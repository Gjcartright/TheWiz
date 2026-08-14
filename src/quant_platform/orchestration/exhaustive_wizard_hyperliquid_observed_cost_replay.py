"""Research-only exhaustive replay with observed Hyperliquid cost evidence."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    MIN_RESEARCH_RANK_TRADES,
    _load_history,
    _mode_settings,
    _orient_history,
    _research_rank_blocker,
    _unique_mode_rows,
)
from quant_platform.wizard_mode_replay import build_local_mode_signal


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_observed_cost_replay.v1"
MIN_OBSERVED_FUNDING_ROWS = 250
MIN_RESEARCH_RANK_BARS = {"1d": 365, "1h": 720}
RESEARCH_ONLY_REASON = (
    "observed_cost_replay_is_not_acceptance_authority;"
    "l2_slippage_is_current_calibration_not_historical_depth;"
    "walk_forward_not_run"
)


def run_exhaustive_wizard_hyperliquid_observed_cost_replay(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Replay every eligible cell on observed funding and pair-specific L2 costs."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    preflight_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_replay_preflight_manifest.json"
    )
    cost_manifest_path = active / "exhaustive_wizard_hyperliquid_cost_evidence_manifest.json"
    canonical_path = active / "exhaustive_wizard_hyperliquid_canonical_replay.csv"
    required = [preflight_manifest_path, cost_manifest_path, canonical_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Observed-cost replay inputs missing: {missing}")

    preflight_manifest = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    cost_manifest = json.loads(cost_manifest_path.read_text(encoding="utf-8"))
    run_id = _text(cost_manifest.get("run_id"))
    preflight_id = _text(cost_manifest.get("replay_preflight_id"))
    history_run_id = _text(cost_manifest.get("history_run_id"))
    funding_evidence_id = _text(cost_manifest.get("funding_evidence_id"))
    cost_evidence_id = _text(cost_manifest.get("cost_evidence_id"))
    if run_id != _text(preflight_manifest.get("run_id")) or preflight_id != _text(
        preflight_manifest.get("replay_preflight_id")
    ):
        raise ValueError("Preflight and cost-evidence identities do not match")

    experiment_path = root / _text(
        preflight_manifest.get("artifacts", {}).get("snapshot_experiment_preflight")
    )
    mode_path = root / _text(preflight_manifest.get("input_snapshots", {}).get("mode_ledger"))
    pair_cost_path = root / _text(
        cost_manifest.get("artifacts", {}).get("snapshot_pair_cost_evidence")
    )
    experiment_cost_path = root / _text(
        cost_manifest.get("artifacts", {}).get("snapshot_experiment_cost_readiness")
    )
    required.extend([experiment_path, mode_path, pair_cost_path, experiment_cost_path])
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Observed-cost replay inputs missing: {missing}")

    experiments = _read_csv(experiment_path)
    modes = _read_csv(mode_path)
    pair_costs = _read_csv(pair_cost_path)
    experiment_costs = _read_csv(experiment_cost_path)
    canonical = _read_csv(canonical_path)
    _require_unique(experiments, "experiment_id")
    _require_unique(pair_costs, "pair_group_id")
    _require_unique(experiment_costs, "experiment_id")
    _require_unique(canonical, "experiment_id")
    if len(experiments) != len(experiment_costs):
        raise ValueError("Cost readiness does not account for every experiment")

    input_paths = {
        "experiment_preflight": experiment_path,
        "mode_ledger": mode_path,
        "pair_cost_evidence": pair_cost_path,
        "experiment_cost_readiness": experiment_cost_path,
        "canonical_replay": canonical_path,
        "cost_manifest": cost_manifest_path,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "as_of": as_of.isoformat(),
        "minimum_observed_funding_rows": MIN_OBSERVED_FUNDING_ROWS,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    replay_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    replay_id = f"hlobserved_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{replay_hash[:8]}"
    cost_snapshot_manifest = root / _text(
        cost_manifest.get("artifacts", {}).get("snapshot_manifest")
    )
    snapshot_dir = cost_snapshot_manifest.parent / "observed_cost_replays" / replay_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = input_dir / source.name
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    paths = {
        "results": active / "exhaustive_wizard_hyperliquid_observed_cost_replay.csv",
        "ranked": active / "exhaustive_wizard_hyperliquid_observed_cost_replay_ranked.csv",
        "trades": active / "exhaustive_wizard_hyperliquid_observed_cost_replay_trades.csv",
        "comparison": active / "exhaustive_wizard_hyperliquid_observed_cost_comparison.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_observed_cost_replay_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_observed_cost_replay_summary.md",
        "snapshot_results": snapshot_dir / "observed_cost_replay.csv",
        "snapshot_ranked": snapshot_dir / "observed_cost_replay_ranked.csv",
        "snapshot_trades": snapshot_dir / "observed_cost_replay_trades.csv",
        "snapshot_comparison": snapshot_dir / "observed_cost_comparison.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    active_pair_group_ids = {_text(value) for value in experiments["pair_group_id"]}
    if "" in active_pair_group_ids:
        raise ValueError("Observed-cost experiment identity missing pair_group_id")
    mode_lookup = _unique_mode_rows(
        modes,
        allowed_pair_group_ids=active_pair_group_ids,
    )
    pair_lookup = _row_lookup(pair_costs, "pair_group_id")
    readiness_lookup = _row_lookup(experiment_costs, "experiment_id")
    canonical_lookup = _row_lookup(canonical, "experiment_id")
    history_cache: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for experiment in experiments.itertuples():
        experiment_id = _text(experiment.experiment_id)
        pair_group_id = _text(experiment.pair_group_id)
        exact_mode = _text(experiment.exact_mode)
        orientation = _text(experiment.orientation)
        pair_cost = pair_lookup.get(pair_group_id)
        readiness = readiness_lookup.get(experiment_id)
        canonical_row = canonical_lookup.get(experiment_id)
        base = _base_row(
            experiment,
            pair_cost=pair_cost,
            canonical_row=canonical_row,
            replay_id=replay_id,
            history_run_id=history_run_id,
            funding_evidence_id=funding_evidence_id,
            cost_evidence_id=cost_evidence_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        if readiness is None:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_COST_EVIDENCE",
                    "replay_blocker": "experiment_cost_readiness_missing",
                }
            )
            continue
        readiness_status = _text(readiness.cost_replay_status)
        if readiness_status not in {
            "READY_FOR_PROVISIONAL_COST_RESEARCH",
            "READY_FOR_COST_CALIBRATED_REPLAY",
        }:
            result_rows.append(
                {
                    **base,
                    "replay_status": readiness_status,
                    "replay_blocker": _text(readiness.cost_replay_blocker),
                }
            )
            continue
        if pair_cost is None:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_COST_EVIDENCE",
                    "replay_blocker": "pair_cost_evidence_missing",
                }
            )
            continue
        mode_row = mode_lookup.get((pair_group_id, exact_mode, orientation))
        if mode_row is None or _text(mode_row.capture_status) != "CAPTURED":
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_MODE_INPUTS",
                    "replay_blocker": "captured_mode_row_missing",
                }
            )
            continue
        if exact_mode.startswith("Dyn"):
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_DYNAMIC_EXPOSURE_RULE",
                    "replay_blocker": (
                        "wizard_dynamic_hedge_exposure_rule_not_observed;"
                        "do_not_invent_dynamic_hedge_window"
                    ),
                }
            )
            continue

        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            cache_key = str(history_path)
            if cache_key not in history_cache:
                raw = _load_history(history_path)
                history_cache[cache_key] = _longest_observed_funding_segment(
                    raw,
                    interval=_text(experiment.hyperliquid_interval),
                )
            raw_history = history_cache[cache_key]
            if len(raw_history) < MIN_OBSERVED_FUNDING_ROWS:
                raise ValueError(
                    f"observed_funding_segment_rows_{len(raw_history)}_below_"
                    f"{MIN_OBSERVED_FUNDING_ROWS}"
                )
            raw_history = raw_history.copy()
            raw_history["slippage_x_model_bps"] = _required_number(
                pair_cost.slippage_x_p95_bps,
                "slippage_x_p95_bps",
            )
            raw_history["slippage_y_model_bps"] = _required_number(
                pair_cost.slippage_y_p95_bps,
                "slippage_y_p95_bps",
            )
            history = _orient_history(raw_history, orientation=orientation)
            paired_ou = mode_lookup.get((pair_group_id, "OU (Spread)", orientation))
            settings, setting_source = _mode_settings(mode_row, paired_ou=paired_ou)
            captured_hedge_ratio = _number(settings.get("hedge_ratio"))
            if captured_hedge_ratio is not None:
                history["hedge_ratio"] = captured_hedge_ratio
            mode_result = build_local_mode_signal(history, settings, exact_mode=exact_mode)
            if mode_result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                result_rows.append(
                    {
                        **base,
                        "replay_status": "BLOCKED_MODE_INPUTS",
                        "replay_blocker": ";".join(mode_result.missing_inputs),
                        "mode_fidelity_status": mode_result.mode_fidelity_status,
                        "mode_fidelity_reason": mode_result.mode_fidelity_reason,
                        "mode_setting_source": setting_source,
                    }
                )
                continue
            costs = CostModel(
                taker_fee_bps=_required_number(pair_cost.taker_fee_bps, "taker_fee_bps"),
                slippage_bps=_required_number(
                    pair_cost.pair_one_way_slippage_bps,
                    "pair_one_way_slippage_bps",
                ),
                execution_risk_bps=_required_number(
                    pair_cost.execution_risk_bps,
                    "execution_risk_bps",
                ),
                funding_bps_per_day=0.0,
                funding_policy=FundingPolicy.SIGNED_REALIZED.value,
            )
            result, ledger = backtest_two_leg_spread_with_ledger(
                history,
                mode_result.signal,
                costs,
                interval=_text(experiment.hyperliquid_interval),
            )
        except Exception as exc:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_REPLAY_ERROR",
                    "replay_blocker": f"{type(exc).__name__}:{exc}",
                }
            )
            continue

        metrics = asdict(result)
        completed = {
            **base,
            **metrics,
            "replay_status": "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE",
            "replay_blocker": "",
            "metric_name": mode_result.metric_name,
            "mode_fidelity_status": mode_result.mode_fidelity_status,
            "mode_fidelity_reason": mode_result.mode_fidelity_reason,
            "mode_setting_source": setting_source,
            "computation_notes": ";".join(mode_result.computation_notes),
            "observed_funding_rows": int(len(history)),
            "acceptance_status": "BLOCKED",
            "acceptance_reason": (
                f"{RESEARCH_ONLY_REASON};{_text(pair_cost.cost_blocker)};"
                f"{mode_result.mode_fidelity_status}"
            ),
            "local_replay_completed": True,
        }
        completed.update(_comparison_metrics(completed, canonical_row))
        result_rows.append(completed)
        for lifecycle, frame in (("closed", ledger.closed_trades), ("open", ledger.open_trades)):
            for trade in frame.to_dict("records"):
                trade_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "exhaustive_run_id": run_id,
                        "observed_cost_replay_id": replay_id,
                        "cost_evidence_id": cost_evidence_id,
                        "experiment_id": experiment_id,
                        "pair_group_id": pair_group_id,
                        "pair": _text(experiment.pair),
                        "exact_mode": exact_mode,
                        "orientation": orientation,
                        "trade_lifecycle": lifecycle,
                        **trade,
                        "backtest_label": True,
                        "paper_label": False,
                        "live_label": False,
                        "live_trading_authorized": False,
                    }
                )

    results = pd.DataFrame(result_rows)
    if len(results) != len(experiments) or results["experiment_id"].nunique() != len(experiments):
        raise ValueError("Observed-cost replay failed complete experiment accounting")
    results["math_version"] = results.get(
        "math_version",
        pd.Series(index=results.index, dtype=object),
    ).fillna("")
    results["research_rank_eligible"] = False
    results["research_rank_blocker"] = "replay_not_complete"
    completed_mask = results["replay_status"].eq("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE")
    rank_blockers = results.loc[completed_mask].apply(_observed_rank_blocker, axis=1)
    results.loc[completed_mask, "research_rank_blocker"] = rank_blockers
    results.loc[completed_mask, "research_rank_eligible"] = rank_blockers.eq("")
    completed_frame = results.loc[completed_mask].copy()
    ranked = _rank_completed(completed_frame)
    comparison_columns = [
        "experiment_id",
        "pair_group_id",
        "pair",
        "wizard_exchange",
        "wizard_timeframe",
        "exact_mode",
        "orientation",
        "trades",
        "profit_factor",
        "sharpe",
        "max_drawdown",
        "total_return",
        "total_funding",
        "canonical_trades",
        "canonical_profit_factor",
        "canonical_sharpe",
        "canonical_max_drawdown",
        "canonical_total_return",
        "canonical_total_funding",
        "trades_delta_vs_canonical",
        "profit_factor_delta_vs_canonical",
        "sharpe_delta_vs_canonical",
        "max_drawdown_delta_vs_canonical",
        "total_return_delta_vs_canonical",
        "total_funding_delta_vs_canonical",
        "research_rank_eligible",
        "research_rank_blocker",
        "acceptance_status",
        "acceptance_reason",
    ]
    comparison = completed_frame.reindex(columns=comparison_columns)
    trades = pd.DataFrame(trade_rows)
    for frame, active_path, snapshot_path in (
        (results, paths["results"], paths["snapshot_results"]),
        (ranked, paths["ranked"], paths["snapshot_ranked"]),
        (trades, paths["trades"], paths["snapshot_trades"]),
        (comparison, paths["comparison"], paths["snapshot_comparison"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    status_counts = _status_counts(results, "replay_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "observed_cost_replay_id": replay_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(results)),
        "unique_experiment_ids": int(results["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(results)),
        "observed_cost_replays_complete": int(
            status_counts.get("OBSERVED_COST_RESEARCH_REPLAY_COMPLETE", 0)
        ),
        "research_rank_eligible_replays": int(results["research_rank_eligible"].map(_truthy).sum()),
        "minimum_research_rank_trades": MIN_RESEARCH_RANK_TRADES,
        "minimum_research_rank_bars": MIN_RESEARCH_RANK_BARS,
        "trade_ledger_rows": int(len(trades)),
        "canonical_replay_leverage": 1.0,
        "funding_policy": FundingPolicy.SIGNED_REALIZED.value,
        "observed_funding_policy": (
            "longest_contiguous_two_leg_observed_segment_selected_by_coverage_only"
        ),
        "slippage_policy": "pair_leg_p95_from_current_public_l2_research_only",
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {name: _relative(path, root) for name, path in snapshot_inputs.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _base_row(
    experiment: object,
    *,
    pair_cost: object | None,
    canonical_row: object | None,
    replay_id: str,
    history_run_id: str,
    funding_evidence_id: str,
    cost_evidence_id: str,
    evidence_paths: object,
    root: Path,
) -> dict[str, object]:
    pair_cost = pair_cost or {}
    orientation = _text(experiment.orientation)
    asset_x = _text(experiment.asset_x)
    asset_y = _text(experiment.asset_y)
    slippage_x = _number(_get(pair_cost, "slippage_x_p95_bps"))
    slippage_y = _number(_get(pair_cost, "slippage_y_p95_bps"))
    replay_slippage_x = slippage_y if orientation == "reverse" else slippage_x
    replay_slippage_y = slippage_x if orientation == "reverse" else slippage_y
    canonical_values = _canonical_metrics(canonical_row)
    return {
        "schema_version": SCHEMA_VERSION,
        "exhaustive_run_id": _text(experiment.exhaustive_run_id),
        "replay_preflight_id": _text(experiment.replay_preflight_id),
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "cost_evidence_id": cost_evidence_id,
        "observed_cost_replay_id": replay_id,
        "experiment_id": _text(experiment.experiment_id),
        "pair_group_id": _text(experiment.pair_group_id),
        "pair": _text(experiment.pair),
        "wizard_exchange": _text(experiment.wizard_exchange),
        "wizard_timeframe": _text(experiment.wizard_timeframe),
        "hyperliquid_interval": _text(experiment.hyperliquid_interval),
        "exact_mode": _text(experiment.exact_mode),
        "orientation": orientation,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "replay_asset_x": asset_y if orientation == "reverse" else asset_x,
        "replay_asset_y": asset_x if orientation == "reverse" else asset_y,
        "scanner_cutoff_at": _text(experiment.scanner_cutoff_at),
        "preflight_status": _text(experiment.preflight_status),
        "preflight_blocker": _text(experiment.preflight_blocker),
        "canonical_replay_leverage": 1.0,
        "observed_funding_rows": 0,
        "funding_both_coverage": _number(_get(pair_cost, "funding_both_coverage")),
        "taker_fee_bps": _number(_get(pair_cost, "taker_fee_bps")),
        "slippage_x_p95_bps": slippage_x,
        "slippage_y_p95_bps": slippage_y,
        "replay_slippage_x_p95_bps": replay_slippage_x,
        "replay_slippage_y_p95_bps": replay_slippage_y,
        "pair_one_way_slippage_bps": _number(_get(pair_cost, "pair_one_way_slippage_bps")),
        "execution_risk_bps": _number(_get(pair_cost, "execution_risk_bps")),
        "funding_policy": FundingPolicy.SIGNED_REALIZED.value,
        "partial_fill_probability": CostModel().partial_fill_probability,
        "cost_evidence_status": _text(_get(pair_cost, "cost_evidence_status")),
        "cost_evidence_blocker": _text(_get(pair_cost, "cost_blocker")),
        "replay_status": "",
        "replay_blocker": "",
        "metric_name": "",
        "mode_fidelity_status": "",
        "mode_fidelity_reason": "",
        "mode_setting_source": "",
        "computation_notes": "",
        "trades": 0,
        "open_trades": 0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "gross_return": 0.0,
        "total_fees": 0.0,
        "total_slippage": 0.0,
        "total_funding": 0.0,
        "total_execution_risk": 0.0,
        "total_partial_fill_cost": 0.0,
        "avg_gross_exposure": 0.0,
        "sharpe_status": "blocked:not_replayed",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "replay_not_complete",
        "local_replay_completed": False,
        **canonical_values,
        "trades_delta_vs_canonical": None,
        "profit_factor_delta_vs_canonical": None,
        "sharpe_delta_vs_canonical": None,
        "max_drawdown_delta_vs_canonical": None,
        "total_return_delta_vs_canonical": None,
        "total_funding_delta_vs_canonical": None,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _longest_observed_funding_segment(
    history: pd.DataFrame,
    *,
    interval: str,
) -> pd.DataFrame:
    required = {"funding_x_realized_bps", "funding_y_realized_bps"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"observed_funding_columns_missing:{';'.join(sorted(missing))}")
    working = history.sort_index().copy()
    valid = (
        working[list(sorted(required))].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    )
    timestamps = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
    expected = pd.Timedelta(hours=1) if interval == "1h" else pd.Timedelta(days=1)
    best_start = 0
    best_end = 0
    current_start = 0
    previous: pd.Timestamp | None = None
    for position, (is_valid, timestamp) in enumerate(
        zip(valid.tolist(), timestamps.tolist(), strict=True)
    ):
        parsed = pd.Timestamp(timestamp) if pd.notna(timestamp) else None
        contiguous = bool(
            parsed is not None
            and previous is not None
            and pd.Timedelta(0) < parsed - previous <= expected * 1.5
        )
        if not is_valid or parsed is None:
            current_start = position + 1
            previous = None
            continue
        if not contiguous:
            current_start = position
        if position + 1 - current_start > best_end - best_start:
            best_start = current_start
            best_end = position + 1
        previous = parsed
    return working.iloc[best_start:best_end].copy()


def _comparison_metrics(
    observed: dict[str, object], canonical_row: object | None
) -> dict[str, object]:
    canonical = _canonical_metrics(canonical_row)
    pairs = {
        "trades": "canonical_trades",
        "profit_factor": "canonical_profit_factor",
        "sharpe": "canonical_sharpe",
        "max_drawdown": "canonical_max_drawdown",
        "total_return": "canonical_total_return",
        "total_funding": "canonical_total_funding",
    }
    result = dict(canonical)
    for current_name, canonical_name in pairs.items():
        current = _number(observed.get(current_name))
        baseline = _number(canonical.get(canonical_name))
        delta_name = f"{current_name}_delta_vs_canonical"
        if current is None or baseline is None:
            result[delta_name] = None
        elif current_name == "profit_factor" and not (
            math.isfinite(current) and math.isfinite(baseline)
        ):
            result[delta_name] = None
        else:
            result[delta_name] = current - baseline
    return result


def _observed_rank_blocker(row: pd.Series) -> str:
    blockers = [value for value in _research_rank_blocker(row).split(";") if value]
    interval = _text(row.get("hyperliquid_interval"))
    minimum = MIN_RESEARCH_RANK_BARS.get(interval, max(MIN_RESEARCH_RANK_BARS.values()))
    observed_rows = int(_number(row.get("observed_funding_rows")) or 0)
    if observed_rows < minimum:
        blockers.append(f"observed_funding_rows<{minimum}")
    return ";".join(blockers)


def _canonical_metrics(row: object | None) -> dict[str, object]:
    return {
        "canonical_trades": _number(_get(row, "trades")),
        "canonical_profit_factor": _number(_get(row, "profit_factor")),
        "canonical_sharpe": _number(_get(row, "sharpe")),
        "canonical_max_drawdown": _number(_get(row, "max_drawdown")),
        "canonical_total_return": _number(_get(row, "total_return")),
        "canonical_total_funding": _number(_get(row, "total_funding")),
    }


def _rank_completed(completed: pd.DataFrame) -> pd.DataFrame:
    if completed.empty:
        return completed
    eligible = completed.loc[completed["research_rank_eligible"].map(_truthy)].sort_values(
        ["profit_factor", "sharpe", "max_drawdown", "trades"],
        ascending=[False, False, True, False],
        na_position="last",
    )
    eligible.insert(0, "research_rank", range(1, len(eligible) + 1))
    ineligible = completed.loc[~completed["research_rank_eligible"].map(_truthy)].sort_values(
        ["trades", "sharpe", "max_drawdown"],
        ascending=[False, False, True],
        na_position="last",
    )
    ineligible.insert(0, "research_rank", "")
    return pd.concat([eligible, ineligible], ignore_index=True)


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, object]:
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _require_unique(frame: pd.DataFrame, key: str) -> None:
    if frame.empty or key not in frame.columns or frame[key].astype(str).duplicated().any():
        raise ValueError(f"Expected non-empty unique {key} rows")


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(status) or "MISSING_STATUS": int(count)
        for status, count in frame[column].value_counts(dropna=False).items()
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Hyperliquid Observed-Cost 1x Replay",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Observed-cost replay: `{summary['observed_cost_replay_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Research replays complete: {summary['observed_cost_replays_complete']}",
            f"- Research-rank eligible: {summary['research_rank_eligible_replays']}",
            "- Minimum ranked evidence span: "
            f"{summary['minimum_research_rank_bars']['1d']} daily bars / "
            f"{summary['minimum_research_rank_bars']['1h']} hourly bars",
            f"- Trade-ledger rows: {summary['trade_ledger_rows']}",
            f"- Funding policy: `{summary['funding_policy']}`",
            f"- Acceptance-eligible replays: {summary['acceptance_eligible_replays']}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "This is a research-only replay. It uses a continuous suffix with observed two-leg funding, official taker fees, pair-leg current L2 p95 slippage, and the retained execution-risk assumption. Current L2 calibration is not historical depth and does not grant acceptance authority.",
            "",
        ]
    )


def _required_number(value: object, name: str) -> float:
    parsed = _number(value)
    if parsed is None or not math.isfinite(parsed):
        raise ValueError(f"{name}_missing_or_nonfinite")
    return parsed


def _number(value: object) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else None


def _get(row: object | None, key: str) -> object:
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
