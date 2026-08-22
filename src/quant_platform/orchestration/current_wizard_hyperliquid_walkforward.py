"""Purged walk-forward validation for the current Wizard board."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    _exposure_hedge_ratio,
    _load_history,
    _orient_history,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward import (
    FALSE_DISCOVERY_RATE,
    FOLD_COUNT,
    _add_statistical_selection_controls,
    _aggregate_candidate,
    _attach_causal_entry_features,
    _build_folds,
    _fit_training_parameters,
    _flat_fold_signal,
    _rank_candidates,
    _walkforward_gate_blockers,
)
from quant_platform.wizard_mode_replay import build_local_mode_signal

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_walkforward.v1"
RESEARCH_ONLY_REASON = (
    "walk_forward_is_research_only;strict_l2_calibration_required;"
    "regime_robustness_not_run;parameter_sensitivity_not_run;"
    "local_formula_approximation;mode_fidelity_parity_not_proven"
)
IDENTITY_COLUMNS = (
    "experiment_id",
    "pair_group_key",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "exact_mode",
    "orientation",
    "asset_x",
    "asset_y",
)


def run_current_wizard_hyperliquid_walkforward(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Run every completed observed-cost cell through purged expanding folds."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    observed_manifest_path = (
        active / "current_wizard_hyperliquid_observed_cost_replay_manifest.json"
    )
    cost_manifest_path = active / "current_wizard_hyperliquid_cost_manifest.json"
    if not observed_manifest_path.exists() or not cost_manifest_path.exists():
        raise FileNotFoundError("Current observed replay and cost manifests are required")
    observed_manifest = _read_json(observed_manifest_path)
    cost_manifest = _read_json(cost_manifest_path)
    observed_artifacts = observed_manifest.get("artifacts", {})
    cost_artifacts = cost_manifest.get("artifacts", {})
    input_paths = {
        "observed_results": root / _text(observed_artifacts.get("snapshot_results")),
        "pair_costs": root / _text(cost_artifacts.get("snapshot_pairs")),
        "observed_manifest": root / _text(observed_artifacts.get("snapshot_manifest")),
        "cost_manifest": root / _text(cost_artifacts.get("snapshot_manifest")),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current walk-forward inputs missing: {missing}")
    observed = pd.read_csv(input_paths["observed_results"])
    pair_costs = pd.read_csv(input_paths["pair_costs"])
    if observed["experiment_id"].duplicated().any():
        raise ValueError("Current observed replay contains duplicate experiment ids")
    if pair_costs["pair_group_key"].duplicated().any():
        raise ValueError("Current pair costs contain duplicate pair keys")
    if _text(observed_manifest.get("cost_evidence_id")) != _text(
        cost_manifest.get("cost_evidence_id")
    ):
        raise ValueError("Current observed replay and cost identities do not match")

    policy = {
        "fold_count": FOLD_COUNT,
        "initial_train_fraction": 0.50,
        "embargo_bars": 20,
        "minimum_train_rows": 180,
        "minimum_test_rows": 30,
        "minimum_aggregate_trades": 10,
        "minimum_profit_factor": 1.10,
        "maximum_drawdown": 0.50,
        "minimum_positive_folds": 3,
        "maximum_positive_fold_concentration": 0.80,
        "false_discovery_rate": FALSE_DISCOVERY_RATE,
        "candidate_policy": "all_completed_observed_cost_replays_no_performance_prefilter",
        "fit_policy": "training_only_parameters_with_20_bar_embargo",
        "fold_position_policy": "start_flat_wait_for_source_flat_force_flat_at_end",
        "dynamic_exposure_policy": "same_causal_rolling_hedge_ratio_as_signal_metric",
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "observed_cost_replay_id": _text(observed_manifest.get("observed_cost_replay_id")),
        "cost_evidence_id": _text(cost_manifest.get("cost_evidence_id")),
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    walkforward_id = "cwwalk_" + sha256(_canonical_json(material).encode()).hexdigest()[:20]
    observed_snapshot_manifest = input_paths["observed_manifest"]
    snapshot_dir = observed_snapshot_manifest.parent / "walkforwards" / walkforward_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = immutable_snapshot_copy(source, input_dir, artifact_name=name)
        snapshot_inputs[name] = target

    pair_lookup = {_text(row.pair_group_key): row for row in pair_costs.itertuples()}
    history_cache: dict[str, pd.DataFrame] = {}
    status_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    bar_rows: list[dict[str, object]] = []
    for observed_row in observed.itertuples():
        pair_key = _text(observed_row.pair_group_key)
        pair_cost = pair_lookup.get(pair_key)
        base = _status_base(
            observed_row,
            pair_cost=pair_cost,
            walkforward_id=walkforward_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        if _text(observed_row.replay_status) != "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE":
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "NOT_RUN_PRIOR_REPLAY_BLOCKED",
                    "walkforward_blocker": _text(observed_row.replay_blocker)
                    or _text(observed_row.replay_status),
                }
            )
            continue
        if pair_cost is None or not bool(pair_cost.provisional_cost_research_ready):
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_INPUTS",
                    "walkforward_blocker": "pair_cost_research_evidence_not_ready",
                }
            )
            continue
        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            if str(history_path) not in history_cache:
                history_cache[str(history_path)] = _load_history(history_path)
            history = _orient_history(
                history_cache[str(history_path)],
                orientation=_text(observed_row.orientation),
            )
            folds = _build_folds(len(history))
            captured_settings = json.loads(_text(observed_row.settings_json))
            costs = _cost_model(pair_cost, timeframe=_text(observed_row.timeframe))
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_INPUTS",
                    "walkforward_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue

        candidate_fold_rows: list[dict[str, object]] = []
        candidate_trades: list[pd.DataFrame] = []
        candidate_bars: list[pd.DataFrame] = []
        candidate_blocker = ""
        exact_mode = _text(observed_row.exact_mode)
        for fold in folds:
            try:
                train = history.iloc[: fold["train_end"]].copy()
                settings, fit_diagnostics = _fit_training_parameters(
                    train,
                    captured_settings,
                    exact_mode=exact_mode,
                )
                signal_history = history.iloc[: fold["test_end"]].copy()
                signal_history["hedge_ratio"] = _exposure_hedge_ratio(
                    signal_history,
                    exact_mode=exact_mode,
                    settings=settings,
                )
                mode_result = build_local_mode_signal(
                    signal_history,
                    settings,
                    exact_mode=exact_mode,
                )
                if mode_result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                    raise ValueError(
                        "mode_signal_blocked:" + ";".join(mode_result.missing_inputs)
                    )
                eval_start = fold["test_start"] - 1
                evaluation = signal_history.iloc[eval_start : fold["test_end"]].copy()
                source_signal = mode_result.signal.reindex(signal_history.index)
                eval_signal = _flat_fold_signal(
                    source_signal,
                    evaluation.index,
                    prior_position=source_signal.iloc[eval_start],
                )
                result, ledger = backtest_two_leg_spread_with_ledger(
                    evaluation,
                    eval_signal,
                    costs,
                    interval=_interval(_text(observed_row.timeframe)),
                )
                closed, bars = _attach_causal_entry_features(
                    closed=ledger.closed_trades,
                    bars=ledger.bar_ledger,
                    signal_history=signal_history,
                    mode_metric=mode_result.metric,
                    metric_name=mode_result.metric_name,
                    settings=settings,
                    exact_mode=exact_mode,
                )
            except Exception as exc:
                candidate_blocker = (
                    f"fold_{fold['fold_number']}:{safe_exception_code(exc)}"
                )
                break
            fold_row = {
                **_identity(observed_row, pair_cost),
                "schema_version": SCHEMA_VERSION,
                "walkforward_id": walkforward_id,
                "observed_cost_replay_id": material["observed_cost_replay_id"],
                "fold_number": fold["fold_number"],
                "fold_status": "COMPLETE",
                "fold_blocker": "",
                "train_start_at": _timestamp(history.iloc[0]["timestamp"]),
                "train_end_at": _timestamp(history.iloc[fold["train_end"] - 1]["timestamp"]),
                "embargo_start_at": _timestamp(history.iloc[fold["train_end"]]["timestamp"]),
                "test_start_at": _timestamp(history.iloc[fold["test_start"]]["timestamp"]),
                "test_end_at": _timestamp(history.iloc[fold["test_end"] - 1]["timestamp"]),
                "train_rows": fold["train_end"],
                "embargo_rows": fold["test_start"] - fold["train_end"],
                "test_rows": fold["test_end"] - fold["test_start"],
                "fitted_hedge_ratio": fit_diagnostics.get("fitted_hedge_ratio"),
                "fitted_ou_mu": fit_diagnostics.get("fitted_ou_mu"),
                "fitted_ou_sigma": fit_diagnostics.get("fitted_ou_sigma"),
                "fit_uses_test_data": False,
                **result.__dict__,
                "fold_positive_after_cost": bool(result.total_return > 0.0),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
            candidate_fold_rows.append(fold_row)
            if not closed.empty:
                closed["fold_number"] = fold["fold_number"]
                candidate_trades.append(closed)
            bars["fold_number"] = fold["fold_number"]
            candidate_bars.append(bars)

        if candidate_blocker or len(candidate_fold_rows) != FOLD_COUNT:
            status_rows.append(
                {
                    **base,
                    "walkforward_status": "BLOCKED_WALK_FORWARD_REPLAY",
                    "walkforward_blocker": candidate_blocker
                    or "incomplete_fold_accounting",
                }
            )
            continue
        aggregate = _aggregate_candidate(
            candidate_fold_rows,
            candidate_trades,
            candidate_bars,
            interval=_interval(_text(observed_row.timeframe)),
        )
        gate_blockers = _walkforward_gate_blockers(aggregate)
        candidate = {
            **base,
            **aggregate,
            "primary_candidate": bool(observed_row.research_rank_eligible),
            "walkforward_status": (
                "PASS_RESEARCH_WALK_FORWARD"
                if not gate_blockers
                else "FAIL_RESEARCH_WALK_FORWARD"
            ),
            "walkforward_blocker": ";".join(gate_blockers),
            "acceptance_status": "BLOCKED",
            "acceptance_reason": (
                RESEARCH_ONLY_REASON + ";" + _text(pair_cost.cost_blocker)
            ).strip(";"),
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        candidate_rows.append(candidate)
        status_rows.append(candidate.copy())
        fold_rows.extend(candidate_fold_rows)
        for ledger_type, frames in (("closed", candidate_trades), ("bar", candidate_bars)):
            for frame in frames:
                for record in frame.reset_index(drop=True).to_dict("records"):
                    payload = {
                        **_identity(observed_row, pair_cost),
                        "schema_version": SCHEMA_VERSION,
                        "walkforward_id": walkforward_id,
                        "observed_cost_replay_id": material["observed_cost_replay_id"],
                        "ledger_type": ledger_type,
                        **record,
                        "backtest_label": True,
                        "paper_label": False,
                        "live_label": False,
                        "live_trading_authorized": False,
                    }
                    (trade_rows if ledger_type == "closed" else bar_rows).append(payload)

    status = pd.DataFrame(status_rows)
    if len(status) != len(observed) or status["experiment_id"].nunique() != len(observed):
        raise ValueError("Current walk-forward failed complete experiment accounting")
    statistical_columns = [
        "family_tests",
        "fold_return_raw_pvalue",
        "fold_return_normal_proxy_status",
        "block_bootstrap_return_pvalue",
        "block_bootstrap_lower_95",
        "block_bootstrap_upper_95",
        "block_bootstrap_status",
        "block_bootstrap_blocker",
        "return_observations",
        "probabilistic_sharpe_probability",
        "probabilistic_sharpe_status",
        "probabilistic_sharpe_blocker",
        "bh_qvalue",
        "false_discovery_rate",
        "hedge_ratio_cv",
        "parameter_stability_status",
        "deflated_sharpe_probability",
        "deflated_sharpe_benchmark",
        "deflated_sharpe_status",
        "statistical_selection_status",
        "statistical_selection_blocker",
    ]
    raw_candidates = pd.DataFrame(candidate_rows)
    if raw_candidates.empty:
        raw_candidates = status.iloc[0:0].copy()
    candidates = _add_statistical_selection_controls(raw_candidates)
    for column in statistical_columns:
        if column not in candidates.columns:
            candidates[column] = pd.Series(dtype=object)
    if not candidates.empty:
        statistics = candidates.set_index("experiment_id")[statistical_columns]
        mask = status["experiment_id"].isin(statistics.index)
        for column in statistical_columns:
            status.loc[mask, column] = status.loc[mask, "experiment_id"].map(
                statistics[column]
            )
    status["statistical_selection_status"] = status.get(
        "statistical_selection_status", pd.Series(index=status.index, dtype=object)
    ).fillna("NOT_EVALUATED")
    status["statistical_selection_blocker"] = status.get(
        "statistical_selection_blocker", pd.Series(index=status.index, dtype=object)
    ).fillna("walk_forward_not_completed")
    folds_frame = _with_empty_schema(
        pd.DataFrame(fold_rows),
        (*IDENTITY_COLUMNS, "schema_version", "walkforward_id", "fold_number"),
    )
    trades = _with_empty_schema(
        pd.DataFrame(trade_rows),
        (*IDENTITY_COLUMNS, "schema_version", "walkforward_id", "ledger_type", "entry_timestamp"),
    )
    bars = _with_empty_schema(
        pd.DataFrame(bar_rows),
        (*IDENTITY_COLUMNS, "schema_version", "walkforward_id", "ledger_type", "timestamp"),
    )
    ranked = _rank_candidates(candidates)
    validation = _validation(observed, status, folds_frame, candidates)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current walk-forward validation failed: " + ",".join(failed))
    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (status, "status", "snapshot_status"),
        (candidates, "candidates", "snapshot_candidates"),
        (ranked, "ranked", "snapshot_ranked"),
        (folds_frame, "folds", "snapshot_folds"),
        (trades, "trades", "snapshot_trades"),
        (bars, "bars", "snapshot_bars"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)
    status_counts = status["walkforward_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "walkforward_id": walkforward_id,
        "experiments_accounted": int(len(status)),
        "experiment_status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status)),
        "walkforward_candidates_completed": int(len(candidates)),
        "primary_candidates_completed": int(candidates.get("primary_candidate", pd.Series(dtype=bool)).astype(bool).sum()),
        "walkforward_passes": int(candidates.get("walkforward_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_WALK_FORWARD").sum()),
        "statistical_selection_passes": int(candidates.get("statistical_selection_status", pd.Series(dtype=str)).eq("PASS").sum()),
        "folds_expected": int(len(candidates) * FOLD_COUNT),
        "folds_complete": int(len(folds_frame)),
        "closed_trade_rows": int(len(trades)),
        "bar_ledger_rows": int(len(bars)),
        "selection_hindsight_used_for_fold_execution": False,
        "canonical_replay_leverage": 1.0,
        "acceptance_eligible_replays": 0,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {name: _relative(path, root) for name, path in snapshot_inputs.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _cost_model(pair_cost: object, *, timeframe: str) -> CostModel:
    return CostModel(
        taker_fee_bps=float(pair_cost.taker_fee_bps),
        slippage_bps=float(pair_cost.pair_one_way_slippage_bps),
        execution_risk_bps=float(pair_cost.execution_risk_bps),
        funding_bps_per_day=0.0,
        bars_per_day=24 if _interval(timeframe) == "1h" else 1,
        funding_policy=FundingPolicy.SIGNED_REALIZED.value,
    )


def _with_empty_schema(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    if frame.empty and len(frame.columns) == 0:
        return pd.DataFrame(columns=list(columns))
    return frame


def _status_base(
    row: object,
    *,
    pair_cost: object | None,
    walkforward_id: str,
    evidence_paths: Any,
    root: Path,
) -> dict[str, object]:
    return {
        **_identity(row, pair_cost),
        "schema_version": SCHEMA_VERSION,
        "walkforward_id": walkforward_id,
        "observed_cost_replay_id": _text(row.observed_cost_replay_id),
        "prior_replay_status": _text(row.replay_status),
        "prior_replay_blocker": _text(row.replay_blocker),
        "primary_candidate": bool(getattr(row, "research_rank_eligible", False)),
        "canonical_replay_leverage": 1.0,
        "walkforward_status": "",
        "walkforward_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "walk_forward_not_complete",
        "acceptance_eligible": False,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _identity(row: object, pair_cost: object | None) -> dict[str, object]:
    return {
        "experiment_id": _text(row.experiment_id),
        "pair_group_key": _text(row.pair_group_key),
        "pair": _text(row.pair),
        "wizard_exchange": _text(row.wizard_exchange),
        "wizard_timeframe": _text(row.timeframe),
        "hyperliquid_interval": _text(getattr(pair_cost, "hyperliquid_interval", "")),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "asset_x": _text(getattr(pair_cost, "asset_x", "")),
        "asset_y": _text(getattr(pair_cost, "asset_y", "")),
    }


def _validation(
    observed: pd.DataFrame,
    status: pd.DataFrame,
    folds: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    fold_counts = folds.groupby("experiment_id").size() if not folds.empty else pd.Series(dtype=int)
    checks = {
        "experiment_count_preserved": len(observed) == len(status),
        "experiment_ids_unique": status["experiment_id"].nunique() == len(status),
        "statuses_accounted": status["walkforward_status"].astype(str).ne("").all(),
        "five_folds_per_candidate": candidates.empty or fold_counts.eq(FOLD_COUNT).all(),
        "fit_uses_no_test_data": folds.empty or not folds["fit_uses_test_data"].astype(bool).any(),
        "embargo_positive": folds.empty or pd.to_numeric(folds["embargo_rows"], errors="coerce").gt(0).all(),
        "folds_end_flat": folds.empty or pd.to_numeric(folds["open_trades"], errors="coerce").fillna(1).eq(0).all(),
        "one_x_only": status["canonical_replay_leverage"].eq(1.0).all(),
        "acceptance_disabled": status["acceptance_status"].eq("BLOCKED").all(),
        "live_trading_disabled": not status["live_trading_authorized"].astype(bool).any(),
    }
    return pd.DataFrame(
        [{"check": check, "status": "PASS" if passed else "FAIL"} for check, passed in checks.items()]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    return {
        "status": active / "current_wizard_hyperliquid_walkforward_status.csv",
        "candidates": active / "current_wizard_hyperliquid_walkforward_candidates.csv",
        "ranked": active / "current_wizard_hyperliquid_walkforward_ranked.csv",
        "folds": active / "current_wizard_hyperliquid_walkforward_folds.csv",
        "trades": active / "current_wizard_hyperliquid_walkforward_trades.csv",
        "bars": active / "current_wizard_hyperliquid_walkforward_bars.csv.gz",
        "validation": active / "current_wizard_hyperliquid_walkforward_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_walkforward_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_walkforward_summary.md",
        "snapshot_status": snapshot / "walkforward_status.csv",
        "snapshot_candidates": snapshot / "walkforward_candidates.csv",
        "snapshot_ranked": snapshot / "walkforward_ranked.csv",
        "snapshot_folds": snapshot / "walkforward_folds.csv",
        "snapshot_trades": snapshot / "walkforward_trades.csv",
        "snapshot_bars": snapshot / "walkforward_bars.csv.gz",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Walk-Forward",
            "",
            f"- Walk-forward: `{summary['walkforward_id']}`",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Candidates completed: {summary['walkforward_candidates_completed']}",
            f"- Practical passes: {summary['walkforward_passes']}",
            f"- Statistical-selection passes: {summary['statistical_selection_passes']}",
            f"- Folds complete: {summary['folds_complete']} / {summary['folds_expected']}",
            "- Selection hindsight used for fold execution: no",
            "- Promotion authority: no",
            "- Live trading authorized: no",
            "",
        ]
    )


def _timestamp(value: object) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.isoformat() if pd.notna(parsed) else ""


def _interval(timeframe: str) -> str:
    return timeframe.lower().replace("daily", "1d").replace("hourly", "1h")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
