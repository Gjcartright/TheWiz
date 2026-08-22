"""Canonical 1x research replays for the exhaustive Wizard run."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import CostModel, backtest_two_leg_spread_with_ledger
from quant_platform.economic_contract import (
    ECONOMIC_CONTRACT_VERSION,
    tail_actions,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import atomic_write_csv, atomic_write_text
from quant_platform.wizard_mode_replay import build_local_mode_signal

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_canonical_replay.v2"
PROVISIONAL_COST_REASON = (
    "provisional_costs;observed_funding_not_attached;observed_slippage_not_attached;"
    "walk_forward_not_run;local_formula_approximation"
)
MIN_RESEARCH_RANK_TRADES = 10


def run_exhaustive_wizard_hyperliquid_canonical_replay(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    cost_model: CostModel | None = None,
) -> CommandResult:
    """Run every eligible exact-mode/orientation cell at canonical 1x exposure."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    costs = cost_model or CostModel()
    active = root / "reports" / "active"
    preflight_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_replay_preflight_manifest.json"
    )
    history_manifest_path = active / "exhaustive_wizard_hyperliquid_history_manifest.json"
    if not preflight_manifest_path.exists() or not history_manifest_path.exists():
        raise FileNotFoundError("Replay preflight and history manifests are required")
    preflight = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    history_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    run_id = _text(preflight.get("run_id"))
    preflight_id = _text(preflight.get("replay_preflight_id"))
    history_run_id = _text(history_manifest.get("history_run_id"))
    if run_id != _text(history_manifest.get("run_id")) or preflight_id != _text(
        history_manifest.get("replay_preflight_id")
    ):
        raise ValueError("History and replay-preflight identities do not match")

    experiment_path = root / _text(
        preflight.get("artifacts", {}).get("snapshot_experiment_preflight")
    )
    mode_path = root / _text(preflight.get("input_snapshots", {}).get("mode_ledger"))
    pair_history_path = root / _text(
        history_manifest.get("artifacts", {}).get("snapshot_pair_results")
    )
    required = [experiment_path, mode_path, pair_history_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Canonical replay inputs missing: {missing}")

    experiments = _read_csv(experiment_path)
    modes = _read_csv(mode_path)
    pair_histories = _read_csv(pair_history_path)
    active_pair_group_ids = {_text(value) for value in experiments["pair_group_id"]}
    if "" in active_pair_group_ids:
        raise ValueError("Canonical replay experiment identity missing pair_group_id")
    mode_lookup = _unique_mode_rows(
        modes,
        allowed_pair_group_ids=active_pair_group_ids,
    )
    pair_lookup = _unique_rows(pair_histories, "pair_group_id")
    cost_payload = asdict(costs)
    material = {
        "schema_version": SCHEMA_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "run_id": run_id,
        "preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "experiment_hash": _file_hash(experiment_path),
        "mode_hash": _file_hash(mode_path),
        "pair_history_hash": _file_hash(pair_history_path),
        "cost_model": cost_payload,
    }
    material_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    replay_id = f"hlcanonical_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{material_hash[:8]}"
    history_snapshot_manifest = root / _text(
        history_manifest.get("artifacts", {}).get("snapshot_manifest")
    )
    snapshot_dir = history_snapshot_manifest.parent / "canonical_replays" / replay_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": active / "exhaustive_wizard_hyperliquid_canonical_replay.csv",
        "ranked": active / "exhaustive_wizard_hyperliquid_canonical_replay_ranked.csv",
        "trades": active / "exhaustive_wizard_hyperliquid_canonical_replay_trades.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_canonical_replay_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_canonical_replay_summary.md",
        "snapshot_results": snapshot_dir / "canonical_replay.csv",
        "snapshot_ranked": snapshot_dir / "canonical_replay_ranked.csv",
        "snapshot_trades": snapshot_dir / "canonical_replay_trades.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }

    history_cache: dict[str, pd.DataFrame] = {}
    result_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for experiment in experiments.itertuples():
        pair_group_id = _text(experiment.pair_group_id)
        exact_mode = _text(experiment.exact_mode)
        orientation = _text(experiment.orientation)
        mode_row = mode_lookup.get((pair_group_id, exact_mode, orientation))
        pair_row = pair_lookup.get(pair_group_id)
        base = _base_result_row(
            experiment,
            replay_id=replay_id,
            history_run_id=history_run_id,
            cost_payload=cost_payload,
            evidence_paths=(experiment_path, mode_path, pair_history_path),
            root=root,
        )
        if _text(experiment.preflight_status) != "READY_FOR_HISTORY":
            result_rows.append(
                {
                    **base,
                    "replay_status": _text(experiment.preflight_status),
                    "replay_blocker": _text(experiment.preflight_blocker),
                }
            )
            continue
        allowed_history_statuses = {
            "READY_FOR_CANONICAL_REPLAY",
            "READY_FOR_SHORT_HISTORY_RESEARCH_REPLAY",
        }
        if pair_row is None or _text(pair_row.history_status) not in allowed_history_statuses:
            blocker = (
                _text(getattr(pair_row, "history_blocker", ""))
                if pair_row is not None
                else "pair_history_result_missing"
            )
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_POINT_IN_TIME_HISTORY",
                    "replay_blocker": blocker or "pair_history_not_ready",
                }
            )
            continue
        history_lane = _text(getattr(pair_row, "history_lane", ""))
        base.update(
            {
                "history_validation_lane": history_lane,
                "history_rows": int(_number(getattr(pair_row, "history_rows", 0)) or 0),
                "history_acceptance_ready": _truthy(
                    getattr(pair_row, "acceptance_history_ready", False)
                ),
                "history_research_ready": _truthy(
                    getattr(pair_row, "research_history_ready", False)
                ),
            }
        )
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

        history_path = root / _text(pair_row.history_path)
        try:
            cache_key = str(history_path)
            if cache_key not in history_cache:
                history_cache[cache_key] = _load_history(history_path)
            raw_history = history_cache[cache_key]
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
                    "replay_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue

        metrics = asdict(result)
        short_history = history_lane == "SHORT_HISTORY_RESEARCH_ONLY"
        result_rows.append(
            {
                **base,
                **metrics,
                "replay_status": (
                    "SHORT_HISTORY_RESEARCH_REPLAY_COMPLETE"
                    if short_history
                    else "RESEARCH_REPLAY_COMPLETE"
                ),
                "replay_blocker": "",
                "metric_name": mode_result.metric_name,
                "mode_fidelity_status": mode_result.mode_fidelity_status,
                "mode_fidelity_reason": mode_result.mode_fidelity_reason,
                "mode_setting_source": setting_source,
                "computation_notes": ";".join(mode_result.computation_notes),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": (
                    f"short_history_research_only;{PROVISIONAL_COST_REASON}"
                    if short_history
                    else PROVISIONAL_COST_REASON
                ),
                "local_replay_completed": True,
            }
        )
        for lifecycle, frame in (
            ("closed", ledger.closed_trades),
            ("open", ledger.open_trades),
        ):
            for trade in frame.to_dict("records"):
                trade_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
                        "exhaustive_run_id": run_id,
                        "canonical_replay_id": replay_id,
                        "experiment_id": _text(experiment.experiment_id),
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
        raise ValueError("Canonical replay failed complete experiment accounting")
    results["research_rank_eligible"] = False
    results["research_rank_blocker"] = "replay_not_complete"
    completed_statuses = {
        "RESEARCH_REPLAY_COMPLETE",
        "SHORT_HISTORY_RESEARCH_REPLAY_COMPLETE",
    }
    completed_mask = results["replay_status"].isin(completed_statuses)
    rank_blockers = results.loc[completed_mask].apply(_research_rank_blocker, axis=1)
    results.loc[completed_mask, "research_rank_blocker"] = rank_blockers
    results.loc[completed_mask, "research_rank_eligible"] = rank_blockers.eq("")
    completed = results.loc[results["replay_status"].isin(completed_statuses)].copy()
    if completed.empty:
        ranked = completed
    else:
        eligible = completed.loc[completed["research_rank_eligible"].astype(bool)].sort_values(
            ["profit_factor", "sharpe", "max_drawdown", "trades"],
            ascending=[False, False, True, False],
            na_position="last",
        )
        eligible.insert(0, "research_rank", range(1, len(eligible) + 1))
        ineligible = completed.loc[~completed["research_rank_eligible"].astype(bool)].sort_values(
            ["trades", "sharpe", "max_drawdown"],
            ascending=[False, False, True],
            na_position="last",
        )
        ineligible.insert(0, "research_rank", "")
        ranked = pd.concat([eligible, ineligible], ignore_index=True)
    trades = pd.DataFrame(trade_rows)
    for frame, active_path, snapshot_path in (
        (results, paths["results"], paths["snapshot_results"]),
        (ranked, paths["ranked"], paths["snapshot_ranked"]),
        (trades, paths["trades"], paths["snapshot_trades"]),
    ):
        atomic_write_csv(frame, active_path, index=False)
        atomic_write_csv(frame, snapshot_path, index=False)

    status_counts = results["replay_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "canonical_replay_id": replay_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(results)),
        "unique_experiment_ids": int(results["experiment_id"].nunique()),
        "research_replays_complete": int(
            status_counts.get("RESEARCH_REPLAY_COMPLETE", 0)
            + status_counts.get("SHORT_HISTORY_RESEARCH_REPLAY_COMPLETE", 0)
        ),
        "full_history_research_replays_complete": int(
            status_counts.get("RESEARCH_REPLAY_COMPLETE", 0)
        ),
        "short_history_research_replays_complete": int(
            status_counts.get("SHORT_HISTORY_RESEARCH_REPLAY_COMPLETE", 0)
        ),
        "blocked_point_in_time_history": int(status_counts.get("BLOCKED_POINT_IN_TIME_HISTORY", 0)),
        "blocked_dynamic_exposure_rule": int(status_counts.get("BLOCKED_DYNAMIC_EXPOSURE_RULE", 0)),
        "blocked_mode_inputs": int(status_counts.get("BLOCKED_MODE_INPUTS", 0)),
        "blocked_replay_errors": int(status_counts.get("BLOCKED_REPLAY_ERROR", 0)),
        "not_applicable_wizard_mode": int(status_counts.get("NOT_APPLICABLE_WIZARD_MODE", 0)),
        "trade_ledger_rows": int(len(trades)),
        "research_rank_eligible_replays": int(results["research_rank_eligible"].astype(bool).sum()),
        "minimum_research_rank_trades": MIN_RESEARCH_RANK_TRADES,
        "canonical_replay_leverage": 1.0,
        "cost_evidence_status": "PROVISIONAL_CONSERVATIVE_DEFAULTS",
        "cost_model": cost_payload,
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "input_hashes": {
            "experiment_preflight": _file_hash(experiment_path),
            "mode_ledger": _file_hash(mode_path),
            "pair_history_results": _file_hash(pair_history_path),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _base_result_row(
    experiment: object,
    *,
    replay_id: str,
    history_run_id: str,
    cost_payload: dict[str, object],
    evidence_paths: tuple[Path, ...],
    root: Path,
) -> dict[str, object]:
    orientation = _text(experiment.orientation)
    asset_x = _text(experiment.asset_x)
    asset_y = _text(experiment.asset_y)
    return {
        "schema_version": SCHEMA_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "exhaustive_run_id": _text(experiment.exhaustive_run_id),
        "replay_preflight_id": _text(experiment.replay_preflight_id),
        "history_run_id": history_run_id,
        "canonical_replay_id": replay_id,
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
        "history_validation_lane": "",
        "history_rows": 0,
        "history_acceptance_ready": False,
        "history_research_ready": False,
        "preflight_status": _text(experiment.preflight_status),
        "preflight_blocker": _text(experiment.preflight_blocker),
        "canonical_replay_leverage": 1.0,
        "taker_fee_bps": cost_payload["taker_fee_bps"],
        "slippage_bps": cost_payload["slippage_bps"],
        "execution_risk_bps": cost_payload["execution_risk_bps"],
        "funding_bps_per_day": cost_payload["funding_bps_per_day"],
        "partial_fill_probability": cost_payload["partial_fill_probability"],
        "cost_evidence_status": "PROVISIONAL_CONSERVATIVE_DEFAULTS",
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
        "evidence_path": ";".join(_relative(path, root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _mode_settings(mode_row: object, *, paired_ou: object | None) -> tuple[dict[str, object], str]:
    exact_mode = _text(mode_row.exact_mode)
    copula_direction_view = "u1_given_u2"
    lower_action, upper_action = tail_actions(
        exact_mode,
        copula_direction_view=copula_direction_view,
    )
    settings: dict[str, object] = {
        "exact_mode": exact_mode,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "capture_confirmed": True,
        "entry_long_operator": _text(getattr(mode_row, "entry_long_operator", "")),
        "entry_long_value": getattr(mode_row, "entry_long", ""),
        "entry_long_position": lower_action.value,
        "entry_short_operator": _text(getattr(mode_row, "entry_short_operator", "")),
        "entry_short_value": getattr(mode_row, "entry_short", ""),
        "entry_short_position": upper_action.value,
        "exit_long_operator": _text(getattr(mode_row, "exit_long_operator", "")),
        "exit_long_value": getattr(mode_row, "exit_long", ""),
        "exit_short_operator": _text(getattr(mode_row, "exit_short_operator", "")),
        "exit_short_value": getattr(mode_row, "exit_short", ""),
        "hedge_ratio": getattr(mode_row, "hedge_ratio", ""),
        "zscore_window": getattr(mode_row, "rolling_window", ""),
        "copula_family": _text(getattr(mode_row, "copula_family", "")),
        "copula_signal_type": "conditional_cdf_tail_dislocation",
        "copula_direction_view": copula_direction_view,
        "copula_entry_lower": getattr(mode_row, "entry_long", ""),
        "copula_entry_upper": getattr(mode_row, "entry_short", ""),
        "copula_exit_lower": getattr(mode_row, "exit_long", ""),
        "copula_exit_upper": getattr(mode_row, "exit_short", ""),
    }
    setting_source = "direct_exact_mode_capture"
    if exact_mode == "OU (ZScoreR)" and paired_ou is not None:
        settings["ou_mu"] = getattr(paired_ou, "ou_mu", "")
        settings["ou_sigma"] = getattr(paired_ou, "ou_sigma", "")
        setting_source = "direct_exact_mode_capture+paired_orientation_ou_spread_parameters"
    else:
        settings["ou_mu"] = getattr(mode_row, "ou_mu", "")
        settings["ou_sigma"] = getattr(mode_row, "ou_sigma", "")
    return settings, setting_source


def _load_history(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = pd.DataFrame(payload.get("history", []))
    if history.empty or not {"timestamp", "price_x", "price_y"}.issubset(history.columns):
        raise ValueError("pair_history_missing_required_columns")
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp")
    history = history.set_index("timestamp", drop=False)
    if history.empty or history.index.duplicated().any():
        raise ValueError("pair_history_timestamp_identity_invalid")
    return history


def _orient_history(history: pd.DataFrame, *, orientation: str) -> pd.DataFrame:
    oriented = history.copy()
    if orientation == "original":
        return oriented
    if orientation != "reverse":
        raise ValueError(f"unsupported_orientation:{orientation}")
    for left, right in (
        ("price_x", "price_y"),
        ("open_x", "open_y"),
        ("volume_x_usd", "volume_y_usd"),
        ("funding_x_bps", "funding_y_bps"),
        ("funding_x_realized_bps", "funding_y_realized_bps"),
        ("slippage_x_model_bps", "slippage_y_model_bps"),
    ):
        if left in oriented.columns and right in oriented.columns:
            original_left = oriented[left].copy()
            oriented[left] = oriented[right]
            oriented[right] = original_left
    return oriented


def _unique_mode_rows(
    frame: pd.DataFrame,
    *,
    allowed_pair_group_ids: set[str] | None = None,
) -> dict[tuple[str, str, str], object]:
    rows: dict[tuple[str, str, str], object] = {}
    for row in frame.itertuples():
        pair_group_id = _text(row.pair_group_id)
        if allowed_pair_group_ids is not None and pair_group_id not in allowed_pair_group_ids:
            continue
        key = (pair_group_id, _text(row.exact_mode), _text(row.orientation))
        if key in rows:
            raise ValueError(f"Duplicate mode-ledger identity: {key}")
        rows[key] = row
    return rows


def _research_rank_blocker(row: pd.Series) -> str:
    blockers: list[str] = []
    if _text(row.get("history_validation_lane")) == "SHORT_HISTORY_RESEARCH_ONLY":
        blockers.append("short_history_research_only")
    trades = int(_number(row.get("trades")) or 0)
    profit_factor = _number(row.get("profit_factor"))
    total_return = _number(row.get("total_return"))
    expectancy = _number(row.get("expectancy"))
    if trades < MIN_RESEARCH_RANK_TRADES:
        blockers.append(f"closed_trades<{MIN_RESEARCH_RANK_TRADES}")
    if profit_factor is None or not math.isfinite(profit_factor):
        blockers.append("profit_factor_not_finite")
    if _text(row.get("sharpe_status")) != "valid":
        blockers.append("sharpe_not_valid")
    if total_return is None or total_return <= 0.0:
        blockers.append("after_cost_total_return_not_positive")
    if expectancy is None or expectancy <= 0.0:
        blockers.append("closed_trade_expectancy_not_positive")
    return ";".join(blockers)


def _unique_rows(frame: pd.DataFrame, key: str) -> dict[str, object]:
    if frame.empty or key not in frame.columns:
        return {}
    if frame[key].astype(str).duplicated().any():
        raise ValueError(f"Expected unique {key} rows")
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Canonical 1x Replay",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Canonical replay: `{summary['canonical_replay_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Research replays complete: {summary['research_replays_complete']}",
            f"- Full-history research replays: {summary['full_history_research_replays_complete']}",
            f"- Short-history research replays: {summary['short_history_research_replays_complete']}",
            f"- Point-in-time history blocked: {summary['blocked_point_in_time_history']}",
            f"- Dynamic exposure rule blocked: {summary['blocked_dynamic_exposure_rule']}",
            f"- Mode inputs blocked: {summary['blocked_mode_inputs']}",
            f"- Replay errors: {summary['blocked_replay_errors']}",
            f"- Trade-ledger rows: {summary['trade_ledger_rows']}",
            f"- Research-rank eligible: {summary['research_rank_eligible_replays']} (minimum {summary['minimum_research_rank_trades']} closed trades, finite profit factor, valid Sharpe, positive after-cost return and expectancy)",
            f"- Acceptance-eligible replays: {summary['acceptance_eligible_replays']}",
            f"- Canonical leverage: {summary['canonical_replay_leverage']}x",
            f"- Cost evidence: `{summary['cost_evidence_status']}`",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Ranks are research diagnostics only. Short-history replays are always rank-ineligible and acceptance-blocked. Observed Hyperliquid funding and slippage, purged walk-forward evidence, stability tests, and deterministic acceptance gates remain required.",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _number(value: object) -> float | None:
    try:
        if value is None or _text(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _truthy(value: object) -> bool:
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
