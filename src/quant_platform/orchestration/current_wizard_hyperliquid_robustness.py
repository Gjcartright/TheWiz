"""Parameter and cost robustness for current-board walk-forward passes."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import CostModel, FundingPolicy, backtest_two_leg_spread_with_ledger
from quant_platform.orchestration.current_wizard_hyperliquid_replay import (
    _exposure_hedge_ratio,
    _load_history,
    _orient_history,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_robustness import (
    BASELINE_TOLERANCE,
    MAX_STRESS_DRAWDOWN,
    MIN_PARAMETER_PASS_RATIO,
    MIN_PARAMETER_POSITIVE_RATIO,
    SCENARIOS,
    _candidate_robustness_summary,
    _scenario_costs_and_history,
    _scenario_settings,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_walkforward import (
    FOLD_COUNT,
    _aggregate_candidate,
    _build_folds,
    _fit_training_parameters,
    _flat_fold_signal,
    _walkforward_gate_blockers,
)
from quant_platform.wizard_mode_replay import build_local_mode_signal


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_robustness.v1"
RESEARCH_ONLY_REASON = (
    "robustness_is_research_only;strict_l2_calibration_required;"
    "mode_fidelity_parity_not_proven"
)


def run_current_wizard_hyperliquid_robustness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Stress every practical current-board walk-forward survivor."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    walk_manifest_path = active / "current_wizard_hyperliquid_walkforward_manifest.json"
    regime_manifest_path = active / "current_wizard_hyperliquid_regime_manifest.json"
    if not walk_manifest_path.exists() or not regime_manifest_path.exists():
        raise FileNotFoundError("Current walk-forward and regime manifests are required")
    walk_manifest = _read_json(walk_manifest_path)
    regime_manifest = _read_json(regime_manifest_path)
    if _text(walk_manifest.get("walkforward_id")) != _text(
        regime_manifest.get("walkforward_id")
    ):
        raise ValueError("Current regime attribution does not match walk-forward")
    artifacts = walk_manifest.get("artifacts", {})
    snapshots = walk_manifest.get("input_snapshots", {})
    input_paths = {
        "status": root / _text(artifacts.get("snapshot_status")),
        "candidates": root / _text(artifacts.get("snapshot_candidates")),
        "pair_costs": root / _text(snapshots.get("pair_costs")),
        "observed_results": root / _text(snapshots.get("observed_results")),
        "walkforward_manifest": root / _text(artifacts.get("snapshot_manifest")),
        "regime_manifest": root / _text(regime_manifest.get("artifacts", {}).get("snapshot_manifest")),
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current robustness inputs missing: {missing}")
    statuses = pd.read_csv(input_paths["status"])
    candidates = pd.read_csv(input_paths["candidates"])
    pair_costs = pd.read_csv(input_paths["pair_costs"])
    observed = pd.read_csv(input_paths["observed_results"])
    for frame, key, label in (
        (statuses, "experiment_id", "status"),
        (candidates, "experiment_id", "candidate"),
        (pair_costs, "pair_group_key", "pair cost"),
        (observed, "experiment_id", "observed replay"),
    ):
        if frame[key].duplicated().any():
            raise ValueError(f"Current {label} input contains duplicate {key}")
    selected = candidates.loc[
        candidates["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ].copy()
    selected_lookup = {
        _text(row.experiment_id): row for row in selected.itertuples()
    }
    observed_lookup = {
        _text(row.experiment_id): row for row in observed.itertuples()
    }
    pair_lookup = {
        _text(row.pair_group_key): row for row in pair_costs.itertuples()
    }
    policy = {
        "candidate_policy": "pass_research_walk_forward_only",
        "scenario_count": len(SCENARIOS),
        "scenarios": list(SCENARIOS),
        "minimum_parameter_positive_ratio": MIN_PARAMETER_POSITIVE_RATIO,
        "minimum_parameter_pass_ratio": MIN_PARAMETER_PASS_RATIO,
        "maximum_stress_drawdown": MAX_STRESS_DRAWDOWN,
        "baseline_reconciliation_tolerance": BASELINE_TOLERANCE,
        "promotion_requires_statistical_selection": True,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "walkforward_id": _text(walk_manifest.get("walkforward_id")),
        "regime_attribution_id": _text(regime_manifest.get("regime_attribution_id")),
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    robustness_id = "cwrobust_" + sha256(_canonical_json(material).encode()).hexdigest()[:20]
    snapshot_manifest = input_paths["walkforward_manifest"]
    snapshot_dir = snapshot_manifest.parent / "robustness" / robustness_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = input_dir / source.name
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    history_cache: dict[str, pd.DataFrame] = {}
    status_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    scenario_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for status in statuses.itertuples():
        experiment_id = _text(status.experiment_id)
        base = _status_base(
            status,
            robustness_id=robustness_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        candidate = selected_lookup.get(experiment_id)
        if candidate is None:
            status_rows.append(
                {
                    **base,
                    "robustness_status": "NOT_SELECTED_PRIOR_WALK_FORWARD_GATE",
                    "robustness_blocker": _text(status.walkforward_blocker)
                    or _text(status.walkforward_status),
                }
            )
            continue
        pair_cost = pair_lookup.get(_text(candidate.pair_group_key))
        observed_row = observed_lookup.get(experiment_id)
        if pair_cost is None or observed_row is None:
            status_rows.append(
                {
                    **base,
                    "robustness_status": "BLOCKED_ROBUSTNESS_INPUTS",
                    "robustness_blocker": "pair_cost_or_observed_settings_missing",
                }
            )
            continue
        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            if str(history_path) not in history_cache:
                history_cache[str(history_path)] = _load_history(history_path)
            history = _orient_history(
                history_cache[str(history_path)],
                orientation=_text(candidate.orientation),
            )
            history["slippage_x_model_bps"] = float(
                pair_cost.pair_one_way_slippage_bps
            )
            history["slippage_y_model_bps"] = float(
                pair_cost.pair_one_way_slippage_bps
            )
            folds = _build_folds(len(history))
            captured_settings = json.loads(_text(observed_row.settings_json))
            baseline_costs = _cost_model(
                pair_cost, timeframe=_text(candidate.wizard_timeframe)
            )
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "robustness_status": "BLOCKED_ROBUSTNESS_INPUTS",
                    "robustness_blocker": f"{type(exc).__name__}:{exc}",
                }
            )
            continue
        exact_mode = _text(candidate.exact_mode)
        experiment_scenarios: list[dict[str, object]] = []
        experiment_blocker = ""
        for scenario in SCENARIOS:
            scenario_name = _text(scenario["scenario"])
            scenario_fold_rows: list[dict[str, object]] = []
            scenario_trades: list[pd.DataFrame] = []
            scenario_bars: list[pd.DataFrame] = []
            for fold in folds:
                try:
                    train = history.iloc[: fold["train_end"]].copy()
                    fitted, _ = _fit_training_parameters(
                        train, captured_settings, exact_mode=exact_mode
                    )
                    settings = _scenario_settings(
                        fitted, train, exact_mode=exact_mode, scenario=scenario
                    )
                    signal_history = history.iloc[: fold["test_end"]].copy()
                    signal_history, scenario_costs = _scenario_costs_and_history(
                        signal_history, baseline_costs, scenario=scenario
                    )
                    exposure = _exposure_hedge_ratio(
                        signal_history,
                        exact_mode=exact_mode,
                        settings=settings,
                    )
                    hedge_factor = _number(scenario.get("hedge_factor"))
                    if exact_mode.startswith("Dyn") and hedge_factor is not None:
                        exposure = exposure * hedge_factor
                    signal_history["hedge_ratio"] = exposure
                    mode_result = build_local_mode_signal(
                        signal_history, settings, exact_mode=exact_mode
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
                        scenario_costs,
                        interval=_interval(_text(candidate.wizard_timeframe)),
                    )
                except Exception as exc:
                    experiment_blocker = (
                        f"{scenario_name}:fold_{fold['fold_number']}:"
                        f"{type(exc).__name__}:{exc}"
                    )
                    break
                fold_row = {
                    **_identity(candidate),
                    "schema_version": SCHEMA_VERSION,
                    "robustness_id": robustness_id,
                    "walkforward_id": material["walkforward_id"],
                    "scenario": scenario_name,
                    "scenario_category": _text(scenario["category"]),
                    "fold_number": fold["fold_number"],
                    "fold_status": "COMPLETE",
                    "fold_blocker": "",
                    "train_rows": fold["train_end"],
                    "embargo_rows": fold["test_start"] - fold["train_end"],
                    "test_rows": fold["test_end"] - fold["test_start"],
                    "fitted_hedge_ratio": _number(settings.get("hedge_ratio")),
                    "fitted_ou_mu": _number(settings.get("ou_mu")),
                    "fitted_ou_sigma": _number(settings.get("ou_sigma")),
                    "fit_uses_test_data": False,
                    **result.__dict__,
                    "acceptance_status": "BLOCKED",
                    "acceptance_reason": RESEARCH_ONLY_REASON,
                    "live_trading_authorized": False,
                }
                scenario_fold_rows.append(fold_row)
                if not ledger.closed_trades.empty:
                    scenario_trades.append(ledger.closed_trades.copy())
                scenario_bars.append(ledger.bar_ledger.copy())
            if experiment_blocker:
                break
            aggregate = _aggregate_candidate(
                scenario_fold_rows,
                scenario_trades,
                scenario_bars,
                interval=_interval(_text(candidate.wizard_timeframe)),
            )
            blockers = _walkforward_gate_blockers(aggregate)
            scenario_row = {
                **_identity(candidate),
                "schema_version": SCHEMA_VERSION,
                "robustness_id": robustness_id,
                "walkforward_id": material["walkforward_id"],
                "scenario": scenario_name,
                "scenario_category": _text(scenario["category"]),
                **aggregate,
                "scenario_status": (
                    "PASS_RESEARCH_SCENARIO"
                    if not blockers
                    else "FAIL_RESEARCH_SCENARIO"
                ),
                "scenario_blocker": ";".join(blockers),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "live_trading_authorized": False,
            }
            experiment_scenarios.append(scenario_row)
            scenario_rows.append(scenario_row)
            fold_rows.extend(scenario_fold_rows)
        if experiment_blocker or len(experiment_scenarios) != len(SCENARIOS):
            status_rows.append(
                {
                    **base,
                    "robustness_status": "BLOCKED_ROBUSTNESS_REPLAY",
                    "robustness_blocker": experiment_blocker
                    or "scenario_accounting_incomplete",
                }
            )
            continue
        candidate_summary = _candidate_robustness_summary(
            candidate, experiment_scenarios
        )
        result = {
            **base,
            **candidate_summary,
            "robustness_status": "ROBUSTNESS_COMPLETE",
            "robustness_blocker": "",
            "acceptance_status": "BLOCKED",
            "acceptance_reason": RESEARCH_ONLY_REASON,
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        status_rows.append(result)
        candidate_rows.append(result)

    status_frame = pd.DataFrame(status_rows)
    candidate_frame = pd.DataFrame(candidate_rows)
    scenario_frame = pd.DataFrame(scenario_rows)
    fold_frame = pd.DataFrame(fold_rows)
    if len(status_frame) != len(statuses) or status_frame["experiment_id"].nunique() != len(statuses):
        raise ValueError("Current robustness failed complete experiment accounting")
    validation = _validation(
        statuses, status_frame, candidate_frame, scenario_frame, fold_frame
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current robustness validation failed: " + ",".join(failed))
    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (status_frame, "status", "snapshot_status"),
        (candidate_frame, "candidates", "snapshot_candidates"),
        (scenario_frame, "scenarios", "snapshot_scenarios"),
        (fold_frame, "folds", "snapshot_folds"),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)
    counts = status_frame["robustness_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "robustness_id": robustness_id,
        "experiments_accounted": int(len(status_frame)),
        "status_counts": counts,
        "experiment_status_accounted": bool(sum(counts.values()) == len(status_frame)),
        "walkforward_research_passes_selected": int(len(selected)),
        "robustness_candidates_complete": int(len(candidate_frame)),
        "research_robustness_passes": int(candidate_frame.get("research_robustness_status", pd.Series(dtype=str)).eq("PASS_RESEARCH_ROBUSTNESS").sum()),
        "statistically_selected_robustness_passes": int(candidate_frame.get("promotion_readiness", pd.Series(dtype=str)).eq("READY_FOR_NEXT_RESEARCH_GATE").sum()),
        "scenario_rows": int(len(scenario_frame)),
        "fold_rows": int(len(fold_frame)),
        "expected_scenario_rows": int(len(candidate_frame) * len(SCENARIOS)),
        "expected_fold_rows": int(len(candidate_frame) * len(SCENARIOS) * FOLD_COUNT),
        "acceptance_eligible_replays": 0,
        "promotion_authority": False,
        "live_trading_authorized": False,
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


def _cost_model(pair_cost: object, *, timeframe: str) -> CostModel:
    return CostModel(
        taker_fee_bps=float(pair_cost.taker_fee_bps),
        slippage_bps=float(pair_cost.pair_one_way_slippage_bps),
        execution_risk_bps=float(pair_cost.execution_risk_bps),
        funding_bps_per_day=0.0,
        bars_per_day=24 if _interval(timeframe) == "1h" else 1,
        funding_policy=FundingPolicy.SIGNED_REALIZED.value,
    )


def _status_base(
    row: object,
    *,
    robustness_id: str,
    evidence_paths: Any,
    root: Path,
) -> dict[str, object]:
    return {
        **_identity(row),
        "schema_version": SCHEMA_VERSION,
        "robustness_id": robustness_id,
        "walkforward_id": _text(row.walkforward_id),
        "prior_walkforward_status": _text(row.walkforward_status),
        "robustness_status": "",
        "robustness_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "robustness_not_complete",
        "acceptance_eligible": False,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _identity(row: object) -> dict[str, object]:
    return {
        "experiment_id": _text(row.experiment_id),
        "pair_group_key": _text(row.pair_group_key),
        "pair": _text(row.pair),
        "wizard_exchange": _text(row.wizard_exchange),
        "wizard_timeframe": _text(row.wizard_timeframe),
        "hyperliquid_interval": _text(row.hyperliquid_interval),
        "exact_mode": _text(row.exact_mode),
        "orientation": _text(row.orientation),
        "asset_x": _text(getattr(row, "asset_x", "")),
        "asset_y": _text(getattr(row, "asset_y", "")),
    }


def _validation(
    prior: pd.DataFrame,
    status: pd.DataFrame,
    candidates: pd.DataFrame,
    scenarios: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    expected_scenarios = len(candidates) * len(SCENARIOS)
    expected_folds = expected_scenarios * FOLD_COUNT
    checks = {
        "experiment_count_preserved": len(prior) == len(status),
        "experiment_ids_unique": status["experiment_id"].nunique() == len(status),
        "statuses_accounted": status["robustness_status"].astype(str).ne("").all(),
        "scenario_accounting": len(scenarios) == expected_scenarios,
        "fold_accounting": len(folds) == expected_folds,
        "fit_uses_no_test_data": folds.empty or not folds["fit_uses_test_data"].astype(bool).any(),
        "folds_end_flat": folds.empty or pd.to_numeric(folds["open_trades"], errors="coerce").fillna(1).eq(0).all(),
        "baseline_reconciles": candidates.empty or pd.to_numeric(candidates["baseline_reconciliation_error"], errors="coerce").le(BASELINE_TOLERANCE).all(),
        "acceptance_disabled": status["acceptance_status"].eq("BLOCKED").all(),
        "live_trading_disabled": not status["live_trading_authorized"].astype(bool).any(),
    }
    return pd.DataFrame(
        [{"check": check, "status": "PASS" if passed else "FAIL"} for check, passed in checks.items()]
    )


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    return {
        "status": active / "current_wizard_hyperliquid_robustness_status.csv",
        "candidates": active / "current_wizard_hyperliquid_robustness_candidates.csv",
        "scenarios": active / "current_wizard_hyperliquid_robustness_scenarios.csv",
        "folds": active / "current_wizard_hyperliquid_robustness_folds.csv",
        "validation": active / "current_wizard_hyperliquid_robustness_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_robustness_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_robustness_summary.md",
        "snapshot_status": snapshot / "robustness_status.csv",
        "snapshot_candidates": snapshot / "robustness_candidates.csv",
        "snapshot_scenarios": snapshot / "robustness_scenarios.csv",
        "snapshot_folds": snapshot / "robustness_folds.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Robustness",
            "",
            f"- Robustness: `{summary['robustness_id']}`",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Candidates complete: {summary['robustness_candidates_complete']}",
            f"- Research robustness passes: {summary['research_robustness_passes']}",
            f"- Statistical robustness passes: {summary['statistically_selected_robustness_passes']}",
            f"- Scenario rows: {summary['scenario_rows']} / {summary['expected_scenario_rows']}",
            f"- Fold rows: {summary['fold_rows']} / {summary['expected_fold_rows']}",
            "- Promotion authority: no",
            "- Live trading authorized: no",
            "",
        ]
    )


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


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
