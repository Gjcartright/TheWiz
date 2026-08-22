"""Parameter and cost robustness for exhaustive Hyperliquid research passes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.economic_contract import y_on_x_log_spread
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_canonical_replay import (
    _load_history,
    _mode_settings,
    _orient_history,
    _unique_mode_rows,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_observed_cost_replay import (
    _longest_observed_funding_segment,
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
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_robustness.v1"
MIN_PARAMETER_POSITIVE_RATIO = 0.75
MIN_PARAMETER_PASS_RATIO = 0.60
MAX_STRESS_DRAWDOWN = 0.60
BASELINE_TOLERANCE = 1e-10
RESEARCH_ONLY_REASON = (
    "robustness_is_research_only;statistical_selection_not_passed;"
    "current_l2_depth_is_point_in_time_not_historical_execution_evidence;"
    "mode_fidelity_parity_not_proven"
)

SCENARIOS: tuple[dict[str, object], ...] = (
    {"scenario": "baseline", "category": "baseline"},
    {"scenario": "entry_tighter_10", "category": "parameter", "entry_factor": 1.10},
    {"scenario": "entry_looser_10", "category": "parameter", "entry_factor": 0.90},
    {"scenario": "window_shorter_20", "category": "parameter", "window_factor": 0.80},
    {"scenario": "window_longer_20", "category": "parameter", "window_factor": 1.20},
    {"scenario": "hedge_ratio_lower_10", "category": "parameter", "hedge_factor": 0.90},
    {"scenario": "hedge_ratio_higher_10", "category": "parameter", "hedge_factor": 1.10},
    {"scenario": "slippage_150", "category": "cost", "slippage_factor": 1.50},
    {"scenario": "taker_fee_plus_2bps", "category": "cost", "fee_add_bps": 2.0},
    {"scenario": "trading_costs_125", "category": "cost", "trading_cost_factor": 1.25},
    {
        "scenario": "adverse_absolute_funding_150",
        "category": "funding",
        "funding_factor": 1.50,
        "adverse_funding": True,
    },
)


def run_exhaustive_wizard_hyperliquid_robustness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Run deterministic parameter and cost stress over practical OOS passes."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    walkforward_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_walkforward_manifest.json"
    )
    regime_manifest_path = active / "exhaustive_wizard_hyperliquid_regime_manifest.json"
    if not walkforward_manifest_path.exists() or not regime_manifest_path.exists():
        raise FileNotFoundError("Walk-forward and regime manifests are required")
    walkforward_manifest = json.loads(
        walkforward_manifest_path.read_text(encoding="utf-8")
    )
    regime_manifest = json.loads(regime_manifest_path.read_text(encoding="utf-8"))
    if _text(regime_manifest.get("walkforward_id")) != _text(
        walkforward_manifest.get("walkforward_id")
    ):
        raise ValueError("Regime attribution does not match the active walk-forward run")

    artifacts = walkforward_manifest.get("artifacts", {})
    snapshots = walkforward_manifest.get("input_snapshots", {})
    input_paths = {
        "walkforward_status": root / _text(artifacts.get("snapshot_status")),
        "walkforward_candidates": root / _text(artifacts.get("snapshot_candidates")),
        "mode_ledger": root / _text(snapshots.get("mode_ledger")),
        "pair_cost_evidence": root / _text(snapshots.get("pair_cost_evidence")),
        "walkforward_manifest": walkforward_manifest_path,
        "regime_manifest": regime_manifest_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Robustness inputs missing: {missing}")

    statuses = _read_csv(input_paths["walkforward_status"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    modes = _read_csv(input_paths["mode_ledger"])
    pair_costs = _read_csv(input_paths["pair_cost_evidence"])
    _require_unique(statuses, "experiment_id")
    _require_unique(candidates, "experiment_id", allow_empty=True)
    _require_unique(pair_costs, "pair_group_id")

    run_id = _text(walkforward_manifest.get("run_id"))
    walkforward_id = _text(walkforward_manifest.get("walkforward_id"))
    regime_attribution_id = _text(regime_manifest.get("regime_attribution_id"))
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
        "run_id": run_id,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": regime_attribution_id,
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    material_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    robustness_id = f"hlrobust_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{material_hash[:8]}"
    walkforward_snapshot_manifest = root / _text(artifacts.get("snapshot_manifest"))
    snapshot_dir = walkforward_snapshot_manifest.parent / "robustness" / robustness_id
    snapshot_input_dir = snapshot_dir / "inputs"
    snapshot_input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = immutable_snapshot_copy(
            source,
            snapshot_input_dir,
            artifact_name=name,
        )
        snapshot_inputs[name] = target

    paths = {
        "status": active / "exhaustive_wizard_hyperliquid_robustness_status.csv",
        "candidates": active / "exhaustive_wizard_hyperliquid_robustness_candidates.csv",
        "scenarios": active / "exhaustive_wizard_hyperliquid_robustness_scenarios.csv",
        "folds": active / "exhaustive_wizard_hyperliquid_robustness_folds.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_robustness_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_robustness_summary.md",
        "snapshot_status": snapshot_dir / "robustness_status.csv",
        "snapshot_candidates": snapshot_dir / "robustness_candidates.csv",
        "snapshot_scenarios": snapshot_dir / "robustness_scenarios.csv",
        "snapshot_folds": snapshot_dir / "robustness_folds.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }

    active_pair_group_ids = {_text(value) for value in statuses["pair_group_id"]}
    if "" in active_pair_group_ids:
        raise ValueError("Robustness status identity missing pair_group_id")
    mode_lookup = _unique_mode_rows(
        modes,
        allowed_pair_group_ids=active_pair_group_ids,
    )
    pair_lookup = _row_lookup(pair_costs, "pair_group_id")
    selected = candidates.loc[
        candidates["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
    ].copy()
    selected_lookup = _row_lookup(selected, "experiment_id")
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
                    "robustness_blocker": (
                        _text(getattr(status, "walkforward_blocker", ""))
                        or _text(getattr(status, "walkforward_status", ""))
                    ),
                }
            )
            continue

        pair_group_id = _text(candidate.pair_group_id)
        exact_mode = _text(candidate.exact_mode)
        orientation = _text(candidate.orientation)
        pair_cost = pair_lookup.get(pair_group_id)
        mode_row = mode_lookup.get((pair_group_id, exact_mode, orientation))
        if pair_cost is None or mode_row is None:
            status_rows.append(
                {
                    **base,
                    "robustness_status": "BLOCKED_ROBUSTNESS_INPUTS",
                    "robustness_blocker": (
                        "pair_cost_evidence_missing"
                        if pair_cost is None
                        else "captured_mode_row_missing"
                    ),
                }
            )
            continue
        try:
            history_path = root / _text(pair_cost.enriched_history_path)
            cache_key = f"{history_path}:{_text(candidate.hyperliquid_interval)}"
            if cache_key not in history_cache:
                history_cache[cache_key] = _longest_observed_funding_segment(
                    _load_history(history_path),
                    interval=_text(candidate.hyperliquid_interval),
                )
            raw_history = history_cache[cache_key].copy()
            raw_history["slippage_x_model_bps"] = _required_number(
                pair_cost.slippage_x_p95_bps,
                "slippage_x_p95_bps",
            )
            raw_history["slippage_y_model_bps"] = _required_number(
                pair_cost.slippage_y_p95_bps,
                "slippage_y_p95_bps",
            )
            history = _orient_history(raw_history, orientation=orientation)
            folds = _build_folds(len(history))
            paired_ou = mode_lookup.get((pair_group_id, "OU (Spread)", orientation))
            captured_settings, setting_source = _mode_settings(mode_row, paired_ou=paired_ou)
            baseline_costs = CostModel(
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
        except Exception as exc:
            status_rows.append(
                {
                    **base,
                    "robustness_status": "BLOCKED_ROBUSTNESS_INPUTS",
                    "robustness_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue

        experiment_scenarios: list[dict[str, object]] = []
        experiment_blocker = ""
        for scenario in SCENARIOS:
            scenario_fold_rows: list[dict[str, object]] = []
            scenario_trades: list[pd.DataFrame] = []
            scenario_bars: list[pd.DataFrame] = []
            scenario_name = _text(scenario["scenario"])
            for fold in folds:
                try:
                    train = history.iloc[: fold["train_end"]].copy()
                    fitted_settings, fit_diagnostics = _fit_training_parameters(
                        train,
                        captured_settings,
                        exact_mode=exact_mode,
                    )
                    scenario_settings = _scenario_settings(
                        fitted_settings,
                        train,
                        exact_mode=exact_mode,
                        scenario=scenario,
                    )
                    signal_history = history.iloc[: fold["test_end"]].copy()
                    hedge_ratio = _number(scenario_settings.get("hedge_ratio"))
                    if hedge_ratio is not None:
                        signal_history["hedge_ratio"] = hedge_ratio
                    signal_history, scenario_costs = _scenario_costs_and_history(
                        signal_history,
                        baseline_costs,
                        scenario=scenario,
                    )
                    mode_result = build_local_mode_signal(
                        signal_history,
                        scenario_settings,
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
                        scenario_costs,
                        interval=_text(candidate.hyperliquid_interval),
                    )
                except Exception as exc:
                    experiment_blocker = (
                        f"{scenario_name}:fold_{fold['fold_number']}:"
                        f"{safe_exception_code(exc)}"
                    )
                    break

                fold_metric = {
                    **_identity_fields(candidate),
                    "schema_version": SCHEMA_VERSION,
                    "robustness_id": robustness_id,
                    "walkforward_id": walkforward_id,
                    "scenario": scenario_name,
                    "scenario_category": _text(scenario["category"]),
                    "fold_number": fold["fold_number"],
                    "fold_status": "COMPLETE",
                    "fold_blocker": "",
                    "train_rows": fold["train_end"],
                    "embargo_rows": fold["test_start"] - fold["train_end"],
                    "test_rows": fold["test_end"] - fold["test_start"],
                    "mode_setting_source": setting_source,
                    "fitted_hedge_ratio": _number(scenario_settings.get("hedge_ratio")),
                    "fitted_ou_mu": _number(scenario_settings.get("ou_mu")),
                    "fitted_ou_sigma": _number(scenario_settings.get("ou_sigma")),
                    "fit_uses_test_data": False,
                    **asdict(result),
                    "acceptance_status": "BLOCKED",
                    "acceptance_reason": RESEARCH_ONLY_REASON,
                    "live_trading_authorized": False,
                }
                scenario_fold_rows.append(fold_metric)
                if not ledger.closed_trades.empty:
                    scenario_trades.append(ledger.closed_trades.copy())
                scenario_bars.append(ledger.bar_ledger.copy())
            if experiment_blocker:
                break
            aggregate = _aggregate_candidate(
                scenario_fold_rows,
                scenario_trades,
                scenario_bars,
                interval=_text(candidate.hyperliquid_interval),
            )
            scenario_blockers = _walkforward_gate_blockers(aggregate)
            scenario_row = {
                **_identity_fields(candidate),
                "schema_version": SCHEMA_VERSION,
                "robustness_id": robustness_id,
                "walkforward_id": walkforward_id,
                "scenario": scenario_name,
                "scenario_category": _text(scenario["category"]),
                **aggregate,
                "scenario_status": (
                    "PASS_RESEARCH_SCENARIO" if not scenario_blockers else "FAIL_RESEARCH_SCENARIO"
                ),
                "scenario_blocker": ";".join(scenario_blockers),
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
                    "robustness_blocker": experiment_blocker or "scenario_accounting_incomplete",
                }
            )
            continue
        summary = _candidate_robustness_summary(candidate, experiment_scenarios)
        result = {
            **base,
            **summary,
            "robustness_status": "ROBUSTNESS_COMPLETE",
            "robustness_blocker": "",
            "acceptance_status": "BLOCKED",
            "acceptance_reason": _downstream_acceptance_reason(candidate),
            "acceptance_eligible": False,
            "live_trading_authorized": False,
        }
        status_rows.append(result)
        candidate_rows.append(result)

    status_frame = pd.DataFrame(status_rows)
    if len(status_frame) != len(statuses) or status_frame["experiment_id"].nunique() != len(
        statuses
    ):
        raise ValueError("Robustness failed complete experiment accounting")
    robustness_status_defaults = {
        "scenarios_complete": 0,
        "parameter_positive_ratio": "",
        "parameter_pass_ratio": "",
        "worst_parameter_drawdown": "",
        "research_robustness_status": "NOT_EVALUATED",
        "research_robustness_blocker": "prior_walk_forward_gate_not_passed",
        "promotion_readiness": "BLOCKED",
        "promotion_blocker": "prior_walk_forward_gate_not_passed",
    }
    for column, default in robustness_status_defaults.items():
        status_frame[column] = status_frame.get(
            column,
            pd.Series(index=status_frame.index, dtype=object),
        ).fillna(default)
    candidate_frame = (
        pd.DataFrame(candidate_rows)
        if candidate_rows
        else status_frame.iloc[0:0].copy()
    )
    scenario_frame = (
        pd.DataFrame(scenario_rows)
        if scenario_rows
        else pd.DataFrame(columns=["experiment_id", "scenario", "category"])
    )
    fold_frame = (
        pd.DataFrame(fold_rows)
        if fold_rows
        else pd.DataFrame(columns=["experiment_id", "scenario", "fold_number"])
    )
    for frame, active_path, snapshot_path in (
        (status_frame, paths["status"], paths["snapshot_status"]),
        (candidate_frame, paths["candidates"], paths["snapshot_candidates"]),
        (scenario_frame, paths["scenarios"], paths["snapshot_scenarios"]),
        (fold_frame, paths["folds"], paths["snapshot_folds"]),
    ):
        atomic_write_csv(frame, active_path, index=False)
        atomic_write_csv(frame, snapshot_path, index=False)

    status_counts = _status_counts(status_frame, "robustness_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "walkforward_id": walkforward_id,
        "regime_attribution_id": regime_attribution_id,
        "robustness_id": robustness_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(status_frame)),
        "unique_experiment_ids": int(status_frame["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status_frame)),
        "walkforward_research_passes_selected": int(len(selected)),
        "robustness_candidates_complete": int(len(candidate_frame)),
        "research_robustness_passes": int(
            candidate_frame.get("research_robustness_status", pd.Series(dtype=str))
            .eq("PASS_RESEARCH_ROBUSTNESS")
            .sum()
        ),
        "statistically_selected_robustness_passes": int(
            candidate_frame.get("promotion_readiness", pd.Series(dtype=str))
            .eq("READY_FOR_NEXT_RESEARCH_GATE")
            .sum()
        ),
        "scenario_rows": int(len(scenario_frame)),
        "fold_rows": int(len(fold_frame)),
        "expected_scenario_rows": int(len(candidate_frame) * len(SCENARIOS)),
        "expected_fold_rows": int(len(candidate_frame) * len(SCENARIOS) * FOLD_COUNT),
        "policy": policy,
        "acceptance_eligible_replays": 0,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _scenario_settings(
    fitted: dict[str, object],
    train: pd.DataFrame,
    *,
    exact_mode: str,
    scenario: dict[str, object],
) -> dict[str, object]:
    settings = dict(fitted)
    entry_factor = _number(scenario.get("entry_factor"))
    if entry_factor is not None:
        if exact_mode == "Copula":
            lower = _required_number(settings.get("copula_entry_lower"), "copula_entry_lower")
            upper = _required_number(settings.get("copula_entry_upper"), "copula_entry_upper")
            if entry_factor > 1.0:
                lower = lower / entry_factor
                upper = 1.0 - (1.0 - upper) / entry_factor
            else:
                lower = min(0.499, lower / max(entry_factor, 1e-6))
                upper = max(0.501, 1.0 - (1.0 - upper) / max(entry_factor, 1e-6))
            settings["copula_entry_lower"] = max(0.001, min(0.499, lower))
            settings["copula_entry_upper"] = min(0.999, max(0.501, upper))
        else:
            settings["entry_long_value"] = _required_number(
                settings.get("entry_long_value"), "entry_long_value"
            ) * entry_factor
            settings["entry_short_value"] = _required_number(
                settings.get("entry_short_value"), "entry_short_value"
            ) * entry_factor

    window_factor = _number(scenario.get("window_factor"))
    if window_factor is not None:
        if exact_mode == "Copula":
            base_window = int(_number(settings.get("copula_window")) or 120)
            settings["copula_window"] = max(20, int(round(base_window * window_factor)))
        else:
            base_window = int(_required_number(settings.get("zscore_window"), "zscore_window"))
            settings["zscore_window"] = max(5, int(round(base_window * window_factor)))

    hedge_factor = _number(scenario.get("hedge_factor"))
    if hedge_factor is not None:
        beta = _required_number(settings.get("hedge_ratio"), "hedge_ratio") * hedge_factor
        settings["hedge_ratio"] = beta
        if exact_mode.startswith("OU"):
            prices = train[["price_x", "price_y"]].apply(pd.to_numeric, errors="coerce")
            prices = prices.loc[
                (prices["price_x"] > 0.0) & (prices["price_y"] > 0.0)
            ].dropna()
            spread = y_on_x_log_spread(
                prices["price_x"], prices["price_y"], beta
            )
            mu = float(spread.mean())
            sigma = float(spread.std(ddof=0))
            if not math.isfinite(mu) or not math.isfinite(sigma) or sigma <= 1e-12:
                raise ValueError("perturbed_ou_parameters_invalid")
            settings["ou_mu"] = mu
            settings["ou_sigma"] = sigma
    return settings


def _scenario_costs_and_history(
    history: pd.DataFrame,
    baseline: CostModel,
    *,
    scenario: dict[str, object],
) -> tuple[pd.DataFrame, CostModel]:
    adjusted = history.copy()
    costs = baseline
    slippage_factor = _number(scenario.get("slippage_factor"))
    if slippage_factor is not None:
        adjusted["slippage_x_model_bps"] *= slippage_factor
        adjusted["slippage_y_model_bps"] *= slippage_factor
    fee_add = _number(scenario.get("fee_add_bps"))
    if fee_add is not None:
        costs = replace(costs, taker_fee_bps=costs.taker_fee_bps + fee_add)
    trading_cost_factor = _number(scenario.get("trading_cost_factor"))
    if trading_cost_factor is not None:
        adjusted["slippage_x_model_bps"] *= trading_cost_factor
        adjusted["slippage_y_model_bps"] *= trading_cost_factor
        costs = replace(
            costs,
            taker_fee_bps=costs.taker_fee_bps * trading_cost_factor,
            execution_risk_bps=costs.execution_risk_bps * trading_cost_factor,
            partial_fill_penalty_bps=(
                costs.partial_fill_penalty_bps * trading_cost_factor
            ),
        )
    funding_factor = _number(scenario.get("funding_factor"))
    if funding_factor is not None:
        adjusted["funding_x_realized_bps"] *= funding_factor
        adjusted["funding_y_realized_bps"] *= funding_factor
    if bool(scenario.get("adverse_funding")):
        costs = replace(
            costs,
            funding_policy=FundingPolicy.CONSERVATIVE_ABSOLUTE_DRAG.value,
        )
    return adjusted, costs


def _candidate_robustness_summary(
    candidate: object,
    scenarios: list[dict[str, object]],
) -> dict[str, object]:
    lookup = {_text(row["scenario"]): row for row in scenarios}
    baseline = lookup["baseline"]
    baseline_error = abs(
        _required_number(baseline.get("aggregate_total_return"), "baseline_total_return")
        - _required_number(candidate.aggregate_total_return, "candidate_total_return")
    )
    parameter_rows = [row for row in scenarios if _text(row["scenario_category"]) == "parameter"]
    parameter_positive = sum(
        (_number(row.get("aggregate_total_return")) or 0.0) > 0.0 for row in parameter_rows
    )
    parameter_passes = sum(
        _text(row.get("scenario_status")) == "PASS_RESEARCH_SCENARIO"
        for row in parameter_rows
    )
    positive_ratio = parameter_positive / len(parameter_rows) if parameter_rows else 0.0
    pass_ratio = parameter_passes / len(parameter_rows) if parameter_rows else 0.0
    worst_parameter_drawdown = max(
        (_number(row.get("aggregate_max_drawdown")) or 0.0) for row in parameter_rows
    )
    required_cost_scenarios = [
        "slippage_150",
        "taker_fee_plus_2bps",
        "trading_costs_125",
        "adverse_absolute_funding_150",
    ]
    cost_positive = {
        name: (_number(lookup[name].get("aggregate_total_return")) or 0.0) > 0.0
        for name in required_cost_scenarios
    }
    research_blockers: list[str] = []
    if baseline_error > BASELINE_TOLERANCE:
        research_blockers.append("baseline_reconciliation_failed")
    if positive_ratio < MIN_PARAMETER_POSITIVE_RATIO:
        research_blockers.append(
            f"parameter_positive_ratio<{MIN_PARAMETER_POSITIVE_RATIO:g}"
        )
    if pass_ratio < MIN_PARAMETER_PASS_RATIO:
        research_blockers.append(f"parameter_pass_ratio<{MIN_PARAMETER_PASS_RATIO:g}")
    if worst_parameter_drawdown > MAX_STRESS_DRAWDOWN:
        research_blockers.append(f"worst_parameter_drawdown>{MAX_STRESS_DRAWDOWN:g}")
    for name, passed in cost_positive.items():
        if not passed:
            research_blockers.append(f"{name}_after_cost_return_not_positive")
    statistical_pass = _text(candidate.statistical_selection_status) == "PASS"
    promotion_blockers = list(research_blockers)
    if not statistical_pass:
        promotion_blockers.append("statistical_selection_not_passed")
    return {
        "scenarios_complete": len(scenarios),
        "baseline_reconciliation_error": baseline_error,
        "parameter_scenarios": len(parameter_rows),
        "parameter_positive_scenarios": parameter_positive,
        "parameter_pass_scenarios": parameter_passes,
        "parameter_positive_ratio": positive_ratio,
        "parameter_pass_ratio": pass_ratio,
        "worst_parameter_drawdown": worst_parameter_drawdown,
        **{f"{name}_positive": passed for name, passed in cost_positive.items()},
        "research_robustness_status": (
            "PASS_RESEARCH_ROBUSTNESS"
            if not research_blockers
            else "FAIL_RESEARCH_ROBUSTNESS"
        ),
        "research_robustness_blocker": ";".join(research_blockers),
        "statistical_selection_status": _text(candidate.statistical_selection_status),
        "bh_qvalue": _number(candidate.bh_qvalue),
        "promotion_readiness": (
            "READY_FOR_NEXT_RESEARCH_GATE" if not promotion_blockers else "BLOCKED"
        ),
        "promotion_blocker": ";".join(promotion_blockers),
    }


def _status_base(
    row: object,
    *,
    robustness_id: str,
    evidence_paths: object,
    root: Path,
) -> dict[str, object]:
    return {
        **_identity_fields(row),
        "schema_version": SCHEMA_VERSION,
        "robustness_id": robustness_id,
        "walkforward_id": _text(getattr(row, "walkforward_id", "")),
        "prior_walkforward_status": _text(getattr(row, "walkforward_status", "")),
        "history_validation_lane": _text(
            getattr(row, "history_validation_lane", "")
        ),
        "walkforward_rank_eligible": _truthy(
            getattr(row, "walkforward_rank_eligible", False)
        ),
        "statistical_selection_status": _text(
            getattr(row, "statistical_selection_status", "")
        ),
        "statistical_selection_blocker": _text(
            getattr(row, "statistical_selection_blocker", "")
        ),
        "robustness_status": "",
        "robustness_blocker": "",
        "acceptance_status": "BLOCKED",
        "acceptance_reason": _downstream_acceptance_reason(row),
        "acceptance_eligible": False,
        "evidence_path": ";".join(_relative(Path(path), root) for path in evidence_paths),
        "live_trading_authorized": False,
    }


def _downstream_acceptance_reason(row: object) -> str:
    reasons: list[str] = []
    if _text(getattr(row, "history_validation_lane", "")) == (
        "SHORT_HISTORY_RESEARCH_ONLY"
    ):
        reasons.append("short_history_research_only")
    reasons.append(RESEARCH_ONLY_REASON)
    selection_status = _text(getattr(row, "statistical_selection_status", ""))
    selection_blocker = _text(getattr(row, "statistical_selection_blocker", ""))
    if selection_status and selection_status != "PASS":
        reasons.append(f"statistical_selection_{selection_status.lower()}")
    if selection_blocker:
        reasons.append(selection_blocker)
    return ";".join(dict.fromkeys(reason for reason in reasons if reason))


def _identity_fields(row: object) -> dict[str, object]:
    return {
        "experiment_id": _text(getattr(row, "experiment_id", "")),
        "pair_group_id": _text(getattr(row, "pair_group_id", "")),
        "pair": _text(getattr(row, "pair", "")),
        "wizard_exchange": _text(getattr(row, "wizard_exchange", "")),
        "wizard_timeframe": _text(getattr(row, "wizard_timeframe", "")),
        "hyperliquid_interval": _text(getattr(row, "hyperliquid_interval", "")),
        "exact_mode": _text(getattr(row, "exact_mode", "")),
        "orientation": _text(getattr(row, "orientation", "")),
    }


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, object]:
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _require_unique(
    frame: pd.DataFrame,
    key: str,
    *,
    allow_empty: bool = False,
) -> None:
    if frame.empty and allow_empty:
        return
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
            "# Exhaustive Hyperliquid Robustness Stress",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Walk-forward: `{summary['walkforward_id']}`",
            f"- Robustness: `{summary['robustness_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Walk-forward research passes selected: {summary['walkforward_research_passes_selected']}",
            f"- Robustness candidates complete: {summary['robustness_candidates_complete']}",
            f"- Research robustness passes: {summary['research_robustness_passes']}",
            f"- Statistically selected robustness passes: {summary['statistically_selected_robustness_passes']}",
            f"- Scenario rows: {summary['scenario_rows']} / {summary['expected_scenario_rows']}",
            f"- Fold rows: {summary['fold_rows']} / {summary['expected_fold_rows']}",
            f"- Acceptance-eligible replays: {summary['acceptance_eligible_replays']}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "This matrix reproduces the baseline and perturbs entries, windows, hedge ratios, slippage, fees, execution costs, and funding over the same purged five-fold path. A research robustness pass cannot override family-wide statistical selection or cost-calibration blockers.",
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


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


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
