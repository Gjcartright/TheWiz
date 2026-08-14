"""Evidence-gated Hyperliquid leverage and margin research surface."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_leverage.v1"
REQUESTED_LEVERAGES = (1.0, 1.5, 2.0, 3.0, 5.0)
REFERENCE_EQUITY_USD = (1_000.0, 10_000.0, 50_000.0)
MARGIN_MODES = ("cross", "isolated")
STRESS_SCENARIOS: tuple[dict[str, object], ...] = (
    {"stress_scenario": "baseline"},
    {"stress_scenario": "downside_volatility_150", "downside_factor": 1.5},
    {"stress_scenario": "basis_gap_2pct_each_fold", "event_loss": 0.02},
    {"stress_scenario": "correlation_break_3pct_each_fold", "event_loss": 0.03},
    {"stress_scenario": "hedge_error_10pct", "hedge_error_factor": 0.10},
    {"stress_scenario": "adverse_funding_200", "funding_factor": 2.0},
    {"stress_scenario": "illiquid_slippage_200", "slippage_factor": 2.0},
    {"stress_scenario": "orphan_leg_5pct_each_fold", "orphan_leg_loss": 0.05},
)
MIN_MAINTENANCE_BUFFER_RATIO = 0.50
MIN_LIQUIDATION_DISTANCE_PCT = 0.15
MAX_LEVERAGED_DRAWDOWN = 0.60
MIN_STRESS_TOTAL_RETURN = -0.50
MAX_MARKET_EVIDENCE_AGE_HOURS = 1.0
RESEARCH_ONLY_REASON = (
    "leverage_surface_is_research_only;testnet_lifecycle_not_proven;"
    "human_live_release_not_authorized"
)
SCENARIO_COLUMNS = [
    "schema_version",
    "leverage_surface_id",
    "experiment_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "exact_mode",
    "orientation",
    "asset_x",
    "asset_y",
    "requested_gross_leverage",
    "effective_gross_leverage",
    "required_exchange_leverage_setting",
    "reference_equity_usd",
    "margin_mode",
    "stress_scenario",
    "hedge_ratio_abs",
    "leg_x_weight",
    "leg_y_weight",
    "leg_x_notional_usd",
    "leg_y_notional_usd",
    "leg_x_margin_table_id",
    "leg_y_margin_table_id",
    "leg_x_tier_max_leverage",
    "leg_y_tier_max_leverage",
    "leg_x_maintenance_margin_usd",
    "leg_y_maintenance_margin_usd",
    "total_initial_margin_usd",
    "total_maintenance_margin_usd",
    "initial_margin_utilization",
    "maintenance_buffer_ratio",
    "minimum_liquidation_distance_pct",
    "aggregate_total_return",
    "maximum_drawdown",
    "minimum_bar_return",
    "ruin_triggered",
    "folds_complete",
    "scenario_status",
    "scenario_blocker",
    "acceptance_status",
    "acceptance_reason",
    "live_trading_authorized",
]
CANDIDATE_COLUMNS = [
    "schema_version",
    "leverage_surface_id",
    "experiment_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "exact_mode",
    "orientation",
    "asset_x",
    "asset_y",
    "prior_concentration_status",
    "scenario_rows_expected",
    "scenario_rows_complete",
    "scenario_rows_passed",
    "baseline_1x_pass",
    "maximum_research_leverage",
    "leverage_surface_status",
    "leverage_surface_blocker",
    "ready_for_testnet_1x_lifecycle",
    "acceptance_status",
    "acceptance_reason",
    "acceptance_eligible",
    "evidence_path",
    "live_trading_authorized",
]


def build_exhaustive_wizard_hyperliquid_leverage_surface(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Build separate leverage/margin scenarios without changing the 1x signal."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    concentration_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_concentration_manifest.json"
    )
    walkforward_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_walkforward_manifest.json"
    )
    if not concentration_manifest_path.exists() or not walkforward_manifest_path.exists():
        raise FileNotFoundError("Concentration and walk-forward manifests are required")
    concentration_manifest = _read_json(concentration_manifest_path)
    walkforward_manifest = _read_json(walkforward_manifest_path)
    if _text(concentration_manifest.get("walkforward_id")) != _text(
        walkforward_manifest.get("walkforward_id")
    ):
        raise ValueError("Concentration and walk-forward identities do not match")

    concentration_artifacts = concentration_manifest.get("artifacts", {})
    walkforward_artifacts = walkforward_manifest.get("artifacts", {})
    input_paths = {
        "concentration_status": root
        / _text(concentration_artifacts.get("snapshot_status")),
        "concentration_manifest": concentration_manifest_path,
        "walkforward_candidates": root
        / _text(walkforward_artifacts.get("snapshot_candidates")),
        "walkforward_bars": root / _text(walkforward_artifacts.get("snapshot_bars")),
        "walkforward_folds": root / _text(walkforward_artifacts.get("snapshot_folds")),
        "walkforward_manifest": walkforward_manifest_path,
        "market_inventory": active / "hyperliquid_testnet_market_inventory.csv",
        "margin_tiers": active / "hyperliquid_testnet_margin_tiers.csv",
        "pair_cost_evidence": active
        / "exhaustive_wizard_hyperliquid_pair_cost_evidence.csv",
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Leverage surface inputs missing: {missing}")

    statuses = _read_csv(input_paths["concentration_status"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    bars = _read_csv(input_paths["walkforward_bars"])
    folds = _read_csv(input_paths["walkforward_folds"])
    inventory = _read_csv(input_paths["market_inventory"])
    tiers = _read_csv(input_paths["margin_tiers"])
    costs = _read_csv(input_paths["pair_cost_evidence"])
    _require_unique(statuses, "experiment_id")
    _require_unique(candidates, "experiment_id", allow_empty=True)
    _require_unique(costs, "pair_group_id", allow_empty=True)
    if statuses.empty:
        raise ValueError("Leverage surface requires a non-empty experiment status ledger")

    selected_ids = set(
        statuses.loc[
            statuses.get("ready_for_leverage_gate", pd.Series(False, index=statuses.index)).map(
                _truthy
            ),
            "experiment_id",
        ].astype(str)
    )
    candidate_lookup = _row_lookup(candidates, "experiment_id")
    cost_lookup = _row_lookup(costs, "pair_group_id")
    inventory_lookup = _inventory_lookup(inventory)
    inventory_age = _evidence_age_hours(inventory.get("checked_at_utc"), as_of)
    tiers_age = _evidence_age_hours(tiers.get("checked_at_utc"), as_of)
    market_evidence_fresh = bool(
        inventory_age is not None
        and tiers_age is not None
        and inventory_age <= MAX_MARKET_EVIDENCE_AGE_HOURS
        and tiers_age <= MAX_MARKET_EVIDENCE_AGE_HOURS
    )

    policy = {
        "candidate_policy": "concentration_ready_only",
        "requested_gross_leverages": list(REQUESTED_LEVERAGES),
        "reference_equity_usd": list(REFERENCE_EQUITY_USD),
        "margin_modes": list(MARGIN_MODES),
        "stress_scenarios": list(STRESS_SCENARIOS),
        "minimum_maintenance_buffer_ratio": MIN_MAINTENANCE_BUFFER_RATIO,
        "minimum_liquidation_distance_pct": MIN_LIQUIDATION_DISTANCE_PCT,
        "maximum_leveraged_drawdown": MAX_LEVERAGED_DRAWDOWN,
        "minimum_stress_total_return": MIN_STRESS_TOTAL_RETURN,
        "maximum_market_evidence_age_hours": MAX_MARKET_EVIDENCE_AGE_HOURS,
        "one_point_five_x_policy": "effective_gross_sizing_with_integer_exchange_setting_ceiling",
        "canonical_1x_signal_unchanged": True,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": _text(concentration_manifest.get("run_id")),
        "walkforward_id": _text(concentration_manifest.get("walkforward_id")),
        "concentration_id": _text(concentration_manifest.get("concentration_id")),
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    digest = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    leverage_surface_id = f"hlleverage_{as_of.strftime('%Y%m%dT%H%M%S%fZ')}_{digest[:8]}"
    concentration_snapshot_manifest = root / _text(
        concentration_artifacts.get("snapshot_manifest")
    )
    snapshot_dir = concentration_snapshot_manifest.parent / "leverage" / leverage_surface_id
    snapshot_inputs_dir = snapshot_dir / "inputs"
    snapshot_inputs_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in input_paths.items():
        target = snapshot_inputs_dir / f"{name}{source.suffix or '.dat'}"
        shutil.copy2(source, target)
        snapshot_inputs[name] = target

    scenario_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    selected_blockers: dict[str, str] = {}
    for experiment_id in sorted(selected_ids):
        candidate = candidate_lookup.get(experiment_id)
        if candidate is None:
            selected_blockers[experiment_id] = "walkforward_candidate_missing"
            continue
        pair_group_id = _text(candidate.get("pair_group_id"))
        cost = cost_lookup.get(pair_group_id)
        if not cost or not _truthy(cost.get("cost_acceptance_ready")):
            selected_blockers[experiment_id] = "cost_calibration_acceptance_not_ready"
            continue
        candidate_bars = bars.loc[bars["experiment_id"].astype(str).eq(experiment_id)].copy()
        candidate_folds = folds.loc[folds["experiment_id"].astype(str).eq(experiment_id)].copy()
        asset_x = _text(candidate.get("asset_x")).upper()
        asset_y = _text(candidate.get("asset_y")).upper()
        market_x = inventory_lookup.get(asset_x)
        market_y = inventory_lookup.get(asset_y)
        hedge_ratio = pd.to_numeric(
            candidate_folds.get("fitted_hedge_ratio"), errors="coerce"
        ).abs().median()
        blockers: list[str] = []
        if not market_evidence_fresh:
            blockers.append("hyperliquid_market_or_margin_tier_evidence_stale")
        if candidate_bars.empty:
            blockers.append("walkforward_bar_ledger_missing")
        if market_x is None or market_y is None:
            blockers.append("hyperliquid_testnet_market_inventory_pair_missing")
        if not math.isfinite(float(hedge_ratio)) or float(hedge_ratio) <= 0.0:
            blockers.append("fitted_hedge_ratio_missing_or_invalid")
        if blockers:
            selected_blockers[experiment_id] = ";".join(blockers)
            continue

        identity = _identity(candidate)
        for requested in REQUESTED_LEVERAGES:
            for equity in REFERENCE_EQUITY_USD:
                for margin_mode in MARGIN_MODES:
                    margin = _margin_metrics(
                        requested_leverage=requested,
                        equity_usd=equity,
                        hedge_ratio_abs=float(hedge_ratio),
                        margin_mode=margin_mode,
                        market_x=market_x,
                        market_y=market_y,
                        tiers=tiers,
                    )
                    for stress in STRESS_SCENARIOS:
                        path = _simulate_leverage_path(
                            candidate_bars,
                            effective_leverage=float(margin["effective_gross_leverage"]),
                            stress=stress,
                            maximum_leg_weight=max(
                                float(margin["leg_x_weight"]),
                                float(margin["leg_y_weight"]),
                            ),
                        )
                        scenario_blockers = _scenario_blockers(
                            requested_leverage=requested,
                            margin=margin,
                            path=path,
                        )
                        margin_mode_not_applicable = (
                            margin_mode == "cross"
                            and "pair_contains_isolated_only_market"
                            in _split_blockers(margin.get("margin_blocker"))
                        )
                        scenario_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "leverage_surface_id": leverage_surface_id,
                                **identity,
                                "requested_gross_leverage": requested,
                                **margin,
                                "reference_equity_usd": equity,
                                "margin_mode": margin_mode,
                                "stress_scenario": _text(stress["stress_scenario"]),
                                **path,
                                "scenario_status": (
                                    "NOT_APPLICABLE_MARGIN_MODE"
                                    if margin_mode_not_applicable
                                    else (
                                        "PASS_RESEARCH_LEVERAGE_SCENARIO"
                                        if not scenario_blockers
                                        else "FAIL_RESEARCH_LEVERAGE_SCENARIO"
                                    )
                                ),
                                "scenario_blocker": ";".join(scenario_blockers),
                                "acceptance_status": "BLOCKED",
                                "acceptance_reason": RESEARCH_ONLY_REASON,
                                "live_trading_authorized": False,
                            }
                        )

        experiment_scenarios = [
            row for row in scenario_rows if row["experiment_id"] == experiment_id
        ]
        expected = (
            len(REQUESTED_LEVERAGES)
            * len(REFERENCE_EQUITY_USD)
            * len(MARGIN_MODES)
            * len(STRESS_SCENARIOS)
        )
        passing_leverages = []
        for requested in REQUESTED_LEVERAGES:
            subset = [
                row
                for row in experiment_scenarios
                if float(row["requested_gross_leverage"]) == requested
                and row["scenario_status"] != "NOT_APPLICABLE_MARGIN_MODE"
            ]
            if subset and all(
                row["scenario_status"] == "PASS_RESEARCH_LEVERAGE_SCENARIO"
                for row in subset
            ):
                passing_leverages.append(requested)
        baseline_pass = 1.0 in passing_leverages
        candidate_blockers = []
        if len(experiment_scenarios) != expected:
            candidate_blockers.append("leverage_scenario_accounting_incomplete")
        if not baseline_pass:
            candidate_blockers.append("canonical_1x_leverage_surface_failed")
        candidate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leverage_surface_id": leverage_surface_id,
                **identity,
                "prior_concentration_status": _text(
                    statuses.loc[
                        statuses["experiment_id"].astype(str).eq(experiment_id),
                        "concentration_status",
                    ].iloc[0]
                ),
                "scenario_rows_expected": expected,
                "scenario_rows_complete": len(experiment_scenarios),
                "scenario_rows_passed": sum(
                    row["scenario_status"] == "PASS_RESEARCH_LEVERAGE_SCENARIO"
                    for row in experiment_scenarios
                ),
                "baseline_1x_pass": baseline_pass,
                "maximum_research_leverage": max(passing_leverages, default=0.0),
                "leverage_surface_status": (
                    "PASS_RESEARCH_LEVERAGE_SURFACE"
                    if not candidate_blockers
                    else "FAIL_RESEARCH_LEVERAGE_SURFACE"
                ),
                "leverage_surface_blocker": ";".join(candidate_blockers),
                "ready_for_testnet_1x_lifecycle": bool(not candidate_blockers),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "acceptance_eligible": False,
                "evidence_path": ";".join(
                    _relative(path, root) for path in snapshot_inputs.values()
                ),
                "live_trading_authorized": False,
            }
        )

    scenario_frame = pd.DataFrame(scenario_rows, columns=SCENARIO_COLUMNS)
    candidate_frame = pd.DataFrame(candidate_rows, columns=CANDIDATE_COLUMNS)
    candidate_result_lookup = _row_lookup(candidate_frame, "experiment_id")
    status_frame = _status_frame(
        statuses,
        selected_ids=selected_ids,
        selected_blockers=selected_blockers,
        candidate_results=candidate_result_lookup,
        leverage_surface_id=leverage_surface_id,
        evidence_paths=snapshot_inputs.values(),
        root=root,
    )
    if len(status_frame) != len(statuses) or status_frame["experiment_id"].nunique() != len(
        statuses
    ):
        raise ValueError("Leverage surface failed complete experiment accounting")
    leverage_status_defaults = {
        "ready_for_testnet_1x_lifecycle": False,
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "prior_concentration_gate_not_passed",
        "acceptance_eligible": False,
    }
    for column, default in leverage_status_defaults.items():
        status_frame[column] = status_frame.get(
            column,
            pd.Series(index=status_frame.index, dtype=object),
        ).fillna(default)

    paths = {
        "status": active / "exhaustive_wizard_hyperliquid_leverage_status.csv",
        "candidates": active / "exhaustive_wizard_hyperliquid_leverage_candidates.csv",
        "scenarios": active / "exhaustive_wizard_hyperliquid_leverage_scenarios.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_leverage_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_leverage_summary.md",
        "snapshot_status": snapshot_dir / "leverage_status.csv",
        "snapshot_candidates": snapshot_dir / "leverage_candidates.csv",
        "snapshot_scenarios": snapshot_dir / "leverage_scenarios.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for frame, active_path, snapshot_path in (
        (status_frame, paths["status"], paths["snapshot_status"]),
        (candidate_frame, paths["candidates"], paths["snapshot_candidates"]),
        (scenario_frame, paths["scenarios"], paths["snapshot_scenarios"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    status_counts = _status_counts(status_frame, "leverage_surface_status")
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": material["run_id"],
        "walkforward_id": material["walkforward_id"],
        "concentration_id": material["concentration_id"],
        "leverage_surface_id": leverage_surface_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(status_frame)),
        "unique_experiment_ids": int(status_frame["experiment_id"].nunique()),
        "status_counts": status_counts,
        "experiment_status_accounted": bool(sum(status_counts.values()) == len(status_frame)),
        "concentration_ready_candidates": len(selected_ids),
        "leverage_candidates_complete": int(len(candidate_frame)),
        "scenario_rows": int(len(scenario_frame)),
        "research_leverage_surface_passes": int(
            candidate_frame.get("leverage_surface_status", pd.Series(dtype=str))
            .eq("PASS_RESEARCH_LEVERAGE_SURFACE")
            .sum()
        ),
        "testnet_1x_lifecycle_ready": int(
            candidate_frame.get("ready_for_testnet_1x_lifecycle", pd.Series(dtype=bool))
            .map(_truthy)
            .sum()
        ),
        "market_inventory_age_hours": inventory_age,
        "margin_tier_age_hours": tiers_age,
        "market_evidence_fresh": market_evidence_fresh,
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
    summary_text = _summary_markdown(summary, candidate_frame)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _margin_metrics(
    *,
    requested_leverage: float,
    equity_usd: float,
    hedge_ratio_abs: float,
    margin_mode: str,
    market_x: dict[str, object],
    market_y: dict[str, object],
    tiers: pd.DataFrame,
) -> dict[str, object]:
    weight_x = hedge_ratio_abs / (1.0 + hedge_ratio_abs)
    weight_y = 1.0 / (1.0 + hedge_ratio_abs)
    effective = requested_leverage
    tier_x: dict[str, object] | None = None
    tier_y: dict[str, object] | None = None
    blockers: list[str] = []
    if margin_mode == "cross" and (
        _truthy(market_x.get("only_isolated")) or _truthy(market_y.get("only_isolated"))
    ):
        blockers.append("pair_contains_isolated_only_market")
    for _ in range(4):
        notional_x = equity_usd * effective * weight_x
        notional_y = equity_usd * effective * weight_y
        tier_x = _tier_for_notional(tiers, market_x.get("margin_table_id"), notional_x)
        tier_y = _tier_for_notional(tiers, market_y.get("margin_table_id"), notional_y)
        if tier_x is None or tier_y is None:
            break
        cap = min(
            _number(market_x.get("max_leverage")) or 0.0,
            _number(market_y.get("max_leverage")) or 0.0,
            _number(tier_x.get("max_leverage")) or 0.0,
            _number(tier_y.get("max_leverage")) or 0.0,
        )
        revised = min(requested_leverage, cap)
        if abs(revised - effective) < 1e-12:
            break
        effective = revised
    if tier_x is None or tier_y is None:
        blockers.append("margin_tier_not_resolved_for_pair_notional")
    if effective <= 0.0:
        blockers.append("effective_leverage_not_positive")
    notional_x = equity_usd * max(effective, 0.0) * weight_x
    notional_y = equity_usd * max(effective, 0.0) * weight_y
    exchange_setting = max(1, int(math.ceil(max(effective, 0.0))))
    maintenance_x = _maintenance_margin(notional_x, tier_x)
    maintenance_y = _maintenance_margin(notional_y, tier_y)
    initial_x = notional_x / exchange_setting if exchange_setting > 0 else math.inf
    initial_y = notional_y / exchange_setting if exchange_setting > 0 else math.inf
    total_initial = initial_x + initial_y
    total_maintenance = maintenance_x + maintenance_y
    maintenance_buffer = (
        (equity_usd - total_maintenance) / equity_usd if equity_usd > 0.0 else -math.inf
    )
    if margin_mode == "cross":
        available = max(equity_usd - total_maintenance, 0.0)
        distances = [
            _liquidation_distance(available, notional_x, maintenance_x),
            _liquidation_distance(available, notional_y, maintenance_y),
        ]
    else:
        distances = [
            _liquidation_distance(max(initial_x - maintenance_x, 0.0), notional_x, maintenance_x),
            _liquidation_distance(max(initial_y - maintenance_y, 0.0), notional_y, maintenance_y),
        ]
    return {
        "effective_gross_leverage": effective,
        "required_exchange_leverage_setting": exchange_setting,
        "hedge_ratio_abs": hedge_ratio_abs,
        "leg_x_weight": weight_x,
        "leg_y_weight": weight_y,
        "leg_x_notional_usd": notional_x,
        "leg_y_notional_usd": notional_y,
        "leg_x_margin_table_id": _integer(market_x.get("margin_table_id")),
        "leg_y_margin_table_id": _integer(market_y.get("margin_table_id")),
        "leg_x_tier_max_leverage": _number((tier_x or {}).get("max_leverage")),
        "leg_y_tier_max_leverage": _number((tier_y or {}).get("max_leverage")),
        "leg_x_maintenance_margin_usd": maintenance_x,
        "leg_y_maintenance_margin_usd": maintenance_y,
        "total_initial_margin_usd": total_initial,
        "total_maintenance_margin_usd": total_maintenance,
        "initial_margin_utilization": total_initial / equity_usd if equity_usd else math.inf,
        "maintenance_buffer_ratio": maintenance_buffer,
        "minimum_liquidation_distance_pct": min(distances),
        "margin_blocker": ";".join(_deduplicate(blockers)),
    }


def _simulate_leverage_path(
    bars: pd.DataFrame,
    *,
    effective_leverage: float,
    stress: dict[str, object],
    maximum_leg_weight: float,
) -> dict[str, object]:
    if bars.empty:
        return {
            "aggregate_total_return": -1.0,
            "maximum_drawdown": 1.0,
            "minimum_bar_return": -1.0,
            "ruin_triggered": True,
            "folds_complete": 0,
        }
    frame = bars.copy()
    for column in (
        "net_return",
        "gross_return",
        "funding",
        "slippage",
        "target_position",
    ):
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce").fillna(0.0)
    frame["fold_number"] = pd.to_numeric(
        frame.get("fold_number"), errors="coerce"
    ).fillna(0).astype(int)
    fold_returns: list[float] = []
    max_drawdown = 0.0
    minimum_bar_return = math.inf
    ruin = False
    for _, fold in frame.groupby("fold_number", sort=True):
        adjusted = fold["net_return"].copy()
        downside_factor = _number(stress.get("downside_factor"))
        if downside_factor is not None:
            adjusted.loc[adjusted < 0.0] *= downside_factor
        hedge_error = _number(stress.get("hedge_error_factor"))
        if hedge_error is not None:
            adjusted -= hedge_error * fold["gross_return"].abs()
        funding_factor = _number(stress.get("funding_factor"))
        if funding_factor is not None and funding_factor > 1.0:
            adjusted -= (funding_factor - 1.0) * fold["funding"].abs()
        slippage_factor = _number(stress.get("slippage_factor"))
        if slippage_factor is not None and slippage_factor > 1.0:
            adjusted -= (slippage_factor - 1.0) * fold["slippage"].abs()
        event_loss = _number(stress.get("event_loss")) or 0.0
        orphan_loss = (_number(stress.get("orphan_leg_loss")) or 0.0) * maximum_leg_weight
        shock = event_loss + orphan_loss
        active = fold.index[fold["target_position"].abs() > 0.0]
        if shock > 0.0 and len(active):
            adjusted.loc[active[0]] -= shock
        leveraged = adjusted * effective_leverage
        minimum_bar_return = min(minimum_bar_return, float(leveraged.min()))
        if (leveraged <= -1.0).any():
            ruin = True
        equity = (1.0 + leveraged).cumprod()
        if (equity <= 0.0).any() or not np.isfinite(equity).all():
            ruin = True
        equity = equity.clip(lower=0.0)
        peak = equity.cummax().replace(0.0, np.nan)
        drawdown = (1.0 - equity / peak).fillna(1.0)
        max_drawdown = max(max_drawdown, float(drawdown.max()))
        fold_returns.append(float(equity.iloc[-1] - 1.0))
    aggregate = float(np.prod([1.0 + value for value in fold_returns]) - 1.0)
    return {
        "aggregate_total_return": aggregate,
        "maximum_drawdown": max_drawdown,
        "minimum_bar_return": minimum_bar_return if math.isfinite(minimum_bar_return) else 0.0,
        "ruin_triggered": ruin,
        "folds_complete": len(fold_returns),
    }


def _scenario_blockers(
    *,
    requested_leverage: float,
    margin: dict[str, object],
    path: dict[str, object],
) -> list[str]:
    blockers = _split_blockers(margin.get("margin_blocker"))
    effective = _number(margin.get("effective_gross_leverage")) or 0.0
    if effective + 1e-12 < requested_leverage:
        blockers.append("requested_leverage_exceeds_pair_tier_limit")
    maintenance_buffer = _number(margin.get("maintenance_buffer_ratio"))
    liquidation_distance = _number(margin.get("minimum_liquidation_distance_pct"))
    maximum_drawdown = _number(path.get("maximum_drawdown"))
    total_return = _number(path.get("aggregate_total_return"))
    if maintenance_buffer is None or maintenance_buffer < MIN_MAINTENANCE_BUFFER_RATIO:
        blockers.append("maintenance_buffer_below_policy")
    if liquidation_distance is None or liquidation_distance < MIN_LIQUIDATION_DISTANCE_PCT:
        blockers.append("liquidation_distance_below_policy")
    if _truthy(path.get("ruin_triggered")):
        blockers.append("leveraged_path_ruin_triggered")
    if maximum_drawdown is None or maximum_drawdown > MAX_LEVERAGED_DRAWDOWN:
        blockers.append("leveraged_drawdown_above_policy")
    if total_return is None or total_return < MIN_STRESS_TOTAL_RETURN:
        blockers.append("stress_total_return_below_policy")
    return _deduplicate(blockers)


def _tier_for_notional(
    tiers: pd.DataFrame,
    table_id: object,
    notional_usd: float,
) -> dict[str, object] | None:
    if tiers.empty:
        return None
    target = _integer(table_id)
    frame = tiers.loc[
        pd.to_numeric(tiers.get("margin_table_id"), errors="coerce").fillna(-1).astype(int).eq(target)
    ].copy()
    if "blocker" in frame.columns:
        frame = frame.loc[frame["blocker"].fillna("").astype(str).eq("")]
    if frame.empty:
        return None
    frame["_lower"] = pd.to_numeric(frame.get("lower_bound_usd"), errors="coerce")
    frame["_upper"] = pd.to_numeric(frame.get("upper_bound_usd"), errors="coerce")
    matches = frame.loc[
        frame["_lower"].le(notional_usd)
        & (frame["_upper"].isna() | frame["_upper"].gt(notional_usd))
    ].sort_values("_lower", ascending=False)
    return matches.iloc[0].to_dict() if not matches.empty else None


def _maintenance_margin(notional: float, tier: dict[str, object] | None) -> float:
    if tier is None:
        return math.inf
    rate = _number(tier.get("maintenance_margin_rate"))
    deduction = _number(tier.get("maintenance_deduction_usd"))
    if rate is None or deduction is None:
        return math.inf
    return max(notional * rate - deduction, 0.0)


def _liquidation_distance(
    margin_available: float,
    notional: float,
    maintenance_margin: float,
) -> float:
    if notional <= 0.0 or not math.isfinite(maintenance_margin):
        return 0.0
    maintenance_rate = maintenance_margin / notional
    return max(margin_available / notional / (1.0 + maintenance_rate), 0.0)


def _status_frame(
    statuses: pd.DataFrame,
    *,
    selected_ids: set[str],
    selected_blockers: dict[str, str],
    candidate_results: dict[str, dict[str, object]],
    leverage_surface_id: str,
    evidence_paths: object,
    root: Path,
) -> pd.DataFrame:
    evidence = ";".join(_relative(Path(path), root) for path in evidence_paths)
    rows: list[dict[str, object]] = []
    for record in statuses.to_dict("records"):
        experiment_id = _text(record.get("experiment_id"))
        result = candidate_results.get(experiment_id)
        if result:
            status = _text(result.get("leverage_surface_status"))
            blocker = _text(result.get("leverage_surface_blocker"))
            ready = _truthy(result.get("ready_for_testnet_1x_lifecycle"))
        elif experiment_id in selected_ids:
            status = "BLOCKED_LEVERAGE_INPUTS"
            blocker = selected_blockers.get(experiment_id, "leverage_candidate_not_materialized")
            ready = False
        else:
            status = "NOT_SELECTED_PRIOR_CONCENTRATION_GATE"
            blocker = (
                _text(record.get("concentration_blocker"))
                or _text(record.get("concentration_status"))
            )
            ready = False
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leverage_surface_id": leverage_surface_id,
                "experiment_id": experiment_id,
                "pair_group_id": _text(record.get("pair_group_id")),
                "pair": _text(record.get("pair")),
                "wizard_exchange": _text(record.get("wizard_exchange")),
                "wizard_timeframe": _text(record.get("wizard_timeframe")),
                "hyperliquid_interval": _text(record.get("hyperliquid_interval")),
                "exact_mode": _text(record.get("exact_mode")),
                "orientation": _text(record.get("orientation")),
                "prior_concentration_status": _text(record.get("concentration_status")),
                "leverage_surface_status": status,
                "leverage_surface_blocker": blocker,
                "ready_for_testnet_1x_lifecycle": ready,
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "acceptance_eligible": False,
                "evidence_path": evidence,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _identity(candidate: dict[str, object]) -> dict[str, str]:
    return {
        key: _text(candidate.get(key))
        for key in (
            "experiment_id",
            "pair_group_id",
            "pair",
            "wizard_exchange",
            "wizard_timeframe",
            "hyperliquid_interval",
            "exact_mode",
            "orientation",
            "asset_x",
            "asset_y",
        )
    }


def _inventory_lookup(frame: pd.DataFrame) -> dict[str, dict[str, object]]:
    if frame.empty:
        return {}
    tradable = frame.get("tradable_perp", pd.Series(False, index=frame.index)).map(_truthy)
    return {
        _text(row.get("asset")).upper(): row
        for row in frame.loc[tradable].to_dict("records")
        if _text(row.get("asset"))
    }


def _evidence_age_hours(values: pd.Series | None, as_of: datetime) -> float | None:
    if values is None or values.empty:
        return None
    timestamps = pd.to_datetime(values, utc=True, errors="coerce").dropna()
    if timestamps.empty:
        return None
    return max((pd.Timestamp(as_of) - timestamps.max()).total_seconds() / 3600.0, 0.0)


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, dict[str, object]]:
    if frame.empty or key not in frame.columns:
        return {}
    return {_text(row.get(key)): row for row in frame.to_dict("records")}


def _summary_markdown(summary: dict[str, object], candidates: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Leverage Surface",
            "",
            "Canonical 1x signals remain unchanged. Leverage alters sizing, margin, liquidation distance, and the loss path; it never creates statistical edge.",
            "The surface uses current Hyperliquid Testnet margin tiers and writes separate rows for every requested leverage, reference equity, permitted margin mode, and stress scenario.",
            "No row authorizes Testnet submission or live trading.",
            "",
            "## Summary",
            "",
            pd.DataFrame(
                [
                    {"metric": "leverage_surface_id", "value": summary["leverage_surface_id"]},
                    {"metric": "experiments", "value": summary["experiments"]},
                    {
                        "metric": "concentration_ready_candidates",
                        "value": summary["concentration_ready_candidates"],
                    },
                    {"metric": "scenario_rows", "value": summary["scenario_rows"]},
                    {
                        "metric": "research_leverage_surface_passes",
                        "value": summary["research_leverage_surface_passes"],
                    },
                    {
                        "metric": "testnet_1x_lifecycle_ready",
                        "value": summary["testnet_1x_lifecycle_ready"],
                    },
                    {"metric": "live_trading_authorized", "value": False},
                ]
            ).to_markdown(index=False),
            "",
            "## Candidate Surfaces",
            "",
            (
                candidates.to_markdown(index=False)
                if not candidates.empty
                else "No experiment passed the prerequisite concentration gate."
            ),
            "",
        ]
    )


def _require_unique(frame: pd.DataFrame, column: str, *, allow_empty: bool = False) -> None:
    if frame.empty and allow_empty:
        return
    if frame.empty or column not in frame.columns or frame[column].astype(str).duplicated().any():
        raise ValueError(f"Leverage surface input requires unique {column}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _status_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    return {
        _text(key): int(value)
        for key, value in frame[column].value_counts(dropna=False).sort_index().items()
    }


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _split_blockers(value: object) -> list[str]:
    return [part.strip() for part in _text(value).split(";") if part.strip()]


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _integer(value: object) -> int:
    number = _number(value)
    return int(number) if number is not None else 0


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
