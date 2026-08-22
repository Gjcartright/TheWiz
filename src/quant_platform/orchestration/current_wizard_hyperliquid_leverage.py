"""Evidence-gated leverage and margin surface for the current Wizard board."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_leverage import (
    MARGIN_MODES,
    REFERENCE_EQUITY_USD,
    REQUESTED_LEVERAGES,
    STRESS_SCENARIOS,
    _evidence_age_hours,
    _inventory_lookup,
    _margin_metrics,
    _scenario_blockers,
    _simulate_leverage_path,
)
from quant_platform.orchestration.snapshot_lineage import (
    existing_snapshot_reference,
    unique_file_bytes,
    verified_snapshot_reference,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_leverage.v1"
MAX_MARKET_EVIDENCE_AGE_HOURS = 1.0
RESEARCH_ONLY_REASON = (
    "current_leverage_surface_is_research_only;canonical_1x_remains_signal_authority;"
    "testnet_order_authority_false;live_trading_not_authorized"
)


def build_current_wizard_hyperliquid_leverage_surface(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Build leverage scenarios only for current 1x research survivors."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    failure_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    )
    walk_manifest_path = active / "current_wizard_hyperliquid_walkforward_manifest.json"
    if not failure_manifest_path.exists() or not walk_manifest_path.exists():
        raise FileNotFoundError(
            "Current failure attribution and walk-forward manifests are required"
        )
    failure_manifest = _read_json(failure_manifest_path)
    walk_manifest = _read_json(walk_manifest_path)
    failure_artifacts = failure_manifest.get("artifacts", {})
    walk_artifacts = walk_manifest.get("artifacts", {})
    input_paths = {
        "failure_attribution": root
        / _text(failure_artifacts.get("snapshot_attribution")),
        "failure_manifest": failure_manifest_path,
        "walkforward_candidates": root
        / _text(walk_artifacts.get("snapshot_candidates")),
        "walkforward_bars": root / _text(walk_artifacts.get("snapshot_bars")),
        "walkforward_folds": root / _text(walk_artifacts.get("snapshot_folds")),
        "walkforward_manifest": walk_manifest_path,
    }
    missing = [str(path) for path in input_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current leverage inputs missing: {missing}")
    statuses = _read_csv(input_paths["failure_attribution"])
    candidates = _read_csv(input_paths["walkforward_candidates"])
    bars = _read_csv(input_paths["walkforward_bars"])
    folds = _read_csv(input_paths["walkforward_folds"])
    _require_unique(statuses, "experiment_id")
    _require_unique(candidates, "experiment_id", allow_empty=True)
    selected_ids = set(
        statuses.loc[
            statuses["leverage_research_eligible"].map(_truthy), "experiment_id"
        ].astype(str)
    )

    inventory_path = active / "hyperliquid_testnet_market_inventory.csv"
    tiers_path = active / "hyperliquid_testnet_margin_tiers.csv"
    costs_path = active / "current_wizard_hyperliquid_pair_cost_evidence.csv"
    if selected_ids:
        selected_required = (inventory_path, tiers_path, costs_path)
        missing_selected = [str(path) for path in selected_required if not path.exists()]
        if missing_selected:
            raise FileNotFoundError(
                "Current selected leverage inputs missing: " + ",".join(missing_selected)
            )
    for name, path in (
        ("market_inventory", inventory_path),
        ("margin_tiers", tiers_path),
        ("pair_cost_evidence", costs_path),
    ):
        if path.exists():
            input_paths[name] = path

    inventory = _read_csv(inventory_path) if inventory_path.exists() else pd.DataFrame()
    tiers = _read_csv(tiers_path) if tiers_path.exists() else pd.DataFrame()
    costs = _read_csv(costs_path) if costs_path.exists() else pd.DataFrame()
    _require_unique(costs, "pair_group_key", allow_empty=True)
    candidate_lookup = _row_lookup(candidates, "experiment_id")
    cost_lookup = _row_lookup(costs, "pair_group_key")
    status_lookup = _row_lookup(statuses, "experiment_id")
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
        "candidate_policy": "one_x_research_survivors_only",
        "requested_gross_leverages": list(REQUESTED_LEVERAGES),
        "reference_equity_usd": list(REFERENCE_EQUITY_USD),
        "margin_modes": list(MARGIN_MODES),
        "stress_scenarios": list(STRESS_SCENARIOS),
        "maximum_market_evidence_age_hours": MAX_MARKET_EVIDENCE_AGE_HOURS,
        "canonical_1x_signal_unchanged": True,
        "leverage_changes_capital_and_loss_geometry_not_signal_edge": True,
        "testnet_requires_separate_execution_acceptance": True,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "failure_attribution_id": _text(
            failure_manifest.get("failure_attribution_id")
        ),
        "walkforward_id": _text(walk_manifest.get("walkforward_id")),
        "as_of": as_of.isoformat(),
        "policy": policy,
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
    }
    leverage_surface_id = "cwleverage_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    failure_snapshot = root / _text(failure_artifacts.get("snapshot_manifest"))
    if not failure_snapshot.exists():
        raise FileNotFoundError("Current failure-attribution snapshot is missing")
    snapshot_dir = failure_snapshot.parent / "leverage" / leverage_surface_id
    input_dir = snapshot_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {
        "failure_attribution": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["failure_attribution"]
        ),
        "failure_manifest": verified_snapshot_reference(
            root=root,
            active_path=failure_manifest_path,
            upstream_manifest=failure_manifest,
            artifact_key="manifest",
        ),
        "walkforward_candidates": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["walkforward_candidates"]
        ),
        "walkforward_bars": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["walkforward_bars"]
        ),
        "walkforward_folds": existing_snapshot_reference(
            root=root, snapshot_path=input_paths["walkforward_folds"]
        ),
        "walkforward_manifest": verified_snapshot_reference(
            root=root,
            active_path=walk_manifest_path,
            upstream_manifest=walk_manifest,
            artifact_key="manifest",
        ),
    }
    input_snapshot_modes = {
        name: "verified_upstream_reference" for name in snapshot_inputs
    }
    for name in ("market_inventory", "margin_tiers", "pair_cost_evidence"):
        source = input_paths.get(name)
        if source is None:
            continue
        target = immutable_snapshot_copy(source, input_dir, artifact_name=name)
        snapshot_inputs[name] = target
        input_snapshot_modes[name] = "local_external_evidence_copy"

    scenario_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    selected_blockers: dict[str, str] = {}
    for experiment_id in sorted(selected_ids):
        candidate = candidate_lookup.get(experiment_id)
        prior = status_lookup.get(experiment_id, {})
        if candidate is None:
            selected_blockers[experiment_id] = "walkforward_candidate_missing"
            continue
        pair_key = _text(candidate.get("pair_group_key"))
        cost = cost_lookup.get(pair_key)
        asset_x = _text(candidate.get("asset_x")).upper()
        asset_y = _text(candidate.get("asset_y")).upper()
        market_x = inventory_lookup.get(asset_x)
        market_y = inventory_lookup.get(asset_y)
        candidate_bars = bars.loc[
            bars["experiment_id"].astype(str).eq(experiment_id)
        ].copy()
        candidate_folds = folds.loc[
            folds["experiment_id"].astype(str).eq(experiment_id)
        ].copy()
        hedge_ratio = pd.to_numeric(
            candidate_folds.get("fitted_hedge_ratio"), errors="coerce"
        ).abs().median()
        blockers: list[str] = []
        if cost is None or not _truthy(cost.get("provisional_cost_research_ready")):
            blockers.append("provisional_observed_cost_evidence_not_ready")
        if not market_evidence_fresh:
            blockers.append("hyperliquid_market_or_margin_tier_evidence_stale")
        if candidate_bars.empty:
            blockers.append("walkforward_bar_ledger_missing")
        if market_x is None or market_y is None:
            blockers.append("hyperliquid_testnet_market_inventory_pair_missing")
        if not math.isfinite(float(hedge_ratio)) or float(hedge_ratio) <= 0.0:
            blockers.append("fitted_hedge_ratio_missing_or_invalid")
        if blockers:
            selected_blockers[experiment_id] = ";".join(dict.fromkeys(blockers))
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
                            effective_leverage=float(
                                margin["effective_gross_leverage"]
                            ),
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
                        margin_not_applicable = (
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
                                "stress_scenario": _text(
                                    stress["stress_scenario"]
                                ),
                                **path,
                                "scenario_status": (
                                    "NOT_APPLICABLE_MARGIN_MODE"
                                    if margin_not_applicable
                                    else (
                                        "PASS_RESEARCH_LEVERAGE_SCENARIO"
                                        if not scenario_blockers
                                        else "FAIL_RESEARCH_LEVERAGE_SCENARIO"
                                    )
                                ),
                                "scenario_blocker": ";".join(scenario_blockers),
                                "acceptance_status": "BLOCKED",
                                "acceptance_reason": RESEARCH_ONLY_REASON,
                                "testnet_order_authority": False,
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
        candidate_blockers: list[str] = []
        if len(experiment_scenarios) != expected:
            candidate_blockers.append("leverage_scenario_accounting_incomplete")
        if not baseline_pass:
            candidate_blockers.append("canonical_1x_leverage_surface_failed")
        testnet_ready = bool(
            not candidate_blockers
            and _truthy(prior.get("testnet_preflight_eligible"))
        )
        if not _truthy(prior.get("testnet_preflight_eligible")):
            candidate_blockers.append("execution_acceptance_not_ready")
        candidate_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leverage_surface_id": leverage_surface_id,
                **identity,
                "prior_one_x_research_status": _text(
                    prior.get("consolidated_status")
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
                    if baseline_pass
                    else "FAIL_RESEARCH_LEVERAGE_SURFACE"
                ),
                "leverage_surface_blocker": ";".join(
                    dict.fromkeys(candidate_blockers)
                ),
                "ready_for_testnet_1x_lifecycle": testnet_ready,
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "acceptance_eligible": False,
                "testnet_order_authority": False,
                "promotion_authority": False,
                "live_trading_authorized": False,
                "evidence_path": ";".join(
                    _relative(path, root) for path in snapshot_inputs.values()
                ),
            }
        )

    scenario_frame = pd.DataFrame(scenario_rows)
    if scenario_frame.empty:
        scenario_frame = pd.DataFrame(
            columns=[
                "schema_version",
                "leverage_surface_id",
                "experiment_id",
                "pair_group_key",
                "pair",
                "exact_mode",
                "orientation",
                "requested_gross_leverage",
                "reference_equity_usd",
                "margin_mode",
                "stress_scenario",
                "scenario_status",
                "scenario_blocker",
                "testnet_order_authority",
                "live_trading_authorized",
            ]
        )
    candidate_frame = pd.DataFrame(candidate_rows)
    if candidate_frame.empty:
        candidate_frame = pd.DataFrame(
            columns=[
                "schema_version",
                "leverage_surface_id",
                "experiment_id",
                "pair_group_key",
                "pair",
                "exact_mode",
                "orientation",
                "scenario_rows_expected",
                "scenario_rows_complete",
                "baseline_1x_pass",
                "maximum_research_leverage",
                "leverage_surface_status",
                "leverage_surface_blocker",
                "ready_for_testnet_1x_lifecycle",
                "testnet_order_authority",
                "live_trading_authorized",
            ]
        )
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
    validation = _validation(
        statuses,
        status_frame,
        candidate_frame,
        scenario_frame,
        selected_ids=selected_ids,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current leverage validation failed: " + ",".join(failed))

    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (status_frame, "status", "snapshot_status"),
        (candidate_frame, "candidates", "snapshot_candidates"),
        (scenario_frame, "scenarios", "snapshot_scenarios"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)
    counts = status_frame["leverage_surface_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "leverage_surface_id": leverage_surface_id,
        "experiments_accounted": int(len(status_frame)),
        "unique_experiment_ids": int(status_frame["experiment_id"].nunique()),
        "status_counts": counts,
        "experiment_status_accounted": bool(sum(counts.values()) == len(status_frame)),
        "one_x_research_survivors_selected": int(len(selected_ids)),
        "leverage_candidates_complete": int(len(candidate_frame)),
        "scenario_rows": int(len(scenario_frame)),
        "expected_scenario_rows": int(
            len(candidate_frame)
            * len(REQUESTED_LEVERAGES)
            * len(REFERENCE_EQUITY_USD)
            * len(MARGIN_MODES)
            * len(STRESS_SCENARIOS)
        ),
        "research_leverage_surface_passes": int(
            candidate_frame.get(
                "leverage_surface_status", pd.Series(dtype=str)
            ).eq("PASS_RESEARCH_LEVERAGE_SURFACE").sum()
        ),
        "testnet_1x_lifecycle_ready": int(
            candidate_frame.get(
                "ready_for_testnet_1x_lifecycle", pd.Series(dtype=bool)
            ).map(_truthy).sum()
        ),
        "market_inventory_age_hours": inventory_age,
        "margin_tier_age_hours": tiers_age,
        "market_evidence_fresh": market_evidence_fresh,
        "acceptance_eligible_replays": 0,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
        "input_snapshot_modes": input_snapshot_modes,
        "referenced_upstream_bytes": unique_file_bytes(
            [
                snapshot_inputs[name]
                for name, mode in input_snapshot_modes.items()
                if mode == "verified_upstream_reference"
            ]
        ),
        "locally_copied_input_bytes": unique_file_bytes(
            [
                snapshot_inputs[name]
                for name, mode in input_snapshot_modes.items()
                if mode == "local_external_evidence_copy"
            ]
        ),
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary, candidate_frame)
    for path in (paths["manifest"], paths["snapshot_manifest"]):
        atomic_write_text(path, manifest_text, encoding="utf-8")
    for path in (paths["summary_md"], paths["snapshot_summary_md"]):
        atomic_write_text(path, summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


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
    del evidence_paths, root
    evidence = "reports/active/current_wizard_hyperliquid_leverage_manifest.json"
    rows: list[dict[str, object]] = []
    for record in statuses.to_dict("records"):
        experiment_id = _text(record.get("experiment_id"))
        candidate = candidate_results.get(experiment_id)
        if candidate is not None:
            status = _text(candidate.get("leverage_surface_status"))
            blocker = _text(candidate.get("leverage_surface_blocker"))
            maximum = _number(candidate.get("maximum_research_leverage")) or 0.0
            ready = _truthy(candidate.get("ready_for_testnet_1x_lifecycle"))
        elif experiment_id in selected_ids:
            status = "BLOCKED_LEVERAGE_INPUTS"
            blocker = selected_blockers.get(
                experiment_id, "leverage_candidate_not_materialized"
            )
            maximum = 0.0
            ready = False
        else:
            status = "NOT_SELECTED_PRIOR_ONE_X_RESEARCH_GATE"
            blocker = _text(record.get("first_blocker")) or _text(
                record.get("consolidated_status")
            )
            maximum = 0.0
            ready = False
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "leverage_surface_id": leverage_surface_id,
                "experiment_id": experiment_id,
                "pair_group_key": _text(record.get("pair_group_key")),
                "pair": _text(record.get("pair")),
                "wizard_exchange": _text(record.get("wizard_exchange")),
                "wizard_timeframe": _text(record.get("wizard_timeframe")),
                "exact_mode": _text(record.get("exact_mode")),
                "orientation": _text(record.get("orientation")),
                "prior_one_x_research_status": _text(
                    record.get("consolidated_status")
                ),
                "leverage_surface_status": status,
                "leverage_surface_blocker": blocker,
                "maximum_research_leverage": maximum,
                "ready_for_testnet_1x_lifecycle": ready,
                "acceptance_status": "BLOCKED",
                "acceptance_reason": RESEARCH_ONLY_REASON,
                "acceptance_eligible": False,
                "testnet_order_authority": False,
                "promotion_authority": False,
                "live_trading_authorized": False,
                "evidence_path": evidence,
            }
        )
    return pd.DataFrame(rows)


def _validation(
    prior: pd.DataFrame,
    status: pd.DataFrame,
    candidates: pd.DataFrame,
    scenarios: pd.DataFrame,
    *,
    selected_ids: set[str],
) -> pd.DataFrame:
    expected_per_candidate = (
        len(REQUESTED_LEVERAGES)
        * len(REFERENCE_EQUITY_USD)
        * len(MARGIN_MODES)
        * len(STRESS_SCENARIOS)
    )
    checks = {
        "experiment_count_preserved": len(prior) == len(status),
        "experiment_ids_unique": status["experiment_id"].nunique() == len(status),
        "status_present": status["leverage_surface_status"].astype(str).ne("").all(),
        "selected_candidates_accounted": len(candidates) <= len(selected_ids),
        "scenario_accounting": len(scenarios) == len(candidates) * expected_per_candidate,
        "canonical_1x_authority_preserved": True,
        "acceptance_disabled": status["acceptance_status"].eq("BLOCKED").all()
        and not status["acceptance_eligible"].astype(bool).any(),
        "testnet_order_authority_disabled": not status[
            "testnet_order_authority"
        ].astype(bool).any(),
        "live_trading_disabled": not status["live_trading_authorized"]
        .astype(bool)
        .any(),
    }
    return pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL"}
            for check, passed in checks.items()
        ]
    )


def _identity(candidate: dict[str, object]) -> dict[str, str]:
    return {
        key: _text(candidate.get(key))
        for key in (
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
    }


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    return {
        "status": active / "current_wizard_hyperliquid_leverage_status.csv",
        "candidates": active / "current_wizard_hyperliquid_leverage_candidates.csv",
        "scenarios": active / "current_wizard_hyperliquid_leverage_scenarios.csv",
        "validation": active / "current_wizard_hyperliquid_leverage_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_leverage_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_leverage_summary.md",
        "snapshot_status": snapshot / "leverage_status.csv",
        "snapshot_candidates": snapshot / "leverage_candidates.csv",
        "snapshot_scenarios": snapshot / "leverage_scenarios.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(
    summary: dict[str, object], candidates: pd.DataFrame
) -> str:
    lines = [
        "# Current Wizard Hyperliquid Leverage Surface",
        "",
        "Canonical 1x remains the statistical-edge authority. Leverage changes sizing, margin, liquidation distance, and loss geometry only.",
        "",
        f"- Leverage surface: `{summary['leverage_surface_id']}`",
        f"- Experiments accounted: {summary['experiments_accounted']}",
        f"- 1x research survivors selected: {summary['one_x_research_survivors_selected']}",
        f"- Candidate surfaces complete: {summary['leverage_candidates_complete']}",
        f"- Scenario rows: {summary['scenario_rows']} / {summary['expected_scenario_rows']}",
        f"- Testnet 1x lifecycle ready: {summary['testnet_1x_lifecycle_ready']}",
        "- Testnet order authority: no",
        "- Live trading authorized: no",
        "",
        "## Candidate Surfaces",
        "",
    ]
    lines.append(
        candidates.to_markdown(index=False)
        if not candidates.empty
        else "No current configuration passed the complete 1x research gate."
    )
    return "\n".join(lines) + "\n"


def _row_lookup(frame: pd.DataFrame, key: str) -> dict[str, dict[str, object]]:
    if frame.empty or key not in frame.columns:
        return {}
    return {_text(row.get(key)): row for row in frame.to_dict("records")}


def _require_unique(
    frame: pd.DataFrame, column: str, *, allow_empty: bool = False
) -> None:
    if frame.empty and allow_empty:
        return
    if frame.empty or column not in frame.columns:
        raise ValueError(f"Current leverage input requires {column}")
    if frame[column].astype(str).duplicated().any():
        raise ValueError(f"Current leverage input requires unique {column}")


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _split_blockers(value: object) -> list[str]:
    return [part.strip() for part in _text(value).split(";") if part.strip()]


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
