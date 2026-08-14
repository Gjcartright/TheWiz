"""Consolidated all-cell failure attribution for the current Wizard board."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.snapshot_lineage import (
    unique_file_bytes,
    verified_snapshot_reference,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_failure_attribution.v1"
ROUTING_SCHEMA_VERSION = "current_wizard_hyperliquid_failure_attribution_routes.v1"

L2_ROUTING_COLUMNS = (
    "experiment_id",
    "pair_group_key",
    "pair",
    "asset_x",
    "asset_y",
    "overall_research_rank",
)

INPUT_FILENAMES = {
    "matrix": "current_wizard_hyperliquid_experiment_matrix.csv",
    "canonical": "current_wizard_hyperliquid_canonical_replay.csv",
    "cost": "current_wizard_hyperliquid_experiment_cost_readiness.csv",
    "observed": "current_wizard_hyperliquid_observed_cost_replay.csv",
    "walkforward": "current_wizard_hyperliquid_walkforward_status.csv",
    "regime": "current_wizard_hyperliquid_regime_status.csv",
    "robustness": "current_wizard_hyperliquid_robustness_status.csv",
    "concentration": "current_wizard_hyperliquid_concentration_status.csv",
}

UPSTREAM_MANIFESTS = {
    "handoff": "current_wizard_hyperliquid_handoff_manifest.json",
    "canonical": "current_wizard_hyperliquid_canonical_replay_manifest.json",
    "cost": "current_wizard_hyperliquid_cost_manifest.json",
    "observed": "current_wizard_hyperliquid_observed_cost_replay_manifest.json",
    "walkforward": "current_wizard_hyperliquid_walkforward_manifest.json",
    "regime": "current_wizard_hyperliquid_regime_manifest.json",
    "robustness": "current_wizard_hyperliquid_robustness_manifest.json",
    "concentration": "current_wizard_hyperliquid_concentration_manifest.json",
}


def build_current_wizard_hyperliquid_failure_attribution(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Join every current-board gate without allowing missing experiment rows."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    input_paths = {name: active / filename for name, filename in INPUT_FILENAMES.items()}
    upstream_manifest_paths = {
        name: active / filename for name, filename in UPSTREAM_MANIFESTS.items()
    }
    robustness_manifest_path = upstream_manifest_paths["robustness"]
    concentration_manifest_path = upstream_manifest_paths["concentration"]
    input_paths["robustness_manifest"] = robustness_manifest_path
    input_paths["concentration_manifest"] = concentration_manifest_path
    missing_paths = [
        str(path)
        for path in (*input_paths.values(), *upstream_manifest_paths.values())
        if not path.exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Current failure-attribution inputs missing: " + ",".join(missing_paths)
        )

    frames = {
        name: pd.read_csv(path, keep_default_na=False)
        for name, path in input_paths.items()
        if name not in {"robustness_manifest", "concentration_manifest"}
    }
    matrix = frames["matrix"]
    if matrix.empty:
        raise ValueError("Current experiment matrix is empty")
    _validate_input_accounting(frames)
    _validate_identity_stability(frames)
    _validate_lineage(
        frames,
        _read_json(robustness_manifest_path),
        _read_json(concentration_manifest_path),
    )

    material = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "experiment_count": int(len(matrix)),
        "input_hashes": {name: _file_hash(path) for name, path in input_paths.items()},
        "authority_policy": {
            "matrix_is_cardinality_authority": True,
            "one_x_research_survivor_requires": [
                "pass_research_walk_forward",
                "pass_research_regime_stability",
                "pass_research_robustness",
                "pass_statistical_selection",
                "pass_cross_cell_concentration",
            ],
            "execution_acceptance_requires": [
                "one_x_research_survivor",
                "strict_l2_cost_calibration",
                "vendor_exact_mode_parity",
            ],
            "leverage_is_research_only": True,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    }
    attribution_id = "cwfailure_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    upstream_manifests = {
        name: _read_json(path) for name, path in upstream_manifest_paths.items()
    }
    concentration_manifest = upstream_manifests["concentration"]
    source_snapshot = root / _text(
        concentration_manifest.get("artifacts", {}).get("snapshot_manifest")
    )
    if not source_snapshot.exists():
        raise FileNotFoundError("Current concentration snapshot manifest is missing")
    snapshot_dir = source_snapshot.parent / "failure_attribution" / attribution_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_specs = {
        "matrix": ("handoff", "experiments"),
        "canonical": ("canonical", "results"),
        "cost": ("cost", "experiments"),
        "observed": ("observed", "results"),
        "walkforward": ("walkforward", "status"),
        "regime": ("regime", "status"),
        "robustness": ("robustness", "status"),
        "concentration": ("concentration", "status"),
        "robustness_manifest": ("robustness", "manifest"),
        "concentration_manifest": ("concentration", "manifest"),
    }
    snapshot_inputs = {
        name: verified_snapshot_reference(
            root=root,
            active_path=source,
            upstream_manifest=upstream_manifests[manifest_name],
            artifact_key=artifact_key,
        )
        for name, source in input_paths.items()
        for manifest_name, artifact_key in (snapshot_specs[name],)
    }

    lookups = {
        name: frame.set_index("experiment_id", drop=False)
        for name, frame in frames.items()
        if name != "matrix"
    }
    rows = [
        _attribution_row(
            matrix_row,
            lookups=lookups,
            attribution_id=attribution_id,
            evidence_paths=snapshot_inputs.values(),
            root=root,
        )
        for matrix_row in matrix.itertuples(index=False)
    ]
    attribution = pd.DataFrame(rows)
    attribution["evidence_path"] = (
        "reports/active/current_wizard_hyperliquid_failure_attribution_manifest.json"
    )
    attribution = _rank_attribution(attribution)
    blocker_summary = _blocker_summary(attribution)
    validation = _validation(matrix, attribution, frames)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError(
            "Current failure-attribution validation failed: " + ",".join(failed)
        )

    paths = _paths(active, snapshot_dir)
    for frame, active_key, snapshot_key in (
        (attribution, "attribution", "snapshot_attribution"),
        (blocker_summary, "blocker_summary", "snapshot_blocker_summary"),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)

    status_counts = attribution["consolidated_status"].value_counts().to_dict()
    stage_counts = attribution["first_blocking_stage"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "failure_attribution_id": attribution_id,
        "concentration_id": _text(concentration_manifest.get("concentration_id")),
        "experiments_accounted": int(len(attribution)),
        "unique_experiment_ids": int(attribution["experiment_id"].nunique()),
        "experiment_status_accounted": bool(
            len(attribution) == len(matrix)
            and attribution["experiment_id"].nunique() == len(matrix)
        ),
        "status_counts": status_counts,
        "first_blocking_stage_counts": stage_counts,
        "one_x_research_survivors": int(
            attribution["one_x_research_survivor"].astype(bool).sum()
        ),
        "leverage_research_eligible": int(
            attribution["leverage_research_eligible"].astype(bool).sum()
        ),
        "strict_cost_calibrated_survivors": int(
            (
                attribution["one_x_research_survivor"].astype(bool)
                & attribution["strict_cost_calibration_ready"].astype(bool)
            ).sum()
        ),
        "vendor_parity_proven_survivors": int(
            (
                attribution["one_x_research_survivor"].astype(bool)
                & attribution["vendor_exact_mode_parity_proven"].astype(bool)
            ).sum()
        ),
        "execution_acceptance_ready": int(
            attribution["execution_acceptance_ready"].astype(bool).sum()
        ),
        "testnet_preflight_eligible": int(
            attribution["testnet_preflight_eligible"].astype(bool).sum()
        ),
        "acceptance_eligible_replays": 0,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
        "input_snapshot_modes": {
            name: "verified_upstream_reference" for name in snapshot_inputs
        },
        "referenced_upstream_bytes": unique_file_bytes(
            list(snapshot_inputs.values())
        ),
        "locally_copied_input_bytes": 0,
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary, attribution)
    for path in (paths["manifest"], paths["snapshot_manifest"]):
        path.write_text(manifest_text, encoding="utf-8")
    for path in (paths["summary_md"], paths["snapshot_summary_md"]):
        path.write_text(summary_text, encoding="utf-8")
    routing = build_current_wizard_hyperliquid_failure_routing_index(
        root=root,
        routes=attribution.loc[:, list(L2_ROUTING_COLUMNS)].copy(),
    )
    paths["route_index"] = routing.paths["route_index"]
    paths["route_manifest"] = routing.paths["route_manifest"]
    paths["immutable_route_index"] = routing.paths["immutable_route_index"]
    paths["immutable_route_manifest"] = routing.paths["immutable_route_manifest"]
    return CommandResult(paths=paths, summary=summary)


def build_current_wizard_hyperliquid_failure_routing_index(
    *,
    root: Path = ROOT,
    routes: pd.DataFrame | None = None,
) -> CommandResult:
    """Publish a compact immutable L2 route index without mutating research lineage."""

    active = root / "reports" / "active"
    source_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    )
    source_manifest = _read_json(source_manifest_path)
    source_id = _text(source_manifest.get("failure_attribution_id"))
    artifacts = source_manifest.get("artifacts", {})
    if not source_id or not isinstance(artifacts, dict):
        raise ValueError("Current failure-attribution manifest is incomplete")
    source_active_path = root / _text(artifacts.get("attribution"))
    source_snapshot_path = root / _text(artifacts.get("snapshot_attribution"))
    if not source_active_path.is_file() or not source_snapshot_path.is_file():
        raise FileNotFoundError("Current failure-attribution evidence is missing")
    source_active_hash = _file_hash(source_active_path)
    source_snapshot_hash = _file_hash(source_snapshot_path)
    if source_active_hash != source_snapshot_hash:
        raise ValueError("Current failure-attribution active/snapshot mismatch")
    if routes is None:
        routes = pd.read_csv(
            source_snapshot_path,
            usecols=list(L2_ROUTING_COLUMNS),
            keep_default_na=False,
        )
    missing = set(L2_ROUTING_COLUMNS) - set(routes.columns)
    if missing:
        raise ValueError(
            "Failure-attribution routing columns missing: " + ",".join(sorted(missing))
        )
    routes = routes.loc[:, list(L2_ROUTING_COLUMNS)].copy()
    experiment_ids = routes["experiment_id"].astype(str)
    expected_rows = int(source_manifest.get("experiments_accounted", -1))
    expected_unique = int(source_manifest.get("unique_experiment_ids", -1))
    if (
        routes.empty
        or experiment_ids.eq("").any()
        or experiment_ids.duplicated().any()
        or len(routes) != expected_rows
        or experiment_ids.nunique() != expected_unique
    ):
        raise ValueError("Failure-attribution routing identity is incomplete")
    route_bytes = routes.to_csv(index=False).encode("utf-8")
    route_hash = sha256(route_bytes).hexdigest()
    source_manifest_hash = _file_hash(source_manifest_path)
    material = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "source_failure_attribution_id": source_id,
        "source_manifest_sha256": source_manifest_hash,
        "source_attribution_sha256": source_active_hash,
        "route_index_sha256": route_hash,
        "route_index_rows": len(routes),
        "route_index_unique_experiment_ids": int(experiment_ids.nunique()),
    }
    route_id = "cwroutes_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    immutable_dir = (
        root / "data" / "research" / "wizard_failure_attribution_routes" / route_id
    )
    immutable_route_path = immutable_dir / "routing_index.csv"
    immutable_manifest_path = immutable_dir / "manifest.json"
    active_route_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes.csv"
    )
    active_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes_manifest.json"
    )
    payload = {
        **material,
        "route_id": route_id,
        "source_manifest_path": _relative(source_manifest_path, root),
        "source_active_attribution_path": _relative(source_active_path, root),
        "source_snapshot_attribution_path": _relative(source_snapshot_path, root),
        "source_as_of": _text(source_manifest.get("as_of")),
        "active_route_index_path": _relative(active_route_path, root),
        "immutable_route_index_path": _relative(immutable_route_path, root),
        "immutable_route_manifest_path": _relative(immutable_manifest_path, root),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    manifest_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write_or_validate_immutable_bytes(route_bytes, immutable_route_path)
    _write_or_validate_immutable_bytes(manifest_bytes, immutable_manifest_path)
    _atomic_write_bytes(route_bytes, active_route_path)
    _atomic_write_bytes(manifest_bytes, active_manifest_path)
    return CommandResult(
        paths={
            "route_index": active_route_path,
            "route_manifest": active_manifest_path,
            "immutable_route_index": immutable_route_path,
            "immutable_route_manifest": immutable_manifest_path,
        },
        summary=payload,
    )


def _attribution_row(
    matrix_row: object,
    *,
    lookups: dict[str, pd.DataFrame],
    attribution_id: str,
    evidence_paths: Any,
    root: Path,
) -> dict[str, object]:
    experiment_id = _text(matrix_row.experiment_id)
    canonical = lookups["canonical"].loc[experiment_id]
    cost = lookups["cost"].loc[experiment_id]
    observed = lookups["observed"].loc[experiment_id]
    walkforward = lookups["walkforward"].loc[experiment_id]
    regime = lookups["regime"].loc[experiment_id]
    robustness = lookups["robustness"].loc[experiment_id]
    concentration = lookups["concentration"].loc[experiment_id]

    matrix_status = _text(matrix_row.experiment_status)
    canonical_status = _text(canonical["replay_status"])
    cost_status = _text(cost["cost_replay_status"])
    observed_status = _text(observed["replay_status"])
    walkforward_status = _text(walkforward["walkforward_status"])
    regime_status = _text(regime["regime_status"])
    robustness_status = _text(robustness["robustness_status"])
    concentration_status = _text(concentration["concentration_status"])

    canonical_complete = canonical_status == "RESEARCH_REPLAY_COMPLETE"
    observed_complete = observed_status == "OBSERVED_COST_RESEARCH_REPLAY_COMPLETE"
    walkforward_pass = walkforward_status == "PASS_RESEARCH_WALK_FORWARD"
    regime_pass = (
        _text(regime["regime_stability_status"])
        == "PASS_RESEARCH_REGIME_STABILITY"
    )
    robustness_pass = (
        _text(robustness["research_robustness_status"])
        == "PASS_RESEARCH_ROBUSTNESS"
    )
    statistical_pass = (
        _text(robustness["statistical_selection_status"]) == "PASS"
        and _text(robustness["promotion_readiness"])
        == "READY_FOR_NEXT_RESEARCH_GATE"
    )
    concentration_pass = _truthy(concentration["ready_for_leverage_gate"])
    one_x_survivor = bool(
        canonical_complete
        and observed_complete
        and walkforward_pass
        and regime_pass
        and robustness_pass
        and statistical_pass
        and concentration_pass
    )
    strict_cost_ready = _truthy(observed["strict_cost_calibration_ready"])
    vendor_parity = _truthy(matrix_row.vendor_parity_claimed)
    execution_ready = bool(one_x_survivor and strict_cost_ready and vendor_parity)

    blocking_stage, blocker, next_action, progress_rank = _first_blocker(
        matrix_row=matrix_row,
        canonical=canonical,
        cost=cost,
        observed=observed,
        walkforward=walkforward,
        regime=regime,
        robustness=robustness,
        concentration=concentration,
        flags={
            "canonical_complete": canonical_complete,
            "observed_complete": observed_complete,
            "walkforward_pass": walkforward_pass,
            "regime_pass": regime_pass,
            "robustness_pass": robustness_pass,
            "statistical_pass": statistical_pass,
            "concentration_pass": concentration_pass,
            "strict_cost_ready": strict_cost_ready,
            "vendor_parity": vendor_parity,
        },
    )
    all_blockers = _all_independent_blockers(
        matrix_row=matrix_row,
        canonical=canonical,
        cost=cost,
        observed=observed,
        walkforward=walkforward,
        regime=regime,
        robustness=robustness,
        concentration=concentration,
        flags={
            "canonical_complete": canonical_complete,
            "observed_complete": observed_complete,
            "walkforward_pass": walkforward_pass,
            "regime_pass": regime_pass,
            "robustness_pass": robustness_pass,
            "statistical_pass": statistical_pass,
            "concentration_pass": concentration_pass,
            "strict_cost_ready": strict_cost_ready,
            "vendor_parity": vendor_parity,
        },
    )
    if execution_ready:
        consolidated_status = "READY_FOR_READ_ONLY_TESTNET_PREFLIGHT"
    elif one_x_survivor:
        consolidated_status = "ONE_X_RESEARCH_SURVIVOR_EXECUTION_BLOCKED"
    else:
        consolidated_status = "BLOCKED_" + blocking_stage.upper()

    return {
        "schema_version": SCHEMA_VERSION,
        "failure_attribution_id": attribution_id,
        "experiment_id": experiment_id,
        "pair_group_key": _text(matrix_row.pair_group_key),
        "pair": _text(matrix_row.pair),
        "wizard_exchange": _text(matrix_row.wizard_exchange),
        "wizard_timeframe": _text(matrix_row.timeframe),
        "exact_mode": _text(matrix_row.exact_mode),
        "orientation": _text(matrix_row.orientation),
        "asset_x": _text(matrix_row.asset_a),
        "asset_y": _text(matrix_row.asset_b),
        "matrix_status": matrix_status,
        "canonical_replay_status": canonical_status,
        "cost_evidence_status": cost_status,
        "observed_cost_replay_status": observed_status,
        "walkforward_status": walkforward_status,
        "regime_status": regime_status,
        "regime_stability_status": _text(regime["regime_stability_status"]),
        "robustness_status": robustness_status,
        "research_robustness_status": _text(
            robustness["research_robustness_status"]
        ),
        "statistical_selection_status": _text(
            robustness["statistical_selection_status"]
        ),
        "concentration_status": concentration_status,
        "concentration_gate_pass": concentration_pass,
        "strict_cost_calibration_ready": strict_cost_ready,
        "mode_fidelity_status": _text(observed["mode_fidelity_status"]),
        "vendor_exact_mode_parity_proven": vendor_parity,
        "one_x_research_survivor": one_x_survivor,
        "leverage_research_eligible": one_x_survivor,
        "execution_acceptance_ready": execution_ready,
        "testnet_preflight_eligible": execution_ready,
        "consolidated_status": consolidated_status,
        "first_blocking_stage": blocking_stage,
        "first_blocker": blocker,
        "all_independent_blockers": ";".join(all_blockers),
        "next_action": next_action,
        "research_progress_rank": progress_rank,
        "observed_trades": _number(observed["trades"]),
        "observed_profit_factor": _number(observed["profit_factor"]),
        "observed_sharpe": _number(observed["sharpe"]),
        "observed_max_drawdown": _number(observed["max_drawdown"]),
        "observed_total_return": _number(observed["total_return"]),
        "walkforward_trades": _number(walkforward["aggregate_trades"]),
        "walkforward_profit_factor": _number(
            walkforward["aggregate_profit_factor"]
        ),
        "walkforward_sharpe": _number(walkforward["aggregate_sharpe"]),
        "walkforward_max_drawdown": _number(
            walkforward["aggregate_max_drawdown"]
        ),
        "walkforward_total_return": _number(
            walkforward["aggregate_total_return"]
        ),
        "walkforward_bh_qvalue": _number(walkforward["bh_qvalue"]),
        "hedge_ratio_cv": _number(walkforward["hedge_ratio_cv"]),
        "regime_profit_concentration": _number(
            regime["regime_profit_concentration"]
        ),
        "worst_regime_expectancy": _number(regime["worst_regime_expectancy"]),
        "robustness_parameter_pass_ratio": _number(
            robustness["parameter_pass_ratio"]
        ),
        "worst_robustness_drawdown": _number(
            robustness["worst_parameter_drawdown"]
        ),
        "leverage_research_only": True,
        "acceptance_status": "BLOCKED",
        "acceptance_reason": (
            "failure_attribution_is_research_only;testnet_order_authority_false;"
            "live_trading_not_authorized"
        ),
        "acceptance_eligible": False,
        "testnet_order_authority": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "evidence_path": ";".join(
            _relative(Path(path), root) for path in evidence_paths
        ),
    }


def _first_blocker(
    *,
    matrix_row: object,
    canonical: pd.Series,
    cost: pd.Series,
    observed: pd.Series,
    walkforward: pd.Series,
    regime: pd.Series,
    robustness: pd.Series,
    concentration: pd.Series,
    flags: dict[str, bool],
) -> tuple[str, str, str, int]:
    matrix_status = _text(matrix_row.experiment_status)
    if matrix_status == "BLOCKED_HYPERLIQUID_MAPPING":
        return (
            "hyperliquid_mapping",
            _text(matrix_row.experiment_blocker) or matrix_status,
            "resolve_symbol_or_venue_mapping",
            0,
        )
    if matrix_status == "NOT_APPLICABLE_LOCAL_IMPLEMENTATION":
        return (
            "local_mode_implementation",
            _text(matrix_row.experiment_blocker) or matrix_status,
            "implement_or_confirm_exact_mode_not_applicable",
            1,
        )
    if matrix_status != "READY_FOR_POINT_IN_TIME_HISTORY":
        return (
            "handoff",
            _text(matrix_row.experiment_blocker) or matrix_status,
            "repair_current_board_handoff",
            1,
        )
    if not flags["canonical_complete"]:
        return (
            "point_in_time_history"
            if _text(canonical["replay_status"]) == "DEFERRED_POINT_IN_TIME_HISTORY"
            else "canonical_replay",
            _text(canonical["replay_blocker"]) or _text(canonical["replay_status"]),
            "fetch_cutoff_bounded_pair_history"
            if _text(canonical["replay_status"]) == "DEFERRED_POINT_IN_TIME_HISTORY"
            else "repair_canonical_replay",
            2,
        )
    if _text(cost["cost_replay_status"]) != "READY_FOR_OBSERVED_COST_RESEARCH":
        return (
            "cost_evidence",
            _text(cost["cost_replay_blocker"]) or _text(cost["cost_replay_status"]),
            "attach_hyperliquid_funding_and_l2_cost_evidence",
            3,
        )
    if not flags["observed_complete"]:
        return (
            "observed_cost_replay",
            _text(observed["replay_blocker"]) or _text(observed["replay_status"]),
            "rerun_observed_cost_replay",
            4,
        )
    if not flags["walkforward_pass"]:
        return (
            "walk_forward",
            _text(walkforward["walkforward_blocker"])
            or _text(walkforward["walkforward_status"]),
            "investigate_walk_forward_failure",
            5,
        )
    if not flags["regime_pass"]:
        return (
            "regime_stability",
            _text(regime["regime_stability_blocker"])
            or _text(regime["regime_status"]),
            "investigate_regime_instability",
            6,
        )
    if not flags["robustness_pass"]:
        return (
            "robustness",
            _text(robustness["research_robustness_blocker"])
            or _text(robustness["robustness_status"]),
            "investigate_parameter_and_cost_fragility",
            7,
        )
    if not flags["statistical_pass"]:
        return (
            "statistical_selection",
            _text(robustness["promotion_blocker"])
            or _text(robustness["statistical_selection_status"]),
            "retain_on_watchlist_or_collect_more_out_of_sample_history",
            8,
        )
    if not flags["concentration_pass"]:
        return (
            "cross_cell_concentration",
            _text(concentration["concentration_blocker"])
            or _text(concentration["concentration_status"]),
            "collect_a_broader_statistically_selected_cohort",
            9,
        )
    if not flags["strict_cost_ready"]:
        return (
            "strict_l2_cost_calibration",
            "strict_l2_calibration_incomplete",
            "collect_strict_l2_calibration_then_rerun",
            10,
        )
    if not flags["vendor_parity"]:
        return (
            "vendor_exact_mode_parity",
            _text(observed["mode_fidelity_reason"])
            or "vendor_custom_series_parity_not_proven",
            "capture_vendor_custom_series_and_prove_mode_parity",
            11,
        )
    return (
        "read_only_testnet_preflight",
        "testnet_preflight_not_yet_run",
        "run_read_only_testnet_preflight",
        12,
    )


def _all_independent_blockers(
    *,
    matrix_row: object,
    canonical: pd.Series,
    cost: pd.Series,
    observed: pd.Series,
    walkforward: pd.Series,
    regime: pd.Series,
    robustness: pd.Series,
    concentration: pd.Series,
    flags: dict[str, bool],
) -> list[str]:
    blockers: list[str] = []
    matrix_status = _text(matrix_row.experiment_status)
    if matrix_status != "READY_FOR_POINT_IN_TIME_HISTORY":
        blockers.append(
            "handoff:" + (_text(matrix_row.experiment_blocker) or matrix_status)
        )
        return blockers
    if not flags["canonical_complete"]:
        blockers.append(
            "canonical:" + (
                _text(canonical["replay_blocker"]) or _text(canonical["replay_status"])
            )
        )
        return blockers
    if _text(cost["cost_replay_status"]) != "READY_FOR_OBSERVED_COST_RESEARCH":
        blockers.append(
            "cost:" + (
                _text(cost["cost_replay_blocker"]) or _text(cost["cost_replay_status"])
            )
        )
        return blockers
    if not flags["observed_complete"]:
        blockers.append(
            "observed:" + (
                _text(observed["replay_blocker"]) or _text(observed["replay_status"])
            )
        )
        return blockers
    if not flags["walkforward_pass"]:
        blockers.append(
            "walk_forward:" + (
                _text(walkforward["walkforward_blocker"])
                or _text(walkforward["walkforward_status"])
            )
        )
    if flags["walkforward_pass"] and not flags["regime_pass"]:
        blockers.append(
            "regime:" + (
                _text(regime["regime_stability_blocker"])
                or _text(regime["regime_status"])
            )
        )
    if flags["walkforward_pass"] and not flags["robustness_pass"]:
        blockers.append(
            "robustness:" + (
                _text(robustness["research_robustness_blocker"])
                or _text(robustness["robustness_status"])
            )
        )
    if (
        flags["walkforward_pass"]
        and flags["regime_pass"]
        and flags["robustness_pass"]
        and not flags["statistical_pass"]
    ):
        blockers.append(
            "statistical:" + (
                _text(robustness["promotion_blocker"])
                or _text(robustness["statistical_selection_status"])
            )
        )
    if flags["statistical_pass"] and not flags["concentration_pass"]:
        blockers.append(
            "concentration:"
            + (
                _text(concentration["concentration_blocker"])
                or _text(concentration["concentration_status"])
            )
        )
    if flags["walkforward_pass"] and not flags["strict_cost_ready"]:
        blockers.append("execution_cost:strict_l2_calibration_incomplete")
    if flags["walkforward_pass"] and not flags["vendor_parity"]:
        blockers.append("mode_fidelity:vendor_custom_series_parity_not_proven")
    return blockers


def _rank_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values(
        [
            "leverage_research_eligible",
            "execution_acceptance_ready",
            "research_progress_rank",
            "walkforward_sharpe",
            "walkforward_profit_factor",
            "experiment_id",
        ],
        ascending=[False, False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranked.insert(0, "overall_research_rank", range(1, len(ranked) + 1))
    survivor_rank = pd.Series(pd.NA, index=ranked.index, dtype="Int64")
    survivors = ranked["one_x_research_survivor"].astype(bool)
    survivor_rank.loc[survivors] = range(1, int(survivors.sum()) + 1)
    ranked.insert(1, "one_x_survivor_rank", survivor_rank)
    return ranked


def _blocker_summary(attribution: pd.DataFrame) -> pd.DataFrame:
    return (
        attribution.groupby(
            [
                "first_blocking_stage",
                "first_blocker",
                "next_action",
                "consolidated_status",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="experiment_count")
        .sort_values(
            ["experiment_count", "first_blocking_stage"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )


def _validation(
    matrix: pd.DataFrame,
    attribution: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    matrix_ids = set(matrix["experiment_id"].astype(str))
    survivor = attribution["one_x_research_survivor"].astype(bool)
    execution_ready = attribution["execution_acceptance_ready"].astype(bool)
    checks = {
        "experiment_count_preserved": len(attribution) == len(matrix),
        "experiment_ids_unique": attribution["experiment_id"].nunique() == len(matrix),
        "experiment_id_set_preserved": set(attribution["experiment_id"].astype(str))
        == matrix_ids,
        "all_stage_inputs_accounted": all(
            len(frame) == len(matrix)
            and set(frame["experiment_id"].astype(str)) == matrix_ids
            for frame in frames.values()
        ),
        "consolidated_status_present": attribution["consolidated_status"]
        .astype(str)
        .ne("")
        .all(),
        "first_blocker_present": attribution["first_blocker"].astype(str).ne("").all(),
        "next_action_present": attribution["next_action"].astype(str).ne("").all(),
        "survivor_requires_walk_forward": (
            ~survivor
            | attribution["walkforward_status"].eq("PASS_RESEARCH_WALK_FORWARD")
        ).all(),
        "survivor_requires_regime_stability": (
            ~survivor
            | attribution["regime_stability_status"].eq(
                "PASS_RESEARCH_REGIME_STABILITY"
            )
        ).all(),
        "survivor_requires_robustness": (
            ~survivor
            | attribution["research_robustness_status"].eq(
                "PASS_RESEARCH_ROBUSTNESS"
            )
        ).all(),
        "survivor_requires_statistical_selection": (
            ~survivor
            | attribution["statistical_selection_status"].eq("PASS")
        ).all(),
        "survivor_requires_cross_cell_concentration": (
            ~survivor | attribution["concentration_gate_pass"].astype(bool)
        ).all(),
        "execution_requires_research_survivor": (~execution_ready | survivor).all(),
        "execution_requires_strict_cost": (
            ~execution_ready
            | attribution["strict_cost_calibration_ready"].astype(bool)
        ).all(),
        "execution_requires_vendor_parity": (
            ~execution_ready
            | attribution["vendor_exact_mode_parity_proven"].astype(bool)
        ).all(),
        "testnet_order_authority_disabled": not attribution[
            "testnet_order_authority"
        ].astype(bool).any(),
        "acceptance_disabled": attribution["acceptance_status"].eq("BLOCKED").all()
        and not attribution["acceptance_eligible"].astype(bool).any(),
        "live_trading_disabled": not attribution["live_trading_authorized"]
        .astype(bool)
        .any(),
    }
    return pd.DataFrame(
        [
            {"check": check, "status": "PASS" if passed else "FAIL"}
            for check, passed in checks.items()
        ]
    )


def _validate_input_accounting(frames: dict[str, pd.DataFrame]) -> None:
    matrix_ids = set(frames["matrix"]["experiment_id"].astype(str))
    for name, frame in frames.items():
        if "experiment_id" not in frame.columns:
            raise ValueError(f"Current {name} input lacks experiment_id")
        if frame["experiment_id"].astype(str).duplicated().any():
            raise ValueError(f"Current {name} input contains duplicate experiment_id")
        ids = set(frame["experiment_id"].astype(str))
        if ids != matrix_ids:
            missing = len(matrix_ids - ids)
            extra = len(ids - matrix_ids)
            raise ValueError(
                f"Current {name} experiment accounting changed: missing={missing},extra={extra}"
            )


def _validate_identity_stability(frames: dict[str, pd.DataFrame]) -> None:
    authority = frames["matrix"].set_index("experiment_id")
    expected = authority[["pair_group_key", "pair", "exact_mode", "orientation"]]
    for name, frame in frames.items():
        if name == "matrix":
            continue
        candidate = frame.set_index("experiment_id")
        shared = [column for column in expected.columns if column in candidate.columns]
        if not shared:
            continue
        left = expected[shared].astype(str).sort_index()
        right = candidate[shared].astype(str).sort_index()
        if not left.equals(right):
            raise ValueError(f"Current {name} experiment identity changed")


def _validate_lineage(
    frames: dict[str, pd.DataFrame],
    robustness_manifest: dict[str, Any],
    concentration_manifest: dict[str, Any],
) -> None:
    robust_id = _single_nonempty(frames["robustness"], "robustness_id")
    if robust_id != _text(robustness_manifest.get("robustness_id")):
        raise ValueError("Current robustness manifest identity does not match status")
    concentration_id = _single_nonempty(frames["concentration"], "concentration_id")
    if concentration_id != _text(concentration_manifest.get("concentration_id")):
        raise ValueError("Current concentration manifest identity does not match status")
    concentration_robustness_id = _text(
        concentration_manifest.get("robustness_id")
    )
    if concentration_robustness_id != robust_id:
        raise ValueError("Current concentration lineage does not match robustness")
    walk_id = _single_nonempty(frames["walkforward"], "walkforward_id")
    regime_walk_id = _single_nonempty(frames["regime"], "walkforward_id")
    robust_walk_id = _single_nonempty(frames["robustness"], "walkforward_id")
    concentration_walk_id = _text(concentration_manifest.get("walkforward_id"))
    if (
        not walk_id
        or walk_id != regime_walk_id
        or walk_id != robust_walk_id
        or walk_id != concentration_walk_id
    ):
        raise ValueError("Current walk-forward lineage does not match downstream stages")
    observed_id = _single_nonempty(frames["observed"], "observed_cost_replay_id")
    walk_observed_id = _single_nonempty(
        frames["walkforward"], "observed_cost_replay_id"
    )
    if not observed_id or observed_id != walk_observed_id:
        raise ValueError("Current observed replay lineage does not match walk-forward")


def _single_nonempty(frame: pd.DataFrame, column: str) -> str:
    values = {_text(value) for value in frame[column].tolist() if _text(value)}
    if len(values) != 1:
        raise ValueError(f"Expected one non-empty {column}, found {len(values)}")
    return next(iter(values))


def _paths(active: Path, snapshot: Path) -> dict[str, Path]:
    return {
        "attribution": active
        / "current_wizard_hyperliquid_failure_attribution.csv",
        "blocker_summary": active
        / "current_wizard_hyperliquid_failure_attribution_summary.csv",
        "validation": active
        / "current_wizard_hyperliquid_failure_attribution_validation.csv",
        "manifest": active
        / "current_wizard_hyperliquid_failure_attribution_manifest.json",
        "summary_md": active
        / "current_wizard_hyperliquid_failure_attribution_summary.md",
        "snapshot_attribution": snapshot / "failure_attribution.csv",
        "snapshot_blocker_summary": snapshot / "failure_attribution_summary.csv",
        "snapshot_validation": snapshot / "validation.csv",
        "snapshot_manifest": snapshot / "manifest.json",
        "snapshot_summary_md": snapshot / "summary.md",
    }


def _summary_markdown(
    summary: dict[str, object], attribution: pd.DataFrame
) -> str:
    lines = [
        "# Current Wizard Hyperliquid Failure Attribution",
        "",
        f"- Attribution: `{summary['failure_attribution_id']}`",
        f"- Experiments accounted: {summary['experiments_accounted']}",
        f"- 1x research survivors: {summary['one_x_research_survivors']}",
        f"- Leverage-research eligible: {summary['leverage_research_eligible']}",
        f"- Strict-cost calibrated survivors: {summary['strict_cost_calibrated_survivors']}",
        f"- Vendor-parity proven survivors: {summary['vendor_parity_proven_survivors']}",
        f"- Execution-acceptance ready: {summary['execution_acceptance_ready']}",
        f"- Testnet-preflight eligible: {summary['testnet_preflight_eligible']}",
        "- Testnet order authority: no",
        "- Live trading authorized: no",
        "",
        "## Research Survivors",
        "",
    ]
    survivors = attribution.loc[attribution["one_x_research_survivor"].astype(bool)]
    if survivors.empty:
        lines.append("- none")
    else:
        for row in survivors.itertuples(index=False):
            lines.append(
                f"- {row.pair} | {row.exact_mode} | {row.orientation} | "
                f"walk-forward Sharpe {_display(row.walkforward_sharpe)} | "
                f"next: {row.next_action}"
            )
    lines.extend(["", "## First Blocking Stages", ""])
    for stage, count in summary["first_blocking_stage_counts"].items():
        lines.append(f"- {stage}: {count}")
    return "\n".join(lines) + "\n"


def _display(value: object) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.4f}"


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )


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


def _write_or_validate_immutable_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"Immutable routing artifact mismatch: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_bytes(payload: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
