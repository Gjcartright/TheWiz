"""Current-board Crypto Wizards to Hyperliquid local-research handoff.

The current API refresh is kept separate from the frozen browser run. Every
current pair receives a vendor-detail status, a Hyperliquid mapping outcome,
and all exact-mode/orientation experiment rows without discovery filtering.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_replay import (
    HISTORY_DAYS,
    MINIMUM_HISTORY_ROWS,
    TIMEFRAME_INTERVALS,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    EXACT_MODES,
    ORIENTATIONS,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_handoff.v1"
PAIR_PAGE_IMPLEMENTED_MODES = tuple(
    mode for mode in EXACT_MODES if mode != "OU (Optimal)"
)


def build_current_wizard_hyperliquid_handoff(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Build complete current-refresh status, experiment, and history ledgers."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    inputs = {
        "refresh_manifest": active / "exhaustive_wizard_api_refresh_manifest.json",
        "pair_detail_queue": active
        / "exhaustive_wizard_api_refresh_pair_detail_queue.csv",
        "source_accounting": active
        / "exhaustive_wizard_api_refresh_source_accounting.csv",
        "hyperliquid_mapping": active
        / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv",
        "api_pilot_manifest": active / "wizard_pair_detail_api_pilot_manifest.json",
        "api_pilot_endpoints": active / "wizard_pair_detail_api_pilot_manifest.csv",
        "api_pilot_coverage": active / "wizard_pair_detail_api_pilot_coverage.csv",
        "frozen_capture_progress": active
        / "exhaustive_wizard_pair_detail_capture_progress.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current Wizard handoff inputs missing: {missing}")

    refresh = _read_json(inputs["refresh_manifest"])
    queue = _read_csv(inputs["pair_detail_queue"])
    source_accounting = _read_csv(inputs["source_accounting"])
    mapping = _read_csv(inputs["hyperliquid_mapping"])
    pilot = _read_json(inputs["api_pilot_manifest"])
    pilot_endpoints = _read_csv(inputs["api_pilot_endpoints"])
    pilot_coverage = _read_csv(inputs["api_pilot_coverage"])
    frozen_progress = _read_csv(inputs["frozen_capture_progress"])
    refresh_id = _text(refresh.get("refresh_id"))
    if not refresh_id:
        raise ValueError("Current Wizard refresh manifest has no refresh_id")
    if queue["pair_group_key"].astype(str).duplicated().any():
        raise ValueError("Current pair-detail queue has duplicate pair_group_key rows")

    input_hashes = {name: _file_hash(path) for name, path in inputs.items()}
    material = {
        "schema_version": SCHEMA_VERSION,
        "refresh_id": refresh_id,
        "input_hashes": input_hashes,
        "exact_modes": EXACT_MODES,
        "orientations": ORIENTATIONS,
        "pair_page_implemented_modes": PAIR_PAGE_IMPLEMENTED_MODES,
        "scanner_boolean_overlays": ("ou_optimal",),
        "canonical_leverage": 1.0,
    }
    handoff_id = "cwhandoff_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    snapshot_dir = (
        root
        / "reports"
        / "snapshots"
        / "current_wizard_hyperliquid"
        / refresh_id
        / handoff_id
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    input_snapshot_dir = snapshot_dir / "inputs"
    input_snapshot_dir.mkdir(parents=True, exist_ok=True)
    input_snapshots: dict[str, Path] = {}
    for name, source in inputs.items():
        destination = input_snapshot_dir / source.name
        shutil.copy2(source, destination)
        input_snapshots[name] = destination

    pair_status = _build_pair_status(
        queue,
        mapping=mapping,
        frozen_progress=frozen_progress,
        pilot=pilot,
        pilot_endpoints=pilot_endpoints,
        pilot_coverage=pilot_coverage,
        handoff_id=handoff_id,
        refresh_id=refresh_id,
        evidence_path=_relative(snapshot_dir / "manifest.json", root),
    )
    experiments = _build_experiment_matrix(
        pair_status,
        source_accounting=source_accounting,
        handoff_id=handoff_id,
        refresh_id=refresh_id,
    )
    pair_history_queue = _build_pair_history_queue(
        pair_status,
        experiments=experiments,
        handoff_id=handoff_id,
        refresh_id=refresh_id,
    )
    asset_fetch_queue = _build_asset_fetch_queue(
        pair_history_queue,
        handoff_id=handoff_id,
        refresh_id=refresh_id,
    )
    validation = _build_validation(
        queue=queue,
        mapping=mapping,
        pair_status=pair_status,
        experiments=experiments,
        pair_history_queue=pair_history_queue,
        pilot=pilot,
    )
    validation.insert(1, "handoff_id", handoff_id)
    failures = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
    if failures:
        raise ValueError("Current Wizard handoff validation failed: " + ",".join(failures))

    paths = {
        "pair_status": active / "current_wizard_pair_detail_status.csv",
        "experiments": active / "current_wizard_hyperliquid_experiment_matrix.csv",
        "pair_history_queue": active
        / "current_wizard_hyperliquid_pair_history_queue.csv",
        "asset_fetch_queue": active
        / "current_wizard_hyperliquid_asset_fetch_queue.csv",
        "validation": active / "current_wizard_hyperliquid_handoff_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_handoff_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_handoff_summary.md",
        "snapshot_pair_status": snapshot_dir / "pair_detail_status.csv",
        "snapshot_experiments": snapshot_dir / "experiment_matrix.csv",
        "snapshot_pair_history_queue": snapshot_dir / "pair_history_queue.csv",
        "snapshot_asset_fetch_queue": snapshot_dir / "asset_fetch_queue.csv",
        "snapshot_validation": snapshot_dir / "validation.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for frame, active_key, snapshot_key in (
        (pair_status, "pair_status", "snapshot_pair_status"),
        (experiments, "experiments", "snapshot_experiments"),
        (pair_history_queue, "pair_history_queue", "snapshot_pair_history_queue"),
        (asset_fetch_queue, "asset_fetch_queue", "snapshot_asset_fetch_queue"),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)

    vendor_counts = pair_status["vendor_pair_detail_status"].value_counts().to_dict()
    experiment_counts = experiments["experiment_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "handoff_id": handoff_id,
        "refresh_id": refresh_id,
        "created_at": as_of.isoformat(),
        "pair_groups": int(len(pair_status)),
        "pair_groups_accounted": int(pair_status["pair_group_key"].nunique()),
        "api_schema_pilot_pairs": int(pair_status["api_schema_pilot_pair"].sum()),
        "current_vendor_pair_detail_complete": int(
            pair_status["current_vendor_pair_detail_complete"].sum()
        ),
        "historical_ui_pair_groups": int(
            pair_status["historical_ui_evidence_available"].sum()
        ),
        "hyperliquid_ready_pair_groups": int(pair_status["hyperliquid_pair_ready"].sum()),
        "hyperliquid_blocked_pair_groups": int(
            (~pair_status["hyperliquid_pair_ready"]).sum()
        ),
        "planned_experiments": int(len(experiments)),
        "local_implemented_mode_experiments": int(
            experiments["local_mode_implemented"].sum()
        ),
        "ready_for_point_in_time_history_experiments": int(
            experiment_counts.get("READY_FOR_POINT_IN_TIME_HISTORY", 0)
        ),
        "mapping_blocked_experiments": int(
            experiment_counts.get("BLOCKED_HYPERLIQUID_MAPPING", 0)
        ),
        "not_applicable_vendor_mode_experiments": int(
            experiment_counts.get("NOT_APPLICABLE_VENDOR_MODE", 0)
        ),
        "pair_history_queue_rows": int(len(pair_history_queue)),
        "pair_history_ready_to_fetch": int(
            pair_history_queue["history_request_status"].eq("READY_TO_FETCH").sum()
        ),
        "asset_interval_fetch_requests": int(len(asset_fetch_queue)),
        "vendor_status_counts": {str(k): int(v) for k, v in vendor_counts.items()},
        "experiment_status_counts": {
            str(k): int(v) for k, v in experiment_counts.items()
        },
        "discovery_prefilter_applied": False,
        "canonical_replay_leverage": 1.0,
        "vendor_parity_claimed": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "input_hashes": input_hashes,
        "input_snapshots": {
            name: _relative(path, root) for name, path in input_snapshots.items()
        },
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _build_pair_status(
    queue: pd.DataFrame,
    *,
    mapping: pd.DataFrame,
    frozen_progress: pd.DataFrame,
    pilot: dict[str, Any],
    pilot_endpoints: pd.DataFrame,
    pilot_coverage: pd.DataFrame,
    handoff_id: str,
    refresh_id: str,
    evidence_path: str,
) -> pd.DataFrame:
    mapping_lookup = {
        _text(row.pair_group_key): row for row in mapping.itertuples()
    }
    frozen_lookup = {
        _text(row.pair_group_key): row for row in frozen_progress.itertuples()
    }
    pilot_pair = _text(pilot.get("pair_group_key"))
    pilot_endpoint_complete = bool(
        pilot.get("pilot_complete", False)
        and len(pilot_endpoints) > 0
        and pilot_endpoints["status"].astype(str).eq("COMPLETED").all()
    )
    missing_coverage = sorted(
        pilot_coverage.loc[
            pilot_coverage["status"].astype(str).ne("FOUND"), "field_group"
        ].astype(str)
    )
    rows: list[dict[str, object]] = []
    for queue_row in queue.itertuples():
        pair_key = _text(queue_row.pair_group_key)
        mapping_row = mapping_lookup.get(pair_key)
        if mapping_row is None:
            raise ValueError(f"Current pair has no Hyperliquid mapping row: {pair_key}")
        frozen_row = frozen_lookup.get(pair_key)
        historical_available = bool(
            frozen_row is not None
            and _text(getattr(frozen_row, "capture_status", "")) == "COMPLETE"
        )
        is_pilot = bool(pair_key == pilot_pair and pilot_endpoint_complete)
        if is_pilot:
            vendor_status = "API_ANALYTICS_COMPLETE_DASHBOARD_FIELDS_MISSING"
            vendor_blockers = [f"missing_{field}" for field in missing_coverage]
            vendor_blockers.append("current_dashboard_ui_capture_missing")
        elif historical_available:
            vendor_status = "HISTORICAL_UI_EVIDENCE_REQUIRES_CURRENT_REFRESH"
            vendor_blockers = ["current_dashboard_pair_detail_refresh_missing"]
        else:
            vendor_status = "NOT_CAPTURED_CURRENT_REFRESH"
            vendor_blockers = [
                "current_dashboard_pair_detail_missing",
                "browser_content_read_timeout",
            ]
        hyperliquid_ready = _truthy(mapping_row.hyperliquid_pair_ready)
        mapping_blocker = _text(mapping_row.hyperliquid_mapping_blocker)
        local_status = (
            "READY_FOR_POINT_IN_TIME_HISTORY"
            if hyperliquid_ready
            else "BLOCKED_HYPERLIQUID_MAPPING"
        )
        local_blocker = "" if hyperliquid_ready else mapping_blocker
        if not hyperliquid_ready and not local_blocker:
            local_blocker = "hyperliquid_mapping_blocker_missing"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "handoff_id": handoff_id,
                "refresh_id": refresh_id,
                "queue_position": int(queue_row.queue_position),
                "pair_group_key": pair_key,
                "pair": _text(queue_row.pair),
                "wizard_exchange": _text(queue_row.wizard_exchange),
                "timeframe": _text(queue_row.timeframe),
                "asset_a": _text(queue_row.asset_a),
                "asset_b": _text(queue_row.asset_b),
                "api_pair_ids": _text(queue_row.api_pair_ids),
                "api_exact_modes_observed": _text(queue_row.api_exact_modes_observed),
                "api_observed_orientations": _text(
                    queue_row.api_observed_orientations
                ),
                "api_captured_at": _text(queue_row.api_captured_at),
                "api_sharpe_min": getattr(queue_row, "api_sharpe_min", ""),
                "api_sharpe_max": getattr(queue_row, "api_sharpe_max", ""),
                "api_returns_total_min": getattr(
                    queue_row, "api_returns_total_min", ""
                ),
                "api_returns_total_max": getattr(
                    queue_row, "api_returns_total_max", ""
                ),
                "historical_ui_evidence_available": historical_available,
                "historical_ui_capture_status": _text(
                    getattr(frozen_row, "capture_status", "")
                    if frozen_row is not None
                    else ""
                ),
                "historical_ui_captured_cells": int(
                    getattr(frozen_row, "captured_planned_cells", 0) or 0
                    if frozen_row is not None
                    else 0
                ),
                "historical_ui_evidence_paths": _text(
                    getattr(frozen_row, "pair_detail_evidence_paths", "")
                    if frozen_row is not None
                    else ""
                ),
                "api_schema_pilot_pair": is_pilot,
                "api_schema_pilot_id": _text(pilot.get("pilot_id")) if is_pilot else "",
                "api_endpoints_complete": int(pilot.get("completed_endpoints", 0) or 0)
                if is_pilot
                else 0,
                "api_fields_observed": int(pilot.get("observed_fields", 0) or 0)
                if is_pilot
                else 0,
                "api_coverage_passes": int(pilot.get("coverage_passes", 0) or 0)
                if is_pilot
                else 0,
                "api_coverage_checks": int(pilot.get("coverage_checks", 0) or 0)
                if is_pilot
                else 0,
                "api_ecm_fields_found": bool(pilot.get("ecm_fields_found", False))
                if is_pilot
                else False,
                "api_dashboard_pair_detail_complete": bool(
                    pilot.get("dashboard_pair_detail_complete", False)
                )
                if is_pilot
                else False,
                "vendor_pair_detail_status": vendor_status,
                "current_vendor_pair_detail_complete": False,
                "vendor_pair_detail_blocker": ";".join(dict.fromkeys(vendor_blockers)),
                "required_exact_modes": ";".join(EXACT_MODES),
                "required_orientations": ";".join(ORIENTATIONS),
                "hyperliquid_market_a": _text(mapping_row.hyperliquid_market_a),
                "hyperliquid_market_b": _text(mapping_row.hyperliquid_market_b),
                "hyperliquid_pair_ready": hyperliquid_ready,
                "hyperliquid_pair_max_leverage": getattr(
                    mapping_row, "hyperliquid_pair_max_leverage", ""
                ),
                "hyperliquid_only_isolated": _truthy(
                    getattr(mapping_row, "hyperliquid_only_isolated", False)
                ),
                "hyperliquid_inventory_checked_at": _text(
                    mapping_row.hyperliquid_inventory_checked_at
                ),
                "hyperliquid_mapping_blocker": mapping_blocker,
                "local_research_status": local_status,
                "local_research_blocker": local_blocker,
                "local_settings_authority": "LOCAL_STANDARDIZED_MATH_V2_NOT_WIZARD_PARITY",
                "canonical_replay_status": "NOT_RUN",
                "acceptance_status": (
                    "BLOCKED_PENDING_LOCAL_1X_REPLAY"
                    if hyperliquid_ready
                    else "BLOCKED_HYPERLIQUID_MAPPING"
                ),
                "discovery_prefilter_applied": False,
                "vendor_parity_claimed": False,
                "promotion_authority": False,
                "live_trading_authorized": False,
                "evidence_path": evidence_path,
            }
        )
    return pd.DataFrame(rows).sort_values("queue_position", kind="mergesort")


def _build_experiment_matrix(
    pair_status: pd.DataFrame,
    *,
    source_accounting: pd.DataFrame,
    handoff_id: str,
    refresh_id: str,
) -> pd.DataFrame:
    observed_modes = {
        key: set(
            source_accounting.loc[
                source_accounting["pair_group_key"].astype(str).eq(key),
                "api_exact_mode",
            ].astype(str)
        )
        for key in pair_status["pair_group_key"].astype(str)
    }
    rows: list[dict[str, object]] = []
    for pair in pair_status.itertuples():
        pair_key = _text(pair.pair_group_key)
        for mode in EXACT_MODES:
            local_implemented = mode in PAIR_PAGE_IMPLEMENTED_MODES
            for orientation in ORIENTATIONS:
                if not _truthy(pair.hyperliquid_pair_ready):
                    status = "BLOCKED_HYPERLIQUID_MAPPING"
                    blocker = _text(pair.hyperliquid_mapping_blocker)
                elif not local_implemented:
                    status = "NOT_APPLICABLE_VENDOR_MODE"
                    blocker = (
                        "ou_optimal_is_scanner_boolean_overlay_not_pair_page_mode"
                    )
                else:
                    status = "READY_FOR_POINT_IN_TIME_HISTORY"
                    blocker = ""
                experiment_id = "cwexp_" + sha256(
                    f"{refresh_id}|{pair_key}|{mode}|{orientation}".encode("utf-8")
                ).hexdigest()[:20]
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "handoff_id": handoff_id,
                        "refresh_id": refresh_id,
                        "experiment_id": experiment_id,
                        "pair_group_key": pair_key,
                        "pair": _text(pair.pair),
                        "wizard_exchange": _text(pair.wizard_exchange),
                        "timeframe": _text(pair.timeframe),
                        "asset_a": _text(pair.asset_a),
                        "asset_b": _text(pair.asset_b),
                        "exact_mode": mode,
                        "orientation": orientation,
                        "experiment_kind": (
                            "exact_mode_replay"
                            if local_implemented
                            else "scanner_overlay_accounting"
                        ),
                        "pair_page_mode_available": local_implemented,
                        "scanner_overlay_name": "ou_optimal" if not local_implemented else "",
                        "mode_observed_in_current_scanner": mode
                        in observed_modes.get(pair_key, set()),
                        "local_mode_implemented": local_implemented,
                        "experiment_status": status,
                        "experiment_blocker": blocker,
                        "canonical_replay_leverage": 1.0,
                        "local_settings_authority": _text(
                            pair.local_settings_authority
                        ),
                        "vendor_pair_detail_status": _text(
                            pair.vendor_pair_detail_status
                        ),
                        "vendor_parity_claimed": False,
                        "discovery_prefilter_applied": False,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                        "evidence_path": _text(pair.evidence_path),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["pair_group_key", "exact_mode", "orientation"], kind="mergesort"
    )


def _build_pair_history_queue(
    pair_status: pd.DataFrame,
    *,
    experiments: pd.DataFrame,
    handoff_id: str,
    refresh_id: str,
) -> pd.DataFrame:
    ready_counts = (
        experiments["experiment_status"]
        .eq("READY_FOR_POINT_IN_TIME_HISTORY")
        .groupby(experiments["pair_group_key"])
        .sum()
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for pair in pair_status.itertuples():
        pair_key = _text(pair.pair_group_key)
        interval = TIMEFRAME_INTERVALS.get(_text(pair.timeframe).lower(), "")
        cutoff = _earliest_timestamp(_text(pair.api_captured_at))
        blockers: list[str] = []
        if not _truthy(pair.hyperliquid_pair_ready):
            blockers.append(_text(pair.hyperliquid_mapping_blocker))
        if not interval:
            blockers.append("unsupported_wizard_timeframe")
        if cutoff is None:
            blockers.append("current_scanner_capture_timestamp_missing")
        ready_experiments = int(ready_counts.get(pair_key, 0))
        if ready_experiments == 0 and _truthy(pair.hyperliquid_pair_ready):
            blockers.append("no_local_mode_experiments_ready")
        blockers = [value for value in dict.fromkeys(blockers) if value]
        request_id = "cwhistory_" + sha256(
            f"{refresh_id}|{pair_key}|{interval}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "handoff_id": handoff_id,
                "refresh_id": refresh_id,
                "history_request_id": request_id,
                "pair_group_key": pair_key,
                "pair": _text(pair.pair),
                "wizard_exchange": _text(pair.wizard_exchange),
                "timeframe": _text(pair.timeframe),
                "hyperliquid_interval": interval,
                "asset_a": _text(pair.asset_a),
                "asset_b": _text(pair.asset_b),
                "hyperliquid_market_a": _text(pair.hyperliquid_market_a),
                "hyperliquid_market_b": _text(pair.hyperliquid_market_b),
                "scanner_cutoff_at": cutoff.isoformat() if cutoff else "",
                "history_days_requested": HISTORY_DAYS.get(interval, 0),
                "minimum_history_rows": MINIMUM_HISTORY_ROWS.get(interval, 0),
                "ready_mode_orientation_experiments": ready_experiments,
                "history_request_status": "READY_TO_FETCH" if not blockers else "BLOCKED",
                "history_request_blocker": ";".join(blockers),
                "canonical_replay_leverage": 1.0,
                "promotion_authority": False,
                "live_trading_authorized": False,
                "evidence_path": _text(pair.evidence_path),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["history_request_status", "pair_group_key"],
        ascending=[False, True],
        kind="mergesort",
    )


def _build_asset_fetch_queue(
    pair_history_queue: pd.DataFrame,
    *,
    handoff_id: str,
    refresh_id: str,
) -> pd.DataFrame:
    ready = pair_history_queue[
        pair_history_queue["history_request_status"].eq("READY_TO_FETCH")
    ]
    requests: dict[tuple[str, str], dict[str, object]] = {}
    for pair in ready.itertuples():
        for asset in (_text(pair.asset_a), _text(pair.asset_b)):
            key = (asset, _text(pair.hyperliquid_interval))
            record = requests.setdefault(
                key,
                {
                    "cutoffs": [],
                    "pair_keys": [],
                    "history_request_ids": [],
                    "history_days_requested": int(pair.history_days_requested),
                    "minimum_history_rows": int(pair.minimum_history_rows),
                },
            )
            record["cutoffs"].append(_text(pair.scanner_cutoff_at))
            record["pair_keys"].append(_text(pair.pair_group_key))
            record["history_request_ids"].append(_text(pair.history_request_id))
    rows: list[dict[str, object]] = []
    for (asset, interval), record in sorted(requests.items()):
        cutoffs = sorted(value for value in record["cutoffs"] if value)
        request_id = "cwasset_" + sha256(
            f"{refresh_id}|{asset}|{interval}|{cutoffs[-1]}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "handoff_id": handoff_id,
                "refresh_id": refresh_id,
                "asset_fetch_request_id": request_id,
                "asset": asset,
                "hyperliquid_interval": interval,
                "fetch_end_at": cutoffs[-1],
                "earliest_pair_cutoff_at": cutoffs[0],
                "history_days_requested": record["history_days_requested"],
                "minimum_history_rows": record["minimum_history_rows"],
                "pair_group_count": len(set(record["pair_keys"])),
                "pair_group_keys": ";".join(sorted(set(record["pair_keys"]))),
                "history_request_ids": ";".join(
                    sorted(set(record["history_request_ids"]))
                ),
                "fetch_status": "READY_TO_FETCH",
                "fetch_blocker": "",
                "network_requests_deduplicated_only": True,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _build_validation(
    *,
    queue: pd.DataFrame,
    mapping: pd.DataFrame,
    pair_status: pd.DataFrame,
    experiments: pd.DataFrame,
    pair_history_queue: pd.DataFrame,
    pilot: dict[str, Any],
) -> pd.DataFrame:
    expected_pairs = len(queue)
    expected_experiments = expected_pairs * len(EXACT_MODES) * len(ORIENTATIONS)
    blocked_vendor = pair_status[~pair_status["current_vendor_pair_detail_complete"]]
    blocked_mapping = pair_status[~pair_status["hyperliquid_pair_ready"]]
    pilot_pair = _text(pilot.get("pair_group_key"))
    pilot_pair_on_current_board = bool(
        pilot_pair
        and queue["pair_group_key"].astype(str).eq(pilot_pair).any()
        and pilot.get("pilot_complete", False)
    )
    checks = [
        (
            "every_current_pair_group_accounted",
            len(pair_status) == expected_pairs
            and pair_status["pair_group_key"].nunique() == expected_pairs,
            f"accounted={len(pair_status)} expected={expected_pairs}",
        ),
        (
            "every_pair_has_hyperliquid_mapping",
            len(mapping) >= expected_pairs
            and pair_status["hyperliquid_pair_ready"].notna().all(),
            f"mapping_rows={len(mapping)} current_pairs={expected_pairs}",
        ),
        (
            "every_vendor_gap_has_blocker",
            blocked_vendor["vendor_pair_detail_blocker"].astype(str).str.len().gt(0).all(),
            f"vendor_incomplete={len(blocked_vendor)}",
        ),
        (
            "every_mapping_gap_has_blocker",
            blocked_mapping["local_research_blocker"].astype(str).str.len().gt(0).all(),
            f"mapping_blocked={len(blocked_mapping)}",
        ),
        (
            "all_exact_modes_and_orientations_planned",
            len(experiments) == expected_experiments
            and experiments["experiment_id"].nunique() == expected_experiments
            and experiments.groupby("pair_group_key").size().eq(
                len(EXACT_MODES) * len(ORIENTATIONS)
            ).all(),
            f"experiments={len(experiments)} expected={expected_experiments}",
        ),
        (
            "history_queue_accounts_every_pair",
            len(pair_history_queue) == expected_pairs
            and pair_history_queue["pair_group_key"].nunique() == expected_pairs,
            f"history_rows={len(pair_history_queue)} expected={expected_pairs}",
        ),
        (
            "api_pilot_overlay_applied_when_pair_present",
            int(pair_status["api_schema_pilot_pair"].sum())
            == (1 if pilot_pair_on_current_board else 0),
            (
                f"pilot_pairs={int(pair_status['api_schema_pilot_pair'].sum())} "
                f"pilot_pair_on_current_board={pilot_pair_on_current_board}"
            ),
        ),
        (
            "no_discovery_prefilter",
            not pair_status["discovery_prefilter_applied"].astype(bool).any()
            and not experiments["discovery_prefilter_applied"].astype(bool).any(),
            "all current rows retained",
        ),
        (
            "no_vendor_parity_claim",
            not pair_status["vendor_parity_claimed"].astype(bool).any()
            and not experiments["vendor_parity_claimed"].astype(bool).any(),
            "local research remains distinct from Wizard parity",
        ),
        (
            "no_promotion_or_live_authority",
            not pair_status["promotion_authority"].astype(bool).any()
            and not pair_status["live_trading_authorized"].astype(bool).any()
            and not experiments["promotion_authority"].astype(bool).any()
            and not experiments["live_trading_authorized"].astype(bool).any(),
            "promotion=false live=false",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
            for check, passed, evidence in checks
        ]
    )


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Handoff",
            "",
            f"- Handoff: `{summary['handoff_id']}`",
            f"- Current refresh: `{summary['refresh_id']}`",
            f"- Pair groups accounted: {summary['pair_groups_accounted']} / {summary['pair_groups']}",
            f"- API schema-pilot pairs: {summary['api_schema_pilot_pairs']}",
            f"- Current complete vendor pair details: {summary['current_vendor_pair_detail_complete']}",
            f"- Historical UI evidence pairs: {summary['historical_ui_pair_groups']}",
            f"- Hyperliquid-ready / blocked pairs: {summary['hyperliquid_ready_pair_groups']} / {summary['hyperliquid_blocked_pair_groups']}",
            f"- Exact-mode/orientation experiments: {summary['planned_experiments']}",
            f"- Ready for point-in-time history: {summary['ready_for_point_in_time_history_experiments']}",
            f"- Pair histories ready to fetch: {summary['pair_history_ready_to_fetch']}",
            f"- Deduplicated asset/interval requests: {summary['asset_interval_fetch_requests']}",
            "- Discovery prefilter applied: `false`",
            "- Canonical replay leverage: `1x`",
            "- Vendor parity claimed: `false`",
            "- Promotion authority: `false`",
            "- Live trading authorized: `false`",
            "",
            "Every current pair remains visible. Missing dashboard fields, unsupported local modes, and unavailable Hyperliquid legs are explicit blockers rather than silent exclusions.",
            "",
        ]
    )


def _earliest_timestamp(value: str) -> datetime | None:
    timestamps: list[datetime] = []
    for item in value.split(";"):
        text = item.strip()
        if not text:
            continue
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            continue
        timestamps.append(_as_utc(parsed))
    return min(timestamps) if timestamps else None


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Required CSV is empty: {path}")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON object is invalid: {path}")
    return payload


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "ready", "pass"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
