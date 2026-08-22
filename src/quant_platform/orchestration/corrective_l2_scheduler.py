"""Cadenced, read-only Hyperliquid depth collection for registered hypotheses."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.hyperliquid import (
    fetch_hyperliquid_funding_history,
    normalize_hyperliquid_funding_history,
    refresh_hyperliquid_execution_cost_snapshot,
)
from quant_platform.orchestration.corrective_data_evidence import (
    build_cost_collection_status,
    build_l2_capture_candidate_set,
    build_pair_cost_stress_surfaces,
    validate_pair_cost_bundle_artifacts,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    ensure_runtime_temp_directory,
    ensure_scheduler_log_directory,
    scheduler_contract,
    scheduler_launch_agent_plist,
    scheduler_log_directory,
    scheduler_python_path,
    scheduler_run_identity,
    write_immutable_json,
    write_launch_agent_plist,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceLockBusy,
    GovernedEvidenceMaintenanceActive,
    acquire_scheduler_lock,
    governed_evidence_write_lock,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    supervise_scheduler_run,
)
from quant_platform.orchestration.current_wizard_hyperliquid_costs import (
    materialize_current_wizard_hyperliquid_cost_evidence,
)
from quant_platform.orchestration.current_wizard_hyperliquid_evidence_command_center import (
    build_current_wizard_hyperliquid_evidence_command_center,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    build_exhaustive_wizard_hyperliquid_mapping_refresh,
)
from quant_platform.runtime_types import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_l2_scheduler.v1"
READINESS_REFRESH_SCHEMA_VERSION = "thewiz.corrective_l2_readiness_refresh.v1"
L2_ACCEPTANCE_LATTICE_SCHEMA_VERSION = "thewiz.l2_acceptance_lattice.v1"
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-l2-cadence"
# Leave enough cadence headroom for capture, immutable-bundle publication, and
# occasional API/runtime drift while retaining 12 observations in two hours.
DEFAULT_INTERVAL_SECONDS = 5 * 60
DEFAULT_NOTIONAL_USD = 1_000.0
DEFAULT_MINIMUM_SAMPLES = 12
DEFAULT_WINDOW_HOURS = 2.0
DEFAULT_FUNDING_HISTORY_DAYS = 500
WIZARD_DAILY_CREDIT_RESET_UTC = "00:00"
MAPPING_REFRESH_INTERVAL_HOURS = 6.0
MAPPING_HARD_STALE_HOURS = 24.0
MAPPING_REFRESH_RUNTIME_TOLERANCE_MINUTES = 15
POST_WINDOW_LOCK_NAMES = (
    ".corrective_daily.lock",
    ".corrective_l2_capture.lock",
    ".corrective_wizard_proof.lock",
    ".corrective_registered_rerun.lock",
)
# The registered gate owns prospective contract rollover. Its active contract is
# therefore a controlled output, while these paths are the exogenous inputs that
# must remain unchanged throughout one local readiness refresh.
POST_WINDOW_SOURCE_PATHS = (
    "reports/active/corrective_l2_capture_status.json",
    "reports/active/corrective_l2_capture_candidates.csv",
    "reports/active/hyperliquid_pair_cost_bundle_pointer.json",
    "reports/active/current_hypothesis_batch.csv",
    "reports/active/current_wizard_hyperliquid_failure_attribution.csv",
    "reports/active/current_wizard_hyperliquid_failure_attribution_manifest.json",
    "reports/active/current_wizard_hyperliquid_experiment_matrix.csv",
    "reports/active/current_wizard_hyperliquid_chain_validation_manifest.json",
    "reports/active/current_wizard_hyperliquid_history_manifest.json",
    "reports/active/walkforward_near_miss_queue.csv",
    "reports/active/hypothesis_ledger_audit.csv",
    "reports/active/acceptance_policy_receipt.json",
    "reports/active/holdout_policy_receipt.json",
    "reports/active/final_1x_survivor_receipt.json",
    "reports/active/wizard_mode_parity.csv",
    "reports/active/corrective_wizard_proof_scheduler_status.json",
    "reports/active/corrective_wizard_next_capture_manifest.json",
    "reports/active/wizard_reset_readiness.json",
    "reports/active/registered_stage5_protocol.json",
    "config/wizard_discovery_policy.json",
    "data/processed/hyperliquid_market_context.csv",
    "data/research/hypothesis_ledger.jsonl",
)
POST_WINDOW_APPEND_ONLY_PATHS = frozenset(
    {
        "data/research/hypothesis_ledger.jsonl",
    }
)


def _normalized_blockers(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    return list(
        dict.fromkeys(str(item).strip() for item in values if str(item).strip())
    )


def _build_l2_acceptance_summary(
    capture_summary: dict[str, Any],
    *,
    post_window_summary: dict[str, Any] | None = None,
    post_window_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive one monotone, zero-trade-authority L2 completion state."""

    summary = dict(capture_summary)
    readiness = dict(post_window_summary or {})
    validation = dict(post_window_validation or {})
    capture_blockers = _normalized_blockers(
        summary.get("capture_blockers", summary.get("blockers", []))
    )
    collector = summary.get("collector_summary", {})
    if not isinstance(collector, dict):
        collector = {}
        capture_blockers.append("l2_collector_summary_invalid")

    candidate_pairs = _safe_int(summary.get("eligible_pairs"))
    collected_pairs = _safe_int(collector.get("pairs"))
    collected = bool(
        not capture_blockers and candidate_pairs > 0 and collected_pairs > 0
    )

    strict_eligible = _safe_int(summary.get("strict_pair_cost_eligible"))
    strict_ready = _safe_int(summary.get("strict_pair_cost_ready"))
    strict_cost_accepted = bool(
        strict_eligible > 0
        and strict_ready == strict_eligible
        and str(summary.get("strict_pair_cost_acceptance_status", "")) == "PASS"
    )
    registered_candidates = _safe_int(
        summary.get("registered_contract_candidates")
    )

    post_window_present = bool(
        post_window_summary is not None and post_window_validation is not None
    )
    readiness_status = str(readiness.get("status", "NOT_RUN")).strip()
    readiness_blockers = _normalized_blockers(readiness.get("blockers", []))
    validation_blockers = _normalized_blockers(validation.get("blockers", []))
    capture_receipt_id = str(summary.get("receipt_id", "")).strip()
    source_ids_match = bool(
        capture_receipt_id
        and str(readiness.get("source_l2_receipt_id", "")).strip()
        == capture_receipt_id
        and str(validation.get("source_l2_receipt_id", "")).strip()
        == capture_receipt_id
    )
    readiness_business_valid = readiness_status in {
        "WAITING_STRICT_L2",
        "NOT_APPLICABLE_NO_REGISTERED_COHORT",
        "PASS_LOCAL_READINESS_REFRESH",
    }
    post_window_evidence_valid = bool(
        post_window_present
        and str(validation.get("status", "")) == "PASS"
        and not validation_blockers
        and source_ids_match
        and readiness_business_valid
    )
    validated = bool(collected and post_window_evidence_valid)

    readiness_counts_match = bool(
        _safe_int(readiness.get("eligible_pairs"), default=-1) == strict_eligible
        and _safe_int(readiness.get("ready_pairs"), default=-1) == strict_ready
        and _safe_int(readiness.get("collecting_pairs"), default=-1) == 0
    )
    cohort_matches = bool(
        registered_candidates > 0
        and _safe_int(
            readiness.get("registered_contract_candidates"), default=-1
        )
        == registered_candidates
    )
    accepted = bool(
        collected
        and validated
        and strict_cost_accepted
        and readiness_counts_match
        and cohort_matches
        and readiness.get("refresh_executed") is True
    )
    gate_status = str(readiness.get("registered_gate_status", "")).strip()
    handoff_status = str(readiness.get("stage4_handoff_status", "")).strip()
    scheduler_authorized = bool(
        accepted
        and readiness_status == "PASS_LOCAL_READINESS_REFRESH"
        and readiness.get("registered_gate_refresh_executed") is True
        and readiness.get("stage4_handoff_refresh_executed") is True
        and gate_status.startswith("PASS")
        and handoff_status.startswith("PASS")
        and readiness.get("stage4_handoff_validation_status") == "PASS"
        and not readiness_blockers
        and _authority_is_zero(summary)
        and _authority_is_zero(readiness)
    )

    lattice_blockers = list(capture_blockers)
    if not collected and not capture_blockers:
        lattice_blockers.append("l2_collection_not_complete")
    if not strict_cost_accepted:
        lattice_blockers.append("l2_strict_cost_acceptance_not_pass")
    if not post_window_present:
        lattice_blockers.append("l2_post_window_validation_pending")
    else:
        lattice_blockers.extend(
            f"l2_post_window_validation:{blocker}"
            for blocker in validation_blockers
        )
        if not source_ids_match:
            lattice_blockers.append("l2_post_window_source_receipt_mismatch")
        if not readiness_business_valid:
            lattice_blockers.append(
                f"l2_post_window_status_not_acceptable:{readiness_status or 'MISSING'}"
            )
        if not readiness_counts_match:
            lattice_blockers.append("l2_post_window_cost_counts_mismatch")
        if not cohort_matches:
            lattice_blockers.append("l2_post_window_cohort_mismatch")
        lattice_blockers.extend(
            f"l2_post_window:{blocker}" for blocker in readiness_blockers
        )
        if accepted and not scheduler_authorized:
            lattice_blockers.append("l2_scheduler_completion_not_authorized")

    complete = bool(collected and validated and accepted and scheduler_authorized)
    collection_state = "COLLECTED" if collected else "BLOCKED"
    if not post_window_present:
        validation_state = "PENDING"
    else:
        validation_state = "VALIDATED" if validated else "BLOCKED"
    if accepted:
        acceptance_state = "ACCEPTED"
    elif not post_window_present and strict_cost_accepted:
        acceptance_state = "PENDING"
    else:
        acceptance_state = "BLOCKED"
    if scheduler_authorized:
        authorization_state = "AUTHORIZED"
    elif not post_window_present:
        authorization_state = "PENDING"
    else:
        authorization_state = "BLOCKED"
    highest_state = "BLOCKED"
    if collected:
        highest_state = "COLLECTED"
    if validated:
        highest_state = "VALIDATED"
    if accepted:
        highest_state = "ACCEPTED"
    if scheduler_authorized:
        highest_state = "AUTHORIZED"

    summary.update(
        {
            "status": "PASS" if complete else "BLOCKED",
            "blockers": list(dict.fromkeys(lattice_blockers)),
            "capture_status": "PASS" if collected else "BLOCKED",
            "capture_blockers": list(dict.fromkeys(capture_blockers)),
            "l2_acceptance_lattice_schema_version": (
                L2_ACCEPTANCE_LATTICE_SCHEMA_VERSION
            ),
            "l2_acceptance_lattice": [
                "COLLECTED",
                "VALIDATED",
                "ACCEPTED",
                "AUTHORIZED",
            ],
            "l2_collection_state": collection_state,
            "l2_validation_state": validation_state,
            "l2_acceptance_state": acceptance_state,
            "l2_authorization_state": authorization_state,
            "l2_highest_state": highest_state,
            "l2_acceptance_lattice_complete": complete,
            "l2_scheduler_completion_authorized": scheduler_authorized,
            "l2_terminal_slot_credit_eligible": complete,
            "post_window_readiness_status": readiness_status,
            "post_window_readiness_validation_status": str(
                validation.get("status", "NOT_RUN")
            ),
            "post_window_readiness": readiness,
            "post_window_readiness_validation": validation,
        }
    )
    return summary


def _build_supervised_l2_result(
    *,
    capture: CommandResult,
    readiness: CommandResult,
    readiness_validation: dict[str, Any],
) -> dict[str, Any]:
    summary = _build_l2_acceptance_summary(
        capture.summary,
        post_window_summary=readiness.summary,
        post_window_validation=readiness_validation,
    )
    paths = {key: str(value) for key, value in capture.paths.items()}
    paths.update(
        {
            f"post_window_readiness_{key}": str(value)
            for key, value in readiness.paths.items()
        }
    )
    return {"summary": summary, "paths": paths}


def run_corrective_l2_capture(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    collector: Callable[..., CommandResult] = refresh_hyperliquid_execution_cost_snapshot,
    funding_materializer: Callable[
        ..., CommandResult
    ] = materialize_current_wizard_hyperliquid_cost_evidence,
    supplemental_funding_materializer: Callable[..., CommandResult] | None = None,
    testnet_inventory_refresher: Callable[..., pd.DataFrame] | None = None,
    mapping_refresher: Callable[..., CommandResult] = (
        build_exhaustive_wizard_hyperliquid_mapping_refresh
    ),
    require_launchd_provenance: bool = False,
) -> CommandResult:
    """Capture one public L2 observation for each eligible registered pair."""

    captured_at = _as_utc(now)
    runtime_identity = scheduler_run_identity(
        root,
        contract=scheduler_contract("hyperliquid_l2"),
        require_launchd=require_launchd_provenance,
    )
    active = root / "reports" / "active"
    receipts = active / "l2_capture_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = active / ".corrective_l2_capture.lock"
    receipt_path = receipts / captured_at.strftime("%Y-%m-%d_%H%M%S.json")
    candidate_path = active / "corrective_l2_capture_candidates.csv"
    candidate_result = {
        "registered_hypotheses": 0,
        "registered_contract_candidates": 0,
        "candidate_pairs": 0,
        "eligible_pairs": 0,
        "stage_two_candidate_pairs": 0,
    }
    eligible_pairs = 0
    blockers: list[str] = []
    operational_warnings: list[str] = []
    capture_summary: dict[str, Any] = {}
    funding_summary: dict[str, Any] = {}
    funding_paths: dict[str, Path] = {}
    supplemental_funding_summary: dict[str, Any] = {}
    supplemental_funding_paths: dict[str, Path] = {}
    funding_refresh_status = "NOT_RUN_NO_ELIGIBLE_CANDIDATES"
    missing_funding_assets: list[str] = []
    remaining_missing_funding_assets: list[str] = []
    evidence_evaluated_at = captured_at
    collection: dict[str, Any] = {
        "ready_assets": 0,
        "collecting_assets": 0,
        "status_path": active / "hyperliquid_cost_collection_status.csv",
    }
    pair_cost_defaults: dict[str, Any] = {
        "pairs": 0,
        "strict_ready_pairs": 0,
        "eligible_pairs": 0,
        "acceptance_status": "BLOCKED",
        "stage_two_eligible_pairs": 0,
        "stage_two_strict_ready_pairs": 0,
        "stage_two_acceptance_status": "BLOCKED",
        "models": root / "data" / "processed" / "hyperliquid_pair_cost_models.csv",
        "stress": active / "hyperliquid_cost_stress.csv",
        "model_snapshot": active / "missing_pair_cost_model_snapshot.csv",
        "stress_snapshot": active / "missing_pair_cost_stress_snapshot.csv",
        "bundle_manifest": active / "missing_pair_cost_bundle.json",
        "bundle_pointer": active / "missing_pair_cost_bundle_pointer.json",
        "bundle_pointer_snapshot": active / "missing_pair_cost_bundle_pointer_snapshot.json",
        "bundle_id": "",
    }
    pair_costs = dict(pair_cost_defaults)
    transition: dict[str, Any] = {
        "path": active / "corrective_l2_post_window_transition.csv",
        "ready_pairs": 0,
        "collecting_pairs": 0,
    }
    mapping_maintenance: dict[str, Any] = {
        "configured": False,
        "due": False,
        "hard_stale_before": False,
        "status": "NOT_CONFIGURED",
        "age_hours_before": None,
        "age_hours_after": None,
        "inventory_refreshed": False,
        "mapping_refresh_id": "",
        "ready_pair_groups": 0,
        "blocked_pair_groups": 0,
        "blocker": "",
        "paths": {},
    }
    command_center_summary: dict[str, Any] = {}
    command_center_paths: dict[str, Path] = {}
    lock_acquired = False
    try:
        acquire_scheduler_lock(lock_path, now=captured_at, timeout_seconds=DEFAULT_INTERVAL_SECONDS)
        lock_acquired = True
        if not runtime_identity["runtime_environment_valid"]:
            raise RuntimeError(
                ";".join(runtime_identity["runtime_environment_blockers"])
            )
        mapping_maintenance = _maintain_exhaustive_mapping(
            root=root,
            now=captured_at,
            inventory_refresher=testnet_inventory_refresher,
            mapping_refresher=mapping_refresher,
        )
        mapping_blocker = str(mapping_maintenance.get("blocker", ""))
        if mapping_blocker:
            if bool(mapping_maintenance.get("hard_stale_before", False)):
                blockers.append(mapping_blocker)
            else:
                operational_warnings.append(mapping_blocker)
        candidate_result = build_l2_capture_candidate_set(root=root)
        candidate_path = Path(candidate_result["path"])
        eligible_pairs = int(candidate_result["eligible_pairs"])
        if eligible_pairs <= 0:
            blockers.append("no_registered_hyperliquid_l2_candidates")
        else:
            capture = collector(
                root=root,
                max_pairs=eligible_pairs,
                notionals=(DEFAULT_NOTIONAL_USD,),
                captured_at=captured_at,
                candidate_path=candidate_path,
                min_samples=DEFAULT_MINIMUM_SAMPLES,
                window_hours=DEFAULT_WINDOW_HOURS,
            )
            capture_summary = dict(capture.summary)
            if int(capture_summary.get("pairs", 0)) <= 0:
                blockers.append("collector_returned_zero_pairs")
        evidence_evaluated_at = captured_at if now is not None else _as_utc(None)
        candidates = _read_csv(candidate_path)
        missing_funding_assets = _missing_candidate_funding_assets(
            root=root,
            candidates=candidates,
        )
        if eligible_pairs > 0 and missing_funding_assets:
            pair_group_keys = _eligible_current_pair_group_keys(
                candidates,
                required_assets=set(missing_funding_assets),
            )
            if pair_group_keys:
                try:
                    funding = funding_materializer(
                        root=root,
                        pair_group_keys=pair_group_keys,
                        now=evidence_evaluated_at,
                        fetch_funding=True,
                    )
                    funding_summary = dict(funding.summary)
                    funding_paths = {key: Path(path) for key, path in funding.paths.items()}
                    funding_refresh_status = "PASS"
                except Exception as exc:  # noqa: BLE001 - persist blocked evidence
                    blockers.append(f"funding_materialization_error:{safe_exception_code(exc)}")
                    funding_refresh_status = "BLOCKED"
            if funding_refresh_status != "BLOCKED":
                remaining_missing_funding_assets = _missing_candidate_funding_assets(
                    root=root,
                    candidates=candidates,
                )
                if remaining_missing_funding_assets:
                    try:
                        supplemental = (
                            supplemental_funding_materializer
                            or materialize_corrective_candidate_funding_assets
                        )(
                            root=root,
                            assets=remaining_missing_funding_assets,
                            now=evidence_evaluated_at,
                        )
                        supplemental_funding_summary = dict(supplemental.summary)
                        supplemental_funding_paths = {
                            key: Path(path) for key, path in supplemental.paths.items()
                        }
                    except Exception as exc:  # noqa: BLE001 - persist blocked evidence
                        blockers.append(
                            f"supplemental_funding_materialization_error:{safe_exception_code(exc)}"
                        )
                        funding_refresh_status = "BLOCKED"
                remaining_missing_funding_assets = _missing_candidate_funding_assets(
                    root=root,
                    candidates=candidates,
                )
                if remaining_missing_funding_assets:
                    blockers.append(
                        "candidate_funding_refresh_incomplete:"
                        + ",".join(remaining_missing_funding_assets)
                    )
                    funding_refresh_status = "BLOCKED"
                else:
                    funding_refresh_status = "PASS"
        elif eligible_pairs > 0:
            funding_refresh_status = "NOT_REQUIRED_ALL_CANDIDATE_ASSETS_COMPLETE"
        collection = build_cost_collection_status(root=root, now=evidence_evaluated_at)
        built_pair_costs = build_pair_cost_stress_surfaces(root=root, now=evidence_evaluated_at)
        pair_costs.update(built_pair_costs)
        blockers.extend(_pair_cost_result_blockers(built_pair_costs))
        for key, fallback in pair_cost_defaults.items():
            if pair_costs.get(key) is None or (
                key
                in {
                    "models",
                    "stress",
                    "model_snapshot",
                    "stress_snapshot",
                    "bundle_manifest",
                    "bundle_pointer",
                    "bundle_pointer_snapshot",
                }
                and not str(pair_costs.get(key, "")).strip()
            ):
                pair_costs[key] = fallback
        transition = build_post_window_transition(
            root=root,
            candidate_path=candidate_path,
            cost_status_path=Path(collection["status_path"]),
            captured_at=evidence_evaluated_at,
        )
        command_center_inputs = (
            active / "current_wizard_hyperliquid_cost_manifest.json",
            active / "current_wizard_hyperliquid_pair_cost_evidence.csv",
            active / "current_wizard_hyperliquid_failure_attribution.csv",
        )
        if all(path.is_file() for path in command_center_inputs):
            try:
                command_center = build_current_wizard_hyperliquid_evidence_command_center(
                    root=root, now=evidence_evaluated_at
                )
                command_center_summary = dict(command_center.summary)
                command_center_paths = {
                    key: Path(path) for key, path in command_center.paths.items()
                }
            except Exception as exc:  # noqa: BLE001 - capture remains independent
                operational_warnings.append(
                    f"evidence_command_center_refresh_error:{safe_exception_code(exc)}"
                )
    except FileExistsError as exc:
        blockers.append(safe_exception_code(exc))
    except Exception as exc:  # noqa: BLE001 - persist a blocked scheduler receipt
        blockers.append(f"l2_capture_error:{safe_exception_code(exc)}")
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        **runtime_identity,
        "captured_at_utc": captured_at.isoformat(),
        "evidence_evaluated_at_utc": evidence_evaluated_at.isoformat(),
        "status": "BLOCKED",
        "capture_blockers": list(blockers),
        "registered_hypotheses": int(candidate_result["registered_hypotheses"]),
        "registered_contract_candidates": int(
            candidate_result.get("registered_contract_candidates", 0)
        ),
        "candidate_pairs": int(candidate_result["candidate_pairs"]),
        "eligible_pairs": eligible_pairs,
        "candidate_routing_index_used": bool(candidate_result.get("routing_index_used", False)),
        "candidate_routing_source_paths": str(candidate_result.get("routing_source_paths", "")),
        "stage_two_candidate_pairs": int(candidate_result.get("stage_two_candidate_pairs", 0)),
        "collector_summary": capture_summary,
        "funding_refresh_status": funding_refresh_status,
        "funding_refresh_missing_assets_before": missing_funding_assets,
        "funding_refresh_missing_assets_after": remaining_missing_funding_assets,
        "funding_materializer_summary": funding_summary,
        "supplemental_funding_materializer_summary": supplemental_funding_summary,
        "funding_public_read_only_endpoint": True,
        "cost_ready_assets": int(collection["ready_assets"]),
        "cost_collecting_assets": int(collection["collecting_assets"]),
        "registered_pair_cost_models": int(pair_costs["pairs"]),
        "strict_pair_cost_ready": int(pair_costs["strict_ready_pairs"]),
        "strict_pair_cost_eligible": int(pair_costs["eligible_pairs"]),
        "strict_pair_cost_acceptance_status": str(pair_costs["acceptance_status"]),
        "stage_two_pair_cost_eligible": int(pair_costs.get("stage_two_eligible_pairs", 0)),
        "stage_two_pair_cost_ready": int(pair_costs.get("stage_two_strict_ready_pairs", 0)),
        "stage_two_pair_cost_acceptance_status": str(
            pair_costs.get("stage_two_acceptance_status", "BLOCKED")
        ),
        "pair_cost_bundle_id": str(pair_costs.get("bundle_id", "")),
        "pair_cost_bundle_manifest_path": _relative(Path(pair_costs["bundle_manifest"]), root),
        "pair_cost_bundle_manifest_sha256": _file_sha256(Path(pair_costs["bundle_manifest"])),
        "pair_cost_bundle_pointer_path": _relative(
            Path(pair_costs["bundle_pointer_snapshot"]), root
        ),
        "pair_cost_bundle_pointer_sha256": _file_sha256(
            Path(pair_costs["bundle_pointer_snapshot"])
        ),
        "active_pair_cost_bundle_pointer_path": _relative(Path(pair_costs["bundle_pointer"]), root),
        "active_pair_cost_bundle_pointer_sha256": _file_sha256(Path(pair_costs["bundle_pointer"])),
        "pair_cost_model_snapshot_path": _relative(Path(pair_costs["model_snapshot"]), root),
        "pair_cost_model_snapshot_sha256": _file_sha256(Path(pair_costs["model_snapshot"])),
        "post_window_ready_pairs": int(transition["ready_pairs"]),
        "post_window_collecting_pairs": int(transition["collecting_pairs"]),
        "post_window_candidate_refresh_executed": False,
        "evidence_command_center_summary": command_center_summary,
        "wizard_daily_credit_reset_utc": WIZARD_DAILY_CREDIT_RESET_UTC,
        "blockers": blockers,
        "operational_warnings": operational_warnings,
        "mapping_maintenance_configured": bool(mapping_maintenance["configured"]),
        "mapping_refresh_due": bool(mapping_maintenance["due"]),
        "mapping_hard_stale_before": bool(mapping_maintenance["hard_stale_before"]),
        "mapping_refresh_status": str(mapping_maintenance["status"]),
        "mapping_age_hours_before": mapping_maintenance["age_hours_before"],
        "mapping_age_hours_after": mapping_maintenance["age_hours_after"],
        "mapping_inventory_refreshed": bool(mapping_maintenance["inventory_refreshed"]),
        "mapping_refresh_id": str(mapping_maintenance["mapping_refresh_id"]),
        "mapping_ready_pair_groups": int(mapping_maintenance["ready_pair_groups"]),
        "mapping_blocked_pair_groups": int(mapping_maintenance["blocked_pair_groups"]),
        "mapping_refresh_blocker": str(mapping_maintenance["blocker"]),
        "mapping_refresh_public_read_only": True,
        "lock_released": not lock_path.exists(),
        "public_read_only_endpoint": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "candidate_evidence_path": _relative(candidate_path, root),
        "candidate_evidence_sha256": _file_sha256(candidate_path),
        "cost_status_path": _relative(Path(collection["status_path"]), root),
        "pair_cost_models_path": _relative(Path(pair_costs["models"]), root),
        "pair_cost_stress_path": _relative(Path(pair_costs["stress"]), root),
        "post_window_transition_path": _relative(Path(transition["path"]), root),
    }
    receipt = _build_l2_acceptance_summary(receipt)
    receipt["receipt_id"] = (
        "l2receipt_"
        + sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    _atomic_json(receipt, receipt_path)
    latest_path = active / "corrective_l2_capture_status.json"
    _atomic_json({**receipt, "receipt_path": _relative(receipt_path, root)}, latest_path)
    return CommandResult(
        paths={
            "capture_receipt": receipt_path,
            "latest_status": latest_path,
            "candidate_set": candidate_path,
            "cost_collection_status": Path(collection["status_path"]),
            "pair_cost_models": Path(pair_costs["models"]),
            "pair_cost_stress": Path(pair_costs["stress"]),
            "pair_cost_bundle_manifest": Path(pair_costs["bundle_manifest"]),
            "pair_cost_bundle_pointer": Path(pair_costs["bundle_pointer"]),
            "pair_cost_bundle_pointer_snapshot": Path(pair_costs["bundle_pointer_snapshot"]),
            "pair_cost_model_snapshot": Path(pair_costs["model_snapshot"]),
            "pair_cost_stress_snapshot": Path(pair_costs["stress_snapshot"]),
            "post_window_transition": Path(transition["path"]),
            **{f"funding_{key}": path for key, path in funding_paths.items()},
            **{
                f"supplemental_funding_{key}": path
                for key, path in supplemental_funding_paths.items()
            },
            **{
                f"mapping_{key}": Path(path)
                for key, path in dict(mapping_maintenance["paths"]).items()
            },
            **{f"command_center_{key}": path for key, path in command_center_paths.items()},
        },
        summary=receipt,
    )


def validate_l2_capture_receipt(
    *, root: Path = ROOT, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the active L2 status against its immutable capture receipt."""

    active_path = root / "reports" / "active" / "corrective_l2_capture_status.json"
    active = receipt or _read_json(active_path)
    blockers: list[str] = []
    receipt_id = str(active.get("receipt_id", "")).strip()
    receipt_relative = str(active.get("receipt_path", "")).strip()
    receipt_path = _safe_root_artifact(root, receipt_relative)
    immutable = _read_json(receipt_path) if receipt_path is not None else {}
    expected_relative = ""
    captured_at = pd.to_datetime(active.get("captured_at_utc"), utc=True, errors="coerce")
    if pd.notna(captured_at):
        expected_relative = (
            f"reports/active/l2_capture_receipts/{captured_at.strftime('%Y-%m-%d_%H%M%S')}.json"
        )
    active_core = {key: value for key, value in active.items() if key != "receipt_path"}
    identity_core = {key: value for key, value in active_core.items() if key != "receipt_id"}
    expected_id = (
        "l2receipt_"
        + sha256(
            json.dumps(identity_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    if active.get("schema_version") != SCHEMA_VERSION:
        blockers.append("l2_capture_receipt_schema_invalid")
    if receipt_id != expected_id:
        blockers.append("l2_capture_receipt_identity_invalid")
    if not expected_relative or receipt_relative != expected_relative:
        blockers.append("l2_capture_receipt_path_invalid")
    if receipt_path is None or not receipt_path.is_file():
        blockers.append("l2_capture_immutable_receipt_missing")
    elif immutable != active_core:
        blockers.append("l2_capture_immutable_receipt_mismatch")
    candidate_path = _safe_root_artifact(
        root, str(active.get("candidate_evidence_path", "")).strip()
    )
    candidate_hash = str(active.get("candidate_evidence_sha256", "")).strip()
    if (
        candidate_path is None
        or not candidate_path.is_file()
        or len(candidate_hash) != 64
        or _file_sha256(candidate_path) != candidate_hash
    ):
        blockers.append("l2_capture_candidate_binding_invalid")
    for label, path_field, hash_field in (
        (
            "pair_cost_bundle_manifest",
            "pair_cost_bundle_manifest_path",
            "pair_cost_bundle_manifest_sha256",
        ),
        (
            "pair_cost_bundle_pointer",
            "pair_cost_bundle_pointer_path",
            "pair_cost_bundle_pointer_sha256",
        ),
        (
            "pair_cost_model_snapshot",
            "pair_cost_model_snapshot_path",
            "pair_cost_model_snapshot_sha256",
        ),
    ):
        artifact_path = _safe_root_artifact(root, str(active.get(path_field, "")).strip())
        artifact_hash = str(active.get(hash_field, "")).strip()
        if (
            artifact_path is None
            or not artifact_path.is_file()
            or len(artifact_hash) != 64
            or _file_sha256(artifact_path) != artifact_hash
        ):
            blockers.append(f"l2_capture_{label}_binding_invalid")
    pointer_snapshot = _safe_root_artifact(
        root, str(active.get("pair_cost_bundle_pointer_path", "")).strip()
    )
    blockers.extend(
        f"l2_capture_pair_cost_pointer:{value}"
        for value in _pair_cost_pointer_blockers(
            root=root,
            pointer_path=pointer_snapshot,
            expected_bundle_id=str(active.get("pair_cost_bundle_id", "")).strip(),
            require_active_model_match=False,
        )
    )
    if not _authority_is_zero(active):
        blockers.append("l2_capture_receipt_authority_violation")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "receipt_id": receipt_id,
        "receipt_path": receipt_relative,
        "candidate_evidence_sha256": candidate_hash,
    }


def run_post_window_readiness_refresh(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    expected_l2_receipt_id: str = "",
    gate_builder: Callable[..., CommandResult] | None = None,
    handoff_builder: Callable[..., CommandResult] | None = None,
    handoff_validator: Callable[..., dict[str, Any]] | None = None,
) -> CommandResult:
    """Refresh local Stage 4 readiness from one exact, fully ready L2 receipt."""

    evaluated_at = _as_utc(now)
    active_dir = root / "reports" / "active"
    active_dir.mkdir(parents=True, exist_ok=True)
    active_l2_path = active_dir / "corrective_l2_capture_status.json"
    l2_status = _read_json(active_l2_path)
    validation = validate_l2_capture_receipt(root=root, receipt=l2_status)
    blockers = list(validation["blockers"])
    source_l2_receipt_id = str(l2_status.get("receipt_id", "")).strip()
    refresh_executed = False
    gate_refresh_executed = False
    handoff_refresh_executed = False
    status = "BLOCKED_L2_RECEIPT"
    gate_status = "NOT_RUN"
    handoff_status = "NOT_RUN"
    handoff_receipt_id = ""
    handoff_validation_status = "NOT_RUN"
    output_paths: dict[str, Path] = {}
    acquired: list[Path] = []

    if expected_l2_receipt_id and source_l2_receipt_id != expected_l2_receipt_id:
        blockers.append("post_window_expected_l2_receipt_changed")
    eligible_pairs = _safe_int(l2_status.get("strict_pair_cost_eligible"))
    ready_pairs = _safe_int(l2_status.get("strict_pair_cost_ready"))
    collecting_pairs = _safe_int(l2_status.get("post_window_collecting_pairs"))
    registered_candidates = _safe_int(l2_status.get("registered_contract_candidates"))
    fully_ready = bool(
        eligible_pairs > 0
        and ready_pairs == eligible_pairs
        and collecting_pairs == 0
        and str(l2_status.get("strict_pair_cost_acceptance_status", "")) == "PASS"
    )
    if not blockers and not fully_ready:
        status = "WAITING_STRICT_L2"
    elif not blockers and registered_candidates <= 0:
        status = "NOT_APPLICABLE_NO_REGISTERED_COHORT"
    elif not blockers:
        source_before = _source_fingerprint(root)
        pointer_blocker_count = len(blockers)
        active_pointer_path = _safe_root_artifact(
            root,
            str(
                l2_status.get(
                    "active_pair_cost_bundle_pointer_path",
                    "reports/active/hyperliquid_pair_cost_bundle_pointer.json",
                )
            ).strip(),
        )
        active_pointer_hash = str(
            l2_status.get("active_pair_cost_bundle_pointer_sha256", "")
        ).strip()
        if (
            active_pointer_path is None
            or active_pointer_path != active_dir / "hyperliquid_pair_cost_bundle_pointer.json"
            or len(active_pointer_hash) != 64
            or _file_sha256(active_pointer_path) != active_pointer_hash
        ):
            blockers.append("post_window_active_pair_cost_pointer_changed")
        else:
            blockers.extend(
                f"post_window_active_pair_cost_pointer:{value}"
                for value in _pair_cost_pointer_blockers(
                    root=root,
                    pointer_path=active_pointer_path,
                    expected_bundle_id=str(l2_status.get("pair_cost_bundle_id", "")).strip(),
                    require_active_model_match=True,
                )
            )
        if len(blockers) > pointer_blocker_count:
            status = "BLOCKED_ACTIVE_ROUTING_POINTER"
        bundle_blocker_count = len(blockers)
        bundle_path = _safe_root_artifact(
            root, str(l2_status.get("pair_cost_bundle_manifest_path", "")).strip()
        )
        model_path = _bundle_model_path(root=root, bundle_path=bundle_path)
        if bundle_path is None or model_path is None:
            blockers.append("post_window_pair_cost_bundle_path_invalid")
        else:
            blockers.extend(
                validate_pair_cost_bundle_artifacts(
                    root=root,
                    bundle_manifest_path=bundle_path,
                    pair_cost_models_path=model_path,
                )
            )
        if len(blockers) > bundle_blocker_count and status != ("BLOCKED_ACTIVE_ROUTING_POINTER"):
            status = "BLOCKED_COST_BUNDLE"
        if not blockers:
            try:
                for name in POST_WINDOW_LOCK_NAMES:
                    lock_path = active_dir / name
                    acquire_scheduler_lock(
                        lock_path,
                        now=evaluated_at,
                        timeout_seconds=4 * 60 * 60,
                    )
                    acquired.append(lock_path)
                locked_status = _read_json(active_l2_path)
                locked_validation = validate_l2_capture_receipt(root=root, receipt=locked_status)
                if locked_validation["status"] != "PASS":
                    blockers.extend(locked_validation["blockers"])
                if str(locked_status.get("receipt_id", "")) != source_l2_receipt_id:
                    blockers.append("post_window_l2_receipt_changed_under_lock")
                if _source_fingerprint(root) != source_before:
                    blockers.append("post_window_source_cohort_changed_before_refresh")
                if blockers:
                    status = "BLOCKED_SOURCE_DRIFT"
                else:
                    if gate_builder is None or handoff_builder is None or handoff_validator is None:
                        from quant_platform.orchestration.corrective_registered_rerun import (
                            build_registered_rerun_gate,
                        )
                        from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
                            build_corrective_stage4_handoff_readiness,
                            validate_stage4_handoff_readiness_receipt,
                        )

                        gate_builder = gate_builder or build_registered_rerun_gate
                        handoff_builder = (
                            handoff_builder or build_corrective_stage4_handoff_readiness
                        )
                        handoff_validator = (
                            handoff_validator or validate_stage4_handoff_readiness_receipt
                        )
                    gate_result = gate_builder(root=root, now=evaluated_at)
                    gate_refresh_executed = True
                    gate_status = str(gate_result.summary.get("status", "BLOCKED"))
                    if not gate_status.startswith("PASS"):
                        blockers.append(
                            f"post_window_registered_gate_status:{gate_status}"
                        )
                    if not _authority_is_zero(gate_result.summary):
                        blockers.append("post_window_registered_gate_authority_violation")
                    if _source_fingerprint(root) != source_before:
                        blockers.append("post_window_source_cohort_changed_during_refresh")
                    refresh_executed = True
                    output_paths.update(
                        {
                            f"registered_gate_{key}": Path(value)
                            for key, value in gate_result.paths.items()
                        }
                    )
                    if not blockers:
                        handoff_result = handoff_builder(root=root, now=evaluated_at)
                        handoff_refresh_executed = True
                        handoff_status = str(
                            handoff_result.summary.get("status", "BLOCKED_STAGE4_HANDOFF")
                        )
                        if not handoff_status.startswith("PASS"):
                            blockers.append(
                                f"post_window_stage4_handoff_status:{handoff_status}"
                            )
                        handoff_receipt_id = str(handoff_result.summary.get("receipt_id", ""))
                        if not _authority_is_zero(handoff_result.summary):
                            blockers.append("post_window_stage4_handoff_authority_violation")
                        handoff_validation = handoff_validator(
                            root=root, receipt=handoff_result.summary
                        )
                        handoff_validation_status = str(handoff_validation.get("status", "BLOCKED"))
                        if handoff_validation_status != "PASS":
                            blockers.extend(
                                f"post_window_stage4_validation:{value}"
                                for value in handoff_validation.get("blockers", [])
                            )
                        if _source_fingerprint(root) != source_before:
                            blockers.append("post_window_source_cohort_changed_during_refresh")
                        output_paths.update(
                            {
                                f"stage4_handoff_{key}": Path(value)
                                for key, value in handoff_result.paths.items()
                            }
                        )
                    status = (
                        "PASS_LOCAL_READINESS_REFRESH"
                        if not blockers
                        else "BLOCKED_LOCAL_READINESS_REFRESH"
                    )
            except FileExistsError:
                status = "DEFERRED_CONCURRENT_PRODUCER"
                blockers.append("post_window_concurrent_producer_active")
            except Exception as exc:  # noqa: BLE001 - publish fail-closed evidence
                status = "BLOCKED_LOCAL_READINESS_REFRESH"
                blockers.append(f"post_window_readiness_refresh_error:{safe_exception_code(exc)}")
            finally:
                for lock_path in reversed(acquired):
                    lock_path.unlink(missing_ok=True)

    payload: dict[str, Any] = {
        "schema_version": READINESS_REFRESH_SCHEMA_VERSION,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "status": status,
        "source_l2_receipt_id": source_l2_receipt_id,
        "source_l2_receipt_path": str(l2_status.get("receipt_path", "")),
        "source_l2_receipt_sha256": _file_sha256(
            _safe_root_artifact(root, str(l2_status.get("receipt_path", "")).strip())
        ),
        "source_candidate_sha256": str(l2_status.get("candidate_evidence_sha256", "")),
        "source_pair_cost_bundle_id": str(l2_status.get("pair_cost_bundle_id", "")),
        "eligible_pairs": eligible_pairs,
        "ready_pairs": ready_pairs,
        "collecting_pairs": collecting_pairs,
        "registered_contract_candidates": registered_candidates,
        "refresh_executed": refresh_executed,
        "registered_gate_refresh_executed": gate_refresh_executed,
        "stage4_handoff_refresh_executed": handoff_refresh_executed,
        "registered_gate_status": gate_status,
        "stage4_handoff_status": handoff_status,
        "stage4_handoff_receipt_id": handoff_receipt_id,
        "stage4_handoff_validation_status": handoff_validation_status,
        "blockers": list(dict.fromkeys(blockers)),
        "external_wizard_call_included": False,
        "candidate_refresh_execution_included": False,
        "rerun_execution_included": False,
        "order_submission_included": False,
        "research_only": True,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_id"] = (
        "l2readiness_"
        + sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    status_path = active_dir / "corrective_l2_readiness_refresh_status.json"
    immutable_path = (
        root / "data" / "research" / "l2_readiness_refresh" / f"{payload['receipt_id']}.json"
    )
    _atomic_json(payload, status_path)
    _write_immutable_json(payload, immutable_path)
    output_paths.update({"status": status_path, "immutable_receipt": immutable_path})
    from quant_platform.orchestration.corrective_canonical_status import (
        build_canonical_program_status,
    )

    canonical = build_canonical_program_status(root=root, now=evaluated_at)
    output_paths.update({f"canonical_{name}": path for name, path in canonical.paths.items()})
    return CommandResult(paths=output_paths, summary=payload)


def validate_post_window_readiness_receipt(
    *, root: Path = ROOT, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate one immutable, zero-authority L2-to-Stage-4 transition receipt."""

    payload = receipt or _read_json(
        root / "reports" / "active" / "corrective_l2_readiness_refresh_status.json"
    )
    blockers: list[str] = []
    receipt_id = str(payload.get("receipt_id", "")).strip()
    material = {key: value for key, value in payload.items() if key != "receipt_id"}
    expected_id = (
        "l2readiness_"
        + sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    immutable_path = root / "data" / "research" / "l2_readiness_refresh" / f"{receipt_id}.json"
    if payload.get("schema_version") != READINESS_REFRESH_SCHEMA_VERSION:
        blockers.append("l2_readiness_refresh_schema_invalid")
    if receipt_id != expected_id:
        blockers.append("l2_readiness_refresh_identity_invalid")
    if not immutable_path.is_file():
        blockers.append("l2_readiness_refresh_immutable_receipt_missing")
    elif _read_json(immutable_path) != payload:
        blockers.append("l2_readiness_refresh_immutable_receipt_mismatch")
    if not _authority_is_zero(payload):
        blockers.append("l2_readiness_refresh_authority_violation")
    for field in (
        "external_wizard_call_included",
        "candidate_refresh_execution_included",
        "rerun_execution_included",
        "order_submission_included",
    ):
        if payload.get(field) is not False:
            blockers.append(f"l2_readiness_refresh_forbidden_action:{field}")
    if payload.get("research_only") is not True:
        blockers.append("l2_readiness_refresh_not_research_only")
    eligible = _safe_int(payload.get("eligible_pairs"), default=-1)
    ready = _safe_int(payload.get("ready_pairs"), default=-1)
    collecting = _safe_int(payload.get("collecting_pairs"), default=-1)
    fully_ready_claim = bool(eligible > 0 and ready == eligible and collecting == 0)

    source_path = _safe_root_artifact(root, str(payload.get("source_l2_receipt_path", "")).strip())
    source_hash = str(payload.get("source_l2_receipt_sha256", "")).strip()
    source_l2 = _read_json(source_path) if source_path is not None else {}
    source_identity_material = {
        key: value for key, value in source_l2.items() if key != "receipt_id"
    }
    expected_source_id = (
        "l2receipt_"
        + sha256(
            json.dumps(source_identity_material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:20]
    )
    if (
        source_path is None
        or not source_path.is_file()
        or len(source_hash) != 64
        or _file_sha256(source_path) != source_hash
    ):
        blockers.append("l2_readiness_refresh_source_receipt_binding_invalid")
    if (
        not source_l2
        or source_l2.get("receipt_id") != expected_source_id
        or source_l2.get("receipt_id") != payload.get("source_l2_receipt_id")
        or not _authority_is_zero(source_l2)
    ):
        blockers.append("l2_readiness_refresh_source_receipt_identity_invalid")
    if source_l2.get("candidate_evidence_sha256") != payload.get("source_candidate_sha256"):
        blockers.append("l2_readiness_refresh_candidate_hash_mismatch")
    if source_l2.get("pair_cost_bundle_id") != payload.get("source_pair_cost_bundle_id"):
        blockers.append("l2_readiness_refresh_cost_bundle_id_mismatch")
    bundle_path = _safe_root_artifact(
        root, str(source_l2.get("pair_cost_bundle_manifest_path", "")).strip()
    )
    model_path = _bundle_model_path(root=root, bundle_path=bundle_path)
    if bundle_path is None or model_path is None:
        blockers.append("l2_readiness_refresh_cost_bundle_path_invalid")
    else:
        bundle = _read_json(bundle_path)
        if _file_sha256(bundle_path) != source_l2.get("pair_cost_bundle_manifest_sha256"):
            blockers.append("l2_readiness_refresh_cost_bundle_hash_mismatch")
        if bundle.get("bundle_id") != payload.get("source_pair_cost_bundle_id") or _file_sha256(
            model_path
        ) != bundle.get("pair_cost_models_sha256"):
            blockers.append("l2_readiness_refresh_cost_model_binding_invalid")
        if fully_ready_claim:
            blockers.extend(
                f"l2_readiness_refresh_cost_bundle:{value}"
                for value in validate_pair_cost_bundle_artifacts(
                    root=root,
                    bundle_manifest_path=bundle_path,
                    pair_cost_models_path=model_path,
                )
            )

    status = str(payload.get("status", ""))
    reported_blockers = payload.get("blockers")
    if not isinstance(reported_blockers, list):
        blockers.append("l2_readiness_refresh_blockers_invalid")
        reported_blockers = []
    executed = payload.get("refresh_executed") is True
    if status == "WAITING_STRICT_L2":
        if executed or eligible <= 0 or (ready == eligible and collecting == 0):
            blockers.append("l2_readiness_refresh_waiting_claim_invalid")
    elif status == "NOT_APPLICABLE_NO_REGISTERED_COHORT":
        if executed or _safe_int(payload.get("registered_contract_candidates")) > 0:
            blockers.append("l2_readiness_refresh_not_applicable_claim_invalid")
    elif status == "PASS_LOCAL_READINESS_REFRESH":
        if (
            not executed
            or reported_blockers
            or payload.get("stage4_handoff_validation_status") != "PASS"
        ):
            blockers.append("l2_readiness_refresh_pass_claim_invalid")
    elif status in {
        "BLOCKED_L2_RECEIPT",
        "BLOCKED_ACTIVE_ROUTING_POINTER",
        "BLOCKED_COST_BUNDLE",
        "BLOCKED_SOURCE_DRIFT",
        "BLOCKED_LOCAL_READINESS_REFRESH",
        "DEFERRED_CONCURRENT_PRODUCER",
    }:
        if not reported_blockers:
            blockers.append("l2_readiness_refresh_blocked_claim_missing_reason")
    else:
        blockers.append("l2_readiness_refresh_status_invalid")

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(dict.fromkeys(blockers)),
        "receipt_id": receipt_id,
        "receipt_status": status,
        "immutable_receipt_path": _relative(immutable_path, root),
        "source_l2_receipt_id": str(payload.get("source_l2_receipt_id", "")),
    }


def _pair_cost_result_blockers(result: dict[str, Any]) -> list[str]:
    """Validate the receipt-facing pair-cost result contract without raising."""

    blockers: list[str] = []
    required_paths = (
        "models",
        "stress",
        "model_snapshot",
        "stress_snapshot",
        "bundle_manifest",
        "bundle_pointer",
        "bundle_pointer_snapshot",
    )
    for key in required_paths:
        value = result.get(key)
        if value is None or not str(value).strip():
            blockers.append(f"pair_cost_evidence_contract_missing:{key}")
            continue
        try:
            if not Path(value).is_file():
                blockers.append(f"pair_cost_evidence_artifact_missing:{key}")
        except (OSError, TypeError, ValueError):
            blockers.append(f"pair_cost_evidence_path_invalid:{key}")
    if not str(result.get("bundle_id", "")).strip():
        blockers.append("pair_cost_evidence_contract_missing:bundle_id")
    return blockers


def materialize_corrective_candidate_funding_assets(
    *,
    root: Path,
    assets: list[str],
    now: datetime,
    funding_fetcher: Callable[..., Path] = fetch_hyperliquid_funding_history,
    history_days: int = DEFAULT_FUNDING_HISTORY_DAYS,
) -> CommandResult:
    """Fetch immutable funding evidence for candidates outside the current board."""

    selected = sorted({str(asset).strip().upper() for asset in assets if str(asset).strip()})
    if not selected:
        raise ValueError("at least one supplemental funding asset is required")
    if history_days <= 0:
        raise ValueError("history_days must be positive")
    captured_at = _as_utc(now)
    request_material = {
        "schema_version": "thewiz.corrective_candidate_funding.v1",
        "captured_at_utc": captured_at.isoformat(),
        "assets": selected,
        "history_days": history_days,
    }
    capture_id = (
        "l2funding_"
        + sha256(
            json.dumps(request_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    capture_root = root / "data" / "research" / "corrective_l2_funding" / capture_id
    raw_root = capture_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_artifacts = []
    for asset in selected:
        raw_path: Path | None = None
        normalized = pd.DataFrame()
        fetch_complete = False
        blockers = []
        try:
            raw_path = Path(
                funding_fetcher(
                    coin=asset,
                    days=history_days,
                    output_dir=raw_root,
                    end_time=captured_at,
                )
            )
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            fetch_complete = payload.get("fetch_complete") is True
            normalized = pd.DataFrame(
                normalize_hyperliquid_funding_history(
                    payload,
                    coin=asset,
                    evidence_path=_relative(raw_path, root),
                )
            )
        except Exception as exc:  # noqa: BLE001 - persist per-asset evidence failure
            blockers.append(f"funding_fetch_failed:{safe_exception_code(exc)}")
        if normalized.empty:
            blockers.append("normalized_funding_empty")
            timestamps = pd.Series(dtype="datetime64[ns, UTC]")
        else:
            timestamps = pd.to_datetime(
                normalized.get("timestamp", pd.Series(dtype=str)),
                utc=True,
                errors="coerce",
                format="mixed",
            )
            if timestamps.isna().any():
                blockers.append("funding_timestamp_parse_invalid")
        post_cutoff_rows = int((timestamps > pd.Timestamp(captured_at)).sum())
        if post_cutoff_rows:
            blockers.append("funding_contains_post_cutoff_rows")
        if not fetch_complete:
            blockers.append("funding_fetch_complete_flag_false")
        bounded = timestamps.loc[timestamps.notna() & timestamps.le(pd.Timestamp(captured_at))]
        if bounded.empty:
            blockers.append("bounded_funding_empty")
        raw_relative = _relative(raw_path, root) if raw_path is not None else ""
        raw_sha256 = _file_sha256(raw_path) if raw_path is not None else ""
        if raw_path is not None:
            raw_artifacts.append(
                {
                    "asset": asset,
                    "path": raw_relative,
                    "sha256": raw_sha256,
                }
            )
        rows.append(
            {
                "schema_version": "thewiz.corrective_candidate_funding.v1",
                "capture_id": capture_id,
                "asset": asset,
                "fetch_cutoff_at": captured_at.isoformat(),
                "history_days_requested": history_days,
                "funding_rows": len(bounded),
                "earliest_funding_at": (bounded.min().isoformat() if not bounded.empty else ""),
                "latest_funding_at": (bounded.max().isoformat() if not bounded.empty else ""),
                "post_cutoff_rows": post_cutoff_rows,
                "fetch_complete_flag": fetch_complete,
                "funding_status": "COMPLETE" if not blockers else "BLOCKED",
                "funding_blocker": ";".join(sorted(set(blockers))),
                "funding_source": "hyperliquid_public_funding_history",
                "funding_source_path": raw_relative,
                "funding_source_sha256": raw_sha256,
                "funding_path": raw_relative,
                "funding_sha256": raw_sha256,
                "promotion_authority": False,
                "order_submission_included": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    snapshot = pd.DataFrame(rows)
    snapshot_path = capture_root / "funding_asset_results.csv"
    _atomic_csv(snapshot, snapshot_path)
    active_path = root / "reports" / "active" / "corrective_l2_funding_asset_results.csv"
    existing = _read_csv(active_path)
    if not existing.empty and "asset" in existing:
        existing = existing.loc[~existing["asset"].astype(str).str.upper().isin(selected)]
    active_frame = pd.concat([existing, snapshot], ignore_index=True, sort=False)
    if not active_frame.empty:
        active_frame = active_frame.sort_values("asset").reset_index(drop=True)
    _atomic_csv(active_frame, active_path)
    status = "PASS" if snapshot["funding_status"].eq("COMPLETE").all() else "BLOCKED"
    receipt = {
        **request_material,
        "capture_id": capture_id,
        "status": status,
        "completed_assets": int(snapshot["funding_status"].eq("COMPLETE").sum()),
        "required_assets": len(snapshot),
        "snapshot_path": _relative(snapshot_path, root),
        "snapshot_sha256": _file_sha256(snapshot_path),
        "active_path": _relative(active_path, root),
        "active_sha256": _file_sha256(active_path),
        "raw_artifacts": raw_artifacts,
        "public_read_only_endpoint": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_path = capture_root / "receipt.json"
    _atomic_json(receipt, receipt_path)
    return CommandResult(
        paths={
            "assets": active_path,
            "snapshot_assets": snapshot_path,
            "receipt": receipt_path,
        },
        summary=receipt,
    )


def _eligible_current_pair_group_keys(
    candidates: pd.DataFrame,
    *,
    required_assets: set[str] | None = None,
) -> list[str]:
    if candidates.empty:
        return []
    eligible = candidates.loc[
        candidates.get("collection_eligible", pd.Series(False, index=candidates.index)).map(_truthy)
    ].copy()
    if "source_family" in eligible:
        eligible = eligible.loc[
            eligible["source_family"].astype(str).isin(("", "current_wizard_family"))
        ]
    if required_assets:
        normalized = {asset.strip().upper() for asset in required_assets if asset.strip()}
        eligible = eligible.loc[
            eligible.get("asset_x", pd.Series("", index=eligible.index))
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(normalized)
            | eligible.get("asset_y", pd.Series("", index=eligible.index))
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(normalized)
        ]
    return list(
        dict.fromkeys(
            value
            for value in eligible.get("pair_group_key", pd.Series(dtype=str))
            .astype(str)
            .str.strip()
            if value
        )
    )


def _missing_candidate_funding_assets(*, root: Path, candidates: pd.DataFrame) -> list[str]:
    required: set[str] = set()
    if not candidates.empty:
        eligible = candidates.loc[
            candidates.get("collection_eligible", pd.Series(False, index=candidates.index)).map(
                _truthy
            )
        ]
        for field in ("asset_x", "asset_y"):
            required.update(
                value
                for value in eligible.get(field, pd.Series(dtype=str))
                .astype(str)
                .str.strip()
                .str.upper()
                if value
            )
    complete: set[str] = set()
    for path in (
        root / "reports" / "active" / "current_wizard_hyperliquid_funding_asset_results.csv",
        root / "reports" / "active" / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        root / "reports" / "active" / "corrective_l2_funding_asset_results.csv",
    ):
        frame = _read_csv(path)
        if frame.empty or not {"asset", "funding_status"}.issubset(frame.columns):
            continue
        complete.update(
            frame.loc[frame["funding_status"].astype(str).eq("COMPLETE"), "asset"]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    return sorted(required - complete)


def build_post_window_transition(
    *,
    root: Path,
    candidate_path: Path,
    cost_status_path: Path,
    captured_at: datetime,
) -> dict[str, Any]:
    """Publish a no-action handoff from L2 collection to the daily Wizard refresh."""

    candidates = _read_csv(candidate_path)
    costs = _read_csv(cost_status_path)
    cost_by_asset = {
        str(row.get("asset", "")).strip().upper(): row
        for row in costs.to_dict("records")
        if str(row.get("asset", "")).strip()
    }
    rows = []
    for candidate in candidates.to_dict("records"):
        if not _truthy(candidate.get("collection_eligible")):
            continue
        assets = (
            str(candidate.get("asset_x", "")).strip().upper(),
            str(candidate.get("asset_y", "")).strip().upper(),
        )
        leg_blockers = []
        leg_ready = []
        for asset in assets:
            status = cost_by_asset.get(asset)
            if status is None:
                leg_ready.append(False)
                leg_blockers.append(f"{asset}:cost_collection_status_missing")
                continue
            ready = bool(
                _truthy(status.get("funding_complete"))
                and _truthy(status.get("strict_l2_cadence_ready"))
                and str(status.get("collection_status", "")).strip() == "READY"
            )
            leg_ready.append(ready)
            if not ready:
                blocker = str(status.get("blocker", "")).strip()
                leg_blockers.append(f"{asset}:{blocker or 'strict_cost_not_ready'}")
        pair_ready = bool(len(leg_ready) == 2 and all(leg_ready))
        rows.append(
            {
                "captured_at_utc": captured_at.isoformat(),
                "pair_group_key": str(candidate.get("pair_group_key", "")).strip(),
                "pair": str(candidate.get("pair", "")).strip(),
                "asset_x": assets[0],
                "asset_y": assets[1],
                "funding_and_strict_l2_ready": pair_ready,
                "post_window_transition_status": (
                    "READY_FOR_POST_WINDOW_WIZARD_REFRESH" if pair_ready else "COLLECTING_STRICT_L2"
                ),
                "blocker": ";".join(leg_blockers),
                "wizard_daily_credit_reset_utc": WIZARD_DAILY_CREDIT_RESET_UTC,
                "next_action": (
                    "daily_research_runner_refreshes_wizard_board_then_rematerializes_point_in_time_cost_evidence"
                    if pair_ready
                    else "continue_read_only_l2_collection"
                ),
                "candidate_refresh_execution_included": False,
                "order_submission_included": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "evidence_path": (
                    f"{_relative(candidate_path, root)};{_relative(cost_status_path, root)}"
                ),
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active" / "corrective_l2_post_window_transition.csv"
    _atomic_csv(frame, path)
    ready_pairs = int(
        frame.get("funding_and_strict_l2_ready", pd.Series(dtype=bool)).map(_truthy).sum()
    )
    return {
        "path": path,
        "pairs": len(frame),
        "ready_pairs": ready_pairs,
        "collecting_pairs": len(frame) - ready_pairs,
        "candidate_refresh_execution_included": False,
        "order_submission_included": False,
        "live_trading_authorized": False,
    }


def _maintain_exhaustive_mapping(
    *,
    root: Path,
    now: datetime,
    inventory_refresher: Callable[..., pd.DataFrame] | None,
    mapping_refresher: Callable[..., CommandResult],
) -> dict[str, Any]:
    active = root / "reports" / "active"
    run_manifest = active / "exhaustive_wizard_hyperliquid_run_manifest.json"
    mapping_path = active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv"
    state: dict[str, Any] = {
        "configured": run_manifest.is_file(),
        "due": False,
        "hard_stale_before": False,
        "status": "NOT_CONFIGURED",
        "age_hours_before": None,
        "age_hours_after": None,
        "inventory_refreshed": False,
        "mapping_refresh_id": "",
        "ready_pair_groups": 0,
        "blocked_pair_groups": 0,
        "blocker": "",
        "paths": {},
    }
    if not state["configured"]:
        return state

    state.update(_existing_mapping_counts(mapping_path))
    source_timestamp = _mapping_source_timestamp(mapping_path)
    age_hours = _age_hours(source_timestamp, now)
    invalid_timestamp = bool(source_timestamp is None or source_timestamp > now)
    state["age_hours_before"] = age_hours
    state["due"] = bool(
        invalid_timestamp or (age_hours is not None and age_hours >= MAPPING_REFRESH_INTERVAL_HOURS)
    )
    state["hard_stale_before"] = bool(
        invalid_timestamp or (age_hours is not None and age_hours >= MAPPING_HARD_STALE_HOURS)
    )
    if not state["due"]:
        state["status"] = "NOT_DUE"
        state["age_hours_after"] = age_hours
        return state

    try:
        refresh_inventory = inventory_refresher
        if refresh_inventory is None:
            from quant_platform.execution import (
                refresh_hyperliquid_testnet_market_inventory,
            )

            refresh_inventory = refresh_hyperliquid_testnet_market_inventory
        inventory = refresh_inventory(root=root)
        inventory_blocker = _inventory_refresh_blocker(inventory, now)
        if inventory_blocker:
            raise ValueError(inventory_blocker)
        state["inventory_refreshed"] = True
        mapping = mapping_refresher(root=root, now=now)
        summary = dict(mapping.summary)
        if int(summary.get("pair_groups", 0) or 0) <= 0:
            raise ValueError("mapping_refresh_returned_zero_pair_groups")
        if summary.get("live_trading_authorized") is not False:
            raise ValueError("mapping_refresh_authority_violation")
        refreshed_timestamp = _mapping_source_timestamp(mapping_path)
        refreshed_age_hours = _age_hours(refreshed_timestamp, now)
        if (
            refreshed_timestamp is None
            or refreshed_timestamp
            > now + timedelta(minutes=MAPPING_REFRESH_RUNTIME_TOLERANCE_MINUTES)
            or refreshed_age_hours is None
            or refreshed_age_hours > 0.25
        ):
            raise ValueError("mapping_refresh_did_not_publish_current_inventory_evidence")
        state.update(
            {
                "status": "PASS",
                "age_hours_after": refreshed_age_hours,
                "mapping_refresh_id": str(summary.get("mapping_refresh_id", "")),
                "ready_pair_groups": int(summary.get("current_ready_pair_groups", 0) or 0),
                "blocked_pair_groups": int(summary.get("current_blocked_pair_groups", 0) or 0),
                "paths": {key: Path(path) for key, path in mapping.paths.items()},
            }
        )
    except Exception as exc:  # noqa: BLE001 - receipt must preserve refresh failure
        severity = "hard_stale" if state["hard_stale_before"] else "warning"
        state["status"] = f"BLOCKED_{severity.upper()}"
        state["blocker"] = f"exhaustive_mapping_refresh_{severity}:{safe_exception_code(exc)}"
        state["age_hours_after"] = _age_hours(_mapping_source_timestamp(mapping_path), now)
    return state


def _mapping_source_timestamp(path: Path) -> datetime | None:
    frame = _read_csv(path)
    if frame.empty or "current_inventory_checked_at" not in frame.columns:
        return None
    timestamps = pd.to_datetime(
        frame["current_inventory_checked_at"],
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    if timestamps.empty:
        return None
    return timestamps.max().to_pydatetime().astimezone(UTC)


def _existing_mapping_counts(path: Path) -> dict[str, object]:
    frame = _read_csv(path)
    if frame.empty:
        return {}
    ready = int(frame.get("current_pair_ready", pd.Series(dtype=object)).map(_truthy).sum())
    identifiers = [
        str(value).strip()
        for value in frame.get("mapping_refresh_id", pd.Series(dtype=object))
        if str(value).strip()
    ]
    return {
        "mapping_refresh_id": identifiers[0] if identifiers else "",
        "ready_pair_groups": ready,
        "blocked_pair_groups": len(frame) - ready,
    }


def _age_hours(timestamp: datetime | None, now: datetime) -> float | None:
    if timestamp is None:
        return None
    return max(0.0, (now - timestamp).total_seconds() / 3600.0)


def _inventory_refresh_blocker(frame: pd.DataFrame, now: datetime) -> str:
    if frame.empty:
        return "testnet_inventory_refresh_returned_empty"
    blockers = sorted(
        {
            str(value).strip()
            for value in frame.get("fetch_blocker", pd.Series(dtype=object))
            if str(value).strip() and str(value).strip().lower() != "nan"
        }
    )
    if blockers:
        return "testnet_inventory_refresh_blocked:" + ";".join(blockers)
    tradable = frame.get("tradable_perp", pd.Series(dtype=object)).map(_truthy)
    if not tradable.any():
        return "testnet_inventory_refresh_has_no_tradable_perps"
    timestamps = pd.to_datetime(
        frame.get("checked_at_utc", pd.Series(dtype=object)),
        utc=True,
        errors="coerce",
        format="mixed",
    ).dropna()
    if timestamps.empty:
        return "testnet_inventory_refresh_timestamp_missing"
    newest = timestamps.max().to_pydatetime().astimezone(UTC)
    if newest > now + timedelta(minutes=MAPPING_REFRESH_RUNTIME_TOLERANCE_MINUTES):
        return "testnet_inventory_refresh_timestamp_in_future"
    if now - newest > timedelta(minutes=15):
        return "testnet_inventory_refresh_timestamp_stale"
    return ""


def install_corrective_l2_launch_agent(
    *,
    root: Path = ROOT,
    interval_seconds: int = DEFAULT_INTERVAL_SECONDS,
    system_path: Path | None = None,
) -> dict[str, Any]:
    """Install, but do not bootstrap, the public-depth LaunchAgent."""

    if interval_seconds < 60:
        raise ValueError("L2 collection interval must be at least 60 seconds")
    python = scheduler_python_path(root)
    if not python.is_file():
        raise FileNotFoundError(f"scheduler Python missing: {python}")
    logs = ensure_scheduler_log_directory(root)
    ensure_runtime_temp_directory(root)
    payload = _launch_agent_plist(
        root=root,
        python=python,
        logs=logs,
        interval_seconds=interval_seconds,
    )
    publication = write_launch_agent_plist(
        root=root,
        label=LAUNCH_AGENT_LABEL,
        payload=payload,
        system_path=system_path,
    )
    return {
        "label": LAUNCH_AGENT_LABEL,
        "plist": publication["workspace_plist"],
        **publication,
        "interval_seconds": interval_seconds,
        "public_read_only_endpoint": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _launch_agent_plist(*, root: Path, python: Path, logs: Path, interval_seconds: int) -> str:
    if python != scheduler_python_path(root):
        raise ValueError("L2 scheduler interpreter must use canonical runtime")
    if logs != scheduler_log_directory(root):
        raise ValueError("L2 scheduler logs must use canonical runtime")
    return scheduler_launch_agent_plist(
        root,
        contract=scheduler_contract("hyperliquid_l2"),
        interval_seconds=interval_seconds,
    )


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        publication_scope="public_l2",
    )


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    write_immutable_json(path, payload, publication_scope="public_l2")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    atomic_write_text(
        path,
        frame.to_csv(index=False),
        publication_scope="public_l2",
    )


def _file_sha256(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_root_artifact(root: Path, value: str) -> Path | None:
    if not value:
        return None
    relative = Path(value)
    if relative.is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _bundle_model_path(*, root: Path, bundle_path: Path | None) -> Path | None:
    if bundle_path is None or not bundle_path.is_file():
        return None
    bundle = _read_json(bundle_path)
    return _safe_root_artifact(root, str(bundle.get("pair_cost_models_path", "")).strip())


def _pair_cost_pointer_blockers(
    *,
    root: Path,
    pointer_path: Path | None,
    expected_bundle_id: str,
    require_active_model_match: bool,
) -> list[str]:
    blockers: list[str] = []
    if pointer_path is None or not pointer_path.is_file():
        return ["pair_cost_pointer_missing"]
    pointer = _read_json(pointer_path)
    pointer_core = {key: value for key, value in pointer.items() if key != "receipt_sha256"}
    pointer_material = {key: value for key, value in pointer_core.items() if key != "pointer_id"}
    expected_pointer_id = (
        "l2costpointer_"
        + sha256(
            json.dumps(pointer_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    expected_receipt_hash = sha256(
        json.dumps(pointer_core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    bundle_id = str(pointer.get("bundle_id", "")).strip()
    bundle_dir = root / "data" / "research" / "l2_cost_model_receipts" / bundle_id
    expected_pointer_path = (
        root / "reports" / "active" / "hyperliquid_pair_cost_bundle_pointer.json"
        if require_active_model_match
        else bundle_dir / "pointer.json"
    )
    manifest_path = _safe_root_artifact(root, str(pointer.get("bundle_manifest_path", "")).strip())
    model_path = _safe_root_artifact(root, str(pointer.get("pair_cost_models_path", "")).strip())
    active_model_path = _safe_root_artifact(
        root, str(pointer.get("active_pair_cost_models_path", "")).strip()
    )
    if (
        pointer.get("schema_version") != "thewiz.l2_cost_model_pointer.v1"
        or not bundle_id.startswith("l2costbundle_")
        or bundle_id != expected_bundle_id
        or pointer_path.resolve() != expected_pointer_path.resolve()
    ):
        blockers.append("pair_cost_pointer_identity_invalid")
    if (
        pointer.get("pointer_id") != expected_pointer_id
        or pointer.get("receipt_sha256") != expected_receipt_hash
    ):
        blockers.append("pair_cost_pointer_receipt_invalid")
    if (
        manifest_path != bundle_dir / "receipt.json"
        or manifest_path is None
        or not manifest_path.is_file()
        or _file_sha256(manifest_path) != str(pointer.get("bundle_manifest_sha256", ""))
    ):
        blockers.append("pair_cost_pointer_manifest_binding_invalid")
    if (
        model_path != bundle_dir / "pair_cost_models.csv"
        or model_path is None
        or not model_path.is_file()
        or _file_sha256(model_path) != str(pointer.get("pair_cost_models_sha256", ""))
    ):
        blockers.append("pair_cost_pointer_model_binding_invalid")
    if active_model_path != root / "data" / "processed" / "hyperliquid_pair_cost_models.csv":
        blockers.append("pair_cost_pointer_active_model_path_invalid")
    elif require_active_model_match and (
        not active_model_path.is_file()
        or _file_sha256(active_model_path) != str(pointer.get("active_pair_cost_models_sha256", ""))
        or _file_sha256(active_model_path) != _file_sha256(model_path)
    ):
        blockers.append("pair_cost_pointer_active_model_binding_invalid")
    if not _authority_is_zero(pointer):
        blockers.append("pair_cost_pointer_authority_violation")
    return blockers


def _source_fingerprint(root: Path) -> dict[str, str]:
    return {
        relative: (
            _append_only_change_token(root / relative)
            if relative in POST_WINDOW_APPEND_ONLY_PATHS
            else _file_sha256(root / relative)
        )
        for relative in POST_WINDOW_SOURCE_PATHS
    }


def _append_only_change_token(path: Path) -> str:
    """Detect same-run ledger mutation without rehashing the full append-only file."""

    try:
        stat = path.stat()
    except OSError:
        return "MISSING"
    return f"stat:{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}"


def _authority_is_zero(payload: dict[str, Any]) -> bool:
    authority_fields = (
        "candidate_promotion_authority",
        "promotion_authority",
        "execution_authority",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    return all(not _truthy(payload.get(field)) for field in authority_fields if field in payload)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        try:
            with governed_evidence_write_lock(
                ROOT, blocking=False, scope="scheduler_config"
            ):
                result: Any = install_corrective_l2_launch_agent()
        except (GovernedEvidenceMaintenanceActive, GovernedEvidenceLockBusy) as exc:
            result = {
                "summary": {
                    "status": "DEFERRED_PHASE00_OR_GOVERNED_LOCK",
                    "blockers": [safe_exception_code(exc)],
                    "live_trading_authorized": False,
                },
                "paths": {},
            }
    else:
        def _run_l2() -> dict[str, Any]:
            capture = run_corrective_l2_capture(require_launchd_provenance=True)
            readiness = run_post_window_readiness_refresh(
                expected_l2_receipt_id=str(capture.summary.get("receipt_id", ""))
            )
            readiness_validation = validate_post_window_readiness_receipt(
                receipt=readiness.summary
            )
            return _build_supervised_l2_result(
                capture=capture,
                readiness=readiness,
                readiness_validation=readiness_validation,
            )

        supervised = supervise_scheduler_run(
            root=ROOT,
            contract_key="hyperliquid_l2",
            publication_scope="public_l2",
            callback=_run_l2,
            require_launchd_provenance=True,
        )
        result = {
            "summary": supervised.result_summary,
            "paths": supervised.result_paths,
            "terminal_receipt": supervised.terminal_receipt,
            "terminal_paths": {
                key: str(value) for key, value in supervised.terminal_paths.items()
            },
        }
        print(json.dumps(result, indent=2, default=str))
        if supervised.exit_code:
            raise SystemExit(supervised.exit_code)
        return
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
