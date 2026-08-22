"""Audit the zero-authority Stage 3 to Stage 4 research handoff."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_registered_learning_protocol import (
    validate_registered_stage5_protocol,
)
from quant_platform.orchestration.corrective_registered_rerun import (
    validate_registered_rerun_contract_identity,
)
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    validate_wizard_reset_readiness_receipt,
    wizard_reset_readiness_state_sha256,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_stage4_handoff_readiness.v4"
MINIMUM_INDEPENDENT_CLUSTERS = 3
Stage5ProtocolValidator = Callable[..., tuple[Path, dict[str, Any]]]
SOURCE_ARTIFACT_KEYS = frozenset(
    {
        "active_contract",
        "immutable_contract",
        "registered_source_receipt",
        "registered_source_matrix",
        "active_gate",
        "active_gate_rows",
        "pending_preflight",
        "pending_preflight_rows",
        "current_hypothesis_batch",
        "active_stage3_manifest",
        "immutable_stage3_manifest",
        "immutable_stage3_reset",
        "stage5_protocol_pointer",
        "immutable_stage5_protocol",
        "proof_scheduler_status",
    }
)


def build_corrective_stage4_handoff_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    stage5_protocol_validator: Stage5ProtocolValidator = (validate_registered_stage5_protocol),
) -> CommandResult:
    """Prove the registered Stage 4 route is ready without running research."""

    checked_at = _as_utc(now)
    active = root / "reports" / "active"
    contract_path = active / "registered_research_rerun_contract.json"
    gate_path = active / "registered_research_rerun_gate.json"
    gate_rows_path = active / "registered_research_rerun_gate.csv"
    preflight_path = active / "registered_rerun_family_preflight.json"
    preflight_rows_path = active / "registered_rerun_family_preflight.csv"
    batch_path = active / "current_hypothesis_batch.csv"
    manifest_path = active / "corrective_wizard_next_capture_manifest.json"
    reset_path = active / "wizard_reset_readiness.json"
    protocol_path = active / "registered_stage5_protocol.json"
    scheduler_path = active / "corrective_wizard_proof_scheduler_status.json"

    contract = _read_json(contract_path)
    gate = _read_json(gate_path)
    gate_rows = _read_csv(gate_rows_path)
    preflight = _read_json(preflight_path)
    preflight_rows = _read_csv(preflight_rows_path)
    batch_rows = _read_csv(batch_path)
    manifest = _read_json(manifest_path)
    reset = _read_json(reset_path)
    protocol = _read_json(protocol_path)
    scheduler = _read_json(scheduler_path)
    checks: list[dict[str, str]] = []

    contract_id = _text(contract.get("contract_id"))
    immutable_contract_relative = _text(contract.get("immutable_contract_path"))
    immutable_contract_path = root / immutable_contract_relative
    immutable_contract = _read_json(immutable_contract_path)
    contract_identity_valid = False
    try:
        validate_registered_rerun_contract_identity(contract)
        contract_identity_valid = True
    except ValueError:
        contract_identity_valid = False
    contract_binding_valid = bool(
        contract.get("schema_version") == "thewiz.corrective_registered_rerun.v1"
        and contract_id.startswith("registeredrerun_")
        and immutable_contract_relative
        == f"data/research/registered_rerun_contracts/{contract_id}.json"
        and immutable_contract == contract
        and contract_identity_valid
    )
    _add_check(
        checks,
        name="active_contract_immutable_binding",
        passed=contract_binding_valid,
        observed=contract_id or "missing",
        evidence_path=immutable_contract_relative or _relative(contract_path, root),
        blocker="active_registered_contract_binding_invalid",
    )

    source_dir = root / "data" / "research" / "registered_rerun_source_families" / contract_id
    source_receipt_path = source_dir / "receipt.json"
    source_receipt = _read_json(source_receipt_path)
    source_matrix_relative = _text(source_receipt.get("source_family_path"))
    source_matrix_path = root / source_matrix_relative
    source_rows = _read_csv(source_matrix_path)
    source_binding_valid = bool(
        source_receipt.get("schema_version") == "thewiz.registered_rerun_source_family.v1"
        and source_receipt.get("contract_id") == contract_id
        and source_matrix_relative
        == f"data/research/registered_rerun_source_families/{contract_id}/experiment_matrix.csv"
        and source_matrix_path.is_file()
        and _file_sha256(source_matrix_path) == contract.get("source_family_sha256")
        and source_receipt.get("source_family_sha256") == contract.get("source_family_sha256")
        and len(source_rows) == _safe_int(contract.get("source_family_rows"), default=-1)
        and _safe_int(source_receipt.get("source_family_rows"), default=-1) == len(source_rows)
    )
    _add_check(
        checks,
        name="active_source_family_binding",
        passed=source_binding_valid,
        observed=f"rows={len(source_rows)}",
        evidence_path=_relative(source_receipt_path, root),
        blocker="active_registered_source_family_binding_invalid",
    )

    candidates = contract.get("registered_candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    candidate_semantic_ids = [
        _text(value.get("semantic_hypothesis_id"))
        for value in candidates
        if isinstance(value, dict)
    ]
    candidate_experiment_ids = [
        _text(value.get("source_experiment_id")) for value in candidates if isinstance(value, dict)
    ]
    candidate_pair_keys = [
        _text(value.get("pair_group_key")) for value in candidates if isinstance(value, dict)
    ]
    candidate_identity_valid = bool(
        candidates
        and len(candidate_semantic_ids) == len(candidates)
        and "" not in candidate_semantic_ids
        and len(set(candidate_semantic_ids)) == len(candidate_semantic_ids)
        and "" not in candidate_experiment_ids
        and len(set(candidate_experiment_ids)) == len(candidate_experiment_ids)
        and "" not in candidate_pair_keys
        and all(_authority_is_zero(value) for value in candidates if isinstance(value, dict))
    )
    _add_check(
        checks,
        name="active_candidate_identity",
        passed=candidate_identity_valid,
        observed=f"candidates={len(candidates)}",
        evidence_path=_relative(contract_path, root),
        blocker="active_registered_candidate_identity_invalid",
    )

    gate_semantic_ids = {_text(row.get("semantic_hypothesis_id")) for row in gate_rows} - {""}
    active_inputs_valid = bool(
        len(gate_rows) == len(candidates)
        and gate_semantic_ids == set(candidate_semantic_ids)
        and all(
            _truthy(row.get("registration_valid"))
            and _truthy(row.get("authority_boundary_safe"))
            and _truthy(row.get("strict_cost_ready_after_registration"))
            and _truthy(row.get("frozen_source_family_identity_match"))
            and _truthy(row.get("frozen_source_family_integrity"))
            and _authority_is_zero(row)
            for row in gate_rows
        )
    )
    _add_check(
        checks,
        name="active_candidate_non_vendor_inputs",
        passed=active_inputs_valid,
        observed=f"ready={sum(_truthy(row.get('strict_cost_ready_after_registration')) for row in gate_rows)}/{len(gate_rows)}",
        evidence_path=_relative(gate_rows_path, root),
        blocker="active_registered_candidate_non_vendor_inputs_invalid",
    )

    pending_hypotheses = _safe_int(preflight.get("hypotheses"), default=-1)
    pending_ready = _safe_int(preflight.get("ready_hypotheses"), default=-1)
    pending_pairs = _safe_int(preflight.get("pairs"), default=-1)
    pending_clusters = _safe_int(preflight.get("independent_clusters"), default=-1)
    pending_preflight_valid = bool(
        preflight.get("schema_version") == "thewiz.registered_rerun_family_preflight.v1"
        and preflight.get("status") == "PASS"
        and pending_hypotheses > 0
        and pending_ready == pending_hypotheses == len(preflight_rows)
        and pending_pairs >= MINIMUM_INDEPENDENT_CLUSTERS
        and pending_clusters >= MINIMUM_INDEPENDENT_CLUSTERS
        and preflight.get("source_family_valid") is True
        and preflight.get("cost_bundle_valid") is True
        and preflight.get("history_manifest_bound") is True
        and preflight.get("vendor_evidence_included") is False
        and preflight.get("rerun_execution_included") is False
        and _authority_is_zero(preflight)
    )
    _add_check(
        checks,
        name="pending_family_independent_preflight",
        passed=pending_preflight_valid,
        observed=(
            f"ready={pending_ready}/{pending_hypotheses};"
            f"pairs={pending_pairs};clusters={pending_clusters}"
        ),
        evidence_path=_relative(preflight_path, root),
        blocker="pending_registered_family_preflight_invalid",
    )

    pending_inputs_valid = bool(
        preflight_rows
        and all(
            _truthy(row.get("registration_ready"))
            and _truthy(row.get("identity_ready"))
            and _truthy(row.get("strict_cost_ready"))
            and _truthy(row.get("history_ready"))
            and _truthy(row.get("non_vendor_preflight_ready"))
            and not _text(row.get("blocker"))
            and _truthy(row.get("vendor_evidence_included")) is False
            and _authority_is_zero(row)
            for row in preflight_rows
        )
    )
    _add_check(
        checks,
        name="pending_family_non_vendor_inputs",
        passed=pending_inputs_valid,
        observed=f"ready={sum(_truthy(row.get('non_vendor_preflight_ready')) for row in preflight_rows)}/{len(preflight_rows)}",
        evidence_path=_relative(preflight_rows_path, root),
        blocker="pending_registered_family_non_vendor_inputs_invalid",
    )

    pending_identity_valid, batch_semantic_ids, batch_variant_count = (
        _pending_family_identity_valid(
            active_semantic_ids=set(candidate_semantic_ids),
            pending_rows=preflight_rows,
            batch_rows=batch_rows,
        )
    )
    pending_semantic_ids = {_text(row.get("semantic_hypothesis_id")) for row in preflight_rows} - {
        ""
    }
    active_pending_overlap = set(candidate_semantic_ids) & pending_semantic_ids
    _add_check(
        checks,
        name="active_pending_family_identity_partition",
        passed=pending_identity_valid,
        observed=(
            f"active={len(set(candidate_semantic_ids))};"
            f"pending={len(pending_semantic_ids)};"
            f"batch={len(batch_semantic_ids)};"
            f"variants={batch_variant_count};"
            f"overlap={len(active_pending_overlap)}"
        ),
        evidence_path=(
            f"{_relative(contract_path, root)};"
            f"{_relative(batch_path, root)};"
            f"{_relative(preflight_rows_path, root)}"
        ),
        blocker="registered_stage4_active_pending_identity_partition_invalid",
    )

    manifest_id = _text(manifest.get("manifest_id"))
    immutable_manifest_relative = _text(manifest.get("immutable_manifest_path"))
    immutable_manifest_path = root / immutable_manifest_relative
    manifest_binding_valid = bool(
        manifest.get("schema_version") == "thewiz.corrective_wizard_next_capture_manifest.v1"
        and manifest.get("status") == "PASS"
        and manifest.get("manifest_enforced") is True
        and manifest_id.startswith("wizardcapture_")
        and immutable_manifest_relative.startswith("data/research/wizard_capture_manifests/")
        and immutable_manifest_path.is_file()
        and _file_sha256(immutable_manifest_path)
        == _text(manifest.get("immutable_manifest_sha256"))
        and _safe_int(manifest.get("pending_calls")) > 0
    )
    _add_check(
        checks,
        name="frozen_stage3_manifest_binding",
        passed=manifest_binding_valid,
        observed=manifest_id or "missing",
        evidence_path=immutable_manifest_relative or _relative(manifest_path, root),
        blocker="stage3_frozen_capture_manifest_binding_invalid",
    )

    reset_receipt_id = _text(reset.get("receipt_id"))
    immutable_reset_relative = f"data/research/wizard_reset_readiness/{reset_receipt_id}.json"
    immutable_reset_path = root / immutable_reset_relative
    immutable_reset = _read_json(immutable_reset_path)
    reset_validation = validate_wizard_reset_readiness_receipt(
        root=root,
        receipt=reset,
    )
    reset_state_sha256 = wizard_reset_readiness_state_sha256(reset)
    reset_identity_valid = bool(
        reset_validation.get("status") == "PASS"
        and reset_validation.get("receipt_id") == reset_receipt_id
        and reset_validation.get("state_sha256") == reset_state_sha256
        and immutable_reset_path.is_file()
        and immutable_reset == reset
    )
    reset_binding_valid = bool(
        reset_identity_valid
        and reset.get("status") == "PASS_RESET_AUTOMATION_READY"
        and reset.get("manifest_id") == manifest_id
        and reset.get("immutable_manifest_path") == immutable_manifest_relative
        and reset.get("immutable_manifest_sha256") == manifest.get("immutable_manifest_sha256")
        and reset.get("launch_agent_loaded") is True
        and _safe_int(reset.get("checks_passed"), default=-1)
        == _safe_int(reset.get("checks_total"), default=-2)
        and _safe_int(reset.get("checks_total")) >= 16
        and reset.get("scheduler_manifest_continuity_valid") is True
        and reset.get("scheduler_manifest_continuity_status")
        in {"PASS_NEW_FROZEN_COHORT", "PASS_PRIOR_UNRESOLVED_COHORT_MATCH"}
        and _authority_is_zero(reset)
    )
    _add_check(
        checks,
        name="stage3_reset_automation_binding",
        passed=reset_binding_valid,
        observed=_text(reset.get("status")) or "missing",
        evidence_path=(f"{_relative(reset_path, root)};{immutable_reset_relative}"),
        blocker="stage3_reset_automation_not_bound_or_ready",
    )

    allowed_gate_statuses = {
        "BLOCKED_VENDOR_PARITY",
        "BLOCKED_PREREQUISITES",
        "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN",
        "PASS_REGISTERED_RERUN_ACCOUNTED",
        "BLOCKED_PENDING_FAMILY_ROLLOVER",
    }
    handoff_wiring_valid = bool(
        gate.get("schema_version") == "thewiz.corrective_registered_rerun.v1"
        and gate.get("contract_id") == contract_id
        and gate.get("status") in allowed_gate_statuses
        and gate.get("registered_rerun_executor_available") is True
        and gate.get("automatic_post_parity_handoff") is True
        and gate.get("family_rerun_scope") == "full_policy_defined_family"
        and gate.get("promotion_evaluation_scope") == "registered_semantic_hypotheses_only"
        and _authority_is_zero(gate)
    )
    _add_check(
        checks,
        name="registered_handoff_wiring",
        passed=handoff_wiring_valid,
        observed=_text(gate.get("status")) or "missing",
        evidence_path=_relative(gate_path, root),
        blocker="registered_stage4_handoff_wiring_invalid",
    )

    rollover_required = pending_semantic_ids != set(candidate_semantic_ids)
    reported_rollover_required = _truthy(gate.get("pending_family_rollover_required"))
    results_accounted = _truthy(gate.get("registered_rerun_results_accounted"))
    sequence_valid = bool(
        pending_semantic_ids
        and reported_rollover_required == rollover_required
        and (
            (rollover_required and not results_accounted)
            or (
                rollover_required
                and results_accounted
                and gate.get("status") == "BLOCKED_PENDING_FAMILY_ROLLOVER"
            )
            or not rollover_required
        )
    )
    handoff_state = _handoff_state(
        gate_status=_text(gate.get("status")),
        rollover_required=rollover_required,
        results_accounted=results_accounted,
    )
    _add_check(
        checks,
        name="registered_generation_sequence",
        passed=sequence_valid,
        observed=handoff_state,
        evidence_path=_relative(gate_path, root),
        blocker="registered_stage4_generation_sequence_invalid",
    )

    protocol_relative = _text(protocol.get("protocol_receipt_path"))
    immutable_protocol_path = root / protocol_relative
    protocol_current_bindings_valid = False
    try:
        validated_protocol_path, validated_protocol = stage5_protocol_validator(root=root)
        protocol_current_bindings_valid = bool(
            validated_protocol_path.resolve() == immutable_protocol_path.resolve()
            and validated_protocol.get("protocol_id") == protocol.get("protocol_id")
            and _authority_is_zero(validated_protocol)
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        protocol_current_bindings_valid = False
    protocol_binding_valid = bool(
        protocol.get("schema_version") == "thewiz.registered_stage5_protocol_pointer.v1"
        and protocol.get("status") == "PASS_PROSPECTIVE_PROTOCOL_REGISTERED"
        and _text(protocol.get("protocol_id")).startswith("stage5protocol_")
        and immutable_protocol_path.is_file()
        and _file_sha256(immutable_protocol_path) == _text(protocol.get("protocol_receipt_sha256"))
        and protocol.get("research_only") is True
        and protocol.get("thresholds_changed_after_results") is False
        and _authority_is_zero(protocol)
        and protocol_current_bindings_valid
    )
    _add_check(
        checks,
        name="prospective_stage5_protocol_binding",
        passed=protocol_binding_valid,
        observed=_text(protocol.get("protocol_id")) or "missing",
        evidence_path=protocol_relative or _relative(protocol_path, root),
        blocker="prospective_stage5_protocol_binding_invalid",
    )

    current_reset = _read_json(reset_path)
    current_reset_validation = validate_wizard_reset_readiness_receipt(
        root=root,
        receipt=current_reset,
    )
    current_reset_state_sha256 = wizard_reset_readiness_state_sha256(current_reset)
    current_reset_binding_valid = bool(
        current_reset_validation.get("status") == "PASS"
        and current_reset_validation.get("receipt_status") == "PASS_RESET_AUTOMATION_READY"
        and current_reset_state_sha256 == reset_state_sha256
    )
    _add_check(
        checks,
        name="stage3_reset_current_state_binding",
        passed=current_reset_binding_valid,
        observed=(_text(current_reset_validation.get("receipt_id")) or "missing"),
        evidence_path=_relative(reset_path, root),
        blocker="stage3_reset_current_state_changed_during_handoff_audit",
    )

    authority_payloads = {
        "contract": contract,
        "gate": gate,
        "preflight": preflight,
        "manifest": manifest,
        "reset": reset,
        "protocol": protocol,
        "scheduler": scheduler,
    }
    authority_safe = all(_authority_is_zero(value) for value in authority_payloads.values())
    _add_check(
        checks,
        name="zero_execution_authority",
        passed=authority_safe,
        observed="research_only" if authority_safe else "authority_violation",
        evidence_path=";".join(
            (
                _relative(contract_path, root),
                _relative(gate_path, root),
                _relative(preflight_path, root),
                _relative(manifest_path, root),
                _relative(protocol_path, root),
                _relative(scheduler_path, root),
            )
        ),
        blocker="stage4_handoff_authority_violation",
    )

    source_paths = {
        "active_contract": contract_path,
        "immutable_contract": immutable_contract_path,
        "registered_source_receipt": source_receipt_path,
        "registered_source_matrix": source_matrix_path,
        "active_gate": gate_path,
        "active_gate_rows": gate_rows_path,
        "pending_preflight": preflight_path,
        "pending_preflight_rows": preflight_rows_path,
        "current_hypothesis_batch": batch_path,
        "active_stage3_manifest": manifest_path,
        "immutable_stage3_manifest": immutable_manifest_path,
        "immutable_stage3_reset": immutable_reset_path,
        "stage5_protocol_pointer": protocol_path,
        "immutable_stage5_protocol": immutable_protocol_path,
        "proof_scheduler_status": scheduler_path,
    }
    source_artifacts = {
        name: _artifact_binding(root=root, path=path)
        for name, path in source_paths.items()
    }
    source_closure_complete = bool(
        set(source_artifacts) == SOURCE_ARTIFACT_KEYS
        and all(binding["sha256"] for binding in source_artifacts.values())
    )
    _add_check(
        checks,
        name="decision_source_closure",
        passed=source_closure_complete,
        observed=f"bound={sum(bool(value['sha256']) for value in source_artifacts.values())}/{len(SOURCE_ARTIFACT_KEYS)}",
        evidence_path=";".join(
            value["path"] for value in source_artifacts.values() if value["path"]
        ),
        blocker="stage4_handoff_decision_source_closure_incomplete",
    )

    blockers = [row["blocker"] for row in checks if row["status"] == "BLOCKED"]
    status = "PASS_STAGE4_HANDOFF_READY" if not blockers else "BLOCKED_STAGE4_HANDOFF"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": status,
        "handoff_state": handoff_state,
        "active_contract_id": contract_id,
        "active_contract_candidates": len(candidates),
        "active_gate_status": _text(gate.get("status")),
        "stage3_manifest_id": manifest_id,
        "stage3_manifest_immutable_path": immutable_manifest_relative,
        "stage3_reset_receipt_id": reset_receipt_id,
        "stage3_reset_state_sha256": reset_state_sha256,
        "pending_family_hypotheses": pending_hypotheses,
        "pending_family_ready": pending_ready,
        "pending_family_pairs": pending_pairs,
        "pending_family_independent_clusters": pending_clusters,
        "pending_family_semantic_batch_coverage": len(pending_semantic_ids & batch_semantic_ids),
        "pending_family_batch_semantic_hypotheses": len(batch_semantic_ids),
        "pending_family_batch_experiment_variants": batch_variant_count,
        "active_pending_semantic_overlap": len(active_pending_overlap),
        "pending_family_rollover_required": rollover_required,
        "stage5_protocol_id": _text(protocol.get("protocol_id")),
        "stage5_protocol_receipt_path": protocol_relative,
        "source_artifacts": source_artifacts,
        "source_closure_sha256": sha256(
            _canonical_json(source_artifacts).encode("utf-8")
        ).hexdigest(),
        "checks_total": len(checks),
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "blockers": blockers,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_id = (
        "stage4handoff_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    payload["receipt_id"] = receipt_id

    paths = {
        "checks": active / "stage4_handoff_readiness_checks.csv",
        "status": active / "stage4_handoff_readiness.json",
        "summary": active / "stage4_handoff_readiness.md",
        "immutable_receipt": (
            root / "data" / "research" / "stage4_handoff_readiness" / f"{receipt_id}.json"
        ),
    }
    _write_csv(checks, paths["checks"])
    _write_json(payload, paths["status"])
    _write_text(paths["summary"], _markdown(payload, checks))
    _write_immutable_json(payload, paths["immutable_receipt"])
    return CommandResult(paths=paths, summary=payload)


def validate_stage4_handoff_readiness_receipt(
    *, root: Path = ROOT, receipt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the active Stage 4 handoff against its immutable twin."""

    payload = receipt or _read_json(root / "reports" / "active" / "stage4_handoff_readiness.json")
    blockers: list[str] = []
    receipt_id = _text(payload.get("receipt_id"))
    material = {key: value for key, value in payload.items() if key != "receipt_id"}
    expected_id = (
        "stage4handoff_" + sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:20]
    )
    immutable_relative = f"data/research/stage4_handoff_readiness/{receipt_id}.json"
    immutable_path = root / immutable_relative
    immutable = _read_json(immutable_path)

    if payload.get("schema_version") != SCHEMA_VERSION:
        blockers.append("stage4_handoff_schema_version_invalid")
    if receipt_id != expected_id:
        blockers.append("stage4_handoff_receipt_id_invalid")
    if not immutable_path.is_file():
        blockers.append("stage4_handoff_immutable_receipt_missing")
    elif immutable != payload:
        blockers.append("stage4_handoff_immutable_receipt_mismatch")
    if payload.get("status") not in {
        "PASS_STAGE4_HANDOFF_READY",
        "BLOCKED_STAGE4_HANDOFF",
    }:
        blockers.append("stage4_handoff_status_invalid")
    checks_total = _safe_int(payload.get("checks_total"), default=-1)
    checks_passed = _safe_int(payload.get("checks_passed"), default=-1)
    reported_blockers = payload.get("blockers")
    if not isinstance(reported_blockers, list):
        blockers.append("stage4_handoff_blockers_invalid")
        reported_blockers = []
    if checks_total <= 0 or checks_passed < 0 or checks_passed > checks_total:
        blockers.append("stage4_handoff_check_accounting_invalid")
    if payload.get("status") == "PASS_STAGE4_HANDOFF_READY" and (
        checks_passed != checks_total or reported_blockers
    ):
        blockers.append("stage4_handoff_pass_claim_invalid")
    if payload.get("status") == "BLOCKED_STAGE4_HANDOFF" and not reported_blockers:
        blockers.append("stage4_handoff_blocked_claim_missing_reason")
    if not _authority_is_zero(payload):
        blockers.append("stage4_handoff_authority_violation")
    blockers.extend(_validate_source_closure(root=root, payload=payload))
    current_reset_validation = validate_wizard_reset_readiness_receipt(root=root)
    current_reset = _read_json(root / "reports" / "active" / "wizard_reset_readiness.json")
    current_reset_state_sha256 = wizard_reset_readiness_state_sha256(current_reset)
    if current_reset_validation.get("status") != "PASS":
        blockers.append("stage4_handoff_current_stage3_reset_receipt_invalid")
    if current_reset_validation.get("receipt_status") != "PASS_RESET_AUTOMATION_READY":
        blockers.append("stage4_handoff_current_stage3_reset_not_ready")
    if (
        not _text(payload.get("stage3_reset_state_sha256"))
        or payload.get("stage3_reset_state_sha256") != current_reset_state_sha256
    ):
        blockers.append("stage4_handoff_current_stage3_reset_state_mismatch")

    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "receipt_id": receipt_id,
        "immutable_receipt_path": immutable_relative,
        "immutable_receipt_sha256": (
            _file_sha256(immutable_path) if immutable_path.is_file() else ""
        ),
        "receipt_status": _text(payload.get("status")),
        "current_stage3_reset_receipt_id": current_reset_validation.get("receipt_id", ""),
        "current_stage3_reset_state_sha256": current_reset_state_sha256,
        "source_closure_sha256": _text(payload.get("source_closure_sha256")),
    }


def _handoff_state(*, gate_status: str, rollover_required: bool, results_accounted: bool) -> str:
    if gate_status == "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN":
        return "READY_TO_EXECUTE_ACTIVE_CONTRACT"
    if rollover_required and results_accounted:
        return "READY_TO_ROLL_PENDING_FAMILY"
    if rollover_required:
        return "READY_AWAITING_STAGE3_THEN_ACTIVE_CONTRACT"
    if results_accounted:
        return "ACTIVE_FAMILY_ACCOUNTED"
    return "READY_AWAITING_STAGE3"


def _pending_family_identity_valid(
    *,
    active_semantic_ids: set[str],
    pending_rows: list[dict[str, str]],
    batch_rows: list[dict[str, str]],
) -> tuple[bool, set[str], int]:
    """Prove the pending cohort is an exact, non-overlapping batch projection."""

    batch_semantic_ids = {_text(row.get("semantic_hypothesis_id")) for row in batch_rows} - {""}
    batch_variant_identities = {
        (
            _text(row.get("semantic_hypothesis_id")),
            _text(row.get("experiment_id")),
        )
        for row in batch_rows
        if _text(row.get("semantic_hypothesis_id")) and _text(row.get("experiment_id"))
    }
    pending_ids = [_text(row.get("semantic_hypothesis_id")) for row in pending_rows]
    pending_semantic_ids = set(pending_ids) - {""}
    required_identity_fields = (
        "equivalence_cluster_id",
        "pair",
        "wizard_timeframe",
        "exact_mode",
        "orientation",
        "registration_cohort_role",
    )
    rows_by_semantic: dict[str, list[dict[str, str]]] = {}
    for row in batch_rows:
        rows_by_semantic.setdefault(_text(row.get("semantic_hypothesis_id")), []).append(row)

    representatives_valid = True
    for pending in pending_rows:
        semantic_id = _text(pending.get("semantic_hypothesis_id"))
        experiment_id = _text(pending.get("experiment_id"))
        variants = rows_by_semantic.get(semantic_id, [])
        variant_experiment_ids = {_text(row.get("experiment_id")) for row in variants} - {""}
        represented_experiment_ids = {
            experiment_id,
            *_semicolon_values(pending.get("alternative_experiment_ids")),
        } - {""}
        representative = next(
            (row for row in variants if _text(row.get("experiment_id")) == experiment_id),
            None,
        )
        representatives_valid = bool(
            representatives_valid
            and representative is not None
            and (semantic_id, experiment_id) in batch_variant_identities
            and represented_experiment_ids == variant_experiment_ids
            and _safe_int(pending.get("source_experiment_count"), default=-1)
            == len(variant_experiment_ids)
            and all(
                _text(pending.get(field)) == _text(representative.get(field))
                for field in required_identity_fields
            )
        )

    valid = bool(
        pending_rows
        and batch_rows
        and "" not in pending_ids
        and len(pending_ids) == len(pending_semantic_ids)
        and len(batch_variant_identities) == len(batch_rows)
        and pending_semantic_ids == batch_semantic_ids
        and active_semantic_ids.isdisjoint(pending_semantic_ids)
        and representatives_valid
    )
    return valid, batch_semantic_ids, len(batch_variant_identities)


def _authority_is_zero(payload: dict[str, Any]) -> bool:
    authority_fields = (
        "candidate_promotion_authority",
        "promotion_authority",
        "execution_authority",
        "order_submission_included",
        "order_submission_performed",
        "testnet_candidate_authority",
        "testnet_order_authority",
        "live_trading_authorized",
        "execution_authority_at_contract",
        "testnet_order_authority_at_contract",
        "live_trading_authorized_at_contract",
    )
    return all(not _truthy(payload.get(field)) for field in authority_fields if field in payload)


def _add_check(
    checks: list[dict[str, str]],
    *,
    name: str,
    passed: bool,
    observed: str,
    evidence_path: str,
    blocker: str,
) -> None:
    checks.append(
        {
            "check": name,
            "status": "PASS" if passed else "BLOCKED",
            "observed": observed,
            "evidence_path": evidence_path,
            "blocker": "" if passed else blocker,
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return []


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["check", "status", "observed", "evidence_path", "blocker"],
        )
        writer.writeheader()
        writer.writerows(rows)
    promote_staged_file(temporary, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable Stage 4 readiness receipt collision: {path}")
        return
    _write_text(path, encoded)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    promote_staged_file(temporary, path)


def _markdown(payload: dict[str, Any], checks: list[dict[str, str]]) -> str:
    lines = [
        "# Stage 4 Handoff Readiness",
        "",
        f"- Status: `{payload['status']}`",
        f"- Handoff state: `{payload['handoff_state']}`",
        f"- Active contract: `{payload['active_contract_id']}`",
        f"- Active gate: `{payload['active_gate_status']}`",
        f"- Stage 3 manifest: `{payload['stage3_manifest_id']}`",
        f"- Decision-source closure: `{payload['source_closure_sha256']}`",
        (
            "- Pending family ready / hypotheses / clusters: "
            f"`{payload['pending_family_ready']} / "
            f"{payload['pending_family_hypotheses']} / "
            f"{payload['pending_family_independent_clusters']}`"
        ),
        (
            "- Active/pending overlap; pending semantic/variant coverage: "
            f"`{payload['active_pending_semantic_overlap']}; "
            f"{payload['pending_family_semantic_batch_coverage']} / "
            f"{payload['pending_family_batch_semantic_hypotheses']}; "
            f"{payload['pending_family_batch_experiment_variants']}`"
        ),
        "- This receipt grants no promotion, Testnet-order, or live-trading authority.",
        "",
        "| Check | Status | Observed | Blocker |",
        "| --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['check']} | {row['status']} | {row['observed']} | {row['blocker']} |"
        for row in checks
    )
    return "\n".join(lines) + "\n"


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_binding(*, root: Path, path: Path) -> dict[str, str]:
    relative = _relative(path, root)
    return {
        "path": relative,
        "sha256": (
            _file_sha256(path)
            if _path_within(path=path, parent=root) and path.is_file()
            else ""
        ),
    }


def _validate_source_closure(*, root: Path, payload: dict[str, Any]) -> list[str]:
    artifacts = payload.get("source_artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != SOURCE_ARTIFACT_KEYS:
        return ["stage4_handoff_source_artifact_set_invalid"]
    expected_paths = _expected_source_paths(root=root, payload=payload)
    if set(expected_paths) != SOURCE_ARTIFACT_KEYS:
        return ["stage4_handoff_source_artifact_set_invalid"]
    blockers: list[str] = []
    expected_closure = sha256(_canonical_json(artifacts).encode("utf-8")).hexdigest()
    if _text(payload.get("source_closure_sha256")) != expected_closure:
        blockers.append("stage4_handoff_source_closure_identity_invalid")
    for name in sorted(SOURCE_ARTIFACT_KEYS):
        binding = artifacts.get(name)
        if not isinstance(binding, dict):
            blockers.append(f"stage4_handoff_source_binding_invalid:{name}")
            continue
        expected_path = expected_paths[name]
        relative = _text(binding.get("path"))
        expected_sha256 = _text(binding.get("sha256"))
        path = root / relative
        if (
            not relative
            or Path(relative).is_absolute()
            or relative != _relative(expected_path, root)
            or not _path_within(path=path, parent=root)
            or not expected_sha256
            or not path.is_file()
            or _file_sha256(path) != expected_sha256
        ):
            blockers.append(f"stage4_handoff_source_binding_invalid:{name}")
    return blockers


def _expected_source_paths(*, root: Path, payload: dict[str, Any]) -> dict[str, Path]:
    active = root / "reports" / "active"
    contract_id = _text(payload.get("active_contract_id"))
    manifest_relative = _text(payload.get("stage3_manifest_immutable_path"))
    reset_receipt_id = _text(payload.get("stage3_reset_receipt_id"))
    protocol_relative = _text(payload.get("stage5_protocol_receipt_path"))
    return {
        "active_contract": active / "registered_research_rerun_contract.json",
        "immutable_contract": (
            root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
        ),
        "registered_source_receipt": (
            root
            / "data"
            / "research"
            / "registered_rerun_source_families"
            / contract_id
            / "receipt.json"
        ),
        "registered_source_matrix": (
            root
            / "data"
            / "research"
            / "registered_rerun_source_families"
            / contract_id
            / "experiment_matrix.csv"
        ),
        "active_gate": active / "registered_research_rerun_gate.json",
        "active_gate_rows": active / "registered_research_rerun_gate.csv",
        "pending_preflight": active / "registered_rerun_family_preflight.json",
        "pending_preflight_rows": active / "registered_rerun_family_preflight.csv",
        "current_hypothesis_batch": active / "current_hypothesis_batch.csv",
        "active_stage3_manifest": active / "corrective_wizard_next_capture_manifest.json",
        "immutable_stage3_manifest": root / manifest_relative,
        "immutable_stage3_reset": (
            root
            / "data"
            / "research"
            / "wizard_reset_readiness"
            / f"{reset_receipt_id}.json"
        ),
        "stage5_protocol_pointer": active / "registered_stage5_protocol.json",
        "immutable_stage5_protocol": root / protocol_relative,
        "proof_scheduler_status": active / "corrective_wizard_proof_scheduler_status.json",
    }


def _path_within(*, path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _semicolon_values(value: Any) -> set[str]:
    return {part.strip() for part in _text(value).split(";") if part.strip()}


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
