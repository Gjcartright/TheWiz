"""Fail-closed Phase 00 decisions for stale mathematical descendants."""

from __future__ import annotations

import csv
import json
import stat
import subprocess
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_active_artifact_envelopes import (
    load_active_artifact_envelope_index,
    load_artifact_envelope_index_for_manifest,
)
from quant_platform.orchestration.corrective_lineage import active_artifact_rows
from quant_platform.orchestration.corrective_phase00_control import (
    ACTIVE_CHECKPOINT,
    DEFAULT_LOCK_WAIT_SECONDS,
    _build_stable_manifest,
    _canonical_json,
    _configuration_fingerprint,
    _phase00_control_publication_authority,
    _stable_manifest_freshness_projection,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    write_immutable_bytes,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    PHASE00_CONTROL_SCOPE,
    _phase00_maintenance_controller_lock,
    read_phase00_maintenance_state,
)

SCHEMA_VERSION = "thewiz.phase00.descendant_control.v1"
MANIFEST_SCHEMA_VERSION = "thewiz.phase00.descendant_manifest.v1"
POINTER_SCHEMA_VERSION = "thewiz.phase00.descendant_pointer.v1"
CONTROL_ROOT = Path("data/research/phase00_control/descendant_controls")
ACTIVE_POINTER = Path("reports/active/phase00_descendant_control.json")
MATH_AUTHORITY = Path("reports/audits/2026-08-20_v2_math_diagnostic_authority.json")
HISTORICAL_IMPACT = Path(
    "reports/audits/2026-08-20_v2_math_diagnostic_historical_impact.csv"
)
ERROR_LEDGER = Path(
    "reports/audits/2026-08-20_v2_math_diagnostic_error_ledger.csv"
)
MAX_SOURCE_BYTES = 32 * 1024**2
ZERO_AUTHORITY = {
    "candidate_promotion_authority": False,
    "training_authority": False,
    "paper_trading_authority": False,
    "testnet_order_authority": False,
    "live_trading_authority": False,
    "order_submission_authority": False,
}
REQUIRED_AUDIT_ARTIFACTS = frozenset(
    {
        "diagnostic_checks",
        "error_ledger",
        "formula_inventory",
        "historical_impact",
        "math_comparison",
        "report",
    }
)
AUDIT_ARTIFACT_PATHS = {
    "diagnostic_checks": "reports/audits/2026-08-20_v2_math_diagnostic_checks.csv",
    "error_ledger": ERROR_LEDGER.as_posix(),
    "formula_inventory": (
        "reports/audits/2026-08-20_v2_math_diagnostic_formula_inventory.csv"
    ),
    "historical_impact": HISTORICAL_IMPACT.as_posix(),
    "math_comparison": (
        "reports/audits/2026-08-20_v2_math_diagnostic_comparison.csv"
    ),
    "report": "reports/audits/2026-08-20_v2_math_diagnostic.md",
}
ACTION_BY_STATUS = {
    "RETAIN": "PRESERVED_INPUT_ONLY",
    "RETAIN_WITH_QUALITY_CHECK": "PRESERVED_INPUT_ONLY_REQUIRES_QUALITY_CHECK",
    "RETAIN_REFERENCE": "PRESERVED_RESEARCH_ONLY",
    "SUPERSEDED": "SUPERSEDED",
    "REEVALUATED_AND_SUPERSEDED": "SUPERSEDED",
    "SUPERSEDE_AND_REBUILD": "INVALIDATED_PENDING_CONTROLLED_REBUILD",
    "RERUN_REQUIRED": "INVALIDATED_PENDING_CONTROLLED_REBUILD",
    "RETRAIN_REQUIRED": "INVALIDATED_PENDING_CONTROLLED_REBUILD",
    "RECOMPUTE_BEFORE_USE": "INVALIDATED_PENDING_CONTROLLED_REBUILD",
    "RETAIN_DIAGNOSTIC": "PRESERVED_RESEARCH_ONLY",
    "RETAIN_NEGATIVE_EVIDENCE": "PRESERVED_RESEARCH_ONLY",
}
DECISION_FIELDS = (
    "decision_id",
    "subject_type",
    "subject_id",
    "artifact_class",
    "prior_status",
    "control_action",
    "reason",
    "required_action",
    "authority_after_audit",
    "source_evidence_path",
    "source_evidence_sha256",
    "preserves_raw_evidence",
    "requires_controlled_rebuild",
    "candidate_promotion_authority",
    "training_authority",
    "paper_trading_authority",
    "testnet_order_authority",
    "live_trading_authority",
    "order_submission_authority",
)


def build_phase00_descendant_invalidation(
    *,
    root: Path,
    now: datetime | None = None,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CommandResult:
    """Invalidate stale descendants while retaining their forensic evidence."""

    root = _canonical_root(root)
    recorded_at = _as_utc(now)
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        maintenance = read_phase00_maintenance_state(root)
        if not maintenance.get("blocking"):
            raise RuntimeError("phase00_descendant_control_requires_active_maintenance")

        checkpoint, checkpoint_manifest = _require_current_checkpoint(
            root, runner=runner
        )
        lineage = _require_complete_lineage(root)
        audit = _load_math_audit_evidence(root)
        decisions = _build_decision_rows(audit)
        decisions_bytes = _csv_bytes(decisions)
        decisions_sha = sha256(decisions_bytes).hexdigest()
        action_counts = _action_counts(decisions)
        source_bindings = audit["source_bindings"]
        source_fingerprint = sha256(
            _canonical_json(source_bindings).encode("utf-8")
        ).hexdigest()
        runtime_fingerprint = sha256(
            _canonical_json(checkpoint_manifest.get("runtime_contracts", [])).encode(
                "utf-8"
            )
        ).hexdigest()
        configuration_fingerprint = _configuration_fingerprint(root)
        material = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": recorded_at.isoformat(),
            "maintenance_id": str(maintenance.get("maintenance_id", "")),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "active_lineage_manifest_id": lineage["manifest_id"],
            "historical_math_authority_sha256": audit[
                "historical_math_authority_sha256"
            ],
            "decisions_sha256": decisions_sha,
            "decision_count": len(decisions),
            "action_counts": action_counts,
        }
        control_id = "phase00descendants_" + sha256(
            _canonical_json(material).encode("utf-8")
        ).hexdigest()[:24]
        output_root = root / CONTROL_ROOT / control_id
        decisions_path = output_root / "decisions.csv"
        manifest_path = output_root / "manifest.json"
        receipt_path = output_root / "receipt.json"
        pointer_path = root / ACTIVE_POINTER
        manifest = {
            **material,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "control_id": control_id,
            "decisions_path": _relative(decisions_path, root),
            "decisions_sha256": decisions_sha,
            "source_bindings": source_bindings,
            "historical_input_hashes": audit["historical_input_hashes"],
            "historical_input_hash_match_count": audit[
                "historical_input_hash_match_count"
            ],
            "historical_input_hash_mismatch_count": audit[
                "historical_input_hash_mismatch_count"
            ],
            "historical_input_hashes_are_current_authority": False,
            "math_incidents_explicitly_controlled": ["MATH-023", "MATH-024"],
            "regenerated_descendant_count": 0,
            "descendant_regeneration_status": (
                "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            ),
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        manifest_file_sha = _json_file_sha256(manifest)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS_STALE_DESCENDANTS_INVALIDATED",
            "control_id": control_id,
            "recorded_at_utc": recorded_at.isoformat(),
            "maintenance_id": material["maintenance_id"],
            "checkpoint_id": material["checkpoint_id"],
            "checkpoint_manifest_sha256": str(
                checkpoint["stable_manifest_file_sha256"]
            ),
            "active_lineage_manifest_id": lineage["manifest_id"],
            "active_lineage_registry_size": lineage["registry_size"],
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": manifest_file_sha,
            "decisions_path": _relative(decisions_path, root),
            "decisions_sha256": decisions_sha,
            "decision_count": len(decisions),
            "action_counts": action_counts,
            "regenerated_descendant_count": 0,
            "descendant_regeneration_status": (
                "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            ),
            "historical_outputs_valid_for_current_decisions": False,
            "current_model_authority": "RESEARCH_ONLY_BLOCKED_PENDING_REBUILD",
            "current_dashboard_order_authority": False,
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        receipt_file_sha = _json_file_sha256(receipt)
        pointer = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "status": receipt["status"],
            "control_id": control_id,
            "recorded_at_utc": recorded_at.isoformat(),
            "receipt_path": _relative(receipt_path, root),
            "receipt_sha256": receipt_file_sha,
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": manifest_file_sha,
            "decisions_path": _relative(decisions_path, root),
            "decisions_sha256": decisions_sha,
            "decision_count": len(decisions),
            "descendant_regeneration_status": (
                "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            ),
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        with _phase00_control_publication_authority(
            root=root,
            run_id=control_id,
            intended_slot_id=material["maintenance_id"],
            source_fingerprint_sha256=source_fingerprint,
            runtime_fingerprint_sha256=runtime_fingerprint,
            configuration_fingerprint_sha256=configuration_fingerprint,
            target_paths=(decisions_path, manifest_path, receipt_path, pointer_path),
        ):
            write_immutable_bytes(
                decisions_path,
                decisions_bytes,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            write_immutable_json(
                manifest_path,
                manifest,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            write_immutable_json(
                receipt_path,
                receipt,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                pointer_path,
                _pretty_json(pointer),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
        observation = observe_phase00_descendant_control(root)
        if observation["validation_status"] != "PASS":
            raise RuntimeError(
                "phase00_descendant_control_postpublication_validation_failed:"
                + ";".join(observation["blockers"])
            )
    return CommandResult(
        paths={
            "descendant_decisions": decisions_path,
            "descendant_manifest": manifest_path,
            "descendant_receipt": receipt_path,
            "active_descendant_control": pointer_path,
        },
        summary=receipt,
    )


def observe_phase00_descendant_control(
    root: Path,
    *,
    allow_superseded_lineage: bool = False,
) -> dict[str, Any]:
    """Validate the active descendant pointer without granting any authority."""

    root = root.resolve()
    pointer_path = root / ACTIVE_POINTER
    if not pointer_path.exists():
        return {
            "validation_status": "NOT_STARTED",
            "status": "NOT_STARTED",
            "descendant_regeneration_status": "NOT_STARTED",
            "blockers": [],
            "authority_flags": dict(ZERO_AUTHORITY),
        }
    blockers: list[str] = []
    try:
        pointer = _load_strict_object(pointer_path)
        _require_schema(pointer, POINTER_SCHEMA_VERSION, "pointer")
        _require_zero_authority(pointer, "pointer")
        control_id = _require_string(pointer, "control_id")
        if not control_id.startswith("phase00descendants_") or len(control_id) != 43:
            raise ValueError("descendant_control_id_invalid")
        receipt_path = _bound_relative_file(root, pointer, "receipt_path")
        manifest_path = _bound_relative_file(root, pointer, "manifest_path")
        decisions_path = _bound_relative_file(root, pointer, "decisions_path")
        expected_root = root / CONTROL_ROOT / control_id
        if receipt_path != expected_root / "receipt.json":
            raise ValueError("descendant_receipt_path_substitution")
        if manifest_path != expected_root / "manifest.json":
            raise ValueError("descendant_manifest_path_substitution")
        if decisions_path != expected_root / "decisions.csv":
            raise ValueError("descendant_decisions_path_substitution")
        _require_hash(receipt_path, pointer, "receipt_sha256")
        _require_hash(manifest_path, pointer, "manifest_sha256")
        _require_hash(decisions_path, pointer, "decisions_sha256")
        receipt = _load_strict_object(receipt_path)
        manifest = _load_strict_object(manifest_path)
        _require_schema(receipt, SCHEMA_VERSION, "receipt")
        _require_schema(manifest, MANIFEST_SCHEMA_VERSION, "manifest")
        _require_zero_authority(receipt, "receipt")
        _require_zero_authority(manifest, "manifest")
        if receipt.get("control_id") != control_id or manifest.get("control_id") != control_id:
            raise ValueError("descendant_control_id_mismatch")
        if pointer.get("status") != "PASS_STALE_DESCENDANTS_INVALIDATED":
            raise ValueError("descendant_pointer_status_invalid")
        if receipt.get("status") != pointer.get("status"):
            raise ValueError("descendant_receipt_status_mismatch")
        _require_cross_binding(pointer, receipt, "manifest_path", "manifest_sha256")
        _require_cross_binding(pointer, receipt, "decisions_path", "decisions_sha256")
        _require_cross_binding(pointer, manifest, "decisions_path", "decisions_sha256")
        expected_regeneration = "INVALIDATED_PENDING_CONTROLLED_REBUILD"
        if any(
            payload.get("descendant_regeneration_status") != expected_regeneration
            for payload in (pointer, receipt, manifest)
        ):
            raise ValueError("descendant_regeneration_status_mismatch")
        decision_count = int(pointer.get("decision_count", -1))
        if decision_count != int(receipt.get("decision_count", -2)) or decision_count != int(
            manifest.get("decision_count", -3)
        ):
            raise ValueError("descendant_decision_count_mismatch")
        decision_rows = _read_csv_rows(decisions_path)
        if decision_count != len(decision_rows) or decision_count != 15:
            raise ValueError("descendant_decision_count_invalid")
        if tuple(decision_rows[0]) != DECISION_FIELDS:
            raise ValueError("descendant_decision_schema_invalid")
        audit = _load_math_audit_evidence(root)
        expected_decisions = _build_decision_rows(audit)
        expected_bytes = _csv_bytes(expected_decisions)
        if decisions_path.read_bytes() != expected_bytes:
            raise ValueError("descendant_decisions_not_reconstructable")
        expected_counts = _action_counts(expected_decisions)
        if any(payload.get("action_counts") != expected_counts for payload in (receipt, manifest)):
            raise ValueError("descendant_action_counts_mismatch")
        if manifest.get("source_bindings") != audit["source_bindings"]:
            raise ValueError("descendant_source_bindings_mismatch")
        if {row.get("subject_id") for row in decision_rows if row.get("subject_type") == "math_incident"} != {
            "MATH-023",
            "MATH-024",
        }:
            raise ValueError("descendant_math_incident_decisions_incomplete")
        for row in decision_rows:
            if any(row.get(field) != "false" for field in ZERO_AUTHORITY):
                raise ValueError("descendant_decision_authority_nonzero")
        material = {
            "schema_version": SCHEMA_VERSION,
            "recorded_at_utc": manifest.get("recorded_at_utc"),
            "maintenance_id": manifest.get("maintenance_id"),
            "checkpoint_id": manifest.get("checkpoint_id"),
            "active_lineage_manifest_id": manifest.get(
                "active_lineage_manifest_id"
            ),
            "historical_math_authority_sha256": manifest.get(
                "historical_math_authority_sha256"
            ),
            "decisions_sha256": manifest.get("decisions_sha256"),
            "decision_count": manifest.get("decision_count"),
            "action_counts": manifest.get("action_counts"),
        }
        expected_control_id = "phase00descendants_" + sha256(
            _canonical_json(material).encode("utf-8")
        ).hexdigest()[:24]
        if expected_control_id != control_id:
            raise ValueError("descendant_control_id_content_mismatch")
        _require_historical_checkpoint_binding(root, receipt)
        lineage_pointer = _load_strict_object(
            root / "reports/active/phase00_active_artifact_lineage.json"
        )
        recorded_lineage_id = str(receipt.get("active_lineage_manifest_id", ""))
        active_lineage_id = str(lineage_pointer.get("manifest_id", ""))
        lineage_mismatch = (
            active_lineage_id != recorded_lineage_id
            or active_lineage_id != manifest.get("active_lineage_manifest_id")
            or int(lineage_pointer.get("registry_size", -1)) != 74
        )
        if lineage_mismatch and not allow_superseded_lineage:
            raise ValueError("descendant_active_lineage_binding_mismatch")
        if lineage_mismatch:
            active_index, active_issues = load_active_artifact_envelope_index(root)
            historical_index, historical_issues = (
                load_artifact_envelope_index_for_manifest(
                    root,
                    recorded_lineage_id,
                )
            )
            if active_issues or len(active_index) != 74:
                raise ValueError("descendant_active_lineage_invalid")
            if historical_issues or len(historical_index) != 74:
                raise ValueError("descendant_historical_lineage_invalid")
            return {
                "validation_status": "PASS_HISTORICAL_ONLY",
                "status": "SUPERSEDED_DESCENDANT_CONTROL_LINEAGE",
                "control_id": control_id,
                "historical_lineage_manifest_id": recorded_lineage_id,
                "active_lineage_manifest_id": active_lineage_id,
                "descendant_regeneration_status": (
                    "SUPERSEDED_REQUIRES_CONTROLLED_REBUILD"
                ),
                "blockers": [],
                "authority_flags": dict(ZERO_AUTHORITY),
            }
    except (OSError, TypeError, ValueError) as exc:
        blockers.append(type(exc).__name__ + ":" + _bounded_exception_code(exc))
        return {
            "validation_status": "BLOCKED",
            "status": "BLOCKED_INVALID_DESCENDANT_CONTROL",
            "descendant_regeneration_status": "BLOCKED_INVALID_CONTROL",
            "blockers": blockers,
            "authority_flags": dict(ZERO_AUTHORITY),
        }
    return {
        "validation_status": "PASS",
        "status": str(pointer["status"]),
        "control_id": str(pointer["control_id"]),
        "receipt_path": str(pointer["receipt_path"]),
        "receipt_sha256": str(pointer["receipt_sha256"]),
        "manifest_path": str(pointer["manifest_path"]),
        "manifest_sha256": str(pointer["manifest_sha256"]),
        "decisions_path": str(pointer["decisions_path"]),
        "decisions_sha256": str(pointer["decisions_sha256"]),
        "decision_count": int(pointer["decision_count"]),
        "descendant_regeneration_status": str(
            pointer["descendant_regeneration_status"]
        ),
        "blockers": [],
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _require_current_checkpoint(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_path = root / ACTIVE_CHECKPOINT
    checkpoint = _load_strict_object(checkpoint_path)
    if (
        checkpoint.get("status") != "PASS_QUIESCED_BASELINE"
        or checkpoint.get("checkpoint_capture_complete") is not True
        or checkpoint.get("snapshot_stable") is not True
        or checkpoint.get("blockers") != []
    ):
        raise RuntimeError("phase00_descendant_control_requires_pass_checkpoint")
    receipt_path = _bound_relative_file(root, checkpoint, "checkpoint_receipt_path")
    checkpoint_id = _require_string(checkpoint, "checkpoint_id")
    expected_receipt = (
        root
        / "data/research/phase00_control/checkpoints"
        / f"{checkpoint_id}.json"
    )
    if receipt_path != expected_receipt:
        raise RuntimeError("phase00_descendant_checkpoint_receipt_path_invalid")
    if _load_strict_object(receipt_path) != checkpoint:
        raise RuntimeError("phase00_descendant_checkpoint_receipt_mismatch")
    manifest_path = _bound_relative_file(root, checkpoint, "stable_manifest_path")
    expected_manifest = (
        root
        / "data/research/phase00_control/manifests"
        / f"{checkpoint_id}.json"
    )
    if manifest_path != expected_manifest:
        raise RuntimeError("phase00_descendant_checkpoint_manifest_path_invalid")
    _require_hash(manifest_path, checkpoint, "stable_manifest_file_sha256")
    manifest = _load_strict_object(manifest_path)
    current_manifest = _build_stable_manifest(root, runner=runner)
    current_sha = sha256(
        _canonical_json(
            _stable_manifest_freshness_projection(current_manifest)
        ).encode("utf-8")
    ).hexdigest()
    if current_sha != checkpoint.get("freshness_manifest_sha256"):
        raise RuntimeError("phase00_descendant_control_checkpoint_is_stale")
    if current_manifest.get("blockers") != []:
        raise RuntimeError("phase00_descendant_control_current_manifest_blocked")
    return checkpoint, manifest


def _require_complete_lineage(root: Path) -> dict[str, Any]:
    index, issues = load_active_artifact_envelope_index(root)
    if issues or len(index) != 74:
        codes = sorted({issue.code for issue in issues})
        raise RuntimeError(
            "phase00_descendant_control_lineage_invalid:" + ";".join(codes)
        )
    rows = active_artifact_rows(root)
    invalid = [
        str(row.get("path", ""))
        for row in rows
        if str(row.get("resolution_status", "")).startswith("UNRESOLVED")
        or row.get("domain_validation_status") not in {"PASS", "NOT_APPLICABLE"}
        or row.get("artifact_role") == "unregistered"
        or row.get("authority_eligible") is not False
    ]
    if len(rows) != 74 or invalid:
        raise RuntimeError(
            "phase00_descendant_control_lineage_incomplete:" + ";".join(invalid)
        )
    pointer = _load_strict_object(
        root / "reports/active/phase00_active_artifact_lineage.json"
    )
    if pointer.get("status") != "SEALED_FAIL_CLOSED" or int(
        pointer.get("registry_size", -1)
    ) != 74:
        raise RuntimeError("phase00_descendant_control_lineage_pointer_invalid")
    return pointer


def _load_math_audit_evidence(root: Path) -> dict[str, Any]:
    authority_path = root / MATH_AUTHORITY
    authority = _load_strict_object(authority_path)
    _require_schema(authority, "thewiz.v2_math_diagnostic.v2", "math_authority")
    if authority.get("promotion_authority") is not False or authority.get(
        "paper_or_live_authority"
    ) is not False:
        raise ValueError("historical_math_authority_is_not_fail_closed")
    artifacts = authority.get("artifacts")
    if not isinstance(artifacts, dict) or frozenset(artifacts) != REQUIRED_AUDIT_ARTIFACTS:
        raise ValueError("historical_math_artifact_set_invalid")
    source_bindings = [
        {
            "name": "math_authority",
            "path": MATH_AUTHORITY.as_posix(),
            "sha256": _file_sha256(authority_path),
        }
    ]
    for name in sorted(artifacts):
        binding = artifacts[name]
        if not isinstance(binding, dict):
            raise TypeError("historical_math_artifact_binding_invalid")
        path = _bound_relative_file(root, binding, "path")
        if _relative(path, root) != AUDIT_ARTIFACT_PATHS[name]:
            raise ValueError("historical_math_artifact_path_substitution")
        _require_hash(path, binding, "sha256")
        source_bindings.append(
            {"name": name, "path": _relative(path, root), "sha256": _file_sha256(path)}
        )
    impact_path = root / HISTORICAL_IMPACT
    impact_rows = _read_csv_rows(impact_path)
    if len(impact_rows) != 13 or any(
        not row.get("artifact_class") or row.get("status") not in ACTION_BY_STATUS
        for row in impact_rows
    ):
        raise ValueError("historical_impact_contract_invalid")
    ledger_rows = _read_csv_rows(root / ERROR_LEDGER)
    incidents = {row.get("incident_id"): row for row in ledger_rows}
    for incident_id in ("MATH-023", "MATH-024"):
        row = incidents.get(incident_id)
        if (
            row is None
            or row.get("severity") != "P0"
            or row.get("status") != "FIXED_THIS_AUDIT"
            or not row.get("invalidated_artifacts")
            or not row.get("wrong_or_inconsistent_math")
            or not row.get("remaining_action")
        ):
            raise ValueError(f"historical_incident_contract_invalid:{incident_id}")
    historical_inputs = authority.get("input_hashes")
    if not isinstance(historical_inputs, dict):
        raise TypeError("historical_input_hashes_invalid")
    input_rows: list[dict[str, Any]] = []
    for relative, expected in sorted(historical_inputs.items()):
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("historical_input_hash_invalid")
        path = _safe_relative_path(root, str(relative))
        observed = _file_sha256(path) if _is_single_regular_file(path) else ""
        input_rows.append(
            {
                "path": str(relative),
                "historical_sha256": str(expected),
                "current_sha256": observed,
                "status": "MATCH" if observed == expected else "STALE_OR_MISSING",
            }
        )
    return {
        "authority": authority,
        "historical_math_authority_sha256": _file_sha256(authority_path),
        "source_bindings": source_bindings,
        "historical_impact_rows": impact_rows,
        "incidents": {key: incidents[key] for key in ("MATH-023", "MATH-024")},
        "historical_input_hashes": input_rows,
        "historical_input_hash_match_count": sum(
            row["status"] == "MATCH" for row in input_rows
        ),
        "historical_input_hash_mismatch_count": sum(
            row["status"] != "MATCH" for row in input_rows
        ),
    }


def _build_decision_rows(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_sha = str(audit["historical_math_authority_sha256"])
    rows: list[dict[str, Any]] = []
    for source in audit["historical_impact_rows"]:
        action = ACTION_BY_STATUS[str(source["status"])]
        row = {
            "subject_type": "artifact_class",
            "subject_id": str(source["artifact_class"]),
            "artifact_class": str(source["artifact_class"]),
            "prior_status": str(source["status"]),
            "control_action": action,
            "reason": str(source["reason"]),
            "required_action": str(source["required_action"]),
            "authority_after_audit": str(source["authority_after_audit"]),
            "source_evidence_path": HISTORICAL_IMPACT.as_posix(),
            "source_evidence_sha256": next(
                binding["sha256"]
                for binding in audit["source_bindings"]
                if binding["name"] == "historical_impact"
            ),
            "preserves_raw_evidence": action.startswith("PRESERVED") or action == "SUPERSEDED",
            "requires_controlled_rebuild": action == "INVALIDATED_PENDING_CONTROLLED_REBUILD",
            **ZERO_AUTHORITY,
        }
        row["decision_id"] = _decision_id(row)
        rows.append(row)
    for incident_id in ("MATH-023", "MATH-024"):
        incident = audit["incidents"][incident_id]
        row = {
            "subject_type": "math_incident",
            "subject_id": incident_id,
            "artifact_class": str(incident["invalidated_artifacts"]),
            "prior_status": str(incident["status"]),
            "control_action": "INVALIDATED_PENDING_CONTROLLED_REBUILD",
            "reason": str(incident["wrong_or_inconsistent_math"]),
            "required_action": str(incident["remaining_action"]),
            "authority_after_audit": "none_until_controlled_rebuild",
            "source_evidence_path": ERROR_LEDGER.as_posix(),
            "source_evidence_sha256": next(
                binding["sha256"]
                for binding in audit["source_bindings"]
                if binding["name"] == "error_ledger"
            ),
            "preserves_raw_evidence": True,
            "requires_controlled_rebuild": True,
            **ZERO_AUTHORITY,
        }
        row["decision_id"] = _decision_id(row)
        rows.append(row)
    if len(rows) != 15 or len({row["decision_id"] for row in rows}) != 15:
        raise ValueError("descendant_decision_set_invalid")
    if not source_sha:
        raise ValueError("historical_math_authority_hash_missing")
    return sorted(rows, key=lambda row: (row["subject_type"], row["subject_id"]))


def _decision_id(row: Mapping[str, Any]) -> str:
    material = {key: row[key] for key in row if key != "decision_id"}
    return "phase00decision_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:24]


def _action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(
        sorted(Counter(str(row["control_action"]) for row in rows).items())
    )


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=DECISION_FIELDS, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: ("true" if value is True else "false" if value is False else value)
                for key, value in row.items()
            }
        )
    return buffer.getvalue().encode("utf-8")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    data = _read_single_regular_bytes(path)
    text = data.decode("utf-8")
    if "\x00" in text:
        raise ValueError("csv_contains_nul")
    reader = csv.DictReader(StringIO(text, newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("csv_header_invalid")
    return [dict(row) for row in reader]


def _load_strict_object(path: Path) -> dict[str, Any]:
    payload = json.loads(
        _read_single_regular_bytes(path).decode("utf-8"),
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, dict):
        raise TypeError("json_root_not_object")
    return payload


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"json_nonfinite_constant:{value}")


def _read_single_regular_bytes(path: Path) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError("source_not_single_regular_file")
    if metadata.st_size > MAX_SOURCE_BYTES:
        raise ValueError("source_too_large")
    return path.read_bytes()


def _is_single_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _bound_relative_file(root: Path, payload: Mapping[str, Any], key: str) -> Path:
    value = _require_string(payload, key)
    path = _safe_relative_path(root, value)
    if not _is_single_regular_file(path):
        raise ValueError(f"{key}_not_single_regular_file")
    return path


def _safe_relative_path(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError("unsafe_relative_path")
    path = root / relative
    if path.resolve().is_relative_to(root) is False:
        raise ValueError("path_escapes_root")
    return path


def _require_hash(path: Path, payload: Mapping[str, Any], key: str) -> None:
    expected = _require_string(payload, key)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{key}_invalid")
    if _file_sha256(path) != expected:
        raise ValueError(f"{key}_mismatch")


def _require_cross_binding(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    path_key: str,
    hash_key: str,
) -> None:
    if left.get(path_key) != right.get(path_key) or left.get(hash_key) != right.get(
        hash_key
    ):
        raise ValueError(f"descendant_cross_binding_mismatch:{path_key}")


def _require_historical_checkpoint_binding(
    root: Path, receipt: Mapping[str, Any]
) -> None:
    checkpoint_id = _require_string(receipt, "checkpoint_id")
    checkpoint_path = (
        root
        / "data/research/phase00_control/checkpoints"
        / f"{checkpoint_id}.json"
    )
    checkpoint = _load_strict_object(checkpoint_path)
    if (
        checkpoint.get("checkpoint_id") != checkpoint_id
        or checkpoint.get("status") != "PASS_QUIESCED_BASELINE"
        or checkpoint.get("checkpoint_capture_complete") is not True
        or checkpoint.get("stable_manifest_file_sha256")
        != receipt.get("checkpoint_manifest_sha256")
    ):
        raise ValueError("descendant_checkpoint_binding_invalid")


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise ValueError(f"{label}_schema_invalid")


def _require_zero_authority(payload: Mapping[str, Any], label: str) -> None:
    flags = payload.get("authority_flags")
    if not isinstance(flags, dict) or flags != ZERO_AUTHORITY:
        raise ValueError(f"{label}_authority_nonzero")


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}_missing")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_file_sha256(payload: Mapping[str, Any]) -> str:
    return sha256(_pretty_json(payload).encode("utf-8")).hexdigest()


def _pretty_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _canonical_root(root: Path) -> Path:
    resolved = root.resolve()
    if not (resolved / ".git").exists():
        raise ValueError("phase00 descendant root is not a Git repository")
    return resolved


def _as_utc(value: datetime | None) -> datetime:
    observed = value or datetime.now(UTC)
    if observed.tzinfo is None:
        raise ValueError("phase00 descendant timestamp must be timezone aware")
    return observed.astimezone(UTC)


def _bounded_exception_code(exc: BaseException) -> str:
    code = type(exc).__name__
    primitive = next(
        (
            value
            for value in exc.args
            if isinstance(value, (str, int, float, bool)) and value is not None
        ),
        "",
    )
    if primitive != "":
        safe = "".join(
            character if character.isalnum() or character in "._:-" else "_"
            for character in str(primitive)[:160]
        )
        return f"{code}:{safe}"
    return code
