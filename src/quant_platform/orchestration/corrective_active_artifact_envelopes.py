"""Immutable, fail-closed decisions for every Phase 00 active artifact."""

from __future__ import annotations

import csv
import json
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_lineage import (
    ACTIVE_ARTIFACT_REGISTRY,
    ArtifactRegistration,
    LineageIssue,
    inspect_registered_artifact,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code

ENVELOPE_SCHEMA_VERSION = "thewiz.phase00.active_artifact_envelope.v1"
MANIFEST_SCHEMA_VERSION = "thewiz.phase00.active_artifact_manifest.v1"
POINTER_SCHEMA_VERSION = "thewiz.phase00.active_artifact_pointer.v1"
IMMUTABLE_ROOT = Path("data/research/phase00_control/active_artifact_lineage")
ACTIVE_POINTER = Path("reports/active/phase00_active_artifact_lineage.json")
ZERO_SHA256 = "0" * 64
MAX_ACTIVE_ARTIFACT_BYTES = 32 * 1024**2
AUTHORITY_FLAGS = (
    "candidate_promotion_authority",
    "paper_trading_authority",
    "testnet_order_authority",
    "live_trading_authority",
    "order_submission_authority",
)
EXPECTED_VALIDATORS = frozenset(
    {
        "aggregate_bundle_validator",
        "blocked_decision_validator",
        "cost_coverage_validator",
        "daily_schedule_validator",
        "integrity_repair_validator",
        "l2_capture_validator",
        "lock_classifier",
        "model_lineage_validator",
        "ownership_and_capture_validator",
        "presentation_classifier",
        "registered_existing_validator",
        "research_snapshot_validator",
        "wizard_mode_domain_validator",
    }
)


@dataclass(frozen=True)
class DomainDecision:
    state: str
    resolution_status: str
    domain_validation_status: str
    blockers: tuple[str, ...]


def build_active_artifact_lineage_bundle(
    *,
    root: Path,
    recorded_at_utc: str,
    source_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    configuration_fingerprint_sha256: str,
) -> dict[str, Any]:
    """Build a complete immutable envelope set without publishing it."""

    root = root.resolve()
    _require_utc(recorded_at_utc)
    fingerprints = {
        "source_fingerprint_sha256": _require_sha256(source_fingerprint_sha256),
        "runtime_fingerprint_sha256": _require_sha256(runtime_fingerprint_sha256),
        "configuration_fingerprint_sha256": _require_sha256(
            configuration_fingerprint_sha256
        ),
    }
    observed_validators = frozenset(
        registration.validator for registration in ACTIVE_ARTIFACT_REGISTRY
    )
    if observed_validators != EXPECTED_VALIDATORS:
        raise ValueError("active artifact domain validator registry is incomplete")

    envelopes: list[dict[str, Any]] = []
    manifest_entries: list[dict[str, Any]] = []
    for registration in ACTIVE_ARTIFACT_REGISTRY:
        envelope = _build_envelope(
            root=root,
            registration=registration,
            recorded_at_utc=recorded_at_utc,
            fingerprints=fingerprints,
        )
        envelope_path = (
            IMMUTABLE_ROOT / "envelopes" / f"{envelope['envelope_id']}.json"
        )
        envelopes.append(
            {
                "path": envelope_path.as_posix(),
                "payload": envelope,
            }
        )
        manifest_entries.append(
            {
                "artifact_path": registration.path,
                "observed_artifact_sha256": (
                    envelope["observed_artifact"]["sha256"]
                    if envelope["observed_artifact"] is not None
                    else ""
                ),
                "envelope_id": envelope["envelope_id"],
                "envelope_path": envelope_path.as_posix(),
                "envelope_file_sha256": _json_file_sha256(envelope),
                "envelope_content_sha256": envelope["content_sha256"],
                "state": envelope["state"],
                "resolution_status": envelope["resolution_status"],
                "domain_validation_status": envelope[
                    "domain_validation_status"
                ],
            }
        )

    manifest_material = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "recorded_at_utc": recorded_at_utc,
        **fingerprints,
        "registry_size": len(ACTIVE_ARTIFACT_REGISTRY),
        "entries": manifest_entries,
        "authority_flags": _zero_authority_flags(),
    }
    manifest_id = "phase00artifactmanifest_" + sha256(
        _canonical_json(manifest_material).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "manifest_id": manifest_id,
        "content_sha256": ZERO_SHA256,
        **manifest_material,
    }
    manifest["content_sha256"] = _content_sha256(manifest)
    manifest_path = IMMUTABLE_ROOT / "manifests" / f"{manifest_id}.json"
    pointer = {
        "schema_version": POINTER_SCHEMA_VERSION,
        "status": "SEALED_FAIL_CLOSED",
        "manifest_id": manifest_id,
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _json_file_sha256(manifest),
        "manifest_content_sha256": manifest["content_sha256"],
        "registry_size": len(ACTIVE_ARTIFACT_REGISTRY),
        "recorded_at_utc": recorded_at_utc,
        **fingerprints,
        "authority_flags": _zero_authority_flags(),
    }
    return {
        "envelopes": envelopes,
        "manifest": manifest,
        "manifest_path": manifest_path.as_posix(),
        "pointer": pointer,
        "pointer_path": ACTIVE_POINTER.as_posix(),
    }


def load_active_artifact_envelope_index(
    root: Path,
) -> tuple[dict[str, dict[str, Any]], tuple[LineageIssue, ...]]:
    """Load one fully validated manifest or return no usable decisions."""

    root = root.resolve()
    pointer_path = root / ACTIVE_POINTER
    try:
        pointer = _load_strict_object(pointer_path, require_single_link=True)
    except (OSError, TypeError, ValueError) as exc:
        if not pointer_path.exists():
            return {}, ()
        return {}, (LineageIssue("ACTIVE_ENVELOPE_POINTER_INVALID", detail=safe_exception_code(exc)),)
    issues = _pointer_issues(root, pointer)
    if issues:
        return {}, tuple(sorted(set(issues)))
    manifest_path = root / str(pointer["manifest_path"])
    try:
        manifest = _load_strict_object(manifest_path, require_single_link=True)
    except (OSError, TypeError, ValueError) as exc:
        return {}, (LineageIssue("ACTIVE_ENVELOPE_MANIFEST_INVALID", detail=safe_exception_code(exc)),)
    issues.extend(_manifest_issues(root, pointer, manifest))
    index: dict[str, dict[str, Any]] = {}
    if not issues:
        for entry in manifest["entries"]:
            envelope_path = root / entry["envelope_path"]
            try:
                envelope = _load_strict_object(
                    envelope_path, require_single_link=True
                )
            except (OSError, TypeError, ValueError) as exc:
                issues.append(
                    LineageIssue(
                        "ACTIVE_ENVELOPE_INVALID",
                        str(entry.get("artifact_path", "")),
                        safe_exception_code(exc),
                    )
                )
                continue
            issues.extend(_envelope_issues(root, entry, envelope))
            artifact_path = str(entry.get("artifact_path", ""))
            if artifact_path in index:
                issues.append(
                    LineageIssue("ACTIVE_ENVELOPE_DUPLICATE_ARTIFACT", artifact_path)
                )
            index[artifact_path] = {
                "entry": dict(entry),
                "envelope": envelope,
            }
    if issues:
        return {}, tuple(sorted(set(issues)))
    return index, ()


def load_artifact_envelope_index_for_manifest(
    root: Path,
    manifest_id: str,
) -> tuple[dict[str, dict[str, Any]], tuple[LineageIssue, ...]]:
    """Validate one content-addressed historical lineage without an active pointer."""

    root = root.resolve()
    if (
        not manifest_id.startswith("phase00artifactmanifest_")
        or len(manifest_id) != 48
    ):
        return {}, (LineageIssue("HISTORICAL_ENVELOPE_MANIFEST_ID_INVALID"),)
    manifest_path = (
        root / IMMUTABLE_ROOT / "manifests" / f"{manifest_id}.json"
    )
    try:
        manifest = _load_strict_object(manifest_path, require_single_link=True)
    except (OSError, TypeError, ValueError) as exc:
        return {}, (
            LineageIssue(
                "HISTORICAL_ENVELOPE_MANIFEST_INVALID",
                detail=safe_exception_code(exc),
            ),
        )
    expected_keys = {
        "manifest_id",
        "content_sha256",
        "schema_version",
        "recorded_at_utc",
        "source_fingerprint_sha256",
        "runtime_fingerprint_sha256",
        "configuration_fingerprint_sha256",
        "registry_size",
        "entries",
        "authority_flags",
    }
    issues: list[LineageIssue] = []
    if set(manifest) != expected_keys:
        issues.append(LineageIssue("HISTORICAL_ENVELOPE_MANIFEST_FIELDS_INVALID"))
    manifest_material = {
        key: manifest.get(key)
        for key in (
            "schema_version",
            "recorded_at_utc",
            "source_fingerprint_sha256",
            "runtime_fingerprint_sha256",
            "configuration_fingerprint_sha256",
            "registry_size",
            "entries",
            "authority_flags",
        )
    }
    expected_id = "phase00artifactmanifest_" + sha256(
        _canonical_json(manifest_material).encode("utf-8")
    ).hexdigest()[:24]
    if manifest.get("manifest_id") != manifest_id or expected_id != manifest_id:
        issues.append(LineageIssue("HISTORICAL_ENVELOPE_MANIFEST_ID_MISMATCH"))
    pointer = {
        "manifest_id": manifest_id,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": _file_sha256(manifest_path),
        "manifest_content_sha256": manifest.get("content_sha256"),
    }
    issues.extend(_manifest_issues(root, pointer, manifest))
    index: dict[str, dict[str, Any]] = {}
    if not issues:
        for entry in manifest["entries"]:
            envelope_path = root / str(entry.get("envelope_path", ""))
            try:
                envelope = _load_strict_object(
                    envelope_path,
                    require_single_link=True,
                )
            except (OSError, TypeError, ValueError) as exc:
                issues.append(
                    LineageIssue(
                        "HISTORICAL_ENVELOPE_INVALID",
                        str(entry.get("artifact_path", "")),
                        safe_exception_code(exc),
                    )
                )
                continue
            issues.extend(_envelope_issues(root, entry, envelope))
            artifact_path = str(entry.get("artifact_path", ""))
            if artifact_path in index:
                issues.append(
                    LineageIssue(
                        "HISTORICAL_ENVELOPE_DUPLICATE_ARTIFACT",
                        artifact_path,
                    )
                )
            index[artifact_path] = {
                "entry": dict(entry),
                "envelope": envelope,
            }
    if issues:
        return {}, tuple(sorted(set(issues)))
    return index, ()


def apply_active_artifact_envelopes(
    rows: list[dict[str, Any]],
    *,
    index: Mapping[str, Mapping[str, Any]],
    load_issues: tuple[LineageIssue, ...] = (),
) -> list[dict[str, Any]]:
    """Apply only exact hash-matched immutable decisions to inventory rows."""

    result: list[dict[str, Any]] = []
    issue_codes = sorted({issue.code for issue in load_issues})
    for supplied in rows:
        row = dict(supplied)
        artifact_path = str(row.get("path", ""))
        binding = index.get(artifact_path)
        row["lineage_envelope_status"] = "MISSING"
        row["lineage_envelope_path"] = ""
        row["lineage_envelope_sha256"] = ""
        row["lineage_envelope_state"] = ""
        if issue_codes:
            row["lineage_envelope_status"] = "INVALID_MANIFEST"
            row["reference_blockers"] = list(
                dict.fromkeys(
                    [*row.get("reference_blockers", []), *issue_codes]
                )
            )
        elif binding is not None:
            entry = binding["entry"]
            envelope = binding["envelope"]
            observed = envelope.get("observed_artifact")
            observed_hash = (
                str(observed.get("sha256", ""))
                if isinstance(observed, dict)
                else ""
            )
            if observed_hash == str(row.get("sha256", "")):
                row["resolution_status"] = envelope["resolution_status"]
                row["domain_validation_status"] = envelope[
                    "domain_validation_status"
                ]
                row["authority_eligible"] = False
                row["lineage_envelope_status"] = "VALID"
                row["lineage_envelope_path"] = entry["envelope_path"]
                row["lineage_envelope_sha256"] = entry[
                    "envelope_file_sha256"
                ]
                row["lineage_envelope_state"] = envelope["state"]
                row["lineage_blockers"] = list(envelope["blockers"])
            else:
                row["lineage_envelope_status"] = "STALE_ARTIFACT_HASH"
                row["resolution_status"] = "UNRESOLVED_STALE_ARTIFACT_ENVELOPE"
                row["domain_validation_status"] = "BLOCKED_STALE_ENVELOPE"
        result.append(row)
    return result


def _build_envelope(
    *,
    root: Path,
    registration: ArtifactRegistration,
    recorded_at_utc: str,
    fingerprints: Mapping[str, str],
) -> dict[str, Any]:
    path = root / registration.path
    observed: dict[str, Any] | None = None
    row: dict[str, Any]
    if path.is_file() and not path.is_symlink():
        observed = {
            "path": registration.path,
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        row = inspect_registered_artifact(root, registration)
    else:
        row = {
            "reference_count": 0,
            "resolved_reference_count": 0,
            "hash_binding_count": 0,
            "hash_match_count": 0,
            "reference_blockers": ["registered_artifact_missing"],
        }
    decision = _domain_decision(
        path=path,
        registration=registration,
        row=row,
        observed=observed,
    )
    target = (
        dict(observed)
        if observed is not None
        and decision.state in {"RESEARCH_ONLY", "EXCLUDED_NON_AUTHORITY"}
        else None
    )
    material = {
        "schema_version": ENVELOPE_SCHEMA_VERSION,
        "recorded_at_utc": recorded_at_utc,
        **fingerprints,
        "artifact_path": registration.path,
        "artifact_role": registration.role,
        "producer": registration.producer,
        "declared_schema": registration.schema,
        "domain_validator": registration.validator,
        "prior_migration_state": registration.migration_state,
        "state": decision.state,
        "resolution_status": decision.resolution_status,
        "domain_validation_status": decision.domain_validation_status,
        "observed_artifact": observed,
        "target": target,
        "reference_summary": {
            "reference_count": int(row.get("reference_count", 0)),
            "resolved_reference_count": int(
                row.get("resolved_reference_count", 0)
            ),
            "hash_binding_count": int(row.get("hash_binding_count", 0)),
            "hash_match_count": int(row.get("hash_match_count", 0)),
            "reference_blockers": sorted(
                {str(value) for value in row.get("reference_blockers", []) if value}
            ),
        },
        "blockers": list(decision.blockers),
        "preserved_original_sha256": (
            observed["sha256"]
            if observed is not None
            and registration.migration_state == "integrity_defect"
            else ""
        ),
        "supersedes_original_without_rehash": bool(
            observed is not None
            and registration.migration_state == "integrity_defect"
        ),
        "authority_flags": _zero_authority_flags(),
    }
    envelope_id = "phase00artifact_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:24]
    envelope = {
        "envelope_id": envelope_id,
        "content_sha256": ZERO_SHA256,
        **material,
    }
    envelope["content_sha256"] = _content_sha256(envelope)
    return envelope


def _domain_decision(
    *,
    path: Path,
    registration: ArtifactRegistration,
    row: Mapping[str, Any],
    observed: Mapping[str, Any] | None,
) -> DomainDecision:
    if registration.validator not in EXPECTED_VALIDATORS:
        return _blocked("DOMAIN_VALIDATOR_UNREGISTERED")
    if observed is None:
        return _blocked("REGISTERED_ARTIFACT_MISSING")
    if registration.validator == "presentation_classifier":
        return _excluded("PRESENTATION_NON_AUTHORITY")
    if registration.validator == "lock_classifier":
        return _excluded("RUNTIME_LOCK_NON_AUTHORITY")
    if registration.validator in {
        "cost_coverage_validator",
        "research_snapshot_validator",
        "aggregate_bundle_validator",
    }:
        csv_blockers = _csv_blockers(path)
        return (
            _blocked(*csv_blockers)
            if csv_blockers
            else _research_only(f"{registration.validator.upper()}_NON_AUTHORITY")
        )
    if registration.validator == "blocked_decision_validator":
        return _validate_blocked_decision(path)
    json_blockers = _json_blockers(path) if path.suffix.lower() == ".json" else ()
    structural_blockers = tuple(
        f"REFERENCE_{str(value).upper()}"
        for value in row.get("reference_blockers", [])
        if value
    )
    if registration.validator == "integrity_repair_validator":
        return _superseded_blocked(
            *(structural_blockers or ("INTEGRITY_REPAIR_REQUIRED",))
        )
    if json_blockers:
        return _blocked(*json_blockers)
    if structural_blockers:
        return _blocked(*structural_blockers)
    blocker_by_validator = {
        "registered_existing_validator": "REGISTERED_DOMAIN_REPLAY_REQUIRED",
        "daily_schedule_validator": "CADENCE_DOMAIN_REPLAY_REQUIRED",
        "l2_capture_validator": "L2_CAPTURE_DOMAIN_REPLAY_REQUIRED",
        "wizard_mode_domain_validator": "WIZARD_MODE_DOMAIN_REPLAY_REQUIRED",
        "ownership_and_capture_validator": "PRODUCER_OWNERSHIP_REPAIR_REQUIRED",
        "model_lineage_validator": "MODEL_LINEAGE_RETRAIN_REQUIRED",
    }
    blocker = blocker_by_validator.get(registration.validator)
    return _blocked(blocker or "DOMAIN_VALIDATOR_FAIL_CLOSED")


def _validate_blocked_decision(path: Path) -> DomainDecision:
    try:
        payload = _load_strict_object(path, require_single_link=False)
    except (OSError, TypeError, ValueError) as exc:
        return _blocked(f"BLOCKED_DECISION_INVALID_{type(exc).__name__.upper()}")
    authority_values = _authority_values(payload)
    if any(value is not False for value in authority_values):
        return DomainDecision(
            state="BLOCKED",
            resolution_status="UNRESOLVED_BLOCKED_DECISION_AUTHORITY_CONFLICT",
            domain_validation_status="BLOCKED_VALIDATION_FAILED",
            blockers=("BLOCKED_DECISION_AUTHORITY_NOT_ZERO",),
        )
    blockers = _payload_blockers(payload)
    if not blockers:
        blockers = ("BLOCKED_DECISION_REASON_MISSING",)
    return DomainDecision(
        state="BLOCKED",
        resolution_status="RESOLVED_IMMUTABLE_BLOCKED",
        domain_validation_status="PASS",
        blockers=blockers,
    )


def _blocked(*blockers: str) -> DomainDecision:
    normalized = tuple(sorted({value for value in blockers if value})) or (
        "FAIL_CLOSED",
    )
    return DomainDecision(
        state="BLOCKED",
        resolution_status="RESOLVED_IMMUTABLE_BLOCKED",
        domain_validation_status="PASS",
        blockers=normalized,
    )


def _superseded_blocked(*blockers: str) -> DomainDecision:
    normalized = tuple(sorted({value for value in blockers if value})) or (
        "INTEGRITY_REPAIR_REQUIRED",
    )
    return DomainDecision(
        state="SUPERSEDED_BLOCKED",
        resolution_status="RESOLVED_IMMUTABLE_SUPERSEDED_BLOCKED",
        domain_validation_status="PASS",
        blockers=normalized,
    )


def _research_only(reason: str) -> DomainDecision:
    return DomainDecision(
        state="RESEARCH_ONLY",
        resolution_status="RESOLVED_IMMUTABLE_RESEARCH_ONLY",
        domain_validation_status="PASS",
        blockers=(reason,),
    )


def _excluded(reason: str) -> DomainDecision:
    return DomainDecision(
        state="EXCLUDED_NON_AUTHORITY",
        resolution_status="EXCLUDED_NON_POINTER",
        domain_validation_status="NOT_APPLICABLE",
        blockers=(reason,),
    )


def _pointer_issues(root: Path, pointer: Mapping[str, Any]) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    if pointer.get("schema_version") != POINTER_SCHEMA_VERSION:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_POINTER_SCHEMA_INVALID"))
    if pointer.get("status") != "SEALED_FAIL_CLOSED":
        issues.append(LineageIssue("ACTIVE_ENVELOPE_POINTER_STATUS_INVALID"))
    path = _safe_relative_path(root, pointer.get("manifest_path"), IMMUTABLE_ROOT)
    if path is None:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_PATH_INVALID"))
    digest = pointer.get("manifest_sha256")
    if not isinstance(digest, str) or not _is_sha256(digest):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_HASH_INVALID"))
    content_digest = pointer.get("manifest_content_sha256")
    if not isinstance(content_digest, str) or not _is_sha256(content_digest):
        issues.append(
            LineageIssue("ACTIVE_ENVELOPE_MANIFEST_CONTENT_HASH_INVALID")
        )
    if not _authority_flags_are_zero(pointer.get("authority_flags")):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_POINTER_AUTHORITY_NONZERO"))
    return issues


def _manifest_issues(
    root: Path, pointer: Mapping[str, Any], manifest: Mapping[str, Any]
) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_SCHEMA_INVALID"))
    if manifest.get("manifest_id") != pointer.get("manifest_id"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_ID_MISMATCH"))
    if _content_sha256(manifest) != manifest.get("content_sha256"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_CONTENT_HASH_MISMATCH"))
    manifest_path = root / str(pointer.get("manifest_path", ""))
    if manifest_path.is_file() and _file_sha256(manifest_path) != pointer.get(
        "manifest_sha256"
    ):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_FILE_HASH_MISMATCH"))
    if manifest.get("content_sha256") != pointer.get("manifest_content_sha256"):
        issues.append(
            LineageIssue("ACTIVE_ENVELOPE_MANIFEST_POINTER_CONTENT_HASH_MISMATCH")
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != len(ACTIVE_ARTIFACT_REGISTRY):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_REGISTRY_COVERAGE_INVALID"))
        return issues
    expected_paths = {registration.path for registration in ACTIVE_ARTIFACT_REGISTRY}
    observed_paths = {
        str(entry.get("artifact_path", ""))
        for entry in entries
        if isinstance(entry, dict)
    }
    if observed_paths != expected_paths or len(observed_paths) != len(entries):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_REGISTRY_PATH_SET_INVALID"))
    if not _authority_flags_are_zero(manifest.get("authority_flags")):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_MANIFEST_AUTHORITY_NONZERO"))
    return issues


def _envelope_issues(
    root: Path, entry: Mapping[str, Any], envelope: Mapping[str, Any]
) -> list[LineageIssue]:
    artifact_path = str(entry.get("artifact_path", ""))
    issues: list[LineageIssue] = []
    registration = next(
        (
            value
            for value in ACTIVE_ARTIFACT_REGISTRY
            if value.path == artifact_path
        ),
        None,
    )
    if registration is None:
        return [LineageIssue("ACTIVE_ENVELOPE_ARTIFACT_UNREGISTERED", artifact_path)]
    if envelope.get("schema_version") != ENVELOPE_SCHEMA_VERSION:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_SCHEMA_INVALID", artifact_path))
    if envelope.get("envelope_id") != entry.get("envelope_id"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_ID_MISMATCH", artifact_path))
    if _content_sha256(envelope) != envelope.get("content_sha256"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_CONTENT_HASH_MISMATCH", artifact_path))
    envelope_path = _safe_relative_path(
        root, entry.get("envelope_path"), IMMUTABLE_ROOT / "envelopes"
    )
    if envelope_path is None:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_PATH_INVALID", artifact_path))
    elif _file_sha256(envelope_path) != entry.get("envelope_file_sha256"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_FILE_HASH_MISMATCH", artifact_path))
    if envelope.get("content_sha256") != entry.get("envelope_content_sha256"):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_ENTRY_HASH_MISMATCH", artifact_path))
    for field, expected in (
        ("artifact_path", registration.path),
        ("artifact_role", registration.role),
        ("producer", registration.producer),
        ("declared_schema", registration.schema),
        ("domain_validator", registration.validator),
        ("prior_migration_state", registration.migration_state),
    ):
        if envelope.get(field) != expected:
            issues.append(
                LineageIssue("ACTIVE_ENVELOPE_REGISTRATION_MISMATCH", artifact_path, field)
            )
    if not _authority_flags_are_zero(envelope.get("authority_flags")):
        issues.append(LineageIssue("ACTIVE_ENVELOPE_AUTHORITY_NONZERO", artifact_path))
    state = envelope.get("state")
    target = envelope.get("target")
    blockers = envelope.get("blockers")
    if state in {"BLOCKED", "SUPERSEDED_BLOCKED"}:
        if target is not None:
            issues.append(LineageIssue("ACTIVE_ENVELOPE_BLOCKED_TARGET_NON_NULL", artifact_path))
        if not isinstance(blockers, list) or not blockers:
            issues.append(LineageIssue("ACTIVE_ENVELOPE_BLOCKERS_MISSING", artifact_path))
    elif state in {"RESEARCH_ONLY", "EXCLUDED_NON_AUTHORITY"}:
        if not isinstance(target, dict) or target != envelope.get("observed_artifact"):
            issues.append(LineageIssue("ACTIVE_ENVELOPE_TARGET_INVALID", artifact_path))
    else:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_STATE_INVALID", artifact_path))
    if state == "SUPERSEDED_BLOCKED":
        observed = envelope.get("observed_artifact")
        preserved = envelope.get("preserved_original_sha256")
        if (
            not isinstance(observed, dict)
            or preserved != observed.get("sha256")
            or envelope.get("supersedes_original_without_rehash") is not True
        ):
            issues.append(
                LineageIssue("ACTIVE_ENVELOPE_ORIGINAL_HASH_NOT_PRESERVED", artifact_path)
            )
    if envelope.get("domain_validation_status") not in {"PASS", "NOT_APPLICABLE"}:
        issues.append(LineageIssue("ACTIVE_ENVELOPE_DOMAIN_VALIDATION_FAILED", artifact_path))
    return issues


def _csv_blockers(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding="utf-8")
        reader = csv.reader(StringIO(text), strict=True)
        rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error):
        return ("CSV_PARSE_FAILED",)
    if not rows or not rows[0] or any(not value.strip() for value in rows[0]):
        return ("CSV_HEADER_INVALID",)
    width = len(rows[0])
    if any(len(row) != width for row in rows[1:]):
        return ("CSV_ROW_WIDTH_MISMATCH",)
    return ()


def _json_blockers(path: Path) -> tuple[str, ...]:
    try:
        _load_strict_object(path, require_single_link=False)
    except (OSError, TypeError, ValueError) as exc:
        return (f"JSON_INVALID_{type(exc).__name__.upper()}",)
    return ()


def _load_strict_object(path: Path, *, require_single_link: bool) -> dict[str, Any]:
    identity = path.lstat()
    if not stat.S_ISREG(identity.st_mode) or path.is_symlink():
        raise ValueError("artifact must be a regular non-symlink file")
    if require_single_link and identity.st_nlink != 1:
        raise ValueError("immutable artifact must have exactly one link")
    if identity.st_size > MAX_ACTIVE_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds strict JSON size limit")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise TypeError("artifact root must be an object")
    _reject_nonfinite_values(payload)
    return payload


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON value: {value}")


def _reject_nonfinite_values(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite_values(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite_values(child)


def _authority_values(payload: Mapping[str, Any]) -> tuple[bool, ...]:
    values: list[bool] = []
    for key, value in _walk_items(payload):
        normalized = key.lower()
        if normalized in AUTHORITY_FLAGS or normalized.endswith("_authorized"):
            values.append(value if isinstance(value, bool) else True)
    return tuple(values) or (False,)


def _payload_blockers(payload: Mapping[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()
    for key, value in _walk_items(payload):
        normalized = key.lower()
        if "blocker" not in normalized and normalized not in {
            "reason",
            "decision_reason",
            "failure_reason",
        }:
            continue
        if isinstance(value, str) and value.strip():
            values.add(value.strip()[:512])
        elif isinstance(value, list):
            values.update(
                str(item).strip()[:512]
                for item in value
                if str(item).strip()
            )
    return tuple(sorted(values))


def _walk_items(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_items(child)


def _safe_relative_path(
    root: Path, supplied: Any, required_prefix: Path
) -> Path | None:
    if not isinstance(supplied, str) or not supplied or "\n" in supplied:
        return None
    candidate = Path(supplied)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (root / candidate).resolve(strict=False)
    prefix = (root / required_prefix).resolve(strict=False)
    if prefix != resolved and prefix not in resolved.parents:
        return None
    if not resolved.is_file() or resolved.is_symlink():
        return None
    return resolved


def _authority_flags_are_zero(value: Any) -> bool:
    return isinstance(value, dict) and all(value.get(key) is False for key in AUTHORITY_FLAGS)


def _zero_authority_flags() -> dict[str, bool]:
    return {key: False for key in AUTHORITY_FLAGS}


def _content_sha256(payload: Mapping[str, Any]) -> str:
    material = dict(payload)
    material["content_sha256"] = ZERO_SHA256
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _json_file_sha256(payload: Mapping[str, Any]) -> str:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str) -> str:
    if not _is_sha256(value):
        raise ValueError("fingerprint must be a lowercase SHA-256 digest")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_utc(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("recorded_at_utc must be UTC")
