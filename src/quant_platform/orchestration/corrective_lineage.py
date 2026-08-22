"""Phase 00 artifact inventory and fail-closed Gate 00F immutable lineage."""

from __future__ import annotations

import fcntl
import json
import os
import re
import stat
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_redaction import safe_exception_code

ACTIVE_PREFIX = "reports/active"
POINTER_NAME_TOKENS = ("authority", "pointer", "status", "checkpoint")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{2,255}$")
REASON_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")
ZERO_SHA256 = "0" * 64

CANONICAL_RECEIPT_TYPES = (
    "source_mapping_receipt",
    "input_manifest",
    "run_manifest",
    "artifact_manifest",
    "data_quality_receipt",
    "authority_receipt",
)
CANONICAL_SCHEMA_VERSIONS = {
    artifact_type: f"thewiz.gate00f.{artifact_type}.v1" for artifact_type in CANONICAL_RECEIPT_TYPES
}
CANONICAL_PARENT_TYPES: dict[str, str | None] = {
    "source_mapping_receipt": None,
    "input_manifest": "source_mapping_receipt",
    "run_manifest": "input_manifest",
    "artifact_manifest": "run_manifest",
    "data_quality_receipt": "artifact_manifest",
    "authority_receipt": "data_quality_receipt",
}
CANONICAL_CHAIN_FROM_AUTHORITY = tuple(reversed(CANONICAL_RECEIPT_TYPES))
VERSION_BINDING_NAMES = (
    "formula",
    "cost",
    "dataset",
    "runtime",
    "source",
    "policy",
)
LINEAGE_STATES = {"ACTIVE", "RESEARCH_ONLY", "INVALIDATED", "SUPERSEDED"}
INVALIDATION_EVENT_SCHEMA_VERSION = "thewiz.gate00f.invalidation_event.v1"
REGENERATION_EVENT_SCHEMA_VERSION = "thewiz.gate00f.regeneration_event.v1"
CONTROL_OUTPUT_PATHS = {
    "reports/active/phase00_active_artifact_lineage.json",
    "reports/active/phase00_control_checkpoint.json",
    "reports/active/phase00_descendant_control.json",
    "reports/active/phase00_closure.json",
    "reports/active/phase00_acceptance_matrix.csv",
    "reports/active/phase00_fault_catalog.csv",
    "reports/active/phase00_maintenance.json",
    "reports/active/phase00_source_evidence_index.csv",
}


@dataclass(frozen=True)
class ArtifactRegistration:
    path: str
    role: str
    producer: str
    schema: str
    validator: str
    migration_state: str
    multi_path_fields: tuple[str, ...] = ()


@dataclass(frozen=True, order=True)
class LineageIssue:
    """One deterministic fail-closed Gate 00F validation issue."""

    code: str
    artifact_id: str = ""
    detail: str = ""


@dataclass(frozen=True)
class LineageValidationResult:
    """Complete structural and authority result for one immutable DAG."""

    valid: bool
    authority_eligible: bool
    issues: tuple[LineageIssue, ...]
    authority_artifact_ids: tuple[str, ...]
    root_artifact_ids: tuple[str, ...]
    topological_order: tuple[str, ...]
    invalidated_artifact_ids: tuple[str, ...]
    regenerated_artifact_ids: tuple[str, ...]

    @property
    def blocker_codes(self) -> tuple[str, ...]:
        return tuple(sorted({issue.code for issue in self.issues}))

    def require_authority(self) -> None:
        """Raise unless the requested authority chain passed every Gate 00F check."""

        if not self.authority_eligible:
            raise LineageValidationError(self.issues)


class LineageValidationError(ValueError):
    """Raised when a caller asks a blocked lineage to confer authority."""

    def __init__(self, issues: Sequence[LineageIssue] | str):
        if isinstance(issues, str):
            self.issues = (LineageIssue("LINEAGE_VALIDATION_ERROR", detail=issues),)
        else:
            self.issues = tuple(issues)
        message = "; ".join(
            f"{issue.code}:{issue.artifact_id}:{issue.detail}" for issue in self.issues
        )
        super().__init__(message or "Gate 00F lineage validation failed")


def _paths(*names: str) -> tuple[str, ...]:
    return tuple(f"{ACTIVE_PREFIX}/{name}" for name in names)


def _group(
    paths: tuple[str, ...],
    *,
    role: str,
    producer: str,
    schema: str,
    validator: str,
    migration_state: str,
    multi_path_fields: tuple[str, ...] = (),
) -> tuple[ArtifactRegistration, ...]:
    return tuple(
        ArtifactRegistration(
            path=path,
            role=role,
            producer=producer,
            schema=schema,
            validator=validator,
            migration_state=migration_state,
            multi_path_fields=multi_path_fields,
        )
        for path in paths
    )


ACTIVE_ARTIFACT_REGISTRY = (
    *_group(
        _paths(
            "daily_research_authority.json",
            "daily_research_plan_authority.json",
            "wizard_dynamic_v2_holdout_status.json",
            "wizard_dynamic_v2_proof_refresh_status.json",
            "wizard_ou_v2_holdout_status.json",
            "wizard_ou_v3_holdout_status.json",
            "wizard_ou_v3_proof_refresh_status.json",
            "wizard_ou_v4_holdout_status.json",
            "wizard_ou_v4_proof_refresh_status.json",
            "wizard_ou_v5_holdout_status.json",
            "wizard_ou_v5_proof_refresh_status.json",
            "wizard_ou_v6_holdout_status.json",
            "wizard_ou_v6_proof_refresh_status.json",
        ),
        role="authority_pointer",
        producer="registered_corrective_authority_publishers",
        schema="producer_declared_json",
        validator="registered_existing_validator",
        migration_state="resolved_existing_parent",
    ),
    *_group(
        _paths(
            "canonical_program_status.json",
            "corrective_l2_readiness_refresh_status.json",
            "corrective_wizard_proof_launcher_status.json",
            "corrective_wizard_proof_scheduler_execution_status.json",
            "corrective_wizard_proof_scheduler_observation_status.json",
            "corrective_wizard_proof_scheduler_status.json",
            "hyperliquid_pair_cost_bundle_pointer.json",
        ),
        role="status_pointer",
        producer="canonical_scheduler_and_cost_publishers",
        schema="producer_declared_json",
        validator="registered_existing_validator",
        migration_state="resolved_existing_parent",
    ),
    *_group(
        _paths("hyperliquid_cost_collection_status.csv"),
        role="analytical_view",
        producer="hyperliquid_cost_collection",
        schema="implicit_csv",
        validator="cost_coverage_validator",
        migration_state="resolved_existing_parent",
    ),
    *_group(
        _paths(
            "canonical_program_status.md",
            "orchestrator_run_status.md",
            "seven_stage_goal_checkpoint.md",
            "v2_implementation_checkpoint_2026-08-07.md",
            "wizard_hyperliquid_checkpoint_2026-08-07.md",
            "wizard_live_journal_status.md",
        ),
        role="presentation",
        producer="presentation_renderers",
        schema="markdown_presentation",
        validator="presentation_classifier",
        migration_state="excluded_non_pointer",
    ),
    *_group(
        _paths("corrective_wizard_proof_launcher_status.json.lock"),
        role="lock",
        producer="wizard_proof_launcher",
        schema="runtime_lock",
        validator="lock_classifier",
        migration_state="excluded_non_pointer",
    ),
    *_group(
        _paths(
            "current_wizard_hyperliquid_concentration_status.csv",
            "current_wizard_hyperliquid_daily_plan_status.csv",
            "current_wizard_hyperliquid_daily_run_status.csv",
            "current_wizard_hyperliquid_leverage_status.csv",
            "current_wizard_hyperliquid_regime_status.csv",
            "current_wizard_hyperliquid_replay_pair_status.csv",
            "current_wizard_hyperliquid_robustness_status.csv",
            "current_wizard_hyperliquid_walkforward_status.csv",
            "current_wizard_pair_detail_status.csv",
        ),
        role="analytical_view",
        producer="current_wizard_hyperliquid_pipeline",
        schema="implicit_csv",
        validator="research_snapshot_validator",
        migration_state="unsealed_analytical_view",
    ),
    *_group(
        _paths(
            "exhaustive_wizard_hyperliquid_concentration_status.csv",
            "exhaustive_wizard_hyperliquid_leverage_status.csv",
            "exhaustive_wizard_hyperliquid_regime_status.csv",
            "exhaustive_wizard_hyperliquid_robustness_status.csv",
            "exhaustive_wizard_hyperliquid_walkforward_status.csv",
        ),
        role="analytical_view",
        producer="exhaustive_wizard_hyperliquid_pipeline",
        schema="implicit_csv",
        validator="research_snapshot_validator",
        migration_state="unsealed_analytical_view",
    ),
    *_group(
        _paths("daily_schedule_plan_status.csv", "daily_schedule_status.csv"),
        role="analytical_view",
        producer="corrective_daily_scheduler",
        schema="implicit_csv",
        validator="daily_schedule_validator",
        migration_state="unsealed_schedule_view",
    ),
    *_group(
        _paths(
            "agent_authority_inventory.csv",
            "corrective_plan_phase_status.csv",
            "dashboard_refresh_status.csv",
            "hyperliquid_authority_state.csv",
            "orchestrator_run_status.csv",
            "seven_stage_goal_checkpoint.csv",
            "wizard_mode_authority.csv",
            "wizard_parity_capture_status.csv",
            "workflow_rl_shadow_status.csv",
        ),
        role="analytical_view",
        producer="aggregate_status_publishers",
        schema="implicit_csv",
        validator="aggregate_bundle_validator",
        migration_state="unsealed_aggregate",
    ),
    *_group(
        _paths("corrective_l2_capture_status.json"),
        role="status_pointer",
        producer="corrective_l2_scheduler",
        schema="thewiz.corrective_l2_scheduler.v1",
        validator="l2_capture_validator",
        migration_state="partial_decision",
    ),
    *_group(
        _paths(
            "corrective_artifact_retention_status.json",
            "live_input_parity_capture_status.json",
            "wizard_copula_behavioral_status.json",
        ),
        role="status_pointer",
        producer="retention_parity_and_copula_publishers",
        schema="producer_declared_json",
        validator="integrity_repair_validator",
        migration_state="integrity_defect",
    ),
    *_group(
        _paths(
            "wizard_copula_behavioral_v2_status.json",
            "wizard_credit_ledger_status.json",
            "wizard_dynamic_v2_activation_status.json",
            "wizard_ou_v3_activation_status.json",
            "wizard_ou_v4_activation_status.json",
            "wizard_ou_v5_activation_status.json",
            "wizard_ou_v6_activation_status.json",
            "wizard_ou_v6_capture_status.json",
        ),
        role="status_pointer",
        producer="wizard_mode_decision_publishers",
        schema="producer_declared_json",
        validator="wizard_mode_domain_validator",
        migration_state="unsealed_decision",
    ),
    *_group(
        _paths(
            "corrective_git_checkpoint.json",
            "crypto_wizards_inspector_status.json",
            "wizard_ou_v3_capture_status.json",
            "wizard_ou_v4_capture_status.json",
            "wizard_ou_v5_capture_status.json",
        ),
        role="status_pointer",
        producer="ownership_to_be_registered",
        schema="producer_declared_json",
        validator="ownership_and_capture_validator",
        migration_state="ownership_unresolved",
    ),
    *_group(
        _paths("model_authority_status.json"),
        role="authority_pointer",
        producer="agent_learning_governance",
        schema="thewiz.agent_learning_governance.v1",
        validator="model_lineage_validator",
        migration_state="ownership_unresolved",
        multi_path_fields=("evidence_path",),
    ),
    *_group(
        _paths(
            "live_canary_executor_status.json",
            "registered_learning_research_status.json",
            "stage6_release_status.json",
        ),
        role="blocked_decision",
        producer="release_and_learning_governance",
        schema="producer_declared_json",
        validator="blocked_decision_validator",
        migration_state="blocked_unsealed",
    ),
)

_REGISTRY_BY_PATH = {entry.path: entry for entry in ACTIVE_ARTIFACT_REGISTRY}
if len(ACTIVE_ARTIFACT_REGISTRY) != 74 or len(_REGISTRY_BY_PATH) != 74:
    raise RuntimeError("Phase 00 active artifact registry must contain 74 unique entries")


def active_artifact_rows(root: Path) -> list[dict[str, Any]]:
    """Inspect all registered candidates and fail closed on unknown candidates."""

    from quant_platform.orchestration.corrective_active_artifact_envelopes import (
        apply_active_artifact_envelopes,
        load_active_artifact_envelope_index,
    )

    root = root.resolve()
    rows = [
        inspect_registered_artifact(root, entry)
        for entry in ACTIVE_ARTIFACT_REGISTRY
        if (root / entry.path).is_file()
    ]
    active = root / ACTIVE_PREFIX
    if active.is_dir():
        for path in sorted(active.iterdir()):
            relative = _relative(path, root)
            if (
                path.is_file()
                and _looks_like_pointer_candidate(path.name)
                and relative not in CONTROL_OUTPUT_PATHS
                and relative not in _REGISTRY_BY_PATH
            ):
                rows.append(_unregistered_row(root, path))
    envelope_index, envelope_issues = load_active_artifact_envelope_index(root)
    rows = apply_active_artifact_envelopes(
        rows,
        index=envelope_index,
        load_issues=envelope_issues,
    )
    return sorted(rows, key=lambda row: str(row["path"]))


def inspect_registered_artifact(root: Path, registration: ArtifactRegistration) -> dict[str, Any]:
    path = root / registration.path
    references = (
        resolve_json_path_references(root, path, registration)
        if path.suffix.lower() == ".json"
        else []
    )
    reference_blockers = [
        str(reference["blocker"]) for reference in references if reference["blocker"]
    ]
    migration = registration.migration_state
    if migration == "excluded_non_pointer":
        resolution = "EXCLUDED_NON_POINTER"
        domain_status = "NOT_APPLICABLE"
    elif migration == "resolved_existing_parent" and not reference_blockers:
        resolution = "RESOLVED_STRUCTURAL" if references else "RESOLVED_CLASSIFIED"
        domain_status = "NOT_EVALUATED"
    else:
        resolution = f"UNRESOLVED_{migration.upper()}"
        domain_status = "BLOCKED_NOT_MIGRATED"
    return {
        "path": registration.path,
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "artifact_role": registration.role,
        "producer": registration.producer,
        "schema": registration.schema,
        "validator": registration.validator,
        "migration_state": registration.migration_state,
        "reference_count": len(references),
        "resolved_reference_count": sum(not reference["blocker"] for reference in references),
        "hash_binding_count": sum(
            reference["hash_status"] != "NOT_PROVIDED" for reference in references
        ),
        "hash_match_count": sum(reference["hash_status"] == "MATCH" for reference in references),
        "reference_blockers": reference_blockers,
        "resolution_status": resolution,
        "domain_validation_status": domain_status,
        "authority_eligible": False,
        "content_captured": False,
    }


def resolve_json_path_references(
    root: Path,
    path: Path,
    registration: ArtifactRegistration,
) -> list[dict[str, Any]]:
    """Resolve only schema-declared path fields and their sibling hash bindings."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [
            {
                "field": "",
                "path_sha256": "",
                "inside_root": False,
                "exists": False,
                "hash_status": "NOT_PROVIDED",
                "blocker": "json_payload_unreadable",
            }
        ]
    references: list[dict[str, Any]] = []
    _collect_path_references(
        root=root,
        value=payload,
        field_path=(),
        references=references,
        multi_path_fields=set(registration.multi_path_fields),
    )
    return references


def _collect_path_references(
    *,
    root: Path,
    value: Any,
    field_path: tuple[str, ...],
    references: list[dict[str, Any]],
    multi_path_fields: set[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*field_path, key_text)
            if isinstance(child, str) and _is_declared_path_field(key_text):
                values = (
                    [part.strip() for part in child.split(";")]
                    if key_text in multi_path_fields
                    else [child.strip()]
                )
                expected_hash = _sibling_expected_hash(value, key_text)
                for index, candidate in enumerate(values):
                    if candidate:
                        suffix = f"[{index}]" if len(values) > 1 else ""
                        references.append(
                            _reference_row(
                                root=root,
                                field=".".join(child_path) + suffix,
                                candidate_text=candidate,
                                expected_hash=(expected_hash if len(values) == 1 else ""),
                            )
                        )
            else:
                _collect_path_references(
                    root=root,
                    value=child,
                    field_path=child_path,
                    references=references,
                    multi_path_fields=multi_path_fields,
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_path_references(
                root=root,
                value=child,
                field_path=(*field_path, str(index)),
                references=references,
                multi_path_fields=multi_path_fields,
            )


def _reference_row(
    *, root: Path, field: str, candidate_text: str, expected_hash: str
) -> dict[str, Any]:
    invalid_text = "://" in candidate_text or "\n" in candidate_text or "=" in candidate_text
    candidate = Path(candidate_text)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve(strict=False)
    inside = not invalid_text and (resolved == root or root in resolved.parents)
    exists = inside and resolved.exists()
    hash_status = "NOT_PROVIDED"
    if expected_hash:
        normalized_hash = expected_hash.strip().lower()
        if not SHA256_PATTERN.fullmatch(normalized_hash):
            hash_status = "INVALID_EXPECTED_HASH"
        elif not exists:
            hash_status = "TARGET_MISSING"
        elif not resolved.is_file():
            hash_status = "TARGET_NOT_FILE"
        else:
            hash_status = "MATCH" if _file_sha256(resolved) == normalized_hash else "MISMATCH"
    blocker = ""
    if invalid_text:
        blocker = "invalid_path_text"
    elif not inside:
        blocker = "path_outside_root"
    elif not exists:
        blocker = "path_missing"
    elif hash_status not in {"NOT_PROVIDED", "MATCH"}:
        blocker = f"hash_{hash_status.lower()}"
    return {
        "field": field,
        "path_sha256": sha256(candidate_text.encode("utf-8")).hexdigest(),
        "inside_root": inside,
        "exists": exists,
        "hash_status": hash_status,
        "blocker": blocker,
    }


def _sibling_expected_hash(parent: dict[str, Any], path_key: str) -> str:
    hash_key = "sha256" if path_key == "path" else f"{path_key[:-5]}_sha256"
    value = parent.get(hash_key, "")
    return value if isinstance(value, str) else ""


def _is_declared_path_field(key: str) -> bool:
    normalized = key.lower()
    return normalized == "path" or normalized.endswith("_path")


def _looks_like_pointer_candidate(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in POINTER_NAME_TOKENS)


def _unregistered_row(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": _file_sha256(path),
        "size_bytes": path.stat().st_size,
        "artifact_role": "unregistered",
        "producer": "unknown",
        "schema": "unknown",
        "validator": "none",
        "migration_state": "unregistered",
        "reference_count": 0,
        "resolved_reference_count": 0,
        "hash_binding_count": 0,
        "hash_match_count": 0,
        "reference_blockers": ["artifact_not_in_registry"],
        "resolution_status": "UNRESOLVED_UNREGISTERED",
        "domain_validation_status": "BLOCKED_UNREGISTERED",
        "authority_eligible": False,
        "content_captured": False,
    }


def _relative(path: Path, root: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve()).as_posix()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    """Hash a receipt's canonical identity material, excluding its hash field."""

    material = dict(receipt)
    material.pop("content_sha256", None)
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def canonical_event_sha256(event: Mapping[str, Any]) -> str:
    """Hash an append-only event without creating a self-reference."""

    material = dict(event)
    material.pop("event_sha256", None)
    return sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def lineage_parent(receipt: Mapping[str, Any]) -> dict[str, str]:
    """Create the only parent-reference shape accepted by Gate 00F."""

    artifact_id = receipt.get("artifact_id")
    artifact_type = receipt.get("artifact_type")
    content_sha256 = receipt.get("content_sha256")
    if not isinstance(artifact_id, str) or not IDENTIFIER_PATTERN.fullmatch(artifact_id):
        raise LineageValidationError("parent artifact_id is invalid")
    if artifact_type not in CANONICAL_RECEIPT_TYPES:
        raise LineageValidationError("parent artifact_type is invalid")
    if not isinstance(content_sha256, str) or not SHA256_PATTERN.fullmatch(content_sha256):
        raise LineageValidationError("parent content_sha256 is invalid")
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "content_sha256": content_sha256,
    }


def build_canonical_receipt(
    *,
    artifact_type: str,
    artifact_id: str,
    created_at_utc: str,
    bindings: Mapping[str, Mapping[str, str]],
    payload: Mapping[str, Any],
    parents: Iterable[Mapping[str, str]] = (),
    lineage_state: str = "ACTIVE",
) -> dict[str, Any]:
    """Build and self-validate one canonical immutable Gate 00F receipt."""

    if artifact_type not in CANONICAL_SCHEMA_VERSIONS:
        raise LineageValidationError("unknown canonical artifact_type")
    receipt: dict[str, Any] = {
        "schema_version": CANONICAL_SCHEMA_VERSIONS[artifact_type],
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "content_sha256": ZERO_SHA256,
        "created_at_utc": created_at_utc,
        "lineage_state": lineage_state,
        "parents": [dict(parent) for parent in parents],
        "bindings": {str(name): dict(binding) for name, binding in bindings.items()},
        "payload": dict(payload),
    }
    receipt["content_sha256"] = canonical_receipt_sha256(receipt)
    issues = _receipt_shape_issues(receipt)
    if issues:
        raise LineageValidationError(issues)
    return receipt


def validate_lineage_dag(
    receipts: Iterable[Mapping[str, Any]],
    *,
    authority_artifact_id: str | None = None,
    events: Iterable[Mapping[str, Any]] = (),
    require_authority: bool = True,
) -> LineageValidationResult:
    """Validate canonical receipts, their DAG, and append-only lineage events."""

    receipt_list = [dict(receipt) for receipt in receipts]
    issues: list[LineageIssue] = []
    indexed: dict[str, dict[str, Any]] = {}
    for position, receipt in enumerate(receipt_list):
        shape_issues = _receipt_shape_issues(receipt, position=position)
        issues.extend(shape_issues)
        artifact_id = receipt.get("artifact_id")
        if not isinstance(artifact_id, str):
            continue
        if artifact_id in indexed:
            issues.append(
                LineageIssue(
                    "DUPLICATE_ARTIFACT_ID",
                    artifact_id,
                    "artifact identity appears more than once",
                )
            )
            continue
        indexed[artifact_id] = receipt

    children: dict[str, set[str]] = defaultdict(set)
    issues.extend(_edge_issues(indexed, children))
    topological_order, cycle_nodes = _topological_order(indexed, children)
    if cycle_nodes:
        issues.append(
            LineageIssue(
                "LINEAGE_CYCLE",
                ",".join(cycle_nodes),
                "the parent graph is not acyclic",
            )
        )

    event_issues, invalidated, regenerated = _validate_lineage_events(
        indexed=indexed,
        children=children,
        events=[dict(event) for event in events],
    )
    issues.extend(event_issues)
    issues.extend(_stale_parent_issues(indexed, invalidated))

    authorities = _selected_authorities(
        indexed=indexed,
        authority_artifact_id=authority_artifact_id,
        require_authority=require_authority,
        issues=issues,
    )
    roots: set[str] = set()
    if not cycle_nodes:
        for authority_id in authorities:
            roots.update(
                _authority_chain_issues(
                    authority_id=authority_id,
                    indexed=indexed,
                    invalidated=invalidated,
                    issues=issues,
                )
            )

    normalized_issues = tuple(sorted(set(issues)))
    valid = not normalized_issues
    return LineageValidationResult(
        valid=valid,
        authority_eligible=bool(authorities) and valid,
        issues=normalized_issues,
        authority_artifact_ids=tuple(sorted(authorities)),
        root_artifact_ids=tuple(sorted(roots)),
        topological_order=topological_order,
        invalidated_artifact_ids=tuple(sorted(invalidated)),
        regenerated_artifact_ids=tuple(sorted(regenerated)),
    )


def validate_gate00f_lineage(
    receipts: Iterable[Mapping[str, Any]],
    *,
    authority_artifact_id: str | None = None,
    events: Iterable[Mapping[str, Any]] = (),
) -> LineageValidationResult:
    """Fail-closed authority-facing alias for the canonical DAG validator."""

    return validate_lineage_dag(
        receipts,
        authority_artifact_id=authority_artifact_id,
        events=events,
        require_authority=True,
    )


validate_immutable_lineage_dag = validate_gate00f_lineage


def require_gate00f_authority(result: LineageValidationResult) -> None:
    """Authority consumers must call this instead of trusting receipt booleans."""

    result.require_authority()


def _receipt_shape_issues(
    receipt: Mapping[str, Any], *, position: int | None = None
) -> list[LineageIssue]:
    artifact_id = receipt.get("artifact_id")
    issue_id = artifact_id if isinstance(artifact_id, str) else f"row:{position}"
    issues: list[LineageIssue] = []
    expected_fields = {
        "schema_version",
        "artifact_type",
        "artifact_id",
        "content_sha256",
        "created_at_utc",
        "lineage_state",
        "parents",
        "bindings",
        "payload",
    }
    if set(receipt) != expected_fields:
        issues.append(
            LineageIssue(
                "RECEIPT_SCHEMA_FIELDS_INVALID",
                issue_id,
                _field_difference(set(receipt), expected_fields),
            )
        )
    artifact_type = receipt.get("artifact_type")
    if artifact_type not in CANONICAL_RECEIPT_TYPES:
        issues.append(LineageIssue("ARTIFACT_TYPE_INVALID", issue_id, str(artifact_type)))
    expected_schema = CANONICAL_SCHEMA_VERSIONS.get(str(artifact_type))
    if receipt.get("schema_version") != expected_schema:
        issues.append(
            LineageIssue(
                "SCHEMA_VERSION_MISMATCH",
                issue_id,
                f"expected {expected_schema!r}",
            )
        )
    if not isinstance(artifact_id, str) or not IDENTIFIER_PATTERN.fullmatch(artifact_id):
        issues.append(LineageIssue("ARTIFACT_ID_INVALID", issue_id, str(artifact_id)))
    content_hash = receipt.get("content_sha256")
    if not isinstance(content_hash, str) or not SHA256_PATTERN.fullmatch(content_hash):
        issues.append(LineageIssue("CONTENT_HASH_INVALID", issue_id, str(content_hash)))
    else:
        try:
            computed_hash = canonical_receipt_sha256(receipt)
        except (TypeError, ValueError) as exc:
            issues.append(LineageIssue("NON_CANONICAL_CONTENT", issue_id, safe_exception_code(exc)))
        else:
            if computed_hash != content_hash:
                issues.append(
                    LineageIssue(
                        "CONTENT_HASH_MISMATCH",
                        issue_id,
                        f"expected {content_hash}, computed {computed_hash}",
                    )
                )
    if not _is_utc_timestamp(receipt.get("created_at_utc")):
        issues.append(
            LineageIssue(
                "CREATED_AT_UTC_INVALID",
                issue_id,
                str(receipt.get("created_at_utc")),
            )
        )
    if receipt.get("lineage_state") not in LINEAGE_STATES:
        issues.append(
            LineageIssue(
                "LINEAGE_STATE_INVALID",
                issue_id,
                str(receipt.get("lineage_state")),
            )
        )
    parents = receipt.get("parents")
    issues.extend(_parent_shape_issues(parents, issue_id))
    if artifact_type in CANONICAL_PARENT_TYPES and isinstance(parents, list):
        expected_parent_type = CANONICAL_PARENT_TYPES[artifact_type]
        expected_count = 0 if expected_parent_type is None else 1
        if len(parents) != expected_count:
            issues.append(
                LineageIssue(
                    "PARENT_CARDINALITY_INVALID",
                    issue_id,
                    f"expected {expected_count}, found {len(parents)}",
                )
            )
        for parent in parents:
            if isinstance(parent, dict) and parent.get("artifact_type") != expected_parent_type:
                issues.append(
                    LineageIssue(
                        "CANONICAL_PARENT_TYPE_MISMATCH",
                        issue_id,
                        (f"expected {expected_parent_type}, found {parent.get('artifact_type')}"),
                    )
                )
    issues.extend(_binding_issues(receipt.get("bindings"), issue_id))
    issues.extend(
        _payload_issues(
            artifact_type if isinstance(artifact_type, str) else "",
            receipt.get("payload"),
            issue_id,
        )
    )
    return issues


def _parent_shape_issues(value: Any, artifact_id: str) -> list[LineageIssue]:
    if not isinstance(value, list):
        return [LineageIssue("PARENTS_SCHEMA_INVALID", artifact_id, "parents must be a list")]
    issues: list[LineageIssue] = []
    seen: set[str] = set()
    required = {"artifact_id", "artifact_type", "content_sha256"}
    for index, parent in enumerate(value):
        if not isinstance(parent, dict) or set(parent) != required:
            issues.append(
                LineageIssue(
                    "PARENT_SCHEMA_INVALID",
                    artifact_id,
                    f"parent[{index}] must contain exactly {sorted(required)}",
                )
            )
            continue
        parent_id = parent.get("artifact_id")
        if not isinstance(parent_id, str) or not IDENTIFIER_PATTERN.fullmatch(parent_id):
            issues.append(LineageIssue("PARENT_ID_INVALID", artifact_id, f"parent[{index}]"))
        elif parent_id in seen:
            issues.append(LineageIssue("DUPLICATE_PARENT", artifact_id, parent_id))
        else:
            seen.add(parent_id)
        if parent.get("artifact_type") not in CANONICAL_RECEIPT_TYPES:
            issues.append(LineageIssue("PARENT_TYPE_INVALID", artifact_id, f"parent[{index}]"))
        parent_hash = parent.get("content_sha256")
        if not isinstance(parent_hash, str) or not SHA256_PATTERN.fullmatch(parent_hash):
            issues.append(LineageIssue("PARENT_HASH_INVALID", artifact_id, f"parent[{index}]"))
    return issues


def _binding_issues(value: Any, artifact_id: str) -> list[LineageIssue]:
    if not isinstance(value, dict) or set(value) != set(VERSION_BINDING_NAMES):
        actual = set(value) if isinstance(value, dict) else set()
        return [
            LineageIssue(
                "VERSION_BINDINGS_INVALID",
                artifact_id,
                _field_difference(actual, set(VERSION_BINDING_NAMES)),
            )
        ]
    issues: list[LineageIssue] = []
    for name in VERSION_BINDING_NAMES:
        binding = value[name]
        if not isinstance(binding, dict) or set(binding) != {"version", "sha256"}:
            issues.append(LineageIssue("VERSION_BINDING_SCHEMA_INVALID", artifact_id, name))
            continue
        version = binding.get("version")
        digest = binding.get("sha256")
        if not isinstance(version, str) or not version.strip():
            issues.append(LineageIssue("VERSION_BINDING_VERSION_INVALID", artifact_id, name))
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            issues.append(LineageIssue("VERSION_BINDING_HASH_INVALID", artifact_id, name))
    return issues


def _payload_issues(artifact_type: str, value: Any, artifact_id: str) -> list[LineageIssue]:
    if not isinstance(value, dict):
        return [LineageIssue("PAYLOAD_SCHEMA_INVALID", artifact_id, "payload must be an object")]
    issues: list[LineageIssue] = []
    hash_fields: tuple[str, ...] = ()
    required_fields: set[str]
    if artifact_type == "source_mapping_receipt":
        required_fields = {"source_snapshot_sha256", "mapping_sha256"}
        hash_fields = ("source_snapshot_sha256", "mapping_sha256")
    elif artifact_type == "input_manifest":
        required_fields = {"input_set_sha256"}
        hash_fields = ("input_set_sha256",)
    elif artifact_type == "run_manifest":
        required_fields = {"run_config_sha256"}
        hash_fields = ("run_config_sha256",)
    elif artifact_type == "artifact_manifest":
        required_fields = {"artifact_sha256"}
        hash_fields = ("artifact_sha256",)
    elif artifact_type == "data_quality_receipt":
        required_fields = {"checks_sha256", "status"}
        hash_fields = ("checks_sha256",)
    elif artifact_type == "authority_receipt":
        required_fields = {"accepted_artifact_id", "decision"}
    else:
        return issues
    allowed_fields = required_fields | {"metadata"}
    if not required_fields.issubset(value) or not set(value).issubset(allowed_fields):
        issues.append(
            LineageIssue(
                "PAYLOAD_FIELDS_INVALID",
                artifact_id,
                _field_difference(set(value), required_fields, allowed_fields),
            )
        )
    for field in hash_fields:
        digest = value.get(field)
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            issues.append(LineageIssue("PAYLOAD_HASH_INVALID", artifact_id, field))
    metadata = value.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        issues.append(LineageIssue("PAYLOAD_METADATA_INVALID", artifact_id, "metadata"))
    if artifact_type == "data_quality_receipt" and value.get("status") not in {
        "PASS",
        "FAIL",
    }:
        issues.append(
            LineageIssue("DATA_QUALITY_STATUS_INVALID", artifact_id, str(value.get("status")))
        )
    if artifact_type == "authority_receipt":
        if value.get("decision") not in {"ACCEPT", "DENY"}:
            issues.append(
                LineageIssue(
                    "AUTHORITY_DECISION_INVALID",
                    artifact_id,
                    str(value.get("decision")),
                )
            )
        accepted_id = value.get("accepted_artifact_id")
        if not isinstance(accepted_id, str) or not IDENTIFIER_PATTERN.fullmatch(accepted_id):
            issues.append(
                LineageIssue("ACCEPTED_ARTIFACT_ID_INVALID", artifact_id, str(accepted_id))
            )
    try:
        _canonical_json(value)
    except (TypeError, ValueError) as exc:
        issues.append(LineageIssue("PAYLOAD_NOT_CANONICAL", artifact_id, safe_exception_code(exc)))
    return issues


def _edge_issues(
    indexed: Mapping[str, Mapping[str, Any]], children: dict[str, set[str]]
) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    for artifact_id, receipt in indexed.items():
        artifact_type = receipt.get("artifact_type")
        parents = receipt.get("parents")
        if not isinstance(parents, list):
            continue
        expected_parent_type = CANONICAL_PARENT_TYPES.get(str(artifact_type))
        expected_count = 0 if expected_parent_type is None else 1
        if len(parents) != expected_count:
            issues.append(
                LineageIssue(
                    "PARENT_CARDINALITY_INVALID",
                    artifact_id,
                    f"expected {expected_count}, found {len(parents)}",
                )
            )
        for parent_reference in parents:
            if not isinstance(parent_reference, dict):
                continue
            parent_id = parent_reference.get("artifact_id")
            if not isinstance(parent_id, str):
                continue
            parent = indexed.get(parent_id)
            if parent is None:
                issues.append(LineageIssue("MISSING_PARENT", artifact_id, parent_id))
                continue
            children[parent_id].add(artifact_id)
            actual_parent_type = parent.get("artifact_type")
            if parent_reference.get("artifact_type") != actual_parent_type:
                issues.append(
                    LineageIssue(
                        "PARENT_TYPE_MISMATCH",
                        artifact_id,
                        parent_id,
                    )
                )
            if actual_parent_type != expected_parent_type:
                issues.append(
                    LineageIssue(
                        "CANONICAL_PARENT_TYPE_MISMATCH",
                        artifact_id,
                        f"expected {expected_parent_type}, found {actual_parent_type}",
                    )
                )
            if parent_reference.get("content_sha256") != parent.get("content_sha256"):
                issues.append(
                    LineageIssue(
                        "PARENT_HASH_MISMATCH",
                        artifact_id,
                        parent_id,
                    )
                )
    return issues


def _topological_order(
    indexed: Mapping[str, Mapping[str, Any]],
    children: Mapping[str, set[str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    indegree = {artifact_id: 0 for artifact_id in indexed}
    for receipt in indexed.values():
        parents = receipt.get("parents")
        if not isinstance(parents, list):
            continue
        for parent in parents:
            if isinstance(parent, dict) and parent.get("artifact_id") in indexed:
                artifact_id = receipt.get("artifact_id")
                if isinstance(artifact_id, str):
                    indegree[artifact_id] += 1
    ready = deque(sorted(key for key, value in indegree.items() if value == 0))
    order: list[str] = []
    while ready:
        artifact_id = ready.popleft()
        order.append(artifact_id)
        for child_id in sorted(children.get(artifact_id, set())):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    cycle_nodes = tuple(sorted(set(indexed) - set(order)))
    return tuple(order), cycle_nodes


def _selected_authorities(
    *,
    indexed: Mapping[str, Mapping[str, Any]],
    authority_artifact_id: str | None,
    require_authority: bool,
    issues: list[LineageIssue],
) -> tuple[str, ...]:
    if not require_authority:
        return ()
    if authority_artifact_id is not None:
        receipt = indexed.get(authority_artifact_id)
        if receipt is None:
            issues.append(
                LineageIssue("AUTHORITY_RECEIPT_MISSING", authority_artifact_id, "not found")
            )
            return ()
        if receipt.get("artifact_type") != "authority_receipt":
            issues.append(
                LineageIssue(
                    "AUTHORITY_ARTIFACT_TYPE_INVALID",
                    authority_artifact_id,
                    str(receipt.get("artifact_type")),
                )
            )
            return ()
        return (authority_artifact_id,)
    authorities = tuple(
        sorted(
            artifact_id
            for artifact_id, receipt in indexed.items()
            if receipt.get("artifact_type") == "authority_receipt"
            and isinstance(receipt.get("payload"), dict)
            and receipt["payload"].get("decision") == "ACCEPT"
        )
    )
    if not authorities:
        issues.append(
            LineageIssue("AUTHORITY_RECEIPT_MISSING", detail="no ACCEPT authority receipt")
        )
    return authorities


def _authority_chain_issues(
    *,
    authority_id: str,
    indexed: Mapping[str, Mapping[str, Any]],
    invalidated: set[str],
    issues: list[LineageIssue],
) -> set[str]:
    chain: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    current_id = authority_id
    while current_id in indexed and current_id not in seen:
        seen.add(current_id)
        receipt = indexed[current_id]
        chain.append(receipt)
        parents = receipt.get("parents")
        if not isinstance(parents, list) or len(parents) != 1:
            break
        parent_id = parents[0].get("artifact_id") if isinstance(parents[0], dict) else None
        if not isinstance(parent_id, str):
            break
        current_id = parent_id
    chain_types = tuple(str(receipt.get("artifact_type")) for receipt in chain)
    if chain_types != CANONICAL_CHAIN_FROM_AUTHORITY:
        issues.append(
            LineageIssue(
                "CANONICAL_CHAIN_INCOMPLETE",
                authority_id,
                f"found {chain_types}",
            )
        )
    roots = {
        str(receipt.get("artifact_id"))
        for receipt in chain
        if receipt.get("artifact_type") == "source_mapping_receipt" and receipt.get("parents") == []
    }
    if len(roots) != 1:
        issues.append(
            LineageIssue(
                "ROOT_CARDINALITY_INVALID",
                authority_id,
                f"expected 1, found {len(roots)}",
            )
        )
    baseline_bindings = chain[-1].get("bindings") if chain else None
    for receipt in chain:
        artifact_id = str(receipt.get("artifact_id"))
        if receipt.get("lineage_state") != "ACTIVE":
            issues.append(
                LineageIssue(
                    "STALE_AUTHORITY_LINEAGE",
                    authority_id,
                    artifact_id,
                )
            )
        if artifact_id in invalidated:
            issues.append(
                LineageIssue(
                    "INVALIDATED_AUTHORITY_LINEAGE",
                    authority_id,
                    artifact_id,
                )
            )
        if receipt.get("bindings") != baseline_bindings:
            issues.append(
                LineageIssue(
                    "VERSION_BINDING_MISMATCH",
                    authority_id,
                    artifact_id,
                )
            )
    authority = indexed[authority_id]
    authority_payload = authority.get("payload")
    if not isinstance(authority_payload, dict) or authority_payload.get("decision") != "ACCEPT":
        issues.append(LineageIssue("AUTHORITY_DECISION_NOT_ACCEPT", authority_id, "fail closed"))
    artifact_ids = {
        str(receipt.get("artifact_id"))
        for receipt in chain
        if receipt.get("artifact_type") == "artifact_manifest"
    }
    accepted_id = (
        authority_payload.get("accepted_artifact_id")
        if isinstance(authority_payload, dict)
        else None
    )
    if accepted_id not in artifact_ids:
        issues.append(LineageIssue("ACCEPTED_ARTIFACT_MISMATCH", authority_id, str(accepted_id)))
    quality_receipts = [
        receipt for receipt in chain if receipt.get("artifact_type") == "data_quality_receipt"
    ]
    if len(quality_receipts) != 1 or quality_receipts[0].get("payload", {}).get("status") != "PASS":
        issues.append(LineageIssue("DATA_QUALITY_NOT_PASS", authority_id, "fail closed"))
    return roots


def _stale_parent_issues(
    indexed: Mapping[str, Mapping[str, Any]], invalidated: set[str]
) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    for artifact_id, receipt in indexed.items():
        if receipt.get("lineage_state") != "ACTIVE" or artifact_id in invalidated:
            continue
        parents = receipt.get("parents")
        if not isinstance(parents, list):
            continue
        for reference in parents:
            if not isinstance(reference, dict):
                continue
            parent_id = reference.get("artifact_id")
            parent = indexed.get(str(parent_id))
            if parent is not None and (
                parent.get("lineage_state") != "ACTIVE" or parent_id in invalidated
            ):
                issues.append(LineageIssue("STALE_PARENT", artifact_id, str(parent_id)))
    return issues


def build_descendant_invalidation_event(
    receipts: Iterable[Mapping[str, Any]],
    *,
    trigger_artifact_id: str,
    reason_code: str,
    recorded_at_utc: str,
    previous_event_sha256: str = ZERO_SHA256,
) -> dict[str, Any]:
    """Create evidence containing the exact transitive reverse-DAG closure."""

    receipt_list = [dict(receipt) for receipt in receipts]
    structural = validate_lineage_dag(receipt_list, require_authority=False)
    if not structural.valid:
        raise LineageValidationError(structural.issues)
    indexed = {str(receipt["artifact_id"]): receipt for receipt in receipt_list}
    trigger = indexed.get(trigger_artifact_id)
    if trigger is None:
        raise LineageValidationError("invalidation trigger does not exist")
    if not REASON_CODE_PATTERN.fullmatch(reason_code):
        raise LineageValidationError("invalidation reason_code is invalid")
    if not _is_utc_timestamp(recorded_at_utc):
        raise LineageValidationError("invalidation timestamp is invalid")
    if not SHA256_PATTERN.fullmatch(previous_event_sha256):
        raise LineageValidationError("previous event hash is invalid")
    children = _children_for(indexed)
    closure = _descendant_closure(children, trigger_artifact_id)
    base: dict[str, Any] = {
        "schema_version": INVALIDATION_EVENT_SCHEMA_VERSION,
        "event_type": "INVALIDATE_DESCENDANTS",
        "recorded_at_utc": recorded_at_utc,
        "previous_event_sha256": previous_event_sha256,
        "trigger_artifact_id": trigger_artifact_id,
        "trigger_artifact_sha256": trigger["content_sha256"],
        "reason_code": reason_code,
        "invalidated": [
            {
                "artifact_id": artifact_id,
                "artifact_sha256": indexed[artifact_id]["content_sha256"],
                "depth": depth,
            }
            for artifact_id, depth in sorted(closure.items())
        ],
    }
    event_id = (
        "lineage_invalidation:" + sha256(_canonical_json(base).encode("utf-8")).hexdigest()[:24]
    )
    event = {"event_id": event_id, "event_sha256": ZERO_SHA256, **base}
    event["event_sha256"] = canonical_event_sha256(event)
    event_issues = _event_shape_issues(event)
    if event_issues:
        raise LineageValidationError(event_issues)
    return event


def build_descendant_regeneration_event(
    receipts: Iterable[Mapping[str, Any]],
    *,
    invalidation_event: Mapping[str, Any],
    replacements: Mapping[str, str],
    recorded_at_utc: str,
    previous_event_sha256: str | None = None,
) -> dict[str, Any]:
    """Bind invalidated identities to new immutable replacements without revival."""

    receipt_list = [dict(receipt) for receipt in receipts]
    structural = validate_lineage_dag(receipt_list, require_authority=False)
    if not structural.valid:
        raise LineageValidationError(structural.issues)
    indexed = {str(receipt["artifact_id"]): receipt for receipt in receipt_list}
    invalidation = dict(invalidation_event)
    event_issues = _event_shape_issues(invalidation)
    if event_issues or invalidation.get("event_type") != "INVALIDATE_DESCENDANTS":
        raise LineageValidationError(event_issues or "regeneration requires an invalidation event")
    if not _is_utc_timestamp(recorded_at_utc):
        raise LineageValidationError("regeneration timestamp is invalid")
    invalidated = {
        row["artifact_id"]: row
        for row in invalidation["invalidated"]
        if isinstance(row, dict) and isinstance(row.get("artifact_id"), str)
    }
    replacement_rows: list[dict[str, str]] = []
    seen_replacements: set[str] = set()
    for invalidated_id, replacement_id in sorted(replacements.items()):
        old = indexed.get(invalidated_id)
        new = indexed.get(replacement_id)
        if invalidated_id not in invalidated or old is None:
            raise LineageValidationError(f"replacement source is not invalidated: {invalidated_id}")
        if new is None or replacement_id == invalidated_id:
            raise LineageValidationError(f"replacement identity is invalid: {replacement_id}")
        if replacement_id in seen_replacements:
            raise LineageValidationError(f"replacement identity is duplicated: {replacement_id}")
        if old.get("artifact_type") != new.get("artifact_type"):
            raise LineageValidationError(f"replacement type differs for {invalidated_id}")
        seen_replacements.add(replacement_id)
        replacement_rows.append(
            {
                "invalidated_artifact_id": invalidated_id,
                "invalidated_artifact_sha256": old["content_sha256"],
                "replacement_artifact_id": replacement_id,
                "replacement_artifact_sha256": new["content_sha256"],
            }
        )
    prior_hash = previous_event_sha256 or str(invalidation["event_sha256"])
    if not SHA256_PATTERN.fullmatch(prior_hash):
        raise LineageValidationError("previous event hash is invalid")
    base: dict[str, Any] = {
        "schema_version": REGENERATION_EVENT_SCHEMA_VERSION,
        "event_type": "REGENERATE_DESCENDANTS",
        "recorded_at_utc": recorded_at_utc,
        "previous_event_sha256": prior_hash,
        "invalidation_event_id": invalidation["event_id"],
        "replacements": replacement_rows,
    }
    event_id = (
        "lineage_regeneration:" + sha256(_canonical_json(base).encode("utf-8")).hexdigest()[:24]
    )
    event = {"event_id": event_id, "event_sha256": ZERO_SHA256, **base}
    event["event_sha256"] = canonical_event_sha256(event)
    event_issues = _event_shape_issues(event)
    if event_issues:
        raise LineageValidationError(event_issues)
    return event


def _validate_lineage_events(
    *,
    indexed: Mapping[str, Mapping[str, Any]],
    children: Mapping[str, set[str]],
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[LineageIssue], set[str], set[str]]:
    issues: list[LineageIssue] = []
    invalidated: set[str] = set()
    regenerated: set[str] = set()
    invalidation_events: dict[str, Mapping[str, Any]] = {}
    seen_event_ids: set[str] = set()
    previous_hash = ZERO_SHA256
    previous_timestamp: datetime | None = None
    for position, event in enumerate(events):
        event_issues = _event_shape_issues(event, position=position)
        issues.extend(event_issues)
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            continue
        if event_id in seen_event_ids:
            issues.append(LineageIssue("DUPLICATE_EVENT_ID", event_id, "event log"))
        seen_event_ids.add(event_id)
        if event.get("previous_event_sha256") != previous_hash:
            issues.append(
                LineageIssue(
                    "EVENT_CHAIN_HASH_MISMATCH",
                    event_id,
                    f"expected {previous_hash}",
                )
            )
        event_hash = event.get("event_sha256")
        if isinstance(event_hash, str) and SHA256_PATTERN.fullmatch(event_hash):
            previous_hash = event_hash
        if event_issues:
            continue
        event_timestamp = _parse_utc_timestamp(str(event["recorded_at_utc"]))
        if previous_timestamp is not None and event_timestamp < previous_timestamp:
            issues.append(
                LineageIssue(
                    "EVENT_TIMESTAMP_OUT_OF_ORDER",
                    event_id,
                    str(event["recorded_at_utc"]),
                )
            )
        previous_timestamp = event_timestamp
        if event.get("event_type") == "INVALIDATE_DESCENDANTS":
            trigger_id = str(event["trigger_artifact_id"])
            trigger = indexed.get(trigger_id)
            if trigger is None:
                issues.append(LineageIssue("INVALIDATION_TRIGGER_MISSING", event_id, trigger_id))
                continue
            if event["trigger_artifact_sha256"] != trigger.get("content_sha256"):
                issues.append(
                    LineageIssue("INVALIDATION_TRIGGER_HASH_MISMATCH", event_id, trigger_id)
                )
            expected_closure = _descendant_closure(children, trigger_id)
            actual_closure = {
                str(row["artifact_id"]): int(row["depth"]) for row in event["invalidated"]
            }
            if actual_closure != expected_closure:
                issues.append(
                    LineageIssue(
                        "INVALIDATION_CLOSURE_MISMATCH",
                        event_id,
                        "event must enumerate the exact reverse-DAG closure",
                    )
                )
            for row in event["invalidated"]:
                artifact_id = str(row["artifact_id"])
                receipt = indexed.get(artifact_id)
                if receipt is None:
                    issues.append(
                        LineageIssue("INVALIDATED_ARTIFACT_MISSING", event_id, artifact_id)
                    )
                elif row["artifact_sha256"] != receipt.get("content_sha256"):
                    issues.append(
                        LineageIssue(
                            "INVALIDATED_ARTIFACT_HASH_MISMATCH",
                            event_id,
                            artifact_id,
                        )
                    )
            invalidated.update(actual_closure)
            invalidation_events[event_id] = event
        elif event.get("event_type") == "REGENERATE_DESCENDANTS":
            source_event = invalidation_events.get(str(event["invalidation_event_id"]))
            if source_event is None:
                issues.append(
                    LineageIssue(
                        "REGENERATION_SOURCE_EVENT_MISSING",
                        event_id,
                        str(event["invalidation_event_id"]),
                    )
                )
                continue
            source_ids = {str(row["artifact_id"]) for row in source_event["invalidated"]}
            invalidated_at = _parse_utc_timestamp(str(source_event["recorded_at_utc"]))
            old_ids: set[str] = set()
            new_ids: set[str] = set()
            for row in event["replacements"]:
                old_id = str(row["invalidated_artifact_id"])
                new_id = str(row["replacement_artifact_id"])
                old = indexed.get(old_id)
                new = indexed.get(new_id)
                if old_id in old_ids or new_id in new_ids:
                    issues.append(LineageIssue("DUPLICATE_REGENERATION_MAPPING", event_id, new_id))
                old_ids.add(old_id)
                new_ids.add(new_id)
                if old_id not in source_ids:
                    issues.append(
                        LineageIssue("REGENERATION_SOURCE_NOT_INVALIDATED", event_id, old_id)
                    )
                if old is None or row["invalidated_artifact_sha256"] != old.get("content_sha256"):
                    issues.append(
                        LineageIssue("REGENERATION_SOURCE_HASH_MISMATCH", event_id, old_id)
                    )
                if new is None or row["replacement_artifact_sha256"] != new.get("content_sha256"):
                    issues.append(
                        LineageIssue("REGENERATION_REPLACEMENT_HASH_MISMATCH", event_id, new_id)
                    )
                if (
                    old is not None
                    and new is not None
                    and old.get("artifact_type") != new.get("artifact_type")
                ):
                    issues.append(LineageIssue("REGENERATION_TYPE_MISMATCH", event_id, new_id))
                if new is not None:
                    replacement_timestamp = _parse_utc_timestamp(str(new["created_at_utc"]))
                    if not invalidated_at <= replacement_timestamp <= event_timestamp:
                        issues.append(
                            LineageIssue(
                                "REGENERATION_REPLACEMENT_TIMESTAMP_INVALID",
                                event_id,
                                new_id,
                            )
                        )
                if new_id in invalidated:
                    issues.append(
                        LineageIssue("REGENERATION_REPLACEMENT_INVALIDATED", event_id, new_id)
                    )
            regenerated.update(new_ids)
    return issues, invalidated, regenerated


def _event_shape_issues(
    event: Mapping[str, Any], *, position: int | None = None
) -> list[LineageIssue]:
    event_id = event.get("event_id")
    issue_id = event_id if isinstance(event_id, str) else f"event:{position}"
    event_type = event.get("event_type")
    common = {
        "schema_version",
        "event_type",
        "event_id",
        "event_sha256",
        "recorded_at_utc",
        "previous_event_sha256",
    }
    if event_type == "INVALIDATE_DESCENDANTS":
        expected_fields = common | {
            "trigger_artifact_id",
            "trigger_artifact_sha256",
            "reason_code",
            "invalidated",
        }
        expected_schema = INVALIDATION_EVENT_SCHEMA_VERSION
    elif event_type == "REGENERATE_DESCENDANTS":
        expected_fields = common | {"invalidation_event_id", "replacements"}
        expected_schema = REGENERATION_EVENT_SCHEMA_VERSION
    else:
        return [LineageIssue("EVENT_TYPE_INVALID", issue_id, str(event_type))]
    issues: list[LineageIssue] = []
    if set(event) != expected_fields:
        issues.append(
            LineageIssue(
                "EVENT_SCHEMA_FIELDS_INVALID",
                issue_id,
                _field_difference(set(event), expected_fields),
            )
        )
    if event.get("schema_version") != expected_schema:
        issues.append(LineageIssue("EVENT_SCHEMA_VERSION_MISMATCH", issue_id, str(expected_schema)))
    if not isinstance(event_id, str) or not IDENTIFIER_PATTERN.fullmatch(event_id):
        issues.append(LineageIssue("EVENT_ID_INVALID", issue_id, str(event_id)))
    event_hash = event.get("event_sha256")
    if not isinstance(event_hash, str) or not SHA256_PATTERN.fullmatch(event_hash):
        issues.append(LineageIssue("EVENT_HASH_INVALID", issue_id, str(event_hash)))
    else:
        try:
            computed_hash = canonical_event_sha256(event)
        except (TypeError, ValueError) as exc:
            issues.append(LineageIssue("EVENT_NOT_CANONICAL", issue_id, safe_exception_code(exc)))
        else:
            if event_hash != computed_hash:
                issues.append(
                    LineageIssue(
                        "EVENT_HASH_MISMATCH",
                        issue_id,
                        f"expected {event_hash}, computed {computed_hash}",
                    )
                )
    if not _is_utc_timestamp(event.get("recorded_at_utc")):
        issues.append(
            LineageIssue(
                "EVENT_TIMESTAMP_INVALID",
                issue_id,
                str(event.get("recorded_at_utc")),
            )
        )
    previous_hash = event.get("previous_event_sha256")
    if not isinstance(previous_hash, str) or not SHA256_PATTERN.fullmatch(previous_hash):
        issues.append(LineageIssue("PREVIOUS_EVENT_HASH_INVALID", issue_id, str(previous_hash)))
    if event_type == "INVALIDATE_DESCENDANTS":
        trigger_id = event.get("trigger_artifact_id")
        trigger_hash = event.get("trigger_artifact_sha256")
        if not isinstance(trigger_id, str) or not IDENTIFIER_PATTERN.fullmatch(trigger_id):
            issues.append(LineageIssue("INVALIDATION_TRIGGER_ID_INVALID", issue_id))
        if not isinstance(trigger_hash, str) or not SHA256_PATTERN.fullmatch(trigger_hash):
            issues.append(LineageIssue("INVALIDATION_TRIGGER_HASH_INVALID", issue_id))
        reason = event.get("reason_code")
        if not isinstance(reason, str) or not REASON_CODE_PATTERN.fullmatch(reason):
            issues.append(LineageIssue("INVALIDATION_REASON_CODE_INVALID", issue_id))
        issues.extend(
            _event_rows_issues(
                event.get("invalidated"),
                issue_id,
                required={"artifact_id", "artifact_sha256", "depth"},
                id_fields=("artifact_id",),
                hash_fields=("artifact_sha256",),
                require_depth=True,
            )
        )
    else:
        invalidation_id = event.get("invalidation_event_id")
        if not isinstance(invalidation_id, str) or not IDENTIFIER_PATTERN.fullmatch(
            invalidation_id
        ):
            issues.append(LineageIssue("REGENERATION_EVENT_ID_INVALID", issue_id))
        issues.extend(
            _event_rows_issues(
                event.get("replacements"),
                issue_id,
                required={
                    "invalidated_artifact_id",
                    "invalidated_artifact_sha256",
                    "replacement_artifact_id",
                    "replacement_artifact_sha256",
                },
                id_fields=(
                    "invalidated_artifact_id",
                    "replacement_artifact_id",
                ),
                hash_fields=(
                    "invalidated_artifact_sha256",
                    "replacement_artifact_sha256",
                ),
            )
        )
    return issues


def _event_rows_issues(
    value: Any,
    event_id: str,
    *,
    required: set[str],
    id_fields: tuple[str, ...],
    hash_fields: tuple[str, ...],
    require_depth: bool = False,
) -> list[LineageIssue]:
    if not isinstance(value, list) or not value:
        return [LineageIssue("EVENT_ROWS_INVALID", event_id, "rows must be a non-empty list")]
    issues: list[LineageIssue] = []
    seen_identities: set[tuple[str, ...]] = set()
    for index, row in enumerate(value):
        if not isinstance(row, dict) or set(row) != required:
            issues.append(LineageIssue("EVENT_ROW_SCHEMA_INVALID", event_id, f"row[{index}]"))
            continue
        identity = tuple(str(row.get(field, "")) for field in id_fields)
        if identity in seen_identities:
            issues.append(LineageIssue("DUPLICATE_EVENT_ROW", event_id, f"row[{index}]"))
        seen_identities.add(identity)
        for field in id_fields:
            item = row.get(field)
            if not isinstance(item, str) or not IDENTIFIER_PATTERN.fullmatch(item):
                issues.append(
                    LineageIssue("EVENT_ROW_ID_INVALID", event_id, f"row[{index}].{field}")
                )
        for field in hash_fields:
            digest = row.get(field)
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                issues.append(
                    LineageIssue(
                        "EVENT_ROW_HASH_INVALID",
                        event_id,
                        f"row[{index}].{field}",
                    )
                )
        if require_depth and (type(row.get("depth")) is not int or int(row["depth"]) < 0):
            issues.append(LineageIssue("EVENT_ROW_DEPTH_INVALID", event_id, f"row[{index}]"))
    return issues


def _children_for(
    indexed: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    children: dict[str, set[str]] = defaultdict(set)
    for artifact_id, receipt in indexed.items():
        for parent in receipt.get("parents", []):
            if isinstance(parent, dict) and parent.get("artifact_id") in indexed:
                children[str(parent["artifact_id"])].add(artifact_id)
    return children


def _descendant_closure(
    children: Mapping[str, set[str]], trigger_artifact_id: str
) -> dict[str, int]:
    depths = {trigger_artifact_id: 0}
    queue = deque([trigger_artifact_id])
    while queue:
        artifact_id = queue.popleft()
        for child_id in sorted(children.get(artifact_id, set())):
            candidate_depth = depths[artifact_id] + 1
            if child_id not in depths or candidate_depth < depths[child_id]:
                depths[child_id] = candidate_depth
                queue.append(child_id)
    return depths


def validate_lineage_files(
    root: Path,
    receipt_paths: Iterable[Path | str],
    *,
    authority_artifact_id: str | None = None,
    event_log_path: Path | str | None = None,
) -> LineageValidationResult:
    """Load immutable files safely, then perform authority-facing validation."""

    root = root.resolve()
    receipts: list[dict[str, Any]] = []
    load_issues: list[LineageIssue] = []
    for supplied_path in receipt_paths:
        try:
            path = _safe_lineage_file(root, supplied_path)
            payload = _strict_json_loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("receipt root must be an object")
            receipts.append(payload)
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
            load_issues.append(LineageIssue("RECEIPT_FILE_INVALID", str(supplied_path), safe_exception_code(exc)))
    events: list[dict[str, Any]] = []
    if event_log_path is not None:
        try:
            event_path = _safe_lineage_file(root, event_log_path)
            events = load_lineage_event_log(event_path)
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            load_issues.append(LineageIssue("EVENT_LOG_INVALID", str(event_log_path), safe_exception_code(exc)))
    result = validate_gate00f_lineage(
        receipts,
        authority_artifact_id=authority_artifact_id,
        events=events,
    )
    if not load_issues:
        return result
    issues = tuple(sorted(set(result.issues + tuple(load_issues))))
    return LineageValidationResult(
        valid=False,
        authority_eligible=False,
        issues=issues,
        authority_artifact_ids=result.authority_artifact_ids,
        root_artifact_ids=result.root_artifact_ids,
        topological_order=result.topological_order,
        invalidated_artifact_ids=result.invalidated_artifact_ids,
        regenerated_artifact_ids=result.regenerated_artifact_ids,
    )


def load_lineage_event_log(path: Path) -> list[dict[str, Any]]:
    """Read and validate an append-only JSONL event hash chain."""

    if path.is_symlink() or not path.is_file():
        raise LineageValidationError("event log is not a regular file")
    if path.stat().st_nlink != 1:
        raise LineageValidationError("event log has multiple hard links")
    events = _parse_event_log(path.read_text(encoding="utf-8"))
    issues = _event_log_chain_issues(events)
    if issues:
        raise LineageValidationError(issues)
    return events


def append_lineage_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append one fsynced event without permitting rewrite or chain forking."""

    if not path.parent.is_dir():
        raise LineageValidationError("event log parent directory does not exist")
    if path.is_symlink():
        raise LineageValidationError("event log cannot be a symlink")
    event_copy = dict(event)
    shape_issues = _event_shape_issues(event_copy)
    if shape_issues:
        raise LineageValidationError(shape_issues)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise LineageValidationError("event log must be a single-link regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        try:
            existing_text = b"".join(chunks).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise LineageValidationError("event log is not UTF-8") from exc
        existing = _parse_event_log(existing_text)
        existing_issues = _event_log_chain_issues(existing)
        if existing_issues:
            raise LineageValidationError(existing_issues)
        expected_previous = str(existing[-1]["event_sha256"]) if existing else ZERO_SHA256
        if event_copy.get("previous_event_sha256") != expected_previous:
            raise LineageValidationError(
                (
                    LineageIssue(
                        "EVENT_CHAIN_HASH_MISMATCH",
                        str(event_copy.get("event_id", "")),
                        f"expected {expected_previous}",
                    ),
                )
            )
        if any(
            existing_event.get("event_id") == event_copy.get("event_id")
            for existing_event in existing
        ):
            raise LineageValidationError(
                (
                    LineageIssue(
                        "DUPLICATE_EVENT_ID",
                        str(event_copy.get("event_id", "")),
                        "event log",
                    ),
                )
            )
        candidate_issues = _event_log_chain_issues([*existing, event_copy])
        if candidate_issues:
            raise LineageValidationError(candidate_issues)
        encoded = (_canonical_json(event_copy) + "\n").encode("utf-8")
        os.lseek(descriptor, 0, os.SEEK_END)
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short append to lineage event log")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parse_event_log(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise LineageValidationError(f"blank event log line at {line_number}")
        try:
            payload = _strict_json_loads(line)
        except ValueError as exc:
            raise LineageValidationError(
                f"invalid event JSON at line {line_number}: {safe_exception_code(exc)}"
            ) from exc
        if not isinstance(payload, dict):
            raise LineageValidationError(f"event at line {line_number} is not an object")
        events.append(payload)
    return events


def _event_log_chain_issues(
    events: Sequence[Mapping[str, Any]],
) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    previous = ZERO_SHA256
    previous_timestamp: datetime | None = None
    seen: set[str] = set()
    for position, event in enumerate(events):
        issues.extend(_event_shape_issues(event, position=position))
        event_id = str(event.get("event_id", f"event:{position}"))
        if event_id in seen:
            issues.append(LineageIssue("DUPLICATE_EVENT_ID", event_id, "log"))
        seen.add(event_id)
        if event.get("previous_event_sha256") != previous:
            issues.append(
                LineageIssue(
                    "EVENT_CHAIN_HASH_MISMATCH",
                    event_id,
                    f"expected {previous}",
                )
            )
        recorded_at = event.get("recorded_at_utc")
        if _is_utc_timestamp(recorded_at):
            event_timestamp = _parse_utc_timestamp(str(recorded_at))
            if previous_timestamp is not None and event_timestamp < previous_timestamp:
                issues.append(
                    LineageIssue(
                        "EVENT_TIMESTAMP_OUT_OF_ORDER",
                        event_id,
                        str(recorded_at),
                    )
                )
            previous_timestamp = event_timestamp
        event_hash = event.get("event_sha256")
        if isinstance(event_hash, str) and SHA256_PATTERN.fullmatch(event_hash):
            previous = event_hash
    return issues


def _safe_lineage_file(root: Path, supplied_path: Path | str) -> Path:
    path = Path(supplied_path)
    lexical = Path(os.path.abspath(path if path.is_absolute() else root / path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError("lineage path is outside the repository root") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError("lineage path contains a symlink")
    resolved = lexical.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError("lineage path is outside the repository root")
    if not lexical.is_file():
        raise ValueError("lineage path is not a regular non-symlink file")
    if lexical.stat().st_nlink != 1:
        raise ValueError("lineage path has multiple hard links")
    return lexical


def _strict_json_loads(text: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON value: {value}")

    try:
        return json.loads(
            text,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = _parse_utc_timestamp(value)
    except ValueError:
        return False
    return parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0


def _parse_utc_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _field_difference(actual: set[str], required: set[str], allowed: set[str] | None = None) -> str:
    allowed = allowed or required
    missing = sorted(required - actual)
    extra = sorted(actual - allowed)
    return f"missing={missing}, extra={extra}"
