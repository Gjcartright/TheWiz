"""Phase 00 maintenance mode and quiesced baseline checkpoints."""

from __future__ import annotations

import csv
import json
import os
import secrets
import stat as stat_module
import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_active_artifact_envelopes import (
    build_active_artifact_lineage_bundle,
    load_active_artifact_envelope_index,
)
from quant_platform.orchestration.corrective_financial_effect_registry import (
    financial_effect_surface_rows,
    unfenced_financial_effect_surface_ids,
)
from quant_platform.orchestration.corrective_lineage import active_artifact_rows
from quant_platform.orchestration.corrective_publication_registry import (
    publication_surface_rows,
    unmigrated_publication_surface_ids,
    unpaired_staging_surface_ids,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    SCHEDULER_CONTRACTS,
    atomic_write_text,
    scheduler_runtime_contract,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    PHASE00_CONTROL_SCOPE,
    PHASE00_MAINTENANCE_MARKER,
    _phase00_maintenance_controller_lock,
    current_publication_lease,
    read_phase00_maintenance_state,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_REPAIR_PROFILE,
    EffectAuthority,
    EffectAuthorityError,
    PublicationAuthoritySession,
    current_publication_authority,
    publication_authority_session,
    require_publication_authority,
)

SCHEMA_VERSION = "thewiz.phase00_control.v1"
DEFAULT_MAINTENANCE_TTL_SECONDS = 2 * 60 * 60
DEFAULT_LOCK_WAIT_SECONDS = 60.0
COMMAND_TIMEOUT_SECONDS = 10.0
MAX_CHECKPOINT_HASH_BYTES = 64 * 1024**2
MAX_PHASE00_CONTROL_PUBLICATION_BYTES = 512 * 1024**2
PHASE00_CONTROL_PUBLICATION_POLICY_VERSION = "thewiz.phase00_control.publication.v1"
CONTROL_ROOT = "data/research/phase00_control"
ACTIVE_CHECKPOINT = "reports/active/phase00_control_checkpoint.json"
ACTIVE_MAINTENANCE = "reports/active/phase00_maintenance.json"
ACTIVE_DESCENDANT_CONTROL = "reports/active/phase00_descendant_control.json"
ACTIVE_CLOSURE = "reports/active/phase00_closure.json"
ACTIVE_ACCEPTANCE_MATRIX = "reports/active/phase00_acceptance_matrix.csv"
ACTIVE_FAULT_CATALOG = "reports/active/phase00_fault_catalog.csv"
CONTROL_OUTPUT_PREFIXES = (
    CONTROL_ROOT,
    "reports/active/phase00_active_artifact_lineage.json",
    ACTIVE_CHECKPOINT,
    ACTIVE_DESCENDANT_CONTROL,
    ACTIVE_CLOSURE,
    ACTIVE_ACCEPTANCE_MATRIX,
    ACTIVE_FAULT_CATALOG,
    ACTIVE_MAINTENANCE,
    "reports/active/phase00_source_evidence_index.csv",
)


def start_phase00_maintenance(
    *,
    root: Path,
    reason: str,
    now: datetime | None = None,
    ttl_seconds: int = DEFAULT_MAINTENANCE_TTL_SECONDS,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CommandResult:
    """Drain governed writers and activate a fail-closed maintenance marker."""

    root = _canonical_root(root)
    started_at = _as_utc(now)
    if ttl_seconds < 60:
        raise ValueError("phase00 maintenance ttl must be at least 60 seconds")
    if not reason.strip():
        raise ValueError("phase00 maintenance reason is required")
    marker = root / PHASE00_MAINTENANCE_MARKER
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        controller_lease = current_publication_lease()
        governed_writers_drained = bool(
            controller_lease is not None
            and controller_lease.maintenance_controller
            and Path(controller_lease.root) == root
        )
        if not governed_writers_drained:
            raise RuntimeError("phase00_controller_lease_not_verified")
        existing = read_phase00_maintenance_state(root)
        if existing["blocking"]:
            raise RuntimeError(
                "phase00_maintenance_already_active:"
                f"{existing.get('maintenance_id', 'unknown')}"
            )
        maintenance_material = {
            "schema_version": SCHEMA_VERSION,
            "root": str(root),
            "started_at_utc": started_at.isoformat(),
            "expires_at_utc": (
                started_at + timedelta(seconds=ttl_seconds)
            ).isoformat(),
            "reason": reason.strip(),
            "controller_pid": os.getpid(),
        }
        maintenance_id = "phase00maint_" + sha256(
            _canonical_json(maintenance_material).encode("utf-8")
        ).hexdigest()[:20]
        unmigrated_surfaces = unmigrated_publication_surface_ids(root)
        unpaired_staging_surfaces = unpaired_staging_surface_ids(root)
        unfenced_financial_surfaces = unfenced_financial_effect_surface_ids(root)
        runtime_observation, runtime_blockers = _runtime_quiescence_observation(
            runner
        )
        receipt = {
            **maintenance_material,
            "maintenance_id": maintenance_id,
            "event": "START",
            "status": "ACTIVE",
            "governed_writers_drained": governed_writers_drained,
            "governed_writer_drain_method": "exclusive_controller_flock",
            "new_governed_writers_blocked": True,
            "runtime_observation": runtime_observation,
            "runtime_blockers": runtime_blockers,
            "known_ungoverned_publication_surfaces": list(
                unmigrated_surfaces
            ),
            "known_unpaired_staging_surfaces": list(
                unpaired_staging_surfaces
            ),
            "known_unfenced_financial_effect_surfaces": list(
                unfenced_financial_surfaces
            ),
            "research_only": True,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        receipt_path = (
            root
            / CONTROL_ROOT
            / "maintenance"
            / f"{maintenance_id}_start.json"
        )
        marker_payload = {
            "schema_version": SCHEMA_VERSION,
            "maintenance_id": maintenance_id,
            "started_at_utc": receipt["started_at_utc"],
            "expires_at_utc": receipt["expires_at_utc"],
            "reason": receipt["reason"],
            "start_receipt_path": _relative(receipt_path, root),
            "start_receipt_sha256": _payload_sha256(receipt),
        }
        active_path = root / ACTIVE_MAINTENANCE
        with _phase00_control_publication_authority(
            root=root,
            run_id=maintenance_id,
            intended_slot_id=maintenance_id,
            source_fingerprint_sha256=sha256(
                _canonical_json(
                    {
                        "maintenance_material": maintenance_material,
                        "publication_surfaces": list(unmigrated_surfaces),
                        "financial_effect_surfaces": list(
                            unfenced_financial_surfaces
                        ),
                    }
                ).encode("utf-8")
            ).hexdigest(),
            runtime_fingerprint_sha256=sha256(
                _canonical_json(runtime_observation).encode("utf-8")
            ).hexdigest(),
            configuration_fingerprint_sha256=_configuration_fingerprint(root),
            target_paths=(receipt_path, marker, active_path),
        ):
            write_immutable_json(
                receipt_path,
                receipt,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                marker,
                _pretty_json(marker_payload),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                active_path,
                _pretty_json(receipt),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
    return CommandResult(
        paths={
            "maintenance_marker": marker,
            "maintenance_receipt": receipt_path,
            "active_maintenance": active_path,
        },
        summary=receipt,
    )


def build_phase00_quiesced_checkpoint(
    *,
    root: Path,
    now: datetime | None = None,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    between_manifests: Callable[[], None] | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> CommandResult:
    """Capture two equal source/evidence manifests while the global lease is held."""

    root = _canonical_root(root)
    observed_at = _as_utc(now)
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        maintenance = read_phase00_maintenance_state(root)
        if not maintenance["blocking"] or maintenance["status"] not in {
            "ACTIVE",
            "BLOCKED_EXPIRED_REQUIRES_EXPLICIT_RESUME",
        }:
            raise RuntimeError("phase00_checkpoint_requires_active_maintenance")
        first = _build_stable_manifest(root, runner=runner)
        first_bytes = _canonical_json(first).encode("utf-8")
        if between_manifests is not None:
            between_manifests()
        second = _build_stable_manifest(root, runner=runner)
        second_bytes = _canonical_json(second).encode("utf-8")
        stable = first_bytes == second_bytes
        stable_manifest = second if stable else {}
        first_sha = sha256(first_bytes).hexdigest()
        second_sha = sha256(second_bytes).hexdigest()
        freshness_manifest_sha = sha256(
            _canonical_json(
                _stable_manifest_freshness_projection(second)
            ).encode("utf-8")
        ).hexdigest()
        source_rows = list(second.get("source_evidence_index", []))
        blockers = list(second.get("blockers", []))
        if not stable:
            blockers.insert(0, "quiesced_manifests_not_byte_identical")
        blockers = list(dict.fromkeys(blockers))
        checkpoint_pass = stable and not blockers
        checkpoint_material = {
            "schema_version": SCHEMA_VERSION,
            "maintenance_id": str(maintenance.get("maintenance_id", "")),
            "observed_at_utc": observed_at.isoformat(),
            "first_manifest_sha256": first_sha,
            "second_manifest_sha256": second_sha,
            "snapshot_stable": stable,
            "stable_manifest_sha256": second_sha if stable else "",
            "freshness_manifest_sha256": (
                freshness_manifest_sha if stable else ""
            ),
            "blockers": blockers,
        }
        checkpoint_id = "phase00checkpoint_" + sha256(
            _canonical_json(checkpoint_material).encode("utf-8")
        ).hexdigest()[:20]
        manifest_path = (
            root / CONTROL_ROOT / "manifests" / f"{checkpoint_id}.json"
        )
        receipt_path = (
            root / CONTROL_ROOT / "checkpoints" / f"{checkpoint_id}.json"
        )
        runtime_fingerprint = sha256(
            _canonical_json(second.get("runtime_contracts", [])).encode(
                "utf-8"
            )
        ).hexdigest()
        configuration_fingerprint = _configuration_fingerprint(root)
        stable_manifest_file_sha256 = ""
        receipt = {
            **checkpoint_material,
            "checkpoint_id": checkpoint_id,
            "status": (
                "PASS_QUIESCED_BASELINE"
                if checkpoint_pass
                else "CAPTURED_BLOCKED_PHASE00"
            ),
            "checkpoint_capture_complete": checkpoint_pass,
            "producer_publication_lease_active": False,
            "controller_holds_exclusive_publication_lease": True,
            "implementation_status": "IMPLEMENTED",
            "descendant_regeneration_status": second.get(
                "descendant_control", {}
            ).get("descendant_regeneration_status", "NOT_STARTED"),
            "descendant_control_status": second.get(
                "descendant_control", {}
            ).get("status", "NOT_STARTED"),
            "phase00_closure_status": second.get("phase00_closure", {}).get(
                "status", "NOT_STARTED"
            ),
            "stable_manifest_path": _relative(manifest_path, root) if stable else "",
            "stable_manifest_file_sha256": stable_manifest_file_sha256,
            "checkpoint_receipt_path": _relative(receipt_path, root),
            "runtime_observation": second.get("runtime_observation", {}),
            "known_ungoverned_publication_surfaces": [
                row["surface_id"]
                for row in second.get("publication_surface_index", [])
                if row["migration_state"] == "UNMIGRATED"
                and row["authority_scope"]
                in {"phase00_blocking", "execution_blocking"}
            ],
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        active_path = root / ACTIVE_CHECKPOINT
        index_path = root / "reports/active/phase00_source_evidence_index.csv"
        with _phase00_control_publication_authority(
            root=root,
            run_id=checkpoint_id,
            intended_slot_id=str(maintenance.get("maintenance_id", "")),
            source_fingerprint_sha256=second_sha,
            runtime_fingerprint_sha256=runtime_fingerprint,
            configuration_fingerprint_sha256=configuration_fingerprint,
            target_paths=(manifest_path, receipt_path, active_path, index_path),
        ):
            if stable:
                write_immutable_json(
                    manifest_path,
                    stable_manifest,
                    publication_scope=PHASE00_CONTROL_SCOPE,
                )
                stable_manifest_file_sha256 = _file_sha256(manifest_path)
                receipt["stable_manifest_file_sha256"] = (
                    stable_manifest_file_sha256
                )
            write_immutable_json(
                receipt_path,
                receipt,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                active_path,
                _pretty_json(receipt),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                index_path,
                _csv_text(source_rows),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
    return CommandResult(
        paths={
            "checkpoint_receipt": receipt_path,
            "stable_manifest": manifest_path,
            "active_checkpoint": active_path,
            "source_evidence_index": index_path,
        },
        summary=receipt,
    )


def seal_phase00_active_artifact_lineage(
    *,
    root: Path,
    now: datetime | None = None,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
) -> CommandResult:
    """Seal all 74 artifact candidates as immutable, non-authoritative decisions."""

    root = _canonical_root(root)
    recorded_at = _as_utc(now)
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        maintenance = read_phase00_maintenance_state(root)
        if not maintenance["blocking"]:
            raise RuntimeError("phase00_lineage_seal_requires_active_maintenance")
        checkpoint = _read_json(root / ACTIVE_CHECKPOINT)
        if (
            checkpoint.get("snapshot_stable") is not True
            or not checkpoint.get("stable_manifest_path")
            or not checkpoint.get("stable_manifest_file_sha256")
        ):
            raise RuntimeError("phase00_lineage_seal_requires_stable_checkpoint")
        stable_manifest_path = root / str(checkpoint["stable_manifest_path"])
        if (
            stable_manifest_path.is_symlink()
            or not stable_manifest_path.is_file()
            or _file_sha256(stable_manifest_path)
            != str(checkpoint["stable_manifest_file_sha256"])
        ):
            raise RuntimeError("phase00_lineage_seal_checkpoint_manifest_invalid")
        stable_manifest = _read_json(stable_manifest_path)
        if not stable_manifest:
            raise RuntimeError("phase00_lineage_seal_checkpoint_manifest_unreadable")
        source_fingerprint = str(checkpoint["stable_manifest_file_sha256"])
        runtime_fingerprint = sha256(
            _canonical_json(stable_manifest.get("runtime_contracts", [])).encode(
                "utf-8"
            )
        ).hexdigest()
        configuration_fingerprint = _configuration_fingerprint(root)
        bundle = build_active_artifact_lineage_bundle(
            root=root,
            recorded_at_utc=recorded_at.isoformat(),
            source_fingerprint_sha256=source_fingerprint,
            runtime_fingerprint_sha256=runtime_fingerprint,
            configuration_fingerprint_sha256=configuration_fingerprint,
        )
        manifest_path = root / bundle["manifest_path"]
        pointer_path = root / bundle["pointer_path"]
        envelope_paths = tuple(
            root / envelope["path"] for envelope in bundle["envelopes"]
        )
        with _phase00_control_publication_authority(
            root=root,
            run_id=str(bundle["manifest"]["manifest_id"]),
            intended_slot_id=str(maintenance.get("maintenance_id", "")),
            source_fingerprint_sha256=source_fingerprint,
            runtime_fingerprint_sha256=runtime_fingerprint,
            configuration_fingerprint_sha256=configuration_fingerprint,
            target_paths=(*envelope_paths, manifest_path, pointer_path),
        ):
            for envelope in bundle["envelopes"]:
                _write_immutable_json_idempotent(
                    root / envelope["path"], envelope["payload"]
                )
            _write_immutable_json_idempotent(
                manifest_path, bundle["manifest"]
            )
            atomic_write_text(
                pointer_path,
                _pretty_json(bundle["pointer"]),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
        envelope_index, envelope_issues = load_active_artifact_envelope_index(root)
        if envelope_issues or len(envelope_index) != 74:
            raise RuntimeError(
                "phase00_active_artifact_lineage_validation_failed:"
                + ";".join(sorted({issue.code for issue in envelope_issues}))
            )
        rows = active_artifact_rows(root)
        unresolved = [
            str(row["path"])
            for row in rows
            if str(row["resolution_status"]).startswith("UNRESOLVED")
            or row["domain_validation_status"] not in {"PASS", "NOT_APPLICABLE"}
            or row["artifact_role"] == "unregistered"
            or row["authority_eligible"] is not False
        ]
        if unresolved or len(rows) != 74:
            raise RuntimeError(
                "phase00_active_artifact_lineage_incomplete:"
                + ";".join(unresolved)
            )
        summary = {
            "schema_version": "thewiz.phase00.active_artifact_seal.v1",
            "status": "PASS_FAIL_CLOSED_LINEAGE_SEAL",
            "maintenance_id": str(maintenance.get("maintenance_id", "")),
            "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
            "recorded_at_utc": recorded_at.isoformat(),
            "registry_rows": len(rows),
            "immutable_envelopes": len(envelope_index),
            "unresolved_rows": 0,
            "authority_eligible_rows": 0,
            "source_fingerprint_sha256": source_fingerprint,
            "runtime_fingerprint_sha256": runtime_fingerprint,
            "configuration_fingerprint_sha256": configuration_fingerprint,
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": _file_sha256(manifest_path),
            "active_pointer_path": _relative(pointer_path, root),
            "active_pointer_sha256": _file_sha256(pointer_path),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
    return CommandResult(
        paths={
            "active_artifact_lineage": pointer_path,
            "active_artifact_manifest": manifest_path,
        },
        summary=summary,
    )


def resume_phase00_maintenance(
    *,
    root: Path,
    maintenance_id: str,
    now: datetime | None = None,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
) -> CommandResult:
    """Write a durable resume receipt and remove the matching marker."""

    root = _canonical_root(root)
    resumed_at = _as_utc(now)
    marker_path = root / PHASE00_MAINTENANCE_MARKER
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        maintenance = read_phase00_maintenance_state(root)
        observed_id = str(maintenance.get("maintenance_id", ""))
        if not maintenance["blocking"]:
            raise RuntimeError("phase00_maintenance_not_active")
        if not maintenance_id or maintenance_id != observed_id:
            raise RuntimeError("phase00_maintenance_id_mismatch")
        checkpoint_path = root / ACTIVE_CHECKPOINT
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("maintenance_id") != maintenance_id:
            raise RuntimeError("phase00_resume_requires_matching_checkpoint")
        if (
            checkpoint.get("status") != "PASS_QUIESCED_BASELINE"
            or checkpoint.get("checkpoint_capture_complete") is not True
            or checkpoint.get("blockers") != []
        ):
            raise RuntimeError("phase00_resume_requires_complete_checkpoint_capture")
        checkpoint_id = str(checkpoint.get("checkpoint_id", ""))
        expected_receipt_path = (
            root / CONTROL_ROOT / "checkpoints" / f"{checkpoint_id}.json"
        )
        declared_receipt_path = root / str(
            checkpoint.get("checkpoint_receipt_path", "")
        )
        if (
            not checkpoint_id.startswith("phase00checkpoint_")
            or declared_receipt_path != expected_receipt_path
            or expected_receipt_path.is_symlink()
            or not expected_receipt_path.is_file()
            or _read_json(expected_receipt_path) != checkpoint
        ):
            raise RuntimeError("phase00_resume_checkpoint_receipt_invalid")
        manifest_path = root / str(checkpoint.get("stable_manifest_path", ""))
        if (
            manifest_path.is_symlink()
            or not manifest_path.is_file()
            or _file_sha256(manifest_path)
            != str(checkpoint.get("stable_manifest_file_sha256", ""))
        ):
            raise RuntimeError("phase00_resume_manifest_invalid")
        receipt_material = {
            "schema_version": SCHEMA_VERSION,
            "maintenance_id": maintenance_id,
            "event": "RESUME",
            "resumed_at_utc": resumed_at.isoformat(),
            "checkpoint_id": str(checkpoint.get("checkpoint_id", "")),
            "checkpoint_status": str(checkpoint.get("status", "BLOCKED")),
            "checkpoint_receipt_sha256": _file_sha256(expected_receipt_path),
            "status": "RESUMED",
            "research_only": True,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        receipt_id = "phase00resume_" + sha256(
            _canonical_json(receipt_material).encode("utf-8")
        ).hexdigest()[:20]
        receipt = {**receipt_material, "receipt_id": receipt_id}
        receipt_path = (
            root / CONTROL_ROOT / "maintenance" / f"{maintenance_id}_resume.json"
        )
        active_path = root / ACTIVE_MAINTENANCE
        stable_manifest = _read_json(manifest_path)
        with _phase00_control_publication_authority(
            root=root,
            run_id=receipt_id,
            intended_slot_id=maintenance_id,
            source_fingerprint_sha256=str(
                checkpoint["stable_manifest_file_sha256"]
            ),
            runtime_fingerprint_sha256=sha256(
                _canonical_json(
                    stable_manifest.get("runtime_contracts", [])
                ).encode("utf-8")
            ).hexdigest(),
            configuration_fingerprint_sha256=_configuration_fingerprint(root),
            target_paths=(receipt_path, active_path),
        ):
            write_immutable_json(
                receipt_path,
                receipt,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            marker_path.unlink()
            _fsync_directory(marker_path.parent)
            atomic_write_text(
                active_path,
                _pretty_json(receipt),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
    return CommandResult(
        paths={
            "resume_receipt": receipt_path,
            "active_maintenance": active_path,
        },
        summary=receipt,
    )


@contextmanager
def _phase00_control_publication_authority(
    *,
    root: Path,
    run_id: str,
    intended_slot_id: str,
    source_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    configuration_fingerprint_sha256: str,
    target_paths: tuple[Path, ...],
) -> Iterator[PublicationAuthoritySession]:
    """Issue narrow file-publication authority for one controller operation."""

    canonical_root = root.resolve()
    targets = tuple(
        dict.fromkeys(Path(os.path.abspath(path)) for path in target_paths)
    )
    if not targets:
        raise EffectAuthorityError("phase00_publication_targets_required")
    if any(
        target != canonical_root and canonical_root not in target.parents
        for target in targets
    ):
        raise EffectAuthorityError("phase00_publication_target_outside_root")

    existing = current_publication_authority()
    if existing is not None:
        session = require_publication_authority(
            root=canonical_root,
            publication_scope=PHASE00_CONTROL_SCOPE,
        )
        if any(
            not any(
                target == prefix or prefix in target.parents
                for prefix in session.allowed_target_prefixes
            )
            for target in targets
        ):
            raise EffectAuthorityError(
                "phase00_publication_target_denied_by_existing_session"
            )
        yield session
        return

    authority = EffectAuthority(
        root=canonical_root,
        secret=secrets.token_bytes(32),
        issuer_id="phase00_control",
        profile=PHASE00_REPAIR_PROFILE,
    )
    with publication_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot_id,
        policy_version=PHASE00_CONTROL_PUBLICATION_POLICY_VERSION,
        source_fingerprint_sha256=source_fingerprint_sha256,
        runtime_fingerprint_sha256=runtime_fingerprint_sha256,
        configuration_fingerprint_sha256=configuration_fingerprint_sha256,
        allowed_scopes=frozenset({PHASE00_CONTROL_SCOPE}),
        allowed_target_prefixes=targets,
        max_total_bytes=MAX_PHASE00_CONTROL_PUBLICATION_BYTES,
    ) as session:
        yield session


def _build_stable_manifest(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    branch = _run_text(
        runner, ["git", "-C", str(root), "branch", "--show-current"]
    )
    head = _run_text(
        runner, ["git", "-C", str(root), "rev-parse", "HEAD"]
    )
    dirty_paths = _dirty_paths(root, runner=runner)
    source_rows = [_file_index_row(root, path) for path in dirty_paths]
    pointer_rows = _active_pointer_rows(root)
    lease_rows = _lease_rows(root)
    publication_rows = publication_surface_rows(root)
    financial_effect_rows = financial_effect_surface_rows(root)
    from quant_platform.orchestration.corrective_phase00_descendants import (
        observe_phase00_descendant_control,
    )

    descendant_control = observe_phase00_descendant_control(
        root,
        allow_superseded_lineage=True,
    )
    from quant_platform.orchestration.corrective_phase00_closure import (
        observe_phase00_closure,
    )

    phase00_closure = observe_phase00_closure(root, verify_current_tree=False)
    runtime_contracts = [
        scheduler_runtime_contract(root, contract=contract)
        for contract in SCHEDULER_CONTRACTS
    ]
    blockers: list[str] = []
    blockers.extend(
        f"unresolved_mutable_pointer:{row['path']}"
        for row in pointer_rows
        if str(row["resolution_status"]).startswith("UNRESOLVED")
    )
    blockers.extend(
        f"phase00_closure_invalid:{blocker}"
        for blocker in phase00_closure.get("blockers", [])
    )
    blockers.extend(
        f"domain_validation_incomplete:{row['path']}:{row['domain_validation_status']}"
        for row in pointer_rows
        if row["artifact_role"]
        in {"authority_pointer", "status_pointer", "blocked_decision"}
        and row["domain_validation_status"] not in {"PASS", "NOT_APPLICABLE"}
    )
    blockers.extend(
        f"unknown_dirty_path:{row['path']}"
        for row in source_rows
        if row["classification"] == "unknown"
    )
    blockers.extend(
        f"source_hash_unavailable:{row['path']}:{row['hash_status']}"
        for row in source_rows
        if row["classification"] != "outside_quant_project_scope"
        and row["artifact_type"] == "file"
        and row["hash_status"] != "HASHED"
    )
    blockers.extend(
        f"ungoverned_publication_surface:{row['surface_id']}:{row['source_path']}:{row['line']}"
        for row in publication_rows
        if row["migration_state"] == "UNMIGRATED"
    )
    unpaired_staging_ids = set(unpaired_staging_surface_ids(root))
    blockers.extend(
        f"unpaired_staging_surface:{row['surface_id']}:{row['source_path']}:{row['line']}"
        for row in publication_rows
        if row["surface_id"] in unpaired_staging_ids
    )
    blockers.extend(
        "unfenced_financial_effect_surface:"
        f"{row['surface_id']}:{row['source_path']}:{row['line']}:{row['method']}"
        for row in financial_effect_rows
        if row["migration_state"] != "MIGRATED"
    )
    runtime_observation, runtime_blockers = _runtime_quiescence_observation(
        runner
    )
    blockers.extend(runtime_blockers)
    blockers.extend(
        f"descendant_control_invalid:{blocker}"
        for blocker in descendant_control.get("blockers", [])
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "git_branch": branch,
        "git_head": head,
        "git_dirty": bool(dirty_paths),
        "dirty_path_count": len(dirty_paths),
        "source_evidence_index": source_rows,
        "active_pointer_index": pointer_rows,
        "lease_index": lease_rows,
        "publication_surface_index": publication_rows,
        "financial_effect_surface_index": financial_effect_rows,
        "runtime_contracts": runtime_contracts,
        "runtime_observation": runtime_observation,
        "descendant_control": descendant_control,
        "phase00_closure": phase00_closure,
        "blockers": list(dict.fromkeys(blockers)),
    }


def _stable_manifest_freshness_projection(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Remove process-ephemeral telemetry from cross-command freshness only."""

    projected = json.loads(_canonical_json(manifest))
    telemetry_paths = {
        ".runtime_control/effect_authority.sqlite3": (
            "CONTROL_JOURNAL_SEMANTIC_ONLY"
        ),
        ".runtime_locks/governed_evidence.lock": "LOCK_TELEMETRY_SEMANTIC_ONLY",
    }
    projected["phase00_closure"] = {
        "cross_process_freshness": "CONTROL_OUTPUT_EXCLUDED"
    }
    projected["source_evidence_index"] = [
        (
            {
                "path": row.get("path", ""),
                "artifact_type": row.get("artifact_type", ""),
                "classification": row.get("classification", ""),
                "hash_status": row.get("hash_status", ""),
                "content_captured": row.get("content_captured", False),
                "implementation_status": row.get("implementation_status", ""),
                "descendant_regeneration_status": row.get(
                    "descendant_regeneration_status", ""
                ),
                "cross_process_freshness": telemetry_paths[str(row.get("path"))],
            }
            if row.get("path") in telemetry_paths
            else row
        )
        for row in projected.get("source_evidence_index", [])
    ]
    projected["lease_index"] = [
        {
            "path": row.get("path", ""),
            "content_captured": row.get("content_captured", False),
            "cross_process_freshness": "LOCK_TELEMETRY_SEMANTIC_ONLY",
        }
        for row in projected.get("lease_index", [])
    ]
    runtime = projected.get("runtime_observation", {})
    if isinstance(runtime, dict):
        processes = runtime.get("processes", {})
        launchd = runtime.get("launchd", [])
        projected["runtime_observation"] = {
            "processes": {
                "status": processes.get("status", ""),
                "producer_count": processes.get("producer_count", -1),
            },
            "launchd": [
                {
                    "service": row.get("service", ""),
                    "status": row.get("status", ""),
                    "loaded": row.get("loaded"),
                }
                for row in launchd
            ],
        }
    return projected


def _dirty_paths(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[str]:
    commands = (
        ["git", "-C", str(root), "diff", "--name-only", "-z"],
        ["git", "-C", str(root), "diff", "--cached", "--name-only", "-z"],
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
    )
    paths: set[str] = set()
    for command in commands:
        completed = _run(command, runner)
        if completed.returncode != 0:
            raise RuntimeError(
                f"git_manifest_command_failed:{command[3]}:{completed.returncode}"
            )
        paths.update(
            value
            for value in (completed.stdout or "").split("\0")
            if value and not _is_control_output(value)
        )
    return sorted(paths)


def _file_index_row(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    classification = _classify_path(relative)
    metadata = _file_metadata(path, relative)
    if path.is_symlink():
        kind = "symlink"
        digest = sha256(os.readlink(path).encode("utf-8")).hexdigest()
        size = len(os.readlink(path).encode("utf-8"))
        hash_status = "HASHED_LINK_TARGET"
    elif path.is_file():
        kind = "file"
        size = path.stat().st_size
        if classification == "outside_quant_project_scope":
            digest = ""
            hash_status = "SKIPPED_OUTSIDE_SCOPE"
        elif size > MAX_CHECKPOINT_HASH_BYTES:
            digest = ""
            hash_status = "BLOCKED_SIZE_LIMIT"
        else:
            digest = _file_sha256(path)
            hash_status = "HASHED"
    else:
        kind = "missing_or_directory"
        digest = ""
        size = 0
        hash_status = "NOT_A_FILE"
    return {
        "path": relative,
        "artifact_type": kind,
        "classification": classification,
        "sha256": digest,
        "hash_status": hash_status,
        "metadata_sha256": sha256(
            _canonical_json(metadata).encode("utf-8")
        ).hexdigest(),
        "size_bytes": size,
        "content_captured": False,
        "implementation_status": "CURRENT_TREE",
        "descendant_regeneration_status": "NOT_EVALUATED",
    }


def _active_pointer_rows(root: Path) -> list[dict[str, Any]]:
    return active_artifact_rows(root)


def _lease_rows(root: Path) -> list[dict[str, Any]]:
    candidates: set[Path] = set()
    runtime_locks = root / ".runtime_locks"
    if runtime_locks.is_dir():
        candidates.update(path for path in runtime_locks.iterdir() if path.is_file())
    active = root / "reports" / "active"
    if active.is_dir():
        candidates.update(path for path in active.glob("*.lock") if path.is_file())
        candidates.update(path for path in active.glob(".*.lock") if path.is_file())
    return [
        {
            "path": _relative(path, root),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
            "content_captured": False,
        }
        for path in sorted(candidates)
        if not _is_control_output(_relative(path, root))
    ]


def _bounded_process_observation(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    completed = _run(
        ["pgrep", "-fal", "quant_platform|corrective_|pytest"],
        runner,
        timeout=COMMAND_TIMEOUT_SECONDS,
    )
    producer_markers = (
        "corrective_daily_scheduler",
        "corrective_l2_scheduler",
        "corrective_wizard_proof_scheduler",
        "corrective_wizard_proof_launcher",
    )
    lines = [
        line
        for line in (completed.stdout or "").splitlines()[:200]
        if "pgrep -fal" not in line
        and any(marker in line for marker in producer_markers)
    ]
    observation_status = (
        "BLOCKED_PRODUCERS_PRESENT"
        if lines
        else "PASS_NO_PRODUCERS"
        if completed.returncode in {0, 1}
        else "PASS_NOT_APPLICABLE"
        if completed.returncode == 127
        else "BLOCKED_OBSERVATION_FAILED"
    )
    return {
        "status": observation_status,
        "returncode": completed.returncode,
        "producer_count": len(lines),
        "output_sha256": sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "output_captured": False,
    }


def _bounded_launchd_observation(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for contract in SCHEDULER_CONTRACTS:
        completed = _run(
            ["launchctl", "print", f"gui/{os.getuid()}/{contract.label}"],
            runner,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        output = (completed.stdout or "")[:1024 * 1024]
        rows.append(
            {
                "service": contract.key,
                "returncode": completed.returncode,
                "loaded": completed.returncode == 0,
                "status": (
                    "BLOCKED_LOADED"
                    if completed.returncode == 0
                    else "PASS_UNLOADED"
                    if completed.returncode == 113
                    else "PASS_NOT_APPLICABLE"
                    if completed.returncode == 127
                    else "BLOCKED_OBSERVATION_FAILED"
                ),
                "output_sha256": sha256(output.encode("utf-8")).hexdigest(),
                "output_truncated": len(completed.stdout or "") > len(output),
                "output_captured": False,
            }
        )
    return rows


def _runtime_quiescence_observation(
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[dict[str, Any], list[str]]:
    processes = _bounded_process_observation(runner)
    launchd = _bounded_launchd_observation(runner)
    blockers: list[str] = []
    if processes["status"] == "BLOCKED_OBSERVATION_FAILED":
        blockers.append("producer_process_observation_failed")
    if int(processes.get("producer_count", 0)) > 0:
        blockers.append("producer_processes_still_running")
    blockers.extend(
        f"launchd_service_not_quiesced:{row['service']}:{row['status']}"
        for row in launchd
        if row["status"] not in {"PASS_UNLOADED", "PASS_NOT_APPLICABLE"}
    )
    return {"processes": processes, "launchd": launchd}, blockers


def _run_text(
    runner: Callable[..., subprocess.CompletedProcess[str]], command: list[str]
) -> str:
    completed = _run(command, runner)
    if completed.returncode != 0:
        raise RuntimeError(f"command_failed:{command[-1]}:{completed.returncode}")
    return (completed.stdout or "").strip()


def _run(
    command: list[str],
    runner: Callable[..., subprocess.CompletedProcess[str]],
    *,
    timeout: float = COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, 124, stdout="", stderr=safe_exception_code(exc))
    except OSError as exc:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"{safe_exception_code(exc)}",
        )


def _classify_path(path: str) -> str:
    if path in {".env", ".env.local"} or path.startswith(".env."):
        return "sensitive_secret_metadata_only"
    if path.startswith(("src/", "scripts/")):
        return "runtime_source"
    if path.startswith("tests/"):
        return "test_source"
    if path.startswith("config/") or path in {
        ".gitignore",
        "pyproject.toml",
        "uv.lock",
    }:
        return "runtime_configuration"
    if path.startswith((".runtime_control/", ".runtime_locks/", ".runtime_agents/")):
        return "runtime_control"
    if path.startswith("reports/active/"):
        return "active_evidence"
    if path.startswith(("docs/", "reports/diagnostics/", "reports/plans/")):
        return "documentation_or_diagnostic"
    if path.startswith(("data/", "models/", "reports/")):
        return "research_evidence"
    if path.startswith("apps/"):
        return "outside_quant_project_scope"
    return "unknown"


def _is_control_output(path: str) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in CONTROL_OUTPUT_PREFIXES)


def _csv_text(rows: list[dict[str, Any]]) -> str:
    fields = [
        "path",
        "artifact_type",
        "classification",
        "sha256",
        "hash_status",
        "metadata_sha256",
        "size_bytes",
        "content_captured",
        "implementation_status",
        "descendant_regeneration_status",
    ]
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _canonical_root(root: Path) -> Path:
    resolved = root.resolve()
    if not (resolved / ".git").exists():
        raise ValueError(f"phase00 root is not a Git repository: {resolved}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        identity = path.lstat()
    except OSError:
        return {}
    if (
        not stat_module.S_ISREG(identity.st_mode)
        or identity.st_nlink != 1
        or identity.st_size > MAX_CHECKPOINT_HASH_BYTES
    ):
        return {}
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, TypeError, UnicodeDecodeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("json_duplicate_key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"json_nonfinite_constant:{value}")


def _payload_sha256(payload: dict[str, Any]) -> str:
    return sha256(_pretty_json(payload).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: Path, relative: str) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError:
        return {"path": relative, "status": "missing"}
    return {
        "path": relative,
        "mode": stat.st_mode,
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "symlink_target": os.readlink(path) if path.is_symlink() else "",
    }


def _write_immutable_json_idempotent(path: Path, payload: dict[str, Any]) -> None:
    expected = _pretty_json(payload).encode("utf-8")
    if path.exists() or path.is_symlink():
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_nlink != 1
            or path.read_bytes() != expected
        ):
            raise RuntimeError(f"immutable Phase 00 collision: {path}")
        return
    write_immutable_json(
        path,
        payload,
        publication_scope=PHASE00_CONTROL_SCOPE,
    )


def _configuration_fingerprint(root: Path) -> str:
    candidates = [root / "pyproject.toml", root / "uv.lock"]
    config = root / "config"
    if config.is_dir():
        candidates.extend(sorted(path for path in config.rglob("*") if path.is_file()))
    rows: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise RuntimeError(f"configuration fingerprint path invalid: {path}")
        if path.stat().st_size > MAX_CHECKPOINT_HASH_BYTES:
            raise RuntimeError(f"configuration fingerprint path too large: {path}")
        rows.append(
            {
                "path": _relative(path, root),
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not rows:
        raise RuntimeError("configuration fingerprint has no inputs")
    return sha256(_canonical_json(rows).encode("utf-8")).hexdigest()


def _pretty_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve(strict=False))


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
