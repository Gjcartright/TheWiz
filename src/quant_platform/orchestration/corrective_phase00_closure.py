"""Exact-tree, no-external-effect Phase 00 verification and closure receipts."""

from __future__ import annotations

import csv
import os
import shlex
import shutil
import stat
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_phase00_control import (
    DEFAULT_LOCK_WAIT_SECONDS,
    _canonical_json,
    _configuration_fingerprint,
    _phase00_control_publication_authority,
)
from quant_platform.orchestration.corrective_phase00_descendants import (
    ZERO_AUTHORITY,
    _file_sha256,
    _json_file_sha256,
    _load_strict_object,
    _pretty_json,
    _relative,
    _require_current_checkpoint,
    observe_phase00_descendant_control,
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

SCHEMA_VERSION = "thewiz.phase00.closure_receipt.v2"
MANIFEST_SCHEMA_VERSION = "thewiz.phase00.closure_manifest.v2"
POINTER_SCHEMA_VERSION = "thewiz.phase00.closure_pointer.v2"
SOURCE_SCHEMA_VERSION = "thewiz.phase00.source_receipt.v2"
RUNTIME_SCHEMA_VERSION = "thewiz.phase00.runtime_receipt.v1"
TEST_SCHEMA_VERSION = "thewiz.phase00.test_receipt.v4"
GIT_SCHEMA_VERSION = "thewiz.phase00.git_receipt.v1"
CONTROL_EVIDENCE_SCHEMA_VERSION = "thewiz.phase00.control_evidence.v1"
CONTROL_ROOT = Path("data/research/phase00_control/closures")
ACTIVE_POINTER = Path("reports/active/phase00_closure.json")
ACTIVE_ACCEPTANCE = Path("reports/active/phase00_acceptance_matrix.csv")
ACTIVE_FAULT_CATALOG = Path("reports/active/phase00_fault_catalog.csv")
HISTORICAL_ACCEPTANCE = Path(
    "reports/diagnostics/2026-08-21_phase00_scheduler_lineage_acceptance.csv"
)
HISTORICAL_FAULT_CATALOG = Path(
    "reports/diagnostics/2026-08-21_phase00_new_test_catalog.csv"
)
CONTROL_EVIDENCE_MAP = Path("config/phase00_control_evidence_map.json")
MAX_CAPTURE_BYTES = 32 * 1024**2
DEFAULT_VERIFICATION_TIMEOUT_SECONDS = 20 * 60.0
PYTEST_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny process-exec (literal \"/usr/bin/security\"))
(deny process-exec (literal \"/usr/bin/curl\"))
(deny process-exec (literal \"/usr/bin/ftp\"))
(deny process-exec (literal \"/usr/bin/nc\"))
(deny process-exec (literal \"/usr/bin/scp\"))
(deny process-exec (literal \"/usr/bin/sftp\"))
(deny process-exec (literal \"/usr/bin/ssh\"))
"""
EXPECTED_LAUNCHD_SERVICES = {
    "daily_research",
    "hyperliquid_l2",
    "wizard_proof",
}
REQUIRED_BUNDLE_FILES = frozenset(
    {
        "source_receipt.json",
        "runtime_receipt.json",
        "test_receipt.json",
        "git_receipt.json",
        "runtime_smoke.stdout.txt",
        "runtime_smoke.stderr.txt",
        "pytest.stdout.txt",
        "pytest.stderr.txt",
        "pytest.xml",
        "ruff.stdout.txt",
        "ruff.stderr.txt",
        "cli_fatal.stdout.txt",
        "cli_fatal.stderr.txt",
        "pip_check.stdout.txt",
        "pip_check.stderr.txt",
        "acceptance_matrix.csv",
        "fault_catalog.csv",
        "control_evidence.json",
    }
)
DEFERRED_OPERATIONAL_FAULTS = frozenset({"T00-030", "T00-035"})
RELOAD_CONTROLS = frozenset({"P00-SL-020", "P00-SL-021"})
ALLOWED_CLOSURE_STATUSES = frozenset(
    {
        "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY",
        "BLOCKED_PHASE00_CLOSURE_VERIFICATION",
    }
)
LEGACY_V1_POINTER_SCHEMA_VERSION = "thewiz.phase00.closure_pointer.v1"
LEGACY_V1_CLOSURE_SCHEMA_VERSION = "thewiz.phase00.closure_receipt.v1"
LEGACY_V1_MANIFEST_SCHEMA_VERSION = "thewiz.phase00.closure_manifest.v1"
LEGACY_V1_REQUIRED_BUNDLE_FILES = frozenset(
    {
        "source_receipt.json",
        "runtime_receipt.json",
        "test_receipt.json",
        "git_receipt.json",
        "pytest.stdout.txt",
        "pytest.stderr.txt",
        "pytest.xml",
        "ruff.stdout.txt",
        "ruff.stderr.txt",
        "cli_fatal.stdout.txt",
        "cli_fatal.stderr.txt",
        "pip_check.stdout.txt",
        "pip_check.stderr.txt",
        "acceptance_matrix.csv",
        "fault_catalog.csv",
    }
)
LEGACY_V1_JSON_SCHEMAS = {
    "source_receipt.json": "thewiz.phase00.source_receipt.v1",
    "runtime_receipt.json": "thewiz.phase00.runtime_receipt.v1",
    "test_receipt.json": "thewiz.phase00.test_receipt.v2",
    "git_receipt.json": "thewiz.phase00.git_receipt.v1",
}


def build_phase00_closure_verification(
    *,
    root: Path,
    now: datetime | None = None,
    wait_timeout_seconds: float = DEFAULT_LOCK_WAIT_SECONDS,
    verification_timeout_seconds: float = DEFAULT_VERIFICATION_TIMEOUT_SECONDS,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
) -> CommandResult:
    """Run exact-tree local verification and publish an immutable closure bundle."""

    root = _canonical_root(root)
    started_at = _as_utc(now)
    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        preflight = _closure_preflight(root, runner=runner)

    verification = _run_verification(
        root,
        runner=runner,
        timeout_seconds=verification_timeout_seconds,
    )

    with _phase00_maintenance_controller_lock(
        root, wait_timeout_seconds=wait_timeout_seconds
    ):
        postflight = _closure_preflight(root, runner=runner)
        completed_at = datetime.now(UTC)
        if now is not None:
            completed_at = started_at
        pre_freshness = str(preflight["checkpoint"]["freshness_manifest_sha256"])
        post_freshness = str(postflight["checkpoint"]["freshness_manifest_sha256"])
        blockers = list(verification["blockers"])
        if pre_freshness != post_freshness:
            blockers.append("source_or_runtime_drift_during_verification")
        if preflight["descendant"]["control_id"] != postflight["descendant"][
            "control_id"
        ]:
            blockers.append("descendant_control_changed_during_verification")
        blockers = list(dict.fromkeys(blockers))
        material = {
            "schema_version": SCHEMA_VERSION,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "maintenance_id": preflight["maintenance_id"],
            "checkpoint_ids": preflight["checkpoint_ids"],
            "freshness_manifest_sha256": pre_freshness,
            "descendant_control_id": preflight["descendant"]["control_id"],
            "pytest_output_sha256": sha256(
                verification["pytest"]["stdout"]
                + verification["pytest"]["stderr"]
                + verification["pytest"]["junit_xml"]
            ).hexdigest(),
            "runtime_smoke_output_sha256": sha256(
                verification["runtime_smoke"]["stdout"]
                + verification["runtime_smoke"]["stderr"]
            ).hexdigest(),
            "ruff_output_sha256": sha256(
                verification["ruff"]["stdout"]
                + verification["ruff"]["stderr"]
            ).hexdigest(),
            "cli_fatal_output_sha256": sha256(
                verification["cli_fatal"]["stdout"]
                + verification["cli_fatal"]["stderr"]
            ).hexdigest(),
            "pip_check_output_sha256": sha256(
                verification["pip_check"]["stdout"]
                + verification["pip_check"]["stderr"]
            ).hexdigest(),
        }
        closure_id = "phase00closure_" + sha256(
            _canonical_json(material).encode("utf-8")
        ).hexdigest()[:24]
        control_evidence = _build_control_evidence(
            root,
            closure_id=closure_id,
            junit_xml=verification["pytest"]["junit_xml"],
        )
        if not control_evidence["implementation_controls_proven"]:
            blockers.append("phase00_acceptance_controls_not_proven")
        if not control_evidence["implementation_faults_proven"]:
            blockers.append("phase00_fault_contracts_not_proven")
        blockers = list(dict.fromkeys(blockers))
        verification_pass = not blockers
        verification["blockers"] = blockers
        output_root = root / CONTROL_ROOT / closure_id
        paths = _bundle_paths(root, output_root)
        source_receipt = _source_receipt(
            closure_id=closure_id,
            preflight=preflight,
            postflight=postflight,
            passed=verification_pass,
        )
        runtime_receipt = _runtime_receipt(
            closure_id=closure_id,
            preflight=postflight,
            passed=verification_pass,
            containment=verification["containment"],
        )
        test_receipt = _test_receipt(
            closure_id=closure_id,
            verification=verification,
            passed=verification_pass,
        )
        git_receipt = _git_receipt(
            closure_id=closure_id,
            preflight=postflight,
        )
        acceptance_rows = _acceptance_rows(
            root,
            closure_id=closure_id,
            evidence=control_evidence,
            checkpoint_ids=preflight["checkpoint_ids"],
        )
        fault_rows = _fault_catalog_rows(
            root,
            closure_id=closure_id,
            evidence=control_evidence,
        )
        acceptance_bytes = _csv_bytes(acceptance_rows)
        fault_bytes = _csv_bytes(fault_rows)
        leaf_payloads: dict[str, bytes] = {
            "source_receipt.json": _json_bytes(source_receipt),
            "runtime_receipt.json": _json_bytes(runtime_receipt),
            "test_receipt.json": _json_bytes(test_receipt),
            "git_receipt.json": _json_bytes(git_receipt),
            "runtime_smoke.stdout.txt": verification["runtime_smoke"]["stdout"],
            "runtime_smoke.stderr.txt": verification["runtime_smoke"]["stderr"],
            "pytest.stdout.txt": verification["pytest"]["stdout"],
            "pytest.stderr.txt": verification["pytest"]["stderr"],
            "pytest.xml": verification["pytest"]["junit_xml"],
            "ruff.stdout.txt": verification["ruff"]["stdout"],
            "ruff.stderr.txt": verification["ruff"]["stderr"],
            "cli_fatal.stdout.txt": verification["cli_fatal"]["stdout"],
            "cli_fatal.stderr.txt": verification["cli_fatal"]["stderr"],
            "pip_check.stdout.txt": verification["pip_check"]["stdout"],
            "pip_check.stderr.txt": verification["pip_check"]["stderr"],
            "acceptance_matrix.csv": acceptance_bytes,
            "fault_catalog.csv": fault_bytes,
            "control_evidence.json": _json_bytes(control_evidence),
        }
        entries = [
            {
                "name": name,
                "path": _relative(output_root / name, root),
                "sha256": sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(leaf_payloads.items())
        ]
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "closure_id": closure_id,
            "created_at_utc": completed_at.isoformat(),
            "entry_count": len(entries),
            "entries": entries,
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        manifest_path = output_root / "manifest.json"
        closure_path = output_root / "closure_receipt.json"
        manifest_sha = _json_file_sha256(manifest)
        status = (
            "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY"
            if verification_pass
            else "BLOCKED_PHASE00_CLOSURE_VERIFICATION"
        )
        closure_receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "closure_id": closure_id,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": manifest_sha,
            "blockers": blockers,
            "phase00_implementation_accepted": verification_pass,
            "phase00_runtime_accepted": False,
            "runtime_activation_authorized": False,
            "reload_performed": False,
            "external_calls_performed": 0,
            "credentials_accessed": False,
            "provider_credits_used": 0,
            "orders_submitted": 0,
            "external_effect_count_basis": (
                "KERNEL_DENY_SANDBOX_PLUS_TEST_CONTAINMENT"
            ),
            "external_effect_containment": dict(verification["containment"]),
            "descendant_regeneration_status": (
                "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            ),
            "model_authority": "RESEARCH_ONLY_BLOCKED_PENDING_REBUILD",
            "immutability_contract": "TAMPER_EVIDENT_WRITE_ONCE_READ_ONLY_MODE",
            "git_reproducibility_status": git_receipt[
                "git_reproducibility_status"
            ],
            "operational_boundaries_remaining": [
                "separately_approved_launchd_reload_and controlled first run",
                "reboot_or_unmount recovery exercise on disposable media",
                "no-order scheduler soak before runtime acceptance",
                "controlled descendant rebuild before ML or model authority",
            ],
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        closure_sha = _json_file_sha256(closure_receipt)
        pointer = {
            "schema_version": POINTER_SCHEMA_VERSION,
            "status": status,
            "closure_id": closure_id,
            "created_at_utc": completed_at.isoformat(),
            "closure_receipt_path": _relative(closure_path, root),
            "closure_receipt_sha256": closure_sha,
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": manifest_sha,
            "test_receipt_path": _relative(
                output_root / "test_receipt.json", root
            ),
            "test_receipt_sha256": sha256(
                leaf_payloads["test_receipt.json"]
            ).hexdigest(),
            "acceptance_matrix_path": _relative(
                output_root / "acceptance_matrix.csv", root
            ),
            "acceptance_matrix_sha256": sha256(acceptance_bytes).hexdigest(),
            "fault_catalog_path": _relative(output_root / "fault_catalog.csv", root),
            "fault_catalog_sha256": sha256(fault_bytes).hexdigest(),
            "control_evidence_path": _relative(
                output_root / "control_evidence.json", root
            ),
            "control_evidence_sha256": sha256(
                leaf_payloads["control_evidence.json"]
            ).hexdigest(),
            "phase00_runtime_accepted": False,
            "runtime_activation_authorized": False,
            "authority_flags": dict(ZERO_AUTHORITY),
        }
        publication_targets = tuple(
            [output_root / name for name in leaf_payloads]
            + [
                manifest_path,
                closure_path,
                paths["active_closure"],
                paths["active_acceptance"],
                paths["active_fault_catalog"],
            ]
        )
        source_fingerprint = sha256(
            _canonical_json(source_receipt).encode("utf-8")
        ).hexdigest()
        runtime_fingerprint = sha256(
            _canonical_json(runtime_receipt).encode("utf-8")
        ).hexdigest()
        with _phase00_control_publication_authority(
            root=root,
            run_id=closure_id,
            intended_slot_id=preflight["maintenance_id"],
            source_fingerprint_sha256=source_fingerprint,
            runtime_fingerprint_sha256=runtime_fingerprint,
            configuration_fingerprint_sha256=_configuration_fingerprint(root),
            target_paths=publication_targets,
        ):
            for name, payload in leaf_payloads.items():
                write_immutable_bytes(
                    output_root / name,
                    payload,
                    publication_scope=PHASE00_CONTROL_SCOPE,
                )
            write_immutable_json(
                manifest_path,
                manifest,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            write_immutable_json(
                closure_path,
                closure_receipt,
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                paths["active_closure"],
                _pretty_json(pointer),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                paths["active_acceptance"],
                acceptance_bytes.decode("utf-8"),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
            atomic_write_text(
                paths["active_fault_catalog"],
                fault_bytes.decode("utf-8"),
                publication_scope=PHASE00_CONTROL_SCOPE,
            )
        observation = observe_phase00_closure(root)
        if observation["validation_status"] != "PASS":
            raise RuntimeError(
                "phase00_closure_postpublication_validation_failed:"
                + ";".join(observation["blockers"])
            )
    return CommandResult(
        paths={
            **paths,
            "closure_receipt": closure_path,
            "closure_manifest": manifest_path,
            "test_receipt": output_root / "test_receipt.json",
            "source_receipt": output_root / "source_receipt.json",
            "runtime_receipt": output_root / "runtime_receipt.json",
            "git_receipt": output_root / "git_receipt.json",
        },
        summary=closure_receipt,
    )


def observe_phase00_closure(
    root: Path,
    *,
    verify_current_tree: bool = True,
) -> dict[str, Any]:
    """Validate the active closure bundle and, normally, its current-tree binding."""

    root = root.resolve()
    pointer_path = root / ACTIVE_POINTER
    if not pointer_path.exists():
        return {
            "validation_status": "NOT_STARTED",
            "status": "NOT_STARTED",
            "blockers": [],
            "runtime_activation_authorized": False,
            "authority_flags": dict(ZERO_AUTHORITY),
    }
    try:
        pointer = _load_strict_object(pointer_path)
        if pointer.get("schema_version") == LEGACY_V1_POINTER_SCHEMA_VERSION:
            return _observe_legacy_v1_closure(root, pointer)
        _require_schema(pointer, POINTER_SCHEMA_VERSION, "closure_pointer")
        _require_zero_authority(pointer, "closure_pointer")
        closure_id = _required_string(pointer, "closure_id")
        if not closure_id.startswith("phase00closure_") or len(closure_id) != 39:
            raise ValueError("closure_id_invalid")
        output_root = root / CONTROL_ROOT / closure_id
        closure_path = _exact_bound_path(
            root,
            pointer,
            "closure_receipt_path",
            output_root / "closure_receipt.json",
        )
        manifest_path = _exact_bound_path(
            root,
            pointer,
            "manifest_path",
            output_root / "manifest.json",
        )
        _require_hash(closure_path, pointer, "closure_receipt_sha256")
        _require_hash(manifest_path, pointer, "manifest_sha256")
        _require_read_only(closure_path, "closure_receipt")
        _require_read_only(manifest_path, "closure_manifest")
        closure = _load_strict_object(closure_path)
        manifest = _load_strict_object(manifest_path)
        _require_schema(closure, SCHEMA_VERSION, "closure_receipt")
        _require_schema(manifest, MANIFEST_SCHEMA_VERSION, "closure_manifest")
        _require_zero_authority(closure, "closure_receipt")
        _require_zero_authority(manifest, "closure_manifest")
        pointer_status = _required_string(pointer, "status")
        if pointer_status not in ALLOWED_CLOSURE_STATUSES:
            raise ValueError("closure_pointer_status_invalid")
        if closure.get("status") != pointer_status:
            raise ValueError("closure_pointer_status_binding_mismatch")
        if closure.get("closure_id") != closure_id or manifest.get("closure_id") != closure_id:
            raise ValueError("closure_id_binding_mismatch")
        if closure.get("manifest_path") != pointer.get("manifest_path") or closure.get(
            "manifest_sha256"
        ) != pointer.get("manifest_sha256"):
            raise ValueError("closure_manifest_cross_binding_mismatch")
        entries = manifest.get("entries")
        if not isinstance(entries, list) or int(manifest.get("entry_count", -1)) != len(
            entries
        ):
            raise ValueError("closure_manifest_entries_invalid")
        names = {str(entry.get("name", "")) for entry in entries if isinstance(entry, dict)}
        if names != REQUIRED_BUNDLE_FILES or len(entries) != len(REQUIRED_BUNDLE_FILES):
            raise ValueError("closure_manifest_file_set_invalid")
        entry_index: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("closure_manifest_entry_not_object")
            name = _required_string(entry, "name")
            path = _exact_bound_path(
                root,
                entry,
                "path",
                output_root / name,
            )
            _require_hash(path, entry, "sha256")
            _require_read_only(path, f"closure_leaf_{name}")
            if int(entry.get("size_bytes", -1)) != path.stat().st_size:
                raise ValueError("closure_manifest_size_mismatch")
            entry_index[name] = entry
        test_receipt = _load_strict_object(output_root / "test_receipt.json")
        source_receipt = _load_strict_object(output_root / "source_receipt.json")
        runtime_receipt = _load_strict_object(output_root / "runtime_receipt.json")
        git_receipt = _load_strict_object(output_root / "git_receipt.json")
        control_evidence = _load_strict_object(output_root / "control_evidence.json")
        _require_schema(test_receipt, TEST_SCHEMA_VERSION, "test_receipt")
        _require_schema(source_receipt, SOURCE_SCHEMA_VERSION, "source_receipt")
        _require_schema(runtime_receipt, RUNTIME_SCHEMA_VERSION, "runtime_receipt")
        _require_schema(git_receipt, GIT_SCHEMA_VERSION, "git_receipt")
        _require_schema(
            control_evidence,
            CONTROL_EVIDENCE_SCHEMA_VERSION,
            "control_evidence",
        )
        for label, payload in (
            ("test_receipt", test_receipt),
            ("source_receipt", source_receipt),
            ("runtime_receipt", runtime_receipt),
            ("git_receipt", git_receipt),
            ("control_evidence", control_evidence),
        ):
            _require_zero_authority(payload, label)
            if payload.get("closure_id") != closure_id:
                raise ValueError(f"{label}_closure_id_mismatch")
        if closure.get("status") == "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY":
            if any(
                payload.get("status") not in {
                    "PASS_EXACT_TREE_LOCAL_VERIFICATION",
                    "PASS_SOURCE_STABLE",
                    "PASS_RUNTIME_QUIESCED",
                    "BOUND_TO_HEAD_AND_DIRTY_MANIFEST",
                }
                for payload in (
                    test_receipt,
                    source_receipt,
                    runtime_receipt,
                    git_receipt,
                )
            ):
                raise ValueError("closure_pass_leaf_status_invalid")
            if closure.get("blockers") != []:
                raise ValueError("closure_pass_has_blockers")
            if (
                control_evidence.get("implementation_controls_proven") is not True
                or control_evidence.get("implementation_faults_proven") is not True
            ):
                raise ValueError("closure_pass_control_evidence_incomplete")
        if pointer.get("runtime_activation_authorized") is not False or closure.get(
            "runtime_activation_authorized"
        ) is not False:
            raise ValueError("closure_runtime_authority_nonzero")
        acceptance_path = root / ACTIVE_ACCEPTANCE
        fault_path = root / ACTIVE_FAULT_CATALOG
        if _file_sha256(acceptance_path) != str(
            pointer.get("acceptance_matrix_sha256", "")
        ):
            raise ValueError("active_acceptance_hash_mismatch")
        if _file_sha256(fault_path) != str(pointer.get("fault_catalog_sha256", "")):
            raise ValueError("active_fault_catalog_hash_mismatch")
        if acceptance_path.read_bytes() != (output_root / "acceptance_matrix.csv").read_bytes():
            raise ValueError("active_acceptance_content_mismatch")
        if fault_path.read_bytes() != (output_root / "fault_catalog.csv").read_bytes():
            raise ValueError("active_fault_catalog_content_mismatch")
        if entry_index["test_receipt.json"]["sha256"] != pointer.get(
            "test_receipt_sha256"
        ):
            raise ValueError("active_test_receipt_binding_mismatch")
        if entry_index["control_evidence.json"]["sha256"] != pointer.get(
            "control_evidence_sha256"
        ):
            raise ValueError("active_control_evidence_binding_mismatch")
        if verify_current_tree:
            current_checkpoint, current_manifest = _require_current_checkpoint(
                root,
                runner=subprocess.run,
            )
            current_descendant = observe_phase00_descendant_control(root)
            _require_current_tree_binding(
                source_receipt=source_receipt,
                git_receipt=git_receipt,
                checkpoint=current_checkpoint,
                manifest=current_manifest,
                descendant=current_descendant,
            )
    except (OSError, ET.ParseError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "validation_status": "BLOCKED",
            "status": "BLOCKED_INVALID_PHASE00_CLOSURE",
            "blockers": [_exception_code(exc)],
            "runtime_activation_authorized": False,
            "authority_flags": dict(ZERO_AUTHORITY),
        }
    return {
        "validation_status": "PASS",
        "status": str(pointer["status"]),
        "closure_id": str(pointer["closure_id"]),
        "closure_receipt_path": str(pointer["closure_receipt_path"]),
        "closure_receipt_sha256": str(pointer["closure_receipt_sha256"]),
        "test_receipt_path": str(pointer["test_receipt_path"]),
        "test_receipt_sha256": str(pointer["test_receipt_sha256"]),
        "runtime_activation_authorized": False,
        "blockers": [],
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _observe_legacy_v1_closure(
    root: Path,
    pointer: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the exact pre-hardening bundle as zero-authority history."""

    _require_schema(pointer, LEGACY_V1_POINTER_SCHEMA_VERSION, "closure_pointer")
    _require_zero_authority(pointer, "closure_pointer")
    closure_id = _required_string(pointer, "closure_id")
    if not closure_id.startswith("phase00closure_") or len(closure_id) != 39:
        raise ValueError("closure_id_invalid")
    output_root = root / CONTROL_ROOT / closure_id
    closure_path = _exact_bound_path(
        root,
        pointer,
        "closure_receipt_path",
        output_root / "closure_receipt.json",
    )
    manifest_path = _exact_bound_path(
        root,
        pointer,
        "manifest_path",
        output_root / "manifest.json",
    )
    _require_hash(closure_path, pointer, "closure_receipt_sha256")
    _require_hash(manifest_path, pointer, "manifest_sha256")
    closure = _load_strict_object(closure_path)
    manifest = _load_strict_object(manifest_path)
    _require_schema(closure, LEGACY_V1_CLOSURE_SCHEMA_VERSION, "closure_receipt")
    _require_schema(manifest, LEGACY_V1_MANIFEST_SCHEMA_VERSION, "closure_manifest")
    _require_zero_authority(closure, "closure_receipt")
    _require_zero_authority(manifest, "closure_manifest")
    pointer_status = _required_string(pointer, "status")
    if pointer_status not in ALLOWED_CLOSURE_STATUSES:
        raise ValueError("closure_pointer_status_invalid")
    if closure.get("status") != pointer_status:
        raise ValueError("closure_pointer_status_binding_mismatch")
    if closure.get("closure_id") != closure_id or manifest.get("closure_id") != closure_id:
        raise ValueError("closure_id_binding_mismatch")
    if closure.get("manifest_path") != pointer.get("manifest_path") or closure.get(
        "manifest_sha256"
    ) != pointer.get("manifest_sha256"):
        raise ValueError("closure_manifest_cross_binding_mismatch")
    if pointer_status == "PASS_PHASE00_IMPLEMENTATION_RESEARCH_ONLY" and closure.get(
        "blockers"
    ) != []:
        raise ValueError("closure_pass_has_blockers")

    entries = manifest.get("entries")
    if not isinstance(entries, list) or int(manifest.get("entry_count", -1)) != len(
        entries
    ):
        raise ValueError("closure_manifest_entries_invalid")
    names = {
        str(entry.get("name", ""))
        for entry in entries
        if isinstance(entry, dict)
    }
    if names != LEGACY_V1_REQUIRED_BUNDLE_FILES or len(entries) != len(
        LEGACY_V1_REQUIRED_BUNDLE_FILES
    ):
        raise ValueError("legacy_closure_manifest_file_set_invalid")
    entry_index: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise TypeError("closure_manifest_entry_not_object")
        name = _required_string(entry, "name")
        path = _exact_bound_path(root, entry, "path", output_root / name)
        _require_hash(path, entry, "sha256")
        if int(entry.get("size_bytes", -1)) != path.stat().st_size:
            raise ValueError("closure_manifest_size_mismatch")
        entry_index[name] = entry

    for name, schema in LEGACY_V1_JSON_SCHEMAS.items():
        payload = _load_strict_object(output_root / name)
        _require_schema(payload, schema, f"legacy_{name}")
        _require_zero_authority(payload, f"legacy_{name}")
        if payload.get("closure_id") != closure_id:
            raise ValueError(f"legacy_{name}_closure_id_mismatch")
    runtime_receipt = _load_strict_object(output_root / "runtime_receipt.json")
    if runtime_receipt.get("runtime_activation_authorized") is not False:
        raise ValueError("legacy_runtime_authority_nonzero")

    pointer_bindings = {
        "test_receipt.json": ("test_receipt_path", "test_receipt_sha256"),
        "acceptance_matrix.csv": (
            "acceptance_matrix_path",
            "acceptance_matrix_sha256",
        ),
        "fault_catalog.csv": ("fault_catalog_path", "fault_catalog_sha256"),
    }
    for name, (path_key, hash_key) in pointer_bindings.items():
        path = _exact_bound_path(root, pointer, path_key, output_root / name)
        _require_hash(path, pointer, hash_key)
        if entry_index[name]["sha256"] != pointer[hash_key]:
            raise ValueError(f"legacy_{name}_pointer_manifest_mismatch")

    if (
        pointer.get("phase00_runtime_accepted") is not False
        or pointer.get("runtime_activation_authorized") is not False
        or closure.get("runtime_activation_authorized") is not False
    ):
        raise ValueError("closure_runtime_authority_nonzero")
    active_acceptance = root / ACTIVE_ACCEPTANCE
    active_faults = root / ACTIVE_FAULT_CATALOG
    if _file_sha256(active_acceptance) != pointer["acceptance_matrix_sha256"]:
        raise ValueError("active_acceptance_hash_mismatch")
    if _file_sha256(active_faults) != pointer["fault_catalog_sha256"]:
        raise ValueError("active_fault_catalog_hash_mismatch")
    if active_acceptance.read_bytes() != (output_root / "acceptance_matrix.csv").read_bytes():
        raise ValueError("active_acceptance_content_mismatch")
    if active_faults.read_bytes() != (output_root / "fault_catalog.csv").read_bytes():
        raise ValueError("active_fault_catalog_content_mismatch")

    return {
        "validation_status": "PASS_HISTORICAL_ONLY",
        "status": "SUPERSEDED_LEGACY_PHASE00_CLOSURE",
        "legacy_status": pointer_status,
        "closure_id": closure_id,
        "closure_receipt_path": str(pointer["closure_receipt_path"]),
        "closure_receipt_sha256": str(pointer["closure_receipt_sha256"]),
        "legacy_immutability_contract": "HASH_BOUND_PRE_READ_ONLY_HARDENING",
        "runtime_activation_authorized": False,
        "blockers": [],
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _require_current_tree_binding(
    *,
    source_receipt: Mapping[str, Any],
    git_receipt: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    manifest: Mapping[str, Any],
    descendant: Mapping[str, Any],
) -> None:
    expected_checkpoint = _required_string(source_receipt, "post_checkpoint_id")
    expected_freshness = _required_string(
        source_receipt,
        "post_freshness_manifest_sha256",
    )
    expected_manifest_sha = _required_string(
        source_receipt,
        "post_stable_manifest_file_sha256",
    )
    if checkpoint.get("checkpoint_id") != expected_checkpoint:
        raise ValueError("closure_current_checkpoint_id_mismatch")
    if checkpoint.get("freshness_manifest_sha256") != expected_freshness:
        raise ValueError("closure_current_freshness_mismatch")
    if checkpoint.get("stable_manifest_file_sha256") != expected_manifest_sha:
        raise ValueError("closure_current_manifest_sha256_mismatch")
    if git_receipt.get("semantic_freshness_sha256") != expected_freshness:
        raise ValueError("closure_git_freshness_binding_mismatch")
    if git_receipt.get("exact_tree_manifest_sha256") != expected_manifest_sha:
        raise ValueError("closure_git_manifest_binding_mismatch")
    for key in ("git_branch", "git_head", "git_dirty", "dirty_path_count"):
        if git_receipt.get(key) != manifest.get(key):
            raise ValueError(f"closure_current_{key}_mismatch")
    if descendant.get("validation_status") != "PASS":
        raise ValueError("closure_current_descendant_control_invalid")
    if descendant.get("control_id") != source_receipt.get("descendant_control_id"):
        raise ValueError("closure_current_descendant_control_id_mismatch")
    if descendant.get("descendant_regeneration_status") != (
        "INVALIDATED_PENDING_CONTROLLED_REBUILD"
    ):
        raise ValueError("closure_current_descendant_regeneration_status_invalid")


def _closure_preflight(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> dict[str, Any]:
    maintenance = read_phase00_maintenance_state(root)
    if not maintenance.get("blocking"):
        raise RuntimeError("phase00_closure_requires_active_maintenance")
    checkpoint, manifest = _require_current_checkpoint(root, runner=runner)
    descendant = observe_phase00_descendant_control(root)
    if descendant.get("validation_status") != "PASS":
        raise RuntimeError("phase00_closure_requires_valid_descendant_control")
    if descendant.get("descendant_regeneration_status") != (
        "INVALIDATED_PENDING_CONTROLLED_REBUILD"
    ):
        raise RuntimeError("phase00_closure_descendant_status_invalid")
    _require_runtime_quiesced(checkpoint)
    checkpoint_ids = _matching_checkpoint_ids(
        root,
        freshness_sha=str(checkpoint["freshness_manifest_sha256"]),
    )
    if len(checkpoint_ids) < 2:
        raise RuntimeError("phase00_closure_requires_two_stable_checkpoints")
    return {
        "maintenance_id": str(maintenance.get("maintenance_id", "")),
        "checkpoint": checkpoint,
        "manifest": manifest,
        "checkpoint_ids": checkpoint_ids[-2:],
        "descendant": descendant,
    }


def _run_verification(
    root: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
    timeout_seconds: float,
) -> dict[str, Any]:
    environment = _sanitized_environment(root)
    python = root / ".venv/bin/python"
    ruff = root / ".venv/bin/ruff"
    if not python.is_file() or not ruff.is_file():
        raise RuntimeError("phase00_closure_canonical_runtime_missing")
    with tempfile.TemporaryDirectory(prefix="thewiz-phase00-") as temporary:
        junit_path = Path(temporary) / "pytest.xml"
        smoke_source = """import json, runpy, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
module = runpy.run_path(str(root / 'src/quant_platform/orchestration/corrective_scheduler_bootstrap.py'))
blockers = module['canonical_runtime_blockers'](root)
print(json.dumps({'status': 'PASS' if not blockers else 'BLOCKED', 'blockers': blockers}, sort_keys=True))
raise SystemExit(0 if not blockers else 78)
"""
        base_smoke_command = [
            str(python),
            "-I",
            "-c",
            smoke_source,
            str(root),
        ]
        base_pytest_command = [
            str(python),
            "-W",
            "error",
            "-m",
            "pytest",
            "-q",
            "-m",
            "not external_network",
            f"--junitxml={junit_path}",
            "tests",
        ]
        smoke_command, containment = _contained_pytest_command(
            base_smoke_command
        )
        pytest_command, pytest_containment = _contained_pytest_command(
            base_pytest_command
        )
        if pytest_containment != containment:
            raise RuntimeError("phase00_containment_contract_inconsistent")
        phase00_scope = _phase00_ruff_scope(root)
        ruff_command = [str(ruff), "check", *phase00_scope]
        cli_fatal_command = [
            str(ruff),
            "check",
            "--select",
            "E9,F63,F7,F82",
            "src/quant_platform/cli.py",
            "tests/test_cli.py",
        ]
        pip_check_command = _dependency_check_command(
            python=python,
            environment=environment,
        )
        smoke_result = _run_captured(
            runner,
            smoke_command,
            root=root,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 60.0),
        )
        if smoke_result["returncode"] == 0:
            pytest_result = _run_captured(
                runner,
                pytest_command,
                root=root,
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
        else:
            pytest_result = _not_run_result(
                pytest_command,
                reason="runtime_smoke_failed",
            )
        junit_xml = (
            junit_path.read_bytes()
            if smoke_result["returncode"] == 0 and junit_path.is_file()
            else b""
        )
        pytest_result["junit_xml"] = _bounded_bytes(junit_xml)
        pytest_result["junit"] = _parse_junit(junit_xml)
        ruff_result = _run_captured(
            runner,
            ruff_command,
            root=root,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 300.0),
        )
        cli_fatal_result = _run_captured(
            runner,
            cli_fatal_command,
            root=root,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 300.0),
        )
        pip_result = _run_captured(
            runner,
            pip_check_command,
            root=root,
            environment=environment,
            timeout_seconds=min(timeout_seconds, 300.0),
        )
    blockers: list[str] = []
    if smoke_result["returncode"] != 0:
        blockers.append(f"runtime_smoke_exit_{smoke_result['returncode']}")
    if pytest_result.get("not_run_reason"):
        blockers.append(f"pytest_not_run:{pytest_result['not_run_reason']}")
    elif pytest_result["returncode"] != 0:
        blockers.append(f"pytest_exit_{pytest_result['returncode']}")
    junit = pytest_result["junit"]
    if (
        int(junit.get("tests", 0)) <= 0
        or int(junit.get("failures", 0)) != 0
        or int(junit.get("errors", 0)) != 0
    ):
        blockers.append("pytest_junit_not_clean")
    if ruff_result["returncode"] != 0:
        blockers.append(f"phase00_ruff_exit_{ruff_result['returncode']}")
    if cli_fatal_result["returncode"] != 0:
        blockers.append(f"cli_fatal_ruff_exit_{cli_fatal_result['returncode']}")
    if pip_result["returncode"] != 0:
        blockers.append(f"pip_check_exit_{pip_result['returncode']}")
    return {
        "runtime_smoke": smoke_result,
        "pytest": pytest_result,
        "ruff": ruff_result,
        "cli_fatal": cli_fatal_result,
        "pip_check": pip_result,
        "containment": containment,
        "blockers": blockers,
    }


def _not_run_result(command: Sequence[str], *, reason: str) -> dict[str, Any]:
    payload = f"NOT_RUN:{reason}\n".encode("ascii")
    return {
        "command": shlex.join(command),
        "returncode": 125,
        "duration_seconds": 0.0,
        "stdout": b"",
        "stderr": payload,
        "stdout_sha256": sha256(b"").hexdigest(),
        "stderr_sha256": sha256(payload).hexdigest(),
        "not_run_reason": reason,
    }


def _run_captured(
    runner: Callable[..., subprocess.CompletedProcess[Any]],
    command: Sequence[str],
    *,
    root: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        completed = runner(
            list(command),
            cwd=root,
            env=dict(environment),
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = int(completed.returncode)
        stdout = _bounded_bytes(_as_bytes(completed.stdout))
        stderr = _bounded_bytes(_as_bytes(completed.stderr))
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = _bounded_bytes(_as_bytes(exc.stdout))
        stderr = b"TimeoutExpired\n"
    except OSError as exc:
        returncode = 127
        stdout = b""
        stderr = (_exception_code(exc) + "\n").encode("ascii", errors="replace")
    return {
        "command": shlex.join(command),
        "returncode": returncode,
        "duration_seconds": round(time.monotonic() - started, 6),
        "stdout": stdout,
        "stderr": stderr,
        "stdout_sha256": sha256(stdout).hexdigest(),
        "stderr_sha256": sha256(stderr).hexdigest(),
    }


def _source_receipt(
    *,
    closure_id: str,
    preflight: Mapping[str, Any],
    postflight: Mapping[str, Any],
    passed: bool,
) -> dict[str, Any]:
    manifest = postflight["manifest"]
    source_rows = manifest.get("source_evidence_index", [])
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "status": "PASS_SOURCE_STABLE" if passed else "BLOCKED_SOURCE_VERIFICATION",
        "closure_id": closure_id,
        "pre_checkpoint_id": preflight["checkpoint"]["checkpoint_id"],
        "post_checkpoint_id": postflight["checkpoint"]["checkpoint_id"],
        "checkpoint_ids": preflight["checkpoint_ids"],
        "pre_freshness_manifest_sha256": preflight["checkpoint"][
            "freshness_manifest_sha256"
        ],
        "post_freshness_manifest_sha256": postflight["checkpoint"][
            "freshness_manifest_sha256"
        ],
        "pre_stable_manifest_file_sha256": preflight["checkpoint"][
            "stable_manifest_file_sha256"
        ],
        "post_stable_manifest_file_sha256": postflight["checkpoint"][
            "stable_manifest_file_sha256"
        ],
        "source_evidence_index_sha256": sha256(
            _canonical_json(source_rows).encode("utf-8")
        ).hexdigest(),
        "dirty_path_count": int(manifest.get("dirty_path_count", -1)),
        "publication_surface_count": len(manifest.get("publication_surface_index", [])),
        "financial_effect_surface_count": len(
            manifest.get("financial_effect_surface_index", [])
        ),
        "blockers": list(manifest.get("blockers", [])),
        "descendant_control_id": postflight["descendant"]["control_id"],
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _runtime_receipt(
    *,
    closure_id: str,
    preflight: Mapping[str, Any],
    passed: bool,
    containment: Mapping[str, Any],
) -> dict[str, Any]:
    observation = preflight["checkpoint"]["runtime_observation"]
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "status": "PASS_RUNTIME_QUIESCED" if passed else "BLOCKED_RUNTIME_VERIFICATION",
        "closure_id": closure_id,
        "maintenance_id": preflight["maintenance_id"],
        "maintenance_blocking": True,
        "runtime_observation": observation,
        "producer_count": int(observation["processes"]["producer_count"]),
        "loaded_launchd_services": [
            row["service"] for row in observation["launchd"] if row["loaded"]
        ],
        "reload_performed": False,
        "external_calls_performed": 0,
        "credentials_accessed": False,
        "provider_credits_used": 0,
        "orders_submitted": 0,
        "external_effect_count_basis": (
            "KERNEL_DENY_SANDBOX_PLUS_TEST_CONTAINMENT"
        ),
        "external_effect_containment": dict(containment),
        "runtime_activation_authorized": False,
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _test_receipt(
    *,
    closure_id: str,
    verification: Mapping[str, Any],
    passed: bool,
) -> dict[str, Any]:
    pytest_result = verification["pytest"]
    return {
        "schema_version": TEST_SCHEMA_VERSION,
        "status": (
            "PASS_EXACT_TREE_LOCAL_VERIFICATION"
            if passed
            else "BLOCKED_EXACT_TREE_LOCAL_VERIFICATION"
        ),
        "closure_id": closure_id,
        "runtime_smoke": _command_receipt(verification["runtime_smoke"]),
        "pytest": _command_receipt(pytest_result, include_junit=True),
        "phase00_ruff": _command_receipt(verification["ruff"]),
        "cli_fatal_ruff": _command_receipt(verification["cli_fatal"]),
        "pip_check": _command_receipt(verification["pip_check"]),
        "warnings_promoted_to_errors": True,
        "external_network_tests_excluded": True,
        "external_calls_performed": 0,
        "credentials_accessed": False,
        "provider_credits_used": 0,
        "orders_submitted": 0,
        "external_effect_count_basis": (
            "KERNEL_DENY_SANDBOX_PLUS_TEST_CONTAINMENT"
        ),
        "external_effect_containment": dict(verification["containment"]),
        "blockers": list(verification["blockers"]),
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _git_receipt(
    *,
    closure_id: str,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = preflight["manifest"]
    dirty = bool(manifest.get("git_dirty"))
    return {
        "schema_version": GIT_SCHEMA_VERSION,
        "status": "BOUND_TO_HEAD_AND_DIRTY_MANIFEST",
        "closure_id": closure_id,
        "git_branch": str(manifest.get("git_branch", "")),
        "git_head": str(manifest.get("git_head", "")),
        "git_dirty": dirty,
        "dirty_path_count": int(manifest.get("dirty_path_count", -1)),
        "exact_tree_manifest_sha256": preflight["checkpoint"][
            "stable_manifest_file_sha256"
        ],
        "semantic_freshness_sha256": preflight["checkpoint"][
            "freshness_manifest_sha256"
        ],
        "git_reproducibility_status": (
            "BOUND_DIRTY_TREE_NOT_REPRODUCIBLE_FROM_GIT_CLONE"
            if dirty
            else "REPRODUCIBLE_FROM_GIT_HEAD"
        ),
        "git_checkpoint_complete": not dirty,
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _acceptance_rows(
    root: Path,
    *,
    closure_id: str,
    evidence: Mapping[str, Any],
    checkpoint_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows = _read_csv_rows(root / HISTORICAL_ACCEPTANCE)
    if len(rows) != 22 or {row.get("control_id") for row in rows} != {
        f"P00-SL-{index:03d}" for index in range(1, 23)
    }:
        raise ValueError("phase00_acceptance_source_contract_invalid")
    result: list[dict[str, Any]] = []
    evidence_by_id = {
        str(item["requirement_id"]): item
        for item in evidence["acceptance_controls"]
    }
    for row in rows:
        control_id = str(row["control_id"])
        control_evidence = evidence_by_id[control_id]
        if control_id in RELOAD_CONTROLS:
            status = "BLOCKED_SEPARATE_OPERATOR_APPROVAL"
            implementation_gate_passed = True
            runtime_activation_gate_passed = False
            reason = "Runtime reload and first-run evidence are explicitly outside this closure."
            next_action = "Require a new explicit operator approval and controlled runtime protocol."
        elif control_evidence["status"] == "PASS_PROVEN":
            status = "PASS"
            implementation_gate_passed = True
            runtime_activation_gate_passed = False
            reason = "Every explicitly mapped JUnit test passed without skip, failure, or error."
            next_action = "Preserve the exact control evidence and source bindings."
        else:
            status = "NOT_PROVEN"
            implementation_gate_passed = False
            runtime_activation_gate_passed = False
            reason = str(control_evidence["reason"])
            next_action = "Add or repair exact mapped tests; rerun the immutable closure."
        result.append(
            {
                **row,
                "status": status,
                "implementation_gate_passed": implementation_gate_passed,
                "runtime_activation_gate_passed": runtime_activation_gate_passed,
                "closure_id": closure_id,
                "evidence_path": (
                    f"data/research/phase00_control/closures/{closure_id}/"
                    "control_evidence.json"
                ),
                "evidence_test_nodes": ";".join(control_evidence["matched_test_nodes"]),
                "checkpoint_ids": ";".join(checkpoint_ids),
                "reason": reason,
                "next_action": next_action,
            }
        )
    return result


def _fault_catalog_rows(
    root: Path,
    *,
    closure_id: str,
    evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = _read_csv_rows(root / HISTORICAL_FAULT_CATALOG)
    if len(rows) != 36 or {row.get("test_id") for row in rows} != {
        f"T00-{index:03d}" for index in range(1, 37)
    }:
        raise ValueError("phase00_fault_catalog_source_contract_invalid")
    result: list[dict[str, Any]] = []
    evidence_by_id = {
        str(item["requirement_id"]): item for item in evidence["fault_contracts"]
    }
    for row in rows:
        test_id = str(row["test_id"])
        fault_evidence = evidence_by_id[test_id]
        if test_id in DEFERRED_OPERATIONAL_FAULTS:
            status = "DEFERRED_OPERATIONAL_NOT_EXECUTED"
            reason = (
                "Requires disposable hardware or elapsed-time runtime operation and cannot "
                "be truthfully simulated as implementation acceptance."
            )
            runtime_activation_blocker = True
        elif fault_evidence["status"] == "PASS_PROVEN":
            status = "PASS_IMPLEMENTED_AND_TESTED"
            reason = "Every explicitly mapped JUnit test passed without skip, failure, or error."
            runtime_activation_blocker = False
        else:
            status = "NOT_PROVEN"
            reason = str(fault_evidence["reason"])
            runtime_activation_blocker = True
        result.append(
            {
                **row,
                "closure_status": status,
                "closure_id": closure_id,
                "evidence_path": (
                    f"data/research/phase00_control/closures/{closure_id}/"
                    "control_evidence.json"
                ),
                "evidence_test_nodes": ";".join(fault_evidence["matched_test_nodes"]),
                "runtime_activation_blocker": runtime_activation_blocker,
                "reason": reason,
                "promotion_authority": False,
                "order_authority": False,
            }
        )
    return result


def _build_control_evidence(
    root: Path,
    *,
    closure_id: str,
    junit_xml: bytes,
) -> dict[str, Any]:
    mapping = _load_control_evidence_map(root / CONTROL_EVIDENCE_MAP)
    cases = _junit_test_cases(junit_xml)
    acceptance = _evaluate_evidence_group(
        mapping["acceptance_controls"],
        cases=cases,
        expected_ids={f"P00-SL-{index:03d}" for index in range(1, 23)},
        operational_ids=RELOAD_CONTROLS,
    )
    faults = _evaluate_evidence_group(
        mapping["fault_contracts"],
        cases=cases,
        expected_ids={f"T00-{index:03d}" for index in range(1, 37)},
        operational_ids=DEFERRED_OPERATIONAL_FAULTS,
    )
    return {
        "schema_version": CONTROL_EVIDENCE_SCHEMA_VERSION,
        "closure_id": closure_id,
        "mapping_path": CONTROL_EVIDENCE_MAP.as_posix(),
        "mapping_sha256": _file_sha256(root / CONTROL_EVIDENCE_MAP),
        "junit_sha256": sha256(junit_xml).hexdigest(),
        "junit_case_count": len(cases),
        "implementation_controls_proven": all(
            item["status"] == "PASS_PROVEN"
            for item in acceptance
            if item["requirement_id"] not in RELOAD_CONTROLS
        ),
        "implementation_faults_proven": all(
            item["status"] == "PASS_PROVEN"
            for item in faults
            if item["requirement_id"] not in DEFERRED_OPERATIONAL_FAULTS
        ),
        "acceptance_controls": acceptance,
        "fault_contracts": faults,
        "authority_flags": dict(ZERO_AUTHORITY),
    }


def _load_control_evidence_map(path: Path) -> dict[str, dict[str, list[str]]]:
    payload = _load_strict_object(path)
    if payload.get("schema_version") != "thewiz.phase00.control_evidence_map.v1":
        raise ValueError("phase00_control_evidence_map_schema_invalid")
    result: dict[str, dict[str, list[str]]] = {}
    for group_name in ("acceptance_controls", "fault_contracts"):
        group = payload.get(group_name)
        if not isinstance(group, dict):
            raise TypeError(f"phase00_control_evidence_map_{group_name}_invalid")
        normalized: dict[str, list[str]] = {}
        for requirement_id, raw_nodes in group.items():
            if not isinstance(requirement_id, str) or not isinstance(raw_nodes, list):
                raise TypeError("phase00_control_evidence_map_entry_invalid")
            nodes = [str(node).strip() for node in raw_nodes]
            if any(not node or "::" not in node for node in nodes):
                raise ValueError("phase00_control_evidence_map_node_invalid")
            if len(nodes) != len(set(nodes)):
                raise ValueError("phase00_control_evidence_map_duplicate_node")
            normalized[requirement_id] = nodes
        result[group_name] = normalized
    return result


def _evaluate_evidence_group(
    mapping: Mapping[str, Sequence[str]],
    *,
    cases: Mapping[str, str],
    expected_ids: set[str],
    operational_ids: frozenset[str],
) -> list[dict[str, Any]]:
    if set(mapping) != expected_ids:
        raise ValueError("phase00_control_evidence_map_id_set_invalid")
    rows: list[dict[str, Any]] = []
    for requirement_id in sorted(expected_ids):
        selectors = list(mapping[requirement_id])
        matched: list[str] = []
        for selector in selectors:
            matches = sorted(
                node
                for node in cases
                if node == selector or node.startswith(f"{selector}[")
            )
            matched.extend(matches)
        matched = list(dict.fromkeys(matched))
        outcomes = [cases[node] for node in matched]
        missing_selectors = [
            selector
            for selector in selectors
            if not any(
                node == selector or node.startswith(f"{selector}[") for node in cases
            )
        ]
        if requirement_id in operational_ids:
            status = "DEFERRED_OPERATIONAL"
            reason = "Physical or elapsed-time operation is outside implementation closure."
        elif not selectors or missing_selectors:
            status = "NOT_PROVEN"
            reason = "One or more required test selectors were absent from JUnit evidence."
        elif any(outcome != "passed" for outcome in outcomes):
            status = "FAILED_EVIDENCE"
            reason = "At least one required JUnit case failed, errored, or was skipped."
        else:
            status = "PASS_PROVEN"
            reason = "All explicitly mapped JUnit cases passed."
        rows.append(
            {
                "requirement_id": requirement_id,
                "status": status,
                "required_test_selectors": selectors,
                "matched_test_nodes": matched,
                "missing_test_selectors": missing_selectors,
                "matched_outcomes": outcomes,
                "reason": reason,
            }
        )
    return rows


def _junit_test_cases(payload: bytes) -> dict[str, str]:
    if not payload:
        return {}
    root = ET.fromstring(payload)
    cases: dict[str, str] = {}
    for case in root.iter("testcase"):
        class_name = str(case.attrib.get("classname", "")).strip()
        name = str(case.attrib.get("name", "")).strip()
        if not class_name or not name:
            raise ValueError("phase00_junit_case_identity_invalid")
        node = f"{class_name}::{name}"
        if node in cases:
            raise ValueError("phase00_junit_duplicate_case_identity")
        if case.find("failure") is not None:
            outcome = "failed"
        elif case.find("error") is not None:
            outcome = "error"
        elif case.find("skipped") is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        cases[node] = outcome
    return cases


def _matching_checkpoint_ids(root: Path, *, freshness_sha: str) -> list[str]:
    matches: list[tuple[str, str]] = []
    checkpoint_root = root / "data/research/phase00_control/checkpoints"
    for path in sorted(checkpoint_root.glob("phase00checkpoint_*.json")):
        try:
            payload = _load_strict_object(path)
        except (OSError, TypeError, ValueError):
            continue
        checkpoint_id = str(payload.get("checkpoint_id", ""))
        manifest_path = root / str(payload.get("stable_manifest_path", ""))
        if (
            checkpoint_id != path.stem
            or payload.get("status") != "PASS_QUIESCED_BASELINE"
            or payload.get("checkpoint_capture_complete") is not True
            or payload.get("blockers") != []
            or payload.get("freshness_manifest_sha256") != freshness_sha
            or payload.get("descendant_control_status")
            != "PASS_STALE_DESCENDANTS_INVALIDATED"
            or payload.get("descendant_regeneration_status")
            != "INVALIDATED_PENDING_CONTROLLED_REBUILD"
            or not _single_regular_file(manifest_path)
            or _file_sha256(manifest_path)
            != payload.get("stable_manifest_file_sha256")
        ):
            continue
        matches.append((str(payload.get("observed_at_utc", "")), checkpoint_id))
    return [checkpoint_id for _, checkpoint_id in sorted(matches)]


def _require_runtime_quiesced(checkpoint: Mapping[str, Any]) -> None:
    observation = checkpoint.get("runtime_observation")
    if not isinstance(observation, dict):
        raise TypeError("phase00_runtime_observation_missing")
    processes = observation.get("processes")
    launchd = observation.get("launchd")
    if (
        not isinstance(processes, dict)
        or processes.get("status") != "PASS_NO_PRODUCERS"
        or int(processes.get("producer_count", -1)) != 0
    ):
        raise RuntimeError("phase00_runtime_producers_not_quiesced")
    if not isinstance(launchd, list) or {
        str(row.get("service", "")) for row in launchd if isinstance(row, dict)
    } != EXPECTED_LAUNCHD_SERVICES:
        raise RuntimeError("phase00_runtime_launchd_set_invalid")
    if any(
        not isinstance(row, dict)
        or row.get("status") != "PASS_UNLOADED"
        or row.get("loaded") is not False
        for row in launchd
    ):
        raise RuntimeError("phase00_runtime_launchd_not_unloaded")


def _phase00_ruff_scope(root: Path) -> list[str]:
    patterns = (
        "src/quant_platform/orchestration/corrective_phase00*.py",
        "src/quant_platform/orchestration/corrective_scheduler*.py",
        "src/quant_platform/orchestration/corrective_publication_registry.py",
        "src/quant_platform/orchestration/corrective_active_artifact_envelopes.py",
        "src/quant_platform/orchestration/corrective_lineage.py",
        "src/quant_platform/orchestration/corrective_effect_guard.py",
        "src/quant_platform/orchestration/corrective_external_effect*.py",
        "src/quant_platform/orchestration/corrective_financial_effect_registry.py",
        "src/quant_platform/orchestration/corrective_order_authority.py",
        "src/quant_platform/orchestration/corrective_redaction.py",
        "src/quant_platform/orchestration/effect_authority.py",
        "src/quant_platform/orchestration/identity_ontology.py",
        "src/quant_platform/orchestration/phase00_ccxt_evidence.py",
        "src/quant_platform/orchestration/venue_policy_registry.py",
        "tests/test_corrective_phase00*.py",
        "tests/test_corrective_scheduler*.py",
        "tests/test_corrective_publication_registry.py",
        "tests/test_corrective_lineage.py",
        "tests/test_corrective_financial_effect_registry.py",
        "tests/test_phase00*.py",
    )
    paths: set[str] = set()
    for pattern in patterns:
        paths.update(
            path.relative_to(root).as_posix()
            for path in root.glob(pattern)
            if path.is_file()
        )
    if not paths:
        raise RuntimeError("phase00_ruff_scope_empty")
    return sorted(paths)


def _sanitized_environment(root: Path) -> dict[str, str]:
    allowed = {
        "HOME",
        "LANG",
        "LC_ALL",
        "PATH",
        "SHELL",
        "TMPDIR",
        "TZ",
        "USER",
    }
    environment = {
        key: value for key, value in os.environ.items() if key in allowed and value
    }
    environment.update(
        {
            "PYTHONPATH": str(root / "src"),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTHONNOUSERSITE": "1",
            "QPA_PHASE00_EXTERNAL_EFFECT_MODE": "NO_EXTERNAL_NO_ORDER",
        }
    )
    return environment


def _dependency_check_command(
    *,
    python: Path,
    environment: Mapping[str, str],
) -> list[str]:
    uv = shutil.which("uv", path=environment.get("PATH"))
    if uv:
        return [uv, "pip", "check", "--python", str(python)]
    return [str(python), "-m", "pip", "check"]


def _contained_pytest_command(
    command: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    sandbox = Path("/usr/bin/sandbox-exec")
    if os.uname().sysname != "Darwin" or not sandbox.is_file():
        raise RuntimeError("phase00_kernel_effect_containment_unavailable")
    profile_sha256 = sha256(PYTEST_SANDBOX_PROFILE.encode("utf-8")).hexdigest()
    return (
        [str(sandbox), "-p", PYTEST_SANDBOX_PROFILE, *command],
        {
            "status": "ENFORCED",
            "mechanism": "macos_sandbox_exec",
            "profile_sha256": profile_sha256,
            "network_denied": True,
            "keychain_security_exec_denied": True,
            "test_fixture_containment_required": True,
        },
    )


def _parse_junit(payload: bytes) -> dict[str, Any]:
    if not payload:
        return {"tests": 0, "failures": 0, "errors": 1, "skipped": 0, "time": 0.0}
    root = ET.fromstring(payload)
    if root.tag == "testsuite" or (
        root.tag == "testsuites" and root.attrib.get("tests") is not None
    ):
        nodes = [root]
    else:
        nodes = list(root.findall(".//testsuite"))
    return {
        "tests": sum(int(node.attrib.get("tests", 0)) for node in nodes),
        "failures": sum(int(node.attrib.get("failures", 0)) for node in nodes),
        "errors": sum(int(node.attrib.get("errors", 0)) for node in nodes),
        "skipped": sum(int(node.attrib.get("skipped", 0)) for node in nodes),
        "time": round(sum(float(node.attrib.get("time", 0.0)) for node in nodes), 6),
    }


def _command_receipt(
    result: Mapping[str, Any],
    *,
    include_junit: bool = False,
) -> dict[str, Any]:
    receipt = {
        "command": result["command"],
        "returncode": int(result["returncode"]),
        "duration_seconds": float(result["duration_seconds"]),
        "stdout_sha256": str(result["stdout_sha256"]),
        "stderr_sha256": str(result["stderr_sha256"]),
    }
    if result.get("not_run_reason"):
        receipt["not_run_reason"] = str(result["not_run_reason"])
    if include_junit:
        receipt["junit"] = dict(result["junit"])
        receipt["junit_sha256"] = sha256(result["junit_xml"]).hexdigest()
    return receipt


def _bundle_paths(root: Path, output_root: Path) -> dict[str, Path]:
    return {
        "bundle_root": output_root,
        "active_closure": root / ACTIVE_POINTER,
        "active_acceptance": root / ACTIVE_ACCEPTANCE,
        "active_fault_catalog": root / ACTIVE_FAULT_CATALOG,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not _single_regular_file(path) or path.stat().st_size > MAX_CAPTURE_BYTES:
        raise ValueError("closure_csv_source_invalid")
    text = path.read_text(encoding="utf-8")
    if "\x00" in text:
        raise ValueError("closure_csv_source_contains_nul")
    reader = csv.DictReader(StringIO(text, newline=""))
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("closure_csv_source_header_invalid")
    return [dict(row) for row in reader]


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("closure_csv_rows_empty")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("closure_csv_row_schema_mismatch")
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: ("true" if value is True else "false" if value is False else value)
                for key, value in row.items()
            }
        )
    return buffer.getvalue().encode("utf-8")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return _pretty_json(payload).encode("utf-8")


def _bounded_bytes(payload: bytes) -> bytes:
    if len(payload) > MAX_CAPTURE_BYTES:
        raise ValueError("phase00_verification_output_too_large")
    return payload


def _as_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace")
    raise TypeError("subprocess_output_type_invalid")


def _exact_bound_path(
    root: Path,
    payload: Mapping[str, Any],
    key: str,
    expected: Path,
) -> Path:
    relative = Path(_required_string(payload, key))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{key}_unsafe")
    path = root / relative
    if path != expected or not _single_regular_file(path):
        raise ValueError(f"{key}_substitution")
    return path


def _require_hash(path: Path, payload: Mapping[str, Any], key: str) -> None:
    expected = _required_string(payload, key)
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError(f"{key}_invalid")
    if _file_sha256(path) != expected:
        raise ValueError(f"{key}_mismatch")


def _require_read_only(path: Path, label: str) -> None:
    if path.stat().st_mode & 0o222:
        raise ValueError(f"{label}_writable")


def _require_schema(payload: Mapping[str, Any], expected: str, label: str) -> None:
    if payload.get("schema_version") != expected:
        raise ValueError(f"{label}_schema_invalid")


def _require_zero_authority(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("authority_flags") != ZERO_AUTHORITY:
        raise ValueError(f"{label}_authority_nonzero")


def _required_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key}_missing")
    return value


def _single_regular_file(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1


def _canonical_root(root: Path) -> Path:
    resolved = root.resolve()
    if not (resolved / ".git").exists():
        raise ValueError("phase00 closure root is not a Git repository")
    return resolved


def _as_utc(value: datetime | None) -> datetime:
    observed = value or datetime.now(UTC)
    if observed.tzinfo is None:
        raise ValueError("phase00 closure timestamp must be timezone aware")
    return observed.astimezone(UTC)


def _exception_code(exc: BaseException) -> str:
    primitive = next(
        (
            value
            for value in exc.args
            if isinstance(value, (str, int, float, bool)) and value is not None
        ),
        "",
    )
    code = type(exc).__name__
    if primitive == "":
        return code
    safe = "".join(
        character if character.isalnum() or character in "._:-" else "_"
        for character in str(primitive)[:160]
    )
    return f"{code}:{safe}"
