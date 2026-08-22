"""Copy-only, hash-verified off-volume archival for Wizard research snapshots."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_csv,
    atomic_write_text,
    promote_staged_directory,
    promote_staged_file,
)
from quant_platform.orchestration.current_wizard_hyperliquid_storage import (
    _archive_destination_preflight,
    _file_hash,
    _tree_inventory,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "wizard_hyperliquid_archive_copy.v1"
ALLOWED_SNAPSHOT_ROOTS = (
    Path("reports/snapshots/current_wizard_hyperliquid"),
    Path("reports/snapshots/exhaustive_wizard_hyperliquid"),
)


def stage_current_wizard_hyperliquid_archive_copy(
    *,
    root: Path = ROOT,
    archive_destination: Path | None,
    approval_id: str,
    now: datetime | None = None,
) -> CommandResult:
    """Copy verified superseded branches off-volume without releasing sources."""

    approval_id = str(approval_id).strip()
    if not approval_id:
        raise ValueError("archive_copy_approval_id_required")
    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    plan_path = active / "current_wizard_hyperliquid_storage_reclamation_plan.csv"
    reclamation_path = (
        active / "current_wizard_hyperliquid_storage_reclamation_manifest.json"
    )
    plan = _read_plan(plan_path)
    reclamation = _read_json(reclamation_path)
    reclamation_id = str(reclamation.get("reclamation_id", "")).strip()
    if not reclamation_id:
        raise ValueError("storage_reclamation_manifest_missing_reclamation_id")
    if plan.empty:
        raise ValueError("storage_reclamation_plan_has_no_candidates")
    if set(plan["reclamation_id"].astype(str)) != {reclamation_id}:
        raise ValueError("storage_reclamation_plan_manifest_id_mismatch")

    safe = plan.loc[plan["safe_to_archive_later"].map(_boolish)].copy()
    if safe.empty:
        raise ValueError("storage_reclamation_plan_has_no_safe_candidates")
    if not safe["hash_verified"].map(_boolish).all():
        raise ValueError("storage_reclamation_safe_candidate_hash_missing")
    preliminary_destination = _archive_destination_preflight(
        root=root,
        archive_destination=archive_destination,
        reclaimable_bytes=0,
        require_safe_candidates=False,
    )
    if not bool(preliminary_destination["archive_copy_preflight_ready"]):
        raise ValueError(str(preliminary_destination["archive_destination_blocker"]))

    destination_root = Path(str(preliminary_destination["archive_destination_path"]))
    archive_root = destination_root / reclamation_id
    bytes_requiring_copy = _bytes_requiring_copy(
        archive_root=archive_root,
        safe=safe,
    )
    destination = _archive_destination_preflight(
        root=root,
        archive_destination=archive_destination,
        reclaimable_bytes=bytes_requiring_copy,
        require_safe_candidates=False,
    )
    if not bool(destination["archive_copy_preflight_ready"]):
        raise ValueError(str(destination["archive_destination_blocker"]))
    archive_root.mkdir(parents=True, exist_ok=True)
    approval_fingerprint = sha256(approval_id.encode("utf-8")).hexdigest()[:16]
    copy_set_id = "cwarchivecopyset_" + sha256(
        _canonical_json(
            {
                "schema_version": SCHEMA_VERSION,
                "reclamation_id": reclamation_id,
                "destination": str(destination_root),
                "approval_fingerprint": approval_fingerprint,
                "candidate_tree_hashes": dict(
                    zip(safe["snapshot_path"], safe["tree_sha256"], strict=False)
                ),
            }
        ).encode("utf-8")
    ).hexdigest()[:20]
    copy_id = "cwarchivecopy_" + sha256(
        f"{copy_set_id}:{as_of.isoformat()}".encode("utf-8")
    ).hexdigest()[:20]

    rows: list[dict[str, object]] = []
    for record in safe.to_dict("records"):
        rows.append(
            _copy_candidate(
                root=root,
                archive_root=archive_root,
                copy_id=copy_id,
                copy_set_id=copy_set_id,
                approval_fingerprint=approval_fingerprint,
                record=record,
            )
        )
    receipt = pd.DataFrame(rows)
    validation = _validation(receipt, expected=len(safe))
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("archive_copy_validation_failed:" + ",".join(failed))

    receipt_material = {
        "schema_version": SCHEMA_VERSION,
        "copy_id": copy_id,
        "copy_set_id": copy_set_id,
        "reclamation_id": reclamation_id,
        "approval_fingerprint": approval_fingerprint,
        "source_reclamation_manifest_sha256": _file_hash(reclamation_path),
        "source_reclamation_plan_sha256": _file_hash(plan_path),
        "destination_root": str(destination_root),
        "archive_root": str(archive_root),
        "candidate_receipts": receipt.to_dict("records"),
    }
    receipt_sha256 = sha256(
        _canonical_json(receipt_material).encode("utf-8")
    ).hexdigest()
    paths = _paths(active)
    validation.insert(1, "copy_id", copy_id)
    destination_receipt = archive_root / "_copy_receipts" / f"{copy_id}.json"
    summary: dict[str, object] = {
        **receipt_material,
        **destination,
        "as_of": as_of.isoformat(),
        "receipt_sha256": receipt_sha256,
        "candidate_branches": int(len(receipt)),
        "copied_this_run": int(receipt["copy_status"].eq("COPIED_VERIFIED").sum()),
        "already_verified": int(receipt["copy_status"].eq("ALREADY_VERIFIED").sum()),
        "bytes_copied_this_run": int(receipt["bytes_copied_this_run"].sum()),
        "bytes_verified": int(receipt["size_bytes"].sum()),
        "bytes_requiring_copy_at_preflight": bytes_requiring_copy,
        "archive_copy_completed": True,
        "archive_release_evidence_complete": True,
        "archive_release_preflight_ready": False,
        "source_release_authorized": False,
        "source_move_performed": False,
        "source_delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "destination_receipt": str(destination_receipt),
        "artifacts": {name: str(path.relative_to(root)) for name, path in paths.items()},
    }
    _atomic_json_write(destination_receipt, summary, copy_id)
    atomic_write_csv(receipt, paths["receipt"], index=False)
    atomic_write_csv(validation, paths["validation"], index=False)
    atomic_write_text(paths["manifest"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["summary_md"], _summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _copy_candidate(
    *,
    root: Path,
    archive_root: Path,
    copy_id: str,
    copy_set_id: str,
    approval_fingerprint: str,
    record: dict[str, Any],
) -> dict[str, object]:
    relative = _safe_relative_snapshot_path(str(record["snapshot_path"]))
    source = (root / relative).resolve()
    _require_allowed_source(root, source)
    if not source.is_dir():
        raise FileNotFoundError(f"archive_source_missing:{relative}")
    if _contains_symlink(source):
        raise ValueError(f"archive_source_contains_symlink:{relative}")

    expected_tree = str(record["tree_sha256"])
    expected_manifest = str(record["manifest_sha256"])
    source_size, source_files, source_tree = _tree_inventory(source)
    source_manifest = source / "manifest.json"
    if source_tree != expected_tree:
        raise ValueError(f"archive_source_tree_hash_changed:{relative}")
    if not source_manifest.is_file() or _file_hash(source_manifest) != expected_manifest:
        raise ValueError(f"archive_source_manifest_hash_changed:{relative}")
    if source_size != int(record["size_bytes"]):
        raise ValueError(f"archive_source_size_changed:{relative}")

    target = (archive_root / relative).resolve()
    if not _is_relative_to(target, archive_root.resolve()):
        raise ValueError(f"archive_target_escaped_root:{relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    copy_status = "ALREADY_VERIFIED"
    bytes_copied = 0
    if target.exists():
        if not target.is_dir() or _contains_symlink(target):
            raise FileExistsError(f"archive_target_conflict:{target}")
        target_size, target_files, target_tree = _tree_inventory(target)
        if (
            target_tree != expected_tree
            or target_size != source_size
            or target_files != source_files
            or _file_hash(target / "manifest.json") != expected_manifest
        ):
            raise FileExistsError(f"archive_target_hash_conflict:{target}")
    else:
        partial = target.with_name(
            f".{target.name}.partial-{copy_id}-{approval_fingerprint}"
        )
        if partial.exists():
            raise FileExistsError(f"archive_partial_target_exists:{partial}")
        shutil.copytree(source, partial, copy_function=shutil.copy2, symlinks=True)
        target_size, target_files, target_tree = _tree_inventory(partial)
        if (
            target_tree != expected_tree
            or target_size != source_size
            or target_files != source_files
            or _file_hash(partial / "manifest.json") != expected_manifest
        ):
            raise ValueError(f"archive_destination_hash_mismatch:{relative}")
        promote_staged_directory(partial, target)
        copy_status = "COPIED_VERIFIED"
        bytes_copied = source_size

    final_size, final_files, final_tree = _tree_inventory(target)
    source_size_after, source_files_after, source_tree_after = _tree_inventory(source)
    source_still_present = source.is_dir()
    if (
        source_tree_after != expected_tree
        or source_size_after != source_size
        or source_files_after != source_files
        or _file_hash(source / "manifest.json") != expected_manifest
    ):
        raise ValueError(f"archive_source_changed_during_copy:{relative}")
    return {
        "schema_version": SCHEMA_VERSION,
        "copy_id": copy_id,
        "copy_set_id": copy_set_id,
        "snapshot_path": relative.as_posix(),
        "source_path": str(source),
        "destination_path": str(target),
        "copy_status": copy_status,
        "size_bytes": final_size,
        "file_count": final_files,
        "source_tree_sha256": source_tree,
        "source_tree_sha256_after_copy": source_tree_after,
        "destination_tree_sha256": final_tree,
        "tree_hash_match": final_tree == source_tree == expected_tree,
        "manifest_sha256": expected_manifest,
        "source_still_present": source_still_present,
        "bytes_copied_this_run": bytes_copied,
        "source_move_performed": False,
        "source_delete_performed": False,
        "live_trading_authorized": False,
    }


def _validation(receipt: pd.DataFrame, *, expected: int) -> pd.DataFrame:
    checks = (
        ("all_candidates_accounted", len(receipt) == expected, f"{len(receipt)}/{expected}"),
        (
            "all_destination_tree_hashes_match",
            not receipt.empty and receipt["tree_hash_match"].map(_boolish).all(),
            "tree_hash_match=true",
        ),
        (
            "all_sources_remain_present",
            not receipt.empty and receipt["source_still_present"].map(_boolish).all(),
            "source_still_present=true",
        ),
        (
            "no_source_move_performed",
            receipt.empty or not receipt["source_move_performed"].map(_boolish).any(),
            "source_move_performed=false",
        ),
        (
            "no_source_delete_performed",
            receipt.empty or not receipt["source_delete_performed"].map(_boolish).any(),
            "source_delete_performed=false",
        ),
        (
            "live_authority_false",
            receipt.empty or not receipt["live_trading_authorized"].map(_boolish).any(),
            "live_trading_authorized=false",
        ),
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "source_release_authorized": False,
                "source_delete_performed": False,
                "live_trading_authorized": False,
            }
            for check, passed, evidence in checks
        ]
    )


def _bytes_requiring_copy(*, archive_root: Path, safe: pd.DataFrame) -> int:
    required = 0
    for record in safe.to_dict("records"):
        relative = _safe_relative_snapshot_path(str(record["snapshot_path"]))
        target = (archive_root / relative).resolve()
        if not _is_relative_to(target, archive_root.resolve()):
            raise ValueError(f"archive_target_escaped_root:{relative}")
        if not target.exists():
            required += int(record["size_bytes"])
            continue
        if not target.is_dir() or _contains_symlink(target):
            raise FileExistsError(f"archive_target_conflict:{target}")
        target_size, _target_files, target_tree = _tree_inventory(target)
        target_manifest = target / "manifest.json"
        if (
            target_tree != str(record["tree_sha256"])
            or target_size != int(record["size_bytes"])
            or not target_manifest.is_file()
            or _file_hash(target_manifest) != str(record["manifest_sha256"])
        ):
            raise FileExistsError(f"archive_target_hash_conflict:{target}")
    return int(required)


def _safe_relative_snapshot_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe_archive_snapshot_path:{value}")
    if not any(_is_relative_to(path, allowed) for allowed in ALLOWED_SNAPSHOT_ROOTS):
        raise ValueError(f"archive_snapshot_path_outside_allowed_roots:{value}")
    return path


def _require_allowed_source(root: Path, source: Path) -> None:
    allowed = [(root / path).resolve() for path in ALLOWED_SNAPSHOT_ROOTS]
    if not any(_is_relative_to(source, path) for path in allowed):
        raise ValueError(f"archive_source_outside_allowed_roots:{source}")


def _contains_symlink(path: Path) -> bool:
    if path.is_symlink():
        return True
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        if any((Path(directory) / name).is_symlink() for name in dirnames):
            return True
        if any((Path(directory) / name).is_symlink() for name in filenames):
            return True
    return False


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _read_plan(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    required = {
        "reclamation_id",
        "snapshot_path",
        "size_bytes",
        "manifest_sha256",
        "tree_sha256",
        "hash_verified",
        "safe_to_archive_later",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("storage_reclamation_plan_missing_columns:" + ",".join(sorted(missing)))
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def _atomic_json_write(path: Path, payload: dict[str, object], copy_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"archive_receipt_exists:{path}")
    temporary = path.with_name(f".{path.name}.{copy_id}.tmp")
    if temporary.exists():
        raise FileExistsError(f"archive_receipt_temporary_exists:{temporary}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    promote_staged_file(temporary, path)


def _paths(active: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_archive_copy"
    return {
        "receipt": active / f"{stem}_receipt.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Archive Copy Receipt",
            "",
            "This receipt proves an off-volume copy. It does not authorize source release or deletion.",
            "",
            f"- copy: {summary['copy_id']}",
            f"- copy set: {summary['copy_set_id']}",
            f"- reclamation plan: {summary['reclamation_id']}",
            f"- candidate branches: {summary['candidate_branches']}",
            f"- copied this run: {summary['copied_this_run']}",
            f"- already verified: {summary['already_verified']}",
            f"- bytes verified: {summary['bytes_verified']}",
            f"- archive copy completed: {summary['archive_copy_completed']}",
            "- source release authorized: false",
            "- source move performed: false",
            "- source delete performed: false",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )
