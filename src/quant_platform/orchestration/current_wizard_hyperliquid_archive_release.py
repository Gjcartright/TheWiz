"""Non-destructive release planning for verified off-volume snapshot copies."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import json
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.current_wizard_hyperliquid_storage import (
    MINIMUM_FREE_BYTES,
    SNAPSHOT_ROOT_NAMES,
    STORAGE_BOOKKEEPING_MANIFEST_NAMES,
    _device_id,
    _file_hash,
    _tree_inventory,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "wizard_hyperliquid_archive_release_dry_run.v1"
PLAN_COLUMNS = (
    "schema_version",
    "release_id",
    "copy_id",
    "snapshot_path",
    "source_path",
    "destination_path",
    "size_bytes",
    "source_tree_sha256",
    "destination_tree_sha256",
    "tree_hash_match",
    "current_lineage_overlap",
    "release_candidate",
    "release_status",
    "blocker",
    "source_release_authorized",
    "source_move_performed",
    "source_delete_performed",
    "order_submission_performed",
    "live_trading_authorized",
)


def build_current_wizard_hyperliquid_archive_release_dry_run(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    available_disk_bytes: int | None = None,
) -> CommandResult:
    """Re-verify copied snapshots and publish an exact, no-delete release plan."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    copy_manifest_path = active / "current_wizard_hyperliquid_archive_copy_manifest.json"
    paths = _paths(active)
    if not copy_manifest_path.is_file():
        return _blocked_result(
            root=root,
            paths=paths,
            as_of=as_of,
            blocker="verified_off_volume_archive_copy_missing",
            evidence_path=copy_manifest_path,
            available_disk_bytes=available_disk_bytes,
        )

    copy_manifest = _read_json(copy_manifest_path)
    copy_id = str(copy_manifest.get("copy_id", "")).strip()
    copy_set_id = str(copy_manifest.get("copy_set_id", "")).strip()
    reclamation_id = str(copy_manifest.get("reclamation_id", "")).strip()
    release_id = "cwarchiverelease_" + sha256(
        f"{copy_id}:{as_of.isoformat()}".encode("utf-8")
    ).hexdigest()[:20]
    global_blockers = _copy_receipt_blockers(
        root=root,
        active=active,
        copy_manifest=copy_manifest,
        copy_manifest_path=copy_manifest_path,
    )
    protected_paths = _current_protected_paths(root, active)
    candidate_receipts = copy_manifest.get("candidate_receipts", [])
    if not isinstance(candidate_receipts, list):
        candidate_receipts = []
        global_blockers.append("copy_manifest_candidate_receipts_invalid")

    rows = [
        _candidate_row(
            root=root,
            release_id=release_id,
            copy_id=copy_id,
            receipt=receipt,
            protected_paths=protected_paths,
            global_blockers=global_blockers,
        )
        for receipt in candidate_receipts
        if isinstance(receipt, dict)
    ]
    if len(rows) != len(candidate_receipts):
        global_blockers.append("copy_manifest_candidate_receipt_type_invalid")
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    if plan.empty:
        plan = pd.DataFrame(
            [
                _blocked_plan_row(
                    release_id=release_id,
                    copy_id=copy_id,
                    blocker=";".join(global_blockers)
                    or "copy_manifest_has_no_candidate_receipts",
                )
            ],
            columns=PLAN_COLUMNS,
        )

    current_free_bytes = int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )
    release_rows = plan.loc[plan["release_candidate"].map(_boolish)]
    recoverable_bytes = int(release_rows["size_bytes"].astype(int).sum())
    projected_free_bytes = current_free_bytes + recoverable_bytes
    projected_floor_met = projected_free_bytes >= MINIMUM_FREE_BYTES
    candidates_accounted = len(rows) == len(candidate_receipts) and bool(rows)
    candidate_roots = [Path(str(row["source_path"])) for row in rows]
    no_candidate_overlap = _candidate_roots_do_not_overlap(candidate_roots)
    release_dry_run_ready = bool(
        not global_blockers
        and candidates_accounted
        and no_candidate_overlap
        and not plan.empty
        and plan["release_candidate"].map(_boolish).all()
        and projected_floor_met
    )
    if not projected_floor_met:
        plan.loc[plan["release_candidate"].map(_boolish), "release_candidate"] = False
        plan.loc[
            plan["release_status"].eq("READY_FOR_SEPARATE_APPROVAL"),
            "release_status",
        ] = "BLOCKED"
        plan.loc[plan["blocker"].astype(str).eq(""), "blocker"] = (
            "projected_free_space_below_required_floor"
        )
    if not no_candidate_overlap:
        plan["release_candidate"] = False
        plan["release_status"] = "BLOCKED"
        plan["blocker"] = plan["blocker"].map(
            lambda value: _join_blockers(str(value), "release_candidate_roots_overlap")
        )
    validation = _validation(
        plan=plan,
        copy_manifest=copy_manifest,
        copy_id=copy_id,
        candidates_accounted=candidates_accounted,
        no_candidate_overlap=no_candidate_overlap,
        projected_floor_met=projected_floor_met,
        global_blockers=global_blockers,
    )
    atomic_write_csv(plan, paths["plan"], index=False)
    atomic_write_csv(validation, paths["validation"], index=False)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "as_of": as_of.isoformat(),
        "copy_id": copy_id,
        "copy_set_id": copy_set_id,
        "reclamation_id": reclamation_id,
        "source_copy_manifest": str(copy_manifest_path.relative_to(root)),
        "source_copy_manifest_sha256": _file_hash(copy_manifest_path),
        "protected_current_paths": len(protected_paths),
        "candidate_branches": len(rows),
        "release_candidates": int(plan["release_candidate"].map(_boolish).sum()),
        "blocked_candidates": int((~plan["release_candidate"].map(_boolish)).sum()),
        "recoverable_bytes": recoverable_bytes,
        "current_free_bytes": current_free_bytes,
        "projected_free_bytes_after_release": projected_free_bytes,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "projected_storage_floor_met": projected_floor_met,
        "copy_receipt_integrity_pass": not global_blockers,
        "candidate_roots_do_not_overlap": no_candidate_overlap,
        "release_dry_run_ready": release_dry_run_ready,
        "release_status": (
            "READY_FOR_SEPARATE_APPROVAL" if release_dry_run_ready else "BLOCKED"
        ),
        "release_blocker": (
            ""
            if release_dry_run_ready
            else _release_blocker(plan, global_blockers, projected_floor_met)
        ),
        "source_release_authorized": False,
        "source_move_performed": False,
        "source_delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: str(path.relative_to(root)) for name, path in paths.items()},
    }
    atomic_write_text(paths["manifest"], json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    atomic_write_text(paths["summary_md"], _summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _copy_receipt_blockers(
    *,
    root: Path,
    active: Path,
    copy_manifest: dict[str, Any],
    copy_manifest_path: Path,
) -> list[str]:
    blockers: list[str] = []
    required_true = ("archive_copy_completed", "archive_release_evidence_complete")
    required_false = (
        "archive_release_preflight_ready",
        "source_release_authorized",
        "source_move_performed",
        "source_delete_performed",
        "order_submission_performed",
        "live_trading_authorized",
    )
    for field in required_true:
        if not _boolish(copy_manifest.get(field, False)):
            blockers.append(f"copy_manifest_{field}_false")
    for field in required_false:
        if _boolish(copy_manifest.get(field, False)):
            blockers.append(f"copy_manifest_{field}_unexpected_true")
    for field in ("copy_id", "copy_set_id", "reclamation_id", "destination_root"):
        if not str(copy_manifest.get(field, "")).strip():
            blockers.append(f"copy_manifest_{field}_missing")

    destination_receipt_text = str(copy_manifest.get("destination_receipt", "")).strip()
    if not destination_receipt_text:
        blockers.append("destination_copy_receipt_path_missing")
    else:
        destination_receipt = Path(destination_receipt_text)
        if not destination_receipt.is_file():
            blockers.append("destination_copy_receipt_missing")
        else:
            destination_payload = _read_json(destination_receipt)
            if _canonical_json(destination_payload) != _canonical_json(copy_manifest):
                blockers.append("active_and_destination_copy_receipts_differ")
    destination_root_text = str(copy_manifest.get("destination_root", "")).strip()
    if destination_root_text:
        destination_root = Path(destination_root_text)
        if not destination_root.is_dir():
            blockers.append("archive_destination_root_missing")
        elif _device_id(destination_root) == _device_id(root):
            blockers.append("archive_destination_is_same_device_as_project")

    receipt_material = {
        key: copy_manifest.get(key)
        for key in (
            "schema_version",
            "copy_id",
            "copy_set_id",
            "reclamation_id",
            "approval_fingerprint",
            "source_reclamation_manifest_sha256",
            "source_reclamation_plan_sha256",
            "destination_root",
            "archive_root",
            "candidate_receipts",
        )
    }
    expected_receipt_hash = sha256(
        _canonical_json(receipt_material).encode("utf-8")
    ).hexdigest()
    if expected_receipt_hash != str(copy_manifest.get("receipt_sha256", "")):
        blockers.append("copy_receipt_material_hash_mismatch")

    plan_path = active / "current_wizard_hyperliquid_storage_reclamation_plan.csv"
    reclamation_path = (
        active / "current_wizard_hyperliquid_storage_reclamation_manifest.json"
    )
    for path, field, missing_blocker, mismatch_blocker in (
        (
            plan_path,
            "source_reclamation_plan_sha256",
            "source_reclamation_plan_missing",
            "source_reclamation_plan_hash_changed",
        ),
        (
            reclamation_path,
            "source_reclamation_manifest_sha256",
            "source_reclamation_manifest_missing",
            "source_reclamation_manifest_hash_changed",
        ),
    ):
        if not path.is_file():
            blockers.append(missing_blocker)
        elif _file_hash(path) != str(copy_manifest.get(field, "")):
            blockers.append(mismatch_blocker)

    receipt_csv = active / "current_wizard_hyperliquid_archive_copy_receipt.csv"
    if not receipt_csv.is_file():
        blockers.append("active_archive_copy_receipt_csv_missing")
    else:
        receipt_frame = pd.read_csv(receipt_csv, keep_default_na=False)
        if set(receipt_frame.get("copy_id", pd.Series(dtype=str)).astype(str)) != {
            str(copy_manifest.get("copy_id", ""))
        }:
            blockers.append("active_archive_copy_receipt_csv_copy_id_mismatch")
        if len(receipt_frame) != len(copy_manifest.get("candidate_receipts", [])):
            blockers.append("active_archive_copy_receipt_csv_count_mismatch")
    if not copy_manifest_path.is_file():
        blockers.append("active_archive_copy_manifest_missing")
    return sorted(set(blockers))


def _candidate_row(
    *,
    root: Path,
    release_id: str,
    copy_id: str,
    receipt: dict[str, Any],
    protected_paths: set[Path],
    global_blockers: list[str],
) -> dict[str, object]:
    blockers = list(global_blockers)
    snapshot_path = str(receipt.get("snapshot_path", "")).strip()
    source = Path(str(receipt.get("source_path", "")))
    destination = Path(str(receipt.get("destination_path", "")))
    expected_tree = str(receipt.get("destination_tree_sha256", ""))
    expected_manifest = str(receipt.get("manifest_sha256", ""))
    expected_size = int(receipt.get("size_bytes", 0) or 0)
    if not _allowed_source(root, source, snapshot_path):
        blockers.append("source_path_outside_allowed_snapshot_roots")
    current_overlap = any(
        _paths_overlap(source.resolve(), path.resolve()) for path in protected_paths
    )
    if current_overlap:
        blockers.append("source_became_current_lineage")

    source_tree = ""
    destination_tree = ""
    source_size = 0
    destination_size = 0
    source_files = 0
    destination_files = 0
    if not source.is_dir():
        blockers.append("source_snapshot_missing")
    else:
        source_size, source_files, source_tree = _tree_inventory(source)
        source_manifest = source / "manifest.json"
        if (
            source_tree != expected_tree
            or source_size != expected_size
            or not source_manifest.is_file()
            or _file_hash(source_manifest) != expected_manifest
        ):
            blockers.append("source_snapshot_hash_changed")
    if not destination.is_dir():
        blockers.append("destination_snapshot_missing")
    else:
        destination_size, destination_files, destination_tree = _tree_inventory(destination)
        destination_manifest = destination / "manifest.json"
        if (
            destination_tree != expected_tree
            or destination_size != expected_size
            or not destination_manifest.is_file()
            or _file_hash(destination_manifest) != expected_manifest
        ):
            blockers.append("destination_snapshot_hash_changed")
    tree_hash_match = bool(
        source_tree
        and source_tree == destination_tree == expected_tree
        and source_size == destination_size == expected_size
        and source_files == destination_files
    )
    if not tree_hash_match:
        blockers.append("source_destination_tree_mismatch")
    blockers = sorted(set(filter(None, blockers)))
    ready = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "copy_id": copy_id,
        "snapshot_path": snapshot_path,
        "source_path": str(source),
        "destination_path": str(destination),
        "size_bytes": expected_size,
        "source_tree_sha256": source_tree,
        "destination_tree_sha256": destination_tree,
        "tree_hash_match": tree_hash_match,
        "current_lineage_overlap": current_overlap,
        "release_candidate": ready,
        "release_status": "READY_FOR_SEPARATE_APPROVAL" if ready else "BLOCKED",
        "blocker": ";".join(blockers),
        "source_release_authorized": False,
        "source_move_performed": False,
        "source_delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def _validation(
    *,
    plan: pd.DataFrame,
    copy_manifest: dict[str, Any],
    copy_id: str,
    candidates_accounted: bool,
    no_candidate_overlap: bool,
    projected_floor_met: bool,
    global_blockers: list[str],
) -> pd.DataFrame:
    checks = (
        ("copy_receipt_integrity", not global_blockers, ";".join(global_blockers)),
        ("copy_id_present", bool(copy_id), copy_id),
        (
            "all_copy_candidates_accounted",
            candidates_accounted,
            f"{len(plan)}/{len(copy_manifest.get('candidate_receipts', []))}",
        ),
        (
            "candidate_roots_do_not_overlap",
            no_candidate_overlap,
            str(len(plan)),
        ),
        (
            "all_source_destination_hashes_match",
            not plan.empty and plan["tree_hash_match"].map(_boolish).all(),
            "tree_hash_match=true",
        ),
        (
            "no_current_lineage_overlap",
            plan.empty or not plan["current_lineage_overlap"].map(_boolish).any(),
            "current_lineage_overlap=false",
        ),
        (
            "projected_storage_floor_met",
            projected_floor_met,
            str(projected_floor_met),
        ),
        (
            "source_release_authority_false",
            plan.empty or not plan["source_release_authorized"].map(_boolish).any(),
            "source_release_authorized=false",
        ),
        (
            "no_source_move_performed",
            plan.empty or not plan["source_move_performed"].map(_boolish).any(),
            "source_move_performed=false",
        ),
        (
            "no_source_delete_performed",
            plan.empty or not plan["source_delete_performed"].map(_boolish).any(),
            "source_delete_performed=false",
        ),
        (
            "live_authority_false",
            plan.empty or not plan["live_trading_authorized"].map(_boolish).any(),
            "live_trading_authorized=false",
        ),
    )
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "BLOCKED",
                "evidence": evidence,
                "source_release_authorized": False,
                "source_delete_performed": False,
                "live_trading_authorized": False,
            }
            for check, passed, evidence in checks
        ]
    )


def _blocked_result(
    *,
    root: Path,
    paths: dict[str, Path],
    as_of: datetime,
    blocker: str,
    evidence_path: Path,
    available_disk_bytes: int | None,
) -> CommandResult:
    paths["plan"].parent.mkdir(parents=True, exist_ok=True)
    release_id = "cwarchiverelease_" + sha256(
        f"missing:{as_of.isoformat()}".encode("utf-8")
    ).hexdigest()[:20]
    plan = pd.DataFrame(
        [_blocked_plan_row(release_id=release_id, copy_id="", blocker=blocker)],
        columns=PLAN_COLUMNS,
    )
    validation = pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": "verified_archive_copy_receipt_available",
                "status": "BLOCKED",
                "evidence": str(evidence_path),
                "source_release_authorized": False,
                "source_delete_performed": False,
                "live_trading_authorized": False,
            },
            {
                "schema_version": SCHEMA_VERSION,
                "check": "no_source_delete_performed",
                "status": "PASS",
                "evidence": "source_delete_performed=false",
                "source_release_authorized": False,
                "source_delete_performed": False,
                "live_trading_authorized": False,
            },
        ]
    )
    current_free_bytes = int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )
    plan.to_csv(paths["plan"], index=False)
    validation.to_csv(paths["validation"], index=False)
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "as_of": as_of.isoformat(),
        "copy_id": "",
        "candidate_branches": 0,
        "release_candidates": 0,
        "blocked_candidates": 0,
        "recoverable_bytes": 0,
        "current_free_bytes": current_free_bytes,
        "projected_free_bytes_after_release": current_free_bytes,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "projected_storage_floor_met": current_free_bytes >= MINIMUM_FREE_BYTES,
        "copy_receipt_integrity_pass": False,
        "release_dry_run_ready": False,
        "release_status": "BLOCKED",
        "release_blocker": blocker,
        "source_release_authorized": False,
        "source_move_performed": False,
        "source_delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: str(path.relative_to(root)) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["summary_md"].write_text(_summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _blocked_plan_row(*, release_id: str, copy_id: str, blocker: str) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "release_id": release_id,
        "copy_id": copy_id,
        "snapshot_path": "",
        "source_path": "",
        "destination_path": "",
        "size_bytes": 0,
        "source_tree_sha256": "",
        "destination_tree_sha256": "",
        "tree_hash_match": False,
        "current_lineage_overlap": False,
        "release_candidate": False,
        "release_status": "BLOCKED",
        "blocker": blocker,
        "source_release_authorized": False,
        "source_move_performed": False,
        "source_delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
    }


def _current_protected_paths(root: Path, active: Path) -> set[Path]:
    protected: set[Path] = set()
    snapshot_roots = [
        (root / "reports" / "snapshots" / name).resolve()
        for name in SNAPSHOT_ROOT_NAMES
    ]
    for path in sorted(active.glob("*_manifest.json")):
        if path.name in STORAGE_BOOKKEEPING_MANIFEST_NAMES:
            continue
        payload = _read_json(path)
        for value in _walk_values(payload):
            text = str(value).strip()
            candidate = None
            if text.startswith("reports/snapshots/"):
                candidate = (root / text).resolve()
            elif text.startswith("/"):
                absolute = Path(text).resolve()
                if any(_is_relative_to(absolute, base) for base in snapshot_roots):
                    candidate = absolute
            if candidate is not None:
                protected.add(candidate)
    return protected


def _allowed_source(root: Path, source: Path, snapshot_path: str) -> bool:
    if not source.is_absolute() or not snapshot_path.startswith("reports/snapshots/"):
        return False
    expected = (root / snapshot_path).resolve()
    if expected != source.resolve():
        return False
    roots = [
        (root / "reports" / "snapshots" / name).resolve()
        for name in SNAPSHOT_ROOT_NAMES
    ]
    return any(_is_relative_to(source.resolve(), allowed) for allowed in roots)


def _candidate_roots_do_not_overlap(paths: list[Path]) -> bool:
    return all(
        not _paths_overlap(left.resolve(), right.resolve())
        for index, left in enumerate(paths)
        for other_index, right in enumerate(paths)
        if index != other_index
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    return _is_relative_to(left, right) or _is_relative_to(right, left)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _release_blocker(
    plan: pd.DataFrame, global_blockers: list[str], projected_floor_met: bool
) -> str:
    blockers = list(global_blockers)
    for value in plan.get("blocker", pd.Series(dtype=str)).astype(str):
        blockers.extend(part for part in value.split(";") if part)
    if not projected_floor_met:
        blockers.append("projected_free_space_below_required_floor")
    return ";".join(sorted(set(blockers))) or "release_dry_run_not_ready"


def _join_blockers(left: str, right: str) -> str:
    return ";".join(sorted(set(filter(None, [*left.split(";"), right]))))


def _paths(active: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_archive_release"
    return {
        "plan": active / f"{stem}_plan.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Archive Release Dry Run",
            "",
            "This is a dry-run release plan. It does not move or delete source evidence.",
            "",
            f"- release dry run: {summary['release_id']}",
            f"- copy: {summary['copy_id'] or 'missing'}",
            f"- status: {summary['release_status']}",
            f"- blocker: {summary['release_blocker'] or 'none'}",
            f"- candidate branches: {summary['candidate_branches']}",
            f"- release candidates: {summary['release_candidates']}",
            f"- recoverable bytes: {summary['recoverable_bytes']}",
            f"- projected storage floor met: {summary['projected_storage_floor_met']}",
            "- source release authorized: false",
            "- source move performed: false",
            "- source delete performed: false",
            "- order submission performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected_json_object:{path}")
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )
