"""Read-only snapshot reclamation planning for the current Wizard pipeline."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "wizard_hyperliquid_storage_reclamation.v2"
MINIMUM_FREE_BYTES = 3 * 1024**3
ARCHIVE_DESTINATION_RESERVE_BYTES = 1 * 1024**3
SNAPSHOT_ID_PATTERN = re.compile(
    r"^(?:"
    r"cwconcentration|cwfailure|cwleverage|cwlearning|cwvalidation|cwcadence|"
    r"ewhl|ewapi|hlmap|hlreplay|hlhistory|hlfunding|hlcanonical|hlcost|"
    r"hlobserved|hlwalk|hlregime|hlrobust|hlconcentration|hlleverage|hlvalidation"
    r")_[A-Za-z0-9]+$"
)
SNAPSHOT_ROOT_NAMES = (
    "current_wizard_hyperliquid",
    "exhaustive_wizard_hyperliquid",
)
STORAGE_BOOKKEEPING_MANIFEST_NAMES = {
    "current_wizard_hyperliquid_storage_reclamation_manifest.json",
    "current_wizard_hyperliquid_archive_copy_manifest.json",
    "current_wizard_hyperliquid_archive_release_manifest.json",
}


def build_current_wizard_hyperliquid_storage_reclamation_plan(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    available_disk_bytes: int | None = None,
    archive_destination: Path | None = None,
) -> CommandResult:
    """Inventory archive candidates without moving, linking, or deleting anything."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    snapshot_roots = [
        root / "reports" / "snapshots" / name for name in SNAPSHOT_ROOT_NAMES
    ]
    snapshot_roots = [path for path in snapshot_roots if path.is_dir()]
    if not snapshot_roots:
        raise FileNotFoundError("No Wizard/Hyperliquid snapshot roots were found")

    manifests: dict[Path, dict[str, Any]] = {}
    for path in sorted(active.glob("*_manifest.json")):
        if path.name in STORAGE_BOOKKEEPING_MANIFEST_NAMES:
            continue
        payload = _read_json(path)
        if _references_snapshot_root(payload):
            manifests[path] = payload
    if not manifests:
        raise FileNotFoundError("No active Wizard/Hyperliquid snapshot manifests were found")
    manifest_paths = sorted(manifests)
    protected_paths = _protected_snapshot_paths(root, manifests.values())
    missing_protected = [str(path) for path in protected_paths if not path.exists()]
    if missing_protected:
        raise FileNotFoundError(
            "Current protected snapshot paths are missing: "
            + ",".join(missing_protected)
        )
    recognized = sorted(
        path
        for snapshot_root in snapshot_roots
        for path in snapshot_root.rglob("*")
        if path.is_dir() and SNAPSHOT_ID_PATTERN.match(path.name)
    )
    candidate_roots: list[Path] = []
    for path in recognized:
        if _contains_any(path, protected_paths):
            continue
        if any(_is_relative_to(path, parent) for parent in candidate_roots):
            continue
        candidate_roots.append(path)

    rows = []
    for path in candidate_roots:
        size_bytes, file_count, tree_sha256 = _tree_inventory(path)
        manifest_present = (path / "manifest.json").is_file()
        manifest_sha256 = (
            _file_hash(path / "manifest.json") if manifest_present else ""
        )
        protected_count = sum(_is_relative_to(item, path) for item in protected_paths)
        hash_verified = bool(tree_sha256 and manifest_sha256)
        safe = bool(manifest_present and protected_count == 0 and hash_verified)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "snapshot_path": _relative(path, root),
                "snapshot_family": path.name.split("_", 1)[0],
                "snapshot_id": path.name,
                "manifest_present": manifest_present,
                "protected_current_paths": protected_count,
                "current_lineage": False,
                "file_count": file_count,
                "size_bytes": size_bytes,
                "manifest_sha256": manifest_sha256,
                "tree_sha256": tree_sha256,
                "hash_verified": hash_verified,
                "safe_to_archive_later": safe,
                "reason": (
                    "superseded_complete_snapshot_branch"
                    if safe
                    else "incomplete_snapshot_requires_manual_review"
                ),
                "recommended_action": (
                    "move_to_external_archive_after_hash_verification"
                    if safe
                    else "review_do_not_move_automatically"
                ),
                "move_performed": False,
                "delete_performed": False,
                "live_trading_authorized": False,
            }
        )
    columns = [
        "schema_version",
        "snapshot_path",
        "snapshot_family",
        "snapshot_id",
        "manifest_present",
        "protected_current_paths",
        "current_lineage",
        "file_count",
        "size_bytes",
        "manifest_sha256",
        "tree_sha256",
        "hash_verified",
        "safe_to_archive_later",
        "reason",
        "recommended_action",
        "move_performed",
        "delete_performed",
        "live_trading_authorized",
    ]
    plan = pd.DataFrame(rows, columns=columns)
    if not plan.empty:
        plan = plan.sort_values(
            ["safe_to_archive_later", "size_bytes"],
            ascending=[False, False],
        ).reset_index(drop=True)
    safe_rows = plan.loc[
        plan.get("safe_to_archive_later", pd.Series(dtype=bool)).astype(bool)
    ]
    reclaimable = int(safe_rows.get("size_bytes", pd.Series(dtype=int)).sum())
    destination = _archive_destination_preflight(
        root=root,
        archive_destination=archive_destination,
        reclaimable_bytes=reclaimable,
    )
    validation = _validation(
        plan=plan,
        protected_paths=protected_paths,
        root=root,
        destination=destination,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Storage reclamation validation failed: " + ",".join(failed))

    material = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "input_hashes": {
            _relative(path, root): _file_hash(path) for path in manifest_paths
        },
        "protected_snapshot_paths": sorted(
            _relative(path, root) for path in protected_paths
        ),
        "candidate_tree_hashes": dict(
            zip(plan["snapshot_path"], plan["tree_sha256"], strict=False)
        ),
    }
    reclamation_id = "cwreclaim_" + sha256(
        _canonical_json(material).encode("utf-8")
    ).hexdigest()[:20]
    plan.insert(1, "reclamation_id", reclamation_id)
    validation.insert(1, "reclamation_id", reclamation_id)
    free_bytes = int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )
    paths = _paths(active)
    plan.to_csv(paths["plan"], index=False)
    validation.to_csv(paths["validation"], index=False)
    summary: dict[str, object] = {
        **material,
        "reclamation_id": reclamation_id,
        "candidate_branches": int(len(plan)),
        "safe_archive_candidate_branches": int(len(safe_rows)),
        "hash_verified_candidate_branches": int(
            plan.get("hash_verified", pd.Series(dtype=bool)).astype(bool).sum()
        ),
        "manual_review_branches": int(len(plan) - len(safe_rows)),
        "protected_snapshot_paths": int(len(protected_paths)),
        "current_free_bytes": free_bytes,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "reclaimable_bytes_if_archived_off_volume": reclaimable,
        "projected_free_bytes_after_safe_archive": free_bytes + reclaimable,
        "storage_floor_met_after_safe_archive": bool(
            free_bytes + reclaimable >= MINIMUM_FREE_BYTES
        ),
        **destination,
        "archive_apply_authority": False,
        "move_performed": False,
        "delete_performed": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
    }
    paths["manifest"].write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths["summary_md"].write_text(_summary_markdown(summary), encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _protected_snapshot_paths(
    root: Path, manifests: Any
) -> set[Path]:
    protected: set[Path] = set()
    for manifest in manifests:
        for value in _walk_values(manifest):
            text = str(value).strip()
            if text.startswith("reports/snapshots/"):
                protected.add((root / text).resolve())
    return protected


def _references_snapshot_root(manifest: dict[str, Any]) -> bool:
    prefixes = tuple(f"reports/snapshots/{name}/" for name in SNAPSHOT_ROOT_NAMES)
    return any(str(value).strip().startswith(prefixes) for value in _walk_values(manifest))


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value


def _contains_any(path: Path, protected_paths: set[Path]) -> bool:
    resolved = path.resolve()
    return any(_is_relative_to(item, resolved) for item in protected_paths)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _tree_inventory(path: Path) -> tuple[int, int, str]:
    size = 0
    count = 0
    digest = sha256()
    for directory, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames.sort()
        for filename in sorted(filenames):
            item = Path(directory) / filename
            try:
                item_size = item.lstat().st_size
                item_hash = _file_hash(item)
                relative = item.relative_to(path).as_posix()
                digest.update(
                    f"{relative}\0{item_size}\0{item_hash}\n".encode("utf-8")
                )
                size += item_size
                count += 1
            except FileNotFoundError:
                continue
    return int(size), int(count), digest.hexdigest()


def _validation(
    *,
    plan: pd.DataFrame,
    protected_paths: set[Path],
    root: Path,
    destination: dict[str, object],
) -> pd.DataFrame:
    safe = plan.loc[
        plan.get("safe_to_archive_later", pd.Series(dtype=bool)).astype(bool)
    ]
    roots = [root / value for value in plan.get("snapshot_path", pd.Series(dtype=str))]
    no_overlap = all(
        not _is_relative_to(right, left)
        for index, left in enumerate(roots)
        for other_index, right in enumerate(roots)
        if index != other_index
    )
    checks = (
        (
            "protected_paths_present",
            bool(protected_paths),
            str(len(protected_paths)),
        ),
        (
            "candidate_roots_do_not_overlap",
            no_overlap,
            str(len(roots)),
        ),
        (
            "safe_candidates_have_manifests",
            safe.empty or safe["manifest_present"].astype(bool).all(),
            "manifest_present=true",
        ),
        (
            "safe_candidates_have_verified_tree_hashes",
            safe.empty
            or (
                safe["hash_verified"].astype(bool).all()
                and safe["manifest_sha256"].astype(str).str.len().eq(64).all()
                and safe["tree_sha256"].astype(str).str.len().eq(64).all()
            ),
            "hash_verified=true;sha256_length=64",
        ),
        (
            "safe_candidates_exclude_current_lineage",
            safe.empty
            or (
                safe["protected_current_paths"].astype(int).eq(0).all()
                and not safe["current_lineage"].astype(bool).any()
            ),
            "protected_current_paths=0",
        ),
        (
            "no_move_performed",
            plan.empty or not plan["move_performed"].astype(bool).any(),
            "move_performed=false",
        ),
        (
            "no_delete_performed",
            plan.empty or not plan["delete_performed"].astype(bool).any(),
            "delete_performed=false",
        ),
        (
            "archive_destination_ready_only_if_off_volume",
            not bool(destination["archive_copy_preflight_ready"])
            or (
                bool(destination["archive_destination_configured"])
                and bool(destination["archive_destination_exists"])
                and bool(destination["archive_destination_writable"])
                and not bool(destination["archive_destination_same_device"])
                and int(destination["archive_destination_free_bytes"])
                >= int(destination["archive_destination_required_bytes"])
            ),
            str(destination["archive_destination_status"]),
        ),
        (
            "archive_release_preflight_false",
            not bool(destination["archive_release_preflight_ready"]),
            "archive_release_preflight_ready=false",
        ),
        (
            "live_authority_false",
            plan.empty or not plan["live_trading_authorized"].astype(bool).any(),
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
                "move_performed": False,
                "delete_performed": False,
                "live_trading_authorized": False,
            }
            for check, passed, evidence in checks
        ]
    )


def _paths(active: Path) -> dict[str, Path]:
    stem = "current_wizard_hyperliquid_storage_reclamation"
    return {
        "plan": active / f"{stem}_plan.csv",
        "validation": active / f"{stem}_validation.csv",
        "manifest": active / f"{stem}_manifest.json",
        "summary_md": active / f"{stem}_summary.md",
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard To Hyperliquid Storage Reclamation Plan",
            "",
            "This is a read-only archive plan. It does not move, link, compress, or delete evidence.",
            "",
            f"- reclamation plan: {summary['reclamation_id']}",
            f"- protected current snapshot paths: {summary['protected_snapshot_paths']}",
            f"- candidate branches: {summary['candidate_branches']}",
            f"- safe archive candidate branches: {summary['safe_archive_candidate_branches']}",
            f"- hash-verified candidate branches: {summary['hash_verified_candidate_branches']}",
            f"- manual-review branches: {summary['manual_review_branches']}",
            f"- current free bytes: {summary['current_free_bytes']}",
            f"- reclaimable bytes if archived off-volume: {summary['reclaimable_bytes_if_archived_off_volume']}",
            f"- projected free bytes: {summary['projected_free_bytes_after_safe_archive']}",
            f"- 3 GiB floor met after safe archive: {summary['storage_floor_met_after_safe_archive']}",
            f"- archive destination: {summary['archive_destination_path'] or 'not configured'}",
            f"- archive destination status: {summary['archive_destination_status']}",
            f"- archive destination blocker: {summary['archive_destination_blocker'] or 'none'}",
            f"- archive copy preflight ready: {summary['archive_copy_preflight_ready']}",
            "- archive apply authority: false",
            "- move performed: false",
            "- delete performed: false",
            "- live trading authorized: false",
            "",
        ]
    )


def _archive_destination_preflight(
    *,
    root: Path,
    archive_destination: Path | None,
    reclaimable_bytes: int,
    require_safe_candidates: bool = True,
) -> dict[str, object]:
    configured_value = str(
        archive_destination
        or os.getenv("WIZARD_HYPERLIQUID_ARCHIVE_DESTINATION", "")
    ).strip()
    required_bytes = int(reclaimable_bytes + ARCHIVE_DESTINATION_RESERVE_BYTES)
    if not configured_value:
        return {
            "archive_destination_configured": False,
            "archive_destination_path": "",
            "archive_destination_exists": False,
            "archive_destination_writable": False,
            "archive_destination_same_device": False,
            "archive_destination_free_bytes": 0,
            "archive_destination_required_bytes": required_bytes,
            "archive_destination_reserve_bytes": ARCHIVE_DESTINATION_RESERVE_BYTES,
            "archive_destination_status": "BLOCKED",
            "archive_destination_blocker": "off_volume_archive_destination_not_configured",
            "archive_copy_preflight_ready": False,
            "archive_release_preflight_ready": False,
        }

    destination = Path(configured_value).expanduser()
    exists = destination.is_dir()
    writable = bool(exists and os.access(destination, os.W_OK))
    same_device = bool(exists and _device_id(destination) == _device_id(root))
    free_bytes = int(shutil.disk_usage(destination).free) if exists else 0
    blockers = []
    if not exists:
        blockers.append("off_volume_archive_destination_missing")
    if exists and not writable:
        blockers.append("off_volume_archive_destination_not_writable")
    if exists and same_device:
        blockers.append("archive_destination_is_same_device_as_project")
    if exists and free_bytes < required_bytes:
        blockers.append("archive_destination_insufficient_free_space")
    if reclaimable_bytes <= 0 and require_safe_candidates:
        blockers.append("no_safe_archive_candidates")
    ready = not blockers
    return {
        "archive_destination_configured": True,
        "archive_destination_path": str(destination.resolve()),
        "archive_destination_exists": exists,
        "archive_destination_writable": writable,
        "archive_destination_same_device": same_device,
        "archive_destination_free_bytes": free_bytes,
        "archive_destination_required_bytes": required_bytes,
        "archive_destination_reserve_bytes": ARCHIVE_DESTINATION_RESERVE_BYTES,
        "archive_destination_status": "READY" if ready else "BLOCKED",
        "archive_destination_blocker": ";".join(blockers),
        "archive_copy_preflight_ready": ready,
        "archive_release_preflight_ready": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _device_id(path: Path) -> int:
    return int(path.resolve().stat().st_dev)


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    return (
        value.astimezone(timezone.utc)
        if value.tzinfo
        else value.replace(tzinfo=timezone.utc)
    )
