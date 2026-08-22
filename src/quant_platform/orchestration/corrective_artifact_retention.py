"""Evidence-aware retention for high-frequency operational receipts and logs."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_bytes,
    atomic_write_text,
    promote_staged_file,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_artifact_retention.v1"
POLICY_PATH = Path("config/corrective_artifact_retention.json")


def run_corrective_artifact_retention(
    *, root: Path = ROOT, now: datetime | None = None, apply: bool = False
) -> CommandResult:
    """Plan or archive only allowlisted, old, unreferenced operational evidence."""

    evaluated_at = _as_utc(now)
    policy_path = root / POLICY_PATH
    policy = _read_json(policy_path)
    _validate_policy(policy)
    referenced_text = _active_reference_text(root)
    cutoff = evaluated_at - timedelta(days=int(policy["receipt_retention_days"]))
    minimum_newest = int(policy["minimum_newest_per_directory"])
    rows: list[dict[str, Any]] = []
    directory_summaries: list[dict[str, Any]] = []
    files_scanned = 0
    for relative_dir in policy["operational_receipt_directories"]:
        source_dir = _safe_root_path(root, str(relative_dir))
        files = (
            sorted(
                (path for path in source_dir.rglob("*") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if source_dir.is_dir()
            else []
        )
        files_scanned += len(files)
        protected_newest = {path.resolve() for path in files[:minimum_newest]}
        directory_candidates = 0
        directory_reclaimable_bytes = 0
        directory_retained_bytes = 0
        for path in files:
            relative = _relative(path, root)
            source_size_bytes = path.stat().st_size
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            referenced = relative in referenced_text
            eligible = bool(
                modified < cutoff and path.resolve() not in protected_newest and not referenced
            )
            archive_path = (
                root
                / "archive"
                / "operational_receipts"
                / evaluated_at.strftime("%Y-%m")
                / relative
            )
            moved = False
            if apply and eligible:
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                if archive_path.exists():
                    if _file_hash(archive_path) != _file_hash(path):
                        raise ValueError(f"retention archive collision: {archive_path}")
                    path.unlink()
                else:
                    promote_staged_file(path, archive_path)
                moved = True
            directory_candidates += int(eligible)
            directory_reclaimable_bytes += source_size_bytes if eligible else 0
            directory_retained_bytes += source_size_bytes if not eligible else 0
            if eligible or referenced:
                rows.append(
                    {
                        "source_path": relative,
                        "source_size_bytes": source_size_bytes,
                        "source_sha256": (
                            _file_hash(archive_path if moved else path) if eligible else ""
                        ),
                        "modified_at_utc": modified.isoformat(),
                        "referenced_by_active_reports": referenced,
                        "protected_as_newest": path.resolve() in protected_newest,
                        "eligible": eligible,
                        "action": (
                            "archived"
                            if moved
                            else "would_archive"
                            if eligible
                            else "retained_active_reference"
                        ),
                        "archive_path": _relative(archive_path, root) if eligible else "",
                    }
                )
        directory_summaries.append(
            {
                "directory": str(relative_dir),
                "files_scanned": len(files),
                "archive_candidates": directory_candidates,
                "projected_reclaimable_bytes": directory_reclaimable_bytes,
                "projected_retained_bytes": directory_retained_bytes,
                "minimum_newest_preserved": min(len(files), minimum_newest),
            }
        )

    rotated_logs = _rotate_logs(root=root, policy=policy, apply=apply)
    candidates = sum(row["eligible"] for row in rows)
    archived = sum(row["action"] == "archived" for row in rows)
    projected_reclaimable_bytes = sum(
        int(row["projected_reclaimable_bytes"])
        for row in directory_summaries
    )
    projected_retained_bytes = sum(
        int(row["projected_retained_bytes"])
        for row in directory_summaries
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "mode": "apply" if apply else "dry_run",
        "policy_path": str(POLICY_PATH),
        "policy_sha256": _file_hash(policy_path),
        "files_scanned": files_scanned,
        "archive_candidates": candidates,
        "files_archived": archived,
        "retention_horizon_days": int(policy["receipt_retention_days"]),
        "projected_reclaimable_bytes": projected_reclaimable_bytes,
        "projected_retained_bytes": projected_retained_bytes,
        "capacity_forecast_status": "PASS_POLICY_BOUNDED",
        "logs_rotated": len(rotated_logs),
        "dry_run_moved_nothing": not apply and archived == 0,
        "scientific_evidence_directories_touched": [],
        "status": "PASS_RETENTION_APPLIED" if apply else "PASS_RETENTION_DRY_RUN",
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "receipt_rows": rows,
        "directory_summaries": directory_summaries,
        "rotated_logs": rotated_logs,
    }
    payload["receipt_id"] = (
        "retention_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    status_path = root / "reports" / "active" / "corrective_artifact_retention_status.json"
    _atomic_json(payload, status_path)
    paths = {"status": status_path}
    if apply:
        manifest = (
            root / "archive" / "operational_receipt_retention" / f"{payload['receipt_id']}.json"
        )
        _write_or_validate_immutable_json(payload, manifest)
        paths["archive_manifest"] = manifest
    return CommandResult(paths=paths, summary=payload)


def _validate_policy(policy: dict[str, Any]) -> None:
    if policy.get("schema_version") != "thewiz.corrective_artifact_retention_policy.v1":
        raise ValueError("retention policy schema is invalid")
    if int(policy.get("receipt_retention_days", 0)) <= 0:
        raise ValueError("receipt_retention_days must be positive")
    if int(policy.get("minimum_newest_per_directory", 0)) <= 0:
        raise ValueError("minimum_newest_per_directory must be positive")
    operational = set(policy.get("operational_receipt_directories", []))
    forbidden = set(policy.get("never_archive_directories", []))
    if not operational or operational & forbidden:
        raise ValueError("retention allowlist is empty or intersects scientific evidence")


def _active_reference_text(root: Path) -> str:
    fragments: list[str] = []
    active = root / "reports" / "active"
    if not active.is_dir():
        return ""
    # Canonical pointers live directly under reports/active. Nested folders are
    # receipt stores, not authorities, and can contain multi-gigabyte snapshots.
    for path in active.iterdir():
        if not path.is_file() or path.stat().st_size > 25_000_000:
            continue
        try:
            fragments.append(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    return "\n".join(fragments)


def _rotate_logs(*, root: Path, policy: dict[str, Any], apply: bool) -> list[dict[str, Any]]:
    threshold = int(policy.get("log_rotation_threshold_bytes", 0))
    keep = int(policy.get("log_rotations_to_keep", 0))
    rotated: list[dict[str, Any]] = []
    for pattern in policy.get("log_paths", []):
        for path in root.glob(str(pattern)):
            if not path.is_file() or path.stat().st_size <= threshold:
                continue
            entry = {
                "path": _relative(path, root),
                "size_before": path.stat().st_size,
                "action": "would_rotate",
            }
            if apply:
                for index in range(keep, 0, -1):
                    source = path.with_name(f"{path.name}.{index}")
                    target = path.with_name(f"{path.name}.{index + 1}")
                    if index == keep:
                        source.unlink(missing_ok=True)
                    elif source.exists():
                        promote_staged_file(source, target)
                atomic_write_bytes(
                    path.with_name(f"{path.name}.1"),
                    path.read_bytes(),
                )
                atomic_write_text(path, "", encoding="utf-8")
                entry["action"] = "rotated"
            rotated.append(entry)
    return rotated


def _safe_root_path(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if Path(relative).is_absolute() or not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"retention path escapes root: {relative}")
    return candidate


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _as_utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        promote_staged_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"immutable retention manifest mismatch: {path}")
        return
    _atomic_text(path, expected)
