"""Verified references to immutable upstream snapshot artifacts."""

from __future__ import annotations

import gzip
from hashlib import sha256
from pathlib import Path
from typing import Any


def verified_snapshot_reference(
    *,
    root: Path,
    active_path: Path,
    upstream_manifest: dict[str, Any],
    artifact_key: str,
) -> Path:
    """Resolve and verify an immutable snapshot counterpart for an active input."""

    artifacts = upstream_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise ValueError("Upstream manifest artifacts must be an object")
    snapshot_value = str(artifacts.get(f"snapshot_{artifact_key}", "")).strip()
    if not snapshot_value:
        raise ValueError(
            f"Upstream manifest has no snapshot_{artifact_key} artifact reference"
        )
    snapshot_path = root / snapshot_value
    _require_immutable_snapshot_path(root, snapshot_path)
    if not active_path.is_file():
        raise FileNotFoundError(f"Active lineage input is missing: {active_path}")
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Immutable lineage reference is missing: {snapshot_path}"
        )
    active_hash = artifact_content_hash(active_path)
    snapshot_hash = artifact_content_hash(snapshot_path)
    if active_hash != snapshot_hash:
        raise ValueError(
            "Active and immutable lineage artifacts differ: "
            f"active={active_path};snapshot={snapshot_path}"
        )
    return snapshot_path


def existing_snapshot_reference(*, root: Path, snapshot_path: Path) -> Path:
    """Validate a path that is already an immutable upstream snapshot artifact."""

    _require_immutable_snapshot_path(root, snapshot_path)
    if not snapshot_path.is_file():
        raise FileNotFoundError(
            f"Immutable lineage reference is missing: {snapshot_path}"
        )
    return snapshot_path


def artifact_content_hash(path: Path) -> str:
    """Hash underlying content, ignoring non-semantic gzip container metadata."""

    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return sha256(handle.read()).hexdigest()
    return sha256(path.read_bytes()).hexdigest()


def unique_file_bytes(paths: list[Path] | tuple[Path, ...]) -> int:
    """Return physical file sizes once per resolved evidence path."""

    unique = {path.resolve() for path in paths}
    return int(sum(path.stat().st_size for path in unique if path.is_file()))


def _require_immutable_snapshot_path(root: Path, path: Path) -> None:
    snapshot_root = (root / "reports" / "snapshots").resolve()
    try:
        path.resolve().relative_to(snapshot_root)
    except ValueError as exc:
        raise ValueError(
            f"Lineage reference is outside immutable snapshot storage: {path}"
        ) from exc
