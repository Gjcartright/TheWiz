"""Path-safe content integrity helpers for dynamic-agent evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path


class EvidenceIntegrityError(ValueError):
    """Raised when evidence is missing, escapes the project root, or changes."""


def resolve_evidence_path(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise EvidenceIntegrityError("evidence path must be a non-empty relative path")
    root_resolved = root.resolve()
    resolved = (root_resolved / relative_path).resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise EvidenceIntegrityError("evidence path escapes the project root")
    if not resolved.is_file():
        raise EvidenceIntegrityError("evidence file does not exist")
    return resolved


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_hash(root: Path, relative_path: str) -> str:
    return sha256_file(resolve_evidence_path(root, relative_path))
