"""Lease-fenced publication helpers for Phase 00 authority artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    PublicationLease,
    publication_lease_for_path,
    validate_current_publication_lease,
)


@dataclass(frozen=True)
class EvidenceStore:
    """Publish under one approved repository root and publication scope."""

    root: Path
    scope: str

    def __post_init__(self) -> None:
        canonical_root = self.root.resolve()
        if not self.scope.strip():
            raise ValueError("evidence store scope is required")
        object.__setattr__(self, "root", canonical_root)

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        resolved = candidate.resolve(strict=False)
        if resolved != self.root and self.root not in resolved.parents:
            raise ValueError("evidence store path escapes approved root")
        if candidate.is_symlink():
            raise ValueError("evidence store target cannot be a symlink")
        return candidate

    def publish_text(self, path: str | Path, payload: str, *, mode: int = 0o600) -> Path:
        target = self.resolve(path)
        with publication_lease_for_path(target, scope=self.scope):
            validate_current_publication_lease(self.root, target=target)
            atomic_write_text(
                target,
                payload,
                mode=mode,
                publication_scope=self.scope,
            )
        return target

    def publish_json(self, path: str | Path, payload: dict[str, Any]) -> Path:
        return self.publish_text(
            path,
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
        )

    def publish_immutable_json(
        self, path: str | Path, payload: dict[str, Any]
    ) -> Path:
        target = self.resolve(path)
        with publication_lease_for_path(target, scope=self.scope):
            validate_current_publication_lease(self.root, target=target)
            write_immutable_json(
                target,
                payload,
                publication_scope=self.scope,
            )
        return target


def current_store_lease(store: EvidenceStore, target: str | Path) -> PublicationLease:
    """Expose the validated lease for receipt and fault-test construction."""

    return validate_current_publication_lease(
        store.root,
        target=store.resolve(target),
        required_scope=store.scope,
    )
