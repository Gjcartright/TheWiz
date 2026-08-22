"""Small cross-process lock primitive shared by scheduled producers."""

from __future__ import annotations

import fcntl
import json
import os
import platform
import stat
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

GOVERNED_EVIDENCE_LOCK = ".runtime_locks/governed_evidence.lock"
SCHEDULER_TERMINAL_EVIDENCE_LOCK = ".runtime_locks/scheduler_terminal_evidence.lock"
PHASE00_MAINTENANCE_MARKER = ".runtime_control/phase00_maintenance.json"
DEFAULT_PUBLICATION_LEASE_TTL_SECONDS = 6 * 60 * 60
PUBLICATION_LEASE_SCHEMA_VERSION = "thewiz.publication_lease.v2"
SCHEDULER_COMPATIBILITY_LOCK_SCHEMA_VERSION = "thewiz.scheduler_compatibility_lock.v2"
SCHEDULER_TERMINAL_SCOPE = "scheduler_terminal"
PHASE00_CONTROL_SCOPE = "phase00_control"
SCHEDULER_TERMINAL_TARGET_PREFIXES = (
    "data/research/scheduler_run_intents",
    "data/research/scheduler_slot_claims",
    "data/research/scheduler_terminal_receipts",
    "reports/active/scheduler_terminal_receipts",
)
PHASE00_CONTROL_TARGET_PREFIXES = (
    ".runtime_control/phase00_maintenance.json",
    "data/research/phase00_control",
    "reports/active/phase00_control_checkpoint.json",
    "reports/active/phase00_descendant_control.json",
    "reports/active/phase00_closure.json",
    "reports/active/phase00_acceptance_matrix.csv",
    "reports/active/phase00_fault_catalog.csv",
    "reports/active/phase00_active_artifact_lineage.json",
    "reports/active/phase00_maintenance.json",
    "reports/active/phase00_source_evidence_index.csv",
)
RESEARCH_TARGET_PREFIXES = (
    "archive",
    "config",
    "data",
    "docs",
    "models",
    "reports",
)
SCHEDULED_RESEARCH_TARGET_PREFIXES = (
    "data",
    "reports",
)
SCHEDULER_CONFIG_TARGET_PREFIXES = (
    ".runtime_control",
    "config",
    "reports",
)
SCOPED_TARGET_PREFIXES = {
    SCHEDULER_TERMINAL_SCOPE: SCHEDULER_TERMINAL_TARGET_PREFIXES,
    PHASE00_CONTROL_SCOPE: PHASE00_CONTROL_TARGET_PREFIXES,
    "research": RESEARCH_TARGET_PREFIXES,
    "daily_research": SCHEDULED_RESEARCH_TARGET_PREFIXES,
    "public_l2": SCHEDULED_RESEARCH_TARGET_PREFIXES,
    "wizard_external_research": SCHEDULED_RESEARCH_TARGET_PREFIXES,
    "wizard_proof_launcher": SCHEDULED_RESEARCH_TARGET_PREFIXES,
    "scheduler_config": SCHEDULER_CONFIG_TARGET_PREFIXES,
}


def _process_start_identity(pid: int | None = None) -> tuple[str, str]:
    target_pid = os.getpid() if pid is None else int(pid)
    proc_stat = Path(f"/proc/{target_pid}/stat")
    if proc_stat.is_file():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
            material = f"{target_pid}:{fields[21]}"
            return sha256(material.encode("utf-8")).hexdigest()[:24], "proc_stat"
        except (OSError, IndexError, UnicodeDecodeError):
            pass
    try:
        completed = subprocess.run(
            ["ps", "-p", str(target_pid), "-o", "lstart="],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = subprocess.CompletedProcess([], 127, stdout="", stderr="")
    observed = (completed.stdout or "").strip()
    if completed.returncode == 0 and observed:
        material = f"{target_pid}:{observed}"
        return sha256(material.encode("utf-8")).hexdigest()[:24], "ps_lstart"
    if target_pid != os.getpid():
        return "", "unavailable"
    material = f"{target_pid}:{time.monotonic_ns()}"
    return sha256(material.encode("utf-8")).hexdigest()[:24], "monotonic_fallback"


def _boot_identity() -> tuple[str, str]:
    boot_id = Path("/proc/sys/kernel/random/boot_id")
    if boot_id.is_file():
        try:
            material = boot_id.read_text(encoding="utf-8").strip()
            if material:
                return sha256(material.encode("utf-8")).hexdigest()[:24], "proc_boot_id"
        except (OSError, UnicodeDecodeError):
            pass
    try:
        completed = subprocess.run(
            ["sysctl", "-n", "kern.boottime"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        completed = subprocess.CompletedProcess([], 127, stdout="", stderr="")
    observed = (completed.stdout or "").strip()
    if completed.returncode == 0 and observed:
        return sha256(observed.encode("utf-8")).hexdigest()[:24], "sysctl_boottime"
    approximate_boot_epoch = int(time.time() - time.monotonic())
    material = f"{platform.node()}:{approximate_boot_epoch}"
    return sha256(material.encode("utf-8")).hexdigest()[:24], "monotonic_fallback"


_PROCESS_START_ID, _PROCESS_START_SOURCE = _process_start_identity()
_BOOT_ID, _BOOT_ID_SOURCE = _boot_identity()


def current_scheduler_process_identity() -> dict[str, object]:
    """Return the process-generation identity shared by locks and run intents."""

    return {
        "pid": os.getpid(),
        "process_start_id": _PROCESS_START_ID,
        "process_start_source": _PROCESS_START_SOURCE,
        "boot_id": _BOOT_ID,
        "boot_id_source": _BOOT_ID_SOURCE,
        "host": platform.node(),
    }


def scheduler_process_owner_matches(state: dict[str, Any]) -> bool | None:
    """Return true for the same live process, false for dead/reused, or unknown."""

    try:
        pid = int(state.get("pid"))
    except (TypeError, ValueError):
        return None
    if not _lock_owner_alive(pid):
        return False
    required = (
        "process_start_id",
        "process_start_source",
        "boot_id",
        "boot_id_source",
        "host",
    )
    if any(not str(state.get(field, "")).strip() for field in required):
        return None
    if state.get("host") != platform.node():
        return False
    if state.get("boot_id") != _BOOT_ID or state.get("boot_id_source") != _BOOT_ID_SOURCE:
        return False
    if pid == os.getpid():
        return bool(
            state.get("process_start_id") == _PROCESS_START_ID
            and state.get("process_start_source") == _PROCESS_START_SOURCE
        )
    observed_id, observed_source = _process_start_identity(pid)
    if not observed_id or observed_source == "unavailable":
        return None
    return bool(
        state.get("process_start_id") == observed_id
        and state.get("process_start_source") == observed_source
    )


@dataclass(frozen=True)
class PublicationLease:
    """Process-bound authority required by governed publication sinks."""

    schema_version: str
    lease_id: str
    generation: int
    root: str
    lock_path: str
    pid: int
    process_start_id: str
    process_start_source: str
    boot_id: str
    boot_id_source: str
    host: str
    run_id: str
    scope: str
    acquired_at_utc: str
    expires_at_utc: str
    maintenance_controller: bool
    maintenance_exempt: bool
    lock_device: int
    lock_inode: int


_CURRENT_PUBLICATION_LEASE: ContextVar[PublicationLease | None] = ContextVar(
    "thewiz_current_publication_lease", default=None
)


class GovernedEvidenceLockBusy(RuntimeError):
    """Raised when another process owns the governed-evidence write lease."""

    def __init__(self, reason_code: str):
        normalized = reason_code.split(":", 1)[0]
        if normalized not in {
            "governed_evidence_write_lock_busy",
            "governed_evidence_write_lock_timeout",
            "phase00_maintenance_active",
        }:
            raise ValueError("governed_evidence_lock_reason_code_invalid")
        super().__init__(reason_code)
        self.evidence_reason_code = normalized
        self.evidence_reason_only = True


class GovernedEvidenceMaintenanceActive(GovernedEvidenceLockBusy):
    """Raised when Phase 00 maintenance denies a new governed writer."""


class PublicationLeaseError(RuntimeError):
    """Raised when a final publication sink has no valid current lease."""


@contextmanager
def governed_evidence_read_lock(
    root: Path,
    *,
    blocking: bool = True,
    wait_timeout_seconds: float | None = None,
    create_if_missing: bool = False,
) -> Iterator[Path]:
    """Hold a shared evidence lock without minting publication authority."""

    canonical_root = root.resolve()
    path = canonical_root / GOVERNED_EVIDENCE_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        if not create_if_missing:
            raise PublicationLeaseError("governed_evidence_read_lock_missing") from None
        create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            create_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            create_flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, create_flags, 0o600)
            created = True
        except FileExistsError:
            descriptor = os.open(path, flags)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise PublicationLeaseError("governed_evidence_read_lock_not_regular")
        _acquire_flock(
            descriptor,
            blocking=blocking,
            wait_timeout_seconds=wait_timeout_seconds,
            shared=True,
        )
        yield path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
        if created:
            try:
                observed = path.lstat()
            except FileNotFoundError:
                observed = None
            if (
                observed is not None
                and observed.st_dev == identity.st_dev
                and observed.st_ino == identity.st_ino
            ):
                path.unlink()


@contextmanager
def governed_evidence_write_lock(
    root: Path,
    *,
    blocking: bool,
    wait_timeout_seconds: float | None = None,
    scope: str = "research",
    run_id: str = "",
    lease_ttl_seconds: int = DEFAULT_PUBLICATION_LEASE_TTL_SECONDS,
) -> Iterator[Path]:
    """Serialize all writes to governed evidence across schedulers and pytest."""

    _require_publication_authority(root, scope=scope)
    existing = current_publication_lease()
    canonical_root = root.resolve()
    if existing is not None and Path(existing.root) == canonical_root:
        if existing.scope != scope:
            raise PublicationLeaseError("publication_lease_scope_reentry_denied")
        validate_current_publication_lease(
            canonical_root,
            required_scope=scope,
        )
        yield Path(existing.lock_path)
        return
    with _governed_evidence_lock(
        canonical_root,
        blocking=blocking,
        wait_timeout_seconds=wait_timeout_seconds,
        maintenance_controller=False,
        scope=scope,
        run_id=run_id,
        lease_ttl_seconds=lease_ttl_seconds,
        lock_relative_path=GOVERNED_EVIDENCE_LOCK,
        maintenance_exempt=False,
    ) as path:
        yield path


@contextmanager
def scheduler_terminal_write_lock(
    root: Path,
    *,
    blocking: bool = True,
    wait_timeout_seconds: float | None = None,
    run_id: str = "",
    lease_ttl_seconds: int = DEFAULT_PUBLICATION_LEASE_TTL_SECONDS,
) -> Iterator[Path]:
    """Acquire the maintenance-safe lane restricted to scheduler terminal evidence."""

    _require_publication_authority(root, scope=SCHEDULER_TERMINAL_SCOPE)
    existing = current_publication_lease()
    canonical_root = root.resolve()
    if existing is not None and Path(existing.root) == canonical_root:
        if existing.scope != SCHEDULER_TERMINAL_SCOPE:
            raise PublicationLeaseError("scheduler_terminal_lease_reentry_denied")
        validate_current_publication_lease(
            canonical_root,
            required_scope=SCHEDULER_TERMINAL_SCOPE,
        )
        yield Path(existing.lock_path)
        return
    with _governed_evidence_lock(
        canonical_root,
        blocking=blocking,
        wait_timeout_seconds=wait_timeout_seconds,
        maintenance_controller=False,
        scope=SCHEDULER_TERMINAL_SCOPE,
        run_id=run_id,
        lease_ttl_seconds=lease_ttl_seconds,
        lock_relative_path=SCHEDULER_TERMINAL_EVIDENCE_LOCK,
        maintenance_exempt=True,
    ) as path:
        yield path


@contextmanager
def _phase00_maintenance_controller_lock(
    root: Path,
    *,
    wait_timeout_seconds: float,
) -> Iterator[Path]:
    """Internal controller lease used only to activate, checkpoint, or resume."""

    with _governed_evidence_lock(
        root,
        blocking=True,
        wait_timeout_seconds=wait_timeout_seconds,
        maintenance_controller=True,
        scope="phase00_control",
        run_id="phase00_maintenance_controller",
        lease_ttl_seconds=DEFAULT_PUBLICATION_LEASE_TTL_SECONDS,
        lock_relative_path=GOVERNED_EVIDENCE_LOCK,
        maintenance_exempt=False,
    ) as path:
        yield path


@contextmanager
def _governed_evidence_lock(
    root: Path,
    *,
    blocking: bool,
    wait_timeout_seconds: float | None,
    maintenance_controller: bool,
    scope: str,
    run_id: str,
    lease_ttl_seconds: int,
    lock_relative_path: str,
    maintenance_exempt: bool,
) -> Iterator[Path]:
    root = root.resolve()
    if not scope.strip():
        raise ValueError("publication lease scope is required")
    if lease_ttl_seconds <= 0:
        raise ValueError("publication lease ttl must be positive")
    path = root / lock_relative_path
    handle, lock_identity = _open_secure_lock_file(root, lock_relative_path)
    acquired = False
    context_token = None
    try:
        _acquire_flock(
            handle.fileno(),
            blocking=blocking,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        acquired = True
        if not maintenance_controller and not maintenance_exempt:
            maintenance = read_phase00_maintenance_state(root)
            if maintenance["blocking"]:
                maintenance_id = str(maintenance.get("maintenance_id", "unknown"))
                raise GovernedEvidenceMaintenanceActive(
                    f"phase00_maintenance_active:{maintenance_id}"
                )
        handle.seek(0)
        previous: dict[str, Any] = {}
        try:
            value = json.load(handle)
            if isinstance(value, dict):
                previous = value
        except (OSError, ValueError, json.JSONDecodeError):
            previous = {}
        try:
            previous_generation = int(previous.get("generation", 0))
        except (TypeError, ValueError):
            previous_generation = 0
        acquired_at = datetime.now(UTC)
        lease = PublicationLease(
            schema_version=PUBLICATION_LEASE_SCHEMA_VERSION,
            lease_id=f"publicationlease_{uuid4().hex}",
            generation=max(previous_generation, 0) + 1,
            root=str(root),
            lock_path=str(path),
            pid=os.getpid(),
            process_start_id=_PROCESS_START_ID,
            process_start_source=_PROCESS_START_SOURCE,
            boot_id=_BOOT_ID,
            boot_id_source=_BOOT_ID_SOURCE,
            host=platform.node(),
            run_id=run_id.strip() or f"pid_{os.getpid()}",
            scope=scope.strip(),
            acquired_at_utc=acquired_at.isoformat(),
            expires_at_utc=(acquired_at + timedelta(seconds=lease_ttl_seconds)).isoformat(),
            maintenance_controller=maintenance_controller,
            maintenance_exempt=maintenance_exempt,
            lock_device=lock_identity.st_dev,
            lock_inode=lock_identity.st_ino,
        )
        handle.seek(0)
        handle.truncate(0)
        json.dump(asdict(lease), handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        context_token = _CURRENT_PUBLICATION_LEASE.set(lease)
        yield path
    finally:
        if context_token is not None:
            _CURRENT_PUBLICATION_LEASE.reset(context_token)
        if acquired:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def current_publication_lease() -> PublicationLease | None:
    """Return the current process-context lease, if one is active."""

    return _CURRENT_PUBLICATION_LEASE.get()


def governed_root_for_path(path: Path) -> Path | None:
    """Find the quant repository root that owns a publication target."""

    candidate = Path(os.path.abspath(path))
    current = current_publication_lease()
    if current is not None:
        current_root = Path(current.root)
        if candidate == current_root or current_root in candidate.parents:
            return current_root
    for parent in (candidate.parent, *candidate.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src" / "quant_platform").is_dir():
            return parent.resolve()
    return None


@contextmanager
def publication_lease_for_path(
    path: Path,
    *,
    scope: str,
    blocking: bool = True,
) -> Iterator[PublicationLease | None]:
    """Acquire or reuse the lease that governs a repository publication path."""

    root = governed_root_for_path(path)
    if root is None:
        yield None
        return
    existing = current_publication_lease()
    if existing is not None and Path(existing.root) == root:
        validate_current_publication_lease(
            root,
            target=path,
            required_scope=scope,
        )
        yield existing
        return
    with governed_evidence_write_lock(root, blocking=blocking, scope=scope):
        lease = validate_current_publication_lease(
            root,
            target=path,
            required_scope=scope,
        )
        yield lease


def _require_publication_authority(root: Path, *, scope: str) -> None:
    from quant_platform.orchestration.effect_authority import (
        EffectAuthorityError,
        require_publication_authority,
    )

    try:
        require_publication_authority(
            root=root,
            publication_scope=scope,
        )
    except EffectAuthorityError as exc:
        raise PublicationLeaseError(str(exc)) from exc


def validate_current_publication_lease(
    root: Path,
    *,
    target: Path | None = None,
    required_scope: str | None = None,
) -> PublicationLease:
    """Fail closed unless the current sink is protected by the live lock generation."""

    root = root.resolve()
    lease = current_publication_lease()
    if lease is None:
        raise PublicationLeaseError("publication_lease_missing")
    if Path(lease.root) != root:
        raise PublicationLeaseError("publication_lease_root_mismatch")
    if lease.pid != os.getpid() or lease.process_start_id != _PROCESS_START_ID:
        raise PublicationLeaseError("publication_lease_process_mismatch")
    if lease.process_start_source != _PROCESS_START_SOURCE:
        raise PublicationLeaseError("publication_lease_process_source_mismatch")
    if lease.boot_id != _BOOT_ID or lease.host != platform.node():
        raise PublicationLeaseError("publication_lease_host_or_boot_mismatch")
    if lease.boot_id_source != _BOOT_ID_SOURCE:
        raise PublicationLeaseError("publication_lease_boot_source_mismatch")
    if required_scope is not None and lease.scope != required_scope:
        raise PublicationLeaseError("publication_lease_scope_mismatch")
    expires = pd.to_datetime(lease.expires_at_utc, utc=True, errors="coerce")
    if pd.isna(expires) or expires.to_pydatetime() <= datetime.now(UTC):
        raise PublicationLeaseError("publication_lease_expired")
    if target is not None:
        if target.is_symlink():
            raise PublicationLeaseError("publication_target_symlink_denied")
        resolved_target = target.resolve(strict=False)
        if resolved_target != root and root not in resolved_target.parents:
            raise PublicationLeaseError("publication_target_outside_root")
        if not _target_allowed_for_scope(root, target, lease.scope):
            raise PublicationLeaseError(f"publication_{lease.scope}_target_denied")
    maintenance = read_phase00_maintenance_state(root)
    terminal_exemption_valid = bool(
        lease.maintenance_exempt
        and lease.scope == SCHEDULER_TERMINAL_SCOPE
        and Path(lease.lock_path) == root / SCHEDULER_TERMINAL_EVIDENCE_LOCK
        and (target is None or _target_allowed_for_scope(root, target, lease.scope))
    )
    if (
        maintenance["blocking"]
        and not lease.maintenance_controller
        and not terminal_exemption_valid
    ):
        raise GovernedEvidenceMaintenanceActive(
            f"phase00_maintenance_active:{maintenance.get('maintenance_id', 'unknown')}"
        )
    state_path = Path(lease.lock_path)
    try:
        descriptor = _open_existing_regular_file(state_path)
        with os.fdopen(descriptor, "r", encoding="utf-8") as state_handle:
            observed_identity = os.fstat(state_handle.fileno())
            state = json.load(state_handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, PublicationLeaseError) as exc:
        raise PublicationLeaseError("publication_lease_state_unreadable") from exc
    if (
        observed_identity.st_dev != lease.lock_device
        or observed_identity.st_ino != lease.lock_inode
    ):
        raise PublicationLeaseError("publication_lease_lock_identity_mismatch")
    expected = {
        "lease_id": lease.lease_id,
        "generation": lease.generation,
        "pid": lease.pid,
        "process_start_id": lease.process_start_id,
        "process_start_source": lease.process_start_source,
        "boot_id_source": lease.boot_id_source,
        "scope": lease.scope,
        "maintenance_exempt": lease.maintenance_exempt,
        "lock_device": lease.lock_device,
        "lock_inode": lease.lock_inode,
    }
    if not isinstance(state, dict) or any(
        state.get(key) != value for key, value in expected.items()
    ):
        raise PublicationLeaseError("publication_lease_generation_mismatch")
    return lease


def _target_allowed_for_scope(root: Path, target: Path, scope: str) -> bool:
    prefixes = SCOPED_TARGET_PREFIXES.get(scope)
    if prefixes is None:
        return False
    resolved = target.resolve(strict=False)
    for prefix in prefixes:
        allowed = (root / prefix).resolve(strict=False)
        if resolved == allowed or allowed in resolved.parents:
            return True
    return False


def _open_secure_lock_file(
    root: Path,
    lock_relative_path: str,
) -> tuple[Any, os.stat_result]:
    relative = Path(lock_relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.name:
        raise PublicationLeaseError("publication_lock_path_invalid")
    parent_descriptor = _open_or_create_directory(root, relative.parent)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(relative.name, flags, 0o600, dir_fd=parent_descriptor)
    except OSError as exc:
        raise PublicationLeaseError("publication_lock_open_failed") from exc
    finally:
        os.close(parent_descriptor)
    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise PublicationLeaseError("publication_lock_not_regular")
        if identity.st_uid != os.getuid():
            raise PublicationLeaseError("publication_lock_owner_mismatch")
        if identity.st_nlink != 1:
            raise PublicationLeaseError("publication_lock_link_count_invalid")
        os.fchmod(descriptor, 0o600)
        path_identity = (root / relative).lstat()
        if path_identity.st_dev != identity.st_dev or path_identity.st_ino != identity.st_ino:
            raise PublicationLeaseError("publication_lock_path_identity_mismatch")
        return os.fdopen(descriptor, "r+", encoding="utf-8"), identity
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory(root: Path, relative: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(root, flags)
    try:
        for part in relative.parts:
            if part in {"", "."}:
                continue
            if part == "..":
                raise PublicationLeaseError("publication_directory_escape")
            try:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_existing_regular_file(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    identity = os.fstat(descriptor)
    if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
        os.close(descriptor)
        raise PublicationLeaseError("publication_state_not_regular")
    return descriptor


def read_phase00_maintenance_state(root: Path) -> dict[str, Any]:
    """Return a fail-closed maintenance state without exposing repository secrets."""

    marker = root / PHASE00_MAINTENANCE_MARKER
    if not marker.exists():
        return {
            "blocking": False,
            "status": "INACTIVE",
            "maintenance_id": "",
            "marker_path": str(marker),
        }
    if marker.is_symlink() or not marker.is_file():
        return {
            "blocking": True,
            "status": "BLOCKED_INVALID_MARKER",
            "maintenance_id": "",
            "marker_path": str(marker),
        }
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "blocking": True,
            "status": "BLOCKED_INVALID_MARKER",
            "maintenance_id": "",
            "marker_path": str(marker),
        }
    if not isinstance(payload, dict) or not str(payload.get("maintenance_id", "")).strip():
        return {
            "blocking": True,
            "status": "BLOCKED_INVALID_MARKER",
            "maintenance_id": "",
            "marker_path": str(marker),
        }
    expires = pd.to_datetime(payload.get("expires_at_utc"), utc=True, errors="coerce")
    expired = bool(pd.notna(expires) and expires.to_pydatetime() <= datetime.now(UTC))
    return {
        **payload,
        "blocking": True,
        "status": ("BLOCKED_EXPIRED_REQUIRES_EXPLICIT_RESUME" if expired else "ACTIVE"),
        "expired": expired,
        "marker_path": str(marker),
    }


def _acquire_flock(
    descriptor: int,
    *,
    blocking: bool,
    wait_timeout_seconds: float | None,
    shared: bool = False,
) -> None:
    if wait_timeout_seconds is not None and wait_timeout_seconds < 0:
        raise ValueError("wait_timeout_seconds must be non-negative")
    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    if not blocking:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GovernedEvidenceLockBusy("governed_evidence_write_lock_busy") from exc
        return
    if wait_timeout_seconds is None:
        fcntl.flock(descriptor, operation)
        return
    deadline = time.monotonic() + wait_timeout_seconds
    while True:
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
            return
        except BlockingIOError as exc:
            if time.monotonic() >= deadline:
                raise GovernedEvidenceLockBusy("governed_evidence_write_lock_timeout") from exc
            time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))


def acquire_scheduler_lock(path: Path, *, now: datetime, timeout_seconds: int) -> None:
    """Acquire a compatibility lock without trusting PID reuse or malformed state."""

    if timeout_seconds <= 0:
        raise ValueError("scheduler lock timeout must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        state: dict[str, Any] = {}
        malformed = False
        try:
            identity = path.lstat()
            if not stat.S_ISREG(identity.st_mode):
                raise FileExistsError("active_scheduler_lock_present")
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                state = value
            else:
                malformed = True
            started = pd.to_datetime(state.get("started_at_utc"), utc=True, errors="coerce")
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            identity = path.lstat()
            started = pd.NaT
            malformed = True
        if pd.isna(started):
            malformed = True
            observed_started = datetime.fromtimestamp(identity.st_mtime, tz=UTC)
        else:
            observed_started = started.to_pydatetime()
        stale = observed_started < now - timedelta(seconds=timeout_seconds)
        owner_matches = _scheduler_lock_owner_matches(state)
        # A positively identified live process owns the lock regardless of whether
        # ancillary age metadata is malformed. Age is only an eviction signal once
        # ownership is absent or conclusively belongs to another process generation.
        if owner_matches is True:
            raise FileExistsError("active_scheduler_lock_present")
        if stale and (malformed or owner_matches is False):
            _quarantine_scheduler_lock(path)
        else:
            raise FileExistsError("active_scheduler_lock_present")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        payload = json.dumps(
            {
                "schema_version": SCHEDULER_COMPATIBILITY_LOCK_SCHEMA_VERSION,
                "pid": os.getpid(),
                "process_start_id": _PROCESS_START_ID,
                "process_start_source": _PROCESS_START_SOURCE,
                "boot_id": _BOOT_ID,
                "boot_id_source": _BOOT_ID_SOURCE,
                "host": platform.node(),
                "started_at_utc": now.isoformat(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        written = handle.write(payload)
        if written != len(payload):
            raise OSError("short scheduler lock write")
        handle.flush()
        os.fsync(handle.fileno())


def _scheduler_lock_owner_matches(state: dict[str, Any]) -> bool | None:
    """Return true for the same process, false for mismatch, or none if ambiguous."""

    return scheduler_process_owner_matches(state)


def _quarantine_scheduler_lock(path: Path) -> Path:
    """Move one stale compatibility lock to deterministic forensic quarantine."""

    payload = path.read_bytes()
    quarantine_dir = path.parent / ".scheduler_lock_quarantine"
    quarantine_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = sha256(payload).hexdigest()[:24]
    quarantine_path = quarantine_dir / f"{path.name}.{digest}.quarantined"
    if quarantine_path.exists():
        if quarantine_path.read_bytes() != payload:
            raise FileExistsError("scheduler_lock_quarantine_collision")
        path.unlink()
    else:
        path.replace(quarantine_path)
    return quarantine_path


def _lock_owner_alive(raw_pid: Any) -> bool:
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
