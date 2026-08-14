"""Shared runtime storage policy for corrective background processes."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

RUNTIME_TEMP_DIRNAME = ".runtime_tmp"
RUNTIME_TEMP_ENV_NAMES = ("TMPDIR", "TMP", "TEMP")
RUNTIME_AGENT_DIRNAME = ".runtime_agents"


def runtime_temp_directory(root: Path) -> Path:
    """Return the repository-local temp directory without following unsafe links."""

    resolved_root = root.resolve()
    candidate = root / RUNTIME_TEMP_DIRNAME
    if candidate.is_symlink():
        raise ValueError("runtime temp directory must not be a symbolic link")
    resolved_candidate = candidate.resolve(strict=False)
    if resolved_candidate.parent != resolved_root:
        raise ValueError("runtime temp directory must remain inside the repository root")
    return candidate


def ensure_runtime_temp_directory(root: Path) -> Path:
    """Create the private workspace temp directory used by LaunchAgents."""

    path = runtime_temp_directory(root)
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"runtime temp path is not a directory: {path}")
    path.chmod(0o700)
    return path


def launch_agent_runtime_environment(root: Path) -> dict[str, str]:
    """Build the exact environment required by corrective LaunchAgents."""

    runtime_temp = str(runtime_temp_directory(root))
    return {
        "PYTHONPATH": str(root / "src"),
        **{name: runtime_temp for name in RUNTIME_TEMP_ENV_NAMES},
    }


def workspace_launch_agent_path(root: Path, label: str) -> Path:
    if not label or "/" in label:
        raise ValueError("invalid LaunchAgent label")
    return root / RUNTIME_AGENT_DIRNAME / f"{label}.plist"


def write_launch_agent_plist(
    *,
    root: Path,
    label: str,
    payload: str,
    system_path: Path | None = None,
) -> dict[str, Any]:
    """Publish a workspace plist and mirror it to LaunchAgents when possible."""

    workspace_path = workspace_launch_agent_path(root, label)
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.parent.chmod(0o700)
    _atomic_text(workspace_path, payload)

    system_path = system_path or (
        Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"
    )
    system_installed = False
    system_blocker = ""
    system_install_mode = "blocked"
    try:
        system_path.parent.mkdir(parents=True, exist_ok=True)
        if system_path.is_file() and system_path.read_text(encoding="utf-8") == payload:
            system_install_mode = "already_current"
        else:
            _atomic_text(system_path, payload)
            system_install_mode = "atomic_replace"
        system_installed = True
    except OSError as exc:
        if exc.errno == errno.ENOSPC and system_path.is_file():
            backup_path = workspace_path.with_suffix(".system-backup.plist")
            _atomic_text(backup_path, system_path.read_text(encoding="utf-8"))
            try:
                _replace_existing_file_in_place(system_path, payload)
                system_installed = True
                system_install_mode = "verified_in_place_existing_allocation"
            except OSError as fallback_exc:
                system_blocker = (
                    "system_launch_agent_install_failed:"
                    f"{type(fallback_exc).__name__}:{fallback_exc.errno}"
                )
        else:
            system_blocker = (
                f"system_launch_agent_install_failed:{type(exc).__name__}:{exc.errno}"
            )
    return {
        "workspace_plist": workspace_path,
        "system_plist": system_path,
        "system_persistence_installed": system_installed,
        "system_install_mode": system_install_mode,
        "system_persistence_blocker": system_blocker,
    }


def _atomic_text(path: Path, payload: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _replace_existing_file_in_place(path: Path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    stat = path.stat()
    allocated_bytes = max(stat.st_blocks * 512, stat.st_size)
    if len(encoded) > allocated_bytes:
        raise OSError(errno.ENOSPC, "payload exceeds existing file allocation", str(path))
    with path.open("r+b", buffering=0) as handle:
        written = handle.write(encoded)
        if written != len(encoded):
            raise OSError(errno.EIO, "short in-place LaunchAgent write", str(path))
        handle.truncate(len(encoded))
        os.fsync(handle.fileno())
    if path.read_bytes() != encoded:
        raise OSError(errno.EIO, "LaunchAgent in-place verification failed", str(path))
