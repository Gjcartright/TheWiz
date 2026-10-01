from __future__ import annotations

import os
import stat
from pathlib import Path


def _check_env_file_provenance(file_stat: os.stat_result) -> None:
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError("env_file_symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("env_file_not_regular")
    if file_stat.st_uid != os.getuid():
        raise ValueError("env_file_not_owned_by_user")
    if file_stat.st_mode & 0o077:
        raise ValueError("env_file_permissions")


def _read_owned_env_text(path: Path) -> str | None:
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    _check_env_file_provenance(before)

    # Validate the opened inode too, so a replacement between lstat and open
    # cannot redirect credential reads to another file.
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ValueError("env_file_changed_before_read") from exc
    try:
        opened = os.fstat(fd)
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise ValueError("env_file_changed_before_read")
        _check_env_file_provenance(opened)
        with os.fdopen(fd, "r", encoding="utf-8") as env_file:
            fd = -1
            return env_file.read()
    finally:
        if fd != -1:
            os.close(fd)


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load simple KEY=value pairs from a local env file."""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    content = _read_owned_env_text(env_path)
    if content is None:
        return loaded

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def load_selected_env_keys(
    path: str | Path,
    *,
    allowed_keys: set[str] | frozenset[str],
    override: bool = False,
) -> dict[str, str]:
    """Load only explicitly authorized keys from a local env file."""

    from quant_platform.orchestration.corrective_effect_guard import (
        assert_credential_access_allowed,
    )

    if not allowed_keys or any(not key.strip() for key in allowed_keys):
        raise ValueError("allowed env keys must be non-empty names")
    normalized_allowed_keys = {key.strip().upper() for key in allowed_keys}
    for key in sorted(normalized_allowed_keys):
        assert_credential_access_allowed(source=key)
    env_path = Path(path)
    loaded: dict[str, str] = {}
    content = _read_owned_env_text(env_path)
    if content is None:
        return loaded
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.upper() not in normalized_allowed_keys:
            continue
        value = _strip_quotes(value.strip())
        if override or key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
