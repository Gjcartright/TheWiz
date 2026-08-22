from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path, override: bool = False) -> dict[str, str]:
    """Load simple KEY=value pairs from a local env file."""
    env_path = Path(path)
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
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
