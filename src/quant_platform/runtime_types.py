"""Lightweight shared runtime values for data daemons and command producers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

TRUE_TOKENS = frozenset({"true", "1", "yes", "y"})


def strict_bool(value: Any) -> bool:
    """Parse persisted gate values without treating a nonempty string as true."""

    if value is None:
        return False
    if isinstance(value, Real):
        return float(value) == 1.0
    return str(value).strip().lower() in TRUE_TOKENS


@dataclass(frozen=True)
class CommandResult:
    paths: dict[str, Path]
    summary: dict[str, object]
