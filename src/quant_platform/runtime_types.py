"""Lightweight shared runtime values for data daemons and command producers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CommandResult:
    paths: dict[str, Path]
    summary: dict[str, object]
