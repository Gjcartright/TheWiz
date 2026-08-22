#!/usr/bin/env python3
"""Build source health and point-in-time lineage from raw captures."""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.evidence_registry import build_evidence_registry

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = build_evidence_registry(root=ROOT)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
