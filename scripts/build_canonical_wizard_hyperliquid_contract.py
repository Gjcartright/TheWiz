#!/usr/bin/env python3
"""Write the shared current/exhaustive Wizard-Hyperliquid contract."""

from __future__ import annotations

import json
from pathlib import Path

from quant_platform.orchestration.canonical_wizard_hyperliquid_contract import (
    build_canonical_pipeline_contract,
)

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    result = build_canonical_pipeline_contract(root=ROOT)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
