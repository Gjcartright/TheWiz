from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.orchestration.corrective_data_evidence import (
    SOURCE_CONTRACTS,
    run_cost_evidence_attacks,
    validate_cost_bundle,
    validate_source_frame,
)


NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "BTC",
                "asset_index": 0,
                "universe_name": "BTC",
                "sz_decimals": 5,
                "max_leverage": 40,
                "is_delisted": False,
                "tradable_perp": True,
                "checked_at_utc": NOW.isoformat(),
            }
        ]
    )


def test_source_contract_fails_closed_on_schema_duplicate_and_stale_data():
    valid = _inventory()
    assert validate_source_frame(valid, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW) == []
    assert validate_source_frame(valid.drop(columns=["tradable_perp"]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)
    assert "duplicate_keys" in validate_source_frame(pd.concat([valid, valid]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)
    stale = valid.assign(checked_at_utc=(NOW - timedelta(days=3)).isoformat())
    assert "source_stale" in validate_source_frame(stale, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)


def test_cost_contract_rejects_stale_sparse_sign_and_stress_manipulation(tmp_path):
    base = {"captured_at": NOW.isoformat(), "samples": 20, "funding_coverage": 0.99, "base_slippage_bps": 2.0, "stress_slippage_bps": 4.0}
    assert validate_cost_bundle(base, now=NOW) == []
    assert validate_cost_bundle({**base, "samples": 1}, now=NOW)
    assert validate_cost_bundle({**base, "base_slippage_bps": -2.0}, now=NOW)
    assert validate_cost_bundle({**base, "stress_slippage_bps": 1.0}, now=NOW)
    result = run_cost_evidence_attacks(root=tmp_path, now=NOW)
    assert result["status"].eq("PASS").all()
    assert not result["manipulated_evidence_improves_readiness"].any()
