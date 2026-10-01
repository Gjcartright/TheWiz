from __future__ import annotations

import pandas as pd

from quant_platform.wizard_candidate_set import wizard_hyperliquid_candidate_set_id


def test_candidate_set_id_is_order_independent_and_changes_with_membership():
    rows = pd.DataFrame(
        [
            {"pair": "BTC/EIGEN", "candidate_config_hash": "a" * 64},
            {"pair": "DOGE/EIGEN", "candidate_config_hash": "b" * 64},
        ]
    )

    first = wizard_hyperliquid_candidate_set_id(rows)
    reordered = wizard_hyperliquid_candidate_set_id(rows.iloc[::-1])
    smaller = wizard_hyperliquid_candidate_set_id(rows.iloc[:1])

    assert first.startswith("whlset_")
    assert first == reordered
    assert first != smaller


def test_candidate_set_identity_distinguishes_monthly_from_minute_timeframes():
    minute = pd.DataFrame([{"pair": "BTC/ETH", "local_interval": "1m", "exact_mode": "Copula"}])
    monthly = pd.DataFrame([{"pair": "BTC/ETH", "local_interval": "1M", "exact_mode": "Copula"}])

    assert wizard_hyperliquid_candidate_set_id(minute) != wizard_hyperliquid_candidate_set_id(monthly)
