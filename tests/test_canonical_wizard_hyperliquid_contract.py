from __future__ import annotations

import json

import pandas as pd

from quant_platform.orchestration.canonical_wizard_hyperliquid_contract import (
    CAPACITY_NOTIONALS_USD,
    MINIMUM_STRICT_L2_SAMPLES,
    build_canonical_pipeline_contract,
)


def test_canonical_contract_freezes_modes_costs_statistics_and_authority(tmp_path):
    result = build_canonical_pipeline_contract(root=tmp_path)
    payload = json.loads(result["contract"].read_text(encoding="utf-8"))
    stages = pd.read_csv(result["stages"])

    assert result["exact_mode_cells_per_pair"] == 14
    assert payload["cost_policy"]["minimum_strict_l2_samples"] == MINIMUM_STRICT_L2_SAMPLES
    assert payload["cost_policy"]["capacity_notionals_usd"] == list(CAPACITY_NOTIONALS_USD)
    assert payload["statistical_policy"]["fold_count_is_not_deflated_sharpe"] is True
    assert payload["authority"]["wizard_promotion_authority"] is False
    assert payload["authority"]["live_trading_authorized"] is False
    assert stages.iloc[-1]["authority"] == "disabled"
