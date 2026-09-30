from __future__ import annotations

import math

import pandas as pd

from quant_platform.backtest import CostModel, backtest_pair


def _zero_costs() -> CostModel:
    return CostModel(
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        execution_risk_bps=0.0,
        funding_bps_per_day=0.0,
        partial_fill_probability=0.0,
        partial_fill_penalty_bps=0.0,
    )


def test_backtest_does_not_encode_no_loss_sample_as_infinite_profit_factor() -> None:
    frame = pd.DataFrame({"spread": [0.0, 0.0, 1.0, 1.0]})
    signal = pd.Series([0.0, 1.0, 1.0, 0.0])

    result = backtest_pair(frame, signal, _zero_costs())

    assert result.trades == 1
    assert math.isnan(result.profit_factor)
    assert result.profit_factor_status == "unavailable"
    assert result.profit_factor_reason == "no_loss_observations"


def test_backtest_preserves_valid_zero_profit_factor_for_all_loss_sample() -> None:
    frame = pd.DataFrame({"spread": [0.0, 0.0, -0.5, -0.5]})
    signal = pd.Series([0.0, 1.0, 1.0, 0.0])

    result = backtest_pair(frame, signal, _zero_costs())

    assert result.trades == 1
    assert result.profit_factor == 0.0
    assert result.profit_factor_status == "valid"
    assert result.profit_factor_reason == ""
