import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_pair,
    backtest_two_leg_spread,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.strategies import zscore_signal


def test_backtest_includes_costs_and_returns_metrics():
    frame = pd.DataFrame(
        {
            "spread": [0.0, 1.0, 2.2, 1.0, 0.1, -1.0, -2.2, -1.0, 0.0],
            "zscore": [0.0, 1.0, 2.2, 1.0, 0.1, -1.0, -2.2, -1.0, 0.0],
        }
    )
    result = backtest_pair(
        frame,
        zscore_signal(frame),
        CostModel(taker_fee_bps=1, slippage_bps=1, execution_risk_bps=1),
    )
    assert result.trades > 0
    assert result.max_drawdown >= 0
    assert result.total_return == result.total_return


def test_two_leg_backtest_accounts_for_execution_cost_components():
    frame = pd.DataFrame(
        {
            "price_x": [100, 101, 102, 101, 100, 99],
            "price_y": [50, 49, 48, 49, 50, 51],
            "spread": [0.0, -1.0, -2.0, -1.0, 0.0, 1.0],
            "hedge_ratio": [1.2] * 6,
            "beta": [0.8] * 6,
            "funding_x_bps": [2.0] * 6,
            "funding_y_bps": [3.0] * 6,
        }
    )
    signal = pd.Series([0, 1, 1, 0, -1, -1], index=frame.index)
    costs = CostModel(
        taker_fee_bps=5,
        slippage_bps=4,
        execution_risk_bps=2,
        funding_bps_per_day=1,
        partial_fill_probability=0.25,
        partial_fill_fraction=0.5,
        partial_fill_penalty_bps=3,
    )

    result = backtest_two_leg_spread(frame, signal, costs)

    assert result.trades > 0
    assert result.total_fees > 0
    assert result.total_slippage > 0
    assert result.total_funding > 0
    assert result.total_execution_risk > 0
    assert result.total_partial_fill_cost > 0
    assert result.avg_gross_exposure > 0


def test_two_leg_backtest_does_not_redailyize_realized_hourly_funding():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC"),
            "price_x": [100.0, 100.0, 100.0],
            "price_y": [50.0, 50.0, 50.0],
            "hedge_ratio": [1.0, 1.0, 1.0],
            "funding_x_realized_bps": [24.0, 24.0, 24.0],
            "funding_y_realized_bps": [0.0, 0.0, 0.0],
        }
    )
    signal = pd.Series([1.0, 1.0, 1.0], index=frame.index)
    costs = CostModel(
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        execution_risk_bps=0.0,
        funding_bps_per_day=0.0,
        partial_fill_probability=0.0,
        partial_fill_penalty_bps=0.0,
        funding_policy=FundingPolicy.SIGNED_REALIZED.value,
    )

    result, ledger = backtest_two_leg_spread_with_ledger(
        frame,
        signal,
        costs,
        interval="1h",
    )

    assert ledger.bar_ledger["funding"].tolist() == pytest.approx([0.0, -0.0012, -0.0012])
    assert result.total_funding == pytest.approx(-0.0024)


def test_two_leg_backtest_weights_leg_specific_slippage_by_hedge_exposure():
    frame = pd.DataFrame(
        {
            "price_x": [100.0, 100.0, 100.0],
            "price_y": [50.0, 50.0, 50.0],
            "hedge_ratio": [3.0, 3.0, 3.0],
            "slippage_x_model_bps": [10.0, 10.0, 10.0],
            "slippage_y_model_bps": [2.0, 2.0, 2.0],
        }
    )
    signal = pd.Series([1.0, 0.0, 0.0], index=frame.index)
    costs = CostModel(
        taker_fee_bps=0.0,
        slippage_bps=999.0,
        execution_risk_bps=0.0,
        funding_bps_per_day=0.0,
        partial_fill_probability=0.0,
        partial_fill_penalty_bps=0.0,
    )

    result, ledger = backtest_two_leg_spread_with_ledger(frame, signal, costs)

    assert ledger.bar_ledger["slippage"].tolist() == pytest.approx([0.0008, 0.0008, 0.0])
    assert result.total_slippage == pytest.approx(0.0016)
