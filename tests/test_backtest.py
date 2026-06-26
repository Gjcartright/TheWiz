import pandas as pd

from quant_platform.backtest import CostModel, backtest_pair, backtest_two_leg_spread
from quant_platform.strategies import zscore_signal


def test_backtest_includes_costs_and_returns_metrics():
    frame = pd.DataFrame(
        {
            "spread": [0.0, 1.0, 2.2, 1.0, 0.1, -1.0, -2.2, -1.0, 0.0],
            "zscore": [0.0, 1.0, 2.2, 1.0, 0.1, -1.0, -2.2, -1.0, 0.0],
        }
    )
    result = backtest_pair(frame, zscore_signal(frame), CostModel(taker_fee_bps=1, slippage_bps=1, execution_risk_bps=1))
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
