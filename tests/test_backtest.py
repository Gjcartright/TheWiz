import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_pair,
    backtest_two_leg_spread,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.strategies import copula_signal, zscore_signal


def test_zscore_signal_holds_until_mean_exit_and_flattens_before_reversal():
    frame = pd.DataFrame(
        {
            "spread": [0.0, 2.2, 1.0, 0.1, 0.0, -2.2, -1.0, -0.1, 2.3, -2.3, -2.4, 0.0],
            "zscore": [0.0, 2.2, 1.0, 0.1, 0.0, -2.2, -1.0, -0.1, 2.3, -2.3, -2.4, 0.0],
        }
    )

    signal = zscore_signal(frame)

    assert signal.tolist() == [0.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0]


def test_copula_signal_uses_distinct_entry_and_neutral_exit_bands():
    frame = pd.DataFrame(
        {
            "conditional_probability_distortion": [
                0.0,
                0.25,
                0.10,
                0.04,
                0.0,
                -0.25,
                -0.10,
                -0.04,
            ]
        }
    )

    signal = copula_signal(frame)

    assert signal.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0]


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


def test_two_leg_backtest_fails_closed_on_missing_hedge_ratio_or_bad_prices():
    frame = pd.DataFrame(
        {
            "price_x": [100.0, 101.0, 102.0],
            "price_y": [50.0, 51.0, 52.0],
        }
    )
    signal = pd.Series([0.0, 1.0, 0.0])
    with pytest.raises(ValueError, match="hedge_ratio"):
        backtest_two_leg_spread(frame, signal)

    with_hedge = frame.assign(hedge_ratio=1.0)
    with_hedge.loc[1, "price_x"] = float("nan")
    with pytest.raises(ValueError, match="complete, finite, and positive"):
        backtest_two_leg_spread(with_hedge, signal)


def test_cost_model_rejects_impossible_parameters():
    with pytest.raises(ValueError, match="slippage_bps"):
        CostModel(slippage_bps=-1.0)
    with pytest.raises(ValueError, match="partial_fill_probability"):
        CostModel(partial_fill_probability=1.1)
    for invalid in (True, 1.5, float("nan")):
        with pytest.raises(ValueError, match="bars_per_day"):
            CostModel(bars_per_day=invalid)
    with pytest.raises(ValueError, match="bars_per_day"):
        CostModel().funding_per_bar(bars_per_day=float("nan"))


@pytest.mark.parametrize("runner", [backtest_pair, backtest_two_leg_spread])
def test_backtest_rejects_unaligned_missing_or_nonnumeric_signal(runner):
    frame = pd.DataFrame(
        {
            "spread": [0.0, 0.02, 0.04],
            "price_x": [100.0, 101.0, 102.0],
            "price_y": [50.0, 50.5, 51.0],
            "hedge_ratio": [1.0, 1.0, 1.0],
        }
    )
    invalid_signals = (
        pd.Series([1.0, 0.0], index=[0, 2]),
        pd.Series([True, True, False], index=frame.index),
        pd.Series([1.0, float("nan"), 0.0], index=frame.index),
        pd.Series([1.0 + 0.0j, 1.0 + 0.0j, 0.0j], index=frame.index),
    )
    for signal in invalid_signals:
        with pytest.raises(ValueError, match="signal"):
            runner(frame, signal)


def test_two_leg_backtest_rejects_negative_modeled_slippage():
    frame = pd.DataFrame(
        {
            "price_x": [100.0, 101.0, 102.0],
            "price_y": [50.0, 50.5, 51.0],
            "hedge_ratio": [1.0, 1.0, 1.0],
            "slippage_x_model_bps": [-10.0, -10.0, -10.0],
            "slippage_y_model_bps": [0.0, 0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="modeled leg slippage"):
        backtest_two_leg_spread(frame, pd.Series([1.0, 1.0, 0.0], index=frame.index))
