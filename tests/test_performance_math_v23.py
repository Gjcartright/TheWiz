from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    backtest_two_leg_spread,
    max_drawdown,
)
from quant_platform.performance_math import (
    MATH_VERSION,
    calculate_annualized_sharpe,
    resolve_annualization,
)
from quant_platform.trade_ledger import build_trade_ledger


def test_annualization_uses_declared_timeframe_and_blocks_unknown():
    daily = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]), interval="daily")
    hourly = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]), interval="1h")
    unknown = calculate_annualized_sharpe(pd.Series([0.01, -0.005, 0.007]))

    assert daily.periods_per_year == 365
    assert hourly.periods_per_year == 8760
    assert unknown.status == "blocked"
    assert math.isnan(unknown.value)


def test_annualization_infers_regular_five_minute_timestamps():
    timestamps = pd.date_range("2026-01-01", periods=20, freq="5min", tz="UTC")
    result = resolve_annualization(timestamps=timestamps)
    assert result.interval == "5m"
    assert result.periods_per_year == 105120


def test_sharpe_uses_sample_standard_deviation_and_blocks_one_observation():
    values = pd.Series([0.01, -0.005, 0.007, 0.002])
    result = calculate_annualized_sharpe(values, interval="1d")
    singleton = calculate_annualized_sharpe(pd.Series([0.01]), interval="1d")

    expected = math.sqrt(365.0) * values.mean() / values.std(ddof=1)
    assert result.value == pytest.approx(expected)
    assert singleton.status == "blocked"
    assert singleton.reason == "insufficient_return_observations"


def test_sharpe_blocks_zero_variance_and_declared_timestamp_mismatch():
    constant = calculate_annualized_sharpe(pd.Series([0.01, 0.01, 0.01]), interval="1d")
    timestamps = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    mismatch = calculate_annualized_sharpe(
        pd.Series([0.01, -0.01, 0.02, -0.01]),
        interval="1d",
        timestamps=timestamps,
    )

    assert constant.status == "blocked"
    assert math.isnan(constant.value)
    assert mismatch.status == "blocked"
    assert mismatch.reason == "declared_interval_timestamp_mismatch:1d!=1h"


def test_max_drawdown_includes_initial_capital_before_first_return():
    assert max_drawdown(pd.Series([0.90, 0.80])) == pytest.approx(0.20)


def test_trade_ledger_counts_closed_trades_and_reconciles_reversal():
    target = pd.Series([0.0, 1.0, 1.0, -1.0, -1.0, 0.0])
    gross = pd.Series([0.0, 0.0, 0.01, 0.005, 0.01, 0.01])
    fees = pd.Series([0.0, 0.001, 0.0, 0.002, 0.0, 0.001])
    ledger = build_trade_ledger(target, gross, {"fees": fees})

    assert ledger.closed_trades["exit_reason"].tolist() == ["signal_reversal", "signal_exit"]
    assert ledger.open_trades.empty
    assert ledger.reconciliation_error < 1e-12


def test_backtest_reports_closed_trade_expectancy_lower_bound():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=8, freq="1D", tz="UTC"),
            "price_x": [100, 101, 102, 101, 100, 99, 100, 101],
            "price_y": [50, 49, 48, 49, 50, 51, 50, 49],
            "hedge_ratio": [1.0] * 8,
        }
    )
    signal = pd.Series([0, 1, 1, 0, 0, -1, -1, 0])
    result = backtest_two_leg_spread(frame, signal, CostModel(funding_bps_per_day=0.0))
    assert result.trades == 2
    assert result.closed_trade_return_std >= 0.0
    assert result.expectancy_lower_95 is not None


def test_open_position_does_not_count_as_closed_trade():
    ledger = build_trade_ledger(
        pd.Series([0.0, 1.0, 1.0]),
        pd.Series([0.0, 0.0, 0.01]),
        {},
    )
    assert ledger.closed_trades.empty
    assert len(ledger.open_trades) == 1


def test_two_leg_backtest_ignores_beta_and_declares_funding_policy():
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC"),
            "price_x": [100, 101, 102, 101, 100, 99],
            "price_y": [50, 49, 48, 49, 50, 51],
            "hedge_ratio": [1.2] * 6,
            "funding_x_bps": [2.0] * 6,
            "funding_y_bps": [3.0] * 6,
        }
    )
    signal = pd.Series([0, 1, 1, 0, -1, 0])
    low_beta = base.assign(beta=0.1)
    high_beta = base.assign(beta=10.0)
    model = CostModel(funding_policy=FundingPolicy.SIGNED_REALIZED.value)

    low = backtest_two_leg_spread(low_beta, signal, model)
    high = backtest_two_leg_spread(high_beta, signal, model)

    assert low.total_return == high.total_return
    assert low.funding_policy == "signed_realized"
    assert low.math_version == MATH_VERSION
    assert low.sharpe_status == "valid"
