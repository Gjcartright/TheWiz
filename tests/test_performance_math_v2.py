from __future__ import annotations

import math

import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    RebalancePolicy,
    backtest_two_leg_spread,
    backtest_two_leg_spread_with_ledger,
    max_drawdown,
)
from quant_platform.performance_math import (
    MATH_VERSION,
    calculate_annualized_sharpe,
    normalize_interval,
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


@pytest.mark.parametrize(
    ("interval", "expected_periods"),
    [
        ("3m", 175_200.0),
        ("30m", 17_520.0),
        ("2h", 4_380.0),
        ("4h", 2_190.0),
        ("8h", 1_095.0),
        ("12h", 730.0),
        ("3d", 365.0 / 3.0),
        ("1w", 365.0 / 7.0),
        ("1M", 12.0),
    ],
)
def test_annualization_supports_declared_hyperliquid_timeframes(interval, expected_periods):
    result = resolve_annualization(interval=interval)

    assert result.status == "valid"
    assert result.periods_per_year == pytest.approx(expected_periods)


def test_monthly_and_minute_labels_remain_distinct():
    assert normalize_interval("1M") == "1M"
    assert normalize_interval("1month") == "1M"
    assert normalize_interval("1m") == "1m"
    monthly = calculate_annualized_sharpe(
        pd.Series([0.01, -0.005, 0.007]), interval="1M"
    )
    minute = calculate_annualized_sharpe(
        pd.Series([0.01, -0.005, 0.007]), interval="1m"
    )

    assert monthly.periods_per_year == 12.0
    assert minute.periods_per_year == 525_600.0


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


def test_two_leg_backtest_fixed_units_holds_quantities_and_records_events():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC"),
            "price_x": [100.0, 100.0, 110.0, 120.0, 120.0],
            "price_y": [50.0, 50.0, 50.0, 50.0, 50.0],
            "hedge_ratio": [1.0, 1.0, 1.4, 1.8, 1.8],
            "hedge_ratio_kind": ["quantity_units"] * 5,
        }
    )
    signal = pd.Series([0.0, 1.0, 1.0, 1.0, 0.0])
    fixed_model = CostModel(
        funding_bps_per_day=0.0,
        execution_risk_bps=0.0,
        partial_fill_probability=0.0,
        rebalance_policy=RebalancePolicy.FIXED_UNITS_UNTIL_EXIT.value,
    )
    target_model = CostModel(
        funding_bps_per_day=0.0,
        execution_risk_bps=0.0,
        partial_fill_probability=0.0,
        rebalance_policy=RebalancePolicy.TARGET_WEIGHTS_EVERY_BAR.value,
    )

    fixed, ledger = backtest_two_leg_spread_with_ledger(frame, signal, fixed_model)
    target, target_ledger = backtest_two_leg_spread_with_ledger(frame, signal, target_model)

    bars = ledger.bar_ledger
    assert fixed.rebalance_policy == RebalancePolicy.FIXED_UNITS_UNTIL_EXIT.value
    assert bars["rebalance_event"].tolist() == ["hold", "entry", "hold", "hold", "exit"]
    assert bars.loc[1:3, "quantity_x_after_rebalance"].nunique() == 1
    assert bars.loc[1:3, "quantity_y_after_rebalance"].nunique() == 1
    assert bars.loc[2:3, ["turnover_x", "turnover_y"]].to_numpy().sum() == pytest.approx(0.0)
    assert target_ledger.bar_ledger.loc[
        2:3, ["turnover_x", "turnover_y"]
    ].to_numpy().sum() > 0.0
    assert "target_update" in set(target_ledger.bar_ledger["rebalance_event"])
    assert fixed.total_fees != pytest.approx(target.total_fees)
    assert fixed.reconciliation_error < 1e-12


def test_two_leg_backtest_still_rejects_unknown_rebalance_policy():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "price_x": [100.0, 101.0, 102.0, 103.0],
            "price_y": [50.0, 50.5, 50.0, 51.0],
            "hedge_ratio": [1.0] * 4,
        }
    )

    with pytest.raises(ValueError, match="unsupported rebalance policy"):
        backtest_two_leg_spread(
            frame,
            pd.Series([0.0, 1.0, 1.0, 0.0]),
            CostModel(rebalance_policy="unknown_policy"),
        )
