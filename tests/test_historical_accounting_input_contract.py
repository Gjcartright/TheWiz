from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    RebalancePolicy,
    backtest_two_leg_spread,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.performance_math import normalize_interval, resolve_annualization


def _pair() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC"),
            "price_x": [100.0, 100.0, 110.0, 110.0],
            "price_y": [50.0, 50.0, 50.0, 50.0],
            "hedge_ratio": [1.0] * 4,
        }
    )


@pytest.mark.parametrize("field", [
    "taker_fee_bps", "slippage_bps", "execution_risk_bps",
    "partial_fill_penalty_bps", "funding_bps_per_day",
    "bars_per_day", "partial_fill_probability", "partial_fill_fraction",
])
@pytest.mark.parametrize("boolean", [True, np.bool_(True)])
def test_cost_model_rejects_boolean_financial_parameters(field, boolean):
    with pytest.raises(ValueError, match=field):
        CostModel(**{field: boolean})


def test_two_leg_rejects_boolean_prices_and_funding():
    frame = _pair()
    frame["price_x"] = [True] * len(frame)
    with pytest.raises(ValueError, match="two-leg prices must be real numeric"):
        backtest_two_leg_spread(frame, pd.Series([0.0, 1.0, 1.0, 0.0]))

    frame = _pair()
    frame["funding_x_bps"] = [np.bool_(True)] * len(frame)
    with pytest.raises(ValueError, match="funding_x_bps contains non-real numeric"):
        backtest_two_leg_spread(frame, pd.Series([0.0, 1.0, 1.0, 0.0]))


@pytest.mark.parametrize("column,value", [
    ("spread_orientation", None),
    ("spread_orientation", ""),
    ("hedge_ratio_kind", None),
    ("hedge_ratio_kind", "unknown"),
])
def test_explicit_economic_metadata_must_be_complete_and_supported(column, value):
    frame = _pair()
    frame[column] = [value] * len(frame)
    with pytest.raises(ValueError, match=column):
        backtest_two_leg_spread(frame, pd.Series([0.0, 1.0, 1.0, 0.0]))


def test_fixed_units_ledger_reconciles_same_currency_pnl():
    frame = _pair()
    costs = CostModel(
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        execution_risk_bps=0.0,
        funding_bps_per_day=0.0,
        partial_fill_probability=0.0,
        rebalance_policy=RebalancePolicy.FIXED_UNITS_UNTIL_EXIT,
    )
    result, ledger = backtest_two_leg_spread_with_ledger(
        frame, pd.Series([0.0, 1.0, 1.0, 0.0]), costs
    )
    assert result.total_return == pytest.approx(-0.05)
    assert ledger.closed_trades["net_pnl"].tolist() == pytest.approx([-0.05])
    assert result.reconciliation_error < 1e-12
    assert ledger.bar_ledger["quantity_basis"].eq("initial_capital_currency_units").all()
    assert ledger.bar_ledger["spread_orientation"].eq("y_on_x").all()


def test_fixed_units_reversal_charges_both_sides_and_closes_two_trades():
    frame = _pair()
    frame["price_x"] = 100.0
    costs = CostModel(
        taker_fee_bps=10.0,
        slippage_bps=0.0,
        execution_risk_bps=0.0,
        funding_bps_per_day=0.0,
        partial_fill_probability=0.0,
        rebalance_policy=RebalancePolicy.FIXED_UNITS_UNTIL_EXIT,
    )
    result, ledger = backtest_two_leg_spread_with_ledger(
        frame, pd.Series([0.0, 1.0, -1.0, 0.0]), costs
    )
    assert ledger.bar_ledger["rebalance_event"].tolist() == [
        "hold", "entry", "reversal", "exit"
    ]
    assert result.trades == 2
    assert result.profit_factor_status == "valid"
    assert result.profit_factor == 0.0
    assert result.total_return < 0.0
    assert result.reconciliation_error < 1e-12


def test_monthly_clock_requires_consecutive_aligned_calendar_months():
    assert normalize_interval("1M") == "1M"
    assert normalize_interval("1m") == "1m"
    starts = pd.date_range("2025-01-01", periods=4, freq="MS", tz="UTC")
    ends = pd.date_range("2025-01-31", periods=4, freq="ME", tz="UTC")
    assert resolve_annualization(interval="1M", timestamps=iter(starts)).status == "valid"
    assert resolve_annualization(interval="1M", timestamps=ends).periods_per_year == 12.0
    assert resolve_annualization(timestamps=starts).interval == "1M"
    gap = starts.delete(1)
    assert resolve_annualization(interval="1M", timestamps=gap).status == "blocked"
    assert resolve_annualization(interval="1m", timestamps=starts).status == "blocked"
