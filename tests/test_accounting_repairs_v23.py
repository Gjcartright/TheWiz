"""Ordinary isolated accounting regressions: no execution or gate authority."""

import math
import sys

import numpy as np
import pandas as pd
import pytest

from quant_platform.backtest import (
    CostModel,
    FundingPolicy,
    _backtest_result,
    backtest_pair,
    backtest_two_leg_spread_with_ledger,
)
from quant_platform.protective_exits import (
    ProtectiveExitEngine,
    ProtectiveExitPolicy,
    ProtectiveLevels,
    protective_levels,
)
from quant_platform.trade_ledger import build_trade_ledger


def _costs(**overrides):
    return CostModel(**{
        "taker_fee_bps": 0.0, "slippage_bps": 0.0,
        "execution_risk_bps": 0.0, "funding_bps_per_day": 0.0,
        "partial_fill_probability": 0.0, "partial_fill_penalty_bps": 0.0,
        **overrides,
    })


def _prices():
    return pd.DataFrame({"price_x": [100.0] * 3, "price_y": [50.0] * 3,
                         "hedge_ratio": [1.0] * 3})


@pytest.mark.parametrize("funding", [0.01, -0.01])
@pytest.mark.parametrize("close_fraction", [0.0, 1.0 / 3.0, 1.0])
def test_holding_funding_never_moves_to_the_new_reversal_trade(funding, close_fraction):
    ledger = build_trade_ledger(
        pd.Series([1.0, -2.0, 0.0]), pd.Series([0.0, 0.0075, 0.0]),
        {"funding": pd.Series([0.0, funding, 0.0])},
        reversal_close_fraction=pd.Series([close_fraction] * 3),
    )
    assert ledger.closed_trades["total_funding"].tolist() == pytest.approx([funding, 0.0])
    assert ledger.closed_trades["profit_after_cost"].tolist() == pytest.approx([0.0075 - funding, 0.0])
    assert ledger.closed_trades["net_pnl"].sum() == pytest.approx(0.0075 - funding)
    assert ledger.reconciliation_error < 1e-12


def test_reversal_cost_split_and_currency_pnl_reconcile_independent_cash_arithmetic():
    ledger = build_trade_ledger(
        pd.Series([1.0, -2.0, 0.0]), pd.Series([0.0, 0.0075, 0.0]),
        {"fees": pd.Series([0.001, 0.003, 0.002]),
         "funding": pd.Series([0.0, 0.01, 0.0])},
        reversal_close_fraction=pd.Series([0.5, 1.0 / 3.0, 0.5]),
    )
    old_end = 0.999 * (1.0 + 0.0075 - 0.01 - 0.001)
    portfolio_end = 0.999 * (1.0 + 0.0075 - 0.01 - 0.003) * 0.998
    closed = ledger.closed_trades
    assert closed["entry_equity"].tolist() == pytest.approx([1.0, old_end])
    assert closed["ending_equity"].tolist() == pytest.approx([old_end, portfolio_end])
    assert closed["net_pnl"].tolist() == pytest.approx([old_end - 1.0, portfolio_end - old_end])
    assert closed["total_fees"].tolist() == pytest.approx([0.002, 0.004])
    assert closed["total_funding"].tolist() == pytest.approx([0.01, 0.0])
    assert closed["net_pnl"].sum() == pytest.approx(portfolio_end - 1.0)
    assert ledger.bar_ledger["net_pnl"].sum() == pytest.approx(portfolio_end - 1.0)
    assert np.prod(1.0 + closed["profit_after_cost"]) == pytest.approx(portfolio_end)
    assert ledger.reconciliation_error < 1e-12


def test_profit_factor_uses_capital_basis_but_retains_trade_normalized_returns():
    gross = pd.Series([0.0, 1.0, 0.0, -0.5])
    ledger = build_trade_ledger(pd.Series([1.0, 0.0, 1.0, 0.0]), gross, {})
    assert ledger.closed_trades["entry_equity"].tolist() == [1.0, 2.0]
    assert ledger.closed_trades["ending_equity"].tolist() == [2.0, 1.0]
    assert ledger.closed_trades["net_pnl"].tolist() == [1.0, -1.0]
    assert ledger.closed_trades["profit_after_cost"].tolist() == [1.0, -0.5]
    result = _backtest_result(ledger, gross_return=gross,
                              gross_exposure=pd.Series([0.0, 1.0, 0.0, 1.0]),
                              costs=_costs(), interval="1d", timestamps=None)
    assert result.profit_factor == 1.0
    assert result.profit_factor_status == "valid"
    assert result.expectancy == 0.25  # Still an explicitly trade-normalized statistic.
    assert result.profit_factor_basis == "closed_trade_net_pnl_in_initial_capital_currency_units"


def test_open_trade_pnl_is_reconciled_but_not_in_closed_trade_profit_factor():
    gross = pd.Series([0.0, 0.2, 0.0, -0.1])
    ledger = build_trade_ledger(pd.Series([1.0, 0.0, -1.0, -1.0]), gross, {})
    result = _backtest_result(ledger, gross_return=gross,
                              gross_exposure=pd.Series([0.0, 1.0, 0.0, 1.0]),
                              costs=_costs(), interval="1d", timestamps=None)
    assert result.trades == 1 and result.open_trades == 1
    assert result.profit_factor_status == "unavailable"
    assert result.profit_factor_reason == "no_loss_observations"
    assert math.isnan(result.profit_factor)
    assert ledger.closed_trades["net_pnl"].sum() + ledger.open_trades["net_pnl"].sum() == pytest.approx(0.08)


def test_first_active_bar_and_final_open_position_have_correct_capital_basis():
    ledger = build_trade_ledger(pd.Series([0.0, 1.0, 1.0]),
                                pd.Series([0.0, 0.0, 0.1]),
                                {"fees": pd.Series([0.0, 0.01, 0.0])})
    assert ledger.closed_trades.empty
    trade = ledger.open_trades.iloc[0]
    assert trade.entry_equity == 1.0
    assert trade.ending_equity == pytest.approx(0.99 * 1.1)
    assert trade.net_pnl == pytest.approx(0.089)
    assert ledger.bar_ledger["equity_before"].tolist() == pytest.approx([1.0, 1.0, 0.99])


def test_realized_fee_rebates_remain_signed_in_ledger():
    ledger = build_trade_ledger(pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0]),
                                {"fees": pd.Series([-0.001, -0.001])})
    assert ledger.closed_trades.iloc[0].total_fees == -0.002
    assert ledger.closed_trades.iloc[0].net_pnl == pytest.approx(1.001 ** 2 - 1.0)


@pytest.mark.parametrize("leg", ["slippage_x_model_bps", "slippage_y_model_bps"])
@pytest.mark.parametrize("invalid", [-1.0, np.nan, np.inf, -np.inf])
def test_each_modeled_slippage_leg_requires_nonnegative_finite_cost(leg, invalid):
    frame = _prices().assign(slippage_x_model_bps=0.0, slippage_y_model_bps=0.0)
    frame.loc[1, leg] = invalid
    with pytest.raises(ValueError):
        backtest_two_leg_spread_with_ledger(frame, pd.Series([1.0, 0.0, 0.0]), _costs(), interval="1h")


@pytest.mark.parametrize("component", ["slippage", "execution_risk", "partial_fill"])
def test_direct_ledger_rejects_negative_adverse_costs(component):
    with pytest.raises(ValueError, match="nonnegative"):
        build_trade_ledger(pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0]),
                           {component: pd.Series([-0.001, 0.0])})


@pytest.mark.parametrize("field", ["target", "gross", "fees", "funding", "fraction"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_ledger_does_not_silently_replace_missing_or_nonfinite_inputs(field, invalid):
    target, gross = pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0])
    costs = {"fees": pd.Series([0.0, 0.0]), "funding": pd.Series([0.0, 0.0])}
    fraction = pd.Series([0.5, 0.5])
    values = {"target": target, "gross": gross, **costs, "fraction": fraction}
    values[field].iloc[1] = invalid
    with pytest.raises(ValueError, match="complete and finite"):
        build_trade_ledger(target, gross, costs, reversal_close_fraction=fraction)


def test_unknown_cost_is_not_silently_ignored():
    with pytest.raises(ValueError, match="unknown cost"):
        build_trade_ledger(pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0]),
                           {"fundng": pd.Series([0.0, 0.01])})


@pytest.mark.parametrize("component", ["gross", "funding", "fees"])
def test_flat_rows_cannot_lose_unowned_economic_activity(component):
    gross = pd.Series([0.0, 0.0])
    costs = {}
    if component == "gross":
        gross.iloc[0] = 0.01
    else:
        costs[component] = pd.Series([0.01, 0.0])
    with pytest.raises(ValueError):
        build_trade_ledger(pd.Series([0.0, 0.0]), gross, costs)


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_invalid_reversal_fractions_are_rejected_not_clipped(fraction):
    with pytest.raises(ValueError, match="within"):
        build_trade_ledger(pd.Series([1.0, -1.0]), pd.Series([0.0, 0.0]), {},
                           reversal_close_fraction=pd.Series([fraction, fraction]))


@pytest.mark.parametrize("source", ["gross", "cost", "fraction"])
def test_supplied_series_must_have_identical_index_order(source):
    target, gross = pd.Series([1.0, 0.0]), pd.Series([0.0, 0.0])
    costs = {"fees": pd.Series([0.0, 0.0])}
    fraction = pd.Series([0.5, 0.5])
    {"gross": gross, "cost": costs["fees"], "fraction": fraction}[source].index = [1, 0]
    with pytest.raises(ValueError, match="identical ledger index"):
        build_trade_ledger(target, gross, costs, reversal_close_fraction=fraction)


def test_duplicate_index_is_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_trade_ledger(pd.Series([1.0, 0.0], index=[0, 0]),
                           pd.Series([0.0, 0.0], index=[0, 0]), {})


@pytest.mark.parametrize("order", [[1, 0, 2], [2, 1, 0], [0, 0, 1]])
def test_datetime_index_itself_is_a_chronological_ledger_contract(order):
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")[order]
    with pytest.raises(ValueError):
        build_trade_ledger(pd.Series([1.0, 1.0, 0.0], index=index),
                           pd.Series([0.0, 0.0, 0.0], index=index), {})


def test_explicit_timestamps_cannot_relabel_a_datetime_index():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    with pytest.raises(ValueError, match="match the datetime ledger index"):
        build_trade_ledger(pd.Series([1.0, 1.0, 0.0], index=index),
                           pd.Series([0.0, 0.0, 0.0], index=index), {},
                           timestamps=index + pd.Timedelta(hours=1))


def test_valid_datetime_index_and_matching_timestamp_series_preserve_order():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    ledger = build_trade_ledger(pd.Series([1.0, 1.0, 0.0], index=index),
                                pd.Series([0.0, 0.01, 0.0], index=index), {},
                                timestamps=pd.Series(index, index=index))
    assert ledger.closed_trades.iloc[0].net_pnl == pytest.approx(0.01)


def test_compounding_overflow_is_explicitly_rejected():
    with pytest.raises(ValueError, match="compounded equity"):
        build_trade_ledger(pd.Series([1.0, 1.0, 0.0]),
                           pd.Series([0.0, 1e308, 1e308]), {})


@pytest.mark.parametrize("interval,freq,divisor", [("1d", "1D", 1.0), ("1h", "1h", 24.0)])
def test_funding_clock_uses_full_declared_grid_not_dummy_return_length(interval, freq, divisor):
    frame = _prices().assign(timestamp=pd.date_range("2026-01-01", periods=3, freq=freq, tz="UTC"))
    _, ledger = backtest_two_leg_spread_with_ledger(
        frame, pd.Series([1.0, 1.0, 0.0]), _costs(funding_bps_per_day=24.0), interval=interval)
    assert ledger.bar_ledger["funding_divisor"].tolist() == [divisor] * 3
    assert ledger.bar_ledger["funding"].tolist() == pytest.approx([0.0, 0.0024 / divisor, 0.0024 / divisor])


@pytest.mark.parametrize("stamps,interval", [
    (["2026-01-01", "2026-01-02", "2026-01-03"], "1h"),
    (["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"], "1h"),
    (["2026-01-01", "2026-01-01", "2026-01-02"], "1d"),
    (["2026-01-01", None, "2026-01-03"], "1d"),
    (None, "unknown"),
])
def test_bad_declared_funding_clock_never_falls_back_to_24(stamps, interval):
    frame = _prices()
    if stamps is not None:
        frame["timestamp"] = stamps
    with pytest.raises(ValueError, match="funding clock"):
        backtest_two_leg_spread_with_ledger(frame, pd.Series([1.0, 1.0, 0.0]), _costs(), interval=interval)


def test_absent_clock_retains_explicit_cost_divisor_without_claiming_sharpe():
    result, ledger = backtest_two_leg_spread_with_ledger(
        _prices(), pd.Series([1.0, 1.0, 0.0]), _costs(bars_per_day=8, funding_bps_per_day=8.0))
    assert ledger.bar_ledger["funding_divisor"].tolist() == [8.0] * 3
    assert result.sharpe_status.startswith("blocked:")


def test_signed_realized_funding_credit_survives_reversal_attribution():
    frame = _prices().assign(funding_x_realized_bps=20.0, funding_y_realized_bps=0.0)
    _, ledger = backtest_two_leg_spread_with_ledger(
        frame, pd.Series([1.0, -1.0, 0.0]),
        _costs(funding_policy=FundingPolicy.SIGNED_REALIZED.value), interval="1h")
    assert ledger.closed_trades["total_funding"].tolist() == pytest.approx([-0.001, 0.001])
    assert ledger.closed_trades["profit_after_cost"].tolist() == pytest.approx([0.001, -0.001])


@pytest.mark.parametrize("invalid", [0, -1, 0.5, np.nan, np.inf, True])
def test_cost_model_divisor_requires_positive_finite_integer(invalid):
    with pytest.raises(ValueError):
        _costs(bars_per_day=invalid)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.nan, np.inf])
def test_explicit_funding_divisor_does_not_use_truthiness_fallback(invalid):
    with pytest.raises(ValueError):
        _costs().funding_per_bar(bars_per_day=invalid)


@pytest.mark.parametrize("backtester", [backtest_pair, backtest_two_leg_spread_with_ledger])
def test_missing_signal_is_not_flat(backtester):
    with pytest.raises(ValueError, match="signal"):
        backtester(_prices().assign(spread=0.0), pd.Series([1.0, np.nan, 0.0]), _costs())


def test_missing_spread_is_not_a_free_holding_bar():
    with pytest.raises(ValueError, match="spread"):
        backtest_pair(pd.DataFrame({"spread": [0.0, np.nan, 0.1]}),
                      pd.Series([1.0, 1.0, 0.0]), _costs())


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("entry", [sys.float_info.max, math.ulp(0.0)])
def test_rejected_protective_domain_does_not_mutate_engine(side, entry):
    engine = ProtectiveExitEngine(ProtectiveExitPolicy(0.5, 0.5))
    before = engine.state
    with pytest.raises(ValueError):
        engine.enter(side=side, entry_price=entry, bar_index=0)
    assert engine.state == before
    assert engine.can_enter(bar_index=0)


@pytest.mark.parametrize("side", ["long", "short"])
def test_rounded_away_protective_distance_is_explicitly_unrepresentable(side):
    with pytest.raises(ValueError, match="not representable"):
        protective_levels(side=side, entry_price=100.0,
                          policy=ProtectiveExitPolicy(1e-300, 1e-300))


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("entry", [1e-300, sys.float_info.max / 4.0])
def test_representable_extreme_levels_remain_supported(side, entry):
    levels = protective_levels(side=side, entry_price=entry,
                               policy=ProtectiveExitPolicy(0.5, 0.5))
    assert all(math.isfinite(value) and value > 0.0 for value in (levels.stop_loss, levels.take_profit))
    assert min(levels.stop_loss, levels.take_profit) < entry < max(levels.stop_loss, levels.take_profit)


@pytest.mark.parametrize("invalid", [0.0, -1.0, np.inf, np.nan])
def test_direct_protective_levels_cannot_hold_invalid_prices(invalid):
    with pytest.raises(ValueError):
        ProtectiveLevels(stop_loss=invalid, take_profit=110.0)
