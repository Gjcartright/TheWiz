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

RPT006_SPREAD_COLUMN = "spread"
RPT006_HEDGE_RATIO_COLUMN = "hedge_ratio"


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


def test_rpt006_entry_exit_level_crossing_truth_table():
    canonical_node_ids = ("node.canonical_net_return_ledger",)
    canonical_gate_ids: tuple[str, ...] = ()
    assert canonical_node_ids == ("node.canonical_net_return_ledger",)
    assert canonical_gate_ids == ()

    cases = [
        "flat",
        "entry_touch_equality",
        "entry_cross",
        "short_remain",
        "short_recede_remain",
        "exit_touch_equality",
        "exit_cross",
        "long_entry_jump",
        "long_entry_equality_remain",
        "long_exit_touch_equality",
        "long_exit_cross",
        "short_entry_jump",
        "opposite_recross_close",
        "opposite_recross_open",
        "long_exit_jump",
        "entry_touch_equality_flat",
    ]
    zscore = pd.Series(
        [0.0, 2.0, 2.1, 2.4, 2.0, 0.25, 0.20, -2.5, -2.0, -0.25, 0.0, 2.5, -2.5, -2.5, 0.0, -2.0]
    )
    frame = pd.DataFrame({RPT006_SPREAD_COLUMN: zscore, "zscore": zscore, "case": cases})
    signal = zscore_signal(frame, entry=2.0, exit_=0.25)
    assert signal.tolist() == [
        0.0,
        0.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        -1.0,
        0.0,
        1.0,
        0.0,
        0.0,
    ]

    events = []
    prior = 0.0
    event_types = {
        (0.0, 0.0): "flat_hold",
        (0.0, 1.0): "enter_long_signal",
        (0.0, -1.0): "enter_short_signal",
        (1.0, 1.0): "long_hold",
        (-1.0, -1.0): "short_hold",
        (1.0, 0.0): "exit_long_signal",
        (-1.0, 0.0): "exit_short_signal",
    }
    for index, new in enumerate(signal):
        events.append(
            {
                "row": index,
                "event_type": event_types[(prior, float(new))],
                "prior_state": prior,
                "new_state": float(new),
                "entry_equality": abs(float(zscore.iloc[index])) == 2.0,
                "exit_equality": abs(float(zscore.iloc[index])) == 0.25,
                "fill_timing": "next_bar",
            }
        )
        prior = float(new)
    assert len(events) == len(frame)
    assert all(event["event_type"] for event in events)
    assert events[1]["entry_equality"] and events[1]["event_type"] == "flat_hold"
    assert events[5]["exit_equality"] and events[5]["event_type"] == "short_hold"
    assert events[12]["event_type"] == "exit_short_signal"
    assert events[13]["event_type"] == "enter_long_signal"

    execution_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(frame), freq="1h", tz="UTC"),
            "price_x": 100.0 + pd.Series(range(len(frame)), dtype=float),
            "price_y": 50.0 + pd.Series(range(len(frame)), dtype=float) * 0.5,
            RPT006_HEDGE_RATIO_COLUMN: 1.0,
        }
    )
    _, ledger = backtest_two_leg_spread_with_ledger(
        execution_frame,
        signal,
        CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            execution_risk_bps=0.0,
            funding_bps_per_day=0.0,
            partial_fill_probability=0.0,
            partial_fill_penalty_bps=0.0,
        ),
        interval="1h",
    )
    bars = ledger.bar_ledger
    pd.testing.assert_series_equal(
        bars["held_weight_x"].iloc[1:].reset_index(drop=True),
        bars["target_weight_x"].iloc[:-1].reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        bars["held_weight_y"].iloc[1:].reset_index(drop=True),
        bars["target_weight_y"].iloc[:-1].reset_index(drop=True),
        check_names=False,
    )
    transitions = signal.ne(signal.shift(1, fill_value=0.0))
    assert (
        (
            bars.loc[transitions, ["held_weight_x", "held_weight_y"]].to_numpy()
            != bars.loc[transitions, ["target_weight_x", "target_weight_y"]].to_numpy()
        )
        .any(axis=1)
        .all()
    )
    assert all(event["fill_timing"] == "next_bar" for event in events)
