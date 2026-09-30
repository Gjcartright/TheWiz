from __future__ import annotations

import math

import pytest

from quant_platform.protective_exits import (
    AmbiguousIntrabarPathError,
    EntryNotAllowedError,
    ExitReason,
    ExitTriggerBasis,
    OHLCBar,
    PositionSide,
    ProtectiveExitEngine,
    ProtectiveExitPolicy,
    SameBarCollisionPolicy,
    protective_levels,
)


def _policy(**overrides: object) -> ProtectiveExitPolicy:
    values = {
        "stop_loss_fraction": 0.05,
        "take_profit_fraction": 0.10,
        **overrides,
    }
    return ProtectiveExitPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("side", "expected_stop", "expected_profit"),
    [
        (PositionSide.LONG, 95.0, 110.0),
        (PositionSide.SHORT, 105.0, 90.0),
    ],
)
def test_levels_are_explicitly_derived_from_entry_price(
    side: PositionSide,
    expected_stop: float,
    expected_profit: float,
) -> None:
    levels = protective_levels(side=side, entry_price=100.0, policy=_policy())

    assert levels.stop_loss == pytest.approx(expected_stop)
    assert levels.take_profit == pytest.approx(expected_profit)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan, True])
def test_entry_price_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(
        (TypeError, ValueError), match="entry_price must be a positive finite number"
    ):
        protective_levels(side=PositionSide.LONG, entry_price=value, policy=_policy())


@pytest.mark.parametrize("field", ["stop_loss_fraction", "take_profit_fraction"])
@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, math.inf, math.nan, True])
def test_thresholds_must_be_positive_finite_fractions_below_one(
    field: str,
    value: float,
) -> None:
    values = {"stop_loss_fraction": 0.05, "take_profit_fraction": 0.10, field: value}
    with pytest.raises((TypeError, ValueError)):
        ProtectiveExitPolicy(**values)


@pytest.mark.parametrize(
    "bar",
    [
        (0.0, 2.0, 1.0, 1.5),
        (1.0, math.inf, 0.5, 1.0),
        (1.0, 0.5, 1.0, 0.75),
        (2.0, 1.5, 1.0, 1.25),
        (1.0, 2.0, 1.5, 1.75),
        (1.5, 2.0, 1.0, 2.5),
    ],
)
def test_ohlc_bar_rejects_nonfinite_or_inconsistent_prices(
    bar: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError):
        OHLCBar(*bar)


@pytest.mark.parametrize(
    ("side", "bar", "reason", "fill"),
    [
        (PositionSide.LONG, OHLCBar(100.0, 104.0, 94.0, 96.0), ExitReason.STOP_LOSS, 95.0),
        (PositionSide.LONG, OHLCBar(100.0, 111.0, 99.0, 109.0), ExitReason.TAKE_PROFIT, 110.0),
        (PositionSide.SHORT, OHLCBar(100.0, 106.0, 96.0, 101.0), ExitReason.STOP_LOSS, 105.0),
        (PositionSide.SHORT, OHLCBar(100.0, 101.0, 89.0, 91.0), ExitReason.TAKE_PROFIT, 90.0),
    ],
)
def test_single_intrabar_trigger_uses_threshold_fill(
    side: PositionSide,
    bar: OHLCBar,
    reason: ExitReason,
    fill: float,
) -> None:
    engine = ProtectiveExitEngine(_policy())
    engine.enter(side=side, entry_price=100.0, bar_index=0)

    result = engine.process_bar(bar_index=0, bar=bar)

    assert result is not None
    assert result.reason is reason
    assert result.trigger_basis is ExitTriggerBasis.INTRABAR
    assert result.fill_price == pytest.approx(fill)
    assert result.bars_held == 1


@pytest.mark.parametrize(
    ("side", "bar", "reason", "fill"),
    [
        (PositionSide.LONG, OHLCBar(90.0, 100.0, 89.0, 98.0), ExitReason.STOP_LOSS, 90.0),
        (PositionSide.LONG, OHLCBar(115.0, 116.0, 108.0, 112.0), ExitReason.TAKE_PROFIT, 115.0),
        (PositionSide.SHORT, OHLCBar(112.0, 114.0, 103.0, 104.0), ExitReason.STOP_LOSS, 112.0),
        (PositionSide.SHORT, OHLCBar(85.0, 92.0, 83.0, 91.0), ExitReason.TAKE_PROFIT, 85.0),
    ],
)
def test_open_gap_precedes_intrabar_and_fills_at_actual_open(
    side: PositionSide,
    bar: OHLCBar,
    reason: ExitReason,
    fill: float,
) -> None:
    engine = ProtectiveExitEngine(_policy())
    engine.enter(side=side, entry_price=100.0, bar_index=0)

    result = engine.process_bar(bar_index=0, bar=bar)

    assert result is not None
    assert result.reason is reason
    assert result.trigger_basis is ExitTriggerBasis.OPEN_GAP
    assert result.fill_price == fill


def test_same_bar_collision_defaults_to_conservative_stop_first() -> None:
    engine = ProtectiveExitEngine(_policy())
    engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=0)

    result = engine.process_bar(
        bar_index=0,
        bar=OHLCBar(open=100.0, high=112.0, low=94.0, close=101.0),
    )

    assert result is not None
    assert result.reason is ExitReason.STOP_LOSS
    assert result.fill_price == pytest.approx(95.0)


def test_same_bar_collision_can_select_take_profit_first() -> None:
    engine = ProtectiveExitEngine(
        _policy(collision_policy=SameBarCollisionPolicy.TAKE_PROFIT_FIRST)
    )
    engine.enter(side=PositionSide.SHORT, entry_price=100.0, bar_index=0)

    result = engine.process_bar(
        bar_index=0,
        bar=OHLCBar(open=100.0, high=106.0, low=89.0, close=101.0),
    )

    assert result is not None
    assert result.reason is ExitReason.TAKE_PROFIT
    assert result.fill_price == pytest.approx(90.0)


def test_fail_closed_collision_does_not_mutate_engine_state() -> None:
    engine = ProtectiveExitEngine(_policy(collision_policy="fail_closed"))
    engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=3)
    before = engine.state

    with pytest.raises(AmbiguousIntrabarPathError, match="touches both"):
        engine.process_bar(
            bar_index=3,
            bar=OHLCBar(open=100.0, high=112.0, low=94.0, close=101.0),
        )

    assert engine.state == before


def test_max_hold_uses_close_after_counting_processed_bars() -> None:
    engine = ProtectiveExitEngine(_policy(max_hold_bars=2))
    engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=4)
    assert (
        engine.process_bar(bar_index=4, bar=OHLCBar(open=100.0, high=103.0, low=98.0, close=102.0))
        is None
    )

    result = engine.process_bar(
        bar_index=7,
        bar=OHLCBar(open=102.0, high=104.0, low=99.0, close=103.0),
    )

    assert result is not None
    assert result.reason is ExitReason.MAX_HOLD
    assert result.trigger_basis is ExitTriggerBasis.BAR_CLOSE
    assert result.fill_price == 103.0
    assert result.bars_held == 2


def test_price_trigger_precedes_max_hold_on_same_bar() -> None:
    engine = ProtectiveExitEngine(_policy(max_hold_bars=1))
    engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=0)

    result = engine.process_bar(
        bar_index=0,
        bar=OHLCBar(open=100.0, high=103.0, low=94.0, close=99.0),
    )

    assert result is not None
    assert result.reason is ExitReason.STOP_LOSS
    assert result.trigger_basis is ExitTriggerBasis.INTRABAR


def test_same_bar_reentry_and_configured_cooldown_are_enforced() -> None:
    engine = ProtectiveExitEngine(_policy(cooldown_bars=2))
    engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=5)
    assert (
        engine.process_bar(bar_index=5, bar=OHLCBar(open=94.0, high=96.0, low=92.0, close=95.0))
        is not None
    )

    for blocked_index in (5, 6, 7):
        assert not engine.can_enter(bar_index=blocked_index)
        with pytest.raises(EntryNotAllowedError):
            engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=blocked_index)

    assert engine.can_enter(bar_index=8)
    position = engine.enter(side=PositionSide.SHORT, entry_price=100.0, bar_index=8)
    assert position.side is PositionSide.SHORT


def test_bar_indices_must_advance_strictly() -> None:
    engine = ProtectiveExitEngine(_policy())
    engine.process_bar(bar_index=2, bar=OHLCBar(100.0, 101.0, 99.0, 100.0))

    with pytest.raises(ValueError, match="strictly increasing"):
        engine.process_bar(bar_index=2, bar=OHLCBar(100.0, 101.0, 99.0, 100.0))


def test_processing_is_prefix_invariant() -> None:
    bars = [
        OHLCBar(100.0, 102.0, 98.0, 101.0),
        OHLCBar(101.0, 104.0, 99.0, 103.0),
        OHLCBar(103.0, 109.0, 102.0, 108.0),
        OHLCBar(108.0, 111.0, 107.0, 110.0),
    ]
    full_engine = ProtectiveExitEngine(_policy())
    full_engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=0)
    full_outputs = []
    full_states = []
    for index, bar in enumerate(bars):
        full_outputs.append(full_engine.process_bar(bar_index=index, bar=bar))
        full_states.append(full_engine.state)

    for prefix_length in range(1, len(bars) + 1):
        prefix_engine = ProtectiveExitEngine(_policy())
        prefix_engine.enter(side=PositionSide.LONG, entry_price=100.0, bar_index=0)
        prefix_outputs = [
            prefix_engine.process_bar(bar_index=index, bar=bar)
            for index, bar in enumerate(bars[:prefix_length])
        ]

        assert prefix_outputs == full_outputs[:prefix_length]
        assert prefix_engine.state == full_states[prefix_length - 1]
