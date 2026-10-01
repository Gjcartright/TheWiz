"""New ordinary numeric contract checks; not original project test cases."""

import math
import sys

import pytest

from quant_platform.protective_exits import (
    PositionSide,
    ProtectiveExitPolicy,
    protective_levels,
)


@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
@pytest.mark.parametrize("entry", [sys.float_info.max, math.ulp(0.0)])
def test_accepted_price_domain_cannot_return_nonpositive_or_nonfinite_levels(side, entry):
    policy = ProtectiveExitPolicy(stop_loss_fraction=0.5, take_profit_fraction=0.5)
    try:
        levels = protective_levels(side=side, entry_price=entry, policy=policy)
    except (ValueError, OverflowError):
        # Explicit unavailable/rejection is acceptable at representation limits.
        return
    assert math.isfinite(levels.stop_loss) and levels.stop_loss > 0.0
    assert math.isfinite(levels.take_profit) and levels.take_profit > 0.0


@pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
def test_ordinary_price_domain_keeps_expected_levels(side):
    levels = protective_levels(
        side=side,
        entry_price=100.0,
        policy=ProtectiveExitPolicy(stop_loss_fraction=0.5, take_profit_fraction=0.5),
    )
    expected = (50.0, 150.0) if side is PositionSide.LONG else (150.0, 50.0)
    assert (levels.stop_loss, levels.take_profit) == expected
