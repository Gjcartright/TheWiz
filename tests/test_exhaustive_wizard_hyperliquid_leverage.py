from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.orchestration.exhaustive_wizard_hyperliquid_leverage import (
    _simulate_leverage_path,
)


def test_leverage_path_drawdown_includes_first_bar_loss_from_fold_capital():
    bars = pd.DataFrame(
        {
            "fold_number": [1, 1],
            "net_return": [-0.10, 0.0],
            "gross_return": [-0.10, 0.0],
            "funding": [0.0, 0.0],
            "slippage": [0.0, 0.0],
            "target_position": [1.0, 1.0],
        }
    )

    result = _simulate_leverage_path(
        bars,
        effective_leverage=1.0,
        stress={},
        maximum_leg_weight=0.5,
    )

    assert result["maximum_drawdown"] == pytest.approx(0.10)
