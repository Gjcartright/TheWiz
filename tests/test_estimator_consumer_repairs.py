"""Direct-consumer regressions from independent estimator review."""

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.economic_contract import rolling_y_on_x_beta
from quant_platform.statistics.math_v2 import attach_math_v2_statistics


def _rows(stamps):
    # The constant price pair explicitly invalidates EG. The documented legacy
    # spread fallback isolates the wrapper's OU clock with a known AR(1) path.
    return [dict(price_x=2.0, price_y=3.0, spread=1.0 + 0.2 * 0.9 ** i,
                 **({"timestamp": stamps[i]} if stamps is not None else {}))
            for i in range(80)]


@pytest.mark.parametrize("defect", ["gap", "duplicate", "reversed", "missing"])
def test_batch_attachment_preserves_present_observation_clock_for_ou(defect):
    stamps = list(pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC"))
    if defect == "gap":
        stamps[40:] = [value + pd.Timedelta(hours=1) for value in stamps[40:]]
    elif defect == "duplicate":
        stamps[40] = stamps[39]
    elif defect == "reversed":
        stamps.reverse()
    else:
        stamps[40] = "bad-timestamp"
    rows = _rows(stamps)
    result = attach_math_v2_statistics(rows, zscore_window=20, min_zscore_window=10)
    ou = next(item for item in result["audit"] if item["metric"] == "ou_ar1_with_intercept")
    assert ou["validity_status"] == "invalid"
    assert all(row["math_v2_half_life"] is None for row in rows)
    assert [row["timestamp"] for row in rows] == stamps
    assert ou["point_in_time"] is False
    assert "batch_fit_requires_walk_forward" in ou["validity_reason"]


@pytest.mark.parametrize("clock", ["regular", "positional", "mixed_iso"])
def test_batch_attachment_regular_or_declared_positional_time_remains_in_bars(clock):
    stamps = list(pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC"))
    if clock == "positional":
        stamps = None
    elif clock == "mixed_iso":
        stamps = [value.isoformat() if i else "2026-01-01" for i, value in enumerate(stamps)]
    rows = _rows(stamps)
    result = attach_math_v2_statistics(rows, zscore_window=20, min_zscore_window=10)
    ou = next(item for item in result["audit"] if item["metric"] == "ou_ar1_with_intercept")
    assert ou["validity_status"] == "valid"
    assert rows[0]["math_v2_half_life"] == pytest.approx(math.log(2) / -math.log(.9), rel=1e-9)
    assert ou["point_in_time"] is False
    assert result["status"] == "partial"  # Other estimators were not made valid.


@pytest.mark.parametrize("leg", ["x", "y"])
@pytest.mark.parametrize("imaginary", [0.0, 1.0])
def test_rolling_beta_rejects_complex_data_without_discarding_imaginary_part(leg, imaginary):
    x = pd.Series(np.arange(6, dtype=float))
    y = 1.0 + 2.0 * x
    if leg == "x":
        x = x.astype(complex) + imaginary * 1j
    else:
        y = y.astype(complex) + imaginary * 1j
    with pytest.raises(ValueError, match="real"):
        rolling_y_on_x_beta(x, y, window=3)
