"""Independent validity and paired-sample regressions for the isolated candidate."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.economic_contract import rolling_y_on_x_beta, y_on_x_beta
from quant_platform.statistics.math_v2 import fit_engle_granger, fit_ou, fit_static_y_on_x_ols


def _prices():
    x = pd.Series(np.exp(np.linspace(1.0, 2.0, 120)))
    y = np.exp(0.4 + 1.7 * np.log(x) + 0.01 * np.sin(np.arange(120)))
    return x, y


@pytest.mark.parametrize("fit", [fit_static_y_on_x_ols, fit_engle_granger])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_price_estimators_reject_incomplete_or_invalid_prices(fit, bad):
    x, y = _prices()
    y.iloc[50] = bad
    result = fit(x, y)
    assert result.validity_status == "invalid"
    assert "residual" not in result.values


@pytest.mark.parametrize("fit", [fit_static_y_on_x_ols, fit_engle_granger])
def test_price_estimators_reject_unidentified_or_misaligned_inputs(fit):
    x, y = _prices()
    assert fit(pd.Series([2.0] * len(x)), y).validity_status == "invalid"
    assert fit(x, y.set_axis(pd.RangeIndex(1, len(y) + 1))).validity_status == "invalid"
    repeated = pd.Index([0] * len(x))
    assert fit(x.set_axis(repeated), y.set_axis(repeated)).validity_status == "invalid"


@pytest.mark.parametrize("swap", [False, True])
def test_static_fit_matches_centered_hand_ols_in_both_orientations(swap):
    x, y = _prices()
    if swap:
        x, y = y, x
    log_x, log_y = np.log(x), np.log(y)
    dx, dy = log_x - log_x.mean(), log_y - log_y.mean()
    expected_beta = float((dx * dy).sum() / (dx * dx).sum())
    expected_alpha = float(log_y.mean() - expected_beta * log_x.mean())
    result = fit_static_y_on_x_ols(x, y)
    assert result.validity_status == "valid"
    assert result.values["hedge_ratio"] == pytest.approx(expected_beta, abs=1e-12)
    assert result.values["alpha"] == pytest.approx(expected_alpha, abs=1e-12)
    np.testing.assert_allclose(result.values["residual"], log_y - expected_alpha - expected_beta * log_x, atol=1e-12)


def test_static_constant_response_is_an_identified_zero_slope_not_execution_eligibility():
    x = pd.Series(np.exp(np.linspace(1.0, 2.0, 80)))
    result = fit_static_y_on_x_ols(x, pd.Series([3.0] * len(x)))
    assert result.validity_status == "valid"
    assert result.values["hedge_ratio"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("level", [0.0, 1.0, 3.0, -3.0])
def test_ou_constant_level_is_unidentified(level):
    result = fit_ou(pd.Series([level] * 100))
    assert result.validity_status == "invalid"
    assert "half_life" not in result.values


def test_near_constant_designs_do_not_manufacture_identification():
    spread = pd.Series(3.0 + np.linspace(0.0, 1e-15, 100))
    assert fit_ou(spread).validity_status == "invalid"
    x = np.exp(pd.Series(1.0 + np.linspace(0.0, 1e-15, 100)))
    assert fit_static_y_on_x_ols(x, pd.Series(np.linspace(2.0, 3.0, 100))).validity_status == "invalid"


def _ou_path(index=None):
    return pd.Series(1.0 + 0.2 * 0.9 ** np.arange(80), index=index)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_ou_rejects_nonfinite_observations_without_compressing_timeline(bad):
    values = _ou_path()
    values.iloc[20] = bad
    result = fit_ou(values)
    assert result.validity_status == "invalid"
    assert "half_life" not in result.values


@pytest.mark.parametrize("defect", ["gap", "duplicate", "reversed", "nat"])
def test_ou_rejects_invalid_timestamp_grid(defect):
    index = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    values = list(index)
    if defect == "gap":
        values[40:] = [value + pd.Timedelta(hours=1) for value in values[40:]]
    elif defect == "duplicate":
        values[40] = values[39]
    elif defect == "reversed":
        values = values[::-1]
    else:
        values[40] = pd.NaT
    result = fit_ou(_ou_path(pd.DatetimeIndex(values)))
    assert result.validity_status == "invalid"
    assert "half_life" not in result.values


def test_ou_regular_datetime_and_positional_grids_keep_declared_time_units():
    values = _ou_path()
    dated = values.set_axis(pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC"))
    first = fit_ou(values, delta_t=0.25)
    second = fit_ou(dated, delta_t=0.25)
    assert first.validity_status == second.validity_status == "valid"
    expected_half_life = 0.25 * math.log(2.0) / -math.log(0.9)
    assert first.values["half_life"] == pytest.approx(expected_half_life, rel=1e-9)
    assert second.values["half_life"] == pytest.approx(expected_half_life, rel=1e-9)
    assert second.values["theta"] == pytest.approx(-math.log(0.9) / 0.25, rel=1e-9)


@pytest.mark.parametrize("delta_t", [1e-320, 1e308])
def test_ou_does_not_label_nonfinite_derived_parameters_valid(delta_t):
    result = fit_ou(_ou_path(), delta_t=delta_t)
    assert result.validity_status == "invalid"
    assert "half_life" not in result.values


def test_ou_insufficient_transition_information_is_invalid():
    result = fit_ou(pd.Series([1.0, 0.9, 0.81]), min_rows=2)
    assert result.validity_status == "invalid"


def _paired_window_beta(x, y):
    pairs = [(float(a), float(b)) for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 3:
        return np.nan
    mx = sum(a for a, _ in pairs) / len(pairs)
    my = sum(b for _, b in pairs) / len(pairs)
    denominator = sum((a - mx) ** 2 for a, _ in pairs)
    sample_variance = denominator / (len(pairs) - 1)
    return sum((a - mx) * (b - my) for a, b in pairs) / denominator if sample_variance > 1e-12 else np.nan


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_rolling_beta_has_one_paired_finite_sample(missing):
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = pd.Series([0.0, 2.0, missing, 6.0, 7.0, 9.0, 10.0])
    result = rolling_y_on_x_beta(x, y, window=4, min_periods=3)
    expected = [_paired_window_beta(x.iloc[max(0, i - 3):i + 1], y.iloc[max(0, i - 3):i + 1]) for i in range(len(x))]
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert result.iloc[3] == pytest.approx(2.0)


def test_rolling_beta_prefix_invariance_and_complete_window_agreement():
    x = pd.Series(np.arange(20, dtype=float))
    y = 3.0 + 2.0 * x
    whole = rolling_y_on_x_beta(x, y, window=5)
    prefix = rolling_y_on_x_beta(x.iloc[:10], y.iloc[:10], window=5)
    pd.testing.assert_series_equal(whole.iloc[:10], prefix)
    assert whole.dropna().eq(2.0).all()
    assert y_on_x_beta(x, y) == pytest.approx(2.0)


def test_rolling_beta_pairs_missing_x_and_y_before_minimum_count():
    x = pd.Series([0.0, np.nan, 2.0, 3.0])
    y = pd.Series([0.0, 1.0, np.nan, 3.0])
    assert rolling_y_on_x_beta(x, y, window=4, min_periods=3).isna().all()


def test_rolling_beta_rejects_mismatched_or_duplicate_indices():
    x = pd.Series([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="index"):
        rolling_y_on_x_beta(x, x.set_axis([1, 2, 3]), window=3)
    with pytest.raises(ValueError, match="index"):
        rolling_y_on_x_beta(x.set_axis([0, 0, 1]), x.set_axis([0, 0, 1]), window=3)


def test_rolling_beta_respects_explicit_zero_minimum_as_invalid():
    with pytest.raises(ValueError):
        rolling_y_on_x_beta(pd.Series([0.0, 1.0]), pd.Series([0.0, 1.0]), window=2, min_periods=0)


def test_finite_nullable_real_observations_keep_valid_point_estimates():
    x, y = _prices()
    result = fit_static_y_on_x_ols(x.astype("Float64"), y.astype("Float64"))
    reference = fit_static_y_on_x_ols(x, y)
    assert result.validity_status == "valid"
    assert result.values["hedge_ratio"] == pytest.approx(reference.values["hedge_ratio"])
    assert fit_ou(_ou_path().astype("Float64")).validity_status == "valid"


@pytest.mark.parametrize("defect", ["numeric_gap", "numeric_reversed", "timedelta_gap"])
def test_ou_retained_numeric_and_timedelta_grids_are_not_compressed(defect):
    coordinates = np.arange(80, dtype=float)
    if defect.endswith("gap"):
        coordinates[40:] += 1.0
    else:
        coordinates = coordinates[::-1]
    index = pd.to_timedelta(coordinates, unit="h") if defect == "timedelta_gap" else pd.Index(coordinates)
    assert fit_ou(_ou_path(index)).validity_status == "invalid"
