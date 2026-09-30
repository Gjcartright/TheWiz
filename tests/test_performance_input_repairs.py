"""Independent clock-declaration and retained-index boundary probes."""

import math

import numpy as np
import pandas as pd
import pytest

from quant_platform.performance_math import (
    calculate_annualized_sharpe,
    normalize_interval,
    resolve_annualization,
)


RETURNS = [0.01, -0.005, 0.007, 0.002]


@pytest.mark.parametrize("hours", [[0, 1, 1, 2], [3, 2, 1, 0], [0, 1, 3, 7], [0, 1, 2, None]])
@pytest.mark.parametrize("declaration", [{"interval": "1h"}, {"periods_per_year": 8760}, {}])
def test_existing_return_datetime_index_is_never_an_absent_grid(hours, declaration):
    stamps = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(hours, unit="h")
    result = calculate_annualized_sharpe(pd.Series(RETURNS, index=stamps), **declaration)
    assert result.status == "blocked"
    assert result.periods_per_year is None
    assert math.isnan(result.value)


def test_existing_return_index_infers_clock_and_checks_explicit_periods():
    stamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    values = pd.Series(RETURNS, index=stamps)
    inferred = calculate_annualized_sharpe(values)
    explicit = calculate_annualized_sharpe(values, periods_per_year=8760)
    assert inferred.status == explicit.status == "valid"
    assert inferred.interval == "1h"
    assert inferred.periods_per_year == explicit.periods_per_year == 8760
    assert inferred.value == explicit.value == pytest.approx(math.sqrt(8760) * np.mean(RETURNS) / np.std(RETURNS, ddof=1))
    assert calculate_annualized_sharpe(values, periods_per_year=365).status == "blocked"
    assert calculate_annualized_sharpe(values, interval="1d").status == "blocked"


@pytest.mark.parametrize("interval", [pd.NA, np.nan, pd.NaT, True, False, ["1h"], np.array(["1h"]), pd.Series(["1h"]), {}])
@pytest.mark.parametrize("call", ["resolve", "sharpe", "explicit"])
def test_invalid_declared_interval_does_not_raise_or_become_absent(interval, call):
    stamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    if call == "resolve":
        result = resolve_annualization(interval=interval, timestamps=stamps)
    else:
        extra = {"periods_per_year": 8760} if call == "explicit" else {}
        result = calculate_annualized_sharpe(RETURNS, interval=interval, timestamps=stamps, **extra)
    assert result.status == "blocked"
    assert result.reason == "invalid_declared_interval"
    assert normalize_interval(interval) is None


@pytest.mark.parametrize("interval", [None, "", " \t "])
def test_only_absent_or_blank_text_interval_allows_grid_inference(interval):
    stamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    result = resolve_annualization(interval=interval, timestamps=stamps)
    assert result.status == "valid"
    assert result.interval == "1h"
    assert calculate_annualized_sharpe(RETURNS, interval=interval, periods_per_year=8760).status == "valid"


@pytest.mark.parametrize("interval", ["1h", " 1 hour ", "HOURLY"])
def test_supported_string_declarations_keep_normalization(interval):
    assert normalize_interval(interval) == "1h"
    assert resolve_annualization(interval=interval).periods_per_year == 8760


@pytest.mark.parametrize("periods", [True, False, np.bool_(True), pd.NA, np.nan, np.inf, -1, 0])
def test_invalid_explicit_periods_are_not_a_numeric_clock(periods):
    result = calculate_annualized_sharpe(RETURNS, periods_per_year=periods)
    assert result.status == "blocked"
    assert result.reason == "invalid_explicit_periods_per_year"


def test_mixed_iso_representations_describe_one_exact_ordered_grid():
    stamps = ["2026-01-01", "2026-01-01T01:00:00Z", "2026-01-01 02:00:00+00:00", "2025-12-31T22:00:00-05:00"]
    result = resolve_annualization(timestamps=stamps)
    assert result.status == "valid"
    assert result.periods_per_year == 8760
    values = pd.Series(RETURNS, index=pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC"))
    assert calculate_annualized_sharpe(values, timestamps=iter(stamps)).status == "valid"


@pytest.mark.parametrize("bad", [None, "not-a-date", "2026-01-01T01:00:00Z", "2026-01-01T04:00:00Z"])
def test_mixed_parser_does_not_discard_missing_duplicate_or_irregular_time(bad):
    stamps = ["2026-01-01", "2026-01-01T01:00:00Z", bad, "2026-01-01T03:00:00Z"]
    assert resolve_annualization(interval="1h", timestamps=stamps).status == "blocked"


def test_equivalent_timezone_index_matches_but_order_is_not_repaired():
    stamps = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    values = pd.Series(RETURNS, index=stamps.tz_convert("America/New_York"))
    assert calculate_annualized_sharpe(values, timestamps=stamps).status == "valid"
    assert calculate_annualized_sharpe(values, timestamps=stamps[::-1]).reason == "return_timestamp_index_mismatch"
