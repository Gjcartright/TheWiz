"""Timeframe-aware performance calculations for Math V2."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

MATH_VERSION = "math-v2.3-venue-clock-execution"
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

PERIODS_PER_YEAR: dict[str, float] = {
    "1m": 525_600,
    "3m": 175_200,
    "5m": 105_120,
    "15m": 35_040,
    "30m": 17_520,
    "1h": 8_760,
    "2h": 4_380,
    "4h": 2_190,
    "6h": 1_460,
    "8h": 1_095,
    "12h": 730,
    "1d": 365,
    "3d": 365.0 / 3.0,
    "1w": 365.0 / 7.0,
    "1M": 12.0,
}

_INTERVAL_ALIASES: dict[str, str] = {
    "1m": "1m",
    "1min": "1m",
    "1mins": "1m",
    "1minute": "1m",
    "1minutes": "1m",
    "3m": "3m",
    "3min": "3m",
    "3mins": "3m",
    "3minute": "3m",
    "3minutes": "3m",
    "5m": "5m",
    "5min": "5m",
    "5mins": "5m",
    "5minute": "5m",
    "5minutes": "5m",
    "15m": "15m",
    "15min": "15m",
    "15mins": "15m",
    "15minute": "15m",
    "15minutes": "15m",
    "30m": "30m",
    "30min": "30m",
    "30mins": "30m",
    "30minute": "30m",
    "30minutes": "30m",
    "1h": "1h",
    "1hr": "1h",
    "1hour": "1h",
    "1hours": "1h",
    "hour": "1h",
    "hourly": "1h",
    "60m": "1h",
    "60min": "1h",
    "2h": "2h",
    "2hr": "2h",
    "2hour": "2h",
    "120m": "2h",
    "4h": "4h",
    "4hr": "4h",
    "4hour": "4h",
    "4hours": "4h",
    "240m": "4h",
    "6h": "6h",
    "6hr": "6h",
    "6hour": "6h",
    "360m": "6h",
    "8h": "8h",
    "8hr": "8h",
    "8hour": "8h",
    "480m": "8h",
    "12h": "12h",
    "12hr": "12h",
    "12hour": "12h",
    "720m": "12h",
    "1d": "1d",
    "1day": "1d",
    "daily": "1d",
    "day": "1d",
    "3d": "3d",
    "3day": "3d",
    "3days": "3d",
    "1w": "1w",
    "1wk": "1w",
    "1week": "1w",
    "weekly": "1w",
    "1mo": "1M",
    "1month": "1M",
    "monthly": "1M",
}


@dataclass(frozen=True)
class Annualization:
    interval: str | None
    periods_per_year: float | None
    status: str
    reason: str


@dataclass(frozen=True)
class SharpeCalculation:
    value: float
    interval: str | None
    periods_per_year: float | None
    status: str
    reason: str


def normalize_interval(interval: object) -> str | None:
    """Normalize supported candle labels without guessing unknown units."""

    # A container's representation must not become a scalar declaration (for
    # example, ['1h'] used to normalize to 1h). Missing sentinels and booleans
    # likewise are not absent declarations and must never be truth-tested.
    if not isinstance(interval, str):
        return None
    raw = interval.strip()
    if raw == "1M":
        return "1M"
    key = re.sub(r"[^a-z0-9]+", "", raw.lower())
    return _INTERVAL_ALIASES.get(key)


def resolve_annualization(
    *,
    interval: object | None = None,
    timestamps: Iterable[object] | pd.Series | pd.Index | None = None,
) -> Annualization:
    """Resolve the 365-day candle clock without discarding bad observations.

    An absent grid permits an explicitly declared clock. A supplied but
    insufficient or invalid grid is different and cannot be overridden by a
    label. Supplied timestamps retain their order, duplicates and missing rows.
    """

    if timestamps is not None:
        try:
            timestamps = tuple(timestamps)
        except TypeError:
            return Annualization(None, None, "blocked", "invalid_timestamp_values")
    normalized = normalize_interval(interval)
    if interval is not None and not isinstance(interval, str):
        return Annualization(None, None, "blocked", "invalid_declared_interval")
    if timestamps is not None and normalized == "1M":
        monthly_state = _calendar_month_grid(timestamps)
        if monthly_state != "verified":
            return Annualization("1M", None, "blocked", monthly_state)
        return Annualization("1M", 12.0, "valid", "declared_monthly_grid_verified")
    if timestamps is not None and interval is None:
        monthly_state = _calendar_month_grid(timestamps)
        if monthly_state == "verified":
            return Annualization("1M", 12.0, "valid", "inferred_from_monthly_grid")
    grid_state, inferred_seconds = _timestamp_grid(timestamps)
    if grid_state not in {"absent", "verified"}:
        return Annualization(normalized, None, "blocked", grid_state)
    if normalized is not None:
        if inferred_seconds is not None:
            inferred = _canonical_interval_for_seconds(inferred_seconds)
            if inferred is None:
                return Annualization(
                    normalized,
                    None,
                    "blocked",
                    "declared_interval_timestamp_grid_unsupported",
                )
            if inferred != normalized:
                return Annualization(
                    normalized,
                    None,
                    "blocked",
                    f"declared_interval_timestamp_mismatch:{normalized}!={inferred}",
                )
        return Annualization(
            normalized,
            float(PERIODS_PER_YEAR[normalized]),
            "valid",
            "declared_interval_verified" if inferred_seconds is not None else "declared_interval",
        )

    if interval is not None and interval.strip():
        return Annualization(None, None, "blocked", "unknown_declared_interval")
    if inferred_seconds is None:
        return Annualization(None, None, "blocked", "unknown_interval")

    inferred = _canonical_interval_for_seconds(inferred_seconds)
    if inferred is None:
        return Annualization(None, None, "blocked", "unsupported_or_irregular_interval")
    return Annualization(
        inferred, float(PERIODS_PER_YEAR[inferred]), "valid", "inferred_from_timestamps"
    )


def calculate_annualized_sharpe(
    returns: pd.Series | Iterable[float],
    *,
    interval: object | None = None,
    timestamps: Iterable[object] | pd.Series | pd.Index | None = None,
    periods_per_year: float | None = None,
) -> SharpeCalculation:
    """Calculate Sharpe and expose whether its annualization is authoritative.

    A return Series' DatetimeIndex is its observation grid when no separate
    timestamps are supplied. Explicit timestamps must match that same index;
    neither declaration may override a malformed or inconsistent grid.
    """

    values = pd.to_numeric(pd.Series(returns), errors="coerce")
    if interval is not None and not isinstance(interval, str):
        return SharpeCalculation(float("nan"), None, None,
                                 "blocked", "invalid_declared_interval")
    if timestamps is None and isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        timestamps = returns.index
    stamps = None if timestamps is None else list(timestamps)
    if stamps is not None and len(stamps) != len(values):
        return SharpeCalculation(float("nan"), normalize_interval(interval), None,
                                 "blocked", "return_timestamp_length_mismatch")
    if stamps is not None and isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        expected = _parse_timestamps(stamps)
        actual = _parse_timestamps(returns.index)
        if not actual.equals(expected):
            return SharpeCalculation(float("nan"), normalize_interval(interval), None,
                                     "blocked", "return_timestamp_index_mismatch")
    if periods_per_year is not None:
        try:
            explicit_periods = (float("nan") if isinstance(periods_per_year, (bool, np.bool_))
                                else float(periods_per_year))
        except (TypeError, ValueError, OverflowError):
            explicit_periods = float("nan")
        if not np.isfinite(explicit_periods) or explicit_periods <= 0.0:
            return SharpeCalculation(
                float("nan"),
                normalize_interval(interval),
                None,
                "blocked",
                "invalid_explicit_periods_per_year",
            )
        if stamps is not None or (interval is not None and interval.strip()):
            declared = resolve_annualization(interval=interval, timestamps=stamps)
            if declared.status != "valid":
                return SharpeCalculation(float("nan"), declared.interval, None,
                                         declared.status, declared.reason)
            if not np.isclose(explicit_periods, declared.periods_per_year, rtol=1e-12, atol=0.0):
                return SharpeCalculation(float("nan"), declared.interval, None,
                                         "blocked", "explicit_periods_timestamp_or_interval_mismatch")
        annualization = Annualization(
            normalize_interval(interval),
            explicit_periods,
            "valid",
            "explicit_periods_per_year",
        )
    else:
        annualization = resolve_annualization(interval=interval, timestamps=stamps)

    if annualization.status != "valid" or annualization.periods_per_year is None:
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            annualization.status,
            annualization.reason,
        )
    if values.empty or not bool(np.isfinite(values).any()):
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            "blocked",
            "no_finite_returns",
        )
    if bool(values.isna().any()) or not bool(np.isfinite(values).all()):
        return SharpeCalculation(float("nan"), annualization.interval,
                                 annualization.periods_per_year, "blocked",
                                 "incomplete_or_nonfinite_returns")
    if len(values) < 2:
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            "blocked",
            "insufficient_return_observations",
        )
    # Historical returns are a sample used to estimate future volatility. Keep
    # this convention aligned with the PSR/DSR inference path.
    standard_deviation = float(values.std(ddof=1))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            "blocked",
            "zero_or_nonfinite_return_variance",
        )
    value = float(np.sqrt(annualization.periods_per_year) * values.mean() / standard_deviation)
    if not np.isfinite(value):
        return SharpeCalculation(float("nan"), annualization.interval,
                                 annualization.periods_per_year, "blocked", "nonfinite_sharpe")
    return SharpeCalculation(
        value,
        annualization.interval,
        annualization.periods_per_year,
        "valid",
        annualization.reason,
    )


def compound_simple_returns(returns: pd.Series | Iterable[float]) -> pd.Series:
    """Compound a finite simple-return stream from unit initial capital."""

    values = pd.to_numeric(pd.Series(returns), errors="coerce")
    if bool(values.isna().any()) or bool((~np.isfinite(values)).any()):
        raise ValueError("simple returns must be complete and finite")
    if bool(values.le(-1.0).any()):
        raise ValueError("simple returns must remain greater than -100%")
    return values.add(1.0).cumprod().rename("equity")


def calculate_max_drawdown(equity: pd.Series | Iterable[float]) -> float:
    """Calculate peak-to-trough drawdown including initial equity of one."""

    values = pd.to_numeric(pd.Series(equity), errors="coerce")
    if values.empty:
        return 0.0
    if (
        bool(values.isna().any())
        or bool((~np.isfinite(values)).any())
        or bool(values.le(0.0).any())
    ):
        raise ValueError("equity must be complete, finite, and positive")
    path = np.concatenate(([1.0], values.to_numpy(dtype=float)))
    peaks = np.maximum.accumulate(path)
    drawdown = np.divide(peaks - path, peaks, out=np.zeros_like(path), where=peaks > 0.0)
    return float(np.max(drawdown))


def _parse_timestamps(timestamps: Iterable[object]) -> pd.DatetimeIndex:
    """Parse each representation without imposing the first value's format.

    Mixed ISO/date/Timestamp representations can describe the same grid. This
    only normalizes their representation and timezone: it does not sort,
    deduplicate, fill, discard or resample any observation.
    """
    return pd.DatetimeIndex(pd.to_datetime(
        list(timestamps), format="mixed", utc=True, errors="coerce"
    ))


def _calendar_month_grid(
    timestamps: Iterable[object] | pd.Series | pd.Index,
) -> str:
    parsed = _parse_timestamps(timestamps)
    if parsed.isna().any():
        return "invalid_timestamp_values"
    if len(parsed) < 3:
        return "insufficient_timestamp_observations"
    months = parsed.year * 12 + parsed.month
    if not bool(np.all(np.diff(months) == 1)):
        return "irregular_monthly_timestamp_grid"
    if len({timestamp.time() for timestamp in parsed}) != 1:
        return "irregular_monthly_timestamp_grid"
    days = parsed.day
    same_day = bool(np.all(days == days[0]))
    month_end = bool(np.all(parsed.is_month_end))
    return "verified" if same_day or month_end else "irregular_monthly_timestamp_grid"


def _timestamp_grid(
    timestamps: Iterable[object] | pd.Series | pd.Index | None,
) -> tuple[str, float | None]:
    if timestamps is None:
        return "absent", None
    parsed = pd.Series(_parse_timestamps(timestamps))
    if parsed.isna().any():
        return "invalid_timestamp_values", None
    if len(parsed) < 3:
        return "insufficient_timestamp_observations", None
    deltas = parsed.diff().dropna().dt.total_seconds()
    if bool(deltas.le(0).any()):
        return "nonincreasing_timestamp_grid", None
    median = float(deltas.median())
    if not np.isfinite(median) or not bool(np.isclose(deltas, median, rtol=1e-9, atol=1e-6).all()):
        return "irregular_timestamp_grid", None
    return "verified", median


def _median_interval_seconds(
    timestamps: Iterable[object] | pd.Series | pd.Index | None,
) -> float | None:
    """Compatibility helper; resolution uses the richer grid state above."""
    state, seconds = _timestamp_grid(timestamps)
    return seconds if state == "verified" else None


def _canonical_interval_for_seconds(seconds: float) -> str | None:
    expected = {
        "1m": 60.0,
        "3m": 180.0,
        "5m": 300.0,
        "15m": 900.0,
        "30m": 1_800.0,
        "1h": 3_600.0,
        "2h": 7_200.0,
        "4h": 14_400.0,
        "6h": 21_600.0,
        "8h": 28_800.0,
        "12h": 43_200.0,
        "1d": 86_400.0,
        "3d": 259_200.0,
        "1w": 604_800.0,
    }
    for interval, target in expected.items():
        if abs(seconds - target) <= max(1e-6, target * 1e-9):
            return interval
    return None
