"""Timeframe-aware performance calculations for Math V2."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np
import pandas as pd


MATH_VERSION = "math-v2"
SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0

PERIODS_PER_YEAR: dict[str, int] = {
    "1m": 525_600,
    "5m": 105_120,
    "15m": 35_040,
    "1h": 8_760,
    "4h": 2_190,
    "1d": 365,
}

_INTERVAL_ALIASES: dict[str, str] = {
    "1m": "1m",
    "1min": "1m",
    "1mins": "1m",
    "1minute": "1m",
    "1minutes": "1m",
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
    "1h": "1h",
    "1hr": "1h",
    "1hour": "1h",
    "60m": "1h",
    "60min": "1h",
    "4h": "4h",
    "4hr": "4h",
    "4hour": "4h",
    "240m": "4h",
    "1d": "1d",
    "1day": "1d",
    "daily": "1d",
    "day": "1d",
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

    key = re.sub(r"[^a-z0-9]+", "", str(interval or "").lower())
    return _INTERVAL_ALIASES.get(key)


def resolve_annualization(
    *,
    interval: object | None = None,
    timestamps: Iterable[object] | pd.Series | pd.Index | None = None,
) -> Annualization:
    """Resolve annualization from a declared interval or regular timestamps."""

    normalized = normalize_interval(interval)
    if normalized is not None:
        return Annualization(normalized, float(PERIODS_PER_YEAR[normalized]), "valid", "declared_interval")

    inferred_seconds = _median_interval_seconds(timestamps)
    if inferred_seconds is None:
        return Annualization(None, None, "blocked", "unknown_interval")

    inferred = _canonical_interval_for_seconds(inferred_seconds)
    if inferred is None:
        return Annualization(None, None, "blocked", "unsupported_or_irregular_interval")
    return Annualization(inferred, float(PERIODS_PER_YEAR[inferred]), "valid", "inferred_from_timestamps")


def calculate_annualized_sharpe(
    returns: pd.Series | Iterable[float],
    *,
    interval: object | None = None,
    timestamps: Iterable[object] | pd.Series | pd.Index | None = None,
    periods_per_year: float | None = None,
) -> SharpeCalculation:
    """Calculate Sharpe and expose whether its annualization is authoritative."""

    values = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if periods_per_year is not None:
        annualization = Annualization(
            normalize_interval(interval),
            float(periods_per_year),
            "valid",
            "explicit_periods_per_year",
        )
    else:
        annualization = resolve_annualization(interval=interval, timestamps=timestamps)

    if annualization.status != "valid" or annualization.periods_per_year is None:
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            annualization.status,
            annualization.reason,
        )
    if values.empty:
        return SharpeCalculation(
            float("nan"),
            annualization.interval,
            annualization.periods_per_year,
            "blocked",
            "no_finite_returns",
        )
    standard_deviation = float(values.std(ddof=0))
    if standard_deviation == 0.0:
        return SharpeCalculation(
            0.0,
            annualization.interval,
            annualization.periods_per_year,
            "valid",
            "zero_volatility",
        )
    value = float(np.sqrt(annualization.periods_per_year) * values.mean() / standard_deviation)
    return SharpeCalculation(
        value,
        annualization.interval,
        annualization.periods_per_year,
        "valid",
        annualization.reason,
    )


def _median_interval_seconds(
    timestamps: Iterable[object] | pd.Series | pd.Index | None,
) -> float | None:
    if timestamps is None:
        return None
    parsed = pd.Series(pd.to_datetime(list(timestamps), utc=True, errors="coerce")).dropna().drop_duplicates().sort_values()
    if len(parsed) < 3:
        return None
    deltas = parsed.diff().dropna().dt.total_seconds()
    deltas = deltas[deltas > 0]
    if deltas.empty:
        return None
    median = float(deltas.median())
    # A heavily irregular series must not receive a precise-looking Sharpe.
    relative_deviation = (deltas - median).abs() / median
    if float((relative_deviation <= 0.05).mean()) < 0.80:
        return None
    return median


def _canonical_interval_for_seconds(seconds: float) -> str | None:
    expected = {
        "1m": 60.0,
        "5m": 300.0,
        "15m": 900.0,
        "1h": 3_600.0,
        "4h": 14_400.0,
        "1d": 86_400.0,
    }
    for interval, target in expected.items():
        if abs(seconds - target) / target <= 0.05:
            return interval
    return None
