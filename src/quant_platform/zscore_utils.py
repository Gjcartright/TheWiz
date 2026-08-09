from __future__ import annotations

from dataclasses import dataclass
import math
import pandas as pd


ZSCORE_WINDOW = 7
ZSCORE_MIN_PERIODS = 7


@dataclass(frozen=True)
class ZScoreParams:
    window: int = ZSCORE_WINDOW
    min_periods: int = ZSCORE_MIN_PERIODS


def rolling_zscore(values: pd.Series | list[float] | list[object], *, window: int = ZSCORE_WINDOW, min_periods: int = ZSCORE_MIN_PERIODS) -> pd.Series:
    """Compute rolling z-score from spread-like values.

    Uses the windowed mean and std of the provided spread input.
    """

    spread = pd.to_numeric(pd.Series(values), errors="coerce")
    window_value = max(1, int(window))
    min_periods_value = max(1, min(int(min_periods), window_value))
    rolling_mean = spread.rolling(window_value, min_periods=min_periods_value).mean()
    rolling_std = spread.rolling(window_value, min_periods=min_periods_value).std()
    return (spread - rolling_mean) / rolling_std


def attach_row_level_zscores(rows: list[dict], *, spread_key: str = "spread", zscore_key: str = "zscore", window: int = ZSCORE_WINDOW, min_periods: int = ZSCORE_MIN_PERIODS) -> None:
    """Mutate row dictionaries in-place with computed rolling z-scores."""

    if not rows:
        return
    if not all(isinstance(row, dict) and spread_key in row for row in rows):
        return

    frame = pd.DataFrame(rows)
    if spread_key not in frame.columns:
        return
    scores = rolling_zscore(frame[spread_key], window=window, min_periods=min_periods)
    for idx, row in enumerate(rows):
        row[zscore_key] = float(scores.iloc[idx]) if pd.notna(scores.iloc[idx]) else float("nan")


def coalesce_zscore(frame: pd.DataFrame, *, prefer_reconstructed: bool = True, prefer_provider: bool = True, prefer_rolling: bool = True) -> pd.Series:
    """Return the canonical z-score stream for a data frame.

    Preference:
    1) reconstructed z-score,
    2) provided z-score,
    3) rolling z-score,
    4) reconstructed from spread as rolling z-score.
    """

    if prefer_reconstructed and "zscore_reconstructed" in frame.columns:
        reconstructed = pd.to_numeric(frame["zscore_reconstructed"], errors="coerce")
        if reconstructed.dropna().any():
            return reconstructed

    if prefer_provider and "zscore" in frame.columns:
        provided = pd.to_numeric(frame["zscore"], errors="coerce")
        if provided.dropna().any():
            return provided

    if prefer_rolling and "rolling_zscore" in frame.columns:
        rolling = pd.to_numeric(frame["rolling_zscore"], errors="coerce")
        if rolling.dropna().any():
            return rolling

    if "spread" in frame.columns:
        return rolling_zscore(frame["spread"])

    return pd.Series([math.nan] * len(frame), index=frame.index)
