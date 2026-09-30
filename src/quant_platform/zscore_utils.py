from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
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


def coalesce_zscore(
    frame: pd.DataFrame,
    *,
    prefer_reconstructed: bool = True,
    prefer_provider: bool = True,
    prefer_rolling: bool = True,
    source_column: str | None = None,
) -> pd.Series:
    """Return the canonical z-score stream for a data frame.

    Preference:
    1) reconstructed z-score,
    2) provided z-score,
    3) rolling z-score,
    4) reconstructed from spread as rolling z-score.

    Select a declared source from the input schema rather than the values in a
    current or future row. Missing values in the selected stream remain missing.
    A growing schema must bind its source before evaluating historical prefixes.
    """

    declared = frame.attrs.get("zscore_source_column")
    if source_column is not None and declared is not None and source_column != declared:
        raise ValueError("explicit z-score source conflicts with the frame binding")
    selected = source_column if source_column is not None else declared
    allowed = {"zscore_reconstructed", "zscore", "rolling_zscore", "spread"}
    if selected is not None:
        if not isinstance(selected, str) or selected not in allowed:
            raise ValueError("unsupported declared z-score source column")
        if selected not in frame.columns:
            result = pd.Series(math.nan, index=frame.index, dtype=float)
        elif selected == "spread":
            result = rolling_zscore(frame[selected])
        else:
            values = pd.to_numeric(frame[selected], errors="coerce")
            result = values.where(np.isfinite(values))
        result.attrs["zscore_source_column"] = selected
        return result

    for enabled, column in (
        (prefer_reconstructed, "zscore_reconstructed"),
        (prefer_provider, "zscore"),
        (prefer_rolling, "rolling_zscore"),
    ):
        if enabled and column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            result = values.where(np.isfinite(values))
            result.attrs["zscore_source_column"] = column
            return result

    if "spread" in frame.columns:
        result = rolling_zscore(frame["spread"])
        result.attrs["zscore_source_column"] = "spread"
        return result

    return pd.Series([math.nan] * len(frame), index=frame.index)
