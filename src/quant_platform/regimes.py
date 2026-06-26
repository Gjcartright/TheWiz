from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


REGIME_LABELS = ("bull", "bear", "range", "crisis")


@dataclass(frozen=True)
class RegimeConfig:
    lookback: int = 20
    trend_threshold: float = 0.02
    crisis_vol_quantile: float = 0.85
    crisis_drawdown_threshold: float = 0.08
    return_columns: tuple[str, ...] = (
        "market_return",
        "benchmark_return",
        "asset_x_return",
        "return",
        "returns",
    )
    preserve_existing: bool = False


def _select_return_series(frame: pd.DataFrame, config: RegimeConfig) -> tuple[pd.Series, str]:
    for column in config.return_columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(0.0), column
    if "price" in frame.columns:
        price = pd.to_numeric(frame["price"], errors="coerce")
        return price.pct_change().fillna(0.0), "price_pct_change"
    if "spread" in frame.columns:
        spread = pd.to_numeric(frame["spread"], errors="coerce")
        scale = spread.abs().rolling(config.lookback, min_periods=2).median().replace(0, np.nan)
        return (spread.diff() / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0), "spread_change_scaled"
    return pd.Series(0.0, index=frame.index), "constant_zero"


def classify_regimes(frame: pd.DataFrame, config: RegimeConfig | None = None) -> pd.DataFrame:
    """Classify rows into bull, bear, range, or crisis regimes.

    The classifier is deliberately deterministic and explainable. It uses rolling
    trend, volatility, and drawdown from the best available return series.
    """

    config = config or RegimeConfig()
    classified = frame.copy()
    returns, source = _select_return_series(classified, config)
    rolling_return = returns.rolling(config.lookback, min_periods=2).sum().fillna(0.0)
    rolling_vol = returns.rolling(config.lookback, min_periods=2).std().fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = (equity.cummax() - equity) / equity.cummax().replace(0, np.nan)
    drawdown = drawdown.fillna(0.0)
    vol_threshold = float(rolling_vol.quantile(config.crisis_vol_quantile)) if len(rolling_vol) else 0.0

    labels = pd.Series("range", index=classified.index, dtype="object")
    labels[rolling_return > config.trend_threshold] = "bull"
    labels[rolling_return < -config.trend_threshold] = "bear"
    labels[(rolling_vol >= vol_threshold) & (drawdown >= config.crisis_drawdown_threshold)] = "crisis"

    classified["classified_regime"] = labels
    classified["regime_return_source"] = source
    classified["regime_rolling_return"] = rolling_return
    classified["regime_rolling_volatility"] = rolling_vol
    classified["regime_drawdown"] = drawdown
    if "regime" not in classified.columns or not config.preserve_existing:
        classified["regime"] = classified["classified_regime"]
    else:
        existing = classified["regime"].fillna("").astype(str).str.lower()
        missing = existing.isin({"", "nan", "none", "unknown"})
        classified.loc[missing, "regime"] = classified.loc[missing, "classified_regime"]
    return classified


def regime_distribution(frame: pd.DataFrame, regime_column: str = "regime") -> pd.DataFrame:
    if regime_column not in frame.columns or frame.empty:
        return pd.DataFrame(columns=["regime", "observations", "observation_share"])
    counts = frame[regime_column].value_counts(dropna=False).rename_axis("regime").reset_index(name="observations")
    counts["observation_share"] = counts["observations"] / counts["observations"].sum()
    return counts.sort_values(["observations", "regime"], ascending=[False, True])


def write_regime_dataset_report(datasets: list, output_path: str | Path, regime_column: str = "regime") -> Path:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        distribution = regime_distribution(dataset.frame, regime_column)
        for row in distribution.to_dict("records"):
            rows.append({"pair": dataset.pair, **row})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["pair", "regime", "observations", "observation_share"]).to_csv(output, index=False)
    return output


def regime_pair_strategy_report(results: pd.DataFrame) -> pd.DataFrame:
    evaluated = results[results["status"] == "evaluated"].copy()
    if evaluated.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "regime",
                "strategy_id",
                "strategy_name",
                "family",
                "runs",
                "eligible_runs",
                "median_profit_factor",
                "median_sharpe",
                "median_max_drawdown",
                "total_trades",
            ]
        )
    return (
        evaluated.groupby(["pair", "regime", "strategy_id", "strategy_name", "family"], as_index=False)
        .agg(
            runs=("status", "count"),
            eligible_runs=("eligible", "sum"),
            median_profit_factor=("profit_factor", "median"),
            median_sharpe=("sharpe", "median"),
            median_max_drawdown=("max_drawdown", "median"),
            total_trades=("trades", "sum"),
        )
        .sort_values(["pair", "regime", "eligible_runs", "median_profit_factor"], ascending=[True, True, False, False])
    )

