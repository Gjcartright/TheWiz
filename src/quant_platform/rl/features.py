from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "zscore",
    "rolling_zscore",
    "spread",
    "spread_slope",
    "realized_volatility_percentile",
    "correlation",
    "hedge_ratio",
    "hedge_ratio_stability",
    "beta",
    "beta_stability",
    "copula_calibration_score",
    "ecm_x",
    "ecm_y",
    "ecm_strength",
    "funding_bps_per_day",
    "crisis_probability",
    "liquidity_score",
    "trade_quality_score",
]

FUTURE_ONLY_COLUMNS = {
    "good_trade",
    "profit_after_cost",
    "max_adverse_excursion",
    "max_favorable_excursion",
    "label_timestamp",
    "exit_timestamp",
    "exit_reason",
    "future_return",
    "realized_return",
}


def leakage_columns(columns: list[str] | pd.Index) -> list[str]:
    lowered = {str(column).lower(): str(column) for column in columns}
    return [original for key, original in lowered.items() if key in FUTURE_ONLY_COLUMNS]


def build_rl_feature_frame(frame: pd.DataFrame, *, allow_future_columns: bool = False) -> pd.DataFrame:
    leaked = leakage_columns(frame.columns)
    if leaked and not allow_future_columns:
        raise ValueError(f"rl_feature_leakage_columns:{','.join(sorted(leaked))}")
    features = pd.DataFrame(index=frame.index)
    for column in FEATURE_COLUMNS:
        if column in frame.columns:
            features[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            features[column] = 0.0
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float64")


def write_feature_schema(path: Path, columns: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "rl_policy_v1",
        "features": columns or FEATURE_COLUMNS,
        "label_source": "backtest_trained",
        "live_enabled": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path
