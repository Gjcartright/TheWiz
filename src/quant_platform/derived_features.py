from __future__ import annotations

import pandas as pd


def add_derived_beta_from_prices(
    frame: pd.DataFrame,
    *,
    price_x_column: str = "price_x",
    price_y_column: str = "price_y",
    beta_column: str = "beta",
    source_column: str = "beta_source",
    min_observations: int = 3,
) -> pd.DataFrame:
    """Add pair beta from leg return covariance when native beta is absent."""

    if beta_column in frame.columns or not {price_x_column, price_y_column}.issubset(frame.columns):
        return frame
    price_x = pd.to_numeric(frame[price_x_column], errors="coerce")
    price_y = pd.to_numeric(frame[price_y_column], errors="coerce")
    returns = pd.DataFrame(
        {
            "x": price_x.pct_change(fill_method=None),
            "y": price_y.pct_change(fill_method=None),
        }
    ).replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(returns) < min_observations:
        return frame
    variance_y = returns["y"].var(ddof=0)
    if pd.isna(variance_y) or float(variance_y) == 0.0:
        return frame
    beta = returns["x"].cov(returns["y"], ddof=0) / variance_y
    if pd.isna(beta):
        return frame
    output = frame.copy()
    output[beta_column] = float(beta)
    output[source_column] = "derived_from_price_returns"
    return output
