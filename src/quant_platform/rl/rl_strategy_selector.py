from __future__ import annotations

import pandas as pd


def approved_strategy_choices(strategy_tests: pd.DataFrame) -> pd.DataFrame:
    if strategy_tests.empty:
        return pd.DataFrame(columns=["pair", "timeframe", "strategy", "accepted"])
    frame = strategy_tests.copy()
    accepted_col = "accepted" if "accepted" in frame.columns else "production_eligible"
    if accepted_col not in frame.columns:
        frame[accepted_col] = False
    return frame[frame[accepted_col].astype(bool)].copy()
