from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class AllocationRule:
    max_pair_weight: float = 0.08
    max_gross_exposure: float = 1.5
    min_score: float = 70.0


def composite_score(score_frame: pd.DataFrame) -> pd.Series:
    weights = {
        "cointegration_score": 0.10,
        "mean_reversion_score": 0.15,
        "copula_dislocation_score": 0.15,
        "tail_risk_score": 0.15,
        "ecm_score": 0.10,
        "backtest_quality_score": 0.15,
        "proprietary_signal_score": 0.08,
        "regime_score": 0.07,
        "execution_quality_score": 0.05,
    }
    missing = [col for col in weights if col not in score_frame]
    if missing:
        raise ValueError(f"missing score columns: {missing}")
    return sum(score_frame[col].astype(float) * weight for col, weight in weights.items())


def allocate(score_frame: pd.DataFrame, rule: AllocationRule | None = None) -> pd.DataFrame:
    rule = rule or AllocationRule()
    ranked = score_frame.copy()
    ranked["composite_score"] = composite_score(ranked)
    ranked = ranked[ranked["composite_score"] >= rule.min_score].sort_values("composite_score", ascending=False)
    if ranked.empty:
        ranked["weight"] = []
        return ranked
    raw = ranked["composite_score"] / ranked["composite_score"].sum()
    ranked["weight"] = raw.clip(upper=rule.max_pair_weight)
    total = ranked["weight"].sum()
    if total > rule.max_gross_exposure:
        ranked["weight"] *= rule.max_gross_exposure / total
    return ranked

