from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


SignalFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class StrategySpec:
    id: int
    name: str
    family: str
    hypothesis: str
    primary_fields: tuple[str, ...]
    required_tests: tuple[str, ...]
    signal_function: SignalFunction | None = None


def zscore_signal(frame: pd.DataFrame, entry: float = 2.0, exit_: float = 0.25) -> pd.Series:
    z = frame["zscore"].astype(float)
    signal = pd.Series(0.0, index=frame.index)
    signal[z > entry] = -1.0
    signal[z < -entry] = 1.0
    signal[z.abs() < exit_] = 0.0
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def _stateful(signal: pd.Series) -> pd.Series:
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _base_zscore_direction(frame: pd.DataFrame, entry: float = 2.0, exit_: float = 0.25) -> pd.Series:
    z = _numeric(frame, "zscore")
    signal = pd.Series(0.0, index=frame.index)
    signal[z > entry] = -1.0
    signal[z < -entry] = 1.0
    signal[z.abs() < exit_] = 0.0
    return signal


def copula_signal(frame: pd.DataFrame, threshold: float = 0.20) -> pd.Series:
    distortion = frame["conditional_probability_distortion"].astype(float)
    signal = pd.Series(0.0, index=frame.index)
    signal[distortion > threshold] = -1.0
    signal[distortion < -threshold] = 1.0
    return signal.replace(0.0, np.nan).ffill().fillna(0.0)


def zscore_ecm_signal(frame: pd.DataFrame) -> pd.Series:
    ecm_strength = _numeric(frame, "ecm_strength")
    return _stateful(_base_zscore_direction(frame).where(ecm_strength >= 0.5, 0.0))


def zscore_copula_signal(frame: pd.DataFrame) -> pd.Series:
    z_signal = _base_zscore_direction(frame)
    copula = _numeric(frame, "conditional_probability_distortion")
    copula_signal_raw = pd.Series(0.0, index=frame.index)
    copula_signal_raw[copula > 0.20] = -1.0
    copula_signal_raw[copula < -0.20] = 1.0
    return _stateful(z_signal.where(z_signal == copula_signal_raw, 0.0))


def ecm_copula_zscore_signal(frame: pd.DataFrame) -> pd.Series:
    ecm_strength = _numeric(frame, "ecm_strength")
    return _stateful(zscore_copula_signal(frame).where(ecm_strength >= 0.5, 0.0))


def dual_conditional_copula_signal(frame: pd.DataFrame) -> pd.Series:
    spread = _numeric(frame, "u1_given_u2") - _numeric(frame, "u2_given_u1")
    signal = pd.Series(0.0, index=frame.index)
    signal[spread > 0.25] = -1.0
    signal[spread < -0.25] = 1.0
    return _stateful(signal)


def tail_event_reversion_signal(frame: pd.DataFrame) -> pd.Series:
    distortion = _numeric(frame, "conditional_probability_distortion")
    tail = _numeric(frame, "tail_dependence", 0.5)
    signal = pd.Series(0.0, index=frame.index)
    signal[(distortion > 0.30) & (tail < 0.65)] = -1.0
    signal[(distortion < -0.30) & (tail < 0.65)] = 1.0
    return _stateful(signal)


def pure_ecm_signal(frame: pd.DataFrame) -> pd.Series:
    strength = _numeric(frame, "ecm_strength")
    adjustment_gap = _numeric(frame, "ecm_y") - _numeric(frame, "ecm_x")
    signal = pd.Series(0.0, index=frame.index)
    signal[(adjustment_gap > 0.10) & (strength >= 0.5)] = 1.0
    signal[(adjustment_gap < -0.10) & (strength >= 0.5)] = -1.0
    return _stateful(signal)


def ecm_leadership_signal(frame: pd.DataFrame) -> pd.Series:
    x = _numeric(frame, "ecm_x")
    y = _numeric(frame, "ecm_y")
    strength = _numeric(frame, "ecm_strength", 0.5)
    signal = pd.Series(0.0, index=frame.index)
    signal[(x.abs() > y.abs() * 1.5) & (strength >= 0.45)] = 1.0
    signal[(y.abs() > x.abs() * 1.5) & (strength >= 0.45)] = -1.0
    return _stateful(signal)


def half_life_optimized_signal(frame: pd.DataFrame) -> pd.Series:
    half_life = _numeric(frame, "half_life", 999.0)
    z = _numeric(frame, "zscore")
    dynamic_entry = (1.5 + (half_life / 48.0)).clip(lower=1.5, upper=3.0)
    signal = pd.Series(0.0, index=frame.index)
    signal[z > dynamic_entry] = -1.0
    signal[z < -dynamic_entry] = 1.0
    signal[z.abs() < 0.25] = 0.0
    return _stateful(signal)


def hurst_filter_signal(frame: pd.DataFrame) -> pd.Series:
    hurst = _numeric(frame, "hurst", 0.5)
    return _stateful(_base_zscore_direction(frame).where(hurst < 0.45, 0.0))


def hurst_half_life_signal(frame: pd.DataFrame) -> pd.Series:
    hurst = _numeric(frame, "hurst", 0.5)
    half_life = _numeric(frame, "half_life", 999.0)
    tradable_decay = half_life.between(2, 48)
    return _stateful(_base_zscore_direction(frame).where((hurst < 0.45) & tradable_decay, 0.0))


def ou_optimal_signal(frame: pd.DataFrame) -> pd.Series:
    ou = _numeric(frame, "ou_optimal", 0.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(ou >= 0.55, 0.0))


def ml_confidence_signal(frame: pd.DataFrame) -> pd.Series:
    confidence = _numeric(frame, "ml_confidence", 0.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(confidence >= 0.60, 0.0))


def profile_match_signal(frame: pd.DataFrame) -> pd.Series:
    profile = _numeric(frame, "profile_match", 0.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(profile >= 0.60, 0.0))


def proprietary_stack_signal(frame: pd.DataFrame) -> pd.Series:
    score = (
        _numeric(frame, "ml_confidence", 0.0)
        + _numeric(frame, "profile_match", 0.0)
        + _numeric(frame, "ou_optimal", 0.0)
    ) / 3.0
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(score >= 0.60, 0.0))


def composite_quant_score_signal(frame: pd.DataFrame) -> pd.Series:
    score = _numeric(frame, "composite_score", 0.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(score >= 70.0, 0.0))


def weighted_voting_signal(frame: pd.DataFrame) -> pd.Series:
    votes = (
        (_numeric(frame, "cointegration_score", 0.0) >= 70).astype(int)
        + (_numeric(frame, "ecm_score", 0.0) >= 60).astype(int)
        + (_numeric(frame, "copula_dislocation_score", 0.0) >= 60).astype(int)
    )
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(votes >= 2, 0.0))


def dynamic_threshold_signal(frame: pd.DataFrame) -> pd.Series:
    hurst = _numeric(frame, "hurst", 0.5)
    half_life = _numeric(frame, "half_life", 24.0)
    ecm = _numeric(frame, "ecm_strength", 0.5)
    z = _numeric(frame, "zscore")
    entry = (2.2 - ecm * 0.4 - (0.5 - hurst).clip(lower=0.0) - (24.0 - half_life).clip(lower=0.0) / 48.0).clip(
        lower=1.25, upper=3.0
    )
    signal = pd.Series(0.0, index=frame.index)
    signal[z > entry] = -1.0
    signal[z < -entry] = 1.0
    signal[z.abs() < 0.25] = 0.0
    return _stateful(signal)


def regime_filtered_signal(frame: pd.DataFrame) -> pd.Series:
    if "regime" not in frame.columns:
        return _stateful(_base_zscore_direction(frame))
    favorable = frame["regime"].astype(str).str.lower().isin({"range", "bull"})
    return _stateful(_base_zscore_direction(frame).where(favorable, 0.0))


def copula_tail_risk_sizing_signal(frame: pd.DataFrame) -> pd.Series:
    tail = _numeric(frame, "tail_dependence", 0.5)
    return _stateful(copula_signal(frame).where(tail < 0.65, 0.0))


def copula_risk_filter_signal(frame: pd.DataFrame) -> pd.Series:
    tail = _numeric(frame, "tail_dependence", 0.5)
    return _stateful(copula_signal(frame).where(tail < 0.50, 0.0))


def copula_dislocation_ranking_signal(frame: pd.DataFrame) -> pd.Series:
    return copula_signal(frame, threshold=0.15)


def copula_regime_signal(frame: pd.DataFrame) -> pd.Series:
    if "regime" not in frame.columns:
        return copula_signal(frame)
    favorable = frame["regime"].astype(str).str.lower().isin({"range", "bull"})
    return _stateful(copula_signal(frame).where(favorable, 0.0))


def copula_persistence_signal(frame: pd.DataFrame) -> pd.Series:
    distortion = _numeric(frame, "conditional_probability_distortion")
    persistent = distortion.rolling(3, min_periods=1).mean()
    signal = pd.Series(0.0, index=frame.index)
    signal[persistent > 0.18] = -1.0
    signal[persistent < -0.18] = 1.0
    return _stateful(signal)


def copula_ecm_signal(frame: pd.DataFrame) -> pd.Series:
    ecm = _numeric(frame, "ecm_strength", 0.0)
    return _stateful(copula_signal(frame).where(ecm >= 0.50, 0.0))


def _return_proxy(frame: pd.DataFrame) -> pd.Series:
    for column in ("market_return", "benchmark_return", "returns", "return"):
        if column in frame.columns:
            return _numeric(frame, column)
    if "spread" in frame.columns:
        return _numeric(frame, "spread").diff().fillna(0.0)
    return pd.Series(0.0, index=frame.index)


def hmm_regime_signal(frame: pd.DataFrame) -> pd.Series:
    returns = _return_proxy(frame)
    vol = returns.rolling(10, min_periods=2).std().fillna(0.0)
    calm = vol <= vol.rolling(50, min_periods=2).quantile(0.60).fillna(vol.median())
    return _stateful(_base_zscore_direction(frame).where(calm, 0.0))


def gmm_regime_signal(frame: pd.DataFrame) -> pd.Series:
    returns = _return_proxy(frame)
    trend = returns.rolling(10, min_periods=2).sum().fillna(0.0)
    favorable = trend.abs() < trend.abs().rolling(50, min_periods=2).quantile(0.70).fillna(trend.abs().median())
    return _stateful(_base_zscore_direction(frame).where(favorable, 0.0))


def kmeans_regime_signal(frame: pd.DataFrame) -> pd.Series:
    returns = _return_proxy(frame)
    vol = returns.rolling(10, min_periods=2).std().fillna(0.0)
    trend = returns.rolling(10, min_periods=2).sum().fillna(0.0)
    favorable = (vol.rank(pct=True) < 0.75) & (trend.abs().rank(pct=True) < 0.75)
    return _stateful(_base_zscore_direction(frame).where(favorable, 0.0))


def pair_ranking_signal(frame: pd.DataFrame) -> pd.Series:
    score = _numeric(frame, "composite_score", 75.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(score >= 70.0, 0.0))


def portfolio_rotation_signal(frame: pd.DataFrame) -> pd.Series:
    score = _numeric(frame, "composite_score", 75.0)
    drawdown = _numeric(frame, "drawdown", 0.0)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where((score >= 70.0) & (drawdown <= 0.15), 0.0))


def risk_adjusted_ranking_signal(frame: pd.DataFrame) -> pd.Series:
    sharpe = _numeric(frame, "sharpe", 1.5)
    cvar = _numeric(frame, "cvar", 0.05)
    drawdown = _numeric(frame, "drawdown", 0.05)
    favorable = (sharpe >= 1.2) & (cvar <= 0.10) & (drawdown <= 0.15)
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(favorable, 0.0))


def meta_model_proxy_signal(frame: pd.DataFrame) -> pd.Series:
    confidence = (
        _numeric(frame, "ml_confidence", 0.55)
        + _numeric(frame, "profile_match", 0.55)
        + _numeric(frame, "ou_optimal", 0.55)
    ) / 3.0
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(confidence >= 0.58, 0.0))


def feature_importance_proxy_signal(frame: pd.DataFrame) -> pd.Series:
    weighted = (
        _numeric(frame, "ecm_strength", 0.5) * 0.35
        + _numeric(frame, "tail_dependence", 0.3).rsub(1.0) * 0.25
        + _numeric(frame, "hurst", 0.5).rsub(0.5).clip(lower=0.0) * 0.80
        + _numeric(frame, "ml_confidence", 0.55) * 0.20
    )
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(weighted >= 0.45, 0.0))


def trade_outcome_predictor_proxy_signal(frame: pd.DataFrame) -> pd.Series:
    probability = _numeric(frame, "trade_success_probability", np.nan)
    if probability.isna().all():
        probability = (
            _numeric(frame, "ml_confidence", 0.55)
            + _numeric(frame, "profile_match", 0.55)
            + _numeric(frame, "ecm_strength", 0.55)
        ) / 3.0
    return _stateful(_base_zscore_direction(frame, entry=1.5).where(probability >= 0.58, 0.0))


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(1, "Classic ZScore Mean Reversion", "zscore", "Spread extremes revert after costs.", ("zscore", "spread", "hedge_ratio"), ("threshold_sweep", "walk_forward"), zscore_signal),
    StrategySpec(2, "ZScore + ECM", "hybrid", "Z-score entries improve when ECM confirms correction.", ("zscore", "ecm_strength", "ecm_x", "ecm_y"), ("incremental_alpha", "ablation"), zscore_ecm_signal),
    StrategySpec(3, "ZScore + Copula", "hybrid", "Linear spread dislocation improves when joint probability is distorted.", ("zscore", "conditional_probabilities", "copula"), ("ablation", "calibration"), zscore_copula_signal),
    StrategySpec(4, "ECM + Copula + ZScore", "hybrid", "Three independent views agree on disequilibrium.", ("zscore", "ecm_strength", "conditional_probabilities"), ("interaction_effects", "walk_forward"), ecm_copula_zscore_signal),
    StrategySpec(5, "Pure Copula", "copula", "Conditional probability distortion is alpha without spread z-score.", ("copula", "conditional_probabilities"), ("calibration", "alpha_without_zscore"), copula_signal),
    StrategySpec(6, "Dual Conditional Copula", "copula", "Both conditional directions identify asymmetric mispricing.", ("conditional_probabilities", "kendall"), ("directional_calibration",), dual_conditional_copula_signal),
    StrategySpec(7, "Tail Event Reversion", "copula", "Tail co-movement extremes mean-revert when systemic risk is contained.", ("tail_dependence", "cvar", "conditional_probabilities"), ("stress_split",), tail_event_reversion_signal),
    StrategySpec(8, "Pure ECM", "ecm", "Adjustment coefficients predict next leg movement.", ("ecm_x", "ecm_y", "ecm_strength"), ("granger_incremental_value",), pure_ecm_signal),
    StrategySpec(9, "ECM Leadership", "ecm", "Leader/follower asymmetry predicts follower catch-up.", ("ecm_x", "ecm_y"), ("lead_lag_stability",), ecm_leadership_signal),
    StrategySpec(10, "Leader/Follower Prediction", "ecm", "One asset leads the equilibrium correction.", ("ecm_x", "ecm_y", "beta"), ("directional_accuracy",), ecm_leadership_signal),
    StrategySpec(11, "Half-Life Optimized", "mean_reversion", "Entries work best when holding horizon matches decay.", ("half_life", "zscore"), ("holding_period_fit",), half_life_optimized_signal),
    StrategySpec(12, "Hurst Filter", "mean_reversion", "Mean-reversion signals work only under anti-persistent regimes.", ("hurst", "zscore"), ("filter_ablation",), hurst_filter_signal),
    StrategySpec(13, "Hurst + Half-Life", "mean_reversion", "Anti-persistence plus tradable decay improves z-score.", ("hurst", "half_life", "zscore"), ("interaction_effects",), hurst_half_life_signal),
    StrategySpec(14, "OU Optimal", "mean_reversion", "OU optimal stopping beats static thresholds.", ("ou_optimal", "half_life", "spread"), ("threshold_comparison",), ou_optimal_signal),
    StrategySpec(15, "ML Confidence", "ml", "Calibrated model confidence ranks profitable trades.", ("ml_confidence",), ("calibration", "leakage_audit"), ml_confidence_signal),
    StrategySpec(16, "Profile Match", "ml", "Historical winner profile similarity predicts outcome.", ("profile_match",), ("nearest_neighbor_validation",), profile_match_signal),
    StrategySpec(17, "Proprietary Signal Stack", "composite", "Internal feature stack improves robust ranking.", ("ml_confidence", "profile_match", "ou_optimal"), ("ablation",), proprietary_stack_signal),
    StrategySpec(18, "Composite Quant Score", "composite", "Explainable factor blend ranks opportunities.", ("composite_score",), ("weight_sensitivity",), composite_quant_score_signal),
    StrategySpec(19, "Weighted Voting Model", "composite", "Independent model votes reduce false positives.", ("cointegration_score", "ecm_score", "copula_dislocation_score"), ("vote_ablation",), weighted_voting_signal),
    StrategySpec(20, "Dynamic Threshold Model", "adaptive", "Thresholds adapt to volatility/regime.", ("zscore", "realized_volatility_percentile", "regime"), ("regime_walk_forward",), dynamic_threshold_signal),
    StrategySpec(21, "HMM Regime Model", "regime", "Hidden-state regimes condition strategy selection.", ("returns", "volatility"), ("state_stability",), hmm_regime_signal),
    StrategySpec(22, "GMM Regime Model", "regime", "Distribution clusters identify market regimes.", ("returns", "volatility"), ("cluster_stability",), gmm_regime_signal),
    StrategySpec(23, "KMeans Regime Model", "regime", "Feature clusters separate favorable and hostile states.", ("returns", "volatility"), ("cluster_stability",), kmeans_regime_signal),
    StrategySpec(24, "Regime Filtered Stat-Arb", "regime", "Stat-arb trades only in regimes with positive expectancy.", ("regime_strategy_match",), ("regime_split",), regime_filtered_signal),
    StrategySpec(25, "Pair Ranking Strategy", "portfolio", "Top-ranked pairs outperform all signals.", ("composite_score",), ("cross_sectional_rank_ic",), pair_ranking_signal),
    StrategySpec(26, "Portfolio Rotation", "portfolio", "Capital rotates to strongest pairs as edge decays.", ("composite_score", "drawdown"), ("turnover_costs",), portfolio_rotation_signal),
    StrategySpec(27, "Risk Adjusted Ranking", "portfolio", "Ranking by return per tail risk improves portfolio PF.", ("sharpe", "cvar", "drawdown"), ("risk_adjusted_ic",), risk_adjusted_ranking_signal),
    StrategySpec(28, "Copula Tail Risk Sizing", "portfolio", "Tail dependence controls size.", ("tail_dependence", "cvar"), ("sizing_ablation",), copula_tail_risk_sizing_signal),
    StrategySpec(29, "Copula Risk Filter", "copula", "Avoid trades when copula implies crash co-dependence.", ("copula", "tail_dependence"), ("tail_filter_ablation",), copula_risk_filter_signal),
    StrategySpec(30, "Meta Model", "ml", "Trade outcome model improves selection.", ("all_features",), ("walk_forward", "calibration"), meta_model_proxy_signal),
    StrategySpec(31, "Feature Importance Model", "ml", "Feature attribution identifies true edge drivers.", ("all_features",), ("permutation_importance",), feature_importance_proxy_signal),
    StrategySpec(32, "Trade Outcome Predictor", "ml", "Predict expected return/drawdown/PF per setup.", ("all_features",), ("target_separation",), trade_outcome_predictor_proxy_signal),
    StrategySpec(33, "Copula Dislocation Ranking Engine", "copula", "Largest calibrated copula dislocations outperform.", ("conditional_probabilities", "copula_calibration_score"), ("rank_ic",), copula_dislocation_ranking_signal),
    StrategySpec(34, "Copula + Regime Strategy", "copula", "Copula edge is regime dependent.", ("conditional_probabilities", "regime"), ("regime_interaction",), copula_regime_signal),
    StrategySpec(35, "Copula Divergence Persistence", "copula", "Persistence of probability distortion predicts trade timing.", ("conditional_probability_distortion",), ("persistence_alpha",), copula_persistence_signal),
    StrategySpec(36, "Copula + ECM Strategy", "hybrid", "Copula dislocation plus correction force improves entries.", ("conditional_probabilities", "ecm_strength"), ("ablation",), copula_ecm_signal),
    StrategySpec(37, "Pure Copula Portfolio", "portfolio", "Portfolio built entirely from copula-ranked dislocations.", ("copula_dislocation_score",), ("portfolio_walk_forward",), copula_dislocation_ranking_signal),
)


STRATEGY_REQUIRED_COLUMNS: dict[int, set[str]] = {
    1: {"spread", "zscore"},
    2: {"spread", "zscore", "ecm_strength"},
    3: {"spread", "zscore", "conditional_probability_distortion"},
    4: {"spread", "zscore", "ecm_strength", "conditional_probability_distortion"},
    5: {"spread", "conditional_probability_distortion"},
    6: {"spread", "u1_given_u2", "u2_given_u1"},
    7: {"spread", "conditional_probability_distortion", "tail_dependence"},
    8: {"spread", "ecm_x", "ecm_y", "ecm_strength"},
    9: {"spread", "ecm_x", "ecm_y"},
    10: {"spread", "ecm_x", "ecm_y"},
    11: {"spread", "zscore", "half_life"},
    12: {"spread", "zscore", "hurst"},
    13: {"spread", "zscore", "hurst", "half_life"},
    14: {"spread", "zscore", "ou_optimal"},
    15: {"spread", "zscore", "ml_confidence"},
    16: {"spread", "zscore", "profile_match"},
    17: {"spread", "zscore", "ml_confidence", "profile_match", "ou_optimal"},
    18: {"spread", "zscore", "composite_score"},
    19: {"spread", "zscore", "cointegration_score", "ecm_score", "copula_dislocation_score"},
    20: {"spread", "zscore", "hurst", "half_life", "ecm_strength"},
    21: {"spread", "zscore"},
    22: {"spread", "zscore"},
    23: {"spread", "zscore"},
    24: {"spread", "zscore", "regime"},
    25: {"spread", "zscore"},
    26: {"spread", "zscore"},
    27: {"spread", "zscore", "sharpe", "cvar", "drawdown"},
    28: {"spread", "conditional_probability_distortion", "tail_dependence"},
    29: {"spread", "conditional_probability_distortion", "tail_dependence"},
    30: {"spread", "zscore"},
    31: {"spread", "zscore"},
    32: {"spread", "zscore"},
    33: {"spread", "conditional_probability_distortion"},
    34: {"spread", "conditional_probability_distortion", "regime"},
    35: {"spread", "conditional_probability_distortion"},
    36: {"spread", "conditional_probability_distortion", "ecm_strength"},
    37: {"spread", "conditional_probability_distortion"},
}


def strategy_rows() -> list[dict[str, str]]:
    return [
        {
            "id": str(s.id),
            "name": s.name,
            "family": s.family,
            "hypothesis": s.hypothesis,
            "primary_fields": ";".join(s.primary_fields),
            "required_tests": ";".join(s.required_tests),
            "executable_signal": str(s.signal_function is not None),
        }
        for s in STRATEGIES
    ]
