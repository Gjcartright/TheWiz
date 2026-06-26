from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FieldRole(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"
    RISK = "risk"
    RANKING = "ranking"
    FILTER = "filter"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class FieldDefinition:
    name: str
    category: str
    role: FieldRole
    direction: str
    description: str
    expected_range: str
    required_tests: tuple[str, ...]


FIELD_DEFINITIONS: tuple[FieldDefinition, ...] = (
    FieldDefinition("cointegration", "relationship", FieldRole.FILTER, "higher_is_better", "Evidence that a linear spread is stationary.", "boolean or p-value/statistic", ("walk_forward_stability", "false_discovery_control", "regime_split")),
    FieldDefinition("hedge_ratio", "relationship", FieldRole.DIAGNOSTIC, "contextual", "Position ratio that defines the spread between assets.", "real number", ("rolling_stability", "trade_pnl_sensitivity")),
    FieldDefinition("beta", "relationship", FieldRole.RISK, "contextual", "Market or pair sensitivity used for exposure normalization.", "real number", ("exposure_neutrality", "drawdown_attribution")),
    FieldDefinition("ecm_x", "ecm", FieldRole.ENTRY, "signed", "Error correction term or adjustment coefficient for asset X.", "real number", ("granger_incremental_value", "lead_lag_stability")),
    FieldDefinition("ecm_y", "ecm", FieldRole.ENTRY, "signed", "Error correction term or adjustment coefficient for asset Y.", "real number", ("granger_incremental_value", "lead_lag_stability")),
    FieldDefinition("ecm_strength", "ecm", FieldRole.RANKING, "higher_is_better", "Magnitude and reliability of error correction.", "0-1 or real number", ("incremental_alpha", "half_life_consistency")),
    FieldDefinition("half_life", "mean_reversion", FieldRole.FILTER, "bounded_optimal", "Estimated time for spread deviation to decay by half.", "positive real number", ("holding_period_fit", "capacity_vs_decay")),
    FieldDefinition("hurst", "mean_reversion", FieldRole.FILTER, "lower_is_better_below_0_5", "Long memory statistic; below 0.5 suggests mean reversion.", "0-1", ("regime_conditioned_predictiveness", "lookback_sensitivity")),
    FieldDefinition("zscore", "mean_reversion", FieldRole.ENTRY, "absolute_higher", "Standardized spread deviation.", "real number", ("threshold_sweep", "decay_after_signal")),
    FieldDefinition("rolling_zscore", "mean_reversion", FieldRole.ENTRY, "absolute_higher", "Z-score computed on rolling windows.", "real number", ("window_sensitivity", "threshold_sweep")),
    FieldDefinition("spread", "mean_reversion", FieldRole.DIAGNOSTIC, "signed", "Hedged price differential.", "real number", ("stationarity", "jump_risk")),
    FieldDefinition("pearson", "dependence", FieldRole.FILTER, "absolute_higher", "Linear correlation.", "-1 to 1", ("nonlinear_incremental_value", "breakdown_detection")),
    FieldDefinition("spearman", "dependence", FieldRole.FILTER, "absolute_higher", "Rank correlation.", "-1 to 1", ("monotonic_dependence", "tail_robustness")),
    FieldDefinition("kendall", "dependence", FieldRole.FILTER, "absolute_higher", "Concordance-based rank dependence.", "-1 to 1", ("copula_consistency", "small_sample_stability")),
    FieldDefinition("copula", "copula", FieldRole.RANKING, "contextual", "Joint distribution model or copula family fit.", "categorical/parameters", ("tail_dependence_alpha", "calibration_error")),
    FieldDefinition("conditional_probabilities", "copula", FieldRole.ENTRY, "distortion_higher", "Conditional probability distortion between observed and expected co-movement.", "0-1", ("probability_calibration", "trade_outcome_prediction")),
    FieldDefinition("sharpe", "backtest", FieldRole.RANKING, "higher_is_better", "Annualized excess return per volatility.", "real number", ("deflated_sharpe", "walk_forward")),
    FieldDefinition("sortino", "backtest", FieldRole.RANKING, "higher_is_better", "Return per downside volatility.", "real number", ("downside_robustness", "walk_forward")),
    FieldDefinition("var", "risk", FieldRole.RISK, "lower_is_better", "Value-at-risk loss estimate.", "positive loss", ("tail_backtest", "exception_rate")),
    FieldDefinition("cvar", "risk", FieldRole.RISK, "lower_is_better", "Expected shortfall beyond VaR.", "positive loss", ("tail_backtest", "stress_periods")),
    FieldDefinition("drawdown", "risk", FieldRole.RISK, "lower_is_better", "Peak-to-trough equity loss.", "0-1", ("max_dd_constraint", "recovery_time")),
    FieldDefinition("win_rate", "backtest", FieldRole.DIAGNOSTIC, "contextual", "Fraction of winning trades.", "0-1", ("payoff_ratio_joint_test", "stability")),
    FieldDefinition("ml_confidence", "ml", FieldRole.RANKING, "calibrated_higher", "Model-estimated probability or confidence for favorable outcome.", "0-1", ("calibration", "feature_leakage", "walk_forward")),
    FieldDefinition("profile_match", "ml", FieldRole.FILTER, "higher_is_better", "Similarity to historically successful trade profiles.", "0-1", ("nearest_neighbor_outcomes", "regime_split")),
    FieldDefinition("ou_optimal", "mean_reversion", FieldRole.ENTRY, "higher_is_better", "Ornstein-Uhlenbeck optimal entry/exit assessment.", "real number or 0-1", ("ou_parameter_stability", "threshold_sweep")),
)


def field_rows() -> list[dict[str, str]]:
    return [
        {
            "name": f.name,
            "category": f.category,
            "role": f.role.value,
            "direction": f.direction,
            "description": f.description,
            "expected_range": f.expected_range,
            "required_tests": ";".join(f.required_tests),
        }
        for f in FIELD_DEFINITIONS
    ]

