# Quant Brain

## cointegration
- Measures: Evidence that a linear spread is stationary.
- Why it exists: Tests whether a pair has a persistent equilibrium relation.
- How it may create edge: Filter/rank pairs before mean-reversion strategies.
- When it fails: Breaks during regime shifts, structural changes, or multiple-testing overfit.
- Research role: filter
- Required tests: walk_forward_stability;false_discovery_control;regime_split

## hedge_ratio
- Measures: Position ratio that defines the spread between assets.
- Why it exists: Defines spread neutrality and leg sizing.
- How it may create edge: Position construction and spread measurement.
- When it fails: Unstable beta creates hidden directional exposure.
- Research role: diagnostic
- Required tests: rolling_stability;trade_pnl_sensitivity

## beta
- Measures: Market or pair sensitivity used for exposure normalization.
- Why it exists: Sensitivity to common factor or paired asset.
- How it may create edge: Exposure normalization and risk attribution.
- When it fails: Nonlinear exposure is missed by linear beta.
- Research role: risk
- Required tests: exposure_neutrality;drawdown_attribution

## ecm_x
- Measures: Error correction term or adjustment coefficient for asset X.
- Why it exists: Adjustment speed of X toward equilibrium.
- How it may create edge: Leader/follower and directional leg prediction.
- When it fails: Spurious adjustment under unstable cointegration.
- Research role: entry
- Required tests: granger_incremental_value;lead_lag_stability

## ecm_y
- Measures: Error correction term or adjustment coefficient for asset Y.
- Why it exists: Adjustment speed of Y toward equilibrium.
- How it may create edge: Leader/follower and directional leg prediction.
- When it fails: Coefficient sign flips across regimes.
- Research role: entry
- Required tests: granger_incremental_value;lead_lag_stability

## ecm_strength
- Measures: Magnitude and reliability of error correction.
- Why it exists: How forcefully the pair corrects deviations.
- How it may create edge: Rank mean-reversion candidates and holding period confidence.
- When it fails: High in-sample strength can be overfit or stale.
- Research role: ranking
- Required tests: incremental_alpha;half_life_consistency

## half_life
- Measures: Estimated time for spread deviation to decay by half.
- Why it exists: Expected decay horizon of spread shock.
- How it may create edge: Set entry horizon, max holding period, and threshold timing.
- When it fails: Invalid when spread is not stationary or phi is unstable.
- Research role: filter
- Required tests: holding_period_fit;capacity_vs_decay

## hurst
- Measures: Long memory statistic; below 0.5 suggests mean reversion.
- Why it exists: H < 0.5 suggests anti-persistence; H > 0.5 suggests trend persistence.
- How it may create edge: Filter mean-reversion vs trend regimes.
- When it fails: Sensitive to sample length, microstructure noise, and jumps.
- Research role: filter
- Required tests: regime_conditioned_predictiveness;lookback_sensitivity

## zscore
- Measures: Standardized spread deviation.
- Why it exists: Distance from estimated equilibrium.
- How it may create edge: Classic entry/exit trigger.
- When it fails: Large z-score may indicate structural break rather than opportunity.
- Research role: entry
- Required tests: threshold_sweep;decay_after_signal

## rolling_zscore
- Measures: Z-score computed on rolling windows.
- Why it exists: Adaptive deviation measure.
- How it may create edge: Threshold model with recent volatility adaptation.
- When it fails: Window choice can chase noise or lag breaks.
- Research role: entry
- Required tests: window_sensitivity;threshold_sweep

## spread
- Measures: Hedged price differential.
- Why it exists: Tradable disequilibrium between hedged legs.
- How it may create edge: Base series for stationarity, z-score, OU, and ECM.
- When it fails: Bad hedge ratio converts spread into directional bet.
- Research role: diagnostic
- Required tests: stationarity;jump_risk

## pearson
- Measures: Linear correlation.
- Why it exists: Linear co-movement.
- How it may create edge: Basic dependence filter and breakdown monitor.
- When it fails: Misses nonlinear and tail dependence.
- Research role: filter
- Required tests: nonlinear_incremental_value;breakdown_detection

## spearman
- Measures: Rank correlation.
- Why it exists: Monotonic dependence robust to nonlinear scaling.
- How it may create edge: Dependence confirmation.
- When it fails: Can miss asymmetric tail structure.
- Research role: filter
- Required tests: monotonic_dependence;tail_robustness

## kendall
- Measures: Concordance-based rank dependence.
- Why it exists: Rank concordance often tied to copula parameters.
- How it may create edge: Copula calibration and dependence stability.
- When it fails: No direct spread trading signal by itself.
- Research role: filter
- Required tests: copula_consistency;small_sample_stability

## copula
- Measures: Joint distribution model or copula family fit.
- Why it exists: Dependence structure independent of marginal distributions.
- How it may create edge: Conditional mispricing, tail dependence, and portfolio risk.
- When it fails: Wrong family or poor calibration creates false dislocation signals.
- Research role: ranking
- Required tests: tail_dependence_alpha;calibration_error

## conditional_probabilities
- Measures: Conditional probability distortion between observed and expected co-movement.
- Why it exists: Observed pair state vs expected conditional state.
- How it may create edge: Pure copula and dual conditional copula entries.
- When it fails: Probability distortions can persist instead of reverting.
- Research role: entry
- Required tests: probability_calibration;trade_outcome_prediction

## sharpe
- Measures: Annualized excess return per volatility.
- Why it exists: Risk-adjusted return quality.
- How it may create edge: Strategy ranking and acceptance tests.
- When it fails: Inflated by non-normal tails, serial correlation, and selection bias.
- Research role: ranking
- Required tests: deflated_sharpe;walk_forward

## sortino
- Measures: Return per downside volatility.
- Why it exists: Reward per downside volatility.
- How it may create edge: Ranking strategies with asymmetric outcomes.
- When it fails: Can ignore crash clustering until enough tail events occur.
- Research role: ranking
- Required tests: downside_robustness;walk_forward

## var
- Measures: Value-at-risk loss estimate.
- Why it exists: Expected threshold loss not exceeded most of the time.
- How it may create edge: Position sizing and risk gating.
- When it fails: Does not describe losses beyond the quantile.
- Research role: risk
- Required tests: tail_backtest;exception_rate

## cvar
- Measures: Expected shortfall beyond VaR.
- Why it exists: Tail severity estimate.
- How it may create edge: Tail-aware sizing and copula risk controls.
- When it fails: Requires enough tail observations or robust stress modeling.
- Research role: risk
- Required tests: tail_backtest;stress_periods

## drawdown
- Measures: Peak-to-trough equity loss.
- Why it exists: Capital impairment from peak.
- How it may create edge: Risk limits, strategy rejection, and kill switches.
- When it fails: Backtest drawdown underestimates unseen structural breaks.
- Research role: risk
- Required tests: max_dd_constraint;recovery_time

## win_rate
- Measures: Fraction of winning trades.
- Why it exists: Hit rate of completed trades.
- How it may create edge: Diagnostics with payoff ratio.
- When it fails: High win rate can hide rare large losses.
- Research role: diagnostic
- Required tests: payoff_ratio_joint_test;stability

## ml_confidence
- Measures: Model-estimated probability or confidence for favorable outcome.
- Why it exists: Meta-model belief in setup quality.
- How it may create edge: Ranking, filtering, and sizing after calibration.
- When it fails: Leakage, nonstationarity, and uncalibrated probabilities.
- Research role: ranking
- Required tests: calibration;feature_leakage;walk_forward

## profile_match
- Measures: Similarity to historically successful trade profiles.
- Why it exists: How much the current setup resembles known winners.
- How it may create edge: Meta-learning filter.
- When it fails: Historical clusters may not survive new regimes.
- Research role: filter
- Required tests: nearest_neighbor_outcomes;regime_split

## ou_optimal
- Measures: Ornstein-Uhlenbeck optimal entry/exit assessment.
- Why it exists: Model-based mean-reversion entry/exit attractiveness.
- How it may create edge: Threshold optimization and expected holding period.
- When it fails: OU assumptions fail under jumps, trends, or changing volatility.
- Research role: entry
- Required tests: ou_parameter_stability;threshold_sweep
