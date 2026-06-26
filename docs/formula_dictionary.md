# Formula Dictionary

## cointegration
- Formula: Engle-Granger/Johansen stationarity test on residual spread: y_t - beta*x_t.
- Market interpretation: Tests whether a pair has a persistent equilibrium relation.
- Use case: Filter/rank pairs before mean-reversion strategies.
- Failure mode: Breaks during regime shifts, structural changes, or multiple-testing overfit.

## hedge_ratio
- Formula: OLS/TLS/Kalman beta in y_t = alpha + beta*x_t + epsilon_t.
- Market interpretation: Defines spread neutrality and leg sizing.
- Use case: Position construction and spread measurement.
- Failure mode: Unstable beta creates hidden directional exposure.

## beta
- Formula: cov(asset, benchmark) / var(benchmark), or pair beta depending on endpoint.
- Market interpretation: Sensitivity to common factor or paired asset.
- Use case: Exposure normalization and risk attribution.
- Failure mode: Nonlinear exposure is missed by linear beta.

## ecm_x
- Formula: Delta x_t = alpha_x * error_{t-1} + lagged deltas + noise.
- Market interpretation: Adjustment speed of X toward equilibrium.
- Use case: Leader/follower and directional leg prediction.
- Failure mode: Spurious adjustment under unstable cointegration.

## ecm_y
- Formula: Delta y_t = alpha_y * error_{t-1} + lagged deltas + noise.
- Market interpretation: Adjustment speed of Y toward equilibrium.
- Use case: Leader/follower and directional leg prediction.
- Failure mode: Coefficient sign flips across regimes.

## ecm_strength
- Formula: Function of adjustment coefficient magnitude, t-statistics, and residual correction reliability.
- Market interpretation: How forcefully the pair corrects deviations.
- Use case: Rank mean-reversion candidates and holding period confidence.
- Failure mode: High in-sample strength can be overfit or stale.

## half_life
- Formula: -ln(2) / ln(phi), where spread_t = c + phi*spread_{t-1} + noise.
- Market interpretation: Expected decay horizon of spread shock.
- Use case: Set entry horizon, max holding period, and threshold timing.
- Failure mode: Invalid when spread is not stationary or phi is unstable.

## hurst
- Formula: Scaling relation E[range/std] ~ n^H.
- Market interpretation: H < 0.5 suggests anti-persistence; H > 0.5 suggests trend persistence.
- Use case: Filter mean-reversion vs trend regimes.
- Failure mode: Sensitive to sample length, microstructure noise, and jumps.

## zscore
- Formula: (spread_t - mean(spread)) / std(spread).
- Market interpretation: Distance from estimated equilibrium.
- Use case: Classic entry/exit trigger.
- Failure mode: Large z-score may indicate structural break rather than opportunity.

## rolling_zscore
- Formula: (spread_t - rolling_mean) / rolling_std.
- Market interpretation: Adaptive deviation measure.
- Use case: Threshold model with recent volatility adaptation.
- Failure mode: Window choice can chase noise or lag breaks.

## spread
- Formula: y_t - beta*x_t - alpha.
- Market interpretation: Tradable disequilibrium between hedged legs.
- Use case: Base series for stationarity, z-score, OU, and ECM.
- Failure mode: Bad hedge ratio converts spread into directional bet.

## pearson
- Formula: cov(x,y)/(std(x)*std(y)).
- Market interpretation: Linear co-movement.
- Use case: Basic dependence filter and breakdown monitor.
- Failure mode: Misses nonlinear and tail dependence.

## spearman
- Formula: Pearson correlation of ranked observations.
- Market interpretation: Monotonic dependence robust to nonlinear scaling.
- Use case: Dependence confirmation.
- Failure mode: Can miss asymmetric tail structure.

## kendall
- Formula: Difference between concordant and discordant pair probabilities.
- Market interpretation: Rank concordance often tied to copula parameters.
- Use case: Copula calibration and dependence stability.
- Failure mode: No direct spread trading signal by itself.

## copula
- Formula: C(u,v) joining marginal CDFs into joint distribution F(x,y)=C(Fx(x),Fy(y)).
- Market interpretation: Dependence structure independent of marginal distributions.
- Use case: Conditional mispricing, tail dependence, and portfolio risk.
- Failure mode: Wrong family or poor calibration creates false dislocation signals.

## conditional_probabilities
- Formula: P(U <= u | V = v) or tail-conditional variants from fitted copula.
- Market interpretation: Observed pair state vs expected conditional state.
- Use case: Pure copula and dual conditional copula entries.
- Failure mode: Probability distortions can persist instead of reverting.

## sharpe
- Formula: annualized mean(return) / std(return).
- Market interpretation: Risk-adjusted return quality.
- Use case: Strategy ranking and acceptance tests.
- Failure mode: Inflated by non-normal tails, serial correlation, and selection bias.

## sortino
- Formula: annualized mean(return) / downside_std(return).
- Market interpretation: Reward per downside volatility.
- Use case: Ranking strategies with asymmetric outcomes.
- Failure mode: Can ignore crash clustering until enough tail events occur.

## var
- Formula: Quantile loss at confidence alpha.
- Market interpretation: Expected threshold loss not exceeded most of the time.
- Use case: Position sizing and risk gating.
- Failure mode: Does not describe losses beyond the quantile.

## cvar
- Formula: Expected loss conditional on loss exceeding VaR.
- Market interpretation: Tail severity estimate.
- Use case: Tail-aware sizing and copula risk controls.
- Failure mode: Requires enough tail observations or robust stress modeling.

## drawdown
- Formula: (equity_peak - equity_t) / equity_peak.
- Market interpretation: Capital impairment from peak.
- Use case: Risk limits, strategy rejection, and kill switches.
- Failure mode: Backtest drawdown underestimates unseen structural breaks.

## win_rate
- Formula: winning_trades / total_trades.
- Market interpretation: Hit rate of completed trades.
- Use case: Diagnostics with payoff ratio.
- Failure mode: High win rate can hide rare large losses.

## ml_confidence
- Formula: Calibrated model probability of target event, usually P(profitable trade).
- Market interpretation: Meta-model belief in setup quality.
- Use case: Ranking, filtering, and sizing after calibration.
- Failure mode: Leakage, nonstationarity, and uncalibrated probabilities.

## profile_match
- Formula: Similarity score between current feature vector and historically profitable clusters.
- Market interpretation: How much the current setup resembles known winners.
- Use case: Meta-learning filter.
- Failure mode: Historical clusters may not survive new regimes.

## ou_optimal
- Formula: OU process dX_t = theta(mu-X_t)dt + sigma dW_t with optimal stopping thresholds.
- Market interpretation: Model-based mean-reversion entry/exit attractiveness.
- Use case: Threshold optimization and expected holding period.
- Failure mode: OU assumptions fail under jumps, trends, or changing volatility.
