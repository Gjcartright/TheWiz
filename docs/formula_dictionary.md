# Formula Dictionary

## cointegration
- Formula: Active local Engle-Granger fits `log(Y_t)=alpha+beta_y_on_x*log(X_t)+e_t` and tests `e_t=log(Y_t)-alpha-beta_y_on_x*log(X_t)`. Johansen is not yet local acceptance authority.
- Market interpretation: Tests whether a pair has a persistent equilibrium relation.
- Use case: Filter/rank pairs before mean-reversion strategies.
- Failure mode: Breaks during regime shifts, structural changes, or multiple-testing overfit.

## hedge_ratio
- Formula: Active static beta is OLS in `log(Y_t)=alpha+beta_y_on_x*log(X_t)+e_t`. The named opposite-leg execution contract requires finite `beta_y_on_x>0`; other estimators and negative-beta economics require separate method/contract IDs.
- Market interpretation: Defines spread neutrality and leg sizing.
- Use case: Position construction and spread measurement.
- Failure mode: Unstable beta creates hidden directional exposure.

## dynamic_hedge_ratio
- Formula: `beta_t=rolling_cov(log(X),log(Y))/rolling_var(log(X))`, in Y-on-X orientation with sample covariance/variance over the declared window. Held-out evaluation is seeded only with the trailing training observations; no future rows or backfill are used.
- Market interpretation: Time-varying relative sensitivity between the two legs.
- Use case: Dynamic Spread and Dynamic ZScoreR signal construction and gross-one sizing, using the same point-in-time beta series for both.
- Failure mode: Missing, nonfinite, or nonpositive beta blocks the current opposite-leg contract; a sign-changing relationship requires a separately named economic strategy rather than a silent beta default.

## beta
- Formula: cov(asset, benchmark) / var(benchmark), or pair beta depending on endpoint.
- Market interpretation: Sensitivity to common factor or paired asset.
- Use case: Exposure normalization and risk attribution.
- Failure mode: Nonlinear exposure is missed by linear beta.

## ecm_x
- Formula: `Delta log(X_t)=gamma_x*e_{t-1}+lagged deltas+noise`, with `e=log(Y)-alpha-beta_y_on_x*log(X)` and expected correction sign `gamma_x>0`.
- Market interpretation: Adjustment speed of X toward equilibrium.
- Use case: Leader/follower and directional leg prediction.
- Failure mode: Spurious adjustment under unstable cointegration.

## ecm_y
- Formula: `Delta log(Y_t)=gamma_y*e_{t-1}+lagged deltas+noise`, with expected correction sign `gamma_y<0`.
- Market interpretation: Adjustment speed of Y toward equilibrium.
- Use case: Leader/follower and directional leg prediction.
- Failure mode: Coefficient sign flips across regimes.

## ecm_strength
- Formula: Share of the two HC1 adjustment coefficients that are significant at 5% and have the canonical expected signs (`gamma_x>0`, `gamma_y<0`).
- Market interpretation: Support for expected-direction correction across the two legs; the score does not measure coefficient magnitude.
- Use case: Rank mean-reversion candidates and holding period confidence.
- Failure mode: High in-sample strength can be overfit or stale.

## half_life
- Formula: `delta_t*ln(2)/(-ln(phi))`, where `spread_t=c+phi*spread_{t-1}+noise`, `0<phi<1`, and `delta_t` is the caller-declared duration of one regular observation interval in the desired output time unit. With `delta_t=1`, the result is in observation intervals.
- Market interpretation: Expected decay horizon of spread shock.
- Use case: Set entry horizon, max holding period, and threshold timing.
- Failure mode: The local fit rejects missing or irregular observations and `phi` outside `(0,1)`. A numerically valid fit alone does not establish stationarity or a reliable trading horizon.

## hurst
- Formula: The local first-order detrended fluctuation analysis (DFA) estimator fits `log(F(n))=a+H*log(n)`, where `F(n)` is the root-mean-square fluctuation after linear detrending of the integrated, mean-centered series in windows of scale `n`.
- Market interpretation: H < 0.5 suggests anti-persistence; H > 0.5 suggests trend persistence.
- Use case: Filter mean-reversion vs trend regimes.
- Failure mode: Sensitive to sample length, microstructure noise, and jumps.

## zscore
- Formula: (spread_t - sample_mean) / sample_std. The observed Crypto Wizards custom-series `zscore` uses sample standard deviation (`ddof=1`) over the supplied sample.
- Market interpretation: Distance from estimated equilibrium.
- Use case: Classic entry/exit trigger.
- Failure mode: A full-sample mean and standard deviation use hindsight. Live/local signals must estimate calibration statistics inside each walk-forward training fold or use a causal expanding calculation.

## rolling_zscore
- Formula: (spread_t - rolling_mean_w) / rolling_sample_std_w. Observed Crypto Wizards `zscore_roll` uses `ddof=1`, the configured `roll_w`, and zero during warmup.
- Market interpretation: Adaptive deviation measure.
- Use case: Threshold model with recent volatility adaptation.
- Failure mode: Window choice can chase noise or lag breaks.

## spread
- Formula: Fitted residual is `log(Y_t)-alpha-beta_y_on_x*log(X_t)`. Replay execution spread is separately named `log(Y_t)-beta_y_on_x*log(X_t)`; z-scores remove the constant, while raw OU modes explicitly center it.
- Market interpretation: Tradable disequilibrium between hedged legs.
- Use case: Base series for stationarity, z-score, OU, and ECM.
- Failure mode: Bad hedge ratio converts spread into directional bet.

## ingested_pair_spread
- Formula: `log(Y_t)-beta_y_on_x*log(X_t)`, constructed only when a finite positive hedge ratio is explicitly present.
- Provenance: New dYdX history imports identify the slope as `beta_y_on_x` and mark a full-history fit as `full_sample_hindsight_research_only`. Generic fixture ingestion does not invent a missing ratio.
- Use case: Research normalization before point-in-time refitting.
- Failure mode: The superseded formula `price_X-beta*price_Y` has opposite orientation and different units; using it under the same field name invalidates cross-source comparisons.

## two_leg_dispatch
- Formula contract: two-leg PnL is selected only when `price_x`, `price_y`, and `hedge_ratio` are all present. Prices without a ratio may use a supplied spread for research-only replay, but cannot enter two-leg accounting.
- Use case: Experiment harness, threshold sweeps, and ML label construction.
- Failure mode: Selecting on prices alone previously reached an implicit `beta=1.0` path or crashed after fail-closed validation was introduced.

## crypto_wizards_static_spread_observed
- Formula: OLS fits `y = alpha + beta*x`, then `spread = y - (alpha + beta*x)`. When the vendor response reports `log_used=true`, it instead fits `log(y) = alpha + beta*log(x)` and uses the log residual.
- Evidence: Reconstructed to numerical precision from the 2026-08-07 BTC/EIGEN and DOGE/EIGEN Hyperliquid custom-series proofs.
- Use case: Vendor formula parity and exact-mode diagnosis.
- Failure mode: The custom-series response fits the complete supplied sample. It is historical/hindsight evidence and cannot be copied directly into a live signal; fit alpha, beta, transform choice, and scale inside each walk-forward training fold.

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
- Formula: `sqrt(periods_per_year)*mean(net simple return)/sample_std(net simple return,ddof=1)`, zero risk-free rate, 365-day crypto annualization. Unknown intervals, timestamp/label mismatches, tiny samples, and zero variance are blocked.
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
- Formula: `max((running_peak-equity_t)/running_peak)` with initial equity `1.0` included before the first bar.
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
- Vendor formula: not exposed by the captured API/dashboard evidence.
- Observed type: scanner boolean annotation attached to an ordinary Static, Dynamic, OU, or Copula source row.
- Market interpretation: vendor screening hint that may identify a cohort for OU-threshold research.
- Use case: true/false stratification within the row's actual mode; never an independent pair-page signal.
- Failure mode: inferring thresholds, spread math, or acceptance authority from the boolean.
