# Hudson & Thames YouTube Research Recommendations

These are machine-assisted inferences from Hudson & Thames captions. They require manual review and local validation.

- videos inventoried: 82
- caption insight windows: 73
- source videos represented: 45
- recommendation rows: 20
- promotion authority: none; no item can authorize Testnet or live execution

## Curated Must-Test Hypotheses

### Copula entry and exit logic matrix

**Recommendation:** Treat AND/OR entry and exit rules as explicit strategy parameters rather than implementation details.

**Interesting test idea:** Prioritize AND-open/OR-exit as a robustness challenger, while preserving the full four-combination matrix for local replay.

**Local validation:** Replay AND/AND, AND/OR, OR/AND, and OR/OR with identical point-in-time copula fits, costs, and signal thresholds.

**Evidence basis:** The reviewed captions say performance is highly sensitive to Boolean signal assembly and describe AND-open/OR-exit as a more stable variation.

**Source windows:** Advanced Pairs Trading: Intro to the Copula Approach [00:14:15](https://www.youtube.com/watch?v=hLV5Roa_Bek&t=855s) | Advanced Pairs Trading: Variations on the Copula Based Mispricing Index Strategy. [00:20:15](https://www.youtube.com/watch?v=t99uCFQL5KI&t=1215s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Copula model-dependence audit

**Recommendation:** Require signal direction and trade lifecycle to survive several statistically plausible copula families.

**Interesting test idea:** Create a copula-consensus score and use family disagreement as a veto, size reduction, or separate regime state.

**Local validation:** Compare top information-criterion families, parameter perturbations, threshold bands, flag reset rules, and divergence stop-loss rules.

**Evidence basis:** The reviewed captions repeatedly warn that mispricing-index signals and position flips depend materially on the selected copula and logic.

**Source windows:** Advanced Pairs Trading: Intro to the Copula Approach [00:30:45](https://www.youtube.com/watch?v=hLV5Roa_Bek&t=1845s) | Advanced Pairs Trading: Variations on the Copula Based Mispricing Index Strategy. [00:16:30](https://www.youtube.com/watch?v=t99uCFQL5KI&t=990s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Vine Copula cohort selection and exit repair

**Recommendation:** Use Vine Copulas only after point-in-time cohort selection demonstrates stable multivariate dependency.

**Interesting test idea:** Test extrema-based cohort selection and replace the documented early Bollinger exit with conditional-density normalization and risk exits.

**Local validation:** Compare random, sector, dependency-clustered, and extrema-selected cohorts; attribute gains separately to selection and exit logic.

**Evidence basis:** The reviewed caption reports weak performance for arbitrary groups, stronger extrema selection, and a tendency for Bollinger exits to close too early.

**Source windows:** Advanced Pairs Trading: Vine Copula Trading Strategy [00:18:00](https://www.youtube.com/watch?v=FB6tNgCZSbo&t=1080s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Cost-aware OU threshold optimization

**Recommendation:** Derive OU entries and exits from expected first-passage behavior and transaction costs instead of fixing every trade at plus or minus two.

**Interesting test idea:** Optimize return and Sharpe per unit time, then compare the analytic thresholds with the regular fixed-threshold styles already in the project.

**Local validation:** Refit OU parameters on trailing training windows, freeze thresholds for the next fold, and run parameter-sensitivity bands after costs.

**Evidence basis:** The reviewed caption derives entry/exit boundaries from first-passage time, risk-free rate, and round-trip transaction cost.

**Source windows:** Advanced Pairs Trading: Optimal Trading Thresholds for the O-U Process [00:05:15](https://www.youtube.com/watch?v=C7iZLMXyIOQ&t=315s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Minimum-profit versus trade-frequency optimization

**Recommendation:** Select spread boundaries by balancing minimum profit per trade against expected trade frequency and holding time.

**Interesting test idea:** Add a minimum-profit feasibility check that rejects thresholds whose gross boundary value cannot clear fees, funding, and slippage.

**Local validation:** Estimate first-passage duration and inter-trade interval point-in-time, then compare profit-per-time with fixed z-score entries.

**Evidence basis:** The reviewed caption explicitly frames threshold choice as a trade-off between profit per trade and trade frequency under transaction costs.

**Source windows:** Pairs Trading: The Cointegration Approach and Minimum Profit Optimization [00:17:16](https://www.youtube.com/watch?v=1zz91G0nR14&t=1036s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Hedge-error stationarity and hedge-ratio tournament

**Recommendation:** Evaluate hedge-ratio estimators by the stationarity and boundedness of realized hedge error, not regression fit alone.

**Interesting test idea:** Run OLS-differences, OLS-levels, ECM, PCA, and dynamic estimators as competing exact hedge-construction modes.

**Local validation:** Freeze each estimator per fold and compare hedge-error stationarity, beta drift, turnover, costs, and spread strategy outcomes.

**Evidence basis:** The reviewed caption distinguishes static and dynamic hedge methods and identifies non-stationary, unbounded hedge error as the central failure.

**Source windows:** Advanced Pairs Trading: Hedge Ratio Estimation Methods [00:02:17](https://www.youtube.com/watch?v=odh5rH3WYJM&t=137s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Calibrated meta-label trade gate and sizing

**Recommendation:** Use a secondary model only after a base strategy proposes a discrete trade, then calibrate its output before using it for size.

**Interesting test idea:** Audit extreme confidence deciles for reversal and compare trade/no-trade gating with risk-constrained Kelly sizing.

**Local validation:** Require enough candidate trades, a non-overfit primary model, calibrated probabilities, take-rate floors, and drawdown attribution.

**Evidence basis:** The reviewed captions distinguish model confidence from probability, show calibration failures at extreme deciles, and warn against daily vectorized use or tiny trade samples.

**Source windows:** Meta-Labeling: Solving for Non Stationarity and Position Sizing [00:21:01](https://www.youtube.com/watch?v=WbgglcXfEzA&t=1261s) | Meta-Labeling: Calibration and Position Sizing [00:55:31](https://www.youtube.com/watch?v=BIBSv_gwBgs&t=3331s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Purged, embargoed, and combinatorial validation

**Recommendation:** Purge overlapping trade labels and embargo observations adjacent to test folds before evaluating any learned gate.

**Interesting test idea:** Use combinatorial purged cross-validation to produce a distribution of backtest paths instead of trusting one walk-forward path.

**Local validation:** Compare walk-forward, purged K-fold, and combinatorial purged folds; report leakage sensitivity, path dispersion, and recent-regime coverage.

**Evidence basis:** The reviewed captions explain label concurrency, purging, embargo, and the benefit of multiple combinatorial backtest paths.

**Source windows:** Modelling: Label Concurrency and Cross Validation [00:15:00](https://www.youtube.com/watch?v=lDTSGK4JMYk&t=900s) | Cross-Validation in Finance and Backtesting [00:14:15](https://www.youtube.com/watch?v=kyjmHHoMc80&t=855s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Clustered feature-importance stability

**Recommendation:** Measure feature importance at the dependency-cluster level so correlated substitutes do not hide or duplicate signal.

**Interesting test idea:** Cluster Wizard, spread, copula, regime, and execution features using several dependence metrics and compare cluster stability across folds.

**Local validation:** Run clustered MDA/MDI with purged folds, multiple random seeds, and a report of informative, redundant, and noise clusters.

**Evidence basis:** The reviewed captions address masking and substitution effects by grouping similar features before measuring importance.

**Source windows:** Clustered Feature Importance Algorithms in Financial Machine Learning: Part 2 [00:00:46](https://www.youtube.com/watch?v=rY4JWAz194Q&t=46s) | Feature Importance Algorithms in Financial Machine Learning: Part 1 [00:12:00](https://www.youtube.com/watch?v=-A7yrsOihNM&t=720s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

### Conditioned portfolio and RL challengers

**Recommendation:** Use regime-conditioned optimization and RL as challengers to deterministic strategy decisions, not as execution authority.

**Interesting test idea:** Train policies across multiple reward functions, action spaces, asset clusters, and regime features; always retain equal-weight and rule-based baselines.

**Local validation:** Freeze environments, include costs and risk in rewards, compare discrete versus continuous sizing, and report baseline-relative out-of-sample failures.

**Evidence basis:** The reviewed captions emphasize state/action/reward design and also show a simple equal-weight baseline beating the conditioned optimizer out of sample.

**Source windows:** Deep Reinforcement Learning for Trading [00:15:00](https://www.youtube.com/watch?v=PwoTb-SxoC0&t=900s) | Conditional Portfolio Optimization [00:33:45](https://www.youtube.com/watch?v=vyjgR5Ln57s&t=2025s)

Confidence: 0.85; review status: assistant_reviewed_2026-08-06.

## Theme Recommendations

### Dependency structure

**Recommendation:** Use correlation and ECM fields as diagnostics, not as standalone proof that a spread will mean-revert.

**Interesting test idea:** Create a dependency-consensus feature that records Pearson, Spearman, Kendall, ECM-X, ECM-Y, and ECM-strength agreement.

**Local validation:** Ablate each dependency field in walk-forward tests and compare consensus, disagreement, and single-metric variants.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Theory-Implied Correlation Matrix (TIC) [00:06:01](https://www.youtube.com/watch?v=yhR1zLQIu9g&t=361s) | Portfolio Optimization Workshop [00:34:31](https://www.youtube.com/watch?v=OeqIGC-WTWo&t=2071s) | Advanced Pairs Trading: The Pearson Distance Approach [00:11:15](https://www.youtube.com/watch?v=fvTQAnjgIaA&t=675s)

Confidence: 0.58; review status: machine_extracted_needs_human_review.

### Risk and position sizing

**Recommendation:** Size pair legs from hedge exposure and volatility rather than equal notional alone.

**Interesting test idea:** Compare beta-neutral, volatility-balanced, and tail-risk-budgeted sizing while preserving a hard pair-level loss budget.

**Local validation:** Replay all accepted hypotheses under equal-notional, beta-neutral, inverse-volatility, and CVaR-budgeted sizing.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Meta-Labeling: Solving for Non Stationarity and Position Sizing [00:20:15](https://www.youtube.com/watch?v=WbgglcXfEzA&t=1215s) | Meta-Labeling: Calibration and Position Sizing [00:35:16](https://www.youtube.com/watch?v=BIBSv_gwBgs&t=2116s) | Meta-Labeling: Theory and Framework [00:39:46](https://www.youtube.com/watch?v=ZCFmZFBtqsQ&t=2386s)

Confidence: 0.56; review status: machine_extracted_needs_human_review.

### Machine learning

**Recommendation:** Use machine learning as an explainable trade-quality gate after the base strategy produces a candidate.

**Interesting test idea:** Measure feature-importance stability across folds and pairs; unstable importance should lower confidence even when headline metrics improve.

**Local validation:** Compare raw, rule-filtered, model-gated, and model-sized variants with take-rate and concentration controls.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Bagging and Boosting in Financial Machine Learning [00:17:15](https://www.youtube.com/watch?v=RMskDStbwi0&t=1035s) | Clustered Feature Importance Algorithms in Financial Machine Learning: Part 2 [00:00:02](https://www.youtube.com/watch?v=rY4JWAz194Q&t=2s) | Meta-Labeling: Solving for Non Stationarity and Position Sizing [00:13:32](https://www.youtube.com/watch?v=WbgglcXfEzA&t=812s)

Confidence: 0.55; review status: machine_extracted_needs_human_review.

### Copula dislocation

**Recommendation:** Keep copula dislocation as a distinct strategy family rather than treating it as another z-score.

**Interesting test idea:** Test copula signals both alone and as confirmation for spread entries, with tail-specific exits and asymmetric risk limits.

**Local validation:** Fit copulas only on prior data; compare standalone, confirmation, and veto roles under costed walk-forward replay.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Advanced Pairs Trading: Vine Copula Trading Strategy [00:18:00](https://www.youtube.com/watch?v=FB6tNgCZSbo&t=1080s) | Advanced Pairs Trading: Variations on the Copula Based Mispricing Index Strategy. [00:03:47](https://www.youtube.com/watch?v=t99uCFQL5KI&t=227s) | Advanced Pairs Trading: Intro to the Copula Approach [00:00:45](https://www.youtube.com/watch?v=hLV5Roa_Bek&t=45s)

Confidence: 0.54; review status: machine_extracted_needs_human_review.

### Stationarity and cointegration

**Recommendation:** Gate pair hypotheses with point-in-time stationarity and cointegration diagnostics before spread testing.

**Interesting test idea:** Treat disagreement between Engle-Granger, Johansen, and ADF as a regime feature instead of silently choosing the favorable test.

**Local validation:** Compare expanding-window Engle-Granger, Johansen, and ADF results; measure break frequency and out-of-sample survival.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Meta-Labeling: Solving for Non Stationarity and Position Sizing [00:01:31](https://www.youtube.com/watch?v=WbgglcXfEzA&t=91s) | Meta-Labeling: Theory and Framework [00:27:02](https://www.youtube.com/watch?v=ZCFmZFBtqsQ&t=1622s) | Pairs Trading: The Cointegration Approach and Minimum Profit Optimization [00:09:46](https://www.youtube.com/watch?v=1zz91G0nR14&t=586s)

Confidence: 0.54; review status: machine_extracted_needs_human_review.

### Regime and serial behavior

**Recommendation:** Estimate regime and serial-dependence features only from information available before each entry.

**Interesting test idea:** Use regime-model disagreement as an uncertainty flag and reduce size instead of forcing a single state label.

**Local validation:** Compare existing deterministic overlays with expanding-window HMM/GMM and autocorrelation features using purged folds.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Ensemble Meta-Labeling [00:45:01](https://www.youtube.com/watch?v=tpLCMVyMOaM&t=2701s) | Meta-Labeling: Solving for Non Stationarity and Position Sizing [00:28:30](https://www.youtube.com/watch?v=WbgglcXfEzA&t=1710s) | Meta-Labeling: Theory and Framework [00:27:02](https://www.youtube.com/watch?v=ZCFmZFBtqsQ&t=1622s)

Confidence: 0.53; review status: machine_extracted_needs_human_review.

### Reinforcement learning

**Recommendation:** Use reinforcement learning to propose and challenge hypotheses, not to bypass deterministic safety and acceptance gates.

**Interesting test idea:** Let the RL research agent search entry, exit, sizing, and regime combinations, then cluster its ideas to find similar Wizard pairs.

**Local validation:** Evaluate policies on frozen candidate sets with turnover, drawdown, and skipped-opportunity penalties before Testnet handoff.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Deep Reinforcement Learning for Trading [00:12:46](https://www.youtube.com/watch?v=PwoTb-SxoC0&t=766s) | Conditional Portfolio Optimization [00:11:17](https://www.youtube.com/watch?v=vyjgR5Ln57s&t=677s) | FinGPT: Open-Source Financial Large Language Models [00:03:45](https://www.youtube.com/watch?v=kX_7ocfTS8g&t=225s)

Confidence: 0.53; review status: machine_extracted_needs_human_review.

### Spread and signal design

**Recommendation:** Preserve each Wizard spread and ZScoreR mode as a separate hypothesis with its own entry and exit behavior.

**Interesting test idea:** Run an exact-mode tournament across Static Spread, Static ZScoreR, Dyn Spread, Dyn ZScoreR, OU Spread, and OU ZScoreR.

**Local validation:** Replay identical pair/timeframe folds for all six modes with entry-only and hard-exit regime overlays.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Advanced Pairs Trading: Sparse Mean Reversion Portfolio Selection [00:09:48](https://www.youtube.com/watch?v=IP7R-gnDF9w&t=588s) | Advanced Pairs Trading: Kalman Filters [00:00:46](https://www.youtube.com/watch?v=YavO2-sNVcs&t=46s) | Advanced Pairs Trading: Optimal Trading Rules [00:03:45](https://www.youtube.com/watch?v=Fllb9C7p7kE&t=225s)

Confidence: 0.52; review status: machine_extracted_needs_human_review.

### Backtest validation

**Recommendation:** Require costed walk-forward and Testnet evidence before promoting any video-derived strategy idea.

**Interesting test idea:** Track how often a strong full-sample result fails after purging, costs, and parameter freezing; make that decay a model feature.

**Local validation:** Run purged walk-forward folds with frozen parameters, sensitivity bands, after-cost metrics, and minimum trade-count gates.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** Cross-Validation in Finance and Backtesting [00:09:47](https://www.youtube.com/watch?v=kyjmHHoMc80&t=587s) | Feature Importance Algorithms in Financial Machine Learning: Part 1 [00:10:30](https://www.youtube.com/watch?v=-A7yrsOihNM&t=630s) | Synthetic Financial Data Generation with MlFinLab [00:02:16](https://www.youtube.com/watch?v=rRscFvpKb2s&t=136s)

Confidence: 0.52; review status: machine_extracted_needs_human_review.

### Execution and market frictions

**Recommendation:** Calibrate fees, funding, and slippage from the exact Hyperliquid market and network used for the replay.

**Interesting test idea:** Maintain separate research and execution feasibility scores so thin markets remain visible without becoming tradable by accident.

**Local validation:** Replay fills against timestamped L2 snapshots and realized Testnet fills; compare assumed versus observed cost distributions.

**Evidence basis:** deterministic theme extraction from caption windows

**Source windows:** An Overview of Financial Data Structures [00:03:45](https://www.youtube.com/watch?v=BumY00BJ61g&t=225s) | Meta-Labeling: Theory and Framework [00:36:47](https://www.youtube.com/watch?v=ZCFmZFBtqsQ&t=2207s) | Optimal Trading Rules Detection with Triple Barrier Labeling [00:09:46](https://www.youtube.com/watch?v=U2CxilKFue4&t=586s)

Confidence: 0.51; review status: machine_extracted_needs_human_review.
