# RL Copula Dashboard Contract

The RL layer may consume Crypto Wizards Copula data only as an observation that
was captured at or before the candidate's `feature_timestamp`.

## Eligible inputs

- Conditional probabilities and their gap
- Copula family and correlation
- Pearson, Spearman, Kendall, Hurst, half-life, hedge ratio, and capital weight
- Dashboard z-score mode values, liquidity, VaR, CVaR, stationarity state, and
  explicit Copula execution blockers

The feature builder records whether a snapshot was attached, stale, or absent.
An orange Engle-Granger badge is encoded as `wizard_copula_engle_granger_trending`.
It is context and a research-only warning, not a stationarity confirmation.

## Exclusions

Dashboard return, Sharpe, Sortino, win rate, closed-trade count, and other
full-sample performance values are not RL features, labels, or rewards. They
remain discovery and human-review evidence only.

## Enforcement

`attach_copula_dashboard_features` selects the newest matched pair/timeframe
snapshot at or before the feature time. A later snapshot is rejected. A snapshot
older than 2.5 hours is marked stale and contributes no Copula values.

`run-rl-research` writes `reports/rl/rl_copula_dashboard_join_audit.csv` and
adds the result to `reports/rl/rl_leakage_audit.csv`.
