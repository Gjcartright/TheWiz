# RL Copula Dashboard Contract

The RL layer may consume Crypto Wizards Copula data only as an observation that
was captured at or before the candidate's `feature_timestamp`.

## Eligible inputs

- Conditional probabilities and their gap
- Copula family and correlation
- Pearson, Spearman, Kendall, Hurst, half-life, hedge ratio, and capital weight
- Dashboard z-score mode values, liquidity, VaR, CVaR, stationarity state, and
  explicit Copula execution blockers
- ECM-X, ECM-Y, and ECM-strength availability; configured Copula entry/exit
  thresholds; directional interpretation; family/tail classification; and a
  snapshot-completeness score

The feature builder records whether a snapshot was attached, stale, or absent.
An orange Engle-Granger badge is encoded as `wizard_copula_engle_granger_trending`.
It is context and a research-only warning, not a stationarity confirmation.

## Exclusions

Dashboard return, Sharpe, Sortino, win rate, closed-trade count, and other
full-sample performance values are not RL features, labels, or rewards. They
remain discovery and human-review evidence only.

## Enforcement

`attach_copula_dashboard_features` requires an exact pair, timeframe, and Wizard
mode match. The source must be a captured `pair_detail_capture` row with an
evidence path, both directional conditional probabilities in `[0, 1]`, and a
named Copula family. Scanner rows, generated missing-timeframe placeholders,
matrix-only summaries without direction values, and incomplete captures cannot
be attached.

The builder selects the newest complete matched snapshot at or before the
feature time. A later snapshot is rejected. A snapshot older than 2.5 hours is
marked stale and contributes no Copula values. The audit records the candidate
and snapshot modes, mode-match result, source-evidence presence, completeness,
age, and rejection reason.

Video-derived Copula ideas are research priors only. They may generate local
tests such as the AND/OR entry-exit matrix or family-consensus challengers, but
they cannot supply an observation, label, reward, promotion, or order decision.

`run-rl-research` writes `reports/rl/rl_copula_dashboard_join_audit.csv` and
adds the result to `reports/rl/rl_leakage_audit.csv`.
