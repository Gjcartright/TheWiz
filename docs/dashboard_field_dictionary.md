# Dashboard Field Dictionary

This dictionary maps Crypto Wizards dashboard/API fields to the local dYdX pair-research pipeline. Fields marked historical/backtest-derived must not be used for live signal generation unless they are captured and recomputed point-in-time.

| Field | Dashboard Source | Local Use | Current Ingestion | Priority |
| --- | --- | --- | --- | --- |
| pair_id/spread_id | scanner, pair detail | Join scanner rows to pair-detail pages and captures. | yes, scanner | must_integrate |
| symbol_1/symbol_2 | scanner, pair detail controls | Defines two-leg universe and local history fetch targets. | yes | must_integrate |
| exchange | scanner, pair detail/API params | Filters dYdX vs other exchanges. | partial | must_integrate |
| interval/period | scanner, pair detail controls/API params | Controls timeframe and sample window. | partial | must_integrate |
| spread/zscore/zscore_roll | spread/zscore charts/API | Core entry, exit, and regime path. | partial local, dashboard raw arrays missing | must_integrate |
| hedge_ratio/x_weighting/y_weighting | scanner, pair header, weighting slider | Sizing and spread construction. | partial | must_integrate |
| Pearson/Spearman/Kendall | correlation/dependency view | Dependency validation and stable-correlation filters. | local computed, dashboard summary partial | must_integrate |
| beta/betas | dependency view | Hedge stability and beta-anchor strategy design. | partial local | must_integrate |
| ecm_x/ecm_y/ecm_strength | ECM dependency views | Error-correction strategy and exits. | raw pair detail partial, not evidence pipeline | must_integrate |
| copula/u1_given_u2/u2_given_u1/tail thresholds | copula view/API | Nonlinear dependency, tail dislocation, and risk filters. | partial | must_integrate |
| Hurst/half_life/ou_optimal | scanner/header/API | Mean-reversion validation and timeout design. | partial | must_integrate |
| Sharpe/Sortino/returns/win_rate/closed trades | backtest metrics | Historical diagnostics only; not live signal features. | local recomputed separately | useful |
| MDD/drawdown/VaR/CVaR/underwater | risk metrics and backtest chart | Risk gates, drawdown controls, paper preflight. | partial local | must_integrate |
| entry/exit thresholds/operators | backtest panel | Reproducible dashboard strategy settings. | not systematic | must_integrate |
| Close N/Stop Loss/ECM min/Corr min | override panel | Exit/risk/regime controls. | not systematic | must_integrate |
| funding/slippage/cost assumptions | not visible in current dashboard capture | Paper-trade realism and stress tests. | local placeholder only | must_integrate |
| scanner filter set | live scanner page | Reproducible discovery snapshots across sort, cointegration, correlation, Hurst, half-life, copula, strategy, symbol, and exchange filters. | not systematic | must_integrate |
| inline strategy comparison returns/Sharpe | live scanner selected-row detail | Strategy-family triage only; must be replayed locally before acceptance. | not ingested | must_integrate |
| open/closed simulated positions | live trades page | Paper-trading validation, live monitoring, and post-trade review. | not ingested | must_integrate |
| Telegram alert configuration | live alerts page | Operational alert delivery only; credentials must stay outside repo. | not ingested by design | useful |

## No-Hindsight Rule

- Scanner/backtest rankings are discovery inputs, not deployable signals.
- Raw spread, z-score, dependency, ECM, copula, entry, return, and underwater arrays must be timestamped and replayed locally before strategy use.
- Any dashboard feature that is calibrated on the full visible sample is `risky_or_hindsight` until it can be reconstructed bar-by-bar.
- Account and alert credential fields are operational metadata only; never store private account data, bot tokens, or chat IDs in research artifacts.
