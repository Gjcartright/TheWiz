# Dashboard Field Dictionary

This dictionary maps Crypto Wizards dashboard/API fields to the local Hyperliquid pair-research pipeline. Fields marked historical or backtest-derived must not be used for live signal generation unless they are captured or recomputed point-in-time. Crypto Wizards is discovery and diagnosis; local Hyperliquid replay is acceptance authority.

| Field | Dashboard Source | Local Use | Current Ingestion | Priority |
| --- | --- | --- | --- | --- |
| authenticated_session_state | protected member routes and navigation | Proves browser capture can access the intended dashboard state. | manual only | must_integrate |
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
| commission/slippage assumptions | pair Preferences and paper form | Wizard/local cost parity and paper realism. | partial; local Hyperliquid profile is separate | must_integrate |
| funding/borrow assumptions | not supplied by Wizard pair backtest | Hyperliquid after-cost replay and direction-specific carry. | local Hyperliquid data/profile | must_integrate |
| scanner filter set | live scanner page | Reproducible discovery snapshots across sort, cointegration, correlation, Hurst, half-life, copula, strategy, symbol, and exchange filters. | not systematic | must_integrate |
| scanner interval | scanner D/H control | Maps Daily and Hourly browser views into existing `interval` and `sweep_interval` identity. | API implemented; browser Hourly coverage missing | must_integrate |
| supporting chart family / spread_type | pair spread chart heading and capture lineage | Preserves Static, Dynamic, or OU chart context independently of the selected exact strategy. | spread_type blocker exists; raw chart label partial | must_integrate |
| inline strategy comparison returns/Sharpe | live scanner selected-row detail | Strategy-family triage only; must be replayed locally before acceptance. | not ingested | must_integrate |
| metric_accounting_state | pair backtest metrics and trade count | Explains nonzero performance with zero closed trades; existing minimum-trade gates still block progression. | progression gate implemented; provenance missing | must_integrate |
| open/closed simulated positions | live trades page | Paper-trading validation, live monitoring, and post-trade review. | not ingested | must_integrate |
| Telegram alert configuration | live alerts page | Operational alert delivery only; credentials must stay outside repo. | not ingested by design | useful |

## No-Hindsight Rule

- Scanner/backtest rankings are discovery inputs, not deployable signals.
- Raw spread, z-score, dependency, ECM, copula, entry, return, and underwater arrays must be timestamped and replayed locally before strategy use.
- Any dashboard feature that is calibrated on the full visible sample is `risky_or_hindsight` until it can be reconstructed bar-by-bar.
- Account and alert credential fields are operational metadata only; never store private account data, bot tokens, or chat IDs in research artifacts.

## Required Observation Key

No dashboard value is uniquely identified by pair alone. Every observation must retain:

| Key | Meaning |
| --- | --- |
| `pair_id` | Crypto Wizards pair identifier when available. |
| `asset_x` / `asset_y` | Ordered orientation; X is the driver/independent leg and Y is the dependent leg in Wizard descriptions. |
| `wizard_exchange` | Binance, Binance US, ByBit, Coinbase, DYDX, Forex, or Stocks/Other. |
| `source_surface` | Scanner browser, pair browser, prescanned API, analytical API, or local replay. |
| `interval` | Scanner uses Daily/Hourly; pair detail uses Daily/4 Hour/1 Hour/5 Min. Preserve the raw label and normalized value. |
| `period` | Lookback observations used for estimation and backtest. |
| `exact_mode` | One exact spread/signal construction, not a generic Z-score label. |
| `spread_type` / `raw_supporting_chart_label` | Existing spread lineage plus the literal chart family visible beside the strategy; may differ from `exact_mode`, especially for Copula. |
| `source_timestamp` | Vendor calculation or backtest timestamp. |
| `captured_at` | Local observation time. |
| `window_start` / `window_end` | Data window used to estimate the field. |
| `config_hash` | Hash of all mode, entry, exit, cost, weighting, and metric settings. |
| `pit_status` | `point_in_time`, `reconstructable`, `historical_only`, `vendor_derived`, or `unsafe_unknown`. |
| `metric_accounting_state` | `closed_realized`, `open_mark_to_market`, `mixed`, or `unknown`; `unknown` blocks acceptance. |
| `evidence_path` | Immutable raw response or browser-capture path. |

## Exact Mode Dictionary

| Dashboard label | Canonical mode | Core construction | Signal field |
| --- | --- | --- | --- |
| Static Spread | `static_spread` | Static OLS/ARMA residual spread. | normalized spread or configured raw spread threshold |
| Static ZScoreR | `static_zscore_r` | Static spread with rolling Z-score. | `zscore_roll` |
| Dyn Spread | `dynamic_spread` | State-space/Kalman dynamic hedge spread. | normalized dynamic spread |
| Dyn ZScoreR | `dynamic_zscore_r` | Dynamic spread with rolling Z-score. | `zscore_roll` |
| OU Spread | `ou_spread` | OU MLE spread with mean-reversion parameters. | normalized OU spread or optimal threshold |
| OU ZScoreR | `ou_zscore_r` | OU spread with rolling Z-score. | `zscore_roll` |
| Copula | `copula` | Conditional dependence/tail dislocation. | `u1_given_u2`, `u2_given_u1`, and configured copula thresholds |

The scanner also exposes family groupings such as Static All, Dynamic All, OU All, and all. Those are discovery filters, not exact modes.

## Scanner And API Fields

| Canonical field | Wizard/API aliases | Interpretation | Point-in-time rule |
| --- | --- | --- | --- |
| `returns_total` | scanner Return; `returns_total` | Total backtest return for the selected sample and settings. | historical only; never substitute a generic `return` field silently |
| `sharpe` | `sharpe`; `sharpe_ratio` | Backtest risk-adjusted return. | historical only |
| `spread_normalized` | `zscore_last`; scanner `norm` | Latest spread normalized over the selected model/window. | current only if source time and causal window are known |
| `zscore_roll` | `zscore_roll_last`; scanner `roll` | Latest rolling Z-score. | current only if rolling window and source time are known |
| `zscore_window` | API field; pair setting | Rolling-Z lookback; zero may mean automatic in the UI. | configuration |
| `profile_match` | scanner dependency badge | Vendor profile-cluster compatibility. | vendor-derived discovery field |
| `u1_given_u2` | X-given-Y directional probability | Copula conditional probability in the displayed orientation. | causal only if family and window were fit without future data |
| `u2_given_u1` | Y-given-X directional probability | Reverse copula conditional probability. | same restriction |
| `corr_copula` | copula correlation | Dependence parameter or vendor correlation summary. | historical/reconstructable |
| `copula` | `copula_name`; Best fit | Gaussian, Student-t, Clayton, Gumbel, or other selected family. | selected family is hindsight-prone |
| `johansen_coint` | Johansen badge | Johansen stationarity/cointegration pass indicator. | historical unless rolled forward; underlying specification required |
| `coint_eg` | Engle-Granger badge | Engle-Granger pass indicator. | historical unless rolled forward |
| `coint_eg_inc_trend` | orange Engle-Granger state | Engle-Granger result includes trend. | record separately from the EG pass boolean |
| `stationarity_state_source` | explicit API boolean or structured badge capture | Proves whether a stationarity value came from data, badge color, text-only capture, or was missing. | `unknown_text_only` must not produce a boolean |
| `metric_accounting_state` | pair-page performance plus closed trades | Distinguishes closed-trade-supported metrics from ambiguous open/mark-to-market results. | ambiguous/unknown blocks labels, rewards, and acceptance |
| `pair_page_data_quality_state` | pair-page warning text and structured diagnostics | Captures missing GARCH or implausible volatility displays by pair/timeframe/mode. | any blocker excludes the cell from comparison |
| `coint_eg_p` | EG p-value | Engle-Granger significance. | preserve window and deterministic terms |
| `hurst` | Hurst | Mean-reversion/persistence diagnostic. | historical unless causally rolled |
| `half_life` | Half Life | Estimated reversion horizon. | historical unless causally rolled |
| `zero_cross` | 0-sigma crossings | Spread normalization frequency. | historical diagnostic |
| `stddev_cross` | 2-sigma crossings | Extreme-spread crossing frequency. | historical diagnostic |
| `hedge_ratio` | hedge r; weighting | Leg relationship used in spread and sizing. | current only with estimation timestamp/window |
| `sym_1_volume` / `sym_2_volume` | 24h volume columns | Vendor-market liquidity hints for both legs. | current snapshot; not Hyperliquid execution liquidity |
| `mdd` | max drawdown | Backtest peak-to-trough loss. | historical only |
| `var` / `cvar` | VaR/CVaR at 99% | Backtest or simulated tail risk. | historical only; method and simulation settings required |
| `ml_confidence` | prescanned field | Vendor ML field with undocumented training lineage. | risky/hindsight; do not use live |
| `backtest_ts` | API timestamp | Vendor backtest calculation time. | source timestamp, not proof of PIT feature construction |

The scanner D/H control and pair-page timeframe control expose different option sets but belong to the existing interval lineage. On 2026-08-11, an otherwise identical live scanner configuration displayed 30 Daily results and 9 Hourly results; those UI counts were not saved as an immutable exhaustive browser capture. The API completed all 30 venue/interval/strategy request cells on 2026-08-12, but that does not prove exhaustive browser infinite-scroll coverage.

## Pair Diagnostic Fields

| Field group | Fields | Use | Current status |
| --- | --- | --- | --- |
| Beta | long-term beta; conditional beta | Hedge stability; beta dislocation; relationship breaks. | local summary/derived support; Wizard chart history incomplete |
| Correlation | Pearson; Spearman; Kendall; conditional correlation | Linear and rank dependency confirmation; break detection. | API/local summaries; Wizard conditional history incomplete |
| ECM static | X lag 1/2; Y lag 1/2; spread lag gamma; significance | Error-correction orientation and stability. | pair capture aliases/availability; raw histories incomplete |
| ECM dynamic | X lag 1; Y lag 1; gamma histories | Time-varying correction behavior. | dashboard capture gap |
| ECM strength | proprietary -1 to +1 series | Vendor correction deviation signal. | accepted by journal/importer but unsafe until timestamped |
| Impulse response | response by horizon and orientation | Exit horizon and recovery-shape research. | dashboard capture gap |
| Volatility/GARCH | long-term vol; beta; gamma; theta; correlation shock/persistence | Calm/crisis regime; sizing; relationship-break risk. | local models exist; Wizard parameter/history capture incomplete |
| Copula contour | prices/returns; 1/5/10%; point and contours | Tail-dislocation confidence and directional confirmation. | summary/journal partial; contour history missing |
| OU parameters | mu; alpha; beta; B; sigma; optimal entry/exit | OU mode parity and threshold research. | exact mode exists; full parameter/settings capture incomplete |

## Backtest Configuration Fields

Every result must capture all of the following before Wizard/local comparison:

- entry-long and entry-short operators and levels
- exit-long and exit-short operators and levels
- rolling-Z window and whether zero means auto
- copula lower/upper entry and exit thresholds
- forced close: Ignore, Half Life, or N=1 through 20
- stop-loss percentage
- minimum ECM deviation and minimum correlation strength
- commission rate and slippage rate
- capital weighting: Even or Risk Reduction
- metric mode: Standard or Aggressive
- simulation runs
- Manual or Auto live update
- input data hash, API version, source timestamp, and capture timestamp

Observed during this inventory: commission `0.10%`, slippage `0.05%`, Risk Reduction weighting, and 1000 simulations. These are observations, not project defaults. Aggressive metrics exclude idle periods and must be classified `risky_or_hindsight` for acceptance.

## Metric Accounting Integrity

- `closed_trades=0` with nonzero Sharpe, return, drawdown, or underwater history is not automatically an error, but its accounting basis is ambiguous.
- Preserve open-position state, realized PnL, unrealized PnL, entry count, closed-trade count, and calculation timestamp separately where available.
- Until those fields reconcile, set `metric_accounting_state=AMBIGUOUS_OPEN_OR_MARK_TO_MARKET`. Existing minimum-closed-trade gates continue to block paid proof and acceptance; the state additionally blocks ML labels and RL rewards from treating ambiguous vendor performance as an outcome.
- Do not infer that a displayed return is realized merely because it appears in the backtest summary.

## Stationarity Display Semantics

- green Johansen: Johansen passes
- green Engle-Granger: Engle-Granger passes without the trend qualification
- orange Engle-Granger: `engle_granger_trending`; do not count as a green confirmation
- no color: that test does not pass

The row label text always contains `Jn` and `EG`; it is not evidence that either test passed. Structured API booleans are authoritative for API rows. Browser captures must store computed color/SVG class. Orange EG maps to `coint_eg=true` plus `coint_eg_inc_trend=true`, not to a separate failed test.

Badge color alone is insufficient acceptance evidence. Preserve or recompute the statistical specification, window, p-value/test statistic, critical values, and timestamp.

## API Coverage And Known Gaps

Documented v1beta GET endpoints: `backtest`, `cointegration`, `copula`, `correlations`, `credits-used`, `prescanned`, `spread`, and `zscores`. Own-data POST is documented for all analytical endpoints except `prescanned` and `credits-used`.

The documented API does not expose full Johansen detail, ECM, ECM strength, impulse responses, GARCH/conditional-volatility histories, raw OHLCV, paper trades, or alerts. Those require authenticated dashboard capture or local recomputation.

During this inventory, documentation pages showed inconsistent route headings/examples for spread, zscores, and credits-used even when the curl example used the expected path. Treat v1beta docs as a guide and enforce live contract tests with sanitized fixtures.

## Authority And Leakage Rules

1. Scanner and dashboard backtests may create a hypothesis only.
2. Wizard current fields may enter a model/RL dataset only after causal timing and freshness are proven.
3. Historical performance, mode winners, best-fit families, `ml_confidence`, and full-window diagnostics are never live features.
4. Hyperliquid acceptance requires fresh local bars, current fees/funding/slippage, purged walk-forward replay, and an evidence path.
5. Backtest, paper, and live labels remain separate.
6. No account, API key, billing, Telegram token, or chat ID belongs in research evidence.
7. Browser captures require a protected-route authentication proof; product-card or download-button wording is not sufficient.
8. Strategy mode, existing spread type/supporting chart context, interval, and metric accounting state must never be inferred from one another.
