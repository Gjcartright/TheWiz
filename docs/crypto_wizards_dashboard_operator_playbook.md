# Crypto Wizards Dashboard Operator Playbook

Last live verification: 2026-08-07

## Purpose

Use Crypto Wizards as a configurable hypothesis laboratory. The scanner discovers and ranks candidates. The pair page diagnoses the relationship and runs a costed custom backtest. Hyperliquid point-in-time replay, walk-forward validation, and Testnet remain the acceptance authority.

The objective is not to maximize the largest displayed Sharpe. It is to find an after-cost Sharpe that survives neighboring settings, enough trades, later data, and Hyperliquid execution assumptions.

## Mandatory Run State

Capture these fields before reading any ranking:

- capture timestamp and scanner update timestamp
- exchange
- timeframe control: Daily or Hourly
- sort order
- cointegration filter
- correlation filter
- Hurst filter
- half-life filter
- copula-arbitrage filter
- exact strategy filter
- symbol filter
- pair-page period count
- commission and slippage settings
- capital-weighting mode
- Standard or Aggressive metrics
- manual or automatic refresh

The D/H control must be verified visually. On 2026-08-07, switching the same DYDX scanner from Hourly to Daily changed the leading candidates and their metrics materially.

## Scanner Workflow

1. Start with Daily, the intended exchange, no symbol filter, Strategy all cases, and diagnostic filters at all cases.
2. Capture every visible row before ranking. Sharpe, return, liquidity, stationarity, and Hyperliquid availability must not remove a source row.
3. Sort and label rows by Sharpe and `returns_total` only after exhaustive capture; thresholds create research views, not source deletion.
4. Keep weak research or execution diagnostics as explicit blockers, not silent exclusions.
5. Run the seven pair-page modes: Static Spread, Static ZScoreR, Dynamic Spread, Dynamic ZScoreR, OU Spread, OU ZScoreR, and Copula. Preserve the scanner's `ou_optimal` boolean on the source row instead of treating it as an eighth mode.
6. Click each retained row to open the same-pair mode comparison.
7. Open the pair workspace for full diagnostics and a custom costed backtest.

The scanner filters are a funnel, not an optimizer. A strict combination can return zero rows. Relax one diagnostic at a time and record which constraint removed the opportunity.

## Exact Modes

| Mode | What changes | Best research use | Main failure mode |
| --- | --- | --- | --- |
| Static Spread | Fixed relationship; normalized full-window spread | Stable hedge relationship and simple mean reversion | Look-ahead and structural-break sensitivity |
| Static ZScoreR | Static spread standardized over a rolling window | Local deviations around a stable relationship | Window dependence and repeated noisy entries |
| Dynamic Spread | Time-varying Kalman/state-space hedge relationship | Drifting beta or changing relative value | Chasing noise and turnover |
| Dynamic ZScoreR | Rolling z-score applied to the dynamic spread | Local dislocation after adapting the hedge | Two adaptive layers can overfit |
| OU Spread | OU-estimated mean-reverting process | Stable process with interpretable mean-reversion parameters | Very low trade count or regime breaks |
| OU ZScoreR | Rolling z-score of the OU spread | Local deviations around an OU process | Extra window may erase the OU advantage |
| Copula | Conditional dependence and tail dislocation | Nonlinear or asymmetric dependence | Not equivalent to z-score; tail fit may be unstable |

Spread and ZScoreR values are not interchangeable. Crypto Wizards describes Spread as a normalized, non-rolling process and ZScoreR as its rolling standardization. Entry values therefore need mode-specific meaning and settings.

`ou_optimal` is a scanner annotation, not a pair-page mode. Compare true and
false cohorts within the same actual mode, timeframe, and orientation. Do not
invent an OU Optimal spread, z-score, entry rule, or acceptance result from the
boolean alone.

## Same-Pair Comparison Protocol

Hold the following constant:

- pair and leg order
- exchange
- timeframe
- data end timestamp
- period count
- commission and slippage
- capital-weighting method
- stop, close-N, ECM, and correlation overrides

Then vary one item at a time:

1. Exact mode.
2. Entry band.
3. Exit band.
4. Close-N or half-life timeout.
5. Stop loss.
6. ECM minimum.
7. Correlation-strength minimum.
8. Capital weighting.

Compare net return, Sharpe, Sortino, closed trades, max drawdown, VaR, CVaR, and the shape of the cumulative-return and underwater curves. Reject a higher Sharpe caused by one lucky trade, much lower participation, or a single discontinuous gain.

## Pair Page Controls

### Research controls

- Timeframes: Daily, 4 Hour, 1 Hour, 5 Min.
- Exact modes: Static Spread, Static ZScoreR, Dyn Spread, Dyn ZScoreR, OU Spread, OU ZScoreR, Copula.
- Lookback: numeric periods.
- Dependence chart: betas, correlation, volatilities, ECM(Y), ECM(X), ECM strength.
- Copula view: prices or returns at 1%, 5%, or 10% tails.

### Strategy controls

- Spread entry long and short conditions.
- Spread exit long and short conditions.
- Rolling-ZScore window, with zero meaning automatic.
- Rolling-ZScore entry and exit conditions.
- Copula lower/upper entry and exit probabilities.
- Close after half-life or a fixed 1 to 20 periods.
- Stop loss.
- Minimum ECM deviation.
- Minimum correlation strength.
- Capital-weighting slider.

### Backtest preferences verified live

- Commission transaction cost: 0.10%.
- Slippage transaction cost: 0.05%.
- Capital weighting: Risk Reduction.
- Metrics: Standard.
- Simulation runs: 1000.
- Live update: Manual.

Manual mode is appropriate for research. Browse quickly, then refresh only retained candidates to obtain current data. Record the refresh timestamp. The refresh control restores the saved run configuration, so unsaved edits must not be assumed to have affected displayed metrics.

## Diagnostic Interpretation

- Cointegration: treat Johansen and Engle-Granger agreement as stronger evidence; disagreement is a feature and confidence penalty, not an invitation to choose the favorable test.
- Orange Engle-Granger state: trending cointegration case. It may favor rolling standardization or a trend-adjusted local model, but it is not live acceptance.
- Correlation: compare Pearson, Spearman, Kendall, and conditional dependence. High linear correlation alone does not prove mean reversion.
- Hurst: lower values support mean-reversion research. Values near or above 1 are warnings, even when scanner Sharpe is high.
- Half-life: match it to the holding horizon. A high Sharpe with one completed trade is not sufficient evidence.
- Zero and two-sigma crossings: use as activity and trade-frequency diagnostics, not standalone quality scores.
- ECM(X), ECM(Y), ECM strength: use direction and adjustment strength to explain which leg corrects and whether the relation is currently active.
- Copula conditional probabilities: use the asymmetry between X given Y and Y given X as a tail-dislocation signal. Test it standalone, as confirmation, and as a veto.
- VaR, CVaR, max drawdown, and underwater curve: use all four; none can be replaced by Sharpe.

## Robust Sharpe Rule

A dashboard Sharpe becomes a research candidate only when:

- scanner Sharpe and return pass the discovery gate
- pair-page costed metrics are captured
- trade count is sufficient for the claimed conclusion
- nearby entry, exit, and lookback settings do not collapse
- no single trade explains most returns
- the same mode survives expanding or rolling walk-forward replay
- Hyperliquid symbols, funding, liquidity, fees, and slippage are mapped
- Testnet forward results remain consistent

Use a plateau, not a peak. Prefer a broad region of acceptable settings over the single highest cell in a parameter grid.

## Video-Derived Operating Lessons

- The scanner is for continuous discovery; clicking a row opens quick charts and the custom pair workspace.
- Manual refresh is recommended while researching; refresh a candidate after its pattern looks useful.
- Transaction-cost assumptions live in Preferences and must be captured with every comparison.
- Spread and rolling ZScoreR are separate signal definitions and require separate thresholds.
- Forward-testing evidence in the Crypto Wizards material shows that backtests can materially overstate profitability.
- The videos emphasize consistency of the equity curve, liquidity, conditional copula fields, hedge ratio, rolling z-score, beta, and ECM behavior over headline return alone.

## Daily Deliverable

For every retained pair, write one row per exact mode with:

- pair, leg order, exchange, timeframe, periods, and update timestamp
- exact mode and signal value
- scanner return, Sharpe, and drawdown
- pair-page costed return, Sharpe, Sortino, trades, drawdown, VaR, and CVaR
- Johansen, Engle-Granger state, Pearson, Spearman, Kendall, Hurst, half-life, ECM fields, and copula fields
- threshold configuration and weighting
- discovery reason, blockers, and next local replay step

No dashboard result directly authorizes a trade.
