# API-First Research Flow

This project now uses a simple four-gate operating model:

```text
Crypto Wizards API discovers
-> Apify/dYdX verifies execution reality
-> local point-in-time tests decide
-> dashboard explains and audits
```

## Gate 1: Crypto Wizards API Discovery

Use Crypto Wizards API first when looking for candidates. Pull scanner rows and then enrich promising pairs with:

- z-score history
- spread history
- backtest history
- cointegration
- copula
- correlations

### Discovery Pass Criteria

The first screen must start with Crypto Wizards Sharpe and `returns_total`:

- Sharpe must be greater than `2.0`.
- `returns_total` must be greater than `20%`.

Z-score, spread family, and copula are not first-pass blockers. They are enrichment and audit fields checked after a pair clears Sharpe and `returns_total`.

Pairs that do not clear Sharpe and `returns_total` are not first-pass candidates.

Tested API intervals:

- `Daily`
- `Hourly`
- `Min5`

The tested API path rejected 4H interval names. Treat dashboard 4H as manual audit only until a working API route is confirmed.

Crypto Wizards evidence can create `WATCH`, `FETCH_MORE_DATA`, or `REJECT`. It cannot create `PROMOTE`.

## Gate 2: Apify/dYdX Execution Screen

Use Apify MCP as the preferred acquisition layer for dYdX execution checks:

- both legs exist
- market is active
- 24h volume is usable
- open interest is usable
- funding is current
- data is not stale

Direct dYdX indexer calls are fallback/diagnostic only.

## Gate 3: Local Acceptance

Only pairs that survive the execution screen should get expensive local tests. Promotion requires local point-in-time evidence:

- raw two-leg history
- hedge/beta/spread derived locally
- costs, slippage, funding drag
- regime overlays
- walk-forward or out-of-sample checks
- enough trades to trust the result

This is the only promotion authority.

## Gate 4: Dashboard Audit

Use the Crypto Wizards dashboard after API pulls to:

- inspect charts visually
- confirm scanner/detail rows
- review UI-only controls or metrics
- explain a decision to a human

Dashboard metrics are not live signal features unless they are captured point-in-time and replayable.

## Current Working Priority

1. `SOL-USD/WLD-USD` daily: first execution-grade test target.
2. `HYPE-USD/TRX-USD` daily: watch liquidity.
3. `DOGE-USD/LTC-USD` hourly: research-only until local evidence improves.
4. `ETH-USD/XRP-USD` hourly: research-only and liquidity unknown.
5. `BTC-USD/DOGE-USD` hourly: statistically interesting but execution screen blocked.

## One-Line Rule

Do not deep-test pairs just because they look good on Crypto Wizards. Use Crypto Wizards to find the trail, Apify/dYdX to see whether the trail is tradable, and local tests to decide whether it is real.
