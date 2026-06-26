## P2 Resumed Diagnostic (2026-06-20)

### What was verified
- Official dYdX docs still describe the same Indexer HTTP API shape for:
  - `Get Candles`
  - `Get Historical Funding`
- So the current fetch blocker is **not proven to be an API route-shape change**.
- In this environment, the practical blocker remains external indexer reachability / DNS.

### Local expansion result
- `dydx-local-pair-universe` was run against the cached manual payload set.
- Result:
  - `139` rows with `status=rebuilt`
  - `14` rows with `status=failed`
  - all successful rows were `rebuilt_existing_pair_history`
  - there were **0 new successful pair builds**
- Therefore the local cache is exhausted for net-new pair creation.

### Acceptance result after rerun
- Commands rerun:
  - `run-pair-detail-experiments --funding-path data/processed/dydx_funding.csv`
  - `priority-readiness`
  - `gap-test`
- P2 remains blocked:
  - `production_eligible=0`
  - `preferred_eligible=0`
  - `max_two_leg_pairs_tested=143`
  - `max_two_leg_passing_pairs=0`
  - top blocker: `passing_pairs<2:37`

### Important diagnosis
The failure mode is now clearer than before:

1. Small-sample high-quality runs exist
- Example pattern: some pair/strategy/cost-bucket rows pass non-trade gates (`profit_factor`, `sharpe`, `drawdown`, `expectancy`) but only have very low trade counts.
- Example from current evidence:
  - `LINK-USD-SOL-USD` / `Copula Dislocation Ranking Engine` / `base`
  - `trades=21`, `profit_factor=1.808057`, `sharpe=2.251688`, `max_drawdown=0.040988`, `expectancy=0.002490`
  - blocker: `trades<100`

2. Large-sample SOL/LINK runs exist, but quality collapses
- Example pattern: long-history `SOL-USD-LINK-USD` rows have high trade counts, including `1088` trades, but then fail production quality gates:
  - `profit_factor<1.8`
  - `sharpe<1.2`
  - `max_drawdown>0.15`
  - `expectancy<=0`

### Interpretation
- This means the problem is **not only** “we need more data.”
- More history helps some micro-runs approach trade thresholds, but the largest current SOL/LINK runs already show that simply extending the same behavior can still fail risk/return gates.
- The next successful unblock attempt likely needs **fresh market history plus additional candidate pairs/regimes**, not just a rerun of the same local cache.

### Best next external action
When network fetch becomes available again:

1. Refresh `SOL-USD` and `LINK-USD` from the official indexer first.
2. Immediately also fetch one or two adjacent candidate pairs with overlapping modern history, especially:
   - `ETH-USD/LINK-USD`
   - `BTC-USD/LINK-USD`
   - `ETH-USD/SOL-USD`
3. Rerun:
   - `run-pair-detail-experiments --funding-path data/processed/dydx_funding.csv`
   - `priority-readiness`
   - `gap-test`
4. Compare whether:
   - `max_two_leg_passing_pairs` rises above `0`
   - any strategy gets at least two passing pairs
   - `production_eligible` changes from `0`
