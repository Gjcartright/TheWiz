# Crypto Wizards + dYdX Quant Research Project Handoff

Date: 2026-06-25

## 1. What This Project Started As

The project started as a research and backtest repo for crypto pair trading. The original goal was to find strong dYdX pairs, test mean-reversion strategies, compare timeframes, and decide which pairs were worth moving toward paper trading.

At the beginning, the work was mostly centered on local backtests and pair research. We were testing pairs such as ETH-SOL, BTC-DOGE, BTC-SOL, ETH-LINK, DOGE-XRP, BTC-DOGE, and later candidates found through Crypto Wizards. The early strategy set included z-score, OU, beta-dislocation, copula-dislocation, ghost/ambush, and regular entry/exit styles.

The first system idea was: find pairs, run strategies, rank results, and decide what to test next.

That changed as we discovered the repo needed stronger evidence control, better data freshness, and a cleaner route from discovery to acceptance.

## 2. The Big Pivot

The project moved from a simple research/backtest repo into a staged research pipeline:

```text
Crypto Wizards / Apify / dYdX / dashboard inputs
→ discovery candidates
→ exact Wizard mode capture
→ local dYdX verification
→ pair universe
→ evidence-gated acceptance
→ trade dataset
→ model gate
→ model-gated backtest
→ quantized scoring only if proven
→ dashboard command center
→ paper/live guardrails
```

The biggest rule we added was this:

Crypto Wizards can discover candidates, but local dYdX point-in-time evidence must approve them.

That means a great Wizard Sharpe ratio alone can never promote a pair to trading. It can only create a hypothesis.

## 3. Why We Changed The Pipeline

Several issues forced the pipeline to become stricter:

1. Some local tests were using stale or short data.
2. Crypto Wizards was not using one single generic z-score.
3. Wizard pair pages expose multiple exact modes: Dynamic Spread, Dynamic ZScoreR, OU Spread, OU ZScoreR, Static Spread, Static ZScoreR, and Copula.
4. A pair could look good on the scanner but fail local after-cost verification.
5. Scanner-level fields were not enough. Pair-detail fields matter.
6. Wizard cost assumptions were not clearly visible on the pair page.
7. ML could easily become misleading if trained on future-looking fields or full-sample dashboard metrics.
8. Dashboard scores could create false confidence unless every row shows evidence, blockers, freshness, and reasons.

So the system was hardened around evidence, freshness, exact mode matching, leakage prevention, and acceptance gates.

## 4. Current Project Goal

The current goal is to build a real evidence-gated quant research system:

```text
research → pair universe → local verification → trade dataset → model gate → quantized scoring → dashboard → paper/live guardrails
```

The system should answer:

- Which pairs are interesting?
- Why are they interesting?
- What exact Wizard mode made them interesting?
- Does local dYdX data confirm the idea?
- Does it still work after costs?
- Is the data fresh enough?
- Is the trade count thick enough?
- Is the pair tradable on dYdX?
- Are we in the right regime?
- Should this be promoted, watched, rejected, or fetched for more data?
- If a model is used, does it improve the strategy out-of-sample?
- Can the dashboard explain every action and blocker?

## 5. Crypto Wizards Discovery Flow

The new first step is Crypto Wizards discovery.

The user-defined first filter became:

- Sharpe ratio above 2.0
- `returns_total` above 20 percent
- Daily timeframe, not 5-minute
- Keep exact mode
- Leave copula out of the initial simple scanner filter, but use copula later as diagnostic evidence

Important correction: the scanner return field should be `returns_total`, not a generic `returns` field.

The first discovery list should be separated into:

1. A pure Sharpe + `returns_total` list, even if execution blockers exist.
2. A research-blockers-only list.
3. A final evidence-gated list.

This was done because the user wanted to see raw opportunity separately from blockers.

## 6. Exact Mode Discovery

We discovered that Crypto Wizards does not calculate one single z-score. It separates spread and z-score logic by mode.

The modes we mapped were:

- Dynamic Spread
- Dynamic ZScoreR
- OU Spread
- OU ZScoreR
- Static Spread
- Static ZScoreR
- Copula

In the code, these map to spread/strategy identifiers:

- Static Spread: spread_id 3, strategy_id 1
- Static ZScoreR: spread_id 3, strategy_id 2
- Dynamic Spread: spread_id 1, strategy_id 1
- Dynamic ZScoreR: spread_id 1, strategy_id 2
- OU Spread: spread_id 2, strategy_id 1
- OU ZScoreR: spread_id 2, strategy_id 2
- Copula: strategy_id 3 style handling

The new rule is:

Do not run local comparison as “generic z-score” if the Wizard candidate came from a specific mode. Capture and test the exact mode.

## 7. Crypto Wizards Pair Page Diagnostics

On the Wizard pair page we found useful diagnostics beyond the scanner:

- Pearson correlation
- Spearman correlation
- Kendall correlation
- Conditional probability charts
- Copula fit
- Copula conditional values
- ECM X
- ECM Y
- ECM strength
- Hurst / mean reversion fields where available
- Static/dynamic/OU spread and z-score fields
- Strategy dropdowns
- Timeframe and period controls
- Backtest machine settings

These fields should be used as diagnostic confirmation, not automatic acceptance.

They help answer:

- Is this pair correlated enough?
- Is the dependency linear or rank-based?
- Is the relationship asymmetric?
- Is there tail dependence?
- Is one leg correcting more than the other?
- Is the spread mean-reverting?
- Which Wizard mode actually generated the signal?

## 8. Wizard Backtest Machine Role

The Wizard backtest machine should be used to generate hypotheses and bedrock settings.

It should capture:

- pair
- timeframe
- period/lookback
- exact mode
- entry thresholds
- exit thresholds
- weighting / hedge values
- Wizard Sharpe
- Wizard return
- drawdown
- win rate
- trade count
- visible strategy parameters

But Wizard results remain discovery evidence only until verified locally.

## 9. Dashboard Exploration Requirement

We decided the scanner page is not the whole product.

The dashboard should be inventoried fully, including:

- scanner page
- pair detail pages
- z-score views
- spread charts
- dependency views
- correlation views
- copula views
- ECM views
- error-correction views
- backtest pages
- risk pages
- stationarity sections
- volume/liquidity fields
- strategy dropdowns
- timeframe controls
- lookback controls
- exchange filters
- export/download buttons
- API links
- docs links
- hidden tabs, popovers, menus, and advanced settings

Planned reports:

- `reports/dashboard_full_inventory.csv`
- `reports/dashboard_full_inventory.md`
- `reports/dashboard_missing_integrations.csv`
- `reports/dashboard_capture_opportunities.csv`
- `docs/dashboard_field_dictionary.md`
- `reports/dashboard_integration_summary.md`

Each feature should be ranked:

- `must_integrate`
- `useful`
- `optional`
- `not_useful`
- `risky_or_hindsight`

Any hindsight-like feature must be marked and blocked from live signal generation unless it can be made point-in-time.

## 10. Repo Strategy Change

Originally, the plan included cleanup and archiving.

We changed that.

The safer plan is:

Leave the repo intact.

Build a clean active layer and artifact index on top of the current repo.

Archive only later, after the active pipeline is stable.

The reason: old reports and scripts may contain evidence. Moving them too early could destroy traceability.

## 11. Active Layer Outputs

The active layer should produce:

- `reports/active/current_state.md`
- `reports/active/current_state.csv`
- `reports/active/artifact_index.csv`
- `reports/active/artifact_index.md`
- `reports/active/canonical_commands.md`

Artifact statuses:

- `active`
- `historical_evidence`
- `scratch`
- `superseded`
- `unknown`
- `do_not_move`

Artifact index fields:

- path
- artifact type
- status
- source system
- created or modified time
- used by active pipeline
- evidence value
- safe to archive later
- reason
- notes

The active layer lets us understand the repo without moving anything.

## 12. Commands Added Or Planned

The project now has or is expected to have commands such as:

```bash
PYTHONPATH=src python -m quant_platform.cli system-check
PYTHONPATH=src python -m quant_platform.cli build-artifact-index
PYTHONPATH=src python -m quant_platform.cli current-state
PYTHONPATH=src python -m quant_platform.cli build-pair-universe
PYTHONPATH=src python -m quant_platform.cli build-trade-dataset
PYTHONPATH=src python -m quant_platform.cli train-trade-gate
PYTHONPATH=src python -m quant_platform.cli run-model-gated-backtest
PYTHONPATH=src python -m quant_platform.cli export-trade-gate-model
PYTHONPATH=src python -m quant_platform.cli build-command-dashboard
```

Wizard-specific commands were added/planned:

```bash
PYTHONPATH=src python -m quant_platform.cli build-wizard-evidence
PYTHONPATH=src python -m quant_platform.cli build-wizard-hypotheses
PYTHONPATH=src python -m quant_platform.cli build-wizard-diagnostic-confirmation
PYTHONPATH=src python -m quant_platform.cli build-wizard-local-parity
PYTHONPATH=src python -m quant_platform.cli build-wizard-exact-mode-capture-queue
PYTHONPATH=src python -m quant_platform.cli build-wizard-research-pack
```

Long-history dYdX command used:

```bash
PYTHONPATH=src python -m quant_platform.cli run-dydx-long-history \
  --asset-x BNB-USD \
  --asset-y STX-USD \
  --pair-id bnb_stx_daily_320_fresh \
  --windows 2 \
  --limit 200 \
  --interval 1DAY \
  --derive-hedge-ratio \
  --allow-stale-fetch
```

## 13. Pair Universe

The pair universe is the master table.

It should include:

- pair
- asset X
- asset Y
- exchange
- dYdX tradable yes/no
- available timeframes
- Wizard pair ID
- exact Wizard mode
- spread ID
- strategy ID
- cointegration score
- copula score
- z-score score
- half-life
- Hurst
- correlation
- funding drag
- volume
- open interest
- local backtest score
- discovery score
- acceptance score
- combined score
- decision bucket
- decision reason
- missing data reason
- source timestamp
- field freshness
- stale reason
- evidence path

Buckets:

- `PROMOTE`
- `WATCH`
- `FETCH_MORE_DATA`
- `REJECT`

Critical rule:

`PROMOTE` requires local point-in-time, costed, walk-forward or comparable acceptance evidence.

Wizard-only evidence can never produce `PROMOTE`.

## 14. Discovery Score vs Acceptance Score

We split scoring into two kinds.

Discovery score can use:

- Crypto Wizards hints
- dashboard scanner hints
- Apify hints
- broad liquidity hints
- correlation hints
- copula hints

Acceptance score can only use:

- local dYdX point-in-time evidence
- after-cost results
- walk-forward results
- regime stability
- trade count
- funding drag
- drawdown
- stale data penalties
- dYdX tradeability

Combined score is dashboard convenience only. It is not promotion authority.

This prevents fake precision.

## 15. Strategy Logic

The project considered both regular and creative entry/exit styles.

Core families:

- z-score mean reversion
- OU mean reversion
- beta dislocation
- copula dislocation
- ghost / ambush entries
- regular threshold entries

Regime overlays:

- all
- range_only
- exclude_crisis
- calm_vol_only
- range_low_tail
- stable_hedge

Regime application styles:

- entry_only
- hard_exit

The goal is not one universal strategy. The goal is to match pair behavior to the right strategy and exit logic.

## 16. Entry And Exit Logic Direction

Regular logic:

- Enter when spread or z-score is stretched.
- Exit when it mean reverts.
- Block entries in bad regimes.
- Use hard exits when regime breaks.

Creative logic explored:

- ambush entries after failed breakout
- ghost entries after phantom dislocation
- tail-aware exits
- beta instability exits
- copula asymmetry entries
- ECM-confirmed entries
- regime fade entries
- volatility compression entries

But all of these must still pass local evidence gates.

## 17. Regime Filters

The user requested regime filters be part of every entry/exit style test, not a separate experiment.

So regime is now part of the strategy test matrix.

Every strategy should be tested across:

- pair
- timeframe
- strategy
- regime
- regime application style
- entry style
- exit style
- costs

The report should include:

- trades
- profit factor
- Sharpe
- max drawdown
- acceptance reason

## 18. ML Dataset Plan

The trade dataset turns historical candidate trades into ML rows.

One row equals one candidate trade entry.

Features must be known at entry time.

Feature groups:

- pair
- timeframe
- strategy
- entry style
- exit style
- z-score
- rolling z-score
- spread slope
- volatility rank
- correlation
- beta stability
- hedge ratio stability
- Hurst
- half-life
- cointegration
- copula fields
- funding drag
- regime
- Wizard confirmation fields
- dYdX liquidity fields

Labels:

- good trade
- profit after cost
- max adverse excursion
- max favorable excursion
- hold bars
- exit reason

Outputs:

- `data/ml/trade_training_dataset.parquet`
- `data/ml/trade_training_dataset.csv`
- `reports/ml/trade_dataset_summary.csv`
- `reports/ml/leakage_audit.csv`

Hard rule:

Dataset build fails if future data leaks into features.

## 19. Model Gate Plan

The model is not a strategy generator.

It is a trade-quality gate.

Start simple:

- logistic regression
- random forest
- gradient boosting
- optional LightGBM / XGBoost if installed

Model output:

```text
trade_quality_score = 0.00 to 1.00
```

Score policy:

- score >= 0.70: full entry
- 0.55 to 0.70: smaller entry
- score < 0.55: skip

Acceptance gates:

- profit factor improves
- drawdown drops
- Sharpe does not materially degrade
- trade count does not collapse
- take-rate remains useful
- score buckets are monotonic
- gains are not concentrated in one pair
- gains are not concentrated in one timeframe
- out-of-sample folds survive
- failure attribution is produced

## 20. Model-Gated Backtest

The model must prove it improves the system before it can affect behavior.

Compare:

- raw strategy
- rule-filtered strategy
- model-gated strategy
- model-sized strategy

Outputs:

- `reports/ml/model_gated_backtest.csv`
- `reports/ml/model_gated_acceptance.csv`
- `reports/ml/model_failure_attribution.csv`

If the model fails, it stays research-only.

No quantization.
No live scoring.
Dashboard shows blocked model state.

## 21. Quantization Plan

Quantization only happens after the model proves value.

Artifacts:

- `models/trade_gate/model.pkl`
- `models/trade_gate/model.onnx`
- `models/trade_gate/model_int8.onnx`
- `models/trade_gate/feature_schema.json`
- `models/trade_gate/metrics.json`
- `models/trade_gate/export_report.json`

Rules:

- no ONNX export unless model-gated backtest passes
- no int8 export unless ONNX parity passes
- quantized predictions must match original predictions within tolerance
- schema mismatch blocks scoring

## 22. Live Scoring Plan

Live scoring flow:

```text
new candle
→ update pair features
→ strategy proposes entry
→ validate feature schema
→ validate freshness
→ validate regime
→ model scores setup
→ enter / skip / resize
→ dashboard logs reason
```

Required blockers:

- dYdX market missing
- stale data
- funding missing or stale
- feature schema mismatch
- crisis regime
- no local acceptance evidence
- Crypto Wizards-only evidence

Outputs:

- `reports/dashboard/live_signals_dashboard.csv`
- `reports/dashboard/blocked_trades_dashboard.csv`
- `reports/dashboard/scoring_audit.csv`

## 23. Dashboard Command Center

The first dashboard version is report-backed, not a full UI.

Views:

- Pair Universe
- Candidate Ranking
- Strategy Tests
- Model Training
- Live Signals
- Blocked Trades
- Data Health
- API/Credit Usage

Every row should include:

- pair
- bucket
- discovery score
- acceptance score
- current regime
- current z-score
- Wizard confirmation
- dYdX tradeability
- model score if available
- action
- reason
- blocker
- stale reason
- evidence path
- next step

Dashboard rule:

Blockers must be as visible as scores.

## 24. Paper And Live Guardrails

Before paper or live trading:

- no trade without dYdX market availability
- no trade without current funding check
- no trade if feature schema mismatches
- no trade if data is stale
- no trade during crisis regime unless explicitly allowed
- no trade from Crypto Wizards/dashboard evidence alone
- every trade logs model score and reason
- every skipped trade logs blocker and reason
- paper outcomes are stored separately from backtest labels

Learning labels must stay separate:

- `backtest_label`
- `paper_label`
- `live_label`

A model trained only on backtests must be marked `backtest_trained`.

It cannot be called live-validated until enough real outcomes exist.

## 25. Archive Later

Archive only after active lineage is stable.

Readiness criteria:

- artifact index runs
- current state identifies active evidence paths
- pair universe builds
- trade dataset builds
- command dashboard builds
- active reports do not depend on unknown artifacts
- archive candidates are marked safe

Archive command should start dry-run only:

```bash
PYTHONPATH=src python -m quant_platform.cli archive-from-index --dry-run
```

Apply only after dry-run is clean:

```bash
PYTHONPATH=src python -m quant_platform.cli archive-from-index --apply
```

Unknown and `do_not_move` artifacts must never be archived automatically.

## 26. Gap Analysis Fixes Embedded In The Plan

The plan now addresses the major gaps:

- Evidence loss: fixed by non-destructive active layer.
- Fake precision: fixed by separating discovery score and acceptance score.
- Crypto Wizards hindsight risk: Wizard is discovery only.
- Wrong z-score comparison: fixed by exact mode capture.
- Scanner-only weakness: fixed by pair page diagnostics and backtest machine capture.
- ML leakage: fixed by entry-time feature rules and leakage audit.
- Model skipping everything: fixed by take-rate and trade count gates.
- Overfitting: fixed by walk-forward, pair concentration, and regime attribution.
- Premature quantization: blocked until model acceptance passes.
- Stale dashboard: fixed by freshness fields, blockers, and evidence paths.
- Paper/live confusion: fixed by separate labels.
- Unsafe archive: fixed by index-first dry-run archive.

## 27. BNB/STX Specific Work

The most recent candidate was:

```text
BNB-USD / STX-USD
```

Crypto Wizards pair page showed:

- Exchange: dYdX
- Timeframe: Daily
- Period: 320
- Exact mode: Static (Spread)
- spread_id: 3
- strategy_id: 1
- Entry Long X: >= 2.00
- Entry Short X: <= -2.00
- Exit Long X: <= 0.00
- Exit Short X: >= 0.00
- BNB weighting: 0.69
- Sharpe: 2.81
- Sortino: 6.64
- Net return: 47.8%
- Annualized return: 56.1%
- Win rate: 100.0%
- Closed trades: 1
- Max drawdown: -4.4%
- Pearson: 77.7%
- Spearman: 64.8%
- Kendall: 47.1%
- Best copula fit: Clayton
- Copula correlation: 55.6%
- BNB given STX: 5.2%
- STX given BNB: 83.4%

Important: cost settings were not visible on the Wizard page.

## 28. BNB/STX Local Data Problem And Fix

Before the fix, local BNB/STX daily data had only:

- 91 rows
- Date range ending 2026-06-09
- Too short for the Wizard 320-day period
- Stale versus the current project date

That made the local comparison unfair and incomplete.

The fix was to pull fresh dYdX daily history:

```bash
PYTHONPATH=src python -m quant_platform.cli run-dydx-long-history \
  --asset-x BNB-USD \
  --asset-y STX-USD \
  --pair-id bnb_stx_daily_320_fresh \
  --windows 2 \
  --limit 200 \
  --interval 1DAY \
  --derive-hedge-ratio \
  --allow-stale-fetch
```

Fresh result:

- Rows: 400
- Date range: 2025-05-21 to 2026-06-24
- File: `data/raw/pair_details/pair_bnb_stx_daily_320_fresh_1day_dydx_long_history_derived_history.json`
- Derived hedge ratio: 191.55742908722416
- Derived beta: 0.42431226654416865

This fixed the data-length and stale-data blocker.

The exact Wizard Static Spread test was then rerun on this fresh 400-row file.

Fresh local verification result:

- Pair: BNB-USD / STX-USD
- Exact mode: Static Spread
- Entry: +2 / -2
- Exit: zero cross
- Cost buckets tested: zero cost, base cost, stress cost
- Trades: 2
- Closed trades: 1
- Profit factor: 4.1021
- Sharpe: 2.4319
- Max drawdown: 39.57%
- Total return: 14.07%
- Acceptance: REJECT
- Acceptance reason: max drawdown > 15%; thin trade count

Conclusion:

The fresh data problem is fixed, but BNB/STX still does not pass local acceptance. It remains research-only because the drawdown is too high and the trade count is too thin.

## 29. Earlier BNB/STX Local Result Before Fresh Data

Using the old 91-row daily history, the local after-cost result was bad:

- Trades: 5
- Entries: 3
- Exits: 2
- Profit factor: 0.4846
- Sharpe: -2.8651
- Max drawdown: 6.84%
- Total return: -3.78%
- Acceptance: REJECT

Reasons:

- local history rows < 320
- total return <= 0
- Sharpe < 1.2
- profit factor < 1.8
- stale data

Cost comparison on old data:

- zero cost: still negative
- base cost: worse
- stress cost: worse again

Conclusion from old data:

The failure was not only because of costs. The local 91-row test did not replicate Wizard.

But the old test was not final because the data was too short and stale.

## 30. Current Status

Done:

- Project direction hardened.
- Active layer plan created.
- Pair universe plan created.
- Discovery vs acceptance scoring rule created.
- Wizard exact-mode handling added/planned.
- Wizard pair page diagnostics identified.
- Wizard backtest machine role defined.
- Regime overlays required for all strategy tests.
- ML dataset and leakage audit plan defined.
- Model gate and model-gated backtest plan defined.
- Quantization gates defined.
- Dashboard command center design defined.
- Paper/live guardrails defined.
- Safe archive-later approach defined.
- BNB/STX exact Wizard settings captured.
- Fresh 400-row BNB/STX daily dYdX history pulled.
- Fresh BNB/STX exact Static Spread local verification completed.
- Dashboard verification output updated.
- Full test suite passed after the verifier update: 286 passed.

Not finished yet:

- More Crypto Wizards daily candidates still need the same exact-mode local verification flow.
- Wizard cost assumptions are still unknown because the pair page did not visibly expose them.
- Full dashboard inventory still needs to be completed.
- ML dataset/model/quantization are designed but not production-proven.
- Paper/live trading is not ready.

## 31. Immediate Next Steps

Step 1:
Treat BNB/STX as research-only for now.

Reason:

- Local Sharpe and profit factor are interesting.
- Local total return is positive but below the original Wizard return.
- Drawdown is too high.
- Closed trade count is too thin.

Step 2:
Keep the Wizard comparison visible:

- Wizard Sharpe 2.81
- Wizard return 47.8%
- Wizard closed trades 1
- Wizard max drawdown -4.4%
- Local Sharpe 2.4319
- Local return 14.07%
- Local closed trades 1
- Local max drawdown 39.57%

Key question:
Why does Wizard show much lower drawdown and higher return than the local dYdX replay?

Step 3:
Check whether the difference comes from:

- Wizard cost assumptions
- Wizard spread construction
- hedge or weighting handling
- exact execution timing
- bar alignment
- mark/close price differences
- funding treatment
- open trade handling

Step 4:
Repeat this flow for more Crypto Wizards candidates:

```text
Sharpe > 2
returns_total > 20%
daily timeframe
capture exact mode
capture pair-page diagnostics
pull local dYdX data
run exact local verification
rank by acceptance evidence
```

## 32. Final Direction

The project is no longer just “find a high Sharpe pair.”

It is now an evidence machine.

Crypto Wizards helps find the spark.
The pair page explains the hypothesis.
The Wizard backtest machine gives bedrock settings.
Local dYdX data verifies or rejects the idea.
The pair universe ranks candidates.
The model only gates trades if it proves incremental value.
The dashboard shows scores, blockers, freshness, and reasons.
Paper/live trading only happens after guardrails pass.

That is the A-to-B route:

```text
old research repo
→ safer active layer
→ exact Crypto Wizards discovery
→ local dYdX proof
→ evidence-gated pair universe
→ leakage-safe ML dataset
→ proven model gate
→ quantized scoring
→ dashboard command center
→ guarded paper/live system
```