# Priority Spine Runbook

Generated from the current P1-P5 readiness reports.

## Project Objective

Source: `/Users/gregc/Documents/Codex/TheWiz-publish-20260625/project_objective.md`
```text
# QUANTIZED DYDX STATISTICAL ARBITRAGE AGENT MEMORY

## PROJECT OBJECTIVE

Build a production-grade statistical arbitrage trading system for dYdX that identifies and trades mean-reverting crypto pairs using:

- Cointegration
- Z-Score
- Hedge Ratio
- Half-Life
- ECM (Error Correction Model)
- Copula Dislocation Analysis
- Machine Learning Trade Scoring
- Automated Risk Management

The system must eventually operate autonomously while remaining fully observable and controllable.

---

# SUCCESS CRITERIA

Primary Success Metric:

Profit Factor >= 1.80

Secondary Metrics:

Win Rate >= 55%

Maximum Drawdown <= 20%

Sharpe Ratio > 1.5

No single pair contributes >10% of total drawdown

All executions logged

No orphaned positions

No unhedged exposure

---

# CORE TRADING THESIS

Markets contain temporary pricing inefficiencies between related assets.

When statistically related assets diverge beyond normal behavior:

1. Open long position on undervalued asset
2. Open short position on overvalued asset
3. Wait for spread reversion
4. Close positions
5. Capture convergence profit

The objective is NOT directional prediction.

The objective IS spread mean reversion.

---

# CURRENT STRATE...
```

## Current Dashboard

| Priority | Area | Status | Blocker | Next Action |
|---|---|---|---|---|
| P1 | crypto_wizards_capture | ready |  |  |
| P2 | strategy_acceptance | ready |  | allow research-gated paper plans |
| P3 | dydx_testnet_readiness | blocked | submit_orders_false | keep order submission disabled until research passes and order adapter is injected |
| P4 | paper_execution_gate | blocked | strategy_or_dydx_gate_not_ready | leave DYDX_TESTNET_SUBMIT_ORDERS=false until research and adapter gates pass |
| P5 | learning_event_store | blocked | missing_learning_events | append realized trade outcomes once research-gated paper signals exist |

## Gap Proof Required

### P3: dydx_testnet_readiness
- Severity: `high`
- Current evidence: `steps_ready=5/7;first_blocker=submit_orders_false`
- Required proof: submit flag, credentials, SDK, indexer, and authenticated order adapter all ready
- Source report: `reports/dydx_execution_checklist.csv`
- Next action: keep order submission disabled until research passes and order adapter is injected

### P4: paper_execution_gate
- Severity: `high`
- Current evidence: `steps_ready=1/4;first_blocker=submit_orders_false`
- Required proof: strategy_acceptance ready and dydx_testnet_readiness ready
- Source report: `reports/paper_execution_preflight.csv`
- Next action: leave DYDX_TESTNET_SUBMIT_ORDERS=false until research and adapter gates pass

### P5: learning_event_store
- Severity: `medium`
- Current evidence: `events=0;outcomes=0;outcomes_remaining=100;ready_for_modeling=False`
- Required proof: paper journal or trade store contains outcome events for later modeling
- Source report: `reports/learning_event_summary.csv`
- Next action: append realized trade outcomes once research-gated paper signals exist

## Ranked Work Queue

| Rank | Gate | Depends On | Blocker | Command/Action |
|---:|---|---|---|---|
| 1 | dydx_testnet_readiness | strategy_acceptance | submit_orders_false | keep order submission disabled until research passes and order adapter is injected |
| 2 | paper_execution_gate | strategy_acceptance;dydx_testnet_readiness | strategy_or_dydx_gate_not_ready | do not submit paper orders yet |
| 3 | learning_event_store | paper_execution_gate | missing_learning_events | append realized trade outcomes once research-gated paper signals exist |

## Operator Commands

- P0 gap analysis checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli gap-analysis-checklist`
- P1 copy browser capture helper: `./scripts/copy_crypto_wizards_capture_helper.sh`
- P1 capture checklist: `PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist`
- P1 browser status after refresh: `await __CW_CAPTURE_STATUS__()`
- P1 browser download after useful status: `await __CW_DOWNLOAD_CAPTURE__()`
- P1 import latest browser download: `PYTHONPATH=src python3 -m quant_platform.cli import-latest-pair-detail-download`
- P1 capture preflight: `PYTHONPATH=src python3 -m quant_platform.cli capture-preflight --json-path /path/to/crypto_wizards_pair_capture.json`
- P2 funding requirements: `PYTHONPATH=src python3 -m quant_platform.cli funding-requirements`
- P2 funding CSV template: `PYTHONPATH=src python3 -m quant_platform.cli funding-template --output-path data/processed/dydx_funding_template.csv`
- P2 funding template check: `PYTHONPATH=src python3 -m quant_platform.cli funding-template-check --input-dir data/processed/dydx_funding_template.csv`
- P2 import funding template: `PYTHONPATH=src python3 -m quant_platform.cli import-funding-template --input-dir data/processed/dydx_funding_template.csv --output-path data/processed/dydx_funding.csv`
- P2 fetch dYdX funding: `PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-funding --market AAVE-USD,ALGO-USD,APT-USD,ARB-USD,AVAX-USD,BNB-USD,BONK-USD,BTC-USD,DOGE-USD,ETC-USD,ETH-USD,LINK-USD,LTC-USD,MATIC-USD,MKR-USD,OP-USD,PENGU-USD,SOL-USD,UNI-USD,XRP-USD`
- P2 funding coverage: `PYTHONPATH=src python3 -m quant_platform.cli funding-coverage --funding-path data/processed/dydx_funding.csv`
- P2 funded research spine: `PYTHONPATH=src python3 -m quant_platform.cli funded-research-spine --funding-path data/processed/dydx_funding.csv`
- P2 strategy acceptance: `PYTHONPATH=src python3 -m quant_platform.cli strategy-acceptance-checklist`
- P2 research unblock plan: `PYTHONPATH=src python3 -m quant_platform.cli research-unblock-plan`
- P2 z-score threshold sweep: `PYTHONPATH=src python3 -m quant_platform.cli zscore-threshold-sweep --funding-path data/processed/dydx_funding.csv`
- P2 dYdX pair expansion plan: `PYTHONPATH=src python3 -m quant_platform.cli dydx-pair-expansion-plan --max-pairs 10 --limit 1000`
- P2 dYdX long-history plan: `PYTHONPATH=src python3 -m quant_platform.cli dydx-long-history-plan --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000`
- P2 run shell-backed dYdX long-history workflow: `bash scripts/run_dydx_long_history.sh --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --funding-path data/processed/dydx_funding.csv`
- P2 shell-backed strict long-history workflow: `bash scripts/run_dydx_long_history.sh --strict --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --funding-path data/processed/dydx_funding.csv`
- P2 run dYdX long-history workflow: `PYTHONPATH=src python3 -m quant_platform.cli run-dydx-long-history --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --derive-hedge-ratio --run-research --research-funding-path data/processed/dydx_funding.csv`
- P2 build dYdX long-history pair: `PYTHONPATH=src python3 -m quant_platform.cli build-dydx-long-history-pair --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --interval 5mins --derive-hedge-ratio --run-research --research-funding-path data/processed/dydx_funding.csv`
- P2 run dYdX pair expansion: `PYTHONPATH=src python3 -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 --run-research`
- P3 adapter contract: `PYTHONPATH=src python3 -m quant_platform.cli dydx-order-adapter-contract`
- P3 dYdX readiness: `PYTHONPATH=src python3 -m quant_platform.cli dydx-execution-checklist`
- P4 paper preflight: `PYTHONPATH=src python3 -m quant_platform.cli paper-execution-preflight`
- P4 paper venue preflight: `PYTHONPATH=src python3 -m quant_platform.cli paper-venue-preflight --pair ETH-BTC`
- P5 learning report: `PYTHONPATH=src python3 -m quant_platform.cli learning-report`
- P5 learning outcome template: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template --output-path data/meta_learning/learning_outcome_template.csv`
- P5 learning outcome template check: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template-check --input-dir data/meta_learning/learning_outcome_template.csv`
- P5 import learning outcomes: `PYTHONPATH=src python3 -m quant_platform.cli import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv --output-path reports/learning_outcome_import_report.csv`
- pre-mortem checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli pre-mortem-checklist`
- post-mortem checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli post-mortem-checklist`
- supreme team checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli supreme-team`
- red-team checkpoint: `PYTHONPATH=src python3 -m quant_platform.cli red-team-checklist`
