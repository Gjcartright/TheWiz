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
| P1 | crypto_wizards_capture | blocked | no_nested_execution_ready_history_candidate_detected | run updated browser capture helper on authenticated pair page |
| P2 | strategy_acceptance | blocked | no_strategy_passes_production_gates | review /Users/gregc/Documents/Codex/TheWiz-publish-20260625/reports/research_unblock_plan.csv and collect the highest-impact missing history/features |
| P3 | dydx_testnet_readiness | blocked | submit_orders_false;missing_wallet_address;missing_private_key;missing_dydx_order_client_adapter | keep order submission disabled until research passes and order adapter is injected |
| P4 | paper_execution_gate | blocked | strategy_or_dydx_gate_not_ready | capture price_x/price_y and rerun two-leg experiments |
| P5 | learning_event_store | blocked | missing_learning_events | append realized trade outcomes once research-gated paper signals exist |

## Gap Proof Required

### P1: crypto_wizards_capture
- Severity: `critical`
- Current evidence: `captures=1;research_spine_ready=1;best_completeness=100.00;next_focus=ready_for_research_spine;missing=none;quality_rows=1;research_usable=1;execution_usable=1;first_quality_blocker=none`
- Required proof: pair-detail capture with spread,zscore,ecm_x,ecm_y,ecm_strength,price_x,price_y history
- Source report: `reports/pair_detail_capture_checklist.csv;reports/pair_detail_quality_report.csv`
- Next action: run updated browser capture helper on authenticated pair page

### P2: strategy_acceptance
- Severity: `critical`
- Current evidence: `steps_ready=3/7;first_blocker=missing_two_leg_backtests`
- Required proof: production-eligible strategy with required two-leg base/stress results across multiple pairs
- Source report: `reports/strategy_acceptance_checklist.csv;reports/research_unblock_plan.csv`
- Next action: review /Users/gregc/Documents/Codex/TheWiz-publish-20260625/reports/research_unblock_plan.csv and collect the highest-impact missing history/features

### P3: dydx_testnet_readiness
- Severity: `high`
- Current evidence: `steps_ready=2/7;first_blocker=missing_wallet_address;missing_private_key`
- Required proof: submit flag, credentials, SDK, indexer, and authenticated order adapter all ready
- Source report: `reports/dydx_execution_checklist.csv`
- Next action: keep order submission disabled until research passes and order adapter is injected

### P4: paper_execution_gate
- Severity: `high`
- Current evidence: `steps_ready=0/4;first_blocker=no_strategy_passes_production_gates`
- Required proof: strategy_acceptance ready and dydx_testnet_readiness ready
- Source report: `reports/paper_execution_preflight.csv`
- Next action: capture price_x/price_y and rerun two-leg experiments

### P5: learning_event_store
- Severity: `medium`
- Current evidence: `events=0;outcomes=0;outcomes_remaining=100;ready_for_modeling=False`
- Required proof: paper journal or trade store contains outcome events for later modeling
- Source report: `reports/learning_event_summary.csv`
- Next action: append realized trade outcomes once research-gated paper signals exist

## Ranked Work Queue

| Rank | Gate | Depends On | Blocker | Command/Action |
|---:|---|---|---|---|
| 1 | crypto_wizards_live_artifacts |  | missing_live_payload_or_dictionary | crawl or import Crypto Wizards payloads |
| 2 | pair_detail_capture_audit | crypto_wizards_live_artifacts | no_nested_execution_ready_history_candidate_detected | run updated browser capture helper on authenticated pair page |
| 3 | strategy_acceptance | pair_detail_two_leg_execution_history | no_strategy_passes_production_gates | review /Users/gregc/Documents/Codex/TheWiz-publish-20260625/reports/research_unblock_plan.csv and collect the highest-impact missing history/features |
| 4 | dydx_testnet_readiness | strategy_acceptance | submit_orders_false;missing_wallet_address;missing_private_key;missing_dydx_order_client_adapter | keep order submission disabled until research passes and order adapter is injected |
| 5 | paper_execution_gate | strategy_acceptance;dydx_testnet_readiness | strategy_or_dydx_gate_not_ready | do not submit paper orders yet |
| 6 | learning_event_store | paper_execution_gate | missing_learning_events | append realized trade outcomes once research-gated paper signals exist |

## Operator Commands

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
- P2 fetch dYdX funding: `PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-funding --market ETH-USD,BTC-USD,SOL-USD`
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
- P5 learning report: `PYTHONPATH=src python3 -m quant_platform.cli learning-report`
- P5 learning outcome template: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template --output-path data/meta_learning/learning_outcome_template.csv`
- P5 learning outcome template check: `PYTHONPATH=src python3 -m quant_platform.cli learning-outcome-template-check --input-dir data/meta_learning/learning_outcome_template.csv`
- P5 import learning outcomes: `PYTHONPATH=src python3 -m quant_platform.cli import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv --output-path reports/learning_outcome_import_report.csv`
