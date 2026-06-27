# Supreme Team Execution Plan (from latest checkpoint)

Latest run: `supreme_team_2026-06-26_181111Z`  
Open actions: **20** (open_actions: 20, pass_gates: 0, critical: 8, high: 8, medium: 4)

Use this plan whenever you ask for the next checkpointed next step.

## Blocked Path A — Deployment Readiness (priority order)

1. **P1 — Restore live Crypto Wizards payload dictionary flow**  
   - **Blocker:** `missing_live_payload_or_dictionary`  
   - **Why now:** both `priority_readiness` and `priority_action_plan` show this is the first dependency for P1 work.  
   - **Next action:** crawl or import fresh Crypto Wizards payloads so latest pair pages are present as first-class evidence.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli crawl-crypto-wizards`
     - `PYTHONPATH=src python -m quant_platform.cli ingest-crypto-wizards-scanner`
     - `PYTHONPATH=src python -m quant_platform.cli verify-crypto-wizards-live-artifacts`

2. **P1 — Refresh pair-detail capture audit for execution-ready pairs**  
   - **Blocker:** `no_nested_execution_ready_history_candidate_detected`  
   - **Why now:** 18 captures are present but no nested execution-ready candidate set (`experiment_ready_paths=0`, `ecm_ready_paths=0`, `two_leg_ready_paths=0`) even though `research_spine_ready=18`.  
   - **Next action:** run updated browser capture helper and import the enriched capture payload (including spread, zscore, ECM fields).  
   - **Commands**
     - `./scripts/copy_crypto_wizards_capture_helper.sh`
     - `PYTHONPATH=src python -m quant_platform.cli capture-preflight --json-path /path/to/crypto_wizards_pair_capture.json`
     - `PYTHONPATH=src python -m quant_platform.cli import-latest-pair-detail-download`
     - `PYTHONPATH=src python -m quant_platform.cli pair-detail-capture-checklist`
     - `PYTHONPATH=src python -m quant_platform.cli pair-detail-quality`

3. **P2 — Unblock strategy acceptance**
   - **Blocker:** `no_strategy_passes_production_gates` (`missing_two_leg_backtests`, `missing_hedge_beta_or_funding_inputs`, `missing_required_cost_buckets`)  
   - **Why now:** this is the top gate for production readiness and paper execution.  
   - **Next action:** rerun targeted execution-realistic pair detail experiments for top candidates and close missing two-leg inputs.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli research-unblock-plan`
     - `PYTHONPATH=src python -m quant_platform.cli strategy-acceptance-checklist`
     - `PYTHONPATH=src python -m quant_platform.cli funding-requirements`
     - `PYTHONPATH=src python -m quant_platform.cli fetch-dydx-funding --market ETH-USD,BTC-USD,SOL-USD`
     - `PYTHONPATH=src python -m quant_platform.cli funding-coverage --funding-path data/processed/dydx_funding.csv`
     - `PYTHONPATH=src python -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 --run-research`
     - `PYTHONPATH=src python -m quant_platform.cli zscore-threshold-sweep --funding-path data/processed/dydx_funding.csv`

4. **P3 — Keep execution lane hard-gated**
   - **Blocker:** `submit_orders_false` (intentional hard-stop)  
   - **Why now:** paper/paper-like order flows remain blocked until strategy and venue lanes are stable.  
   - **Next action:** do not override; keep `DYDX_TESTNET_SUBMIT_ORDERS=false` until gates clear.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli dydx-execution-checklist`
     - `PYTHONPATH=src python -m quant_platform.cli dydx-order-adapter-contract`
     - `PYTHONPATH=src python -m quant_platform.cli paper-plan --pair <PAIR> --strategy-id <STRATEGY_ID> --signal <SIGNAL>` (after strategy acceptance clears)

5. **P4 — Do not submit paper yet**
   - **Blocker:** `strategy_or_dydx_gate_not_ready`  
   - **Next action:** run `paper-execution-preflight` after strategy + dYdX acceptance clears and use it as pre-trade stop condition.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli paper-execution-preflight`
     - `PYTHONPATH=src python -m quant_platform.cli paper-venue-preflight --pair ETH-BTC`

6. **P5 — Build learning outcomes before any model score rule changes**
   - **Blocker:** `missing_learning_events` (`events=0`, `ready_for_modeling=False`)  
   - **Next action:** start paper outcome collection once paper signals are cleared, then feed into model-learning inputs.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli learning-outcome-template --output-path data/meta_learning/learning_outcome_template.csv`
     - `PYTHONPATH=src python -m quant_platform.cli import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv --output-path reports/learning_outcome_import_report.csv`
     - `PYTHONPATH=src python -m quant_platform.cli learning-report`

## Blocked Path B — RL + Model Acceptance Follow-up

7. **RL status check**
   - Latest: `run-rl-research accepted=False` with blocker `rl_acceptance_gates_not_met`; `rl_training_report` blocked by `rl_live_use_blocked`.  
   - **Next action:** keep RL in research mode, complete acceptance gate inputs from live-readiness and strategy gates, then rerun RL pipeline.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli run-rl-research`
     - `PYTHONPATH=src python -m quant_platform.cli train-rl-ppo`
     - `PYTHONPATH=src python -m quant_platform.cli export-rl-policy`
     - `PYTHONPATH=src python -m quant_platform.cli export-rl-policy --apply` (when `accepted=True`)

8. **ML trade-gate follow-up**
   - Latest: `model_gated_backtest` did not pass (`model_gated_backtest_not_accepted`), incremental edge gate failed.  
   - **Next action:** compare strategy-level failure attribution, then run alternative candidate-focused passes after acceptance blockers are resolved.  
   - **Commands**
     - `PYTHONPATH=src python -m quant_platform.cli run-model-gated-backtest`
     - `PYTHONPATH=src python -m quant_platform.cli strategy-family-sweep`
     - `PYTHONPATH=src python -m quant_platform.cli strategy-family-sweep-failure-attribution`

9. **Checkpoint cadence**
   - After each major unblocking batch, run:
     - `PYTHONPATH=src python -m quant_platform.cli supreme-team`
     - `PYTHONPATH=src python -m quant_platform.cli priority-runbook`
     - `PYTHONPATH=src python -m quant_platform.cli paper-execution-preflight`

## Recommended execution sequence (Supreme Team order)

1) P1-crypto_wizards_live_artifacts → 2) P1-pair_detail_capture_audit → 3) P2-strategy_acceptance → 4) P3-dydx_testnet_readiness → 5) P4-paper_execution_gate → 6) P5-learning_event_store, with RL/ML checks run in parallel after each stage's unblocking artifacts refresh.
