# Teacher Council And Student Stack

## Purpose

The Teacher Council organizes deterministic statistical strategies, independent critics, supervised students, and later adaptive challengers without transferring execution authority to a model.

```text
Crypto Wizards discovery
  -> Hyperliquid point-in-time feature reconstruction
  -> seven Math V2 exact-mode teachers
  -> dependency, regime, risk, cost, execution, and outcome critics
  -> deterministic Teacher Council
  -> advisory router and outcome students
  -> shadow research only
  -> separate paper/Testnet authorization controls
```

Crypto Wizards is discovery-only. A Wizard Sharpe ratio, return, exact mode, chart, or conditional probability can nominate a hypothesis but cannot vote toward acceptance, become a training label, or authorize a trade.

On the Expansion checkout, set `UV_PROJECT_ENVIRONMENT` to the Mac internal
runtime directory and run `uv sync --extra dev --locked` before the commands below.

The seven exact-mode names identify local specialist lanes; they do not certify parity with the corresponding Wizard formulas.

## Exact-Mode Teachers

Every council context requires one proposal from each specialist:

1. Static Spread
2. Static ZScoreR
3. Dyn Spread
4. Dyn ZScoreR
5. OU Spread
6. OU ZScoreR
7. Copula

Each local proposal must identify its entry, exit, invalidation, regime, expected after-cost return, lower-bound after-cost return, confidence, uncertainty, formula version, Math V2 version, timestamps, and evidence paths. A specialist may explicitly abstain when its mode does not fit the context.

## Critics

Every context also requires six independent assessments:

- Dependency
- Regime
- Risk
- Cost
- Execution
- Outcome

A veto overrides every teacher and student score. Warnings, unknown states, stale evidence, future timestamps, missing critics, or non-point-in-time evidence force abstention or a blocked state.

## Student Stack

The mixture router predicts a probability for every exact mode plus `abstain`. The distributional outcome student forecasts positive-return probability, after-cost return bounds, MAE, MFE, holding time, structural-break risk, execution-failure risk, and uncertainty.

Students are advisory. They can force additional abstention when support, completeness, uncertainty, mode agreement, or return bounds are weak. They cannot override a critic veto or turn a blocked teacher packet into a shadow test.

The current topology trains one dedicated supervised specialist per exact mode and keeps an inverse-frequency weighted pooled baseline for comparison. This preserves each mode's distinct signal geometry while preventing the higher-frequency Copula lane from dominating aggregate diagnostics.

## Learning Gates

The canonical student dataset is `data/ml/student_teacher_training_dataset.csv`. Training is blocked unless the audit confirms:

- features precede labels
- every row is point-in-time
- every row uses Math V2
- Wizard is never label authority
- dashboard hindsight is absent
- after-cost outcomes and evidence paths are present
- both outcome classes exist
- all seven exact modes are represented
- pair and mode concentration are bounded
- multiple pairs and timeframes are represented

The primary evidence lane uses five purged, expanding daily folds. A separate five-fold 4-hour lane is local robustness evidence; it does not rewrite the Wizard-discovered timeframe. Both lanes write one row per closed out-of-sample trade, including entry-time features, after-cost outcome, MAE, MFE, hold bars, exit reason, and evidence lineage.

Contextual-bandit evaluation additionally requires a behavior policy with logged nonzero probability for alternative actions. Deterministic replay propensities are valid supervised lineage but provide no counterfactual support, so they intentionally leave the bandit lane blocked. That block does not invalidate supervised shadow training. Offline RL remains blocked until genuine exploratory Testnet outcomes supply support.

Model evaluation uses timestamp-grouped chronological folds. Every training label that reaches or crosses the earliest test entry is purged globally, regardless of pair, and a per-pair embargo then removes the final surviving training periods before each test window. The fold and prediction evidence record the global purge result, pair coverage, timestamp coverage, test start, and maximum surviving training-label timestamp. Stage 5 fails closed unless every surviving training label ends before its fold's test window begins.

Daily and auxiliary 4-hour strategy tests are controlled as one research family. Benjamini-Hochberg correction is applied across both lanes, while an idempotent alpha-spending registry charges only changed evidence families. Re-running identical evidence does not spend the budget again. Auxiliary results remain robustness evidence and never rewrite the Wizard-discovered primary timeframe.

## Runtime Files

Input event streams:

- `reports/orchestration/teacher_council/teacher_proposals.jsonl`
- `reports/orchestration/teacher_council/critic_assessments.jsonl`
- `reports/orchestration/teacher_council/student_router_predictions.jsonl`
- `reports/orchestration/teacher_council/student_outcome_forecasts.jsonl`

Generated controls:

- `reports/orchestration/teacher_council/teacher_registry.csv`
- `reports/orchestration/teacher_council/wizard_discovery_hypotheses.csv`
- `reports/orchestration/teacher_council/council_decisions.csv`
- `reports/orchestration/teacher_council/teacher_council_readiness.csv`
- `reports/orchestration/teacher_council/student_training_readiness.csv`
- `reports/orchestration/teacher_council/student_training_schema.csv`
- `reports/orchestration/teacher_council/teacher_council.md`
- `reports/active/hyperliquid_run_manifest.json`
- `reports/active/hyperliquid_run_candidates.csv`
- `reports/orchestration/teacher_council/walkforward_fold_evidence.csv`
- `reports/orchestration/teacher_council/walkforward_trade_events.csv`
- `reports/orchestration/teacher_council/auxiliary_4h_walkforward_mode_summary.csv`
- `reports/orchestration/teacher_council/auxiliary_4h_walkforward_trade_events.csv`
- `reports/orchestration/teacher_council/student_mode_training_manifest.csv`
- `reports/orchestration/teacher_council/statistical_selection_controls.csv`
- `reports/orchestration/teacher_council/portfolio_critic.csv`
- `reports/active/hyperliquid_testnet_margin_snapshot.csv`
- `reports/active/hyperliquid_testnet_lifecycle_gate.csv`
- `reports/active/hyperliquid_authority_state.csv`
- `reports/orchestration/hyperliquid_run/research_cycle_receipt.csv`

## Canonical Command

```bash
uv run --locked python -m quant_platform.cli \
  run-hyperliquid-research-cycle --collect-l2
```

This command freezes one candidate-set identity, refreshes its public L2 sample,
rebuilds matching cost evidence, runs purged walk-forward tests for all seven
exact modes, materializes teacher and critic packets, audits student readiness,
applies the portfolio critic, runs the no-order Testnet preflight and lifecycle
gate, and writes a hash-backed receipt. It never submits an order.

For an offline replay that reuses collected L2 evidence, omit `--collect-l2`.

## Testnet Approval

The approval template is non-approved by default. Before a bounded lifecycle test, the user must complete the exact payload and explicitly set approval. A separate Keychain-backed secret then signs a canonical hash containing the run ID, candidate-set ID, master and agent addresses, expiry, notional limit, and both exact legs:

```bash
uv run --locked python -m quant_platform.cli \
  hyperliquid-testnet-sign-smoke-approval
```

The signing command submits no order. Any later edit to a signed field invalidates both the payload hash and signature. The lifecycle gate still requires submit, partial-fill recovery, cancel/unwind, reconciliation, idempotency, and emergency-flatten receipts.

The canonical run may remain `RESEARCH_ONLY` even when supervised students are ready. Strategy selection, critic, portfolio, margin, approval, lifecycle, and realized-paper gates remain independent. `SHADOW_TEST` never means paper or live execution authorization.
