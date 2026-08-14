# Pre-Goal Resumption Corrective Plan

Plan ID: `pre_goal_resumption_20260812T035708Z`

Status: **PLANNED - GOAL REMAINS PAUSED**

## Purpose

This is the durable source plan for repairing the dashboard self-inspection,
gap-analysis, premortem, and red-team findings before the seven-stage goal is
resumed. The detailed generated review packet is stored under:

`reports/supreme_team/pre_goal_resumption_2026-08-12/`

That directory is intentionally ignored as generated review output. This file
is the Git-eligible plan. The generated packet must also receive an explicit
artifact-index and encrypted-recovery disposition during implementation.

Passing this plan grants only permission to resume research at the existing
stage states. It grants no promotion, ML/RL, Testnet, or live authority.

## Current Stage State

| Stage | State | Continuity rule |
| --- | --- | --- |
| 1 | IN_PROGRESS, 2 of 7 daily receipts | Continue immutable daily collection; never fabricate a missing day |
| 2 | IN_PROGRESS, strict cost evidence missing | Continue read-only history and L2 evidence under frozen identity |
| 3 | IN_PROGRESS, parity/reconciliation incomplete | Run Wizard work only under immutable manifest and credit controls |
| 4 | BLOCKED | Wait for Stage 3 and the registered acceptance contract |
| 5 | BLOCKED | Keep ML and RL research-only until accepted Stage 4 evidence exists |
| 6 | BLOCKED | Keep Testnet no-order until candidate and lifecycle gates pass |
| 7 | BLOCKED | Keep live unauthorized until every prior gate and explicit approval pass |

## Repair Architecture

Use three isolated lanes:

1. **Evidence lane:** preserve time-bound Stage 1 and Stage 2 collection under a
   frozen runtime. Stage 3 collection is allowed only when its immutable
   manifest, credit budget, and source identity match.
2. **Corrective lane:** implement scope, capture, provenance, runtime, and test
   repairs against a quant-only baseline.
3. **Authority lane:** remain frozen. No corrective artifact may grant candidate
   promotion, model authority, order submission, or live authority.

## Phase A: Freeze Scope And Producer Identity

1. Build a quant-only allowlist for source, configuration, tests, governing
   documentation, and authoritative evidence.
2. Correct artifact-index ownership so unrelated `apps/the-ave` files are not
   active quant-pipeline evidence.
3. Record commit, dirty quant diff hash, config hashes, scheduler plist hashes,
   canonical interpreter, dependency fingerprint, and lockfile hash.
4. Bind every producer receipt to producer name, source hash, config hash,
   interpreter, dependency fingerprint, start time, and end time.
5. Add producer-lock and mid-run-drift checks around report builders.
6. Quarantine ambiguous receipts without deleting historical evidence.

Exit: every running producer has one reproducible identity and zero order
authority.

## Phase B: Implement The Pre-Resumption Gate

Add:

```bash
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli build-pre-goal-resumption-gate
```

Outputs:

- `reports/active/pre_goal_resumption_gate.csv`
- `reports/active/pre_goal_resumption_gate.json`
- `reports/active/pre_goal_resumption_gate.md`

Checks:

- seven-stage checkpoint hash and timestamp;
- Supreme Team packet hash and timestamp;
- quant baseline and producer identity;
- dashboard artifact registration and hashes;
- authenticated capture contract;
- scanner and pair-page completeness contracts;
- exact-mode, supporting-chart, and accounting lineage;
- canonical runtime identity;
- required test receipt;
- repair-window evidence reconciliation;
- false Testnet and live authority.

The only passing state is `PASS_RESUMPTION_CONTROL_READY`. It means research can
resume from the preserved stage states. It cannot mark any stage PASS.

## Phase C: Harden Crypto Wizards Capture

### Authentication

Extend existing browser metadata with requested route, final route,
authentication result, member-navigation proof, redirect result, and capture
time. Do not persist credentials, account details, API keys, Telegram values,
wallet secrets, or private keys.

### Scanner

Keep the existing `interval`, `sweep_interval`, and `scanner_interval` lineage.
Require:

- five venues by Daily and Hourly;
- all-case filters without Sharpe or return thresholds;
- raw response hashes and timestamps;
- duplicate preservation;
- declared, raw, and parsed row-count reconciliation;
- five consecutive no-growth bottom fetches for infinite-scroll completion;
- zero missing or unexpected venue-interval cells.

### Pair Pages

Build the expected universe from the complete scanner manifest. Preserve pair
ID and orientation. Capture every unique pair page and all seven exact modes,
or store an explicit unavailable reason and timestamp. Missing pairs remain in
the denominator.

### Mode And Accounting Lineage

Keep these fields independent:

- `exact_mode`
- `spread_type`
- `raw_supporting_chart_label`
- scanner and pair-page interval
- entries and closed trades
- open-position state
- realized and unrealized PnL
- `metric_accounting_state`

An unknown accounting state remains discovery-only and cannot enter ML labels
or RL rewards.

Exit: incomplete or unauthenticated evidence is blocked accurately and cannot
be reported as complete.

## Phase D: Artifact And Runtime Durability

1. Register and hash all five dashboard audit reports.
2. Assign every artifact a Git, generated, recovery-only, or secret disposition.
3. Keep the generated Supreme Team packet under the existing ignored reports
   policy, but include it in the encrypted evidence snapshot and artifact index.
4. Make `.venv312` the canonical operator and scheduler runtime.
5. Add Python, NumPy, scikit-learn, lockfile, and environment fingerprints to
   system-check and scheduler receipts.
6. Fail with `BLOCKED_RUNTIME_IDENTITY` when the wrong interpreter or dependency
   fingerprint is used.

Exit: a clean restore can identify and verify the same source, runtime, active
reports, and evidence disposition.

## Phase E: Red-Team Tests

Require negative tests for:

- forged authentication strings;
- protected-route redirects and missing member navigation;
- Daily rows relabeled Hourly;
- missing venue-interval cells;
- forged pagination completion;
- omitted, duplicated, or orientation-collapsed pair IDs;
- missing exact-mode cells;
- Copula mode overwritten by Dynamic support-chart context;
- zero-closed-trade performance used as ML labels or RL rewards;
- stale Supreme Team/checkpoint bindings;
- active producer and mid-run source drift;
- report existence without PASS status and matching identity;
- pre-resumption PASS used as downstream authority.

Run all verification from `.venv312`, including the focused dashboard suite,
the full required suite, system-check, scheduler readiness, artifact indexing,
and current-state regeneration.

Exit: every red-team mutation fails closed and required tests pass.

## Phase F: Reconcile And Resume

1. Reconcile every repair-window Stage 1, Stage 2, and Stage 3 receipt to the
   approved frozen runtime identity.
2. Quarantine and explain ambiguous receipts.
3. Refresh the canonical seven-stage checkpoint.
4. Generate a new Supreme Team packet bound to that checkpoint hash.
5. Build the pre-resumption gate.
6. Resume only when it reports `PASS_RESUMPTION_CONTROL_READY`.

After resumption, Stage 1 and Stage 2 continue from verified progress, Stage 3
continues its current parity and reconciliation work, and Stages 4 through 7
remain blocked until their original evidence gates pass.

## Definition Of Done

- pre-resumption gate passes;
- checkpoint and Supreme Team hashes match;
- quant scope and artifact dispositions are explicit;
- repair-window receipts are reconciled;
- dashboard auth and completeness cannot be self-asserted;
- pair-page coverage cannot silently omit pairs or modes;
- ambiguous mode/accounting data cannot enter ML or RL;
- canonical runtime passes required tests;
- original stage identities, contracts, statuses, and blockers remain intact;
- `testnet_order_authority=false`;
- `live_trading_authorized=false`.
