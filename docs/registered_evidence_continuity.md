# Registered Evidence Continuity

## Purpose

Daily Crypto Wizards discovery is a moving research board. A registered Stage 4
hypothesis is an immutable prospective test. The moving board must not erase or
replace the evidence required by the frozen test.

## Candidate Lanes

The read-only Hyperliquid L2 collector maintains two explicit lanes:

- `registered_rerun_contract`: frozen candidates that control Stage 2 acceptance.
- `current_wizard_family`: rotating exploratory candidates that continue collecting
  evidence but do not invalidate an already registered test.

If the same asset pair appears in both lanes, the frozen contract identity wins the
deduplication. Every row remains research-only and carries no Testnet or live order
authority.

## Stage 2 Authority

Stage 2 readiness is calculated only for `stage_two_candidate=true` rows. Before a
rerun contract exists, the current registered hypothesis batch supplies those rows.
After a contract exists, only the immutable contract candidates control Stage 2.

The broader discovery set is still reported separately as exploratory cost coverage.
Missing exploratory evidence is visible, but it cannot make a completed frozen test
disappear every time the daily board changes.

## Frozen Source Family

Each registered rerun contract is bound to the exact full experiment matrix hash and
row count that existed when the contract was created. The matrix is copied into:

`data/research/registered_rerun_source_families/<contract_id>/`

The receipt records the original snapshot, immutable copy, SHA-256 hash, row count,
and zero-authority boundary. Gate identity is validated against this frozen matrix,
not the latest daily matrix.

## Immutable Cost Bundles

Every L2 cost-model build now creates an immutable bundle under:

`data/research/l2_cost_model_receipts/<bundle_id>/`

The bundle retains:

- candidate set
- cost-collection status
- strict L2 window used by the model
- fee profile
- pair cost models
- stress surface
- receipt with hashes and row counts

The active CSV files remain convenient current views. The immutable bundle is the
evidence authority for historical gate decisions.

## Isolated Stage 4 Replay

The production registered rerun does not execute against `reports/active`. After
the gate is ready, the executor creates a contract-specific root under:

`data/research/registered_rerun_workspaces/<execution_id>/`

The workspace is seeded with copies of the frozen Wizard handoff and experiment
matrix, its compatible point-in-time history, the ready receipt and proof lineage,
the immutable strict-cost bundle, frozen funding evidence, policy receipts, and
venue constraint files. A workspace receipt binds every seeded input by SHA-256.

All canonical replay, observed-cost, walk-forward, regime, robustness,
concentration, failure-attribution, leverage, learning-ledger, and chain-validation
outputs are written inside this private root. Daily discovery can rotate while the
registered replay runs without changing either its inputs or outputs. A changed
workspace input blocks execution.

After reconciliation, the exact conclusive receipt is copied into the canonical
append-only conclusion store. This is the only bridge that permits a later daily
board to create a new contract generation; the completed contract and all of its
evidence remain immutable and linked as the prior generation.

The main checkpoint may recognize Stage 4 only from either the legacy accounted
gate or a content-hash-valid immutable execution receipt with a conclusive registered
outcome. The active execution status by itself is not sufficient.

## Mode-Specific Wizard Evidence

Stage 3 does not pretend every endpoint exposes the same mathematical object.
The registered gate requires exact custom-series formula reconstruction for the
12 Static, Dynamic, and OU mode-orientation cells. For the two Copula orientations,
it requires repeatable, orientation-symmetric behavioral output from the dedicated
Copula endpoint, complete provenance, and an immutable cohort receipt. Behavioral
evidence cannot satisfy a non-Copula formula cell, and Copula behavioral evidence is
never mislabeled as formula parity.

The mode-parity report also separates absence from disproof. A completed vendor
response that exceeds the frozen comparator tolerance is recorded as
`vendor_response_captured_but_formula_parity_not_proven`, together with its observed
vendor status, maximum absolute error, and request/response provenance. It is not
reported as missing evidence and it cannot receive formula authority. This is how
the Dynamic v1 mismatch remains visible while the preregistered Dynamic v2 candidate
waits for its disjoint four-cell holdout.

Within one bounded proof cycle, a client-rejected batch may advance to unrelated
registered cells only when every failed identity has an immutable current-UTC-day
request receipt and the proof runner confirms those unique identities are
quarantined from same-day reselection. A generic zero-progress batch still stops
immediately. Quarantined failures remain blockers, do not count as captured
responses, and do not increase the daily batch or credit ceilings; the continuation
only prevents one malformed mode payload from starving the rest of the frozen queue.

The daily Stage 3 proof lane has one shared 68-credit reservation: at most 60 credits
for 30 custom-series exact-mode requests and 8 credits for the four two-repeat
Copula behavioral cells. Exact-mode and Copula calls use the same reservation and
the same reconciliation receipt. The scheduler records attempted and completed
credits separately for each endpoint class. A request counts as completed only when
its response is captured in that cycle; failed or missing responses remain
uncompleted attempted credits. A claimed Copula parity pass with incomplete response
accounting is invalidated and cannot contribute accepted evidence or trigger the
registered Stage 4 handoff.

Stage 4 does not trust the active scheduler summary by itself. Before creating or
reusing a registered-rerun-ready receipt, it recomputes the proof-cycle content ID,
binds every material parity and credit field to that cycle, and independently
validates the sealed proof-lane reservation and reconciliation in the UTC-day credit
ledger. Attempted credits must equal completed credits, unresolved attempted credits
must be zero, and the ledger's request and credit counts must match the proof cycle.
Any active-summary contradiction, altered cycle, missing ledger receipt, or partial
response accounting blocks the handoff.

## Failure Rules

The pipeline fails closed when:

- the active contract and immutable contract differ;
- a frozen candidate cannot be matched to its ledger record;
- either pair leg is not currently tradable on Hyperliquid;
- the frozen source-family hash or row count changes;
- a cost-bundle file changes after publication;
- an isolated workspace input changes after its receipt is published;
- a stage runner points at a source family other than the frozen matrix;
- strict funding, L2 cadence, fee freshness, depth, or cost requirements fail.

None of these receipts grant promotion, Testnet, or live order authority.

## Daily Execution Versus Planning State

The seven-day cadence advances only from immutable dated execution receipts. A
planning run writes to `current_wizard_hyperliquid_daily_plan_*`,
`daily_schedule_plan_status.csv`, and `daily_research_plan_authority.json`. It cannot
overwrite the current executed manifest, stage status, scheduler status, or research
authority files.

The selected execution receipt for a calendar day remains frozen after its first
fully validated pass. From 2026-08-11 onward, receipt validity also requires all 19
stage semantic receipts, including the full Wizard sweep, raw-capture bindings,
credit reservation/reconciliation evidence, artifact hashes, and zero order
authority. Cadence qualification is stricter than legacy receipt validity: every
counted day must prove that semantic contract. A pre-cutoff run can qualify only
when its immutable artifacts prove the same facts; missing manifest convenience
fields may be replaced by a hash-bound resolved-command isolation audit, but
missing semantic evidence is never grandfathered into the seven-day count.
A blocked or failed executed run now returns a nonzero LaunchAgent exit in addition
to publishing its immutable failure evidence.

`daily_schedule_status.csv` separates the latest install action from accumulated
execution evidence. `install_action_status=INSTALLED_NOT_STARTED` describes only the
plist write performed by that invocation. `scheduler_status` advances to
`INSTALLED_WITH_QUALIFYING_RECEIPTS` after validated daily receipts exist and to
`INSTALLED_CADENCE_ACCEPTED` only after seven consecutive qualifying calendar days.
This distinction cannot satisfy the seven-day gate early; it prevents a successful
receipt history from being mislabeled as a scheduler that has never run.

## Automatic Stage 5 Handoff Recovery

The installed proof launcher is also the research-only recovery path between an
accepted Stage 4 execution and registered Stage 5 learning. If Stage 4 is hash-valid,
accepted, bound to the active checkpoint, and contains at least three independent
supporting clusters, three independent supporting pairs, three full-survivor
clusters, three full-survivor pairs, and three final survivors, the launcher may
retry a missing or transient Stage 5 handoff on the same UTC day.

This internal continuation:

- invokes no Crypto Wizards endpoint and consumes no additional Wizard credit;
- reuses the immutable accepted Stage 4 execution and frozen survivor identities;
- runs no Testnet or live order action;
- preserves zero promotion, candidate, Testnet-order, and live authority;
- treats `PASS_RESEARCH_LEARNING_GATES` and
  `REJECTED_RESEARCH_LEARNING_GATES` as terminal only when the immutable learning
  receipt passes the same full registered-learning reconciliation used by the
  canonical checkpoint: safe path and file hash, receipt self-hash, current Stage 4
  execution lineage, every required artifact role and hash, prospective protocol
  binding, and recomputed supervised, RL, and governance acceptance checks;
- accepts an active `ALREADY_COMPLETE` pointer only when its underlying immutable
  receipt independently reconciles to a verified PASS or verified REJECTED result;
- returns a nonzero launcher exit when the scheduler claims a terminal learning
  result but the immutable learning receipt is missing or invalid.

Malformed or unverified Stage 4 acceptance evidence does not trigger recovery.
Fully reconciled Stage 5 PASS or REJECTED evidence is never retried automatically;
a merely structural or rehashed receipt remains recoverable and cannot suppress the
handoff. The canonical seven-stage checkpoint independently repeats the audit before
it can publish `Stage 5 = PASS`. It validates the active terminal pointer,
the immutable Stage 4 execution embedded in the learning receipt, the receipt file
hash and self-hash, all required artifact roles and artifact hashes, the prospective
protocol binding, and then recomputes the supervised, RL, and governance acceptance
checks from those artifacts. A mutable status flag cannot satisfy Stage 5. A
reconciled rejection is preserved as valid research evidence but remains a blocked
Stage 5 outcome and grants no Testnet authority.

## Stage 4 Generation Rollover

When an active registered contract concludes while preregistered current-family
hypotheses remain unaccounted, the proof launcher may request a same-day internal
generation continuation. The continuation makes no Wizard request. Completion is
recognized only when the active contract ID changes, the new contract names the
prior contract and advances the generation, the new execution receipt is hash-valid
and authority-free, and the refreshed Stage 4 checkpoint is terminal and bound to
the new contract.

A fully accounted family that produces fewer than the frozen minimum of three
independent full-survivor clusters and pairs is a conclusive research rejection, even
when one or two individual hypotheses survive. It cannot remain indefinitely marked
as an accounting inconsistency, and it cannot launch Stage 5, Testnet, or live
trading. Only a fully accounted family meeting every breadth gate receives the
accepted Stage 4 terminal outcome.

## Proof Scheduler Concurrency And Checkpoint Publication

The proof LaunchAgent polls every ten minutes, but a proof-cycle lock remains valid
for four hours. Lock age alone cannot evict a process whose recorded PID is still
alive. This prevents a long exact-mode, Copula, registered-rerun, or learning handoff
from overlapping with the next polling event and duplicating vendor requests.

Every actual external proof attempt refreshes the seven-stage checkpoint, including
attempts that capture no usable response. The checkpoint therefore records the
attempt blocker and next eligible UTC reset instead of leaving a stale success or
stale readiness state. Preflight-only and same-day deferred launcher checks continue
to make no external request.

## Reviewed Dynamic-v2 Continuation

Dynamic-v2 supersession remains explicitly reviewed and is never applied by the
launcher. After all four disjoint holdout cells pass and a reviewer applies the
immutable research-only activation, the launcher may run one same-day internal
continuation to recompute the already captured Dynamic response histories. No new
Wizard request is made.

The continuation requires a canonical activation ID, a hash-valid immutable
activation receipt, a bound proof snapshot, a substantive review note, the submitted
review-packet ID, generation 2, and zero promotion/Testnet/live authority. Completion
requires the scheduler and active refresh status to agree on generation 2, the same
activation ID, at least four refreshed cells, and exact reconstruction for every
captured Dynamic row. A forged activation or an unbound scheduler success claim
cannot trigger or complete the continuation.

### Explicit Dynamic-v2 Review Workflow

After the disjoint four-cell holdout passes, build and inspect the immutable review
packet without applying it:

```bash
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli review-wizard-dynamic-v2
```

The command must report `READY_REQUIRES_EXPLICIT_APPLY`. Review
`reports/active/wizard_dynamic_v2_review_packet.md` and use the exact
`dynamicv2review_...` identifier printed in that packet. Apply only that reviewed
packet with a named reviewer and a substantive note:

```bash
PYTHONPATH=src .venv312/bin/python -m quant_platform.cli review-wizard-dynamic-v2 \
  --apply-dynamic-v2 \
  --dynamic-v2-reviewer "REVIEWER_NAME" \
  --dynamic-v2-review-note "Reviewed all four disjoint holdout cells and immutable evidence bindings." \
  --dynamic-v2-review-packet-id "dynamicv2review_EXACT_PACKET_ID"
```

A successful apply reports `APPLIED_RESEARCH_COMPARATOR_ONLY` and performs only a
local recomputation of already captured Dynamic proof histories. It makes no Wizard
request and grants no candidate, Testnet-order, or live authority. A rejected apply
prints its blockers, performs no refresh, and exits with status 2 so automation
cannot mistake a blocked review for an applied activation.

## Stage 5 Eligibility-First Model Selection

Registered model selection uses only the isolated model-selection folds. A model is
selection-eligible only when the frozen `promising` checks pass, including positive
profit-factor, Sharpe, and expectancy deltas, non-worsening drawdown, and the minimum
10 percent take rate. Eligible models rank ahead of diagnostic-only models even when
a sparse model has a higher raw quality score.

When no model is eligible, the best diagnostic model may still be serialized for
research, but the manifest reports
`NO_ELIGIBLE_MODEL_DIAGNOSTIC_ONLY`. The downstream acceptance check requires
`selection_candidate_eligible=true`, so a diagnostic winner cannot be accepted,
quantized, promoted, or handed to Testnet. Untouched evaluation folds remain outside
model selection and are used only after the selection decision is frozen.

## Stage 5 Joint ML/RL Profitability Floor

Model-gated and reinforcement-learning acceptance use the same frozen minimum
10 percent participation floor. RL must also produce a strictly positive
net-after-cost total return in both validation and untouched held-out testing.
Each accepted Stage 4 survivor must independently meet the same positive-return
condition in both splits. Relative improvement over a worse losing baseline is
diagnostic evidence only and cannot pass registered learning or reach Testnet.

## 2026-08-10 Runtime Continuity Check

The autonomous Hyperliquid L2 LaunchAgent published
`reports/active/l2_capture_receipts/2026-08-10_215405.json` without a manual
capture override. All twelve assets used by the seven registered candidate pairs
had complete funding evidence and current source timestamps. The frozen rolling
gate remained correctly blocked at 11 of 12 strict observations, with roughly 115
minutes of coverage, because an earlier scheduling gap was still inside the
two-hour window. No early observation was inserted to manufacture a pass.

The Stage 2 capture, Stage 3 exact-mode proof launcher and scheduler, and Stage 4
registered-rerun handoff were re-verified together: 100 focused tests passed. The
corrective program checkpoint was then rebuilt at `2026-08-10T21:57:30Z` and
continued to report zero survivors, zero submitted orders, and false Testnet and
live authority. The next external Wizard proof attempt remains eligible only after
the `2026-08-11T00:00:00Z` credit reset.

A full unbounded pytest invocation was stopped after 89 passing tests because the
next active-pipeline cases read the production pair-detail corpus and can rewrite
operational reports. It is not recorded as a full-suite pass. The isolated
seven-stage control suite then passed 378 tests in 43.69 seconds. Its only warnings
identified the deprecated RL idea-scout UTC timestamp call; the implementation now
uses a timezone-aware `Timestamp.now`, and all 39 RL orchestration tests pass
without those warnings. Autonomous L2 receipts at `22:03Z` and `22:12Z` also prove
that the bounded audit did not starve market evidence collection.

## Active-Pipeline Test Isolation

The active-pipeline builders now derive every active, dashboard, ML-report, and
archive output from the caller-supplied repository root. Their tests use isolated
temporary repositories with explicit output-containment assertions, so a test run
cannot scan or rewrite the production research corpus. The complete
`tests/test_active_pipeline.py` module passes 48 tests in 19.12 seconds, and the
combined active-pipeline plus bounded seven-stage control set passes 426 tests in
56.71 seconds. No assertion was removed or weakened. The earlier unbounded run is
still not represented as a full-suite pass.

## Stage 2 Cadence Remediation

The receipt published at `2026-08-10T22:35:06Z` retained 11 strict observations
over roughly 116 minutes for every required asset. Its timestamp series showed
that the previous eight-minute LaunchAgent interval had insufficient operational
margin once process startup, public L2 collection, immutable-bundle publication,
and cost-stress calculation were included. Waiting under that configuration could
continue rolling an old sample out whenever a new sample arrived.

The read-only scheduler default and installed LaunchAgent interval are therefore
five minutes. The strict acceptance policy is unchanged: at least 12 independent
observations, at least 100 minutes of span, complete funding evidence, fresh fee
provenance, and pair-specific liquidity/slippage evidence are still required.
Twenty focused scheduler and corrective-program tests pass, lint is clean, and the
installed service reports a 300-second interval. This configuration change grants
no candidate, Testnet-order, or live-trading authority; Stage 2 advances only when
new immutable receipts satisfy the unchanged evidence gate.

### Stage 2 Passing Receipt

The next scheduled run published
`reports/active/l2_capture_receipts/2026-08-10_225208.json`. All twelve required
assets had 12 strict vendor-timestamp observations spanning approximately 111
minutes, with zero local-capture timestamp fallbacks. All seven selected pair-cost
models reported `STRICT_OBSERVED`, and the receipt reported both
`strict_pair_cost_acceptance_status=PASS` and
`stage_two_pair_cost_acceptance_status=PASS`.

The canonical corrective checkpoint was rebuilt at
`2026-08-10T22:54:43.224436Z`. It marks Stage 2 and the `venue_data_costs` phase
`PASS`, while recording seven post-window-ready pairs, zero corrective-program
orders, false Testnet authority, and false live authority. Rolling collection
continues for freshness; Stage 3 remains the next dependency and cannot make its
next bounded external proof attempt before the `2026-08-11T00:00:00Z` Wizard
credit reset.

## Stage 3-To-Stage 4 Active-Contract Continuation

The proof launcher now recognizes both forms of same-day registered-rerun work
after a verified Wizard proof attempt: rotating a conclusively accounted contract
into the next cohort, and executing an active cohort that the checkpoint has
already materialized. The latter case previously fell through the daily external
attempt guard and could wait until the next UTC day even though it required no
additional Wizard request.

Both continuation paths invoke the heavy scheduler with
`--internal-continuation-only`. The scheduler therefore requires a verified prior
daily attempt, makes no external proof call, and grants no candidate, Testnet-order,
or live authority. Completion must bind the scheduler receipt to the hashed,
immutable registered execution receipt and a conclusive Stage 4 outcome. If another
local contract generation is materialized, the next launcher cycle can continue it
under the same controls. Twenty-one launcher tests and the 94-test proof-scheduler,
registered-rerun, executor, and checkpoint set pass; lint and `git diff --check`
are clean.

The complete repository suite subsequently passed 1,214 tests in 609.17 seconds.
The run did not starve autonomous evidence collection: the L2 service published
another `PASS` receipt at `2026-08-10T23:16:50Z`, and the Wizard launcher continued
its bounded same-day deferral with an empty error log. The deprecated pandas UTC
call identified by the suite in the learning-outcome CLI was replaced with a
timezone-aware call; its six focused tests pass without that warning. Existing
small-sample NumPy correlation warnings remain diagnostic and do not alter gates.

## Supreme Team Canonical Authority

The generic Supreme Team previously promoted duplicated `CS*` rows from the older
Hyperliquid authority lane. That lane still reported `cost_evidence_not_ready`
after the canonical seven-stage checkpoint had proven Stage 2 `PASS`. The source
reports remain available as legacy diagnostics, but they cannot act as release
authority when a valid seven-stage checkpoint exists.

The Supreme Team now validates all seven unique stage rows and uses them as its
canonical worklist. Its `2026-08-10T23:34:55Z` checkpoint reports six open actions:
Stages 1 and 3 through 7. Stage 2 is correctly counted as passed, and the stale
legacy cost blocker is absent. A regression test proves that a legacy
`CS1 cost_evidence_not_ready` row cannot override a valid Stage 2 receipt. All 166
CLI tests pass, and no Testnet or live authority is granted by this reporting fix.

## Post-Reset Stage 3 Handoff

The frozen manifest `wizardcapture_617c2dc0bbfad54f8b87` contains 13 calls and
18 planned credits: one remaining exact-mode Copula backtest response, four
prospective OU-v3 holdout responses, and eight repeatable Copula-v2 behavioral
responses. Successful transport and reconciliation of those calls is necessary,
but it is not by itself a Stage 3 pass.

The post-reset sequence is deliberately split into evidence-preserving steps:

1. Execute only the calls bound to the immutable capture manifest.
2. Reconcile every request, response, credit, and lane against that manifest.
3. Evaluate the four OU-v3 holdout cells and the four Copula-v2 orientation cells.
4. Keep the already passing Dynamic-v2 review packet immutable until the frozen
   manifest is reconciled.
5. Apply Dynamic-v2 and OU-v3 only through their exact reviewed packet IDs. These
   are research-comparator activations, not candidate or order approvals.
6. Refresh the existing Dynamic and OU proof rows locally. The refresh consumes no
   additional Wizard credits and must bind its output to the immutable activation.
7. Rebuild parity and require 24 non-Copula formula proof cells plus four Copula
   behavioral cells before Stage 3 can complete.
8. Only then may the registered Stage 4 replay consume the completed Stage 3
   receipt.

Neither comparator activates automatically. A scheduler receipt that captured all
13 calls but lacks the reviewed activations and activation-bound local refreshes
remains incomplete. Every step retains `candidate_promotion_authority=false`,
`testnet_order_authority=false`, and `live_trading_authorized=false`.

## Immutable Capture-Source Lineage

The immutable call manifest alone previously froze requests, lane accounting, and
credit totals, while its source-artifact hash list remained only in the mutable
active pointer. Rebuilding that pointer could therefore change the aggregate
source fingerprint without changing the manifest ID. The calls stayed frozen,
but the provenance claim was weaker than the readiness report implied.

Every enforced capture manifest now has a deterministic immutable source receipt
under `data/research/wizard_capture_manifest_sources/`. The receipt binds the
manifest ID, immutable manifest path and hash, the canonical source-artifact
fingerprint, and point-in-time snapshots of every source artifact. Mutable queue,
audit, proof-ledger, budget, and comparator files may continue to evolve, but the
unresolved cohort always reuses its original source receipt and snapshots.

Reset readiness verifies the active pointer against that immutable receipt. The
proof scheduler independently repeats the same verification for both the candidate
and any carried-forward unresolved cohort before credit reservation or a vendor
call. Snapshot tampering, active-pointer source-list rewrites, missing receipts,
manifest-binding drift, and source-receipt identity drift all fail closed. The
existing OU-v4 cohort was migrated before its next eligible external execution;
its source receipt records `binding_origin=retrofit_before_external_execution`.
The migration changes no call, credit, request hash, candidate authority, or
trading authority.

### Pre-Call Manifest Enforcement

The scheduler treats the immutable manifest as an external-call allowlist, not
only as a post-call accounting report. Before any vendor runner receives
`execute=true`, the scheduler derives a separate call allowance for
`exact_mode_backtest`, `ou_v3_holdout`, `ou_v4_holdout`, and
`copula_behavioral`. A lane with zero manifested calls remains read-only. Exact
mode batches are additionally capped at the manifested call count before each
batch starts.

The shared credit ledger reserves `planned_credits` from the enforced manifest;
the broader proof-lane ceiling remains capacity information only. Post-call lane
and credit reconciliation still runs as defense in depth. Verified prior-day
reservation and reconciliation evidence clears stale blocker telemetry instead
of publishing contradictory `VERIFIED` and `BLOCKED` fields.

For the unresolved cohort `wizardcapture_df07a6d4defca7b866bd`, this permits only
eight `ou_v4_holdout` calls and 16 credits. Exact mode, OU-v3, and Copula runners
cannot make external calls during that capture. This restriction grants no
candidate, Testnet-order, or live-trading authority.

## Prospective Stage 5 Learning Protocol

Stage 5 model and reinforcement-learning choices are frozen before the future
Stage 4 survivor cohort exists. The canonical registration command is:

`PYTHONPATH=src .venv312/bin/python -m quant_platform.cli register-stage5-protocol`

The active pointer is written to
`reports/active/registered_stage5_protocol.json`, and the immutable receipt is
stored under `data/research/registered_stage5_protocols/<protocol_id>.json`.
The receipt binds the protocol configuration, survivor-support policy, twelve source
roles, Python and learning-library versions, and the model families available in
the registered runtime. It freezes chronological split rules, embargo, threshold
grid, minimum take rates, monotonicity and concentration gates, RL partitioning,
and the requirement that both model and RL acceptance pass.

The current prospective receipt is
`stage5protocol_729cf67b839aaf41eacb`. Its supervised concentration contract
requires complete, normalized taken-trade distributions for `pair`,
`timeframe`, and `regime`, with no contributor above `0.65`. Reconciliation
checks both recomputed predictions and the bound concentration artifact, so a
missing dimension, malformed share total, or concentrated selection cannot be
hidden by a passing aggregate return.

Registration is prospective and zero-authority. It does not create a candidate,
accept a model, permit a Testnet order, or authorize live trading. When Stage 4
eventually completes, the registered learning run rejects a protocol registered
after that completion timestamp, any changed bound source or policy, a mismatched
runtime, an invalid immutable receipt, or missing independent-survivor evidence.
Failed model or RL acceptance remains research-only and blocks quantization,
Testnet handoff, and live scoring.

## Post-Reset Automation Readiness Receipt

Before the `2026-08-12T00:00:00Z` Wizard credit reset, the project published a
point-in-time readiness receipt with the command:

`PYTHONPATH=src .venv312/bin/python -m quant_platform.cli build-wizard-reset-readiness`

The audit validates the installed and loaded LaunchAgent, its exact repository and
`--execute` command binding, the ten-minute polling interval, launcher heartbeat,
API-key configuration without exposing the secret, shared credit headroom, UTC
reset alignment, frozen-manifest accounting, and the immutable manifest digest.
It also reopens the immutable authenticated-browser receipt, verifies both raw
route-observation hashes, and requires its validity horizon to cover the exact
registered capture timestamp. A receipt that is fresh at audit time but expires
before reset therefore blocks before any Crypto Wizards request.
It also binds readiness to the scheduler's current manifest-continuity decision:
the active and candidate manifest identities, paths, and hashes must agree, both
candidate and source bindings must pass, and scheduler-detected cohort drift must
remain false.
Its active reports are `reports/active/wizard_reset_readiness.{json,md}` and
`reports/active/wizard_reset_readiness_checks.csv`; each run also writes an
immutable receipt under `data/research/wizard_reset_readiness/`.

The current readiness contract contains 21 fail-closed checks. A production
`PASS_RESET_AUTOMATION_READY` receipt proves only that the bounded research capture can
start safely after reset. Runtime credit preflight remains mandatory, and the
receipt grants no candidate promotion, Testnet-order, or live-trading authority.
The canonical Stage 3 checkpoint therefore remains `IN_PROGRESS` until response,
credit, and manifest reconciliation pass and the reviewed comparator and formula
parity requirements are satisfied.

Every newly finalized proof-scheduler cycle also requires a write-once,
content-addressed twin under `data/research/wizard_proof_scheduler_receipts/`.
The twin is published only after parity refresh, manifest and credit
reconciliation, checkpoint refresh, and any registered Stage 4/Stage 5 handoff
have finished. The launcher rejects a prospectively marked cycle when its twin
is missing or differs byte-for-byte from the operational cycle receipt. Legacy
pre-control receipts remain eligible only for the conservative same-day
duplicate-call guard; they cannot satisfy the new immutable-cycle requirement.
The canonical seven-stage checkpoint independently rescans those cycle receipts,
selects the newest execution cycle, and requires its byte-identical immutable twin
and complete Stage 3 evidence contract before it can publish `Stage 3 = PASS`.
Mutable scheduler status, launcher status, or a legacy execution receipt cannot
substitute for that final receipt. If all response, parity, credit, manifest, and
reconciliation evidence is otherwise complete but the twin is absent or invalid,
the checkpoint stops at `repair_final_immutable_scheduler_execution_receipt`.

The readiness contract also protects scheduler scratch storage. The daily,
five-minute L2, and ten-minute Wizard-proof LaunchAgents bind `TMPDIR`, `TMP`, and
`TEMP` to the private Expansion-workspace directory `.runtime_tmp`. Their plist
generators publish a canonical copy under `.runtime_agents` before mirroring the
system LaunchAgents directory. If the system volume cannot create an atomic
temporary file, the installer backs up the existing plist to Expansion and may
reuse only that file's already allocated block, followed by byte-for-byte and
plist validation. The readiness audit fails if the temp binding is changed, the
directory is missing or unwritable, or less than 1 GiB remains on that workspace
filesystem. This prevents a full system disk from silently disabling the next
evidence capture while preserving ordinary login persistence.

## Unified Live Scheduler Runtime Receipt

Publishing a plist does not prove that launchd loaded that exact definition. The
project therefore audits all three corrective services together with:

`PYTHONPATH=src python -m quant_platform.cli build-scheduler-runtime-readiness`

For the daily research, five-minute Hyperliquid L2, and ten-minute Wizard-proof
services, the audit verifies the workspace plist, system plist, byte-identical
SHA-256 mirror, executable, module and action, repository root, log paths,
`PYTHONPATH`, all three workspace-temp variables, schedule, loaded plist path,
and last exit state. A never-run daily service may report `(never exited)`; a
nonzero exit or any static-versus-live mismatch blocks readiness. The shared
runtime directory must also be private, writable, non-symlinked, and have at
least 1 GiB free.

Active evidence is written to
`reports/active/scheduler_runtime_readiness_checks.csv` and
`reports/active/scheduler_runtime_readiness.{json,md}`. Every run also freezes a
content-addressed receipt under `data/research/scheduler_runtime_readiness/`.
The canonical corrective plan rebuilds this receipt, surfaces it in Stages 1-3,
and blocks the Stage 3 external capture when the live Wizard service does not
match the reviewed contract. Low system-volume free space remains visible as an
operational warning because scheduler code, logs, and temporary writes are bound
to Expansion; it is not silently omitted or confused with trading authority.

This audit starts no service, consumes no Wizard credit, promotes no candidate,
and grants no Testnet-order or live-trading authority.

## Unified Comparator Human-Review Boundary

Dynamic-v2 and the newest registered OU generation share one fail-closed review
control generated by:

`PYTHONPATH=src python -m quant_platform.cli build-wizard-comparator-review-control`

The control validates each current review packet, immutable packet digest,
supersession gate, Supreme Team packet binding, activation state, and zero-authority
flags. It deliberately separates `review_ready` from `apply_window_open`: a valid
packet may be reviewed while a frozen capture is outstanding, but it cannot be
applied until that capture is reconciled. Applying a comparator still requires
the exact current packet ID, a named human reviewer, and a substantive review
note. No automatic apply path exists.

Dynamic-v2, OU-v4, and OU-v5 additionally enforce that boundary inside their activation
functions. An explicit apply must load a `PASS_ADVISORY_ONLY` Supreme Team receipt
that recommends activation for the exact current four-cell or eight-cell packet.
Each activation receipt stores the Supreme review ID, immutable path, and SHA-256
digest; every later activation load revalidates its content-addressed identity,
packet binding, complete cell semantics, and zero-authority flags. The lightweight
launcher and unified review queue independently enforce the same bindings. Calling
the CLI or Python activation functions directly therefore cannot bypass Supreme
Team review, and a rehashed but semantically altered review or activation is
rejected.

The active queue and status are written to
`reports/active/wizard_comparator_review_queue.csv` and
`reports/active/wizard_comparator_review_control.{json,md}`. Supreme Team output
is written to `reports/supreme_team/wizard_comparator_review_checkpoint.{json,md}`;
immutable receipts are stored under
`data/research/wizard_comparator_review_controls/`. The canonical corrective plan
refreshes this control automatically and surfaces its counts, receipt ID, blocker,
and next action in Stage 3.

The current production state is
`WAITING_FOR_FROZEN_CAPTURE_RECONCILIATION`: Dynamic-v2 is the one review-ready
comparator, OU-v5 awaits eight prospective holdout responses, no apply window is
open, and no comparator is applied. OU-v4 remains derivation-only and cannot block
the current generation after OU-v5 registration. This grants no candidate
promotion, Testnet-order, or live-trading authority.

## Stage 2 Mapping Freshness Maintenance

The five-minute L2 scheduler now maintains the exhaustive Wizard-to-Hyperliquid
Testnet mapping under the same read-only lock as cost collection. When the
exhaustive run is configured, it checks the source inventory timestamp on every
cycle. A mapping younger than six hours requires no network work. At six hours,
the scheduler refreshes the public Testnet market inventory and then republishes
the point-in-time mapping snapshot.

Mapping maintenance is isolated from strict pair-cost acceptance. A refresh
failure before the 24-hour source SLA is recorded as an operational warning while
valid cost evidence remains usable. A failure after 24 hours blocks the scheduler
receipt because the venue mapping is then materially stale. Each L2 receipt records
the mapping age, due state, refresh status, warnings, blocker, inventory-refresh
state, and mapping receipt ID. The Stage 2 checkpoint surfaces those fields.

The production lane was exercised against the public Testnet API: 210 inventory
rows, 157 tradable perpetuals, zero fetch-blocked rows, 129 mapped Wizard pair
groups, 26 execution-map-ready groups, and zero inventory or mapping drift. The
subsequent L2 receipt retained both strict cost gates at `PASS`, reported no
mapping warning or blocker, and retained false Testnet and live authority.

## Stage 4 Post-Reset Handoff Readiness

The registered Stage 3-to-Stage 4 route can be audited before any downstream
replay with:

`PYTHONPATH=src python -m quant_platform.cli build-stage4-handoff-readiness`

The same audit now runs automatically immediately after every registered-rerun
gate refresh inside the canonical corrective plan. This keeps the active receipt
bound to the exact contract, preflight, cost, manifest, and protocol state used
by the checkpoint; the standalone command remains available for focused audits.

The audit validates the immutable active contract and recomputes its material
identity, binds the frozen source family, validates the active candidate's
registration and strict-cost evidence, and proves the active and pending semantic
cohorts do not overlap. It also requires the pending representatives to cover the
current hypothesis batch exactly, including every collapsed venue experiment
variant. The remaining checks cover the pending family's history and cost
preflight, the three-pair and three-cluster breadth contract, the frozen Stage 3
manifest and reset automation binding, automatic registered executor wiring,
generation order, the prospective Stage 5 protocol, and the zero-authority
boundary. It does not run the replay or accept a strategy.

The reset binding requires the v3 readiness receipt, recomputes its content ID,
and requires a byte-equivalent immutable twin under
`data/research/wizard_reset_readiness/`. It also carries forward the scheduler's
validated manifest-continuity status. Editing and rehashing only the mutable
reset report therefore cannot release Stage 4.

The active reports are `reports/active/stage4_handoff_readiness.{json,md}` and
`reports/active/stage4_handoff_readiness_checks.csv`; immutable receipts are
stored under `data/research/stage4_handoff_readiness/`. The first production
receipt reported `PASS_STAGE4_HANDOFF_READY`; the current contract contains 13
checks after the cohort-partition and reset-receipt hardening. Each production
receipt records the bound active contract, pending family, frozen Stage 3
manifest, immutable reset receipt, and prospective Stage 5 protocol identities.

The v2 handoff receipt is itself content-addressed. The canonical seven-stage
checkpoint recomputes its receipt ID, requires the exact immutable twin, validates
check accounting and zero authority, and converts any mutable-only, mismatched,
rehashed, or malformed `PASS` claim into a blocked evidence state. The handoff
receipt remains an audit control; the registered executor independently enforces
its Stage 3 and immutable-contract gates.

The current handoff state is
`READY_AWAITING_STAGE3_THEN_ACTIVE_CONTRACT`: Stage 3 must first complete its
reviewed, reconciled parity chain; the active generation must then conclude
immutably; only afterward may the pending three-cluster family roll forward.
This readiness receipt grants no promotion, Testnet-order, or live authority.

## Stage 6 Candidate-Bound Collateral Readiness

Every canonical corrective-plan refresh now rebuilds the status-only collateral
preflight immediately after the release gate regenerates the active Testnet
candidate. The preflight binds the current candidate receipt ID and file hash, the
current approval ID and bounded one-use transfer amount, and a fresh public Testnet
margin snapshot. This prevents the seven-stage checkpoint from reporting a
collateral artifact that belongs to an older candidate generation.

The canonical path may read public Testnet account state, but it cannot access the
agent key, reserve or transfer collateral, submit an order, or grant collateral,
Testnet-order, or live authority. It rejects any preflight builder result that sets
one of those status-only fields true. A real transfer remains a separate explicit,
one-use action requiring a current immutable preflight, signed approval, environment
enable, exact acknowledgement, and execution request.

The production exercise bound candidate
`blockedtestnetcandidate_7a7388c39b2adbea3636` to the current candidate file hash
and a public margin snapshot captured at `2026-08-11T18:35:43.369611+00:00`.
The account still showed zero perpetual collateral and `995.075565` Testnet spot
USDC; the preflight remained blocked, accessed no key, attempted no transfer, sent
no order, and granted no authority.
