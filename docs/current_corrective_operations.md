# Current Corrective Operations

## Crypto Wizards Browser Authentication Gate

Stage 3 requires a fresh immutable browser-auth readiness receipt whenever `config/wizard_browser_auth_contract.json` exists. Run the scanner and pair-detail browser helpers with explicit requested URLs, save both JSON artifacts under `data/raw/crypto_wizards/browser_auth/`, then build the receipt:

```bash
uv run --locked python scripts/build_wizard_browser_auth_readiness.py
```

The gate fails closed on public redirects, login or verification forms, missing account/member navigation, missing route controls, stale evidence, shared/tampered source files, or legacy self-asserted captures. Its immutable receipt publishes the earliest validity horizon across the two required routes. Reset readiness binds that receipt by ID, path, and SHA-256 and requires the horizon to cover the registered external-attempt timestamp; evidence that is fresh now but expires before the capture window is blocked before any vendor request. The gate grants research evidence only and cannot authorize promotion, Testnet orders, or live orders.

## Purpose

This is the operator entrypoint for the active Crypto Wizards to Hyperliquid research program. The project remains research-only until each authority gate passes in order. No file's existence grants trading authority.

On the exFAT Expansion checkout, set `UV_PROJECT_ENVIRONMENT` to a directory on the Mac internal drive, then run `uv sync --extra dev --locked` before using the commands below. The on-drive `.venv` is incomplete and must not be used as the command interpreter. Commands here describe operator procedures; they do not grant a provider call or order permission.

## Phase 00 Run-Control Status

As of 2026-08-22, the Phase 00 implementation has fail-closed controls for
publication, external effects, provider credits, credentials, orders, scheduler
provenance, terminal states, immutable identity, venue policy, typed lineage,
and CCXT evidence. All 74 active artifact candidates are sealed by immutable,
domain-validated, zero-authority envelopes. The current descendant control
preserves raw evidence, supersedes obsolete math outputs, and explicitly
invalidates stale fixture, strategy, dataset, model, and dashboard descendants,
including MATH-023 and MATH-024, pending controlled rebuild.

The canonical workspace and user LaunchAgent plist files remain unloaded. No
external request, credential access, Wizard or Apify credit use, Testnet action,
or order submission is part of Phase 00 verification. Runtime activation is a
later, separately approved operation; a passing implementation closure never
authorizes a reload, provider call, or order.

Current authority is read from these fail-closed control surfaces:

- `reports/active/phase00_control_checkpoint.json`
- `reports/active/phase00_active_artifact_lineage.json`
- `reports/active/phase00_descendant_control.json`
- `reports/active/phase00_closure.json`
- `reports/active/phase00_acceptance_matrix.csv`
- `reports/active/phase00_fault_catalog.csv`

The old 57-of-74 readiness report and its 17 blockers are historical diagnostic
evidence, not current Phase 00 authority. Reload controls P00-SL-020 and
P00-SL-021 remain deliberately blocked until separate operator approval and a
controlled first-run protocol. Hardware reboot/unmount recovery and elapsed
no-order soak tests also remain operational evidence gates rather than simulated
implementation passes.

Phase 00 maintenance and checkpoint commands are fail-closed and non-destructive:

```bash
uv run --locked python -m quant_platform.cli \
  phase00-maintenance-start --phase00-reason OPERATOR_REASON

uv run --locked python -m quant_platform.cli phase00-checkpoint

uv run --locked python -m quant_platform.cli phase00-lineage-seal

uv run --locked python -m quant_platform.cli phase00-checkpoint

uv run --locked python -m quant_platform.cli \
  phase00-descendant-invalidate

uv run --locked python -m quant_platform.cli phase00-checkpoint

uv run --locked python -m quant_platform.cli phase00-checkpoint

uv run --locked python -m quant_platform.cli phase00-closure-verify

uv run --locked python -m quant_platform.cli \
  phase00-maintenance-resume --phase00-maintenance-id MAINTENANCE_ID
```

Maintenance prevents governed producers from publishing, drains active leases
within a bounded timeout, and requires two stable manifests before resume. It
does not move or delete evidence. An expired or malformed maintenance marker
continues to block until an explicit, matching resume succeeds.

The repeated checkpoints are required dependency barriers, not redundant
commands. The first checkpoint gives the lineage sealer a stable source root.
Lineage publication changes a freshness-tracked pointer, so a new checkpoint is
required before descendant invalidation. During that one checkpoint only, an
otherwise fully valid zero-authority descendant control bound to the superseded
immutable lineage is reported as `PASS_HISTORICAL_ONLY`; normal observation
continues to block it. Descendant publication changes another
freshness-tracked pointer; closure therefore requires two later checkpoint
receipts with the same semantic freshness and the invalidated descendant status.
Skipping either barrier fails closed with a stale-checkpoint or
two-stable-checkpoints blocker.

Each checkpoint retains its complete byte-level manifest. It also records a
semantic freshness hash for safe comparison across controller processes. That
projection excludes only changing authority-journal bytes, lock-file bytes, and
raw process-probe return telemetry; it still binds source, tests, configuration,
evidence, active pointers, publication surfaces, financial effects, blockers,
producer presence, and launchd state. Two unchanged checkpoints must share that
semantic hash, while each checkpoint's internal first and second full manifests
must remain byte-identical.

All Phase 00 control commands skip `.env.local`. Closure tests run under a
sanitized environment with warnings promoted to errors, plugin autoload
disabled, external-network-marked tests excluded, and no API, token, secret, or
private-key environment variables. The immutable closure bundle contains exact
source, runtime, test, Git, acceptance, and fault-catalog receipts. A dirty-tree
receipt is hash-bound but is not reproducible from a clean Git clone; the receipt
states that limitation directly.

The first production closure attempt,
`phase00closure_c28351a03d07f4bf4978c4c0`, remains immutable failed evidence.
It correctly blocked on seven full-suite failures and an unavailable
`python -m pip check`. The failures exposed stale provider tests that did not
enter the required external-effect authority context and a full-suite CLI env
load that could leak process state between tests. The permanent corrections are:

- mocked provider calls must use the same authority and reservation sessions as
  production code;
- `NO_EXTERNAL_NO_ORDER` verification suppresses CLI env-file hydration for all
  commands, not only Phase 00 commands;
- the dependency integrity gate uses `uv pip check --python .venv/bin/python`
  when `uv` is available and fails closed if neither supported checker works;
- failed closure bundles have no implementation, runtime, model, or order
  authority and are superseded only by a new exact-tree passing bundle.

The later v2 closure attempt
`phase00closure_134c1b03661e7051c74d1fc1` also remains immutable failed
evidence. Its contained suite found two scheduler compatibility-lock failures
that appeared only under the macOS deny-network sandbox. When the OS process
start probe fell back to a monotonic token, re-probing the current PID produced
a different token and could misclassify its live lock as a reused process. The
lock validator now compares the current PID against the stable process and boot
identity captured by the running module. Other inaccessible live owners remain
unknown and non-evictable by age alone. The original two failures and an
unstable-reprobe regression pass both directly and under the exact closure
sandbox profile.

Closure schema v1 bundles predate read-only hardening. The v2 observer accepts
only an exact, hash-bound, path-bound, zero-authority v1 bundle as
`PASS_HISTORICAL_ONLY`; it never treats that evidence as a current-tree pass.
Unknown legacy schemas, path substitutions, hash mismatches, nonzero authority,
or changed active acceptance/fault mirrors remain blocking. This bounded bridge
exists only so a fresh v2 checkpoint and closure can supersede valid historical
evidence without a schema-upgrade deadlock.

Canonical runtime inspection commands are also research-only:

```bash
uv run --locked python -m quant_platform.cli build-canonical-runtime-contract
uv run --locked python -m quant_platform.cli build-scheduler-runtime-readiness
```

Do not reload the three LaunchAgents until an operator separately approves a
controlled first-run protocol. A plist existing on disk is not proof that it is
loaded, healthy, current, or authorized.

## Authority Hierarchy

1. `reports/active/seven_stage_goal_checkpoint.csv` is the canonical program status.
2. `reports/active/model_authority_status.json` governs ML and RL use.
3. `reports/active/hyperliquid_authority_state.csv` governs the current Hyperliquid run identity.
4. Immutable receipts under `data/research/` prove source, configuration, and result lineage.
5. `reports/active/current_state.csv` and dashboard reports are operator views regenerated from those authorities.

If two surfaces disagree, the stricter immutable authority wins. Historical V2, dYdX, Binance, GMX, Injective, and legacy dashboard reports do not override the current seven-stage checkpoint.

## Seven Stages

| Stage | Required proof | Current transition rule |
| --- | --- | --- |
| 1 | Seven distinct, semantically complete daily research receipts | Time-bound; never fabricate or backfill a calendar day |
| 2 | Hyperliquid two-leg history plus pair-specific fees, L2 slippage, funding, and liquidity | Maintain rolling public read-only collection |
| 3 | Crypto Wizards exact-mode provenance and formula/behavior parity | Use the frozen credit-bounded capture manifest; reconcile before comparator changes |
| 4 | Costed walk-forward, regime, robustness, equivalence, and independent breadth acceptance | Requires Stage 3 and an immutable registered rerun conclusion |
| 5 | Leakage-safe ML and RL incremental out-of-sample acceptance | Requires an accepted Stage 4 survivor set; research-only on failure |
| 6 | Reconciled Hyperliquid Testnet lifecycle and sufficient realized samples | Requires candidate authority, collateral, signed one-run approval, and all preflights |
| 7 | Minimal live canary | Requires every prior gate plus explicit, current, evidence-bound user approval |

Stage 6's real Hyperliquid SDK path requires the default canonical approval
validator and a durable execution-state journal. An OS advisory lock serializes
the full validate-to-submit transition across processes. The signed one-run
approval is revalidated with the latest journal immediately before submission
preparation and again after leverage configuration but before the bulk action.
Its v10 issue/expiry window may not exceed 15 minutes, its candidate receipt may
not be older than two hours, and its entry features may not be older than five
minutes. Every signed approval ID is bound to an immutable identity receipt;
changing or reusing that ID with another payload fails closed. Custom approval
validators remain available only to injected fake-exchange test harnesses;
they cannot be attached to the real SDK path.

The normal Testnet close does not reuse entry freshness as a reason to trap an
open position. After a journaled entry, the immutable signed approval retains
only its reduce-only exit policy: the same two markets, opposite sides, no more
than the recorded open size, a 50 bps maximum exit band, and IOC submission.
The original $25 cap remains an entry cap and is not reapplied after price
appreciation. A missing, changed, terminal, or already-used execution journal
blocks the normal exit; uncertain states use the recovery command, which never
retries entry.

The governed operator is two-step and status-first. Neither command grants
persistent authority:

```bash
uv run --locked python -m quant_platform.cli \
  hyperliquid-testnet-pair-execution-preflight \
  --testnet-pair-action entry --order-approval-id APPROVAL_ID

uv run --locked python -m quant_platform.cli \
  run-hyperliquid-testnet-pair-execution \
  --testnet-pair-action entry \
  --testnet-pair-preflight-id PREFLIGHT_ID \
  --order-approval-id APPROVAL_ID
```

The second command shown above is deliberately status-only. A real Testnet
action additionally requires the matching action acknowledgement, the explicit
execute flag, both Testnet environment enables, a fresh immutable preflight,
unchanged source hashes, and an unconsumed one-use reservation. Stage gates and
the signed approval must already be valid; these commands never submit live
orders.

The Testnet candidate queue binds the immutable L2 cost-bundle receipt and its
frozen `pair_cost_models.csv` snapshot, not the rolling mutable latest-cost
file. Cost freshness is checked separately when the queue is built and again
through the candidate/preflight age contracts. A normal five-minute L2 refresh
therefore cannot invalidate a sealed candidate mid-review, while any mutation
of the bound immutable bundle still fails closed.

The registered rerun gate, private replay-workspace builder, and release gate
all use the same complete cost-bundle validator. It re-hashes the frozen
candidate set, cost-status table, strict L2 window, fee profile, and pair-model
table; recomputes bundle, receipt, and per-pair model identities; requires exact
eligible-candidate/model coverage; reconciles funding readiness and minimum L2
sample/span claims; and recomputes the model's p95 slippage, spread, depth, fee,
execution-risk, and round-trip cost fields from the frozen inputs. Missing,
substituted, re-hashed-but-semantic-drifted, provisional, or authority-bearing
evidence blocks the Stage 2-to-Stage 4 handoff.

The frozen strict-L2 window may contain hash-bound observations for assets that
are not in the selected model cohort. Those additional observations are valid
collection evidence and do not affect a selected pair's per-asset statistics.
Validation therefore requires exact candidate-to-model identity and complete
L2 coverage for every modeled asset, then recomputes each modeled asset from
the frozen window; it does not require unrelated observed assets to be removed.

Stage 7 also requires the approval, preflight, authorization, reservation,
execution, and outcome receipts to share the current deterministic live-canary
implementation-bundle hash. Any change to the control plane, preflight,
submission/recovery executor, or outcome validator invalidates stale evidence
before key access.

Immediately before any new canary reservation, the executor also revalidates
the canonical candidate receipt and re-hashes every queue-bound Stage 4,
model, cost, market, and lifecycle source. It requires the candidate timestamp,
signed approval, registered policy receipt and pointer, realized Testnet sample,
Supreme Team checkpoint, and complete raw/snapshot/calculation parity lineage to
remain current and mutually bound. A caller-supplied execution clock is
forbidden, and the real clock is refreshed again inside the exclusive executor
lock. Any drift blocks before key access or reservation. Recovery-only mode
continues to use the frozen reservation so an already-open incident can still
be flattened after normal authority expires.

The control-plane builder applies the same fail-closed principle before it can
register a live policy or write `AUTHORIZED_FOR_ONE_LIVE_CANARY`. Caller
booleans are only corroborating assertions: the builder independently validates
the canonical queue-bound candidate and its age, every realized Testnet sample
row, and the blocker-free Supreme Team checkpoint. A claimed `PASS` cannot
create a misleading authorization artifact when those files disagree.

When any Stage 6 prerequisite is not ready, the control plane also revokes the
active executor-readiness snapshot. It writes a fresh `BLOCKED_UPSTREAM`
preflight bound to the current candidate, policy, and executor implementation,
with credential and network checks explicitly disabled. This prevents an old
mainnet account check from surviving an upstream regression. A later Stage 6
pass still requires a separately requested, fresh read-only preflight; the
control plane never reads the keychain or contacts mainnet automatically.

## Safe Status Refresh

These commands inspect or rebuild research status. They do not submit orders:

```bash
uv run --locked python -m quant_platform.cli system-check
uv run --locked python -m quant_platform.cli build-wizard-credit-budget
uv run --locked python -m quant_platform.orchestration.corrective_wizard_surface_inventory
uv run --locked python -m quant_platform.cli build-wizard-next-capture-manifest
uv run --locked python -m quant_platform.cli build-scheduler-runtime-readiness
uv run --locked python -m quant_platform.cli build-wizard-reset-readiness
uv run --locked python -m quant_platform.cli build-stage4-handoff-readiness
uv run --locked python -m quant_platform.cli build-corrective-agent-governance
uv run --locked python -m quant_platform.cli complete-corrective-plan
uv run --locked python -m quant_platform.cli build-artifact-index
uv run --locked python -m quant_platform.cli current-state
```

The Wizard surface-inventory command validates, rather than regenerates, the
governed inspection evidence. It requires exactly 14 documented API contracts,
29 dashboard areas, 28 explicit integration gaps, 21 capture opportunities, 12
browser/API parity surfaces, and 17 registered optimization dimensions. The
active status is bound to an immutable receipt and a 43-row API/dashboard
consumer matrix. Inventory completeness never grants promotion or execution
authority; the recorded integration gaps remain `IN_PROGRESS` until their own
point-in-time and parity gates pass. The legacy
`scripts/build_dashboard_full_inventory.py` entry point now delegates to this
validator so it cannot silently overwrite the newer inventory with its June
schema.

## Scheduled Producers

- `com.thewiz.corrective-research-daily`: one bounded daily evidence cycle.
- `com.thewiz.corrective-l2-cadence`: public read-only Hyperliquid depth and cost evidence.
- `com.thewiz.corrective-wizard-proof`: credit-bounded Wizard proof capture after the registered UTC reset.

Each L2 cadence run also refreshes the prospective evidence command center:

```bash
uv run --locked python -m quant_platform.cli \
  build-current-wizard-hyperliquid-evidence-command-center
```

Its pair and asset ledgers distinguish the immutable current cutoff from rolling
evidence intended for the next Wizard snapshot. Prospective collection rows have
no Stage 2, Testnet-order, promotion, or live-trading authority.

Every scheduled L2 process now has two separate transactions. The first writes
the immutable public-data capture receipt, an immutable cost-bundle pointer
snapshot at `data/research/l2_cost_model_receipts/<bundle_id>/pointer.json`, and
the mutable active routing pointer. Historical L2 validation binds the immutable
snapshot, so a later legitimate checkpoint rebuild cannot invalidate an older
receipt merely by rotating the active pointer. Only after that receipt exists
does the second transaction evaluate a local
L2-to-Stage-4 readiness refresh. While any eligible pair is still collecting,
it publishes `reports/active/corrective_l2_readiness_refresh_status.json` with
`WAITING_STRICT_L2` and calls neither the registered gate nor the Stage 4
builder. The status has an immutable twin under
`data/research/l2_readiness_refresh/`.

When every eligible pair is strict-ready, the refresher reopens the exact L2
receipt, verifies its candidate-set, cost-manifest, pointer, and model-snapshot
hashes, reruns the complete immutable cost-bundle validator, and acquires the
daily, L2, Wizard-proof, and registered-rerun locks in deterministic order. It
also requires the mutable routing pointer to still identify that exact source
bundle; a later rotation defers the handoff without damaging historical proof. It
then rebuilds the local registered-family gate and Stage 4 handoff, validates
the resulting handoff receipt, and rechecks all exogenous source hashes. An
active producer, candidate/source drift, invalid bundle, failed handoff
validation, or any nonzero authority stops the sequence and writes a blocked
or deferred receipt. This transition performs no Wizard request, candidate
refresh, registered rerun, Testnet action, or order submission.

Stage 1 counts only consecutive calendar dates whose selected receipt and
immutable archive copy match exactly. Each qualifying date must bind the exact
ordered 19-stage topology, all dated semantic receipts and artifact snapshots,
the 30-cell Wizard full sweep, and its 300-credit reservation and
reconciliation. Substituting a stage name, copying a receipt to another date,
or setting any promotion, Testnet, order-submission, or live authority field
revokes the date. The daily command-isolation audit blocks every registered
Stage 3 proof command plus direct `corrective_wizard_*` module or script
execution; the daily discovery lane cannot silently execute the parity lane.

OU-v5 persists an immutable per-call intent before any vendor request and an
immutable per-call completion receipt only after the response has landed. The
completion receipt binds the contract, intent path and hash, request path and
hash, response path and hash, call identity, UTC attempt, and two-credit cost.
Frozen capture reconciliation independently reopens and validates every OU-v5
completion receipt and commits its path and hash into the immutable
reconciliation outcome. A response with an intent but no valid completion is
`BLOCKED_EVIDENCE`, and mutating a completion after reconciliation revokes the
previous PASS during downstream validation.
The proof launcher and final immutable scheduler-execution audit also reopen
that reconciliation receipt and rerun the full validator before reporting
Stage 3 complete. Copied status, count, or PASS fields cannot authorize a Stage
4 transition when the immutable reconciliation is missing, stale, or mutated.
When an OU-v5 prospective registration exists, both the launcher and the
registered Stage 4 rerun gate independently validate the current registration,
the immutable eight-cell evaluation, raw request/response/intent/completion
lineage, activation state, and proof refresh. Omitting the current registration
or copying OU-v5 PASS fields from an unbound scheduler status fails closed.
If the
registered response is still missing, an intent from any UTC date blocks both
the capture runner and reset-readiness gate; crossing midnight does not permit
an automatic retry. The ambiguous attempt must be reconciled explicitly before
another external request can be authorized.

After capture, the OU-v5 v2 immutable evaluation commits the exact evaluation
CSV, request hashes, response hashes, per-call completion hashes, cell identities,
pass counts, comparator source, and result identity. Review packets, activation,
and the parity gate independently
revalidate those commitments. A locally edited status or detail table therefore
cannot become reviewed comparator evidence.

The registered Stage 4 executor also requires exact cohort identity coverage.
Every immutable semantic hypothesis must appear once in the ready gate, with the
same source experiment, pair key, pair, exact mode, and orientation. Duplicate
rows cannot replace a missing hypothesis while preserving the expected row count.

All three must fail closed. Scheduler receipts must show zero order authority and zero submitted orders.

The daily research producer is exactly-once at the qualifying calendar-day
boundary. Before invoking any external stage, it validates the selected receipt
for the current UTC date, including all semantic, raw-response, credit, and
zero-authority bindings. A valid existing `PASS` is reused and the external
pipeline is not invoked again. Receipt selection is serialized with an
OS-backed per-day lock, so scheduler, import, or manual publisher processes
cannot race to replace the first valid pass. Every distinct attempt is still
archived immutably, invalid dates are rejected before publication, and a failed
or invalid selected attempt may be replaced by a later valid pass.

Before the daily producer reaches the Wizard discovery stage, its parent
process resolves `CRYPTO_WIZARDS_API_KEY` for the child environment. A process
environment value takes precedence; otherwise only an owner-matching,
non-symlink `.env.local` or `.env` with no group or other permission bits may
supply it. Missing or insecure credential evidence blocks before the Wizard
network call. Reports persist only credential status, source filename, and any
insecure filename; they never persist the value. This preflight supplements the
CLI's normal `.env.local` loading and keeps the launch-agent plist secret-free.

Crypto Wizards credit authority is also fail-closed across processes. The
discovery lane and proof lane reserve under one UTC-day ledger lock. Only a
newly published reservation authorizes chargeable requests. A reused receipt
authorizes another attempt only when immutable reconciliation proves that every
prior cycle attempted exactly zero credits; an unreconciled reservation or any
positive attempted spend has zero replay authority. This prevents a process
crash between vendor calls and reconciliation from silently repeating spend.
All direct chargeable Wizard CLI routes are plan/preflight-only; external
exact-mode, OU-v3/v4/v5, and copula captures must run through the installed
shared-credit proof scheduler.

The L2 producer may overlap a status rebuild that intentionally holds
`.corrective_l2_capture.lock`. That overlap must write a complete blocked cycle
receipt, make no collector call, leave the existing lock owner untouched, and
exit without an unhandled exception. The next five-minute public read-only cycle
must restore a current `PASS` receipt before scheduler runtime readiness returns
to 61/61.

Pair-cost helper schema drift must also produce a complete `BLOCKED` cycle
receipt instead of raising during receipt construction. Runtime readiness keeps
append-only scheduler stderr for incident evidence and reports its byte count
and timestamp. Nonempty stderr is classified `HISTORICAL_RECOVERED` only when
launchd's current exit is safe and a newer stdout write exists; otherwise it is
flagged for operator review. Log content is not copied into readiness receipts.

## Frozen OU-v4 and OU-v5 Reset Sequence

1. Before `next_external_attempt_eligible_at`, the Wizard producer must report
   `DEFERRED_CAPTURE_MANIFEST_WINDOW` and make zero calls. The direct
   `run-wizard-ou-v4-holdout --execute-wizard-proof` command may inspect the
   same immutable-manifest and eligibility state but always returns
   `BLOCKED_SHARED_CREDIT_ORCHESTRATION` with zero calls; only the governed
   proof scheduler may execute the cohort.
2. The first eligible cycle may execute only the eight calls registered in
   `wizardcapture_df07a6d4defca7b866bd`, with an attempted-credit ceiling of 16.
   Pre-spend validation binds every call ID, pair, mode, orientation,
   observation count, request path and hash, response path, and credit cost to
   the preregistered OU-v4 contract; matching only the lane count is not enough.
   Immutable manifests and their frozen source snapshots use the same durable,
   atomic, no-overwrite publication rule as vendor responses. Daily credit
   reservations and reconciliations follow that rule as well, so a partial
   ledger receipt cannot be mistaken for valid accounting.
   The active manifest also reclassifies every current source against those
   snapshots on each rebuild. `EXACT_MATCH` and a verified
   `GENERATED_METADATA_ONLY` change to top-level `generated_at_utc` remain
   eligible; any queue, configuration, missing-file, non-JSON, or structural
   change becomes `SCIENTIFIC_OR_STRUCTURAL_DRIFT` and blocks before an
   external request. Reset readiness independently recomputes this result from
   the immutable receipt instead of trusting the active manifest declaration.
   Each response is durably written to a temporary file and atomically linked
   into its immutable final path, so an interrupted write cannot masquerade as
   a captured cell. A durable per-call intent is published before each vendor
   request. If a process dies after that intent but before response persistence,
   the missing response is treated as ambiguous and cannot be retried that UTC
   day. After a partial vendor failure, the next eligible day reuses intact
   responses and requests only the still-missing cells. Final reconciliation
   binds every OU-v4 response to all of its retained intent paths, content
   hashes, UTC attempt dates, and multi-day recovery count; missing, altered,
   deleted, or internally inconsistent intent evidence fails closed.
3. That cycle persists the raw responses, evaluates the prospective OU-v4
   holdout, creates advisory review artifacts, and reconciles all eight manifest
   calls into an immutable receipt. It cannot activate a comparator or advance
   Stage 4 while reconciliation is pending or invalid. A scientifically failed
   holdout is a valid operational outcome when the frozen 8/8 reconciliation is
   complete: the launcher exits cleanly, keeps the comparator inactive, and
   records the research failure as the Stage 3 blocker. Malformed, missing, or
   unreconciled evidence still exits nonzero.
4. The next internal-only cycle spends zero Wizard credits, revalidates the
   immutable 8/8 reconciliation, and opens the explicit research-comparator
   review window only when the holdout and Supreme Team review pass.
   A newly built capture manifest validates only contract cells whose responses
   are still pending. Validation of the immutable historical cohort continues
   to require all eight exact preregistered cells. This prevents a completed
   cohort from being misclassified as an empty-call contract mismatch without
   weakening the original frozen-cohort proof.
5. Rebuild `build-wizard-comparator-review-control` and inspect
   `reports/active/wizard_comparator_review_control.md`. The generated preflight
   commands bind to the current immutable Dynamic-v2 and current OU packet IDs. An
   apply template is emitted only when the frozen-capture mutation blocker is
   gone and that comparator's evidence and Supreme Team bindings pass.
6. Dynamic-v2 and the current OU generation each require the current review-packet ID,
   a named reviewer, and a separate substantive review note. Do not hand-copy a
   packet ID from an older receipt. Application remains research-only, refreshes
   only the activation-bound proof rows, and grants no candidate, Testnet, or
   live authority.
7. Only the resulting 28/28 accepted Stage 3 proof can unlock the already
   registered Stage 4 costed walk-forward contract. Stage 5 and all order paths
   remain downstream evidence gates.

The failed OU-v4 cohort is derivation-only. It may be used to attribute selector
errors and preregister a new hypothesis, but it must never be relabeled as a
holdout or reused to validate that hypothesis. Any OU-v5 check must use a
disjoint, frozen cohort with a new implementation hash and a new immutable
capture manifest.

OU-v5 is now that registered successor cohort. Its eight calls cover two new
pair groups, four assets not used by prior OU holdouts, both OU display modes,
and both orientations. The frozen hypothesis separates three decisions that
OU-v4 incorrectly coupled: price transform, the Engle-Granger trend diagnostic,
and the OU profile-kernel branch. Crypto Wizards documents `inc_trend` as a
cointegration diagnostic; it is not treated as a documented OU formula-branch
switch.

The OU-v5 operating sequence is:

1. Rebuild `build-wizard-credit-budget`; the scheduled ceiling is 408 credits,
   including a separate 16-credit OU-v5 ceiling.
2. Rebuild `build-wizard-next-capture-manifest`. The pending manifest must bind
   exactly eight OU-v5 calls and 16 credits while no v5 responses exist. A
   completed cohort produces zero pending v5 calls but does not weaken
   validation of the original immutable eight-call manifest.
3. Wait for `next_external_attempt_eligible_at`. The scheduler fails closed
   before that time. The direct `run-wizard-ou-v5-holdout
   --execute-wizard-proof` command remains zero-call and blocked even after
   eligibility because it cannot bypass shared credit orchestration. After a
   same-day scheduler attempt, the lightweight launcher must
   not invoke local reconciliation while the frozen capture still has pending
   or blocked calls; local-only continuation becomes eligible after response
   accounting reaches zero pending and zero blocked calls. Active launcher
   status is serialized and ordered by `completed_at_utc`. A genuinely earlier
   completed result cannot overwrite newer state, while an older-started heavy
   process that finishes later with newly reconciled evidence remains eligible
   to publish that evidence. Every launcher decision is also written once under
   `data/research/wizard_proof_launcher_receipts/`; the mutable active status
   binds to that receipt by ID, path, and SHA-256, and reset readiness rejects a
   missing, rewritten, path-shifted, or authority-bearing receipt. A manual
   non-execute inspection still receives an immutable receipt, but it cannot
   replace an existing execute-mode operational heartbeat. This prevents a
   read-only check from disabling reset automation while stale execute-mode
   heartbeats continue to fail readiness normally.
   Non-execute scheduler receipts report `api_key_check_performed=false` and
   `api_key_source=not_checked_non_execute`; they do not load secret files.
   Execute mode performs the credential check with process environment first,
   then `.env.local`, then `.env`. Secret files must have mode `0600`; an
   insecure file is rejected without loading its value.
4. At the eligible reset, each request receives a durable pre-call intent and
   each landed response receives a durable post-call completion receipt.
   Reconciliation requires all eight completions to bind to their request,
   response hash, contract, intent path, intent hash, UTC attempt date, and
   credit cost. A copied response, an intent-only crash orphan, or a response
   without completion evidence cannot enter evaluation.
5. The scheduler may capture and evaluate v5, build its immutable review packet,
   and run the advisory Supreme Team review, but it always invokes activation as
   preflight-only. It cannot activate a
   comparator, promote a candidate, submit a Testnet order, or authorize live
   trading. A failed v5 result is recorded as a scientifically negative,
   operationally reconciled outcome. The scheduler then writes a separate
   immutable failure-attribution receipt. That receipt reconciles the active
   evaluation against the immutable cell counts and response hashes, preserves
   passing orientations, separates transform, trend, profile-branch, and formula
   failures, and runs the four Supreme Team lenses. It grants no activation or
   trading authority.
6. Only a prospective 8/8 v5 pass across formula, transform, Engle-Granger trend
   diagnostic, and OU profile-branch checks can replace the v4 failure as current
   OU formula evidence.
7. After that pass, run `build-wizard-ou-v5-supreme-review`, inspect the exact
   immutable packet and advisory receipt, then use `review-wizard-ou-v5` with the
   exact packet ID, a named reviewer, a substantive note, and `--apply-ou-v5`.
   No scheduler path can supply those human fields or apply automatically.
8. A valid research-only activation selects comparator generation 5 and refreshes
   only existing captured OU proof rows. The refresh binds every row to the
   activation ID and digest, verifies raw vendor hashes are unchanged, and must
   reconstruct all eight current OU rows before Stage 3 can pass.
9. If v5 fails, its observations are consumed and cannot validate a successor.
   Automatic successor design and registration remain disabled. A new OU cohort
   requires a Supreme Team review of the immutable attribution followed by a new
   hypothesis, implementation hash, frozen predictions, disjoint assets and time
   window, zero vendor responses at registration, and a new credit-bounded
   capture manifest.

After evaluation, the canonical checkpoint uses a deterministic OU-v5 outcome
router. A failed result routes to failure attribution and then Supreme Team
review. A passing result routes in order through immutable-evidence repair,
supersession/review-packet repair, Supreme Team repair, explicit human research
activation, and activation-bound proof refresh. It cannot fall back to a generic
"collect more evidence" instruction once a terminal v5 result exists.

## Registered Stage 4 Continuation

Stage 4 may require more than one immutable registered generation. The active
contract runs first. If its conclusion is fully accounted but a disjoint,
preflight-passed family remains, the next launcher cycle rotates to a successor
contract and then executes it. These continuation cycles are local-only: they
must use `--internal-continuation-only`, make no Crypto Wizards requests, and
cannot bypass the frozen source-family, policy, cost, or history bindings.

The launcher may continue through additional registered generations until the
current family reaches one of two terminal outcomes:

- accepted survivors satisfy the frozen independent-pair and independent-cluster
  breadth policy; or
- every registered hypothesis in the current family is conclusively rejected.

A partial generation, unrotated contract, missing conclusion, or one-pair
acceptance claim remains `IN_PROGRESS` or fails closed. Stage 5 cannot start
until the canonical Stage 4 checkpoint proves the whole-cohort terminal state.

## Registered Stage 5 Protocol

The supervised-model and RL protocol is frozen prospectively under
`data/research/registered_stage5_protocols/` before Stage 4 produces an
accepted cohort. The active pointer alone is not sufficient evidence. Stage 4
handoff readiness invokes the canonical Stage 5 validator and rechecks the
configuration, support policy, runtime/model availability, and all bound source
hashes. Any implementation or policy drift blocks the handoff until a new
prospective protocol is registered. This check grants no model, promotion,
testnet, order, or live-trading authority.

The registered supervised protocol is executable evidence, not descriptive
metadata. Training receives the frozen split count, minimum training rows, and
embargo explicitly. Acceptance then reconstructs every expected test-row
assignment from the immutable active dataset and requires exact `trade_id`,
fold, phase, and protocol-parameter coverage. A missing adverse row, an added
row, or a moved row fails even when every downstream summary is recomputed to
match the altered prediction artifact.

Protocol `stage5protocol_729cf67b839aaf41eacb` also freezes supervised
taken-trade concentration across `pair`, `timeframe`, and `regime`. For each
dimension, the evidence shares must be finite, non-negative, sum to one, and
have no contributor above `0.65`. The check is run independently against the
recomputed prediction evidence and the hash-bound concentration artifact.
Missing, reordered, expanded, malformed, or overconcentrated dimensions fail
Stage 5; gain concentration and pair-only concentration remain additional
controls rather than substitutes.

The RL lane follows the same rule. Its registered train and validation
fractions, minimum split rows, and entry-threshold calibration quantile are
passed into the runner and bound into split, execution, and lineage artifacts.
Reconciliation rebuilds the partitions from the active dataset and rejects
parameter drift. The active prospective protocol ID and immutable receipt
binding are published in `reports/active/registered_stage5_protocol.json`. The
ID intentionally changes whenever a bound implementation or policy changes.
The active protocol remains research-only and can be used only after a future
accepted Stage 4 execution that completes after that protocol was registered.

The Stage 4 handoff v4 receipt binds every decision-bearing Stage 3-to-Stage 5
input: the active and immutable registered contract, frozen source family,
registered gate, pending-family preflight and batch, active and immutable
capture manifest, immutable reset receipt, prospective Stage 5 protocol, and
proof-scheduler status. Each fixed-path artifact is sealed by SHA-256 in a
single source-closure identity. A changed gate state, orientation, contract,
review input, or scheduler result invalidates the old handoff receipt instead
of leaving a stale PASS in the program checkpoint.

The receipt also binds the Stage 3 reset audit through a
stable safety-state hash. Audit timestamps, heartbeat age, and receipt IDs may
change during an equivalent observation refresh without invalidating the
handoff. A changed manifest, scheduler state, credential state, source drift,
blocker, authority flag, or other safety-relevant reset field changes that hash
and fails the handoff validator immediately. The builder rechecks the active
reset state just before publication, and every canonical consumer repeats the
current-state comparison, closing both overlapping-build and later-regression
windows.

## Cost Evidence Binding

`reports/active/hyperliquid_pair_cost_bundle_pointer.json` is the current
handoff from the rolling public L2 cost builder to the registered Stage 4
preflight. It binds the active pair-cost table to one immutable bundle under
`data/research/l2_cost_model_receipts/` and records hashes for the bundle
receipt, immutable model, and active model.

Stage 4 recomputes the bundle and receipt identities, requires the active and
immutable model hashes to match, and verifies that every authority flag remains
false. A missing pointer may use the older scheduler-bound receipt for legacy
evidence; a present but stale, malformed, drifted, or rehashed pointer fails
closed and cannot fall back to older evidence.

The registered rerun also audits funding refreshes per invocation. Its active
status distinguishes a refresh request from reaching the cost stage and from
evidenced Hyperliquid API fetches. A blocked, planned, or already-complete call
cannot claim a refresh merely because `--fetch-funding` was supplied. A
successful immutable receipt records fetched and cache-reused asset counts and
binds the same facts to the cost-stage row. This is research-data authority
only and never grants order, promotion, testnet, or live-trading authority.

## Stage 6 No-Order Protocol

The immutable Testnet candidate queue is the authority for the Stage 5 handoff.
It binds the registered learning ID, immutable learning receipt path and hash,
registered Stage 5 protocol ID and hash, and registered Stage 4 execution ID.
The selected candidate must repeat those values exactly. Rebuilding and
resealing a candidate around substituted learning lineage fails queue
validation even when the candidate's own identity hash is internally valid.
The queue also proves that the learning receipt path/hash is present in its
source-artifact hash set.

That same registered Stage 5 identity is part of the signed one-run approval,
the `hyperliquid-testnet-lifecycle-v3` exchange receipt, the immutable lifecycle
archive, and the validated lifecycle index. Sample sufficiency filters on the
dataset, model artifact, registered learning receipt, Stage 5 protocol, and
registered Stage 4 execution as one cohort. Evidence from an obsolete learning
run cannot be pooled with the active run merely because both runs produced the
same model hash. Approval schema `hyperliquid-testnet-smoke-v10` invalidates all
pre-lineage approvals and requires a fresh explicit approval for the exact
candidate and learning chain.

Every no-order Testnet preflight rebuilds the deterministic pair-lifecycle
protocol from current code and inputs. It does not trust a prior mutable
protocol manifest. The preflight recomputes the protocol identity, requires all
recovery scenarios, checks current source bindings, and verifies zero execution
and live authority. A rebuild error becomes a blocked preflight check; it does
not fall back to stale protocol evidence or submit an order.

## Stage 7 Live-Canary Handoff

Stage 7 cannot trust a mutable sample CSV or a self-asserted Supreme Team JSON.
After Stage 6 passes, `build_stage6_release_receipt` writes an immutable
`thewiz.stage6_release_receipt.v1` artifact and an active pointer. The receipt
binds the exact candidate and queue, complete registered Stage 5 lineage,
frozen Testnet sample policy, validated lifecycle index, every archived
lifecycle receipt hash, sample-sufficiency CSV, and Supreme Team checkpoint.

The live approval schema is `thewiz.live_canary_user_approval.v3`; authorization
is `thewiz.live_canary_authorization.v5`. Both bind the immutable Stage 6 receipt
ID, path, and file hash. Immediately before any key access, the manual executor
revalidates the candidate, archive closure, sample reconstruction, frozen
policy, Supreme Team checkpoint, current parity evidence, implementation
bundle, and immutable Stage 6 receipt. A forged all-PASS CSV, a substituted
learning run, or any post-authorization source change fails closed before
reservation or signing.

## Stop Conditions

Stop progression and keep `RESEARCH_ONLY` when any of these is true:

- evidence identity or hash does not match;
- data is stale or incomplete;
- Wizard exact-mode parity is unproven;
- costed statistical breadth is below policy;
- ML or RL fails out of sample, monotonicity, take-rate, or concentration gates;
- candidate, wallet, collateral, funding, paired sizing, or reconciliation proof is missing;
- explicit one-run authority is absent, stale, or bound to different evidence.

## Recovery

The canonical source checkout is `/Volumes/Expansion/Crypto Wizard` on the mounted Expansion drive. The loaded `com.thewiz.nightly-savepoint` LaunchAgent runs at midnight, copying a dated project and LocalRuntime snapshot to Expansion and the Mac internal drive and pushing a source-only mirror to the private GitHub backup repository. The installed script matches `scripts/ops/nightly_savepoint.py` byte for byte. Check the three destination results and manifest hash with:

```bash
python scripts/ops/nightly_savepoint.py --status
```

The dated receipt is stored under `/Users/gregc/Backups/TheWiz/state/`. A current source commit also needs its own complete Git bundle and independent restore proof under `/Users/gregc/Backups/TheWiz/gate0-checkpoints/<commit>/`; the latest head-tied Gate 0 source receipt names that bundle, its SHA-256, the clean restore, and the corresponding GitHub CI runs. The nightly snapshot and exact-commit bundle cover different points in time.

For an additional manual redacted package from a clean reviewed checkout, use the locked Mac internal Python environment:

```bash
'/Users/gregc/Library/Application Support/TheWizRuntime/venv/bin/python' scripts/build_corrective_checkpoint.py
'/Users/gregc/Library/Application Support/TheWizRuntime/venv/bin/python' scripts/build_current_recovery_checkpoint.py --destination /Users/gregc/Backups/TheWiz/manual-recovery
```

This package includes committed source through a Git bundle and snapshots eligible changed source when present. Its restore drill checks the exact Git head, source hashes, compilation, and selected smoke tests. A disposable clean-checkout drill on 2026-09-30 passed 15 smoke tests with zero changed-source rows. The older encrypted-workspace evidence variants remain preserved for separate source review; they are not part of the scheduled midnight save point.
