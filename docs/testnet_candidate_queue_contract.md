# Testnet Candidate Queue Contract

## Purpose

The Testnet candidate queue makes the prospective sample policy reachable without
weakening it. A single accepted pair cannot generate evidence for a policy that
requires three independent pairs.

## Upstream Requirements

- Stage 4 must produce at least three independent full-survivor clusters.
- Every queued experiment must be an accepted final 1x survivor.
- The registered Stage 5 ML and RL evidence must pass out of sample.
- The seven-day research cadence must pass.
- Both Hyperliquid markets and the pair-specific observed cost model must be current.
- The frozen sample policy determines the minimum independent-pair count.

## Selection Rule

Only rows without local or global blockers are `READY`. The active research
candidate is the ready row with the fewest validated exact-experiment lifecycle
samples for the current training-dataset and model-artifact lineage. Pair sample
count is the second ordering key and experiment ID is the deterministic tie
breaker. This prevents two accepted modes on one pair from being treated as the
same Testnet hypothesis.

The queue does not select a candidate unless the number of distinct ready pairs
meets the frozen sample policy. It selects one candidate at a time; after a
validated lifecycle is archived, rebuilding the queue can rotate selection toward
the next largest exact-experiment deficit.

The frozen production policy cannot be weakened below 30 closed paired
lifecycles over 14 observation days, three independent pairs, and three observed
regimes. A final sample pass also requires every accepted experiment to appear
and at least five independent lifecycles for each accepted experiment, each pair,
and each regime. Aggregate concentration checks therefore cannot be satisfied by
token one-observation coverage.

## Evidence And Tamper Controls

- The complete queue is stored as an immutable, content-addressed JSON receipt.
- The queue revalidates the registered Stage 5 receipt through the same full
  Stage 4 validator used by the learning orchestrator. Its bound Stage 4
  execution must exist at
  `data/research/registered_rerun_executions/{contract_id}.json`; relocating the
  execution receipt and recomputing downstream hashes cannot create a Testnet
  candidate.
- Every queue build regenerates `daily_cadence_acceptance.csv` from the immutable
  dated daily receipts. A hand-written or stale `PASS` row is never release
  evidence, and a rebuild, path, or column-contract failure blocks selection.
- Queue receipt and pointer v4 expose cadence validation status and the exact
  validation blocker. Earlier queue-receipt versions cannot validate as current.
- The active candidate identity binds the queue ID, path, file hash, rank, sample
  count, and required/ready pair counts.
- Candidate validation independently replays the queue ID, receipt hash, policy
  coverage, exact-experiment deficit ranking, selected row, model lineage, and
  authority flags.
- Before any new Testnet action, candidate validation re-hashes every current
  source artifact bound by the immutable queue. Missing or changed cost, market,
  cadence, Stage 4, or Stage 5 evidence invalidates the sealed candidate even if
  the queue has not been rebuilt.
- Lifecycle validation independently reconciles fill market, side, order ID, and
  aggregate quantity to the approved two-leg entry and opposite-side exit; event
  status labels alone are not evidence.
- Every lifecycle archive copies the immutable queue receipt and every source
  artifact named by that queue. Historical replay validates the frozen source
  bundle and its manifest hashes, so later active-source changes cannot erase
  valid history and archived-source tampering invalidates the lifecycle.
- The latest CSV and JSON pointer are operational views only; they are not the
  immutable authority artifact.

## Authority Boundary

Queue construction, candidate selection, and lifecycle archiving never submit an
order and never grant Testnet or live authority. Explicit, candidate-bound user
approval and all later execution gates remain separate requirements.
