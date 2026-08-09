# The Wizard V2 Run Pipeline

## Purpose

V2 is a controlled restart inside the existing repository. Legacy code and evidence remain intact, but no V2 authority decision may combine mutable global reports directly.

The V2 contract is:

```text
current source evidence
-> atomic run snapshot
-> artifact hashes and identity validation
-> stage-specific semantic gates
-> sealed immutable run
-> status-only dashboard publication
-> completed pointer only when every research gate passes
```

## Commands

Create and seal a new preflight run from the declared source artifacts:

```bash
PYTHONPATH=src python -m quant_platform.cli build-v2-preflight-run
```

Validate an existing sealed run without reading global reports:

```bash
PYTHONPATH=src python -m quant_platform.cli validate-v2-run --run-id <run_id>
```

Republish a sealed run's status:

```bash
PYTHONPATH=src python -m quant_platform.cli publish-v2-run-status --run-id <run_id>
```

## Run Layout

Each run is written atomically to `runs/<run_id>/`:

```text
manifest.json
artifact_registry.csv
stage_status.csv
run_summary.md
seal.json
discovery/
wizard_capture/
vendor_proof/
hyperliquid_data/
costs/
local_replay/
walkforward/
council/
authority/authority.json
```

The directory is immutable after creation. `seal.json` records every file and content hash. Adding, deleting, or changing a run file invalidates the seal.

## Identity Contract

One run has exactly one:

- `run_id`
- `candidate_set_id`
- ordered pair set
- Wizard discovery `policy_hash`

Raw legacy artifacts may be bound through the run manifest envelope. Decision artifacts must embed the candidate-set identity. Wizard queue and proof artifacts must also embed the policy hash.

Cross-run evidence fails closed. An old walk-forward result, council decision, proof, or local comparison cannot vote on a current candidate set.
Pair-level artifacts must cover exactly the run's candidate pairs. Carrying the right candidate-set ID with missing or unrelated pairs also fails closed.

## Stage Gates

### Discovery

- Wizard sweep candidates exist and are non-empty.

### Wizard Capture

- pair settings are confirmed and complete
- queue candidate identity and policy match the run
- discovery gate passes
- paid-proof gate passes
- Wizard source is fresh

### Vendor Proof

- proof is completed
- proof uses scanner-horizon parity
- candidate identity and policy match

### Hyperliquid Data

- local pair history is ready
- both-leg funding is ready
- funding coverage is at least 95%

### Costs

- fee model is ready
- time-spaced size-aware slippage model is ready

### Local Replay

- observation count matches the Wizard horizon
- comparison is explicitly acceptance-valid

### Walk-Forward

- candidate identity matches
- selection controls pass when present

### Council

- council identity matches
- council status is `SHADOW_TEST`
- council does not abstain
- portfolio critic passes

### Authority

Authority passes only if every preceding stage passes. V2 preflight never grants paper, live, or execution authority.

## Publication Rules

Every attempt updates:

- `reports/active/v2_latest_attempt.json`
- `reports/dashboard/v2_run_status.csv`

A blocked or invalid run cannot update:

- `reports/active/v2_latest_completed_run.json`
- `reports/dashboard/v2_latest_completed_run.json`

The dashboard receives only a status row for blocked runs. It does not receive partial actionable candidate data.

## Operating Rule

Never repair or append to a sealed run. Correct the upstream evidence and create a new run. The previous run remains the immutable explanation of what was known and why the decision was blocked at that time.
