# Live Canary Input Parity Contract

## Purpose

Stage 7 may not rely on self-reported `schema_match`, `units_match`, or
`calculation_match` flags. Testnet/live parity is proven only by independently
validating hash-bound artifacts under
`thewiz.testnet_live_input_parity_evidence.v2`.

Passing this contract is a prerequisite only. It does not authorize an order,
enable the live executor, permit reuse, or grant persistent trading authority.

## Required Inputs

Exactly one Testnet snapshot and one live snapshot are required for each input:

1. `market_metadata`
2. `mark_and_mid_prices`
3. `funding_rate_and_timestamp`
4. `l2_depth_and_slippage`
5. `size_precision_and_minimum_notional`
6. `margin_and_liquidation_inputs`

All 12 snapshot paths must be present and unique.

## Snapshot Contract

Each snapshot must use `thewiz.live_input_parity_snapshot.v1` and contain:

- the exact input name
- `testnet` or `live` environment identity
- the current candidate receipt ID
- an embedded UTC capture timestamp
- the complete normalized numeric field set for that input
- the exact canonical units for every field
- a hash-bound local calculation artifact
- hash-bound raw Hyperliquid `/info` receipts for the required source requests
- `capture_complete=true`
- `private_key_accessed=false`
- `order_submission_performed=false`
- `live_trading_authorized=false`
- a valid receipt hash

The normalized fields and units are authoritative in
`PARITY_INPUT_CONTRACTS` in
`src/quant_platform/orchestration/corrective_live_canary.py`.

## Independent Validation

The control plane loads both snapshots and independently verifies:

- snapshot and outer evidence hashes
- exact field coverage and finite numeric values
- exact canonical units
- candidate and environment binding
- freshness and maximum Testnet/live timestamp skew
- equality between row timestamps and embedded timestamps
- use of the same unchanged calculation artifact in both environments
- exact read-only source request types for each input
- raw source receipt hashes, environment, `/info` endpoint, and capture timestamp
- absence of private-key access and order submission in every source receipt
- unique input rows and unique snapshot paths

Market values may differ between Testnet and live. Their schemas, units, and
calculation implementation may not differ.

The read-only collector is available as:

```bash
PYTHONPATH=src python -m quant_platform.cli capture-live-input-parity-evidence
```

It makes no request unless the sealed candidate, realized Stage 6 sample,
Supreme Team checkpoint, live-canary policy, and public account address all
pass first. The standalone command rebuilds the Stage 6 sample and Supreme
Team checkpoint before trusting their active files. It never reads a private
key and has no order endpoint.

The release-gate workflow invokes the collector automatically after Stage 6
passes. A current parity receipt is reused without another request. Once an
approval has an approval ID, wallet signature, or `approved=true`, its parity
receipt is preserved and cannot be replaced automatically. If that committed
receipt expires, the release gate blocks until an explicit approval reset;
it does not silently recapture or rewrite signed material.

## Fail-Closed Rules

Parity is blocked when any artifact is missing, stale, reused, malformed,
rehashed with a false claim, unit-drifted, calculation-drifted, or bound to a
different candidate. Legacy v1 self-attested evidence is not accepted.

Even after parity passes, Stage 7 remains blocked until the realized Testnet
sample policy, Supreme Team checkpoint, exact wallet-signed approval, one-use
authorization, and manual live executor controls all pass.

## Executor Implementation Binding

The live-canary executor contract is bound to one deterministic implementation
bundle, not only the order-submission module. The bundle contains the control
plane, read-only preflight, one-use submission/recovery executor, and outcome
validator. Its canonical path-to-SHA-256 map produces
`executor_implementation_bundle_sha256` and the content-addressed
`executor_contract_id`.

The bundle hash is carried through the read-only preflight, wallet approval,
immutable authorization, atomic reservation, execution receipt, executor
status, and outcome validation. A change to any bundled module rotates the
contract and invalidates stale approval or preflight evidence before key
access. Rehashing an old receipt does not restore validity.

The final submission boundary does not rely only on those sealed receipts. It
replays the canonical candidate validator against every queue-bound source,
checks candidate age against `maximum_input_age_seconds`, verifies the
registered policy receipt and pointer, re-hashes the realized Testnet and
Supreme Team evidence bound into the signed approval, and reruns all parity
snapshot, calculation, and raw-source checks. The executor refreshes the real
clock after acquiring its exclusive lock and rejects a caller-supplied clock
for any new execution. These checks occur before key access and atomic
reservation; recovery-only flattening remains available from the frozen
reservation after normal evidence expires.

Current receipt contracts are:

- `thewiz.hyperliquid_live_canary_executor_preflight.v4`
- `thewiz.live_canary_user_approval.v3`
- `thewiz.live_canary_authorization.v5`
- `thewiz.live_canary_reservation.v2`
- `thewiz.live_canary_execution_receipt.v2`
- `thewiz.live_canary_outcome.v4`

These bindings prove implementation identity only. They never grant order
authority, approval reuse, scaling, leverage, or another canary.
