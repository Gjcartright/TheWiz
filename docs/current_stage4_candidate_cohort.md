# Current-Family Stage 4 Candidate Cohort

## Purpose

Stage 4 requires at least three independent strategy clusters and at least three
distinct canonical asset pairs to pass every 1x research gate. Candidate
preparation must therefore make at least three distinct current-family pairs
testable without pretending that registration is acceptance.

## Cohort Construction

The active cohort is built from
`current_wizard_hyperliquid_failure_attribution.csv` only.

1. Retain current walk-forward near misses and their empirical equivalence
   clusters.
2. Read `minimum_independent_supporting_clusters` from the frozen acceptance
   policy.
3. Fill any pair deficit using the lowest `overall_research_rank` current-family
   rows that have point-in-time history and a completed canonical replay.
4. Keep one experiment per canonical asset pair.
5. Register every selected semantic hypothesis before its next cost/parity test.

Prospective additions have `acceptance_evidence_weight=0.0`. Their temporary
`prospective_pair_*` identifiers organize registration only. They do not count
as supporting or surviving clusters unless a later frozen rerun independently
passes walk-forward, multiplicity, regime, robustness, concentration, strict
cost, and exact-mode parity gates.

## Independence Contract

The final acceptance decision checks two dimensions separately:

- empirical equivalence clusters prevent duplicated strategy behavior from
  counting more than once;
- canonical asset pairs prevent multiple modes or orientations of one market
  pair from counting as independent breadth.

Both the statistical supporting set and the final 1x survivor set must meet the
frozen minimum in both dimensions. The final survivor receipt lists the exact
canonical pairs, and the registered executor verifies that this list equals the
accepted pair identities in the immutable contract conclusion. A claimed
three-cluster result from one pair fails closed.

## Family Boundary

Rows from `exhaustive_wizard_hyperliquid_canonical_replay.csv` are historical
diagnostics. They are written to
`corrective_l2_historical_diagnostics.csv` with
`collection_eligible=false`. They cannot satisfy current Stage 2 cost breadth,
enter the current registered rerun contract, or count toward Stage 4 breadth.

## Automatic Cost Evidence

The eight-minute read-only L2 scheduler checks every eligible current-family
candidate leg for complete funding evidence. If a leg is missing, it invokes
the bounded Hyperliquid public funding-history materializer for the entire
current cohort before rebuilding cost status. Completed funding is reused from
immutable current or exhaustive evidence; missing legs are fetched without a
wallet, signing key, or order endpoint. The scheduler remains blocked if the
funding materialization fails.

## Authority Boundary

Candidate registration grants no promotion, Testnet-order, or live-trading
authority. The active immutable rerun contract is not rewritten. A larger
contract generation can be issued only after the preceding generation has a
complete accepted/rejected conclusion receipt.
