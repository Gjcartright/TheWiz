# Effect authority and collection safety source review

**Decision:** keep the active checkout as the research-only source. Preserve the later LocalRuntime implementation as a candidate dependency closure. Do not cherry-pick its permit fields, collection guard, or release policy into the active tree. External effects, Testnet orders, and live trading remain blocked.

## Verified incompatibility

The LocalRuntime `corrective_order_authority.py` passes `authority_key_id` and `provider_id` to `EffectRequest`; those are the only two added lines in that file's runtime diff. In the active checkout, both `EffectPermit` and `EffectRequest` use Pydantic `extra="forbid"` and have neither field. A read-only model-field probe in the locked active environment confirmed this. Copying that order-authority diff alone would fail request construction. The later `effect_authority.py` changes the data model and adds a much larger durable-effect/journal surface, so a two-line port cannot provide its contract.

The runtime `corrective_effect_guard.py` adds admission checks before credential, network, and Keychain windows, plus a network-denial context manager. Its `collection_admission_barrier.py` is absent from the active checkout. That module explicitly has no activation API: when a dedicated collection root has been installed it denies credential/network effects, while a process with no installed collection root follows the legacy path. Its own docstring says it is not a boundary against hostile same-process Python. These source semantics need a process and owner-admission design review before integration.

The runtime acceptance manifest also adds `maximum_strict_l2_gap_minutes: 15`. Runtime data-evidence and policy-validation modules reference that field; the active source does not. Copying the manifest entry alone would not implement the later gap check or establish a current acceptance policy.

## Scope and next check

The dated AST comparison lists 176 changed Python paths, 123 with runtime-only top-level/class-member names, and 1,506 such names in total. Six active files changed after the earlier four-root hash freeze; none of the runtime files in this set did, and the CSV records current hashes on both sides. In this safety cohort, `effect_authority.py` has 95 runtime-only names and `corrective_external_effects.py` has 52. These are syntax counts, not proof of correctness or independent features. See `GATE0_RUNTIME_SYMBOL_TRIAGE_2026-09-30.csv` for exact current-file hashes and names.

To reconsider the candidate, freeze a dependency-complete runtime source/config/lock set; inspect permit signatures, durable journal rollback/identity behavior, owner admission, process boundary, and release-policy binding; then run its focused safety tests in a disposable locked environment. Compare the result against the active research-only path with explicit no-effect fixtures. Produce an immutable evidence receipt before any code selection or authority change. The 811-path custody queue still holds the unresolved source candidates.
