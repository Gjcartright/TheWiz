# Math Desk source reconciliation

**Status:** research-only dependency audit. No Math Desk code was copied into the working checkout, no MATLAB Engine connection was claimed, and no trading or market archive state was changed.

## Source identity

- The working checkout has `src/quant_platform/statistics/math_v2.py` and related paper/math modules, but no `src/quant_platform/math_desk/` package.
- LocalRuntime and the recovery project each have **27 Python Math Desk modules**, byte-identical across those two copies in the 2026-09-29 hash manifest. That agreement makes them one corroborated source candidate, not an accepted canonical source.
- Their `statistics/math_v2.py` and `research_paper_reproduction.py` differ from the working checkout. The later `math_v2.py` adds tighter temporal/sample identity and Engle-Granger/ECM validation. Replacing the working file wholesale would need compatibility review.
- LocalRuntime `pyproject.toml` and `uv.lock` add an optional `math-research` dependency profile (`sympy`, `pyvinecopulib`, `arch`, and `cvxpy`) absent from the active project contract. The active locked environment cannot stand in for that candidate's dependency receipt.
- `audit/evidence_freeze_2026-09-29/source_reconciliation.csv` identifies each file and SHA-256. `missing_import_edges.csv` maps internal dependencies absent from the working checkout.

## Isolated diagnostic

An isolated copy of the working source plus the 27 Math Desk files and later `math_v2.py` failed test collection: 11 Math Desk test modules could not import. First-layer missing dependencies were `load_or_create_effect_authority_secret`, `ROLLING_SAMPLE_IMPLEMENTATION_VERSION`, `positive_log_price`, and `scripts.build_self_inspection_scorecard`. This proves the package is not a safe standalone copy into the current checkout.

A second isolated copy used the **coherent LocalRuntime source/config/tests/scripts**, a locked dependency environment, and copied available docs/reports and the 100-caption video-brain folder. The Math Desk selection ran **89 passing tests, 18 failures, and 67 setup errors**. Many errors require five governed files that were not found in the working, recovery, LocalRuntime, reconstruction, or older GitHub source trees examined:

1. `docs/math_desk_constitution.md`
2. `docs/math_desk_m3_core_contract.md`
3. `reports/plans/2026-08-22_math_desk_m0_m2_implementation_plan.md`
4. `reports/plans/2026-08-23_math_desk_10_of_10_corrective_plan.md`
5. `reports/plans/2026-08-23_math_desk_m3_core_implementation_plan.md`

The recovery and working Git histories examined did not show these paths. They may be in an unmounted historical workspace or genuinely missing; neither deletion nor successful prior certification is proven. The 18 failures need a clean rerun after source custody is resolved. The current test result is **inconclusive for the Math Desk algorithms** and confirms that the available copy is not fully reproducible as packaged.

## Test-by-test follow-up on 2026-09-30

The same isolated LocalRuntime source and 11 Math Desk test modules were rerun under `NO_EXTERNAL_NO_ORDER` into an off-drive JUnit file. All 38 Math Desk source/test files match the current LocalRuntime bytes. The result reproduced **89 passed, 18 failed, and 67 errors** across 174 cases. The 85 nonpasses break down as 80 cases blocked by the five missing governed files (including one subprocess whose exact command was rerun and raised that error), one missing `reports/active/wizard_research_journal.csv`, and four cases initially blocked by a missing lecture index in the isolated copy. The exact cases and JUnit hashes are in `audit/MATH_DESK_NONPASS_2026-09-30.csv` and `audit/MATH_DESK_DIAGNOSTIC_2026-09-30.json`.

The lecture index exists with identical SHA-256 in LocalRuntime and recovery. After copying its exact bytes into the disposable workspace, the focused risk-census selection had eight passes and **four setup errors** on `KeyError: 'review_status'`. The available index has six columns; the candidate code also requires `review_status`, `topic_tags`, and `transcript_text_stored`. This is a real source/data-schema mismatch, not merely an omitted test fixture. No review statuses or other values were invented. The 89 baseline passes are genuine focused passes, but the blocked cases do not establish an algorithm verdict.

## Required integration path

1. Resolve the five governed-source files through a verified checkpoint or mark them unavailable; do not fabricate replacements that masquerade as original evidence. Resolve the active journal evidence and the lecture-index schema from an authoritative source before risk-census validation.
2. Freeze one candidate Math Desk source hash set and compare its imports, tests, configuration, and `math_v2.py` behavior against the active checkout. Port dependency-complete units only.
3. Use a disposable locked environment to run the Math Desk suite, then add focused current-checkout regression tests for temporal/sample identity, orientation, ledger/cost accounting, and authority flags.
4. Verify MATLAB Engine using the selected Python runtime, if a MATLAB parity lane is still required; the installed R2026a application alone is insufficient.
5. Produce a new immutable Math Desk run receipt with source/config/data hashes and explicit research-only authority. Do not reuse August statuses as current results.
