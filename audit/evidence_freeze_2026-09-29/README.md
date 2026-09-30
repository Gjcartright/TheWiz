# Forensic evidence freeze and first red-team probes

This is a **dated research-only snapshot** created from 2026-09-29 21:04–21:08 America/New_York (UTC times in `summary.json`). It does not certify a strategy, approve a research run, or grant Testnet/live authority. Files were hashed sequentially, with size and modification time compared before and after each read. The cross-root set is not an atomic filesystem snapshot; the forum task can add material later. `post_freeze_delta.csv` records the immediate catalog and policy-receipt changes made after the baseline hash pass.

The committed copies of `source_reconciliation.csv` and `missing_import_edges.csv` use LF line endings for Git review. Their field values and row order are unchanged; the original CRLF files remain in the dated private savepoint.

## Files

- `file_manifest.csv`: 1,929 working-checkout files, 2,861 recovery-project files, 8,040 LocalRuntime files, and 59 Wizard Papers files, each with relative path, size, modification time, SHA-256, and stability status. Environments, Git objects, Finder sidecars, caches, build outputs, symlinks, and secret-like names were excluded. The read-only LocalRuntime capture archive was not edited.
- `git_state.json`: working branch, HEAD, remote refs, porcelain status, and staged/unstaged diff checks. The pre-existing staged work remains intact.
- `evidence_register.csv`: 76 rows: 16 cross-cutting components/inventories plus all 49 historical forensic findings and 11 historical corrective actions. Each historical finding/action is explicitly `NOT_REVALIDATED`; the row records whether its named evidence path exists in each inspected tree, not whether that evidence passes now. No owner or supersession link was invented.
- `source_reconciliation.csv`: 1,096 relative paths in source/config/docs/scripts/tests/apps across working, recovery, and runtime trees. Using this scope: 378 working/runtime hashes match, 191 differ, 526 are runtime-only, and one is working-only. These counts intentionally differ from the handoff's broader eligible-text scope (177 changed, 702 runtime-only); filter definitions are different. No runtime code was overlaid.
- `missing_import_edges.csv`: a conservative AST map for runtime-only Python modules. It identifies 307 references from 109 runtime-only modules to internal modules absent from the working checkout, including Math Desk dependencies. This is a port-planning map, not proof all imports fail at runtime or a replacement for an execution test.
- `video_course_custody_summary.json`: rechecked registry/caption/course counts and distinguishes physical VTT files from Finder sidecars and unavailable source claims.
- `rt01_rt02_receipt.json`: test fixtures, observed results, source hashes, limits, and fail-closed authority state.
- `post_freeze_delta.csv`: current hashes for files changed after the baseline freeze.

## RT-01: stale authority

The working `reports/red_team/latest_red_team.md` is a June 27 dYdX checkpoint with a P2 strategy-acceptance PASS. Current `final_1x_survivor_receipt.json` and `testnet_candidate_receipt.json` are absent. Two targeted tests in an isolated locked environment passed: a stale registered gate and a forged active plus immutable Stage-4 contract were rejected. This is **partial evidence**, not an end-to-end forged Hyperliquid release-chain test: no current complete receipt chain exists to mutate.

## RT-02: conflicting numerical gates

Before the fix, an isolated synthetic candidate with 15 trades, PF 1.2, Sharpe 0.7, and 20% drawdown passed `_walkforward_gate_blockers` with no blockers. It failed the stricter `config/research.yaml` targets (100 trades, PF 1.8, Sharpe 1.2, max 15% drawdown). A policy-integrity receipt returned `PASS` and `promotion_authority=true` solely because source-contract hashes matched. That did **not** establish an end-to-end promotion bypass, but the receipt overstated what hash validation proved.

The working code now always sets `promotion_authority=false` on the policy-integrity receipt while retaining its source-hash `PASS` status. A new focused regression test and four other focused checks passed in the isolated copy. The broader governance/statistical-remediation/release-gate set passed **63 tests**; program/registered-rerun sets passed **103 tests**. The isolated locked environment passed `uv pip check` and `uv lock --check`. The full test suite and a complete RT-01/RT-02 release-chain mutation were not run. A versioned numerical resolver is still required before any acceptance or promotion claim.

The active checkout's existing `.venv` remains incomplete: read-only `uv pip check --python .venv/bin/python` found five dependency incompatibilities (missing `attrs`, `langsmith`, `packaging` for two packages, and `pycryptodome`). No repair was made to that environment; the passing tests above used a disposable locked copy.

## Operative state

The archive's historical integrity pass is separate from strategy performance. The research run contract remains `DRAFT_BLOCKED`. Current pair-level account fees, funding, L2/impact, formula parity, correction policy, and numerical gate resolution are not bound to one run. Research acceptance, Testnet order authority, and live authority remain blocked.
