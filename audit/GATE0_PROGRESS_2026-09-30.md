# Gate 0 progress and open decisions

**Status:** Open. This is a source and custody receipt, not a research acceptance or trading authorization.

## Canonical source candidate

- Writable checkout: `/Volumes/Expansion/Crypto Wizard` on the mounted Expansion drive.
- Branch at inventory: `codex/wiz-v3-source-reconciliation`; prior committed head: `b98d5b9447a109938b57b239a7d458f47606d7c9`.
- The internal `TheWiz-LocalRuntime`, forensic copies, candidate tree, and encrypted sparsebundles are not writable source authorities. The encrypted bundles remain unmounted.
- Local staged, unstaged, and untracked changes still need a reviewed source commit. The public `TheWiz` code branch has not yet received them.

## Custody and restore

- Midnight save point: `/Volumes/Expansion/TheWizNightlySavepoints/2026-09-30`.
- Mac internal copy: `/Users/gregc/Backups/TheWiz/savepoints/2026-09-30`.
- Private GitHub source backup: `Gjcartright/TheWiz-nightly-backup`, branch `codex/nightly-backup`, commit `1143bb6b3afc7bf84b5f1428d7be079650ee5084`.
- The dated receipt records `PASS` separately for all three destinations at `/Users/gregc/Backups/TheWiz/state/2026-09-30.json`.
- A new restore directory at `/Volumes/Expansion/TheWizRestoreRehearsal-2026-09-30` was copied from the Mac internal save point. Before any test modifications, its 10,787 manifest entries matched the stored SHA-256 manifest `b44b6fc5e485b7d447b1e8b422f8973aa9917b23b94a434f77c5f177145d423e` exactly. The restored project remains disposable; later environment and test files do not alter the source backup.
- The 553 historical path/hash variants were separately preserved on the Mac internal drive at `/Users/gregc/Backups/TheWiz/source-delta-evidence/2026-09-30`: 528 distinct blobs, 35,911,036 bytes, all copied and rehashed. Its manifest SHA-256 is `2d0d5a213b91d966a8677f7d62ea9a4b6a6ac0d125dec509099ae49aab10970b`. This is custody protection, not a source selection.

## Source delta

The dated 856-row same-path drift scan is normalized in `GATE0_SOURCE_DELTA_REGISTER_2026-09-30.csv` and its JSON summary. There are 427 paths and 553 distinct path/hash variants: 190 match a known Git ancestor, 133 are retained as historical evidence rather than source, and 230 still require semantic review. None of those 230 hashes is content in the currently reachable Git object set. This register does not claim that the pending 230 were ported or adjudicated.

The candidate branch contains a collection startup verifier and a precollection poll launcher absent from the active tree; their support modules and configuration are also absent. They are preserved for later integration review and are not a safe drop-in replacement. Historical live-canary and scheduler variants require safety-specific review before any source selection.

Static symbol triage parsed 213 historical Python variants and found 57 with top-level functions, classes, or test cases absent from the active counterpart. Notable examples include the forensic `active_pipeline.py` point-in-time history and local-authority functions, `backtest.py` hedge/rebalance helpers, and `statistics/math_v2.py` integration-order/Johansen functions. These are priority review items, not proof the active implementation is wrong or that the old code is safe to import. The full list is in `GATE0_SYMBOL_TRIAGE_2026-09-30.csv`.

Direct diff review found specific acceptance and math gaps requiring a code decision: the forensic `experiments.py` includes lower-bound expectancy, flat-end-state, finite-profit-factor, and ledger-reconciliation checks absent from the active counterpart; its `backtest.py` supports fixed-units holding and explicit spread orientation; its `statistics/math_v2.py` has an I(1) diagnostic and Johansen estimator absent from the active counterpart. No historical implementation has been copied. Current strategy acceptance stays blocked while those differences are resolved and independently checked.

## Patch review and runtime

The pending collector patch had a concrete transport defect: it read the Hyperliquid response after leaving the response context. The active patch now reads inside the context and has a focused regression check. The old active `.venv` is incomplete (`pytest` missing and `_distutils_hack` import error), so it is not a valid dependency receipt. The locked offline environment built successfully on the Mac internal drive (105 installed packages; `uv pip check` passed) and `uv run --locked` selects it when `UV_PROJECT_ENVIRONMENT` is set. A disposable copy of the same lock and relevant code passed 54 focused tests, compilation, and correctness lint.

The bare graph CLI dry-run correctly refused governed report writes without publication authority. `scripts/ops/run_langgraph_dry_run.py` now gives that diagnostic only a scoped, file-only report authority. In the disposable restore it produced seven report files for 25 dry-run stages, eight agent lanes, and eight workflow edges, with no graph blocker. All 25 status rows say `dry_run`; no stage action or strategy acceptance is inferred. A second discovery dry-run also passed under the same private journal after the full suite released its lock. The helper creates the private authority journal under `.runtime_control` in the disposable root. Code-branch CI is still pending.

The first full disposable suite finished with 2,358 passes and three failures, all caused by macOS `._*.py` AppleDouble files on the exFAT volume being included in Python-source inventories. The financial-effect scanner and two repository-wide AST tests now skip that metadata prefix. A focused rerun of the affected checks passed 11 tests. The patched full suite then passed **2,362 tests** in 365.48 seconds. `.runtime_control/` is now Git-ignored to keep the graph authority journal out of the private GitHub source mirror.

## Remaining Gate 0 exit work

1. Finish semantic decisions for the 230 historical source/test/document variants and bind an immutable source commit with a reviewed patch record.
2. Record the locked environment and LangGraph report results against that exact commit in a disposable checkout.
3. Push a reviewable code branch, obtain green CI for the commit, and tie the restore/backup receipt to it.

No Gate 1 data qualification, strategy acceptance, Testnet order, or live trading authority follows from this progress receipt.
