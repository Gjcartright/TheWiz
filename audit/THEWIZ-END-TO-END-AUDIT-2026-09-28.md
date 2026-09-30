# TheWiz V3 end-to-end audit

**Audit date:** 2026-09-28
**Working branch:** `codex/wiz-v3-source-reconciliation`
**Base commit:** `b98d5b9447a109938b57b239a7d458f47606d7c9`
**Scope:** accessible GitHub sources, local source copies, project architecture, quant/research contracts, capture evidence, safety controls, and offline diagnostics.

## Executive assessment

TheWiz is a large research and evidence-control platform for crypto statistical arbitrage. It contains pair discovery, vendor-mode capture, point-in-time replay, two-leg accounting, cost and risk models, walk-forward and regime gates, portfolio and execution interfaces, ML/RL research lanes, a LangGraph workflow, and substantial publication and authority controls.

The software and archived-capture integrity checks pass after the fixes below. The project is **not ready to claim a profitable pairs strategy, continuous observation, release readiness, or trading readiness**. The current archived observation has cadence gaps; no pair models or statistical acceptance runs were performed in this audit; provenance for later runtime changes and owner-only release inputs remain unresolved; and GitHub's default branch does not contain the application source.

The project source is now checked out under the Expansion drive at `/Volumes/Expansion/Crypto Wizard`. The audit report and graph are stored in this folder. Code edits are staged on the local branch for review and have not been committed or pushed.

## 1. Source and repository inventory

### TheWiz GitHub repository

Repository: <https://github.com/Gjcartright/TheWiz>

| Branch | Head | Audit result |
| --- | --- | --- |
| `main` (default) | `f5678b1a96690509c51d933c47bdc2811de463f0` | Contains only `docs/project_full_handoff_summary.md`; no application code. |
| `codex/corrective-evidence-plan` | `b98d5b9447a109938b57b239a7d458f47606d7c9` | Current application source branch; 1,898 tracked blobs. |
| `codex/specialist-strategy-scoreboard` | `3c5dd8674908567d1098b2104efefe1b97d96a66` | Same history, 12 commits behind the current code branch. |

The current code branch is 18 commits ahead of `main`. GitHub reports no pull requests or issues, and branch protection is disabled on all three branches. The code branch has a CI workflow, but those checks are not required by branch protection. The local working branch tracks `origin/codex/corrective-evidence-plan`; its fixes are staged and local only.

### Other repositories visible in the GitHub account

The authenticated account listing returned ten repositories:

| Repository | Relationship to TheWiz audit |
| --- | --- |
| `TheWiz` | Direct project source. |
| `research_public` | Historical quant research archive. Its tree includes pairs-trading and cointegration course notebooks and Quantopian templates; no current dependency from TheWiz was found. |
| `BearZ-VS-BullZ-DaBlock` | Tiny 2023 repository; no matching pair/quant source paths found. |
| `BVBD` | Empty repository. |
| `BZVBZ` | Private JavaScript repository; no pair/quant path names found in the tree scan. Not semantically audited file by file. |
| `living-story-os` | Separate private Python project. |
| `DentalAI` | Separate private project. |
| `Dental-agent` | Separate private project. |
| `dental` | Separate public project; empty/default branch metadata. |
| `The-Ave` | Separate JavaScript project. |

`research_public` is useful as historical educational material, not as a code dependency or evidence for current strategy performance. The other personal repositories were inventoried by metadata and tree paths; they were not cloned or reviewed file by file because no connection to TheWiz was indicated.

### Local source copies and storage

| Location | State |
| --- | --- |
| `/Volumes/Expansion/Crypto Wizard` | Now the working Git checkout, based on `b98d5b9`; on the requested Expansion drive. Git is configured locally with `core.filemode=false` because this mounted volume reports every regular file as executable. |
| `/Users/gregc/Documents/ChatGPT/TheWiz V3 GitHub` | Earlier clone of the same source branch. The reviewed staged patch was transferred into the Expansion checkout. |
| `/Users/gregc/TheWiz-LocalRuntime` | 2.7 GB unversioned runtime snapshot. Its `AGENTS.md` says to keep the snapshot and archived evidence read-only. It has later source and tests than GitHub. |
| `/Users/gregc/Documents/ChatGPT/TheWiz V3 Reconstruction` | Unversioned reconstruction copied from LocalRuntime, not an independent upstream source. Its baseline says runtime-control authority files and local environments were excluded. |
| `/Volumes/CodexWorkspace/Codex/TheWiz-publish-20260625` | Not present/mounted. This prevents claiming the changes are installed in the designated canonical checkout. |
| `/Volumes/Expansion/CodexWorkspace.sparsebundle` | Present but unavailable because it remains locked. It was not modified or inspected by this audit. |

The Expansion checkout initially contained only an empty Git repository. Fetching the project exposed AppleDouble `._*` metadata sidecars inside `.git/objects/pack`; Git reported a non-monotonic pack index. Removing only those generated sidecars and setting this checkout's file-mode handling restored a clean tree before the code patch was applied. `git fsck --full --verbose` then completed successfully.

## 2. Project components, piece by piece

| Layer | Main responsibilities |
| --- | --- |
| Discovery and vendor evidence | Crypto Wizards scanner/API and pair-detail capture, exact strategy mode and timeframe identity, source response archives, field dictionaries, and credit accounting. Vendor results create hypotheses; they do not independently approve trades. |
| Market-data ingestion | Hyperliquid, dYdX, Binance/CCXT, Yahoo, and fixture readers. The code supports public market-data acquisition and point-in-time history workflows. |
| Data contracts | Symbol, interval, bar-time, closure, lineage, source/config/formula identity, universe lifecycle, freshness, correction/supersession, and exact-mode controls. |
| Pair and feature research | Z-score, static/dynamic/OU spread modes, copula, ECM, Hurst, half-life, regime and related strategy families. The registry and experiment harness compare families, pairs, regimes, and cost buckets. |
| Quant math and accounting | Hedge-ratio contracts, two-leg weights, lagged returns, trade timing, trade ledger, Sharpe/drawdown/profit factor, funding, fees, slippage, execution risk, and partial-fill cost assumptions. The legacy scalar spread proxy is explicitly distinct from currency-denominated two-leg P&L. |
| Statistical acceptance | Point-in-time replay, walk-forward folds, regime stability, parameter robustness, family-wide multiple-testing/FDR controls, pair concentration, cost stress, and independent support clusters. |
| Evidence and authority | Hash-bound evidence registries, immutable publication helpers, effect/publication authority, release and cadence gates, redaction, recovery controls, and fail-closed test/owner-admission contracts. |
| Orchestration | CLI stage contracts wrapped by LangGraph. The runtime graph initializes state, runs selected existing stages in order, routes fail-fast/report-only outcomes, and writes workflow reports. The graph itself does not prove that any stage ran or passed. |
| Portfolio and execution | Portfolio/risk interfaces, dYdX/Hyperliquid adapters, market compatibility, journal/watch paths, Testnet candidate gates, and a live path disabled by default. These interfaces do not mean an order was submitted. |
| ML and RL research | Trade labels, leakage audit, walk-forward model gates, advisory student/shadow designs, and RL candidate/handoff paths. The offline foundation expressly does not train, promote, recommend, paper trade, or execute. |
| Dashboards and auxiliary apps | Report builders, field dictionaries, operator guides, and the bundled static `apps/the-ave` web asset. These are not the quant platform's execution authority. |

The current GitHub code branch contains 262 Python source files under `src/quant_platform` and 186 Python test files. The runtime snapshot contains additional source, tests, reports, configuration, and offline research packages; see the reconciliation findings below.

## 3. Python, Node, and graph setup

The core runtime is Python. `pyproject.toml` requires Python 3.11 or newer; CI uses Python 3.12, and the lock pins the primary NumPy/Pandas/SciPy stack. The reproducible path is `uv sync --extra dev --locked` followed by `uv run --locked ...`.

There is no `package.json` or npm lockfile. Node.js is auxiliary: the audited host has Node `v22.18.0` and npm `10.9.3`; browser-capture scripts expect a host-provided `tab.playwright` object and are not standalone Playwright programs. The optional TradeStaq MCP setup uses `npx` and downloads a package on first use. That optional network install was not run.

The two relevant graphs are different:

1. The **research/authority graph** moves from candidate discovery through exact setup capture, immutable data, validation, point-in-time replay, observed costs, walk-forward acceptance, journal/model review, and separately approved venue gates.
2. The **LangGraph control graph** is `initialize → run_stage loop → finalize`; its `run_stage` node dispatches the existing stage contracts. Eight documented lanes describe project intake, capture, local verification, journal, execution compatibility, ML, RL, and red-team/review.

The full component graph is in [`thewiz_component_graph.mmd`](thewiz_component_graph.mmd). It distinguishes workflow edges from evidence that a stage actually ran.

## 4. Pair-trading research and acceptance

The project correctly separates discovery from acceptance. Crypto Wizards exact-mode evidence is intended to be compared against local point-in-time replay with matched pair, mode, timeframe, lookback, timing, hedge/sizing, costs, and trade lifecycle. Acceptance is based on local market evidence and observed costs, not a high vendor Sharpe alone.

The versioned `config/acceptance_policy_manifest.json` is policy version `2026-08-09.1`. It specifies five walk-forward folds, at least three positive folds and ten completed trades, base and stress cost buckets, an `ALL` regime, FDR control, independent support clusters, regime/parameter/concentration checks, and strict funding/fee/L2 evidence. Testnet and live are separate gates; live is false by default and requires explicit authorization.

There are two current threshold inputs, not one. `config/research.yaml` sets higher performance targets: 100 completed trades, profit factor 1.8, Sharpe 1.2, maximum drawdown 15%, and at least two pairs. The acceptance manifest hash-binds `research.yaml`, but the inspected code and docs do not specify whether these gates are conjunctive or which promotion stage applies each one. The README and architecture now preserve both inputs and state a conservative fail-closed rule: require both until a versioned resolver clarifies their relationship. No policy threshold values were changed.

The June handoff records a BNB/STX daily example with 400 rows, one closed trade, PF 4.1021, Sharpe 2.4319, 39.57% maximum drawdown, and 14.07% total return; it was rejected for high drawdown and thin sample. This is historical report evidence, not a current accepted strategy.

No pair-model fitting, pair ranking, strategy backtest, walk-forward acceptance, or performance claim was made in this audit. The multi-asset capture's 300 combinations per capture are candidate combinations, not tested models.

## 5. Archived capture evidence

The separate read-only audit of `LocalRuntime/data/research/multi_asset_captures` found:

- 160 receipt-chain entries and all recorded raw-file hashes passed the official verifier; terminal receipt `universe-20260927T101733Z`.
- 4,000 candle files and 3,320,647 rows validated; zero invalid files, zero conflicting repeated closed-candle fields, and zero missing in-file hourly intervals.
- Point-in-time same-day universe membership yielded 3,625/3,625 eligible asset-hour observations, with zero eligible candle gaps.
- The archive contains 35 distinct asset names over time and 25 assets per capture.
- Cadence remains incomplete: 158 of 168 inferred hourly slots, ten inferred misses, one duplicate slot, and one extra receipt just beyond the inferred window. The schedule grid is inferred because no authoritative schedule manifest was found.

These checks establish stored candle integrity and eligible bar coverage. They do not establish uninterrupted collection, tested pair performance, or profitability. The existing source instructions require the archive to remain immutable and require an owner-approved schedule manifest before any collection restart.

## 5.1 Cost-evidence readiness

The source contract hash checks pass for the manifest-bound `research.yaml`, discovery policy, and Hyperliquid perpetual cost profile. That verifies file identity, not that a complete cost contract is bound to this capture run.

- The pinned account fee profile is dated 2026-08-05 (54 days old at audit time), says account-specific fees were not captured, and directs use of a conservative base tier. The manifest's maximum fee age is 30 days.
- A separate generic fee reference dated 2026-09-16 is more recent, but its page response carried stale-cache metadata, it has no explicit effective date, is not account-specific, and is marked `production_equivalent=false`. It cannot replace the pinned account profile.
- A 2026-09-16 L2 calibration has 12 samples per asset at 600-second spacing over 110 minutes for BTC and ETH. It meets the manifest's sample-count and span shape for those two assets only; it is calibration-only, earns no cadence credit, and is marked `production_equivalent=false`. The captured universe has 25 assets per run.
- The manifest requires recent pair-specific fees, 95% funding coverage, and positive stressed expectancy. No run-bound pair-cost manifest connects fee tier, funding, impact, formula, and the full point-in-time universe to these 160 captures.

Therefore these cost artifacts are useful research inputs, but pair-level after-cost readiness is **not established**. Obtain or approve current run-bound cost inputs before quantitative pair acceptance.

## 5.2 Research run contract draft

I prepared [`research_run_contract_draft.json`](research_run_contract_draft.json) as the next reviewable artifact. It binds the current policy/config hashes to the verified 160-receipt archive identity, captures the terminal receipt hash and observed coverage gaps, and preserves both acceptance threshold sets as required gates. It explicitly blocks pair analysis until the formula, universe lifecycle, correction policy, current pair-cost evidence, and policy relationship are approved and run-bound. The draft does not authorize collection, training, promotion, or trading.

The deeper design review is in [`PAIRS_RESEARCH_DESIGN_2026-09-29.md`](PAIRS_RESEARCH_DESIGN_2026-09-29.md). Its recommended resolver maps the manifest to a non-promoting research-candidate decision and `research.yaml` to a stricter later promotion gate; owner approval is required before adopting that interpretation.

Detailed evidence and CSVs are in `/Users/gregc/.codex/visualizations/2026/09/28/01a0e848-5c1e-7143-9268-74ceb8140e14/`.

## 6. Source drift and reconciliation risk

The eligible-text-file reconciliation compared 1,878 GitHub files with 2,571 LocalRuntime files:

- 1,692 identical common paths;
- 177 changed common paths (98 source modules, 66 tests, 13 docs/config/scripts/build files);
- 9 GitHub-only paths;
- 702 LocalRuntime-only paths, including 208 source files, 234 tests, and 173 reports.

An AST import-map audit found 22 of 35 changed safety/release modules depend on 24 internal modules absent from the GitHub checkout, leaving 77 missing internal import edges. The later runtime code is therefore a dependency-coupled bundle, not a safe set of individual cherry-picks. No broad overlay was performed. The encrypted workspace was excluded as requested; provenance for those later source changes remains unresolved.

## 7. Verified findings and fixes

| Priority | Finding | Action/result |
| --- | --- | --- |
| P1 | TheWiz default branch contains only the handoff doc; application source sits on an unprotected feature branch. | Confirmed through GitHub branch/tree/compare APIs. Source fixes are staged locally; no remote push or branch change was made. Protecting and merging the code branch remains a repository-owner action. |
| P1 | The collector wrote evidence with direct `os.open`/`os.rename`, bypassing the project's publication authority; the full suite caught three unpaired publication surfaces. | Replaced with `create_exclusive_bytes` and `promote_staged_directory`; added publication preflight. Each public Hyperliquid request now consumes an exact one-use network permit before its transport callback. |
| P1 | LocalRuntime is unversioned and substantially diverges from GitHub. | Preserved it read-only; recorded subsystem counts and missing dependency edges; did not overlay it into the checkout. |
| P1 | The seven-day capture was not continuous and has no authoritative schedule manifest. | Kept the archive unchanged and report the cadence gaps as inferred, not canonical. |
| P1 | Two current threshold inputs have an undefined relationship: the manifest gates and stricter targets in `research.yaml`. | Updated README and architecture to preserve both, fail closed by requiring both, and require a versioned resolver before changing policy. |
| P1 | Available cost artifacts do not establish pair-level cost readiness for the captured universe. | Recorded profile age/account-specific gaps, limited BTC/ETH calibration scope, and the missing run-bound pair-cost contract; no policy or archive data was changed. |
| P2 | Collector metadata accepted non-finite numeric values, and API-provided asset names were embedded in evidence filenames. | Excluded non-finite/negative metadata from universe ranking, rejected non-finite thresholds, percent-encoded asset labels for paths, and added regression cases. |
| P1 | GitHub Actions called `python -m pip check` inside a uv environment that has no `pip` module. The local locked environment reproduced `No module named pip`. | Changed CI to `uv pip check --python .venv/bin/python`; verified the command against the installed locked runtime (`135` packages compatible). |
| P2 | Workspace validation hardcoded an obsolete path and required `.env.local`/`python-dotenv` for a local Git identity check. | Made root discovery script-relative, removed credential inspection, and records branch/head/upstream and working-tree state. Syntax and behavior passed in an isolated checkout. |
| P2 | Workspace validation counted a clean tree as one changed line. | Corrected empty-status handling and verified the clean disposable checkout reports zero changes. |
| P1 | The capture verifier counted every matching folder without proving every folder belonged to the terminal receipt chain; the collector also allowed an independent root or branch from an old receipt. | Verification now requires an exact match between capture folders and the terminal chain; collection may only extend the current terminal receipt. Added disconnected-folder, new-root, and branch regression cases. |
| P2 | AppleDouble `._*.py` sidecars on the Expansion volume were misread as unreadable Python by the source publication scanner. | The scanner now skips AppleDouble metadata files; real-volume publication-surface and staging-pair checks both return zero blockers. |
| P2 | Malformed metadata could repeat an asset name or supply booleans as numeric market values, yielding a capture the verifier would reject. | Rejects duplicate symbols and invalid numeric types before candle requests or publication. |
| P2 | Setup docs used `.venv311`, bare `pytest`, and mutable/unlocked install instructions. | Updated Quick Start and LangGraph commands to use the locked `uv` environment, and documented that dry-run workflows still write reports. |
| P2 | Node setup and the graph were ambiguous. | Documented Python as the core runtime, Node as an optional helper/MCP layer, and supplied the component/authority graph. |

## 8. Diagnostic inventory and results

| Diagnostic | Result | Scope/meaning |
| --- | --- | --- |
| Full pytest suite | **2,359 passed** in 239 seconds | Reproduced in a disposable copy after `uv sync --extra dev --locked`, using `uv run --locked python -m pytest -q`; no operational report or retained capture archive was a write target. |
| Focused collector + publication registry tests | **47 passed** | Includes raw/schema/receipt checks, fail-closed file and network authority, linear receipt-chain validation, non-finite/duplicate/invalid metadata rejection, and safe asset path labels. |
| Python source compilation | Passed | `compileall` over `src` and `scripts` in the disposable copy. |
| CI Ruff checks | Passed | `ruff check src tests scripts --select E9,F63,F7,F82`. |
| Collector Ruff checks | Passed | Full lint on collector, verifier, and tests. |
| Lockfile resolution | Passed | `uv lock --check`; 140 packages resolved. |
| Locked dependency tree | Passed | `uv tree --locked --depth 1`; core and optional dev/dYdX/RL/YouTube dependencies listed. |
| Installed dependency consistency | Passed | `uv pip check`; 135 packages checked in the available runtime. |
| Workspace validator | Passed | `bash -n` and execution in a clean disposable checkout; it reported branch/head and zero pre-run changes, then wrote its status report there. |
| CI workflow YAML | Parsed | PyYAML accepted `.github/workflows/quality.yml`. |
| Git object integrity | Passed | `git fsck --full --verbose` after removing generated AppleDouble metadata sidecars. |
| Capture receipt/hash verification | Passed, 160 receipts | Earlier read-only official verifier; all raw hashes passed. |
| Candle schema and coverage audit | Passed, 4,000 files | Strict candle shape/time/value checks, repeated closed-field reconciliation, and point-in-time membership coverage. |
| Cadence audit | **Not continuous** | 158/168 inferred slots; ten missing and one duplicate; no authoritative schedule manifest. |
| Offline research foundation tests | **16 passed** | Synthetic/temporary-file tests for alignment, gaps, statistics, cost filters, receipt hashes, correction-aware ledgers, and observe-only contracts. No active captures were modified; this does not run pair analysis on the archived market data. |

### Self-inspection and score

**Audit quality: 9/10 after the final scrub (initial self-score 8/10).** The source, architecture, safety controls, capture archive, and test surface were audited broadly, and the conclusions distinguish data integrity from strategy performance. The final pass caught authority, chain-accounting, and mounted-volume metadata gaps and added regression coverage. Pair-level empirical validation is still unperformed because the required policy and run-bound cost inputs are missing; this score does not imply a strategy passed.

The review found implementation gaps and one verification-command issue, all corrected above. Non-finite metadata could distort ranking, asset names were used in raw filenames, public reads did not consume network permits, and the verifier did not prove every folder was connected to the chain. The collector now rejects malformed/duplicate metadata, encodes asset labels, consumes exact public-network permits, and validates a single linear chain. The source scanner ignores AppleDouble sidecars. The first fresh `uv run pytest` attempt lacked the dev extra and resolved the host pytest executable, producing a NumPy/scikit-learn ABI collection error. The README and CI now use `uv run --locked python -m pytest`, and the full suite passes after syncing the locked dev extra. An apparent duplicate candle-alignment predicate during review was a display overlap between adjacent excerpts; the source contained only one predicate.

### Available next-stage diagnostics

The codebase and contracts provide additional tests that were **not run** in this audit because they require owner custody, external services, or a scoped release decision:

- source identity, startup preflight, fixed-route schedule and cadence checks;
- runtime/process/boot identity, handoff, watchdog, restart, duplicate prevention, and release-chain tests;
- exact Crypto Wizards mode parity, local Hyperliquid replay, two-leg after-cost performance, L2/funding coverage, walk-forward folds, regime and parameter stability, concentration, and FDR diagnostics;
- venue compatibility and Testnet candidate evidence checks;
- live-canary and order-authority tests requiring explicit authorization.

The LocalRuntime's `RECOVERY_STATUS.md` documents historical batches: 30 startup-preflight tests, 92 startup-verifier tests, 7 fixed-route tests, 23 process/boot checks, 2 barrier tests, 39 release/accounting/admission tests, and 157 failure/handoff/watchdog/restart/duplicate-prevention tests. These are historical receipts; they were not rerun against the current source checkout. Passing those batches would still not qualify a real service release.

## 9. Not run

- No Hyperliquid, dYdX, Crypto Wizards, or other market/API network request.
- No browser automation, npx package installation, or Node helper execution.
- No credential, Keychain, authority key, or `.runtime_control` value was read, copied, generated, or printed.
- No write to the LocalRuntime archive or locked sparse bundle.

## 10. Next steps

1. Use this Expansion checkout as the working source copy; preserve the current source branch and immutable LocalRuntime data while reconciling provenance for later runtime changes.
2. Complete the missing approvals and bindings listed in the research run contract draft; keep the archive read-only and preserve the both-gates-required interim rule.
3. Reverify the receipt chain and raw hashes immediately before any approved pair analysis, then run point-in-time, after-cost, multiple-testing-controlled walk-forward research. The 300 combinations per capture remain universe candidates until tested.
4. Review the staged source diff, then merge it into GitHub's default branch through an owner-reviewed change and enable required CI branch protection.
5. Restore exact owner-held release/journal/schedule inputs from custody only if a later release stage needs them; do not recreate missing material.
6. Obtain an owner-approved schedule manifest and publication authority before any new capture.
7. Keep Testnet and live stages separately gated and explicitly authorized.
