# Exhaustive Crypto Wizards To Hyperliquid Goal Plan

## Goal

Build one reproducible research-to-testnet pipeline that starts with every row exposed by the Crypto Wizards crypto scanner and ends with evidence-gated Hyperliquid perpetual testnet validation. Dashboard observations are discovery and hypothesis evidence. Only point-in-time, costed local replays and complete testnet lifecycle evidence can advance a configuration.

This plan does not authorize live trading.

## 2026-08-08 Execution Decision Update

Hyperliquid Testnet is the only execution-validation venue in this goal, and it is treated as a perpetual account with leverage capability. Crypto Wizards remains the exhaustive discovery, configuration, and diagnostic source; it is not an execution authority. Hyperliquid market data, local point-in-time replay, risk evidence, current account state, and the deterministic safety gate decide whether a configuration may progress to a bounded Testnet lifecycle test.

The implementation order is fixed:

1. account for every Crypto Wizards pair, exact mode, and orientation without a Sharpe or return intake filter;
2. map both legs to current Hyperliquid Testnet perpetuals and retain every unavailable pair with an explicit blocker;
3. repair and complete the rolling two-hour L2 cost-calibration window so fees, slippage, funding, and freshness use one coherent time basis;
4. prove statistical edge with canonical 1x, point-in-time, after-cost walk-forward evidence;
5. require false-discovery, stability, regime, concentration, and robustness gates before leverage analysis;
6. evaluate 1x, 1.5x, 2x, 3x, and 5x as separate leverage/margin scenarios, capped by both legs, portfolio policy, maintenance margin, and liquidation buffer;
7. refresh the Hyperliquid agent, perpetual-margin balance, open-position, market-rule, and order-submission preflight immediately before any Testnet action;
8. run a bounded two-leg Testnet lifecycle at 1x first, then permit only leverage scenarios with their own accepted risk record;
9. journal fills, failures, recovery actions, and realized Testnet outcomes as separate labels for the learning system;
10. keep live trading disabled until a later, explicit human release decision.

Current authority remains `RESEARCH_ONLY`: the rolling L2 cost calibration and no-order Testnet preflight are complete, but zero candidates pass the family-wide statistical-selection gate and the perpetual account currently has zero collateral. Leverage cannot bypass either blocker.

## Non-Negotiable Rules

- Sweep every configured Crypto Wizards crypto venue and timeframe with all-case filters.
- Do not filter intake by Sharpe, return, liquidity, stationarity, copula, trade count, or Hyperliquid availability.
- Preserve duplicate source observations and assign stable evidence identifiers.
- Test the seven pair-page modes: Static Spread, Static ZScoreR, Dyn Spread, Dyn ZScoreR, OU Spread, OU ZScoreR, and Copula.
- Preserve `ou_optimal` as the true/false scanner annotation on its source row. It may stratify those seven modes, but it is not an eighth pair-page mode or an independent signal.
- Test original and reverse orientations because hedge ratios, error-correction behavior, and copula conditionals may be asymmetric.
- Keep unavailable or ambiguous Hyperliquid mappings with explicit blockers.
- Treat Hyperliquid Testnet as a perpetual execution-validation venue; leverage availability does not authorize leverage use.
- Run canonical replays at 1x gross portfolio leverage before leverage. Leverage cannot rescue a weak unlevered strategy.
- Keep every leverage and margin result in a separate scenario record; never overwrite the canonical 1x result.
- Separate research, replay, testnet, and live authority.
- Never use future data or full-sample dashboard hindsight as a live feature.
- Record every pass, failure, skip, and blocker. No silent drops.
- Treat each dashboard snapshot as immutable and time-bounded; never assume a saved board is still current.
- Interleave scanner and pair-detail capture by venue/timeframe cell so board drift cannot silently invalidate the route queue.

## Implementation Phases

### 1. Exhaustive Run Contract

Status: complete.

Define immutable source-row IDs, pair-group IDs, exact modes, both orientations, canonical 1x replay status, Hyperliquid mapping status, blockers, evidence paths, and no-live-trading authority.

### 2. Live Scanner Capture

Status: complete for the 2026-08-07 20:29 UTC interleaved refresh checkpoint.

Capture Binance, Binance US, ByBit, Coinbase, and dYdX across Daily and Hourly. Use all-case filters and scroll until repeated bottom fetches add no rows. Preserve every visible metric and raw field.

Current checkpoint:

- 10 of 10 venue/timeframe cells complete
- 300 source rows retained
- 247 venue/timeframe pair groups
- zero discovery thresholds
- full source-row accounting
- immutable run ID `ewhl_5cf7c50e51f6fee5010c`

The prior snapshots remain immutable evidence. A live refresh showed that pair membership had already changed, so the active run points to the complete 20:29 UTC 10-cell snapshot instead of overwriting any earlier raw files.

### 3. Pair-Detail And Exact-Settings Capture

Status: in progress.

Create one work item for every pair group. Resolve the pair-page route, open the pair detail, cycle through every exact mode and both orientations, and capture:

- spread and z-score series
- lookback, interval, entry, exit, and hedge settings
- Wizard fee and cost assumptions
- Pearson, Spearman, Kendall, and conditional dependency fields
- Copula family, conditional probabilities, tail fields, and arbitrage state
- ECM X, ECM Y, and ECM strength
- Johansen and Engle-Granger states and statistics
- Hurst, half-life, volume, liquidity, and risk fields
- backtest trades, return, Sharpe, drawdown, and exported payloads

Scheduling may put Hyperliquid-ready pairs first, but it cannot remove blocked pairs from the queue.

Operationally, capture is cell-interleaved and resumable:

1. capture one venue/timeframe scanner cell with all-case filters;
2. write its immutable scanner evidence immediately;
3. open each unique pair from that same live cell while it is still available;
4. write one immutable pair bundle after every pair;
5. reconcile the bundle to venue, timeframe, and normalized assets, never the session-local route number;
6. checkpoint captured, unavailable, disappeared, and pending reverse-orientation cells before changing scanner cell.

Current board checkpoint:

- the active progress ledger accounts for all 247 pair groups: 209 `COMPLETE`, 35 `NOT_CAPTURED`, two `ROUTE_UNAVAILABLE`, and one `PARTIAL`
- the mode ledger contains 2,947 verified captured cells, 422 explicit `NOT_AVAILABLE_ON_PAIR_PAGE` cells, 32 pair-route-unavailable cells, and seven pending reverse-orientation recalculations; 544 of the 3,952 planned experiment cells still have no pair-detail row because their pair group has not been captured
- the remaining queue is preserved without substitution; 2026-08-08 retries could list the signed-in Chrome Pair 423 tab but claiming or reading its page content timed out, while the in-app fallback redirected to sign-in, so no dashboard values were guessed or backfilled from stale evidence
- a fresh authenticated API sweep completed at 2026-08-08 11:09 UTC across all 30 configured venue/timeframe/strategy cells with no Sharpe or return threshold, using 300 credits and returning 944 rows
- overlay `cwouoverlay_cf3ba114a2e07d608c56` accounts for all 944 scanner rows and preserves 163 `ou_optimal=true` and 781 `false` observations across their actual base mode and orientation; one invalid DOGE/DOGE source row is retained under the original accounting orientation with its Hyperliquid blocker, zero flagged rows pass walk-forward or family-wide selection, and the overlay has no promotion or execution authority
- the API refresh bridge accounts for all 944 rows and compares 500 current API pair groups with the immutable 247-pair browser run: eight match, 492 are new API discoveries, and 239 frozen pairs are absent from the current API refresh; all 739 union groups remain visible
- all 500 current API pair groups are queued for authenticated pair-detail capture; API-only evidence has no promotion or live-trading authority and does not mutate the frozen exhaustive run
- all 739 union pair groups are mapped against the same frozen 2026-08-08 Hyperliquid Testnet perpetual inventory: 186 resolve to both legs and 553 remain visible with explicit missing, ambiguous, or normalization blockers; among the 500 current API groups, 109 resolve and 391 are blocked; mapping readiness does not bypass pair-detail or local replay gates
- the pair-detail acquisition plan shows that a full documented GET bundle would cost 87,000 credits for the current queue and still omit ECM; the authenticated dashboard remains the complete zero-credit vendor lane, while local Hyperliquid recomputation is the scalable acceptance lane and must never be labeled Wizard parity
- a bounded ETH/WIF API pair-detail pilot now completes all six documented GET endpoints and inventories 85 fields; the corrected Copula backtest uses the documented 0.05 entry and 0.50 exit levels, the one-endpoint retry consumed exactly six credits, and failed-attempt history is retained; 11 of 15 coverage groups are present, while ECM x/y/strength and dashboard entry/exit controls remain explicitly missing
- current-board handoff `cwhandoff_1cc93f005dcc6da38e59` accounts for all 500 API pair groups. Its frozen 8,000-cell compatibility matrix contains 7,000 real pair-page mode/orientation cells plus 1,000 legacy OU Optimal accounting cells. The latter are now semantically `NOT_APPLICABLE_VENDOR_MODE`, not an unimplemented signal; the immutable overlay ledger carries every real scanner boolean to its actual mode. The handoff has 1,526 local cells ready for point-in-time history across 109 Hyperliquid-ready pairs, 6,256 cells blocked by explicit mapping failures, and 95 deduplicated asset/interval fetch requests.
- full history run `cwhistoryrun_20260808T133322858175Z_7c6a3ca4` attempts all 109 Hyperliquid-ready current-board pairs, preserves all 500 pair statuses and all 95 deduplicated asset requests, completes 74 asset histories and 75 synchronized pair histories, and retains explicit blockers for 34 selected pairs plus all 391 mapping-blocked pairs
- current-board canonical replay `cwcanonical_20260808T133906487917Z_6a068731` accounts for all 8,000 frozen compatibility cells at 1x: 1,047 replays complete, 213 are sample-supported for research ranking, 476 lack point-in-time history, 218 mapped legacy OU Optimal pseudo-cells are non-applicable under the corrected vendor semantics, two local fits fail, one replay error is preserved, and all 6,256 mapping failures remain visible; 12,299 closed-trade rows are retained and no replay has acceptance authority
- current-board cost evidence `cwcost_e92ec2fec2ec2221eb61` accounts for all 500 pairs and all 8,000 cells, completes 58 required funding assets using 33 point-in-time cache records and 25 bounded network fetches, marks 44 pairs and 614 experiments ready for provisional observed-cost research, blocks 31 selected pairs on funding/L2 evidence, and finds zero strictly L2-calibrated pairs
- observed-cost replay `cwobserved_63acae76d6672c7a40a5` accounts for all 8,000 cells, completes 614 canonical 1x replays, identifies 130 sample-supported research candidates, writes 7,559 trade rows, and grants zero acceptance or live-trading authority
- purged expanding walk-forward `cwwalk_e3f19e6cf67c65cc74dc` accounts for all 8,000 cells, evaluates 614 candidates and 130 primary sample-supported candidates across 3,070 of 3,070 planned folds, and writes 11,985 closed-trade rows plus 492,496 point-in-time bar rows; 22 cells pass the practical research gate, but zero pass the 10% Benjamini-Hochberg family-wide selection gate and selection hindsight is false
- causal regime attribution `cwregime_fbc9f759291255ec1169` evaluates all 22 practical walk-forward passes using prior-only thresholds; 11 pass the research regime-stability diagnostic, 84 detail rows, 16,213 enriched bars, and 654 trades reconcile, and the regime diagnostic cannot override the empty statistical-selection cohort
- robustness `cwrobust_6c71ca6138c952cde45b` reconciles 242 of 242 parameter/cost scenarios and 1,210 of 1,210 fold replays for all 22 candidates; 14 pass the research robustness diagnostic, but zero retain family-wide statistical selection
- cross-cell concentration `cwconcentration_bdb7208f0a9497faba34` accounts for all 8,000 cells and evaluates 22 practical walk-forward passes, 14 research robustness passes, and zero statistically selected robustness passes across pair, timeframe, Wizard venue, exact mode, orientation, and causal regime; the empty promotion cohort fails closed and cannot pass by default
- full-cohort failure attribution `cwfailure_92d364bdc4d96f23afe7` joins every stage, including concentration, with all 8,000 frozen experiment IDs accounted: 6,256 first fail Hyperliquid mapping, 592 fail walk-forward, 479 fail canonical replay, 433 fail cost evidence, 218 legacy pseudo-cells were originally labeled as awaiting OU Optimal implementation, 11 fail regime stability, nine fail statistical selection, and two fail robustness. The new overlay supersedes only that 218-cell interpretation; it does not alter any frozen replay result.
- the earlier ETH/WIF reverse-Copula pilot survivor is superseded by the full-family false-discovery result; the authoritative current cohort has zero 1x research survivors, zero leverage-research candidates, zero execution-acceptance-ready cells, and zero Testnet-preflight-eligible cells
- leverage surface `cwleverage_35e9f0d22576a0a75720` preserves all 8,000 experiment statuses and correctly creates zero candidate surfaces and zero scenarios because no configuration survived 1x; refreshed market and margin evidence is inside the one-hour research threshold, but Testnet order authority remains false and live trading remains unauthorized
- dated learning ledger `cwlearning_4ad0ec6cb800e44c9558` preserves one research-outcome record for every one of the 8,000 cells with zero training-eligible aggregate summaries, zero paper labels, zero live labels, zero realized outcomes, no Wizard label authority, and no order submission
- final frozen-chain validation `cwvalidation_b54199c02a3d49b7c2a9` passes all 161 explicit lineage, source-row accounting, all-cell accounting, active/snapshot parity, stage-validation, Testnet-state, learning-separation, mode-fidelity, and no-live-authority checks across 14 stages and all 8,000 compatibility cells; the chain now independently validates the 944-row exhaustive refresh and 163/781 OU Optimal overlay split, while trade eligibility remains `BLOCKED_NO_ONE_X_RESEARCH_SURVIVOR`
- operating cadence `cwcadence_f2c5d2dfae3b4cf3a1b9` publishes a 19-stage daily/conditional research route from storage preflight through the monitor dashboard, including OU Optimal outcome stratification immediately after walk-forward; executable plan-only run `cwdaily_e5683876c89061df154b` resolved the complete current `READY_TO_FETCH` history queue and then failed closed before running any stage because only `707,780,608` bytes (about 675 MiB) were free against the 3 GiB safety floor; all 19 stages remain blocked or not started, every scheduled stage has order authority disabled, Testnet execution is excluded from the schedule, and the separate live-lock artifact is permanently research-only
- `run-current-wizard-hyperliquid-daily-pipeline` makes that validated route executable. Its safe default is plan-only; `--execute-daily-pipeline` is required to run stages. It resolves every current `READY_TO_FETCH` pair key at runtime, rechecks the 3 GiB floor before each stage, stops on the first storage, input, or command failure, writes active and dated stage evidence, and contains no Testnet-order or live stage.
- canonical CSV, Markdown, JSON, optional Parquet, and daily-run artifacts now use same-directory atomic replacement: a complete temporary copy is flushed and synchronized before replacement, the parent directory is synchronized afterward where supported, and failed temporary writes are removed while the prior artifact remains untouched. Fault-injection tests prove both required CSV and optional Parquet evidence survive a partial `ENOSPC` write byte-for-byte. Removing only 529,332 KiB of generated Python `__pycache__` files restored local free space without removing research evidence, but the deep pipeline remains correctly blocked below 3 GiB.
- objective-level completion audit `cwcompletion_493a0bff64bd5c53ba60` evaluates 12 explicit requirements from hash-bound current manifests instead of inferring success from file existence: nine are `PROVEN`, leverage/margin and actual Testnet lifecycle coverage are `CONDITIONALLY_PROVEN` because the valid 1x survivor cohort is empty, zero are unproven, and the fresh executable 19-stage cycle is the sole `BLOCKED` requirement. Completion authority is false, order submission is false, and live trading remains unauthorized. Rebuild it with `build-current-wizard-hyperliquid-completion-audit` after every material chain change.
- downstream immutable snapshots now use hash-verified upstream references instead of recursively copying existing evidence: concentration, failure attribution, leverage, and learning reference about 156 MB of immutable inputs while physically copying only the roughly 0.33 MB of current external market, margin-tier, and pair-cost evidence required by leverage
- read-only reclamation plan `cwreclaim_875a88424b6d4f23f14c` protects 325 current and exhaustive lineage paths and identifies 18 complete superseded branches totaling 3,700,991,944 bytes; all 18 branches have deterministic manifest and tree SHA-256 evidence, no move or deletion was performed, archive authority remains false, no external archive volume is currently mounted, and a later verified off-volume archive of the listed branches would restore the 3 GiB operating floor
- configure `WIZARD_HYPERLIQUID_ARCHIVE_DESTINATION` or pass `--wizard-archive-destination /Volumes/<drive>/TheWizArchive` to inspect an existing destination; copy preflight becomes ready only when the directory is writable, resides on a different filesystem device, and has capacity for every safe candidate plus a 1 GiB reserve. This inspection never copies, moves, or deletes evidence, and release preflight remains disabled.
- after a destination passes inspection, `stage-current-wizard-hyperliquid-archive-copy` requires both the explicit destination and a one-run `--archive-copy-approval-id`. It copies only safe, hash-verified branches through unique temporary directories, re-hashes the complete source and destination trees, atomically promotes verified copies, supports idempotent retries, and writes local plus off-volume receipts. It never moves or deletes a source branch and cannot authorize source release, Testnet orders, or live trading.
- `plan-current-wizard-hyperliquid-archive-release` is a separate no-delete dry run. It re-verifies the active and off-volume receipts, every source and destination tree, current research-lineage references, filesystem separation, candidate-root isolation, and the projected 3 GiB local floor. A candidate that becomes active lineage is vetoed immediately. Even a fully passing plan reports only `READY_FOR_SEPARATE_APPROVAL`; release authority and deletion remain false.
- current production dry run `cwarchiverelease_8cbeadda4408b974ffad` is correctly `BLOCKED` with `verified_off_volume_archive_copy_missing`, zero release candidates, zero released bytes, and false source-release, deletion, order, and live authority.
- the read-only Testnet refresh at 2026-08-08 18:29 UTC verifies the configured master, approved agent, keychain key, SDK, local signature, and disabled submission switch. Current Testnet metadata contains 210 perpetual records: 157 tradable, 53 delisted, 19 isolated-only, and zero fetch-blocked. USD 995.075565 remains in Testnet spot USDC while perpetual account value is zero, so `testnet_usdc_requires_spot_to_perp_transfer` remains an account-readiness blocker and no transfer or order was attempted.
- deterministic Testnet protocol `cwtestnetprotocol_115162c859d8b169c949` passes all 9 required no-order scenarios and 23 state transitions, including stale and ineligible setup blocks, 1x-before-leverage enforcement, partial orphan flattening, duplicate-submit prevention, reduce-only exits, and restart reconciliation. Its content identity binds the frozen research manifests plus the exact simulator, executor, approval/lifecycle gate, and read-only exchange-evidence source hashes, so the protocol ID is regenerated whenever any bound safety code changes. It accounts for all zero current leverage candidates, explicitly records `simulation_is_testnet_proof=false`, and grants no execution, order, or live authority.
- the real Testnet lifecycle gate now requires a structured `hyperliquid-testnet-lifecycle-v3` receipt bound to the signed approval, run, candidate set, deterministic protocol ID, complete registered Stage 5 learning/protocol/execution lineage, ordered exchange references, recovery semantics, final flat account state, and receipt hash. Older receipts and the former six-boolean receipt format are rejected. Production remains `BLOCKED`: deterministic simulation passes, but no eligible pair, signed one-run approval, structured exchange receipt, or perpetual collateral exists.
- `build-hyperliquid-testnet-lifecycle-gate` is the canonical read-only gate rebuild. It loads the configured local environment through the normal CLI path, writes the active CSV and Markdown evidence, and never submits an order.
- `capture-hyperliquid-testnet-lifecycle-evidence` is a separate read-only exchange-evidence step. It cannot load the agent key or submit, cancel, or close an order. It joins the hash-valid executor state to Testnet `userFillsByTime`, clearinghouse state, and open orders; only two terminal fills can create an entry or exit event, and only a flat, order-free account can create reconciliation evidence. Partial fills are retained as anomalies instead of being promoted.
- the final Testnet executor now verifies the exact signed approval artifact instead of accepting any nonempty runtime ID. Approval schema `hyperliquid-testnet-smoke-v10` binds the requested leverage, cross or isolated margin mode, preserved 1x proof ID, accepted leverage-scenario ID, exact entry legs, reduce-only exit policy, maximum exit slippage, current candidate receipt, complete registered Stage 5 lineage, and current entry features to the run, candidate set, wallet identities, and expiry. Every signed approval ID also has an immutable identity receipt. Runtime market, side, size, entry-price, leverage, margin, approval-ID, identity-receipt, learning-lineage, candidate-freshness, or feature-freshness drift blocks entry. A journal-linked close retains only immutable reduce-only exit authority after entry freshness expires; it cannot change markets, reverse direction, enlarge size, or reopen risk.
- the executor now configures the approved leverage and margin mode on both perpetual markets before the single bulk entry. Leverage above 1x is blocked unless both the 1x Testnet proof and leverage-scenario evidence IDs are present. The production template is migrated to non-approved v3 at 1x cross; its signature and approval ID remain empty.
- the bounded pair envelope is USD 25 total, with at least USD 10 notional on each leg. This narrow headroom is necessary to satisfy Hyperliquid's USD 10 minimum perpetual order on both legs after lot-size rounding; it does not grant order authority or increase leverage. Immediately before wallet loading, the executor fetches current Testnet metadata and rejects missing or delisted markets, sizes or prices that violate `szDecimals` and tick rules, requested leverage above either market's maximum, and cross margin on isolated-only markets.
- a clean actual Testnet receipt must prove two-leg entry, two-leg exit, final reconciliation, and duplicate-submit blocking. Partial-fill recovery, cancel-and-unwind, and emergency flatten evidence become mandatory only when the receipt records the corresponding anomaly. The deterministic no-order protocol always exercises those failure branches; it cannot substitute for actual exchange evidence.
- partial, unconfirmed, and transport-unknown pair submissions no longer stop at manual recovery. The executor cancels known and discovered pair orders, queries clearinghouse state, closes only nonzero orphan positions through the SDK's reduce-only market-close path, and re-queries positions and open orders until terminal flat reconciliation is proven or a hard recovery blocker is recorded. It never retries the original pair entry.
- one-run approval consumption is enforced by the hash-bound execution journal at the actual executor boundary. Once an entry submission has been attempted, the same approval ID cannot submit that entry again, even after a reduce-only exit; a different approval cannot start while an earlier lifecycle is nonterminal. Entry and exit attempts are journaled separately.
- resting or merely acknowledged orders cannot satisfy lifecycle evidence. Both legs must carry terminal exchange fill status and distinct exchange references before either the entry or exit event can pass the structured receipt gate.
- the production adapter writes an atomic, hash-bound `hyperliquid_testnet_pair_execution_state.json` before submission and marks the attempt before the network call. `hyperliquid-testnet-recover-pair-state --order-approval-id <signed_one_run_approval_id>` can recover an interrupted Testnet attempt from this journal without replaying entry. No production execution-state file currently exists because no candidate was eligible and no order was attempted.
- lightweight monitor dashboard publishes 183 files, including the objective-level completion audit, read-only lifecycle evidence capture, and fail-closed daily-run status, and shows protocol simulation `PASS`, 9 of 9 scenarios, zero actual Testnet candidates, no order submission, and no live authority. System health is 150 of 157 checks ready; the seven remaining blockers are preserved rather than hidden.
- local OU ZScoreR is explicitly flagged as equivalent to local Static ZScoreR when a constant OU mean is removed by rolling standardization; the system does not count those duplicate outputs as independent evidence

- Coinbase Hourly DOT/ZRO captured across all seven pair-page modes and both verified asset orientations
- Binance Daily BTC/MOVE captured across all seven pair-page modes and both verified asset orientations
- Binance Daily ETH/FIL captured across all seven pair-page modes and both verified asset orientations
- Binance Daily BTC/NXPC captured across all seven pair-page modes and both verified asset orientations
- Binance Daily SOL/TURBO captured across all seven pair-page modes and both verified asset orientations
- Binance Daily BTC/LAYER captured across all seven pair-page modes and both verified asset orientations
- Binance Daily SOL/SUSHI captured across all seven pair-page modes and both verified asset orientations
- Binance Daily ADA/ONDO captured across all seven pair-page modes and both verified asset orientations
- Binance Daily ETH/MORPHO captured for both ETHUSDC and ETHUSDT scanner variants, each in both verified asset orientations
- Binance Daily SOL/WIF captured for both WIFUSDC and WIFUSDT scanner variants, each in both verified asset orientations
- Binance Daily DOGE/MAGIC captured across all seven pair-page modes and both verified asset orientations; research is complete while Hyperliquid execution remains blocked because MAGIC has no mapped Testnet perpetual
- Binance Daily DOGE/EDU, DOGE/SEI, and BTC/NEWT captured across all seven pair-page modes and both verified asset orientations; each is research-complete while retaining its point-in-time missing Testnet perpetual blocker
- Binance Daily DOGE/FLOW and BANK/NXPC captured across all seven pair-page modes and both verified asset orientations while retaining their FLOW and BANK execution blockers
- Binance Daily AWE/MAGIC captured across all seven pair-page modes and both verified asset orientations; research is complete while both AWE and MAGIC remain explicit missing-market blockers
- Binance Daily BTC/ERA and HOME/STG captured across all seven pair-page modes and both verified asset orientations while retaining their single-leg and dual-leg execution blockers
- Binance Daily ETH/RPL captured for both ETHUSDC and ETHUSDT scanner variants, each in both verified asset orientations, while retaining its RPL execution blocker
- Binance Daily DOGE/RPL captured across all seven pair-page modes and both verified asset orientations while retaining its RPL execution blocker
- Binance Daily BCH/NXPC and BTC/LSK captured across all seven pair-page modes and both verified asset orientations while retaining their BCH and LSK execution blockers
- Binance Daily AWE/BNT and AWE/PEOPLE captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg and AWE-only execution blockers
- Binance Daily STO/VIC captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market execution blockers
- Binance Daily CTK/LUMIA and B2/HBAR captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg and B2-only execution blockers
- Binance Hourly BTC/MEGA, KAITO/ORDI, and BTC/MOVE captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance Hourly ETH/SUI captured for all four observed ETHUSDC/ETHUSDT and SUIUSDC/SUIUSDT scanner combinations, each in both verified asset orientations
- Binance Hourly BTC/STRK captured across all seven pair-page modes and both verified asset orientations while retaining its STRK execution blocker
- Binance Hourly SKHYNIX/WDC, DEXE/RVN, and MYX/TENCENT captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg execution blockers
- Binance Hourly ACE/GUN and CFG/MOVE captured across all seven pair-page modes and both verified asset orientations while retaining their GUN-only and CFG-only execution blockers
- Binance Hourly 1000000MOG/SLX captured across all seven pair-page modes and both verified asset orientations while retaining the exact multiplier-prefixed asset identity and both execution blockers
- Binance Hourly ALLO/MITO and CL/STO captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg execution blockers
- Binance Hourly AERO/SKHY captured across all seven pair-page modes and both verified asset orientations while retaining only its SKHY execution blocker
- Binance Hourly AAVE/ONT captured across all seven pair-page modes and both verified asset orientations while retaining only its ONT execution blocker
- Binance Hourly SKYAI/ZEREBRO and ALAB/BOME captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg execution blockers
- Binance Hourly BANK/TAC captured across all seven pair-page modes and both verified asset orientations while retaining its dual-leg execution blockers
- Binance Hourly BTC/CARV and BTC/USELESS captured across all seven pair-page modes and both verified asset orientations while retaining their CARV-only and USELESS-only execution blockers; BTC/USELESS preserves its exact BTCUSDC scanner market
- Binance Hourly AKT/CYS captured across all seven pair-page modes and both verified asset orientations while retaining its dual-leg execution blockers
- Binance Hourly GRIFFAIN/OPEN and BTC/SYN captured across all seven pair-page modes and both verified asset orientations while retaining only their OPEN and SYN execution blockers
- Binance Hourly BTC/GUN and BLESS/HOME captured across all seven pair-page modes and both verified asset orientations while retaining their GUN-only and dual-leg execution blockers
- Binance US Daily BNB/ME captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance US Daily AVAX/SUI and BNB/SUI captured for both observed SUIUSD and SUIUSDT scanner variants, each in both verified asset orientations, with both Hyperliquid legs mapped
- Binance US Daily BNB/ETH captured for both observed BNBUSD and BNBUSDT scanner variants against the exact ETHBTC scanner market, each in both verified asset orientations, with both Hyperliquid legs mapped
- Binance US Daily BNB/FIL and BNB/NEO captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance US Daily SOL/SUI captured for all five observed SOLUSD/SOLUSDC/SOLUSDT and SUIUSD/SUIUSDT scanner combinations, each in both verified asset orientations, with both Hyperliquid legs mapped
- Binance US Daily FORTH/NEAR captured across all seven pair-page modes and both verified asset orientations while retaining its FORTH execution blocker
- Binance US Daily BCH/BTC captured for all three observed BTCUSD, BTCUSDC, and BTCUSDT scanner variants, each in both verified asset orientations, while retaining its BCH execution blocker
- Binance US Daily ETC/LAZIO, BNB/PEPE, BNB/KNC, PAXG/ZEC, and LINK/SOL captured across all seven pair-page modes and both verified asset orientations while retaining their LAZIO, PEPE, KNC, ZEC, and LINK execution blockers
- Binance US Daily BNB/GRT captured for the two exact observed BNBUSD/GRTUSD and BNBUSDT/GRTUSDT scanner combinations, each in both verified asset orientations, while retaining its GRT execution blocker
- Binance US Daily SOL/VTHO captured across all seven pair-page modes and both verified asset orientations while preserving the exact SOLBTC scanner market and retaining its VTHO execution blocker
- Binance US Hourly BTC/CELO captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance US Hourly DOGE/SUI captured for the three exact observed DOGEUSDT/SUIUSD, DOGEUSDT/SUIUSDT, and DOGEUSD/SUIUSDT scanner combinations, each in both verified asset orientations, with both Hyperliquid legs mapped
- Binance US Hourly BTC/WLFI and KAITO/LDO captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance US Hourly AVAX/SUI and ETH/SUSHI captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Binance US Hourly PUMP/SYS, IMX/TROLL, POL/REQ, AVAX/DUSK, and BONK/WLFI captured across all seven pair-page modes and both verified asset orientations while retaining their SYS, TROLL, REQ, DUSK, and BONK execution blockers
- Binance US Hourly ADA/SHIB captured for the two exact observed ADAUSD/SHIBUSD and ADAUSD/SHIBUSDT scanner combinations, each in both verified asset orientations, while retaining its SHIB execution blocker
- Binance US Hourly AVAX/ZEC, AUDIO/HBAR, TROLL/XLM, and CTSI/FARTCOIN captured across all seven pair-page modes and both verified asset orientations while retaining their ZEC, AUDIO, TROLL, and CTSI execution blockers
- Binance US Hourly COTI/FORTH and BONK/ENJ captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market blockers for each pair
- Binance US Hourly LDO/TLM, EIGEN/PORTO, and ETH/THETA captured across all seven pair-page modes and both verified asset orientations while retaining their TLM, PORTO, and THETA execution blockers
- Binance US Hourly ADA/PEPE captured for both exact ADAUSD/PEPEUSDT and ADAUSD/PEPEUSD scanner variants, each in both verified asset orientations, while retaining its PEPE execution blocker
- Bybit Daily BNB/ONDO, AVAX/BIGTIME, DYM/IO, ADA/IO, ETH/MORPHO, LDO/MORPHO, and DOGE/NXPC captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Bybit Daily ADA/ONDO captured for both exact ADAUSDT/ONDOPERP and ADAUSDT/ONDOUSDT scanner variants, each in both verified asset orientations, with both Hyperliquid legs mapped
- Bybit Daily AVAX/ENS captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Bybit Daily BNB/MOVR, AVAX/MAGIC, AWE/ENS, DBR/DYDX, MNT/NOT, BNB/CARV, and LINK/MORPHO captured across all seven pair-page modes and both verified asset orientations while retaining their MOVR, MAGIC, AWE, DBR, MNT, CARV, and LINK execution blockers
- Bybit Daily AWE/BNT captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market execution blockers
- Bybit Daily AVAX/CETUS, BNB/GRT, BNB/KNC, BNB/KSM, DOGE/NEWT, BTC/LSK, and AVAX/CGPT captured across all seven pair-page modes and both verified asset orientations while retaining their CETUS, GRT, KNC, KSM, NEWT, LSK, and CGPT execution blockers
- Bybit Daily 1000LUNC/ENJ and DOT/ENJ captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market blockers for each pair
- Bybit Daily 1000PEPE/IOTA captured across all seven pair-page modes and both verified asset orientations while preserving the exact multiplier-prefixed asset identity and retaining its 1000PEPE execution blocker
- Bybit Hourly DOGE/EIGEN, AERO/KAITO, and CC/LDO captured across all seven pair-page modes and both verified asset orientations with both Hyperliquid legs mapped
- Bybit Hourly MANTA/MVLL and AXTI/IBM captured across all seven pair-page modes and both verified asset orientations while retaining their MVLL-only and dual-leg execution blockers
- Bybit Hourly AAOI/AXTI, FLY/H, BICO/ELSA, and 1000TAG/EWJ captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market blockers for each pair
- Bybit Hourly APE/COHR, ADA/AKT, BLESS/HYPER, ATOM/DRIFT, CC/ETHFI, and ACE/GUN captured across all seven pair-page modes and both verified asset orientations while retaining their COHR, AKT, BLESS, DRIFT, ETHFI, and GUN execution blockers
- Bybit Hourly CC/ETHFI captured for both exact CCUSDT/ETHFIPERP and CCUSDT/ETHFIUSDT scanner variants, each in both verified asset orientations
- Bybit Hourly DRAM/ETHUSDT is terminally accounted as route unavailable: the authenticated Custom Analysis form returned `No data found for 'ETHUSDT-07AUG26'`; all 16 planned cells retain this evidence-backed blocker without substituting generic ETH
- Bybit Hourly CBRS/FLNC and CRDO/KAIA captured across all seven pair-page modes and both verified asset orientations while retaining both missing-market blockers for each pair
- Bybit Hourly AKT/AVNT, DOGE/ETHFI, and DYDX/ERA captured across all seven pair-page modes and both verified asset orientations while retaining their AKT, ETHFI, and ERA execution blockers
- Bybit Hourly BTCUSDT/GUN captured for the exact dated `BTCUSDT-25SEP26` contract and `GUNUSDT` in both verified asset orientations; the original route exposes six conditional charts while the reverse route explicitly records that conditional charts are unavailable
- Bybit Hourly HEI/MYX and ESPORTS/EUL captured across all seven pair-page modes and both verified asset orientations while retaining their dual-leg Hyperliquid Testnet mapping blockers
- Coinbase Daily ADA/RENDER and ICP/PYTH captured from the exact `RENDER-USD`/`ADA-USDT` and `ICP-BTC`/`PYTH-USD` scanner markets in both verified asset orientations, with both Hyperliquid legs mapped
- Coinbase Daily ALGO/PUMP captured from all three exact `PUMP-USD` pairings with `ALGO-GBP`, `ALGO-EUR`, and `ALGO-USD`; all six original/reverse route bundles are preserved, with both Hyperliquid legs mapped
- Coinbase Daily PUMP/TURBO and WIF/ZRO captured from the exact scanner markets in both verified asset orientations, with both Hyperliquid legs mapped
- Coinbase Daily PUMP/WIF, PUMP/AERO, ENS/MORPHO, DOGE/MORPHO, and ADA/AERO captured from their exact USD and USDT scanner markets in both verified asset orientations, with both Hyperliquid legs mapped
- Coinbase Daily PUMP/MAGIC, FARM/ATOM, FLOKI/RENDER, CRV/MORPHO, ADA/GNO, and DASH/ZEC captured from their exact scanner markets in both verified asset orientations while retaining their MAGIC, FARM, FLOKI, CRV, GNO, and ZEC execution blockers
- FARM/ATOM preserves the exact mixed-quote `FARM-USD`/`ATOM-BTC` source route, while ADA/GNO preserves `ADA-EUR`/`GNO-USD`; neither route is rewritten to a convenient USD-only dashboard hypothesis
- Coinbase Daily BAT/XRP, DOGE/GIGA, ENA/TIA, FET/MSOL, AMP/WIF, GNO/TURBO, EIGEN/FLOKI, DOGE/SHIB, ATH/IO, and BONK/DOGE captured from their exact USD, USDT, EUR, and GBP scanner markets in both verified asset orientations while retaining every current missing-market execution blocker
- Coinbase Daily CRO/AGLD is terminally accounted as route unavailable: the authenticated Custom Analysis form returned `No data found for 'CRO-USDT'`; all 16 planned cells retain that exact evidence without substituting another CRO market
- Coinbase Hourly LDO/NEAR, FIL/SUI, RSR/STX, AXS/ZRO, FIL/JTO, and DOGE/SUI captured from their exact hourly scanner markets in both verified asset orientations, with both Hyperliquid legs mapped
- Coinbase Hourly POL/ZRO captured from the exact `ZRO-USD`/`POL-USD` route in both verified asset orientations, with both Hyperliquid legs mapped
- Coinbase Hourly ICNT/KAITO, HONEY/AAVE, VVV/ETHFI, GWEI/ALLO, BONK/PYTH, DRIFT/ATOM, BCH/FET, SHIB/MOODENG, EIGEN/AKT, CRO/AAVE, PEPE/ADA, FLR/STRK, ETH/ETHFI, TNSR/ALLO, PAXG/DOT, USDC/ATOM, and DRIFT/AKT captured from their exact scanner markets in both verified asset orientations while retaining every current single-leg or dual-leg execution blocker
- the exact `CRO-USD`/`AAVE-USD` Hourly route is available and fully captured even though the separate `CRO-USDT`/`AGLD-USD` Daily route is unavailable, proving that route status is market- and timeframe-specific rather than an asset-wide assumption
- USDC/ATOM preserves the exact mixed-quote `USDC-EUR`/`ATOM-GBP` Hourly route in both orientations without substituting USD markets
- Coinbase Hourly BASED1/EIGEN and DOT/W captured from the exact `BASED1-USD`/`EIGEN-USD` and `DOT-USD`/`W-USD` routes in both verified asset orientations while retaining their current BASED1 and DOT execution blockers
- dYdX Daily AVAX/JUP, AXS/JTO, DYM/TAO, SNX/W, SOL/XMR, NEAR/TAO, ETH/JTO, BNB/IO, BERA/TAO, ATOM/INJ, DYM/TIA, ADA/EIGEN, ADA/JUP, HYPE/RENDER, and DYM/EIGEN captured from their exact dashed-symbol markets in both verified asset orientations, with both Hyperliquid legs mapped
- dYdX Daily BTC/ZRO and BCH/BLAST captured from their exact dashed-symbol markets across all seven pair-page modes and both verified asset orientations; BTC/ZRO maps to both Hyperliquid legs while BCH/BLAST retains its current missing-market blocker
- dYdX Daily EUR/FIL has a complete original-orientation bundle from the exact `EUR-USD`/`FIL-USD` route; its real reverse `FIL-USD`/`EUR-USD` route remains pending after a browser-control interruption and is not misclassified as route unavailable
- the authenticated dYdX Custom Analysis form enforces a maximum period of 360; all dYdX Daily route creation now uses 360 explicitly instead of the 365-period default used for other venues
- pair-route orientation proof now distinguishes delayed rendering from invalid identity: exact inputs and rendered asset labels may settle through a bounded proof loop, transient chart controls may retry the same proven route once, and genuine no-data responses remain terminal blockers
- dated-contract orientation validation now uses exact raw scanner symbols, preventing `BTCUSDT-25SEP26` from being incorrectly reinterpreted as generic BTC during verification
- every reverse orientation uses a separate Custom Analysis route whose inputs and rendered asset X/Y labels prove the recalculation
- six conditional chart views, six dependency views, and 21 backtest chart views preserved
- optional conditional charts are explicitly marked unavailable when the current pair/timeframe page does not expose them
- OU Optimal explicitly accounted as unavailable on the current pair page
- 3,408 planned mode/orientation cells accounted across 213 retained pair groups, with no coverage failures
- 2,947 supported cells captured, 422 pair-page mode cells unavailable, 32 route-unavailable cells, and seven supported reverse cells pending
- 209 of 247 active-queue pair groups are captured complete, two additional groups are terminally route-unavailable, EUR/FIL is original-only, and 35 remain untouched; BTC/MOVE is retained as separate complete historical evidence from the prior board snapshot
- the next unresolved work item is the real dYdX Daily EUR/FIL reverse route, followed by dYdX Daily ATH/FIL; neither may be substituted, silently dropped, or promoted past remaining gates
- ETH/MORPHO preserves all four ETHUSDC/ETHUSDT original/reverse route bundles; selected cells retain candidate counts and every superseded evidence path
- SOL/WIF preserves all four WIFUSDC/WIFUSDT original/reverse route bundles under the same candidate-lineage rule
- ETH/RPL preserves all four ETHUSDC/ETHUSDT original/reverse route bundles under the same candidate-lineage rule
- ETH/SUI preserves all eight original/reverse route bundles across its four exact quote-market combinations under the same candidate-lineage rule
- AVAX/SUI and BNB/SUI each preserve all four original/reverse route bundles across their SUIUSD and SUIUSDT scanner-market variants under the same candidate-lineage rule
- BNB/ETH preserves all four original/reverse route bundles across its BNBUSD and BNBUSDT scanner-market variants against ETHBTC under the same candidate-lineage rule
- SOL/SUI preserves all ten original/reverse route bundles across its five exact quote-market combinations under the same candidate-lineage rule
- BCH/BTC preserves all six original/reverse route bundles across its BTCUSD, BTCUSDC, and BTCUSDT scanner-market variants under the same candidate-lineage rule
- BNB/GRT preserves all four original/reverse route bundles for its two observed same-quote scanner combinations without inventing unobserved cross-quote combinations
- DOGE/SUI preserves all six original/reverse route bundles for its three observed scanner-market combinations without inventing the unobserved DOGEUSD/SUIUSD combination
- ADA/SHIB preserves all four original/reverse route bundles for its two observed scanner-market combinations under the same candidate-lineage rule
- ADA/PEPE preserves all four original/reverse route bundles for its two observed PEPEUSD and PEPEUSDT scanner-market variants under the same candidate-lineage rule
- ADA/ONDO preserves all four original/reverse route bundles for its two observed ONDOPERP and ONDOUSDT scanner-market variants under the same candidate-lineage rule
- CC/ETHFI preserves all four original/reverse route bundles for its two observed ETHFIPERP and ETHFIUSDT scanner-market variants under the same candidate-lineage rule
- ALGO/PUMP preserves all six original/reverse route bundles across its three exact GBP, EUR, and USD ALGO quote variants; every selected cell records six capture candidates and up to five superseded evidence paths
- ETH/FIL and BTC/NXPC original and reverse results differ across asymmetric modes, including rolling z-score and spread variants, proving that the second route is a recalculation rather than a label swap
- Wizard costs recorded as current client defaults but blocked from local parity until their charging semantics are verified
- configured Wizard API authentication is working as of 2026-08-08; the scanner sweep and bounded API schema pilot are complete, while exhaustive pair-detail parity still requires authenticated dashboard/UI capture for ECM and dashboard-only settings and remains blocked where Chrome content reads time out

### 4. Hyperliquid Mapping And Market Evidence

Status: complete for the 2026-08-08 05:01 UTC point-in-time refresh; recurring refresh remains required before replay or Testnet use.

Normalize both assets, resolve each to exactly one Hyperliquid testnet perpetual, and collect point-in-time market metadata, candles, order-book depth, funding, size and price increments, leverage limits, margin restrictions, and freshness. Ambiguous or missing legs remain blocked and visible.

Current checkpoint:

- 81 pair groups map to both Hyperliquid legs
- 166 pair groups are retained with explicit mapping blockers
- 3,952 exact-mode/orientation experiments remain in the exhaustive matrix
- immutable mapping refresh `hlmap_5bdc15114d2589b20f46` links all 247 pair groups to the frozen Wizard run without rewriting its original inventory
- 157 tradable Testnet perpetuals remain available; no market additions, removals, leverage-limit changes, isolated-margin changes, or pair-mapping drift occurred versus the frozen inventory
- every refreshed row preserves current maximum leverage, isolated-only requirements, blockers, both inventory evidence paths, and `live_trading_authorized=false`
- the Testnet inventory now also persists every referenced margin table and tier with notional bounds, tier maximum leverage, official maintenance-margin rate `1 / (2 * max_leverage)`, and recursive maintenance deduction; table IDs below 50 are represented as their documented single tier instead of being treated as missing

### 5. Canonical Local Replays

Status: point-in-time history, canonical 1x replay, exhaustive funding evidence, 103-pair rolling L2 calibration, pair-level cost bridging, observed-cost 1x research replay, purged walk-forward, causal regime attribution, robustness testing, cross-cell concentration, and the leverage/margin engine are complete. Exact-mode parity for uncaptured Wizard pair pages, a non-empty statistically selected cohort, candidate leverage scenarios, and acceptance remain pending.

Build synchronized point-in-time histories and replay every applicable pair, timeframe, exact mode, and orientation at 1x gross portfolio leverage, where:

```text
gross_portfolio_leverage = (absolute_notional_leg_x + absolute_notional_leg_y) / account_equity
```

Include fees on both legs, observed slippage assumptions, funding on both legs, latency sensitivity, missing bars, delistings, hedge-ratio drift, leg-notional imbalance, and no-lookahead parameter fitting. The canonical replay remains the unambiguous comparison baseline for every later leverage scenario.

Required comparisons include in-sample diagnostics, purged walk-forward folds, regime slices, parameter perturbations, cost stress, and failure attribution.

Current checkpoint:

- immutable replay preflight `hlreplay_ff90fb05f7622471b3d1` accounts for all 3,952 experiment IDs with no discovery prefilter and no identity collapse
- 980 captured, verified, supported mode/orientation cells across 70 pair groups are ready for point-in-time history
- 140 OU Optimal cells are explicitly not applicable because the current pair page does not expose that mode; 176 cells remain pending pair-detail capture
- 2,656 cells retain current Hyperliquid mapping blockers, while zero Testnet-ready cells are missing a corresponding tradable Hyperliquid mainnet history market
- 247 pair-history work items remain visible; only the 68 asset/interval network fetches are deduplicated
- all history requests end at the earliest scanner-capture timestamp for their pair group, preventing post-snapshot candles from leaking into the dashboard-comparison replay
- Wizard operator enums `Gte`, `Lte`, `Gt`, `Lt`, and `Eq` now map directly into the local threshold engine without relabeling or silently defaulting an invalid rule
- immutable history run `hlhistory_20260808T052353614209Z_209cb198` accounts for all 68 deduplicated asset/timeframe requests and all 247 pair-history work items
- 59 asset histories completed and nine daily assets remain explicitly blocked for having only 394 to 743 candles against the 750-row minimum; there were no transport failures, malformed timestamp histories, or post-snapshot candles
- 53 pair groups have synchronized point-in-time histories ready for replay; the other 17 otherwise mapped pair groups remain visible with the dependent short-history blocker
- immutable canonical replay `hlcanonical_20260808T054413287381Z_1057b30c` accounts for all 3,952 experiment IDs and writes 33,450 typed backtest trade-ledger rows
- 529 Static, OU, and causal local-Copula cells completed at 1x; 238 cells retain short-history blockers, 212 dynamic cells are blocked because the dashboard did not expose a reproducible point-in-time dynamic hedge-exposure rule, one reverse Copula cell is blocked for a missing captured family, and zero replay calculations failed
- the local Copula fallback uses only trailing returns at each candle to build rolling Gaussian conditional probabilities; it is marked as a local approximation and never presented as parity with the captured Wizard copula family
- only 53 completed cells receive a numbered research rank after requiring at least 10 closed trades, finite profit factor, valid Sharpe, positive closed-trade expectancy, and positive full-path after-cost return; all other completed cells remain in the ranking file with explicit rank blockers
- canonical costs are still labeled `PROVISIONAL_CONSERVATIVE_DEFAULTS`: 5 bps taker fee, 4 bps slippage, 2 bps execution-risk allowance, 1 bp daily funding drag, and an expected partial-fill penalty; observed Hyperliquid funding, depth-based slippage, and latency calibration are required before any acceptance decision
- every canonical result remains `acceptance_status=BLOCKED`, `acceptance_eligible=false`, and `live_trading_authorized=false`
- immutable funding evidence `hlfunding_f2569b7727e484be3788` completed all 46 deduplicated asset requests with zero post-cutoff rows, zero timestamp failures, zero retries, and zero canonical-history hash changes
- funding coverage accounts for all 247 pair groups: 43 meet the 95% full-history threshold, 10 retain explicit partial-coverage blockers, and 194 retain their earlier pair-history blockers
- funding is stored as signed realized basis points per aligned candle; the backtest engine no longer divides already-realized hourly funding by 24, and reverse orientation swaps both price and funding legs
- the L2 collector, pair-cost model, cadence report, research-cycle defaults, and CLI now share one canonical rule: 12 complete samples per leg at a ten-minute cadence inside a rolling two-hour window; future rows are excluded from historical `as_of` builds, old rows roll out without creating a permanent expiration state, and stale model/window mismatches become `model_rebuild_due`
- the public L2 sweep accounts for all 103 currently routable unique Hyperliquid pairs from the exhaustive inventory; after the 2026-08-08 10:02 UTC capture, all 103 have 12 complete samples per leg inside the rolling two-hour window
- the isolated 06:33 UTC observation rolled out at the prior interval, so only current rolling evidence contributes to these counts
- all 103 rolling sample counts reconcile exactly between raw evidence and the active pair-cost model; the cadence reports 103 calibrated, zero waiting, zero rebuild-due, zero infeasible, and zero expired rows at the calibration checkpoint
- immutable cost bridge `hlcost_78d1de5e31af23f99178` reconciles all 247 pair groups and all 3,952 experiments, with 43 pair groups ready for calibrated replay and 51 ready for provisional observed-cost research
- provisional readiness requires at least 250 rows in one contiguous two-leg observed-funding segment, a fresh official fee profile, at least one complete L2 sample per leg, and explicit pair-leg slippage estimates; partial or missing evidence never becomes a silent zero
- immutable observed-cost replay `hlobserved_20260808T073735129590Z_9a85f0da` accounts for all 3,952 experiments, completes 509 Static, OU, and Copula cells, writes 8,015 typed trade-ledger rows, preserves the 204 dynamic-rule blockers, and retains the one known reverse-Copula family blocker with zero replay errors
- observed-cost replays use signed realized funding, official taker fees, pair-leg L2 p95 slippage weighted by actual hedge exposure, the retained execution-risk allowance, and the longest contiguous funding segment selected only by data availability
- only 29 observed-cost cells receive a numbered research rank after the prior trade and performance checks plus a minimum span of 365 daily bars or 720 hourly bars; short 316-hour segments remain visible but cannot produce misleading annualized research ranks
- the current top research-only cells are TAO/NEAR reverse OU ZScoreR, ZRO/WIF reverse Copula, TAO/DYM reverse Static ZScoreR, AVAX/ENS original Copula, and DYM/TIA original Static ZScoreR
- every observed-cost result remains `acceptance_status=BLOCKED`, `acceptance_eligible=false`, and `live_trading_authorized=false`; current L2 p95 estimates are calibration evidence, not historical depth or execution authority
- immutable purged walk-forward run `hlwalk_20260808T073830405157Z_8b3763c1` accounts for all 3,952 experiments and evaluates every completed observed-cost replay whose history can support the declared folds; full-sample research rank is a label only and is not used to select fold execution
- 300 Static, OU, and Copula cells completed all five expanding folds, producing 1,500 fold rows, 2,678 closed-trade rows, and 116,840 bar-ledger rows; all folds use a 20-bar embargo, training-only execution hedge-ratio fitting for every mode, training-only OU fitting, a flat start with inherited positions suppressed, and a forced flat end
- 209 otherwise completed replay cells remain explicitly blocked because their 273- or 316-bar observed-funding histories cannot support the 180-row training floor plus three 20-bar embargoes and 30-row tests
- 28 cells pass the current practical research walk-forward gate and 272 fail it; only seven of the 29 full-sample primary candidates pass, demonstrating why full-sample rank is not acceptance evidence
- every candidate also receives a one-sided fold-return p-value, Benjamini-Hochberg q-value across the full 300-cell family, and fitted hedge-ratio coefficient-of-variation check; zero candidates currently pass the 10% false-discovery selection gate, so there is no statistically selected promotion cohort
- active and immutable walk-forward artifacts have byte-identical hashes, fold IDs are unique, fold date order is valid, fit leakage flags are false for all 1,500 folds, aggregate returns reconcile to fold compounding, and no fold ends with an open trade
- immutable causal regime attribution `hlregime_20260808T074456253998Z_9872b0fb` accounts for all 3,952 experiments and attributes all 28 practical walk-forward passes using trailing 20-bar pair volatility, trailing 60-bar return correlation, and prior-only expanding thresholds shifted by one bar
- the regime layer writes 11,590 enriched bar rows, 609 enriched trade rows, and 87 candidate/regime detail rows with zero unmatched timestamps, zero future-data flags, exact trade-count reconciliation, and byte-identical active/snapshot artifacts
- DYM/TIA original Copula is the only cell that passes the current research regime-stability diagnostic, but it remains blocked because its family-wide statistical-selection status is blocked; regime diagnostics cannot override false-discovery control
- immutable robustness run `hlrobust_20260808T075056228614Z_68f9a56d` accounts for all 3,952 experiments and subjects all 28 practical walk-forward passes to 11 scenarios over the same five-fold path: baseline, entry thresholds +/-10%, windows +/-20%, fitted hedge ratios +/-10%, 1.5x slippage, +2 bps taker fee, 1.25x aggregate trading costs, and 1.5x adverse absolute funding
- the robustness matrix contains exactly 308 candidate/scenario rows and 1,540 scenario/fold rows with unique identities, zero test-data fit flags, byte-identical active/snapshot artifacts, and baseline return reconciliation within `2.22e-16`
- 19 cells pass the declared research robustness diagnostics and nine fail; adverse absolute funding and parameter sensitivity explain the failures, while zero cells are ready for the next promotion gate because family-wide statistical selection still has zero passes
- every walk-forward result remains research-only with `acceptance_status=BLOCKED`, `acceptance_eligible=false`, and `live_trading_authorized=false`; L2 calibration, exact-mode parity, a statistically selected and sufficiently broad cohort, candidate leverage scenarios, and Testnet lifecycle proof remain required

### 6. Acceptance And Ranking

Status: calibrated cost replay, purged expanding walk-forward, causal regime attribution, parameter perturbation, cost stress, family-wide false-discovery control, and cross-cell concentration are complete. A non-empty statistically selected promotion cohort remains pending.

Rank only after local evidence exists. Keep discovery metrics separate from acceptance metrics. Report trades, after-cost return, profit factor, Sharpe, drawdown, expectancy, turnover, capacity, stability, regime behavior, rejection reason, and evidence path.

Current concentration checkpoint:

- immutable concentration run `hlconcentration_20260808T104841496693Z_e49b39a9` accounts for all 3,952 experiments and separates the 28 practical walk-forward passes, 19 research robustness passes, and zero statistically selected robustness passes
- six declared dimensions are measured with positive after-cost profit share, closed-trade share, HHI, and effective breadth: canonical pair, Wizard timeframe, Wizard exchange, exact mode, orientation, and causal regime
- the 19-candidate robustness cohort fails breadth because all survivors are Daily, BNB/FIL contributes 37.6% of positive profit against a 35% cap, and Copula contributes 82.4% of trades against a 70% cap
- the practical cohort passes pair, venue, mode, orientation, and regime diagnostics but still fails timeframe breadth because no Hourly candidate survives the practical walk-forward gate
- the statistically selected cohort is empty, so `promotion_concentration_pass=false`, `ready_for_leverage_gate=false`, `acceptance_eligible=false`, and `live_trading_authorized=false`

### 7. Leverage And Margin Surface

Status: the leverage/margin scenario engine is implemented, tested, and exercised against the current immutable evidence chain. Candidate scenarios remain blocked until a canonical 1x replay passes the statistical, cost, stability, concentration, and drawdown gates.

Evaluate leverage separately across allowed Hyperliquid levels and margin modes. The scenario grid starts at 1x and may include 1.5x, 2x, 3x, and 5x, but each scenario is capped by the stricter leg-level limit, configured portfolio limit, minimum liquidation buffer, and current market rules. Record the effective leverage rather than only the requested leverage.

For every scenario:

- keep isolated-margin and cross-margin results separate;
- recalculate fees, funding, maintenance margin, liquidation distance, and return on equity;
- shock both mark prices, basis, correlation, hedge ratio, volatility, funding, and liquidity;
- simulate partial fills, delayed second legs, orphan legs, rejected amendments, and forced flattening;
- reject configurations that improve headline return while worsening risk-adjusted acceptance, ruin probability, or operational recoverability;
- preserve the 1x result as the authority for statistical edge.

Leverage changes capital efficiency and loss geometry, not statistical edge.

The leverage graduation contract is deterministic:

1. `1x_RESEARCH_ACCEPTED`: the unlevered configuration passes point-in-time, after-cost, walk-forward, false-discovery, regime, robustness, concentration, and drawdown gates;
2. `1x_TESTNET_PROVEN`: the same configuration completes a bounded two-leg Testnet entry, monitoring, exit, and reconciliation lifecycle at 1x;
3. `LEVERAGE_SCENARIO_ACCEPTED`: one requested leverage and margin-mode scenario independently passes current market limits, maintenance margin, liquidation-buffer, funding, liquidity, basis, partial-fill, and orphan-leg stress;
4. `LEVERAGE_TESTNET_PROVEN`: that exact scenario completes its own bounded Testnet lifecycle without unresolved reconciliation or safety exceptions;
5. `LIVE_DISABLED`: even a proven Testnet scenario remains ineligible for live submission until a separate explicit release decision.

Graduation is sequential. A candidate may advance only one leverage level at a time, and a failed level falls back to the highest previously proven level. The system must preserve the failed scenario and its reason; it must never rewrite the 1x result or search for a larger leverage value to hide a failure.

Current leverage checkpoint:

- immutable leverage run `hlleverage_20260808T104841496693Z_6ceb1c0f` accounts for all 3,952 experiment IDs without changing their canonical 1x signals
- the engine evaluates requested gross leverage of 1x, 1.5x, 2x, 3x, and 5x across reference equity of USD 1,000, USD 10,000, and USD 50,000; cross and isolated margin; current pair-level leverage caps; official maintenance tiers; and eight baseline/stress paths
- 1.5x uses 1.5x effective gross sizing with the exchange leverage setting rounded up to the required integer, so requested sizing and exchange configuration remain separately auditable
- all 3,952 cells are `NOT_SELECTED_PRIOR_CONCENTRATION_GATE`; zero candidate surfaces and zero scenario rows were created because the statistically selected concentration cohort is empty
- current market and margin-tier evidence was fresh at build time; `testnet_1x_lifecycle_ready=0`, `acceptance_eligible_replays=0`, and `live_trading_authorized=false`

### 8. Hyperliquid Testnet Lifecycle

Status: the 2026-08-08 10:11 UTC no-order preflight verified the Testnet URL, master and dedicated agent identities, keychain access, SDK, agent-key/address match, master authorization, and local signing. The refreshed inventory contains 157 tradable Testnet perpetual markets. Order submission remains disabled. The matching read-only margin snapshot found USD 995.075565 of Testnet spot USDC, zero perpetual account value, zero open positions, and the explicit blocker `testnet_usdc_requires_spot_to_perp_transfer`.

Before any real Testnet lifecycle, run `validate-current-wizard-hyperliquid-testnet-protocol`. This deterministic no-order simulator exercises stale-signal rejection, acceptance and collateral blocks, 1x-before-leverage enforcement, complete two-leg entry and reduce-only exit, partial orphan recovery, unconfirmed-response reconciliation without duplicate submission, and restart recovery. A passing simulation is only a software-control prerequisite: `simulation_is_testnet_proof=false`, actual Testnet proof remains zero, and order and live authority remain false. The existing signed one-run approval, actual exchange receipts, reconciliation evidence, perpetual collateral, and eligible-candidate gates remain independent requirements.

For accepted configurations, prove market lookup, precision rounding, minimum notional, selected leverage, selected margin mode, available margin, liquidation buffer, two-leg submission, partial fills, cancellation, retry, stale signal rejection, orphan-leg flattening, reduce-only exits, restart recovery, reconciliation, and complete audit logging. Testnet validation begins at 1x. A higher-leverage scenario can run only after its own leverage-surface acceptance record exists.

Testnet success does not prove profitability and does not authorize live trading; it proves only that the selected configuration can complete and recover from the required execution lifecycle safely in the simulated venue.

The spot-to-perpetual transfer is an account-readiness task, not a reason to bypass the research gates. Even after the transfer, no order can be submitted until a configuration passes the 1x evidence chain, the leverage surface marks it ready, and a separate runtime-only two-leg approval is present.

Stage 6 sample policy v4 closes execution-attempt selection bias. Every actual pair-executor invocation is bound to its immutable preflight, one-use reservation, candidate identity, model artifact, and registered Stage 5 lineage. A submitted attempt must resolve to either a validated lifecycle archive or an explicit terminal failure; unresolved submissions block release. Terminal failures remain in the execution-failure-rate denominator instead of disappearing because they could not satisfy the successful-lifecycle archive gate.

Read-only lifecycle capture also preserves pair-level failure evidence. A one-leg-filled and one-leg-missing outcome is recorded as an orphan-position anomaly even when only one exchange order reference exists. Any proven cancel-and-unwind, partial-fill recovery, or emergency reduce-only flatten is recorded from the hash-valid executor state. Recovery evidence never manufactures a two-leg entry or exit and never converts an unsuccessful lifecycle into a valid sample.

### 9. Learning System

Status: complete for dated, aggregate research outcomes; trade-entry training rows and realized Testnet outcomes remain pending their respective lifecycle events.

Store dated candidate, replay, rejection, testnet, and later live outcomes separately. Feed ML, reinforcement learning, teacher-student, and specialist agents only point-in-time features and clearly typed labels. Learning components propose or rank research; deterministic safety gates retain authority.

Current learning checkpoint:

- immutable learning ledger `hllearning_20260808T104841496693Z_2f761bdb` is bound to validation `hlvalidation_20260808T104841496693Z_1717acdc`
- all 3,952 experiment IDs have exactly one dated `backtest_research_outcome` record; the ledger contains 3,443 pre-evaluation blockers, 481 evaluated-but-not-promoted replays, and 28 practical walk-forward passes that were not promoted
- all seven validation-stage manifests are existence-checked, SHA-256 hashed, and copied into the immutable learning snapshot
- aggregate experiment summaries are explicitly `training_eligible=false`; they are not appended to `trades.jsonl`, and post-evaluation metrics cannot become entry-time features
- paper and live labels are empty, realized-outcome availability is false, Crypto Wizards is not label authority, no order was submitted, and live trading remains unauthorized

### 10. Reproducibility And Live Readiness

Status: deterministic downstream orchestration is implemented; authenticated capture automation and final end-to-end completion remain pending.

Publish coverage, source hashes, settings, code version, environment, history windows, cost snapshots, walk-forward folds, rankings, risk surfaces, testnet journals, learning lineage, blockers, and stale-data checks. Live remains disabled until a separate human-approved release gate passes.

`run-exhaustive-wizard-hyperliquid-validation` rebuilds cost bridging, observed-cost replay, purged walk-forward, regime attribution, robustness, concentration, and leverage in fixed order from frozen evidence with one shared timestamp. It verifies identical exhaustive run identity, equal experiment counts, complete status accounting, and false live-trading authority at every stage, then writes an immutable validation manifest. It intentionally does not claim to automate authenticated Wizard capture and cannot submit orders.

## Calibrated Validation Checkpoint: 2026-08-08 10:48 UTC

- validation `hlvalidation_20260808T104841496693Z_1717acdc` completed all seven deterministic stages against exhaustive run `ewhl_5cf7c50e51f6fee5010c`; every stage accounts for all 3,952 planned experiments and points to an immutable snapshot manifest
- all 103 mapped pair cost models have 12 complete samples per leg inside the rolling two-hour window; raw cadence counts and persisted model counts match, with zero waiting, rebuild-due, infeasible, or expired rows at the calibration checkpoint
- cadence projection now simulates future rolling-window expiry instead of equating the current sample deficit with the number of future captures; impossible target/window/cadence combinations fail closed as `calibration_schedule_infeasible`
- Hyperliquid timestamp parsing accepts mixed ISO-8601 precision, preventing valid whole-second exchange timestamps from being silently dropped when the same evidence column also contains fractional-second timestamps
- calibrated estimated pair round-trip cost is 19.4790 bps at the median, 29.2422 bps at the 95th percentile, and 49.0692 bps for the highest-cost pair, MNT/NOT
- cost bridge `hlcost_f5a2d808d5a39ad4b752` finds 43 pair groups ready for calibrated cost replay and 51 pair groups ready for provisional cost research; all other pair groups retain explicit evidence blockers
- observed-cost replay `hlobserved_20260808T104841496693Z_8a944bdc` completes 509 cells and writes 8,015 closed-trade-ledger rows; all results remain research-only
- walk-forward `hlwalk_20260808T104841496693Z_3c005f4c` completes 300 five-fold candidates; 28 pass the practical research gate and zero pass the 10% Benjamini-Hochberg family-wide selection gate
- regime attribution `hlregime_20260808T104841496693Z_60f017db` finds one research regime-stability pass, while robustness `hlrobust_20260808T104841496693Z_3002a3e0` finds 19 research robustness passes; neither diagnostic can override the zero statistical-selection cohort
- concentration `hlconcentration_20260808T104841496693Z_e49b39a9` remains blocked, so leverage `hlleverage_20260808T104841496693Z_6ceb1c0f` correctly creates zero candidates, zero scenarios, and zero Testnet lifecycle-ready configurations
- learning ledger `hllearning_20260808T104841496693Z_2f761bdb` accounts for all 3,952 experiment outcomes without creating trade-entry training rows or paper/live labels
- refreshed Testnet inventory and margin-tier evidence are fresh in the leverage manifest; the no-order preflight passes, submissions remain disabled, perpetual collateral remains zero, no order was sent, and live trading remains unauthorized

## Current Artifacts

- `reports/active/current_wizard_pair_detail_status.csv`
- `reports/active/current_wizard_hyperliquid_experiment_matrix.csv`
- `reports/active/current_wizard_hyperliquid_pair_history_queue.csv`
- `reports/active/current_wizard_hyperliquid_asset_fetch_queue.csv`
- `reports/active/current_wizard_hyperliquid_handoff_validation.csv`
- `reports/active/current_wizard_hyperliquid_handoff_manifest.json`
- `reports/active/current_wizard_hyperliquid_asset_history_results.csv`
- `reports/active/current_wizard_hyperliquid_pair_history_results.csv`
- `reports/active/current_wizard_hyperliquid_history_validation.csv`
- `reports/active/current_wizard_hyperliquid_history_manifest.json`
- `reports/active/current_wizard_hyperliquid_canonical_replay.csv`
- `reports/active/current_wizard_hyperliquid_canonical_replay_ranked.csv`
- `reports/active/current_wizard_hyperliquid_canonical_replay_trades.csv`
- `reports/active/current_wizard_hyperliquid_replay_pair_status.csv`
- `reports/active/current_wizard_hyperliquid_replay_validation.csv`
- `reports/active/current_wizard_hyperliquid_canonical_replay_manifest.json`
- `reports/active/current_wizard_hyperliquid_funding_asset_results.csv`
- `reports/active/current_wizard_hyperliquid_pair_cost_evidence.csv`
- `reports/active/current_wizard_hyperliquid_experiment_cost_readiness.csv`
- `reports/active/current_wizard_hyperliquid_cost_manifest.json`
- `reports/active/current_wizard_hyperliquid_observed_cost_replay.csv`
- `reports/active/current_wizard_hyperliquid_observed_cost_replay_ranked.csv`
- `reports/active/current_wizard_hyperliquid_observed_cost_replay_trades.csv`
- `reports/active/current_wizard_hyperliquid_observed_cost_replay_manifest.json`
- `reports/active/current_wizard_hyperliquid_walkforward_status.csv`
- `reports/active/current_wizard_hyperliquid_walkforward_candidates.csv`
- `reports/active/current_wizard_hyperliquid_walkforward_folds.csv`
- `reports/active/current_wizard_hyperliquid_walkforward_trades.csv`
- `reports/active/current_wizard_hyperliquid_walkforward_bars.csv.gz`
- `reports/active/current_wizard_hyperliquid_walkforward_manifest.json`
- `reports/active/current_wizard_hyperliquid_regime_status.csv`
- `reports/active/current_wizard_hyperliquid_regime_candidates.csv`
- `reports/active/current_wizard_hyperliquid_regime_detail.csv`
- `reports/active/current_wizard_hyperliquid_regime_bars.csv.gz`
- `reports/active/current_wizard_hyperliquid_regime_trades.csv`
- `reports/active/current_wizard_hyperliquid_regime_manifest.json`
- `reports/active/current_wizard_hyperliquid_robustness_status.csv`
- `reports/active/current_wizard_hyperliquid_robustness_candidates.csv`
- `reports/active/current_wizard_hyperliquid_robustness_scenarios.csv`
- `reports/active/current_wizard_hyperliquid_robustness_folds.csv`
- `reports/active/current_wizard_hyperliquid_robustness_manifest.json`
- `reports/active/current_wizard_hyperliquid_concentration_status.csv`
- `reports/active/current_wizard_hyperliquid_concentration_cohorts.csv`
- `reports/active/current_wizard_hyperliquid_concentration_dimensions.csv`
- `reports/active/current_wizard_hyperliquid_concentration_contributors.csv`
- `reports/active/current_wizard_hyperliquid_concentration_validation.csv`
- `reports/active/current_wizard_hyperliquid_concentration_manifest.json`
- `reports/active/current_wizard_hyperliquid_failure_attribution.csv`
- `reports/active/current_wizard_hyperliquid_failure_attribution_summary.csv`
- `reports/active/current_wizard_hyperliquid_failure_attribution_validation.csv`
- `reports/active/current_wizard_hyperliquid_failure_attribution_manifest.json`
- `reports/active/current_wizard_hyperliquid_leverage_status.csv`
- `reports/active/current_wizard_hyperliquid_leverage_candidates.csv`
- `reports/active/current_wizard_hyperliquid_leverage_scenarios.csv`
- `reports/active/current_wizard_hyperliquid_leverage_validation.csv`
- `reports/active/current_wizard_hyperliquid_leverage_manifest.json`
- `reports/active/current_wizard_hyperliquid_learning_ledger.csv.gz`
- `reports/active/current_wizard_hyperliquid_learning_ledger.jsonl.gz`
- `reports/active/current_wizard_hyperliquid_learning_validation.csv`
- `reports/active/current_wizard_hyperliquid_learning_manifest.json`
- `reports/active/current_wizard_ou_optimal_overlay_ledger.csv`
- `reports/active/current_wizard_ou_optimal_overlay_coverage.csv`
- `reports/active/current_wizard_ou_optimal_overlay_validation.csv`
- `reports/active/current_wizard_ou_optimal_overlay_manifest.json`
- `reports/active/current_wizard_ou_optimal_overlay_summary.md`
- `reports/active/current_wizard_hyperliquid_chain_validation.csv`
- `reports/active/current_wizard_hyperliquid_chain_validation_manifest.json`
- `reports/active/current_wizard_hyperliquid_chain_validation_summary.md`
- `reports/active/current_wizard_hyperliquid_operating_cadence.csv`
- `reports/active/current_wizard_hyperliquid_operating_cadence_validation.csv`
- `reports/active/current_wizard_hyperliquid_operating_cadence_manifest.json`
- `reports/active/current_wizard_hyperliquid_operating_cadence_summary.md`
- `reports/active/current_wizard_hyperliquid_live_lock.csv`
- `reports/active/current_wizard_hyperliquid_storage_efficiency.csv`
- `reports/active/current_wizard_hyperliquid_storage_reclamation_plan.csv`
- `reports/active/current_wizard_hyperliquid_storage_reclamation_validation.csv`
- `reports/active/current_wizard_hyperliquid_storage_reclamation_manifest.json`
- `reports/active/current_wizard_hyperliquid_storage_reclamation_summary.md`
- `reports/active/current_wizard_hyperliquid_archive_copy_receipt.csv`
- `reports/active/current_wizard_hyperliquid_archive_copy_validation.csv`
- `reports/active/current_wizard_hyperliquid_archive_copy_manifest.json`
- `reports/active/current_wizard_hyperliquid_archive_copy_summary.md`
- `reports/active/current_wizard_hyperliquid_archive_release_plan.csv`
- `reports/active/current_wizard_hyperliquid_archive_release_validation.csv`
- `reports/active/current_wizard_hyperliquid_archive_release_manifest.json`
- `reports/active/current_wizard_hyperliquid_archive_release_summary.md`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_scenarios.csv`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_transitions.csv`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_candidate_coverage.csv`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_validation.csv`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_manifest.json`
- `reports/active/current_wizard_hyperliquid_testnet_protocol_summary.md`
- `data/meta_learning/current_wizard_hyperliquid_research_outcomes.csv.gz`
- `data/meta_learning/current_wizard_hyperliquid_research_outcomes.jsonl.gz`
- `reports/active/wizard_pair_detail_api_pilot_manifest.json`
- `reports/active/wizard_pair_detail_api_pilot_manifest.csv`
- `reports/active/wizard_pair_detail_api_pilot_fields.csv`
- `reports/active/wizard_pair_detail_api_pilot_coverage.csv`
- `reports/active/wizard_pair_detail_api_pilot_attempt_history.csv`
- `reports/active/exhaustive_wizard_dashboard_capture_manifest.csv`
- `reports/active/exhaustive_wizard_dashboard_capture_validation.csv`
- `reports/active/exhaustive_wizard_dashboard_rows.csv`
- `reports/active/exhaustive_wizard_source_row_ledger.csv`
- `reports/active/exhaustive_wizard_pair_ledger.csv`
- `reports/active/exhaustive_wizard_pair_detail_capture_queue.csv`
- `reports/active/exhaustive_wizard_pair_detail_mode_ledger.csv`
- `reports/active/exhaustive_wizard_pair_detail_coverage_validation.csv`
- `reports/active/exhaustive_wizard_pair_detail_capture_progress.csv`
- `reports/active/exhaustive_wizard_pair_detail_capture_summary.md`
- `reports/active/exhaustive_wizard_experiment_matrix.csv`
- `reports/active/exhaustive_wizard_coverage_validation.csv`
- `reports/active/exhaustive_wizard_hyperliquid_run_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_mapping_refresh.csv`
- `reports/active/exhaustive_wizard_hyperliquid_inventory_drift.csv`
- `reports/active/exhaustive_wizard_hyperliquid_mapping_refresh_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_mapping_refresh_summary.md`
- `reports/active/hyperliquid_testnet_margin_tiers.csv`
- `reports/active/exhaustive_wizard_hyperliquid_replay_preflight.csv`
- `reports/active/exhaustive_wizard_hyperliquid_pair_history_queue.csv`
- `reports/active/exhaustive_wizard_hyperliquid_asset_fetch_queue.csv`
- `reports/active/exhaustive_wizard_hyperliquid_replay_preflight_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_replay_preflight_summary.md`
- `reports/active/exhaustive_wizard_hyperliquid_funding_pair_coverage.csv`
- `reports/active/exhaustive_wizard_hyperliquid_funding_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_pair_cost_evidence.csv`
- `reports/active/exhaustive_wizard_hyperliquid_experiment_cost_readiness.csv`
- `reports/active/exhaustive_wizard_hyperliquid_cost_evidence_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_observed_cost_replay.csv`
- `reports/active/exhaustive_wizard_hyperliquid_observed_cost_replay_ranked.csv`
- `reports/active/exhaustive_wizard_hyperliquid_observed_cost_replay_trades.csv`
- `reports/active/exhaustive_wizard_hyperliquid_observed_cost_comparison.csv`
- `reports/active/exhaustive_wizard_hyperliquid_observed_cost_replay_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_status.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_candidates.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_ranked.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_folds.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_trades.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_bars.csv`
- `reports/active/exhaustive_wizard_hyperliquid_walkforward_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_regime_status.csv`
- `reports/active/exhaustive_wizard_hyperliquid_regime_candidates.csv`
- `reports/active/exhaustive_wizard_hyperliquid_regime_detail.csv`
- `reports/active/exhaustive_wizard_hyperliquid_regime_bars.csv`
- `reports/active/exhaustive_wizard_hyperliquid_regime_trades.csv`
- `reports/active/exhaustive_wizard_hyperliquid_regime_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_robustness_status.csv`
- `reports/active/exhaustive_wizard_hyperliquid_robustness_candidates.csv`
- `reports/active/exhaustive_wizard_hyperliquid_robustness_scenarios.csv`
- `reports/active/exhaustive_wizard_hyperliquid_robustness_folds.csv`
- `reports/active/exhaustive_wizard_hyperliquid_robustness_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_status.csv`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_cohorts.csv`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_dimensions.csv`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_contributors.csv`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_concentration_summary.md`
- `reports/active/exhaustive_wizard_hyperliquid_leverage_status.csv`
- `reports/active/exhaustive_wizard_hyperliquid_leverage_candidates.csv`
- `reports/active/exhaustive_wizard_hyperliquid_leverage_scenarios.csv`
- `reports/active/exhaustive_wizard_hyperliquid_leverage_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_leverage_summary.md`
- `reports/active/exhaustive_wizard_hyperliquid_validation_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_validation_summary.md`
- `reports/active/exhaustive_wizard_hyperliquid_learning_ledger.csv`
- `reports/active/exhaustive_wizard_hyperliquid_learning_ledger.jsonl`
- `reports/active/exhaustive_wizard_hyperliquid_learning_manifest.json`
- `reports/active/exhaustive_wizard_hyperliquid_learning_summary.md`
- `data/meta_learning/exhaustive_wizard_hyperliquid_research_outcomes.csv`
- `data/meta_learning/exhaustive_wizard_hyperliquid_research_outcomes.jsonl`

## Completion Criteria

The goal is complete only when every captured source row is accounted for, every pair group has pair-detail status, every experiment has a replay result or explicit blocker, every accepted leveraged configuration has a preserved 1x baseline plus stress evidence and a recorded maximum proven leverage, every Testnet candidate completes the required lifecycle checks at each promoted level, every learning record has point-in-time lineage, all reports reproduce from one command, and live trading is still disabled pending a separate authorization decision.
