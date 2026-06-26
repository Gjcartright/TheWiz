# Quantized Statistical Arbitrage Platform

Research-first platform for discovering, validating, ranking, and eventually executing crypto statistical arbitrage opportunities. The system is designed to prove where edge comes from rather than assuming that z-score, cointegration, ECM, copula, or any single indicator is alpha.

## Current State

This repository contains the production scaffold:

- API extraction engine for Crypto Wizards field discovery and response archiving.
- Field, formula, and quant brain dictionaries.
- Explainable 0-100 feature scoring.
- Automatic feature-score enrichment inside the experiment harness for composite, voting, and portfolio strategies.
- Strategy research registry covering all required strategy families.
- Unified experiment harness for pair, regime, and cost-bucket evaluation.
- Executable deterministic signals for z-score, ECM, copula, half-life, Hurst, OU, proprietary filter, dynamic threshold, and regime-filtered strategy families.
- Backtest accounting for fees, funding, slippage, and execution risk.
- Two-leg spread backtesting when leg prices are available, including hedge ratio, beta, per-leg funding, slippage, execution risk, and partial-fill assumptions.
- Portfolio ranking and risk allocation primitives.
- Meta-learning trade store schema.
- dYdX execution adapter interface with dry-run, testnet indexer reads, and gated paper-trading modes.

Live API calls require credentials and exact endpoint details. Until then, research can run from archived JSON/CSV snapshots in `data/raw`.

## Non-Negotiable Research Rules

- No edge is assumed.
- Every field must be documented, scored, and tested.
- Every strategy must be evaluated after realistic costs.
- No strategy is successful without profit factor >= 1.8, Sharpe > 1.2, max drawdown < 15%, 100+ completed trades, and multi-pair robustness.
- Portfolio construction ranks opportunities; it does not trade every signal.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m quant_platform.cli build-dictionaries
python -m quant_platform.cli ingest-fixtures
python -m quant_platform.cli run-demo-backtest
python -m quant_platform.cli run-demo-experiments
python -m quant_platform.cli run-fixture-experiments
```

Generated research artifacts are written to `docs/` and `reports/`.

## Local Keys In VS Code

Create the private local env file:

```bash
python3 scripts/setup_local_env.py
```

Then open `.env.local` in VS Code and fill in:

- `CRYPTO_WIZARDS_BASE_URL`
- `CRYPTO_WIZARDS_API_KEY`
- `CRYPTO_WIZARDS_ENDPOINTS`
- `DYDX_TESTNET_WALLET_ADDRESS`
- `DYDX_TESTNET_PRIVATE_KEY`
- `DYDX_TESTNET_SUBMIT_ORDERS`
- `QPA_ACCEPTANCE_REPORT_PATH`

`.env.local` is ignored by git. `.env.example` is the safe template to keep in the repo.

If the canonical branch truth comes from a reviewed rerun artifact instead of `reports/acceptance_report.csv`, set `QPA_ACCEPTANCE_REPORT_PATH` in `.env.local`. Readiness, preflight, and paper-plan commands will use that file as the acceptance source.

VS Code launch configs are available under Run and Debug:

- `Quant: Check Live Config`
- `Quant: Run Fixture Experiments`
- `Quant: Paper Plan`

Experiment reports are written to:

- `reports/experiment_results.csv`
- `reports/ablation_report.csv`
- `reports/acceptance_report.csv`
- `reports/strategy_summary.csv`
- `reports/regime_summary.csv`
- `reports/regime_pair_strategy_report.csv`
- `reports/strategy_coverage.csv`
- `reports/priority_readiness.csv`
- `reports/priority_action_plan.csv`
- `reports/priority_spine_dashboard.csv`
- `reports/priority_gap_test.csv`
- `reports/paper_trading_journal.csv`
  and `data/meta_learning/trades.jsonl` for learning-event capture

`reports/experiment_results.csv` includes `backtest_mode` and input coverage flags such as `has_price_x`, `has_price_y`, `has_hedge_ratio`, `has_beta`, `has_funding_x`, and `has_funding_y`. Use these columns to distinguish spread-level tests from execution-realistic two-leg tests.

When leg prices are present and native beta is missing, fixture and pair-detail ingestion derive `beta` from leg return covariance and mark `beta_source=derived_from_price_returns`. Funding is not fabricated; production acceptance still requires real `funding_x_bps` and `funding_y_bps` evidence.

Use `--funding-path /path/to/funding.csv` or `--funding-path /path/to/funding.json` with `run-fixture-experiments` or `run-pair-detail-experiments` to merge real per-market funding history into pair datasets. Funding rows should include a market identifier such as `market`, `ticker`, or `symbol`; a value such as `funding_bps`, `funding_rate_bps`, `rate_bps`, `rate`, or `funding_rate`; and optionally a timestamp such as `timestamp`, `effective_at`, or `effectiveAt`. Decimal rates are converted to basis points; bps columns are kept as bps.

Use `python -m quant_platform.cli funding-requirements` to write `reports/funding_requirements.csv` and print the dYdX markets required by the latest experiment pairs. If live fetching is unavailable, use `python -m quant_platform.cli funding-template --output-path data/processed/dydx_funding_template.csv` to create a `market,timestamp,funding_bps` template for the required markets, then fill it with real funding observations. Validate it with `python -m quant_platform.cli funding-template-check --input-dir data/processed/dydx_funding_template.csv`, then normalize it into `data/processed/dydx_funding.csv` with `python -m quant_platform.cli import-funding-template --input-dir data/processed/dydx_funding_template.csv --output-path data/processed/dydx_funding.csv` before coverage and research. If the dYdX indexer adapter is available, use the printed market list with `python -m quant_platform.cli fetch-dydx-funding --market ETH-USD,BTC-USD,SOL-USD --output-path data/processed/dydx_funding.csv` to fetch and normalize funding directly. For saved dYdX indexer funding responses, use `python -m quant_platform.cli export-dydx-funding --json-path /path/to/funding_payload.json --market ETH-USD --output-path data/processed/dydx_funding.csv`, then pass the output file to `--funding-path`. `--json-path` can also point at a directory of per-market JSON payloads; names such as `ETH-USD_funding.json` and `BTC-USD_funding.json` are used to fill missing market identifiers before one combined CSV is written.

For normal dYdX market/funding refreshes, use Apify MCP actor snapshots as the primary acquisition path and ingest them with the same local normalizers. `export-dydx-funding` accepts fields such as `nextFundingRate`, `next_funding`, `scrapedAt`, and `timeMs`, which lets you import Apify actor snapshots before running `funding-coverage` and `funded-research-spine`. Direct dYdX indexer fetches remain available as diagnostics or recovery paths.

Before rerunning acceptance, use `python -m quant_platform.cli funding-requirements` to confirm the needed dYdX markets, fetch or export those markets into `data/processed/dydx_funding.csv`, then use `python -m quant_platform.cli funding-coverage --funding-path data/processed/dydx_funding.csv` to write `reports/funding_coverage.csv`. The coverage command checks the latest experiment pairs, or a specific `--pair ETH-BTC`, and reports whether both dYdX leg markets have funding rows. To run the whole guarded P2 path, use `python -m quant_platform.cli funded-research-spine --funding-path data/processed/dydx_funding.csv`; it refuses to rerun research until funding coverage is complete, then refreshes research spine and strategy acceptance.

`reports/strategy_summary.csv` and `reports/regime_summary.csv` include execution-economics aggregates: median gross return, median net return, median cost drag, total fees, total slippage, total funding, total execution-risk cost, total partial-fill cost, and median gross exposure. Use these columns to reject strategies whose apparent edge is consumed by realistic execution costs.

`reports/acceptance_report.csv` includes `required_backtest_mode`, `required_two_leg_inputs`, `two_leg_pairs_tested`, `two_leg_execution_input_pairs`, and `two_leg_passing_pairs`. `reports/strategy_acceptance_checklist.csv` summarizes whether experiment results exist, two-leg coverage exists, hedge/beta/funding execution assumptions are present, required cost buckets are present, and whether any strategy is production/preferred eligible. The default production gate requires `two_leg` plus explicit leg prices, hedge ratio, beta, and per-leg funding evidence, so spread-only or default-assumption results cannot unlock dYdX paper execution even if their headline metrics look strong.

Acceptance gates are applied twice:

- Run-level: each evaluated pair/regime/cost bucket must pass PF, Sharpe, drawdown, trade count, and expectancy thresholds.
- Strategy-level: production eligibility requires passing two-leg results across at least two pairs, explicit two-leg execution inputs, and both base and stress cost buckets in the `ALL` regime slice.

Ablation reports compare hybrid/enhanced strategies against simpler baselines on matched `pair/regime/cost_bucket` rows, so ECM, copula, Hurst, half-life, OU, regime filters, and proprietary components have to prove incremental value.

Fixture ingestion writes:

- `docs/crypto_wizards_fixture_field_dictionary.csv`
- `reports/fixture_ingestion_summary.csv`

## Repository Layout

- `src/quant_platform/api_extraction.py` - Crypto Wizards endpoint and field discovery.
- `src/quant_platform/field_registry.py` - canonical field definitions.
- `src/quant_platform/formula_registry.py` - formula and interpretation intelligence.
- `src/quant_platform/feature_engine.py` - explainable normalized scores.
- `src/quant_platform/fixture_ingestion.py` - archived Crypto Wizards JSON/CSV fixture ingestion.
- `src/quant_platform/strategies.py` - strategy family registry and signal hooks.
- `src/quant_platform/backtest.py` - cost-aware pair-trade backtester.
- `src/quant_platform/experiments.py` - unified experiment harness and report writer.
- `src/quant_platform/portfolio.py` - ranking and allocation.
- `src/quant_platform/execution.py` - dYdX adapter contracts.
- `src/quant_platform/meta_learning.py` - trade/event store schema.
- `docs/architecture.md` - platform architecture.
- `docs/dydx_v4_reference.md` - dYdX v4 client and testnet setup notes.
- `docs/quant_brain.md` - field-by-field research reasoning.
- `docs/formula_dictionary.md` - field formulas and failure modes.
- `docs/field_dictionary.csv` - canonical field dictionary.

## Execution Modes

- `dry_run`: local-only simulation. No network order submission.
- `paper`: dYdX testnet-backed paper trading. Order submission remains disabled by default until wallet credentials, official client wiring, and risk gates are configured.
- `live`: reserved for production execution and intentionally not enabled in the scaffold.

Paper spread execution is research-gated: a strategy must be marked `production_eligible` in `reports/acceptance_report.csv` before a dYdX testnet paper order plan is created.

Authenticated dYdX testnet submission remains blocked unless `submit_orders` is enabled, credentials are explicitly supplied, and an authenticated order-client adapter is injected. The dYdX indexer adapter can be wired earlier for market/funding reads; the scaffold does not fake exchange-side fills.

Use `python -m quant_platform.cli paper-plan --pair ETH-BTC --strategy-id 1 --signal 1 --hedge-ratio 1 --beta 1 --notional-usd 1000` to test the research-gated execution handoff. The command reads `reports/acceptance_report.csv`, rejects non-eligible strategies, and only then creates two-leg dYdX paper intents. Every attempted paper handoff appends an auditable row to `reports/paper_trading_journal.csv` with plan status, reason, blockers, intents, and fills.

Use `python -m quant_platform.cli learning-report` to write `reports/learning_event_summary.csv`. The report summarizes paper handoff events, research-rejected handoffs, dYdX-config-blocked handoffs, blocked fills, submitted fills, and `data/meta_learning/trades.jsonl` outcome records so the P5 learning gate can distinguish audit evidence from model-ready realized outcomes.

After a paper trade has a realized result, append it to the learning store with `python -m quant_platform.cli append-learning-outcome --pair ETH-BTC --strategy-id 1 --realized-return 0.012 --signal 1 --hedge-ratio 1.2 --beta 0.9 --notional-usd 1000 --regime range --trade-id paper-001`. For batch/manual collection, you can seed `data/meta_learning/learning_outcome_template.csv` directly from submitted paper journal rows with `python -m quant_platform.cli seed-learning-outcome-template --input-dir reports/paper_trading_journal.csv --output-path data/meta_learning/learning_outcome_template.csv`, then fill `realized_return`, validate it with `python -m quant_platform.cli learning-outcome-template-check --input-dir data/meta_learning/learning_outcome_template.csv`, then import ready rows with `python -m quant_platform.cli import-learning-outcomes --input-dir data/meta_learning/learning_outcome_template.csv --output-path reports/learning_outcome_import_report.csv`. Then rerun `learning-report` and `priority-readiness`. P5 only becomes model-ready after enough realized outcome events exist; audit-only paper handoffs do not count as training labels.

## ML Trade Filter Layer

The ML layer is an additive filter on top of the existing pair-construction, signal, risk, execution, and journaling spine. It does not replace strategy logic.

Contract:

- one row = one candidate trade entry event
- label = profitable after costs under the current exit logic
- role = filter only (`take` or `skip`)

Build a leakage-safe candidate-trade dataset from pair histories:

```bash
PYTHONPATH=src python -m quant_platform.cli build-ml-trade-filter-dataset \
  --input-dir data/raw/pair_details \
  --funding-path data/processed/dydx_funding.csv \
  --output-path reports/ml_trade_filter_dataset.csv
```

Run baseline walk-forward validation and compare rule-only versus rule-plus-filter:

```bash
PYTHONPATH=src python -m quant_platform.cli train-ml-trade-filter \
  --input-dir reports/ml_trade_filter_dataset.csv \
  --output-dir reports/ml_trade_filter \
  --walkforward-splits 5 \
  --min-train-rows 100
```

Outputs include:

- `ml_trade_filter_dataset.csv`
- `ml_trade_filter_walkforward_folds.csv`
- `ml_trade_filter_walkforward_predictions.csv`
- `ml_trade_filter_comparison_summary.csv`
- `ml_trade_filter_best_model.pkl`
- `ml_trade_filter_manifest.json`

The baseline model set currently includes:

- logistic regression
- XGBoost when installed
- LightGBM when installed
- local boosted-tree fallback when neither external package is available

Shadow-mode inference is available without changing live behavior:

```bash
PYTHONPATH=src python -m quant_platform.cli shadow-ml-trade-filter \
  --input-dir reports/ml_trade_filter_dataset.csv \
  --model-path reports/ml_trade_filter/ml_trade_filter_best_model.pkl \
  --output-path reports/ml_trade_filter_shadow_predictions.csv
```

To compare saved walk-forward shadow decisions across a narrowed branch/pair set:

```bash
PYTHONPATH=src python -m quant_platform.cli compare-ml-shadow-models \
  --input-dir reports/ml_trade_filter/ml_trade_filter_walkforward_predictions.csv \
  --pair BTC-USD-SOL-USD,DOGE-USD-ETH-USD,BTC-USD-ETH-USD,BTC-USD-DOGE-USD,BTC-USD-LINK-USD
```

This writes `ml_trade_filter_branch_model_comparison.csv` and `ml_trade_filter_branch_pair_comparison.csv` beside the prediction artifact.

Do not wire the filter into live or paper behavior changes until the comparison summary shows consistent improvement in after-cost trading metrics such as profit factor, Sharpe, drawdown, expectancy, and trade count.

## Separate Family And Combo Testing

If you want every strategy family run separately, plus combo tests built from the best strategy in each family, use:

```bash
PYTHONPATH=src python -m quant_platform.cli strategy-family-matrix \
  --input-dir data/raw/pair_details \
  --funding-path data/processed/dydx_funding.csv \
  --pair BTC-USD-SOL-USD,DOGE-USD-ETH-USD,BTC-USD-LINK-USD \
  --output-dir reports/strategy_family_matrix
```

This workflow writes:

- `family_registry.csv`
- `family_separate_summary.csv`
- `family_best_strategies.csv`
- `family_combo_summary.csv`
- `family_combo_pair_summary.csv`
- `family_combo_detail.csv`

## Research Quantization Layer

Once a family matrix exists, you can quantize the current candidates into a ranked decision table without changing strategy execution:

```bash
PYTHONPATH=src python -m quant_platform.cli research-quantization \
  --input-dir reports/strategy_family_matrix_canonical \
  --output-dir reports/strategy_family_matrix_canonical/quantized \
  --top-n 10
```

This writes:

- `research_quantization_ranked.csv`
- `research_quantization_top.csv`
- `research_quantization_summary.csv`
- `research_quantization_runbook.md`

The quantization layer scores each family or combo using current gate status, passing-pair coverage, Sharpe, profit factor, drawdown, and trade count. It is meant to rank and bucket candidates into `promote_now`, `shadow_ready`, `watchlist`, or `reject` while keeping the existing trading spine unchanged.
- `family_matrix_runbook.md`

It also creates per-family report folders under:

- `reports/strategy_family_matrix/families/<family>/`

Interpretation:

- separate-family runs show which strategy wins inside each family
- combo runs test pairwise combinations of the best strategy from each family
- one full-stack combo is also produced using all family winners together

Use this workflow when you want clean family isolation instead of one blended sweep.

Use `python -m quant_platform.cli check-dydx-config` to verify dYdX testnet readiness. Use `python -m quant_platform.cli dydx-order-adapter-contract` to write `reports/dydx_order_adapter_contract.csv` and validate that the configured adapter imports and exposes `place_order(intent, config)` without sending an order. Use `python -m quant_platform.cli dydx-execution-checklist` to write `reports/dydx_execution_checklist.csv`, which separates indexer reads, credentials, SDK availability, submit flag, order-adapter wiring, research acceptance, and the final paper-submission gate. Install the official v4 Python client with `pip install -e ".[dev,dydx]"` when you are ready to wire indexer reads and authenticated paper submission.

Authenticated paper order submission requires an explicit local adapter: set `DYDX_TESTNET_ORDER_CLIENT_ADAPTER=module:object` in `.env.local`, where the object or zero-argument class implements `place_order(intent, config) -> FillReport`. This hook is intentionally separate from credentials and `DYDX_TESTNET_SUBMIT_ORDERS`; all gates must pass before orders are submitted.

For local handoff and journal testing only, the built-in record-only adapter can be configured with `DYDX_TESTNET_ORDER_CLIENT_ADAPTER=quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter`. It satisfies the adapter contract and returns `paper_recorded_not_submitted`, but it does not place dYdX testnet orders and must not be treated as proof of authenticated exchange submission. The readiness checklist keeps this adapter blocked with `record_only_dydx_order_client_adapter`; replace it with an authenticated dYdX order adapter before enabling exchange-side paper submission.

Use `python -m quant_platform.cli priority-readiness` for a consolidated gate report covering Crypto Wizards live artifacts, pair-detail history, nested capture candidates, strategy acceptance, dYdX testnet readiness, paper execution readiness, and the learning event store. It writes `reports/priority_readiness.csv`, `reports/priority_action_plan.csv`, and `reports/priority_spine_dashboard.csv`. P5 is ready only once model-ready realized outcome events exist; blocked paper handoffs remain useful audit records but do not unlock learning.

Use `python -m quant_platform.cli paper-execution-preflight` to write `reports/paper_execution_preflight.csv`. This decomposes P4 into strategy acceptance, dYdX testnet readiness, paper submission gate, and paper journal audit readiness, with the next action inherited from the first blocked dependency.

Use `python -m quant_platform.cli priority-actions` to print the ranked blocked-gate work queue derived from the latest P1-P5 readiness state.

Use `python -m quant_platform.cli priority-dashboard` to print the one-row-per-priority dashboard with the current blocker, key metric, source report, and next action.

Use `python -m quant_platform.cli priority-runbook` to write `reports/priority_runbook.md`, a Markdown operator runbook combining the dashboard, gap proof requirements, ranked work queue, and exact P1-P5 commands.

Use `python -m quant_platform.cli gap-test` to write `reports/priority_gap_test.csv`, which classifies each open P1-P5 gap by severity, current evidence, required proof, source report, and next action.

Use `python -m quant_platform.cli strategy-acceptance-checklist` to print the P2 acceptance checklist and write `reports/strategy_acceptance_checklist.csv`.

Use `python -m quant_platform.cli strategy-failure-attribution` after experiments to write `reports/strategy_failure_attribution.csv`. This explains why each strategy failed or stayed data-blocked, including missing feature columns, too few trades, weak profit factor, weak Sharpe, negative expectancy, drawdown, and cost drag.

Use `python -m quant_platform.cli research-unblock-plan` after failure attribution to write `reports/research_unblock_plan.csv`. This turns the current P2 blockers into a ranked action plan, including estimated extra 5-minute history needed for 100/250-trade evidence, missing Crypto Wizards feature fields by strategy family, pair-universe quality blockers, and the paper-trading gate status.

Use `python -m quant_platform.cli zscore-threshold-sweep --funding-path data/processed/dydx_funding.csv` to write `reports/zscore_threshold_sweep.csv` and `reports/zscore_threshold_sweep_summary.csv`. This tests whether lower or higher z-score entry thresholds materially improve trade count and quality under the same two-leg fees, funding, slippage, execution-risk, and partial-fill assumptions. It is diagnostic only; it does not relax production gates or mark a strategy accepted.

Use `bash scripts/run_dydx_long_history.sh --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --windows 12 --limit 1000 --funding-path data/processed/dydx_funding.csv` to generate the long-history plan, fetch the candle windows with curl, merge/deduplicate them, build the two-leg pair history, and rerun the guarded research path in one command. The shell runner keeps partial progress by default when a fetch fails; add `--strict` if you want the whole run to stop on the first failed window. The lower-level `dydx-long-history-plan` and `build-dydx-long-history-pair` commands remain available when you want to inspect or stage the pieces separately, and `run-dydx-long-history` remains the Python equivalent when the local resolver can reach the indexer directly.

Use `python -m quant_platform.cli dydx-pair-expansion-plan --max-pairs 10 --limit 1000` to write `reports/dydx_pair_expansion_plan.csv`. This creates a ranked queue of additional two-leg dYdX 5-minute pair candidates, skips pairs already tested, shows missing funding markets, and prints copy-ready `fetch-dydx-two-leg-data --derive-hedge-ratio --run-research` commands for expanding the research universe.

Use `python -m quant_platform.cli run-dydx-pair-expansion --max-pairs 1 --limit 1000 --run-research` to execute the first fresh ranked pair from that queue and write `reports/dydx_pair_expansion_run.csv`. The runner records completed and failed pair fetches separately, builds candle-derived pair history with derived hedge ratio/beta, normalizes funding, and reruns the guarded research path only when requested. It does not bypass strategy acceptance or dYdX paper-trading gates.

Use `python -m quant_platform.cli research-spine` to run the guarded research backbone: refresh pair-detail reports, refresh readiness, and run pair-detail experiments only when the required history is present. Pass `--funding-path data/processed/dydx_funding.csv` to merge dYdX funding into those experiments in the same run. By default it requires two-leg price history; `--allow-spread-only` is available for local diagnostics but does not satisfy production eligibility.

Live Crypto Wizards crawling is configured through environment variable names in `config/research.yaml`; fixture ingestion remains the verified default path until real endpoint URLs and credentials are supplied.

Use `python -m quant_platform.cli check-live-config` to verify required Crypto Wizards environment variables before attempting a live crawl. Endpoints can be supplied as `--endpoint pairs=/v1/pairs` or through `CRYPTO_WIZARDS_ENDPOINTS`.

For the official 5-minute Crypto Wizards research path, use the documented v1beta API instead of browser-copied candle workarounds:

```bash
PYTHONPATH=src python3 -m quant_platform.cli crypto-wizards-min5-request-template --asset-x BNB-USD --asset-y STX-USD
PYTHONPATH=src python3 -m quant_platform.cli crawl-crypto-wizards-min5 --max-pairs 10 --priority Sharpe --cw-strategy Spread --exchange Dydx --period 320 --spread-type Static --roll-w 42
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-quality
PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments
PYTHONPATH=src python3 -m quant_platform.cli strategy-acceptance-checklist
PYTHONPATH=src python3 -m quant_platform.cli gap-test
```

The request-template command writes `reports/crypto_wizards_min5_api_requests.csv` with ready-to-copy `/v1beta/prescanned`, `/v1beta/zscores`, and `/v1beta/backtest` URLs plus curl commands. It uses `X-api-key: ${CRYPTO_WIZARDS_API_KEY}` as a placeholder and does not print the real key.

This command first calls `/v1beta/prescanned`, then calls `/v1beta/zscores` with `interval=Min5&with_history=true` for the selected candidates. The imported histories contain official Crypto Wizards spread, z-score, rolling z-score, hedge ratio, half-life, and Hurst where returned. They are research-usable without raw leg prices, but execution remains blocked until dYdX leg prices, funding, beta, hedge ratio, slippage, and fill assumptions are present.

To fetch official Min5 histories and immediately run the research harness:

```bash
PYTHONPATH=src python3 -m quant_platform.cli crawl-crypto-wizards-min5 --max-pairs 10 --run-research
```

For stronger Crypto Wizards-native research evidence, prefer the backtest endpoint when credits allow:

```bash
PYTHONPATH=src python3 -m quant_platform.cli crawl-crypto-wizards-min5-backtest --max-pairs 10 --run-research
```

This calls `/v1beta/backtest` with `interval=Min5&with_history=true` for prescanned candidates and imports CW backtest metrics plus spread/z-score history.

If the Codex shell cannot resolve `api.cryptowizards.net`, download or save a `/v1beta/zscores` response from another terminal/browser and import it directly:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-crypto-wizards-zscores \
  --json-path /path/to/bnb_stx_min5_zscores.json \
  --asset-x BNB-USD \
  --asset-y STX-USD \
  --exchange Dydx \
  --interval Min5 \
  --period 320 \
  --spread-type Static \
  --roll-w 42 \
  --run-research
```

For a saved `/v1beta/backtest` response, use:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-crypto-wizards-backtest \
  --json-path /path/to/bnb_stx_min5_backtest.json \
  --asset-x BNB-USD \
  --asset-y STX-USD \
  --exchange Dydx \
  --interval Min5 \
  --period 320 \
  --spread-type Static \
  --roll-w 42 \
  --run-research
```

Use `python -m quant_platform.cli diagnose-crypto-wizards` to test the configured Crypto Wizards endpoint from the current machine. It masks secrets and reports base URL presence, API key presence, endpoint count, DNS status, HTTP status, and request errors. It also writes `reports/crypto_wizards_diagnostic.csv`.

If `dns_ok` is false, the machine cannot resolve the API host; fix local DNS/network/VPN before retrying:

```bash
PYTHONPATH=src python3 -m quant_platform.cli crawl-crypto-wizards
```

For a full local network check from VS Code Terminal or macOS Terminal, run:

```bash
./scripts/check_crypto_wizards_network.sh
```

If Python networking is blocked but your Terminal can reach the API with `curl`, run the curl-based crawler:

```bash
./scripts/crawl_crypto_wizards_with_curl.sh
```

On success it writes:

- `data/raw/prescanned.json`
- `docs/crypto_wizards_live_field_dictionary.csv`

Then verify the live crawl artifacts:

```bash
PYTHONPATH=src python3 -m quant_platform.cli verify-crypto-wizards-live-artifacts
```

This also writes `reports/crypto_wizards_live_coverage.csv`, which compares the live Crypto Wizards fields against the canonical platform fields and strategy requirements. If the live endpoint does not expose ECM values, the verifier reports `live_ecm_fields_present: False` and marks ECM-dependent strategies as data-blocked.

The Crypto Wizards scanner is only the first research layer. Pair detail pages such as `/wizards/zscore/pair/<id>` expose deeper dashboard metrics, including ECM chart options. Archive dashboard pair-detail exports under `data/raw/pair_details/*.json`, then run:

```bash
PYTHONPATH=src python3 -m quant_platform.cli ingest-pair-details
PYTHONPATH=src python3 -m quant_platform.cli verify-crypto-wizards-live-artifacts
```

This writes `reports/pair_detail_research_snapshots.csv`, `reports/pair_detail_field_dictionary.csv`, `reports/pair_detail_history_coverage.csv`, `reports/pair_detail_capture_audit.csv`, and `reports/pair_detail_capture_checklist.csv`. Baseline pair-detail experiments require a snapshot payload with a `history` or `series` array containing at least `spread` and `zscore`; execution-realistic production acceptance also needs `price_x`, `price_y`, `hedge_ratio`, `beta`, and per-leg funding fields. The capture checklist includes grouped missing-field columns, `capture_completeness_score`, capture source counts such as `capture_fetches`, `capture_worker_messages`, and `capture_payload_sources`, `required_field_locations`, and `execution_assumption_locations` so each discovered field is tied back to `history`, `snapshot`, or a concrete nested JSON path such as an IndexedDB, fetch, XHR, script, or worker payload candidate.

For downloaded dashboard captures, use:

```bash
./scripts/copy_crypto_wizards_capture_helper.sh
PYTHONPATH=src python3 -m quant_platform.cli import-latest-pair-detail-download
PYTHONPATH=src python3 -m quant_platform.cli capture-preflight --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli inspect-pair-detail-capture --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli import-pair-detail-capture --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist
```

The helper command copies `scripts/capture_crypto_wizards_pair_detail.js` to the macOS clipboard. Paste it into the authenticated Crypto Wizards pair-page browser console, run `await __CW_CAPTURE_STATUS__()` once, click the pair page refresh/recalculate icon, then run `await __CW_CAPTURE_STATUS__()` again. The status table reports `fetches`, `xhrs`, `worker_messages`, and `wasm_extracts`; any increase in those counts is useful evidence. If the status shows useful payloads, run `await __CW_DOWNLOAD_CAPTURE__()`. The `import-latest-pair-detail-download` command pulls the newest matching capture from your Downloads folder, archives it under `data/raw/pair_details/`, refreshes reports, and prints the same readiness checks.

The preflight command writes `reports/pair_detail_capture_preflight.csv` without archiving the candidate capture. The inspect command prints readiness without writing files. Use one of those first to confirm the capture contains the needed history. The importer archives the capture under `data/raw/pair_details/`, refreshes pair-detail reports, and prints whether the payload is baseline experiment-ready, ECM-history-ready, and two-leg execution-ready. The checklist command prints the current archived capture queue with missing required fields and the next capture focus. These commands report nested candidate JSON paths from the capture audit so hidden worker/fetch payloads can be identified quickly.

In the browser console, run `await __CW_DOWNLOAD_CAPTURE__()` after installing `scripts/capture_crypto_wizards_pair_detail.js` and refreshing/recalculating the pair page. The helper captures fetch/XHR responses, passive worker-to-page messages with worker IDs and script URLs, research-looking storage, inline scripts, resource URLs, and readable IndexedDB stores. Fetch responses include bounded text plus best-effort JSON parsing even when the server does not advertise a JSON content type.

For higher trade count research, use 5-minute dYdX candle responses from the Crypto Wizards pair page:

```bash
PYTHONPATH=src python3 -m quant_platform.cli dydx-two-leg-request-template \
  --pair BNB-USD-STX-USD \
  --pair-id 1 \
  --hedge-ratio 1.36 \
  --beta 1.36
PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candles --json-path /path/to/BNB-USD-5min-response.txt
PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candles --json-path /path/to/STX-USD-5min-response.txt
PYTHONPATH=src python3 -m quant_platform.cli build-dydx-pair-history \
  --left-candles data/raw/dydx_candles/BNB-USD_5MINS_candles.json \
  --right-candles data/raw/dydx_candles/STX-USD_5MINS_candles.json \
  --asset-x BNB-USD \
  --asset-y STX-USD \
  --pair-id 1 \
  --interval 5mins \
  --hedge-ratio 1.36 \
  --beta 1.36 \
  --zscore-window 320
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-quality
PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments
```

The request-template command writes `reports/dydx_two_leg_data_requests.csv` with ready-to-copy dYdX indexer candle and historical funding URLs for both legs, plus the local import/build/funding merge commands.

Defaults:
- `--indexer-base` falls back to `QPA_INDEXER_BASE` or `https://indexer.dydx.trade`.
- `--indexer-scheme` falls back to `QPA_INDEXER_SCHEME` when set, so `--indexer-scheme` output and fetched URLs honor an environment toggle (e.g., `http`).
- For testnet evidence, pass `--indexer-base https://indexer.v4testnet.dydx.exchange`.

If the current machine can reach the public dYdX indexer, use the one-shot fetch path instead of downloading each URL manually:

```bash
PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-two-leg-data \
  --pair BNB-USD-STX-USD \
  --pair-id 1 \
  --hedge-ratio 1.36 \
  --beta 1.36 \
  --limit 100 \
  --indexer-scheme https \
  --run-research
```

This fetches both 5-minute candle legs and both historical funding payloads, builds the two-leg pair history, normalizes funding into `data/processed/dydx_funding.csv`, checks funding coverage, and then runs the guarded funded research spine when `--run-research` is present. Candle-derived histories do not fabricate funding columns; production acceptance requires real funding merged through `--funding-path`.

If DNS is blocked on this machine, you can harden the indexer path without editing code by setting:

```bash
export QPA_INDEXER_BASES="https://indexer.dydx.trade,https://indexer.v4testnet.dydx.exchange"
export QPA_INDEXER_HOST_IP_HINTS="indexer.dydx.trade:172.66.166.30,104.20.40.161"
export QPA_DISABLE_SCHEME_FALLBACK=1  # optional: only use configured URL scheme (as configured)
export QPA_INDEXER_SCHEME=http        # optional: force all indexer URL fetches to http
export QPA_USE_REQUESTS_FETCH=false    # optional: skip Python requests fetch attempt, use curl transport directly
```
You can pass the same scheme override directly in shell fetch scripts:

```bash
QPA_INDEXER_SCHEME=http \
  bash scripts/fetch_dydx_two_leg_shell.sh --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --indexer-scheme http
QPA_INDEXER_SCHEME=http \
  bash scripts/run_dydx_long_history.sh --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --indexer-scheme http

For air-gapped/manual payload workflows, use `data/raw/dydx_inbox/` as a staging area and then use the same shell runner in `--skip-fetch` mode after placing these four files into `data/raw/dydx_manual/`:
- `SOL-USD_5MINS_candles.json`
- `SOL-USD_funding.json`
- `LINK-USD_5MINS_candles.json`
- `LINK-USD_funding.json`

```bash
bash scripts/fetch_dydx_two_leg_shell.sh \
  --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link \
  --out-dir data/raw/dydx_manual \
  --skip-fetch \
  --indexer-scheme http \
  --funding-path data/processed/dydx_funding.csv
```

If you only have the payloads and want to run via CLI directly, use `fetch-dydx-two-leg-data` in offline mode:

```bash
PYTHONPATH=src python3 -m quant_platform.cli fetch-dydx-two-leg-data \
  --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link \
  --download-dir data/raw/dydx_manual --skip-fetch \
  --funding-path data/processed/dydx_funding.csv \
  --derive-hedge-ratio --run-research
```
```

Use Apify MCP actor outputs as the standard way to populate funding and candle artifacts
before ingesting them through existing CLI importers. Common actor endpoints in the
Apify ecosystem are:

```bash
parseforge/dydx-v4-perpetual-markets-scraper
parseforge/dydx-markets-scraper
fraktalapi/funding-pulse
api_merge/coinglass-coin-markets
parseforge/hyperliquid-perp-funding-scraper
parseforge/gmx-arbitrum-stats-scraper
parseforge/gmx-arbitrum-prices-scraper
louisdeconinck/coinmarketcap-crypto-scraper
real1ty/coingecko
muhammetakkurtt/coinmarketcap-scraper
muhammetakkurtt/arkham-intelligence-wallet-data-scraper
gentle_cloud/cryptocurrency-market-data-scraper
moving_beacon-owner1/my-actor-14
muhammetakkurtt/dexscreener-scraper
muhammetakkurtt/dexscreener-realtime-monitor
```

After downloading actor JSON output, place candle payloads at `data/raw/dydx_manual/*.json`
and run:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candles --json-path /path/to/market.json
PYTHONPATH=src python3 -m quant_platform.cli export-dydx-funding --json-path /path/to/funding.json --market SOL-USD --output-path data/processed/dydx_funding.csv
PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments --funding-path data/processed/dydx_funding.csv
```

For a project-level Apify setup guide and repo-specific staging flow, see:

```text
docs/apify_integration.md
```

The code attempts each configured base, then each configured/fallback IP with `Host`-header preservation and TLS fallback before curl fallback.

In Chrome DevTools Network, switch the Crypto Wizards pair page interval to `5 Min`, filter requests by `candles`, and copy the `Response` body from the `200` requests for each leg. The importer accepts either full JSON such as `{"candles": [...]}` or the pasted object-list fragment Chrome displays in the Response tab. The generated pair history includes local spread/z-score reconstruction and provisional ECM estimates; native Crypto Wizards ECM remains better evidence when available. The quality report separates research usability from execution usability, so derived candles can be used for discovery while placeholder funding remains blocked from paper execution.

To capture visible scanner pairs in one browser pass, paste `scripts/capture_crypto_wizards_5min_pair_bundle.js` into the authenticated Crypto Wizards scanner console, then run:

```js
await __CW_5MIN_PAIRS__()
await __CW_DOWNLOAD_5MIN_BUNDLE__({ maxPairs: 10, limit: 100 })
```

Import the downloaded bundle with:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candle-bundle --json-path /path/to/crypto_wizards_5min_pair_bundle.json
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-quality
PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments
PYTHONPATH=src python3 -m quant_platform.cli strategy-acceptance-checklist
```

This bundle route uses the browser session to discover visible Crypto Wizards pairs, then fetches `5MINS` candles directly from the dYdX indexer for both legs. It is the preferred manual bridge until a server-side API endpoint for all pair detail histories is available.

If the Codex shell cannot resolve DNS but your browser can download the API JSON, save it outside the repo and import it:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-crypto-wizards-payload --json-path /path/to/downloaded.json --endpoint-name prescanned
PYTHONPATH=src python3 -m quant_platform.cli verify-crypto-wizards-live-artifacts
```
