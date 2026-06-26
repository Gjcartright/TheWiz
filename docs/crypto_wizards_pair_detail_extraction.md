# Crypto Wizards Pair Detail Extraction

## Current Finding

`/v1beta/prescanned` is a scanner surface, not the full research surface. The authenticated dashboard pair route exposes richer pair-level research data:

```text
https://cryptowizards.net/wizards/zscore/pair/<pair_id>?origin=scanner
```

The captured dashboard page for `pair/1` includes:

- `ecm (y)` chart option with value `ecm_y`
- `ecm (x)` chart option with value `ecm_x`
- `ecm strength` chart option with value `ecm_strength`
- `ECM Deviation (min)%` backtest override
- copula family and conditional probabilities
- Pearson, Spearman, Kendall
- hedge ratio, Hurst, backtest metrics, VaR, CVaR, drawdown

Archived current snapshot:

```text
data/raw/pair_details/pair_1_dashboard_snapshot.json
```

Generated reports:

```text
reports/pair_detail_research_snapshots.csv
reports/pair_detail_field_dictionary.csv
reports/pair_detail_history_coverage.csv
reports/pair_detail_capture_audit.csv
```

## Internal Route Clues

The pair page uses this route chunk:

```text
/_build/assets/_id_-CjQlYwQE.js
```

That chunk imports:

```text
/_build/assets/zscore_library-IlN0_w2C.js
```

The pair route chunk calls a function imported as `Ct(...)` with:

```text
symbol_1
symbol_2
exchange
interval
period
spread_id
strategy_id
btSettingsCA
```

The callback recognizes progress states:

```text
completed_prices
completed_econometrics
completed_garch
completed_ecm
completed_backtest
```

This strongly suggests the dashboard computes or fetches price, spread, ECM, and backtest data through the `zscore_library` chunk.

## Remaining Extraction Gap

The current browser-visible snapshot proves ECM availability but does not expose raw historical arrays for:

- `spread`
- `zscore`
- `ecm_x`
- `ecm_y`
- `ecm_strength`

Until those arrays are captured, the pair-detail snapshot is research metadata, not an experiment-ready backtest dataset.

The local importer accepts these payload shapes:

- `history`: list of row dictionaries
- `series`: list of row dictionaries
- nested `viewItem.history` or `viewItem.series`
- parallel arrays inside `viewItem`, `result`, `data`, or the root payload, such as `spread`, `zscore`, `ecm_x`, `ecm_y`, `ecm_strength`
- browser capture fields such as `fetches`, `xhrs`, `worker_messages`, `storage`, `indexeddb`, and inline `scripts`
- DevTools HAR-style exports where JSON is stored under `log.entries[].response.content.text`

`reports/pair_detail_history_coverage.csv` reports:

- `experiment_ready`: true when `spread` and `zscore` history exist
- `ecm_history_ready`: true when `ecm_x`, `ecm_y`, and `ecm_strength` history also exist
- `two_leg_execution_ready`: true when `price_x` and `price_y` are also present so the two-leg backtester can use hedge ratio, beta, funding, slippage, execution-risk, and partial-fill assumptions
- `execution_assumption_notes`: defaults being used, such as `beta_default_1.0` or `funding_cost_model_default`

`reports/pair_detail_capture_audit.csv` recursively scans the imported dashboard capture and reports candidate JSON paths, candidate type, row count, columns, and readiness flags. Use it when a browser capture contains nested `fetches`, `xhrs`, `worker_messages`, `storage`, `indexeddb`, inline `scripts`, `viewItem`, `result`, or `data` payloads and the first-pass importer still says no history was detected.

`reports/pair_detail_capture_checklist.csv` combines top-level history coverage and nested candidate audit data into one row per archived capture. It reports found required fields, missing required fields, grouped missing baseline/ECM/two-leg/execution-assumption fields, a 0-100 `capture_completeness_score`, capture source counts such as `capture_fetches`, `capture_xhrs`, `capture_worker_messages`, and `capture_payload_sources`, the best candidate JSON path, whether the capture is import-ready, whether it is research-spine-ready, and the next capture focus. It also includes `required_field_locations` and `execution_assumption_locations`, which map each discovered field to `history`, `snapshot`, or the nested JSON path where the field was found. The `capture_operator_hint` column distinguishes a true browser capture from a static dashboard snapshot; `not_a_browser_capture` means paste the helper into the authenticated pair page, click refresh/recalculate, run `await __CW_CAPTURE_STATUS__()`, then download with `await __CW_DOWNLOAD_CAPTURE__()`.

## Preferred Next Target

In an authenticated browser session:

1. Open a pair detail page.
2. Paste `scripts/capture_crypto_wizards_pair_detail.js` into the browser console.
3. Click the pair page refresh/recalculate control.
4. Run `await __CW_DOWNLOAD_CAPTURE__()` in the browser console.
5. Inspect the downloaded JSON before archiving it:

```bash
PYTHONPATH=src python3 -m quant_platform.cli capture-preflight --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli inspect-pair-detail-capture --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli import-pair-detail-capture --json-path /path/to/crypto_wizards_pair_1_capture.json
PYTHONPATH=src python3 -m quant_platform.cli pair-detail-capture-checklist
```

The preflight command writes `reports/pair_detail_capture_preflight.csv` without archiving the candidate capture. The inspect command prints the same readiness summary without creating `data/raw/pair_details/` files or report files. Import only captures that contain useful history. The import command archives the capture under `data/raw/pair_details/`, refreshes reports, and prints:

- `history_rows_detected`
- `experiment_ready`
- `ecm_history_ready`
- `two_leg_execution_ready`
- missing baseline fields
- missing ECM fields
- missing two-leg fields
- execution assumption notes
- number of nested candidate paths
- experiment-ready JSON paths
- ECM-ready JSON paths
- found required fields
- missing required fields
- next capture focus
- two-leg-ready JSON paths

The capture helper records `fetch`, `XMLHttpRequest`, worker messages, worker script URLs, matching resource URLs, research-looking inline scripts, research-looking IndexedDB stores, and research-looking local/session storage keys. Fetch responses store bounded response text and attempt JSON parsing even when the content type is not JSON, because dashboard APIs sometimes return useful payloads with loose headers. The helper also writes `capture_summary` counts into the downloaded JSON so preflight can immediately show whether network, worker, storage, script, or resource payloads were captured. After clicking refresh/recalculate on the pair page, run `await __CW_CAPTURE_STATUS__()` in the browser console to print payload counts, required field hits, missing field groups, the next capture focus, and `capture_operator_hint` without downloading a file. For a console checklist, run `await __CW_CAPTURE_RUNBOOK__()`. `capture_summary.field_quality` gives the same browser-side hint in the downloaded capture before Python import. Worker messages include passive worker-to-page messages plus `worker_id` and `script_url` metadata so ECM or history payloads computed off the main thread can be traced back to the worker source. Sensitive storage values such as auth tokens and private keys are redacted. If the capture contains worker messages with the full `viewItem` output, preserve that under `viewItem` or leave it in `worker_messages`; the local importer can inspect both shapes.

For production-style research, prefer captures that include:

- `spread`
- `zscore`
- `price_x`
- `price_y`
- `ecm_x`
- `ecm_y`
- `ecm_strength`
- `hedge_ratio`
- `beta` if available
- `funding_x_bps` and `funding_y_bps` if available

The importer also recognizes common aliases, including:

- `asset_x_prices`, `symbol1_prices`, `symbol_1_closes`, `series_1_closes`, `close_x`
- `asset_y_prices`, `symbol2_prices`, `symbol_2_closes`, `series_2_closes`, `close_y`
- `pair_beta`, `pair_betas`, `beta_pair`
- `asset_x_funding_bps`, `symbol1_funding_bps`, `symbol_1_funding_rate`
- `asset_y_funding_bps`, `symbol2_funding_bps`, `symbol_2_funding_rate`

If the helper only captures requests and worker inputs, inspect loaded route chunks and the `zscore_library` implementation next. The target is the function behind `Ct(...)`, because the route chunk assigns its return value to `viewItem`.

As a fallback, export a browser Network HAR from the authenticated pair-detail page and import it with the same command:

```bash
PYTHONPATH=src python3 -m quant_platform.cli import-pair-detail-capture --json-path /path/to/crypto_wizards_pair_detail.har
```

The importer parses JSON strings inside HAR response content and the capture audit reports paths with `#json` when a candidate came from decoded text.

Expected payload shape for the local harness:

```json
{
  "pair_id": "1",
  "pair": "BNB-USD-STX-USD",
  "asset_x": "BNB-USD",
  "asset_y": "STX-USD",
  "exchange": "dydx",
  "hedge_ratio": 1.36,
  "viewItem": {
    "spread": [-0.2, -0.1],
    "zscore": [-2.2, -1.1],
    "ecm_x": [-0.4, -0.2],
    "ecm_y": [-0.1, -0.05],
    "ecm_strength": [0.7, 0.7]
  },
  "history": [
    {
      "spread": -0.2,
      "zscore": -2.2,
      "ecm_x": -0.4,
      "ecm_y": -0.1,
      "ecm_strength": 0.7
    }
  ]
}
```

With that history present, run:

```bash
PYTHONPATH=src python3 -m quant_platform.cli ingest-pair-details
PYTHONPATH=src python3 -m quant_platform.cli run-pair-detail-experiments
```
