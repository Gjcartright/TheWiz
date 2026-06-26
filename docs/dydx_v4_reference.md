# dYdX v4 Execution Reference

Source:

- https://github.com/dydxprotocol
- https://github.com/dydxprotocol/v4-clients
- https://github.com/dydxprotocol/v4-clients/tree/main/v4-client-py-v2

## Official Client

The relevant Python package for dYdX Chain v4 is:

```bash
pip install dydx-v4-client
```

The Python import namespace is:

```python
import dydx_v4_client
```

This repo exposes the dependency as an optional install extra:

```bash
pip install -e ".[dev,dydx]"
```

## Network Setup

The dYdX Python client docs say testnet can use the predefined `TESTNET` network, or a custom testnet via `make_testnet`.

The docs also warn that `node_url` should not include an `https://` prefix when using the SDK network builder.

Current scaffold defaults:

```text
node_url=oegs-testnet.dydx.exchange:443
rest_indexer=https://indexer.v4testnet.dydx.exchange
websocket_indexer=wss://indexer.v4testnet.dydx.exchange/v4/ws
faucet_url=https://faucet.v4testnet.dydx.exchange
```

Always confirm the latest public endpoints from dYdX network resources before using real funds or testnet automation.

## Local Env Values

Fill these in `.env.local`:

```env
DYDX_TESTNET_WALLET_ADDRESS=
DYDX_TESTNET_PRIVATE_KEY=
DYDX_TESTNET_SUBMIT_ORDERS=false
```

Keep `DYDX_TESTNET_SUBMIT_ORDERS=false` until:

- research acceptance gates pass,
- the account is funded on testnet,
- the official `dydx-v4-client` package is installed,
- an authenticated client adapter is explicitly injected,
- paper execution is monitored through `reports/paper_trading_journal.csv`.

## Local Readiness Check

Run:

```bash
python -m quant_platform.cli check-dydx-config
python -m quant_platform.cli dydx-execution-checklist
```

The config command reports whether credentials are present, whether `dydx_v4_client` is importable, whether the dYdX indexer adapter is wired, whether an authenticated order-client adapter is wired, and which blockers remain. It does not print private keys.

The checklist command writes `reports/dydx_execution_checklist.csv`. It splits readiness into indexer market data, testnet credentials, SDK availability, submit flag, order-client adapter, research acceptance, and the final paper-submission gate.

The platform separates read and write readiness:

- `dydx_indexer_adapter_wired`: market/funding reads can use the official v4 indexer client.
- `dydx_order_client_adapter_wired`: signed testnet order placement has an injected authenticated client.
- `ready_for_paper_submission`: true only when credentials, submit flag, SDK availability, and order adapter wiring all pass.

The order adapter is loaded from `DYDX_TESTNET_ORDER_CLIENT_ADAPTER=module:object`. The target object, factory, or zero-argument class must implement `place_order(intent, config) -> FillReport`. Keeping this as an explicit local adapter prevents the paper engine from silently faking exchange-side fills.

Validate the adapter contract before enabling submission with `python -m quant_platform.cli dydx-order-adapter-contract`. The command writes `reports/dydx_order_adapter_contract.csv`, confirms the adapter imports, and checks that `place_order` accepts both `intent` and `config` without placing an order.

For local handoff and journal tests, this repo includes a record-only adapter:

```env
DYDX_TESTNET_ORDER_CLIENT_ADAPTER=quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter
```

It satisfies the adapter contract and returns fills with `status=paper_recorded_not_submitted`, but it does not sign, send, or verify dYdX testnet orders. The readiness checklist reports it as `record_only_dydx_order_client_adapter`, so it cannot unlock exchange-side paper submission. Use it to test the research-to-execution handoff only; replace it with an authenticated dYdX client adapter before claiming exchange-side paper execution.

Historical funding from the indexer can be exported to CSV or JSON and supplied to research runs with `--funding-path`. Accepted columns include `market`, `ticker`, or `symbol`; funding values as `funding_bps`, `funding_rate_bps`, `rate_bps`, `rate`, or `funding_rate`; and optional timestamps such as `timestamp`, `effective_at`, or `effectiveAt`. Decimal funding rates are converted to basis points before they are merged into `funding_x_bps` and `funding_y_bps`.

### Scheme Control (HTTP vs HTTPS)

If you need deterministic scheme behavior on a constrained environment, force a single scheme and disable fallback:

```bash
QPA_INDEXER_SCHEME=http QPA_DISABLE_SCHEME_FALLBACK=1 PYTHONPATH=src /opt/anaconda3/bin/python3 -m quant_platform.cli dydx-two-leg-request-template --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link --limit 1000 --indexer-base https://indexer.dydx.trade
```

That command (and fetch/build commands using `--indexer-scheme`) now keep to the requested scheme only.

If Python DNS is flaky in your environment but `curl` can still resolve external hosts, force curl transport first:

```bash
QPA_USE_REQUESTS_FETCH=false QPA_INDEXER_SCHEME=http PYTHONPATH=src /opt/anaconda3/bin/python3 -m quant_platform.cli fetch-dydx-two-leg-data --pair SOL-USD-LINK-USD --pair-id sol_link --limit 1000
```

### Alternative Funding Import Source (non-indexer)

If the public dYdX indexer is unreachable from your environment (for example DNS or external access restrictions), you can still run P2 funding coverage by feeding official/offline snapshots into the same funding normalizer.

- Any JSON with funding-rate rows can be imported with `export-dydx-funding`.
- The normalizer now accepts common alternative field names, including:
  - Market: `market`, `ticker`, `symbol`, `marketId`, `market_id`, `productId`, `product_id`
  - Value: `funding_rate`, `rate`, `nextFundingRate`, `next_funding`, `next_funding_rate`
  - Time: `scrapedAt`, `nextFundingTime`, `timeMs`, plus standard indexer time fields

This is the normal import path for Apify-backed snapshots after exporting the payload JSON locally. The primary MCP configuration includes the dYdX actors plus broader crypto-market sources such as Coinglass, Hyperliquid funding, GMX stats, CoinMarketCap, DexScreener, and general cryptocurrency market-data scrapers.

When the official indexer adapter is available, fetch and normalize funding directly:

```bash
python -m quant_platform.cli fetch-dydx-funding \
  --market ETH-USD,BTC-USD,SOL-USD \
  --output-path data/processed/dydx_funding.csv
```

For a saved raw indexer response, convert it first:

```bash
python -m quant_platform.cli export-dydx-funding \
  --json-path /path/to/funding_payload.json \
  --market ETH-USD \
  --output-path data/processed/dydx_funding.csv
```

For multiple saved market responses, place them in one directory with market names in the filenames, such as `ETH-USD_funding.json` and `BTC-USD_funding.json`, then run:

```bash
python -m quant_platform.cli export-dydx-funding \
  --json-path /path/to/funding_payloads \
  --output-path data/processed/dydx_funding.csv
```
