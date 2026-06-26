# Apify Integration

This project uses Apify MCP as the preferred dYdX market/funding acquisition path.
Direct dYdX indexer fetches remain available as a diagnostic or recovery path, but they
are not the default operating assumption for new refreshes.

## Operating Rule

For all future refreshes, use this order:

1. Apify MCP
2. Local cache
3. Direct dYdX indexer fallback

Do not start a new research refresh by calling `indexer.dydx.trade` directly. The direct
indexer is only for diagnostics, cache rebuild fallback, or explicit user-approved
recovery. This prevents DNS or public-indexer connectivity issues from blocking pair
selection and strategy validation.

If a pair needs fresh candles or funding, the blocker should read:

```text
waiting_for_apify_mcp_refresh
```

not:

```text
waiting_for_dydx_indexer
```

## What this connection is for

Use Apify to collect market/funding payloads for:
- `SOL-USD`
- `LINK-USD`
- `ETH-USD`
- `BTC-USD`

Then stage those payloads into the repo's existing offline/manual ingestion path.

This project does **not** use Apify as a separate analytics engine. Apify is the primary
data acquisition layer; the local quant platform remains the research, acceptance,
execution-gating, and learning system.

## Primary Apify MCP server URL

```text
https://mcp.apify.com/?tools=actors,docs,parseforge/dydx-v4-perpetual-markets-scraper,parseforge/dydx-markets-scraper,fraktalapi/funding-pulse,api_merge/coinglass-coin-markets,parseforge/hyperliquid-perp-funding-scraper,parseforge/gmx-arbitrum-stats-scraper,louisdeconinck/coinmarketcap-crypto-scraper,gentle_cloud/cryptocurrency-market-data-scraper,moving_beacon-owner1/my-actor-14,muhammetakkurtt/dexscreener-scraper,muhammetakkurtt/dexscreener-realtime-monitor
```

## Expanded Apify MCP server URL

The expanded source set requested on 2026-06-25 is:

```text
https://mcp.apify.com/?tools=actors,docs,muhammetakkurtt/arkham-intelligence-wallet-data-scraper,louisdeconinck/coinmarketcap-crypto-scraper,moving_beacon-owner1/my-actor-14,real1ty/coingecko,muhammetakkurtt/coinmarketcap-scraper,parseforge/gmx-arbitrum-stats-scraper,parseforge/gmx-arbitrum-prices-scraper
```

Use this source set as an additive MCP configuration for broader discovery and market-context enrichment. It does not replace the dYdX/funding actors above; the dYdX actors remain the execution-venue authority for dYdX acceptance.

Recommended MCP config:

```json
{
  "mcpServers": {
    "apify": {
      "url": "https://mcp.apify.com/?tools=actors,docs,parseforge/dydx-v4-perpetual-markets-scraper,parseforge/dydx-markets-scraper,fraktalapi/funding-pulse,api_merge/coinglass-coin-markets,parseforge/hyperliquid-perp-funding-scraper,parseforge/gmx-arbitrum-stats-scraper,louisdeconinck/coinmarketcap-crypto-scraper,gentle_cloud/cryptocurrency-market-data-scraper,moving_beacon-owner1/my-actor-14,muhammetakkurtt/dexscreener-scraper,muhammetakkurtt/dexscreener-realtime-monitor"
    }
  }
}
```

Expanded MCP config:

```json
{
  "mcpServers": {
    "apify-expanded": {
      "url": "https://mcp.apify.com/?tools=actors,docs,muhammetakkurtt/arkham-intelligence-wallet-data-scraper,louisdeconinck/coinmarketcap-crypto-scraper,moving_beacon-owner1/my-actor-14,real1ty/coingecko,muhammetakkurtt/coinmarketcap-scraper,parseforge/gmx-arbitrum-stats-scraper,parseforge/gmx-arbitrum-prices-scraper"
    }
  }
}
```

Use OAuth when the MCP client supports it. If a bearer token is required, keep it in
private local configuration and never commit it to project examples.

## Recommended actors

- `parseforge/dydx-v4-perpetual-markets-scraper`
- `parseforge/dydx-markets-scraper`
- `fraktalapi/funding-pulse`
- `api_merge/coinglass-coin-markets`
- `parseforge/hyperliquid-perp-funding-scraper`
- `parseforge/gmx-arbitrum-stats-scraper`
- `louisdeconinck/coinmarketcap-crypto-scraper`
- `gentle_cloud/cryptocurrency-market-data-scraper`
- `moving_beacon-owner1/my-actor-14`
- `muhammetakkurtt/dexscreener-scraper`
- `muhammetakkurtt/dexscreener-realtime-monitor`

## Expanded actor roles

| Actor | Role | Promotion authority | Notes |
| --- | --- | --- | --- |
| `muhammetakkurtt/arkham-intelligence-wallet-data-scraper` | Wallet/entity intelligence, whale flow, concentration and risk context. | No | Useful for risk overlays, not live signal authority without point-in-time capture. |
| `louisdeconinck/coinmarketcap-crypto-scraper` | Broad market cap, volume, price and ranking context. | No | Already available as a dedicated Apify MCP tool in the current session. |
| `moving_beacon-owner1/my-actor-14` | KuCoin OHLCV and real-time prices. | No | Already available as a dedicated Apify MCP tool in the current session; useful for cross-venue history/context. |
| `real1ty/coingecko` | CoinGecko market, token, category and liquidity context. | No | Use via generic Apify actor call unless a dedicated tool is exposed. |
| `muhammetakkurtt/coinmarketcap-scraper` | Alternate CoinMarketCap scrape path. | No | Use as a redundancy check against the Louis Deconinck CoinMarketCap actor. |
| `parseforge/gmx-arbitrum-stats-scraper` | GMX Arbitrum derivatives stats. | No | Useful for GMX venue context and cross-venue discovery. |
| `parseforge/gmx-arbitrum-prices-scraper` | GMX Arbitrum price feed/context. | No | Useful for GMX venue price context; not dYdX execution authority. |

## Source routing rule

Use the expanded actors to improve discovery, liquidity context, market regime context, and venue selection.

Do not use them to promote a dYdX strategy unless the pair also passes:

- dYdX market availability
- dYdX liquidity/funding checks
- Daily local dYdX replay when the Wizard candidate is Daily
- after-cost acceptance gates
- enough closed trades

## Project wiring

Environment placeholders are in:

```text
.env.example
```

Key values:
- `APIFY_API_TOKEN`
- `APIFY_MCP_SERVER_URL`
- `APIFY_DYDX_MARKETS_ACTOR`
- `APIFY_DYDX_FUNDING_ACTOR`

## Data path

1. Run the configured Apify actor through MCP and download/export the actor output.
2. Stage raw files in:

```text
data/raw/dydx_inbox/
```

3. Rename approved payloads into the repo's expected format:
- `<MARKET>_5MINS_candles.json`
- `<MARKET>_funding.json`

4. Move the approved files into:

```text
data/raw/dydx_manual/
```

5. Run the existing offline/manual rebuild path.

## Current primary ingestion flow

For funding snapshots:

```bash
PYTHONPATH=src python3 -m quant_platform.cli export-dydx-funding \
  --json-path data/raw/dydx_manual \
  --output-path data/processed/dydx_funding.csv
```

For pair rebuilds from manual payloads:

```bash
bash scripts/fetch_dydx_two_leg_shell.sh \
  --asset-x SOL-USD --asset-y LINK-USD --pair-id sol_link \
  --out-dir data/raw/dydx_manual \
  --skip-fetch \
  --funding-path data/processed/dydx_funding.csv
```

## Important constraint

Apify output is accepted into the primary ingestion path when exported payloads include:
- aligned timestamps
- usable market identifiers
- per-market funding values or fields the normalizer already understands

The project already accepts several non-indexer funding field names, including:
- `nextFundingRate`
- `next_funding`
- `scrapedAt`
- `timeMs`

That makes Apify snapshots valid first-class funding inputs for the local import path.
