# Venue Account Capability Evidence

Dynamic shadow controls read `reports/active/venue_account_capabilities.csv`.
This file is factual capability evidence, not a credential store. Never place
API keys, account balances, addresses, or secrets in it.

## Required Columns

```csv
schema_version,venue,account_eligible,product_type,supports_short_leg,captured_at_utc
venue_capabilities.v1,dydx,false,perp,true,2026-07-24T19:00:00Z
```

- `schema_version`: exactly `venue_capabilities.v1`.
- `venue`: normalized route venue, for example `dydx` or `hyperliquid`.
- `account_eligible`: whether the specific account is verified and allowed to
  use that venue/product. This is not inferred from public market data.
- `product_type`: such as `perp` or `spot`.
- `supports_short_leg`: whether the exact account/product can support the pair
  trade's short leg.
- `captured_at_utc`: timezone-aware observation time. It expires after 2.5
  hours for dynamic shadow routing.

Missing, malformed, unversioned, future-dated, or stale rows result in a
venue veto. A verified capability row only permits research-route validation;
it does not authorize paper or live trading.
