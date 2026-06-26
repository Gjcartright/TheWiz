# Strategy Family Sweep Notes

## Pair Pack

- `BTC-USD-SOL-USD`
- `DOGE-USD-SOL-USD`
- `SOL-USD-XRP-USD`
- `SOL-USD-LINK-USD`

## Sweep Readout

- strategies_run: 37
- families_seen: 10
- production_eligible_strategies: 0
- preferred_eligible_strategies: 0

## Best Families

- `mean_reversion` -> `Hurst Filter` (passing_pairs=0, sharpe=0.106, pf=inf, dd=0.152)
- `composite` -> `Composite Quant Score` (passing_pairs=0, sharpe=0.000, pf=inf, dd=0.000)
- `copula` -> `Copula Risk Filter` (passing_pairs=0, sharpe=0.000, pf=inf, dd=0.000)
- `ecm` -> `Pure ECM` (passing_pairs=0, sharpe=0.000, pf=inf, dd=0.000)
- `hybrid` -> `ZScore + ECM` (passing_pairs=0, sharpe=0.000, pf=inf, dd=0.000)

## Promotion Shortlist

- `mean_reversion` / `Hurst Filter` because `top_family_placeholder`
- `composite` / `Composite Quant Score` because `top_family_placeholder`
- `copula` / `Copula Risk Filter` because `top_family_placeholder`
