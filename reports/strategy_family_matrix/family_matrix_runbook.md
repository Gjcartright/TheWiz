# Family Matrix Runbook

This workflow runs every family separately, then builds combo tests from the best strategy in each family.

## What you get

- a separate report folder for each family
- a best-strategy row for each family
- family-combo tests from size 2 through size 4
- one full-stack combo using all best families when the family count is larger than the max combo size

## Separate Family Ranking

- `composite` -> `Composite Quant Score` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `copula` -> `Copula Risk Filter` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `ecm` -> `Pure ECM` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `hybrid` -> `ZScore + ECM` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `ml` -> `Feature Importance Model` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `portfolio` -> `Pair Ranking Strategy` (passing_pairs=0, sharpe=0.000, pf=0.000, trades=0)
- `mean_reversion` -> `Hurst Filter` (passing_pairs=0, sharpe=-0.703, pf=0.000, trades=16)
- `regime` -> `KMeans Regime Model` (passing_pairs=0, sharpe=-0.800, pf=0.375, trades=44)
- `adaptive` -> `Dynamic Threshold Model` (passing_pairs=0, sharpe=-1.348, pf=0.315, trades=60)
- `zscore` -> `Classic ZScore Mean Reversion` (passing_pairs=0, sharpe=-1.654, pf=0.275, trades=60)

## Top Combo Rows

- `full_stack_376` [adaptive;composite;copula;ecm;hybrid;mean_reversion;ml;portfolio;regime;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=180)
- `combo_4way_264` [composite;copula;mean_reversion;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_279` [composite;ecm;mean_reversion;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_289` [composite;hybrid;mean_reversion;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_298` [composite;mean_reversion;ml;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_300` [composite;mean_reversion;portfolio;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_314` [copula;ecm;mean_reversion;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_324` [copula;hybrid;mean_reversion;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_333` [copula;mean_reversion;ml;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)
- `combo_4way_335` [copula;mean_reversion;portfolio;zscore] (passing_pairs=0, sharpe=0.000, pf=0.000, trades=76)

## Combo Sizes

- `2`-way combos: 45
- `3`-way combos: 120
- `4`-way combos: 210
- `10`-way combos: 1
