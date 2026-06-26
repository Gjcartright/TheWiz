# Strategy Family Failure Attribution

## Dominant Family Failures

- `mean_reversion` -> `no_passing_pairs` (best_strategy=`Hurst Filter`, blockers=`median_sharpe:4;passing_pairs:4;total_trades:4;worst_drawdown:4;median_profit_factor:2`)
- `composite` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Composite Quant Score`, blockers=`median_sharpe:3;passing_pairs:3;total_trades:3;median_profit_factor:1;worst_drawdown:1`)
- `copula` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Copula Risk Filter`, blockers=`median_sharpe:7;passing_pairs:7;median_profit_factor:6;worst_drawdown:6;total_trades:5`)
- `ecm` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Pure ECM`, blockers=`median_sharpe:3;passing_pairs:3;total_trades:3`)
- `hybrid` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`ZScore + ECM`, blockers=`median_sharpe:4;passing_pairs:4;total_trades:4;median_profit_factor:1;worst_drawdown:1`)
- `ml` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Trade Outcome Predictor`, blockers=`median_sharpe:5;passing_pairs:5;total_trades:5;worst_drawdown:4;median_profit_factor:3`)
- `portfolio` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Pair Ranking Strategy`, blockers=`median_sharpe:5;passing_pairs:5;total_trades:3;median_profit_factor:2;worst_drawdown:2`)
- `regime` -> `no_passing_pairs` (best_strategy=`KMeans Regime Model`, blockers=`median_profit_factor:4;median_sharpe:4;passing_pairs:4;total_trades:4;worst_drawdown:4`)
- `adaptive` -> `no_passing_pairs` (best_strategy=`Dynamic Threshold Model`, blockers=`median_profit_factor:1;median_sharpe:1;passing_pairs:1;total_trades:1;worst_drawdown:1`)
- `zscore` -> `no_passing_pairs` (best_strategy=`Classic ZScore Mean Reversion`, blockers=`median_profit_factor:1;median_sharpe:1;passing_pairs:1;total_trades:1;worst_drawdown:1`)
