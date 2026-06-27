# Strategy Family Failure Attribution

## Dominant Family Failures

- `composite` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Composite Quant Score`, blockers=`median_sharpe:3;passing_pairs:3;total_trades:3;two_leg_execution_input_pairs<2:3;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:3`)
- `copula` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Copula Risk Filter`, blockers=`median_sharpe:7;passing_pairs:7;total_trades:7;two_leg_execution_input_pairs<2:7;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:7`)
- `ecm` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Leader/Follower Prediction`, blockers=`median_sharpe:3;passing_pairs:3;total_trades:3;two_leg_execution_input_pairs<2:3;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:3`)
- `hybrid` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`ZScore + ECM`, blockers=`median_sharpe:4;passing_pairs:4;total_trades:4;two_leg_execution_input_pairs<2:4;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:4`)
- `ml` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Trade Outcome Predictor`, blockers=`median_sharpe:5;passing_pairs:5;total_trades:5;two_leg_execution_input_pairs<2:5;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:5`)
- `portfolio` -> `too_few_trades_and_no_passing_pairs` (best_strategy=`Risk Adjusted Ranking`, blockers=`median_sharpe:5;passing_pairs:5;total_trades:5;two_leg_execution_input_pairs<2:5;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:5`)
- `mean_reversion` -> `no_passing_pairs` (best_strategy=`Hurst + Half-Life`, blockers=`median_sharpe:4;passing_pairs:4;total_trades:4;two_leg_execution_input_pairs<2:4;two_leg_missing_inputs:BTC-USD-SOL-USD[funding_x+funding_y]:4`)
- `regime` -> `no_passing_pairs` (best_strategy=`KMeans Regime Model`, blockers=`median_profit_factor:4;median_sharpe:4;passing_pairs:4;total_trades:4;two_leg_execution_input_pairs<2:4`)
- `adaptive` -> `no_passing_pairs` (best_strategy=`Dynamic Threshold Model`, blockers=`median_profit_factor:1;median_sharpe:1;passing_pairs:1;total_trades:1;two_leg_execution_input_pairs<2:1`)
- `zscore` -> `no_passing_pairs` (best_strategy=`Classic ZScore Mean Reversion`, blockers=`median_profit_factor:1;median_sharpe:1;passing_pairs:1;total_trades:1;two_leg_execution_input_pairs<2:1`)
