# Crypto Wizards API Reference

Source:

- https://github.com/CryptoWizardsNet/cwizards-api-service-examples
- https://api.cryptowizards.net

## Auth

Crypto Wizards uses an API key header:

```text
X-api-key: <your_api_key>
Content-Type: application/json
```

## Base URL

```text
https://api.cryptowizards.net
```

## Documented GET Endpoints

These endpoints ask Crypto Wizards to provide the input market data and therefore use more credits.

| Name | Method | Path | Credits | Required Parameters |
| --- | --- | --- | --- | --- |
| backtest | GET | `/v1beta/backtest` | 6 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period`, `strategy` |
| cointegration | GET | `/v1beta/cointegration` | 5 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period` |
| copula | GET | `/v1beta/copula` | 5 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period` |
| correlations | GET | `/v1beta/correlations` | 5 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period` |
| credits_used | GET | `/v1beta/credits-used` | 0 | none beyond API key |
| prescanned | GET | `/v1beta/prescanned` | 10 | `priority`, `strategy` |
| spread | GET | `/v1beta/spread` | 5 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period` |
| zscores | GET | `/v1beta/zscores` | 5 | `symbol_1`, `symbol_2`, `exchange`, `interval`, `period` |

Useful prescanned example:

```env
CRYPTO_WIZARDS_ENDPOINTS=prescanned=/v1beta/prescanned?priority=Sharpe&strategy=Spread
```

## Documented POST Endpoints

These endpoints accept user-supplied price series and use fewer credits.

| Name | Method | Path | Credits | Required Body Fields |
| --- | --- | --- | --- | --- |
| backtest | POST | `/v1beta/backtest` | 2 | `params`, `bt_inputs` |
| cointegration | POST | `/v1beta/cointegration` | 1 | `series_1_closes`, `series_2_closes` |
| copula | POST | `/v1beta/copula` | 1 | `series_1_closes`, `series_2_closes` |
| correlations | POST | `/v1beta/correlations` | 1 | `series_1_closes`, `series_2_closes` |
| spread | POST | `/v1beta/spread` | 1 | `series_1_closes`, `series_2_closes` |
| zscores | POST | `/v1beta/zscores` | 1 | `series_1_closes`, `series_2_closes` |

## Fields Seen In Docs

Prescanned response examples include:

- `pair_id`
- `spread_id`
- `sym_1_volume`
- `sym_2_volume`
- `sym_1_volatility_lt`
- `sym_2_volatility_lt`
- `strategy_id`
- `spread_type`
- `strategy`
- `symbol_1`
- `symbol_2`
- `profile_match`
- `exchange`
- `interval`
- `period`
- `x_weighting`
- `y_weighting`
- `sharpe`
- `returns_total`
- `win_rate`
- `closed`
- `mdd`
- `var`
- `cvar`
- `johansen_coint`
- `coint_eg`
- `coint_eg_inc_trend`
- `coint_eg_p`
- `zero_cross`
- `stddev_cross`
- `hurst`
- `half_life`
- `hedge_ratio`
- `zscore_last`
- `zscore_roll_last`
- `zscore_window`
- `ou_optimal`
- `copula_id`
- `copula`
- `corr_copula`
- `u1_given_u2`
- `u2_given_u1`
- `ml_confidence`
- `mini_zscore`
- `backtest_ts`

Other endpoint examples include:

- `p_value`
- `t_stat`
- `cv`
- `is_coint`
- `inc_trend`
- `spread`
- `zscore`
- `zscore_roll`
- `last_zscore`
- `sigma0crossings`
- `sigma2crossings`
- `log_used`
- `kendall`
- `spearman`
- `pearson`
- `copula_name`
- `strat_returns`
- `annual_return`
- `mean_period_return`
- `total_return`
- `max_drawdown`
- `sharpe_ratio`
- `sortino_ratio`
- `bt_returns`
