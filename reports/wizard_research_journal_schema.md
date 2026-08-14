# Wizard Research Journal Schema

This document defines the canonical capture shape for Crypto Wizards research.

The goal is to preserve enough context to:

- explain why a pair was selected
- replay the exact page-two setup later
- compare the same pair across timeframes and strategies
- label paper outcomes for ML / RL
- avoid losing information when the dashboard changes

## Capture Grain

We should store research at two linked grains:

1. `scanner_capture`
   - one row per highlighted pair seen on the scanner at a specific timestamp
2. `pair_detail_capture`
   - one row per `scanner_capture x timeframe x strategy` combination on page two

That means one scanner pick can expand into multiple page-two records:

- `Daily`
- `4 Hour`
- `1 Hour`
- `5 Min`

and, where available:

- `Static (Spread)`
- `Static (ZScoreR)`
- `Dyn (Spread)`
- `Dyn (ZScoreR)`
- `OU (Spread)`
- `OU (ZScoreR)`
- `Copula`

## Core Keys

Every record should carry these keys so all layers join cleanly:

- `capture_id`
- `capture_timestamp_utc`
- `scanner_refresh_timestamp_utc`
- `pair`
- `asset_x`
- `asset_y`
- `venue`
- `exchange_lane`
- `timeframe`
- `strategy_label`
- `page_route`
- `source_path`
- `evidence_path`
- `capture_context`
- `source_cycle_id`
- `source_run_type`

## Layer 1: Scanner Capture

This is the page-one packet. It explains why a pair entered the funnel.

### A. Session and filter context

- refresh pressed before scan
- scanner sort mode
- cointegration filter
- correlation filter
- hurst filter
- half-life filter
- copula filter
- strategy filter
- exchange filter
- symbol filter text
- row index on scanner
- visible row count at capture time

### B. Pair identity

- pair
- asset_x
- asset_y
- asset_x_raw
- asset_y_raw
- normalized_pair
- pair_id

### C. Dashboard metrics

- volume_x
- volume_y
- updated_at_utc
- strategy_family_from_row
- strategy_variant_from_row
- zscore_norm_value
- zscore_roll_value
- zscore_green_source
- dependency_profile
- dependency_x_over_y
- dependency_y_over_x
- correlation_value
- coint_johansen_flag
- coint_engle_granger_flag
- johansen_badge_state
- engle_granger_badge_state
- hurst_value
- half_life_value
- sigma_0_count
- sigma_1_count
- sigma_2_count
- var_99
- cvar_99
- max_drawdown
- return_total
- sharpe

### D. Raw evidence

- scanner_row_text_raw
- scanner_row_screenshot_path
- scanner_snapshot_path

## Layer 2: Pair Detail Capture

This is the page-two packet. It explains whether the pair is actually tradable and how the setup is configured.

Each record is one exact page-two state for one timeframe and one selected strategy.

### A. Pair detail context

- page_route
- pair_tab_id
- timeframe
- strategy_label
- periods_input
- periods_analyzed
- page_detail_screenshot_path
- page_detail_full_screenshot_path
- detail_capture_timestamp_utc
- changed_vs_last_capture
- change_reason

### B. Top strip metrics

- price_x
- price_y
- return_x_pct
- return_y_pct
- volume_x_top
- volume_y_top
- lt_vol_x
- lt_vol_y
- dependency_x_over_y_top
- dependency_y_over_x_top
- coint_johansen_top
- coint_engle_granger_top
- johansen_badge_state
- engle_granger_badge_state
- hurst_top
- half_life_top
- correlation_top
- hedge_ratio_top
- sigma_0_count_top
- sigma_2_count_top
- lt_beta
- max_drawdown_top
- return_total_top
- sharpe_top

### C. Main chart state

- chart_left_mode
- chart_left_series_x_label
- chart_left_series_y_label
- chart_left_latest_x_value
- chart_left_latest_y_value
- chart_right_mode
- chart_right_latest_value

### D. Strategy diagnostics

- ou_mu
- ou_alpha
- ou_beta
- ou_B
- ou_sigma

### E. Correlation block

- pearson_rho
- spearman_rho
- kendall_tau
- conditional_chart_value

### F. Copula block

- copula_best_fit
- copula_correlation_rho
- copula_x_given_y
- copula_y_given_x
- ecm_x_available
- ecm_y_available
- ecm_strength_available
- copula_chart_mode
- copula_direction_view
- copula_entry_lower
- copula_entry_upper
- copula_exit_lower
- copula_exit_upper

### G. Copula interpretation and stationarity legend

The journal must retain the dashboard's badge meaning, rather than reducing it
to a generic yes/no flag:

- green Johansen: confirmed
- green Engle-Granger: confirmed
- orange Engle-Granger: trending
- no colored badge: not_confirmed
- unavailable visual color: unknown

The derived Copula packet records:

- stationarity_summary
- copula_x_given_y_pct
- copula_y_given_x_pct
- copula_probability_gap_pct
- copula_signal_status
- copula_rich_asset
- copula_cheap_asset
- copula_trade_direction
- copula_entry_confirmation
- copula_exit_condition
- copula_journal_status
- copula_execution_blockers

An orange Engle-Granger state is a trend-sensitive research condition, not a
green stationarity confirmation. A Copula direction is therefore a paper
hypothesis until liquidity, costs, venue compatibility, and current-data
checks pass.

### H. Secondary diagnostic chart

- secondary_chart_mode
- secondary_chart_latest_value

### I. Trade rules

- entry_long_operator
- entry_long_value
- entry_long_unit
- entry_short_operator
- entry_short_value
- entry_short_unit
- exit_long_operator
- exit_long_value
- exit_long_unit
- exit_short_operator
- exit_short_value
- exit_short_unit

### J. Overrides

- close_n_periods_mode
- stop_loss_pct
- ecm_deviation_min_pct
- corr_strength_min_pct

### K. Capital weighting

- capital_weighting_slider_value
- capital_weighting_asset

### L. Backtest performance block

- sharpe_detail
- sortino_detail
- net_return_detail
- annualized_return_detail
- mean_period_return_detail
- win_rate
- closed_trades
- max_drawdown_detail
- var_99_detail
- cvar_99_detail
- var_sim
- cvar_sim

### L. Bottom chart state

- bottom_chart_mode
- bottom_chart_legend_mode
- bottom_chart_latest_net_value
- bottom_chart_latest_x_value
- bottom_chart_latest_y_value

## Required Repeats Across Timeframes

For each selected pair, the same page-two packet should be captured for:

- `Daily`
- `4 Hour`
- `1 Hour`
- `5 Min`

For each timeframe row, always store:

- which timeframe options were present
- whether the timeframe was checked
- whether values changed versus the last capture
- whether the page failed to load
- whether the pair disappeared before capture finished

## Recommended Derived Labels

These are not raw page values, but we should compute and store them because they are useful for ML / RL.

- `readiness_label`
  - `not_ready`
  - `near_trigger`
  - `triggered`
  - `invalidated`
- `zscore_signal_type`
  - `norm_only`
  - `roll_only`
  - `both`
  - `none`
- `risk_gate_pass`
- `return_gate_pass`
- `sharpe_gate_pass`
- `drawdown_gate_pass`
- `profit_factor_gate_pass`
- `paper_candidate_status`
- `execution_compatible`
- `account_state_blocked`
- `orphan_leg_blocker`

## Recommended Change Tracking

For every page-two record, store deltas from the previous matching record:

- `delta_zscore_norm`
- `delta_zscore_roll`
- `delta_correlation`
- `delta_hurst`
- `delta_half_life`
- `delta_hedge_ratio`
- `delta_return_total`
- `delta_sharpe`
- `delta_max_drawdown`
- `delta_capital_weighting`

Also track whether any trade-rule value changed:

- `rules_changed_flag`
- `overrides_changed_flag`
- `strategy_changed_flag`
- `timeframe_changed_flag`

## Outcome Linkage

Once a paper trade exists, link all journal records to outcome tracking with:

- `paper_trade_id`
- `paper_entry_timestamp_utc`
- `paper_exit_timestamp_utc`
- `paper_entry_notional_usd`
- `paper_exit_notional_usd`
- `paper_realized_return`
- `paper_outcome_label`
- `paper_exit_reason`

## Failure and Missing-State Fields

We should always store why something could not be captured.

- `capture_status`
- `capture_blocker`
- `pair_unavailable_flag`
- `timeframe_unavailable_flag`
- `strategy_unavailable_flag`
- `page_not_recognized_flag`
- `no_data_flag`
- `unchanged_alias_skipped_flag`

## Why This Structure Helps ML / RL

This structure gives us:

- a discovery layer from page one
- a configuration layer from page two
- a timeframe dimension
- a strategy dimension
- an outcome layer from paper trading

That means we can train on:

- what got selected
- what the exact setup looked like
- what changed before entry
- what actually won or lost later

This is the minimum structure needed to stop relying on memory and start building a reliable learning loop.
