# Live Two-Hour Dashboard Run

Every two-hour run begins with a new live Crypto Wizards dashboard sweep. A
journal rebuild without that capture is historical processing and is never
called current.

## Run Profiles

- `monitor`: a fast checkpoint that reuses the most recent captured evidence,
  rebuilds health and paper-readiness gates, and reports missing or stale
  artifacts. It never creates new discovery, acceptance, or execution
  authority.
- `deep`: the full evidence pass. It rebuilds the Wizard research pack, current
  multi-venue history readiness, pair universe, and venue-specific readiness
  reports before publishing the dashboard. Use this after a new scanner capture,
  a pair-detail settings capture, a new history import, or a material data issue.

The current monitor command is:

```bash
PYTHONPATH=src python -m quant_platform.cli build-command-dashboard --dashboard-refresh-profile monitor
```

The deep command is the default:

```bash
PYTHONPATH=src python -m quant_platform.cli build-command-dashboard
```

1. Capture every current scanner row, active filter, exchange, sort order, and
   source timestamp.
2. For every scanner candidate and current timeframe, capture these exact modes:
   Static Spread, Static ZScoreR, Dynamic Spread, Dynamic ZScoreR, OU Spread,
   OU ZScoreR, and Copula.
3. Every mode requires current spread/z-score charts, hedge ratio, correlation,
   Hurst, half-life, ECM X/Y/strength, entry/exit rules, explicit X/Y position
   mapping, liquidity, risk, and pair-page evidence. Dynamic modes also record
   their hedge-ratio method and window; OU modes record their displayed `mu` and
   `sigma`; Copula modes record family, signal type, directional view, and both
   entry and exit thresholds.
4. Copula additionally requires its family, correlation, both conditional
   probabilities, probability gap, stationarity badges, direction, and capital
   weighting.
5. For each fresh screen-pass setup, create or refresh
   `reports/active/wizard_mode_matrix_capture_queue.csv` and then create or
   refresh `reports/templates/wizard_pair_settings_capture_template.csv`. The
   matrix creates one visible-capture work item for each of the seven modes;
   record the
   visible pair-page settings in that template, include a timestamped screenshot,
   export, or observed request path, and mark the row confirmed only after it
   has been checked against the live dashboard.
6. Import the completed settings CSV with
   `import-wizard-pair-settings-capture --input-dir <capture.csv>`. The importer
   rejects guessed, incomplete, or mode-inconsistent rows and publishes only
   validated settings into active pair-page evidence.
7. A detail page captured before the current scanner timestamp is historical,
   not current. The capture queue marks it `capture_required`.
8. `full_detail_captured` means all required fields were captured in the current
   run. `mode_unavailable` is explicit and never counted as a pass.
9. For every Hyperliquid-routed candidate, capture a fresh public market context,
   refresh matching two-leg history, and take a public L2 book sample at the
   configured per-leg notionals. These requests are research-only and cannot
   place Testnet or live orders.
10. During the two-hour window, take one L2 sample every ten minutes. A pair can
   pass the slippage calibration input only after twelve complete samples for
   both legs inside the current rolling two-hour window. A single book snapshot
   is never calibration. Samples older than two hours stay in immutable history
   but stop counting automatically; they never place a pair into a permanent
   expired state. If the raw rolling samples are sufficient but the saved cost
   model was built with a different window or count, the cadence reports
   `model_rebuild_due` instead of treating the pair as calibrated.
11. After importing confirmed settings, run
   `build-wizard-mode-comparison`. It replays each captured mode against the
   same matching local two-leg history under zero-cost, provisional-base, and
   provisional-stress cases. It records whether the history is same-venue,
   cross-venue, or venue-unknown; cross-venue and venue-unknown results stay
   research-only and must not support acceptance.
12. Only then build the journal, compare the seven modes, refresh venue-specific
   local data/costs, and send point-in-time observations to research or RL.
13. If the live sweep fails, publish `dashboard_capture_failed`; do not produce
   new actionable candidates.

`wizard_mode_replay_capability.csv` is the mode-specific research checklist.
It reports which exact inputs are present for Static Spread, Static ZScoreR,
Dynamic Spread, Dynamic ZScoreR, OU Spread, OU ZScoreR, and Copula before a
local replay starts. A ready row authorizes only a documented local formula
approximation. It is not a claim that Crypto Wizards math has been reproduced;
a bounded custom-series proof is still required before anything is labeled
vendor-exact or contributes to acceptance.

`wizard_mode_comparison.csv` is the side-by-side research result once a visible
settings capture and matching local history exist. It has a row for each mode
and provisional cost case, but all rows are explicitly ineligible for
acceptance, paper, and execution. It is designed to make differences among
Static, Dynamic, OU, and Copula hypotheses inspectable without calling a local
formula approximation a Wizard-equivalent result.

`wizard_replay_handoff.csv` separates two research states. An exploratory local
replay requires a fresh exact-mode setup, captured settings, and matching
history; if venue economics are incomplete, it must use a documented
cost-sensitivity range and remains research-only. An acceptance replay also
requires validated venue fee, slippage, and funding or borrow inputs. Neither
state authorizes paper or live execution.

`wizard_exploratory_cost_sensitivity.csv` materializes the research range for
every handoff setup. It includes a gross-only upper bound plus provisional base
and stress cases. Execution risk stays in each provisional calculation, but it
does not prevent an exploratory replay. Every row is explicitly ineligible for
acceptance, paper, and execution until venue economics are validated.

The automated outputs are:

- `reports/active/wizard_strategy_mode_capture_queue.csv`
- `reports/active/wizard_copula_detail_capture_queue.csv`
- `reports/active/wizard_two_hour_copula_report.csv`
- `reports/active/wizard_research_journal.csv`
- `reports/active/multi_venue_history_readiness.csv`
- `reports/active/wizard_replay_handoff.csv`
- `reports/active/wizard_mode_matrix_capture_queue.csv`
- `reports/active/wizard_mode_replay_capability.csv`
- `reports/active/wizard_mode_comparison.csv`
- `reports/active/wizard_exploratory_cost_sensitivity.csv`
- `reports/templates/wizard_pair_settings_capture_template.csv`
- `reports/active/wizard_pair_settings_capture_validation.csv`
- `reports/active/dashboard_refresh_status.csv`
- `reports/active/hyperliquid_l2_slippage_samples.csv`
- `reports/active/hyperliquid_pair_cost_model.csv`
- `reports/active/hyperliquid_evidence_cadence.csv`

The canonical public-depth commands share the same defaults: twelve complete
samples per leg, ten minutes between captures, and a rolling two-hour window.
They can be overridden explicitly for research tests, but the collector, model,
and cadence must receive the same values:

```bash
PYTHONPATH=src python -m quant_platform.cli refresh-hyperliquid-execution-cost-snapshot \
  --slippage-min-samples 12 --slippage-window-hours 2
PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-pair-cost-model \
  --slippage-min-samples 12 --slippage-window-hours 2
PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-evidence-cadence \
  --slippage-min-samples 12 --slippage-window-hours 2 --slippage-cadence-minutes 10
```
