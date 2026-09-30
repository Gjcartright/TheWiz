# V2 Mathematical Validation Contract

## Authority Order

1. Raw source evidence with hashes and timestamps.
2. Explicit local formula contracts with executable identity tests.
3. Point-in-time local replay on frozen training/test boundaries.
4. Costed, purged walk-forward validation and multiple-testing controls.
5. Rebuilt datasets and out-of-sample model evaluation.
6. Paper/testnet evidence.
7. Live authority only after every preceding gate passes.

Crypto Wizards dashboard and API results are discovery, interpretation, and parity evidence. They are not local promotion authority unless the exact mode, input series, transform, orientation, fitting scope, thresholds, sizing, costs, and trade lifecycle have all been reproduced point in time.

## Canonical Pair Identity

Every result must store:

- `asset_x`
- `asset_y`
- dependent variable: `Y`
- regressor: `X`
- `hedge_ratio_orientation=beta_y_on_x`
- transform: level or log
- intercept policy
- residual formula
- signal convention
- sizing convention
- timeframe and observed timestamp grid
- train/test boundaries
- math and economic-contract versions

The active local relation is:

```text
log(Y_t) = alpha + beta_y_on_x * log(X_t) + e_t
e_t      = log(Y_t) - alpha - beta_y_on_x * log(X_t)
```

Engle-Granger and ECM results must bind this orientation to the same complete,
ordered paired sample. A coefficient or residual from another sample is not
interchangeable with the relationship under evaluation.

The replay execution spread omits the constant:

```text
s_t = log(Y_t) - beta_y_on_x * log(X_t)
```

This is not called the fitted residual. Translation-invariant z-scores remove `alpha`; raw OU-level calculations must center explicitly.

Ingestion follows the same identity. A normalizer may construct the generic
`spread` field from leg prices only when a finite positive `beta_y_on_x` is
explicitly present. It may not substitute a return beta or `1.0`. A hedge ratio
derived from an entire imported history is labeled
`full_sample_hindsight_research_only` and must be refit inside each training fold
before acceptance.

## Economic Direction

```text
signal +1 = short X / long Y
signal -1 = long X / short Y
w_X = -signal * beta / (1 + abs(beta))
w_Y =  signal        / (1 + abs(beta))
```

This named contract requires finite `beta>0`. A negative-beta relation may be researched only under a separately named portfolio and action contract.

For `e_t=Y-alpha-beta*X`, a positive error means Y is rich relative to X. Error correction therefore expects `gamma_x>0` and/or `gamma_y<0`.

## Performance Math

```text
gross_return_t = w_X,t-1*r_X,t + w_Y,t-1*r_Y,t
net_return_t   = gross_return_t - fees_t - slippage_t - funding_t
                 - execution_risk_t - partial_fill_t
equity_t       = product_{i<=t}(1 + net_return_i)
total_return   = equity_T - 1
Sharpe         = sqrt(periods_per_year) * mean(net_return)
                 / sample_std(net_return, ddof=1)
max_drawdown   = max((running_peak - equity) / running_peak), including equity_0=1
```

Unknown timeframe, declared/observed timeframe mismatch, fewer than two returns, zero variance, missing/nonpositive prices, and missing/nonpositive hedge ratios fail closed.

Dynamic beta is estimated causally as rolling OLS in the same Y-on-X direction. The
first held-out estimate is seeded only with the trailing training window; the resulting
point-in-time beta series is shared by signal generation and position sizing. A fold is
blocked if any evaluated beta is missing, nonfinite, or nonpositive under this named
opposite-leg contract.

## Crypto Wizards Boundaries

- Static spread and z-score reconstruction is diagnostic only because the captured custom-series response fits the complete submitted sample.
- Dynamic is officially Kalman-based, but exact state/update/transform details are not certified locally.
- OU is officially MLE-based, but local terminal holdout evidence does not support general parity.
- Copula conditionals are documented, but family selection and calibration history are not fully reconstructed.
- Wizard backtest summary metrics do not expose enough trade-ledger and cost-timing detail for exact performance parity.
- Dashboard colors are captured as categorical UI state. They never replace raw statistics or become live features without a point-in-time field contract.

## Prohibited Patterns

- Testing only a p-value for an orientation-sensitive estimator.
- Swapping estimator arguments to compensate for an undocumented implementation direction.
- Reusing a math version after changing a formula, sign, unit, timing, or default.
- Defaulting a missing hedge ratio to `1.0` in an acceptance backtest.
- Deriving generic spread as `price_X-beta*price_Y` under the active Y-on-X log contract.
- Selecting two-leg accounting from prices alone without an explicit hedge ratio.
- Forward-filling a missing acceptance price.
- Calling an uncentered spread a fitted residual.
- Calling a local Dynamic, OU, or Gaussian Copula approximation "Wizard exact."
- Using full-sample Wizard statistics as live features.
- Comparing returns or Sharpe without matching timeframe, sizing, costs, and trade lifecycle.
- Training ML/RL/student models from artifacts whose math lineage is stale.

## Required Diagnostic Command

```bash
uv run python -m quant_platform.cli build-v2-math-diagnostic
uv run python -m quant_platform.cli build-math-v2-acceptance
uv run python -m quant_platform.cli reevaluate-current-wizard-hyperliquid-math \
  --old-replay-manifest <frozen-manifest.json>
uv run python -m pytest
```

The re-evaluation command is audit-only. It requires explicit immutable legacy
handoff/history receipts, writes a separate `math_reevaluations` snapshot, and always
sets promotion and live authority to false. The ordinary canonical replay still rejects
legacy economic contracts.

Do not use bare `pytest`: this workspace previously had a stale executable shebang that selected the global Anaconda interpreter instead of the locked environment.

## Formula Change Protocol

1. Add a known-data identity or adversarial failure test before changing code.
2. Change the central implementation and every consumer in one migration.
3. Bump math, economic, settings, and dataset/model lineage versions as applicable.
4. Generate the diagnostic and inspect every blocked check.
5. Freeze old outputs as historical evidence.
6. Regenerate descendants in dependency order.
7. Compare old/new results on identical source hashes.
8. Restore downstream authority only after independent acceptance gates pass.

## Incident Classes

The machine-readable error ledger records every known formula or control defect,
including defects already repaired. In addition to estimator, ECM, performance,
cost, and lineage errors, it now records two ingestion-boundary failures:

- `MATH-023`: legacy fixture/dYdX spread used the reciprocal level contract and
  could invent `beta=1.0`; new imports use the canonical Y-on-X log identity.
- `MATH-024`: several dispatchers selected two-leg math from two prices alone;
  all audited dispatchers now require `price_x`, `price_y`, and `hedge_ratio`.

Old artifacts remain evidence of what happened, but neither class retains ranking,
training, paper, or trading authority until rebuilt from raw observations.

The companion `reports/audits/2026-08-20_v2_math_diagnostic_comparison.csv`
provides the direct old-local vs corrected-local vs Crypto Wizards comparison for
orientation, all seven modes, ECM, sizing, performance, costs, and learning labels.

The machine-readable historical authority receipt is
`reports/audits/2026-08-20_v2_math_diagnostic_authority.json`. It records
`BLOCKED_PENDING_CONTROLLED_REGENERATION` and is retained as fail-closed Phase 00
evidence. Current Math V2 consumers separately require the source- and
report-bound marker at `reports/active/math_v2_acceptance.json`; neither receipt
alone grants strategy, training, paper, or live authority.
