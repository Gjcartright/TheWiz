# Math V2 And Teacher Evidence Boundary

## Implemented

On the Expansion checkout, set `UV_PROJECT_ENVIRONMENT` to the Mac internal
runtime directory and run `uv sync --extra dev --locked` before these commands.

The active backtest path uses `math-v2.3-venue-clock-execution` and provides:

- interval-aware annualization for 1m, 5m, 15m, 1h, 4h, and 1d;
- blocked Sharpe when interval evidence is unknown or irregular;
- explicit open, close, reversal, and resize lifecycle accounting;
- closed-trade-only profit factor, expectancy, win rate, and trade count;
- compounded trade returns that reconcile to portfolio equity;
- separate fee, slippage, funding, execution-risk, and partial-fill ledgers;
- `signed_realized` and `conservative_absolute_drag` funding policies;
- hedge-ratio-only default exposure, with beta retained as a diagnostic;
- both ddof=0 and ddof=1 rolling z-scores;
- actual Engle-Granger, residual ADF, OU/AR(1), DFA Hurst, robust ECM, and
  Gaussian-copula conditional-CDF estimators;
- explicit invalid states instead of absolute-valued or clipped half-life;
- namespaced `research_proxy_*` fields in newly built pair histories.

Run the controlled gate with:

```bash
uv run --locked python -m quant_platform.orchestration.dynamic_cli --stage math_v2
```

The generated marker is machine-authored at
`reports/active/math_v2_acceptance.json`. The current source declares 15
reconciliation and 8 statistical checks. A consumer accepts the marker only
when all checks pass and its version, source hashes, check identities, and
report hashes match the loaded implementation and adjacent acceptance reports.
Changing a bound source requires a new marker and reports.

## Authority Boundary

`status=passed` means the core local math library passed its synthetic and
reconciliation checks. It does not mean a strategy, pair, or model passed.

The marker intentionally keeps these states blocked:

- Crypto Wizards exact-mode parity;
- point-in-time walk-forward signal authority;
- student training authority;
- paper or live execution authority.

Batch-fit statistics in a pair-history artifact are labeled as batch fits.
They may diagnose a series, but they cannot become entry-time features until
the same estimator is rerun inside a walk-forward training window.

## Teacher Adapter Inputs

The council adapter reads only:

- `reports/active/math_v2_teacher_inputs.csv`
- `reports/active/math_v2_critic_inputs.csv`

Those files are now generated automatically from the active Hyperliquid
research bundle by `teacher_evidence_materializer`. Each context uses the same
history hash, cost version, training window, and later test window across all
seven specialists. The materializer ignores legacy derived/proxy statistics
in history JSON and rebuilds its inputs from timestamps, two-leg prices, and
observed funding.

The seven local modes are clearly labeled `local_validated_estimator`; they
are not labeled `vendor_exact`. Crypto Wizards identifies the nominated mode,
while the other six local specialists provide comparative shadow evidence.

Independent critics cover dependency, regime, risk, cost, execution, and
outcome. A critic veto remains stronger than every teacher confidence score.

Their schemas are written to:

- `reports/orchestration/teacher_council/teacher_input_schema.csv`
- `reports/orchestration/teacher_council/critic_input_schema.csv`

For each context, the adapter requires exactly seven unique exact-mode rows
and six unique critic rows. The venue must be Hyperliquid, math must match
`MATH_VERSION` (`math-v2.3-venue-clock-execution`), timestamps must be point-in-time confirmed, and every row must have
an evidence path. Wizard rows cannot be converted into local votes.

An incomplete or invalid context emits no events. The adapter also clears old
JSONL streams so stale proposals cannot survive a failed refresh.

Run the complete boundary with:

```bash
uv run --locked python -m quant_platform.orchestration.dynamic_cli --stage teacher_council
```

An earlier production-repo evidence run materialized five complete contexts:
35 teacher proposals and 30 critic assessments. Its decisions were blocked by
thin L2 calibration, weak trade counts, nonpositive out-of-sample lower bounds,
and dependency/regime warnings. A fresh marker and teacher materialization are
required to assess the current source. Ranked repairs are written to
`reports/orchestration/teacher_council/teacher_evidence_next_actions.csv`.

## Remaining Statistical Work

These items are not claimed complete by the core marker:

- Wizard custom-series parity for all seven exact modes;
- integration-order validation and Johansen tests;
- structural-break and rolling parameter-stability audits;
- candidate-universe multiple-testing correction;
- non-Gaussian copula family comparison and out-of-sample family selection;
- walk-forward feature materialization for teacher and student datasets.
