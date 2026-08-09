# Wizard to Hyperliquid Mode Fidelity

## Purpose

Crypto Wizards is the discovery and hypothesis layer. Hyperliquid is a separate
venue with its own price history, funding, order-book depth, fees, and execution
constraints. A good Wizard row is not automatically a valid Hyperliquid trade.

The workflow therefore uses three distinct evidence layers:

1. **Wizard discovery hypothesis**: a current Wizard capture identifies the pair,
   exact mode, timeframe, period, and reported risk/reward metrics.
2. **Vendor exact-mode proof**: the Crypto Wizards custom-series backtest receives
   the matching Hyperliquid close/open series and the captured settings. This shows
   how the Wizard engine behaves on the venue's own price series.
3. **Venue-native after-cost acceptance**: a local two-leg replay applies the
   actual venue fee profile, observed funding, calibrated L2 slippage, and local
   execution assumptions. This is the only layer allowed to inform promotion.

## Exact Modes

| Wizard mode | Vendor strategy | Vendor spread type |
| --- | --- | --- |
| Static (Spread) | `Spread` | `Static` |
| Static (ZScoreR) | `ZScoreRoll` | `Static` |
| Dyn (Spread) | `Spread` | `Dynamic` |
| Dyn (ZScoreR) | `ZScoreRoll` | `Dynamic` |
| OU (Spread) | `Spread` | `OU` |
| OU (ZScoreR) | `ZScoreRoll` | `OU` |
| Copula | `Copula` | capture the effective setting |

The exact mode is not merely a label. A generic local z-score calculation is
stored as a `proxy_only` result. It may help research, but it cannot be treated
as an exact Wizard replay or used as Wizard acceptance evidence.

## OU Optimal Scanner Overlay

`ou_optimal` is a boolean annotation returned on a scanner/API source row. It is
not offered as an eighth mode on the pair page. Every true or false observation
must therefore remain attached to that row's actual exact mode and orientation.

The current authoritative accounting is:

- overlay `cwouoverlay_cf3ba114a2e07d608c56`;
- 944 of 944 source rows accounted;
- 163 true and 781 false observations;
- zero independent OU Optimal pair-page captures;
- zero flagged walk-forward or family-wide statistical passes;
- no acceptance, promotion, Testnet-order, or live authority.

A separately implemented local OU optimal-stopping hypothesis may still be
researched, but it must be named as a local hypothesis and cannot be presented
as Crypto Wizards pair-page parity.

## Queue Gate

Run:

```bash
PYTHONPATH=src python -m quant_platform.cli build-hyperliquid-wizard-hypothesis-queue
```

For each local Hyperliquid pair/timeframe, the queue requires:

- local two-leg history is present and fresh;
- a matching Wizard pair and timeframe exists;
- a valid exact mode is recorded;
- Wizard Sharpe is at least `1.75` and normalized `returns_total` is at least
  `10%`;
- the source capture is current, healthy, and does not report a retrieval
  failure;
- the capture contains entry, exit, capital-weight, slippage, and commission
  settings; and
- Copula captures explicitly record the effective spread type.

A blocked row is not a rejected strategy. It means the evidence is insufficient
for a comparable mode replay. The row remains available for the next dashboard
capture or rescan.

## Current Outputs

- `reports/active/hyperliquid_wizard_hypothesis_queue.csv`
- `reports/active/hyperliquid_wizard_hypothesis_queue.md`
- `reports/dashboard/hyperliquid_wizard_hypothesis_dashboard.csv`

The queue is also part of the `venue_evidence` orchestration group. It never
submits orders, changes a pair's promotion bucket, or spends Wizard API credits.

## Bounded Custom-Series Proof

The safe default is a no-credit preflight:

```bash
PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-wizard-mode-proofs
```

It reads only `vendor_custom_series_eligible=true` rows, validates their local
two-leg prices and captured parameters, and writes the proof queue reports. It
does not call the Crypto Wizards API.

An actual vendor request is deliberately double-gated, capped at three rows, and
must be consciously enabled only after reviewing the preflight report:

```bash
QPA_ENABLE_WIZARD_CUSTOM_SERIES_PROOF=true \
PYTHONPATH=src python -m quant_platform.cli run-hyperliquid-wizard-mode-proofs \
  --execute-wizard-proof --max-pairs 3
```

Each executed request and response is saved without API secrets under
`data/raw/crypto_wizards_custom_series_proofs/`. A completed proof is still
research evidence only; `promotion_allowed` remains false until the independent
local after-cost replay passes.

The official POST contract accepts 50 to 1,100 aligned observations and requires
open and close arrays for both legs. The proof runner therefore uses the most
recent 1,100 aligned Hyperliquid candles, omits disabled zero-valued stop/forced
exit controls, preserves completed proofs on rerun, and never repeats a completed
pair/mode request.

## Observed Static Formula

The 2026-08-07 proofs established the Static formula to numerical precision:

```text
raw branch: spread = y - (alpha + beta*x)
log branch: spread = log(y) - (alpha + beta*log(x))
zscore:     full-sample sample standard deviation, ddof=1
zscore_roll: rolling roll_w sample standard deviation, ddof=1; warmup=0
```

The vendor chooses the raw or log branch and reports that choice as `log_used`.
This is exact historical reconstruction, not a live-safe estimator: the OLS fit
and ordinary z-score use the complete submitted sample. Local acceptance must
fit the transform, alpha, beta, and scale within each walk-forward training fold.

## Replay and Acceptance

Only rows with `vendor_custom_series_eligible=true` may make the bounded
custom-series API request. That request creates vendor-mode evidence, not an
execution decision. It must be followed by the local after-cost replay.

The local replay remains blocked from promotion until its mode fidelity is
`vendor_exact`, its slippage model is calibrated, funding is current, and its
normal acceptance gates pass.
