# Registered Stage 5 Survivor Lineage

Stage 5 may run only after an immutable Stage 4 execution receipt concludes that
at least three registered hypotheses are accepted survivors across at least three
empirical equivalence clusters and three distinct canonical asset pairs.

The Stage 4 survivor receipt and execution receipt bind supporting-cluster,
full-survivor-cluster, supporting-pair, and full-survivor-pair counts, plus the
exact canonical-pair and experiment identities. Stage 5 revalidates those values
against the frozen contract before building any dataset. Multiple modes,
orientations, or parameterizations of one pair cannot manufacture independent
breadth.

## Dataset Scope

The learning dataset keeps the complete strict-cost, point-in-time Wizard mode
family for accepted pair groups. This gives the model both positive and negative
examples without changing the frozen Stage 4 acceptance decision.

Every row retains:

- exact Wizard mode and orientation;
- Stage 4 contract and execution identifiers;
- source experiment identifier;
- registered semantic hypothesis identifier when applicable;
- registered outcome: accepted, rejected, or unregistered full-family context;
- an explicit accepted Stage 4 survivor flag;
- hash-bound walk-forward, bars, cost, regime, and conclusion evidence paths.

Registered identities and Stage 4 outcomes are audit fields, not model features.
The model may use orientation because it is known when the trade is proposed.
It may not use experiment IDs, registered IDs, survivor flags, or outcomes.

## Required Stage 5 Proof

Stage 5 fails closed unless every accepted Stage 4 survivor has:

1. At least 10 rows in the promoted training dataset.
2. At least 10 globally label-purged, pair-aware out-of-sample predictions with
   matching pair, experiment, exact mode, and orientation. Every training label
   in each fold must end before the earliest test entry, including labels from
   other pairs.
3. Coverage across at least three distinct held-out folds.
4. Coverage across at least two distinct out-of-sample regimes. A hypothesis
   may remain timeframe-specific, so multiple timeframes are not fabricated.
5. At least five out-of-sample trades selected by the model and a survivor-level
   take rate of at least 10%.
6. Complete finite realized-return coverage for every selected trade and a
   strictly positive aggregate after-cost out-of-sample return.

Every registered trade is also bound back to the immutable Stage 4 candidate
identity. Its experiment ID, pair, pair-group key, exact mode, and orientation
must all match the contract together. Preserving an accepted experiment ID while
substituting another pair, mode, or orientation is rejected even when aggregate
mode-orientation coverage and artifact hashes remain internally consistent.

Every selected-model prediction is reconciled one-to-one to the immutable
active dataset by trade ID. Pair, timeframe, venue, strategy identity, regime,
exact mode, orientation, registered contract/execution/hypothesis identity,
survivor flags, feature/entry/exit/label timestamps, profitability label, and
realized after-cost return must match. Duplicate, unknown, substituted, or
outcome-altered predictions fail closed even when their fold summaries agree.

Model selection and model acceptance use separate chronological evidence. The
earlier globally purged OOS folds are marked `model_selection` and may choose
the model family. Later folds are marked `untouched_evaluation`; only those
rows may determine acceptance, score monotonicity, concentration, or survivor
support. At least two selection folds and three untouched evaluation folds are
required by the predeclared support policy. Selection trade outcomes must all
resolve before the first untouched entry. Mixed phases, reused fold IDs,
interleaved timestamps, or reports computed from selection rows fail closed.

The winning model is replayed from the raw, hash-bound selection predictions.
For every candidate model and selection fold, Stage 5 rebuilds baseline and
filtered profit factor, Sharpe, drawdown, expectancy, take rate, and the
predeclared selection score. The deterministic leaderboard and winner must
match the published artifact and `best_model`; a forged winner or edited score
fails even when that model looks good on untouched data.

These thresholds live in
`config/registered_survivor_oos_support_policy.json`. The policy is created
before Stage 5 results, recorded in every survivor-attribution row, and
hash-bound into the immutable registered-learning receipt. Changing the policy
creates a new learning identity; it cannot reinterpret an existing receipt.
The frozen policy also declares the minimum filtered and gated model trades,
minimum populated score buckets, maximum model gain and pair-selection
concentration, and the minimum RL trades and maximum RL concentration. These
anti-overfit thresholds are therefore reviewable inputs to the protocol rather
than hidden constants selected after results are known.

RL is bound to the same survivor identity. Its per-trade research log preserves
the registered hypothesis, experiment, pair, exact mode, and orientation as
audit-only fields. Each survivor must have at least five rows and two entered
trades in both validation and the untouched held-out test, at least a 10% take
rate in each split, complete net-after-cost return coverage, and strictly
positive aggregate returns in both validation and the held-out test. A globally
successful RL summary cannot substitute for missing survivor-level evidence.

The per-trade RL execution log is also the metric authority. Stage 5 rebuilds
validation and held-out-test baseline and policy metrics directly from that log,
including trades, take rate, profit factor, Sharpe, drawdown, total return, and
pair/timeframe/regime selection and positive-PnL concentration. The published RL
evaluation and acceptance reports must match the rebuilt values; internally
consistent but forged summary reports fail closed.

RL chronology is independently replayed from the immutable active dataset.
Stage 5 recomputes the train/validation/test boundaries, globally purges every
label that crosses the next split boundary regardless of pair, compares the
published split audit with that replay, and verifies that the validation and
held-out-test trade IDs and entry-time identity fields exactly match the
recomputed partitions. A forged purge flag or a substituted held-out trade
cannot pass receipt validation.

The global model must still pass profit factor, Sharpe, drawdown, take-rate,
minimum-trade, score monotonicity, and pair/timeframe/regime/strategy
concentration gates. Reinforcement learning must independently pass validation
and untouched held-out testing with positive after-cost return, a minimum 10%
take rate, leakage controls, and concentration controls. Improving on a losing
baseline while remaining unprofitable is not acceptance.

## Prospective Model-Family Repair (2026-08-11)

The original Stage 5 design named logistic regression, random forest, and
gradient boosting, but the executable candidate set omitted random forest. The
repair added a regularized random-forest candidate with fixed parameters and
registered protocol version `2026-08-11.2` as immutable receipt
`stage5protocol_56c4520628407109c894` at
`2026-08-11T18:45:06.829478+00:00`, before the diagnostic run. The existing
chronological splits, selection rules, acceptance gates, and authority boundary
were unchanged.

All three model families were then evaluated on the same purged evidence.
Logistic regression remained the selection-fold winner. Random forest ranked
second with a 36.83% selection take rate, +0.0562 profit-factor delta, and
+6.5271 Sharpe delta, but failed the untouched evaluation: its take rate was
39.84%, filtered profit factor was 0.8243, and profit-factor delta was -0.0689.
The selected logistic model also failed untouched profitability and score-bucket
monotonicity. This is a genuine negative result, not permission to switch models
after seeing evaluation outcomes. Stage 5, quantization, Testnet candidate,
Testnet order, and live-trading authority therefore remain false.

Protocol version `2026-08-11.3`, registered before any new diagnostic as
`stage5protocol_a582f43b2f7b210406ee` at
`2026-08-11T19:07:49.546321+00:00`, prospectively adds the Stage 4
contract-identity binding described above. It does not reinterpret the
`2026-08-11.2` diagnostic and does not grant model, Testnet, or live authority.

Protocol version `2026-08-11.5`, registered as
`stage5protocol_dd2570b74b5911e3e4aa` at
`2026-08-11T19:24:50.357794+00:00`, also hash-binds the Stage 4 contract and
execution-validator modules and recomputes production contract IDs from their
frozen content. The verifier recognizes both the historical and current
content-material schemas; a partial schema downgrade does not validate. Editing
a candidate identity and rehashing downstream receipts cannot create a
different accepted hypothesis under the original contract ID.

Protocol version `2026-08-11.6`, registered as
`stage5protocol_06a100f40fb7af7e316b` at
`2026-08-11T19:38:14.132162+00:00`, prospectively binds the strengthened Stage
3-to-4 handoff validator. The completed scheduler cycle must be published to
the immutable evidence root before Stage 4 is invoked. Stage 4 validates and
binds that byte-identical immutable copy; a rehashed mutable scheduler file is
not sufficient handoff evidence.

Protocol version `2026-08-12.1` adds one more direct-call invariant: a Stage 4
execution receipt is valid only at the canonical immutable registry path
`data/research/registered_rerun_executions/{contract_id}.json`. Copying an
otherwise valid receipt to an active, scratch, or caller-selected path cannot
start the dataset, model, or RL stages. It was prospectively registered as
`stage5protocol_88170a22b0541582b366` at
`2026-08-12T20:27:53.698568+00:00`; it remains research-only and grants no
downstream authority.

## Immutable Evidence

The registered learning receipt hash-binds:

- the Stage 4 execution receipt;
- promoted dataset receipt and pointer;
- model metrics, lineage, predictions, backtest, and acceptance;
- score buckets, pair concentration, and gain concentration;
- deterministic model-selection leaderboard;
- model failure attribution and pair support;
- accepted-survivor model attribution;
- RL acceptance, evaluation, split, leakage, schema, and lineage;
- RL per-trade execution backtest and accepted-survivor attribution;
- model authority status.

Receipt validation recomputes model, RL, concentration, and survivor-attribution
checks from those artifacts. Rehashing an edited attribution report does not make
it valid.

## Authority Boundary

Passing Stage 5 proves research usefulness only. It grants no Testnet candidate,
Testnet order, promotion, quantization, or live-trading authority. Those remain
separate downstream gates.
