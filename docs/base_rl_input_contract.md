# Base RL Input Contract

The base RL lane is the authoritative research-only control lane for paper-handoff decisions.

## Allowed inputs

- Accepted pair-history dataset built by `build-trade-dataset`
- Current shortlist and pair-universe surfaces
- Current readiness gates from:
  - `reports/priority_readiness.csv`
  - `reports/paper_execution_preflight.csv`
  - `reports/ml/model_gated_acceptance.csv`

## Forbidden inputs

- Any future-only outcome field not available at decision time
- Any historical pass artifact used as a substitute for the current blocked state
- Any augmented research prior that silently changes the base lane decision

## Required checks

- Dataset exists and is readable
- Pair coverage is computed against the current shortlist, not an old historical batch
- Paper handoff stays `research_only` unless strategy, paper, model, and pair-specific RL support all pass together

## Authoritative outputs

- `reports/rl/base_rl_training_report.csv`
- `reports/rl/base_rl_evaluation_report.csv`
- `reports/rl/base_rl_pair_coverage.csv`
- `reports/rl/base_rl_blocked_actions.csv`
- `reports/rl/base_rl_paper_handoff_status.csv`

The file `base_rl_paper_handoff_status.csv` is the single source of truth for the current base RL go/no-go answer.
