# Specialist Strategy Scoreboard

This report gives each strategy family its own specialist lane while keeping promotion authority outside the specialist score.
A high score means run more local testing, not accept a strategy for trading.

## Strategy Specialists

| strategy_family   |   combined_score | decision        | blocker                          | next_step                                                      |
|:------------------|-----------------:|:----------------|:---------------------------------|:---------------------------------------------------------------|
| Static Spread     |             0.47 | FETCH_MORE_DATA | missing_local_replay             | capture Wizard exact mode and local history for Static Spread  |
| Static ZScoreR    |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for Static ZScoreR |
| Dyn Spread        |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for Dyn Spread     |
| Dyn ZScoreR       |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for Dyn ZScoreR    |
| OU Spread         |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for OU Spread      |
| OU ZScoreR        |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for OU ZScoreR     |
| Copula            |             0.27 | FETCH_MORE_DATA | missing_strategy_family_evidence | capture Wizard exact mode and local history for Copula         |

## Horizontal Agents

| agent           |   average_score |   covered_strategy_families | role                                         |
|:----------------|----------------:|----------------------------:|:---------------------------------------------|
| research_agent  |          0.1429 |                           1 | find and rank hypotheses                     |
| memory_agent    |          0.2    |                           7 | record what each specialist learned          |
| test_agent      |          0      |                           0 | verify ideas with local replay               |
| reference_agent |          1      |                           7 | keep formulas and field definitions explicit |
| rl_agent        |          0.2    |                           7 | suggest ideas and similar-pair fingerprints  |

## Guardrails

- Specialist rows cannot promote a pair or strategy by themselves.
- Crypto Wizards evidence stays discovery-only.
- RL evidence stays idea-only until local after-cost replay proves it.
- Missing evidence creates a blocker or next step instead of a hidden score.
