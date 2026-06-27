# RL Idea Supreme Team Report

Date: 2026-06-26  
Source run: `supreme_team_2026-06-26_183630Z`  

## Supreme Team Summary

I ran a full Supreme Team checkpoint before this RL-specific pass.  
The latest checkpoint reports:

- `open_actions`: 20
- `critical`: 8
- `high`: 8
- `medium`: 4

All 20 global open actions are still valid, and the top systemic blockers are now:

1. `crypto_wizards_capture` missing nested execution-ready history
2. `strategy_acceptance` not passing production gates
3. `dydx_testnet_readiness` still blocked on `submit_orders_false`
4. `paper_execution_gate` waiting on strategy + execution gates
5. `learning_event_store` has no outcome events yet

This confirms the RL ideas cannot be production-authoritative yet.

## RL Idea-Specific Gap Analysis

### G1 — No real RL policy trained yet
- Current `reports/rl/rl_training_report.csv` shows `policy = safe_quantile_baseline`.
- The policy is currently a return-quantile baseline, not an RL network.
- Evidence: `reports/rl/rl_training_report.csv`  
  (`status=research_only`, `live_enabled=False`, `blocker=rl_live_use_blocked`).

### G2 — Critical dependency not installed
- `stable-baselines3` is not available, so PPO path is blocked.
- Evidence: `reports/rl/rl_ppo_dependency_report.csv`
  (`dependency=stable-baselines3`, `status=missing`, `blocker=missing_optional_rl_dependency`).

### G3 — RL acceptance fail is expected and traceable
- `rl_acceptance_report` has `accepted=False` and `blocker=rl_acceptance_gates_not_met`.
- RL policy currently improves drawdown/trade count versus baseline but does not clear the acceptance envelope.
- Evidence: `reports/rl/rl_acceptance_report.csv`.

### G4 — No RL idea artifacts are materialized
- Pipeline defines `rl_idea_agent` outputs (`reports/agents/rl_ideas.csv`,
  `reports/agents/rl_pair_similarity.csv`) but they are not currently produced by any stage.
- Evidence: `reports/agents` directory currently lacks both files.

### G5 — Quantization/export remains disabled by design
- `rl_quantization_parity.csv` shows `accepted=False` and blocker `rl_acceptance_not_passed`.
- This is correct hard gate behavior, but it means RL ideas cannot be executed/ scored live.

### G6 — Execution path still explicitly blocked
- Live action flow includes hard stops:
  - `rl_acceptance_not_passed`
  - `rl_live_submission_blocked_in_v1`
- Evidence: `src/quant_platform/rl/rl_execution_manager.py`.

## RL Idea Pre-Mortem (How this could fail next)

1. **Over-promote RL idea output before acceptance gates**  
   Mitigation: keep `rl_idea_agent` marked `none_rl_hint_only`.

2. **Train on missing/too-narrow context and overfit to history**  
Mitigation: require walk-forward style comparison before acceptance and concentration checks.

3. **Ship model artifacts despite acceptance fail**  
   Mitigation: keep existing `export_rl_policy` and `quantization` blockers tied to `accepted=True`.

4. **Assume venue execution is available because RL research runs**  
   Mitigation: tie all RL actions to current venue readiness checks and paper/trade execution preflight.

5. **Let stale capture/data freshness become RL features**  
   Mitigation: keep leakage audit clean and enforce freshness on feature source inputs in orchestrator before RL run.

## RL Idea Post-Mortem (How it could still hide mistakes)

1. If RL appears to improve raw metrics on one narrow pair/timeframe, concentration checks might still pass visually but not substantively.  
   Fix: add explicit pair/timeframe concentration ceilings in reporting and escalate repeated concentration drift.

2. If safe-quantile proxy remains “best available,” RL idea claims can drift from actual policy strength.  
   Fix: tag all strategy-stage outputs with `policy_type` and require proof of trained policy source for model claims.

3. If missing artifacts accumulate (rl_ideas / pair similarity), RL lane becomes an untestable black box.  
   Fix: make those outputs required artifacts for the RL specialist queue and fail queue completeness if absent.

4. If dependency installation appears available but RL is not called, this can give false confidence.  
   Fix: report RL dependency + acceptance + execution blocker together as one integrated status card.

## RL Idea Red Team Review

1. **False confidence from partial success**
   - Risk: one non-robust RL proxy looks good by chance.
   - Control: require out-of-sample acceptance and concentration constraints before any RL action.

2. **Attack vector: silent bypass through specialist score**
   - Risk: RL score contributes to dashboards despite not being accepted.
   - Control: ensure `rl_score` contribution is advisory (`rl_idea_agent` promotion_authority = hint only).

3. **Data poisoning / stale features**
   - Risk: RL uses stale pair features due to delayed pair-detail updates.
   - Control: require freshness checks and `stale` flags on feature sources before RL research.

4. **Adversarial market regime**
   - Risk: RL ideas overfit to pre/post events that are gone during execution windows.
   - Control: use regime stratified reporting (already present in execution backtest fields) and reject unbalanced regimes.

5. **Venue mismatch**
   - Risk: RL outputs generated from discovery lanes without venue tradability.
   - Control: route RL ideas through venue evidence and require tradability status before simulation.

## Action Plan (Next 8 steps)

1. **Create RL idea artifact extractor stage**
   - Build `reports/agents/rl_ideas.csv` from RL evaluation/training backtest + top feature fingerprints.
   - Build `reports/agents/rl_pair_similarity.csv` using local pair-attribute neighbors.

2. **Install RL dependency and harden PPO training path**
   - Resolve `stable-baselines3` availability.
   - Update `train_ppo_research_policy` so dependency success moves stage from “scaffold-only” to “attempt training” with file outputs.

3. **Introduce a true policy training/validation branch**
   - Add explicit `rl_policy_training_report.csv` with at least:
     - policy name, timesteps, eval metrics, train/eval split dates, seed.
   - Keep current safe quantile baseline as fallback only.

4. **Tighten RL acceptance logic**
   - Add `rl_coverage_score` and `policy_concentration_score` dimensions to acceptance report.
   - Require minimal pair diversity and execution cadence before `accepted=True`.

5. **Wire RL artifacts to `reports/gate` and dashboard**
   - Include RL training-policy provenance and acceptance status in specialist scoreboard and command dashboard.

6. **Generate RL idea task outputs as mandatory queue evidence**
- In `build_mini_agent_orchestration`, require queue tasks to mark `rl_idea_agent` as complete only when both RL artifact files exist.

7. **Run RL Supreme Team slice**
   - Keep regular Supreme Team.
   - Add/update RL-specific checklist file in this directory after each run.

8. **Keep execution hard-gates**
   - Preserve `rl_live_submission_blocked_in_v1` and acceptance gate until all above are complete.
   - Only after RL acceptance pass and quantization parity, allow RL-aware non-default experiments.

## Recommended next command set

```bash
PYTHONPATH=src python3 -m quant_platform.cli run-orchestrator --stage rl
PYTHONPATH=src python3 -m quant_platform.cli run-orchestrator --stage agents
PYTHONPATH=src python3 -m quant_platform.cli supreme-team
PYTHONPATH=src python3 -m quant_platform.cli report-specialist-scoreboard
PYTHONPATH=src python3 -m quant_platform.cli build-command-dashboard
PYTHONPATH=src python3 -m quant_platform.cli train-rl-ppo
```

This preserves evidence-first execution and keeps the RL idea as research-enhancement until it meets acceptance criteria.
