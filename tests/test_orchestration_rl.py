from __future__ import annotations

import pandas as pd
import pytest

from quant_platform.active_pipeline import build_command_dashboard
from quant_platform.orchestration import run_orchestrator
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.rl.features import build_rl_feature_frame
from quant_platform.rl.pair_trading_env import PairTradingEnv
from quant_platform.rl.quantization import export_rl_policy
from quant_platform.rl.rl_idea_engine import run_rl_idea_scout
from quant_platform.rl.rl_backtest import run_rl_research
from quant_platform.rl.train_ppo import train_ppo_research_policy


def test_orchestrator_dry_run_records_stage_contracts():
    result = run_orchestrator(stage="discovery", dry_run=True, pair_id="BNB-USD-STX-USD")

    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert not frame.empty
    assert {"run_id", "pair_id", "stage", "status", "blocker", "evidence_path", "next_step"}.issubset(frame.columns)
    assert set(frame["status"]) == {"dry_run"}
    assert (frame["pair_id"] == "BNB-USD-STX-USD").all()


def test_orchestrator_report_only_writes_spine_audit():
    result = run_orchestrator(stage="verification", report_only=True)

    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert "project_spine_audit" in set(frame["stage"])
    assert result.paths["orchestrator_status_md"].exists()


def test_mini_agent_orchestration_writes_registry_and_queue(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "priority_rank": 1,
                "pair": "BLUR-USD/ETHFI-USD",
                "sharpe": 1.906,
                "returns_total": 0.544,
                "mode_blocker": "missing_exact_mode",
            }
        ]
    ).to_csv(active / "wizard_exact_mode_capture_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSDT-TRUMPUSDT",
                "wizard_exchange": "binance",
                "readiness_status": "ready_to_fetch",
                "next_step": "fetch_binance_candles_then_track_cost_slippage_funding_or_borrow_assumptions",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)

    result = build_mini_agent_orchestration(root=tmp_path)
    registry = pd.read_csv(result.paths["mini_agent_registry"])
    queue = pd.read_csv(result.paths["next_action_queue"])

    assert {"discovery_agent", "rl_idea_agent", "red_team_agent"}.issubset(set(registry["agent"]))
    assert {"capture_exact_mode", "fetch_or_replay_venue_history", "run_rl_idea_scout"}.issubset(set(queue["task_type"]))
    assert not queue["promotion_allowed"].astype(bool).any()


def test_orchestrator_assistant_writes_task_cards_and_agent_memory(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "priority_rank": 1,
                "pair": "BLUR-USD/ETHFI-USD",
                "sharpe": 1.906,
                "returns_total": 0.544,
                "mode_blocker": "missing_exact_mode",
            }
        ]
    ).to_csv(active / "wizard_exact_mode_capture_queue.csv", index=False)

    result = build_orchestrator_assistant(root=tmp_path)
    tasks = pd.read_csv(result.paths["orchestrator_assistant_tasks"])
    learning = pd.read_csv(result.paths["agent_learning_summary"])
    cards_text = result.paths["task_cards"].read_text(encoding="utf-8")

    assert {"task_id", "assigned_agent", "blocking_condition", "assistant_decision"}.issubset(tasks.columns)
    assert "capture_exact_mode" in set(tasks["task_type"])
    assert not tasks["promotion_allowed"].astype(bool).any()
    assert "discovery_agent" in set(learning["agent"])
    assert (tmp_path / "data" / "agent_memory" / "discovery_agent.jsonl").exists()
    assert "promotion_allowed" in cards_text


def test_specialist_scoreboard_covers_exact_mode_families_without_promotion(tmp_path):
    processed = tmp_path / "data" / "processed"
    active = tmp_path / "reports" / "active"
    docs = tmp_path / "docs"
    processed.mkdir(parents=True)
    active.mkdir(parents=True)
    docs.mkdir(parents=True)
    (docs / "formula_dictionary.md").write_text("# Formulas\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/STX-USD",
                "exact_mode": "Static (Spread)",
                "passes_sharpe_gate": True,
                "sharpe": 2.1,
                "returns_total": 0.22,
            },
            {
                "pair": "SOL-USD/WLD-USD",
                "exact_mode": "OU (ZScoreR)",
                "passes_sharpe_gate": True,
                "sharpe": 2.4,
                "returns_total": 0.31,
            },
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)
    pd.DataFrame(
        [
            {
                "strategy_family": "Static Spread",
                "accepted": False,
                "sharpe": 1.5,
                "profit_factor": 2.1,
                "max_drawdown": 0.1,
                "closed_trades": 4,
            }
        ]
    ).to_csv(active / "binance_exact_mode_strategy_sweep_2026-06-25.csv", index=False)

    result = build_specialist_scoreboard(root=tmp_path)
    scoreboard = pd.read_csv(result.paths["specialist_strategy_scoreboard"])

    assert set(scoreboard["strategy_family"]) == {
        "Static Spread",
        "Static ZScoreR",
        "Dyn Spread",
        "Dyn ZScoreR",
        "OU Spread",
        "OU ZScoreR",
        "Copula",
    }
    assert not scoreboard["promotion_allowed"].astype(bool).any()
    assert "PROMOTE_TESTING" in set(scoreboard["decision"])
    assert scoreboard.loc[scoreboard["strategy_family"] == "Copula", "blocker"].iloc[0] == "missing_strategy_family_evidence"
    assert result.paths["specialist_strategy_scoreboard_md"].exists()


def test_orchestrator_agents_stage_runs_mini_agent_reports():
    result = run_orchestrator(stage="agents", report_only=True)
    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert "mini_agents" in set(frame["stage"])
    assert "orchestrator_assistant" in set(frame["stage"])
    assert "specialist_scoreboard" in set(frame["stage"])
    assert frame.loc[frame["stage"] == "mini_agents", "status"].iloc[0] == "passed"


def test_orchestrator_supreme_team_stage_runs_and_records_checkpoint(tmp_path):
    result = run_orchestrator(stage="supreme_team", report_only=True, root=tmp_path)
    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert "supreme_team_checkpoint" in set(frame["stage"])
    assert frame.loc[frame["stage"] == "supreme_team_checkpoint", "status"].iloc[0] == "passed"
    evidence = str(frame.loc[frame["stage"] == "supreme_team_checkpoint", "evidence_path"].iloc[0])
    assert "; " in evidence or ";" in evidence


def test_rl_feature_builder_blocks_future_columns():
    frame = pd.DataFrame({"zscore": [1.0], "profit_after_cost": [0.1]})

    with pytest.raises(ValueError, match="rl_feature_leakage_columns"):
        build_rl_feature_frame(frame)


def test_pair_trading_env_blocks_invalid_and_stale_actions():
    frame = pd.DataFrame({"zscore": [0.0, 2.1, -0.5], "spread": [1.0, 1.2, 1.1]})
    env = PairTradingEnv(frame, stale=True)

    _, info = env.reset()
    assert info["blocked"] is False
    _, reward, terminated, truncated, info = env.step(1)

    assert info["blocked"] is True
    assert info["blocker"] == "stale_data_blocks_position_action"
    assert reward < 0
    assert terminated is False
    assert truncated is False
    assert env.blocked_actions


def test_rl_research_writes_research_only_reports():
    result = run_rl_research(pair_id="")

    training = pd.read_csv(result.paths["training_report"])
    acceptance = pd.read_csv(result.paths["acceptance_report"])
    blocked = pd.read_csv(result.paths["blocked_actions"])

    assert {"status", "blocker", "live_enabled", "rows"}.issubset(training.columns)
    assert {"accepted", "blocker", "acceptance_reason"}.issubset(acceptance.columns)
    assert {"rl_action", "rl_reason", "blocker", "live_enabled"}.issubset(blocked.columns)
    assert not training["live_enabled"].astype(bool).any()


def test_dashboard_includes_orchestrator_and_rl_views():
    run_orchestrator(stage="rl", report_only=True)
    result = build_command_dashboard()

    assert "orchestrator_run_status" in result.paths
    assert "supreme_team_checkpoint" in result.paths
    assert "rl_research_status" in result.paths
    assert "rl_acceptance" in result.paths
    assert "quantization_readiness" in result.paths


def test_orchestrator_all_runs_checkpoint_before_dashboard():
    result = run_orchestrator(stage="all", report_only=True)
    frame = pd.read_csv(result.paths["orchestrator_status"])

    stage_order = list(frame["stage"])
    if "supreme_team_checkpoint" in stage_order and "build_dashboard" in stage_order:
        assert stage_order.index("supreme_team_checkpoint") < stage_order.index("build_dashboard")


def test_train_ppo_reports_dependency_status_without_live_enablement():
    result = train_ppo_research_policy(pair_id="")

    dependency = pd.read_csv(result.paths["ppo_dependency_report"])

    assert {"dependency", "status", "blocker", "live_enabled"}.issubset(dependency.columns)
    assert not dependency["live_enabled"].astype(bool).any()


def test_export_rl_policy_blocks_until_acceptance_passes():
    run_rl_research(pair_id="")
    result = export_rl_policy()

    parity = pd.read_csv(result.paths["parity_csv"])

    assert result.summary["exported"] is False
    assert result.summary["blocker"] == "rl_acceptance_not_passed"
    assert parity["blocker"].iloc[0] == "rl_acceptance_not_passed"


def test_run_rl_idea_scout_generates_hypothesis_artifacts(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "timeframe": "1d",
                "strategy": "Static Spread",
                "regime": "trend",
                "exact_mode": "Static Spread",
                "profit_after_cost": 0.15,
                "zscore": 1.2,
                "spread": 0.08,
                "closed_trades": 4,
                "spread_slope": 0.11,
                "beta_stability": 0.5,
            },
            {
                "pair": "SOL-USD/WLD-USD",
                "timeframe": "1d",
                "strategy": "OU Spread",
                "regime": "range",
                "exact_mode": "OU Spread",
                "profit_after_cost": 0.06,
                "zscore": -1.1,
                "spread": 0.04,
                "closed_trades": 6,
                "spread_slope": -0.03,
                "beta_stability": 0.64,
            },
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_rl_idea_scout(root=tmp_path, top_ideas=1, similarity_k=1)
    ideas = pd.read_csv(result.paths["rl_ideas"])
    sim = pd.read_csv(result.paths["rl_pair_similarity"])
    summary = pd.read_csv(result.paths["rl_idea_summary"])

    assert int(summary.loc[0, "generated_ideas"]) == 1
    assert int(summary.loc[0, "generated_similar_pairs"]) == len(sim)
    assert "pair" in ideas.columns
    assert "similar_pair" in sim.columns
    assert {"generated_ideas", "generated_similar_pairs", "policy_type", "evidence_source"}.issubset(summary.columns)
    assert not ideas.empty


def test_orchestrator_rl_stage_includes_idea_scout(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "profit_after_cost": 0.11,
                "timeframe": "1d",
                "strategy": "Static Spread",
                "regime": "range",
                "exact_mode": "Static Spread",
            }
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_orchestrator(stage="rl", root=tmp_path, report_only=True)
    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert "run_rl_idea_scout" in set(frame["stage"])
    assert frame.loc[frame["stage"] == "run_rl_idea_scout", "status"].iloc[0] == "passed"
