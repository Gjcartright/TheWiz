from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_platform.active_pipeline import build_command_dashboard
from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration import run_langgraph_agent_workflow, run_orchestrator
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.rl.features import build_rl_feature_frame
from quant_platform.rl.pair_trading_env import PairTradingEnv
from quant_platform.rl.quantization import export_rl_policy
from quant_platform.rl.rl_idea_engine import run_rl_idea_scout
from quant_platform.rl.rl_learning_agent import (
    _chronological_rl_partitions,
    _policy_grid,
    run_magicka_learning_cycle,
    run_sequential_thinking_magicka,
)
from quant_platform.rl.brain_cycle import build_brain_readiness_report, run_brain_cycle
from quant_platform.rl.rl_acceptance import return_summary, rl_acceptance_report
from quant_platform.rl.rl_backtest import run_rl_research, simulate_strategy_returns
from quant_platform.rl.train_ppo import train_ppo_research_policy


def test_orchestrator_dry_run_records_stage_contracts():
    result = run_orchestrator(stage="discovery", dry_run=True, pair_id="BNB-USD-STX-USD")

    frame = pd.read_csv(result.paths["orchestrator_status"])

    assert not frame.empty
    assert {"run_id", "pair_id", "stage", "status", "blocker", "evidence_path", "next_step"}.issubset(frame.columns)
    assert set(frame["status"]) == {"dry_run"}
    assert (frame["pair_id"] == "BNB-USD-STX-USD").all()


def test_langgraph_agent_workflow_dry_run_writes_graph_manifests(tmp_path):
    result = run_langgraph_agent_workflow(
        stage="discovery",
        dry_run=True,
        pair_id="BNB-USD-STX-USD",
        root=tmp_path,
    )

    frame = pd.read_csv(result.paths["orchestrator_status"])
    lanes = pd.read_csv(result.paths["langgraph_agent_lanes"])
    edges = pd.read_csv(result.paths["langgraph_agent_edges"])
    markdown = result.paths["langgraph_agent_workflow_md"].read_text(encoding="utf-8")

    assert not frame.empty
    assert set(frame["status"]) == {"dry_run"}
    assert {"wizard_capture_agent", "execution_guard_agent", "base_rl_agent"}.issubset(set(lanes["agent"]))
    assert {"from", "to", "condition"}.issubset(edges.columns)
    assert "LangGraph Agent Workflow" in markdown
    assert result.summary["graph"] == "langgraph_agent_workflow"


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

    assert {"youtube_research_brain", "discovery_agent", "rl_idea_agent", "red_team_agent"}.issubset(
        set(registry["agent"])
    )
    assert {"capture_exact_mode", "fetch_or_replay_venue_history", "run_rl_idea_scout"}.issubset(set(queue["task_type"]))
    assert "run_sequential_thinking_magicka" in set(queue["task_type"])
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


def test_orchestrator_assistant_records_rl_task_outcomes(tmp_path):
    agents = tmp_path / "reports" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    rl_reports = tmp_path / "reports" / "rl"
    rl_reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "research_only", "blocker": "rl_live_use_blocked", "live_enabled": False, "rows": 2, "features": 6, "policy": "safe_quantile_baseline"}]).to_csv(
        rl_reports / "rl_training_report.csv",
        index=False,
    )
    pd.DataFrame([{"accepted": True, "blocker": "", "acceptance_reason": "passed", "raw_profit_factor": 1.1, "rl_profit_factor": 1.6, "raw_drawdown": 0.09, "rl_drawdown": 0.08, "rl_trades": 40, "rl_take_rate": 0.2, "pair_concentration": 0.5, "timeframe_concentration": 0.5}]).to_csv(
        rl_reports / "rl_acceptance_report.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "generated_ideas": 2,
                "generated_similar_pairs": 1,
                "policy_type": "safe_quantile_baseline",
                "blocker": "",
                "evidence_source": "data/ml/trade_training_dataset.csv",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ]
    ).to_csv(agents / "rl_idea_summary.csv", index=False)
    pd.DataFrame([{"pair": "BTC-USD/ETH-USD", "similar_pair": "SOL-USD/LINK-USD", "similarity_rank": 1}]).to_csv(
        agents / "rl_pair_similarity.csv",
        index=False,
    )

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSD-TRUMPUSD",
                "wizard_exchange": "binance",
                "readiness_status": "ready_to_fetch",
                "next_step": "fetch_candles",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)

    result = build_orchestrator_assistant(root=tmp_path)
    lines_idea = (tmp_path / "data" / "agent_memory" / "rl_idea_agent.jsonl").read_text(encoding="utf-8").splitlines()
    lines_sim = (tmp_path / "data" / "agent_memory" / "rl_similarity_agent.jsonl").read_text(encoding="utf-8").splitlines()

    assert any('"outcome_known": true' in line.lower() for line in lines_idea)
    assert any('"outcome_label": "passed"' in line for line in lines_idea)
    assert any('"outcome_known": true' in line.lower() for line in lines_sim)
    assert any('"outcome_label": "validated"' in line for line in lines_sim)
    # sequential-thinking magicka memory appears once it is evaluated from the queue
    lines_seq = (tmp_path / "data" / "agent_memory" / "sequential_thinking_magicka_agent.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("run_sequential_thinking_magicka" in line for line in lines_seq)


def test_mini_agent_queue_uses_agent_effectiveness_to_prioritize(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    agent_reports = tmp_path / "reports" / "agents"
    agent_reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSD-USDLTCUSD",
                "wizard_exchange": "binance",
                "readiness_status": "ready_for_replay",
                "next_step": "fetch replayed venue history",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)
    rl_reports = tmp_path / "reports" / "rl"
    rl_reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "",
                "live_enabled": False,
                "rows": 2,
                "features": 4,
                "policy": "safe_quantile_baseline",
            }
        ]
    ).to_csv(rl_reports / "rl_training_report.csv", index=False)
    pd.DataFrame(
        [
            {
                "accepted": True,
                "blocker": "",
                "acceptance_reason": "passed",
                "raw_profit_factor": 1.1,
                "rl_profit_factor": 1.4,
                "raw_drawdown": 0.09,
                "rl_drawdown": 0.08,
                "rl_trades": 35,
                "rl_take_rate": 0.22,
                "pair_concentration": 0.5,
                "timeframe_concentration": 0.5,
                "pair_pnl_concentration": 0.5,
                "timeframe_pnl_concentration": 0.5,
                "regime_concentration": 0.5,
                "regime_pnl_concentration": 0.5,
            }
        ]
    ).to_csv(rl_reports / "rl_acceptance_report.csv", index=False)
    pd.DataFrame({"agent": ["rl_idea_agent", "venue_evidence_agent"], "effectiveness_score": [0.95, 0.0], "reason": ["good", "low"]}).to_csv(
        agent_reports / "agent_effectiveness.csv",
        index=False,
    )

    result = build_mini_agent_orchestration(root=tmp_path)
    queue = pd.read_csv(result.paths["next_action_queue"])

    first_task = queue.iloc[0]
    assert first_task["assigned_agent"] == "rl_idea_agent"
    assert str(first_task["task_type"]) in {"run_rl_idea_scout", "extract_rl_similarity_candidates"}
    assert float(first_task["agent_effectiveness"]) > float(queue.iloc[-1]["agent_effectiveness"])


def test_orchestrator_assistant_uses_paper_learning_signal_for_rl_outcomes(tmp_path):
    agents = tmp_path / "reports" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "agents" / "rl_idea_summary.csv").write_text(
        "generated_ideas,generated_similar_pairs,policy_type,blocker,evidence_source,generated_at\n2,1,safe_quantile_baseline,,data/ml/trade_training_dataset.csv,2026-01-01T00:00:00Z\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "agents" / "rl_pair_similarity.csv").write_text(
        "idea_seed,pair,timeframe,similar_pair,similar_timeframe,similarity_rank,distance,shared_features,generated_at\n",
        encoding="utf-8",
    )
    rl_reports = tmp_path / "reports" / "rl"
    rl_reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "research_only", "blocker": "", "live_enabled": False, "rows": 2, "features": 6, "policy": "safe_quantile_baseline"}]).to_csv(
        rl_reports / "rl_training_report.csv",
        index=False,
    )
    pd.DataFrame([{"accepted": True, "blocker": "", "acceptance_reason": "passed", "raw_profit_factor": 1.1, "rl_profit_factor": 1.6, "raw_drawdown": 0.09, "rl_drawdown": 0.08, "rl_trades": 40, "rl_take_rate": 0.2, "pair_concentration": 0.5, "timeframe_concentration": 0.5}]).to_csv(
        rl_reports / "rl_acceptance_report.csv",
        index=False,
    )

    trade_store = tmp_path / "data" / "meta_learning" / "trades.jsonl"
    trade_store.parent.mkdir(parents=True)
    trade_store.write_text(
        "\n".join(
            [
                "{\"trade_id\":\"1\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"pair\":\"BTC\",\"strategy\":\"rl\",\"regime\":\"\",\"features\":{},\"signal\":{},\"execution\":{},\"outcome\":{\"realized_return\": 0.5}}",
                "{\"trade_id\":\"2\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"pair\":\"BTC\",\"strategy\":\"rl\",\"regime\":\"\",\"features\":{},\"signal\":{},\"execution\":{},\"outcome\":{\"realized_return\": 0.4}}",
                "{\"trade_id\":\"3\",\"timestamp\":\"2026-01-01T00:00:00Z\",\"pair\":\"BTC\",\"strategy\":\"rl\",\"regime\":\"\",\"features\":{},\"signal\":{},\"execution\":{},\"outcome\":{\"realized_return\": -0.1}}",
            ]
        ),
        encoding="utf-8",
    )

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "ETHUSD-TRUMPUSD",
                "wizard_exchange": "binance",
                "readiness_status": "ready_for_replay",
                "next_step": "fetch replay history",
            }
        ]
    ).to_csv(active / "multi_venue_history_readiness_2026-06-25.csv", index=False)

    build_orchestrator_assistant(root=tmp_path)
    lines = (tmp_path / "data" / "agent_memory" / "rl_idea_agent.jsonl").read_text(encoding="utf-8").splitlines()
    assert any("rl_idea_task_generated:2;paper_profit_ratio=0.6667" in line for line in lines)


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
    assert "youtube_research_brain" in set(frame["stage"])
    assert "orchestrator_assistant" in set(frame["stage"])
    assert "specialist_scoreboard" in set(frame["stage"])
    assert frame.loc[frame["stage"] == "mini_agents", "status"].iloc[0] == "passed"
    assert frame.loc[frame["stage"] == "youtube_research_brain", "status"].iloc[0] == "passed"


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
    assert "brain_readiness_report" in result.paths
    assert "brain_candidate_rollup" in result.paths
    assert "brain_readiness_trend" in result.paths


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
    assert {"stop_loss_hint", "take_profit_hint", "session_loss_cap_hint", "risk_control_style"}.issubset(set(ideas.columns))
    assert "similar_pair" in sim.columns
    assert {"generated_ideas", "generated_similar_pairs", "policy_type", "evidence_source"}.issubset(summary.columns)
    assert not ideas.empty
    assert ideas.iloc[0]["strategy"] == "Static Spread"
    assert ideas.iloc[0]["timeframe"] == "1d"


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


def test_magicka_learning_cycle_runs_and_writes_artifacts(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "trade_id": "T1",
                "profit_after_cost": 0.12,
                "entry_abs_zscore": 1.8,
                "trade_bars": 10,
                "hold_bars": 8,
                "realized_return": 0.12,
                "strategy_name": "Static Spread",
                "timeframe": "1h",
                "strategy_id": "S1",
                "entry_bar_index": 1,
                "exit_bar_index": 11,
                "signal_side": "long",
                "max_adverse_excursion": 0.02,
                "max_favorable_excursion": 0.15,
            }
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_magicka_learning_cycle(root=tmp_path, policy_candidates=4)
    summary = pd.read_csv(result.paths["summary"])
    backtests = pd.read_csv(result.paths["backtests"])
    ideas = pd.read_csv(result.paths["learning_ideas"])
    agent_frame = pd.read_csv(result.paths["agent_experiments"])

    assert not summary.empty
    assert len(summary) > 0
    assert not backtests.empty
    assert {"policy_name", "winner"}.issubset(set(summary.columns))
    assert {"policy_name", "pair", "idea_type", "generated_at", "exit_reason", "stop_triggered"}.issubset(set(ideas.columns))
    assert {"stop_loss_pct", "session_loss_cap_pct"}.issubset(set(summary.columns))
    assert not agent_frame.empty


def test_rl_policy_grid_covers_every_parameter_axis_before_expansion():
    policies = _policy_grid(12)

    assert len(policies) == 12
    assert {policy["entry_threshold"] for policy in policies} == {0.55, 0.60, 0.65, 0.70, 0.75, 0.80}
    assert {policy["hold_cap_pct"] for policy in policies} == {0.15, 0.30, 0.45}
    assert {policy["volatility_penalty_weight"] for policy in policies} == {0.15, 0.25, 0.35}
    assert {policy["stop_loss_pct"] for policy in policies} == {0.03, 0.05, 0.07}
    assert {policy["session_loss_cap_pct"] for policy in policies} == {0.10, 0.15}


def test_rl_simulator_uses_net_strategy_return_without_second_cost_or_short_sign_flip():
    frame = pd.DataFrame(
        [
            {
                "trade_id": "T1",
                "pair": "BTC-USD/ETH-USD",
                "profit_after_cost": 0.10,
                "trade_cost_drag": 0.20,
                "entry_abs_zscore": 2.0,
                "trade_bars": 10,
                "hold_bars": 10,
                "signal_side": "short_spread",
                "max_adverse_excursion": 0.0,
                "max_favorable_excursion": 0.0,
                "timeframe": "1h",
            }
        ]
    )
    policy = {
        "policy_name": "mechanics_check",
        "entry_threshold": 1.0,
        "hold_cap_pct": 1.0,
        "volatility_penalty_weight": 0.0,
        "min_hold_bars": 10,
        "stop_loss_pct": 0.50,
        "take_profit_pct": 0.50,
        "max_trade_drawdown_pct": 0.50,
        "session_loss_cap_pct": 1.0,
    }

    result = simulate_strategy_returns(frame, policy)

    assert result["returns"].iloc[0] == pytest.approx(0.10)
    assert result["frame"].iloc[0]["return_basis"] == "net_after_cost_strategy_return"
    assert "no_second_cost_charge" in result["frame"].iloc[0]["cost_treatment"]


def test_rl_session_loss_cap_blocks_only_later_trades_and_resets_next_day():
    frame = pd.DataFrame(
        {
            "trade_id": ["T1", "T2", "T3"],
            "pair": ["BTC-USD/ETH-USD"] * 3,
            "profit_after_cost": [-0.20, 0.10, 0.10],
            "entry_abs_zscore": [2.0] * 3,
            "trade_bars": [1] * 3,
            "hold_bars": [1] * 3,
            "signal_side": ["long_spread"] * 3,
            "max_adverse_excursion": [0.0] * 3,
            "max_favorable_excursion": [0.0] * 3,
            "timeframe": ["1h"] * 3,
            "entry_timestamp": [
                "2026-08-01T01:00:00Z",
                "2026-08-01T02:00:00Z",
                "2026-08-02T01:00:00Z",
            ],
        }
    )
    policy = {
        "entry_threshold": 1.0,
        "hold_cap_pct": 1.0,
        "volatility_penalty_weight": 0.0,
        "stop_loss_pct": 0.50,
        "take_profit_pct": 0.50,
        "max_trade_drawdown_pct": 0.50,
        "session_loss_cap_pct": 0.10,
    }

    result = simulate_strategy_returns(frame, policy)

    assert result["frame"]["simulation_reason"].tolist() == [
        "entered",
        "session_loss_cap",
        "entered",
    ]
    assert result["returns"].tolist() == pytest.approx([-0.20, 0.10])
    assert result["frame"]["session_scope"].eq("pair_utc_day_proxy").all()


def test_rl_acceptance_requires_validation_and_untouched_test_evidence():
    full_sample_only = pd.DataFrame(
        [
            {
                "variant": "non_rl_baseline",
                "trades": 100,
                "take_rate": 1.0,
                "profit_factor": 1.0,
                "sharpe": 0.5,
                "max_drawdown": 0.20,
                "pair_concentration": 0.5,
                "timeframe_concentration": 0.5,
                "pair_pnl_concentration": 0.5,
                "timeframe_pnl_concentration": 0.5,
                "regime_concentration": 0.5,
                "regime_pnl_concentration": 0.5,
            },
            {
                "variant": "safe_rl_policy",
                "trades": 50,
                "take_rate": 0.5,
                "profit_factor": 2.0,
                "sharpe": 1.0,
                "max_drawdown": 0.10,
                "pair_concentration": 0.5,
                "timeframe_concentration": 0.5,
                "pair_pnl_concentration": 0.5,
                "timeframe_pnl_concentration": 0.5,
                "regime_concentration": 0.5,
                "regime_pnl_concentration": 0.5,
            },
        ]
    )
    rejected = rl_acceptance_report(full_sample_only)

    assert not bool(rejected.iloc[0]["accepted"])
    assert rejected.iloc[0]["blocker"] == "missing_rl_out_of_sample_evidence"

    oos = pd.concat(
        [
            full_sample_only.assign(evaluation_split="validation"),
            full_sample_only.assign(evaluation_split="held_out_test"),
        ],
        ignore_index=True,
    )
    accepted = rl_acceptance_report(oos)

    assert bool(accepted.iloc[0]["accepted"])
    assert bool(accepted.iloc[0]["validation_passed"])
    assert bool(accepted.iloc[0]["held_out_test_passed"])


def test_rl_concentration_separates_opportunity_mix_from_pnl_dependency():
    source = pd.DataFrame(
        {
            "timeframe": ["5m"] * 90 + ["1h"] * 10,
            "pair": ["A-B"] * 100,
            "regime": ["range"] * 100,
        }
    )
    selected = pd.concat([source.iloc[:9], source.iloc[90:91]], ignore_index=True)
    returns = pd.Series([0.01] * 10)

    summary = return_summary(
        "policy",
        selected,
        returns,
        len(source),
        source_frame=source,
    )

    assert summary["raw_timeframe_concentration"] == pytest.approx(0.9)
    assert summary["timeframe_concentration"] == pytest.approx(0.5)
    assert summary["timeframe_pnl_concentration"] == pytest.approx(0.9)


def test_rl_chronological_split_globally_purges_overlapping_labels():
    entry_times = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    label_times = entry_times + pd.Timedelta(hours=1)
    label_times = pd.Series(label_times)
    label_times.iloc[110:120] = entry_times[130]
    frame = pd.DataFrame(
        {
            "pair": [f"P{index % 5}-USD/Q{index % 7}-USD" for index in range(200)],
            "timeframe": ["1h" if index % 2 else "4h" for index in range(200)],
            "feature_timestamp": entry_times,
            "label_timestamp": label_times,
            "profit_after_cost": [0.01 if index % 3 else -0.01 for index in range(200)],
        }
    )

    _, partitions, audit = _chronological_rl_partitions(frame)

    assert audit["status"].eq("ready").all()
    assert int(audit.loc[audit["split"] == "train", "purged_overlap_rows"].iloc[0]) == 10
    validation_start = pd.Timestamp(audit.loc[audit["split"] == "train", "validation_start"].iloc[0])
    assert pd.to_datetime(partitions["train"]["label_timestamp"], utc=True).lt(validation_start).all()


def test_magicka_oos_selection_rejects_single_pair_and_timeframe_concentration(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    entry_times = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    pd.DataFrame(
        {
            "pair": ["BTC-USD/ETH-USD"] * 200,
            "trade_id": [f"T{index}" for index in range(200)],
            "profit_after_cost": [0.03 if index % 3 else -0.01 for index in range(200)],
            "entry_abs_zscore": [0.5 + (index % 20) / 10 for index in range(200)],
            "trade_bars": [10] * 200,
            "hold_bars": [8] * 200,
            "strategy_name": ["Static Spread"] * 200,
            "timeframe": ["1h"] * 200,
            "signal_side": ["long_spread"] * 200,
            "max_adverse_excursion": [0.01] * 200,
            "max_favorable_excursion": [0.04] * 200,
            "feature_timestamp": entry_times,
            "label_timestamp": entry_times + pd.Timedelta(hours=1),
        }
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_magicka_learning_cycle(root=tmp_path, policy_candidates=12)
    summary = pd.read_csv(result.paths["summary"])
    split_audit = pd.read_csv(result.paths["split_audit"])
    best = json.loads(result.paths["best_policy"].read_text(encoding="utf-8"))

    assert split_audit["status"].eq("ready").all()
    assert summary["winner"].eq(1).sum() == 1
    assert best["policy_selection_status"] == "REJECTED"
    assert best["status"] == "blocked"
    assert best["live_enabled"] is False
    assert "pair_concentration" in best["validation_gate_failures"]
    assert "timeframe_selection_concentration" in best["validation_gate_failures"]
    assert "timeframe_pnl_concentration" in best["validation_gate_failures"]


def test_rl_learning_cycle_marks_stop_loss_when_risk_limit_hit(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "trade_id": "T1",
                "profit_after_cost": -0.20,
                "entry_abs_zscore": 2.1,
                "trade_bars": 12,
                "hold_bars": 10,
                "realized_return": -0.20,
                "strategy_name": "Static Spread",
                "timeframe": "1h",
                "strategy_id": "S1",
                "entry_bar_index": 1,
                "exit_bar_index": 13,
                "signal_side": "long",
                "max_adverse_excursion": 0.20,
                "max_favorable_excursion": 0.01,
            }
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_magicka_learning_cycle(root=tmp_path, policy_candidates=2)
    backtests = pd.read_csv(result.paths["backtests"])

    assert "exit_reason" in backtests.columns
    assert "stop_triggered" in backtests.columns
    assert backtests["exit_reason"].astype(str).str.contains("stop_loss").any()
    assert backtests["stop_triggered"].astype(bool).any()


def test_run_sequential_thinking_magicka_cycle_writes_recommendations(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "trade_id": "T1",
                "profit_after_cost": 0.12,
                "entry_abs_zscore": 1.8,
                "trade_bars": 10,
                "hold_bars": 8,
                "realized_return": 0.12,
                "strategy_name": "Static Spread",
                "timeframe": "1h",
                "strategy_id": "S1",
                "entry_bar_index": 1,
                "exit_bar_index": 11,
                "signal_side": "long",
            }
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)
    reports_rl = tmp_path / "reports" / "rl"
    reports_rl.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "status": "research_only",
                "blocker": "",
                "live_enabled": False,
                "rows": 2,
                "features": 6,
                "policy": "safe_quantile_baseline",
            }
        ]
    ).to_csv(reports_rl / "rl_training_report.csv", index=False)

    # Provide minimal upstream artifacts expected by the coaching logic.
    pd.DataFrame(
        [
            {
                "accepted": True,
                "blocker": "",
                "acceptance_reason": "passed",
                "raw_profit_factor": 1.1,
                "rl_profit_factor": 1.4,
                "raw_drawdown": 0.09,
                "rl_drawdown": 0.07,
                "rl_trades": 35,
                "rl_take_rate": 0.22,
                "pair_concentration": 0.5,
                "timeframe_concentration": 0.5,
            }
        ]
    ).to_csv(reports_rl / "rl_acceptance_report.csv", index=False)
    (tmp_path / "reports" / "ml").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports" / "dashboard").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"gated_profit_factor": 1.3, "gated_trades": 30, "gated_drawdown": 0.1, "accepted": True, "blocker": ""}]
    ).to_csv(tmp_path / "reports" / "ml" / "model_gated_acceptance.csv", index=False)
    pd.DataFrame([]).to_csv(tmp_path / "reports" / "dashboard" / "blocked_trades_dashboard.csv", index=False)

    result = run_sequential_thinking_magicka(root=tmp_path, policy_candidates=0, max_recommendations=4)
    recommendations = pd.read_csv(result.paths["recommendations"])
    memory = pd.read_csv(result.paths["sequential_magicka_memory"])

    assert "pair_id" in set(recommendations.columns)
    assert not recommendations.empty
    assert set(recommendations["focus_area"].astype(str)).intersection({"risk_controls", "learning_bootstrap"})
    assert {"cycle_id", "pair_filter", "status", "recommendation_count"}.issubset(set(memory.columns))
    assert int(memory["recommendation_count"].iloc[0]) >= 0


def test_run_brain_cycle_writes_candidates_and_rollup(tmp_path):
    data_ml = tmp_path / "data" / "ml"
    data_ml.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "trade_id": "T1",
                "profit_after_cost": 0.12,
                "entry_abs_zscore": 1.8,
                "trade_bars": 10,
                "hold_bars": 8,
                "realized_return": 0.12,
                "strategy_name": "Static Spread",
                "timeframe": "1h",
                "strategy_id": "S1",
                "entry_bar_index": 1,
                "exit_bar_index": 11,
                "signal_side": "long",
            }
        ]
    ).to_csv(data_ml / "trade_training_dataset.csv", index=False)

    result = run_brain_cycle(root=tmp_path, pair_id="BTC-USD/ETH-USD", policy_candidates=4, max_recommendations=3)
    candidates = pd.read_csv(result.paths["candidate_csv"])
    rollup = pd.read_csv(result.paths["candidate_rollup"])
    summary_text = result.paths["cycle_summary"].read_text(encoding="utf-8")

    assert "cycle_id" in result.summary
    assert not candidates.empty
    assert "candidate_count" in result.summary
    assert int(result.summary["candidate_count"]) >= 1
    assert {"pair", "variant", "entry_logic", "status", "schema_version"}.issubset(set(candidates.columns))
    assert not rollup.empty
    assert "\"status\"" in summary_text
    assert "ready" in summary_text or "blocked" in summary_text


def test_run_brain_cycle_blocks_when_readiness_below_threshold(tmp_path, monkeypatch):
    candidate_summary = pd.DataFrame([{"cycle_id": "2026", "pair": "BTC-USD/ETH-USD", "profit_factor": 1.0, "status": "ready", "entry_threshold": 1.0, "policy_name": "low_edge", "hold_cap_pct": 0.0, "total_return": 0.0, "max_drawdown": 0.0}])
    recommendations = pd.DataFrame(
        [
            {"pair_id": "BTC-USD/ETH-USD", "proposed_change": "lower confidence", "focus_area": "entry", "priority": "low", "cycle_id": "2026"},
            {"pair_id": "BTC-USD/ETH-USD", "proposed_change": "hold more", "focus_area": "exit", "priority": "low", "cycle_id": "2026"},
        ]
    )
    candidate_summary_path = tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    recommendations_path = tmp_path / "reports" / "rl" / "sequential_thinking_magicka_recommendations.csv"
    candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_summary_path.write_text(candidate_summary.to_csv(index=False), encoding="utf-8")
    recommendations_path.write_text(recommendations.to_csv(index=False), encoding="utf-8")

    def fake_magicka(*, root, pair_id, policy_candidates):
        assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
        assert policy_candidates == 4
        return CommandResult(
            summary={"status": "ready"},
            paths={"summary": candidate_summary_path},
        )

    def fake_seq(*, root, pair_id, max_recommendations):
        assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
        assert max_recommendations == 3
        return CommandResult(
            summary={"status": "ready"},
            paths={"recommendations": recommendations_path},
        )

    monkeypatch.setattr("quant_platform.rl.brain_cycle.run_magicka_learning_cycle", fake_magicka)
    monkeypatch.setattr("quant_platform.rl.brain_cycle.run_sequential_thinking_magicka", fake_seq)

    result = run_brain_cycle(
        root=tmp_path,
        pair_id="BTC-USD/ETH-USD",
        policy_candidates=4,
        max_recommendations=3,
        readiness_threshold=0.99,
    )

    assert result.summary["readiness_gate"] == "hold"
    assert result.summary["status"] == "blocked"
    assert result.summary["readiness_score"] < 0.99


def test_run_brain_cycle_readiness_matrix_and_trend_capture(tmp_path, monkeypatch):
    trend_path = tmp_path / "reports" / "brain" / "paper_readiness_trend.csv"

    def make_learning_summary(pf: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "cycle_id": "2026",
                    "pair": "BTC-USD/ETH-USD",
                    "profit_factor": pf,
                    "status": "ready",
                    "entry_threshold": 1.0,
                    "policy_name": "edge_probe",
                    "hold_cap_pct": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "provider": "dydx",
                    "provider_source": "dydx-indexer",
                    "provider_rows": 1200,
                    "provider_quality_score": 0.91,
                }
            ]
        )

    def make_recommendations(high_priority: bool = True, count: int = 2) -> pd.DataFrame:
        priority = "high" if high_priority else "low"
        return pd.DataFrame(
            [
                {
                    "pair_id": "BTC-USD/ETH-USD",
                    "proposed_change": f"rec_{idx + 1}",
                    "focus_area": "entry",
                    "priority": priority,
                    "cycle_id": "2026",
                }
                for idx in range(count)
            ]
        )

    candidate_summary_path = tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    recommendations_path = tmp_path / "reports" / "rl" / "sequential_thinking_magicka_recommendations.csv"
    candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)

    def run_cycle_with_inputs(*, summary: pd.DataFrame, recommendations: pd.DataFrame, readiness_threshold: float) -> dict:
        candidate_summary_path.write_text(summary.to_csv(index=False), encoding="utf-8")
        recommendations_path.write_text(recommendations.to_csv(index=False), encoding="utf-8")

        def fake_magicka(*, root, pair_id, policy_candidates):
            assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
            assert policy_candidates == 4
            return CommandResult(summary={"status": "ready"}, paths={"summary": candidate_summary_path})

        def fake_seq(*, root, pair_id, max_recommendations):
            assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
            assert max_recommendations == 3
            return CommandResult(summary={"status": "ready"}, paths={"recommendations": recommendations_path})

        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_magicka_learning_cycle", fake_magicka)
        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_sequential_thinking_magicka", fake_seq)

        return run_brain_cycle(
            root=tmp_path,
            pair_id="BTC-USD/ETH-USD",
            policy_candidates=4,
            max_recommendations=3,
            readiness_threshold=readiness_threshold,
        ).summary

    scenarios = [
        ("low_quality_blocked", 0.75, make_learning_summary(1.0), make_recommendations(high_priority=True, count=2), "blocked", "hold"),
        ("high_quality_pass", 0.75, make_learning_summary(4.0), make_recommendations(high_priority=True, count=2), "ready", "pass"),
        ("high_quality_hold_strict", 0.95, make_learning_summary(4.0), make_recommendations(high_priority=True, count=2), "blocked", "hold"),
    ]

    rows: list[dict[str, object]] = []
    for pair_name, threshold, summary, recommendations, expected_status, expected_gate in scenarios:
        run_summary = run_cycle_with_inputs(
            summary=summary,
            recommendations=recommendations,
            readiness_threshold=threshold,
        )
        row = {
            "scenario": pair_name,
            "readiness_threshold": threshold,
            "status": run_summary["status"],
            "readiness_gate": run_summary["readiness_gate"],
            "ready_count": run_summary["ready_count"],
            "blocked_count": run_summary["blocked_count"],
            "readiness_score": run_summary["readiness_score"],
            "candidate_count": run_summary["candidate_count"],
        }
        assert run_summary["status"] == expected_status
        assert run_summary["readiness_gate"] == expected_gate
        rows.append(row)

    trend_frame = pd.DataFrame(rows)
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    trend_frame.to_csv(trend_path, index=False)

    saved = pd.read_csv(trend_path)
    assert len(saved) == len(scenarios)
    assert list(saved["scenario"]) == ["low_quality_blocked", "high_quality_pass", "high_quality_hold_strict"]
    assert set(saved["status"]) == {"ready", "blocked"}
    assert list(saved["readiness_gate"]) == ["hold", "pass", "hold"]
    assert saved["readiness_score"].iloc[1] > saved["readiness_score"].iloc[0]
    assert saved["readiness_score"].iloc[2] < 1.0


def test_run_brain_cycle_appends_readiness_trend(tmp_path, monkeypatch):
    candidate_summary_path = tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    recommendations_path = tmp_path / "reports" / "rl" / "sequential_thinking_magicka_recommendations.csv"
    candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_summary_path.write_text(
        pd.DataFrame(
            [
                {
                    "cycle_id": "20260628T001",
                    "pair": "BTC-USD/ETH-USD",
                    "profit_factor": 4.0,
                    "status": "ready",
                    "entry_threshold": 1.0,
                    "policy_name": "edge_probe",
                    "hold_cap_pct": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "provider": "dydx",
                    "provider_source": "history_dydx",
                    "provider_rows": 1500,
                    "provider_quality_score": 0.91,
                }
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )
    recommendations_path.write_text(
        pd.DataFrame(
            [
                {
                    "pair_id": "BTC-USD/ETH-USD",
                    "proposed_change": "increase_window",
                    "focus_area": "entry",
                    "priority": "high",
                    "cycle_id": "20260628T001",
                }
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )

    def fake_magicka(*, root, pair_id, policy_candidates):
        return CommandResult(summary={"status": "ready"}, paths={"summary": candidate_summary_path})

    def fake_seq(*, root, pair_id, max_recommendations):
        return CommandResult(summary={"status": "ready"}, paths={"recommendations": recommendations_path})

    monkeypatch.setattr("quant_platform.rl.brain_cycle.run_magicka_learning_cycle", fake_magicka)
    monkeypatch.setattr("quant_platform.rl.brain_cycle.run_sequential_thinking_magicka", fake_seq)

    run_brain_cycle(root=tmp_path, pair_id="BTC-USD/ETH-USD", policy_candidates=2, max_recommendations=1)
    run_brain_cycle(root=tmp_path, pair_id="BTC-USD/ETH-USD", policy_candidates=2, max_recommendations=1)

    trend = pd.read_csv(tmp_path / "reports" / "brain" / "paper_readiness_trend.csv")
    assert len(trend) == 2
    assert set(trend["status"]).issubset({"ready", "blocked"})
def test_brain_readiness_report_appends_readiness_scorecard(tmp_path, monkeypatch):
    scorecard_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard.csv"
    scorecard_source_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard_rollup.csv"

    def make_learning_summary(pf: float, status: str = "ready") -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "cycle_id": "2026",
                    "pair": "BTC-USD/ETH-USD",
                    "profit_factor": pf,
                    "status": status,
                    "entry_threshold": 1.0,
                    "policy_name": "edge_probe",
                    "hold_cap_pct": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                    "provider": "dydx",
                    "provider_source": "dydx-indexer",
                    "provider_rows": 1200,
                    "provider_quality_score": 0.91,
                }
            ]
        )

    def make_recommendations(priority: str = "low", count: int = 2) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pair_id": "BTC-USD/ETH-USD",
                    "proposed_change": f"rec_{idx + 1}",
                    "focus_area": "entry",
                    "priority": priority,
                    "cycle_id": "2026",
                }
                for idx in range(count)
            ]
        )

    candidate_summary_path = tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    recommendations_path = tmp_path / "reports" / "rl" / "sequential_thinking_magicka_recommendations.csv"
    candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)

    def run_cycle_and_capture(summary: pd.DataFrame, recommendations: pd.DataFrame, readiness_threshold: float) -> dict:
        candidate_summary_path.write_text(summary.to_csv(index=False), encoding="utf-8")
        recommendations_path.write_text(recommendations.to_csv(index=False), encoding="utf-8")

        def fake_magicka(*, root, pair_id, policy_candidates):
            assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
            assert policy_candidates == 4
            return CommandResult(summary={"status": "ready"}, paths={"summary": candidate_summary_path})

        def fake_seq(*, root, pair_id, max_recommendations):
            assert pair_id == "BTC-USD/ETH-USD".replace("/", "-")
            assert max_recommendations == 3
            return CommandResult(summary={"status": "ready"}, paths={"recommendations": recommendations_path})

        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_magicka_learning_cycle", fake_magicka)
        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_sequential_thinking_magicka", fake_seq)

        run_result = run_brain_cycle(
            root=tmp_path,
            pair_id="BTC-USD/ETH-USD",
            policy_candidates=4,
            max_recommendations=3,
            readiness_threshold=readiness_threshold,
        )
        scorecard_source_path.parent.mkdir(parents=True, exist_ok=True)
        source = pd.read_csv(run_result.paths["candidate_csv"]).copy()
        source["cycle_id"] = str(run_result.summary["cycle_id"])
        source.to_csv(scorecard_source_path, index=False)
        report = build_brain_readiness_report(
            root=tmp_path,
            candidate_rollup_path=scorecard_source_path,
            score_threshold=readiness_threshold,
        ).summary
        return {"run": run_result.summary, "report": report}

    scenarios = [
        {
            "name": "low_quality_strict_gate",
            "threshold": 0.8,
            "learning": make_learning_summary(1.0, status="blocked"),
            "recommendations": make_recommendations("low", count=2),
            "expected_run_status": "blocked",
            "expected_gate": "hold",
        },
        {
            "name": "high_quality_mid_gate",
            "threshold": 0.8,
            "learning": make_learning_summary(4.0, status="ready"),
            "recommendations": make_recommendations("low", count=2),
            "expected_run_status": "ready",
            "expected_gate": "pass",
        },
        {
            "name": "high_quality_strict_gate",
            "threshold": 0.95,
            "learning": make_learning_summary(4.0, status="ready"),
            "recommendations": make_recommendations("low", count=2),
            "expected_run_status": "blocked",
            "expected_gate": "hold",
        },
    ]

    score_rows: list[dict[str, object]] = []
    for scenario in scenarios:
        captured = run_cycle_and_capture(
            summary=scenario["learning"],
            recommendations=scenario["recommendations"],
            readiness_threshold=scenario["threshold"],
        )
        run_summary = captured["run"]
        report_summary = captured["report"]
        assert run_summary["status"] == scenario["expected_run_status"]
        assert run_summary["readiness_gate"] == scenario["expected_gate"]
        assert report_summary["score_gate"] == scenario["expected_gate"]
        score_rows.append(
            {
                "scenario": scenario["name"],
                "pair": "BTC-USD/ETH-USD",
                "threshold": scenario["threshold"],
                "status": run_summary["status"],
                "ready_count": run_summary["ready_count"],
                "blocked_count": run_summary["blocked_count"],
                "readiness_score": run_summary["readiness_score"],
                "readiness_gate": run_summary["readiness_gate"],
                "provider": "dydx",
                "provider_source": "dydx-indexer",
                "provider_rows": 1200,
                "provider_quality_score": 0.91,
            }
        )

    scorecard = pd.DataFrame(score_rows)
    scorecard["run_pass"] = scorecard["readiness_gate"].eq("pass").astype(int)
    scorecard["pass_rate"] = scorecard["run_pass"].mean()
    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(scorecard_path, index=False)

    saved = pd.read_csv(scorecard_path)
    assert len(saved) == len(scenarios)
    assert set(saved["readiness_gate"]) == {"hold", "pass"}
    assert saved["pass_rate"].iloc[-1] == pytest.approx(1 / 3)
    assert saved["readiness_score"].iloc[1] > saved["readiness_score"].iloc[0]
    assert saved["readiness_score"].iloc[2] >= saved["readiness_score"].iloc[1]


def test_brain_readiness_scorecard_is_append_only_and_rolling_rate(tmp_path, monkeypatch):
    scorecard_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard.csv"
    scorecard_source_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard_rollup.csv"
    candidate_summary_path = tmp_path / "reports" / "rl" / "rl_learning_cycle_summary.csv"
    recommendations_path = tmp_path / "reports" / "rl" / "sequential_thinking_magicka_recommendations.csv"
    candidate_summary_path.parent.mkdir(parents=True, exist_ok=True)

    def make_learning_summary(pf: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "cycle_id": "2026",
                    "pair": "BTC-USD/ETH-USD",
                    "profit_factor": pf,
                    "status": "ready",
                    "entry_threshold": 1.0,
                    "policy_name": "edge_probe",
                    "hold_cap_pct": 0.0,
                    "total_return": 0.0,
                    "max_drawdown": 0.0,
                }
            ]
        )

    def make_recommendations() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "pair_id": "BTC-USD/ETH-USD",
                    "proposed_change": "rec_1",
                    "focus_area": "entry",
                    "priority": "low",
                    "cycle_id": "2026",
                },
                {
                    "pair_id": "BTC-USD/ETH-USD",
                    "proposed_change": "rec_2",
                    "focus_area": "exit",
                    "priority": "low",
                    "cycle_id": "2026",
                },
            ]
        )

    def run_and_append(pf: float, threshold: float) -> dict:
        candidate_summary_path.write_text(make_learning_summary(pf).to_csv(index=False), encoding="utf-8")
        recommendations_path.write_text(make_recommendations().to_csv(index=False), encoding="utf-8")

        def fake_magicka(*, root, pair_id, policy_candidates):
            return CommandResult(summary={"status": "ready"}, paths={"summary": candidate_summary_path})

        def fake_seq(*, root, pair_id, max_recommendations):
            return CommandResult(summary={"status": "ready"}, paths={"recommendations": recommendations_path})

        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_magicka_learning_cycle", fake_magicka)
        monkeypatch.setattr("quant_platform.rl.brain_cycle.run_sequential_thinking_magicka", fake_seq)

        run_result = run_brain_cycle(
            root=tmp_path,
            pair_id="BTC-USD/ETH-USD",
            policy_candidates=4,
            max_recommendations=3,
            readiness_threshold=threshold,
        )
        source = pd.read_csv(run_result.paths["candidate_csv"]).copy()
        source["cycle_id"] = str(run_result.summary["cycle_id"])
        source.to_csv(scorecard_source_path, index=False)
        report = build_brain_readiness_report(
            root=tmp_path,
            candidate_rollup_path=scorecard_source_path,
            score_threshold=threshold,
        ).summary
        return {"run": run_result.summary, "report": report}

    run_rows = [
        run_and_append(1.2, 0.8),
        run_and_append(4.0, 0.8),
        run_and_append(4.0, 0.95),
    ]

    scorecard_path.parent.mkdir(parents=True, exist_ok=True)
    # First write
    first_frame = pd.DataFrame(
        [
            {
                "scenario": "first_window",
                "pair": "BTC-USD/ETH-USD",
                "threshold": 0.8,
                "status": r["run"]["status"],
                "ready_count": r["run"]["ready_count"],
                "blocked_count": r["run"]["blocked_count"],
                "readiness_score": r["run"]["readiness_score"],
                "readiness_gate": r["run"]["readiness_gate"],
                "provider": "dydx",
                "provider_source": "dydx-indexer",
                "provider_rows": 1200,
                "provider_quality_score": 0.91,
            }
            for r in run_rows[:2]
        ]
    )
    first_frame["run_pass"] = first_frame["readiness_gate"].eq("pass").astype(int)
    first_frame["pass_rate"] = first_frame["run_pass"].expanding().mean()
    first_frame.to_csv(scorecard_path, index=False)

    # Append a new run
    appended = pd.DataFrame(
        [
            {
                "scenario": "appended_window",
                "pair": "BTC-USD/ETH-USD",
                "threshold": 0.95,
                "status": run_rows[2]["run"]["status"],
                "ready_count": run_rows[2]["run"]["ready_count"],
                "blocked_count": run_rows[2]["run"]["blocked_count"],
                "readiness_score": run_rows[2]["run"]["readiness_score"],
                "readiness_gate": run_rows[2]["run"]["readiness_gate"],
                "provider": "dydx",
                "provider_source": "dydx-indexer",
                "provider_rows": 1200,
                "provider_quality_score": 0.91,
            }
        ]
    )
    appended["run_pass"] = appended["readiness_gate"].eq("pass").astype(int)
    appended["pass_rate"] = (first_frame["run_pass"].sum() + appended["run_pass"]) / 3
    appended.to_csv(scorecard_path, mode="a", index=False, header=False)

    combined = pd.read_csv(scorecard_path)
    assert len(combined) == 3
    assert combined.iloc[0]["status"] in {"ready", "blocked"}
    assert combined["pass_rate"].notna().any()
    assert combined["pass_rate"].iloc[-1] == pytest.approx(1 / 3)

    assert all(combined["provider"] == "dydx")

    # rolling check: last window computed only from available rows and stays within [0, 1]
    assert all(0.0 <= float(v) <= 1.0 for v in combined["pass_rate"] if pd.notna(v))


def test_build_brain_readiness_report_rolls_up_provider_context(tmp_path):
    scorecard_source_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard_rollup.csv"
    scorecard_source_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "cycle_id": "20260628T001",
                "pair": "ETH-USD/BTC-USD",
                "status": "ready",
                "confidence": 0.66,
                "provider": "dydx",
                "provider_source": "history_dydx",
                "provider_rows": 800,
                "provider_quality_score": 0.89,
            },
            {
                "cycle_id": "20260628T001",
                "pair": "ETH-USD/BTC-USD",
                "status": "ready",
                "confidence": 0.84,
                "provider": "binance",
                "provider_source": "spot_history",
                "provider_rows": 240,
                "provider_quality_score": 0.76,
            },
        ]
    ).to_csv(scorecard_source_path, index=False)

    result = build_brain_readiness_report(
        root=tmp_path,
        candidate_rollup_path=scorecard_source_path,
        score_threshold=0.7,
    )

    assert result.summary["provider"] == "binance;dydx"
    assert result.summary["provider_source"] == "history_dydx;spot_history"
    assert result.summary["provider_rows"] == 1040
    assert result.summary["provider_mix"] == "mixed"
    assert result.summary["provider_quality_score"] == pytest.approx((0.89 + 0.76) / 2)


def test_brain_readiness_provider_quality_trend(tmp_path):
    scorecard_source_path = tmp_path / "reports" / "brain" / "paper_readiness_scorecard_rollup.csv"
    trend_path = tmp_path / "reports" / "brain" / "provider_quality_trend.csv"
    scorecard_source_path.parent.mkdir(parents=True, exist_ok=True)

    source_a = pd.DataFrame(
        [
            {
                "cycle_id": "20260628T0100Z",
                "pair": "ETH-USD/BTC-USD",
                "status": "ready",
                "confidence": 0.75,
                "provider": "dydx",
                "provider_source": "history_dydx",
                "provider_rows": 1100,
                "provider_quality_score": 0.82,
            },
            {
                "cycle_id": "20260628T0100Z",
                "pair": "SOL-USD/AVAX-USD",
                "status": "ready",
                "confidence": 0.55,
                "provider": "dydx",
                "provider_source": "history_dydx",
                "provider_rows": 900,
                "provider_quality_score": 0.70,
            },
        ]
    )

    source_b = pd.DataFrame(
        [
            {
                "cycle_id": "20260628T0200Z",
                "pair": "ETH-USD/BTC-USD",
                "status": "ready",
                "confidence": 0.80,
                "provider": "binance",
                "provider_source": "spot_history",
                "provider_rows": 950,
                "provider_quality_score": 0.68,
            },
            {
                "cycle_id": "20260628T0200Z",
                "pair": "SOL-USD/AVAX-USD",
                "status": "ready",
                "confidence": 0.79,
                "provider": "binance",
                "provider_source": "spot_history",
                "provider_rows": 1050,
                "provider_quality_score": 0.74,
            },
        ]
    )

    scorecard_source_path.write_text(source_a.to_csv(index=False), encoding="utf-8")
    report_a = build_brain_readiness_report(
        root=tmp_path,
        candidate_rollup_path=scorecard_source_path,
        score_threshold=0.7,
    ).summary

    scorecard_source_path.write_text(source_b.to_csv(index=False), encoding="utf-8")
    report_b = build_brain_readiness_report(
        root=tmp_path,
        candidate_rollup_path=scorecard_source_path,
        score_threshold=0.7,
    ).summary

    trend = pd.DataFrame(
        [
            {
                "cycle_id": report_a["cycle_id"],
                "provider": "dydx",
                "provider_quality_score": report_a["provider_quality_score"],
                "provider_rows": report_a["provider_rows"],
            },
            {
                "cycle_id": report_b["cycle_id"],
                "provider": "binance",
                "provider_quality_score": report_b["provider_quality_score"],
                "provider_rows": report_b["provider_rows"],
            },
        ]
    )
    trend["quality_delta_vs_prev"] = trend["provider_quality_score"].diff()
    trend.to_csv(trend_path, index=False)

    saved = pd.read_csv(trend_path)
    assert list(saved["provider"]) == ["dydx", "binance"]
    assert saved.loc[0, "provider_quality_score"] == pytest.approx((0.82 + 0.70) / 2)
    assert saved.loc[1, "provider_quality_score"] == pytest.approx((0.68 + 0.74) / 2)
    assert saved.loc[1, "quality_delta_vs_prev"] == pytest.approx((0.68 + 0.74) / 2 - (0.82 + 0.70) / 2)
    assert pd.isna(saved.loc[0, "quality_delta_vs_prev"])
    assert report_a["provider"] == "dydx"
    assert report_b["provider"] == "binance"
