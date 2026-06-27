from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import (
    ROOT,
    build_artifact_index,
    build_command_dashboard,
    build_pair_universe,
    build_trade_dataset,
    current_state,
    export_trade_gate_model,
    run_model_gated_backtest,
    system_check,
    train_trade_gate,
)
from quant_platform.orchestration.reporting import stage_result_from_command, write_project_spine_audit
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.rl.rl_backtest import run_rl_research
from quant_platform.rl.rl_idea_engine import run_rl_idea_scout
from quant_platform.wizard_evidence import build_wizard_research_pack
from quant_platform.wizard_local_verification import verify_wizard_local_mode


STAGE_GROUPS: dict[str, list[str]] = {
    "all": [
        "system_check",
        "build_artifact_index",
        "current_state",
        "project_spine_audit",
        "discover_wizard_candidates",
        "verify_wizard_local_mode",
        "build_pair_universe",
        "build_trade_dataset",
        "train_trade_gate",
        "run_model_gated_backtest",
        "run_rl_research",
        "run_rl_idea_scout",
        "mini_agents",
        "orchestrator_assistant",
        "specialist_scoreboard",
        "export_model",
        "supreme_team_checkpoint",
        "build_dashboard",
        "paper_trade_readiness_check",
    ],
    "discovery": ["build_artifact_index", "current_state", "discover_wizard_candidates", "build_pair_universe"],
    "verification": ["system_check", "project_spine_audit", "verify_wizard_local_mode", "build_pair_universe"],
    "model": ["build_trade_dataset", "train_trade_gate", "run_model_gated_backtest", "export_model"],
    "rl": ["run_rl_research", "run_rl_idea_scout", "mini_agents", "orchestrator_assistant", "specialist_scoreboard"],
    "agents": ["mini_agents", "orchestrator_assistant", "specialist_scoreboard"],
    "dashboard": ["build_dashboard"],
    "supreme_team": ["supreme_team_checkpoint"],
}


def stages_for_group(group: str) -> list[str]:
    return STAGE_GROUPS.get(group, [group])


def run_stage(stage: str, state: OrchestratorState, root: Path = ROOT) -> StageResult:
    if state.dry_run:
        return StageResult(stage=stage, status=StageStatus.DRY_RUN, reason="stage_would_run", next_step="remove --dry-run to execute")
    if state.report_only and stage not in {
        "project_spine_audit",
        "build_dashboard",
        "run_rl_research",
        "run_rl_idea_scout",
        "mini_agents",
        "orchestrator_assistant",
        "specialist_scoreboard",
        "supreme_team_checkpoint",
    }:
        return StageResult(stage=stage, status=StageStatus.SKIPPED, reason="report_only_mode", next_step="run without --report-only")

    stage_functions: dict[str, Callable[[], StageResult]] = {
        "system_check": lambda: stage_result_from_command(stage, system_check(root=root)),
        "build_artifact_index": lambda: stage_result_from_command(stage, build_artifact_index(root=root)),
        "current_state": lambda: stage_result_from_command(stage, current_state(root=root)),
        "project_spine_audit": lambda: _spine_audit_result(stage, root),
        "discover_wizard_candidates": lambda: stage_result_from_command(stage, build_wizard_research_pack(root=root)),
        "verify_wizard_local_mode": lambda: stage_result_from_command(stage, verify_wizard_local_mode(root=root)),
        "build_pair_universe": lambda: stage_result_from_command(stage, build_pair_universe(root=root)),
        "build_trade_dataset": lambda: stage_result_from_command(stage, build_trade_dataset(root=root)),
        "train_trade_gate": lambda: stage_result_from_command(stage, train_trade_gate(root=root)),
        "run_model_gated_backtest": lambda: stage_result_from_command(stage, run_model_gated_backtest(root=root)),
        "run_rl_research": lambda: stage_result_from_command(stage, run_rl_research(root=root, pair_id=state.pair_id)),
        "run_rl_idea_scout": lambda: stage_result_from_command(stage, run_rl_idea_scout(root=root)),
        "mini_agents": lambda: stage_result_from_command(stage, build_mini_agent_orchestration(root=root)),
        "orchestrator_assistant": lambda: stage_result_from_command(stage, build_orchestrator_assistant(root=root)),
        "specialist_scoreboard": lambda: stage_result_from_command(stage, build_specialist_scoreboard(root=root)),
        "export_model": lambda: stage_result_from_command(stage, export_trade_gate_model(root=root)),
        "build_dashboard": lambda: stage_result_from_command(stage, build_command_dashboard(root=root)),
        "supreme_team_checkpoint": lambda: _supreme_team_result(stage, root=root),
        "paper_trade_readiness_check": lambda: StageResult(
            stage=stage,
            status=StageStatus.BLOCKED,
            blocker="paper_live_blocked_until_all_gates_pass",
            reason="orchestrator_v1_is_research_only",
            evidence_path=str(root / "reports" / "dashboard" / "blocked_trades_dashboard.csv"),
            next_step="review dashboard blockers before paper trading",
        ),
    }
    if stage not in stage_functions:
        return StageResult(stage=stage, status=StageStatus.FAILED, blocker="unknown_stage", next_step="check run-orchestrator --stage")
    try:
        return stage_functions[stage]()
    except Exception as exc:
        return StageResult(stage=stage, status=StageStatus.FAILED, blocker=type(exc).__name__, reason=str(exc), next_step="inspect stage evidence and rerun")


def _spine_audit_result(stage: str, root: Path) -> StageResult:
    path = write_project_spine_audit(root)
    return StageResult(stage=stage, status=StageStatus.PASSED, reason="spine_audit_written", evidence_path=str(path), rows=1)


def _supreme_team_result(stage: str, root: Path) -> StageResult:
    from quant_platform import cli

    original_root = cli.ROOT
    try:
        cli.ROOT = root
        csv_path, md_path = cli.print_supreme_team_checkpoint(run_dir=root / "reports" / "supreme_team")
    finally:
        cli.ROOT = original_root
    summary = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    return StageResult(
        stage=stage,
        status=StageStatus.PASSED,
        reason="supreme_team_checkpoint_written",
        evidence_path=f"{csv_path};{md_path}",
        rows=int(len(summary)),
    )
