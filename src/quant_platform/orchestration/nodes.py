from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import (
    ROOT,
    build_artifact_index,
    build_command_dashboard,
    build_market_venue_context,
    build_multi_venue_history_readiness,
    build_pair_universe,
    build_trade_dataset,
    build_venue_route_scorecard,
    current_state,
    export_trade_gate_model,
    run_model_gated_backtest,
    system_check,
    train_trade_gate,
)
from quant_platform.hyperliquid import (
    build_hyperliquid_evidence_cadence,
    build_hyperliquid_research_bundle,
    refresh_hyperliquid_execution_cost_snapshot,
    refresh_hyperliquid_market_context,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.dynamic_stage_runner import DYNAMIC_STAGE_NAMES, run_dynamic_stage
from quant_platform.orchestration.mini_agents import build_mini_agent_orchestration
from quant_platform.orchestration.orchestrator_assistant import build_orchestrator_assistant
from quant_platform.orchestration.reporting import (
    stage_result_from_command,
    write_project_spine_audit,
)
from quant_platform.orchestration.specialist_scoreboard import build_specialist_scoreboard
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus
from quant_platform.rl.rl_backtest import run_rl_research
from quant_platform.rl.rl_idea_engine import run_rl_idea_scout
from quant_platform.rl.rl_learning_agent import (
    run_magicka_learning_cycle,
    run_sequential_thinking_magicka,
)
from quant_platform.wizard_control_plane import build_wizard_control_plane
from quant_platform.wizard_evidence import build_wizard_replay_handoff, build_wizard_research_pack
from quant_platform.wizard_hyperliquid_bridge import build_hyperliquid_wizard_hypothesis_queue
from quant_platform.wizard_local_verification import verify_wizard_local_mode
from quant_platform.youtube_brain import run_youtube_brain_cycle

STAGE_GROUPS: dict[str, list[str]] = {
    "all": [
        "system_check",
        "build_artifact_index",
        "current_state",
        "project_spine_audit",
        "wizard_control_plane",
        "discover_wizard_candidates",
        "youtube_research_brain",
        "build_multi_venue_history_readiness",
        "build_wizard_replay_handoff",
        "verify_wizard_local_mode",
        "build_pair_universe",
        "build_trade_dataset",
        "train_trade_gate",
        "run_model_gated_backtest",
        "run_rl_research",
        "run_rl_idea_scout",
        "run_magicka_learning",
        "run_sequential_thinking_magicka",
        "mini_agents",
        "orchestrator_assistant",
        "specialist_scoreboard",
        "export_model",
        "supreme_team_checkpoint",
        "build_dashboard",
        "paper_trade_readiness_check",
    ],
    "discovery": ["build_artifact_index", "current_state", "wizard_control_plane", "discover_wizard_candidates", "youtube_research_brain", "build_multi_venue_history_readiness", "build_wizard_replay_handoff", "build_pair_universe"],
    "verification": ["system_check", "project_spine_audit", "verify_wizard_local_mode", "build_pair_universe"],
    "venue_evidence": [
        "wizard_control_plane",
        "discover_wizard_candidates",
        "build_multi_venue_history_readiness",
        "refresh_hyperliquid_market_context",
        "build_hyperliquid_research_bundle",
        "build_hyperliquid_wizard_hypothesis_queue",
        "refresh_hyperliquid_execution_cost_snapshot",
        "build_hyperliquid_evidence_cadence",
        "build_market_venue_context",
        "build_venue_route_scorecard",
        "build_wizard_replay_handoff",
        "current_state",
        "build_dashboard",
    ],
    "model": ["build_trade_dataset", "train_trade_gate", "run_model_gated_backtest", "export_model"],
    "rl": [
        "run_rl_research",
        "run_rl_idea_scout",
        "run_magicka_learning",
        "run_sequential_thinking_magicka",
        "mini_agents",
        "orchestrator_assistant",
        "specialist_scoreboard",
    ],
    "agents": ["youtube_research_brain", "mini_agents", "orchestrator_assistant", "specialist_scoreboard"],
    "copula_shadow": ["copula_shadow_comparison", "dynamic_rollout_gate"],
    "dynamic_rollout": ["dynamic_rollout_gate"],
    "dynamic_supreme_team": ["dynamic_supreme_team_checkpoint"],
    "dashboard": ["build_dashboard"],
    "monitoring": ["system_check", "wizard_control_plane", "current_state", "build_monitor_dashboard"],
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
        "build_monitor_dashboard",
        "run_rl_research",
        "run_rl_idea_scout",
        "run_sequential_thinking_magicka",
        "wizard_control_plane",
        "mini_agents",
        "orchestrator_assistant",
        "specialist_scoreboard",
        "copula_shadow_comparison",
        "dynamic_rollout_gate",
        "dynamic_supreme_team_checkpoint",
        "supreme_team_checkpoint",
    }:
        return StageResult(stage=stage, status=StageStatus.SKIPPED, reason="report_only_mode", next_step="run without --report-only")
    if stage in DYNAMIC_STAGE_NAMES:
        return run_dynamic_stage(stage, state, root)

    stage_functions: dict[str, Callable[[], StageResult]] = {
        "system_check": lambda: stage_result_from_command(stage, system_check(root=root)),
        "build_artifact_index": lambda: stage_result_from_command(stage, build_artifact_index(root=root)),
        "current_state": lambda: stage_result_from_command(stage, current_state(root=root)),
        "project_spine_audit": lambda: _spine_audit_result(stage, root),
        "wizard_control_plane": lambda: _wizard_control_plane_result(stage, root),
        "discover_wizard_candidates": lambda: _wizard_discovery_result(stage, root),
        "youtube_research_brain": lambda: stage_result_from_command(
            stage, run_youtube_brain_cycle(root=root)
        ),
        "build_multi_venue_history_readiness": lambda: stage_result_from_command(stage, build_multi_venue_history_readiness(root=root)),
        "build_wizard_replay_handoff": lambda: stage_result_from_command(stage, build_wizard_replay_handoff(root=root)),
        "verify_wizard_local_mode": lambda: stage_result_from_command(stage, verify_wizard_local_mode(root=root)),
        "build_pair_universe": lambda: stage_result_from_command(stage, build_pair_universe(root=root)),
        "refresh_hyperliquid_market_context": lambda: stage_result_from_command(stage, refresh_hyperliquid_market_context(root=root)),
        "build_hyperliquid_research_bundle": lambda: stage_result_from_command(stage, build_hyperliquid_research_bundle(root=root)),
        "build_hyperliquid_wizard_hypothesis_queue": lambda: stage_result_from_command(
            stage, build_hyperliquid_wizard_hypothesis_queue(root=root)
        ),
        "refresh_hyperliquid_execution_cost_snapshot": lambda: stage_result_from_command(
            stage, refresh_hyperliquid_execution_cost_snapshot(root=root)
        ),
        "build_hyperliquid_evidence_cadence": lambda: stage_result_from_command(stage, build_hyperliquid_evidence_cadence(root=root)),
        "build_market_venue_context": lambda: stage_result_from_command(stage, build_market_venue_context(root=root)),
        "build_venue_route_scorecard": lambda: stage_result_from_command(stage, build_venue_route_scorecard(root=root)),
        "build_trade_dataset": lambda: stage_result_from_command(stage, build_trade_dataset(root=root)),
        "train_trade_gate": lambda: stage_result_from_command(stage, train_trade_gate(root=root)),
        "run_model_gated_backtest": lambda: stage_result_from_command(stage, run_model_gated_backtest(root=root)),
        "run_rl_research": lambda: stage_result_from_command(stage, run_rl_research(root=root, pair_id=state.pair_id)),
        "run_magicka_learning": lambda: stage_result_from_command(
            stage, run_magicka_learning_cycle(root=root, pair_id=state.pair_id, policy_candidates=12)
        ),
        "run_sequential_thinking_magicka": lambda: stage_result_from_command(
            stage, run_sequential_thinking_magicka(root=root, pair_id=state.pair_id, max_recommendations=12)
        ),
        "run_rl_idea_scout": lambda: stage_result_from_command(stage, run_rl_idea_scout(root=root)),
        "mini_agents": lambda: stage_result_from_command(stage, build_mini_agent_orchestration(root=root)),
        "orchestrator_assistant": lambda: stage_result_from_command(stage, build_orchestrator_assistant(root=root)),
        "specialist_scoreboard": lambda: stage_result_from_command(stage, build_specialist_scoreboard(root=root)),
        "copula_shadow_comparison": lambda: _copula_shadow_result(stage, state=state, root=root),
        "dynamic_rollout_gate": lambda: _dynamic_rollout_result(stage, root=root),
        "dynamic_supreme_team_checkpoint": lambda: _dynamic_supreme_team_result(stage, root=root),
        "export_model": lambda: stage_result_from_command(stage, export_trade_gate_model(root=root)),
        "build_dashboard": lambda: stage_result_from_command(stage, build_command_dashboard(root=root)),
        "build_monitor_dashboard": lambda: stage_result_from_command(
            stage, build_command_dashboard(root=root, refresh_profile="monitor")
        ),
        "supreme_team_checkpoint": lambda: _supreme_team_result(stage, root=root),
        "paper_trade_readiness_check": lambda: _paper_readiness_checkpoint_result(stage, root=root),
    }
    if stage not in stage_functions:
        return StageResult(stage=stage, status=StageStatus.FAILED, blocker="unknown_stage", next_step="check run-orchestrator --stage")
    try:
        return stage_functions[stage]()
    except Exception as exc:
        return StageResult(stage=stage, status=StageStatus.FAILED, blocker=type(exc).__name__, reason=safe_exception_code(exc), next_step="inspect stage evidence and rerun")


def _spine_audit_result(stage: str, root: Path) -> StageResult:
    path = write_project_spine_audit(root)
    return StageResult(stage=stage, status=StageStatus.PASSED, reason="spine_audit_written", evidence_path=str(path), rows=1)


def _wizard_control_plane_result(stage: str, root: Path) -> StageResult:
    result = build_wizard_control_plane(root=root)
    ready = bool(result.summary.get("ready", False))
    return StageResult(
        stage=stage,
        status=StageStatus.PASSED if ready else StageStatus.BLOCKED,
        blocker="" if ready else str(result.summary.get("blocker", "wizard_control_plane_blocked")),
        reason="wizard_ranking_authority_ready" if ready else "wizard_ranking_authority_blocked",
        evidence_path=str(result.paths["wizard_control_plane_summary"]),
        next_step=(
            "build the Wizard research pack"
            if ready
            else "execute a complete fresh Wizard sweep, then rerun wizard-control-plane"
        ),
        rows=int(result.summary.get("candidate_rows", 0)),
    )


def _wizard_discovery_result(stage: str, root: Path) -> StageResult:
    control = build_wizard_control_plane(root=root)
    if not bool(control.summary.get("ready", False)):
        return StageResult(
            stage=stage,
            status=StageStatus.BLOCKED,
            blocker=str(control.summary.get("blocker", "wizard_control_plane_blocked")),
            reason="wizard_discovery_not_rebuilt_from_partial_or_stale_sweep",
            evidence_path=str(control.paths["wizard_control_plane_summary"]),
            next_step="execute a complete fresh Wizard sweep, then rerun discovery",
            rows=0,
        )
    return stage_result_from_command(stage, build_wizard_research_pack(root=root))


def _copula_shadow_result(stage: str, *, state: OrchestratorState, root: Path) -> StageResult:
    return run_dynamic_stage(stage, state, root)


def _dynamic_rollout_result(stage: str, *, root: Path) -> StageResult:
    return run_dynamic_stage(stage, OrchestratorState(report_only=True, root=root), root)


def _dynamic_supreme_team_result(stage: str, *, root: Path) -> StageResult:
    return run_dynamic_stage(stage, OrchestratorState(report_only=True, root=root), root)


def _paper_readiness_checkpoint_result(stage: str, root: Path) -> StageResult:
    from quant_platform import cli

    original_root = cli.ROOT
    try:
        cli.ROOT = root
        result = cli.paper_readiness_checkpoint(readiness_threshold=0.65)
    finally:
        cli.ROOT = original_root

    summary = result.get("summary", {})
    ready_rows = int(summary.get("ready_rows", 0))
    total_rows = ready_rows + int(summary.get("blocked_rows", 0))
    passed = total_rows > 0 and ready_rows == total_rows
    checkpoint_path = result["paths"].get("paper_readiness_checkpoint", root / "reports" / "paper_readiness_checkpoint.csv")

    return StageResult(
        stage=stage,
        status=StageStatus.PASSED if passed else StageStatus.BLOCKED,
        blocker="" if passed else "paper_readiness_checkpoint_not_ready",
        reason="paper_readiness_checkpoint_completed" if passed else "paper_readiness_checkpoint_blocked",
        evidence_path=str(checkpoint_path),
        next_step="continue to paper readiness planning" if passed else "review blockers and rerun",
        rows=total_rows,
    )


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
