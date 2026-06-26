from __future__ import annotations

from pathlib import Path

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.orchestration.nodes import run_stage, stages_for_group
from quant_platform.orchestration.reporting import write_orchestrator_reports
from quant_platform.orchestration.state import OrchestratorState, StageStatus


def run_orchestrator(
    *,
    stage: str = "all",
    pair_id: str = "",
    dry_run: bool = False,
    force_refresh: bool = False,
    fail_fast: bool = False,
    report_only: bool = False,
    root: Path = ROOT,
) -> CommandResult:
    state = OrchestratorState(
        stage_group=stage,
        pair_id=pair_id,
        dry_run=dry_run,
        force_refresh=force_refresh,
        fail_fast=fail_fast,
        report_only=report_only,
        root=root,
    )
    for stage_name in stages_for_group(stage):
        result = run_stage(stage_name, state, root)
        state.append(result)
        if fail_fast and result.status in {StageStatus.BLOCKED, StageStatus.FAILED}:
            break
    return write_orchestrator_reports(state, root)
