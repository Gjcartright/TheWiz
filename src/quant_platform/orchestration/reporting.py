from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.orchestration.events import append_stage_event
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus


def write_orchestrator_reports(state: OrchestratorState, root: Path = ROOT) -> CommandResult:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    status_path = active / "orchestrator_run_status.csv"
    md_path = active / "orchestrator_run_status.md"
    events_path = active / "orchestrator_events.jsonl"

    frame = pd.DataFrame([result.to_row(state.run_id, state.pair_id) for result in state.results])
    frame.to_csv(status_path, index=False)
    md_path.write_text(_status_markdown(frame, state), encoding="utf-8")
    for result in state.results:
        append_stage_event(events_path, run_id=state.run_id, pair_id=state.pair_id, result=result)
    return CommandResult(
        paths={"orchestrator_status": status_path, "orchestrator_status_md": md_path, "orchestrator_events": events_path},
        summary={
            "run_id": state.run_id,
            "stages": len(state.results),
            "blocked": int(sum(result.blocker != "" for result in state.results)),
            "failed": int(sum(result.status.value == "failed" for result in state.results)),
        },
    )


def write_project_spine_audit(root: Path = ROOT) -> Path:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    path = active / "project_spine_audit.md"
    checks = _spine_checks(root)
    rows = ["# Project Spine Audit", "", "| Area | Status | Evidence |", "| --- | --- | --- |"]
    for area, evidence in checks.items():
        status = "present" if evidence.exists() else "missing"
        rows.append(f"| {area} | {status} | `{evidence.relative_to(root) if evidence.exists() else evidence}` |")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def stage_result_from_command(stage: str, result: CommandResult, reason: str = "completed") -> StageResult:
    evidence = ";".join(str(path) for path in result.paths.values())
    rows = int(result.summary.get("rows", result.summary.get("dashboard_files", result.summary.get("artifacts", 0))) or 0)
    return StageResult(stage=stage, status=StageStatus.PASSED, reason=reason, evidence_path=evidence, rows=rows)


def _spine_checks(root: Path) -> dict[str, Path]:
    return {
        "active_pipeline": root / "src" / "quant_platform" / "active_pipeline.py",
        "wizard_evidence": root / "src" / "quant_platform" / "wizard_evidence.py",
        "dydx_candles": root / "src" / "quant_platform" / "dydx_candles.py",
        "backtest": root / "src" / "quant_platform" / "backtest.py",
        "model_gate": root / "src" / "quant_platform" / "ml_filter.py",
        "quantization": root / "src" / "quant_platform" / "research_quantization.py",
        "pair_universe": root / "data" / "processed" / "pair_universe.csv",
        "trade_dataset": root / "data" / "ml" / "trade_training_dataset.csv",
        "dashboard": root / "reports" / "dashboard" / "command_center.md",
    }


def _status_markdown(frame: pd.DataFrame, state: OrchestratorState) -> str:
    lines = [
        "# Orchestrator Run Status",
        "",
        f"- Run ID: `{state.run_id}`",
        f"- Stage group: `{state.stage_group}`",
        f"- Pair ID: `{state.pair_id}`",
        f"- Dry run: `{state.dry_run}`",
        f"- Report only: `{state.report_only}`",
        "",
    ]
    if frame.empty:
        lines.append("No stages ran.")
    else:
        lines.append(frame[["stage", "status", "blocker", "reason", "next_step"]].to_markdown(index=False))
    return "\n".join(lines) + "\n"
