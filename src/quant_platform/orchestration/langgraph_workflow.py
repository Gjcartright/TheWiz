from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.orchestration.nodes import STAGE_GROUPS, run_stage, stages_for_group
from quant_platform.orchestration.reporting import write_orchestrator_reports
from quant_platform.orchestration.state import OrchestratorState, StageResult, StageStatus


class LangGraphWorkflowState(TypedDict, total=False):
    run_id: str
    stage_group: str
    stage_names: list[str]
    stage_index: int
    pair_id: str
    dry_run: bool
    force_refresh: bool
    fail_fast: bool
    report_only: bool
    root: str
    results: list[dict[str, object]]
    blocked: bool
    current_stage: str
    next_action: str


AGENT_LANES: list[dict[str, str]] = [
    {
        "agent": "intake_scout",
        "lane": "project_scrub",
        "stage_group": "discovery",
        "owns": "artifact index, current state, project spine, docs/report inventory",
        "primary_evidence": "reports/active/artifact_index.csv;reports/active/current_state.csv",
        "promotion_rule": "no promotion; discovery only",
    },
    {
        "agent": "wizard_capture_agent",
        "lane": "crypto_wizards_capture",
        "stage_group": "discovery",
        "owns": "scanner capture, pair-detail capture, exact Wizard setup identity, timeframe matrix",
        "primary_evidence": "reports/active/wizard_research_scanner_capture.csv;reports/active/wizard_research_pair_detail_capture.csv",
        "promotion_rule": "Wizard evidence can create hypotheses, never direct paper authorization",
    },
    {
        "agent": "verification_agent",
        "lane": "local_verification",
        "stage_group": "verification",
        "owns": "exact-mode local replay, after-cost tests, pair universe, research blockers",
        "primary_evidence": "reports/active/wizard_local_verification_batch.csv;data/processed/pair_universe.csv",
        "promotion_rule": "requires local after-cost and forward-walk support",
    },
    {
        "agent": "journal_agent",
        "lane": "paper_watch_journal",
        "stage_group": "all",
        "owns": "paper journal entries, entry/exit snapshots, live monitor refresh, outcome labels",
        "primary_evidence": "reports/paper_trading_journal.csv;reports/active/live_paper_trade_monitor.csv",
        "promotion_rule": "closed rows require verified exit snapshot before learning labels",
    },
    {
        "agent": "execution_guard_agent",
        "lane": "execution_compatibility",
        "stage_group": "all",
        "owns": "dYdX compatibility table, account-state blocker, Injective spot-first mirror lane",
        "primary_evidence": "reports/active/dydx_execution_market_compatibility.csv;reports/active/injective_spot_first_candidate_shortlist.csv",
        "promotion_rule": "paper submits only through confirmed markets; journal-only if route is unconfirmed",
    },
    {
        "agent": "ml_gate_agent",
        "lane": "machine_learning_gate",
        "stage_group": "model",
        "owns": "trade dataset, leakage audit, walk-forward model, gated backtest",
        "primary_evidence": "data/ml/trade_training_dataset.csv;reports/ml/model_gated_acceptance.csv",
        "promotion_rule": "model must improve out-of-sample edge or remain advisory",
    },
    {
        "agent": "base_rl_agent",
        "lane": "reinforcement_learning",
        "stage_group": "rl",
        "owns": "base RL route candidates, pair coverage, feedback dataset, paper handoff status",
        "primary_evidence": "reports/rl/base_rl_route_candidates.csv;reports/rl/base_rl_paper_handoff_status.csv",
        "promotion_rule": "paper_authorized only when strategy, execution, account, model, and pair support gates all pass",
    },
    {
        "agent": "review_board_agent",
        "lane": "gap_pre_red_review",
        "stage_group": "supreme_team",
        "owns": "gap analysis, pre-mortem, postmortem, red-team, supreme-team checkpoint",
        "primary_evidence": "reports/priority_gap_test.csv;reports/pre_mortem/latest_pre_mortem.md;reports/red_team/latest_red_team.md",
        "promotion_rule": "current-state blockers are formal review gaps",
    },
]


WORKFLOW_EDGES: list[dict[str, str]] = [
    {"from": "intake_scout", "to": "wizard_capture_agent", "condition": "project surfaces indexed"},
    {"from": "wizard_capture_agent", "to": "verification_agent", "condition": "candidate has exact setup/timeframe evidence"},
    {"from": "verification_agent", "to": "journal_agent", "condition": "research candidate can be watched"},
    {"from": "journal_agent", "to": "execution_guard_agent", "condition": "entry/exit snapshots and account state refreshed"},
    {"from": "execution_guard_agent", "to": "ml_gate_agent", "condition": "candidate is execution-compatible or journal-only research"},
    {"from": "ml_gate_agent", "to": "base_rl_agent", "condition": "model/feature truth refreshed"},
    {"from": "base_rl_agent", "to": "review_board_agent", "condition": "handoff report refreshed"},
    {"from": "review_board_agent", "to": "intake_scout", "condition": "next repair loop scheduled"},
]


def build_langgraph_agent_workflow():
    graph = StateGraph(LangGraphWorkflowState)
    graph.add_node("initialize", _initialize)
    graph.add_node("run_stage", _run_stage_node)
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "initialize")
    graph.add_conditional_edges(
        "initialize",
        _route_after_initialize,
        {"run_stage": "run_stage", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "run_stage",
        _route_after_stage,
        {"run_stage": "run_stage", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_langgraph_agent_workflow(
    *,
    stage: str = "all",
    pair_id: str = "",
    dry_run: bool = False,
    force_refresh: bool = False,
    fail_fast: bool = False,
    report_only: bool = False,
    root: Path = ROOT,
) -> CommandResult:
    graph = build_langgraph_agent_workflow()
    initial_state: LangGraphWorkflowState = {
        "stage_group": stage,
        "pair_id": pair_id,
        "dry_run": dry_run,
        "force_refresh": force_refresh,
        "fail_fast": fail_fast,
        "report_only": report_only,
        "root": str(root),
        "results": [],
    }
    final_state = graph.invoke(initial_state)
    state = _orchestrator_state_from_graph(final_state)
    reports = write_orchestrator_reports(state, root)
    workflow_paths = write_langgraph_workflow_manifest(root=root, final_state=final_state)
    return CommandResult(
        paths={**reports.paths, **workflow_paths},
        summary={
            **reports.summary,
            "graph": "langgraph_agent_workflow",
            "stage_group": stage,
            "agent_lanes": len(AGENT_LANES),
            "workflow_edges": len(WORKFLOW_EDGES),
            "blocked": int(state.blocked),
        },
    )


def write_langgraph_workflow_manifest(
    root: Path = ROOT,
    final_state: LangGraphWorkflowState | None = None,
) -> dict[str, Path]:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    lane_path = active / "langgraph_agent_lanes.csv"
    edge_path = active / "langgraph_agent_edges.csv"
    state_path = active / "langgraph_agent_workflow_state.json"
    md_path = active / "langgraph_agent_workflow.md"

    lanes = pd.DataFrame(AGENT_LANES)
    edges = pd.DataFrame(WORKFLOW_EDGES)
    lanes.to_csv(lane_path, index=False)
    edges.to_csv(edge_path, index=False)
    state_path.write_text(json.dumps(_json_safe_state(final_state or {}), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_workflow_markdown(lanes, edges, final_state or {}), encoding="utf-8")
    return {
        "langgraph_agent_lanes": lane_path,
        "langgraph_agent_edges": edge_path,
        "langgraph_agent_workflow_state": state_path,
        "langgraph_agent_workflow_md": md_path,
    }


def _initialize(state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    stage_group = str(state.get("stage_group", "all") or "all")
    return {
        **state,
        "run_id": str(state.get("run_id") or datetime.now(timezone.utc).strftime("lg_%Y%m%dT%H%M%SZ")),
        "stage_names": stages_for_group(stage_group),
        "stage_index": int(state.get("stage_index", 0) or 0),
        "results": list(state.get("results", [])),
        "blocked": False,
        "next_action": "run first stage",
    }


def _run_stage_node(state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    stage_names = list(state.get("stage_names", []))
    stage_index = int(state.get("stage_index", 0) or 0)
    if stage_index >= len(stage_names):
        return {**state, "next_action": "finalize"}

    root = Path(str(state.get("root") or ROOT))
    prior_results = [_stage_result_from_row(row) for row in state.get("results", [])]
    orchestrator_state = OrchestratorState(
        run_id=str(state.get("run_id", "")),
        stage_group=str(state.get("stage_group", "all")),
        pair_id=str(state.get("pair_id", "") or ""),
        dry_run=bool(state.get("dry_run", False)),
        force_refresh=bool(state.get("force_refresh", False)),
        fail_fast=bool(state.get("fail_fast", False)),
        report_only=bool(state.get("report_only", False)),
        root=root,
        results=prior_results,
    )
    stage_name = stage_names[stage_index]
    result = run_stage(stage_name, orchestrator_state, root)
    result_rows = [*state.get("results", []), _stage_result_to_row(result)]
    blocked = result.status in {StageStatus.BLOCKED, StageStatus.FAILED}
    return {
        **state,
        "current_stage": stage_name,
        "stage_index": stage_index + 1,
        "results": result_rows,
        "blocked": bool(state.get("blocked", False) or blocked),
        "next_action": result.next_step or ("stop at blocker" if blocked else "run next stage"),
    }


def _finalize(state: LangGraphWorkflowState) -> LangGraphWorkflowState:
    if not state.get("stage_names"):
        return {**state, "next_action": "no stages selected"}
    if bool(state.get("blocked", False)):
        return {**state, "next_action": "repair graph blockers and rerun"}
    return {**state, "next_action": "workflow completed"}


def _route_after_initialize(state: LangGraphWorkflowState) -> str:
    return "run_stage" if state.get("stage_names") else "finalize"


def _route_after_stage(state: LangGraphWorkflowState) -> str:
    if bool(state.get("fail_fast", False)) and bool(state.get("blocked", False)):
        return "finalize"
    if int(state.get("stage_index", 0) or 0) >= len(state.get("stage_names", [])):
        return "finalize"
    return "run_stage"


def _orchestrator_state_from_graph(state: LangGraphWorkflowState) -> OrchestratorState:
    return OrchestratorState(
        run_id=str(state.get("run_id", "")),
        stage_group=str(state.get("stage_group", "all")),
        pair_id=str(state.get("pair_id", "") or ""),
        dry_run=bool(state.get("dry_run", False)),
        force_refresh=bool(state.get("force_refresh", False)),
        fail_fast=bool(state.get("fail_fast", False)),
        report_only=bool(state.get("report_only", False)),
        root=Path(str(state.get("root") or ROOT)),
        results=[_stage_result_from_row(row) for row in state.get("results", [])],
        metadata={
            "graph": "langgraph_agent_workflow",
            "current_stage": state.get("current_stage", ""),
            "next_action": state.get("next_action", ""),
        },
    )


def _stage_result_to_row(result: StageResult) -> dict[str, object]:
    return {
        "stage": result.stage,
        "status": result.status.value,
        "blocker": result.blocker,
        "reason": result.reason,
        "evidence_path": result.evidence_path,
        "next_step": result.next_step,
        "rows": result.rows,
    }


def _stage_result_from_row(row: dict[str, object]) -> StageResult:
    status_value = str(row.get("status", StageStatus.SKIPPED.value))
    try:
        status = StageStatus(status_value)
    except ValueError:
        status = StageStatus.FAILED
    return StageResult(
        stage=str(row.get("stage", "")),
        status=status,
        blocker=str(row.get("blocker", "") or ""),
        reason=str(row.get("reason", "") or ""),
        evidence_path=str(row.get("evidence_path", "") or ""),
        next_step=str(row.get("next_step", "") or ""),
        rows=int(row.get("rows", 0) or 0),
    )


def _json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(state, default=str))


def _workflow_markdown(
    lanes: pd.DataFrame,
    edges: pd.DataFrame,
    state: LangGraphWorkflowState,
) -> str:
    lines = [
        "# LangGraph Agent Workflow",
        "",
        "This is the project-level agent workflow. It wraps the existing research, journal, execution, ML, RL, and review stages in a LangGraph state machine.",
        "",
        "## Graph",
        "",
        "```mermaid",
        "flowchart TD",
    ]
    for _, row in edges.iterrows():
        left = str(row["from"])
        right = str(row["to"])
        condition = str(row["condition"])
        lines.append(f'  {left}["{left}"] -->|"{condition}"| {right}["{right}"]')
    lines.extend(["```", "", "## Agent Lanes", ""])
    lines.append(lanes.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Current Run",
            "",
            f"- Run ID: `{state.get('run_id', '')}`",
            f"- Stage group: `{state.get('stage_group', '')}`",
            f"- Stage index: `{state.get('stage_index', 0)}`",
            f"- Blocked: `{state.get('blocked', False)}`",
            f"- Next action: `{state.get('next_action', '')}`",
            "",
            "## Guardrails",
            "",
            "- Wizard scanner rows are discovery evidence, not trade authorization.",
            "- Paper/live submission requires confirmed execution compatibility for every leg.",
            "- Browser-confirmed or journal-only trades stay in the live journal lane until exchange proof exists.",
            "- Closed journal outcomes are learning labels only after verified exit snapshots and prices exist.",
            "- Red-team and pre-mortem checks include current-state blockers, so top-level green cannot outrun exchange reality.",
        ]
    )
    return "\n".join(lines) + "\n"
