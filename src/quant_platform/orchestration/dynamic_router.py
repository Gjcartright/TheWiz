"""Bounded, shadow-only routing for dynamic-agent research tasks."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_platform.orchestration.contracts import AgentCapability, CandidateIdentity, TaskCard
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger


COPULA_FRESHNESS_SECONDS = 2 * 60 * 60 + 30 * 60
DATA_HEALTH_TASK_SECONDS = 15 * 60

CAPABILITIES: dict[str, AgentCapability] = {
    "copula_research_agent": AgentCapability(agent="copula_research_agent", allowed_actions=("capture_copula_evidence",)),
    "copula_reference_agent": AgentCapability(agent="copula_reference_agent", allowed_actions=("validate_copula_reference",)),
    "copula_test_agent": AgentCapability(agent="copula_test_agent", allowed_actions=("run_copula_local_replay",)),
    "venue_evidence_agent": AgentCapability(agent="venue_evidence_agent", allowed_actions=("verify_venue_route",)),
    "cost_risk_agent": AgentCapability(agent="cost_risk_agent", allowed_actions=("estimate_after_cost",)),
    "data_health_agent": AgentCapability(agent="data_health_agent", allowed_actions=("refresh_source_snapshot",)),
}


class RoutingRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate: CandidateIdentity
    event_type: Literal["copula_dislocation"]
    source_timestamp: datetime
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    task_budget: int = Field(default=5, ge=1, le=20)
    requested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_timestamp", "requested_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("routing timestamps must be timezone-aware")
        return value


class RoutePlan(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    route_status: Literal["shadow_routed", "blocked"]
    reason: str
    task_ids: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...]


def route_shadow_request(request: RoutingRequest, *, ledger: DynamicAgentLedger) -> RoutePlan:
    """Route a fresh Copula event to bounded research work, never execution."""

    age_seconds = (request.requested_at - request.source_timestamp).total_seconds()
    if age_seconds < 0:
        task = _create_task(
            request,
            agent="data_health_agent",
            action="refresh_source_snapshot",
            priority="critical",
            expires_at=request.requested_at + timedelta(seconds=DATA_HEALTH_TASK_SECONDS),
        )
        ledger.enqueue(task)
        plan = RoutePlan(
            candidate_id=request.candidate.candidate_id,
            route_status="shadow_routed",
            reason="future_copula_source_only_data_health_task_created",
            task_ids=(task.task_id,),
            evidence_paths=request.evidence_paths,
        )
        _write_route_plan(ledger.directory, plan)
        return plan
    if age_seconds > COPULA_FRESHNESS_SECONDS:
        task = _create_task(
            request,
            agent="data_health_agent",
            action="refresh_source_snapshot",
            priority="critical",
            expires_at=request.requested_at + timedelta(seconds=DATA_HEALTH_TASK_SECONDS),
        )
        ledger.enqueue(task)
        plan = RoutePlan(
            candidate_id=request.candidate.candidate_id,
            route_status="shadow_routed",
            reason="stale_copula_source_only_data_health_task_created",
            task_ids=(task.task_id,),
            evidence_paths=request.evidence_paths,
        )
        _write_route_plan(ledger.directory, plan)
        return plan

    route = (
        ("copula_research_agent", "capture_copula_evidence", "high", ()),
        ("copula_reference_agent", "validate_copula_reference", "high", ()),
        ("copula_test_agent", "run_copula_local_replay", "high", ("capture_copula_evidence", "validate_copula_reference")),
        ("venue_evidence_agent", "verify_venue_route", "medium", ("capture_copula_evidence",)),
        ("cost_risk_agent", "estimate_after_cost", "medium", ("run_copula_local_replay",)),
    )
    if request.task_budget < len(route):
        return RoutePlan(
            candidate_id=request.candidate.candidate_id,
            route_status="blocked",
            reason="insufficient_task_budget_for_required_copula_review",
            evidence_paths=request.evidence_paths,
        )

    expires_at = request.source_timestamp + timedelta(seconds=COPULA_FRESHNESS_SECONDS)
    tasks_by_action: dict[str, TaskCard] = {}
    for agent, action, priority, dependency_actions in route:
        task = _create_task(
            request,
            agent=agent,
            action=action,
            priority=priority,
            dependencies=tuple(tasks_by_action[dependency].task_id for dependency in dependency_actions),
            expires_at=expires_at,
        )
        tasks_by_action[action] = task
    tasks = tuple(tasks_by_action[action] for _, action, _, _ in route)
    for task in tasks:
        ledger.enqueue(task)
    plan = RoutePlan(
        candidate_id=request.candidate.candidate_id,
        route_status="shadow_routed",
        reason="fresh_copula_event_routed_for_research_only",
        task_ids=tuple(task.task_id for task in tasks),
        evidence_paths=request.evidence_paths,
    )
    _write_route_plan(ledger.directory, plan)
    return plan


def _create_task(
    request: RoutingRequest,
    *,
    agent: str,
    action: str,
    priority: str,
    expires_at: datetime,
    dependencies: tuple[str, ...] = (),
) -> TaskCard:
    capability = CAPABILITIES[agent]
    if action not in capability.allowed_actions:
        raise ValueError(f"agent {agent} cannot perform {action}")
    raw = "|".join(
        [request.candidate.candidate_id, agent, action, request.source_timestamp.isoformat(), *request.evidence_paths]
    )
    token = sha256(raw.encode("utf-8")).hexdigest()[:16]
    return TaskCard(
        task_id=f"task_{token}",
        candidate=request.candidate,
        assigned_agent=agent,
        action=action,
        priority=priority,  # type: ignore[arg-type]
        idempotency_key=f"idem_{token}",
        evidence_paths=request.evidence_paths,
        dependencies=dependencies,
        expires_at=expires_at,
    )


def _write_route_plan(directory: Path, plan: RoutePlan) -> None:
    path = directory / "route_plans.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(plan.model_dump(mode="json"), sort_keys=True) + "\n")
