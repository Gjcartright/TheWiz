from __future__ import annotations

from datetime import datetime, timedelta, timezone

from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger
from quant_platform.orchestration.dynamic_router import RoutingRequest, route_shadow_request


def _request(*, age_seconds: int = 0, task_budget: int = 5) -> RoutingRequest:
    now = datetime.now(timezone.utc)
    return RoutingRequest(
        candidate=CandidateIdentity(
            pair="ETC-USD/SYRUP-USD",
            venue="crypto_wizards",
            strategy_family="Copula",
            timeframe="2h",
            lookback=320,
            formula_version="copula-v1",
        ),
        event_type="copula_dislocation",
        source_timestamp=now - timedelta(seconds=age_seconds),
        requested_at=now,
        evidence_paths=("reports/paper_trading_journal.csv",),
        task_budget=task_budget,
    )


def test_fresh_copula_event_routes_only_research_tasks(tmp_path):
    plan = route_shadow_request(_request(), ledger=DynamicAgentLedger(tmp_path))

    assert plan.route_status == "shadow_routed"
    assert len(plan.task_ids) == 5
    assert "research_only" in plan.reason


def test_stale_copula_event_routes_only_data_health(tmp_path):
    plan = route_shadow_request(_request(age_seconds=2 * 60 * 60 + 31 * 60), ledger=DynamicAgentLedger(tmp_path))

    assert plan.route_status == "shadow_routed"
    assert len(plan.task_ids) == 1
    assert "stale" in plan.reason


def test_future_copula_event_routes_only_data_health(tmp_path):
    request = _request(age_seconds=-1)
    plan = route_shadow_request(request, ledger=DynamicAgentLedger(tmp_path))
    assert plan.route_status == "shadow_routed"
    assert len(plan.task_ids) == 1
    assert "future" in plan.reason


def test_router_blocks_when_required_review_budget_is_not_available(tmp_path):
    plan = route_shadow_request(_request(task_budget=4), ledger=DynamicAgentLedger(tmp_path))

    assert plan.route_status == "blocked"
    assert plan.task_ids == ()


def test_router_assigns_expiry_and_dependency_order(tmp_path):
    ledger = DynamicAgentLedger(tmp_path)
    request = _request()
    plan = route_shadow_request(request, ledger=ledger)
    tasks = {ledger.get_task(task_id).action: ledger.get_task(task_id) for task_id in plan.task_ids}
    local_replay = tasks["run_copula_local_replay"]
    assert all(task.expires_at is not None for task in tasks.values())
    assert len(local_replay.dependencies) == 2
    try:
        ledger.lease(local_replay.task_id, worker_id="worker", lease_seconds=30, now=request.requested_at)
    except Exception as exc:
        assert "dependencies" in str(exc)
    else:
        raise AssertionError("dependent task was leased before prerequisites completed")

    for action in ("capture_copula_evidence", "validate_copula_reference"):
        task = tasks[action]
        ledger.lease(task.task_id, worker_id=action, lease_seconds=30, now=request.requested_at)
        ledger.complete(task.task_id, worker_id=action, now=request.requested_at)
    assert ledger.lease(local_replay.task_id, worker_id="worker", lease_seconds=30, now=request.requested_at).action == "run_copula_local_replay"
