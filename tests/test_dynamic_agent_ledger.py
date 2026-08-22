from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.orchestration.contracts import (
    CandidateIdentity,
    EvidencePacket,
    TaskCard,
    TaskStatus,
)
from quant_platform.orchestration.dynamic_ledger import DynamicAgentLedger, DynamicLedgerError


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(
        pair="ETC-USD/SYRUP-USD",
        venue="crypto_wizards",
        strategy_family="Copula",
        timeframe="2h",
        lookback=320,
        formula_version="copula-v1",
    )


def _task() -> TaskCard:
    return TaskCard(
        task_id="task-copula-1",
        candidate=_candidate(),
        assigned_agent="copula_test_agent",
        action="run_local_replay",
        idempotency_key="copula-replay-1",
    )


def test_enqueue_is_idempotent_and_uses_atomic_snapshot(tmp_path):
    ledger = DynamicAgentLedger(tmp_path)
    first = ledger.enqueue(_task())
    second = ledger.enqueue(_task())

    assert first.task_id == second.task_id
    assert (tmp_path / "reports" / "orchestration" / "dynamic_agents" / "tasks.json").exists()
    assert (tmp_path / "reports" / "orchestration" / "dynamic_agents" / "task_events.jsonl").read_text().count("enqueued") == 1


def test_leased_task_requires_owner_and_recovers_after_expiry(tmp_path):
    ledger = DynamicAgentLedger(tmp_path)
    ledger.enqueue(_task())
    now = datetime.now(timezone.utc)
    leased = ledger.lease("task-copula-1", worker_id="worker-a", lease_seconds=10, now=now)

    assert leased.status == TaskStatus.LEASED
    with pytest.raises(DynamicLedgerError):
        ledger.complete("task-copula-1", worker_id="worker-b")

    recovered = ledger.recover_expired_leases(now=now + timedelta(seconds=11))
    assert recovered[0].status == TaskStatus.QUEUED
    final = ledger.lease("task-copula-1", worker_id="worker-b", lease_seconds=10, now=now + timedelta(seconds=12))
    assert ledger.complete(final.task_id, worker_id="worker-b").status == TaskStatus.COMPLETED


def test_evidence_records_are_append_only_and_conflicts_are_rejected(tmp_path):
    ledger = DynamicAgentLedger(tmp_path)
    now = datetime.now(timezone.utc)
    packet = EvidencePacket(
        packet_id="packet-1",
        candidate_id=_candidate().candidate_id,
        producing_agent="copula_research_agent",
        evidence_type="dashboard_snapshot",
        event_timestamp=now,
        source_timestamp=now,
        point_in_time_status="confirmed",
        formula_version="copula-v1",
        test_configuration_hash="snapshot-hash",
        evidence_content_hash="a" * 64,
        finding="conditional probability gap captured",
        evidence_paths=("reports/paper_trading_journal.csv",),
    )

    assert ledger.append_evidence(packet) is True
    assert ledger.append_evidence(packet) is False
    with pytest.raises(DynamicLedgerError):
        ledger.append_evidence(packet.model_copy(update={"finding": "changed after write"}))


def test_only_one_worker_can_lease_a_task_during_a_same_process_race(tmp_path):
    DynamicAgentLedger(tmp_path).enqueue(_task())

    def attempt(worker_id: str) -> str:
        try:
            DynamicAgentLedger(tmp_path).lease("task-copula-1", worker_id=worker_id, lease_seconds=30)
            return "leased"
        except DynamicLedgerError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(copy_context().run, attempt, worker_id)
            for worker_id in ("worker-a", "worker-b")
        ]
        outcomes = [future.result() for future in futures]

    assert sorted(outcomes) == ["leased", "rejected"]
    final = DynamicAgentLedger(tmp_path).get_task("task-copula-1")
    assert final is not None
    assert final.status == TaskStatus.LEASED
    assert final.attempt_count == 1


def test_expired_task_cannot_be_completed_after_its_lease(tmp_path):
    now = datetime.now(timezone.utc)
    task = _task().model_copy(update={"expires_at": now + timedelta(seconds=1)})
    ledger = DynamicAgentLedger(tmp_path)
    ledger.enqueue(task)
    ledger.lease(task.task_id, worker_id="worker", lease_seconds=30, now=now)
    with pytest.raises(DynamicLedgerError, match="expired before completion"):
        ledger.complete(task.task_id, worker_id="worker", now=now + timedelta(seconds=2))
    assert ledger.get_task(task.task_id).status == TaskStatus.EXPIRED


def test_expired_dependent_task_is_marked_expired_before_dependency_failure(tmp_path):
    now = datetime.now(timezone.utc)
    prerequisite = _task().model_copy(update={"task_id": "task-prerequisite", "idempotency_key": "prerequisite"})
    dependent = _task().model_copy(update={"task_id": "task-dependent", "idempotency_key": "dependent", "dependencies": (prerequisite.task_id,), "expires_at": now - timedelta(seconds=1)})
    ledger = DynamicAgentLedger(tmp_path)
    ledger.enqueue(prerequisite); ledger.enqueue(dependent)
    with pytest.raises(DynamicLedgerError, match="expired before it could be leased"):
        ledger.lease(dependent.task_id, worker_id="worker", lease_seconds=30, now=now)
    assert ledger.get_task(dependent.task_id).status == TaskStatus.EXPIRED


def test_legacy_evidence_without_integrity_hash_is_historical_only(tmp_path):
    ledger = DynamicAgentLedger(tmp_path)
    legacy = {"packet_id": "legacy-1", "candidate_id": _candidate().candidate_id, "producing_agent": "old_agent", "evidence_type": "dashboard_snapshot", "event_timestamp": datetime.now(timezone.utc).isoformat(), "source_timestamp": datetime.now(timezone.utc).isoformat(), "point_in_time_status": "confirmed", "formula_version": "copula-v1", "test_configuration_hash": "legacy", "finding": "old", "evidence_paths": ["reports/old.csv"]}
    (ledger.directory / "evidence.jsonl").write_text(json.dumps(legacy) + "\n", encoding="utf-8")
    assert ledger.read_evidence() == ()
