"""Local durable records for the dynamic-agent control plane.

This is intentionally a file-backed, single-host ledger. It gives the router
idempotency and lease safety before the project adds a database or distributed
workers.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import RLock
from typing import Iterator, TypeVar

from pydantic import ValidationError

from quant_platform.orchestration.contracts import (
    ComparisonEvent,
    DecisionRecord,
    EvidencePacket,
    OutcomeRecord,
    TaskCard,
    TaskStatus,
    VetoRecord,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_append_text,
    promote_staged_file,
)

ROOT = Path(__file__).resolve().parents[3]


T = TypeVar("T", EvidencePacket, DecisionRecord, VetoRecord, OutcomeRecord, ComparisonEvent)


# flock protects independent processes. A keyed in-process lock closes the
# separate race window for multiple worker threads sharing one interpreter.
_THREAD_LOCKS: dict[str, RLock] = {}
_THREAD_LOCKS_GUARD = RLock()


class DynamicLedgerError(RuntimeError):
    """Raised when a dynamic-agent record or task transition is invalid."""


class DynamicAgentLedger:
    """Append-only evidence records with an atomically maintained task snapshot."""

    def __init__(self, root: Path = ROOT) -> None:
        self.directory = root / "reports" / "orchestration" / "dynamic_agents"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.directory / "tasks.json"
        self.task_events_path = self.directory / "task_events.jsonl"
        self.lock_path = self.directory / ".ledger.lock"

    def enqueue(self, task: TaskCard) -> TaskCard:
        """Store a new task or return the existing task for the same idempotency key."""

        with self._locked():
            tasks = self._read_tasks()
            existing = next(
                (row for row in tasks.values() if row["task"].get("idempotency_key") == task.idempotency_key),
                None,
            )
            if existing is not None:
                existing_task = TaskCard.model_validate(existing["task"])
                if existing_task.candidate.candidate_id != task.candidate.candidate_id or existing_task.action != task.action:
                    raise DynamicLedgerError("idempotency key is already assigned to a different task")
                return existing_task
            if task.task_id in tasks:
                raise DynamicLedgerError("task_id already exists with a different idempotency key")
            tasks[task.task_id] = {"task": task.model_dump(mode="json")}
            self._write_tasks(tasks)
            self._append_jsonl(self.task_events_path, self._task_event(task, "enqueued"))
            return task

    def get_task(self, task_id: str) -> TaskCard | None:
        with self._locked():
            row = self._read_tasks().get(task_id)
            return TaskCard.model_validate(row["task"]) if row else None

    def lease(self, task_id: str, *, worker_id: str, lease_seconds: int, now: datetime | None = None) -> TaskCard:
        if lease_seconds <= 0:
            raise DynamicLedgerError("lease_seconds must be positive")
        now = now or datetime.now(timezone.utc)
        with self._locked():
            tasks = self._read_tasks()
            task = self._required_task(tasks, task_id)
            task = self._recover_expired_lease(task, now)
            if task.status != TaskStatus.QUEUED:
                raise DynamicLedgerError(f"task is not available for lease: {task.status}")
            if task.expires_at and task.expires_at <= now:
                task = task.model_copy(update={"status": TaskStatus.EXPIRED})
                tasks[task_id] = {"task": task.model_dump(mode="json")}
                self._write_tasks(tasks)
                self._append_jsonl(self.task_events_path, self._task_event(task, "expired_before_lease"))
                raise DynamicLedgerError("task expired before it could be leased")
            incomplete_dependencies = [
                dependency for dependency in task.dependencies
                if dependency not in tasks or TaskCard.model_validate(tasks[dependency]["task"]).status != TaskStatus.COMPLETED
            ]
            if incomplete_dependencies:
                raise DynamicLedgerError("task dependencies are not completed")
            if task.attempt_count >= task.max_attempts:
                task = task.model_copy(update={"status": TaskStatus.FAILED})
                tasks[task_id] = {"task": task.model_dump(mode="json")}
                self._write_tasks(tasks)
                self._append_jsonl(self.task_events_path, self._task_event(task, "attempt_limit_reached"))
                raise DynamicLedgerError("task retry limit reached")
            leased = task.model_copy(
                update={
                    "status": TaskStatus.LEASED,
                    "lease_owner": worker_id,
                    "lease_expires_at": now + timedelta(seconds=lease_seconds),
                    "attempt_count": task.attempt_count + 1,
                }
            )
            tasks[task_id] = {"task": leased.model_dump(mode="json")}
            self._write_tasks(tasks)
            self._append_jsonl(self.task_events_path, self._task_event(leased, "leased"))
            return leased

    def complete(self, task_id: str, *, worker_id: str, now: datetime | None = None) -> TaskCard:
        return self._transition_leased_task(task_id, worker_id=worker_id, status=TaskStatus.COMPLETED, event="completed", now=now)

    def block(self, task_id: str, *, worker_id: str, now: datetime | None = None) -> TaskCard:
        return self._transition_leased_task(task_id, worker_id=worker_id, status=TaskStatus.BLOCKED, event="blocked", now=now)

    def recover_expired_leases(self, now: datetime | None = None) -> list[TaskCard]:
        now = now or datetime.now(timezone.utc)
        recovered: list[TaskCard] = []
        with self._locked():
            tasks = self._read_tasks()
            for task_id, row in list(tasks.items()):
                task = TaskCard.model_validate(row["task"])
                updated = self._recover_expired_lease(task, now)
                if updated != task:
                    tasks[task_id] = {"task": updated.model_dump(mode="json")}
                    recovered.append(updated)
                    self._append_jsonl(self.task_events_path, self._task_event(updated, "lease_recovered"))
            if recovered:
                self._write_tasks(tasks)
        return recovered

    def append_evidence(self, packet: EvidencePacket) -> bool:
        return self._append_immutable("evidence.jsonl", "packet_id", packet.packet_id, packet)

    def read_evidence(self) -> tuple[EvidencePacket, ...]:
        """Return only current-schema packets; legacy packets remain historical-only."""

        with self._locked():
            packets: list[EvidencePacket] = []
            for row in self._read_jsonl(self.directory / "evidence.jsonl"):
                try:
                    packets.append(EvidencePacket.model_validate(row))
                except ValidationError:
                    # Old packets lack mandatory integrity lineage and can never
                    # satisfy a hardened rollout gate.
                    continue
            return tuple(packets)

    def append_decision(self, record: DecisionRecord) -> bool:
        return self._append_immutable("decisions.jsonl", "decision_id", record.decision_id, record)

    def append_veto(self, record: VetoRecord) -> bool:
        return self._append_immutable("vetoes.jsonl", "veto_id", record.veto_id, record)

    def append_outcome(self, record: OutcomeRecord) -> bool:
        return self._append_immutable("outcomes.jsonl", "outcome_id", record.outcome_id, record)

    def append_comparison_event(self, record: ComparisonEvent) -> bool:
        return self._append_immutable("copula_comparison_events.jsonl", "event_id", record.event_id, record)

    def _transition_leased_task(
        self,
        task_id: str,
        *,
        worker_id: str,
        status: TaskStatus,
        event: str,
        now: datetime | None,
    ) -> TaskCard:
        now = now or datetime.now(timezone.utc)
        with self._locked():
            tasks = self._read_tasks()
            task = self._required_task(tasks, task_id)
            if task.status != TaskStatus.LEASED or task.lease_owner != worker_id:
                raise DynamicLedgerError("only the active lease owner can change a leased task")
            if task.expires_at and task.expires_at <= now:
                expired = task.model_copy(update={"status": TaskStatus.EXPIRED, "lease_owner": "", "lease_expires_at": None})
                tasks[task_id] = {"task": expired.model_dump(mode="json")}
                self._write_tasks(tasks)
                self._append_jsonl(self.task_events_path, self._task_event(expired, "expired_before_completion"))
                raise DynamicLedgerError("task expired before completion")
            updated = task.model_copy(update={"status": status, "lease_owner": "", "lease_expires_at": None})
            tasks[task_id] = {"task": updated.model_dump(mode="json")}
            self._write_tasks(tasks)
            self._append_jsonl(self.task_events_path, self._task_event(updated, event))
            return updated

    def _recover_expired_lease(self, task: TaskCard, now: datetime) -> TaskCard:
        if task.status != TaskStatus.LEASED or task.lease_expires_at is None or task.lease_expires_at > now:
            return task
        status = TaskStatus.FAILED if task.attempt_count >= task.max_attempts else TaskStatus.QUEUED
        return task.model_copy(update={"status": status, "lease_owner": "", "lease_expires_at": None})

    def _append_immutable(self, filename: str, key_name: str, key: str, record: T) -> bool:
        path = self.directory / filename
        payload = record.model_dump(mode="json")
        with self._locked():
            for existing in self._read_jsonl(path):
                if existing.get(key_name) != key:
                    continue
                if existing != payload:
                    raise DynamicLedgerError(f"immutable {key_name} already exists with different content")
                return False
            self._append_jsonl(path, payload)
            return True

    def _required_task(self, tasks: dict[str, dict[str, object]], task_id: str) -> TaskCard:
        row = tasks.get(task_id)
        if not row:
            raise DynamicLedgerError("task does not exist")
        return TaskCard.model_validate(row["task"])

    def _read_tasks(self) -> dict[str, dict[str, object]]:
        if not self.tasks_path.exists():
            return {}
        try:
            payload = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DynamicLedgerError("task snapshot is not valid JSON") from exc
        return payload if isinstance(payload, dict) else {}

    def _write_tasks(self, tasks: dict[str, dict[str, object]]) -> None:
        self._atomic_write(self.tasks_path, json.dumps(tasks, indent=2, sort_keys=True) + "\n")

    def _append_jsonl(self, path: Path, payload: dict[str, object]) -> None:
        atomic_append_text(path, json.dumps(payload, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> Iterator[dict[str, object]]:
        if not path.exists():
            return iter(())
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return iter(rows)

    def _task_event(self, task: TaskCard, event: str) -> dict[str, object]:
        return {
            "event": event,
            "event_timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task.task_id,
            "candidate_id": task.candidate.candidate_id,
            "status": task.status.value,
            "lease_owner": task.lease_owner,
            "attempt_count": task.attempt_count,
        }

    def _atomic_write(self, path: Path, text: str) -> None:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(text)
            temporary_path = Path(handle.name)
        promote_staged_file(temporary_path, path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        import fcntl

        lock_key = str(self.directory.resolve())
        with _THREAD_LOCKS_GUARD:
            thread_lock = _THREAD_LOCKS.setdefault(lock_key, RLock())
        with thread_lock:
            with self.lock_path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
