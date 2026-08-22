from __future__ import annotations

import json
import multiprocessing
import os
import queue
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from quant_platform.orchestration import corrective_scheduler_lock as scheduler_lock
from quant_platform.orchestration import effect_authority
from quant_platform.orchestration.corrective_external_effect_policy import (
    EXTERNAL_EFFECT_POLICY_PATH,
)
from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    launch_agent_runtime_environment,
    scheduler_contract,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GOVERNED_EVIDENCE_LOCK,
    acquire_scheduler_lock,
    current_publication_lease,
)
from quant_platform.orchestration.corrective_scheduler_supervisor import (
    supervise_scheduler_run,
)

NOW = datetime(2026, 8, 22, 12, 7, 42, tzinfo=UTC)
OLD = datetime(2020, 1, 1, tzinfo=UTC)
OWNER_PID = 41_001
FAKE_HOST = "phase00-fake-host"
CURRENT_BOOT_ID = "boot-current"
CURRENT_BOOT_SOURCE = "fake-boot-table"
CURRENT_PROCESS_START_ID = "test-runner-process-generation"
CURRENT_PROCESS_START_SOURCE = "fake-process-table"
PROCESS_TIMEOUT_SECONDS = 30.0
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _FakeProcess:
    alive: bool
    start_id: str
    start_source: str = CURRENT_PROCESS_START_SOURCE


@dataclass(frozen=True)
class _FakeProcessTable:
    processes: dict[int, _FakeProcess]

    def owner_alive(self, raw_pid: Any) -> bool:
        try:
            process = self.processes.get(int(raw_pid))
        except (TypeError, ValueError):
            return False
        return bool(process and process.alive)

    def process_start_identity(self, pid: int | None = None) -> tuple[str, str]:
        process = self.processes.get(os.getpid() if pid is None else int(pid))
        if process is None or not process.alive:
            return "", "unavailable"
        return process.start_id, process.start_source


def _prepare_scheduler_root(root: Path) -> None:
    source = root / "src" / "quant_platform"
    source.mkdir(parents=True)
    (source / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='phase00-contention-fixture'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    policy = root / EXTERNAL_EFFECT_POLICY_PATH
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_bytes((PROJECT_ROOT / EXTERNAL_EFFECT_POLICY_PATH).read_bytes())


def _scheduler_process(
    root_text: str,
    *,
    contract_key: str,
    publication_scope: str,
    label: str,
    entered_event: Any,
    release_event: Any,
    result_queue: Any,
) -> None:
    root = Path(root_text)
    callback_calls = 0
    try:
        contract = scheduler_contract(contract_key)
        os.environ.update(launch_agent_runtime_environment(root, contract=contract))

        def commit_callback() -> dict[str, Any]:
            nonlocal callback_calls
            callback_calls += 1
            lease = current_publication_lease()
            if lease is None:
                raise AssertionError("scheduler callback entered without a publication lease")
            entered_event.set()
            if not release_event.wait(PROCESS_TIMEOUT_SECONDS):
                raise TimeoutError(f"contention release timed out for {label}")
            commit_started_ns = time.monotonic_ns()
            marker = root / "reports" / "active" / "phase00_contention_commits" / f"{label}.json"
            atomic_write_text(
                marker,
                json.dumps(
                    {
                        "label": label,
                        "pid": os.getpid(),
                        "lease_generation": lease.generation,
                        "lease_process_start_id": lease.process_start_id,
                    },
                    sort_keys=True,
                )
                + "\n",
                publication_scope=publication_scope,
            )
            commit_completed_ns = time.monotonic_ns()
            return {
                "summary": {
                    "status": "PASS",
                    "blockers": [],
                    "external_calls": 0,
                    "external_credits_reserved": 0,
                    "external_credits_consumed": 0,
                    "external_credits_reconciled": 0,
                    "order_attempts": 0,
                    "order_submissions": 0,
                    "authority_advanced": False,
                    "promotion_authority": False,
                    "live_trading_authorized": False,
                    "commit_started_ns": commit_started_ns,
                    "commit_completed_ns": commit_completed_ns,
                    "lease_pid": lease.pid,
                    "lease_generation": lease.generation,
                    "lease_process_start_id": lease.process_start_id,
                    "lease_boot_id": lease.boot_id,
                },
                "paths": {"commit_marker": str(marker)},
            }

        result = supervise_scheduler_run(
            root=root,
            contract_key=contract_key,
            publication_scope=publication_scope,
            callback=commit_callback,
            now=NOW,
            require_launchd_provenance=True,
        )
        result_queue.put(
            {
                "kind": "result",
                "label": label,
                "pid": os.getpid(),
                "callback_calls": callback_calls,
                "run_id": result.run_id,
                "summary": result.result_summary,
                "terminal_receipt": result.terminal_receipt,
                "terminal_paths": {name: str(path) for name, path in result.terminal_paths.items()},
                "exit_code": result.exit_code,
            }
        )
    except Exception as exc:  # noqa: BLE001 - forward any child failure to pytest.
        result_queue.put(
            {
                "kind": "error",
                "label": label,
                "pid": os.getpid(),
                "callback_calls": callback_calls,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )


def _run_real_process_contention(
    root: Path,
    *,
    contender_contract_key: str,
    contender_publication_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    winner_entered = context.Event()
    winner_release = context.Event()
    contender_entered = context.Event()
    contender_release = context.Event()
    contender_release.set()
    result_queue = context.Queue()
    winner = context.Process(
        target=_scheduler_process,
        kwargs={
            "root_text": str(root),
            "contract_key": "hyperliquid_l2",
            "publication_scope": "public_l2",
            "label": "winner",
            "entered_event": winner_entered,
            "release_event": winner_release,
            "result_queue": result_queue,
        },
        name="phase00-winner",
    )
    contender = context.Process(
        target=_scheduler_process,
        kwargs={
            "root_text": str(root),
            "contract_key": contender_contract_key,
            "publication_scope": contender_publication_scope,
            "label": "contender",
            "entered_event": contender_entered,
            "release_event": contender_release,
            "result_queue": result_queue,
        },
        name="phase00-contender",
    )
    first: dict[str, Any] | None = None
    second: dict[str, Any] | None = None
    try:
        winner.start()
        assert winner_entered.wait(PROCESS_TIMEOUT_SECONDS), (
            "winner never entered its callback under the governed lock"
        )
        contender.start()
        first = result_queue.get(timeout=PROCESS_TIMEOUT_SECONDS)
        assert first["label"] == "contender", first.get("traceback", first)
        contender.join(PROCESS_TIMEOUT_SECONDS)
        assert contender.exitcode == 0
        assert not contender_entered.is_set(), (
            "contender callback entered while winner owned the governed lock"
        )
        winner_release.set()
        second = result_queue.get(timeout=PROCESS_TIMEOUT_SECONDS)
        assert second["label"] == "winner", second.get("traceback", second)
        winner.join(PROCESS_TIMEOUT_SECONDS)
        assert winner.exitcode == 0
    except queue.Empty:
        pytest.fail("scheduler subprocess failed to return a bounded contention result")
    finally:
        winner_release.set()
        contender_release.set()
        for process in (contender, winner):
            if process.pid is not None and process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(5)
        result_queue.close()
        result_queue.join_thread()
    assert first is not None and second is not None
    assert first["kind"] == "result", first.get("traceback", first)
    assert second["kind"] == "result", second.get("traceback", second)
    return second, first


def _assert_durable_terminal_receipt(result: dict[str, Any]) -> dict[str, Any]:
    receipt_path = Path(result["terminal_paths"]["terminal_receipt"])
    assert receipt_path.is_file()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == result["terminal_receipt"]
    return persisted


@pytest.mark.parametrize(
    (
        "contender_contract_key",
        "contender_publication_scope",
        "expected_retryable",
        "expected_blocker",
    ),
    [
        pytest.param(
            "hyperliquid_l2",
            "public_l2",
            False,
            "scheduler_intended_slot_already_claimed",
            id="same-producer",
        ),
        pytest.param(
            "wizard_proof",
            "wizard_external_research",
            False,
            "governed_evidence_write_lock_busy",
            id="cross-producer",
        ),
    ],
)
def test_real_scheduler_process_contention_has_one_commit_and_one_noncredit_terminal(
    tmp_path: Path,
    contender_contract_key: str,
    contender_publication_scope: str,
    expected_retryable: bool,
    expected_blocker: str,
) -> None:
    _prepare_scheduler_root(tmp_path)

    winner, contender = _run_real_process_contention(
        tmp_path,
        contender_contract_key=contender_contract_key,
        contender_publication_scope=contender_publication_scope,
    )

    assert winner["pid"] != contender["pid"]
    assert winner["callback_calls"] == 1
    assert contender["callback_calls"] == 0
    assert winner["exit_code"] == 0
    assert contender["exit_code"] == 2

    passing = _assert_durable_terminal_receipt(winner)
    deferred = _assert_durable_terminal_receipt(contender)
    assert passing["terminal_status"] == "PASS"
    assert passing["intended_slot_credit"] is True
    assert deferred["terminal_status"] == "DEFERRED"
    assert deferred["business_state"] == "DEFERRED"
    assert deferred["retryable"] is expected_retryable
    assert deferred["intended_slot_credit"] is False
    assert deferred["external_calls"] == 0
    assert deferred["external_credits_reserved"] == 0
    assert deferred["external_credits_consumed"] == 0
    assert deferred["external_credits_reconciled"] == 0
    assert deferred["blockers"] == [expected_blocker]

    markers = sorted(
        (tmp_path / "reports" / "active" / "phase00_contention_commits").glob("*.json")
    )
    assert [path.name for path in markers] == ["winner.json"]
    marker = json.loads(markers[0].read_text(encoding="utf-8"))
    assert marker["pid"] == winner["pid"]
    assert marker["lease_generation"] == winner["summary"]["lease_generation"]

    lock_state = json.loads((tmp_path / GOVERNED_EVIDENCE_LOCK).read_text(encoding="utf-8"))
    assert lock_state["pid"] == winner["pid"]
    assert lock_state["generation"] == winner["summary"]["lease_generation"]
    assert lock_state["process_start_id"] == winner["summary"]["lease_process_start_id"]
    assert lock_state["boot_id"] == winner["summary"]["lease_boot_id"]

    terminal_receipts = list(
        (tmp_path / "data" / "research" / "scheduler_terminal_receipts").rglob("*.json")
    )
    assert len(terminal_receipts) == 2


def test_claimed_cross_producer_contention_is_nonretryable_and_cannot_reexecute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_scheduler_root(tmp_path)
    _, contender = _run_real_process_contention(
        tmp_path,
        contender_contract_key="wizard_proof",
        contender_publication_scope="wizard_external_research",
    )
    assert contender["terminal_receipt"]["retryable"] is False

    contract = scheduler_contract("wizard_proof")
    for name, value in launch_agent_runtime_environment(tmp_path, contract=contract).items():
        monkeypatch.setenv(name, value)
    callback_calls: list[str] = []
    test_authority = effect_authority._CURRENT_PUBLICATION_AUTHORITY.set(None)
    try:
        retry = supervise_scheduler_run(
            root=tmp_path,
            contract_key="wizard_proof",
            publication_scope="wizard_external_research",
            callback=lambda: {
                "summary": {
                    "status": "PASS",
                    "blockers": [],
                    "external_calls": 0,
                    "order_attempts": 0,
                    "order_submissions": 0,
                },
                "paths": callback_calls.append("called") or {},
            },
            now=NOW,
            require_launchd_provenance=True,
        )
    finally:
        effect_authority._CURRENT_PUBLICATION_AUTHORITY.reset(test_authority)

    assert callback_calls == []
    assert retry.terminal_receipt["terminal_status"] == "DEFERRED"
    assert retry.terminal_receipt["intended_slot_credit"] is False
    assert retry.terminal_receipt["retryable"] is False
    assert retry.terminal_receipt["blockers"] == ["scheduler_intended_slot_already_claimed"]


def _install_fake_process_observations(
    monkeypatch: pytest.MonkeyPatch,
    table: _FakeProcessTable,
) -> None:
    monkeypatch.setattr(scheduler_lock, "_lock_owner_alive", table.owner_alive)
    monkeypatch.setattr(
        scheduler_lock,
        "_process_start_identity",
        table.process_start_identity,
    )
    monkeypatch.setattr(scheduler_lock, "_PROCESS_START_ID", CURRENT_PROCESS_START_ID)
    monkeypatch.setattr(
        scheduler_lock,
        "_PROCESS_START_SOURCE",
        CURRENT_PROCESS_START_SOURCE,
    )
    monkeypatch.setattr(scheduler_lock, "_BOOT_ID", CURRENT_BOOT_ID)
    monkeypatch.setattr(scheduler_lock, "_BOOT_ID_SOURCE", CURRENT_BOOT_SOURCE)
    monkeypatch.setattr(scheduler_lock.platform, "node", lambda: FAKE_HOST)


def _write_compatibility_lock(
    path: Path,
    *,
    started_at: str,
    process_start_id: str = "owner-generation-1",
    process_start_source: str = CURRENT_PROCESS_START_SOURCE,
    boot_id: str = CURRENT_BOOT_ID,
    boot_id_source: str = CURRENT_BOOT_SOURCE,
    modified_at: datetime = OLD,
) -> bytes:
    payload = {
        "schema_version": scheduler_lock.SCHEDULER_COMPATIBILITY_LOCK_SCHEMA_VERSION,
        "pid": OWNER_PID,
        "process_start_id": process_start_id,
        "process_start_source": process_start_source,
        "boot_id": boot_id,
        "boot_id_source": boot_id_source,
        "host": FAKE_HOST,
        "started_at_utc": started_at,
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))
    return path.read_bytes()


def _quarantined_payloads(path: Path) -> list[bytes]:
    quarantine = path.parent / ".scheduler_lock_quarantine"
    return [candidate.read_bytes() for candidate in quarantine.glob("*.quarantined")]


def test_compatibility_lock_preserves_stale_but_live_process_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _FakeProcessTable({OWNER_PID: _FakeProcess(alive=True, start_id="owner-generation-1")})
    _install_fake_process_observations(monkeypatch, table)
    lock = tmp_path / "live-owner.lock"
    original = _write_compatibility_lock(lock, started_at=OLD.isoformat())

    with pytest.raises(FileExistsError, match="active_scheduler_lock_present"):
        acquire_scheduler_lock(lock, now=NOW, timeout_seconds=60)

    assert lock.read_bytes() == original
    assert _quarantined_payloads(lock) == []


def test_current_process_lock_uses_stable_identity_when_reprobe_is_unstable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _FakeProcessTable(
        {
            os.getpid(): _FakeProcess(
                alive=True,
                start_id=CURRENT_PROCESS_START_ID,
            )
        }
    )
    _install_fake_process_observations(monkeypatch, table)
    monkeypatch.setattr(
        scheduler_lock,
        "_process_start_identity",
        lambda _pid=None: ("unstable-fallback", "monotonic_fallback"),
    )
    state = {
        "pid": os.getpid(),
        "process_start_id": CURRENT_PROCESS_START_ID,
        "process_start_source": CURRENT_PROCESS_START_SOURCE,
        "boot_id": CURRENT_BOOT_ID,
        "boot_id_source": CURRENT_BOOT_SOURCE,
        "host": FAKE_HOST,
    }

    assert scheduler_lock.scheduler_process_owner_matches(state) is True
    assert (
        scheduler_lock.scheduler_process_owner_matches(
            {**state, "process_start_id": "tampered-generation"}
        )
        is False
    )


@pytest.mark.parametrize(
    (
        "owner_process",
        "process_start_id",
        "process_start_source",
        "boot_id",
        "boot_id_source",
    ),
    [
        pytest.param(
            _FakeProcess(alive=False, start_id="owner-generation-1"),
            "owner-generation-1",
            CURRENT_PROCESS_START_SOURCE,
            CURRENT_BOOT_ID,
            CURRENT_BOOT_SOURCE,
            id="dead-owner",
        ),
        pytest.param(
            _FakeProcess(alive=True, start_id="owner-generation-2"),
            "owner-generation-1",
            CURRENT_PROCESS_START_SOURCE,
            CURRENT_BOOT_ID,
            CURRENT_BOOT_SOURCE,
            id="pid-reuse-new-process-generation",
        ),
        pytest.param(
            _FakeProcess(alive=True, start_id="owner-generation-1"),
            "owner-generation-1",
            CURRENT_PROCESS_START_SOURCE,
            "boot-before-reboot",
            CURRENT_BOOT_SOURCE,
            id="boot-change",
        ),
        pytest.param(
            _FakeProcess(alive=True, start_id="owner-generation-1"),
            "owner-generation-1",
            CURRENT_PROCESS_START_SOURCE,
            CURRENT_BOOT_ID,
            "boot-source-before-reboot",
            id="reboot-identity-source-change",
        ),
    ],
)
def test_stale_nonowner_generation_is_quarantined_and_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_process: _FakeProcess,
    process_start_id: str,
    process_start_source: str,
    boot_id: str,
    boot_id_source: str,
) -> None:
    table = _FakeProcessTable({OWNER_PID: owner_process})
    _install_fake_process_observations(monkeypatch, table)
    lock = tmp_path / "replaceable-owner.lock"
    original = _write_compatibility_lock(
        lock,
        started_at=OLD.isoformat(),
        process_start_id=process_start_id,
        process_start_source=process_start_source,
        boot_id=boot_id,
        boot_id_source=boot_id_source,
    )

    acquire_scheduler_lock(lock, now=NOW, timeout_seconds=60)

    replacement = json.loads(lock.read_text(encoding="utf-8"))
    assert replacement["pid"] == os.getpid()
    assert replacement["process_start_id"] == CURRENT_PROCESS_START_ID
    assert replacement["process_start_source"] == CURRENT_PROCESS_START_SOURCE
    assert replacement["boot_id"] == CURRENT_BOOT_ID
    assert replacement["boot_id_source"] == CURRENT_BOOT_SOURCE
    assert replacement["host"] == FAKE_HOST
    assert replacement["started_at_utc"] == NOW.isoformat()
    assert _quarantined_payloads(lock) == [original]


def test_recent_previous_boot_identity_remains_fail_closed_until_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _FakeProcessTable({OWNER_PID: _FakeProcess(alive=True, start_id="owner-generation-1")})
    _install_fake_process_observations(monkeypatch, table)
    lock = tmp_path / "recent-previous-boot.lock"
    original = _write_compatibility_lock(
        lock,
        started_at=NOW.isoformat(),
        boot_id="boot-before-reboot",
        modified_at=NOW,
    )

    with pytest.raises(FileExistsError, match="active_scheduler_lock_present"):
        acquire_scheduler_lock(lock, now=NOW, timeout_seconds=60)

    assert lock.read_bytes() == original
    assert _quarantined_payloads(lock) == []


def test_malformed_age_cannot_evict_a_proven_live_process_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _FakeProcessTable({OWNER_PID: _FakeProcess(alive=True, start_id="owner-generation-1")})
    _install_fake_process_observations(monkeypatch, table)
    lock = tmp_path / "live-malformed-age.lock"
    original = _write_compatibility_lock(lock, started_at="not-a-timestamp")

    with pytest.raises(FileExistsError, match="active_scheduler_lock_present"):
        acquire_scheduler_lock(lock, now=NOW, timeout_seconds=60)

    assert lock.read_bytes() == original
    assert _quarantined_payloads(lock) == []


def test_old_malformed_age_with_dead_owner_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _FakeProcessTable({OWNER_PID: _FakeProcess(alive=False, start_id="owner-generation-1")})
    _install_fake_process_observations(monkeypatch, table)
    lock = tmp_path / "dead-malformed-age.lock"
    original = _write_compatibility_lock(lock, started_at="not-a-timestamp")

    acquire_scheduler_lock(lock, now=NOW, timeout_seconds=60)

    replacement = json.loads(lock.read_text(encoding="utf-8"))
    assert replacement["process_start_id"] == CURRENT_PROCESS_START_ID
    assert replacement["boot_id"] == CURRENT_BOOT_ID
    assert _quarantined_payloads(lock) == [original]
