from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    launch_agent_runtime_environment,
    scheduler_contract,
    scheduler_run_identity,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    GovernedEvidenceMaintenanceActive,
    PublicationLeaseError,
    scheduler_terminal_write_lock,
)
from quant_platform.orchestration.corrective_scheduler_terminal import (
    TERMINAL_STATE_RULES,
    build_scheduler_run_intent,
    build_scheduler_terminal_receipt,
    claim_scheduler_slot,
    new_scheduler_run_id,
    publish_scheduler_run_intent,
    publish_scheduler_terminal_receipt,
    scheduler_terminal_exit_code,
    terminal_receipt_exists,
)


def _identity(root: Path) -> dict:
    contract = scheduler_contract("hyperliquid_l2")
    environment = launch_agent_runtime_environment(root, contract=contract)
    return scheduler_run_identity(
        root,
        contract=contract,
        environment=environment,
        require_launchd=True,
    )


def _prepare_root(root: Path) -> None:
    (root / "src" / "quant_platform").mkdir(parents=True)
    (root / "src" / "quant_platform" / "fixture.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_terminal_receipt_is_immutable_and_only_pass_returns_zero(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)
    identity = _identity(tmp_path)
    intent = build_scheduler_run_intent(
        run_id=run_id,
        intended_slot="2026-08-21T12:00:00Z/PT5M",
        runtime_identity=identity,
        started_at=started,
    )
    publish_scheduler_run_intent(tmp_path, intent)
    receipt = build_scheduler_terminal_receipt(
        run_id=run_id,
        intended_slot="2026-08-21T12:00:00Z/PT5M",
        runtime_identity=identity,
        started_at=started,
        completed_at=started + timedelta(seconds=3),
        terminal_status="PASS",
        process_health="HEALTHY",
        business_state="PASS",
        retryable=False,
        intended_slot_credit=True,
        result_summary={"status": "PASS"},
    )
    claim_scheduler_slot(
        tmp_path,
        run_id=run_id,
        intended_slot="2026-08-21T12:00:00Z/PT5M",
        runtime_identity=identity,
        claimed_at=started,
    )

    paths = publish_scheduler_terminal_receipt(tmp_path, receipt)
    publish_scheduler_terminal_receipt(tmp_path, receipt)

    assert paths["terminal_receipt"].is_file()
    assert paths["terminal_pointer"].is_file()
    assert terminal_receipt_exists(tmp_path, scheduler_key="hyperliquid_l2", run_id=run_id)
    assert scheduler_terminal_exit_code(receipt) == 0


def test_non_pass_requires_blocker_and_never_receives_slot_credit(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    identity = _identity(tmp_path)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)

    with pytest.raises(ValueError, match="requires at least one blocker"):
        build_scheduler_terminal_receipt(
            run_id=run_id,
            intended_slot="slot",
            runtime_identity=identity,
            started_at=started,
            completed_at=started,
            terminal_status="BLOCKED",
            process_health="HEALTHY",
            business_state="BLOCKED",
            retryable=True,
            intended_slot_credit=False,
        )

    receipt = build_scheduler_terminal_receipt(
        run_id=run_id,
        intended_slot="slot",
        runtime_identity=identity,
        started_at=started,
        completed_at=started,
        terminal_status="DEFERRED",
        process_health="HEALTHY",
        business_state="DEFERRED",
        retryable=True,
        intended_slot_credit=False,
        blockers=["governed_lock_busy"],
    )
    assert scheduler_terminal_exit_code(receipt) == 2


@pytest.mark.parametrize("terminal_status", sorted(TERMINAL_STATE_RULES))
def test_every_terminal_status_has_one_enforced_state_rule(
    tmp_path: Path,
    terminal_status: str,
) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    rule = TERMINAL_STATE_RULES[terminal_status]
    retryable = False not in rule.retryable_values

    receipt = build_scheduler_terminal_receipt(
        run_id=new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started),
        intended_slot="slot",
        runtime_identity=_identity(tmp_path),
        started_at=started,
        completed_at=started,
        terminal_status=rule.terminal_status,
        process_health=rule.process_health,
        business_state=rule.business_state,
        retryable=retryable,
        intended_slot_credit=rule.intended_slot_credit,
        blockers=[] if terminal_status == "PASS" else ["terminal_test_blocker"],
    )

    assert scheduler_terminal_exit_code(receipt) == rule.exit_code


@pytest.mark.parametrize(
    ("terminal_status", "process_health", "business_state", "retryable", "credit"),
    [
        ("PASS", "DEGRADED", "PASS", False, True),
        ("BLOCKED", "HEALTHY", "FAILED", False, False),
        ("INCOMPLETE", "DEGRADED", "INCOMPLETE", False, False),
        ("FAILED", "FAILED", "FAILED", True, False),
        ("DEFERRED", "HEALTHY", "DEFERRED", False, True),
    ],
)
def test_terminal_state_rule_rejects_cross_field_contradictions(
    tmp_path: Path,
    terminal_status: str,
    process_health: str,
    business_state: str,
    retryable: bool,
    credit: bool,
) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)

    with pytest.raises(ValueError, match="terminal"):
        build_scheduler_terminal_receipt(
            run_id=new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started),
            intended_slot="slot",
            runtime_identity=_identity(tmp_path),
            started_at=started,
            completed_at=started,
            terminal_status=terminal_status,
            process_health=process_health,
            business_state=business_state,
            retryable=retryable,
            intended_slot_credit=credit,
            blockers=[] if terminal_status == "PASS" else ["terminal_test_blocker"],
        )


def test_research_scheduler_receipt_rejects_order_activity(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    identity = _identity(tmp_path)

    with pytest.raises(ValueError, match="cannot contain order activity"):
        build_scheduler_terminal_receipt(
            run_id=new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started),
            intended_slot="slot",
            runtime_identity=identity,
            started_at=started,
            completed_at=started,
            terminal_status="FAILED",
            process_health="FAILED",
            business_state="FAILED",
            retryable=False,
            intended_slot_credit=False,
            blockers=["unexpected_order_attempt"],
            order_attempts=1,
        )


def test_terminal_receipt_rejects_tampering(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    identity = _identity(tmp_path)
    receipt = build_scheduler_terminal_receipt(
        run_id=new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started),
        intended_slot="slot",
        runtime_identity=identity,
        started_at=started,
        completed_at=started,
        terminal_status="PASS",
        process_health="HEALTHY",
        business_state="PASS",
        retryable=False,
        intended_slot_credit=True,
    )
    receipt["terminal_status"] = "BLOCKED"

    with pytest.raises(ValueError, match="payload hash mismatch"):
        publish_scheduler_terminal_receipt(tmp_path, receipt)


def test_terminal_lane_remains_available_during_maintenance(tmp_path: Path) -> None:
    _prepare_root(tmp_path)
    marker = tmp_path / ".runtime_control" / "phase00_maintenance.json"
    marker.parent.mkdir()
    marker.write_text(
        '{"maintenance_id":"maintenance_fixture","status":"ACTIVE"}\n',
        encoding="utf-8",
    )
    started = datetime(2026, 8, 21, 12, tzinfo=UTC)
    identity = _identity(tmp_path)
    run_id = new_scheduler_run_id(scheduler_key="hyperliquid_l2", now=started)
    intent = build_scheduler_run_intent(
        run_id=run_id,
        intended_slot="slot",
        runtime_identity=identity,
        started_at=started,
    )
    receipt = build_scheduler_terminal_receipt(
        run_id=run_id,
        intended_slot="slot",
        runtime_identity=identity,
        started_at=started,
        completed_at=started,
        terminal_status="DEFERRED",
        process_health="HEALTHY",
        business_state="DEFERRED",
        retryable=True,
        intended_slot_credit=False,
        blockers=["phase00_maintenance_active"],
    )

    publish_scheduler_run_intent(tmp_path, intent)
    paths = publish_scheduler_terminal_receipt(tmp_path, receipt)

    assert paths["terminal_receipt"].is_file()
    with pytest.raises(GovernedEvidenceMaintenanceActive):
        atomic_write_text(tmp_path / "reports" / "ordinary.json", "{}\n")


def test_terminal_lane_rejects_non_terminal_target(tmp_path: Path) -> None:
    _prepare_root(tmp_path)

    with (
        scheduler_terminal_write_lock(tmp_path),
        pytest.raises(PublicationLeaseError, match="terminal_target_denied"),
    ):
        atomic_write_text(
            tmp_path / "reports" / "active" / "authority.json",
            "{}\n",
            publication_scope="scheduler_terminal",
        )
