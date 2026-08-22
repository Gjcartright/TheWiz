from __future__ import annotations

import json
import os
import plistlib
import subprocess
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_runtime import (
    SCHEDULER_LOG_MAX_BYTES,
    launch_agent_runtime_environment,
    scheduler_contract,
    scheduler_launch_agent_payload,
    scheduler_log_directory,
    scheduler_python_path,
    scheduler_run_identity,
    workspace_launch_agent_path,
)
from quant_platform.orchestration.corrective_scheduler_runtime_readiness import (
    LAUNCHCTL_OUTPUT_MAX_BYTES,
    SCHEDULER_CONTRACTS,
    build_corrective_scheduler_runtime_readiness,
)
from quant_platform.orchestration.corrective_scheduler_terminal import (
    build_scheduler_terminal_receipt,
    claim_scheduler_slot,
    new_scheduler_run_id,
    publish_scheduler_terminal_receipt,
)

NOW = datetime(2026, 8, 11, 17, 30, tzinfo=UTC)
DiskUsage = namedtuple("DiskUsage", "total used free")


def test_all_three_live_scheduler_contracts_pass_and_remain_zero_authority(
    tmp_path: Path,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    runner = _runner(tmp_path, system_dir)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "PASS_SCHEDULER_RUNTIME_READY"
    assert result.summary["agents_ready"] == 3
    assert result.summary["checks_passed"] == result.summary["checks_total"]
    assert result.summary["orders_submitted"] == 0
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.summary["operational_warnings"]
    immutable = json.loads(result.paths["immutable_receipt"].read_text(encoding="utf-8"))
    assert immutable["receipt_id"] == result.summary["receipt_id"]
    daily = next(row for row in result.summary["services"] if row["service"] == "daily_research")
    assert daily["live_last_exit_code"] == "(never exited)"
    assert all(
        stream["status"] == "PASS"
        and stream["mode"] == "0600"
        and stream["redaction_status"] == "PASS_NO_SECRET_PATTERNS"
        and len(stream["content_sha256"]) == 64
        for service in result.summary["services"]
        for stream in (service["stdout_log"], service["stderr_log"])
    )


def test_system_workspace_plist_mismatch_blocks(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    target = system_dir / "com.thewiz.corrective-l2-cadence.plist"
    payload = plistlib.loads(target.read_bytes())
    payload["StartInterval"] = 301
    target.write_bytes(plistlib.dumps(payload, sort_keys=True))

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "hyperliquid_l2_system_workspace_plist_mismatch" in result.summary["blockers"]


def test_live_tmpdir_mismatch_blocks(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    runner = _runner(
        tmp_path,
        system_dir,
        live_mutations={"com.thewiz.corrective-wizard-proof": {"tmpdir": "/tmp"}},
    )

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "wizard_proof_live_environment_mismatch" in result.summary["blockers"]


@pytest.mark.parametrize("mutation", ["path", "program"])
def test_live_identity_mismatch_blocks(tmp_path: Path, mutation: str) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    runner = _runner(
        tmp_path,
        system_dir,
        live_mutations={"com.thewiz.corrective-research-daily": {mutation: "/wrong/runtime"}},
    )

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    expected = (
        "daily_research_live_loaded_plist_path_mismatch"
        if mutation == "path"
        else "daily_research_live_program_mismatch"
    )
    assert expected in result.summary["blockers"]


def test_nonzero_last_exit_blocks(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    runner = _runner(
        tmp_path,
        system_dir,
        live_mutations={"com.thewiz.corrective-l2-cadence": {"last_exit": "70"}},
    )

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert "hyperliquid_l2_live_last_exit_unsafe" in result.summary["blockers"]


def test_nonempty_stderr_with_newer_success_is_reported_as_recovered_warning(
    tmp_path: Path,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    logs = tmp_path / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True, exist_ok=True)
    stderr_path = logs / "l2.stderr.log"
    stdout_path = logs / "l2.stdout.log"
    stderr_path.write_text("historical traceback\n", encoding="utf-8")
    stdout_path.write_text("newer successful receipt\n", encoding="utf-8")
    stderr_time = NOW.timestamp() - 600
    stdout_time = NOW.timestamp() - 300
    os.utime(stderr_path, (stderr_time, stderr_time))
    os.utime(stdout_path, (stdout_time, stdout_time))

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    l2 = next(row for row in result.summary["services"] if row["service"] == "hyperliquid_l2")
    assert result.summary["status"] == "PASS_SCHEDULER_RUNTIME_READY"
    assert l2["stderr_classification"] == "HISTORICAL_RECOVERED"
    assert l2["stderr_bytes"] == len("historical traceback\n")
    assert l2["stdout_modified_at_utc"] > l2["stderr_modified_at_utc"]
    assert any(
        warning.startswith("hyperliquid_l2_stderr_historical_recovered:")
        for warning in result.summary["operational_warnings"]
    )


def test_immutable_receipt_collision_fails_before_active_aliases_change(
    tmp_path: Path,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    arguments = {
        "root": tmp_path,
        "now": NOW,
        "runner": _runner(tmp_path, system_dir),
        "disk_usage": _disk_usage,
        "system_launch_agents_dir": system_dir,
    }
    result = build_corrective_scheduler_runtime_readiness(**arguments)
    active_before = result.paths["status"].read_text(encoding="utf-8")
    result.paths["immutable_receipt"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable scheduler runtime receipt collision"):
        build_corrective_scheduler_runtime_readiness(**arguments)

    assert result.paths["status"].read_text(encoding="utf-8") == active_before


def test_source_mutation_after_install_blocks_static_runtime_identity(
    tmp_path: Path,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    source = tmp_path / "src" / "quant_platform" / "mutated.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 'changed-after-install'\n", encoding="utf-8")

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "daily_research_static_environment_mismatch" in result.summary["blockers"]
    assert all(
        service["runtime_contract_sha256"] for service in result.summary["runtime_contracts"]
    )


def test_launchctl_timeout_is_bounded_and_blocks(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    base_runner = _runner(tmp_path, system_dir)

    def timeout_runner(command, **kwargs):
        if command[0] == "launchctl":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return base_runner(command, **kwargs)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=timeout_runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "daily_research_launchctl_timeout" in result.summary["blockers"]
    assert all(
        service["launchctl_status"] == "TIMEOUT"
        and service["launchctl_probe"]["detail_code"] == "launchctl_timeout"
        and "stdout" not in service["launchctl_probe"]
        and "stderr" not in service["launchctl_probe"]
        for service in result.summary["services"]
    )


@pytest.mark.parametrize(
    ("fault", "expected_status", "expected_blocker"),
    [
        ("timeout", "TIMEOUT", "daily_research_launchctl_timeout"),
        ("os_error", "ERROR", "daily_research_launchctl_error"),
        ("unknown_format", "UNSUPPORTED", "daily_research_launchctl_unsupported"),
        ("oversized", "UNSUPPORTED", "daily_research_launchctl_unsupported"),
    ],
)
def test_non_state_launchctl_results_never_satisfy_unloaded_policy(
    tmp_path: Path,
    fault: str,
    expected_status: str,
    expected_blocker: str,
) -> None:
    _write_desired_state(tmp_path, state="UNLOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    base_runner = _runner(tmp_path, system_dir)

    def fault_runner(command, **kwargs):
        if command[0] != "launchctl":
            return base_runner(command, **kwargs)
        if fault == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if fault == "os_error":
            raise OSError("offline launchctl fixture")
        output = (
            "x" * (LAUNCHCTL_OUTPUT_MAX_BYTES + 1)
            if fault == "oversized"
            else "new launchctl format without reviewed fields\n"
        )
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=fault_runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert expected_blocker in result.summary["blockers"]
    assert all(
        service["launchctl_status"] == expected_status
        and len(service["launchctl_probe"]["stdout_sha256"]) == 64
        and len(service["launchctl_probe"]["stderr_sha256"]) == 64
        and len(service["launchctl_probe"]["detail_code"]) <= 128
        and set(service["launchctl_probe"])
        == {
            "status",
            "return_code",
            "detail_code",
            "stdout_bytes",
            "stderr_bytes",
            "stdout_sha256",
            "stderr_sha256",
            "output_limit_bytes",
        }
        for service in result.summary["services"]
    )


def test_unrecognized_nonzero_launchctl_response_is_error_not_unloaded(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="UNLOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    base_runner = _runner(tmp_path, system_dir)

    def permission_runner(command, **kwargs):
        if command[0] == "launchctl":
            return subprocess.CompletedProcess(
                command,
                113,
                stdout="",
                stderr="permission denied",
            )
        return base_runner(command, **kwargs)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=permission_runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert all(
        service["launchctl_status"] == "ERROR"
        and service["launchctl_probe"]["detail_code"] == "launchctl_nonzero_exit_113"
        for service in result.summary["services"]
    )


def test_service_not_found_text_cannot_hide_loaded_payload(tmp_path: Path) -> None:
    _write_desired_state(tmp_path, state="UNLOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    base_runner = _runner(tmp_path, system_dir)

    def contradictory_runner(command, **kwargs):
        if command[0] == "launchctl":
            return subprocess.CompletedProcess(
                command,
                113,
                stdout="unexpected loaded payload",
                stderr="service not found",
            )
        return base_runner(command, **kwargs)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=contradictory_runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert all(service["launchctl_status"] == "ERROR" for service in result.summary["services"])


def test_log_retention_policy_must_match_runtime_contract(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    _write_log_retention_policy(
        tmp_path,
        log_rotation_threshold_bytes=SCHEDULER_LOG_MAX_BYTES + 1,
    )

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "scheduler_log_retention_policy_invalid" in result.summary["blockers"]
    assert result.summary["log_policy_observation"]["violations"] == [
        "log_rotation_threshold_bytes"
    ]


@pytest.mark.parametrize("mutation", ["mode", "symlink"])
def test_log_directory_contract_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    logs = scheduler_log_directory(tmp_path)
    if mutation == "mode":
        logs.chmod(0o755)
    else:
        target = logs.with_name("schedule_logs_target")
        logs.rename(target)
        logs.symlink_to(target, target_is_directory=True)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "scheduler_log_directory_invalid" in result.summary["blockers"]
    expected_violation = "mode_invalid" if mutation == "mode" else "symlink_not_allowed"
    assert expected_violation in result.summary["log_directory_observation"]["violations"]


@pytest.mark.parametrize(
    ("mutation", "expected_blocker"),
    [
        ("mode", "daily_research_stdout_log_mode_invalid"),
        ("oversized", "daily_research_stdout_log_size_exceeded"),
        ("symlink", "daily_research_stdout_log_symlink_not_allowed"),
        ("not_regular", "daily_research_stdout_log_not_regular"),
    ],
)
def test_log_file_contract_fails_closed(
    tmp_path: Path,
    mutation: str,
    expected_blocker: str,
) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    path = scheduler_log_directory(tmp_path) / "daily.stdout.log"
    if mutation == "mode":
        path.chmod(0o644)
    elif mutation == "oversized":
        path.write_bytes(b"")
        path.chmod(0o600)
        with path.open("r+b") as handle:
            handle.truncate(SCHEDULER_LOG_MAX_BYTES + 1)
    elif mutation == "symlink":
        path.unlink()
        path.symlink_to(scheduler_log_directory(tmp_path) / "daily.stderr.log")
    else:
        path.unlink()
        path.mkdir()

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert expected_blocker in result.summary["blockers"]


def test_log_redaction_blocks_without_exposing_secret(tmp_path: Path) -> None:
    system_dir = _prepare_scheduler_contracts(tmp_path)
    secret = "should-never-appear-in-readiness-evidence"
    path = scheduler_log_directory(tmp_path) / "l2.stderr.log"
    path.write_text(f"CRYPTO_WIZARDS_API_KEY={secret}\n", encoding="utf-8")
    path.chmod(0o600)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    encoded_summary = json.dumps(result.summary, sort_keys=True)
    l2 = next(
        service for service in result.summary["services"] if service["service"] == "hyperliquid_l2"
    )
    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "hyperliquid_l2_stderr_log_secret_pattern_detected" in result.summary["blockers"]
    assert l2["stderr_log"]["redaction_status"] == ("BLOCKED_SECRET_PATTERN_DETECTED")
    assert l2["stderr_log"]["secret_pattern_match_count"] == 1
    assert secret not in encoded_summary


def test_public_installers_publish_exact_readiness_contract(tmp_path: Path) -> None:
    from quant_platform.orchestration.corrective_daily_scheduler import (
        install_daily_launch_agent,
    )
    from quant_platform.orchestration.corrective_l2_scheduler import (
        install_corrective_l2_launch_agent,
    )
    from quant_platform.orchestration.corrective_wizard_proof_scheduler import (
        install_corrective_wizard_proof_launch_agent,
    )

    system_dir = _prepare_runtime_identity(tmp_path)
    results = (
        install_daily_launch_agent(
            root=tmp_path,
            system_path=system_dir / "com.thewiz.corrective-research-daily.plist",
        ),
        install_corrective_l2_launch_agent(
            root=tmp_path,
            system_path=system_dir / "com.thewiz.corrective-l2-cadence.plist",
        ),
        install_corrective_wizard_proof_launch_agent(
            root=tmp_path,
            system_path=system_dir / "com.thewiz.corrective-wizard-proof.plist",
        ),
    )

    readiness = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert all(result["status"] == "PERSISTED_NOT_LOADED" for result in results)
    assert readiness.summary["status"] == "PASS_SCHEDULER_RUNTIME_READY"


def test_explicit_unloaded_desired_state_is_ready_without_terminal_receipts(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="UNLOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    base_runner = _runner(tmp_path, system_dir)

    def unloaded_runner(command, **kwargs):
        if command[0] == "launchctl":
            return subprocess.CompletedProcess(
                command,
                113,
                stdout="",
                stderr="service not found",
            )
        return base_runner(command, **kwargs)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=unloaded_runner,
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "PASS_SCHEDULER_RUNTIME_READY"
    assert result.summary["desired_state_explicit"] is True
    assert all(
        service["desired_state"] == "UNLOADED"
        and service["observed_loaded"] is False
        and service["launchctl_status"] == "UNLOADED"
        and service["launchctl_probe"]["return_code"] == 113
        and service["launchctl_probe"]["detail_code"] == "launchctl_service_not_found"
        for service in result.summary["services"]
    )


def test_explicit_unloaded_desired_state_blocks_unexpected_loaded_service(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="UNLOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )
    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "hyperliquid_l2_launch_agent_unexpectedly_loaded" in result.summary["blockers"]


def test_explicit_loaded_desired_state_requires_terminal_receipt(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="LOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )
    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "hyperliquid_l2_terminal_pointer_missing_or_nonregular" in result.summary["blockers"]


def test_explicit_loaded_state_accepts_only_canonical_current_terminal_receipts(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="LOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    for contract in SCHEDULER_CONTRACTS:
        _publish_terminal_pass(tmp_path, contract)

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "PASS_SCHEDULER_RUNTIME_READY"


def test_schema_less_substituted_terminal_receipt_cannot_pass_readiness(
    tmp_path: Path,
) -> None:
    _write_desired_state(tmp_path, state="LOADED")
    system_dir = _prepare_scheduler_contracts(tmp_path)
    for contract in SCHEDULER_CONTRACTS:
        _publish_terminal_pass(tmp_path, contract)
    pointer_path = (
        tmp_path
        / "reports"
        / "active"
        / "scheduler_terminal_receipts"
        / "hyperliquid_l2_latest.json"
    )
    valid_pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer_path.write_text(
        json.dumps(
            {
                "receipt_id": valid_pointer["receipt_id"],
                "receipt_path": valid_pointer["receipt_path"],
                "receipt_sha256": valid_pointer["receipt_sha256"],
            }
        ),
        encoding="utf-8",
    )

    result = build_corrective_scheduler_runtime_readiness(
        root=tmp_path,
        now=NOW,
        runner=_runner(tmp_path, system_dir),
        disk_usage=_disk_usage,
        system_launch_agents_dir=system_dir,
    )

    assert result.summary["status"] == "BLOCKED_SCHEDULER_RUNTIME"
    assert "hyperliquid_l2_terminal_pointer_schema_fields_invalid" in result.summary["blockers"]


def _publish_terminal_pass(root: Path, contract) -> None:
    completed = NOW - timedelta(seconds=5)
    started = completed - timedelta(seconds=2)
    runtime_contract = scheduler_contract(contract.key)
    environment = launch_agent_runtime_environment(root, contract=runtime_contract)
    identity = scheduler_run_identity(
        root,
        contract=runtime_contract,
        environment=environment,
        require_launchd=True,
    )
    run_id = new_scheduler_run_id(scheduler_key=contract.key, now=started)
    intended_slot = f"{completed.isoformat()}/test"
    claim_scheduler_slot(
        root,
        run_id=run_id,
        intended_slot=intended_slot,
        runtime_identity=identity,
        claimed_at=started,
    )
    receipt = build_scheduler_terminal_receipt(
        run_id=run_id,
        intended_slot=intended_slot,
        runtime_identity=identity,
        started_at=started,
        completed_at=completed,
        terminal_status="PASS",
        process_health="HEALTHY",
        business_state="PASS",
        retryable=False,
        intended_slot_credit=True,
    )
    publish_scheduler_terminal_receipt(root, receipt)


def _write_desired_state(root: Path, *, state: str) -> None:
    path = root / "config" / "scheduler_desired_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "thewiz.scheduler_desired_state.v1",
                "generation": 1,
                "services": {contract.key: state for contract in SCHEDULER_CONTRACTS},
                "reason": "test_policy",
                "reload_requires_separate_operator_approval": True,
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )


def _prepare_runtime_identity(root: Path) -> Path:
    python = scheduler_python_path(root)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    (root / ".venv" / "pyvenv.cfg").write_text("home = fixture\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='0.0.0'\nrequires-python='>=3.11'\n",
        encoding="utf-8",
    )
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    runtime_temp = root / ".runtime_tmp"
    runtime_temp.mkdir()
    runtime_temp.chmod(0o700)
    logs = scheduler_log_directory(root)
    logs.mkdir(parents=True)
    logs.chmod(0o700)
    for contract in SCHEDULER_CONTRACTS:
        for name in (contract.stdout_name, contract.stderr_name):
            log_path = logs / name
            log_path.touch(mode=0o600)
            log_path.chmod(0o600)
    _write_log_retention_policy(root)
    system_dir = root / "system_launch_agents"
    system_dir.mkdir()
    return system_dir


def _write_log_retention_policy(
    root: Path,
    **overrides: object,
) -> Path:
    payload: dict[str, object] = {
        "schema_version": "thewiz.corrective_artifact_retention_policy.v1",
        "operational_receipt_directories": ["data/research/operational"],
        "receipt_retention_days": 14,
        "minimum_newest_per_directory": 1,
        "log_paths": ["reports/active/schedule_logs/*.log"],
        "log_rotation_threshold_bytes": SCHEDULER_LOG_MAX_BYTES,
        "log_retention_days": 14,
        "log_rotations_to_keep": 8,
        "log_file_mode": "0600",
        "log_symlinks_allowed": False,
        "log_secret_content_allowed": False,
        "never_archive_directories": ["data/research/scientific"],
    }
    payload.update(overrides)
    path = root / "config" / "corrective_artifact_retention.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prepare_scheduler_contracts(root: Path) -> Path:
    system_dir = _prepare_runtime_identity(root)
    for contract in SCHEDULER_CONTRACTS:
        payload = scheduler_launch_agent_payload(root, contract=contract)
        encoded = plistlib.dumps(payload, sort_keys=True)
        workspace = workspace_launch_agent_path(root, contract.label)
        workspace.parent.mkdir(parents=True, exist_ok=True)
        workspace.write_bytes(encoded)
        (system_dir / f"{contract.label}.plist").write_bytes(encoded)
    return system_dir


def _runner(
    root: Path,
    system_dir: Path,
    *,
    live_mutations: dict[str, dict[str, str]] | None = None,
):
    live_mutations = live_mutations or {}
    contracts = {contract.label: contract for contract in SCHEDULER_CONTRACTS}

    def run(
        command: list[str],
        *,
        capture_output: bool,
        text: bool,
        check: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check, timeout
        if command[0] == "uv":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="All installed packages are compatible\n",
                stderr="",
            )
        if command[0] != "launchctl":
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "executable": str(scheduler_python_path(root)),
                        "version_info": [3, 12, 7],
                        "packages": [["fixture", "0.0.0"]],
                    }
                ),
                stderr="",
            )
        label = command[-1].rsplit("/", 1)[-1]
        contract = contracts[label]
        mutation = live_mutations.get(label, {})
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=_launchctl_payload(
                root,
                system_dir,
                contract,
                mutation=mutation,
            ),
            stderr="",
        )

    return run


def _launchctl_payload(
    root: Path,
    system_dir: Path,
    contract,
    *,
    mutation: dict[str, str],
) -> str:
    payload = scheduler_launch_agent_payload(root, contract=contract)
    python = str(scheduler_python_path(root))
    environment = dict(payload["EnvironmentVariables"])
    path = mutation.get("path", str(system_dir / f"{contract.label}.plist"))
    program = mutation.get("program", python)
    environment["TMPDIR"] = mutation.get("tmpdir", environment["TMPDIR"])
    last_exit = mutation.get(
        "last_exit",
        "(never exited)" if contract.key == "daily_research" else "0",
    )
    lines = [
        f"gui/{os.getuid()}/{contract.label} = {{",
        "\tactive count = 0",
        f"\tpath = {path}",
        "\ttype = LaunchAgent",
        "\tstate = not running",
        "",
        f"\tprogram = {program}",
        "\targuments = {",
        *[f"\t\t{argument}" for argument in payload["ProgramArguments"]],
        "\t}",
        "",
        f"\tworking directory = {root}",
        "",
        f"\tstdout path = {payload['StandardOutPath']}",
        f"\tstderr path = {payload['StandardErrorPath']}",
        "\tenvironment = {",
        *[f"\t\t{key} => {value}" for key, value in environment.items()],
        f"\t\tXPC_SERVICE_NAME => {contract.label}",
        "\t}",
        "\truns = 0",
        f"\tlast exit code = {last_exit}",
    ]
    if contract.interval_seconds is not None:
        lines.append(f"\trun interval = {contract.interval_seconds} seconds")
    else:
        lines.extend(
            [
                "\tevent triggers = {",
                "\t\tdaily => {",
                "\t\t\tdescriptor = {",
                f'\t\t\t\t"Minute" => {contract.calendar_minute}',
                f'\t\t\t\t"Hour" => {contract.calendar_hour}',
                "\t\t\t}",
                "\t\t}",
                "\t}",
            ]
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _disk_usage(path: Path) -> DiskUsage:
    if path == Path.home():
        return DiskUsage(total=10 * 1024**3, used=10 * 1024**3, free=256 * 1024**2)
    return DiskUsage(total=20 * 1024**3, used=1 * 1024**3, free=19 * 1024**3)
