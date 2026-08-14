from __future__ import annotations

import json
import os
import plistlib
import subprocess
from collections import namedtuple
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_platform.orchestration.corrective_runtime import (
    launch_agent_runtime_environment,
    workspace_launch_agent_path,
)
from quant_platform.orchestration.corrective_scheduler_runtime_readiness import (
    SCHEDULER_CONTRACTS,
    build_corrective_scheduler_runtime_readiness,
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
    immutable = json.loads(
        result.paths["immutable_receipt"].read_text(encoding="utf-8")
    )
    assert immutable["receipt_id"] == result.summary["receipt_id"]
    daily = next(
        row for row in result.summary["services"] if row["service"] == "daily_research"
    )
    assert daily["live_last_exit_code"] == "(never exited)"


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
        live_mutations={
            "com.thewiz.corrective-research-daily": {mutation: "/wrong/runtime"}
        },
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

    l2 = next(
        row for row in result.summary["services"] if row["service"] == "hyperliquid_l2"
    )
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


def _prepare_scheduler_contracts(root: Path) -> Path:
    python = root / ".venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    runtime_temp = root / ".runtime_tmp"
    runtime_temp.mkdir()
    runtime_temp.chmod(0o700)
    system_dir = root / "system_launch_agents"
    system_dir.mkdir()
    environment = launch_agent_runtime_environment(root)
    logs = root / "reports" / "active" / "schedule_logs"
    for contract in SCHEDULER_CONTRACTS:
        payload: dict[str, object] = {
            "Label": contract.label,
            "ProgramArguments": [
                str(python),
                "-m",
                contract.module,
                contract.action,
            ],
            "WorkingDirectory": str(root),
            "EnvironmentVariables": environment,
            "RunAtLoad": False,
            "StandardOutPath": str(logs / contract.stdout_name),
            "StandardErrorPath": str(logs / contract.stderr_name),
        }
        if contract.interval_seconds is not None:
            payload["StartInterval"] = contract.interval_seconds
        else:
            payload["StartCalendarInterval"] = {
                "Hour": contract.calendar_hour,
                "Minute": contract.calendar_minute,
            }
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
    ) -> subprocess.CompletedProcess[str]:
        del capture_output, text, check
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
    python = str(root / ".venv312" / "bin" / "python")
    runtime_temp = str(root / ".runtime_tmp")
    path = mutation.get("path", str(system_dir / f"{contract.label}.plist"))
    program = mutation.get("program", python)
    tmpdir = mutation.get("tmpdir", runtime_temp)
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
        f"\t\t{python}",
        "\t\t-m",
        f"\t\t{contract.module}",
        f"\t\t{contract.action}",
        "\t}",
        "",
        f"\tworking directory = {root}",
        "",
        f"\tstdout path = {root / 'reports' / 'active' / 'schedule_logs' / contract.stdout_name}",
        f"\tstderr path = {root / 'reports' / 'active' / 'schedule_logs' / contract.stderr_name}",
        "\tenvironment = {",
        f"\t\tTMP => {runtime_temp}",
        f"\t\tTEMP => {runtime_temp}",
        f"\t\tPYTHONPATH => {root / 'src'}",
        f"\t\tTMPDIR => {tmpdir}",
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
