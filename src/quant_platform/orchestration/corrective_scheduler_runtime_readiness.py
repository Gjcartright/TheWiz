"""Prove that every corrective LaunchAgent matches its reviewed runtime contract."""

from __future__ import annotations

import csv
import json
import os
import plistlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    launch_agent_runtime_environment,
    runtime_temp_directory,
    workspace_launch_agent_path,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_scheduler_runtime_readiness.v1"
MIN_RUNTIME_TEMP_FREE_BYTES = 1024**3
SYSTEM_DISK_WARNING_BYTES = 1024**3


@dataclass(frozen=True)
class SchedulerContract:
    key: str
    label: str
    module: str
    action: str
    stdout_name: str
    stderr_name: str
    interval_seconds: int | None = None
    calendar_hour: int | None = None
    calendar_minute: int | None = None


SCHEDULER_CONTRACTS = (
    SchedulerContract(
        key="daily_research",
        label="com.thewiz.corrective-research-daily",
        module="quant_platform.orchestration.corrective_daily_scheduler",
        action="--execute",
        stdout_name="daily.stdout.log",
        stderr_name="daily.stderr.log",
        calendar_hour=6,
        calendar_minute=15,
    ),
    SchedulerContract(
        key="hyperliquid_l2",
        label="com.thewiz.corrective-l2-cadence",
        module="quant_platform.orchestration.corrective_l2_scheduler",
        action="--capture",
        stdout_name="l2.stdout.log",
        stderr_name="l2.stderr.log",
        interval_seconds=300,
    ),
    SchedulerContract(
        key="wizard_proof",
        label="com.thewiz.corrective-wizard-proof",
        module="quant_platform.orchestration.corrective_wizard_proof_launcher",
        action="--execute",
        stdout_name="wizard_proof.stdout.log",
        stderr_name="wizard_proof.stderr.log",
        interval_seconds=600,
    ),
)


def build_corrective_scheduler_runtime_readiness(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    system_launch_agents_dir: Path | None = None,
    minimum_runtime_temp_free_bytes: int = MIN_RUNTIME_TEMP_FREE_BYTES,
    system_disk_warning_bytes: int = SYSTEM_DISK_WARNING_BYTES,
) -> CommandResult:
    """Audit all live scheduler definitions without starting jobs or placing orders."""

    checked_at = _as_utc(now)
    runtime_temp = runtime_temp_directory(root)
    expected_environment = launch_agent_runtime_environment(root)
    system_launch_agents_dir = system_launch_agents_dir or (
        Path.home() / "Library" / "LaunchAgents"
    )
    checks: list[dict[str, str]] = []
    service_summaries: list[dict[str, Any]] = []

    temp_ready, temp_observed, runtime_free_bytes = _runtime_temp_viability(
        runtime_temp,
        disk_usage=disk_usage,
        minimum_free_bytes=minimum_runtime_temp_free_bytes,
    )
    _add_check(
        checks,
        service="all",
        name="workspace_runtime_temp_viability",
        passed=temp_ready,
        observed=temp_observed,
        expected=f"private_writable_non_symlink;free_bytes>={minimum_runtime_temp_free_bytes}",
        evidence_path=str(runtime_temp),
        blocker="scheduler_workspace_runtime_temp_not_viable",
    )

    operational_warnings: list[str] = []
    system_free_bytes: int | None = None
    try:
        system_free_bytes = int(disk_usage(Path.home()).free)
        if system_free_bytes < system_disk_warning_bytes:
            operational_warnings.append(
                "system_volume_free_space_low:"
                f"{system_free_bytes}<{system_disk_warning_bytes}"
            )
    except OSError as exc:
        operational_warnings.append(
            f"system_volume_free_space_unavailable:{type(exc).__name__}"
        )

    for contract in SCHEDULER_CONTRACTS:
        service_checks_before = len(checks)
        expected = _expected_contract(root, contract, expected_environment)
        workspace_path = workspace_launch_agent_path(root, contract.label)
        system_path = system_launch_agents_dir / f"{contract.label}.plist"
        workspace_payload, workspace_error = _read_plist(workspace_path)
        _, system_error = _read_plist(system_path)

        _add_check(
            checks,
            service=contract.key,
            name="workspace_plist_parse",
            passed=not workspace_error,
            observed="valid" if not workspace_error else workspace_error,
            expected="valid_dictionary_plist",
            evidence_path=str(workspace_path),
            blocker=f"{contract.key}_workspace_plist_invalid",
        )
        _add_check(
            checks,
            service=contract.key,
            name="system_plist_parse",
            passed=not system_error,
            observed="valid" if not system_error else system_error,
            expected="valid_dictionary_plist",
            evidence_path=str(system_path),
            blocker=f"{contract.key}_system_plist_invalid",
        )
        workspace_sha = _file_sha256(workspace_path)
        system_sha = _file_sha256(system_path)
        mirror_valid = bool(
            workspace_sha and system_sha and workspace_sha == system_sha
        )
        _add_check(
            checks,
            service=contract.key,
            name="system_workspace_byte_identity",
            passed=mirror_valid,
            observed=f"workspace={workspace_sha};system={system_sha}",
            expected="identical_sha256",
            evidence_path=f"{workspace_path};{system_path}",
            blocker=f"{contract.key}_system_workspace_plist_mismatch",
        )
        _add_static_contract_checks(
            checks,
            contract=contract,
            payload=workspace_payload,
            expected=expected,
            evidence_path=workspace_path,
        )

        live_output, launch_error = _launchctl_print(contract.label, runner)
        live = _parse_launchctl_print(live_output) if not launch_error else {}
        _add_check(
            checks,
            service=contract.key,
            name="launch_agent_loaded",
            passed=not launch_error,
            observed="loaded" if not launch_error else launch_error,
            expected=f"gui/{os.getuid()}/{contract.label}",
            evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
            blocker=f"{contract.key}_launch_agent_not_loaded",
        )
        _add_live_contract_checks(
            checks,
            contract=contract,
            live=live,
            raw_output=live_output,
            expected=expected,
            system_path=system_path,
            loaded=not launch_error,
        )
        live_last_exit = str(live.get("last exit code", ""))
        log_observation = _scheduler_log_observation(
            root=root,
            contract=contract,
            live_exit_safe=not launch_error
            and live_last_exit in {"0", "(never exited)"},
        )
        if log_observation["stderr_classification"] != "CLEAN":
            operational_warnings.append(
                f"{contract.key}_stderr_"
                f"{str(log_observation['stderr_classification']).lower()}:"
                f"bytes={log_observation['stderr_bytes']};"
                f"modified_at_utc={log_observation['stderr_modified_at_utc']}"
            )
        service_checks = checks[service_checks_before:]
        service_blockers = [
            row["blocker"] for row in service_checks if row["status"] == "BLOCKED"
        ]
        service_summaries.append(
            {
                "service": contract.key,
                "label": contract.label,
                "status": "PASS" if not service_blockers else "BLOCKED",
                "checks_total": len(service_checks),
                "checks_passed": sum(
                    row["status"] == "PASS" for row in service_checks
                ),
                "workspace_plist": str(workspace_path),
                "system_plist": str(system_path),
                "workspace_plist_sha256": workspace_sha,
                "system_plist_sha256": system_sha,
                "live_state": str(live.get("state", "")),
                "live_runs": _safe_int(live.get("runs"), default=-1),
                "live_last_exit_code": live_last_exit,
                **log_observation,
                "blockers": service_blockers,
            }
        )

    blockers = [row["blocker"] for row in checks if row["status"] == "BLOCKED"]
    agents_ready = sum(row["status"] == "PASS" for row in service_summaries)
    status = (
        "PASS_SCHEDULER_RUNTIME_READY"
        if not blockers and agents_ready == len(SCHEDULER_CONTRACTS)
        else "BLOCKED_SCHEDULER_RUNTIME"
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": status,
        "agents_expected": len(SCHEDULER_CONTRACTS),
        "agents_ready": agents_ready,
        "checks_total": len(checks),
        "checks_passed": sum(row["status"] == "PASS" for row in checks),
        "runtime_temp_path": str(runtime_temp),
        "runtime_temp_ready": temp_ready,
        "runtime_temp_free_bytes": runtime_free_bytes,
        "minimum_runtime_temp_free_bytes": minimum_runtime_temp_free_bytes,
        "system_volume_free_bytes": system_free_bytes,
        "system_disk_warning_bytes": system_disk_warning_bytes,
        "operational_warnings": operational_warnings,
        "services": service_summaries,
        "blockers": blockers,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "orders_submitted": 0,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_id = (
        "schedulerruntime_"
        + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    payload["receipt_id"] = receipt_id

    active = root / "reports" / "active"
    immutable = root / "data" / "research" / "scheduler_runtime_readiness"
    paths = {
        "checks": active / "scheduler_runtime_readiness_checks.csv",
        "status": active / "scheduler_runtime_readiness.json",
        "summary": active / "scheduler_runtime_readiness.md",
        "immutable_receipt": immutable / f"{receipt_id}.json",
    }
    _write_immutable_json(payload, paths["immutable_receipt"])
    _write_csv(checks, paths["checks"])
    _write_json(payload, paths["status"])
    _write_text(paths["summary"], _markdown(payload, checks))
    payload["evidence_paths"] = {
        key: _relative(path, root) for key, path in paths.items()
    }
    return CommandResult(paths=paths, summary=payload)


def _scheduler_log_observation(
    *, root: Path, contract: SchedulerContract, live_exit_safe: bool
) -> dict[str, Any]:
    """Classify append-only scheduler stderr without reading or publishing its content."""

    logs = root / "reports" / "active" / "schedule_logs"
    stdout_path = logs / contract.stdout_name
    stderr_path = logs / contract.stderr_name
    stdout_bytes, stdout_modified = _log_stat(stdout_path)
    stderr_bytes, stderr_modified = _log_stat(stderr_path)
    if stderr_bytes <= 0:
        classification = "CLEAN"
    elif (
        live_exit_safe
        and stdout_modified is not None
        and stderr_modified is not None
        and stdout_modified > stderr_modified
    ):
        classification = "HISTORICAL_RECOVERED"
    else:
        classification = "REQUIRES_OPERATOR_REVIEW"
    return {
        "stdout_path": str(stdout_path),
        "stdout_bytes": stdout_bytes,
        "stdout_modified_at_utc": _timestamp_text(stdout_modified),
        "stderr_path": str(stderr_path),
        "stderr_bytes": stderr_bytes,
        "stderr_modified_at_utc": _timestamp_text(stderr_modified),
        "stderr_classification": classification,
    }


def _log_stat(path: Path) -> tuple[int, datetime | None]:
    try:
        stat = path.stat()
    except OSError:
        return 0, None
    return stat.st_size, datetime.fromtimestamp(stat.st_mtime, tz=UTC)


def _timestamp_text(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _expected_contract(
    root: Path,
    contract: SchedulerContract,
    environment: dict[str, str],
) -> dict[str, Any]:
    logs = root / "reports" / "active" / "schedule_logs"
    expected: dict[str, Any] = {
        "label": contract.label,
        "arguments": [
            str(root / ".venv312" / "bin" / "python"),
            "-m",
            contract.module,
            contract.action,
        ],
        "working_directory": str(root),
        "environment": environment,
        "stdout_path": str(logs / contract.stdout_name),
        "stderr_path": str(logs / contract.stderr_name),
    }
    if contract.interval_seconds is not None:
        expected["interval_seconds"] = contract.interval_seconds
    else:
        expected["calendar"] = {
            "Hour": contract.calendar_hour,
            "Minute": contract.calendar_minute,
        }
    return expected


def _add_static_contract_checks(
    checks: list[dict[str, str]],
    *,
    contract: SchedulerContract,
    payload: dict[str, Any],
    expected: dict[str, Any],
    evidence_path: Path,
) -> None:
    items = (
        ("static_label", payload.get("Label"), expected["label"]),
        (
            "static_program_arguments",
            payload.get("ProgramArguments"),
            expected["arguments"],
        ),
        (
            "static_working_directory",
            payload.get("WorkingDirectory"),
            expected["working_directory"],
        ),
        (
            "static_environment",
            payload.get("EnvironmentVariables"),
            expected["environment"],
        ),
        ("static_stdout_path", payload.get("StandardOutPath"), expected["stdout_path"]),
        ("static_stderr_path", payload.get("StandardErrorPath"), expected["stderr_path"]),
    )
    for name, observed, wanted in items:
        _add_check(
            checks,
            service=contract.key,
            name=name,
            passed=observed == wanted,
            observed=_compact(observed),
            expected=_compact(wanted),
            evidence_path=str(evidence_path),
            blocker=f"{contract.key}_{name}_mismatch",
        )
    if contract.interval_seconds is not None:
        observed_schedule: Any = payload.get("StartInterval")
        expected_schedule: Any = expected["interval_seconds"]
    else:
        observed_schedule = payload.get("StartCalendarInterval")
        expected_schedule = expected["calendar"]
    _add_check(
        checks,
        service=contract.key,
        name="static_schedule",
        passed=observed_schedule == expected_schedule,
        observed=_compact(observed_schedule),
        expected=_compact(expected_schedule),
        evidence_path=str(evidence_path),
        blocker=f"{contract.key}_static_schedule_mismatch",
    )


def _add_live_contract_checks(
    checks: list[dict[str, str]],
    *,
    contract: SchedulerContract,
    live: dict[str, Any],
    raw_output: str,
    expected: dict[str, Any],
    system_path: Path,
    loaded: bool,
) -> None:
    live_environment = live.get("environment", {})
    observed_environment = (
        {key: live_environment.get(key) for key in expected["environment"]}
        if isinstance(live_environment, dict)
        else {}
    )
    items = (
        ("live_loaded_plist_path", live.get("path"), str(system_path)),
        ("live_program", live.get("program"), expected["arguments"][0]),
        ("live_program_arguments", live.get("arguments"), expected["arguments"]),
        (
            "live_working_directory",
            live.get("working directory"),
            expected["working_directory"],
        ),
        ("live_stdout_path", live.get("stdout path"), expected["stdout_path"]),
        ("live_stderr_path", live.get("stderr path"), expected["stderr_path"]),
        ("live_environment", observed_environment, expected["environment"]),
    )
    for name, observed, wanted in items:
        _add_check(
            checks,
            service=contract.key,
            name=name,
            passed=loaded and observed == wanted,
            observed=_compact(observed),
            expected=_compact(wanted),
            evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
            blocker=f"{contract.key}_{name}_mismatch",
        )
    if contract.interval_seconds is not None:
        observed_schedule = _safe_int(live.get("run interval"), default=-1)
        expected_schedule = contract.interval_seconds
        schedule_valid = loaded and observed_schedule == expected_schedule
    else:
        observed_schedule = live.get("calendar", {})
        expected_schedule = expected["calendar"]
        schedule_valid = loaded and observed_schedule == expected_schedule
    _add_check(
        checks,
        service=contract.key,
        name="live_schedule",
        passed=schedule_valid,
        observed=_compact(observed_schedule),
        expected=_compact(expected_schedule),
        evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
        blocker=f"{contract.key}_live_schedule_mismatch",
    )
    last_exit = str(live.get("last exit code", ""))
    last_exit_safe = loaded and last_exit in {"0", "(never exited)"}
    _add_check(
        checks,
        service=contract.key,
        name="live_last_exit_safe",
        passed=last_exit_safe,
        observed=last_exit or "missing",
        expected="0_or_never_exited",
        evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
        blocker=f"{contract.key}_live_last_exit_unsafe",
    )
    if loaded and contract.calendar_hour is not None and not live.get("calendar"):
        # Preserve raw evidence in the observed field when launchctl changes formatting.
        checks[-2]["observed"] = _compact(raw_output[-1000:])


def _launchctl_print(
    label: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> tuple[str, str]:
    try:
        completed = runner(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return "", f"launchctl_error:{type(exc).__name__}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "not_loaded").strip()
        return completed.stdout or "", f"launchctl_returncode_{completed.returncode}:{detail}"
    return completed.stdout or "", ""


def _parse_launchctl_print(payload: str) -> dict[str, Any]:
    lines = payload.splitlines()
    parsed: dict[str, Any] = {}
    for line in lines:
        if line.startswith("\t") and not line.startswith("\t\t") and " = " in line:
            key, value = line.strip().split(" = ", 1)
            parsed[key] = value
    parsed["arguments"] = _parse_launchctl_list(lines, "arguments")
    parsed["environment"] = _parse_launchctl_environment(lines)
    calendar: dict[str, int] = {}
    for name in ("Hour", "Minute"):
        match = re.search(rf'^\s*"{name}"\s*=>\s*(\d+)\s*$', payload, re.MULTILINE)
        if match:
            calendar[name] = int(match.group(1))
    parsed["calendar"] = calendar
    interval = re.search(r"^\s*run interval = (\d+) seconds\s*$", payload, re.MULTILINE)
    if interval:
        parsed["run interval"] = int(interval.group(1))
    return parsed


def _parse_launchctl_list(lines: list[str], name: str) -> list[str]:
    opening = f"\t{name} = {{"
    try:
        start = lines.index(opening) + 1
    except ValueError:
        return []
    values: list[str] = []
    for line in lines[start:]:
        if line == "\t}":
            break
        if line.startswith("\t\t"):
            values.append(line.strip())
    return values


def _parse_launchctl_environment(lines: list[str]) -> dict[str, str]:
    opening = "\tenvironment = {"
    try:
        start = lines.index(opening) + 1
    except ValueError:
        return {}
    values: dict[str, str] = {}
    for line in lines[start:]:
        if line == "\t}":
            break
        if " => " in line:
            key, value = line.strip().split(" => ", 1)
            values[key] = value
    return values


def _read_plist(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        return {}, "missing"
    try:
        with path.open("rb") as handle:
            payload = plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException) as exc:
        return {}, f"invalid:{type(exc).__name__}"
    if not isinstance(payload, dict):
        return {}, "not_dictionary"
    return payload, ""


def _runtime_temp_viability(
    path: Path,
    *,
    disk_usage: Callable[[Path], Any],
    minimum_free_bytes: int,
) -> tuple[bool, str, int | None]:
    if minimum_free_bytes < 0:
        return False, "minimum_free_bytes_invalid", None
    if path.is_symlink():
        return False, "symbolic_link_not_allowed", None
    if not path.is_dir():
        return False, "directory_missing", None
    try:
        mode = path.stat().st_mode
        free_bytes = int(disk_usage(path).free)
        if path.stat().st_mode & 0o077:
            return False, f"directory_not_private;mode={mode:o}", free_bytes
        if not mode & 0o200:
            return False, f"owner_write_permission_missing;free_bytes={free_bytes}", free_bytes
        if free_bytes < minimum_free_bytes:
            return (
                False,
                f"insufficient_free_bytes:{free_bytes}<{minimum_free_bytes}",
                free_bytes,
            )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".thewiz-scheduler-runtime-probe-",
            dir=path,
        ) as probe:
            probe.write(b"ready")
            probe.flush()
    except OSError as exc:
        return False, f"write_probe_failed:{type(exc).__name__}", None
    return True, f"private_writable;free_bytes={free_bytes}", free_bytes


def _add_check(
    checks: list[dict[str, str]],
    *,
    service: str,
    name: str,
    passed: bool,
    observed: str,
    expected: str,
    evidence_path: str,
    blocker: str,
) -> None:
    checks.append(
        {
            "service": service,
            "check": name,
            "status": "PASS" if passed else "BLOCKED",
            "observed": observed,
            "expected": expected,
            "evidence_path": evidence_path,
            "blocker": "" if passed else blocker,
        }
    )


def _write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "service",
                "check",
                "status",
                "observed",
                "expected",
                "evidence_path",
                "blocker",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable scheduler runtime receipt collision: {path}")
        return
    _write_text(path, encoded)


def _write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _markdown(payload: dict[str, Any], checks: list[dict[str, str]]) -> str:
    lines = [
        "# Scheduler Runtime Readiness",
        "",
        f"- Status: `{payload['status']}`",
        f"- Agents ready: `{payload['agents_ready']}/{payload['agents_expected']}`",
        f"- Checks passed: `{payload['checks_passed']}/{payload['checks_total']}`",
        f"- Runtime temp ready: `{payload['runtime_temp_ready']}`",
        f"- Operational warnings: `{';'.join(payload['operational_warnings']) or 'none'}`",
        f"- Receipt: `{payload['receipt_id']}`",
        "- Audit is read-only and grants no Testnet or live-order authority.",
        "",
        "| Service | Check | Status | Observed | Blocker |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        f"| {row['service']} | {row['check']} | {row['status']} | "
        f"{row['observed']} | {row['blocker']} |"
        for row in checks
    )
    return "\n".join(lines) + "\n"


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _compact(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value if value is not None else "missing")


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
