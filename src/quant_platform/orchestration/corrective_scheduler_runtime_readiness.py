"""Prove that every corrective LaunchAgent matches its reviewed runtime contract."""

from __future__ import annotations

import csv
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    SCHEDULER_CONTRACTS,
    SCHEDULER_LOG_MAX_BYTES,
    SCHEDULER_LOG_RETENTION_DAYS,
    SCHEDULER_PROBE_TIMEOUT_SECONDS,
    SchedulerContract,
    promote_staged_file,
    runtime_temp_directory,
    scheduler_launch_agent_payload,
    scheduler_log_directory,
    scheduler_python_path,
    scheduler_runtime_contract,
    workspace_launch_agent_path,
)
from quant_platform.orchestration.corrective_scheduler_terminal import (
    load_validated_scheduler_terminal_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_scheduler_runtime_readiness.v2"
DESIRED_STATE_SCHEMA_VERSION = "thewiz.scheduler_desired_state.v1"
DESIRED_STATE_PATH = "config/scheduler_desired_state.json"
RETENTION_POLICY_PATH = "config/corrective_artifact_retention.json"
RETENTION_POLICY_SCHEMA_VERSION = "thewiz.corrective_artifact_retention_policy.v1"
MIN_RUNTIME_TEMP_FREE_BYTES = 1024**3
SYSTEM_DISK_WARNING_BYTES = 1024**3
LAUNCHCTL_OUTPUT_MAX_BYTES = 1024**2
SCHEDULER_LOG_DIRECTORY_MODE = 0o700
SCHEDULER_LOG_FILE_MODE = 0o600
SCHEDULER_LOG_GLOB = "reports/active/schedule_logs/*.log"
LaunchctlStatus = Literal["LOADED", "UNLOADED", "TIMEOUT", "UNSUPPORTED", "ERROR"]
SECRET_LOG_PATTERNS = (
    re.compile(
        rb"(?:CRYPTO_WIZARDS_API_KEY|APIFY(?:_API)?_TOKEN|"
        rb"HYPERLIQUID_PRIVATE_KEY|PRIVATE_KEY)\s*[:=]\s*[\"']?[^\s,;\"']+",
        re.IGNORECASE,
    ),
    re.compile(rb"(?:Authorization|X-api-key)\s*[:=]\s*[^\r\n]+", re.IGNORECASE),
    re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
)


@dataclass(frozen=True)
class LaunchctlProbe:
    status: LaunchctlStatus
    return_code: int | None
    detail_code: str
    stdout_bytes: int
    stderr_bytes: int
    stdout_sha256: str
    stderr_sha256: str
    parsed: dict[str, Any]

    def evidence(self) -> dict[str, Any]:
        """Return bounded evidence without publishing launchctl output."""

        return {
            "status": self.status,
            "return_code": self.return_code,
            "detail_code": self.detail_code,
            "stdout_bytes": self.stdout_bytes,
            "stderr_bytes": self.stderr_bytes,
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "output_limit_bytes": LAUNCHCTL_OUTPUT_MAX_BYTES,
        }


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

    from quant_platform.active_pipeline import CommandResult

    checked_at = _as_utc(now)
    runtime_temp = runtime_temp_directory(root)
    system_launch_agents_dir = system_launch_agents_dir or (
        Path.home() / "Library" / "LaunchAgents"
    )
    checks: list[dict[str, str]] = []
    service_summaries: list[dict[str, Any]] = []
    desired_states, desired_state_explicit, desired_state_blocker = _load_desired_states(root)
    _add_check(
        checks,
        service="all",
        name="scheduler_desired_state_policy",
        passed=not desired_state_blocker,
        observed=(_compact(desired_states) if not desired_state_blocker else desired_state_blocker),
        expected="exact_service_set;LOADED_or_UNLOADED;zero_authority",
        evidence_path=str(root / DESIRED_STATE_PATH),
        blocker=desired_state_blocker or "scheduler_desired_state_invalid",
    )

    interpreter_observation = _interpreter_observation(root, runner)
    _add_check(
        checks,
        service="all",
        name="locked_interpreter_probe",
        passed=interpreter_observation["status"] == "PASS",
        observed=_compact(interpreter_observation),
        expected="canonical_executable;python>=3.11;dependency_check=PASS",
        evidence_path=str(scheduler_python_path(root)),
        blocker="locked_scheduler_interpreter_invalid",
    )
    for dependency_name in ("pyproject.toml", "uv.lock"):
        dependency_path = root / dependency_name
        _add_check(
            checks,
            service="all",
            name=f"{dependency_name}_present",
            passed=dependency_path.is_file(),
            observed="present" if dependency_path.is_file() else "missing",
            expected="present",
            evidence_path=str(dependency_path),
            blocker=f"scheduler_dependency_identity_missing:{dependency_name}",
        )

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
    log_policy_observation = _log_retention_policy_observation(root)
    _add_check(
        checks,
        service="all",
        name="canonical_log_retention_policy",
        passed=log_policy_observation["status"] == "PASS",
        observed=_compact(log_policy_observation),
        expected=(
            "schema=v1;canonical_glob;max_bytes=10485760;retention_days=14;"
            "file_mode=0600;symlinks=false;secret_content=false;rotations=1..32"
        ),
        evidence_path=str(root / RETENTION_POLICY_PATH),
        blocker=str(log_policy_observation["blocker"]),
    )
    log_directory_observation = _log_directory_observation(root)
    _add_check(
        checks,
        service="all",
        name="canonical_log_directory",
        passed=log_directory_observation["status"] == "PASS",
        observed=_compact(log_directory_observation),
        expected="regular_directory;not_symlink;mode=0700",
        evidence_path=str(scheduler_log_directory(root)),
        blocker=str(log_directory_observation["blocker"]),
    )

    operational_warnings: list[str] = []
    system_free_bytes: int | None = None
    try:
        system_free_bytes = int(disk_usage(Path.home()).free)
        if system_free_bytes < system_disk_warning_bytes:
            operational_warnings.append(
                f"system_volume_free_space_low:{system_free_bytes}<{system_disk_warning_bytes}"
            )
    except OSError as exc:
        operational_warnings.append(f"system_volume_free_space_unavailable:{type(exc).__name__}")

    for contract in SCHEDULER_CONTRACTS:
        service_checks_before = len(checks)
        expected = _expected_contract(root, contract)
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
        mirror_valid = bool(workspace_sha and system_sha and workspace_sha == system_sha)
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

        launchctl_probe = _probe_launchctl(contract, runner)
        live = launchctl_probe.parsed if launchctl_probe.status == "LOADED" else {}
        desired_state = desired_states[contract.key]
        observed_loaded = launchctl_probe.status == "LOADED"
        desired_state_matches = launchctl_probe.status == desired_state
        _add_check(
            checks,
            service=contract.key,
            name="launch_agent_desired_state",
            passed=desired_state_matches,
            observed=_compact(launchctl_probe.evidence()),
            expected=desired_state,
            evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
            blocker=_launchctl_policy_blocker(
                contract=contract,
                desired_state=desired_state,
                observed_status=launchctl_probe.status,
            ),
        )
        if desired_state == "LOADED" and launchctl_probe.status == "LOADED":
            _add_live_contract_checks(
                checks,
                contract=contract,
                live=live,
                expected=expected,
                system_path=system_path,
            )
        elif desired_state == "UNLOADED" and launchctl_probe.status == "UNLOADED":
            _add_check(
                checks,
                service=contract.key,
                name="live_contract_not_required_while_unloaded",
                passed=True,
                observed="UNLOADED",
                expected="UNLOADED",
                evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
                blocker=f"{contract.key}_unexpected_live_contract_requirement",
            )
        if desired_state_explicit and desired_state == "LOADED":
            terminal = _terminal_receipt_observation(
                root=root,
                contract=contract,
                checked_at=checked_at,
            )
            _add_check(
                checks,
                service=contract.key,
                name="fresh_terminal_pass_receipt",
                passed=terminal["status"] == "PASS",
                observed=_compact(terminal),
                expected="fresh_hash_valid_terminal_PASS_zero_authority",
                evidence_path=str(terminal["evidence_path"]),
                blocker=str(terminal.get("blocker", "terminal_receipt_invalid")),
            )
        live_last_exit = str(live.get("last exit code", ""))
        log_observation = _scheduler_log_observation(
            root=root,
            contract=contract,
            live_exit_safe=launchctl_probe.status == "LOADED"
            and live_last_exit in {"0", "(never exited)"},
        )
        for stream in ("stdout", "stderr"):
            stream_observation = log_observation[f"{stream}_log"]
            _add_check(
                checks,
                service=contract.key,
                name=f"{stream}_log_contract",
                passed=stream_observation["status"] == "PASS",
                observed=_compact(stream_observation),
                expected=(
                    "regular_file;not_symlink;mode=0600;bytes<=10485760;"
                    "redaction=PASS_NO_SECRET_PATTERNS"
                ),
                evidence_path=str(stream_observation["path"]),
                blocker=f"{contract.key}_{stream}_{stream_observation['blocker']}",
            )
        if log_observation["stderr_classification"] != "CLEAN":
            operational_warnings.append(
                f"{contract.key}_stderr_"
                f"{str(log_observation['stderr_classification']).lower()}:"
                f"bytes={log_observation['stderr_bytes']};"
                f"modified_at_utc={log_observation['stderr_modified_at_utc']}"
            )
        service_checks = checks[service_checks_before:]
        service_blockers = [row["blocker"] for row in service_checks if row["status"] == "BLOCKED"]
        service_summaries.append(
            {
                "service": contract.key,
                "label": contract.label,
                "desired_state": desired_state,
                "observed_loaded": observed_loaded,
                "launchctl_status": launchctl_probe.status,
                "launchctl_probe": launchctl_probe.evidence(),
                "status": "PASS" if not service_blockers else "BLOCKED",
                "checks_total": len(service_checks),
                "checks_passed": sum(row["status"] == "PASS" for row in service_checks),
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
        "desired_state_path": str(root / DESIRED_STATE_PATH),
        "desired_state_explicit": desired_state_explicit,
        "desired_states": desired_states,
        "runtime_temp_ready": temp_ready,
        "runtime_temp_free_bytes": runtime_free_bytes,
        "minimum_runtime_temp_free_bytes": minimum_runtime_temp_free_bytes,
        "system_volume_free_bytes": system_free_bytes,
        "system_disk_warning_bytes": system_disk_warning_bytes,
        "operational_warnings": operational_warnings,
        "log_policy_observation": log_policy_observation,
        "log_directory_observation": log_directory_observation,
        "interpreter_observation": interpreter_observation,
        "runtime_contracts": [
            scheduler_runtime_contract(root, contract=contract) for contract in SCHEDULER_CONTRACTS
        ],
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
        "schedulerruntime_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
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
    payload["evidence_paths"] = {key: _relative(path, root) for key, path in paths.items()}
    return CommandResult(paths=paths, summary=payload)


def _load_desired_states(
    root: Path,
) -> tuple[dict[str, str], bool, str]:
    expected_keys = {contract.key for contract in SCHEDULER_CONTRACTS}
    default = {key: "LOADED" for key in expected_keys}
    path = root / DESIRED_STATE_PATH
    if not path.exists():
        return default, False, ""
    if path.is_symlink() or not path.is_file():
        return default, True, "scheduler_desired_state_not_regular"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return default, True, "scheduler_desired_state_unreadable"
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "generation",
        "services",
        "reason",
        "reload_requires_separate_operator_approval",
        "promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    }:
        return default, True, "scheduler_desired_state_schema_fields_invalid"
    if payload.get("schema_version") != DESIRED_STATE_SCHEMA_VERSION:
        return default, True, "scheduler_desired_state_schema_version_invalid"
    generation = payload.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        return default, True, "scheduler_desired_state_generation_invalid"
    services = payload.get("services")
    if not isinstance(services, dict) or set(services) != expected_keys:
        return default, True, "scheduler_desired_state_service_set_invalid"
    normalized = {str(key): str(value).strip().upper() for key, value in services.items()}
    if any(value not in {"LOADED", "UNLOADED"} for value in normalized.values()):
        return default, True, "scheduler_desired_state_value_invalid"
    if (
        payload.get("reload_requires_separate_operator_approval") is not True
        or payload.get("promotion_authority") is not False
        or payload.get("testnet_order_authority") is not False
        or payload.get("live_trading_authorized") is not False
    ):
        return default, True, "scheduler_desired_state_authority_invalid"
    if not str(payload.get("reason", "")).strip():
        return default, True, "scheduler_desired_state_reason_missing"
    return normalized, True, ""


def _terminal_receipt_observation(
    *,
    root: Path,
    contract: SchedulerContract,
    checked_at: datetime,
) -> dict[str, Any]:
    pointer_path = (
        root / "reports" / "active" / "scheduler_terminal_receipts" / f"{contract.key}_latest.json"
    )
    result: dict[str, Any] = {
        "status": "BLOCKED",
        "blocker": f"{contract.key}_terminal_pointer_missing",
        "evidence_path": str(pointer_path),
    }
    current_runtime = scheduler_runtime_contract(root, contract=contract)
    expected_runtime_identity = {
        "capability_profile": contract.capability_profile,
        **{
            field: current_runtime[field]
            for field in (
                "capability_profile_sha256",
                "runtime_contract_sha256",
                "source_fingerprint_sha256",
                "configuration_fingerprint_sha256",
                "dependency_fingerprint_sha256",
                "interpreter_fingerprint_sha256",
                "schedule_fingerprint_sha256",
            )
        },
    }
    try:
        receipt, receipt_path = load_validated_scheduler_terminal_receipt(
            root,
            scheduler_key=contract.key,
            expected_runtime_identity=expected_runtime_identity,
            require_launchd=True,
        )
    except (OSError, ValueError) as exc:
        token = str(exc).split(":", 1)[0]
        return {**result, "blocker": f"{contract.key}_{token}"}
    if (
        not isinstance(receipt, dict)
        or receipt.get("scheduler_key") != contract.key
        or receipt.get("terminal_status") != "PASS"
        or receipt.get("intended_slot_credit") is not True
        or receipt.get("authority_advanced") is not False
        or receipt.get("promotion_authority") is not False
        or receipt.get("live_trading_authorized") is not False
        or receipt.get("order_submissions") != 0
        or receipt.get("blockers") != []
    ):
        return {
            **result,
            "blocker": f"{contract.key}_terminal_receipt_semantics_invalid",
            "evidence_path": str(receipt_path),
        }
    completed = _parse_timestamp(receipt.get("completed_at_utc"))
    if completed is None:
        return {
            **result,
            "blocker": f"{contract.key}_terminal_receipt_time_invalid",
            "evidence_path": str(receipt_path),
        }
    maximum_age_seconds = (
        contract.interval_seconds * 2 if contract.interval_seconds is not None else 36 * 60 * 60
    )
    age_seconds = (checked_at - completed).total_seconds()
    if age_seconds < 0 or age_seconds > maximum_age_seconds:
        return {
            **result,
            "blocker": f"{contract.key}_terminal_receipt_stale",
            "evidence_path": str(receipt_path),
            "age_seconds": age_seconds,
            "maximum_age_seconds": maximum_age_seconds,
        }
    return {
        "status": "PASS",
        "blocker": "",
        "evidence_path": str(receipt_path),
        "receipt_id": receipt["receipt_id"],
        "age_seconds": age_seconds,
        "maximum_age_seconds": maximum_age_seconds,
    }


def _parse_timestamp(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _log_retention_policy_observation(root: Path) -> dict[str, Any]:
    path = root / RETENTION_POLICY_PATH
    base: dict[str, Any] = {
        "status": "BLOCKED",
        "blocker": "scheduler_log_retention_policy_missing",
        "path": str(path),
        "policy_sha256": "",
        "schema_version": "",
        "log_paths": [],
        "log_rotation_threshold_bytes": None,
        "log_retention_days": None,
        "log_rotations_to_keep": None,
        "log_file_mode": "",
        "log_symlinks_allowed": None,
        "log_secret_content_allowed": None,
        "violations": [],
    }
    try:
        metadata = path.lstat()
    except OSError:
        return base
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        return {
            **base,
            "blocker": "scheduler_log_retention_policy_not_regular",
        }
    try:
        encoded = path.read_bytes()
    except OSError:
        return {
            **base,
            "blocker": "scheduler_log_retention_policy_unreadable",
        }
    if len(encoded) > 64 * 1024:
        return {
            **base,
            "blocker": "scheduler_log_retention_policy_too_large",
        }
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {
            **base,
            "blocker": "scheduler_log_retention_policy_invalid_json",
            "policy_sha256": sha256(encoded).hexdigest(),
        }
    if not isinstance(payload, dict):
        return {
            **base,
            "blocker": "scheduler_log_retention_policy_not_object",
            "policy_sha256": sha256(encoded).hexdigest(),
        }
    rotations = payload.get("log_rotations_to_keep")
    violations: list[str] = []
    if payload.get("schema_version") != RETENTION_POLICY_SCHEMA_VERSION:
        violations.append("schema_version")
    if payload.get("log_paths") != [SCHEDULER_LOG_GLOB]:
        violations.append("log_paths")
    if payload.get("log_rotation_threshold_bytes") != SCHEDULER_LOG_MAX_BYTES:
        violations.append("log_rotation_threshold_bytes")
    if payload.get("log_retention_days") != SCHEDULER_LOG_RETENTION_DAYS:
        violations.append("log_retention_days")
    if isinstance(rotations, bool) or not isinstance(rotations, int) or not 1 <= rotations <= 32:
        violations.append("log_rotations_to_keep")
    if payload.get("log_file_mode") != "0600":
        violations.append("log_file_mode")
    if payload.get("log_symlinks_allowed") is not False:
        violations.append("log_symlinks_allowed")
    if payload.get("log_secret_content_allowed") is not False:
        violations.append("log_secret_content_allowed")
    return {
        **base,
        "status": "PASS" if not violations else "BLOCKED",
        "blocker": "" if not violations else "scheduler_log_retention_policy_invalid",
        "policy_sha256": sha256(encoded).hexdigest(),
        "schema_version": str(payload.get("schema_version", "")),
        "log_paths": payload.get("log_paths", []),
        "log_rotation_threshold_bytes": payload.get("log_rotation_threshold_bytes"),
        "log_retention_days": payload.get("log_retention_days"),
        "log_rotations_to_keep": rotations,
        "log_file_mode": str(payload.get("log_file_mode", "")),
        "log_symlinks_allowed": payload.get("log_symlinks_allowed"),
        "log_secret_content_allowed": payload.get("log_secret_content_allowed"),
        "violations": violations,
    }


def _log_directory_observation(root: Path) -> dict[str, Any]:
    path = scheduler_log_directory(root)
    base: dict[str, Any] = {
        "status": "BLOCKED",
        "blocker": "scheduler_log_directory_missing",
        "path": str(path),
        "regular_directory": False,
        "symlink": False,
        "mode": "",
    }
    try:
        metadata = path.lstat()
    except OSError:
        return base
    symlink = stat.S_ISLNK(metadata.st_mode)
    regular_directory = stat.S_ISDIR(metadata.st_mode)
    mode = stat.S_IMODE(metadata.st_mode)
    violations: list[str] = []
    if symlink:
        violations.append("symlink_not_allowed")
    if not regular_directory:
        violations.append("not_directory")
    if mode != SCHEDULER_LOG_DIRECTORY_MODE:
        violations.append("mode_invalid")
    return {
        **base,
        "status": "PASS" if not violations else "BLOCKED",
        "blocker": "" if not violations else "scheduler_log_directory_invalid",
        "regular_directory": regular_directory,
        "symlink": symlink,
        "mode": f"{mode:04o}",
        "violations": violations,
    }


def _scheduler_log_observation(
    *, root: Path, contract: SchedulerContract, live_exit_safe: bool
) -> dict[str, Any]:
    """Inspect log metadata and redaction without publishing log contents."""

    logs = scheduler_log_directory(root)
    stdout_path = logs / contract.stdout_name
    stderr_path = logs / contract.stderr_name
    stdout = _scheduler_log_file_observation(stdout_path)
    stderr = _scheduler_log_file_observation(stderr_path)
    stdout_bytes = int(stdout["bytes"])
    stderr_bytes = int(stderr["bytes"])
    stdout_modified = _parse_timestamp(stdout["modified_at_utc"])
    stderr_modified = _parse_timestamp(stderr["modified_at_utc"])
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
        "stdout_log": stdout,
        "stderr_path": str(stderr_path),
        "stderr_bytes": stderr_bytes,
        "stderr_modified_at_utc": _timestamp_text(stderr_modified),
        "stderr_log": stderr,
        "stderr_classification": classification,
    }


def _scheduler_log_file_observation(path: Path) -> dict[str, Any]:
    base: dict[str, Any] = {
        "status": "BLOCKED",
        "blocker": "log_missing",
        "path": str(path),
        "regular_file": False,
        "symlink": False,
        "mode": "",
        "bytes": 0,
        "maximum_bytes": SCHEDULER_LOG_MAX_BYTES,
        "modified_at_utc": "",
        "content_sha256": "",
        "redaction_status": "BLOCKED_UNVERIFIED",
        "secret_pattern_match_count": 0,
        "scan_limit_bytes": SCHEDULER_LOG_MAX_BYTES,
        "violations": [],
    }
    try:
        metadata = path.lstat()
    except OSError:
        return base
    symlink = stat.S_ISLNK(metadata.st_mode)
    regular_file = stat.S_ISREG(metadata.st_mode)
    mode = stat.S_IMODE(metadata.st_mode)
    size = int(metadata.st_size)
    violations: list[str] = []
    if symlink:
        violations.append("log_symlink_not_allowed")
    if not regular_file:
        violations.append("log_not_regular")
    if mode != SCHEDULER_LOG_FILE_MODE:
        violations.append("log_mode_invalid")
    if size > SCHEDULER_LOG_MAX_BYTES:
        violations.append("log_size_exceeded")
    digest = ""
    redaction_status = "BLOCKED_UNVERIFIED"
    secret_matches = 0
    if regular_file and not symlink and size <= SCHEDULER_LOG_MAX_BYTES:
        try:
            content = path.read_bytes()
        except OSError:
            violations.append("log_unreadable")
            redaction_status = "BLOCKED_UNREADABLE"
        else:
            digest = sha256(content).hexdigest()
            secret_matches = sum(1 for pattern in SECRET_LOG_PATTERNS if pattern.search(content))
            if secret_matches:
                violations.append("log_secret_pattern_detected")
                redaction_status = "BLOCKED_SECRET_PATTERN_DETECTED"
            else:
                redaction_status = "PASS_NO_SECRET_PATTERNS"
    elif size > SCHEDULER_LOG_MAX_BYTES:
        redaction_status = "BLOCKED_SIZE_EXCEEDED_UNSCANNED"
    return {
        **base,
        "status": "PASS" if not violations else "BLOCKED",
        "blocker": "" if not violations else violations[0],
        "regular_file": regular_file,
        "symlink": symlink,
        "mode": f"{mode:04o}",
        "bytes": size,
        "modified_at_utc": datetime.fromtimestamp(
            metadata.st_mtime,
            tz=UTC,
        ).isoformat(),
        "content_sha256": digest,
        "redaction_status": redaction_status,
        "secret_pattern_match_count": secret_matches,
        "violations": violations,
    }


def _timestamp_text(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _expected_contract(
    root: Path,
    contract: SchedulerContract,
) -> dict[str, Any]:
    payload = scheduler_launch_agent_payload(root, contract=contract)
    expected: dict[str, Any] = {
        "label": payload["Label"],
        "arguments": payload["ProgramArguments"],
        "working_directory": payload["WorkingDirectory"],
        "environment": payload["EnvironmentVariables"],
        "stdout_path": payload["StandardOutPath"],
        "stderr_path": payload["StandardErrorPath"],
        "run_at_load": payload["RunAtLoad"],
        "process_type": payload.get("ProcessType"),
        "low_priority_io": payload.get("LowPriorityIO", False),
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
        ("static_run_at_load", payload.get("RunAtLoad"), expected["run_at_load"]),
        ("static_process_type", payload.get("ProcessType"), expected["process_type"]),
        (
            "static_low_priority_io",
            payload.get("LowPriorityIO", False),
            expected["low_priority_io"],
        ),
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
    expected: dict[str, Any],
    system_path: Path,
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
            passed=observed == wanted,
            observed=_compact(observed),
            expected=_compact(wanted),
            evidence_path=f"launchctl:gui/{os.getuid()}/{contract.label}",
            blocker=f"{contract.key}_{name}_mismatch",
        )
    if contract.interval_seconds is not None:
        observed_schedule = _safe_int(live.get("run interval"), default=-1)
        expected_schedule = contract.interval_seconds
        schedule_valid = observed_schedule == expected_schedule
    else:
        observed_schedule = live.get("calendar", {})
        expected_schedule = expected["calendar"]
        schedule_valid = observed_schedule == expected_schedule
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
    last_exit_safe = last_exit in {"0", "(never exited)"}
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


def _interpreter_observation(
    root: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    python = scheduler_python_path(root)
    if not python.is_file():
        return {
            "status": "BLOCKED",
            "blocker": "canonical_interpreter_missing",
            "expected_executable": str(python),
        }
    script = (
        "import importlib.metadata as m,json,sys;"
        "packages=sorted((d.metadata.get('Name',''),d.version) "
        "for d in m.distributions());"
        "print(json.dumps({'executable':sys.executable,"
        "'version_info':list(sys.version_info[:3]),'packages':packages},"
        "sort_keys=True))"
    )
    try:
        probe = runner(
            [str(python), "-I", "-c", script],
            capture_output=True,
            text=True,
            check=False,
            timeout=SCHEDULER_PROBE_TIMEOUT_SECONDS,
        )
        dependency = runner(
            ["uv", "pip", "check", "--python", str(python)],
            capture_output=True,
            text=True,
            check=False,
            timeout=SCHEDULER_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "BLOCKED",
            "blocker": "interpreter_or_dependency_probe_timeout",
            "expected_executable": str(python),
        }
    except OSError as exc:
        return {
            "status": "BLOCKED",
            "blocker": f"interpreter_probe_error:{type(exc).__name__}",
            "expected_executable": str(python),
        }
    if probe.returncode != 0:
        return {
            "status": "BLOCKED",
            "blocker": f"interpreter_probe_returncode:{probe.returncode}",
            "expected_executable": str(python),
        }
    if len((probe.stdout or "").encode("utf-8")) > 2 * 1024 * 1024:
        return {
            "status": "BLOCKED",
            "blocker": "interpreter_probe_output_too_large",
            "expected_executable": str(python),
        }
    try:
        payload = json.loads(probe.stdout or "{}")
        version = tuple(int(value) for value in payload.get("version_info", []))
        executable_matches = Path(str(payload.get("executable", ""))).resolve() == python.resolve()
        package_fingerprint = sha256(
            _canonical_json(payload.get("packages", [])).encode("utf-8")
        ).hexdigest()
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {
            "status": "BLOCKED",
            "blocker": "interpreter_probe_payload_invalid",
            "expected_executable": str(python),
        }
    blockers: list[str] = []
    if not executable_matches:
        blockers.append("interpreter_executable_mismatch")
    if version < (3, 11):
        blockers.append("interpreter_python_version_unsupported")
    if dependency.returncode != 0:
        blockers.append(f"uv_dependency_check_returncode:{dependency.returncode}")
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "expected_executable": str(python),
        "resolved_executable": str(python.resolve()),
        "observed_executable": str(payload.get("executable", "")),
        "python_version": ".".join(str(value) for value in version),
        "package_count": len(payload.get("packages", [])),
        "package_inventory_sha256": package_fingerprint,
        "dependency_check_status": "PASS" if dependency.returncode == 0 else "BLOCKED",
        "dependency_check_output_sha256": sha256(
            ((dependency.stdout or "") + (dependency.stderr or "")).encode("utf-8")
        ).hexdigest(),
    }


def _probe_launchctl(
    contract: SchedulerContract,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> LaunchctlProbe:
    command = [
        "launchctl",
        "print",
        f"gui/{os.getuid()}/{contract.label}",
    ]
    try:
        completed = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=SCHEDULER_PROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        return _launchctl_probe(
            status="TIMEOUT",
            return_code=None,
            detail_code="launchctl_timeout",
            stdout=_subprocess_text(exc.output),
            stderr=_subprocess_text(exc.stderr),
        )
    except OSError as exc:
        return _launchctl_probe(
            status="ERROR",
            return_code=None,
            detail_code=f"launchctl_os_error_{type(exc).__name__}"[:128],
            stdout="",
            stderr="",
        )
    stdout = _subprocess_text(completed.stdout)
    stderr = _subprocess_text(completed.stderr)
    if max(_encoded_size(stdout), _encoded_size(stderr)) > LAUNCHCTL_OUTPUT_MAX_BYTES:
        return _launchctl_probe(
            status="UNSUPPORTED",
            return_code=completed.returncode,
            detail_code="launchctl_output_too_large",
            stdout=stdout,
            stderr=stderr,
        )
    if completed.returncode != 0:
        status: LaunchctlStatus = (
            "UNLOADED"
            if _is_exact_launchctl_unloaded(
                return_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )
            else "ERROR"
        )
        return _launchctl_probe(
            status=status,
            return_code=completed.returncode,
            detail_code=(
                "launchctl_service_not_found"
                if status == "UNLOADED"
                else f"launchctl_nonzero_exit_{completed.returncode}"[:128]
            ),
            stdout=stdout,
            stderr=stderr,
        )
    parsed = _parse_launchctl_print(stdout)
    if not _launchctl_payload_supported(
        payload=stdout,
        parsed=parsed,
        contract=contract,
    ):
        return _launchctl_probe(
            status="UNSUPPORTED",
            return_code=completed.returncode,
            detail_code="launchctl_format_unsupported",
            stdout=stdout,
            stderr=stderr,
        )
    return _launchctl_probe(
        status="LOADED",
        return_code=completed.returncode,
        detail_code="launchctl_loaded_supported",
        stdout=stdout,
        stderr=stderr,
        parsed=parsed,
    )


def _launchctl_probe(
    *,
    status: LaunchctlStatus,
    return_code: int | None,
    detail_code: str,
    stdout: str,
    stderr: str,
    parsed: dict[str, Any] | None = None,
) -> LaunchctlProbe:
    stdout_bytes = stdout.encode("utf-8", errors="replace")
    stderr_bytes = stderr.encode("utf-8", errors="replace")
    return LaunchctlProbe(
        status=status,
        return_code=return_code,
        detail_code=detail_code[:128],
        stdout_bytes=len(stdout_bytes),
        stderr_bytes=len(stderr_bytes),
        stdout_sha256=sha256(stdout_bytes).hexdigest(),
        stderr_sha256=sha256(stderr_bytes).hexdigest(),
        parsed={} if parsed is None else parsed,
    )


def _subprocess_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else ""


def _encoded_size(value: str) -> int:
    return len(value.encode("utf-8", errors="replace"))


def _is_exact_launchctl_unloaded(
    *,
    return_code: int,
    stdout: str,
    stderr: str,
) -> bool:
    if return_code != 113:
        return False
    normalized_stdout = " ".join(stdout.lower().split())
    normalized_stderr = " ".join(stderr.lower().split())
    if normalized_stdout not in {"", "bad request."}:
        return False
    return "could not find service" in normalized_stderr or normalized_stderr in {
        "service not found",
        "bad request. service not found",
    }


def _launchctl_payload_supported(
    *,
    payload: str,
    parsed: dict[str, Any],
    contract: SchedulerContract,
) -> bool:
    first_line = next((line.strip() for line in payload.splitlines() if line.strip()), "")
    if first_line != f"gui/{os.getuid()}/{contract.label} = {{":
        return False
    required_scalars = (
        "path",
        "state",
        "program",
        "working directory",
        "stdout path",
        "stderr path",
        "runs",
        "last exit code",
    )
    if any(not str(parsed.get(name, "")).strip() for name in required_scalars):
        return False
    if not isinstance(parsed.get("arguments"), list) or len(parsed["arguments"]) < 4:
        return False
    if not isinstance(parsed.get("environment"), dict) or not parsed["environment"]:
        return False
    if contract.interval_seconds is not None:
        return isinstance(parsed.get("run interval"), int)
    calendar = parsed.get("calendar")
    return isinstance(calendar, dict) and set(calendar) == {"Hour", "Minute"}


def _launchctl_policy_blocker(
    *,
    contract: SchedulerContract,
    desired_state: str,
    observed_status: LaunchctlStatus,
) -> str:
    if observed_status == desired_state:
        return ""
    if observed_status == "TIMEOUT":
        return f"{contract.key}_launchctl_timeout"
    if observed_status == "UNSUPPORTED":
        return f"{contract.key}_launchctl_unsupported"
    if observed_status == "ERROR":
        return f"{contract.key}_launchctl_error"
    if observed_status == "UNLOADED":
        return f"{contract.key}_launch_agent_not_loaded"
    if observed_status == "LOADED":
        return f"{contract.key}_launch_agent_unexpectedly_loaded"
    return f"{contract.key}_launchctl_state_mismatch"


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
    promote_staged_file(temporary, path)


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
    promote_staged_file(temporary, path)


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
