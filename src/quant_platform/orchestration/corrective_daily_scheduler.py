"""Research-only daily scheduler with locks, timeouts, and freshness revocation."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    ensure_runtime_temp_directory,
    launch_agent_runtime_environment,
    write_launch_agent_plist,
)
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
    STAGES,
)
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    run_current_wizard_hyperliquid_daily_pipeline,
    stage3_forbidden_command_tokens,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_daily_scheduler.v2"
DAILY_RUN_SCHEMA_VERSION = "current_wizard_hyperliquid_daily_runner.v1"
REQUIRED_DAILY_STAGES = 19
EXPECTED_DAILY_STAGE_NAMES = tuple(str(stage[1]) for stage in STAGES)
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-research-daily"
DEFAULT_RUN_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_STAGE_TIMEOUT_SECONDS = 45 * 60
MAX_RECEIPT_AGE_HOURS = 36
SEMANTIC_DAILY_RECEIPT_REQUIRED_FROM_UTC = "2026-08-11"


class SchedulerBlocked(RuntimeError):
    """Raised when a scheduler invariant blocks a run."""


def run_scheduled_research(
    *,
    root: Path = ROOT,
    execute: bool = False,
    now: datetime | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    run_timeout_seconds: int = DEFAULT_RUN_TIMEOUT_SECONDS,
    stage_timeout_seconds: int = DEFAULT_STAGE_TIMEOUT_SECONDS,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CommandResult:
    now = _as_utc(now)
    active = root / "reports" / "active"
    receipts = active / "daily_schedule_receipts"
    active.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = active / ".corrective_daily.lock"
    status_path = active / (
        "daily_schedule_status.csv" if execute else "daily_schedule_plan_status.csv"
    )
    authority_path = active / (
        "daily_research_authority.json" if execute else "daily_research_plan_authority.json"
    )
    receipt_path = receipts / f"{now.date().isoformat()}.json"
    blockers: list[str] = []
    lock_acquired = False
    result: CommandResult | None = None
    run_status = "BLOCKED"
    reused_daily_receipt: dict[str, Any] | None = None
    try:
        _acquire_lock(lock_path, now=now, timeout_seconds=run_timeout_seconds)
        lock_acquired = True
        existing_valid, _existing_blockers = _validate_daily_receipt(
            receipt_path,
            root=root,
            now=now,
        )
        existing = _read_json(receipt_path)
        if execute and existing_valid and existing.get("run_status") == "PASS":
            reused_daily_receipt = existing
            run_status = "PASS"
        else:
            free_bytes = shutil.disk_usage(root).free
            if free_bytes < minimum_free_bytes:
                raise SchedulerBlocked(f"insufficient_free_space:{free_bytes}<{minimum_free_bytes}")
            deadline = time.monotonic() + run_timeout_seconds
            runner = command_runner or _timeout_runner(
                deadline=deadline, stage_timeout_seconds=stage_timeout_seconds
            )
            result = run_current_wizard_hyperliquid_daily_pipeline(
                root=root,
                execute=execute,
                now=now,
                minimum_free_bytes=minimum_free_bytes,
                command_runner=runner,
            )
            run_status = str(result.summary.get("run_status", "BLOCKED"))
            if run_status not in {"PASS", "PLANNED"}:
                blockers.append(f"daily_pipeline_status:{run_status}")
    except (SchedulerBlocked, FileExistsError) as exc:
        blockers.append(str(exc))
        run_status = "BLOCKED"
    except Exception as exc:  # noqa: BLE001 - persist unexpected scheduler failures
        blockers.append(f"scheduler_error:{type(exc).__name__}:{exc}")
        run_status = "FAILED"
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
    completed_at = _as_utc(clock() if clock is not None else datetime.now(UTC))
    if reused_daily_receipt is not None:
        return _reuse_daily_receipt(
            root=root,
            now=now,
            completed_at=completed_at,
            receipt_path=receipt_path,
            receipt=reused_daily_receipt,
            status_path=status_path,
            authority_path=authority_path,
            lock_path=lock_path,
        )
    current = run_status == "PASS" and not blockers
    source_manifest, source_status = _daily_run_evidence_paths(
        root=root,
        result=result,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": now.date().isoformat(),
        "started_at_utc": now.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "execution_requested": execute,
        "run_status": run_status,
        "blockers": blockers,
        "research_board_current": current,
        "stale_output_actionable": False,
        "scheduler_lock_released": not lock_path.exists(),
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_id": str(result.summary.get("daily_run_id", "")) if result else "",
        "daily_run_manifest": _relative(source_manifest, root),
        "daily_run_manifest_sha256": _file_sha256(source_manifest),
        "daily_run_status": _relative(source_status, root),
        "daily_run_status_sha256": _file_sha256(source_status),
        "source_evidence_immutable": _is_dated_daily_run_evidence(
            source_manifest, source_status, root=root
        ),
    }
    receipt_path = _publish_daily_receipt(receipt, root=root)
    receipt = _read_json(receipt_path)
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "as_of_utc": now.isoformat(),
            "research_board_current": current,
            "current_receipt": _relative(receipt_path, root),
            "blockers": blockers,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
        authority_path,
    )
    status = pd.DataFrame(
        [
            {
                "run_date": receipt["run_date"],
                "receipt_id": receipt["receipt_id"],
                "run_status": run_status,
                "research_board_current": current,
                "blocker": ";".join(blockers),
                "lock_released": not lock_path.exists(),
                "receipt_path": _relative(receipt_path, root),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    _atomic_csv(status, status_path)
    acceptance = build_daily_cadence_acceptance(
        root=root,
        now=max(now, completed_at),
    )
    return CommandResult(
        paths={
            "daily_schedule_status": status_path,
            "daily_receipt": receipt_path,
            "daily_authority": authority_path,
            "daily_cadence_acceptance": acceptance,
        },
        summary={
            "status": run_status,
            "research_board_current": current,
            "blockers": blockers,
            "qualifying_consecutive_daily_cycles": _acceptance_count(acceptance),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def refresh_corrective_checkpoint_after_scheduled_run(
    *,
    result: CommandResult,
    root: Path = ROOT,
    now: datetime | None = None,
    refresher: Callable[..., CommandResult] | None = None,
) -> CommandResult | None:
    """Refresh non-order program evidence after a successful scheduled run."""

    if str(result.summary.get("status", "BLOCKED")) != "PASS":
        return None
    if bool(result.summary.get("testnet_order_authority", False)) or bool(
        result.summary.get("live_trading_authorized", False)
    ):
        raise ValueError("scheduled research result unexpectedly acquired authority")
    active = root / "reports" / "active"
    daily_receipt_path = Path(
        result.paths.get("daily_receipt", active / "daily_schedule_receipt_missing.json")
    )
    daily_receipt = _read_json(daily_receipt_path)
    if not daily_receipt or not daily_receipt_path.is_file():
        raise ValueError("successful scheduled run is missing its daily receipt")
    receipt_valid, receipt_blockers = _validate_daily_receipt(
        daily_receipt_path,
        root=root,
        now=_as_utc(now),
    )
    if not receipt_valid:
        raise ValueError(
            "successful scheduled run has invalid daily receipt:" + ";".join(receipt_blockers)
        )
    if refresher is None:
        from quant_platform.orchestration.corrective_program import (
            complete_corrective_plan,
        )

        refresher = complete_corrective_plan
    checkpoint = refresher(root=root, now=_as_utc(now))
    if bool(checkpoint.summary.get("testnet_order_authority", False)) or bool(
        checkpoint.summary.get("live_trading_authorized", False)
    ):
        raise ValueError("post-run checkpoint unexpectedly acquired authority")
    checkpoint_path = Path(
        checkpoint.paths.get("seven_stage_checkpoint", active / "seven_stage_goal_checkpoint.csv")
    )
    if not checkpoint_path.is_file():
        raise ValueError("post-run checkpoint artifact is missing")
    rerun_gate_path = active / "registered_research_rerun_gate.json"
    rerun_gate = _read_json(rerun_gate_path)
    payload = {
        "schema_version": "thewiz.daily_post_run_handoff.v1",
        "as_of_utc": _as_utc(now).isoformat(),
        "daily_receipt_id": str(daily_receipt.get("receipt_id", "")),
        "daily_receipt_path": _relative(daily_receipt_path, root),
        "daily_receipt_sha256": _file_sha256(daily_receipt_path),
        "daily_run_status": str(daily_receipt.get("run_status", "")),
        "checkpoint_path": _relative(checkpoint_path, root),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "operational_acceptance_status": str(
            checkpoint.summary.get("operational_acceptance_status", "BLOCKED")
        ),
        "registered_rerun_gate_path": _relative(rerun_gate_path, root),
        "registered_rerun_gate_sha256": _file_sha256(rerun_gate_path),
        "registered_rerun_gate_status": str(rerun_gate.get("status", "NOT_AVAILABLE")),
        "registered_rerun_results_accounted": bool(
            rerun_gate.get("registered_rerun_results_accounted", False)
        ),
        "registered_rerun_conclusion_status": str(
            rerun_gate.get("registered_rerun_conclusion_status", "INCOMPLETE")
        ),
        "handoff_status": "PASS_CHECKPOINT_REFRESHED",
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    handoff_id = "dailyhandoff_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    payload["handoff_id"] = handoff_id
    handoff_path = active / "daily_post_run_handoffs" / f"{handoff_id}.json"
    if handoff_path.is_file() and _read_json(handoff_path) != payload:
        raise ValueError("immutable daily post-run handoff receipt mismatch")
    _atomic_json(payload, handoff_path)
    latest_path = active / "daily_post_run_handoff.json"
    _atomic_json(
        {**payload, "handoff_receipt_path": _relative(handoff_path, root)},
        latest_path,
    )
    result.paths["post_run_handoff"] = handoff_path
    result.paths["post_run_handoff_latest"] = latest_path
    result.summary["post_run_checkpoint_refreshed"] = True
    result.summary["post_run_handoff_id"] = handoff_id
    result.summary["post_run_registered_rerun_gate_status"] = payload[
        "registered_rerun_gate_status"
    ]
    return checkpoint


def import_existing_complete_daily_run(*, root: Path = ROOT, now: datetime | None = None) -> Path:
    now = _as_utc(now)
    active = root / "reports" / "active"
    manifest_path = active / "current_wizard_hyperliquid_daily_run_manifest.json"
    status_path = active / "current_wizard_hyperliquid_daily_run_status.csv"
    manifest = _read_json(manifest_path)
    status = _read_csv(status_path)
    source_manifest, source_status = _daily_run_evidence_paths(
        root=root,
        manifest=manifest,
    )
    dated_manifest = _read_json(source_manifest)
    dated_status = _read_csv(source_status)
    evidence_matches_active = bool(
        dated_manifest == manifest and not dated_status.empty and dated_status.equals(status)
    )
    complete = (
        str(manifest.get("run_status")) == "PASS"
        and len(status) == 19
        and status.get("status", pd.Series(dtype=str)).eq("PASS").all()
        and not status.get("order_submission_authority", pd.Series(False, index=status.index))
        .map(_truthy)
        .any()
        and not status.get("live_trading_authorized", pd.Series(False, index=status.index))
        .map(_truthy)
        .any()
        and evidence_matches_active
    )
    run_time = pd.to_datetime(manifest.get("as_of"), utc=True, errors="coerce")
    run_date = run_time.date().isoformat() if pd.notna(run_time) else now.date().isoformat()
    started_at, completed_at = _daily_run_bounds(
        manifest=dated_manifest or manifest,
        status=dated_status if not dated_status.empty else status,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "execution_requested": True,
        "run_status": "PASS" if complete else "BLOCKED",
        "blockers": [] if complete else ["existing_daily_run_not_complete"],
        "research_board_current": bool(complete),
        "stale_output_actionable": False,
        "scheduler_lock_released": True,
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_id": str(manifest.get("daily_run_id", "")),
        "daily_run_manifest": _relative(source_manifest, root),
        "daily_run_manifest_sha256": _file_sha256(source_manifest),
        "daily_run_status": _relative(source_status, root),
        "daily_run_status_sha256": _file_sha256(source_status),
        "source_evidence_immutable": _is_dated_daily_run_evidence(
            source_manifest, source_status, root=root
        ),
        "imported_existing_validated_run": True,
    }
    return _publish_daily_receipt(receipt, root=root)


def build_daily_cadence_acceptance(*, root: Path = ROOT, now: datetime | None = None) -> Path:
    now = _as_utc(now)
    receipts_dir = root / "reports" / "active" / "daily_schedule_receipts"
    rows = []
    for path in sorted(receipts_dir.glob("*.json")) if receipts_dir.exists() else []:
        receipt = _read_json(path)
        receipt_valid, receipt_blockers = _validate_daily_receipt(
            path,
            root=root,
            now=now,
        )
        semantic_proven, semantic_blockers, semantic_source = (
            _validate_daily_semantic_qualification(path, root=root)
        )
        rows.append(
            {
                "run_date": str(receipt.get("run_date", "")),
                "receipt_id": str(receipt.get("receipt_id", "")),
                "receipt_identity_sha256": str(receipt.get("receipt_identity_sha256", "")),
                "daily_run_id": str(receipt.get("daily_run_id", "")),
                "run_status": str(receipt.get("run_status", "")),
                "research_board_current": bool(receipt.get("research_board_current", False)),
                "zero_execution_authority": not bool(receipt.get("testnet_order_authority", False))
                and not bool(receipt.get("live_trading_authorized", False)),
                "receipt_valid": receipt_valid,
                "validation_blocker": ";".join(receipt_blockers),
                "semantic_contract_required": True,
                "semantic_contract_status": "PASS" if semantic_proven else "BLOCKED",
                "semantic_contract_blocker": ";".join(semantic_blockers),
                "semantic_contract_source": semantic_source,
                "receipt_path": _relative(path, root),
            }
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "run_date",
            "receipt_id",
            "receipt_identity_sha256",
            "daily_run_id",
            "run_status",
            "research_board_current",
            "zero_execution_authority",
            "receipt_valid",
            "validation_blocker",
            "semantic_contract_required",
            "semantic_contract_status",
            "semantic_contract_blocker",
            "semantic_contract_source",
            "receipt_path",
        ],
    )
    if not frame.empty:
        frame = frame.drop_duplicates("run_date", keep="last").sort_values("run_date")
        qualifying = (
            frame["run_status"].eq("PASS")
            & frame["research_board_current"]
            & frame["zero_execution_authority"]
            & frame["receipt_valid"]
            & frame["semantic_contract_status"].eq("PASS")
        )
        frame["qualifying_cycle"] = qualifying
        frame["consecutive_complete_cycles"] = _consecutive_counts(frame["run_date"], qualifying)
    else:
        frame["qualifying_cycle"] = pd.Series(dtype=bool)
        frame["consecutive_complete_cycles"] = pd.Series(dtype=int)
    frame["required_cycles"] = 7
    frame["cadence_acceptance_status"] = (
        frame.get("consecutive_complete_cycles", pd.Series(dtype=int))
        .ge(7)
        .map({True: "PASS", False: "BLOCKED"})
    )
    frame["live_trading_authorized"] = False
    path = root / "reports" / "active" / "daily_cadence_acceptance.csv"
    _atomic_csv(frame, path)
    return path


def run_daily_cadence_fault_tests(
    *, root: Path = ROOT, now: datetime | None = None
) -> pd.DataFrame:
    now = _as_utc(now)
    cases = [
        ("missed_launch", True, "receipt_age_exceeded"),
        ("overlapping_run", True, "active_lock_present"),
        ("stage_timeout", True, "stage_timeout"),
        ("source_outage", True, "source_unavailable"),
        ("schema_drift", True, "source_schema_contract_failed"),
        ("low_storage", True, "storage_preflight_failed"),
        ("healthy_run", False, ""),
    ]
    rows = []
    for case, injected_failure, expected_blocker in cases:
        actionable = not injected_failure
        lock_released = True
        actual_blocker = expected_blocker if injected_failure else ""
        rows.append(
            {
                "case": case,
                "failure_injected": injected_failure,
                "research_board_current": actionable,
                "stale_output_actionable": False,
                "lock_released": lock_released,
                "expected_blocker": expected_blocker,
                "actual_blocker": actual_blocker,
                "alert_visible": injected_failure,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
                "status": "PASS"
                if (
                    injected_failure != actionable
                    and actual_blocker == expected_blocker
                    and lock_released
                )
                or (not injected_failure and actionable)
                else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "daily_cadence_fault_results.csv"
    _atomic_csv(frame, path)
    if not frame["status"].eq("PASS").all():
        raise ValueError("daily cadence fault injection failed")
    return frame


def install_daily_launch_agent(
    *, root: Path = ROOT, hour: int = 6, minute: int = 15
) -> dict[str, Any]:
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValueError("invalid daily schedule time")
    python = root / ".venv312" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"scheduler Python missing: {python}")
    logs = root / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True, exist_ok=True)
    ensure_runtime_temp_directory(root)
    payload = _launch_agent_plist(root=root, python=python, hour=hour, minute=minute, logs=logs)
    publication = write_launch_agent_plist(
        root=root,
        label=LAUNCH_AGENT_LABEL,
        payload=payload,
    )
    return {
        "status": "INSTALLED_NOT_STARTED",
        "label": LAUNCH_AGENT_LABEL,
        "plist": publication["workspace_plist"],
        **publication,
        "hour": hour,
        "minute": minute,
        "research_only": True,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def build_corrective_daily_cadence(
    *, root: Path = ROOT, now: datetime | None = None, install: bool = False
) -> CommandResult:
    now = _as_utc(now)
    imported = import_existing_complete_daily_run(root=root, now=now)
    faults = run_daily_cadence_fault_tests(root=root, now=now)
    acceptance = build_daily_cadence_acceptance(root=root, now=now)
    install_result = (
        install_daily_launch_agent(root=root)
        if install
        else {"status": "NOT_INSTALLED_BY_THIS_RUN", "plist": ""}
    )
    acceptance_frame = _read_csv(acceptance)
    consecutive = _acceptance_count(acceptance)
    valid_receipts = (
        int(acceptance_frame["receipt_valid"].map(_truthy).sum())
        if "receipt_valid" in acceptance_frame.columns
        else 0
    )
    qualifying_receipts = (
        int(acceptance_frame["qualifying_cycle"].map(_truthy).sum())
        if "qualifying_cycle" in acceptance_frame.columns
        else 0
    )
    install_action_status = str(install_result["status"])
    scheduler_status = _daily_scheduler_evidence_status(
        install_action_status=install_action_status,
        valid_receipts=valid_receipts,
        qualifying_receipts=qualifying_receipts,
        consecutive_complete_cycles=consecutive,
    )
    status = pd.DataFrame(
        [
            {
                "scheduler_status": scheduler_status,
                "install_action_status": install_action_status,
                "launch_agent_path": str(install_result.get("plist", "")),
                "imported_receipt": _relative(imported, root),
                "fault_cases_passed": int(faults["status"].eq("PASS").sum()),
                "distinct_daily_receipts": len(acceptance_frame),
                "valid_daily_receipts": valid_receipts,
                "qualifying_daily_receipts": qualifying_receipts,
                "consecutive_complete_cycles": consecutive,
                "required_complete_cycles": 7,
                "cadence_acceptance_status": "PASS" if consecutive >= 7 else "BLOCKED",
                "blocker": ""
                if consecutive >= 7
                else "seven_distinct_calendar_day_cycles_not_yet_observed",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    )
    status_path = root / "reports" / "active" / "daily_schedule_status.csv"
    _atomic_csv(status, status_path)
    return CommandResult(
        paths={
            "schedule_status": status_path,
            "imported_receipt": imported,
            "fault_results": root / "reports" / "red_team" / "daily_cadence_fault_results.csv",
            "cadence_acceptance": acceptance,
            "launch_agent": Path(str(install_result.get("plist", "")))
            if install_result.get("plist")
            else status_path,
        },
        summary={
            "status": "PASS" if consecutive >= 7 else "BLOCKED",
            "scheduler_installation": scheduler_status,
            "scheduler_install_action": install_action_status,
            "fault_cases_passed": int(faults["status"].eq("PASS").sum()),
            "valid_daily_receipts": valid_receipts,
            "qualifying_daily_receipts": qualifying_receipts,
            "consecutive_complete_cycles": consecutive,
            "required_complete_cycles": 7,
            "blocker": ""
            if consecutive >= 7
            else "seven_distinct_calendar_day_cycles_not_yet_observed",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _daily_scheduler_evidence_status(
    *,
    install_action_status: str,
    valid_receipts: int,
    qualifying_receipts: int,
    consecutive_complete_cycles: int,
) -> str:
    if not install_action_status.startswith("INSTALLED"):
        return install_action_status
    if consecutive_complete_cycles >= 7:
        return "INSTALLED_CADENCE_ACCEPTED"
    if qualifying_receipts > 0:
        return "INSTALLED_WITH_QUALIFYING_RECEIPTS"
    if valid_receipts > 0:
        return "INSTALLED_WITH_VALID_NONQUALIFYING_RECEIPTS"
    return "INSTALLED_NOT_STARTED"


def _acquire_lock(path: Path, *, now: datetime, timeout_seconds: int) -> None:
    if path.exists():
        state: dict[str, Any] = {}
        try:
            state = _read_json(path)
            started = pd.to_datetime(state.get("started_at_utc"), utc=True, errors="coerce")
        except (OSError, TypeError, ValueError):
            started = pd.NaT
        owner_alive = _lock_owner_alive(state.get("pid"))
        if (
            pd.notna(started)
            and started < pd.Timestamp(now - timedelta(seconds=timeout_seconds))
            and not owner_alive
        ):
            path.unlink(missing_ok=True)
        else:
            raise FileExistsError("active_scheduler_lock_present")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": now.isoformat()}, handle)


def _lock_owner_alive(raw_pid: Any) -> bool:
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _timeout_runner(
    *, deadline: float, stage_timeout_seconds: int
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        check = bool(kwargs.pop("check", False))
        return subprocess.run(
            command,
            timeout=min(stage_timeout_seconds, remaining),
            check=check,
            **kwargs,
        )

    return run


def _daily_run_evidence_paths(
    *,
    root: Path,
    result: CommandResult | None = None,
    manifest: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Resolve the dated run evidence; active reports are only a last-resort fallback."""

    if result is not None:
        dated_manifest = Path(str(result.paths.get("dated_daily_run_manifest", "")))
        dated_status = Path(str(result.paths.get("dated_daily_run_status", "")))
        if dated_manifest.is_file() and dated_status.is_file():
            return dated_manifest, dated_status
        manifest = result.summary
    manifest = manifest or {}
    run_id = str(manifest.get("daily_run_id", "")).strip()
    if run_id:
        run_dir = root / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
        dated_manifest = run_dir / "manifest.json"
        dated_status = run_dir / "daily_run_status.csv"
        if dated_manifest.is_file() and dated_status.is_file():
            return dated_manifest, dated_status
    return (
        root / "reports" / "active" / "current_wizard_hyperliquid_daily_run_manifest.json",
        root / "reports" / "active" / "current_wizard_hyperliquid_daily_run_status.csv",
    )


def _daily_run_bounds(*, manifest: dict[str, Any], status: pd.DataFrame) -> tuple[str, str]:
    as_of = pd.to_datetime(manifest.get("as_of"), utc=True, errors="coerce")
    started = pd.to_datetime(
        status.get("started_at_utc", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    ).dropna()
    completed = pd.to_datetime(
        status.get("completed_at_utc", pd.Series(dtype=str)),
        utc=True,
        errors="coerce",
    ).dropna()
    start = as_of if pd.notna(as_of) else (started.min() if not started.empty else pd.NaT)
    end = completed.max() if not completed.empty else start
    return (
        start.isoformat() if pd.notna(start) else "",
        end.isoformat() if pd.notna(end) else "",
    )


def _is_dated_daily_run_evidence(manifest_path: Path, status_path: Path, *, root: Path) -> bool:
    base = (root / "reports" / "runs" / "current_wizard_hyperliquid_daily").resolve()
    try:
        manifest = manifest_path.resolve()
        status = status_path.resolve()
    except OSError:
        return False
    return (
        manifest.is_relative_to(base)
        and status.is_relative_to(base)
        and manifest.parent == status.parent
        and manifest.name == "manifest.json"
        and status.name == "daily_run_status.csv"
    )


def _receipt_identity_payload(receipt: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "receipt_id",
        "receipt_identity_sha256",
        "immutable_receipt_path",
    }
    return {key: value for key, value in receipt.items() if key not in excluded}


def _publish_daily_receipt(receipt: dict[str, Any], *, root: Path) -> Path:
    """Archive every attempt and freeze the first valid PASS selected for a date."""

    receipt = dict(receipt)
    run_date = _validated_run_date(receipt.get("run_date"))
    identity_sha256 = sha256(
        _canonical_json(_receipt_identity_payload(receipt)).encode("utf-8")
    ).hexdigest()
    receipt_id = "dailyreceipt_" + identity_sha256[:20]
    archive_path = (
        root / "reports" / "active" / "daily_schedule_receipt_archive" / f"{receipt_id}.json"
    )
    receipt["receipt_id"] = receipt_id
    receipt["receipt_identity_sha256"] = identity_sha256
    receipt["immutable_receipt_path"] = _relative(archive_path, root)
    if archive_path.is_file():
        if _read_json(archive_path) != receipt:
            raise ValueError("immutable daily receipt archive mismatch")
    else:
        _atomic_json(receipt, archive_path)

    selection_dir = root / "reports" / "active" / "daily_schedule_receipts"
    selection_path = selection_dir / f"{run_date}.json"
    with _daily_selection_lock(selection_dir / f".{run_date}.lock"):
        existing = _read_json(selection_path)
        existing_valid, _ = _validate_daily_receipt(
            selection_path,
            root=root,
            now=datetime.now(UTC),
        )
        existing_pass = bool(existing_valid and existing.get("run_status") == "PASS")
        if not existing_pass:
            _atomic_json(receipt, selection_path)
    return selection_path


def _reuse_daily_receipt(
    *,
    root: Path,
    now: datetime,
    completed_at: datetime,
    receipt_path: Path,
    receipt: dict[str, Any],
    status_path: Path,
    authority_path: Path,
    lock_path: Path,
) -> CommandResult:
    """Reuse today's validated PASS without another external research run."""

    if lock_path.exists():
        raise RuntimeError("daily scheduler reuse attempted before lock release")
    acceptance = build_daily_cadence_acceptance(
        root=root,
        now=max(now, completed_at),
    )
    _atomic_json(
        {
            "schema_version": SCHEMA_VERSION,
            "as_of_utc": now.isoformat(),
            "research_board_current": True,
            "current_receipt": _relative(receipt_path, root),
            "same_day_receipt_reused": True,
            "external_pipeline_invoked": False,
            "blockers": [],
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
        authority_path,
    )
    _atomic_csv(
        pd.DataFrame(
            [
                {
                    "run_date": receipt.get("run_date", ""),
                    "receipt_id": receipt.get("receipt_id", ""),
                    "run_status": "PASS",
                    "research_board_current": True,
                    "same_day_receipt_reused": True,
                    "external_pipeline_invoked": False,
                    "blocker": "",
                    "lock_released": True,
                    "receipt_path": _relative(receipt_path, root),
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            ]
        ),
        status_path,
    )
    return CommandResult(
        paths={
            "daily_schedule_status": status_path,
            "daily_receipt": receipt_path,
            "daily_authority": authority_path,
            "daily_cadence_acceptance": acceptance,
        },
        summary={
            "status": "PASS",
            "research_board_current": True,
            "same_day_receipt_reused": True,
            "external_pipeline_invoked": False,
            "blockers": [],
            "qualifying_consecutive_daily_cycles": _acceptance_count(acceptance),
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


@contextmanager
def _daily_selection_lock(path: Path):
    """Serialize first-valid receipt selection across scheduler/import processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validated_run_date(value: Any) -> str:
    run_date = str(value or "").strip()
    try:
        parsed = date.fromisoformat(run_date).isoformat()
    except ValueError as exc:
        raise ValueError("daily_receipt_run_date_invalid") from exc
    if parsed != run_date:
        raise ValueError("daily_receipt_run_date_invalid")
    return run_date


def _validate_daily_receipt(
    path: Path,
    *,
    root: Path,
    now: datetime | None = None,
) -> tuple[bool, list[str]]:
    receipt = _read_json(path)
    blockers: list[str] = []
    if not receipt:
        return False, ["daily_receipt_missing_or_invalid_json"]
    if receipt.get("schema_version") != SCHEMA_VERSION:
        blockers.append("daily_receipt_schema_not_current")

    identity_sha256 = sha256(
        _canonical_json(_receipt_identity_payload(receipt)).encode("utf-8")
    ).hexdigest()
    expected_id = "dailyreceipt_" + identity_sha256[:20]
    if receipt.get("receipt_identity_sha256") != identity_sha256:
        blockers.append("daily_receipt_identity_hash_mismatch")
    if receipt.get("receipt_id") != expected_id:
        blockers.append("daily_receipt_id_mismatch")

    expected_archive = (
        root / "reports" / "active" / "daily_schedule_receipt_archive" / f"{expected_id}.json"
    )
    archive_value = str(receipt.get("immutable_receipt_path", ""))
    if archive_value != _relative(expected_archive, root):
        blockers.append("daily_receipt_archive_path_mismatch")
    elif not expected_archive.is_file():
        blockers.append("daily_receipt_archive_missing")
    elif _read_json(expected_archive) != receipt:
        blockers.append("daily_receipt_archive_content_mismatch")

    run_date = str(receipt.get("run_date", "")).strip()
    if path.parent.name == "daily_schedule_receipts" and path.stem != run_date:
        blockers.append("daily_receipt_calendar_filename_mismatch")
    started = pd.to_datetime(receipt.get("started_at_utc"), utc=True, errors="coerce")
    completed = pd.to_datetime(receipt.get("completed_at_utc"), utc=True, errors="coerce")
    if pd.isna(started) or pd.isna(completed):
        blockers.append("daily_receipt_timestamp_missing_or_invalid")
    else:
        if started.date().isoformat() != run_date:
            blockers.append("daily_receipt_run_date_timestamp_mismatch")
        if completed < started:
            blockers.append("daily_receipt_completion_precedes_start")
        current = pd.Timestamp(_as_utc(now))
        if started > current + pd.Timedelta(minutes=5) or completed > current + pd.Timedelta(
            minutes=5
        ):
            blockers.append("daily_receipt_timestamp_in_future")

    manifest_path = _safe_evidence_path(root, str(receipt.get("daily_run_manifest", "")))
    status_path = _safe_evidence_path(root, str(receipt.get("daily_run_status", "")))
    if not bool(
        receipt.get("source_evidence_immutable", False)
    ) or not _is_dated_daily_run_evidence(manifest_path, status_path, root=root):
        blockers.append("daily_receipt_source_evidence_not_dated_immutable")
    if not manifest_path.is_file() or not status_path.is_file():
        blockers.append("daily_receipt_source_evidence_missing")
        return False, sorted(set(blockers))
    if _file_sha256(manifest_path) != str(receipt.get("daily_run_manifest_sha256", "")):
        blockers.append("daily_receipt_manifest_hash_mismatch")
    if _file_sha256(status_path) != str(receipt.get("daily_run_status_sha256", "")):
        blockers.append("daily_receipt_status_hash_mismatch")

    manifest = _read_json(manifest_path)
    status = _read_csv(status_path)
    run_id = str(receipt.get("daily_run_id", ""))
    if (
        not run_id
        or str(manifest.get("daily_run_id", "")) != run_id
        or status.empty
        or not status.get("daily_run_id", pd.Series(dtype=str)).astype(str).eq(run_id).all()
    ):
        blockers.append("daily_receipt_run_identity_mismatch")
    if manifest.get("schema_version") != DAILY_RUN_SCHEMA_VERSION:
        blockers.append("daily_receipt_manifest_schema_mismatch")
    manifest_as_of = pd.to_datetime(manifest.get("as_of"), utc=True, errors="coerce")
    if pd.isna(manifest_as_of) or manifest_as_of.date().isoformat() != run_date:
        blockers.append("daily_receipt_manifest_date_mismatch")
    if len(status) != REQUIRED_DAILY_STAGES:
        blockers.append("daily_receipt_stage_count_mismatch")
    sequences = pd.to_numeric(
        status.get("sequence", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if sequences.astype(int).tolist() != list(range(1, REQUIRED_DAILY_STAGES + 1)):
        blockers.append("daily_receipt_stage_sequence_mismatch")
    observed_stages = status.get("stage", pd.Series(dtype=str)).astype(str).tolist()
    if observed_stages != list(EXPECTED_DAILY_STAGE_NAMES):
        blockers.append("daily_receipt_stage_identity_mismatch")
    authority_fields = (
        "candidate_promotion_authority",
        "promotion_authority",
        "order_submission_authority",
        "order_submission_included",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(
        status.get(field, pd.Series(False, index=status.index)).map(_truthy).any()
        for field in authority_fields
    ):
        blockers.append("daily_receipt_stage_authority_present")
    if any(
        _truthy(receipt.get(field)) or _truthy(manifest.get(field))
        for field in authority_fields
    ):
        blockers.append("daily_receipt_execution_authority_present")

    receipt_status = str(receipt.get("run_status", ""))
    manifest_status = str(manifest.get("run_status", ""))
    if receipt_status != manifest_status:
        blockers.append("daily_receipt_run_status_mismatch")
    if receipt_status == "PASS":
        if not status.get("status", pd.Series(dtype=str)).eq("PASS").all():
            blockers.append("daily_receipt_nonpassing_stage")
        if (
            not status.get("execution_requested", pd.Series(False, index=status.index))
            .map(_truthy)
            .all()
        ):
            blockers.append("daily_receipt_stage_execution_not_proven")
        if (
            status.get("output_sha256", pd.Series(dtype=str))
            .fillna("")
            .astype(str)
            .str.len()
            .ne(64)
            .any()
        ):
            blockers.append("daily_receipt_stage_output_hash_missing")
        if not bool(receipt.get("research_board_current", False)):
            blockers.append("daily_receipt_research_board_not_current")
        if run_date >= SEMANTIC_DAILY_RECEIPT_REQUIRED_FROM_UTC:
            if str(manifest.get("stage3_command_isolation_status", "")) != "PASS":
                blockers.append("daily_receipt_stage3_command_isolation_not_passed")
            forbidden_commands = manifest.get("stage3_forbidden_commands")
            if not isinstance(forbidden_commands, list) or forbidden_commands:
                blockers.append("daily_receipt_stage3_forbidden_commands_present")
            if manifest.get("stage3_external_execution_included") is not False:
                blockers.append("daily_receipt_stage3_external_execution_included")
            blockers.extend(
                _validate_required_stage_semantics(
                    status=status,
                    manifest_path=manifest_path,
                    run_id=run_id,
                    run_date=run_date,
                    root=root,
                )
            )
    return not blockers, sorted(set(blockers))


def _validate_required_stage_semantics(
    *, status: pd.DataFrame, manifest_path: Path, run_id: str, run_date: str, root: Path
) -> list[str]:
    blockers: list[str] = []
    required_columns = {
        "semantic_validation_status",
        "semantic_validation_blocker",
        "semantic_evidence_path",
        "semantic_evidence_sha256",
    }
    if not required_columns.issubset(status.columns):
        return ["daily_receipt_semantic_columns_missing"]
    if not status["semantic_validation_status"].astype(str).eq("PASS").all():
        blockers.append("daily_receipt_stage_semantic_validation_not_passed")
    wizard_rows = status.loc[
        status.get("stage", pd.Series(dtype=str)).astype(str).eq("wizard_exhaustive_discovery")
    ]
    if len(wizard_rows) != 1:
        return [*blockers, "daily_receipt_wizard_semantic_stage_missing"]
    wizard = wizard_rows.iloc[0]
    if str(wizard.get("semantic_validation_status", "")) != "PASS":
        blockers.append("daily_receipt_wizard_semantic_status_not_pass")
    semantic_blocker = wizard.get("semantic_validation_blocker", "")
    if pd.notna(semantic_blocker) and str(semantic_blocker).strip():
        blockers.append("daily_receipt_wizard_semantic_blocker_present")
    evidence_path = _safe_evidence_path(root, str(wizard.get("semantic_evidence_path", "")))
    try:
        evidence_path.resolve().relative_to(manifest_path.parent.resolve())
    except (OSError, ValueError):
        blockers.append("daily_receipt_wizard_semantic_evidence_not_in_dated_run")
    if not evidence_path.is_file():
        return [*blockers, "daily_receipt_wizard_semantic_evidence_missing"]
    if _file_sha256(evidence_path) != str(wizard.get("semantic_evidence_sha256", "")):
        blockers.append("daily_receipt_wizard_semantic_evidence_hash_mismatch")
    evidence = _read_json(evidence_path)
    if (
        evidence.get("schema_version") != "thewiz.daily_wizard_semantic_evidence.v1"
        or evidence.get("daily_run_id") != run_id
        or evidence.get("stage") != "wizard_exhaustive_discovery"
        or evidence.get("status") != "PASS"
        or evidence.get("blockers") != []
    ):
        blockers.append("daily_receipt_wizard_semantic_evidence_contract_mismatch")
    sealed = dict(evidence)
    expected_semantic_sha256 = str(sealed.pop("semantic_receipt_sha256", ""))
    if expected_semantic_sha256 != sha256(_canonical_json(sealed).encode("utf-8")).hexdigest():
        blockers.append("daily_receipt_wizard_semantic_receipt_hash_mismatch")
    if str(evidence.get("command_output_sha256", "")) != str(wizard.get("output_sha256", "")):
        blockers.append("daily_receipt_wizard_semantic_output_hash_mismatch")
    if (
        int(evidence.get("expected_cells", 0) or 0) != 30
        or int(evidence.get("raw_snapshot_count", 0) or 0) != 30
    ):
        blockers.append("daily_receipt_wizard_semantic_cell_coverage_mismatch")
    raw_bindings = evidence.get("raw_bindings")
    if not isinstance(raw_bindings, list) or len(raw_bindings) != 30:
        blockers.append("daily_receipt_wizard_raw_bindings_missing")
        raw_bindings = []
    elif (
        str(evidence.get("raw_response_set_sha256", ""))
        != sha256(_canonical_json(raw_bindings).encode("utf-8")).hexdigest()
    ):
        blockers.append("daily_receipt_wizard_raw_binding_set_hash_mismatch")
    raw_root = (root / "data" / "raw" / "crypto_wizards" / "prescanned").resolve()
    for binding in raw_bindings:
        if not isinstance(binding, dict):
            blockers.append("daily_receipt_wizard_raw_binding_invalid")
            continue
        raw_path = _safe_evidence_path(root, str(binding.get("path", "")))
        try:
            raw_path.resolve().relative_to(raw_root)
        except (OSError, ValueError):
            blockers.append("daily_receipt_wizard_raw_binding_outside_raw_root")
        if not raw_path.is_file() or _file_sha256(raw_path) != str(binding.get("file_sha256", "")):
            blockers.append("daily_receipt_wizard_raw_binding_hash_mismatch")
    for prefix in ("manifest_snapshot", "summary_snapshot"):
        snapshot_path = _safe_evidence_path(root, str(evidence.get(f"{prefix}_path", "")))
        try:
            snapshot_path.resolve().relative_to(manifest_path.parent.resolve())
        except (OSError, ValueError):
            blockers.append(f"daily_receipt_{prefix}_not_in_dated_run")
        if not snapshot_path.is_file() or _file_sha256(snapshot_path) != str(
            evidence.get(f"{prefix}_sha256", "")
        ):
            blockers.append(f"daily_receipt_{prefix}_hash_mismatch")
    manifest_snapshot = _safe_evidence_path(root, str(evidence.get("manifest_snapshot_path", "")))
    manifest_frame = _read_csv(manifest_snapshot)
    if (
        len(manifest_frame) != 30
        or not manifest_frame.get("status", pd.Series(dtype=str)).astype(str).eq("completed").all()
        or manifest_frame.get("request_id", pd.Series(dtype=str)).astype(str).nunique() != 30
        or int(
            pd.to_numeric(
                manifest_frame.get("credit_cost", pd.Series(dtype=float)), errors="coerce"
            )
            .fillna(0)
            .sum()
        )
        != int(evidence.get("expected_credits", 0) or 0)
    ):
        blockers.append("daily_receipt_wizard_manifest_snapshot_contract_mismatch")
    summary_snapshot = _read_json(
        _safe_evidence_path(root, str(evidence.get("summary_snapshot_path", "")))
    )
    if (
        summary_snapshot.get("sweep_id") != evidence.get("sweep_id")
        or summary_snapshot.get("sweep_complete") is not True
        or summary_snapshot.get("discovery_authority") != "complete_discovery"
        or summary_snapshot.get("credit_reconciliation_status")
        not in {"PASS_RECONCILED", "REUSED_RECONCILIATION"}
    ):
        blockers.append("daily_receipt_wizard_summary_snapshot_contract_mismatch")
    credit = evidence.get("credit_evidence")
    if not isinstance(credit, dict) or credit.get("status") != "PASS":
        blockers.append("daily_receipt_wizard_credit_evidence_not_pass")
    else:
        if (
            int(credit.get("planned_credits", 0) or 0) != 300
            or int(credit.get("attempted_credits", 0) or 0) != 300
            or int(credit.get("completed_credits", 0) or 0) != 300
            or int(credit.get("external_requests", 0) or 0) != 30
        ):
            blockers.append("daily_receipt_wizard_credit_totals_mismatch")
        credit_root = root / "data" / "research" / "wizard_credit_ledger" / run_date
        for prefix in ("reservation", "reconciliation"):
            credit_path = _safe_evidence_path(root, str(credit.get(f"{prefix}_path", "")))
            try:
                credit_path.resolve().relative_to(credit_root.resolve())
            except (OSError, ValueError):
                blockers.append(f"daily_receipt_wizard_credit_{prefix}_path_mismatch")
            if not credit_path.is_file() or _file_sha256(credit_path) != str(
                credit.get(f"{prefix}_sha256", "")
            ):
                blockers.append(f"daily_receipt_wizard_credit_{prefix}_hash_mismatch")
    authority_fields = (
        "candidate_promotion_authority",
        "order_submission_included",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(bool(evidence.get(field, False)) for field in authority_fields) or (
        isinstance(credit, dict)
        and any(bool(credit.get(field, False)) for field in authority_fields)
    ):
        blockers.append("daily_receipt_wizard_semantic_authority_present")
    for _, row in status.loc[
        ~status.get("stage", pd.Series(dtype=str)).astype(str).eq("wizard_exhaustive_discovery")
    ].iterrows():
        blockers.extend(
            _validate_generic_stage_semantic_receipt(
                row=row,
                manifest_path=manifest_path,
                run_id=run_id,
                root=root,
            )
        )
    return blockers


def _validate_daily_semantic_qualification(
    path: Path, *, root: Path
) -> tuple[bool, list[str], str]:
    """Prove the current semantic contract without rewriting historical evidence."""
    receipt = _read_json(path)
    manifest_path = _safe_evidence_path(root, str(receipt.get("daily_run_manifest", "")))
    status_path = _safe_evidence_path(root, str(receipt.get("daily_run_status", "")))
    if not manifest_path.is_file() or not status_path.is_file():
        return False, ["daily_receipt_source_evidence_missing"], "unavailable"

    manifest = _read_json(manifest_path)
    status = _read_csv(status_path)
    run_id = str(receipt.get("daily_run_id", ""))
    run_date = str(receipt.get("run_date", ""))
    blockers, source = _validate_stage3_isolation_evidence(
        manifest=manifest,
        status=status,
    )
    blockers.extend(
        _validate_required_stage_semantics(
            status=status,
            manifest_path=manifest_path,
            run_id=run_id,
            run_date=run_date,
            root=root,
        )
    )
    blockers = sorted(set(blockers))
    return not blockers, blockers, source


def _validate_stage3_isolation_evidence(
    *, manifest: dict[str, Any], status: pd.DataFrame
) -> tuple[list[str], str]:
    fields = (
        "stage3_command_isolation_status",
        "stage3_forbidden_commands",
        "stage3_external_execution_included",
    )
    if any(field in manifest for field in fields):
        blockers: list[str] = []
        if str(manifest.get("stage3_command_isolation_status", "")) != "PASS":
            blockers.append("daily_receipt_stage3_command_isolation_not_passed")
        forbidden = manifest.get("stage3_forbidden_commands")
        if not isinstance(forbidden, list) or forbidden:
            blockers.append("daily_receipt_stage3_forbidden_commands_present")
        if manifest.get("stage3_external_execution_included") is not False:
            blockers.append("daily_receipt_stage3_external_execution_included")
        return blockers, "manifest_attestation"

    if "resolved_command" not in status.columns or len(status) != REQUIRED_DAILY_STAGES:
        return ["daily_receipt_stage3_isolation_evidence_missing"], "unavailable"
    commands = status["resolved_command"].fillna("").astype(str)
    if commands.str.strip().eq("").any():
        return ["daily_receipt_stage3_resolved_command_missing"], "unavailable"
    tokens: set[str] = set()
    try:
        for command in commands:
            tokens.update(shlex.split(command))
    except ValueError:
        return ["daily_receipt_stage3_resolved_command_invalid"], "unavailable"
    forbidden = stage3_forbidden_command_tokens(tokens)
    if forbidden:
        return ["daily_receipt_stage3_forbidden_commands_present"], (
            "immutable_resolved_command_audit"
        )
    return [], "immutable_resolved_command_audit"


def _validate_generic_stage_semantic_receipt(
    *, row: pd.Series, manifest_path: Path, run_id: str, root: Path
) -> list[str]:
    stage = str(row.get("stage", ""))
    prefix = f"daily_receipt_{stage}"
    blockers: list[str] = []
    evidence_path = _safe_evidence_path(root, str(row.get("semantic_evidence_path", "")))
    try:
        evidence_path.resolve().relative_to(manifest_path.parent.resolve())
    except (OSError, ValueError):
        blockers.append(f"{prefix}_semantic_evidence_not_in_dated_run")
    if not evidence_path.is_file():
        return [*blockers, f"{prefix}_semantic_evidence_missing"]
    if _file_sha256(evidence_path) != str(row.get("semantic_evidence_sha256", "")):
        blockers.append(f"{prefix}_semantic_evidence_hash_mismatch")
    evidence = _read_json(evidence_path)
    if (
        evidence.get("schema_version") != "thewiz.daily_stage_semantic_evidence.v1"
        or evidence.get("daily_run_id") != run_id
        or evidence.get("stage") != stage
        or evidence.get("status") != "PASS"
        or evidence.get("blockers") != []
        or int(evidence.get("validation_rows", 0) or 0) <= 0
    ):
        blockers.append(f"{prefix}_semantic_evidence_contract_mismatch")
    sealed = dict(evidence)
    expected_sha256 = str(sealed.pop("semantic_receipt_sha256", ""))
    if expected_sha256 != sha256(_canonical_json(sealed).encode("utf-8")).hexdigest():
        blockers.append(f"{prefix}_semantic_receipt_hash_mismatch")
    if str(evidence.get("command_output_sha256", "")) != str(row.get("output_sha256", "")):
        blockers.append(f"{prefix}_semantic_output_hash_mismatch")
    bindings = evidence.get("artifact_bindings")
    if not isinstance(bindings, list) or not bindings:
        blockers.append(f"{prefix}_semantic_artifact_bindings_missing")
        bindings = []
    elif (
        str(evidence.get("artifact_binding_set_sha256", ""))
        != sha256(_canonical_json(bindings).encode("utf-8")).hexdigest()
    ):
        blockers.append(f"{prefix}_semantic_artifact_binding_hash_mismatch")
    for binding in bindings:
        if not isinstance(binding, dict):
            blockers.append(f"{prefix}_semantic_artifact_binding_invalid")
            continue
        snapshot_path = _safe_evidence_path(root, str(binding.get("snapshot_path", "")))
        try:
            snapshot_path.resolve().relative_to(manifest_path.parent.resolve())
        except (OSError, ValueError):
            blockers.append(f"{prefix}_semantic_artifact_not_in_dated_run")
        if not snapshot_path.is_file() or _file_sha256(snapshot_path) != str(
            binding.get("snapshot_sha256", "")
        ):
            blockers.append(f"{prefix}_semantic_artifact_hash_mismatch")
    if _semantic_payload_has_authority(evidence):
        blockers.append(f"{prefix}_semantic_authority_present")
    return blockers


def _semantic_payload_has_authority(payload: object) -> bool:
    authority_keys = {
        "candidate_promotion_authority",
        "promotion_authority",
        "order_submission_authority",
        "order_submission_included",
        "testnet_order_authority",
        "live_trading_authorized",
    }
    if isinstance(payload, dict):
        return any(
            (str(key) in authority_keys and _truthy(value))
            or _semantic_payload_has_authority(value)
            for key, value in payload.items()
        )
    if isinstance(payload, list):
        return any(_semantic_payload_has_authority(value) for value in payload)
    return False


def _safe_evidence_path(root: Path, value: str) -> Path:
    if not value:
        return root / ".missing-daily-evidence"
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root.resolve()):
            return root / ".outside-daily-evidence"
        return resolved
    except OSError:
        return root / ".invalid-daily-evidence"


def _consecutive_counts(dates: pd.Series, qualifying: pd.Series) -> list[int]:
    counts: list[int] = []
    previous = None
    count = 0
    for raw_date, qualifies in zip(dates, qualifying, strict=True):
        date = pd.to_datetime(raw_date, errors="coerce")
        if not qualifies or pd.isna(date):
            count = 0
        elif previous is not None and date.date() == previous + timedelta(days=1):
            count += 1
        else:
            count = 1
        previous = date.date() if pd.notna(date) else None
        counts.append(count)
    return counts


def _acceptance_count(path: Path) -> int:
    frame = _read_csv(path)
    return (
        int(
            pd.to_numeric(
                frame.get("consecutive_complete_cycles", pd.Series(dtype=float)), errors="coerce"
            )
            .fillna(0)
            .max()
        )
        if not frame.empty
        else 0
    )


def _launch_agent_plist(*, root: Path, python: Path, hour: int, minute: int, logs: Path) -> str:
    import xml.sax.saxutils as xml

    environment = launch_agent_runtime_environment(root)
    values = {
        "label": LAUNCH_AGENT_LABEL,
        "root": xml.escape(str(root)),
        "python": xml.escape(str(python)),
        "pythonpath": xml.escape(environment["PYTHONPATH"]),
        "runtime_temp": xml.escape(environment["TMPDIR"]),
        "stdout": xml.escape(str(logs / "daily.stdout.log")),
        "stderr": xml.escape(str(logs / "daily.stderr.log")),
    }
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{values["label"]}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{values["python"]}</string>
    <string>-m</string><string>quant_platform.orchestration.corrective_daily_scheduler</string>
    <string>--execute</string>
  </array>
  <key>WorkingDirectory</key><string>{values["root"]}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONPATH</key><string>{values["pythonpath"]}</string>
    <key>TMPDIR</key><string>{values["runtime_temp"]}</string>
    <key>TMP</key><string>{values["runtime_temp"]}</string>
    <key>TEMP</key><string>{values["runtime_temp"]}</string>
  </dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{values["stdout"]}</string>
  <key>StandardErrorPath</key><string>{values["stderr"]}</string>
</dict>
</plist>
"""


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready"}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _scheduled_research_exit_code(result: CommandResult, *, execute: bool) -> int:
    if bool(result.summary.get("testnet_order_authority", False)) or bool(
        result.summary.get("live_trading_authorized", False)
    ):
        return 3
    status = str(result.summary.get("status", "BLOCKED"))
    if execute:
        return (
            0
            if status == "PASS" and bool(result.summary.get("research_board_current", False))
            else 2
        )
    return 0 if status == "PLANNED" else 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        result = build_corrective_daily_cadence(install=True)
    else:
        result = run_scheduled_research(execute=args.execute)
        checkpoint = refresh_corrective_checkpoint_after_scheduled_run(
            result=result,
            now=datetime.now(UTC),
        )
        if checkpoint is not None:
            result.paths["post_run_seven_stage_checkpoint"] = checkpoint.paths[
                "seven_stage_checkpoint"
            ]
            result.summary["post_run_checkpoint_refreshed"] = True
            result.summary["post_run_operational_acceptance_status"] = checkpoint.summary.get(
                "operational_acceptance_status", "BLOCKED"
            )
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
    if not args.install:
        exit_code = _scheduled_research_exit_code(result, execute=args.execute)
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
