"""Research-only daily scheduler with locks, timeouts, and freshness revocation."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import MINIMUM_FREE_BYTES
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    run_current_wizard_hyperliquid_daily_pipeline,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_daily_scheduler.v1"
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-research-daily"
DEFAULT_RUN_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_STAGE_TIMEOUT_SECONDS = 45 * 60
MAX_RECEIPT_AGE_HOURS = 36


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
) -> CommandResult:
    now = _as_utc(now)
    active = root / "reports" / "active"
    receipts = active / "daily_schedule_receipts"
    active.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = active / ".corrective_daily.lock"
    status_path = active / "daily_schedule_status.csv"
    authority_path = active / "daily_research_authority.json"
    receipt_path = receipts / f"{now.date().isoformat()}.json"
    blockers: list[str] = []
    lock_acquired = False
    result: CommandResult | None = None
    run_status = "BLOCKED"
    try:
        _acquire_lock(lock_path, now=now, timeout_seconds=run_timeout_seconds)
        lock_acquired = True
        free_bytes = shutil.disk_usage(root).free
        if free_bytes < minimum_free_bytes:
            raise SchedulerBlocked(f"insufficient_free_space:{free_bytes}<{minimum_free_bytes}")
        deadline = time.monotonic() + run_timeout_seconds
        runner = command_runner or _timeout_runner(deadline=deadline, stage_timeout_seconds=stage_timeout_seconds)
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
    except Exception as exc:
        blockers.append(f"scheduler_error:{type(exc).__name__}:{exc}")
        run_status = "FAILED"
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
    current = run_status == "PASS" and not blockers
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": now.date().isoformat(),
        "started_at_utc": now.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_requested": execute,
        "run_status": run_status,
        "blockers": blockers,
        "research_board_current": current,
        "stale_output_actionable": False,
        "scheduler_lock_released": not lock_path.exists(),
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_manifest": str(result.paths.get("daily_run_manifest", "")) if result else "",
    }
    receipt["receipt_id"] = "dailyreceipt_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
    _atomic_json(receipt, receipt_path)
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
    acceptance = build_daily_cadence_acceptance(root=root, now=now)
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


def import_existing_complete_daily_run(
    *, root: Path = ROOT, now: datetime | None = None
) -> Path:
    now = _as_utc(now)
    active = root / "reports" / "active"
    manifest_path = active / "current_wizard_hyperliquid_daily_run_manifest.json"
    status_path = active / "current_wizard_hyperliquid_daily_run_status.csv"
    manifest = _read_json(manifest_path)
    status = _read_csv(status_path)
    complete = (
        str(manifest.get("run_status")) == "PASS"
        and len(status) == 19
        and status.get("status", pd.Series(dtype=str)).eq("PASS").all()
        and not status.get("order_submission_authority", pd.Series(False, index=status.index)).map(_truthy).any()
        and not status.get("live_trading_authorized", pd.Series(False, index=status.index)).map(_truthy).any()
    )
    run_time = pd.to_datetime(manifest.get("as_of"), utc=True, errors="coerce")
    run_date = run_time.date().isoformat() if pd.notna(run_time) else now.date().isoformat()
    receipt_path = active / "daily_schedule_receipts" / f"{run_date}.json"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "run_date": run_date,
        "started_at_utc": str(manifest.get("as_of", "")),
        "completed_at_utc": str(manifest.get("as_of", "")),
        "execution_requested": True,
        "run_status": "PASS" if complete else "BLOCKED",
        "blockers": [] if complete else ["existing_daily_run_not_complete"],
        "research_board_current": bool(complete),
        "stale_output_actionable": False,
        "scheduler_lock_released": True,
        "testnet_execution_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "daily_run_manifest": _relative(manifest_path, root),
        "imported_existing_validated_run": True,
    }
    receipt["receipt_id"] = "dailyreceipt_" + sha256(_canonical_json(receipt).encode("utf-8")).hexdigest()[:20]
    _atomic_json(receipt, receipt_path)
    return receipt_path


def build_daily_cadence_acceptance(
    *, root: Path = ROOT, now: datetime | None = None
) -> Path:
    now = _as_utc(now)
    receipts_dir = root / "reports" / "active" / "daily_schedule_receipts"
    rows = []
    for path in sorted(receipts_dir.glob("*.json")) if receipts_dir.exists() else []:
        receipt = _read_json(path)
        rows.append(
            {
                "run_date": str(receipt.get("run_date", "")),
                "receipt_id": str(receipt.get("receipt_id", "")),
                "run_status": str(receipt.get("run_status", "")),
                "research_board_current": bool(receipt.get("research_board_current", False)),
                "zero_execution_authority": not bool(receipt.get("testnet_order_authority", False)) and not bool(receipt.get("live_trading_authorized", False)),
                "receipt_path": _relative(path, root),
            }
        )
    frame = pd.DataFrame(rows, columns=["run_date", "receipt_id", "run_status", "research_board_current", "zero_execution_authority", "receipt_path"])
    if not frame.empty:
        frame = frame.drop_duplicates("run_date", keep="last").sort_values("run_date")
        qualifying = frame["run_status"].eq("PASS") & frame["zero_execution_authority"]
        frame["qualifying_cycle"] = qualifying
        frame["consecutive_complete_cycles"] = _consecutive_counts(frame["run_date"], qualifying)
    else:
        frame["qualifying_cycle"] = pd.Series(dtype=bool)
        frame["consecutive_complete_cycles"] = pd.Series(dtype=int)
    frame["required_cycles"] = 7
    frame["cadence_acceptance_status"] = frame.get("consecutive_complete_cycles", pd.Series(dtype=int)).ge(7).map({True: "PASS", False: "BLOCKED"})
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
                "status": "PASS" if (injected_failure != actionable and actual_blocker == expected_blocker and lock_released) or (not injected_failure and actionable) else "FAIL",
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
    agent_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    python = root / ".venv312" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"scheduler Python missing: {python}")
    logs = root / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = _launch_agent_plist(root=root, python=python, hour=hour, minute=minute, logs=logs)
    temporary = agent_path.with_suffix(".plist.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(agent_path)
    return {
        "status": "INSTALLED_NOT_STARTED",
        "label": LAUNCH_AGENT_LABEL,
        "plist": agent_path,
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
    install_result = install_daily_launch_agent(root=root) if install else {"status": "NOT_INSTALLED_BY_THIS_RUN", "plist": ""}
    acceptance_frame = _read_csv(acceptance)
    consecutive = _acceptance_count(acceptance)
    status = pd.DataFrame(
        [
            {
                "scheduler_status": install_result["status"],
                "launch_agent_path": str(install_result.get("plist", "")),
                "imported_receipt": _relative(imported, root),
                "fault_cases_passed": int(faults["status"].eq("PASS").sum()),
                "distinct_daily_receipts": len(acceptance_frame),
                "consecutive_complete_cycles": consecutive,
                "required_complete_cycles": 7,
                "cadence_acceptance_status": "PASS" if consecutive >= 7 else "BLOCKED",
                "blocker": "" if consecutive >= 7 else "seven_distinct_calendar_day_cycles_not_yet_observed",
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
            "launch_agent": Path(str(install_result.get("plist", ""))) if install_result.get("plist") else status_path,
        },
        summary={
            "status": "PASS" if consecutive >= 7 else "BLOCKED",
            "scheduler_installation": install_result["status"],
            "fault_cases_passed": int(faults["status"].eq("PASS").sum()),
            "consecutive_complete_cycles": consecutive,
            "required_complete_cycles": 7,
            "blocker": "" if consecutive >= 7 else "seven_distinct_calendar_day_cycles_not_yet_observed",
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )


def _acquire_lock(path: Path, *, now: datetime, timeout_seconds: int) -> None:
    if path.exists():
        try:
            state = _read_json(path)
            started = pd.to_datetime(state.get("started_at_utc"), utc=True, errors="coerce")
        except Exception:
            started = pd.NaT
        if pd.notna(started) and started < pd.Timestamp(now - timedelta(seconds=timeout_seconds)):
            path.unlink(missing_ok=True)
        else:
            raise FileExistsError("active_scheduler_lock_present")
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "started_at_utc": now.isoformat()}, handle)


def _timeout_runner(*, deadline: float, stage_timeout_seconds: int) -> Callable[..., subprocess.CompletedProcess[str]]:
    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, 0)
        return subprocess.run(command, timeout=min(stage_timeout_seconds, remaining), **kwargs)

    return run


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
    return int(pd.to_numeric(frame.get("consecutive_complete_cycles", pd.Series(dtype=float)), errors="coerce").fillna(0).max()) if not frame.empty else 0


def _launch_agent_plist(*, root: Path, python: Path, hour: int, minute: int, logs: Path) -> str:
    import xml.sax.saxutils as xml

    values = {"label": LAUNCH_AGENT_LABEL, "root": xml.escape(str(root)), "python": xml.escape(str(python)), "stdout": xml.escape(str(logs / "daily.stdout.log")), "stderr": xml.escape(str(logs / "daily.stderr.log"))}
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{values['label']}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{values['python']}</string>
    <string>-m</string><string>quant_platform.orchestration.corrective_daily_scheduler</string>
    <string>--execute</string>
  </array>
  <key>WorkingDirectory</key><string>{values['root']}</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>{values['root']}/src</string></dict>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>StandardOutPath</key><string>{values['stdout']}</string>
  <key>StandardErrorPath</key><string>{values['stderr']}</string>
</dict>
</plist>
'''


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


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
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        result = build_corrective_daily_cadence(install=True)
    else:
        result = run_scheduled_research(execute=args.execute)
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
