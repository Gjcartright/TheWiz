"""Cadenced, read-only Hyperliquid depth collection for registered hypotheses."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from quant_platform.active_pipeline import CommandResult
from quant_platform.hyperliquid import refresh_hyperliquid_execution_cost_snapshot
from quant_platform.orchestration.corrective_daily_scheduler import _acquire_lock
from quant_platform.orchestration.corrective_data_evidence import (
    build_cost_collection_status,
    build_l2_capture_candidate_set,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_l2_scheduler.v1"
LAUNCH_AGENT_LABEL = "com.thewiz.corrective-l2-cadence"
DEFAULT_INTERVAL_SECONDS = 10 * 60
DEFAULT_NOTIONAL_USD = 1_000.0
DEFAULT_MINIMUM_SAMPLES = 12
DEFAULT_WINDOW_HOURS = 2.0


def run_corrective_l2_capture(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    collector: Callable[..., CommandResult] = refresh_hyperliquid_execution_cost_snapshot,
) -> CommandResult:
    """Capture one public L2 observation for each eligible registered pair."""

    captured_at = _as_utc(now)
    active = root / "reports" / "active"
    receipts = active / "l2_capture_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    lock_path = active / ".corrective_l2_capture.lock"
    receipt_path = receipts / captured_at.strftime("%Y-%m-%d_%H%M%S.json")
    candidate_result = build_l2_capture_candidate_set(root=root)
    candidate_path = Path(candidate_result["path"])
    eligible_pairs = int(candidate_result["eligible_pairs"])
    blockers: list[str] = []
    capture_summary: dict[str, Any] = {}
    lock_acquired = False
    try:
        _acquire_lock(lock_path, now=captured_at, timeout_seconds=DEFAULT_INTERVAL_SECONDS)
        lock_acquired = True
        if eligible_pairs <= 0:
            blockers.append("no_registered_hyperliquid_l2_candidates")
        else:
            capture = collector(
                root=root,
                max_pairs=eligible_pairs,
                notionals=(DEFAULT_NOTIONAL_USD,),
                captured_at=captured_at,
                candidate_path=candidate_path,
                min_samples=DEFAULT_MINIMUM_SAMPLES,
                window_hours=DEFAULT_WINDOW_HOURS,
            )
            capture_summary = dict(capture.summary)
            if int(capture_summary.get("pairs", 0)) <= 0:
                blockers.append("collector_returned_zero_pairs")
    except FileExistsError as exc:
        blockers.append(str(exc))
    except Exception as exc:
        blockers.append(f"l2_capture_error:{type(exc).__name__}:{exc}")
    finally:
        if lock_acquired:
            lock_path.unlink(missing_ok=True)
    collection = build_cost_collection_status(root=root, now=captured_at)
    status = "PASS" if not blockers else "BLOCKED"
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": captured_at.isoformat(),
        "status": status,
        "registered_hypotheses": int(candidate_result["registered_hypotheses"]),
        "candidate_pairs": int(candidate_result["candidate_pairs"]),
        "eligible_pairs": eligible_pairs,
        "collector_summary": capture_summary,
        "cost_ready_assets": int(collection["ready_assets"]),
        "cost_collecting_assets": int(collection["collecting_assets"]),
        "blockers": blockers,
        "lock_released": not lock_path.exists(),
        "public_read_only_endpoint": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "candidate_evidence_path": _relative(candidate_path, root),
        "cost_status_path": _relative(Path(collection["status_path"]), root),
    }
    receipt["receipt_id"] = "l2receipt_" + sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    _atomic_json(receipt, receipt_path)
    latest_path = active / "corrective_l2_capture_status.json"
    _atomic_json({**receipt, "receipt_path": _relative(receipt_path, root)}, latest_path)
    return CommandResult(
        paths={
            "capture_receipt": receipt_path,
            "latest_status": latest_path,
            "candidate_set": candidate_path,
            "cost_collection_status": Path(collection["status_path"]),
        },
        summary=receipt,
    )


def install_corrective_l2_launch_agent(
    *, root: Path = ROOT, interval_seconds: int = DEFAULT_INTERVAL_SECONDS
) -> dict[str, Any]:
    """Install, but do not bootstrap, the public-depth LaunchAgent."""

    if interval_seconds < 60:
        raise ValueError("L2 collection interval must be at least 60 seconds")
    python = root / ".venv312" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"scheduler Python missing: {python}")
    agent_path = Path.home() / "Library" / "LaunchAgents" / f"{LAUNCH_AGENT_LABEL}.plist"
    agent_path.parent.mkdir(parents=True, exist_ok=True)
    logs = root / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = _launch_agent_plist(
        root=root,
        python=python,
        logs=logs,
        interval_seconds=interval_seconds,
    )
    temporary = agent_path.with_suffix(".plist.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(agent_path)
    return {
        "status": "INSTALLED_NOT_STARTED",
        "label": LAUNCH_AGENT_LABEL,
        "plist": agent_path,
        "interval_seconds": interval_seconds,
        "public_read_only_endpoint": True,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _launch_agent_plist(
    *, root: Path, python: Path, logs: Path, interval_seconds: int
) -> str:
    import xml.sax.saxutils as xml

    values = {
        "label": LAUNCH_AGENT_LABEL,
        "root": xml.escape(str(root)),
        "python": xml.escape(str(python)),
        "stdout": xml.escape(str(logs / "l2.stdout.log")),
        "stderr": xml.escape(str(logs / "l2.stderr.log")),
    }
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{values['label']}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{values['python']}</string>
    <string>-m</string><string>quant_platform.orchestration.corrective_l2_scheduler</string>
    <string>--capture</string>
  </array>
  <key>WorkingDirectory</key><string>{values['root']}</string>
  <key>EnvironmentVariables</key><dict><key>PYTHONPATH</key><string>{values['root']}/src</string></dict>
  <key>StartInterval</key><integer>{interval_seconds}</integer>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
  <key>LowPriorityIO</key><true/>
  <key>StandardOutPath</key><string>{values['stdout']}</string>
  <key>StandardErrorPath</key><string>{values['stderr']}</string>
</dict>
</plist>
'''


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    if args.install:
        result: Any = install_corrective_l2_launch_agent()
    else:
        result = run_corrective_l2_capture()
        result = {
            "summary": result.summary,
            "paths": {key: str(value) for key, value in result.paths.items()},
        }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
