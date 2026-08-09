"""Fail-closed executor for the validated 19-stage daily research cadence."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Callable

import pandas as pd

from quant_platform.active_pipeline import CommandResult, _write_csv, _write_json, _write_text
from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
    STAGES,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_daily_runner.v1"


def run_current_wizard_hyperliquid_daily_pipeline(
    *,
    root: Path = ROOT,
    execute: bool = False,
    now: datetime | None = None,
    minimum_free_bytes: int = MINIMUM_FREE_BYTES,
    available_disk_bytes: int | None = None,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> CommandResult:
    """Plan or execute the research cadence; never submit Testnet or live orders."""

    as_of = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    run_id = "cwdaily_" + sha256(
        f"{as_of.isoformat()}|{execute}|{minimum_free_bytes}".encode("utf-8")
    ).hexdigest()[:20]
    active = root / "reports" / "active"
    run_dir = root / "reports" / "runs" / "current_wizard_hyperliquid_daily" / run_id
    active.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=False)
    runner = command_runner or subprocess.run
    rows: list[dict[str, object]] = []
    halted = False
    for sequence, stage, cadence, _, _, _ in STAGES:
        free_bytes = _free_bytes(root, available_disk_bytes)
        storage_ready = free_bytes >= minimum_free_bytes
        command, command_blocker = _stage_command(root, stage)
        status = "PLANNED"
        blocker = ""
        return_code: int | None = None
        output_hash = ""
        started_at = ""
        completed_at = ""
        if halted:
            status = "NOT_STARTED_DEPENDENCY"
            blocker = "prior_stage_not_successful"
        elif not storage_ready:
            status = "BLOCKED_STORAGE" if sequence == 1 else "NOT_STARTED_STORAGE"
            blocker = (
                f"insufficient_free_space:free_bytes={free_bytes};"
                f"required_bytes={minimum_free_bytes}"
            )
            halted = True
        elif command_blocker:
            status = "BLOCKED_INPUT"
            blocker = command_blocker
            halted = True
        elif execute:
            started = datetime.now(timezone.utc)
            started_at = started.isoformat()
            env = os.environ.copy()
            source_path = str(root / "src")
            env["PYTHONPATH"] = (
                source_path
                if not env.get("PYTHONPATH")
                else source_path + os.pathsep + env["PYTHONPATH"]
            )
            try:
                result = runner(
                    command,
                    cwd=root,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return_code = int(result.returncode)
                output = (result.stdout or "") + (result.stderr or "")
                output_hash = sha256(output.encode("utf-8")).hexdigest()
                if return_code == 0:
                    status = "PASS"
                else:
                    status = "FAILED"
                    blocker = f"stage_command_exit_code:{return_code}"
                    halted = True
            except Exception as exc:
                status = "FAILED"
                blocker = f"stage_command_error:{type(exc).__name__}"
                halted = True
            completed_at = datetime.now(timezone.utc).isoformat()
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "daily_run_id": run_id,
                "sequence": sequence,
                "stage": stage,
                "cadence": cadence,
                "status": status,
                "blocker": blocker,
                "free_disk_bytes_before_stage": free_bytes,
                "minimum_free_disk_bytes": minimum_free_bytes,
                "resolved_command": shlex.join(command),
                "return_code": return_code,
                "output_sha256": output_hash,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
                "execution_requested": execute,
                "order_submission_authority": False,
                "live_trading_authorized": False,
            }
        )

    frame = pd.DataFrame(rows)
    if frame["status"].eq("PASS").all():
        run_status = "PASS"
    elif frame["status"].eq("PLANNED").all():
        run_status = "PLANNED"
    elif frame["status"].isin({"BLOCKED_STORAGE", "NOT_STARTED_STORAGE"}).any():
        run_status = "BLOCKED_STORAGE"
    elif frame["status"].eq("FAILED").any():
        run_status = "FAILED"
    else:
        run_status = "BLOCKED"
    summary = {
        "schema_version": SCHEMA_VERSION,
        "daily_run_id": run_id,
        "as_of": as_of.isoformat(),
        "run_status": run_status,
        "execution_requested": execute,
        "stages": len(frame),
        "stages_passed": int(frame["status"].eq("PASS").sum()),
        "stages_planned": int(frame["status"].eq("PLANNED").sum()),
        "stages_blocked_or_not_started": int(
            frame["status"].isin(
                {
                    "BLOCKED_STORAGE",
                    "NOT_STARTED_STORAGE",
                    "BLOCKED_INPUT",
                    "NOT_STARTED_DEPENDENCY",
                    "FAILED",
                }
            ).sum()
        ),
        "minimum_free_disk_bytes": minimum_free_bytes,
        "final_free_disk_bytes": _free_bytes(root, available_disk_bytes),
        "testnet_execution_included": False,
        "order_submission_authority": False,
        "live_trading_authorized": False,
    }
    active_csv = active / "current_wizard_hyperliquid_daily_run_status.csv"
    active_manifest = active / "current_wizard_hyperliquid_daily_run_manifest.json"
    active_md = active / "current_wizard_hyperliquid_daily_run_summary.md"
    run_csv = run_dir / "daily_run_status.csv"
    run_manifest = run_dir / "manifest.json"
    _atomic_csv(frame, active_csv)
    _atomic_csv(frame, run_csv)
    _atomic_json(summary, active_manifest)
    _atomic_json(summary, run_manifest)
    _atomic_text(_markdown(summary, frame), active_md)
    return CommandResult(
        paths={
            "daily_run_status": active_csv,
            "daily_run_manifest": active_manifest,
            "daily_run_summary": active_md,
            "dated_daily_run_status": run_csv,
            "dated_daily_run_manifest": run_manifest,
        },
        summary=summary,
    )


def _stage_command(root: Path, stage: str) -> tuple[list[str], str]:
    prefix = [sys.executable, "-m", "quant_platform.cli"]
    commands: dict[str, list[str]] = {
        "storage_preflight": ["system-check"],
        "wizard_exhaustive_discovery": [
            "crypto-wizards-full-sweep",
            "--execute-wizard-sweep",
        ],
        "wizard_refresh_accounting": ["build-exhaustive-wizard-api-refresh-delta"],
        "hyperliquid_market_inventory": ["hyperliquid-testnet-market-inventory"],
        "pair_mode_orientation_handoff": ["build-current-wizard-hyperliquid-handoff"],
        "canonical_one_x_replay": ["run-current-wizard-hyperliquid-canonical-replay"],
        "observed_cost_replay": ["run-current-wizard-hyperliquid-observed-cost-replay"],
        "purged_walkforward": ["run-current-wizard-hyperliquid-walkforward"],
        "ou_optimal_outcome_stratification": ["build-current-wizard-ou-optimal-overlay"],
        "causal_regime_attribution": ["build-current-wizard-hyperliquid-regime-attribution"],
        "robustness_stress": ["run-current-wizard-hyperliquid-robustness"],
        "cross_cell_concentration": ["build-current-wizard-hyperliquid-concentration"],
        "failure_attribution": ["build-current-wizard-hyperliquid-failure-attribution"],
        "conditional_leverage_surface": ["build-current-wizard-hyperliquid-leverage-surface"],
        "dated_learning_ledger": ["build-current-wizard-hyperliquid-learning-ledger"],
        "frozen_chain_validation": ["validate-current-wizard-hyperliquid-chain"],
        "monitor_dashboard": [
            "build-command-dashboard",
            "--dashboard-refresh-profile",
            "monitor",
        ],
    }
    if stage in {"point_in_time_history", "funding_liquidity_cost_evidence"}:
        pair_keys = _ready_pair_group_keys(root)
        if not pair_keys:
            return prefix, "ready_pair_group_keys_missing"
        command = (
            "materialize-current-wizard-hyperliquid-history"
            if stage == "point_in_time_history"
            else "materialize-current-wizard-hyperliquid-cost-evidence"
        )
        args = [command, "--current-pair-group-keys", ",".join(pair_keys)]
        if stage == "point_in_time_history":
            args.extend(["--minimum-free-disk-mib", str(MINIMUM_FREE_BYTES // 1024**2)])
        return [*prefix, *args], ""
    args = commands.get(stage)
    return ([*prefix, *args], "") if args is not None else (prefix, "unknown_daily_stage")


def _ready_pair_group_keys(root: Path) -> tuple[str, ...]:
    path = root / "reports" / "active" / "current_wizard_hyperliquid_pair_history_queue.csv"
    try:
        frame = pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        return ()
    if not {"pair_group_key", "history_request_status"}.issubset(frame.columns):
        return ()
    ready = frame.loc[
        frame["history_request_status"].astype(str).eq("READY_TO_FETCH"),
        "pair_group_key",
    ]
    return tuple(sorted(set(ready.astype(str).str.strip()) - {""}))


def _free_bytes(root: Path, available_disk_bytes: int | None) -> int:
    return int(
        available_disk_bytes
        if available_disk_bytes is not None
        else shutil.disk_usage(root).free
    )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _write_csv(frame, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    _write_json(path, payload)


def _atomic_text(text: str, path: Path) -> None:
    _write_text(path, text)


def _markdown(summary: dict[str, object], frame: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Daily Run",
            "",
            f"- Run: `{summary['daily_run_id']}`",
            f"- Status: `{summary['run_status']}`",
            f"- Execution requested: `{summary['execution_requested']}`",
            "- Testnet execution included: `False`",
            "- Live trading authorized: `False`",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )
