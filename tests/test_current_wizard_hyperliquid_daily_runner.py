from __future__ import annotations

from datetime import datetime, timezone
import subprocess

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_cadence import (
    MINIMUM_FREE_BYTES,
)
from quant_platform.orchestration.current_wizard_hyperliquid_daily_runner import (
    run_current_wizard_hyperliquid_daily_pipeline,
)


def _write_pair_queue(root):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_group_key": "binance|daily|BTC|ETH",
                "history_request_status": "READY_TO_FETCH",
            },
            {
                "pair_group_key": "binance|daily|SOL|WLD",
                "history_request_status": "READY_TO_FETCH",
            },
            {
                "pair_group_key": "blocked|pair",
                "history_request_status": "BLOCKED",
            },
        ]
    ).to_csv(active / "current_wizard_hyperliquid_pair_history_queue.csv", index=False)


def test_daily_runner_blocks_before_any_command_when_storage_is_low(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def forbidden_runner(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("no command may run below the storage floor")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
        available_disk_bytes=MINIMUM_FREE_BYTES - 1,
        command_runner=forbidden_runner,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "BLOCKED_STORAGE"
    assert len(frame) == 19
    assert frame.iloc[0]["status"] == "BLOCKED_STORAGE"
    assert frame.iloc[1:]["status"].eq("NOT_STARTED_DEPENDENCY").all()
    assert calls == []
    assert frame["order_submission_authority"].astype(str).str.lower().eq("false").all()
    assert frame["live_trading_authorized"].astype(str).str.lower().eq("false").all()


def test_daily_runner_executes_all_research_stages_and_resolves_pair_keys(tmp_path):
    _write_pair_queue(tmp_path)
    calls = []

    def successful_runner(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        execute=True,
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
        command_runner=successful_runner,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "PASS"
    assert len(calls) == 19
    assert frame["status"].eq("PASS").all()
    history = next(command for command in calls if "materialize-current-wizard-hyperliquid-history" in command)
    keys = history[history.index("--current-pair-group-keys") + 1]
    assert keys == "binance|daily|BTC|ETH,binance|daily|SOL|WLD"
    assert "blocked|pair" not in keys
    flattened = " ".join(" ".join(command) for command in calls).lower()
    assert "submit-pair" not in flattened
    assert "sign-smoke-approval" not in flattened
    assert "live" not in flattened


def test_daily_runner_default_is_plan_only(tmp_path):
    _write_pair_queue(tmp_path)

    result = run_current_wizard_hyperliquid_daily_pipeline(
        root=tmp_path,
        now=datetime(2026, 8, 8, 14, 0, tzinfo=timezone.utc),
        available_disk_bytes=MINIMUM_FREE_BYTES + 1,
    )
    frame = pd.read_csv(result.paths["daily_run_status"], keep_default_na=False)

    assert result.summary["run_status"] == "PLANNED"
    assert result.summary["execution_requested"] is False
    assert frame["status"].eq("PLANNED").all()
