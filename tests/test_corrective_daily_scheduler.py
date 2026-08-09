from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from quant_platform.orchestration.corrective_daily_scheduler import (
    _acquire_lock,
    build_daily_cadence_acceptance,
    run_daily_cadence_fault_tests,
)


NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)


def test_overlap_lock_fails_closed_and_stale_lock_recovers(tmp_path):
    lock = tmp_path / "lock"
    _acquire_lock(lock, now=NOW, timeout_seconds=60)
    try:
        _acquire_lock(lock, now=NOW, timeout_seconds=60)
    except FileExistsError:
        pass
    else:
        raise AssertionError("overlapping run must be blocked")
    lock.write_text(json.dumps({"started_at_utc": "2020-01-01T00:00:00+00:00"}))
    _acquire_lock(lock, now=NOW, timeout_seconds=60)


def test_fault_suite_revokes_current_status(tmp_path):
    frame = run_daily_cadence_fault_tests(root=tmp_path, now=NOW)
    assert frame["status"].eq("PASS").all()
    assert not frame.loc[frame["failure_injected"], "research_board_current"].any()
    assert not frame["stale_output_actionable"].any()


def test_seven_distinct_consecutive_receipts_are_required(tmp_path):
    receipts = tmp_path / "reports" / "active" / "daily_schedule_receipts"
    receipts.mkdir(parents=True)
    for day in range(1, 8):
        payload = {"run_date": f"2026-08-{day:02d}", "receipt_id": str(day), "run_status": "PASS", "research_board_current": True, "testnet_order_authority": False, "live_trading_authorized": False}
        (receipts / f"2026-08-{day:02d}.json").write_text(json.dumps(payload))
    path = build_daily_cadence_acceptance(root=tmp_path, now=NOW)
    frame = pd.read_csv(path)
    assert frame["consecutive_complete_cycles"].max() == 7
    assert frame.iloc[-1]["cadence_acceptance_status"] == "PASS"
