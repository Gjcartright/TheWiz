from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_l2_scheduler import (
    _launch_agent_plist,
    run_corrective_l2_capture,
)


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _write_candidate_inputs(root: Path) -> None:
    active = root / "reports" / "active"
    processed = root / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [{"experiment_id": "exp-1", "confirmation_role": "near_miss_remediation"}]
    ).to_csv(active / "current_hypothesis_batch.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "exp-1",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "overall_research_rank": 1,
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_failure_attribution.csv", index=False)
    pd.DataFrame(
        [{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)
    pd.DataFrame(
        [
            {"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500},
            {"asset": "WIF", "funding_status": "COMPLETE", "funding_rows": 500},
        ]
    ).to_csv(active / "current_wizard_hyperliquid_funding_asset_results.csv", index=False)


def test_capture_is_public_read_only_and_writes_receipt(tmp_path):
    _write_candidate_inputs(tmp_path)
    calls = []

    def collector(**kwargs):
        calls.append(kwargs)
        return CommandResult(paths={}, summary={"pairs": 1, "slippage_models_ready": 0})

    result = run_corrective_l2_capture(root=tmp_path, now=NOW, collector=collector)

    assert result.summary["status"] == "PASS"
    assert result.summary["eligible_pairs"] == 1
    assert result.summary["order_submission_included"] is False
    assert result.summary["testnet_order_authority"] is False
    assert calls[0]["notionals"] == (1000.0,)
    assert calls[0]["min_samples"] == 12
    assert result.paths["capture_receipt"].exists()


def test_launch_agent_uses_ten_minute_read_only_cadence(tmp_path):
    python = tmp_path / ".venv312" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    logs = tmp_path / "reports" / "active" / "schedule_logs"
    logs.mkdir(parents=True)

    plist = _launch_agent_plist(
        root=tmp_path,
        python=python,
        logs=logs,
        interval_seconds=600,
    )

    assert "<key>StartInterval</key><integer>600</integer>" in plist
    assert "corrective_l2_scheduler" in plist
    assert "--capture" in plist
    assert "--execute" not in plist
    assert "order" not in plist.lower()
