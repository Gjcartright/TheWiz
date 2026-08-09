from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd

from quant_platform.wizard_hyperliquid_bridge import build_hyperliquid_wizard_hypothesis_queue


def _write_bundle(root, *, pair: str = "BNB-USD-WLD-USD", asset_y: str = "WLD") -> None:
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": pair,
                "asset_x": "BNB",
                "asset_y": asset_y,
                "venue": "hyperliquid",
                "interval": "1d",
                "history_path": "data/raw/pair_details/hyperliquid.json",
                "history_rows": 500,
                "history_latest_candle_at": "2026-08-05T11:30:00Z",
                "history_ready": True,
                "evidence_path": "reports/active/hyperliquid_research_bundle.csv",
            }
        ]
    ).to_csv(active / "hyperliquid_research_bundle.csv", index=False)
    (active / "wizard_control_plane_summary.json").write_text(
        json.dumps({"ready": True, "status": "ready", "blocker": ""}),
        encoding="utf-8",
    )


def _write_evidence(root, *, source_path: str) -> None:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/WLD-USD",
                "asset_x": "BNB-USD",
                "asset_y": "WLD-USD",
                "interval": "daily",
                "period": 320,
                "exact_mode": "Static (Spread)",
                "sharpe": 2.0,
                "returns_total": 0.20,
                "closed_trades": 5,
                "discovery_config_hash": "a" * 64,
                "candidate_config_hash": "b" * 64,
                "settings_config_hash": "c" * 64,
                "source_path": source_path,
                "evidence_path": source_path,
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)


def test_fresh_complete_mode_capture_is_eligible_for_vendor_proof(tmp_path):
    _write_bundle(tmp_path)
    capture = tmp_path / "reports" / "active" / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "asset_x": "BNB-USD",
                        "asset_y": "WLD-USD",
                        "interval": "daily",
                        "exact_mode": "Static (Spread)",
                        "capture_timestamp_utc": "2026-08-05T11:00:00Z",
                        "spread_type": "Static",
                        "raw_text": "current complete dashboard capture",
                        "backtest_settings": {
                            "entry_level": 2.0,
                            "exit_level": 0.0,
                            "x_weighting": 0.5,
                            "slippage_rate": 0.0005,
                            "commission_rate": 0.0005,
                            "roll_w": 42,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_evidence(tmp_path, source_path="reports/active/capture.json")

    result = build_hyperliquid_wizard_hypothesis_queue(
        root=tmp_path,
        as_of=datetime(2026, 8, 5, 12, tzinfo=timezone.utc),
    )
    frame = pd.read_csv(result.paths["hypothesis_queue"])
    row = frame.iloc[0]

    assert result.summary["vendor_custom_series_eligible"] == 1
    assert bool(row["wizard_control_plane_ready"])
    assert row["discovery_config_hash"] == "a" * 64
    assert row["candidate_config_hash"] == "b" * 64
    assert row["settings_config_hash"] == "c" * 64
    assert row["hypothesis_status"] == "READY_FOR_VENDOR_MODE_PROOF"
    assert row["vendor_request_strategy"] == "Spread"
    assert row["vendor_request_spread_type"] == "Static"
    assert row["entry_level"] == 2.0
    assert row["commission_rate"] == 0.0005
    assert row["roll_w"] == 42.0
    assert bool(row["passes_research_spend_gate"])
    assert row["min_closed_trades_for_proof"] == 5
    assert row["proof_observations"] == 320
    assert len(row["discovery_policy_hash"]) == 64
    assert not row["promotion_allowed"]


def test_stale_failed_capture_is_blocked_even_when_wizard_metrics_are_strong(tmp_path):
    _write_bundle(tmp_path)
    capture = tmp_path / "reports" / "active" / "capture.json"
    capture.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "asset_x": "BNB-USD",
                        "asset_y": "WLD-USD",
                        "interval": "daily",
                        "exact_mode": "Static (Spread)",
                        "capture_timestamp_utc": "2026-07-01T11:00:00Z",
                        "raw_text": "Data retrieval failed. Please refresh your browser.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_evidence(tmp_path, source_path="reports/active/capture.json")

    result = build_hyperliquid_wizard_hypothesis_queue(
        root=tmp_path,
        as_of=datetime(2026, 8, 5, 12, tzinfo=timezone.utc),
    )
    row = pd.read_csv(result.paths["hypothesis_queue"]).iloc[0]

    assert result.summary["vendor_custom_series_eligible"] == 0
    assert row["hypothesis_status"] == "BLOCKED"
    assert "wizard_source_stale_or_timestamp_missing" in row["blocker"]
    assert "wizard_capture_data_retrieval_failed" in row["blocker"]
    assert "wizard_backtest_settings_incomplete" in row["blocker"]


def test_live_dashboard_capture_is_used_but_missing_backtest_settings_still_block(tmp_path):
    _write_bundle(tmp_path, pair="BNB-USD-IO-USD", asset_y="IO")
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [
            {
                "dashboard_rank": 1,
                "pair": "BNB-USD/IO-USD",
                "asset_x": "BNB-USD",
                "asset_y": "IO-USD",
                "wizard_exchange": "DYDX",
                "timeframe": "Daily",
                "exact_mode": "Ou (Spread)",
                "returns_total_pct": 34.8,
                "sharpe": 2.39,
                "updated_at_utc": "2026-Aug-05 22:13:42 UTC",
            }
        ]
    ).to_csv(active / "wizard_dydx_daily_live_workflow.csv", index=False)

    result = build_hyperliquid_wizard_hypothesis_queue(
        root=tmp_path,
        as_of=datetime(2026, 8, 6, 0, tzinfo=timezone.utc),
    )
    row = pd.read_csv(result.paths["hypothesis_queue"]).iloc[0]

    assert row["exact_mode"] == "OU (Spread)"
    assert row["wizard_sharpe"] == 2.39
    assert row["wizard_returns_total_pct"] == 34.8
    assert row["wizard_capture_health"] == "captured_healthy"
    assert "wizard_backtest_settings_incomplete" in row["blocker"]
    assert "wizard_pair_not_in_evidence" not in row["blocker"]
    assert not row["vendor_custom_series_eligible"]


def test_fresh_prescanned_api_fields_override_stale_dashboard_and_leave_costs_blocked(tmp_path):
    _write_bundle(tmp_path, pair="BNB-USD-IO-USD", asset_y="IO")
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [
            {
                "pair": "BNB-USD/IO-USD",
                "asset_x": "BNB-USD",
                "asset_y": "IO-USD",
                "interval": "Daily",
                "period": 320,
                "exact_mode": "OU (Spread)",
                "spread_id": 2,
                "strategy_id": 1,
                "returns_total": 0.348,
                "returns_total_pct": 34.8,
                "sharpe": 2.39,
                "capture_timestamp_utc": "2026-08-05T23:30:00Z",
                "spread_type": "ou",
                "entry_level": 2.0,
                "exit_level": 0.0,
                "x_weighting": 0.5,
                "roll_w": 60,
            }
        ]
    ).to_csv(active / "crypto_wizards_live_scanner_capture.csv", index=False)
    pd.DataFrame(
        [
            {
                "dashboard_rank": 3,
                "pair": "BNB-USD/IO-USD",
                "asset_x": "BNB-USD",
                "asset_y": "IO-USD",
                "wizard_exchange": "DYDX",
                "timeframe": "Daily",
                "exact_mode": "OU (Spread)",
                "returns_total_pct": 20.0,
                "sharpe": 2.0,
                "updated_at_utc": "2026-Aug-04 22:13:42 UTC",
            }
        ]
    ).to_csv(active / "wizard_dydx_daily_live_workflow.csv", index=False)

    result = build_hyperliquid_wizard_hypothesis_queue(
        root=tmp_path,
        as_of=datetime(2026, 8, 6, 0, tzinfo=timezone.utc),
    )
    row = pd.read_csv(result.paths["hypothesis_queue"]).iloc[0]

    assert row["wizard_sharpe"] == 2.39
    assert row["wizard_returns_total_pct"] == 34.8
    assert row["entry_level"] == 2.0
    assert row["exit_level"] == 0.0
    assert row["x_weighting"] == 0.5
    assert row["roll_w"] == 60.0
    assert row["wizard_settings_missing"] == "slippage_rate;commission_rate"
    assert "wizard_backtest_settings_incomplete" in row["blocker"]
    assert row["wizard_source_fresh"]
    assert not row["vendor_custom_series_eligible"]


def test_validated_compact_symbol_capture_joins_hourly_hyperliquid_history(tmp_path):
    _write_bundle(tmp_path, pair="BTCUSDT / EIGENPERP", asset_y="EIGEN")
    active = tmp_path / "reports" / "active"
    bundle = pd.read_csv(active / "hyperliquid_research_bundle.csv")
    bundle.loc[0, "asset_x"] = "BTC"
    bundle.loc[0, "interval"] = "1h"
    bundle.loc[0, "history_latest_candle_at"] = "2026-08-07T14:30:00Z"
    bundle.to_csv(active / "hyperliquid_research_bundle.csv", index=False)
    candidate_hash = "b" * 64
    pd.DataFrame(
        [
            {
                "pair": "BTCUSDT / EIGENPERP",
                "asset_x": "BTCUSDT",
                "asset_y": "EIGENPERP",
                "wizard_exchange": "ByBit",
                "timeframe": "Hourly",
                "period": 365,
                "exact_mode": "static_spread",
                "sharpe": 1.98,
                "returns_total": 0.193,
                "closed_trades": 1,
                "candidate_config_hash": candidate_hash,
                "discovery_config_hash": "a" * 64,
                "source_timestamp": "2026-08-07T12:03:54Z",
            }
        ]
    ).to_csv(active / "wizard_sweep_settings_capture_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "BTCUSDT / EIGENPERP",
                "asset_x": "BTCUSDT",
                "asset_y": "EIGENPERP",
                "interval": "hourly",
                "period": 365,
                "exact_mode": "Static (Spread)",
                "spread_type": "Static",
                "candidate_config_hash": candidate_hash,
                "settings_config_hash": "c" * 64,
                "capture_confirmed": True,
                "backtest_settings_complete": True,
                "capture_timestamp_utc": "2026-08-07T14:16:10Z",
                "capture_evidence_path": "reports/active/live_capture.md",
                "entry_long_value": 2.0,
                "entry_short_value": -2.0,
                "exit_long_value": 0.0,
                "exit_short_value": 0.0,
                "x_weighting": 0.0547,
                "slippage_rate": 0.0005,
                "commission_rate": 0.001,
                "zscore_window": 72,
                "stop_loss_rate": 0.0,
                "exit_n_periods": 0,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture_settings.csv", index=False)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTCUSDT/EIGENPERP",
                "asset_x": "BTCUSDT",
                "asset_y": "EIGENPERP",
                "interval": "hourly",
                "exact_mode": "Static (Spread)",
                "source_timestamp": "2026-08-07T14:16:10Z",
                "source_path": "reports/active/incomplete_normalized_row.md",
            }
        ]
    ).to_csv(processed / "wizard_evidence.csv", index=False)

    result = build_hyperliquid_wizard_hypothesis_queue(
        root=tmp_path,
        as_of=datetime(2026, 8, 7, 15, tzinfo=timezone.utc),
    )
    row = pd.read_csv(result.paths["hypothesis_queue"]).iloc[0]

    assert row["exact_mode"] == "Static (Spread)"
    assert row["vendor_request_spread_type"] == "Static"
    assert row["entry_level"] == 2.0
    assert row["exit_level"] == 0.0
    assert row["roll_w"] == 72.0
    assert row["wizard_closed_trades"] == 1.0
    assert row["wizard_sharpe"] == 1.98
    assert row["wizard_returns_total_pct"] == 19.3
    assert bool(row["passes_wizard_discovery_gate"])
    assert not bool(row["passes_research_spend_gate"])
    assert not bool(row["vendor_custom_series_eligible"])
    assert "insufficient_closed_trades_for_paid_proof" in row["blocker"]
