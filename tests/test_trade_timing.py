from __future__ import annotations

import pandas as pd

from quant_platform import cli
from quant_platform.trade_timing import trade_timing_comparison_report_frame, trade_timing_comparison_summary


def test_trade_timing_comparison_finds_z2_entry_and_mean_exit():
    history = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "pair": "ETH-BTC", "spread": 0.0, "zscore": 0.2},
            {"timestamp": "2026-01-01T00:05:00Z", "pair": "ETH-BTC", "spread": 1.0, "zscore": 1.2},
            {"timestamp": "2026-01-01T00:10:00Z", "pair": "ETH-BTC", "spread": 2.0, "zscore": 2.1},
            {"timestamp": "2026-01-01T00:15:00Z", "pair": "ETH-BTC", "spread": 1.4, "zscore": 1.4},
            {"timestamp": "2026-01-01T00:20:00Z", "pair": "ETH-BTC", "spread": 0.2, "zscore": 0.0},
            {"timestamp": "2026-01-01T00:25:00Z", "pair": "ETH-BTC", "spread": 0.1, "zscore": -0.2},
        ]
    )
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    trades = pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "pair": "ETH-BTC",
                "side": "short_spread",
                "entry_timestamp_current": "2026-01-01T00:15:00Z",
                "exit_timestamp_current": "2026-01-01T00:25:00Z",
            }
        ]
    )

    report = trade_timing_comparison_report_frame(trades, history)

    row = report.iloc[0]
    assert row["entry_timestamp_signal_z2"] == "2026-01-01T00:10:00+00:00"
    assert row["exit_timestamp_signal_mean"] == "2026-01-01T00:20:00+00:00"
    assert row["entry_delay_minutes_current_minus_signal"] == 5.0
    assert row["exit_delay_minutes_current_minus_signal"] == 5.0
    assert row["entry_zscore_signal_z2"] == 2.1
    assert row["exit_zscore_signal_mean"] == 0.0

    summary = trade_timing_comparison_summary(report)
    assert summary.loc[0, "trades"] == 1
    assert summary.loc[0, "entry_signals_found"] == 1
    assert summary.loc[0, "late_entries"] == 1


def test_cli_trade_timing_comparison_report_writes_detail_and_summary(tmp_path):
    trades_path = tmp_path / "trades.csv"
    history_path = tmp_path / "history.csv"
    output_path = tmp_path / "trade_timing_report.csv"

    pd.DataFrame(
        [
            {
                "trade_id": "t1",
                "pair": "ETH-BTC",
                "side": "short_spread",
                "entry_timestamp_current": "2026-01-01T00:15:00Z",
                "exit_timestamp_current": "2026-01-01T00:25:00Z",
            }
        ]
    ).to_csv(trades_path, index=False)
    pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "pair": "ETH-BTC", "spread": 0.0, "zscore": 0.2},
            {"timestamp": "2026-01-01T00:05:00Z", "pair": "ETH-BTC", "spread": 1.0, "zscore": 1.2},
            {"timestamp": "2026-01-01T00:10:00Z", "pair": "ETH-BTC", "spread": 2.0, "zscore": 2.1},
            {"timestamp": "2026-01-01T00:15:00Z", "pair": "ETH-BTC", "spread": 1.4, "zscore": 1.4},
            {"timestamp": "2026-01-01T00:20:00Z", "pair": "ETH-BTC", "spread": 0.2, "zscore": 0.0},
            {"timestamp": "2026-01-01T00:25:00Z", "pair": "ETH-BTC", "spread": 0.1, "zscore": -0.2},
        ]
    ).to_csv(history_path, index=False)

    report = cli.trade_timing_comparison_report(trades_path, history_path, output_path)

    assert len(report) == 1
    assert output_path.exists()
    summary_path = tmp_path / "trade_timing_report_summary.csv"
    assert summary_path.exists()

    written = pd.read_csv(output_path, dtype=str).fillna("")
    assert written.loc[0, "entry_timestamp_signal_z2"] == "2026-01-01T00:10:00+00:00"
