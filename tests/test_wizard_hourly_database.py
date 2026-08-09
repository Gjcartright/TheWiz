from pathlib import Path

from quant_platform.wizard_hourly_database import build_hourly_database, build_hourly_database_from_live_scanner


def _row(strategy: str, *, timeframe: str = "Daily", annualized: str = "10.0%", sharpe: str = "1.0", trades: str = "3", best_fit: str = "clayton", conditional_chart: str = "71.9%"):
    return {
        "pair": "BNB-USD/STX-USD",
        "periods": "320",
        "timeframe": timeframe,
        "strategy": strategy,
        "best_fit": best_fit,
        "conditional_chart": conditional_chart,
        "sharpe_metric": sharpe,
        "sortino_metric": "2.0",
        "net_return_metric": "8.0%",
        "annualized_return_metric": annualized,
        "win_rate_metric": "100.0%",
        "closed_trades_metric": trades,
        "max_drawdown_metric": "-4.0%",
        "top_strip": {"corr": "83.2%", "hurst": "0.95", "half_life": "73", "returns": "1.2%", "sharpe": "0.31"},
    }


def test_build_hourly_database_writes_append_and_alias(tmp_path: Path):
    current = [_row("Static (Spread)"), _row("Copula", annualized="3.7%", sharpe="0.62", trades="2")]
    paths = build_hourly_database(current_rows=current, output_dir=tmp_path, scan_timestamp="2026-07-03T20:00:00+00:00")

    assert paths["variant_database"].exists()
    assert paths["current_alias"].exists()
    assert paths["review_queue"].exists()
    assert paths["candidate_queue"].exists()
    assert paths["pair_queue"].exists()
    assert paths["strategy_queue"].exists()


def test_build_hourly_database_marks_changed_rows_on_second_scan(tmp_path: Path):
    build_hourly_database(
        current_rows=[_row("Static (Spread)", annualized="10.0%", sharpe="1.0")],
        output_dir=tmp_path,
        scan_timestamp="2026-07-03T20:00:00+00:00",
    )
    paths = build_hourly_database(
        current_rows=[_row("Static (Spread)", annualized="12.0%", sharpe="1.2")],
        output_dir=tmp_path,
        scan_timestamp="2026-07-03T21:00:00+00:00",
    )

    review_text = paths["review_queue"].read_text(encoding="utf-8")
    candidate_text = paths["candidate_queue"].read_text(encoding="utf-8")

    assert "changed" in review_text
    assert "review_now" in candidate_text


def test_build_hourly_database_blocks_high_anomaly_candidates(tmp_path: Path):
    current = [_row("", timeframe="", best_fit="", conditional_chart="")]
    paths = build_hourly_database(current_rows=current, output_dir=tmp_path, scan_timestamp="2026-07-03T20:00:00+00:00")

    anomalies = paths["anomalies"].read_text(encoding="utf-8")
    candidates = paths["candidate_queue"].read_text(encoding="utf-8")

    assert "blank_variant_fields" in anomalies
    assert "blocked_by_anomaly" in candidates


def test_build_hourly_database_dedupes_same_scan_timestamp(tmp_path: Path):
    build_hourly_database(
        current_rows=[_row("Static (Spread)")],
        output_dir=tmp_path,
        scan_timestamp="2026-07-03T20:00:00+00:00",
    )
    paths = build_hourly_database(
        current_rows=[_row("Static (Spread)")],
        output_dir=tmp_path,
        scan_timestamp="2026-07-03T20:00:00+00:00",
    )

    variant_rows = paths["variant_database"].read_text(encoding="utf-8").strip().splitlines()
    assert len(variant_rows) == 2


def test_build_hourly_database_pair_queue_marks_top_hourly_targets(tmp_path: Path):
    current = [
        _row("Static (Spread)", annualized="12.0%", sharpe="1.2"),
        _row("Copula", annualized="3.7%", sharpe="0.62", trades="2"),
        {
            **_row("OU (Spread)", timeframe="4 Hour", annualized="17.0%", sharpe="1.5"),
            "pair": "SOL-USD/HYPE-USD",
        },
    ]
    paths = build_hourly_database(current_rows=current, output_dir=tmp_path, scan_timestamp="2026-07-03T20:00:00+00:00", hourly_pair_limit=2)

    pair_queue = paths["pair_queue"].read_text(encoding="utf-8")

    assert "BNB-USD/STX-USD" in pair_queue
    assert "SOL-USD/HYPE-USD" in pair_queue
    assert "recommended_this_hour" in pair_queue


def test_build_hourly_database_from_live_scanner_replaces_stale_pair_queue(tmp_path: Path):
    output = tmp_path / "reports" / "active" / "wizard_hourly_database"
    output.mkdir(parents=True)
    build_hourly_database(
        current_rows=[_row("OU (Spread)", timeframe="Live", annualized="67.5%", sharpe="1.4")],
        output_dir=output,
        scan_timestamp="2026-07-04T13:20:00+00:00",
        source_name="wizard_scanner_live_dom",
    )

    collected = tmp_path / "data" / "collected" / "wizard_scanner_static_spread_visible_live"
    collected.mkdir(parents=True)
    (collected / "wizard_scanner_static_spread_visible_live_latest.json").write_text(
        """
        {
          "captured_at": "2026-07-04T19:52:57.957Z",
          "scanner_filters": {"strategy": "Static (Spread)"},
          "rows": [
            {"pair": "EIGEN-USD / XRP-USD", "pair_x": "EIGEN-USD", "pair_y": "XRP-USD", "reward_sharpe": "2.04"},
            {"pair": "DOGE-USD / SYRUP-USD", "pair_x": "DOGE-USD", "pair_y": "SYRUP-USD", "reward_sharpe": "2.03"}
          ]
        }
        """,
        encoding="utf-8",
    )

    paths = build_hourly_database_from_live_scanner(root=tmp_path, output_dir=output, hourly_pair_limit=2)

    pair_queue = paths["pair_queue"].read_text(encoding="utf-8")
    candidate_queue = paths["candidate_queue"].read_text(encoding="utf-8")

    assert "EIGEN-USD/XRP-USD" in pair_queue
    assert "DOGE-USD/SYRUP-USD" in pair_queue
    assert "BNB-USD/STX-USD" not in pair_queue
    assert "Static (Spread)" in candidate_queue


def test_build_hourly_database_from_live_scanner_merges_multiple_strategy_feeds(tmp_path: Path):
    output = tmp_path / "reports" / "active" / "wizard_hourly_database"
    output.mkdir(parents=True)

    static_dir = tmp_path / "data" / "collected" / "wizard_scanner_static_spread_visible_live"
    static_dir.mkdir(parents=True)
    (static_dir / "wizard_scanner_static_spread_visible_live_latest.json").write_text(
        """
        {
          "captured_at": "2026-07-04T19:52:57.957Z",
          "scanner_filters": {"strategy": "Static (Spread)"},
          "rows": [
            {"pair": "EIGEN-USD / XRP-USD", "pair_x": "EIGEN-USD", "pair_y": "XRP-USD", "reward_sharpe": "2.04"}
          ]
        }
        """,
        encoding="utf-8",
    )

    dyn_dir = tmp_path / "data" / "collected" / "wizard_scanner_dynamic_spread_visible_live"
    dyn_dir.mkdir(parents=True)
    (dyn_dir / "wizard_scanner_dynamic_spread_visible_live_latest.json").write_text(
        """
        {
          "captured_at": "2026-07-04T19:58:57.957Z",
          "scanner_filters": {"strategy": "Dynamic (Spread)"},
          "rows": [
            {"pair": "DOGE-USD / SYRUP-USD", "pair_x": "DOGE-USD", "pair_y": "SYRUP-USD", "reward_sharpe": "2.22"}
          ]
        }
        """,
        encoding="utf-8",
    )

    paths = build_hourly_database_from_live_scanner(root=tmp_path, output_dir=output, hourly_pair_limit=2)

    candidate_queue = paths["candidate_queue"].read_text(encoding="utf-8")
    strategy_queue = paths["strategy_queue"].read_text(encoding="utf-8")

    assert "Static (Spread)" in candidate_queue
    assert "Dynamic (Spread)" in candidate_queue
    assert "Static (Spread)" in strategy_queue
    assert "Dynamic (Spread)" in strategy_queue
