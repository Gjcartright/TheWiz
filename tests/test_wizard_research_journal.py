from pathlib import Path

import pandas as pd

from quant_platform.wizard_research_journal import (
    _apply_stationarity_and_copula_interpretation,
    _attach_wizard_configuration_lineage,
    _build_copula_detail_capture_queue,
    _build_detail_records,
    _build_strategy_mode_capture_queue,
    _build_two_hour_copula_report,
)


def test_detail_records_preserve_top_metrics_separately_from_detail_metrics(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "interval": "Daily",
                "exact_mode": "Static Spread",
                "correlation_top": 0.81,
                "pearson": 0.72,
                "corr_copula": 0.63,
                "return_total_top": 28.0,
                "returns_total": 0.19,
                "returns_total_pct": 19.0,
                "sharpe_top": 2.4,
                "sharpe": 1.8,
            }
        ]
    ).to_csv(active / "crypto_wizards_pair_page_capture.csv", index=False)

    detail = _build_detail_records(
        [],
        tmp_path,
        pd.DataFrame(),
        {},
        {},
        {},
        {},
        {},
    )
    row = detail.loc[detail["timeframe"].eq("Daily")].iloc[0]

    assert row["correlation_top"] == 0.81
    assert row["correlation_value"] == 0.63
    assert row["return_total_top"] == 28.0
    assert row["annualized_return_detail"] == 19.0
    assert row["sharpe_top"] == 2.4
    assert row["sharpe_detail"] == 1.8


def test_journal_lineage_joins_exact_mode_configuration_without_inventing_old_hashes(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair": "BTC-USD / ETH-USD",
                "timeframe": "Daily",
                "exact_mode": "static_spread",
                "discovery_config_hash": "a" * 64,
                "candidate_config_hash": "b" * 64,
            }
        ]
    ).to_csv(active / "wizard_sweep_settings_capture_queue.csv", index=False)
    source = pd.DataFrame(
        [
            {
                "pair": "BTC-USD/ETH-USD",
                "timeframe": "Daily",
                "strategy_label": "Static (Spread)",
            },
            {
                "pair": "SOL-USD/AVAX-USD",
                "timeframe": "Daily",
                "strategy_label": "Static (Spread)",
            },
        ]
    )

    result = _attach_wizard_configuration_lineage(source, tmp_path)

    assert result.loc[0, "config_schema_version"] == "wizard_run_config.v1"
    assert result.loc[0, "discovery_config_hash"] == "a" * 64
    assert result.loc[0, "candidate_config_hash"] == "b" * 64
    assert result.loc[1, "candidate_config_hash"] == ""


def test_copula_interpretation_marks_orange_engle_granger_as_research_only():
    record = {
        "asset_x": "ARB-USD",
        "asset_y": "DOGE-USD",
        "copula_x_given_y": 0.993,
        "copula_y_given_x": 0.001,
        "johansen_badge_state": "not_confirmed",
        "engle_granger_badge_state": "trending",
        "volume_x_top": 29000,
        "volume_y_top": 173600,
    }

    _apply_stationarity_and_copula_interpretation(record)

    assert record["stationarity_summary"] == "engle_granger_trending"
    assert record["copula_signal_status"] == "strong_asymmetric_dislocation"
    assert record["copula_rich_asset"] == "ARB-USD"
    assert record["copula_cheap_asset"] == "DOGE-USD"
    assert record["copula_trade_direction"] == "short_x_long_y"
    assert record["copula_journal_status"] == "research_only"
    assert record["copula_execution_blockers"] == "engle_granger_trending"


def test_two_hour_copula_report_joins_scanner_and_detail_context(tmp_path: Path):
    scanner = {
        "journal_layer": "scanner_capture",
        "capture_timestamp_utc": "2026-07-14T22:57:59Z",
        "pair": "ARB-USD-DOGE-USD",
        "asset_x": "ARB-USD",
        "asset_y": "DOGE-USD",
        "venue": "dydx",
        "strategy_label": "OU (Spread)",
        "volume_x": 29000,
        "volume_y": 173600,
        "dependency_x_over_y": 0.984,
        "dependency_y_over_x": 0.004,
        "correlation_value": 0.966,
        "johansen_badge_state": "not_confirmed",
        "engle_granger_badge_state": "trending",
        "hurst_value": 0.82,
        "half_life_value": 9.1,
        "scanner_copula_filter": "Copula (arbitrage)",
    }
    detail = {
        "journal_layer": "pair_detail_capture",
        "capture_timestamp_utc": "2026-07-14T23:00:00Z",
        "detail_capture_timestamp_utc": "2026-07-14T23:00:00Z",
        "pair": "ARB-USD-DOGE-USD",
        "asset_x": "ARB-USD",
        "asset_y": "DOGE-USD",
        "venue": "dydx",
        "timeframe": "Daily",
        "periods_analyzed": 320,
        "strategy_label": "OU (Spread)",
        "copula_best_fit": "studentt",
        "copula_correlation_rho": 0.966,
        "copula_x_given_y": 0.993,
        "copula_y_given_x": 0.001,
        "johansen_badge_state": "not_confirmed",
        "engle_granger_badge_state": "trending",
        "volume_x_top": 29000,
        "volume_y_top": 173600,
    }
    _apply_stationarity_and_copula_interpretation(scanner)
    _apply_stationarity_and_copula_interpretation(detail)

    csv_path, md_path, rows = _build_two_hour_copula_report(pd.DataFrame([scanner, detail]), tmp_path)
    report = pd.read_csv(csv_path)

    assert rows == 1
    assert md_path.exists()
    assert report.loc[0, "copula_trade_direction"] == "short_x_long_y"
    assert report.loc[0, "stationarity_summary"] == "engle_granger_trending"
    assert report.loc[0, "copula_journal_status"] == "research_only"


def test_two_hour_copula_report_does_not_cross_timeframes_when_scanner_names_one(tmp_path: Path):
    scanner = {
        "journal_layer": "scanner_capture",
        "capture_timestamp_utc": "2026-07-14T22:57:59Z",
        "pair": "ARB-USD-DOGE-USD",
        "timeframe": "5 Min",
        "scanner_copula_filter": "Copula (arbitrage)",
    }
    daily_detail = {
        "journal_layer": "pair_detail_capture",
        "capture_timestamp_utc": "2026-07-14T23:00:00Z",
        "pair": "ARB-USD-DOGE-USD",
        "timeframe": "Daily",
        "copula_x_given_y": 0.993,
        "copula_y_given_x": 0.001,
    }
    _apply_stationarity_and_copula_interpretation(daily_detail)

    csv_path, _, rows = _build_two_hour_copula_report(pd.DataFrame([scanner, daily_detail]), tmp_path)
    report = pd.read_csv(csv_path)

    assert rows == 1
    assert pd.isna(report.loc[0, "copula_trade_direction"])
    assert report.loc[0, "copula_detail_capture_status"] == "dashboard_capture_required"


def test_copula_arbitrage_scanner_row_creates_required_detail_capture_task(tmp_path: Path):
    scanner = {
        "journal_layer": "scanner_capture",
        "capture_timestamp_utc": "2026-07-15T09:00:00Z",
        "pair": "ETC-USD/SYRUP-USD",
        "timeframe": "Daily",
        "venue": "dydx",
        "scanner_copula_filter": "Copula (arbitrage)",
        "scanner_snapshot_path": "scanner-evidence.json",
    }

    output, rows, pending = _build_copula_detail_capture_queue(
        pd.DataFrame([scanner]), tmp_path, as_of=pd.Timestamp("2026-07-15T10:00:00Z")
    )
    queue = pd.read_csv(output)

    assert rows == 1
    assert pending == 1
    assert queue.loc[0, "capture_status"] == "capture_required"
    assert "conditional_x_given_y" in queue.loc[0, "missing_fields"]
    assert "copula_best_fit" in queue.loc[0, "required_fields"]


def test_every_scanner_candidate_requires_all_spread_modes_and_copula(tmp_path: Path):
    scanner = {
        "journal_layer": "scanner_capture",
        "capture_timestamp_utc": "2026-07-15T09:00:00Z",
        "pair": "ETC-USD/SYRUP-USD",
        "timeframe": "Daily",
        "venue": "dydx",
        "scanner_snapshot_path": "scanner-evidence.json",
    }

    output, rows, pending = _build_strategy_mode_capture_queue(
        pd.DataFrame([scanner]), tmp_path, as_of=pd.Timestamp("2026-07-15T10:00:00Z")
    )
    queue = pd.read_csv(output)

    assert rows == 7
    assert pending == 7
    assert set(queue["required_mode"]) == {
        "Static (Spread)", "Static (ZScoreR)", "Dyn (Spread)", "Dyn (ZScoreR)",
        "OU (Spread)", "OU (ZScoreR)", "Copula",
    }
    assert set(queue["capture_status"]) == {"capture_required"}
    assert queue.loc[queue["required_mode"].eq("Copula"), "missing_fields"].iloc[0].startswith("fresh_live_pair_page_capture")


def test_stale_scanner_capture_blocks_mode_queue(tmp_path: Path):
    scanner = {
        "journal_layer": "scanner_capture",
        "capture_timestamp_utc": "2026-07-15T06:00:00Z",
        "pair": "ETC-USD/SYRUP-USD",
        "timeframe": "Daily",
    }

    output, _, pending = _build_strategy_mode_capture_queue(
        pd.DataFrame([scanner]), tmp_path, as_of=pd.Timestamp("2026-07-15T10:00:00Z")
    )
    queue = pd.read_csv(output)

    assert pending == 7
    assert set(queue["capture_status"]) == {"dashboard_capture_required"}
    assert queue.loc[0, "missing_fields"].startswith("fresh_live_dashboard_scanner_capture")
