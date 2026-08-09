from pathlib import Path

from quant_platform.wizard_pair_matrix import (
    compare_hourly_matrix,
    detect_matrix_anomalies,
    normalize_matrix_rows,
    write_hourly_review_artifacts,
)


def _sample_row(**overrides):
    row = {
        "pair": "BNB-USD/STX-USD",
        "periods": "320",
        "timeframe": "Daily",
        "strategy": "Static (Spread)",
        "best_fit": "clayton",
        "conditional_chart": "71.9%",
        "sharpe_metric": "2.47",
        "sortino_metric": "5.94",
        "net_return_metric": "38.0%",
        "annualized_return_metric": "44.4%",
        "win_rate_metric": "100.0%",
        "closed_trades_metric": "1",
        "max_drawdown_metric": "-4.0%",
        "top_strip": {"corr": "83.2%", "hurst": "0.95", "half_life": "73", "returns": "1.2%", "sharpe": "0.31"},
    }
    row.update(overrides)
    return row


def test_normalize_matrix_rows_parses_numeric_metrics():
    rows = normalize_matrix_rows([_sample_row()])
    row = rows[0]

    assert row["conditional_chart"] == 71.9
    assert row["sharpe"] == 2.47
    assert row["annualized_return_pct"] == 44.4
    assert row["closed_trades"] == 1.0
    assert row["top_corr"] == 83.2


def test_detect_matrix_anomalies_flags_blank_and_duplicate_5min_rows():
    rows = [
        _sample_row(timeframe="5 Min", strategy="Static (Spread)", best_fit="", conditional_chart="", sharpe_metric="0.62", net_return_metric="3.3%", annualized_return_metric="3.7%", closed_trades_metric="2", max_drawdown_metric="-0.4%"),
        _sample_row(timeframe="5 Min", strategy="Static (ZScoreR)", best_fit="", conditional_chart="", sharpe_metric="0.62", net_return_metric="3.3%", annualized_return_metric="3.7%", closed_trades_metric="2", max_drawdown_metric="-0.4%"),
        _sample_row(pair="", timeframe="", strategy="", best_fit="", conditional_chart=""),
    ]

    anomalies = detect_matrix_anomalies(rows)
    types = {(row["timeframe"], row["strategy"], row["anomaly"]) for row in anomalies}

    assert ("", "", "blank_variant_fields") in types
    assert ("5 Min", "Static (Spread)", "missing_best_fit") in types
    assert ("5 Min", "Static (ZScoreR)", "duplicate_outcome_signature") in types


def test_compare_hourly_matrix_marks_changed_and_unchanged_rows():
    prior = [_sample_row(), _sample_row(strategy="Copula", sharpe_metric="0.10")]
    current = [_sample_row(), _sample_row(strategy="Copula", sharpe_metric="0.62")]

    rows = compare_hourly_matrix(current, prior)
    status = {(row["strategy"], row["review_status"]) for row in rows}

    assert ("Static (Spread)", "unchanged") in status
    assert ("Copula", "changed") in status


def test_write_hourly_review_artifacts_writes_outputs(tmp_path: Path):
    current = [_sample_row(), _sample_row(timeframe="5 Min", strategy="Copula", best_fit="", conditional_chart="")]
    prior = [_sample_row()]

    paths = write_hourly_review_artifacts(current_rows=current, prior_rows=prior, output_dir=tmp_path)

    assert paths["normalized"].exists()
    assert paths["review"].exists()
    assert paths["anomalies"].exists()
