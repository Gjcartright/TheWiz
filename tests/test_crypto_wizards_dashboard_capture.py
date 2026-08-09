import json

import pandas as pd

from quant_platform.crypto_wizards_dashboard_capture import (
    EXPECTED_CRYPTO_VENUES,
    EXPECTED_TIMEFRAMES,
    ingest_exhaustive_wizard_dashboard_captures,
)


def _capture_payload(exchange: str, timeframe: str, *, pagination_complete: bool = True) -> dict:
    pair = "BTCUSDT ETHUSDT" if exchange != "dydx" else "BTC-USD ETH-USD"
    return {
        "capture_metadata": {
            "schema_version": "wizard_scanner_dom_capture.v1",
            "captured_at": "2026-08-07T12:00:00Z",
            "source_url": "https://cryptowizards.net/wizards/zscore/scanner",
            "pagination_mechanism": "infinite_scroll",
            "pagination_completion_signal": "five_consecutive_bottom_fetches_without_growth",
            "pagination_complete": pagination_complete,
            "loaded_row_count": 1,
        },
        "scanner_filters": {
            "priority": "sharpe",
            "interval": timeframe,
            "exchange": exchange,
            "cointegration": "Null",
            "correlation": "Null",
            "hurst": "Null",
            "half_life": "Null",
            "copula": False,
            "strategy": "Null",
            "symbol": "",
        },
        "scanner_rows": [
            {
                "pair": pair,
                "sharpe": -4.25,
                "return_total": -0.70,
                "spread_type": "OU",
                "custom_dom_field": {"badge": "orange"},
            }
        ],
    }


def _write_complete_capture_set(input_dir, *, incomplete_cell: str | None = None) -> None:
    input_dir.mkdir(parents=True)
    for exchange in EXPECTED_CRYPTO_VENUES:
        for timeframe in EXPECTED_TIMEFRAMES:
            cell = f"{exchange}|{timeframe}"
            payload = _capture_payload(
                exchange,
                timeframe,
                pagination_complete=cell != incomplete_cell,
            )
            (input_dir / f"{exchange}_{timeframe}.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )


def test_ingest_exhaustive_dashboard_capture_preserves_every_row_and_raw_field(tmp_path):
    input_dir = tmp_path / "captures"
    _write_complete_capture_set(input_dir)

    result = ingest_exhaustive_wizard_dashboard_captures(root=tmp_path, input_dir=input_dir)

    rows = pd.read_csv(result.paths["rows"])
    manifest = pd.read_csv(result.paths["capture_manifest"])
    validation = pd.read_csv(result.paths["coverage_validation"])
    assert result.summary["all_expected_cells_complete"] is True
    assert result.summary["completed_cells"] == 10
    assert len(rows) == 10
    assert (rows["sharpe"] == -4.25).all()
    assert (rows["return_total"] == -0.70).all()
    assert rows["dashboard_raw_custom_dom_field"].str.contains("orange").all()
    assert rows["pair"].duplicated().any()
    assert manifest["capture_status"].eq("COMPLETE").all()
    assert validation["status"].eq("PASS").all()


def test_ingest_exhaustive_dashboard_capture_blocks_incomplete_scroll_cell(tmp_path):
    input_dir = tmp_path / "captures"
    _write_complete_capture_set(input_dir, incomplete_cell="bybit|hourly")

    result = ingest_exhaustive_wizard_dashboard_captures(root=tmp_path, input_dir=input_dir)

    manifest = pd.read_csv(result.paths["capture_manifest"])
    validation = pd.read_csv(result.paths["coverage_validation"])
    blocked = manifest.loc[manifest["cell_id"].eq("bybit|hourly")].iloc[0]
    assert result.summary["all_expected_cells_complete"] is False
    assert blocked["capture_status"] == "BLOCKED"
    assert "infinite_scroll_completion_not_proven" in blocked["blocker"]
    assert validation.loc[
        validation["check"].eq("all_expected_crypto_cells_complete"), "status"
    ].iloc[0] == "FAIL"
