import json

import pandas as pd
import pytest

from quant_platform.crypto_wizards_scanner import scanner_rows_from_payload, write_scanner_reports


def test_scanner_rows_from_payload_extracts_visible_scanner_columns():
    payload = {
        "captured_at": "2026-06-22T19:00:00Z",
        "url": "https://cryptowizards.net/wizards/zscore/scanner",
        "scanner_filters": {
            "priority": "Sharpe (highest)",
            "count": "Count (all cases)",
            "correlation": "Correl (all cases)",
            "hurst": "Hurst (all cases)",
            "half_life": "Halflife (all cases)",
            "copula": "Copula (any)",
            "strategy": "Strategy (all cases)",
            "exchange": "DYDX",
        },
        "scanner_rows": [
            {
                "cells": [
                    "INJ-USD ZRO-USD",
                    "3.0K 548",
                    "sparkline-points",
                    "2026-Jun-22 22:36:27 UTC",
                    "none copula",
                    "-2.63 norm -1.45 roll",
                    "Profile 56.8% x/y 0.6% y/x 11.7% corr",
                    "Jn EG 1.10 hurst 36.0 half life 16 0σ 3 1σ 2 2σ",
                    "-1.9% VaR -3.0% CVaR -4.9% mdd",
                    "50.4% return 2.92 sharpe",
                ]
            }
        ],
    }

    row = scanner_rows_from_payload(payload, source_path="scanner.json")[0]

    assert row.pair == "INJ-USD-ZRO-USD"
    assert row.asset_x == "INJ-USD"
    assert row.asset_y == "ZRO-USD"
    assert row.volume_x == 3000
    assert row.volume_y == 548
    assert row.strategy_label == "none copula"
    assert row.zscore_norm == -2.63
    assert row.zscore_roll == -1.45
    assert row.dependency_profile == "profile"
    assert row.dependency_x_over_y == 0.568
    assert row.dependency_y_over_x == 0.006
    assert row.correlation == pytest.approx(0.117)
    assert row.jn_flag is True
    assert row.eg_flag is True
    assert row.hurst == 1.10
    assert row.half_life == 36.0
    assert row.sigma_0_count == 16
    assert row.sigma_1_count == 3
    assert row.sigma_2_count == 2
    assert row.var == -0.019
    assert row.cvar == -0.03
    assert row.mdd == -0.049
    assert row.return_total == 0.504
    assert row.sharpe == 2.92
    assert row.scanner_priority == "Sharpe (highest)"
    assert row.wizard_exchange == "dydx"
    assert row.asset_x_normalized == "INJ-USD"
    assert row.asset_y_normalized == "ZRO-USD"
    assert row.normalized_pair == "INJ-USD-ZRO-USD"
    assert row.wizard_promotion_allowed is True


def test_scanner_rows_normalize_non_dydx_wizard_symbols():
    payload = {
        "scanner_filters": {"priority": "Sharpe (highest)", "exchange": "Binance"},
        "scanner_rows": [
            {
                "cells": [
                    "ETHUSDT TRUMPUSDT",
                    "",
                    "",
                    "",
                    "",
                    "-0.64 norm 0.98 roll",
                    "",
                    "",
                    "",
                    "20.5% return 3.37 sharpe",
                ]
            }
        ],
    }

    row = scanner_rows_from_payload(payload, source_path="scanner_binance.json")[0]

    assert row.wizard_exchange == "binance"
    assert row.asset_x == "ETHUSDT"
    assert row.asset_y == "TRUMPUSDT"
    assert row.asset_x_normalized == "ETH-USDT"
    assert row.asset_y_normalized == "TRUMP-USDT"
    assert row.asset_x_base == "ETH"
    assert row.asset_y_base == "TRUMP"
    assert row.asset_x_quote == "USDT"
    assert row.asset_y_quote == "USDT"
    assert row.asset_x_canonical_usd == "ETH-USD"
    assert row.asset_y_canonical_usd == "TRUMP-USD"
    assert row.normalized_pair == "ETH-USDT-TRUMP-USDT"
    assert row.wizard_exchange_lane == "binance_research_lane"
    assert row.wizard_promotion_allowed is False


def test_write_scanner_reports_writes_rows_and_field_dictionary(tmp_path):
    input_dir = tmp_path / "scanner"
    output_dir = tmp_path / "reports"
    input_dir.mkdir()
    (input_dir / "capture.json").write_text(
        json.dumps(
            {
                "scanner_rows": [
                    {
                        "pair": "HYPE-USD-TRX-USD",
                        "asset_x": "HYPE-USD",
                        "asset_y": "TRX-USD",
                        "strategy_label": "static spread",
                        "sharpe": 2.82,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = write_scanner_reports(input_dir, output_dir)

    rows = pd.read_csv(paths["rows"])
    fields = pd.read_csv(paths["fields"])
    assert rows.loc[0, "pair"] == "HYPE-USD-TRX-USD"
    assert rows.loc[0, "strategy_label"] == "static spread"
    assert rows.loc[0, "wizard_exchange"] == "dydx"
    assert rows.loc[0, "asset_x_normalized"] == "HYPE-USD"
    assert "sharpe" in set(fields["field"])
