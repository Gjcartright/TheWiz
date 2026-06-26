import json

import requests

from quant_platform.crypto_wizards_history import (
    CryptoWizardsHistoryRequest,
    crawl_prescanned_backtest_histories,
    crawl_prescanned_zscores_histories,
    official_min5_request_rows,
    payload_from_backtest_history,
    payload_from_zscores_history,
)


def test_payload_from_zscores_history_normalizes_official_min5_response():
    request = CryptoWizardsHistoryRequest("BNB-USD", "STX-USD", period=320)
    response = {
        "data": {"zscore": -1.8, "zscore_roll": -1.1},
        "history": {
            "spread": [-0.2, -0.1, 0.0],
            "zscore": [-2.0, -1.0, 0.0],
            "zscore_roll": [-1.8, -0.8, 0.1],
            "hedge_ratio": 1.36,
            "half_life": 9.5,
            "hurst": 0.71,
        },
    }

    payload = payload_from_zscores_history(
        request,
        response,
        prescanned_row={
            "pair_id": 1,
            "ml_confidence": 0.62,
            "profile_match": True,
            "ou_optimal": True,
            "u1_given_u2": 0.73,
            "u2_given_u1": 0.18,
            "copula": "clayton",
            "sharpe": 1.9,
            "mdd": -0.04,
            "cvar": -0.03,
            "closed": 12,
        },
    )

    assert payload["pair"] == "BNB-USD-STX-USD"
    assert payload["interval"] == "min5"
    assert payload["hedge_ratio"] == 1.36
    assert payload["half_life"] == 9.5
    assert payload["hurst"] == 0.71
    assert payload["sharpe"] == 1.9
    assert payload["drawdown"] == -0.04
    assert payload["prescanned"]["ml_confidence"] == 0.62
    first_row = payload["history"][0]
    assert first_row["timestamp"] == 0
    assert first_row["spread"] == -0.2
    assert first_row["zscore"] == -2.0
    assert first_row["rolling_zscore"] == -1.8
    assert first_row["ml_confidence"] == 0.62
    assert first_row["profile_match"] is True
    assert first_row["ou_optimal"] is True
    assert first_row["copula"] == "clayton"
    assert first_row["conditional_probability_distortion"] == 0.55
    assert first_row["completed_trades"] == 12
    assert first_row["drawdown"] == -0.04


def test_official_min5_request_rows_builds_safe_curl_templates():
    rows = official_min5_request_rows(symbol_1="BNB-USD", symbol_2="STX-USD")

    assert [row["request_name"] for row in rows] == [
        "prescanned_min5_pairs",
        "pair_min5_zscores_history",
        "pair_min5_backtest_history",
    ]
    assert "interval=Min5" in rows[0]["url"]
    assert "symbol_1=BNB-USD" in rows[1]["url"]
    assert "with_history=true" in rows[1]["url"]
    assert "/v1beta/backtest" in rows[2]["url"]
    assert "${CRYPTO_WIZARDS_API_KEY}" in rows[1]["curl"]
    assert "secret" not in rows[1]["curl"].lower()
    assert "import-crypto-wizards-backtest" in rows[2]["import_command"]


def test_crawl_prescanned_zscores_histories_writes_pair_payloads(monkeypatch, tmp_path):
    responses = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, headers=None, timeout=None):
        responses.append({"url": url, "params": params, "headers": headers})
        if url.endswith("/v1beta/prescanned"):
            return FakeResponse(
                [
                    {
                        "pair_id": 1,
                        "symbol_1": "BNB-USD",
                        "symbol_2": "STX-USD",
                        "period": 320,
                        "zscore_window": 42,
                        "hedge_ratio": 1.36,
                    }
                ]
            )
        return FakeResponse(
            {
                "data": {"zscore": 0.0, "zscore_roll": 0.1},
                "history": {
                    "spread": [-0.1, 0.0],
                    "zscore": [-1.0, 0.0],
                    "zscore_roll": [-0.8, 0.1],
                },
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)

    paths = crawl_prescanned_zscores_histories(api_key="secret", output_dir=tmp_path, max_pairs=1)

    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["pair"] == "BNB-USD-STX-USD"
    assert payload["history"][1]["zscore"] == 0.0
    assert responses[0]["params"]["interval"] == "Min5"
    assert responses[1]["params"]["with_history"] == "true"
    assert responses[1]["headers"]["X-api-key"] == "secret"


def test_payload_from_backtest_history_normalizes_metrics_and_history():
    request = CryptoWizardsHistoryRequest("BNB-USD", "STX-USD", period=320)
    response = {
        "data": {
            "strat_returns": {
                "annual_return": 0.38,
                "mean_period_return": 0.001,
                "total_return": 0.12,
            },
            "max_drawdown": -0.04,
            "sharpe_ratio": 2.1,
            "sortino_ratio": 3.2,
            "cvar": -0.03,
            "var": -0.02,
            "win_rate": 0.61,
        },
        "history": {
            "spread_stats": {
                "spread": [-0.2, 0.0],
                "zscore": [-2.0, 0.0],
                "zscore_roll": [-1.8, 0.1],
                "hedge_ratio": 1.36,
                "half_life": 9.5,
                "hurst": 0.71,
            },
            "bt_returns": [1.0, 1.02],
        },
    }

    payload = payload_from_backtest_history(request, response)

    assert payload["source_url"].endswith("/v1beta/backtest")
    assert payload["sharpe"] == 2.1
    assert payload["returns_total"] == 0.12
    assert payload["history"][0]["spread"] == -0.2
    assert payload["history"][0]["bt_return"] == 1.0
    assert payload["history"][0]["sharpe"] == 2.1
    assert payload["history"][1]["bt_return"] == 1.02


def test_crawl_prescanned_backtest_histories_writes_pair_payloads(monkeypatch, tmp_path):
    responses = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, headers=None, timeout=None):
        responses.append({"url": url, "params": params, "headers": headers})
        if url.endswith("/v1beta/prescanned"):
            return FakeResponse(
                [
                    {
                        "pair_id": 1,
                        "symbol_1": "BNB-USD",
                        "symbol_2": "STX-USD",
                        "period": 320,
                        "zscore_window": 42,
                        "x_weighting": 0.69,
                    }
                ]
            )
        return FakeResponse(
            {
                "data": {"sharpe_ratio": 2.1, "strat_returns": {"total_return": 0.12}},
                "history": {
                    "spread_stats": {
                        "spread": [-0.1, 0.0],
                        "zscore": [-1.0, 0.0],
                        "zscore_roll": [-0.8, 0.1],
                    },
                    "bt_returns": [1.0, 1.01],
                },
            }
        )

    monkeypatch.setattr(requests, "get", fake_get)

    paths = crawl_prescanned_backtest_histories(api_key="secret", output_dir=tmp_path, max_pairs=1)

    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["pair"] == "BNB-USD-STX-USD"
    assert payload["history"][1]["bt_return"] == 1.01
    assert responses[1]["url"].endswith("/v1beta/backtest")
    assert responses[1]["params"]["strategy"] == "Spread"
    assert responses[1]["params"]["x_weighting"] == 0.69
