import json
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

import pytest
import requests

from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_catalog import BASE_URL, ENDPOINTS
from quant_platform.crypto_wizards_history import (
    CryptoWizardsCustomSeriesAnalyticsRequest,
    CryptoWizardsCustomSeriesBacktestRequest,
    CryptoWizardsCustomSeriesCopulaRequest,
    CryptoWizardsHistoryRequest,
    crawl_prescanned_backtest_histories,
    crawl_prescanned_zscores_histories,
    fetch_custom_series_analytics,
    fetch_custom_series_backtest,
    fetch_custom_series_copula,
    official_min5_request_rows,
    payload_from_backtest_history,
    payload_from_zscores_history,
)
from quant_platform.orchestration.corrective_effect_guard import phase00_effect_guard
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    external_effect_authority_session,
    read_authorized_credential,
)
from quant_platform.orchestration.effect_authority import EffectAuthority

HASH = "a" * 64


@contextmanager
def _wizard_effects(root: Path, *, secret: str = "test-key"):
    authority = EffectAuthority(
        root=root,
        secret=b"w" * 32,
        issuer_id="wizard-history-test",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    targets = frozenset(
        f"{BASE_URL}{endpoint.path}"
        for endpoint in ENDPOINTS
        if endpoint.method in {"GET", "POST"} and endpoint.path.startswith("/")
    )
    with (
        external_effect_authority_session(
            authority=authority,
            run_id="wizard-history-run",
            intended_slot_id="wizard-history-slot",
            source_fingerprint_sha256=HASH,
            runtime_fingerprint_sha256=HASH,
            configuration_fingerprint_sha256=HASH,
            provider_id="crypto_wizards",
            account_scope_id="wizard-history-test-account",
            reservation_id="wizard-history-reservation",
            reservation_sha256=sha256(b"wizard-history-reservation").hexdigest(),
            allowed_targets=targets,
            allowed_credential_keys=frozenset({"CRYPTO_WIZARDS_API_KEY"}),
            max_total_requests=20,
            max_total_credits=100,
        ),
        phase00_effect_guard(),
    ):
        read_authorized_credential(
            "CRYPTO_WIZARDS_API_KEY",
            reader=lambda _key: secret,
        )
        yield authority


def test_custom_series_backtest_request_preserves_captured_mode_and_cost_settings():
    request = CryptoWizardsCustomSeriesBacktestRequest(
        series_1_opens=tuple(99.5 + index for index in range(50)),
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_opens=tuple(49.5 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
        strategy="ZScoreRoll",
        spread_type="OU",
        roll_w=42,
        entry_level=2.0,
        exit_level=0.0,
        x_weighting=0.5,
        slippage_rate=0.0005,
        commission_rate=0.0005,
        stop_loss_rate=0.1,
    )

    payload = request.payload()

    assert payload["params"]["strategy"] == "ZScoreRoll"
    assert payload["params"]["spread_type"] == "Ou"
    assert payload["bt_inputs"]["commission_rate"] == 0.0005
    assert len(payload["params"]["series_1_opens"]) == 50
    assert len(payload["params"]["series_2_opens"]) == 50
    assert payload["bt_inputs"]["stop_loss_rate"] == 0.1


def test_get_backtest_defaults_match_verified_dashboard_costs_and_omit_stop():
    params = CryptoWizardsHistoryRequest(
        symbol_1="BTCUSDT",
        symbol_2="ETHUSDT",
        exchange="Binance",
        interval="Daily",
        period=365,
    ).backtest_params()

    assert params["commission_rate"] == 0.001
    assert params["slippage_rate"] == 0.0005
    assert "stop_loss_rate" not in params


def test_fetch_custom_series_backtest_posts_typed_payload(monkeypatch, tmp_path):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": {"sharpe_ratio": 1.8}}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    request = CryptoWizardsCustomSeriesBacktestRequest(
        series_1_opens=tuple(99.5 + index for index in range(50)),
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_opens=tuple(49.5 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
        strategy="Spread",
        spread_type="Static",
        roll_w=42,
        entry_level=2.0,
        exit_level=0.0,
        x_weighting=0.5,
        slippage_rate=0.0005,
        commission_rate=0.0005,
    )

    with _wizard_effects(tmp_path) as authority:
        response = fetch_custom_series_backtest(request, api_key="test-key")

    assert response["data"]["sharpe_ratio"] == 1.8
    assert seen["url"].endswith("/v1beta/backtest")
    assert seen["json"]["params"]["strategy"] == "Spread"
    assert seen["headers"]["X-api-key"] == "test-key"
    accounting = authority.run_accounting(
        run_id="wizard-history-run",
        intended_slot_id="wizard-history-slot",
    )
    assert accounting["external_calls"] == 1
    assert accounting["external_credits_consumed"] == 2
    assert accounting["accounting_complete"] is True


def test_fetch_custom_series_copula_posts_only_matching_close_series(
    monkeypatch, tmp_path
):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "copula_name": "clayton",
                "u1_given_u2": 0.91,
                "u2_given_u1": 0.08,
            }

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update({"url": url, "json": json, "headers": headers})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    request = CryptoWizardsCustomSeriesCopulaRequest(
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
    )

    with _wizard_effects(tmp_path):
        response = fetch_custom_series_copula(request, api_key="test-key")

    assert response["copula_name"] == "clayton"
    assert seen["url"].endswith("/v1beta/copula")
    assert set(seen["json"]) == {"series_1_closes", "series_2_closes"}
    assert len(seen["json"]["series_1_closes"]) == 50


@pytest.mark.parametrize(
    ("endpoint", "expected_extra"),
    [
        ("cointegration", {"spread_type", "roll_w", "with_history"}),
        ("correlations", set()),
        ("spread", {"spread_type", "roll_w", "with_history"}),
        ("zscores", {"spread_type", "roll_w", "with_history"}),
    ],
)
def test_fetch_custom_series_analytics_uses_typed_endpoint_payload(
    monkeypatch, tmp_path, endpoint, expected_extra
):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"endpoint": endpoint}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.update({"url": url, "json": json, "headers": headers})
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    request = CryptoWizardsCustomSeriesAnalyticsRequest(
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
    )

    with _wizard_effects(tmp_path):
        response = fetch_custom_series_analytics(endpoint, request, api_key="test-key")

    assert response["endpoint"] == endpoint
    assert seen["url"].endswith(f"/v1beta/{endpoint}")
    assert set(seen["json"]) == {
        "series_1_closes",
        "series_2_closes",
        *expected_extra,
    }


def test_post_error_preserves_bounded_vendor_diagnostic(monkeypatch, tmp_path):
    class FakeResponse:
        status_code = 400
        text = '{"detail":"spread_type must be Ou"}'

        def raise_for_status(self):
            raise requests.HTTPError("400 Client Error")

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: FakeResponse())
    request = CryptoWizardsCustomSeriesCopulaRequest(
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
    )

    with (
        _wizard_effects(tmp_path),
        pytest.raises(CryptoWizardsFetchError, match="spread_type must be Ou"),
    ):
        fetch_custom_series_copula(request, api_key="test-key")


def test_official_wizard_request_without_effect_authority_is_blocked(monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: pytest.fail("wire call must remain blocked"),
    )
    request = CryptoWizardsCustomSeriesCopulaRequest(
        series_1_closes=tuple(100.0 + index for index in range(50)),
        series_2_closes=tuple(50.0 + index for index in range(50)),
    )

    with pytest.raises(CryptoWizardsFetchError, match="reserved effect authority"):
        fetch_custom_series_copula(request, api_key="test-key")


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
    assert len(payload["config_hash"]) == 64
    assert payload["wizard_configuration"]["exact_mode"] == "static_zscores_bundle"
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
    assert first_row["config_hash"] == payload["config_hash"]


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

    with _wizard_effects(tmp_path, secret="secret"):
        paths = crawl_prescanned_zscores_histories(
            api_key="secret",
            output_dir=tmp_path,
            max_pairs=1,
        )

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
    assert payload["wizard_configuration"]["exact_mode"] == "static_spread"
    assert payload["history"][0]["config_hash"] == payload["config_hash"]


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

    with _wizard_effects(tmp_path, secret="secret"):
        paths = crawl_prescanned_backtest_histories(
            api_key="secret",
            output_dir=tmp_path,
            max_pairs=1,
        )

    assert len(paths) == 1
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["pair"] == "BNB-USD-STX-USD"
    assert payload["history"][1]["bt_return"] == 1.01
    assert responses[1]["url"].endswith("/v1beta/backtest")
    assert responses[1]["params"]["strategy"] == "Spread"
    assert responses[1]["params"]["x_weighting"] == 0.69
