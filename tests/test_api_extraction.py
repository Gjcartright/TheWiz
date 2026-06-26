import pytest
import requests

from quant_platform.api_extraction import (
    CryptoWizardsExtractor,
    CryptoWizardsFetchError,
    CryptoWizardsLiveConfig,
    EndpointSpec,
    parse_endpoint_specs,
)


def test_parse_endpoint_specs_accepts_env_style_list():
    specs = parse_endpoint_specs("pairs=/v1/pairs,metrics:https://api.example.test/v1/metrics")

    assert specs == (
        EndpointSpec(name="pairs", url="/v1/pairs"),
        EndpointSpec(name="metrics", url="https://api.example.test/v1/metrics"),
    )


def test_extractor_joins_relative_endpoint_urls(tmp_path):
    extractor = CryptoWizardsExtractor("https://api.example.test/root", archive_dir=tmp_path)

    assert extractor.endpoint_url(EndpointSpec("pairs", "/v1/pairs")) == "https://api.example.test/root/v1/pairs"
    assert (
        extractor.endpoint_url(EndpointSpec("metrics", "https://other.example.test/v1/metrics"))
        == "https://other.example.test/v1/metrics"
    )


def test_live_config_reports_missing_requirements(monkeypatch):
    monkeypatch.delenv("CRYPTO_WIZARDS_BASE_URL", raising=False)
    monkeypatch.delenv("CRYPTO_WIZARDS_ENDPOINTS", raising=False)

    config = CryptoWizardsLiveConfig.from_env()

    assert config.missing_requirements() == ["CRYPTO_WIZARDS_BASE_URL", "CRYPTO_WIZARDS_ENDPOINTS"]
    with pytest.raises(ValueError, match="Crypto Wizards live config missing"):
        CryptoWizardsExtractor.from_live_config(config)


def test_live_config_loads_env(monkeypatch):
    monkeypatch.setenv("CRYPTO_WIZARDS_BASE_URL", "https://api.example.test")
    monkeypatch.setenv("CRYPTO_WIZARDS_API_KEY", "secret")
    monkeypatch.setenv("CRYPTO_WIZARDS_ENDPOINTS", "pairs=/v1/pairs")

    config = CryptoWizardsLiveConfig.from_env()

    assert config.base_url == "https://api.example.test"
    assert config.api_key == "secret"
    assert config.endpoints == (EndpointSpec(name="pairs", url="/v1/pairs"),)
    assert config.missing_requirements() == []


def test_fetch_wraps_request_errors(monkeypatch, tmp_path):
    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("dns failed")

    monkeypatch.setattr(requests, "request", raise_connection_error)
    extractor = CryptoWizardsExtractor("https://api.example.test", archive_dir=tmp_path)

    with pytest.raises(CryptoWizardsFetchError, match="failed to fetch endpoint pairs"):
        extractor.fetch(EndpointSpec("pairs", "/v1/pairs"))


def test_fetch_uses_crypto_wizards_api_key_header(monkeypatch, tmp_path):
    seen = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_request(method, url, params=None, headers=None, timeout=None):
        seen["method"] = method
        seen["url"] = url
        seen["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(requests, "request", fake_request)
    extractor = CryptoWizardsExtractor("https://api.example.test", api_key="secret", archive_dir=tmp_path)

    payload = extractor.fetch(EndpointSpec("pairs", "/v1beta/prescanned"))

    assert payload == {"ok": True}
    assert seen["headers"]["X-api-key"] == "secret"
    assert seen["headers"]["Content-Type"] == "application/json"
    assert "Authorization" not in seen["headers"]


def test_diagnose_endpoint_reports_dns_and_http_status(monkeypatch, tmp_path):
    class FakeResponse:
        ok = True
        status_code = 200
        text = '{"ok": true}'

    monkeypatch.setattr("socket.getaddrinfo", lambda *args, **kwargs: [("ok",)])
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: FakeResponse())
    extractor = CryptoWizardsExtractor("https://api.example.test", api_key="secret", archive_dir=tmp_path)

    diagnostic = extractor.diagnose_endpoint(EndpointSpec("prescanned", "/v1beta/prescanned"))

    assert diagnostic.name == "prescanned"
    assert diagnostic.dns_ok is True
    assert diagnostic.http_ok is True
    assert diagnostic.status_code == 200
    assert diagnostic.error == ""


def test_diagnose_endpoint_reports_request_error(monkeypatch, tmp_path):
    def raise_dns(*args, **kwargs):
        raise OSError("dns failed")

    def raise_connection_error(*args, **kwargs):
        raise requests.exceptions.ConnectionError("connection failed")

    monkeypatch.setattr("socket.getaddrinfo", raise_dns)
    monkeypatch.setattr(requests, "request", raise_connection_error)
    extractor = CryptoWizardsExtractor("https://api.example.test", api_key="secret", archive_dir=tmp_path)

    diagnostic = extractor.diagnose_endpoint(EndpointSpec("prescanned", "/v1beta/prescanned"))

    assert diagnostic.dns_ok is False
    assert "dns failed" in diagnostic.dns_error
    assert diagnostic.http_ok is False
    assert diagnostic.status_code is None
    assert "connection failed" in diagnostic.error
