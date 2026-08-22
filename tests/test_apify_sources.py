from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import requests

from quant_platform.active_pipeline import _planned_source_rows
from quant_platform.apify_sources import (
    APIFY_API_TARGET,
    _extract_usage_details,
    _run_actor_fetch,
    parse_apify_sources_from_mcp_url,
    refresh_apify_sources,
)
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    external_effect_authority_session,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
)

HASH = "a" * 64


class _Response:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self._payload


class _ApifyClient:
    def __init__(self, *, reported_credits: object = 2) -> None:
        self.reported_credits = reported_credits
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(("POST", url, kwargs))
        return _Response({"data": {"id": "run-1"}})

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(("GET", url, kwargs))
        if "/actor-runs/" in url:
            billing = (
                {}
                if self.reported_credits is None
                else {"credits": self.reported_credits, "currency": "USD"}
            )
            return _Response(
                {
                    "data": {
                        "id": "run-1",
                        "status": "SUCCEEDED",
                        "defaultDatasetId": "dataset-1",
                        "billing": billing,
                    }
                }
            )
        return _Response([{"symbol": "BTC", "price": 100.0}])


class _ResponseLossApifyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def post(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(("POST", url, kwargs))
        raise requests.ConnectionError("response lost after actor start")

    def get(self, url: str, **kwargs: object) -> _Response:
        self.calls.append(("GET", url, kwargs))
        raise AssertionError("provider polling must not follow a lost start response")


def _authority(
    root: Path,
    *,
    max_requests: int = 3,
    max_credits: int = 2,
    run_id: str = "apify-test-run",
    reservation_id: str = "apify-reservation-1",
):
    authority = EffectAuthority(
        root=root,
        secret=b"apify-test-authority-secret-32-bytes-minimum",
        issuer_id="apify-test-supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    return external_effect_authority_session(
        authority=authority,
        run_id=run_id,
        intended_slot_id="apify-test-slot",
        source_fingerprint_sha256=HASH,
        runtime_fingerprint_sha256=HASH,
        configuration_fingerprint_sha256=HASH,
        provider_id="apify",
        account_scope_id="apify-research-account",
        reservation_id=reservation_id,
        reservation_sha256=sha256(b"apify-reservation").hexdigest(),
        allowed_targets=frozenset({APIFY_API_TARGET}),
        allowed_credential_keys=frozenset({"APIFY_API_TOKEN"}),
        max_total_requests=max_requests,
        max_total_credits=max_credits,
        allowed_call_contracts=frozenset(
            {
                ExternalEffectCallContract(
                    operation="APIFY_ACTOR_START",
                    method="POST",
                    target=APIFY_API_TARGET,
                    credit_units_per_request=None,
                ),
                ExternalEffectCallContract(
                    operation="APIFY_ACTOR_POLL",
                    method="GET",
                    target=APIFY_API_TARGET,
                    credit_units_per_request=0,
                ),
                ExternalEffectCallContract(
                    operation="APIFY_DATASET_FETCH",
                    method="GET",
                    target=APIFY_API_TARGET,
                    credit_units_per_request=0,
                ),
            }
        ),
    )


def _fake_actor_fetcher(client: _ApifyClient):
    def fetch(**kwargs: object):
        return _run_actor_fetch(
            **kwargs,
            request_client=client,
            sleep_fn=lambda _seconds: None,
            jitter_fn=lambda _low, _high: 0.0,
            clock_fn=lambda: 0.0,
        )

    return fetch


def test_parse_apify_sources_from_mcp_url_handles_actors_docs_and_slugged_actors():
    url = (
        "https://mcp.apify.com/?tools=actors,docs,"
        "parseforge/dydx-v4-perpetual-markets-scraper,"
        "parseforge/dydx-markets-scraper,"
        "muhammetakkurtt/coinmarketcap-scraper"
    )
    sources = parse_apify_sources_from_mcp_url(url)
    assert "apify/actors" in sources
    assert "apify/docs" in sources
    assert "parseforge/dydx-v4-perpetual-markets-scraper" in sources
    assert "muhammetakkurtt/coinmarketcap-scraper" in sources
    assert len(sources) == len(set(sources))


def test_refresh_apify_sources_without_fetch_generates_coverage(tmp_path: Path):
    mcp_url = "https://mcp.apify.com/?tools=actors,docs,parseforge/dydx-markets-scraper"
    result = refresh_apify_sources(root=tmp_path, mcp_url=mcp_url, do_fetch=False)

    assert result.coverage_path.exists()
    assert result.manifest_path.exists()
    assert result.source_count == 3
    assert result.sampled_count == 0
    frame = pd.read_csv(result.coverage_path)
    assert set(frame["source_id"]) == {
        "apify/actors",
        "apify/docs",
        "parseforge/dydx-markets-scraper",
    }


def test_apify_fetch_requires_explicit_credit_ceiling_before_authority(tmp_path: Path):
    with pytest.raises(ValueError, match="positive actor_credit_ceiling"):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            api_token="secret",
        )


def test_apify_fetch_requires_external_effect_session_before_token_or_network(tmp_path: Path):
    calls: list[dict[str, object]] = []
    with pytest.raises(EffectAuthorityError, match="apify_external_effect_session_missing"):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            api_token="secret",
            actor_credit_ceiling=2,
            actor_fetcher=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []


def test_apify_missing_authorized_token_blocks_before_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    calls: list[dict[str, object]] = []
    with (
        _authority(tmp_path) as session,
        pytest.raises(
            EffectAuthorityError,
            match="authorized_credential_missing",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            actor_credit_ceiling=2,
            actor_fetcher=lambda **kwargs: calls.append(kwargs),
        )
    assert calls == []
    assert len(session.credential_receipts) == 1
    assert session.network_receipts == []
    assert session.credit_receipts == []


def test_authorized_apify_fetch_accounts_every_request_and_credit_without_token_leak(
    tmp_path: Path,
):
    client = _ApifyClient(reported_credits=2)
    with _authority(tmp_path) as session:
        result = refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(client),
        )

    assert result.sampled_count == 1
    assert result.failed_count == 0
    assert session.consumed_requests == 3
    assert session.consumed_credits == 2
    assert len(session.credential_receipts) == 1
    assert len(session.network_receipts) == 3
    assert len(session.credit_receipts) == 1
    assert all("apify-secret-token" not in url for _method, url, _kwargs in client.calls)
    assert all(
        "token" not in (kwargs.get("params") or {}) for _method, _url, kwargs in client.calls
    )
    manifest = pd.read_csv(result.manifest_path).iloc[0]
    assert manifest["credit_reconciliation_status"] == (
        "PASS_REPORTED_USAGE_WITHIN_AUTHORIZED_CEILING"
    )
    assert int(manifest["effect_network_requests"]) == 3
    assert int(manifest["effect_credit_units"]) == 2
    assert manifest["effect_reservation_id"] == "apify-reservation-1"
    intent_path = Path(str(manifest["actor_intent_path"]))
    terminal_path = Path(str(manifest["actor_terminal_path"]))
    assert intent_path.is_file()
    assert terminal_path.is_file()
    intent = json.loads(intent_path.read_text(encoding="utf-8"))
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert intent["attempt_id"] == manifest["actor_attempt_id"]
    assert terminal["attempt_id"] == manifest["actor_attempt_id"]
    assert terminal["provider_status"] == "sampled"
    assert terminal["reconciliation_required"] is False
    assert terminal["order_submission_included"] is False
    assert len(terminal["provider_response_bindings"]) == 3
    assert int(manifest["provider_response_count"]) == 3
    for binding in terminal["provider_response_bindings"]:
        response_path = tmp_path / binding["path"]
        assert response_path.is_file()
        assert sha256(response_path.read_bytes()).hexdigest() == binding["sha256"]
        assert "apify-secret-token" not in response_path.read_text(encoding="utf-8")
    coverage = pd.read_csv(result.coverage_path)
    assert coverage.loc[0, "sample_status"] == "sampled"


def test_apify_response_evidence_failure_leaves_effect_unknown_and_blocks_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _ApifyClient(reported_credits=2)

    def fail_response_record(**_kwargs: object) -> str:
        raise OSError("simulated Apify response evidence failure")

    monkeypatch.setattr(
        "quant_platform.apify_sources._record_apify_provider_response",
        fail_response_record,
    )
    with (
        _authority(tmp_path) as session,
        pytest.raises(
            OSError,
            match="simulated Apify response evidence failure",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url=(
                "https://mcp.apify.com/?tools="
                "parseforge/dydx-markets-scraper"
            ),
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(client),
        )

    assert len(client.calls) == 1
    assert session.authority.outcome(session.network_receipts[0].permit_id) == (
        "UNKNOWN"
    )
    assert session.authority.outcome(session.credit_receipts[0].permit_id) == (
        "UNKNOWN"
    )
    assert len(
        list(
            (
                tmp_path
                / "data"
                / "research"
                / "apify_actor_attempts"
                / "intents"
            ).glob("*.json")
        )
    ) == 1
    assert list(
        (
            tmp_path
            / "data"
            / "research"
            / "apify_actor_attempts"
            / "terminals"
        ).glob("*.json")
    ) == []


def test_apify_lost_start_response_requires_reconciliation_and_blocks_later_run(
    tmp_path: Path,
):
    source = "parseforge/dydx-markets-scraper"
    mcp_url = f"https://mcp.apify.com/?tools={source}"
    first_client = _ResponseLossApifyClient()

    with (
        _authority(
            tmp_path,
            run_id="apify-response-loss-run-1",
            reservation_id="apify-response-loss-reservation-1",
        ),
        pytest.raises(
            EffectAuthorityError,
            match="apify_actor_start_reconciliation_required",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url=mcp_url,
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(first_client),
        )

    assert len(first_client.calls) == 1
    manifest = pd.read_csv(
        tmp_path / "reports" / "active" / "apify_source_capture_manifest.csv"
    ).iloc[0]
    intent_path = Path(str(manifest["actor_intent_path"]))
    terminal_path = Path(str(manifest["actor_terminal_path"]))
    assert intent_path.is_file()
    assert terminal_path.is_file()
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    assert terminal["provider_status"] == "network_error:ConnectionError"
    assert terminal["reconciliation_required"] is True
    assert "apify-secret-token" not in intent_path.read_text(encoding="utf-8")
    assert "apify-secret-token" not in terminal_path.read_text(encoding="utf-8")

    second_client = _ApifyClient()
    with (
        _authority(
            tmp_path,
            run_id="apify-response-loss-run-2",
            reservation_id="apify-response-loss-reservation-2",
        ),
        pytest.raises(
            EffectAuthorityError,
            match="apify_actor_start_reconciliation_required",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url=mcp_url,
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(second_client),
        )
    assert second_client.calls == []


def test_apify_missing_terminal_blocks_retry_before_provider_call(tmp_path: Path):
    source = "parseforge/dydx-markets-scraper"
    mcp_url = f"https://mcp.apify.com/?tools={source}"

    with (
        _authority(
            tmp_path,
            run_id="apify-crash-run-1",
            reservation_id="apify-crash-reservation-1",
        ),
        pytest.raises(RuntimeError, match="simulated process death"),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url=mcp_url,
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("simulated process death")
            ),
        )

    intents = list(
        (
            tmp_path
            / "data"
            / "research"
            / "apify_actor_attempts"
            / "intents"
        ).glob("*.json")
    )
    terminals = list(
        (
            tmp_path
            / "data"
            / "research"
            / "apify_actor_attempts"
            / "terminals"
        ).glob("*.json")
    )
    assert len(intents) == 1
    assert terminals == []

    second_client = _ApifyClient()
    with (
        _authority(
            tmp_path,
            run_id="apify-crash-run-2",
            reservation_id="apify-crash-reservation-2",
        ),
        pytest.raises(
            EffectAuthorityError,
            match="apify_actor_start_reconciliation_required",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url=mcp_url,
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(second_client),
        )
    assert second_client.calls == []


@pytest.mark.parametrize(
    ("reported_credits", "expected_status"),
    [
        (None, "unknown_reported_usage"),
        (3, "blocked_reported_usage_exceeds_authorized_ceiling"),
    ],
)
def test_apify_usage_without_reconciliation_cannot_publish_accepted_sample(
    tmp_path: Path,
    reported_credits: object,
    expected_status: str,
):
    client = _ApifyClient(reported_credits=reported_credits)
    with _authority(tmp_path):
        result = refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(client),
        )

    assert result.sampled_count == 0
    assert result.failed_count == 1
    manifest = pd.read_csv(result.manifest_path).iloc[0]
    assert manifest["run_status"] == expected_status
    assert not Path(str(manifest["output_path"])).exists()


def test_apify_request_budget_exhaustion_blocks_before_unpermitted_dataset_call(
    tmp_path: Path,
):
    client = _ApifyClient(reported_credits=2)
    with (
        _authority(tmp_path, max_requests=2),
        pytest.raises(
            EffectAuthorityError,
            match="external_request_budget_exhausted",
        ),
    ):
        refresh_apify_sources(
            root=tmp_path,
            mcp_url="https://mcp.apify.com/?tools=parseforge/dydx-markets-scraper",
            api_token="apify-secret-token",
            actor_credit_ceiling=2,
            actor_fetcher=_fake_actor_fetcher(client),
        )
    assert len(client.calls) == 2


def test_planned_source_rows_includes_all_sources_and_marks_context_only_not_promotable():
    source_coverage = pd.DataFrame(
        [
            {
                "source_id": "fraktalapi/funding-pulse",
                "category": "cross_exchange_perp_feed",
                "status": "checked",
                "sample_status": "not_sampled",
                "evidence": "reports/active/apify_mcp_source_coverage_2026-06-25.csv",
                "limitations": "",
            },
            {
                "source_id": "parseforge/dydx-v4-perpetual-markets-scraper",
                "category": "perp_market_feed",
                "status": "checked",
                "sample_status": "sampled",
                "limitations": "requires dYdX pair feed",
            },
            {
                "source_id": "parseforge/gmx-arbitrum-stats-scraper",
                "category": "defi_perp_feed",
                "status": "checked",
                "sample_status": "not_sampled",
                "limitations": "requires defi context",
            },
            {
                "source_id": "bybit-screener",
                "category": "supplemental",
                "status": "checked",
                "sample_status": "needs_api_key",
                "limitations": "",
            },
        ]
    )

    rows = _planned_source_rows(source_coverage)
    assert len(rows) >= 3
    byrow = {row["venue"]: row for row in rows}
    assert byrow["cross_exchange"]["source_system"] == "funding_pulse"
    assert byrow["dydx"]["execution_authority"] is True
    assert byrow["gmx"]["promotion_allowed"] is False
    assert byrow["bybit"]["promotion_allowed"] is False


def test_extract_usage_details_reads_common_apify_fields():
    payload = {
        "id": "run_123",
        "startedAt": "2026-06-27T00:00:00.000Z",
        "finishedAt": "2026-06-27T00:00:05.123Z",
        "stats": {"runTimeMillis": 5120},
        "meta": {"datasetId": "ds_1"},
        "billing": {"cost": 1.42, "currency": "USD", "credits": 142},
        "status": "SUCCEEDED",
    }

    usage = _extract_usage_details(payload, "run_123")

    assert usage["run_id"] == "run_123"
    assert usage["started_at_utc"] == "2026-06-27T00:00:00.000Z"
    assert usage["finished_at_utc"] == "2026-06-27T00:00:05.123Z"
    assert usage["duration_ms"] == 5120
    assert usage["usage_currency"] == "USD"
    assert usage["usage_amount"] == 1.42
    assert usage["usage_credits"] == 142
    assert "datasetId" in str(usage["usage_meta"])
