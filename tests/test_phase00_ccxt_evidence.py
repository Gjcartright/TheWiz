from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import ccxt
import pytest

from quant_platform.orchestration.phase00_ccxt_evidence import (
    CcxtCaptureRequest,
    CcxtEvidenceContract,
    CcxtEvidenceRuntime,
    write_ccxt_capture_evidence,
)

CONTRACT_PATH = Path(__file__).resolve().parents[1] / "config" / "ccxt_evidence_contract.json"
SOURCE_REVISION = "a" * 64
MINUTE_MS = 60_000


class RequestTimeout(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


def _market(symbol: str = "BTC/USDT") -> dict[str, object]:
    base, quote = symbol.split("/")
    return {
        "id": symbol.replace("/", ""),
        "symbol": symbol,
        "base": base,
        "quote": quote,
        "type": "spot",
        "spot": True,
        "swap": False,
        "future": False,
        "contract": False,
        "settle": None,
        "contractSize": None,
        "linear": None,
        "inverse": None,
        "marginModes": {"cross": True, "isolated": False},
        "active": True,
        "limits": {"amount": {"min": 0.0001, "max": None}},
        "precision": {"amount": 0.0001, "price": 0.01},
    }


def _bar(timestamp: int, *, close: float | None = None, volume: float = 2.0) -> list[float]:
    price = 100.0 + timestamp / MINUTE_MS
    close_price = price if close is None else close
    return [
        timestamp,
        price,
        max(price, close_price) + 1.0,
        min(price, close_price) - 1.0,
        close_price,
        volume,
    ]


class FakeExchange:
    id = "binance"
    has: ClassVar[dict[str, object]] = {"fetchOHLCV": True}
    timeframes: ClassVar[dict[str, str]] = {"1m": "1m", "5m": "5m"}

    def __init__(
        self,
        page_fn: Callable[[int, int], object],
        *,
        markets: dict[str, dict[str, object]] | None = None,
        load_failure_at: int | None = None,
    ) -> None:
        self.page_fn = page_fn
        self.markets = markets or {"BTC/USDT": _market()}
        self.load_failure_at = load_failure_at
        self.load_calls: list[bool] = []
        self.fetch_calls: list[dict[str, object]] = []
        self.forbidden_calls: list[str] = []

    def load_markets(self, *, reload: bool = False):
        self.load_calls.append(reload)
        if self.load_failure_at == len(self.load_calls):
            raise RequestTimeout("message must not enter a receipt")
        return self.markets

    def fetch_ohlcv(
        self,
        *,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
        params: dict[str, object],
    ):
        self.fetch_calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "limit": limit,
                "params": dict(params),
            }
        )
        return self.page_fn(since, limit)

    def create_order(self, *args, **kwargs):
        self.forbidden_calls.append("create_order")
        raise AssertionError("private/order method reached")

    def private_get(self, *args, **kwargs):
        self.forbidden_calls.append("private_get")
        raise AssertionError("private method reached")


def _contract() -> CcxtEvidenceContract:
    return CcxtEvidenceContract.from_path(CONTRACT_PATH)


def _request(
    *,
    since_ms: int = 0,
    end_ms: int = 3 * MINUTE_MS,
    captured_at_ms: int = 3 * MINUTE_MS,
    timeframe: str = "1m",
    page_limit: int = 3,
    force_market_reload: bool = False,
) -> CcxtCaptureRequest:
    return CcxtCaptureRequest(
        source_id="offline_fixture",
        lane="binance_spot",
        exchange_id="binance",
        symbol="BTC/USDT",
        timeframe=timeframe,
        since_ms=since_ms,
        end_ms=end_ms,
        captured_at_ms=captured_at_ms,
        page_limit=page_limit,
        force_market_reload=force_market_reload,
    )


def _runtime(
    exchange: FakeExchange,
    *,
    package_version: str | None = "4.4.26",
    source_revision: str = SOURCE_REVISION,
    factory_calls: list[tuple[str, dict[str, object]]] | None = None,
) -> CcxtEvidenceRuntime:
    calls = factory_calls if factory_calls is not None else []

    def factory(exchange_id: str, options: dict[str, object]) -> object:
        calls.append((exchange_id, options))
        return exchange

    return CcxtEvidenceRuntime(
        contract=_contract(),
        exchange_factory=factory,
        package_version=package_version,
        source_revision=source_revision,
        runtime_fingerprint={"python": "offline-test", "platform": "fixture"},
    )


def test_contract_is_exactly_pinned_public_only_and_lane_specific():
    contract = _contract()

    assert contract.pinned_version == "4.4.26"
    assert contract.payload["phase00"] == {
        "public_data_only": True,
        "credentials_allowed": False,
        "sandbox_allowed": False,
        "account_eligibility_authority": False,
        "order_authority": False,
    }
    assert contract.payload["methods"]["allowed_public"] == ["load_markets", "fetch_ohlcv"]
    assert "create_order" in contract.payload["methods"]["forbidden"]
    lanes = contract.payload["lanes"]
    assert len(lanes) == 12
    registry_ids = [lane["parameter_registry_id"] for lane in lanes.values()]
    assert len(registry_ids) == len(set(registry_ids))
    assert (
        lanes["binance_spot"]["fetch_ohlcv_params"]
        != lanes["bybit_linear_perp"]["fetch_ohlcv_params"]
    )


def test_ccxt_contract_project_lock_and_installed_runtime_share_exact_version():
    root = CONTRACT_PATH.parents[1]
    contract = _contract()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))

    requirement = f"ccxt=={contract.pinned_version}"
    assert requirement in pyproject["project"]["dependencies"]
    locked_ccxt = [package for package in lock["package"] if package.get("name") == "ccxt"]
    assert len(locked_ccxt) == 1
    assert locked_ccxt[0]["version"] == contract.pinned_version
    project_package = next(
        package
        for package in lock["package"]
        if package.get("name") == pyproject["project"]["name"]
    )
    locked_requirement = next(
        item for item in project_package["metadata"]["requires-dist"] if item.get("name") == "ccxt"
    )
    assert locked_requirement["specifier"] == f"=={contract.pinned_version}"
    assert ccxt.__version__ == contract.pinned_version


def test_runtime_fingerprint_rejects_credential_shaped_fields():
    exchange = FakeExchange(lambda since, limit: [])
    with pytest.raises(ValueError, match="runtime_fingerprint_contains_credentials"):
        CcxtEvidenceRuntime(
            contract=_contract(),
            exchange_factory=lambda exchange_id, options: exchange,
            package_version="4.4.26",
            source_revision=SOURCE_REVISION,
            runtime_fingerprint={"api_key": "must-not-be-recorded"},
        )


@pytest.mark.parametrize(
    ("package_version", "blocker"),
    [(None, "ccxt_package_absent"), ("4.4.25", "ccxt_package_version_mismatch")],
)
def test_absent_or_wrong_package_fails_before_exchange_factory(package_version, blocker):
    calls: list[tuple[str, dict[str, object]]] = []
    exchange = FakeExchange(lambda since, limit: [])
    runtime = _runtime(exchange, package_version=package_version, factory_calls=calls)

    result = runtime.capture(_request())

    assert result.status == "terminal_failure"
    assert blocker in result.blockers
    assert calls == []
    assert result.runtime["package_version"] == package_version
    assert len(result.capture_id) == 64


def test_load_markets_capabilities_metadata_reuse_and_no_private_access():
    bars = [_bar(index * MINUTE_MS) for index in range(3)]
    exchange = FakeExchange(lambda since, limit: [row for row in bars if row[0] >= since][:limit])
    factory_calls: list[tuple[str, dict[str, object]]] = []
    runtime = _runtime(exchange, factory_calls=factory_calls)

    first = runtime.capture(_request())
    second = runtime.capture(_request(captured_at_ms=4 * MINUTE_MS))

    assert first.accepted
    assert second.accepted
    assert len(factory_calls) == 1
    assert factory_calls[0] == (
        "binance",
        {"options": {"defaultType": "spot"}, "enableRateLimit": True, "timeout": 120000},
    )
    assert exchange.load_calls == [False]
    assert first.market_provenance["cache_status"] == "miss"
    assert second.market_provenance["cache_status"] == "fresh_reused"
    assert first.market == {
        "venue_market_id": "BTCUSDT",
        "unified_symbol": "BTC/USDT",
        "base": "BTC",
        "quote": "USDT",
        "type": "spot",
        "spot": True,
        "swap": False,
        "future": False,
        "contract": False,
        "settle": "",
        "contract_size": None,
        "linear": None,
        "inverse": None,
        "margin_modes": ["cross"],
        "active": True,
        "limits": {"amount": {"max": None, "min": 0.0001}},
        "precision": {"amount": 0.0001, "price": 0.01},
    }
    assert len(first.market_provenance["market_snapshot_hash"]) == 64
    assert len(first.market_provenance["load_markets_request_hash"]) == 64
    assert (
        first.market_provenance["load_markets_response_hash"]
        == first.market_provenance["market_snapshot_hash"]
    )
    assert exchange.forbidden_calls == []


def test_pagination_dedup_order_incomplete_exclusion_volume_units_and_hashes():
    pages = {
        0: [_bar(2 * MINUTE_MS), _bar(0), _bar(MINUTE_MS), _bar(MINUTE_MS), _bar(3 * MINUTE_MS)],
        4 * MINUTE_MS: [_bar(5 * MINUTE_MS), _bar(4 * MINUTE_MS)],
    }
    exchange = FakeExchange(lambda since, limit: pages.get(since, []))
    runtime = _runtime(exchange)
    request = _request(
        end_ms=6 * MINUTE_MS,
        captured_at_ms=5 * MINUTE_MS + 30_000,
        page_limit=6,
    )

    result = runtime.capture(request)

    assert result.accepted
    assert [row["timestamp_ms"] for row in result.normalized_ohlcv] == [
        0,
        MINUTE_MS,
        2 * MINUTE_MS,
        3 * MINUTE_MS,
        4 * MINUTE_MS,
    ]
    assert result.coverage_receipt["identical_duplicate_count"] == 1
    assert result.coverage_receipt["incomplete_bar_count"] == 1
    assert result.coverage_receipt["exact_reconciliation"] is True
    assert result.coverage_receipt["expected_bar_count"] == 5
    assert result.coverage_receipt["returned_complete_bar_count"] == 5
    assert len(result.raw_ohlcv) == 7
    assert all(len(item["row_hash"]) == 64 for item in result.raw_ohlcv)
    assert all(len(receipt["request_hash"]) == 64 for receipt in result.request_receipts)
    assert all(len(receipt["response_hash"]) == 64 for receipt in result.request_receipts)
    assert [receipt["status"] for receipt in result.request_receipts] == ["partial", "success"]
    bar = result.normalized_ohlcv[0]
    assert bar["raw_base_volume_unit"] == "base_asset_units"
    assert bar["raw_quote_volume"] is None
    assert bar["raw_quote_volume_unit"] == "unavailable"
    assert bar["usd_notional_volume"] is None
    assert bar["usd_notional_volume_source"] == "unavailable"
    assert result.freshness_receipt["market_metadata_order"] == "first_order"
    assert result.freshness_receipt["ohlcv_order"] == "second_order"


def test_exact_maximum_page_count_is_not_falsely_rejected():
    exchange = FakeExchange(lambda since, limit: [_bar(since)])
    result = _runtime(exchange).capture(
        _request(end_ms=100 * MINUTE_MS, captured_at_ms=100 * MINUTE_MS, page_limit=1)
    )

    assert result.accepted
    assert len(result.request_receipts) == 100
    assert result.coverage_receipt["returned_complete_bar_count"] == 100


def test_response_larger_than_bounded_request_is_rejected():
    exchange = FakeExchange(lambda since, limit: [_bar(since), _bar(since + MINUTE_MS)])
    result = _runtime(exchange).capture(_request(page_limit=1))

    assert result.status == "data_quality_failure"
    assert "fetch_ohlcv:response_exceeds_requested_limit" in result.blockers
    assert result.request_receipts[0]["status"] == "malformed"


@pytest.mark.parametrize(
    ("failure", "expected_status", "expected_receipt_status"),
    [
        (RequestTimeout("secret message"), "retryable_failure", "retryable_failure"),
        (RateLimitExceeded("secret message"), "retryable_failure", "retryable_failure"),
        (RuntimeError("secret message"), "terminal_failure", "terminal_failure"),
    ],
)
def test_fetch_failures_have_deterministic_redacted_receipts(
    failure, expected_status, expected_receipt_status
):
    def fail(_since: int, _limit: int):
        raise failure

    result = _runtime(FakeExchange(fail)).capture(_request())
    serialized = json.dumps(result.receipt_dict(), sort_keys=True)

    assert result.status == expected_status
    assert result.request_receipts[0]["status"] == expected_receipt_status
    assert result.request_receipts[0]["exception_class"] == type(failure).__name__
    assert "secret message" not in serialized
    assert not result.accepted


@pytest.mark.parametrize(
    ("page_fn", "expected_status", "expected_blocker"),
    [
        (
            lambda since, limit: {"not": "a page"},
            "data_quality_failure",
            "fetch_ohlcv:malformed_page",
        ),
        (
            lambda since, limit: object(),
            "data_quality_failure",
            "fetch_ohlcv:malformed_page",
        ),
        (lambda since, limit: [], "empty", "zero_complete_rows"),
        (
            lambda since, limit: [_bar(0), _bar(2 * MINUTE_MS)] if since == 0 else [],
            "partial",
            "partial_history_gap",
        ),
    ],
)
def test_malformed_empty_and_partial_history_fail_closed(
    page_fn, expected_status, expected_blocker
):
    result = _runtime(FakeExchange(page_fn)).capture(_request())

    assert result.status == expected_status
    assert expected_blocker in result.blockers
    assert result.coverage_receipt["exact_reconciliation"] is False
    assert not result.accepted


def test_conflicting_duplicate_is_a_data_quality_failure():
    result = _runtime(
        FakeExchange(
            lambda since, limit: (
                [_bar(0), _bar(0, close=101.5), _bar(MINUTE_MS)]
                if since == 0
                else [_bar(2 * MINUTE_MS)]
                if since == 2 * MINUTE_MS
                else []
            )
        )
    ).capture(_request(page_limit=4))

    assert result.status == "data_quality_failure"
    assert "conflicting_duplicate_bar" in result.blockers
    assert result.coverage_receipt["conflicting_duplicate_timestamps_ms"] == [0]
    assert not result.accepted


def test_stale_market_cache_forces_reload_and_reload_failure_is_deterministic():
    bars = [_bar(index * MINUTE_MS) for index in range(3)]
    exchange = FakeExchange(
        lambda since, limit: [row for row in bars if row[0] >= since][:limit],
        load_failure_at=2,
    )
    runtime = _runtime(exchange)

    assert runtime.capture(_request()).accepted
    stale = runtime.capture(_request(captured_at_ms=3_800_000))

    assert stale.status == "retryable_failure"
    assert stale.blockers == ("load_markets:retryable_failure:RequestTimeout",)
    assert stale.market_provenance["cache_status"] == "stale_reloaded"
    assert stale.market_provenance["exception_class"] == "RequestTimeout"
    assert exchange.load_calls == [False, True]


def test_force_reload_is_explicit_even_on_first_load():
    bars = [_bar(index * MINUTE_MS) for index in range(3)]
    exchange = FakeExchange(lambda since, limit: [row for row in bars if row[0] >= since][:limit])

    result = _runtime(exchange).capture(_request(force_market_reload=True))

    assert result.accepted
    assert exchange.load_calls == [True]
    assert result.market_provenance["cache_status"] == "miss"
    assert result.market_provenance["force_reload"] is True


@pytest.mark.parametrize(
    "request_case",
    [
        _request(since_ms=1),
        _request(end_ms=MINUTE_MS, captured_at_ms=0),
        _request(timeframe="13x"),
        _request(end_ms=10 * MINUTE_MS, captured_at_ms=2 * MINUTE_MS),
    ],
)
def test_invalid_utc_boundaries_fail_before_factory(request_case):
    calls: list[tuple[str, dict[str, object]]] = []
    result = _runtime(FakeExchange(lambda since, limit: []), factory_calls=calls).capture(
        request_case
    )

    assert result.status == "terminal_failure"
    assert calls == []
    assert result.blockers


def test_capability_and_timeframe_must_be_explicit():
    no_capability = FakeExchange(lambda since, limit: [])
    no_capability.has = {"fetchOHLCV": "emulated"}
    capability_result = _runtime(no_capability).capture(_request())

    no_timeframe = FakeExchange(lambda since, limit: [])
    no_timeframe.timeframes = {"5m": "5m"}
    timeframe_result = _runtime(no_timeframe).capture(_request())

    assert capability_result.blockers == ("fetch_ohlcv_capability_not_explicitly_true",)
    assert timeframe_result.blockers == ("timeframe_not_supported",)
    assert no_capability.load_calls == []
    assert no_timeframe.load_calls == []


def test_evidence_writer_is_idempotent_and_refuses_mutation(tmp_path):
    bars = [_bar(index * MINUTE_MS) for index in range(3)]
    exchange = FakeExchange(lambda since, limit: [row for row in bars if row[0] >= since][:limit])
    result = _runtime(exchange).capture(_request())

    first = write_ccxt_capture_evidence(result, root=tmp_path)
    second = write_ccxt_capture_evidence(result, root=tmp_path)

    assert first == second
    receipt = json.loads(first["receipt"].read_text(encoding="utf-8"))
    assert receipt["capture_id"] == result.capture_id
    assert receipt["status"] == "accepted"
    assert receipt["normalized_row_count"] == 3
    first["raw_ohlcv"].chmod(0o600)
    first["raw_ohlcv"].write_text("[]\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable_ccxt_evidence_conflict"):
        write_ccxt_capture_evidence(result, root=tmp_path)
