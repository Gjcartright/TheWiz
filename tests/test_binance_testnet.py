from __future__ import annotations

from dataclasses import replace

import pandas as pd

from quant_platform.binance_testnet import (
    BinanceSpotTestnetOrderAdapter,
    BinanceTestnetConfig,
    BinanceUsdmTestnetOrderAdapter,
    binance_testnet_pair_preflight,
    binance_testnet_preflight,
    execute_binance_testnet_pair,
)
from quant_platform.execution import ExecutionMode, OrderIntent, build_execution_venue, validate_venue_order_client_adapter


class FakeResponse:
    ok = True
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"orderId": 123, "origQty": "0.25", "avgPrice": "100.5"}


class FakeSession:
    def __init__(self):
        self.posts: list[dict[str, object]] = []
        self.gets: list[str] = []

    def post(self, url, *, params, headers, timeout):
        self.posts.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return FakeResponse()

    def get(self, url, *, timeout):
        self.gets.append(url)
        return FakeResponse()


class FakeExchangeInfoResponse(FakeResponse):
    def json(self):
        return {
            "symbols": [
                {
                    "symbol": "BTCUSDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001"},
                        {"filterType": "MIN_NOTIONAL", "minNotional": "5"},
                    ],
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "TRADING",
                    "filters": [
                        {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
                        {"filterType": "LOT_SIZE", "stepSize": "0.01", "minQty": "0.01"},
                    ],
                },
            ]
        }


class FakeExchangeInfoSession(FakeSession):
    def get(self, url, *, timeout):
        self.gets.append(url)
        return FakeExchangeInfoResponse()


class SecondLegFailureAdapter(BinanceUsdmTestnetOrderAdapter):
    def __init__(self):
        super().__init__(session=FakeSession())
        self.calls = 0

    def place_order(self, intent, config=None):
        self.calls += 1
        if self.calls == 1:
            return super().place_order(intent, config)
        if self.calls == 2:
            return replace(
                super().place_order(intent, config),
                order_id="second-leg-blocked",
                status="paper_blocked_exchange_rejected",
            )
        return super().place_order(intent, config)


def test_spot_testnet_blocks_without_credentials_or_submit_flag():
    adapter = BinanceSpotTestnetOrderAdapter(session=FakeSession())
    fill = adapter.place_order(OrderIntent(market="ETH-USDT", side="BUY", size=0.25))

    assert fill.status.startswith("paper_blocked_")
    assert "submit_orders_false" in fill.status
    assert "missing_api_key" in fill.status


def test_usdm_testnet_submits_only_to_official_testnet_endpoint():
    session = FakeSession()
    config = replace(
        BinanceTestnetConfig.usdm_testnet(),
        api_key="test-key",
        api_secret="test-secret",
        submit_orders=True,
    )
    adapter = BinanceUsdmTestnetOrderAdapter(session=session)

    fill = adapter.place_order(OrderIntent(market="BTC-USDT", side="SELL", size=0.25, limit_price=100.5), config)

    assert fill.status == "paper_submitted"
    assert session.posts[0]["url"] == "https://testnet.binancefuture.com/fapi/v1/order"
    assert session.posts[0]["params"]["symbol"] == "BTCUSDT"
    assert session.posts[0]["params"]["signature"]


def test_testnet_adapter_refuses_non_testnet_base_url():
    session = FakeSession()
    config = replace(
        BinanceTestnetConfig.spot_testnet(),
        base_url="https://api.binance.com",
        api_key="test-key",
        api_secret="test-secret",
        submit_orders=True,
    )
    fill = BinanceSpotTestnetOrderAdapter(session=session).place_order(OrderIntent(market="ETHUSDT", side="BUY", size=1), config)

    assert "unsafe_non_testnet_base_url" in fill.status
    assert session.posts == []


def test_preflight_masks_secrets_and_writes_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_KEY", "spot-key")
    monkeypatch.setenv("BINANCE_SPOT_TESTNET_API_SECRET", "spot-secret")
    monkeypatch.setenv("BINANCE_USDM_TESTNET_API_KEY", "futures-key")
    monkeypatch.setenv("BINANCE_USDM_TESTNET_API_SECRET", "futures-secret")

    frame = binance_testnet_preflight(root=tmp_path, session=FakeSession())

    assert isinstance(frame, pd.DataFrame)
    assert set(frame["lane"]) == {"spot", "usdm_futures"}
    assert bool(frame["api_key_present"].all())
    assert "spot-key" not in (tmp_path / "reports" / "active" / "binance_testnet_preflight.csv").read_text()
    assert (tmp_path / "reports" / "active" / "binance_testnet_preflight.md").exists()


def test_testnet_adapters_are_available_through_generic_venue_factory():
    config = BinanceTestnetConfig.usdm_testnet()
    adapter = BinanceUsdmTestnetOrderAdapter(session=FakeSession())
    venue = build_execution_venue("binance_usdm_testnet", config=config, order_client=adapter)

    fill = venue.place_order(OrderIntent(market="BTCUSDT", side="BUY", size=0.1))

    assert fill.status.startswith("paper_blocked_")
    assert validate_venue_order_client_adapter("binance_spot_testnet")["valid"] is True


def test_pair_preflight_reads_symbol_rules_and_keeps_submission_separate(tmp_path):
    frame = binance_testnet_pair_preflight(
        asset_x="BTC-USD",
        asset_y="ETH-USD",
        config=BinanceTestnetConfig.usdm_testnet(),
        root=tmp_path,
        session=FakeExchangeInfoSession(),
    )

    btc = frame.set_index("symbol").loc["BTCUSDT"]
    assert btc["tick_size"] == "0.10"
    assert btc["step_size"] == "0.001"
    assert bool(btc["tradable"]) is True
    assert (tmp_path / "reports" / "active" / "binance_testnet_pair_preflight.csv").exists()


def test_pair_execution_attempts_usdm_rollback_after_second_leg_failure(tmp_path):
    config = replace(
        BinanceTestnetConfig.usdm_testnet(),
        api_key="test-key",
        api_secret="test-secret",
        submit_orders=True,
    )
    adapter = SecondLegFailureAdapter()
    intents = (
        OrderIntent(market="BTCUSDT", side="BUY", size=0.01),
        OrderIntent(market="ETHUSDT", side="SELL", size=0.1),
    )

    result = execute_binance_testnet_pair(
        intents=intents,
        config=config,
        adapter=adapter,
        journal_path=tmp_path / "journal.csv",
    )

    assert result.status == "second_leg_failed_rollback_attempted"
    assert result.rollback_fills[0].status == "paper_submitted"
    rollback_request = adapter.session.posts[-1]["params"]
    assert rollback_request["side"] == "SELL"
    assert rollback_request["reduceOnly"] == "true"
    assert (tmp_path / "journal.csv").exists()
