import pytest

from quant_platform.execution import (
    DydxNetworkConfig,
    ExecutionMode,
    FillReport,
    OrderIntent,
    PaperDydxExecution,
    SpreadOrderPlan,
    append_paper_trading_record,
    build_dydx_indexer_adapter,
    build_dydx_order_client_adapter,
    build_market_neutral_spread_intents,
    build_execution_venue,
    DydxV4IndexerAdapter,
    block_paper_plan_for_execution_config,
    dydx_readiness_report,
    paper_trading_record,
    build_research_gated_paper_plan,
    submit_paper_plan,
    validate_dydx_order_client_adapter,
)


class FakeDydxClient:
    def __init__(self):
        self.orders = []

    def place_order(self, intent, config):
        self.orders.append((intent, config))
        return FillReport(
            order_id="fake-testnet-order",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.01,
            slippage_bps=1.0,
            status="paper_submitted",
        )


class FakeMarketsClient:
    async def get_perpetual_markets(self, market=None):
        return {"markets": {market: {"ticker": market, "status": "ACTIVE"}}}

    async def get_perpetual_market_historical_funding(self, market, limit=None):
        return {"historicalFunding": [{"ticker": market, "rate": "0.0001"}], "limit": limit}


class FakeIndexerClient:
    def __init__(self):
        self.markets = FakeMarketsClient()


def test_default_execution_venue_is_local_dry_run():
    venue = build_execution_venue()

    fill = venue.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0, limit_price=2500.0))

    assert fill.order_id == "dry-run"
    assert fill.status == "not_sent"


def test_paper_execution_uses_dydx_testnet_endpoints():
    config = DydxNetworkConfig.paper_testnet()
    venue = build_execution_venue(config)

    market_data = venue.market_data("ETH-USD")

    assert isinstance(venue, PaperDydxExecution)
    assert market_data["status"] == "paper"
    assert market_data["network"] == "dydx_testnet"
    assert market_data["rest_indexer"] == "https://indexer.v4testnet.dydx.exchange"
    assert market_data["websocket_indexer"] == "wss://indexer.v4testnet.dydx.exchange/v4/ws"
    assert config.faucet_url == "https://faucet.v4testnet.dydx.exchange"


def test_paper_execution_can_use_injected_indexer_market_data_client():
    config = DydxNetworkConfig.paper_testnet()
    indexer = DydxV4IndexerAdapter(config, raw_client=FakeIndexerClient())
    venue = PaperDydxExecution(config, market_data_client=indexer)

    market_data = venue.market_data("ETH-USD")
    funding = venue.funding("ETH-USD")

    assert market_data["source"] == "dydx_v4_indexer"
    assert market_data["payload"]["markets"]["ETH-USD"]["status"] == "ACTIVE"
    assert funding["payload"]["historicalFunding"][0]["rate"] == "0.0001"


def test_paper_execution_blocks_order_submission_by_default():
    venue = PaperDydxExecution()

    fill = venue.place_order(OrderIntent(market="BTC-USD", side="SELL", size=0.25, limit_price=65000.0))

    assert fill.order_id == "paper-not-submitted"
    assert fill.status == "paper_blocked_submit_orders_false"


def test_live_execution_is_not_enabled_from_factory():
    config = DydxNetworkConfig(mode=ExecutionMode.LIVE)

    with pytest.raises(NotImplementedError):
        build_execution_venue(config)


def test_paper_execution_blocks_submission_when_credentials_missing():
    config = DydxNetworkConfig.paper_testnet()
    config = DydxNetworkConfig(
        mode=config.mode,
        node_url=config.node_url,
        rest_indexer=config.rest_indexer,
        websocket_indexer=config.websocket_indexer,
        faucet_url=config.faucet_url,
        submit_orders=True,
    )
    venue = PaperDydxExecution(config)

    fill = venue.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0))

    assert fill.order_id == "paper-missing-credentials"
    assert fill.status == "paper_blocked_missing_credentials"


def test_paper_execution_blocks_when_authenticated_client_missing():
    base = DydxNetworkConfig.paper_testnet()
    config = DydxNetworkConfig(
        mode=base.mode,
        node_url=base.node_url,
        rest_indexer=base.rest_indexer,
        websocket_indexer=base.websocket_indexer,
        faucet_url=base.faucet_url,
        submit_orders=True,
        wallet_address="wallet",
        private_key="private",
    )
    venue = PaperDydxExecution(config)

    fill = venue.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0))

    assert fill.order_id == "paper-missing-client"
    assert fill.status == "paper_blocked_missing_client"


def test_paper_execution_uses_injected_dydx_client_when_ready():
    base = DydxNetworkConfig.paper_testnet()
    config = DydxNetworkConfig(
        mode=base.mode,
        node_url=base.node_url,
        rest_indexer=base.rest_indexer,
        websocket_indexer=base.websocket_indexer,
        faucet_url=base.faucet_url,
        submit_orders=True,
        wallet_address="wallet",
        private_key="private",
    )
    client = FakeDydxClient()
    venue = PaperDydxExecution(config, client=client)

    fill = venue.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0))

    assert fill.status == "paper_submitted"
    assert len(client.orders) == 1


def test_build_dydx_order_client_adapter_loads_module_object(tmp_path, monkeypatch):
    adapter_module = tmp_path / "fake_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport

class FakeOrderAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="loaded-adapter",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    adapter = build_dydx_order_client_adapter("fake_order_adapter:FakeOrderAdapter")
    fill = adapter.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0), DydxNetworkConfig.paper_testnet())

    assert fill.order_id == "loaded-adapter"
    assert fill.status == "paper_submitted"


def test_record_only_dydx_order_adapter_records_without_exchange_submission():
    adapter = build_dydx_order_client_adapter(
        "quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter"
    )
    config = DydxNetworkConfig.paper_testnet()

    fill = adapter.place_order(OrderIntent(market="ETH-USD", side="BUY", size=1.0), config)

    assert fill.order_id == "record-only-1"
    assert fill.status == "paper_recorded_not_submitted"
    assert adapter.exchange_submission_capable is False
    assert adapter.record_only is True
    assert adapter.orders[0]["market"] == "ETH-USD"
    assert adapter.orders[0]["submit_orders"] is False


def test_validate_record_only_dydx_order_adapter_contract():
    report = validate_dydx_order_client_adapter(
        "quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter"
    )

    assert report["configured"] is True
    assert report["importable"] is True
    assert report["has_place_order"] is True
    assert report["signature_accepts_intent_config"] is True
    assert report["exchange_submission_capable"] is False
    assert report["record_only"] is True
    assert report["valid"] is True


def test_validate_dydx_order_client_adapter_checks_contract_without_submitting(tmp_path, monkeypatch):
    adapter_module = tmp_path / "bad_order_adapter.py"
    adapter_module.write_text(
        """
class BadOrderAdapter:
    def place_order(self, intent):
        raise RuntimeError("should not be called")
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    report = validate_dydx_order_client_adapter("bad_order_adapter:BadOrderAdapter")

    assert report["configured"] is True
    assert report["importable"] is True
    assert report["has_place_order"] is True
    assert report["signature_accepts_intent_config"] is False
    assert report["valid"] is False
    assert report["error"] == "place_order must accept intent and config arguments"


def test_paper_testnet_config_loads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)

    config = DydxNetworkConfig.paper_testnet_from_env()

    assert config.mode == ExecutionMode.PAPER
    assert config.submit_orders is True
    assert config.wallet_address == "wallet"
    assert config.private_key == "private"
    assert config.paper_trading_blockers() == []


def test_paper_testnet_config_loads_endpoint_overrides_from_env(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_NODE_URL", "custom-node:443")
    monkeypatch.setenv("DYDX_TESTNET_REST_INDEXER", "https://custom-indexer")
    monkeypatch.setenv("DYDX_TESTNET_WEBSOCKET_INDEXER", "wss://custom-indexer/ws")
    monkeypatch.setenv("DYDX_TESTNET_FAUCET_URL", "https://custom-faucet")

    config = DydxNetworkConfig.paper_testnet_from_env()

    assert config.node_url == "custom-node:443"
    assert config.rest_indexer == "https://custom-indexer"
    assert config.websocket_indexer == "wss://custom-indexer/ws"
    assert config.faucet_url == "https://custom-faucet"


def test_paper_testnet_config_reports_blockers_when_not_ready(monkeypatch):
    monkeypatch.delenv("DYDX_TESTNET_WALLET_ADDRESS", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_PRIVATE_KEY", raising=False)
    monkeypatch.delenv("DYDX_TESTNET_SUBMIT_ORDERS", raising=False)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)

    config = DydxNetworkConfig.paper_testnet_from_env()

    assert config.paper_trading_blockers() == [
        "submit_orders_false",
        "missing_wallet_address",
        "missing_private_key",
        "missing_dydx_v4_client",
    ]


def test_dydx_readiness_report_masks_secret_values(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)
    monkeypatch.setattr("quant_platform.execution.dydx_indexer_adapter_available", lambda: False)

    report = dydx_readiness_report()

    assert report["wallet_address_present"] is True
    assert report["private_key_present"] is True
    assert report["dydx_v4_client_installed"] is False
    assert report["dydx_indexer_adapter_wired"] is False
    assert report["dydx_order_client_adapter_wired"] is False
    assert report["ready_for_paper_submission"] is False
    assert report["blockers"] == ["missing_dydx_v4_client", "missing_dydx_order_client_adapter"]


def test_dydx_readiness_requires_order_client_adapter(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)

    report = dydx_readiness_report()

    assert report["dydx_v4_client_installed"] is True
    assert report["dydx_indexer_adapter_wired"] is True
    assert report["ready_for_paper_submission"] is False
    assert report["blockers"] == ["missing_dydx_order_client_adapter"]


def test_build_dydx_indexer_adapter_returns_none_when_sdk_missing(monkeypatch):
    monkeypatch.setattr("quant_platform.execution.dydx_indexer_adapter_available", lambda: False)

    adapter = build_dydx_indexer_adapter(DydxNetworkConfig.paper_testnet())

    assert adapter is None


def test_market_neutral_spread_intents_use_hedge_ratio_and_beta():
    left, right = build_market_neutral_spread_intents("ETH-BTC", "LONG_SPREAD", 1000.0, hedge_ratio=2.0, beta=0.5)

    assert left.market == "ETH-USD"
    assert left.side == "SELL"
    assert right.market == "BTC-USD"
    assert right.side == "BUY"
    assert left.size == pytest.approx(500.0)
    assert right.size == pytest.approx(500.0)


def test_research_gated_paper_plan_blocks_rejected_strategy():
    import pandas as pd

    acceptance = pd.DataFrame(
        [{"strategy_id": 1, "production_eligible": False, "acceptance_reason": "passing_pairs<2"}]
    )

    plan = build_research_gated_paper_plan(
        {"pair": "ETH-BTC", "strategy_id": 1, "signal": 1, "hedge_ratio": 1.0, "beta": 1.0},
        acceptance,
        notional_usd=1000,
    )

    assert plan.status == "blocked"
    assert plan.reason == "research_rejected:passing_pairs<2"
    assert plan.intents == ()


def test_research_gated_paper_plan_submits_to_testnet_adapter_when_accepted():
    import pandas as pd

    acceptance = pd.DataFrame([{"strategy_id": 1, "production_eligible": True, "acceptance_reason": "passed"}])
    venue = PaperDydxExecution()

    plan = build_research_gated_paper_plan(
        {"pair": "ETH-BTC", "strategy_id": 1, "signal": -1, "hedge_ratio": 1.5, "beta": 1.0},
        acceptance,
        notional_usd=1000,
    )
    fills = submit_paper_plan(plan, venue)

    assert plan.status == "paper_ready"
    assert len(plan.intents) == 2
    assert [fill.status for fill in fills] == ["paper_blocked_submit_orders_false", "paper_blocked_submit_orders_false"]


def test_block_paper_plan_for_execution_config_preserves_intents_and_blocks_submission():
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )

    blocked = block_paper_plan_for_execution_config(plan, ["submit_orders_false", "missing_private_key"])

    assert blocked.status == "blocked"
    assert blocked.reason == "dydx_not_ready:submit_orders_false;missing_private_key"
    assert blocked.intents == plan.intents
    assert submit_paper_plan(blocked, PaperDydxExecution()) == []


def test_paper_trading_record_persists_plan_and_fill_details(tmp_path):
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )
    fill = FillReport(
        order_id="paper-not-submitted",
        market="ETH-USD",
        side="BUY",
        size=10.0,
        avg_price=0.0,
        fee=0.0,
        slippage_bps=0.0,
        status="paper_blocked_submit_orders_false",
    )

    path = append_paper_trading_record(
        paper_trading_record(plan, fills=[fill], blockers=["submit_orders_false"]),
        tmp_path / "paper_trading_journal.csv",
    )

    import pandas as pd

    journal = pd.read_csv(path)
    assert list(journal["pair"]) == ["ETH-BTC"]
    assert list(journal["plan_status"]) == ["paper_ready"]
    assert list(journal["blockers"]) == ["submit_orders_false"]
    assert "ETH-USD" in journal["intents_json"].iloc[0]
    assert "paper_blocked_submit_orders_false" in journal["fills_json"].iloc[0]
