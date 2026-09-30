import json
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import pandas as pd
import pytest

import quant_platform.dydx_sdk_order_adapter as dydx_adapter_module
import quant_platform.execution as execution_module
from quant_platform.dydx_sdk_order_adapter import DydxSdkOrderAdapter
from quant_platform.execution import (
    DydxNetworkConfig,
    DydxV4IndexerAdapter,
    ExecutionMode,
    FillReport,
    OrderIntent,
    PaperDydxExecution,
    PaperVenueExecution,
    SpreadOrderPlan,
    _split_pair,
    append_paper_outcome_record,
    append_paper_trading_record,
    block_paper_plan_for_execution_config,
    build_dydx_indexer_adapter,
    build_dydx_order_client_adapter,
    build_execution_venue,
    build_market_neutral_spread_intents,
    build_research_gated_paper_plan,
    build_venue_order_client_adapter,
    dydx_readiness_report,
    gmx_execution_compatibility_snapshot,
    hyperliquid_execution_compatibility_snapshot,
    injective_execution_compatibility_snapshot,
    paper_trading_record,
    refresh_current_paper_watch_positions,
    refresh_dydx_execution_compatibility_table,
    refresh_gmx_execution_compatibility_table,
    refresh_gmx_testnet_candidate_shortlist,
    refresh_hyperliquid_execution_compatibility_table,
    refresh_hyperliquid_testnet_candidate_shortlist,
    refresh_hyperliquid_testnet_market_inventory,
    refresh_injective_execution_compatibility_table,
    refresh_injective_mirror_candidate_queue,
    refresh_injective_spot_first_candidate_shortlist,
    refresh_injective_spot_supported_pair_universe,
    refresh_live_paper_trade_monitor,
    refresh_non_eth_route_submit_queue,
    refresh_paper_trade_decision_report,
    submit_paper_plan,
    validate_dydx_order_client_adapter,
    validate_venue_order_client_adapter,
)
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    external_effect_issuer_session,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
)


@pytest.mark.parametrize(
    "live",
    [
        {"checked": False, "open_markets": [], "positions": [], "blocker": "account_state_unverified"},
        {"checked": True, "open_markets": [], "positions": [], "blocker": "account_state_stale"},
    ],
)
def test_browser_flat_claim_cannot_replace_blocked_live_account_state(tmp_path, monkeypatch, live):
    execution_module.write_browser_account_state_override(True, root=tmp_path)
    monkeypatch.setattr(execution_module, "dydx_account_state_snapshot", lambda config=None: dict(live))

    effective = execution_module.effective_dydx_account_state_snapshot(root=tmp_path)

    assert effective["checked"] is live["checked"]
    assert effective["open_markets"] == live["open_markets"]
    assert effective["positions"] == live["positions"]
    assert effective["blocker"] == live["blocker"]
    assert effective["source"] == "dydx_indexer"
    assert effective["browser_override_authority"] is False


def test_dydx_sdk_order_adapter_client_id_stays_in_valid_range():
    adapter = DydxSdkOrderAdapter()
    adapter._client_id_seed = DydxSdkOrderAdapter._MAX_CLIENT_ID - 1

    first = adapter._next_client_id()
    second = adapter._next_client_id()

    assert first == DydxSdkOrderAdapter._MAX_CLIENT_ID
    assert second == 1


def test_hyperliquid_margin_tiers_use_official_maintenance_formula():
    meta = {
        "universe": [
            {"name": "SOL", "marginTableId": 10, "maxLeverage": 10},
            {"name": "DOGE", "marginTableId": 52, "maxLeverage": 10},
        ],
        "marginTables": [
            [
                52,
                {
                    "description": "tiered 10x",
                    "marginTiers": [
                        {"lowerBound": "0", "maxLeverage": 10},
                        {"lowerBound": "20000", "maxLeverage": 5},
                        {"lowerBound": "100000", "maxLeverage": 3},
                    ],
                },
            ]
        ],
    }

    tiers = execution_module._hyperliquid_margin_tier_frame(
        meta,
        universe=meta["universe"],
        checked_at="2026-08-08T08:00:00+00:00",
    )

    table_10 = tiers.loc[tiers["margin_table_id"].eq(10)].iloc[0]
    assert table_10["max_leverage"] == 10
    assert table_10["maintenance_margin_rate"] == pytest.approx(0.05)
    assert table_10["maintenance_deduction_usd"] == 0.0
    assert table_10["description"] == "single_tier_table_id_below_50"

    table_52 = tiers.loc[tiers["margin_table_id"].eq(52)].reset_index(drop=True)
    assert table_52["upper_bound_usd"].tolist()[:2] == [20000.0, 100000.0]
    assert table_52.loc[1, "maintenance_margin_rate"] == pytest.approx(0.1)
    assert table_52.loc[1, "maintenance_deduction_usd"] == pytest.approx(1000.0)
    assert table_52.loc[2, "maintenance_margin_rate"] == pytest.approx(1 / 6)
    assert table_52.loc[2, "maintenance_deduction_usd"] == pytest.approx(
        1000.0 + 100000.0 * ((1 / 6) - 0.1)
    )
    assert tiers["blocker"].eq("").all()


def test_hyperliquid_margin_tiers_retain_unresolved_referenced_table():
    tiers = execution_module._hyperliquid_margin_tier_frame(
        {"marginTables": []},
        universe=[{"name": "TEST", "marginTableId": 88, "maxLeverage": 7}],
    )

    assert len(tiers) == 1
    assert tiers.loc[0, "margin_table_id"] == 88
    assert tiers.loc[0, "blocker"] == "referenced_margin_table_missing_from_meta"


def test_dydx_sdk_order_adapter_reduce_only_requires_gate00g_before_close_position(monkeypatch, tmp_path):
    active = tmp_path / "reports" / "active"
    monkeypatch.setattr(dydx_adapter_module, "ROOT", tmp_path)
    monkeypatch.setattr(
        dydx_adapter_module,
        "EXECUTION_ATTEMPT_LOG",
        active / "dydx_execution_market_attempts.csv",
    )
    adapter = DydxSdkOrderAdapter()
    called = {}

    async def fake_close_position(intent, config):
        called["market"] = intent.market
        called["size"] = intent.size
        called["reduce_only"] = intent.reduce_only
        return {"txhash": "abc"}

    monkeypatch.setattr(adapter, "_close_position", fake_close_position)
    monkeypatch.setattr(adapter, "_capture_market_state", lambda market, config: __import__("asyncio").sleep(0, result={}))
    monkeypatch.setattr(adapter, "_confirm_market_state", lambda intent, config, before: __import__("asyncio").sleep(0, result={"confirmed": True, "avg_price": 0.0}))
    monkeypatch.setattr(adapter, "_run", lambda awaitable: __import__("asyncio").run(awaitable))

    config = DydxNetworkConfig.paper_testnet()
    config = DydxNetworkConfig(
        mode=config.mode,
        node_url=config.node_url,
        rest_indexer=config.rest_indexer,
        websocket_indexer=config.websocket_indexer,
        faucet_url=config.faucet_url,
        submit_orders=True,
        wallet_address="wallet",
        private_key="secret",
    )

    with pytest.raises(
        execution_module.EffectAuthorityError,
        match="gate00g_order_authority_missing",
    ):
        adapter.place_order(
            OrderIntent(
                market="ETH-USD",
                side="SELL",
                size=0.003,
                limit_price=1.0,
                reduce_only=True,
            ),
            config,
        )
    assert called == {}


def test_dydx_sdk_private_order_sink_requires_consumed_authorization(monkeypatch):
    adapter = DydxSdkOrderAdapter()
    captured = {}

    class FakeNode:
        async def latest_block_height(self):
            return 100

        async def place_order(self, wallet, order):
            captured["order"] = order
            return object()

    class FakeWallet:
        address = "wallet"

    class FakeMarket:
        def __init__(self, payload):
            self.payload = payload

        def order_id(self, address, subaccount_number, client_id, flags):
            return "order-id"

        def order(self, **kwargs):
            return kwargs

    class FakeMarketsClient:
        async def get_perpetual_markets(self, market):
            return {"markets": {market: {"oraclePrice": "1.0"}}}

    class FakeIndexerClient:
        def __init__(self, rest_indexer):
            self.markets = FakeMarketsClient()

    async def fake_connect_node(config):
        return FakeNode()

    async def fake_build_wallet(node, config):
        return FakeWallet()

    monkeypatch.setattr(adapter, "_connect_node", fake_connect_node)
    monkeypatch.setattr(adapter, "_build_wallet", fake_build_wallet)
    monkeypatch.setattr(dydx_adapter_module, "Market", FakeMarket)
    monkeypatch.setattr(dydx_adapter_module, "IndexerClient", FakeIndexerClient)

    config = DydxNetworkConfig.paper_testnet()
    with pytest.raises(TypeError, match="spec.*authorization"):
        adapter._run(
            adapter._place_order(
                OrderIntent(market="STX-USD", side="SELL", size=1.0),
                config,
            )
        )
    assert captured == {}


def test_dydx_sdk_order_adapter_non_reduce_attempts_configured_then_oegs_fallback():
    adapter = DydxSdkOrderAdapter()
    config = DydxNetworkConfig(
        mode=ExecutionMode.PAPER,
        node_url="test-dydx-grpc.kingnodes.com:443",
        rest_indexer="https://indexer.v4testnet.dydx.exchange",
        websocket_indexer="wss://indexer.v4testnet.dydx.exchange/v4/ws",
        faucet_url="https://faucet.v4testnet.dydx.exchange",
        submit_orders=True,
        wallet_address="wallet",
        private_key="secret",
    )

    attempts = adapter._attempt_specs(OrderIntent(market="BNB-USD", side="BUY", size=0.01), config)

    assert [attempt.submit_mode for attempt in attempts] == ["standard_order", "standard_order"]
    assert attempts[0].config.node_url == "test-dydx-grpc.kingnodes.com:443"
    assert attempts[1].config.node_url is None
    assert attempts[1].route_label == "sdk_default_route:sdk_builtin_testnet_node"


def test_dydx_sdk_order_adapter_default_route_uses_sdk_builtin_testnet_node():
    config = DydxNetworkConfig(
        mode=ExecutionMode.PAPER,
        node_url="test-dydx-grpc.kingnodes.com:443",
        rest_indexer="https://indexer.v4testnet.dydx.exchange",
        websocket_indexer="wss://indexer.v4testnet.dydx.exchange/v4/ws",
        faucet_url="https://faucet.v4testnet.dydx.exchange",
        submit_orders=True,
        wallet_address="wallet",
        private_key="secret",
    )

    default_config = DydxSdkOrderAdapter._default_route_config(config)

    assert default_config.node_url is None


def test_refresh_dydx_execution_compatibility_table_uses_any_confirmed_route_as_gating_factor(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    attempts = tmp_path / "reports" / "active" / "dydx_execution_market_attempts.csv"
    attempts.parent.mkdir(parents=True)
    attempts.write_text(
        "\n".join(
            [
                "timestamp_utc,market,side,size,reduce_only,route_label,submit_mode,node_url,rest_indexer,order_id,avg_price,status,confirmed",
                "2026-07-07T17:00:00+00:00,BNB-USD,BUY,1.0,False,configured_route:test,standard_order,test,idx,one,1.0,confirmed_on_exchange,True",
                "2026-07-07T17:05:00+00:00,BNB-USD,BUY,1.0,False,sdk_default_route:sdk_builtin_testnet_node,standard_order,,idx,two,1.0,broadcast_accepted_unconfirmed,False",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_dydx_execution_compatibility_table(root=tmp_path)

    row = frame.set_index("market").loc["BNB-USD"]
    assert bool(row["compatible_for_paper_submit"]) is True
    assert row["last_status"] == "broadcast_accepted_unconfirmed"
    assert row["confirmed_attempts"] == 1
    assert row["unconfirmed_attempts"] == 1
    assert row["confirmed_route_labels"] == "configured_route:test"


def test_refresh_dydx_execution_compatibility_table_requires_latest_confirmation(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    attempts = tmp_path / "reports" / "active" / "dydx_execution_market_attempts.csv"
    attempts.parent.mkdir(parents=True)
    attempts.write_text(
        "\n".join(
            [
                "timestamp_utc,market,side,size,reduce_only,route_label,submit_mode,node_url,rest_indexer,order_id,avg_price,status,confirmed",
                "2026-07-07T17:00:00+00:00,ETH-USD,BUY,1.0,False,configured_route:test,standard_order,test,idx,one,1.0,broadcast_accepted_unconfirmed,False",
                "2026-07-07T17:05:00+00:00,ETH-USD,BUY,1.0,False,sdk_default_route:sdk_builtin_testnet_node,standard_order,,idx,two,1.0,confirmed_on_exchange,True",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_dydx_execution_compatibility_table(root=tmp_path)

    row = frame.set_index("market").loc["ETH-USD"]
    assert bool(row["compatible_for_paper_submit"]) is True
    assert row["last_status"] == "confirmed_on_exchange"
    assert row["confirmed_attempts"] == 1
    assert row["unconfirmed_attempts"] == 1


def test_refresh_non_eth_route_submit_queue_updates_submit_state_from_compatibility(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "dydx_execution_market_compatibility.csv").write_text(
        "\n".join(
            [
                "market,compatible_for_paper_submit,confirmed_attempts,unconfirmed_attempts,blocked_attempts,total_attempts,last_status,last_route_label,last_submit_mode,last_attempt_timestamp_utc,confirmed_route_labels",
                "BNB-USD,True,1,0,0,1,confirmed_on_exchange,configured_route:test,standard_order,2026-07-07T17:00:00+00:00,configured_route:test",
                "LDO-USD,False,0,1,0,1,broadcast_accepted_unconfirmed,configured_route:test,standard_order,2026-07-07T17:01:00+00:00,",
            ]
        ),
        encoding="utf-8",
    )
    (active / "non_eth_route_submit_queue.csv").write_text(
        "\n".join(
            [
                "submit_priority_rank,pair,candidate_id,setup_identity,lane_bias,timeframes,decision_bucket,route_markets,route_market_statuses,all_route_markets_confirmed,current_submit_state,next_action,evidence_path",
                "1,BNB-USD/LDO-USD,candidate-1,setup-1,wizard,daily,PROMOTE,BNB-USD;LDO-USD,,,wait_for_exchange_confirmation,,evidence.csv",
                "2,BNB-USD/BNB-USD,candidate-2,setup-2,wizard,daily,PROMOTE,BNB-USD,,,wait_for_exchange_confirmation,,evidence.csv",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_non_eth_route_submit_queue(root=tmp_path)

    first = frame.iloc[0]
    second = frame.iloc[1]
    assert first["route_market_statuses"] == "BNB-USD:confirmed_route_available@configured_route:test;LDO-USD:broadcast_accepted_unconfirmed"
    assert bool(first["all_route_markets_confirmed"]) is False
    assert first["current_submit_state"] == "wait_for_exchange_confirmation"
    assert bool(second["all_route_markets_confirmed"]) is True
    assert second["current_submit_state"] == "ready_for_paper_submit"


def test_refresh_injective_execution_compatibility_table_marks_testnet_spot_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    route_candidates = tmp_path / "reports" / "rl" / "base_rl_route_candidates.csv"
    route_candidates.parent.mkdir(parents=True)
    route_candidates.write_text(
        "\n".join(
            [
                "pair,candidate_id",
                "BNB-USD/ETC-USD,candidate-1",
                "BNB-USD/LDO-USD,candidate-2",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quant_platform.execution._injective_market_indexes",
        lambda: (
            {"BNB": ["BNB/USDT"], "ETC": ["ETC/USDT"]},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            {"BNB": ["BNB/USDT"], "ETC": ["ETC/USDT"]},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            [],
        ),
    )

    frame = refresh_injective_execution_compatibility_table(root=tmp_path)

    supported = frame.set_index("pair").loc["BNB-USD/ETC-USD"]
    blocked = frame.set_index("pair").loc["BNB-USD/LDO-USD"]
    assert bool(supported["mirrorable_for_paper"]) is True
    assert bool(supported["both_legs_testnet_spot"]) is True
    assert supported["injective_execution_mode"] == "injective_testnet_spot"
    assert bool(supported["both_legs_testnet_derivative"]) is True
    assert bool(blocked["mirrorable_for_paper"]) is False
    assert "missing_testnet_spot:LDO" in blocked["mirror_blocker"]


def test_refresh_injective_execution_compatibility_table_falls_back_to_route_submit_queue(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "non_eth_route_submit_queue.csv").write_text(
        "\n".join(
            [
                "submit_priority_rank,pair,candidate_id",
                "1,BNB-USD/ETC-USD,candidate-1",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quant_platform.execution._injective_market_indexes",
        lambda: (
            {"BNB": ["BNB/USDT"], "ETC": ["ETC/USDT"]},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            {"BNB": ["BNB/USDT"], "ETC": ["ETC/USDT"]},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            [],
        ),
    )

    frame = refresh_injective_execution_compatibility_table(root=tmp_path)

    row = frame.set_index("pair").loc["BNB-USD/ETC-USD"]
    assert bool(row["mirrorable_for_paper"]) is True
    assert row["candidate_id"] == "candidate-1"


def test_refresh_injective_execution_compatibility_table_blocks_derivative_only_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    route_candidates = tmp_path / "reports" / "rl" / "base_rl_route_candidates.csv"
    route_candidates.parent.mkdir(parents=True)
    route_candidates.write_text(
        "\n".join(
            [
                "pair,candidate_id",
                "BNB-USD/ETC-USD,candidate-1",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quant_platform.execution._injective_market_indexes",
        lambda: (
            {"BNB": ["BNB/USDT"], "ETC": ["ETC/USDT"]},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            {},
            {"BNB": ["BNB/USDC PERP"], "ETC": ["ETC/USDC PERP"]},
            [],
        ),
    )

    frame = refresh_injective_execution_compatibility_table(root=tmp_path)

    row = frame.set_index("pair").loc["BNB-USD/ETC-USD"]
    assert bool(row["mirrorable_for_paper"]) is False
    assert row["injective_execution_mode"] == ""
    assert "injective_spot_required" in row["mirror_blocker"]
    assert "missing_testnet_spot:BNB" in row["mirror_blocker"]
    assert "missing_testnet_spot:ETC" in row["mirror_blocker"]


def test_refresh_injective_mirror_candidate_queue_follows_compatibility_table(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    route_candidates = tmp_path / "reports" / "rl" / "base_rl_route_candidates.csv"
    route_candidates.parent.mkdir(parents=True)
    route_candidates.write_text(
        "\n".join(
            [
                "pair,candidate_id,setup_identity,available_timeframes,decision_bucket",
                "BNB-USD/ETC-USD,candidate-1,BNB-USD|ETC-USD|daily|ou_(spread),daily,PROMOTE",
                "BNB-USD/LDO-USD,candidate-2,BNB-USD|LDO-USD|daily|ou_(spread),daily,PROMOTE",
            ]
        ),
        encoding="utf-8",
    )
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "injective_execution_market_compatibility.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id,asset_x,asset_y,mainnet_spot_x,mainnet_derivative_x,testnet_spot_x,testnet_derivative_x,mainnet_spot_y,mainnet_derivative_y,testnet_spot_y,testnet_derivative_y,both_legs_found_mainnet,both_legs_found_testnet,both_legs_testnet_spot,both_legs_testnet_derivative,injective_execution_mode,mirrorable_for_paper,mirror_blocker,checked_at_utc",
                "BNB-USD/ETC-USD,candidate-1,BNB,ETC,BNB/USDT,BNB/USDC PERP,BNB/USDT,BNB/USDC PERP,ETC/USDT,ETC/USDC PERP,ETC/USDT,ETC/USDC PERP,True,True,True,True,injective_testnet_spot,True,,2026-07-07T00:00:00+00:00",
                "BNB-USD/LDO-USD,candidate-2,BNB,LDO,BNB/USDT,BNB/USDC PERP,BNB/USDT,BNB/USDC PERP,,,,True,False,False,False,,False,missing_testnet_spot:LDO,2026-07-07T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_injective_mirror_candidate_queue(root=tmp_path)

    first = frame.set_index("pair").loc["BNB-USD/ETC-USD"]
    second = frame.set_index("pair").loc["BNB-USD/LDO-USD"]
    assert bool(first["injective_mirrorable"]) is True
    assert first["injective_execution_mode"] == "injective_testnet_spot"
    assert first["next_action"] == "paper_plus_injective_spot"
    assert bool(second["injective_mirrorable"]) is False
    assert second["next_action"] == "paper_only_until_injective_spot_support_exists"


def test_refresh_injective_spot_supported_pair_universe_marks_supported_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "pair_universe.csv").write_text(
        "\n".join(
            [
                "pair,combined_score,acceptance_score,decision_bucket,decision_reason,available_timeframes,best_execution_venue",
                "ETH-USD-TIA-USD,31.9,14.9,REJECT,global_filter,daily,dydx",
                "ETH-USD-SOL-USD,32.8,14.9,WATCH,needs_refresh,4HOUR,hyperliquid",
                "BNB-USD-LDO-USD,29.0,10.0,PROMOTE,ready,daily,dydx",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_platform.execution._injective_market_indexes",
        lambda: (
            {"ETH": ["ETH/USDT"], "SOL": ["SOL/USDT"], "TIA": ["TIA/USDT"]},
            {},
            {"ETH": ["ETH/INJ"], "SOL": ["SOL/USDT"], "TIA": ["TIA/USDT"]},
            {},
            [],
        ),
    )

    frame = refresh_injective_spot_supported_pair_universe(root=tmp_path)

    eth_tia = frame.set_index("pair").loc["ETH-USD-TIA-USD"]
    eth_sol = frame.set_index("pair").loc["ETH-USD-SOL-USD"]
    bnb_ldo = frame.set_index("pair").loc["BNB-USD-LDO-USD"]
    assert bool(eth_tia["injective_supported_for_spot"]) is True
    assert bool(eth_tia["both_legs_testnet_spot"]) is True
    assert eth_tia["injective_execution_mode"] == "injective_testnet_spot"
    assert bool(eth_sol["injective_supported_for_spot"]) is True
    assert bool(bnb_ldo["injective_supported_for_spot"]) is False
    assert "missing_testnet_spot:BNB" in bnb_ldo["injective_support_blocker"]
    assert "missing_testnet_spot:LDO" in bnb_ldo["injective_support_blocker"]


def test_refresh_injective_spot_first_candidate_shortlist_prioritizes_supported_watch_over_reject(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    (processed / "pair_universe.csv").write_text(
        "\n".join(
            [
                "pair,combined_score,acceptance_score,decision_bucket,decision_reason,available_timeframes,best_execution_venue",
                "ETH-USD-TIA-USD,31.9,14.9,REJECT,global_filter,daily,dydx",
                "ETH-USD-SOL-USD,32.8,14.9,WATCH,needs_refresh,4HOUR,hyperliquid",
                "BNB-USD-LDO-USD,29.0,10.0,PROMOTE,ready,daily,dydx",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "quant_platform.execution._injective_market_indexes",
        lambda: (
            {"ETH": ["ETH/USDT"], "SOL": ["SOL/USDT"], "TIA": ["TIA/USDT"]},
            {},
            {"ETH": ["ETH/INJ"], "SOL": ["SOL/USDT"], "TIA": ["TIA/USDT"]},
            {},
            [],
        ),
    )

    frame = refresh_injective_spot_first_candidate_shortlist(root=tmp_path, max_pairs=10)

    assert list(frame["pair"]) == ["ETH-USD-SOL-USD", "ETH-USD-TIA-USD"]
    first = frame.iloc[0]
    second = frame.iloc[1]
    assert first["injective_lane_status"] == "injective_supported_research_watch"
    assert first["next_action"] == "refresh_research_then_paper_plus_injective_spot"
    assert second["injective_lane_status"] == "injective_supported_but_research_rejected"
    assert second["next_action"] == "repair_research_filters_before_any_submit"


def test_injective_execution_compatibility_snapshot_reports_paper_only_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "injective_execution_market_compatibility.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id,asset_x,asset_y,mainnet_spot_x,mainnet_derivative_x,testnet_spot_x,testnet_derivative_x,mainnet_spot_y,mainnet_derivative_y,testnet_spot_y,testnet_derivative_y,both_legs_found_mainnet,both_legs_found_testnet,both_legs_testnet_spot,both_legs_testnet_derivative,injective_execution_mode,mirrorable_for_paper,mirror_blocker,checked_at_utc",
                "BNB-USD/ETC-USD,candidate-1,BNB,ETC,BNB/USDT,BNB/USDC PERP,BNB/USDT,BNB/USDC PERP,ETC/USDT,ETC/USDC PERP,ETC/USDT,ETC/USDC PERP,True,True,True,True,injective_testnet_spot,True,,2026-07-07T00:00:00+00:00",
                "BNB-USD/LDO-USD,candidate-2,BNB,LDO,BNB/USDT,BNB/USDC PERP,BNB/USDT,BNB/USDC PERP,,,,True,False,False,False,,False,missing_testnet_spot:LDO,2026-07-07T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = injective_execution_compatibility_snapshot(["BNB-USD/ETC-USD", "BNB-USD/LDO-USD"], root=tmp_path)

    assert snapshot["checked"] is True
    assert snapshot["mirrorable_pairs"] == ["BNB-USD/ETC-USD"]
    assert snapshot["spot_pairs"] == ["BNB-USD/ETC-USD"]
    assert snapshot["derivative_pairs"] == []
    assert snapshot["paper_only_pairs"] == ["BNB-USD/LDO-USD"]
    assert snapshot["blocker"] == "injective_testnet_spot_support_incomplete"


def test_refresh_gmx_execution_compatibility_table_marks_testnet_perp_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "fresh_live_dashboard_pair_queue.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id",
                "BTC-USD/ETH-USD,candidate-1",
                "BNB-USD/ETH-USD,candidate-2",
            ]
        ),
        encoding="utf-8",
    )
    (active / "gmx_testnet_market_inventory.csv").write_text(
        "\n".join(
            [
                "asset,market_symbol,tradable_perp,fetch_blocker",
                "BTC,BTC/USD [BTC-USDC.SG],True,",
                "ETH,ETH/USD [WETH-USDC.SG],True,",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_gmx_execution_compatibility_table(root=tmp_path)

    supported = frame.set_index("pair").loc["BTC-USD/ETH-USD"]
    blocked = frame.set_index("pair").loc["BNB-USD/ETH-USD"]
    assert bool(supported["mirrorable_for_paper"]) is True
    assert bool(supported["both_legs_testnet_perp"]) is True
    assert supported["gmx_execution_mode"] == "gmx_arbitrum_sepolia_perp"
    assert bool(blocked["mirrorable_for_paper"]) is False
    assert "missing_gmx_testnet_perp:BNB" in blocked["mirror_blocker"]


def test_refresh_gmx_testnet_candidate_shortlist_uses_supported_pairs_only(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    (processed / "pair_universe.csv").write_text(
        "\n".join(
            [
                "pair,combined_score,acceptance_score,decision_bucket,decision_reason,available_timeframes,best_execution_venue",
                "BTC-USD-ETH-USD,113.0,75.0,PROMOTE,ready,5MINS,dydx",
                "BNB-USD-ETH-USD,99.0,70.0,PROMOTE,ready,daily,dydx",
            ]
        ),
        encoding="utf-8",
    )
    (active / "gmx_testnet_market_inventory.csv").write_text(
        "\n".join(
            [
                "asset,market_symbol,tradable_perp,fetch_blocker",
                "BTC,BTC/USD [BTC-USDC.SG],True,",
                "ETH,ETH/USD [WETH-USDC.SG],True,",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_gmx_testnet_candidate_shortlist(root=tmp_path, max_pairs=10)

    assert list(frame["pair"]) == ["BTC-USD-ETH-USD"]
    first = frame.iloc[0]
    assert first["gmx_lane_status"] == "gmx_supported_and_promoted"
    assert first["gmx_execution_mode"] == "gmx_arbitrum_sepolia_perp"


def test_gmx_execution_compatibility_snapshot_reports_incomplete_pairs(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "gmx_execution_market_compatibility.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id,asset_x,asset_y,testnet_perp_x,testnet_perp_y,both_legs_testnet_perp,gmx_execution_mode,mirrorable_for_paper,mirror_blocker,checked_at_utc",
                "BTC-USD/ETH-USD,candidate-1,BTC,ETH,BTC/USD [BTC-USDC.SG],ETH/USD [WETH-USDC.SG],True,gmx_arbitrum_sepolia_perp,True,,2026-07-10T00:00:00+00:00",
                "BNB-USD/ETH-USD,candidate-2,BNB,ETH,,ETH/USD [WETH-USDC.SG],False,,False,missing_gmx_testnet_perp:BNB,2026-07-10T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = gmx_execution_compatibility_snapshot(["BTC-USD/ETH-USD", "BNB-USD/ETH-USD"], root=tmp_path)

    assert snapshot["checked"] is True
    assert snapshot["mirrorable_pairs"] == ["BTC-USD/ETH-USD"]
    assert snapshot["paper_only_pairs"] == ["BNB-USD/ETH-USD"]
    assert snapshot["blocker"] == "gmx_testnet_perp_support_incomplete"


def test_refresh_hyperliquid_execution_compatibility_table_marks_testnet_perp_pairs(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    (processed / "pair_universe.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id",
                "BTC-USD-HYPE-USD,candidate-1",
                "BNB-USD-BOGUS-USD,candidate-2",
            ]
        ),
        encoding="utf-8",
    )
    (active / "hyperliquid_testnet_market_inventory.csv").write_text(
        "\n".join(
            [
                "asset,asset_index,universe_name,tradable_perp,fetch_blocker",
                "BTC,3,BTC,True,",
                "HYPE,150,HYPE,True,",
                "BNB,6,BNB,True,",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_hyperliquid_execution_compatibility_table(root=tmp_path)

    supported = frame.set_index("pair").loc["BTC-USD-HYPE-USD"]
    blocked = frame.set_index("pair").loc["BNB-USD-BOGUS-USD"]
    assert bool(supported["mirrorable_for_paper"]) is True
    assert bool(supported["both_legs_testnet_perp"]) is True
    assert supported["hyperliquid_execution_mode"] == "hyperliquid_testnet_perp"
    assert bool(blocked["mirrorable_for_paper"]) is False
    assert "missing_hyperliquid_testnet_perp:BOGUS" in blocked["mirror_blocker"]


def test_hyperliquid_market_inventory_rejects_string_false_tradability(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([
        {"asset": "BTC", "asset_index": 3, "universe_name": "BTC", "tradable_perp": "False"},
        {"asset": "ETH", "asset_index": 4, "universe_name": "ETH", "tradable_perp": "True"},
        {"asset": "DOGE", "asset_index": 5, "universe_name": "DOGE", "tradable_perp": "unknown"},
    ]).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)

    market_index, _ = execution_module._hyperliquid_market_indexes(root=tmp_path)

    assert set(market_index) == {"ETH"}


def test_hyperliquid_shortlist_rejects_string_false_mirrorability(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([
        {"pair": "BTC-USD-HYPE-USD", "mirrorable_for_paper": "False"},
        {"pair": "ETH-USD-SOL-USD", "mirrorable_for_paper": "unknown"},
    ]).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)

    shortlist = refresh_hyperliquid_testnet_candidate_shortlist(root=tmp_path)

    assert shortlist.empty


def test_refresh_hyperliquid_testnet_candidate_shortlist_uses_supported_pairs_only(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    (processed / "pair_universe.csv").write_text(
        "\n".join(
            [
                "pair,combined_score,acceptance_score,decision_bucket,decision_reason,available_timeframes,best_execution_venue",
                "BTC-USD-HYPE-USD,113.0,75.0,PROMOTE,ready,5MINS,hyperliquid",
                "BNB-USD-BOGUS-USD,99.0,70.0,PROMOTE,ready,5MINS,hyperliquid",
            ]
        ),
        encoding="utf-8",
    )
    (active / "hyperliquid_testnet_market_inventory.csv").write_text(
        "\n".join(
            [
                "asset,asset_index,universe_name,tradable_perp,fetch_blocker",
                "BTC,3,BTC,True,",
                "HYPE,150,HYPE,True,",
                "BNB,6,BNB,True,",
            ]
        ),
        encoding="utf-8",
    )

    frame = refresh_hyperliquid_testnet_candidate_shortlist(root=tmp_path, max_pairs=10)

    assert list(frame["pair"]) == ["BTC-USD-HYPE-USD"]
    first = frame.iloc[0]
    assert first["hyperliquid_lane_status"] == "hyperliquid_supported_promoted_preflight_blocked"
    assert first["hyperliquid_execution_mode"] == "hyperliquid_testnet_perp"
    assert bool(first["order_preflight_ready"]) is False
    assert "missing_or_invalid_hyperliquid_master_address" in first["order_preflight_blocker"]


def test_hyperliquid_execution_compatibility_snapshot_reports_preflight_blocker(tmp_path, monkeypatch):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.delenv("HYPERLIQUID_PAPER_ORDER_ADAPTER", raising=False)
    (active / "hyperliquid_execution_market_compatibility.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id,asset_x,asset_y,testnet_perp_x,testnet_perp_y,both_legs_testnet_perp,hyperliquid_execution_mode,mirrorable_for_paper,mirror_blocker,checked_at_utc",
                "BTC-USD/HYPE-USD,candidate-1,BTC,HYPE,BTC#3,HYPE#150,True,hyperliquid_testnet_perp,True,,2026-07-10T00:00:00+00:00",
                "BNB-USD/BOGUS-USD,candidate-2,BNB,BOGUS,BNB#6,,False,,False,missing_hyperliquid_testnet_perp:BOGUS,2026-07-10T00:00:00+00:00",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = hyperliquid_execution_compatibility_snapshot(["BTC-USD/HYPE-USD", "BNB-USD/BOGUS-USD"], root=tmp_path)

    assert snapshot["checked"] is True
    assert snapshot["mirrorable_pairs"] == ["BTC-USD/HYPE-USD"]
    assert snapshot["paper_only_pairs"] == ["BNB-USD/BOGUS-USD"]
    assert snapshot["blocker"] == "hyperliquid_testnet_perp_support_incomplete"
    assert snapshot["order_preflight_ready"] is False
    assert "missing_or_invalid_hyperliquid_master_address" in snapshot["order_preflight_blocker"]


def test_split_pair_supports_full_usd_pair_format():
    assert _split_pair("BTC-USD-LDO-USD") == ("BTC-USD", "LDO-USD")


def test_refresh_paper_trade_decision_report_recommends_close_for_open_pair_that_left_shortlist(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    reports = tmp_path / "reports"
    (reports / "active").mkdir(parents=True)
    journal = reports / "paper_trading_journal.csv"
    journal.write_text(
        "\n".join(
            [
                "timestamp_utc,pair,strategy_id,plan_status,plan_reason,blockers,intents_json,fills_json,trade_id,venue,lifecycle_status,opened_timestamp_utc,closed_timestamp_utc,entry_snapshot_json,exit_snapshot_json,realized_return,outcome_label",
                '2026-07-08T00:00:00+00:00,BNB-USD/ETC-USD,1,paper_submitted,accepted,,[],[],trade-1,dydx,open,2026-07-08T00:00:00+00:00,,{},{},,',
            ]
        ),
        encoding="utf-8",
    )
    (reports / "paper_execution_preflight.csv").write_text(
        "\n".join(
            [
                "step,ready,status,blocker,evidence,next_action",
                "paper_submission_gate,True,ready,,,",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "active" / "non_eth_route_submit_queue.csv").write_text(
        "submit_priority_rank,pair,candidate_id,setup_identity,lane_bias,timeframes,decision_bucket,route_markets,route_market_statuses,all_route_markets_confirmed,current_submit_state,next_action,evidence_path\n",
        encoding="utf-8",
    )
    (reports / "active" / "injective_mirror_candidate_queue.csv").write_text(
        "pair,candidate_id,strategy_mode,timeframe,paper_priority,injective_mirrorable,injective_execution_mode,mirror_blocker,next_action\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quant_platform.active_pipeline.paper_candidate_shortlist_rows",
        lambda root, max_pairs=10: pd.DataFrame(
            [
                {
                    "shortlist_rank": 1,
                    "pair": "ATOM-USD/ETH-USD",
                    "candidate_id": "candidate-1",
                    "best_execution_venue": "dydx",
                }
            ]
        ),
    )

    frame = refresh_paper_trade_decision_report(root=tmp_path)
    row = frame.set_index("pair").loc["BNB-USD/ETC-USD"]
    assert row["recommendation"] == "close"
    assert row["reason"] == "pair_left_shortlist"


def test_refresh_paper_trade_decision_report_prefers_injective_spot_open_when_supported(tmp_path, monkeypatch):
    monkeypatch.setattr("quant_platform.execution.ROOT", tmp_path)
    reports = tmp_path / "reports"
    (reports / "active").mkdir(parents=True)
    (reports / "paper_trading_journal.csv").write_text(
        "timestamp_utc,pair,strategy_id,plan_status,plan_reason,blockers,intents_json,fills_json,trade_id,venue,lifecycle_status,opened_timestamp_utc,closed_timestamp_utc,entry_snapshot_json,exit_snapshot_json,realized_return,outcome_label\n",
        encoding="utf-8",
    )
    (reports / "paper_execution_preflight.csv").write_text(
        "\n".join(
            [
                "step,ready,status,blocker,evidence,next_action",
                "paper_submission_gate,False,blocked,strategy_or_dydx_gate_not_ready,,",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "active" / "non_eth_route_submit_queue.csv").write_text(
        "\n".join(
            [
                "submit_priority_rank,pair,candidate_id,setup_identity,lane_bias,timeframes,decision_bucket,route_markets,route_market_statuses,all_route_markets_confirmed,current_submit_state,next_action,evidence_path",
                "1,ATOM-USD/ETH-USD,candidate-1,setup-1,wizard,daily,PROMOTE,ATOM-USD;ETH-USD,,,wait_for_exchange_confirmation,,evidence.csv",
            ]
        ),
        encoding="utf-8",
    )
    (reports / "active" / "injective_mirror_candidate_queue.csv").write_text(
        "\n".join(
            [
                "pair,candidate_id,strategy_mode,timeframe,paper_priority,injective_mirrorable,injective_execution_mode,mirror_blocker,next_action",
                "ATOM-USD/ETH-USD,candidate-1,setup-1,daily,PROMOTE,True,injective_testnet_spot,,paper_plus_injective_spot",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "quant_platform.active_pipeline.paper_candidate_shortlist_rows",
        lambda root, max_pairs=10: pd.DataFrame(
            [
                {
                    "shortlist_rank": 1,
                    "pair": "ATOM-USD/ETH-USD",
                    "candidate_id": "candidate-1",
                    "best_execution_venue": "dydx",
                }
            ]
        ),
    )

    frame = refresh_paper_trade_decision_report(root=tmp_path)
    row = frame.set_index("pair").loc["ATOM-USD/ETH-USD"]
    assert row["recommendation"] == "open"
    assert row["venue"] == "injective_spot"
    assert row["reason"] == "injective_testnet_spot_supported"


def test_market_neutral_spread_intents_use_asset_sizes_when_prices_present():
    left, right = build_market_neutral_spread_intents(
        pair="BTC-USD-LDO-USD",
        side="LONG_SPREAD",
        notional_usd=10.0,
        hedge_ratio=96489.364422,
        beta=0.554325,
        venue="dydx",
        price_x=62998.30126,
        price_y=0.32175498,
    )

    assert left.market == "BTC-USD"
    assert right.market == "LDO-USD"
    assert left.side == "SELL"
    assert right.side == "BUY"
    assert 0 < left.size < 0.001
    assert right.size > 1.0


def test_market_neutral_spread_intents_quantize_to_step_sizes():
    left, right = build_market_neutral_spread_intents(
        pair="BTC-USD-LDO-USD",
        side="LONG_SPREAD",
        notional_usd=30.0,
        hedge_ratio=96489.364422,
        beta=0.554325,
        venue="dydx",
        price_x=62998.30126,
        price_y=0.32175498,
        step_size_x=0.0001,
        step_size_y=1.0,
    )

    assert left.size == 0.0001
    assert right.size == 73.0


def test_research_gated_paper_plan_blocks_when_quantized_size_hits_zero():
    import pandas as pd

    acceptance = pd.DataFrame([{"strategy_id": 9001, "production_eligible": True, "acceptance_reason": "passed"}])

    plan = build_research_gated_paper_plan(
        {
            "pair": "BTC-USD-LDO-USD",
            "strategy_id": 9001,
            "signal": 1.0,
            "hedge_ratio": 96489.364422,
            "beta": 0.554325,
            "price_x": 62998.30126,
            "price_y": 0.32175498,
            "step_size_x": 0.0001,
            "step_size_y": 1.0,
        },
        acceptance,
        notional_usd=10.0,
    )

    assert plan.status == "blocked"
    assert plan.reason == "venue_size_below_minimum:BTC-USD"


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


class FakeVenueClient:
    def __init__(self):
        self.orders = []

    def place_order(self, intent, config):
        self.orders.append((intent, config))
        return FillReport(
            order_id="fake-venue-order",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
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


def test_paper_execution_blocks_missing_client_without_inspecting_credentials():
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

    assert fill.order_id == "paper-missing-client"
    assert fill.status == "paper_blocked_missing_client"


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


def test_paper_execution_denies_unfenced_injected_dydx_client():
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

    assert fill.status == "paper_blocked_gate00g_order_authority_required"
    assert len(client.orders) == 0


def test_build_dydx_order_client_adapter_denies_unapproved_module_before_import(tmp_path, monkeypatch):
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

    with pytest.raises(
        execution_module.EffectAuthorityError,
        match="gate00g_order_adapter_path_denied",
    ):
        build_dydx_order_client_adapter("fake_order_adapter:FakeOrderAdapter")
    assert "fake_order_adapter" not in __import__("sys").modules


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


def test_build_dydx_order_client_adapter_falls_back_to_builtin_sdk_adapter(monkeypatch):
    monkeypatch.delenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", raising=False)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)

    adapter = build_dydx_order_client_adapter()

    assert adapter is not None
    assert adapter.exchange_submission_capable is True
    assert adapter.record_only is False


def test_validate_dydx_order_client_adapter_uses_builtin_sdk_fallback(monkeypatch):
    monkeypatch.delenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER", raising=False)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: True)

    report = validate_dydx_order_client_adapter()

    assert report["configured"] is True
    assert report["importable"] is True
    assert report["has_place_order"] is True
    assert report["signature_accepts_intent_config"] is True
    assert report["exchange_submission_capable"] is True
    assert report["record_only"] is False
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
    assert report["importable"] is False
    assert report["has_place_order"] is False
    assert report["signature_accepts_intent_config"] is False
    assert report["valid"] is False
    assert report["error"] == "gate00g_order_adapter_path_denied"


def test_venue_specific_order_client_denies_unapproved_env_adapter(tmp_path, monkeypatch):
    adapter_module = tmp_path / "venue_order_adapter.py"
    adapter_module.write_text(
        """
from quant_platform.execution import FillReport


class VenueAdapter:
    def place_order(self, intent, config):
        return FillReport(
            order_id="loaded-venue-adapter",
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
    monkeypatch.setenv("HYPERLIQUID_PAPER_ORDER_ADAPTER", "venue_order_adapter:VenueAdapter")

    with pytest.raises(
        execution_module.EffectAuthorityError,
        match="gate00g_order_adapter_path_denied",
    ):
        build_venue_order_client_adapter("hyperliquid")
    assert "venue_order_adapter" not in __import__("sys").modules


def test_build_execution_venue_blocks_unfenced_generic_client():
    client = FakeVenueClient()
    venue = build_execution_venue("hyperliquid", config=DydxNetworkConfig.paper_testnet(), order_client=client)

    fill = venue.place_order(OrderIntent(market="BTC-USD", side="BUY", size=0.25))

    assert isinstance(venue, PaperVenueExecution)
    assert fill.status == "paper_blocked_gate00g_order_authority_required"
    assert client.orders == []


def test_validate_venue_order_client_adapter_checks_venue_contract():
    report = validate_venue_order_client_adapter(
        "hyperliquid",
        adapter_path="quant_platform.dydx_record_only_adapter:RecordOnlyDydxOrderAdapter",
    )

    assert report["configured"] is True
    assert report["venue"] == "hyperliquid"
    assert report["valid"] is True


def test_paper_testnet_config_loads_credentials_from_env(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setenv("DYDX_TESTNET_ORDER_APPROVAL_ID", "approval-1")
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
    monkeypatch.delenv("DYDX_TESTNET_ORDER_APPROVAL_ID", raising=False)
    monkeypatch.setattr("quant_platform.execution.dydx_v4_client_installed", lambda: False)

    config = DydxNetworkConfig.paper_testnet_from_env()

    assert config.paper_trading_blockers() == [
        "submit_orders_false",
        "missing_wallet_address",
        "missing_private_key",
        "missing_order_approval_id",
        "missing_dydx_v4_client",
    ]


def test_dydx_readiness_report_masks_secret_values(monkeypatch):
    monkeypatch.setenv("DYDX_TESTNET_WALLET_ADDRESS", "wallet")
    monkeypatch.setenv("DYDX_TESTNET_PRIVATE_KEY", "private")
    monkeypatch.setenv("DYDX_TESTNET_SUBMIT_ORDERS", "true")
    monkeypatch.setenv("DYDX_TESTNET_ORDER_APPROVAL_ID", "approval-1")
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
    monkeypatch.setenv("DYDX_TESTNET_ORDER_APPROVAL_ID", "approval-1")
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
    assert [fill.status for fill in fills] == ["paper_blocked_submit_orders_false"]


def test_submit_paper_plan_denies_unfenced_coordinated_pair_path():
    class PairVenue:
        pair_submission_capable = True

        def __init__(self):
            self.calls = []

        def submit_pair(self, intents):
            self.calls.append(intents)
            return SimpleNamespace(
                status="pair_submitted",
                reason="submitted",
                fills=tuple(
                    FillReport(
                        order_id=f"pair-{index}",
                        market=intent.market,
                        side=intent.side,
                        size=intent.size,
                        avg_price=float(intent.limit_price or 0.0),
                        fee=0.0,
                        slippage_bps=0.0,
                        status="paper_submitted",
                    )
                    for index, intent in enumerate(intents)
                ),
            )

        def place_order(self, intent):
            raise AssertionError("single-leg path must not be used")

    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(
            OrderIntent(market="ETH-USD", side="BUY", size=1.0, limit_price=3000.0),
            OrderIntent(market="BTC-USD", side="SELL", size=0.05, limit_price=60000.0),
        ),
        venue="hyperliquid",
    )
    venue = PairVenue()

    fills = submit_paper_plan(plan, venue)

    assert venue.calls == []
    assert len(fills) == 2
    assert all(
        fill.status == "paper_blocked_gate00g_execution_venue_denied"
        for fill in fills
    )


def test_submit_paper_plan_turns_unfenced_pair_venue_into_auditable_blocked_fills():
    class BlockedPairVenue:
        pair_submission_capable = True

        def submit_pair(self, intents):
            return SimpleNamespace(
                status="pair_blocked",
                reason="missing_explicit_hyperliquid_order_approval",
                fills=(),
            )

    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(
            OrderIntent(market="ETH-USD", side="BUY", size=1.0, limit_price=3000.0),
            OrderIntent(market="BTC-USD", side="SELL", size=0.05, limit_price=60000.0),
        ),
        venue="hyperliquid",
    )

    fills = submit_paper_plan(plan, BlockedPairVenue())
    record = paper_trading_record(plan, fills=fills)

    assert len(fills) == 2
    assert all(
        fill.status == "paper_blocked_gate00g_execution_venue_denied"
        for fill in fills
    )
    assert record.plan_status == "blocked"
    assert "gate00g_execution_venue_denied" in record.plan_reason


def test_submit_paper_plan_denies_unfenced_sequential_adapter():
    class SequentialClient:
        pair_submission_capable = False

        def __init__(self):
            self.calls = []

        def place_order(self, intent, config=None):
            self.calls.append(intent)
            return FillReport(
                order_id=f"sequential-{len(self.calls)}",
                market=intent.market,
                side=intent.side,
                size=intent.size,
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status="paper_submitted",
            )

    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(
            OrderIntent(market="ETH-USD", side="BUY", size=1.0),
            OrderIntent(market="BTC-USD", side="SELL", size=0.05),
        ),
        venue="coinbase",
    )
    client = SequentialClient()
    venue = PaperVenueExecution("coinbase", client)

    fills = submit_paper_plan(plan, venue)

    assert client.calls == []
    assert len(fills) == 1
    assert fills[0].status == "paper_blocked_gate00g_order_authority_required"


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
    assert list(journal["plan_status"]) == ["blocked"]
    assert list(journal["blockers"]) == ["submit_orders_false"]
    assert "ETH-USD" in journal["intents_json"].iloc[0]
    assert "paper_blocked_submit_orders_false" in journal["fills_json"].iloc[0]


def test_paper_trading_record_merges_dashboard_snapshot(tmp_path):
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )

    record = paper_trading_record(
        plan,
        dashboard_snapshot={
            "dashboard_strategy": "Static (Spread)",
            "dashboard_zscore_norm": -0.53,
            "dashboard_candidate_count": 3,
            "dashboard_candidates": [
                {"dashboard_strategy": "Static (Spread)", "dashboard_sharpe": 2.71},
                {"dashboard_strategy": "Copula", "dashboard_sharpe": 1.82},
            ],
        },
    )

    entry_snapshot = execution_module._safe_json_dict(record.entry_snapshot_json)
    assert entry_snapshot["dashboard_strategy"] == "Static (Spread)"
    assert entry_snapshot["dashboard_zscore_norm"] == -0.53
    assert entry_snapshot["dashboard_candidate_count"] == 3
    assert len(entry_snapshot["dashboard_candidates"]) == 2


def test_refresh_current_paper_watch_positions_only_keeps_open_trades(tmp_path):
    journal_path = tmp_path / "paper_trading_journal.csv"
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )
    fill = FillReport(
        order_id="ok-1",
        market="ETH-USD",
        side="BUY",
        size=10.0,
        avg_price=1.0,
        fee=0.0,
        slippage_bps=0.0,
        status="paper_submitted",
    )
    append_paper_trading_record(paper_trading_record(plan, fills=[fill]), journal_path)

    watch_path = refresh_current_paper_watch_positions(journal_path, tmp_path / "watch.csv")

    import pandas as pd

    watch = pd.read_csv(watch_path)
    assert list(watch["pair"]) == ["ETH-BTC"]
    assert list(watch["lifecycle_status"]) == ["open"]


def test_append_paper_outcome_record_closes_open_trade_and_removes_from_watch(tmp_path):
    journal_path = tmp_path / "paper_trading_journal.csv"
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=7,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )
    fill = FillReport(
        order_id="ok-2",
        market="ETH-USD",
        side="BUY",
        size=10.0,
        avg_price=1.0,
        fee=0.0,
        slippage_bps=0.0,
        status="paper_submitted",
    )
    first = paper_trading_record(plan, fills=[fill])
    append_paper_trading_record(first, journal_path)

    _, close_record = append_paper_outcome_record(
        journal_path,
        trade_id=first.trade_id,
        realized_return=0.025,
    )
    watch_path = refresh_current_paper_watch_positions(journal_path, tmp_path / "watch.csv")

    import pandas as pd

    journal = pd.read_csv(journal_path)
    watch = pd.read_csv(watch_path)
    assert close_record.plan_status == "paper_completed"
    assert close_record.lifecycle_status == "closed"
    assert close_record.outcome_label == "win"
    assert journal["trade_id"].nunique() == 1
    assert watch.empty


def test_append_paper_outcome_record_uses_dashboard_exit_snapshot_lookup(tmp_path, monkeypatch):
    journal_path = tmp_path / "paper_trading_journal.csv"
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=7,
        status="paper_ready",
        reason="accepted",
        intents=(OrderIntent(market="ETH-USD", side="BUY", size=10.0),),
    )
    fill = FillReport(
        order_id="ok-3",
        market="ETH-USD",
        side="BUY",
        size=10.0,
        avg_price=1.0,
        fee=0.0,
        slippage_bps=0.0,
        status="paper_submitted",
    )
    first = paper_trading_record(plan, fills=[fill])
    append_paper_trading_record(first, journal_path)

    monkeypatch.setattr(
        execution_module,
        "_lookup_dashboard_trade_snapshot",
        lambda **kwargs: {
            "dashboard_strategy": "Static (Spread)",
            "dashboard_sharpe": 2.71,
            "dashboard_candidate_count": 2,
            "dashboard_candidates": [
                {"dashboard_strategy": "Static (Spread)", "dashboard_sharpe": 2.71},
                {"dashboard_strategy": "Copula", "dashboard_sharpe": 1.82},
            ],
        },
    )

    _, close_record = append_paper_outcome_record(
        journal_path,
        trade_id=first.trade_id,
        realized_return=-0.01,
    )

    exit_snapshot = execution_module._safe_json_dict(close_record.exit_snapshot_json)
    assert exit_snapshot["dashboard_strategy"] == "Static (Spread)"
    assert exit_snapshot["dashboard_candidate_count"] == 2
    assert len(exit_snapshot["dashboard_candidates"]) == 2


def test_refresh_live_paper_trade_monitor_builds_dashboard_first_watch_views(tmp_path):
    reports = tmp_path / "reports"
    active = reports / "active"
    active.mkdir(parents=True)
    journal_path = reports / "paper_trading_journal.csv"
    plan = SpreadOrderPlan(
        pair="BNB-USD/ETC-USD",
        strategy_id=7,
        status="paper_ready",
        reason="accepted",
        intents=(
            OrderIntent(market="BNB-USD", side="BUY", size=0.01),
            OrderIntent(market="ETC-USD", side="SELL", size=10.0),
        ),
        venue="dydx",
    )
    fills = [
        FillReport(
            order_id="ok-1",
            market="BNB-USD",
            side="BUY",
            size=0.01,
            avg_price=100.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        ),
        FillReport(
            order_id="ok-2",
            market="ETC-USD",
            side="SELL",
            size=10.0,
            avg_price=20.0,
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        ),
    ]
    record = paper_trading_record(plan, fills=fills)
    append_paper_trading_record(record, journal_path)

    pd.DataFrame(
        [
            {
                "decision_rank": 1,
                "pair": "BNB-USD/ETC-USD",
                "candidate_id": "wizard:bnb-usd-etc-usd:ou_(spread):daily",
                "venue": "dydx",
                "recommendation": "hold",
                "priority": "high",
                "reason": "still_shortlisted_monitor",
                "current_submit_state": "manual_dashboard_watch",
                "injective_execution_mode": "",
                "lifecycle_status": "open",
                "paper_gate_ready": False,
                "last_event_timestamp_utc": "2026-07-08T00:00:00+00:00",
                "evidence_path": "reports/active/paper_trade_decision_report.csv",
            }
        ]
    ).to_csv(active / "paper_trade_decision_report.csv", index=False)

    pd.DataFrame(
        [
            {
                "journal_layer": "pair_detail_capture",
                "pair": "BNB-USD/ETC-USD",
                "timeframe": "Daily",
                "strategy_label": "OU (Spread)",
                "detail_capture_timestamp_utc": "2026-07-08T01:00:00+00:00",
                "capture_status": "captured",
                "readiness_label": "decision_hold",
                "paper_candidate_status": "hold",
                "zscore_green_source": "rolling",
                "zscore_norm_value": 1.2,
                "zscore_roll_value": 2.1,
                "correlation_top": 0.82,
                "hurst_top": 0.31,
                "half_life_top": 18.0,
                "return_total_top": 28.4,
                "sharpe_top": 1.8,
                "max_drawdown_top": 9.4,
                "price_x": 654.2,
                "price_y": 24.1,
                "return_x_pct": 1.2,
                "return_y_pct": -0.6,
                "volume_x_top": 1120000.0,
                "volume_y_top": 510000.0,
                "dependency_x_over_y_top": 63.0,
                "dependency_y_over_x_top": 14.0,
                "coint_johansen_top": True,
                "coint_engle_granger_top": False,
                "hedge_ratio_top": 0.44,
                "lt_beta": 1.1,
                "changed_vs_last_capture": True,
                "delta_zscore_norm": 0.1,
                "delta_zscore_roll": 0.2,
                "delta_return_total": 1.5,
                "delta_sharpe": 0.1,
            },
            {
                "journal_layer": "pair_detail_capture",
                "pair": "BNB-USD/ETC-USD",
                "timeframe": "4 Hour",
                "strategy_label": "OU (Spread)",
                "detail_capture_timestamp_utc": "2026-07-08T01:00:00+00:00",
                "capture_status": "captured",
                "readiness_label": "decision_hold",
                "paper_candidate_status": "hold",
                "zscore_green_source": "normal",
                "zscore_norm_value": 1.7,
                "zscore_roll_value": 1.0,
                "correlation_top": 0.8,
                "hurst_top": 0.29,
                "half_life_top": 11.0,
                "return_total_top": 21.0,
                "sharpe_top": 1.4,
                "max_drawdown_top": 8.0,
                "price_x": 654.2,
                "price_y": 24.1,
                "return_x_pct": 1.2,
                "return_y_pct": -0.6,
                "volume_x_top": 1120000.0,
                "volume_y_top": 510000.0,
                "dependency_x_over_y_top": 63.0,
                "dependency_y_over_x_top": 14.0,
                "coint_johansen_top": True,
                "coint_engle_granger_top": False,
                "hedge_ratio_top": 0.44,
                "lt_beta": 1.1,
                "changed_vs_last_capture": False,
                "delta_zscore_norm": 0.0,
                "delta_zscore_roll": 0.0,
                "delta_return_total": 0.0,
                "delta_sharpe": 0.0,
            },
        ]
    ).to_csv(active / "wizard_research_journal.csv", index=False)

    paths = refresh_live_paper_trade_monitor(root=tmp_path)

    summary = pd.read_csv(paths["summary"])
    timeframe = pd.read_csv(paths["timeframe"])
    markdown = paths["markdown"].read_text(encoding="utf-8")

    assert list(summary["pair"]) == ["BNB-USD/ETC-USD"]
    row = summary.iloc[0]
    assert row["monitor_status"] == "active_on_dashboard"
    assert row["latest_dashboard_timeframe"] == "Daily"
    assert row["latest_dashboard_strategy"] == "OU (Spread)"
    assert bool(row["changed_since_last_capture"]) is True
    assert row["current_zscore_roll"] == 2.1
    assert set(timeframe["timeframe"]) == {"Daily", "4 Hour"}
    assert "BNB-USD/ETC-USD" in markdown


def test_hyperliquid_public_inventory_requires_authority_before_transport(
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []

    with pytest.raises(
        EffectAuthorityError,
        match="hyperliquid_public_effect_issuer_missing",
    ):
        refresh_hyperliquid_testnet_market_inventory(
            root=tmp_path,
            now=datetime(2026, 8, 22, 12, tzinfo=UTC),
            post_json_fetcher=lambda payload, **_: calls.append(payload),
        )

    assert calls == []


def test_hyperliquid_public_inventory_is_zero_credit_and_evidence_bound(
    tmp_path,
) -> None:
    target = "https://api.hyperliquid-testnet.xyz/info"
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"hyperliquid-public-test-authority",
        issuer_id="hyperliquid-public-test-supervisor",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    contracts = frozenset(
        {
            ExternalEffectCallContract(
                operation="HYPERLIQUID_TESTNET_META",
                method="POST",
                target=target,
                credit_units_per_request=0,
            ),
            ExternalEffectCallContract(
                operation="HYPERLIQUID_TESTNET_ALL_MIDS",
                method="POST",
                target=target,
                credit_units_per_request=0,
            ),
        }
    )
    calls: list[str] = []

    def fetch(payload, **_):
        request_type = payload["type"]
        calls.append(request_type)
        if request_type == "meta":
            return {
                "universe": [
                    {
                        "name": "BTC",
                        "szDecimals": 5,
                        "maxLeverage": 40,
                        "marginTableId": 0,
                        "isDelisted": False,
                    }
                ],
                "marginTables": [],
            }
        return {"BTC": "60000"}

    with external_effect_issuer_session(
        authority=authority,
        run_id="hyperliquid-public-run",
        intended_slot_id="hyperliquid-public-slot",
        source_fingerprint_sha256="a" * 64,
        runtime_fingerprint_sha256="b" * 64,
        configuration_fingerprint_sha256="c" * 64,
        provider_id="hyperliquid_public",
        account_scope_id="hyperliquid:testnet:public_research",
        allowed_targets=frozenset({target}),
        allowed_credential_keys=frozenset(),
        max_total_requests=2,
        max_total_credits=0,
        allowed_call_contracts=contracts,
    ):
        frame = refresh_hyperliquid_testnet_market_inventory(
            root=tmp_path,
            now=datetime(2026, 8, 22, 12, tzinfo=UTC),
            post_json_fetcher=fetch,
        )

    accounting = authority.run_accounting(
        run_id="hyperliquid-public-run",
        intended_slot_id="hyperliquid-public-slot",
    )
    manifest_path = (
        tmp_path
        / "reports"
        / "active"
        / "hyperliquid_testnet_market_inventory_evidence.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert calls == ["meta", "allMids"]
    assert frame.loc[0, "asset"] == "BTC"
    assert frame.loc[0, "mid_price"] == pytest.approx(60000.0)
    assert accounting["external_calls"] == 2
    assert accounting["external_credits_reserved"] == 0
    assert accounting["external_credits_consumed"] == 0
    assert accounting["accounting_complete"] is True
    assert manifest["external_requests"] == 2
    assert manifest["external_credits"] == 0
    assert manifest["blockers"] == []
    assert manifest["order_submission_included"] is False
    assert manifest["live_trading_authorized"] is False
    assert len(manifest["response_bindings"]) == 2
    for binding in manifest["response_bindings"]:
        path = tmp_path / binding["path"]
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest() == binding["sha256"]
