from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

import asyncio
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from dydx_v4_client import OrderFlags
from dydx_v4_client.indexer.rest.constants import OrderExecution, OrderType
from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
from dydx_v4_client.key_pair import KeyPair
from dydx_v4_client.network import TESTNET, make_testnet
from dydx_v4_client.node.builder import Builder
from dydx_v4_client.node.client import NodeClient
from dydx_v4_client.node.market import Market
from dydx_v4_client.wallet import Wallet
from v4_proto.dydxprotocol.clob.order_pb2 import Order

from quant_platform.execution import (
    DydxNetworkConfig,
    FillReport,
    OrderIntent,
    refresh_dydx_execution_compatibility_table,
)
from quant_platform.orchestration.corrective_order_authority import (
    DYDX_TESTNET_ADAPTER_ID,
    ConsumedOrderAuthorization,
    CorrectiveOrderAuthority,
    OrderEffectSpec,
    claim_effect_dispatch,
    exact_notional,
    require_consumed_authorization,
    require_order_authority,
)
from quant_platform.orchestration.effect_authority import EffectKind
from quant_platform.orchestration.venue_policy_registry import VenueLane

ROOT = Path(__file__).resolve().parents[2]
EXECUTION_ATTEMPT_LOG = ROOT / "reports" / "active" / "dydx_execution_market_attempts.csv"
EXECUTION_COMPATIBILITY_TABLE = ROOT / "reports" / "active" / "dydx_execution_market_compatibility.csv"


@dataclass(frozen=True)
class _AttemptSpec:
    route_label: str
    submit_mode: str
    config: DydxNetworkConfig


class DydxSdkOrderAdapter:
    """Authenticated dYdX v4 order adapter backed by the installed SDK."""

    _MAX_CLIENT_ID = 2**31 - 1
    _DEFAULT_MARKET_SLIPPAGE_PCT = 10.0
    _DEFAULT_CONFIRM_TIMEOUT_SECS = 20.0
    _DEFAULT_CONFIRM_POLL_SECS = 1.0

    exchange_submission_capable = True
    record_only = False
    gate00g_order_authority_enforced = True

    def __init__(
        self,
        *,
        order_authority: CorrectiveOrderAuthority | None = None,
    ) -> None:
        self._client_id_seed = int(time.time() * 1000) % self._MAX_CLIENT_ID
        self._order_authority = order_authority

    def place_order(self, intent: OrderIntent, config: DydxNetworkConfig) -> FillReport:
        authority = require_order_authority(self._order_authority)
        attempts = self._attempt_specs(intent, config)
        if not attempts:
            raise ValueError("gate00g_dydx_order_route_missing")
        first_spec = self._order_effect_spec(authority, intent, attempts[0])
        first_authorization = authority.consume(first_spec)
        if not config.wallet_address:
            raise ValueError("DYDX_TESTNET_WALLET_ADDRESS is required for authenticated order submission")
        if not config.private_key:
            raise ValueError("DYDX_TESTNET_PRIVATE_KEY is required for authenticated order submission")
        if not config.rest_indexer:
            raise ValueError("DYDX_TESTNET_REST_INDEXER is required for authenticated order submission")
        before_state = self._run(
            self._capture_market_state(
                intent.market,
                config,
                spec=first_spec,
                authorization=first_authorization,
            )
        )
        last_fill: FillReport | None = None
        for index, attempt in enumerate(attempts):
            if index == 0:
                spec = first_spec
                authorization = first_authorization
            else:
                spec = self._order_effect_spec(authority, intent, attempt)
                authorization = authority.consume(spec)
            response = self._run(
                self._submit_order(
                    intent,
                    attempt,
                    spec=spec,
                    authorization=authorization,
                )
            )
            order_id = self._response_txhash(response) or f"dydx-sdk-{self._client_id_seed}"
            confirmation = self._run(
                self._confirm_market_state(
                    intent,
                    attempt.config,
                    before_state,
                    spec=spec,
                    authorization=authorization,
                )
            )
            avg_price = confirmation.get("avg_price")
            if avg_price is None:
                avg_price = float(intent.limit_price or 0.0)
            status = "confirmed_on_exchange" if confirmation.get("confirmed") else "broadcast_accepted_unconfirmed"
            last_fill = FillReport(
                order_id=order_id,
                market=intent.market,
                side=intent.side,
                size=float(intent.size),
                avg_price=float(avg_price),
                fee=0.0,
                slippage_bps=0.0,
                status=status,
            )
            self._record_attempt(intent, attempt, last_fill)
            if confirmation.get("confirmed"):
                return last_fill
            if not intent.reduce_only:
                return last_fill
        return last_fill or FillReport(
            order_id="dydx-sdk-no-attempt",
            market=intent.market,
            side=intent.side,
            size=float(intent.size),
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="broadcast_accepted_unconfirmed",
        )

    async def _submit_order(
        self,
        intent: OrderIntent,
        attempt: _AttemptSpec,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ):
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        if attempt.submit_mode == "reduce_only_market":
            return await self._place_order(
                intent,
                attempt.config,
                spec=spec,
                authorization=authorization,
            )
        if intent.reduce_only:
            return await self._close_position(
                intent,
                attempt.config,
                spec=spec,
                authorization=authorization,
            )
        return await self._place_order(
            intent,
            attempt.config,
            spec=spec,
            authorization=authorization,
        )

    def _attempt_specs(self, intent: OrderIntent, config: DydxNetworkConfig) -> list[_AttemptSpec]:
        attempts = [
            _AttemptSpec(
                route_label=self._route_label(config, configured=True),
                submit_mode="close_position" if intent.reduce_only else "standard_order",
                config=config,
            )
        ]
        default_config = self._default_route_config(config)
        if not intent.reduce_only:
            if self._route_key(default_config) != self._route_key(config):
                attempts.append(
                    _AttemptSpec(
                        route_label=self._route_label(default_config, configured=False),
                        submit_mode="standard_order",
                        config=default_config,
                    )
                )
            return attempts
        if self._route_key(default_config) != self._route_key(config):
            attempts.append(
                _AttemptSpec(
                    route_label=self._route_label(default_config, configured=False),
                    submit_mode="close_position",
                    config=default_config,
                )
            )
        attempts.append(
            _AttemptSpec(
                route_label=self._route_label(config, configured=True),
                submit_mode="reduce_only_market",
                config=config,
            )
        )
        if self._route_key(default_config) != self._route_key(config):
            attempts.append(
                _AttemptSpec(
                    route_label=self._route_label(default_config, configured=False),
                    submit_mode="reduce_only_market",
                    config=default_config,
                )
            )
        return attempts

    @staticmethod
    def _order_effect_spec(
        authority: CorrectiveOrderAuthority,
        intent: OrderIntent,
        attempt: _AttemptSpec,
    ) -> OrderEffectSpec:
        operation = (
            "close_position"
            if attempt.submit_mode == "close_position"
            else "place_order"
        )
        config = attempt.config
        node_target = str(config.node_url or "sdk_builtin_testnet_node").strip()
        indexer_target = str(config.rest_indexer or "missing_indexer").strip()
        return authority.spec(
            effect_kind=EffectKind.ORDER_SUBMISSION,
            environment="testnet",
            adapter_id=DYDX_TESTNET_ADAPTER_ID,
            target=f"dydx-testnet:{node_target}|{indexer_target}",
            operation=operation,
            venue_id="dydx",
            product_lane_id=VenueLane.DYDX_PERP.value,
            account_scope_id=str(config.wallet_address or ""),
            instrument_id=str(intent.market).upper(),
            side=str(intent.side).lower(),
            size=intent.size,
            notional=exact_notional(intent.size, intent.limit_price),
            leverage=1,
            reduce_only=bool(intent.reduce_only),
            proposal_id=str(config.order_approval_id or ""),
            client_reference=attempt.route_label,
        )

    @staticmethod
    def _default_route_config(config: DydxNetworkConfig) -> DydxNetworkConfig:
        base = DydxNetworkConfig.paper_testnet()
        return replace(
            config,
            # Let the SDK's bundled TESTNET node config supply the best-available
            # default route instead of pinning a hostname that may vary by release.
            node_url=None,
            rest_indexer=config.rest_indexer or base.rest_indexer,
            websocket_indexer=config.websocket_indexer or base.websocket_indexer,
            faucet_url=config.faucet_url or base.faucet_url,
        )

    @staticmethod
    def _route_key(config: DydxNetworkConfig) -> str:
        return "|".join(
            [
                str(config.node_url or "").strip(),
                str(config.rest_indexer or "").strip(),
            ]
        )

    @staticmethod
    def _route_label(config: DydxNetworkConfig, configured: bool) -> str:
        prefix = "configured_route" if configured else "sdk_default_route"
        node = str(config.node_url or "").strip() or "sdk_builtin_testnet_node"
        return f"{prefix}:{node}"

    def _record_attempt(self, intent: OrderIntent, attempt: _AttemptSpec, fill: FillReport) -> None:
        row = pd.DataFrame(
            [
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "market": str(intent.market),
                    "side": str(intent.side),
                    "size": float(intent.size),
                    "reduce_only": bool(intent.reduce_only),
                    "route_label": attempt.route_label,
                    "submit_mode": attempt.submit_mode,
                    "node_url": str(attempt.config.node_url or ""),
                    "rest_indexer": str(attempt.config.rest_indexer or ""),
                    "order_id": str(fill.order_id),
                    "avg_price": float(fill.avg_price),
                    "status": str(fill.status),
                    "confirmed": str(fill.status) == "confirmed_on_exchange",
                }
            ]
        )
        EXECUTION_ATTEMPT_LOG.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(row, EXECUTION_ATTEMPT_LOG, mode="a", header=not EXECUTION_ATTEMPT_LOG.exists(), index=False)
        self._refresh_execution_compatibility_table()

    def _refresh_execution_compatibility_table(self) -> None:
        refresh_dydx_execution_compatibility_table(ROOT)

    async def _place_order(
        self,
        intent: OrderIntent,
        config: DydxNetworkConfig,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ):
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        node = await self._connect_node(
            config,
            spec=spec,
            authorization=authorization,
        )
        wallet = await self._build_wallet(
            node,
            config,
            spec=spec,
            authorization=authorization,
        )
        indexer = IndexerClient(config.rest_indexer)
        payload = await indexer.markets.get_perpetual_markets(intent.market)
        market_payload = payload.get("markets", {}).get(intent.market)
        if not isinstance(market_payload, dict):
            raise ValueError(f"missing dYdX market metadata for {intent.market}")
        market = Market(market_payload)
        client_id = self._next_client_id()
        order_id = market.order_id(wallet.address, 0, client_id, OrderFlags.SHORT_TERM)
        current_height = await node.latest_block_height()
        order_type = OrderType.LIMIT if intent.limit_price is not None else OrderType.MARKET
        order_execution = OrderExecution.DEFAULT if intent.limit_price is not None else OrderExecution.IOC
        order_side = Order.Side.SIDE_BUY if str(intent.side).upper() == "BUY" else Order.Side.SIDE_SELL
        oracle_price = float(market_payload.get("oraclePrice") or 0.0)
        reference_price = float(intent.limit_price or oracle_price or 0.0)
        if reference_price <= 0:
            raise ValueError(f"unable to resolve order price for {intent.market}")
        if intent.limit_price is None:
            if str(intent.side).upper() == "BUY":
                reference_price = oracle_price * (1 + self._DEFAULT_MARKET_SLIPPAGE_PCT / 100.0)
            else:
                reference_price = oracle_price * (1 - self._DEFAULT_MARKET_SLIPPAGE_PCT / 100.0)
        order = market.order(
            order_id=order_id,
            order_type=order_type,
            time_in_force=None,
            side=order_side,
            size=float(intent.size),
            price=reference_price,
            reduce_only=bool(intent.reduce_only),
            good_til_block=current_height + 20,
            execution=order_execution,
        )
        claim_effect_dispatch(
            authorization,
            owner=authority,
            spec=spec,
        )
        return await node.place_order(wallet, order)

    async def _close_position(
        self,
        intent: OrderIntent,
        config: DydxNetworkConfig,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ):
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        node = await self._connect_node(
            config,
            spec=spec,
            authorization=authorization,
        )
        wallet = await self._build_wallet(
            node,
            config,
            spec=spec,
            authorization=authorization,
        )
        indexer = IndexerClient(config.rest_indexer)
        payload = await indexer.markets.get_perpetual_markets(intent.market)
        market_payload = payload.get("markets", {}).get(intent.market)
        if not isinstance(market_payload, dict):
            raise ValueError(f"missing dYdX market metadata for {intent.market}")
        market = Market(market_payload)
        client_id = self._next_client_id()
        claim_effect_dispatch(
            authorization,
            owner=authority,
            spec=spec,
        )
        return await node.close_position(
            wallet,
            str(config.wallet_address),
            0,
            market,
            client_id,
            Decimal(str(intent.size)),
            slippage_pct=self._DEFAULT_MARKET_SLIPPAGE_PCT,
        )

    async def _capture_market_state(
        self,
        market: str,
        config: DydxNetworkConfig,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ) -> dict[str, Any]:
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        indexer = IndexerClient(str(config.rest_indexer))
        address = str(config.wallet_address)
        fills_payload = await self._safe_indexer_call(
            indexer.account.get_subaccount_fills,
            address,
            0,
            ticker=market,
            limit=20,
        )
        orders_payload = await self._safe_indexer_call(
            indexer.account.get_subaccount_orders,
            address,
            0,
            ticker=market,
            limit=20,
            return_latest_orders=True,
        )
        positions_payload = await self._safe_indexer_call(
            indexer.account.get_subaccount_perpetual_positions,
            address,
            0,
            limit=100,
        )
        fills = self._filter_market_rows(self._rows_from_payload(fills_payload, ("fills",)), market)
        orders = self._filter_market_rows(self._rows_from_payload(orders_payload, ("orders",)), market)
        positions = self._filter_market_rows(self._rows_from_payload(positions_payload, ("positions", "perpetualPositions")), market)
        return {
            "fill_ids": {self._row_identity(row) for row in fills},
            "order_ids": {self._row_identity(row) for row in orders},
            "position_size": self._position_size(positions),
            "avg_price": self._latest_price(fills) or self._latest_price(orders),
        }

    @staticmethod
    async def _safe_indexer_call(method, *args, **kwargs) -> dict[str, Any]:
        try:
            payload = await method(*args, **kwargs)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    async def _confirm_market_state(
        self,
        intent: OrderIntent,
        config: DydxNetworkConfig,
        before_state: dict[str, Any],
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ) -> dict[str, Any]:
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        deadline = time.time() + self._DEFAULT_CONFIRM_TIMEOUT_SECS
        while time.time() < deadline:
            current_state = await self._capture_market_state(
                intent.market,
                config,
                spec=spec,
                authorization=authorization,
            )
            if self._market_state_confirms(intent, before_state, current_state):
                return {"confirmed": True, "avg_price": current_state.get("avg_price")}
            await asyncio.sleep(self._DEFAULT_CONFIRM_POLL_SECS)
        return {"confirmed": False, "avg_price": before_state.get("avg_price")}

    def _market_state_confirms(
        self,
        intent: OrderIntent,
        before_state: dict[str, Any],
        current_state: dict[str, Any],
    ) -> bool:
        if current_state["fill_ids"] - before_state["fill_ids"]:
            return True
        if current_state["order_ids"] - before_state["order_ids"]:
            return True
        before_size = float(before_state.get("position_size") or 0.0)
        current_size = float(current_state.get("position_size") or 0.0)
        epsilon = max(abs(float(intent.size)) * 0.05, 1e-9)
        if bool(intent.reduce_only):
            return abs(current_size) < max(abs(before_size) - epsilon, 0.0)
        expected_direction = 1.0 if str(intent.side).upper() == "BUY" else -1.0
        return (current_size - before_size) * expected_direction > epsilon

    async def _build_wallet(
        self,
        node: NodeClient,
        config: DydxNetworkConfig,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ) -> Wallet:
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        secret = str(config.private_key or "").strip()
        if " " in secret:
            return await Wallet.from_mnemonic(node, secret, str(config.wallet_address))
        account = await node.get_account(str(config.wallet_address))
        return Wallet(
            KeyPair.from_hex(secret),
            account.account_number,
            account.sequence,
        )

    def _next_client_id(self) -> int:
        self._client_id_seed = (self._client_id_seed % self._MAX_CLIENT_ID) + 1
        return self._client_id_seed

    @staticmethod
    def _rows_from_payload(payload: object, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            return []
        for key in keys:
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return []

    @staticmethod
    def _filter_market_rows(rows: list[dict[str, Any]], market: str) -> list[dict[str, Any]]:
        target = str(market or "").upper()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            row_market = str(
                row.get("market")
                or row.get("ticker")
                or row.get("perpetualMarket")
                or row.get("clobPairId")
                or ""
            ).upper()
            if row_market == target:
                filtered.append(row)
        return filtered

    @staticmethod
    def _row_identity(row: dict[str, Any]) -> str:
        for key in ("id", "fillId", "orderId", "clientId", "createdAtHeight", "createdAt"):
            value = row.get(key)
            if value not in (None, ""):
                return str(value)
        return str(sorted(row.items()))

    @staticmethod
    def _position_size(rows: list[dict[str, Any]]) -> float:
        for row in rows:
            for key in ("size", "sumOpen", "netSize", "quantity"):
                value = row.get(key)
                if value not in (None, ""):
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        continue
        return 0.0

    @staticmethod
    def _latest_price(rows: list[dict[str, Any]]) -> float | None:
        latest: float | None = None
        for row in rows:
            for key in ("price", "fillPrice", "averagePrice"):
                value = row.get(key)
                if value in (None, ""):
                    continue
                try:
                    latest = float(value)
                    break
                except (TypeError, ValueError):
                    continue
        return latest

    @staticmethod
    def _response_txhash(response: object) -> str:
        tx_response = getattr(response, "tx_response", None)
        txhash = getattr(tx_response, "txhash", "") if tx_response is not None else ""
        return str(txhash or "").strip()

    async def _connect_node(
        self,
        config: DydxNetworkConfig,
        *,
        spec: OrderEffectSpec,
        authorization: ConsumedOrderAuthorization | None,
    ) -> NodeClient:
        authority = require_order_authority(self._order_authority)
        require_consumed_authorization(
            authorization,
            owner=authority,
            spec=spec,
        )
        node_url = str(config.node_url or "").strip()
        if node_url:
            network = make_testnet(
                node_url=node_url,
                rest_indexer=str(config.rest_indexer or TESTNET.rest_indexer),
                websocket_indexer=str(config.websocket_indexer or TESTNET.websocket_indexer),
            )
            return await NodeClient.connect(network.node)
        return await NodeClient.connect(TESTNET.node)

    @staticmethod
    def _run(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("DydxSdkOrderAdapter cannot run inside an active asyncio event loop")
