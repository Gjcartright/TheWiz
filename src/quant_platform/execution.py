from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
import asyncio
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Protocol

import pandas as pd


class ExecutionMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    LIVE = "live"


@dataclass(frozen=True)
class DydxNetworkConfig:
    mode: ExecutionMode = ExecutionMode.DRY_RUN
    node_url: str | None = None
    rest_indexer: str | None = None
    websocket_indexer: str | None = None
    faucet_url: str | None = None
    submit_orders: bool = False
    wallet_address: str | None = None
    private_key: str | None = None

    @classmethod
    def paper_testnet(cls) -> "DydxNetworkConfig":
        return cls(
            mode=ExecutionMode.PAPER,
            node_url="oegs-testnet.dydx.exchange:443",
            rest_indexer="https://indexer.v4testnet.dydx.exchange",
            websocket_indexer="wss://indexer.v4testnet.dydx.exchange/v4/ws",
            faucet_url="https://faucet.v4testnet.dydx.exchange",
            submit_orders=False,
        )

    @classmethod
    def paper_testnet_from_env(
        cls,
        wallet_address_env: str = "DYDX_TESTNET_WALLET_ADDRESS",
        private_key_env: str = "DYDX_TESTNET_PRIVATE_KEY",
        submit_orders_env: str = "DYDX_TESTNET_SUBMIT_ORDERS",
        node_url_env: str = "DYDX_TESTNET_NODE_URL",
        rest_indexer_env: str = "DYDX_TESTNET_REST_INDEXER",
        websocket_indexer_env: str = "DYDX_TESTNET_WEBSOCKET_INDEXER",
        faucet_url_env: str = "DYDX_TESTNET_FAUCET_URL",
    ) -> "DydxNetworkConfig":
        base = cls.paper_testnet()
        return cls(
            mode=base.mode,
            node_url=os.getenv(node_url_env, base.node_url or ""),
            rest_indexer=os.getenv(rest_indexer_env, base.rest_indexer or ""),
            websocket_indexer=os.getenv(websocket_indexer_env, base.websocket_indexer or ""),
            faucet_url=os.getenv(faucet_url_env, base.faucet_url or ""),
            submit_orders=os.getenv(submit_orders_env, "").lower() in {"1", "true", "yes"},
            wallet_address=os.getenv(wallet_address_env),
            private_key=os.getenv(private_key_env),
        )

    def paper_trading_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.mode != ExecutionMode.PAPER:
            blockers.append("mode_not_paper")
        if not self.submit_orders:
            blockers.append("submit_orders_false")
        if not self.wallet_address:
            blockers.append("missing_wallet_address")
        if not self.private_key:
            blockers.append("missing_private_key")
        if not dydx_v4_client_installed():
            blockers.append("missing_dydx_v4_client")
        return blockers


@dataclass(frozen=True)
class OrderIntent:
    market: str
    side: str
    size: float
    limit_price: float | None = None
    reduce_only: bool = False


@dataclass(frozen=True)
class FillReport:
    order_id: str
    market: str
    side: str
    size: float
    avg_price: float
    fee: float
    slippage_bps: float
    status: str


@dataclass(frozen=True)
class SpreadOrderPlan:
    pair: str
    strategy_id: int
    status: str
    reason: str
    intents: tuple[OrderIntent, ...] = ()


@dataclass(frozen=True)
class PaperTradingRecord:
    timestamp_utc: str
    pair: str
    strategy_id: int
    plan_status: str
    plan_reason: str
    blockers: str
    intents_json: str
    fills_json: str


class ExecutionVenue(Protocol):
    def market_data(self, market: str) -> dict: ...
    def place_order(self, intent: OrderIntent) -> FillReport: ...
    def positions(self) -> list[dict]: ...
    def funding(self, market: str) -> dict: ...


class DydxOrderClient(Protocol):
    def place_order(self, intent: OrderIntent, config: DydxNetworkConfig) -> FillReport: ...


class DydxMarketDataClient(Protocol):
    def market_data(self, market: str) -> dict: ...
    def funding(self, market: str) -> dict: ...


def dydx_v4_client_installed() -> bool:
    return importlib.util.find_spec("dydx_v4_client") is not None


def dydx_indexer_adapter_available() -> bool:
    return importlib.util.find_spec("dydx_v4_client.indexer.rest.indexer_client") is not None


def build_dydx_order_client_adapter(adapter_path: str | None = None) -> DydxOrderClient | None:
    adapter_path = adapter_path or os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER")
    if not adapter_path:
        return None
    if ":" not in adapter_path:
        raise ValueError("DYDX_TESTNET_ORDER_CLIENT_ADAPTER must be formatted as module:object")
    module_name, object_name = adapter_path.split(":", 1)
    module = importlib.import_module(module_name)
    adapter = getattr(module, object_name)
    if inspect.isclass(adapter) or not hasattr(adapter, "place_order"):
        adapter = adapter()
    if not hasattr(adapter, "place_order"):
        raise TypeError(f"dYdX order adapter {adapter_path} does not define place_order")
    return adapter


def validate_dydx_order_client_adapter(adapter_path: str | None = None) -> dict[str, object]:
    adapter_path = adapter_path or os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER")
    report: dict[str, object] = {
        "adapter_path": adapter_path or "",
        "configured": bool(adapter_path),
        "importable": False,
        "has_place_order": False,
        "signature_accepts_intent_config": False,
        "exchange_submission_capable": False,
        "record_only": False,
        "valid": False,
        "error": "",
    }
    if not adapter_path:
        report["error"] = "DYDX_TESTNET_ORDER_CLIENT_ADAPTER is not set"
        return report
    try:
        adapter = build_dydx_order_client_adapter(adapter_path)
        place_order = getattr(adapter, "place_order", None)
        report["importable"] = True
        report["has_place_order"] = callable(place_order)
        report["signature_accepts_intent_config"] = _place_order_accepts_intent_config(place_order)
        report["exchange_submission_capable"] = bool(getattr(adapter, "exchange_submission_capable", True))
        report["record_only"] = bool(getattr(adapter, "record_only", False))
        report["valid"] = bool(report["has_place_order"] and report["signature_accepts_intent_config"])
        if not report["signature_accepts_intent_config"]:
            report["error"] = "place_order must accept intent and config arguments"
    except Exception as exc:
        report["error"] = str(exc)
    return report


def _place_order_accepts_intent_config(place_order: object) -> bool:
    if not callable(place_order):
        return False
    try:
        signature = inspect.signature(place_order)
    except (TypeError, ValueError):
        return False
    required_positionals = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        and parameter.default is parameter.empty
    ]
    has_varargs = any(parameter.kind == parameter.VAR_POSITIONAL for parameter in signature.parameters.values())
    return has_varargs or len(required_positionals) <= 2 <= len(
        [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD, parameter.VAR_POSITIONAL)
        ]
    )


def dydx_readiness_report(
    config: DydxNetworkConfig | None = None,
    order_client_wired: bool = False,
    indexer_adapter_wired: bool | None = None,
) -> dict[str, object]:
    config = config or DydxNetworkConfig.paper_testnet_from_env()
    blockers = config.paper_trading_blockers()
    if indexer_adapter_wired is None:
        indexer_adapter_wired = bool(config.rest_indexer) and dydx_indexer_adapter_available()
    if not order_client_wired:
        blockers.append("missing_dydx_order_client_adapter")
    return {
        "mode": config.mode.value,
        "node_url": config.node_url,
        "rest_indexer": config.rest_indexer,
        "websocket_indexer": config.websocket_indexer,
        "faucet_url": config.faucet_url,
        "submit_orders": config.submit_orders,
        "wallet_address_present": bool(config.wallet_address),
        "private_key_present": bool(config.private_key),
        "dydx_v4_client_installed": dydx_v4_client_installed(),
        "dydx_indexer_adapter_wired": indexer_adapter_wired,
        "dydx_order_client_adapter_wired": order_client_wired,
        "ready_for_paper_submission": not blockers,
        "blockers": blockers,
    }


class DryRunDydxExecution:
    """Safe dYdX adapter placeholder that records intent without placing live orders."""

    def market_data(self, market: str) -> dict:
        return {"market": market, "status": "dry_run", "bid": None, "ask": None}

    def place_order(self, intent: OrderIntent) -> FillReport:
        return FillReport(
            order_id="dry-run",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="not_sent",
        )

    def positions(self) -> list[dict]:
        return []

    def funding(self, market: str) -> dict:
        return {"market": market, "status": "dry_run", "funding_rate": None}


class PaperDydxExecution:
    """dYdX testnet-backed paper trading adapter.

    This adapter keeps order submission disabled by default until wallet credentials
    and the official dYdX client wiring are explicitly configured.
    """

    def __init__(
        self,
        config: DydxNetworkConfig | None = None,
        client: DydxOrderClient | None = None,
        market_data_client: DydxMarketDataClient | None = None,
    ) -> None:
        self.config = config or DydxNetworkConfig.paper_testnet()
        self.client = client
        self.market_data_client = market_data_client
        if self.config.mode != ExecutionMode.PAPER:
            raise ValueError("PaperDydxExecution requires ExecutionMode.PAPER")

    def market_data(self, market: str) -> dict:
        if self.market_data_client is not None:
            return self.market_data_client.market_data(market)
        return {
            "market": market,
            "status": "paper",
            "network": "dydx_testnet",
            "rest_indexer": self.config.rest_indexer,
            "websocket_indexer": self.config.websocket_indexer,
        }

    def place_order(self, intent: OrderIntent) -> FillReport:
        if not self.config.submit_orders:
            return FillReport(
                order_id="paper-not-submitted",
                market=intent.market,
                side=intent.side,
                size=intent.size,
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status="paper_blocked_submit_orders_false",
            )
        if not self.config.wallet_address or not self.config.private_key:
            return FillReport(
                order_id="paper-missing-credentials",
                market=intent.market,
                side=intent.side,
                size=intent.size,
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status="paper_blocked_missing_credentials",
            )
        if self.client is None:
            return FillReport(
                order_id="paper-missing-client",
                market=intent.market,
                side=intent.side,
                size=intent.size,
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status="paper_blocked_missing_client",
            )
        return self.client.place_order(intent, self.config)

    def positions(self) -> list[dict]:
        return []

    def funding(self, market: str) -> dict:
        if self.market_data_client is not None:
            return self.market_data_client.funding(market)
        return {
            "market": market,
            "status": "paper",
            "network": "dydx_testnet",
            "funding_rate": None,
        }


class DydxV4IndexerAdapter:
    """Synchronous wrapper around the official dYdX v4 indexer client."""

    def __init__(self, config: DydxNetworkConfig, raw_client: object | None = None) -> None:
        self.config = config
        if raw_client is not None:
            self.client = raw_client
            return
        if not config.rest_indexer:
            raise ValueError("dYdX rest_indexer is required for indexer market data")
        from dydx_v4_client.indexer.rest.indexer_client import IndexerClient

        self.client = IndexerClient(config.rest_indexer)

    def market_data(self, market: str) -> dict:
        payload = self._run(self.client.markets.get_perpetual_markets(market))
        return {
            "market": market,
            "status": "paper",
            "network": "dydx_testnet",
            "rest_indexer": self.config.rest_indexer,
            "source": "dydx_v4_indexer",
            "payload": payload,
        }

    def funding(self, market: str) -> dict:
        payload = self._run(self.client.markets.get_perpetual_market_historical_funding(market, limit=1))
        return {
            "market": market,
            "status": "paper",
            "network": "dydx_testnet",
            "source": "dydx_v4_indexer",
            "payload": payload,
        }

    @staticmethod
    def _run(awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(awaitable)
        raise RuntimeError("DydxV4IndexerAdapter cannot run inside an active asyncio event loop")


def build_dydx_indexer_adapter(config: DydxNetworkConfig | None = None) -> DydxV4IndexerAdapter | None:
    config = config or DydxNetworkConfig.paper_testnet_from_env()
    if not dydx_indexer_adapter_available() or not config.rest_indexer:
        return None
    return DydxV4IndexerAdapter(config)


def build_execution_venue(
    config: DydxNetworkConfig | None = None,
    order_client: DydxOrderClient | None = None,
    market_data_client: DydxMarketDataClient | None = None,
) -> ExecutionVenue:
    config = config or DydxNetworkConfig()
    if config.mode == ExecutionMode.DRY_RUN:
        return DryRunDydxExecution()
    if config.mode == ExecutionMode.PAPER:
        return PaperDydxExecution(config, client=order_client, market_data_client=market_data_client)
    raise NotImplementedError("Live dYdX execution must be implemented behind explicit risk gates.")


def build_market_neutral_spread_intents(
    pair: str,
    side: str,
    notional_usd: float,
    hedge_ratio: float,
    beta: float = 1.0,
) -> tuple[OrderIntent, OrderIntent]:
    left, right = _split_pair(pair)
    signal = 1.0 if side.upper() == "LONG_SPREAD" else -1.0
    hedge = abs(float(hedge_ratio or 1.0))
    beta_abs = abs(float(beta or 1.0))
    gross_scale = 1.0 + hedge * beta_abs
    right_notional = notional_usd / gross_scale
    left_notional = notional_usd * hedge * beta_abs / gross_scale
    left_side = "SELL" if signal > 0 else "BUY"
    right_side = "BUY" if signal > 0 else "SELL"
    return (
        OrderIntent(market=_dydx_market(left), side=left_side, size=left_notional),
        OrderIntent(market=_dydx_market(right), side=right_side, size=right_notional),
    )


def build_research_gated_paper_plan(
    signal_row: dict,
    acceptance_report: pd.DataFrame,
    notional_usd: float,
) -> SpreadOrderPlan:
    strategy_id = int(signal_row["strategy_id"])
    pair = str(signal_row["pair"])
    matches = acceptance_report[acceptance_report["strategy_id"] == strategy_id]
    if matches.empty:
        return SpreadOrderPlan(pair=pair, strategy_id=strategy_id, status="blocked", reason="strategy_missing_acceptance")
    accepted = bool(matches["production_eligible"].iloc[0])
    if not accepted:
        reason = str(matches["acceptance_reason"].iloc[0])
        return SpreadOrderPlan(pair=pair, strategy_id=strategy_id, status="blocked", reason=f"research_rejected:{reason}")
    signal = float(signal_row.get("signal", 0.0))
    if signal == 0:
        return SpreadOrderPlan(pair=pair, strategy_id=strategy_id, status="blocked", reason="no_trade_signal")
    side = "LONG_SPREAD" if signal > 0 else "SHORT_SPREAD"
    intents = build_market_neutral_spread_intents(
        pair=pair,
        side=side,
        notional_usd=notional_usd,
        hedge_ratio=float(signal_row.get("hedge_ratio", 1.0)),
        beta=float(signal_row.get("beta", 1.0)),
    )
    return SpreadOrderPlan(pair=pair, strategy_id=strategy_id, status="paper_ready", reason="accepted", intents=intents)


def submit_paper_plan(plan: SpreadOrderPlan, venue: ExecutionVenue) -> list[FillReport]:
    if plan.status != "paper_ready":
        return []
    return [venue.place_order(intent) for intent in plan.intents]


def block_paper_plan_for_execution_config(plan: SpreadOrderPlan, blockers: list[str]) -> SpreadOrderPlan:
    if plan.status != "paper_ready" or not blockers:
        return plan
    return SpreadOrderPlan(
        pair=plan.pair,
        strategy_id=plan.strategy_id,
        status="blocked",
        reason=f"dydx_not_ready:{';'.join(blockers)}",
        intents=plan.intents,
    )


def paper_trading_record(
    plan: SpreadOrderPlan,
    fills: list[FillReport] | None = None,
    blockers: list[str] | None = None,
) -> PaperTradingRecord:
    return PaperTradingRecord(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        pair=plan.pair,
        strategy_id=plan.strategy_id,
        plan_status=plan.status,
        plan_reason=plan.reason,
        blockers=";".join(blockers or []),
        intents_json=json.dumps([asdict(intent) for intent in plan.intents], sort_keys=True),
        fills_json=json.dumps([asdict(fill) for fill in fills or []], sort_keys=True),
    )


def append_paper_trading_record(record: PaperTradingRecord, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(record)])
    row.to_csv(output, mode="a", header=not output.exists(), index=False)
    return output


def _split_pair(pair: str) -> tuple[str, str]:
    normalized = pair.replace("/", "-")
    parts = normalized.split("-")
    if len(parts) != 2:
        raise ValueError(f"pair must have two assets separated by '-' or '/': {pair}")
    return parts[0].upper(), parts[1].upper()


def _dydx_market(asset: str) -> str:
    if asset.endswith("-USD"):
        return asset
    return f"{asset}-USD"
