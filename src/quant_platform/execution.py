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
import math
import ssl
from typing import Protocol
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

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


CONFIRMED_PAPER_FILL_STATUSES = {"paper_submitted", "confirmed_on_exchange"}


@dataclass(frozen=True)
class SpreadOrderPlan:
    pair: str
    strategy_id: int
    status: str
    reason: str
    intents: tuple[OrderIntent, ...] = ()
    venue: str = "dydx"


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
    trade_id: str = ""
    venue: str = "dydx"
    lifecycle_status: str = "audit_only"
    opened_timestamp_utc: str = ""
    closed_timestamp_utc: str = ""
    entry_snapshot_json: str = "{}"
    exit_snapshot_json: str = "{}"
    realized_return: str = ""
    outcome_label: str = ""


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


class VenueOrderClient(Protocol):
    def place_order(self, intent: OrderIntent, config: object | None = None) -> FillReport: ...


_VENUE_PAPER_ADAPTER_ENV = {
    "dydx": "DYDX_TESTNET_ORDER_CLIENT_ADAPTER",
    "gmx": "GMX_TESTNET_ORDER_CLIENT_ADAPTER",
    "hyperliquid": "HYPERLIQUID_PAPER_ORDER_ADAPTER",
    "binance": "BINANCE_PAPER_ORDER_ADAPTER",
    "binanceus": "BINANCEUS_PAPER_ORDER_ADAPTER",
    "coinbase": "COINBASE_PAPER_ORDER_ADAPTER",
    "bybit": "BYBIT_PAPER_ORDER_ADAPTER",
    "binance_spot_testnet": "BINANCE_SPOT_TESTNET_ORDER_ADAPTER",
    "binance_usdm_testnet": "BINANCE_USDM_TESTNET_ORDER_ADAPTER",
}

_DEFAULT_DYDX_SDK_ADAPTER_PATH = "quant_platform.dydx_sdk_order_adapter:DydxSdkOrderAdapter"
_DEFAULT_VENUE_PAPER_ADAPTER_PATHS = {
    "hyperliquid": "quant_platform.hyperliquid_testnet:HyperliquidTestnetPairAdapter",
    "binance_spot_testnet": "quant_platform.binance_testnet:BinanceSpotTestnetOrderAdapter",
    "binance_usdm_testnet": "quant_platform.binance_testnet:BinanceUsdmTestnetOrderAdapter",
}
ROOT = Path(__file__).resolve().parents[2]
DYDX_EXECUTION_COMPATIBILITY_TABLE = ROOT / "reports" / "active" / "dydx_execution_market_compatibility.csv"
DYDX_EXECUTION_ATTEMPT_LOG = ROOT / "reports" / "active" / "dydx_execution_market_attempts.csv"
NON_ETH_ROUTE_SUBMIT_QUEUE = ROOT / "reports" / "active" / "non_eth_route_submit_queue.csv"
CURRENT_PAPER_WATCH_POSITIONS = ROOT / "reports" / "active" / "current_paper_watch_positions.csv"
PAPER_TRADE_RULEBOOK_PATH = ROOT / "reports" / "active" / "paper_trade_rulebook.md"
PAPER_TRADE_DECISION_REPORT = ROOT / "reports" / "active" / "paper_trade_decision_report.csv"
LIVE_PAPER_TRADE_MONITOR = ROOT / "reports" / "active" / "live_paper_trade_monitor.csv"
LIVE_PAPER_TRADE_TIMEFRAME_MONITOR = ROOT / "reports" / "active" / "live_paper_trade_timeframe_monitor.csv"
LIVE_PAPER_TRADE_MONITOR_MD = ROOT / "reports" / "active" / "live_paper_trade_monitor.md"
INJECTIVE_EXECUTION_COMPATIBILITY_TABLE = ROOT / "reports" / "active" / "injective_execution_market_compatibility.csv"
INJECTIVE_MIRROR_CANDIDATE_QUEUE = ROOT / "reports" / "active" / "injective_mirror_candidate_queue.csv"
GMX_TESTNET_MARKET_INVENTORY = ROOT / "reports" / "active" / "gmx_testnet_market_inventory.csv"
GMX_EXECUTION_COMPATIBILITY_TABLE = ROOT / "reports" / "active" / "gmx_execution_market_compatibility.csv"
GMX_TESTNET_CANDIDATE_SHORTLIST = ROOT / "reports" / "active" / "gmx_testnet_candidate_shortlist.csv"
HYPERLIQUID_TESTNET_MARKET_INVENTORY = ROOT / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
HYPERLIQUID_EXECUTION_COMPATIBILITY_TABLE = ROOT / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv"
HYPERLIQUID_TESTNET_CANDIDATE_SHORTLIST = ROOT / "reports" / "active" / "hyperliquid_testnet_candidate_shortlist.csv"
BROWSER_ACCOUNT_STATE_OVERRIDE = ROOT / "reports" / "active" / "browser_account_state_override.json"
INJECTIVE_MAINNET_SPOT_MARKETS_URL = "https://sentry.lcd.injective.network/injective/exchange/v1beta1/spot/markets"
INJECTIVE_MAINNET_DERIVATIVE_MARKETS_URL = "https://sentry.lcd.injective.network/injective/exchange/v1beta1/derivative/markets"
INJECTIVE_TESTNET_SPOT_MARKETS_URL = "https://testnet.sentry.lcd.injective.network/injective/exchange/v1beta1/spot/markets"
INJECTIVE_TESTNET_DERIVATIVE_MARKETS_URL = "https://testnet.sentry.lcd.injective.network/injective/exchange/v1beta1/derivative/markets"
GMX_ARBITRUM_SEPOLIA_CHAIN_ID = 421614
GMX_ARBITRUM_SEPOLIA_API_URL = "https://gmx-api-arbitrum-sepolia-yp6pp.ondigitalocean.app/v1"
GMX_ARBITRUM_SEPOLIA_TEST_API_URL = "https://arbitrum-sepolia-test.gmxapi.ai/v1"
HYPERLIQUID_TESTNET_INFO_URL = "https://api.hyperliquid-testnet.xyz/info"
HYPERLIQUID_MARGIN_TIERS_SOURCE_URL = (
    "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/margin-tiers"
)
HYPERLIQUID_TESTNET_ACCOUNT_ADDRESS_ENV = "HYPERLIQUID_TESTNET_ACCOUNT_ADDRESS"
HYPERLIQUID_TESTNET_SECRET_KEY_ENV = "HYPERLIQUID_TESTNET_SECRET_KEY"
HYPERLIQUID_TESTNET_SUBMIT_ORDERS_ENV = "HYPERLIQUID_TESTNET_SUBMIT_ORDERS"


def normalize_venue_name(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("_", " ").replace("-", " ").replace("/", " ")
    text = " ".join(text.split())
    text = text.replace("coinglass ", "").strip()

    parts = text.split()
    if parts == ["binance", "us"]:
        return "binanceus"
    if len(parts) > 1 and any(part in {"binance"} for part in parts) and any(part == "us" for part in parts):
        return "binanceus"

    tokens = [part for part in parts if part and part != "exchange"]
    if not tokens:
        return ""
    text = " ".join(tokens)
    if text in {"dyd", "dydx testnet", "dydx_testnet", "dydx-testnet", "dydx"}:
        return "dydx"
    if text in {"binanceus", "binance_us", "binance-us", "binance us"}:
        return "binanceus"
    if text in {"binance spot testnet", "binance_spot_testnet", "binance-spot-testnet"}:
        return "binance_spot_testnet"
    if text in {
        "binance usdm testnet",
        "binance usdm demo",
        "binance futures testnet",
        "binance_usdm_testnet",
        "binance-usdm-testnet",
    }:
        return "binance_usdm_testnet"
    if text in {"coinbase", "coinbase pro", "coinbase_pro", "coinbase-pro"}:
        return "coinbase"
    if text in {"bybit", "by_bit", "by-bit"}:
        return "bybit"
    if text == "coinglass":
        return ""
    return text


def venue_order_adapter_env(venue: str) -> str:
    venue = normalize_venue_name(venue)
    return _VENUE_PAPER_ADAPTER_ENV.get(venue, "")


def venue_has_paper_adapter(venue: str) -> bool:
    env = venue_order_adapter_env(venue)
    normalized = normalize_venue_name(venue)
    return bool(os.getenv(env, "").strip()) or normalized in _DEFAULT_VENUE_PAPER_ADAPTER_PATHS


def _parse_adapter_path(adapter_path: str) -> tuple[str, str]:
    if ":" not in adapter_path:
        raise ValueError("venue order adapter path must be formatted as module:object")
    module_name, object_name = adapter_path.split(":", 1)
    return module_name, object_name


def dydx_v4_client_installed() -> bool:
    return importlib.util.find_spec("dydx_v4_client") is not None


def dydx_indexer_adapter_available() -> bool:
    try:
        return importlib.util.find_spec("dydx_v4_client.indexer.rest.indexer_client") is not None
    except ModuleNotFoundError:
        return False


def build_dydx_order_client_adapter(adapter_path: str | None = None) -> DydxOrderClient | None:
    return build_venue_order_client_adapter("dydx", adapter_path=adapter_path)


def build_venue_order_client_adapter(venue: str, adapter_path: str | None = None) -> VenueOrderClient | None:
    venue = normalize_venue_name(venue)
    if adapter_path is None:
        if venue == "dydx":
            adapter_path = os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER")
            if not adapter_path and dydx_v4_client_installed():
                adapter_path = _DEFAULT_DYDX_SDK_ADAPTER_PATH
        else:
            env_name = venue_order_adapter_env(venue)
            adapter_path = os.getenv(env_name) if env_name else ""
            if not adapter_path:
                adapter_path = _DEFAULT_VENUE_PAPER_ADAPTER_PATHS.get(venue, "")
    if not adapter_path:
        return None
    env = venue_order_adapter_env(venue)
    if env and os.getenv(env) and adapter_path == os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER"):
        adapter_path = os.getenv(env, "")
    if not adapter_path:
        return None
    module_name, object_name = _parse_adapter_path(adapter_path)
    module = importlib.import_module(module_name)
    adapter = getattr(module, object_name)
    if inspect.isclass(adapter) or not hasattr(adapter, "place_order"):
        adapter = adapter()
    if not hasattr(adapter, "place_order"):
        raise TypeError(f"dYdX order adapter {adapter_path} does not define place_order")
    return adapter


def validate_dydx_order_client_adapter(adapter_path: str | None = None) -> dict[str, object]:
    return validate_venue_order_client_adapter("dydx", adapter_path=adapter_path)


def validate_venue_order_client_adapter(venue: str, adapter_path: str | None = None) -> dict[str, object]:
    venue = normalize_venue_name(venue)
    if adapter_path is None:
        if venue == "dydx":
            adapter_path = os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER")
            if not adapter_path and dydx_v4_client_installed():
                adapter_path = _DEFAULT_DYDX_SDK_ADAPTER_PATH
        else:
            env = venue_order_adapter_env(venue)
            adapter_path = os.getenv(env, "") if env else ""
            if not adapter_path:
                adapter_path = _DEFAULT_VENUE_PAPER_ADAPTER_PATHS.get(venue, "")
    env = venue_order_adapter_env(venue)
    if env and os.getenv(env) and adapter_path == os.getenv("DYDX_TESTNET_ORDER_CLIENT_ADAPTER"):
        adapter_path = os.getenv(env, "")
    report: dict[str, object] = {
        "venue": normalize_venue_name(venue),
        "adapter_path": adapter_path or "",
        "configured": bool(adapter_path),
        "importable": False,
        "has_place_order": False,
        "signature_accepts_intent_config": False,
        "has_submit_pair": False,
        "signature_accepts_pair_config": False,
        "pair_submission_capable": False,
        "exchange_submission_capable": False,
        "record_only": False,
        "valid": False,
        "error": "",
    }
    if not adapter_path:
        if env:
            report["error"] = f"{env} is not set"
        else:
            report["error"] = "DYDX_TESTNET_ORDER_CLIENT_ADAPTER is not set"
        return report
    try:
        adapter = build_venue_order_client_adapter(venue, adapter_path=adapter_path)
        place_order = getattr(adapter, "place_order", None)
        report["importable"] = True
        report["has_place_order"] = callable(place_order)
        report["signature_accepts_intent_config"] = _place_order_accepts_intent_config(place_order)
        submit_pair = getattr(adapter, "submit_pair", None)
        report["has_submit_pair"] = callable(submit_pair)
        report["signature_accepts_pair_config"] = _place_order_accepts_intent_config(submit_pair)
        report["pair_submission_capable"] = bool(
            getattr(adapter, "pair_submission_capable", False)
            and report["has_submit_pair"]
            and report["signature_accepts_pair_config"]
        )
        report["exchange_submission_capable"] = bool(getattr(adapter, "exchange_submission_capable", True))
        report["record_only"] = bool(getattr(adapter, "record_only", False))
        report["valid"] = bool(report["has_place_order"] and report["signature_accepts_intent_config"])
        if not report["signature_accepts_intent_config"]:
            report["error"] = "place_order must accept intent and config arguments"
    except Exception as exc:
        report["error"] = str(exc)
    return report


def _place_order_call(place_order, intent: OrderIntent, config: object | None = None):
    try:
        return place_order(intent, config)
    except TypeError:
        return place_order(intent)


def fill_report_confirmed(fill: FillReport) -> bool:
    return str(fill.status or "").strip().lower() in CONFIRMED_PAPER_FILL_STATUSES


def load_dydx_execution_attempts(root: Path = ROOT) -> pd.DataFrame:
    path = root / "reports" / "active" / "dydx_execution_market_attempts.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def _status_is_confirmed_for_paper_submit(status: object) -> bool:
    if status is None:
        return False
    normalized = str(status).strip().lower()
    return normalized in {"confirmed_on_exchange", "confirmed", "paper_submitted"}


def _injective_fetch_json(url: str) -> dict[str, object]:
    request = urlopen(url, context=ssl._create_unverified_context(), timeout=60)
    with request as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, dict):
        return data
    raise ValueError(f"unexpected Injective payload type for {url}")


def _injective_symbol_index(markets: list[dict[str, object]], field: str) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for row in markets:
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        base = ticker.split("/", 1)[0].strip().upper()
        if not base:
            continue
        index.setdefault(base, []).append(ticker)
    for symbol in index:
        index[symbol] = sorted(set(index[symbol]))
    return index


def _injective_market_indexes() -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], dict[str, list[str]], list[str]]:
    blockers: list[str] = []
    mainnet_spot: dict[str, list[str]] = {}
    mainnet_derivative: dict[str, list[str]] = {}
    testnet_spot: dict[str, list[str]] = {}
    testnet_derivative: dict[str, list[str]] = {}
    endpoints = [
        ("mainnet_spot", INJECTIVE_MAINNET_SPOT_MARKETS_URL),
        ("mainnet_derivative", INJECTIVE_MAINNET_DERIVATIVE_MARKETS_URL),
        ("testnet_spot", INJECTIVE_TESTNET_SPOT_MARKETS_URL),
        ("testnet_derivative", INJECTIVE_TESTNET_DERIVATIVE_MARKETS_URL),
    ]
    for label, url in endpoints:
        try:
            payload = _injective_fetch_json(url)
        except (URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"{label}_fetch_failed:{type(exc).__name__}")
            continue
        if label.endswith("derivative"):
            markets = [row.get("market", row) for row in payload.get("markets", []) if isinstance(row, dict)]
        else:
            markets = [row for row in payload.get("markets", []) if isinstance(row, dict)]
        index = _injective_symbol_index(markets, label)
        if label == "mainnet_spot":
            mainnet_spot = index
        elif label == "mainnet_derivative":
            mainnet_derivative = index
        elif label == "testnet_spot":
            testnet_spot = index
        else:
            testnet_derivative = index
    return mainnet_spot, mainnet_derivative, testnet_spot, testnet_derivative, blockers


def _injective_assets_for_pair(pair: str) -> tuple[str, str]:
    left, right = _split_pair(pair)
    return left.split("-", 1)[0].upper(), right.split("-", 1)[0].upper()


def refresh_injective_execution_compatibility_table(root: Path = ROOT) -> pd.DataFrame:
    route_candidates = _read_csv_or_empty(root / "reports" / "rl" / "base_rl_route_candidates.csv")
    if route_candidates.empty or "pair" not in route_candidates.columns:
        fallback_queue = _read_csv_or_empty(root / "reports" / "active" / "non_eth_route_submit_queue.csv")
        if not fallback_queue.empty and "pair" in fallback_queue.columns:
            fallback = fallback_queue.copy()
            if "candidate_id" not in fallback.columns:
                fallback["candidate_id"] = ""
            route_candidates = fallback[["pair", "candidate_id"]].copy()
    output = root / "reports" / "active" / "injective_execution_market_compatibility.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair",
        "candidate_id",
        "asset_x",
        "asset_y",
        "mainnet_spot_x",
        "mainnet_derivative_x",
        "testnet_spot_x",
        "testnet_derivative_x",
        "mainnet_spot_y",
        "mainnet_derivative_y",
        "testnet_spot_y",
        "testnet_derivative_y",
        "both_legs_found_mainnet",
        "both_legs_found_testnet",
        "both_legs_testnet_spot",
        "both_legs_testnet_derivative",
        "injective_execution_mode",
        "mirrorable_for_paper",
        "mirror_blocker",
        "checked_at_utc",
    ]
    if route_candidates.empty or "pair" not in route_candidates.columns:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame
    mainnet_spot, mainnet_derivative, testnet_spot, testnet_derivative, fetch_blockers = _injective_market_indexes()
    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for _, row in route_candidates.iterrows():
        pair = str(row.get("pair", "")).strip()
        if not pair:
            continue
        asset_x, asset_y = _injective_assets_for_pair(pair)
        main_spot_x = mainnet_spot.get(asset_x, [])
        main_deriv_x = mainnet_derivative.get(asset_x, [])
        test_spot_x = testnet_spot.get(asset_x, [])
        test_deriv_x = testnet_derivative.get(asset_x, [])
        main_spot_y = mainnet_spot.get(asset_y, [])
        main_deriv_y = mainnet_derivative.get(asset_y, [])
        test_spot_y = testnet_spot.get(asset_y, [])
        test_deriv_y = testnet_derivative.get(asset_y, [])
        both_main = bool((main_spot_x or main_deriv_x) and (main_spot_y or main_deriv_y))
        both_test = bool((test_spot_x or test_deriv_x) and (test_spot_y or test_deriv_y))
        both_test_spot = bool(test_spot_x and test_spot_y)
        both_test_derivative = bool(test_deriv_x and test_deriv_y)
        execution_mode = ""
        if both_test_spot:
            execution_mode = "injective_testnet_spot"
        mirrorable = bool(execution_mode) and not fetch_blockers
        blockers: list[str] = []
        if fetch_blockers:
            blockers.extend(fetch_blockers)
        if both_test_derivative and not both_test_spot:
            blockers.append("injective_spot_required")
        if not both_test_spot:
            if not test_spot_x:
                blockers.append(f"missing_testnet_spot:{asset_x}")
            if not test_spot_y:
                blockers.append(f"missing_testnet_spot:{asset_y}")
        rows.append(
            {
                "pair": pair,
                "candidate_id": str(row.get("candidate_id", "")),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "mainnet_spot_x": ";".join(main_spot_x),
                "mainnet_derivative_x": ";".join(main_deriv_x),
                "testnet_spot_x": ";".join(test_spot_x),
                "testnet_derivative_x": ";".join(test_deriv_x),
                "mainnet_spot_y": ";".join(main_spot_y),
                "mainnet_derivative_y": ";".join(main_deriv_y),
                "testnet_spot_y": ";".join(test_spot_y),
                "testnet_derivative_y": ";".join(test_deriv_y),
                "both_legs_found_mainnet": both_main,
                "both_legs_found_testnet": both_test,
                "both_legs_testnet_spot": both_test_spot,
                "both_legs_testnet_derivative": both_test_derivative,
                "injective_execution_mode": execution_mode,
                "mirrorable_for_paper": mirrorable,
                "mirror_blocker": ";".join(blockers),
                "checked_at_utc": checked_at,
            }
        )
    frame = pd.DataFrame(rows, columns=columns).sort_values(
        ["mirrorable_for_paper", "pair"],
        ascending=[False, True],
    )
    frame.to_csv(output, index=False)
    return frame


def refresh_injective_mirror_candidate_queue(root: Path = ROOT) -> pd.DataFrame:
    route_candidates = _read_csv_or_empty(root / "reports" / "rl" / "base_rl_route_candidates.csv")
    compatibility = _read_csv_or_empty(root / "reports" / "active" / "injective_execution_market_compatibility.csv")
    output = root / "reports" / "active" / "injective_mirror_candidate_queue.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair",
        "candidate_id",
        "strategy_mode",
        "timeframe",
        "paper_priority",
        "injective_mirrorable",
        "injective_execution_mode",
        "mirror_blocker",
        "next_action",
    ]
    if route_candidates.empty or compatibility.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame
    merged = route_candidates.merge(
        compatibility[["pair", "candidate_id", "injective_execution_mode", "mirrorable_for_paper", "mirror_blocker"]],
        on=["pair", "candidate_id"],
        how="left",
    )
    rows: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        injective_mirrorable = _boolish(row.get("mirrorable_for_paper", False))
        injective_execution_mode = str(row.get("injective_execution_mode", "") or "").strip()
        timeframe_options = str(row.get("available_timeframes", "")).split(",")
        timeframe = timeframe_options[0].strip() if timeframe_options else ""
        rows.append(
            {
                "pair": str(row.get("pair", "")),
                "candidate_id": str(row.get("candidate_id", "")),
                "strategy_mode": str(row.get("setup_identity", "")),
                "timeframe": timeframe,
                "paper_priority": str(row.get("decision_bucket", "")),
                "injective_mirrorable": injective_mirrorable,
                "injective_execution_mode": injective_execution_mode,
                "mirror_blocker": str(row.get("mirror_blocker", "")),
                "next_action": (
                    "paper_plus_injective_spot"
                    if injective_mirrorable
                    else "paper_only_until_injective_spot_support_exists"
                ),
            }
        )
    frame = pd.DataFrame(rows, columns=columns).sort_values(
        ["injective_mirrorable", "paper_priority", "pair"],
        ascending=[False, True, True],
    )
    frame.to_csv(output, index=False)
    return frame


def refresh_injective_spot_supported_pair_universe(root: Path = ROOT) -> pd.DataFrame:
    pair_universe = _read_csv_or_empty(root / "data" / "processed" / "pair_universe.csv")
    output = root / "reports" / "active" / "injective_spot_supported_pair_universe.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair",
        "asset_x",
        "asset_y",
        "combined_score",
        "acceptance_score",
        "decision_bucket",
        "decision_reason",
        "available_timeframes",
        "source_best_execution_venue",
        "mainnet_spot_x",
        "testnet_spot_x",
        "mainnet_spot_y",
        "testnet_spot_y",
        "both_legs_mainnet_spot",
        "both_legs_testnet_spot",
        "injective_execution_mode",
        "injective_supported_for_spot",
        "injective_support_blocker",
        "evidence_path",
        "checked_at_utc",
    ]
    if pair_universe.empty or "pair" not in pair_universe.columns:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    mainnet_spot, _mainnet_derivative, testnet_spot, _testnet_derivative, fetch_blockers = _injective_market_indexes()
    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    for _, row in pair_universe.iterrows():
        pair = str(row.get("pair", "") or "").strip().replace("_", "-")
        if not pair:
            continue
        try:
            asset_x, asset_y = _injective_assets_for_pair(pair.replace("-", "-"))
        except ValueError:
            continue
        main_spot_x = mainnet_spot.get(asset_x, [])
        test_spot_x = testnet_spot.get(asset_x, [])
        main_spot_y = mainnet_spot.get(asset_y, [])
        test_spot_y = testnet_spot.get(asset_y, [])
        both_mainnet_spot = bool(main_spot_x and main_spot_y)
        both_testnet_spot = bool(test_spot_x and test_spot_y)
        blockers: list[str] = list(fetch_blockers)
        if not test_spot_x:
            blockers.append(f"missing_testnet_spot:{asset_x}")
        if not test_spot_y:
            blockers.append(f"missing_testnet_spot:{asset_y}")
        rows.append(
            {
                "pair": pair.replace("-", "-", 0),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "combined_score": pd.to_numeric(pd.Series([row.get("combined_score")]), errors="coerce").fillna(0.0).iloc[0],
                "acceptance_score": pd.to_numeric(pd.Series([row.get("acceptance_score")]), errors="coerce").fillna(0.0).iloc[0],
                "decision_bucket": str(row.get("decision_bucket", "") or "").strip(),
                "decision_reason": str(row.get("decision_reason", "") or "").strip(),
                "available_timeframes": str(row.get("available_timeframes", "") or "").strip(),
                "source_best_execution_venue": str(row.get("best_execution_venue", "") or "").strip(),
                "mainnet_spot_x": ";".join(main_spot_x),
                "testnet_spot_x": ";".join(test_spot_x),
                "mainnet_spot_y": ";".join(main_spot_y),
                "testnet_spot_y": ";".join(test_spot_y),
                "both_legs_mainnet_spot": both_mainnet_spot,
                "both_legs_testnet_spot": both_testnet_spot,
                "injective_execution_mode": "injective_testnet_spot" if both_testnet_spot else "",
                "injective_supported_for_spot": both_testnet_spot and not fetch_blockers,
                "injective_support_blocker": "" if both_testnet_spot and not fetch_blockers else ";".join(blockers),
                "evidence_path": str(root / "data" / "processed" / "pair_universe.csv"),
                "checked_at_utc": checked_at,
            }
        )
    frame = pd.DataFrame(rows, columns=columns).sort_values(
        ["injective_supported_for_spot", "combined_score", "acceptance_score", "pair"],
        ascending=[False, False, False, True],
    )
    frame.to_csv(output, index=False)
    return frame


def refresh_injective_spot_first_candidate_shortlist(root: Path = ROOT, max_pairs: int = 10) -> pd.DataFrame:
    supported_universe = refresh_injective_spot_supported_pair_universe(root=root)
    output = root / "reports" / "active" / "injective_spot_first_candidate_shortlist.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "shortlist_rank",
        "pair",
        "asset_x",
        "asset_y",
        "injective_execution_mode",
        "injective_testnet_spot_x",
        "injective_testnet_spot_y",
        "source_decision_bucket",
        "source_decision_reason",
        "combined_score",
        "acceptance_score",
        "available_timeframes",
        "injective_lane_status",
        "next_action",
        "evidence_path",
    ]
    if supported_universe.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    shortlist = supported_universe[
        supported_universe.get("injective_supported_for_spot", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ].copy()
    if shortlist.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    bucket_rank = {"PROMOTE": 0, "WATCH": 1, "REJECT": 2}
    shortlist["_bucket_rank"] = shortlist.get("decision_bucket", pd.Series(dtype=object)).astype(str).map(bucket_rank).fillna(9)
    shortlist = shortlist.sort_values(
        ["_bucket_rank", "combined_score", "acceptance_score", "pair"],
        ascending=[True, False, False, True],
    ).head(max_pairs).copy()

    lane_status: list[str] = []
    next_actions: list[str] = []
    for _, row in shortlist.iterrows():
        bucket = str(row.get("decision_bucket", "") or "").strip().upper()
        if bucket == "PROMOTE":
            lane_status.append("injective_supported_and_promoted")
            next_actions.append("paper_plus_injective_spot")
        elif bucket == "WATCH":
            lane_status.append("injective_supported_research_watch")
            next_actions.append("refresh_research_then_paper_plus_injective_spot")
        else:
            lane_status.append("injective_supported_but_research_rejected")
            next_actions.append("repair_research_filters_before_any_submit")
    shortlist["injective_lane_status"] = lane_status
    shortlist["next_action"] = next_actions
    shortlist = shortlist.reset_index(drop=True)
    shortlist.insert(0, "shortlist_rank", shortlist.index + 1)
    shortlist["evidence_path"] = str(root / "reports" / "active" / "injective_spot_supported_pair_universe.csv")
    shortlist = shortlist.rename(
        columns={
            "testnet_spot_x": "injective_testnet_spot_x",
            "testnet_spot_y": "injective_testnet_spot_y",
            "decision_bucket": "source_decision_bucket",
            "decision_reason": "source_decision_reason",
        }
    )
    shortlist = shortlist.drop(columns=["_bucket_rank"], errors="ignore")
    frame = shortlist.reindex(columns=columns, fill_value="").copy()
    frame.to_csv(output, index=False)
    return frame


def load_injective_execution_compatibility(root: Path = ROOT) -> pd.DataFrame:
    path = root / "reports" / "active" / "injective_execution_market_compatibility.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def injective_execution_compatibility_snapshot(
    pairs: list[str] | tuple[str, ...] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    table = load_injective_execution_compatibility(root=root)
    normalized_pairs = [str(value or "").strip() for value in (pairs or []) if str(value or "").strip()]
    if table.empty or "pair" not in table.columns:
        return {
            "checked": False,
            "table_available": False,
            "route_pairs": normalized_pairs,
            "mirrorable_pairs": [],
            "spot_pairs": [],
            "derivative_pairs": [],
            "paper_only_pairs": normalized_pairs,
            "missing_pairs": normalized_pairs,
            "blocker": "injective_execution_compatibility_table_missing",
        }
    working = table.copy()
    working["pair"] = working["pair"].astype(str).str.strip()
    if normalized_pairs:
        working = working[working["pair"].isin(normalized_pairs)].copy()
    mirrorable_pairs: list[str] = []
    spot_pairs: list[str] = []
    derivative_pairs: list[str] = []
    paper_only_pairs: list[str] = []
    for _, row in working.iterrows():
        pair = str(row.get("pair", "")).strip()
        mirrorable = str(row.get("mirrorable_for_paper", "")).strip().lower() in {"true", "1", "yes"}
        execution_mode = str(row.get("injective_execution_mode", "") or "").strip()
        if mirrorable:
            mirrorable_pairs.append(pair)
            if execution_mode == "injective_testnet_spot":
                spot_pairs.append(pair)
            elif execution_mode == "injective_testnet_derivative":
                derivative_pairs.append(pair)
        else:
            paper_only_pairs.append(pair)
    observed_pairs = set(working["pair"].astype(str).tolist())
    missing_pairs = [pair for pair in normalized_pairs if pair not in observed_pairs]
    blocker = ""
    if normalized_pairs:
        if missing_pairs:
            blocker = "injective_pair_compatibility_missing"
        elif paper_only_pairs:
            blocker = "injective_testnet_spot_support_incomplete"
    else:
        blocker = "" if spot_pairs else "injective_no_spot_pairs"
    return {
        "checked": True,
        "table_available": True,
        "route_pairs": normalized_pairs,
        "mirrorable_pairs": sorted(set(mirrorable_pairs)),
        "spot_pairs": sorted(set(spot_pairs)),
        "derivative_pairs": sorted(set(derivative_pairs)),
        "paper_only_pairs": sorted(set(paper_only_pairs)),
        "missing_pairs": missing_pairs,
        "blocker": blocker,
    }


def _gmx_fetch_json(url: str) -> list[dict[str, object]]:
    request = urlopen(url, context=ssl._create_unverified_context(), timeout=60)
    with request as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    raise ValueError(f"unexpected GMX payload type for {url}")


def _gmx_token_index(tokens: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for token in tokens:
        for key in ("address", "wrappedAddress"):
            address = str(token.get(key, "") or "").lower().strip()
            if address:
                index[address] = token
    return index


def _gmx_symbol_from_market(row: dict[str, object], token_index: dict[str, dict[str, object]]) -> str:
    index_address = str(row.get("indexTokenAddress", "") or "").lower().strip()
    token = token_index.get(index_address, {})
    symbol = str(token.get("baseSymbol", "") or token.get("symbol", "") or "").upper().strip()
    if symbol == "WETH":
        symbol = "ETH"
    if symbol:
        return symbol
    market_symbol = str(row.get("symbol", "") or row.get("name", "") or "").strip()
    if "/" in market_symbol:
        return market_symbol.split("/", 1)[0].strip().upper()
    return market_symbol.split(" ", 1)[0].strip().upper()


def _gmx_scaled_usd(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value) / 1e30
    except (TypeError, ValueError, OverflowError):
        return None


def _gmx_market_payloads(
    base_url: str = GMX_ARBITRUM_SEPOLIA_API_URL,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[str]]:
    blockers: list[str] = []

    def _fetch(path: str) -> list[dict[str, object]]:
        try:
            return _gmx_fetch_json(f"{base_url.rstrip('/')}/{path.lstrip('/')}")
        except (URLError, OSError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"{path.strip('/').replace('/', '_')}_fetch_failed:{type(exc).__name__}")
            return []

    markets = _fetch("/markets")
    market_info = _fetch("/markets/info")
    tickers = _fetch("/markets/tickers")
    tokens = _fetch("/tokens")
    return markets, market_info, tickers, tokens, blockers


def _gmx_inventory_frame(
    base_url: str = GMX_ARBITRUM_SEPOLIA_API_URL,
) -> tuple[pd.DataFrame, list[str]]:
    markets, market_info, tickers, tokens, blockers = _gmx_market_payloads(base_url)
    token_index = _gmx_token_index(tokens)
    info_by_market = {str(row.get("marketTokenAddress", "") or "").lower(): row for row in market_info}
    ticker_by_market = {str(row.get("marketTokenAddress", "") or "").lower(): row for row in tickers}
    rows: list[dict[str, object]] = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for row in markets:
        market_token = str(row.get("marketTokenAddress", "") or "").lower()
        info = info_by_market.get(market_token, {})
        ticker = ticker_by_market.get(market_token, {})
        symbol = str(row.get("symbol", "") or info.get("name", "") or "").strip()
        index_symbol = _gmx_symbol_from_market(row, token_index)
        is_disabled = _boolish(info.get("isDisabled", False))
        is_spot_only = _boolish(row.get("isSpotOnly", info.get("isSpotOnly", False)))
        is_listed = _boolish(row.get("isListed", True))
        tradable_perp = bool(index_symbol and is_listed and not is_disabled and not is_spot_only)
        rows.append(
            {
                "chain_id": GMX_ARBITRUM_SEPOLIA_CHAIN_ID,
                "api_url": base_url,
                "market_symbol": symbol,
                "asset": index_symbol,
                "market_token_address": row.get("marketTokenAddress", ""),
                "index_token_address": row.get("indexTokenAddress", ""),
                "long_token_address": row.get("longTokenAddress", ""),
                "short_token_address": row.get("shortTokenAddress", ""),
                "is_listed": is_listed,
                "is_disabled": is_disabled,
                "is_spot_only": is_spot_only,
                "tradable_perp": tradable_perp,
                "mark_price_usd": _gmx_scaled_usd(ticker.get("markPrice")),
                "min_position_size_usd": _gmx_scaled_usd(row.get("minPositionSizeUsd") or info.get("minPositionSizeUsd")),
                "min_collateral_usd": _gmx_scaled_usd(row.get("minCollateralUsd") or info.get("minCollateralUsd")),
                "open_interest_long_usd": _gmx_scaled_usd(info.get("openInterestLongUsd") or ticker.get("openInterestLongUsd")),
                "open_interest_short_usd": _gmx_scaled_usd(info.get("openInterestShortUsd") or ticker.get("openInterestShortUsd")),
                "checked_at_utc": checked_at,
            }
        )
    columns = [
        "chain_id",
        "api_url",
        "market_symbol",
        "asset",
        "market_token_address",
        "index_token_address",
        "long_token_address",
        "short_token_address",
        "is_listed",
        "is_disabled",
        "is_spot_only",
        "tradable_perp",
        "mark_price_usd",
        "min_position_size_usd",
        "min_collateral_usd",
        "open_interest_long_usd",
        "open_interest_short_usd",
        "checked_at_utc",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["tradable_perp", "asset", "market_symbol"], ascending=[False, True, True]).reset_index(drop=True)
    return frame, blockers


def refresh_gmx_testnet_market_inventory(
    root: Path = ROOT,
    base_url: str = GMX_ARBITRUM_SEPOLIA_API_URL,
) -> pd.DataFrame:
    output = root / "reports" / "active" / "gmx_testnet_market_inventory.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, blockers = _gmx_inventory_frame(base_url)
    if blockers and frame.empty:
        frame = pd.DataFrame(
            [
                {
                    "chain_id": GMX_ARBITRUM_SEPOLIA_CHAIN_ID,
                    "api_url": base_url,
                    "market_symbol": "",
                    "asset": "",
                    "market_token_address": "",
                    "index_token_address": "",
                    "long_token_address": "",
                    "short_token_address": "",
                    "is_listed": False,
                    "is_disabled": True,
                    "is_spot_only": False,
                    "tradable_perp": False,
                    "mark_price_usd": None,
                    "min_position_size_usd": None,
                    "min_collateral_usd": None,
                    "open_interest_long_usd": None,
                    "open_interest_short_usd": None,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fetch_blocker": ";".join(blockers),
                }
            ]
        )
    elif blockers:
        frame = frame.copy()
        frame["fetch_blocker"] = ";".join(blockers)
    else:
        frame = frame.copy()
        frame["fetch_blocker"] = ""
    frame.to_csv(output, index=False)
    return frame


def _gmx_market_indexes(root: Path = ROOT) -> tuple[dict[str, list[str]], list[str]]:
    inventory = _read_csv_or_empty(root / "reports" / "active" / "gmx_testnet_market_inventory.csv")
    if inventory.empty or "asset" not in inventory.columns:
        inventory = refresh_gmx_testnet_market_inventory(root=root)
    blockers: list[str] = []
    if not inventory.empty and "fetch_blocker" in inventory.columns:
        blockers = sorted(
            {
                item.strip()
                for value in inventory.get("fetch_blocker", pd.Series(dtype=object)).tolist()
                for item in str(value or "").split(";")
                if item.strip() and item.strip().lower() != "nan"
            }
        )
    market_index: dict[str, list[str]] = {}
    if inventory.empty:
        return market_index, blockers or ["gmx_testnet_market_inventory_empty"]
    working = inventory.copy()
    tradable = working.get("tradable_perp", pd.Series(dtype=bool)).fillna(False).astype(bool)
    for _, row in working.loc[tradable].iterrows():
        asset = str(row.get("asset", "") or "").upper().strip()
        market = str(row.get("market_symbol", "") or "").strip()
        if asset and market:
            market_index.setdefault(asset, []).append(market)
    for asset in market_index:
        market_index[asset] = sorted(set(market_index[asset]))
    return market_index, blockers


def _gmx_assets_for_pair(pair: str) -> tuple[str, str]:
    left, right = _split_pair(pair)
    return left.split("-", 1)[0].upper(), right.split("-", 1)[0].upper()


def _gmx_pair_key(pair: object) -> str:
    try:
        asset_x, asset_y = _gmx_assets_for_pair(str(pair or ""))
    except ValueError:
        return str(pair or "").strip().upper().replace("/", "-")
    return f"{asset_x}-{asset_y}"


def _gmx_candidate_source_frame(root: Path) -> pd.DataFrame:
    candidates = _read_csv_or_empty(root / "reports" / "rl" / "base_rl_route_candidates.csv")
    if not candidates.empty and "pair" in candidates.columns:
        return candidates.copy()
    for path in [
        root / "reports" / "active" / "fresh_live_dashboard_pair_queue.csv",
        root / "reports" / "active" / "non_eth_route_submit_queue.csv",
        root / "data" / "processed" / "pair_universe.csv",
    ]:
        frame = _read_csv_or_empty(path)
        if not frame.empty and "pair" in frame.columns:
            working = frame.copy()
            if "candidate_id" not in working.columns:
                working["candidate_id"] = ""
            return working
    return pd.DataFrame()


def refresh_gmx_execution_compatibility_table(root: Path = ROOT) -> pd.DataFrame:
    candidates = _gmx_candidate_source_frame(root)
    output = root / "reports" / "active" / "gmx_execution_market_compatibility.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair",
        "candidate_id",
        "asset_x",
        "asset_y",
        "testnet_perp_x",
        "testnet_perp_y",
        "both_legs_testnet_perp",
        "gmx_execution_mode",
        "mirrorable_for_paper",
        "mirror_blocker",
        "checked_at_utc",
    ]
    if candidates.empty or "pair" not in candidates.columns:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    market_index, fetch_blockers = _gmx_market_indexes(root=root)
    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in candidates.iterrows():
        pair = str(row.get("pair", "") or "").strip().replace("_", "-")
        if not pair:
            continue
        candidate_id = str(row.get("candidate_id", "") or "")
        key = (pair, candidate_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            asset_x, asset_y = _gmx_assets_for_pair(pair)
        except ValueError:
            continue
        testnet_x = market_index.get(asset_x, [])
        testnet_y = market_index.get(asset_y, [])
        both_perp = bool(testnet_x and testnet_y)
        blockers: list[str] = list(fetch_blockers)
        if not testnet_x:
            blockers.append(f"missing_gmx_testnet_perp:{asset_x}")
        if not testnet_y:
            blockers.append(f"missing_gmx_testnet_perp:{asset_y}")
        execution_mode = "gmx_arbitrum_sepolia_perp" if both_perp and not fetch_blockers else ""
        rows.append(
            {
                "pair": pair,
                "candidate_id": candidate_id,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "testnet_perp_x": ";".join(testnet_x),
                "testnet_perp_y": ";".join(testnet_y),
                "both_legs_testnet_perp": both_perp,
                "gmx_execution_mode": execution_mode,
                "mirrorable_for_paper": bool(execution_mode),
                "mirror_blocker": ";".join(blockers),
                "checked_at_utc": checked_at,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["mirrorable_for_paper", "pair"], ascending=[False, True]).reset_index(drop=True)
    frame.to_csv(output, index=False)
    return frame


def refresh_gmx_testnet_candidate_shortlist(root: Path = ROOT, max_pairs: int = 10) -> pd.DataFrame:
    compatibility = refresh_gmx_execution_compatibility_table(root=root)
    universe = _read_csv_or_empty(root / "data" / "processed" / "pair_universe.csv")
    output = root / "reports" / "active" / "gmx_testnet_candidate_shortlist.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "shortlist_rank",
        "pair",
        "candidate_id",
        "asset_x",
        "asset_y",
        "gmx_execution_mode",
        "gmx_testnet_perp_x",
        "gmx_testnet_perp_y",
        "source_decision_bucket",
        "source_decision_reason",
        "combined_score",
        "acceptance_score",
        "available_timeframes",
        "gmx_lane_status",
        "next_action",
        "evidence_path",
    ]
    if compatibility.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    supported = compatibility[
        compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ].copy()
    if supported.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    if not universe.empty and "pair" in universe.columns:
        keep_cols = [
            col
            for col in ["pair", "decision_bucket", "decision_reason", "combined_score", "acceptance_score", "available_timeframes"]
            if col in universe.columns
        ]
        universe_meta = universe[keep_cols].copy()
        universe_meta["_gmx_pair_key"] = universe_meta["pair"].map(_gmx_pair_key)
        universe_meta = universe_meta.drop(columns=["pair"]).drop_duplicates("_gmx_pair_key")
        supported["_gmx_pair_key"] = supported["pair"].map(_gmx_pair_key)
        supported = supported.merge(universe_meta, on="_gmx_pair_key", how="left")

    bucket_rank = {"PROMOTE": 0, "WATCH": 1, "FETCH_MORE_DATA": 2, "REJECT": 3}
    supported["_bucket_rank"] = supported.get("decision_bucket", pd.Series(dtype=object)).astype(str).map(bucket_rank).fillna(9)
    supported["combined_score"] = pd.to_numeric(supported.get("combined_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    supported["acceptance_score"] = pd.to_numeric(supported.get("acceptance_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    supported = supported.sort_values(
        ["_bucket_rank", "combined_score", "acceptance_score", "pair"],
        ascending=[True, False, False, True],
    ).head(max_pairs).copy()

    lane_status: list[str] = []
    next_actions: list[str] = []
    for _, row in supported.iterrows():
        bucket = str(row.get("decision_bucket", "") or "").strip().upper()
        if bucket == "PROMOTE":
            lane_status.append("gmx_supported_and_promoted")
            next_actions.append("paper_journal_plus_gmx_testnet_order_adapter_probe")
        elif bucket == "WATCH":
            lane_status.append("gmx_supported_research_watch")
            next_actions.append("refresh_research_then_gmx_adapter_probe")
        else:
            lane_status.append("gmx_supported_but_research_not_promoted")
            next_actions.append("repair_research_filters_before_gmx_submit")
    supported["gmx_lane_status"] = lane_status
    supported["next_action"] = next_actions
    supported = supported.reset_index(drop=True)
    supported.insert(0, "shortlist_rank", supported.index + 1)
    supported["evidence_path"] = str(root / "reports" / "active" / "gmx_execution_market_compatibility.csv")
    supported = supported.rename(
        columns={
            "testnet_perp_x": "gmx_testnet_perp_x",
            "testnet_perp_y": "gmx_testnet_perp_y",
            "decision_bucket": "source_decision_bucket",
            "decision_reason": "source_decision_reason",
        }
    )
    supported = supported.drop(columns=["_bucket_rank", "_gmx_pair_key"], errors="ignore")
    frame = supported.reindex(columns=columns, fill_value="")
    frame.to_csv(output, index=False)
    return frame


def load_gmx_execution_compatibility(root: Path = ROOT) -> pd.DataFrame:
    path = root / "reports" / "active" / "gmx_execution_market_compatibility.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def gmx_execution_compatibility_snapshot(
    pairs: list[str] | tuple[str, ...] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    table = load_gmx_execution_compatibility(root=root)
    normalized_pairs = [str(value or "").strip() for value in (pairs or []) if str(value or "").strip()]
    if table.empty or "pair" not in table.columns:
        return {
            "checked": False,
            "table_available": False,
            "route_pairs": normalized_pairs,
            "mirrorable_pairs": [],
            "paper_only_pairs": normalized_pairs,
            "missing_pairs": normalized_pairs,
            "blocker": "gmx_execution_compatibility_table_missing",
        }
    working = table.copy()
    working["pair"] = working["pair"].astype(str).str.strip()
    if normalized_pairs:
        working = working[working["pair"].isin(normalized_pairs)].copy()
    mirrorable_pairs: list[str] = []
    paper_only_pairs: list[str] = []
    for _, row in working.iterrows():
        pair = str(row.get("pair", "")).strip()
        mirrorable = _boolish(row.get("mirrorable_for_paper", False))
        if mirrorable:
            mirrorable_pairs.append(pair)
        else:
            paper_only_pairs.append(pair)
    observed_pairs = set(working["pair"].astype(str).tolist())
    missing_pairs = [pair for pair in normalized_pairs if pair not in observed_pairs]
    if normalized_pairs:
        if missing_pairs:
            blocker = "gmx_pair_compatibility_missing"
        elif paper_only_pairs:
            blocker = "gmx_testnet_perp_support_incomplete"
        else:
            blocker = ""
    else:
        blocker = "" if mirrorable_pairs else "gmx_no_supported_testnet_pairs"
    return {
        "checked": True,
        "table_available": True,
        "route_pairs": normalized_pairs,
        "mirrorable_pairs": sorted(set(mirrorable_pairs)),
        "paper_only_pairs": sorted(set(paper_only_pairs)),
        "missing_pairs": missing_pairs,
        "blocker": blocker,
    }


def _hyperliquid_post_json(
    payload: dict[str, object],
    *,
    info_url: str = HYPERLIQUID_TESTNET_INFO_URL,
    timeout: int = 20,
) -> object:
    request = Request(
        info_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    context = ssl.create_default_context()
    with urlopen(request, timeout=timeout, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def _hyperliquid_safe_float(value: object) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _hyperliquid_inventory_frame(
    *,
    info_url: str = HYPERLIQUID_TESTNET_INFO_URL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    blockers: list[str] = []
    try:
        meta = _hyperliquid_post_json({"type": "meta"}, info_url=info_url)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        meta = {}
        blockers.append(f"hyperliquid_meta_fetch_failed:{type(exc).__name__}")
    try:
        mids = _hyperliquid_post_json({"type": "allMids"}, info_url=info_url)
    except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        mids = {}
        blockers.append(f"hyperliquid_all_mids_fetch_failed:{type(exc).__name__}")

    rows: list[dict[str, object]] = []
    checked_at = datetime.now(timezone.utc).isoformat()
    universe = meta.get("universe", []) if isinstance(meta, dict) else []
    mids_map = mids if isinstance(mids, dict) else {}
    for asset_index, row in enumerate(universe):
        if not isinstance(row, dict):
            continue
        asset = str(row.get("name", "") or "").strip()
        is_delisted = _boolish(row.get("isDelisted", False))
        tradable_perp = bool(asset and not is_delisted)
        rows.append(
            {
                "api_url": info_url,
                "market_type": "perp",
                "asset": asset,
                "asset_index": asset_index,
                "universe_name": asset,
                "sz_decimals": row.get("szDecimals", ""),
                "max_leverage": row.get("maxLeverage", ""),
                "margin_table_id": row.get("marginTableId", ""),
                "only_isolated": _boolish(row.get("onlyIsolated", False)),
                "margin_mode": row.get("marginMode", ""),
                "is_delisted": is_delisted,
                "tradable_perp": tradable_perp,
                "mid_price": _hyperliquid_safe_float(mids_map.get(asset)),
                "checked_at_utc": checked_at,
            }
        )
    columns = [
        "api_url",
        "market_type",
        "asset",
        "asset_index",
        "universe_name",
        "sz_decimals",
        "max_leverage",
        "margin_table_id",
        "only_isolated",
        "margin_mode",
        "is_delisted",
        "tradable_perp",
        "mid_price",
        "checked_at_utc",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["tradable_perp", "asset_index"], ascending=[False, True]).reset_index(drop=True)
    margin_tiers = _hyperliquid_margin_tier_frame(
        meta,
        universe=universe,
        info_url=info_url,
        checked_at=checked_at,
    )
    return frame, margin_tiers, blockers


def _hyperliquid_margin_tier_frame(
    meta: object,
    *,
    universe: object | None = None,
    info_url: str = HYPERLIQUID_TESTNET_INFO_URL,
    checked_at: str | None = None,
) -> pd.DataFrame:
    """Normalize the maintenance-margin tables returned by Hyperliquid meta."""

    columns = [
        "api_url",
        "margin_table_id",
        "description",
        "tier_number",
        "lower_bound_usd",
        "upper_bound_usd",
        "max_leverage",
        "maintenance_margin_rate",
        "maintenance_deduction_usd",
        "checked_at_utc",
        "source_system",
        "source_url",
        "blocker",
    ]
    payload = meta if isinstance(meta, dict) else {}
    raw_tables = payload.get("marginTables", [])
    table_payloads: dict[int, dict[str, object]] = {}
    if isinstance(raw_tables, list):
        for item in raw_tables:
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[1], dict):
                continue
            try:
                table_id = int(item[0])
            except (TypeError, ValueError):
                continue
            table_payloads[table_id] = item[1]

    universe_rows = universe if isinstance(universe, list) else payload.get("universe", [])
    if not isinstance(universe_rows, list):
        universe_rows = []
    referenced: dict[int, int] = {}
    for row in universe_rows:
        if not isinstance(row, dict):
            continue
        try:
            table_id = int(row.get("marginTableId"))
            max_leverage = int(row.get("maxLeverage"))
        except (TypeError, ValueError):
            continue
        referenced[table_id] = max(referenced.get(table_id, 0), max_leverage)

    for table_id, max_leverage in referenced.items():
        if table_id in table_payloads or table_id >= 50:
            continue
        table_payloads[table_id] = {
            "description": "single_tier_table_id_below_50",
            "marginTiers": [{"lowerBound": "0", "maxLeverage": max_leverage}],
        }

    rows: list[dict[str, object]] = []
    timestamp = checked_at or datetime.now(timezone.utc).isoformat()
    for table_id in sorted(set(table_payloads) | set(referenced)):
        table = table_payloads.get(table_id)
        if not isinstance(table, dict):
            rows.append(
                {
                    "api_url": info_url,
                    "margin_table_id": table_id,
                    "description": "",
                    "tier_number": "",
                    "lower_bound_usd": "",
                    "upper_bound_usd": "",
                    "max_leverage": referenced.get(table_id, ""),
                    "maintenance_margin_rate": "",
                    "maintenance_deduction_usd": "",
                    "checked_at_utc": timestamp,
                    "source_system": "hyperliquid_testnet_meta",
                    "source_url": HYPERLIQUID_MARGIN_TIERS_SOURCE_URL,
                    "blocker": "referenced_margin_table_missing_from_meta",
                }
            )
            continue
        raw_tiers = table.get("marginTiers", [])
        tiers: list[tuple[float, int]] = []
        if isinstance(raw_tiers, list):
            for tier in raw_tiers:
                if not isinstance(tier, dict):
                    continue
                lower = _hyperliquid_safe_float(tier.get("lowerBound"))
                leverage = _hyperliquid_safe_float(tier.get("maxLeverage"))
                if lower is None or leverage is None or leverage <= 0:
                    continue
                tiers.append((lower, int(leverage)))
        tiers.sort(key=lambda value: value[0])
        previous_rate = 0.0
        deduction = 0.0
        for tier_number, (lower, max_leverage) in enumerate(tiers):
            rate = 1.0 / (2.0 * max_leverage)
            if tier_number > 0:
                deduction += lower * (rate - previous_rate)
            upper: float | str = tiers[tier_number + 1][0] if tier_number + 1 < len(tiers) else ""
            rows.append(
                {
                    "api_url": info_url,
                    "margin_table_id": table_id,
                    "description": str(table.get("description", "") or ""),
                    "tier_number": tier_number,
                    "lower_bound_usd": lower,
                    "upper_bound_usd": upper,
                    "max_leverage": max_leverage,
                    "maintenance_margin_rate": rate,
                    "maintenance_deduction_usd": deduction,
                    "checked_at_utc": timestamp,
                    "source_system": "hyperliquid_testnet_meta",
                    "source_url": HYPERLIQUID_MARGIN_TIERS_SOURCE_URL,
                    "blocker": "",
                }
            )
            previous_rate = rate
    return pd.DataFrame(rows, columns=columns)


def refresh_hyperliquid_testnet_market_inventory(
    root: Path = ROOT,
    info_url: str = HYPERLIQUID_TESTNET_INFO_URL,
) -> pd.DataFrame:
    output = root / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv"
    margin_output = root / "reports" / "active" / "hyperliquid_testnet_margin_tiers.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame, margin_tiers, blockers = _hyperliquid_inventory_frame(info_url=info_url)
    if blockers and frame.empty:
        frame = pd.DataFrame(
            [
                {
                    "api_url": info_url,
                    "market_type": "perp",
                    "asset": "",
                    "asset_index": "",
                    "universe_name": "",
                    "sz_decimals": "",
                    "max_leverage": "",
                    "margin_table_id": "",
                    "only_isolated": False,
                    "margin_mode": "",
                    "is_delisted": True,
                    "tradable_perp": False,
                    "mid_price": None,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "fetch_blocker": ";".join(blockers),
                }
            ]
        )
    elif blockers:
        frame = frame.copy()
        frame["fetch_blocker"] = ";".join(blockers)
    else:
        frame = frame.copy()
        frame["fetch_blocker"] = ""
    frame.to_csv(output, index=False)
    margin_tiers.to_csv(margin_output, index=False)
    return frame


def _hyperliquid_market_indexes(root: Path = ROOT) -> tuple[dict[str, list[str]], list[str]]:
    inventory = _read_csv_or_empty(root / "reports" / "active" / "hyperliquid_testnet_market_inventory.csv")
    if inventory.empty or "asset" not in inventory.columns:
        inventory = refresh_hyperliquid_testnet_market_inventory(root=root)
    blockers: list[str] = []
    if not inventory.empty and "fetch_blocker" in inventory.columns:
        blockers = sorted(
            {
                item.strip()
                for value in inventory.get("fetch_blocker", pd.Series(dtype=object)).tolist()
                for item in str(value or "").split(";")
                if item.strip() and item.strip().lower() != "nan"
            }
        )
    market_index: dict[str, list[str]] = {}
    if inventory.empty:
        return market_index, blockers or ["hyperliquid_testnet_market_inventory_empty"]
    tradable = inventory.get("tradable_perp", pd.Series(dtype=bool)).fillna(False).astype(bool)
    for _, row in inventory.loc[tradable].iterrows():
        asset = str(row.get("asset", "") or "").upper().strip()
        label = str(row.get("universe_name", "") or row.get("asset", "") or "").strip()
        asset_index = row.get("asset_index", "")
        if asset and label:
            market_index.setdefault(asset, []).append(f"{label}#{asset_index}")
    for asset in market_index:
        market_index[asset] = sorted(set(market_index[asset]))
    return market_index, blockers


def _hyperliquid_assets_for_pair(pair: str) -> tuple[str, str]:
    left, right = _split_pair(pair)
    return left.split("-", 1)[0].upper(), right.split("-", 1)[0].upper()


def _hyperliquid_pair_key(pair: object) -> str:
    try:
        asset_x, asset_y = _hyperliquid_assets_for_pair(str(pair or ""))
    except ValueError:
        return str(pair or "").strip().upper().replace("/", "-")
    return f"{asset_x}-{asset_y}"


def _hyperliquid_candidate_source_frame(root: Path) -> pd.DataFrame:
    for path in [
        root / "data" / "processed" / "pair_universe.csv",
        root / "reports" / "rl" / "base_rl_route_candidates.csv",
        root / "reports" / "active" / "fresh_live_dashboard_pair_queue.csv",
        root / "reports" / "active" / "non_eth_route_submit_queue.csv",
    ]:
        frame = _read_csv_or_empty(path)
        if not frame.empty and "pair" in frame.columns:
            working = frame.copy()
            if "candidate_id" not in working.columns:
                working["candidate_id"] = ""
            return working
    return pd.DataFrame()


def hyperliquid_testnet_order_preflight_status() -> dict[str, object]:
    """Return configuration-only Hyperliquid readiness without reading a secret.

    The live no-order preflight lives in ``quant_platform.hyperliquid_testnet``.
    This lightweight status is safe to call from reports and shortlist refreshes.
    """

    from quant_platform.hyperliquid_testnet import HyperliquidTestnetConfig, hyperliquid_sdk_installed

    config = HyperliquidTestnetConfig.paper_testnet_from_env()
    adapter_contract = validate_venue_order_client_adapter("hyperliquid")
    account_configured = bool(config.master_address)
    agent_configured = bool(config.agent_address)
    keychain_configured = bool(config.keychain_service)
    submit_orders_enabled = bool(config.submit_orders)
    pair_executor_available = bool(
        hyperliquid_sdk_installed()
        and adapter_contract.get("pair_submission_capable")
    )
    blockers: list[str] = list(config.configuration_blockers())
    if not adapter_contract.get("configured"):
        blockers.append("missing_hyperliquid_order_client_adapter")
    elif not adapter_contract.get("valid"):
        blockers.append(f"hyperliquid_order_client_adapter_invalid:{adapter_contract.get('error')}")
    elif not adapter_contract.get("pair_submission_capable"):
        blockers.append("hyperliquid_pair_submission_not_supported")
    elif not adapter_contract.get("exchange_submission_capable"):
        blockers.append("record_only_hyperliquid_order_client_adapter")
    if not pair_executor_available:
        blockers.append("missing_hyperliquid_python_sdk")
    if not submit_orders_enabled:
        blockers.append(f"{HYPERLIQUID_TESTNET_SUBMIT_ORDERS_ENV.lower()}_false")
    ready = bool(
        adapter_contract.get("valid")
        and adapter_contract.get("pair_submission_capable")
        and adapter_contract.get("exchange_submission_capable")
        and pair_executor_available
        and not config.configuration_blockers()
        and account_configured
        and agent_configured
        and keychain_configured
        and submit_orders_enabled
    )
    return {
        "ready": ready,
        "adapter_configured": bool(adapter_contract.get("configured")),
        "adapter_valid": bool(adapter_contract.get("valid")),
        "exchange_submission_capable": bool(adapter_contract.get("exchange_submission_capable")),
        "single_leg_order_path_blocked": bool(adapter_contract.get("record_only")),
        "pair_executor_available": pair_executor_available,
        "account_address_configured": account_configured,
        "master_address_configured": account_configured,
        "agent_address_configured": agent_configured,
        "keychain_service_configured": keychain_configured,
        # Kept for existing report schemas. This indicates configured Keychain
        # storage rather than a raw secret environment variable.
        "secret_key_configured": keychain_configured,
        "submit_orders_enabled": submit_orders_enabled,
        "blocker": ";".join(sorted(set(blockers))),
        "adapter_path": str(adapter_contract.get("adapter_path") or ""),
    }


def refresh_hyperliquid_execution_compatibility_table(root: Path = ROOT) -> pd.DataFrame:
    candidates = _hyperliquid_candidate_source_frame(root)
    output = root / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "pair",
        "candidate_id",
        "asset_x",
        "asset_y",
        "testnet_perp_x",
        "testnet_perp_y",
        "both_legs_testnet_perp",
        "hyperliquid_execution_mode",
        "mirrorable_for_paper",
        "mirror_blocker",
        "checked_at_utc",
    ]
    if candidates.empty or "pair" not in candidates.columns:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    market_index, fetch_blockers = _hyperliquid_market_indexes(root=root)
    checked_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in candidates.iterrows():
        pair = str(row.get("pair", "") or "").strip().replace("_", "-")
        if not pair:
            continue
        candidate_id = str(row.get("candidate_id", "") or "")
        key = (pair, candidate_id)
        if key in seen:
            continue
        seen.add(key)
        try:
            asset_x, asset_y = _hyperliquid_assets_for_pair(pair)
        except ValueError:
            continue
        testnet_x = market_index.get(asset_x, [])
        testnet_y = market_index.get(asset_y, [])
        both_perp = bool(testnet_x and testnet_y)
        blockers: list[str] = list(fetch_blockers)
        if not testnet_x:
            blockers.append(f"missing_hyperliquid_testnet_perp:{asset_x}")
        if not testnet_y:
            blockers.append(f"missing_hyperliquid_testnet_perp:{asset_y}")
        execution_mode = "hyperliquid_testnet_perp" if both_perp and not fetch_blockers else ""
        rows.append(
            {
                "pair": pair,
                "candidate_id": candidate_id,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "testnet_perp_x": ";".join(testnet_x),
                "testnet_perp_y": ";".join(testnet_y),
                "both_legs_testnet_perp": both_perp,
                "hyperliquid_execution_mode": execution_mode,
                "mirrorable_for_paper": bool(execution_mode),
                "mirror_blocker": ";".join(blockers),
                "checked_at_utc": checked_at,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["mirrorable_for_paper", "pair"], ascending=[False, True]).reset_index(drop=True)
    frame.to_csv(output, index=False)
    return frame


def refresh_hyperliquid_testnet_candidate_shortlist(root: Path = ROOT, max_pairs: int = 10) -> pd.DataFrame:
    compatibility = refresh_hyperliquid_execution_compatibility_table(root=root)
    universe = _read_csv_or_empty(root / "data" / "processed" / "pair_universe.csv")
    output = root / "reports" / "active" / "hyperliquid_testnet_candidate_shortlist.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "shortlist_rank",
        "pair",
        "candidate_id",
        "asset_x",
        "asset_y",
        "hyperliquid_execution_mode",
        "hyperliquid_testnet_perp_x",
        "hyperliquid_testnet_perp_y",
        "source_decision_bucket",
        "source_decision_reason",
        "combined_score",
        "acceptance_score",
        "available_timeframes",
        "hyperliquid_lane_status",
        "order_preflight_ready",
        "order_preflight_blocker",
        "order_adapter_configured",
        "account_address_configured",
        "secret_key_configured",
        "submit_orders_enabled",
        "next_action",
        "evidence_path",
    ]
    if compatibility.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    supported = compatibility[
        compatibility.get("mirrorable_for_paper", pd.Series(dtype=bool)).fillna(False).astype(bool)
    ].copy()
    if supported.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame

    if not universe.empty and "pair" in universe.columns:
        keep_cols = [
            col
            for col in ["pair", "decision_bucket", "decision_reason", "combined_score", "acceptance_score", "available_timeframes"]
            if col in universe.columns
        ]
        universe_meta = universe[keep_cols].copy()
        universe_meta["_hyperliquid_pair_key"] = universe_meta["pair"].map(_hyperliquid_pair_key)
        universe_meta = universe_meta.drop(columns=["pair"]).drop_duplicates("_hyperliquid_pair_key")
        supported["_hyperliquid_pair_key"] = supported["pair"].map(_hyperliquid_pair_key)
        supported = supported.merge(universe_meta, on="_hyperliquid_pair_key", how="left")

    bucket_rank = {"PROMOTE": 0, "WATCH": 1, "FETCH_MORE_DATA": 2, "REJECT": 3}
    supported["_bucket_rank"] = supported.get("decision_bucket", pd.Series(dtype=object)).astype(str).map(bucket_rank).fillna(9)
    supported["combined_score"] = pd.to_numeric(supported.get("combined_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    supported["acceptance_score"] = pd.to_numeric(supported.get("acceptance_score", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    supported = supported.sort_values(
        ["_bucket_rank", "combined_score", "acceptance_score", "pair"],
        ascending=[True, False, False, True],
    ).head(max_pairs).copy()

    preflight = hyperliquid_testnet_order_preflight_status()
    lane_status: list[str] = []
    next_actions: list[str] = []
    for _, row in supported.iterrows():
        bucket = str(row.get("decision_bucket", "") or "").strip().upper()
        if bool(preflight["ready"]) and bucket == "PROMOTE":
            lane_status.append("hyperliquid_supported_and_preflight_ready")
            next_actions.append("paper_journal_then_manual_submit_approval")
        elif bucket == "PROMOTE":
            lane_status.append("hyperliquid_supported_promoted_preflight_blocked")
            next_actions.append("configure_hyperliquid_order_adapter_wallet_and_submit_flag")
        elif bucket == "WATCH":
            lane_status.append("hyperliquid_supported_research_watch")
            next_actions.append("refresh_research_then_hyperliquid_preflight")
        else:
            lane_status.append("hyperliquid_supported_but_research_not_promoted")
            next_actions.append("repair_research_filters_before_hyperliquid_submit")
    supported["hyperliquid_lane_status"] = lane_status
    supported["order_preflight_ready"] = bool(preflight["ready"])
    supported["order_preflight_blocker"] = str(preflight["blocker"])
    supported["order_adapter_configured"] = bool(preflight["adapter_configured"])
    supported["account_address_configured"] = bool(preflight["account_address_configured"])
    supported["secret_key_configured"] = bool(preflight["secret_key_configured"])
    supported["submit_orders_enabled"] = bool(preflight["submit_orders_enabled"])
    supported["next_action"] = next_actions
    supported = supported.reset_index(drop=True)
    supported.insert(0, "shortlist_rank", supported.index + 1)
    supported["evidence_path"] = str(root / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv")
    supported = supported.rename(
        columns={
            "testnet_perp_x": "hyperliquid_testnet_perp_x",
            "testnet_perp_y": "hyperliquid_testnet_perp_y",
            "decision_bucket": "source_decision_bucket",
            "decision_reason": "source_decision_reason",
        }
    )
    supported = supported.drop(columns=["_bucket_rank", "_hyperliquid_pair_key"], errors="ignore")
    frame = supported.reindex(columns=columns, fill_value="")
    frame.to_csv(output, index=False)
    return frame


def load_hyperliquid_execution_compatibility(root: Path = ROOT) -> pd.DataFrame:
    path = root / "reports" / "active" / "hyperliquid_execution_market_compatibility.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def hyperliquid_execution_compatibility_snapshot(
    pairs: list[str] | tuple[str, ...] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    table = load_hyperliquid_execution_compatibility(root=root)
    normalized_pairs = [str(value or "").strip() for value in (pairs or []) if str(value or "").strip()]
    if table.empty or "pair" not in table.columns:
        return {
            "checked": False,
            "table_available": False,
            "route_pairs": normalized_pairs,
            "mirrorable_pairs": [],
            "paper_only_pairs": normalized_pairs,
            "missing_pairs": normalized_pairs,
            "blocker": "hyperliquid_execution_compatibility_table_missing",
        }
    working = table.copy()
    working["pair"] = working["pair"].astype(str).str.strip()
    if normalized_pairs:
        working = working[working["pair"].isin(normalized_pairs)].copy()
    mirrorable_pairs: list[str] = []
    paper_only_pairs: list[str] = []
    for _, row in working.iterrows():
        pair = str(row.get("pair", "")).strip()
        mirrorable = _boolish(row.get("mirrorable_for_paper", False))
        if mirrorable:
            mirrorable_pairs.append(pair)
        else:
            paper_only_pairs.append(pair)
    observed_pairs = set(working["pair"].astype(str).tolist())
    missing_pairs = [pair for pair in normalized_pairs if pair not in observed_pairs]
    if normalized_pairs:
        if missing_pairs:
            blocker = "hyperliquid_pair_compatibility_missing"
        elif paper_only_pairs:
            blocker = "hyperliquid_testnet_perp_support_incomplete"
        else:
            blocker = ""
    else:
        blocker = "" if mirrorable_pairs else "hyperliquid_no_supported_testnet_pairs"
    preflight = hyperliquid_testnet_order_preflight_status()
    return {
        "checked": True,
        "table_available": True,
        "route_pairs": normalized_pairs,
        "mirrorable_pairs": sorted(set(mirrorable_pairs)),
        "paper_only_pairs": sorted(set(paper_only_pairs)),
        "missing_pairs": missing_pairs,
        "blocker": blocker,
        "order_preflight_ready": bool(preflight["ready"]),
        "order_preflight_blocker": str(preflight["blocker"]),
    }


def refresh_dydx_execution_compatibility_table(root: Path = ROOT) -> pd.DataFrame:
    attempts = load_dydx_execution_attempts(root=root)
    output = root / "reports" / "active" / "dydx_execution_market_compatibility.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "market",
        "compatible_for_paper_submit",
        "confirmed_attempts",
        "unconfirmed_attempts",
        "blocked_attempts",
        "total_attempts",
        "last_status",
        "last_route_label",
        "last_submit_mode",
        "last_attempt_timestamp_utc",
        "confirmed_route_labels",
    ]
    if attempts.empty:
        frame = pd.DataFrame(columns=columns)
        frame.to_csv(output, index=False)
        return frame
    attempts = attempts.copy()
    attempts["confirmed"] = attempts.get("confirmed", pd.Series(dtype=object)).astype(str).str.lower().isin({"true", "1", "yes"})
    rows: list[dict[str, object]] = []
    for market, frame in attempts.groupby("market", sort=True):
        ordered = frame.sort_values("timestamp_utc")
        last = ordered.iloc[-1]
        statuses = ordered["status"].astype(str)
        confirmed_routes = sorted(set(ordered.loc[ordered["confirmed"], "route_label"].astype(str)))
        last_status = str(last.get("status", "")).strip().lower()
        confirmed_route_available = bool(confirmed_routes)
        rows.append(
            {
                "market": str(market),
                "compatible_for_paper_submit": confirmed_route_available,
                "confirmed_attempts": int(ordered["confirmed"].sum()),
                "unconfirmed_attempts": int((statuses == "broadcast_accepted_unconfirmed").sum()),
                "blocked_attempts": int(statuses.str.startswith("paper_blocked").sum()),
                "total_attempts": int(len(ordered)),
                "last_status": last_status,
                "last_route_label": str(last.get("route_label", "")),
                "last_submit_mode": str(last.get("submit_mode", "")),
                "last_attempt_timestamp_utc": str(last.get("timestamp_utc", "")),
                "confirmed_route_labels": ";".join(confirmed_routes),
            }
        )
    compatibility = pd.DataFrame(rows, columns=columns).sort_values(
        ["compatible_for_paper_submit", "market"],
        ascending=[False, True],
    )
    compatibility.to_csv(output, index=False)
    return compatibility


def load_dydx_execution_compatibility(root: Path = ROOT) -> pd.DataFrame:
    path = root / "reports" / "active" / "dydx_execution_market_compatibility.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def dydx_execution_compatibility_snapshot(
    markets: list[str] | tuple[str, ...] | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    table = load_dydx_execution_compatibility(root=root)
    normalized_markets = [str(value or "").strip().upper() for value in (markets or []) if str(value or "").strip()]
    if table.empty or "market" not in table.columns:
        return {
            "checked": False,
            "table_available": False,
            "route_markets": normalized_markets,
            "compatible_markets": [],
            "incompatible_markets": normalized_markets,
            "missing_markets": normalized_markets,
            "blocker": "execution_compatibility_table_missing",
        }
    working = table.copy()
    working["market"] = working["market"].astype(str).str.upper()
    if normalized_markets:
        working = working[working["market"].isin(normalized_markets)].copy()
    compatible_markets: list[str] = []
    incompatible_markets: list[str] = []
    for _, row in working.iterrows():
        market = str(row.get("market", "")).upper()
        compatible = str(row.get("compatible_for_paper_submit", "")).strip().lower() in {"true", "1", "yes"}
        if compatible:
            compatible_markets.append(market)
        else:
            incompatible_markets.append(market)
    observed_markets = set(working["market"].astype(str).tolist())
    missing_markets = [market for market in normalized_markets if market not in observed_markets]
    if normalized_markets:
        if missing_markets:
            blocker = "route_market_compatibility_missing"
        elif incompatible_markets:
            blocker = "route_market_unconfirmed_on_exchange"
        else:
            blocker = ""
    else:
        blocker = "" if compatible_markets else "execution_compatibility_missing_confirmed_market"
    return {
        "checked": True,
        "table_available": True,
        "route_markets": normalized_markets,
        "compatible_markets": sorted(set(compatible_markets)),
        "incompatible_markets": sorted(set(incompatible_markets)),
        "missing_markets": missing_markets,
        "blocker": blocker,
    }


def dydx_market_confirmed_for_paper_submit(market: str, root: Path = ROOT) -> tuple[bool, str]:
    table = load_dydx_execution_compatibility(root=root)
    if table.empty or "market" not in table.columns:
        return False, "no_execution_compatibility_evidence"
    matches = table[table["market"].astype(str).str.upper() == str(market or "").upper()]
    if matches.empty:
        return False, "market_not_in_execution_compatibility_table"
    row = matches.iloc[-1]
    compatible = str(row.get("compatible_for_paper_submit", "")).strip().lower() in {"true", "1", "yes"}
    if compatible:
        return True, ""
    return False, str(row.get("last_status", "") or "market_not_confirmed_on_exchange")


def refresh_non_eth_route_submit_queue(root: Path = ROOT, queue_path: Path | None = None) -> pd.DataFrame:
    path = queue_path or (root / "reports" / "active" / "non_eth_route_submit_queue.csv")
    if not path.exists():
        return pd.DataFrame()
    try:
        queue = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if queue.empty:
        return queue
    compatibility = load_dydx_execution_compatibility(root=root)
    compatibility_map: dict[str, dict[str, object]] = {}
    if not compatibility.empty and "market" in compatibility.columns:
        for _, row in compatibility.iterrows():
            compatibility_map[str(row.get("market", "")).strip().upper()] = {
                "last_status": str(row.get("last_status", "") or "").strip(),
                "compatible": _boolish(row.get("compatible_for_paper_submit", False)),
                "confirmed_routes": str(row.get("confirmed_route_labels", "") or "").strip(),
            }

    def _route_markets(value: object) -> list[str]:
        return [part.strip().upper() for part in str(value or "").split(";") if part.strip()]

    working = queue.copy()
    statuses: list[str] = []
    confirmed_flags: list[bool] = []
    submit_states: list[str] = []
    next_actions: list[str] = []
    for _, row in working.iterrows():
        markets = _route_markets(row.get("route_markets", ""))
        market_statuses = []
        all_confirmed = bool(markets)
        missing_any = False
        for market in markets:
            market_record = compatibility_map.get(market)
            status = "market_not_in_execution_compatibility_table"
            compatible = False
            if market_record:
                status = str(market_record.get("last_status", "") or "").strip() or "market_not_in_execution_compatibility_table"
                compatible = bool(market_record.get("compatible", False))
            route_suffix = ""
            confirmed_routes = ""
            if market_record:
                raw_routes = str(market_record.get("confirmed_routes", "") or "").strip()
                if raw_routes and raw_routes.lower() != "nan":
                    confirmed_routes = raw_routes
            if confirmed_routes:
                route_suffix = f"@{confirmed_routes}"
            market_statuses.append(f"{market}:{'confirmed_route_available' if compatible else status}{route_suffix}")
            if not compatible:
                all_confirmed = False
            if status == "market_not_in_execution_compatibility_table":
                missing_any = True
        statuses.append(";".join(market_statuses))
        confirmed_flags.append(all_confirmed)
        if all_confirmed:
            submit_states.append("ready_for_paper_submit")
            next_actions.append("route through confirmed markets only")
        elif missing_any:
            submit_states.append("wait_for_compatibility_evidence")
            next_actions.append("probe missing route markets before any paper submit")
        else:
            submit_states.append("wait_for_exchange_confirmation")
            next_actions.append("re-probe these two markets before any paper submit")
    working["route_market_statuses"] = statuses
    working["all_route_markets_confirmed"] = confirmed_flags
    working["current_submit_state"] = submit_states
    working["next_action"] = next_actions
    working.to_csv(path, index=False)
    return working


def dydx_account_state_snapshot(config: DydxNetworkConfig | None = None) -> dict[str, object]:
    config = config or DydxNetworkConfig.paper_testnet_from_env()
    if not config.wallet_address or not config.rest_indexer:
        return {
            "checked": False,
            "open_markets": [],
            "positions": [],
            "blocker": "account_state_unverified",
        }
    if not dydx_indexer_adapter_available():
        return {
            "checked": False,
            "open_markets": [],
            "positions": [],
            "blocker": "dydx_indexer_unavailable_for_account_state",
        }
    try:
        from dydx_v4_client.indexer.rest.indexer_client import IndexerClient
    except Exception:
        return {
            "checked": False,
            "open_markets": [],
            "positions": [],
            "blocker": "dydx_indexer_unavailable_for_account_state",
        }

    async def _fetch_positions() -> dict[str, object]:
        indexer = IndexerClient(str(config.rest_indexer))
        payload = await indexer.account.get_subaccount_perpetual_positions(str(config.wallet_address), 0, limit=100)
        rows = payload.get("positions") or payload.get("perpetualPositions") or []
        aggregated: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            market = str(row.get("market") or row.get("ticker") or row.get("perpetualMarket") or "").upper()
            if not market:
                continue
            size = row.get("size") or row.get("sumOpen") or row.get("netSize") or row.get("quantity") or 0
            try:
                aggregated[market] = aggregated.get(market, 0.0) + float(size)
            except (TypeError, ValueError):
                continue
        positions = [
            {"market": market, "size": size}
            for market, size in sorted(aggregated.items())
            if abs(float(size)) > 1e-12
        ]
        return {"positions": positions}

    try:
        payload = DydxV4IndexerAdapter._run(_fetch_positions())
    except Exception:
        return {
            "checked": False,
            "open_markets": [],
            "positions": [],
            "blocker": "account_state_unverified",
        }
    positions = payload.get("positions", []) if isinstance(payload, dict) else []
    open_markets = [str(row.get("market", "")) for row in positions if str(row.get("market", ""))]
    return {
        "checked": True,
        "open_markets": open_markets,
        "positions": positions,
        "blocker": "orphan_leg_open" if open_markets else "",
    }


def load_browser_account_state_override(root: Path = ROOT) -> dict[str, object]:
    path = root / "reports" / "active" / "browser_account_state_override.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_browser_account_state_override(
    confirmed_flat: bool,
    root: Path = ROOT,
    note: str = "",
    source: str = "browser_user_confirmed",
) -> Path:
    path = root / "reports" / "active" / "browser_account_state_override.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "confirmed_flat": bool(confirmed_flat),
        "source": str(source),
        "note": str(note),
        "confirmed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def effective_dydx_account_state_snapshot(
    config: DydxNetworkConfig | None = None,
    root: Path = ROOT,
) -> dict[str, object]:
    live = dydx_account_state_snapshot(config=config)
    override = load_browser_account_state_override(root=root)
    live_open_markets = [str(market) for market in live.get("open_markets", []) if str(market)]
    live_positions = live.get("positions", [])
    live_has_open_positions = bool(live_open_markets or live_positions)
    if str(override.get("confirmed_flat", "")).strip().lower() in {"true", "1", "yes"}:
        if bool(live.get("checked", False)) and live_has_open_positions:
            live = dict(live)
            live["source"] = "dydx_indexer_browser_override_conflict"
            live["browser_override_confirmed_at_utc"] = str(override.get("confirmed_at_utc", ""))
            live["browser_override_note"] = str(override.get("note", ""))
            live["browser_override_conflict"] = True
            live["blocker"] = str(live.get("blocker", "") or "orphan_leg_open")
            return live
        return {
            "checked": True,
            "open_markets": [],
            "positions": [],
            "blocker": "",
            "source": "browser_override",
            "browser_override_confirmed_at_utc": str(override.get("confirmed_at_utc", "")),
            "browser_override_note": str(override.get("note", "")),
            "live_snapshot_checked": bool(live.get("checked", False)),
            "live_snapshot_open_markets": live.get("open_markets", []),
            "live_snapshot_positions": live.get("positions", []),
            "live_snapshot_blocker": str(live.get("blocker", "")),
        }
    live = dict(live)
    live["source"] = "dydx_indexer"
    return live


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
        if not intent.reduce_only:
            compatible, compatibility_reason = dydx_market_confirmed_for_paper_submit(intent.market)
            if not compatible:
                return FillReport(
                    order_id="paper-market-incompatible",
                    market=intent.market,
                    side=intent.side,
                    size=intent.size,
                    avg_price=float(intent.limit_price or 0.0),
                    fee=0.0,
                    slippage_bps=0.0,
                    status=f"paper_blocked_market_unconfirmed:{compatibility_reason}",
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


class PaperVenueExecution:
    """Generic paper execution adapter for venues backed by an injected order client."""

    def __init__(self, venue: str, client: VenueOrderClient, config: object | None = None) -> None:
        self.venue = normalize_venue_name(venue) or "unknown"
        self.client = client
        self.config = config
        self.pair_submission_capable = bool(
            getattr(client, "pair_submission_capable", False)
            and callable(getattr(client, "submit_pair", None))
        )

    def market_data(self, market: str) -> dict:
        return {
            "market": market,
            "status": "paper",
            "venue": self.venue,
            "source": "paper_venue_execution",
        }

    def place_order(self, intent: OrderIntent) -> FillReport:
        config = self.config or {"venue": self.venue, "mode": ExecutionMode.PAPER.value}
        return _place_order_call(self.client.place_order, intent, config)

    def submit_pair(self, intents: tuple[OrderIntent, ...]):
        submit_pair = getattr(self.client, "submit_pair", None)
        if not callable(submit_pair):
            raise NotImplementedError(f"{self.venue} adapter does not support coordinated pair submission")
        config = self.config or {"venue": self.venue, "mode": ExecutionMode.PAPER.value}
        return _place_order_call(submit_pair, intents, config)

    def positions(self) -> list[dict]:
        if hasattr(self.client, "positions"):
            try:
                return self.client.positions()
            except Exception:
                return []
        return []

    def funding(self, market: str) -> dict:
        if hasattr(self.client, "funding"):
            return getattr(self.client, "funding")(market)
        return {"market": market, "status": "paper", "venue": self.venue}


def build_execution_venue(
    venue: str | DydxNetworkConfig = "dydx",
    config: object | None = None,
    order_client: VenueOrderClient | None = None,
    market_data_client: DydxMarketDataClient | None = None,
) -> ExecutionVenue:
    # Backward-compatible signature: callers historically passed config as first arg.
    if isinstance(venue, DydxNetworkConfig):
        config = venue
        venue_name = "dydx"
    else:
        venue_name = normalize_venue_name(venue)
    config = config or DydxNetworkConfig()
    if venue_name != "dydx":
        if getattr(config, "mode", ExecutionMode.PAPER) != ExecutionMode.PAPER:
            raise NotImplementedError(f"Live {venue_name} execution must be implemented behind explicit risk gates.")
        if order_client is None:
            return UnsupportedVenueExecution(venue_name)
        return PaperVenueExecution(venue_name, order_client, config=config)
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
    venue: str = "dydx",
    price_x: float | None = None,
    price_y: float | None = None,
    step_size_x: float | None = None,
    step_size_y: float | None = None,
) -> tuple[OrderIntent, OrderIntent]:
    left, right = _split_pair(pair)
    signal = 1.0 if side.upper() == "LONG_SPREAD" else -1.0
    hedge = abs(float(hedge_ratio or 1.0))
    beta_abs = abs(float(beta or 1.0))
    if price_x and price_y and float(price_x) > 0 and float(price_y) > 0:
        hedge = hedge * float(price_y) / float(price_x)
    gross_scale = 1.0 + hedge * beta_abs
    right_notional = notional_usd / gross_scale
    left_notional = notional_usd * hedge * beta_abs / gross_scale
    left_side = "SELL" if signal > 0 else "BUY"
    right_side = "BUY" if signal > 0 else "SELL"
    left_size = left_notional / float(price_x) if price_x and float(price_x) > 0 else left_notional
    right_size = right_notional / float(price_y) if price_y and float(price_y) > 0 else right_notional
    left_size = _quantize_size_to_step(left_size, step_size_x)
    right_size = _quantize_size_to_step(right_size, step_size_y)
    return (
        OrderIntent(market=_venue_market(left, venue), side=left_side, size=left_size),
        OrderIntent(market=_venue_market(right, venue), side=right_side, size=right_size),
    )


def _quantize_size_to_step(size: float, step_size: float | None) -> float:
    if not step_size or float(step_size) <= 0:
        return float(size)
    step = float(step_size)
    units = math.floor(float(size) / step)
    return round(units * step, 12)


def build_research_gated_paper_plan(
    signal_row: dict,
    acceptance_report: pd.DataFrame,
    notional_usd: float,
    venue: str = "dydx",
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
        venue=venue,
        price_x=float(signal_row.get("price_x", 0.0) or 0.0),
        price_y=float(signal_row.get("price_y", 0.0) or 0.0),
        step_size_x=float(signal_row.get("step_size_x", 0.0) or 0.0),
        step_size_y=float(signal_row.get("step_size_y", 0.0) or 0.0),
    )
    invalid_markets = [intent.market for intent in intents if float(intent.size) <= 0]
    if invalid_markets:
        return SpreadOrderPlan(
            pair=pair,
            strategy_id=strategy_id,
            status="blocked",
            reason=f"venue_size_below_minimum:{','.join(invalid_markets)}",
            venue=venue,
        )
    return SpreadOrderPlan(
        pair=pair,
        strategy_id=strategy_id,
        status="paper_ready",
        reason="accepted",
        intents=intents,
        venue=venue,
    )


def submit_paper_plan(plan: SpreadOrderPlan, venue: ExecutionVenue) -> list[FillReport]:
    if plan.status != "paper_ready":
        return []
    pair_submit = getattr(venue, "submit_pair", None)
    if bool(getattr(venue, "pair_submission_capable", False)) and callable(pair_submit) and len(plan.intents) == 2:
        result = pair_submit(plan.intents)
        result_fills = list(getattr(result, "fills", ()) or ())
        if result_fills:
            return result_fills
        result_status = str(getattr(result, "status", "pair_submission_error") or "pair_submission_error")
        result_reason = str(getattr(result, "reason", "pair_submission_failed") or "pair_submission_failed")
        blocked = result_status == "pair_blocked"
        fill_status = (
            f"paper_blocked_pair_submission:{result_reason}"
            if blocked
            else f"paper_submission_rejected:{result_status}:{result_reason}"
        )
        return [
            FillReport(
                order_id=result_status,
                market=intent.market,
                side=intent.side,
                size=float(intent.size),
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status=fill_status,
            )
            for intent in plan.intents
        ]
    fills: list[FillReport] = []
    for intent in plan.intents:
        fill = venue.place_order(intent)
        fills.append(fill)
        if not fill_report_confirmed(fill):
            break
    return fills


def _plan_status_from_fills(plan: SpreadOrderPlan, fills: list[FillReport] | None) -> tuple[str, str]:
    if not fills:
        return plan.status, plan.reason
    statuses = [str(fill.status or "").strip().lower() for fill in fills]
    if all(status in CONFIRMED_PAPER_FILL_STATUSES for status in statuses) and len(fills) == len(plan.intents):
        return "paper_submitted", "exchange_confirmed"
    if any(status.startswith("paper_blocked") for status in statuses):
        return "blocked", f"execution_blocked:{';'.join(statuses)}"
    if any(status == "broadcast_accepted_unconfirmed" for status in statuses):
        return "execution_unconfirmed", f"exchange_unconfirmed:{';'.join(statuses)}"
    return "execution_partial", f"pair_incomplete:{';'.join(statuses)}"


def block_paper_plan_for_execution_config(plan: SpreadOrderPlan, blockers: list[str]) -> SpreadOrderPlan:
    if plan.status != "paper_ready" or not blockers:
        return plan
    venue = plan.venue or "dydx"
    return SpreadOrderPlan(
        pair=plan.pair,
        strategy_id=plan.strategy_id,
        status="blocked",
        reason=f"{venue}_not_ready:{';'.join(blockers)}",
        intents=plan.intents,
        venue=venue,
    )


def paper_trading_record(
    plan: SpreadOrderPlan,
    fills: list[FillReport] | None = None,
    blockers: list[str] | None = None,
    dashboard_snapshot: dict[str, object] | None = None,
) -> PaperTradingRecord:
    timestamp = datetime.now(timezone.utc).isoformat()
    derived_status, derived_reason = _plan_status_from_fills(plan, fills)
    entry_snapshot = {
        "pair": plan.pair,
        "strategy_id": plan.strategy_id,
        "venue": plan.venue or "dydx",
        "plan_status": derived_status,
        "plan_reason": derived_reason,
        "intents": [asdict(intent) for intent in plan.intents],
        "fills": [asdict(fill) for fill in fills or []],
    }
    if dashboard_snapshot:
        entry_snapshot.update(dashboard_snapshot)
    lifecycle_status = "audit_only"
    opened_timestamp = ""
    if derived_status in {"paper_submitted", "confirmed_on_exchange"}:
        lifecycle_status = "open"
        opened_timestamp = timestamp
    elif derived_status == "blocked":
        lifecycle_status = "blocked"
    elif derived_status == "execution_unconfirmed":
        lifecycle_status = "unconfirmed"
    elif derived_status == "execution_partial":
        lifecycle_status = "partial"
    return PaperTradingRecord(
        timestamp_utc=timestamp,
        pair=plan.pair,
        strategy_id=plan.strategy_id,
        plan_status=derived_status,
        plan_reason=derived_reason,
        blockers=";".join(blockers or []),
        intents_json=json.dumps([asdict(intent) for intent in plan.intents], sort_keys=True),
        fills_json=json.dumps([asdict(fill) for fill in fills or []], sort_keys=True),
        trade_id=f"{plan.pair}:{plan.strategy_id}:{plan.venue or 'dydx'}:{uuid4().hex[:12]}",
        venue=plan.venue or "dydx",
        lifecycle_status=lifecycle_status,
        opened_timestamp_utc=opened_timestamp,
        closed_timestamp_utc="",
        entry_snapshot_json=json.dumps(entry_snapshot, sort_keys=True),
        exit_snapshot_json="{}",
        realized_return="",
        outcome_label="",
    )


def append_paper_trading_record(record: PaperTradingRecord, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([asdict(record)])
    existing = _read_csv(output)
    if existing.empty and not output.exists():
        row.to_csv(output, index=False)
        return output
    combined = pd.concat([existing, row], ignore_index=True, sort=False).fillna("")
    combined.to_csv(output, index=False)
    return output


def append_paper_outcome_record(
    journal_path: str | Path,
    *,
    trade_id: str | None = None,
    pair: str | None = None,
    strategy_id: int | None = None,
    realized_return: float,
    outcome_label: str | None = None,
    exit_snapshot: dict[str, object] | None = None,
) -> tuple[Path, PaperTradingRecord]:
    journal = _read_csv(Path(journal_path))
    if journal.empty:
        raise ValueError("paper_trading_journal is missing or empty")
    working = journal.copy()
    if "trade_id" not in working.columns:
        working["trade_id"] = ""
    if "timestamp_utc" not in working.columns:
        working["timestamp_utc"] = ""
    matches = working
    if trade_id is not None:
        matches = matches[matches["trade_id"].astype(str) == str(trade_id)]
    else:
        if pair is not None:
            matches = matches[matches.get("pair", pd.Series(dtype=object)).astype(str) == str(pair)]
        if strategy_id is not None:
            matches = matches[pd.to_numeric(matches.get("strategy_id", pd.Series(dtype=object)), errors="coerce") == float(strategy_id)]
    if matches.empty:
        raise ValueError("no matching paper trade found to close")
    base = matches.sort_values("timestamp_utc").iloc[-1].to_dict()
    timestamp = datetime.now(timezone.utc).isoformat()
    realized = float(realized_return)
    label = outcome_label or ("win" if realized > 0 else "loss" if realized < 0 else "flat")
    resolved_exit_snapshot = dict(exit_snapshot or {})
    if not resolved_exit_snapshot:
        entry_snapshot = _safe_json_dict(base.get("entry_snapshot_json", "{}"))
        resolved_exit_snapshot = _lookup_dashboard_trade_snapshot(
            pair=str(base.get("pair", "")),
            strategy_label=str(entry_snapshot.get("dashboard_strategy", "") or ""),
        )
    record = PaperTradingRecord(
        timestamp_utc=timestamp,
        pair=str(base.get("pair", "")),
        strategy_id=int(float(base.get("strategy_id", 0) or 0)),
        plan_status="paper_completed",
        plan_reason="realized_outcome_recorded",
        blockers="",
        intents_json=str(base.get("intents_json", "[]") or "[]"),
        fills_json=str(base.get("fills_json", "[]") or "[]"),
        trade_id=str(base.get("trade_id", "")),
        venue=str(base.get("venue", "dydx") or "dydx"),
        lifecycle_status="closed",
        opened_timestamp_utc=str(base.get("opened_timestamp_utc", "")),
        closed_timestamp_utc=timestamp,
        entry_snapshot_json=str(base.get("entry_snapshot_json", "{}") or "{}"),
        exit_snapshot_json=json.dumps(resolved_exit_snapshot, sort_keys=True),
        realized_return=str(realized),
        outcome_label=label,
    )
    return append_paper_trading_record(record, journal_path), record


def _normalized_pair_key(value: object) -> str:
    return str(value or "").strip().upper().replace("/", "-")


def _lookup_dashboard_trade_snapshot(
    *,
    pair: str,
    strategy_label: str = "",
    root: Path = ROOT,
) -> dict[str, object]:
    scanner_path = root / "reports" / "active" / "wizard_research_scanner_capture.csv"
    frame = _read_csv(scanner_path)
    if frame.empty or "pair" not in frame.columns:
        return {}

    working = frame.copy()
    if "journal_layer" in working.columns:
        working = working[working["journal_layer"].astype(str) == "scanner_capture"].copy()
    if working.empty:
        return {}

    pair_key = _normalized_pair_key(pair)
    working["_pair_key"] = working["pair"].map(_normalized_pair_key)
    matches = working[working["_pair_key"] == pair_key].copy()
    if matches.empty:
        return {}

    strategy_hint = str(strategy_label or "").strip()
    if strategy_hint:
        strategy_matches = matches[matches.get("strategy_label", pd.Series(dtype=object)).astype(str) == strategy_hint].copy()
        if not strategy_matches.empty:
            matches = strategy_matches

    matches["_capture_ts"] = pd.to_datetime(matches.get("capture_timestamp_utc", pd.Series(dtype=object)), utc=True, errors="coerce")
    matches["_row_rank"] = pd.to_numeric(matches.get("row_index", pd.Series(dtype=object)), errors="coerce")
    matches = matches.sort_values(["_capture_ts", "_row_rank"], ascending=[False, True], na_position="last")

    def _snapshot_row(row: pd.Series) -> dict[str, object]:
        return {
            "dashboard_strategy": str(row.get("strategy_label", "") or ""),
            "dashboard_timeframe": str(row.get("timeframe", "") or ""),
            "dashboard_row_index": row.get("row_index", ""),
            "dashboard_capture_timestamp_utc": str(row.get("capture_timestamp_utc", "") or ""),
            "dashboard_refresh_timestamp_utc": str(row.get("scanner_refresh_timestamp_utc", "") or ""),
            "dashboard_updated_at_utc": str(row.get("updated_at_utc", "") or ""),
            "dashboard_strategy_family": str(row.get("strategy_family_from_row", "") or ""),
            "dashboard_strategy_variant": str(row.get("strategy_variant_from_row", "") or ""),
            "dashboard_zscore_source": row.get("zscore_green_source", ""),
            "dashboard_zscore_norm": row.get("zscore_norm_value", ""),
            "dashboard_zscore_roll": row.get("zscore_roll_value", ""),
            "dashboard_dependency_profile": row.get("dependency_profile", ""),
            "dashboard_profile_x_over_y": row.get("dependency_x_over_y", ""),
            "dashboard_profile_y_over_x": row.get("dependency_y_over_x", ""),
            "dashboard_corr": row.get("correlation_value", ""),
            "dashboard_johansen": row.get("coint_johansen_flag", ""),
            "dashboard_engle_granger": row.get("coint_engle_granger_flag", ""),
            "dashboard_hurst": row.get("hurst_value", ""),
            "dashboard_half_life": row.get("half_life_value", ""),
            "dashboard_sigma_0": row.get("sigma_0_count", ""),
            "dashboard_sigma_1": row.get("sigma_1_count", ""),
            "dashboard_sigma_2": row.get("sigma_2_count", ""),
            "dashboard_var_99": row.get("var_99", ""),
            "dashboard_cvar_99": row.get("cvar_99", ""),
            "dashboard_mdd_pct": row.get("max_drawdown", ""),
            "dashboard_return_pct": row.get("return_total", ""),
            "dashboard_sharpe": row.get("sharpe", ""),
            "dashboard_volume_x": row.get("volume_x", ""),
            "dashboard_volume_y": row.get("volume_y", ""),
            "dashboard_scanner_row_text_raw": row.get("scanner_row_text_raw", ""),
        }

    candidate_rows = [_snapshot_row(row) for _, row in matches.iterrows()]
    primary = candidate_rows[0] if candidate_rows else {}
    primary["dashboard_candidate_count"] = len(candidate_rows)
    primary["dashboard_candidates"] = candidate_rows
    return primary


def refresh_current_paper_watch_positions(
    journal_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    source = Path(journal_path)
    output = Path(output_path or CURRENT_PAPER_WATCH_POSITIONS)
    frame = _read_csv(source)
    columns = [
        "trade_id",
        "pair",
        "strategy_id",
        "venue",
        "opened_timestamp_utc",
        "last_event_timestamp_utc",
        "lifecycle_status",
        "plan_status",
        "plan_reason",
        "entry_snapshot_json",
        "realized_return",
        "outcome_label",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        pd.DataFrame(columns=columns).to_csv(output, index=False)
        return output
    working = frame.copy()
    if "trade_id" not in working.columns:
        working["trade_id"] = ""
    if "venue" not in working.columns:
        working["venue"] = "dydx"
    if "lifecycle_status" not in working.columns:
        fallback = working.get("plan_status", pd.Series(dtype=object)).astype(str)
        working["lifecycle_status"] = fallback.map(
            lambda value: "open"
            if value in {"paper_submitted", "confirmed_on_exchange"}
            else "closed" if value in {"paper_completed", "completed", "closed"} else "audit_only"
        )
    if "opened_timestamp_utc" not in working.columns:
        working["opened_timestamp_utc"] = ""
    if "entry_snapshot_json" not in working.columns:
        working["entry_snapshot_json"] = "{}"
    latest = working.sort_values("timestamp_utc").groupby("trade_id", dropna=False, as_index=False).tail(1)
    latest = latest[latest["lifecycle_status"].astype(str).isin({"open", "monitoring", "unconfirmed", "partial"})].copy()
    if latest.empty:
        pd.DataFrame(columns=columns).to_csv(output, index=False)
        return output
    latest["last_event_timestamp_utc"] = latest.get("timestamp_utc", pd.Series(dtype=object)).astype(str)
    latest = latest.reindex(columns=columns, fill_value="")
    latest.to_csv(output, index=False)
    return output


def refresh_paper_trade_rulebook(output_path: str | Path | None = None) -> Path:
    output = Path(output_path or PAPER_TRADE_RULEBOOK_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Paper Trade Rulebook",
        "",
        "This file defines the conservative operating rules for the recurring paper loop.",
        "",
        "## Open Rules",
        "",
        "1. Only consider pairs that are currently on the paper shortlist.",
        "2. The pair must still be in a promotable decision bucket.",
        "3. If the primary dYdX route is confirmed, prefer the dYdX paper route.",
        "4. If the dYdX route is not confirmed but Injective testnet spot supports both legs, queue the pair for the Injective spot lane.",
        "5. If neither route is available, keep the pair in watch mode and journal the blocker.",
        "",
        "## Hold Rules",
        "",
        "1. Hold open paper trades that are still on the shortlist and do not show a broken lifecycle state.",
        "2. Hold monitored trades when the venue route is still unavailable but the research case remains active.",
        "3. Keep blocked or queued trades in research mode until a route clears.",
        "",
        "## Close Rules",
        "",
        "1. Close a paper trade when the pair falls off the shortlist.",
        "2. Close a paper trade when its latest lifecycle state is partial or unconfirmed and the state persists into the next review window.",
        "3. Close a paper trade when a later journal event already marked the trade completed.",
        "",
        "## Current Limitations",
        "",
        "- This rulebook does not yet use live mark-to-market PnL for stop-loss or take-profit exits.",
        "- Until PnL snapshots are wired in, open/hold/close decisions are driven by shortlist status, route readiness, and journal state.",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def _safe_json_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _maybe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("%", "").replace(",", "")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _extract_dydx_reference_price(payload: object, market: str) -> float | None:
    if not isinstance(payload, dict):
        return None
    markets = payload.get("markets")
    if not isinstance(markets, dict):
        return None
    market_payload = markets.get(market)
    if not isinstance(market_payload, dict):
        return None
    for key in ("oraclePrice", "indexPrice", "price"):
        number = _maybe_float(market_payload.get(key))
        if number is not None and number > 0:
            return number
    return None


def _lookup_dydx_reference_prices(markets: set[str]) -> dict[str, dict[str, object]]:
    if not markets:
        return {}
    adapter = build_dydx_indexer_adapter(DydxNetworkConfig.paper_testnet_from_env())
    if adapter is None:
        return {}
    captured_at = datetime.now(timezone.utc).isoformat()
    prices: dict[str, dict[str, object]] = {}
    for market in sorted(markets):
        try:
            payload = adapter.market_data(market)
        except Exception:
            continue
        reference_price = _extract_dydx_reference_price(payload.get("payload"), market)
        if reference_price is None:
            continue
        prices[market] = {
            "price": reference_price,
            "source": "dydx_oracle_price",
            "captured_at_utc": captured_at,
        }
    return prices


def _match_pair_detail_price_snapshot(
    pair_detail: pd.DataFrame,
    *,
    pair: str,
    strategy_label: str = "",
    timeframe: str = "",
) -> dict[str, object]:
    def _clean_scalar(value: object) -> object:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return value

    if pair_detail.empty:
        return {}
    matched = pair_detail[pair_detail.get("pair", pd.Series(dtype=object)).astype(str) == str(pair)].copy()
    if matched.empty:
        return {}
    if strategy_label and "strategy_label" in matched.columns:
        exact = matched[matched["strategy_label"].astype(str) == str(strategy_label)].copy()
        if not exact.empty:
            matched = exact
    if timeframe and "timeframe" in matched.columns:
        exact_timeframe = matched[matched["timeframe"].astype(str) == str(timeframe)].copy()
        if not exact_timeframe.empty:
            matched = exact_timeframe
    sort_cols: list[str] = []
    if "detail_capture_timestamp_utc" in matched.columns:
        matched["_detail_capture_ts"] = pd.to_datetime(matched["detail_capture_timestamp_utc"], utc=True, errors="coerce")
        sort_cols.append("_detail_capture_ts")
    if sort_cols:
        matched = matched.sort_values(sort_cols, ascending=False, na_position="last")
    row = matched.iloc[0].to_dict()
    return {
        "wizard_price_x": _clean_scalar(row.get("price_x", "")),
        "wizard_price_y": _clean_scalar(row.get("price_y", "")),
        "wizard_price_capture_timestamp_utc": _clean_scalar(row.get("detail_capture_timestamp_utc", "")),
        "wizard_return_x_pct": _clean_scalar(row.get("return_x_pct", "")),
        "wizard_return_y_pct": _clean_scalar(row.get("return_y_pct", "")),
        "wizard_volume_x_top": _clean_scalar(row.get("volume_x_top", "")),
        "wizard_volume_y_top": _clean_scalar(row.get("volume_y_top", "")),
    }


def refresh_paper_trade_price_journal(root: Path = ROOT) -> Path:
    journal_path = root / "reports" / "paper_trading_journal.csv"
    if not journal_path.exists():
        return journal_path
    journal = _read_csv(journal_path)
    if journal.empty:
        return journal_path
    pair_detail_path = root / "reports" / "active" / "wizard_research_pair_detail_capture.csv"
    pair_detail = _read_csv_or_empty(pair_detail_path)
    if not pair_detail.empty and "journal_layer" in pair_detail.columns:
        pair_detail = pair_detail[pair_detail["journal_layer"].astype(str) == "pair_detail_capture"].copy()
    dydx_markets: set[str] = set()
    for _, row in journal.iterrows():
        if str(row.get("venue", "") or "dydx").strip().lower() != "dydx":
            continue
        pair = str(row.get("pair", "") or "")
        try:
            asset_x, asset_y = _split_pair(pair)
        except ValueError:
            continue
        dydx_markets.add(_venue_market(asset_x, "dydx"))
        dydx_markets.add(_venue_market(asset_y, "dydx"))
    dydx_reference_prices = _lookup_dydx_reference_prices(dydx_markets)

    changed = False
    for idx, row in journal.iterrows():
        row_changed = False
        pair = str(row.get("pair", "") or "")
        venue = str(row.get("venue", "") or "dydx")
        try:
            asset_x, asset_y = _split_pair(pair)
        except ValueError:
            continue
        strategy_label = ""
        timeframe = ""
        entry_snapshot = _safe_json_dict(row.get("entry_snapshot_json"))
        if entry_snapshot:
            strategy_label = str(entry_snapshot.get("dashboard_strategy", "") or "")
            timeframe = str(entry_snapshot.get("dashboard_timeframe", "") or "")
        detail_prices = _match_pair_detail_price_snapshot(
            pair_detail,
            pair=pair,
            strategy_label=strategy_label,
            timeframe=timeframe,
        )
        if detail_prices:
            for source_key, dest_key in (
                ("wizard_price_x", "wizard_entry_price_x"),
                ("wizard_price_y", "wizard_entry_price_y"),
                ("wizard_price_capture_timestamp_utc", "wizard_entry_price_capture_timestamp_utc"),
                ("wizard_return_x_pct", "wizard_entry_return_x_pct"),
                ("wizard_return_y_pct", "wizard_entry_return_y_pct"),
                ("wizard_volume_x_top", "wizard_entry_volume_x_top"),
                ("wizard_volume_y_top", "wizard_entry_volume_y_top"),
            ):
                if not str(entry_snapshot.get(dest_key, "")).strip() and str(detail_prices.get(source_key, "")).strip():
                    entry_snapshot[dest_key] = detail_prices[source_key]
                    row_changed = True
        if venue.strip().lower() == "dydx":
            left_market = _venue_market(asset_x, "dydx")
            right_market = _venue_market(asset_y, "dydx")
            left_reference = dydx_reference_prices.get(left_market, {})
            right_reference = dydx_reference_prices.get(right_market, {})
            if left_reference and not str(entry_snapshot.get("venue_entry_price_x", "")).strip():
                entry_snapshot["venue_entry_price_x"] = left_reference.get("price", "")
                entry_snapshot["venue_entry_price_source_x"] = left_reference.get("source", "")
                entry_snapshot["venue_entry_price_captured_at_utc_x"] = left_reference.get("captured_at_utc", "")
                row_changed = True
            if right_reference and not str(entry_snapshot.get("venue_entry_price_y", "")).strip():
                entry_snapshot["venue_entry_price_y"] = right_reference.get("price", "")
                entry_snapshot["venue_entry_price_source_y"] = right_reference.get("source", "")
                entry_snapshot["venue_entry_price_captured_at_utc_y"] = right_reference.get("captured_at_utc", "")
                row_changed = True
        if row_changed:
            journal.at[idx, "entry_snapshot_json"] = json.dumps(entry_snapshot, sort_keys=True)
            changed = True
        exit_snapshot = _safe_json_dict(row.get("exit_snapshot_json"))
        if not exit_snapshot:
            continue
        exit_row_changed = False
        exit_detail_prices = _match_pair_detail_price_snapshot(
            pair_detail,
            pair=pair,
            strategy_label=str(exit_snapshot.get("dashboard_strategy", "") or strategy_label),
            timeframe=str(exit_snapshot.get("dashboard_timeframe", "") or timeframe),
        )
        for source_key, dest_key in (
            ("wizard_price_x", "wizard_exit_price_x"),
            ("wizard_price_y", "wizard_exit_price_y"),
            ("wizard_price_capture_timestamp_utc", "wizard_exit_price_capture_timestamp_utc"),
        ):
            if exit_detail_prices and not str(exit_snapshot.get(dest_key, "")).strip() and str(exit_detail_prices.get(source_key, "")).strip():
                exit_snapshot[dest_key] = exit_detail_prices[source_key]
                exit_row_changed = True
        if venue.strip().lower() == "dydx":
            left_market = _venue_market(asset_x, "dydx")
            right_market = _venue_market(asset_y, "dydx")
            left_reference = dydx_reference_prices.get(left_market, {})
            right_reference = dydx_reference_prices.get(right_market, {})
            if left_reference and not str(exit_snapshot.get("venue_exit_price_x", "")).strip():
                exit_snapshot["venue_exit_price_x"] = left_reference.get("price", "")
                exit_snapshot["venue_exit_price_source_x"] = left_reference.get("source", "")
                exit_snapshot["venue_exit_price_captured_at_utc_x"] = left_reference.get("captured_at_utc", "")
                exit_row_changed = True
            if right_reference and not str(exit_snapshot.get("venue_exit_price_y", "")).strip():
                exit_snapshot["venue_exit_price_y"] = right_reference.get("price", "")
                exit_snapshot["venue_exit_price_source_y"] = right_reference.get("source", "")
                exit_snapshot["venue_exit_price_captured_at_utc_y"] = right_reference.get("captured_at_utc", "")
                exit_row_changed = True
        if exit_row_changed:
            journal.at[idx, "exit_snapshot_json"] = json.dumps(exit_snapshot, sort_keys=True)
            changed = True
    if changed:
        journal.to_csv(journal_path, index=False)
    return journal_path


def _monitor_timeframe_sort_key(value: object) -> tuple[int, str]:
    text = str(value or "").strip()
    order = {"Daily": 0, "4 Hour": 1, "1 Hour": 2, "5 Min": 3, "Live": 4}
    return order.get(text, 99), text


def _monitor_markdown(summary: pd.DataFrame, timeframe: pd.DataFrame) -> str:
    lines = [
        "# Live Paper Trade Monitor",
        "",
        "This report tracks paper trades from the Wizard dashboard side, even when venue execution evidence is incomplete.",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Open watch rows: {len(summary)}",
        f"- Timeframe detail rows: {len(timeframe)}",
        "",
    ]
    if not summary.empty:
        preview_cols = [
            "pair",
            "monitor_status",
            "recommendation",
            "latest_dashboard_timeframe",
            "latest_dashboard_strategy",
            "current_zscore_roll",
            "current_zscore_norm",
            "current_sharpe",
            "current_return_total",
            "current_max_drawdown",
        ]
        preview = summary[[col for col in preview_cols if col in summary.columns]].head(20)
        lines.extend(["## Open Monitor", "", "```text", preview.to_string(index=False), "```", ""])
    if not timeframe.empty:
        preview_cols = [
            "pair",
            "timeframe",
            "strategy_label",
            "capture_status",
            "readiness_label",
            "paper_candidate_status",
            "zscore_roll_value",
            "zscore_norm_value",
            "sharpe_top",
            "return_total_top",
        ]
        preview = timeframe[[col for col in preview_cols if col in timeframe.columns]].head(30)
        lines.extend(["## Timeframe Detail", "", "```text", preview.to_string(index=False), "```", ""])
    return "\n".join(lines)


def refresh_live_paper_trade_monitor(root: Path = ROOT) -> dict[str, Path]:
    journal_path = root / "reports" / "paper_trading_journal.csv"
    watch_path = refresh_current_paper_watch_positions(journal_path, root / "reports" / "active" / "current_paper_watch_positions.csv")
    decision_path = root / "reports" / "active" / "paper_trade_decision_report.csv"
    research_path = root / "reports" / "active" / "wizard_research_journal.csv"
    summary_path = root / "reports" / "active" / "live_paper_trade_monitor.csv"
    timeframe_path = root / "reports" / "active" / "live_paper_trade_timeframe_monitor.csv"
    md_path = root / "reports" / "active" / "live_paper_trade_monitor.md"

    watch = _read_csv_or_empty(watch_path)
    decisions = _read_csv_or_empty(decision_path)
    research = _read_csv_or_empty(research_path)

    summary_columns = [
        "trade_id",
        "pair",
        "strategy_id",
        "venue",
        "opened_timestamp_utc",
        "last_event_timestamp_utc",
        "lifecycle_status",
        "plan_status",
        "plan_reason",
        "monitor_status",
        "monitor_reason",
        "recommendation",
        "decision_reason",
        "latest_dashboard_capture_timestamp_utc",
        "latest_dashboard_timeframe",
        "latest_dashboard_strategy",
        "latest_readiness_label",
        "latest_paper_candidate_status",
        "current_submit_state",
        "current_zscore_source",
        "current_zscore_norm",
        "current_zscore_roll",
        "current_correlation",
        "current_hurst",
        "current_half_life",
        "current_return_total",
        "current_sharpe",
        "current_max_drawdown",
        "wizard_entry_price_x",
        "wizard_entry_price_y",
        "venue_entry_price_x",
        "venue_entry_price_y",
        "price_x",
        "price_y",
        "return_x_pct",
        "return_y_pct",
        "volume_x_top",
        "volume_y_top",
        "dependency_x_over_y_top",
        "dependency_y_over_x_top",
        "coint_johansen_top",
        "coint_engle_granger_top",
        "hedge_ratio_top",
        "lt_beta",
        "changed_since_last_capture",
        "entry_snapshot_json",
    ]
    timeframe_columns = [
        "trade_id",
        "pair",
        "venue",
        "opened_timestamp_utc",
        "timeframe",
        "strategy_label",
        "detail_capture_timestamp_utc",
        "capture_status",
        "readiness_label",
        "paper_candidate_status",
        "recommendation",
        "zscore_green_source",
        "zscore_norm_value",
        "zscore_roll_value",
        "correlation_top",
        "hurst_top",
        "half_life_top",
        "return_total_top",
        "sharpe_top",
        "max_drawdown_top",
        "price_x",
        "price_y",
        "return_x_pct",
        "return_y_pct",
        "volume_x_top",
        "volume_y_top",
        "dependency_x_over_y_top",
        "dependency_y_over_x_top",
        "coint_johansen_top",
        "coint_engle_granger_top",
        "hedge_ratio_top",
        "lt_beta",
        "changed_vs_last_capture",
    ]

    if watch.empty:
        pd.DataFrame(columns=summary_columns).to_csv(summary_path, index=False)
        pd.DataFrame(columns=timeframe_columns).to_csv(timeframe_path, index=False)
        md_path.write_text(_monitor_markdown(pd.DataFrame(columns=summary_columns), pd.DataFrame(columns=timeframe_columns)), encoding="utf-8")
        return {"summary": summary_path, "timeframe": timeframe_path, "markdown": md_path}

    detail = research.copy()
    if not detail.empty and "journal_layer" in detail.columns:
        detail = detail[detail["journal_layer"].astype(str) == "pair_detail_capture"].copy()
    if not detail.empty and "pair" in detail.columns:
        detail["pair"] = detail["pair"].astype(str)
    if not decisions.empty and "pair" in decisions.columns:
        decisions["pair"] = decisions["pair"].astype(str)

    summary_rows: list[dict[str, object]] = []
    timeframe_rows: list[dict[str, object]] = []

    for _, watch_row in watch.iterrows():
        pair = str(watch_row.get("pair", "") or "")
        trade_id = str(watch_row.get("trade_id", "") or "")
        venue = str(watch_row.get("venue", "") or "paper")
        pair_detail = detail[detail.get("pair", pd.Series(dtype=object)).astype(str) == pair].copy() if not detail.empty else pd.DataFrame()
        if not pair_detail.empty:
            pair_detail["_capture_ts"] = pd.to_datetime(
                pair_detail.get("detail_capture_timestamp_utc", pd.Series(dtype=object)),
                utc=True,
                errors="coerce",
            )
            pair_detail = pair_detail.sort_values(
                by=["_capture_ts", "timeframe", "strategy_label"],
                ascending=[False, True, True],
                key=lambda col: col.map(_monitor_timeframe_sort_key) if col.name == "timeframe" else col,
                na_position="last",
            )
        latest = pair_detail.iloc[0].to_dict() if not pair_detail.empty else {}
        decision_row = {}
        if not decisions.empty:
            matched = decisions[decisions["pair"].astype(str) == pair].copy()
            if not matched.empty:
                decision_row = matched.sort_values("decision_rank", na_position="last").iloc[0].to_dict()
        entry_snapshot = _safe_json_dict(watch_row.get("entry_snapshot_json"))
        if pair_detail.empty and entry_snapshot:
            monitor_status = "entry_snapshot_only"
            monitor_reason = "live monitor is using the saved dashboard snapshot until fresh pair-detail capture arrives"
        elif pair_detail.empty:
            monitor_status = "missing_from_dashboard"
            monitor_reason = "open trade is not present in the latest dashboard research capture"
        elif any(str(value or "").strip().lower() == "captured" for value in pair_detail.get("capture_status", pd.Series(dtype=object)).tolist()):
            monitor_status = "active_on_dashboard"
            monitor_reason = "latest dashboard capture is available for this watched pair"
        else:
            monitor_status = "needs_refresh"
            monitor_reason = "dashboard rows exist but the latest capture is incomplete"
        recommendation = str(decision_row.get("recommendation", "") or ("hold" if entry_snapshot else "hold"))
        decision_reason = str(decision_row.get("reason", "") or monitor_reason)
        if monitor_status == "entry_snapshot_only" and str(watch_row.get("plan_reason", "") or "") == "manual_dashboard_watch_seed":
            recommendation = "hold"
            decision_reason = "hold until the next hourly refresh replaces the seed snapshot with a fresh dashboard capture"
        changed = bool(
            not pair_detail.empty
            and (
                pair_detail.get("changed_vs_last_capture", pd.Series(dtype=bool)).fillna(False).astype(bool).any()
                or pair_detail.get("delta_zscore_norm", pd.Series(dtype=float)).fillna(0).astype(float).ne(0).any()
                or pair_detail.get("delta_zscore_roll", pd.Series(dtype=float)).fillna(0).astype(float).ne(0).any()
                or pair_detail.get("delta_return_total", pd.Series(dtype=float)).fillna(0).astype(float).ne(0).any()
                or pair_detail.get("delta_sharpe", pd.Series(dtype=float)).fillna(0).astype(float).ne(0).any()
            )
        )
        fallback_timeframe = str(entry_snapshot.get("dashboard_timeframe", "") or "Live") if entry_snapshot else ""
        fallback_strategy = str(entry_snapshot.get("dashboard_strategy", "") or "")

        summary_rows.append(
            {
                "trade_id": trade_id,
                "pair": pair,
                "strategy_id": watch_row.get("strategy_id", ""),
                "venue": venue,
                "opened_timestamp_utc": watch_row.get("opened_timestamp_utc", ""),
                "last_event_timestamp_utc": watch_row.get("last_event_timestamp_utc", ""),
                "lifecycle_status": watch_row.get("lifecycle_status", ""),
                "plan_status": watch_row.get("plan_status", ""),
                "plan_reason": watch_row.get("plan_reason", ""),
                "monitor_status": monitor_status,
                "monitor_reason": monitor_reason,
                "recommendation": recommendation,
                "decision_reason": decision_reason,
                "latest_dashboard_capture_timestamp_utc": latest.get("detail_capture_timestamp_utc", "") or watch_row.get("opened_timestamp_utc", ""),
                "latest_dashboard_timeframe": latest.get("timeframe", "") or fallback_timeframe,
                "latest_dashboard_strategy": latest.get("strategy_label", "") or fallback_strategy,
                "latest_readiness_label": latest.get("readiness_label", ""),
                "latest_paper_candidate_status": latest.get("paper_candidate_status", ""),
                "current_submit_state": decision_row.get("current_submit_state", ""),
                "current_zscore_source": latest.get("zscore_green_source", "") or entry_snapshot.get("dashboard_zscore_source", ""),
                "current_zscore_norm": latest.get("zscore_norm_value", "") or entry_snapshot.get("dashboard_zscore_norm", ""),
                "current_zscore_roll": latest.get("zscore_roll_value", "") or entry_snapshot.get("dashboard_zscore_roll", ""),
                "current_correlation": latest.get("correlation_top", "") or entry_snapshot.get("dashboard_corr", ""),
                "current_hurst": latest.get("hurst_top", "") or entry_snapshot.get("dashboard_hurst", ""),
                "current_half_life": latest.get("half_life_top", "") or entry_snapshot.get("dashboard_half_life", ""),
        "current_return_total": latest.get("return_total_top", "") or entry_snapshot.get("dashboard_return_pct", ""),
        "current_sharpe": latest.get("sharpe_top", "") or entry_snapshot.get("dashboard_sharpe", ""),
        "current_max_drawdown": latest.get("max_drawdown_top", "") or entry_snapshot.get("dashboard_mdd_pct", ""),
        "wizard_entry_price_x": entry_snapshot.get("wizard_entry_price_x", ""),
        "wizard_entry_price_y": entry_snapshot.get("wizard_entry_price_y", ""),
        "venue_entry_price_x": entry_snapshot.get("venue_entry_price_x", ""),
        "venue_entry_price_y": entry_snapshot.get("venue_entry_price_y", ""),
        "price_x": latest.get("price_x", "") or entry_snapshot.get("wizard_entry_price_x", "") or entry_snapshot.get("venue_entry_price_x", ""),
        "price_y": latest.get("price_y", "") or entry_snapshot.get("wizard_entry_price_y", "") or entry_snapshot.get("venue_entry_price_y", ""),
                "return_x_pct": latest.get("return_x_pct", ""),
                "return_y_pct": latest.get("return_y_pct", ""),
                "volume_x_top": latest.get("volume_x_top", ""),
                "volume_y_top": latest.get("volume_y_top", ""),
                "dependency_x_over_y_top": latest.get("dependency_x_over_y_top", ""),
                "dependency_y_over_x_top": latest.get("dependency_y_over_x_top", ""),
                "coint_johansen_top": latest.get("coint_johansen_top", ""),
                "coint_engle_granger_top": latest.get("coint_engle_granger_top", ""),
                "hedge_ratio_top": latest.get("hedge_ratio_top", ""),
                "lt_beta": latest.get("lt_beta", ""),
                "changed_since_last_capture": changed,
                "entry_snapshot_json": json.dumps(entry_snapshot, sort_keys=True),
            }
        )

        if pair_detail.empty and entry_snapshot:
            timeframe_rows.append(
                {
                    "trade_id": trade_id,
                    "pair": pair,
                    "venue": venue,
                    "opened_timestamp_utc": watch_row.get("opened_timestamp_utc", ""),
                    "timeframe": fallback_timeframe,
                    "strategy_label": fallback_strategy,
                    "detail_capture_timestamp_utc": watch_row.get("opened_timestamp_utc", ""),
                    "capture_status": "entry_snapshot_only",
                    "readiness_label": "",
                    "paper_candidate_status": "",
                    "recommendation": recommendation,
                    "zscore_green_source": entry_snapshot.get("dashboard_zscore_source", ""),
                    "zscore_norm_value": entry_snapshot.get("dashboard_zscore_norm", ""),
                    "zscore_roll_value": entry_snapshot.get("dashboard_zscore_roll", ""),
                    "correlation_top": entry_snapshot.get("dashboard_corr", ""),
                    "hurst_top": entry_snapshot.get("dashboard_hurst", ""),
                    "half_life_top": entry_snapshot.get("dashboard_half_life", ""),
                    "return_total_top": entry_snapshot.get("dashboard_return_pct", ""),
                    "sharpe_top": entry_snapshot.get("dashboard_sharpe", ""),
                    "max_drawdown_top": entry_snapshot.get("dashboard_mdd_pct", ""),
                    "price_x": entry_snapshot.get("wizard_entry_price_x", "") or entry_snapshot.get("venue_entry_price_x", ""),
                    "price_y": entry_snapshot.get("wizard_entry_price_y", "") or entry_snapshot.get("venue_entry_price_y", ""),
                    "return_x_pct": entry_snapshot.get("wizard_entry_return_x_pct", ""),
                    "return_y_pct": entry_snapshot.get("wizard_entry_return_y_pct", ""),
                    "volume_x_top": entry_snapshot.get("wizard_entry_volume_x_top", ""),
                    "volume_y_top": entry_snapshot.get("wizard_entry_volume_y_top", ""),
                    "dependency_x_over_y_top": entry_snapshot.get("dashboard_profile_x_over_y", ""),
                    "dependency_y_over_x_top": entry_snapshot.get("dashboard_profile_y_over_x", ""),
                    "coint_johansen_top": entry_snapshot.get("dashboard_johansen", ""),
                    "coint_engle_granger_top": entry_snapshot.get("dashboard_engle_granger", ""),
                    "hedge_ratio_top": "",
                    "lt_beta": "",
                    "changed_vs_last_capture": False,
                }
            )

        for _, detail_row in pair_detail.iterrows():
            timeframe_rows.append(
                {
                    "trade_id": trade_id,
                    "pair": pair,
                    "venue": venue,
                    "opened_timestamp_utc": watch_row.get("opened_timestamp_utc", ""),
                    "timeframe": detail_row.get("timeframe", ""),
                    "strategy_label": detail_row.get("strategy_label", ""),
                    "detail_capture_timestamp_utc": detail_row.get("detail_capture_timestamp_utc", ""),
                    "capture_status": detail_row.get("capture_status", ""),
                    "readiness_label": detail_row.get("readiness_label", ""),
                    "paper_candidate_status": detail_row.get("paper_candidate_status", ""),
                    "recommendation": recommendation,
                    "zscore_green_source": detail_row.get("zscore_green_source", ""),
                    "zscore_norm_value": detail_row.get("zscore_norm_value", ""),
                    "zscore_roll_value": detail_row.get("zscore_roll_value", ""),
                    "correlation_top": detail_row.get("correlation_top", ""),
                    "hurst_top": detail_row.get("hurst_top", ""),
                    "half_life_top": detail_row.get("half_life_top", ""),
                    "return_total_top": detail_row.get("return_total_top", ""),
                    "sharpe_top": detail_row.get("sharpe_top", ""),
                    "max_drawdown_top": detail_row.get("max_drawdown_top", ""),
                    "price_x": detail_row.get("price_x", ""),
                    "price_y": detail_row.get("price_y", ""),
                    "return_x_pct": detail_row.get("return_x_pct", ""),
                    "return_y_pct": detail_row.get("return_y_pct", ""),
                    "volume_x_top": detail_row.get("volume_x_top", ""),
                    "volume_y_top": detail_row.get("volume_y_top", ""),
                    "dependency_x_over_y_top": detail_row.get("dependency_x_over_y_top", ""),
                    "dependency_y_over_x_top": detail_row.get("dependency_y_over_x_top", ""),
                    "coint_johansen_top": detail_row.get("coint_johansen_top", ""),
                    "coint_engle_granger_top": detail_row.get("coint_engle_granger_top", ""),
                    "hedge_ratio_top": detail_row.get("hedge_ratio_top", ""),
                    "lt_beta": detail_row.get("lt_beta", ""),
                    "changed_vs_last_capture": detail_row.get("changed_vs_last_capture", ""),
                }
            )

    summary = pd.DataFrame(summary_rows).reindex(columns=summary_columns, fill_value="")
    timeframe = pd.DataFrame(timeframe_rows).reindex(columns=timeframe_columns, fill_value="")
    if not summary.empty:
        summary = summary.sort_values(by=["monitor_status", "opened_timestamp_utc", "pair"], ascending=[True, False, True], na_position="last")
    if not timeframe.empty:
        timeframe = timeframe.sort_values(
            by=["pair", "timeframe", "strategy_label", "detail_capture_timestamp_utc"],
            ascending=[True, True, True, False],
            key=lambda col: col.map(_monitor_timeframe_sort_key) if col.name == "timeframe" else col,
            na_position="last",
        )
    summary.to_csv(summary_path, index=False)
    timeframe.to_csv(timeframe_path, index=False)
    md_path.write_text(_monitor_markdown(summary, timeframe), encoding="utf-8")
    return {"summary": summary_path, "timeframe": timeframe_path, "markdown": md_path}


def refresh_paper_trade_decision_report(root: Path = ROOT) -> pd.DataFrame:
    journal_path = root / "reports" / "paper_trading_journal.csv"
    watch_path = refresh_current_paper_watch_positions(journal_path, root / "reports" / "active" / "current_paper_watch_positions.csv")
    refresh_paper_trade_rulebook(root / "reports" / "active" / "paper_trade_rulebook.md")

    from quant_platform.active_pipeline import paper_candidate_shortlist_rows

    shortlist = paper_candidate_shortlist_rows(root=root, max_pairs=10)
    preflight = _read_csv(root / "reports" / "paper_execution_preflight.csv")
    route_queue = _read_csv(root / "reports" / "active" / "non_eth_route_submit_queue.csv")
    injective_queue = _read_csv(root / "reports" / "active" / "injective_mirror_candidate_queue.csv")
    watch = _read_csv(watch_path)

    paper_gate_ready = bool(
        not preflight.empty
        and "ready" in preflight.columns
        and preflight["ready"].fillna(False).astype(bool).all()
    )

    route_by_pair = {}
    if not route_queue.empty and "pair" in route_queue.columns:
        route_by_pair = {
            str(row.get("pair", "")): row
            for _, row in route_queue.sort_values("submit_priority_rank" if "submit_priority_rank" in route_queue.columns else "pair").iterrows()
        }
    injective_by_pair = {}
    if not injective_queue.empty and "pair" in injective_queue.columns:
        injective_by_pair = {
            str(row.get("pair", "")): row
            for _, row in injective_queue.iterrows()
        }

    shortlist_pairs = set(shortlist.get("pair", pd.Series(dtype=object)).astype(str)) if not shortlist.empty else set()
    watch_pairs = set(watch.get("pair", pd.Series(dtype=object)).astype(str)) if not watch.empty else set()

    columns = [
        "decision_rank",
        "pair",
        "candidate_id",
        "venue",
        "recommendation",
        "priority",
        "reason",
        "current_submit_state",
        "injective_execution_mode",
        "lifecycle_status",
        "paper_gate_ready",
        "last_event_timestamp_utc",
        "evidence_path",
    ]
    rows: list[dict[str, object]] = []

    if not watch.empty:
        for _, row in watch.sort_values("last_event_timestamp_utc").iterrows():
            pair = str(row.get("pair", ""))
            lifecycle_status = str(row.get("lifecycle_status", "") or "")
            reason = "still_shortlisted_monitor"
            recommendation = "hold"
            priority = "medium"
            if pair not in shortlist_pairs:
                recommendation = "close"
                priority = "high"
                reason = "pair_left_shortlist"
            elif lifecycle_status in {"partial", "unconfirmed"}:
                recommendation = "close"
                priority = "high"
                reason = f"repair_or_close_lingering_{lifecycle_status}"
            rows.append(
                {
                    "decision_rank": 0,
                    "pair": pair,
                    "candidate_id": "",
                    "venue": str(row.get("venue", "paper") or "paper"),
                    "recommendation": recommendation,
                    "priority": priority,
                    "reason": reason,
                    "current_submit_state": "",
                    "injective_execution_mode": "",
                    "lifecycle_status": lifecycle_status,
                    "paper_gate_ready": paper_gate_ready,
                    "last_event_timestamp_utc": str(row.get("last_event_timestamp_utc", "") or ""),
                    "evidence_path": f"{watch_path};{journal_path}",
                }
            )

    if not shortlist.empty:
        for _, row in shortlist.sort_values("shortlist_rank" if "shortlist_rank" in shortlist.columns else "pair").iterrows():
            pair = str(row.get("pair", ""))
            if pair in watch_pairs:
                continue
            route_row = route_by_pair.get(pair)
            injective_row = injective_by_pair.get(pair)
            submit_state = str(route_row.get("current_submit_state", "") or "") if route_row is not None else ""
            injective_mode = str(injective_row.get("injective_execution_mode", "") or "") if injective_row is not None else ""
            recommendation = "watch"
            priority = "medium"
            reason = "route_not_ready_keep_under_review"
            venue = str(row.get("best_execution_venue", "") or "paper")
            if paper_gate_ready and submit_state == "ready_for_paper_submit":
                recommendation = "open"
                priority = "high"
                reason = "dydx_route_confirmed_for_paper"
                venue = "dydx"
            elif injective_mode == "injective_testnet_spot":
                recommendation = "open"
                priority = "high"
                reason = "injective_testnet_spot_supported"
                venue = "injective_spot"
            rows.append(
                {
                    "decision_rank": 0,
                    "pair": pair,
                    "candidate_id": str(row.get("candidate_id", "") or ""),
                    "venue": venue,
                    "recommendation": recommendation,
                    "priority": priority,
                    "reason": reason,
                    "current_submit_state": submit_state,
                    "injective_execution_mode": injective_mode,
                    "lifecycle_status": "not_open",
                    "paper_gate_ready": paper_gate_ready,
                    "last_event_timestamp_utc": "",
                    "evidence_path": ";".join(
                        [
                            str(root / "reports" / "paper_execution_preflight.csv"),
                            str(root / "reports" / "active" / "non_eth_route_submit_queue.csv"),
                            str(root / "reports" / "active" / "injective_mirror_candidate_queue.csv"),
                        ]
                    ),
                }
            )

    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        frame.to_csv(root / "reports" / "active" / "paper_trade_decision_report.csv", index=False)
        return frame

    priority_order = {"high": 0, "medium": 1, "low": 2}
    action_order = {"close": 0, "open": 1, "hold": 2, "watch": 3}
    frame["_priority_sort"] = frame["priority"].map(priority_order).fillna(9)
    frame["_action_sort"] = frame["recommendation"].map(action_order).fillna(9)
    frame = frame.sort_values(
        ["_priority_sort", "_action_sort", "pair"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    frame["decision_rank"] = frame.index + 1
    frame = frame.drop(columns=["_priority_sort", "_action_sort"])
    frame.to_csv(root / "reports" / "active" / "paper_trade_decision_report.csv", index=False)
    return frame


def _split_pair(pair: str) -> tuple[str, str]:
    text = str(pair or "").strip().upper()
    if "/" in text:
        left, right = [part.strip() for part in text.split("/", 1)]
        if left and right:
            return left, right
    normalized = text.replace("/", "-")
    parts = [part for part in normalized.split("-") if part]
    if len(parts) == 2:
        return parts[0], parts[1]
    if len(parts) == 4 and parts[1] == "USD" and parts[3] == "USD":
        return f"{parts[0]}-USD", f"{parts[2]}-USD"
    raise ValueError(f"pair must have two assets separated by '-' or '/': {pair}")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame()


def _dydx_market(asset: str) -> str:
    if asset.endswith("-USD"):
        return asset
    return f"{asset}-USD"


def _venue_market(asset: str, venue: str) -> str:
    normalized = _dydx_market(asset)
    if (venue or "").lower() != "dydx":
        return normalized
    return normalized


class UnsupportedVenueExecution:
    """Temporary execution adapter for venues that are recognized but not yet wired."""

    def __init__(self, venue: str) -> None:
        self.venue = venue or "unknown"

    def market_data(self, market: str) -> dict:
        return {"market": market, "status": "unsupported_venue", "venue": self.venue}

    def place_order(self, intent: OrderIntent) -> FillReport:
        return FillReport(
            order_id=f"{self.venue}-unsupported-paper",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_venue_not_supported",
        )

    def positions(self) -> list[dict]:
        return []

    def funding(self, market: str) -> dict:
        return {"market": market, "status": "unsupported_venue", "venue": self.venue}
