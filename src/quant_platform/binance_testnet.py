from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from quant_platform.execution import ExecutionMode, FillReport, OrderIntent


ROOT = Path(__file__).resolve().parents[2]
BINANCE_SPOT_TESTNET_URL = "https://testnet.binance.vision"
BINANCE_USDM_TESTNET_URL = "https://testnet.binancefuture.com"
BINANCE_TESTNET_PAIR_PREFLIGHT_REPORT = ROOT / "reports" / "active" / "binance_testnet_pair_preflight.csv"
BINANCE_TESTNET_PAIR_EXECUTION_JOURNAL = ROOT / "reports" / "paper_trading_journal_binance_testnet.csv"


@dataclass(frozen=True)
class BinancePairExecutionResult:
    lane: str
    status: str
    reason: str
    fills: tuple[FillReport, ...] = ()
    rollback_fills: tuple[FillReport, ...] = ()


@dataclass(frozen=True)
class BinanceTestnetConfig:
    """Credentials and endpoints for one explicitly non-production Binance lane."""

    lane: str
    mode: ExecutionMode = ExecutionMode.PAPER
    base_url: str = ""
    api_key: str | None = None
    api_secret: str | None = None
    submit_orders: bool = False
    recv_window_ms: int = 5_000

    @classmethod
    def spot_testnet(cls) -> "BinanceTestnetConfig":
        return cls(lane="spot", base_url=BINANCE_SPOT_TESTNET_URL)

    @classmethod
    def usdm_testnet(cls) -> "BinanceTestnetConfig":
        return cls(lane="usdm_futures", base_url=BINANCE_USDM_TESTNET_URL)

    @classmethod
    def spot_testnet_from_env(cls) -> "BinanceTestnetConfig":
        base = cls.spot_testnet()
        return cls(
            lane=base.lane,
            mode=base.mode,
            base_url=os.getenv("BINANCE_SPOT_TESTNET_BASE_URL", base.base_url).strip(),
            api_key=os.getenv("BINANCE_SPOT_TESTNET_API_KEY") or None,
            api_secret=os.getenv("BINANCE_SPOT_TESTNET_API_SECRET") or None,
            submit_orders=_env_truthy("BINANCE_SPOT_TESTNET_SUBMIT_ORDERS"),
            recv_window_ms=_env_int("BINANCE_SPOT_TESTNET_RECV_WINDOW_MS", base.recv_window_ms),
        )

    @classmethod
    def usdm_testnet_from_env(cls) -> "BinanceTestnetConfig":
        base = cls.usdm_testnet()
        return cls(
            lane=base.lane,
            mode=base.mode,
            base_url=os.getenv("BINANCE_USDM_TESTNET_BASE_URL", base.base_url).strip(),
            api_key=os.getenv("BINANCE_USDM_TESTNET_API_KEY") or None,
            api_secret=os.getenv("BINANCE_USDM_TESTNET_API_SECRET") or None,
            submit_orders=_env_truthy("BINANCE_USDM_TESTNET_SUBMIT_ORDERS"),
            recv_window_ms=_env_int("BINANCE_USDM_TESTNET_RECV_WINDOW_MS", base.recv_window_ms),
        )

    def paper_trading_blockers(self) -> list[str]:
        blockers: list[str] = []
        if self.mode != ExecutionMode.PAPER:
            blockers.append("mode_not_paper")
        if not _is_testnet_url(self.base_url, self.lane):
            blockers.append("unsafe_non_testnet_base_url")
        if not self.submit_orders:
            blockers.append("submit_orders_false")
        if not self.api_key:
            blockers.append("missing_api_key")
        if not self.api_secret:
            blockers.append("missing_api_secret")
        return blockers


class _BinanceTestnetOrderAdapter:
    exchange_submission_capable = True
    record_only = False
    lane = ""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()

    def place_order(self, intent: OrderIntent, config: BinanceTestnetConfig | object | None = None) -> FillReport:
        resolved = self._resolve_config(config)
        blockers = resolved.paper_trading_blockers()
        if blockers:
            return FillReport(
                order_id="binance-testnet-blocked",
                market=intent.market,
                side=intent.side,
                size=float(intent.size),
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status=f"paper_blocked_{';'.join(blockers)}",
            )
        if float(intent.size) <= 0:
            return FillReport(
                order_id="binance-testnet-invalid-size",
                market=intent.market,
                side=intent.side,
                size=float(intent.size),
                avg_price=float(intent.limit_price or 0.0),
                fee=0.0,
                slippage_bps=0.0,
                status="paper_blocked_invalid_size",
            )
        payload = self._submit(intent, resolved)
        return FillReport(
            order_id=str(payload.get("orderId") or payload.get("clientOrderId") or "binance-testnet-submitted"),
            market=intent.market,
            side=intent.side,
            size=float(payload.get("origQty") or intent.size),
            avg_price=float(payload.get("avgPrice") or intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )

    def _resolve_config(self, config: BinanceTestnetConfig | object | None) -> BinanceTestnetConfig:
        if isinstance(config, BinanceTestnetConfig) and config.lane == self.lane:
            return config
        if self.lane == "spot":
            return BinanceTestnetConfig.spot_testnet_from_env()
        return BinanceTestnetConfig.usdm_testnet_from_env()

    def _submit(self, intent: OrderIntent, config: BinanceTestnetConfig) -> dict[str, Any]:
        params = self._order_params(intent, config)
        params["timestamp"] = int(datetime.now(timezone.utc).timestamp() * 1000)
        params["recvWindow"] = int(config.recv_window_ms)
        encoded = urlencode(params)
        params["signature"] = hmac.new(
            str(config.api_secret).encode("utf-8"),
            encoded.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        response = self.session.post(
            f"{config.base_url.rstrip('/')}{self._order_path()}",
            params=params,
            headers={"X-MBX-APIKEY": str(config.api_key)},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected Binance testnet order response")
        return payload

    def _order_params(self, intent: OrderIntent, config: BinanceTestnetConfig) -> dict[str, object]:
        params: dict[str, object] = {
            "symbol": _normalize_market(intent.market, lane=config.lane),
            "side": str(intent.side).upper(),
            "type": "LIMIT" if intent.limit_price is not None else "MARKET",
            "quantity": _decimal_text(intent.size),
        }
        if intent.limit_price is not None:
            params["price"] = _decimal_text(intent.limit_price)
            params["timeInForce"] = "GTC"
        if intent.reduce_only and self.lane == "usdm_futures":
            params["reduceOnly"] = "true"
        return params

    def _order_path(self) -> str:
        raise NotImplementedError


class BinanceSpotTestnetOrderAdapter(_BinanceTestnetOrderAdapter):
    """Signed Spot Testnet adapter. It cannot access production endpoints."""

    lane = "spot"

    def _order_path(self) -> str:
        return "/api/v3/order"


class BinanceUsdmTestnetOrderAdapter(_BinanceTestnetOrderAdapter):
    """Signed USD-M Futures Testnet adapter. It cannot access production endpoints."""

    lane = "usdm_futures"

    def _order_path(self) -> str:
        return "/fapi/v1/order"


def binance_testnet_preflight(
    *,
    root: Path = ROOT,
    session: requests.Session | None = None,
    probe_network: bool = True,
) -> pd.DataFrame:
    """Write non-secret readiness rows for both Binance non-production lanes."""

    http = session or requests.Session()
    rows: list[dict[str, object]] = []
    for config in [BinanceTestnetConfig.spot_testnet_from_env(), BinanceTestnetConfig.usdm_testnet_from_env()]:
        network_status = "not_checked"
        network_error = ""
        if probe_network:
            try:
                response = http.get(f"{config.base_url.rstrip('/')}{_time_path(config.lane)}", timeout=15)
                network_status = "reachable" if response.ok else f"http_{response.status_code}"
            except requests.RequestException as exc:
                network_status = "unreachable"
                network_error = type(exc).__name__
        blockers = config.paper_trading_blockers()
        if network_status != "reachable":
            blockers.append(f"network_{network_status}")
        rows.append(
            {
                "lane": config.lane,
                "mode": config.mode.value,
                "base_url": config.base_url,
                "testnet_base_url": _is_testnet_url(config.base_url, config.lane),
                "api_key_present": bool(config.api_key),
                "api_secret_present": bool(config.api_secret),
                "submit_orders": bool(config.submit_orders),
                "network_status": network_status,
                "network_error": network_error,
                "ready_for_testnet_submit": not blockers,
                "blockers": ";".join(blockers),
                "next_action": _next_action(blockers),
                "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
    frame = pd.DataFrame(rows)
    output_dir = root / "reports" / "active"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "binance_testnet_preflight.csv", index=False)
    lines = ["# Binance Testnet Preflight", "", "Testnet and demo credentials are isolated from production keys.", ""]
    lines.append(frame.to_markdown(index=False))
    (output_dir / "binance_testnet_preflight.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def binance_testnet_pair_preflight(
    *,
    asset_x: str,
    asset_y: str,
    config: BinanceTestnetConfig,
    root: Path = ROOT,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Validate both legs against the selected Binance non-production venue."""

    http = session or requests.Session()
    symbols = [_normalize_market(asset_x, lane=config.lane), _normalize_market(asset_y, lane=config.lane)]
    shared_blockers = _preflight_blockers_without_submission(config)
    exchange_info: dict[str, Any] = {}
    fetch_error = ""
    try:
        response = http.get(f"{config.base_url.rstrip('/')}{_exchange_info_path(config.lane)}", timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected Binance exchangeInfo response")
        exchange_info = payload
    except (requests.RequestException, ValueError) as exc:
        fetch_error = type(exc).__name__

    symbol_index = {
        str(item.get("symbol", "")).upper(): item
        for item in exchange_info.get("symbols", [])
        if isinstance(item, dict)
    }
    rows: list[dict[str, object]] = []
    checked_at = datetime.now(timezone.utc).isoformat()
    for symbol in symbols:
        market = symbol_index.get(symbol, {})
        filters = {
            str(item.get("filterType", "")).upper(): item
            for item in market.get("filters", [])
            if isinstance(item, dict)
        }
        blockers = list(shared_blockers)
        if fetch_error:
            blockers.append(f"exchange_info_fetch_failed:{fetch_error}")
        if not market and not fetch_error:
            blockers.append("symbol_not_found_on_testnet")
        status = str(market.get("status") or market.get("contractStatus") or "").upper()
        if market and status not in {"TRADING"}:
            blockers.append(f"symbol_not_trading:{status or 'unknown'}")
        lot = filters.get("LOT_SIZE", {})
        price = filters.get("PRICE_FILTER", {})
        notional = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
        rows.append(
            {
                "lane": config.lane,
                "symbol": symbol,
                "base_url": config.base_url,
                "symbol_status": status,
                "tradable": bool(market) and status == "TRADING" and not blockers,
                "tick_size": str(price.get("tickSize", "")),
                "step_size": str(lot.get("stepSize", "")),
                "min_qty": str(lot.get("minQty", "")),
                "min_notional": str(notional.get("minNotional", notional.get("notional", ""))),
                "testnet_submit_enabled": bool(config.submit_orders),
                "api_key_present": bool(config.api_key),
                "api_secret_present": bool(config.api_secret),
                "blockers": ";".join(blockers),
                "next_action": _pair_preflight_next_action(blockers),
                "checked_at_utc": checked_at,
            }
        )
    frame = pd.DataFrame(rows)
    output = root / "reports" / "active" / "binance_testnet_pair_preflight.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def execute_binance_testnet_pair(
    *,
    intents: tuple[OrderIntent, OrderIntent],
    config: BinanceTestnetConfig,
    adapter: _BinanceTestnetOrderAdapter | None = None,
    journal_path: Path = BINANCE_TESTNET_PAIR_EXECUTION_JOURNAL,
) -> BinancePairExecutionResult:
    """Submit a two-leg testnet plan and attempt to flatten any confirmed first leg on failure.

    This function is intentionally not exposed through the CLI. Calling it requires
    a dedicated testnet config with ``submit_orders=True`` supplied by application code.
    """

    if len(intents) != 2:
        raise ValueError("Binance testnet pair execution requires exactly two intents")
    client = adapter or (
        BinanceSpotTestnetOrderAdapter() if config.lane == "spot" else BinanceUsdmTestnetOrderAdapter()
    )
    fills: list[FillReport] = []
    rollback_fills: list[FillReport] = []
    for intent in intents:
        fill = client.place_order(intent, config)
        fills.append(fill)
        if str(fill.status) == "paper_submitted":
            continue
        if len(fills) == 1:
            result = BinancePairExecutionResult(
                lane=config.lane,
                status="blocked_before_first_leg",
                reason=str(fill.status),
                fills=tuple(fills),
            )
            _append_pair_execution_result(result, intents=intents, journal_path=journal_path)
            return result
        first_intent = intents[0]
        rollback = client.place_order(_reverse_intent(first_intent, lane=config.lane), config)
        rollback_fills.append(rollback)
        result = BinancePairExecutionResult(
            lane=config.lane,
            status="second_leg_failed_rollback_attempted",
            reason=str(fill.status),
            fills=tuple(fills),
            rollback_fills=tuple(rollback_fills),
        )
        _append_pair_execution_result(result, intents=intents, journal_path=journal_path)
        return result
    result = BinancePairExecutionResult(
        lane=config.lane,
        status="pair_submitted",
        reason="both_legs_submitted",
        fills=tuple(fills),
    )
    _append_pair_execution_result(result, intents=intents, journal_path=journal_path)
    return result


def _append_pair_execution_result(
    result: BinancePairExecutionResult,
    *,
    intents: tuple[OrderIntent, OrderIntent],
    journal_path: Path,
) -> None:
    row = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "lane": result.lane,
                "status": result.status,
                "reason": result.reason,
                "intents_json": json.dumps([intent.__dict__ for intent in intents], sort_keys=True),
                "fills_json": json.dumps([fill.__dict__ for fill in result.fills], sort_keys=True),
                "rollback_fills_json": json.dumps([fill.__dict__ for fill in result.rollback_fills], sort_keys=True),
            }
        ]
    )
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(journal_path, mode="a", header=not journal_path.exists(), index=False)


def _reverse_intent(intent: OrderIntent, *, lane: str) -> OrderIntent:
    side = "SELL" if str(intent.side).upper() == "BUY" else "BUY"
    return OrderIntent(
        market=intent.market,
        side=side,
        size=float(intent.size),
        limit_price=intent.limit_price,
        reduce_only=lane == "usdm_futures",
    )


def _preflight_blockers_without_submission(config: BinanceTestnetConfig) -> list[str]:
    blockers: list[str] = []
    if config.mode != ExecutionMode.PAPER:
        blockers.append("mode_not_paper")
    if not _is_testnet_url(config.base_url, config.lane):
        blockers.append("unsafe_non_testnet_base_url")
    return blockers


def _exchange_info_path(lane: str) -> str:
    return "/api/v3/exchangeInfo" if lane == "spot" else "/fapi/v1/exchangeInfo"


def _pair_preflight_next_action(blockers: list[str]) -> str:
    if not blockers:
        return "validate_order_size_against_tick_lot_and_notional_rules"
    if "symbol_not_found_on_testnet" in blockers:
        return "choose_a_pair_with_both_legs_listed_on_this_testnet"
    if any(blocker.startswith("symbol_not_trading") for blocker in blockers):
        return "wait_for_or_select_trading_symbols"
    if any(blocker.startswith("exchange_info_fetch_failed") for blocker in blockers):
        return "resolve_testnet_network_access"
    return "resolve_testnet_configuration_blockers"


def _time_path(lane: str) -> str:
    return "/api/v3/time" if lane == "spot" else "/fapi/v1/time"


def _normalize_market(market: str, *, lane: str | None = None) -> str:
    """Map dashboard-style symbols to the contract symbols used by a Binance lane."""

    normalized = "".join(character for character in str(market).upper() if character.isalnum())
    if lane == "usdm_futures" and normalized.endswith("USD") and not normalized.endswith("USDT"):
        return f"{normalized}T"
    return normalized


def _decimal_text(value: float) -> str:
    return format(float(value), ".12f").rstrip("0").rstrip(".")


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _is_testnet_url(base_url: str, lane: str) -> bool:
    value = str(base_url or "").strip().lower()
    allowed = {"https://testnet.binance.vision"} if lane == "spot" else {"https://testnet.binancefuture.com"}
    return value.rstrip("/") in allowed


def _next_action(blockers: list[str]) -> str:
    if not blockers:
        return "run_controlled_testnet_smoke_order"
    if any(blocker.startswith("unsafe_non_testnet") for blocker in blockers):
        return "restore_official_testnet_base_url"
    if "missing_api_key" in blockers or "missing_api_secret" in blockers:
        return "configure_dedicated_testnet_credentials"
    if "submit_orders_false" in blockers:
        return "keep_submission_disabled_until_smoke_test_is_approved"
    if any(blocker.startswith("network_") for blocker in blockers):
        return "resolve_testnet_network_access"
    return "resolve_testnet_preflight_blockers"
