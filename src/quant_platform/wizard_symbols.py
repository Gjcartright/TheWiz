from __future__ import annotations

from dataclasses import dataclass
import re


USD_QUOTES = {"USD", "USDT", "USDC"}
KNOWN_QUOTES = ("USDT", "USDC", "USD", "PERP", "BTC", "ETH", "EUR", "GBP")
WIZARD_EXCHANGE_ALIASES = {
    "binance": "binance",
    "binance us": "binanceus",
    "binanceus": "binanceus",
    "bybit": "bybit",
    "by bit": "bybit",
    "coinbase": "coinbase",
    "dydx": "dydx",
    "dyd": "dydx",
    "forex": "forex",
    "stocks": "stocks",
    "stocks/other": "stocks",
}
WIZARD_RESEARCH_ONLY_EXCHANGES = {"binance", "binanceus", "bybit", "coinbase"}


@dataclass(frozen=True)
class WizardSymbol:
    raw_symbol: str | None
    normalized_symbol: str | None
    base_asset: str | None
    quote_asset: str | None
    canonical_usd_symbol: str | None
    symbol_format: str


def normalize_wizard_exchange(value: object | None, default: str | None = None) -> str | None:
    text = str(value or default or "").strip()
    if not text:
        return None
    lowered = re.sub(r"\s+", " ", text).lower()
    return WIZARD_EXCHANGE_ALIASES.get(lowered, lowered.replace(" ", ""))


def normalize_wizard_symbol(symbol: object | None, wizard_exchange: object | None = None) -> WizardSymbol:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return WizardSymbol(None, None, None, None, None, "missing")

    cleaned = re.sub(r"[^A-Z0-9/-]", "", raw)
    if "-" in cleaned:
        parts = [part for part in cleaned.split("-") if part]
        if len(parts) >= 2:
            base, quote = parts[0], parts[1]
            return WizardSymbol(
                raw_symbol=cleaned,
                normalized_symbol=f"{base}-{quote}",
                base_asset=base,
                quote_asset=quote,
                canonical_usd_symbol=_canonical_usd_symbol(base, quote),
                symbol_format="hyphenated",
            )

    for quote in KNOWN_QUOTES:
        if cleaned.endswith(quote) and len(cleaned) > len(quote):
            base = cleaned[: -len(quote)]
            return WizardSymbol(
                raw_symbol=cleaned,
                normalized_symbol=f"{base}-{quote}",
                base_asset=base,
                quote_asset=quote,
                canonical_usd_symbol=_canonical_usd_symbol(base, quote),
                symbol_format="concatenated",
            )

    return WizardSymbol(
        raw_symbol=cleaned,
        normalized_symbol=cleaned or None,
        base_asset=cleaned or None,
        quote_asset=None,
        canonical_usd_symbol=None,
        symbol_format="unknown",
    )


def wizard_exchange_lane(wizard_exchange: object | None) -> str:
    exchange = normalize_wizard_exchange(wizard_exchange, default="dydx")
    if exchange == "dydx":
        return "dydx_discovery_lane"
    if exchange in WIZARD_RESEARCH_ONLY_EXCHANGES:
        return f"{exchange}_research_lane"
    if exchange in {"forex", "stocks"}:
        return f"{exchange}_out_of_scope_lane"
    return "unknown_wizard_exchange_lane"


def wizard_exchange_promotion_allowed(wizard_exchange: object | None) -> bool:
    return normalize_wizard_exchange(wizard_exchange, default="dydx") == "dydx"


def _canonical_usd_symbol(base: str | None, quote: str | None) -> str | None:
    if not base or quote not in USD_QUOTES:
        return None
    return f"{base}-USD"
