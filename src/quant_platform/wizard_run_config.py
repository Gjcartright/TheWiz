from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re


WIZARD_CONFIG_SCHEMA_VERSION = "wizard_run_config.v1"


@dataclass(frozen=True)
class WizardRunConfiguration:
    """Versioned identity for one Wizard observation, backtest, or discovery request."""

    source: str
    symbol_1: str | None = None
    symbol_2: str | None = None
    wizard_exchange: str | None = None
    interval: str | None = None
    period: int | None = None
    strategy: str | None = None
    spread_type: str | None = None
    exact_mode: str | None = None
    roll_window: int | None = None
    with_history: bool | None = None
    priority: str | None = None
    entry_level: float | None = None
    exit_level: float | None = None
    entry_long_operator: str | None = None
    entry_long_level: float | None = None
    entry_short_operator: str | None = None
    entry_short_level: float | None = None
    exit_long_operator: str | None = None
    exit_long_level: float | None = None
    exit_short_operator: str | None = None
    exit_short_level: float | None = None
    copula_entry_lower: float | None = None
    copula_entry_upper: float | None = None
    copula_exit_lower: float | None = None
    copula_exit_upper: float | None = None
    forced_close_mode: str | None = None
    exit_n_periods: int | None = None
    stop_loss_rate: float | None = None
    ecm_deviation_min: float | None = None
    correlation_strength_min: float | None = None
    commission_rate: float | None = None
    slippage_rate: float | None = None
    x_weighting: float | None = None
    y_weighting: float | None = None
    capital_weighting: str | None = None
    metric_mode: str | None = None
    simulation_runs: int | None = None
    live_update: str | None = None
    input_data_hash: str | None = None
    api_version: str = "v1beta"
    schema_version: str = WIZARD_CONFIG_SCHEMA_VERSION

    def canonical_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["source"] = canonical_token(self.source)
        values["symbol_1"] = canonical_symbol(self.symbol_1)
        values["symbol_2"] = canonical_symbol(self.symbol_2)
        values["wizard_exchange"] = canonical_wizard_exchange(self.wizard_exchange)
        values["interval"] = canonical_wizard_interval(self.interval)
        values["strategy"] = canonical_wizard_strategy(self.strategy)
        values["spread_type"] = canonical_wizard_spread_type(self.spread_type)
        values["priority"] = canonical_token(self.priority)
        values["exact_mode"] = canonical_token(
            self.exact_mode
            or canonical_exact_mode(
                strategy=values["strategy"],
                spread_type=values["spread_type"],
            )
        )
        for key in (
            "entry_long_operator",
            "entry_short_operator",
            "exit_long_operator",
            "exit_short_operator",
            "forced_close_mode",
            "capital_weighting",
            "metric_mode",
            "live_update",
            "api_version",
            "schema_version",
        ):
            values[key] = canonical_token(values[key])
        values["input_data_hash"] = canonical_hash(self.input_data_hash)
        return values

    @property
    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @property
    def config_hash(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()


def canonical_exact_mode(*, strategy: object, spread_type: object) -> str:
    strategy_token = canonical_wizard_strategy(strategy)
    spread_token = canonical_wizard_spread_type(spread_type)
    if strategy_token == "copula":
        return "copula"
    if strategy_token == "zscore_roll":
        return f"{spread_token or 'all'}_zscore_r"
    if strategy_token == "spread":
        return f"{spread_token or 'all'}_spread"
    if strategy_token:
        return f"{spread_token or 'all'}_{strategy_token}"
    return f"{spread_token}_spread" if spread_token else "unknown"


def canonical_token(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or None


def canonical_symbol(value: object) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", "", str(value).strip().upper())
    return text or None


def canonical_hash(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def canonical_wizard_exchange(value: object) -> str | None:
    if value is None:
        return None
    return canonical_token(api_wizard_exchange(value))


def canonical_wizard_interval(value: object) -> str | None:
    if value is None:
        return None
    token = normalized_key(value)
    mapping = {
        "daily": "daily",
        "1d": "daily",
        "hourly": "hourly",
        "1hour": "hourly",
        "1h": "hourly",
        "4hour": "4_hour",
        "4h": "4_hour",
        "5min": "5_min",
        "min5": "5_min",
        "5m": "5_min",
    }
    return mapping.get(token, canonical_token(value))


def canonical_wizard_strategy(value: object) -> str | None:
    if value is None:
        return None
    token = normalized_key(value)
    mapping = {
        "spread": "spread",
        "zscoreroll": "zscore_roll",
        "zscorer": "zscore_roll",
        "copula": "copula",
        "zscores": "zscores",
    }
    return mapping.get(token, canonical_token(value))


def canonical_wizard_spread_type(value: object) -> str | None:
    if value is None or not str(value).strip():
        return None
    token = canonical_token(value)
    return {"dyn": "dynamic", "ols": "static"}.get(str(token), token)


def api_wizard_exchange(value: object) -> str:
    token = normalized_key(value)
    mapping = {
        "binance": "Binance",
        "binanceus": "BinanceUs",
        "bybit": "ByBit",
        "coinbase": "Coinbase",
        "dydx": "Dydx",
    }
    if token not in mapping:
        raise ValueError(f"unsupported Crypto Wizards crypto exchange: {value}")
    return mapping[token]


def api_wizard_discovery_interval(value: object) -> str:
    token = normalized_key(value)
    mapping = {
        "daily": "Daily",
        "1d": "Daily",
        "hourly": "Hourly",
        "1hour": "Hourly",
        "1h": "Hourly",
    }
    if token not in mapping:
        raise ValueError(f"unsupported Crypto Wizards prescanned interval: {value}")
    return mapping[token]


def api_wizard_strategy(value: object) -> str:
    token = normalized_key(value)
    mapping = {
        "spread": "Spread",
        "zscoreroll": "ZScoreRoll",
        "zscorer": "ZScoreRoll",
        "copula": "Copula",
        "zscores": "ZScores",
    }
    if token not in mapping:
        raise ValueError(f"unsupported Crypto Wizards strategy: {value}")
    return mapping[token]


def normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
