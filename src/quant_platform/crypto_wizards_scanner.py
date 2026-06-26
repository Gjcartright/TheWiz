from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json
import re

import pandas as pd

from quant_platform.fixture_ingestion import snake_case
from quant_platform.wizard_symbols import (
    normalize_wizard_exchange,
    normalize_wizard_symbol,
    wizard_exchange_lane,
    wizard_exchange_promotion_allowed,
)


SCANNER_COLUMNS = [
    "source_path",
    "source_url",
    "captured_at",
    "scanner_priority",
    "scanner_count_filter",
    "scanner_correlation_filter",
    "scanner_hurst_filter",
    "scanner_half_life_filter",
    "scanner_copula_filter",
    "scanner_strategy_filter",
    "scanner_symbol_filter",
    "scanner_exchange",
    "wizard_exchange",
    "wizard_exchange_lane",
    "wizard_promotion_allowed",
    "row_index",
    "pair",
    "asset_x",
    "asset_y",
    "asset_x_raw",
    "asset_y_raw",
    "asset_x_normalized",
    "asset_y_normalized",
    "asset_x_base",
    "asset_y_base",
    "asset_x_quote",
    "asset_y_quote",
    "asset_x_canonical_usd",
    "asset_y_canonical_usd",
    "normalized_pair",
    "pair_id",
    "volume_x",
    "volume_y",
    "min_volume",
    "spread_snapshot",
    "updated_at",
    "strategy_label",
    "zscore_norm",
    "zscore_roll",
    "dependency_profile",
    "dependency_x_over_y",
    "dependency_y_over_x",
    "correlation",
    "jn_flag",
    "eg_flag",
    "hurst",
    "half_life",
    "sigma_0_count",
    "sigma_1_count",
    "sigma_2_count",
    "var",
    "cvar",
    "mdd",
    "return_total",
    "sharpe",
    "raw_pair_cell",
    "raw_volume_cell",
    "raw_spread_cell",
    "raw_updated_cell",
    "raw_strategy_cell",
    "raw_zscore_cell",
    "raw_dependency_cell",
    "raw_stationarity_cell",
    "raw_risk_cell",
    "raw_reward_cell",
]


@dataclass(frozen=True)
class CryptoWizardsScannerRow:
    source_path: str | None = None
    source_url: str | None = None
    captured_at: str | None = None
    scanner_priority: str | None = None
    scanner_count_filter: str | None = None
    scanner_correlation_filter: str | None = None
    scanner_hurst_filter: str | None = None
    scanner_half_life_filter: str | None = None
    scanner_copula_filter: str | None = None
    scanner_strategy_filter: str | None = None
    scanner_symbol_filter: str | None = None
    scanner_exchange: str | None = None
    wizard_exchange: str | None = None
    wizard_exchange_lane: str | None = None
    wizard_promotion_allowed: bool | None = None
    row_index: int | None = None
    pair: str | None = None
    asset_x: str | None = None
    asset_y: str | None = None
    asset_x_raw: str | None = None
    asset_y_raw: str | None = None
    asset_x_normalized: str | None = None
    asset_y_normalized: str | None = None
    asset_x_base: str | None = None
    asset_y_base: str | None = None
    asset_x_quote: str | None = None
    asset_y_quote: str | None = None
    asset_x_canonical_usd: str | None = None
    asset_y_canonical_usd: str | None = None
    normalized_pair: str | None = None
    pair_id: str | None = None
    volume_x: float | None = None
    volume_y: float | None = None
    min_volume: float | None = None
    spread_snapshot: str | None = None
    updated_at: str | None = None
    strategy_label: str | None = None
    zscore_norm: float | None = None
    zscore_roll: float | None = None
    dependency_profile: str | None = None
    dependency_x_over_y: float | None = None
    dependency_y_over_x: float | None = None
    correlation: float | None = None
    jn_flag: bool | None = None
    eg_flag: bool | None = None
    hurst: float | None = None
    half_life: float | None = None
    sigma_0_count: int | None = None
    sigma_1_count: int | None = None
    sigma_2_count: int | None = None
    var: float | None = None
    cvar: float | None = None
    mdd: float | None = None
    return_total: float | None = None
    sharpe: float | None = None
    raw_pair_cell: str | None = None
    raw_volume_cell: str | None = None
    raw_spread_cell: str | None = None
    raw_updated_cell: str | None = None
    raw_strategy_cell: str | None = None
    raw_zscore_cell: str | None = None
    raw_dependency_cell: str | None = None
    raw_stationarity_cell: str | None = None
    raw_risk_cell: str | None = None
    raw_reward_cell: str | None = None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def load_scanner_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_scanner_rows(input_dir: str | Path) -> list[CryptoWizardsScannerRow]:
    rows: list[CryptoWizardsScannerRow] = []
    for path in sorted(Path(input_dir).glob("*.json")):
        rows.extend(scanner_rows_from_payload(load_scanner_payload(path), source_path=str(path)))
    return rows


def scanner_rows_from_payload(payload: Any, source_path: str | None = None) -> list[CryptoWizardsScannerRow]:
    payload = _parse_json_string(payload)
    if isinstance(payload, list):
        return [_scanner_row_from_record(record, {}, source_path, idx) for idx, record in enumerate(payload)]
    if not isinstance(payload, dict):
        return []

    filters = payload.get("scanner_filters") or payload.get("filters") or {}
    context = {
        "source_url": payload.get("url") or payload.get("source_url"),
        "captured_at": payload.get("captured_at"),
        "scanner_priority": _filter_value(filters, "priority"),
        "scanner_count_filter": _filter_value(filters, "count"),
        "scanner_correlation_filter": _filter_value(filters, "correlation", "correl"),
        "scanner_hurst_filter": _filter_value(filters, "hurst"),
        "scanner_half_life_filter": _filter_value(filters, "half_life", "halflife"),
        "scanner_copula_filter": _filter_value(filters, "copula"),
        "scanner_strategy_filter": _filter_value(filters, "strategy"),
        "scanner_symbol_filter": _filter_value(filters, "symbol"),
        "scanner_exchange": _filter_value(filters, "exchange"),
    }
    candidates = (
        payload.get("scanner_rows")
        or payload.get("rows")
        or payload.get("items")
        or payload.get("data")
        or []
    )
    candidates = _parse_json_string(candidates)
    if isinstance(candidates, dict):
        candidates = candidates.get("scanner_rows") or candidates.get("rows") or candidates.get("items") or []
    if not isinstance(candidates, list):
        return []
    return [_scanner_row_from_record(record, context, source_path, idx) for idx, record in enumerate(candidates)]


def scanner_field_rows(rows: list[CryptoWizardsScannerRow]) -> list[dict[str, object]]:
    field_rows: list[dict[str, object]] = []
    for row in rows:
        for field, value in row.to_row().items():
            if value is None or value == "":
                continue
            field_rows.append(
                {
                    "field": field,
                    "type": type(value).__name__,
                    "example": str(value)[:160],
                    "source": "crypto_wizards_scanner",
                    "pair": row.pair or "",
                }
            )
    if not field_rows:
        return []
    frame = pd.DataFrame(field_rows).drop_duplicates(subset=["field", "source", "pair"])
    return frame.sort_values(["source", "field", "pair"]).to_dict("records")


def write_scanner_reports(input_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_scanner_rows(input_dir)
    rows_path = output / "crypto_wizards_scanner_rows.csv"
    fields_path = output / "crypto_wizards_scanner_field_dictionary.csv"
    pd.DataFrame([row.to_row() for row in rows], columns=SCANNER_COLUMNS).to_csv(rows_path, index=False)
    pd.DataFrame(scanner_field_rows(rows)).to_csv(fields_path, index=False)
    return {"rows": rows_path, "fields": fields_path}


def _scanner_row_from_record(
    record: Any,
    context: dict[str, Any],
    source_path: str | None,
    row_index: int,
) -> CryptoWizardsScannerRow:
    parsed = _parse_json_string(record)
    if not isinstance(parsed, dict):
        parsed = {"raw_pair_cell": str(record)}
    normalized = {snake_case(str(key)): value for key, value in parsed.items()}
    cells = _row_cells(normalized)

    pair_cell = _first_text(normalized, "raw_pair_cell", "pair_cell", "pair") or _cell(cells, 0)
    volume_cell = _first_text(normalized, "raw_volume_cell", "volume_cell", "volume") or _cell(cells, 1)
    spread_cell = _first_text(normalized, "raw_spread_cell", "spread_cell", "spread_snapshot", "spread") or _cell(cells, 2)
    updated_cell = _first_text(normalized, "raw_updated_cell", "updated_cell", "updated_at", "updated") or _cell(cells, 3)
    strategy_cell = _first_text(normalized, "raw_strategy_cell", "strategy_cell", "strategy", "strategy_label") or _cell(cells, 4)
    zscore_cell = _first_text(normalized, "raw_zscore_cell", "zscore_cell", "zscore") or _cell(cells, 5)
    dependency_cell = _first_text(normalized, "raw_dependency_cell", "dependency_cell", "dependency") or _cell(cells, 6)
    stationarity_cell = _first_text(normalized, "raw_stationarity_cell", "stationarity_cell", "stationarity") or _cell(cells, 7)
    risk_cell = _first_text(normalized, "raw_risk_cell", "risk_cell", "volatility_and_risk", "risk") or _cell(cells, 8)
    reward_cell = _first_text(normalized, "raw_reward_cell", "reward_cell", "reward") or _cell(cells, 9)

    wizard_exchange = normalize_wizard_exchange(context.get("scanner_exchange") or normalized.get("exchange"), default="dydx")
    asset_x, asset_y = _pair_assets(normalized, pair_cell)
    x_symbol = normalize_wizard_symbol(asset_x, wizard_exchange)
    y_symbol = normalize_wizard_symbol(asset_y, wizard_exchange)
    normalized_pair = (
        f"{x_symbol.normalized_symbol}-{y_symbol.normalized_symbol}"
        if x_symbol.normalized_symbol and y_symbol.normalized_symbol
        else None
    )
    volumes = _numbers(volume_cell)
    zscores = _numbers(zscore_cell)
    dependency_values = _numbers(dependency_cell)
    stationarity_values = _numbers(stationarity_cell)
    risk_values = _numbers(risk_cell)
    reward_values = _numbers(reward_cell)
    sigma_counts = _sigma_counts(stationarity_cell)

    return CryptoWizardsScannerRow(
        source_path=source_path,
        source_url=_text_or_none(context.get("source_url") or normalized.get("source_url") or normalized.get("url")),
        captured_at=_text_or_none(context.get("captured_at") or normalized.get("captured_at")),
        scanner_priority=_text_or_none(context.get("scanner_priority")),
        scanner_count_filter=_text_or_none(context.get("scanner_count_filter")),
        scanner_correlation_filter=_text_or_none(context.get("scanner_correlation_filter")),
        scanner_hurst_filter=_text_or_none(context.get("scanner_hurst_filter")),
        scanner_half_life_filter=_text_or_none(context.get("scanner_half_life_filter")),
        scanner_copula_filter=_text_or_none(context.get("scanner_copula_filter")),
        scanner_strategy_filter=_text_or_none(context.get("scanner_strategy_filter")),
        scanner_symbol_filter=_text_or_none(context.get("scanner_symbol_filter")),
        scanner_exchange=wizard_exchange,
        wizard_exchange=wizard_exchange,
        wizard_exchange_lane=wizard_exchange_lane(wizard_exchange),
        wizard_promotion_allowed=wizard_exchange_promotion_allowed(wizard_exchange),
        row_index=_safe_int(normalized.get("row_index"), row_index),
        pair=_first_text(normalized, "pair") or (f"{asset_x}-{asset_y}" if asset_x and asset_y else None),
        asset_x=asset_x,
        asset_y=asset_y,
        asset_x_raw=x_symbol.raw_symbol,
        asset_y_raw=y_symbol.raw_symbol,
        asset_x_normalized=x_symbol.normalized_symbol,
        asset_y_normalized=y_symbol.normalized_symbol,
        asset_x_base=x_symbol.base_asset,
        asset_y_base=y_symbol.base_asset,
        asset_x_quote=x_symbol.quote_asset,
        asset_y_quote=y_symbol.quote_asset,
        asset_x_canonical_usd=x_symbol.canonical_usd_symbol,
        asset_y_canonical_usd=y_symbol.canonical_usd_symbol,
        normalized_pair=normalized_pair,
        pair_id=_first_text(normalized, "pair_id") or _pair_id(asset_x, asset_y),
        volume_x=_safe_float(normalized.get("volume_x"), volumes[0] if len(volumes) > 0 else None),
        volume_y=_safe_float(normalized.get("volume_y"), volumes[1] if len(volumes) > 1 else None),
        min_volume=_safe_float(normalized.get("min_volume"), min(volumes[:2]) if len(volumes) >= 2 else None),
        spread_snapshot=spread_cell,
        updated_at=updated_cell,
        strategy_label=_first_text(normalized, "strategy_label", "strategy") or _clean_lines(strategy_cell),
        zscore_norm=_safe_float(normalized.get("zscore_norm"), zscores[0] if len(zscores) > 0 else None),
        zscore_roll=_safe_float(normalized.get("zscore_roll"), zscores[1] if len(zscores) > 1 else None),
        dependency_profile=_first_text(normalized, "dependency_profile") or _dependency_profile(dependency_cell),
        dependency_x_over_y=_safe_float(
            normalized.get("dependency_x_over_y"), dependency_values[0] if len(dependency_values) > 0 else None
        ),
        dependency_y_over_x=_safe_float(
            normalized.get("dependency_y_over_x"), dependency_values[1] if len(dependency_values) > 1 else None
        ),
        correlation=_safe_float(normalized.get("correlation"), dependency_values[2] if len(dependency_values) > 2 else None),
        jn_flag=_safe_bool(normalized.get("jn_flag"), _contains_token(stationarity_cell, "Jn")),
        eg_flag=_safe_bool(normalized.get("eg_flag"), _contains_token(stationarity_cell, "EG")),
        hurst=_safe_float(normalized.get("hurst"), _labeled_number(stationarity_cell, "hurst") or (stationarity_values[0] if stationarity_values else None)),
        half_life=_safe_float(
            normalized.get("half_life"),
            _labeled_number(stationarity_cell, "half life", "halflife") or (stationarity_values[1] if len(stationarity_values) > 1 else None),
        ),
        sigma_0_count=_safe_int(normalized.get("sigma_0_count"), sigma_counts.get(0)),
        sigma_1_count=_safe_int(normalized.get("sigma_1_count"), sigma_counts.get(1)),
        sigma_2_count=_safe_int(normalized.get("sigma_2_count"), sigma_counts.get(2)),
        var=_safe_float(normalized.get("var"), risk_values[0] if len(risk_values) > 0 else None),
        cvar=_safe_float(normalized.get("cvar"), risk_values[1] if len(risk_values) > 1 else None),
        mdd=_safe_float(normalized.get("mdd"), risk_values[2] if len(risk_values) > 2 else None),
        return_total=_safe_float(normalized.get("return_total"), reward_values[0] if len(reward_values) > 0 else None),
        sharpe=_safe_float(normalized.get("sharpe"), reward_values[1] if len(reward_values) > 1 else None),
        raw_pair_cell=pair_cell,
        raw_volume_cell=volume_cell,
        raw_spread_cell=spread_cell,
        raw_updated_cell=updated_cell,
        raw_strategy_cell=strategy_cell,
        raw_zscore_cell=zscore_cell,
        raw_dependency_cell=dependency_cell,
        raw_stationarity_cell=stationarity_cell,
        raw_risk_cell=risk_cell,
        raw_reward_cell=reward_cell,
    )


def _row_cells(record: dict[str, Any]) -> list[str]:
    raw = record.get("cells") or record.get("row_cells") or record.get("raw_cells")
    if isinstance(raw, list):
        return [_clean_lines(str(cell)) for cell in raw]
    return []


def _cell(cells: list[str], index: int) -> str | None:
    return cells[index] if len(cells) > index else None


def _first_text(record: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = record.get(key)
        if value is None:
            continue
        text = _clean_lines(str(value))
        if text:
            return text
    return None


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = _clean_lines(str(value))
    return text or None


def _clean_lines(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value.replace("\u03c3", "σ")).strip()
    return text or None


def _parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _filter_value(filters: Any, *keys: str) -> str | None:
    if not isinstance(filters, dict):
        return None
    normalized = {snake_case(str(key)): value for key, value in filters.items()}
    for key in keys:
        value = normalized.get(snake_case(key))
        if value is not None:
            return _text_or_none(value)
    return None


def _pair_assets(record: dict[str, Any], pair_cell: str | None) -> tuple[str | None, str | None]:
    asset_x = _first_text(record, "asset_x", "symbol_1", "symbol1", "base_asset")
    asset_y = _first_text(record, "asset_y", "symbol_2", "symbol2", "quote_asset")
    if asset_x and asset_y:
        return asset_x, asset_y
    tokens = re.findall(r"\b[A-Z0-9]+-USD\b", pair_cell or "")
    if len(tokens) >= 2:
        return tokens[0], tokens[1]
    tokens = _symbol_tokens(pair_cell)
    if len(tokens) >= 2:
        return tokens[0], tokens[1]
    return asset_x, asset_y


def _symbol_tokens(text: str | None) -> list[str]:
    if not text:
        return []
    tokens = re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)?\b", text.upper())
    useful = []
    quote_markers = ("-USD", "-USDT", "-USDC", "-BTC", "-ETH", "USDT", "USDC", "USD", "PERP")
    ignored = {"PAIR", "ZSCORE", "REWARD", "SHARPE", "NORM", "ROLL", "NEW", "DYD", "DYDX"}
    for token in tokens:
        if token in ignored:
            continue
        if any(marker in token for marker in quote_markers):
            useful.append(token)
    return useful


def _pair_id(asset_x: str | None, asset_y: str | None) -> str | None:
    if not asset_x or not asset_y:
        return None
    return f"{asset_x.lower().replace('-usd', '')}_{asset_y.lower().replace('-usd', '')}"


def _numbers(text: str | None) -> list[float]:
    if not text:
        return []
    values: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?\s*[%kKmMbB]?", text):
        raw = match.group(0).strip()
        values.append(_parse_number(raw))
    return values


def _parse_number(raw: str) -> float:
    text = raw.strip().replace(",", "")
    multiplier = 1.0
    if text.endswith("%"):
        text = text[:-1].strip()
        multiplier = 0.01
    elif text[-1:].lower() == "k":
        text = text[:-1]
        multiplier = 1_000.0
    elif text[-1:].lower() == "m":
        text = text[:-1]
        multiplier = 1_000_000.0
    elif text[-1:].lower() == "b":
        text = text[:-1]
        multiplier = 1_000_000_000.0
    return float(text) * multiplier


def _safe_float(value: Any, fallback: Any = None) -> float | None:
    candidate = value if value is not None else fallback
    if candidate is None or candidate == "":
        return None
    try:
        if isinstance(candidate, str):
            return _parse_number(candidate)
        return float(candidate)
    except (TypeError, ValueError, IndexError):
        return None


def _safe_int(value: Any, fallback: Any = None) -> int | None:
    candidate = value if value is not None else fallback
    if candidate is None or candidate == "":
        return None
    try:
        return int(float(str(candidate).replace(",", "")))
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any, fallback: bool | None = None) -> bool | None:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "pass", "passed"}:
        return True
    if text in {"false", "0", "no", "n", "fail", "failed"}:
        return False
    return fallback


def _contains_token(text: str | None, token: str) -> bool | None:
    if not text:
        return None
    pattern = rf"\b{re.escape(token)}\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def _dependency_profile(text: str | None) -> str | None:
    if not text:
        return None
    if re.search(r"\bprofile\b", text, flags=re.IGNORECASE):
        return "profile"
    return None


def _labeled_number(text: str | None, *labels: str) -> float | None:
    if not text:
        return None
    for label in labels:
        match = re.search(rf"([-+]?\d+(?:\.\d+)?)\s*{re.escape(label)}", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(rf"{re.escape(label)}\s*([-+]?\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _sigma_counts(text: str | None) -> dict[int, int]:
    counts: dict[int, int] = {}
    if not text:
        return counts
    for match in re.finditer(r"(\d+)\s*([012])\s*σ", text):
        counts[int(match.group(2))] = int(match.group(1))
    return counts
