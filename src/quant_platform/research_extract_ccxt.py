from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.research_extraction import EXTRACTION_COLUMNS, normalize_extraction_rows
from quant_platform.research_ingestion import active_research_sources, mark_research_sources_processed

try:
    import ccxt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ccxt = None


@dataclass(frozen=True)
class _CcxtSourceSpec:
    exchange: str
    symbols: list[str]
    timeframes: list[str]
    limit: int


def ccxt_research_paths(root: Path = ROOT) -> dict[str, Path]:
    base = root / "data" / "processed" / "research_knowledge"
    return {
        "ccxt_research_rows": base / "ccxt_research_rows.csv",
        "ccxt_research_features": base / "ccxt_research_features.csv",
        "ccxt_research_rules": base / "ccxt_research_rules.csv",
        "ccxt_research_summary": base / "ccxt_research_summary.txt",
    }


def extract_research_knowledge(root: Path = ROOT) -> CommandResult:
    registry = active_research_sources(root)
    ccxt_sources = registry[registry["source_type"].astype(str) == "ccxt"].copy() if not registry.empty else pd.DataFrame()

    if ccxt is None:
        rows = []
        summary_lines = [
            "ccxt module unavailable (ccxt/ccxt not installed)",
            "install with: pip install ccxt",
            "no_ccxt_rows_collected",
        ]
    else:
        rows = _collect_ccxt_rows(ccxt_sources)
        summary_lines = [
            f"sources={len(ccxt_sources)}",
            f"rows={len(rows)}",
            "mode=ccxt_research_enabled",
        ]

    normalized = normalize_extraction_rows(rows)
    paths = ccxt_research_paths(root=root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(paths["ccxt_research_rows"], index=False)
    normalized[normalized["row_type"].isin(["risk_prior", "mode_preference", "regime_condition", "pair_filter", "anti_pattern", "execution_warning"])].to_csv(
        paths["ccxt_research_rules"], index=False
    )
    normalized[normalized["row_type"] == "feature_idea"].to_csv(paths["ccxt_research_features"], index=False)
    paths["ccxt_research_summary"].write_text("\n".join(summary_lines), encoding="utf-8")

    if not ccxt_sources.empty:
        mark_research_sources_processed("ccxt", root=root)

    return CommandResult(
        paths={
            **paths,
            "research_rows": paths["ccxt_research_rows"],
        },
        summary={"rows": int(len(normalized)), "ccxt_rows": int(len(rows)), "youtube_rows": 0, "sources": int(len(ccxt_sources))},
    )


def _collect_ccxt_rows(sources: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for _, source in sources.iterrows():
        spec = _parse_ccxt_spec(str(source.get("source_path_or_url", "")))
        if spec is None:
            continue
        rows.extend(_extract_rows_for_source(source=source, spec=spec))
    return rows


def _extract_rows_for_source(*, source: pd.Series, spec: _CcxtSourceSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if ccxt is None:
        return rows
    exchange_id = spec.exchange.lower()
    if not hasattr(ccxt, exchange_id):
        return rows
    exchange_class = getattr(ccxt, exchange_id)
    instance = exchange_class({"enableRateLimit": True, "timeout": 120_000})
    for symbol in spec.symbols:
        for timeframe in spec.timeframes:
            try:
                payload = instance.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=spec.limit)
            except Exception:
                continue
            if not isinstance(payload, list) or len(payload) < 8:
                continue
            metric_rows = _rows_from_ohlcv(
                payload=payload,
                source=source,
                exchange=exchange_id,
                symbol=symbol,
                timeframe=timeframe,
            )
            rows.extend(metric_rows)
    return rows


def _rows_from_ohlcv(*, payload: list[list[object]], source: pd.Series, exchange: str, symbol: str, timeframe: str) -> list[dict[str, object]]:
    rows = [item for item in payload if isinstance(item, (list, tuple))]
    if len(rows) < 8:
        return []

    close = pd.Series([_safe_float(item[4]) for item in rows if len(item) >= 5], dtype=float)
    volume = pd.Series([_safe_float(item[5]) for item in rows if len(item) >= 6], dtype=float)
    high = pd.Series([_safe_float(item[2]) for item in rows if len(item) >= 3], dtype=float)
    low = pd.Series([_safe_float(item[3]) for item in rows if len(item) >= 4], dtype=float)

    if len(close) < 8:
        return []

    returns = close.pct_change().dropna()
    if returns.empty:
        return []
    volatility = float(returns.std(ddof=0))
    avg_volume = float(volume.mean() if not volume.empty else 0.0)
    return_mean = float(returns.mean())
    drawdown = _estimate_drawdown(close)
    trend_slope = _estimate_trend(close)
    regime = "regime_flat"
    if volatility >= 0.008:
        regime = "regime_high_volatility"
    elif volatility <= 0.002:
        regime = "regime_low_volatility"
    if trend_slope > 0.0001:
        regime = f"{regime}_trend_up"
    elif trend_slope < -0.0001:
        regime = f"{regime}_trend_down"

    pair_filter = _normalize_pair_filter(symbol)
    base = _base_row(source, exchange, symbol, timeframe)
    rows: list[dict[str, object]] = []
    rows.append(
        {
            **base,
            "row_type": "feature_idea",
            "feature_name": f"ccxt_volatility_{exchange}_{timeframe}_{pair_filter}",
            "pair_filter": pair_filter,
            "timeframe_preference": timeframe,
            "notes": f"rows={len(close)};volatility={volatility:.8f};mean_return={return_mean:.8f};trend={trend_slope:.6f}",
            "regime_condition": regime,
        }
    )
    rows.append(
        {
            **base,
            "row_type": "feature_idea",
            "feature_name": f"ccxt_liquidity_{exchange}_{timeframe}_{pair_filter}",
            "pair_filter": pair_filter,
            "timeframe_preference": timeframe,
            "notes": f"rows={len(close)};avg_volume={avg_volume:.4f};drawdown={drawdown:.6f}",
            "risk_rule": f"avg_volume={avg_volume:.6f}",
        }
    )
    rows.append(
        {
            **base,
            "row_type": "regime_condition",
            "regime_condition": regime,
            "pair_filter": pair_filter,
            "timeframe_preference": timeframe,
            "notes": f"volatility={volatility:.8f};drawdown={drawdown:.6f};trend={trend_slope:.6f}",
        }
    )

    if volatility >= 0.0:
        rows.append(
            {
                **base,
                "row_type": "risk_prior",
                "risk_rule": f"ccxt_availability={exchange}:{symbol}:{timeframe};volatility={volatility:.6f};drawdown={drawdown:.6f}",
                "pair_filter": pair_filter,
                "timeframe_preference": timeframe,
                "notes": f"rows={len(close)};return_mean={return_mean:.8f};trend={trend_slope:.6f}",
            }
        )
    if len(close) < 40:
        rows.append(
            {
                **base,
                "row_type": "anti_pattern",
                "pair_filter": pair_filter,
                "anti_pattern": f"insufficient_history:{exchange}:{symbol}:{timeframe};bars<{len(close)}",
                "notes": f"rows={len(close)}",
            }
        )
    if len(close) >= 40 and high.notna().any() and low.notna().any():
        rows.append(
            {
                **base,
                "row_type": "pair_filter",
                "pair_filter": pair_filter,
                "notes": f"symbol={symbol};timeframe={timeframe};exchange={exchange};window={len(close)}",
            }
        )
    if avg_volume < 1.0:
        rows.append(
            {
                **base,
                "row_type": "anti_pattern",
                "pair_filter": pair_filter,
                "anti_pattern": f"low_liquidity:{exchange}:{symbol}:{timeframe};avg_volume<{avg_volume:.4f}",
                "notes": f"rows={len(close)};avg_volume={avg_volume:.4f}",
            }
        )
        rows.append(
            {
                **base,
                "row_type": "risk_prior",
                "risk_rule": f"skip_if_illiquid:{exchange}:{symbol}:{timeframe};avg_volume={avg_volume:.4f}",
                "pair_filter": pair_filter,
                "notes": f"rows={len(close)};avg_volume={avg_volume:.4f}",
            }
        )
    if avg_volume == 0:
        rows.append(
            {
                **base,
                "row_type": "execution_warning",
                "execution_warning": f"zero_volume:{exchange}:{symbol}:{timeframe}",
                "pair_filter": pair_filter,
                "notes": f"rows={len(close)};avg_volume={avg_volume:.4f}",
            }
        )
    return rows


def _parse_ccxt_spec(raw: str) -> _CcxtSourceSpec | None:
    value = raw.strip()
    if not value:
        return None

    if value.startswith("ccxt://"):
        value = value[len("ccxt://") :].strip()

    # Format: exchange:symbol1,symbol2;timeframe=5m;limit=100
    if ":" in value and "{" not in value and "[" not in value and ";" in value:
        exchange_part, rest = value.split(":", 1)
        body = rest
    elif ":" in value and "{" not in value and "[" not in value and ";" not in value and "," in value:
        exchange_part, body = value.split(":", 1)
    elif value.endswith(".json") and Path(value).exists():
        return _parse_ccxt_json_manifest(Path(value))
    else:
        return None

    exchange = exchange_part.strip().lower()
    if not exchange or not body:
        return None

    symbols_part, timeframe_part, limit_part = body, "", ""
    if ";" in body:
        first, *options = body.split(";")
        symbols_part = first
        for option in options:
            key, _, value = option.partition("=")
            key = key.strip().lower()
            value = value.strip()
            if key == "timeframe":
                timeframe_part = value
            elif key == "limit":
                limit_part = value
    symbols = [symbol.strip() for symbol in symbols_part.split(",") if symbol.strip()]
    if not symbols:
        return None
    timeframes = [tf.strip() for tf in (timeframe_part.split(",") if timeframe_part else ["5m"])]
    try:
        limit = int(limit_part) if limit_part else 300
    except ValueError:
        limit = 300
    timeframes = [tf for tf in timeframes if tf]
    if not timeframes:
        timeframes = ["5m"]
    return _CcxtSourceSpec(exchange=exchange, symbols=symbols, timeframes=timeframes, limit=limit)


def _parse_ccxt_json_manifest(path: Path) -> _CcxtSourceSpec | None:
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    import json

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    exchange = str(data.get("exchange", "")).strip().lower()
    symbols = [str(item).strip() for item in data.get("symbols", []) if str(item).strip()]
    timeframes = [str(item).strip() for item in data.get("timeframes", []) if str(item).strip()]
    limit = int(data.get("limit", 300) or 300)
    if not exchange or not symbols:
        return None
    return _CcxtSourceSpec(exchange=exchange, symbols=symbols, timeframes=timeframes or ["5m"], limit=limit)


def _base_row(source: pd.Series, exchange: str, symbol: str, timeframe: str) -> dict[str, object]:
    return {
        "strategy_family": "",
        "mode_preference": "",
        "entry_logic": "",
        "exit_logic": "",
        "risk_rule": "",
        "feature_name": "",
        "pair_filter": "",
        "timeframe_preference": "",
        "regime_condition": "",
        "anti_pattern": "",
        "execution_warning": "",
        "confidence": float(source.get("confidence", 0.5) or 0.5),
        "source_id": str(source.get("source_id", "")),
        "source_type": "ccxt",
        "source_title": str(source.get("title", "")),
        "evidence_path": f"{source.get('source_path_or_url', '')}#{symbol}:{timeframe}",
        "review_status": str(source.get("review_status", "unreviewed")),
        "notes": f"exchange={exchange};symbol={symbol};timeframe={timeframe}",
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _normalize_pair_filter(symbol: str) -> str:
    return str(symbol or "").replace("/", "-").replace("_", "-").upper()


def _estimate_drawdown(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    cumulative = (1 + close.pct_change().fillna(0.0)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def _estimate_trend(close: pd.Series) -> float:
    if close.empty:
        return 0.0
    if len(close) < 2:
        return 0.0
    first = _safe_float(close.iloc[0])
    last = _safe_float(close.iloc[-1])
    if first == 0:
        return 0.0
    return (last - first) / first
