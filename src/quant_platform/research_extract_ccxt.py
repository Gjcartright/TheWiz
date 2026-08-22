from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import ROOT, CommandResult
from quant_platform.orchestration.phase00_ccxt_evidence import (
    CcxtCaptureRequest,
    CcxtEvidenceRuntime,
    write_ccxt_capture_evidence,
)
from quant_platform.research_extraction import normalize_extraction_rows
from quant_platform.research_ingestion import (
    active_research_sources,
    mark_research_sources_processed,
)


@dataclass(frozen=True)
class _CcxtSourceSpec:
    exchange: str
    lane: str
    symbols: list[str]
    timeframes: list[str]
    limit: int
    since_ms: int | None
    end_ms: int | None
    captured_at_ms: int | None
    force_market_reload: bool


def ccxt_research_paths(root: Path = ROOT) -> dict[str, Path]:
    base = root / "data" / "processed" / "research_knowledge"
    return {
        "ccxt_research_rows": base / "ccxt_research_rows.csv",
        "ccxt_research_features": base / "ccxt_research_features.csv",
        "ccxt_research_rules": base / "ccxt_research_rules.csv",
        "ccxt_research_summary": base / "ccxt_research_summary.txt",
    }


def extract_research_knowledge(
    root: Path = ROOT,
    *,
    evidence_runtime: CcxtEvidenceRuntime | None = None,
) -> CommandResult:
    registry = active_research_sources(root)
    ccxt_sources = (
        registry[registry["source_type"].astype(str) == "ccxt"].copy()
        if not registry.empty
        else pd.DataFrame()
    )

    accepted_sources = 0
    failed_sources = len(ccxt_sources)
    capture_count = 0
    if evidence_runtime is None:
        rows = []
        summary_lines = [
            f"sources={len(ccxt_sources)}",
            "accepted_sources=0",
            f"failed_sources={len(ccxt_sources)}",
            "capture_count=0",
            "blocker=ccxt_evidence_runtime_not_injected",
            "no_ccxt_rows_collected",
        ]
    else:
        rows, accepted_sources, failed_sources, capture_count = _collect_ccxt_rows(
            ccxt_sources,
            evidence_runtime=evidence_runtime,
            root=root,
        )
        summary_lines = [
            f"sources={len(ccxt_sources)}",
            f"accepted_sources={accepted_sources}",
            f"failed_sources={failed_sources}",
            f"capture_count={capture_count}",
            f"rows={len(rows)}",
            "mode=phase00_ccxt_evidence_contract",
        ]

    normalized = normalize_extraction_rows(rows)
    paths = ccxt_research_paths(root=root)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(normalized, paths["ccxt_research_rows"], index=False)
    atomic_write_csv(normalized[
        normalized["row_type"].isin(
            [
                "risk_prior",
                "mode_preference",
                "regime_condition",
                "pair_filter",
                "anti_pattern",
                "execution_warning",
            ]
        )
    ], paths["ccxt_research_rules"], index=False)
    atomic_write_csv(normalized[normalized["row_type"] == "feature_idea"], paths["ccxt_research_features"], index=False)
    atomic_write_text(paths["ccxt_research_summary"], "\n".join(summary_lines), encoding="utf-8")

    all_sources_accepted = (
        not ccxt_sources.empty
        and accepted_sources == len(ccxt_sources)
        and failed_sources == 0
        and bool(rows)
    )
    if all_sources_accepted:
        mark_research_sources_processed("ccxt", root=root)

    return CommandResult(
        paths={
            **paths,
            "research_rows": paths["ccxt_research_rows"],
        },
        summary={
            "rows": len(normalized),
            "ccxt_rows": len(rows),
            "youtube_rows": 0,
            "sources": len(ccxt_sources),
            "accepted_sources": int(accepted_sources),
            "failed_sources": int(failed_sources),
            "capture_count": int(capture_count),
            "processed": bool(all_sources_accepted),
        },
    )


def _collect_ccxt_rows(
    sources: pd.DataFrame,
    *,
    evidence_runtime: CcxtEvidenceRuntime,
    root: Path,
) -> tuple[list[dict[str, object]], int, int, int]:
    rows: list[dict[str, object]] = []
    accepted_sources = 0
    failed_sources = 0
    capture_count = 0
    for _, source in sources.iterrows():
        spec = _parse_ccxt_spec(str(source.get("source_path_or_url", "")))
        if spec is None:
            failed_sources += 1
            continue
        source_rows, source_accepted, source_capture_count = _extract_rows_for_source(
            source=source,
            spec=spec,
            evidence_runtime=evidence_runtime,
            root=root,
        )
        rows.extend(source_rows)
        capture_count += source_capture_count
        if source_accepted and source_rows:
            accepted_sources += 1
        else:
            failed_sources += 1
    return rows, accepted_sources, failed_sources, capture_count


def _extract_rows_for_source(
    *,
    source: pd.Series,
    spec: _CcxtSourceSpec,
    evidence_runtime: CcxtEvidenceRuntime,
    root: Path,
) -> tuple[list[dict[str, object]], bool, int]:
    rows: list[dict[str, object]] = []
    if not spec.lane or spec.since_ms is None or spec.end_ms is None or spec.captured_at_ms is None:
        return rows, False, 0
    all_captures_accepted = True
    capture_count = 0
    for symbol in spec.symbols:
        for timeframe in spec.timeframes:
            request = CcxtCaptureRequest(
                source_id=str(source.get("source_id", "")),
                lane=spec.lane,
                exchange_id=spec.exchange,
                symbol=symbol,
                timeframe=timeframe,
                since_ms=spec.since_ms,
                end_ms=spec.end_ms,
                captured_at_ms=spec.captured_at_ms,
                page_limit=spec.limit,
                force_market_reload=spec.force_market_reload,
            )
            capture = evidence_runtime.capture(request)
            evidence_paths = write_ccxt_capture_evidence(capture, root=root)
            capture_count += 1
            if not capture.accepted:
                all_captures_accepted = False
                continue
            payload = [
                [
                    bar["timestamp_ms"],
                    bar["open"],
                    bar["high"],
                    bar["low"],
                    bar["close"],
                    bar["raw_base_volume"],
                ]
                for bar in capture.normalized_ohlcv
            ]
            metric_rows = _rows_from_ohlcv(
                payload=payload,
                source=source,
                exchange=spec.exchange,
                symbol=symbol,
                timeframe=timeframe,
                evidence_path=str(evidence_paths["receipt"]),
            )
            if not metric_rows:
                all_captures_accepted = False
            rows.extend(metric_rows)
    return rows, all_captures_accepted, capture_count


def _rows_from_ohlcv(
    *,
    payload: list[list[object]],
    source: pd.Series,
    exchange: str,
    symbol: str,
    timeframe: str,
    evidence_path: str = "",
) -> list[dict[str, object]]:
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
    base = _base_row(
        source,
        exchange,
        symbol,
        timeframe,
        evidence_path=evidence_path,
    )
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
    elif (
        ":" in value and "{" not in value and "[" not in value and ";" not in value and "," in value
    ):
        exchange_part, body = value.split(":", 1)
    elif value.endswith(".json") and Path(value).exists():
        return _parse_ccxt_json_manifest(Path(value))
    else:
        return None

    exchange = exchange_part.strip().lower()
    if not exchange or not body:
        return None

    symbols_part = body
    options_by_name: dict[str, str] = {}
    if ";" in body:
        first, *options = body.split(";")
        symbols_part = first
        for option in options:
            key, separator, value = option.partition("=")
            key = key.strip().lower()
            value = value.strip()
            allowed_options = {
                "lane",
                "timeframe",
                "limit",
                "since_ms",
                "end_ms",
                "captured_at_ms",
                "force_market_reload",
            }
            if separator != "=" or key not in allowed_options or not value:
                return None
            options_by_name[key] = value
    symbols = [symbol.strip() for symbol in symbols_part.split(",") if symbol.strip()]
    if not symbols:
        return None
    timeframe_part = options_by_name.get("timeframe", "")
    timeframes = [tf.strip() for tf in (timeframe_part.split(",") if timeframe_part else ["5m"])]
    limit = _parse_optional_int(options_by_name.get("limit"), default=300)
    if limit is None:
        return None
    timeframes = [tf for tf in timeframes if tf]
    if not timeframes:
        timeframes = ["5m"]
    force_reload = options_by_name.get("force_market_reload", "false").lower()
    if force_reload not in {"0", "1", "false", "true", "no", "yes"}:
        return None
    return _CcxtSourceSpec(
        exchange=exchange,
        lane=options_by_name.get("lane", "").strip(),
        symbols=symbols,
        timeframes=timeframes,
        limit=limit,
        since_ms=_parse_optional_int(options_by_name.get("since_ms")),
        end_ms=_parse_optional_int(options_by_name.get("end_ms")),
        captured_at_ms=_parse_optional_int(options_by_name.get("captured_at_ms")),
        force_market_reload=force_reload in {"1", "true", "yes"},
    )


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
    if not isinstance(data.get("symbols"), list) or not isinstance(
        data.get("timeframes", []), list
    ):
        return None
    exchange = str(data.get("exchange", "")).strip().lower()
    symbols = [str(item).strip() for item in data.get("symbols", []) if str(item).strip()]
    timeframes = [str(item).strip() for item in data.get("timeframes", []) if str(item).strip()]
    limit = _parse_optional_int(data.get("limit"), default=300)
    force_market_reload = data.get("force_market_reload", False)
    if not exchange or not symbols or limit is None or not isinstance(force_market_reload, bool):
        return None
    return _CcxtSourceSpec(
        exchange=exchange,
        lane=str(data.get("lane", "")).strip(),
        symbols=symbols,
        timeframes=timeframes or ["5m"],
        limit=limit,
        since_ms=_parse_optional_int(data.get("since_ms")),
        end_ms=_parse_optional_int(data.get("end_ms")),
        captured_at_ms=_parse_optional_int(data.get("captured_at_ms")),
        force_market_reload=force_market_reload,
    )


def _base_row(
    source: pd.Series,
    exchange: str,
    symbol: str,
    timeframe: str,
    *,
    evidence_path: str = "",
) -> dict[str, object]:
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
        "evidence_path": evidence_path
        or f"{source.get('source_path_or_url', '')}#{symbol}:{timeframe}",
        "review_status": str(source.get("review_status", "unreviewed")),
        "notes": f"exchange={exchange};symbol={symbol};timeframe={timeframe}",
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _parse_optional_int(value: object, *, default: int | None = None) -> int | None:
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None


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
