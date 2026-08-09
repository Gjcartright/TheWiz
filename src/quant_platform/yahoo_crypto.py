from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult, ROOT, current_multi_venue_history_readiness_path
from quant_platform.dydx_candles import build_pair_history_from_candles
from quant_platform.experiments import PairDataset
from quant_platform.funding import enrich_pair_dataset_with_funding, normalize_funding_rows


YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com"
SUPPORTED_INTERVALS = {"1d", "1h", "1wk", "1mo"}
RESEARCH_ONLY_EXCHANGES = {"binance", "binanceus", "bybit", "coinbase"}
ZSCORE_WINDOW = 7
ZSCORE_MIN_WINDOW = 7


def fetch_yahoo_crypto_candles(
    *,
    symbol: str,
    interval: str = "1d",
    lookback_range: str = "max",
    output_dir: str | Path | None = None,
    base_url: str = YAHOO_CHART_BASE_URL,
    timeout: int = 30,
) -> Path:
    clean_symbol = _to_yahoo_symbol(symbol)
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported Yahoo interval: {interval}")
    params = {
        "interval": interval,
        "range": lookback_range,
        "includePrePost": "false",
        "events": "div,splits,capitalGains",
    }
    response = requests.get(
        f"{base_url.rstrip('/')}/v8/finance/chart/{clean_symbol}",
        params=params,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    payload = response.json()
    candles = normalize_yahoo_crypto_chart(payload, symbol=clean_symbol, interval=interval)
    if not candles:
        raise ValueError(f"no Yahoo candles returned for {clean_symbol} {interval}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "yahoo_crypto_candles")
    output_base.mkdir(parents=True, exist_ok=True)
    output = output_base / f"{clean_symbol}_{interval}_candles.json"
    output.write_text(
        json.dumps(
            {
                "source": "yahoo_crypto",
                "base_url": base_url,
                "request": {"symbol": clean_symbol, **params},
                "candles": candles,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def normalize_yahoo_crypto_chart(payload: Any, *, symbol: str, interval: str) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        return []
    chart = payload.get("chart", {})
    results = chart.get("result", []) if isinstance(chart, dict) else []
    if not results or not isinstance(results[0], dict):
        return []
    result = results[0]
    timestamps = result.get("timestamp", []) or []
    indicators = result.get("indicators", {}) if isinstance(result.get("indicators"), dict) else {}
    quote = indicators.get("quote", []) if isinstance(indicators, dict) else []
    rows = quote[0] if quote and isinstance(quote[0], dict) else {}
    opens = rows.get("open", []) or []
    highs = rows.get("high", []) or []
    lows = rows.get("low", []) or []
    closes = rows.get("close", []) or []
    volumes = rows.get("volume", []) or []
    candles: list[dict[str, object]] = []
    for idx, ts in enumerate(timestamps):
        close = _safe_float(_value_at(closes, idx))
        if close <= 0:
            continue
        volume = _safe_float(_value_at(volumes, idx))
        candles.append(
            {
                "startedAt": datetime.fromtimestamp(float(ts), timezone.utc).isoformat().replace("+00:00", "Z"),
                "ticker": _to_yahoo_symbol(symbol),
                "resolution": interval,
                "open": _safe_float(_value_at(opens, idx)),
                "high": _safe_float(_value_at(highs, idx)),
                "low": _safe_float(_value_at(lows, idx)),
                "close": close,
                "baseVolume": volume,
                "usdVolume": volume * close,
                "source": "yahoo_crypto",
            }
        )
    return sorted(candles, key=lambda candle: str(candle["startedAt"]))


def build_yahoo_crypto_pair_history(
    *,
    asset_x: str,
    asset_y: str,
    interval: str = "1d",
    pair_id: str | None = None,
    hedge_ratio: float | None = None,
    beta: float | None = None,
    zscore_window: int = ZSCORE_WINDOW,
    candle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    left_symbol = _to_yahoo_symbol(asset_x)
    right_symbol = _to_yahoo_symbol(asset_y)
    candle_base = Path(candle_dir or ROOT / "data" / "raw" / "yahoo_crypto_candles")
    left_path = candle_base / f"{left_symbol}_{interval}_candles.json"
    right_path = candle_base / f"{right_symbol}_{interval}_candles.json"
    missing = [str(path) for path in [left_path, right_path] if not path.exists()]
    if missing:
        raise ValueError(f"missing Yahoo candle files: {missing}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "pair_details")
    clean_pair_id = pair_id or f"{left_symbol.lower()}_{right_symbol.lower()}_yahoo_crypto_{interval}"
    output = output_base / f"pair_{_safe_filename(clean_pair_id)}_yahoo_crypto_{interval}_derived_history.json"
    path = build_pair_history_from_candles(
        left_path=left_path,
        right_path=right_path,
        output_path=output,
        pair_id=clean_pair_id,
        asset_x=left_symbol,
        asset_y=right_symbol,
        hedge_ratio=hedge_ratio,
        beta=beta,
        interval=interval,
        zscore_window=zscore_window,
        min_zscore_window=ZSCORE_MIN_WINDOW if zscore_window == ZSCORE_WINDOW else min(20, max(2, zscore_window // 4)),
    )
    _rewrite_pair_history_as_yahoo(path)
    return path


def refresh_yahoo_crypto_pair_history(
    *,
    asset_x: str,
    asset_y: str,
    interval: str = "1d",
    lookback_range: str = "max",
    pair_id: str | None = None,
    hedge_ratio: float | None = None,
    beta: float | None = None,
    zscore_window: int = ZSCORE_WINDOW,
    candle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    fetch_yahoo_crypto_candles(symbol=asset_x, interval=interval, lookback_range=lookback_range, output_dir=candle_dir)
    fetch_yahoo_crypto_candles(symbol=asset_y, interval=interval, lookback_range=lookback_range, output_dir=candle_dir)
    return build_yahoo_crypto_pair_history(
        asset_x=asset_x,
        asset_y=asset_y,
        interval=interval,
        pair_id=pair_id,
        hedge_ratio=hedge_ratio,
        beta=beta,
        zscore_window=zscore_window,
        candle_dir=candle_dir,
        output_dir=output_dir,
    )


def build_yahoo_crypto_lane_report(root: Path = ROOT) -> CommandResult:
    readiness = _read_csv(current_multi_venue_history_readiness_path(root))
    if readiness.empty:
        frame = pd.DataFrame(columns=["symbol", "exchange", "status", "daily_candle_rows", "candle_path", "next_action"])
    else:
        rows: list[dict[str, object]] = []
        seen: set[tuple[str, str]] = set()
        for _, row in readiness.iterrows():
            exchange = str(row.get("wizard_exchange", "") or "").strip().lower()
            if exchange not in RESEARCH_ONLY_EXCHANGES:
                continue
            for key in ("asset_x", "asset_y"):
                value = str(row.get(key, "") or "").strip()
                if not value:
                    continue
                symbol = _to_yahoo_symbol(value)
                marker = (exchange, symbol)
                if marker in seen:
                    continue
                seen.add(marker)
                rows.append(_yahoo_symbol_status(root, symbol, exchange))
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["status", "exchange", "symbol"]).reset_index(drop=True)
    active = root / "reports" / "active"
    csv_path = active / "yahoo_crypto_history_readiness.csv"
    md_path = active / "yahoo_crypto_history_readiness.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _yahoo_lane_markdown(frame))
    return CommandResult(
        paths={"yahoo_crypto_history_readiness": csv_path, "yahoo_crypto_history_readiness_md": md_path},
        summary={
            "symbols": len(frame),
            "ready_symbols": int(frame["status"].astype(str).eq("history_ready").sum()) if not frame.empty else 0,
            "blocked_symbols": int(frame["status"].astype(str).ne("history_ready").sum()) if not frame.empty else 0,
        },
    )


def refresh_yahoo_research_candidates(
    *,
    root: Path = ROOT,
    max_pairs: int = 5,
    interval: str = "1d",
    lookback_range: str = "max",
    zscore_window: int = ZSCORE_WINDOW,
) -> CommandResult:
    readiness = _read_csv(current_multi_venue_history_readiness_path(root))
    columns = [
        "rank",
        "pair",
        "wizard_exchange",
        "asset_x",
        "asset_y",
        "readiness_status",
        "refresh_status",
        "history_path",
        "error",
        "refresh_command",
    ]
    if readiness.empty:
        candidates = _derived_yahoo_research_candidates(root, max_pairs=max_pairs)
    else:
        candidates = readiness.copy()
        candidates = candidates[candidates.get("wizard_exchange", pd.Series(dtype=str)).astype(str).str.lower().isin(RESEARCH_ONLY_EXCHANGES)]
        candidates = candidates[candidates.get("readiness_status", pd.Series(dtype=str)).astype(str).eq("ready_to_fetch")]
        candidates["rank"] = pd.to_numeric(candidates.get("rank", pd.Series(dtype=float)), errors="coerce").fillna(999999)
        candidates = candidates.sort_values(["rank"]).head(max_pairs).reset_index(drop=True)
        if candidates.empty:
            candidates = _derived_yahoo_research_candidates(root, max_pairs=max_pairs)
    if candidates.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        rows: list[dict[str, object]] = []
        for _, row in candidates.iterrows():
            asset_x = str(row.get("asset_x", "") or "")
            asset_y = str(row.get("asset_y", "") or "")
            command = (
                "PYTHONPATH=src python -m quant_platform.cli refresh-yahoo-crypto-pair-history "
                f"--asset-x {asset_x} --asset-y {asset_y} --interval {interval} --range {lookback_range} --zscore-window {zscore_window}"
            )
            try:
                path = refresh_yahoo_crypto_pair_history(
                    asset_x=asset_x,
                    asset_y=asset_y,
                    interval=interval,
                    lookback_range=lookback_range,
                    zscore_window=zscore_window,
                )
                rows.append(
                    {
                        "rank": int(row.get("rank", 0) or 0),
                        "pair": str(row.get("pair", "") or ""),
                        "wizard_exchange": str(row.get("wizard_exchange", "") or ""),
                        "asset_x": asset_x,
                        "asset_y": asset_y,
                        "readiness_status": str(row.get("readiness_status", "") or ""),
                        "refresh_status": "refreshed",
                        "history_path": str(path),
                        "error": "",
                        "refresh_command": command,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "rank": int(row.get("rank", 0) or 0),
                        "pair": str(row.get("pair", "") or ""),
                        "wizard_exchange": str(row.get("wizard_exchange", "") or ""),
                        "asset_x": asset_x,
                        "asset_y": asset_y,
                        "readiness_status": str(row.get("readiness_status", "") or ""),
                        "refresh_status": "failed",
                        "history_path": "",
                        "error": str(exc),
                        "refresh_command": command,
                    }
                )
        frame = pd.DataFrame(rows, columns=columns)
    active = root / "reports" / "active"
    csv_path = active / "yahoo_crypto_research_refresh.csv"
    md_path = active / "yahoo_crypto_research_refresh.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _yahoo_refresh_markdown(frame, interval=interval, lookback_range=lookback_range))
    return CommandResult(
        paths={"yahoo_crypto_research_refresh": csv_path, "yahoo_crypto_research_refresh_md": md_path},
        summary={
            "pairs": len(frame),
            "refreshed": int(frame["refresh_status"].astype(str).eq("refreshed").sum()) if not frame.empty else 0,
            "failed": int(frame["refresh_status"].astype(str).eq("failed").sum()) if not frame.empty else 0,
        },
    )


def backfill_yahoo_crypto_funding(
    *,
    funding_path: Path | str,
    root: Path = ROOT,
    pair_dir: Path | str | None = None,
) -> CommandResult:
    funding_file = Path(funding_path)
    if not funding_file.exists():
        raise ValueError(f"funding file not found: {funding_file}")
    funding_rows = normalize_funding_rows(pd.read_csv(funding_file))
    if funding_rows.empty:
        raise ValueError(f"no normalized funding rows found in {funding_file}")
    base = Path(pair_dir or root / "data" / "raw" / "pair_details")
    rows: list[dict[str, object]] = []
    for path in sorted(base.glob("*yahoo_crypto_1d_derived_history.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        history = payload.get("history", [])
        pair = str(payload.get("pair") or "")
        if not pair or not isinstance(history, list) or not history:
            continue
        dataset = PairDataset(pair, pd.DataFrame(history))
        enriched = enrich_pair_dataset_with_funding(dataset, funding_rows)
        frame = enriched.frame.copy()
        left_asset = str(payload.get("asset_x") or "").strip() or pair.split("-")[0]
        right_asset = str(payload.get("asset_y") or "").strip() or pair.split("-")[-2]
        left_market = left_asset if left_asset.endswith("-USD") else f"{left_asset}-USD"
        right_market = right_asset if right_asset.endswith("-USD") else f"{right_asset}-USD"
        _fill_missing_funding_column(frame, funding_rows, left_market, "funding_x_bps")
        _fill_missing_funding_column(frame, funding_rows, right_market, "funding_y_bps")
        funding_x_ready = "funding_x_bps" in frame.columns and frame["funding_x_bps"].notna().any()
        funding_y_ready = "funding_y_bps" in frame.columns and frame["funding_y_bps"].notna().any()
        if funding_x_ready and funding_y_ready:
            payload["history"] = frame.where(pd.notna(frame), None).to_dict("records")
            path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(
                {
                    "pair": pair,
                    "path": str(path),
                    "status": "backfilled",
                    "funding_x_ready": True,
                    "funding_y_ready": True,
                }
            )
        else:
            rows.append(
                {
                    "pair": pair,
                    "path": str(path),
                    "status": "missing_markets",
                    "funding_x_ready": bool(funding_x_ready),
                    "funding_y_ready": bool(funding_y_ready),
                }
            )
    frame = pd.DataFrame(rows, columns=["pair", "path", "status", "funding_x_ready", "funding_y_ready"])
    active = root / "reports" / "active"
    csv_path = active / "yahoo_crypto_funding_backfill.csv"
    md_path = active / "yahoo_crypto_funding_backfill.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _yahoo_funding_backfill_markdown(frame, funding_file))
    return CommandResult(
        paths={"yahoo_crypto_funding_backfill": csv_path, "yahoo_crypto_funding_backfill_md": md_path},
        summary={
            "pairs": len(frame),
            "backfilled": int(frame["status"].astype(str).eq("backfilled").sum()) if not frame.empty else 0,
            "missing_markets": int(frame["status"].astype(str).eq("missing_markets").sum()) if not frame.empty else 0,
        },
    )


def _fill_missing_funding_column(frame: pd.DataFrame, funding_rows: pd.DataFrame, market: str, column: str) -> None:
    market_rows = funding_rows[funding_rows["market"].astype(str) == str(market)]
    if market_rows.empty:
        return
    fallback = float(pd.to_numeric(market_rows["funding_bps"], errors="coerce").dropna().iloc[-1])
    if column not in frame.columns:
        frame[column] = fallback
        return
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame[column] = frame[column].fillna(fallback)


def _derived_yahoo_research_candidates(root: Path, max_pairs: int) -> pd.DataFrame:
    lanes = _read_csv(root / "reports" / "active" / "venue_lane_classification.csv")
    if lanes.empty:
        return pd.DataFrame()
    anchors = lanes[lanes.get("best_lane", pd.Series(dtype=str)).astype(str).isin(["dydx_execution_candidate", "dydx_execution_watch"])].copy()
    targets = lanes[lanes.get("best_lane", pd.Series(dtype=str)).astype(str).isin(["hyperliquid_research_candidate", "hyperliquid_watch"])].copy()
    if anchors.empty or targets.empty:
        return pd.DataFrame()
    anchor_assets = [str(v) for v in anchors.get("asset", pd.Series(dtype=str)).dropna().astype(str).tolist() if str(v).strip()]
    target_rows = targets.to_dict("records")
    rows: list[dict[str, object]] = []
    rank = 1
    seen: set[str] = set()
    for target in target_rows:
        target_asset = str(target.get("asset", "") or "").strip().upper()
        if not target_asset:
            continue
        for anchor in anchor_assets:
            anchor_asset = str(anchor or "").strip().upper()
            if not anchor_asset or anchor_asset == target_asset:
                continue
            pair_key = f"{anchor_asset}/{target_asset}"
            if pair_key in seen:
                continue
            seen.add(pair_key)
            rows.append(
                {
                    "rank": rank,
                    "pair": f"{anchor_asset}-USD/{target_asset}-USD",
                    "wizard_exchange": "binance",
                    "asset_x": f"{anchor_asset}-USD",
                    "asset_y": f"{target_asset}-USD",
                    "readiness_status": "derived_research_fetch",
                }
            )
            rank += 1
            if len(rows) >= max_pairs:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def _yahoo_symbol_status(root: Path, symbol: str, exchange: str) -> dict[str, object]:
    path = root / "data" / "raw" / "yahoo_crypto_candles" / f"{symbol}_1d_candles.json"
    rows = _candle_count(path)
    status = "history_ready" if rows >= 120 else "missing_yahoo_daily_history"
    return {
        "symbol": symbol,
        "exchange": exchange,
        "status": status,
        "daily_candle_rows": rows,
        "candle_path": _rel(path, root) if path.exists() else "",
        "next_action": "build_yahoo_crypto_pair_history" if status == "history_ready" else "fetch_yahoo_crypto_candles_1d",
    }


def _rewrite_pair_history_as_yahoo(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exchange"] = "yahoo_crypto"
    payload["source_note"] = (
        "Derived from Yahoo Finance public crypto history. This is research evidence only; "
        "do not treat it as execution-truth or venue-specific slippage, fee, funding, or borrow evidence."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _to_yahoo_symbol(value: object) -> str:
    text = str(value or "").upper().replace("/", "-").replace("_", "-").strip()
    if text.endswith("-USD"):
        return text
    if text.endswith("-USDT"):
        return f"{text[:-5]}-USD"
    if text.endswith("USDT") and "-" not in text and len(text) > 4:
        return f"{text[:-4]}-USD"
    if text.endswith("USD") and "-" not in text and len(text) > 3:
        return f"{text[:-3]}-USD"
    if "-" not in text:
        return f"{text}-USD"
    return text


def _value_at(values: list[Any], idx: int) -> Any:
    return values[idx] if idx < len(values) else None


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _candle_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    candles = payload.get("candles", []) if isinstance(payload, dict) else []
    return len(candles) if isinstance(candles, list) else 0


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _yahoo_lane_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Yahoo Crypto History Readiness\n\nNo research-lane symbols were found.\n"
    counts = frame["status"].value_counts().reset_index()
    counts.columns = ["status", "rows"]
    return "\n".join(
        [
            "# Yahoo Crypto History Readiness",
            "",
            "This report tracks whether research-only crypto venue assets have local Yahoo daily candle history ready for pair replay.",
            "",
            "## Status Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Symbols",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )


def _yahoo_refresh_markdown(frame: pd.DataFrame, *, interval: str, lookback_range: str) -> str:
    if frame.empty:
        return "# Yahoo Crypto Research Refresh\n\nNo ready-to-fetch research candidates were found.\n"
    return "\n".join(
        [
            "# Yahoo Crypto Research Refresh",
            "",
            f"Refreshed research-lane pair history using Yahoo crypto candles with interval `{interval}` and range `{lookback_range}`.",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )


def _yahoo_funding_backfill_markdown(frame: pd.DataFrame, funding_file: Path) -> str:
    if frame.empty:
        return "# Yahoo Crypto Funding Backfill\n\nNo Yahoo crypto pair-history files were found.\n"
    return "\n".join(
        [
            "# Yahoo Crypto Funding Backfill",
            "",
            f"Backfilled Yahoo crypto pair-history funding columns from `{funding_file}`.",
            "",
            frame.to_markdown(index=False),
            "",
        ]
    )
