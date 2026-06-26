from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.dydx_candles import build_pair_history_from_candles


BINANCE_API_URL = "https://api.binance.com"
BINANCE_FALLBACK_API_URLS = ("https://data-api.binance.vision",)
SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}


def fetch_binance_spot_candles(
    *,
    symbol: str,
    interval: str = "1d",
    limit: int = 1000,
    output_dir: str | Path | None = None,
    base_url: str = BINANCE_API_URL,
    timeout: int = 30,
) -> Path:
    clean_symbol = _clean_symbol(symbol)
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported Binance interval: {interval}")
    params = {"symbol": clean_symbol, "interval": interval, "limit": limit}
    payload, used_base_url = _fetch_klines_with_fallbacks(
        base_url=base_url,
        params=params,
        timeout=timeout,
    )
    candles = normalize_binance_spot_klines(payload, symbol=clean_symbol, interval=interval)
    if not candles:
        raise ValueError(f"no Binance candles returned for {clean_symbol} {interval}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "binance_spot_candles")
    output_base.mkdir(parents=True, exist_ok=True)
    output = output_base / f"{clean_symbol}_{interval}_candles.json"
    output.write_text(
        json.dumps(
            {
                "source": "binance_spot",
                "base_url": used_base_url,
                "request": params,
                "candles": candles,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def _fetch_klines_with_fallbacks(*, base_url: str, params: dict[str, object], timeout: int) -> tuple[Any, str]:
    urls = [base_url, *BINANCE_FALLBACK_API_URLS]
    last_error: Exception | None = None
    for candidate in urls:
        try:
            response = requests.get(f"{candidate.rstrip('/')}/api/v3/klines", params=params, timeout=timeout)
            response.raise_for_status()
            return response.json(), candidate
        except requests.HTTPError as exc:
            last_error = exc
            if exc.response is not None and exc.response.status_code in {451, 403, 418, 429}:
                continue
            raise
        except requests.RequestException as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("Binance kline fetch failed without an error")


def normalize_binance_spot_klines(payload: Any, *, symbol: str, interval: str) -> list[dict[str, object]]:
    rows = payload if isinstance(payload, list) else payload.get("klines", []) if isinstance(payload, dict) else []
    candles: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 8:
            continue
        timestamp = datetime.fromtimestamp(float(row[0]) / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")
        close = _safe_float(row[4])
        quote_volume = _safe_float(row[7])
        base_volume = _safe_float(row[5])
        candles.append(
            {
                "startedAt": timestamp,
                "ticker": _clean_symbol(symbol),
                "resolution": interval,
                "open": _safe_float(row[1]),
                "high": _safe_float(row[2]),
                "low": _safe_float(row[3]),
                "close": close,
                "baseVolume": base_volume,
                "usdVolume": quote_volume if quote_volume > 0 else base_volume * close,
                "source": "binance_spot",
            }
        )
    return sorted(candles, key=lambda candle: str(candle["startedAt"]))


def build_binance_spot_pair_history(
    *,
    asset_x: str,
    asset_y: str,
    interval: str = "1d",
    pair_id: str | None = None,
    hedge_ratio: float | None = None,
    beta: float | None = None,
    zscore_window: int = 320,
    candle_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    left_symbol = _clean_symbol(asset_x)
    right_symbol = _clean_symbol(asset_y)
    candle_base = Path(candle_dir or ROOT / "data" / "raw" / "binance_spot_candles")
    left_path = candle_base / f"{left_symbol}_{interval}_candles.json"
    right_path = candle_base / f"{right_symbol}_{interval}_candles.json"
    missing = [str(path) for path in [left_path, right_path] if not path.exists()]
    if missing:
        raise ValueError(f"missing Binance candle files: {missing}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "pair_details")
    clean_pair_id = pair_id or f"{left_symbol.lower()}_{right_symbol.lower()}_binance_spot_{interval}"
    output = output_base / f"pair_{_safe_filename(clean_pair_id)}_binance_spot_{interval}_derived_history.json"
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
        min_zscore_window=min(20, max(2, zscore_window // 4)),
    )
    _rewrite_pair_history_as_binance(path)
    return path


def build_binance_spot_lane_report(root: Path = ROOT) -> CommandResult:
    readiness = _read_csv(root / "reports" / "active" / "multi_venue_history_readiness_2026-06-25.csv")
    if readiness.empty:
        frame = pd.DataFrame(columns=["symbol", "status", "daily_candle_rows", "candle_path", "next_action"])
    else:
        symbols = sorted(
            {
                _clean_symbol(value)
                for col in ["asset_x", "asset_y"]
                for value in readiness[readiness["wizard_exchange"].astype(str) == "binance"].get(col, [])
                if str(value or "").strip()
            }
        )
        rows = [_binance_symbol_status(root, symbol) for symbol in symbols]
        frame = pd.DataFrame(rows)
    active = root / "reports" / "active"
    csv_path = active / "binance_spot_history_readiness.csv"
    md_path = active / "binance_spot_history_readiness.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _binance_lane_markdown(frame))
    return CommandResult(
        paths={"binance_spot_history_readiness": csv_path, "binance_spot_history_readiness_md": md_path},
        summary={
            "symbols": len(frame),
            "ready_symbols": int(frame["status"].astype(str).eq("history_ready").sum()) if not frame.empty else 0,
            "blocked_symbols": int(frame["status"].astype(str).ne("history_ready").sum()) if not frame.empty else 0,
        },
    )


def _binance_symbol_status(root: Path, symbol: str) -> dict[str, object]:
    path = root / "data" / "raw" / "binance_spot_candles" / f"{symbol}_1d_candles.json"
    rows = _candle_count(path)
    status = "history_ready" if rows >= 120 else "missing_binance_daily_history"
    return {
        "symbol": symbol,
        "status": status,
        "daily_candle_rows": rows,
        "candle_path": _rel(path, root) if path.exists() else "",
        "next_action": "build_binance_pair_history" if status == "history_ready" else "fetch_binance_spot_candles_1d",
    }


def _rewrite_pair_history_as_binance(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exchange"] = "binance_spot"
    payload["source_note"] = (
        "Derived from Binance public spot klines. This is research evidence only; do not promote until "
        "Binance-specific fees, slippage, and borrow or shortability assumptions are merged and tested."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _binance_lane_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Binance Spot History Readiness\n\nNo Binance symbols were found.\n"
    counts = frame["status"].value_counts().reset_index()
    counts.columns = ["status", "rows"]
    return "\n".join(
        [
            "# Binance Spot History Readiness",
            "",
            "This report tracks whether Binance-routed Wizard assets have local daily candle history ready for pair replay.",
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


def _candle_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    candles = payload.get("candles", []) if isinstance(payload, dict) else []
    return len(candles) if isinstance(candles, list) else 0


def _clean_symbol(value: object) -> str:
    return str(value or "").upper().replace("-", "").replace("/", "").strip()


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


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
