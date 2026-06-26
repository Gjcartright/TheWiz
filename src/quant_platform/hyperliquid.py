from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.dydx_candles import build_pair_history_from_candles


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"}


def fetch_hyperliquid_candles(
    *,
    coin: str,
    interval: str = "1d",
    days: int = 500,
    output_dir: str | Path | None = None,
    end_time: datetime | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
) -> Path:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported Hyperliquid interval: {interval}")
    clean_coin = _coin(coin)
    end_dt = end_time or datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    body = {
        "type": "candleSnapshot",
        "req": {
            "coin": clean_coin,
            "interval": interval,
            "startTime": int(start_dt.timestamp() * 1000),
            "endTime": int(end_dt.timestamp() * 1000),
        },
    }
    response = requests.post(info_url, json=body, headers={"Content-Type": "application/json"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    candles = normalize_hyperliquid_candles(payload, coin=clean_coin, interval=interval)
    if not candles:
        raise ValueError(f"no Hyperliquid candles returned for {clean_coin} {interval}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "hyperliquid_candles")
    output_base.mkdir(parents=True, exist_ok=True)
    output = output_base / f"{clean_coin}_{interval}_candles.json"
    output.write_text(
        json.dumps(
            {
                "source": "hyperliquid",
                "info_url": info_url,
                "request": body,
                "candles": candles,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def normalize_hyperliquid_candles(payload: Any, *, coin: str, interval: str) -> list[dict[str, object]]:
    rows = payload if isinstance(payload, list) else payload.get("candles", []) if isinstance(payload, dict) else []
    candles: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp_ms = row.get("t") or row.get("time") or row.get("timestamp")
        if timestamp_ms is None:
            continue
        timestamp = datetime.fromtimestamp(float(timestamp_ms) / 1000.0, timezone.utc).isoformat().replace("+00:00", "Z")
        volume = _safe_float(row.get("v"))
        close = _safe_float(row.get("c"))
        candles.append(
            {
                "startedAt": timestamp,
                "ticker": f"{_coin(coin)}-USD",
                "resolution": interval,
                "open": _safe_float(row.get("o")),
                "high": _safe_float(row.get("h")),
                "low": _safe_float(row.get("l")),
                "close": close,
                "baseVolume": volume,
                "usdVolume": volume * close,
                "source": "hyperliquid",
            }
        )
    return sorted(candles, key=lambda candle: str(candle["startedAt"]))


def build_hyperliquid_pair_history(
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
    left_coin = _coin(asset_x)
    right_coin = _coin(asset_y)
    candle_base = Path(candle_dir or ROOT / "data" / "raw" / "hyperliquid_candles")
    left_path = candle_base / f"{left_coin}_{interval}_candles.json"
    right_path = candle_base / f"{right_coin}_{interval}_candles.json"
    missing = [str(path) for path in [left_path, right_path] if not path.exists()]
    if missing:
        raise ValueError(f"missing Hyperliquid candle files: {missing}")
    output_base = Path(output_dir or ROOT / "data" / "raw" / "pair_details")
    clean_pair_id = pair_id or f"{left_coin.lower()}_{right_coin.lower()}_hyperliquid_{interval}"
    output = output_base / f"pair_{_safe_filename(clean_pair_id)}_hyperliquid_{interval}_derived_history.json"
    path = build_pair_history_from_candles(
        left_path=left_path,
        right_path=right_path,
        output_path=output,
        pair_id=clean_pair_id,
        asset_x=f"{left_coin}-USD",
        asset_y=f"{right_coin}-USD",
        hedge_ratio=hedge_ratio,
        beta=beta,
        interval=interval,
        zscore_window=zscore_window,
        min_zscore_window=min(20, max(2, zscore_window // 4)),
    )
    _rewrite_pair_history_as_hyperliquid(path)
    return path


def build_hyperliquid_lane_report(root: Path = ROOT) -> CommandResult:
    lanes = _read_csv(root / "reports" / "active" / "venue_lane_classification.csv")
    if lanes.empty:
        rows = []
    else:
        rows = [_hyperliquid_lane_row(root, row) for _, row in lanes.iterrows()]
        rows = [row for row in rows if row["hyperliquid_lane"]]
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["status", "asset"]).reset_index(drop=True)
    active = root / "reports" / "active"
    csv_path = active / "hyperliquid_lane_readiness.csv"
    md_path = active / "hyperliquid_lane_readiness.md"
    _write_csv(frame, csv_path)
    _write_text(md_path, _hyperliquid_lane_markdown(frame))
    return CommandResult(
        paths={"hyperliquid_lane_readiness": csv_path, "hyperliquid_lane_readiness_md": md_path},
        summary={
            "assets": len(frame),
            "ready_assets": int(frame["status"].astype(str).eq("history_ready").sum()) if not frame.empty else 0,
            "blocked_assets": int(frame["status"].astype(str).ne("history_ready").sum()) if not frame.empty else 0,
        },
    )


def _hyperliquid_lane_row(root: Path, row: pd.Series) -> dict[str, object]:
    asset = str(row.get("asset", "") or "")
    coin = _coin(asset)
    lane = str(row.get("hyperliquid_lane", "") or "")
    candle_dir = root / "data" / "raw" / "hyperliquid_candles"
    daily_path = candle_dir / f"{coin}_1d_candles.json"
    rows = _candle_count(daily_path)
    status = "history_ready" if rows >= 120 else "missing_hyperliquid_daily_history"
    return {
        "asset": asset,
        "coin": coin,
        "hyperliquid_lane": lane,
        "status": status,
        "daily_candle_rows": rows,
        "candle_path": _rel(daily_path, root) if daily_path.exists() else "",
        "blocker": "" if status == "history_ready" else "fetch_hyperliquid_candles_1d",
        "next_action": "build_pair_history_for_hyperliquid_candidates" if status == "history_ready" else "fetch_hyperliquid_daily_candles",
    }


def _rewrite_pair_history_as_hyperliquid(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["exchange"] = "hyperliquid"
    payload["source_note"] = (
        "Derived from Hyperliquid candleSnapshot candles. This is Hyperliquid research evidence only; "
        "do not use it to promote dYdX execution. Funding and slippage must be merged from Hyperliquid-specific sources."
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _hyperliquid_lane_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Hyperliquid Lane Readiness\n\nNo Hyperliquid lane assets were found.\n"
    counts = frame["status"].value_counts().reset_index()
    counts.columns = ["status", "rows"]
    return "\n".join(
        [
            "# Hyperliquid Lane Readiness",
            "",
            "This report tracks whether Hyperliquid-routed assets have local daily candle history ready for pair replay.",
            "",
            "## Status Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Assets",
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


def _coin(value: str) -> str:
    text = str(value or "").upper().strip()
    if text.endswith("-USD"):
        text = text[:-4]
    if text.endswith("/USD"):
        text = text[:-4]
    if text.endswith("USD") and "-" not in text and "/" not in text and len(text) > 3:
        text = text[:-3]
    return text.replace("_", "-")


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
