from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_LONG = ROOT / "data" / "raw" / "dydx_long_history"
RAW_SINGLE = ROOT / "data" / "raw" / "dydx_candles"
PROCESSED = ROOT / "data" / "processed" / "evidence_pipeline"
REPORTS = ROOT / "reports" / "evidence_pipeline"

PAIRS = {
    "ETH-SOL": ("eth_sol", "ETH-USD", "SOL-USD"),
    "BTC-DOGE": ("btc_doge", "BTC-USD", "DOGE-USD"),
    "BTC-SOL": ("btc_sol", "BTC-USD", "SOL-USD"),
    "ETH-LINK": ("eth_link", "ETH-USD", "LINK-USD"),
    "DOGE-XRP": ("doge_xrp", "DOGE-USD", "XRP-USD"),
}

TIMEFRAMES = {
    "5m": "5MINS",
    "15m": "15MINS",
    "1h": "1HOUR",
    "4h": "4HOURS",
    "1d": "1DAY",
}


def read_candles(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for candle in data.get("candles", []):
            rows.append(
                {
                    "timestamp": candle.get("startedAt"),
                    "open": candle.get("open"),
                    "high": candle.get("high"),
                    "low": candle.get("low"),
                    "close": candle.get("close"),
                    "usd_volume": candle.get("usdVolume"),
                    "trades": candle.get("trades"),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in ("open", "high", "low", "close", "usd_volume", "trades"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(subset=["timestamp", "close"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")
    )


def long_paths(pair_dir: str, ticker: str, timeframe_api: str) -> list[Path]:
    return sorted((RAW_LONG / pair_dir).glob(f"window_*/{ticker}_{timeframe_api}_candles.json"))


def single_paths(ticker: str, timeframe_api: str) -> list[Path]:
    path = RAW_SINGLE / f"{ticker}_{timeframe_api}_candles.json"
    return [path] if path.exists() else []


def build_pair_frame(
    pair_dir: str,
    asset_x: str,
    asset_y: str,
    timeframe: str,
    timeframe_api: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x_paths = long_paths(pair_dir, asset_x, timeframe_api)
    y_paths = long_paths(pair_dir, asset_y, timeframe_api)
    source_type = "long_history"
    source_quality = "full_pair_long_history"

    if not x_paths or not y_paths:
        x_paths = single_paths(asset_x, timeframe_api)
        y_paths = single_paths(asset_y, timeframe_api)
        source_type = "single_window"
        source_quality = "single_window_built"

    x_frame = read_candles(x_paths)
    y_frame = read_candles(y_paths)
    meta = {
        "timeframe": timeframe,
        "timeframe_api": timeframe_api,
        "source_type": source_type,
        "source_quality": source_quality,
        "asset_x_paths": len(x_paths),
        "asset_y_paths": len(y_paths),
        "asset_x_rows": len(x_frame),
        "asset_y_rows": len(y_frame),
    }
    if x_frame.empty or y_frame.empty:
        meta["source_quality"] = "missing_fetch_required"
        return pd.DataFrame(), meta

    joined = (
        x_frame[["open", "high", "low", "close", "usd_volume", "trades"]]
        .add_prefix("x_")
        .join(y_frame[["open", "high", "low", "close", "usd_volume", "trades"]].add_prefix("y_"), how="inner")
        .dropna(subset=["x_close", "y_close"])
    )
    joined = joined[(joined["x_close"] > 0) & (joined["y_close"] > 0)]
    if joined.empty:
        return joined, meta

    log_x = np.log(joined["x_close"])
    log_y = np.log(joined["y_close"])
    ret_x = log_x.diff()
    ret_y = log_y.diff()
    joined["spread"] = log_x - log_y
    joined["return_x"] = ret_x
    joined["return_y"] = ret_y
    joined["rolling_corr_96"] = ret_x.rolling(96, min_periods=20).corr(ret_y)
    beta = ret_x.rolling(96, min_periods=20).cov(ret_y) / ret_y.rolling(96, min_periods=20).var().replace(0, np.nan)
    joined["beta_96"] = beta.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-5.0, 5.0).abs()
    joined["min_usd_volume"] = joined[["x_usd_volume", "y_usd_volume"]].min(axis=1)
    joined["pair_timeframe"] = timeframe
    joined["source_type"] = source_type
    joined["source_quality"] = meta["source_quality"]
    return joined.reset_index(), meta


def fetch_target(pair: str, asset: str, timeframe: str, timeframe_api: str, reason: str) -> dict[str, object]:
    return {
        "pair": pair,
        "missing_asset": asset,
        "timeframe": timeframe,
        "timeframe_api": timeframe_api,
        "target_file": f"data/raw/dydx_candles/{asset}_{timeframe_api}_candles.json",
        "reason": reason,
        "suggested_action": "fetch_dydx_candles",
    }


def table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    coverage_rows: list[dict[str, object]] = []
    fetch_rows: list[dict[str, object]] = []

    for pair, (pair_dir, asset_x, asset_y) in PAIRS.items():
        for timeframe, timeframe_api in TIMEFRAMES.items():
            frame, meta = build_pair_frame(pair_dir, asset_x, asset_y, timeframe, timeframe_api)
            output_file = ""
            available = len(frame) >= 120
            if available:
                output_file = f"{pair.lower().replace('-', '_')}_{timeframe}_pair_history.csv"
                frame.to_csv(PROCESSED / output_file, index=False)
            else:
                if not single_paths(asset_x, timeframe_api) and not long_paths(pair_dir, asset_x, timeframe_api):
                    fetch_rows.append(fetch_target(pair, asset_x, timeframe, timeframe_api, "asset_x_missing_locally"))
                if not single_paths(asset_y, timeframe_api) and not long_paths(pair_dir, asset_y, timeframe_api):
                    fetch_rows.append(fetch_target(pair, asset_y, timeframe, timeframe_api, "asset_y_missing_locally"))

            coverage_rows.append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "available": available,
                    "source_quality": meta["source_quality"],
                    "source_type": meta["source_type"],
                    "paired_rows": len(frame),
                    "asset_x_rows": meta["asset_x_rows"],
                    "asset_y_rows": meta["asset_y_rows"],
                    "asset_x_paths": meta["asset_x_paths"],
                    "asset_y_paths": meta["asset_y_paths"],
                    "processed_file": output_file,
                    "blocker": "" if available else "missing_or_insufficient_local_history",
                }
            )

    coverage = pd.DataFrame(coverage_rows)
    fetch_targets = pd.DataFrame(fetch_rows).drop_duplicates() if fetch_rows else pd.DataFrame()
    coverage.to_csv(REPORTS / "phase2_pair_coverage_build_inventory.csv", index=False)
    fetch_targets.to_csv(REPORTS / "phase2_missing_fetch_targets.csv", index=False)

    quality_summary = (
        coverage.groupby(["source_quality", "available"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["available", "source_quality"], ascending=[False, True])
    )

    coverage_columns = [
        "pair",
        "timeframe",
        "available",
        "source_quality",
        "paired_rows",
        "asset_x_rows",
        "asset_y_rows",
        "processed_file",
        "blocker",
    ]
    fetch_columns = ["pair", "missing_asset", "timeframe", "target_file", "reason", "suggested_action"]
    report = [
        "# Evidence Pipeline Phase 2 Coverage Build",
        "",
        "Local coverage build for the required five-pair, five-timeframe matrix.",
        "",
        "## Summary",
        "",
        f"- Required pair/timeframe cells: {len(coverage):,}",
        f"- Built/available cells: {int(coverage['available'].sum()):,}",
        f"- Missing cells: {int((~coverage['available']).sum()):,}",
        f"- Full pair long-history cells: {int((coverage['source_quality'] == 'full_pair_long_history').sum()):,}",
        f"- Local single-window built cells: {int((coverage['source_quality'] == 'single_window_built').sum()):,}",
        f"- Fetch-required cells: {int((coverage['source_quality'] == 'missing_fetch_required').sum()):,}",
        "",
        "## Source Quality Summary",
        "",
        table(quality_summary, list(quality_summary.columns)),
        "",
        "## Coverage Matrix",
        "",
        table(coverage, coverage_columns),
        "",
        "## Missing Fetch Targets",
        "",
        table(fetch_targets, fetch_columns) if not fetch_targets.empty else "_No fetch targets remain._",
        "",
        "## Notes",
        "",
        "- `full_pair_long_history` means pair-specific long-history windows already existed locally.",
        "- `single_window_built` means the pair history was built from local standalone asset candle files; usable for research, but weaker than pair-specific long-history coverage.",
        "- `missing_fetch_required` means the local files needed to build that pair/timeframe are absent.",
    ]
    (REPORTS / "phase2_pair_coverage_build_report.md").write_text("\n".join(report) + "\n")

    print(
        json.dumps(
            {
                "coverage_cells": int(len(coverage)),
                "available_cells": int(coverage["available"].sum()),
                "missing_cells": int((~coverage["available"]).sum()),
                "processed_dir": str(PROCESSED),
                "report": str(REPORTS / "phase2_pair_coverage_build_report.md"),
                "coverage_csv": str(REPORTS / "phase2_pair_coverage_build_inventory.csv"),
                "fetch_targets_csv": str(REPORTS / "phase2_missing_fetch_targets.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
