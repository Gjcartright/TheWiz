from __future__ import annotations

import json
import os
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from quant_platform.funding import normalize_funding_rows

DYDX_INDEXER_BASE = os.getenv("QPA_INDEXER_BASE", "https://indexer.dydx.trade").strip()
MAX_DYDX_REQUEST_LIMIT = 1000
MIN_DYDX_REQUEST_LIMIT = 1


def _clamp_request_limit(limit: int) -> int:
    try:
        sanitized = int(float(limit)) if isinstance(limit, str) else int(limit)
    except (TypeError, ValueError):
        return MAX_DYDX_REQUEST_LIMIT
    if sanitized < MIN_DYDX_REQUEST_LIMIT:
        return MIN_DYDX_REQUEST_LIMIT
    if sanitized > MAX_DYDX_REQUEST_LIMIT:
        return MAX_DYDX_REQUEST_LIMIT
    return sanitized


def dydx_two_leg_request_rows(
    *,
    asset_x: str,
    asset_y: str,
    pair_id: str = "manual",
    hedge_ratio: float = 1.0,
    beta: float | None = None,
    resolution: str = "5MINS",
    limit: int = 100,
    to_iso: str | None = None,
    from_iso: str | None = None,
    indexer_base: str = DYDX_INDEXER_BASE,
    output_dir: str | Path = "data/raw/dydx_manual",
    zscore_window: int = 320,
) -> list[dict[str, str]]:
    left = _dydx_market(asset_x)
    right = _dydx_market(asset_y)
    output_base = Path(output_dir)
    rows: list[dict[str, str]] = []
    for leg_name, market in (("asset_x", left), ("asset_y", right)):
        candle_path = output_base / f"{market}_{resolution}_candles.json"
        rows.append(
            _request_template_row(
                name=f"{leg_name}_candles_{resolution.lower()}",
                url=_dydx_indexer_url(
                    indexer_base,
                    f"/v4/candles/perpetualMarkets/{market}",
                    {
                        "resolution": resolution,
                        "fromISO": from_iso,
                        "toISO": to_iso,
                        "limit": _clamp_request_limit(limit),
                    },
                ),
                save_as=candle_path,
                import_command=(
                    "PYTHONPATH=src python3 -m quant_platform.cli import-dydx-candles "
                    f"--json-path {candle_path}"
                ),
                notes=f"Download {resolution} candles for {market}; this supplies {'price_x' if leg_name == 'asset_x' else 'price_y'}.",
            )
        )
        funding_path = output_base / f"{market}_funding.json"
        rows.append(
            _request_template_row(
                name=f"{leg_name}_historical_funding",
                url=_dydx_indexer_url(
                    indexer_base,
                    f"/v4/historicalFunding/{market}",
                    {"limit": _clamp_request_limit(limit)},
                ),
                save_as=funding_path,
                import_command=(
                    "PYTHONPATH=src python3 -m quant_platform.cli export-dydx-funding "
                    f"--json-path {funding_path} --market {market} --output-path data/processed/dydx_funding.csv"
                ),
                notes=f"Download historical funding for {market}; combine both legs before running funding coverage.",
            )
        )

    beta_value = beta if beta is not None else hedge_ratio
    rows.append(
        {
            "request_name": "build_two_leg_pair_history",
            "method": "LOCAL",
            "url": "",
            "curl": "",
            "save_as": str(Path("data/raw/pair_details") / f"pair_{pair_id}_{resolution.lower()}_dydx_candles_derived_history.json"),
            "import_command": (
                "PYTHONPATH=src python3 -m quant_platform.cli build-dydx-pair-history "
                f"--left-candles data/raw/dydx_candles/{left}_{resolution}_candles.json "
                f"--right-candles data/raw/dydx_candles/{right}_{resolution}_candles.json "
                f"--asset-x {left} --asset-y {right} --pair-id {pair_id} --interval {resolution.lower()} "
                f"--hedge-ratio {hedge_ratio} --beta {beta_value} --zscore-window {zscore_window}"
            ),
            "notes": "Run after both candle imports. Funding is merged later with --funding-path, not fabricated into this file.",
        }
    )
    rows.append(
        {
            "request_name": "merge_funding_and_rerun_research",
            "method": "LOCAL",
            "url": "",
            "curl": "",
            "save_as": "data/processed/dydx_funding.csv",
            "import_command": (
                "PYTHONPATH=src python3 -m quant_platform.cli export-dydx-funding "
                f"--json-path {output_base} --output-path data/processed/dydx_funding.csv && "
                "PYTHONPATH=src python3 -m quant_platform.cli funded-research-spine "
                "--funding-path data/processed/dydx_funding.csv"
            ),
            "notes": "Normalize all saved funding payloads, verify funding coverage, and rerun the guarded P2 research path.",
        }
    )
    return rows


def load_loose_candle_payload(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = _parse_loose_candle_text(text)
    candles = _candles_from_payload(payload)
    if not candles:
        raise ValueError(f"no candles found in {path}")
    return candles


def archive_dydx_candles(input_path: str | Path, output_dir: str | Path) -> Path:
    candles = load_loose_candle_payload(input_path)
    ticker = str(candles[0].get("ticker") or "UNKNOWN")
    resolution = str(candles[0].get("resolution") or "UNKNOWN")
    output = Path(output_dir) / f"{ticker}_{resolution}_candles.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"candles": candles}, indent=2, sort_keys=True), encoding="utf-8")
    return output


def merge_dydx_candle_windows(
    *,
    input_dir: str | Path,
    market: str,
    resolution: str = "5MINS",
    output_dir: str | Path,
) -> Path:
    market_name = _dydx_market(market)
    candidates = sorted(Path(input_dir).glob(f"**/{market_name}_{resolution}_candles.json"))
    if not candidates:
        raise ValueError(f"no windowed dYdX candle files found for {market_name} {resolution} in {input_dir}")

    by_timestamp: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        for candle in load_loose_candle_payload(candidate):
            timestamp = _timestamp(candle)
            if not timestamp:
                continue
            by_timestamp[timestamp] = candle
    if not by_timestamp:
        raise ValueError(f"no timestamped candles found for {market_name} {resolution} in {input_dir}")

    candles = [by_timestamp[timestamp] for timestamp in sorted(by_timestamp)]
    output = Path(output_dir) / f"{market_name}_{resolution}_candles.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"candles": candles}, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_pair_history_from_windowed_candles(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    pair_output_dir: str | Path,
    pair_id: str,
    asset_x: str,
    asset_y: str,
    hedge_ratio: float | None,
    beta: float | None = None,
    resolution: str = "5MINS",
    interval: str | None = None,
    zscore_window: int = 320,
    derive_hedge_ratio: bool = False,
    funding_path: str | Path | None = None,
) -> dict[str, Path]:
    left = _dydx_market(asset_x)
    right = _dydx_market(asset_y)
    left_path = merge_dydx_candle_windows(
        input_dir=input_dir,
        market=left,
        resolution=resolution,
        output_dir=output_dir,
    )
    right_path = merge_dydx_candle_windows(
        input_dir=input_dir,
        market=right,
        resolution=resolution,
        output_dir=output_dir,
    )
    pair_path = (
        Path(pair_output_dir)
        / f"pair_{_safe_filename(pair_id)}_{resolution.lower()}_dydx_long_history_derived_history.json"
    )
    build_pair_history_from_candles(
        left_path=left_path,
        right_path=right_path,
        output_path=pair_path,
        pair_id=pair_id,
        asset_x=left,
        asset_y=right,
        hedge_ratio=None if derive_hedge_ratio else hedge_ratio,
        beta=None if derive_hedge_ratio else beta,
        interval=interval or resolution.lower(),
        zscore_window=zscore_window,
        funding_path=funding_path,
    )
    return {"left_candles": left_path, "right_candles": right_path, "pair_history": pair_path}


def build_pair_history_from_candles(
    *,
    left_path: str | Path,
    right_path: str | Path,
    output_path: str | Path,
    pair_id: str,
    asset_x: str,
    asset_y: str,
    hedge_ratio: float | None,
    beta: float | None = None,
    interval: str | None = None,
    zscore_window: int = 320,
    min_zscore_window: int = 20,
    derive_ecm: bool = True,
    funding_path: str | Path | None = None,
    funding_rows: pd.DataFrame | None = None,
) -> Path:
    left = load_loose_candle_payload(left_path)
    right = load_loose_candle_payload(right_path)
    left_by_time = {_timestamp(row): row for row in left if _timestamp(row)}
    right_by_time = {_timestamp(row): row for row in right if _timestamp(row)}
    timestamps = sorted(set(left_by_time).intersection(right_by_time))
    if not timestamps:
        raise ValueError("no overlapping candle timestamps")

    price_pairs = [(_candle_price(left_by_time[timestamp]), _candle_price(right_by_time[timestamp])) for timestamp in timestamps]
    final_hedge_ratio = hedge_ratio if hedge_ratio is not None else _estimate_price_hedge_ratio(price_pairs)
    final_beta = beta if beta is not None else _estimate_return_beta(price_pairs)

    rows: list[dict[str, Any]] = []
    for timestamp in timestamps:
        x = left_by_time[timestamp]
        y = right_by_time[timestamp]
        price_x = _candle_price(x)
        price_y = _candle_price(y)
        spread = price_x - final_hedge_ratio * price_y
        rows.append(
            {
                "timestamp": timestamp,
                "price_x": price_x,
                "price_y": price_y,
                "spread": spread,
                "hedge_ratio": final_hedge_ratio,
                "beta": final_beta,
                "volume_x_usd": _safe_float(x.get("usdVolume"), 0.0),
                "volume_y_usd": _safe_float(y.get("usdVolume"), 0.0),
            }
        )

    _attach_zscores(rows, zscore_window=zscore_window, min_window=min_zscore_window)
    ecm_derivation: dict[str, Any] | None = None
    if derive_ecm:
        ecm_derivation = _attach_provisional_ecm(rows)
    _attach_provisional_research_features(rows)
    if funding_path is not None or funding_rows is not None:
        _attach_funding_to_rows(
            rows,
            funding_path,
            funding_rows=funding_rows,
            asset_x=asset_x,
            asset_y=asset_y,
        )

    resolution = interval or str(left[0].get("resolution") or right[0].get("resolution") or "unknown").lower()
    payload: dict[str, Any] = {
        "pair_id": pair_id,
        "pair": f"{asset_x}-{asset_y}",
        "asset_x": asset_x,
        "asset_y": asset_y,
        "exchange": "dydx",
        "interval": resolution.lower(),
        "period": len(rows),
        "strategy_mode": "static",
        "hedge_ratio": final_hedge_ratio,
        "hedge_ratio_source": "operator" if hedge_ratio is not None else "derived_price_ols",
        "beta": final_beta,
        "beta_source": "operator" if beta is not None else "derived_return_covariance",
        "ecm_x_available": bool(derive_ecm),
        "ecm_y_available": bool(derive_ecm),
        "ecm_strength_available": bool(derive_ecm),
        "source_note": (
            "Derived from manually copied dYdX candle responses observed on Crypto Wizards pair page. "
            "Spread and zscore are reconstructed locally. Funding is not fabricated; merge real dYdX funding "
            "with --funding-path before production acceptance."
        ),
        "history": rows,
    }
    if ecm_derivation:
        payload["ecm_derivation"] = ecm_derivation
        payload["source_note"] += " ECM fields are provisional derived estimates, not native Crypto Wizards ECM payload values."
    payload["source_note"] += (
        " Additional research columns such as conditional_probability_distortion, half_life, hurst, tail_dependence,"
        " and model confidence inputs are provisional derived features used to exercise the strategy research harness."
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def backfill_provisional_pair_history_features(
    input_dir: str | Path,
    *,
    pattern: str = "pair_*_derived_history.json",
) -> list[Path]:
    root = Path(input_dir)
    written: list[Path] = []
    for path in sorted(root.glob(pattern)):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = payload.get("history")
        if not isinstance(history, list) or not history:
            continue
        rows = [row for row in history if isinstance(row, dict)]
        if not rows:
            continue
        _attach_provisional_research_features(rows, overwrite=True)
        payload["history"] = rows
        if "source_note" in payload:
            source_note = str(payload.get("source_note") or "")
            if "provisional derived features used to exercise the strategy research harness" not in source_note:
                payload["source_note"] = source_note + (
                    " Additional research columns such as conditional_probability_distortion, half_life, hurst,"
                    " tail_dependence, and model confidence inputs are provisional derived features used to exercise"
                    " the strategy research harness."
                )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        written.append(path)
    return written


def _attach_funding_to_rows(
    rows: list[dict[str, Any]],
    funding_path: str | Path,
    *,
    funding_rows: pd.DataFrame | None = None,
    asset_x: str,
    asset_y: str,
) -> None:
    if funding_rows is None:
        funding_file = Path(funding_path)
        if not funding_file.exists():
            return
        funding = normalize_funding_rows(pd.read_csv(funding_file))
    else:
        funding = funding_rows
    if funding.empty:
        return
    _attach_leg_funding_to_rows(rows, funding, _dydx_market(asset_x), "funding_x_bps")
    _attach_leg_funding_to_rows(rows, funding, _dydx_market(asset_y), "funding_y_bps")


def _attach_leg_funding_to_rows(
    rows: list[dict[str, Any]],
    funding: pd.DataFrame,
    market: str,
    output_column: str,
) -> None:
    if output_column in rows[0]:
        return
    market_funding = funding[funding["market"] == market].copy()
    if market_funding.empty:
        return
    if "timestamp" not in rows[0] or market_funding["timestamp"].isna().all():
        latest = market_funding["funding_bps"].dropna()
        if not latest.empty:
            for row in rows:
                row[output_column] = float(latest.iloc[-1])
        return

    frame = pd.DataFrame(rows).reset_index().rename(columns={"index": "_row_index"})
    frame["_timestamp"] = (
        pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        .astype("datetime64[ns, UTC]")
    )
    right = market_funding.dropna(subset=["timestamp"]).copy()
    right["_timestamp"] = (
        pd.to_datetime(right["timestamp"], errors="coerce", utc=True)
        .astype("datetime64[ns, UTC]")
    ).sort_values()
    if right.empty:
        latest = market_funding["funding_bps"].dropna()
        if not latest.empty:
            for row in rows:
                row[output_column] = float(latest.iloc[-1])
        return

    frame["_timestamp"] = (
        pd.to_datetime(frame["_timestamp"], utc=True, errors="coerce")
        .astype("datetime64[ns, UTC]")
    )
    right["_timestamp"] = (
        pd.to_datetime(right["_timestamp"], utc=True, errors="coerce")
        .astype("datetime64[ns, UTC]")
    )

    merged = pd.merge_asof(
        frame.sort_values("_timestamp"),
        right[["_timestamp", "funding_bps"]].sort_values("_timestamp"),
        on="_timestamp",
        direction="backward",
    ).sort_values("_row_index")
    values = pd.Series(merged["funding_bps"]).ffill().bfill().to_list()
    for row, value in zip(rows, values):
        if value is not None and not pd.isna(value):
            row[output_column] = float(value)


def import_dydx_candle_bundle(
    input_path: str | Path,
    *,
    candle_output_dir: str | Path,
    pair_output_dir: str | Path,
    hedge_ratio_by_pair: dict[str, float] | None = None,
    default_hedge_ratio: float = 1.0,
    zscore_window: int = 320,
) -> list[Path]:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    pairs = payload.get("pairs") if isinstance(payload, dict) else None
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("bundle does not contain pairs")

    written: list[Path] = []
    candle_dir = Path(candle_output_dir)
    pair_dir = Path(pair_output_dir)
    candle_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    for index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, dict):
            continue
        asset_x = str(pair.get("asset_x") or "")
        asset_y = str(pair.get("asset_y") or "")
        if not asset_x or not asset_y:
            continue
        pair_id = str(pair.get("pair_id") or index)
        pair_name = str(pair.get("pair") or f"{asset_x}-{asset_y}")
        legs = pair.get("legs")
        if not isinstance(legs, dict):
            continue
        left_candles = _bundle_leg_candles(legs.get("asset_x"))
        right_candles = _bundle_leg_candles(legs.get("asset_y"))
        if not left_candles or not right_candles:
            continue

        left_path = candle_dir / f"{asset_x}_5MINS_candles.json"
        right_path = candle_dir / f"{asset_y}_5MINS_candles.json"
        left_path.write_text(json.dumps({"candles": left_candles}, indent=2, sort_keys=True), encoding="utf-8")
        right_path.write_text(json.dumps({"candles": right_candles}, indent=2, sort_keys=True), encoding="utf-8")

        hedge_ratio = (hedge_ratio_by_pair or {}).get(pair_name, default_hedge_ratio)
        output = pair_dir / f"pair_{_safe_filename(pair_id)}_5mins_{_safe_filename(asset_x)}_{_safe_filename(asset_y)}_dydx_candles_derived_history.json"
        written.append(
            build_pair_history_from_candles(
                left_path=left_path,
                right_path=right_path,
                output_path=output,
                pair_id=pair_id,
                asset_x=asset_x,
                asset_y=asset_y,
                hedge_ratio=hedge_ratio,
                beta=hedge_ratio,
                interval="5mins",
                zscore_window=zscore_window,
            )
        )
    return written


def _parse_loose_candle_text(text: str) -> Any:
    stripped = text.strip()
    if stripped.endswith("] }") or stripped.endswith("]\n}") or stripped.endswith("]\r\n}"):
        return json.loads('{"candles": [' + stripped.rsplit("]", 1)[0].rstrip().rstrip(",") + "]}")
    return json.loads("[" + stripped.rstrip().rstrip(",") + "]")


def _candles_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("candles", "data", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _bundle_leg_candles(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    json_payload = value.get("json")
    return _candles_from_payload(json_payload)


def _dydx_indexer_url(base_url: str, path: str, params: dict[str, object | None]) -> str:
    clean_params = {key: value for key, value in params.items() if value not in (None, "")}
    return f"{base_url.rstrip('/')}{path}?{urlencode(clean_params)}"


def _request_template_row(
    *,
    name: str,
    url: str,
    save_as: Path,
    import_command: str,
    notes: str,
) -> dict[str, str]:
    return {
        "request_name": name,
        "method": "GET",
        "url": url,
        "curl": f"curl -L '{url}' -H 'Content-Type: application/json' -o '{save_as}'",
        "save_as": str(save_as),
        "import_command": import_command,
        "notes": notes,
    }


def _dydx_market(asset: str) -> str:
    parts = [part for part in str(asset).upper().replace("/", "-").split("-") if part]
    if len(parts) >= 2 and parts[-1] == "USD":
        return f"{parts[0]}-USD"
    if len(parts) == 1:
        return f"{parts[0]}-USD"
    return str(asset).upper()


def _estimate_price_hedge_ratio(price_pairs: list[tuple[float, float]]) -> float:
    x_values = [x for x, y in price_pairs if x > 0 and y > 0]
    y_values = [y for x, y in price_pairs if x > 0 and y > 0]
    if len(x_values) < 2 or len(y_values) < 2:
        return 1.0
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    variance_y = sum((value - y_mean) ** 2 for value in y_values)
    if variance_y <= 0:
        ratios = [x / y for x, y in zip(x_values, y_values) if y > 0]
        return mean(ratios) if ratios else 1.0
    covariance_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    hedge_ratio = covariance_xy / variance_y
    if not math.isfinite(hedge_ratio) or hedge_ratio == 0:
        ratios = [x / y for x, y in zip(x_values, y_values) if y > 0]
        return mean(ratios) if ratios else 1.0
    return hedge_ratio


def _estimate_return_beta(price_pairs: list[tuple[float, float]]) -> float:
    if len(price_pairs) < 3:
        return 1.0
    returns_x = [_log_return(price_pairs[idx][0], price_pairs[idx - 1][0]) for idx in range(1, len(price_pairs))]
    returns_y = [_log_return(price_pairs[idx][1], price_pairs[idx - 1][1]) for idx in range(1, len(price_pairs))]
    x_mean = mean(returns_x)
    y_mean = mean(returns_y)
    variance_y = sum((value - y_mean) ** 2 for value in returns_y)
    if variance_y <= 0:
        return 1.0
    beta = sum((x - x_mean) * (y - y_mean) for x, y in zip(returns_x, returns_y)) / variance_y
    return beta if math.isfinite(beta) and beta != 0 else 1.0


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _timestamp(row: dict[str, Any]) -> str:
    return str(row.get("startedAt") or row.get("started_at") or row.get("timestamp") or "")


def _candle_price(row: dict[str, Any]) -> float:
    return _safe_float(row.get("close") or row.get("orderbookMidPriceClose") or row.get("midClose"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _attach_zscores(rows: list[dict[str, Any]], *, zscore_window: int, min_window: int) -> None:
    spreads = [_safe_float(row["spread"]) for row in rows]
    full_mean = mean(spreads)
    full_std = pstdev(spreads) or 1.0
    for idx, row in enumerate(rows):
        window = spreads[max(0, idx - zscore_window + 1) : idx + 1]
        if len(window) >= min_window:
            window_mean = mean(window)
            window_std = pstdev(window) or full_std
        else:
            window_mean = full_mean
            window_std = full_std
        row["zscore"] = (_safe_float(row["spread"]) - window_mean) / window_std


def _attach_provisional_ecm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spreads = [_safe_float(row["spread"]) for row in rows]
    spread_mean = mean(spreads)
    spread_std = pstdev(spreads) or 1.0
    lagged_spread_z = [0.0] + [(spread - spread_mean) / spread_std for spread in spreads[:-1]]
    returns_x = [0.0] + [_log_return(rows[i]["price_x"], rows[i - 1]["price_x"]) for i in range(1, len(rows))]
    returns_y = [0.0] + [_log_return(rows[i]["price_y"], rows[i - 1]["price_y"]) for i in range(1, len(rows))]
    gamma_x = _slope(lagged_spread_z[1:], returns_x[1:])
    gamma_y = _slope(lagged_spread_z[1:], returns_y[1:])
    strength = min(1.0, abs(gamma_x - gamma_y) * 100.0)
    for idx, row in enumerate(rows):
        row["ecm_x"] = gamma_x * lagged_spread_z[idx]
        row["ecm_y"] = gamma_y * lagged_spread_z[idx]
        row["ecm_strength"] = strength
    return {
        "method": "provisional_ols_lagged_spread_z_to_next_log_returns",
        "native_crypto_wizards_ecm": False,
        "gamma_x": gamma_x,
        "gamma_y": gamma_y,
        "ecm_strength": strength,
    }


def _attach_provisional_research_features(rows: list[dict[str, Any]], *, overwrite: bool = False) -> None:
    if not rows:
        return

    frame = pd.DataFrame(rows)
    if "zscore" not in frame.columns or "spread" not in frame.columns:
        return

    spread = pd.to_numeric(frame["spread"], errors="coerce").fillna(0.0)
    zscore = pd.to_numeric(frame["zscore"], errors="coerce").fillna(0.0)
    spread_delta = spread.diff().fillna(0.0)
    lagged_spread = spread.shift(1).fillna(spread.iloc[0])
    lagged_zscore = zscore.shift(1).fillna(zscore.iloc[0])
    window = min(max(len(frame) // 4, 12), 48)
    min_periods = min(8, window)

    phi = _safe_series_slope(lagged_spread.iloc[1:].tolist(), spread.iloc[1:].tolist())
    phi = max(min(phi, 0.99), -0.99)
    if phi == 0:
        half_life = 999.0
    else:
        half_life = abs(math.log(2.0) / math.log(abs(phi)))
    half_life = float(min(max(half_life, 2.0), 240.0))
    hurst = float(min(max(0.5 - 0.25 * (1.0 - phi), 0.05), 0.95))

    spread_mean = float(spread.mean()) if not spread.empty else 0.0
    spread_std = float(spread.std(ddof=0)) if float(spread.std(ddof=0) or 0.0) > 0 else 1.0
    spread_z = (spread - spread_mean) / spread_std
    cpd = np.tanh(zscore / 3.0)
    u1_given_u2 = np.clip(0.5 + cpd / 2.0, 0.0, 1.0)
    u2_given_u1 = np.clip(0.5 - cpd / 2.0, 0.0, 1.0)

    downside = spread_delta.where(spread_delta < 0.0, 0.0)
    downside_std = float(downside.std(ddof=0)) if float(downside.std(ddof=0) or 0.0) > 0 else float(spread_delta.std(ddof=0) or 1.0)
    mean_delta = float(spread_delta.mean())
    total_std = float(spread_delta.std(ddof=0) or 1.0)
    sharpe = float((mean_delta / total_std) * math.sqrt(max(len(spread_delta), 1)))
    sortino = float((mean_delta / downside_std) * math.sqrt(max(len(spread_delta), 1))) if downside_std else sharpe
    var = float(abs(spread_delta.quantile(0.05)))
    cvar = float(abs(spread_delta[spread_delta <= spread_delta.quantile(0.05)].mean()) if not spread_delta.empty else 0.0)
    cumulative = spread_delta.cumsum()
    rolling_max = cumulative.cummax()
    drawdown = float((rolling_max - cumulative).max()) if not cumulative.empty else 0.0
    win_rate = float((spread_delta > 0.0).mean()) if not spread_delta.empty else 0.5
    tail_dependence = float(np.clip(abs(cpd).mean() + abs(spread_z).rolling(20, min_periods=1).mean().iloc[-1] / 10.0, 0.0, 1.0))
    corr = pd.Series(spread).corr(pd.Series(frame["price_x"]), method="pearson")
    corr = 0.0 if pd.isna(corr) else float(corr)
    cointegration_pvalue = float(np.clip(1.0 - abs(corr), 0.0, 1.0))
    stability_series = spread.pct_change().abs().rolling(20, min_periods=1).mean()
    stability_value = float(stability_series.iloc[-1]) if not stability_series.empty and not pd.isna(stability_series.iloc[-1]) else 0.0
    hedge_ratio_stability = float(np.clip(1.0 - stability_value, 0.0, 1.0))
    realized_volatility_percentile = float(np.clip((spread_delta.abs().rank(pct=True).iloc[-1] if len(spread_delta) else 0.5), 0.0, 1.0))
    crisis_probability = float(np.clip(realized_volatility_percentile * 0.8, 0.0, 1.0))
    volume_x = pd.to_numeric(frame["volume_x_usd"], errors="coerce").fillna(0.0) if "volume_x_usd" in frame.columns else pd.Series(0.0, index=frame.index)
    liquidity_source = volume_x.rolling(20, min_periods=1).mean().iloc[-1] if not volume_x.empty else 0.0
    liquidity_score = float(np.clip(liquidity_source / 10_000.0, 0.0, 1.0))
    bid_ask_spread_bps = float(np.clip(20.0 - liquidity_score * 15.0, 1.0, 30.0))
    slippage_bps = float(np.clip(15.0 - liquidity_score * 10.0, 1.0, 25.0))
    funding_x = pd.to_numeric(frame["funding_x_bps"], errors="coerce").fillna(0.0) if "funding_x_bps" in frame.columns else pd.Series(0.0, index=frame.index)
    funding_y = pd.to_numeric(frame["funding_y_bps"], errors="coerce").fillna(0.0) if "funding_y_bps" in frame.columns else pd.Series(0.0, index=frame.index)
    funding_bps_per_day = float((funding_x.abs() + funding_y.abs()).mean())

    local_corr = lagged_spread.rolling(window, min_periods=min_periods).corr(spread)
    local_std_spread = spread.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    local_std_lagged = lagged_spread.rolling(window, min_periods=min_periods).std(ddof=0).replace(0.0, np.nan)
    phi_series = (local_corr * (local_std_spread / local_std_lagged)).replace([np.inf, -np.inf], np.nan)
    phi_series = phi_series.clip(-0.99, 0.99).fillna(phi)
    abs_phi = phi_series.abs().clip(lower=1.0e-6)
    half_life_series = abs(np.log(2.0) / np.log(abs_phi))
    half_life_series = half_life_series.replace([np.inf, -np.inf], np.nan).fillna(half_life).clip(2.0, 240.0)
    hurst_series = (0.5 - 0.25 * (1.0 - phi_series)).clip(0.05, 0.95)

    local_abs_cpd = abs(cpd).rolling(window, min_periods=1).mean().clip(0.0, 1.0)
    local_cpd_strength = (local_abs_cpd * 2.0).clip(0.0, 1.0)
    zscore_relief = (1.0 - abs(zscore) / 4.0).clip(0.0, 1.0)
    zscore_extension = (abs(zscore) / 3.5).clip(0.0, 1.0)
    delta_scale = spread_delta.abs().rolling(window, min_periods=1).mean().replace(0.0, np.nan)
    reversal_quality = (1.0 - (spread_delta.abs() / delta_scale)).clip(0.0, 1.0).fillna(0.5)
    local_cumulative = spread_delta.cumsum()
    local_peak = local_cumulative.rolling(window, min_periods=1).max()
    local_drawdown = (local_peak - local_cumulative).clip(lower=0.0)
    drawdown_scale = spread_delta.abs().rolling(window, min_periods=1).sum().replace(0.0, np.nan)
    normalized_drawdown = (local_drawdown / drawdown_scale).clip(0.0, 1.0).fillna(0.0)
    local_return = spread_delta.rolling(window, min_periods=2).sum().fillna(0.0)
    local_vol = spread_delta.rolling(window, min_periods=2).std(ddof=0).fillna(0.0)
    return_scale = spread_delta.abs().rolling(window, min_periods=2).mean().replace(0.0, np.nan)
    trend_pressure = (local_return / return_scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-2.0, 2.0)
    bullish_trend = trend_pressure.clip(lower=0.0) / 2.0
    local_vol_percentile = local_vol.rank(pct=True).clip(0.0, 1.0)
    crisis_like = (0.6 * local_vol_percentile + 0.4 * normalized_drawdown).clip(0.0, 1.0)
    bull_like = (bullish_trend * (1.0 - normalized_drawdown) * (1.0 - local_vol_percentile)).clip(0.0, 1.0)
    hurst_ml_term = (1.0 - abs(hurst_series - 0.35) / 0.35).clip(0.0, 1.0)
    hurst_ou_term = (1.0 - abs(hurst_series - 0.30) / 0.30).clip(0.0, 1.0)
    half_life_term = (1.0 - half_life_series / 240.0).clip(0.0, 1.0)

    ml_confidence_series = (
        0.24
        + 0.22 * hurst_ml_term
        + 0.16 * local_cpd_strength
        + 0.18 * reversal_quality
        + 0.12 * zscore_relief
        + 0.10 * (1.0 - normalized_drawdown)
        + 0.08 * crisis_like * zscore_extension
        - 0.10 * bull_like * bullish_trend
    ).clip(0.0, 1.0)
    profile_match_series = (
        0.22
        + 0.24 * (1.0 - local_abs_cpd)
        + 0.16 * zscore_relief
        + 0.16 * reversal_quality
        + 0.14 * crisis_like * zscore_extension
        + 0.08 * crisis_like * local_cpd_strength
        - 0.18 * bull_like * bullish_trend
        - 0.10 * bull_like * normalized_drawdown
    ).clip(0.0, 1.0)
    ou_optimal_series = (
        0.22
        + 0.28 * hurst_ou_term
        + 0.23 * half_life_term
        + 0.12 * zscore_relief
        + 0.12 * reversal_quality
        + 0.08 * crisis_like * zscore_extension
        - 0.12 * bull_like * bullish_trend
    ).clip(0.0, 1.0)
    ml_confidence = float(ml_confidence_series.iloc[-1])
    profile_match = float(profile_match_series.iloc[-1])
    ou_optimal = float(ou_optimal_series.iloc[-1])
    composite_score = float(np.clip(100.0 * np.nanmean([
        np.clip(1.0 - cointegration_pvalue, 0.0, 1.0),
        np.clip(1.0 - abs(hurst - 0.35) / 0.35, 0.0, 1.0),
        np.clip(1.0 - abs(cpd).mean(), 0.0, 1.0),
        np.clip(1.0 - drawdown, 0.0, 1.0),
    ]), 0.0, 100.0))

    for idx, row in enumerate(rows):
        updates = {
            "conditional_probability_distortion": float(cpd.iloc[idx]),
            "conditional_probabilities": True,
            "u1_given_u2": float(u1_given_u2.iloc[idx]),
            "u2_given_u1": float(u2_given_u1.iloc[idx]),
            "half_life": float(half_life_series.iloc[idx]),
            "hurst": float(hurst_series.iloc[idx]),
            "tail_dependence": tail_dependence,
            "cointegration_pvalue": cointegration_pvalue,
            "hedge_ratio_stability": hedge_ratio_stability,
            "sharpe": sharpe,
            "sortino": sortino,
            "var": var,
            "cvar": cvar,
            "drawdown": drawdown,
            "win_rate": win_rate,
            "ml_confidence": float(ml_confidence_series.iloc[idx]),
            "profile_match": float(profile_match_series.iloc[idx]),
            "ou_optimal": float(ou_optimal_series.iloc[idx]),
            "realized_volatility_percentile": realized_volatility_percentile,
            "crisis_probability": crisis_probability,
            "regime_strategy_match": 0.5 if row.get("regime") is None else 0.7,
            "bid_ask_spread_bps": bid_ask_spread_bps,
            "slippage_bps": slippage_bps,
            "funding_bps_per_day": funding_bps_per_day,
            "liquidity_score": liquidity_score,
            "copula_calibration_score": float(np.clip(0.5 + abs(cpd.iloc[idx]) / 2.0, 0.0, 1.0)),
            "composite_score": composite_score,
        }
        if overwrite:
            row.update(updates)
        else:
            for key, value in updates.items():
                row.setdefault(key, value)


def _safe_series_slope(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _log_return(current: Any, previous: Any) -> float:
    current_float = _safe_float(current)
    previous_float = _safe_float(previous)
    return math.log(current_float / previous_float) if current_float > 0 and previous_float > 0 else 0.0


def _slope(xs: list[float], ys: list[float]) -> float:
    if not xs or not ys:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
