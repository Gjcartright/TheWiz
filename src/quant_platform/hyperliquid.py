from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult, ROOT
from quant_platform.dydx_candles import build_pair_history_from_candles


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
SUPPORTED_INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "3d", "1w", "1M"}
ZSCORE_WINDOW = 7
ZSCORE_MIN_WINDOW = 7
HYPERLIQUID_MARKET_CONTEXT_COLUMNS = [
    "asset",
    "venue",
    "tradable",
    "volume_24h",
    "open_interest",
    "open_interest_usd",
    "funding_rate",
    "liquidity_usd",
    "transaction_count_24h",
    "market_cap",
    "source_timestamp",
    "source_system",
    "source_status",
    "source_role",
    "execution_authority",
    "promotion_allowed",
    "venue_lane",
    "liquidity_bucket",
    "funding_pulse_status",
    "blocker",
    "evidence_path",
    "notes",
    "mark_price",
]
HYPERLIQUID_RESEARCH_BUNDLE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "interval",
    "candidate_source",
    "testnet_pair_compatible",
    "history_path",
    "history_rows",
    "required_history_rows",
    "history_ready",
    "candle_x_fetched_at",
    "candle_y_fetched_at",
    "history_latest_candle_at",
    "history_status",
    "blocker",
    "next_step",
    "evidence_path",
]
HYPERLIQUID_FUNDING_HISTORY_COLUMNS = [
    "coin",
    "market",
    "timestamp",
    "time_ms",
    "funding_rate",
    "funding_bps",
    "premium",
    "source_system",
    "evidence_path",
]
HYPERLIQUID_FUNDING_COVERAGE_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "interval",
    "history_path",
    "history_rows",
    "funding_x_hourly_rows",
    "funding_y_hourly_rows",
    "funding_x_daily_rows",
    "funding_y_daily_rows",
    "funding_x_aligned_rows",
    "funding_y_aligned_rows",
    "both_legs_aligned_rows",
    "funding_coverage_pct",
    "funding_ready",
    "funding_cost_policy",
    "blocker",
    "next_step",
    "evidence_path",
]
DEFAULT_RESEARCH_INTERVALS = ("1d", "5m")
HYPERLIQUID_FEE_SOURCE_URL = "https://hyperliquid.gitbook.io/hyperliquid-docs/trading/fees"
HYPERLIQUID_INFO_SOURCE_URL = "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint"
DEFAULT_HYPERLIQUID_COST_PROFILE = {
    "profile_id": "hyperliquid_perp_base_tier_taker_v1",
    "venue": "hyperliquid",
    "instrument": "perpetual",
    "fee_tier_assumption": "base_tier_0_conservative",
    "execution_style": "taker",
    "taker_fee_bps": 4.5,
    "maker_fee_bps": 1.5,
    "execution_risk_bps": 2.0,
    "fee_source_url": HYPERLIQUID_FEE_SOURCE_URL,
    "fee_source_checked_at": "2026-08-05",
    "account_specific_fee_status": "not_captured_use_conservative_base_tier",
    "slippage_method": "public_l2_book_depth",
}
HYPERLIQUID_SLIPPAGE_SAMPLE_COLUMNS = [
    "sample_id",
    "pair",
    "asset",
    "venue",
    "notional_usd",
    "source_timestamp",
    "captured_at",
    "best_bid",
    "best_ask",
    "mid_price",
    "top_of_book_spread_bps",
    "buy_complete",
    "buy_average_price",
    "buy_slippage_bps",
    "buy_available_notional_usd",
    "sell_complete",
    "sell_average_price",
    "sell_slippage_bps",
    "sell_available_notional_usd",
    "one_way_slippage_bps",
    "book_levels_bid",
    "book_levels_ask",
    "blocker",
    "evidence_path",
]
HYPERLIQUID_PAIR_COST_MODEL_COLUMNS = [
    "pair",
    "asset_x",
    "asset_y",
    "venue",
    "leg_notional_usd",
    "fee_profile_id",
    "fee_tier_assumption",
    "execution_style",
    "taker_fee_bps",
    "maker_fee_bps",
    "execution_risk_bps",
    "fee_source_url",
    "fee_source_checked_at",
    "account_specific_fee_status",
    "cost_model_status",
    "cost_model_ready",
    "slippage_model_status",
    "slippage_model_ready",
    "slippage_samples_x",
    "slippage_samples_y",
    "required_slippage_samples",
    "slippage_window_hours",
    "slippage_x_p95_bps",
    "slippage_y_p95_bps",
    "pair_one_way_slippage_bps",
    "estimated_pair_round_trip_cost_bps",
    "freshest_sample_at",
    "oldest_qualifying_sample_at",
    "blocker",
    "next_step",
    "evidence_path",
]
HYPERLIQUID_EVIDENCE_CADENCE_COLUMNS = [
    "pair",
    "venue",
    "leg_notional_usd",
    "window_start_at",
    "samples_x",
    "samples_y",
    "minimum_samples",
    "required_samples",
    "oldest_sample_x_at",
    "oldest_sample_y_at",
    "latest_sample_x_at",
    "latest_sample_y_at",
    "first_sample_at",
    "latest_sample_at",
    "next_sample_due_at",
    "projected_captures_to_calibration",
    "projected_calibration_at",
    "projected_capture_feasible",
    "calibration_deadline_at",
    "cadence_minutes",
    "window_hours",
    "model_required_samples",
    "model_window_hours",
    "model_window_matches",
    "model_samples_match",
    "model_rebuild_required",
    "status",
    "blocker",
    "next_step",
    "evidence_path",
]
DEFAULT_SLIPPAGE_NOTIONALS = (250.0, 1_000.0, 5_000.0)
DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES = 12
DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS = 2
DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES = 10


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
    fetched_at = datetime.now(timezone.utc)
    snapshot = output_base / "snapshots" / fetched_at.strftime("%Y-%m-%d_%H%M%S") / output.name
    record = {
        "source": "hyperliquid",
        "info_url": info_url,
        "request": body,
        "candles": candles,
        "fetched_at": fetched_at.isoformat(),
        "snapshot_path": str(snapshot),
    }
    serialized = json.dumps(record, indent=2, sort_keys=True)
    output.write_text(serialized, encoding="utf-8")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(serialized, encoding="utf-8")
    return output


def fetch_hyperliquid_funding_history(
    *,
    coin: str,
    days: int = 500,
    output_dir: str | Path | None = None,
    end_time: datetime | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
    max_pages: int = 100,
    page_interval_seconds: float = 1.5,
    max_retries: int = 6,
) -> Path:
    """Fetch complete public funding history using Hyperliquid time pagination."""
    if days <= 0 or max_pages <= 0 or max_retries < 0 or page_interval_seconds < 0:
        raise ValueError("days and max_pages must be positive; retry and pacing values cannot be negative")
    clean_coin = _coin(coin)
    end_dt = _utc_datetime(end_time or datetime.now(timezone.utc))
    start_ms = int((end_dt - timedelta(days=days)).timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    next_start = start_ms
    records: list[dict[str, object]] = []
    requests_made: list[dict[str, object]] = []
    output_base = Path(output_dir or ROOT / "data" / "raw" / "hyperliquid_funding")
    checkpoint_path = output_base / f"{clean_coin}_funding.json"
    if checkpoint_path.exists():
        try:
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            checkpoint = {}
        checkpoint_records = checkpoint.get("funding", []) if isinstance(checkpoint, dict) else []
        checkpoint_times = [
            int(row["time"])
            for row in checkpoint_records
            if isinstance(row, dict) and row.get("time") is not None
        ]
        if checkpoint_times and min(checkpoint_times) <= start_ms + 3_600_000:
            records.extend(row for row in checkpoint_records if isinstance(row, dict))
            next_start = max(checkpoint_times) + 1

    for _ in range(max_pages):
        request_body = {
            "type": "fundingHistory",
            "coin": clean_coin,
            "startTime": next_start,
            "endTime": end_ms,
        }
        response = None
        for attempt in range(max_retries + 1):
            response = requests.post(
                info_url,
                json=request_body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if response.status_code != 429:
                break
            if attempt >= max_retries:
                break
            retry_after = _safe_float(response.headers.get("Retry-After"))
            delay = retry_after if retry_after > 0 else min(30.0, 1.5 * (2**attempt))
            time.sleep(delay)
        if response is None:
            raise RuntimeError("Hyperliquid fundingHistory request did not produce a response")
        response.raise_for_status()
        page = response.json()
        if not isinstance(page, list):
            raise ValueError("Hyperliquid fundingHistory response must be a list")
        requests_made.append({**request_body, "rows": len(page)})
        page_records = [row for row in page if isinstance(row, dict)]
        records.extend(page_records)
        output_base.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "source": "hyperliquid_public_funding_history",
                    "info_url": info_url,
                    "coin": clean_coin,
                    "requests": requests_made,
                    "funding": records,
                    "fetch_complete": False,
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        page_times = [int(row["time"]) for row in page_records if row.get("time") is not None]
        if not page_times or len(page_records) < 500:
            break
        last_time = max(page_times)
        if last_time >= end_ms:
            break
        advanced_start = last_time + 1
        if advanced_start <= next_start:
            raise RuntimeError("Hyperliquid fundingHistory pagination did not advance")
        next_start = advanced_start
        if page_interval_seconds:
            time.sleep(page_interval_seconds)
    else:
        raise RuntimeError(f"Hyperliquid fundingHistory pagination exceeded {max_pages} pages for {clean_coin}")

    deduplicated = {
        (str(row.get("coin", clean_coin)), int(row.get("time", 0))): row
        for row in records
        if row.get("time") is not None
    }
    ordered = sorted(deduplicated.values(), key=lambda row: int(row.get("time", 0)))
    captured_at = datetime.now(timezone.utc)
    return _write_hyperliquid_funding_record(
        coin=clean_coin,
        records=ordered,
        requests_made=requests_made,
        captured_at=captured_at,
        output_dir=output_base,
        info_url=info_url,
    )


def normalize_hyperliquid_funding_history(
    payload: Any,
    *,
    coin: str | None = None,
    evidence_path: str = "",
) -> list[dict[str, object]]:
    """Normalize hourly Hyperliquid funding rates into canonical bps rows."""
    if isinstance(payload, dict):
        records = payload.get("funding", payload.get("records", []))
        default_coin = _coin(str(payload.get("coin", coin or "")))
    else:
        records = payload
        default_coin = _coin(coin or "")
    if not isinstance(records, list):
        return []

    rows: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        time_ms = record.get("time")
        funding_rate = record.get("fundingRate", record.get("funding_rate"))
        if time_ms is None or funding_rate is None:
            continue
        try:
            timestamp = datetime.fromtimestamp(float(time_ms) / 1000.0, timezone.utc).isoformat()
            rate = float(funding_rate)
        except (TypeError, ValueError, OSError):
            continue
        row_coin = _coin(str(record.get("coin", default_coin)))
        if not row_coin:
            continue
        rows.append(
            {
                "coin": row_coin,
                "market": f"{row_coin}-USD",
                "timestamp": timestamp,
                "time_ms": int(float(time_ms)),
                "funding_rate": rate,
                "funding_bps": rate * 10_000.0,
                "premium": _safe_float(record.get("premium")),
                "source_system": "hyperliquid_public_funding_history",
                "evidence_path": evidence_path,
            }
        )
    return sorted(rows, key=lambda row: (str(row["coin"]), int(row["time_ms"])))


def refresh_hyperliquid_funding_history(
    root: Path = ROOT,
    max_pairs: int = 5,
    *,
    candidate_path: str | Path | None = None,
    days: int = 500,
    end_time: datetime | None = None,
    funding_payloads: dict[str, Any] | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
) -> CommandResult:
    """Fetch, normalize, and point-in-time align funding for active pairs."""
    candidates = _hyperliquid_research_candidates(
        root,
        max_pairs=max_pairs,
        candidate_path=Path(candidate_path) if candidate_path else None,
    )
    coins = sorted({_coin(asset) for row in candidates for asset in (row["asset_x"], row["asset_y"])})
    raw_dir = root / "data" / "raw" / "hyperliquid_funding"
    captured_at = _utc_datetime(end_time or datetime.now(timezone.utc))
    normalized_rows: list[dict[str, object]] = []
    raw_paths: list[Path] = []
    for coin_name in coins:
        if funding_payloads is not None:
            supplied = funding_payloads.get(coin_name, [])
            records = supplied.get("funding", supplied) if isinstance(supplied, dict) else supplied
            path = _write_hyperliquid_funding_record(
                coin=coin_name,
                records=[row for row in records if isinstance(row, dict)] if isinstance(records, list) else [],
                requests_made=[{"type": "fundingHistory", "coin": coin_name, "source": "supplied_payload"}],
                captured_at=captured_at,
                output_dir=raw_dir,
                info_url=info_url,
            )
        else:
            path = fetch_hyperliquid_funding_history(
                coin=coin_name,
                days=days,
                output_dir=raw_dir,
                end_time=end_time,
                info_url=info_url,
                timeout=timeout,
            )
        raw_paths.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized_rows.extend(
            normalize_hyperliquid_funding_history(payload, coin=coin_name, evidence_path=_rel(path, root))
        )

    funding = pd.DataFrame(normalized_rows, columns=HYPERLIQUID_FUNDING_HISTORY_COLUMNS)
    if not funding.empty:
        funding = funding.drop_duplicates(["coin", "time_ms"], keep="last").sort_values(["coin", "time_ms"]).reset_index(drop=True)
    funding_path = root / "data" / "processed" / "hyperliquid_funding.csv"
    _write_csv(funding, funding_path)
    coverage = _merge_hyperliquid_funding_into_pair_histories(
        root=root,
        candidates=candidates,
        funding=funding,
        funding_path=funding_path,
    )
    coverage_path = root / "reports" / "active" / "hyperliquid_funding_coverage.csv"
    coverage_md = root / "reports" / "active" / "hyperliquid_funding_coverage.md"
    _write_csv(coverage, coverage_path)
    _write_text(coverage_md, _hyperliquid_funding_coverage_markdown(coverage))
    return CommandResult(
        paths={
            "hyperliquid_funding": funding_path,
            "hyperliquid_funding_coverage": coverage_path,
            "hyperliquid_funding_coverage_md": coverage_md,
        },
        summary={
            "pairs": len(coverage),
            "coins": len(coins),
            "hourly_rows": len(funding),
            "funding_ready": int(coverage["funding_ready"].astype(bool).sum()) if not coverage.empty else 0,
            "raw_evidence_files": len(raw_paths),
        },
    )


def refresh_hyperliquid_market_context(
    root: Path = ROOT,
    *,
    payload: Any | None = None,
    captured_at: datetime | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
) -> CommandResult:
    """Capture a read-only Hyperliquid market snapshot for venue research.

    This calls only Hyperliquid's public ``info`` endpoint. It does not read a
    wallet, use a key, create an agent, or submit an order.
    """
    captured = _utc_datetime(captured_at or datetime.now(timezone.utc))
    request_body = {"type": "metaAndAssetCtxs"}
    if payload is None:
        response = requests.post(
            info_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

    raw_path = root / "data" / "raw" / "hyperliquid_market_snapshots" / f"{captured.strftime('%Y-%m-%d_%H%M%S')}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        json.dumps(
            {
                "source": "hyperliquid_public_info",
                "info_url": info_url,
                "request": request_body,
                "captured_at": captured.isoformat(),
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    evidence_path = _rel(raw_path, root)
    rows = normalize_hyperliquid_market_context(payload, captured_at=captured, evidence_path=evidence_path)
    frame = pd.DataFrame(rows, columns=HYPERLIQUID_MARKET_CONTEXT_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["asset"]).reset_index(drop=True)
    output = root / "data" / "processed" / "hyperliquid_market_context.csv"
    snapshot = root / "data" / "processed" / "hyperliquid_market_context_snapshots" / f"{captured.strftime('%Y-%m-%d_%H%M%S')}.csv"
    summary = root / "reports" / "active" / "hyperliquid_market_context_summary.csv"
    summary_md = root / "reports" / "active" / "hyperliquid_market_context_summary.md"
    _write_csv(frame, output)
    _write_csv(frame, snapshot)
    _write_csv(_hyperliquid_market_context_summary(frame), summary)
    _write_text(summary_md, _hyperliquid_market_context_markdown(frame, captured, evidence_path))
    return CommandResult(
        paths={
            "hyperliquid_market_context": output,
            "snapshot": snapshot,
            "summary": summary,
            "summary_md": summary_md,
            "raw_snapshot": raw_path,
        },
        summary={
            "markets": int(len(frame)),
            "tradable_markets": int(frame["tradable"].astype(bool).sum()) if not frame.empty else 0,
            "captured_at": captured.isoformat(),
        },
    )


def normalize_hyperliquid_market_context(
    payload: Any,
    *,
    captured_at: datetime,
    evidence_path: str,
) -> list[dict[str, object]]:
    """Normalize Hyperliquid ``metaAndAssetCtxs`` into venue-context rows."""
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Hyperliquid metaAndAssetCtxs payload must contain metadata and asset contexts")
    metadata, contexts = payload[0], payload[1]
    if not isinstance(metadata, dict) or not isinstance(contexts, list):
        raise ValueError("Hyperliquid metaAndAssetCtxs payload has invalid metadata or contexts")
    universe = metadata.get("universe", [])
    if not isinstance(universe, list):
        raise ValueError("Hyperliquid metadata is missing the universe list")

    timestamp = _utc_datetime(captured_at).isoformat()
    rows: list[dict[str, object]] = []
    for market, context in zip(universe, contexts):
        if not isinstance(market, dict) or not isinstance(context, dict):
            continue
        asset = _coin(str(market.get("name", "")))
        if not asset:
            continue
        mark_price = _safe_float(context.get("markPx") or context.get("midPx") or context.get("oraclePx"))
        open_interest = _safe_float(context.get("openInterest"))
        volume_24h = _safe_float(context.get("dayNtlVlm"))
        tradable = not bool(market.get("isDelisted", False))
        rows.append(
            {
                "asset": asset,
                "venue": "hyperliquid",
                "tradable": tradable,
                "volume_24h": volume_24h,
                "open_interest": open_interest,
                "open_interest_usd": open_interest * mark_price,
                "funding_rate": _safe_float(context.get("funding")),
                "liquidity_usd": volume_24h,
                "transaction_count_24h": "",
                "market_cap": "",
                "source_timestamp": timestamp,
                "source_system": "hyperliquid_public_api",
                "source_status": "captured",
                "source_role": "market_data_funding_context",
                "execution_authority": tradable,
                "promotion_allowed": False,
                "venue_lane": "hyperliquid_research_candidate",
                "liquidity_bucket": _liquidity_bucket(volume_24h),
                "funding_pulse_status": "not_required_for_hyperliquid_snapshot",
                "blocker": "requires_pair_history_cost_slippage_and_preflight",
                "evidence_path": evidence_path,
                "notes": "Public Hyperliquid mainnet market snapshot. It supports research context only and cannot authorize a trade.",
                "mark_price": mark_price,
            }
        )
    return rows


def build_hyperliquid_research_bundle(
    root: Path = ROOT,
    max_pairs: int = 5,
    *,
    intervals: tuple[str, ...] = DEFAULT_RESEARCH_INTERVALS,
    daily_days: int = 500,
    intraday_days: int = 30,
    refresh: bool = True,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
    candidate_path: str | Path | None = None,
    output_stem: str = "hyperliquid_research_bundle",
) -> CommandResult:
    """Create fresh Hyperliquid two-leg histories for mainnet-tradable research pairs.

    The bundle is local research evidence. It does not score a strategy, set a
    cost assumption, read credentials, or submit a Testnet order.
    """
    clean_intervals = tuple(dict.fromkeys(str(interval).strip() for interval in intervals if str(interval).strip()))
    unsupported = sorted(set(clean_intervals) - SUPPORTED_INTERVALS)
    if unsupported:
        raise ValueError(f"unsupported Hyperliquid intervals: {unsupported}")
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be a plain filename stem")

    candidates = _hyperliquid_research_candidates(
        root,
        max_pairs=max_pairs,
        candidate_path=Path(candidate_path) if candidate_path else None,
    )
    candle_dir = root / "data" / "raw" / "hyperliquid_candles"
    history_dir = root / "data" / "raw" / "pair_details"
    fetch_results: dict[tuple[str, str], Path | Exception] = {}
    if refresh:
        coins = sorted({_coin(asset) for candidate in candidates for asset in (candidate["asset_x"], candidate["asset_y"])})
        for interval in clean_intervals:
            days = daily_days if interval in {"1d", "3d", "1w", "1M"} else intraday_days
            for coin in coins:
                try:
                    fetch_results[(coin, interval)] = fetch_hyperliquid_candles(
                        coin=coin,
                        interval=interval,
                        days=days,
                        output_dir=candle_dir,
                        info_url=info_url,
                        timeout=timeout,
                    )
                except Exception as exc:  # Preserve other pair work when a market is unavailable.
                    fetch_results[(coin, interval)] = exc

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for interval in clean_intervals:
            left_coin = _coin(candidate["asset_x"])
            right_coin = _coin(candidate["asset_y"])
            left_path = candle_dir / f"{left_coin}_{interval}_candles.json"
            right_path = candle_dir / f"{right_coin}_{interval}_candles.json"
            fetch_errors = [
                result
                for result in (fetch_results.get((left_coin, interval)), fetch_results.get((right_coin, interval)))
                if isinstance(result, Exception)
            ]
            metadata_x = _candle_metadata(left_path)
            metadata_y = _candle_metadata(right_path)
            history_path = ""
            history_rows = 0
            latest_candle_at = ""
            required_rows = _required_history_rows(interval)
            blocker = ""
            if fetch_errors:
                blocker = "hyperliquid_candle_fetch_failed"
            elif not left_path.exists() or not right_path.exists():
                blocker = "missing_hyperliquid_two_leg_candles"
            else:
                try:
                    path = build_hyperliquid_pair_history(
                        asset_x=candidate["asset_x"],
                        asset_y=candidate["asset_y"],
                        interval=interval,
                        pair_id=f"{left_coin.lower()}_{right_coin.lower()}",
                        candle_dir=candle_dir,
                        output_dir=history_dir,
                    )
                    history_path = _rel(path, root)
                    history_rows, latest_candle_at = _pair_history_metadata(path)
                except Exception as exc:
                    blocker = f"hyperliquid_pair_history_build_failed:{type(exc).__name__}"

            candle_fresh = _timestamp_is_fresh(metadata_x.get("fetched_at", "")) and _timestamp_is_fresh(metadata_y.get("fetched_at", ""))
            history_ready = bool(not blocker and history_rows >= required_rows and candle_fresh)
            if not blocker and history_rows < required_rows:
                blocker = "insufficient_hyperliquid_history"
            elif not blocker and not candle_fresh:
                blocker = "stale_hyperliquid_candle_evidence"
            status = "history_ready_for_local_replay" if history_ready else "history_blocked"
            evidence_paths = [candidate["evidence_path"], _rel(left_path, root), _rel(right_path, root), history_path]
            rows.append(
                {
                    "pair": candidate["pair"],
                    "asset_x": _coin(candidate["asset_x"]),
                    "asset_y": _coin(candidate["asset_y"]),
                    "venue": "hyperliquid",
                    "interval": interval,
                    "candidate_source": candidate["candidate_source"],
                    "testnet_pair_compatible": bool(candidate.get("testnet_pair_compatible", False)),
                    "history_path": history_path,
                    "history_rows": history_rows,
                    "required_history_rows": required_rows,
                    "history_ready": history_ready,
                    "candle_x_fetched_at": metadata_x.get("fetched_at", ""),
                    "candle_y_fetched_at": metadata_y.get("fetched_at", ""),
                    "history_latest_candle_at": latest_candle_at,
                    "history_status": status,
                    "blocker": blocker,
                    "next_step": "collect_venue_cost_slippage_funding_and_preflight" if history_ready else "repair_hyperliquid_history_evidence",
                    "evidence_path": ";".join(value for value in evidence_paths if value),
                }
            )

    funding_path = root / "data" / "processed" / "hyperliquid_funding.csv"
    funding = _read_csv(funding_path)
    if not funding.empty:
        _merge_hyperliquid_funding_into_pair_histories(
            root=root,
            candidates=candidates,
            funding=funding,
            funding_path=funding_path,
        )

    frame = pd.DataFrame(rows, columns=HYPERLIQUID_RESEARCH_BUNDLE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["history_ready", "pair", "interval"], ascending=[False, True, True]).reset_index(drop=True)
    output = root / "reports" / "active" / f"{output_stem}.csv"
    markdown = root / "reports" / "active" / f"{output_stem}.md"
    _write_csv(frame, output)
    _write_text(markdown, _hyperliquid_research_bundle_markdown(frame))
    return CommandResult(
        paths={"hyperliquid_research_bundle": output, "hyperliquid_research_bundle_md": markdown},
        summary={
            "pairs": int(frame["pair"].nunique()) if not frame.empty else 0,
            "histories": int(len(frame)),
            "history_ready": int(frame["history_ready"].astype(bool).sum()) if not frame.empty else 0,
            "blocked": int((~frame["history_ready"].astype(bool)).sum()) if not frame.empty else 0,
        },
    )


def fetch_hyperliquid_l2_book(
    *,
    coin: str,
    output_dir: str | Path | None = None,
    payload: Any | None = None,
    captured_at: datetime | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
) -> Path:
    """Capture one read-only L2 book snapshot for a Hyperliquid perpetual."""
    clean_coin = _coin(coin)
    captured = _utc_datetime(captured_at or datetime.now(timezone.utc))
    request_body = {"type": "l2Book", "coin": clean_coin}
    if payload is None:
        response = requests.post(
            info_url,
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    _l2_book_levels(payload)
    output_base = Path(output_dir or ROOT / "data" / "raw" / "hyperliquid_l2_books")
    output = output_base / captured.strftime("%Y-%m-%d_%H%M%S") / f"{clean_coin}_l2_book.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "source": "hyperliquid_public_info",
                "info_url": info_url,
                "request": request_body,
                "captured_at": captured.isoformat(),
                "payload": payload,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output


def refresh_hyperliquid_execution_cost_snapshot(
    root: Path = ROOT,
    max_pairs: int = 5,
    *,
    notionals: tuple[float, ...] = DEFAULT_SLIPPAGE_NOTIONALS,
    captured_at: datetime | None = None,
    book_payloads: dict[str, Any] | None = None,
    info_url: str = HYPERLIQUID_INFO_URL,
    timeout: int = 30,
    candidate_path: str | Path | None = None,
    min_samples: int = DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    window_hours: float = DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
) -> CommandResult:
    """Append public L2 depth samples and rebuild pair cost/slippage evidence.

    The command calls only the public ``l2Book`` endpoint. Its output is a
    point-in-time research estimate and never an order or execution request.
    """
    clean_notionals = tuple(sorted({float(value) for value in notionals if float(value) > 0}))
    if not clean_notionals:
        raise ValueError("at least one positive notional is required")
    if min_samples <= 0 or window_hours <= 0:
        raise ValueError("min_samples and window_hours must be positive")
    candidates = _hyperliquid_research_candidates(
        root,
        max_pairs=max_pairs,
        candidate_path=Path(candidate_path) if candidate_path else None,
    )
    if not candidates:
        return _write_empty_hyperliquid_cost_reports(root, reason="no_hyperliquid_research_candidates")

    capture = _utc_datetime(captured_at or datetime.now(timezone.utc))
    book_dir = root / "data" / "raw" / "hyperliquid_l2_books"
    books: dict[str, Path | Exception] = {}
    for asset in sorted({_coin(value) for candidate in candidates for value in (candidate["asset_x"], candidate["asset_y"])}):
        try:
            books[asset] = fetch_hyperliquid_l2_book(
                coin=asset,
                output_dir=book_dir,
                payload=(book_payloads or {}).get(asset),
                captured_at=capture,
                info_url=info_url,
                timeout=timeout,
            )
        except Exception as exc:  # A thin or remapped market should not block other pairs.
            books[asset] = exc

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        for asset in (_coin(candidate["asset_x"]), _coin(candidate["asset_y"])):
            result = books.get(asset)
            if isinstance(result, Path):
                rows.extend(
                    _hyperliquid_slippage_rows_from_book(
                        pair=candidate["pair"],
                        asset=asset,
                        raw_path=result,
                        notionals=clean_notionals,
                        root=root,
                    )
                )
                continue
            blocker = f"hyperliquid_l2_book_fetch_failed:{type(result).__name__}" if isinstance(result, Exception) else "missing_hyperliquid_l2_book"
            for notional in clean_notionals:
                rows.append(_blocked_hyperliquid_slippage_row(candidate["pair"], asset, notional, capture, blocker))

    samples_path = root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv"
    existing = _read_csv(samples_path)
    samples = pd.concat([existing, pd.DataFrame(rows, columns=HYPERLIQUID_SLIPPAGE_SAMPLE_COLUMNS)], ignore_index=True)
    if not samples.empty:
        samples = samples.drop_duplicates(["sample_id"], keep="last").sort_values(["source_timestamp", "pair", "asset", "notional_usd"]).reset_index(drop=True)
    _write_csv(samples.reindex(columns=HYPERLIQUID_SLIPPAGE_SAMPLE_COLUMNS, fill_value=""), samples_path)
    sample_report = root / "reports" / "active" / "hyperliquid_l2_slippage_samples.csv"
    _write_csv(samples.reindex(columns=HYPERLIQUID_SLIPPAGE_SAMPLE_COLUMNS, fill_value=""), sample_report)
    collected_times = pd.to_datetime(
        pd.Series([row.get("source_timestamp", "") for row in rows], dtype=object),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    newest_collected = collected_times.max() if collected_times.notna().any() else pd.NaT
    model_as_of = max(capture, newest_collected.to_pydatetime()) if pd.notna(newest_collected) else capture
    model = build_hyperliquid_pair_cost_model(
        root=root,
        max_pairs=max_pairs,
        leg_notional_usd=1_000.0 if 1_000.0 in clean_notionals else clean_notionals[0],
        min_samples=min_samples,
        window_hours=window_hours,
        as_of=model_as_of,
        candidate_path=candidate_path,
    )
    sample_markdown = root / "reports" / "active" / "hyperliquid_l2_slippage_samples.md"
    _write_text(
        sample_markdown,
        _hyperliquid_l2_slippage_markdown(
            samples,
            clean_notionals,
            min_samples=min_samples,
            window_hours=window_hours,
        ),
    )
    return CommandResult(
        paths={
            "hyperliquid_l2_slippage_samples": sample_report,
            "hyperliquid_l2_slippage_samples_md": sample_markdown,
            **model.paths,
        },
        summary={
            "pairs": len(candidates),
            "samples_written": len(rows),
            "unique_samples": len(samples),
            "model_as_of": model_as_of.isoformat(),
            **model.summary,
        },
    )


def build_hyperliquid_pair_cost_model(
    root: Path = ROOT,
    max_pairs: int = 5,
    *,
    leg_notional_usd: float = 1_000.0,
    min_samples: int = DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    window_hours: float = DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
    as_of: datetime | None = None,
    candidate_path: str | Path | None = None,
) -> CommandResult:
    """Build a conservative pair cost model from fee evidence and L2 samples."""
    if leg_notional_usd <= 0:
        raise ValueError("leg_notional_usd must be positive")
    if min_samples <= 0 or window_hours <= 0:
        raise ValueError("min_samples and window_hours must be positive")
    candidates = _hyperliquid_research_candidates(
        root,
        max_pairs=max_pairs,
        candidate_path=Path(candidate_path) if candidate_path else None,
    )
    samples_path = root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv"
    samples = _read_csv(samples_path)
    profile, profile_path = _hyperliquid_cost_profile(root)
    now = _utc_datetime(as_of or datetime.now(timezone.utc))
    rows = [
        _hyperliquid_pair_cost_model_row(
            candidate,
            samples,
            profile=profile,
            profile_path=profile_path,
            samples_path=samples_path,
            leg_notional_usd=leg_notional_usd,
            min_samples=min_samples,
            window_hours=window_hours,
            as_of=now,
        )
        for candidate in candidates
    ]
    frame = pd.DataFrame(rows, columns=HYPERLIQUID_PAIR_COST_MODEL_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["slippage_model_ready", "cost_model_ready", "pair"], ascending=[False, False, True]).reset_index(drop=True)
    output = root / "reports" / "active" / "hyperliquid_pair_cost_model.csv"
    markdown = root / "reports" / "active" / "hyperliquid_pair_cost_model.md"
    _write_csv(frame, output)
    _write_text(markdown, _hyperliquid_pair_cost_model_markdown(frame))
    return CommandResult(
        paths={"hyperliquid_pair_cost_model": output, "hyperliquid_pair_cost_model_md": markdown},
        summary={
            "pairs": len(frame),
            "cost_models_ready": int(frame["cost_model_ready"].astype(bool).sum()) if not frame.empty else 0,
            "slippage_models_ready": int(frame["slippage_model_ready"].astype(bool).sum()) if not frame.empty else 0,
        },
    )


def _project_l2_calibration(
    *,
    times_x: pd.Series,
    times_y: pd.Series,
    now: pd.Timestamp,
    first_capture_at: pd.Timestamp,
    target_samples: int,
    cadence_minutes: int,
    window_hours: float,
) -> tuple[int | None, pd.Timestamp]:
    """Project captures needed after accounting for rolling-window expiry."""
    window = pd.Timedelta(hours=window_hours)
    cadence = pd.Timedelta(minutes=cadence_minutes)
    observed_x = [pd.Timestamp(value) for value in times_x if pd.notna(value)]
    observed_y = [pd.Timestamp(value) for value in times_y if pd.notna(value)]
    if len(observed_x) >= target_samples and len(observed_y) >= target_samples:
        return 0, now

    maximum_captures = target_samples + int(window / cadence) + 2
    for capture_count in range(1, maximum_captures + 1):
        capture_at = first_capture_at + cadence * (capture_count - 1)
        cutoff = capture_at - window
        observed_x = [value for value in observed_x if cutoff <= value <= capture_at]
        observed_y = [value for value in observed_y if cutoff <= value <= capture_at]
        observed_x.append(capture_at)
        observed_y.append(capture_at)
        if len(observed_x) >= target_samples and len(observed_y) >= target_samples:
            return capture_count, capture_at
    return None, pd.NaT


def build_hyperliquid_evidence_cadence(
    root: Path = ROOT,
    *,
    target_samples: int = DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    cadence_minutes: int = DEFAULT_SLIPPAGE_CALIBRATION_CADENCE_MINUTES,
    window_hours: float = DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
    as_of: datetime | None = None,
) -> CommandResult:
    """Report the next public-depth evidence task for each active pair.

    This is a scheduler-facing plan only. It neither sleeps nor submits any
    exchange request, so orchestration can call it safely between explicit
    snapshot jobs.
    """
    if target_samples <= 0 or cadence_minutes <= 0 or window_hours <= 0:
        raise ValueError("target_samples, cadence_minutes, and window_hours must be positive")
    model_path = root / "reports" / "active" / "hyperliquid_pair_cost_model.csv"
    samples_path = root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv"
    model = _read_csv(model_path)
    samples = _read_csv(samples_path)
    now = _utc_datetime(as_of or datetime.now(timezone.utc))
    window_start = pd.Timestamp(now) - pd.Timedelta(hours=window_hours)
    rows: list[dict[str, object]] = []
    for _, row in model.iterrows():
        record = row.to_dict()
        pair = str(record.get("pair", "") or "")
        asset_x = _coin(str(record.get("asset_x", "")))
        asset_y = _coin(str(record.get("asset_y", "")))
        notional = _safe_float(record.get("leg_notional_usd"))
        leg_x = _qualifying_slippage_samples(
            samples,
            pair=pair,
            asset=asset_x,
            notional_usd=notional,
            start_at=window_start,
            end_at=pd.Timestamp(now),
        )
        leg_y = _qualifying_slippage_samples(
            samples,
            pair=pair,
            asset=asset_y,
            notional_usd=notional,
            start_at=window_start,
            end_at=pd.Timestamp(now),
        )
        times_x = leg_x["_source_timestamp"] if not leg_x.empty else pd.Series(dtype="datetime64[ns, UTC]")
        times_y = leg_y["_source_timestamp"] if not leg_y.empty else pd.Series(dtype="datetime64[ns, UTC]")
        timestamps = pd.concat([times_x, times_y], ignore_index=True)
        first = timestamps.min() if not timestamps.empty else pd.NaT
        latest = timestamps.max() if not timestamps.empty else pd.NaT
        oldest_x = times_x.min() if not times_x.empty else pd.NaT
        oldest_y = times_y.min() if not times_y.empty else pd.NaT
        latest_x = times_x.max() if not times_x.empty else pd.NaT
        latest_y = times_y.max() if not times_y.empty else pd.NaT
        samples_x = int(len(leg_x))
        samples_y = int(len(leg_y))
        minimum = min(samples_x, samples_y)
        required = max(0, target_samples - minimum)
        leg_latest = [value for value in (latest_x, latest_y) if pd.notna(value)]
        due_basis = min(leg_latest) if len(leg_latest) == 2 else pd.NaT
        due = pd.Timestamp(now) if pd.isna(due_basis) else due_basis + pd.Timedelta(minutes=cadence_minutes)
        projected_captures, projected_at = _project_l2_calibration(
            times_x=times_x,
            times_y=times_y,
            now=pd.Timestamp(now),
            first_capture_at=max(pd.Timestamp(now), due),
            target_samples=target_samples,
            cadence_minutes=cadence_minutes,
            window_hours=window_hours,
        )
        projection_feasible = projected_captures is not None
        deadline = pd.NaT if pd.isna(first) else first + pd.Timedelta(hours=window_hours)
        model_required = int(_safe_float(record.get("required_slippage_samples")))
        model_window = _safe_float(record.get("slippage_window_hours"))
        model_window_matches = abs(model_window - float(window_hours)) < 1e-9
        model_samples_match = (
            int(_safe_float(record.get("slippage_samples_x"))) == samples_x
            and int(_safe_float(record.get("slippage_samples_y"))) == samples_y
        )
        model_matches = bool(
            _truthy(record.get("slippage_model_ready"))
            and model_required == target_samples
            and model_window_matches
            and model_samples_match
        )
        calibrated = minimum >= target_samples and model_matches
        model_rebuild_required = minimum >= target_samples and not model_matches
        if not projection_feasible:
            status = "calibration_schedule_infeasible"
            blocker = "target_samples_exceed_rolling_window_cadence_capacity"
            next_step = "lower_target_samples_or_increase_window_or_reduce_cadence"
        elif calibrated:
            status = "calibrated"
            blocker = ""
            next_step = "run_costed_hyperliquid_local_replay"
        elif model_rebuild_required:
            status = "model_rebuild_due"
            blocker = "slippage_cost_model_window_or_counts_stale"
            next_step = "rebuild_pair_cost_model_with_matching_rolling_window"
        elif now >= due:
            status = "capture_due"
            blocker = ""
            next_step = "run refresh-hyperliquid-execution-cost-snapshot"
        else:
            status = "waiting_for_next_capture"
            blocker = ""
            next_step = "wait_until_next_l2_capture_due"
        rows.append(
            {
                "pair": pair,
                "venue": "hyperliquid",
                "leg_notional_usd": notional,
                "window_start_at": window_start.isoformat(),
                "samples_x": samples_x,
                "samples_y": samples_y,
                "minimum_samples": minimum,
                "required_samples": required,
                "oldest_sample_x_at": oldest_x.isoformat() if pd.notna(oldest_x) else "",
                "oldest_sample_y_at": oldest_y.isoformat() if pd.notna(oldest_y) else "",
                "latest_sample_x_at": latest_x.isoformat() if pd.notna(latest_x) else "",
                "latest_sample_y_at": latest_y.isoformat() if pd.notna(latest_y) else "",
                "first_sample_at": first.isoformat() if pd.notna(first) else "",
                "latest_sample_at": latest.isoformat() if pd.notna(latest) else "",
                "next_sample_due_at": due.isoformat(),
                "projected_captures_to_calibration": projected_captures if projection_feasible else "",
                "projected_calibration_at": projected_at.isoformat() if pd.notna(projected_at) else "",
                "projected_capture_feasible": projection_feasible,
                "calibration_deadline_at": deadline.isoformat() if pd.notna(deadline) else "",
                "cadence_minutes": cadence_minutes,
                "window_hours": window_hours,
                "model_required_samples": model_required,
                "model_window_hours": model_window,
                "model_window_matches": model_window_matches,
                "model_samples_match": model_samples_match,
                "model_rebuild_required": model_rebuild_required,
                "status": status,
                "blocker": blocker,
                "next_step": next_step,
                "evidence_path": f"{samples_path};{model_path}",
            }
        )
    frame = pd.DataFrame(rows, columns=HYPERLIQUID_EVIDENCE_CADENCE_COLUMNS)
    if not frame.empty:
        status_order = {
            "calibration_schedule_infeasible": 0,
            "model_rebuild_due": 1,
            "capture_due": 2,
            "waiting_for_next_capture": 3,
            "calibrated": 4,
        }
        frame = frame.assign(_status_rank=frame["status"].map(status_order).fillna(9)).sort_values(["_status_rank", "pair"]).drop(columns="_status_rank").reset_index(drop=True)
    output = root / "reports" / "active" / "hyperliquid_evidence_cadence.csv"
    markdown = root / "reports" / "active" / "hyperliquid_evidence_cadence.md"
    _write_csv(frame, output)
    _write_text(markdown, _hyperliquid_evidence_cadence_markdown(frame))
    return CommandResult(
        paths={"hyperliquid_evidence_cadence": output, "hyperliquid_evidence_cadence_md": markdown},
        summary={
            "pairs": len(frame),
            "capture_due": int(frame["status"].eq("capture_due").sum()) if not frame.empty else 0,
            "waiting": int(frame["status"].eq("waiting_for_next_capture").sum()) if not frame.empty else 0,
            "calibrated": int(frame["status"].eq("calibrated").sum()) if not frame.empty else 0,
            "model_rebuild_due": int(frame["status"].eq("model_rebuild_due").sum()) if not frame.empty else 0,
            "schedule_infeasible": int(frame["status"].eq("calibration_schedule_infeasible").sum())
            if not frame.empty
            else 0,
            "expired": 0,
        },
    )


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
    zscore_window: int = ZSCORE_WINDOW,
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
        min_zscore_window=ZSCORE_MIN_WINDOW if zscore_window == ZSCORE_WINDOW else min(20, max(2, zscore_window // 4)),
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


def _l2_book_levels(payload: Any) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not isinstance(payload, dict):
        raise ValueError("Hyperliquid l2Book payload must be an object")
    levels = payload.get("levels", [])
    if not isinstance(levels, list) or len(levels) < 2:
        raise ValueError("Hyperliquid l2Book payload must contain bid and ask levels")
    bids = [level for level in levels[0] if isinstance(level, dict)] if isinstance(levels[0], list) else []
    asks = [level for level in levels[1] if isinstance(level, dict)] if isinstance(levels[1], list) else []
    if not bids or not asks:
        raise ValueError("Hyperliquid l2Book payload has no usable bid or ask levels")
    return bids, asks


def _book_timestamp(payload: dict[str, object], captured_at: object) -> str:
    value = payload.get("time")
    try:
        if value is not None:
            return datetime.fromtimestamp(float(value) / 1000.0, timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        pass
    timestamp = pd.to_datetime(captured_at, utc=True, errors="coerce")
    return timestamp.isoformat() if pd.notna(timestamp) else ""


def _market_order_estimate(levels: list[dict[str, object]], *, notional_usd: float, mid_price: float, side: str) -> dict[str, object]:
    remaining = float(notional_usd)
    filled_notional = 0.0
    filled_size = 0.0
    available_notional = 0.0
    for level in levels:
        price = _safe_float(level.get("px"))
        size = _safe_float(level.get("sz"))
        if price <= 0 or size <= 0:
            continue
        level_notional = price * size
        available_notional += level_notional
        take_notional = min(remaining, level_notional)
        filled_notional += take_notional
        filled_size += take_notional / price
        remaining -= take_notional
        if remaining <= 1e-9:
            break
    complete = remaining <= 1e-6
    average_price = filled_notional / filled_size if filled_size > 0 else 0.0
    if not complete or average_price <= 0 or mid_price <= 0:
        slippage_bps: float | str = ""
    elif side == "buy":
        slippage_bps = (average_price / mid_price - 1.0) * 10_000.0
    else:
        slippage_bps = (mid_price / average_price - 1.0) * 10_000.0
    return {
        "complete": complete,
        "average_price": average_price if average_price > 0 else "",
        "slippage_bps": slippage_bps,
        "available_notional_usd": available_notional,
    }


def _hyperliquid_slippage_rows_from_book(
    *,
    pair: str,
    asset: str,
    raw_path: Path,
    notionals: tuple[float, ...],
    root: Path,
) -> list[dict[str, object]]:
    record = json.loads(raw_path.read_text(encoding="utf-8"))
    payload = record.get("payload", {}) if isinstance(record, dict) else {}
    if not isinstance(payload, dict):
        raise ValueError("Hyperliquid L2 raw record is missing its payload")
    bids, asks = _l2_book_levels(payload)
    best_bid = _safe_float(bids[0].get("px"))
    best_ask = _safe_float(asks[0].get("px"))
    mid_price = (best_bid + best_ask) / 2.0 if best_bid > 0 and best_ask > 0 else 0.0
    spread_bps = ((best_ask - best_bid) / mid_price * 10_000.0) if mid_price > 0 else 0.0
    captured_at = str(record.get("captured_at", "") or "")
    source_timestamp = _book_timestamp(payload, captured_at)
    evidence_path = _rel(raw_path, root)
    rows: list[dict[str, object]] = []
    for notional in notionals:
        buy = _market_order_estimate(asks, notional_usd=notional, mid_price=mid_price, side="buy")
        sell = _market_order_estimate(bids, notional_usd=notional, mid_price=mid_price, side="sell")
        buy_bps = pd.to_numeric(pd.Series([buy["slippage_bps"]]), errors="coerce").iloc[0]
        sell_bps = pd.to_numeric(pd.Series([sell["slippage_bps"]]), errors="coerce").iloc[0]
        one_way = max(float(buy_bps), float(sell_bps)) if pd.notna(buy_bps) and pd.notna(sell_bps) else ""
        complete = bool(buy["complete"]) and bool(sell["complete"])
        rows.append(
            {
                "sample_id": f"{pair}|{asset}|{notional:.8f}|{source_timestamp}",
                "pair": pair,
                "asset": asset,
                "venue": "hyperliquid",
                "notional_usd": notional,
                "source_timestamp": source_timestamp,
                "captured_at": captured_at,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "mid_price": mid_price,
                "top_of_book_spread_bps": spread_bps,
                "buy_complete": bool(buy["complete"]),
                "buy_average_price": buy["average_price"],
                "buy_slippage_bps": buy["slippage_bps"],
                "buy_available_notional_usd": buy["available_notional_usd"],
                "sell_complete": bool(sell["complete"]),
                "sell_average_price": sell["average_price"],
                "sell_slippage_bps": sell["slippage_bps"],
                "sell_available_notional_usd": sell["available_notional_usd"],
                "one_way_slippage_bps": one_way,
                "book_levels_bid": len(bids),
                "book_levels_ask": len(asks),
                "blocker": "" if complete else "insufficient_l2_depth_for_notional",
                "evidence_path": evidence_path,
            }
        )
    return rows


def _blocked_hyperliquid_slippage_row(pair: str, asset: str, notional: float, captured_at: datetime, blocker: str) -> dict[str, object]:
    timestamp = captured_at.isoformat()
    return {
        "sample_id": f"{pair}|{asset}|{notional:.8f}|{timestamp}",
        "pair": pair,
        "asset": asset,
        "venue": "hyperliquid",
        "notional_usd": notional,
        "source_timestamp": timestamp,
        "captured_at": timestamp,
        "best_bid": "",
        "best_ask": "",
        "mid_price": "",
        "top_of_book_spread_bps": "",
        "buy_complete": False,
        "buy_average_price": "",
        "buy_slippage_bps": "",
        "buy_available_notional_usd": "",
        "sell_complete": False,
        "sell_average_price": "",
        "sell_slippage_bps": "",
        "sell_available_notional_usd": "",
        "one_way_slippage_bps": "",
        "book_levels_bid": 0,
        "book_levels_ask": 0,
        "blocker": blocker,
        "evidence_path": "",
    }


def _hyperliquid_cost_profile(root: Path) -> tuple[dict[str, object], Path]:
    path = root / "config" / "hyperliquid_perp_cost_profile.json"
    profile = dict(DEFAULT_HYPERLIQUID_COST_PROFILE)
    if path.exists():
        try:
            configured = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            configured = {}
        if isinstance(configured, dict):
            profile.update(configured)
    return profile, path


def _hyperliquid_pair_cost_model_row(
    candidate: dict[str, str],
    samples: pd.DataFrame,
    *,
    profile: dict[str, object],
    profile_path: Path,
    samples_path: Path,
    leg_notional_usd: float,
    min_samples: int,
    window_hours: float,
    as_of: datetime,
) -> dict[str, object]:
    window_end = pd.Timestamp(as_of)
    window_start = window_end - pd.Timedelta(hours=window_hours)
    pair = candidate["pair"]
    asset_x = _coin(candidate["asset_x"])
    asset_y = _coin(candidate["asset_y"])

    def leg(asset: str) -> tuple[int, float | str, str, str]:
        subset = _qualifying_slippage_samples(
            samples,
            pair=pair,
            asset=asset,
            notional_usd=leg_notional_usd,
            start_at=window_start,
            end_at=window_end,
        )
        if subset.empty:
            return 0, "", "", ""
        slippage = subset["_slippage_bps"]
        count = int(len(slippage))
        p95 = float(slippage.quantile(0.95)) if count else ""
        fresh = subset["_source_timestamp"]
        return count, p95, fresh.max().isoformat() if fresh.notna().any() else "", fresh.min().isoformat() if fresh.notna().any() else ""

    count_x, p95_x, freshest_x, oldest_x = leg(asset_x)
    count_y, p95_y, freshest_y, oldest_y = leg(asset_y)
    slippage_ready = bool(count_x >= min_samples and count_y >= min_samples)
    p95_values = [value for value in (p95_x, p95_y) if isinstance(value, float)]
    pair_slippage = float(sum(p95_values) / len(p95_values)) if len(p95_values) == 2 else ""
    fee_bps = _safe_float(profile.get("taker_fee_bps"))
    execution_risk_bps = _safe_float(profile.get("execution_risk_bps"))
    cost_ready = bool(fee_bps > 0 and str(profile.get("fee_source_url", "")).strip())
    estimated_cost = 2.0 * (fee_bps + float(pair_slippage) + execution_risk_bps) if isinstance(pair_slippage, float) else ""
    blockers = []
    if not cost_ready:
        blockers.append("invalid_hyperliquid_fee_profile")
    if count_x < min_samples:
        blockers.append("insufficient_l2_slippage_samples_asset_x")
    if count_y < min_samples:
        blockers.append("insufficient_l2_slippage_samples_asset_y")
    freshest = max((value for value in (freshest_x, freshest_y) if value), default="")
    oldest = min((value for value in (oldest_x, oldest_y) if value), default="")
    return {
        "pair": pair,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "venue": "hyperliquid",
        "leg_notional_usd": leg_notional_usd,
        "fee_profile_id": str(profile.get("profile_id", "")),
        "fee_tier_assumption": str(profile.get("fee_tier_assumption", "")),
        "execution_style": str(profile.get("execution_style", "")),
        "taker_fee_bps": fee_bps,
        "maker_fee_bps": _safe_float(profile.get("maker_fee_bps")),
        "execution_risk_bps": execution_risk_bps,
        "fee_source_url": str(profile.get("fee_source_url", "")),
        "fee_source_checked_at": str(profile.get("fee_source_checked_at", "")),
        "account_specific_fee_status": str(profile.get("account_specific_fee_status", "")),
        "cost_model_status": "official_base_tier_conservative_fee_profile" if cost_ready else "invalid_hyperliquid_fee_profile",
        "cost_model_ready": cost_ready,
        "slippage_model_status": "l2_depth_calibrated_p95" if slippage_ready else "insufficient_l2_depth_samples",
        "slippage_model_ready": slippage_ready,
        "slippage_samples_x": count_x,
        "slippage_samples_y": count_y,
        "required_slippage_samples": min_samples,
        "slippage_window_hours": window_hours,
        "slippage_x_p95_bps": p95_x,
        "slippage_y_p95_bps": p95_y,
        "pair_one_way_slippage_bps": pair_slippage,
        "estimated_pair_round_trip_cost_bps": estimated_cost,
        "freshest_sample_at": freshest,
        "oldest_qualifying_sample_at": oldest,
        "blocker": ";".join(blockers),
        "next_step": "collect_more_l2_samples_across_the_two_hour_window" if not slippage_ready else "run_costed_hyperliquid_local_replay",
        "evidence_path": ";".join(
            value
            for value in [
                _rel(profile_path, root=profile_path.parents[1]) if profile_path.exists() else str(profile_path),
                str(samples_path),
                candidate.get("evidence_path", ""),
            ]
            if value
        ),
    }


def _write_empty_hyperliquid_cost_reports(root: Path, *, reason: str) -> CommandResult:
    frame = pd.DataFrame(columns=HYPERLIQUID_PAIR_COST_MODEL_COLUMNS)
    output = root / "reports" / "active" / "hyperliquid_pair_cost_model.csv"
    markdown = root / "reports" / "active" / "hyperliquid_pair_cost_model.md"
    _write_csv(frame, output)
    _write_text(markdown, f"# Hyperliquid Pair Cost Model\n\nBlocked: `{reason}`.\n")
    return CommandResult(
        paths={"hyperliquid_pair_cost_model": output, "hyperliquid_pair_cost_model_md": markdown},
        summary={"pairs": 0, "cost_models_ready": 0, "slippage_models_ready": 0, "blocker": reason},
    )


def _hyperliquid_l2_slippage_markdown(
    frame: pd.DataFrame,
    notionals: tuple[float, ...],
    *,
    min_samples: int = DEFAULT_SLIPPAGE_CALIBRATION_MIN_SAMPLES,
    window_hours: float = DEFAULT_SLIPPAGE_CALIBRATION_WINDOW_HOURS,
) -> str:
    if frame.empty:
        return "# Hyperliquid L2 Slippage Samples\n\nNo samples captured.\n"
    complete = frame.get("buy_complete", pd.Series(False, index=frame.index)).map(_truthy) & frame.get("sell_complete", pd.Series(False, index=frame.index)).map(_truthy)
    summary = pd.DataFrame(
        [
            {"metric": "samples", "value": len(frame)},
            {"metric": "complete_samples", "value": int(complete.sum())},
            {"metric": "notionals_usd", "value": ",".join(f"{value:g}" for value in notionals)},
            {
                "metric": "calibration_rule",
                "value": f"{min_samples} complete samples per leg within a rolling {window_hours:g}h window",
            },
        ]
    )
    view = frame[
        ["pair", "asset", "notional_usd", "one_way_slippage_bps", "buy_complete", "sell_complete", "top_of_book_spread_bps", "source_timestamp", "blocker"]
    ].tail(50)
    return "\n".join(
        [
            "# Hyperliquid L2 Slippage Samples",
            "",
            "Point-in-time estimates from public L2 depth. They are calibration inputs, not promises of future fills.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Latest Samples",
            "",
            view.to_markdown(index=False),
            "",
        ]
    )


def _hyperliquid_pair_cost_model_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Hyperliquid Pair Cost Model\n\nNo pair cost model rows are available.\n"
    summary = pd.DataFrame(
        [
            {"metric": "pairs", "value": len(frame)},
            {"metric": "cost_models_ready", "value": int(frame["cost_model_ready"].astype(bool).sum())},
            {"metric": "slippage_models_ready", "value": int(frame["slippage_model_ready"].astype(bool).sum())},
        ]
    )
    view = frame[
        ["pair", "leg_notional_usd", "taker_fee_bps", "pair_one_way_slippage_bps", "slippage_samples_x", "slippage_samples_y", "slippage_model_status", "blocker", "next_step"]
    ]
    return "\n".join(
        [
            "# Hyperliquid Pair Cost Model",
            "",
            "Fees use the documented conservative base perpetual tier until account-specific fee evidence is captured.",
            "Slippage becomes ready only after repeated complete public L2 samples; execution risk remains an explicit non-blocking cost assumption.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Pair Models",
            "",
            view.to_markdown(index=False),
            "",
        ]
    )


def _qualifying_slippage_samples(
    samples: pd.DataFrame,
    *,
    pair: str,
    asset: str,
    notional_usd: float,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    if samples.empty or notional_usd <= 0:
        return pd.DataFrame()
    subset = samples[
        (samples.get("pair", pd.Series(dtype=str)).astype(str) == pair)
        & (samples.get("asset", pd.Series(dtype=str)).astype(str) == asset)
    ].copy()
    if subset.empty:
        return pd.DataFrame()
    notionals = pd.to_numeric(
        subset.get("notional_usd", pd.Series(index=subset.index, dtype=float)),
        errors="coerce",
    )
    complete = subset.get("buy_complete", pd.Series(False, index=subset.index)).map(_truthy) & subset.get("sell_complete", pd.Series(False, index=subset.index)).map(_truthy)
    timestamps = pd.to_datetime(
        subset.get("source_timestamp", pd.Series(index=subset.index, dtype=object)),
        utc=True,
        errors="coerce",
        format="mixed",
    )
    slippage = pd.to_numeric(
        subset.get("one_way_slippage_bps", pd.Series(index=subset.index, dtype=float)),
        errors="coerce",
    )
    mask = (
        ((notionals - notional_usd).abs() < 1e-6)
        & complete
        & timestamps.notna()
        & slippage.notna()
    )
    if start_at is not None:
        mask &= timestamps >= start_at
    if end_at is not None:
        mask &= timestamps <= end_at
    qualified = subset.loc[mask].copy()
    if qualified.empty:
        return pd.DataFrame()
    qualified["_source_timestamp"] = timestamps.loc[mask]
    qualified["_slippage_bps"] = slippage.loc[mask]
    return qualified.sort_values("_source_timestamp").reset_index(drop=True)


def _hyperliquid_evidence_cadence_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Hyperliquid Evidence Cadence\n\nNo pair cost-model rows are available.\n"
    counts = frame["status"].value_counts().reset_index()
    counts.columns = ["status", "pairs"]
    view = frame[
        [
            "pair",
            "minimum_samples",
            "required_samples",
            "projected_captures_to_calibration",
            "projected_calibration_at",
            "next_sample_due_at",
            "calibration_deadline_at",
            "status",
            "blocker",
            "next_step",
        ]
    ]
    return "\n".join(
        [
            "# Hyperliquid Evidence Cadence",
            "",
            "A deterministic, scheduler-facing plan for public L2 depth collection. It never sends orders.",
            "The pair slippage model requires 12 complete samples per leg inside the current rolling two-hour window.",
            "Historical samples remain preserved, but samples older than the rolling window no longer count and cannot permanently expire a pair.",
            "Projected captures simulate future cadence and rolling-window expiry; this can exceed the current sample deficit.",
            "An impossible target/window/cadence combination is blocked as calibration_schedule_infeasible instead of waiting forever.",
            "A model-rebuild-due row means the raw rolling evidence is sufficient but the persisted model was built with different or stale window inputs.",
            "",
            "## Status Counts",
            "",
            counts.to_markdown(index=False),
            "",
            "## Next Evidence Tasks",
            "",
            view.to_markdown(index=False),
            "",
        ]
    )


def _hyperliquid_market_context_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(
        [
            {"metric": "markets", "value": len(frame)},
            {"metric": "tradable_markets", "value": int(frame["tradable"].astype(bool).sum())},
            {"metric": "total_24h_notional_usd", "value": float(pd.to_numeric(frame["volume_24h"], errors="coerce").fillna(0.0).sum())},
            {"metric": "total_open_interest_usd", "value": float(pd.to_numeric(frame["open_interest_usd"], errors="coerce").fillna(0.0).sum())},
            {"metric": "promotion_allowed_rows", "value": 0},
        ]
    )


def _hyperliquid_market_context_markdown(frame: pd.DataFrame, captured_at: datetime, evidence_path: str) -> str:
    if frame.empty:
        return "# Hyperliquid Market Context\n\nNo Hyperliquid market rows were captured.\n"
    summary = _hyperliquid_market_context_summary(frame)
    view = frame[
        ["asset", "tradable", "volume_24h", "open_interest_usd", "funding_rate", "liquidity_bucket", "blocker", "evidence_path"]
    ].head(50)
    return "\n".join(
        [
            "# Hyperliquid Market Context",
            "",
            "Read-only public market evidence captured from Hyperliquid's `metaAndAssetCtxs` endpoint.",
            "It is research context, not account authority and not permission to submit orders.",
            "",
            f"- captured at: {captured_at.isoformat()}",
            f"- immutable raw evidence: `{evidence_path}`",
            "- promotion allowed by this artifact: no",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Top Markets",
            "",
            view.to_markdown(index=False),
            "",
        ]
    )


def _hyperliquid_research_candidates(
    root: Path,
    *,
    max_pairs: int,
    candidate_path: Path | None = None,
) -> list[dict[str, object]]:
    active = root / "reports" / "active"
    compatibility_path = active / "hyperliquid_execution_market_compatibility.csv"
    compatibility = _read_csv(compatibility_path)
    compatible: dict[str, dict[str, object]] = {}
    for _, row in compatibility.iterrows():
        record = row.to_dict()
        if not _truthy(record.get("both_legs_testnet_perp")):
            continue
        asset_x = _coin(str(record.get("asset_x", "")))
        asset_y = _coin(str(record.get("asset_y", "")))
        pair = str(record.get("pair", "") or "")
        key = _pair_key(asset_x, asset_y)
        if pair and key:
            compatible[key] = {
                "pair": pair,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "testnet_pair_compatible": True,
            }
    selected: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_candidates(frame: pd.DataFrame, *, source: str, evidence_path: Path, sort_columns: list[str] | None = None) -> None:
        if frame.empty:
            return
        ordered = frame
        if sort_columns:
            present = [column for column in sort_columns if column in ordered.columns]
            if present:
                ordered = ordered.sort_values(present, ascending=[True] * len(present), na_position="last")
        for _, row in ordered.iterrows():
            record = row.to_dict()
            asset_x = _coin(str(record.get("asset_x", "")))
            asset_y = _coin(str(record.get("asset_y", "")))
            key = _pair_key(asset_x, asset_y)
            compatible_record = compatible.get(key)
            if not compatible_record or key in seen:
                continue
            selected.append(
                {
                    "pair": str(record.get("pair", "") or compatible_record["pair"]),
                    "asset_x": compatible_record["asset_x"],
                    "asset_y": compatible_record["asset_y"],
                    "testnet_pair_compatible": bool(compatible_record.get("testnet_pair_compatible", False)),
                    "candidate_source": source,
                    "evidence_path": _rel(evidence_path, root),
                }
            )
            seen.add(key)
            if len(selected) >= max_pairs:
                return

    if candidate_path is not None:
        explicit = _read_csv(candidate_path)
        if "wizard_discovery_pass" in explicit.columns:
            explicit = explicit.loc[explicit["wizard_discovery_pass"].map(_truthy)].copy()
        inventory = _read_csv(active / "hyperliquid_testnet_market_inventory.csv")
        tradable_assets: set[str] = set()
        if not inventory.empty and "asset" in inventory.columns:
            tradable = inventory.get("tradable_perp", pd.Series(False, index=inventory.index)).map(_truthy)
            tradable_assets = {_coin(value) for value in inventory.loc[tradable, "asset"].astype(str)}
        mainnet = _read_csv(root / "data" / "processed" / "hyperliquid_market_context.csv")
        mainnet_tradable_assets: set[str] = set()
        if not mainnet.empty and "asset" in mainnet.columns:
            mainnet_tradable = mainnet.get("tradable", pd.Series(False, index=mainnet.index)).map(_truthy)
            mainnet_tradable_assets = {_coin(value) for value in mainnet.loc[mainnet_tradable, "asset"].astype(str)}
        for _, row in explicit.iterrows():
            asset_x = _coin(str(row.get("asset_x", "")))
            asset_y = _coin(str(row.get("asset_y", "")))
            key = _pair_key(asset_x, asset_y)
            testnet_compatible = bool(key and {asset_x, asset_y}.issubset(tradable_assets))
            mainnet_compatible = bool(key and {asset_x, asset_y}.issubset(mainnet_tradable_assets))
            if testnet_compatible or mainnet_compatible:
                compatible.setdefault(
                    key,
                    {
                        "pair": str(row.get("pair", "") or f"{asset_x}-USD-{asset_y}-USD"),
                        "asset_x": asset_x,
                        "asset_y": asset_y,
                        "testnet_pair_compatible": testnet_compatible,
                    },
                )
        add_candidates(
            explicit,
            source="explicit_hyperliquid_candidate_file",
            evidence_path=candidate_path,
            sort_columns=["dashboard_rank", "shortlist_rank"],
        )
        # An explicit candidate file is an allow-list. Never substitute unrelated
        # shortlist or universe pairs when one of its symbols cannot be routed.
        return selected[:max_pairs]
    if not compatible:
        return []
    shortlist_path = active / "hyperliquid_testnet_candidate_shortlist.csv"
    add_candidates(_read_csv(shortlist_path), source="hyperliquid_testnet_candidate_shortlist", evidence_path=shortlist_path, sort_columns=["shortlist_rank"])
    if len(selected) < max_pairs:
        universe_path = root / "data" / "processed" / "pair_universe.csv"
        universe = _read_csv(universe_path)
        if not universe.empty and "combined_score" in universe.columns:
            universe = universe.assign(_rank=pd.to_numeric(universe["combined_score"], errors="coerce").fillna(float("-inf")))
            universe = universe.sort_values("_rank", ascending=False)
        add_candidates(universe, source="pair_universe_hyperliquid_compatible", evidence_path=universe_path)
    if len(selected) < max_pairs:
        for key, record in sorted(compatible.items(), key=lambda item: item[1]["pair"]):
            if key in seen:
                continue
            selected.append(
                {
                    "pair": record["pair"],
                    "asset_x": record["asset_x"],
                    "asset_y": record["asset_y"],
                    "testnet_pair_compatible": bool(record.get("testnet_pair_compatible", False)),
                    "candidate_source": "hyperliquid_testnet_compatibility",
                    "evidence_path": _rel(compatibility_path, root),
                }
            )
            seen.add(key)
            if len(selected) >= max_pairs:
                break
    return selected[:max_pairs]


def _candle_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    candles = payload.get("candles", [])
    latest = ""
    if isinstance(candles, list) and candles:
        last = candles[-1]
        if isinstance(last, dict):
            latest = str(last.get("startedAt", "") or "")
    return {
        "fetched_at": str(payload.get("fetched_at", "") or ""),
        "rows": len(candles) if isinstance(candles, list) else 0,
        "latest_candle_at": latest,
    }


def _write_hyperliquid_funding_record(
    *,
    coin: str,
    records: list[dict[str, object]],
    requests_made: list[dict[str, object]],
    captured_at: datetime,
    output_dir: Path,
    info_url: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    captured = _utc_datetime(captured_at)
    output = output_dir / f"{coin}_funding.json"
    snapshot = output_dir / "snapshots" / captured.strftime("%Y-%m-%d_%H%M%S") / output.name
    payload = {
        "source": "hyperliquid_public_funding_history",
        "info_url": info_url,
        "coin": coin,
        "requests": requests_made,
        "funding": records,
        "fetch_complete": True,
        "fetched_at": captured.isoformat(),
        "snapshot_path": str(snapshot),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    output.write_text(serialized, encoding="utf-8")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(serialized, encoding="utf-8")
    return output


def _merge_hyperliquid_funding_into_pair_histories(
    *,
    root: Path,
    candidates: list[dict[str, str]],
    funding: pd.DataFrame,
    funding_path: Path,
) -> pd.DataFrame:
    prepared = funding.copy()
    if not prepared.empty:
        prepared["timestamp"] = pd.to_datetime(
            prepared["timestamp"], utc=True, errors="coerce", format="mixed"
        )
        prepared["funding_bps"] = pd.to_numeric(prepared["funding_bps"], errors="coerce")
        prepared = prepared.dropna(subset=["timestamp", "funding_bps"])
        prepared["day"] = prepared["timestamp"].dt.floor("D")
    daily = (
        prepared.groupby(["coin", "day"], as_index=False)["funding_bps"].sum()
        if not prepared.empty
        else pd.DataFrame(columns=["coin", "day", "funding_bps"])
    )
    daily_maps = {
        coin: dict(zip(group["day"], group["funding_bps"]))
        for coin, group in daily.groupby("coin")
    }
    hourly_counts = prepared["coin"].value_counts().to_dict() if not prepared.empty else {}
    daily_counts = daily["coin"].value_counts().to_dict() if not daily.empty else {}
    rows: list[dict[str, object]] = []

    for candidate in candidates:
        asset_x = _coin(candidate["asset_x"])
        asset_y = _coin(candidate["asset_y"])
        pair_key = _pair_key(asset_x, asset_y)
        matching: list[tuple[Path, dict[str, Any]]] = []
        for path in sorted((root / "data" / "raw" / "pair_details").glob("*hyperliquid*derived_history.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict) or str(payload.get("exchange", "")).lower() != "hyperliquid":
                continue
            if _pair_key(str(payload.get("asset_x", "")), str(payload.get("asset_y", ""))) == pair_key:
                matching.append((path, payload))

        if not matching:
            rows.append(
                _hyperliquid_funding_coverage_row(
                    candidate=candidate,
                    interval="",
                    history_path=None,
                    history_rows=0,
                    aligned_x=0,
                    aligned_y=0,
                    aligned_both=0,
                    hourly_counts=hourly_counts,
                    daily_counts=daily_counts,
                    funding_path=funding_path,
                    root=root,
                )
            )
            continue

        for history_path, payload in matching:
            history = payload.get("history", [])
            if not isinstance(history, list):
                history = []
            aligned_x = 0
            aligned_y = 0
            aligned_both = 0
            map_x = daily_maps.get(asset_x, {})
            map_y = daily_maps.get(asset_y, {})
            for point in history:
                if not isinstance(point, dict):
                    continue
                timestamp = pd.to_datetime(point.get("timestamp", point.get("startedAt")), utc=True, errors="coerce")
                day = timestamp.floor("D") if pd.notna(timestamp) else pd.NaT
                funding_x = map_x.get(day) if pd.notna(day) else None
                funding_y = map_y.get(day) if pd.notna(day) else None
                point["funding_x_bps"] = float(funding_x) if funding_x is not None else None
                point["funding_y_bps"] = float(funding_y) if funding_y is not None else None
                point["funding_bps_per_day"] = (
                    abs(float(funding_x)) + abs(float(funding_y))
                    if funding_x is not None and funding_y is not None
                    else None
                )
                aligned_x += int(funding_x is not None)
                aligned_y += int(funding_y is not None)
                aligned_both += int(funding_x is not None and funding_y is not None)
            payload["history"] = history
            payload["funding_source"] = "hyperliquid_public_funding_history"
            payload["funding_units"] = "bps_per_utc_day_sum_of_hourly_funding_rates"
            payload["funding_alignment"] = "same_utc_day_no_future_fill"
            payload["funding_cost_policy"] = "conservative_absolute_leg_drag"
            payload["funding_evidence_path"] = _rel(funding_path, root)
            history_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            rows.append(
                _hyperliquid_funding_coverage_row(
                    candidate=candidate,
                    interval=str(payload.get("interval", "")),
                    history_path=history_path,
                    history_rows=len(history),
                    aligned_x=aligned_x,
                    aligned_y=aligned_y,
                    aligned_both=aligned_both,
                    hourly_counts=hourly_counts,
                    daily_counts=daily_counts,
                    funding_path=funding_path,
                    root=root,
                )
            )
    return pd.DataFrame(rows, columns=HYPERLIQUID_FUNDING_COVERAGE_COLUMNS)


def _hyperliquid_funding_coverage_row(
    *,
    candidate: dict[str, str],
    interval: str,
    history_path: Path | None,
    history_rows: int,
    aligned_x: int,
    aligned_y: int,
    aligned_both: int,
    hourly_counts: dict[str, int],
    daily_counts: dict[str, int],
    funding_path: Path,
    root: Path,
) -> dict[str, object]:
    asset_x = _coin(candidate["asset_x"])
    asset_y = _coin(candidate["asset_y"])
    coverage = aligned_both / history_rows if history_rows else 0.0
    blockers = []
    if history_path is None:
        blockers.append("missing_matching_hyperliquid_pair_history")
    if hourly_counts.get(asset_x, 0) == 0:
        blockers.append("missing_hyperliquid_funding_asset_x")
    if hourly_counts.get(asset_y, 0) == 0:
        blockers.append("missing_hyperliquid_funding_asset_y")
    if history_rows and coverage < 0.95:
        blockers.append("hyperliquid_funding_coverage_below_95pct")
    ready = bool(history_rows and coverage >= 0.95 and not blockers)
    evidence = [str(candidate.get("evidence_path", "")), _rel(funding_path, root)]
    if history_path is not None:
        evidence.append(_rel(history_path, root))
    return {
        "pair": candidate["pair"],
        "asset_x": asset_x,
        "asset_y": asset_y,
        "interval": interval,
        "history_path": _rel(history_path, root) if history_path is not None else "",
        "history_rows": history_rows,
        "funding_x_hourly_rows": int(hourly_counts.get(asset_x, 0)),
        "funding_y_hourly_rows": int(hourly_counts.get(asset_y, 0)),
        "funding_x_daily_rows": int(daily_counts.get(asset_x, 0)),
        "funding_y_daily_rows": int(daily_counts.get(asset_y, 0)),
        "funding_x_aligned_rows": aligned_x,
        "funding_y_aligned_rows": aligned_y,
        "both_legs_aligned_rows": aligned_both,
        "funding_coverage_pct": coverage * 100.0,
        "funding_ready": ready,
        "funding_cost_policy": "conservative_absolute_leg_drag",
        "blocker": ";".join(blockers),
        "next_step": "rerun_costed_walk_forward" if ready else "repair_hyperliquid_funding_coverage",
        "evidence_path": ";".join(value for value in evidence if value),
    }


def _hyperliquid_funding_coverage_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Hyperliquid Funding Coverage\n\nNo active pair rows were available.\n"
    summary = pd.DataFrame(
        [
            {"metric": "pair_histories", "value": len(frame)},
            {"metric": "funding_ready", "value": int(frame["funding_ready"].astype(bool).sum())},
            {"metric": "minimum_required_coverage_pct", "value": 95.0},
            {"metric": "future_fill_allowed", "value": False},
            {"metric": "cost_policy", "value": "conservative_absolute_leg_drag"},
        ]
    )
    view = frame[
        [
            "pair",
            "interval",
            "history_rows",
            "both_legs_aligned_rows",
            "funding_coverage_pct",
            "funding_ready",
            "blocker",
            "next_step",
        ]
    ]
    return "\n".join(
        [
            "# Hyperliquid Funding Coverage",
            "",
            "Hourly public funding rates are summed by UTC day and aligned only to the same daily bar.",
            "Missing history is left missing; no future funding observation is backfilled into an earlier bar.",
            "Backtests charge absolute funding on both legs as a conservative drag rather than assuming a credit.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Pair Coverage",
            "",
            view.to_markdown(index=False),
            "",
        ]
    )


def _pair_history_metadata(path: Path) -> tuple[int, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, ""
    history = payload.get("history", []) if isinstance(payload, dict) else []
    if not isinstance(history, list) or not history:
        return 0, ""
    last = history[-1] if isinstance(history[-1], dict) else {}
    return len(history), str(last.get("startedAt", last.get("timestamp", "")) or "")


def _required_history_rows(interval: str) -> int:
    return 120 if interval in {"1d", "3d", "1w", "1M"} else 1_000


def _timestamp_is_fresh(value: object, *, max_age_hours: int = 24) -> bool:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return False
    return bool(pd.Timestamp.now(tz="UTC") - timestamp <= pd.Timedelta(hours=max_age_hours))


def _pair_key(asset_x: str, asset_y: str) -> str:
    assets = sorted(value for value in (_coin(asset_x), _coin(asset_y)) if value)
    return "|".join(assets)


def _truthy(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _utc_datetime(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _liquidity_bucket(volume_24h: float) -> str:
    if volume_24h >= 100_000_000:
        return "deep"
    if volume_24h >= 10_000_000:
        return "usable"
    if volume_24h > 0:
        return "thin"
    return "unknown"


def _hyperliquid_research_bundle_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "# Hyperliquid Research Bundle\n\nNo Testnet-compatible Hyperliquid candidates were available.\n"
    summary = pd.DataFrame(
        [
            {"metric": "pairs", "value": int(frame["pair"].nunique())},
            {"metric": "pair_timeframes", "value": len(frame)},
            {"metric": "history_ready", "value": int(frame["history_ready"].astype(bool).sum())},
            {"metric": "blocked", "value": int((~frame["history_ready"].astype(bool)).sum())},
        ]
    )
    view = frame[
        ["pair", "interval", "history_rows", "required_history_rows", "history_ready", "history_status", "blocker", "next_step"]
    ]
    return "\n".join(
        [
            "# Hyperliquid Research Bundle",
            "",
            "Fresh two-leg candle histories for Testnet-compatible research pairs.",
            "This proves only local history availability. It does not satisfy cost, slippage, funding, strategy acceptance, or order-submission gates.",
            "",
            "## Summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Pair-Timeframe Evidence",
            "",
            view.to_markdown(index=False),
            "",
        ]
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
    text = str(value or "").upper().strip().replace("_", "-")
    if text.endswith("-USD"):
        text = text[:-4]
    if text.endswith("/USD"):
        text = text[:-4]
    if "-" not in text and "/" not in text:
        for suffix in ("USDT", "USDC", "PERP", "USD"):
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                break
    return text


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
