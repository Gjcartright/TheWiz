"""Point-in-time Hyperliquid funding evidence for the exhaustive run."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import time
from typing import Callable

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.hyperliquid import (
    fetch_hyperliquid_funding_history,
    normalize_hyperliquid_funding_history,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_funding_evidence.v1"
MINIMUM_PAIR_FUNDING_COVERAGE = 0.95


def materialize_exhaustive_hyperliquid_funding_evidence(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    max_assets: int = 0,
    max_attempts: int = 2,
    fetcher: Callable[..., Path] = fetch_hyperliquid_funding_history,
    sleep: Callable[[float], None] = time.sleep,
) -> CommandResult:
    """Fetch resumable funding histories and write immutable enriched pair copies.

    ``max_assets=0`` processes every pending asset. A positive value bounds only
    new network fetches; every queued asset and pair still appears in the output.
    """

    if max_assets < 0 or max_attempts <= 0:
        raise ValueError("max_assets cannot be negative and max_attempts must be positive")
    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    history_manifest_path = active / "exhaustive_wizard_hyperliquid_history_manifest.json"
    if not history_manifest_path.exists():
        raise FileNotFoundError("Exhaustive Hyperliquid history manifest is missing")
    history_manifest = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    run_id = _text(history_manifest.get("run_id"))
    preflight_id = _text(history_manifest.get("replay_preflight_id"))
    history_run_id = _text(history_manifest.get("history_run_id"))
    artifacts = history_manifest.get("artifacts", {})
    asset_history_path = root / _text(artifacts.get("snapshot_asset_results"))
    pair_history_path = root / _text(artifacts.get("snapshot_pair_results"))
    history_snapshot_manifest = root / _text(artifacts.get("snapshot_manifest"))
    required = [asset_history_path, pair_history_path, history_snapshot_manifest]
    missing = [str(path) for path in required if not path.exists()]
    if not run_id or missing:
        raise FileNotFoundError(f"Funding evidence inputs missing: {missing}")

    assets = _read_csv(asset_history_path)
    pairs = _read_csv(pair_history_path)
    queue = _build_funding_queue(
        assets,
        run_id=run_id,
        preflight_id=preflight_id,
        history_run_id=history_run_id,
        evidence_path=_relative(asset_history_path, root),
    )
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "history_run_id": history_run_id,
        "asset_history_hash": _file_hash(asset_history_path),
        "pair_history_hash": _file_hash(pair_history_path),
        "minimum_pair_funding_coverage": MINIMUM_PAIR_FUNDING_COVERAGE,
        "queue": queue.to_dict("records"),
    }
    funding_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    funding_evidence_id = f"hlfunding_{funding_hash[:20]}"
    snapshot_dir = history_snapshot_manifest.parent / "funding_evidence" / funding_evidence_id
    asset_dir = snapshot_dir / "assets"
    pair_dir = snapshot_dir / "pairs"
    checkpoint_dir = snapshot_dir / "checkpoints" / as_of.strftime("%Y%m%dT%H%M%S%fZ")
    for directory in (asset_dir, pair_dir, checkpoint_dir):
        directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "queue": active / "exhaustive_wizard_hyperliquid_funding_queue.csv",
        "asset_results": active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        "pair_coverage": active / "exhaustive_wizard_hyperliquid_funding_pair_coverage.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_funding_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_funding_summary.md",
        "snapshot_queue": snapshot_dir / "funding_queue.csv",
        "snapshot_asset_results": snapshot_dir / "asset_results.csv",
        "snapshot_pair_coverage": snapshot_dir / "pair_coverage.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
        "checkpoint_asset_results": checkpoint_dir / "asset_results.csv",
        "checkpoint_pair_coverage": checkpoint_dir / "pair_coverage.csv",
    }
    queue.to_csv(paths["queue"], index=False)
    queue.to_csv(paths["snapshot_queue"], index=False)

    fetched_assets = 0
    asset_rows: list[dict[str, object]] = []
    for request in queue.itertuples():
        asset = _text(request.asset)
        cutoff = _parse_timestamp(request.fetch_end_at)
        output_path = asset_dir / f"{asset}_funding.json"
        metadata = _funding_file_metadata(output_path, asset=asset, cutoff=cutoff)
        attempts = 0
        last_error = ""
        if not metadata["funding_complete"] and (max_assets == 0 or fetched_assets < max_assets):
            fetched_assets += 1
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    returned = fetcher(
                        coin=asset,
                        days=int(request.history_days_requested),
                        output_dir=asset_dir,
                        end_time=cutoff,
                        page_interval_seconds=0.0,
                    )
                    output_path = Path(returned)
                    break
                except Exception as exc:  # Keep the asset and allow a later resume.
                    last_error = f"{type(exc).__name__}:{exc}"
                    if attempt < max_attempts:
                        sleep(float(2 ** (attempt - 1)))
            metadata = _funding_file_metadata(output_path, asset=asset, cutoff=cutoff)

        if metadata["funding_complete"]:
            status = "COMPLETE"
            blocker = ""
        elif last_error:
            status = "RETRYABLE_FETCH_BLOCKER"
            blocker = f"hyperliquid_funding_fetch_failed:{last_error}"
        elif output_path.exists():
            status = "RETRYABLE_INCOMPLETE_CHECKPOINT"
            blocker = _text(metadata["blocker"]) or "funding_checkpoint_incomplete"
        else:
            status = "PENDING"
            blocker = "funding_fetch_not_started"
        asset_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "history_run_id": history_run_id,
                "funding_evidence_id": funding_evidence_id,
                "funding_request_id": _text(request.funding_request_id),
                "asset": asset,
                "fetch_start_at": _text(request.fetch_start_at),
                "fetch_end_at": _text(request.fetch_end_at),
                "history_days_requested": int(request.history_days_requested),
                "attempts_this_checkpoint": attempts,
                "funding_rows": int(metadata["rows"]),
                "earliest_funding_at": _text(metadata["earliest"]),
                "latest_funding_at": _text(metadata["latest"]),
                "post_cutoff_rows": int(metadata["post_cutoff_rows"]),
                "timestamp_parse_valid": bool(metadata["timestamp_parse_valid"]),
                "fetch_complete_flag": bool(metadata["fetch_complete_flag"]),
                "funding_status": status,
                "funding_blocker": blocker,
                "funding_path": _relative(output_path, root) if output_path.exists() else "",
                "funding_sha256": _file_hash(output_path),
                "queue_evidence_path": _relative(paths["snapshot_queue"], root),
                "live_trading_authorized": False,
            }
        )
    asset_results = pd.DataFrame(asset_rows)
    asset_lookup = {_text(row.asset): row for row in asset_results.itertuples()}
    funding_cache: dict[str, pd.DataFrame] = {}
    pair_rows: list[dict[str, object]] = []
    for pair in pairs.itertuples():
        pair_rows.append(
            _materialize_pair_funding(
                pair,
                asset_lookup=asset_lookup,
                funding_cache=funding_cache,
                pair_dir=pair_dir,
                root=root,
                run_id=run_id,
                preflight_id=preflight_id,
                history_run_id=history_run_id,
                funding_evidence_id=funding_evidence_id,
            )
        )
    pair_coverage = pd.DataFrame(pair_rows)
    for frame, active_path, snapshot_path, checkpoint_path in (
        (
            asset_results,
            paths["asset_results"],
            paths["snapshot_asset_results"],
            paths["checkpoint_asset_results"],
        ),
        (
            pair_coverage,
            paths["pair_coverage"],
            paths["snapshot_pair_coverage"],
            paths["checkpoint_pair_coverage"],
        ),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)
        frame.to_csv(checkpoint_path, index=False)

    pair_status_counts = {
        _text(status) or "MISSING_STATUS": int(count)
        for status, count in pair_coverage["funding_status"].value_counts(dropna=False).items()
    }
    known_pair_statuses = {
        "READY_FOR_COST_REPLAY",
        "PARTIAL_FUNDING_COVERAGE",
        "PENDING_ASSET_FUNDING",
        "BLOCKED_PAIR_HISTORY",
        "BLOCKED_FUNDING_ALIGNMENT",
    }
    unclassified_pair_statuses = int(
        (~pair_coverage["funding_status"].isin(known_pair_statuses)).sum()
    )
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "checkpoint_created_at": as_of.isoformat(),
        "asset_requests": int(len(asset_results)),
        "assets_fetched_this_checkpoint": fetched_assets,
        "asset_funding_complete": int(asset_results["funding_status"].eq("COMPLETE").sum()),
        "asset_funding_pending": int(asset_results["funding_status"].eq("PENDING").sum()),
        "asset_funding_retryable": int(
            asset_results["funding_status"].str.startswith("RETRYABLE").sum()
        ),
        "pair_work_items": int(len(pair_coverage)),
        "pairs_ready_for_cost_replay": int(
            pair_coverage["funding_status"].eq("READY_FOR_COST_REPLAY").sum()
        ),
        "pairs_partial_funding_coverage": int(
            pair_coverage["funding_status"].eq("PARTIAL_FUNDING_COVERAGE").sum()
        ),
        "pairs_pending_asset_funding": int(
            pair_coverage["funding_status"].eq("PENDING_ASSET_FUNDING").sum()
        ),
        "pairs_blocked_pair_history": int(
            pair_coverage["funding_status"].eq("BLOCKED_PAIR_HISTORY").sum()
        ),
        "pairs_blocked_funding_alignment": int(
            pair_coverage["funding_status"].eq("BLOCKED_FUNDING_ALIGNMENT").sum()
        ),
        "pairs_unclassified_status": unclassified_pair_statuses,
        "pair_status_counts": pair_status_counts,
        "pair_status_accounted": bool(sum(pair_status_counts.values()) == len(pair_coverage)),
        "minimum_pair_funding_coverage": MINIMUM_PAIR_FUNDING_COVERAGE,
        "alignment_policy": "same_realized_utc_hour_or_day_no_future_fill",
        "all_asset_funding_complete": bool(asset_results["funding_status"].eq("COMPLETE").all()),
        "live_trading_authorized": False,
        "input_hashes": {
            "asset_history_results": _file_hash(asset_history_path),
            "pair_history_results": _file_hash(pair_history_path),
            "funding_queue": _file_hash(paths["snapshot_queue"]),
        },
        "funding_queue_policy": {
            "assets": int(len(queue)),
            "request_start_from_earliest_candle": True,
            "request_end_at_frozen_cutoff": True,
            "network_identity_deduplicated_by_asset": True,
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _build_funding_queue(
    assets: pd.DataFrame,
    *,
    run_id: str,
    preflight_id: str,
    history_run_id: str,
    evidence_path: str,
) -> pd.DataFrame:
    complete = assets.loc[assets["history_status"].eq("COMPLETE")].copy()
    rows: list[dict[str, object]] = []
    for asset, group in complete.groupby("asset", sort=True):
        cutoffs = [
            timestamp
            for value in group["fetch_end_at"]
            if (timestamp := _parse_timestamp(value)) is not None
        ]
        starts = [
            timestamp
            for value in group["earliest_candle_at"]
            if (timestamp := _parse_timestamp(value)) is not None
        ]
        cutoff = min(cutoffs) if cutoffs else None
        start = min(starts) if starts else None
        days = (
            max(1, int(math.ceil((cutoff - start).total_seconds() / 86_400.0)) + 2)
            if cutoff is not None and start is not None
            else int(pd.to_numeric(group["history_days_requested"], errors="coerce").max())
        )
        request_material = f"{run_id}|{asset}|{start}|{cutoff}|{days}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "history_run_id": history_run_id,
                "funding_request_id": "hlfund_"
                + sha256(request_material.encode("utf-8")).hexdigest()[:20],
                "asset": _text(asset),
                "fetch_start_at": start.isoformat() if start else "",
                "fetch_end_at": cutoff.isoformat() if cutoff else "",
                "history_days_requested": days,
                "asset_history_request_count": int(len(group)),
                "asset_history_intervals": ";".join(
                    sorted(set(group["hyperliquid_interval"].astype(str)))
                ),
                "funding_status": "NOT_FETCHED",
                "evidence_path": evidence_path,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _funding_file_metadata(
    path: Path,
    *,
    asset: str,
    cutoff: datetime | None,
) -> dict[str, object]:
    empty = {
        "rows": 0,
        "earliest": "",
        "latest": "",
        "post_cutoff_rows": 0,
        "timestamp_parse_valid": False,
        "fetch_complete_flag": False,
        "funding_complete": False,
        "blocker": "funding_file_missing",
    }
    if not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {**empty, "blocker": "funding_file_unreadable"}
    records = payload.get("funding", []) if isinstance(payload, dict) else []
    normalized = normalize_hyperliquid_funding_history(payload, coin=asset)
    timestamps = [
        timestamp
        for row in normalized
        if (timestamp := _parse_timestamp(row.get("timestamp"))) is not None
    ]
    rows = len(records) if isinstance(records, list) else 0
    parse_valid = rows > 0 and len(timestamps) == rows and cutoff is not None
    post_cutoff_rows = (
        sum(timestamp > cutoff for timestamp in timestamps) if cutoff is not None else 0
    )
    fetch_complete_flag = (
        bool(payload.get("fetch_complete")) if isinstance(payload, dict) else False
    )
    funding_complete = bool(
        fetch_complete_flag and parse_valid and post_cutoff_rows == 0 and rows > 0
    )
    blocker = ""
    if not fetch_complete_flag:
        blocker = "funding_fetch_complete_flag_false"
    elif not parse_valid:
        blocker = "funding_timestamp_parse_invalid"
    elif post_cutoff_rows:
        blocker = "funding_contains_post_cutoff_rows"
    return {
        "rows": rows,
        "earliest": min(timestamps).isoformat() if timestamps else "",
        "latest": max(timestamps).isoformat() if timestamps else "",
        "post_cutoff_rows": post_cutoff_rows,
        "timestamp_parse_valid": parse_valid,
        "fetch_complete_flag": fetch_complete_flag,
        "funding_complete": funding_complete,
        "blocker": blocker,
    }


def _materialize_pair_funding(
    pair: object,
    *,
    asset_lookup: dict[str, object],
    funding_cache: dict[str, pd.DataFrame],
    pair_dir: Path,
    root: Path,
    run_id: str,
    preflight_id: str,
    history_run_id: str,
    funding_evidence_id: str,
) -> dict[str, object]:
    asset_x = _text(pair.asset_x)
    asset_y = _text(pair.asset_y)
    x_result = asset_lookup.get(asset_x)
    y_result = asset_lookup.get(asset_y)
    base = {
        "schema_version": SCHEMA_VERSION,
        "exhaustive_run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "funding_evidence_id": funding_evidence_id,
        "pair_group_id": _text(pair.pair_group_id),
        "pair": _text(pair.pair),
        "wizard_exchange": _text(pair.wizard_exchange),
        "wizard_timeframe": _text(pair.wizard_timeframe),
        "hyperliquid_interval": _text(pair.hyperliquid_interval),
        "asset_x": asset_x,
        "asset_y": asset_y,
        "history_rows": int(getattr(pair, "history_rows", 0) or 0),
        "funding_x_aligned_rows": 0,
        "funding_y_aligned_rows": 0,
        "funding_both_aligned_rows": 0,
        "funding_longest_contiguous_rows": 0,
        "funding_x_coverage": 0.0,
        "funding_y_coverage": 0.0,
        "funding_both_coverage": 0.0,
        "funding_status": "",
        "funding_blocker": "",
        "funding_acceptance_ready": False,
        "enriched_history_path": "",
        "enriched_history_sha256": "",
        "evidence_path": _text(getattr(pair, "evidence_path", "")),
        "live_trading_authorized": False,
    }
    if _text(pair.history_status) != "READY_FOR_CANONICAL_REPLAY":
        return {
            **base,
            "funding_status": "BLOCKED_PAIR_HISTORY",
            "funding_blocker": _text(pair.history_blocker) or "pair_history_not_ready",
        }
    unavailable = [
        asset
        for asset, result in ((asset_x, x_result), (asset_y, y_result))
        if result is None or _text(result.funding_status) != "COMPLETE"
    ]
    if unavailable:
        return {
            **base,
            "funding_status": "PENDING_ASSET_FUNDING",
            "funding_blocker": ";".join(
                f"asset_funding_not_complete:{asset}" for asset in unavailable
            ),
        }
    history_path = root / _text(pair.history_path)
    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        history = pd.DataFrame(payload.get("history", []))
        if history.empty or "timestamp" not in history.columns:
            raise ValueError("pair_history_missing_timestamp")
        history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
        if history["timestamp"].isna().any():
            raise ValueError("pair_history_timestamp_invalid")
        funding_x = _load_funding_frame(root / _text(x_result.funding_path), asset_x, funding_cache)
        funding_y = _load_funding_frame(root / _text(y_result.funding_path), asset_y, funding_cache)
        frequency = "h" if _text(pair.hyperliquid_interval) == "1h" else "D"
        map_x = _funding_bucket_map(funding_x, frequency=frequency)
        map_y = _funding_bucket_map(funding_y, frequency=frequency)
        buckets = history["timestamp"].dt.floor(frequency)
        aligned_x = buckets.map(map_x)
        aligned_y = buckets.map(map_y)
        history["funding_x_bps"] = aligned_x
        history["funding_y_bps"] = aligned_y
        history["funding_x_realized_bps"] = aligned_x
        history["funding_y_realized_bps"] = aligned_y
        history["funding_bps_per_day"] = aligned_x.abs() + aligned_y.abs()
        history["funding_realized_abs_bps_per_bar"] = aligned_x.abs() + aligned_y.abs()
        rows = len(history)
        count_x = int(aligned_x.notna().sum())
        count_y = int(aligned_y.notna().sum())
        count_both = int((aligned_x.notna() & aligned_y.notna()).sum())
        longest_contiguous = _longest_contiguous_funding_run(
            aligned_x.notna() & aligned_y.notna(),
            history["timestamp"],
            frequency=frequency,
        )
        coverage_x = count_x / rows if rows else 0.0
        coverage_y = count_y / rows if rows else 0.0
        coverage_both = count_both / rows if rows else 0.0
        payload["history"] = _json_records(history)
        payload["funding_source"] = "hyperliquid_public_funding_history"
        payload["funding_alignment"] = (
            f"same_realized_utc_{'hour' if frequency == 'h' else 'day'}_no_future_fill"
        )
        payload["funding_cost_policy"] = "signed_rates_preserved_for_backtest_policy"
        payload["funding_rate_semantics"] = "realized_bps_per_aligned_candle"
        payload["funding_evidence_id"] = funding_evidence_id
        output_path = pair_dir / f"{_safe_filename(_text(pair.pair_group_id))}_funding.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        return {
            **base,
            "funding_status": "BLOCKED_FUNDING_ALIGNMENT",
            "funding_blocker": f"{type(exc).__name__}:{exc}",
        }
    ready = bool(
        coverage_x >= MINIMUM_PAIR_FUNDING_COVERAGE and coverage_y >= MINIMUM_PAIR_FUNDING_COVERAGE
    )
    blockers = []
    if coverage_x < MINIMUM_PAIR_FUNDING_COVERAGE:
        blockers.append("insufficient_funding_coverage_asset_x")
    if coverage_y < MINIMUM_PAIR_FUNDING_COVERAGE:
        blockers.append("insufficient_funding_coverage_asset_y")
    evidence = [
        _text(getattr(pair, "evidence_path", "")),
        _text(x_result.funding_path),
        _text(y_result.funding_path),
        _relative(output_path, root),
    ]
    return {
        **base,
        "history_rows": rows,
        "funding_x_aligned_rows": count_x,
        "funding_y_aligned_rows": count_y,
        "funding_both_aligned_rows": count_both,
        "funding_longest_contiguous_rows": longest_contiguous,
        "funding_x_coverage": coverage_x,
        "funding_y_coverage": coverage_y,
        "funding_both_coverage": coverage_both,
        "funding_status": "READY_FOR_COST_REPLAY" if ready else "PARTIAL_FUNDING_COVERAGE",
        "funding_blocker": ";".join(blockers),
        "funding_acceptance_ready": ready,
        "enriched_history_path": _relative(output_path, root),
        "enriched_history_sha256": _file_hash(output_path),
        "evidence_path": ";".join(value for value in evidence if value),
    }


def _load_funding_frame(
    path: Path,
    asset: str,
    cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    key = str(path)
    if key not in cache:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(normalize_hyperliquid_funding_history(payload, coin=asset))
        if frame.empty:
            raise ValueError(f"normalized_funding_empty:{asset}")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame["funding_bps"] = pd.to_numeric(frame["funding_bps"], errors="coerce")
        cache[key] = frame.dropna(subset=["timestamp", "funding_bps"])
    return cache[key]


def _funding_bucket_map(frame: pd.DataFrame, *, frequency: str) -> dict[pd.Timestamp, float]:
    grouped = (
        frame.assign(bucket=frame["timestamp"].dt.floor(frequency))
        .groupby("bucket")["funding_bps"]
        .sum()
    )
    return grouped.to_dict()


def _longest_contiguous_funding_run(
    valid: pd.Series,
    timestamps: pd.Series,
    *,
    frequency: str,
) -> int:
    expected = pd.Timedelta(hours=1) if frequency == "h" else pd.Timedelta(days=1)
    longest = 0
    current = 0
    previous: pd.Timestamp | None = None
    for is_valid, timestamp in zip(valid.tolist(), timestamps.tolist(), strict=True):
        parsed = pd.Timestamp(timestamp)
        contiguous = bool(
            previous is not None and pd.Timedelta(0) < parsed - previous <= expected * 1.5
        )
        if not is_valid:
            current = 0
            previous = None
            continue
        current = current + 1 if contiguous else 1
        longest = max(longest, current)
        previous = parsed
    return longest


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    working = frame.copy()
    working["timestamp"] = working["timestamp"].map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return working.where(pd.notna(working), None).to_dict("records")


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Hyperliquid Funding Evidence",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Funding evidence: `{summary['funding_evidence_id']}`",
            f"- Asset funding complete: {summary['asset_funding_complete']} / {summary['asset_requests']}",
            f"- Assets fetched this checkpoint: {summary['assets_fetched_this_checkpoint']}",
            f"- Asset funding pending: {summary['asset_funding_pending']}",
            f"- Asset funding retryable: {summary['asset_funding_retryable']}",
            f"- Pairs ready for cost replay: {summary['pairs_ready_for_cost_replay']} / {summary['pair_work_items']}",
            f"- Pairs with partial funding coverage: {summary['pairs_partial_funding_coverage']}",
            f"- Pairs pending asset funding: {summary['pairs_pending_asset_funding']}",
            f"- Pairs blocked by pair history: {summary['pairs_blocked_pair_history']}",
            f"- Pairs blocked by funding alignment: {summary['pairs_blocked_funding_alignment']}",
            f"- Unclassified pair statuses: {summary['pairs_unclassified_status']}",
            f"- All pair statuses accounted: {summary['pair_status_accounted']}",
            f"- Minimum pair funding coverage: {summary['minimum_pair_funding_coverage']:.0%}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Funding is aligned only to the same realized UTC hour or day. Missing buckets remain missing; no future fill or dashboard hindsight is used.",
            "",
        ]
    )


def _safe_filename(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_" for character in value
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_timestamp(value: object) -> datetime | None:
    timestamp = pd.to_datetime(_text(value), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
