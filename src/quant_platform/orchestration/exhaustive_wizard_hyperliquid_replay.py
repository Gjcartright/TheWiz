"""Replay preflight for the exhaustive Wizard to Hyperliquid run.

The preflight accounts for every experiment before any history is fetched. It
deduplicates network requests, never pair groups, modes, or orientations.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Callable

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.hyperliquid import (
    build_hyperliquid_pair_history,
    fetch_hyperliquid_candles,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_replay_preflight.v1"
HISTORY_SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_history.v1"
TIMEFRAME_INTERVALS = {"daily": "1d", "hourly": "1h"}
HISTORY_DAYS = {"1d": 1_500, "1h": 210}
MINIMUM_HISTORY_ROWS = {"1d": 750, "1h": 1_000}
ASSET_HISTORY_RESULT_COLUMNS = [
    "schema_version",
    "exhaustive_run_id",
    "replay_preflight_id",
    "history_run_id",
    "asset_fetch_request_id",
    "asset",
    "hyperliquid_interval",
    "fetch_end_at",
    "history_days_requested",
    "minimum_history_rows",
    "attempts",
    "history_rows",
    "earliest_candle_at",
    "latest_candle_at",
    "timestamp_parse_valid",
    "post_snapshot_rows",
    "timestamp_bound_valid",
    "history_status",
    "history_blocker",
    "history_path",
    "history_sha256",
    "queue_evidence_path",
    "live_trading_authorized",
]
PAIR_HISTORY_RESULT_COLUMNS = [
    "schema_version",
    "exhaustive_run_id",
    "replay_preflight_id",
    "history_run_id",
    "history_request_id",
    "pair_group_id",
    "pair",
    "wizard_exchange",
    "wizard_timeframe",
    "hyperliquid_interval",
    "asset_x",
    "asset_y",
    "scanner_cutoff_at",
    "minimum_history_rows",
    "history_rows",
    "earliest_candle_at",
    "latest_candle_at",
    "timestamp_parse_valid",
    "post_snapshot_rows",
    "timestamp_bound_valid",
    "ready_mode_orientation_cells",
    "history_status",
    "history_blocker",
    "history_path",
    "history_sha256",
    "evidence_path",
    "canonical_replay_leverage",
    "live_trading_authorized",
]


def materialize_exhaustive_wizard_hyperliquid_history(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    max_attempts: int = 3,
    fetcher: Callable[..., Path] = fetch_hyperliquid_candles,
    sleep: Callable[[float], None] = time.sleep,
) -> CommandResult:
    """Fetch bounded public candles and build every ready pair-group history."""

    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    preflight_manifest_path = (
        active / "exhaustive_wizard_hyperliquid_replay_preflight_manifest.json"
    )
    if not preflight_manifest_path.exists():
        raise FileNotFoundError("Exhaustive replay preflight manifest is missing")
    preflight = json.loads(preflight_manifest_path.read_text(encoding="utf-8"))
    run_id = _text(preflight.get("run_id"))
    preflight_id = _text(preflight.get("replay_preflight_id"))
    artifacts = preflight.get("artifacts", {})
    asset_queue_path = root / _text(artifacts.get("snapshot_asset_fetch_queue"))
    pair_queue_path = root / _text(artifacts.get("snapshot_pair_history_queue"))
    if (
        not run_id
        or not preflight_id
        or not asset_queue_path.exists()
        or not pair_queue_path.exists()
    ):
        raise ValueError("Replay preflight does not reference complete immutable queues")
    asset_queue = _read_csv(asset_queue_path)
    pair_queue = _read_csv(pair_queue_path)
    timestamp_token = as_of.strftime("%Y%m%dT%H%M%S%fZ")
    history_run_id = f"hlhistory_{timestamp_token}_{_file_hash(asset_queue_path)[:8]}"
    preflight_snapshot_dir = (root / _text(artifacts.get("snapshot_manifest"))).parent
    history_dir = preflight_snapshot_dir / "history_runs" / history_run_id
    asset_dir = history_dir / "assets"
    pair_dir = history_dir / "pairs"
    asset_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "asset_results": active / "exhaustive_wizard_hyperliquid_asset_history_results.csv",
        "pair_results": active / "exhaustive_wizard_hyperliquid_pair_history_results.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_history_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_history_summary.md",
        "snapshot_asset_results": history_dir / "asset_history_results.csv",
        "snapshot_pair_results": history_dir / "pair_history_results.csv",
        "snapshot_manifest": history_dir / "manifest.json",
        "snapshot_summary_md": history_dir / "summary.md",
    }

    asset_rows: list[dict[str, object]] = []
    for request in asset_queue.itertuples():
        asset = _text(request.asset)
        interval = _text(request.hyperliquid_interval)
        cutoff = _parse_timestamp(request.fetch_end_at)
        days = int(request.history_days_requested)
        minimum_rows = int(request.minimum_history_rows)
        output_path: Path | None = None
        blocker = ""
        attempts = 0
        last_error = ""
        if cutoff is None:
            blocker = "asset_fetch_cutoff_missing"
        else:
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    output_path = fetcher(
                        coin=asset,
                        interval=interval,
                        days=days,
                        output_dir=asset_dir,
                        end_time=cutoff,
                    )
                    break
                except Exception as exc:  # Preserve every remaining request.
                    last_error = f"{type(exc).__name__}:{exc}"
                    if attempt < max_attempts:
                        sleep(float(2 ** (attempt - 1)))
            if output_path is None:
                blocker = f"hyperliquid_candle_fetch_failed:{last_error}"
        metadata = _candle_file_metadata(output_path, cutoff=cutoff)
        if not blocker and metadata["rows"] < minimum_rows:
            blocker = "insufficient_point_in_time_history"
        if not blocker and not metadata["timestamp_parse_valid"]:
            blocker = "history_timestamp_missing_or_invalid"
        if not blocker and metadata["post_snapshot_rows"] > 0:
            blocker = "history_contains_post_snapshot_candles"
        status = "COMPLETE" if not blocker else "BLOCKED"
        asset_rows.append(
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "history_run_id": history_run_id,
                "asset_fetch_request_id": _text(request.asset_fetch_request_id),
                "asset": asset,
                "hyperliquid_interval": interval,
                "fetch_end_at": cutoff.isoformat() if cutoff else "",
                "history_days_requested": days,
                "minimum_history_rows": minimum_rows,
                "attempts": attempts,
                "history_rows": metadata["rows"],
                "earliest_candle_at": metadata["earliest"],
                "latest_candle_at": metadata["latest"],
                "timestamp_parse_valid": metadata["timestamp_parse_valid"],
                "post_snapshot_rows": metadata["post_snapshot_rows"],
                "timestamp_bound_valid": metadata["timestamp_bound_valid"],
                "history_status": status,
                "history_blocker": blocker,
                "history_path": _relative(output_path, root) if output_path else "",
                "history_sha256": _file_hash(output_path) if output_path else "",
                "queue_evidence_path": _relative(asset_queue_path, root),
                "live_trading_authorized": False,
            }
        )
    asset_results = pd.DataFrame(asset_rows, columns=ASSET_HISTORY_RESULT_COLUMNS)
    asset_lookup = {
        (_text(row.asset), _text(row.hyperliquid_interval)): row
        for row in asset_results.itertuples()
    }

    pair_rows: list[dict[str, object]] = []
    for request in pair_queue.itertuples():
        asset_x = _text(request.asset_x)
        asset_y = _text(request.asset_y)
        interval = _text(request.hyperliquid_interval)
        x_result = asset_lookup.get((asset_x, interval))
        y_result = asset_lookup.get((asset_y, interval))
        blockers = _split_blockers(getattr(request, "history_request_blocker", ""))
        pair_path: Path | None = None
        rows = 0
        earliest = ""
        latest = ""
        timestamp_parse_valid = False
        post_snapshot_rows = 0
        timestamp_bound_valid = False
        if _text(request.history_request_status) != "READY_TO_FETCH":
            blockers.append("pair_history_request_not_ready")
        for asset, result in ((asset_x, x_result), (asset_y, y_result)):
            if result is None:
                blockers.append(f"asset_history_result_missing:{asset}")
            elif _text(result.history_status) != "COMPLETE":
                blockers.extend(
                    _split_blockers(result.history_blocker) or [f"asset_history_blocked:{asset}"]
                )
        if not blockers:
            try:
                pair_path = build_hyperliquid_pair_history(
                    asset_x=asset_x,
                    asset_y=asset_y,
                    interval=interval,
                    pair_id=_text(request.pair_group_id),
                    candle_dir=asset_dir,
                    output_dir=pair_dir,
                )
                pair_metadata = _pair_history_metadata(
                    pair_path,
                    cutoff=_parse_timestamp(request.scanner_cutoff_at),
                )
                rows = pair_metadata["rows"]
                earliest = pair_metadata["earliest"]
                latest = pair_metadata["latest"]
                timestamp_parse_valid = pair_metadata["timestamp_parse_valid"]
                post_snapshot_rows = pair_metadata["post_snapshot_rows"]
                timestamp_bound_valid = pair_metadata["timestamp_bound_valid"]
                if rows < int(request.minimum_history_rows):
                    blockers.append("insufficient_aligned_pair_history")
                if not timestamp_parse_valid:
                    blockers.append("pair_history_timestamp_missing_or_invalid")
                if post_snapshot_rows > 0:
                    blockers.append("pair_history_contains_post_snapshot_candles")
            except Exception as exc:
                blockers.append(f"hyperliquid_pair_history_build_failed:{type(exc).__name__}:{exc}")
        blockers = list(dict.fromkeys(blockers))
        status = "READY_FOR_CANONICAL_REPLAY" if not blockers else "BLOCKED"
        evidence = [
            _relative(pair_queue_path, root),
            _text(getattr(x_result, "history_path", "")) if x_result else "",
            _text(getattr(y_result, "history_path", "")) if y_result else "",
            _relative(pair_path, root) if pair_path else "",
        ]
        pair_rows.append(
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "history_run_id": history_run_id,
                "history_request_id": _text(request.history_request_id),
                "pair_group_id": _text(request.pair_group_id),
                "pair": _text(request.pair),
                "wizard_exchange": _text(request.wizard_exchange),
                "wizard_timeframe": _text(request.wizard_timeframe),
                "hyperliquid_interval": interval,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "scanner_cutoff_at": _text(request.scanner_cutoff_at),
                "minimum_history_rows": int(request.minimum_history_rows),
                "history_rows": rows,
                "earliest_candle_at": earliest,
                "latest_candle_at": latest,
                "timestamp_parse_valid": timestamp_parse_valid,
                "post_snapshot_rows": post_snapshot_rows,
                "timestamp_bound_valid": timestamp_bound_valid,
                "ready_mode_orientation_cells": int(request.ready_mode_orientation_cells),
                "history_status": status,
                "history_blocker": ";".join(blockers),
                "history_path": _relative(pair_path, root) if pair_path else "",
                "history_sha256": _file_hash(pair_path) if pair_path else "",
                "evidence_path": ";".join(value for value in evidence if value),
                "canonical_replay_leverage": 1.0,
                "live_trading_authorized": False,
            }
        )
    pair_results = pd.DataFrame(pair_rows, columns=PAIR_HISTORY_RESULT_COLUMNS)
    for frame, active_path, snapshot_path in (
        (asset_results, paths["asset_results"], paths["snapshot_asset_results"]),
        (pair_results, paths["pair_results"], paths["snapshot_pair_results"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    summary: dict[str, object] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "history_run_id": history_run_id,
        "created_at": as_of.isoformat(),
        "asset_requests": int(len(asset_results)),
        "asset_histories_complete": int(asset_results["history_status"].eq("COMPLETE").sum()),
        "asset_histories_blocked": int(asset_results["history_status"].eq("BLOCKED").sum()),
        "pair_work_items": int(len(pair_results)),
        "pair_histories_ready": int(
            pair_results["history_status"].eq("READY_FOR_CANONICAL_REPLAY").sum()
        ),
        "pair_histories_blocked": int(pair_results["history_status"].eq("BLOCKED").sum()),
        "post_snapshot_asset_violations": int(
            asset_results["post_snapshot_rows"].astype(int).gt(0).sum()
        ),
        "post_snapshot_pair_violations": int(
            pair_results["post_snapshot_rows"].astype(int).gt(0).sum()
        ),
        "source_api": "hyperliquid_public_mainnet_info",
        "canonical_replay_leverage": 1.0,
        "live_trading_authorized": False,
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _history_summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def build_exhaustive_wizard_hyperliquid_replay_preflight(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Materialize complete replay and point-in-time history work ledgers."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifest_path = active / "exhaustive_wizard_hyperliquid_run_manifest.json"
    mapping_path = active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv"
    progress_path = active / "exhaustive_wizard_pair_detail_capture_progress.csv"
    mode_ledger_path = active / "exhaustive_wizard_pair_detail_mode_ledger.csv"
    mainnet_path = root / "data" / "processed" / "hyperliquid_market_context.csv"
    required = [manifest_path, mapping_path, progress_path, mode_ledger_path, mainnet_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Replay preflight inputs missing: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = _text(manifest.get("run_id"))
    immutable = manifest.get("immutable_snapshot", {})
    immutable_artifacts = immutable.get("artifacts", {}) if isinstance(immutable, dict) else {}
    experiment_path = root / _text(immutable_artifacts.get("experiment_matrix"))
    pair_path = root / _text(immutable_artifacts.get("pair_ledger"))
    if not run_id or not experiment_path.exists() or not pair_path.exists():
        raise ValueError(
            "Exhaustive run manifest does not reference the frozen experiment and pair ledgers"
        )

    experiments = _read_csv(experiment_path)
    pairs = _read_csv(pair_path)
    mapping = _read_csv(mapping_path)
    progress = _read_csv(progress_path)
    modes = _read_csv(mode_ledger_path)
    mainnet = _read_csv(mainnet_path)
    input_paths = {
        "experiment_matrix": experiment_path,
        "pair_ledger": pair_path,
        "mapping_refresh": mapping_path,
        "pair_detail_progress": progress_path,
        "mode_ledger": mode_ledger_path,
        "mainnet_market_context": mainnet_path,
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "input_hashes": {key: _file_hash(path) for key, path in input_paths.items()},
        "timeframe_intervals": TIMEFRAME_INTERVALS,
        "history_days": HISTORY_DAYS,
        "minimum_history_rows": MINIMUM_HISTORY_ROWS,
    }
    preflight_hash = sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    preflight_id = f"hlreplay_{preflight_hash[:20]}"
    run_snapshot = root / _text(immutable.get("directory"))
    snapshot_dir = run_snapshot / "replay_preflights" / preflight_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    paths = {
        "experiment_preflight": active / "exhaustive_wizard_hyperliquid_replay_preflight.csv",
        "pair_history_queue": active / "exhaustive_wizard_hyperliquid_pair_history_queue.csv",
        "asset_fetch_queue": active / "exhaustive_wizard_hyperliquid_asset_fetch_queue.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_replay_preflight_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_replay_preflight_summary.md",
        "snapshot_experiment_preflight": snapshot_dir / "experiment_preflight.csv",
        "snapshot_pair_history_queue": snapshot_dir / "pair_history_queue.csv",
        "snapshot_asset_fetch_queue": snapshot_dir / "asset_fetch_queue.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    input_snapshot_paths = {
        key: snapshot_dir / "inputs" / path.name for key, path in input_paths.items()
    }
    for key, source in input_paths.items():
        destination = input_snapshot_paths[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    experiment_preflight = _build_experiment_preflight(
        experiments,
        mapping=mapping,
        progress=progress,
        modes=modes,
        mainnet=mainnet,
        run_id=run_id,
        preflight_id=preflight_id,
        input_evidence_paths={
            key: _relative(path, root) for key, path in input_snapshot_paths.items()
        },
    )
    pair_history_queue = _build_pair_history_queue(
        pairs,
        mapping=mapping,
        progress=progress,
        experiment_preflight=experiment_preflight,
        run_id=run_id,
        preflight_id=preflight_id,
        evidence_path=_relative(paths["snapshot_experiment_preflight"], root),
    )
    asset_fetch_queue = _build_asset_fetch_queue(
        pair_history_queue,
        run_id=run_id,
        preflight_id=preflight_id,
        evidence_path=_relative(paths["snapshot_pair_history_queue"], root),
    )
    for frame, active_path, snapshot_path in (
        (
            experiment_preflight,
            paths["experiment_preflight"],
            paths["snapshot_experiment_preflight"],
        ),
        (pair_history_queue, paths["pair_history_queue"], paths["snapshot_pair_history_queue"]),
        (asset_fetch_queue, paths["asset_fetch_queue"], paths["snapshot_asset_fetch_queue"]),
    ):
        frame.to_csv(active_path, index=False)
        frame.to_csv(snapshot_path, index=False)

    status_counts = experiment_preflight["preflight_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "replay_preflight_id": preflight_id,
        "created_at": as_of.isoformat(),
        "experiments": int(len(experiment_preflight)),
        "unique_experiment_ids": int(experiment_preflight["experiment_id"].nunique()),
        "ready_for_history_experiments": int(status_counts.get("READY_FOR_HISTORY", 0)),
        "not_applicable_mode_experiments": int(status_counts.get("NOT_APPLICABLE_WIZARD_MODE", 0)),
        "pending_pair_detail_experiments": int(status_counts.get("PENDING_PAIR_DETAIL", 0)),
        "mapping_blocked_experiments": int(status_counts.get("BLOCKED_HYPERLIQUID_MAPPING", 0)),
        "mainnet_history_blocked_experiments": int(
            status_counts.get("BLOCKED_MAINNET_HISTORY_MARKET", 0)
        ),
        "pair_history_work_items": int(len(pair_history_queue)),
        "pair_history_ready_to_fetch": int(
            pair_history_queue["history_request_status"].eq("READY_TO_FETCH").sum()
        ),
        "asset_interval_fetch_requests": int(len(asset_fetch_queue)),
        "history_cutoff_policy": "earliest_scanner_capture_timestamp_per_pair_group",
        "network_request_deduplication_only": True,
        "discovery_prefilter_applied": False,
        "canonical_replay_leverage": 1.0,
        "live_trading_authorized": False,
        "input_hashes": material["input_hashes"],
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
        "input_snapshots": {
            key: _relative(path, root) for key, path in input_snapshot_paths.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _build_experiment_preflight(
    experiments: pd.DataFrame,
    *,
    mapping: pd.DataFrame,
    progress: pd.DataFrame,
    modes: pd.DataFrame,
    mainnet: pd.DataFrame,
    run_id: str,
    preflight_id: str,
    input_evidence_paths: dict[str, str],
) -> pd.DataFrame:
    mapping_rows = _unique_rows(mapping, "pair_group_id")
    progress_rows = _unique_rows(progress, "pair_group_id")
    mode_rows = {
        (_text(row.pair_group_id), _text(row.exact_mode), _text(row.orientation)): row
        for row in modes.itertuples()
    }
    mainnet_assets = {
        _text(row.asset).upper()
        for row in mainnet.itertuples()
        if _truthy(getattr(row, "tradable", False))
    }
    rows: list[dict[str, object]] = []
    for experiment in experiments.itertuples():
        pair_group_id = _text(experiment.pair_group_id)
        mapping_row = mapping_rows.get(pair_group_id)
        progress_row = progress_rows.get(pair_group_id)
        mode_row = mode_rows.get(
            (pair_group_id, _text(experiment.exact_mode), _text(experiment.orientation))
        )
        asset_x = _text(experiment.asset_x).upper()
        asset_y = _text(experiment.asset_y).upper()
        mapping_ready = bool(
            mapping_row is not None and _truthy(getattr(mapping_row, "current_pair_ready", False))
        )
        mainnet_missing = sorted(
            {asset for asset in (asset_x, asset_y) if asset not in mainnet_assets}
        )
        interval = TIMEFRAME_INTERVALS.get(_text(experiment.timeframe).lower(), "")
        cutoff_at = _earliest_timestamp(
            getattr(progress_row, "scanner_capture_timestamps", "") if progress_row else ""
        )
        mode_status = _text(getattr(mode_row, "capture_status", "")) if mode_row else ""
        orientation_verified = bool(
            mode_row is not None and _truthy(getattr(mode_row, "orientation_verified", False))
        )
        blockers: list[str] = []
        if mapping_row is None:
            blockers.append("mapping_refresh_row_missing")
        elif not mapping_ready:
            blockers.extend(
                _split_blockers(getattr(mapping_row, "current_mapping_blocker", ""))
                or ["hyperliquid_pair_not_ready"]
            )
        if mainnet_missing:
            blockers.extend(f"mainnet_market_data_unavailable:{asset}" for asset in mainnet_missing)
        if not interval:
            blockers.append("unsupported_wizard_timeframe")
        if not cutoff_at:
            blockers.append("scanner_capture_cutoff_missing")
        if mode_row is None:
            blockers.append("pair_detail_mode_cell_missing")
        elif mode_status == "NOT_AVAILABLE_ON_PAIR_PAGE":
            blockers.append("wizard_mode_not_available_on_pair_page")
        elif mode_status != "CAPTURED":
            blockers.extend(
                _split_blockers(getattr(mode_row, "capture_blocker", ""))
                or ["pair_detail_mode_not_captured"]
            )
        elif not orientation_verified:
            blockers.append("pair_detail_orientation_not_verified")

        if not mapping_ready:
            status = "BLOCKED_HYPERLIQUID_MAPPING"
        elif mainnet_missing:
            status = "BLOCKED_MAINNET_HISTORY_MARKET"
        elif mode_status == "NOT_AVAILABLE_ON_PAIR_PAGE":
            status = "NOT_APPLICABLE_WIZARD_MODE"
        elif mode_status == "CAPTURED" and orientation_verified and interval and cutoff_at:
            status = "READY_FOR_HISTORY"
        else:
            status = "PENDING_PAIR_DETAIL"
        history_request_id = ""
        if interval:
            history_request_id = (
                "hlhist_"
                + sha256(
                    f"{run_id}|{pair_group_id}|{interval}|{cutoff_at}".encode("utf-8")
                ).hexdigest()[:20]
            )
        evidence = [
            input_evidence_paths["experiment_matrix"],
            input_evidence_paths["mapping_refresh"],
            input_evidence_paths["pair_detail_progress"],
            input_evidence_paths["mainnet_market_context"],
        ]
        mode_evidence = _text(getattr(mode_row, "evidence_path", "")) if mode_row else ""
        if mode_evidence:
            evidence.append(mode_evidence)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "experiment_id": _text(experiment.experiment_id),
                "pair_group_id": pair_group_id,
                "pair": _text(experiment.pair),
                "wizard_exchange": _text(experiment.wizard_exchange),
                "wizard_timeframe": _text(experiment.timeframe),
                "hyperliquid_interval": interval,
                "exact_mode": _text(experiment.exact_mode),
                "orientation": _text(experiment.orientation),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "mapping_ready": mapping_ready,
                "mainnet_history_market_ready": not mainnet_missing,
                "pair_detail_mode_status": mode_status or "MISSING",
                "orientation_verified": orientation_verified,
                "scanner_cutoff_at": cutoff_at,
                "history_request_id": history_request_id,
                "preflight_status": status,
                "preflight_blocker": ";".join(dict.fromkeys(blockers)),
                "mode_evidence_path": mode_evidence,
                "evidence_path": ";".join(dict.fromkeys(evidence)),
                "discovery_prefilter_applied": False,
                "canonical_replay_leverage": 1.0,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(experiments) or frame["experiment_id"].nunique() != len(experiments):
        raise ValueError("Replay preflight failed complete experiment accounting")
    return frame


def _build_pair_history_queue(
    pairs: pd.DataFrame,
    *,
    mapping: pd.DataFrame,
    progress: pd.DataFrame,
    experiment_preflight: pd.DataFrame,
    run_id: str,
    preflight_id: str,
    evidence_path: str,
) -> pd.DataFrame:
    mapping_rows = _unique_rows(mapping, "pair_group_id")
    progress_rows = _unique_rows(progress, "pair_group_id")
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples():
        pair_group_id = _text(pair.pair_group_id)
        mapping_row = mapping_rows.get(pair_group_id)
        progress_row = progress_rows.get(pair_group_id)
        cells = experiment_preflight.loc[experiment_preflight["pair_group_id"].eq(pair_group_id)]
        ready_cells = int(cells["preflight_status"].eq("READY_FOR_HISTORY").sum())
        interval = TIMEFRAME_INTERVALS.get(_text(pair.timeframe).lower(), "")
        cutoff_at = _earliest_timestamp(
            getattr(progress_row, "scanner_capture_timestamps", "") if progress_row else ""
        )
        request_id = _text(cells["history_request_id"].iloc[0]) if not cells.empty else ""
        blockers = sorted(
            {
                blocker
                for value in cells["preflight_blocker"].tolist()
                for blocker in _split_blockers(value)
                if blocker != "wizard_mode_not_available_on_pair_page"
            }
        )
        status = "READY_TO_FETCH" if ready_cells > 0 else "BLOCKED"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "history_request_id": request_id,
                "pair_group_id": pair_group_id,
                "pair_group_key": _text(getattr(pair, "pair_group_key", "")),
                "pair": _text(pair.pair),
                "wizard_exchange": _text(pair.wizard_exchange),
                "wizard_timeframe": _text(pair.timeframe),
                "hyperliquid_interval": interval,
                "asset_x": _text(pair.asset_x).upper(),
                "asset_y": _text(pair.asset_y).upper(),
                "scanner_cutoff_at": cutoff_at,
                "history_days_requested": HISTORY_DAYS.get(interval, 0),
                "minimum_history_rows": MINIMUM_HISTORY_ROWS.get(interval, 0),
                "ready_mode_orientation_cells": ready_cells,
                "planned_mode_orientation_cells": int(len(cells)),
                "pair_detail_capture_status": _text(
                    getattr(progress_row, "capture_status", "MISSING")
                    if progress_row
                    else "MISSING"
                ),
                "hyperliquid_mapping_ready": bool(
                    mapping_row is not None
                    and _truthy(getattr(mapping_row, "current_pair_ready", False))
                ),
                "history_request_status": status,
                "history_request_blocker": ";".join(blockers),
                "evidence_path": evidence_path,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _build_asset_fetch_queue(
    pair_history_queue: pd.DataFrame,
    *,
    run_id: str,
    preflight_id: str,
    evidence_path: str,
) -> pd.DataFrame:
    requests: list[dict[str, str]] = []
    ready = pair_history_queue.loc[
        pair_history_queue["history_request_status"].eq("READY_TO_FETCH")
    ]
    for row in ready.itertuples():
        for asset in (_text(row.asset_x), _text(row.asset_y)):
            requests.append(
                {
                    "asset": asset,
                    "interval": _text(row.hyperliquid_interval),
                    "cutoff_at": _text(row.scanner_cutoff_at),
                    "pair_group_id": _text(row.pair_group_id),
                    "history_request_id": _text(row.history_request_id),
                }
            )
    if not requests:
        return pd.DataFrame(
            columns=[
                "schema_version",
                "exhaustive_run_id",
                "replay_preflight_id",
                "asset_fetch_request_id",
                "asset",
                "hyperliquid_interval",
                "fetch_end_at",
                "fetch_start_at",
                "history_days_requested",
                "minimum_history_rows",
                "pair_group_count",
                "pair_group_ids",
                "history_request_ids",
                "fetch_status",
                "evidence_path",
                "live_trading_authorized",
            ]
        )
    frame = pd.DataFrame(requests)
    rows: list[dict[str, object]] = []
    for (asset, interval), group in frame.groupby(["asset", "interval"], sort=True):
        timestamps = [
            timestamp
            for value in group["cutoff_at"].tolist()
            if (timestamp := _parse_timestamp(value)) is not None
        ]
        cutoff = min(timestamps) if timestamps else None
        days = HISTORY_DAYS.get(interval, 0)
        request_material = f"{run_id}|{asset}|{interval}|{cutoff.isoformat() if cutoff else ''}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "replay_preflight_id": preflight_id,
                "asset_fetch_request_id": "hlfetch_"
                + sha256(request_material.encode("utf-8")).hexdigest()[:20],
                "asset": asset,
                "hyperliquid_interval": interval,
                "fetch_end_at": cutoff.isoformat() if cutoff else "",
                "fetch_start_at": (cutoff - timedelta(days=days)).isoformat()
                if cutoff and days
                else "",
                "history_days_requested": days,
                "minimum_history_rows": MINIMUM_HISTORY_ROWS.get(interval, 0),
                "pair_group_count": int(group["pair_group_id"].nunique()),
                "pair_group_ids": ";".join(sorted(set(group["pair_group_id"]))),
                "history_request_ids": ";".join(sorted(set(group["history_request_id"]))),
                "fetch_status": "NOT_FETCHED",
                "evidence_path": evidence_path,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _unique_rows(frame: pd.DataFrame, key: str) -> dict[str, object]:
    if frame.empty or key not in frame.columns:
        return {}
    if frame[key].astype(str).duplicated().any():
        raise ValueError(f"Expected unique {key} rows")
    return {_text(getattr(row, key)): row for row in frame.itertuples()}


def _earliest_timestamp(value: object) -> str:
    timestamps = [
        timestamp
        for item in _text(value).split(";")
        if (timestamp := _parse_timestamp(item)) is not None
    ]
    return min(timestamps).isoformat() if timestamps else ""


def _parse_timestamp(value: object) -> datetime | None:
    timestamp = pd.to_datetime(_text(value), utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _split_blockers(value: object) -> list[str]:
    return [item.strip() for item in _text(value).split(";") if item.strip()]


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Replay Preflight",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Preflight: `{summary['replay_preflight_id']}`",
            f"- Experiments accounted: {summary['unique_experiment_ids']} / {summary['experiments']}",
            f"- Ready for point-in-time history: {summary['ready_for_history_experiments']}",
            f"- Wizard modes not available: {summary['not_applicable_mode_experiments']}",
            f"- Pending pair detail: {summary['pending_pair_detail_experiments']}",
            f"- Hyperliquid mapping blocked: {summary['mapping_blocked_experiments']}",
            f"- Mainnet history market blocked: {summary['mainnet_history_blocked_experiments']}",
            f"- Pair-history requests ready: {summary['pair_history_ready_to_fetch']} / {summary['pair_history_work_items']}",
            f"- Deduplicated asset/interval fetches: {summary['asset_interval_fetch_requests']}",
            f"- Canonical replay leverage: {summary['canonical_replay_leverage']}x",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Network fetches are deduplicated for efficiency. Pair groups, modes, orientations, blockers, and evidence lineage are never deduplicated or filtered.",
            "",
        ]
    )


def _candle_file_metadata(path: Path | None, *, cutoff: datetime | None) -> dict[str, object]:
    return _json_series_metadata(
        path, series_key="candles", timestamp_key="startedAt", cutoff=cutoff
    )


def _pair_history_metadata(path: Path | None, *, cutoff: datetime | None) -> dict[str, object]:
    return _json_series_metadata(
        path, series_key="history", timestamp_key="timestamp", cutoff=cutoff
    )


def _json_series_metadata(
    path: Path | None,
    *,
    series_key: str,
    timestamp_key: str,
    cutoff: datetime | None,
) -> dict[str, object]:
    empty = {
        "rows": 0,
        "earliest": "",
        "latest": "",
        "timestamp_parse_valid": False,
        "post_snapshot_rows": 0,
        "timestamp_bound_valid": False,
    }
    if path is None or not path.exists():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return empty
    series = payload.get(series_key, []) if isinstance(payload, dict) else []
    if not isinstance(series, list) or not series:
        return empty
    timestamps = [
        timestamp
        for row in series
        if isinstance(row, dict)
        and (timestamp := _parse_timestamp(row.get(timestamp_key))) is not None
    ]
    parse_valid = cutoff is not None and len(timestamps) == len(series)
    post_snapshot_rows = (
        sum(timestamp > cutoff for timestamp in timestamps) if cutoff is not None else 0
    )
    return {
        "rows": len(series),
        "earliest": min(timestamps).isoformat() if timestamps else "",
        "latest": max(timestamps).isoformat() if timestamps else "",
        "timestamp_parse_valid": parse_valid,
        "post_snapshot_rows": post_snapshot_rows,
        "timestamp_bound_valid": bool(parse_valid and post_snapshot_rows == 0),
    }


def _history_summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Point-In-Time History",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Replay preflight: `{summary['replay_preflight_id']}`",
            f"- History run: `{summary['history_run_id']}`",
            f"- Asset histories complete: {summary['asset_histories_complete']} / {summary['asset_requests']}",
            f"- Asset histories blocked: {summary['asset_histories_blocked']}",
            f"- Pair histories ready for canonical replay: {summary['pair_histories_ready']} / {summary['pair_work_items']}",
            f"- Pair histories blocked: {summary['pair_histories_blocked']}",
            f"- Asset histories with post-snapshot rows: {summary['post_snapshot_asset_violations']}",
            f"- Pair histories with post-snapshot rows: {summary['post_snapshot_pair_violations']}",
            f"- Canonical replay leverage: {summary['canonical_replay_leverage']}x",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Every history is bounded by the frozen scanner timestamp. Failed, short, malformed, and mapping-blocked work remains visible with an explicit blocker; no research identity is silently dropped.",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "ready"}


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
