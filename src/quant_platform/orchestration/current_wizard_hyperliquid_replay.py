"""Bounded history and canonical 1x replays for the current Wizard board.

The current scanner refresh can contain hundreds of pair groups.  This module
requires an explicit selection for network and disk work while still emitting
one status row for every pair and every planned mode/orientation cell.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.backtest import CostModel, backtest_two_leg_spread_with_ledger
from quant_platform.economic_contract import (
    ECONOMIC_CONTRACT_VERSION,
    rolling_y_on_x_beta,
    tail_actions,
    y_on_x_log_spread,
)
from quant_platform.hyperliquid import (
    build_hyperliquid_pair_history,
    fetch_hyperliquid_candles,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import atomic_write_csv, atomic_write_text
from quant_platform.statistics.math_v2 import fit_engle_granger, fit_ou
from quant_platform.wizard_mode_replay import build_local_mode_signal

ROOT = Path(__file__).resolve().parents[3]
HISTORY_SCHEMA_VERSION = "current_wizard_hyperliquid_history.v1"
REPLAY_SCHEMA_VERSION = "current_wizard_hyperliquid_canonical_replay.v2"
LOCAL_SETTINGS_VERSION = "current_local_standardized_math_v2.v4_y_on_x"
MATH_IMPLEMENTATION_PATHS = (
    "src/quant_platform/economic_contract.py",
    "src/quant_platform/performance_math.py",
    "src/quant_platform/backtest.py",
    "src/quant_platform/trade_ledger.py",
    "src/quant_platform/wizard_mode_replay.py",
    "src/quant_platform/statistics/math_v2.py",
    "src/quant_platform/orchestration/current_wizard_hyperliquid_replay.py",
)
DEFAULT_MINIMUM_FREE_DISK_BYTES = 512 * 1024 * 1024
ESTIMATED_ASSET_BYTES = 8 * 1024 * 1024
ESTIMATED_PAIR_BYTES = 4 * 1024 * 1024
MINIMUM_RESEARCH_RANK_TRADES = 10
PROVISIONAL_ACCEPTANCE_BLOCKER = (
    "provisional_costs;observed_funding_not_attached;"
    "observed_slippage_not_attached;walk_forward_not_run;"
    "regime_robustness_not_run;local_formula_approximation"
)


@dataclass(frozen=True)
class CurrentReplayPolicy:
    """Causal local settings used consistently across current-board cells."""

    train_fraction: float = 0.70
    minimum_train_rows: int = 320
    minimum_test_rows: int = 120
    zscore_window: int = 60
    dynamic_window: int = 90
    copula_window: int = 120
    entry_zscore: float = 2.0
    exit_zscore: float = 0.0
    copula_entry_lower: float = 0.10
    copula_entry_upper: float = 0.90
    copula_exit_lower: float = 0.45
    copula_exit_upper: float = 0.55


def materialize_current_wizard_hyperliquid_history(
    *,
    pair_group_keys: Iterable[str],
    root: Path = ROOT,
    now: datetime | None = None,
    max_attempts: int = 3,
    minimum_free_disk_bytes: int = DEFAULT_MINIMUM_FREE_DISK_BYTES,
    available_disk_bytes: int | None = None,
    fetcher: Callable[..., Path] = fetch_hyperliquid_candles,
    sleep: Callable[[float], None] = time.sleep,
) -> CommandResult:
    """Fetch only explicitly selected pairs and account for the full board."""

    selected = tuple(dict.fromkeys(_text(value) for value in pair_group_keys if _text(value)))
    if not selected:
        raise ValueError("pair_group_keys must explicitly select at least one pair")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if minimum_free_disk_bytes < 0:
        raise ValueError("minimum_free_disk_bytes cannot be negative")

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    handoff_manifest_path = active / "current_wizard_hyperliquid_handoff_manifest.json"
    if not handoff_manifest_path.exists():
        raise FileNotFoundError("Current Wizard Hyperliquid handoff manifest is missing")
    handoff = _read_json(handoff_manifest_path)
    handoff_id = _text(handoff.get("handoff_id"))
    refresh_id = _text(handoff.get("refresh_id"))
    artifacts = handoff.get("artifacts", {})
    pair_queue_path = root / _text(artifacts.get("snapshot_pair_history_queue"))
    asset_queue_path = root / _text(artifacts.get("snapshot_asset_fetch_queue"))
    handoff_snapshot_manifest = root / _text(artifacts.get("snapshot_manifest"))
    required = (pair_queue_path, asset_queue_path, handoff_snapshot_manifest)
    missing = [str(path) for path in required if not path.exists()]
    if not handoff_id or not refresh_id or missing:
        raise ValueError(f"Current handoff does not reference complete immutable queues: {missing}")

    pair_queue = _read_csv(pair_queue_path)
    asset_queue = _read_csv(asset_queue_path)
    known = set(pair_queue["pair_group_key"].astype(str))
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"Unknown current pair_group_keys: {unknown}")
    ready_selected = set(
        pair_queue.loc[
            pair_queue["pair_group_key"].astype(str).isin(selected)
            & pair_queue["history_request_status"].astype(str).eq("READY_TO_FETCH"),
            "pair_group_key",
        ].astype(str)
    )
    selected_asset_mask = asset_queue["pair_group_keys"].map(
        lambda value: bool(ready_selected.intersection(_split_values(value)))
    )
    selected_asset_count = int(selected_asset_mask.sum())
    estimated_bytes = (
        selected_asset_count * ESTIMATED_ASSET_BYTES + len(ready_selected) * ESTIMATED_PAIR_BYTES
    )
    free_bytes = (
        int(available_disk_bytes)
        if available_disk_bytes is not None
        else int(shutil.disk_usage(root).free)
    )
    storage_preflight_passed = free_bytes - estimated_bytes >= minimum_free_disk_bytes

    material = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "handoff_id": handoff_id,
        "refresh_id": refresh_id,
        "selected_pair_group_keys": selected,
        "pair_queue_sha256": _file_hash(pair_queue_path),
        "asset_queue_sha256": _file_hash(asset_queue_path),
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "estimated_materialization_bytes": estimated_bytes,
    }
    token = as_of.strftime("%Y%m%dT%H%M%S%fZ")
    history_run_id = (
        "cwhistoryrun_"
        + token
        + "_"
        + sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:8]
    )
    history_dir = handoff_snapshot_manifest.parent / "history_runs" / history_run_id
    asset_dir = history_dir / "assets"
    pair_dir = history_dir / "pairs"
    asset_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "asset_results": active / "current_wizard_hyperliquid_asset_history_results.csv",
        "pair_results": active / "current_wizard_hyperliquid_pair_history_results.csv",
        "validation": active / "current_wizard_hyperliquid_history_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_history_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_history_summary.md",
        "snapshot_asset_results": history_dir / "asset_history_results.csv",
        "snapshot_pair_results": history_dir / "pair_history_results.csv",
        "snapshot_validation": history_dir / "validation.csv",
        "snapshot_manifest": history_dir / "manifest.json",
        "snapshot_summary_md": history_dir / "summary.md",
    }

    asset_rows: list[dict[str, object]] = []
    for request in asset_queue.itertuples():
        asset = _text(request.asset)
        interval = _text(request.hyperliquid_interval)
        request_pairs = _split_values(request.pair_group_keys)
        request_selected = bool(ready_selected.intersection(request_pairs))
        cutoff = _parse_timestamp(request.fetch_end_at)
        output_path: Path | None = None
        attempts = 0
        blocker = ""
        if not request_selected:
            status = "DEFERRED_NOT_SELECTED"
            blocker = "bounded_run_pair_not_selected"
        elif not storage_preflight_passed:
            status = "BLOCKED_STORAGE_PREFLIGHT"
            blocker = "minimum_free_disk_reserve_would_be_breached"
        elif cutoff is None:
            status = "BLOCKED"
            blocker = "asset_fetch_cutoff_missing"
        else:
            last_error = ""
            for attempt in range(1, max_attempts + 1):
                attempts = attempt
                try:
                    output_path = fetcher(
                        coin=asset,
                        interval=interval,
                        days=int(request.history_days_requested),
                        output_dir=asset_dir,
                        end_time=cutoff,
                    )
                    break
                except Exception as exc:  # Every failed request remains in the ledger.
                    last_error = f"{safe_exception_code(exc)}"
                    if attempt < max_attempts:
                        sleep(float(2 ** (attempt - 1)))
            metadata = _candle_metadata(output_path, cutoff=cutoff)
            blockers: list[str] = []
            if output_path is None:
                blockers.append(f"hyperliquid_candle_fetch_failed:{last_error}")
            if metadata["rows"] < int(request.minimum_history_rows):
                blockers.append("insufficient_point_in_time_history")
            if not metadata["timestamp_parse_valid"]:
                blockers.append("history_timestamp_missing_or_invalid")
            if int(metadata["post_cutoff_rows"]) > 0:
                blockers.append("history_contains_post_cutoff_candles")
            blocker = ";".join(dict.fromkeys(blockers))
            status = "COMPLETE" if not blocker else "BLOCKED"
        metadata = _candle_metadata(output_path, cutoff=cutoff)
        asset_rows.append(
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "handoff_id": handoff_id,
                "refresh_id": refresh_id,
                "history_run_id": history_run_id,
                "asset_fetch_request_id": _text(request.asset_fetch_request_id),
                "asset": asset,
                "hyperliquid_interval": interval,
                "fetch_end_at": cutoff.isoformat() if cutoff else "",
                "selected_for_materialization": request_selected,
                "selected_pair_group_count": len(ready_selected.intersection(request_pairs)),
                "history_days_requested": int(request.history_days_requested),
                "minimum_history_rows": int(request.minimum_history_rows),
                "attempts": attempts,
                "history_rows": metadata["rows"],
                "earliest_candle_at": metadata["earliest"],
                "latest_candle_at": metadata["latest"],
                "timestamp_parse_valid": metadata["timestamp_parse_valid"],
                "post_cutoff_rows": metadata["post_cutoff_rows"],
                "timestamp_bound_valid": metadata["timestamp_bound_valid"],
                "history_status": status,
                "history_blocker": blocker,
                "history_path": _relative(output_path, root) if output_path else "",
                "history_sha256": _file_hash(output_path) if output_path else "",
                "queue_evidence_path": _relative(asset_queue_path, root),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    asset_results = pd.DataFrame(asset_rows)
    asset_lookup = {
        (_text(row.asset), _text(row.hyperliquid_interval)): row
        for row in asset_results.itertuples()
    }

    pair_rows: list[dict[str, object]] = []
    for request in pair_queue.itertuples():
        pair_key = _text(request.pair_group_key)
        asset_x = _text(request.asset_a)
        asset_y = _text(request.asset_b)
        interval = _text(request.hyperliquid_interval)
        selected_for_run = pair_key in selected
        ready_in_handoff = _text(request.history_request_status) == "READY_TO_FETCH"
        cutoff = _parse_timestamp(request.scanner_cutoff_at)
        pair_path: Path | None = None
        blockers: list[str] = []
        if not ready_in_handoff:
            status = "BLOCKED_HANDOFF_REQUEST"
            blockers.extend(_split_values(request.history_request_blocker))
            blockers.append("pair_history_request_not_ready")
        elif not selected_for_run:
            status = "DEFERRED_NOT_SELECTED"
            blockers.append("bounded_run_pair_not_selected")
        elif not storage_preflight_passed:
            status = "BLOCKED_STORAGE_PREFLIGHT"
            blockers.append("minimum_free_disk_reserve_would_be_breached")
        else:
            x_result = asset_lookup.get((asset_x, interval))
            y_result = asset_lookup.get((asset_y, interval))
            for asset, result in ((asset_x, x_result), (asset_y, y_result)):
                if result is None:
                    blockers.append(f"asset_history_result_missing:{asset}")
                elif _text(result.history_status) != "COMPLETE":
                    blockers.extend(
                        _split_values(result.history_blocker) or [f"asset_history_blocked:{asset}"]
                    )
            if not blockers:
                try:
                    pair_path = build_hyperliquid_pair_history(
                        asset_x=asset_x,
                        asset_y=asset_y,
                        interval=interval,
                        pair_id=pair_key,
                        candle_dir=asset_dir,
                        output_dir=pair_dir,
                    )
                    _trim_pair_history_to_cutoff(pair_path, cutoff=cutoff)
                except Exception as exc:
                    blockers.append(
                        f"hyperliquid_pair_history_build_failed:{safe_exception_code(exc)}"
                    )
            metadata = _pair_metadata(pair_path, cutoff=cutoff)
            if metadata["rows"] < int(request.minimum_history_rows):
                blockers.append("insufficient_aligned_pair_history")
            if not metadata["timestamp_parse_valid"]:
                blockers.append("pair_history_timestamp_missing_or_invalid")
            if int(metadata["post_cutoff_rows"]) > 0:
                blockers.append("pair_history_contains_post_cutoff_candles")
            status = "READY_FOR_CANONICAL_1X_REPLAY" if not blockers else "BLOCKED"
        metadata = _pair_metadata(pair_path, cutoff=cutoff)
        blockers = list(dict.fromkeys(value for value in blockers if value))
        evidence = [
            _relative(pair_queue_path, root),
            _relative(pair_path, root) if pair_path else "",
        ]
        pair_rows.append(
            {
                "schema_version": HISTORY_SCHEMA_VERSION,
                "handoff_id": handoff_id,
                "refresh_id": refresh_id,
                "history_run_id": history_run_id,
                "history_request_id": _text(request.history_request_id),
                "pair_group_key": pair_key,
                "pair": _text(request.pair),
                "wizard_exchange": _text(request.wizard_exchange),
                "timeframe": _text(request.timeframe),
                "hyperliquid_interval": interval,
                "asset_a": asset_x,
                "asset_b": asset_y,
                "hyperliquid_market_a": _text(request.hyperliquid_market_a),
                "hyperliquid_market_b": _text(request.hyperliquid_market_b),
                "scanner_cutoff_at": cutoff.isoformat() if cutoff else "",
                "selected_for_materialization": selected_for_run,
                "minimum_history_rows": int(request.minimum_history_rows),
                "history_rows": metadata["rows"],
                "earliest_candle_at": metadata["earliest"],
                "latest_candle_at": metadata["latest"],
                "timestamp_parse_valid": metadata["timestamp_parse_valid"],
                "post_cutoff_rows": metadata["post_cutoff_rows"],
                "timestamp_bound_valid": metadata["timestamp_bound_valid"],
                "ready_mode_orientation_experiments": int(
                    request.ready_mode_orientation_experiments
                ),
                "history_status": status,
                "history_blocker": ";".join(blockers),
                "history_path": _relative(pair_path, root) if pair_path else "",
                "history_sha256": _file_hash(pair_path) if pair_path else "",
                "evidence_path": ";".join(value for value in evidence if value),
                "canonical_replay_leverage": 1.0,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    pair_results = pd.DataFrame(pair_rows)
    validation = _history_validation(
        pair_queue=pair_queue,
        pair_results=pair_results,
        asset_results=asset_results,
        selected=selected,
        storage_preflight_passed=storage_preflight_passed,
    )
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current history validation failed: " + ",".join(failed))

    for frame, active_key, snapshot_key in (
        (asset_results, "asset_results", "snapshot_asset_results"),
        (pair_results, "pair_results", "snapshot_pair_results"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)

    pair_counts = pair_results["history_status"].value_counts().to_dict()
    asset_counts = asset_results["history_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "handoff_id": handoff_id,
        "refresh_id": refresh_id,
        "history_run_id": history_run_id,
        "created_at": as_of.isoformat(),
        "selected_pair_group_keys": list(selected),
        "selected_pair_groups": len(selected),
        "selected_ready_pair_groups": len(ready_selected),
        "pair_groups_accounted": int(len(pair_results)),
        "asset_requests_accounted": int(len(asset_results)),
        "pair_status_counts": {str(key): int(value) for key, value in pair_counts.items()},
        "asset_status_counts": {str(key): int(value) for key, value in asset_counts.items()},
        "pair_histories_ready": int(pair_counts.get("READY_FOR_CANONICAL_1X_REPLAY", 0)),
        "storage_preflight_passed": storage_preflight_passed,
        "available_disk_bytes": free_bytes,
        "estimated_materialization_bytes": estimated_bytes,
        "minimum_free_disk_bytes_after_run": minimum_free_disk_bytes,
        "network_scope_explicit_and_bounded": True,
        "source_api": "hyperliquid_public_mainnet_info",
        "canonical_replay_leverage": 1.0,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "input_hashes": {
            "handoff_manifest": _file_hash(handoff_snapshot_manifest),
            "pair_history_queue": _file_hash(pair_queue_path),
            "asset_fetch_queue": _file_hash(asset_queue_path),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _history_summary(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def run_current_wizard_hyperliquid_canonical_replay(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    policy: CurrentReplayPolicy | None = None,
    cost_model: CostModel | None = None,
    handoff_manifest_path: Path | None = None,
    history_manifest_path: Path | None = None,
    output_dir: Path | None = None,
    legacy_contract_math_audit: bool = False,
) -> CommandResult:
    """Attempt every current cell and run selected ready cells at normalized 1x."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    policy = policy or CurrentReplayPolicy()
    _validate_replay_policy(policy)
    costs = cost_model or CostModel()
    active = output_dir or root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    if legacy_contract_math_audit and (
        handoff_manifest_path is None or history_manifest_path is None or output_dir is None
    ):
        raise ValueError(
            "Legacy-contract math audit requires explicit handoff/history manifests "
            "and a separate output_dir"
        )
    handoff_manifest_path = handoff_manifest_path or (
        root / "reports" / "active" / "current_wizard_hyperliquid_handoff_manifest.json"
    )
    history_manifest_path = history_manifest_path or (
        root / "reports" / "active" / "current_wizard_hyperliquid_history_manifest.json"
    )
    if not handoff_manifest_path.exists() or not history_manifest_path.exists():
        raise FileNotFoundError("Current handoff and bounded history manifests are required")
    handoff = _read_json(handoff_manifest_path)
    history_manifest = _read_json(history_manifest_path)
    source_contract_version = _text(handoff.get("economic_contract_version"))
    if source_contract_version != ECONOMIC_CONTRACT_VERSION and not legacy_contract_math_audit:
        raise ValueError(
            "Current handoff predates the canonical seven-mode economic contract; "
            "rebuild the handoff before replay"
        )
    input_contract_status = (
        "CURRENT_CONTRACT"
        if source_contract_version == ECONOMIC_CONTRACT_VERSION
        else "LEGACY_FROZEN_INPUT_MATH_REEVALUATION_ONLY"
    )
    handoff_id = _text(handoff.get("handoff_id"))
    refresh_id = _text(handoff.get("refresh_id"))
    history_run_id = _text(history_manifest.get("history_run_id"))
    if handoff_id != _text(history_manifest.get("handoff_id")) or refresh_id != _text(
        history_manifest.get("refresh_id")
    ):
        raise ValueError("Current history and handoff identities do not match")
    experiment_path = root / _text(handoff.get("artifacts", {}).get("snapshot_experiments"))
    pair_history_path = root / _text(
        history_manifest.get("artifacts", {}).get("snapshot_pair_results")
    )
    history_snapshot_manifest = root / _text(
        history_manifest.get("artifacts", {}).get("snapshot_manifest")
    )
    missing = [
        str(path)
        for path in (experiment_path, pair_history_path, history_snapshot_manifest)
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Current canonical replay inputs missing: {missing}")

    experiments = _read_csv(experiment_path)
    pair_histories = _read_csv(pair_history_path)
    pair_lookup = {_text(row.pair_group_key): row for row in pair_histories.itertuples()}
    cost_payload = asdict(costs)
    policy_payload = asdict(policy)
    math_implementation_hashes = _math_implementation_hashes()
    material = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "handoff_id": handoff_id,
        "history_run_id": history_run_id,
        "experiment_sha256": _file_hash(experiment_path),
        "pair_history_sha256": _file_hash(pair_history_path),
        "cost_model": cost_payload,
        "replay_policy": policy_payload,
        "settings_version": LOCAL_SETTINGS_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "source_handoff_economic_contract_version": source_contract_version,
        "input_contract_status": input_contract_status,
        "historical_math_reevaluation": legacy_contract_math_audit,
        "math_implementation_hashes": math_implementation_hashes,
    }
    token = as_of.strftime("%Y%m%dT%H%M%S%fZ")
    replay_id = (
        "cwcanonical_"
        + token
        + "_"
        + sha256(_canonical_json(material).encode("utf-8")).hexdigest()[:8]
    )
    snapshot_collection = (
        "math_reevaluations" if legacy_contract_math_audit else "canonical_replays"
    )
    snapshot_dir = history_snapshot_manifest.parent / snapshot_collection / replay_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results": active / "current_wizard_hyperliquid_canonical_replay.csv",
        "ranked": active / "current_wizard_hyperliquid_canonical_replay_ranked.csv",
        "trades": active / "current_wizard_hyperliquid_canonical_replay_trades.csv",
        "pair_status": active / "current_wizard_hyperliquid_replay_pair_status.csv",
        "validation": active / "current_wizard_hyperliquid_replay_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_canonical_replay_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_canonical_replay_summary.md",
        "snapshot_results": snapshot_dir / "canonical_replay.csv",
        "snapshot_ranked": snapshot_dir / "canonical_replay_ranked.csv",
        "snapshot_trades": snapshot_dir / "canonical_replay_trades.csv",
        "snapshot_pair_status": snapshot_dir / "pair_status.csv",
        "snapshot_validation": snapshot_dir / "validation.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }

    history_cache: dict[str, pd.DataFrame] = {}
    fit_cache: dict[tuple[str, str], dict[str, Any]] = {}
    result_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []
    for experiment in experiments.itertuples():
        pair_key = _text(experiment.pair_group_key)
        exact_mode = _text(experiment.exact_mode)
        orientation = _text(experiment.orientation)
        pair_row = pair_lookup.get(pair_key)
        base = _replay_base_row(
            experiment,
            replay_id=replay_id,
            history_run_id=history_run_id,
            cost_payload=cost_payload,
            evidence_paths=(experiment_path, pair_history_path),
            root=root,
        )
        experiment_status = _text(experiment.experiment_status)
        if experiment_status != "READY_FOR_POINT_IN_TIME_HISTORY":
            result_rows.append(
                {
                    **base,
                    "replay_status": experiment_status,
                    "replay_blocker": _text(experiment.experiment_blocker),
                }
            )
            continue
        if pair_row is None:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_POINT_IN_TIME_HISTORY",
                    "replay_blocker": "pair_history_result_missing",
                }
            )
            continue
        history_status = _text(pair_row.history_status)
        if history_status == "DEFERRED_NOT_SELECTED":
            result_rows.append(
                {
                    **base,
                    "replay_status": "DEFERRED_POINT_IN_TIME_HISTORY",
                    "replay_blocker": "bounded_run_pair_not_selected",
                }
            )
            continue
        if history_status != "READY_FOR_CANONICAL_1X_REPLAY":
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_POINT_IN_TIME_HISTORY",
                    "replay_blocker": _text(pair_row.history_blocker) or "pair_history_not_ready",
                }
            )
            continue

        history_path = root / _text(pair_row.history_path)
        try:
            cache_key = str(history_path)
            if cache_key not in history_cache:
                history_cache[cache_key] = _load_history(history_path)
            oriented = _orient_history(history_cache[cache_key], orientation=orientation)
            fit_key = (pair_key, orientation)
            if fit_key not in fit_cache:
                fit_cache[fit_key] = _fit_orientation_context(oriented, policy=policy)
            fit = fit_cache[fit_key]
            settings, settings_blockers = _local_settings_for_mode(
                exact_mode,
                fit=fit,
                policy=policy,
            )
            if settings_blockers:
                result_rows.append(
                    {
                        **base,
                        "replay_status": "BLOCKED_LOCAL_FIT",
                        "replay_blocker": ";".join(settings_blockers),
                        **_fit_fields(fit),
                    }
                )
                continue
            test = fit["test"].copy()
            test["hedge_ratio"] = _exposure_hedge_ratio(
                test,
                exact_mode=exact_mode,
                settings=settings,
                prior_history=fit["train"],
            )
            hedge_ratio_blocker = _hedge_ratio_contract_blocker(test["hedge_ratio"])
            if hedge_ratio_blocker:
                result_rows.append(
                    {
                        **base,
                        "replay_status": "BLOCKED_ECONOMIC_CONTRACT",
                        "replay_blocker": hedge_ratio_blocker,
                        **_fit_fields(fit),
                    }
                )
                continue
            mode_result = build_local_mode_signal(test, settings, exact_mode=exact_mode)
            if mode_result.mode_replay_status != "READY_FOR_RESEARCH_REPLAY":
                result_rows.append(
                    {
                        **base,
                        "replay_status": "BLOCKED_MODE_INPUTS",
                        "replay_blocker": ";".join(mode_result.missing_inputs),
                        **_fit_fields(fit),
                    }
                )
                continue
            result, ledger = backtest_two_leg_spread_with_ledger(
                test,
                mode_result.signal,
                costs,
                interval=_text(experiment.timeframe)
                .lower()
                .replace("daily", "1d")
                .replace("hourly", "1h"),
            )
        except Exception as exc:
            result_rows.append(
                {
                    **base,
                    "replay_status": "BLOCKED_REPLAY_ERROR",
                    "replay_blocker": f"{safe_exception_code(exc)}",
                }
            )
            continue

        metrics = asdict(result)
        fit_fields = _fit_fields(fit)
        result_rows.append(
            {
                **base,
                **metrics,
                **fit_fields,
                "replay_status": "RESEARCH_REPLAY_COMPLETE",
                "replay_blocker": "",
                "metric_name": mode_result.metric_name,
                "mode_fidelity_status": mode_result.mode_fidelity_status,
                "mode_fidelity_reason": mode_result.mode_fidelity_reason,
                "mode_setting_source": LOCAL_SETTINGS_VERSION,
                "settings_json": _canonical_json(settings),
                "entry_style": _entry_style(settings, exact_mode),
                "exit_style": _exit_style(settings, exact_mode),
                "mode_equivalence_warning": _mode_equivalence_warning(exact_mode),
                "computation_notes": ";".join(mode_result.computation_notes),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": PROVISIONAL_ACCEPTANCE_BLOCKER,
                "local_replay_completed": True,
            }
        )
        for lifecycle, frame in (
            ("closed", ledger.closed_trades),
            ("open", ledger.open_trades),
        ):
            for trade in frame.to_dict("records"):
                trade_rows.append(
                    {
                        "schema_version": REPLAY_SCHEMA_VERSION,
                        "handoff_id": handoff_id,
                        "history_run_id": history_run_id,
                        "canonical_replay_id": replay_id,
                        "experiment_id": _text(experiment.experiment_id),
                        "pair_group_key": pair_key,
                        "pair": _text(experiment.pair),
                        "exact_mode": exact_mode,
                        "orientation": orientation,
                        "trade_lifecycle": lifecycle,
                        **trade,
                        "backtest_label": True,
                        "paper_label": False,
                        "live_label": False,
                        "promotion_authority": False,
                        "live_trading_authorized": False,
                    }
                )

    results = pd.DataFrame(result_rows)
    if len(results) != len(experiments) or results["experiment_id"].nunique() != len(experiments):
        raise ValueError("Current canonical replay failed complete experiment accounting")
    results["research_rank_eligible"] = False
    results["research_rank_blocker"] = "replay_not_complete"
    complete_mask = results["replay_status"].eq("RESEARCH_REPLAY_COMPLETE")
    rank_blockers = results.loc[complete_mask].apply(_rank_blocker, axis=1)
    results.loc[complete_mask, "research_rank_blocker"] = rank_blockers
    results.loc[complete_mask, "research_rank_eligible"] = rank_blockers.eq("")
    completed = results.loc[complete_mask].copy()
    if completed.empty:
        ranked = completed
    else:
        eligible = completed.loc[completed["research_rank_eligible"].astype(bool)].sort_values(
            ["profit_factor", "sharpe", "max_drawdown", "trades"],
            ascending=[False, False, True, False],
            na_position="last",
        )
        eligible.insert(0, "research_rank", range(1, len(eligible) + 1))
        ineligible = completed.loc[~completed["research_rank_eligible"].astype(bool)].sort_values(
            ["trades", "sharpe", "max_drawdown"],
            ascending=[False, False, True],
            na_position="last",
        )
        ineligible.insert(0, "research_rank", "")
        ranked = pd.concat([eligible, ineligible], ignore_index=True)
    trades = pd.DataFrame(trade_rows)
    pair_status = _build_replay_pair_status(pair_histories, results)
    validation = _replay_validation(experiments=experiments, results=results)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current canonical replay validation failed: " + ",".join(failed))

    for frame, active_key, snapshot_key in (
        (results, "results", "snapshot_results"),
        (ranked, "ranked", "snapshot_ranked"),
        (trades, "trades", "snapshot_trades"),
        (pair_status, "pair_status", "snapshot_pair_status"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)

    status_counts = results["replay_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "handoff_id": handoff_id,
        "refresh_id": refresh_id,
        "history_run_id": history_run_id,
        "canonical_replay_id": replay_id,
        "created_at": as_of.isoformat(),
        "experiments_accounted": int(len(results)),
        "unique_experiment_ids": int(results["experiment_id"].nunique()),
        "pair_groups_accounted": int(pair_status["pair_group_key"].nunique()),
        "replay_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "replay_status_count_total": int(sum(status_counts.values())),
        "research_replays_complete": int(status_counts.get("RESEARCH_REPLAY_COMPLETE", 0)),
        "deferred_point_in_time_history": int(
            status_counts.get("DEFERRED_POINT_IN_TIME_HISTORY", 0)
        ),
        "blocked_point_in_time_history": int(status_counts.get("BLOCKED_POINT_IN_TIME_HISTORY", 0)),
        "blocked_local_fit": int(status_counts.get("BLOCKED_LOCAL_FIT", 0)),
        "blocked_replay_error": int(status_counts.get("BLOCKED_REPLAY_ERROR", 0)),
        "blocked_economic_contract": int(status_counts.get("BLOCKED_ECONOMIC_CONTRACT", 0)),
        "not_applicable_vendor_mode": int(status_counts.get("NOT_APPLICABLE_VENDOR_MODE", 0)),
        "mapping_blocked_experiments": int(status_counts.get("BLOCKED_HYPERLIQUID_MAPPING", 0)),
        "trade_ledger_rows": int(len(trades)),
        "research_rank_eligible_replays": int(results["research_rank_eligible"].astype(bool).sum()),
        "minimum_research_rank_trades": MINIMUM_RESEARCH_RANK_TRADES,
        "canonical_replay_leverage": 1.0,
        "train_only_parameter_fit": True,
        "test_only_performance_measurement": True,
        "settings_version": LOCAL_SETTINGS_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "source_handoff_economic_contract_version": source_contract_version,
        "input_contract_status": input_contract_status,
        "historical_math_reevaluation": legacy_contract_math_audit,
        "source_handoff_manifest_sha256": _file_hash(handoff_manifest_path),
        "source_history_manifest_sha256": _file_hash(history_manifest_path),
        "math_implementation_hashes": math_implementation_hashes,
        "cost_evidence_status": "PROVISIONAL_CONSERVATIVE_DEFAULTS",
        "cost_model": cost_payload,
        "replay_policy": policy_payload,
        "acceptance_eligible_replays": 0,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "input_hashes": {
            "experiment_matrix": _file_hash(experiment_path),
            "pair_history_results": _file_hash(pair_history_path),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _replay_summary(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _fit_orientation_context(
    history: pd.DataFrame,
    *,
    policy: CurrentReplayPolicy,
) -> dict[str, Any]:
    rows = len(history)
    split = int(math.floor(rows * policy.train_fraction))
    split = max(policy.minimum_train_rows, split)
    split = min(split, rows - policy.minimum_test_rows)
    blockers: list[str] = []
    if split < policy.minimum_train_rows:
        blockers.append(f"train_rows<{policy.minimum_train_rows}")
    if rows - split < policy.minimum_test_rows:
        blockers.append(f"test_rows<{policy.minimum_test_rows}")
    train = history.iloc[: max(split, 0)].copy()
    test = history.iloc[max(split, 0) :].copy()
    engle = fit_engle_granger(train["price_x"], train["price_y"])
    if engle.validity_status == "valid":
        hedge_ratio = float(engle.values["hedge_ratio"])
        static_spread = y_on_x_log_spread(
            train["price_x"],
            train["price_y"],
            hedge_ratio,
        )
        ou = fit_ou(static_spread)
    else:
        hedge_ratio = float("nan")
        static_spread = pd.Series(np.nan, index=train.index)
        ou = fit_ou(static_spread)
    return {
        "rows": rows,
        "split": split,
        "train": train,
        "test": test,
        "engle": engle,
        "hedge_ratio": hedge_ratio,
        "static_spread": static_spread,
        "ou": ou,
        "base_blockers": tuple(blockers),
    }


def _local_settings_for_mode(
    exact_mode: str,
    *,
    fit: dict[str, Any],
    policy: CurrentReplayPolicy,
) -> tuple[dict[str, object], tuple[str, ...]]:
    mode = _text(exact_mode)
    blockers = list(fit["base_blockers"])
    static_family = mode.startswith("Static") or mode.startswith("OU")
    if static_family or mode == "Copula":
        if fit["engle"].validity_status != "valid":
            blockers.append(f"engle_granger_invalid:{fit['engle'].validity_reason}")
        elif not math.isfinite(float(fit["hedge_ratio"])) or float(fit["hedge_ratio"]) <= 0.0:
            blockers.append("unsupported_nonpositive_y_on_x_hedge_ratio")
    if mode.startswith("OU") and fit["ou"].validity_status != "valid":
        blockers.append(f"ou_fit_invalid:{fit['ou'].validity_reason}")

    lower_action, upper_action = tail_actions(
        mode,
        copula_direction_view="u1_given_u2",
    )
    ou_mu = 0.0
    ou_sigma = 1.0
    if fit["ou"].validity_status == "valid":
        ou_mu = float(fit["ou"].values["mu"])
        phi = float(fit["ou"].values["phi"])
        innovation = float(fit["ou"].values["innovation_sigma"])
        denominator = math.sqrt(max(1.0 - phi**2, np.finfo(float).eps))
        ou_sigma = innovation / denominator
        if not math.isfinite(ou_sigma) or ou_sigma <= 0.0:
            blockers.append("ou_stationary_sigma_invalid")
            ou_sigma = 1.0
    settings: dict[str, object] = {
        "capture_confirmed": True,
        "entry_long_operator": "<=",
        "entry_long_value": -abs(policy.entry_zscore),
        "entry_long_position": lower_action.value,
        "entry_short_operator": ">=",
        "entry_short_value": abs(policy.entry_zscore),
        "entry_short_position": upper_action.value,
        "exit_long_operator": ">=",
        "exit_long_value": policy.exit_zscore,
        "exit_short_operator": "<=",
        "exit_short_value": policy.exit_zscore,
        "hedge_ratio": fit["hedge_ratio"],
        "zscore_window": policy.zscore_window,
        "dynamic_hedge_ratio_method": "history_captured_hedge_ratio",
        "dynamic_hedge_ratio_window": policy.dynamic_window,
        "dynamic_hedge_ratio_source": ("causal_rolling_y_on_x_ols_seeded_with_training_history"),
        "ou_mu": ou_mu,
        "ou_sigma": ou_sigma,
        "ou_sigma_definition": "train_ar1_stationary_standard_deviation",
        "copula_family": "gaussian",
        "copula_signal_type": "conditional_cdf_tail",
        "copula_direction_view": "u1_given_u2",
        "copula_window": policy.copula_window,
        "copula_entry_lower": policy.copula_entry_lower,
        "copula_entry_upper": policy.copula_entry_upper,
        "copula_exit_lower": policy.copula_exit_lower,
        "copula_exit_upper": policy.copula_exit_upper,
        "settings_authority": "LOCAL_STANDARDIZED_MATH_V2_NOT_WIZARD_PARITY",
        "settings_version": LOCAL_SETTINGS_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
    }
    if mode == "OU (Spread)":
        settings["entry_long_value"] = -abs(policy.entry_zscore) * ou_sigma
        settings["entry_short_value"] = abs(policy.entry_zscore) * ou_sigma
        settings["exit_long_value"] = policy.exit_zscore * ou_sigma
        settings["exit_short_value"] = policy.exit_zscore * ou_sigma
    return settings, tuple(dict.fromkeys(blockers))


def _exposure_hedge_ratio(
    history: pd.DataFrame,
    *,
    exact_mode: str,
    settings: dict[str, object],
    prior_history: pd.DataFrame | None = None,
) -> pd.Series:
    if not exact_mode.startswith("Dyn"):
        return pd.Series(float(settings["hedge_ratio"]), index=history.index)
    window = int(settings["dynamic_hedge_ratio_window"])
    seed = prior_history.tail(window - 1) if prior_history is not None else history.iloc[0:0]
    combined = pd.concat([seed, history])
    log_x = np.log(pd.to_numeric(combined["price_x"], errors="coerce"))
    log_y = np.log(pd.to_numeric(combined["price_y"], errors="coerce"))
    return rolling_y_on_x_beta(log_x, log_y, window=window).reindex(history.index)


def _hedge_ratio_contract_blocker(hedge_ratio: pd.Series) -> str:
    values = pd.to_numeric(hedge_ratio, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        return "point_in_time_hedge_ratio_missing_or_nonfinite"
    if values.le(0.0).any():
        return "point_in_time_hedge_ratio_nonpositive_opposing_legs_unsupported"
    return ""


def _load_history(path: Path) -> pd.DataFrame:
    payload = _read_json(path)
    frame = pd.DataFrame(payload.get("history", []))
    required = {"timestamp", "price_x", "price_y"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("pair history has no timestamped two-leg prices")
    columns = [
        column
        for column in (
            "timestamp",
            "price_x",
            "price_y",
            "funding_x_bps",
            "funding_y_bps",
            "funding_x_realized_bps",
            "funding_y_realized_bps",
            "volume_x_usd",
            "volume_y_usd",
        )
        if column in frame.columns
    ]
    frame = frame[columns].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for column in columns:
        if column != "timestamp":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "price_x", "price_y"])
    frame = frame[(frame["price_x"] > 0.0) & (frame["price_y"] > 0.0)]
    frame = frame.drop_duplicates("timestamp", keep="last").sort_values("timestamp")
    return frame.set_index("timestamp", drop=False)


def _orient_history(history: pd.DataFrame, *, orientation: str) -> pd.DataFrame:
    if orientation == "original":
        return history.copy()
    if orientation != "reverse":
        raise ValueError(f"unsupported_orientation:{orientation}")
    result = history.copy()
    for left, right in (
        ("price_x", "price_y"),
        ("funding_x_bps", "funding_y_bps"),
        ("funding_x_realized_bps", "funding_y_realized_bps"),
        ("volume_x_usd", "volume_y_usd"),
    ):
        if left in result.columns and right in result.columns:
            result[left], result[right] = history[right].copy(), history[left].copy()
    return result


def _replay_base_row(
    experiment: object,
    *,
    replay_id: str,
    history_run_id: str,
    cost_payload: dict[str, object],
    evidence_paths: tuple[Path, ...],
    root: Path,
) -> dict[str, object]:
    orientation = _text(experiment.orientation)
    asset_a = _text(experiment.asset_a)
    asset_b = _text(experiment.asset_b)
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "economic_contract_version": ECONOMIC_CONTRACT_VERSION,
        "handoff_id": _text(experiment.handoff_id),
        "refresh_id": _text(experiment.refresh_id),
        "history_run_id": history_run_id,
        "canonical_replay_id": replay_id,
        "experiment_id": _text(experiment.experiment_id),
        "pair_group_key": _text(experiment.pair_group_key),
        "pair": _text(experiment.pair),
        "wizard_exchange": _text(experiment.wizard_exchange),
        "timeframe": _text(experiment.timeframe),
        "exact_mode": _text(experiment.exact_mode),
        "orientation": orientation,
        "asset_a": asset_a,
        "asset_b": asset_b,
        "replay_asset_x": asset_b if orientation == "reverse" else asset_a,
        "replay_asset_y": asset_a if orientation == "reverse" else asset_b,
        "original_experiment_status": _text(experiment.experiment_status),
        "original_experiment_blocker": _text(experiment.experiment_blocker),
        "canonical_replay_leverage": 1.0,
        "taker_fee_bps": cost_payload["taker_fee_bps"],
        "slippage_bps": cost_payload["slippage_bps"],
        "execution_risk_bps": cost_payload["execution_risk_bps"],
        "funding_bps_per_day": cost_payload["funding_bps_per_day"],
        "partial_fill_probability": cost_payload["partial_fill_probability"],
        "cost_evidence_status": "PROVISIONAL_CONSERVATIVE_DEFAULTS",
        "replay_status": "",
        "replay_blocker": "",
        "metric_name": "",
        "mode_fidelity_status": "",
        "mode_fidelity_reason": "",
        "mode_setting_source": "",
        "settings_json": "",
        "entry_style": "",
        "exit_style": "",
        "mode_equivalence_warning": "",
        "computation_notes": "",
        "train_rows": 0,
        "test_rows": 0,
        "train_start": "",
        "train_end": "",
        "test_start": "",
        "test_end": "",
        "engle_granger_status": "",
        "engle_granger_pvalue": np.nan,
        "hedge_ratio": np.nan,
        "ou_fit_status": "",
        "ou_phi": np.nan,
        "ou_half_life": np.nan,
        "trades": 0,
        "open_trades": 0,
        "profit_factor": 0.0,
        "expectancy": 0.0,
        "expectancy_lower_95": np.nan,
        "sharpe": 0.0,
        "sharpe_status": "blocked",
        "max_drawdown": 0.0,
        "win_rate": 0.0,
        "total_return": 0.0,
        "gross_return": 0.0,
        "total_fees": 0.0,
        "total_slippage": 0.0,
        "total_funding": 0.0,
        "total_execution_risk": 0.0,
        "total_partial_fill_cost": 0.0,
        "avg_gross_exposure": 0.0,
        "reconciliation_error": 0.0,
        "acceptance_status": "BLOCKED",
        "acceptance_reason": "replay_not_complete",
        "local_replay_completed": False,
        "discovery_prefilter_applied": False,
        "vendor_parity_claimed": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "evidence_path": ";".join(_relative(path, root) for path in evidence_paths),
    }


def _fit_fields(fit: dict[str, Any]) -> dict[str, object]:
    train = fit["train"]
    test = fit["test"]
    engle = fit["engle"]
    ou = fit["ou"]
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_start": train.index[0].isoformat() if not train.empty else "",
        "train_end": train.index[-1].isoformat() if not train.empty else "",
        "test_start": test.index[0].isoformat() if not test.empty else "",
        "test_end": test.index[-1].isoformat() if not test.empty else "",
        "engle_granger_status": engle.validity_status,
        "engle_granger_pvalue": engle.values.get("cointegration_pvalue", np.nan),
        "hedge_ratio": fit["hedge_ratio"],
        "ou_fit_status": ou.validity_status,
        "ou_phi": ou.values.get("phi", np.nan),
        "ou_half_life": ou.values.get("half_life", np.nan),
    }


def _build_replay_pair_status(
    pair_histories: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for pair in pair_histories.itertuples():
        pair_key = _text(pair.pair_group_key)
        cells = results.loc[results["pair_group_key"].astype(str).eq(pair_key)]
        completed = cells.loc[cells["replay_status"].eq("RESEARCH_REPLAY_COMPLETE")]
        supported = completed.loc[completed["research_rank_eligible"].astype(bool)]
        best = (
            supported.sort_values(
                ["profit_factor", "sharpe", "max_drawdown", "trades"],
                ascending=[False, False, True, False],
                na_position="last",
            ).iloc[0]
            if not supported.empty
            else pd.Series(dtype=object)
        )
        statuses = cells["replay_status"].value_counts().to_dict()
        rows.append(
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "handoff_id": _text(pair.handoff_id),
                "history_run_id": _text(pair.history_run_id),
                "pair_group_key": pair_key,
                "pair": _text(pair.pair),
                "wizard_exchange": _text(pair.wizard_exchange),
                "timeframe": _text(pair.timeframe),
                "history_status": _text(pair.history_status),
                "planned_cells": len(cells),
                "completed_replay_cells": int(statuses.get("RESEARCH_REPLAY_COMPLETE", 0)),
                "deferred_cells": int(statuses.get("DEFERRED_POINT_IN_TIME_HISTORY", 0)),
                "blocked_cells": int(
                    len(cells)
                    - statuses.get("RESEARCH_REPLAY_COMPLETE", 0)
                    - statuses.get("DEFERRED_POINT_IN_TIME_HISTORY", 0)
                ),
                "sample_supported_cells": int(len(supported)),
                "best_selection_scope": (
                    "sample_supported_research_cells" if not supported.empty else "NONE"
                ),
                "best_exact_mode": _text(best.get("exact_mode", "")),
                "best_orientation": _text(best.get("orientation", "")),
                "best_trades": _integer(best.get("trades", 0)),
                "best_profit_factor": _number(best.get("profit_factor")),
                "best_sharpe": _number(best.get("sharpe")),
                "best_max_drawdown": _number(best.get("max_drawdown")),
                "acceptance_status": "BLOCKED",
                "acceptance_reason": (
                    PROVISIONAL_ACCEPTANCE_BLOCKER
                    if not supported.empty
                    else "no_sample_supported_1x_cell;" + PROVISIONAL_ACCEPTANCE_BLOCKER
                    if not completed.empty
                    else _text(pair.history_blocker) or "canonical_1x_replay_not_complete"
                ),
                "canonical_replay_leverage": 1.0,
                "promotion_authority": False,
                "live_trading_authorized": False,
                "evidence_path": _text(pair.evidence_path),
            }
        )
    return pd.DataFrame(rows)


def _history_validation(
    *,
    pair_queue: pd.DataFrame,
    pair_results: pd.DataFrame,
    asset_results: pd.DataFrame,
    selected: tuple[str, ...],
    storage_preflight_passed: bool,
) -> pd.DataFrame:
    selected_results = pair_results.loc[pair_results["pair_group_key"].astype(str).isin(selected)]
    unselected_ready = pair_results.loc[
        ~pair_results["pair_group_key"].astype(str).isin(selected)
        & pair_queue["history_request_status"].astype(str).eq("READY_TO_FETCH").to_numpy()
    ]
    checks = [
        (
            "every_current_pair_accounted",
            len(pair_results) == len(pair_queue)
            and pair_results["pair_group_key"].nunique() == len(pair_queue),
            f"accounted={len(pair_results)} expected={len(pair_queue)}",
        ),
        (
            "selected_pairs_accounted",
            selected_results["pair_group_key"].nunique() == len(selected),
            f"accounted={selected_results['pair_group_key'].nunique()} expected={len(selected)}",
        ),
        (
            "unselected_ready_pairs_explicitly_deferred",
            unselected_ready["history_status"].eq("DEFERRED_NOT_SELECTED").all(),
            f"deferred={int(unselected_ready['history_status'].eq('DEFERRED_NOT_SELECTED').sum())} rows={len(unselected_ready)}",
        ),
        (
            "ready_histories_have_no_post_cutoff_rows",
            pd.to_numeric(
                pair_results.loc[
                    pair_results["history_status"].eq("READY_FOR_CANONICAL_1X_REPLAY"),
                    "post_cutoff_rows",
                ],
                errors="coerce",
            )
            .fillna(1)
            .eq(0)
            .all(),
            "point-in-time pair histories are cutoff bounded",
        ),
        (
            "storage_failure_prevents_network_completion",
            storage_preflight_passed or not asset_results["history_status"].eq("COMPLETE").any(),
            f"storage_preflight_passed={storage_preflight_passed}",
        ),
        (
            "no_live_authority",
            not pair_results["live_trading_authorized"].astype(bool).any()
            and not asset_results["live_trading_authorized"].astype(bool).any(),
            "history materialization is research-only",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
            for name, passed, detail in checks
        ]
    )


def _replay_validation(*, experiments: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    ready = experiments["experiment_status"].eq("READY_FOR_POINT_IN_TIME_HISTORY")
    ready_results = results.loc[ready.to_numpy()]
    attempted_statuses = {
        "RESEARCH_REPLAY_COMPLETE",
        "DEFERRED_POINT_IN_TIME_HISTORY",
        "BLOCKED_POINT_IN_TIME_HISTORY",
        "BLOCKED_LOCAL_FIT",
        "BLOCKED_ECONOMIC_CONTRACT",
        "BLOCKED_MODE_INPUTS",
        "BLOCKED_REPLAY_ERROR",
    }
    checks = [
        (
            "every_experiment_accounted",
            len(results) == len(experiments)
            and results["experiment_id"].nunique() == len(experiments),
            f"accounted={len(results)} expected={len(experiments)}",
        ),
        (
            "every_locally_ready_cell_attempted_or_deferred",
            ready_results["replay_status"].isin(attempted_statuses).all(),
            f"ready_cells={int(ready.sum())}",
        ),
        (
            "canonical_replay_is_one_x",
            pd.to_numeric(results["canonical_replay_leverage"], errors="coerce").eq(1.0).all(),
            "all cells use normalized 1x gross exposure",
        ),
        (
            "no_promotion_authority",
            not results["promotion_authority"].astype(bool).any(),
            "canonical replay is research evidence only",
        ),
        (
            "no_live_authority",
            not results["live_trading_authorized"].astype(bool).any(),
            "canonical replay cannot submit orders",
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "PASS" if passed else "FAIL", "detail": detail}
            for name, passed, detail in checks
        ]
    )


def _rank_blocker(row: pd.Series) -> str:
    blockers: list[str] = []
    if _integer(row.get("trades")) < MINIMUM_RESEARCH_RANK_TRADES:
        blockers.append(f"closed_trades<{MINIMUM_RESEARCH_RANK_TRADES}")
    if _text(row.get("sharpe_status")) != "valid":
        blockers.append("sharpe_invalid")
    if not math.isfinite(float(row.get("profit_factor", float("nan")))):
        blockers.append("profit_factor_nonfinite")
    if abs(float(row.get("reconciliation_error", 0.0))) > 1e-10:
        blockers.append("ledger_reconciliation_error")
    return ";".join(blockers)


def _entry_style(settings: dict[str, object], exact_mode: str) -> str:
    if exact_mode == "Copula":
        return (
            f"causal_gaussian_conditional_cdf:"
            f"{settings['copula_entry_lower']}/{settings['copula_entry_upper']}"
        )
    return f"fixed_sigma_threshold:{settings['entry_long_value']}/{settings['entry_short_value']}"


def _exit_style(settings: dict[str, object], exact_mode: str) -> str:
    if exact_mode == "Copula":
        return (
            f"conditional_normalization:"
            f"{settings['copula_exit_lower']}/{settings['copula_exit_upper']}"
        )
    return f"mean_cross:{settings['exit_long_value']}/{settings['exit_short_value']}"


def _mode_equivalence_warning(exact_mode: str) -> str:
    if exact_mode == "OU (ZScoreR)":
        return (
            "local_ou_zscorer_equals_static_zscorer_when_constant_ou_mu_is_removed_"
            "by_rolling_standardization"
        )
    return ""


def _trim_pair_history_to_cutoff(path: Path, *, cutoff: datetime | None) -> None:
    if cutoff is None:
        return
    payload = _read_json(path)
    history = payload.get("history", [])
    if not isinstance(history, list):
        raise ValueError("pair history payload has no history list")
    kept: list[dict[str, object]] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        timestamp = _parse_timestamp(row.get("timestamp"))
        if timestamp is not None and timestamp <= cutoff:
            kept.append(row)
    payload["history"] = kept
    payload["period"] = len(kept)
    payload["point_in_time_cutoff_at"] = cutoff.isoformat()
    payload["post_cutoff_rows_removed"] = len(history) - len(kept)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _candle_metadata(path: Path | None, *, cutoff: datetime | None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if path is not None and path.exists():
        payload = _read_json(path)
        candidate = payload.get("candles", [])
        rows = candidate if isinstance(candidate, list) else []
    return _timestamp_metadata(rows, timestamp_field="startedAt", cutoff=cutoff)


def _pair_metadata(path: Path | None, *, cutoff: datetime | None) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    if path is not None and path.exists():
        payload = _read_json(path)
        candidate = payload.get("history", [])
        rows = candidate if isinstance(candidate, list) else []
    return _timestamp_metadata(rows, timestamp_field="timestamp", cutoff=cutoff)


def _timestamp_metadata(
    rows: list[dict[str, object]],
    *,
    timestamp_field: str,
    cutoff: datetime | None,
) -> dict[str, object]:
    timestamps = [
        timestamp
        for row in rows
        if (timestamp := _parse_timestamp(row.get(timestamp_field))) is not None
    ]
    valid = bool(rows) and len(timestamps) == len(rows)
    post_cutoff = sum(timestamp > cutoff for timestamp in timestamps) if cutoff else 0
    return {
        "rows": len(rows),
        "earliest": min(timestamps).isoformat() if timestamps else "",
        "latest": max(timestamps).isoformat() if timestamps else "",
        "timestamp_parse_valid": valid,
        "post_cutoff_rows": post_cutoff,
        "timestamp_bound_valid": bool(valid and cutoff is not None and post_cutoff == 0),
    }


def _validate_replay_policy(policy: CurrentReplayPolicy) -> None:
    if not 0.0 < policy.train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if (
        min(
            policy.minimum_train_rows,
            policy.minimum_test_rows,
            policy.zscore_window,
            policy.dynamic_window,
            policy.copula_window,
        )
        <= 1
    ):
        raise ValueError("replay row and window policies must exceed one")


def _history_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Bounded History",
            "",
            f"- History run: `{summary['history_run_id']}`",
            f"- Selected pair groups: {summary['selected_pair_groups']}",
            f"- All pair groups accounted: {summary['pair_groups_accounted']}",
            f"- Pair histories ready: {summary['pair_histories_ready']}",
            f"- Storage preflight passed: {summary['storage_preflight_passed']}",
            f"- Estimated bytes: {summary['estimated_materialization_bytes']}",
            "- Network scope: explicit selected pairs only",
            "- Promotion authority: false",
            "- Live trading authorized: false",
            "",
        ]
    )


def _replay_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Canonical 1x Replay",
            "",
            f"- Canonical replay: `{summary['canonical_replay_id']}`",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Pair groups accounted: {summary['pair_groups_accounted']}",
            f"- Research replays complete: {summary['research_replays_complete']}",
            f"- Deferred cells: {summary['deferred_point_in_time_history']}",
            f"- Replay errors: {summary['blocked_replay_error']}",
            f"- Economic-contract blockers: {summary['blocked_economic_contract']}",
            f"- Rank-eligible research cells: {summary['research_rank_eligible_replays']}",
            f"- Settings version: `{summary['settings_version']}`",
            f"- Economic contract: `{summary['economic_contract_version']}`",
            f"- Hash-bound math files: {len(summary['math_implementation_hashes'])}",
            "- Parameters: fitted on the training slice only",
            "- Performance: measured on the held-out test slice only",
            "- Exposure: normalized 1x; leverage scenarios are separate",
            "- Acceptance eligible: 0",
            "- Promotion authority: false",
            "- Live trading authorized: false",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _math_implementation_hashes() -> dict[str, str]:
    repository_root = Path(__file__).resolve().parents[3]
    hashes: dict[str, str] = {}
    for relative_path in MATH_IMPLEMENTATION_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Canonical replay implementation file missing: {path}")
        hashes[relative_path] = _file_hash(path)
    return hashes


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _split_values(value: object) -> set[str]:
    return {part.strip() for part in _text(value).split(";") if part.strip()}


def _parse_timestamp(value: object) -> datetime | None:
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return timestamp.to_pydatetime()


def _integer(value: object) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
