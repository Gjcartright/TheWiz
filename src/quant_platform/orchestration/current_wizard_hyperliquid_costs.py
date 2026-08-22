"""Point-in-time Hyperliquid cost evidence for the current Wizard board."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from quant_platform.hyperliquid import (
    fetch_hyperliquid_funding_history,
    normalize_hyperliquid_funding_history,
)
from quant_platform.orchestration.canonical_wizard_hyperliquid_contract import (
    MAXIMUM_FEE_EVIDENCE_AGE_DAYS,
    MINIMUM_FUNDING_COVERAGE,
    MINIMUM_PROVISIONAL_FUNDED_ROWS,
    MINIMUM_PROVISIONAL_L2_SAMPLES,
    MINIMUM_STRICT_L2_SAMPLES,
    PROVISIONAL_L2_WINDOW_HOURS,
    REFERENCE_LEG_NOTIONAL_USD,
    STRICT_L2_WINDOW_HOURS,
)
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    atomic_copy_file,
    atomic_write_csv,
    atomic_write_text,
    immutable_snapshot_copy,
)
from quant_platform.runtime_types import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_cost_evidence.v1"
LEG_NOTIONAL_USD = REFERENCE_LEG_NOTIONAL_USD


def materialize_current_wizard_hyperliquid_cost_evidence(
    *,
    pair_group_keys: Iterable[str],
    root: Path = ROOT,
    now: datetime | None = None,
    fetch_funding: bool = True,
    funding_fetcher: Callable[..., Path] = fetch_hyperliquid_funding_history,
    max_funding_workers: int = 4,
) -> CommandResult:
    """Build exact-pair funding and L2 evidence without authorizing execution.

    Network work is bounded to the explicitly selected pair keys. Every pair
    and experiment on the current board still receives an explicit status.
    """

    selected = tuple(dict.fromkeys(_text(value) for value in pair_group_keys if _text(value)))
    if not selected:
        raise ValueError("pair_group_keys must explicitly select at least one pair")
    if max_funding_workers <= 0:
        raise ValueError("max_funding_workers must be positive")
    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    history_manifest_path = active / "current_wizard_hyperliquid_history_manifest.json"
    replay_manifest_path = active / "current_wizard_hyperliquid_canonical_replay_manifest.json"
    if not history_manifest_path.exists() or not replay_manifest_path.exists():
        raise FileNotFoundError("Current history and canonical replay manifests are required")
    history_manifest = _read_json(history_manifest_path)
    replay_manifest = _read_json(replay_manifest_path)
    history_artifacts = history_manifest.get("artifacts", {})
    replay_artifacts = replay_manifest.get("artifacts", {})
    pair_results_path = root / _text(history_artifacts.get("snapshot_pair_results"))
    experiment_path = root / _text(replay_artifacts.get("snapshot_results"))
    l2_path = root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv"
    fee_profile_path = root / "config" / "hyperliquid_perp_cost_profile.json"
    required = (pair_results_path, experiment_path, l2_path, fee_profile_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Current cost evidence inputs missing: {missing}")

    pairs = pd.read_csv(pair_results_path)
    experiments = pd.read_csv(experiment_path)
    known = set(pairs["pair_group_key"].astype(str))
    unknown = sorted(set(selected) - known)
    if unknown:
        raise ValueError(f"Unknown current pair_group_keys: {unknown}")
    if pairs["pair_group_key"].duplicated().any():
        raise ValueError("Current pair history contains duplicate pair_group_key values")
    if experiments["experiment_id"].duplicated().any():
        raise ValueError("Current canonical replay contains duplicate experiment_id values")

    profile = _read_json(fee_profile_path)
    samples = _normalize_l2_samples(pd.read_csv(l2_path))
    cache_ledgers = {
        "prior_current_funding_assets": active
        / "current_wizard_hyperliquid_funding_asset_results.csv",
        "exhaustive_funding_assets": active
        / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
    }
    cache_ledgers = {
        name: path for name, path in cache_ledgers.items() if path.exists()
    }
    material = {
        "schema_version": SCHEMA_VERSION,
        "handoff_id": _text(history_manifest.get("handoff_id")),
        "refresh_id": _text(history_manifest.get("refresh_id")),
        "history_run_id": _text(history_manifest.get("history_run_id")),
        "canonical_replay_id": _text(replay_manifest.get("canonical_replay_id")),
        "selected_pair_group_keys": selected,
        "as_of": as_of.isoformat(),
        "input_hashes": {
            "pair_results": _file_hash(pair_results_path),
            "canonical_replay": _file_hash(experiment_path),
            "l2_samples": _file_hash(l2_path),
            "fee_profile": _file_hash(fee_profile_path),
            **{
                name: _file_hash(path) for name, path in cache_ledgers.items()
            },
        },
        "max_funding_workers": max_funding_workers,
    }
    cost_evidence_id = "cwcost_" + sha256(_canonical_json(material).encode()).hexdigest()[:20]
    history_snapshot_manifest = root / _text(history_artifacts.get("snapshot_manifest"))
    snapshot_dir = history_snapshot_manifest.parent / "cost_evidence" / cost_evidence_id
    funding_dir = snapshot_dir / "funding_assets"
    enriched_dir = snapshot_dir / "enriched_pairs"
    input_dir = snapshot_dir / "inputs"
    for directory in (funding_dir, enriched_dir, input_dir):
        directory.mkdir(parents=True, exist_ok=True)
    snapshot_inputs: dict[str, Path] = {}
    for name, source in {
        "pair_results": pair_results_path,
        "canonical_replay": experiment_path,
        "l2_samples": l2_path,
        "fee_profile": fee_profile_path,
        **cache_ledgers,
    }.items():
        target = immutable_snapshot_copy(source, input_dir, artifact_name=name)
        snapshot_inputs[name] = target

    selected_ready = pairs.loc[
        pairs["pair_group_key"].astype(str).isin(selected)
        & pairs["history_status"].astype(str).eq("READY_FOR_CANONICAL_1X_REPLAY")
    ].copy()
    asset_requests = _asset_requests(selected_ready)
    cache_lookup = _funding_cache_lookup(
        asset_requests=asset_requests,
        ledger_paths=cache_ledgers.values(),
        root=root,
    )
    asset_results: dict[str, tuple[dict[str, object], pd.DataFrame | None]] = {}
    with ThreadPoolExecutor(
        max_workers=min(max_funding_workers, max(1, len(asset_requests)))
    ) as executor:
        future_assets = {
            executor.submit(
                copy_context().run,
                _materialize_funding_asset,
                asset=asset,
                request=request,
                funding_dir=funding_dir,
                root=root,
                cost_evidence_id=cost_evidence_id,
                cache_record=cache_lookup.get(asset),
                fetch_funding=fetch_funding,
                funding_fetcher=funding_fetcher,
            ): asset
            for asset, request in sorted(asset_requests.items())
        }
        for future in as_completed(future_assets):
            asset = future_assets[future]
            asset_results[asset] = future.result()
    asset_rows = [asset_results[asset][0] for asset in sorted(asset_results)]
    funding_frames = {
        asset: result[1]
        for asset, result in sorted(asset_results.items())
        if result[1] is not None
    }
    assets = pd.DataFrame(asset_rows)

    pair_rows = [
        _pair_cost_row(
            pair,
            selected=set(selected),
            funding_frames=funding_frames,
            samples=samples,
            profile=profile,
            profile_path=fee_profile_path,
            l2_path=l2_path,
            enriched_dir=enriched_dir,
            root=root,
            as_of=as_of,
            cost_evidence_id=cost_evidence_id,
        )
        for pair in pairs.itertuples()
    ]
    pair_frame = pd.DataFrame(pair_rows)
    pair_lookup = {
        _text(row.pair_group_key): row for row in pair_frame.itertuples()
    }
    experiment_rows = [
        _experiment_cost_row(
            experiment,
            pair_cost=pair_lookup.get(_text(experiment.pair_group_key)),
            cost_evidence_id=cost_evidence_id,
        )
        for experiment in experiments.itertuples()
    ]
    experiment_frame = pd.DataFrame(experiment_rows)
    validation = _validation(pairs, pair_frame, experiments, experiment_frame, assets)
    if not validation["status"].eq("PASS").all():
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Current cost evidence validation failed: " + ",".join(failed))

    paths = {
        "assets": active / "current_wizard_hyperliquid_funding_asset_results.csv",
        "pairs": active / "current_wizard_hyperliquid_pair_cost_evidence.csv",
        "experiments": active / "current_wizard_hyperliquid_experiment_cost_readiness.csv",
        "validation": active / "current_wizard_hyperliquid_cost_validation.csv",
        "manifest": active / "current_wizard_hyperliquid_cost_manifest.json",
        "summary_md": active / "current_wizard_hyperliquid_cost_summary.md",
        "snapshot_assets": snapshot_dir / "funding_asset_results.csv",
        "snapshot_pairs": snapshot_dir / "pair_cost_evidence.csv",
        "snapshot_experiments": snapshot_dir / "experiment_cost_readiness.csv",
        "snapshot_validation": snapshot_dir / "validation.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    for frame, active_key, snapshot_key in (
        (assets, "assets", "snapshot_assets"),
        (pair_frame, "pairs", "snapshot_pairs"),
        (experiment_frame, "experiments", "snapshot_experiments"),
        (validation, "validation", "snapshot_validation"),
    ):
        atomic_write_csv(frame, paths[active_key], index=False)
        atomic_write_csv(frame, paths[snapshot_key], index=False)
    pair_counts = pair_frame["cost_evidence_status"].value_counts().to_dict()
    experiment_counts = experiment_frame["cost_replay_status"].value_counts().to_dict()
    summary: dict[str, object] = {
        **material,
        "cost_evidence_id": cost_evidence_id,
        "pair_groups_accounted": int(len(pair_frame)),
        "pair_status_counts": pair_counts,
        "pair_status_accounted": bool(sum(pair_counts.values()) == len(pair_frame)),
        "pairs_ready_for_observed_cost_research": int(
            pair_frame["provisional_cost_research_ready"].astype(bool).sum()
        ),
        "pairs_strict_cost_calibrated": int(
            pair_frame["cost_acceptance_ready"].astype(bool).sum()
        ),
        "experiments_accounted": int(len(experiment_frame)),
        "experiment_status_counts": experiment_counts,
        "funding_assets_complete": int(
            assets.get("funding_status", pd.Series(dtype=str)).eq("COMPLETE").sum()
        ),
        "funding_assets_reused_from_cache": int(
            assets.get("funding_source", pd.Series(dtype=str))
            .eq("reused_point_in_time_cache")
            .sum()
        ),
        "funding_assets_fetched_from_network": int(
            assets.get("funding_source", pd.Series(dtype=str))
            .eq("hyperliquid_public_api")
            .sum()
        ),
        "max_funding_workers": max_funding_workers,
        "strict_l2_window_hours": STRICT_L2_WINDOW_HOURS,
        "provisional_l2_window_hours": PROVISIONAL_L2_WINDOW_HOURS,
        "minimum_strict_l2_samples": MINIMUM_STRICT_L2_SAMPLES,
        "minimum_funding_coverage": MINIMUM_FUNDING_COVERAGE,
        "promotion_authority": False,
        "live_trading_authorized": False,
        "artifacts": {name: _relative(path, root) for name, path in paths.items()},
        "input_snapshots": {
            name: _relative(path, root) for name, path in snapshot_inputs.items()
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _asset_requests(pairs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    requests: dict[str, dict[str, Any]] = {}
    for row in pairs.itertuples():
        cutoff = _parse_timestamp(row.scanner_cutoff_at)
        earliest = _parse_timestamp(row.earliest_candle_at)
        if cutoff is None or earliest is None:
            continue
        days = max(2, int(math.ceil((cutoff - earliest).total_seconds() / 86_400.0)) + 2)
        for asset in (_text(row.asset_a), _text(row.asset_b)):
            prior = requests.get(asset)
            if prior is None:
                requests[asset] = {"cutoff": cutoff, "days": days}
            else:
                prior["cutoff"] = max(prior["cutoff"], cutoff)
                prior["days"] = max(int(prior["days"]), days)
    return requests


def _funding_cache_lookup(
    *,
    asset_requests: dict[str, dict[str, Any]],
    ledger_paths: Iterable[Path],
    root: Path,
) -> dict[str, dict[str, object]]:
    candidates: dict[str, list[dict[str, object]]] = {}
    for ledger_path in ledger_paths:
        try:
            ledger = pd.read_csv(ledger_path, keep_default_na=False)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        required = {"asset", "funding_status", "funding_path"}
        if not required.issubset(ledger.columns):
            continue
        for row in ledger.itertuples(index=False):
            asset = _text(row.asset)
            request = asset_requests.get(asset)
            if request is None or _text(row.funding_status) != "COMPLETE":
                continue
            source_path = root / _text(row.funding_path)
            if not source_path.exists():
                continue
            source_cutoff = _parse_timestamp(
                getattr(row, "fetch_cutoff_at", "")
                or getattr(row, "fetch_end_at", "")
            )
            requested_cutoff = request["cutoff"]
            if source_cutoff is None or source_cutoff > requested_cutoff:
                continue
            candidates.setdefault(asset, []).append(
                {
                    "source_path": source_path,
                    "source_cutoff": source_cutoff,
                    "source_ledger": ledger_path,
                    "source_sha256": _file_hash(source_path),
                }
            )
    return {
        asset: max(rows, key=lambda row: row["source_cutoff"])
        for asset, rows in candidates.items()
    }


def _materialize_funding_asset(
    *,
    asset: str,
    request: dict[str, Any],
    funding_dir: Path,
    root: Path,
    cost_evidence_id: str,
    cache_record: dict[str, object] | None,
    fetch_funding: bool,
    funding_fetcher: Callable[..., Path],
) -> tuple[dict[str, object], pd.DataFrame | None]:
    cutoff = request["cutoff"]
    days = int(request["days"])
    funding_path = funding_dir / f"{asset}_funding.json"
    blocker = ""
    funding_source = ""
    source_path = ""
    source_sha256 = ""
    if cache_record is not None:
        cached_path = Path(cache_record["source_path"])
        atomic_copy_file(cached_path, funding_path, immutable=True)
        funding_source = "reused_point_in_time_cache"
        source_path = _relative(cached_path, root)
        source_sha256 = _text(cache_record.get("source_sha256"))
    elif fetch_funding:
        try:
            fetched = funding_fetcher(
                coin=asset,
                days=days,
                output_dir=funding_dir,
                end_time=cutoff,
            )
            if fetched.resolve() != funding_path.resolve():
                atomic_copy_file(fetched, funding_path, immutable=True)
            funding_source = "hyperliquid_public_api"
            source_path = _relative(fetched, root)
            source_sha256 = _file_hash(fetched)
        except Exception as exc:
            blocker = f"funding_fetch_failed:{safe_exception_code(exc)}"
    elif not funding_path.exists():
        blocker = "funding_fetch_disabled_and_snapshot_missing"
    frame, metadata = _load_bounded_funding(
        funding_path,
        asset=asset,
        cutoff=cutoff,
    )
    if not blocker:
        blocker = _text(metadata.get("blocker"))
    row = {
        "schema_version": SCHEMA_VERSION,
        "cost_evidence_id": cost_evidence_id,
        "asset": asset,
        "fetch_cutoff_at": cutoff.isoformat(),
        "history_days_requested": days,
        "funding_rows": int(metadata["rows"]),
        "earliest_funding_at": metadata["earliest"],
        "latest_funding_at": metadata["latest"],
        "post_cutoff_rows": int(metadata["post_cutoff_rows"]),
        "fetch_complete_flag": bool(metadata["fetch_complete_flag"]),
        "funding_status": "COMPLETE" if not blocker else "BLOCKED",
        "funding_blocker": blocker,
        "funding_source": funding_source,
        "funding_source_path": source_path,
        "funding_source_sha256": source_sha256,
        "funding_path": _relative(funding_path, root) if funding_path.exists() else "",
        "funding_sha256": _file_hash(funding_path) if funding_path.exists() else "",
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    return row, frame if not blocker else None


def _pair_cost_row(
    pair: object,
    *,
    selected: set[str],
    funding_frames: dict[str, pd.DataFrame],
    samples: pd.DataFrame,
    profile: dict[str, object],
    profile_path: Path,
    l2_path: Path,
    enriched_dir: Path,
    root: Path,
    as_of: datetime,
    cost_evidence_id: str,
) -> dict[str, object]:
    pair_key = _text(pair.pair_group_key)
    asset_x = _text(pair.asset_a)
    asset_y = _text(pair.asset_b)
    base = {
        "schema_version": SCHEMA_VERSION,
        "cost_evidence_id": cost_evidence_id,
        "pair_group_key": pair_key,
        "pair": _text(pair.pair),
        "wizard_exchange": _text(pair.wizard_exchange),
        "timeframe": _text(pair.timeframe),
        "hyperliquid_interval": _text(pair.hyperliquid_interval),
        "asset_x": asset_x,
        "asset_y": asset_y,
        "selected_for_cost_evidence": pair_key in selected,
        "funding_x_coverage": 0.0,
        "funding_y_coverage": 0.0,
        "funding_both_coverage": 0.0,
        "funding_longest_contiguous_rows": 0,
        "funding_status": "NOT_EVALUATED",
        "strict_l2_samples_x": 0,
        "strict_l2_samples_y": 0,
        "provisional_l2_samples_x": 0,
        "provisional_l2_samples_y": 0,
        "strict_slippage_ready": False,
        "provisional_slippage_ready": False,
        "slippage_x_p95_bps": "",
        "slippage_y_p95_bps": "",
        "pair_one_way_slippage_bps": "",
        "taker_fee_bps": _safe_float(profile.get("taker_fee_bps")),
        "execution_risk_bps": _safe_float(profile.get("execution_risk_bps")),
        "estimated_pair_round_trip_cost_bps": "",
        "cost_evidence_status": "",
        "cost_blocker": "",
        "provisional_cost_research_ready": False,
        "cost_acceptance_ready": False,
        "enriched_history_path": "",
        "evidence_path": _text(pair.evidence_path),
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    if pair_key not in selected:
        return {**base, "cost_evidence_status": "DEFERRED_NOT_SELECTED", "cost_blocker": "bounded_cost_run_pair_not_selected"}
    if _text(pair.history_status) != "READY_FOR_CANONICAL_1X_REPLAY":
        return {
            **base,
            "cost_evidence_status": "BLOCKED_POINT_IN_TIME_HISTORY",
            "cost_blocker": _text(pair.history_blocker) or "pair_history_not_ready",
        }
    cutoff = _parse_timestamp(pair.scanner_cutoff_at)
    if cutoff is None:
        return {**base, "cost_evidence_status": "BLOCKED_CUTOFF", "cost_blocker": "scanner_cutoff_missing"}
    funding_x = funding_frames.get(asset_x)
    funding_y = funding_frames.get(asset_y)
    blockers: list[str] = []
    if funding_x is None:
        blockers.append("funding_asset_x_unavailable")
    if funding_y is None:
        blockers.append("funding_asset_y_unavailable")
    history_path = root / _text(pair.history_path)
    coverage_x = coverage_y = coverage_both = 0.0
    longest = 0
    enriched_path: Path | None = None
    if funding_x is not None and funding_y is not None:
        try:
            payload = _read_json(history_path)
            history = pd.DataFrame(payload.get("history", []))
            if history.empty or "timestamp" not in history:
                raise ValueError("pair_history_missing_timestamp")
            history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
            if history["timestamp"].isna().any():
                raise ValueError("pair_history_timestamp_invalid")
            frequency = "h" if _text(pair.hyperliquid_interval) == "1h" else "D"
            map_x = _funding_bucket_map(funding_x, frequency=frequency)
            map_y = _funding_bucket_map(funding_y, frequency=frequency)
            buckets = history["timestamp"].dt.floor(frequency)
            aligned_x = buckets.map(map_x)
            aligned_y = buckets.map(map_y)
            history["funding_x_realized_bps"] = aligned_x
            history["funding_y_realized_bps"] = aligned_y
            history["funding_x_bps"] = aligned_x
            history["funding_y_bps"] = aligned_y
            both = aligned_x.notna() & aligned_y.notna()
            coverage_x = float(aligned_x.notna().mean())
            coverage_y = float(aligned_y.notna().mean())
            coverage_both = float(both.mean())
            longest = _longest_contiguous(both, history["timestamp"], frequency)
            payload["history"] = _json_records(history)
            payload["funding_source"] = "hyperliquid_public_funding_history"
            payload["funding_alignment"] = f"same_realized_utc_{frequency}_bucket_no_future_fill"
            payload["cost_evidence_id"] = cost_evidence_id
            enriched_path = enriched_dir / f"{_safe_filename(pair_key)}_funding.json"
            atomic_write_text(enriched_path, json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            blockers.append(f"funding_alignment_failed:{safe_exception_code(exc)}")
    if coverage_x < MINIMUM_FUNDING_COVERAGE:
        blockers.append("insufficient_funding_coverage_asset_x")
    if coverage_y < MINIMUM_FUNDING_COVERAGE:
        blockers.append("insufficient_funding_coverage_asset_y")

    strict_x = _asset_slippage(samples, asset_x, cutoff, STRICT_L2_WINDOW_HOURS)
    strict_y = _asset_slippage(samples, asset_y, cutoff, STRICT_L2_WINDOW_HOURS)
    provisional_x = _asset_slippage(samples, asset_x, cutoff, PROVISIONAL_L2_WINDOW_HOURS)
    provisional_y = _asset_slippage(samples, asset_y, cutoff, PROVISIONAL_L2_WINDOW_HOURS)
    strict_ready = len(strict_x) >= MINIMUM_STRICT_L2_SAMPLES and len(strict_y) >= MINIMUM_STRICT_L2_SAMPLES
    provisional_ready = len(provisional_x) >= MINIMUM_PROVISIONAL_L2_SAMPLES and len(provisional_y) >= MINIMUM_PROVISIONAL_L2_SAMPLES
    chosen_x = strict_x if strict_ready else provisional_x
    chosen_y = strict_y if strict_ready else provisional_y
    p95_x = float(chosen_x["one_way_slippage_bps"].quantile(0.95)) if not chosen_x.empty else None
    p95_y = float(chosen_y["one_way_slippage_bps"].quantile(0.95)) if not chosen_y.empty else None
    pair_slippage = (p95_x + p95_y) / 2.0 if p95_x is not None and p95_y is not None else None
    fee_ready = _fee_profile_ready(profile, cutoff)
    if not fee_ready:
        blockers.append("fee_profile_missing_or_stale")
    if not strict_ready:
        blockers.append("strict_l2_calibration_incomplete")
    if not provisional_ready:
        blockers.append("provisional_l2_evidence_incomplete")
    provisional_research_ready = bool(
        fee_ready and provisional_ready and longest >= MINIMUM_PROVISIONAL_FUNDED_ROWS
    )
    acceptance_ready = bool(
        fee_ready
        and strict_ready
        and coverage_x >= MINIMUM_FUNDING_COVERAGE
        and coverage_y >= MINIMUM_FUNDING_COVERAGE
    )
    if acceptance_ready:
        status = "STRICT_COST_EVIDENCE_READY"
    elif provisional_research_ready:
        status = "PROVISIONAL_OBSERVED_COST_RESEARCH_READY"
    else:
        status = "BLOCKED_COST_EVIDENCE"
    fee = _safe_float(profile.get("taker_fee_bps"))
    execution_risk = _safe_float(profile.get("execution_risk_bps"))
    round_trip = 2.0 * (fee + pair_slippage + execution_risk) if pair_slippage is not None else ""
    evidence = [
        _text(pair.evidence_path),
        _relative(history_path, root),
        _relative(enriched_path, root) if enriched_path else "",
        _relative(profile_path, root),
        _relative(l2_path, root),
    ]
    return {
        **base,
        "funding_x_coverage": coverage_x,
        "funding_y_coverage": coverage_y,
        "funding_both_coverage": coverage_both,
        "funding_longest_contiguous_rows": longest,
        "funding_status": "COMPLETE" if coverage_both > 0 else "BLOCKED",
        "strict_l2_samples_x": len(strict_x),
        "strict_l2_samples_y": len(strict_y),
        "provisional_l2_samples_x": len(provisional_x),
        "provisional_l2_samples_y": len(provisional_y),
        "strict_slippage_ready": strict_ready,
        "provisional_slippage_ready": provisional_ready,
        "slippage_x_p95_bps": p95_x if p95_x is not None else "",
        "slippage_y_p95_bps": p95_y if p95_y is not None else "",
        "pair_one_way_slippage_bps": pair_slippage if pair_slippage is not None else "",
        "estimated_pair_round_trip_cost_bps": round_trip,
        "cost_evidence_status": status,
        "cost_blocker": ";".join(dict.fromkeys(blockers)),
        "provisional_cost_research_ready": provisional_research_ready,
        "cost_acceptance_ready": acceptance_ready,
        "enriched_history_path": _relative(enriched_path, root) if enriched_path else "",
        "evidence_path": ";".join(value for value in evidence if value),
    }


def _experiment_cost_row(
    experiment: object,
    *,
    pair_cost: object | None,
    cost_evidence_id: str,
) -> dict[str, object]:
    base = {
        "schema_version": SCHEMA_VERSION,
        "cost_evidence_id": cost_evidence_id,
        "experiment_id": _text(experiment.experiment_id),
        "pair_group_key": _text(experiment.pair_group_key),
        "pair": _text(experiment.pair),
        "exact_mode": _text(experiment.exact_mode),
        "orientation": _text(experiment.orientation),
        "canonical_replay_status": _text(experiment.replay_status),
        "cost_replay_status": "",
        "cost_replay_blocker": "",
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    if pair_cost is None:
        return {**base, "cost_replay_status": "BLOCKED_PAIR_COST_MISSING", "cost_replay_blocker": "pair_cost_evidence_missing"}
    pair_status = _text(pair_cost.cost_evidence_status)
    if _text(experiment.replay_status) != "RESEARCH_REPLAY_COMPLETE":
        return {
            **base,
            "cost_replay_status": "NOT_RUN_CANONICAL_REPLAY_BLOCKED",
            "cost_replay_blocker": _text(experiment.replay_blocker) or _text(experiment.replay_status),
        }
    if bool(pair_cost.provisional_cost_research_ready):
        return {
            **base,
            "cost_replay_status": "READY_FOR_OBSERVED_COST_RESEARCH",
            "cost_replay_blocker": "" if bool(pair_cost.cost_acceptance_ready) else "strict_cost_acceptance_not_ready",
        }
    return {
        **base,
        "cost_replay_status": "BLOCKED_PAIR_COST_EVIDENCE",
        "cost_replay_blocker": _text(pair_cost.cost_blocker) or pair_status,
    }


def _normalize_l2_samples(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"asset", "notional_usd", "source_timestamp", "one_way_slippage_bps"}
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    work = frame.copy()
    work["asset"] = work["asset"].astype(str).str.upper()
    work["notional_usd"] = pd.to_numeric(work["notional_usd"], errors="coerce")
    work["source_timestamp"] = pd.to_datetime(work["source_timestamp"], utc=True, errors="coerce")
    work["one_way_slippage_bps"] = pd.to_numeric(work["one_way_slippage_bps"], errors="coerce")
    valid = work["source_timestamp"].notna() & work["one_way_slippage_bps"].notna()
    if "blocker" in work:
        valid &= work["blocker"].fillna("").astype(str).eq("")
    for column in ("buy_complete", "sell_complete"):
        if column in work:
            valid &= work[column].map(_truthy)
    return work.loc[valid].drop_duplicates(
        ["asset", "notional_usd", "source_timestamp"], keep="last"
    )


def _asset_slippage(
    samples: pd.DataFrame,
    asset: str,
    cutoff: datetime,
    window_hours: float,
) -> pd.DataFrame:
    if samples.empty:
        return samples
    end = pd.Timestamp(cutoff)
    start = end - pd.Timedelta(hours=window_hours)
    notional = pd.to_numeric(samples["notional_usd"], errors="coerce")
    return samples.loc[
        samples["asset"].eq(asset)
        & notional.sub(LEG_NOTIONAL_USD).abs().le(1e-6)
        & samples["source_timestamp"].between(start, end, inclusive="both")
    ].copy()


def _load_bounded_funding(
    path: Path,
    *,
    asset: str,
    cutoff: datetime,
) -> tuple[pd.DataFrame, dict[str, object]]:
    empty = pd.DataFrame(columns=["timestamp", "funding_bps"])
    metadata = {
        "rows": 0,
        "earliest": "",
        "latest": "",
        "post_cutoff_rows": 0,
        "fetch_complete_flag": False,
        "blocker": "funding_file_missing",
    }
    if not path.exists():
        return empty, metadata
    try:
        payload = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return empty, {**metadata, "blocker": "funding_file_unreadable"}
    normalized = pd.DataFrame(normalize_hyperliquid_funding_history(payload, coin=asset))
    if normalized.empty:
        return empty, {**metadata, "blocker": "normalized_funding_empty"}
    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"], utc=True, errors="coerce", format="mixed"
    )
    normalized["funding_bps"] = pd.to_numeric(normalized["funding_bps"], errors="coerce")
    invalid = int(normalized["timestamp"].isna().sum())
    post_cutoff = int((normalized["timestamp"] > pd.Timestamp(cutoff)).sum())
    bounded = normalized.loc[
        normalized["timestamp"].notna()
        & normalized["funding_bps"].notna()
        & normalized["timestamp"].le(pd.Timestamp(cutoff))
    ].copy()
    complete = bool(payload.get("fetch_complete", False))
    blockers = []
    if invalid:
        blockers.append("funding_timestamp_parse_invalid")
    if post_cutoff:
        blockers.append("funding_contains_post_cutoff_rows")
    if not complete:
        blockers.append("funding_fetch_complete_flag_false")
    if bounded.empty:
        blockers.append("bounded_funding_empty")
    return bounded, {
        "rows": len(bounded),
        "earliest": bounded["timestamp"].min().isoformat() if not bounded.empty else "",
        "latest": bounded["timestamp"].max().isoformat() if not bounded.empty else "",
        "post_cutoff_rows": post_cutoff,
        "fetch_complete_flag": complete,
        "blocker": ";".join(blockers),
    }


def _funding_bucket_map(frame: pd.DataFrame, *, frequency: str) -> dict[pd.Timestamp, float]:
    return (
        frame.assign(bucket=frame["timestamp"].dt.floor(frequency))
        .groupby("bucket")["funding_bps"]
        .sum()
        .to_dict()
    )


def _longest_contiguous(valid: pd.Series, timestamps: pd.Series, frequency: str) -> int:
    expected = pd.Timedelta(hours=1) if frequency == "h" else pd.Timedelta(days=1)
    longest = current = 0
    previous: pd.Timestamp | None = None
    for is_valid, timestamp in zip(valid.tolist(), timestamps.tolist(), strict=True):
        parsed = pd.Timestamp(timestamp)
        contiguous = bool(previous is not None and pd.Timedelta(0) < parsed - previous <= expected * 1.5)
        if not is_valid:
            current = 0
            previous = None
            continue
        current = current + 1 if contiguous else 1
        longest = max(longest, current)
        previous = parsed
    return longest


def _fee_profile_ready(profile: dict[str, object], cutoff: datetime) -> bool:
    checked = _parse_timestamp(profile.get("fee_source_checked_at"))
    return bool(
        _safe_float(profile.get("taker_fee_bps")) > 0
        and _text(profile.get("fee_source_url"))
        and checked is not None
        and checked <= cutoff
        and cutoff - checked <= pd.Timedelta(days=MAXIMUM_FEE_EVIDENCE_AGE_DAYS)
    )


def _validation(
    pairs: pd.DataFrame,
    pair_costs: pd.DataFrame,
    experiments: pd.DataFrame,
    experiment_costs: pd.DataFrame,
    assets: pd.DataFrame,
) -> pd.DataFrame:
    checks = {
        "pair_count_preserved": len(pairs) == len(pair_costs),
        "pair_keys_unique": pair_costs["pair_group_key"].nunique() == len(pair_costs),
        "experiment_count_preserved": len(experiments) == len(experiment_costs),
        "experiment_ids_unique": experiment_costs["experiment_id"].nunique() == len(experiment_costs),
        "pair_status_accounted": pair_costs["cost_evidence_status"].astype(str).ne("").all(),
        "experiment_status_accounted": experiment_costs["cost_replay_status"].astype(str).ne("").all(),
        "funding_has_no_post_cutoff_rows": assets.empty or pd.to_numeric(assets["post_cutoff_rows"], errors="coerce").fillna(1).eq(0).all(),
        "live_trading_disabled": not pair_costs["live_trading_authorized"].astype(bool).any() and not experiment_costs["live_trading_authorized"].astype(bool).any(),
    }
    return pd.DataFrame(
        [{"check": check, "status": "PASS" if passed else "FAIL"} for check, passed in checks.items()]
    )


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    work = frame.copy()
    work["timestamp"] = work["timestamp"].map(
        lambda value: value.isoformat() if pd.notna(value) else None
    )
    return work.where(pd.notna(work), None).to_dict("records")


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Current Wizard Hyperliquid Cost Evidence",
            "",
            f"- Cost evidence: `{summary['cost_evidence_id']}`",
            f"- Pair groups accounted: {summary['pair_groups_accounted']}",
            f"- Experiments accounted: {summary['experiments_accounted']}",
            f"- Funding assets complete: {summary['funding_assets_complete']}",
            f"- Pairs ready for observed-cost research: {summary['pairs_ready_for_observed_cost_research']}",
            f"- Pairs with strict cost calibration: {summary['pairs_strict_cost_calibrated']}",
            "- L2 observations are deduplicated by asset, timestamp, and notional.",
            "- Strict and provisional L2 windows are reported separately.",
            "- Promotion authority: no",
            "- Live trading authorized: no",
            "",
        ]
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parse_timestamp(value: object) -> datetime | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "ready", "complete"}


def _safe_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
