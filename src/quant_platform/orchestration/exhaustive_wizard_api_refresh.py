"""Bridge a complete Crypto Wizards API refresh into the frozen research run.

The API is a discovery surface, not acceptance authority. This module accounts
for every API row, compares current pair membership with the immutable browser
capture, and creates the next pair-detail capture queue without mutating the
frozen exhaustive run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.crypto_wizards_sweep import (
    WIZARD_CRYPTO_EXCHANGES,
    WIZARD_DISCOVERY_INTERVALS,
    WIZARD_DISCOVERY_PRIORITIES,
    WIZARD_DISCOVERY_STRATEGIES,
    build_wizard_sweep_cells,
)
from quant_platform.orchestration.exhaustive_wizard_hyperliquid_run import (
    EXACT_MODES,
    ORIENTATIONS,
)
from quant_platform.wizard_run_config import canonical_wizard_interval
from quant_platform.wizard_symbols import normalize_wizard_exchange, normalize_wizard_symbol


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_api_refresh.v1"
DISCOVERY_POLICY = "exhaustive_no_prefilter"
PAIR_PAGE_EXACT_MODES = tuple(mode for mode in EXACT_MODES if mode != "OU (Optimal)")


def build_exhaustive_wizard_api_refresh_delta(
    *,
    root: Path = ROOT,
    candidates_path: Path | None = None,
    sweep_manifest_path: Path | None = None,
    sweep_summary_path: Path | None = None,
    exhaustive_manifest_path: Path | None = None,
    inventory_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Account for a complete API sweep and compare it with frozen UI evidence."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    candidates_path = candidates_path or active / "wizard_sweep_candidates.csv"
    sweep_manifest_path = sweep_manifest_path or active / "wizard_sweep_manifest.csv"
    sweep_summary_path = sweep_summary_path or active / "wizard_sweep_summary.json"
    exhaustive_manifest_path = (
        exhaustive_manifest_path
        or active / "exhaustive_wizard_hyperliquid_run_manifest.json"
    )
    inventory_path = inventory_path or active / "hyperliquid_testnet_market_inventory.csv"

    candidates = _read_csv_required(candidates_path)
    sweep_manifest = _read_csv_required(sweep_manifest_path)
    sweep_summary = _read_json_required(sweep_summary_path)
    exhaustive_manifest = _read_json_required(exhaustive_manifest_path)
    inventory = _read_csv_required(inventory_path)
    run_id = _text(exhaustive_manifest.get("run_id"))
    if not run_id:
        raise ValueError("Frozen exhaustive run manifest has no run_id")
    frozen_pair_path = _immutable_pair_ledger_path(root, exhaustive_manifest)
    frozen_pairs = _read_csv_required(frozen_pair_path)

    _validate_complete_sweep(candidates, sweep_manifest, sweep_summary, root=root)
    source_accounting = _build_api_source_accounting(candidates)
    api_groups = _build_api_pair_groups(source_accounting)
    frozen_groups = _build_frozen_pair_groups(frozen_pairs)
    delta = _build_delta(api_groups, frozen_groups)
    hyperliquid_mapping = _build_hyperliquid_mapping(delta, inventory)
    pair_detail_queue = _build_pair_detail_queue(delta)
    pair_detail_queue = pair_detail_queue.merge(
        hyperliquid_mapping[
            [
                "pair_group_key",
                "hyperliquid_market_a",
                "hyperliquid_market_b",
                "hyperliquid_pair_ready",
                "hyperliquid_pair_max_leverage",
                "hyperliquid_only_isolated",
                "hyperliquid_inventory_checked_at",
                "hyperliquid_mapping_blocker",
                "hyperliquid_research_status",
            ]
        ],
        on="pair_group_key",
        how="left",
        validate="one_to_one",
    )
    credit_limit = int(sweep_summary.get("credit_limit", 1000) or 1000)
    reserved_credits = int(sweep_summary.get("reserved_credits", 100) or 0)
    credits_used_after_estimate = int(sweep_summary.get("credits_used_before", 0) or 0) + int(
        sweep_summary.get("completed_credits", 0) or 0
    )
    credits_available_after_reserve = max(
        credit_limit - reserved_credits - credits_used_after_estimate,
        0,
    )
    acquisition_plan = _build_pair_detail_acquisition_plan(
        pair_groups=len(api_groups),
        credits_available=credits_available_after_reserve,
    )

    input_hashes = {
        "api_candidates": _file_hash(candidates_path),
        "api_sweep_manifest": _file_hash(sweep_manifest_path),
        "api_sweep_summary": _file_hash(sweep_summary_path),
        "frozen_exhaustive_manifest": _file_hash(exhaustive_manifest_path),
        "frozen_pair_ledger": _file_hash(frozen_pair_path),
        "hyperliquid_testnet_inventory": _file_hash(inventory_path),
    }
    refresh_material = {
        "schema_version": SCHEMA_VERSION,
        "exhaustive_run_id": run_id,
        "sweep_id": _text(sweep_summary.get("sweep_id")),
        "input_hashes": input_hashes,
        "discovery_policy": DISCOVERY_POLICY,
    }
    refresh_id = "ewapi_" + sha256(_canonical_json(refresh_material).encode()).hexdigest()[:20]
    for frame in (
        source_accounting,
        delta,
        hyperliquid_mapping,
        pair_detail_queue,
        acquisition_plan,
    ):
        frame.insert(1, "refresh_id", refresh_id)

    validation = _build_validation(
        source_accounting=source_accounting,
        delta=delta,
        hyperliquid_mapping=hyperliquid_mapping,
        pair_detail_queue=pair_detail_queue,
        acquisition_plan=acquisition_plan,
        candidates=candidates,
        frozen_pairs=frozen_pairs,
        sweep_manifest=sweep_manifest,
    )
    validation.insert(1, "refresh_id", refresh_id)
    failures = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
    if failures:
        raise ValueError("API refresh accounting failed: " + ",".join(failures))

    snapshot_dir = (
        root
        / "reports"
        / "snapshots"
        / "exhaustive_wizard_hyperliquid"
        / run_id
        / "api_refreshes"
        / refresh_id
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "source_accounting": active / "exhaustive_wizard_api_refresh_source_accounting.csv",
        "delta": active / "exhaustive_wizard_api_refresh_delta.csv",
        "hyperliquid_mapping": active
        / "exhaustive_wizard_api_refresh_hyperliquid_mapping.csv",
        "pair_detail_queue": active
        / "exhaustive_wizard_api_refresh_pair_detail_queue.csv",
        "pair_detail_acquisition_plan": active
        / "exhaustive_wizard_pair_detail_acquisition_plan.csv",
        "validation": active / "exhaustive_wizard_api_refresh_validation.csv",
        "manifest": active / "exhaustive_wizard_api_refresh_manifest.json",
        "summary_md": active / "exhaustive_wizard_api_refresh_summary.md",
        "snapshot_source_accounting": snapshot_dir / "source_accounting.csv",
        "snapshot_delta": snapshot_dir / "pair_membership_delta.csv",
        "snapshot_hyperliquid_mapping": snapshot_dir / "hyperliquid_mapping.csv",
        "snapshot_pair_detail_queue": snapshot_dir / "pair_detail_queue.csv",
        "snapshot_pair_detail_acquisition_plan": snapshot_dir
        / "pair_detail_acquisition_plan.csv",
        "snapshot_validation": snapshot_dir / "validation.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
        "snapshot_api_candidates": snapshot_dir / "api_candidates.csv",
        "snapshot_api_sweep_manifest": snapshot_dir / "api_sweep_manifest.csv",
        "snapshot_api_sweep_summary": snapshot_dir / "api_sweep_summary.json",
        "snapshot_frozen_pair_ledger": snapshot_dir / "frozen_pair_ledger.csv",
        "snapshot_hyperliquid_inventory": snapshot_dir / "hyperliquid_inventory.csv",
    }
    for frame, active_key, snapshot_key in (
        (source_accounting, "source_accounting", "snapshot_source_accounting"),
        (delta, "delta", "snapshot_delta"),
        (
            hyperliquid_mapping,
            "hyperliquid_mapping",
            "snapshot_hyperliquid_mapping",
        ),
        (pair_detail_queue, "pair_detail_queue", "snapshot_pair_detail_queue"),
        (
            acquisition_plan,
            "pair_detail_acquisition_plan",
            "snapshot_pair_detail_acquisition_plan",
        ),
        (validation, "validation", "snapshot_validation"),
    ):
        frame.to_csv(paths[active_key], index=False)
        frame.to_csv(paths[snapshot_key], index=False)
    _copy(candidates_path, paths["snapshot_api_candidates"])
    _copy(sweep_manifest_path, paths["snapshot_api_sweep_manifest"])
    _copy(sweep_summary_path, paths["snapshot_api_sweep_summary"])
    _copy(frozen_pair_path, paths["snapshot_frozen_pair_ledger"])
    _copy(inventory_path, paths["snapshot_hyperliquid_inventory"])

    status_counts = delta["membership_status"].value_counts().to_dict()
    mapping_ready = int(hyperliquid_mapping["hyperliquid_pair_ready"].astype(bool).sum())
    current_mapping = hyperliquid_mapping[
        hyperliquid_mapping["in_api_refresh"].astype(bool)
    ]
    current_mapping_ready = int(current_mapping["hyperliquid_pair_ready"].astype(bool).sum())
    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "refresh_id": refresh_id,
        "created_at": as_of.isoformat(),
        "exhaustive_run_id": run_id,
        "sweep_id": _text(sweep_summary.get("sweep_id")),
        "sweep_complete": True,
        "discovery_policy": DISCOVERY_POLICY,
        "discovery_prefilters": {
            "sharpe": None,
            "returns_total": None,
            "liquidity": None,
            "stationarity": None,
            "copula": None,
        },
        "api_source_rows": int(len(candidates)),
        "api_source_rows_accounted": int(len(source_accounting)),
        "api_source_row_ids_unique": int(source_accounting["api_source_row_id"].nunique()),
        "ou_optimal_true_source_rows": int(source_accounting["api_ou_optimal"].astype(bool).sum()),
        "ou_optimal_true_pair_groups": int(
            source_accounting.loc[
                source_accounting["api_ou_optimal"].astype(bool), "pair_group_key"
            ].nunique()
        ),
        "api_pair_groups": int(source_accounting["pair_group_key"].nunique()),
        "frozen_source_rows": int(pd.to_numeric(frozen_pairs["source_row_count"], errors="coerce").fillna(0).sum()),
        "frozen_pair_groups": int(len(frozen_pairs)),
        "union_pair_groups": int(len(delta)),
        "matched_frozen_pairs": int(status_counts.get("MATCHED_FROZEN_PAIR", 0)),
        "new_api_discoveries": int(status_counts.get("NEW_API_DISCOVERY", 0)),
        "missing_from_api_refresh": int(status_counts.get("MISSING_FROM_API_REFRESH", 0)),
        "pair_detail_queue_rows": int(len(pair_detail_queue)),
        "pair_detail_queue_complete": len(pair_detail_queue)
        == int(source_accounting["pair_group_key"].nunique()),
        "hyperliquid_pair_groups_mapped": int(len(hyperliquid_mapping)),
        "hyperliquid_ready_pair_groups": mapping_ready,
        "hyperliquid_blocked_pair_groups": int(len(hyperliquid_mapping) - mapping_ready),
        "current_api_hyperliquid_pair_groups_mapped": int(len(current_mapping)),
        "current_api_hyperliquid_ready_pair_groups": current_mapping_ready,
        "current_api_hyperliquid_blocked_pair_groups": int(
            len(current_mapping) - current_mapping_ready
        ),
        "credit_limit": credit_limit,
        "reserved_credits": reserved_credits,
        "credits_used_after_sweep_estimate": credits_used_after_estimate,
        "credits_available_after_reserve_estimate": credits_available_after_reserve,
        "pair_detail_acquisition_lanes": int(len(acquisition_plan)),
        "pair_detail_browser_route_status": "BLOCKED_CHROME_CONTENT_READ_TIMEOUT",
        "exact_modes_required": list(EXACT_MODES),
        "orientations_required": list(ORIENTATIONS),
        "no_silent_drops": True,
        "discovery_authority": "API_REFRESH_DISCOVERY_ONLY",
        "promotion_authority": False,
        "live_trading_authorized": False,
        "next_step": "capture every queued pair in the authenticated Wizard pair-detail UI",
        "input_hashes": input_hashes,
        "inputs": {
            "api_candidates": _relative(candidates_path, root),
            "api_sweep_manifest": _relative(sweep_manifest_path, root),
            "api_sweep_summary": _relative(sweep_summary_path, root),
            "frozen_exhaustive_manifest": _relative(exhaustive_manifest_path, root),
            "frozen_pair_ledger": _relative(frozen_pair_path, root),
            "hyperliquid_testnet_inventory": _relative(inventory_path, root),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
        "immutable_snapshot": _relative(snapshot_dir, root),
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    paths["manifest"].write_text(manifest_text, encoding="utf-8")
    paths["snapshot_manifest"].write_text(manifest_text, encoding="utf-8")
    paths["summary_md"].write_text(summary_text, encoding="utf-8")
    paths["snapshot_summary_md"].write_text(summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _validate_complete_sweep(
    candidates: pd.DataFrame,
    manifest: pd.DataFrame,
    summary: dict[str, Any],
    *,
    root: Path,
) -> None:
    expected_cells = build_wizard_sweep_cells(
        sweep_id=_text(summary.get("sweep_id")) or "validation",
        exchanges=WIZARD_CRYPTO_EXCHANGES,
        intervals=WIZARD_DISCOVERY_INTERVALS,
        strategies=WIZARD_DISCOVERY_STRATEGIES,
        priorities=WIZARD_DISCOVERY_PRIORITIES,
    )
    expected_request_ids = {cell.request_id for cell in expected_cells}
    required_manifest = {"request_id", "status", "row_count", "evidence_path"}
    required_candidates = {
        "sweep_id",
        "request_id",
        "sweep_exchange",
        "sweep_interval",
        "sweep_strategy",
        "sweep_complete",
        "discovery_authority",
        "symbol_1",
        "symbol_2",
    }
    if not required_manifest.issubset(manifest.columns):
        raise ValueError("Wizard sweep manifest is missing required columns")
    if not required_candidates.issubset(candidates.columns):
        raise ValueError("Wizard sweep candidates are missing required columns")
    actual_request_ids = set(manifest["request_id"].astype(str))
    if actual_request_ids != expected_request_ids or len(manifest) != len(expected_cells):
        raise ValueError("Wizard API refresh does not contain the expected 30 request cells")
    if not manifest["status"].astype(str).eq("completed").all():
        raise ValueError("Wizard API refresh contains an incomplete request cell")
    if not _truthy(summary.get("sweep_complete")):
        raise ValueError("Wizard API refresh summary is not complete")
    if _text(summary.get("discovery_authority")) != "complete_discovery":
        raise ValueError("Wizard API refresh lacks complete discovery authority")
    if int(summary.get("completed_cells", 0) or 0) != len(expected_cells):
        raise ValueError("Wizard API refresh summary has the wrong completed cell count")
    expected_rows = int(pd.to_numeric(manifest["row_count"], errors="coerce").fillna(0).sum())
    if expected_rows != len(candidates) or int(summary.get("candidate_rows", -1)) != len(candidates):
        raise ValueError("Wizard API refresh candidate count does not match its manifest")
    actual_counts = candidates.groupby("request_id", dropna=False).size().to_dict()
    for row in manifest.itertuples():
        if actual_counts.get(str(row.request_id), 0) != int(row.row_count):
            raise ValueError(f"Wizard API request row count mismatch: {row.request_id}")
        evidence = root / _text(row.evidence_path)
        if not evidence.exists():
            raise FileNotFoundError(f"Wizard API raw evidence missing: {evidence}")
    if not candidates["sweep_complete"].map(_truthy).all():
        raise ValueError("Wizard API candidate row is not marked sweep_complete")
    if not candidates["discovery_authority"].astype(str).eq("complete_discovery").all():
        raise ValueError("Wizard API candidate row lacks complete discovery authority")
    sweep_ids = set(candidates["sweep_id"].astype(str))
    if sweep_ids != {_text(summary.get("sweep_id"))}:
        raise ValueError("Wizard API candidate sweep_id does not match the sweep summary")


def _build_api_source_accounting(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_index, (_, candidate) in enumerate(candidates.iterrows()):
        exchange = normalize_wizard_exchange(candidate.get("sweep_exchange")) or ""
        interval = canonical_wizard_interval(candidate.get("sweep_interval")) or ""
        symbol_x = normalize_wizard_symbol(candidate.get("symbol_1"), exchange)
        symbol_y = normalize_wizard_symbol(candidate.get("symbol_2"), exchange)
        asset_x = symbol_x.base_asset or ""
        asset_y = symbol_y.base_asset or ""
        blockers: list[str] = []
        if not exchange:
            blockers.append("wizard_exchange_unresolved")
        if not interval:
            blockers.append("wizard_interval_unresolved")
        if not asset_x:
            blockers.append("wizard_symbol_1_unresolved")
        if not asset_y:
            blockers.append("wizard_symbol_2_unresolved")
        fingerprint = sha256(
            _canonical_json({str(key): _json_value(value) for key, value in candidate.items()}).encode()
        ).hexdigest()
        source_id = "wapirow_" + sha256(
            f"{fingerprint}|{row_index}".encode()
        ).hexdigest()[:20]
        if exchange and interval and asset_x and asset_y:
            assets = sorted((asset_x.upper(), asset_y.upper()))
            pair_group_key = "|".join((exchange, interval, *assets))
        else:
            pair_group_key = f"unresolved|{source_id}"
        exact_mode = _api_exact_mode(candidate)
        api_symbol_1 = _text(candidate.get("symbol_1"))
        api_symbol_2 = _text(candidate.get("symbol_2"))
        api_exchange = _text(candidate.get("sweep_exchange") or candidate.get("exchange"))
        api_interval = _text(candidate.get("sweep_interval") or candidate.get("interval"))
        api_period = _text(candidate.get("period"))
        api_roll_window = _text(candidate.get("zscore_window"))
        request_variant = _canonical_json(
            {
                "symbol_1": api_symbol_1,
                "symbol_2": api_symbol_2,
                "exchange": api_exchange,
                "interval": api_interval,
                "period": api_period,
                "roll_w": api_roll_window,
            }
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "api_source_row_id": source_id,
                "api_source_row_index": row_index,
                "source_row_fingerprint": fingerprint,
                "accounting_status": "ACCOUNTED",
                "discovery_policy": DISCOVERY_POLICY,
                "discovery_prefilter_applied": False,
                "sweep_id": _text(candidate.get("sweep_id")),
                "request_id": _text(candidate.get("request_id")),
                "pair_group_key": pair_group_key,
                "wizard_exchange": exchange,
                "timeframe": interval,
                "asset_x": asset_x,
                "asset_y": asset_y,
                "orientation": f"{asset_x}/{asset_y}" if asset_x and asset_y else "",
                "pair": f"{asset_x}-USD-{asset_y}-USD" if asset_x and asset_y else "",
                "pair_id": _text(candidate.get("pair_id")),
                "spread_id": _text(candidate.get("spread_id")),
                "api_symbol_1": api_symbol_1,
                "api_symbol_2": api_symbol_2,
                "api_exchange": api_exchange,
                "api_interval": api_interval,
                "api_period": api_period,
                "api_roll_window": api_roll_window,
                "api_request_variant": request_variant,
                "requested_strategy_family": _text(candidate.get("sweep_strategy")),
                "returned_strategy": _text(candidate.get("strategy")),
                "spread_type": _text(candidate.get("spread_type")),
                "api_exact_mode": exact_mode,
                "api_ou_optimal": _truthy(candidate.get("ou_optimal")),
                "sharpe": _number(candidate.get("sharpe")),
                "returns_total": _number(candidate.get("returns_total")),
                "sweep_rank": _number(candidate.get("sweep_rank")),
                "sweep_captured_at": _text(candidate.get("sweep_captured_at")),
                "sweep_source_timestamp": _text(candidate.get("sweep_source_timestamp")),
                "evidence_path": _text(candidate.get("sweep_evidence_path")),
                "normalization_blocker": ";".join(blockers),
                "pair_detail_required": True,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(candidates) or frame["api_source_row_id"].nunique() != len(candidates):
        raise ValueError("Wizard API source-row accounting is not one-to-one")
    return frame


def _build_api_pair_groups(source: pd.DataFrame) -> dict[str, dict[str, object]]:
    groups: dict[str, dict[str, object]] = {}
    for key, group in source.groupby("pair_group_key", sort=True, dropna=False):
        first = group.iloc[0]
        assets = sorted(
            value
            for value in {
                _text(first.get("asset_x")).upper(),
                _text(first.get("asset_y")).upper(),
            }
            if value
        )
        groups[str(key)] = {
            "pair_group_key": str(key),
            "wizard_exchange": _text(first.get("wizard_exchange")),
            "timeframe": _text(first.get("timeframe")),
            "asset_a": assets[0] if assets else "",
            "asset_b": assets[1] if len(assets) > 1 else "",
            "api_source_row_count": int(len(group)),
            "api_source_row_ids": _join(group["api_source_row_id"]),
            "api_pair_ids": _join(group["pair_id"]),
            "api_pair_detail_urls": _join(
                pd.Series(
                    [
                        f"https://cryptowizards.net/wizards/zscore/pair/{pair_id}"
                        for pair_id in group["pair_id"].astype(str)
                        if pair_id
                    ]
                )
            ),
            "api_request_variants": " || ".join(
                sorted({_text(value) for value in group["api_request_variant"] if _text(value)})
            ),
            "api_observed_orientations": _join(group["orientation"]),
            "api_requested_strategy_families": _join(group["requested_strategy_family"]),
            "api_returned_strategies": _join(group["returned_strategy"]),
            "api_exact_modes_observed": _join(group["api_exact_mode"]),
            "api_ou_optimal_source_rows": int(group["api_ou_optimal"].astype(bool).sum()),
            "api_ou_optimal_observed_any": bool(group["api_ou_optimal"].astype(bool).any()),
            "api_ou_optimal_exact_modes": _join(
                group.loc[group["api_ou_optimal"].astype(bool), "api_exact_mode"]
            ),
            "api_sharpe_min": _numeric_extreme(group["sharpe"], "min"),
            "api_sharpe_max": _numeric_extreme(group["sharpe"], "max"),
            "api_returns_total_min": _numeric_extreme(group["returns_total"], "min"),
            "api_returns_total_max": _numeric_extreme(group["returns_total"], "max"),
            "api_captured_at": _join(group["sweep_captured_at"]),
            "api_source_timestamps": _join(group["sweep_source_timestamp"]),
            "api_evidence_paths": _join(group["evidence_path"]),
            "normalization_blocker": _join(group["normalization_blocker"]),
        }
    return groups


def _build_frozen_pair_groups(frozen: pd.DataFrame) -> dict[str, dict[str, object]]:
    if "pair_group_key" not in frozen.columns or frozen["pair_group_key"].duplicated().any():
        raise ValueError("Frozen pair ledger must contain one unique pair_group_key per row")
    return {str(row.pair_group_key): row._asdict() for row in frozen.itertuples(index=False)}


def _build_delta(
    api_groups: dict[str, dict[str, object]],
    frozen_groups: dict[str, dict[str, object]],
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "pair_group_key",
        "wizard_exchange",
        "timeframe",
        "asset_a",
        "asset_b",
        "pair",
        "membership_status",
        "in_api_refresh",
        "in_frozen_ui_run",
        "api_source_row_count",
        "api_source_row_ids",
        "api_pair_ids",
        "api_pair_detail_urls",
        "api_request_variants",
        "api_observed_orientations",
        "api_requested_strategy_families",
        "api_returned_strategies",
        "api_exact_modes_observed",
        "api_ou_optimal_source_rows",
        "api_ou_optimal_observed_any",
        "api_ou_optimal_exact_modes",
        "api_sharpe_min",
        "api_sharpe_max",
        "api_returns_total_min",
        "api_returns_total_max",
        "api_captured_at",
        "api_source_timestamps",
        "api_evidence_paths",
        "frozen_source_row_count",
        "frozen_observed_orientations",
        "frozen_sharpe_min",
        "frozen_sharpe_max",
        "frozen_returns_total_min",
        "frozen_returns_total_max",
        "frozen_hyperliquid_pair_ready",
        "frozen_hyperliquid_mapping_blocker",
        "frozen_evidence_paths",
        "normalization_blocker",
        "pair_detail_required",
        "research_queue_status",
        "promotion_authority",
        "live_trading_authorized",
    ]
    rows: list[dict[str, object]] = []
    for key in sorted(set(api_groups) | set(frozen_groups)):
        api = api_groups.get(key, {})
        frozen = frozen_groups.get(key, {})
        in_api = bool(api)
        in_frozen = bool(frozen)
        if in_api and in_frozen:
            membership_status = "MATCHED_FROZEN_PAIR"
        elif in_api:
            membership_status = "NEW_API_DISCOVERY"
        else:
            membership_status = "MISSING_FROM_API_REFRESH"
        key_parts = key.split("|")
        exchange = _text(api.get("wizard_exchange") or frozen.get("wizard_exchange"))
        timeframe = _text(api.get("timeframe") or frozen.get("timeframe"))
        asset_a = _text(api.get("asset_a"))
        asset_b = _text(api.get("asset_b"))
        if not asset_a and len(key_parts) == 4:
            asset_a, asset_b = key_parts[2], key_parts[3]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_group_key": key,
                "wizard_exchange": exchange,
                "timeframe": timeframe,
                "asset_a": asset_a,
                "asset_b": asset_b,
                "pair": f"{asset_a}-USD-{asset_b}-USD" if asset_a and asset_b else "",
                "membership_status": membership_status,
                "in_api_refresh": in_api,
                "in_frozen_ui_run": in_frozen,
                "api_source_row_count": int(api.get("api_source_row_count", 0) or 0),
                "api_source_row_ids": _text(api.get("api_source_row_ids")),
                "api_pair_ids": _text(api.get("api_pair_ids")),
                "api_pair_detail_urls": _text(api.get("api_pair_detail_urls")),
                "api_request_variants": _text(api.get("api_request_variants")),
                "api_observed_orientations": _text(api.get("api_observed_orientations")),
                "api_requested_strategy_families": _text(api.get("api_requested_strategy_families")),
                "api_returned_strategies": _text(api.get("api_returned_strategies")),
                "api_exact_modes_observed": _text(api.get("api_exact_modes_observed")),
                "api_ou_optimal_source_rows": int(
                    api.get("api_ou_optimal_source_rows", 0) or 0
                ),
                "api_ou_optimal_observed_any": bool(
                    api.get("api_ou_optimal_observed_any", False)
                ),
                "api_ou_optimal_exact_modes": _text(
                    api.get("api_ou_optimal_exact_modes")
                ),
                "api_sharpe_min": api.get("api_sharpe_min"),
                "api_sharpe_max": api.get("api_sharpe_max"),
                "api_returns_total_min": api.get("api_returns_total_min"),
                "api_returns_total_max": api.get("api_returns_total_max"),
                "api_captured_at": _text(api.get("api_captured_at")),
                "api_source_timestamps": _text(api.get("api_source_timestamps")),
                "api_evidence_paths": _text(api.get("api_evidence_paths")),
                "frozen_source_row_count": int(float(frozen.get("source_row_count", 0) or 0)),
                "frozen_observed_orientations": _text(frozen.get("observed_orientations")),
                "frozen_sharpe_min": _number(frozen.get("observed_sharpe_min")),
                "frozen_sharpe_max": _number(frozen.get("observed_sharpe_max")),
                "frozen_returns_total_min": _number(frozen.get("observed_returns_total_min")),
                "frozen_returns_total_max": _number(frozen.get("observed_returns_total_max")),
                "frozen_hyperliquid_pair_ready": _truthy(frozen.get("hyperliquid_pair_ready")),
                "frozen_hyperliquid_mapping_blocker": _text(frozen.get("hyperliquid_mapping_blocker")),
                "frozen_evidence_paths": _text(frozen.get("source_evidence_paths")),
                "normalization_blocker": _text(api.get("normalization_blocker")),
                "pair_detail_required": in_api,
                "research_queue_status": (
                    "PAIR_DETAIL_CAPTURE_REQUIRED"
                    if in_api
                    else "HISTORICAL_FROZEN_PAIR_NOT_IN_CURRENT_API_REFRESH"
                ),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_hyperliquid_mapping(delta: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    required = {"asset", "tradable_perp"}
    if not required.issubset(inventory.columns):
        raise ValueError("Hyperliquid inventory is missing asset or tradable_perp")
    inventory_blockers = sorted(
        {
            item.strip()
            for value in inventory.get("fetch_blocker", pd.Series(dtype=object)).tolist()
            for item in _text(value).split(";")
            if item.strip()
        }
    )
    market_index: dict[str, list[dict[str, object]]] = {}
    suffix_index: dict[str, list[dict[str, object]]] = {}
    for _, market_row in inventory.iterrows():
        if not _truthy(market_row.get("tradable_perp")):
            continue
        asset = _text(market_row.get("asset")).upper()
        if not asset:
            continue
        market = {
            "asset": asset,
            "universe_name": _text(market_row.get("universe_name")) or asset,
            "asset_index": _text(market_row.get("asset_index")),
            "max_leverage": _number(market_row.get("max_leverage")),
            "only_isolated": _truthy(market_row.get("only_isolated")),
            "checked_at": _text(market_row.get("checked_at_utc")),
        }
        market_index.setdefault(asset, []).append(market)
        if ":" in asset:
            suffix_index.setdefault(asset.rsplit(":", 1)[-1], []).append(market)
    for suffix, markets in suffix_index.items():
        market_index.setdefault(suffix, markets)

    rows: list[dict[str, object]] = []
    for row in delta.itertuples(index=False):
        asset_a = _text(row.asset_a).upper()
        asset_b = _text(row.asset_b).upper()
        markets_a = market_index.get(asset_a, [])
        markets_b = market_index.get(asset_b, [])
        blockers = list(inventory_blockers)
        normalization_blocker = _text(row.normalization_blocker)
        if normalization_blocker:
            blockers.extend(item for item in normalization_blocker.split(";") if item)
        for label, asset, markets in (
            ("a", asset_a, markets_a),
            ("b", asset_b, markets_b),
        ):
            if not asset:
                blockers.append(f"wizard_asset_{label}_unresolved")
            elif not markets:
                blockers.append(f"missing_hyperliquid_testnet_perp:{asset}")
            elif len(markets) > 1:
                blockers.append(f"ambiguous_hyperliquid_testnet_perp:{asset}")
        blockers = list(dict.fromkeys(blockers))
        ready = bool(len(markets_a) == 1 and len(markets_b) == 1 and not blockers)
        leverage_values = [
            market["max_leverage"]
            for market in [*markets_a, *markets_b]
            if market["max_leverage"] is not None
        ]
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_group_key": row.pair_group_key,
                "wizard_exchange": row.wizard_exchange,
                "timeframe": row.timeframe,
                "asset_a": asset_a,
                "asset_b": asset_b,
                "pair": row.pair,
                "membership_status": row.membership_status,
                "in_api_refresh": bool(row.in_api_refresh),
                "in_frozen_ui_run": bool(row.in_frozen_ui_run),
                "hyperliquid_market_a": _market_labels(markets_a),
                "hyperliquid_market_b": _market_labels(markets_b),
                "hyperliquid_market_a_count": len(markets_a),
                "hyperliquid_market_b_count": len(markets_b),
                "hyperliquid_leg_a_ready": len(markets_a) == 1,
                "hyperliquid_leg_b_ready": len(markets_b) == 1,
                "hyperliquid_pair_ready": ready,
                "hyperliquid_pair_max_leverage": (
                    min(float(value) for value in leverage_values)
                    if len(leverage_values) >= 2
                    else None
                ),
                "hyperliquid_only_isolated": any(
                    bool(market["only_isolated"]) for market in [*markets_a, *markets_b]
                ),
                "hyperliquid_inventory_checked_at": ";".join(
                    sorted(
                        {
                            _text(market["checked_at"])
                            for market in [*markets_a, *markets_b]
                            if _text(market["checked_at"])
                        }
                    )
                ),
                "hyperliquid_mapping_blocker": ";".join(blockers),
                "hyperliquid_research_status": (
                    "READY_FOR_PAIR_DETAIL_AND_HISTORY"
                    if ready
                    else "RETAINED_WITH_EXECUTION_BLOCKER"
                ),
                "canonical_1x_replay_status": "BLOCKED_PENDING_PAIR_DETAIL_AND_HISTORY",
                "leverage_surface_status": "NOT_RUN_UNTIL_1X_REPLAY_PASSES",
                "testnet_status": "NOT_RUN",
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["hyperliquid_pair_ready", "wizard_exchange", "timeframe", "asset_a", "asset_b"],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_pair_detail_queue(delta: pd.DataFrame) -> pd.DataFrame:
    queue = delta[delta["in_api_refresh"].astype(bool)].copy()
    queue["capture_reason"] = queue["membership_status"].map(
        {
            "NEW_API_DISCOVERY": "new_api_pair_requires_authenticated_pair_detail_capture",
            "MATCHED_FROZEN_PAIR": "current_api_pair_requires_fresh_pair_detail_refresh",
        }
    )
    queue["required_exact_modes"] = ";".join(PAIR_PAGE_EXACT_MODES)
    queue["required_scanner_overlays"] = "ou_optimal"
    queue["required_orientations"] = ";".join(ORIENTATIONS)
    queue["pair_detail_capture_status"] = "NOT_CAPTURED_FOR_CURRENT_API_REFRESH"
    queue["acceptance_status"] = "BLOCKED_PENDING_PAIR_DETAIL_AND_LOCAL_REPLAY"
    queue["promotion_authority"] = False
    queue["live_trading_authorized"] = False
    queue = queue.sort_values(
        ["wizard_exchange", "timeframe", "asset_a", "asset_b"], kind="mergesort"
    ).reset_index(drop=True)
    queue.insert(2, "queue_position", range(1, len(queue) + 1))
    return queue


def _build_pair_detail_acquisition_plan(
    *,
    pair_groups: int,
    credits_available: int,
) -> pd.DataFrame:
    lane_specs = [
        {
            "lane": "authenticated_dashboard_pair_page",
            "method": "browser_ui",
            "request_bundle": "all_pair_page_modes_and_both_orientations",
            "credits_per_pair": 0,
            "exact_mode_coverage": "complete_if_capture_succeeds",
            "ecm_coverage": "dashboard_ecm_x_ecm_y_ecm_strength",
            "dashboard_settings_coverage": "complete_visible_controls_and_backtest_overrides",
            "wizard_venue_data": True,
            "hyperliquid_data": False,
            "authority": "vendor_pair_detail_evidence_after_capture",
            "operational_status": "BLOCKED_CHROME_CONTENT_READ_TIMEOUT",
            "recommended_use": "primary_complete_capture_lane_after_browser_recovery",
        },
        {
            "lane": "documented_get_full_bundle",
            "method": "wizard_api_get",
            "request_bundle": "spread_3x2+zscores_3x2+backtest_7x2+cointegration_2+copula_2+correlations_2",
            "credits_per_pair": 174,
            "exact_mode_coverage": "documented_analytics_and_backtests",
            "ecm_coverage": "not_exposed",
            "dashboard_settings_coverage": "request_settings_only",
            "wizard_venue_data": True,
            "hyperliquid_data": False,
            "authority": "api_discovery_and_vendor_backtest_evidence_only",
            "operational_status": "PLANNED_NOT_EXECUTED",
            "recommended_use": "bounded_parity_pilots_only",
        },
        {
            "lane": "documented_get_nonredundant_bundle",
            "method": "wizard_api_get",
            "request_bundle": "zscores_3x2+backtest_7x2+cointegration_2+copula_2+correlations_2",
            "credits_per_pair": 144,
            "exact_mode_coverage": "documented_analytics_and_backtests_without_separate_spread_calls",
            "ecm_coverage": "not_exposed",
            "dashboard_settings_coverage": "request_settings_only",
            "wizard_venue_data": True,
            "hyperliquid_data": False,
            "authority": "api_discovery_and_vendor_backtest_evidence_only",
            "operational_status": "PLANNED_NOT_EXECUTED",
            "recommended_use": "lowest_cost_vendor_exact_mode_pilot",
        },
        {
            "lane": "documented_get_original_orientation_pilot",
            "method": "wizard_api_get",
            "request_bundle": "spread_1+zscores_1+backtest_1+cointegration_1+copula_1+correlations_1",
            "credits_per_pair": 31,
            "exact_mode_coverage": "single_mode_single_orientation_pilot",
            "ecm_coverage": "not_exposed",
            "dashboard_settings_coverage": "request_settings_only",
            "wizard_venue_data": True,
            "hyperliquid_data": False,
            "authority": "api_schema_probe_only",
            "operational_status": "PLANNED_NOT_EXECUTED",
            "recommended_use": "schema_and_parity_probe_not_exhaustive_capture",
        },
        {
            "lane": "custom_series_post_full_bundle",
            "method": "wizard_api_post",
            "request_bundle": "spread_3x2+zscores_3x2+backtest_7x2+cointegration_2+copula_2+correlations_2",
            "credits_per_pair": 46,
            "exact_mode_coverage": "documented_custom_series_analytics_and_backtests",
            "ecm_coverage": "not_exposed",
            "dashboard_settings_coverage": "caller_supplied_settings_only",
            "wizard_venue_data": False,
            "hyperliquid_data": True,
            "authority": "local_hyperliquid_vendor_math_comparison_only",
            "operational_status": "PLANNED_NOT_EXECUTED",
            "recommended_use": "selected_math_parity_tests_after_history_is_frozen",
        },
        {
            "lane": "local_hyperliquid_recomputation",
            "method": "local_point_in_time",
            "request_bundle": "all_supported_local_exact_modes_and_orientations",
            "credits_per_pair": 0,
            "exact_mode_coverage": "local_implementation_not_vendor_identity",
            "ecm_coverage": "local_ecm_only",
            "dashboard_settings_coverage": "not_vendor_settings",
            "wizard_venue_data": False,
            "hyperliquid_data": True,
            "authority": "local_acceptance_evidence_after_validation",
            "operational_status": "AVAILABLE_AFTER_HYPERLIQUID_HISTORY_MAPPING",
            "recommended_use": "canonical_acceptance_lane_never_label_as_wizard_parity",
        },
    ]
    rows: list[dict[str, object]] = []
    for spec in lane_specs:
        per_pair = int(spec["credits_per_pair"])
        credit_capacity = pair_groups if per_pair == 0 else min(pair_groups, credits_available // per_pair)
        operational_capacity = credit_capacity
        if spec["operational_status"] == "BLOCKED_CHROME_CONTENT_READ_TIMEOUT":
            operational_capacity = 0
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                **spec,
                "current_pair_groups": pair_groups,
                "full_queue_credits": per_pair * pair_groups,
                "credits_available_after_reserve_estimate": credits_available,
                "credit_capacity_pairs_today": credit_capacity,
                "operational_capacity_pairs_now": operational_capacity,
                "execute_now": False,
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _build_validation(
    *,
    source_accounting: pd.DataFrame,
    delta: pd.DataFrame,
    hyperliquid_mapping: pd.DataFrame,
    pair_detail_queue: pd.DataFrame,
    acquisition_plan: pd.DataFrame,
    candidates: pd.DataFrame,
    frozen_pairs: pd.DataFrame,
    sweep_manifest: pd.DataFrame,
) -> pd.DataFrame:
    api_groups = int(source_accounting["pair_group_key"].nunique())
    frozen_groups = int(frozen_pairs["pair_group_key"].nunique())
    expected_union = len(
        set(source_accounting["pair_group_key"].astype(str))
        | set(frozen_pairs["pair_group_key"].astype(str))
    )
    mapping_blockers_complete = (
        hyperliquid_mapping.loc[
            ~hyperliquid_mapping["hyperliquid_pair_ready"].astype(bool),
            "hyperliquid_mapping_blocker",
        ]
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )
    checks = [
        ("complete_30_cell_api_sweep", len(sweep_manifest) == 30 and sweep_manifest["status"].astype(str).eq("completed").all(), f"cells={len(sweep_manifest)}"),
        ("every_api_source_row_accounted", len(source_accounting) == len(candidates), f"accounted={len(source_accounting)} expected={len(candidates)}"),
        ("api_source_row_ids_unique", source_accounting["api_source_row_id"].nunique() == len(candidates), f"unique={source_accounting['api_source_row_id'].nunique()} expected={len(candidates)}"),
        ("every_frozen_pair_group_accounted", frozen_groups == len(frozen_pairs), f"groups={frozen_groups} rows={len(frozen_pairs)}"),
        ("union_pair_membership_complete", len(delta) == expected_union, f"union={len(delta)} expected={expected_union}"),
        ("current_pair_detail_queue_complete", len(pair_detail_queue) == api_groups, f"queue={len(pair_detail_queue)} api_groups={api_groups}"),
        ("every_union_pair_mapped_to_hyperliquid", len(hyperliquid_mapping) == expected_union, f"mapped={len(hyperliquid_mapping)} union={expected_union}"),
        ("every_unavailable_hyperliquid_pair_has_blocker", mapping_blockers_complete, f"blocked={int((~hyperliquid_mapping['hyperliquid_pair_ready'].astype(bool)).sum())}"),
        ("pair_detail_acquisition_plan_is_nonexecuting", not acquisition_plan["execute_now"].astype(bool).any(), f"lanes={len(acquisition_plan)}"),
        ("no_discovery_prefilter", not source_accounting["discovery_prefilter_applied"].astype(bool).any(), "all rows retained"),
        ("no_promotion_authority", not delta["promotion_authority"].astype(bool).any() and not pair_detail_queue["promotion_authority"].astype(bool).any(), "discovery only"),
        ("no_live_trading_authority", not delta["live_trading_authorized"].astype(bool).any() and not pair_detail_queue["live_trading_authorized"].astype(bool).any(), "live disabled"),
    ]
    return pd.DataFrame(
        [
            {
                "schema_version": SCHEMA_VERSION,
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
            }
            for check, passed, evidence in checks
        ]
    )


def _api_exact_mode(row: pd.Series) -> str:
    family = _text(row.get("sweep_strategy")).lower()
    if family == "copula":
        return "Copula"
    spread = _text(row.get("spread_type")).lower()
    spread_label = {"static": "Static", "dynamic": "Dyn", "dyn": "Dyn", "ou": "OU"}.get(
        spread, spread.title() or "Unknown"
    )
    strategy_label = "ZScoreR" if family in {"zscoreroll", "zscore_roll"} else "Spread"
    return f"{spread_label} ({strategy_label})"


def _immutable_pair_ledger_path(root: Path, manifest: dict[str, Any]) -> Path:
    immutable = manifest.get("immutable_snapshot", {})
    artifacts = immutable.get("artifacts", {}) if isinstance(immutable, dict) else {}
    relative = _text(artifacts.get("pair_ledger"))
    if not relative:
        raise ValueError("Frozen exhaustive run manifest has no immutable pair ledger")
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Frozen immutable pair ledger missing: {path}")
    return path


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard API Refresh",
            "",
            f"- Refresh: `{summary['refresh_id']}`",
            f"- API sweep: `{summary['sweep_id']}`",
            f"- Frozen exhaustive run: `{summary['exhaustive_run_id']}`",
            f"- API rows accounted: {summary['api_source_rows_accounted']} / {summary['api_source_rows']}",
            f"- Current API pair groups: {summary['api_pair_groups']}",
            f"- Frozen UI pair groups: {summary['frozen_pair_groups']}",
            f"- Union pair groups retained: {summary['union_pair_groups']}",
            f"- Matched frozen pairs: {summary['matched_frozen_pairs']}",
            f"- New API discoveries: {summary['new_api_discoveries']}",
            f"- Frozen pairs missing from current API refresh: {summary['missing_from_api_refresh']}",
            f"- Current pair-detail queue: {summary['pair_detail_queue_rows']}",
            f"- Hyperliquid-ready union pair groups: {summary['hyperliquid_ready_pair_groups']}",
            f"- Hyperliquid-blocked union pair groups: {summary['hyperliquid_blocked_pair_groups']}",
            f"- Hyperliquid-ready current API pair groups: {summary['current_api_hyperliquid_ready_pair_groups']}",
            f"- Estimated credits remaining after reserve: {summary['credits_available_after_reserve_estimate']}",
            f"- Pair-detail acquisition lanes: {summary['pair_detail_acquisition_lanes']}",
            f"- Browser pair-detail route: `{summary['pair_detail_browser_route_status']}`",
            f"- No silent drops: `{str(summary['no_silent_drops']).lower()}`",
            "- Promotion authority: `false`",
            "- Live trading authorized: `false`",
            "",
            "The API refresh is discovery evidence only. Every current pair must still receive authenticated pair-detail capture and local Hyperliquid validation.",
            "",
        ]
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    if frame.empty:
        raise ValueError(f"Required CSV is empty: {path}")
    return frame


def _read_json_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Required JSON object is invalid: {path}")
    return payload


def _join(values: pd.Series) -> str:
    return ";".join(sorted({_text(value) for value in values if _text(value)}))


def _market_labels(markets: list[dict[str, object]]) -> str:
    return ";".join(
        f"{_text(market.get('universe_name'))}"
        + (f"#{_text(market.get('asset_index'))}" if _text(market.get("asset_index")) else "")
        for market in markets
    )


def _numeric_extreme(values: pd.Series, method: str) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    result = numeric.min() if method == "min" else numeric.max()
    return float(result)


def _number(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "complete"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def _json_value(value: object) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
