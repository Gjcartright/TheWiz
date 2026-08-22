"""Exhaustive Crypto Wizards to Hyperliquid research-run contract.

This module deliberately separates intake coverage from strategy acceptance.
Every source row is retained, no Sharpe/return filter is applied, and every
derived pair remains visible even when Hyperliquid cannot execute one or both
legs. Later replay and testnet stages consume this immutable work definition.
"""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import atomic_write_text

from quant_platform.orchestration.corrective_runtime import atomic_write_csv

from quant_platform.orchestration.corrective_runtime import atomic_write_bytes

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable

import pandas as pd

from quant_platform.crypto_wizards_dashboard_capture import (
    EXPECTED_CRYPTO_VENUES,
    EXPECTED_TIMEFRAMES,
)
from quant_platform.economic_contract import CANONICAL_WIZARD_MODES
from quant_platform.runtime_types import CommandResult
from quant_platform.wizard_symbols import normalize_wizard_exchange, normalize_wizard_symbol

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_run.v4"
MAPPING_REFRESH_SCHEMA_VERSION = "exhaustive_wizard_hyperliquid_mapping_refresh.v2"
DISCOVERY_POLICY = "exhaustive_no_prefilter"
ORIENTATION_POLICY = "both_orientations_for_all_modes"
EXACT_MODES = CANONICAL_WIZARD_MODES
ORIENTATIONS = ("original", "reverse")
PAIR_DETAIL_REQUIRED_FIELDS = (
    "pair_detail_url_or_id",
    "pair_detail_capture_timestamp",
    "dashboard_exchange",
    "timeframe",
    "lookback_or_period",
    "exact_mode",
    "orientation",
    "spread_series",
    "zscore_series",
    "entry_thresholds",
    "exit_thresholds",
    "hedge_ratio_or_weights",
    "wizard_cost_assumptions",
    "backtest_trade_count",
    "backtest_return",
    "backtest_sharpe",
    "backtest_drawdown",
    "pearson_returns",
    "spearman_returns",
    "kendall_returns",
    "conditional_dependency_values",
    "copula_family",
    "copula_tail_or_arbitrage_fields",
    "ecm_x",
    "ecm_y",
    "ecm_strength",
    "johansen_state_and_statistic",
    "engle_granger_state_and_statistic",
    "hurst",
    "half_life",
    "volume_and_liquidity_fields",
    "raw_export_or_payload",
)


@dataclass(frozen=True)
class HyperliquidMarket:
    asset: str
    universe_name: str
    asset_index: str
    max_leverage: float | None
    only_isolated: bool
    checked_at: str

    @property
    def label(self) -> str:
        suffix = f"#{self.asset_index}" if self.asset_index else ""
        return f"{self.universe_name or self.asset}{suffix}"


def build_exhaustive_wizard_hyperliquid_run(
    *,
    root: Path = ROOT,
    source_path: Path | None = None,
    inventory_path: Path | None = None,
    sweep_manifest_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Build an exhaustive, non-trading work definition from one Wizard snapshot."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        dashboard_source = active / "exhaustive_wizard_dashboard_rows.csv"
        source_path = (
            dashboard_source
            if dashboard_source.exists()
            else active / "wizard_sweep_candidates.csv"
        )
    inventory_path = inventory_path or active / "hyperliquid_testnet_market_inventory.csv"
    if sweep_manifest_path is None:
        dashboard_manifest = active / "exhaustive_wizard_dashboard_capture_manifest.csv"
        sweep_manifest_path = (
            dashboard_manifest
            if source_path.name == "exhaustive_wizard_dashboard_rows.csv"
            and dashboard_manifest.exists()
            else active / "wizard_sweep_manifest.csv"
        )

    source = _read_csv(source_path)
    inventory = _read_csv(inventory_path)
    sweep_manifest = _read_csv(sweep_manifest_path)
    source_hash = _file_hash(source_path)
    inventory_hash = _file_hash(inventory_path)
    run_material = {
        "schema_version": SCHEMA_VERSION,
        "source_hash": source_hash,
        "inventory_hash": inventory_hash,
        "exact_modes": EXACT_MODES,
        "orientations": ORIENTATIONS,
        "discovery_policy": DISCOVERY_POLICY,
    }
    run_hash = sha256(_canonical_json(run_material).encode("utf-8")).hexdigest()
    run_id = f"ewhl_{run_hash[:20]}"

    source_ledger = _build_source_ledger(source, run_id=run_id)
    market_index, inventory_blockers = _market_index(inventory)
    pair_ledger = _build_pair_ledger(
        source_ledger,
        run_id=run_id,
        market_index=market_index,
        inventory_blockers=inventory_blockers,
    )
    pair_detail_queue = _build_pair_detail_capture_queue(
        pair_ledger,
        source_ledger,
        run_id=run_id,
    )
    experiment_matrix = _build_experiment_matrix(pair_ledger, run_id=run_id)

    all_pages_proven = _all_pages_proven(source)
    sweep_complete = _sweep_complete(source, sweep_manifest)
    validation = _build_validation(
        source=source,
        source_ledger=source_ledger,
        pair_ledger=pair_ledger,
        pair_detail_queue=pair_detail_queue,
        experiment_matrix=experiment_matrix,
        source_path=source_path,
        inventory_path=inventory_path,
        sweep_complete=sweep_complete,
        all_pages_proven=all_pages_proven,
    )
    hard_failures = validation.loc[validation["status"].eq("FAIL"), "check"].tolist()
    blockers = validation.loc[validation["status"].eq("BLOCKED"), "reason"].tolist()
    if hard_failures:
        authority = "INVALID_RUN_DEFINITION"
    elif not all_pages_proven:
        authority = "PARTIAL_CAPTURE_RESEARCH_ONLY"
    else:
        authority = "EXHAUSTIVE_RESEARCH_DEFINITION_READY"

    paths = {
        "manifest": active / "exhaustive_wizard_hyperliquid_run_manifest.json",
        "source_row_ledger": active / "exhaustive_wizard_source_row_ledger.csv",
        "pair_ledger": active / "exhaustive_wizard_pair_ledger.csv",
        "pair_detail_capture_queue": active / "exhaustive_wizard_pair_detail_capture_queue.csv",
        "experiment_matrix": active / "exhaustive_wizard_experiment_matrix.csv",
        "coverage_validation": active / "exhaustive_wizard_coverage_validation.csv",
        "summary_md": active / "exhaustive_wizard_hyperliquid_run_summary.md",
    }
    snapshot_dir = root / "reports" / "snapshots" / "exhaustive_wizard_hyperliquid" / run_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_paths = {
        "manifest": snapshot_dir / "manifest.json",
        "source": snapshot_dir / "source.csv",
        "sweep_manifest": snapshot_dir / "sweep_manifest.csv",
        "hyperliquid_inventory": snapshot_dir / "hyperliquid_inventory.csv",
        "source_row_ledger": snapshot_dir / "source_row_ledger.csv",
        "pair_ledger": snapshot_dir / "pair_ledger.csv",
        "pair_detail_capture_queue": snapshot_dir / "pair_detail_capture_queue.csv",
        "experiment_matrix": snapshot_dir / "experiment_matrix.csv",
        "coverage_validation": snapshot_dir / "coverage_validation.csv",
        "summary_md": snapshot_dir / "summary.md",
    }
    output_frames = {
        "source_row_ledger": source_ledger,
        "pair_ledger": pair_ledger,
        "pair_detail_capture_queue": pair_detail_queue,
        "experiment_matrix": experiment_matrix,
        "coverage_validation": validation,
    }
    for key, frame in output_frames.items():
        atomic_write_csv(frame, paths[key], index=False)
        atomic_write_csv(frame, snapshot_paths[key], index=False)
    _copy_snapshot_input(source_path, snapshot_paths["source"])
    _copy_snapshot_input(sweep_manifest_path, snapshot_paths["sweep_manifest"])
    _copy_snapshot_input(inventory_path, snapshot_paths["hyperliquid_inventory"])
    paths["snapshot_manifest"] = snapshot_paths["manifest"]
    paths["snapshot_summary_md"] = snapshot_paths["summary_md"]

    summary: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": as_of.isoformat(),
        "authority": authority,
        "discovery_policy": DISCOVERY_POLICY,
        "discovery_prefilters": {
            "sharpe": None,
            "returns_total": None,
            "trade_count": None,
            "liquidity": None,
            "stationarity": None,
            "copula": None,
        },
        "source_rows": int(len(source)),
        "accounted_source_rows": int(len(source_ledger)),
        "unique_source_row_ids": int(source_ledger["source_row_id"].nunique())
        if not source_ledger.empty
        else 0,
        "pair_groups": int(len(pair_ledger)),
        "pair_detail_capture_queue_rows": int(len(pair_detail_queue)),
        "pair_detail_capture_status": "NOT_CAPTURED",
        "hyperliquid_ready_pair_groups": int(pair_ledger["hyperliquid_pair_ready"].sum())
        if not pair_ledger.empty
        else 0,
        "hyperliquid_blocked_pair_groups": int((~pair_ledger["hyperliquid_pair_ready"]).sum())
        if not pair_ledger.empty
        else 0,
        "exact_modes": list(EXACT_MODES),
        "orientation_policy": ORIENTATION_POLICY,
        "orientations": list(ORIENTATIONS),
        "planned_experiments": int(len(experiment_matrix)),
        "canonical_replay_leverage": 1.0,
        "leverage_and_margin_surface_status": "NOT_RUN_SEPARATE_STAGE",
        "testnet_validation_status": "NOT_RUN",
        "sweep_cells_complete": sweep_complete,
        "all_dashboard_pages_proven": all_pages_proven,
        "blockers": list(dict.fromkeys(str(value) for value in blockers if str(value))),
        "hard_failures": hard_failures,
        "live_trading_authorized": False,
        "source": {
            "path": _relative(source_path, root),
            "sha256": source_hash,
            "source_type": (
                "live_dashboard_exhaustive_capture"
                if source_path.name == "exhaustive_wizard_dashboard_rows.csv"
                else "prescanned_top_opportunities_api"
            ),
        },
        "hyperliquid_inventory": {
            "path": _relative(inventory_path, root),
            "sha256": inventory_hash,
        },
        "sweep_manifest": {
            "path": _relative(sweep_manifest_path, root),
            "sha256": _file_hash(sweep_manifest_path),
        },
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
        "immutable_snapshot": {
            "directory": _relative(snapshot_dir, root),
            "source": _relative(snapshot_paths["source"], root),
            "sweep_manifest": _relative(snapshot_paths["sweep_manifest"], root),
            "hyperliquid_inventory": _relative(snapshot_paths["hyperliquid_inventory"], root),
            "artifacts": {key: _relative(path, root) for key, path in snapshot_paths.items()},
        },
    }
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    summary_text = _summary_markdown(summary)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(snapshot_paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(snapshot_paths["summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def build_exhaustive_wizard_hyperliquid_mapping_refresh(
    *,
    root: Path = ROOT,
    inventory_path: Path | None = None,
    manifest_path: Path | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Re-map a frozen exhaustive run against current Testnet market rules."""

    as_of = _as_utc(now or datetime.now(timezone.utc))
    active = root / "reports" / "active"
    manifest_path = manifest_path or active / "exhaustive_wizard_hyperliquid_run_manifest.json"
    inventory_path = inventory_path or active / "hyperliquid_testnet_market_inventory.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Exhaustive run manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = _text(manifest.get("run_id"))
    immutable = manifest.get("immutable_snapshot", {})
    immutable_artifacts = immutable.get("artifacts", {}) if isinstance(immutable, dict) else {}
    pair_path = root / _text(immutable_artifacts.get("pair_ledger"))
    baseline_inventory_path = root / _text(immutable.get("hyperliquid_inventory"))
    if not run_id or not pair_path.exists() or not baseline_inventory_path.exists():
        raise ValueError("Exhaustive run manifest does not reference a complete immutable snapshot")
    if not inventory_path.exists():
        raise FileNotFoundError(f"Current Hyperliquid inventory not found: {inventory_path}")

    pairs = _read_csv(pair_path)
    baseline_inventory = _read_csv(baseline_inventory_path)
    current_inventory = _read_csv(inventory_path)
    baseline_index, baseline_inventory_blockers = _market_index(baseline_inventory)
    current_index, current_inventory_blockers = _market_index(current_inventory)
    baseline_hash = _file_hash(baseline_inventory_path)
    current_hash = _file_hash(inventory_path)
    refresh_material = {
        "schema_version": MAPPING_REFRESH_SCHEMA_VERSION,
        "run_id": run_id,
        "baseline_inventory_hash": baseline_hash,
        "current_inventory_hash": current_hash,
    }
    refresh_hash = sha256(_canonical_json(refresh_material).encode("utf-8")).hexdigest()
    refresh_id = f"hlmap_{refresh_hash[:20]}"
    snapshot_dir = root / _text(immutable.get("directory")) / "mapping_refreshes" / refresh_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    paths = {
        "mapping": active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv",
        "inventory_drift": active / "exhaustive_wizard_hyperliquid_inventory_drift.csv",
        "manifest": active / "exhaustive_wizard_hyperliquid_mapping_refresh_manifest.json",
        "summary_md": active / "exhaustive_wizard_hyperliquid_mapping_refresh_summary.md",
        "snapshot_mapping": snapshot_dir / "pair_mapping.csv",
        "snapshot_inventory_drift": snapshot_dir / "inventory_drift.csv",
        "snapshot_inventory": snapshot_dir / "current_inventory.csv",
        "snapshot_manifest": snapshot_dir / "manifest.json",
        "snapshot_summary_md": snapshot_dir / "summary.md",
    }
    mapping = _build_mapping_refresh_rows(
        pairs,
        run_id=run_id,
        refresh_id=refresh_id,
        baseline_index=baseline_index,
        current_index=current_index,
        baseline_inventory_blockers=baseline_inventory_blockers,
        current_inventory_blockers=current_inventory_blockers,
        baseline_evidence_path=_relative(baseline_inventory_path, root),
        current_evidence_path=_relative(paths["snapshot_inventory"], root),
    )
    inventory_drift = _build_inventory_drift_rows(
        baseline_inventory,
        current_inventory,
        run_id=run_id,
        refresh_id=refresh_id,
        baseline_evidence_path=_relative(baseline_inventory_path, root),
        current_evidence_path=_relative(paths["snapshot_inventory"], root),
    )
    atomic_write_csv(mapping, paths["mapping"], index=False)
    atomic_write_csv(mapping, paths["snapshot_mapping"], index=False)
    atomic_write_csv(inventory_drift, paths["inventory_drift"], index=False)
    atomic_write_csv(inventory_drift, paths["snapshot_inventory_drift"], index=False)
    _copy_snapshot_input(inventory_path, paths["snapshot_inventory"])

    changed_inventory = int(inventory_drift["drift_status"].ne("UNCHANGED").sum())
    mapping_drift = int(mapping["mapping_drift"].astype(bool).sum())
    ready = int(mapping["current_pair_ready"].astype(bool).sum())
    blocked = int((~mapping["current_pair_ready"].astype(bool)).sum())
    summary: dict[str, object] = {
        "schema_version": MAPPING_REFRESH_SCHEMA_VERSION,
        "run_id": run_id,
        "mapping_refresh_id": refresh_id,
        "created_at": as_of.isoformat(),
        "pair_groups": int(len(mapping)),
        "current_ready_pair_groups": ready,
        "current_blocked_pair_groups": blocked,
        "mapping_drift_pair_groups": mapping_drift,
        "inventory_rows": int(len(inventory_drift)),
        "inventory_drift_rows": changed_inventory,
        "baseline_inventory_sha256": baseline_hash,
        "current_inventory_sha256": current_hash,
        "baseline_inventory_path": _relative(baseline_inventory_path, root),
        "current_inventory_path": _relative(paths["snapshot_inventory"], root),
        "authority": "POINT_IN_TIME_TESTNET_MAPPING_EVIDENCE",
        "live_trading_authorized": False,
        "artifacts": {key: _relative(path, root) for key, path in paths.items()},
    }
    summary_text = _mapping_refresh_markdown(summary)
    manifest_text = json.dumps(summary, indent=2, sort_keys=True)
    atomic_write_text(paths["manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_manifest"], manifest_text, encoding="utf-8")
    atomic_write_text(paths["summary_md"], summary_text, encoding="utf-8")
    atomic_write_text(paths["snapshot_summary_md"], summary_text, encoding="utf-8")
    return CommandResult(paths=paths, summary=summary)


def _build_mapping_refresh_rows(
    pairs: pd.DataFrame,
    *,
    run_id: str,
    refresh_id: str,
    baseline_index: dict[str, list[HyperliquidMarket]],
    current_index: dict[str, list[HyperliquidMarket]],
    baseline_inventory_blockers: list[str],
    current_inventory_blockers: list[str],
    baseline_evidence_path: str,
    current_evidence_path: str,
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "exhaustive_run_id",
        "mapping_refresh_id",
        "pair_group_id",
        "pair_group_key",
        "wizard_exchange",
        "timeframe",
        "pair",
        "asset_x",
        "asset_y",
        "baseline_pair_ready",
        "current_pair_ready",
        "baseline_market_x",
        "current_market_x",
        "baseline_market_y",
        "current_market_y",
        "baseline_pair_max_leverage",
        "current_pair_max_leverage",
        "baseline_only_isolated",
        "current_only_isolated",
        "baseline_mapping_blocker",
        "current_mapping_blocker",
        "current_inventory_checked_at",
        "mapping_drift",
        "mapping_drift_reasons",
        "mapping_status",
        "baseline_evidence_path",
        "current_evidence_path",
        "live_trading_authorized",
    ]
    rows: list[dict[str, object]] = []
    for pair in pairs.itertuples():
        baseline = _pair_market_state(
            str(pair.asset_x),
            str(pair.asset_y),
            market_index=baseline_index,
            inventory_blockers=baseline_inventory_blockers,
        )
        current = _pair_market_state(
            str(pair.asset_x),
            str(pair.asset_y),
            market_index=current_index,
            inventory_blockers=current_inventory_blockers,
        )
        reasons: list[str] = []
        for field, reason in (
            ("pair_ready", "pair_readiness_changed"),
            ("market_x", "market_x_changed"),
            ("market_y", "market_y_changed"),
            ("mapping_blocker", "mapping_blocker_changed"),
            ("pair_max_leverage", "pair_max_leverage_changed"),
            ("only_isolated", "isolated_margin_rule_changed"),
        ):
            if not _mapping_values_equal(baseline[field], current[field]):
                reasons.append(reason)
        rows.append(
            {
                "schema_version": MAPPING_REFRESH_SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "mapping_refresh_id": refresh_id,
                "pair_group_id": _text(getattr(pair, "pair_group_id", "")),
                "pair_group_key": _text(getattr(pair, "pair_group_key", "")),
                "wizard_exchange": _text(getattr(pair, "wizard_exchange", "")),
                "timeframe": _text(getattr(pair, "timeframe", "")),
                "pair": _text(getattr(pair, "pair", "")),
                "asset_x": _text(pair.asset_x),
                "asset_y": _text(pair.asset_y),
                "baseline_pair_ready": baseline["pair_ready"],
                "current_pair_ready": current["pair_ready"],
                "baseline_market_x": baseline["market_x"],
                "current_market_x": current["market_x"],
                "baseline_market_y": baseline["market_y"],
                "current_market_y": current["market_y"],
                "baseline_pair_max_leverage": baseline["pair_max_leverage"],
                "current_pair_max_leverage": current["pair_max_leverage"],
                "baseline_only_isolated": baseline["only_isolated"],
                "current_only_isolated": current["only_isolated"],
                "baseline_mapping_blocker": baseline["mapping_blocker"],
                "current_mapping_blocker": current["mapping_blocker"],
                "current_inventory_checked_at": current["checked_at"],
                "mapping_drift": bool(reasons),
                "mapping_drift_reasons": ";".join(reasons),
                "mapping_status": "READY" if current["pair_ready"] else "BLOCKED",
                "baseline_evidence_path": baseline_evidence_path,
                "current_evidence_path": current_evidence_path,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _pair_market_state(
    asset_x: str,
    asset_y: str,
    *,
    market_index: dict[str, list[HyperliquidMarket]],
    inventory_blockers: list[str],
) -> dict[str, object]:
    markets_x = market_index.get(asset_x.upper(), [])
    markets_y = market_index.get(asset_y.upper(), [])
    blockers = list(inventory_blockers)
    if asset_x.upper() == asset_y.upper():
        blockers.append("identical_canonical_assets")
    if not markets_x:
        blockers.append(f"missing_hyperliquid_testnet_perp:{asset_x}")
    elif len(markets_x) > 1:
        blockers.append(f"ambiguous_hyperliquid_testnet_perp:{asset_x}")
    if not markets_y:
        blockers.append(f"missing_hyperliquid_testnet_perp:{asset_y}")
    elif len(markets_y) > 1:
        blockers.append(f"ambiguous_hyperliquid_testnet_perp:{asset_y}")
    leverage_values = [
        market.max_leverage
        for market in [*markets_x, *markets_y]
        if market.max_leverage is not None
    ]
    checked_at = sorted(
        {market.checked_at for market in [*markets_x, *markets_y] if market.checked_at}
    )
    return {
        "pair_ready": bool(len(markets_x) == 1 and len(markets_y) == 1 and not blockers),
        "market_x": ";".join(market.label for market in markets_x),
        "market_y": ";".join(market.label for market in markets_y),
        "pair_max_leverage": min(leverage_values) if len(leverage_values) >= 2 else None,
        "only_isolated": any(market.only_isolated for market in [*markets_x, *markets_y]),
        "mapping_blocker": ";".join(dict.fromkeys(blockers)),
        "checked_at": ";".join(checked_at),
    }


def _build_inventory_drift_rows(
    baseline_inventory: pd.DataFrame,
    current_inventory: pd.DataFrame,
    *,
    run_id: str,
    refresh_id: str,
    baseline_evidence_path: str,
    current_evidence_path: str,
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "exhaustive_run_id",
        "mapping_refresh_id",
        "asset",
        "baseline_present",
        "current_present",
        "baseline_tradable_perp",
        "current_tradable_perp",
        "baseline_market_labels",
        "current_market_labels",
        "baseline_max_leverage",
        "current_max_leverage",
        "baseline_only_isolated",
        "current_only_isolated",
        "drift_status",
        "drift_reasons",
        "baseline_evidence_path",
        "current_evidence_path",
        "live_trading_authorized",
    ]
    baseline = _inventory_rule_index(baseline_inventory)
    current = _inventory_rule_index(current_inventory)
    rows: list[dict[str, object]] = []
    for asset in sorted(set(baseline) | set(current)):
        old = baseline.get(asset)
        new = current.get(asset)
        reasons: list[str] = []
        if old is None:
            status = "ADDED"
            reasons.append("asset_added")
        elif new is None:
            status = "REMOVED"
            reasons.append("asset_removed")
        else:
            for field, reason in (
                ("tradable_perp", "tradability_changed"),
                ("market_labels", "market_identity_changed"),
                ("max_leverage", "max_leverage_changed"),
                ("only_isolated", "isolated_margin_rule_changed"),
            ):
                if not _mapping_values_equal(old[field], new[field]):
                    reasons.append(reason)
            status = "CHANGED" if reasons else "UNCHANGED"
        rows.append(
            {
                "schema_version": MAPPING_REFRESH_SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "mapping_refresh_id": refresh_id,
                "asset": asset,
                "baseline_present": old is not None,
                "current_present": new is not None,
                "baseline_tradable_perp": old["tradable_perp"] if old else False,
                "current_tradable_perp": new["tradable_perp"] if new else False,
                "baseline_market_labels": old["market_labels"] if old else "",
                "current_market_labels": new["market_labels"] if new else "",
                "baseline_max_leverage": old["max_leverage"] if old else None,
                "current_max_leverage": new["max_leverage"] if new else None,
                "baseline_only_isolated": old["only_isolated"] if old else False,
                "current_only_isolated": new["only_isolated"] if new else False,
                "drift_status": status,
                "drift_reasons": ";".join(reasons),
                "baseline_evidence_path": baseline_evidence_path,
                "current_evidence_path": current_evidence_path,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _inventory_rule_index(inventory: pd.DataFrame) -> dict[str, dict[str, object]]:
    if inventory.empty or "asset" not in inventory.columns:
        return {}
    records: dict[str, dict[str, object]] = {}
    for asset, group in inventory.groupby(inventory["asset"].astype(str).str.upper()):
        if not asset:
            continue
        labels = sorted(
            {
                f"{_text(row.get('universe_name')) or asset}#{_text(row.get('asset_index'))}"
                for _, row in group.iterrows()
            }
        )
        tradable = bool(
            group.get("tradable_perp", pd.Series(False, index=group.index)).map(_truthy).any()
        )
        leverage_values = [
            value
            for value in group.get("max_leverage", pd.Series(dtype=object)).map(_number)
            if value is not None
        ]
        records[asset] = {
            "tradable_perp": tradable,
            "market_labels": ";".join(labels),
            "max_leverage": min(leverage_values) if leverage_values else None,
            "only_isolated": bool(
                group.get("only_isolated", pd.Series(False, index=group.index)).map(_truthy).any()
            ),
        }
    return records


def _mapping_values_equal(left: object, right: object) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is not None or right_number is not None:
        return left_number == right_number
    return left == right


def _mapping_refresh_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard To Hyperliquid Mapping Refresh",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Refresh: `{summary['mapping_refresh_id']}`",
            f"- Point-in-time pair groups: {summary['pair_groups']}",
            f"- Hyperliquid-ready pair groups: {summary['current_ready_pair_groups']}",
            f"- Retained with mapping blockers: {summary['current_blocked_pair_groups']}",
            f"- Pair mappings changed: {summary['mapping_drift_pair_groups']}",
            f"- Inventory rules changed: {summary['inventory_drift_rows']}",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "The frozen Wizard run and its original Hyperliquid inventory remain unchanged. This refresh is separate point-in-time venue evidence.",
            "",
        ]
    )


def _build_source_ledger(source: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    metadata = [
        "schema_version",
        "exhaustive_run_id",
        "source_row_id",
        "source_row_fingerprint",
        "duplicate_occurrence",
        "accounting_status",
        "discovery_policy",
        "discovery_prefilter_applied",
        "canonical_wizard_exchange",
        "canonical_interval",
        "canonical_asset_x",
        "canonical_asset_y",
        "canonical_pair",
        "pair_group_key",
        "normalization_blocker",
    ]
    if source.empty:
        return pd.DataFrame(columns=metadata + list(source.columns))

    fingerprints = [_row_fingerprint(row) for _, row in source.iterrows()]
    occurrences: Counter[str] = Counter()
    rows: list[dict[str, object]] = []
    for (_, source_row), fingerprint in zip(source.iterrows(), fingerprints):
        occurrence = occurrences[fingerprint]
        occurrences[fingerprint] += 1
        source_row_id = (
            "wrow_" + sha256(f"{fingerprint}|{occurrence}".encode("utf-8")).hexdigest()[:20]
        )
        exchange = (
            normalize_wizard_exchange(
                _first(source_row, "sweep_exchange", "wizard_exchange", "exchange"),
                default="dydx",
            )
            or ""
        )
        interval = _token(
            _first(source_row, "scanner_interval", "sweep_interval", "interval", "timeframe")
        )
        symbol_x = normalize_wizard_symbol(_first(source_row, "symbol_1", "asset_x"), exchange)
        symbol_y = normalize_wizard_symbol(_first(source_row, "symbol_2", "asset_y"), exchange)
        asset_x = symbol_x.base_asset or ""
        asset_y = symbol_y.base_asset or ""
        normalization_blockers: list[str] = []
        if not asset_x:
            normalization_blockers.append("wizard_symbol_1_unresolved")
        if not asset_y:
            normalization_blockers.append("wizard_symbol_2_unresolved")
        if asset_x and asset_y and asset_x.upper() == asset_y.upper():
            normalization_blockers.append("identical_canonical_assets")
        canonical_pair = f"{asset_x}-USD-{asset_y}-USD" if asset_x and asset_y else ""
        pair_group_key = _pair_group_key(exchange, interval, asset_x, asset_y)
        row = {
            "schema_version": SCHEMA_VERSION,
            "exhaustive_run_id": run_id,
            "source_row_id": source_row_id,
            "source_row_fingerprint": fingerprint,
            "duplicate_occurrence": occurrence,
            "accounting_status": "ACCOUNTED",
            "discovery_policy": DISCOVERY_POLICY,
            "discovery_prefilter_applied": False,
            "canonical_wizard_exchange": exchange,
            "canonical_interval": interval,
            "canonical_asset_x": asset_x,
            "canonical_asset_y": asset_y,
            "canonical_pair": canonical_pair,
            "pair_group_key": pair_group_key,
            "normalization_blocker": ";".join(normalization_blockers),
        }
        row.update(source_row.to_dict())
        rows.append(row)
    return pd.DataFrame(
        rows, columns=metadata + [column for column in source.columns if column not in metadata]
    )


def _build_pair_ledger(
    source_ledger: pd.DataFrame,
    *,
    run_id: str,
    market_index: dict[str, list[HyperliquidMarket]],
    inventory_blockers: list[str],
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "exhaustive_run_id",
        "pair_group_id",
        "pair_group_key",
        "wizard_exchange",
        "timeframe",
        "asset_x",
        "asset_y",
        "pair",
        "observed_orientations",
        "source_row_count",
        "source_row_ids",
        "source_evidence_paths",
        "observed_sharpe_min",
        "observed_sharpe_max",
        "observed_returns_total_min",
        "observed_returns_total_max",
        "hyperliquid_market_x",
        "hyperliquid_market_y",
        "hyperliquid_market_x_count",
        "hyperliquid_market_y_count",
        "hyperliquid_leg_x_ready",
        "hyperliquid_leg_y_ready",
        "hyperliquid_pair_ready",
        "hyperliquid_pair_max_leverage",
        "hyperliquid_only_isolated",
        "hyperliquid_inventory_checked_at",
        "hyperliquid_mapping_blocker",
        "research_status",
        "live_trading_authorized",
    ]
    if source_ledger.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    usable = source_ledger[source_ledger["pair_group_key"].astype(str).str.len().gt(0)].copy()
    for group_key, group in usable.groupby("pair_group_key", sort=True):
        first = group.sort_values("source_row_id").iloc[0]
        asset_x = str(first["canonical_asset_x"])
        asset_y = str(first["canonical_asset_y"])
        markets_x = market_index.get(asset_x.upper(), [])
        markets_y = market_index.get(asset_y.upper(), [])
        blockers = list(inventory_blockers)
        blockers.extend(
            sorted(
                {
                    blocker
                    for value in group["normalization_blocker"].tolist()
                    for blocker in _text(value).split(";")
                    if blocker
                }
            )
        )
        if not markets_x:
            blockers.append(f"missing_hyperliquid_testnet_perp:{asset_x}")
        elif len(markets_x) > 1:
            blockers.append(f"ambiguous_hyperliquid_testnet_perp:{asset_x}")
        if not markets_y:
            blockers.append(f"missing_hyperliquid_testnet_perp:{asset_y}")
        elif len(markets_y) > 1:
            blockers.append(f"ambiguous_hyperliquid_testnet_perp:{asset_y}")
        ready = bool(len(markets_x) == 1 and len(markets_y) == 1 and not blockers)
        leverage_values = [
            market.max_leverage
            for market in [*markets_x, *markets_y]
            if market.max_leverage is not None
        ]
        pair_max_leverage = min(leverage_values) if len(leverage_values) >= 2 else None
        orientations = sorted(
            {
                f"{row.canonical_asset_x}/{row.canonical_asset_y}"
                for row in group.itertuples()
                if row.canonical_asset_x and row.canonical_asset_y
            }
        )
        evidence = sorted(
            {
                _text(value)
                for column in ("capture_path", "sweep_evidence_path", "source_path")
                if column in group.columns
                for value in group[column].tolist()
                if _text(value)
            }
        )
        checked_at = sorted(
            {market.checked_at for market in [*markets_x, *markets_y] if market.checked_at}
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "pair_group_id": "wpair_" + sha256(str(group_key).encode("utf-8")).hexdigest()[:20],
                "pair_group_key": group_key,
                "wizard_exchange": first["canonical_wizard_exchange"],
                "timeframe": first["canonical_interval"],
                "asset_x": asset_x,
                "asset_y": asset_y,
                "pair": f"{asset_x}-USD-{asset_y}-USD",
                "observed_orientations": ";".join(orientations),
                "source_row_count": int(len(group)),
                "source_row_ids": ";".join(sorted(group["source_row_id"].astype(str))),
                "source_evidence_paths": ";".join(evidence),
                "observed_sharpe_min": _numeric_extreme(group, "sharpe", "min"),
                "observed_sharpe_max": _numeric_extreme(group, "sharpe", "max"),
                "observed_returns_total_min": _numeric_extreme_any(
                    group, ("returns_total", "return_total"), "min"
                ),
                "observed_returns_total_max": _numeric_extreme_any(
                    group, ("returns_total", "return_total"), "max"
                ),
                "hyperliquid_market_x": ";".join(market.label for market in markets_x),
                "hyperliquid_market_y": ";".join(market.label for market in markets_y),
                "hyperliquid_market_x_count": len(markets_x),
                "hyperliquid_market_y_count": len(markets_y),
                "hyperliquid_leg_x_ready": len(markets_x) == 1,
                "hyperliquid_leg_y_ready": len(markets_y) == 1,
                "hyperliquid_pair_ready": ready,
                "hyperliquid_pair_max_leverage": pair_max_leverage,
                "hyperliquid_only_isolated": any(
                    market.only_isolated for market in [*markets_x, *markets_y]
                ),
                "hyperliquid_inventory_checked_at": ";".join(checked_at),
                "hyperliquid_mapping_blocker": ";".join(dict.fromkeys(blockers)),
                "research_status": "READY_FOR_HISTORY_AND_SETTINGS"
                if ready
                else "RETAINED_WITH_EXECUTION_BLOCKER",
                "live_trading_authorized": False,
            }
        )

    unresolved = source_ledger[source_ledger["pair_group_key"].astype(str).str.len().eq(0)]
    for row in unresolved.itertuples():
        group_key = f"unresolved|{row.source_row_id}"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "pair_group_id": "wpair_" + sha256(group_key.encode("utf-8")).hexdigest()[:20],
                "pair_group_key": group_key,
                "wizard_exchange": row.canonical_wizard_exchange,
                "timeframe": row.canonical_interval,
                "asset_x": row.canonical_asset_x,
                "asset_y": row.canonical_asset_y,
                "pair": row.canonical_pair,
                "observed_orientations": "",
                "source_row_count": 1,
                "source_row_ids": row.source_row_id,
                "source_evidence_paths": _text(getattr(row, "sweep_evidence_path", "")),
                "observed_sharpe_min": _number(getattr(row, "sharpe", None)),
                "observed_sharpe_max": _number(getattr(row, "sharpe", None)),
                "observed_returns_total_min": _number(getattr(row, "returns_total", None)),
                "observed_returns_total_max": _number(getattr(row, "returns_total", None)),
                "hyperliquid_market_x": "",
                "hyperliquid_market_y": "",
                "hyperliquid_market_x_count": 0,
                "hyperliquid_market_y_count": 0,
                "hyperliquid_leg_x_ready": False,
                "hyperliquid_leg_y_ready": False,
                "hyperliquid_pair_ready": False,
                "hyperliquid_pair_max_leverage": None,
                "hyperliquid_only_isolated": False,
                "hyperliquid_inventory_checked_at": "",
                "hyperliquid_mapping_blocker": row.normalization_blocker
                or "wizard_pair_unresolved",
                "research_status": "RETAINED_WITH_NORMALIZATION_BLOCKER",
                "live_trading_authorized": False,
            }
        )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(
            ["hyperliquid_pair_ready", "wizard_exchange", "timeframe", "pair_group_id"],
            ascending=[False, True, True, True],
        )
        .reset_index(drop=True)
    )


def _build_experiment_matrix(pair_ledger: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    columns = [
        "schema_version",
        "exhaustive_run_id",
        "experiment_id",
        "pair_group_id",
        "wizard_exchange",
        "timeframe",
        "exact_mode",
        "orientation",
        "asset_x",
        "asset_y",
        "pair",
        "orientation_policy",
        "orientation_reason",
        "discovery_prefilter_applied",
        "baseline_notional_multiplier",
        "baseline_margin_mode",
        "canonical_replay_status",
        "canonical_replay_blocker",
        "hyperliquid_pair_ready",
        "hyperliquid_pair_max_leverage",
        "hyperliquid_mapping_blocker",
        "leverage_surface_status",
        "testnet_validation_status",
        "testnet_blocker",
        "live_trading_authorized",
        "source_row_ids",
        "evidence_path",
    ]
    rows: list[dict[str, object]] = []
    for pair in pair_ledger.itertuples():
        for exact_mode in EXACT_MODES:
            for orientation in ORIENTATIONS:
                asset_x, asset_y = (pair.asset_x, pair.asset_y)
                if orientation == "reverse":
                    asset_x, asset_y = asset_y, asset_x
                identity = "|".join([pair.pair_group_id, exact_mode, orientation, "baseline_1x"])
                mapping_blocker = _text(pair.hyperliquid_mapping_blocker)
                testnet_blocker = mapping_blocker or "canonical_1x_replay_not_completed"
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "exhaustive_run_id": run_id,
                        "experiment_id": "wexp_"
                        + sha256(identity.encode("utf-8")).hexdigest()[:20],
                        "pair_group_id": pair.pair_group_id,
                        "wizard_exchange": pair.wizard_exchange,
                        "timeframe": pair.timeframe,
                        "exact_mode": exact_mode,
                        "orientation": orientation,
                        "asset_x": asset_x,
                        "asset_y": asset_y,
                        "pair": f"{asset_x}-USD-{asset_y}-USD" if asset_x and asset_y else "",
                        "orientation_policy": ORIENTATION_POLICY,
                        "orientation_reason": _orientation_reason(exact_mode),
                        "discovery_prefilter_applied": False,
                        "baseline_notional_multiplier": 1.0,
                        "baseline_margin_mode": "not_applicable_local_replay",
                        "canonical_replay_status": "NOT_RUN",
                        "canonical_replay_blocker": "point_in_time_history_and_exact_mode_settings_not_validated",
                        "hyperliquid_pair_ready": bool(pair.hyperliquid_pair_ready),
                        "hyperliquid_pair_max_leverage": pair.hyperliquid_pair_max_leverage,
                        "hyperliquid_mapping_blocker": mapping_blocker,
                        "leverage_surface_status": "NOT_RUN_UNTIL_CANONICAL_REPLAY_PASSES",
                        "testnet_validation_status": "NOT_RUN"
                        if pair.hyperliquid_pair_ready
                        else "BLOCKED",
                        "testnet_blocker": testnet_blocker,
                        "live_trading_authorized": False,
                        "source_row_ids": pair.source_row_ids,
                        "evidence_path": pair.source_evidence_paths,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def _build_pair_detail_capture_queue(
    pair_ledger: pd.DataFrame,
    source_ledger: pd.DataFrame,
    *,
    run_id: str,
) -> pd.DataFrame:
    columns = [
        "schema_version",
        "exhaustive_run_id",
        "pair_detail_queue_id",
        "pair_group_id",
        "pair_group_key",
        "capture_order",
        "capture_order_policy",
        "accounting_status",
        "wizard_exchange",
        "timeframe",
        "pair",
        "asset_x",
        "asset_y",
        "scanner_cell_id",
        "scanner_capture_timestamps",
        "scanner_asset_x_raw",
        "scanner_asset_y_raw",
        "observed_scanner_mode_labels",
        "modes_to_capture",
        "orientations_to_capture",
        "source_row_count",
        "source_row_ids",
        "source_evidence_paths",
        "pair_detail_url",
        "pair_detail_route_status",
        "required_fields",
        "capture_status",
        "capture_blocker",
        "pair_presence_recheck_required",
        "pair_presence_status",
        "capture_cadence_policy",
        "hyperliquid_pair_ready",
        "hyperliquid_mapping_blocker",
        "next_step",
        "live_trading_authorized",
    ]
    if pair_ledger.empty:
        return pd.DataFrame(columns=columns)

    source_groups = {
        str(key): group
        for key, group in source_ledger.groupby("pair_group_key", sort=False)
        if str(key)
    }
    ordered_pairs = pair_ledger.sort_values(
        ["wizard_exchange", "timeframe", "hyperliquid_pair_ready", "pair_group_id"],
        ascending=[True, True, False, True],
    )
    rows: list[dict[str, object]] = []
    for capture_order, pair in enumerate(ordered_pairs.itertuples(), start=1):
        source_group = source_groups.get(str(pair.pair_group_key), pd.DataFrame())
        detail_urls = _nonempty_unique_values(
            source_group,
            (
                "dashboard_raw_pair_detail_url",
                "dashboard_raw_pair_detail_href",
                "pair_detail_url",
                "pair_url",
            ),
        )
        route_status = "CAPTURED" if len(detail_urls) == 1 else "MISSING"
        if len(detail_urls) > 1:
            route_status = "AMBIGUOUS"
        route_blocker = {
            "CAPTURED": "",
            "MISSING": "pair_detail_route_not_captured_in_scanner_snapshot",
            "AMBIGUOUS": "multiple_pair_detail_routes_captured",
        }[route_status]
        mode_labels = _nonempty_unique_values(
            source_group,
            (
                "dashboard_raw_raw_strategy_cell",
                "raw_strategy_cell",
                "dashboard_raw_spread_type",
                "strategy_label",
            ),
        )
        capture_timestamps = _nonempty_unique_values(
            source_group,
            (
                "sweep_captured_at",
                "captured_at",
                "capture_timestamp",
                "source_timestamp",
            ),
        )
        raw_asset_x = _nonempty_unique_values(
            source_group,
            ("asset_x_raw", "dashboard_raw_asset_x", "symbol_1"),
        )
        raw_asset_y = _nonempty_unique_values(
            source_group,
            ("asset_y_raw", "dashboard_raw_asset_y", "symbol_2"),
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "exhaustive_run_id": run_id,
                "pair_detail_queue_id": "wdetail_"
                + sha256(str(pair.pair_group_id).encode("utf-8")).hexdigest()[:20],
                "pair_group_id": pair.pair_group_id,
                "pair_group_key": pair.pair_group_key,
                "capture_order": capture_order,
                "capture_order_policy": "scanner_cell_first_then_hyperliquid_ready_no_rows_filtered",
                "accounting_status": "ACCOUNTED",
                "wizard_exchange": pair.wizard_exchange,
                "timeframe": pair.timeframe,
                "pair": pair.pair,
                "asset_x": pair.asset_x,
                "asset_y": pair.asset_y,
                "scanner_cell_id": f"{pair.wizard_exchange}|{pair.timeframe}",
                "scanner_capture_timestamps": ";".join(capture_timestamps),
                "scanner_asset_x_raw": ";".join(raw_asset_x),
                "scanner_asset_y_raw": ";".join(raw_asset_y),
                "observed_scanner_mode_labels": ";".join(mode_labels),
                "modes_to_capture": ";".join(EXACT_MODES),
                "orientations_to_capture": ";".join(ORIENTATIONS),
                "source_row_count": pair.source_row_count,
                "source_row_ids": pair.source_row_ids,
                "source_evidence_paths": pair.source_evidence_paths,
                "pair_detail_url": ";".join(detail_urls),
                "pair_detail_route_status": route_status,
                "required_fields": ";".join(PAIR_DETAIL_REQUIRED_FIELDS),
                "capture_status": "NOT_CAPTURED",
                "capture_blocker": route_blocker or "pair_detail_fields_not_captured",
                "pair_presence_recheck_required": True,
                "pair_presence_status": "NOT_RECHECKED",
                "capture_cadence_policy": "capture_pair_detail_immediately_within_same_live_scanner_cell",
                "hyperliquid_pair_ready": bool(pair.hyperliquid_pair_ready),
                "hyperliquid_mapping_blocker": pair.hyperliquid_mapping_blocker,
                "next_step": (
                    "open_captured_pair_detail_route_and_capture_all_modes"
                    if route_status == "CAPTURED"
                    else "recheck_pair_in_same_live_scanner_cell_then_capture_route_and_all_modes"
                ),
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _build_validation(
    *,
    source: pd.DataFrame,
    source_ledger: pd.DataFrame,
    pair_ledger: pd.DataFrame,
    pair_detail_queue: pd.DataFrame,
    experiment_matrix: pd.DataFrame,
    source_path: Path,
    inventory_path: Path,
    sweep_complete: bool,
    all_pages_proven: bool,
) -> pd.DataFrame:
    expected_experiments = len(pair_ledger) * len(EXACT_MODES) * len(ORIENTATIONS)
    unique_ids = source_ledger["source_row_id"].nunique() if not source_ledger.empty else 0
    checks = [
        _check(
            "wizard_source_snapshot_available",
            source_path.exists(),
            "Wizard candidate snapshot is missing.",
        ),
        _check("wizard_source_rows_nonzero", not source.empty, "Wizard snapshot contains no rows."),
        _check(
            "every_source_row_accounted",
            len(source) == len(source_ledger) and unique_ids == len(source),
            f"source={len(source)} ledger={len(source_ledger)} unique_ids={unique_ids}",
        ),
        _check(
            "zero_discovery_prefilters",
            bool(
                source_ledger.empty
                or not source_ledger["discovery_prefilter_applied"].astype(bool).any()
            ),
            "A source row was filtered before exhaustive accounting.",
        ),
        _check("all_sweep_cells_complete", sweep_complete, "Wizard sweep cells are incomplete."),
        _blocked_check(
            "all_dashboard_pages_proven",
            all_pages_proven,
            "wizard_all_pages_capture_not_proven",
        ),
        _blocked_check(
            "all_symbols_normalized",
            bool(
                source_ledger.empty
                or source_ledger["normalization_blocker"].fillna("").eq("").all()
            ),
            "one_or_more_wizard_symbols_unresolved",
        ),
        _check(
            "mode_orientation_matrix_complete",
            len(experiment_matrix) == expected_experiments
            and experiment_matrix["experiment_id"].nunique() == expected_experiments,
            f"expected={expected_experiments} actual={len(experiment_matrix)}",
        ),
        _check(
            "every_pair_group_has_pair_detail_work_item",
            len(pair_detail_queue) == len(pair_ledger)
            and pair_detail_queue["pair_group_id"].nunique() == len(pair_ledger),
            f"pair_groups={len(pair_ledger)} queue_rows={len(pair_detail_queue)}",
        ),
        _blocked_check(
            "all_pairs_hyperliquid_mapped",
            bool(
                not pair_ledger.empty and pair_ledger["hyperliquid_pair_ready"].astype(bool).all()
            ),
            f"hyperliquid_unavailable_pair_groups={int((~pair_ledger['hyperliquid_pair_ready'].astype(bool)).sum()) if not pair_ledger.empty else 0}",
        ),
        _check(
            "hyperliquid_inventory_available",
            inventory_path.exists(),
            "Hyperliquid testnet inventory is missing.",
        ),
        _check(
            "live_trading_disabled",
            bool(
                experiment_matrix.empty
                or not experiment_matrix["live_trading_authorized"].astype(bool).any()
            ),
            "An exhaustive research row incorrectly authorized live trading.",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "status", "reason"])


def _market_index(inventory: pd.DataFrame) -> tuple[dict[str, list[HyperliquidMarket]], list[str]]:
    if inventory.empty or "asset" not in inventory.columns:
        return {}, ["hyperliquid_testnet_market_inventory_missing_or_empty"]
    blockers = sorted(
        {
            item.strip()
            for value in inventory.get("fetch_blocker", pd.Series(dtype=object)).tolist()
            for item in _text(value).split(";")
            if item.strip()
        }
    )
    by_key: dict[str, list[HyperliquidMarket]] = {}
    suffixes: dict[str, list[HyperliquidMarket]] = {}
    for _, row in inventory.iterrows():
        if not _truthy(row.get("tradable_perp", False)):
            continue
        asset = _text(row.get("asset")).upper()
        if not asset:
            continue
        market = HyperliquidMarket(
            asset=asset,
            universe_name=_text(row.get("universe_name")) or asset,
            asset_index=_text(row.get("asset_index")),
            max_leverage=_number(row.get("max_leverage")),
            only_isolated=_truthy(row.get("only_isolated", False)),
            checked_at=_text(row.get("checked_at_utc")),
        )
        by_key.setdefault(asset, []).append(market)
        if ":" in asset:
            suffixes.setdefault(asset.rsplit(":", 1)[-1], []).append(market)
    for suffix, markets in suffixes.items():
        if suffix not in by_key:
            by_key[suffix] = markets
    return by_key, blockers


def _sweep_complete(source: pd.DataFrame, manifest: pd.DataFrame) -> bool:
    if source.empty:
        return False
    if "pagination_complete" in source.columns:
        source_complete = source["pagination_complete"].map(_truthy).all()
        if manifest.empty:
            return bool(source_complete)
        statuses = manifest.get("capture_status", pd.Series(dtype=object)).astype(str).str.upper()
        return bool(source_complete and not statuses.empty and statuses.eq("COMPLETE").all())
    source_complete = (
        source.get("sweep_complete", pd.Series(False, index=source.index)).map(_truthy).all()
    )
    if manifest.empty:
        return bool(source_complete)
    statuses = manifest.get("status", pd.Series(dtype=object)).astype(str).str.lower()
    return bool(source_complete and not statuses.empty and statuses.eq("completed").all())


def _all_pages_proven(source: pd.DataFrame) -> bool:
    dashboard_required = {
        "capture_id",
        "scanner_exchange",
        "scanner_interval",
        "pagination_complete",
        "discovery_prefilter_applied",
    }
    if not source.empty and dashboard_required.issubset(source.columns):
        expected_cells = {
            f"{venue}|{timeframe}"
            for venue in EXPECTED_CRYPTO_VENUES
            for timeframe in EXPECTED_TIMEFRAMES
        }
        actual_cells = {
            f"{normalize_wizard_exchange(row.scanner_exchange) or ''}|{_token(row.scanner_interval)}"
            for row in source[["scanner_exchange", "scanner_interval"]]
            .drop_duplicates()
            .itertuples()
        }
        return bool(
            expected_cells == actual_cells
            and source["pagination_complete"].map(_truthy).all()
            and not source["discovery_prefilter_applied"].map(_truthy).any()
        )
    required = {"sweep_page", "sweep_total_pages", "sweep_page_complete"}
    if source.empty or not required.issubset(source.columns):
        return False
    if not source["sweep_page_complete"].map(_truthy).all():
        return False
    grouped = source.groupby(["request_id"], dropna=False)
    for _, group in grouped:
        pages = {
            int(value) for value in pd.to_numeric(group["sweep_page"], errors="coerce").dropna()
        }
        totals = {
            int(value)
            for value in pd.to_numeric(group["sweep_total_pages"], errors="coerce").dropna()
        }
        if len(totals) != 1 or pages != set(range(1, next(iter(totals)) + 1)):
            return False
    return True


def _pair_group_key(exchange: str, interval: str, asset_x: str, asset_y: str) -> str:
    if not asset_x or not asset_y:
        return ""
    assets = sorted([asset_x.upper(), asset_y.upper()])
    return "|".join([exchange, interval, *assets])


def _orientation_reason(exact_mode: str) -> str:
    if exact_mode == "Copula":
        return "conditional_copula_probabilities_are_directional"
    return "hedge_ratio_regression_and_signal_definition_can_be_directional"


def _row_fingerprint(row: pd.Series) -> str:
    payload = {str(key): _json_value(value) for key, value in sorted(row.to_dict().items())}
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _numeric_extreme(frame: pd.DataFrame, column: str, method: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.min() if method == "min" else values.max())


def _numeric_extreme_any(frame: pd.DataFrame, columns: Iterable[str], method: str) -> float | None:
    for column in columns:
        if column in frame.columns:
            return _numeric_extreme(frame, column, method)
    return None


def _nonempty_unique_values(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    values: set[str] = set()
    for column in columns:
        if column not in frame.columns:
            continue
        values.update(_text(value) for value in frame[column].tolist() if _text(value))
    return sorted(values)


def _check(name: str, passed: bool, reason: str) -> dict[str, str]:
    return {
        "check": name,
        "status": "PASS" if passed else "FAIL",
        "reason": "" if passed else reason,
    }


def _blocked_check(name: str, passed: bool, reason: str) -> dict[str, str]:
    return {
        "check": name,
        "status": "PASS" if passed else "BLOCKED",
        "reason": "" if passed else reason,
    }


def _summary_markdown(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Exhaustive Wizard to Hyperliquid Run",
            "",
            f"- Run: `{summary['run_id']}`",
            f"- Authority: `{summary['authority']}`",
            f"- Discovery policy: `{summary['discovery_policy']}`",
            f"- Source rows accounted: {summary['accounted_source_rows']} / {summary['source_rows']}",
            f"- Pair groups: {summary['pair_groups']}",
            f"- Pair-detail work items: {summary['pair_detail_capture_queue_rows']}",
            f"- Hyperliquid-ready pair groups: {summary['hyperliquid_ready_pair_groups']}",
            f"- Retained with Hyperliquid blockers: {summary['hyperliquid_blocked_pair_groups']}",
            f"- Planned exact-mode/orientation experiments: {summary['planned_experiments']}",
            f"- Sweep cells complete: `{str(summary['sweep_cells_complete']).lower()}`",
            f"- All dashboard pages proven: `{str(summary['all_dashboard_pages_proven']).lower()}`",
            f"- Live trading authorized: `{str(summary['live_trading_authorized']).lower()}`",
            "",
            "Sharpe, return, trade count, liquidity, stationarity, and copula values are retained as observations. None are intake filters.",
            "Canonical replay is fixed at 1x. Leverage and margin analysis is a separate later stage and cannot rescue a failing 1x strategy.",
            "Unavailable or ambiguous Hyperliquid mappings remain in the ledgers with explicit blockers.",
            "",
        ]
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return pd.DataFrame()


def _file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return sha256(path.read_bytes()).hexdigest()


def _copy_snapshot_input(source: Path, destination: Path) -> None:
    if source.exists() and source.is_file():
        atomic_write_bytes(destination, source.read_bytes())
    else:
        atomic_write_text(destination, "", encoding="utf-8")


def _first(row: pd.Series, *columns: str) -> object:
    for column in columns:
        value = row.get(column, "")
        if _text(value):
            return value
    return ""


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _number(value: object) -> float | None:
    try:
        if value is None or _text(value) == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _token(value: object) -> str:
    return "_".join(_text(value).strip().lower().replace("-", " ").split())


def _text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _json_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
