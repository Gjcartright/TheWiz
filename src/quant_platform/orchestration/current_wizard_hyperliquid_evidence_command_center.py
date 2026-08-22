"""Prospective L2 coverage and evidence funnel for the current Wizard board."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.canonical_wizard_hyperliquid_contract import (
    MINIMUM_PROVISIONAL_L2_SAMPLES,
    MINIMUM_STRICT_L2_SAMPLES,
    PROVISIONAL_L2_WINDOW_HOURS,
    REFERENCE_LEG_NOTIONAL_USD,
    STRICT_L2_WINDOW_HOURS,
)
from quant_platform.runtime_types import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "current_wizard_hyperliquid_evidence_command_center.v1"
CAPTURE_CADENCE_MINUTES = 5
STAGES = (
    "handoff",
    "hyperliquid_mapping",
    "canonical_replay",
    "cost_evidence",
    "observed_cost_replay",
    "walkforward",
    "regime",
    "robustness",
    "concentration",
    "leverage",
    "testnet_preflight",
    "execution_acceptance",
)


def build_current_wizard_hyperliquid_evidence_command_center(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    """Build prospective collection coverage without changing acceptance state."""

    as_of = _as_utc(now)
    active = root / "reports" / "active"
    dashboard = root / "reports" / "dashboard"
    active.mkdir(parents=True, exist_ok=True)
    dashboard.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "cost_manifest": active / "current_wizard_hyperliquid_cost_manifest.json",
        "pair_costs": active / "current_wizard_hyperliquid_pair_cost_evidence.csv",
        "funding_assets": active
        / "current_wizard_hyperliquid_funding_asset_results.csv",
        "l2_samples": root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv",
        "failure_attribution": active
        / "current_wizard_hyperliquid_failure_attribution.csv",
        "failure_manifest": active
        / "current_wizard_hyperliquid_failure_attribution_manifest.json",
    }
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evidence command-center inputs missing: {missing}")

    cost_manifest = _read_json(source_paths["cost_manifest"])
    selected_keys = {
        _text(value)
        for value in cost_manifest.get("selected_pair_group_keys", [])
        if _text(value)
    }
    if not selected_keys:
        raise ValueError("Current cost manifest has no selected pair groups")
    pair_costs = pd.read_csv(source_paths["pair_costs"], keep_default_na=False)
    funding_assets = pd.read_csv(source_paths["funding_assets"], keep_default_na=False)
    failure = pd.read_csv(source_paths["failure_attribution"], keep_default_na=False)
    samples = _normalize_l2_samples(
        pd.read_csv(source_paths["l2_samples"], keep_default_na=False)
    )
    selected_pairs = pair_costs.loc[
        pair_costs["pair_group_key"].astype(str).isin(selected_keys)
        & pair_costs["selected_for_cost_evidence"].map(_truthy)
    ].copy()
    if set(selected_pairs["pair_group_key"].astype(str)) != selected_keys:
        raise ValueError("Selected current-board pair evidence is incomplete")

    assets = _build_asset_coverage(
        selected_pairs=selected_pairs,
        funding_assets=funding_assets,
        samples=samples,
        as_of=as_of,
        evidence_paths=source_paths,
        root=root,
    )
    pairs = _build_pair_coverage(
        selected_pairs=selected_pairs,
        assets=assets,
        as_of=as_of,
        evidence_paths=source_paths,
        root=root,
    )
    funnel = _build_funnel(failure=failure, root=root, source_paths=source_paths)
    blockers = _build_blocker_summary(failure=failure, root=root, source_paths=source_paths)
    pair_funnel = _build_pair_funnel(failure=failure, pairs=pairs)
    validation = _build_validation(
        selected_keys=selected_keys,
        selected_pairs=selected_pairs,
        assets=assets,
        pairs=pairs,
        funnel=funnel,
        failure=failure,
    )

    input_hashes = {name: _file_hash(path) for name, path in source_paths.items()}
    identity = {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of.isoformat(),
        "cost_evidence_id": _text(cost_manifest.get("cost_evidence_id")),
        "selected_pair_group_keys": sorted(selected_keys),
        "input_hashes": input_hashes,
        "strict_l2_window_hours": STRICT_L2_WINDOW_HOURS,
        "provisional_l2_window_hours": PROVISIONAL_L2_WINDOW_HOURS,
        "minimum_strict_l2_samples": MINIMUM_STRICT_L2_SAMPLES,
        "minimum_provisional_l2_samples": MINIMUM_PROVISIONAL_L2_SAMPLES,
        "reference_leg_notional_usd": REFERENCE_LEG_NOTIONAL_USD,
    }
    command_center_id = "cwevidence_" + sha256(
        _canonical_json(identity).encode("utf-8")
    ).hexdigest()[:20]
    snapshot_dir = (
        root
        / "reports"
        / "snapshots"
        / "current_wizard_hyperliquid"
        / "evidence_command_centers"
        / command_center_id
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    active_paths = {
        "assets": active / "current_wizard_hyperliquid_asset_evidence_coverage.csv",
        "pairs": active / "current_wizard_hyperliquid_pair_evidence_coverage.csv",
        "collection_plan": active
        / "current_wizard_hyperliquid_l2_collection_plan.csv",
        "funnel": active / "current_wizard_hyperliquid_evidence_funnel.csv",
        "blockers": active / "current_wizard_hyperliquid_evidence_blockers.csv",
        "pair_funnel": active
        / "current_wizard_hyperliquid_pair_evidence_funnel.csv",
        "validation": active
        / "current_wizard_hyperliquid_evidence_command_center_validation.csv",
        "manifest": active
        / "current_wizard_hyperliquid_evidence_command_center_manifest.json",
        "summary_md": active
        / "current_wizard_hyperliquid_evidence_command_center.md",
    }
    dashboard_paths = {
        key: dashboard / path.name
        for key, path in active_paths.items()
        if key not in {"manifest", "summary_md"}
    }
    snapshot_paths = {
        "assets": snapshot_dir / "asset_evidence_coverage.csv",
        "pairs": snapshot_dir / "pair_evidence_coverage.csv",
        "collection_plan": snapshot_dir / "l2_collection_plan.csv",
        "funnel": snapshot_dir / "evidence_funnel.csv",
        "blockers": snapshot_dir / "evidence_blockers.csv",
        "pair_funnel": snapshot_dir / "pair_evidence_funnel.csv",
        "validation": snapshot_dir / "validation.csv",
        "manifest": snapshot_dir / "manifest.json",
        "summary_md": snapshot_dir / "summary.md",
    }
    frames = {
        "assets": assets,
        "pairs": pairs,
        "collection_plan": _collection_plan(pairs),
        "funnel": funnel,
        "blockers": blockers,
        "pair_funnel": pair_funnel,
        "validation": validation,
    }
    for key, frame in frames.items():
        _write_csv(frame, active_paths[key])
        _write_csv(frame, dashboard_paths[key])
        _write_csv(frame, snapshot_paths[key])

    summary = {
        "command_center_id": command_center_id,
        "as_of": as_of.isoformat(),
        "current_snapshot_pair_groups": len(pairs),
        "rolling_assets": len(assets),
        "rolling_strict_ready_assets": int(assets["rolling_strict_l2_ready"].sum()),
        "rolling_provisional_ready_assets": int(
            assets["rolling_provisional_l2_ready"].sum()
        ),
        "rolling_strict_ready_pairs": int(pairs["rolling_strict_l2_ready"].sum()),
        "rolling_provisional_ready_pairs": int(
            pairs["rolling_provisional_l2_ready"].sum()
        ),
        "capture_due_assets": int(assets["capture_due"].sum()),
        "current_snapshot_strict_cost_ready_pairs": int(
            pairs["current_snapshot_strict_cost_ready"].sum()
        ),
        "experiments": int(failure["experiment_id"].nunique()),
        "acceptance_ready_experiments": int(
            failure.get(
                "execution_acceptance_ready", pd.Series(False, index=failure.index)
            ).map(_truthy).sum()
        ),
        "validation_status": (
            "PASS" if validation["status"].eq("PASS").all() else "BLOCKED"
        ),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    summary_text = _summary_markdown(summary=summary, funnel=funnel, assets=assets)
    _write_text(summary_text, active_paths["summary_md"])
    _write_text(summary_text, snapshot_paths["summary_md"])
    manifest = {
        **identity,
        **summary,
        "artifacts": {
            **{key: _relative(path, root) for key, path in active_paths.items()},
            **{
                f"snapshot_{key}": _relative(path, root)
                for key, path in snapshot_paths.items()
            },
        },
        "output_hashes": {
            key: _file_hash(path)
            for key, path in snapshot_paths.items()
            if key != "manifest"
        },
    }
    _write_json(manifest, active_paths["manifest"])
    _write_json(manifest, snapshot_paths["manifest"])
    if summary["validation_status"] != "PASS":
        failed = validation.loc[validation["status"].ne("PASS"), "check"].tolist()
        raise ValueError("Evidence command-center validation failed: " + ",".join(failed))
    return CommandResult(
        paths={**active_paths, **{f"dashboard_{k}": v for k, v in dashboard_paths.items()}},
        summary=summary,
    )


def _build_asset_coverage(
    *,
    selected_pairs: pd.DataFrame,
    funding_assets: pd.DataFrame,
    samples: pd.DataFrame,
    as_of: datetime,
    evidence_paths: dict[str, Path],
    root: Path,
) -> pd.DataFrame:
    required_assets = sorted(
        set(selected_pairs["asset_x"].astype(str).str.upper())
        | set(selected_pairs["asset_y"].astype(str).str.upper())
    )
    funding_lookup = {
        _text(row.get("asset")).upper(): row
        for row in funding_assets.to_dict("records")
        if _text(row.get("asset"))
    }
    now = pd.Timestamp(as_of)
    rows: list[dict[str, Any]] = []
    for asset in required_assets:
        asset_samples = samples.loc[samples["asset"].eq(asset)].copy()
        strict = asset_samples.loc[
            asset_samples["source_timestamp"].between(
                now - pd.Timedelta(hours=STRICT_L2_WINDOW_HOURS), now, inclusive="both"
            )
        ]
        provisional = asset_samples.loc[
            asset_samples["source_timestamp"].between(
                now - pd.Timedelta(hours=PROVISIONAL_L2_WINDOW_HOURS),
                now,
                inclusive="both",
            )
        ]
        strict_count = len(strict)
        provisional_count = len(provisional)
        strict_ready = strict_count >= MINIMUM_STRICT_L2_SAMPLES
        provisional_ready = provisional_count >= MINIMUM_PROVISIONAL_L2_SAMPLES
        latest = asset_samples["source_timestamp"].max() if not asset_samples.empty else pd.NaT
        next_due = (
            now
            if pd.isna(latest)
            else latest + pd.Timedelta(minutes=CAPTURE_CADENCE_MINUTES)
        )
        capture_due = bool(now >= next_due)
        funding = funding_lookup.get(asset, {})
        funding_complete = _text(funding.get("funding_status")) == "COMPLETE" and _truthy(
            funding.get("fetch_complete_flag")
        )
        latest_funding = pd.to_datetime(
            funding.get("latest_funding_at"), utc=True, errors="coerce"
        )
        if strict_ready:
            status = "STRICT_ROLLING_L2_READY"
            next_action = "maintain_rolling_l2_collection"
        elif provisional_ready:
            status = "PROVISIONAL_ROLLING_L2_READY"
            next_action = "continue_collection_until_strict_target"
        elif capture_due:
            status = "CAPTURE_DUE"
            next_action = "capture_public_hyperliquid_l2"
        else:
            status = "WAITING_FOR_NEXT_CAPTURE"
            next_action = "wait_until_next_capture_due"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "asset": asset,
                "pair_group_count": int(
                    (
                        selected_pairs["asset_x"].astype(str).str.upper().eq(asset)
                        | selected_pairs["asset_y"].astype(str).str.upper().eq(asset)
                    ).sum()
                ),
                "reference_leg_notional_usd": REFERENCE_LEG_NOTIONAL_USD,
                "rolling_strict_l2_samples": strict_count,
                "rolling_provisional_l2_samples": provisional_count,
                "strict_sample_deficit": max(0, MINIMUM_STRICT_L2_SAMPLES - strict_count),
                "provisional_sample_deficit": max(
                    0, MINIMUM_PROVISIONAL_L2_SAMPLES - provisional_count
                ),
                "rolling_strict_l2_ready": strict_ready,
                "rolling_provisional_l2_ready": provisional_ready,
                "latest_l2_sample_at": _timestamp_text(latest),
                "next_capture_due_at": _timestamp_text(next_due),
                "capture_due": capture_due,
                "funding_cache_complete": funding_complete,
                "funding_rows": _int_value(funding.get("funding_rows")),
                "latest_funding_at": _timestamp_text(latest_funding),
                "funding_refresh_required_at_next_cutoff": bool(
                    pd.isna(latest_funding) or latest_funding < now
                ),
                "rolling_collection_status": status,
                "next_action": next_action,
                "evidence_as_of": as_of.isoformat(),
                "evidence_path": ";".join(
                    (
                        _relative(evidence_paths["l2_samples"], root),
                        _relative(evidence_paths["funding_assets"], root),
                        _relative(evidence_paths["cost_manifest"], root),
                    )
                ),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rolling_strict_l2_ready", "strict_sample_deficit", "asset"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _build_pair_coverage(
    *,
    selected_pairs: pd.DataFrame,
    assets: pd.DataFrame,
    as_of: datetime,
    evidence_paths: dict[str, Path],
    root: Path,
) -> pd.DataFrame:
    lookup = {row.asset: row for row in assets.itertuples()}
    rows: list[dict[str, Any]] = []
    for row in selected_pairs.sort_values("pair_group_key").itertuples():
        asset_x = _text(row.asset_x).upper()
        asset_y = _text(row.asset_y).upper()
        x = lookup[asset_x]
        y = lookup[asset_y]
        strict_ready = bool(x.rolling_strict_l2_ready and y.rolling_strict_l2_ready)
        provisional_ready = bool(
            x.rolling_provisional_l2_ready and y.rolling_provisional_l2_ready
        )
        current_ready = _truthy(row.cost_acceptance_ready)
        if strict_ready:
            rolling_status = "STRICT_L2_READY_FOR_NEXT_WIZARD_CUTOFF"
            next_action = "maintain_collection_and_refresh_funding_at_next_cutoff"
        elif provisional_ready:
            rolling_status = "PROVISIONAL_L2_READY_FOR_NEXT_WIZARD_CUTOFF"
            next_action = "continue_collection_until_both_legs_are_strict"
        else:
            rolling_status = "COLLECTING_L2_FOR_NEXT_WIZARD_CUTOFF"
            next_action = "capture_due_assets_for_both_legs"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_group_key": _text(row.pair_group_key),
                "pair": _text(row.pair),
                "wizard_exchange": _text(row.wizard_exchange),
                "timeframe": _text(row.timeframe),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "current_snapshot_cost_status": _text(row.cost_evidence_status),
                "current_snapshot_cost_blocker": _text(row.cost_blocker),
                "current_snapshot_strict_l2_samples_x": _int_value(
                    row.strict_l2_samples_x
                ),
                "current_snapshot_strict_l2_samples_y": _int_value(
                    row.strict_l2_samples_y
                ),
                "current_snapshot_provisional_l2_samples_x": _int_value(
                    row.provisional_l2_samples_x
                ),
                "current_snapshot_provisional_l2_samples_y": _int_value(
                    row.provisional_l2_samples_y
                ),
                "current_snapshot_strict_cost_ready": current_ready,
                "current_snapshot_repairable_with_future_l2": False,
                "current_snapshot_cutoff_rule": (
                    "immutable_point_in_time_cutoff;post_cutoff_samples_excluded"
                ),
                "rolling_strict_l2_samples_x": int(x.rolling_strict_l2_samples),
                "rolling_strict_l2_samples_y": int(y.rolling_strict_l2_samples),
                "rolling_provisional_l2_samples_x": int(
                    x.rolling_provisional_l2_samples
                ),
                "rolling_provisional_l2_samples_y": int(
                    y.rolling_provisional_l2_samples
                ),
                "rolling_strict_l2_ready": strict_ready,
                "rolling_provisional_l2_ready": provisional_ready,
                "rolling_pair_status": rolling_status,
                "capture_due": bool(x.capture_due or y.capture_due),
                "next_capture_due_at": min(
                    value for value in (x.next_capture_due_at, y.next_capture_due_at) if value
                ),
                "next_action": next_action,
                "evidence_as_of": as_of.isoformat(),
                "evidence_path": ";".join(
                    (
                        _relative(evidence_paths["pair_costs"], root),
                        _relative(evidence_paths["l2_samples"], root),
                        _relative(evidence_paths["cost_manifest"], root),
                    )
                ),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["rolling_strict_l2_ready", "rolling_provisional_l2_ready", "pair_group_key"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def _build_funnel(
    *, failure: pd.DataFrame, root: Path, source_paths: dict[str, Path]
) -> pd.DataFrame:
    if failure["experiment_id"].astype(str).duplicated().any():
        raise ValueError("Failure attribution has duplicate experiment identities")
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    first_stage = failure["first_blocking_stage"].fillna("").astype(str)
    unknown = sorted(set(first_stage) - set(STAGES) - {""})
    if unknown:
        raise ValueError(f"Unknown first-blocking stages: {unknown}")
    first_index = first_stage.map(stage_order).fillna(len(STAGES)).astype(int)
    rows: list[dict[str, Any]] = []
    total = len(failure)
    for index, stage in enumerate(STAGES):
        reached = int(first_index.ge(index).sum())
        blocked = int(first_stage.eq(stage).sum())
        passed = int(first_index.gt(index).sum())
        blockers = failure.loc[first_stage.eq(stage), "first_blocker"].astype(str)
        top_blocker = blockers.value_counts().index[0] if not blockers.empty else ""
        if reached == 0:
            status = "NOT_REACHED"
        elif passed == reached:
            status = "PASS"
        elif passed == 0:
            status = "BLOCKED"
        else:
            status = "PARTIAL"
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "stage_order": index + 1,
                "stage": stage,
                "population_experiments": total,
                "reached_stage": reached,
                "passed_stage": passed,
                "blocked_at_stage": blocked,
                "not_reached": total - reached,
                "stage_conversion_rate": passed / reached if reached else 0.0,
                "cumulative_conversion_rate": passed / total if total else 0.0,
                "status": status,
                "top_blocker": top_blocker,
                "next_action": _stage_next_action(stage),
                "evidence_path": _relative(source_paths["failure_attribution"], root),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(rows)


def _build_blocker_summary(
    *, failure: pd.DataFrame, root: Path, source_paths: dict[str, Path]
) -> pd.DataFrame:
    grouped = (
        failure.groupby(["first_blocking_stage", "first_blocker"], dropna=False)
        .agg(
            experiments=("experiment_id", "nunique"),
            pair_groups=("pair_group_key", "nunique"),
            best_research_rank=("overall_research_rank", "min"),
        )
        .reset_index()
    )
    grouped["share_of_experiments"] = grouped["experiments"] / max(1, len(failure))
    grouped["next_action"] = grouped["first_blocking_stage"].map(_stage_next_action)
    grouped["evidence_path"] = _relative(source_paths["failure_attribution"], root)
    grouped["promotion_authority"] = False
    grouped["testnet_order_authority"] = False
    grouped["live_trading_authorized"] = False
    return grouped.sort_values(
        ["experiments", "best_research_rank"], ascending=[False, True]
    ).reset_index(drop=True)


def _build_pair_funnel(*, failure: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    stage_order = {stage: index for index, stage in enumerate(STAGES)}
    work = failure.copy()
    work["_stage_order"] = work["first_blocking_stage"].map(stage_order).fillna(len(STAGES))
    records: list[dict[str, Any]] = []
    coverage = {row.pair_group_key: row for row in pairs.itertuples()}
    for pair_group_key, group in work.groupby("pair_group_key", sort=True):
        earliest_order = int(group["_stage_order"].min())
        earliest_stage = STAGES[earliest_order] if earliest_order < len(STAGES) else "accepted"
        stage_rows = group.loc[group["_stage_order"].eq(earliest_order)]
        blocker_counts = stage_rows["first_blocker"].astype(str).value_counts()
        blocker = blocker_counts.index[0] if not blocker_counts.empty else ""
        coverage_row = coverage.get(_text(pair_group_key))
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "pair_group_key": _text(pair_group_key),
                "pair": _text(group.iloc[0].get("pair")),
                "wizard_exchange": _text(group.iloc[0].get("wizard_exchange")),
                "wizard_timeframe": _text(group.iloc[0].get("wizard_timeframe")),
                "asset_x": _text(group.iloc[0].get("asset_x")),
                "asset_y": _text(group.iloc[0].get("asset_y")),
                "experiments": int(group["experiment_id"].nunique()),
                "best_research_rank": _int_value(group["overall_research_rank"].min()),
                "first_blocking_stage": earliest_stage,
                "experiments_blocked_at_first_stage": len(stage_rows),
                "first_blocker": blocker,
                "selected_for_current_cost_evidence": coverage_row is not None,
                "current_snapshot_cost_status": (
                    coverage_row.current_snapshot_cost_status if coverage_row else "NOT_SELECTED"
                ),
                "rolling_pair_status": (
                    coverage_row.rolling_pair_status if coverage_row else "NOT_IN_COLLECTION_PLAN"
                ),
                "rolling_strict_l2_ready": bool(
                    coverage_row.rolling_strict_l2_ready if coverage_row else False
                ),
                "next_action": _stage_next_action(earliest_stage),
                "evidence_path": _text(group.iloc[0].get("evidence_path")),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["best_research_rank", "pair_group_key"]
    ).reset_index(drop=True)


def _collection_plan(pairs: pd.DataFrame) -> pd.DataFrame:
    plan = pairs[
        [
            "pair_group_key",
            "pair",
            "asset_x",
            "asset_y",
            "capture_due",
            "next_capture_due_at",
            "rolling_strict_l2_ready",
            "rolling_provisional_l2_ready",
            "rolling_pair_status",
            "next_action",
            "evidence_path",
            "promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        ]
    ].copy()
    plan.insert(0, "schema_version", SCHEMA_VERSION)
    plan["collection_eligible"] = True
    plan["collection_role"] = "prospective_next_wizard_cutoff_only"
    return plan.sort_values(
        ["capture_due", "rolling_strict_l2_ready", "pair_group_key"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _build_validation(
    *,
    selected_keys: set[str],
    selected_pairs: pd.DataFrame,
    assets: pd.DataFrame,
    pairs: pd.DataFrame,
    funnel: pd.DataFrame,
    failure: pd.DataFrame,
) -> pd.DataFrame:
    expected_assets = set(selected_pairs["asset_x"].astype(str).str.upper()) | set(
        selected_pairs["asset_y"].astype(str).str.upper()
    )
    authority_columns = (
        "promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    checks = [
        (
            "selected_pair_groups_accounted",
            set(pairs["pair_group_key"].astype(str)) == selected_keys,
            f"expected={len(selected_keys)};actual={len(pairs)}",
        ),
        (
            "selected_assets_accounted",
            set(assets["asset"].astype(str)) == expected_assets,
            f"expected={len(expected_assets)};actual={len(assets)}",
        ),
        (
            "failure_experiment_identity_unique",
            not failure["experiment_id"].astype(str).duplicated().any(),
            f"rows={len(failure)};unique={failure['experiment_id'].nunique()}",
        ),
        (
            "funnel_population_constant",
            funnel["population_experiments"].eq(len(failure)).all(),
            f"experiments={len(failure)}",
        ),
        (
            "funnel_first_blockers_accounted",
            int(funnel["blocked_at_stage"].sum())
            == int(failure["first_blocking_stage"].astype(str).ne("").sum()),
            (
                f"funnel={int(funnel['blocked_at_stage'].sum())};"
                f"failure={int(failure['first_blocking_stage'].astype(str).ne('').sum())}"
            ),
        ),
        (
            "future_l2_cannot_repair_current_snapshot",
            not pairs["current_snapshot_repairable_with_future_l2"].map(_truthy).any(),
            "post_cutoff_samples_are_prospective_only",
        ),
        (
            "no_execution_authority",
            all(
                not frame[column].map(_truthy).any()
                for frame in (assets, pairs, funnel)
                for column in authority_columns
            ),
            "research_collection_only",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": check,
                "status": "PASS" if passed else "FAIL",
                "detail": detail,
            }
            for check, passed, detail in checks
        ]
    )


def _normalize_l2_samples(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "asset",
        "notional_usd",
        "source_timestamp",
        "one_way_slippage_bps",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    work = frame.copy()
    work["asset"] = work["asset"].astype(str).str.upper()
    work["notional_usd"] = pd.to_numeric(work["notional_usd"], errors="coerce")
    work["source_timestamp"] = pd.to_datetime(
        work["source_timestamp"], utc=True, errors="coerce", format="mixed"
    )
    work["one_way_slippage_bps"] = pd.to_numeric(
        work["one_way_slippage_bps"], errors="coerce"
    )
    valid = (
        work["source_timestamp"].notna()
        & work["one_way_slippage_bps"].notna()
        & work["notional_usd"].sub(REFERENCE_LEG_NOTIONAL_USD).abs().le(1e-6)
    )
    if "blocker" in work:
        valid &= work["blocker"].fillna("").astype(str).eq("")
    for column in ("buy_complete", "sell_complete"):
        if column in work:
            valid &= work[column].map(_truthy)
    return work.loc[valid].drop_duplicates(
        ["asset", "notional_usd", "source_timestamp"], keep="last"
    )


def _stage_next_action(stage: str) -> str:
    return {
        "handoff": "repair_wizard_pair_detail_or_handoff_contract",
        "hyperliquid_mapping": "refresh_hyperliquid_inventory_and_symbol_mapping",
        "canonical_replay": "repair_point_in_time_history_or_replay_math",
        "cost_evidence": "maintain_pre_cutoff_l2_collection_and_refresh_funding",
        "observed_cost_replay": "run_observed_cost_replay",
        "walkforward": "run_purged_walkforward_validation",
        "regime": "run_regime_attribution",
        "robustness": "run_parameter_and_cost_stress",
        "concentration": "run_pair_timeframe_and_regime_concentration_checks",
        "leverage": "build_research_only_leverage_surface",
        "testnet_preflight": "validate_hyperliquid_testnet_preflight",
        "execution_acceptance": "retain_live_lock_until_all_acceptance_gates_pass",
        "accepted": "review_for_controlled_testnet_protocol",
    }.get(_text(stage), "review_unclassified_stage")


def _summary_markdown(
    *, summary: dict[str, Any], funnel: pd.DataFrame, assets: pd.DataFrame
) -> str:
    lines = [
        "# Current Wizard -> Hyperliquid Evidence Command Center",
        "",
        f"- Command center ID: `{summary['command_center_id']}`",
        f"- Evidence as of: `{summary['as_of']}`",
        f"- Selected current-board pairs: **{summary['current_snapshot_pair_groups']}**",
        f"- Required Hyperliquid assets: **{summary['rolling_assets']}**",
        (
            "- Rolling strict L2 coverage: "
            f"**{summary['rolling_strict_ready_assets']}/{summary['rolling_assets']} assets**, "
            f"**{summary['rolling_strict_ready_pairs']}/"
            f"{summary['current_snapshot_pair_groups']} pairs**"
        ),
        (
            "- Rolling provisional L2 coverage: "
            f"**{summary['rolling_provisional_ready_assets']}/"
            f"{summary['rolling_assets']} assets**, "
            f"**{summary['rolling_provisional_ready_pairs']}/"
            f"{summary['current_snapshot_pair_groups']} pairs**"
        ),
        f"- Assets due for capture: **{summary['capture_due_assets']}**",
        (
            "- Current frozen snapshot strict-cost-ready pairs: "
            f"**{summary['current_snapshot_strict_cost_ready_pairs']}**"
        ),
        f"- Acceptance-ready experiments: **{summary['acceptance_ready_experiments']}**",
        "- Order authority: **false**",
        "",
        "## Cutoff Rule",
        "",
        (
            "A book sample captured after a Wizard scanner cutoff cannot repair that frozen "
            "snapshot. Rolling collection is prospective evidence for the next snapshot."
        ),
        "",
        "## Funnel",
        "",
        "| Stage | Reached | Passed | Blocked Here | Conversion |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in funnel.itertuples():
        lines.append(
            f"| {row.stage} | {row.reached_stage} | {row.passed_stage} | "
            f"{row.blocked_at_stage} | {row.stage_conversion_rate:.1%} |"
        )
    due = assets.loc[assets["capture_due"].map(_truthy), "asset"].astype(str).tolist()
    lines.extend(
        [
            "",
            "## Immediate Action",
            "",
            (
                "Run the read-only L2 cadence for the current collection plan. "
                f"Due assets: {', '.join(due[:20])}{' ...' if len(due) > 20 else ''}"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _as_utc(value: datetime | None) -> datetime:
    result = value or datetime.now(UTC)
    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _int_value(value: Any) -> int:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0
    return int(parsed) if math.isfinite(parsed) else 0


def _timestamp_text(value: Any) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed.isoformat() if pd.notna(parsed) else ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _write_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    promote_staged_file(temporary, path)
