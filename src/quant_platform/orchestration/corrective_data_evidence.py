"""Corrective Hyperliquid mapping, source contracts, history, and cost evidence."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import atomic_write_text, promote_staged_file
from quant_platform.runtime_types import CommandResult

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_data_evidence.v1"
L2_TIMESTAMP_POLICY = "vendor_source_timestamp_else_local_capture_receipt_for_cadence_only"
PAIR_COST_BUNDLE_POINTER_SCHEMA_VERSION = "thewiz.l2_cost_model_pointer.v1"
PAIR_COST_BUNDLE_POINTER_NAME = "hyperliquid_pair_cost_bundle_pointer.json"
PAIR_COST_BUNDLE_INPUTS = {
    "candidate_set": ("candidate_set.csv", "candidate_set_sha256"),
    "cost_status": ("cost_status.csv", "cost_status_sha256"),
    "strict_l2_window": ("strict_l2_window.csv", "l2_samples_sha256"),
    "fee_profile": ("fee_profile.json", "fee_profile_sha256"),
}


SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "hyperliquid_inventory": {
        "required": [
            "asset",
            "asset_index",
            "universe_name",
            "sz_decimals",
            "max_leverage",
            "is_delisted",
            "tradable_perp",
            "checked_at_utc",
        ],
        "key": ["asset"],
        "timestamp": "checked_at_utc",
        "maximum_age_hours": 24.0,
        "minimum_rows": 1,
    },
    "wizard_mapping": {
        "required": [
            "pair_group_key",
            "pair",
            "asset_x",
            "asset_y",
            "current_pair_ready",
            "mapping_status",
            "current_inventory_checked_at",
        ],
        "key": ["pair_group_key"],
        "timestamp": "current_inventory_checked_at",
        "maximum_age_hours": 24.0,
        "minimum_rows": 1,
    },
    "hyperliquid_l2": {
        "required": [
            "sample_id",
            "asset",
            "notional_usd",
            "source_timestamp",
            "one_way_slippage_bps",
            "buy_complete",
            "sell_complete",
        ],
        "key": ["sample_id"],
        "timestamp": "source_timestamp",
        "maximum_age_hours": 24.0,
        "minimum_rows": 1,
    },
}


def validate_source_frame(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    *,
    now: datetime | None = None,
) -> list[str]:
    blockers: list[str] = []
    required = list(contract["required"])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        blockers.append("missing_columns:" + ",".join(missing))
        return blockers
    if len(frame) < int(contract.get("minimum_rows", 1)):
        blockers.append("row_truncation")
    keys = list(contract.get("key", []))
    if keys and frame.duplicated(keys).any():
        blockers.append("duplicate_keys")
    for column in required:
        null_ratio = float(frame[column].isna().mean()) if len(frame) else 1.0
        if null_ratio > 0.5:
            blockers.append(f"null_spike:{column}:{null_ratio:.3f}")
    timestamp_column = str(contract.get("timestamp", ""))
    if timestamp_column:
        raw_timestamps = frame[timestamp_column]
        populated = raw_timestamps.loc[
            raw_timestamps.notna() & raw_timestamps.astype(str).str.strip().ne("")
        ]
        timestamps = pd.to_datetime(populated, format="mixed", utc=True, errors="coerce")
        if not populated.empty and timestamps.isna().any():
            blockers.append("timestamp_type_drift_or_null")
        elif not timestamps.empty:
            cutoff = _as_utc(now) - pd.Timedelta(hours=float(contract["maximum_age_hours"]))
            if timestamps.max() < cutoff:
                blockers.append("source_stale")
    return list(dict.fromkeys(blockers))


def build_hyperliquid_market_manifest(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    active = root / "reports" / "active"
    mapping_path = active / "exhaustive_wizard_hyperliquid_mapping_refresh.csv"
    inventory_path = active / "hyperliquid_testnet_market_inventory.csv"
    mapping = _read_csv(mapping_path)
    inventory = _read_csv(inventory_path)
    mapping_blockers = validate_source_frame(mapping, SOURCE_CONTRACTS["wizard_mapping"], now=now)
    inventory_blockers = validate_source_frame(
        inventory, SOURCE_CONTRACTS["hyperliquid_inventory"], now=now
    )
    inventory_by_asset = {
        asset: group
        for asset, group in inventory.groupby(
            inventory.get("asset", pd.Series(dtype=str)).astype(str).str.upper()
        )
    }
    requested_assets: dict[str, set[str]] = {}
    for row in mapping.to_dict("records"):
        for column in ("asset_x", "asset_y"):
            asset = _text(row.get(column)).upper()
            if asset:
                requested_assets.setdefault(asset, set()).add(_text(row.get("pair_group_key")))
    rows: list[dict[str, Any]] = []
    for asset, pairs in sorted(requested_assets.items()):
        matches = inventory_by_asset.get(asset, pd.DataFrame())
        tradable = (
            matches.loc[
                matches.get("tradable_perp", pd.Series(False, index=matches.index)).map(_truthy)
            ]
            if not matches.empty
            else matches
        )
        if len(matches) > 1:
            classification = "alias_conflict"
            blocker = "multiple_hyperliquid_contracts_for_asset"
        elif matches.empty:
            classification = "structurally_unavailable"
            blocker = "hyperliquid_contract_missing"
        elif _truthy(matches.iloc[0].get("is_delisted")):
            classification = "delisted"
            blocker = "hyperliquid_contract_delisted"
        elif tradable.empty:
            classification = "structurally_unavailable"
            blocker = "hyperliquid_contract_not_tradable"
        elif asset != _text(matches.iloc[0].get("universe_name")).upper():
            classification = "exact_alias"
            blocker = ""
        else:
            classification = "supported"
            blocker = ""
        contract = matches.iloc[0] if len(matches) == 1 else pd.Series(dtype=object)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "asset": asset,
                "normalized_symbol": asset,
                "classification": classification,
                "requested_pair_count": len(pairs),
                "market_type": _text(contract.get("market_type")),
                "asset_index": contract.get("asset_index", pd.NA),
                "universe_name": _text(contract.get("universe_name")),
                "size_decimals": contract.get("sz_decimals", pd.NA),
                "maximum_leverage": contract.get("max_leverage", pd.NA),
                "margin_table_id": contract.get("margin_table_id", pd.NA),
                "only_isolated": _truthy(contract.get("only_isolated")),
                "is_delisted": _truthy(contract.get("is_delisted")),
                "tradable_perp": bool(not blocker),
                "source_timestamp": _text(contract.get("checked_at_utc")),
                "blocker": blocker,
                "evidence_path": _relative(inventory_path, root),
                "promotion_authority": False,
                "live_trading_authorized": False,
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = root / "data" / "processed" / "hyperliquid_market_manifest.csv"
    _atomic_csv(manifest, manifest_path)
    classifications = dict(zip(manifest.get("asset", []), manifest.get("classification", [])))
    pair_rows = []
    for row in mapping.to_dict("records"):
        x, y = _text(row.get("asset_x")).upper(), _text(row.get("asset_y")).upper()
        blockers = []
        for asset in (x, y):
            status = classifications.get(asset, "structurally_unavailable")
            if status not in {"supported", "exact_alias"}:
                blockers.append(f"{asset}:{status}")
        if mapping_blockers:
            blockers.extend(mapping_blockers)
        if inventory_blockers:
            blockers.extend(inventory_blockers)
        pair_rows.append(
            {
                "pair_group_key": _text(row.get("pair_group_key")),
                "pair": _text(row.get("pair")),
                "asset_x": x,
                "asset_y": y,
                "asset_x_classification": classifications.get(x, "structurally_unavailable"),
                "asset_y_classification": classifications.get(y, "structurally_unavailable"),
                "mapping_status": "PASS" if not blockers else "BLOCKED",
                "blocker": ";".join(dict.fromkeys(blockers)),
                "evidence_path": f"{_relative(mapping_path, root)};{_relative(manifest_path, root)}",
                "live_trading_authorized": False,
            }
        )
    audit = pd.DataFrame(pair_rows)
    audit_path = active / "hyperliquid_mapping_audit.csv"
    _atomic_csv(audit, audit_path)
    return {
        "manifest": manifest_path,
        "audit": audit_path,
        "assets": len(manifest),
        "pairs": len(audit),
        "mapping_passes": int(audit.get("mapping_status", pd.Series(dtype=str)).eq("PASS").sum()),
        "source_contract_blockers": mapping_blockers + inventory_blockers,
        "status": "PASS" if not mapping_blockers and not inventory_blockers else "BLOCKED",
        "live_trading_authorized": False,
    }


def build_external_source_contract_attacks(
    *, root: Path = ROOT, now: datetime | None = None
) -> pd.DataFrame:
    now = _as_utc(now)
    fixtures = root / "data" / "fixtures" / "hostile_source_responses"
    fixtures.mkdir(parents=True, exist_ok=True)
    base = pd.DataFrame(
        [
            {
                "asset": "BTC",
                "asset_index": 0,
                "universe_name": "BTC",
                "sz_decimals": 5,
                "max_leverage": 40,
                "is_delisted": False,
                "tradable_perp": True,
                "checked_at_utc": now.isoformat(),
            },
            {
                "asset": "ETH",
                "asset_index": 1,
                "universe_name": "ETH",
                "sz_decimals": 4,
                "max_leverage": 25,
                "is_delisted": False,
                "tradable_perp": True,
                "checked_at_utc": now.isoformat(),
            },
        ]
    )
    cases: dict[str, pd.DataFrame] = {
        "valid": base,
        "row_truncation": base.iloc[0:0],
        "pagination_loss": base.iloc[:1],
        "null_spike": base.assign(max_leverage=pd.NA),
        "type_drift": base.assign(checked_at_utc="not-a-timestamp"),
        "stale_timestamp": base.assign(checked_at_utc=(now - pd.Timedelta(days=3)).isoformat()),
        "duplicate_key": pd.concat([base, base.iloc[:1]], ignore_index=True),
        "missing_column": base.drop(columns=["tradable_perp"]),
    }
    rows = []
    contract = {**SOURCE_CONTRACTS["hyperliquid_inventory"], "minimum_rows": 2}
    for case, frame in cases.items():
        fixture_path = fixtures / f"hyperliquid_inventory_{case}.json"
        atomic_write_text(fixture_path, frame.to_json(orient="records", date_format="iso", indent=2) + "\n", encoding="utf-8")
        blockers = validate_source_frame(frame, contract, now=now)
        expected_pass = case == "valid"
        actual_pass = not blockers
        rows.append(
            {
                "case": case,
                "expected_status": "PASS" if expected_pass else "BLOCKED",
                "actual_status": "PASS" if actual_pass else "BLOCKED",
                "blocker": ";".join(blockers),
                "fixture_path": _relative(fixture_path, root),
                "current_board_publishable": actual_pass,
                "live_trading_authorized": False,
                "status": "PASS" if expected_pass == actual_pass else "FAIL",
            }
        )
    result = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "external_source_contract_results.csv"
    _atomic_csv(result, path)
    if not result["status"].eq("PASS").all():
        raise ValueError("external source contract attack failed")
    return result


def build_history_remediation(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "reports" / "active"
    history_path = active / "current_wizard_hyperliquid_pair_history_results.csv"
    history = _read_csv(history_path)
    failure = _read_csv(active / "current_wizard_hyperliquid_failure_attribution.csv")
    rank = (
        failure.groupby("pair_group_key", dropna=False)["overall_research_rank"]
        .min()
        .rename("best_research_rank")
        if not failure.empty
        else pd.Series(dtype=float, name="best_research_rank")
    )
    coverage = history.copy()
    coverage["aligned_history_ready"] = (
        coverage.get("history_status", pd.Series(dtype=str)).eq("READY_FOR_CANONICAL_1X_REPLAY")
        & coverage.get("timestamp_parse_valid", pd.Series(False, index=coverage.index)).map(_truthy)
        & coverage.get("timestamp_bound_valid", pd.Series(False, index=coverage.index)).map(_truthy)
        & pd.to_numeric(coverage.get("post_cutoff_rows", 0), errors="coerce").fillna(1).eq(0)
        & pd.to_numeric(coverage.get("history_rows", 0), errors="coerce")
        .fillna(0)
        .ge(
            pd.to_numeric(coverage.get("minimum_history_rows", 0), errors="coerce").fillna(math.inf)
        )
    )
    coverage = coverage.join(rank, on="pair_group_key")
    coverage["history_priority"] = coverage["best_research_rank"].fillna(10**9)
    coverage["history_remediation_reason"] = coverage.apply(_history_reason, axis=1)
    selected = coverage.get(
        "selected_for_materialization", pd.Series(False, index=coverage.index)
    ).map(_truthy)
    history_status = coverage.get("history_status", pd.Series("", index=coverage.index)).astype(str)
    history_blocker = (
        coverage.get("history_blocker", pd.Series("", index=coverage.index)).fillna("").astype(str)
    )
    insufficient_asset_age = history_blocker.str.contains(
        "insufficient_point_in_time_history", regex=False
    )
    coverage["history_coverage_class"] = "STRUCTURALLY_BLOCKED"
    coverage.loc[history_status.eq("DEFERRED_NOT_SELECTED"), "history_coverage_class"] = (
        "DEFERRED_NOT_SELECTED"
    )
    coverage.loc[selected & ~coverage["aligned_history_ready"], "history_coverage_class"] = (
        "ACTIVE_REMEDIATION"
    )
    coverage.loc[selected & insufficient_asset_age, "history_coverage_class"] = (
        "INSUFFICIENT_ASSET_AGE"
    )
    coverage.loc[coverage["aligned_history_ready"], "history_coverage_class"] = "READY"
    coverage["promotion_authority"] = False
    coverage["live_trading_authorized"] = False
    coverage_path = active / "hyperliquid_history_coverage.csv"
    _atomic_csv(coverage, coverage_path)
    queue = (
        coverage.loc[coverage["history_coverage_class"].eq("ACTIVE_REMEDIATION")]
        .sort_values(["history_priority", "pair_group_key"])
        .copy()
    )
    queue["next_action"] = "refetch_both_legs_to_declared_cutoff_then_validate_alignment"
    queue_path = active / "hyperliquid_history_remediation_queue.csv"
    _atomic_csv(queue, queue_path)
    return {
        "coverage": coverage_path,
        "queue": queue_path,
        "pairs": len(coverage),
        "ready": int(coverage["aligned_history_ready"].sum()),
        "queued": len(queue),
        "deferred": int(coverage["history_coverage_class"].eq("DEFERRED_NOT_SELECTED").sum()),
        "structurally_blocked": int(
            coverage["history_coverage_class"].eq("STRUCTURALLY_BLOCKED").sum()
        ),
        "insufficient_asset_age": int(
            coverage["history_coverage_class"].eq("INSUFFICIENT_ASSET_AGE").sum()
        ),
        "status": "PASS" if not coverage.empty else "BLOCKED",
        "live_trading_authorized": False,
    }


def build_l2_capture_candidate_set(*, root: Path = ROOT) -> dict[str, Any]:
    """Normalize moving discovery and frozen registered hypotheses into an L2 allow-list."""

    active = root / "reports" / "active"
    hypothesis_path = active / "current_hypothesis_batch.csv"
    attribution_path = active / "current_wizard_hyperliquid_failure_attribution.csv"
    attribution_route_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes.csv"
    )
    attribution_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    )
    attribution_route_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes_manifest.json"
    )
    exhaustive_replay_path = active / "exhaustive_wizard_hyperliquid_canonical_replay.csv"
    market_path = root / "data" / "processed" / "hyperliquid_market_context.csv"
    contract_path = active / "registered_research_rerun_contract.json"
    ledger_path = root / "data" / "research" / "hypothesis_ledger.jsonl"
    hypotheses = _read_csv(hypothesis_path)
    attribution, attribution_source_paths, routing_index_used = (
        _read_l2_failure_attribution_routes(
            root=root,
            attribution_path=attribution_path,
            route_path=attribution_route_path,
            source_manifest_path=attribution_manifest_path,
            route_manifest_path=attribution_route_manifest_path,
        )
    )
    markets = _read_csv(market_path)
    contract = _read_json(contract_path)
    columns = [
        "experiment_id",
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "overall_research_rank",
        "confirmation_role",
        "source_family",
        "semantic_hypothesis_id",
        "registered_contract_id",
        "registered_contract_candidate",
        "stage_two_candidate",
        "collection_eligible",
        "blocker",
        "evidence_path",
        "testnet_order_authority",
        "live_trading_authorized",
    ]
    if hypotheses.empty or attribution.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        required_hypotheses = {"experiment_id", "confirmation_role"}
        required_attribution = {
            "experiment_id",
            "pair_group_key",
            "pair",
            "asset_x",
            "asset_y",
            "overall_research_rank",
        }
        if not required_hypotheses.issubset(hypotheses.columns):
            raise ValueError("hypothesis batch is missing L2 candidate identity columns")
        if not required_attribution.issubset(attribution.columns):
            raise ValueError("failure attribution is missing L2 candidate routing columns")
        hypothesis_columns = ["experiment_id", "confirmation_role"]
        if "semantic_hypothesis_id" in hypotheses:
            hypothesis_columns.append("semantic_hypothesis_id")
        joined = hypotheses[hypothesis_columns].merge(
            attribution[list(required_attribution)],
            on="experiment_id",
            how="left",
            validate="one_to_one",
        )
        tradable_assets = set()
        if not markets.empty and {"asset", "tradable"}.issubset(markets.columns):
            tradable_assets = {
                _text(value).upper()
                for value in markets.loc[markets["tradable"].map(_truthy), "asset"]
            }
        rows = []
        for row in joined.to_dict("records"):
            asset_x = _text(row.get("asset_x")).upper()
            asset_y = _text(row.get("asset_y")).upper()
            blockers = []
            if not asset_x or not asset_y:
                blockers.append("normalized_pair_legs_missing")
            elif not {asset_x, asset_y}.issubset(tradable_assets):
                blockers.append("one_or_both_legs_not_currently_tradable_on_hyperliquid")
            rows.append(
                {
                    "experiment_id": _text(row.get("experiment_id")),
                    "pair_group_key": _text(row.get("pair_group_key")),
                    "pair": _text(row.get("pair")),
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "overall_research_rank": _finite(
                        row.get("overall_research_rank"), default=math.inf
                    ),
                    "confirmation_role": _text(row.get("confirmation_role")),
                    "source_family": "current_wizard_family",
                    "semantic_hypothesis_id": _text(row.get("semantic_hypothesis_id")),
                    "registered_contract_id": "",
                    "registered_contract_candidate": False,
                    "stage_two_candidate": False,
                    "collection_eligible": not blockers,
                    "blocker": ";".join(blockers),
                    "evidence_path": (
                        f"{_relative(hypothesis_path, root)};"
                        f"{';'.join(attribution_source_paths)};"
                        f"{_relative(market_path, root)}"
                    ),
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
        frame = pd.DataFrame(rows, columns=columns)
        frame = frame.sort_values(
            ["collection_eligible", "overall_research_rank", "experiment_id"],
            ascending=[False, True, True],
        )
        frame["_pair_identity"] = frame.apply(
            lambda row: "|".join(sorted((_text(row["asset_x"]), _text(row["asset_y"])))),
            axis=1,
        )
        frame = frame.drop_duplicates("_pair_identity", keep="first").drop(
            columns=["_pair_identity"]
        )
    registered_rows = _registered_contract_l2_rows(
        root=root,
        contract=contract,
        contract_path=contract_path,
        ledger_path=ledger_path,
        market_path=market_path,
        markets=markets,
    )
    if registered_rows:
        frame = pd.concat(
            [frame, pd.DataFrame(registered_rows, columns=columns)],
            ignore_index=True,
            sort=False,
        )
    elif not frame.empty:
        # Before a frozen rerun contract exists, the current registered batch is
        # the Stage 2 collection authority. Once a contract exists, only its
        # immutable candidates control Stage 2 acceptance.
        frame["stage_two_candidate"] = True
    current_board_rows = _current_wizard_board_l2_rows(
        root=root,
        market_path=market_path,
        markets=markets,
    )
    if current_board_rows:
        frame = pd.concat(
            [frame, pd.DataFrame(current_board_rows, columns=columns)],
            ignore_index=True,
            sort=False,
        )
    exhaustive = _read_csv(exhaustive_replay_path)
    historical_diagnostic_rows: list[dict[str, Any]] = []
    exhaustive_required = {
        "experiment_id",
        "pair_group_id",
        "pair",
        "asset_x",
        "asset_y",
        "research_rank_eligible",
    }
    if not exhaustive.empty and exhaustive_required.issubset(exhaustive.columns):
        eligible = exhaustive.loc[exhaustive["research_rank_eligible"].map(_truthy)].copy()
        for metric in ("profit_factor", "sharpe", "total_return"):
            if metric in eligible.columns:
                eligible[metric] = pd.to_numeric(eligible[metric], errors="coerce")
        sort_columns = [
            metric
            for metric in ("profit_factor", "sharpe", "total_return")
            if metric in eligible.columns
        ]
        if sort_columns:
            eligible = eligible.sort_values(
                sort_columns,
                ascending=[False] * len(sort_columns),
                na_position="last",
            )
        tradable_assets = set()
        if not markets.empty and {"asset", "tradable"}.issubset(markets.columns):
            tradable_assets = {
                _text(value).upper()
                for value in markets.loc[markets["tradable"].map(_truthy), "asset"]
            }
        for rank, row in enumerate(eligible.to_dict("records"), start=1):
            asset_x = _text(row.get("asset_x")).upper()
            asset_y = _text(row.get("asset_y")).upper()
            blockers = ["historical_exhaustive_family_not_current_registration_authority"]
            if not asset_x or not asset_y:
                blockers.append("normalized_pair_legs_missing")
            elif not {asset_x, asset_y}.issubset(tradable_assets):
                blockers.append("one_or_both_legs_not_currently_tradable_on_hyperliquid")
            historical_diagnostic_rows.append(
                {
                    "experiment_id": _text(row.get("experiment_id")),
                    "pair_group_key": _text(row.get("pair_group_id")),
                    "pair": _text(row.get("pair")),
                    "asset_x": asset_x,
                    "asset_y": asset_y,
                    "overall_research_rank": 1_000 + rank,
                    "confirmation_role": "exhaustive_research_rank_eligible",
                    "source_family": "historical_exhaustive_family",
                    "semantic_hypothesis_id": "",
                    "registered_contract_id": "",
                    "registered_contract_candidate": False,
                    "stage_two_candidate": False,
                    "collection_eligible": False,
                    "blocker": ";".join(blockers),
                    "evidence_path": (
                        f"{_relative(exhaustive_replay_path, root)};{_relative(market_path, root)}"
                    ),
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    if not frame.empty:
        frame["_pair_identity"] = frame.apply(
            lambda row: "|".join(sorted((_text(row["asset_x"]), _text(row["asset_y"])))),
            axis=1,
        )
        frame = (
            frame.sort_values(
                [
                    "stage_two_candidate",
                    "registered_contract_candidate",
                    "collection_eligible",
                    "overall_research_rank",
                    "experiment_id",
                ],
                ascending=[False, False, False, True, True],
            )
            .drop_duplicates("_pair_identity", keep="first")
            .drop(columns=["_pair_identity"])
            .reset_index(drop=True)
        )
    path = active / "corrective_l2_capture_candidates.csv"
    _atomic_csv(frame, path)
    historical_path = active / "corrective_l2_historical_diagnostics.csv"
    historical = pd.DataFrame(historical_diagnostic_rows, columns=columns)
    if not historical.empty:
        historical["_pair_identity"] = historical.apply(
            lambda row: "|".join(sorted((_text(row["asset_x"]), _text(row["asset_y"])))),
            axis=1,
        )
        historical = (
            historical.sort_values(["overall_research_rank", "experiment_id"])
            .drop_duplicates("_pair_identity", keep="first")
            .drop(columns=["_pair_identity"])
            .reset_index(drop=True)
        )
    _atomic_csv(historical, historical_path)
    return {
        "path": path,
        "historical_diagnostics_path": historical_path,
        "registered_hypotheses": len(registered_rows) if contract else len(hypotheses),
        "registered_contract_candidates": len(registered_rows),
        "current_board_collection_pairs": len(current_board_rows),
        "candidate_pairs": len(frame),
        "historical_diagnostic_pairs": len(historical),
        "eligible_pairs": int(
            frame.get("collection_eligible", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "stage_two_candidate_pairs": int(
            frame.get("stage_two_candidate", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "status": "PASS"
        if frame.get("collection_eligible", pd.Series(dtype=bool)).map(_truthy).any()
        else "BLOCKED",
        "routing_index_used": routing_index_used,
        "routing_source_paths": ";".join(attribution_source_paths),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _current_wizard_board_l2_rows(
    *,
    root: Path,
    market_path: Path,
    markets: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Expose the frozen board's selected pairs to prospective L2 collection.

    These rows expand collection coverage only. They never become registered
    hypotheses, Stage 2 candidates, promotion evidence, or order authority.
    Fresh samples are intended for a later Wizard cutoff and cannot repair the
    already-frozen board that selected the pairs.
    """

    active = root / "reports" / "active"
    manifest_path = active / "current_wizard_hyperliquid_cost_manifest.json"
    manifest = _read_json(manifest_path)
    if not manifest:
        return []
    cost_evidence_id = _text(manifest.get("cost_evidence_id"))
    selected = {
        _text(value)
        for value in manifest.get("selected_pair_group_keys", [])
        if _text(value)
    }
    artifacts = manifest.get("artifacts", {})
    if not cost_evidence_id or not selected or not isinstance(artifacts, dict):
        return []
    pairs_relative = _text(artifacts.get("snapshot_pairs") or artifacts.get("pairs"))
    pairs_path = root / pairs_relative
    pairs = _read_csv(pairs_path)
    required = {
        "cost_evidence_id",
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "selected_for_cost_evidence",
    }
    if pairs.empty or not required.issubset(pairs.columns):
        return []
    if set(pairs["cost_evidence_id"].astype(str)) != {cost_evidence_id}:
        raise ValueError("current Wizard pair-cost evidence identity mismatch")
    selected_rows = pairs.loc[
        pairs["selected_for_cost_evidence"].map(_truthy)
        & pairs["pair_group_key"].astype(str).isin(selected)
    ].copy()
    if set(selected_rows["pair_group_key"].astype(str)) != selected:
        raise ValueError("current Wizard selected pair-cost set is incomplete")
    tradable_assets = set()
    if not markets.empty and {"asset", "tradable"}.issubset(markets.columns):
        tradable_assets = {
            _text(value).upper()
            for value in markets.loc[markets["tradable"].map(_truthy), "asset"]
        }
    rows: list[dict[str, Any]] = []
    for rank, row in enumerate(
        selected_rows.sort_values(["pair_group_key"]).to_dict("records"), start=1
    ):
        asset_x = _text(row.get("asset_x")).upper()
        asset_y = _text(row.get("asset_y")).upper()
        blockers: list[str] = []
        if not asset_x or not asset_y:
            blockers.append("normalized_pair_legs_missing")
        elif not {asset_x, asset_y}.issubset(tradable_assets):
            blockers.append("one_or_both_legs_not_currently_tradable_on_hyperliquid")
        pair_group_key = _text(row.get("pair_group_key"))
        rows.append(
            {
                "experiment_id": f"prospective_collection::{pair_group_key}",
                "pair_group_key": pair_group_key,
                "pair": _text(row.get("pair")),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "overall_research_rank": 100_000 + rank,
                "confirmation_role": "prospective_next_wizard_cutoff_cost_coverage",
                "source_family": "current_wizard_cost_selected_collection",
                "semantic_hypothesis_id": "",
                "registered_contract_id": "",
                "registered_contract_candidate": False,
                "stage_two_candidate": False,
                "collection_eligible": not blockers,
                "blocker": ";".join(blockers),
                "evidence_path": ";".join(
                    (
                        _relative(manifest_path, root),
                        _relative(pairs_path, root),
                        _relative(market_path, root),
                    )
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return rows


def _read_l2_failure_attribution_routes(
    *,
    root: Path,
    attribution_path: Path,
    route_path: Path,
    source_manifest_path: Path,
    route_manifest_path: Path,
) -> tuple[pd.DataFrame, list[str], bool]:
    """Use the compact hash-bound route index when the producer published one."""

    source_manifest = _read_json(source_manifest_path)
    route_manifest = _read_json(route_manifest_path)
    route_relative = _relative(route_path, root)
    if route_manifest:
        advertised_route = _text(route_manifest.get("active_route_index_path"))
        expected_hash = _text(route_manifest.get("route_index_sha256"))
        source_id = _text(source_manifest.get("failure_attribution_id"))
        expected_source_id = _text(
            route_manifest.get("source_failure_attribution_id")
        )
        expected_source_manifest_hash = _text(
            route_manifest.get("source_manifest_sha256")
        )
        if (
            advertised_route != route_relative
            or not expected_hash
            or not source_id
            or expected_source_id != source_id
            or expected_source_manifest_hash != _file_sha256(source_manifest_path)
        ):
            raise ValueError("failure attribution route-index manifest is incomplete")
        if not route_path.is_file() or _file_sha256(route_path) != expected_hash:
            raise ValueError("failure attribution route-index hash mismatch")
        routes = _read_csv(route_path)
        required = {
            "experiment_id",
            "pair_group_key",
            "pair",
            "asset_x",
            "asset_y",
            "overall_research_rank",
        }
        if not required.issubset(routes.columns):
            raise ValueError("failure attribution route index is missing routing columns")
        experiment_ids = routes["experiment_id"].astype(str)
        if experiment_ids.eq("").any() or experiment_ids.duplicated().any():
            raise ValueError("failure attribution route index identity is invalid")
        return (
            routes,
            [
                route_relative,
                _relative(route_manifest_path, root),
                _relative(source_manifest_path, root),
            ],
            True,
        )
    return _read_csv(attribution_path), [_relative(attribution_path, root)], False


def _registered_contract_l2_rows(
    *,
    root: Path,
    contract: dict[str, Any],
    contract_path: Path,
    ledger_path: Path,
    market_path: Path,
    markets: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not contract:
        return []
    immutable_rel = _text(contract.get("immutable_contract_path"))
    immutable_path = root / immutable_rel
    if not immutable_rel or _read_json(immutable_path) != contract:
        raise ValueError("registered rerun contract immutable copy mismatch")
    contract_id = _text(contract.get("contract_id"))
    candidates = contract.get("registered_candidates", [])
    if not contract_id or not isinstance(candidates, list) or not candidates:
        raise ValueError("registered rerun contract candidates are missing")
    ledger = _read_jsonl(ledger_path)
    tradable_assets = set()
    if not markets.empty and {"asset", "tradable"}.issubset(markets.columns):
        tradable_assets = {
            _text(value).upper()
            for value in markets.loc[markets["tradable"].map(_truthy), "asset"]
        }
    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise TypeError("registered rerun contract candidate is invalid")
        semantic_id = _text(candidate.get("semantic_hypothesis_id"))
        record_hash = _text(candidate.get("ledger_record_hash"))
        matches = [
            record
            for record in ledger
            if _text(record.get("semantic_hypothesis_id")) == semantic_id
            and _text(record.get("record_hash")) == record_hash
        ]
        blockers: list[str] = []
        material: dict[str, Any] = {}
        if len(matches) != 1:
            blockers.append("registered_contract_ledger_record_missing_or_ambiguous")
        else:
            raw_material = matches[0].get("semantic_material", {})
            if isinstance(raw_material, dict):
                material = raw_material
            if _text(matches[0].get("source_experiment_id")) != _text(
                candidate.get("source_experiment_id")
            ):
                blockers.append("registered_contract_experiment_identity_mismatch")
        assets = material.get("assets", [])
        if not isinstance(assets, list) or len(assets) != 2:
            blockers.append("registered_contract_assets_missing")
            asset_x = ""
            asset_y = ""
        else:
            asset_x, asset_y = (_text(asset).upper() for asset in assets)
        if _text(material.get("execution_venue")).lower() != "hyperliquid":
            blockers.append("registered_contract_execution_venue_not_hyperliquid")
        if not asset_x or not asset_y:
            blockers.append("normalized_pair_legs_missing")
        elif not {asset_x, asset_y}.issubset(tradable_assets):
            blockers.append("one_or_both_legs_not_currently_tradable_on_hyperliquid")
        rows.append(
            {
                "experiment_id": _text(candidate.get("source_experiment_id")),
                "pair_group_key": _text(candidate.get("pair_group_key")),
                "pair": _text(candidate.get("pair")),
                "asset_x": asset_x,
                "asset_y": asset_y,
                "overall_research_rank": rank,
                "confirmation_role": "frozen_registered_rerun_contract",
                "source_family": "registered_rerun_contract",
                "semantic_hypothesis_id": semantic_id,
                "registered_contract_id": contract_id,
                "registered_contract_candidate": True,
                "stage_two_candidate": True,
                "collection_eligible": not blockers,
                "blocker": ";".join(dict.fromkeys(blockers)),
                "evidence_path": ";".join(
                    (
                        _relative(contract_path, root),
                        _relative(immutable_path, root),
                        _relative(ledger_path, root),
                        _relative(market_path, root),
                    )
                ),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    return rows


def build_cost_collection_status(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    now = _as_utc(now)
    active = root / "reports" / "active"
    policy = _cost_gate_policy(root)
    strict_window_hours = float(policy.get("strict_l2_window_hours", 2.0))
    minimum_samples = int(policy.get("minimum_strict_l2_samples", 12))
    minimum_span_minutes = float(policy.get("minimum_strict_l2_span_minutes", 100.0))
    maximum_gap_minutes = float(policy.get("maximum_strict_l2_gap_minutes", 15.0))
    current_funding_path = active / "current_wizard_hyperliquid_funding_asset_results.csv"
    exhaustive_funding_path = active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv"
    corrective_l2_funding_path = active / "corrective_l2_funding_asset_results.csv"
    current_funding = _read_csv(current_funding_path).assign(
        _funding_source_artifact=_relative(current_funding_path, root)
    )
    exhaustive_funding = _read_csv(exhaustive_funding_path).assign(
        _funding_source_artifact=_relative(exhaustive_funding_path, root)
    )
    corrective_l2_funding = _read_csv(corrective_l2_funding_path).assign(
        _funding_source_artifact=_relative(corrective_l2_funding_path, root)
    )
    funding = pd.concat(
        [current_funding, exhaustive_funding, corrective_l2_funding],
        ignore_index=True,
        sort=False,
    )
    l2 = _read_csv(root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv")
    # Exhaustive evidence may close a funding sub-gate for an active asset, but it
    # must not silently widen the active cost-collection universe.
    assets = sorted(
        set(current_funding.get("asset", pd.Series(dtype=str)).astype(str))
        | set(l2.get("asset", pd.Series(dtype=str)).astype(str))
    )
    rows = []
    for asset in assets:
        f = funding.loc[funding.get("asset", pd.Series(dtype=str)).astype(str).eq(asset)]
        s = l2.loc[l2.get("asset", pd.Series(dtype=str)).astype(str).eq(asset)]
        observations = _complete_l2_observations(s).drop_duplicates(
            "effective_l2_observation_timestamp", keep="last"
        )
        observations = observations.loc[
            observations["effective_l2_observation_timestamp"] <= pd.Timestamp(now)
        ]
        strict_observations = observations.loc[
            observations["effective_l2_observation_timestamp"]
            >= pd.Timestamp(now) - pd.Timedelta(hours=strict_window_hours)
        ]
        provisional_observations = observations.loc[
            observations["effective_l2_observation_timestamp"]
            >= pd.Timestamp(now) - pd.Timedelta(hours=24)
        ]
        strict_timestamps = strict_observations["effective_l2_observation_timestamp"]
        strict = len(strict_observations)
        provisional = len(provisional_observations)
        strict_local_fallbacks = int(
            strict_observations["l2_local_capture_timestamp_fallback"].sum()
        )
        provisional_local_fallbacks = int(
            provisional_observations["l2_local_capture_timestamp_fallback"].sum()
        )
        vendor_timestamps = observations["source_timestamp"].dropna()
        span_minutes = (
            float((strict_timestamps.max() - strict_timestamps.min()).total_seconds() / 60.0)
            if len(strict_timestamps) >= 2
            else 0.0
        )
        ordered_timestamps = strict_timestamps.sort_values()
        observation_gaps = ordered_timestamps.diff().dropna().dt.total_seconds() / 60.0
        max_gap_minutes = (
            float(observation_gaps.max()) if not observation_gaps.empty else math.inf
        )
        latest_age_minutes = (
            float((pd.Timestamp(now) - ordered_timestamps.max()).total_seconds() / 60.0)
            if not ordered_timestamps.empty
            else math.inf
        )
        cadence_ready = bool(
            strict >= minimum_samples
            and span_minutes >= minimum_span_minutes
            and max_gap_minutes <= maximum_gap_minutes
            and latest_age_minutes <= maximum_gap_minutes
        )
        funding_complete_rows = f.get("funding_status", pd.Series("", index=f.index, dtype=str)).eq(
            "COMPLETE"
        )
        if "fetch_complete_flag" in f:
            fetch_complete = f["fetch_complete_flag"]
            funding_complete_rows &= fetch_complete.isna() | fetch_complete.map(_truthy)
        if "timestamp_parse_valid" in f:
            timestamps_valid = f["timestamp_parse_valid"]
            funding_complete_rows &= timestamps_valid.isna() | timestamps_valid.map(_truthy)
        if "post_cutoff_rows" in f:
            post_cutoff_rows = pd.to_numeric(f["post_cutoff_rows"], errors="coerce").fillna(0)
            funding_complete_rows &= post_cutoff_rows.eq(0)
        complete_funding = f.loc[funding_complete_rows]
        funding_complete = not complete_funding.empty
        funding_rows = (
            int(
                pd.to_numeric(complete_funding.get("funding_rows", 0), errors="coerce")
                .fillna(0)
                .max()
            )
            if funding_complete
            else 0
        )
        funding_evidence_paths = sorted(
            {
                _text(value)
                for value in complete_funding.get("_funding_source_artifact", pd.Series(dtype=str))
                if _text(value)
            }
        )
        funding_detail_paths = sorted(
            {
                _text(value)
                for value in complete_funding.get("funding_path", pd.Series(dtype=str))
                if _text(value)
            }
        )
        blockers = []
        if not funding_complete:
            blockers.append("funding_incomplete")
        if strict < minimum_samples:
            blockers.append("strict_l2_sample_target_not_met")
        elif span_minutes < minimum_span_minutes:
            blockers.append("strict_l2_observation_span_not_met")
        if strict >= 2 and max_gap_minutes > maximum_gap_minutes:
            blockers.append("strict_l2_maximum_gap_exceeded")
        if strict >= 1 and latest_age_minutes > maximum_gap_minutes:
            blockers.append("strict_l2_latest_observation_stale")
        rows.append(
            {
                "asset": asset,
                "funding_rows": funding_rows,
                "funding_complete": funding_complete,
                "funding_complete_source_count": len(funding_evidence_paths),
                "funding_evidence_path": ";".join(funding_evidence_paths),
                "funding_detail_path": ";".join(funding_detail_paths),
                "strict_l2_samples": strict,
                "provisional_l2_samples": provisional,
                "strict_l2_local_capture_timestamp_fallbacks": strict_local_fallbacks,
                "provisional_l2_local_capture_timestamp_fallbacks": provisional_local_fallbacks,
                "strict_l2_span_minutes": span_minutes,
                "strict_l2_max_gap_minutes": (
                    max_gap_minutes if math.isfinite(max_gap_minutes) else ""
                ),
                "strict_l2_latest_age_minutes": (
                    latest_age_minutes if math.isfinite(latest_age_minutes) else ""
                ),
                "strict_l2_cadence_ready": cadence_ready,
                "minimum_strict_l2_samples": minimum_samples,
                "minimum_strict_l2_span_minutes": minimum_span_minutes,
                "maximum_strict_l2_gap_minutes": maximum_gap_minutes,
                "l2_timestamp_policy": L2_TIMESTAMP_POLICY,
                "latest_l2_at": (
                    observations["effective_l2_observation_timestamp"].max().isoformat()
                    if not observations.empty
                    else ""
                ),
                "latest_l2_vendor_at": (
                    vendor_timestamps.max().isoformat() if not vendor_timestamps.empty else ""
                ),
                "collection_status": "READY" if not blockers else "COLLECTING",
                "blocker": ";".join(blockers),
                "next_action": "continue_2h_l2_and_funding_collection"
                if blockers
                else "maintain_rolling_collection",
                "live_trading_authorized": False,
            }
        )
    frame = pd.DataFrame(rows)
    path = active / "hyperliquid_cost_collection_status.csv"
    _atomic_csv(frame, path)
    return {
        "status_path": path,
        "assets": len(frame),
        "ready_assets": int(frame.get("collection_status", pd.Series(dtype=str)).eq("READY").sum()),
        "collecting_assets": int(
            frame.get("collection_status", pd.Series(dtype=str)).eq("COLLECTING").sum()
        ),
        "status": "PASS" if not frame.empty else "BLOCKED",
        "live_trading_authorized": False,
    }


def build_pair_cost_stress_surfaces(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    """Build strict cost models for every registered L2 candidate.

    Candidate identity comes from the unified corrective candidate set, so a pair
    discovered by the exhaustive lane cannot disappear merely because it is absent
    from the latest current-Wizard cost table. The model remains research-only and
    is rebuilt from the rolling public L2 window and explicit funding provenance.
    """

    now = _as_utc(now)
    active = root / "reports" / "active"
    candidate_path = active / "corrective_l2_capture_candidates.csv"
    status_path = active / "hyperliquid_cost_collection_status.csv"
    l2_path = root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv"
    profile_path = root / "config" / "hyperliquid_perp_cost_profile.json"
    candidates = _read_csv(candidate_path)
    candidates = candidates.loc[
        candidates.get("collection_eligible", pd.Series(False, index=candidates.index)).map(_truthy)
    ].copy()
    if "source_family" in candidates:
        candidates = candidates.loc[
            candidates["source_family"].astype(str).ne(
                "current_wizard_cost_selected_collection"
            )
        ].copy()
    statuses = _read_csv(status_path)
    status_by_asset = {
        _text(row.get("asset")).upper(): row
        for row in statuses.to_dict("records")
        if _text(row.get("asset"))
    }
    l2 = _strict_l2_window(_read_csv(l2_path), now=now, root=root)
    policy = _cost_gate_policy(root)
    minimum_samples = int(policy.get("minimum_strict_l2_samples", 12))
    minimum_span_minutes = float(policy.get("minimum_strict_l2_span_minutes", 100.0))
    maximum_gap_minutes = float(policy.get("maximum_strict_l2_gap_minutes", 15.0))
    profile = _read_json(profile_path)
    fee = _finite(profile.get("taker_fee_bps"))
    execution_risk = _finite(profile.get("execution_risk_bps"))
    fee_checked = pd.to_datetime(profile.get("fee_source_checked_at"), utc=True, errors="coerce")
    fee_fresh = bool(
        math.isfinite(fee)
        and fee >= 0
        and math.isfinite(execution_risk)
        and execution_risk >= 0
        and _text(profile.get("fee_source_url"))
        and pd.notna(fee_checked)
        and fee_checked <= pd.Timestamp(now)
        and pd.Timestamp(now) - fee_checked <= pd.Timedelta(days=30)
    )
    candidate_bytes = candidate_path.read_bytes() if candidate_path.is_file() else b""
    cost_status_bytes = status_path.read_bytes() if status_path.is_file() else b""
    fee_profile_bytes = profile_path.read_bytes() if profile_path.is_file() else b""
    source_hashes = {
        "candidate_set_sha256": sha256(candidate_bytes).hexdigest(),
        "cost_status_sha256": sha256(cost_status_bytes).hexdigest(),
        "l2_samples_sha256": sha256(_csv_bytes(l2)).hexdigest(),
        "fee_profile_sha256": sha256(fee_profile_bytes).hexdigest(),
        "l2_source_file_sha256": _file_sha256(l2_path),
    }
    bundle_material = {"model_as_of_utc": now.isoformat(), **source_hashes}
    bundle_id = "l2costbundle_" + sha256(
        json.dumps(bundle_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    bundle_dir = root / "data" / "research" / "l2_cost_model_receipts" / bundle_id
    bundle_inputs = {
        "candidate_set": bundle_dir / "candidate_set.csv",
        "cost_status": bundle_dir / "cost_status.csv",
        "strict_l2_window": bundle_dir / "strict_l2_window.csv",
        "fee_profile": bundle_dir / "fee_profile.json",
    }
    _write_or_validate_immutable_bytes(candidate_bytes, bundle_inputs["candidate_set"])
    _write_or_validate_immutable_bytes(cost_status_bytes, bundle_inputs["cost_status"])
    _write_or_validate_immutable_bytes(_csv_bytes(l2), bundle_inputs["strict_l2_window"])
    _write_or_validate_immutable_bytes(fee_profile_bytes, bundle_inputs["fee_profile"])
    input_hashes = {
        "candidate_set_sha256": _file_sha256(bundle_inputs["candidate_set"]),
        "cost_status_sha256": _file_sha256(bundle_inputs["cost_status"]),
        "l2_samples_sha256": _file_sha256(bundle_inputs["strict_l2_window"]),
        "fee_profile_sha256": _file_sha256(bundle_inputs["fee_profile"]),
        "l2_source_file_sha256": source_hashes["l2_source_file_sha256"],
    }
    model_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dict("records"):
        asset_x = _text(candidate.get("asset_x")).upper()
        asset_y = _text(candidate.get("asset_y")).upper()
        status_x = status_by_asset.get(asset_x, {})
        status_y = status_by_asset.get(asset_y, {})
        stats_x = _strict_asset_cost_statistics(l2, asset_x)
        stats_y = _strict_asset_cost_statistics(l2, asset_y)
        blockers: list[str] = []
        for label, asset, status, stats in (
            ("x", asset_x, status_x, stats_x),
            ("y", asset_y, status_y, stats_y),
        ):
            if not status:
                blockers.append(f"asset_{label}_cost_status_missing:{asset}")
                continue
            if not _truthy(status.get("funding_complete")):
                blockers.append(f"asset_{label}_funding_incomplete:{asset}")
            if not _truthy(status.get("strict_l2_cadence_ready")):
                blockers.append(f"asset_{label}_strict_l2_not_ready:{asset}")
            if _text(status.get("collection_status")) != "READY":
                blockers.append(f"asset_{label}_cost_collection_not_ready:{asset}")
            if not stats["statistics_complete"]:
                blockers.append(f"asset_{label}_strict_l2_statistics_incomplete:{asset}")
            if stats["samples"] < minimum_samples:
                blockers.append(f"asset_{label}_strict_l2_sample_target_not_met:{asset}")
            if stats["span_minutes"] < minimum_span_minutes:
                blockers.append(f"asset_{label}_strict_l2_span_not_met:{asset}")
            if stats["max_gap_minutes"] > maximum_gap_minutes:
                blockers.append(f"asset_{label}_strict_l2_maximum_gap_exceeded:{asset}")
            if (
                not math.isfinite(stats["available_notional_p05_usd"])
                or stats["available_notional_p05_usd"] < 1_000.0
            ):
                blockers.append(f"asset_{label}_depth_below_test_notional:{asset}")
        if not fee_fresh:
            blockers.append("fee_profile_missing_stale_or_invalid")
        p95_x = stats_x["slippage_p95_bps"]
        p95_y = stats_y["slippage_p95_bps"]
        pair_slippage = (
            (p95_x + p95_y) / 2.0 if math.isfinite(p95_x) and math.isfinite(p95_y) else math.nan
        )
        round_trip = (
            2.0 * (fee + pair_slippage + execution_risk)
            if fee_fresh and math.isfinite(pair_slippage)
            else math.nan
        )
        ready = not blockers and math.isfinite(round_trip)
        evidence_paths = [
            _relative(candidate_path, root),
            _relative(status_path, root),
            _relative(l2_path, root),
            _relative(profile_path, root),
            *(_relative(path, root) for path in bundle_inputs.values()),
            _text(status_x.get("funding_evidence_path")),
            _text(status_y.get("funding_evidence_path")),
            _text(status_x.get("funding_detail_path")),
            _text(status_y.get("funding_detail_path")),
        ]
        identity_payload = {
            "pair_group_key": _text(candidate.get("pair_group_key")),
            "model_as_of_utc": now.isoformat(),
            **input_hashes,
        }
        model_rows.append(
            {
                "schema_version": "thewiz.registered_pair_cost_model.v1",
                "cost_model_id": "hlpaircost_"
                + sha256(json.dumps(identity_payload, sort_keys=True).encode("utf-8")).hexdigest()[
                    :20
                ],
                "model_as_of_utc": now.isoformat(),
                "pair_group_key": _text(candidate.get("pair_group_key")),
                "pair": _text(candidate.get("pair")),
                "execution_venue": "hyperliquid",
                "instrument": "perpetual",
                "asset_x": asset_x,
                "asset_y": asset_y,
                "confirmation_role": _text(candidate.get("confirmation_role")),
                "source_family": _text(candidate.get("source_family")),
                "semantic_hypothesis_id": _text(candidate.get("semantic_hypothesis_id")),
                "registered_contract_id": _text(candidate.get("registered_contract_id")),
                "registered_contract_candidate": _truthy(
                    candidate.get("registered_contract_candidate")
                ),
                "stage_two_candidate": _truthy(
                    candidate.get("stage_two_candidate", True)
                ),
                "overall_research_rank": candidate.get("overall_research_rank"),
                "selected_for_cost_evidence": True,
                "funding_x_complete": _truthy(status_x.get("funding_complete")),
                "funding_y_complete": _truthy(status_y.get("funding_complete")),
                "funding_x_rows": int(_finite(status_x.get("funding_rows"), default=0)),
                "funding_y_rows": int(_finite(status_y.get("funding_rows"), default=0)),
                "strict_l2_samples_x": stats_x["samples"],
                "strict_l2_samples_y": stats_y["samples"],
                "strict_l2_local_capture_timestamp_fallbacks_x": stats_x[
                    "local_capture_timestamp_fallbacks"
                ],
                "strict_l2_local_capture_timestamp_fallbacks_y": stats_y[
                    "local_capture_timestamp_fallbacks"
                ],
                "l2_timestamp_policy": L2_TIMESTAMP_POLICY,
                "strict_l2_span_minutes_x": stats_x["span_minutes"],
                "strict_l2_span_minutes_y": stats_y["span_minutes"],
                "strict_l2_max_gap_minutes_x": stats_x["max_gap_minutes"],
                "strict_l2_max_gap_minutes_y": stats_y["max_gap_minutes"],
                "maximum_strict_l2_gap_minutes": maximum_gap_minutes,
                "strict_l2_start_at": min(
                    value for value in (stats_x["start_at"], stats_y["start_at"]) if value
                )
                if stats_x["start_at"] and stats_y["start_at"]
                else "",
                "strict_l2_end_at": max(stats_x["end_at"], stats_y["end_at"]),
                "slippage_x_p95_bps": p95_x,
                "slippage_y_p95_bps": p95_y,
                "pair_one_way_slippage_bps": pair_slippage,
                "spread_x_p95_bps": stats_x["spread_p95_bps"],
                "spread_y_p95_bps": stats_y["spread_p95_bps"],
                "available_notional_x_p05_usd": stats_x["available_notional_p05_usd"],
                "available_notional_y_p05_usd": stats_y["available_notional_p05_usd"],
                "notional_per_leg_usd": 1_000.0,
                "fee_profile_id": _text(profile.get("profile_id")),
                "fee_source_checked_at": _text(profile.get("fee_source_checked_at")),
                "fee_profile_fresh": fee_fresh,
                "taker_fee_bps": fee,
                "execution_risk_bps": execution_risk,
                "cost_normalization": "pair_gross_capital_two_legs_open_and_close",
                "funding_cost_treatment": (
                    "direction_and_hold_dependent_applied_in_point_in_time_replay"
                ),
                "estimated_pair_round_trip_cost_bps": round_trip,
                "cost_acceptance_ready": ready,
                "strict_observed_cost_ready": ready,
                "cost_model_status": (
                    "STRICT_OBSERVED" if ready else "BLOCKED_PROVISIONAL_OR_MISSING"
                ),
                "cost_model_blocker": ";".join(dict.fromkeys(blockers)),
                **input_hashes,
                "source_cost_evidence_path": ";".join(
                    dict.fromkeys(path for path in evidence_paths if path)
                ),
                "evidence_path": ";".join(dict.fromkeys(path for path in evidence_paths if path)),
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    models = pd.DataFrame(model_rows)
    model_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    _atomic_csv(models, model_path)
    model_snapshot_path = bundle_dir / "pair_cost_models.csv"
    _write_or_validate_immutable_bytes(model_path.read_bytes(), model_snapshot_path)
    rows = []
    for row in models.to_dict("records"):
        base = _finite(row.get("estimated_pair_round_trip_cost_bps"))
        for scenario, multiplier, additive in (
            ("base", 1.0, 0.0),
            ("stress", 1.5, 2.0),
            ("tail", 2.0, 5.0),
        ):
            cost = base * multiplier + additive if math.isfinite(base) else math.nan
            ready = bool(row.get("strict_observed_cost_ready")) and math.isfinite(cost)
            rows.append(
                {
                    "pair_group_key": _text(row.get("pair_group_key")),
                    "pair": _text(row.get("pair")),
                    "scenario": scenario,
                    "round_trip_cost_bps": cost,
                    "strict_observed_cost_ready": ready,
                    "blocker": ""
                    if ready
                    else _text(row.get("cost_model_blocker")) or "strict_observed_cost_missing",
                    "evidence_path": _text(row.get("evidence_path")),
                    "promotion_authority": False,
                    "testnet_order_authority": False,
                    "live_trading_authorized": False,
                }
            )
    stress = pd.DataFrame(rows)
    stress_path = active / "hyperliquid_cost_stress.csv"
    _atomic_csv(stress, stress_path)
    stress_snapshot_path = bundle_dir / "cost_stress.csv"
    _write_or_validate_immutable_bytes(stress_path.read_bytes(), stress_snapshot_path)
    bundle_manifest_core = {
        "schema_version": "thewiz.l2_cost_model_receipt.v1",
        "bundle_id": bundle_id,
        "model_as_of_utc": now.isoformat(),
        "input_paths": {
            key: _relative(path, root) for key, path in bundle_inputs.items()
        },
        "input_hashes": input_hashes,
        "pair_cost_models_path": _relative(model_snapshot_path, root),
        "pair_cost_models_sha256": _file_sha256(model_snapshot_path),
        "cost_stress_path": _relative(stress_snapshot_path, root),
        "cost_stress_sha256": _file_sha256(stress_snapshot_path),
        "pair_cost_model_rows": len(models),
        "cost_stress_rows": len(stress),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    bundle_manifest = {
        **bundle_manifest_core,
        "receipt_id": "l2costreceipt_"
        + sha256(
            json.dumps(bundle_manifest_core, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20],
    }
    bundle_manifest_path = bundle_dir / "receipt.json"
    _write_or_validate_immutable_bytes(
        (json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        bundle_manifest_path,
    )
    pointer_material = {
        "schema_version": PAIR_COST_BUNDLE_POINTER_SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "bundle_id": bundle_id,
        "bundle_manifest_path": _relative(bundle_manifest_path, root),
        "bundle_manifest_sha256": _file_sha256(bundle_manifest_path),
        "pair_cost_models_path": _relative(model_snapshot_path, root),
        "pair_cost_models_sha256": _file_sha256(model_snapshot_path),
        "active_pair_cost_models_path": _relative(model_path, root),
        "active_pair_cost_models_sha256": _file_sha256(model_path),
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    pointer = {
        **pointer_material,
        "pointer_id": "l2costpointer_"
        + sha256(
            json.dumps(pointer_material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:20],
    }
    pointer["receipt_sha256"] = sha256(
        json.dumps(pointer, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    pointer_snapshot_path = bundle_dir / "pointer.json"
    _write_or_validate_immutable_bytes(
        (json.dumps(pointer, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        pointer_snapshot_path,
    )
    pointer_path = active / PAIR_COST_BUNDLE_POINTER_NAME
    _atomic_json(pointer, pointer_path)
    stage_two_candidates = candidates.loc[
        candidates.get(
            "stage_two_candidate", pd.Series(True, index=candidates.index)
        ).map(_truthy)
    ]
    stage_two_models = models.loc[
        models.get(
            "stage_two_candidate", pd.Series(True, index=models.index)
        ).map(_truthy)
    ]
    stage_two_eligible = len(stage_two_candidates)
    stage_two_ready = int(
        stage_two_models.get(
            "strict_observed_cost_ready", pd.Series(dtype=bool)
        ).map(_truthy).sum()
    )
    return {
        "models": model_path,
        "stress": stress_path,
        "model_snapshot": model_snapshot_path,
        "stress_snapshot": stress_snapshot_path,
        "bundle_manifest": bundle_manifest_path,
        "bundle_pointer": pointer_path,
        "bundle_pointer_snapshot": pointer_snapshot_path,
        "bundle_id": bundle_id,
        "pairs": len(models),
        "eligible_pairs": len(candidates),
        "strict_ready_pairs": int(
            models.get("strict_observed_cost_ready", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "stage_two_eligible_pairs": stage_two_eligible,
        "stage_two_strict_ready_pairs": stage_two_ready,
        "stage_two_acceptance_status": (
            "PASS"
            if stage_two_eligible > 0
            and len(stage_two_models) == stage_two_eligible
            and stage_two_ready == stage_two_eligible
            else "BLOCKED"
        ),
        "status": "PASS" if len(models) == len(candidates) and len(models) > 0 else "BLOCKED",
        "acceptance_status": (
            "PASS"
            if len(models) == len(candidates)
            and len(models) > 0
            and models["strict_observed_cost_ready"].map(_truthy).all()
            else "BLOCKED"
        ),
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _complete_l2_observations(frame: pd.DataFrame) -> pd.DataFrame:
    """Return complete L2 observations with an auditable cadence timestamp."""

    work = frame.copy()
    if "source_timestamp" not in work:
        work["source_timestamp"] = ""
    if "captured_at" not in work:
        work["captured_at"] = ""
    source_raw = work["source_timestamp"].fillna("").astype(str).str.strip()
    source_timestamp = pd.to_datetime(source_raw, format="mixed", utc=True, errors="coerce")
    captured_at = pd.to_datetime(work["captured_at"], format="mixed", utc=True, errors="coerce")
    local_fallback = source_raw.eq("") & captured_at.notna()
    work["source_timestamp"] = source_timestamp
    work["captured_at"] = captured_at
    work["effective_l2_observation_timestamp"] = source_timestamp.mask(local_fallback, captured_at)
    work["l2_local_capture_timestamp_fallback"] = local_fallback
    work["l2_observation_timestamp_source"] = "vendor_source_timestamp"
    work.loc[local_fallback, "l2_observation_timestamp_source"] = "local_capture_receipt_fallback"
    work.loc[
        work["effective_l2_observation_timestamp"].isna(),
        "l2_observation_timestamp_source",
    ] = "missing_or_invalid"
    complete = (
        work.get("buy_complete", pd.Series(False, index=work.index)).map(_truthy)
        & work.get("sell_complete", pd.Series(False, index=work.index)).map(_truthy)
        & work["effective_l2_observation_timestamp"].notna()
    )
    return work.loc[complete].sort_values("effective_l2_observation_timestamp")


def _strict_l2_window(
    frame: pd.DataFrame,
    *,
    now: datetime,
    root: Path,
) -> pd.DataFrame:
    required = {
        "asset",
        "notional_usd",
        "source_timestamp",
        "one_way_slippage_bps",
        "buy_complete",
        "sell_complete",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    policy = _cost_gate_policy(root)
    start = pd.Timestamp(now) - pd.Timedelta(hours=float(policy.get("strict_l2_window_hours", 2.0)))
    work = _complete_l2_observations(frame)
    work["asset"] = work["asset"].astype(str).str.upper()
    work["notional_usd"] = pd.to_numeric(work["notional_usd"], errors="coerce")
    work["one_way_slippage_bps"] = pd.to_numeric(work["one_way_slippage_bps"], errors="coerce")
    valid = (
        work["effective_l2_observation_timestamp"].between(
            start, pd.Timestamp(now), inclusive="both"
        )
        & work["notional_usd"].sub(1_000.0).abs().le(1e-6)
        & work["one_way_slippage_bps"].notna()
    )
    if "blocker" in work:
        valid &= work["blocker"].fillna("").astype(str).str.strip().eq("")
    return (
        work.loc[valid]
        .sort_values("effective_l2_observation_timestamp")
        .drop_duplicates(
            ["asset", "notional_usd", "effective_l2_observation_timestamp"],
            keep="last",
        )
    )


def _strict_asset_cost_statistics(frame: pd.DataFrame, asset: str) -> dict[str, Any]:
    samples = frame.loc[frame.get("asset", pd.Series(dtype=str)).eq(asset)].copy()
    timestamps = samples.get(
        "effective_l2_observation_timestamp",
        pd.Series(dtype="datetime64[ns, UTC]"),
    )
    span = (
        float((timestamps.max() - timestamps.min()).total_seconds() / 60.0)
        if len(timestamps) >= 2
        else 0.0
    )
    ordered_timestamps = timestamps.sort_values()
    gaps = ordered_timestamps.diff().dropna().dt.total_seconds() / 60.0
    max_gap = float(gaps.max()) if not gaps.empty else math.inf
    slippage = pd.to_numeric(
        samples.get("one_way_slippage_bps", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    spread = pd.to_numeric(
        samples.get("top_of_book_spread_bps", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    buy_available = pd.to_numeric(
        samples.get("buy_available_notional_usd", pd.Series(dtype=float)),
        errors="coerce",
    )
    sell_available = pd.to_numeric(
        samples.get("sell_available_notional_usd", pd.Series(dtype=float)),
        errors="coerce",
    )
    available_sides = pd.concat([buy_available, sell_available], axis=1)
    available = available_sides.loc[available_sides.notna().all(axis=1)].min(axis=1).dropna()
    return {
        "samples": len(samples),
        "local_capture_timestamp_fallbacks": int(
            samples.get(
                "l2_local_capture_timestamp_fallback",
                pd.Series(False, index=samples.index),
            )
            .map(_truthy)
            .sum()
        ),
        "span_minutes": span,
        "max_gap_minutes": max_gap,
        "start_at": timestamps.min().isoformat() if len(timestamps) else "",
        "end_at": timestamps.max().isoformat() if len(timestamps) else "",
        "slippage_p95_bps": float(slippage.quantile(0.95)) if not slippage.empty else math.nan,
        "spread_p95_bps": float(spread.quantile(0.95)) if not spread.empty else math.nan,
        "available_notional_p05_usd": float(available.quantile(0.05))
        if not available.empty
        else math.nan,
        "statistics_complete": bool(
            len(samples) > 0
            and len(slippage) == len(samples)
            and len(spread) == len(samples)
            and len(available) == len(samples)
        ),
    }


def validate_cost_bundle(bundle: dict[str, Any], *, now: datetime | None = None) -> list[str]:
    now = _as_utc(now)
    blockers = []
    timestamp = pd.to_datetime(bundle.get("captured_at"), utc=True, errors="coerce")
    if pd.isna(timestamp) or timestamp < now - pd.Timedelta(hours=2):
        blockers.append("cost_evidence_stale")
    samples = int(bundle.get("samples", 0) or 0)
    if samples < 12:
        blockers.append("cost_evidence_sparse")
    funding_coverage = _finite(bundle.get("funding_coverage"))
    if not math.isfinite(funding_coverage) or funding_coverage < 0.95:
        blockers.append("funding_coverage_insufficient")
    base = _finite(bundle.get("base_slippage_bps"))
    stress = _finite(bundle.get("stress_slippage_bps"))
    if not math.isfinite(base) or base < 0 or base > 500:
        blockers.append("slippage_unit_or_sign_invalid")
    if not math.isfinite(stress) or stress < base:
        blockers.append("stress_surface_cherry_picked_or_sign_invalid")
    return blockers


def validate_pair_cost_bundle_artifacts(
    *,
    root: Path,
    bundle_manifest_path: Path,
    pair_cost_models_path: Path | None = None,
) -> list[str]:
    """Validate the complete immutable input-to-model lineage for one cost bundle."""

    blockers: list[str] = []
    root = root.resolve()
    manifest_path = bundle_manifest_path.resolve()
    try:
        manifest_path.relative_to(root)
    except ValueError:
        return ["pair_cost_bundle_manifest_path_unsafe"]
    manifest = _read_json(manifest_path)
    bundle_id = _text(manifest.get("bundle_id"))
    bundle_dir = root / "data" / "research" / "l2_cost_model_receipts" / bundle_id
    if (
        manifest.get("schema_version") != "thewiz.l2_cost_model_receipt.v1"
        or not bundle_id.startswith("l2costbundle_")
        or manifest_path != bundle_dir / "receipt.json"
    ):
        blockers.append("pair_cost_bundle_manifest_identity_invalid")

    input_paths = manifest.get("input_paths")
    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_paths, dict) or not isinstance(input_hashes, dict):
        return list(
            dict.fromkeys(
                [*blockers, "pair_cost_bundle_input_manifest_invalid"]
            )
        )
    required_hashes = {
        hash_key for _, hash_key in PAIR_COST_BUNDLE_INPUTS.values()
    } | {"l2_source_file_sha256"}
    if any(not _valid_sha256(input_hashes.get(key)) for key in required_hashes):
        blockers.append("pair_cost_bundle_input_hash_invalid")

    resolved_inputs: dict[str, Path] = {}
    for input_name, (filename, hash_key) in PAIR_COST_BUNDLE_INPUTS.items():
        expected = bundle_dir / filename
        candidate = _safe_root_artifact(root, input_paths.get(input_name))
        if candidate != expected:
            blockers.append(f"pair_cost_bundle_{input_name}_path_invalid")
            continue
        resolved_inputs[input_name] = candidate
        if not candidate.is_file():
            blockers.append(f"pair_cost_bundle_{input_name}_missing")
        elif _file_sha256(candidate) != _text(input_hashes.get(hash_key)):
            blockers.append(f"pair_cost_bundle_{input_name}_hash_mismatch")

    expected_model_path = bundle_dir / "pair_cost_models.csv"
    manifest_model_path = _safe_root_artifact(
        root, manifest.get("pair_cost_models_path")
    )
    supplied_model_path = (
        pair_cost_models_path.resolve()
        if pair_cost_models_path is not None
        else manifest_model_path
    )
    if manifest_model_path != expected_model_path or supplied_model_path != expected_model_path:
        blockers.append("pair_cost_bundle_model_path_invalid")
    if supplied_model_path is None or not supplied_model_path.is_file():
        blockers.append("pair_cost_bundle_model_missing")
        models = pd.DataFrame()
    else:
        if _file_sha256(supplied_model_path) != _text(
            manifest.get("pair_cost_models_sha256")
        ):
            blockers.append("pair_cost_bundle_model_hash_mismatch")
        models = _read_csv(supplied_model_path)

    manifest_core = {key: value for key, value in manifest.items() if key != "receipt_id"}
    expected_receipt_id = "l2costreceipt_" + sha256(
        json.dumps(manifest_core, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    bundle_material = {
        "model_as_of_utc": _text(manifest.get("model_as_of_utc")),
        **input_hashes,
    }
    expected_bundle_id = "l2costbundle_" + sha256(
        json.dumps(bundle_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    if bundle_id != expected_bundle_id or _text(manifest.get("receipt_id")) != expected_receipt_id:
        blockers.append("pair_cost_bundle_receipt_identity_invalid")
    if any(
        manifest.get(field) is not False
        for field in (
            "promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        blockers.append("pair_cost_bundle_authority_invalid")

    if models.empty:
        blockers.append("pair_cost_bundle_models_empty")
        return list(dict.fromkeys(blockers))
    try:
        expected_rows = int(manifest.get("pair_cost_model_rows", -1))
    except (TypeError, ValueError):
        expected_rows = -1
    if expected_rows != len(models) or expected_rows <= 0:
        blockers.append("pair_cost_bundle_model_row_count_mismatch")

    required_model_columns = {
        "schema_version",
        "cost_model_id",
        "model_as_of_utc",
        "pair_group_key",
        "pair",
        "execution_venue",
        "instrument",
        "asset_x",
        "asset_y",
        "confirmation_role",
        "source_family",
        "semantic_hypothesis_id",
        "registered_contract_id",
        "registered_contract_candidate",
        "stage_two_candidate",
        "selected_for_cost_evidence",
        "funding_x_complete",
        "funding_y_complete",
        "funding_x_rows",
        "funding_y_rows",
        "strict_l2_samples_x",
        "strict_l2_samples_y",
        "strict_l2_local_capture_timestamp_fallbacks_x",
        "strict_l2_local_capture_timestamp_fallbacks_y",
        "strict_l2_span_minutes_x",
        "strict_l2_span_minutes_y",
        "strict_l2_max_gap_minutes_x",
        "strict_l2_max_gap_minutes_y",
        "maximum_strict_l2_gap_minutes",
        "strict_l2_start_at",
        "strict_l2_end_at",
        "slippage_x_p95_bps",
        "slippage_y_p95_bps",
        "pair_one_way_slippage_bps",
        "spread_x_p95_bps",
        "spread_y_p95_bps",
        "available_notional_x_p05_usd",
        "available_notional_y_p05_usd",
        "fee_profile_id",
        "fee_source_checked_at",
        "fee_profile_fresh",
        "taker_fee_bps",
        "execution_risk_bps",
        "estimated_pair_round_trip_cost_bps",
        "cost_acceptance_ready",
        "strict_observed_cost_ready",
        "cost_model_status",
        "cost_model_blocker",
        *required_hashes,
        "promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    }
    if not required_model_columns.issubset(models.columns):
        blockers.append("pair_cost_bundle_model_schema_invalid")
        return list(dict.fromkeys(blockers))
    if models["pair_group_key"].map(_text).eq("").any() or models[
        "pair_group_key"
    ].map(_text).duplicated().any():
        blockers.append("pair_cost_bundle_model_pair_identity_invalid")

    candidates = _read_csv(resolved_inputs.get("candidate_set", Path()))
    statuses = _read_csv(resolved_inputs.get("cost_status", Path()))
    l2 = _read_csv(resolved_inputs.get("strict_l2_window", Path()))
    profile = _read_json(resolved_inputs.get("fee_profile", Path()))
    required_candidate_columns = {
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "confirmation_role",
        "source_family",
        "semantic_hypothesis_id",
        "registered_contract_id",
        "registered_contract_candidate",
        "stage_two_candidate",
        "collection_eligible",
        "testnet_order_authority",
        "live_trading_authorized",
    }
    if candidates.empty or not required_candidate_columns.issubset(candidates.columns):
        blockers.append("pair_cost_bundle_candidate_schema_invalid")
        eligible = pd.DataFrame()
    else:
        eligible = candidates.loc[candidates["collection_eligible"].map(_truthy)].copy()
        if (
            eligible.empty
            or eligible["pair_group_key"].map(_text).eq("").any()
            or eligible["pair_group_key"].map(_text).duplicated().any()
            or eligible["testnet_order_authority"].map(_truthy).any()
            or eligible["live_trading_authorized"].map(_truthy).any()
        ):
            blockers.append("pair_cost_bundle_candidate_identity_invalid")
    model_keys = set(models["pair_group_key"].map(_text))
    candidate_keys = (
        set(eligible["pair_group_key"].map(_text)) if not eligible.empty else set()
    )
    if model_keys != candidate_keys:
        blockers.append("pair_cost_bundle_candidate_model_coverage_mismatch")

    relevant_assets = {
        _text(value).upper()
        for column in ("asset_x", "asset_y")
        for value in models[column]
        if _text(value)
    }
    required_status_columns = {
        "asset",
        "funding_rows",
        "funding_complete",
        "strict_l2_samples",
        "strict_l2_span_minutes",
        "strict_l2_max_gap_minutes",
        "strict_l2_latest_age_minutes",
        "maximum_strict_l2_gap_minutes",
        "strict_l2_cadence_ready",
        "minimum_strict_l2_samples",
        "minimum_strict_l2_span_minutes",
        "l2_timestamp_policy",
        "collection_status",
        "live_trading_authorized",
    }
    status_by_asset: dict[str, dict[str, Any]] = {}
    if statuses.empty or not required_status_columns.issubset(statuses.columns):
        blockers.append("pair_cost_bundle_cost_status_schema_invalid")
    else:
        statuses = statuses.copy()
        statuses["asset"] = statuses["asset"].map(_text).str.upper()
        relevant_status = statuses.loc[statuses["asset"].isin(relevant_assets)]
        counts = relevant_status["asset"].value_counts()
        if set(counts.index) != relevant_assets or not counts.eq(1).all():
            blockers.append("pair_cost_bundle_cost_status_coverage_invalid")
        else:
            status_by_asset = {
                row["asset"]: row for row in relevant_status.to_dict("records")
            }
            if any(
                not _truthy(row.get("funding_complete"))
                or not _truthy(row.get("strict_l2_cadence_ready"))
                or _text(row.get("collection_status")) != "READY"
                or _truthy(row.get("live_trading_authorized"))
                for row in status_by_asset.values()
            ):
                blockers.append("pair_cost_bundle_cost_status_not_ready")

    required_l2_columns = {
        "asset",
        "notional_usd",
        "buy_complete",
        "sell_complete",
        "one_way_slippage_bps",
        "top_of_book_spread_bps",
        "buy_available_notional_usd",
        "sell_available_notional_usd",
        "effective_l2_observation_timestamp",
        "l2_local_capture_timestamp_fallback",
    }
    l2_stats: dict[str, dict[str, Any]] = {}
    if l2.empty or not required_l2_columns.issubset(l2.columns):
        blockers.append("pair_cost_bundle_l2_schema_invalid")
    else:
        l2 = l2.copy()
        l2["asset"] = l2["asset"].map(_text).str.upper()
        l2["effective_l2_observation_timestamp"] = pd.to_datetime(
            l2["effective_l2_observation_timestamp"],
            format="mixed",
            utc=True,
            errors="coerce",
        )
        malformed = (
            l2["effective_l2_observation_timestamp"].isna()
            | ~l2["buy_complete"].map(_truthy)
            | ~l2["sell_complete"].map(_truthy)
            | pd.to_numeric(l2["notional_usd"], errors="coerce")
            .sub(1_000.0)
            .abs()
            .gt(1e-6)
        )
        if "blocker" in l2.columns:
            malformed |= l2["blocker"].map(_text).ne("")
        if malformed.any() or not relevant_assets.issubset(set(l2["asset"])):
            blockers.append("pair_cost_bundle_l2_content_invalid")
        if l2.duplicated(
            ["asset", "notional_usd", "effective_l2_observation_timestamp"]
        ).any():
            blockers.append("pair_cost_bundle_l2_duplicate_observation")
        l2_stats = {
            asset: _strict_asset_cost_statistics(l2, asset)
            for asset in relevant_assets
        }
        expected_maximum_gap = float(
            _cost_gate_policy(root).get("maximum_strict_l2_gap_minutes", 15.0)
        )
        model_as_of = pd.to_datetime(
            manifest.get("model_as_of_utc"), utc=True, errors="coerce"
        )
        if any(not stats["statistics_complete"] for stats in l2_stats.values()):
            blockers.append("pair_cost_bundle_l2_statistics_incomplete")
        for asset in relevant_assets:
            stats = l2_stats.get(asset, {})
            status = status_by_asset.get(asset, {})
            end_at = pd.to_datetime(stats.get("end_at"), utc=True, errors="coerce")
            expected_latest_age = (
                float((model_as_of - end_at).total_seconds() / 60.0)
                if pd.notna(model_as_of) and pd.notna(end_at)
                else math.inf
            )
            if (
                not stats
                or not status
                or not _number_matches(
                    status.get("strict_l2_samples"), stats.get("samples")
                )
                or not _number_matches(
                    status.get("strict_l2_span_minutes"),
                    stats.get("span_minutes"),
                )
                or not _number_matches(
                    status.get("strict_l2_max_gap_minutes"),
                    stats.get("max_gap_minutes"),
                )
                or not _number_matches(
                    status.get("strict_l2_latest_age_minutes"),
                    expected_latest_age,
                )
                or _finite(stats.get("samples"))
                < _finite(status.get("minimum_strict_l2_samples"))
                or _finite(stats.get("span_minutes"))
                < _finite(status.get("minimum_strict_l2_span_minutes"))
                or _finite(stats.get("max_gap_minutes"))
                > _finite(status.get("maximum_strict_l2_gap_minutes"))
                or expected_latest_age
                > _finite(status.get("maximum_strict_l2_gap_minutes"))
                or not _number_matches(
                    status.get("maximum_strict_l2_gap_minutes"),
                    expected_maximum_gap,
                )
                or _text(status.get("l2_timestamp_policy")) != L2_TIMESTAMP_POLICY
            ):
                blockers.append("pair_cost_bundle_cost_status_l2_mismatch")

    candidate_by_key = (
        {
            _text(row.get("pair_group_key")): row
            for row in eligible.to_dict("records")
        }
        if not eligible.empty
        else {}
    )
    fee = _finite(profile.get("taker_fee_bps"))
    execution_risk = _finite(profile.get("execution_risk_bps"))
    if (
        _text(profile.get("profile_id")) == ""
        or _text(profile.get("venue")) != "hyperliquid"
        or _text(profile.get("instrument")) != "perpetual"
        or not math.isfinite(fee)
        or fee < 0
        or not math.isfinite(execution_risk)
        or execution_risk < 0
    ):
        blockers.append("pair_cost_bundle_fee_profile_invalid")

    identity_fields = (
        "pair",
        "asset_x",
        "asset_y",
        "confirmation_role",
        "source_family",
        "semantic_hypothesis_id",
        "registered_contract_id",
    )
    for model in models.to_dict("records"):
        pair_key = _text(model.get("pair_group_key"))
        candidate = candidate_by_key.get(pair_key, {})
        asset_x = _text(model.get("asset_x")).upper()
        asset_y = _text(model.get("asset_y")).upper()
        stats_x = l2_stats.get(asset_x, {})
        stats_y = l2_stats.get(asset_y, {})
        status_x = status_by_asset.get(asset_x, {})
        status_y = status_by_asset.get(asset_y, {})
        identity_payload = {
            "pair_group_key": pair_key,
            "model_as_of_utc": _text(manifest.get("model_as_of_utc")),
            **input_hashes,
        }
        expected_model_id = "hlpaircost_" + sha256(
            json.dumps(identity_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        if (
            _text(model.get("schema_version"))
            != "thewiz.registered_pair_cost_model.v1"
            or _text(model.get("cost_model_id")) != expected_model_id
            or _text(model.get("model_as_of_utc"))
            != _text(manifest.get("model_as_of_utc"))
            or _text(model.get("execution_venue")) != "hyperliquid"
            or _text(model.get("instrument")) != "perpetual"
            or any(_text(model.get(key)) != _text(input_hashes.get(key)) for key in required_hashes)
        ):
            blockers.append("pair_cost_bundle_model_lineage_invalid")
        if (
            not candidate
            or any(
                _text(model.get(field)).upper()
                != _text(candidate.get(field)).upper()
                for field in identity_fields
            )
            or _truthy(model.get("registered_contract_candidate"))
            != _truthy(candidate.get("registered_contract_candidate"))
            or _truthy(model.get("stage_two_candidate"))
            != _truthy(candidate.get("stage_two_candidate"))
        ):
            blockers.append("pair_cost_bundle_candidate_model_identity_mismatch")
        if (
            not _truthy(model.get("selected_for_cost_evidence"))
            or not _truthy(model.get("funding_x_complete"))
            or not _truthy(model.get("funding_y_complete"))
            or not _truthy(model.get("fee_profile_fresh"))
            or not _truthy(model.get("cost_acceptance_ready"))
            or not _truthy(model.get("strict_observed_cost_ready"))
            or _text(model.get("cost_model_status")) != "STRICT_OBSERVED"
            or _text(model.get("cost_model_blocker"))
            or any(
                _truthy(model.get(field))
                for field in (
                    "promotion_authority",
                    "testnet_order_authority",
                    "live_trading_authorized",
                )
            )
        ):
            blockers.append("pair_cost_bundle_model_readiness_invalid")
        if (
            not status_x
            or not status_y
            or not _number_matches(model.get("funding_x_rows"), status_x.get("funding_rows"))
            or not _number_matches(model.get("funding_y_rows"), status_y.get("funding_rows"))
        ):
            blockers.append("pair_cost_bundle_model_funding_mismatch")
        if not stats_x or not stats_y:
            blockers.append("pair_cost_bundle_model_l2_coverage_missing")
            continue
        pair_start = (
            min(stats_x["start_at"], stats_y["start_at"])
            if stats_x["start_at"] and stats_y["start_at"]
            else ""
        )
        pair_end = max(stats_x["end_at"], stats_y["end_at"])
        numeric_statistics = {
            "strict_l2_samples_x": stats_x["samples"],
            "strict_l2_samples_y": stats_y["samples"],
            "strict_l2_local_capture_timestamp_fallbacks_x": stats_x[
                "local_capture_timestamp_fallbacks"
            ],
            "strict_l2_local_capture_timestamp_fallbacks_y": stats_y[
                "local_capture_timestamp_fallbacks"
            ],
            "strict_l2_span_minutes_x": stats_x["span_minutes"],
            "strict_l2_span_minutes_y": stats_y["span_minutes"],
            "strict_l2_max_gap_minutes_x": stats_x["max_gap_minutes"],
            "strict_l2_max_gap_minutes_y": stats_y["max_gap_minutes"],
            "maximum_strict_l2_gap_minutes": expected_maximum_gap,
            "slippage_x_p95_bps": stats_x["slippage_p95_bps"],
            "slippage_y_p95_bps": stats_y["slippage_p95_bps"],
            "spread_x_p95_bps": stats_x["spread_p95_bps"],
            "spread_y_p95_bps": stats_y["spread_p95_bps"],
            "available_notional_x_p05_usd": stats_x[
                "available_notional_p05_usd"
            ],
            "available_notional_y_p05_usd": stats_y[
                "available_notional_p05_usd"
            ],
        }
        pair_slippage = (
            stats_x["slippage_p95_bps"] + stats_y["slippage_p95_bps"]
        ) / 2.0
        round_trip = 2.0 * (fee + pair_slippage + execution_risk)
        numeric_statistics.update(
            {
                "pair_one_way_slippage_bps": pair_slippage,
                "taker_fee_bps": fee,
                "execution_risk_bps": execution_risk,
                "estimated_pair_round_trip_cost_bps": round_trip,
            }
        )
        if (
            any(
                not _number_matches(model.get(field), expected)
                for field, expected in numeric_statistics.items()
            )
            or _text(model.get("strict_l2_start_at")) != pair_start
            or _text(model.get("strict_l2_end_at")) != pair_end
        ):
            blockers.append("pair_cost_bundle_model_l2_statistics_mismatch")
        if (
            _text(model.get("fee_profile_id")) != _text(profile.get("profile_id"))
            or _text(model.get("fee_source_checked_at"))
            != _text(profile.get("fee_source_checked_at"))
        ):
            blockers.append("pair_cost_bundle_model_fee_profile_mismatch")

    return list(dict.fromkeys(blockers))


def _safe_root_artifact(root: Path, value: Any) -> Path | None:
    text = _text(value)
    if not text:
        return None
    relative = Path(text)
    if relative.is_absolute():
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _valid_sha256(value: Any) -> bool:
    text = _text(value).lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _number_matches(actual: Any, expected: Any) -> bool:
    actual_number = _finite(actual)
    expected_number = _finite(expected)
    return bool(
        math.isfinite(actual_number)
        and math.isfinite(expected_number)
        and math.isclose(
            actual_number,
            expected_number,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    )


def run_cost_evidence_attacks(*, root: Path = ROOT, now: datetime | None = None) -> pd.DataFrame:
    now = _as_utc(now)
    base = {
        "captured_at": now.isoformat(),
        "samples": 24,
        "funding_coverage": 0.99,
        "base_slippage_bps": 3.0,
        "stress_slippage_bps": 6.0,
    }
    cases = {
        "valid": base,
        "stale": {**base, "captured_at": (now - pd.Timedelta(days=1)).isoformat()},
        "sparse": {**base, "samples": 2},
        "cherry_picked": {**base, "stress_slippage_bps": 1.0},
        "unit_flipped": {**base, "base_slippage_bps": 3000.0, "stress_slippage_bps": 6000.0},
        "sign_flipped": {**base, "base_slippage_bps": -3.0, "stress_slippage_bps": -1.0},
        "funding_gap": {**base, "funding_coverage": 0.4},
    }
    rows = []
    for case, bundle in cases.items():
        blockers = validate_cost_bundle(bundle, now=now)
        expected_pass = case == "valid"
        actual_pass = not blockers
        rows.append(
            {
                "case": case,
                "expected_status": "PASS" if expected_pass else "BLOCKED",
                "actual_status": "PASS" if actual_pass else "BLOCKED",
                "blocker": ";".join(blockers),
                "manipulated_evidence_improves_readiness": bool(not expected_pass and actual_pass),
                "live_trading_authorized": False,
                "status": "PASS" if expected_pass == actual_pass else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    path = root / "reports" / "red_team" / "cost_evidence_attack_results.csv"
    _atomic_csv(frame, path)
    if not frame["status"].eq("PASS").all():
        raise ValueError("one or more hostile cost bundles evaded validation")
    return frame


def build_corrective_data_evidence(
    *, root: Path = ROOT, now: datetime | None = None
) -> CommandResult:
    now = _as_utc(now)
    market = build_hyperliquid_market_manifest(root=root, now=now)
    source_attacks = build_external_source_contract_attacks(root=root, now=now)
    history = build_history_remediation(root=root)
    candidates = build_l2_capture_candidate_set(root=root)
    collection = build_cost_collection_status(root=root, now=now)
    costs = build_pair_cost_stress_surfaces(root=root, now=now)
    cost_attacks = run_cost_evidence_attacks(root=root, now=now)
    readiness_refresh = _l2_readiness_refresh_evidence(root=root)
    paths = {
        "market_manifest": Path(market["manifest"]),
        "mapping_audit": Path(market["audit"]),
        "external_source_contract_results": root
        / "reports"
        / "red_team"
        / "external_source_contract_results.csv",
        "history_remediation_queue": Path(history["queue"]),
        "history_coverage": Path(history["coverage"]),
        "l2_capture_candidates": Path(candidates["path"]),
        "cost_collection_status": Path(collection["status_path"]),
        "pair_cost_models": Path(costs["models"]),
        "pair_cost_bundle_pointer": Path(costs["bundle_pointer"]),
        "cost_stress": Path(costs["stress"]),
        "cost_evidence_attack_results": root
        / "reports"
        / "red_team"
        / "cost_evidence_attack_results.csv",
    }
    if readiness_refresh["status_path"].is_file():
        paths["l2_readiness_refresh_status"] = readiness_refresh["status_path"]
    if readiness_refresh["immutable_receipt_path"].is_file():
        paths["l2_readiness_refresh_immutable_receipt"] = readiness_refresh[
            "immutable_receipt_path"
        ]
    return CommandResult(
        paths=paths,
        summary={
            "status": (
                "PASS"
                if market["status"] == "PASS"
                and history["status"] == "PASS"
                and costs["status"] == "PASS"
                and costs["stage_two_acceptance_status"] == "PASS"
                else "BLOCKED"
            ),
            "infrastructure_status": "PASS"
            if market["status"] == "PASS"
            and history["status"] == "PASS"
            and costs["status"] == "PASS"
            else "BLOCKED",
            "mapped_assets": market["assets"],
            "mapped_pairs": market["pairs"],
            "mapping_passes": market["mapping_passes"],
            "history_ready_pairs": history["ready"],
            "history_queued_pairs": history["queued"],
            "history_deferred_pairs": history["deferred"],
            "history_structurally_blocked_pairs": history["structurally_blocked"],
            "history_insufficient_asset_age_pairs": history["insufficient_asset_age"],
            "l2_capture_eligible_pairs": candidates["eligible_pairs"],
            "cost_collection_ready_assets": collection["ready_assets"],
            "cost_collection_collecting_assets": collection["collecting_assets"],
            "strict_cost_eligible_pairs": costs["eligible_pairs"],
            "strict_cost_ready_pairs": costs["strict_ready_pairs"],
            "stage_two_strict_cost_eligible_pairs": costs[
                "stage_two_eligible_pairs"
            ],
            "stage_two_strict_cost_ready_pairs": costs[
                "stage_two_strict_ready_pairs"
            ],
            "stage_two_strict_cost_acceptance_status": costs[
                "stage_two_acceptance_status"
            ],
            "l2_readiness_refresh_validation_status": readiness_refresh[
                "validation_status"
            ],
            "l2_readiness_refresh_status": readiness_refresh["receipt_status"],
            "l2_readiness_refresh_receipt_id": readiness_refresh["receipt_id"],
            "l2_readiness_refresh_source_receipt_id": readiness_refresh[
                "source_l2_receipt_id"
            ],
            "l2_readiness_refresh_executed": readiness_refresh[
                "refresh_executed"
            ],
            "l2_readiness_refresh_gate_executed": readiness_refresh[
                "gate_executed"
            ],
            "l2_readiness_refresh_handoff_executed": readiness_refresh[
                "handoff_executed"
            ],
            "l2_readiness_refresh_ready_pairs": readiness_refresh["ready_pairs"],
            "l2_readiness_refresh_eligible_pairs": readiness_refresh[
                "eligible_pairs"
            ],
            "l2_readiness_refresh_blockers": readiness_refresh["blockers"],
            "source_attack_cases": len(source_attacks),
            "cost_attack_cases": len(cost_attacks),
            "live_trading_authorized": False,
        },
    )


def _l2_readiness_refresh_evidence(*, root: Path) -> dict[str, Any]:
    status_path = root / "reports" / "active" / "corrective_l2_readiness_refresh_status.json"
    payload = _read_json(status_path)
    if not payload:
        return {
            "validation_status": "NOT_AVAILABLE",
            "receipt_status": "NOT_AVAILABLE",
            "receipt_id": "",
            "source_l2_receipt_id": "",
            "refresh_executed": False,
            "gate_executed": False,
            "handoff_executed": False,
            "ready_pairs": 0,
            "eligible_pairs": 0,
            "blockers": ["l2_readiness_refresh_receipt_missing"],
            "status_path": status_path,
            "immutable_receipt_path": Path(),
        }
    try:
        from quant_platform.orchestration.corrective_l2_scheduler import (
            validate_post_window_readiness_receipt,
        )

        validation = validate_post_window_readiness_receipt(
            root=root, receipt=payload
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must fail closed
        validation = {
            "status": "BLOCKED",
            "blockers": [
                f"l2_readiness_refresh_validation_error:{safe_exception_code(exc)}"
            ],
            "immutable_receipt_path": "",
        }
    immutable_relative = str(validation.get("immutable_receipt_path", "")).strip()
    immutable_path = (
        (root / immutable_relative).resolve() if immutable_relative else Path()
    )
    return {
        "validation_status": str(validation.get("status", "BLOCKED")),
        "receipt_status": str(payload.get("status", "BLOCKED")),
        "receipt_id": str(payload.get("receipt_id", "")),
        "source_l2_receipt_id": str(payload.get("source_l2_receipt_id", "")),
        "refresh_executed": _truthy(payload.get("refresh_executed")),
        "gate_executed": _truthy(
            payload.get("registered_gate_refresh_executed")
        ),
        "handoff_executed": _truthy(
            payload.get("stage4_handoff_refresh_executed")
        ),
        "ready_pairs": int(_finite(payload.get("ready_pairs"), default=0)),
        "eligible_pairs": int(_finite(payload.get("eligible_pairs"), default=0)),
        "blockers": list(validation.get("blockers", [])),
        "status_path": status_path,
        "immutable_receipt_path": immutable_path,
    }


def _history_reason(row: pd.Series) -> str:
    blockers = []
    if not _truthy(row.get("timestamp_parse_valid")):
        blockers.append("timestamp_parse_invalid")
    if not _truthy(row.get("timestamp_bound_valid")):
        blockers.append("future_timestamp_or_cutoff_invalid")
    if int(_finite(row.get("post_cutoff_rows"), default=1)) > 0:
        blockers.append("post_cutoff_rows_present")
    if _finite(row.get("history_rows"), default=0) < _finite(
        row.get("minimum_history_rows"), default=math.inf
    ):
        blockers.append("insufficient_aligned_rows")
    source = _text(row.get("history_blocker"))
    if source:
        blockers.append(source)
    return ";".join(dict.fromkeys(blockers))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cost_gate_policy(root: Path) -> dict[str, Any]:
    path = root / "config" / "acceptance_policy_manifest.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    gates = payload.get("cost_gates", {}) if isinstance(payload, dict) else {}
    return gates if isinstance(gates, dict) else {}


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    promote_staged_file(temporary, path)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _write_or_validate_immutable_bytes(payload: bytes, path: Path) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable L2 cost evidence changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    promote_staged_file(temporary, path)


def _finite(value: Any, *, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "pass",
        "ready",
        "complete",
    }


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    result = build_corrective_data_evidence()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
