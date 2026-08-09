"""Corrective Hyperliquid mapping, source contracts, history, and cost evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_data_evidence.v1"


SOURCE_CONTRACTS: dict[str, dict[str, Any]] = {
    "hyperliquid_inventory": {
        "required": ["asset", "asset_index", "universe_name", "sz_decimals", "max_leverage", "is_delisted", "tradable_perp", "checked_at_utc"],
        "key": ["asset"],
        "timestamp": "checked_at_utc",
        "maximum_age_hours": 24.0,
        "minimum_rows": 1,
    },
    "wizard_mapping": {
        "required": ["pair_group_key", "pair", "asset_x", "asset_y", "current_pair_ready", "mapping_status", "current_inventory_checked_at"],
        "key": ["pair_group_key"],
        "timestamp": "current_inventory_checked_at",
        "maximum_age_hours": 24.0,
        "minimum_rows": 1,
    },
    "hyperliquid_l2": {
        "required": ["sample_id", "asset", "notional_usd", "source_timestamp", "one_way_slippage_bps", "buy_complete", "sell_complete"],
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
        populated = raw_timestamps.loc[raw_timestamps.notna() & raw_timestamps.astype(str).str.strip().ne("")]
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
    inventory_blockers = validate_source_frame(inventory, SOURCE_CONTRACTS["hyperliquid_inventory"], now=now)
    inventory_by_asset = {
        asset: group for asset, group in inventory.groupby(inventory.get("asset", pd.Series(dtype=str)).astype(str).str.upper())
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
        tradable = matches.loc[matches.get("tradable_perp", pd.Series(False, index=matches.index)).map(_truthy)] if not matches.empty else matches
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
        fixture_path.write_text(frame.to_json(orient="records", date_format="iso", indent=2) + "\n", encoding="utf-8")
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


def build_history_remediation(
    *, root: Path = ROOT
) -> dict[str, Any]:
    active = root / "reports" / "active"
    history_path = active / "current_wizard_hyperliquid_pair_history_results.csv"
    history = _read_csv(history_path)
    failure = _read_csv(active / "current_wizard_hyperliquid_failure_attribution.csv")
    rank = (
        failure.groupby("pair_group_key", dropna=False)["overall_research_rank"].min().rename("best_research_rank")
        if not failure.empty
        else pd.Series(dtype=float, name="best_research_rank")
    )
    coverage = history.copy()
    coverage["aligned_history_ready"] = (
        coverage.get("history_status", pd.Series(dtype=str)).eq("READY_FOR_CANONICAL_1X_REPLAY")
        & coverage.get("timestamp_parse_valid", pd.Series(False, index=coverage.index)).map(_truthy)
        & coverage.get("timestamp_bound_valid", pd.Series(False, index=coverage.index)).map(_truthy)
        & pd.to_numeric(coverage.get("post_cutoff_rows", 0), errors="coerce").fillna(1).eq(0)
        & pd.to_numeric(coverage.get("history_rows", 0), errors="coerce").fillna(0).ge(pd.to_numeric(coverage.get("minimum_history_rows", 0), errors="coerce").fillna(math.inf))
    )
    coverage = coverage.join(rank, on="pair_group_key")
    coverage["history_priority"] = coverage["best_research_rank"].fillna(10**9)
    coverage["history_remediation_reason"] = coverage.apply(_history_reason, axis=1)
    selected = coverage.get(
        "selected_for_materialization", pd.Series(False, index=coverage.index)
    ).map(_truthy)
    history_status = coverage.get(
        "history_status", pd.Series("", index=coverage.index)
    ).astype(str)
    history_blocker = coverage.get(
        "history_blocker", pd.Series("", index=coverage.index)
    ).fillna("").astype(str)
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
    queue = coverage.loc[
        coverage["history_coverage_class"].eq("ACTIVE_REMEDIATION")
    ].sort_values(["history_priority", "pair_group_key"]).copy()
    queue["next_action"] = "refetch_both_legs_to_declared_cutoff_then_validate_alignment"
    queue_path = active / "hyperliquid_history_remediation_queue.csv"
    _atomic_csv(queue, queue_path)
    return {
        "coverage": coverage_path,
        "queue": queue_path,
        "pairs": len(coverage),
        "ready": int(coverage["aligned_history_ready"].sum()),
        "queued": len(queue),
        "deferred": int(
            coverage["history_coverage_class"].eq("DEFERRED_NOT_SELECTED").sum()
        ),
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
    """Normalize registered hypotheses into an explicit Hyperliquid allow-list."""

    active = root / "reports" / "active"
    hypothesis_path = active / "current_hypothesis_batch.csv"
    attribution_path = active / "current_wizard_hyperliquid_failure_attribution.csv"
    market_path = root / "data" / "processed" / "hyperliquid_market_context.csv"
    hypotheses = _read_csv(hypothesis_path)
    attribution = _read_csv(attribution_path)
    markets = _read_csv(market_path)
    columns = [
        "experiment_id",
        "pair_group_key",
        "pair",
        "asset_x",
        "asset_y",
        "overall_research_rank",
        "confirmation_role",
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
        joined = hypotheses[["experiment_id", "confirmation_role"]].merge(
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
                    "collection_eligible": not blockers,
                    "blocker": ";".join(blockers),
                    "evidence_path": (
                        f"{_relative(hypothesis_path, root)};"
                        f"{_relative(attribution_path, root)};{_relative(market_path, root)}"
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
    path = active / "corrective_l2_capture_candidates.csv"
    _atomic_csv(frame, path)
    return {
        "path": path,
        "registered_hypotheses": len(hypotheses),
        "candidate_pairs": len(frame),
        "eligible_pairs": int(
            frame.get("collection_eligible", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "status": "PASS"
        if frame.get("collection_eligible", pd.Series(dtype=bool)).map(_truthy).any()
        else "BLOCKED",
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def build_cost_collection_status(
    *, root: Path = ROOT, now: datetime | None = None
) -> dict[str, Any]:
    now = _as_utc(now)
    active = root / "reports" / "active"
    policy = _cost_gate_policy(root)
    strict_window_hours = float(policy.get("strict_l2_window_hours", 2.0))
    minimum_samples = int(policy.get("minimum_strict_l2_samples", 12))
    minimum_span_minutes = float(policy.get("minimum_strict_l2_span_minutes", 100.0))
    funding = _read_csv(active / "current_wizard_hyperliquid_funding_asset_results.csv")
    l2 = _read_csv(root / "data" / "processed" / "hyperliquid_l2_slippage_samples.csv")
    assets = sorted(set(funding.get("asset", pd.Series(dtype=str)).astype(str)) | set(l2.get("asset", pd.Series(dtype=str)).astype(str)))
    rows = []
    for asset in assets:
        f = funding.loc[funding.get("asset", pd.Series(dtype=str)).astype(str).eq(asset)]
        s = l2.loc[l2.get("asset", pd.Series(dtype=str)).astype(str).eq(asset)]
        if not s.empty:
            complete = (
                s.get("buy_complete", pd.Series(False, index=s.index)).map(_truthy)
                & s.get("sell_complete", pd.Series(False, index=s.index)).map(_truthy)
            )
            timestamps = pd.to_datetime(
                s.loc[complete, "source_timestamp"], format="mixed", utc=True, errors="coerce"
            ).dropna().drop_duplicates().sort_values()
        else:
            timestamps = pd.Series(dtype="datetime64[ns, UTC]")
        strict_timestamps = timestamps.loc[
            timestamps >= pd.Timestamp(now) - pd.Timedelta(hours=strict_window_hours)
        ]
        provisional_timestamps = timestamps.loc[
            timestamps >= pd.Timestamp(now) - pd.Timedelta(hours=24)
        ]
        strict = int(len(strict_timestamps))
        provisional = int(len(provisional_timestamps))
        span_minutes = (
            float((strict_timestamps.max() - strict_timestamps.min()).total_seconds() / 60.0)
            if len(strict_timestamps) >= 2
            else 0.0
        )
        cadence_ready = strict >= minimum_samples and span_minutes >= minimum_span_minutes
        funding_complete = bool(not f.empty and f.get("funding_status", pd.Series(dtype=str)).eq("COMPLETE").all())
        blockers = []
        if not funding_complete:
            blockers.append("funding_incomplete")
        if strict < minimum_samples:
            blockers.append("strict_l2_sample_target_not_met")
        elif span_minutes < minimum_span_minutes:
            blockers.append("strict_l2_observation_span_not_met")
        rows.append(
            {
                "asset": asset,
                "funding_rows": int(pd.to_numeric(f.get("funding_rows", 0), errors="coerce").fillna(0).max()) if not f.empty else 0,
                "funding_complete": funding_complete,
                "strict_l2_samples": strict,
                "provisional_l2_samples": provisional,
                "strict_l2_span_minutes": span_minutes,
                "strict_l2_cadence_ready": cadence_ready,
                "minimum_strict_l2_samples": minimum_samples,
                "minimum_strict_l2_span_minutes": minimum_span_minutes,
                "latest_l2_at": timestamps.max().isoformat() if not timestamps.empty else "",
                "collection_status": "READY" if not blockers else "COLLECTING",
                "blocker": ";".join(blockers),
                "next_action": "continue_2h_l2_and_funding_collection" if blockers else "maintain_rolling_collection",
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
        "collecting_assets": int(frame.get("collection_status", pd.Series(dtype=str)).eq("COLLECTING").sum()),
        "status": "PASS" if not frame.empty else "BLOCKED",
        "live_trading_authorized": False,
    }


def build_pair_cost_stress_surfaces(*, root: Path = ROOT) -> dict[str, Any]:
    active = root / "reports" / "active"
    source_path = active / "current_wizard_hyperliquid_pair_cost_evidence.csv"
    source = _read_csv(source_path)
    models = source.copy()
    models["strict_observed_cost_ready"] = models.get("cost_acceptance_ready", pd.Series(False, index=models.index)).map(_truthy)
    models["cost_model_status"] = models["strict_observed_cost_ready"].map({True: "STRICT_OBSERVED", False: "BLOCKED_PROVISIONAL_OR_MISSING"})
    models["cost_model_blocker"] = models.get("cost_blocker", "").fillna("")
    models["source_cost_evidence_path"] = _relative(source_path, root)
    models["promotion_authority"] = False
    models["live_trading_authorized"] = False
    model_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    _atomic_csv(models, model_path)
    rows = []
    for row in models.to_dict("records"):
        base = _finite(row.get("estimated_pair_round_trip_cost_bps"))
        for scenario, multiplier, additive in (("base", 1.0, 0.0), ("stress", 1.5, 2.0), ("tail", 2.0, 5.0)):
            cost = base * multiplier + additive if math.isfinite(base) else math.nan
            ready = bool(row.get("strict_observed_cost_ready")) and math.isfinite(cost)
            rows.append(
                {
                    "pair_group_key": _text(row.get("pair_group_key")),
                    "pair": _text(row.get("pair")),
                    "scenario": scenario,
                    "round_trip_cost_bps": cost,
                    "strict_observed_cost_ready": ready,
                    "blocker": "" if ready else _text(row.get("cost_model_blocker")) or "strict_observed_cost_missing",
                    "evidence_path": _text(row.get("evidence_path")),
                    "promotion_authority": False,
                    "live_trading_authorized": False,
                }
            )
    stress = pd.DataFrame(rows)
    stress_path = active / "hyperliquid_cost_stress.csv"
    _atomic_csv(stress, stress_path)
    return {
        "models": model_path,
        "stress": stress_path,
        "pairs": len(models),
        "strict_ready_pairs": int(models["strict_observed_cost_ready"].sum()),
        "status": "PASS" if not models.empty else "BLOCKED",
        "live_trading_authorized": False,
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


def run_cost_evidence_attacks(
    *, root: Path = ROOT, now: datetime | None = None
) -> pd.DataFrame:
    now = _as_utc(now)
    base = {"captured_at": now.isoformat(), "samples": 24, "funding_coverage": 0.99, "base_slippage_bps": 3.0, "stress_slippage_bps": 6.0}
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
    costs = build_pair_cost_stress_surfaces(root=root)
    cost_attacks = run_cost_evidence_attacks(root=root, now=now)
    paths = {
        "market_manifest": Path(market["manifest"]),
        "mapping_audit": Path(market["audit"]),
        "external_source_contract_results": root / "reports" / "red_team" / "external_source_contract_results.csv",
        "history_remediation_queue": Path(history["queue"]),
        "history_coverage": Path(history["coverage"]),
        "l2_capture_candidates": Path(candidates["path"]),
        "cost_collection_status": Path(collection["status_path"]),
        "pair_cost_models": Path(costs["models"]),
        "cost_stress": Path(costs["stress"]),
        "cost_evidence_attack_results": root / "reports" / "red_team" / "cost_evidence_attack_results.csv",
    }
    return CommandResult(
        paths=paths,
        summary={
            "status": "PASS" if market["status"] == "PASS" and history["status"] == "PASS" and costs["status"] == "PASS" and costs["strict_ready_pairs"] > 0 else "BLOCKED",
            "infrastructure_status": "PASS" if market["status"] == "PASS" and history["status"] == "PASS" and costs["status"] == "PASS" else "BLOCKED",
            "mapped_assets": market["assets"],
            "mapped_pairs": market["pairs"],
            "mapping_passes": market["mapping_passes"],
            "history_ready_pairs": history["ready"],
            "history_queued_pairs": history["queued"],
            "history_deferred_pairs": history["deferred"],
            "history_structurally_blocked_pairs": history["structurally_blocked"],
            "history_insufficient_asset_age_pairs": history[
                "insufficient_asset_age"
            ],
            "l2_capture_eligible_pairs": candidates["eligible_pairs"],
            "cost_collection_ready_assets": collection["ready_assets"],
            "cost_collection_collecting_assets": collection["collecting_assets"],
            "strict_cost_ready_pairs": costs["strict_ready_pairs"],
            "source_attack_cases": len(source_attacks),
            "cost_attack_cases": len(cost_attacks),
            "live_trading_authorized": False,
        },
    )


def _history_reason(row: pd.Series) -> str:
    blockers = []
    if not _truthy(row.get("timestamp_parse_valid")):
        blockers.append("timestamp_parse_invalid")
    if not _truthy(row.get("timestamp_bound_valid")):
        blockers.append("future_timestamp_or_cutoff_invalid")
    if int(_finite(row.get("post_cutoff_rows"), default=1)) > 0:
        blockers.append("post_cutoff_rows_present")
    if _finite(row.get("history_rows"), default=0) < _finite(row.get("minimum_history_rows"), default=math.inf):
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
    temporary.replace(path)


def _finite(value: Any, *, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _truthy(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes", "pass", "ready", "complete"}


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    result = build_corrective_data_evidence()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
