from __future__ import annotations

import json
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_data_evidence import (
    L2_TIMESTAMP_POLICY,
    _strict_asset_cost_statistics,
)


def publish_valid_pair_cost_bundle(
    *,
    root: Path,
    now,
    specs: list[dict[str, Any]],
    extra_l2_assets: tuple[str, ...] = (),
) -> dict[str, Path]:
    """Publish a minimal production-shaped immutable pair-cost bundle for tests."""

    candidate_rows = []
    for rank, spec in enumerate(specs, start=1):
        candidate_rows.append(
            {
                "experiment_id": spec.get("experiment_id", f"experiment-{rank}"),
                "pair_group_key": spec["pair_group_key"],
                "pair": spec["pair"],
                "asset_x": spec["asset_x"],
                "asset_y": spec["asset_y"],
                "overall_research_rank": rank,
                "confirmation_role": spec.get("confirmation_role", "test_fixture"),
                "source_family": spec.get("source_family", "test_source_family"),
                "semantic_hypothesis_id": spec.get(
                    "semantic_hypothesis_id", f"hypothesis-{rank}"
                ),
                "registered_contract_id": spec.get("registered_contract_id", ""),
                "registered_contract_candidate": spec.get(
                    "registered_contract_candidate", False
                ),
                "stage_two_candidate": True,
                "collection_eligible": True,
                "blocker": "",
                "evidence_path": "test_fixture",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    model_assets = {
            str(value).upper()
            for row in candidate_rows
            for value in (row["asset_x"], row["asset_y"])
    }
    assets = sorted(model_assets | {asset.upper() for asset in extra_l2_assets})
    l2_rows = []
    for asset_number, asset in enumerate(assets, start=1):
        for sample_number in range(3):
            timestamp = now - timedelta(minutes=2 - sample_number)
            slippage = float(asset_number + sample_number) / 10.0
            l2_rows.append(
                {
                    "sample_id": f"{asset}-{sample_number}",
                    "pair": f"{asset}-USD",
                    "asset": asset,
                    "venue": "hyperliquid",
                    "notional_usd": 1_000.0,
                    "source_timestamp": timestamp.isoformat(),
                    "captured_at": timestamp.isoformat(),
                    "best_bid": 99.0,
                    "best_ask": 101.0,
                    "mid_price": 100.0,
                    "top_of_book_spread_bps": slippage / 2.0,
                    "buy_complete": True,
                    "buy_average_price": 100.0,
                    "buy_slippage_bps": slippage,
                    "buy_available_notional_usd": 2_000.0 + asset_number,
                    "sell_complete": True,
                    "sell_average_price": 100.0,
                    "sell_slippage_bps": slippage,
                    "sell_available_notional_usd": 1_900.0 + asset_number,
                    "one_way_slippage_bps": slippage,
                    "book_levels_bid": 5,
                    "book_levels_ask": 5,
                    "blocker": "",
                    "evidence_path": "test_fixture",
                    "effective_l2_observation_timestamp": timestamp.isoformat(),
                    "l2_local_capture_timestamp_fallback": False,
                    "l2_observation_timestamp_source": "vendor_source_timestamp",
                }
            )
    l2 = pd.DataFrame(l2_rows)
    l2_for_stats = l2.copy()
    l2_for_stats["effective_l2_observation_timestamp"] = pd.to_datetime(
        l2_for_stats["effective_l2_observation_timestamp"], utc=True
    )
    stats = {
        asset: _strict_asset_cost_statistics(l2_for_stats, asset) for asset in assets
    }
    statuses = pd.DataFrame(
        [
            {
                "asset": asset,
                "funding_rows": 100,
                "funding_complete": True,
                "funding_complete_source_count": 1,
                "funding_evidence_path": "test_fixture",
                "funding_detail_path": "test_fixture",
                "strict_l2_samples": stats[asset]["samples"],
                "provisional_l2_samples": 0,
                "strict_l2_local_capture_timestamp_fallbacks": 0,
                "provisional_l2_local_capture_timestamp_fallbacks": 0,
                "strict_l2_span_minutes": stats[asset]["span_minutes"],
                "strict_l2_cadence_ready": True,
                "minimum_strict_l2_samples": 3,
                "minimum_strict_l2_span_minutes": 2.0,
                "l2_timestamp_policy": L2_TIMESTAMP_POLICY,
                "latest_l2_at": stats[asset]["end_at"],
                "latest_l2_vendor_at": stats[asset]["end_at"],
                "collection_status": "READY",
                "blocker": "",
                "next_action": "ready",
                "live_trading_authorized": False,
            }
            for asset in assets
        ]
    )
    profile = {
        "profile_id": "hyperliquid-test-fee-profile",
        "venue": "hyperliquid",
        "instrument": "perpetual",
        "fee_tier_assumption": "test",
        "execution_style": "taker",
        "taker_fee_bps": 4.5,
        "maker_fee_bps": 1.5,
        "execution_risk_bps": 2.0,
        "fee_source_url": "https://example.test/hyperliquid-fees",
        "fee_source_checked_at": now.date().isoformat(),
        "account_specific_fee_status": "test_fixture",
        "slippage_method": "public_l2_book_depth",
    }
    candidate_bytes = candidates.to_csv(index=False).encode("utf-8")
    status_bytes = statuses.to_csv(index=False).encode("utf-8")
    l2_bytes = l2.to_csv(index=False).encode("utf-8")
    profile_bytes = (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    input_hashes = {
        "candidate_set_sha256": sha256(candidate_bytes).hexdigest(),
        "cost_status_sha256": sha256(status_bytes).hexdigest(),
        "l2_samples_sha256": sha256(l2_bytes).hexdigest(),
        "fee_profile_sha256": sha256(profile_bytes).hexdigest(),
        "l2_source_file_sha256": "f" * 64,
    }
    model_as_of = now.isoformat()
    bundle_material = {"model_as_of_utc": model_as_of, **input_hashes}
    bundle_id = "l2costbundle_" + sha256(
        json.dumps(bundle_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    bundle_dir = root / "data" / "research" / "l2_cost_model_receipts" / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    input_payloads = {
        "candidate_set": (bundle_dir / "candidate_set.csv", candidate_bytes),
        "cost_status": (bundle_dir / "cost_status.csv", status_bytes),
        "strict_l2_window": (bundle_dir / "strict_l2_window.csv", l2_bytes),
        "fee_profile": (bundle_dir / "fee_profile.json", profile_bytes),
    }
    for path, payload in input_payloads.values():
        path.write_bytes(payload)

    models = []
    for candidate in candidate_rows:
        asset_x = str(candidate["asset_x"]).upper()
        asset_y = str(candidate["asset_y"]).upper()
        stats_x = stats[asset_x]
        stats_y = stats[asset_y]
        pair_slippage = (
            stats_x["slippage_p95_bps"] + stats_y["slippage_p95_bps"]
        ) / 2.0
        identity_payload = {
            "pair_group_key": candidate["pair_group_key"],
            "model_as_of_utc": model_as_of,
            **input_hashes,
        }
        models.append(
            {
                "schema_version": "thewiz.registered_pair_cost_model.v1",
                "cost_model_id": "hlpaircost_"
                + sha256(
                    json.dumps(identity_payload, sort_keys=True).encode("utf-8")
                ).hexdigest()[:20],
                "model_as_of_utc": model_as_of,
                "pair_group_key": candidate["pair_group_key"],
                "pair": candidate["pair"],
                "execution_venue": "hyperliquid",
                "instrument": "perpetual",
                "asset_x": asset_x,
                "asset_y": asset_y,
                "confirmation_role": candidate["confirmation_role"],
                "source_family": candidate["source_family"],
                "semantic_hypothesis_id": candidate["semantic_hypothesis_id"],
                "registered_contract_id": candidate["registered_contract_id"],
                "registered_contract_candidate": candidate[
                    "registered_contract_candidate"
                ],
                "stage_two_candidate": True,
                "overall_research_rank": candidate["overall_research_rank"],
                "selected_for_cost_evidence": True,
                "funding_x_complete": True,
                "funding_y_complete": True,
                "funding_x_rows": 100,
                "funding_y_rows": 100,
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
                "strict_l2_start_at": min(stats_x["start_at"], stats_y["start_at"]),
                "strict_l2_end_at": max(stats_x["end_at"], stats_y["end_at"]),
                "slippage_x_p95_bps": stats_x["slippage_p95_bps"],
                "slippage_y_p95_bps": stats_y["slippage_p95_bps"],
                "pair_one_way_slippage_bps": pair_slippage,
                "spread_x_p95_bps": stats_x["spread_p95_bps"],
                "spread_y_p95_bps": stats_y["spread_p95_bps"],
                "available_notional_x_p05_usd": stats_x[
                    "available_notional_p05_usd"
                ],
                "available_notional_y_p05_usd": stats_y[
                    "available_notional_p05_usd"
                ],
                "notional_per_leg_usd": 1_000.0,
                "fee_profile_id": profile["profile_id"],
                "fee_source_checked_at": profile["fee_source_checked_at"],
                "fee_profile_fresh": True,
                "taker_fee_bps": profile["taker_fee_bps"],
                "execution_risk_bps": profile["execution_risk_bps"],
                "cost_normalization": "pair_gross_capital_two_legs_open_and_close",
                "funding_cost_treatment": "direction_and_hold_dependent_applied_in_point_in_time_replay",
                "estimated_pair_round_trip_cost_bps": 2.0
                * (
                    profile["taker_fee_bps"]
                    + pair_slippage
                    + profile["execution_risk_bps"]
                ),
                "cost_acceptance_ready": True,
                "strict_observed_cost_ready": True,
                "cost_model_status": "STRICT_OBSERVED",
                "cost_model_blocker": "",
                **input_hashes,
                "source_cost_evidence_path": "test_fixture",
                "evidence_path": "test_fixture",
                "promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    model_frame = pd.DataFrame(models)
    model_path = bundle_dir / "pair_cost_models.csv"
    model_frame.to_csv(model_path, index=False)
    active_path = root / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    active_path.parent.mkdir(parents=True, exist_ok=True)
    model_frame.to_csv(active_path, index=False)
    model_hash = sha256(model_path.read_bytes()).hexdigest()
    manifest_core = {
        "schema_version": "thewiz.l2_cost_model_receipt.v1",
        "bundle_id": bundle_id,
        "model_as_of_utc": model_as_of,
        "input_paths": {
            key: str(path.relative_to(root))
            for key, (path, _) in input_payloads.items()
        },
        "input_hashes": input_hashes,
        "pair_cost_models_path": str(model_path.relative_to(root)),
        "pair_cost_models_sha256": model_hash,
        "cost_stress_path": "",
        "cost_stress_sha256": "",
        "pair_cost_model_rows": len(model_frame),
        "cost_stress_rows": 0,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    manifest = {
        **manifest_core,
        "receipt_id": "l2costreceipt_"
        + sha256(json.dumps(manifest_core, sort_keys=True).encode("utf-8")).hexdigest()[
            :20
        ],
    }
    manifest_path = bundle_dir / "receipt.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pointer_material = {
        "schema_version": "thewiz.l2_cost_model_pointer.v1",
        "generated_at_utc": model_as_of,
        "bundle_id": bundle_id,
        "bundle_manifest_path": str(manifest_path.relative_to(root)),
        "bundle_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
        "pair_cost_models_path": str(model_path.relative_to(root)),
        "pair_cost_models_sha256": model_hash,
        "active_pair_cost_models_path": str(active_path.relative_to(root)),
        "active_pair_cost_models_sha256": sha256(active_path.read_bytes()).hexdigest(),
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
    pointer_snapshot_path.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pointer_path = (
        root / "reports" / "active" / "hyperliquid_pair_cost_bundle_pointer.json"
    )
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(
        json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "active": active_path,
        "bundle": manifest_path,
        "models": model_path,
        "pointer": pointer_path,
        "pointer_snapshot": pointer_snapshot_path,
        **{key: path for key, (path, _) in input_payloads.items()},
    }
