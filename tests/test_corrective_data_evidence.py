from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration import corrective_data_evidence, corrective_l2_scheduler
from quant_platform.orchestration.corrective_data_evidence import (
    SOURCE_CONTRACTS,
    _l2_readiness_refresh_evidence,
    build_cost_collection_status,
    build_history_remediation,
    build_l2_capture_candidate_set,
    build_pair_cost_stress_surfaces,
    run_cost_evidence_attacks,
    validate_cost_bundle,
    validate_pair_cost_bundle_artifacts,
    validate_source_frame,
)
from tests.pair_cost_bundle_support import publish_valid_pair_cost_bundle

NOW = datetime(2026, 8, 9, tzinfo=UTC)


def _inventory() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset": "BTC",
                "asset_index": 0,
                "universe_name": "BTC",
                "sz_decimals": 5,
                "max_leverage": 40,
                "is_delisted": False,
                "tradable_perp": True,
                "checked_at_utc": NOW.isoformat(),
            }
        ]
    )


def test_l2_readiness_refresh_diagnostics_are_optional_and_validated(
    tmp_path, monkeypatch
):
    missing = _l2_readiness_refresh_evidence(root=tmp_path)
    assert missing["validation_status"] == "NOT_AVAILABLE"
    assert missing["receipt_status"] == "NOT_AVAILABLE"

    active = tmp_path / "reports" / "active"
    immutable = tmp_path / "data" / "research" / "l2_readiness_refresh" / "receipt.json"
    active.mkdir(parents=True)
    immutable.parent.mkdir(parents=True)
    immutable.write_text("{}\n", encoding="utf-8")
    payload = {
        "status": "WAITING_STRICT_L2",
        "receipt_id": "l2readiness_test",
        "source_l2_receipt_id": "l2receipt_test",
        "refresh_executed": False,
        "registered_gate_refresh_executed": False,
        "stage4_handoff_refresh_executed": False,
        "ready_pairs": 1,
        "eligible_pairs": 4,
    }
    (active / "corrective_l2_readiness_refresh_status.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    monkeypatch.setattr(
        corrective_l2_scheduler,
        "validate_post_window_readiness_receipt",
        lambda **_: {
            "status": "PASS",
            "blockers": [],
            "immutable_receipt_path": str(immutable.relative_to(tmp_path)),
        },
    )

    evidence = _l2_readiness_refresh_evidence(root=tmp_path)

    assert evidence["validation_status"] == "PASS"
    assert evidence["receipt_status"] == "WAITING_STRICT_L2"
    assert evidence["receipt_id"] == "l2readiness_test"
    assert evidence["ready_pairs"] == 1
    assert evidence["eligible_pairs"] == 4
    assert evidence["immutable_receipt_path"] == immutable


def test_source_contract_fails_closed_on_schema_duplicate_and_stale_data():
    valid = _inventory()
    assert validate_source_frame(valid, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW) == []
    assert validate_source_frame(
        valid.drop(columns=["tradable_perp"]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW
    )
    assert "duplicate_keys" in validate_source_frame(
        pd.concat([valid, valid]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW
    )
    stale = valid.assign(checked_at_utc=(NOW - timedelta(days=3)).isoformat())
    assert "source_stale" in validate_source_frame(
        stale, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW
    )


def test_cost_contract_rejects_stale_sparse_sign_and_stress_manipulation(tmp_path):
    base = {
        "captured_at": NOW.isoformat(),
        "samples": 20,
        "funding_coverage": 0.99,
        "base_slippage_bps": 2.0,
        "stress_slippage_bps": 4.0,
    }
    assert validate_cost_bundle(base, now=NOW) == []
    assert validate_cost_bundle({**base, "samples": 1}, now=NOW)
    assert validate_cost_bundle({**base, "base_slippage_bps": -2.0}, now=NOW)
    assert validate_cost_bundle({**base, "stress_slippage_bps": 1.0}, now=NOW)
    result = run_cost_evidence_attacks(root=tmp_path, now=NOW)
    assert result["status"].eq("PASS").all()
    assert not result["manipulated_evidence_improves_readiness"].any()


@pytest.mark.parametrize(
    ("artifact", "expected_blocker"),
    (
        ("candidate_set", "pair_cost_bundle_candidate_set_hash_mismatch"),
        ("cost_status", "pair_cost_bundle_cost_status_hash_mismatch"),
        ("strict_l2_window", "pair_cost_bundle_strict_l2_window_hash_mismatch"),
        ("fee_profile", "pair_cost_bundle_fee_profile_hash_mismatch"),
    ),
)
def test_pair_cost_bundle_rejects_every_frozen_input_mutation(
    tmp_path, artifact, expected_blocker
):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "hyperliquid|daily|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "asset_x": "ETH",
                "asset_y": "BTC",
            }
        ],
    )
    path = artifacts[artifact]
    path.write_bytes(path.read_bytes() + b"\n")

    blockers = validate_pair_cost_bundle_artifacts(
        root=tmp_path,
        bundle_manifest_path=artifacts["bundle"],
        pair_cost_models_path=artifacts["models"],
    )

    assert expected_blocker in blockers


def test_pair_cost_bundle_allows_hash_bound_l2_assets_outside_model_cohort(tmp_path):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "hyperliquid|daily|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "asset_x": "ETH",
                "asset_y": "BTC",
            }
        ],
        extra_l2_assets=("SOL",),
    )

    blockers = validate_pair_cost_bundle_artifacts(
        root=tmp_path,
        bundle_manifest_path=artifacts["bundle"],
        pair_cost_models_path=artifacts["models"],
    )

    assert blockers == []


def test_pair_cost_bundle_rejects_rehashed_candidate_substitution(tmp_path):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "hyperliquid|daily|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "asset_x": "ETH",
                "asset_y": "BTC",
            }
        ],
    )
    candidates = pd.read_csv(artifacts["candidate_set"], keep_default_na=False)
    candidates.loc[0, "pair"] = "ETH-USD-SOL-USD"
    candidates.to_csv(artifacts["candidate_set"], index=False)
    manifest = json.loads(artifacts["bundle"].read_text(encoding="utf-8"))
    manifest["input_hashes"]["candidate_set_sha256"] = sha256(
        artifacts["candidate_set"].read_bytes()
    ).hexdigest()
    models = pd.read_csv(artifacts["models"], keep_default_na=False)
    for key, digest in manifest["input_hashes"].items():
        models[key] = digest
    identity = {
        "pair_group_key": models.loc[0, "pair_group_key"],
        "model_as_of_utc": manifest["model_as_of_utc"],
        **manifest["input_hashes"],
    }
    models.loc[0, "cost_model_id"] = "hlpaircost_" + sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    models.to_csv(artifacts["models"], index=False)
    manifest["pair_cost_models_sha256"] = sha256(
        artifacts["models"].read_bytes()
    ).hexdigest()
    manifest_core = {key: value for key, value in manifest.items() if key != "receipt_id"}
    manifest["receipt_id"] = "l2costreceipt_" + sha256(
        json.dumps(manifest_core, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    artifacts["bundle"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    blockers = validate_pair_cost_bundle_artifacts(
        root=tmp_path,
        bundle_manifest_path=artifacts["bundle"],
        pair_cost_models_path=artifacts["models"],
    )

    assert "pair_cost_bundle_candidate_model_identity_mismatch" in blockers
    assert "pair_cost_bundle_receipt_identity_invalid" in blockers


def test_pair_cost_bundle_rejects_exploratory_only_model_after_reseal(tmp_path):
    artifacts = publish_valid_pair_cost_bundle(
        root=tmp_path,
        now=NOW,
        specs=[
            {
                "pair_group_key": "hyperliquid|daily|ETH|BTC",
                "pair": "ETH-USD-BTC-USD",
                "asset_x": "ETH",
                "asset_y": "BTC",
            }
        ],
    )
    models = pd.read_csv(artifacts["models"], keep_default_na=False)
    models.loc[0, "strict_observed_cost_ready"] = False
    models.loc[0, "cost_acceptance_ready"] = False
    models.loc[0, "cost_model_status"] = "BLOCKED_PROVISIONAL_OR_MISSING"
    models.loc[0, "cost_model_blocker"] = "exploratory_only"
    models.to_csv(artifacts["models"], index=False)
    manifest = json.loads(artifacts["bundle"].read_text(encoding="utf-8"))
    manifest["pair_cost_models_sha256"] = sha256(
        artifacts["models"].read_bytes()
    ).hexdigest()
    manifest_core = {key: value for key, value in manifest.items() if key != "receipt_id"}
    manifest["receipt_id"] = "l2costreceipt_" + sha256(
        json.dumps(manifest_core, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    artifacts["bundle"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    blockers = validate_pair_cost_bundle_artifacts(
        root=tmp_path,
        bundle_manifest_path=artifacts["bundle"],
        pair_cost_models_path=artifacts["models"],
    )

    assert "pair_cost_bundle_model_readiness_invalid" in blockers


def test_registered_hypotheses_are_normalized_for_public_l2_collection(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame([{"experiment_id": "exp-1", "confirmation_role": "near_miss_remediation"}]).to_csv(
        active / "current_hypothesis_batch.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "experiment_id": "exp-1",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "overall_research_rank": 1,
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_failure_attribution.csv", index=False)
    pd.DataFrame([{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]).to_csv(
        processed / "hyperliquid_market_context.csv", index=False
    )

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])

    assert result["eligible_pairs"] == 1
    assert frame.loc[0, "asset_x"] == "ETH"
    assert frame.loc[0, "asset_y"] == "WIF"
    assert bool(frame.loc[0, "collection_eligible"]) is True
    assert bool(frame.loc[0, "testnet_order_authority"]) is False


def test_current_wizard_selected_pairs_expand_collection_not_stage_two(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    snapshot = tmp_path / "reports" / "snapshots" / "current" / "pair_costs.csv"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "cost_evidence_id": "cwcost-test",
                "pair_group_key": "coinbase|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "selected_for_cost_evidence": True,
            }
        ]
    ).to_csv(snapshot, index=False)
    (active / "current_wizard_hyperliquid_cost_manifest.json").write_text(
        json.dumps(
            {
                "cost_evidence_id": "cwcost-test",
                "selected_pair_group_keys": ["coinbase|daily|ETH|WIF"],
                "artifacts": {
                    "snapshot_pairs": str(snapshot.relative_to(tmp_path)),
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])

    assert result["current_board_collection_pairs"] == 1
    assert result["eligible_pairs"] == 1
    assert result["stage_two_candidate_pairs"] == 0
    assert frame.loc[0, "source_family"] == "current_wizard_cost_selected_collection"
    assert bool(frame.loc[0, "collection_eligible"]) is True
    assert bool(frame.loc[0, "stage_two_candidate"]) is False
    assert bool(frame.loc[0, "testnet_order_authority"]) is False


def test_l2_candidate_builder_uses_hash_bound_failure_route_index(
    tmp_path, monkeypatch
):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [{"experiment_id": "exp-1", "confirmation_role": "near_miss_remediation"}]
    ).to_csv(active / "current_hypothesis_batch.csv", index=False)
    route_path = active / "current_wizard_hyperliquid_failure_attribution_routes.csv"
    pd.DataFrame(
        [
            {
                "experiment_id": "exp-1",
                "pair_group_key": "binance|daily|ETH|WIF",
                "pair": "ETH-USD-WIF-USD",
                "asset_x": "ETH",
                "asset_y": "WIF",
                "overall_research_rank": 1,
            }
        ]
    ).to_csv(route_path, index=False)
    attribution_path = active / "current_wizard_hyperliquid_failure_attribution.csv"
    attribution_path.write_text("must_not_be_read\n", encoding="utf-8")
    manifest_path = active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    manifest_path.write_text(
        json.dumps({"failure_attribution_id": "cwfailure-test"}),
        encoding="utf-8",
    )
    route_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes_manifest.json"
    )
    route_manifest_path.write_text(
        json.dumps(
            {
                "active_route_index_path": "reports/active/current_wizard_hyperliquid_failure_attribution_routes.csv",
                "route_index_sha256": sha256(route_path.read_bytes()).hexdigest(),
                "source_failure_attribution_id": "cwfailure-test",
                "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)
    original_read_csv = corrective_data_evidence._read_csv

    def guarded_read_csv(path):
        assert Path(path) != attribution_path
        return original_read_csv(path)

    monkeypatch.setattr(corrective_data_evidence, "_read_csv", guarded_read_csv)

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])

    assert result["routing_index_used"] is True
    assert result["eligible_pairs"] == 1
    assert frame.loc[0, "pair"] == "ETH-USD-WIF-USD"
    assert "failure_attribution_routes.csv" in frame.loc[0, "evidence_path"]
    manifest_path.write_text(
        json.dumps({"failure_attribution_id": "cwfailure-drifted"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="route-index manifest is incomplete"):
        build_l2_capture_candidate_set(root=tmp_path)


def test_l2_candidate_builder_rejects_tampered_failure_route_index(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    route_path = active / "current_wizard_hyperliquid_failure_attribution_routes.csv"
    route_path.write_text("experiment_id,pair\nexp-1,ETH-WIF\n", encoding="utf-8")
    manifest_path = active / "current_wizard_hyperliquid_failure_attribution_manifest.json"
    manifest_path.write_text(
        json.dumps({"failure_attribution_id": "cwfailure-test"}), encoding="utf-8"
    )
    route_manifest_path = (
        active / "current_wizard_hyperliquid_failure_attribution_routes_manifest.json"
    )
    route_manifest_path.write_text(
        json.dumps(
            {
                "active_route_index_path": "reports/active/current_wizard_hyperliquid_failure_attribution_routes.csv",
                "route_index_sha256": "0" * 64,
                "source_failure_attribution_id": "cwfailure-test",
                "source_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="route-index hash mismatch"):
        build_l2_capture_candidate_set(root=tmp_path)


def test_frozen_registered_candidate_survives_daily_board_rotation(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    research = tmp_path / "data" / "research"
    contracts = research / "registered_rerun_contracts"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    contracts.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_id": "new-exp",
                "semantic_hypothesis_id": "new-hypothesis",
                "confirmation_role": "near_miss_remediation",
            }
        ]
    ).to_csv(active / "current_hypothesis_batch.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "new-exp",
                "pair_group_key": "binance|daily|SOL|WIF",
                "pair": "SOL-USD-WIF-USD",
                "asset_x": "SOL",
                "asset_y": "WIF",
                "overall_research_rank": 1,
            }
        ]
    ).to_csv(active / "current_wizard_hyperliquid_failure_attribution.csv", index=False)
    pd.DataFrame(
        [
            {"asset": asset, "tradable": True}
            for asset in ("SOL", "WIF", "ETH", "PYTH")
        ]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)
    contract = {
        "contract_id": "registeredrerun-frozen",
        "immutable_contract_path": (
            "data/research/registered_rerun_contracts/registeredrerun-frozen.json"
        ),
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "frozen-hypothesis",
                "source_experiment_id": "frozen-exp",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "pair": "ETH-USD-PYTH-USD",
                "ledger_record_hash": "ledger-frozen",
            }
        ],
    }
    (active / "registered_research_rerun_contract.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (contracts / "registeredrerun-frozen.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (research / "hypothesis_ledger.jsonl").write_text(
        json.dumps(
            {
                "semantic_hypothesis_id": "frozen-hypothesis",
                "source_experiment_id": "frozen-exp",
                "record_hash": "ledger-frozen",
                "semantic_material": {
                    "assets": ["ETH", "PYTH"],
                    "execution_venue": "hyperliquid",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])

    assert result["registered_contract_candidates"] == 1
    assert result["candidate_pairs"] == 2
    assert result["stage_two_candidate_pairs"] == 1
    frozen = frame.loc[frame["pair_group_key"].eq("dydx|daily|ETH|PYTH")].iloc[0]
    current = frame.loc[frame["pair_group_key"].eq("binance|daily|SOL|WIF")].iloc[0]
    assert bool(frozen["registered_contract_candidate"]) is True
    assert bool(frozen["stage_two_candidate"]) is True
    assert frozen["source_family"] == "registered_rerun_contract"
    assert bool(current["stage_two_candidate"]) is False
    assert not frame["testnet_order_authority"].astype(bool).any()


def test_exhaustive_research_rankable_pairs_remain_diagnostic_only(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_id": "exhaustive-exp-1",
                "pair_group_id": "dydx|daily|ADA|ALGO",
                "pair": "ADA-USD-ALGO-USD",
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "research_rank_eligible": True,
                "profit_factor": 1.31,
                "sharpe": 0.23,
                "total_return": 0.11,
            },
            {
                "experiment_id": "exhaustive-exp-2",
                "pair_group_id": "dydx|daily|ADA|ALGO",
                "pair": "ALGO-USD-ADA-USD",
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "research_rank_eligible": False,
                "profit_factor": 0.98,
                "sharpe": -0.02,
                "total_return": -0.09,
            },
        ]
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_canonical_replay.csv",
        index=False,
    )
    pd.DataFrame([{"asset": "ADA", "tradable": True}, {"asset": "ALGO", "tradable": True}]).to_csv(
        processed / "hyperliquid_market_context.csv", index=False
    )

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])
    diagnostic = pd.read_csv(result["historical_diagnostics_path"])

    assert result["eligible_pairs"] == 0
    assert frame.empty
    assert diagnostic.loc[0, "pair"] == "ADA-USD-ALGO-USD"
    assert diagnostic.loc[0, "source_family"] == "historical_exhaustive_family"
    assert bool(diagnostic.loc[0, "collection_eligible"]) is False
    assert (
        "historical_exhaustive_family_not_current_registration_authority"
        in diagnostic.loc[0, "blocker"]
    )
    assert bool(diagnostic.loc[0, "testnet_order_authority"]) is False


def test_cost_collection_counts_distinct_observations_and_requires_time_span(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "acceptance_policy_manifest.json").write_text(
        '{"cost_gates":{"strict_l2_window_hours":2,"minimum_strict_l2_samples":12,"minimum_strict_l2_span_minutes":100}}'
    )
    pd.DataFrame([{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv", index=False
    )
    timestamps = [NOW - timedelta(minutes=5 * offset) for offset in range(12)]
    rows = [
        {
            "pair": pair,
            "asset": "ETH",
            "source_timestamp": timestamp.isoformat(),
            "buy_complete": True,
            "sell_complete": True,
        }
        for pair in ("ETH-WIF", "ETH-BTC")
        for timestamp in timestamps
    ]
    pd.DataFrame(rows).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_cost_collection_status(root=tmp_path, now=NOW)
    row = pd.read_csv(result["status_path"]).iloc[0]

    assert row["strict_l2_samples"] == 12
    assert row["strict_l2_span_minutes"] == 55
    assert bool(row["strict_l2_cadence_ready"]) is False
    assert "strict_l2_observation_span_not_met" in row["blocker"]


def test_cost_collection_uses_local_receipt_only_for_blank_vendor_timestamp(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "acceptance_policy_manifest.json").write_text(
        '{"cost_gates":{"strict_l2_window_hours":2,"minimum_strict_l2_samples":12,"minimum_strict_l2_span_minutes":100}}'
    )
    pd.DataFrame([{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    timestamps = [NOW - timedelta(minutes=10 * offset) for offset in range(12)]
    rows = []
    for index, timestamp in enumerate(timestamps):
        rows.append(
            {
                "asset": "ETH",
                "source_timestamp": "" if index == 5 else timestamp.isoformat(),
                "captured_at": timestamp.isoformat(),
                "buy_complete": True,
                "sell_complete": True,
            }
        )
    rows.extend(
        [
            {
                "asset": "ETH",
                "source_timestamp": "not-a-timestamp",
                "captured_at": (NOW - timedelta(minutes=7)).isoformat(),
                "buy_complete": True,
                "sell_complete": True,
            },
            {
                "asset": "ETH",
                "source_timestamp": "",
                "captured_at": "",
                "buy_complete": True,
                "sell_complete": True,
            },
            {
                "asset": "ETH",
                "source_timestamp": (NOW + timedelta(minutes=1)).isoformat(),
                "captured_at": NOW.isoformat(),
                "buy_complete": True,
                "sell_complete": True,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_cost_collection_status(root=tmp_path, now=NOW)
    row = pd.read_csv(result["status_path"]).iloc[0]

    assert row["strict_l2_samples"] == 12
    assert row["strict_l2_local_capture_timestamp_fallbacks"] == 1
    assert row["strict_l2_span_minutes"] == 110
    assert bool(row["strict_l2_cadence_ready"]) is True
    assert row["l2_timestamp_policy"] == (
        "vendor_source_timestamp_else_local_capture_receipt_for_cadence_only"
    )
    assert row["collection_status"] == "READY"


@pytest.mark.parametrize(
    ("minutes_ago", "blocker"),
    [
        (
            [110 - offset for offset in range(11)] + [0],
            "strict_l2_maximum_gap_exceeded",
        ),
        (
            [16 + 9.3 * offset for offset in range(12)],
            "strict_l2_latest_observation_stale",
        ),
    ],
)
def test_cost_collection_rejects_strict_l2_cadence_holes(
    tmp_path: Path, minutes_ago: list[float], blocker: str
) -> None:
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "acceptance_policy_manifest.json").write_text(
        '{"cost_gates":{"strict_l2_window_hours":2,"minimum_strict_l2_samples":12,'
        '"minimum_strict_l2_span_minutes":100,"maximum_strict_l2_gap_minutes":15}}',
        encoding="utf-8",
    )
    pd.DataFrame(
        [{"asset": "ETH", "funding_status": "COMPLETE", "funding_rows": 500}]
    ).to_csv(active / "current_wizard_hyperliquid_funding_asset_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": "ETH",
                "source_timestamp": (NOW - timedelta(minutes=minutes)).isoformat(),
                "buy_complete": True,
                "sell_complete": True,
            }
            for minutes in minutes_ago
        ]
    ).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_cost_collection_status(root=tmp_path, now=NOW)
    row = pd.read_csv(result["status_path"]).iloc[0]

    assert row["strict_l2_samples"] == 12
    assert row["strict_l2_span_minutes"] >= 100
    assert bool(row["strict_l2_cadence_ready"]) is False
    assert row["collection_status"] == "COLLECTING"
    assert blocker in row["blocker"]


def test_cost_collection_uses_exhaustive_funding_with_explicit_provenance(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "acceptance_policy_manifest.json").write_text(
        '{"cost_gates":{"strict_l2_window_hours":2,"minimum_strict_l2_samples":12,"minimum_strict_l2_span_minutes":100}}'
    )
    pd.DataFrame([{"asset": "ADA", "funding_status": "BLOCKED", "funding_rows": 0}]).to_csv(
        active / "current_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "asset": "ADA",
                "funding_status": "COMPLETE",
                "funding_rows": 24537,
                "fetch_complete_flag": True,
                "timestamp_parse_valid": True,
                "post_cutoff_rows": 0,
                "funding_path": "reports/snapshots/exhaustive/ADA_funding.json",
            }
        ]
    ).to_csv(
        active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "asset": "ADA",
                "source_timestamp": NOW.isoformat(),
                "buy_complete": True,
                "sell_complete": True,
            }
        ]
    ).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_cost_collection_status(root=tmp_path, now=NOW)
    row = pd.read_csv(result["status_path"]).iloc[0]

    assert bool(row["funding_complete"]) is True
    assert row["funding_rows"] == 24537
    assert row["funding_complete_source_count"] == 1
    assert (
        row["funding_evidence_path"]
        == "reports/active/exhaustive_wizard_hyperliquid_funding_asset_results.csv"
    )
    assert row["funding_detail_path"] == ("reports/snapshots/exhaustive/ADA_funding.json")
    assert "funding_incomplete" not in row["blocker"]
    assert "strict_l2_sample_target_not_met" in row["blocker"]

    invalid_exhaustive = pd.read_csv(
        active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv"
    )
    invalid_exhaustive["post_cutoff_rows"] = 1
    invalid_exhaustive.to_csv(
        active / "exhaustive_wizard_hyperliquid_funding_asset_results.csv",
        index=False,
    )

    invalid_result = build_cost_collection_status(root=tmp_path, now=NOW)
    invalid_row = pd.read_csv(invalid_result["status_path"]).iloc[0]

    assert bool(invalid_row["funding_complete"]) is False
    assert invalid_row["funding_rows"] == 0
    assert "funding_incomplete" in invalid_row["blocker"]


def test_pair_cost_models_cover_every_registered_candidate_lane(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    config.mkdir(parents=True)
    (config / "acceptance_policy_manifest.json").write_text(
        '{"cost_gates":{"strict_l2_window_hours":2}}', encoding="utf-8"
    )
    (config / "hyperliquid_perp_cost_profile.json").write_text(
        json.dumps(
            {
                "profile_id": "test-profile",
                "taker_fee_bps": 4.5,
                "execution_risk_bps": 2.0,
                "fee_source_url": "https://example.test/fees",
                "fee_source_checked_at": "2026-08-01",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "pair_group_key": "current-pair",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "confirmation_role": "near_miss_remediation",
                "overall_research_rank": 1,
                "collection_eligible": True,
            },
            {
                "pair_group_key": "exhaustive-pair",
                "pair": "XLM-USD-HBAR-USD",
                "asset_x": "XLM",
                "asset_y": "HBAR",
                "confirmation_role": "exhaustive_research_rank_eligible",
                "overall_research_rank": 1001,
                "collection_eligible": True,
            },
        ]
    ).to_csv(active / "corrective_l2_capture_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "asset": asset,
                "funding_complete": True,
                "funding_rows": 500,
                "strict_l2_cadence_ready": True,
                "strict_l2_samples": 12,
                "strict_l2_span_minutes": 100.0,
                "strict_l2_max_gap_minutes": 100 / 11,
                "strict_l2_latest_age_minutes": 0.0,
                "minimum_strict_l2_samples": 12,
                "minimum_strict_l2_span_minutes": 100.0,
                "maximum_strict_l2_gap_minutes": 15.0,
                "l2_timestamp_policy": (
                    "vendor_source_timestamp_else_local_capture_receipt_for_cadence_only"
                ),
                "collection_status": "READY",
                "funding_evidence_path": f"funding/{asset}.csv",
            }
            for asset in ("ETH", "PYTH", "XLM", "HBAR")
        ]
    ).to_csv(active / "hyperliquid_cost_collection_status.csv", index=False)
    timestamps = [NOW - timedelta(minutes=100 * index / 11) for index in range(12)]
    l2_samples = pd.DataFrame(
        [
            {
                "asset": asset,
                "notional_usd": 1000.0,
                "source_timestamp": timestamp.isoformat(),
                "captured_at": timestamp.isoformat(),
                "one_way_slippage_bps": 1.0 + asset_index / 10,
                "top_of_book_spread_bps": 0.5,
                "buy_available_notional_usd": 5000.0,
                "sell_available_notional_usd": 4500.0,
                "buy_complete": True,
                "sell_complete": True,
                "blocker": "",
            }
            for asset_index, asset in enumerate(("ETH", "PYTH", "XLM", "HBAR"))
            for timestamp in timestamps
        ]
    )
    fallback_row = l2_samples.loc[
        l2_samples["asset"].eq("ETH") & l2_samples["source_timestamp"].eq(timestamps[5].isoformat())
    ].index[0]
    l2_samples.loc[fallback_row, "source_timestamp"] = ""
    l2_samples.to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)

    result = build_pair_cost_stress_surfaces(root=tmp_path, now=NOW)
    models = pd.read_csv(result["models"])
    stress = pd.read_csv(result["stress"])

    assert result["eligible_pairs"] == 2
    assert result["strict_ready_pairs"] == 2
    assert result["stage_two_eligible_pairs"] == 2
    assert result["stage_two_strict_ready_pairs"] == 2
    assert result["stage_two_acceptance_status"] == "PASS"
    assert result["bundle_manifest"].is_file()
    assert result["bundle_pointer"].is_file()
    assert result["bundle_pointer_snapshot"].is_file()
    assert result["model_snapshot"].is_file()
    assert result["stress_snapshot"].is_file()
    bundle_manifest = json.loads(result["bundle_manifest"].read_text())
    bundle_pointer = json.loads(result["bundle_pointer"].read_text())
    assert result["bundle_pointer_snapshot"].read_bytes() == result[
        "bundle_pointer"
    ].read_bytes()
    assert bundle_manifest["pair_cost_model_rows"] == 2
    assert bundle_pointer["schema_version"] == "thewiz.l2_cost_model_pointer.v1"
    assert bundle_pointer["bundle_id"] == result["bundle_id"]
    assert bundle_pointer["bundle_manifest_sha256"] == sha256(
        result["bundle_manifest"].read_bytes()
    ).hexdigest()
    assert bundle_pointer["pair_cost_models_sha256"] == sha256(
        result["model_snapshot"].read_bytes()
    ).hexdigest()
    assert bundle_pointer["active_pair_cost_models_sha256"] == sha256(
        result["models"].read_bytes()
    ).hexdigest()
    assert bundle_pointer["promotion_authority"] is False
    assert bundle_pointer["testnet_order_authority"] is False
    assert bundle_pointer["live_trading_authorized"] is False
    strict_window_path = tmp_path / bundle_manifest["input_paths"]["strict_l2_window"]
    assert bundle_manifest["input_hashes"]["l2_samples_sha256"] == (
        sha256(strict_window_path.read_bytes()).hexdigest()
    )
    assert result["acceptance_status"] == "PASS"
    assert set(models["pair_group_key"]) == {"current-pair", "exhaustive-pair"}
    assert models["strict_observed_cost_ready"].astype(bool).all()
    assert models["fee_profile_sha256"].str.len().eq(64).all()
    assert models["l2_samples_sha256"].str.len().eq(64).all()
    current = models.loc[models["pair_group_key"].eq("current-pair")].iloc[0]
    assert current["strict_l2_local_capture_timestamp_fallbacks_x"] == 1
    assert current["strict_l2_local_capture_timestamp_fallbacks_y"] == 0
    assert current["l2_timestamp_policy"] == (
        "vendor_source_timestamp_else_local_capture_receipt_for_cadence_only"
    )
    assert len(stress) == 6
    assert not stress["testnet_order_authority"].astype(bool).any()
    assert not stress["live_trading_authorized"].astype(bool).any()

    l2 = pd.read_csv(processed / "hyperliquid_l2_slippage_samples.csv")
    pd.concat(
        [l2.loc[l2["asset"] != "HBAR"], l2.loc[l2["asset"] == "HBAR"].head(1)],
        ignore_index=True,
    ).to_csv(processed / "hyperliquid_l2_slippage_samples.csv", index=False)
    blocked = build_pair_cost_stress_surfaces(root=tmp_path, now=NOW)
    blocked_models = pd.read_csv(blocked["models"])
    exhaustive = blocked_models.loc[blocked_models["pair_group_key"].eq("exhaustive-pair")].iloc[0]

    assert blocked["bundle_id"] != result["bundle_id"]
    assert blocked["strict_ready_pairs"] == 1
    assert blocked["acceptance_status"] == "BLOCKED"
    assert "asset_y_strict_l2_sample_target_not_met:HBAR" in exhaustive["cost_model_blocker"]


def test_history_remediation_excludes_intentionally_deferred_and_structural_rows(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    common = {
        "timestamp_parse_valid": True,
        "timestamp_bound_valid": True,
        "post_cutoff_rows": 0,
        "minimum_history_rows": 100,
        "history_rows": 0,
        "history_blocker": "missing_history",
    }
    pd.DataFrame(
        [
            {
                **common,
                "pair_group_key": "a",
                "selected_for_materialization": True,
                "history_status": "BLOCKED_FETCH",
            },
            {
                **common,
                "pair_group_key": "b",
                "selected_for_materialization": False,
                "history_status": "DEFERRED_NOT_SELECTED",
            },
            {
                **common,
                "pair_group_key": "c",
                "selected_for_materialization": False,
                "history_status": "BLOCKED_HANDOFF_REQUEST",
            },
            {
                **common,
                "pair_group_key": "d",
                "selected_for_materialization": True,
                "history_status": "BLOCKED",
                "history_blocker": "insufficient_point_in_time_history",
            },
        ]
    ).to_csv(active / "current_wizard_hyperliquid_pair_history_results.csv", index=False)

    result = build_history_remediation(root=tmp_path)
    coverage = pd.read_csv(result["coverage"])
    queue = pd.read_csv(result["queue"])

    assert result["queued"] == 1
    assert result["deferred"] == 1
    assert result["structurally_blocked"] == 1
    assert result["insufficient_asset_age"] == 1
    assert queue["pair_group_key"].tolist() == ["a"]
    assert set(coverage["history_coverage_class"]) == {
        "ACTIVE_REMEDIATION",
        "DEFERRED_NOT_SELECTED",
        "STRUCTURALLY_BLOCKED",
        "INSUFFICIENT_ASSET_AGE",
    }
