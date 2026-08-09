from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from quant_platform.orchestration.corrective_data_evidence import (
    SOURCE_CONTRACTS,
    build_cost_collection_status,
    build_history_remediation,
    build_l2_capture_candidate_set,
    run_cost_evidence_attacks,
    validate_cost_bundle,
    validate_source_frame,
)


NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)


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


def test_source_contract_fails_closed_on_schema_duplicate_and_stale_data():
    valid = _inventory()
    assert validate_source_frame(valid, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW) == []
    assert validate_source_frame(valid.drop(columns=["tradable_perp"]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)
    assert "duplicate_keys" in validate_source_frame(pd.concat([valid, valid]), SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)
    stale = valid.assign(checked_at_utc=(NOW - timedelta(days=3)).isoformat())
    assert "source_stale" in validate_source_frame(stale, SOURCE_CONTRACTS["hyperliquid_inventory"], now=NOW)


def test_cost_contract_rejects_stale_sparse_sign_and_stress_manipulation(tmp_path):
    base = {"captured_at": NOW.isoformat(), "samples": 20, "funding_coverage": 0.99, "base_slippage_bps": 2.0, "stress_slippage_bps": 4.0}
    assert validate_cost_bundle(base, now=NOW) == []
    assert validate_cost_bundle({**base, "samples": 1}, now=NOW)
    assert validate_cost_bundle({**base, "base_slippage_bps": -2.0}, now=NOW)
    assert validate_cost_bundle({**base, "stress_slippage_bps": 1.0}, now=NOW)
    result = run_cost_evidence_attacks(root=tmp_path, now=NOW)
    assert result["status"].eq("PASS").all()
    assert not result["manipulated_evidence_improves_readiness"].any()


def test_registered_hypotheses_are_normalized_for_public_l2_collection(tmp_path):
    active = tmp_path / "reports" / "active"
    processed = tmp_path / "data" / "processed"
    active.mkdir(parents=True)
    processed.mkdir(parents=True)
    pd.DataFrame(
        [{"experiment_id": "exp-1", "confirmation_role": "near_miss_remediation"}]
    ).to_csv(active / "current_hypothesis_batch.csv", index=False)
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
    pd.DataFrame(
        [{"asset": "ETH", "tradable": True}, {"asset": "WIF", "tradable": True}]
    ).to_csv(processed / "hyperliquid_market_context.csv", index=False)

    result = build_l2_capture_candidate_set(root=tmp_path)
    frame = pd.read_csv(result["path"])

    assert result["eligible_pairs"] == 1
    assert frame.loc[0, "asset_x"] == "ETH"
    assert frame.loc[0, "asset_y"] == "WIF"
    assert bool(frame.loc[0, "collection_eligible"]) is True
    assert bool(frame.loc[0, "testnet_order_authority"]) is False


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
