from __future__ import annotations

import hashlib
import json

import pandas as pd

from quant_platform.orchestration.corrective_agent_governance import (
    build_agent_authority_inventory,
    build_learning_label_contract,
    build_model_authority_status,
    run_agent_authority_fuzz_tests,
    validate_agent_packet,
)


def _packet():
    return {
        "candidate_id": "candidate_1234567890abcdef1234",
        "model_version": "v1",
        "feature_schema_version": "f1",
        "confidence": 0.5,
        "label_source": "backtest_label",
        "feature_timestamp": "2026-08-01T00:00:00Z",
        "label_timestamp": "2026-08-02T00:00:00Z",
        "evidence_hash": "a" * 64,
        "requested_action": "research",
    }


def test_agent_packets_cannot_request_execution():
    assert validate_agent_packet(_packet()) == []
    assert "agent_execution_action_forbidden" in validate_agent_packet({**_packet(), "requested_action": "submit_live_order"})


def test_inventory_has_no_current_live_submit_authority(tmp_path):
    frame = build_agent_authority_inventory(root=tmp_path)
    assert not frame["can_submit_live_orders"].any()
    assert not frame["can_change_acceptance_policy"].any()
    assert not frame["can_override_veto"].any()


def test_forged_packets_all_fail_closed(tmp_path):
    frame = run_agent_authority_fuzz_tests(root=tmp_path)
    assert frame["status"].eq("PASS").all()
    assert not frame["live_trading_authorized"].any()


def test_learning_label_contract_requires_compounded_fractional_returns(tmp_path):
    path = tmp_path / "reports" / "ml" / "leakage_audit.csv"
    path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "feature_timestamp": "2026-08-01T00:00:00Z",
                "label_timestamp": "2026-08-02T00:00:00Z",
                "uses_future_data": False,
                "uses_dashboard_hindsight": False,
                "leakage_blocker": "",
                "profit_after_cost": -0.25,
                "return_unit": "fraction_of_equity",
                "return_aggregation": "compounded_bar_returns_zero_floor",
            }
        ]
    ).to_csv(path, index=False)
    dataset_path = tmp_path / "data" / "ml" / "trade_training_dataset.csv"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "pair": "ETH-PYTH",
                "timeframe": "1d",
                "source_venue": "hyperliquid",
                "regime": "range",
            }
        ]
    ).to_csv(dataset_path, index=False)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    (dataset_path.parent / "active_trade_dataset.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE_RESEARCH_DATASET",
                "dataset_id": "tradedataset-test",
                "active_dataset_path": "data/ml/trade_training_dataset.csv",
                "active_dataset_sha256": dataset_hash,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "selection_status": "SELECTED_CANONICAL",
                "source_venue": "hyperliquid",
            }
        ]
    ).to_csv(path.parent / "trade_dataset_source_selection.csv", index=False)

    result = build_learning_label_contract(root=tmp_path)

    assert result["summary"]["status"] == "PASS"
    assert result["summary"]["invalid_fractional_return_rows"] == 0
    assert result["summary"]["compounded_return_rows"] == 1
    assert result["summary"]["selected_canonical_hyperliquid_histories"] == 1
    assert result["summary"]["dataset_hash_matches_pointer"] is True


def test_learning_label_contract_rejects_post_outcome_memory_features(tmp_path):
    reports = tmp_path / "reports" / "ml"
    reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "feature_timestamp": "2026-08-01T00:00:00Z",
                "label_timestamp": "2026-08-02T00:00:00Z",
                "uses_future_data": False,
                "uses_dashboard_hindsight": False,
                "leakage_blocker": "",
                "profit_after_cost": 0.01,
                "return_unit": "fraction_of_equity",
                "return_aggregation": "compounded_bar_returns_zero_floor",
            }
        ]
    ).to_csv(reports / "leakage_audit.csv", index=False)
    dataset_path = tmp_path / "data" / "ml" / "trade_training_dataset.csv"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_id": "trade-1",
                "pair": "ETH-PYTH",
                "timeframe": "1d",
                "source_venue": "hyperliquid",
                "regime": "range",
                "shared_outcome_win_rate": 0.75,
            }
        ]
    ).to_csv(dataset_path, index=False)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    (dataset_path.parent / "active_trade_dataset.json").write_text(
        json.dumps(
            {
                "status": "ACTIVE_RESEARCH_DATASET",
                "dataset_id": "tradedataset-test",
                "active_dataset_path": "data/ml/trade_training_dataset.csv",
                "active_dataset_sha256": dataset_hash,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"selection_status": "SELECTED_CANONICAL"}]).to_csv(
        reports / "trade_dataset_source_selection.csv", index=False
    )

    result = build_learning_label_contract(root=tmp_path)

    assert result["summary"]["status"] == "BLOCKED"
    assert "post_outcome_memory_features_present" in result["summary"][
        "lineage_blockers"
    ]


def test_learning_label_contract_rejects_duplicate_conflicting_trade_identity(
    tmp_path,
):
    reports = tmp_path / "reports" / "ml"
    reports.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "feature_timestamp": "2026-08-01T00:00:00Z",
                "label_timestamp": "2026-08-02T00:00:00Z",
                "uses_future_data": False,
                "uses_dashboard_hindsight": False,
                "leakage_blocker": "",
                "profit_after_cost": 0.01,
                "return_unit": "fraction_of_equity",
                "return_aggregation": "compounded_bar_returns_zero_floor",
            }
        ]
    ).to_csv(reports / "leakage_audit.csv", index=False)
    dataset_path = tmp_path / "data" / "ml" / "trade_training_dataset.csv"
    dataset_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "trade_id": "duplicate",
                "pair": "ETH-PYTH",
                "timeframe": "1d",
                "source_venue": "hyperliquid",
                "regime": regime,
            }
            for regime in ("range", "crisis")
        ]
    ).to_csv(dataset_path, index=False)
    pd.DataFrame(
        [{"selection_status": "SELECTED_CANONICAL"}]
    ).to_csv(reports / "trade_dataset_source_selection.csv", index=False)

    result = build_learning_label_contract(root=tmp_path)

    assert result["summary"]["status"] == "BLOCKED"
    assert result["summary"]["duplicate_trade_id_rows"] == 2
    assert result["summary"]["conflicting_trade_ids"] == 1
    assert "conflicting_trade_identity_context" in result["summary"][
        "lineage_blockers"
    ]


def test_model_authority_surfaces_rl_out_of_sample_failure(tmp_path):
    model_dir = tmp_path / "models" / "trade_gate"
    ml_dir = tmp_path / "reports" / "ml"
    rl_dir = tmp_path / "reports" / "rl"
    model_dir.mkdir(parents=True)
    ml_dir.mkdir(parents=True)
    rl_dir.mkdir(parents=True)
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "accepted": False,
                "best_model": "gradient_boosting",
                "median_take_rate": 0.06,
                "score_buckets_monotonic": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame([{"accepted": False}]).to_csv(
        ml_dir / "model_gated_acceptance.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "accepted": False,
                "out_of_sample_evidence": True,
                "validation_passed": False,
                "held_out_test_passed": False,
                "validation_gate_failures": "minimum_trades",
                "held_out_test_gate_failures": "minimum_take_rate",
            }
        ]
    ).to_csv(rl_dir / "rl_acceptance_report.csv", index=False)
    pd.DataFrame([{"status": "ready", "global_label_purge": True}]).to_csv(
        rl_dir / "rl_split_audit.csv", index=False
    )

    result = build_model_authority_status(root=tmp_path)
    summary = result["summary"]

    assert summary["rl_global_label_purge_ready"] is True
    assert summary["rl_out_of_sample_accepted"] is False
    assert "rl_out_of_sample_acceptance_not_met" in summary["blockers"]
    assert summary["model_authority"] == "RESEARCH_ONLY"
