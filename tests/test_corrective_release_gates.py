from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pandas as pd
import pytest

import quant_platform.orchestration.corrective_release_gates as release_gates
import quant_platform.orchestration.hyperliquid_learning_and_risk as learning_risk
from quant_platform.hyperliquid_testnet import (
    HyperliquidPairExecutionResult,
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
)
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
    _payload_hash,
    _testnet_sample_policy_id,
    _validated_testnet_candidate_receipt,
    archive_validated_testnet_lifecycle,
    build_corrective_release_gates,
    build_stage6_release_receipt,
    build_testnet_candidate_queue,
    build_testnet_candidate_receipt,
    build_testnet_sample_sufficiency,
    validate_stage6_release_evidence,
)
from quant_platform.orchestration.corrective_testnet_pair_execution import (
    ENABLE_ENV,
    ENTRY_ACKNOWLEDGEMENT,
    build_testnet_pair_execution_preflight,
    run_testnet_pair_execution,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    validate_current_wizard_hyperliquid_testnet_protocol,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_CANDIDATE_BINDING_FIELDS,
    _testnet_receipt_payload_hash,
    sign_testnet_smoke_approval,
    write_testnet_smoke_approval_template,
)
from tests.candidate_queue_support import seal_candidate_with_valid_queue
from tests.pair_cost_bundle_support import publish_valid_pair_cost_bundle

APPROVAL_SECRET = "corrective-release-gate-test-secret"
REAL_DAILY_CADENCE_BUILDER = release_gates.build_daily_cadence_acceptance


def _write_policy(root, **overrides):
    (root / "config").mkdir(exist_ok=True)
    policy = {
        "schema_version": "thewiz.testnet_sample_sufficiency.v5",
        "policy_version": "test-policy-v1",
        "effective_at_utc": "2026-07-01T00:00:00Z",
        "minimum_closed_paired_lifecycles": 30,
        "minimum_observation_days": 14,
        "minimum_independent_pairs": 3,
        "minimum_observed_regimes": 3,
        "minimum_lifecycles_per_candidate": 5,
        "minimum_lifecycles_per_pair": 5,
        "minimum_lifecycles_per_regime": 5,
        "maximum_unresolved_orphan_legs": 0,
        "maximum_unreconciled_orders": 0,
        "maximum_execution_failure_rate": 0.02,
        "minimum_positive_lifecycle_share": 0.55,
        "minimum_after_cost_mean_lcb_95_usd": 0.0,
        "after_cost_mean_lcb_95_method": "student_t_two_sided_95_lower_bound",
        "maximum_pair_sample_share": 0.5,
        "maximum_regime_sample_share": 0.5,
        "requires_positive_after_cost_expectancy": True,
        "requires_no_risk_policy_override": True,
        "requires_prospective_feature_and_outcome_logging": True,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **overrides,
    }
    (root / "config" / "testnet_sample_sufficiency_policy.json").write_text(
        json.dumps(policy),
        encoding="utf-8",
    )


def _publish_test_pair_cost_bundle(root, *, now):
    matrix = pd.read_csv(
        root / "reports" / "active" / "current_wizard_hyperliquid_experiment_matrix.csv",
        keep_default_na=False,
    )
    specs = [
        {
            "experiment_id": row["experiment_id"],
            "pair_group_key": row["pair_group_key"],
            "pair": row["pair"],
            "asset_x": row["asset_a"],
            "asset_y": row["asset_b"],
            "semantic_hypothesis_id": f"hypothesis-{row['experiment_id']}",
        }
        for row in matrix.to_dict("records")
    ]
    return publish_valid_pair_cost_bundle(root=root, now=now, specs=specs)["models"]


def _write_candidate_queue_upstream(root, monkeypatch, *, now, pair_samples, candidate_count=3):
    _write_policy(root)
    active = root / "reports" / "active"
    processed = root / "data" / "processed"
    learning_dir = root / "data" / "learning"
    active.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    learning_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("experiment-btc-eth", "BTC-USD-ETH-USD", "BTC", "ETH"),
        ("experiment-sol-hype", "SOL-USD-HYPE-USD", "SOL", "HYPE"),
        ("experiment-doge-xrp", "DOGE-USD-XRP-USD", "DOGE", "XRP"),
    ][:candidate_count]
    matrix_rows = []
    cost_rows = []
    market_rows = []
    for experiment_id, pair, asset_x, asset_y in specs:
        pair_group_key = f"hyperliquid|daily|{asset_x}|{asset_y}"
        matrix_rows.append(
            {
                "experiment_id": experiment_id,
                "pair_group_key": pair_group_key,
                "pair": pair,
                "asset_a": asset_x,
                "asset_b": asset_y,
                "timeframe": "daily",
                "exact_mode": "OU Optimal",
                "orientation": "original",
            }
        )
        cost_rows.append(
            {
                "pair_group_key": pair_group_key,
                "cost_model_id": f"cost-{asset_x.lower()}-{asset_y.lower()}",
                "strict_observed_cost_ready": True,
                "model_as_of_utc": now.isoformat(),
            }
        )
        market_rows.extend(
            {
                "asset": asset,
                "tradable_perp": True,
                "source_timestamp": now.isoformat(),
            }
            for asset in (asset_x, asset_y)
        )
    pd.DataFrame(matrix_rows).to_csv(
        active / "current_wizard_hyperliquid_experiment_matrix.csv", index=False
    )
    matrix_path = active / "current_wizard_hyperliquid_experiment_matrix.csv"
    pd.DataFrame(cost_rows).to_csv(processed / "hyperliquid_pair_cost_models.csv", index=False)
    _publish_test_pair_cost_bundle(root, now=now)
    pd.DataFrame(market_rows).drop_duplicates("asset").to_csv(
        processed / "hyperliquid_market_manifest.csv", index=False
    )
    cadence_path = active / "daily_cadence_acceptance.csv"
    pd.DataFrame(
        [
            {
                "run_date": now.date().isoformat(),
                "receipt_valid": True,
                "semantic_contract_status": "PASS",
                "qualifying_cycle": True,
                "consecutive_complete_cycles": 7,
                "required_cycles": 7,
                "cadence_acceptance_status": "PASS",
                "zero_execution_authority": True,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(cadence_path, index=False)
    monkeypatch.setattr(
        release_gates,
        "build_daily_cadence_acceptance",
        lambda **_: cadence_path,
    )
    survivor = {
        "receipt_status": "PASS",
        "testnet_candidate_authority": True,
        "thresholds_changed_after_results": False,
        "final_one_x_survivors": len(specs),
        "final_experiment_ids": [row[0] for row in specs],
        "acceptance_policy_id": "acceptance-policy-1",
    }
    survivor_path = active / "final_1x_survivor_receipt.json"
    survivor_path.write_text(json.dumps(survivor), encoding="utf-8")
    contract_id = "registered-stage4-contract-accepted-1"
    contract_path = (
        root / "data" / "research" / "registered_rerun_contracts" / f"{contract_id}.json"
    )
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "contract_id": contract_id,
        "immutable_contract_path": str(contract_path.relative_to(root)),
        "source_family_sha256": sha256(matrix_path.read_bytes()).hexdigest(),
        "source_family_rows": len(matrix_rows),
        "registered_candidates": [
            {
                "semantic_hypothesis_id": f"hypothesis-{index}",
                "source_experiment_id": experiment_id,
                "pair_group_key": f"hyperliquid|daily|{asset_x}|{asset_y}",
                "pair": pair,
                "exact_mode": "OU Optimal",
                "orientation": "original",
            }
            for index, (experiment_id, pair, asset_x, asset_y) in enumerate(specs, start=1)
        ],
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    release_gates.resolve_registered_source_family(
        root=root,
        contract=contract,
        current_matrix_path=matrix_path,
    )
    conclusion = {
        "contract_id": contract_id,
        "conclusion_status": "ACCEPTED_REGISTERED_SURVIVORS",
        "accepted_registered_hypotheses": len(specs),
        "outcomes": {
            f"hypothesis-{index}": "ACCEPTED_SURVIVOR" for index in range(1, len(specs) + 1)
        },
        "final_survivor_receipt_sha256": sha256(survivor_path.read_bytes()).hexdigest(),
    }
    conclusion_path = learning_dir / "stage4-conclusion.json"
    conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")
    learning_path = learning_dir / "registered-learning.json"
    learning_path.write_text("{}", encoding="utf-8")
    learning = {
        "status": "PASS_RESEARCH_LEARNING_GATES",
        "stage5_research_gate_pass": True,
        "training_accepted_oos": True,
        "model_accepted_oos": True,
        "rl_accepted_oos": True,
        "agent_governance_pass": True,
        "training_dataset_id": "dataset-accepted-1",
        "learning_id": "learning-accepted-1",
        "registered_execution_id": "execution-accepted-1",
        "registered_stage5_protocol_id": "stage5-protocol-accepted-1",
        "registered_stage5_protocol_sha256": "d" * 64,
    }
    model = {
        "out_of_sample_incremental_edge_accepted": True,
        "rl_out_of_sample_accepted": True,
        "exact_mode_trade_provenance_ready": True,
        "strict_cost_training_evidence_ready": True,
        "training_dataset_lineage_ready": True,
        "model_active_dataset_lineage_matches": True,
        "model_artifact_hash_matches": True,
        "model_artifact_sha256": "a" * 64,
        "median_take_rate": 0.5,
        "minimum_take_rate": 0.1,
    }
    stage4 = {
        "contract_id": contract_id,
        "source_family_sha256": contract["source_family_sha256"],
        "conclusion_path": str(conclusion_path.relative_to(root)),
        "conclusion_sha256": sha256(conclusion_path.read_bytes()).hexdigest(),
        "accepted_registered_hypotheses": len(specs),
        "stage4_final_one_x_survivors": len(specs),
        "stage4_final_experiment_ids": [row[0] for row in specs],
    }
    lifecycle_rows = [
        {
            "pair": pair,
            "model_training_dataset_id": learning["training_dataset_id"],
            "model_artifact_sha256": model["model_artifact_sha256"],
            "registered_learning_id": learning["learning_id"],
            "registered_learning_receipt_path": str(learning_path.relative_to(root)),
            "registered_learning_receipt_sha256": sha256(learning_path.read_bytes()).hexdigest(),
            "registered_stage5_protocol_id": learning["registered_stage5_protocol_id"],
            "registered_stage5_protocol_sha256": learning["registered_stage5_protocol_sha256"],
            "registered_execution_id": learning["registered_execution_id"],
        }
        for pair, count in pair_samples.items()
        for _ in range(count)
    ]
    monkeypatch.setattr(
        release_gates,
        "_validated_registered_learning",
        lambda **_: (learning, model, stage4, learning_path, ""),
    )
    monkeypatch.setattr(
        release_gates,
        "_validated_lifecycle_index",
        lambda **_: pd.DataFrame(lifecycle_rows),
    )
    return specs


def test_candidate_queue_blocks_when_three_independent_ready_pairs_are_absent(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={},
        candidate_count=2,
    )

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)
    candidate = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]

    assert queue["selected"] == {}
    assert queue["ready_candidate_pairs"] == 2
    assert queue["required_candidate_pairs"] == 3
    assert "testnet_candidate_queue_independent_pair_coverage_below_policy" in queue["blockers"]
    assert candidate["candidate_status"] == "BLOCKED"
    assert candidate["testnet_order_authority"] is False


def test_candidate_queue_rejects_forged_mutable_cadence_pass(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={},
    )
    monkeypatch.setattr(
        release_gates,
        "build_daily_cadence_acceptance",
        REAL_DAILY_CADENCE_BUILDER,
    )

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)
    rebuilt = pd.read_csv(tmp_path / "reports" / "active" / "daily_cadence_acceptance.csv")

    assert rebuilt.empty
    assert queue["selected"] == {}
    assert queue["cadence_pass"] is False
    assert queue["cadence_evidence_validated"] is True
    assert "seven_day_research_cadence_not_proven" in queue["blockers"]
    pointer = json.loads(queue["queue_receipt"].read_text(encoding="utf-8"))
    assert pointer["cadence_evidence_validated"] is True
    assert pointer["cadence_acceptance_status"] == "BLOCKED"
    assert pointer["cadence_validation_blocker"] == ""


def test_candidate_queue_fails_closed_when_cadence_rebuild_fails(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={},
    )

    def fail_rebuild(**_):
        raise OSError("injected cadence rebuild failure")

    monkeypatch.setattr(
        release_gates,
        "build_daily_cadence_acceptance",
        fail_rebuild,
    )

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)

    assert queue["selected"] == {}
    assert queue["cadence_pass"] is False
    assert queue["cadence_evidence_validated"] is False
    assert "seven_day_research_cadence_not_proven" in queue["blockers"]
    assert "daily_cadence_acceptance_rebuild_failed:OSError" in queue["blockers"]
    pointer = json.loads(queue["queue_receipt"].read_text(encoding="utf-8"))
    assert pointer["cadence_evidence_validated"] is False
    assert pointer["cadence_acceptance_status"] == "BLOCKED"
    assert (
        pointer["cadence_validation_blocker"] == "daily_cadence_acceptance_rebuild_failed:OSError"
    )


def test_zero_survivor_queue_reports_missing_prerequisites_without_false_lineage_failure(
    tmp_path,
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_policy(tmp_path)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "final_1x_survivor_receipt.json").write_text(
        json.dumps(
            {
                "receipt_status": "ZERO_SURVIVORS",
                "final_one_x_survivors": 0,
                "final_experiment_ids": [],
                "testnet_candidate_authority": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "registered_learning_research_status.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED_STAGE4",
                "stage5_research_gate_pass": False,
                "testnet_order_authority": False,
            }
        ),
        encoding="utf-8",
    )

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)
    receipt = json.loads(queue["queue_receipt"].read_text(encoding="utf-8"))

    assert queue["selected"] == {}
    assert "valid_multi_pair_final_one_x_survivor_receipt_missing" in queue["blockers"]
    assert "accepted_ml_rl_oos_authority_missing" in queue["blockers"]
    assert not any("lineage" in blocker for blocker in queue["blockers"])
    assert not any(
        blocker.startswith("registered_learning_validation_failed:")
        for blocker in queue["blockers"]
    )
    assert receipt["immutable_stage4_candidate_identity_ready"] is False
    assert receipt["candidate_selection_ready"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False


def test_candidate_queue_selects_the_current_model_pair_with_largest_sample_deficit(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)
    candidate = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    valid, blockers = _validated_testnet_candidate_receipt(root=tmp_path, candidate=candidate)
    rows = pd.read_csv(queue["queue"])

    assert queue["selected"]["candidate_experiment_id"] == "experiment-sol-hype"
    assert queue["selected"]["validated_current_model_pair_samples"] == 1
    assert queue["ready_candidate_pairs"] == 3
    assert candidate["candidate_queue_selection_rank"] == 1
    assert candidate["candidate_status"] == "READY_FOR_NO_ORDER_PREFLIGHT"
    assert valid is True
    assert blockers == []
    assert rows["selected_active"].sum() == 1
    assert not rows["testnet_order_authority"].map(bool).any()


def test_ready_candidate_receipt_is_stable_across_identical_read_only_refreshes(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )

    first = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    second = build_testnet_candidate_receipt(root=tmp_path, now=now + timedelta(minutes=1))[
        "receipt"
    ]

    assert first["candidate_status"] == "READY_FOR_NO_ORDER_PREFLIGHT"
    assert second["candidate_receipt_id"] == first["candidate_receipt_id"]
    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert second["generated_at_utc"] == first["generated_at_utc"]
    valid, blockers = _validated_testnet_candidate_receipt(root=tmp_path, candidate=second)
    assert valid is True
    assert blockers == []


def test_ready_candidate_receipt_rotates_when_bound_source_hash_changes(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )
    first = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    cost_path = tmp_path / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    cost_path.write_text(cost_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    _publish_test_pair_cost_bundle(root=tmp_path, now=now + timedelta(minutes=1))

    second = build_testnet_candidate_receipt(root=tmp_path, now=now + timedelta(minutes=1))[
        "receipt"
    ]

    assert second["candidate_status"] == "READY_FOR_NO_ORDER_PREFLIGHT"
    assert second["candidate_receipt_id"] != first["candidate_receipt_id"]
    assert second["receipt_sha256"] != first["receipt_sha256"]


def test_sealed_candidate_survives_mutable_latest_cost_refresh(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )
    candidate = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    cost_path = tmp_path / "data" / "processed" / "hyperliquid_pair_cost_models.csv"
    cost_path.write_text(
        cost_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    valid, blockers = _validated_testnet_candidate_receipt(
        root=tmp_path,
        candidate=candidate,
    )

    assert valid is True
    assert blockers == []


def test_sealed_candidate_rejects_immutable_cost_snapshot_tampering(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )
    candidate = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    snapshot_relative = next(
        path
        for path in candidate["source_artifact_hashes"]
        if "data/research/l2_cost_model_receipts/" in path
        and path.endswith("/pair_cost_models.csv")
    )
    snapshot = tmp_path / snapshot_relative
    snapshot.write_text(
        snapshot.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    valid, blockers = _validated_testnet_candidate_receipt(
        root=tmp_path,
        candidate=candidate,
    )

    assert valid is False
    assert f"testnet_candidate_queue_source_artifact_changed:{snapshot_relative}" in blockers


def test_ready_candidate_receipt_is_not_reused_after_cost_freshness_expires(tmp_path, monkeypatch):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )
    first = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]

    expired = build_testnet_candidate_receipt(root=tmp_path, now=now + timedelta(hours=3))[
        "receipt"
    ]

    assert first["candidate_status"] == "READY_FOR_NO_ORDER_PREFLIGHT"
    assert expired["candidate_status"] == "BLOCKED"
    assert expired["candidate_receipt_id"] != first["candidate_receipt_id"]
    assert "testnet_candidate_queue_has_no_selected_candidate" in expired["blockers"]


def test_candidate_queue_uses_frozen_stage4_family_after_active_matrix_rotates(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={
            "BTC-USD-ETH-USD": 5,
            "SOL-USD-HYPE-USD": 1,
            "DOGE-USD-XRP-USD": 3,
        },
    )
    active_matrix = (
        tmp_path / "reports" / "active" / "current_wizard_hyperliquid_experiment_matrix.csv"
    )
    pd.DataFrame(
        [
            {
                "experiment_id": "different-daily-refresh",
                "pair_group_key": "hyperliquid|daily|ARB|OP",
                "pair": "ARB-USD-OP-USD",
                "asset_a": "ARB",
                "asset_b": "OP",
                "timeframe": "daily",
                "exact_mode": "Copula",
                "orientation": "reverse",
            }
        ]
    ).to_csv(active_matrix, index=False)

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)

    assert queue["selected"]["candidate_experiment_id"] == "experiment-sol-hype"
    assert queue["selected"]["pair"] == "SOL-USD-HYPE-USD"
    assert queue["selected"]["immutable_stage4_candidate_identity_ready"] is True
    assert queue["blockers"] == []


def test_candidate_queue_rejects_contract_identity_that_differs_from_frozen_family(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={},
    )
    contract_path = (
        tmp_path
        / "data"
        / "research"
        / "registered_rerun_contracts"
        / "registered-stage4-contract-accepted-1.json"
    )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["registered_candidates"][0]["pair"] = "FORGED-PAIR"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    queue = build_testnet_candidate_queue(root=tmp_path, now=now)

    assert queue["selected"] == {}
    assert any(
        blocker.startswith("stage4_survivor_identity_lineage_invalid:")
        for blocker in queue["blockers"]
    )
    rows = pd.read_csv(queue["queue"])
    assert not rows["immutable_stage4_candidate_identity_ready"].map(bool).any()
    assert not rows["testnet_order_authority"].map(bool).any()


def test_tampered_immutable_candidate_queue_invalidates_the_selected_candidate(
    tmp_path, monkeypatch
):
    now = datetime(2026, 8, 10, 12, tzinfo=UTC)
    _write_candidate_queue_upstream(
        tmp_path,
        monkeypatch,
        now=now,
        pair_samples={},
    )
    candidate = build_testnet_candidate_receipt(root=tmp_path, now=now)["receipt"]
    queue_path = tmp_path / candidate["candidate_queue_path"]
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["rows"][0]["pair"] = "FORGED-PAIR"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")

    valid, blockers = _validated_testnet_candidate_receipt(root=tmp_path, candidate=candidate)

    assert valid is False
    assert "testnet_candidate_queue_artifact_invalid" in blockers
    assert "testnet_candidate_queue_receipt_invalid" in blockers


def test_no_order_preflight_accepts_fresh_ready_margin_contract(tmp_path, monkeypatch):
    now = datetime(2026, 8, 11, 12, tzinfo=UTC)
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    candidate = {
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_receipt_id": "testnetcandidate_ready",
        "candidate_experiment_id": "experiment-ready",
        "pair_group_key": "hyperliquid|daily|BTC|ETH",
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
        "timeframe": "daily",
        "exact_mode": "OU Optimal",
        "orientation": "original",
        "cost_model_id": "cost-ready",
        "model_training_dataset_id": "dataset-ready",
        "model_artifact_sha256": "a" * 64,
        "registered_learning_id": "learning-ready",
        "registered_learning_receipt_path": "data/research/learning-ready.json",
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_stage5_protocol_id": "stage5-protocol-ready",
        "registered_stage5_protocol_sha256": "c" * 64,
        "registered_execution_id": "stage4-execution-ready",
        "hyperliquid_markets_current": True,
        "strict_cost_model_current": True,
    }
    monkeypatch.setattr(
        release_gates,
        "build_testnet_candidate_receipt",
        lambda **_: {"receipt": candidate},
    )
    pd.DataFrame(
        [
            {
                "checked_at_utc": now.isoformat(),
                "network": "testnet",
                "ready_for_no_order_preflight": True,
                "submit_orders_enabled": False,
                "agent_key_matches_address": True,
                "agent_authorized_for_master": True,
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_preflight.csv", index=False)
    pd.DataFrame(
        [
            {
                "checked_at_utc": now.isoformat(),
                "network": "testnet",
                "withdrawable_usd": 100.0,
                "point_in_time_status": "confirmed",
                "status": "READY",
                "blockers": "",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_margin_snapshot.csv", index=False)
    approval = {
        **{field: candidate[field] for field in TESTNET_CANDIDATE_BINDING_FIELDS},
        "network": "testnet",
        "one_run_only": True,
        "max_total_notional_usd": 25.0,
        "legs": [
            {"market": "BTC", "side": "BUY", "size": 0.001, "limit_price": 10_000.0},
            {"market": "ETH", "side": "SELL", "size": 0.005, "limit_price": 2_000.0},
        ],
    }
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(approval), encoding="utf-8"
    )
    scenarios = [
        {
            "scenario_id": scenario,
            "scenario_pass": True,
            "simulation_only": True,
            "order_submission_performed": False,
        }
        for scenario in (
            "partial_orphan_recovery",
            "unconfirmed_response_reconciliation",
            "restart_open_pair_reconciliation",
        )
    ]
    (active / "current_wizard_hyperliquid_testnet_protocol_manifest.json").write_text(
        json.dumps(
            {
                "protocol_status": "PASS",
                "simulation_is_testnet_proof": False,
                "order_submission_performed": False,
                "scenario_results": scenarios,
            }
        ),
        encoding="utf-8",
    )

    result = release_gates.build_no_order_preflight(root=tmp_path, now=now)
    checks = result["frame"].set_index("check")

    assert result["status"] == "PASS"
    assert checks.loc["margin_collateral", "status"] == "PASS"
    assert checks.loc["testnet_protocol_current_binding", "status"] == "PASS"
    assert checks["testnet_order_authority"].eq(False).all()
    assert checks["live_trading_authorized"].eq(False).all()
    refreshed_protocol = json.loads(
        (active / "current_wizard_hyperliquid_testnet_protocol_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert refreshed_protocol["protocol_id"].startswith("cwtestnetprotocol_")
    assert len(refreshed_protocol["source_hashes"]) >= 4


def test_no_order_preflight_blocks_protocol_revalidation_failure(
    tmp_path,
    monkeypatch,
):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    monkeypatch.setattr(
        release_gates,
        "build_testnet_candidate_receipt",
        lambda **_: {
            "receipt": {
                "candidate_status": "BLOCKED",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        },
    )

    def fail_protocol_revalidation(**_):
        raise ValueError("injected protocol drift")

    monkeypatch.setattr(
        release_gates,
        "validate_current_wizard_hyperliquid_testnet_protocol",
        fail_protocol_revalidation,
    )

    result = release_gates.build_no_order_preflight(
        root=tmp_path,
        now=datetime(2026, 8, 11, 12, tzinfo=UTC),
    )
    checks = result["frame"].set_index("check")

    assert result["status"] == "BLOCKED"
    assert checks.loc["testnet_protocol_current_binding", "status"] == "BLOCKED"
    assert checks.loc["testnet_protocol_current_binding", "blocker"].startswith(
        "testnet_protocol_revalidation_failed:ValueError:"
    )
    assert checks.loc["partial_fill_contingency", "status"] == "BLOCKED"
    assert checks["testnet_order_authority"].eq(False).all()
    assert checks["live_trading_authorized"].eq(False).all()


def _write_valid_lifecycle_source(
    root,
    *,
    sample_number,
    started,
    assets=("BTC", "ETH"),
    regime="range",
    net_after_cost=0.05,
    strategy_signal_id=None,
    model_training_dataset_id="dataset-accepted-1",
    model_artifact_sha256="a" * 64,
    registered_learning_id="registered-learning-accepted-1",
    registered_stage5_protocol_id="stage5-protocol-accepted-1",
    registered_stage5_protocol_sha256="d" * 64,
    registered_execution_id="registered-execution-accepted-1",
    approval_time=None,
):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    approval_issued = approval_time or datetime.now(UTC)
    pd.DataFrame([{"ready_for_no_order_preflight": True}]).to_csv(
        active / "hyperliquid_testnet_preflight.csv", index=False
    )
    asset_x, asset_y = assets
    candidate_experiment_id = f"experiment-{asset_x.lower()}-{asset_y.lower()}"
    binding = {
        "candidate_experiment_id": candidate_experiment_id,
        "pair_group_key": f"hyperliquid|daily|{asset_x}|{asset_y}",
        "pair": f"{asset_x}-USD-{asset_y}-USD",
        "timeframe": "daily",
        "exact_mode": "OU Optimal",
        "orientation": "original",
        "cost_model_id": f"cost-model-{sample_number}",
        "model_training_dataset_id": model_training_dataset_id,
        "model_artifact_sha256": model_artifact_sha256,
    }
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at_utc": (approval_issued - timedelta(minutes=2)).isoformat(),
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_leverage": 1.0,
        "asset_x": asset_x,
        "asset_y": asset_y,
        "registered_learning_id": registered_learning_id,
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_stage5_protocol_id": registered_stage5_protocol_id,
        "registered_stage5_protocol_sha256": registered_stage5_protocol_sha256,
        "registered_execution_id": registered_execution_id,
        "survivor_receipt_id": "survivor-accepted-1",
        "source_artifact_hashes": {"stage5": "c" * 64},
        "blockers": [],
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **binding,
    }
    candidate_cohort = (
        ("BTC", "ETH"),
        ("SOL", "HYPE"),
        ("DOGE", "XRP"),
    )
    support_candidates = tuple(
        {
            "experiment_id": f"experiment-{support_x.lower()}-{support_y.lower()}",
            "pair": f"{support_x}-USD-{support_y}-USD",
            "asset_x": support_x,
            "asset_y": support_y,
        }
        for support_x, support_y in candidate_cohort
        if (support_x, support_y) != (asset_x, asset_y)
    )
    candidate = seal_candidate_with_valid_queue(
        root=root,
        receipt=candidate,
        support_candidates=support_candidates,
    )
    binding = {field: candidate[field] for field in TESTNET_CANDIDATE_BINDING_FIELDS}
    (active / "testnet_candidate_receipt.json").write_text(json.dumps(candidate), encoding="utf-8")
    run_id = f"run-{sample_number}"
    candidate_set_id = f"set-{sample_number}"
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": run_id, "candidate_set_id": candidate_set_id}),
        encoding="utf-8",
    )
    protocol = validate_current_wizard_hyperliquid_testnet_protocol(root=root)
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-agent",
    )
    approval_path = active / "hyperliquid_testnet_smoke_approval.json"
    approval_path.unlink(missing_ok=True)
    template = write_testnet_smoke_approval_template(root=root, config=config)
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    feature_time = approval_issued - timedelta(minutes=1)
    entry_context = {
        "feature_timestamp_utc": feature_time.isoformat(),
        "regime": regime,
        "trade_quality_score": 0.75,
        "strategy_signal_id": strategy_signal_id or f"signal-{sample_number}",
    }
    prices = (100.0, 50.0)
    approval.update(
        {
            "approved": True,
            "approval_id": f"approval-{sample_number}",
            "issued_at_utc": approval_issued.isoformat(),
            "expires_at_utc": (approval_issued + timedelta(minutes=10)).isoformat(),
            "entry_context": entry_context,
            "legs": [
                {
                    "market": asset_x,
                    "side": "BUY",
                    "size": 10.0 / prices[0],
                    "limit_price": prices[0],
                },
                {
                    "market": asset_y,
                    "side": "SELL",
                    "size": 10.0 / prices[1],
                    "limit_price": prices[1],
                },
            ],
        }
    )
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(root=root, approval_secret=APPROVAL_SECRET, config=config)
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    fee_per_fill = 0.005
    total_fees = fee_per_fill * 4
    gross_pnl = net_after_cost + total_fees

    def fill(coin, order_id, trade_id, closed_pnl, side, price, size):
        return {
            "order_id": order_id,
            "trade_id": trade_id,
            "transaction_hash": "0x" + f"{sample_number * 100 + trade_id:064x}",
            "coin": coin,
            "intent_side": "BUY" if side == "B" else "SELL",
            "exchange_side": side,
            "size": size,
            "price": price,
            "expected_limit_price": price,
            "fee_usd": fee_per_fill,
            "fee_token": "USDC",
            "closed_pnl_usd": closed_pnl,
            "implementation_shortfall_vs_limit_usd": 0.0,
            "fill_time_ms": int(started.timestamp() * 1000) + trade_id,
            "economics_complete": True,
        }

    entry_refs = [f"entry-{sample_number}-x", f"entry-{sample_number}-y"]
    exit_refs = [f"exit-{sample_number}-x", f"exit-{sample_number}-y"]
    events = [
        {
            "event_id": f"event-{sample_number}-1",
            "event_type": "two_leg_entry",
            "timestamp_utc": (started + timedelta(seconds=1)).isoformat(),
            "exchange_reference_ids": entry_refs,
            "leg_x_status": "filled",
            "leg_y_status": "filled",
            "fill_economics_complete": True,
            "fill_evidence": [
                fill(
                    asset_x,
                    entry_refs[0],
                    sample_number * 10 + 1,
                    0.0,
                    "B",
                    prices[0],
                    10.0 / prices[0],
                ),
                fill(
                    asset_y,
                    entry_refs[1],
                    sample_number * 10 + 2,
                    0.0,
                    "A",
                    prices[1],
                    10.0 / prices[1],
                ),
            ],
        },
        {
            "event_id": f"event-{sample_number}-2",
            "event_type": "two_leg_exit",
            "timestamp_utc": (started + timedelta(seconds=2)).isoformat(),
            "exchange_reference_ids": exit_refs,
            "leg_x_status": "filled",
            "leg_y_status": "filled",
            "fill_economics_complete": True,
            "fill_evidence": [
                fill(
                    asset_x,
                    exit_refs[0],
                    sample_number * 10 + 3,
                    gross_pnl / 2,
                    "A",
                    prices[0],
                    10.0 / prices[0],
                ),
                fill(
                    asset_y,
                    exit_refs[1],
                    sample_number * 10 + 4,
                    gross_pnl / 2,
                    "B",
                    prices[1],
                    10.0 / prices[1],
                ),
            ],
        },
        {
            "event_id": f"event-{sample_number}-3",
            "event_type": "reconciled",
            "timestamp_utc": (started + timedelta(seconds=3)).isoformat(),
            "exchange_reference_ids": [f"reconcile-{sample_number}"],
        },
        {
            "event_id": f"event-{sample_number}-4",
            "event_type": "idempotency",
            "timestamp_utc": (started + timedelta(seconds=4)).isoformat(),
            "exchange_reference_ids": [f"idempotency-{sample_number}"],
            "duplicate_submit_blocked": True,
        },
    ]
    receipt = {
        "receipt_version": "hyperliquid-testnet-lifecycle-v3",
        "receipt_source": "hyperliquid_testnet_lifecycle_evidence_capture",
        "actual_testnet": True,
        "network": "testnet",
        "approval_id": approval["approval_id"],
        "run_id": run_id,
        "candidate_set_id": candidate_set_id,
        "protocol_id": protocol.summary["protocol_id"],
        **binding,
        "risk_override_applied": False,
        "entry_context": entry_context,
        "started_at_utc": started.isoformat(),
        "completed_at_utc": (started + timedelta(seconds=6)).isoformat(),
        "events": events,
        "funding_events": [],
        "funding_query_complete": True,
        "economics": {
            "economics_complete": True,
            "terminal_fill_count": 4,
            "gross_realized_pnl_usd": gross_pnl,
            "fees_usd": total_fees,
            "funding_pnl_usd": 0.0,
            "implementation_shortfall_vs_limit_usd": 0.0,
            "net_realized_pnl_after_cost_usd": net_after_cost,
            "slippage_treatment": "diagnostic_only_already_reflected_in_realized_pnl_not_double_subtracted",
        },
        "final_state": {
            "reconciled": True,
            "position_x": 0.0,
            "position_y": 0.0,
            "open_order_count": 0,
            "account_state_timestamp_utc": (started + timedelta(seconds=5)).isoformat(),
        },
    }
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return config, receipt


def _record_testnet_attempt(
    root,
    *,
    config,
    now,
    monkeypatch,
    status="pair_recovered_flat",
    reconciled=True,
):
    active = root / "reports" / "active"
    approval_path = active / "hyperliquid_testnet_smoke_approval.json"
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    approval["approval_id"] = "approval-terminal-failure"
    approval_path.write_text(json.dumps(approval), encoding="utf-8")
    markets = [str(leg["market"]) for leg in approval["legs"]]

    def approval_validator(approval_id, root, config, scope, checked_at):
        return {
            "status": "PASS",
            "blockers": [],
            "execution_allowed": True,
            "testnet_order_authority": True,
            "authority_scope": scope,
            "live_trading_authorized": False,
        }

    def info_client(payload):
        if payload["type"] == "meta":
            return {
                "universe": [
                    {"name": market, "szDecimals": 4, "maxLeverage": 50} for market in markets
                ]
            }
        if payload["type"] == "clearinghouseState":
            return {"assetPositions": []}
        if payload["type"] == "openOrders":
            return []
        raise AssertionError(payload)

    def failed_submit_pair(self, intents, resolved):
        del self, intents, resolved
        return HyperliquidPairExecutionResult(
            status=status,
            reason="test_attempt_result",
            order_submission_performed=True,
            reconciled=reconciled,
            live_trading_authorized=False,
        )

    monkeypatch.setattr(
        HyperliquidTestnetPairExecutor,
        "submit_pair",
        failed_submit_pair,
    )

    preflight = build_testnet_pair_execution_preflight(
        root=root,
        action="entry",
        approval_id=approval["approval_id"],
        config=config,
        now=now,
        approval_validator=approval_validator,
        candidate_validator=lambda root, candidate: (True, []),
        info_client=info_client,
    )
    monkeypatch.setenv(ENABLE_ENV, "true")
    result = run_testnet_pair_execution(
        root=root,
        action="entry",
        preflight_id=preflight.summary["preflight_id"],
        approval_id=approval["approval_id"],
        acknowledgement=ENTRY_ACKNOWLEDGEMENT,
        execute=True,
        config=HyperliquidTestnetConfig(
            master_address=config.master_address,
            agent_address=config.agent_address,
            keychain_service=config.keychain_service,
            submit_orders=True,
        ),
        now=now,
    )
    assert result.summary["status"] == status


def test_failed_submitted_attempt_is_in_execution_failure_denominator(tmp_path, monkeypatch):
    _write_policy(tmp_path)
    started = datetime(2026, 7, 8, 12, tzinfo=UTC)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=8,
        started=started,
        net_after_cost=0.10,
    )
    archived = archive_validated_testnet_lifecycle(
        root=tmp_path,
        now=started,
        approval_secret=APPROVAL_SECRET,
        config=config,
    )
    assert archived["status"] == "PASS", archived
    _record_testnet_attempt(
        tmp_path,
        config=config,
        now=started + timedelta(minutes=2),
        monkeypatch=monkeypatch,
    )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    checks = result["frame"].set_index("check")

    assert checks.loc["execution_attempt_evidence_valid", "status"] == "PASS"
    assert checks.loc["submitted_attempts_accounted", "status"] == "PASS"
    assert float(checks.loc["execution_failure_rate", "observed"]) == pytest.approx(0.5)
    assert checks.loc["execution_failure_rate", "status"] == "BLOCKED"
    assert result["current_model_terminal_execution_attempts"] == 2
    assert result["current_model_failed_execution_attempts"] == 1


def test_unresolved_submitted_attempt_blocks_sample_release(tmp_path, monkeypatch):
    _write_policy(tmp_path)
    started = datetime(2026, 7, 8, 12, tzinfo=UTC)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=9,
        started=started,
        net_after_cost=0.10,
    )
    archived = archive_validated_testnet_lifecycle(
        root=tmp_path,
        now=started,
        approval_secret=APPROVAL_SECRET,
        config=config,
    )
    assert archived["status"] == "PASS", archived
    _record_testnet_attempt(
        tmp_path,
        config=config,
        now=started + timedelta(minutes=2),
        monkeypatch=monkeypatch,
        status="pair_submitted",
        reconciled=False,
    )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    checks = result["frame"].set_index("check")

    assert checks.loc["execution_attempt_evidence_valid", "status"] == "PASS"
    assert checks.loc["submitted_attempts_accounted", "status"] == "BLOCKED"
    assert int(checks.loc["submitted_attempts_accounted", "observed"]) == 1
    assert result["current_model_pending_execution_attempts"] == 1


def test_zero_survivor_blocks_all_testnet_and_live_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "testnet_sample_sufficiency_policy.json").write_text(
        json.dumps(
            {
                "minimum_closed_paired_lifecycles": 30,
                "minimum_observation_days": 14,
                "minimum_independent_pairs": 3,
                "minimum_observed_regimes": 3,
                "maximum_unresolved_orphan_legs": 0,
            }
        )
    )
    (active / "final_1x_survivor_receipt.json").write_text(
        json.dumps({"receipt_status": "ZERO_SURVIVORS", "final_experiment_ids": []})
    )
    result = build_corrective_release_gates(root=tmp_path)
    assert result.summary["status"] == "BLOCKED"
    assert result.summary["orders_submitted"] == 0
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    authorization = json.loads(result.paths["live_canary_authorization"].read_text())
    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["live_trading_authorized"] is False


def test_validated_lifecycle_archive_is_idempotent(tmp_path):
    _write_policy(tmp_path)
    config, receipt = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=1,
        started=datetime(2026, 7, 1, 12, tzinfo=UTC),
    )

    first = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )
    second = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )
    index = pd.read_csv(first["index"])

    assert first["status"] == "PASS"
    assert first["already_archived"] is False
    assert second["status"] == "PASS"
    assert second["already_archived"] is True
    assert len(index) == 1
    assert index.iloc[0]["receipt_hash"] == receipt["receipt_hash"]


def test_archived_candidate_source_tamper_invalidates_lifecycle(tmp_path):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=60,
        started=datetime(2026, 7, 1, 12, tzinfo=UTC),
    )
    archived = archive_validated_testnet_lifecycle(
        root=tmp_path,
        approval_secret=APPROVAL_SECRET,
        config=config,
    )
    archive_path = archived["archive"]
    source_path = next(archive_path.glob("candidate_source_*"))
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "tampered\n",
        encoding="utf-8",
    )

    valid, blockers = release_gates._validate_lifecycle_archive(
        archive_path=archive_path,
        receipt_hash=str(archived["receipt_hash"]),
    )
    sufficiency = build_testnet_sample_sufficiency(root=tmp_path)

    assert valid is False
    assert any(
        blocker.startswith("testnet_lifecycle_archive_artifact_invalid:candidate_source_")
        for blocker in blockers
    )
    assert sufficiency["validated_lifecycle_samples"] == 0


def test_sample_policy_cannot_rotate_after_first_lifecycle(tmp_path):
    _write_policy(tmp_path)
    first_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=7,
        started=datetime(2026, 7, 7, 12, tzinfo=UTC),
    )
    first = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=first_config
    )
    assert first["status"] == "PASS"

    _write_policy(tmp_path, minimum_positive_lifecycle_share=0.01)
    second_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=8,
        started=datetime(2026, 7, 8, 12, tzinfo=UTC),
    )
    second = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=second_config
    )

    assert second["status"] == "BLOCKED"
    assert "testnet_sample_policy_changed_after_samples_started" in second["blockers"]
    assert len(pd.read_csv(first["index"])) == 1
    sufficiency = build_testnet_sample_sufficiency(root=tmp_path)
    frozen = (
        sufficiency["frame"].loc[sufficiency["frame"]["check"].eq("frozen_sample_policy")].iloc[0]
    )
    assert frozen["status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_closed_paired_lifecycles", 29),
        ("minimum_lifecycles_per_candidate", 4),
        ("minimum_lifecycles_per_pair", 4),
        ("minimum_lifecycles_per_regime", 4),
        ("maximum_execution_failure_rate", 0.021),
        ("minimum_positive_lifecycle_share", 0.54),
        ("minimum_after_cost_mean_lcb_95_usd", -0.01),
        ("after_cost_mean_lcb_95_method", "normal_approximation_1_96"),
        ("maximum_pair_sample_share", 0.51),
        ("maximum_regime_sample_share", 0.51),
        ("requires_prospective_feature_and_outcome_logging", False),
    ],
)
def test_pre_sample_policy_weakening_is_rejected(tmp_path, field, value):
    _write_policy(tmp_path, **{field: value})
    policy = json.loads(
        (tmp_path / "config" / "testnet_sample_sufficiency_policy.json").read_text(encoding="utf-8")
    )

    assert _testnet_sample_policy_id(policy) == ""


def test_invalid_or_missing_receipt_hash_is_not_indexed(tmp_path):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=2,
        started=datetime(2026, 7, 2, 12, tzinfo=UTC),
    )
    receipt_path = tmp_path / "reports" / "active" / "hyperliquid_testnet_smoke_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["receipt_hash"] = ""
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )

    assert result["status"] == "BLOCKED"
    assert not result["index"].exists()


def test_simulated_or_economically_tampered_receipt_is_not_indexed(tmp_path):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=3,
        started=datetime(2026, 7, 3, 12, tzinfo=UTC),
    )
    receipt_path = tmp_path / "reports" / "active" / "hyperliquid_testnet_smoke_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["actual_testnet"] = False
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    simulated = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )
    assert simulated["status"] == "BLOCKED"
    assert not simulated["index"].exists()

    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=4,
        started=datetime(2026, 7, 4, 12, tzinfo=UTC),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["economics"]["net_realized_pnl_after_cost_usd"] += 1.0
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    tampered = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )
    assert tampered["status"] == "BLOCKED"
    assert not tampered["index"].exists()


@pytest.mark.parametrize(
    "attack",
    ["same_coin", "wrong_side", "wrong_size", "extra_order_reference"],
)
def test_rehashed_fill_intent_mismatch_is_not_indexed(tmp_path, attack):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=500,
        started=datetime(2026, 7, 5, 12, tzinfo=UTC),
    )
    receipt_path = tmp_path / "reports" / "active" / "hyperliquid_testnet_smoke_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    entry = next(event for event in receipt["events"] if event["event_type"] == "two_leg_entry")
    if attack == "same_coin":
        entry["fill_evidence"][1]["coin"] = entry["fill_evidence"][0]["coin"]
    elif attack == "wrong_side":
        entry["fill_evidence"][0]["intent_side"] = "SELL"
    elif attack == "wrong_size":
        entry["fill_evidence"][0]["size"] /= 2
    else:
        entry["exchange_reference_ids"].append("unapproved-extra-order")
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    result = archive_validated_testnet_lifecycle(
        root=tmp_path,
        approval_secret=APPROVAL_SECRET,
        config=config,
    )

    assert result["status"] == "BLOCKED"
    assert "validated_testnet_lifecycle_gate_not_passed" in result["blockers"]
    assert not result["index"].exists()


def test_forged_candidate_binding_is_not_indexed(tmp_path):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=5,
        started=datetime(2026, 7, 5, 12, tzinfo=UTC),
    )
    candidate_path = tmp_path / "reports" / "active" / "testnet_candidate_receipt.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["candidate_experiment_id"] = "forged-experiment"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )

    assert result["status"] == "BLOCKED"
    assert not result["index"].exists()


def test_rehashed_candidate_cannot_escape_immutable_identity(tmp_path):
    _write_policy(tmp_path)
    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=6,
        started=datetime(2026, 7, 6, 12, tzinfo=UTC),
    )
    candidate_path = tmp_path / "reports" / "active" / "testnet_candidate_receipt.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["candidate_experiment_id"] = "forged-but-rehashed"
    candidate["receipt_sha256"] = _payload_hash(candidate)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")

    result = archive_validated_testnet_lifecycle(
        root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
    )

    assert result["status"] == "BLOCKED"
    assert "testnet_candidate_identity_content_invalid" in result["blockers"]
    assert not result["index"].exists()


def test_resealed_candidate_cannot_substitute_registered_stage5_lineage(tmp_path):
    base = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_experiment_id": "experiment-stage5-binding",
        "pair_group_key": "hyperliquid|daily|BTC|ETH",
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
        "timeframe": "daily",
        "exact_mode": "OU Optimal",
        "orientation": "original",
        "cost_model_id": "cost-stage5-binding",
        "candidate_leverage": 1.0,
        "model_training_dataset_id": "dataset-stage5-binding",
        "model_artifact_sha256": "a" * 64,
        "registered_learning_id": "learning-stage5-binding",
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_execution_id": "execution-stage5-binding",
        "survivor_receipt_id": "survivor-stage5-binding",
        "source_artifact_hashes": {"data/research/fixture_stage5_receipt.json": "fixture"},
        "blockers": [],
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    candidate = seal_candidate_with_valid_queue(root=tmp_path, receipt=base)
    forged = dict(candidate)
    for field in (
        "candidate_receipt_id",
        "receipt_id",
        "candidate_identity_path",
        "candidate_identity_sha256",
        "receipt_sha256",
    ):
        forged.pop(field, None)
    forged["registered_learning_id"] = "learning-substituted"
    forged["registered_learning_receipt_sha256"] = "f" * 64
    forged = release_gates._seal_testnet_candidate_receipt(
        receipt=forged,
        root=tmp_path,
    )

    valid, blockers = _validated_testnet_candidate_receipt(
        root=tmp_path,
        candidate=forged,
    )

    assert valid is False
    assert "testnet_candidate_queue_lineage_mismatch" in blockers


def test_duplicate_exchange_fills_do_not_count_as_independent_samples(tmp_path):
    _write_policy(tmp_path)
    first_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=10,
        started=datetime(2026, 7, 10, 12, tzinfo=UTC),
    )
    assert (
        archive_validated_testnet_lifecycle(
            root=tmp_path, approval_secret=APPROVAL_SECRET, config=first_config
        )["status"]
        == "PASS"
    )
    second_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=11,
        started=datetime(2026, 7, 11, 12, tzinfo=UTC),
    )
    receipt_path = tmp_path / "reports" / "active" / "hyperliquid_testnet_smoke_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["events"][0]["fill_evidence"][0]["trade_id"] = 101
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert (
        archive_validated_testnet_lifecycle(
            root=tmp_path, approval_secret=APPROVAL_SECRET, config=second_config
        )["status"]
        == "PASS"
    )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    duplicate = (
        result["frame"].loc[result["frame"]["check"].eq("duplicate_exchange_evidence")].iloc[0]
    )

    assert duplicate["status"] == "BLOCKED"
    assert int(duplicate["observed"]) == 2


def test_reused_strategy_signal_does_not_count_as_independent_sample(tmp_path):
    _write_policy(tmp_path)
    for sample_number in (20, 21):
        config, _ = _write_valid_lifecycle_source(
            tmp_path,
            sample_number=sample_number,
            started=datetime(2026, 7, sample_number, 12, tzinfo=UTC),
            strategy_signal_id="same-prospective-signal",
        )
        assert (
            archive_validated_testnet_lifecycle(
                root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
            )["status"]
            == "PASS"
        )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    duplicate = (
        result["frame"].loc[result["frame"]["check"].eq("duplicate_strategy_signals")].iloc[0]
    )

    assert duplicate["status"] == "BLOCKED"
    assert int(duplicate["observed"]) == 2


def test_positive_mean_outlier_cannot_mask_mostly_losing_samples(tmp_path):
    _write_policy(tmp_path)
    for sample_number, net_after_cost in zip((30, 31, 32), (-0.01, -0.01, 0.08), strict=True):
        config, _ = _write_valid_lifecycle_source(
            tmp_path,
            sample_number=sample_number,
            started=datetime(2026, 7, sample_number - 20, 12, tzinfo=UTC),
            net_after_cost=net_after_cost,
        )
        assert (
            archive_validated_testnet_lifecycle(
                root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
            )["status"]
            == "PASS"
        )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    checks = result["frame"].set_index("check")

    assert checks.loc["positive_after_cost_expectancy", "status"] == "PASS"
    assert checks.loc["positive_lifecycle_share", "status"] == "BLOCKED"
    assert checks.loc["after_cost_mean_lcb_95_usd", "status"] == "BLOCKED"


def test_after_cost_mean_lcb_uses_student_t_small_sample_bound():
    values = pd.Series([0.01, 0.02, 0.03])

    observed = release_gates._mean_lcb_95(values)
    expected = 0.02 - 4.302652729911275 * (0.01 / math.sqrt(3))
    normal_approximation = 0.02 - 1.96 * (0.01 / math.sqrt(3))

    assert observed == pytest.approx(expected)
    assert observed < normal_approximation


def test_thirty_validated_receipts_pass_full_sample_policy(tmp_path, monkeypatch):
    _write_policy(tmp_path)
    pairs = [("BTC", "ETH"), ("SOL", "HYPE"), ("DOGE", "XRP")]
    regimes = ["range", "bull", "bear"]
    base = datetime(2026, 7, 1, 12, tzinfo=UTC)

    class FrozenDateTime(datetime):
        current = base

        @classmethod
        def now(cls, tz=None):
            return cls.current if tz is not None else cls.current.replace(tzinfo=None)

    monkeypatch.setattr(learning_risk, "datetime", FrozenDateTime)
    for index in range(30):
        sample_time = base + timedelta(days=index % 14, minutes=index)
        FrozenDateTime.current = sample_time
        config, _ = _write_valid_lifecycle_source(
            tmp_path,
            sample_number=100 + index,
            started=sample_time,
            assets=pairs[index % len(pairs)],
            regime=regimes[index % len(regimes)],
            net_after_cost=0.05,
            approval_time=sample_time,
        )
        archived = archive_validated_testnet_lifecycle(
            root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
        )
        assert archived["status"] == "PASS"

    result = build_testnet_sample_sufficiency(root=tmp_path)
    index = pd.read_csv(tmp_path / "data" / "testnet" / "lifecycle_index.csv")

    assert len(index) == 30
    assert result["status"] == "PASS"
    assert result["frame"]["status"].eq("PASS").all()

    supreme_path = tmp_path / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
    supreme = {
        "schema_version": release_gates.SCHEMA_VERSION,
        "generated_at_utc": FrozenDateTime.current.isoformat(),
        "checkpoint_status": "PASS",
        "blockers": [],
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    supreme_path.parent.mkdir(parents=True, exist_ok=True)
    supreme_path.write_text(json.dumps(supreme), encoding="utf-8")
    monkeypatch.setattr(
        release_gates,
        "build_testnet_supreme_team_checkpoint",
        lambda **_: {
            "path": supreme_path,
            "checkpoint": supreme,
        },
    )
    stage6 = build_stage6_release_receipt(
        root=tmp_path,
        now=FrozenDateTime.current,
    )
    candidate = json.loads(
        (tmp_path / "reports" / "active" / "testnet_candidate_receipt.json").read_text()
    )
    valid, receipt, receipt_path, blockers = validate_stage6_release_evidence(
        root=tmp_path,
        candidate=candidate,
        sample_evidence_path=result["path"],
        supreme_evidence_path=supreme_path,
    )

    assert stage6["status"] == "PASS"
    assert valid is True
    assert blockers == []
    assert receipt_path == stage6["receipt_path"]
    assert receipt["validated_lifecycle_count"] == 30
    assert len(receipt["validated_lifecycle_receipt_hashes"]) == 30

    result["path"].write_text("check,status\nforged,PASS\n", encoding="utf-8")
    forged_valid, _, _, forged_blockers = validate_stage6_release_evidence(
        root=tmp_path,
        candidate=candidate,
        sample_evidence_path=result["path"],
        supreme_evidence_path=supreme_path,
    )

    assert forged_valid is False
    assert any(blocker.startswith("stage6_release_source_changed:") for blocker in forged_blockers)


def test_token_pair_candidate_and_regime_breadth_cannot_pass(tmp_path):
    _write_policy(tmp_path)
    pairs = [("BTC", "ETH")] * 15 + [("SOL", "HYPE")] * 14 + [("DOGE", "XRP")]
    regimes = ["range"] * 15 + ["bull"] * 14 + ["bear"]
    base = datetime(2026, 7, 1, 12, tzinfo=UTC)
    for index, (assets, regime) in enumerate(zip(pairs, regimes, strict=True)):
        config, _ = _write_valid_lifecycle_source(
            tmp_path,
            sample_number=300 + index,
            started=base + timedelta(days=index % 14, minutes=index),
            assets=assets,
            regime=regime,
            net_after_cost=0.05,
        )
        assert (
            archive_validated_testnet_lifecycle(
                root=tmp_path,
                approval_secret=APPROVAL_SECRET,
                config=config,
            )["status"]
            == "PASS"
        )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    checks = result["frame"].set_index("check")

    assert checks.loc["pair_sample_concentration", "status"] == "PASS"
    assert checks.loc["regime_sample_concentration", "status"] == "PASS"
    assert checks.loc["accepted_candidate_coverage", "status"] == "PASS"
    assert checks.loc["lifecycles_per_accepted_candidate", "status"] == "BLOCKED"
    assert checks.loc["lifecycles_per_pair", "status"] == "BLOCKED"
    assert checks.loc["lifecycles_per_regime", "status"] == "BLOCKED"
    assert result["status"] == "BLOCKED"


def test_obsolete_model_lifecycles_cannot_unlock_active_model_cohort(tmp_path):
    _write_policy(tmp_path)
    base = datetime(2026, 7, 1, 12, tzinfo=UTC)
    for index in range(3):
        config, _ = _write_valid_lifecycle_source(
            tmp_path,
            sample_number=200 + index,
            started=base + timedelta(days=index),
            model_training_dataset_id="obsolete-dataset",
            model_artifact_sha256="b" * 64,
        )
        assert (
            archive_validated_testnet_lifecycle(
                root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
            )["status"]
            == "PASS"
        )

    config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=203,
        started=base + timedelta(days=3),
        model_training_dataset_id="active-dataset",
        model_artifact_sha256="c" * 64,
    )
    assert (
        archive_validated_testnet_lifecycle(
            root=tmp_path, approval_secret=APPROVAL_SECRET, config=config
        )["status"]
        == "PASS"
    )

    result = build_testnet_sample_sufficiency(root=tmp_path)
    checks = result["frame"].set_index("check")

    assert result["status"] == "BLOCKED"
    assert result["validated_lifecycle_samples"] == 4
    assert result["current_model_cohort_samples"] == 1
    assert result["excluded_obsolete_model_samples"] == 3
    assert checks.loc["active_model_lineage_bound", "status"] == "PASS"
    assert int(checks.loc["closed_paired_lifecycles", "observed"]) == 1
    assert checks.loc["closed_paired_lifecycles", "status"] == "BLOCKED"


def test_obsolete_registered_learning_lifecycle_cannot_join_active_stage5_cohort(
    tmp_path,
):
    _write_policy(tmp_path)
    base = datetime(2026, 7, 1, 12, tzinfo=UTC)
    obsolete_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=220,
        started=base,
        registered_learning_id="obsolete-registered-learning",
        registered_stage5_protocol_id="obsolete-stage5-protocol",
        registered_stage5_protocol_sha256="e" * 64,
        registered_execution_id="obsolete-stage4-execution",
    )
    assert (
        archive_validated_testnet_lifecycle(
            root=tmp_path,
            approval_secret=APPROVAL_SECRET,
            config=obsolete_config,
        )["status"]
        == "PASS"
    )

    active_config, _ = _write_valid_lifecycle_source(
        tmp_path,
        sample_number=221,
        started=base + timedelta(days=1),
        registered_learning_id="active-registered-learning",
        registered_stage5_protocol_id="active-stage5-protocol",
        registered_stage5_protocol_sha256="f" * 64,
        registered_execution_id="active-stage4-execution",
    )
    assert (
        archive_validated_testnet_lifecycle(
            root=tmp_path,
            approval_secret=APPROVAL_SECRET,
            config=active_config,
        )["status"]
        == "PASS"
    )

    result = build_testnet_sample_sufficiency(root=tmp_path)

    assert result["validated_lifecycle_samples"] == 2
    assert result["current_model_cohort_samples"] == 1
    assert result["excluded_obsolete_model_samples"] == 1
