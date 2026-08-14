from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_statistical_remediation import (
    _build_registered_candidate_cohort,
    _jaccard,
    _missing_proof,
    _missing_proofs,
    _overlay_registered_pair_cost_evidence,
    build_corrective_statistical_remediation,
    build_independent_strategy_breadth,
)

NOW = datetime(2026, 8, 9, 19, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("readiness_status", "readiness_blockers"),
    [
        ("PASS_STAGE4_HANDOFF_READY", []),
        ("BLOCKED_STAGE4_HANDOFF", ["contract_binding_invalid"]),
    ],
)
def test_statistical_remediation_refreshes_stage4_handoff_after_registered_gate(
    tmp_path, monkeypatch, readiness_status, readiness_blockers
):
    events = []
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_strategy_equivalence_clusters",
        lambda **_: {
            "path": active / "clusters.csv",
            "clusters": 1,
            "duplicate_rows": 0,
        },
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_near_miss_queue",
        lambda **_: {
            "queue": active / "queue.csv",
            "markdown": active / "queue.md",
            "batch": active / "batch.csv",
            "candidates": 1,
        },
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_corrective_research_funnel",
        lambda **_: {"path": active / "funnel.csv", "markdown": active / "funnel.md"},
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_independent_strategy_breadth",
        lambda **_: {
            "path": active / "breadth.csv",
            "supporting_clusters": 0,
            "full_survivor_clusters": 0,
            "supporting_pairs": 0,
            "full_survivor_pairs": 0,
        },
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "issue_final_one_x_survivor_receipt",
        lambda **_: {
            "path": active / "final.json",
            "receipt": {
                "receipt_status": "ZERO_SURVIVORS",
                "final_one_x_survivors": 0,
                "blockers": ["zero_final_survivors"],
            },
        },
    )

    rerun_paths = {
        "contract": active / "contract.json",
        "gate": active / "gate.csv",
        "summary": active / "gate.json",
        "family_preflight": active / "preflight.csv",
        "family_preflight_summary": active / "preflight.json",
    }
    rerun_summary = {
        "status": "BLOCKED_VENDOR_PARITY",
        "registered_candidates": 1,
        "candidate_gates_ready": 0,
        "pending_family_preflight_status": "PASS",
        "pending_family_hypotheses": 3,
        "pending_family_ready": 3,
        "pending_family_pairs": 3,
        "pending_family_independent_clusters": 3,
        "pending_family_rollover_required": True,
        "registered_rerun_results_accounted": False,
        "registered_rerun_conclusion_status": "INCOMPLETE",
        "accepted_registered_hypotheses": 0,
        "rejected_registered_hypotheses": 0,
        "next_action": "reconcile_frozen_wizard_capture_manifest",
    }

    def rerun_gate(**_):
        events.append("registered_gate")
        return CommandResult(paths=rerun_paths, summary=rerun_summary)

    def readiness(**_):
        assert events == ["registered_gate"]
        events.append("stage4_readiness")
        return CommandResult(
            paths={
                "checks": active / "stage4_checks.csv",
                "status": active / "stage4.json",
                "summary": active / "stage4.md",
                "immutable_receipt": tmp_path / "data" / "stage4_receipt.json",
            },
            summary={
                "status": readiness_status,
                "handoff_state": "READY_AWAITING_STAGE3",
                "receipt_id": "stage4handoff_test",
                "checked_at_utc": NOW.isoformat(),
                "checks_passed": 12 if not readiness_blockers else 11,
                "checks_total": 12,
                "blockers": readiness_blockers,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_registered_rerun_gate",
        rerun_gate,
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_corrective_stage4_handoff_readiness",
        readiness,
    )

    result = build_corrective_statistical_remediation(root=tmp_path, now=NOW)

    assert events == ["registered_gate", "stage4_readiness"]
    assert result.summary["stage4_handoff_readiness_status"] == readiness_status
    assert result.summary["stage4_handoff_receipt_id"] == "stage4handoff_test"
    assert result.summary["stage4_handoff_blockers"] == readiness_blockers
    assert result.paths["stage4_handoff_readiness"] == active / "stage4.json"
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_trade_overlap_clusters_exact_duplicates():
    left = {("a", "b", "long"), ("c", "d", "short")}
    assert _jaccard(left, set(left)) == 1.0
    assert _jaccard(left, {("x", "y", "long")}) == 0.0


def test_missing_proof_order_is_fail_closed():
    row = pd.Series(
        {
            "strict_cost_calibration_ready": False,
            "vendor_exact_mode_parity_proven": False,
            "statistical_selection_status": "PASS",
            "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
            "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
            "concentration_gate_pass": True,
        }
    )
    assert _missing_proof(row) == "strict_observed_cost_calibration"
    assert _missing_proofs(row) == [
        "strict_observed_cost_calibration",
        "vendor_formula_parity",
    ]


def test_registered_cost_overlay_advances_only_fresh_identity_matched_evidence():
    source = pd.DataFrame(
        [
            {
                "pair_group_key": "pair-1",
                "pair": "ETH-USD-PYTH-USD",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "strict_cost_calibration_ready": False,
                "vendor_exact_mode_parity_proven": False,
                "statistical_selection_status": "PASS",
                "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
                "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
                "concentration_gate_pass": True,
                "all_independent_blockers": (
                    "execution_cost:strict_l2_calibration_incomplete;"
                    "mode_fidelity:vendor_custom_series_parity_not_proven"
                ),
                "evidence_path": "reports/source.csv",
            }
        ]
    )
    model = pd.DataFrame(
        [
            {
                "pair_group_key": "pair-1",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "model_as_of_utc": NOW.isoformat(),
                "strict_observed_cost_ready": True,
                "cost_acceptance_ready": True,
                "cost_model_status": "STRICT_OBSERVED",
                "estimated_pair_round_trip_cost_bps": 18.5,
                "cost_model_id": "cost-1",
                "candidate_set_sha256": "a" * 64,
                "cost_status_sha256": "b" * 64,
                "l2_samples_sha256": "c" * 64,
                "fee_profile_sha256": "d" * 64,
                "evidence_path": "reports/cost-evidence.csv",
            }
        ]
    )

    overlaid = _overlay_registered_pair_cost_evidence(source, model, now=NOW)
    row = overlaid.iloc[0]

    assert bool(row["source_strict_cost_calibration_ready"]) is False
    assert bool(row["strict_cost_calibration_ready"]) is True
    assert row["strict_cost_overlay_status"] == "PASS"
    assert "execution_cost:" not in row["all_independent_blockers"]
    assert _missing_proof(row) == "vendor_formula_parity"

    stale = _overlay_registered_pair_cost_evidence(
        source, model, now=NOW + timedelta(hours=3)
    ).iloc[0]

    assert bool(stale["strict_cost_calibration_ready"]) is False
    assert "registered_pair_cost_model_stale_or_future" in stale["strict_cost_overlay_blocker"]
    assert "execution_cost:" in stale["all_independent_blockers"]


def test_registered_candidate_cohort_fills_breadth_from_current_family_only():
    failure = pd.DataFrame(
        [
            {
                "experiment_id": "exp-eth-pyth",
                "pair_group_key": "dydx|daily|ETH|PYTH",
                "asset_x": "ETH",
                "asset_y": "PYTH",
                "matrix_status": "READY_FOR_POINT_IN_TIME_HISTORY",
                "canonical_replay_status": "RESEARCH_REPLAY_COMPLETE",
                "overall_research_rank": 1,
            },
            {
                "experiment_id": "exp-btc-wld",
                "pair_group_key": "dydx|daily|BTC|WLD",
                "asset_x": "BTC",
                "asset_y": "WLD",
                "matrix_status": "READY_FOR_POINT_IN_TIME_HISTORY",
                "canonical_replay_status": "RESEARCH_REPLAY_COMPLETE",
                "overall_research_rank": 15,
            },
            {
                "experiment_id": "exp-eth-ldo",
                "pair_group_key": "dydx|daily|ETH|LDO",
                "asset_x": "ETH",
                "asset_y": "LDO",
                "matrix_status": "READY_FOR_POINT_IN_TIME_HISTORY",
                "canonical_replay_status": "RESEARCH_REPLAY_COMPLETE",
                "overall_research_rank": 16,
            },
            {
                "experiment_id": "exp-old-family",
                "pair_group_key": "historical|daily|XLM|HBAR",
                "asset_x": "XLM",
                "asset_y": "HBAR",
                "matrix_status": "BLOCKED_HYPERLIQUID_MAPPING",
                "canonical_replay_status": "NOT_RUN",
                "overall_research_rank": 2,
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {
                "experiment_id": "exp-eth-pyth",
                "equivalence_cluster_id": "eq-eth-pyth",
                "cluster_member_count": 1,
                "independent_evidence_weight": 1.0,
            }
        ]
    )

    cohort = _build_registered_candidate_cohort(
        failure,
        clusters,
        required_pairs=3,
    ).sort_values("overall_research_rank")

    assert list(cohort["experiment_id"]) == [
        "exp-eth-pyth",
        "exp-btc-wld",
        "exp-eth-ldo",
    ]
    prospective = cohort.loc[
        cohort["registration_cohort_role"].eq("prospective_current_family_breadth")
    ]
    assert len(prospective) == 2
    assert prospective["acceptance_evidence_weight"].eq(0.0).all()
    assert prospective["equivalence_cluster_id"].str.startswith("prospective_pair_").all()


def test_breadth_cannot_pass_with_three_supporting_clusters_but_one_full_survivor(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        [
            {
                "experiment_id": f"experiment-{index}",
                "equivalence_cluster_id": f"cluster-{index}",
                "statistical_selection_status": "PASS",
                "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
                "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
                "one_x_research_survivor": index == 0,
                "all_current_research_gates_pass": index == 0,
                "cluster_member_count": 1,
                "asset_x": f"ASSET-{index}",
                "asset_y": f"HEDGE-{index}",
            }
            for index in range(3)
        ]
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_corrective_research_funnel",
        lambda **_: {"frame": frame},
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "_minimum_independent_clusters",
        lambda _: 3,
    )

    result = build_independent_strategy_breadth(root=tmp_path, now=NOW)

    assert result["supporting_clusters"] == 3
    assert result["full_survivor_clusters"] == 1
    assert result["supporting_pairs"] == 3
    assert result["full_survivor_pairs"] == 1
    assert result["status"] == "BLOCKED"


def test_breadth_passes_only_with_three_independent_full_survivor_clusters(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        [
            {
                "experiment_id": f"experiment-{index}",
                "equivalence_cluster_id": f"cluster-{index}",
                "statistical_selection_status": "PASS",
                "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
                "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
                "one_x_research_survivor": True,
                "all_current_research_gates_pass": True,
                "cluster_member_count": 1,
                "asset_x": f"ASSET-{index}",
                "asset_y": f"HEDGE-{index}",
            }
            for index in range(3)
        ]
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_corrective_research_funnel",
        lambda **_: {"frame": frame},
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "_minimum_independent_clusters",
        lambda _: 3,
    )

    result = build_independent_strategy_breadth(root=tmp_path, now=NOW)

    assert result["supporting_clusters"] == 3
    assert result["full_survivor_clusters"] == 3
    assert result["supporting_pairs"] == 3
    assert result["full_survivor_pairs"] == 3
    assert result["status"] == "PASS"


def test_three_clusters_from_one_pair_cannot_manufacture_independent_breadth(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame(
        [
            {
                "experiment_id": f"experiment-{index}",
                "equivalence_cluster_id": f"cluster-{index}",
                "statistical_selection_status": "PASS",
                "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
                "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
                "one_x_research_survivor": True,
                "all_current_research_gates_pass": True,
                "cluster_member_count": 1,
                "asset_x": "SOL",
                "asset_y": "TURBO",
            }
            for index in range(3)
        ]
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "build_corrective_research_funnel",
        lambda **_: {"frame": frame},
    )
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_statistical_remediation."
        "_minimum_independent_clusters",
        lambda _: 3,
    )

    result = build_independent_strategy_breadth(root=tmp_path, now=NOW)

    assert result["supporting_clusters"] == 3
    assert result["full_survivor_clusters"] == 3
    assert result["supporting_pairs"] == 1
    assert result["full_survivor_pairs"] == 1
    assert result["status"] == "BLOCKED"
