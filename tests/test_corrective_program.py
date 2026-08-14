from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd
import pytest

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_program import (
    TASK_DEFINITION_OVERRIDES,
    TASK_STATUS,
    _capture_manifest_continuity_pass,
    _capture_reconciliation_pass,
    _classify_checkpoint_evidence_paths,
    _daily_cadence_task_status,
    _next_ou_v5_action,
    _next_ou_v6_action,
    _next_strict_cost_action,
    _next_wizard_proof_action,
    _ou_v4_immutable_evidence_complete,
    _ou_v6_immutable_evidence_complete,
    _refresh_testnet_collateral_preflight,
    _registered_contract_identity_valid,
    _registered_rerun_task_status,
    _stage_five_blockers,
    _stage_four_completion_state,
    _strict_cost_task_status,
    _write_seven_stage_checkpoint,
    complete_corrective_plan,
)
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    wizard_reset_readiness_state_sha256,
)
from tests.browser_auth_support import write_browser_auth_binding
from tests.capture_reconciliation_support import (
    write_valid_capture_reconciliation_evidence,
)


def test_checkpoint_evidence_classification_distinguishes_planned_and_required(tmp_path):
    existing = tmp_path / "reports" / "active" / "existing.json"
    existing.parent.mkdir(parents=True)
    existing.write_text("{}", encoding="utf-8")

    material, planned, missing = _classify_checkpoint_evidence_paths(
        root=tmp_path,
        evidence_path=(
            "reports/active/existing.json;"
            "reports/active/planned.json;"
            "required:reports/active/required.json"
        ),
    )

    assert material == "reports/active/existing.json"
    assert planned == "reports/active/planned.json"
    assert missing == "reports/active/required.json"


def test_stage_five_blockers_distinguish_waiting_from_invalid_terminal_evidence():
    waiting = _stage_five_blockers(
        stage_five_pass=False,
        stage_four_complete=False,
        registered_learning={
            "status": "BLOCKED_STAGE4",
            "blocker": "FileNotFoundError:registered Stage 4 execution receipt is unavailable",
        },
        registered_learning_available=True,
        registered_learning_audit={
            "evidence_valid": False,
            "blockers": ["ValueError:registered learning active status is not terminal"],
        },
        stage5_protocol_pass=True,
        model_blockers=["model_incremental_edge_not_accepted"],
    )
    invalid_terminal = _stage_five_blockers(
        stage_five_pass=False,
        stage_four_complete=True,
        registered_learning={
            "status": "PASS_RESEARCH_LEARNING_GATES",
            "blocker": "",
        },
        registered_learning_available=True,
        registered_learning_audit={
            "evidence_valid": False,
            "blockers": ["ValueError:registered learning receipt hash mismatch"],
        },
        stage5_protocol_pass=True,
        model_blockers=["model_incremental_edge_not_accepted"],
    )

    assert waiting == "registered_stage4_execution_receipt_not_yet_available"
    assert "FileNotFoundError" not in waiting
    assert "not terminal" not in waiting
    assert "model_incremental_edge_not_accepted" not in waiting
    assert invalid_terminal == "ValueError:registered learning receipt hash mismatch"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"required": False}, ""),
        ({}, "run_governed_ou_v5_holdout_after_utc_reset"),
        (
            {"immutable_evidence_complete": True},
            "repair_ou_v5_immutable_holdout_evidence",
        ),
        ({"evaluation": {"status": "FAIL"}}, "build_ou_v5_failure_attribution"),
        (
            {
                "evaluation": {"status": "FAIL"},
                "failure_attribution": {"status": "PASS_FAILURE_ATTRIBUTION_COMPLETE"},
            },
            "supreme_team_review_ou_v5_failure_before_successor_preregistration",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
            },
            "build_or_repair_ou_v5_supersession_review_packet",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "review_packet": {"status": "READY_FOR_EXPLICIT_REVIEW"},
            },
            "build_or_repair_ou_v5_supreme_team_review",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "review_packet": {"status": "READY_FOR_EXPLICIT_REVIEW"},
                "supreme_review": {"status": "PASS_ADVISORY_ONLY"},
            },
            "human_review_ou_v5_exact_packet_and_apply_research_comparator",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "review_packet": {"status": "READY_FOR_EXPLICIT_REVIEW"},
                "supreme_review": {"status": "PASS_ADVISORY_ONLY"},
                "activation": {"status": "APPLIED_RESEARCH_COMPARATOR_ONLY"},
            },
            "refresh_ou_v5_activation_bound_current_proof_rows",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "review_packet": {"status": "READY_FOR_EXPLICIT_REVIEW"},
                "supreme_review": {"status": "PASS_ADVISORY_ONLY"},
                "activation": {"status": "APPLIED_RESEARCH_COMPARATOR_ONLY"},
                "proof_refresh": {"status": "PASS"},
            },
            "",
        ),
    ],
)
def test_ou_v5_next_action_routes_every_terminal_substate(overrides, expected):
    values = {
        "required": True,
        "immutable_evidence_complete": False,
        "evaluation": {},
        "failure_attribution": {},
        "review_packet": {},
        "supreme_review": {},
        "activation": {},
        "proof_refresh": {},
        **overrides,
    }

    assert _next_ou_v5_action(**values) == expected


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"required": False}, ""),
        ({}, "run_governed_terminal_ou_v6_holdout_after_utc_reset"),
        (
            {"immutable_evidence_complete": True},
            "repair_ou_v6_immutable_holdout_evidence",
        ),
        (
            {"evaluation": {"status": "FAIL"}},
            "conclusively_close_local_ou_exact_parity_after_terminal_v6_failure",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
            },
            "build_and_human_review_ou_v6_terminal_success_activation_packet",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "activation": {"status": "APPLIED_RESEARCH_COMPARATOR_ONLY"},
            },
            "refresh_ou_v6_activation_bound_current_proof_rows",
        ),
        (
            {
                "immutable_evidence_complete": True,
                "evaluation": {"status": "PASS"},
                "activation": {"status": "APPLIED_RESEARCH_COMPARATOR_ONLY"},
                "proof_refresh": {"status": "PASS"},
            },
            "",
        ),
    ],
)
def test_ou_v6_next_action_is_terminal_and_never_routes_to_v7(overrides, expected):
    values = {
        "required": True,
        "immutable_evidence_complete": False,
        "evaluation": {},
        "activation": {},
        "proof_refresh": {},
        **overrides,
    }

    assert _next_ou_v6_action(**values) == expected
    assert "v7" not in expected


def test_ou_v6_completion_requires_terminal_policy_and_immutable_research_evidence(tmp_path):
    immutable = (
        tmp_path
        / "data"
        / "research"
        / "wizard_ou_v6_holdout_evaluations"
        / "ouv6holdout_test.json"
    )
    immutable.parent.mkdir(parents=True)
    immutable.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evaluation = {
        "status": "PASS",
        "required_cells": 8,
        "passed_cells": 8,
        "final_successor_iteration": True,
        "successor_after_v6_failure_allowed": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "result_id": immutable.stem,
        "immutable_result_path": str(immutable.relative_to(tmp_path)),
        "immutable_result_sha256": sha256(immutable.read_bytes()).hexdigest(),
    }

    assert _ou_v6_immutable_evidence_complete(tmp_path, evaluation)
    assert not _ou_v6_immutable_evidence_complete(
        tmp_path,
        {**evaluation, "successor_after_v6_failure_allowed": True},
    )


def test_ou_v4_completion_requires_immutable_research_only_evidence(tmp_path):
    immutable = (
        tmp_path
        / "data"
        / "research"
        / "wizard_ou_v4_holdout_evaluations"
        / "ouv4holdout_test.json"
    )
    immutable.parent.mkdir(parents=True)
    immutable.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evaluation = {
        "status": "PASS",
        "required_cells": 8,
        "passed_cells": 8,
        "comparator_supersession_eligible": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "result_id": immutable.stem,
        "immutable_result_path": str(immutable.relative_to(tmp_path)),
        "immutable_result_sha256": sha256(immutable.read_bytes()).hexdigest(),
    }

    assert _ou_v4_immutable_evidence_complete(tmp_path, evaluation)
    assert not _ou_v4_immutable_evidence_complete(
        tmp_path,
        {**evaluation, "live_trading_authorized": True},
    )
    immutable.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    assert not _ou_v4_immutable_evidence_complete(tmp_path, evaluation)


def test_capture_reconciliation_requires_positive_fully_bound_cohort():
    assert _capture_reconciliation_pass(
        {
            "status": "PASS",
            "required_calls": 13,
            "completed_calls": 13,
            "pending_calls": 0,
            "blocked_calls": 0,
        }
    )
    assert not _capture_reconciliation_pass(
        {
            "status": "NOT_REQUIRED",
            "required_calls": 0,
            "completed_calls": 0,
            "pending_calls": 0,
            "blocked_calls": 0,
        }
    )
    assert not _capture_reconciliation_pass(
        {
            "status": "PASS",
            "required_calls": 13,
            "completed_calls": 12,
            "pending_calls": 1,
            "blocked_calls": 0,
        }
    )


def test_capture_manifest_continuity_requires_valid_nondrifted_candidate():
    valid = {
        "capture_manifest_continuity_status": ("PASS_PRIOR_UNRESOLVED_COHORT_MATCH"),
        "capture_manifest_continuity_valid": True,
        "capture_manifest_candidate_binding_valid": True,
        "capture_manifest_candidate_source_binding_valid": True,
        "capture_manifest_id": "wizardcapture_test",
        "capture_manifest_source_receipt_id": "wizardcapturesources_test",
        "capture_manifest_source_receipt_path": (
            "data/research/wizard_capture_manifest_sources/wizardcapture_test.json"
        ),
        "capture_manifest_source_receipt_sha256": "a" * 64,
        "capture_manifest_source_artifacts_sha256": "b" * 64,
        "capture_manifest_drift_detected": False,
    }
    assert _capture_manifest_continuity_pass(valid)
    assert not _capture_manifest_continuity_pass({**valid, "capture_manifest_drift_detected": True})
    assert not _capture_manifest_continuity_pass(
        {**valid, "capture_manifest_candidate_binding_valid": False}
    )
    assert not _capture_manifest_continuity_pass(
        {**valid, "capture_manifest_candidate_source_binding_valid": False}
    )
    assert not _capture_manifest_continuity_pass(
        {**valid, "capture_manifest_continuity_status": "BLOCKED_COHORT_DRIFT"}
    )


def _write_stage_four_policy_lineage(root):
    config = root / "config"
    active = root / "reports" / "active"
    config.mkdir(parents=True, exist_ok=True)
    active.mkdir(parents=True, exist_ok=True)
    acceptance_policy = {
        "schema_version": "thewiz.acceptance_policy.v1",
        "research_gates": {"minimum_independent_supporting_clusters": 3},
    }
    acceptance_path = config / "acceptance_policy_manifest.json"
    acceptance_path.write_text(json.dumps(acceptance_policy), encoding="utf-8")
    holdout_path = config / "research_holdout_policy.json"
    holdout_path.write_text(json.dumps({"schema_version": "holdout-test"}), encoding="utf-8")
    acceptance_id = "acceptance-test"
    holdout_id = "holdout-test"
    (active / "acceptance_policy_receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "policy_id": acceptance_id,
                "policy_path": "config/acceptance_policy_manifest.json",
                "policy_sha256": sha256(acceptance_path.read_bytes()).hexdigest(),
                "source_contracts_match": True,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "holdout_policy_receipt.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "policy_id": holdout_id,
                "policy_path": "config/research_holdout_policy.json",
                "policy_sha256": sha256(holdout_path.read_bytes()).hexdigest(),
                "final_holdout_locked": True,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    return acceptance_id, holdout_id


def _write_verified_wizard_execution(root):
    capture_evidence = write_valid_capture_reconciliation_evidence(root)
    payload = {
        "schema_version": "thewiz.corrective_wizard_proof_scheduler.v1",
        "attempt_date_utc": "2026-08-11",
        "started_at_utc": "2026-08-11T12:30:00+00:00",
        "status": "COMPLETE_ACCEPTED_MODE_EVIDENCE",
        "execution_requested": True,
        "final_immutable_receipt_required": True,
        "queue_eligible": 13,
        "responses_captured_after": 13,
        "accepted_mode_evidence_cells": 13,
        "parity_refresh_status": "PASS",
        "credit_reconciliation_status": "PASS_RECONCILED",
        **capture_evidence,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    payload["receipt_id"] = "wizardproof_" + sha256(canonical.encode("utf-8")).hexdigest()[:20]
    active = (
        root
        / "reports"
        / "active"
        / "wizard_proof_scheduler_receipts"
        / "2026-08-11_123000_000000.json"
    )
    active.parent.mkdir(parents=True, exist_ok=True)
    active.write_text(json.dumps(payload), encoding="utf-8")
    immutable = (
        root
        / "data"
        / "research"
        / "wizard_proof_scheduler_receipts"
        / f"{payload['receipt_id']}.json"
    )
    immutable.parent.mkdir(parents=True, exist_ok=True)
    immutable.write_bytes(active.read_bytes())
    return payload["receipt_id"]


def _write_stage_four_contract(
    root,
    semantic_ids,
    *,
    acceptance_id,
    holdout_id,
    generation,
    prior_contract_id="",
    active=False,
):
    material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": [
            {"semantic_hypothesis_id": semantic_id} for semantic_id in semantic_ids
        ],
        "acceptance_policy_id": acceptance_id,
        "holdout_policy_id": holdout_id,
        "discovery_policy_sha256": "a" * 64,
        "source_family_sha256": "b" * 64,
        "source_family_rows": 100,
        "generation": generation,
        "prior_contract_id": prior_contract_id,
    }
    contract_id = (
        "registeredrerun_"
        + sha256(json.dumps(material, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[
            :20
        ]
    )
    contract = {
        **material,
        "contract_id": contract_id,
        "full_family_multiplicity_required": True,
        "promotion_evaluation_registered_only": True,
        "threshold_changes_after_contract_permitted": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    immutable = root / "data" / "research" / "registered_rerun_contracts"
    immutable.mkdir(parents=True, exist_ok=True)
    (immutable / f"{contract_id}.json").write_text(json.dumps(contract), encoding="utf-8")
    if active:
        active_path = root / "reports" / "active"
        active_path.mkdir(parents=True, exist_ok=True)
        (active_path / "registered_research_rerun_contract.json").write_text(
            json.dumps(contract), encoding="utf-8"
        )
    return contract


def _write_stage_four_conclusion(root, contract, outcomes):
    accepted = sum(value == "ACCEPTED_SURVIVOR" for value in outcomes.values())
    rejected = sum(value == "REJECTED_BY_FROZEN_GATES" for value in outcomes.values())
    conclusion = {
        "schema_version": "thewiz.corrective_registered_rerun_conclusion.v2",
        "contract_id": contract["contract_id"],
        "registered_hypotheses": len(outcomes),
        "accepted_registered_hypotheses": accepted,
        "rejected_registered_hypotheses": rejected,
        "conclusion_status": (
            "ACCEPTED_REGISTERED_SURVIVORS" if accepted else "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"
        ),
        "outcomes": outcomes,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    conclusion["conclusion_id"] = (
        "registeredconclusion_"
        + sha256(
            json.dumps(conclusion, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:20]
    )
    path = root / "data" / "research" / "registered_rerun_conclusions"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{contract['contract_id']}.json").write_text(json.dumps(conclusion), encoding="utf-8")


def _write_stage_four_final_receipt(
    root,
    *,
    acceptance_id,
    holdout_id,
    final_survivors,
    independent_clusters=0,
):
    full_survivor_breadth = independent_clusters if final_survivors else 0
    path = root / "reports" / "active"
    path.mkdir(parents=True, exist_ok=True)
    receipt_path = path / "final_1x_survivor_receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": "thewiz.final_one_x_survivor_receipt.v1",
                "receipt_status": "PASS" if final_survivors else "ZERO_SURVIVORS",
                "acceptance_policy_id": acceptance_id,
                "holdout_policy_id": holdout_id,
                "final_one_x_survivors": final_survivors,
                "independent_supporting_clusters": independent_clusters,
                "independent_full_survivor_clusters": full_survivor_breadth,
                "independent_supporting_pairs": independent_clusters,
                "independent_full_survivor_pairs": full_survivor_breadth,
                "final_experiment_ids": [f"experiment-{index}" for index in range(final_survivors)],
                "final_canonical_pairs": [
                    f"ASSET-{index}-PAIR-{index}" for index in range(full_survivor_breadth)
                ],
                "blockers": [] if final_survivors else ["zero_final_survivors"],
                "testnet_candidate_authority": bool(final_survivors),
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    conclusions = root / "data" / "research" / "registered_rerun_conclusions"
    if conclusions.is_dir():
        for conclusion_path in conclusions.glob("*.json"):
            conclusion = json.loads(conclusion_path.read_text(encoding="utf-8"))
            conclusion["final_survivor_receipt_sha256"] = sha256(
                receipt_path.read_bytes()
            ).hexdigest()
            conclusion.pop("conclusion_id", None)
            conclusion["conclusion_id"] = (
                "registeredconclusion_"
                + sha256(
                    json.dumps(conclusion, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()[:20]
            )
            conclusion_path.write_text(json.dumps(conclusion), encoding="utf-8")


def test_all_41_tasks_have_explicit_corrective_status():
    assert set(TASK_STATUS) == {f"T{index:02d}" for index in range(1, 42)}


def test_program_checkpoint_refuses_active_producer_lock(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    lock = active / ".corrective_daily.lock"
    lock.write_text(
        json.dumps(
            {
                "pid": 123,
                "started_at_utc": "2026-08-09T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError, match="active_scheduler_lock_present"):
        complete_corrective_plan(root=tmp_path, now=datetime(2026, 8, 9, 12, 1, tzinfo=UTC))

    assert lock.exists()


@pytest.mark.parametrize(
    ("policy_amount", "expected_amount"),
    [(23.0, 23.0), (100.0, 25.0), ("invalid", 25.0)],
)
def test_canonical_collateral_preflight_refresh_is_status_only_and_bounded(
    tmp_path,
    policy_amount,
    expected_amount,
):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(
            {
                "approval_id": "approval-current",
                "collateral_transfer_policy": {"amount_usd": policy_amount},
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def builder(**kwargs):
        calls.append(kwargs)
        return CommandResult(
            paths={"preflight": active / "testnet_collateral_transfer_preflight.json"},
            summary={
                "status": "BLOCKED",
                "candidate_receipt_id": "candidate-current",
                "agent_key_accessed": False,
                "transfer_attempted": False,
                "order_submission_performed": False,
                "testnet_collateral_transfer_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    now = datetime(2026, 8, 11, 18, 30, tzinfo=UTC)
    result = _refresh_testnet_collateral_preflight(
        root=tmp_path,
        now=now,
        builder=builder,
    )

    assert result.summary["status"] == "BLOCKED"
    assert len(calls) == 1
    assert calls[0]["root"] == tmp_path
    assert calls[0]["approval_id"] == "approval-current"
    assert calls[0]["amount_usd"] == expected_amount
    assert calls[0]["now"] == now


def test_canonical_collateral_preflight_rejects_authority_broadening(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "hyperliquid_testnet_smoke_approval.json").write_text("{}")

    def unsafe_builder(**_):
        return CommandResult(
            paths={},
            summary={
                "agent_key_accessed": True,
                "transfer_attempted": False,
                "order_submission_performed": False,
                "testnet_collateral_transfer_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    with pytest.raises(RuntimeError, match="status-only contract"):
        _refresh_testnet_collateral_preflight(
            root=tmp_path,
            now=datetime(2026, 8, 11, 18, 30, tzinfo=UTC),
            builder=unsafe_builder,
        )


def test_conditional_execution_tasks_never_marked_unconditionally_complete():
    for task in ("T33", "T34", "T35", "T36", "T39", "T40", "T41"):
        assert TASK_STATUS[task][0] != "completed"


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({}, ("in_progress_time_observation", "daily_cadence_0_of_7_cycles_observed")),
        (
            {"consecutive_complete_cycles": 4, "required_complete_cycles": 7},
            ("in_progress_time_observation", "daily_cadence_4_of_7_cycles_observed"),
        ),
        (
            {"consecutive_complete_cycles": 7, "required_complete_cycles": 7},
            ("completed", "daily_cadence_7_of_7_cycles_observed"),
        ),
        (
            {"consecutive_complete_cycles": 8, "required_complete_cycles": 7},
            ("completed", "daily_cadence_7_of_7_cycles_observed"),
        ),
        (
            {"consecutive_complete_cycles": -1, "required_complete_cycles": 0},
            ("in_progress_time_observation", "daily_cadence_0_of_7_cycles_observed"),
        ),
        (
            {"consecutive_complete_cycles": "tampered", "required_complete_cycles": []},
            ("in_progress_time_observation", "daily_cadence_0_of_7_cycles_observed"),
        ),
    ],
)
def test_daily_cadence_task_status_tracks_authoritative_receipt(summary, expected):
    assert _daily_cadence_task_status(summary) == expected


def test_ou_optimal_overlay_stays_in_progress_until_both_orientations_are_observed():
    assert TASK_STATUS["T17"][0] == "in_progress_evidence_collection"
    assert "scanner_overlay" in TASK_STATUS["T17"][1]
    definition = TASK_DEFINITION_OVERRIDES["T17"]
    assert "seven pair-page exact modes" in definition["task"]
    assert "scanner overlay" in definition["task"]
    assert "eight exact modes" not in definition["task"]


def test_stage_three_requires_surface_inventory_when_contract_is_enabled(tmp_path):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    contract = tmp_path / "config/wizard_surface_inventory_contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}\n", encoding="utf-8")
    phases = {
        "venue_data_costs": result(
            strict_cost_ready_pairs=1,
            strict_cost_eligible_pairs=1,
            history_queued_pairs=0,
            history_ready_pairs=1,
        ),
        "wizard_parity": result(status="PASS", blocker=""),
        "statistical_remediation": result(
            final_one_x_survivors=0,
            registered_rerun_candidates=0,
            blockers=["waiting_for_stage3"],
        ),
        "daily_cadence": result(consecutive_complete_cycles=0, blocker="need_seven_days"),
        "agent_learning_governance": result(
            model_authority="RESEARCH_ONLY",
            blockers=["waiting_for_stage4"],
        ),
        "conditional_release_gates": result(testnet_sample_sufficient=False),
    }

    csv_path, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    stage_three = pd.read_csv(csv_path).loc[lambda frame: frame["stage"].eq(3)].iloc[0]

    assert stage_three["status"] == "BLOCKED"
    assert stage_three["next_action"] == "repair_wizard_surface_inventory_contract"
    assert "wizard_surface_inventory_not_ready" in stage_three["blocker"]
    assert "surface_inventory_enforced=True" in stage_three["evidence_progress"]
    assert not bool(stage_three["testnet_order_authority"])
    assert not bool(stage_three["live_trading_authorized"])


def test_stage_three_requires_verified_browser_auth_receipt_when_enabled(tmp_path):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    contract = tmp_path / "config/wizard_browser_auth_contract.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}\n", encoding="utf-8")
    phases = {
        "venue_data_costs": result(
            strict_cost_ready_pairs=1,
            strict_cost_eligible_pairs=1,
            history_queued_pairs=0,
            history_ready_pairs=1,
        ),
        "wizard_parity": result(status="PASS", blocker=""),
        "statistical_remediation": result(
            final_one_x_survivors=0,
            registered_rerun_candidates=0,
            blockers=["waiting_for_stage3"],
        ),
        "daily_cadence": result(consecutive_complete_cycles=0, blocker="need_seven_days"),
        "agent_learning_governance": result(
            model_authority="RESEARCH_ONLY",
            blockers=["waiting_for_stage4"],
        ),
        "conditional_release_gates": result(testnet_sample_sufficient=False),
    }

    csv_path, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    stage_three = pd.read_csv(csv_path).loc[lambda frame: frame["stage"].eq(3)].iloc[0]

    assert stage_three["status"] == "BLOCKED"
    assert stage_three["next_action"] == (
        "capture_fresh_authenticated_scanner_and_pair_route_receipts"
    )
    assert "wizard_browser_auth_readiness_not_proven" in stage_three["blocker"]
    assert "browser_auth_enforced=True" in stage_three["evidence_progress"]
    assert not bool(stage_three["testnet_order_authority"])
    assert not bool(stage_three["live_trading_authorized"])


def test_seven_stage_checkpoint_never_grants_order_authority(tmp_path):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    phases = {
        "venue_data_costs": result(
            strict_cost_ready_pairs=0,
            history_queued_pairs=1,
            history_ready_pairs=2,
            l2_readiness_refresh_status="WAITING_STRICT_L2",
            l2_readiness_refresh_validation_status="PASS",
            l2_readiness_refresh_receipt_id="l2readiness_test",
            l2_readiness_refresh_ready_pairs=1,
            l2_readiness_refresh_eligible_pairs=4,
            l2_readiness_refresh_gate_executed=False,
            l2_readiness_refresh_handoff_executed=False,
        ),
        "wizard_parity": result(status="BLOCKED", blocker="parity_missing"),
        "statistical_remediation": result(
            final_one_x_survivors=0,
            near_miss_candidates=1,
            registered_rerun_candidates=1,
            registered_rerun_candidate_gates_ready=0,
            registered_rerun_gate_status="BLOCKED_VENDOR_PARITY",
            registered_rerun_results_accounted=False,
            registered_rerun_conclusion_status="INCOMPLETE",
            registered_rerun_next_action="complete_vendor_parity",
            independent_supporting_clusters=0,
            blockers=["zero_survivors"],
        ),
        "daily_cadence": result(
            consecutive_complete_cycles=1,
            blocker="need_seven_days",
        ),
        "agent_learning_governance": result(
            model_authority="RESEARCH_ONLY",
            blockers=["incremental_edge_missing"],
        ),
        "conditional_release_gates": result(testnet_sample_sufficient=False),
    }
    scheduler = tmp_path / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    scheduler.parent.mkdir(parents=True)
    (scheduler.parent / "corrective_l2_capture_status.json").write_text(
        json.dumps(
            {
                "mapping_refresh_status": "NOT_DUE",
                "mapping_refresh_due": False,
                "mapping_age_hours_after": 1.5,
                "mapping_refresh_id": "hlmap_test",
                "mapping_ready_pair_groups": 26,
                "mapping_blocked_pair_groups": 103,
                "operational_warnings": [],
                "order_submission_included": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    scheduler.write_text(
        json.dumps(
            {
                "status": "PLANNED",
                "queue_eligible": 28,
                "completed_after": 0,
                "input_audit_ready": 28,
                "credit_budget_status": "PASS",
                "scheduled_credit_ceiling": 360,
                "credit_headroom_after_reserve": 540,
                "proof_lane_credit_ceiling": 68,
                "credit_reservation_status": "PASS",
                "credit_reconciliation_status": "PASS_RECONCILED",
                "capture_manifest_continuity_status": ("PASS_PRIOR_UNRESOLVED_COHORT_MATCH"),
                "capture_manifest_continuity_valid": True,
                "capture_manifest_candidate_binding_valid": True,
                "capture_manifest_candidate_source_binding_valid": True,
                "capture_manifest_id": "wizardcapture_test",
                "capture_manifest_source_receipt_id": "wizardcapturesources_test",
                "capture_manifest_source_receipt_path": (
                    "data/research/wizard_capture_manifest_sources/wizardcapture_test.json"
                ),
                "capture_manifest_source_receipt_sha256": "a" * 64,
                "capture_manifest_source_artifacts_sha256": "b" * 64,
                "capture_manifest_drift_detected": False,
                "remaining_vendor_responses": 28,
                "configured_proof_request_capacity": 30,
                "remaining_custom_series_credits": 56,
                "next_cohort_readiness": ("CAPACITY_READY_RUNTIME_PREFLIGHT_REQUIRED"),
                "next_utc_reset_at": "2026-08-11T00:00:00+00:00",
                "next_external_attempt_eligible_at": ("2026-08-11T00:00:00+00:00"),
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "corrective_wizard_proof_launcher_status.json").write_text(
        json.dumps(
            {
                "launcher_status": "DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT",
                "same_day_attempt_state": "ATTEMPTED",
                "heavy_scheduler_invoked": False,
                "latest_status_capture_manifest_binding_complete": True,
                "latest_immutable_execution_capture_manifest_binding_complete": False,
                "latest_status_stage3_evidence_complete": False,
                "latest_immutable_execution_stage3_evidence_complete": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "corrective_wizard_next_capture_manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "capture_state": "DEFERRED_UNTIL_UTC_RESET",
                "manifest_id": "wizardcapture_test",
                "pending_calls": 13,
                "planned_credits": 18,
                "next_external_attempt_eligible_at": ("2026-08-11T00:00:00+00:00"),
            }
        ),
        encoding="utf-8",
    )
    browser_auth_binding = write_browser_auth_binding(
        tmp_path,
        captured_at=datetime(2026, 8, 11, 11, 55, tzinfo=UTC),
        required_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
    )
    reset_material = {
        "schema_version": "thewiz.corrective_wizard_reset_readiness.v5",
        "status": "PASS_RESET_AUTOMATION_READY",
        "checks_passed": 12,
        "checks_total": 12,
        "launch_agent_loaded": True,
        "blockers": [],
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        **browser_auth_binding,
    }
    reset_receipt_id = (
        "wizardresetreadiness_"
        + sha256(
            json.dumps(reset_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
    )
    reset_receipt = {**reset_material, "receipt_id": reset_receipt_id}
    (scheduler.parent / "wizard_reset_readiness.json").write_text(
        json.dumps(reset_receipt),
        encoding="utf-8",
    )
    immutable_reset = (
        tmp_path / "data" / "research" / "wizard_reset_readiness" / f"{reset_receipt_id}.json"
    )
    immutable_reset.parent.mkdir(parents=True)
    immutable_reset.write_text(json.dumps(reset_receipt), encoding="utf-8")
    scheduler_runtime_path = scheduler.parent / "scheduler_runtime_readiness.json"
    scheduler_runtime_pass = {
        "status": "PASS_SCHEDULER_RUNTIME_READY",
        "agents_ready": 3,
        "agents_expected": 3,
        "checks_passed": 49,
        "checks_total": 49,
        "receipt_id": "schedulerruntime_test",
        "blockers": [],
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    scheduler_runtime_path.write_text(
        json.dumps(scheduler_runtime_pass),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_comparator_review_control.json").write_text(
        json.dumps(
            {
                "status": "WAITING_FOR_PROSPECTIVE_EVIDENCE",
                "comparators": 2,
                "review_ready": 0,
                "apply_ready": 0,
                "applied": 0,
                "receipt_id": "comparatorreview_test",
                "control_blockers": [],
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    stage4_contract_id = "registeredrerun_test"
    stage4_manifest_path = "data/research/wizard_capture_manifests/wizardcapture_test.json"
    stage4_protocol_path = "data/research/registered_stage5_protocols/stage5protocol_test.json"
    stage4_source_paths = {
        "active_contract": "reports/active/registered_research_rerun_contract.json",
        "immutable_contract": (
            f"data/research/registered_rerun_contracts/{stage4_contract_id}.json"
        ),
        "registered_source_receipt": (
            f"data/research/registered_rerun_source_families/{stage4_contract_id}/receipt.json"
        ),
        "registered_source_matrix": (
            "data/research/registered_rerun_source_families/"
            f"{stage4_contract_id}/experiment_matrix.csv"
        ),
        "active_gate": "reports/active/registered_research_rerun_gate.json",
        "active_gate_rows": "reports/active/registered_research_rerun_gate.csv",
        "pending_preflight": "reports/active/registered_rerun_family_preflight.json",
        "pending_preflight_rows": "reports/active/registered_rerun_family_preflight.csv",
        "current_hypothesis_batch": "reports/active/current_hypothesis_batch.csv",
        "active_stage3_manifest": ("reports/active/corrective_wizard_next_capture_manifest.json"),
        "immutable_stage3_manifest": stage4_manifest_path,
        "immutable_stage3_reset": (f"data/research/wizard_reset_readiness/{reset_receipt_id}.json"),
        "stage5_protocol_pointer": "reports/active/registered_stage5_protocol.json",
        "immutable_stage5_protocol": stage4_protocol_path,
        "proof_scheduler_status": ("reports/active/corrective_wizard_proof_scheduler_status.json"),
    }
    for relative in stage4_source_paths.values():
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.exists():
            source.write_text("{}\n", encoding="utf-8")
    stage4_source_artifacts = {
        name: {
            "path": relative,
            "sha256": sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for name, relative in stage4_source_paths.items()
    }
    stage4_material = {
        "schema_version": "thewiz.corrective_stage4_handoff_readiness.v4",
        "status": "PASS_STAGE4_HANDOFF_READY",
        "handoff_state": "READY_AWAITING_STAGE3_THEN_ACTIVE_CONTRACT",
        "checked_at_utc": "2026-08-11T12:00:00+00:00",
        "checks_passed": 12,
        "checks_total": 12,
        "active_contract_id": stage4_contract_id,
        "stage3_manifest_immutable_path": stage4_manifest_path,
        "stage3_reset_receipt_id": reset_receipt_id,
        "stage3_reset_state_sha256": wizard_reset_readiness_state_sha256(reset_receipt),
        "stage5_protocol_receipt_path": stage4_protocol_path,
        "source_artifacts": stage4_source_artifacts,
        "source_closure_sha256": sha256(
            json.dumps(
                stage4_source_artifacts,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
        "blockers": [],
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    stage4_receipt_id = (
        "stage4handoff_"
        + sha256(
            json.dumps(stage4_material, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()[:20]
    )
    stage4_receipt = {**stage4_material, "receipt_id": stage4_receipt_id}
    (scheduler.parent / "stage4_handoff_readiness.json").write_text(
        json.dumps(stage4_receipt), encoding="utf-8"
    )
    immutable_stage4 = (
        tmp_path / "data" / "research" / "stage4_handoff_readiness" / f"{stage4_receipt_id}.json"
    )
    immutable_stage4.parent.mkdir(parents=True)
    immutable_stage4.write_text(json.dumps(stage4_receipt), encoding="utf-8")
    (scheduler.parent / "corrective_wizard_capture_reconciliation.json").write_text(
        json.dumps(
            {
                "status": "PENDING",
                "required_calls": 13,
                "completed_calls": 0,
                "reconciliation_id": "",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "registered_learning_research_status.json").write_text(
        json.dumps(
            {
                "status": "PASS_RESEARCH_LEARNING_GATES",
                "registered_execution_id": "registered-execution-1",
                "stage5_research_gate_pass": True,
                "blocker": "",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_dynamic_v2_holdout_status.json").write_text(
        json.dumps(
            {
                "status": "WAITING_FOR_HOLDOUT",
                "required_cells": 4,
                "passed_cells": 0,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_dynamic_v2_supersession_gate.json").write_text(
        json.dumps({"status": "BLOCKED"}), encoding="utf-8"
    )
    (scheduler.parent / "wizard_dynamic_v2_activation_status.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED",
                "comparator_generation": 2,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_dynamic_v2_proof_refresh_status.json").write_text(
        json.dumps(
            {
                "status": "NOT_ACTIVE",
                "captured_dynamic_rows": 0,
                "exact_dynamic_rows": 0,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_dynamic_v2_review_packet.json").write_text(
        json.dumps(
            {
                "status": "WAITING_FOR_HOLDOUT",
                "review_packet_id": "",
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_ou_trend_selector_v1_receipt.json").write_text(
        json.dumps(
            {
                "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
                "derivation_cells": 8,
                "derivation_matched_cells": 8,
                "holdout_prediction_cells": 4,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_ou_v3_capture_status.json").write_text(
        json.dumps(
            {
                "status": "PLANNED",
                "responses_available": 0,
                "required_responses": 4,
                "evaluation_status": "WAITING_VENDOR_RESPONSES",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (scheduler.parent / "wizard_copula_behavioral_status.json").write_text(
        json.dumps(
            {
                "status": "WAITING_FOR_COPULA_BACKTEST_RESPONSES",
                "expected_cells": 4,
                "behavioral_cells_passed": 0,
                "provenance_cells_complete": 0,
                "behavioral_parity_proven": False,
                "formula_parity_proven": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    csv_path, md_path = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    frame = pd.read_csv(csv_path)

    assert len(frame) == 7
    assert not frame["testnet_order_authority"].astype(bool).any()
    assert not frame["live_trading_authorized"].astype(bool).any()
    stage_one_progress = frame.loc[frame["stage"].eq(1), "evidence_progress"].iloc[0]
    assert stage_one_progress.startswith("1/7;")
    assert "wizard_semantic_receipt_required_from=2026-08-11" in stage_one_progress
    assert "full_sweep_raw_and_credit_binding_required=True" in stage_one_progress
    assert "scheduler_runtime=PASS_SCHEDULER_RUNTIME_READY" in stage_one_progress
    assert "scheduler_agents=3/3" in stage_one_progress
    assert "scheduler_runtime_receipt=schedulerruntime_test" in stage_one_progress
    stage_two_progress = frame.loc[frame["stage"].eq(2), "evidence_progress"].iloc[0]
    assert "mapping_maintenance=NOT_DUE" in stage_two_progress
    assert "mapping_refresh_due=False" in stage_two_progress
    assert "mapping_age_hours=1.5" in stage_two_progress
    assert "mapping_refresh_id=hlmap_test" in stage_two_progress
    assert "mapping_ready_pair_groups=26/129" in stage_two_progress
    assert "mapping_maintenance_warnings=0" in stage_two_progress
    assert "scheduler_runtime=PASS_SCHEDULER_RUNTIME_READY" in stage_two_progress
    assert "l2_readiness_refresh=WAITING_STRICT_L2" in stage_two_progress
    assert "l2_readiness_validation=PASS" in stage_two_progress
    assert "l2_readiness_receipt=l2readiness_test" in stage_two_progress
    assert "l2_readiness_pairs=1/4" in stage_two_progress
    assert "l2_readiness_gate_executed=False" in stage_two_progress
    assert "l2_readiness_handoff_executed=False" in stage_two_progress
    stage_three = frame.loc[frame["stage"].eq(3)].iloc[0]
    assert stage_three["status"] == "IN_PROGRESS"
    assert "vendor_responses=0/28" in stage_three["evidence_progress"]
    assert "formula_proofs=0/24" in stage_three["evidence_progress"]
    assert "accepted_mode_evidence=0/28" in stage_three["evidence_progress"]
    assert "proof_launcher=DEFERRED_SAME_UTC_DAY_LIGHTWEIGHT" in stage_three["evidence_progress"]
    assert "proof_launcher_same_day_state=ATTEMPTED" in stage_three["evidence_progress"]
    assert "proof_launcher_heavy_invoked=False" in stage_three["evidence_progress"]
    assert "proof_launcher_status_manifest_bound=True" in stage_three["evidence_progress"]
    assert "proof_launcher_immutable_manifest_bound=False" in stage_three["evidence_progress"]
    assert "proof_launcher_status_stage3_complete=False" in stage_three["evidence_progress"]
    assert "proof_launcher_immutable_stage3_complete=False" in stage_three["evidence_progress"]
    assert (
        "final_immutable_execution=BLOCKED_FINAL_IMMUTABLE_EXECUTION"
        in stage_three["evidence_progress"]
    )
    assert "final_immutable_execution_receipt=" in stage_three["evidence_progress"]
    assert "final_immutable_execution_stage3_complete=False" in stage_three["evidence_progress"]
    assert "proof_input_audit=28/28" in stage_three["evidence_progress"]
    assert "credit_budget=PASS" in stage_three["evidence_progress"]
    assert "proof_lane_credit_ceiling=68" in stage_three["evidence_progress"]
    assert "capture_manifest=PASS" in stage_three["evidence_progress"]
    assert "capture_manifest_calls=13" in stage_three["evidence_progress"]
    assert "capture_manifest_credits=18" in stage_three["evidence_progress"]
    assert "reset_automation=PASS_RESET_AUTOMATION_READY" in stage_three["evidence_progress"]
    assert "reset_automation_checks=12/12" in stage_three["evidence_progress"]
    assert "reset_automation_launch_agent_loaded=True" in stage_three["evidence_progress"]
    assert "scheduler_runtime=PASS_SCHEDULER_RUNTIME_READY" in stage_three["evidence_progress"]
    assert "scheduler_agents=3/3" in stage_three["evidence_progress"]
    assert "scheduler_runtime_receipt=schedulerruntime_test" in stage_three["evidence_progress"]
    assert (
        "comparator_review_control=WAITING_FOR_PROSPECTIVE_EVIDENCE"
        in (stage_three["evidence_progress"])
    )
    assert "comparator_review_ready=0/2" in stage_three["evidence_progress"]
    assert "comparator_apply_ready=0/2" in stage_three["evidence_progress"]
    assert "comparator_applied=0/2" in stage_three["evidence_progress"]
    assert "comparator_review_receipt=comparatorreview_test" in stage_three["evidence_progress"]
    assert "capture_manifest_continuity_valid=True" in stage_three["evidence_progress"]
    assert "capture_reconciliation=PENDING" in stage_three["evidence_progress"]
    assert "capture_reconciled_calls=0/13" in stage_three["evidence_progress"]
    assert "credit_reservation=PASS" in stage_three["evidence_progress"]
    assert "credit_reconciliation=PASS_RECONCILED" in stage_three["evidence_progress"]
    assert "next_cohort_remaining=28" in stage_three["evidence_progress"]
    assert "next_cohort_capacity=30" in stage_three["evidence_progress"]
    assert "next_cohort_required_credits=56" in stage_three["evidence_progress"]
    assert (
        "next_cohort_readiness=CAPACITY_READY_RUNTIME_PREFLIGHT_REQUIRED"
        in stage_three["evidence_progress"]
    )
    assert "next_utc_reset_at=2026-08-11T00:00:00+00:00" in stage_three["evidence_progress"]
    assert (
        "next_external_attempt_eligible_at=2026-08-11T00:00:00+00:00"
        in stage_three["evidence_progress"]
    )
    assert "dynamic_v2_holdout=WAITING_FOR_HOLDOUT" in stage_three["evidence_progress"]
    assert "dynamic_v2_cells=0/4" in stage_three["evidence_progress"]
    assert "dynamic_v2_supersession=BLOCKED" in stage_three["evidence_progress"]
    assert "dynamic_v2_activation=BLOCKED" in stage_three["evidence_progress"]
    assert "dynamic_v2_generation=1" in stage_three["evidence_progress"]
    assert "dynamic_v2_proof_refresh=NOT_ACTIVE" in stage_three["evidence_progress"]
    assert "dynamic_v2_exact_rows=0/0" in stage_three["evidence_progress"]
    assert "dynamic_v2_review_packet=WAITING_FOR_HOLDOUT" in stage_three["evidence_progress"]
    assert "dynamic_v2_review_packet_id=" in stage_three["evidence_progress"]
    assert (
        "ou_trend_selector=REGISTERED_WAITING_VENDOR_RESPONSES" in stage_three["evidence_progress"]
    )
    assert "ou_selector_derivation=8/8" in stage_three["evidence_progress"]
    assert "ou_selector_predictions=4" in stage_three["evidence_progress"]
    assert "ou_v3_capture=PLANNED" in stage_three["evidence_progress"]
    assert "ou_v3_responses=0/4" in stage_three["evidence_progress"]
    assert "ou_v3_evaluation=WAITING_VENDOR_RESPONSES" in stage_three["evidence_progress"]
    assert "ou_v3_local_selector_proven=False" in stage_three["evidence_progress"]
    assert (
        "copula_behavioral=WAITING_FOR_COPULA_BACKTEST_RESPONSES"
        in stage_three["evidence_progress"]
    )
    assert "copula_behavioral_cells=0/4" in stage_three["evidence_progress"]
    assert "copula_provenance_cells=0/4" in stage_three["evidence_progress"]
    assert "copula_behavioral_parity=False" in stage_three["evidence_progress"]
    assert "copula_formula_parity=False" in stage_three["evidence_progress"]
    assert "wizard_dynamic_v2_holdout_status.json" in stage_three["evidence_path"]
    assert "wizard_dynamic_v2_activation_status.json" in stage_three["evidence_path"]
    assert "wizard_dynamic_v2_proof_refresh_status.json" in stage_three["evidence_path"]
    assert "wizard_dynamic_v2_review_packet.json" in stage_three["evidence_path"]
    assert "wizard_ou_trend_selector_v1_receipt.json" in stage_three["evidence_path"]
    assert "wizard_ou_v3_capture_status.json" in stage_three["evidence_path"]
    assert "wizard_copula_behavioral_status.json" in stage_three["evidence_path"]
    assert "wizard_comparator_review_control.json" in stage_three["evidence_path"]
    assert "scheduler_runtime_readiness.json" in stage_three["evidence_path"]
    assert "data/research/wizard_proof_scheduler_receipts" in stage_three["evidence_path"]
    assert stage_three["next_action"] == ("collect_and_evaluate_remaining_comparator_holdout_cells")
    stage_four = frame.loc[frame["stage"].eq(4)].iloc[0]
    assert "stage4_handoff_readiness=PASS_STAGE4_HANDOFF_READY" in stage_four["evidence_progress"]
    assert (
        "stage4_handoff_state=READY_AWAITING_STAGE3_THEN_ACTIVE_CONTRACT"
        in stage_four["evidence_progress"]
    )
    assert "stage4_handoff_checks=12/12" in stage_four["evidence_progress"]
    assert f"stage4_handoff_receipt_id={stage4_receipt_id}" in stage_four["evidence_progress"]
    assert "stage4_handoff_checked_at=2026-08-11T12:00:00+00:00" in stage_four["evidence_progress"]
    assert "stage4_handoff_readiness.json" in stage_four["evidence_path"]
    stage_five = frame.loc[frame["stage"].eq(5)].iloc[0]
    assert stage_five["status"] == "BLOCKED"
    assert (
        "registered_learning_audit=BLOCKED_REGISTERED_LEARNING_EVIDENCE"
        in stage_five["evidence_progress"]
    )
    assert "registered_learning_evidence_valid=False" in stage_five["evidence_progress"]
    assert "stage5_research_gate_pass=False" in stage_five["evidence_progress"]
    assert stage_five["next_action"] == "register_stage5_protocol_before_stage4_completion"
    stage_six = frame.loc[frame["stage"].eq(6)].iloc[0]
    assert stage_six["status"] == "BLOCKED"
    assert "wallet_agent_preflight=False" in stage_six["evidence_progress"]
    assert "sample_artifact_pass=False" in stage_six["evidence_progress"]
    assert "hyperliquid_testnet_preflight.csv" in stage_six["evidence_path"]
    assert "testnet_lifecycle_execution_receipt.json" in stage_six["evidence_path"]
    assert md_path.exists()

    phases["wizard_parity"] = result(status="PASS", blocker="")
    scheduler_payload = json.loads(scheduler.read_text(encoding="utf-8"))
    scheduler.write_text(
        json.dumps(
            {
                **scheduler_payload,
                "status": "COMPLETE_ACCEPTED_MODE_EVIDENCE",
                "queue_eligible": 13,
                "completed_after": 13,
                "responses_captured_after": 13,
            }
        ),
        encoding="utf-8",
    )
    reconciliation_path = scheduler.parent / "corrective_wizard_capture_reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "required_calls": 13,
                "completed_calls": 13,
                "pending_calls": 0,
                "blocked_calls": 0,
                "reconciliation_id": "reconciliation_test",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    no_final_csv, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    no_final_stage_three = pd.read_csv(no_final_csv).loc[lambda frame: frame["stage"].eq(3)].iloc[0]
    assert no_final_stage_three["status"] == "BLOCKED"
    assert no_final_stage_three["next_action"] == (
        "repair_final_immutable_scheduler_execution_receipt"
    )

    immutable_receipt_id = _write_verified_wizard_execution(tmp_path)
    final_csv, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    final_stage_three = pd.read_csv(final_csv).loc[lambda frame: frame["stage"].eq(3)].iloc[0]
    assert final_stage_three["status"] == "PASS"
    assert (
        "final_immutable_execution=PASS_FINAL_IMMUTABLE_EXECUTION"
        in (final_stage_three["evidence_progress"])
    )
    assert (
        f"final_immutable_execution_receipt={immutable_receipt_id}"
        in (final_stage_three["evidence_progress"])
    )
    assert (
        "final_immutable_execution_stage3_complete=True" in (final_stage_three["evidence_progress"])
    )
    assert not bool(final_stage_three["testnet_order_authority"])
    assert not bool(final_stage_three["live_trading_authorized"])

    phases["wizard_parity"] = result(status="BLOCKED", blocker="parity_missing")
    reconciliation_path.write_text(
        json.dumps(
            {
                "status": "PENDING",
                "required_calls": 13,
                "completed_calls": 0,
                "reconciliation_id": "",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    scheduler_runtime_path.write_text(
        json.dumps(
            {
                **scheduler_runtime_pass,
                "status": "BLOCKED_SCHEDULER_RUNTIME",
                "agents_ready": 2,
                "checks_passed": 48,
                "blockers": ["wizard_proof_live_environment_mismatch"],
            }
        ),
        encoding="utf-8",
    )
    runtime_blocked_csv, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    runtime_blocked_frame = pd.read_csv(runtime_blocked_csv)
    runtime_blocked_stage_one = runtime_blocked_frame.loc[
        runtime_blocked_frame["stage"].eq(1)
    ].iloc[0]
    runtime_blocked_stage_three = runtime_blocked_frame.loc[
        runtime_blocked_frame["stage"].eq(3)
    ].iloc[0]
    assert runtime_blocked_stage_one["next_action"] == ("repair_scheduler_runtime_contract")
    assert runtime_blocked_stage_three["status"] == "BLOCKED"
    assert "wizard_proof_live_environment_mismatch" in runtime_blocked_stage_three["blocker"]
    assert runtime_blocked_stage_three["next_action"] == (
        "repair_scheduler_runtime_contract_before_external_calls"
    )
    scheduler_runtime_path.write_text(
        json.dumps(scheduler_runtime_pass),
        encoding="utf-8",
    )

    (scheduler.parent / "wizard_comparator_review_control.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED_REVIEW_CONTROL",
                "comparators": 2,
                "review_ready": 0,
                "apply_ready": 0,
                "applied": 0,
                "receipt_id": "comparatorreview_blocked",
                "control_blockers": ["dynamic_v2_supreme_packet_binding_invalid"],
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    control_blocked_csv, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    control_blocked_frame = pd.read_csv(control_blocked_csv)
    control_blocked_stage_three = control_blocked_frame.loc[
        control_blocked_frame["stage"].eq(3)
    ].iloc[0]
    assert "wizard_comparator_review_control_blocked" in control_blocked_stage_three["blocker"]
    assert control_blocked_stage_three["next_action"] == (
        "repair_comparator_review_evidence_bindings"
    )

    (scheduler.parent / "stage4_handoff_readiness.json").write_text(
        json.dumps(
            {
                "status": "BLOCKED_STAGE4_HANDOFF",
                "handoff_state": "READY_AWAITING_STAGE3",
                "receipt_id": "stage4handoff_blocked",
                "checked_at_utc": "2026-08-11T12:05:00+00:00",
                "checks_passed": 11,
                "checks_total": 12,
                "blockers": ["contract_binding_invalid"],
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    blocked_csv, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    blocked_frame = pd.read_csv(blocked_csv)
    blocked_stage_four = blocked_frame.loc[blocked_frame["stage"].eq(4)].iloc[0]
    assert (
        "stage4_handoff_readiness_failed:stage4_handoff_schema_version_invalid"
        in (blocked_stage_four["blocker"])
    )


def test_seven_stage_checkpoint_accepts_current_stage_six_and_completed_canary_artifacts(
    tmp_path,
):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "testnet_candidate_receipt.json").write_text(
        json.dumps(
            {
                "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
                "blockers": [],
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "testnet_lifecycle_execution_receipt.json").write_text(
        json.dumps(
            {
                "lifecycle_status": "COMPLETE_RECONCILED",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "testnet_collateral_transfer_preflight.json").write_text(
        json.dumps(
            {
                "status": "NO_TRANSFER_REQUIRED",
                "agent_key_accessed": False,
                "transfer_attempted": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "tradable_perp": True,
                "fetch_blocker": "",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_market_inventory.csv", index=False)
    pd.DataFrame(
        [
            {
                "ready_for_no_order_preflight": True,
                "ready_for_testnet_submit": False,
                "order_submission_performed": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_preflight.csv", index=False)
    pd.DataFrame(
        [
            {
                "status": "READY",
                "account_value_usd": 25.0,
                "withdrawable_usd": 25.0,
                "spot_usdc_usd": 970.0,
                "blockers": "",
            }
        ]
    ).to_csv(active / "hyperliquid_testnet_margin_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {"check": "configuration", "status": "PASS"},
            {"check": "candidate_binding", "status": "PASS"},
        ]
    ).to_csv(active / "testnet_no_order_preflight.csv", index=False)
    pd.DataFrame(
        [
            {
                "check": "closed_paired_lifecycles",
                "status": "PASS",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
            {
                "check": "after_cost_mean_lcb_95_usd",
                "status": "PASS",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        ]
    ).to_csv(active / "realized_testnet_sample_sufficiency.csv", index=False)
    authorization_path = active / "live_canary_authorization.json"
    authorization_path.write_text(
        json.dumps(
            {
                "authorization_status": "AUTHORIZED_FOR_ONE_LIVE_CANARY",
                "manual_executor_available": True,
                "user_authorization_present": True,
                "authorization_reusable": False,
                "canary_execution_authority": True,
                "live_trading_authorized": False,
                "blockers": [],
            }
        ),
        encoding="utf-8",
    )
    (active / "live_canary_execution_receipt.json").write_text(
        json.dumps({"execution_id": "livecanaryexec_test"}), encoding="utf-8"
    )
    review_path = tmp_path / "reports" / "supreme_team" / "live_canary_post_canary_review.json"
    review_path.parent.mkdir(parents=True)
    review_path.write_text(json.dumps({"review_status": "PASS"}), encoding="utf-8")
    outcome = {
        "schema_version": "thewiz.live_canary_outcome.v4",
        "generated_at_utc": "2026-08-11T12:00:00+00:00",
        "canary_status": "PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY",
        "authorization_receipt_sha256": sha256(authorization_path.read_bytes()).hexdigest(),
        "execution_id": "livecanaryexec_test",
        "execution_receipt_path": "reports/active/live_canary_execution_receipt.json",
        "execution_blockers": [],
        "post_canary_review_path": ("reports/supreme_team/live_canary_post_canary_review.json"),
        "post_canary_review_blockers": [],
        "orders_submitted": 4,
        "fills_observed": 4,
        "reconciled_flat": True,
        "repeat_authorized": False,
        "scaling_authorized": False,
        "leverage_authorized": False,
        "new_explicit_authorization_required": True,
        "canary_execution_authority": False,
        "live_trading_authorized": False,
    }
    outcome["receipt_sha256"] = sha256(
        json.dumps(
            outcome,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    (active / "live_canary_outcome_evaluation.json").write_text(
        json.dumps(outcome), encoding="utf-8"
    )
    phases = {
        "venue_data_costs": result(),
        "wizard_parity": result(),
        "statistical_remediation": result(),
        "daily_cadence": result(),
        "agent_learning_governance": result(),
        "conditional_release_gates": result(
            testnet_sample_status="PASS",
            testnet_sample_sufficient=False,
        ),
    }

    csv_path, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    stage_six = pd.read_csv(csv_path).loc[lambda frame: frame["stage"].eq(6)].iloc[0]

    assert stage_six["status"] == "PASS"
    assert "sample_release_status=PASS" in stage_six["evidence_progress"]
    assert "sample_artifact_pass=True" in stage_six["evidence_progress"]
    assert "collateral_preflight=NO_TRANSFER_REQUIRED" in stage_six["evidence_progress"]
    assert stage_six["next_action"] == "stage6_complete_await_stage7_policy_gate"
    assert pd.isna(stage_six["blocker"])
    stage_seven = pd.read_csv(csv_path).loc[lambda frame: frame["stage"].eq(7)].iloc[0]
    assert stage_seven["status"] == "PASS"
    assert (
        "outcome=PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY" in stage_seven["evidence_progress"]
    )
    assert "ongoing_live_authority=False" in stage_seven["evidence_progress"]
    assert stage_seven["next_action"] == "one_live_canary_complete_no_further_authority"
    assert pd.isna(stage_seven["blocker"])


def test_stage_six_cannot_pass_from_sample_artifact_alone(tmp_path):
    def result(**summary):
        return CommandResult(paths={}, summary=summary)

    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "check": "closed_paired_lifecycles",
                "status": "PASS",
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ]
    ).to_csv(active / "realized_testnet_sample_sufficiency.csv", index=False)
    phases = {
        "venue_data_costs": result(),
        "wizard_parity": result(),
        "statistical_remediation": result(),
        "daily_cadence": result(),
        "agent_learning_governance": result(),
        "conditional_release_gates": result(testnet_sample_status="PASS"),
    }

    csv_path, _ = _write_seven_stage_checkpoint(
        root=tmp_path,
        phases=phases,
        completion={"live_trading_authorized": False},
    )
    frame = pd.read_csv(csv_path)
    stage_six = frame.loc[frame["stage"].eq(6)].iloc[0]
    stage_seven = frame.loc[frame["stage"].eq(7)].iloc[0]

    assert stage_six["status"] == "BLOCKED"
    assert "sample_artifact_pass=True" in stage_six["evidence_progress"]
    assert "testnet_candidate:MISSING" in stage_six["blocker"]
    assert "testnet_wallet_agent_preflight_not_ready" in stage_six["blocker"]
    assert "testnet_no_order_preflight_missing_or_empty" in stage_six["blocker"]
    assert "testnet_lifecycle:MISSING" in stage_six["blocker"]
    assert stage_seven["status"] == "BLOCKED"


def test_registered_rerun_task_is_not_complete_when_only_ready():
    assert _registered_rerun_task_status(
        {
            "registered_rerun_gate_status": "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN",
            "registered_rerun_results_accounted": False,
        }
    ) == (
        "in_progress_evidence_collection",
        "registered_full_family_rerun_ready_awaiting_post_ready_chain",
    )


def test_stage_four_does_not_pass_on_one_contract_survivor_with_pending_cohort(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    contract = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        contract,
        {"hypothesis-1": "ACCEPTED_SURVIVOR"},
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=1,
        independent_clusters=3,
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {"semantic_hypothesis_id": ["hypothesis-2", "hypothesis-3"]}
        ),
        statistics={
            "final_one_x_survivors": 1,
            "independent_supporting_clusters": 3,
            "pending_family_preflight_status": "PASS",
        },
    )

    assert state["complete"] is False
    assert state["active_contract_conclusive"] is True
    assert state["target_hypotheses"] == 3
    assert state["accounted_hypotheses"] == 1
    assert state["unaccounted_hypotheses"] == 2
    assert state["next_action"].startswith("advance_registered_rerun")

    blocked = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {"semantic_hypothesis_id": ["hypothesis-2", "hypothesis-3"]}
        ),
        statistics={
            "final_one_x_survivors": 1,
            "independent_supporting_clusters": 3,
            "pending_family_preflight_status": "BLOCKED",
        },
    )
    assert blocked["next_action"] == ("repair_pending_family_non_vendor_preflight_before_rollover")


def test_stage_four_collapses_only_identity_consistent_duplicate_experiment_rows(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    contract = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        contract,
        {"hypothesis-1": "REJECTED_BY_FROZEN_GATES"},
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=0,
    )
    common = {
        "semantic_hypothesis_id": "hypothesis-1",
        "equivalence_cluster_id": "cluster-1",
        "pair": "ETH-USD-PYTH-USD",
        "wizard_timeframe": "daily",
        "exact_mode": "Copula",
        "orientation": "reverse",
    }
    consistent = pd.DataFrame(
        [
            {**common, "experiment_id": "experiment-1"},
            {**common, "experiment_id": "experiment-2"},
        ]
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=consistent,
        statistics={"final_one_x_survivors": 0},
    )

    assert state["complete"] is True
    assert state["target_hypotheses"] == 1
    assert "pending_current_family_semantic_ids_invalid_or_duplicated" not in state["blocker"]

    conflicting = consistent.copy()
    conflicting.loc[1, "pair"] = "ETH-USD-NEAR-USD"
    blocked = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=conflicting,
        statistics={"final_one_x_survivors": 0},
    )

    assert blocked["complete"] is False
    assert "pending_current_family_semantic_ids_invalid_or_duplicated" in blocked["blocker"]


def test_stage_four_does_not_pass_on_generation_one_zero_with_pending_cohort(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    contract = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        contract,
        {"hypothesis-1": "REJECTED_BY_FROZEN_GATES"},
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=0,
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {"semantic_hypothesis_id": ["hypothesis-2", "hypothesis-3"]}
        ),
        statistics={
            "final_one_x_survivors": 0,
            "independent_supporting_clusters": 0,
            "pending_family_preflight_status": "PASS",
        },
    )

    assert state["complete"] is False
    assert state["terminal_outcome"] == "INCOMPLETE_CURRENT_FAMILY"
    assert state["unaccounted_hypotheses"] == 2
    assert state["next_action"].startswith("advance_registered_rerun")


def test_stage_four_passes_only_after_generation_chain_and_breadth_are_complete(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    first = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
    )
    _write_stage_four_conclusion(
        tmp_path,
        first,
        {"hypothesis-1": "REJECTED_BY_FROZEN_GATES"},
    )
    second = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-2", "hypothesis-3", "hypothesis-4"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=2,
        prior_contract_id=first["contract_id"],
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        second,
        {
            "hypothesis-2": "ACCEPTED_SURVIVOR",
            "hypothesis-3": "ACCEPTED_SURVIVOR",
            "hypothesis-4": "ACCEPTED_SURVIVOR",
        },
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=3,
        independent_clusters=3,
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {
                "semantic_hypothesis_id": [
                    "hypothesis-2",
                    "hypothesis-3",
                    "hypothesis-4",
                ]
            }
        ),
        statistics={
            "final_one_x_survivors": 3,
            "independent_supporting_clusters": 3,
            "independent_full_survivor_clusters": 3,
            "independent_supporting_pairs": 3,
            "independent_full_survivor_pairs": 3,
        },
    )

    assert state["complete"] is True
    assert state["terminal_outcome"] == "ACCEPTED_VALID_INDEPENDENT_SURVIVORS"
    assert state["contract_generations"] == 2
    assert state["target_hypotheses"] == 4
    assert state["accounted_hypotheses"] == 4
    assert state["unaccounted_hypotheses"] == 0


def test_stage_four_accepts_modern_selection_bound_contract_identity():
    material = {
        "schema_version": "thewiz.corrective_registered_rerun.v1",
        "registered_candidates": [
            {
                "semantic_hypothesis_id": "hypothesis-modern",
                "source_experiment_id": "experiment-modern",
            }
        ],
        "acceptance_policy_id": "acceptance-modern",
        "holdout_policy_id": "holdout-modern",
        "discovery_policy_sha256": "a" * 64,
        "source_family_sha256": "b" * 64,
        "source_family_rows": 3264,
        "candidate_selection_path": "reports/active/registered_rerun_family_preflight.csv",
        "candidate_selection_sha256": "c" * 64,
        "candidate_selection_status": "PASS",
        "generation": 2,
        "prior_contract_id": "registeredrerun_prior",
    }
    contract = {
        **material,
        "contract_id": "registeredrerun_"
        + sha256(
            json.dumps(
                material,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20],
        "full_family_multiplicity_required": True,
        "promotion_evaluation_registered_only": True,
        "threshold_changes_after_contract_permitted": False,
        "promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }

    assert _registered_contract_identity_valid(
        contract,
        acceptance_policy_id="acceptance-modern",
        holdout_policy_id="holdout-modern",
    )
    assert not _registered_contract_identity_valid(
        {**contract, "candidate_selection_path": "reports/active/tampered.csv"},
        acceptance_policy_id="acceptance-modern",
        holdout_policy_id="holdout-modern",
    )


def test_stage_four_can_conclusively_reject_only_after_full_generation_accounting(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    first = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
    )
    _write_stage_four_conclusion(
        tmp_path,
        first,
        {"hypothesis-1": "REJECTED_BY_FROZEN_GATES"},
    )
    second = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-2", "hypothesis-3"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=2,
        prior_contract_id=first["contract_id"],
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        second,
        {
            "hypothesis-2": "ACCEPTED_SURVIVOR",
            "hypothesis-3": "ACCEPTED_SURVIVOR",
        },
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=0,
        independent_clusters=2,
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {"semantic_hypothesis_id": ["hypothesis-2", "hypothesis-3"]}
        ),
        statistics={
            "final_one_x_survivors": 0,
            "independent_supporting_clusters": 2,
            "independent_full_survivor_clusters": 0,
            "independent_supporting_pairs": 2,
            "independent_full_survivor_pairs": 0,
        },
    )

    assert state["complete"] is True
    assert state["terminal_outcome"] == "CONCLUSIVE_REJECTION_CURRENT_FAMILY"
    assert state["target_hypotheses"] == 3
    assert state["accounted_hypotheses"] == 3


def test_stage_four_conclusively_rejects_fully_accounted_under_breadth_family(
    tmp_path,
):
    acceptance_id, holdout_id = _write_stage_four_policy_lineage(tmp_path)
    contract = _write_stage_four_contract(
        tmp_path,
        ["hypothesis-1", "hypothesis-2"],
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        generation=1,
        active=True,
    )
    _write_stage_four_conclusion(
        tmp_path,
        contract,
        {
            "hypothesis-1": "ACCEPTED_SURVIVOR",
            "hypothesis-2": "ACCEPTED_SURVIVOR",
        },
    )
    _write_stage_four_final_receipt(
        tmp_path,
        acceptance_id=acceptance_id,
        holdout_id=holdout_id,
        final_survivors=2,
        independent_clusters=2,
    )

    state = _stage_four_completion_state(
        root=tmp_path,
        pending_candidate_cohort=pd.DataFrame(
            {"semantic_hypothesis_id": ["hypothesis-1", "hypothesis-2"]}
        ),
        statistics={
            "final_one_x_survivors": 2,
            "independent_supporting_clusters": 2,
            "independent_full_survivor_clusters": 2,
            "independent_supporting_pairs": 2,
            "independent_full_survivor_pairs": 2,
        },
    )

    assert state["complete"] is True
    assert state["terminal_outcome"] == "CONCLUSIVE_REJECTION_CURRENT_FAMILY"
    assert state["target_hypotheses"] == 2
    assert state["accounted_hypotheses"] == 2
    assert state["unaccounted_hypotheses"] == 0
    assert state["next_action"] == (
        "close_current_family_as_conclusively_rejected_and_refresh_discovery"
    )
    assert "independent_strategy_breadth_2_of_3" in state["blocker"]


def test_registered_rerun_task_accepts_only_hash_valid_execution_receipt(
    tmp_path,
):
    receipt = {
        "status": "PASS_REGISTERED_RERUN_ACCOUNTED",
        "conclusion_status": "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt["receipt_sha256"] = sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / "data" / "research" / "execution.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    execution = {
        "execution_receipt_path": str(path.relative_to(tmp_path)),
        "execution_receipt_sha256": sha256(path.read_bytes()).hexdigest(),
        "conclusion_status": "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
    }
    assert (
        _registered_rerun_task_status(
            {"registered_rerun_gate_status": "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"},
            execution=execution,
            root=tmp_path,
        )[0]
        == "completed"
    )
    path.write_text("{}", encoding="utf-8")
    assert (
        _registered_rerun_task_status(
            {"registered_rerun_gate_status": "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"},
            execution=execution,
            root=tmp_path,
        )[0]
        == "in_progress_evidence_collection"
    )
    assert (
        _registered_rerun_task_status(
            {
                "registered_rerun_gate_status": "PASS_REGISTERED_RERUN_ACCOUNTED",
                "registered_rerun_results_accounted": True,
                "registered_rerun_conclusion_status": ("CONCLUSIVE_ZERO_REGISTERED_SURVIVORS"),
            }
        )[0]
        == "completed"
    )


def test_strict_cost_action_waits_for_every_pair_before_refresh():
    assert _next_strict_cost_action(
        {
            "strict_cost_ready_pairs": 0,
            "strict_cost_eligible_pairs": 3,
            "post_window_ready_pairs": 3,
            "post_window_eligible_pairs": 3,
        }
    ) == (
        "refresh_wizard_candidate_after_completed_l2_window_then_rematerialize_"
        "point_in_time_cost_evidence"
    )
    assert (
        _next_strict_cost_action(
            {
                "strict_cost_ready_pairs": 1,
                "strict_cost_eligible_pairs": 3,
                "post_window_ready_pairs": 1,
                "post_window_eligible_pairs": 3,
            }
        )
        == "l2_scheduler_collects_every_5_minutes_until_strict_window_is_complete"
    )
    assert _next_strict_cost_action(
        {"strict_cost_ready_pairs": 3, "strict_cost_eligible_pairs": 3}
    ).startswith("maintain_rolling_l2")


def test_strict_cost_task_completes_only_when_every_registered_pair_is_ready():
    assert _strict_cost_task_status(
        {"strict_cost_ready_pairs": 2, "strict_cost_eligible_pairs": 3}
    ) == ("blocked_by_evidence", "strict_pair_cost_models_2_of_3")
    assert _strict_cost_task_status(
        {"strict_cost_ready_pairs": 3, "strict_cost_eligible_pairs": 3}
    ) == ("completed", "strict_pair_cost_models_3_of_3")


def test_wizard_proof_action_is_driven_by_scheduler_and_queue_state():
    assert (
        _next_wizard_proof_action(
            scheduler_status="DEFERRED_CREDIT_RESET",
            queue_completed=0,
            queue_eligible=28,
        )
        == "run_bounded_exact_mode_proofs_after_0000_utc_credit_reset"
    )
    assert (
        _next_wizard_proof_action(
            scheduler_status="DEFERRED_SAME_UTC_DAY",
            queue_completed=4,
            queue_eligible=28,
            responses_captured=8,
        )
        == "wait_for_next_utc_credit_reset_then_resume_bounded_exact_mode_proofs"
    )
    assert _next_wizard_proof_action(
        scheduler_status="COMPLETE_QUEUE",
        queue_completed=28,
        queue_eligible=28,
    ).startswith("refresh_formula_parity")
    assert _next_wizard_proof_action(
        scheduler_status="COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE",
        queue_completed=4,
        queue_eligible=28,
        responses_captured=28,
    ).startswith("implement_mode_specific_comparators")
