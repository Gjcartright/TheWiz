"""Run every non-order corrective phase and publish one truthful program status."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_agent_governance import (
    build_corrective_agent_governance,
)
from quant_platform.orchestration.corrective_daily_scheduler import (
    SEMANTIC_DAILY_RECEIPT_REQUIRED_FROM_UTC,
    _acquire_lock,
    build_corrective_daily_cadence,
)
from quant_platform.orchestration.corrective_data_evidence import (
    build_corrective_data_evidence,
    build_l2_capture_candidate_set,
)
from quant_platform.orchestration.corrective_governance import build_corrective_governance
from quant_platform.orchestration.corrective_l2_scheduler import (
    build_post_window_transition,
)
from quant_platform.orchestration.corrective_registered_learning import (
    latest_verified_registered_learning,
)
from quant_platform.orchestration.corrective_registered_rerun import (
    contract_conclusively_accounted,
    validate_registered_rerun_contract_identity,
)
from quant_platform.orchestration.corrective_release_gates import build_corrective_release_gates
from quant_platform.orchestration.corrective_scheduler_runtime_readiness import (
    build_corrective_scheduler_runtime_readiness,
)
from quant_platform.orchestration.corrective_stage4_handoff_readiness import (
    validate_stage4_handoff_readiness_receipt,
)
from quant_platform.orchestration.corrective_statistical_remediation import (
    build_corrective_statistical_remediation,
)
from quant_platform.orchestration.corrective_testnet_collateral_transfer import (
    build_testnet_collateral_transfer_preflight,
)
from quant_platform.orchestration.corrective_wizard_browser_auth import (
    validate_wizard_browser_auth_readiness,
)
from quant_platform.orchestration.corrective_wizard_comparator_review_control import (
    build_corrective_wizard_comparator_review_control,
)
from quant_platform.orchestration.corrective_wizard_parity import build_corrective_wizard_parity
from quant_platform.orchestration.corrective_wizard_proof_launcher import (
    latest_verified_immutable_scheduler_execution,
)
from quant_platform.orchestration.corrective_wizard_surface_inventory import (
    validate_wizard_surface_inventory_readiness,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_program_completion.v1"


TASK_STATUS = {
    **{f"T{index:02d}": ("completed", "implemented_and_verified") for index in range(1, 12)},
    "T12": (
        "in_progress_evidence_collection",
        "selected_pair_history_remediation_is_evidence_driven",
    ),
    "T13": ("in_progress_evidence_collection", "strict_l2_cadence_collection_is_active"),
    "T14": ("blocked_by_evidence", "zero_pair_cost_models_have_strict_observed_cost_acceptance"),
    "T15": ("completed", "seven_hostile_cost_bundles_fail_closed"),
    "T16": ("completed", "all_pair_page_mode_orientation_cells_accounted"),
    "T17": (
        "in_progress_evidence_collection",
        "ou_optimal_scanner_overlay_provenance_requires_both_orientations",
    ),
    "T18": ("completed", "parity_comparator_and_seven_formula_mutations_verified"),
    "T19": ("completed", "vendor_parity_separated_from_local_approximation"),
    "T20": ("completed", "42_raw_passes_clustered_into_33_effective_clusters"),
    "T21": ("completed", "42_near_misses_have_one_missing_proof_and_next_test"),
    "T22": ("completed", "bounded_hypothesis_batch_registered_before_new_tests"),
    "T23": (
        "blocked_pending_new_evidence",
        "rerun_forbidden_until_new_history_strict_cost_and_parity_evidence_arrive",
    ),
    "T24": ("completed", "independent_breadth_gate_materialized_and_failed_at_one_of_three"),
    "T25": ("completed", "explicit_zero_survivor_receipt_issued"),
    "T26": ("completed", "research_only_launch_agent_loaded_for_0615_local"),
    "T27": ("completed", "seven_scheduler_fault_cases_passed"),
    "T28": ("in_progress_time_observation", "one_of_seven_distinct_calendar_day_cycles_observed"),
    "T29": ("completed", "29_agent_and_learning_authority_edges_inventoried"),
    "T30": ("completed", "eight_forged_agent_packets_fail_closed"),
    "T31": ("completed", "2338_learning_rows_audited_under_label_and_time_contract"),
    "T32": (
        "blocked_pending_model_and_testnet_evidence",
        "model_incremental_edge_and_realized_testnet_sample_not_proven",
    ),
    "T33": ("blocked_by_safety_gate", "valid_final_one_x_survivor_receipt_missing"),
    "T34": ("blocked_by_dependency", "testnet_candidate_not_opened"),
    "T35": ("not_executed_safety_gate", "no_order_preflight_and_explicit_authorization_missing"),
    "T36": ("blocked_by_dependency", "zero_realized_testnet_lifecycles"),
    "T37": (
        "completed_blocked_checkpoint",
        "supreme_team_checkpoint_written_with_realized_evidence_gap",
    ),
    "T38": ("completed", "prospective_live_canary_policy_frozen"),
    "T39": ("blocked_by_dependency", "no_testnet_candidate_for_shadow_live_input_parity"),
    "T40": ("not_authorized", "live_prerequisites_and_exact_order_user_authorization_missing"),
    "T41": ("not_executed_safety_gate", "no_live_canary_was_authorized_or_submitted"),
}

TASK_DEFINITION_OVERRIDES = {
    "T17": {
        "task": (
            "Create golden fixtures for all seven pair-page exact modes and both "
            "orientations, then account for OU Optimal as a scanner overlay"
        ),
        "acceptance_gate": (
            "fixtures cover Copula, Dyn Spread, Dyn ZScoreR, OU Spread, OU ZScoreR, "
            "Static Spread, and Static ZScoreR in both orientations; OU Optimal "
            "ordered-symbol provenance is separately accounted"
        ),
        "stop_condition": (
            "stop if OU Optimal is classified as an independent pair-page mode or "
            "historical overlay provenance is used as a live signal"
        ),
    }
}


def complete_corrective_plan(*, root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    """Publish one cross-stage snapshot while daily and L2 producers are quiescent."""

    now = _as_utc(now)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    lock_paths = [
        active / ".corrective_daily.lock",
        active / ".corrective_l2_capture.lock",
    ]
    acquired: list[Path] = []
    try:
        for lock_path in lock_paths:
            _acquire_lock(lock_path, now=now, timeout_seconds=4 * 60 * 60)
            acquired.append(lock_path)
        return _complete_corrective_plan_unlocked(root=root, now=now)
    finally:
        for lock_path in reversed(acquired):
            lock_path.unlink(missing_ok=True)


def _complete_corrective_plan_unlocked(*, root: Path, now: datetime) -> CommandResult:
    phases = {
        "governance": build_corrective_governance(root=root, now=now),
        "venue_data_costs": build_corrective_data_evidence(root=root, now=now),
        "wizard_parity": build_corrective_wizard_parity(root=root),
        "statistical_remediation": build_corrective_statistical_remediation(root=root, now=now),
        "daily_cadence": build_corrective_daily_cadence(root=root, now=now, install=True),
        "agent_learning_governance": build_corrective_agent_governance(root=root, now=now),
        "conditional_release_gates": build_corrective_release_gates(root=root, now=now),
    }
    collateral_preflight = _refresh_testnet_collateral_preflight(
        root=root,
        now=now,
    )
    phases["conditional_release_gates"].paths.update(
        {f"collateral_{name}": path for name, path in collateral_preflight.paths.items()}
    )
    phases["conditional_release_gates"].summary.update(
        {
            "testnet_collateral_preflight_status": collateral_preflight.summary.get(
                "status", "BLOCKED"
            ),
            "testnet_collateral_candidate_receipt_id": collateral_preflight.summary.get(
                "candidate_receipt_id", ""
            ),
            "testnet_collateral_agent_key_accessed": False,
            "testnet_collateral_transfer_attempted": False,
        }
    )
    scheduler_runtime = build_corrective_scheduler_runtime_readiness(
        root=root,
        now=now,
    )
    phases["daily_cadence"].paths.update(
        {f"scheduler_runtime_{name}": path for name, path in scheduler_runtime.paths.items()}
    )
    phases["daily_cadence"].summary.update(
        {
            "scheduler_runtime_status": scheduler_runtime.summary.get(
                "status", "BLOCKED_SCHEDULER_RUNTIME"
            ),
            "scheduler_runtime_agents_ready": int(
                scheduler_runtime.summary.get("agents_ready", 0) or 0
            ),
            "scheduler_runtime_agents_expected": int(
                scheduler_runtime.summary.get("agents_expected", 3) or 3
            ),
            "scheduler_runtime_receipt_id": str(scheduler_runtime.summary.get("receipt_id", "")),
        }
    )
    comparator_review = build_corrective_wizard_comparator_review_control(
        root=root,
        now=now,
    )
    phases["wizard_parity"].paths.update(
        {f"comparator_review_{name}": path for name, path in comparator_review.paths.items()}
    )
    phases["wizard_parity"].summary.update(
        {
            "comparator_review_control_status": comparator_review.summary.get(
                "status", "BLOCKED_REVIEW_CONTROL"
            ),
            "comparator_review_ready": int(comparator_review.summary.get("review_ready", 0) or 0),
            "comparator_apply_ready": int(comparator_review.summary.get("apply_ready", 0) or 0),
        }
    )
    final_l2_candidates = build_l2_capture_candidate_set(root=root)
    post_window_transition = build_post_window_transition(
        root=root,
        candidate_path=Path(final_l2_candidates["path"]),
        cost_status_path=(root / "reports" / "active" / "hyperliquid_cost_collection_status.csv"),
        captured_at=now,
    )
    phases["venue_data_costs"].paths["post_window_transition"] = Path(
        post_window_transition["path"]
    )
    phases["venue_data_costs"].summary.update(
        {
            "post_window_ready_pairs": int(post_window_transition["ready_pairs"]),
            "post_window_collecting_pairs": int(post_window_transition["collecting_pairs"]),
            "post_window_eligible_pairs": int(post_window_transition["pairs"]),
        }
    )
    plan_path = root / "reports" / "active" / "corrective_execution_plan.csv"
    plan = _read_csv(plan_path)
    if plan.empty or set(TASK_STATUS) - set(plan.get("task_id", pd.Series(dtype=str)).astype(str)):
        raise ValueError("corrective execution plan is missing one or more canonical tasks")
    for task_id, fields in TASK_DEFINITION_OVERRIDES.items():
        mask = plan["task_id"].astype(str).eq(task_id)
        for field, value in fields.items():
            plan.loc[mask, field] = value
    task_status = dict(TASK_STATUS)
    venue = phases["venue_data_costs"].summary
    history_queue = int(venue.get("history_queued_pairs", 0) or 0)
    if history_queue == 0:
        task_status["T12"] = (
            "completed",
            "selected_pair_histories_ready_zero_active_remediation_rows",
        )
    l2_ready = int(venue.get("post_window_ready_pairs", 0) or 0)
    l2_eligible = int(venue.get("post_window_eligible_pairs", 0) or 0)
    task_status["T13"] = (
        (
            "completed"
            if l2_eligible > 0 and l2_ready == l2_eligible
            else "in_progress_evidence_collection"
        ),
        (f"strict_l2_ready_pairs_{l2_ready}_of_{l2_eligible}"),
    )
    task_status["T14"] = _strict_cost_task_status(venue)
    wizard = phases["wizard_parity"].summary
    ou_accounted = int(wizard.get("ou_optimal_orientations_accounted", 0) or 0)
    ou_expected = int(wizard.get("ou_optimal_expected_orientations", 2) or 2)
    task_status["T17"] = (
        (
            "completed"
            if ou_expected > 0 and ou_accounted == ou_expected
            else "in_progress_evidence_collection"
        ),
        f"ou_optimal_scanner_overlay_provenance_{ou_accounted}_of_{ou_expected}",
    )
    task_status["T28"] = _daily_cadence_task_status(phases["daily_cadence"].summary)
    statistics = phases["statistical_remediation"].summary
    registered_execution = _read_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    pending_candidate_cohort = _read_csv(
        root / "reports" / "active" / "current_hypothesis_batch.csv"
    )
    stage_four_state = _stage_four_completion_state(
        root=root,
        pending_candidate_cohort=pending_candidate_cohort,
        statistics=statistics,
    )
    task_status["T23"] = _registered_rerun_task_status(
        statistics,
        execution=registered_execution,
        root=root,
        current_family_accounting=stage_four_state,
    )
    registered_learning = _read_json(
        root / "reports" / "active" / "registered_learning_research_status.json"
    )
    registered_learning_audit = latest_verified_registered_learning(root=root)
    verified_stage5_pass = bool(
        registered_learning_audit.get("evidence_valid", False)
        and registered_learning_audit.get("stage5_research_gate_pass", False)
    )
    task_status["T32"] = (
        ("completed" if verified_stage5_pass else "blocked_pending_model_and_testnet_evidence"),
        (
            "registered_ml_rl_out_of_sample_gates_passed"
            if verified_stage5_pass
            else _join_blockers(
                [
                    *registered_learning_audit.get("blockers", []),
                    str(
                        registered_learning.get(
                            "blocker",
                            "model_incremental_edge_and_oos_rl_not_proven",
                        )
                    ),
                ]
            )
        ),
    )
    plan["status"] = plan["task_id"].map(lambda task: task_status[str(task)][0])
    plan["current_blocker_or_completion_reason"] = plan["task_id"].map(
        lambda task: task_status[str(task)][1]
    )
    plan["last_evaluated_at_utc"] = now.isoformat()
    _atomic_csv(plan, plan_path)
    completed = int(plan["status"].astype(str).str.startswith("completed").sum())
    in_progress = int(plan["status"].astype(str).str.startswith("in_progress").sum())
    blocked = len(plan) - completed - in_progress
    phase_rows = []
    for name, result in phases.items():
        phase_rows.append(
            {
                "phase": name,
                "status": str(result.summary.get("status", "BLOCKED")),
                "blocker": _summary_blocker(result.summary),
                "artifact_count": len(result.paths),
                "testnet_order_authority": bool(
                    result.summary.get("testnet_order_authority", False)
                ),
                "live_trading_authorized": bool(
                    result.summary.get("live_trading_authorized", False)
                ),
                "evidence_paths": ";".join(_relative(path, root) for path in result.paths.values()),
            }
        )
    phase_frame = pd.DataFrame(phase_rows)
    unsafe = bool(
        phase_frame["testnet_order_authority"].any() or phase_frame["live_trading_authorized"].any()
    )
    if unsafe:
        raise ValueError("corrective program unexpectedly acquired order authority")
    phase_path = root / "reports" / "active" / "corrective_plan_phase_status.csv"
    _atomic_csv(phase_frame, phase_path)
    final_receipt = _read_json(root / "reports" / "active" / "final_1x_survivor_receipt.json")
    daily_cycles = int(phases["daily_cadence"].summary.get("consecutive_complete_cycles", 0) or 0)
    strict_cost_ready_pairs, strict_cost_eligible_pairs = _stage_two_cost_counts(venue)
    full_survivor_clusters = int(statistics.get("independent_full_survivor_clusters", 0) or 0)
    full_survivor_pairs = int(statistics.get("independent_full_survivor_pairs", 0) or 0)
    primary_blockers = [
        "vendor_formula_parity_unproven",
        f"independent_full_survivor_breadth_{full_survivor_clusters}_of_three",
        f"independent_full_survivor_pair_breadth_{full_survivor_pairs}_of_three",
        f"daily_cadence_{daily_cycles}_of_seven",
        "realized_testnet_sample_absent",
    ]
    if scheduler_runtime.summary.get("status") != "PASS_SCHEDULER_RUNTIME_READY":
        primary_blockers.insert(0, "scheduler_runtime_contract_not_ready")
    if not bool(registered_learning.get("stage5_research_gate_pass", False)):
        primary_blockers.insert(3, "model_incremental_edge_not_accepted")
    if strict_cost_eligible_pairs <= 0 or strict_cost_ready_pairs < strict_cost_eligible_pairs:
        strict_cost_action = _next_strict_cost_action(venue)
        primary_blockers.insert(
            0,
            (
                "strict_pair_cost_requires_post_window_candidate_refresh"
                if strict_cost_action.startswith("refresh_wizard_candidate")
                else "strict_pair_cost_evidence_missing"
            ),
        )
    completion = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "implementation_status": "COMPLETE",
        "operational_acceptance_status": "BLOCKED",
        "total_tasks": len(plan),
        "completed_or_completed_checkpoint_tasks": completed,
        "in_progress_tasks": in_progress,
        "blocked_or_not_authorized_tasks": blocked,
        "phase_statuses": {row["phase"]: row["status"] for row in phase_rows},
        "final_one_x_survivors": int(final_receipt.get("final_one_x_survivors", 0) or 0),
        "current_decision": "CONTINUE_RESEARCH_ONLY",
        "orders_submitted_by_corrective_program": 0,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "primary_blockers": primary_blockers,
        "next_automatic_action": (
            _next_strict_cost_action(venue)
            + ";"
            + str(
                statistics.get(
                    "registered_rerun_next_action",
                    "await_registered_research_rerun_evidence",
                )
            )
            + ";daily_research_scheduler_collects_next_calendar_day_evidence_at_0615_local"
        ),
        "evidence_path": "reports/active/corrective_execution_plan.csv;reports/active/corrective_plan_phase_status.csv;reports/active/final_1x_survivor_receipt.json;reports/active/scheduler_runtime_readiness.json",
        "current_l2_candidate_pairs": int(final_l2_candidates["candidate_pairs"]),
        "current_l2_eligible_pairs": int(final_l2_candidates["eligible_pairs"]),
        "post_window_ready_pairs": int(post_window_transition["ready_pairs"]),
        "post_window_collecting_pairs": int(post_window_transition["collecting_pairs"]),
    }
    completion_path = root / "reports" / "active" / "corrective_plan_completion.json"
    completion_md = root / "reports" / "active" / "corrective_plan_completion.md"
    _atomic_json(completion, completion_path)
    completion_md.write_text(_completion_markdown(completion, plan, phase_frame), encoding="utf-8")
    seven_stage_path, seven_stage_md = _write_seven_stage_checkpoint(
        root=root,
        phases=phases,
        completion=completion,
        registered_learning_audit=registered_learning_audit,
        now=now,
    )
    return CommandResult(
        paths={
            "execution_plan": plan_path,
            "phase_status": phase_path,
            "completion_receipt": completion_path,
            "completion_summary": completion_md,
            "seven_stage_checkpoint": seven_stage_path,
            "seven_stage_checkpoint_markdown": seven_stage_md,
            "current_l2_candidates": Path(final_l2_candidates["path"]),
            **{
                f"phase_{phase}_{name}": path
                for phase, result in phases.items()
                for name, path in result.paths.items()
            },
        },
        summary=completion,
    )


def _daily_cadence_task_status(summary: dict[str, Any]) -> tuple[str, str]:
    """Project the authoritative cadence receipt into the canonical task ledger."""

    try:
        observed = max(int(summary.get("consecutive_complete_cycles", 0) or 0), 0)
    except (TypeError, ValueError):
        observed = 0
    try:
        required = int(summary.get("required_complete_cycles", 7) or 7)
    except (TypeError, ValueError):
        required = 7
    if required <= 0:
        required = 7
    status = "completed" if observed >= required else "in_progress_time_observation"
    return status, f"daily_cadence_{min(observed, required)}_of_{required}_cycles_observed"


def _refresh_testnet_collateral_preflight(
    *,
    root: Path,
    now: datetime,
    builder: Callable[..., CommandResult] = build_testnet_collateral_transfer_preflight,
) -> CommandResult:
    """Bind status-only collateral evidence to the candidate just regenerated."""

    approval = _read_json(root / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json")
    policy = approval.get("collateral_transfer_policy")
    raw_amount = policy.get("amount_usd", 25.0) if isinstance(policy, dict) else 25.0
    try:
        amount_usd = float(raw_amount)
    except (TypeError, ValueError):
        amount_usd = 25.0
    if not 20.0 <= amount_usd <= 25.0:
        amount_usd = 25.0
    result = builder(
        root=root,
        approval_id=str(approval.get("approval_id", "")),
        amount_usd=amount_usd,
        now=now,
    )
    forbidden_true_fields = (
        "agent_key_accessed",
        "transfer_attempted",
        "order_submission_performed",
        "testnet_collateral_transfer_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(result.summary.get(field) is not False for field in forbidden_true_fields):
        raise RuntimeError("canonical collateral preflight violated status-only contract")
    return result


def _write_seven_stage_checkpoint(
    *,
    root: Path,
    phases: dict[str, CommandResult],
    completion: dict[str, Any],
    registered_learning_audit: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    now = _as_utc(now)
    venue = phases["venue_data_costs"].summary
    wizard = phases["wizard_parity"].summary
    statistics = phases["statistical_remediation"].summary
    daily = phases["daily_cadence"].summary
    learning = phases["agent_learning_governance"].summary
    release = phases["conditional_release_gates"].summary
    latest_l2_capture = _read_json(
        root / "reports" / "active" / "corrective_l2_capture_status.json"
    )
    scheduler_runtime = _read_json(root / "reports" / "active" / "scheduler_runtime_readiness.json")
    scheduler_runtime_status = str(scheduler_runtime.get("status", "NOT_AUDITED"))
    scheduler_runtime_ready = bool(
        scheduler_runtime_status == "PASS_SCHEDULER_RUNTIME_READY"
        and int(scheduler_runtime.get("agents_ready", 0) or 0)
        == int(scheduler_runtime.get("agents_expected", 3) or 3)
    )
    scheduler_runtime_blocker = (
        ""
        if scheduler_runtime_ready
        else "scheduler_runtime_not_ready:"
        + (
            ",".join(str(value) for value in scheduler_runtime.get("blockers", []) if str(value))
            or scheduler_runtime_status
        )
    )
    wizard_proof_scheduler = _read_json(
        root / "reports" / "active" / "corrective_wizard_proof_scheduler_status.json"
    )
    wizard_input_audit = _read_json(
        root / "reports" / "active" / "exhaustive_wizard_mode_proof_input_audit.json"
    )
    wizard_proof_launcher = _read_json(
        root / "reports" / "active" / "corrective_wizard_proof_launcher_status.json"
    )
    immutable_wizard_execution = latest_verified_immutable_scheduler_execution(root=root)
    wizard_capture_manifest = _read_json(
        root / "reports" / "active" / "corrective_wizard_next_capture_manifest.json"
    )
    wizard_reset_readiness = _read_json(root / "reports" / "active" / "wizard_reset_readiness.json")
    wizard_surface_contract_enforced = (
        root / "config" / "wizard_surface_inventory_contract.json"
    ).is_file()
    wizard_surface_inventory = _read_json(
        root / "reports" / "active" / "wizard_surface_inventory_readiness.json"
    )
    wizard_surface_validation = (
        validate_wizard_surface_inventory_readiness(root=root)
        if wizard_surface_contract_enforced
        else {"status": "NOT_ENFORCED", "blocker": ""}
    )
    wizard_surface_inventory_ready = bool(
        not wizard_surface_contract_enforced or wizard_surface_validation.get("status") == "PASS"
    )
    wizard_browser_auth_contract_enforced = (
        root / "config" / "wizard_browser_auth_contract.json"
    ).is_file()
    wizard_browser_auth = _read_json(
        root / "reports" / "active" / "wizard_browser_auth_readiness.json"
    )
    wizard_browser_auth_validation = (
        validate_wizard_browser_auth_readiness(root=root, now=now)
        if wizard_browser_auth_contract_enforced
        else {"status": "NOT_ENFORCED", "blocker": ""}
    )
    wizard_browser_auth_ready = bool(
        not wizard_browser_auth_contract_enforced
        or wizard_browser_auth_validation.get("status") == "PASS"
    )
    wizard_comparator_review = _read_json(
        root / "reports" / "active" / "wizard_comparator_review_control.json"
    )
    stage4_handoff_readiness = _read_json(
        root / "reports" / "active" / "stage4_handoff_readiness.json"
    )
    stage4_handoff_validation = validate_stage4_handoff_readiness_receipt(
        root=root,
        receipt=stage4_handoff_readiness,
    )
    if stage4_handoff_validation["status"] != "PASS":
        stage4_handoff_readiness = {
            **stage4_handoff_readiness,
            "status": "BLOCKED_STAGE4_HANDOFF_EVIDENCE",
            "blockers": list(stage4_handoff_validation["blockers"]),
        }
    wizard_capture_reconciliation = _read_json(
        root / "reports" / "active" / "corrective_wizard_capture_reconciliation.json"
    )
    dynamic_holdout = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_holdout_status.json"
    )
    dynamic_supersession_gate = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_supersession_gate.json"
    )
    dynamic_activation = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_activation_status.json"
    )
    dynamic_proof_refresh = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_proof_refresh_status.json"
    )
    dynamic_review_packet = _read_json(
        root / "reports" / "active" / "wizard_dynamic_v2_review_packet.json"
    )
    dynamic_supreme_review = _read_json(
        root / "reports" / "supreme_team" / "wizard_dynamic_v2_review.json"
    )
    ou_trend_selector = _read_json(
        root / "reports" / "active" / "wizard_ou_trend_selector_v1_receipt.json"
    )
    ou_v3_capture = _read_json(root / "reports" / "active" / "wizard_ou_v3_capture_status.json")
    ou_v3_evaluation = _read_json(root / "reports" / "active" / "wizard_ou_v3_holdout_status.json")
    ou_v3_supersession_gate = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_supersession_gate.json"
    )
    ou_v3_review_packet = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_review_packet.json"
    )
    ou_v3_activation = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_activation_status.json"
    )
    ou_v3_proof_refresh = _read_json(
        root / "reports" / "active" / "wizard_ou_v3_proof_refresh_status.json"
    )
    ou_v4_receipt = _read_json(root / "reports" / "active" / "wizard_ou_v4_holdout_receipt.json")
    ou_v4_capture = _read_json(root / "reports" / "active" / "wizard_ou_v4_capture_status.json")
    ou_v4_evaluation = _read_json(root / "reports" / "active" / "wizard_ou_v4_holdout_status.json")
    ou_v4_supersession_gate = _read_json(
        root / "reports" / "active" / "wizard_ou_v4_supersession_gate.json"
    )
    ou_v4_review_packet = _read_json(
        root / "reports" / "active" / "wizard_ou_v4_review_packet.json"
    )
    ou_v4_supreme_review = _read_json(
        root / "reports" / "supreme_team" / "wizard_ou_v4_review.json"
    )
    ou_v4_activation = _read_json(
        root / "reports" / "active" / "wizard_ou_v4_activation_status.json"
    )
    ou_v4_proof_refresh = _read_json(
        root / "reports" / "active" / "wizard_ou_v4_proof_refresh_status.json"
    )
    ou_v5_receipt = _read_json(root / "reports" / "active" / "wizard_ou_v5_holdout_receipt.json")
    ou_v5_capture = _read_json(root / "reports" / "active" / "wizard_ou_v5_capture_status.json")
    ou_v5_evaluation = _read_json(root / "reports" / "active" / "wizard_ou_v5_holdout_status.json")
    ou_v5_failure_attribution = _read_json(
        root / "reports" / "active" / "wizard_ou_v5_failure_attribution.json"
    )
    ou_v5_supersession_gate = _read_json(
        root / "reports" / "active" / "wizard_ou_v5_supersession_gate.json"
    )
    ou_v5_review_packet = _read_json(
        root / "reports" / "active" / "wizard_ou_v5_review_packet.json"
    )
    ou_v5_supreme_review = _read_json(
        root / "reports" / "supreme_team" / "wizard_ou_v5_review.json"
    )
    ou_v5_activation = _read_json(
        root / "reports" / "active" / "wizard_ou_v5_activation_status.json"
    )
    ou_v5_proof_refresh = _read_json(
        root / "reports" / "active" / "wizard_ou_v5_proof_refresh_status.json"
    )
    ou_v6_receipt = _read_json(root / "reports" / "active" / "wizard_ou_v6_holdout_receipt.json")
    ou_v6_capture = _read_json(root / "reports" / "active" / "wizard_ou_v6_capture_status.json")
    ou_v6_evaluation = _read_json(root / "reports" / "active" / "wizard_ou_v6_holdout_status.json")
    ou_v6_activation = _read_json(
        root / "reports" / "active" / "wizard_ou_v6_activation_status.json"
    )
    ou_v6_proof_refresh = _read_json(
        root / "reports" / "active" / "wizard_ou_v6_proof_refresh_status.json"
    )
    ou_v4_required = bool(
        (root / "config" / "wizard_ou_comparator_v4_holdout.json").is_file() or ou_v4_receipt
    )
    ou_v4_evidence_complete = bool(
        not ou_v4_required or _ou_v4_immutable_evidence_complete(root, ou_v4_evaluation)
    )
    ou_v5_required = bool(
        (root / "config" / "wizard_ou_comparator_v5_holdout.json").is_file() or ou_v5_receipt
    )
    ou_v5_evidence_complete = bool(
        not ou_v5_required or _ou_v5_immutable_evidence_complete(root, ou_v5_evaluation)
    )
    ou_v6_required = bool(
        (root / "config" / "wizard_ou_comparator_v6_holdout.json").is_file() or ou_v6_receipt
    )
    ou_v6_evidence_complete = bool(
        not ou_v6_required or _ou_v6_immutable_evidence_complete(root, ou_v6_evaluation)
    )
    current_ou_evidence_complete = bool(
        ou_v6_evidence_complete
        if ou_v6_required
        else ou_v5_evidence_complete
        if ou_v5_required
        else ou_v4_evidence_complete
    )
    copula_v2_status = root / "reports" / "active" / "wizard_copula_behavioral_v2_status.json"
    copula_behavioral = _read_json(
        copula_v2_status
        if copula_v2_status.is_file()
        else root / "reports" / "active" / "wizard_copula_behavioral_status.json"
    )
    registered_execution = _read_json(
        root / "reports" / "active" / "registered_research_rerun_execution.json"
    )
    registered_learning = _read_json(
        root / "reports" / "active" / "registered_learning_research_status.json"
    )
    if registered_learning_audit is None:
        registered_learning_audit = latest_verified_registered_learning(root=root)
    registered_stage5_protocol = _read_json(
        root / "reports" / "active" / "registered_stage5_protocol.json"
    )
    pending_candidate_cohort = _read_csv(
        root / "reports" / "active" / "current_hypothesis_batch.csv"
    )
    pending_candidate_pairs = int(
        pending_candidate_cohort.get("pair", pd.Series(dtype=str)).astype(str).nunique()
    )
    prospective_candidate_pairs = int(
        pending_candidate_cohort.loc[
            pending_candidate_cohort.get("registration_cohort_role", pd.Series(dtype=str))
            .astype(str)
            .eq("prospective_current_family_breadth")
        ]
        .get("pair", pd.Series(dtype=str))
        .astype(str)
        .nunique()
    )
    proof_queue_eligible = int(wizard_proof_scheduler.get("queue_eligible", 0) or 0)
    proof_queue_completed = int(wizard_proof_scheduler.get("completed_after", 0) or 0)
    proof_responses_captured = int(wizard_proof_scheduler.get("responses_captured_after", 0) or 0)
    proof_input_ready = int(
        wizard_input_audit.get("ready_rows", wizard_proof_scheduler.get("input_audit_ready", 0))
        or 0
    )
    proof_input_retry_safe = int(
        wizard_input_audit.get(
            "retry_safe_rows",
            wizard_proof_scheduler.get("input_audit_retry_safe", proof_input_ready),
        )
        or 0
    )
    proof_payloads_changed_after_4xx = int(
        wizard_input_audit.get(
            "changed_after_vendor_4xx_rows",
            wizard_proof_scheduler.get("input_audit_changed_after_vendor_4xx", 0),
        )
        or 0
    )
    proof_payloads_unchanged_after_4xx = int(
        wizard_input_audit.get(
            "unchanged_vendor_4xx_rows",
            wizard_proof_scheduler.get("input_audit_unchanged_vendor_4xx", 0),
        )
        or 0
    )
    copula_expected_cells = int(copula_behavioral.get("expected_cells", 0) or 0)
    formula_proofs_expected = int(
        wizard.get(
            "formula_queue_cells_expected",
            max(proof_queue_eligible - copula_expected_cells, 0),
        )
        or 0
    )
    formula_proofs_passed = int(
        wizard.get("formula_queue_cells_passed", proof_queue_completed) or 0
    )
    accepted_mode_cells = int(
        wizard.get(
            "accepted_queue_cells",
            formula_proofs_passed
            + (
                int(copula_behavioral.get("behavioral_cells_passed", 0) or 0)
                if bool(copula_behavioral.get("behavioral_parity_proven", False))
                else 0
            ),
        )
        or 0
    )
    proof_scheduler_status = str(wizard_proof_scheduler.get("status", "NOT_CONFIGURED"))
    proof_scheduler_installed = any(
        path.is_file()
        for path in (
            root / ".runtime_agents" / "com.thewiz.corrective-wizard-proof.plist",
            Path.home() / "Library" / "LaunchAgents" / "com.thewiz.corrective-wizard-proof.plist",
        )
    )
    proof_collection_active = bool(
        proof_queue_eligible > proof_queue_completed
        and proof_scheduler_status != "COMPLETE_RESPONSE_CAPTURE_FORMULA_PROOF_INCOMPLETE"
        and (proof_scheduler_status != "NOT_CONFIGURED" or proof_scheduler_installed)
    )
    capture_reconciliation_required = int(
        wizard_capture_reconciliation.get("required_calls", 0) or 0
    )
    capture_reconciliation_completed = int(
        wizard_capture_reconciliation.get("completed_calls", 0) or 0
    )
    capture_reconciliation_status = str(
        wizard_capture_reconciliation.get("status", "NOT_CONFIGURED")
    )
    capture_reconciliation_pass = _capture_reconciliation_pass(wizard_capture_reconciliation)
    capture_manifest_continuity_status = str(
        wizard_proof_scheduler.get("capture_manifest_continuity_status", "NOT_CONFIGURED")
    )
    capture_manifest_continuity_pass = _capture_manifest_continuity_pass(wizard_proof_scheduler)
    immutable_wizard_execution_valid = bool(immutable_wizard_execution.get("valid", False))
    immutable_wizard_execution_stage3_complete = bool(
        immutable_wizard_execution.get("stage3_evidence_complete", False)
    )
    mutable_stage3_evidence_complete = bool(
        str(wizard.get("status", "BLOCKED")) == "PASS"
        and capture_reconciliation_pass
        and capture_manifest_continuity_pass
        and scheduler_runtime_ready
        and wizard_surface_inventory_ready
        and wizard_browser_auth_ready
        and current_ou_evidence_complete
    )
    wizard_stage_blockers = [str(wizard.get("blocker", ""))]
    if not scheduler_runtime_ready:
        wizard_stage_blockers.append(scheduler_runtime_blocker)
    if not wizard_surface_inventory_ready:
        wizard_stage_blockers.append(
            "wizard_surface_inventory_not_ready:"
            + str(wizard_surface_validation.get("blocker", "invalid_inventory_contract"))
        )
    if not wizard_browser_auth_ready:
        wizard_stage_blockers.append(
            "wizard_browser_auth_readiness_not_proven:"
            + str(wizard_browser_auth_validation.get("blocker", "invalid_browser_auth_readiness"))
        )
    if not capture_manifest_continuity_pass:
        wizard_stage_blockers.append(
            "capture_manifest_continuity_not_passed:"
            f"{capture_manifest_continuity_status}:"
            f"drift={wizard_proof_scheduler.get('capture_manifest_drift_detected')}"
        )
    if not capture_reconciliation_pass:
        wizard_stage_blockers.append(
            "capture_manifest_reconciliation_not_passed:"
            f"{capture_reconciliation_completed}_of_"
            f"{capture_reconciliation_required}:"
            f"{capture_reconciliation_status}"
        )
    if not immutable_wizard_execution_valid:
        wizard_stage_blockers.append(
            "final_immutable_scheduler_execution_not_verified:"
            + str(immutable_wizard_execution.get("status", "NOT_AUDITED"))
            + ":"
            + ",".join(
                str(value) for value in immutable_wizard_execution.get("blockers", []) if str(value)
            )
        )
    elif not immutable_wizard_execution_stage3_complete:
        wizard_stage_blockers.append(
            "final_immutable_scheduler_execution_stage3_incomplete:"
            + str(immutable_wizard_execution.get("receipt_id", ""))
        )
    if ou_v6_required and not ou_v6_evidence_complete:
        wizard_stage_blockers.append(
            "ou_v6_prospective_evidence_incomplete:"
            f"registration={ou_v6_receipt.get('status', 'MISSING')}:"
            f"capture={ou_v6_capture.get('status', 'NOT_CAPTURED')}:"
            f"evaluation={ou_v6_evaluation.get('status', 'NOT_EVALUATED')}:"
            f"responses={int(ou_v6_capture.get('responses_available', 0) or 0)}/"
            f"{int(ou_v6_capture.get('required_responses', 8) or 8)}"
        )
    if ou_v6_required and ou_v6_evaluation.get("status") == "FAIL":
        wizard_stage_blockers.extend(
            [
                "ou_v6_terminal_failure_exact_local_ou_parity_rejected",
                "ou_v6_successor_prohibited_after_terminal_failure",
            ]
        )
    if (
        ou_v6_required
        and ou_v6_evidence_complete
        and ou_v6_activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
    ):
        wizard_stage_blockers.append(
            "ou_v6_explicit_reviewed_activation_not_applied:"
            f"activation={ou_v6_activation.get('status', 'NOT_IMPLEMENTED')}"
        )
    if (
        ou_v6_required
        and ou_v6_activation.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and ou_v6_proof_refresh.get("status") != "PASS"
    ):
        wizard_stage_blockers.append(
            "ou_v6_activation_bound_proof_refresh_not_passed:"
            f"{ou_v6_proof_refresh.get('status', 'NOT_REFRESHED')}"
        )
    if not ou_v6_required and ou_v5_required and not ou_v5_evidence_complete:
        wizard_stage_blockers.append(
            "ou_v5_prospective_evidence_incomplete:"
            f"registration={ou_v5_receipt.get('status', 'MISSING')}:"
            f"capture={ou_v5_capture.get('status', 'NOT_CAPTURED')}:"
            f"evaluation={ou_v5_evaluation.get('status', 'NOT_EVALUATED')}:"
            f"responses={int(ou_v5_capture.get('responses_available', 0) or 0)}/"
            f"{int(ou_v5_capture.get('required_responses', 8) or 8)}"
        )
    if (
        not ou_v6_required
        and ou_v5_required
        and ou_v5_evaluation.get("status") == "FAIL"
        and ou_v5_failure_attribution.get("status") != "PASS_FAILURE_ATTRIBUTION_COMPLETE"
    ):
        wizard_stage_blockers.append("ou_v5_failure_attribution_not_complete")
    if (
        not ou_v6_required
        and ou_v5_required
        and ou_v5_evidence_complete
        and ou_v5_activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
    ):
        wizard_stage_blockers.append(
            "ou_v5_explicit_reviewed_activation_not_applied:"
            f"packet={ou_v5_review_packet.get('status', 'NOT_BUILT')}:"
            f"supreme={ou_v5_supreme_review.get('status', 'NOT_REVIEWED')}:"
            f"activation={ou_v5_activation.get('status', 'NOT_PLANNED')}"
        )
    if (
        not ou_v6_required
        and ou_v5_required
        and ou_v5_activation.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and ou_v5_proof_refresh.get("status") != "PASS"
    ):
        wizard_stage_blockers.append(
            "ou_v5_activation_bound_proof_refresh_not_passed:"
            f"{ou_v5_proof_refresh.get('status', 'NOT_REFRESHED')}"
        )
    if not ou_v6_required and not ou_v5_required and ou_v4_required and not ou_v4_evidence_complete:
        wizard_stage_blockers.append(
            "ou_v4_prospective_evidence_incomplete:"
            f"registration={ou_v4_receipt.get('status', 'MISSING')}:"
            f"capture={ou_v4_capture.get('status', 'NOT_CAPTURED')}:"
            f"evaluation={ou_v4_evaluation.get('status', 'NOT_EVALUATED')}:"
            f"responses={int(ou_v4_capture.get('responses_available', 0) or 0)}/"
            f"{int(ou_v4_capture.get('required_responses', 8) or 8)}"
        )
    if (
        not ou_v6_required
        and not ou_v5_required
        and ou_v4_required
        and ou_v4_evidence_complete
        and ou_v4_activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
    ):
        wizard_stage_blockers.append(
            "ou_v4_explicit_reviewed_activation_not_applied:"
            f"packet={ou_v4_review_packet.get('status', 'NOT_BUILT')}:"
            f"supreme={ou_v4_supreme_review.get('status', 'NOT_REVIEWED')}:"
            f"activation={ou_v4_activation.get('status', 'NOT_PLANNED')}"
        )
    if (
        not ou_v6_required
        and not ou_v5_required
        and ou_v4_required
        and ou_v4_activation.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
        and ou_v4_proof_refresh.get("status") != "PASS"
    ):
        wizard_stage_blockers.append(
            "ou_v4_activation_bound_proof_refresh_not_passed:"
            f"{ou_v4_proof_refresh.get('status', 'NOT_REFRESHED')}"
        )
    if wizard_reset_readiness and str(wizard_reset_readiness.get("status", "")) != (
        "PASS_RESET_AUTOMATION_READY"
    ):
        wizard_stage_blockers.append(
            "wizard_reset_automation_not_ready:"
            + ",".join(
                str(value) for value in wizard_reset_readiness.get("blockers", []) if str(value)
            )
        )
    if str(wizard_comparator_review.get("status", "")) == "BLOCKED_REVIEW_CONTROL":
        wizard_stage_blockers.append(
            "wizard_comparator_review_control_blocked:"
            + ",".join(
                str(value)
                for value in wizard_comparator_review.get("control_blockers", [])
                if str(value)
            )
        )
    wizard_stage_blocker = ";".join(blocker for blocker in wizard_stage_blockers if blocker)
    strict_cost_next_action = _next_strict_cost_action(venue)
    strict_cost_ready, strict_cost_eligible = _stage_two_cost_counts(venue)
    strict_cost_complete = bool(
        strict_cost_eligible > 0 and strict_cost_ready == strict_cost_eligible
    )
    strict_cost_blocker = (
        ""
        if strict_cost_complete
        else (
            "strict_pair_cost_requires_post_window_candidate_refresh"
            if strict_cost_next_action.startswith("refresh_wizard_candidate")
            else "strict_pair_cost_evidence_missing"
        )
    )
    rerun_gate_status = str(statistics.get("registered_rerun_gate_status", "NOT_EVALUATED"))
    rerun_conclusion = str(statistics.get("registered_rerun_conclusion_status", "INCOMPLETE"))
    execution_conclusion = (
        str(registered_execution.get("conclusion_status", "INCOMPLETE"))
        if _registered_execution_receipt_valid(root, registered_execution)
        else "INCOMPLETE"
    )
    final_survivors = int(statistics.get("final_one_x_survivors", 0) or 0)
    stage_four_state = _stage_four_completion_state(
        root=root,
        pending_candidate_cohort=pending_candidate_cohort,
        statistics=statistics,
    )
    stage_four_complete = bool(stage_four_state["complete"])
    registered_learning_available = bool(registered_learning)
    stage5_protocol_pass = bool(
        registered_stage5_protocol.get("status") == "PASS_PROSPECTIVE_PROTOCOL_REGISTERED"
    )
    stage_five_evidence_valid = bool(registered_learning_audit.get("evidence_valid", False))
    stage_five_pass = bool(
        stage_five_evidence_valid
        and registered_learning_audit.get("stage5_research_gate_pass", False)
    )
    stage_five_status = (
        "PASS"
        if stage_five_pass
        else "IN_PROGRESS"
        if str(registered_learning.get("status", "")) in {"PLANNED", "RUNNING"}
        else "BLOCKED"
    )
    model_blockers = learning.get("model_blockers", learning.get("blockers", []))
    if isinstance(model_blockers, str):
        model_blockers = [model_blockers]
    stage_five_blocker = _stage_five_blockers(
        stage_five_pass=stage_five_pass,
        stage_four_complete=stage_four_complete,
        registered_learning=registered_learning,
        registered_learning_available=registered_learning_available,
        registered_learning_audit=registered_learning_audit,
        stage5_protocol_pass=stage5_protocol_pass,
        model_blockers=model_blockers,
    )
    active = root / "reports" / "active"
    testnet_candidate = _read_json(active / "testnet_candidate_receipt.json")
    testnet_prospective = _read_json(active / "testnet_prospective_cohort_readiness.json")
    testnet_lifecycle = _read_json(active / "testnet_lifecycle_execution_receipt.json")
    testnet_no_order = _read_csv(active / "testnet_no_order_preflight.csv")
    testnet_samples = _read_csv(active / "realized_testnet_sample_sufficiency.csv")
    testnet_inventory = _read_csv(active / "hyperliquid_testnet_market_inventory.csv")
    testnet_account_preflight = _read_csv(active / "hyperliquid_testnet_preflight.csv")
    testnet_margin = _read_csv(active / "hyperliquid_testnet_margin_snapshot.csv")
    testnet_collateral_preflight = _read_json(active / "testnet_collateral_transfer_preflight.json")
    account_row = (
        testnet_account_preflight.iloc[-1].to_dict() if not testnet_account_preflight.empty else {}
    )
    margin_row = testnet_margin.iloc[-1].to_dict() if not testnet_margin.empty else {}
    inventory_tradable = int(
        testnet_inventory.get("tradable_perp", pd.Series(False, index=testnet_inventory.index))
        .map(_truthy)
        .sum()
    )
    inventory_fetch_blocked = int(
        testnet_inventory.get("fetch_blocker", pd.Series("", index=testnet_inventory.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
        .sum()
    )
    account_preflight_ready = bool(
        _truthy(account_row.get("ready_for_no_order_preflight"))
        and not _truthy(account_row.get("ready_for_testnet_submit"))
        and not _truthy(account_row.get("order_submission_performed"))
        and not _truthy(account_row.get("testnet_order_authority"))
        and not _truthy(account_row.get("live_trading_authorized"))
    )
    margin_status = str(margin_row.get("status", "MISSING"))
    margin_blockers = str(margin_row.get("blockers", ""))
    collateral_required = "testnet_usdc_requires_spot_to_perp_transfer" in margin_blockers
    collateral_preflight_status = str(testnet_collateral_preflight.get("status", "MISSING"))
    no_order_passed = int(
        testnet_no_order.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").sum()
    )
    no_order_checks = len(testnet_no_order)
    no_order_blocked_checks = (
        testnet_no_order.loc[
            testnet_no_order.get("status", pd.Series(dtype=str)).astype(str).ne("PASS"),
            "check",
        ]
        .dropna()
        .astype(str)
        .tolist()
        if not testnet_no_order.empty and {"check", "status"}.issubset(testnet_no_order.columns)
        else []
    )
    sample_artifact_pass = bool(
        not testnet_samples.empty
        and {
            "status",
            "testnet_order_authority",
            "live_trading_authorized",
        }.issubset(testnet_samples.columns)
        and testnet_samples.get("status", pd.Series(dtype=str)).astype(str).eq("PASS").all()
        and not testnet_samples.get(
            "testnet_order_authority",
            pd.Series(False, index=testnet_samples.index),
        )
        .map(_truthy)
        .any()
        and not testnet_samples.get(
            "live_trading_authorized",
            pd.Series(False, index=testnet_samples.index),
        )
        .map(_truthy)
        .any()
    )
    candidate_status = str(testnet_candidate.get("candidate_status", "MISSING"))
    prospective_testnet_status = str(testnet_prospective.get("status", "MISSING"))
    prospective_testnet_pairs = int(testnet_prospective.get("cohort_pairs", 0) or 0)
    prospective_testnet_ready_pairs = int(
        testnet_prospective.get("prospectively_ready_pairs", 0) or 0
    )
    prospective_testnet_authority_safe = bool(
        testnet_prospective.get("candidate_selection_performed") is False
        and testnet_prospective.get("collateral_transfer_attempted") is False
        and testnet_prospective.get("order_submission_performed") is False
        and testnet_prospective.get("candidate_promotion_authority") is False
        and testnet_prospective.get("testnet_order_authority") is False
        and testnet_prospective.get("live_trading_authorized") is False
    )
    lifecycle_status = str(testnet_lifecycle.get("lifecycle_status", "MISSING"))
    candidate_authority_safe = bool(
        testnet_candidate.get("testnet_order_authority") is False
        and testnet_candidate.get("live_trading_authorized") is False
    )
    lifecycle_authority_safe = bool(
        testnet_lifecycle.get("testnet_order_authority") is False
        and testnet_lifecycle.get("live_trading_authorized") is False
    )
    no_order_operational_pass = bool(no_order_checks > 0 and no_order_passed == no_order_checks)
    stage_six_pass = bool(
        release.get("testnet_sample_status") == "PASS"
        and sample_artifact_pass
        and candidate_status == "READY_FOR_NO_ORDER_PREFLIGHT"
        and candidate_authority_safe
        and lifecycle_status == "COMPLETE_RECONCILED"
        and lifecycle_authority_safe
        and account_preflight_ready
        and margin_status in {"PASS", "READY"}
        and no_order_operational_pass
        and inventory_tradable > 0
    )
    stage_six_blockers: list[str] = []
    if candidate_status != "READY_FOR_NO_ORDER_PREFLIGHT":
        candidate_blockers = testnet_candidate.get("blockers", [])
        if isinstance(candidate_blockers, list):
            stage_six_blockers.extend(str(value) for value in candidate_blockers)
            if not candidate_blockers:
                stage_six_blockers.append(f"testnet_candidate:{candidate_status}")
        else:
            stage_six_blockers.append(str(candidate_blockers))
    if not candidate_authority_safe:
        stage_six_blockers.append("testnet_candidate_authority_flags_not_fail_closed")
    if testnet_prospective and not prospective_testnet_authority_safe:
        stage_six_blockers.append("testnet_prospective_cohort_authority_flags_not_fail_closed")
    if inventory_tradable <= 0:
        stage_six_blockers.append("testnet_tradable_perp_inventory_missing")
    if not account_preflight_ready:
        stage_six_blockers.append("testnet_wallet_agent_preflight_not_ready")
    if no_order_checks <= 0:
        stage_six_blockers.append("testnet_no_order_preflight_missing_or_empty")
    elif no_order_blocked_checks:
        stage_six_blockers.append(
            "testnet_no_order_checks_blocked:" + ",".join(no_order_blocked_checks)
        )
    if margin_status not in {"PASS", "READY"}:
        stage_six_blockers.append(str(margin_row.get("blockers", "testnet_margin_not_ready")))
        if collateral_required and collateral_preflight_status not in {
            "READY_REQUIRES_EXPLICIT_EXECUTION",
            "NO_TRANSFER_REQUIRED",
        }:
            stage_six_blockers.append(
                f"testnet_collateral_transfer_preflight:{collateral_preflight_status}"
            )
    if lifecycle_status != "COMPLETE_RECONCILED":
        stage_six_blockers.append(f"testnet_lifecycle:{lifecycle_status}")
    if not lifecycle_authority_safe:
        stage_six_blockers.append("testnet_lifecycle_authority_flags_not_fail_closed")
    if not sample_artifact_pass:
        stage_six_blockers.append("realized_testnet_sample_sufficiency_not_passed")
    stage_six_next_action = (
        "stage6_complete_await_stage7_policy_gate"
        if stage_six_pass
        else "remain_no_order_until_final_survivor_receipt_grants_candidate_authority"
        if candidate_status != "READY_FOR_NO_ORDER_PREFLIGHT"
        else "repair_testnet_wallet_agent_read_only_preflight"
        if not account_preflight_ready
        else "repair_testnet_no_order_operational_checks"
        if not no_order_operational_pass
        else "refresh_testnet_tradable_perp_inventory"
        if inventory_tradable <= 0
        else "obtain_exact_one_use_signed_collateral_transfer_approval_and_rerun_preflight"
        if collateral_required
        and collateral_preflight_status != "READY_REQUIRES_EXPLICIT_EXECUTION"
        else "await_explicit_user_confirmation_for_one_use_testnet_collateral_transfer"
        if collateral_required
        else "repair_testnet_perp_margin_readiness"
        if margin_status not in {"PASS", "READY"}
        else "execute_only_explicitly_approved_candidate_bound_testnet_lifecycle"
        if lifecycle_status != "COMPLETE_RECONCILED"
        else "collect_independent_reconciled_testnet_lifecycles_to_frozen_sample_policy"
    )
    live_authorization_path = active / "live_canary_authorization.json"
    live_authorization = _read_json(live_authorization_path)
    live_outcome = _read_json(active / "live_canary_outcome_evaluation.json")
    stage_seven_pass = bool(
        stage_six_pass
        and _live_canary_outcome_pass(
            root=root,
            authorization_path=live_authorization_path,
            outcome=live_outcome,
        )
    )
    stage_seven_blockers: list[str] = []
    authorization_blockers = live_authorization.get("blockers", [])
    if isinstance(authorization_blockers, list):
        stage_seven_blockers.extend(str(value) for value in authorization_blockers)
    outcome_execution_blockers = live_outcome.get("execution_blockers", [])
    if isinstance(outcome_execution_blockers, list):
        stage_seven_blockers.extend(str(value) for value in outcome_execution_blockers)
    outcome_review_blockers = live_outcome.get("post_canary_review_blockers", [])
    if isinstance(outcome_review_blockers, list):
        stage_seven_blockers.extend(str(value) for value in outcome_review_blockers)
    if not stage_seven_pass:
        stage_seven_blockers.append("one_live_canary_complete_reconciled_outcome_not_proven")
    stage_seven_next_action = (
        "one_live_canary_complete_no_further_authority"
        if stage_seven_pass
        else "remain_no_live_order_until_stage6_and_live_policy_gates_pass"
        if not stage_six_pass
        else "obtain_exact_one_run_user_authorization_for_manual_live_canary"
        if live_authorization.get("authorization_status") != "AUTHORIZED_FOR_ONE_LIVE_CANARY"
        else "execute_and_reconcile_exactly_one_approved_live_canary_then_run_post_canary_review"
    )
    ou_v6_next_action = _next_ou_v6_action(
        required=ou_v6_required,
        immutable_evidence_complete=ou_v6_evidence_complete,
        evaluation=ou_v6_evaluation,
        activation=ou_v6_activation,
        proof_refresh=ou_v6_proof_refresh,
    )
    ou_v5_next_action = _next_ou_v5_action(
        required=ou_v5_required,
        immutable_evidence_complete=ou_v5_evidence_complete,
        evaluation=ou_v5_evaluation,
        failure_attribution=ou_v5_failure_attribution,
        review_packet=ou_v5_review_packet,
        supreme_review=ou_v5_supreme_review,
        activation=ou_v5_activation,
        proof_refresh=ou_v5_proof_refresh,
    )
    rows = [
        {
            "stage": 1,
            "objective": "seven_distinct_daily_research_receipts",
            "status": "PASS"
            if int(daily.get("consecutive_complete_cycles", 0)) >= 7 and scheduler_runtime_ready
            else "IN_PROGRESS",
            "evidence_progress": (
                f"{int(daily.get('consecutive_complete_cycles', 0))}/7;"
                f"wizard_semantic_receipt_required_from={SEMANTIC_DAILY_RECEIPT_REQUIRED_FROM_UTC};"
                "full_sweep_raw_and_credit_binding_required=True;"
                "semantic_qualification_required_for_all_receipts=True;"
                f"scheduler_runtime={scheduler_runtime_status};"
                f"scheduler_agents={int(scheduler_runtime.get('agents_ready', 0) or 0)}/"
                f"{int(scheduler_runtime.get('agents_expected', 3) or 3)};"
                f"scheduler_runtime_receipt={scheduler_runtime.get('receipt_id', '')}"
            ),
            "blocker": _join_blockers([str(daily.get("blocker", "")), scheduler_runtime_blocker]),
            "next_action": (
                "repair_scheduler_runtime_contract"
                if not scheduler_runtime_ready
                else "daily_research_scheduler_runs_at_0615_local"
            ),
            "evidence_path": (
                "reports/active/daily_cadence_acceptance.csv;"
                "reports/active/daily_schedule_receipts;"
                "reports/runs/current_wizard_hyperliquid_daily;"
                "reports/active/scheduler_runtime_readiness_checks.csv;"
                "reports/active/scheduler_runtime_readiness.json;"
                "reports/active/scheduler_runtime_readiness.md;"
                "data/research/scheduler_runtime_readiness"
            ),
        },
        {
            "stage": 2,
            "objective": "hyperliquid_history_and_strict_pair_cost_evidence",
            "status": (
                "PASS"
                if strict_cost_complete and int(venue.get("history_queued_pairs", 0)) == 0
                else "IN_PROGRESS"
            ),
            "evidence_progress": (
                f"history_ready={int(venue.get('history_ready_pairs', 0))};"
                f"history_active_remediation={int(venue.get('history_queued_pairs', 0))};"
                f"history_insufficient_asset_age={int(venue.get('history_insufficient_asset_age_pairs', 0))};"
                f"strict_cost_ready_pairs={strict_cost_ready}/{strict_cost_eligible};"
                f"exploratory_cost_ready_pairs={int(venue.get('strict_cost_ready_pairs', 0))}/"
                f"{int(venue.get('strict_cost_eligible_pairs', 0))};"
                f"post_window_ready_pairs={int(venue.get('post_window_ready_pairs', 0))}/"
                f"{int(venue.get('post_window_eligible_pairs', 0))};"
                f"l2_readiness_refresh={venue.get('l2_readiness_refresh_status', 'NOT_AVAILABLE')};"
                f"l2_readiness_validation={venue.get('l2_readiness_refresh_validation_status', 'NOT_AVAILABLE')};"
                f"l2_readiness_receipt={venue.get('l2_readiness_refresh_receipt_id', '')};"
                f"l2_readiness_pairs={int(venue.get('l2_readiness_refresh_ready_pairs', 0))}/"
                f"{int(venue.get('l2_readiness_refresh_eligible_pairs', 0))};"
                f"l2_readiness_gate_executed={bool(venue.get('l2_readiness_refresh_gate_executed', False))};"
                f"l2_readiness_handoff_executed={bool(venue.get('l2_readiness_refresh_handoff_executed', False))};"
                f"mapping_maintenance={latest_l2_capture.get('mapping_refresh_status', 'NOT_RECORDED')};"
                f"mapping_refresh_due={bool(latest_l2_capture.get('mapping_refresh_due', False))};"
                f"mapping_age_hours={latest_l2_capture.get('mapping_age_hours_after', '')};"
                f"mapping_refresh_id={latest_l2_capture.get('mapping_refresh_id', '')};"
                f"mapping_ready_pair_groups={int(latest_l2_capture.get('mapping_ready_pair_groups', 0) or 0)}/"
                f"{int(latest_l2_capture.get('mapping_ready_pair_groups', 0) or 0) + int(latest_l2_capture.get('mapping_blocked_pair_groups', 0) or 0)};"
                f"mapping_maintenance_warnings={len(latest_l2_capture.get('operational_warnings', []))};"
                f"scheduler_runtime={scheduler_runtime_status};"
                f"scheduler_runtime_receipt={scheduler_runtime.get('receipt_id', '')}"
            ),
            "blocker": strict_cost_blocker,
            "next_action": strict_cost_next_action,
            "evidence_path": (
                "reports/active/hyperliquid_history_coverage.csv;"
                "reports/active/hyperliquid_cost_collection_status.csv;"
                "data/processed/hyperliquid_pair_cost_models.csv;"
                "reports/active/hyperliquid_cost_stress.csv;"
                "reports/active/corrective_l2_post_window_transition.csv;"
                "reports/active/corrective_l2_capture_status.json;"
                "reports/active/corrective_l2_readiness_refresh_status.json;"
                "data/research/l2_readiness_refresh;"
                "reports/active/scheduler_runtime_readiness.json"
            ),
        },
        {
            "stage": 3,
            "objective": "crypto_wizards_exact_mode_provenance_and_formula_parity",
            "status": (
                "PASS"
                if mutable_stage3_evidence_complete
                and immutable_wizard_execution_valid
                and immutable_wizard_execution_stage3_complete
                else "IN_PROGRESS"
                if scheduler_runtime_ready
                and (
                    proof_collection_active
                    or capture_reconciliation_status == "PENDING"
                    or (ou_v6_required and not ou_v6_evidence_complete)
                    or (not ou_v6_required and ou_v5_required and not ou_v5_evidence_complete)
                    or (
                        not ou_v6_required
                        and not ou_v5_required
                        and ou_v4_required
                        and not ou_v4_evidence_complete
                    )
                )
                else "BLOCKED"
            ),
            "evidence_progress": (
                f"pair_page_modes={int(wizard.get('fixture_cells_accounted', 0)) // 2};"
                f"surface_inventory_enforced={wizard_surface_contract_enforced};"
                f"surface_inventory={wizard_surface_inventory.get('status', 'NOT_AUDITED')};"
                f"surface_inventory_receipt={wizard_surface_inventory.get('receipt_id', '')};"
                f"surface_inventory_checks={int(wizard_surface_inventory.get('checks_passed', 0) or 0)}/"
                f"{int(wizard_surface_inventory.get('checks_total', 0) or 0)};"
                f"surface_inventory_api_endpoints={int(wizard_surface_inventory.get('api_endpoint_count', 0) or 0)};"
                f"surface_inventory_dashboard_areas={int(wizard_surface_inventory.get('dashboard_surface_count', 0) or 0)};"
                f"surface_inventory_integration_status={wizard_surface_inventory.get('integration_status', 'NOT_AUDITED')};"
                f"browser_auth_enforced={wizard_browser_auth_contract_enforced};"
                f"browser_auth={wizard_browser_auth.get('status', 'NOT_AUDITED')};"
                f"browser_auth_receipt={wizard_browser_auth.get('receipt_id', '')};"
                f"browser_auth_routes={','.join(str(value) for value in wizard_browser_auth_validation.get('route_kinds', []))};"
                f"ou_optimal_orientations={int(wizard.get('ou_optimal_orientations_accounted', 0))}/"
                f"{int(wizard.get('ou_optimal_expected_orientations', 2))};"
                f"vendor_parity_cells={int(wizard.get('vendor_parity_cells', 0))};"
                f"vendor_responses={proof_responses_captured}/{proof_queue_eligible};"
                f"formula_proofs={formula_proofs_passed}/{formula_proofs_expected};"
                f"accepted_mode_evidence={accepted_mode_cells}/{proof_queue_eligible};"
                f"proof_scheduler={proof_scheduler_status};"
                f"proof_scheduler_installed={proof_scheduler_installed};"
                f"proof_launcher={wizard_proof_launcher.get('launcher_status', 'NOT_CONFIGURED')};"
                f"proof_launcher_same_day_state={wizard_proof_launcher.get('same_day_attempt_state', 'NOT_EVALUATED')};"
                f"proof_launcher_heavy_invoked={bool(wizard_proof_launcher.get('heavy_scheduler_invoked', False))};"
                f"proof_launcher_status_manifest_bound={bool(wizard_proof_launcher.get('latest_status_capture_manifest_binding_complete', False))};"
                f"proof_launcher_immutable_manifest_bound={bool(wizard_proof_launcher.get('latest_immutable_execution_capture_manifest_binding_complete', False))};"
                f"proof_launcher_status_stage3_complete={bool(wizard_proof_launcher.get('latest_status_stage3_evidence_complete', False))};"
                f"proof_launcher_immutable_stage3_complete={bool(wizard_proof_launcher.get('latest_immutable_execution_stage3_evidence_complete', False))};"
                f"final_immutable_execution={immutable_wizard_execution.get('status', 'NOT_AUDITED')};"
                f"final_immutable_execution_receipt={immutable_wizard_execution.get('receipt_id', '')};"
                f"final_immutable_execution_stage3_complete={immutable_wizard_execution_stage3_complete};"
                f"proof_input_audit={proof_input_ready}/{proof_queue_eligible};"
                f"proof_retry_safe={proof_input_retry_safe}/{proof_queue_eligible};"
                f"proof_payloads_changed_after_4xx={proof_payloads_changed_after_4xx};"
                f"proof_payloads_unchanged_after_4xx={proof_payloads_unchanged_after_4xx};"
                f"credit_budget={wizard_proof_scheduler.get('credit_budget_status', 'NOT_CHECKED')};"
                f"scheduled_credits={int(wizard_proof_scheduler.get('scheduled_credit_ceiling', 0) or 0)};"
                f"credit_headroom={int(wizard_proof_scheduler.get('credit_headroom_after_reserve', 0) or 0)};"
                f"proof_lane_credit_ceiling={int(wizard_proof_scheduler.get('proof_lane_credit_ceiling', 0) or 0)};"
                f"capture_manifest={wizard_capture_manifest.get('status', 'NOT_CONFIGURED')};"
                f"capture_manifest_state={wizard_capture_manifest.get('capture_state', 'NOT_CONFIGURED')};"
                f"capture_manifest_id={wizard_capture_manifest.get('manifest_id', '')};"
                f"capture_manifest_calls={int(wizard_capture_manifest.get('pending_calls', 0) or 0)};"
                f"capture_manifest_credits={int(wizard_capture_manifest.get('planned_credits', 0) or 0)};"
                f"capture_manifest_next_eligible={wizard_capture_manifest.get('next_external_attempt_eligible_at', '')};"
                f"reset_automation={wizard_reset_readiness.get('status', 'NOT_AUDITED')};"
                f"reset_automation_checks={int(wizard_reset_readiness.get('checks_passed', 0) or 0)}/"
                f"{int(wizard_reset_readiness.get('checks_total', 0) or 0)};"
                f"reset_automation_launch_agent_loaded={bool(wizard_reset_readiness.get('launch_agent_loaded', False))};"
                f"scheduler_runtime={scheduler_runtime_status};"
                f"scheduler_agents={int(scheduler_runtime.get('agents_ready', 0) or 0)}/"
                f"{int(scheduler_runtime.get('agents_expected', 3) or 3)};"
                f"scheduler_runtime_receipt={scheduler_runtime.get('receipt_id', '')};"
                f"comparator_review_control={wizard_comparator_review.get('status', 'NOT_AUDITED')};"
                f"comparator_review_ready={int(wizard_comparator_review.get('review_ready', 0) or 0)}/"
                f"{int(wizard_comparator_review.get('comparators', 2) or 2)};"
                f"comparator_apply_ready={int(wizard_comparator_review.get('apply_ready', 0) or 0)}/"
                f"{int(wizard_comparator_review.get('comparators', 2) or 2)};"
                f"comparator_applied={int(wizard_comparator_review.get('applied', 0) or 0)}/"
                f"{int(wizard_comparator_review.get('comparators', 2) or 2)};"
                f"comparator_review_receipt={wizard_comparator_review.get('receipt_id', '')};"
                f"capture_manifest_continuity={capture_manifest_continuity_status};"
                f"capture_manifest_continuity_valid={capture_manifest_continuity_pass};"
                f"capture_manifest_drift={wizard_proof_scheduler.get('capture_manifest_drift_detected')};"
                f"capture_reconciliation={capture_reconciliation_status};"
                f"capture_reconciled_calls={capture_reconciliation_completed}/"
                f"{capture_reconciliation_required};"
                f"capture_reconciliation_id={wizard_capture_reconciliation.get('reconciliation_id', '')};"
                f"credit_reservation={wizard_proof_scheduler.get('credit_reservation_status', 'NOT_RECORDED')};"
                f"credit_reconciliation={wizard_proof_scheduler.get('credit_reconciliation_status', 'NOT_RECORDED')};"
                f"next_cohort_remaining={int(wizard_proof_scheduler.get('remaining_vendor_responses', 0) or 0)};"
                f"next_cohort_capacity={int(wizard_proof_scheduler.get('configured_proof_request_capacity', 0) or 0)};"
                f"next_cohort_required_credits={int(wizard_proof_scheduler.get('remaining_custom_series_credits', 0) or 0)};"
                f"next_cohort_readiness={wizard_proof_scheduler.get('next_cohort_readiness', 'NOT_EVALUATED')};"
                f"next_utc_reset_at={wizard_proof_scheduler.get('next_utc_reset_at', '')};"
                f"next_external_attempt_eligible_at={wizard_proof_scheduler.get('next_external_attempt_eligible_at', '')};"
                f"dynamic_v2_holdout={dynamic_holdout.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_cells={int(dynamic_holdout.get('passed_cells', 0) or 0)}/"
                f"{int(dynamic_holdout.get('required_cells', 4) or 4)};"
                f"dynamic_v2_supersession={dynamic_supersession_gate.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_activation={dynamic_activation.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_generation={2 if dynamic_activation.get('status') == 'APPLIED_RESEARCH_COMPARATOR_ONLY' else 1};"
                f"dynamic_v2_proof_refresh={dynamic_proof_refresh.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_exact_rows={int(dynamic_proof_refresh.get('exact_dynamic_rows', 0) or 0)}/"
                f"{int(dynamic_proof_refresh.get('captured_dynamic_rows', 0) or 0)};"
                f"dynamic_v2_review_packet={dynamic_review_packet.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_review_packet_id={dynamic_review_packet.get('review_packet_id', '')};"
                f"dynamic_v2_supreme_review={dynamic_supreme_review.get('status', 'NOT_CONFIGURED')};"
                f"dynamic_v2_supreme_recommendation={dynamic_supreme_review.get('recommendation', '')};"
                f"ou_trend_selector={ou_trend_selector.get('status', 'NOT_CONFIGURED')};"
                f"ou_selector_derivation={int(ou_trend_selector.get('derivation_matched_cells', 0) or 0)}/"
                f"{int(ou_trend_selector.get('derivation_cells', 8) or 8)};"
                f"ou_selector_predictions={int(ou_trend_selector.get('holdout_prediction_cells', 0) or 0)};"
                f"ou_v3_capture={ou_v3_capture.get('status', 'NOT_CONFIGURED')};"
                f"ou_v3_responses={int(ou_v3_capture.get('responses_available', 0) or 0)}/"
                f"{int(ou_v3_capture.get('required_responses', 4) or 4)};"
                f"ou_v3_evaluation={ou_v3_capture.get('evaluation_status', 'NOT_EVALUATED')};"
                f"ou_v3_local_selector_proven="
                f"{bool(ou_v3_evaluation.get('local_point_in_time_trend_selector_proven', False))};"
                f"ou_v3_supersession={ou_v3_supersession_gate.get('status', 'NOT_CONFIGURED')};"
                f"ou_v3_review_packet={ou_v3_review_packet.get('status', 'NOT_CONFIGURED')};"
                f"ou_v3_review_packet_id={ou_v3_review_packet.get('review_packet_id', '')};"
                f"ou_v3_activation={ou_v3_activation.get('status', 'NOT_CONFIGURED')};"
                f"ou_v3_generation={3 if ou_v3_activation.get('status') == 'APPLIED_RESEARCH_COMPARATOR_ONLY' else 1};"
                f"ou_v3_proof_refresh={ou_v3_proof_refresh.get('status', 'NOT_CONFIGURED')};"
                f"ou_v3_exact_rows={int(ou_v3_proof_refresh.get('exact_ou_rows', 0) or 0)}/"
                f"{int(ou_v3_proof_refresh.get('captured_ou_rows', 0) or 0)};"
                f"ou_v4_registration={ou_v4_receipt.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_capture={ou_v4_capture.get('status', 'NOT_CAPTURED')};"
                f"ou_v4_responses={int(ou_v4_capture.get('responses_available', 0) or 0)}/"
                f"{int(ou_v4_capture.get('required_responses', 8) or 8)};"
                f"ou_v4_evaluation={ou_v4_evaluation.get('status', 'NOT_EVALUATED')};"
                f"ou_v4_cells={int(ou_v4_evaluation.get('passed_cells', 0) or 0)}/"
                f"{int(ou_v4_evaluation.get('required_cells', 8) or 8)};"
                f"ou_v4_transform_cells={int(ou_v4_evaluation.get('transform_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v4_trend_cells={int(ou_v4_evaluation.get('trend_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v4_formula_cells={int(ou_v4_evaluation.get('formula_parity_passed_cells', 0) or 0)};"
                f"ou_v4_immutable_evidence_complete={ou_v4_evidence_complete};"
                f"ou_v4_supersession={ou_v4_supersession_gate.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_review_packet={ou_v4_review_packet.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_review_packet_id={ou_v4_review_packet.get('review_packet_id', '')};"
                f"ou_v4_supreme_review={ou_v4_supreme_review.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_supreme_recommendation={ou_v4_supreme_review.get('recommendation', '')};"
                f"ou_v4_activation={ou_v4_activation.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_generation={4 if ou_v4_activation.get('status') == 'APPLIED_RESEARCH_COMPARATOR_ONLY' else 1};"
                f"ou_v4_proof_refresh={ou_v4_proof_refresh.get('status', 'NOT_CONFIGURED')};"
                f"ou_v4_exact_rows={int(ou_v4_proof_refresh.get('exact_ou_rows', 0) or 0)}/"
                f"{int(ou_v4_proof_refresh.get('captured_ou_rows', 0) or 0)};"
                f"ou_v5_registration={ou_v5_receipt.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_capture={ou_v5_capture.get('status', 'NOT_CAPTURED')};"
                f"ou_v5_responses={int(ou_v5_capture.get('responses_available', 0) or 0)}/"
                f"{int(ou_v5_capture.get('required_responses', 8) or 8)};"
                f"ou_v5_evaluation={ou_v5_evaluation.get('status', 'NOT_EVALUATED')};"
                f"ou_v5_failure_attribution={ou_v5_failure_attribution.get('status', 'NOT_BUILT')};"
                f"ou_v5_failure_attribution_id={ou_v5_failure_attribution.get('attribution_id', '')};"
                f"ou_v5_cells={int(ou_v5_evaluation.get('passed_cells', 0) or 0)}/"
                f"{int(ou_v5_evaluation.get('required_cells', 8) or 8)};"
                f"ou_v5_transform_cells={int(ou_v5_evaluation.get('transform_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v5_trend_cells={int(ou_v5_evaluation.get('trend_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v5_profile_branch_cells={int(ou_v5_evaluation.get('profile_branch_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v5_formula_cells={int(ou_v5_evaluation.get('formula_parity_passed_cells', 0) or 0)};"
                f"ou_v5_immutable_evidence_complete={ou_v5_evidence_complete};"
                f"ou_v5_supersession={ou_v5_supersession_gate.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_review_packet={ou_v5_review_packet.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_review_packet_id={ou_v5_review_packet.get('review_packet_id', '')};"
                f"ou_v5_supreme_review={ou_v5_supreme_review.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_supreme_recommendation={ou_v5_supreme_review.get('recommendation', '')};"
                f"ou_v5_activation={ou_v5_activation.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_generation={5 if ou_v5_activation.get('status') == 'APPLIED_RESEARCH_COMPARATOR_ONLY' else 1};"
                f"ou_v5_proof_refresh={ou_v5_proof_refresh.get('status', 'NOT_CONFIGURED')};"
                f"ou_v5_exact_rows={int(ou_v5_proof_refresh.get('exact_ou_rows', 0) or 0)}/"
                f"{int(ou_v5_proof_refresh.get('captured_ou_rows', 0) or 0)};"
                f"ou_v6_registration={ou_v6_receipt.get('status', 'NOT_CONFIGURED')};"
                f"ou_v6_capture={ou_v6_capture.get('status', 'NOT_CAPTURED')};"
                f"ou_v6_responses={int(ou_v6_capture.get('responses_available', 0) or 0)}/"
                f"{int(ou_v6_capture.get('required_responses', 8) or 8)};"
                f"ou_v6_evaluation={ou_v6_evaluation.get('status', 'NOT_EVALUATED')};"
                f"ou_v6_cells={int(ou_v6_evaluation.get('passed_cells', 0) or 0)}/"
                f"{int(ou_v6_evaluation.get('required_cells', 8) or 8)};"
                f"ou_v6_transform_cells={int(ou_v6_evaluation.get('transform_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v6_trend_cells={int(ou_v6_evaluation.get('trend_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v6_profile_branch_cells={int(ou_v6_evaluation.get('profile_branch_selector_parity_passed_cells', 0) or 0)};"
                f"ou_v6_formula_cells={int(ou_v6_evaluation.get('formula_parity_passed_cells', 0) or 0)};"
                f"ou_v6_immutable_evidence_complete={ou_v6_evidence_complete};"
                f"ou_v6_final_successor_iteration={bool(ou_v6_receipt.get('final_successor_iteration', False))};"
                f"ou_v6_successor_after_failure_allowed={bool(ou_v6_receipt.get('successor_after_v6_failure_allowed', False))};"
                f"ou_v6_activation={ou_v6_activation.get('status', wizard_proof_scheduler.get('ou_v6_activation_status', 'NOT_CONFIGURED'))};"
                f"ou_v6_generation={int(wizard_proof_scheduler.get('ou_v6_comparator_generation', 1) or 1)};"
                f"ou_v6_proof_refresh={ou_v6_proof_refresh.get('status', wizard_proof_scheduler.get('ou_v6_proof_refresh_status', 'NOT_CONFIGURED'))};"
                f"ou_v6_terminal_failure={bool(wizard_proof_scheduler.get('ou_v6_terminal_failure', False) or ou_v6_evaluation.get('status') == 'FAIL')};"
                f"copula_behavioral={copula_behavioral.get('status', 'NOT_CONFIGURED')};"
                f"copula_behavioral_cells={int(copula_behavioral.get('behavioral_cells_passed', 0) or 0)}/"
                f"{int(copula_behavioral.get('expected_cells', 4) or 4)};"
                f"copula_provenance_cells={int(copula_behavioral.get('provenance_cells_complete', 0) or 0)}/"
                f"{int(copula_behavioral.get('expected_cells', 4) or 4)};"
                f"copula_behavioral_parity={bool(copula_behavioral.get('behavioral_parity_proven', False))};"
                f"copula_formula_parity={bool(copula_behavioral.get('formula_parity_proven', False))}"
            ),
            "blocker": wizard_stage_blocker or "vendor_formula_parity_unproven",
            "next_action": (
                "repair_wizard_surface_inventory_contract"
                if not wizard_surface_inventory_ready
                else "capture_fresh_authenticated_scanner_and_pair_route_receipts"
                if not wizard_browser_auth_ready
                else "repair_scheduler_runtime_contract_before_external_calls"
                if not scheduler_runtime_ready
                else "repair_frozen_capture_manifest_continuity_before_external_calls"
                if not capture_manifest_continuity_pass
                else "repair_final_immutable_scheduler_execution_receipt"
                if mutable_stage3_evidence_complete
                and not immutable_wizard_execution_stage3_complete
                else ou_v6_next_action
                if ou_v6_next_action
                else "review_ou_v4_failure_attribution_without_activation"
                if not ou_v6_required
                and not ou_v5_required
                and ou_v4_required
                and ou_v4_evaluation.get("status") == "FAIL"
                else ou_v5_next_action
                if not ou_v6_required and ou_v5_next_action
                else "run_governed_ou_v4_holdout_after_utc_reset"
                if not ou_v6_required
                and not ou_v5_required
                and ou_v4_required
                and not ou_v4_evidence_complete
                else "human_review_ou_v4_exact_packet_and_apply_research_comparator"
                if not ou_v6_required
                and not ou_v5_required
                and ou_v4_required
                and ou_v4_review_packet.get("status") == "READY_FOR_EXPLICIT_REVIEW"
                and ou_v4_supreme_review.get("status") == "PASS_ADVISORY_ONLY"
                and ou_v4_activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY"
                else "refresh_ou_v4_activation_bound_current_proof_rows"
                if not ou_v6_required
                and not ou_v5_required
                and ou_v4_required
                and ou_v4_activation.get("status") == "APPLIED_RESEARCH_COMPARATOR_ONLY"
                and ou_v4_proof_refresh.get("status") != "PASS"
                else _next_stage_three_action(
                    comparator_review_status=str(
                        wizard_comparator_review.get("status", "NOT_AUDITED")
                    ),
                    scheduler_status=proof_scheduler_status,
                    queue_completed=proof_queue_completed,
                    queue_eligible=proof_queue_eligible,
                    responses_captured=proof_responses_captured,
                )
            ),
            "evidence_path": (
                "reports/active/wizard_mode_parity.csv;"
                "config/wizard_surface_inventory_contract.json;"
                "reports/active/wizard_surface_inventory_checks.csv;"
                "reports/active/wizard_surface_integration_matrix.csv;"
                "reports/active/wizard_surface_inventory_readiness.json;"
                "data/research/wizard_surface_inventory_readiness;"
                "config/wizard_browser_auth_contract.json;"
                "reports/active/wizard_browser_auth_checks.csv;"
                "reports/active/wizard_browser_auth_readiness.json;"
                "data/research/wizard_browser_auth_readiness;"
                "reports/active/wizard_ou_optimal_overlay_provenance.csv;"
                "reports/active/exhaustive_wizard_exact_mode_proof_queue.csv;"
                "reports/active/exhaustive_wizard_mode_proof_input_audit.json;"
                "reports/active/corrective_wizard_proof_scheduler_status.json;"
                "reports/active/corrective_wizard_proof_launcher_status.json;"
                "data/research/wizard_proof_scheduler_receipts;"
                "reports/active/corrective_wizard_next_capture_manifest.csv;"
                "reports/active/corrective_wizard_next_capture_manifest.json;"
                "reports/active/corrective_wizard_next_capture_manifest.md;"
                "reports/active/wizard_reset_readiness_checks.csv;"
                "reports/active/wizard_reset_readiness.json;"
                "reports/active/wizard_reset_readiness.md;"
                "reports/active/scheduler_runtime_readiness_checks.csv;"
                "reports/active/scheduler_runtime_readiness.json;"
                "reports/active/scheduler_runtime_readiness.md;"
                "data/research/scheduler_runtime_readiness;"
                "reports/active/wizard_comparator_review_queue.csv;"
                "reports/active/wizard_comparator_review_control.json;"
                "reports/active/wizard_comparator_review_control.md;"
                "reports/supreme_team/wizard_comparator_review_checkpoint.json;"
                "reports/supreme_team/wizard_comparator_review_checkpoint.md;"
                "reports/active/corrective_wizard_capture_reconciliation.csv;"
                "reports/active/corrective_wizard_capture_reconciliation.json;"
                "reports/active/corrective_wizard_capture_reconciliation.md;"
                "reports/active/wizard_dynamic_v2_holdout_status.json;"
                "reports/active/wizard_dynamic_v2_holdout_evaluation.csv;"
                "reports/active/wizard_dynamic_v2_supersession_gate.json;"
                "reports/active/wizard_dynamic_v2_activation_status.json;"
                "reports/active/wizard_dynamic_v2_proof_refresh_status.json;"
                "reports/active/wizard_dynamic_v2_review_packet.json;"
                "reports/active/wizard_dynamic_v2_review_packet.md;"
                "reports/supreme_team/wizard_dynamic_v2_review.json;"
                "reports/supreme_team/wizard_dynamic_v2_review.md;"
                "config/wizard_ou_trend_selector_v1_holdout.json;"
                "reports/active/wizard_ou_trend_selector_v1_receipt.json;"
                "reports/active/wizard_ou_trend_selector_v1_derivation.csv;"
                "reports/active/wizard_ou_trend_selector_v1_predictions.csv;"
                "reports/active/wizard_ou_v3_capture_status.json;"
                "reports/active/wizard_ou_v3_holdout_status.json;"
                "reports/active/wizard_ou_v3_holdout_evaluation.csv;"
                "reports/active/wizard_ou_v3_supersession_gate.json;"
                "reports/active/wizard_ou_v3_review_packet.json;"
                "reports/active/wizard_ou_v3_review_packet.md;"
                "reports/active/wizard_ou_v3_activation_status.json;"
                "reports/active/wizard_ou_v3_proof_refresh_status.json;"
                "config/wizard_ou_comparator_v4_holdout.json;"
                "reports/active/wizard_ou_v4_holdout_receipt.json;"
                "reports/active/wizard_ou_v4_derivation.csv;"
                "reports/active/wizard_ou_v4_predictions.csv;"
                "reports/active/wizard_ou_v4_capture_status.json;"
                "reports/active/wizard_ou_v4_holdout_status.json;"
                "reports/active/wizard_ou_v4_holdout_evaluation.csv;"
                "reports/active/wizard_ou_v4_supersession_gate.json;"
                "reports/active/wizard_ou_v4_review_packet.json;"
                "reports/active/wizard_ou_v4_review_packet.md;"
                "reports/supreme_team/wizard_ou_v4_review.json;"
                "reports/supreme_team/wizard_ou_v4_review.md;"
                "reports/active/wizard_ou_v4_activation_status.json;"
                "reports/active/wizard_ou_v4_proof_refresh_status.json;"
                "data/research/wizard_ou_v4_holdout_evaluations;"
                "data/research/wizard_ou_v4_review_packets;"
                "data/research/wizard_ou_v4_supreme_reviews;"
                "data/research/wizard_ou_v4_comparator_activations;"
                "config/wizard_ou_comparator_v5_holdout.json;"
                "reports/active/wizard_ou_v5_holdout_receipt.json;"
                "reports/active/wizard_ou_v5_derivation.csv;"
                "reports/active/wizard_ou_v5_predictions.csv;"
                "reports/active/wizard_ou_v5_capture_status.json;"
                "reports/active/wizard_ou_v5_holdout_status.json;"
                "reports/active/wizard_ou_v5_holdout_evaluation.csv;"
                "reports/active/wizard_ou_v5_failure_attribution.csv;"
                "reports/active/wizard_ou_v5_failure_attribution.json;"
                "reports/supreme_team/wizard_ou_v5_failure_checkpoint.md;"
                "reports/active/wizard_ou_v5_supersession_gate.json;"
                "reports/active/wizard_ou_v5_review_packet.json;"
                "reports/active/wizard_ou_v5_review_packet.md;"
                "reports/supreme_team/wizard_ou_v5_review.json;"
                "reports/supreme_team/wizard_ou_v5_review.md;"
                "reports/active/wizard_ou_v5_activation_status.json;"
                "reports/active/wizard_ou_v5_proof_refresh_status.json;"
                "data/research/wizard_ou_v5_holdout_evaluations;"
                "data/research/wizard_ou_v5_failure_attributions;"
                "data/research/wizard_ou_v5_review_packets;"
                "data/research/wizard_ou_v5_supreme_reviews;"
                "data/research/wizard_ou_v5_comparator_activations;"
                "config/wizard_ou_comparator_v6_holdout.json;"
                "reports/active/wizard_ou_v6_holdout_receipt.json;"
                "reports/active/wizard_ou_v6_derivation.csv;"
                "reports/active/wizard_ou_v6_predictions.csv;"
                "reports/active/wizard_ou_v6_capture_status.json;"
                "reports/active/wizard_ou_v6_holdout_status.json;"
                "reports/active/wizard_ou_v6_holdout_evaluation.csv;"
                "reports/active/wizard_ou_v6_activation_status.json;"
                "reports/active/wizard_ou_v6_proof_refresh_status.json;"
                "data/research/wizard_ou_v6_holdout;"
                "data/research/wizard_ou_v6_holdout_evaluations;"
                "reports/active/wizard_copula_behavioral_status.json;"
                "reports/active/wizard_copula_behavioral_evaluation.csv;"
                "reports/active/wizard_copula_behavioral_v2_status.json;"
                "reports/active/wizard_copula_behavioral_v2_evaluation.csv;"
                "reports/active/wizard_copula_behavioral_v2_contract_receipt.json;"
                "reports/active/exhaustive_wizard_copula_proof_queue_v2.csv;"
                "reports/active/wizard_copula_source_ledger_v2.csv;"
                "reports/active/wizard_credit_budget_contract.json;"
                "reports/active/wizard_credit_ledger_status.json;"
                "data/research/wizard_credit_ledger"
            ),
        },
        {
            "stage": 4,
            "objective": "costed_statistical_acceptance_and_independent_breadth",
            "status": (
                "PASS"
                if stage_four_complete
                else "IN_PROGRESS"
                if stage_four_state["status"] == "IN_PROGRESS"
                or rerun_gate_status == "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"
                else "BLOCKED"
            ),
            "evidence_progress": (
                f"active_contract_candidates={int(statistics.get('registered_rerun_candidates', 0))};"
                f"pending_current_family_cohort_pairs={pending_candidate_pairs};"
                f"prospective_zero_weight_pairs={prospective_candidate_pairs};"
                f"current_family_target_hypotheses={int(stage_four_state['target_hypotheses'])};"
                f"current_family_accounted_hypotheses={int(stage_four_state['accounted_hypotheses'])};"
                f"current_family_unaccounted_hypotheses={int(stage_four_state['unaccounted_hypotheses'])};"
                f"active_contract_id={stage_four_state['active_contract_id']};"
                f"contract_generations={int(stage_four_state['contract_generations'])};"
                f"contract_chain_valid={bool(stage_four_state['contract_chain_valid'])};"
                f"active_contract_conclusive={bool(stage_four_state['active_contract_conclusive'])};"
                f"final_receipt_conclusion_bound={bool(stage_four_state['final_receipt_conclusion_bound'])};"
                f"frozen_minimum_independent_clusters={int(stage_four_state['minimum_independent_clusters'])};"
                f"stage4_terminal_outcome={stage_four_state['terminal_outcome']};"
                f"registered_gates_ready={int(statistics.get('registered_rerun_candidate_gates_ready', 0))};"
                f"pending_family_preflight={statistics.get('pending_family_preflight_status', 'NOT_EVALUATED')};"
                f"pending_family_ready={int(statistics.get('pending_family_ready', 0))}/"
                f"{int(statistics.get('pending_family_hypotheses', 0))};"
                f"pending_family_pairs={int(statistics.get('pending_family_pairs', 0))};"
                f"pending_family_clusters={int(statistics.get('pending_family_independent_clusters', 0))};"
                f"pending_family_rollover_required="
                f"{bool(statistics.get('pending_family_rollover_required', False))};"
                f"rerun_gate={rerun_gate_status};"
                f"rerun_results_accounted={bool(statistics.get('registered_rerun_results_accounted', False))};"
                f"rerun_conclusion={rerun_conclusion};"
                f"execution_conclusion={execution_conclusion};"
                f"registered_executor={registered_execution.get('status', 'NOT_EVALUATED')};"
                f"stage4_handoff_readiness={stage4_handoff_readiness.get('status', 'NOT_AUDITED')};"
                f"stage4_handoff_state={stage4_handoff_readiness.get('handoff_state', 'NOT_AUDITED')};"
                f"stage4_handoff_receipt_id={stage4_handoff_readiness.get('receipt_id', '')};"
                f"stage4_handoff_checked_at={stage4_handoff_readiness.get('checked_at_utc', '')};"
                f"stage4_handoff_checks={int(stage4_handoff_readiness.get('checks_passed', 0) or 0)}/"
                f"{int(stage4_handoff_readiness.get('checks_total', 0) or 0)};"
                f"independent_clusters={int(statistics.get('independent_supporting_clusters', 0))};"
                f"independent_full_survivor_clusters="
                f"{int(statistics.get('independent_full_survivor_clusters', 0))};"
                f"independent_pairs={int(statistics.get('independent_supporting_pairs', 0))};"
                f"independent_full_survivor_pairs="
                f"{int(statistics.get('independent_full_survivor_pairs', 0))};"
                f"final_survivors={final_survivors}"
            ),
            "blocker": (
                ""
                if stage_four_complete
                else _join_blockers(
                    [
                        str(stage_four_state["blocker"]),
                        (
                            "pending_family_non_vendor_preflight_failed"
                            if statistics.get("pending_family_preflight_status")
                            not in {None, "PASS"}
                            else ""
                        ),
                        (
                            "stage4_handoff_readiness_failed:"
                            + ",".join(
                                str(value)
                                for value in stage4_handoff_readiness.get("blockers", [])
                                if str(value)
                            )
                            if stage4_handoff_readiness
                            and stage4_handoff_readiness.get("status")
                            != "PASS_STAGE4_HANDOFF_READY"
                            else ""
                        ),
                        *statistics.get("blockers", []),
                    ]
                )
            ),
            "next_action": str(stage_four_state["next_action"]),
            "evidence_path": (
                "reports/active/corrective_research_funnel.csv;"
                "reports/active/final_1x_survivor_receipt.json;"
                "reports/active/acceptance_policy_receipt.json;"
                "reports/active/holdout_policy_receipt.json;"
                "reports/active/registered_research_rerun_contract.json;"
                "reports/active/registered_research_rerun_gate.csv;"
                "reports/active/registered_research_rerun_gate.json;"
                "reports/active/registered_rerun_family_preflight.csv;"
                "reports/active/registered_rerun_family_preflight.json;"
                "reports/active/stage4_handoff_readiness_checks.csv;"
                "reports/active/stage4_handoff_readiness.json;"
                "reports/active/stage4_handoff_readiness.md;"
                "reports/active/registered_research_rerun_execution.json;"
                "data/research/registered_rerun_contracts;"
                "data/research/stage4_handoff_readiness;"
                "data/research/registered_rerun_executions;"
                "data/research/registered_rerun_conclusions"
            ),
        },
        {
            "stage": 5,
            "objective": "leakage_safe_ml_and_rl_out_of_sample_acceptance",
            "status": stage_five_status,
            "evidence_progress": (
                f"registered_learning={registered_learning.get('status', 'NOT_EVALUATED')!s};"
                f"registered_execution={registered_learning.get('registered_execution_id', '')!s};"
                f"registered_learning_audit={registered_learning_audit.get('status', 'NOT_AUDITED')!s};"
                f"registered_learning_evidence_valid={stage_five_evidence_valid};"
                f"registered_learning_receipt={registered_learning_audit.get('learning_id', '')!s};"
                f"stage5_protocol={registered_stage5_protocol.get('status', 'NOT_REGISTERED')!s};"
                f"stage5_protocol_id={registered_stage5_protocol.get('protocol_id', '')!s};"
                f"stage5_research_gate_pass={stage_five_pass};"
                f"model_authority={learning.get('model_authority', 'RESEARCH_ONLY')!s};"
                f"active_dataset={learning.get('active_dataset_id', '')!s};"
                f"model_lineage={bool(learning.get('model_active_dataset_lineage_matches', False))};"
                f"model_hash={bool(learning.get('model_artifact_hash_matches', False))};"
                f"model_take_rate={learning.get('model_median_take_rate', '')!s};"
                f"exact_mode_provenance={bool(learning.get('exact_mode_trade_provenance_ready', False))};"
                f"strict_cost_training={bool(learning.get('strict_cost_training_evidence_ready', False))};"
                f"rl_global_purge={bool(learning.get('rl_global_label_purge_ready', False))};"
                f"rl_lineage={bool(learning.get('rl_active_dataset_lineage_matches', False))};"
                f"rl_validation={bool(learning.get('rl_validation_passed', False))};"
                f"rl_held_out_test={bool(learning.get('rl_held_out_test_passed', False))}"
            ),
            "blocker": stage_five_blocker,
            "next_action": (
                "advance_to_testnet_candidate_review_without_order_authority"
                if stage_five_pass
                else "register_stage5_protocol_before_stage4_completion"
                if not stage5_protocol_pass
                else "repair_registered_learning_immutable_evidence"
                if registered_learning.get("status")
                in {
                    "PASS_RESEARCH_LEARNING_GATES",
                    "REJECTED_RESEARCH_LEARNING_GATES",
                    "ALREADY_COMPLETE",
                }
                and not stage_five_evidence_valid
                else "complete_registered_stage4_before_stage5_learning"
                if registered_learning.get("status") == "BLOCKED_STAGE4" and not stage_four_complete
                else "register_next_stage4_survivor_family_after_verified_stage5_rejection"
                if stage_five_evidence_valid
                else "run_registered_learning_from_immutable_stage4_receipt;"
                "repair_model_economics_score_monotonicity_and_timeframe_support;"
                "rerun_rl_only_on_registered_strict_cost_dataset"
            ),
            "evidence_path": (
                "reports/active/registered_learning_research_status.json;"
                "reports/active/registered_stage5_protocol.json;"
                "data/research/registered_stage5_protocols;"
                "data/research/registered_learning;reports/active/model_authority_status.json;"
                "reports/active/learning_label_audit.json;reports/rl/rl_acceptance_report.csv;"
                "reports/rl/rl_split_audit.csv"
            ),
        },
        {
            "stage": 6,
            "objective": "hyperliquid_testnet_lifecycle_and_realized_sample",
            "status": "PASS" if stage_six_pass else "BLOCKED",
            "evidence_progress": (
                f"candidate={candidate_status};"
                f"candidate_authority_safe={candidate_authority_safe};"
                f"prospective_cohort={prospective_testnet_status};"
                f"prospective_pairs={prospective_testnet_ready_pairs}/{prospective_testnet_pairs};"
                f"prospective_authority_safe={prospective_testnet_authority_safe};"
                f"inventory_rows={len(testnet_inventory)};"
                f"tradable_perps={inventory_tradable};"
                f"inventory_fetch_blocked={inventory_fetch_blocked};"
                f"wallet_agent_preflight={account_preflight_ready};"
                f"margin_status={margin_status};"
                f"perp_account_value_usd={margin_row.get('account_value_usd', '')!s};"
                f"withdrawable_usd={margin_row.get('withdrawable_usd', '')!s};"
                f"spot_usdc_usd={margin_row.get('spot_usdc_usd', '')!s};"
                f"collateral_required={collateral_required};"
                f"collateral_preflight={collateral_preflight_status};"
                f"collateral_agent_key_accessed={bool(testnet_collateral_preflight.get('agent_key_accessed', False))};"
                f"collateral_transfer_attempted={bool(testnet_collateral_preflight.get('transfer_attempted', False))};"
                f"no_order_checks={no_order_passed}/{no_order_checks};"
                f"lifecycle={lifecycle_status};"
                f"lifecycle_authority_safe={lifecycle_authority_safe};"
                f"sample_release_status={release.get('testnet_sample_status', 'MISSING')!s};"
                f"sample_artifact_pass={sample_artifact_pass}"
            ),
            "blocker": "" if stage_six_pass else _join_blockers(stage_six_blockers),
            "next_action": stage_six_next_action,
            "evidence_path": (
                "reports/active/hyperliquid_testnet_market_inventory.csv;"
                "reports/active/testnet_prospective_cohort_readiness.csv;"
                "reports/active/testnet_prospective_cohort_readiness.json;"
                "reports/active/hyperliquid_testnet_preflight.csv;"
                "reports/active/hyperliquid_testnet_margin_snapshot.csv;"
                "reports/active/testnet_collateral_transfer_preflight.json;"
                "reports/active/testnet_collateral_transfer_execution.json;"
                "reports/active/testnet_candidate_receipt.json;"
                "reports/active/testnet_no_order_preflight.csv;"
                "reports/active/testnet_lifecycle_execution_receipt.json;"
                "data/testnet/lifecycle_index.csv;"
                "reports/active/realized_testnet_sample_sufficiency.csv;"
                "reports/supreme_team/testnet_evidence_checkpoint.json"
            ),
        },
        {
            "stage": 7,
            "objective": "minimal_live_canary_after_every_gate",
            "status": "PASS" if stage_seven_pass else "BLOCKED",
            "evidence_progress": (
                f"authorization={live_authorization.get('authorization_status', 'MISSING')!s};"
                f"manual_executor={bool(live_authorization.get('manual_executor_available', False))};"
                f"user_authorization={bool(live_authorization.get('user_authorization_present', False))};"
                f"outcome={live_outcome.get('canary_status', 'MISSING')!s};"
                f"orders_submitted={int(live_outcome.get('orders_submitted', 0) or 0)};"
                f"fills_observed={int(live_outcome.get('fills_observed', 0) or 0)};"
                f"reconciled_flat={bool(live_outcome.get('reconciled_flat', False))};"
                f"repeat_authorized={bool(live_outcome.get('repeat_authorized', False))};"
                f"scaling_authorized={bool(live_outcome.get('scaling_authorized', False))};"
                f"ongoing_live_authority={bool(live_outcome.get('live_trading_authorized', False))}"
            ),
            "blocker": "" if stage_seven_pass else _join_blockers(stage_seven_blockers),
            "next_action": stage_seven_next_action,
            "evidence_path": (
                "config/live_canary_policy.json;"
                "reports/active/testnet_live_input_parity.csv;"
                "reports/active/live_canary_user_approval.json;"
                "reports/active/live_canary_authorization.json;"
                "reports/active/live_canary_execution_receipt.json;"
                "reports/supreme_team/live_canary_post_canary_review.json;"
                "reports/active/live_canary_outcome_evaluation.json"
            ),
        },
    ]
    frame = pd.DataFrame(rows)
    frame["testnet_order_authority"] = False
    frame["live_trading_authorized"] = False
    csv_path = root / "reports" / "active" / "seven_stage_goal_checkpoint.csv"
    md_path = root / "reports" / "active" / "seven_stage_goal_checkpoint.md"
    _atomic_csv(frame, csv_path)
    md_path.write_text(
        "# Seven-Stage Goal Checkpoint\n\n"
        + frame.to_markdown(index=False)
        + "\n\nNo Testnet or live orders are authorized by this checkpoint.\n",
        encoding="utf-8",
    )
    return csv_path, md_path


def _capture_reconciliation_pass(evidence: dict[str, Any]) -> bool:
    try:
        required = int(evidence.get("required_calls", 0) or 0)
        completed = int(evidence.get("completed_calls", 0) or 0)
        pending = int(evidence.get("pending_calls", 0) or 0)
        blocked = int(evidence.get("blocked_calls", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        str(evidence.get("status", "")) == "PASS"
        and required > 0
        and completed == required
        and pending == 0
        and blocked == 0
    )


def _capture_manifest_continuity_pass(evidence: dict[str, Any]) -> bool:
    manifest_id = str(evidence.get("capture_manifest_id", "")).strip()
    source_receipt_id = str(evidence.get("capture_manifest_source_receipt_id", "")).strip()
    source_receipt_path = str(evidence.get("capture_manifest_source_receipt_path", "")).strip()
    source_receipt_sha256 = (
        str(evidence.get("capture_manifest_source_receipt_sha256", "")).strip().lower()
    )
    source_artifacts_sha256 = (
        str(evidence.get("capture_manifest_source_artifacts_sha256", "")).strip().lower()
    )

    def valid_digest(value: str) -> bool:
        return len(value) == 64 and all(character in "0123456789abcdef" for character in value)

    return bool(
        evidence.get("capture_manifest_continuity_valid") is True
        and evidence.get("capture_manifest_candidate_binding_valid") is True
        and evidence.get("capture_manifest_candidate_source_binding_valid") is True
        and evidence.get("capture_manifest_drift_detected") is False
        and str(evidence.get("capture_manifest_continuity_status", "")).startswith("PASS_")
        and manifest_id.startswith("wizardcapture_")
        and source_receipt_id.startswith("wizardcapturesources_")
        and source_receipt_path
        == f"data/research/wizard_capture_manifest_sources/{manifest_id}.json"
        and valid_digest(source_receipt_sha256)
        and valid_digest(source_artifacts_sha256)
    )


_PENDING_SEMANTIC_IDENTITY_COLUMNS = (
    "equivalence_cluster_id",
    "pair",
    "wizard_timeframe",
    "exact_mode",
    "orientation",
)


def _pending_semantic_hypothesis_ids(
    pending_candidate_cohort: pd.DataFrame,
) -> tuple[set[str], bool]:
    """Collapse consistent experiment rows without hiding identity conflicts."""

    semantic_column = "semantic_hypothesis_id"
    if pending_candidate_cohort.empty:
        return set(), True
    if semantic_column not in pending_candidate_cohort:
        return set(), False

    normalized = pending_candidate_cohort[semantic_column].map(
        lambda value: str(value).strip() if pd.notna(value) else ""
    )
    pending_ids = {value for value in normalized if value}
    if normalized.eq("").any():
        return pending_ids, False

    duplicate_ids = set(normalized[normalized.duplicated(keep=False)])
    if not duplicate_ids:
        return pending_ids, True
    if any(column not in pending_candidate_cohort for column in _PENDING_SEMANTIC_IDENTITY_COLUMNS):
        return pending_ids, False

    cohort = pending_candidate_cohort.assign(_semantic_id=normalized)
    for semantic_id in duplicate_ids:
        rows = cohort.loc[cohort["_semantic_id"] == semantic_id]
        identities = {
            tuple(str(value).strip() if pd.notna(value) else "" for value in row)
            for row in rows.loc[:, _PENDING_SEMANTIC_IDENTITY_COLUMNS].itertuples(
                index=False, name=None
            )
        }
        if len(identities) != 1 or any(not value for value in next(iter(identities))):
            return pending_ids, False
    return pending_ids, True


def _stage_four_completion_state(
    *,
    root: Path,
    pending_candidate_cohort: pd.DataFrame,
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Require immutable whole-cohort accounting before Stage 4 can finish."""

    policy_path = root / "config" / "acceptance_policy_manifest.json"
    holdout_policy_path = root / "config" / "research_holdout_policy.json"
    acceptance_policy = _read_json_fail_closed(policy_path)
    acceptance_receipt = _read_json_fail_closed(
        root / "reports" / "active" / "acceptance_policy_receipt.json"
    )
    holdout_receipt = _read_json_fail_closed(
        root / "reports" / "active" / "holdout_policy_receipt.json"
    )
    try:
        minimum_clusters = int(
            acceptance_policy.get("research_gates", {}).get(
                "minimum_independent_supporting_clusters", 0
            )
        )
    except (TypeError, ValueError):
        minimum_clusters = 0
    policy_lineage_valid = bool(
        acceptance_policy.get("schema_version") == "thewiz.acceptance_policy.v1"
        and minimum_clusters > 0
        and acceptance_receipt.get("status") == "PASS"
        and acceptance_receipt.get("policy_path") == "config/acceptance_policy_manifest.json"
        and acceptance_receipt.get("policy_sha256") == _file_sha256(policy_path)
        and bool(acceptance_receipt.get("source_contracts_match", False))
        and not bool(acceptance_receipt.get("testnet_order_authority", False))
        and not bool(acceptance_receipt.get("live_trading_authorized", False))
        and holdout_receipt.get("status") == "PASS"
        and holdout_receipt.get("policy_path") == "config/research_holdout_policy.json"
        and holdout_receipt.get("policy_sha256") == _file_sha256(holdout_policy_path)
        and bool(holdout_receipt.get("final_holdout_locked", False))
        and not bool(holdout_receipt.get("testnet_order_authority", False))
        and not bool(holdout_receipt.get("live_trading_authorized", False))
    )

    pending_ids, pending_ids_valid = _pending_semantic_hypothesis_ids(pending_candidate_cohort)
    accounting = _registered_current_family_accounting(
        root=root,
        pending_ids=pending_ids,
        acceptance_policy_id=str(acceptance_receipt.get("policy_id", "")),
        holdout_policy_id=str(holdout_receipt.get("policy_id", "")),
    )

    final_receipt_path = root / "reports" / "active" / "final_1x_survivor_receipt.json"
    final_receipt = _read_json_fail_closed(final_receipt_path)
    active_conclusion = _read_json_fail_closed(
        root
        / "data"
        / "research"
        / "registered_rerun_conclusions"
        / f"{accounting['active_contract_id']}.json"
    )
    final_receipt_conclusion_bound = bool(
        not accounting["active_contract_conclusive"]
        or (
            active_conclusion.get("contract_id") == accounting["active_contract_id"]
            and active_conclusion.get("final_survivor_receipt_sha256")
            == _file_sha256(final_receipt_path)
        )
    )
    summary_survivors = int(statistics.get("final_one_x_survivors", 0) or 0)
    receipt_survivors = int(final_receipt.get("final_one_x_survivors", 0) or 0)
    independent_clusters = int(statistics.get("independent_supporting_clusters", 0) or 0)
    full_survivor_clusters = int(statistics.get("independent_full_survivor_clusters", 0) or 0)
    independent_pairs = int(statistics.get("independent_supporting_pairs", 0) or 0)
    full_survivor_pairs = int(statistics.get("independent_full_survivor_pairs", 0) or 0)
    expected_receipt_status = "PASS" if receipt_survivors > 0 else "ZERO_SURVIVORS"
    final_receipt_lineage_valid = bool(
        final_receipt.get("schema_version") == "thewiz.final_one_x_survivor_receipt.v1"
        and final_receipt.get("acceptance_policy_id") == acceptance_receipt.get("policy_id")
        and final_receipt.get("holdout_policy_id") == holdout_receipt.get("policy_id")
        and final_receipt.get("receipt_status") == expected_receipt_status
        and final_receipt_conclusion_bound
        and receipt_survivors == summary_survivors
        and int(final_receipt.get("independent_supporting_clusters", 0) or 0)
        == independent_clusters
        and int(final_receipt.get("independent_full_survivor_clusters", 0) or 0)
        == full_survivor_clusters
        and int(final_receipt.get("independent_supporting_pairs", 0) or 0) == independent_pairs
        and int(final_receipt.get("independent_full_survivor_pairs", 0) or 0) == full_survivor_pairs
        and isinstance(final_receipt.get("final_canonical_pairs", []), list)
        and len(final_receipt.get("final_canonical_pairs", [])) == full_survivor_pairs
        and (
            (
                receipt_survivors > 0
                and not final_receipt.get("blockers", [])
                and bool(final_receipt.get("testnet_candidate_authority", False))
                and len(final_receipt.get("final_experiment_ids", [])) == receipt_survivors
            )
            or (
                receipt_survivors == 0
                and not bool(final_receipt.get("testnet_candidate_authority", False))
            )
        )
        and not bool(final_receipt.get("testnet_order_authority", False))
        and not bool(final_receipt.get("live_trading_authorized", False))
    )
    whole_cohort_accounted = bool(
        policy_lineage_valid
        and pending_ids_valid
        and accounting["contract_chain_valid"]
        and accounting["all_target_hypotheses_accounted"]
        and final_receipt_lineage_valid
    )
    breadth_met = bool(
        minimum_clusters > 0
        and independent_clusters >= minimum_clusters
        and full_survivor_clusters >= minimum_clusters
        and independent_pairs >= minimum_clusters
        and full_survivor_pairs >= minimum_clusters
        and summary_survivors >= minimum_clusters
    )
    accepted_completion = bool(
        whole_cohort_accounted
        and accounting["accepted_hypotheses"] > 0
        and summary_survivors > 0
        and breadth_met
    )
    conclusive_rejection = bool(
        whole_cohort_accounted
        and not accepted_completion
        and (summary_survivors == 0 or not breadth_met)
    )
    complete = bool(accepted_completion or conclusive_rejection)
    terminal_outcome = (
        "ACCEPTED_VALID_INDEPENDENT_SURVIVORS"
        if accepted_completion
        else "CONCLUSIVE_REJECTION_CURRENT_FAMILY"
        if conclusive_rejection
        else "INCOMPLETE_CURRENT_FAMILY"
    )

    blockers: list[str] = []
    if not policy_lineage_valid:
        blockers.append("frozen_acceptance_or_holdout_policy_lineage_invalid")
    if not pending_ids_valid:
        blockers.append("pending_current_family_semantic_ids_invalid_or_duplicated")
    if not accounting["contract_chain_valid"]:
        blockers.append("registered_contract_chain_invalid")
    if accounting["invalid_conclusion_contracts"]:
        blockers.append(
            "invalid_or_missing_registered_conclusions:"
            + ",".join(accounting["invalid_conclusion_contracts"])
        )
    if accounting["unaccounted_ids"]:
        blockers.append(
            "current_family_hypotheses_unaccounted:" + str(len(accounting["unaccounted_ids"]))
        )
    if not final_receipt_lineage_valid:
        blockers.append("final_survivor_receipt_lineage_invalid_or_stale")
    if not final_receipt_conclusion_bound:
        blockers.append("final_survivor_receipt_not_bound_to_active_conclusion")
    if whole_cohort_accounted and summary_survivors > 0 and not breadth_met:
        blockers.append(
            f"independent_strategy_breadth_{independent_clusters}_of_{minimum_clusters}"
        )
    if (
        whole_cohort_accounted
        and summary_survivors == 0
        and accounting["accepted_hypotheses"] > 0
        and breadth_met
    ):
        blockers.append("registered_acceptance_and_final_survivor_receipt_disagree")

    gate_status = str(statistics.get("registered_rerun_gate_status", "NOT_EVALUATED"))
    if complete and accepted_completion:
        next_action = "run_registered_stage5_learning_from_immutable_stage4_survivors"
    elif complete:
        next_action = "close_current_family_as_conclusively_rejected_and_refresh_discovery"
    elif not accounting["active_contract_id"]:
        next_action = "materialize_first_registered_current_family_contract"
    elif not accounting["contract_chain_valid"]:
        next_action = "repair_registered_contract_chain_before_any_rerun"
    elif accounting["active_contract_conclusive"] and accounting["unaccounted_ids"]:
        next_action = (
            "advance_registered_rerun_to_unaccounted_current_family_cohort"
            if statistics.get("pending_family_preflight_status") == "PASS"
            else "repair_pending_family_non_vendor_preflight_before_rollover"
        )
    elif not accounting["active_contract_conclusive"]:
        next_action = (
            "execute_active_registered_rerun_after_ready_chain"
            if gate_status == "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"
            else str(
                statistics.get(
                    "registered_rerun_next_action",
                    "await_active_registered_rerun_evidence",
                )
            )
        )
    elif not final_receipt_lineage_valid:
        next_action = "refresh_final_survivor_receipt_from_accounted_registered_results"
    else:
        next_action = "repair_stage4_terminal_accounting_inconsistency"

    in_progress = bool(
        not complete
        and accounting["contract_chain_valid"]
        and accounting["active_contract_id"]
        and (
            accounting["active_contract_conclusive"]
            or gate_status == "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN"
        )
    )
    return {
        "status": "PASS" if complete else "IN_PROGRESS" if in_progress else "BLOCKED",
        "complete": complete,
        "terminal_outcome": terminal_outcome,
        "blocker": _join_blockers(blockers),
        "next_action": next_action,
        "minimum_independent_clusters": minimum_clusters,
        "independent_supporting_clusters": independent_clusters,
        "independent_full_survivor_clusters": full_survivor_clusters,
        "independent_supporting_pairs": independent_pairs,
        "independent_full_survivor_pairs": full_survivor_pairs,
        "final_one_x_survivors": summary_survivors,
        "policy_lineage_valid": policy_lineage_valid,
        "final_receipt_lineage_valid": final_receipt_lineage_valid,
        "final_receipt_conclusion_bound": final_receipt_conclusion_bound,
        "pending_ids_valid": pending_ids_valid,
        **accounting,
    }


def _registered_current_family_accounting(
    *,
    root: Path,
    pending_ids: set[str],
    acceptance_policy_id: str,
    holdout_policy_id: str,
) -> dict[str, Any]:
    active_path = root / "reports" / "active" / "registered_research_rerun_contract.json"
    active = _read_json_fail_closed(active_path)
    active_contract_id = str(active.get("contract_id", "")).strip()
    contracts: list[dict[str, Any]] = []
    chain_valid = bool(active_contract_id)
    visited: set[str] = set()
    current_id = active_contract_id
    while current_id:
        if current_id in visited or len(visited) >= 100:
            chain_valid = False
            break
        visited.add(current_id)
        immutable_path = (
            root / "data" / "research" / "registered_rerun_contracts" / f"{current_id}.json"
        )
        contract = _read_json_fail_closed(immutable_path)
        if not contract or contract.get("contract_id") != current_id:
            chain_valid = False
            break
        if current_id == active_contract_id and contract != active:
            chain_valid = False
            break
        if not _registered_contract_identity_valid(
            contract,
            acceptance_policy_id=acceptance_policy_id,
            holdout_policy_id=holdout_policy_id,
        ):
            chain_valid = False
            break
        contracts.append(contract)
        current_id = str(contract.get("prior_contract_id", "")).strip()

    target_ids = set(pending_ids)
    accounted_outcomes: dict[str, str] = {}
    conflicting_outcome_ids: set[str] = set()
    invalid_conclusion_contracts: list[str] = []
    valid_conclusion_contracts: list[str] = []
    for contract in contracts:
        contract_id = str(contract["contract_id"])
        candidate_ids = _registered_contract_candidate_ids(contract)
        target_ids.update(candidate_ids)
        conclusion = _read_json_fail_closed(
            root / "data" / "research" / "registered_rerun_conclusions" / f"{contract_id}.json"
        )
        if not contract_conclusively_accounted(contract, conclusion):
            invalid_conclusion_contracts.append(contract_id)
            continue
        valid_conclusion_contracts.append(contract_id)
        for semantic_id, outcome in conclusion["outcomes"].items():
            previous = accounted_outcomes.get(str(semantic_id))
            if previous is not None and previous != str(outcome):
                conflicting_outcome_ids.add(str(semantic_id))
            accounted_outcomes[str(semantic_id)] = str(outcome)
    if conflicting_outcome_ids:
        chain_valid = False
    unaccounted_ids = sorted(target_ids - set(accounted_outcomes))
    all_target_accounted = bool(
        chain_valid
        and target_ids
        and not invalid_conclusion_contracts
        and not conflicting_outcome_ids
        and not unaccounted_ids
    )
    return {
        "active_contract_id": active_contract_id,
        "active_contract_conclusive": bool(active_contract_id in valid_conclusion_contracts),
        "contract_chain_valid": chain_valid,
        "contract_generations": len(contracts),
        "target_hypotheses": len(target_ids),
        "accounted_hypotheses": len(accounted_outcomes),
        "unaccounted_hypotheses": len(unaccounted_ids),
        "unaccounted_ids": unaccounted_ids,
        "all_target_hypotheses_accounted": all_target_accounted,
        "accepted_hypotheses": sum(
            outcome == "ACCEPTED_SURVIVOR" for outcome in accounted_outcomes.values()
        ),
        "rejected_hypotheses": sum(
            outcome == "REJECTED_BY_FROZEN_GATES" for outcome in accounted_outcomes.values()
        ),
        "valid_conclusion_contracts": valid_conclusion_contracts,
        "invalid_conclusion_contracts": invalid_conclusion_contracts,
        "conflicting_outcome_ids": sorted(conflicting_outcome_ids),
    }


def _registered_contract_candidate_ids(contract: dict[str, Any]) -> set[str]:
    candidates = contract.get("registered_candidates", [])
    if not isinstance(candidates, list):
        return set()
    return {
        str(candidate.get("semantic_hypothesis_id", "")).strip()
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("semantic_hypothesis_id", "")).strip()
    }


def _registered_contract_identity_valid(
    contract: dict[str, Any], *, acceptance_policy_id: str, holdout_policy_id: str
) -> bool:
    candidate_ids = _registered_contract_candidate_ids(contract)
    candidates = contract.get("registered_candidates", [])
    if (
        not candidate_ids
        or not isinstance(candidates, list)
        or len(candidate_ids) != len(candidates)
    ):
        return False
    if not str(contract.get("contract_id", "")).startswith("registeredrerun_"):
        return False
    try:
        validate_registered_rerun_contract_identity(contract)
    except (TypeError, ValueError):
        return False
    return bool(
        contract.get("schema_version") == "thewiz.corrective_registered_rerun.v1"
        and contract.get("acceptance_policy_id") == acceptance_policy_id
        and contract.get("holdout_policy_id") == holdout_policy_id
        and len(str(contract.get("discovery_policy_sha256", ""))) == 64
        and len(str(contract.get("source_family_sha256", ""))) == 64
        and int(contract.get("source_family_rows", 0) or 0) > 0
        and bool(contract.get("full_family_multiplicity_required", False))
        and bool(contract.get("promotion_evaluation_registered_only", False))
        and not bool(contract.get("threshold_changes_after_contract_permitted", True))
        and not bool(contract.get("promotion_authority", False))
        and not bool(contract.get("testnet_order_authority", False))
        and not bool(contract.get("live_trading_authorized", False))
    )


def _read_json_fail_closed(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _ou_v4_immutable_evidence_complete(root: Path, evaluation: dict[str, Any]) -> bool:
    relative = str(evaluation.get("immutable_result_path", "")).strip()
    result_id = str(evaluation.get("result_id", "")).strip()
    expected_hash = str(evaluation.get("immutable_result_sha256", "")).strip()
    if not relative or not result_id or len(expected_hash) != 64:
        return False
    path = Path(relative)
    path = path if path.is_absolute() else root / path
    evidence_root = (root / "data" / "research" / "wizard_ou_v4_holdout_evaluations").resolve()
    try:
        path.resolve().relative_to(evidence_root)
    except ValueError:
        return False
    required = int(evaluation.get("required_cells", 0) or 0)
    passed = int(evaluation.get("passed_cells", 0) or 0)
    return bool(
        evaluation.get("status") == "PASS"
        and required == 8
        and passed == required
        and bool(evaluation.get("comparator_supersession_eligible", False))
        and bool(evaluation.get("research_only", False))
        and not bool(evaluation.get("candidate_promotion_authority", True))
        and not bool(evaluation.get("testnet_order_authority", True))
        and not bool(evaluation.get("live_trading_authorized", True))
        and path.stem == result_id
        and path.is_file()
        and _file_sha256(path) == expected_hash
    )


def _ou_v5_immutable_evidence_complete(root: Path, evaluation: dict[str, Any]) -> bool:
    relative = str(evaluation.get("immutable_result_path", "")).strip()
    result_id = str(evaluation.get("result_id", "")).strip()
    expected_hash = str(evaluation.get("immutable_result_sha256", "")).strip()
    if not relative or not result_id or len(expected_hash) != 64:
        return False
    path = Path(relative)
    path = path if path.is_absolute() else root / path
    evidence_root = (root / "data" / "research" / "wizard_ou_v5_holdout_evaluations").resolve()
    try:
        path.resolve().relative_to(evidence_root)
    except ValueError:
        return False
    required = int(evaluation.get("required_cells", 0) or 0)
    passed = int(evaluation.get("passed_cells", 0) or 0)
    return bool(
        evaluation.get("status") == "PASS"
        and required == 8
        and passed == required
        and bool(evaluation.get("research_only", False))
        and not bool(evaluation.get("candidate_promotion_authority", True))
        and not bool(evaluation.get("testnet_order_authority", True))
        and not bool(evaluation.get("live_trading_authorized", True))
        and path.stem == result_id
        and path.is_file()
        and _file_sha256(path) == expected_hash
    )


def _ou_v6_immutable_evidence_complete(root: Path, evaluation: dict[str, Any]) -> bool:
    relative = str(evaluation.get("immutable_result_path", "")).strip()
    result_id = str(evaluation.get("result_id", "")).strip()
    expected_hash = str(evaluation.get("immutable_result_sha256", "")).strip()
    if not relative or not result_id or len(expected_hash) != 64:
        return False
    path = Path(relative)
    path = path if path.is_absolute() else root / path
    evidence_root = (root / "data" / "research" / "wizard_ou_v6_holdout_evaluations").resolve()
    try:
        path.resolve().relative_to(evidence_root)
    except ValueError:
        return False
    required = int(evaluation.get("required_cells", 0) or 0)
    passed = int(evaluation.get("passed_cells", 0) or 0)
    return bool(
        evaluation.get("status") == "PASS"
        and required == 8
        and passed == required
        and bool(evaluation.get("final_successor_iteration", False))
        and not bool(evaluation.get("successor_after_v6_failure_allowed", True))
        and bool(evaluation.get("research_only", False))
        and not bool(evaluation.get("candidate_promotion_authority", True))
        and not bool(evaluation.get("testnet_order_authority", True))
        and not bool(evaluation.get("live_trading_authorized", True))
        and path.stem == result_id
        and path.is_file()
        and _file_sha256(path) == expected_hash
    )


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _join_blockers(values: list[Any]) -> str:
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in ordered:
            ordered.append(text)
    return ";".join(ordered)


def _stage_five_blockers(
    *,
    stage_five_pass: bool,
    stage_four_complete: bool,
    registered_learning: dict[str, Any],
    registered_learning_available: bool,
    registered_learning_audit: dict[str, Any],
    stage5_protocol_pass: bool,
    model_blockers: list[Any],
) -> str:
    """Separate expected Stage 4 waiting from claimed Stage 5 evidence failure."""

    if stage_five_pass:
        return ""
    learning_status = str(registered_learning.get("status", ""))
    terminal_statuses = {
        "PASS_RESEARCH_LEARNING_GATES",
        "REJECTED_RESEARCH_LEARNING_GATES",
        "ALREADY_COMPLETE",
    }
    waiting_for_stage_four = bool(learning_status == "BLOCKED_STAGE4" and not stage_four_complete)
    values: list[Any] = [
        (
            "registered_stage4_execution_receipt_not_yet_available"
            if waiting_for_stage_four
            else str(registered_learning.get("blocker", ""))
            if registered_learning_available
            else "registered_learning_status_missing"
        ),
        "registered_stage5_protocol_missing_or_blocked" if not stage5_protocol_pass else "",
    ]
    if learning_status in terminal_statuses:
        values.extend(registered_learning_audit.get("blockers", []))
    if bool(registered_learning_audit.get("evidence_valid", False)):
        values.extend(model_blockers)
    return _join_blockers(values)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "pass", "ready"}


def _live_canary_outcome_pass(
    *, root: Path, authorization_path: Path, outcome: dict[str, Any]
) -> bool:
    supplied_hash = str(outcome.get("receipt_sha256", ""))
    core = {key: value for key, value in outcome.items() if key != "receipt_sha256"}
    try:
        expected_hash = sha256(
            json.dumps(
                core,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError):
        return False
    execution_path = root / "reports" / "active" / "live_canary_execution_receipt.json"
    review_path = root / "reports" / "supreme_team" / "live_canary_post_canary_review.json"
    return bool(
        outcome.get("schema_version") == "thewiz.live_canary_outcome.v4"
        and outcome.get("canary_status") == "PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY"
        and supplied_hash == expected_hash
        and outcome.get("authorization_receipt_sha256") == _file_sha256(authorization_path)
        and outcome.get("execution_receipt_path")
        == "reports/active/live_canary_execution_receipt.json"
        and execution_path.is_file()
        and outcome.get("post_canary_review_path")
        == "reports/supreme_team/live_canary_post_canary_review.json"
        and review_path.is_file()
        and outcome.get("execution_blockers") == []
        and outcome.get("post_canary_review_blockers") == []
        and int(outcome.get("orders_submitted", 0) or 0) == 4
        and int(outcome.get("fills_observed", 0) or 0) == 4
        and outcome.get("reconciled_flat") is True
        and outcome.get("repeat_authorized") is False
        and outcome.get("scaling_authorized") is False
        and outcome.get("leverage_authorized") is False
        and outcome.get("new_explicit_authorization_required") is True
        and outcome.get("canary_execution_authority") is False
        and outcome.get("live_trading_authorized") is False
    )


def _next_strict_cost_action(venue: dict[str, Any]) -> str:
    strict_pairs, eligible_pairs = _stage_two_cost_counts(venue)
    post_window_ready = int(venue.get("post_window_ready_pairs", 0) or 0)
    post_window_eligible = int(venue.get("post_window_eligible_pairs", 0) or 0)
    if eligible_pairs > 0 and strict_pairs == eligible_pairs:
        return "maintain_rolling_l2_collection_and_monitor_freshness"
    if post_window_eligible > 0 and post_window_ready == post_window_eligible:
        return "refresh_wizard_candidate_after_completed_l2_window_then_rematerialize_point_in_time_cost_evidence"
    return "l2_scheduler_collects_every_5_minutes_until_strict_window_is_complete"


def _strict_cost_task_status(venue: dict[str, Any]) -> tuple[str, str]:
    strict_ready, strict_eligible = _stage_two_cost_counts(venue)
    status = (
        "completed"
        if strict_eligible > 0 and strict_ready == strict_eligible
        else "blocked_by_evidence"
    )
    return status, f"strict_pair_cost_models_{strict_ready}_of_{strict_eligible}"


def _stage_two_cost_counts(venue: dict[str, Any]) -> tuple[int, int]:
    ready = int(
        venue.get(
            "stage_two_strict_cost_ready_pairs",
            venue.get("strict_cost_ready_pairs", 0),
        )
        or 0
    )
    eligible = int(
        venue.get(
            "stage_two_strict_cost_eligible_pairs",
            venue.get("strict_cost_eligible_pairs", 0),
        )
        or 0
    )
    return ready, eligible


def _registered_rerun_task_status(
    statistics: dict[str, Any],
    *,
    execution: dict[str, Any] | None = None,
    root: Path | None = None,
    current_family_accounting: dict[str, Any] | None = None,
) -> tuple[str, str]:
    if current_family_accounting is not None:
        if bool(current_family_accounting.get("complete", False)):
            return (
                "completed",
                "registered_current_family_accounted_"
                + str(
                    current_family_accounting.get("terminal_outcome", "unknown_terminal_outcome")
                ).lower(),
            )
        if bool(current_family_accounting.get("active_contract_conclusive", False)):
            return (
                "in_progress_evidence_collection",
                "registered_contract_generation_accounted_but_current_family_has_"
                + str(current_family_accounting.get("unaccounted_hypotheses", 0))
                + "_unaccounted_hypotheses",
            )
        if bool(current_family_accounting.get("active_contract_id", "")):
            gate_status = str(statistics.get("registered_rerun_gate_status", "NOT_EVALUATED"))
            if gate_status != "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN":
                return (
                    "blocked_pending_new_evidence",
                    f"active_registered_contract_gate_{gate_status.lower()}",
                )
            return (
                "in_progress_evidence_collection",
                "active_registered_contract_awaiting_conclusive_immutable_outcomes",
            )
    if (
        execution
        and root is not None
        and _registered_execution_receipt_valid(root, execution)
        and str(execution.get("conclusion_status", ""))
        in {
            "ACCEPTED_REGISTERED_SURVIVORS",
            "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        }
    ):
        return (
            "completed",
            "registered_full_family_rerun_accounted_" + str(execution["conclusion_status"]).lower(),
        )
    gate_status = str(statistics.get("registered_rerun_gate_status", "NOT_EVALUATED"))
    if bool(statistics.get("registered_rerun_results_accounted", False)):
        conclusion = str(statistics.get("registered_rerun_conclusion_status", "INCOMPLETE"))
        return "completed", f"registered_full_family_rerun_accounted_{conclusion.lower()}"
    if gate_status == "READY_FOR_NEXT_SCHEDULED_RESEARCH_RERUN":
        return (
            "in_progress_evidence_collection",
            "registered_full_family_rerun_ready_awaiting_post_ready_chain",
        )
    return (
        "blocked_pending_new_evidence",
        f"registered_full_family_rerun_gate_{gate_status.lower()}",
    )


def _registered_execution_receipt_valid(root: Path, execution: dict[str, Any]) -> bool:
    relative = str(execution.get("execution_receipt_path", "")).strip()
    expected_file_hash = str(execution.get("execution_receipt_sha256", "")).strip()
    if not relative or len(expected_file_hash) != 64:
        return False
    path = root / relative
    if not path.is_file() or sha256(path.read_bytes()).hexdigest() != expected_file_hash:
        return False
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    embedded = str(receipt.pop("receipt_sha256", "")).strip()
    payload_hash = sha256(
        json.dumps(
            receipt,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return bool(
        embedded == payload_hash
        and receipt.get("status") == "PASS_REGISTERED_RERUN_ACCOUNTED"
        and receipt.get("conclusion_status")
        in {
            "ACCEPTED_REGISTERED_SURVIVORS",
            "CONCLUSIVE_ZERO_REGISTERED_SURVIVORS",
        }
        and not bool(receipt.get("order_submission_performed"))
        and not bool(receipt.get("testnet_order_authority"))
        and not bool(receipt.get("live_trading_authorized"))
    )


def _next_wizard_proof_action(
    *,
    scheduler_status: str,
    queue_completed: int,
    queue_eligible: int,
    responses_captured: int = 0,
) -> str:
    if queue_eligible > 0 and queue_completed >= queue_eligible:
        return "refresh_formula_parity_then_capture_missing_ou_optimal_orientation"
    if queue_eligible > 0 and responses_captured >= queue_eligible:
        return "implement_mode_specific_comparators_from_preserved_vendor_series"
    if scheduler_status == "DEFERRED_SAME_UTC_DAY":
        return "wait_for_next_utc_credit_reset_then_resume_bounded_exact_mode_proofs"
    if scheduler_status in {
        "PLANNED",
        "DEFERRED_CREDIT_RESET",
        "PARTIAL_CREDIT_DEFERRED",
    }:
        return "run_bounded_exact_mode_proofs_after_0000_utc_credit_reset"
    if scheduler_status == "BLOCKED_NO_PROGRESS":
        return "repair_first_failing_exact_mode_proof_input_before_retry"
    return "capture_authenticated_comparable_vendor_numeric_series"


def _next_stage_three_action(
    *,
    comparator_review_status: str,
    scheduler_status: str,
    queue_completed: int,
    queue_eligible: int,
    responses_captured: int = 0,
) -> str:
    if comparator_review_status == "BLOCKED_REVIEW_CONTROL":
        return "repair_comparator_review_evidence_bindings"
    if comparator_review_status == "WAITING_FOR_FROZEN_CAPTURE_RECONCILIATION":
        return "complete_and_reconcile_frozen_stage3_capture_before_apply"
    if comparator_review_status == "READY_FOR_EXPLICIT_HUMAN_REVIEW":
        return "human_reviews_exact_comparator_packet_ids_before_research_only_apply"
    if comparator_review_status == "WAITING_FOR_PROSPECTIVE_EVIDENCE":
        return "collect_and_evaluate_remaining_comparator_holdout_cells"
    return _next_wizard_proof_action(
        scheduler_status=scheduler_status,
        queue_completed=queue_completed,
        queue_eligible=queue_eligible,
        responses_captured=responses_captured,
    )


def _next_ou_v5_action(
    *,
    required: bool,
    immutable_evidence_complete: bool,
    evaluation: dict[str, Any],
    failure_attribution: dict[str, Any],
    review_packet: dict[str, Any],
    supreme_review: dict[str, Any],
    activation: dict[str, Any],
    proof_refresh: dict[str, Any],
) -> str:
    """Route every registered OU-v5 outcome without granting authority."""

    if not required:
        return ""
    evaluation_status = str(evaluation.get("status", "NOT_EVALUATED"))
    if evaluation_status == "FAIL":
        if failure_attribution.get("status") != "PASS_FAILURE_ATTRIBUTION_COMPLETE":
            return "build_ou_v5_failure_attribution"
        return "supreme_team_review_ou_v5_failure_before_successor_preregistration"
    if not immutable_evidence_complete:
        return "run_governed_ou_v5_holdout_after_utc_reset"
    if evaluation_status != "PASS":
        return "repair_ou_v5_immutable_holdout_evidence"
    if review_packet.get("status") != "READY_FOR_EXPLICIT_REVIEW":
        return "build_or_repair_ou_v5_supersession_review_packet"
    if supreme_review.get("status") != "PASS_ADVISORY_ONLY":
        return "build_or_repair_ou_v5_supreme_team_review"
    if activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        return "human_review_ou_v5_exact_packet_and_apply_research_comparator"
    if proof_refresh.get("status") != "PASS":
        return "refresh_ou_v5_activation_bound_current_proof_rows"
    return ""


def _next_ou_v6_action(
    *,
    required: bool,
    immutable_evidence_complete: bool,
    evaluation: dict[str, Any],
    activation: dict[str, Any],
    proof_refresh: dict[str, Any],
) -> str:
    """Route the final OU successor without permitting a post-failure v7."""

    if not required:
        return ""
    evaluation_status = str(evaluation.get("status", "NOT_EVALUATED"))
    if evaluation_status == "FAIL":
        return "conclusively_close_local_ou_exact_parity_after_terminal_v6_failure"
    if not immutable_evidence_complete:
        return "run_governed_terminal_ou_v6_holdout_after_utc_reset"
    if evaluation_status != "PASS":
        return "repair_ou_v6_immutable_holdout_evidence"
    if activation.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        return "build_and_human_review_ou_v6_terminal_success_activation_packet"
    if proof_refresh.get("status") != "PASS":
        return "refresh_ou_v6_activation_bound_current_proof_rows"
    return ""


def _completion_markdown(
    completion: dict[str, Any], plan: pd.DataFrame, phases: pd.DataFrame
) -> str:
    blocked = plan.loc[~plan["status"].astype(str).str.startswith(("completed", "in_progress"))]
    lines = [
        "# Corrective Plan Completion",
        "",
        f"- Implementation: `{completion['implementation_status']}`",
        f"- Operational acceptance: `{completion['operational_acceptance_status']}`",
        f"- Tasks completed/checkpointed: `{completion['completed_or_completed_checkpoint_tasks']}/{completion['total_tasks']}`",
        f"- Tasks in evidence collection: `{completion['in_progress_tasks']}`",
        f"- Tasks blocked or not authorized: `{completion['blocked_or_not_authorized_tasks']}`",
        f"- Final 1x survivors: `{completion['final_one_x_survivors']}`",
        "- Orders submitted by this program: `0`",
        "- Live trading authorized: `false`",
        "",
        "## Phase Status",
        "",
        phases.to_markdown(index=False),
        "",
        "## Remaining Gates",
        "",
    ]
    lines.extend(
        f"- `{row.task_id}` {row.task}: `{row.status}` - {row.current_blocker_or_completion_reason}"
        for row in blocked.itertuples()
    )
    lines.extend(
        [
            "",
            "The implementation is complete, but the evidence is not. The system remains research-only and will not cross Testnet or live gates until the prospective requirements are actually observed.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_blocker(summary: dict[str, Any]) -> str:
    blocker = summary.get("blocker", summary.get("blockers", ""))
    if isinstance(blocker, list):
        return ";".join(str(item) for item in blocker)
    return str(blocker or "")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


if __name__ == "__main__":
    result = complete_corrective_plan()
    print(
        json.dumps(
            {
                "summary": result.summary,
                "paths": {key: str(value) for key, value in result.paths.items()},
            },
            indent=2,
        )
    )
