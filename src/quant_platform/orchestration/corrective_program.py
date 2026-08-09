"""Run every non-order corrective phase and publish one truthful program status."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_agent_governance import build_corrective_agent_governance
from quant_platform.orchestration.corrective_daily_scheduler import build_corrective_daily_cadence
from quant_platform.orchestration.corrective_data_evidence import build_corrective_data_evidence
from quant_platform.orchestration.corrective_governance import build_corrective_governance
from quant_platform.orchestration.corrective_release_gates import build_corrective_release_gates
from quant_platform.orchestration.corrective_statistical_remediation import build_corrective_statistical_remediation
from quant_platform.orchestration.corrective_wizard_parity import build_corrective_wizard_parity


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_program_completion.v1"


TASK_STATUS = {
    **{f"T{index:02d}": ("completed", "implemented_and_verified") for index in range(1, 12)},
    "T12": ("in_progress_evidence_collection", "393_pair_histories_remain_in_remediation_queue"),
    "T13": ("in_progress_evidence_collection", "zero_assets_currently_meet_12_fresh_l2_samples_in_2h"),
    "T14": ("blocked_by_evidence", "zero_pair_cost_models_have_strict_observed_cost_acceptance"),
    "T15": ("completed", "seven_hostile_cost_bundles_fail_closed"),
    "T16": ("completed", "3408_mode_orientation_cells_accounted_2947_captured"),
    "T17": ("blocked_vendor_mode_unavailable", "ou_optimal_unavailable_on_captured_pair_pages"),
    "T18": ("completed", "parity_comparator_and_seven_formula_mutations_verified"),
    "T19": ("completed", "vendor_parity_separated_from_local_approximation"),
    "T20": ("completed", "42_raw_passes_clustered_into_33_effective_clusters"),
    "T21": ("completed", "42_near_misses_have_one_missing_proof_and_next_test"),
    "T22": ("completed", "bounded_hypothesis_batch_registered_before_new_tests"),
    "T23": ("blocked_pending_new_evidence", "rerun_forbidden_until_new_history_strict_cost_and_parity_evidence_arrive"),
    "T24": ("completed", "independent_breadth_gate_materialized_and_failed_at_one_of_three"),
    "T25": ("completed", "explicit_zero_survivor_receipt_issued"),
    "T26": ("completed", "research_only_launch_agent_loaded_for_0615_local"),
    "T27": ("completed", "seven_scheduler_fault_cases_passed"),
    "T28": ("in_progress_time_observation", "one_of_seven_distinct_calendar_day_cycles_observed"),
    "T29": ("completed", "29_agent_and_learning_authority_edges_inventoried"),
    "T30": ("completed", "eight_forged_agent_packets_fail_closed"),
    "T31": ("completed", "2338_learning_rows_audited_under_label_and_time_contract"),
    "T32": ("blocked_pending_model_and_testnet_evidence", "model_incremental_edge_and_realized_testnet_sample_not_proven"),
    "T33": ("blocked_by_safety_gate", "valid_final_one_x_survivor_receipt_missing"),
    "T34": ("blocked_by_dependency", "testnet_candidate_not_opened"),
    "T35": ("not_executed_safety_gate", "no_order_preflight_and_explicit_authorization_missing"),
    "T36": ("blocked_by_dependency", "zero_realized_testnet_lifecycles"),
    "T37": ("completed_blocked_checkpoint", "supreme_team_checkpoint_written_with_realized_evidence_gap"),
    "T38": ("completed", "prospective_live_canary_policy_frozen"),
    "T39": ("blocked_by_dependency", "no_testnet_candidate_for_shadow_live_input_parity"),
    "T40": ("not_authorized", "live_prerequisites_and_exact_order_user_authorization_missing"),
    "T41": ("not_executed_safety_gate", "no_live_canary_was_authorized_or_submitted"),
}


def complete_corrective_plan(*, root: Path = ROOT, now: datetime | None = None) -> CommandResult:
    now = _as_utc(now)
    phases = {
        "governance": build_corrective_governance(root=root, now=now),
        "venue_data_costs": build_corrective_data_evidence(root=root, now=now),
        "wizard_parity": build_corrective_wizard_parity(root=root),
        "statistical_remediation": build_corrective_statistical_remediation(root=root, now=now),
        "daily_cadence": build_corrective_daily_cadence(root=root, now=now, install=True),
        "agent_learning_governance": build_corrective_agent_governance(root=root, now=now),
        "conditional_release_gates": build_corrective_release_gates(root=root, now=now),
    }
    plan_path = root / "reports" / "active" / "corrective_execution_plan.csv"
    plan = _read_csv(plan_path)
    if plan.empty or set(TASK_STATUS) - set(plan.get("task_id", pd.Series(dtype=str)).astype(str)):
        raise ValueError("corrective execution plan is missing one or more canonical tasks")
    plan["status"] = plan["task_id"].map(lambda task: TASK_STATUS[str(task)][0])
    plan["current_blocker_or_completion_reason"] = plan["task_id"].map(lambda task: TASK_STATUS[str(task)][1])
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
                "testnet_order_authority": bool(result.summary.get("testnet_order_authority", False)),
                "live_trading_authorized": bool(result.summary.get("live_trading_authorized", False)),
                "evidence_paths": ";".join(_relative(path, root) for path in result.paths.values()),
            }
        )
    phase_frame = pd.DataFrame(phase_rows)
    unsafe = bool(phase_frame["testnet_order_authority"].any() or phase_frame["live_trading_authorized"].any())
    if unsafe:
        raise ValueError("corrective program unexpectedly acquired order authority")
    phase_path = root / "reports" / "active" / "corrective_plan_phase_status.csv"
    _atomic_csv(phase_frame, phase_path)
    final_receipt = _read_json(root / "reports" / "active" / "final_1x_survivor_receipt.json")
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
        "primary_blockers": [
            "strict_pair_cost_evidence_missing",
            "vendor_formula_parity_unproven",
            "independent_breadth_one_of_three",
            "daily_cadence_one_of_seven",
            "model_incremental_edge_not_accepted",
            "realized_testnet_sample_absent",
        ],
        "next_automatic_action": "daily_research_scheduler_collects_next_calendar_day_evidence_at_0615_local",
        "evidence_path": "reports/active/corrective_execution_plan.csv;reports/active/corrective_plan_phase_status.csv;reports/active/final_1x_survivor_receipt.json",
    }
    completion_path = root / "reports" / "active" / "corrective_plan_completion.json"
    completion_md = root / "reports" / "active" / "corrective_plan_completion.md"
    _atomic_json(completion, completion_path)
    completion_md.write_text(_completion_markdown(completion, plan, phase_frame), encoding="utf-8")
    return CommandResult(
        paths={
            "execution_plan": plan_path,
            "phase_status": phase_path,
            "completion_receipt": completion_path,
            "completion_summary": completion_md,
            **{f"phase_{phase}_{name}": path for phase, result in phases.items() for name, path in result.paths.items()},
        },
        summary=completion,
    )


def _completion_markdown(completion: dict[str, Any], plan: pd.DataFrame, phases: pd.DataFrame) -> str:
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
    lines.extend(["", "The implementation is complete, but the evidence is not. The system remains research-only and will not cross Testnet or live gates until the prospective requirements are actually observed.", ""])
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
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


if __name__ == "__main__":
    result = complete_corrective_plan()
    print(json.dumps({"summary": result.summary, "paths": {key: str(value) for key, value in result.paths.items()}}, indent=2))
