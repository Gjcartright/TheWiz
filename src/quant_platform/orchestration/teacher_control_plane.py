"""Report-backed control plane for teacher proposals, critics, and students."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ValidationError

from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import atomic_write_csv, atomic_write_text
from quant_platform.orchestration.student_readiness import write_student_training_readiness
from quant_platform.orchestration.teacher_contracts import (
    EXACT_MODES,
    MATH_V2,
    REQUIRED_CRITICS,
    CouncilDecision,
    CriticAssessment,
    StudentOutcomeForecast,
    StudentRouterPrediction,
    TeacherProposal,
    normalize_exact_mode,
)
from quant_platform.orchestration.teacher_council import arbitrate_teacher_council

ROOT = Path(__file__).resolve().parents[3]
T = TypeVar("T", bound=BaseModel)


def build_teacher_council_control_plane(*, root: Path = ROOT) -> dict[str, object]:
    """Validate event streams, run councils, and expose every current blocker."""

    directory = root / "reports" / "orchestration" / "teacher_council"
    directory.mkdir(parents=True, exist_ok=True)
    registry_path = directory / "teacher_registry.csv"
    proposal_report_path = directory / "teacher_proposals.csv"
    critic_report_path = directory / "critic_assessments.csv"
    decision_report_path = directory / "council_decisions.csv"
    decision_jsonl_path = directory / "council_decisions.jsonl"
    errors_path = directory / "event_validation_errors.csv"
    readiness_path = directory / "teacher_council_readiness.csv"
    markdown_path = directory / "teacher_council.md"
    wizard_path = directory / "wizard_discovery_hypotheses.csv"

    registry = _teacher_registry()
    atomic_write_csv(registry, registry_path, index=False)
    wizard = _wizard_discovery_hypotheses(root)
    atomic_write_csv(wizard, wizard_path, index=False)

    proposals, proposal_errors = _read_models(directory / "teacher_proposals.jsonl", TeacherProposal)
    assessments, assessment_errors = _read_models(directory / "critic_assessments.jsonl", CriticAssessment)
    router_predictions, router_errors = _read_models(
        directory / "student_router_predictions.jsonl", StudentRouterPrediction
    )
    outcome_forecasts, outcome_errors = _read_models(
        directory / "student_outcome_forecasts.jsonl", StudentOutcomeForecast
    )
    errors = pd.DataFrame([*proposal_errors, *assessment_errors, *router_errors, *outcome_errors])
    atomic_write_csv(errors, errors_path, index=False)

    proposal_frame = pd.DataFrame([_proposal_row(record) for record in proposals])
    critic_frame = pd.DataFrame([_critic_row(record) for record in assessments])
    atomic_write_csv(proposal_frame, proposal_report_path, index=False)
    atomic_write_csv(critic_frame, critic_report_path, index=False)

    decisions = _run_councils(proposals, assessments, router_predictions, outcome_forecasts)
    decision_frame = pd.DataFrame([_decision_row(record) for record in decisions])
    atomic_write_csv(decision_frame, decision_report_path, index=False)
    atomic_write_text(decision_jsonl_path, "".join(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n" for record in decisions), encoding="utf-8")

    student = write_student_training_readiness(root=root)
    readiness = _readiness(
        root=root,
        registry=registry,
        wizard=wizard,
        proposals=proposals,
        assessments=assessments,
        decisions=decisions,
        validation_errors=errors,
        supervised_status=str(student["supervised_status"]),
        bandit_status=str(student["bandit_status"]),
    )
    atomic_write_csv(readiness, readiness_path, index=False)
    shadow_blocked = readiness["status"].eq("BLOCKED") & readiness["blocking_for_shadow"].astype(bool)
    overall = "BLOCKED" if shadow_blocked.any() else "READY_FOR_SHADOW_ONLY"
    atomic_write_text(markdown_path, _control_plane_markdown(
            readiness,
            registry=registry,
            decisions=decision_frame,
            overall=overall,
        ), encoding="utf-8")
    return {
        "status": overall,
        "teacher_count": len(EXACT_MODES),
        "critic_count": len(REQUIRED_CRITICS),
        "wizard_hypotheses": len(wizard),
        "proposal_count": len(proposals),
        "assessment_count": len(assessments),
        "decision_count": len(decisions),
        "validation_error_count": len(errors),
        "registry": registry_path,
        "wizard_discovery": wizard_path,
        "proposals": proposal_report_path,
        "critics": critic_report_path,
        "decisions": decision_report_path,
        "decisions_jsonl": decision_jsonl_path,
        "validation_errors": errors_path,
        "readiness": readiness_path,
        "markdown": markdown_path,
        "student_readiness": student["audit"],
        "student_schema": student["schema"],
    }


def _teacher_registry() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mode in EXACT_MODES:
        rows.append(
            {
                "component": _slug(mode.value) + "_teacher",
                "component_type": "exact_mode_teacher",
                "specialization": mode.value,
                "authority": "local_point_in_time_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "teacher_proposal",
                "prerequisite": "Math V2 exact-mode replay",
            }
        )
    for critic_type in REQUIRED_CRITICS:
        rows.append(
            {
                "component": f"{critic_type.value}_critic",
                "component_type": "critic",
                "specialization": critic_type.value,
                "authority": "local_point_in_time_or_realized_outcome",
                "can_veto": True,
                "can_promote": False,
                "can_execute": False,
                "output": "critic_assessment",
                "prerequisite": "fresh evidence with lineage",
            }
        )
    rows.extend(
        [
            {
                "component": "crypto_wizards_discovery_teacher",
                "component_type": "discovery_teacher",
                "specialization": "candidate and exact-mode hypothesis",
                "authority": "discovery_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "wizard_discovery_hypothesis",
                "prerequisite": "fresh Wizard capture",
            },
            {
                "component": "mixture_router_student",
                "component_type": "student",
                "specialization": "mode routing plus abstain",
                "authority": "advisory_shadow_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "student_router_prediction",
                "prerequisite": "student readiness audit passes",
            },
            {
                "component": "distributional_outcome_student",
                "component_type": "student",
                "specialization": "after-cost return, MAE, MFE, holding time, failure risk",
                "authority": "advisory_shadow_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "student_outcome_forecast",
                "prerequisite": "student readiness audit passes",
            },
            {
                "component": "contextual_bandit_allocator",
                "component_type": "adaptive_challenger",
                "specialization": "choose among already-safe strategy variants plus abstain",
                "authority": "shadow_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "shadow_allocation",
                "prerequisite": "logged action propensities and off-policy evaluation",
            },
            {
                "component": "offline_rl_challenger",
                "component_type": "future_challenger",
                "specialization": "sequential entry, hold, resize, and exit",
                "authority": "research_only",
                "can_veto": False,
                "can_promote": False,
                "can_execute": False,
                "output": "shadow_policy_proposal",
                "prerequisite": "supervised, bandit, and Testnet outcome gates pass",
            },
        ]
    )
    return pd.DataFrame(rows)


def _wizard_discovery_hypotheses(root: Path) -> pd.DataFrame:
    source = root / "data" / "processed" / "wizard_evidence.csv"
    frame = _read_csv(source)
    columns = [
        "pair",
        "source_exchange",
        "target_venue",
        "timeframe",
        "exact_mode",
        "sharpe",
        "returns_total",
        "source_timestamp",
        "source_fresh",
        "discovery_gate_pass",
        "teacher_action",
        "authority",
        "student_label_eligible",
        "local_vote_eligible",
        "reason",
        "evidence_path",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        try:
            mode = normalize_exact_mode(str(row.get("exact_mode", ""))).value
        except ValueError:
            continue
        sharpe_pass = _boolish(row.get("passes_sharpe_gate"))
        return_pass = _number(row.get("returns_total")) >= _number(row.get("discovery_min_returns_total"), default=0.10)
        rows.append(
            {
                "pair": str(row.get("pair", "")),
                "source_exchange": str(row.get("exchange", "")),
                "target_venue": "hyperliquid",
                "timeframe": str(row.get("interval", "")),
                "exact_mode": mode,
                "sharpe": _number(row.get("sharpe"), default=float("nan")),
                "returns_total": _number(row.get("returns_total"), default=float("nan")),
                "source_timestamp": str(row.get("source_timestamp", "")),
                "source_fresh": _boolish(row.get("source_fresh")),
                "discovery_gate_pass": bool(sharpe_pass and return_pass),
                "teacher_action": "abstain",
                "authority": "discovery_only",
                "student_label_eligible": False,
                "local_vote_eligible": False,
                "reason": "Wizard may nominate exact mode; Hyperliquid Math V2 replay and critics must decide",
                "evidence_path": str(row.get("evidence_path", row.get("source_path", source))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _run_councils(
    proposals: list[TeacherProposal],
    assessments: list[CriticAssessment],
    router_predictions: list[StudentRouterPrediction],
    outcome_forecasts: list[StudentOutcomeForecast],
) -> list[CouncilDecision]:
    contexts = {
        record.context.context_id: record.context
        for record in [*proposals, *assessments]
    }
    routers = {record.context_id: record for record in router_predictions}
    outcomes = {record.context_id: record for record in outcome_forecasts}
    decisions: list[CouncilDecision] = []
    for context_id, context in sorted(contexts.items()):
        context_proposals = tuple(record for record in proposals if record.context.context_id == context_id)
        context_assessments = tuple(record for record in assessments if record.context.context_id == context_id)
        decisions.append(
            arbitrate_teacher_council(
                context=context,
                proposals=context_proposals,
                assessments=context_assessments,
                router_prediction=routers.get(context_id),
                outcome_forecast=outcomes.get(context_id),
            )
        )
    return decisions


def _read_models(path: Path, model: type[T]) -> tuple[list[T], list[dict[str, object]]]:
    records: list[T] = []
    errors: list[dict[str, object]] = []
    if not path.exists():
        return records, errors
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(model.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            errors.append(
                {
                    "path": str(path),
                    "line_number": line_number,
                    "model": model.__name__,
                    "error": safe_exception_code(exc),
                }
            )
    return records, errors


def _readiness(
    *,
    root: Path,
    registry: pd.DataFrame,
    wizard: pd.DataFrame,
    proposals: list[TeacherProposal],
    assessments: list[CriticAssessment],
    decisions: list[CouncilDecision],
    validation_errors: pd.DataFrame,
    supervised_status: str,
    bandit_status: str,
) -> pd.DataFrame:
    math_marker = root / "reports" / "active" / "math_v2_acceptance.json"
    math_pass = _math_v2_marker_passes(math_marker)
    checks = [
        ("seven_exact_mode_teachers", registry["component_type"].eq("exact_mode_teacher").sum() == 7, 7, 7, "teacher_registry_incomplete", True),
        ("six_independent_critics", registry["component_type"].eq("critic").sum() == 6, 6, 6, "critic_registry_incomplete", True),
        ("wizard_discovery_connected", not wizard.empty, len(wizard), "> 0", "wizard_discovery_hypotheses_missing", True),
        ("math_v2_accepted", math_pass, str(math_marker), "status=passed", "math_v2_not_accepted", True),
        ("teacher_proposals_present", bool(proposals), len(proposals), ">= 7 per context", "teacher_proposal_stream_empty", True),
        ("critic_assessments_present", bool(assessments), len(assessments), ">= 6 per context", "critic_assessment_stream_empty", True),
        ("event_contracts_valid", validation_errors.empty, len(validation_errors), 0, "invalid_teacher_event_contracts", True),
        ("council_decisions_present", bool(decisions), len(decisions), "> 0", "no_council_decisions", True),
        ("supervised_student_ready", supervised_status != "BLOCKED", supervised_status, "READY_FOR_SHADOW_TRAINING", "student_dataset_not_ready", True),
        ("contextual_bandit_ready", bandit_status != "BLOCKED", bandit_status, "READY_FOR_SHADOW_EVALUATION", "bandit_dataset_not_ready", False),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "status": "PASS" if passed else "BLOCKED",
                "observed": observed,
                "required": required,
                "blocker": "" if passed else blocker,
                "execution_allowed": False,
                "next_step": _readiness_next_step(name, passed),
                "blocking_for_shadow": blocking_for_shadow,
            }
            for name, passed, observed, required, blocker, blocking_for_shadow in checks
        ]
    )


def _math_v2_marker_passes(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        str(payload.get("status", "")).strip().lower() == "passed"
        and payload.get("math_version") == MATH_V2
        and payload.get("acceptance_scope") == "core_math_library"
        and payload.get("all_checks_passed") is True
        and payload.get("generated_by") == "quant_platform.math_v2_acceptance"
    )


def _proposal_row(record: TeacherProposal) -> dict[str, object]:
    return {
        "run_id": record.context.run_id,
        "candidate_set_id": record.context.candidate_set_id,
        "setup_identity": record.context.setup_identity,
        "proposal_id": record.proposal_id,
        "context_id": record.context.context_id,
        "candidate_id": record.candidate.candidate_id,
        "pair": record.context.pair,
        "venue": record.context.venue,
        "timeframe": record.context.timeframe,
        "exact_mode": record.exact_mode.value,
        "action": record.proposed_action.value,
        "confidence": record.confidence,
        "uncertainty": record.uncertainty,
        "expected_net_return": record.expected_net_return,
        "lower_bound_net_return": record.lower_bound_net_return,
        "authority": record.authority.value,
        "point_in_time_status": record.point_in_time_status,
        "math_version": record.math_version,
        "blockers": ";".join(record.blockers),
        "evidence_paths": ";".join(record.evidence_paths),
    }


def _critic_row(record: CriticAssessment) -> dict[str, object]:
    return {
        "run_id": record.context.run_id,
        "candidate_set_id": record.context.candidate_set_id,
        "setup_identity": record.context.setup_identity,
        "assessment_id": record.assessment_id,
        "context_id": record.context.context_id,
        "pair": record.context.pair,
        "critic_type": record.critic_type.value,
        "verdict": record.verdict.value,
        "score": record.score,
        "reason": record.reason,
        "authority": record.authority.value,
        "point_in_time_status": record.point_in_time_status,
        "blocker_codes": ";".join(record.blocker_codes),
        "evidence_paths": ";".join(record.evidence_paths),
    }


def _decision_row(record: CouncilDecision) -> dict[str, object]:
    return {
        "run_id": record.context.run_id,
        "candidate_set_id": record.context.candidate_set_id,
        "setup_identity": record.context.setup_identity,
        "decision_id": record.decision_id,
        "context_id": record.context.context_id,
        "pair": record.context.pair,
        "venue": record.context.venue,
        "timeframe": record.context.timeframe,
        "status": record.status.value,
        "action": record.action.value,
        "research_preference": record.research_preference.value,
        "selected_mode": record.selected_mode.value if record.selected_mode else "",
        "weighted_confidence": record.weighted_confidence,
        "teacher_disagreement": record.teacher_disagreement,
        "uncertainty_penalty": record.uncertainty_penalty,
        "lower_bound_net_return": record.lower_bound_net_return,
        "blocker_codes": ";".join(record.blocker_codes),
        "reason": record.reason,
        "shadow_only": record.shadow_only,
        "execution_allowed": record.execution_allowed,
        "evidence_paths": ";".join(record.evidence_paths),
    }


def _control_plane_markdown(
    readiness: pd.DataFrame,
    *,
    registry: pd.DataFrame,
    decisions: pd.DataFrame,
    overall: str,
) -> str:
    lines = [
        "# Teacher Council And Student Stack",
        "",
        f"- Overall status: **{overall}**",
        "- Crypto Wizards authority: **discovery only**",
        "- Local authority: **Hyperliquid point-in-time Math V2 replay**",
        "- Council authority: **shadow research only**",
        "- Paper/live execution authority: **none**",
        "",
        "## Readiness",
        "",
        readiness.to_markdown(index=False),
        "",
        "## Components",
        "",
        registry.to_markdown(index=False),
        "",
        "## Current Decisions",
        "",
        decisions.to_markdown(index=False) if not decisions.empty else "No validated council contexts are available yet.",
        "",
    ]
    return "\n".join(lines)


def _readiness_next_step(name: str, passed: bool) -> str:
    if passed:
        return "retain and monitor"
    steps = {
        "wizard_discovery_connected": "refresh canonical Crypto Wizards discovery evidence",
        "math_v2_accepted": "complete Math V2 repairs and write the reviewed acceptance marker",
        "teacher_proposals_present": "emit all seven local exact-mode teacher proposals for one context",
        "critic_assessments_present": "emit dependency, regime, risk, cost, execution, and outcome assessments",
        "event_contracts_valid": "repair invalid JSONL records before arbitration",
        "council_decisions_present": "supply complete proposals and critics, then rerun the council",
        "supervised_student_ready": "build the leakage-safe student training dataset",
        "contextual_bandit_ready": "log action propensities and pass off-policy readiness",
    }
    return steps.get(name, "repair the blocked registry requirement")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _boolish(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _number(value: object, *, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number


def _slug(value: str) -> str:
    return "_".join(str(value).lower().replace("zscorer", "zscore_r").split())
