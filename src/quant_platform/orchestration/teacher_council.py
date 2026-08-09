"""Deterministic arbitration for the seven-mode teacher council."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256

from quant_platform.orchestration.teacher_contracts import (
    CouncilContext,
    CouncilDecision,
    CouncilStatus,
    CriticAssessment,
    CriticType,
    CriticVerdict,
    EvidenceAuthority,
    ExactMode,
    MATH_V2,
    REQUIRED_CRITICS,
    StudentOutcomeForecast,
    StudentRouterPrediction,
    TeacherAction,
    TeacherProposal,
)


@dataclass(frozen=True)
class TeacherCouncilPolicy:
    """Thresholds for shadow research, never paper or live execution."""

    max_evidence_age_hours: float = 24.0
    max_teacher_disagreement: float = 0.35
    max_uncertainty: float = 0.40
    min_weighted_confidence: float = 0.55
    min_student_support: float = 0.60
    min_feature_completeness: float = 0.95
    max_structural_break_probability: float = 0.35
    max_execution_failure_probability: float = 0.25
    required_math_version: str = MATH_V2


def arbitrate_teacher_council(
    *,
    context: CouncilContext,
    proposals: tuple[TeacherProposal, ...],
    assessments: tuple[CriticAssessment, ...],
    router_prediction: StudentRouterPrediction | None = None,
    outcome_forecast: StudentOutcomeForecast | None = None,
    policy: TeacherCouncilPolicy | None = None,
    now: datetime | None = None,
) -> CouncilDecision:
    """Aggregate local teacher evidence with hard vetoes and mandatory abstention."""

    policy = policy or TeacherCouncilPolicy()
    now = now or datetime.now(timezone.utc)
    proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
    assessment_ids = tuple(assessment.assessment_id for assessment in assessments)
    blockers = _integrity_blockers(
        context=context,
        proposals=proposals,
        assessments=assessments,
        policy=policy,
        now=now,
    )

    eligible = tuple(proposal for proposal in proposals if _is_acceptance_teacher(proposal, policy=policy, now=now))
    vote = _teacher_vote(eligible)
    research_preference = vote["action"]
    selected = vote["selected"]
    confidence = vote["confidence"]
    disagreement = vote["disagreement"]
    uncertainty = vote["uncertainty"]
    lower_bound = vote["lower_bound"]

    if any(assessment.verdict == CriticVerdict.VETO for assessment in assessments):
        blockers.add("hard_critic_veto")
    if any(assessment.verdict == CriticVerdict.WARN for assessment in assessments):
        blockers.add("critic_warning_requires_review")
    if any(assessment.verdict == CriticVerdict.UNKNOWN for assessment in assessments):
        blockers.add("critic_unknown_requires_more_data")
    if not eligible and any(proposal.authority == EvidenceAuthority.DISCOVERY_ONLY for proposal in proposals):
        blockers.add("discovery_only_cannot_vote")
    if vote["action"] in {TeacherAction.ABSTAIN, TeacherAction.FLAT}:
        blockers.add("teacher_council_prefers_no_entry")
    if eligible:
        if disagreement > policy.max_teacher_disagreement:
            blockers.add("teacher_disagreement_above_limit")
        if uncertainty > policy.max_uncertainty:
            blockers.add("teacher_uncertainty_above_limit")
        if confidence < policy.min_weighted_confidence:
            blockers.add("teacher_confidence_below_minimum")
        if lower_bound is None:
            blockers.add("missing_after_cost_lower_bound")
        elif lower_bound <= 0.0:
            blockers.add("nonpositive_after_cost_lower_bound")

    _apply_student_abstention(
        blockers,
        context=context,
        selected=selected,
        router_prediction=router_prediction,
        outcome_forecast=outcome_forecast,
        policy=policy,
    )

    hard_blockers = {
        blocker
        for blocker in blockers
        if blocker.startswith(("context_", "duplicate_", "missing_exact_mode", "missing_critic", "future_", "stale_"))
        or blocker
        in {
            "hard_critic_veto",
            "math_v2_required",
            "point_in_time_teacher_evidence_required",
            "point_in_time_critic_evidence_required",
            "local_critic_authority_required",
        }
    }
    if hard_blockers:
        status = CouncilStatus.BLOCKED
    elif blockers == {"discovery_only_cannot_vote", "teacher_council_prefers_no_entry"}:
        status = CouncilStatus.RESEARCH_ONLY
    elif blockers:
        status = CouncilStatus.ABSTAIN
    else:
        status = CouncilStatus.SHADOW_TEST

    action = vote["action"] if status == CouncilStatus.SHADOW_TEST else TeacherAction.ABSTAIN
    selected_mode = selected.exact_mode if selected is not None else None
    reason = "complete_math_v2_teacher_packet_ready_for_shadow_test" if not blockers else ";".join(sorted(blockers))
    evidence_paths = _evidence_paths(proposals, assessments, router_prediction, outcome_forecast)
    token = sha256(
        "|".join(
            [
                context.context_id,
                status.value,
                action.value,
                reason,
                *proposal_ids,
                *assessment_ids,
            ]
        ).encode("utf-8")
    ).hexdigest()[:20]
    return CouncilDecision(
        decision_id=f"teacher_decision_{token}",
        context=context,
        status=status,
        action=action,
        research_preference=research_preference,
        selected_mode=selected_mode,
        weighted_confidence=confidence,
        teacher_disagreement=disagreement,
        uncertainty_penalty=uncertainty,
        lower_bound_net_return=lower_bound,
        proposal_ids=proposal_ids,
        assessment_ids=assessment_ids,
        blocker_codes=tuple(sorted(blockers)),
        reason=reason,
        evidence_paths=evidence_paths or ("reports/orchestration/teacher_council/teacher_registry.csv",),
        created_at=now,
    )


def _integrity_blockers(
    *,
    context: CouncilContext,
    proposals: tuple[TeacherProposal, ...],
    assessments: tuple[CriticAssessment, ...],
    policy: TeacherCouncilPolicy,
    now: datetime,
) -> set[str]:
    blockers: set[str] = set()
    if len({proposal.proposal_id for proposal in proposals}) != len(proposals):
        blockers.add("duplicate_proposal_id")
    if len({proposal.exact_mode for proposal in proposals}) != len(proposals):
        blockers.add("duplicate_exact_mode_teacher")
    if len({assessment.assessment_id for assessment in assessments}) != len(assessments):
        blockers.add("duplicate_assessment_id")
    if len({assessment.critic_type for assessment in assessments}) != len(assessments):
        blockers.add("duplicate_critic_type")
    if any(proposal.context.context_id != context.context_id for proposal in proposals):
        blockers.add("context_teacher_mismatch")
    if any(assessment.context.context_id != context.context_id for assessment in assessments):
        blockers.add("context_critic_mismatch")

    observed_modes = {proposal.exact_mode for proposal in proposals}
    for mode in ExactMode:
        if mode not in observed_modes:
            blockers.add(f"missing_exact_mode:{mode.value}")
    observed_critics = {assessment.critic_type for assessment in assessments}
    for critic_type in REQUIRED_CRITICS:
        if critic_type not in observed_critics:
            blockers.add(f"missing_critic:{critic_type.value}")

    for proposal in proposals:
        _add_freshness_blockers(
            blockers,
            prefix=f"teacher:{proposal.exact_mode.value}",
            timestamp=proposal.source_timestamp,
            now=now,
            max_age_hours=policy.max_evidence_age_hours,
        )
        if proposal.authority == EvidenceAuthority.LOCAL_POINT_IN_TIME:
            if proposal.math_version != policy.required_math_version:
                blockers.add("math_v2_required")
            if proposal.point_in_time_status != "confirmed":
                blockers.add("point_in_time_teacher_evidence_required")
        if proposal.blockers:
            blockers.update(f"teacher_blocker:{code}" for code in proposal.blockers)

    for assessment in assessments:
        _add_freshness_blockers(
            blockers,
            prefix=f"critic:{assessment.critic_type.value}",
            timestamp=assessment.source_timestamp,
            now=now,
            max_age_hours=policy.max_evidence_age_hours,
        )
        if assessment.authority not in {
            EvidenceAuthority.LOCAL_POINT_IN_TIME,
            EvidenceAuthority.REALIZED_OUTCOME,
        }:
            blockers.add("local_critic_authority_required")
        if assessment.point_in_time_status != "confirmed":
            blockers.add("point_in_time_critic_evidence_required")
        blockers.update(f"critic_blocker:{code}" for code in assessment.blocker_codes)
    return blockers


def _add_freshness_blockers(
    blockers: set[str], *, prefix: str, timestamp: datetime, now: datetime, max_age_hours: float
) -> None:
    age_hours = (now - timestamp).total_seconds() / 3600.0
    normalized = prefix.lower().replace(" ", "_")
    if age_hours < 0:
        blockers.add(f"future_{normalized}")
    elif age_hours > max_age_hours:
        blockers.add(f"stale_{normalized}")


def _is_acceptance_teacher(
    proposal: TeacherProposal, *, policy: TeacherCouncilPolicy, now: datetime
) -> bool:
    age_hours = (now - proposal.source_timestamp).total_seconds() / 3600.0
    return (
        proposal.authority == EvidenceAuthority.LOCAL_POINT_IN_TIME
        and proposal.point_in_time_status == "confirmed"
        and proposal.math_version == policy.required_math_version
        and not proposal.blockers
        and 0.0 <= age_hours <= policy.max_evidence_age_hours
    )


def _teacher_vote(proposals: tuple[TeacherProposal, ...]) -> dict[str, object]:
    if not proposals:
        return {
            "action": TeacherAction.ABSTAIN,
            "selected": None,
            "confidence": 0.0,
            "disagreement": 1.0,
            "uncertainty": 1.0,
            "lower_bound": None,
        }

    weights = {proposal.proposal_id: proposal.confidence * (1.0 - proposal.uncertainty) for proposal in proposals}
    action_weights = {action: 0.0 for action in TeacherAction}
    for proposal in proposals:
        action_weights[proposal.proposed_action] += weights[proposal.proposal_id]
    winner = max(action_weights, key=lambda action: (action_weights[action], action.value))
    total_weight = sum(weights.values())
    winner_weight = action_weights[winner]
    supporters = tuple(proposal for proposal in proposals if proposal.proposed_action == winner)
    selected = max(
        supporters,
        key=lambda proposal: (
            weights[proposal.proposal_id],
            proposal.lower_bound_net_return if proposal.lower_bound_net_return is not None else float("-inf"),
            proposal.exact_mode.value,
        ),
    )
    confidence = _weighted_average(
        ((proposal.confidence, weights[proposal.proposal_id]) for proposal in supporters),
        default=0.0,
    )
    uncertainty = _weighted_average(
        ((proposal.uncertainty, weights[proposal.proposal_id]) for proposal in supporters),
        default=1.0,
    )
    lower_bounds = tuple(
        (proposal.lower_bound_net_return, weights[proposal.proposal_id])
        for proposal in supporters
        if proposal.lower_bound_net_return is not None
    )
    lower_bound = _weighted_average(lower_bounds, default=None)
    disagreement = 1.0 if total_weight <= 0.0 else 1.0 - (winner_weight / total_weight)
    return {
        "action": winner,
        "selected": selected,
        "confidence": min(1.0, max(0.0, confidence)),
        "disagreement": min(1.0, max(0.0, disagreement)),
        "uncertainty": min(1.0, max(0.0, uncertainty)),
        "lower_bound": lower_bound,
    }


def _weighted_average(values, *, default):
    rows = tuple(values)
    total = sum(weight for _, weight in rows)
    if not rows or total <= 0.0:
        return default
    return sum(float(value) * weight for value, weight in rows) / total


def _apply_student_abstention(
    blockers: set[str],
    *,
    context: CouncilContext,
    selected: TeacherProposal | None,
    router_prediction: StudentRouterPrediction | None,
    outcome_forecast: StudentOutcomeForecast | None,
    policy: TeacherCouncilPolicy,
) -> None:
    if router_prediction is not None:
        if router_prediction.context_id != context.context_id:
            blockers.add("student_router_context_mismatch")
        if router_prediction.training_support_score < policy.min_student_support:
            blockers.add("student_training_support_below_minimum")
        if router_prediction.feature_completeness_score < policy.min_feature_completeness:
            blockers.add("student_feature_completeness_below_minimum")
        if router_prediction.uncertainty > policy.max_uncertainty:
            blockers.add("student_router_uncertainty_above_limit")
        top_label = max(router_prediction.mode_probabilities, key=router_prediction.mode_probabilities.get)
        if top_label == TeacherAction.ABSTAIN.value:
            blockers.add("student_router_prefers_abstain")
        elif selected is not None and top_label != selected.exact_mode.value:
            blockers.add("student_teacher_mode_disagreement")
    if outcome_forecast is not None:
        if outcome_forecast.context_id != context.context_id:
            blockers.add("student_outcome_context_mismatch")
        if outcome_forecast.lower_bound_net_return <= 0.0:
            blockers.add("student_nonpositive_after_cost_lower_bound")
        if outcome_forecast.uncertainty > policy.max_uncertainty:
            blockers.add("student_outcome_uncertainty_above_limit")
        if outcome_forecast.structural_break_probability > policy.max_structural_break_probability:
            blockers.add("student_structural_break_risk_above_limit")
        if outcome_forecast.execution_failure_probability > policy.max_execution_failure_probability:
            blockers.add("student_execution_failure_risk_above_limit")


def _evidence_paths(
    proposals: tuple[TeacherProposal, ...],
    assessments: tuple[CriticAssessment, ...],
    router_prediction: StudentRouterPrediction | None,
    outcome_forecast: StudentOutcomeForecast | None,
) -> tuple[str, ...]:
    paths: list[str] = []
    for record in (*proposals, *assessments):
        paths.extend(record.evidence_paths)
    if router_prediction is not None:
        paths.extend(router_prediction.evidence_paths)
    if outcome_forecast is not None:
        paths.extend(outcome_forecast.evidence_paths)
    return tuple(dict.fromkeys(path for path in paths if str(path).strip()))
