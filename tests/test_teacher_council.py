from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.teacher_contracts import (
    CouncilContext,
    CriticAssessment,
    CriticType,
    CriticVerdict,
    EvidenceAuthority,
    ExactMode,
    StudentRouterPrediction,
    TeacherAction,
    TeacherProposal,
)
from quant_platform.orchestration.teacher_council import arbitrate_teacher_council
from quant_platform.statistics.math_v2 import MATH_VERSION

NOW = datetime(2026, 8, 6, 18, tzinfo=UTC)


def _context() -> CouncilContext:
    return CouncilContext(
        pair="SOL-USD/WLD-USD",
        venue="hyperliquid",
        timeframe="1d",
        lookback=320,
        source_snapshot_id="snapshot-1",
        source_timestamp=NOW,
    )


def _proposal(
    mode: ExactMode,
    *,
    action: TeacherAction = TeacherAction.SHORT_X_LONG_Y,
    authority: EvidenceAuthority = EvidenceAuthority.LOCAL_POINT_IN_TIME,
    source_system: str = "hyperliquid_local_replay",
    math_version: str = MATH_VERSION,
) -> TeacherProposal:
    context = _context()
    return TeacherProposal(
        proposal_id=f"proposal-{mode.name.lower()}",
        context=context,
        candidate=CandidateIdentity(
            pair=context.pair,
            venue=context.venue,
            strategy_family=mode.value,
            timeframe=context.timeframe,
            lookback=context.lookback,
            formula_version=f"{mode.name.lower()}-v2",
        ),
        teacher_id=f"{mode.name.lower()}-teacher",
        exact_mode=mode,
        proposed_action=action,
        confidence=0.80,
        uncertainty=0.10,
        expected_net_return=0.025,
        lower_bound_net_return=0.008,
        expected_holding_bars=8,
        entry_style="mode-specific entry",
        exit_style="mode-specific exit",
        invalidation_condition="dependency break",
        required_regime="range",
        source_system=source_system,
        authority=authority,
        point_in_time_status="confirmed",
        formula_version=f"{mode.name.lower()}-v2",
        math_version=math_version,
        source_timestamp=NOW,
        evidence_paths=(f"reports/evidence/{mode.name.lower()}.csv",),
        created_at=NOW,
    )


def _critics(*, veto: CriticType | None = None) -> tuple[CriticAssessment, ...]:
    context = _context()
    return tuple(
        CriticAssessment(
            assessment_id=f"critic-{critic_type.value}",
            context=context,
            critic_id=f"{critic_type.value}-critic",
            critic_type=critic_type,
            verdict=CriticVerdict.VETO if critic_type == veto else CriticVerdict.PASS,
            score=0.9,
            reason="verified" if critic_type != veto else "unsafe",
            authority=(
                EvidenceAuthority.REALIZED_OUTCOME
                if critic_type == CriticType.OUTCOME
                else EvidenceAuthority.LOCAL_POINT_IN_TIME
            ),
            point_in_time_status="confirmed",
            source_timestamp=NOW,
            blocker_codes=("unsafe",) if critic_type == veto else (),
            evidence_paths=(f"reports/evidence/{critic_type.value}.csv",),
            created_at=NOW,
        )
        for critic_type in CriticType
    )


def test_complete_math_v2_council_can_only_authorize_shadow_test():
    proposals = tuple(_proposal(mode) for mode in ExactMode)
    decision = arbitrate_teacher_council(
        context=_context(),
        proposals=proposals,
        assessments=_critics(),
        now=NOW,
    )

    assert decision.status == "SHADOW_TEST"
    assert decision.action == "short_x_long_y"
    assert decision.selected_mode in set(ExactMode)
    assert decision.promotion_allowed is False
    assert decision.execution_allowed is False


def test_previous_math_version_requires_new_evidence_after_repair():
    proposals = tuple(_proposal(mode, math_version="math-v2.1-y-on-x") for mode in ExactMode)
    decision = arbitrate_teacher_council(
        context=_context(), proposals=proposals, assessments=_critics(), now=NOW
    )
    assert decision.status == "BLOCKED"
    assert "math_v2_required" in decision.blocker_codes
    assert decision.execution_allowed is False


def test_missing_mode_math_v1_or_critic_veto_blocks_council():
    missing = tuple(_proposal(mode) for mode in tuple(ExactMode)[:-1])
    missing_decision = arbitrate_teacher_council(
        context=_context(), proposals=missing, assessments=_critics(), now=NOW
    )
    assert missing_decision.status == "BLOCKED"
    assert any(code.startswith("missing_exact_mode") for code in missing_decision.blocker_codes)

    math_v1 = tuple(_proposal(mode, math_version="math-v1") for mode in ExactMode)
    math_decision = arbitrate_teacher_council(
        context=_context(), proposals=math_v1, assessments=_critics(), now=NOW
    )
    assert math_decision.status == "BLOCKED"
    assert "math_v2_required" in math_decision.blocker_codes

    vetoed = arbitrate_teacher_council(
        context=_context(),
        proposals=tuple(_proposal(mode) for mode in ExactMode),
        assessments=_critics(veto=CriticType.EXECUTION),
        now=NOW,
    )
    assert vetoed.status == "BLOCKED"
    assert vetoed.action == "abstain"


def test_wizard_evidence_is_discovery_only_and_cannot_vote():
    with pytest.raises(ValidationError, match="discovery-only"):
        _proposal(
            ExactMode.COPULA,
            authority=EvidenceAuthority.LOCAL_POINT_IN_TIME,
            source_system="crypto_wizards_dashboard",
        )

    discovery = tuple(
        _proposal(
            mode,
            authority=EvidenceAuthority.DISCOVERY_ONLY,
            source_system="crypto_wizards_dashboard",
            math_version="wizard-display-v1",
        )
        for mode in ExactMode
    )
    decision = arbitrate_teacher_council(
        context=_context(), proposals=discovery, assessments=_critics(), now=NOW
    )
    assert decision.status == "RESEARCH_ONLY"
    assert decision.action == "abstain"
    assert "discovery_only_cannot_vote" in decision.blocker_codes


def test_student_router_can_only_add_an_abstention():
    probabilities = {mode.value: 0.0 for mode in ExactMode}
    probabilities[ExactMode.COPULA.value] = 0.90
    probabilities[TeacherAction.ABSTAIN.value] = 0.10
    router = StudentRouterPrediction(
        prediction_id="router-1",
        context_id=_context().context_id,
        model_version="router-v1",
        feature_schema_version="features-v1",
        mode_probabilities=probabilities,
        uncertainty=0.10,
        training_support_score=0.90,
        feature_completeness_score=1.0,
        evidence_paths=("models/router/metrics.json",),
        generated_at=NOW,
    )
    decision = arbitrate_teacher_council(
        context=_context(),
        proposals=tuple(_proposal(mode) for mode in ExactMode),
        assessments=_critics(),
        router_prediction=router,
        now=NOW,
    )
    assert decision.status == "ABSTAIN"
    assert decision.action == "abstain"
    assert "student_teacher_mode_disagreement" in decision.blocker_codes
