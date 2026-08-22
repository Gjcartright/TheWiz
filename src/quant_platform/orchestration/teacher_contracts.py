"""Versioned contracts for the shadow-only teacher council and student stack."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.economic_contract import (
    EXACT_MODES,
    ExactMode,
    TradeAction as TeacherAction,
    normalize_exact_mode,
)
from quant_platform.orchestration.contracts import CandidateIdentity, normalize_pair
from quant_platform.performance_math import MATH_VERSION


TEACHER_COUNCIL_SCHEMA_VERSION = "teacher_council.v1"
MATH_V2 = MATH_VERSION


class EvidenceAuthority(StrEnum):
    DISCOVERY_ONLY = "discovery_only"
    LOCAL_POINT_IN_TIME = "local_point_in_time"
    REALIZED_OUTCOME = "realized_outcome"


class CriticType(StrEnum):
    DEPENDENCY = "dependency"
    REGIME = "regime"
    RISK = "risk"
    COST = "cost"
    EXECUTION = "execution"
    OUTCOME = "outcome"


REQUIRED_CRITICS: tuple[CriticType, ...] = tuple(CriticType)


class CriticVerdict(StrEnum):
    PASS = "pass"
    WARN = "warn"
    VETO = "veto"
    UNKNOWN = "unknown"


class CouncilStatus(StrEnum):
    SHADOW_TEST = "SHADOW_TEST"
    RESEARCH_ONLY = "RESEARCH_ONLY"
    BLOCKED = "BLOCKED"
    ABSTAIN = "ABSTAIN"


def council_context_id_for(
    *,
    pair: str,
    venue: str,
    timeframe: str,
    lookback: int,
    source_snapshot_id: str,
    run_id: str = "",
    candidate_set_id: str = "",
    setup_identity: str = "",
) -> str:
    identity = [
        normalize_pair(pair),
        str(venue or "").strip().lower(),
        str(timeframe or "").strip().lower(),
        str(int(lookback)),
        str(source_snapshot_id or "").strip(),
    ]
    lineage = [str(run_id or "").strip(), str(candidate_set_id or "").strip(), str(setup_identity or "").strip()]
    if any(lineage) and not all(lineage):
        raise ValueError("run lineage fields must be supplied together")
    canonical = "|".join([*identity, *lineage] if all(lineage) else identity)
    if "||" in canonical or canonical.endswith("|"):
        raise ValueError("council context identity fields cannot be blank")
    return f"teacher_context_{sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


class CouncilContext(BaseModel):
    """One pair/timeframe snapshot shared by all seven exact-mode teachers."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    pair: str
    venue: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    lookback: int = Field(gt=0)
    source_snapshot_id: str = Field(min_length=1)
    source_timestamp: datetime
    run_id: str = ""
    candidate_set_id: str = ""
    setup_identity: str = ""
    context_id: str = ""

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, value: str) -> str:
        return normalize_pair(value)

    @field_validator("venue", "timeframe", "source_snapshot_id")
    @classmethod
    def _require_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("context identity field cannot be blank")
        return normalized

    @field_validator("source_timestamp")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _set_or_validate_context_id(self) -> "CouncilContext":
        expected = council_context_id_for(
            pair=self.pair,
            venue=self.venue,
            timeframe=self.timeframe,
            lookback=self.lookback,
            source_snapshot_id=self.source_snapshot_id,
            run_id=self.run_id,
            candidate_set_id=self.candidate_set_id,
            setup_identity=self.setup_identity,
        )
        if self.context_id and self.context_id != expected:
            raise ValueError("context_id does not match the canonical context identity")
        object.__setattr__(self, "context_id", expected)
        return self


class TeacherProposal(BaseModel):
    """A mode specialist's auditable proposal; it has no execution authority."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    proposal_id: str = Field(min_length=1)
    context: CouncilContext
    candidate: CandidateIdentity
    teacher_id: str = Field(min_length=1)
    exact_mode: ExactMode
    proposed_action: TeacherAction
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    expected_net_return: float | None = None
    lower_bound_net_return: float | None = None
    expected_holding_bars: float | None = Field(default=None, ge=0.0)
    entry_style: str = ""
    exit_style: str = ""
    invalidation_condition: str = ""
    required_regime: str = ""
    source_system: str = Field(min_length=1)
    authority: EvidenceAuthority
    point_in_time_status: Literal["confirmed", "unknown", "hindsight_blocked"]
    formula_version: str = Field(min_length=1)
    math_version: str = Field(min_length=1)
    source_timestamp: datetime
    blockers: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("exact_mode", mode="before")
    @classmethod
    def _normalize_mode(cls, value: str | ExactMode) -> ExactMode:
        return normalize_exact_mode(value)

    @field_validator("source_timestamp", "created_at")
    @classmethod
    def _require_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_lineage_and_identity(self) -> "TeacherProposal":
        if self.candidate.pair != self.context.pair:
            raise ValueError("teacher candidate pair does not match council context")
        if self.candidate.venue.lower() != self.context.venue.lower():
            raise ValueError("teacher candidate venue does not match council context")
        if self.candidate.timeframe.lower() != self.context.timeframe.lower():
            raise ValueError("teacher candidate timeframe does not match council context")
        if self.candidate.lookback != self.context.lookback:
            raise ValueError("teacher candidate lookback does not match council context")
        if normalize_exact_mode(self.candidate.strategy_family) != self.exact_mode:
            raise ValueError("teacher candidate strategy family does not match exact mode")
        if "wizard" in self.source_system.lower() and self.authority != EvidenceAuthority.DISCOVERY_ONLY:
            raise ValueError("Crypto Wizards evidence is discovery-only")
        if self.lower_bound_net_return is not None and self.expected_net_return is not None:
            if self.lower_bound_net_return > self.expected_net_return:
                raise ValueError("lower-bound return cannot exceed expected return")
        if self.proposed_action not in {TeacherAction.ABSTAIN, TeacherAction.FLAT}:
            required = (self.entry_style, self.exit_style, self.invalidation_condition)
            if not all(str(value).strip() for value in required):
                raise ValueError("active teacher proposals require entry, exit, and invalidation logic")
        return self


class CriticAssessment(BaseModel):
    """An independent critic assessment. Vetoes override every model score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    assessment_id: str = Field(min_length=1)
    context: CouncilContext
    critic_id: str = Field(min_length=1)
    critic_type: CriticType
    verdict: CriticVerdict
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = Field(min_length=1)
    authority: EvidenceAuthority
    point_in_time_status: Literal["confirmed", "unknown", "hindsight_blocked"]
    source_timestamp: datetime
    blocker_codes: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("source_timestamp", "created_at")
    @classmethod
    def _require_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class StudentRouterPrediction(BaseModel):
    """Advisory mixture-of-experts output that may abstain but never authorize."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    prediction_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    feature_schema_version: str = Field(min_length=1)
    mode_probabilities: dict[str, float]
    uncertainty: float = Field(ge=0.0, le=1.0)
    training_support_score: float = Field(ge=0.0, le=1.0)
    feature_completeness_score: float = Field(ge=0.0, le=1.0)
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    shadow_only: Literal[True] = True

    @field_validator("generated_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_probability_distribution(self) -> "StudentRouterPrediction":
        expected = {mode.value for mode in EXACT_MODES} | {TeacherAction.ABSTAIN.value}
        if set(self.mode_probabilities) != expected:
            raise ValueError("router probabilities must cover all exact modes plus abstain")
        if any(value < 0.0 or value > 1.0 for value in self.mode_probabilities.values()):
            raise ValueError("router probabilities must be between zero and one")
        if abs(sum(self.mode_probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("router probabilities must sum to one")
        return self


class StudentOutcomeForecast(BaseModel):
    """Distributional after-cost forecast used only to tighten council abstention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    forecast_id: str = Field(min_length=1)
    context_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    probability_positive: float = Field(ge=0.0, le=1.0)
    expected_net_return: float
    lower_bound_net_return: float
    median_net_return: float
    upper_bound_net_return: float
    expected_mae: float = Field(ge=0.0)
    expected_mfe: float = Field(ge=0.0)
    expected_holding_bars: float = Field(ge=0.0)
    structural_break_probability: float = Field(ge=0.0, le=1.0)
    execution_failure_probability: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    shadow_only: Literal[True] = True

    @field_validator("generated_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_interval(self) -> "StudentOutcomeForecast":
        if not self.lower_bound_net_return <= self.median_net_return <= self.upper_bound_net_return:
            raise ValueError("outcome forecast quantiles are not ordered")
        return self


class CouncilDecision(BaseModel):
    """Final council result. It can enter shadow testing but cannot execute."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[TEACHER_COUNCIL_SCHEMA_VERSION] = TEACHER_COUNCIL_SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    context: CouncilContext
    status: CouncilStatus
    action: TeacherAction
    research_preference: TeacherAction
    selected_mode: ExactMode | None = None
    weighted_confidence: float = Field(ge=0.0, le=1.0)
    teacher_disagreement: float = Field(ge=0.0, le=1.0)
    uncertainty_penalty: float = Field(ge=0.0, le=1.0)
    lower_bound_net_return: float | None = None
    proposal_ids: tuple[str, ...] = ()
    assessment_ids: tuple[str, ...] = ()
    blocker_codes: tuple[str, ...] = ()
    reason: str = Field(min_length=1)
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    shadow_only: Literal[True] = True
    promotion_allowed: Literal[False] = False
    execution_allowed: Literal[False] = False

    @field_validator("created_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _enforce_abstention_boundary(self) -> "CouncilDecision":
        if self.status != CouncilStatus.SHADOW_TEST and self.action != TeacherAction.ABSTAIN:
            raise ValueError("non-shadow council decisions must abstain")
        if self.status == CouncilStatus.SHADOW_TEST and self.action == TeacherAction.ABSTAIN:
            raise ValueError("shadow-test decisions require a proposed action")
        return self
