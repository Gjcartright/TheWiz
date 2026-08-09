"""Versioned contracts for the deterministic dynamic-agent control plane."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DYNAMIC_AGENT_SCHEMA_VERSION = "dynamic_agents.v1"


class TaskStatus(StrEnum):
    QUEUED = "queued"
    LEASED = "leased"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    EXPIRED = "expired"
    FAILED = "failed"


class DecisionBucket(StrEnum):
    REJECT = "REJECT"
    FETCH_MORE_DATA = "FETCH_MORE_DATA"
    WATCH = "WATCH"
    TEST = "TEST"
    PAPER_AUTHORIZED = "PAPER_AUTHORIZED"


class OutcomeSource(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


def normalize_symbol(value: str) -> str:
    """Normalize a leg without guessing its venue-specific market symbol."""

    normalized = re.sub(r"\s+", "", str(value or "").upper()).replace("_", "-")
    if not normalized:
        raise ValueError("symbol cannot be blank")
    return normalized


def normalize_pair(value: str) -> str:
    """Normalize pair separators while preserving leg order for hedge semantics."""

    raw = str(value or "").strip()
    parts = [part for part in re.split(r"\s*/\s*", raw) if part]
    if len(parts) != 2:
        raise ValueError("pair must contain exactly two legs separated by '/'")
    return f"{normalize_symbol(parts[0])}/{normalize_symbol(parts[1])}"


def candidate_id_for(
    *,
    pair: str,
    venue: str,
    strategy_family: str,
    timeframe: str,
    lookback: int,
    formula_version: str,
) -> str:
    """Create a stable ID for one exact strategy configuration, not just a pair."""

    canonical = "|".join(
        [
            normalize_pair(pair),
            str(venue or "").strip().lower(),
            str(strategy_family or "").strip().lower(),
            str(timeframe or "").strip().lower(),
            str(int(lookback)),
            str(formula_version or "").strip().lower(),
        ]
    )
    if "||" in canonical or canonical.endswith("|"):
        raise ValueError("candidate identity fields cannot be blank")
    return f"candidate_{sha256(canonical.encode('utf-8')).hexdigest()[:20]}"


class CandidateIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    pair: str
    venue: str = Field(min_length=1)
    strategy_family: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    lookback: int = Field(gt=0)
    formula_version: str = Field(min_length=1)
    candidate_id: str = ""

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, value: str) -> str:
        return normalize_pair(value)

    @field_validator("venue", "strategy_family", "timeframe", "formula_version")
    @classmethod
    def _require_trimmed_text(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("identity field cannot be blank")
        return normalized

    @model_validator(mode="after")
    def _set_or_validate_candidate_id(self) -> "CandidateIdentity":
        expected = candidate_id_for(
            pair=self.pair,
            venue=self.venue,
            strategy_family=self.strategy_family,
            timeframe=self.timeframe,
            lookback=self.lookback,
            formula_version=self.formula_version,
        )
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("candidate_id does not match the canonical identity")
        object.__setattr__(self, "candidate_id", expected)
        return self


class TaskCard(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    task_id: str = Field(min_length=1)
    candidate: CandidateIdentity
    assigned_agent: str = Field(min_length=1)
    action: str = Field(min_length=1)
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    status: TaskStatus = TaskStatus.QUEUED
    idempotency_key: str = Field(min_length=1)
    evidence_paths: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    lease_owner: str = ""
    lease_expires_at: datetime | None = None
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)

    @field_validator("created_at", "expires_at", "lease_expires_at")
    @classmethod
    def _require_utc_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class EvidencePacket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    packet_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    producing_agent: str = Field(min_length=1)
    evidence_type: str = Field(min_length=1)
    event_timestamp: datetime
    source_timestamp: datetime
    point_in_time_status: Literal["confirmed", "unknown", "hindsight_blocked"]
    formula_version: str = Field(min_length=1)
    test_configuration_hash: str = Field(min_length=1)
    evidence_content_hash: str = Field(min_length=1)
    finding: str = Field(min_length=1)
    confidence_band: Literal["low", "medium", "high"] = "low"
    blockers: tuple[str, ...] = ()
    evidence_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("event_timestamp", "source_timestamp")
    @classmethod
    def _require_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class ComparisonEvent(BaseModel):
    """Immutable comparison of sequential and dynamic Copula decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    event_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    configuration_id: str = Field(min_length=1)
    source_snapshot_id: str = Field(min_length=1)
    pair: str
    venue: str = Field(min_length=1)
    timeframe: str = Field(min_length=1)
    source_timestamp: datetime
    sequential_decision: DecisionBucket
    sequential_reason: str = Field(min_length=1)
    sequential_evidence_path: str = ""
    dynamic_decision: DecisionBucket
    dynamic_reason: str = Field(min_length=1)
    dynamic_veto_count: int = Field(ge=0)
    dynamic_evidence_count: int = Field(ge=0)
    evidence_packet_ids: tuple[str, ...] = Field(min_length=1)
    comparison: Literal["exact_agreement", "both_non_promoting", "requires_review"]
    shadow_only: Literal[True] = True
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("pair")
    @classmethod
    def _normalize_pair(cls, value: str) -> str:
        return normalize_pair(value)

    @field_validator("source_timestamp", "created_at")
    @classmethod
    def _require_timezones(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class VetoRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    veto_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    vetoing_agent: str = Field(min_length=1)
    blocker_code: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    evidence_paths: tuple[str, ...] = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DecisionRecord(BaseModel):
    """Arbiter output. Dynamic agents cannot authorize live execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    decision_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    decision: DecisionBucket
    reason: str = Field(min_length=1)
    evidence_packet_ids: tuple[str, ...] = Field(min_length=1)
    veto_ids: tuple[str, ...] = ()
    next_task_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _enforce_veto_policy(self) -> "DecisionRecord":
        if self.decision == DecisionBucket.PAPER_AUTHORIZED:
            raise ValueError("dynamic-agent decisions cannot authorize paper execution")
        return self


class OutcomeRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[DYNAMIC_AGENT_SCHEMA_VERSION] = DYNAMIC_AGENT_SCHEMA_VERSION
    outcome_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    source: OutcomeSource
    observed_at: datetime
    label: str = Field(min_length=1)
    evidence_paths: tuple[str, ...] = Field(min_length=1)

    @field_validator("observed_at")
    @classmethod
    def _require_observed_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return value


class AgentCapability(BaseModel):
    """Dynamic research agents have no direct execution capability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent: str = Field(min_length=1)
    allowed_actions: tuple[str, ...] = ()
    can_read_secrets: bool = False
    can_submit_orders: bool = False

    @model_validator(mode="after")
    def _forbid_direct_execution(self) -> "AgentCapability":
        if self.can_read_secrets or self.can_submit_orders:
            raise ValueError("dynamic-agent capabilities cannot include secrets or order submission")
        return self
