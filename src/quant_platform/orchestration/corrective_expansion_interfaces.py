"""Pure, fail-closed Phase 00 contracts for future platform expansion.

This module intentionally contains no I/O, credential access, scheduling,
publication, permit issuance, or order submission.  It defines immutable
interfaces and deterministic validators for evidence produced elsewhere.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.orchestration.identity_ontology import (
    IDENTITY_ONTOLOGY_VERSION,
    AccountScopeIdentity,
    VenueIdentity,
    VenueInstrumentIdentity,
    VenueProductLaneIdentity,
)
from quant_platform.orchestration.venue_policy_registry import (
    PHASE00_VENUE_POLICIES,
    VENUE_POLICY_SCHEMA_VERSION,
    VenueLane,
    venue_policy,
)

EXPANSION_INTERFACE_SCHEMA_VERSION = "thewiz.phase00_expansion_interfaces.v1"
EXPANSION_CONTRACT_VERSION = "2026-08-21.1"

DEFAULT_INSTRUMENT_SOURCE_PRECEDENCE = (
    "venue_public_api",
    "ccxt_public",
    "crypto_wizards",
    "apify_public",
)
INSTRUMENT_CONSENSUS_FIELDS = (
    "venue_id",
    "product_lane_id",
    "venue_market_id",
    "unified_symbol",
    "base_asset_id",
    "quote_asset_id",
    "settle_asset_id",
    "instrument_id",
)


def _payload_hash(kind: str, payload: object) -> str:
    if isinstance(payload, BaseModel):
        payload = payload.model_dump(mode="json")
    encoded = json.dumps(
        {"kind": kind, "payload": payload},
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


def _identifier(kind: str, payload: object) -> str:
    return f"{kind}_{_payload_hash(kind, payload)[:24]}"


def _aware(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value


def _decimal(value: object, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be a finite decimal")
    return result


def _unique(items: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[EXPANSION_INTERFACE_SCHEMA_VERSION] = EXPANSION_INTERFACE_SCHEMA_VERSION
    contract_version: Literal[EXPANSION_CONTRACT_VERSION] = EXPANSION_CONTRACT_VERSION


# ---------------------------------------------------------------------------
# Canonical instrument registry and source reconciliation


class InstrumentSourceObservation(_ContractModel):
    source_system: str = Field(min_length=1)
    observed_at: datetime
    freshness_deadline: datetime
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    instrument: VenueInstrumentIdentity

    @field_validator("source_system")
    @classmethod
    def _source_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("source_system cannot be blank")
        return normalized

    @field_validator("observed_at", "freshness_deadline")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime, info) -> datetime:
        return _aware(value, field=info.field_name)

    @model_validator(mode="after")
    def _freshness_window(self) -> InstrumentSourceObservation:
        if self.freshness_deadline <= self.observed_at:
            raise ValueError("freshness_deadline must follow observed_at")
        return self


class InstrumentReconciliationPolicy(_ContractModel):
    policy_id: str = Field(min_length=1)
    source_precedence: tuple[str, ...] = Field(min_length=1)
    consensus_fields: tuple[
        Literal[
            "venue_id",
            "product_lane_id",
            "venue_market_id",
            "unified_symbol",
            "base_asset_id",
            "quote_asset_id",
            "settle_asset_id",
            "instrument_id",
        ],
        ...,
    ] = INSTRUMENT_CONSENSUS_FIELDS
    reject_unknown_sources: Literal[True] = True
    reject_stale_observations: Literal[True] = True
    reject_identity_conflicts: Literal[True] = True

    @field_validator("source_precedence")
    @classmethod
    def _normalize_precedence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip().lower() for item in value)
        if any(not item for item in normalized):
            raise ValueError("source_precedence cannot contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source_precedence must be unique")
        return normalized

    @field_validator("consensus_fields")
    @classmethod
    def _complete_consensus(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if set(value) != set(INSTRUMENT_CONSENSUS_FIELDS):
            raise ValueError("all canonical instrument identity fields require consensus")
        if len(value) != len(set(value)):
            raise ValueError("consensus_fields must be unique")
        return value


class InstrumentReconciliationRequest(_ContractModel):
    policy: InstrumentReconciliationPolicy
    as_of: datetime
    observations: tuple[InstrumentSourceObservation, ...] = Field(min_length=1)

    @field_validator("as_of")
    @classmethod
    def _as_of_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="as_of")


class InstrumentReconciliationReceipt(_ContractModel):
    decision: Literal["RECONCILED", "ABSTAIN"]
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    receipt_id: str = Field(min_length=1)
    canonical_instrument: VenueInstrumentIdentity | None = None
    authoritative_source: str | None = None
    ordered_sources: tuple[str, ...]
    blockers: tuple[str, ...]

    @model_validator(mode="after")
    def _decision_is_consistent(self) -> InstrumentReconciliationReceipt:
        if self.decision == "RECONCILED":
            if self.canonical_instrument is None or not self.authoritative_source:
                raise ValueError("reconciled receipt requires instrument and source")
            if self.blockers:
                raise ValueError("reconciled receipt cannot contain blockers")
        elif self.canonical_instrument is not None or self.authoritative_source is not None:
            raise ValueError("abstention cannot identify an authoritative instrument")
        return self


class CanonicalInstrumentRegistryEntry(_ContractModel):
    instrument: VenueInstrumentIdentity
    reconciliation_receipt_id: str = Field(min_length=1)
    authoritative_source: str = Field(min_length=1)
    source_snapshot_hashes: tuple[str, ...] = Field(min_length=1)
    effective_at: datetime
    registry_entry_id: str = ""

    @field_validator("source_snapshot_hashes")
    @classmethod
    def _snapshot_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("source_snapshot_hashes must be unique")
        if any(
            len(item) != 64 or any(char not in "0123456789abcdef" for char in item)
            for item in value
        ):
            raise ValueError("source_snapshot_hashes must contain lowercase sha256 values")
        return value

    @field_validator("effective_at")
    @classmethod
    def _effective_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="effective_at")

    @model_validator(mode="after")
    def _bind_registry_entry(self) -> CanonicalInstrumentRegistryEntry:
        payload = self.model_dump(mode="json", exclude={"registry_entry_id"})
        expected = _identifier("instrument_registry", payload)
        if self.registry_entry_id and self.registry_entry_id != expected:
            raise ValueError("registry_entry_id does not match canonical payload")
        object.__setattr__(self, "registry_entry_id", expected)
        return self


def reconcile_instrument(
    request: InstrumentReconciliationRequest,
) -> InstrumentReconciliationReceipt:
    """Reconcile immutable observations or abstain on any evidence ambiguity."""

    blockers: list[str] = []
    precedence = {source: rank for rank, source in enumerate(request.policy.source_precedence)}
    seen_sources: set[str] = set()
    for observation in request.observations:
        if observation.source_system not in precedence:
            blockers.append(f"unknown_source:{observation.source_system}")
        if observation.source_system in seen_sources:
            blockers.append(f"duplicate_source_observation:{observation.source_system}")
        seen_sources.add(observation.source_system)
        if observation.observed_at > request.as_of:
            blockers.append(f"future_observation:{observation.source_system}")
        if request.as_of > observation.freshness_deadline:
            blockers.append(f"stale_observation:{observation.source_system}")

    ordered = tuple(
        sorted(
            (item.source_system for item in request.observations),
            key=lambda source: (precedence.get(source, len(precedence)), source),
        )
    )
    recognized = [item for item in request.observations if item.source_system in precedence]
    recognized.sort(key=lambda item: (precedence[item.source_system], item.source_system))
    canonical = recognized[0] if recognized else None
    if canonical is not None:
        for observation in recognized[1:]:
            for field in request.policy.consensus_fields:
                if getattr(observation.instrument, field) != getattr(canonical.instrument, field):
                    blockers.append(
                        f"identity_conflict:{field}:{canonical.source_system}:{observation.source_system}"
                    )

    blockers_tuple = _unique(tuple(blockers))
    request_hash = _payload_hash("instrument_reconciliation_request", request)
    decision = "ABSTAIN" if blockers_tuple or canonical is None else "RECONCILED"
    receipt_payload = {
        "decision": decision,
        "request_hash": request_hash,
        "canonical_instrument_id": canonical.instrument.instrument_id
        if decision == "RECONCILED" and canonical
        else None,
        "authoritative_source": canonical.source_system
        if decision == "RECONCILED" and canonical
        else None,
        "ordered_sources": ordered,
        "blockers": blockers_tuple,
    }
    return InstrumentReconciliationReceipt(
        decision=decision,
        request_hash=request_hash,
        receipt_id=_identifier("instrument_reconciliation", receipt_payload),
        canonical_instrument=canonical.instrument
        if decision == "RECONCILED" and canonical
        else None,
        authoritative_source=canonical.source_system
        if decision == "RECONCILED" and canonical
        else None,
        ordered_sources=ordered,
        blockers=blockers_tuple,
    )


# ---------------------------------------------------------------------------
# Unknown-exchange onboarding


class OnboardingState(StrEnum):
    DISCOVERED = "discovered"
    QUARANTINED = "quarantined"
    IDENTITY_VERIFIED = "identity_verified"
    PUBLIC_DATA_VERIFIED = "public_data_verified"
    COSTS_VERIFIED = "costs_verified"
    EXECUTION_CONTRACT_VERIFIED = "execution_contract_verified"
    TESTNET_ELIGIBLE = "testnet_eligible"
    REJECTED = "rejected"


LEGAL_ONBOARDING_TRANSITIONS: dict[OnboardingState, frozenset[OnboardingState]] = {
    OnboardingState.DISCOVERED: frozenset({OnboardingState.QUARANTINED, OnboardingState.REJECTED}),
    OnboardingState.QUARANTINED: frozenset(
        {OnboardingState.IDENTITY_VERIFIED, OnboardingState.REJECTED}
    ),
    OnboardingState.IDENTITY_VERIFIED: frozenset(
        {OnboardingState.PUBLIC_DATA_VERIFIED, OnboardingState.REJECTED}
    ),
    OnboardingState.PUBLIC_DATA_VERIFIED: frozenset(
        {OnboardingState.COSTS_VERIFIED, OnboardingState.REJECTED}
    ),
    OnboardingState.COSTS_VERIFIED: frozenset(
        {OnboardingState.EXECUTION_CONTRACT_VERIFIED, OnboardingState.REJECTED}
    ),
    OnboardingState.EXECUTION_CONTRACT_VERIFIED: frozenset(
        {OnboardingState.TESTNET_ELIGIBLE, OnboardingState.REJECTED}
    ),
    OnboardingState.TESTNET_ELIGIBLE: frozenset(),
    OnboardingState.REJECTED: frozenset(),
}
KNOWN_PHASE00_VENUES = frozenset(policy.venue for policy in PHASE00_VENUE_POLICIES)


class UnknownExchangeOnboardingRecord(_ContractModel):
    venue: VenueIdentity
    state: OnboardingState = OnboardingState.DISCOVERED
    discovered_at: datetime
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    onboarding_id: str = ""

    @field_validator("discovered_at")
    @classmethod
    def _discovered_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="discovered_at")

    @model_validator(mode="after")
    def _unknown_and_bound(self) -> UnknownExchangeOnboardingRecord:
        if self.venue.venue_code in KNOWN_PHASE00_VENUES:
            raise ValueError("known Phase 00 venue cannot enter unknown-exchange onboarding")
        payload = self.model_dump(mode="json", exclude={"onboarding_id", "state"})
        expected = _identifier("unknown_venue_onboarding", payload)
        if self.onboarding_id and self.onboarding_id != expected:
            raise ValueError("onboarding_id does not match canonical payload")
        object.__setattr__(self, "onboarding_id", expected)
        return self


class OnboardingTransitionRequest(_ContractModel):
    record: UnknownExchangeOnboardingRecord
    onboarding_id: str = Field(min_length=1)
    from_state: OnboardingState
    to_state: OnboardingState
    requested_at: datetime
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("requested_at")
    @classmethod
    def _requested_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="requested_at")

    @model_validator(mode="after")
    def _bind_current_record(self) -> OnboardingTransitionRequest:
        if self.onboarding_id != self.record.onboarding_id:
            raise ValueError("transition onboarding_id does not match current record")
        if self.from_state != self.record.state:
            raise ValueError("transition from_state does not match current record")
        if self.requested_at < self.record.discovered_at:
            raise ValueError("transition cannot predate discovery")
        return self


class OnboardingTransitionReceipt(_ContractModel):
    onboarding_id: str
    from_state: OnboardingState
    requested_state: OnboardingState
    resulting_state: OnboardingState
    allowed: bool
    blockers: tuple[str, ...]
    evidence_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    transition_receipt_id: str

    @model_validator(mode="after")
    def _decision_consistency(self) -> OnboardingTransitionReceipt:
        if self.allowed and (self.blockers or self.resulting_state != self.requested_state):
            raise ValueError("allowed transition receipt is inconsistent")
        if not self.allowed and (not self.blockers or self.resulting_state != self.from_state):
            raise ValueError("denied transition receipt is inconsistent")
        return self


def evaluate_onboarding_transition(
    request: OnboardingTransitionRequest,
) -> OnboardingTransitionReceipt:
    legal = request.to_state in LEGAL_ONBOARDING_TRANSITIONS[request.from_state]
    blockers = () if legal else ("illegal_onboarding_transition",)
    resulting_state = request.to_state if legal else request.from_state
    payload = {
        **request.model_dump(mode="json"),
        "resulting_state": resulting_state,
        "allowed": legal,
        "blockers": blockers,
    }
    return OnboardingTransitionReceipt(
        onboarding_id=request.onboarding_id,
        from_state=request.from_state,
        requested_state=request.to_state,
        resulting_state=resulting_state,
        allowed=legal,
        blockers=blockers,
        evidence_hash=request.evidence_hash,
        transition_receipt_id=_identifier("onboarding_transition", payload),
    )


# ---------------------------------------------------------------------------
# Deterministic venue selection


class EvidenceLane(StrEnum):
    BACKTEST = "backtest"
    SHADOW = "shadow"
    TESTNET = "testnet"
    CANARY = "canary"
    LIVE = "live"


class VenueCostEvidence(_ContractModel):
    observed_at: datetime
    freshness_deadline: datetime
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    fee_bps: Decimal
    slippage_bps: Decimal
    funding_bps: Decimal
    borrow_short_bps: Decimal
    network_bps: Decimal
    execution_risk_bps: Decimal
    all_in_cost_bps: Decimal

    @field_validator(
        "fee_bps",
        "slippage_bps",
        "funding_bps",
        "borrow_short_bps",
        "network_bps",
        "execution_risk_bps",
        "all_in_cost_bps",
        mode="before",
    )
    @classmethod
    def _finite_costs(cls, value: object, info) -> Decimal:
        return _decimal(value, field=info.field_name)

    @field_validator("observed_at", "freshness_deadline")
    @classmethod
    def _cost_timestamps_are_aware(cls, value: datetime, info) -> datetime:
        return _aware(value, field=info.field_name)

    @model_validator(mode="after")
    def _cost_contract(self) -> VenueCostEvidence:
        if self.freshness_deadline <= self.observed_at:
            raise ValueError("cost freshness_deadline must follow observed_at")
        nonnegative = (
            self.fee_bps,
            self.slippage_bps,
            self.borrow_short_bps,
            self.network_bps,
            self.execution_risk_bps,
        )
        if any(item < 0 for item in nonnegative):
            raise ValueError(
                "fee, slippage, borrow, network, and execution risk costs cannot be negative"
            )
        expected = sum(nonnegative, self.funding_bps)
        if self.all_in_cost_bps != expected:
            raise ValueError("all_in_cost_bps must equal the complete cost formula")
        return self


class VenueSelectorCandidate(_ContractModel):
    lane: VenueLane
    instrument: VenueInstrumentIdentity
    account_scope: AccountScopeIdentity
    observed_at: datetime
    freshness_deadline: datetime
    market_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    costs: VenueCostEvidence
    candidate_id: str = ""

    @field_validator("observed_at", "freshness_deadline")
    @classmethod
    def _market_timestamps_are_aware(cls, value: datetime, info) -> datetime:
        return _aware(value, field=info.field_name)

    @model_validator(mode="after")
    def _complete_identity(self) -> VenueSelectorCandidate:
        if self.freshness_deadline <= self.observed_at:
            raise ValueError("market freshness_deadline must follow observed_at")
        policy = venue_policy(self.lane)
        expected_venue_id = VenueIdentity(venue_code=policy.venue).venue_id
        if self.instrument.venue_id != expected_venue_id:
            raise ValueError("instrument venue identity does not match lane")
        expected_product_lane_id = VenueProductLaneIdentity(
            venue_id=expected_venue_id,
            product_lane=self.lane.value,
            settlement_asset_id=self.instrument.settle_asset_id,
        ).product_lane_id
        if self.instrument.product_lane_id != expected_product_lane_id:
            raise ValueError("instrument product lane identity does not match lane")
        if self.account_scope.venue_id != expected_venue_id:
            raise ValueError("account venue identity does not match lane")
        payload = self.model_dump(mode="json", exclude={"candidate_id"})
        expected = _identifier("venue_selector_candidate", payload)
        if self.candidate_id and self.candidate_id != expected:
            raise ValueError("candidate_id does not match canonical payload")
        object.__setattr__(self, "candidate_id", expected)
        return self


class VenueSelectorInput(_ContractModel):
    selection_id: str = Field(min_length=1)
    evidence_lane: EvidenceLane
    as_of: datetime
    candidates: tuple[VenueSelectorCandidate, ...] = Field(min_length=1)

    @field_validator("as_of")
    @classmethod
    def _selector_as_of_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="as_of")


class VenueSelectorReceipt(_ContractModel):
    selection_id: str
    evidence_lane: EvidenceLane
    decision: Literal["SELECT", "ABSTAIN"]
    selected_candidate_id: str | None = None
    selected_instrument_id: str | None = None
    ranked_candidate_ids: tuple[str, ...]
    stable_tie_break: Literal["all_in_cost_bps,venue_id,lane,instrument_id,account_scope_id"] = (
        "all_in_cost_bps,venue_id,lane,instrument_id,account_scope_id"
    )
    blockers: tuple[str, ...]
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    selector_receipt_id: str

    @model_validator(mode="after")
    def _selector_decision_consistency(self) -> VenueSelectorReceipt:
        if self.decision == "SELECT":
            if not self.selected_candidate_id or not self.selected_instrument_id or self.blockers:
                raise ValueError("selected receipt is incomplete or blocked")
        elif self.selected_candidate_id is not None or self.selected_instrument_id is not None:
            raise ValueError("abstention cannot select a candidate")
        return self


def _selector_route_key(candidate: VenueSelectorCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.instrument.venue_id,
        candidate.lane.value,
        candidate.instrument.instrument_id,
        candidate.account_scope.account_scope_id,
    )


def select_venue(request: VenueSelectorInput) -> VenueSelectorReceipt:
    """Select deterministically from complete fresh evidence, otherwise abstain."""

    blockers: list[str] = []
    route_keys: set[tuple[str, str, str, str]] = set()
    for candidate in request.candidates:
        route_key = _selector_route_key(candidate)
        if route_key in route_keys:
            blockers.append(f"ambiguous_duplicate_route:{candidate.instrument.instrument_id}")
        route_keys.add(route_key)
        if candidate.account_scope.environment != request.evidence_lane.value and not (
            request.evidence_lane in {EvidenceLane.BACKTEST, EvidenceLane.SHADOW}
            and candidate.account_scope.environment == "research"
        ):
            blockers.append(f"account_environment_mismatch:{candidate.candidate_id}")
        policy = venue_policy(candidate.lane)
        if request.evidence_lane == EvidenceLane.TESTNET and not policy.testnet_progression_allowed:
            blockers.append(f"lane_not_testnet_eligible:{candidate.lane.value}")
        if request.evidence_lane in {EvidenceLane.CANARY, EvidenceLane.LIVE}:
            blockers.append(f"phase00_effect_lane_forbidden:{request.evidence_lane.value}")
        if candidate.observed_at > request.as_of:
            blockers.append(f"future_market_evidence:{candidate.candidate_id}")
        if request.as_of > candidate.freshness_deadline:
            blockers.append(f"stale_market_evidence:{candidate.candidate_id}")
        if candidate.costs.observed_at > request.as_of:
            blockers.append(f"future_cost_evidence:{candidate.candidate_id}")
        if request.as_of > candidate.costs.freshness_deadline:
            blockers.append(f"stale_cost_evidence:{candidate.candidate_id}")

    ranked = tuple(
        sorted(
            request.candidates,
            key=lambda candidate: (
                candidate.costs.all_in_cost_bps,
                *_selector_route_key(candidate),
            ),
        )
    )
    blockers_tuple = _unique(tuple(blockers))
    selected = ranked[0] if ranked and not blockers_tuple else None
    decision = "SELECT" if selected is not None else "ABSTAIN"
    input_hash = _payload_hash("venue_selector_input", request)
    payload = {
        "selection_id": request.selection_id,
        "decision": decision,
        "selected_candidate_id": selected.candidate_id if selected else None,
        "ranked_candidate_ids": tuple(item.candidate_id for item in ranked),
        "blockers": blockers_tuple,
        "input_hash": input_hash,
    }
    return VenueSelectorReceipt(
        selection_id=request.selection_id,
        evidence_lane=request.evidence_lane,
        decision=decision,
        selected_candidate_id=selected.candidate_id if selected else None,
        selected_instrument_id=selected.instrument.instrument_id if selected else None,
        ranked_candidate_ids=tuple(item.candidate_id for item in ranked),
        blockers=blockers_tuple,
        input_hash=input_hash,
        selector_receipt_id=_identifier("venue_selector", payload),
    )


# ---------------------------------------------------------------------------
# Two-leg journal and recovery authority references (no execution)


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class TwoLegState(StrEnum):
    PROPOSED = "proposed"
    AUTHORITY_CONFIRMED = "authority_confirmed"
    FIRST_LEG_PENDING = "first_leg_pending"
    FIRST_LEG_CONFIRMED = "first_leg_confirmed"
    SECOND_LEG_PENDING = "second_leg_pending"
    HEDGED = "hedged"
    RECOVERY_REQUIRED = "recovery_required"
    RECOVERY_AUTHORIZED = "recovery_authorized"
    RECOVERY_PENDING = "recovery_pending"
    COMPLETED = "completed"
    RECOVERED = "recovered"
    ABORTED = "aborted"
    FAILED = "failed"
    UNKNOWN = "unknown"


TERMINAL_TWO_LEG_STATES = frozenset(
    {
        TwoLegState.COMPLETED,
        TwoLegState.RECOVERED,
        TwoLegState.ABORTED,
        TwoLegState.FAILED,
        TwoLegState.UNKNOWN,
    }
)
LEGAL_TWO_LEG_TRANSITIONS: dict[TwoLegState, frozenset[TwoLegState]] = {
    TwoLegState.PROPOSED: frozenset({TwoLegState.AUTHORITY_CONFIRMED, TwoLegState.ABORTED}),
    TwoLegState.AUTHORITY_CONFIRMED: frozenset(
        {TwoLegState.FIRST_LEG_PENDING, TwoLegState.ABORTED}
    ),
    TwoLegState.FIRST_LEG_PENDING: frozenset(
        {
            TwoLegState.FIRST_LEG_CONFIRMED,
            TwoLegState.RECOVERY_REQUIRED,
            TwoLegState.UNKNOWN,
        }
    ),
    TwoLegState.FIRST_LEG_CONFIRMED: frozenset(
        {TwoLegState.SECOND_LEG_PENDING, TwoLegState.RECOVERY_REQUIRED}
    ),
    TwoLegState.SECOND_LEG_PENDING: frozenset(
        {TwoLegState.HEDGED, TwoLegState.RECOVERY_REQUIRED, TwoLegState.UNKNOWN}
    ),
    TwoLegState.HEDGED: frozenset({TwoLegState.COMPLETED, TwoLegState.RECOVERY_REQUIRED}),
    TwoLegState.RECOVERY_REQUIRED: frozenset(
        {TwoLegState.RECOVERY_AUTHORIZED, TwoLegState.UNKNOWN}
    ),
    TwoLegState.RECOVERY_AUTHORIZED: frozenset({TwoLegState.RECOVERY_PENDING, TwoLegState.UNKNOWN}),
    TwoLegState.RECOVERY_PENDING: frozenset(
        {TwoLegState.RECOVERED, TwoLegState.FAILED, TwoLegState.UNKNOWN}
    ),
    TwoLegState.COMPLETED: frozenset(),
    TwoLegState.RECOVERED: frozenset(),
    TwoLegState.ABORTED: frozenset(),
    TwoLegState.FAILED: frozenset(),
    TwoLegState.UNKNOWN: frozenset(),
}
RECOVERY_AUTHORITY_STATES = frozenset(
    {TwoLegState.RECOVERY_AUTHORIZED, TwoLegState.RECOVERY_PENDING, TwoLegState.RECOVERED}
)


class ExecutionLegContract(_ContractModel):
    leg_name: Literal["leg_a", "leg_b"]
    instrument: VenueInstrumentIdentity
    account_scope: AccountScopeIdentity
    side: OrderSide
    quantity: Decimal
    client_order_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)

    @field_validator("quantity", mode="before")
    @classmethod
    def _positive_quantity(cls, value: object) -> Decimal:
        quantity = _decimal(value, field="quantity")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        return quantity

    @model_validator(mode="after")
    def _same_venue_identity(self) -> ExecutionLegContract:
        if self.instrument.venue_id != self.account_scope.venue_id:
            raise ValueError("leg instrument and account venue identities must match")
        return self


class TwoLegExecutionPlan(_ContractModel):
    proposal_id: str = Field(min_length=1)
    evidence_lane: EvidenceLane
    created_at: datetime
    leg_a: ExecutionLegContract
    leg_b: ExecutionLegContract
    plan_id: str = ""

    @field_validator("created_at")
    @classmethod
    def _created_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="created_at")

    @model_validator(mode="after")
    def _pair_and_bind(self) -> TwoLegExecutionPlan:
        if self.leg_a.leg_name != "leg_a" or self.leg_b.leg_name != "leg_b":
            raise ValueError("two-leg plan leg names are misplaced")
        if self.leg_a.instrument.instrument_id == self.leg_b.instrument.instrument_id:
            raise ValueError("two-leg plan instruments must differ")
        if self.leg_a.side == self.leg_b.side:
            raise ValueError("two-leg plan requires opposing sides")
        if self.leg_a.idempotency_key == self.leg_b.idempotency_key:
            raise ValueError("leg idempotency keys must differ")
        expected_environment = (
            "research"
            if self.evidence_lane in {EvidenceLane.BACKTEST, EvidenceLane.SHADOW}
            else "testnet"
            if self.evidence_lane == EvidenceLane.TESTNET
            else "live"
        )
        observed_environments = {
            self.leg_a.account_scope.environment,
            self.leg_b.account_scope.environment,
        }
        if observed_environments != {expected_environment}:
            raise ValueError("two-leg account environments do not match evidence lane")
        payload = self.model_dump(mode="json", exclude={"plan_id"})
        expected = _identifier("two_leg_plan", payload)
        if self.plan_id and self.plan_id != expected:
            raise ValueError("plan_id does not match canonical payload")
        object.__setattr__(self, "plan_id", expected)
        return self


class RecoveryAuthorityReference(_ContractModel):
    authority_receipt_id: str = Field(min_length=1)
    authority_receipt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    permit_id: str = Field(min_length=1)
    plan_id: str = Field(min_length=1)
    allowed_action: Literal["cancel_open_leg", "flatten_filled_leg", "complete_missing_leg"]
    scope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: datetime
    expires_at: datetime

    @field_validator("issued_at", "expires_at")
    @classmethod
    def _authority_times_are_aware(cls, value: datetime, info) -> datetime:
        return _aware(value, field=info.field_name)

    @model_validator(mode="after")
    def _authority_window(self) -> RecoveryAuthorityReference:
        if self.expires_at <= self.issued_at:
            raise ValueError("recovery authority expiry must follow issuance")
        return self


class TwoLegJournalEvent(_ContractModel):
    sequence: int = Field(gt=0)
    event_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    occurred_at: datetime
    from_state: TwoLegState
    to_state: TwoLegState
    recovery_authority: RecoveryAuthorityReference | None = None

    @field_validator("occurred_at")
    @classmethod
    def _occurred_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="occurred_at")


class TwoLegExecutionJournal(_ContractModel):
    plan: TwoLegExecutionPlan
    initial_state: Literal[TwoLegState.PROPOSED] = TwoLegState.PROPOSED
    events: tuple[TwoLegJournalEvent, ...]


class TwoLegJournalValidationReceipt(_ContractModel):
    plan_id: str
    valid: bool
    terminal: bool
    final_state: TwoLegState
    event_count: int = Field(ge=0)
    blockers: tuple[str, ...]
    journal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_receipt_id: str

    @model_validator(mode="after")
    def _validation_consistency(self) -> TwoLegJournalValidationReceipt:
        if self.valid == bool(self.blockers):
            raise ValueError("journal validation decision is inconsistent")
        if self.terminal != (self.final_state in TERMINAL_TWO_LEG_STATES):
            raise ValueError("journal terminal flag is inconsistent")
        return self


def validate_two_leg_journal(
    journal: TwoLegExecutionJournal,
    *,
    as_of: datetime,
    trusted_recovery_authority_receipt_ids: frozenset[str] = frozenset(),
    trusted_recovery_authority_receipt_hashes: frozenset[str] = frozenset(),
) -> TwoLegJournalValidationReceipt:
    """Validate journal structure and trusted recovery references without effects."""

    _aware(as_of, field="as_of")
    blockers: list[str] = []
    current = TwoLegState.PROPOSED
    previous_time = journal.plan.created_at
    event_ids: set[str] = set()
    idempotency_keys: set[str] = set()
    for expected_sequence, event in enumerate(journal.events, start=1):
        if event.sequence != expected_sequence:
            blockers.append(f"noncontiguous_sequence:{event.sequence}")
        if event.event_id in event_ids:
            blockers.append(f"duplicate_event_id:{event.event_id}")
        event_ids.add(event.event_id)
        if event.idempotency_key in idempotency_keys:
            blockers.append(f"duplicate_idempotency_key:{event.idempotency_key}")
        idempotency_keys.add(event.idempotency_key)
        if event.occurred_at < previous_time:
            blockers.append(f"nonmonotonic_event_time:{event.event_id}")
        if event.occurred_at > as_of:
            blockers.append(f"future_event:{event.event_id}")
        previous_time = event.occurred_at
        if current in TERMINAL_TWO_LEG_STATES:
            blockers.append(f"transition_after_terminal:{current.value}")
        if event.from_state != current:
            blockers.append(f"state_chain_mismatch:{event.event_id}")
        if event.to_state not in LEGAL_TWO_LEG_TRANSITIONS[current]:
            blockers.append(f"illegal_two_leg_transition:{current.value}:{event.to_state.value}")
        if event.to_state in RECOVERY_AUTHORITY_STATES:
            authority = event.recovery_authority
            if authority is None:
                blockers.append(f"recovery_authority_missing:{event.event_id}")
            else:
                if authority.authority_receipt_id not in trusted_recovery_authority_receipt_ids:
                    blockers.append(f"recovery_authority_untrusted:{event.event_id}")
                if (
                    authority.authority_receipt_hash
                    not in trusted_recovery_authority_receipt_hashes
                ):
                    blockers.append(f"recovery_authority_hash_untrusted:{event.event_id}")
                if authority.plan_id != journal.plan.plan_id:
                    blockers.append(f"recovery_authority_plan_mismatch:{event.event_id}")
                if as_of < authority.issued_at:
                    blockers.append(f"recovery_authority_not_yet_valid:{event.event_id}")
                if as_of > authority.expires_at:
                    blockers.append(f"recovery_authority_expired:{event.event_id}")
        elif event.recovery_authority is not None:
            blockers.append(f"recovery_authority_out_of_scope:{event.event_id}")
        current = event.to_state

    blockers_tuple = _unique(tuple(blockers))
    journal_hash = _payload_hash("two_leg_journal", journal)
    payload = {
        "plan_id": journal.plan.plan_id,
        "journal_hash": journal_hash,
        "final_state": current,
        "blockers": blockers_tuple,
    }
    return TwoLegJournalValidationReceipt(
        plan_id=journal.plan.plan_id,
        valid=not blockers_tuple,
        terminal=current in TERMINAL_TWO_LEG_STATES,
        final_state=current,
        event_count=len(journal.events),
        blockers=blockers_tuple,
        journal_hash=journal_hash,
        validation_receipt_id=_identifier("two_leg_journal_validation", payload),
    )


# ---------------------------------------------------------------------------
# Proposal-only learning and specialist capabilities


class ProposalCapabilityRole(StrEnum):
    TEACHER = "teacher"
    STUDENT = "student"
    ML = "ml"
    RL = "rl"
    SHADOW = "shadow"
    SPECIALIST = "specialist"


class ProposalOutputType(StrEnum):
    PROPOSAL = "proposal"
    SCORE = "score"
    EXPLANATION = "explanation"
    HYPOTHESIS = "hypothesis"
    EVALUATION = "evaluation"


class ProposalCapabilityContract(_ContractModel):
    role: ProposalCapabilityRole
    capability_name: str = Field(min_length=1)
    specialist_domain: str | None = None
    allowed_outputs: tuple[ProposalOutputType, ...] = Field(min_length=1)
    execution_authority: Literal[False] = False
    can_mint_permits: Literal[False] = False
    can_submit_orders: Literal[False] = False
    can_publish_live_evidence: Literal[False] = False
    capability_id: str = ""

    @model_validator(mode="after")
    def _proposal_only(self) -> ProposalCapabilityContract:
        if len(self.allowed_outputs) != len(set(self.allowed_outputs)):
            raise ValueError("allowed_outputs must be unique")
        if self.role == ProposalCapabilityRole.SPECIALIST and not (
            self.specialist_domain and self.specialist_domain.strip()
        ):
            raise ValueError("specialist capability requires specialist_domain")
        if self.role != ProposalCapabilityRole.SPECIALIST and self.specialist_domain is not None:
            raise ValueError("only specialist capability may declare specialist_domain")
        payload = self.model_dump(mode="json", exclude={"capability_id"})
        expected = _identifier("proposal_capability", payload)
        if self.capability_id and self.capability_id != expected:
            raise ValueError("capability_id does not match canonical payload")
        object.__setattr__(self, "capability_id", expected)
        return self


class ProposalCapabilityRegistry(_ContractModel):
    contracts: tuple[ProposalCapabilityContract, ...] = Field(min_length=6)

    @model_validator(mode="after")
    def _all_roles_present(self) -> ProposalCapabilityRegistry:
        required = set(ProposalCapabilityRole)
        observed = {contract.role for contract in self.contracts}
        if observed != required:
            missing = sorted(role.value for role in required - observed)
            raise ValueError(f"capability registry missing roles: {','.join(missing)}")
        ids = [contract.capability_id for contract in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("capability registry contains duplicate contracts")
        return self


class ProposalOnlyReceipt(_ContractModel):
    capability: ProposalCapabilityContract
    capability_id: str = ""
    output_type: ProposalOutputType
    evidence_lane: EvidenceLane
    input_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    proposal_payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    produced_at: datetime
    proposal_only: Literal[True] = True
    execution_authority: Literal[False] = False
    can_mint_permits: Literal[False] = False
    permit_id: Literal[""] = ""
    authority_receipt_id: Literal[""] = ""
    proposal_receipt_id: str = ""

    @field_validator("produced_at")
    @classmethod
    def _produced_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="produced_at")

    @model_validator(mode="after")
    def _bind_proposal_receipt(self) -> ProposalOnlyReceipt:
        if self.output_type not in self.capability.allowed_outputs:
            raise ValueError("proposal output type is not allowed by capability")
        if self.capability_id and self.capability_id != self.capability.capability_id:
            raise ValueError("capability_id does not match embedded capability")
        object.__setattr__(self, "capability_id", self.capability.capability_id)
        payload = self.model_dump(mode="json", exclude={"proposal_receipt_id"})
        expected = _identifier("proposal_only_receipt", payload)
        if self.proposal_receipt_id and self.proposal_receipt_id != expected:
            raise ValueError("proposal_receipt_id does not match canonical payload")
        object.__setattr__(self, "proposal_receipt_id", expected)
        return self


# ---------------------------------------------------------------------------
# Non-mixable outcome labels


class OutcomeLabel(_ContractModel):
    outcome_id: str = Field(min_length=1)
    evidence_lane: EvidenceLane
    source_run_id: str = Field(min_length=1)
    observed_at: datetime
    outcome_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("observed_at")
    @classmethod
    def _outcome_time_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="observed_at")


class OutcomeLabelPartition(_ContractModel):
    evidence_lane: EvidenceLane
    labels: tuple[OutcomeLabel, ...] = Field(min_length=1)
    partition_id: str = ""

    @model_validator(mode="after")
    def _one_lane_only(self) -> OutcomeLabelPartition:
        observed_lanes = {label.evidence_lane for label in self.labels}
        if observed_lanes != {self.evidence_lane}:
            raise ValueError("outcome label partition cannot mix evidence lanes")
        outcome_ids = [label.outcome_id for label in self.labels]
        if len(outcome_ids) != len(set(outcome_ids)):
            raise ValueError("outcome label partition contains duplicate outcomes")
        payload = self.model_dump(mode="json", exclude={"partition_id"})
        expected = _identifier("outcome_label_partition", payload)
        if self.partition_id and self.partition_id != expected:
            raise ValueError("partition_id does not match canonical payload")
        object.__setattr__(self, "partition_id", expected)
        return self


# ---------------------------------------------------------------------------
# Complete held-out validation receipts


class HeldOutDimension(StrEnum):
    VENUE = "venue"
    PAIR = "pair"
    TIME = "time"
    REGIME = "regime"


class HeldOutMetrics(_ContractModel):
    trade_count: int = Field(gt=0)
    profit_factor: Decimal
    sharpe: Decimal
    max_drawdown: Decimal
    expectancy: Decimal

    @field_validator("profit_factor", "sharpe", "max_drawdown", "expectancy", mode="before")
    @classmethod
    def _finite_metrics(cls, value: object, info) -> Decimal:
        return _decimal(value, field=info.field_name)

    @model_validator(mode="after")
    def _metric_ranges(self) -> HeldOutMetrics:
        if self.profit_factor < 0:
            raise ValueError("profit_factor cannot be negative")
        if self.max_drawdown < 0:
            raise ValueError("max_drawdown cannot be negative")
        return self


class HeldOutDimensionReceipt(_ContractModel):
    dimension: HeldOutDimension
    source_dataset_id: str = Field(min_length=1)
    split_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_members: tuple[str, ...] = Field(min_length=1)
    held_out_members: tuple[str, ...] = Field(min_length=1)
    training_sample_count: int = Field(gt=0)
    held_out_sample_count: int = Field(gt=0)
    training_end: datetime | None = None
    held_out_start: datetime | None = None
    evaluated_at: datetime
    metrics: HeldOutMetrics
    dimension_receipt_id: str = ""

    @field_validator("evaluated_at")
    @classmethod
    def _evaluated_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="evaluated_at")

    @field_validator("training_members", "held_out_members")
    @classmethod
    def _normalize_members(cls, value: tuple[str, ...], info) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError(f"{info.field_name} cannot contain blank members")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{info.field_name} must be unique")
        return normalized

    @field_validator("training_end", "held_out_start")
    @classmethod
    def _optional_times_are_aware(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _aware(value, field=info.field_name)

    @model_validator(mode="after")
    def _held_out_contract(self) -> HeldOutDimensionReceipt:
        if set(self.training_members) & set(self.held_out_members):
            raise ValueError("training and held-out members must be disjoint")
        if self.dimension == HeldOutDimension.TIME:
            if self.training_end is None or self.held_out_start is None:
                raise ValueError("time-held-out receipt requires temporal boundaries")
            if self.training_end >= self.held_out_start:
                raise ValueError("time-held-out boundary must follow training window")
        elif self.training_end is not None or self.held_out_start is not None:
            raise ValueError("temporal boundaries are exclusive to time-held-out evidence")
        payload = self.model_dump(mode="json", exclude={"dimension_receipt_id"})
        expected = _identifier("held_out_dimension", payload)
        if self.dimension_receipt_id and self.dimension_receipt_id != expected:
            raise ValueError("dimension_receipt_id does not match canonical payload")
        object.__setattr__(self, "dimension_receipt_id", expected)
        return self


class CompleteHeldOutValidationReceipt(_ContractModel):
    validation_run_id: str = Field(min_length=1)
    model_or_strategy_id: str = Field(min_length=1)
    source_dataset_id: str = Field(min_length=1)
    source_label_lane: EvidenceLane
    dimensions: tuple[HeldOutDimensionReceipt, ...] = Field(min_length=4, max_length=4)
    created_at: datetime
    validation_receipt_id: str = ""

    @field_validator("created_at")
    @classmethod
    def _created_at_is_aware(cls, value: datetime) -> datetime:
        return _aware(value, field="created_at")

    @model_validator(mode="after")
    def _all_dimensions_are_present(self) -> CompleteHeldOutValidationReceipt:
        required = set(HeldOutDimension)
        observed = {receipt.dimension for receipt in self.dimensions}
        if observed != required or len(observed) != len(self.dimensions):
            missing = sorted(item.value for item in required - observed)
            raise ValueError(f"held-out validation dimensions incomplete: {','.join(missing)}")
        if any(item.source_dataset_id != self.source_dataset_id for item in self.dimensions):
            raise ValueError("held-out dimension source_dataset_id mismatch")
        payload = self.model_dump(mode="json", exclude={"validation_receipt_id"})
        expected = _identifier("complete_held_out_validation", payload)
        if self.validation_receipt_id and self.validation_receipt_id != expected:
            raise ValueError("validation_receipt_id does not match canonical payload")
        object.__setattr__(self, "validation_receipt_id", expected)
        return self


IDENTITY_BINDINGS = {
    "identity_ontology_version": IDENTITY_ONTOLOGY_VERSION,
    "venue_policy_schema_version": VENUE_POLICY_SCHEMA_VERSION,
}
