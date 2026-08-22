"""Fail-closed Phase 00 policy for every supported venue product lane."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.orchestration.identity_ontology import IDENTITY_ONTOLOGY_VERSION

VENUE_POLICY_SCHEMA_VERSION = "thewiz.venue_policy.v2"
HYPERLIQUID_TESTNET_ADAPTER = (
    "quant_platform.hyperliquid_testnet:HyperliquidTestnetOrderAdapter"
)


class VenueLane(StrEnum):
    HYPERLIQUID_PERP = "hyperliquid_perp"
    DYDX_PERP = "dydx_perp"
    BINANCE_US_SPOT = "binance_us_spot"
    BINANCE_SPOT = "binance_spot"
    BINANCE_MARGIN = "binance_margin"
    BINANCE_USDM_PERP = "binance_usdm_perp"
    BINANCE_COINM_PERP = "binance_coinm_perp"
    BYBIT_SPOT = "bybit_spot"
    BYBIT_LINEAR_PERP = "bybit_linear_perp"
    BYBIT_INVERSE_PERP = "bybit_inverse_perp"
    COINBASE_ADVANCED_SPOT = "coinbase_advanced_spot"
    COINBASE_ADVANCED_PERP = "coinbase_advanced_perp"


class VenueLanePolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[VENUE_POLICY_SCHEMA_VERSION] = VENUE_POLICY_SCHEMA_VERSION
    identity_ontology_version: Literal[IDENTITY_ONTOLOGY_VERSION] = IDENTITY_ONTOLOGY_VERSION
    lane: VenueLane
    venue: str
    product_type: Literal["spot", "margin", "linear_perp", "inverse_perp"]
    activation_enabled: Literal[False] = False
    live_enabled: Literal[False] = False
    authenticated_access_allowed: Literal[False] = False
    order_submission_allowed: Literal[False] = False
    account_mutation_allowed: Literal[False] = False
    public_market_data_allowed: bool = True
    testnet_progression_allowed: bool = False
    allowed_testnet_adapters: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _testnet_policy(self) -> VenueLanePolicy:
        allowed = self.lane == VenueLane.HYPERLIQUID_PERP
        if self.testnet_progression_allowed != allowed:
            raise ValueError("only hyperliquid_perp may declare testnet progression")
        if allowed and self.allowed_testnet_adapters != (HYPERLIQUID_TESTNET_ADAPTER,):
            raise ValueError("hyperliquid testnet adapter allowlist mismatch")
        if not allowed and self.allowed_testnet_adapters:
            raise ValueError("non-Hyperliquid lanes cannot allow testnet adapters")
        return self


_LANE_DEFINITIONS: tuple[tuple[VenueLane, str, str], ...] = (
    (VenueLane.HYPERLIQUID_PERP, "hyperliquid", "linear_perp"),
    (VenueLane.DYDX_PERP, "dydx", "linear_perp"),
    (VenueLane.BINANCE_US_SPOT, "binance_us", "spot"),
    (VenueLane.BINANCE_SPOT, "binance", "spot"),
    (VenueLane.BINANCE_MARGIN, "binance", "margin"),
    (VenueLane.BINANCE_USDM_PERP, "binance", "linear_perp"),
    (VenueLane.BINANCE_COINM_PERP, "binance", "inverse_perp"),
    (VenueLane.BYBIT_SPOT, "bybit", "spot"),
    (VenueLane.BYBIT_LINEAR_PERP, "bybit", "linear_perp"),
    (VenueLane.BYBIT_INVERSE_PERP, "bybit", "inverse_perp"),
    (VenueLane.COINBASE_ADVANCED_SPOT, "coinbase_advanced", "spot"),
    (VenueLane.COINBASE_ADVANCED_PERP, "coinbase_advanced", "linear_perp"),
)


PHASE00_VENUE_POLICIES: tuple[VenueLanePolicy, ...] = tuple(
    VenueLanePolicy(
        lane=lane,
        venue=venue,
        product_type=product_type,
        testnet_progression_allowed=lane == VenueLane.HYPERLIQUID_PERP,
        allowed_testnet_adapters=(HYPERLIQUID_TESTNET_ADAPTER,)
        if lane == VenueLane.HYPERLIQUID_PERP
        else (),
    )
    for lane, venue, product_type in _LANE_DEFINITIONS
)


class VenueCapabilityEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[VENUE_POLICY_SCHEMA_VERSION] = VENUE_POLICY_SCHEMA_VERSION
    lane: VenueLane
    venue: str
    account_scope: str = Field(min_length=1)
    source_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    freshness_deadline: datetime
    instrument_id: str = Field(min_length=1)
    long_executable: bool = False
    short_executable: bool = False
    margin_available: bool = False
    perpetual_available: bool = False
    borrow_available: bool = False
    inventory_available: bool = False
    locate_available: bool = False
    recall_supported: bool = False
    jurisdiction_blockers: tuple[str, ...] = ()
    account_blockers: tuple[str, ...] = ()

    @field_validator("captured_at", "freshness_deadline")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("venue capability timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _semantic_rules(self) -> VenueCapabilityEvidence:
        policy = venue_policy(self.lane)
        if self.venue.lower() != policy.venue:
            raise ValueError("venue capability does not match lane policy")
        if self.freshness_deadline <= self.captured_at:
            raise ValueError("freshness deadline must follow capture")
        if policy.product_type == "spot" and self.short_executable:
            raise ValueError("spot capability cannot declare short execution")
        return self


class VenuePolicyDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[VENUE_POLICY_SCHEMA_VERSION] = VENUE_POLICY_SCHEMA_VERSION
    lane: str
    allowed: bool
    blockers: tuple[str, ...]
    account_scope: str
    source_snapshot_hash: str


def venue_policy(lane: VenueLane | str) -> VenueLanePolicy:
    try:
        normalized = VenueLane(str(lane))
    except ValueError as exc:
        raise KeyError(f"unknown venue lane: {lane}") from exc
    return next(policy for policy in PHASE00_VENUE_POLICIES if policy.lane == normalized)


def evaluate_capability(
    *,
    lane: VenueLane | str,
    account_scope: str,
    source_snapshot_hash: str,
    now: datetime,
    evidence: VenueCapabilityEvidence | None,
) -> VenuePolicyDecision:
    blockers: list[str] = []
    try:
        policy = venue_policy(lane)
    except KeyError:
        return VenuePolicyDecision(
            lane=str(lane),
            allowed=False,
            blockers=("unknown_venue_lane",),
            account_scope=account_scope,
            source_snapshot_hash=source_snapshot_hash,
        )
    if not account_scope.strip():
        blockers.append("account_scope_missing")
    if evidence is None:
        blockers.append("capability_evidence_missing")
    else:
        if evidence.lane != policy.lane:
            blockers.append("capability_lane_mismatch")
        if evidence.account_scope != account_scope:
            blockers.append("capability_account_scope_mismatch")
        if evidence.source_snapshot_hash != source_snapshot_hash:
            blockers.append("capability_source_snapshot_mismatch")
        if now.tzinfo is None or now > evidence.freshness_deadline:
            blockers.append("capability_evidence_stale")
        blockers.extend(evidence.jurisdiction_blockers)
        blockers.extend(evidence.account_blockers)
    if not policy.activation_enabled:
        blockers.append("lane_activation_disabled")
    if policy.live_enabled:
        blockers.append("phase00_live_policy_invalid")
    return VenuePolicyDecision(
        lane=policy.lane,
        allowed=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        account_scope=account_scope,
        source_snapshot_hash=source_snapshot_hash,
    )


def evaluate_testnet_route(
    *,
    lane: VenueLane | str,
    environment: str,
    adapter: str,
    account_scope: str,
    source_snapshot_hash: str,
    requested_at: datetime,
    freshness_deadline: datetime,
) -> VenuePolicyDecision:
    blockers: list[str] = []
    try:
        policy = venue_policy(lane)
    except KeyError:
        return VenuePolicyDecision(
            lane=str(lane),
            allowed=False,
            blockers=("unknown_venue_lane",),
            account_scope=account_scope,
            source_snapshot_hash=source_snapshot_hash,
        )
    if environment != "testnet":
        blockers.append("testnet_environment_required")
    if not policy.testnet_progression_allowed:
        blockers.append("testnet_lane_denied")
    if adapter not in policy.allowed_testnet_adapters:
        blockers.append("testnet_adapter_denied")
    if not account_scope.strip():
        blockers.append("account_scope_missing")
    if requested_at.tzinfo is None or freshness_deadline.tzinfo is None:
        blockers.append("testnet_freshness_timestamp_invalid")
    elif requested_at > freshness_deadline:
        blockers.append("testnet_route_stale")
    if not re_full_sha256(source_snapshot_hash):
        blockers.append("source_snapshot_hash_invalid")
    return VenuePolicyDecision(
        lane=policy.lane,
        allowed=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        account_scope=account_scope,
        source_snapshot_hash=source_snapshot_hash,
    )


def re_full_sha256(value: str) -> bool:
    import re

    return re.fullmatch(r"[0-9a-f]{64}", str(value or "")) is not None
