from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from quant_platform.orchestration.venue_policy_registry import (
    HYPERLIQUID_TESTNET_ADAPTER,
    PHASE00_VENUE_POLICIES,
    VenueCapabilityEvidence,
    VenueLane,
    evaluate_capability,
    evaluate_testnet_route,
)

HASH = "b" * 64
NOW = datetime(2026, 8, 21, tzinfo=UTC)


def test_registry_contains_all_twelve_disabled_lanes() -> None:
    assert len(PHASE00_VENUE_POLICIES) == 12
    assert {policy.lane for policy in PHASE00_VENUE_POLICIES} == set(VenueLane)
    assert all(policy.activation_enabled is False for policy in PHASE00_VENUE_POLICIES)
    assert all(policy.live_enabled is False for policy in PHASE00_VENUE_POLICIES)
    assert [
        policy.lane
        for policy in PHASE00_VENUE_POLICIES
        if policy.testnet_progression_allowed
    ] == [VenueLane.HYPERLIQUID_PERP]


def test_only_exact_hyperliquid_testnet_route_is_eligible() -> None:
    allowed = evaluate_testnet_route(
        lane=VenueLane.HYPERLIQUID_PERP,
        environment="testnet",
        adapter=HYPERLIQUID_TESTNET_ADAPTER,
        account_scope="hyperliquid-testnet-agent",
        source_snapshot_hash=HASH,
        requested_at=NOW,
        freshness_deadline=NOW + timedelta(minutes=5),
    )
    denied = evaluate_testnet_route(
        lane=VenueLane.DYDX_PERP,
        environment="testnet",
        adapter="quant_platform.dydx_sdk_order_adapter:DydxSdkOrderAdapter",
        account_scope="dydx-testnet",
        source_snapshot_hash=HASH,
        requested_at=NOW,
        freshness_deadline=NOW + timedelta(minutes=5),
    )

    assert allowed.allowed is True
    assert denied.allowed is False
    assert "testnet_lane_denied" in denied.blockers


def test_unknown_or_stale_route_fails_closed() -> None:
    unknown = evaluate_testnet_route(
        lane="injective_perp",
        environment="testnet",
        adapter="adapter",
        account_scope="account",
        source_snapshot_hash=HASH,
        requested_at=NOW,
        freshness_deadline=NOW + timedelta(minutes=1),
    )
    stale = evaluate_testnet_route(
        lane=VenueLane.HYPERLIQUID_PERP,
        environment="testnet",
        adapter=HYPERLIQUID_TESTNET_ADAPTER,
        account_scope="account",
        source_snapshot_hash=HASH,
        requested_at=NOW,
        freshness_deadline=NOW - timedelta(seconds=1),
    )

    assert unknown.blockers == ("unknown_venue_lane",)
    assert "testnet_route_stale" in stale.blockers


def test_spot_capability_cannot_claim_short_execution() -> None:
    with pytest.raises(ValueError, match="spot capability"):
        VenueCapabilityEvidence(
            lane=VenueLane.BINANCE_US_SPOT,
            venue="binance_us",
            account_scope="account",
            source_snapshot_hash=HASH,
            captured_at=NOW,
            freshness_deadline=NOW + timedelta(minutes=5),
            instrument_id="instrument_btc_usd",
            short_executable=True,
        )


def test_capability_lookup_requires_exact_fresh_evidence_and_stays_inactive() -> None:
    evidence = VenueCapabilityEvidence(
        lane=VenueLane.HYPERLIQUID_PERP,
        venue="hyperliquid",
        account_scope="account",
        source_snapshot_hash=HASH,
        captured_at=NOW,
        freshness_deadline=NOW + timedelta(minutes=5),
        instrument_id="instrument_btc",
        long_executable=True,
        short_executable=True,
        perpetual_available=True,
    )
    decision = evaluate_capability(
        lane=VenueLane.HYPERLIQUID_PERP,
        account_scope="account",
        source_snapshot_hash=HASH,
        now=NOW,
        evidence=evidence,
    )

    assert decision.allowed is False
    assert decision.blockers == ("lane_activation_disabled",)
