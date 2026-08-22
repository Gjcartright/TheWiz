from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_platform.orchestration.identity_ontology import (
    CanonicalAssetIdentity,
    IdentityMappingReceipt,
    OrderedStrategyPairIdentity,
    Phase00CandidateIdentity,
    UnorderedEconomicPairIdentity,
)

HASH = "a" * 64


def _candidate(*, asset_x: str = "asset_btc", asset_y: str = "asset_eth") -> Phase00CandidateIdentity:
    return Phase00CandidateIdentity(
        source_system="crypto_wizards",
        wizard_pair_id_or_source_candidate_id="pair-7",
        ordered_asset_x=asset_x,
        ordered_asset_y=asset_y,
        scanner_venue="dydx",
        scanner_instrument_x="BTC-USD",
        scanner_instrument_y="ETH-USD",
        target_venue="hyperliquid",
        target_product_lane="hyperliquid_perp",
        target_instrument_x="BTC",
        target_instrument_y="ETH",
        account_scope="account_scope_testnet",
        strategy_mode="ou_zscorer",
        formula_version="ou.v2",
        timeframe="1d",
        lookback=120,
        scanner_cutoff=datetime(2026, 8, 21, tzinfo=UTC),
        source_snapshot_hash=HASH,
    )


def test_ordered_orientation_changes_candidate_but_not_economic_pair() -> None:
    forward = _candidate()
    reverse = _candidate(asset_x="asset_eth", asset_y="asset_btc")

    assert forward.candidate_id != reverse.candidate_id
    assert forward.ordered_pair_id != reverse.ordered_pair_id
    assert forward.economic_pair_id == reverse.economic_pair_id


def test_unordered_pair_is_deterministic() -> None:
    first = UnorderedEconomicPairIdentity(asset_a_id="asset_btc", asset_b_id="asset_eth")
    second = UnorderedEconomicPairIdentity(asset_a_id="asset_eth", asset_b_id="asset_btc")

    assert first == second


def test_scanner_or_target_venue_mutation_changes_candidate_identity() -> None:
    baseline = _candidate()
    changed = baseline.model_copy(update={"target_venue": "dydx", "candidate_id": ""})
    rebuilt = Phase00CandidateIdentity.model_validate(changed.model_dump())

    assert baseline.candidate_id != rebuilt.candidate_id


def test_identity_rejects_naive_cutoff_and_bad_hash() -> None:
    payload = _candidate().model_dump()
    payload["scanner_cutoff"] = datetime.fromisoformat("2026-08-21")
    payload["candidate_id"] = ""
    with pytest.raises(ValueError, match="timezone-aware"):
        Phase00CandidateIdentity.model_validate(payload)
    payload["scanner_cutoff"] = datetime(2026, 8, 21, tzinfo=UTC)
    payload["source_snapshot_hash"] = "bad"
    with pytest.raises(ValueError, match="sha256"):
        Phase00CandidateIdentity.model_validate(payload)


def test_canonical_asset_and_mapping_receipt_are_source_bound() -> None:
    asset = CanonicalAssetIdentity(
        source_system="hyperliquid",
        source_asset_id="@1",
        canonical_symbol="btc",
    )
    mapping = IdentityMappingReceipt(
        source_system="hyperliquid",
        source_id="@1",
        canonical_type="asset",
        canonical_id=asset.asset_id,
        source_snapshot_hash=HASH,
        mapped_at=datetime(2026, 8, 21, tzinfo=UTC),
    )

    assert asset.canonical_symbol == "BTC"
    assert mapping.mapping_receipt_id.startswith("identity_mapping_")


def test_ordered_pair_rejects_wrong_economic_link() -> None:
    with pytest.raises(ValueError, match="economic_pair_id"):
        OrderedStrategyPairIdentity(
            ordered_asset_x_id="asset_btc",
            ordered_asset_y_id="asset_eth",
            economic_pair_id="economic_pair_wrong",
        )
