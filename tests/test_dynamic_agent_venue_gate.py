from __future__ import annotations

import pandas as pd

from quant_platform.orchestration.contracts import CandidateIdentity, EvidencePacket, VetoRecord
from quant_platform.orchestration.venue_gate import VenuePolicy, evaluate_venue_gate


def _candidate() -> CandidateIdentity:
    return CandidateIdentity(pair="BNB-USD/WLD-USD", venue="dydx", strategy_family="Copula", timeframe="1d", lookback=320, formula_version="copula-v1")


def test_dydx_gate_requires_both_confirmed_legs(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    pd.DataFrame([{"market": "BNB-USD", "compatible_for_paper_submit": True}, {"market": "WLD-USD", "compatible_for_paper_submit": True}]).to_csv(active / "dydx_execution_market_compatibility.csv", index=False)
    result = evaluate_venue_gate(_candidate(), policy=VenuePolicy(venue="dydx", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"), root=tmp_path)
    assert isinstance(result, EvidencePacket)


def test_spot_policy_vetoes_a_short_leg_without_exception(tmp_path):
    result = evaluate_venue_gate(_candidate(), policy=VenuePolicy(venue="binance_us", account_eligible=True, product_type="spot", supports_short_leg=False, capability_status="verified"), root=tmp_path)
    assert isinstance(result, VetoRecord)
    assert result.blocker_code == "spot_short_leg_unavailable"


def test_dydx_gate_rejects_false_string_in_persisted_compatibility(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    pd.DataFrame([
        {"market": "BNB-USD", "compatible_for_paper_submit": "true"},
        {"market": "WLD-USD", "compatible_for_paper_submit": "False "},
    ]).to_csv(active / "dydx_execution_market_compatibility.csv", index=False)
    result = evaluate_venue_gate(
        _candidate(),
        policy=VenuePolicy(venue="dydx", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"),
        root=tmp_path,
    )
    assert isinstance(result, VetoRecord)
    assert result.blocker_code == "both_legs_not_confirmed"


def test_hyperliquid_gate_rejects_false_string_in_persisted_compatibility(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    pd.DataFrame([
        {"pair": "BNB-USD-WLD-USD", "mirrorable_for_paper": "False "},
    ]).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    candidate = CandidateIdentity(pair="BNB-USD/WLD-USD", venue="hyperliquid", strategy_family="Copula", timeframe="1d", lookback=320, formula_version="copula-v1")
    result = evaluate_venue_gate(
        candidate,
        policy=VenuePolicy(venue="hyperliquid", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"),
        root=tmp_path,
    )
    assert isinstance(result, VetoRecord)
    assert result.blocker_code == "both_legs_not_confirmed"


def test_hyperliquid_gate_accepts_explicit_true_compatibility(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    pd.DataFrame([
        {"pair": "BNB-USD-WLD-USD", "mirrorable_for_paper": "true"},
    ]).to_csv(active / "hyperliquid_execution_market_compatibility.csv", index=False)
    candidate = CandidateIdentity(pair="BNB-USD/WLD-USD", venue="hyperliquid", strategy_family="Copula", timeframe="1d", lookback=320, formula_version="copula-v1")
    result = evaluate_venue_gate(
        candidate,
        policy=VenuePolicy(venue="hyperliquid", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"),
        root=tmp_path,
    )
    assert isinstance(result, EvidencePacket)
