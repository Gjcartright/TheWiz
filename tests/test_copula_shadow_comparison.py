from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from quant_platform.orchestration.contracts import CandidateIdentity
from quant_platform.orchestration.copula_shadow_comparison import build_copula_shadow_comparison
from quant_platform.orchestration.venue_gate import VenuePolicy


def test_shadow_comparison_keeps_sequential_and_dynamic_decisions_separate(tmp_path):
    active = tmp_path / "reports" / "active"; active.mkdir(parents=True)
    now = datetime.now(timezone.utc).isoformat()
    pd.DataFrame([{"pair": "BNB-USD/WLD-USD", "detail_capture_timestamp_utc": now}]).to_csv(active / "wizard_two_hour_copula_report.csv", index=False)
    pd.DataFrame([{"pair": "BNB-USD/WLD-USD", "acceptance": "ACCEPT", "acceptance_reason": "passed"}]).to_csv(active / "bnbusd_wldusd_copula_verification_after_cost.csv", index=False)
    pd.DataFrame([{"case": "base"}]).to_csv(active / "bnbusd_wldusd_copula_verification_cost_comparison.csv", index=False)
    pd.DataFrame([{"market": "BNB-USD", "compatible_for_paper_submit": True}, {"market": "WLD-USD", "compatible_for_paper_submit": True}]).to_csv(active / "dydx_execution_market_compatibility.csv", index=False)
    (tmp_path / "docs").mkdir(); (tmp_path / "docs" / "formula_dictionary.md").write_text("formula", encoding="utf-8")
    candidate = CandidateIdentity(pair="BNB-USD/WLD-USD", venue="dydx", strategy_family="Copula", timeframe="1d", lookback=320, formula_version="copula-v1")
    result = build_copula_shadow_comparison(candidate=candidate, root=tmp_path, venue_policy=VenuePolicy(venue="dydx", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"))
    row = pd.read_csv(result["csv"]).iloc[0]
    assert row["sequential_decision"] == "TEST"
    assert row["dynamic_decision"] == "TEST"
    assert row["comparison"] == "exact_agreement"
    assert bool(row["shadow_only"])
    duplicate = build_copula_shadow_comparison(candidate=candidate, root=tmp_path, venue_policy=VenuePolicy(venue="dydx", account_eligible=True, product_type="perp", supports_short_leg=True, capability_status="verified"))
    assert duplicate["appended"] is False
    events = (tmp_path / "reports" / "orchestration" / "dynamic_agents" / "copula_comparison_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == 1
