from __future__ import annotations

import pandas as pd

from quant_platform.orchestration.corrective_statistical_remediation import _jaccard, _missing_proof


def test_trade_overlap_clusters_exact_duplicates():
    left = {("a", "b", "long"), ("c", "d", "short")}
    assert _jaccard(left, set(left)) == 1.0
    assert _jaccard(left, {("x", "y", "long")}) == 0.0


def test_missing_proof_order_is_fail_closed():
    row = pd.Series(
        {
            "strict_cost_calibration_ready": False,
            "vendor_exact_mode_parity_proven": False,
            "statistical_selection_status": "PASS",
            "regime_stability_status": "PASS_RESEARCH_REGIME_STABILITY",
            "research_robustness_status": "PASS_RESEARCH_ROBUSTNESS",
            "concentration_gate_pass": True,
        }
    )
    assert _missing_proof(row) == "strict_observed_cost_calibration"
