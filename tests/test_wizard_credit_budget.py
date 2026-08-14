from __future__ import annotations

import pandas as pd

from quant_platform.wizard_credit_budget import build_wizard_credit_budget_contract


def test_current_scheduled_wizard_lanes_fit_with_large_headroom(tmp_path):
    result = build_wizard_credit_budget_contract(root=tmp_path)
    frame = pd.read_csv(result.paths["budget_contract"])

    assert result.summary["status"] == "PASS"
    assert result.summary["discovery_sweep_credit_ceiling"] == 300
    assert result.summary["exact_mode_proof_credit_ceiling"] == 60
    assert result.summary["copula_behavioral_credit_ceiling"] == 8
    assert result.summary["ou_v3_prospective_credit_ceiling"] == 8
    assert result.summary["ou_v4_prospective_credit_ceiling"] == 16
    assert result.summary["ou_v5_prospective_credit_ceiling"] == 16
    assert result.summary["ou_v6_prospective_credit_ceiling"] == 16
    assert result.summary["scheduled_credit_ceiling"] == 424
    assert result.summary["headroom_after_reserve"] == 476
    assert (
        frame.loc[
            frame["lane"].eq("ou_v4_prospective_holdout"),
            "maximum_requests",
        ].iat[0]
        == 8
    )
    assert (
        frame.loc[
            frame["lane"].eq("ou_v5_prospective_holdout"),
            "maximum_requests",
        ].iat[0]
        == 8
    )
    assert (
        frame.loc[
            frame["lane"].eq("ou_v6_prospective_holdout"),
            "maximum_requests",
        ].iat[0]
        == 8
    )
    assert not frame["order_submission_capability"].astype(bool).any()
    assert result.summary["live_trading_authorized"] is False


def test_credit_contract_fails_closed_when_scheduled_lanes_outgrow_budget(tmp_path):
    result = build_wizard_credit_budget_contract(
        root=tmp_path,
        proof_max_batches=101,
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["headroom_after_reserve"] < 0
    assert result.summary["blocker"] == "scheduled_wizard_lanes_exceed_daily_budget"
