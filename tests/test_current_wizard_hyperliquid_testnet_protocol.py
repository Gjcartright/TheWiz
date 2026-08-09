from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    REQUIRED_SCENARIOS,
    validate_current_wizard_hyperliquid_testnet_protocol,
)


def test_protocol_exercises_required_recovery_paths_without_orders(
    tmp_path: Path,
) -> None:
    result = validate_current_wizard_hyperliquid_testnet_protocol(root=tmp_path)
    scenarios = pd.read_csv(result.paths["scenarios"], keep_default_na=False)
    transitions = pd.read_csv(result.paths["transitions"], keep_default_na=False)
    validation = pd.read_csv(result.paths["validation"], keep_default_na=False)

    assert result.summary["protocol_status"] == "PASS"
    assert set(scenarios["scenario_id"]) == REQUIRED_SCENARIOS
    assert scenarios["scenario_pass"].astype(bool).all()
    assert scenarios["terminal_flat"].astype(bool).all()
    assert validation["status"].eq("PASS").all()
    assert result.summary["leverage_candidates_accounted"] == 0
    assert result.summary["actual_testnet_lifecycle_proven"] == 0
    assert result.summary["simulation_only"] is True
    assert result.summary["simulation_is_testnet_proof"] is False
    assert result.summary["order_submission_performed"] is False
    assert result.summary["live_trading_authorized"] is False
    assert not transitions["order_submission_performed"].astype(bool).any()

    partial = transitions.loc[
        transitions["scenario_id"].eq("partial_orphan_recovery")
    ]
    assert "CANCEL_REMAINDER_THEN_REDUCE_ONLY_FLATTEN" in set(partial["action"])
    assert partial["reduce_only_required"].astype(bool).any()

    unconfirmed = transitions.loc[
        transitions["scenario_id"].eq("unconfirmed_response_reconciliation")
    ]
    assert "QUERY_ACCOUNT_AND_ORDERS_NO_RETRY" in set(unconfirmed["action"])
    assert unconfirmed["duplicate_submit_blocked"].astype(bool).any()


def test_protocol_accounts_for_each_leverage_candidate_but_claims_no_proof(
    tmp_path: Path,
) -> None:
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "candidate_id": "candidate-1",
                "pair": "BTC-ETH",
                "requested_leverage": 2.0,
                "one_x_research_accepted": True,
                "leverage_scenario_accepted": True,
            }
        ]
    ).to_csv(
        active / "current_wizard_hyperliquid_leverage_candidates.csv",
        index=False,
    )
    (active / "current_wizard_hyperliquid_leverage_manifest.json").write_text(
        json.dumps({"leverage_candidates_complete": 1}),
        encoding="utf-8",
    )

    result = validate_current_wizard_hyperliquid_testnet_protocol(root=tmp_path)
    coverage = pd.read_csv(
        result.paths["candidate_coverage"], keep_default_na=False
    )

    assert result.summary["protocol_status"] == "PASS"
    assert result.summary["leverage_candidates_expected"] == 1
    assert result.summary["leverage_candidates_accounted"] == 1
    assert coverage["candidate_id"].tolist() == ["candidate-1"]
    assert coverage["protocol_simulation_pass"].astype(bool).all()
    assert not coverage["actual_testnet_lifecycle_proven"].astype(bool).any()
    assert not coverage["execution_authority"].astype(bool).any()
    assert not coverage["order_submission_performed"].astype(bool).any()
    assert not coverage["live_trading_authorized"].astype(bool).any()


def test_protocol_id_is_stable_for_identical_inputs(tmp_path: Path) -> None:
    first = validate_current_wizard_hyperliquid_testnet_protocol(
        root=tmp_path,
        now=datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc),
    )
    second = validate_current_wizard_hyperliquid_testnet_protocol(
        root=tmp_path,
        now=datetime(2026, 8, 8, 13, 0, tzinfo=timezone.utc),
    )

    assert first.summary["protocol_id"] == second.summary["protocol_id"]
    assert first.summary["as_of"] != second.summary["as_of"]
