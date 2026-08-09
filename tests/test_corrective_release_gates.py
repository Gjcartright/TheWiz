from __future__ import annotations

import json

from quant_platform.orchestration.corrective_release_gates import build_corrective_release_gates


def test_zero_survivor_blocks_all_testnet_and_live_authority(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "testnet_sample_sufficiency_policy.json").write_text(
        json.dumps(
            {
                "minimum_closed_paired_lifecycles": 30,
                "minimum_observation_days": 14,
                "minimum_independent_pairs": 3,
                "minimum_observed_regimes": 3,
                "maximum_unresolved_orphan_legs": 0,
            }
        )
    )
    (active / "final_1x_survivor_receipt.json").write_text(
        json.dumps({"receipt_status": "ZERO_SURVIVORS", "final_experiment_ids": []})
    )
    result = build_corrective_release_gates(root=tmp_path)
    assert result.summary["status"] == "BLOCKED"
    assert result.summary["orders_submitted"] == 0
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    authorization = json.loads(result.paths["live_canary_authorization"].read_text())
    assert authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert authorization["live_trading_authorized"] is False
