from __future__ import annotations

from quant_platform.orchestration.corrective_agent_governance import (
    build_agent_authority_inventory,
    run_agent_authority_fuzz_tests,
    validate_agent_packet,
)


def _packet():
    return {
        "candidate_id": "candidate_1234567890abcdef1234",
        "model_version": "v1",
        "feature_schema_version": "f1",
        "confidence": 0.5,
        "label_source": "backtest_label",
        "feature_timestamp": "2026-08-01T00:00:00Z",
        "label_timestamp": "2026-08-02T00:00:00Z",
        "evidence_hash": "a" * 64,
        "requested_action": "research",
    }


def test_agent_packets_cannot_request_execution():
    assert validate_agent_packet(_packet()) == []
    assert "agent_execution_action_forbidden" in validate_agent_packet({**_packet(), "requested_action": "submit_live_order"})


def test_inventory_has_no_current_live_submit_authority(tmp_path):
    frame = build_agent_authority_inventory(root=tmp_path)
    assert not frame["can_submit_live_orders"].any()
    assert not frame["can_change_acceptance_policy"].any()
    assert not frame["can_override_veto"].any()


def test_forged_packets_all_fail_closed(tmp_path):
    frame = run_agent_authority_fuzz_tests(root=tmp_path)
    assert frame["status"].eq("PASS").all()
    assert not frame["live_trading_authorized"].any()
