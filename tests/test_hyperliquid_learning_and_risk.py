from __future__ import annotations

import json

import pandas as pd

from quant_platform.hyperliquid_testnet import HyperliquidTestnetConfig
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    _testnet_receipt_payload_hash,
    build_testnet_lifecycle_gate,
    sign_testnet_smoke_approval,
    validate_testnet_smoke_approval,
    write_testnet_smoke_approval_template,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    validate_current_wizard_hyperliquid_testnet_protocol,
)


def test_smoke_approval_template_is_non_approved_and_fail_closed(tmp_path):
    result = write_testnet_smoke_approval_template(root=tmp_path)
    approval = json.loads(result["approval"].read_text(encoding="utf-8"))
    gate = build_testnet_lifecycle_gate(root=tmp_path)
    checks = pd.read_csv(gate["gate"])

    assert approval["approved"] is False
    assert approval["network"] == "testnet"
    assert approval["approval_version"] == "hyperliquid-testnet-smoke-v4"
    assert approval["max_total_notional_usd"] == 25.0
    assert approval["exit_policy"]["reduce_only"] is True
    assert gate["status"] == "BLOCKED"
    assert "explicit_testnet_smoke_approval_missing" in set(checks["blocker"].dropna())


def test_lifecycle_gate_requires_a_complete_bounded_receipt(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([{"ready_for_no_order_preflight": True}]).to_csv(
        active / "hyperliquid_testnet_preflight.csv",
        index=False,
    )
    manifest = {"run_id": "run-1", "candidate_set_id": "set-1"}
    (active / "hyperliquid_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-agent",
    )
    template = write_testnet_smoke_approval_template(root=tmp_path, config=config)
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    approval.update(
        {
            "approved": True,
            "approval_id": "bounded-smoke-001",
            "expires_at_utc": "2099-01-01T00:00:00Z",
            "legs": [
                {"market": "BTC", "side": "BUY", "size": 0.0002, "limit_price": 60_000.0},
                {"market": "ETH", "side": "SELL", "size": 0.004, "limit_price": 3_000.0},
            ],
        }
    )
    template["approval"].write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(root=tmp_path, approval_secret="separate-approval-secret", config=config)
    approval_validation = validate_testnet_smoke_approval(
        approval_id="bounded-smoke-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    mismatch_validation = validate_testnet_smoke_approval(
        approval_id="wrong-runtime-id",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    assert approval_validation["status"] == "PASS"
    assert mismatch_validation["status"] == "BLOCKED"
    assert "runtime_hyperliquid_order_approval_id_mismatch" in mismatch_validation[
        "blockers"
    ]
    protocol = validate_current_wizard_hyperliquid_testnet_protocol(root=tmp_path)
    event_types = [
        "two_leg_entry",
        "two_leg_exit",
        "reconciled",
        "idempotency",
    ]
    events = []
    for index, event_type in enumerate(event_types, start=1):
        event = {
            "event_id": f"event-{index}",
            "event_type": event_type,
            "timestamp_utc": f"2026-08-08T12:00:{index:02d}Z",
            "exchange_reference_ids": [f"reference-{index}"],
        }
        if event_type in {"two_leg_entry", "two_leg_exit"}:
            event.update(
                {
                    "exchange_reference_ids": [
                        f"{event_type}-order-x",
                        f"{event_type}-order-y",
                    ],
                    "leg_x_status": "filled",
                    "leg_y_status": "filled",
                }
            )
        if event_type == "idempotency":
            event["duplicate_submit_blocked"] = True
        events.append(event)
    receipt = {
        "receipt_version": "hyperliquid-testnet-lifecycle-v2",
        "receipt_source": "hyperliquid_testnet_lifecycle_evidence_capture",
        "actual_testnet": True,
        "network": "testnet",
        "approval_id": "bounded-smoke-001",
        "run_id": "run-1",
        "candidate_set_id": "set-1",
        "protocol_id": protocol.summary["protocol_id"],
        "started_at_utc": "2026-08-08T12:00:00Z",
        "completed_at_utc": "2026-08-08T12:01:00Z",
        "events": events,
        "final_state": {
            "reconciled": True,
            "position_x": 0.0,
            "position_y": 0.0,
            "open_order_count": 0,
            "account_state_timestamp_utc": "2026-08-08T12:00:30Z",
        },
    }
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

    result = build_testnet_lifecycle_gate(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )

    assert result["status"] == "PASS"

    receipt["events"][0]["leg_x_status"] = "resting"
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    resting_gate = build_testnet_lifecycle_gate(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    resting_checks = pd.read_csv(resting_gate["gate"])
    assert resting_gate["status"] == "BLOCKED"
    assert "two_leg_entry_not_proven" in set(resting_checks["blocker"].dropna())

    receipt["events"][0]["leg_x_status"] = "filled"

    receipt["events"].append(
        {
            "event_id": "event-anomaly",
            "event_type": "execution_anomaly",
            "anomaly_type": "partial_fill",
            "timestamp_utc": "2026-08-08T12:00:05Z",
            "exchange_reference_ids": ["anomaly-query-1"],
        }
    )
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    anomaly_gate = build_testnet_lifecycle_gate(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    anomaly_checks = pd.read_csv(anomaly_gate["gate"])
    assert anomaly_gate["status"] == "BLOCKED"
    assert "partial_fill_recovery_not_proven" in set(
        anomaly_checks["blocker"].dropna()
    )


def test_lifecycle_gate_rejects_boolean_only_receipt(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([{"ready_for_no_order_preflight": True}]).to_csv(
        active / "hyperliquid_testnet_preflight.csv", index=False
    )
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(
            {
                "two_leg_submit": True,
                "partial_fill_recovery": True,
                "cancel_and_unwind": True,
                "reconciled": True,
                "idempotency": True,
                "emergency_flatten": True,
            }
        ),
        encoding="utf-8",
    )
    validate_current_wizard_hyperliquid_testnet_protocol(root=tmp_path)

    result = build_testnet_lifecycle_gate(root=tmp_path)
    checks = pd.read_csv(result["gate"])

    assert result["status"] == "BLOCKED"
    assert "testnet_lifecycle_receipt_schema_invalid" in set(
        checks["blocker"].dropna()
    )


def test_signed_approval_binds_exact_leverage_margin_and_evidence_ids(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "run-3x", "candidate_set_id": "set-3x"}),
        encoding="utf-8",
    )
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-agent",
        requested_leverage=3,
        margin_mode="isolated",
        one_x_testnet_proof_id="one-x-proof-1",
        leverage_scenario_id="scenario-3x-1",
    )
    template = write_testnet_smoke_approval_template(root=tmp_path, config=config)
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    approval.update(
        {
            "approved": True,
            "approval_id": "bounded-3x-001",
            "expires_at_utc": "2099-01-01T00:00:00Z",
            "legs": [
                {
                    "market": "BTC",
                    "side": "BUY",
                    "size": 0.0002,
                    "limit_price": 60_000.0,
                },
                {
                    "market": "ETH",
                    "side": "SELL",
                    "size": 0.004,
                    "limit_price": 3_000.0,
                },
            ],
        }
    )
    template["approval"].write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )

    valid = validate_testnet_smoke_approval(
        approval_id="bounded-3x-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    drifted = validate_testnet_smoke_approval(
        approval_id="bounded-3x-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=HyperliquidTestnetConfig(
            master_address=config.master_address,
            agent_address=config.agent_address,
            keychain_service=config.keychain_service,
            requested_leverage=2,
            margin_mode="isolated",
            one_x_testnet_proof_id="one-x-proof-1",
            leverage_scenario_id="scenario-3x-1",
        ),
    )

    assert valid["status"] == "PASS"
    assert drifted["status"] == "BLOCKED"
    assert "testnet_smoke_requested_leverage_mismatch" in drifted["blockers"]


def test_lifecycle_gate_rejects_tampered_signed_payload(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame([{"ready_for_no_order_preflight": True}]).to_csv(
        active / "hyperliquid_testnet_preflight.csv", index=False
    )
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "run-1", "candidate_set_id": "set-1"}), encoding="utf-8"
    )
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-agent",
    )
    template = write_testnet_smoke_approval_template(root=tmp_path, config=config)
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    approval.update(
        {
            "approved": True,
            "approval_id": "bounded-smoke-001",
            "expires_at_utc": "2099-01-01T00:00:00Z",
            "legs": [
                {"market": "BTC", "side": "BUY", "size": 0.0002, "limit_price": 60_000.0},
                {"market": "ETH", "side": "SELL", "size": 0.004, "limit_price": 3_000.0},
            ],
        }
    )
    template["approval"].write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(root=tmp_path, approval_secret="separate-approval-secret", config=config)
    signed = json.loads(template["approval"].read_text(encoding="utf-8"))
    signed["legs"][0]["size"] = 0.00021
    template["approval"].write_text(json.dumps(signed), encoding="utf-8")

    result = build_testnet_lifecycle_gate(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    checks = pd.read_csv(result["gate"])

    assert result["status"] == "BLOCKED"
    assert "testnet_smoke_payload_hash_mismatch" in set(checks["blocker"].dropna())
