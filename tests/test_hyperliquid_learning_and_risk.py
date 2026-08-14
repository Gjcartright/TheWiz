from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from quant_platform.hyperliquid_testnet import HyperliquidTestnetConfig
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
    _candidate_identity_core,
    _payload_hash,
)
from quant_platform.orchestration.current_wizard_hyperliquid_testnet_protocol import (
    validate_current_wizard_hyperliquid_testnet_protocol,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_CANDIDATE_BINDING_FIELDS,
    _testnet_receipt_payload_hash,
    build_testnet_lifecycle_gate,
    sign_testnet_smoke_approval,
    validate_testnet_smoke_approval,
    write_testnet_smoke_approval_template,
)
from tests.candidate_queue_support import seal_candidate_with_valid_queue

BASE_CANDIDATE_BINDING = {
    "candidate_experiment_id": "experiment-1",
    "pair_group_key": "hyperliquid|daily|BTC|ETH",
    "pair": "BTC-USD-ETH-USD",
    "timeframe": "daily",
    "exact_mode": "OU Optimal",
    "orientation": "original",
    "cost_model_id": "cost-model-1",
    "model_training_dataset_id": "dataset-1",
    "model_artifact_sha256": "a" * 64,
}
CANDIDATE_RECEIPT = {
    "schema_version": CANDIDATE_SCHEMA_VERSION,
    "generated_at_utc": (datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
    "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
    "candidate_leverage": 1.0,
    "asset_x": "BTC",
    "asset_y": "ETH",
    "registered_learning_id": "registered-learning-1",
    "registered_learning_receipt_sha256": "b" * 64,
    "registered_execution_id": "registered-execution-1",
    "survivor_receipt_id": "survivor-1",
    "source_artifact_hashes": {"stage5": "c" * 64},
    "blockers": [],
    "order_submission_performed": False,
    "testnet_order_authority": False,
    "live_trading_authorized": False,
    **BASE_CANDIDATE_BINDING,
}
CANDIDATE_BINDING = {
    "candidate_receipt_id": "testnetcandidate_"
    + _payload_hash(_candidate_identity_core(CANDIDATE_RECEIPT))[:20],
    **BASE_CANDIDATE_BINDING,
}

ENTRY_CONTEXT = {
    "feature_timestamp_utc": (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
    "regime": "range",
    "trade_quality_score": 0.74,
    "strategy_signal_id": "signal-1",
}


def _approval_window():
    issued = datetime.now(UTC)
    return {
        "issued_at_utc": issued.isoformat(),
        "expires_at_utc": (issued + timedelta(minutes=10)).isoformat(),
    }


def _write_testnet_candidate(active, *, generated_at_utc=None):
    receipt = dict(CANDIDATE_RECEIPT)
    if generated_at_utc is not None:
        receipt["generated_at_utc"] = generated_at_utc
    candidate = seal_candidate_with_valid_queue(
        receipt=receipt, root=active.parents[1]
    )
    CANDIDATE_BINDING.clear()
    CANDIDATE_BINDING.update(
        {
            field: candidate[field]
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        }
    )
    (active / "testnet_candidate_receipt.json").write_text(
        json.dumps(candidate),
        encoding="utf-8",
    )


def _write_approvable_payload(
    root,
    *,
    approval_id="bounded-smoke-immutable-001",
    candidate_generated_at_utc=None,
    feature_timestamp_utc=None,
):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    _write_testnet_candidate(
        active,
        generated_at_utc=candidate_generated_at_utc,
    )
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "run-immutable", "candidate_set_id": "set-immutable"}),
        encoding="utf-8",
    )
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-agent",
    )
    template = write_testnet_smoke_approval_template(root=root, config=config)
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    approval.update(
        {
            "approved": True,
            "approval_id": approval_id,
            **_approval_window(),
            "entry_context": {
                **ENTRY_CONTEXT,
                "feature_timestamp_utc": feature_timestamp_utc
                or (datetime.now(UTC) - timedelta(minutes=1)).isoformat(),
            },
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
    return template, config


def test_smoke_approval_template_is_non_approved_and_fail_closed(tmp_path):
    result = write_testnet_smoke_approval_template(root=tmp_path)
    approval = json.loads(result["approval"].read_text(encoding="utf-8"))
    gate = build_testnet_lifecycle_gate(root=tmp_path)
    checks = pd.read_csv(gate["gate"])

    assert approval["approved"] is False
    assert approval["network"] == "testnet"
    assert approval["approval_version"] == "hyperliquid-testnet-smoke-v10"
    assert approval["max_total_notional_usd"] == 25.0
    assert approval["maximum_exit_slippage_bps"] == 50.0
    assert approval["collateral_transfer_policy"] == {
        "approved": False,
        "direction": "spot_to_perp",
        "amount_usd": 25.0,
        "one_use": True,
        "maximum_transfer_attempts": 1,
    }
    assert approval["exit_policy"]["reduce_only"] is True
    assert gate["status"] == "BLOCKED"
    assert "explicit_testnet_smoke_approval_missing" in set(checks["blocker"].dropna())


def test_signed_approval_is_bound_to_an_immutable_identity_artifact(tmp_path):
    template, config = _write_approvable_payload(tmp_path)

    signed = sign_testnet_smoke_approval(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    identity_path = signed["approval_identity"]
    active_payload = json.loads(template["approval"].read_text(encoding="utf-8"))

    assert identity_path.is_file()
    assert signed["approval_identity_sha256"]
    assert json.loads(identity_path.read_text(encoding="utf-8")) == active_payload

    active_payload["execution_authority"] = "mutated_non_signed_display_field"
    template["approval"].write_text(json.dumps(active_payload), encoding="utf-8")
    validation = validate_testnet_smoke_approval(
        approval_id="bounded-smoke-immutable-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )

    assert validation["status"] == "BLOCKED"
    assert (
        "testnet_smoke_approval_identity_artifact_mismatch"
        in validation["blockers"]
    )
    with pytest.raises(ValueError, match="already bound to a different payload"):
        sign_testnet_smoke_approval(
            root=tmp_path,
            approval_secret="separate-approval-secret",
            config=config,
        )


def test_expired_entry_approval_retains_only_immutable_reduce_only_exit_authority(
    tmp_path,
):
    template, config = _write_approvable_payload(tmp_path)
    sign_testnet_smoke_approval(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    approval = json.loads(template["approval"].read_text(encoding="utf-8"))
    after_expiry = datetime.fromisoformat(approval["expires_at_utc"]) + timedelta(
        minutes=10
    )

    active = tmp_path / "reports" / "active"
    (active / "testnet_candidate_receipt.json").write_text(
        json.dumps({"candidate_status": "ROTATED"}), encoding="utf-8"
    )
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "new-run", "candidate_set_id": "new-set"}),
        encoding="utf-8",
    )

    entry = validate_testnet_smoke_approval(
        approval_id="bounded-smoke-immutable-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
        validation_scope="entry",
        now=after_expiry,
    )
    exit_authority = validate_testnet_smoke_approval(
        approval_id="bounded-smoke-immutable-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
        validation_scope="exit",
        now=after_expiry,
    )

    assert entry["status"] == "BLOCKED"
    assert "testnet_smoke_approval_missing_or_expired" in entry["blockers"]
    assert exit_authority["status"] == "PASS"
    assert exit_authority["authority_scope"] == "exit"
    assert exit_authority["execution_allowed"] is True
    assert exit_authority["testnet_order_authority"] is True
    assert exit_authority["live_trading_authorized"] is False


@pytest.mark.parametrize(
    "candidate_time",
    [
        lambda: datetime.now(UTC) - timedelta(hours=3),
        lambda: datetime.now(UTC) + timedelta(minutes=1),
    ],
)
def test_approval_signing_rejects_stale_or_future_candidate(tmp_path, candidate_time):
    _, config = _write_approvable_payload(
        tmp_path,
        candidate_generated_at_utc=candidate_time().isoformat(),
    )

    with pytest.raises(ValueError, match="testnet_smoke_candidate_receipt_stale_or_future"):
        sign_testnet_smoke_approval(
            root=tmp_path,
            approval_secret="separate-approval-secret",
            config=config,
        )


@pytest.mark.parametrize(
    "feature_time",
    [
        lambda: datetime.now(UTC) - timedelta(minutes=6),
        lambda: datetime.now(UTC) + timedelta(minutes=1),
    ],
)
def test_approval_signing_rejects_stale_or_future_entry_features(tmp_path, feature_time):
    _, config = _write_approvable_payload(
        tmp_path,
        feature_timestamp_utc=feature_time().isoformat(),
    )

    with pytest.raises(ValueError, match="testnet_smoke_entry_context_stale_or_future"):
        sign_testnet_smoke_approval(
            root=tmp_path,
            approval_secret="separate-approval-secret",
            config=config,
        )


def test_lifecycle_gate_requires_a_complete_bounded_receipt(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    _write_testnet_candidate(active)
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
            **_approval_window(),
            "entry_context": ENTRY_CONTEXT,
            "legs": [
                {"market": "BTC", "side": "BUY", "size": 0.0002, "limit_price": 60_000.0},
                {"market": "ETH", "side": "SELL", "size": 0.004, "limit_price": 3_000.0},
            ],
        }
    )
    template["approval"].write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(
        root=tmp_path, approval_secret="separate-approval-secret", config=config
    )
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
    assert approval_validation["execution_allowed"] is True
    assert approval_validation["testnet_order_authority"] is True
    assert approval_validation["live_trading_authorized"] is False
    assert mismatch_validation["status"] == "BLOCKED"
    assert mismatch_validation["execution_allowed"] is False
    assert mismatch_validation["testnet_order_authority"] is False
    assert "runtime_hyperliquid_order_approval_id_mismatch" in mismatch_validation["blockers"]
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
            closed_pnls = [0.0, 0.0] if event_type == "two_leg_entry" else [0.04, 0.03]
            fill_phase_offset_ms = 0 if event_type == "two_leg_entry" else 10_000
            event.update(
                {
                    "exchange_reference_ids": [
                        f"{event_type}-order-x",
                        f"{event_type}-order-y",
                    ],
                    "leg_x_status": "filled",
                    "leg_y_status": "filled",
                    "fill_economics_complete": True,
                    "fill_evidence": [
                        {
                            "order_id": f"{event_type}-order-x",
                            "trade_id": f"{event_type}-trade-x",
                            "transaction_hash": "0x" + "1" * 64,
                            "coin": "BTC",
                            "intent_side": "BUY" if event_type == "two_leg_entry" else "SELL",
                            "exchange_side": "B" if event_type == "two_leg_entry" else "A",
                            "size": 0.0002,
                            "price": 60_000.0,
                            "expected_limit_price": 60_000.0,
                            "fee_usd": 0.005,
                            "fee_token": "USDC",
                            "closed_pnl_usd": closed_pnls[0],
                            "implementation_shortfall_vs_limit_usd": 0.0,
                            "fill_time_ms": (
                                1_786_190_400_000 + fill_phase_offset_ms + index
                            ),
                            "economics_complete": True,
                        },
                        {
                            "order_id": f"{event_type}-order-y",
                            "trade_id": f"{event_type}-trade-y",
                            "transaction_hash": "0x" + "2" * 64,
                            "coin": "ETH",
                            "intent_side": "SELL" if event_type == "two_leg_entry" else "BUY",
                            "exchange_side": "A" if event_type == "two_leg_entry" else "B",
                            "size": 0.004,
                            "price": 3_000.0,
                            "expected_limit_price": 3_000.0,
                            "fee_usd": 0.005,
                            "fee_token": "USDC",
                            "closed_pnl_usd": closed_pnls[1],
                            "implementation_shortfall_vs_limit_usd": 0.0,
                            "fill_time_ms": (
                                1_786_190_400_100 + fill_phase_offset_ms + index
                            ),
                            "economics_complete": True,
                        },
                    ],
                }
            )
        if event_type == "idempotency":
            event["duplicate_submit_blocked"] = True
        events.append(event)
    receipt = {
        "receipt_version": "hyperliquid-testnet-lifecycle-v3",
        "receipt_source": "hyperliquid_testnet_lifecycle_evidence_capture",
        "actual_testnet": True,
        "network": "testnet",
        "approval_id": "bounded-smoke-001",
        "run_id": "run-1",
        "candidate_set_id": "set-1",
        "protocol_id": protocol.summary["protocol_id"],
        **CANDIDATE_BINDING,
        "risk_override_applied": False,
        "entry_context": ENTRY_CONTEXT,
        "started_at_utc": "2026-08-08T12:00:00Z",
        "completed_at_utc": "2026-08-08T12:01:00Z",
        "events": events,
        "funding_events": [],
        "funding_query_complete": True,
        "economics": {
            "economics_complete": True,
            "terminal_fill_count": 4,
            "gross_realized_pnl_usd": 0.07,
            "fees_usd": 0.02,
            "funding_pnl_usd": 0.0,
            "implementation_shortfall_vs_limit_usd": 0.0,
            "net_realized_pnl_after_cost_usd": 0.05,
            "slippage_treatment": "diagnostic_only_already_reflected_in_realized_pnl_not_double_subtracted",
        },
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

    original_trade_id = receipt["events"][0]["fill_evidence"][1]["trade_id"]
    receipt["events"][0]["fill_evidence"][1]["trade_id"] = receipt["events"][0][
        "fill_evidence"
    ][0]["trade_id"]
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    duplicate_fill_gate = build_testnet_lifecycle_gate(
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    duplicate_fill_checks = pd.read_csv(duplicate_fill_gate["gate"])
    assert duplicate_fill_gate["status"] == "BLOCKED"
    assert "testnet_lifecycle_after_cost_economics_missing_or_invalid" in set(
        duplicate_fill_checks["blocker"].dropna()
    )
    receipt["events"][0]["fill_evidence"][1]["trade_id"] = original_trade_id
    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    (active / "hyperliquid_testnet_smoke_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )

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
    assert "partial_fill_recovery_not_proven" in set(anomaly_checks["blocker"].dropna())


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
    assert "testnet_lifecycle_receipt_schema_invalid" in set(checks["blocker"].dropna())


def test_signed_approval_binds_exact_leverage_margin_and_evidence_ids(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    _write_testnet_candidate(active)
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
            **_approval_window(),
            "entry_context": ENTRY_CONTEXT,
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
    assert valid["execution_allowed"] is True
    assert valid["testnet_order_authority"] is True
    assert drifted["status"] == "BLOCKED"
    assert drifted["execution_allowed"] is False
    assert drifted["testnet_order_authority"] is False
    assert "testnet_smoke_requested_leverage_mismatch" in drifted["blockers"]

    overlong = json.loads(template["approval"].read_text(encoding="utf-8"))
    overlong["expires_at_utc"] = (
        pd.Timestamp(overlong["issued_at_utc"]) + pd.Timedelta(minutes=16)
    ).isoformat()
    template["approval"].write_text(json.dumps(overlong), encoding="utf-8")
    overlong_validation = validate_testnet_smoke_approval(
        approval_id="bounded-3x-001",
        root=tmp_path,
        approval_secret="separate-approval-secret",
        config=config,
    )
    assert overlong_validation["status"] == "BLOCKED"
    assert (
        "testnet_smoke_approval_missing_or_expired"
        in overlong_validation["blockers"]
    )


def test_lifecycle_gate_rejects_tampered_signed_payload(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    _write_testnet_candidate(active)
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
            **_approval_window(),
            "entry_context": ENTRY_CONTEXT,
            "legs": [
                {"market": "BTC", "side": "BUY", "size": 0.0002, "limit_price": 60_000.0},
                {"market": "ETH", "side": "SELL", "size": 0.004, "limit_price": 3_000.0},
            ],
        }
    )
    template["approval"].write_text(json.dumps(approval), encoding="utf-8")
    sign_testnet_smoke_approval(
        root=tmp_path, approval_secret="separate-approval-secret", config=config
    )
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
