from __future__ import annotations

import json
from datetime import datetime, timedelta

from eth_account import Account

from quant_platform.execution import OrderIntent
from quant_platform.hyperliquid_testnet import (
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
)
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
    _candidate_identity_core,
    _payload_hash,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_CANDIDATE_BINDING_FIELDS,
    _testnet_receipt_checks,
)
from quant_platform.orchestration.hyperliquid_testnet_lifecycle_evidence import (
    capture_hyperliquid_testnet_lifecycle_evidence,
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
    "generated_at_utc": "2026-08-08T11:58:00Z",
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
    "feature_timestamp_utc": "2026-08-08T11:59:00Z",
    "regime": "range",
    "trade_quality_score": 0.74,
    "strategy_signal_id": "signal-1",
}


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _EvidenceSession:
    def __init__(self, *, fills, positions, open_orders, funding=None):
        self.fills = fills
        self.positions = positions
        self.open_orders = open_orders
        self.funding = [] if funding is None else funding
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if json["type"] == "userFillsByTime":
            return _Response(self.fills)
        if json["type"] == "clearinghouseState":
            return _Response({"assetPositions": self.positions})
        if json["type"] == "openOrders":
            return _Response(self.open_orders)
        if json["type"] == "userFunding":
            return _Response(self.funding)
        raise AssertionError(f"unexpected info request: {json}")


def _config():
    return HyperliquidTestnetConfig(
        master_address=Account.create().address,
        agent_address=Account.create().address,
        keychain_service="unused-read-only-test-service",
        order_approval_id="approval-1",
    )


def _entry_intents():
    return (
        OrderIntent("BTC-USD", "BUY", 0.00016, 64_000.0),
        OrderIntent("ETH-USD", "SELL", 0.0034, 3_000.0),
    )


def _exit_intents():
    return (
        OrderIntent("BTC-USD", "SELL", 0.00016, 64_000.0, reduce_only=True),
        OrderIntent("ETH-USD", "BUY", 0.0034, 3_000.0, reduce_only=True),
    )


def _fill(*, oid, coin, size, price, fee, closed_pnl, time_ms, side, trade_id):
    return {
        "oid": oid,
        "coin": coin,
        "sz": str(size),
        "px": str(price),
        "fee": str(fee),
        "closedPnl": str(closed_pnl),
        "time": time_ms,
        "side": side,
        "tid": trade_id,
        "hash": "0x" + f"{trade_id:064x}",
        "feeToken": "USDC",
    }


def _write_identity_artifacts(root, config):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    candidate = seal_candidate_with_valid_queue(
        receipt=CANDIDATE_RECEIPT,
        root=root,
    )
    candidate_binding = {field: candidate[field] for field in TESTNET_CANDIDATE_BINDING_FIELDS}
    approval = {
        "approval_version": "hyperliquid-testnet-smoke-v10",
        "approval_id": "approval-1",
        "master_address": config.master_address,
        "agent_address": config.agent_address,
        "risk_override_applied": False,
        "entry_context": ENTRY_CONTEXT,
        "legs": [
            {
                "market": "BTC",
                "side": "BUY",
                "size": 0.00016,
                "limit_price": 64_000.0,
            },
            {
                "market": "ETH",
                "side": "SELL",
                "size": 0.0034,
                "limit_price": 3_000.0,
            },
        ],
        **candidate_binding,
    }
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(approval), encoding="utf-8"
    )
    (active / "testnet_candidate_receipt.json").write_text(json.dumps(candidate), encoding="utf-8")
    (active / "hyperliquid_run_manifest.json").write_text(
        json.dumps({"run_id": "run-1", "candidate_set_id": "set-1"}),
        encoding="utf-8",
    )
    protocol = {
        "protocol_id": "protocol-1",
        "protocol_status": "PASS",
        "simulation_is_testnet_proof": False,
        "order_submission_performed": False,
    }
    (active / "current_wizard_hyperliquid_testnet_protocol_manifest.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    return active, approval, protocol


def test_read_only_capture_builds_exchange_backed_lifecycle_receipt(tmp_path):
    config = _config()
    active, approval, protocol = _write_identity_artifacts(tmp_path, config)
    state_path = active / "hyperliquid_testnet_pair_execution_state.json"
    executor = HyperliquidTestnetPairExecutor(state_path=state_path)
    entry_state = executor._new_execution_state(_entry_intents(), config)
    entry_state.update(
        {
            "phase": "AWAITING_EXCHANGE_CONFIRMATION",
            "entry_submit_attempted": True,
            "exchange_reference_ids": ["101", "102"],
            "exchange_references": [
                {"market": "BTC", "order_id": "101"},
                {"market": "ETH", "order_id": "102"},
            ],
        }
    )
    executor._persist_execution_state(entry_state)
    started = datetime.fromisoformat(entry_state["created_at_utc"])
    entry_session = _EvidenceSession(
        fills=[
            _fill(
                oid=101,
                coin="BTC",
                size="0.00016",
                price="64005",
                fee="0.005",
                closed_pnl="0",
                time_ms=int(started.timestamp() * 1000) + 1_000,
                side="B",
                trade_id=1001,
            ),
            _fill(
                oid=102,
                coin="ETH",
                size="0.0034",
                price="2999",
                fee="0.005",
                closed_pnl="0",
                time_ms=int(started.timestamp() * 1000) + 1_100,
                side="A",
                trade_id=1002,
            ),
        ],
        positions=[
            {"position": {"coin": "BTC", "szi": "0.00016"}},
            {"position": {"coin": "ETH", "szi": "-0.0034"}},
        ],
        open_orders=[],
    )

    entry = capture_hyperliquid_testnet_lifecycle_evidence(
        root=tmp_path,
        config=config,
        session=entry_session,
        now=started + timedelta(seconds=10),
    )

    assert entry["status"] == "PARTIAL"
    assert entry["read_only"] is True
    assert entry["signing_key_loaded"] is False
    assert entry["order_submission_performed"] is False
    assert [request["json"]["type"] for request in entry_session.requests] == [
        "userFillsByTime",
        "clearinghouseState",
        "openOrders",
        "userFunding",
    ]

    exit_state = executor._new_execution_state(_exit_intents(), config)
    exit_state.update(
        {
            "phase": "AWAITING_EXCHANGE_CONFIRMATION",
            "entry_submit_attempted": True,
            "exit_submit_attempted": True,
            "entry_execution_state_id": entry_state["execution_state_id"],
            "duplicate_entry_blocked": True,
            "duplicate_entry_blocked_at_utc": (started + timedelta(seconds=15)).isoformat(),
            "exchange_reference_ids": ["201", "202"],
            "exchange_references": [
                {"market": "BTC", "order_id": "201"},
                {"market": "ETH", "order_id": "202"},
            ],
        }
    )
    executor._persist_execution_state(exit_state)
    exit_session = _EvidenceSession(
        fills=[
            _fill(
                oid=201,
                coin="BTC",
                size="0.00016",
                price="64010",
                fee="0.005",
                closed_pnl="0.04",
                time_ms=int(started.timestamp() * 1000) + 18_000,
                side="A",
                trade_id=2001,
            ),
            _fill(
                oid=202,
                coin="ETH",
                size="0.0034",
                price="2998",
                fee="0.005",
                closed_pnl="0.03",
                time_ms=int(started.timestamp() * 1000) + 18_100,
                side="B",
                trade_id=2002,
            ),
        ],
        positions=[],
        open_orders=[],
    )

    exit_result = capture_hyperliquid_testnet_lifecycle_evidence(
        root=tmp_path,
        config=config,
        session=exit_session,
        now=started + timedelta(seconds=20),
    )

    assert exit_result["status"] == "COMPLETE"
    receipt = json.loads(
        (active / "hyperliquid_testnet_smoke_receipt.json").read_text(encoding="utf-8")
    )
    assert {event["event_type"] for event in receipt["events"]} == {
        "two_leg_entry",
        "two_leg_exit",
        "reconciled",
        "idempotency",
    }
    assert receipt["final_state"]["reconciled"] is True
    assert receipt["entry_context"] == ENTRY_CONTEXT
    assert receipt["economics"]["economics_complete"] is True
    assert receipt["economics"]["terminal_fill_count"] == 4
    checks = _testnet_receipt_checks(
        receipt,
        approval=approval,
        protocol=protocol,
        root=tmp_path,
    )
    assert all(passed for _, passed, _ in checks)


def test_read_only_capture_records_partial_fill_as_anomaly(tmp_path):
    config = _config()
    active, _, _ = _write_identity_artifacts(tmp_path, config)
    state_path = active / "hyperliquid_testnet_pair_execution_state.json"
    executor = HyperliquidTestnetPairExecutor(state_path=state_path)
    state = executor._new_execution_state(_entry_intents(), config)
    state.update(
        {
            "entry_submit_attempted": True,
            "exchange_reference_ids": ["101", "102"],
        }
    )
    executor._persist_execution_state(state)
    started = datetime.fromisoformat(state["created_at_utc"])
    session = _EvidenceSession(
        fills=[
            _fill(
                oid=101,
                coin="BTC",
                size="0.00008",
                price="64005",
                fee="0.0025",
                closed_pnl="0",
                time_ms=int(started.timestamp() * 1000) + 1_000,
                side="B",
                trade_id=3001,
            ),
            _fill(
                oid=102,
                coin="ETH",
                size="0.0034",
                price="2999",
                fee="0.005",
                closed_pnl="0",
                time_ms=int(started.timestamp() * 1000) + 1_100,
                side="A",
                trade_id=3002,
            ),
        ],
        positions=[{"position": {"coin": "ETH", "szi": "-0.0034"}}],
        open_orders=[],
    )

    result = capture_hyperliquid_testnet_lifecycle_evidence(
        root=tmp_path,
        config=config,
        session=session,
        now=started + timedelta(seconds=5),
    )

    assert result["status"] == "PARTIAL"
    assert result["blockers"] == "hyperliquid_two_leg_terminal_fill_not_yet_proven"
    receipt = json.loads(
        (active / "hyperliquid_testnet_smoke_receipt.json").read_text(encoding="utf-8")
    )
    anomaly_types = {
        event["anomaly_type"]
        for event in receipt["events"]
        if event["event_type"] == "execution_anomaly"
    }
    assert anomaly_types == {"partial_fill", "orphan_position"}


def test_read_only_capture_records_one_leg_orphan_and_recovery(tmp_path):
    config = _config()
    active, _, _ = _write_identity_artifacts(tmp_path, config)
    state_path = active / "hyperliquid_testnet_pair_execution_state.json"
    executor = HyperliquidTestnetPairExecutor(state_path=state_path)
    state = executor._new_execution_state(_entry_intents(), config)
    state.update(
        {
            "entry_submit_attempted": True,
            "exchange_reference_ids": ["101"],
            "exchange_references": [{"market": "BTC", "order_id": "101"}],
            "recovery_actions": [
                "block_duplicate_entry_retry",
                "cancel_pair_open_orders",
                "reduce_only_flatten:BTC",
                "verify_flat_pair_state",
            ],
            "reconciled": True,
            "phase": "FLAT_RECONCILED",
        }
    )
    executor._persist_execution_state(state)
    started = datetime.fromisoformat(state["created_at_utc"])
    session = _EvidenceSession(
        fills=[
            _fill(
                oid=101,
                coin="BTC",
                size="0.00016",
                price="64005",
                fee="0.005",
                closed_pnl="0",
                time_ms=int(started.timestamp() * 1000) + 1_000,
                side="B",
                trade_id=4001,
            )
        ],
        positions=[],
        open_orders=[],
    )

    result = capture_hyperliquid_testnet_lifecycle_evidence(
        root=tmp_path,
        config=config,
        session=session,
        now=started + timedelta(seconds=5),
    )

    assert result["status"] == "PARTIAL"
    receipt = json.loads(
        (active / "hyperliquid_testnet_smoke_receipt.json").read_text(encoding="utf-8")
    )
    event_types = {event["event_type"] for event in receipt["events"]}
    assert event_types == {
        "execution_anomaly",
        "cancel_and_unwind",
        "emergency_flatten",
    }
    anomaly = next(
        event for event in receipt["events"] if event["event_type"] == "execution_anomaly"
    )
    assert anomaly["anomaly_type"] == "orphan_position"
    recovery_events = [
        event for event in receipt["events"] if event["event_type"] != "execution_anomaly"
    ]
    assert all(event["reduce_only"] is True for event in recovery_events)
    assert all(event["reconciled_flat"] is True for event in recovery_events)
    assert not any(event["event_type"] == "two_leg_entry" for event in receipt["events"])
