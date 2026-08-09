from __future__ import annotations

from datetime import datetime, timedelta
import json

from eth_account import Account

from quant_platform.execution import OrderIntent
from quant_platform.hyperliquid_testnet import (
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    _testnet_receipt_checks,
)
from quant_platform.orchestration.hyperliquid_testnet_lifecycle_evidence import (
    capture_hyperliquid_testnet_lifecycle_evidence,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _EvidenceSession:
    def __init__(self, *, fills, positions, open_orders):
        self.fills = fills
        self.positions = positions
        self.open_orders = open_orders
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if json["type"] == "userFillsByTime":
            return _Response(self.fills)
        if json["type"] == "clearinghouseState":
            return _Response({"assetPositions": self.positions})
        if json["type"] == "openOrders":
            return _Response(self.open_orders)
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


def _write_identity_artifacts(root, config):
    active = root / "reports" / "active"
    active.mkdir(parents=True)
    approval = {
        "approval_version": "hyperliquid-testnet-smoke-v4",
        "approval_id": "approval-1",
        "master_address": config.master_address,
        "agent_address": config.agent_address,
    }
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(approval), encoding="utf-8"
    )
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
            {"oid": 101, "coin": "BTC", "sz": "0.00016"},
            {"oid": 102, "coin": "ETH", "sz": "0.0034"},
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
    ]

    exit_state = executor._new_execution_state(_exit_intents(), config)
    exit_state.update(
        {
            "phase": "AWAITING_EXCHANGE_CONFIRMATION",
            "entry_submit_attempted": True,
            "exit_submit_attempted": True,
            "entry_execution_state_id": entry_state["execution_state_id"],
            "duplicate_entry_blocked": True,
            "duplicate_entry_blocked_at_utc": (
                started + timedelta(seconds=15)
            ).isoformat(),
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
            {"oid": 201, "coin": "BTC", "sz": "0.00016"},
            {"oid": 202, "coin": "ETH", "sz": "0.0034"},
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
        (active / "hyperliquid_testnet_smoke_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert {event["event_type"] for event in receipt["events"]} == {
        "two_leg_entry",
        "two_leg_exit",
        "reconciled",
        "idempotency",
    }
    assert receipt["final_state"]["reconciled"] is True
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
            {"oid": 101, "coin": "BTC", "sz": "0.00008"},
            {"oid": 102, "coin": "ETH", "sz": "0.0034"},
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
        (active / "hyperliquid_testnet_smoke_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["events"][0]["event_type"] == "execution_anomaly"
    assert receipt["events"][0]["anomaly_type"] == "partial_fill"
