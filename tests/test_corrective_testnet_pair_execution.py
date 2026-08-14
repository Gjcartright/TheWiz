from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

from quant_platform.hyperliquid_testnet import (
    HyperliquidPairExecutionResult,
    HyperliquidTestnetConfig,
)
from quant_platform.orchestration.corrective_testnet_pair_execution import (
    ENABLE_ENV,
    ENTRY_ACKNOWLEDGEMENT,
    EXIT_ACKNOWLEDGEMENT,
    build_testnet_pair_execution_preflight,
    run_testnet_pair_execution,
)


def _config(*, submit_orders=False):
    return HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address="0x" + "2" * 40,
        keychain_service="test-keychain",
        submit_orders=submit_orders,
    )


def _approval(approval_id="approval-1"):
    return {
        "approval_id": approval_id,
        "candidate_receipt_id": "testnetcandidate_candidate1",
        "candidate_experiment_id": "experiment-1",
        "pair_group_key": "hyperliquid|hourly|BTC|ETH",
        "pair": "BTC-USD-ETH-USD",
        "timeframe": "hourly",
        "exact_mode": "Dyn (ZScoreR)",
        "orientation": "reverse",
        "cost_model_id": "cost-model-1",
        "model_training_dataset_id": "dataset-1",
        "model_artifact_sha256": "a" * 64,
        "registered_learning_id": "learning-1",
        "registered_learning_receipt_path": "data/research/registered_learning/learning-1.json",
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_stage5_protocol_id": "protocol-1",
        "registered_stage5_protocol_sha256": "c" * 64,
        "registered_execution_id": "execution-1",
        "entry_context": {
            "feature_timestamp_utc": "2026-08-14T00:00:00Z",
            "regime": "range",
            "trade_quality_score": 0.75,
            "strategy_signal_id": "signal-1",
        },
        "maximum_exit_slippage_bps": 50.0,
        "legs": [
            {"market": "BTC", "side": "BUY", "size": 0.0002, "limit_price": 60_000.0},
            {"market": "ETH", "side": "SELL", "size": 0.004, "limit_price": 3_000.0},
        ],
    }


def _write_sources(root, approval_id="approval-1"):
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(_approval(approval_id)), encoding="utf-8"
    )
    (active / "testnet_candidate_receipt.json").write_text(
        json.dumps({"candidate_receipt_id": "testnetcandidate_candidate1"}),
        encoding="utf-8",
    )
    return active


def _approval_validator(approval_id, root, config, scope, now):
    return {
        "status": "PASS",
        "blockers": [],
        "execution_allowed": True,
        "testnet_order_authority": True,
        "authority_scope": scope,
        "live_trading_authorized": False,
    }


def _info_client(*, positions=None):
    positions = positions or []

    def fetch(payload):
        if payload["type"] == "meta":
            return {
                "universe": [
                    {"name": "BTC", "szDecimals": 5, "maxLeverage": 50},
                    {"name": "ETH", "szDecimals": 4, "maxLeverage": 50},
                ]
            }
        if payload["type"] == "clearinghouseState":
            return {"assetPositions": positions}
        if payload["type"] == "openOrders":
            return []
        if payload["type"] == "allMids":
            return {"BTC": "70000", "ETH": "2500"}
        raise AssertionError(payload)

    return fetch


def _write_entry_state(active, approval_id="approval-1"):
    state = {
        "schema_version": "hyperliquid_testnet_pair_execution_state.v1",
        "execution_state_id": "entry-state-1",
        "network": "testnet",
        "master_address": "0x" + "1" * 40,
        "agent_address": "0x" + "2" * 40,
        "order_approval_id": approval_id,
        "approval_validated_at_entry": True,
        "intents": [
            {
                "market": "BTC",
                "side": "BUY",
                "size": 0.0002,
                "limit_price": 60_000.0,
                "reduce_only": False,
            },
            {
                "market": "ETH",
                "side": "SELL",
                "size": 0.004,
                "limit_price": 3_000.0,
                "reduce_only": False,
            },
        ],
        "submit_kind": "entry",
        "phase": "AWAITING_EXCHANGE_CONFIRMATION",
        "entry_submit_attempted": True,
        "exit_submit_attempted": False,
        "live_trading_authorized": False,
    }
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    state["state_hash"] = sha256(canonical.encode("utf-8")).hexdigest()
    (active / "hyperliquid_testnet_pair_execution_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    def submit_pair(self, intents, config):
        self.calls.append((intents, config))
        return HyperliquidPairExecutionResult(
            status="pair_submitted",
            reason="submitted",
            order_submission_performed=True,
            live_trading_authorized=False,
        )


def test_status_only_entry_preflight_never_invokes_executor(tmp_path, monkeypatch):
    _write_sources(tmp_path)
    preflight = build_testnet_pair_execution_preflight(
        root=tmp_path,
        action="entry",
        approval_id="approval-1",
        config=_config(),
        approval_validator=_approval_validator,
        candidate_validator=lambda root, candidate: (True, []),
        info_client=_info_client(),
    )
    fake = _FakeExecutor()
    monkeypatch.delenv(ENABLE_ENV, raising=False)

    result = run_testnet_pair_execution(
        root=tmp_path,
        action="entry",
        preflight_id=preflight.summary["preflight_id"],
        approval_id="approval-1",
        config=_config(submit_orders=True),
        pair_executor=fake,
    )

    assert preflight.summary["status"] == "READY_REQUIRES_EXPLICIT_EXECUTION"
    assert result.summary["status"] == "NO_SUBMISSION_STATUS_ONLY"
    assert "testnet_pair_execute_flag_not_set" in result.summary["blockers"]
    assert result.summary["order_submission_performed"] is False
    assert fake.calls == []


def test_explicit_entry_executes_exact_sealed_intents_once(tmp_path, monkeypatch):
    _write_sources(tmp_path)
    now = datetime.now(UTC)
    preflight = build_testnet_pair_execution_preflight(
        root=tmp_path,
        action="entry",
        approval_id="approval-1",
        config=_config(),
        now=now,
        approval_validator=_approval_validator,
        candidate_validator=lambda root, candidate: (True, []),
        info_client=_info_client(),
    )
    fake = _FakeExecutor()
    monkeypatch.setenv(ENABLE_ENV, "true")

    result = run_testnet_pair_execution(
        root=tmp_path,
        action="entry",
        preflight_id=preflight.summary["preflight_id"],
        approval_id="approval-1",
        acknowledgement=ENTRY_ACKNOWLEDGEMENT,
        execute=True,
        config=_config(submit_orders=True),
        now=now,
        pair_executor=fake,
    )
    replay = run_testnet_pair_execution(
        root=tmp_path,
        action="entry",
        preflight_id=preflight.summary["preflight_id"],
        approval_id="approval-1",
        acknowledgement=ENTRY_ACKNOWLEDGEMENT,
        execute=True,
        config=_config(submit_orders=True),
        now=now,
        pair_executor=fake,
    )

    assert result.summary["status"] == "pair_submitted"
    assert result.summary["order_submission_performed"] is True
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.summary["execution_invoked"] is True
    assert result.summary["candidate_experiment_id"] == "experiment-1"
    assert result.summary["registered_learning_id"] == "learning-1"
    assert result.summary["immutable_preflight_sha256"]
    assert len(fake.calls) == 1
    intents = fake.calls[0][0]
    assert [(intent.market, intent.side, intent.reduce_only) for intent in intents] == [
        ("BTC", "BUY", False),
        ("ETH", "SELL", False),
    ]
    assert replay.summary["status"] == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "testnet_pair_preflight_already_reserved_or_consumed" in replay.summary["blockers"]


def test_exit_preflight_uses_current_positions_and_bounded_ioc_limits(tmp_path, monkeypatch):
    active = _write_sources(tmp_path)
    _write_entry_state(active)
    positions = [
        {"position": {"coin": "BTC", "szi": "0.0002"}},
        {"position": {"coin": "ETH", "szi": "-0.004"}},
    ]

    preflight = build_testnet_pair_execution_preflight(
        root=tmp_path,
        action="exit",
        approval_id="approval-1",
        config=_config(),
        approval_validator=_approval_validator,
        info_client=_info_client(positions=positions),
    )

    assert preflight.summary["status"] == "READY_REQUIRES_EXPLICIT_EXECUTION"
    assert preflight.summary["agent_key_accessed"] is False
    intents = preflight.summary["intents"]
    assert [(row["market"], row["side"], row["reduce_only"]) for row in intents] == [
        ("BTC", "SELL", True),
        ("ETH", "BUY", True),
    ]
    assert intents[0]["limit_price"] <= 70_000.0
    assert intents[1]["limit_price"] >= 2_500.0

    fake = _FakeExecutor()
    monkeypatch.setenv(ENABLE_ENV, "true")
    result = run_testnet_pair_execution(
        root=tmp_path,
        action="exit",
        preflight_id=preflight.summary["preflight_id"],
        approval_id="approval-1",
        acknowledgement=EXIT_ACKNOWLEDGEMENT,
        execute=True,
        config=_config(submit_orders=True),
        pair_executor=fake,
    )

    assert result.summary["status"] == "pair_submitted"
    assert len(fake.calls) == 1
    assert all(intent.reduce_only for intent in fake.calls[0][0])


def test_execution_blocks_source_drift_before_executor_or_key_access(tmp_path, monkeypatch):
    active = _write_sources(tmp_path)
    preflight = build_testnet_pair_execution_preflight(
        root=tmp_path,
        action="entry",
        approval_id="approval-1",
        config=_config(),
        approval_validator=_approval_validator,
        candidate_validator=lambda root, candidate: (True, []),
        info_client=_info_client(),
    )
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps({**_approval(), "pair": "DRIFTED"}), encoding="utf-8"
    )
    fake = _FakeExecutor()
    monkeypatch.setenv(ENABLE_ENV, "true")

    result = run_testnet_pair_execution(
        root=tmp_path,
        action="entry",
        preflight_id=preflight.summary["preflight_id"],
        approval_id="approval-1",
        acknowledgement=ENTRY_ACKNOWLEDGEMENT,
        execute=True,
        config=_config(submit_orders=True),
        pair_executor=fake,
    )

    assert result.summary["status"] == "BLOCKED_BEFORE_KEY_ACCESS"
    assert any(
        blocker.startswith("testnet_pair_preflight_source_changed:")
        for blocker in result.summary["blockers"]
    )
    assert fake.calls == []
