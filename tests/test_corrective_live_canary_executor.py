from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from eth_account import Account

from quant_platform.orchestration import corrective_live_canary_executor as executor

NOW = datetime(2026, 8, 11, 12, tzinfo=UTC)


def _candidate():
    return {
        "candidate_receipt_id": "testnetcandidate_accepted",
        "receipt_sha256": "a" * 64,
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
    }


def _config(account):
    return executor.HyperliquidLiveCanaryConfig(
        network="mainnet",
        base_url=executor.HYPERLIQUID_MAINNET_URL,
        master_address="0x" + "1" * 40,
        agent_address=account.address,
        keychain_service="thewiz-hyperliquid-live-agent",
    )


def _info(master):
    def fetch(request):
        if request["type"] == "userRole":
            return {"role": "agent", "data": {"user": master}}
        if request["type"] == "meta":
            return {"universe": [{"name": "BTC"}, {"name": "ETH"}]}
        if request["type"] == "clearinghouseState":
            return {
                "marginSummary": {"accountValue": "100"},
                "withdrawable": "75",
                "assetPositions": [],
            }
        if request["type"] == "openOrders":
            return []
        raise AssertionError(f"unexpected request: {request}")

    return fetch


def test_read_only_mainnet_preflight_passes_without_submission_or_authority(
    tmp_path, monkeypatch
):
    account = Account.create()
    config = _config(account)
    monkeypatch.setattr(executor.importlib.util, "find_spec", lambda name: object())

    path = executor.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=_candidate(),
        policy_id="livecanarypolicy_test",
        config=config,
        info_client=_info(config.master_address),
        keychain_reader=lambda service, address: account.key.hex(),
    )
    receipt = json.loads(path.read_text())

    assert receipt["preflight_status"] == "PASS_READ_ONLY"
    assert receipt["exact_mainnet_endpoint"] is True
    assert receipt["agent_authorized_for_master"] is True
    assert receipt["account_flat"] is True
    assert receipt["no_open_orders"] is True
    assert receipt["order_submission_method_present"] is False
    assert receipt["private_key_persisted"] is False
    assert receipt["order_submission_performed"] is False
    assert receipt["canary_execution_authority"] is False
    assert receipt["live_trading_authorized"] is False
    assert account.key.hex().lower().removeprefix("0x") not in path.read_text().lower()

    valid, blockers = executor.validate_live_canary_executor_preflight(
        receipt=receipt,
        candidate=_candidate(),
        policy_id="livecanarypolicy_test",
        as_of=NOW + timedelta(seconds=30),
        maximum_age_seconds=60,
    )
    assert valid is True
    assert blockers == []


def test_testnet_endpoint_and_raw_key_environment_fail_before_external_checks(
    tmp_path, monkeypatch
):
    account = Account.create()
    config = executor.HyperliquidLiveCanaryConfig(
        network="testnet",
        base_url="https://api.hyperliquid-testnet.xyz",
        master_address="0x" + "1" * 40,
        agent_address=account.address,
        keychain_service="service",
    )
    monkeypatch.setenv("HYPERLIQUID_LIVE_AGENT_PRIVATE_KEY", account.key.hex())
    monkeypatch.setattr(executor.importlib.util, "find_spec", lambda name: object())

    def external_call_forbidden(request):
        raise AssertionError("configuration failure must not make external calls")

    path = executor.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=_candidate(),
        policy_id="livecanarypolicy_test",
        config=config,
        info_client=external_call_forbidden,
        keychain_reader=lambda service, address: account.key.hex(),
    )
    receipt = json.loads(path.read_text())

    assert receipt["preflight_status"] == "BLOCKED"
    assert "live_canary_executor_network_not_mainnet" in receipt["blockers"]
    assert "live_canary_executor_mainnet_endpoint_invalid" in receipt["blockers"]
    assert "live_canary_executor_raw_private_key_env_forbidden" in receipt["blockers"]
    assert receipt["order_submission_performed"] is False


def test_preflight_tamper_staleness_and_candidate_drift_fail_closed(
    tmp_path, monkeypatch
):
    account = Account.create()
    config = _config(account)
    monkeypatch.setattr(executor.importlib.util, "find_spec", lambda name: object())
    path = executor.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=_candidate(),
        policy_id="livecanarypolicy_test",
        config=config,
        info_client=_info(config.master_address),
        keychain_reader=lambda service, address: account.key.hex(),
    )
    receipt = json.loads(path.read_text())
    drifted = {**_candidate(), "pair": "SOL-USD-ETH-USD"}
    receipt["account_flat"] = False

    valid, blockers = executor.validate_live_canary_executor_preflight(
        receipt=receipt,
        candidate=drifted,
        policy_id="livecanarypolicy_test",
        as_of=NOW + timedelta(minutes=2),
        maximum_age_seconds=60,
    )

    assert valid is False
    assert "live_canary_executor_preflight_hash_invalid" in blockers
    assert "live_canary_executor_preflight_binding_mismatch" in blockers
    assert "live_canary_executor_preflight_stale_or_future" in blockers
    assert "live_canary_executor_read_only_preflight_not_passed" in blockers


def test_environment_flags_cannot_create_submission_method_or_authority(
    tmp_path, monkeypatch
):
    account = Account.create()
    config = _config(account)
    monkeypatch.setenv("HYPERLIQUID_LIVE_SUBMIT_ORDERS", "true")
    monkeypatch.setenv("LIVE_TRADING_AUTHORIZED", "true")
    monkeypatch.setattr(executor.importlib.util, "find_spec", lambda name: object())

    path = executor.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=_candidate(),
        policy_id="livecanarypolicy_test",
        config=config,
        info_client=_info(config.master_address),
        keychain_reader=lambda service, address: account.key.hex(),
    )
    receipt = json.loads(path.read_text())

    assert receipt["preflight_status"] == "PASS_READ_ONLY"
    assert receipt["order_submission_method_present"] is False
    assert receipt["canary_execution_authority"] is False
    assert receipt["live_trading_authorized"] is False
