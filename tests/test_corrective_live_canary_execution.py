from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from eth_account import Account
from eth_account.messages import encode_defunct

from quant_platform.orchestration import corrective_live_canary_execution as live_execution
from quant_platform.orchestration import corrective_live_canary_executor as preflight
from quant_platform.orchestration.corrective_live_canary import (
    _canonical_json,
    _payload_hash,
    _policy_id,
    build_live_canary_control_plane,
)
from quant_platform.orchestration.corrective_live_canary_execution import (
    EXECUTOR_RESERVATION_SCHEMA_VERSION,
    LIVE_ACKNOWLEDGEMENT,
    LIVE_ENABLE_ENV,
    live_canary_executor_contract,
    run_live_canary_executor,
)
from quant_platform.orchestration.corrective_release_gates import (
    CANDIDATE_SCHEMA_VERSION,
)
from tests.candidate_queue_support import seal_candidate_with_valid_queue
from tests.test_corrective_live_canary import (
    NOW,
    _build,
    _install_mock_stage6_release,
    _policy,
    _write_json,
    _write_parity,
)


class SimulatedMainnet:
    def __init__(self, master_address: str, *, partial_entry: bool = False) -> None:
        self.master_address = master_address
        self.partial_entry = partial_entry
        self.positions = {"BTC": 0.0, "ETH": 0.0}
        self.fills: list[dict[str, object]] = []
        self.bulk_requests: list[list[dict[str, object]]] = []
        self.leverage_updates: list[tuple[int, str, bool]] = []
        self.market_closes: list[str] = []
        self.cancels: list[list[dict[str, object]]] = []
        self.next_order_id = 100

    def info(self, request):
        request_type = request["type"]
        if request_type == "userRole":
            return {"role": "agent", "data": {"user": self.master_address}}
        if request_type == "meta":
            return {
                "universe": [
                    {"name": "BTC", "szDecimals": 5, "maxLeverage": 40},
                    {"name": "ETH", "szDecimals": 4, "maxLeverage": 25},
                ]
            }
        if request_type == "allMids":
            return {"BTC": "60000", "ETH": "3000"}
        if request_type == "clearinghouseState":
            return {
                "marginSummary": {"accountValue": "100"},
                "withdrawable": "75",
                "assetPositions": [
                    {"position": {"coin": coin, "szi": str(size)}}
                    for coin, size in self.positions.items()
                    if abs(size) > 1e-12
                ],
            }
        if request_type == "openOrders":
            return []
        if request_type == "userFillsByTime":
            return [
                row
                for row in self.fills
                if int(request["startTime"]) <= int(row["time"]) <= int(request["endTime"])
            ]
        if request_type == "userFunding":
            return []
        raise AssertionError(f"unexpected info request: {request}")

    def exchange(self, wallet, config):
        outer = self

        class Exchange:
            def update_leverage(self, leverage, coin, is_cross):
                outer.leverage_updates.append((leverage, coin, is_cross))
                return {"status": "ok"}

            def bulk_orders(self, requests):
                outer.bulk_requests.append(requests)
                is_exit = all(bool(row["reduce_only"]) for row in requests)
                selected = requests if is_exit or not outer.partial_entry else requests[:1]
                timestamp_ms = int(time.time() * 1000) + 2
                for row in selected:
                    coin = str(row["coin"])
                    size = float(row["sz"])
                    side = "B" if bool(row["is_buy"]) else "A"
                    if is_exit:
                        outer.positions[coin] = 0.0
                    else:
                        outer.positions[coin] = size if side == "B" else -size
                    outer.next_order_id += 1
                    outer.fills.append(
                        {
                            "coin": coin,
                            "side": side,
                            "sz": str(size),
                            "px": str(row["limit_px"]),
                            "fee": "0.01",
                            "feeToken": "USDC",
                            "time": timestamp_ms,
                            "oid": outer.next_order_id,
                            "hash": f"0x{outer.next_order_id:064x}",
                        }
                    )
                time.sleep(0.004)
                return {
                    "status": "ok",
                    "response": {
                        "type": "order",
                        "data": {
                            "statuses": [
                                {"filled": {"oid": index + 1}}
                                for index, _ in enumerate(selected)
                            ]
                        },
                    },
                }

            def bulk_cancel(self, requests):
                outer.cancels.append(requests)
                return {"status": "ok"}

            def market_close(self, coin, sz=None, slippage=None):
                outer.market_closes.append(coin)
                outer.positions[coin] = 0.0
                return {"status": "ok"}

        return Exchange()


def _executable_candidate(root: Path) -> dict[str, object]:
    receipt = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "generated_at_utc": NOW.isoformat(),
        "candidate_status": "READY_FOR_NO_ORDER_PREFLIGHT",
        "candidate_experiment_id": "experiment-1",
        "candidate_leverage": 1.0,
        "pair_group_key": "BTC__ETH",
        "pair": "BTC-USD-ETH-USD",
        "asset_x": "BTC",
        "asset_y": "ETH",
        "timeframe": "daily",
        "exact_mode": "OU Optimal",
        "orientation": "original",
        "cost_model_id": "cost-1",
        "model_training_dataset_id": "dataset-1",
        "model_artifact_sha256": "a" * 64,
        "registered_learning_id": "learning-1",
        "registered_learning_receipt_sha256": "b" * 64,
        "registered_execution_id": "execution-1",
        "survivor_receipt_id": "survivor-1",
        "cadence_pass": True,
        "model_oos_authority_ready": True,
        "strict_cost_model_current": True,
        "hyperliquid_markets_current": True,
        "blockers": [],
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "source_artifact_hashes": {},
        "evidence_path": "",
    }
    candidate = seal_candidate_with_valid_queue(root=root, receipt=receipt)
    _write_json(
        root / "reports" / "active" / "testnet_candidate_receipt.json",
        candidate,
    )
    return candidate


def _authorized_fixture(tmp_path, monkeypatch):
    signer = Account.create()
    policy = _policy(signer.address)
    candidate = _executable_candidate(tmp_path)
    _install_mock_stage6_release(monkeypatch, tmp_path, candidate)
    _write_json(tmp_path / "config" / "live_canary_policy.json", policy)
    parity = _write_parity(tmp_path, policy, candidate)
    first = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    approval_path = first["approval"]
    approval = json.loads(approval_path.read_text())
    approval.update(
        {
            "approved": True,
            "issued_at_utc": NOW.isoformat(),
            "expires_at_utc": (NOW + timedelta(minutes=10)).isoformat(),
            "authorized_signer_address": signer.address,
            "parity_receipt_id": parity["parity_receipt_id"],
            "legs": [
                {
                    "market": "BTC",
                    "side": "BUY",
                    "size": 0.0002,
                    "limit_price": 60_006.0,
                },
                {
                    "market": "ETH",
                    "side": "SELL",
                    "size": 0.004,
                    "limit_price": 2_999.7,
                },
            ],
            "maximum_total_notional_usd": 25.0,
            "maximum_slippage_bps": 10.0,
            "nonce": "one-use-executor-test",
        }
    )
    core = {
        key: value
        for key, value in approval.items()
        if key not in {"approval_id", "wallet_signature"}
    }
    approval["approval_id"] = "livecanaryapproval_" + _payload_hash(core)[:20]
    signed = {key: value for key, value in approval.items() if key != "wallet_signature"}
    approval["wallet_signature"] = Account.sign_message(
        encode_defunct(text=_canonical_json(signed)), signer.key
    ).signature.hex()
    _write_json(approval_path, approval)

    config = preflight.HyperliquidLiveCanaryConfig(
        network="mainnet",
        base_url=preflight.HYPERLIQUID_MAINNET_URL,
        master_address="0x" + "1" * 40,
        agent_address=signer.address,
        keychain_service="thewiz-live-canary-test",
    )
    market = SimulatedMainnet(str(config.master_address))
    preflight.build_live_canary_executor_preflight(
        root=tmp_path,
        now=NOW,
        candidate=candidate,
        policy_id=_policy_id(policy),
        config=config,
        info_client=market.info,
        keychain_reader=lambda service, address: signer.key.hex(),
    )
    ready = _build(
        tmp_path,
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        candidate=candidate,
    )
    authorization = json.loads(ready["authorization"].read_text())
    assert authorization["authorization_status"] == "AUTHORIZED_FOR_ONE_LIVE_CANARY"
    return {
        "signer": signer,
        "config": config,
        "market": market,
        "candidate": candidate,
        "approval": approval,
        "authorization_path": ready["authorization"],
        "authorization_sha256": sha256(ready["authorization"].read_bytes()).hexdigest(),
    }


def _execute(tmp_path, fixture, monkeypatch, *, market=None, key_reader=None):
    monkeypatch.setenv(LIVE_ENABLE_ENV, "true")
    monkeypatch.setattr(live_execution, "_utc_now", lambda: NOW)
    selected_market = market or fixture["market"]
    return run_live_canary_executor(
        root=tmp_path,
        execute=True,
        authorization_sha256=fixture["authorization_sha256"],
        approval_id=fixture["approval"]["approval_id"],
        acknowledgement=LIVE_ACKNOWLEDGEMENT,
        config=fixture["config"],
        info_client=selected_market.info,
        keychain_reader=key_reader
        or (lambda service, address: fixture["signer"].key.hex()),
        exchange_factory=selected_market.exchange,
        waiter=lambda seconds: None,
    )


def test_executor_contract_rotates_when_any_internal_module_changes(
    tmp_path, monkeypatch
):
    relative_paths = live_execution.EXECUTOR_IMPLEMENTATION_RELATIVE_PATHS
    for index, relative_path in enumerate(relative_paths):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"module-{index}\n", encoding="utf-8")
    monkeypatch.setattr(live_execution, "ROOT", Path(tmp_path))

    before = live_canary_executor_contract()
    changed_path = relative_paths[-1]
    (tmp_path / changed_path).write_text("module-changed\n", encoding="utf-8")
    after = live_canary_executor_contract()

    assert set(before["executor_implementation_bundle"]) == {
        path.as_posix() for path in relative_paths
    }
    assert before["executor_source_sha256"] == after["executor_source_sha256"]
    assert (
        before["executor_implementation_bundle_sha256"]
        != after["executor_implementation_bundle_sha256"]
    )
    assert before["executor_contract_id"] != after["executor_contract_id"]


def test_default_path_is_status_only_and_never_reads_key_or_network(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("status-only path must not touch network or keychain")

    result = run_live_canary_executor(
        root=tmp_path,
        now=NOW,
        config=fixture["config"],
        info_client=forbidden,
        keychain_reader=forbidden,
        exchange_factory=forbidden,
    )

    assert result.status == "NO_SUBMISSION_STATUS_ONLY"
    assert "live_canary_execute_flag_not_set" in result.blockers
    assert result.order_submission_performed is False
    assert result.live_trading_authorized is False
    assert not (
        tmp_path / "data" / "live" / "canary_authorization_reservations"
    ).exists()


def test_tampered_preflight_blocks_before_key_access_and_reservation(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    preflight_path = (
        tmp_path
        / "reports"
        / "active"
        / "hyperliquid_live_canary_executor_preflight.json"
    )
    receipt = json.loads(preflight_path.read_text())
    receipt["account_flat"] = False
    _write_json(preflight_path, receipt)
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(tmp_path, fixture, monkeypatch, key_reader=key_reader)

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "live_canary_executor_preflight_hash_invalid" in result.blockers
    assert touched is False
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


def test_rehashed_stale_implementation_bundle_blocks_before_key_access(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    preflight_path = (
        tmp_path
        / "reports"
        / "active"
        / "hyperliquid_live_canary_executor_preflight.json"
    )
    receipt = json.loads(preflight_path.read_text())
    receipt["executor_implementation_bundle_sha256"] = "0" * 64
    receipt["receipt_sha256"] = _payload_hash(receipt)
    _write_json(preflight_path, receipt)
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(tmp_path, fixture, monkeypatch, key_reader=key_reader)

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert (
        "live_canary_executor_preflight_contract_binding_mismatch"
        in result.blockers
    )
    assert touched is False


def test_candidate_source_drift_blocks_before_key_access_and_reservation(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    source = tmp_path / "data" / "research" / "fixture_stage5_receipt.json"
    source.write_text("changed after live authorization\n", encoding="utf-8")
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(
        tmp_path,
        fixture,
        monkeypatch,
        key_reader=key_reader,
    )

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "live_canary_current_candidate_evidence_invalid" in result.blockers
    assert (
        "testnet_candidate_queue_source_artifact_changed:"
        "data/research/fixture_stage5_receipt.json"
    ) in result.blockers
    assert touched is False
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


def test_parity_raw_source_drift_blocks_before_key_access_and_reservation(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    evidence = json.loads(
        (
            tmp_path
            / "reports"
            / "active"
            / "testnet_live_input_parity_evidence.json"
        ).read_text()
    )
    snapshot_path = tmp_path / evidence["inputs"][0]["testnet_artifact_path"]
    snapshot = json.loads(snapshot_path.read_text())
    source_descriptor = next(iter(snapshot["source_artifacts"].values()))
    source_path = tmp_path / source_descriptor["path"]
    source_path.write_text("changed after live authorization\n", encoding="utf-8")
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(
        tmp_path,
        fixture,
        monkeypatch,
        key_reader=key_reader,
    )

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "live_canary_current_input_parity_evidence_invalid" in result.blockers
    assert touched is False
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


def test_sample_evidence_drift_blocks_before_key_access_and_reservation(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    sample_path = (
        tmp_path / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
    )
    sample_path.write_text("check,status\nall,BLOCKED\n", encoding="utf-8")
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(
        tmp_path,
        fixture,
        monkeypatch,
        key_reader=key_reader,
    )

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert (
        "live_canary_testnet_sample_evidence_invalid_or_changed"
        in result.blockers
    )
    assert touched is False
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "expected_blocker"),
    (
        (
            "reports/supreme_team/testnet_evidence_checkpoint.json",
            "checkpoint_status",
            "BLOCKED",
            "live_canary_supreme_team_evidence_invalid_or_changed",
        ),
        (
            "data/live/active_canary_policy.json",
            "live_trading_authorized",
            True,
            "live_canary_registered_policy_evidence_invalid",
        ),
    ),
)
def test_current_control_artifact_drift_blocks_before_key_access(
    tmp_path,
    monkeypatch,
    relative_path,
    field,
    value,
    expected_blocker,
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    path = tmp_path / relative_path
    payload = json.loads(path.read_text())
    payload[field] = value
    if "receipt_sha256" in payload:
        payload["receipt_sha256"] = _payload_hash(payload)
    _write_json(path, payload)
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(
        tmp_path,
        fixture,
        monkeypatch,
        key_reader=key_reader,
    )

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert expected_blocker in result.blockers
    assert touched is False


def test_execute_rejects_caller_supplied_time_override_before_network_or_key(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv(LIVE_ENABLE_ENV, "true")

    def forbidden(*args, **kwargs):
        raise AssertionError("time override must block before network or key access")

    result = run_live_canary_executor(
        root=tmp_path,
        execute=True,
        authorization_sha256=fixture["authorization_sha256"],
        approval_id=fixture["approval"]["approval_id"],
        acknowledgement=LIVE_ACKNOWLEDGEMENT,
        now=NOW,
        config=fixture["config"],
        info_client=forbidden,
        keychain_reader=forbidden,
        exchange_factory=forbidden,
    )

    assert result.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "live_canary_execution_time_override_forbidden" in result.blockers
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


def test_environment_acknowledgement_and_exact_hash_each_fail_before_key_access(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(live_execution, "_utc_now", lambda: NOW)

    def forbidden(*args, **kwargs):
        raise AssertionError("failed explicit gates must not touch network or keychain")

    common = {
        "root": tmp_path,
        "execute": True,
        "approval_id": fixture["approval"]["approval_id"],
        "config": fixture["config"],
        "info_client": forbidden,
        "keychain_reader": forbidden,
        "exchange_factory": forbidden,
    }
    monkeypatch.delenv(LIVE_ENABLE_ENV, raising=False)
    missing_environment = run_live_canary_executor(
        **common,
        authorization_sha256=fixture["authorization_sha256"],
        acknowledgement=LIVE_ACKNOWLEDGEMENT,
    )
    assert "live_canary_environment_enable_missing" in missing_environment.blockers

    monkeypatch.setenv(LIVE_ENABLE_ENV, "true")
    wrong_acknowledgement = run_live_canary_executor(
        **common,
        authorization_sha256=fixture["authorization_sha256"],
        acknowledgement="wrong",
    )
    assert "live_canary_exact_acknowledgement_missing" in wrong_acknowledgement.blockers

    wrong_hash = run_live_canary_executor(
        **common,
        authorization_sha256="0" * 64,
        acknowledgement=LIVE_ACKNOWLEDGEMENT,
    )
    assert (
        "live_canary_authorization_hash_acknowledgement_mismatch"
        in wrong_hash.blockers
    )


def test_existing_process_lock_fails_closed_without_reservation_or_key_access(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    lock = tmp_path / "data" / "live" / ".live_canary_executor.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("existing process\n", encoding="utf-8")
    touched = False

    def key_reader(service, address):
        nonlocal touched
        touched = True
        return fixture["signer"].key.hex()

    result = _execute(tmp_path, fixture, monkeypatch, key_reader=key_reader)

    assert result.status == "BLOCKED_EXECUTOR_LOCK_PRESENT"
    assert touched is False
    assert not (
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{fixture['approval']['approval_id']}.json"
    ).exists()


def test_exact_ioc_entry_reduce_only_exit_and_replay_block(tmp_path, monkeypatch):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    result = _execute(tmp_path, fixture, monkeypatch)

    assert result.status == "EXECUTED_ONE_CANARY_PENDING_SUPREME_REVIEW"
    assert result.order_submission_performed is True
    assert result.reconciled_flat is True
    assert len(fixture["market"].bulk_requests) == 2
    entry, exit_requests = fixture["market"].bulk_requests
    assert all(row["order_type"] == {"limit": {"tif": "Ioc"}} for row in entry)
    assert all(row["reduce_only"] is False for row in entry)
    assert all(row["order_type"] == {"limit": {"tif": "Ioc"}} for row in exit_requests)
    assert all(row["reduce_only"] is True for row in exit_requests)
    assert fixture["market"].leverage_updates == [
        (1, "BTC", True),
        (1, "ETH", True),
    ]
    execution = json.loads(
        (tmp_path / "reports" / "active" / "live_canary_execution_receipt.json").read_text()
    )
    assert len(execution["entry_fills"]) == 2
    assert len(execution["exit_fills"]) == 2
    assert execution["entry_retry_attempted"] is False
    assert execution["reconciled_flat"] is True
    assert execution["live_trading_authorized"] is False

    refreshed = build_live_canary_control_plane(
        root=tmp_path,
        now=datetime.now(UTC) + timedelta(seconds=1),
        candidate=fixture["candidate"],
        candidate_valid=True,
        sample_status="PASS",
        supreme_status="PASS",
        sample_evidence_path=(
            tmp_path / "reports" / "active" / "realized_testnet_sample_sufficiency.csv"
        ),
        supreme_evidence_path=(
            tmp_path / "reports" / "supreme_team" / "testnet_evidence_checkpoint.json"
        ),
    )
    refreshed_authorization = json.loads(refreshed["authorization"].read_text())
    refreshed_outcome = json.loads(refreshed["outcome"].read_text())
    assert refreshed_authorization["authorization_status"] == "NOT_AUTHORIZED"
    assert refreshed_authorization["canary_execution_authority"] is False
    assert refreshed_outcome["canary_status"] == "BLOCKED_POST_CANARY_REVIEW"
    assert not any(
        blocker.startswith("canary_execution_without_authority")
        for blocker in refreshed_outcome["execution_blockers"]
    )

    touched = False

    def forbidden_key(service, address):
        nonlocal touched
        touched = True
        raise AssertionError("replay must block before key access")

    replay = _execute(
        tmp_path,
        fixture,
        monkeypatch,
        key_reader=forbidden_key,
    )
    assert replay.status == "BLOCKED_BEFORE_KEY_ACCESS"
    assert "live_canary_approval_already_reserved_or_used" in replay.blockers
    assert touched is False


def test_partial_entry_never_retries_and_recovers_flat(tmp_path, monkeypatch):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    partial = SimulatedMainnet(
        str(fixture["config"].master_address), partial_entry=True
    )
    result = _execute(tmp_path, fixture, monkeypatch, market=partial)

    assert result.status == "INCIDENT_RECOVERED_FLAT"
    assert result.order_submission_performed is True
    assert result.reconciled_flat is True
    assert len(partial.bulk_requests) == 1
    assert partial.market_closes == ["BTC"]
    assert partial.positions == {"BTC": 0.0, "ETH": 0.0}
    assert not (
        tmp_path / "reports" / "active" / "live_canary_execution_receipt.json"
    ).exists()
    incident = json.loads(
        (tmp_path / "reports" / "active" / "live_canary_executor_incident.json").read_text()
    )
    state = json.loads(
        (tmp_path / "reports" / "active" / "live_canary_executor_state.json").read_text()
    )
    assert "entry_not_exactly_filled_no_retry" in incident["incident"]
    assert incident["reconciled_flat"] is True
    assert state["entry_retry_allowed"] is False
    assert state["entry_submit_attempted"] is True


def test_recovery_only_uses_frozen_reservation_after_active_authority_expires(
    tmp_path, monkeypatch
):
    fixture = _authorized_fixture(tmp_path, monkeypatch)
    authorization = json.loads(fixture["authorization_path"].read_text())
    approval_id = fixture["approval"]["approval_id"]
    reservation = {
        "schema_version": EXECUTOR_RESERVATION_SCHEMA_VERSION,
        "reserved_at_utc": NOW.isoformat(),
        "approval_id": approval_id,
        "authorization_receipt_sha256": fixture["authorization_sha256"],
        "authorization_id": authorization["authorization_id"],
        "authorization_receipt_path": authorization[
            "immutable_authorization_path"
        ],
        "candidate_receipt_id": fixture["candidate"]["candidate_receipt_id"],
        "executor_contract_id": live_canary_executor_contract()[
            "executor_contract_id"
        ],
        "executor_implementation_bundle_sha256": live_canary_executor_contract()[
            "executor_implementation_bundle_sha256"
        ],
        "master_address": fixture["config"].master_address,
        "agent_address": fixture["config"].agent_address,
        "entry_retry_allowed": False,
        "authorization_reusable": False,
        "live_trading_authorized": False,
    }
    reservation["reservation_id"] = (
        "livecanaryreservation_" + _payload_hash(reservation)[:20]
    )
    reservation["receipt_sha256"] = _payload_hash(reservation)
    _write_json(
        tmp_path
        / "data"
        / "live"
        / "canary_authorization_reservations"
        / f"{approval_id}.json",
        reservation,
    )
    _write_json(
        fixture["authorization_path"],
        {
            "schema_version": "thewiz.live_canary_authorization.v5",
            "authorization_status": "NOT_AUTHORIZED",
            "canary_execution_authority": False,
            "live_trading_authorized": False,
        },
    )
    recovery_market = SimulatedMainnet(str(fixture["config"].master_address))
    recovery_market.positions["BTC"] = 0.0002
    monkeypatch.setenv(LIVE_ENABLE_ENV, "true")

    result = run_live_canary_executor(
        root=tmp_path,
        execute=True,
        recover_only=True,
        authorization_sha256=fixture["authorization_sha256"],
        approval_id=approval_id,
        acknowledgement=LIVE_ACKNOWLEDGEMENT,
        now=NOW + timedelta(days=1),
        config=fixture["config"],
        info_client=recovery_market.info,
        keychain_reader=lambda service, address: fixture["signer"].key.hex(),
        exchange_factory=recovery_market.exchange,
        waiter=lambda seconds: None,
    )

    assert result.status == "INCIDENT_RECOVERED_FLAT"
    assert result.reconciled_flat is True
    assert recovery_market.market_closes == ["BTC"]
    assert recovery_market.positions == {"BTC": 0.0, "ETH": 0.0}
    assert len(recovery_market.bulk_requests) == 0
