from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from eth_account import Account

from quant_platform.hyperliquid_testnet import HyperliquidTestnetConfig
from quant_platform.orchestration import corrective_order_authority as order_gate
from quant_platform.orchestration import (
    corrective_testnet_collateral_transfer as collateral_transfer,
)
from quant_platform.orchestration.corrective_order_authority import (
    HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID,
    CorrectiveOrderAuthority,
    OrderAuthorityIdentity,
    issue_gate00g_permit,
)
from quant_platform.orchestration.corrective_testnet_collateral_transfer import (
    ENABLE_ENV,
    EXECUTION_ACKNOWLEDGEMENT,
    build_testnet_collateral_transfer_preflight,
    run_testnet_collateral_transfer,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityProfile,
    EffectKind,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_APPROVAL_VERSION,
)
from quant_platform.orchestration.venue_policy_registry import VenueLane

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
MAX_STALE_SECONDS = 301
BEFORE_MARGIN = {
    "status": "BLOCKED",
    "blockers": "testnet_usdc_requires_spot_to_perp_transfer",
    "account_value_usd": 0.0,
    "margin_used_usd": 0.0,
    "withdrawable_usd": 0.0,
    "spot_usdc_usd": 1000.0,
    "open_positions": 0,
}
AFTER_MARGIN = {
    "status": "READY",
    "blockers": "",
    "account_value_usd": 25.0,
    "margin_used_usd": 0.0,
    "withdrawable_usd": 25.0,
    "spot_usdc_usd": 975.0,
    "open_positions": 0,
}


class FakeExchange:
    def __init__(self, *, response=None, error: Exception | None = None):
        self.response = response or {"status": "ok", "response": {"type": "default"}}
        self.error = error
        self.calls: list[tuple[float, bool]] = []

    def usd_class_transfer(self, amount: float, to_perp: bool):
        self.calls.append((amount, to_perp))
        if self.error is not None:
            raise self.error
        return self.response


def _ready_context(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    wallet = Account.create()
    config = HyperliquidTestnetConfig(
        master_address="0x" + "1" * 40,
        agent_address=wallet.address,
        keychain_service="testnet-agent",
    )
    candidate = {
        "candidate_receipt_id": "testnetcandidate_fixture1234567890",
    }
    approval = {
        "approval_version": TESTNET_APPROVAL_VERSION,
        "approval_id": "approval-1",
        "payload_hash": "a" * 64,
        "collateral_transfer_policy": {
            "approved": True,
            "direction": "spot_to_perp",
            "amount_usd": 25.0,
            "one_use": True,
            "maximum_transfer_attempts": 1,
        },
    }
    (active / "testnet_candidate_receipt.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    (active / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(approval), encoding="utf-8"
    )
    return config, wallet, candidate, approval


def _ready_preflight(tmp_path, config):
    return build_testnet_collateral_transfer_preflight(
        root=tmp_path,
        config=config,
        approval_id="approval-1",
        amount_usd=25.0,
        now=NOW,
        candidate_validator=lambda _root, _candidate: (True, []),
        approval_validator=lambda _approval, _root, _config: {
            "execution_allowed": True,
            "blockers": [],
        },
        margin_reader=lambda _config: dict(BEFORE_MARGIN),
    )


class GateClock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def _collateral_gate(
    tmp_path,
    config,
    preflight_id,
    monkeypatch,
    *,
    ttl_seconds: int = 60,
):
    def policy(lane: str):
        assert lane == VenueLane.HYPERLIQUID_PERP.value
        return SimpleNamespace(
            venue="hyperliquid",
            activation_enabled=True,
            live_enabled=False,
            authenticated_access_allowed=True,
            order_submission_allowed=True,
            account_mutation_allowed=True,
            testnet_progression_allowed=True,
            allowed_testnet_adapters=(
                HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID,
            ),
        )

    monkeypatch.setattr(order_gate, "venue_policy", policy)
    clock = GateClock()
    primitive = EffectAuthority(
        root=tmp_path / "gate00g_collateral",
        secret=b"c" * 32,
        issuer_id="collateral-gate00g-test",
        profile=EffectAuthorityProfile(
            name="GATE00G_COLLATERAL_TEST_ONLY",
            allowed_effects=frozenset({EffectKind.ACCOUNT_MUTATION}),
            max_ttl_seconds=120,
        ),
        clock=clock,
    )
    identity = OrderAuthorityIdentity(
        run_id="run-collateral-gate00g",
        intended_slot_id="slot-collateral-gate00g",
        account_scope_id=str(config.master_address),
        proposal_id="approval-1",
        model_version="model-collateral-gate00g",
        formula_version="formula-collateral-gate00g",
        source_fingerprint_sha256="a" * 64,
        runtime_fingerprint_sha256="b" * 64,
        configuration_fingerprint_sha256="c" * 64,
    )
    draft = CorrectiveOrderAuthority(
        authority=primitive,
        identity=identity,
        permits=(),
    )
    spec = collateral_transfer._collateral_transfer_effect_spec(
        authority=draft,
        config=config,
        preflight_id=str(preflight_id),
        approval_id="approval-1",
        amount_usd=25.0,
    )
    permit = issue_gate00g_permit(
        authority=primitive,
        spec=spec,
        ttl_seconds=ttl_seconds,
    )
    return (
        CorrectiveOrderAuthority(
            authority=primitive,
            identity=identity,
            permits=(permit,),
        ),
        clock,
        spec,
    )


def test_collateral_preflight_is_immutable_no_key_and_no_order(tmp_path):
    config, _, _, _ = _ready_context(tmp_path)

    result = _ready_preflight(tmp_path, config)

    assert result.summary["status"] == "READY_REQUIRES_EXPLICIT_EXECUTION"
    assert result.summary["transfer_required"] is True
    assert result.summary["agent_key_accessed"] is False
    assert result.summary["transfer_attempted"] is False
    assert result.summary["order_submission_performed"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    assert result.paths["immutable_preflight"].is_file()
    assert result.summary["immutable_sha256"]


def test_collateral_preflight_blocks_unsafe_or_unapproved_input(tmp_path):
    config, _, _, _ = _ready_context(tmp_path)
    unsafe = HyperliquidTestnetConfig(
        master_address=config.master_address,
        agent_address=config.agent_address,
        keychain_service=config.keychain_service,
        base_url="https://api.hyperliquid.xyz",
    )

    result = build_testnet_collateral_transfer_preflight(
        root=tmp_path,
        config=unsafe,
        approval_id="approval-1",
        amount_usd=25.0,
        now=NOW,
        candidate_validator=lambda _root, _candidate: (False, ["candidate_invalid"]),
        approval_validator=lambda _approval, _root, _config: {
            "execution_allowed": False,
            "blockers": ["approval_invalid"],
        },
        margin_reader=lambda _config: dict(BEFORE_MARGIN),
    )

    assert result.summary["status"] == "BLOCKED"
    assert "unsafe_non_testnet_base_url" in result.summary["blockers"]
    assert "candidate_invalid" in result.summary["blockers"]
    assert "approval_invalid" in result.summary["blockers"]
    assert result.summary["transfer_attempted"] is False


def test_collateral_executor_defaults_to_status_only_before_key_access(
    tmp_path,
    monkeypatch,
):
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    config, _, _, _ = _ready_context(tmp_path)
    preflight = _ready_preflight(tmp_path, config)
    key_reads: list[str] = []

    result = run_testnet_collateral_transfer(
        root=tmp_path,
        config=config,
        preflight_id=str(preflight.summary["preflight_id"]),
        approval_id="approval-1",
        amount_usd=25.0,
        execute=False,
        now=NOW + timedelta(seconds=30),
        keychain_reader=lambda _service, account: key_reads.append(account) or "",
    )

    assert result.summary["status"] == "BLOCKED_NO_TRANSFER"
    assert "testnet_collateral_execute_flag_not_set" in result.summary["blockers"]
    assert "testnet_collateral_environment_enable_missing" in result.summary["blockers"]
    assert result.summary["agent_key_accessed"] is False
    assert result.summary["transfer_attempted"] is False
    assert key_reads == []
    assert not (tmp_path / "data" / "testnet" / "collateral_transfer_reservations").exists()


def test_collateral_executor_transfers_once_and_blocks_replay(tmp_path, monkeypatch):
    config, wallet, _, _ = _ready_context(tmp_path)
    preflight = _ready_preflight(tmp_path, config)
    monkeypatch.setenv(ENABLE_ENV, "true")
    exchange = FakeExchange()
    key_reads: list[str] = []
    gate, _, _ = _collateral_gate(
        tmp_path,
        config,
        preflight.summary["preflight_id"],
        monkeypatch,
    )

    result = run_testnet_collateral_transfer(
        root=tmp_path,
        config=config,
        preflight_id=str(preflight.summary["preflight_id"]),
        approval_id="approval-1",
        amount_usd=25.0,
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        execute=True,
        now=NOW + timedelta(seconds=30),
        keychain_reader=lambda _service, account: key_reads.append(account)
        or wallet.key.hex(),
        exchange_factory=lambda _wallet, _config: exchange,
        margin_reader=lambda _config: dict(AFTER_MARGIN),
        sleeper=lambda _seconds: None,
        order_authority=gate,
    )

    assert result.summary["status"] == "PASS_TRANSFER_RECONCILED"
    assert result.summary["transfer_attempted"] is True
    assert result.summary["transfer_reconciled"] is True
    assert result.summary["automatic_retry_performed"] is False
    assert result.summary["order_submission_performed"] is False
    assert result.summary["testnet_order_authority"] is False
    assert exchange.calls == [(25.0, True)]
    assert key_reads == [wallet.address]
    reservation = json.loads(result.paths["reservation"].read_text(encoding="utf-8"))
    assert reservation["status"] == "CONSUMED_NO_RETRY"

    replay = run_testnet_collateral_transfer(
        root=tmp_path,
        config=config,
        preflight_id=str(preflight.summary["preflight_id"]),
        approval_id="approval-1",
        amount_usd=25.0,
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        execute=True,
        now=NOW + timedelta(seconds=60),
        keychain_reader=lambda _service, account: key_reads.append(account)
        or wallet.key.hex(),
        exchange_factory=lambda _wallet, _config: exchange,
        margin_reader=lambda _config: dict(AFTER_MARGIN),
        sleeper=lambda _seconds: None,
        order_authority=gate,
    )
    assert replay.summary["status"] == "BLOCKED_NO_TRANSFER"
    assert (
        "testnet_collateral_transfer_approval_already_reserved_or_consumed"
        in replay.summary["blockers"]
    )
    assert exchange.calls == [(25.0, True)]
    assert key_reads == [wallet.address]


def test_collateral_executor_reconciles_unknown_response_without_retry(
    tmp_path,
    monkeypatch,
):
    config, wallet, _, _ = _ready_context(tmp_path)
    preflight = _ready_preflight(tmp_path, config)
    monkeypatch.setenv(ENABLE_ENV, "true")
    exchange = FakeExchange(error=TimeoutError("unknown response"))
    gate, _, _ = _collateral_gate(
        tmp_path,
        config,
        preflight.summary["preflight_id"],
        monkeypatch,
    )

    result = run_testnet_collateral_transfer(
        root=tmp_path,
        config=config,
        preflight_id=str(preflight.summary["preflight_id"]),
        approval_id="approval-1",
        amount_usd=25.0,
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        execute=True,
        now=NOW + timedelta(seconds=30),
        keychain_reader=lambda _service, _account: wallet.key.hex(),
        exchange_factory=lambda _wallet, _config: exchange,
        margin_reader=lambda _config: dict(AFTER_MARGIN),
        sleeper=lambda _seconds: None,
        order_authority=gate,
    )

    assert result.summary["status"] == "PASS_RECONCILED_AFTER_UNKNOWN_RESPONSE"
    assert result.summary["transfer_response_ok"] is False
    assert result.summary["transfer_reconciled"] is True
    assert result.summary["automatic_retry_performed"] is False
    assert exchange.calls == [(25.0, True)]


def test_collateral_gate00g_missing_stale_replayed_and_tampered_authority(
    tmp_path,
    monkeypatch,
):
    scenarios = ("missing", "stale", "replayed", "tampered")
    expected = {
        "missing": "authority_missing",
        "stale": "expired",
        "replayed": "consumed",
        "tampered": "signature_invalid",
    }
    for index, scenario in enumerate(scenarios):
        case_root = tmp_path / f"case_{index}"
        config, wallet, _, _ = _ready_context(case_root)
        preflight = _ready_preflight(case_root, config)
        monkeypatch.setenv(ENABLE_ENV, "true")
        gate, clock, spec = _collateral_gate(
            case_root,
            config,
            preflight.summary["preflight_id"],
            monkeypatch,
            ttl_seconds=1 if scenario == "stale" else 60,
        )
        if scenario == "missing":
            selected_gate = None
        elif scenario == "stale":
            clock.now += timedelta(seconds=2)
            selected_gate = gate
        elif scenario == "replayed":
            gate.consume(spec)
            selected_gate = gate
        else:
            tampered = gate.permits[0].model_copy(update={"signature": "0" * 64})
            selected_gate = CorrectiveOrderAuthority(
                authority=gate.authority,
                identity=gate.identity,
                permits=(tampered,),
            )
        key_reads: list[str] = []
        exchange = FakeExchange()

        def read_key(_service, account, *, reads=key_reads, owner=wallet):
            reads.append(account)
            return owner.key.hex()

        def build_exchange(_wallet, _config, *, instance=exchange):
            return instance

        result = run_testnet_collateral_transfer(
            root=case_root,
            config=config,
            preflight_id=str(preflight.summary["preflight_id"]),
            approval_id="approval-1",
            amount_usd=25.0,
            acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
            execute=True,
            now=NOW + timedelta(seconds=30),
            keychain_reader=read_key,
            exchange_factory=build_exchange,
            margin_reader=lambda _config: dict(AFTER_MARGIN),
            sleeper=lambda _seconds: None,
            order_authority=selected_gate,
        )

        assert result.summary["status"] == "BLOCKED_NO_TRANSFER"
        assert expected[scenario] in ";".join(result.summary["blockers"]).lower()
        assert key_reads == []
        assert exchange.calls == []
        assert not (
            case_root / "data" / "testnet" / "collateral_transfer_reservations"
        ).exists()


def test_collateral_executor_blocks_stale_or_changed_evidence_before_key(
    tmp_path,
    monkeypatch,
):
    config, wallet, _, approval = _ready_context(tmp_path)
    preflight = _ready_preflight(tmp_path, config)
    monkeypatch.setenv(ENABLE_ENV, "true")
    approval["payload_hash"] = "b" * 64
    (tmp_path / "reports" / "active" / "hyperliquid_testnet_smoke_approval.json").write_text(
        json.dumps(approval), encoding="utf-8"
    )
    key_reads: list[str] = []

    result = run_testnet_collateral_transfer(
        root=tmp_path,
        config=config,
        preflight_id=str(preflight.summary["preflight_id"]),
        approval_id="approval-1",
        amount_usd=25.0,
        acknowledgement=EXECUTION_ACKNOWLEDGEMENT,
        execute=True,
        now=NOW + timedelta(seconds=MAX_STALE_SECONDS),
        keychain_reader=lambda _service, account: key_reads.append(account)
        or wallet.key.hex(),
    )

    assert result.summary["status"] == "BLOCKED_NO_TRANSFER"
    assert "testnet_collateral_preflight_stale_or_future" in result.summary["blockers"]
    assert "testnet_collateral_approval_binding_changed" in result.summary["blockers"]
    assert key_reads == []
