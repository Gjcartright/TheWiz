from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import quant_platform.execution as execution_module
from quant_platform.binance_testnet import (
    BINANCE_SPOT_TESTNET_URL,
    BinanceSpotTestnetOrderAdapter,
    BinanceTestnetConfig,
)
from quant_platform.dydx_sdk_order_adapter import DydxSdkOrderAdapter
from quant_platform.execution import (
    DydxNetworkConfig,
    ExecutionMode,
    OrderIntent,
    PaperVenueExecution,
    SpreadOrderPlan,
    build_venue_order_client_adapter,
    submit_paper_plan,
)
from quant_platform.hyperliquid_testnet import (
    HYPERLIQUID_TESTNET_URL,
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairAdapter,
    HyperliquidTestnetPairExecutor,
)
from quant_platform.orchestration import (
    corrective_live_canary_execution as live_canary_execution,
)
from quant_platform.orchestration import corrective_order_authority as order_gate
from quant_platform.orchestration import (
    corrective_testnet_collateral_transfer as collateral_transfer,
)
from quant_platform.orchestration.corrective_order_authority import (
    BINANCE_SPOT_TESTNET_ADAPTER_ID,
    BINANCE_USDM_TESTNET_ADAPTER_ID,
    DYDX_TESTNET_ADAPTER_ID,
    HYPERLIQUID_LIVE_CANARY_ADAPTER_ID,
    HYPERLIQUID_TESTNET_ADAPTER_ID,
    HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID,
    CorrectiveOrderAuthority,
    OrderAuthorityIdentity,
    issue_gate00g_permit,
    require_consumed_authorization,
)
from quant_platform.orchestration.effect_authority import (
    PHASE00_DENY_ALL_PROFILE,
    EffectAuthority,
    EffectAuthorityError,
    EffectAuthorityProfile,
    EffectKind,
)
from quant_platform.orchestration.venue_policy_registry import VenueLane

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
ACCOUNT = "0x" + "1" * 40
AGENT = "0x" + "2" * 40
PROPOSAL = "proposal-gate00g-1"


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


class FakeResponse:
    ok = True
    status_code = 200

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {"orderId": 17, "origQty": "1", "avgPrice": "25"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.post_calls: list[tuple[tuple, dict]] = []

    def post(self, *args, **kwargs) -> FakeResponse:
        self.post_calls.append((args, kwargs))
        return FakeResponse()


class FakeExchange:
    def __init__(self) -> None:
        self.leverage_calls: list[tuple[tuple, dict]] = []
        self.bulk_order_calls: list[list[dict]] = []
        self.bulk_cancel_calls: list[list[dict]] = []
        self.market_close_calls: list[tuple[tuple, dict]] = []

    def update_leverage(self, *args, **kwargs) -> None:
        self.leverage_calls.append((args, kwargs))

    def bulk_orders(self, requests: list[dict]) -> dict:
        self.bulk_order_calls.append(requests)
        return {}

    def bulk_cancel(self, requests: list[dict]) -> None:
        self.bulk_cancel_calls.append(requests)

    def market_close(self, *args, **kwargs) -> None:
        self.market_close_calls.append((args, kwargs))


def _identity(
    *,
    account: str = ACCOUNT,
    proposal: str = PROPOSAL,
    run_id: str = "run-gate00g-1",
) -> OrderAuthorityIdentity:
    return OrderAuthorityIdentity(
        run_id=run_id,
        intended_slot_id="slot-gate00g-1",
        account_scope_id=account,
        proposal_id=proposal,
        model_version="model-gate00g-1",
        formula_version="formula-gate00g-1",
        source_fingerprint_sha256=HASH_A,
        runtime_fingerprint_sha256=HASH_B,
        configuration_fingerprint_sha256=HASH_C,
    )


def _effect_authority(
    tmp_path: Path,
    clock: Clock,
    *,
    profile: EffectAuthorityProfile | None = None,
) -> EffectAuthority:
    return EffectAuthority(
        root=tmp_path,
        secret=b"g" * 32,
        issuer_id="gate00g-test-supervisor",
        profile=profile
        or EffectAuthorityProfile(
            name="GATE00G_TEST_ONLY",
            allowed_effects=frozenset(
                {EffectKind.ORDER_SUBMISSION, EffectKind.ACCOUNT_MUTATION}
            ),
            max_ttl_seconds=120,
        ),
        clock=clock,
    )


def _enable_test_policies(monkeypatch: pytest.MonkeyPatch) -> None:
    lanes = {
        VenueLane.HYPERLIQUID_PERP.value: (
            "hyperliquid",
            (
                HYPERLIQUID_TESTNET_ADAPTER_ID,
                HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID,
            ),
        ),
        VenueLane.DYDX_PERP.value: ("dydx", (DYDX_TESTNET_ADAPTER_ID,)),
        VenueLane.BINANCE_SPOT.value: (
            "binance",
            (BINANCE_SPOT_TESTNET_ADAPTER_ID,),
        ),
        VenueLane.BINANCE_USDM_PERP.value: (
            "binance",
            (BINANCE_USDM_TESTNET_ADAPTER_ID,),
        ),
    }

    def resolver(lane: str):
        normalized = str(lane)
        venue, adapters = lanes[normalized]
        return SimpleNamespace(
            lane=normalized,
            venue=venue,
            activation_enabled=True,
            live_enabled=False,
            authenticated_access_allowed=True,
            order_submission_allowed=True,
            account_mutation_allowed=True,
            testnet_progression_allowed=True,
            allowed_testnet_adapters=adapters,
        )

    monkeypatch.setattr(order_gate, "venue_policy", resolver)


def _binance_spec(gate: CorrectiveOrderAuthority, **overrides):
    values = {
        "effect_kind": EffectKind.ORDER_SUBMISSION,
        "environment": "testnet",
        "adapter_id": BINANCE_SPOT_TESTNET_ADAPTER_ID,
        "target": f"{BINANCE_SPOT_TESTNET_URL}/api/v3/order",
        "operation": "place_order",
        "venue_id": "binance",
        "product_lane_id": VenueLane.BINANCE_SPOT.value,
        "account_scope_id": gate.identity.account_scope_id,
        "instrument_id": "ETHUSDT",
        "side": "buy",
        "size": "1",
        "notional": "25",
        "leverage": "1",
        "reduce_only": False,
        "proposal_id": gate.identity.proposal_id,
    }
    values.update(overrides)
    return gate.spec(**values)


def _gate_with_permits(
    authority: EffectAuthority,
    identity: OrderAuthorityIdentity,
    specs: tuple,
    *,
    ttl_seconds: int = 60,
) -> CorrectiveOrderAuthority:
    permits = tuple(
        issue_gate00g_permit(
            authority=authority,
            spec=spec,
            ttl_seconds=ttl_seconds,
        )
        for spec in specs
    )
    return CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=permits,
    )


def test_current_phase00_policy_and_profile_cannot_mint_order_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    authority = _effect_authority(tmp_path / "policy", clock)
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=_identity(),
        permits=(),
    )
    spec = _binance_spec(draft)
    with pytest.raises(EffectAuthorityError, match="venue_activation_disabled"):
        issue_gate00g_permit(authority=authority, spec=spec)

    new_path_specs = (
        draft.spec(
            effect_kind=EffectKind.ORDER_SUBMISSION,
            environment="live",
            adapter_id=HYPERLIQUID_LIVE_CANARY_ADAPTER_ID,
            target="https://api.hyperliquid.xyz/exchange",
            operation="bulk_orders",
            venue_id="hyperliquid",
            product_lane_id=VenueLane.HYPERLIQUID_PERP.value,
            account_scope_id=draft.identity.account_scope_id,
            instrument_id="BTC",
            side="buy",
            size="0.001",
            notional="60",
            leverage="1",
            proposal_id=draft.identity.proposal_id,
        ),
        draft.spec(
            effect_kind=EffectKind.ACCOUNT_MUTATION,
            environment="testnet",
            adapter_id=HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID,
            target=f"{HYPERLIQUID_TESTNET_URL}/exchange",
            operation="usd_class_transfer",
            venue_id="hyperliquid",
            product_lane_id=VenueLane.HYPERLIQUID_PERP.value,
            account_scope_id=draft.identity.account_scope_id,
            instrument_id="USDC",
            size="25",
            notional="25",
            proposal_id=draft.identity.proposal_id,
        ),
    )
    for new_spec in new_path_specs:
        with pytest.raises(
            EffectAuthorityError,
            match="activation_disabled|live_lane_denied|testnet_adapter_denied",
        ):
            issue_gate00g_permit(authority=authority, spec=new_spec)

    _enable_test_policies(monkeypatch)
    phase00 = _effect_authority(
        tmp_path / "phase00",
        clock,
        profile=PHASE00_DENY_ALL_PROFILE,
    )
    with pytest.raises(EffectAuthorityError, match="phase00_profile_forbidden"):
        issue_gate00g_permit(authority=phase00, spec=spec)
    for new_spec in new_path_specs:
        with pytest.raises(EffectAuthorityError, match="phase00_profile_forbidden"):
            issue_gate00g_permit(authority=phase00, spec=new_spec)


@pytest.mark.parametrize(
    ("identity", "overrides"),
    [
        (_identity(run_id="wrong-run"), {}),
        (_identity(account="0x" + "4" * 40), {}),
        (_identity(proposal="wrong-proposal"), {}),
        (_identity(), {"venue_id": "dydx"}),
        (_identity(), {"product_lane_id": VenueLane.BINANCE_USDM_PERP.value}),
        (_identity(), {"adapter_id": BINANCE_USDM_TESTNET_ADAPTER_ID}),
        (_identity(), {"target": f"{BINANCE_SPOT_TESTNET_URL}/wrong"}),
        (_identity(), {"operation": "cancel_order"}),
        (_identity(), {"instrument_id": "BTCUSDT"}),
        (_identity(), {"side": "sell"}),
        (_identity(), {"size": "2", "notional": "50"}),
        (_identity(), {"notional": "26"}),
        (_identity(), {"leverage": "2"}),
    ],
)
def test_exact_scope_mismatches_deny_without_consuming_base_permit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity: OrderAuthorityIdentity,
    overrides: dict,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    base_identity = _identity()
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=base_identity,
        permits=(),
    )
    base_spec = _binance_spec(draft)
    permit = issue_gate00g_permit(authority=authority, spec=base_spec)
    mismatched = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(permit,),
    )
    changed = dict(overrides)
    changed.setdefault("account_scope_id", identity.account_scope_id)
    changed.setdefault("proposal_id", identity.proposal_id)
    spec = _binance_spec(mismatched, **changed)
    with pytest.raises(EffectAuthorityError):
        mismatched.consume(spec)
    assert authority.state(permit.permit_id) == "ISSUED"


def test_missing_stale_revoked_replayed_and_tampered_permits_deny(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity()
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    spec = _binance_spec(draft)
    with pytest.raises(EffectAuthorityError, match="permit_missing"):
        draft.consume(spec)

    stale = issue_gate00g_permit(authority=authority, spec=spec, ttl_seconds=1)
    clock.now += timedelta(seconds=2)
    stale_gate = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(stale,),
    )
    with pytest.raises(EffectAuthorityError, match="expired"):
        stale_gate.consume(spec)

    clock.now += timedelta(seconds=1)
    revoked = issue_gate00g_permit(authority=authority, spec=spec)
    authority.revoke(revoked.permit_id, reason="test")
    revoked_gate = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(revoked,),
    )
    with pytest.raises(EffectAuthorityError, match="REVOKED"):
        revoked_gate.consume(spec)

    replayed = issue_gate00g_permit(authority=authority, spec=spec)
    replay_gate = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(replayed,),
    )
    replay_gate.consume(spec)
    with pytest.raises(EffectAuthorityError, match="CONSUMED"):
        replay_gate.consume(spec)

    valid = issue_gate00g_permit(authority=authority, spec=spec)
    tampered = valid.model_copy(update={"signature": "0" * 64})
    tampered_gate = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(tampered,),
    )
    with pytest.raises(EffectAuthorityError, match="signature_invalid"):
        tampered_gate.consume(spec)
    assert authority.state(valid.permit_id) == "ISSUED"


def test_binance_consumes_exact_permit_before_signing_and_http_and_replay_denies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity(account="binance-test-account")
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    spec = _binance_spec(draft)
    gate = _gate_with_permits(authority, identity, (spec,))
    session = FakeSession()
    adapter = BinanceSpotTestnetOrderAdapter(
        session=session,
        order_authority=gate,
    )
    config = BinanceTestnetConfig(
        lane="spot",
        mode=ExecutionMode.PAPER,
        base_url=BINANCE_SPOT_TESTNET_URL,
        api_key="local-test-key",
        api_secret="local-test-secret",
        submit_orders=True,
        account_scope_id=identity.account_scope_id,
        order_approval_id=identity.proposal_id,
    )
    intent = OrderIntent(
        market="ETHUSDT",
        side="BUY",
        size=1,
        limit_price=25,
    )
    fill = adapter.place_order(intent, config)
    assert fill.status == "paper_submitted"
    assert len(session.post_calls) == 1
    assert authority.state(gate.permits[0].permit_id) == "CONSUMED"
    with pytest.raises(EffectAuthorityError, match="CONSUMED"):
        adapter.place_order(intent, config)
    assert len(session.post_calls) == 1


def test_private_sink_cannot_replay_a_consumed_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity(account="binance-private-sink-account")
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    spec = _binance_spec(draft)
    gate = _gate_with_permits(authority, identity, (spec,))
    authorization = gate.consume(spec)
    session = FakeSession()
    adapter = BinanceSpotTestnetOrderAdapter(
        session=session,
        order_authority=gate,
    )
    config = BinanceTestnetConfig(
        lane="spot",
        mode=ExecutionMode.PAPER,
        base_url=BINANCE_SPOT_TESTNET_URL,
        api_key="local-test-key",
        api_secret="local-test-secret",
        submit_orders=True,
        account_scope_id=identity.account_scope_id,
        order_approval_id=identity.proposal_id,
    )
    intent = OrderIntent("ETHUSDT", "BUY", 1, 25)

    adapter._submit(
        intent,
        config,
        spec=spec,
        authorization=authorization,
    )
    with pytest.raises(EffectAuthorityError, match="dispatch_replayed"):
        adapter._submit(
            intent,
            config,
            spec=spec,
            authorization=authorization,
        )
    assert len(session.post_calls) == 1


def test_consumed_permit_that_expires_before_dispatch_is_denied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity(account="binance-stale-dispatch-account")
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    spec = _binance_spec(draft)
    gate = _gate_with_permits(authority, identity, (spec,), ttl_seconds=1)
    authorization = gate.consume(spec)
    clock.now += timedelta(seconds=2)
    session = FakeSession()
    adapter = BinanceSpotTestnetOrderAdapter(
        session=session,
        order_authority=gate,
    )
    with pytest.raises(EffectAuthorityError, match="authorization_stale"):
        adapter._submit(
            OrderIntent("ETHUSDT", "BUY", 1, 25),
            BinanceTestnetConfig(
                lane="spot",
                mode=ExecutionMode.PAPER,
                base_url=BINANCE_SPOT_TESTNET_URL,
                api_key="must-not-be-read",
                api_secret="must-not-be-read",
                submit_orders=True,
                account_scope_id=identity.account_scope_id,
                order_approval_id=identity.proposal_id,
            ),
            spec=spec,
            authorization=authorization,
        )
    assert session.post_calls == []


def test_missing_authority_stops_all_three_venue_sinks_before_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binance_session = FakeSession()
    binance = BinanceSpotTestnetOrderAdapter(session=binance_session)
    binance_config = BinanceTestnetConfig(
        lane="spot",
        mode=ExecutionMode.PAPER,
        base_url=BINANCE_SPOT_TESTNET_URL,
        api_key="must-not-be-read",
        api_secret="must-not-be-read",
        submit_orders=True,
        account_scope_id="binance-account",
        order_approval_id=PROPOSAL,
    )
    intent = OrderIntent("ETHUSDT", "BUY", 1, 25)
    with pytest.raises(EffectAuthorityError, match="authority_missing"):
        binance.place_order(intent, binance_config)
    assert binance_session.post_calls == []

    dydx = DydxSdkOrderAdapter()
    touched = {"capture": 0}

    async def capture(*args, **kwargs):
        touched["capture"] += 1
        return {}

    monkeypatch.setattr(dydx, "_capture_market_state", capture)
    with pytest.raises(EffectAuthorityError, match="authority_missing"):
        dydx.place_order(
            OrderIntent("ETH-USD", "BUY", 1, 25),
            DydxNetworkConfig(
                mode=ExecutionMode.PAPER,
                submit_orders=True,
                wallet_address=ACCOUNT,
                private_key="must-not-be-read",
                rest_indexer="https://local.invalid",
                order_approval_id=PROPOSAL,
            ),
        )
    assert touched["capture"] == 0

    exchange = FakeExchange()
    hyper_session = FakeSession()
    key_reads = {"count": 0}

    def key_reader(*args):
        key_reads["count"] += 1
        return "3" * 64

    hyper = HyperliquidTestnetPairExecutor(
        session=hyper_session,
        keychain_reader=key_reader,
        exchange_factory=lambda *args: exchange,
        approval_validator=lambda *args: (),
    )
    hyper_config = _hyper_config()
    result = hyper.submit_pair(_hyper_intents(), hyper_config)
    assert result.status == "pair_blocked"
    assert "authority_missing" in result.reason
    assert key_reads["count"] == 0
    assert hyper_session.post_calls == []
    assert exchange.leverage_calls == []
    assert exchange.bulk_order_calls == []


def test_binance_transport_is_not_constructed_before_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructed = {"count": 0}

    def construct_session():
        constructed["count"] += 1
        raise AssertionError("transport construction must remain behind Gate 00G")

    monkeypatch.setattr(
        "quant_platform.binance_testnet.requests.Session",
        construct_session,
    )
    adapter = BinanceSpotTestnetOrderAdapter()
    with pytest.raises(EffectAuthorityError, match="authority_missing"):
        adapter.place_order(
            OrderIntent("ETHUSDT", "BUY", 1, 25),
            BinanceTestnetConfig(
                lane="spot",
                mode=ExecutionMode.PAPER,
                base_url=BINANCE_SPOT_TESTNET_URL,
                api_key="must-not-be-read",
                api_secret="must-not-be-read",
                submit_orders=True,
                account_scope_id="binance-account",
                order_approval_id=PROPOSAL,
            ),
        )
    assert constructed["count"] == 0

    fake_session = FakeSession()
    fake_authority_adapter = BinanceSpotTestnetOrderAdapter(
        session=fake_session,
        order_authority=SimpleNamespace(),
    )
    with pytest.raises(EffectAuthorityError, match="authority_type_invalid"):
        fake_authority_adapter.place_order(
            OrderIntent("ETHUSDT", "BUY", 1, 25),
            BinanceTestnetConfig(
                lane="spot",
                mode=ExecutionMode.PAPER,
                base_url=BINANCE_SPOT_TESTNET_URL,
                api_key="must-not-be-read",
                api_secret="must-not-be-read",
                submit_orders=True,
                account_scope_id="binance-account",
                order_approval_id=PROPOSAL,
            ),
        )
    assert fake_session.post_calls == []


def _hyper_config() -> HyperliquidTestnetConfig:
    return HyperliquidTestnetConfig(
        mode=ExecutionMode.PAPER,
        network="testnet",
        base_url=HYPERLIQUID_TESTNET_URL,
        master_address=ACCOUNT,
        agent_address=AGENT,
        keychain_service="local-test-service",
        submit_orders=True,
        max_pair_notional_usd=100,
        order_approval_id=PROPOSAL,
        requested_leverage=1,
    )


def _hyper_intents() -> tuple[OrderIntent, OrderIntent]:
    return (
        OrderIntent("BTC-USD", "BUY", 1, 25),
        OrderIntent("ETH-USD", "SELL", 1, 25),
    )


def _hyper_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CorrectiveOrderAuthority, EffectAuthority]:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity()
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    config = _hyper_config()
    intents = _hyper_intents()
    specs = tuple(
        HyperliquidTestnetPairExecutor._account_mutation_spec(
            draft,
            intent,
            config,
            operation="update_leverage",
        )
        for intent in intents
    ) + tuple(
        HyperliquidTestnetPairExecutor._order_effect_spec(
            draft,
            intent,
            config,
            operation="bulk_orders",
        )
        for intent in intents
    )
    return _gate_with_permits(authority, identity, specs), authority


def _prepare_hyper_executor(
    gate: CorrectiveOrderAuthority,
    exchange: FakeExchange,
    approval_validator,
    monkeypatch: pytest.MonkeyPatch,
) -> HyperliquidTestnetPairExecutor:
    executor = HyperliquidTestnetPairExecutor(
        keychain_reader=lambda *args: "3" * 64,
        exchange_factory=lambda *args: exchange,
        approval_validator=approval_validator,
        order_authority=gate,
    )
    config = _hyper_config()
    executor._validated_wallet = object()
    executor._validated_wallet_binding = (
        str(config.keychain_service),
        str(config.agent_address).lower(),
    )
    monkeypatch.setattr(
        executor,
        "no_order_preflight",
        lambda config=None: {"ready_for_no_order_preflight": True, "blockers": ""},
    )
    monkeypatch.setattr(executor, "_pair_account_blockers", lambda *args: [])
    monkeypatch.setattr(executor, "_pair_market_rule_blockers", lambda *args: [])
    monkeypatch.setattr(executor, "_pair_exit_price_blockers", lambda *args: [])
    return executor


def test_hyperliquid_rechecks_approval_before_leverage_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, _ = _hyper_gate(tmp_path, monkeypatch)
    exchange = FakeExchange()
    approvals = {"calls": 0}

    def approval_validator(*args):
        approvals["calls"] += 1
        return () if approvals["calls"] < 3 else ("approval_revoked",)

    executor = _prepare_hyper_executor(
        gate,
        exchange,
        approval_validator,
        monkeypatch,
    )
    result = executor.submit_pair(_hyper_intents(), _hyper_config())
    assert result.status == "pair_blocked"
    assert "approval_revoked" in result.reason
    assert exchange.leverage_calls == []
    assert exchange.bulk_order_calls == []


def test_hyperliquid_rechecks_approval_before_each_leverage_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, _ = _hyper_gate(tmp_path, monkeypatch)
    exchange = FakeExchange()
    approvals = {"calls": 0}

    def approval_validator(*args):
        approvals["calls"] += 1
        if approvals["calls"] < 5:
            return ()
        return ("approval_revoked_between_leverage_legs",)

    executor = _prepare_hyper_executor(
        gate,
        exchange,
        approval_validator,
        monkeypatch,
    )
    result = executor.submit_pair(_hyper_intents(), _hyper_config())
    assert result.status == "pair_configuration_error"
    assert len(exchange.leverage_calls) == 1
    assert exchange.bulk_order_calls == []


def test_hyperliquid_local_fake_consumes_leverage_and_order_permits_before_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate, authority = _hyper_gate(tmp_path, monkeypatch)
    exchange = FakeExchange()
    executor = _prepare_hyper_executor(
        gate,
        exchange,
        lambda *args: (),
        monkeypatch,
    )
    result = executor.submit_pair(_hyper_intents(), _hyper_config())
    assert result.status == "pair_rejected"
    assert len(exchange.leverage_calls) == 2
    assert len(exchange.bulk_order_calls) == 1
    assert all(authority.state(permit.permit_id) == "CONSUMED" for permit in gate.permits)


def test_dydx_unconfirmed_opening_order_is_not_retried_on_second_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_test_policies(monkeypatch)
    clock = Clock()
    authority = _effect_authority(tmp_path, clock)
    identity = _identity()
    draft = CorrectiveOrderAuthority(
        authority=authority,
        identity=identity,
        permits=(),
    )
    config = DydxNetworkConfig(
        mode=ExecutionMode.PAPER,
        node_url="configured.test:443",
        rest_indexer="https://configured.invalid",
        submit_orders=True,
        wallet_address=ACCOUNT,
        private_key="local-only-secret",
        order_approval_id=PROPOSAL,
    )
    intent = OrderIntent("ETH-USD", "BUY", 1, 25)
    provisional = DydxSdkOrderAdapter(order_authority=draft)
    first_attempt = provisional._attempt_specs(intent, config)[0]
    spec = provisional._order_effect_spec(draft, intent, first_attempt)
    gate = _gate_with_permits(authority, identity, (spec,))
    adapter = DydxSdkOrderAdapter(order_authority=gate)
    calls = {"submits": 0}

    async def capture(*args, **kwargs):
        require_consumed_authorization(
            kwargs["authorization"],
            owner=gate,
            spec=kwargs["spec"],
        )
        return {"fill_ids": set(), "order_ids": set(), "position_size": 0.0}

    async def submit(*args, **kwargs):
        require_consumed_authorization(
            kwargs["authorization"],
            owner=gate,
            spec=kwargs["spec"],
        )
        calls["submits"] += 1
        return object()

    async def confirm(*args, **kwargs):
        return {"confirmed": False, "avg_price": None}

    monkeypatch.setattr(adapter, "_capture_market_state", capture)
    monkeypatch.setattr(adapter, "_submit_order", submit)
    monkeypatch.setattr(adapter, "_confirm_market_state", confirm)
    monkeypatch.setattr(adapter, "_record_attempt", lambda *args: None)
    fill = adapter.place_order(intent, config)
    assert fill.status == "broadcast_accepted_unconfirmed"
    assert calls["submits"] == 1


def test_dynamic_adapter_and_custom_execution_venue_bypasses_are_denied_before_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported = {"count": 0}

    def import_module(name: str):
        imported["count"] += 1
        raise AssertionError("untrusted adapter import must not run")

    monkeypatch.setattr("quant_platform.execution.importlib.import_module", import_module)
    with pytest.raises(EffectAuthorityError, match="adapter_path_denied"):
        build_venue_order_client_adapter("dydx", "untrusted.module:Adapter")
    assert imported["count"] == 0

    class CustomVenue:
        def __init__(self) -> None:
            self.calls = 0

        def place_order(self, intent):
            self.calls += 1
            raise AssertionError("custom execution venue must not run")

    venue = CustomVenue()
    plan = SpreadOrderPlan(
        pair="ETH-BTC",
        strategy_id=1,
        status="paper_ready",
        reason="test",
        intents=(OrderIntent("ETH-USD", "BUY", 1, 25),),
    )
    fills = submit_paper_plan(plan, venue)
    assert fills[0].status == "paper_blocked_gate00g_execution_venue_denied"
    assert venue.calls == 0
    with pytest.raises(EffectAuthorityError, match="order_callable_denied"):
        execution_module._place_order_call(
            venue.place_order,
            OrderIntent("ETHUSDT", "BUY", 1, 25),
        )
    assert venue.calls == 0

    trusted = BinanceSpotTestnetOrderAdapter(session=FakeSession())
    wrapper = PaperVenueExecution("binance", trusted)
    wrapper.client = venue
    mutated = wrapper.place_order(OrderIntent("ETHUSDT", "BUY", 1, 25))
    assert mutated.status == "paper_blocked_gate00g_order_authority_required"
    assert venue.calls == 0

    class CustomPairExecutor:
        def __init__(self) -> None:
            self.calls = 0

        def submit_pair(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("substituted pair executor must not run")

    custom_pair_executor = CustomPairExecutor()
    pair_adapter = HyperliquidTestnetPairAdapter(executor=custom_pair_executor)
    with pytest.raises(EffectAuthorityError, match="pair_executor_denied"):
        pair_adapter.submit_pair(_hyper_intents(), _hyper_config())
    assert custom_pair_executor.calls == 0


def test_gate00g_direct_sink_inventory_and_private_sink_signatures_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "src/quant_platform/hyperliquid_testnet.py": (
            {"update_leverage", "bulk_orders", "bulk_cancel", "market_close"},
            {
                ("_submit_pair_locked", "update_leverage"),
                ("_submit_pair_locked", "bulk_orders"),
                ("_recover_pair_to_flat", "bulk_cancel"),
                ("_recover_pair_to_flat", "market_close"),
            },
        ),
        "src/quant_platform/binance_testnet.py": (
            {"post"},
            {("_submit", "post")},
        ),
        "src/quant_platform/dydx_sdk_order_adapter.py": (
            {"place_order", "close_position"},
            {
                ("_place_order", "place_order"),
                ("_close_position", "close_position"),
            },
        ),
        "src/quant_platform/orchestration/corrective_live_canary_execution.py": (
            {"update_leverage", "bulk_orders", "bulk_cancel", "market_close"},
            {
                ("run_live_canary_executor", "update_leverage"),
                ("run_live_canary_executor", "bulk_orders"),
                ("_recover_to_flat", "bulk_cancel"),
                ("_recover_to_flat", "market_close"),
            },
        ),
        "src/quant_platform/orchestration/corrective_testnet_collateral_transfer.py": (
            {"usd_class_transfer"},
            {("run_testnet_collateral_transfer", "usd_class_transfer")},
        ),
    }

    class SinkVisitor(ast.NodeVisitor):
        def __init__(self, sink_names: set[str]) -> None:
            self.function = ""
            self.calls: set[tuple[str, str]] = set()
            self.sink_names = sink_names

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            previous = self.function
            self.function = node.name
            self.generic_visit(node)
            self.function = previous

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._visit_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._visit_function(node)

        def visit_Call(self, node: ast.Call) -> None:
            if (
                self.function
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in self.sink_names
            ):
                self.calls.add((self.function, node.func.attr))
            self.generic_visit(node)

    for relative, (sink_names, exact_calls) in expected.items():
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        visitor = SinkVisitor(sink_names)
        visitor.visit(tree)
        assert visitor.calls == exact_calls

    fenced_sources = {
        BinanceSpotTestnetOrderAdapter._submit: "claim_effect_dispatch",
        DydxSdkOrderAdapter._place_order: "claim_effect_dispatch",
        DydxSdkOrderAdapter._close_position: "claim_effect_dispatch",
        HyperliquidTestnetPairExecutor._submit_pair_locked: (
            "_claim_pair_authorization"
        ),
        HyperliquidTestnetPairExecutor._recover_pair_to_flat: (
            "claim_effect_dispatch"
        ),
        live_canary_execution.run_live_canary_executor: "claim_effect_dispatch",
        live_canary_execution._recover_to_flat: "claim_effect_dispatch",
        collateral_transfer.run_testnet_collateral_transfer: (
            "claim_effect_dispatch"
        ),
    }
    for method, required_fence in fenced_sources.items():
        assert required_fence in inspect.getsource(method)

    for method in (
        BinanceSpotTestnetOrderAdapter._submit,
        DydxSdkOrderAdapter._place_order,
        DydxSdkOrderAdapter._close_position,
        DydxSdkOrderAdapter._build_wallet,
        DydxSdkOrderAdapter._connect_node,
        HyperliquidTestnetPairExecutor._load_wallet,
        HyperliquidTestnetPairExecutor._build_exchange,
        live_canary_execution._load_wallet,
        live_canary_execution._build_exchange,
    ):
        parameters = inspect.signature(method).parameters
        assert "spec" in parameters
        assert "authorization" in parameters

    assert (
        "order_authority"
        in inspect.signature(live_canary_execution.run_live_canary_executor).parameters
    )
    assert (
        "order_authority"
        in inspect.signature(collateral_transfer.run_testnet_collateral_transfer).parameters
    )
