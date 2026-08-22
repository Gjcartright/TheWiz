from __future__ import annotations

import fcntl
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest
from eth_account import Account

import quant_platform.hyperliquid_testnet as hyperliquid_module
from quant_platform.execution import OrderIntent, hyperliquid_testnet_order_preflight_status
from quant_platform.hyperliquid_testnet import (
    HYPERLIQUID_AGENT_KEYCHAIN_CREDENTIAL_ID,
    HYPERLIQUID_INFO_OPERATIONS,
    HYPERLIQUID_TESTNET_INFO_URL,
    HyperliquidPairExecutionResult,
    HyperliquidTestnetConfig,
    HyperliquidTestnetOrderAdapter,
    HyperliquidTestnetPairAdapter,
    HyperliquidTestnetPairExecutor,
    _approval_intent_blockers,
    _exit_price_band_blockers,
    _signed_approval_blockers,
    hyperliquid_testnet_margin_snapshot,
    write_hyperliquid_testnet_margin_snapshot,
    write_hyperliquid_testnet_preflight_report,
)
from quant_platform.orchestration.corrective_external_effects import (
    RESEARCH_EXTERNAL_EFFECT_PROFILE,
    ExternalEffectCallContract,
    external_effect_authority_session,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
)

_RealHyperliquidTestnetPairExecutor = HyperliquidTestnetPairExecutor


class _LegacyAuthorizedSpec(SimpleNamespace):
    def payload_sha256(self) -> str:
        material = json.dumps(vars(self), sort_keys=True, default=str)
        return sha256(material.encode("utf-8")).hexdigest()


class _LegacyConsumedAuthority:
    def state(self, permit_id: str) -> str:
        return "CONSUMED"


class _LegacyBehaviorOrderAuthority:
    """Local-only double preserving legacy executor tests behind the new gate."""

    def __init__(self) -> None:
        self.authority = _LegacyConsumedAuthority()
        self._counter = 0
        self._dispatched: set[str] = set()
        self._permits: dict[str, SimpleNamespace] = {}

    def spec(self, **kwargs):
        return _LegacyAuthorizedSpec(**kwargs)

    def consume(self, spec):
        self._counter += 1
        permit_id = f"legacy-{self._counter}"
        nonce = f"legacy-nonce-{self._counter}"
        self._permits[permit_id] = SimpleNamespace(
            nonce=nonce,
            effect_kind=spec.effect_kind,
        )
        return SimpleNamespace(
            receipt=SimpleNamespace(
                permit_id=permit_id,
                nonce=nonce,
                effect_kind=spec.effect_kind,
            ),
            spec_sha256=spec.payload_sha256(),
            _owner=self,
        )

    def consume_all(self, specs):
        return tuple(self.consume(spec) for spec in specs)

    def _permit_by_id(self, permit_id):
        return self._permits[permit_id]

    def claim_dispatch(self, authorization, spec) -> None:
        if authorization is None or authorization._owner is not self:
            raise ValueError("legacy_authorization_owner_mismatch")
        if authorization.spec_sha256 != spec.payload_sha256():
            raise ValueError("legacy_authorization_scope_mismatch")
        permit_id = authorization.receipt.permit_id
        if permit_id in self._dispatched:
            raise ValueError("legacy_authorization_dispatch_replayed")
        self._dispatched.add(permit_id)


def HyperliquidTestnetPairExecutor(*args, **kwargs):
    kwargs.setdefault("order_authority", _LegacyBehaviorOrderAuthority())
    return _RealHyperliquidTestnetPairExecutor(*args, **kwargs)


@pytest.fixture(autouse=True)
def _allow_explicit_legacy_authority_double(monkeypatch):
    real_require = hyperliquid_module.require_order_authority

    def require(authority):
        if isinstance(authority, _LegacyBehaviorOrderAuthority):
            return authority
        return real_require(authority)

    monkeypatch.setattr(hyperliquid_module, "require_order_authority", require)


@pytest.fixture(autouse=True)
def _authorize_local_preflight_effects(tmp_path):
    authority = EffectAuthority(
        root=tmp_path,
        secret=b"h" * 32,
        issuer_id="hyperliquid-preflight-test",
        profile=RESEARCH_EXTERNAL_EFFECT_PROFILE,
    )
    with external_effect_authority_session(
        authority=authority,
        run_id="hyperliquid-preflight-test-run",
        intended_slot_id="hyperliquid-preflight-test-slot",
        source_fingerprint_sha256="a" * 64,
        runtime_fingerprint_sha256="b" * 64,
        configuration_fingerprint_sha256="c" * 64,
        provider_id="hyperliquid_testnet_preflight",
        account_scope_id="hyperliquid:testnet:agent_preflight",
        reservation_id=f"reservation-{tmp_path.name}",
        reservation_sha256=sha256(str(tmp_path).encode()).hexdigest(),
        allowed_targets=frozenset({HYPERLIQUID_TESTNET_INFO_URL}),
        allowed_credential_keys=frozenset(
            {HYPERLIQUID_AGENT_KEYCHAIN_CREDENTIAL_ID}
        ),
        max_total_requests=64,
        max_total_credits=0,
        allowed_call_contracts=frozenset(
            {
                ExternalEffectCallContract(
                    operation=operation,
                    method="POST",
                    target=HYPERLIQUID_TESTNET_INFO_URL,
                    credit_units_per_request=0,
                )
                for operation in HYPERLIQUID_INFO_OPERATIONS.values()
            }
        ),
    ):
        yield


def test_default_executor_gate_rejects_pass_without_explicit_testnet_authority(monkeypatch):
    from quant_platform.orchestration import hyperliquid_learning_and_risk

    monkeypatch.setattr(
        hyperliquid_learning_and_risk,
        "validate_testnet_smoke_approval",
        lambda **kwargs: {
            "status": "PASS",
            "blockers": [],
            "execution_allowed": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    config, _ = _test_config(
        submit_orders=True,
        order_approval_id="authority-missing",
    )

    assert _signed_approval_blockers("authority-missing", config) == [
        "hyperliquid_testnet_explicit_order_authority_not_granted"
    ]


def test_default_executor_requests_exit_scoped_authority_for_risk_reduction(
    monkeypatch,
):
    from quant_platform.orchestration import hyperliquid_learning_and_risk

    scopes = []

    def validate(**kwargs):
        scopes.append(kwargs["validation_scope"])
        return {
            "status": "PASS",
            "blockers": [],
            "execution_allowed": True,
            "testnet_order_authority": True,
            "authority_scope": kwargs["validation_scope"],
            "live_trading_authorized": False,
        }

    monkeypatch.setattr(
        hyperliquid_learning_and_risk,
        "validate_testnet_smoke_approval",
        validate,
    )
    config, _ = _test_config(
        submit_orders=True,
        order_approval_id="exit-scope",
    )

    assert _signed_approval_blockers(
        "exit-scope", config, validation_scope="exit"
    ) == []
    assert scopes == ["exit"]


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class _RoleSession:
    def __init__(self, master_address: str):
        self.master_address = master_address
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if json.get("type") == "clearinghouseState":
            return _Response({"assetPositions": []})
        if json.get("type") == "openOrders":
            return _Response([])
        if json.get("type") == "meta":
            return _Response(_market_meta())
        return _Response({"role": "agent", "data": {"user": self.master_address}})


class _MarginSession:
    def __init__(self):
        self.requests = []

    def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if json.get("type") == "spotClearinghouseState":
            return _Response({"balances": [{"coin": "USDC", "total": "75.0"}]})
        return _Response(
            {
                "marginSummary": {
                    "accountValue": "100.0",
                    "totalMarginUsed": "25.0",
                    "totalNtlPos": "40.0",
                },
                "withdrawable": "70.0",
                "assetPositions": [
                    {"position": {"coin": "BTC", "szi": "0.001"}},
                    {"position": {"coin": "ETH", "szi": "0"}},
                ],
            }
        )


class _SpotOnlyMarginSession(_MarginSession):
    def post(self, url, json, timeout):
        if json.get("type") == "spotClearinghouseState":
            return _Response({"balances": [{"coin": "USDC", "total": "995.0"}]})
        return _Response(
            {
                "marginSummary": {"accountValue": "0", "totalMarginUsed": "0", "totalNtlPos": "0"},
                "withdrawable": "0",
                "assetPositions": [],
            }
        )


class _FakeExchange:
    def __init__(self):
        self.requests = []
        self.cancel_requests = []
        self.close_requests = []
        self.leverage_updates = []

    def update_leverage(self, leverage, coin, is_cross=True):
        self.leverage_updates.append({"leverage": leverage, "coin": coin, "is_cross": is_cross})
        return {"status": "ok"}

    def bulk_orders(self, requests):
        self.requests.append(requests)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"resting": {"oid": 123}},
                        {"resting": {"oid": 124}},
                    ]
                }
            },
        }

    def bulk_cancel(self, requests):
        self.cancel_requests.append(requests)
        return {"status": "ok"}

    def market_close(self, coin, sz=None):
        self.close_requests.append({"coin": coin, "sz": sz})
        return {"status": "ok"}


class _PartialExchange(_FakeExchange):
    def bulk_orders(self, requests):
        self.requests.append(requests)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"resting": {"oid": 123}},
                        {"error": "Insufficient margin"},
                    ]
                }
            },
        }


class _FilledPartialExchange(_FakeExchange):
    def bulk_orders(self, requests):
        self.requests.append(requests)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"oid": 321, "totalSz": "0.0001"}},
                        {"error": "Insufficient margin"},
                    ]
                }
            },
        }


class _UnconfirmedExchange(_FakeExchange):
    def bulk_orders(self, requests):
        self.requests.append(requests)
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{}, {}]}},
        }


class _UnknownResponseExchange(_FakeExchange):
    def bulk_orders(self, requests):
        self.requests.append(requests)
        raise TimeoutError("response lost after submission attempt")


class _SequencedAccountSession(_RoleSession):
    def __init__(self, master_address, *, states, open_orders):
        super().__init__(master_address)
        self.states = list(states)
        self.orders = list(open_orders)

    def post(self, url, json, timeout):
        self.requests.append({"url": url, "json": json, "timeout": timeout})
        if json.get("type") == "clearinghouseState":
            return _Response(self.states.pop(0))
        if json.get("type") == "openOrders":
            return _Response(self.orders.pop(0))
        if json.get("type") == "meta":
            return _Response(_market_meta())
        return _Response({"role": "agent", "data": {"user": self.master_address}})


class _CustomMetaSession(_RoleSession):
    def __init__(self, master_address, meta):
        super().__init__(master_address)
        self.meta = meta

    def post(self, url, json, timeout):
        if json.get("type") == "meta":
            self.requests.append({"url": url, "json": json, "timeout": timeout})
            return _Response(self.meta)
        return super().post(url, json, timeout)


def _positions(**sizes):
    return {
        "assetPositions": [
            {"position": {"coin": coin, "szi": str(size)}} for coin, size in sizes.items()
        ]
    }


def _market_meta():
    return {
        "universe": [
            {
                "name": "BTC",
                "szDecimals": 5,
                "maxLeverage": 50,
                "onlyIsolated": False,
                "isDelisted": False,
            },
            {
                "name": "ETH",
                "szDecimals": 4,
                "maxLeverage": 50,
                "onlyIsolated": False,
                "isDelisted": False,
            },
        ]
    }


class _StubPairExecutor:
    def __init__(self):
        self.calls = []

    def submit_pair(self, intents, config):
        self.calls.append((intents, config))
        return HyperliquidPairExecutionResult(status="pair_blocked", reason="stub_blocked")


def _test_config(
    *,
    submit_orders: bool = False,
    order_approval_id: str | None = None,
    requested_leverage: int = 1,
    margin_mode: str = "cross",
    one_x_testnet_proof_id: str | None = None,
    leverage_scenario_id: str | None = None,
):
    master = Account.create()
    agent = Account.create()
    config = HyperliquidTestnetConfig(
        master_address=master.address,
        agent_address=agent.address,
        keychain_service="TheWiz Test Keychain",
        submit_orders=submit_orders,
        order_approval_id=order_approval_id,
        requested_leverage=requested_leverage,
        margin_mode=margin_mode,
        one_x_testnet_proof_id=one_x_testnet_proof_id,
        leverage_scenario_id=leverage_scenario_id,
    )
    return config, agent.key.hex()


def _pair_intents():
    return (
        OrderIntent(market="BTC-USD", side="BUY", size=0.00016, limit_price=64_000.0),
        OrderIntent(market="ETH-USD", side="SELL", size=0.0034, limit_price=3_000.0),
    )


def _pair_exit_intents():
    return (
        OrderIntent(
            market="BTC-USD",
            side="SELL",
            size=0.00016,
            limit_price=64_000.0,
            reduce_only=True,
        ),
        OrderIntent(
            market="ETH-USD",
            side="BUY",
            size=0.0034,
            limit_price=3_000.0,
            reduce_only=True,
        ),
    )


def _approval_payload():
    return {
        "approval_version": "hyperliquid-testnet-smoke-v10",
        "max_total_notional_usd": 25.0,
        "maximum_exit_slippage_bps": 50.0,
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
        "exit_policy": {
            "reduce_only": True,
            "opposite_side": True,
            "market_scope": "approved_entry_markets",
            "maximum_size": "approved_entry_size_per_leg",
            "entry_notional_cap_not_reapplied_to_exit": True,
        },
    }


def test_runtime_intents_must_match_signed_entry_and_exit_policy():
    approval = _approval_payload()
    matching_entry = _pair_intents()
    drifted_entry = (
        matching_entry[0],
        OrderIntent(
            market="ETH-USD",
            side="SELL",
            size=0.0033,
            limit_price=3_000.0,
        ),
    )
    valid_exit = (
        OrderIntent(
            market="BTC-USD",
            side="SELL",
            size=0.00016,
            limit_price=64_000.0,
            reduce_only=True,
        ),
        OrderIntent(
            market="ETH-USD",
            side="BUY",
            size=0.0034,
            limit_price=3_000.0,
            reduce_only=True,
        ),
    )
    oversized_exit = (
        valid_exit[0],
        OrderIntent(
            market="ETH-USD",
            side="BUY",
            size=0.0035,
            limit_price=3_000.0,
            reduce_only=True,
        ),
    )

    assert _approval_intent_blockers(approval, matching_entry) == []
    assert _approval_intent_blockers(approval, drifted_entry) == [
        "hyperliquid_runtime_entry_size_mismatch:1"
    ]
    assert _approval_intent_blockers(approval, valid_exit) == []
    assert "hyperliquid_runtime_exit_size_exceeds_approval:1" in (
        _approval_intent_blockers(approval, oversized_exit)
    )

    malformed = dict(approval)
    malformed["maximum_exit_slippage_bps"] = "not-a-number"
    assert "hyperliquid_runtime_exit_slippage_policy_invalid" in (
        _approval_intent_blockers(malformed, valid_exit)
    )

    appreciated_exit = (
        OrderIntent(
            market="BTC-USD",
            side="SELL",
            size=0.00016,
            limit_price=100_000.0,
            reduce_only=True,
        ),
        valid_exit[1],
    )
    assert _approval_intent_blockers(approval, appreciated_exit) == []


def test_reduce_only_exit_price_must_be_marketable_and_inside_signed_band():
    valid = _pair_exit_intents()
    mids = {"BTC": 64_000.0, "ETH": 3_000.0}
    too_wide = (
        OrderIntent(
            market="BTC-USD",
            side="SELL",
            size=0.00016,
            limit_price=63_000.0,
            reduce_only=True,
        ),
        valid[1],
    )
    not_marketable = (
        OrderIntent(
            market="BTC-USD",
            side="SELL",
            size=0.00016,
            limit_price=64_100.0,
            reduce_only=True,
        ),
        valid[1],
    )

    assert _exit_price_band_blockers(valid, mids, 50.0) == []
    assert "hyperliquid_runtime_exit_price_below_slippage_band:0" in (
        _exit_price_band_blockers(too_wide, mids, 50.0)
    )
    assert "hyperliquid_runtime_exit_limit_not_marketable:0" in (
        _exit_price_band_blockers(not_marketable, mids, 50.0)
    )


def test_no_order_preflight_verifies_authorized_agent_and_local_signature():
    config, secret = _test_config()
    session = _RoleSession(config.master_address or "")
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
    )

    result = executor.no_order_preflight(config)

    assert result["ready_for_no_order_preflight"] is True
    assert result["ready_for_testnet_submit"] is False
    assert result["agent_authorized_for_master"] is True
    assert result["agent_key_matches_address"] is True
    assert result["local_signature_created"] is True
    assert result["blockers"] == ""
    assert result["order_submission_performed"] is False
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert secret not in str(result)
    assert session.requests[0]["json"] == {"type": "userRole", "user": config.agent_address}


def test_no_order_preflight_never_grants_submit_readiness_from_environment_toggle():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="not-validated-by-read-only-preflight",
    )
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
    )

    result = executor.no_order_preflight(config)

    assert result["ready_for_no_order_preflight"] is True
    assert result["submit_orders_enabled"] is True
    assert result["ready_for_testnet_submit"] is False
    assert result["order_submission_performed"] is False
    assert result["testnet_order_authority"] is False
    assert result["live_trading_authorized"] is False
    assert result["next_action"] == "create_a_runtime_only_order_approval_before_any_two_leg_submission"


def test_preflight_report_never_contains_agent_private_key(tmp_path):
    config, secret = _test_config()
    frame = write_hyperliquid_testnet_preflight_report(
        root=tmp_path,
        config=config,
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
    )

    markdown = (tmp_path / "reports" / "active" / "hyperliquid_testnet_preflight.md").read_text(
        encoding="utf-8"
    )
    assert bool(frame.iloc[0]["ready_for_no_order_preflight"]) is True
    assert secret not in markdown


def test_margin_snapshot_is_read_only_and_computes_buffer(tmp_path):
    config, _ = _test_config()
    session = _MarginSession()

    result = hyperliquid_testnet_margin_snapshot(config, session=session)
    frame = write_hyperliquid_testnet_margin_snapshot(root=tmp_path, config=config, session=session)

    assert result["status"] == "READY"
    assert result["margin_buffer"] == 0.75
    assert result["margin_utilization"] == 0.25
    assert result["spot_usdc_usd"] == 75.0
    assert result["open_positions"] == 1
    assert session.requests[0]["json"] == {
        "type": "clearinghouseState",
        "user": config.master_address,
    }
    assert frame.iloc[0]["status"] == "READY"
    assert (tmp_path / "reports" / "active" / "hyperliquid_testnet_margin_snapshot.md").exists()


def test_margin_snapshot_identifies_spot_to_perp_transfer_blocker():
    config, _ = _test_config()

    result = hyperliquid_testnet_margin_snapshot(config, session=_SpotOnlyMarginSession())

    assert result["status"] == "BLOCKED"
    assert result["spot_usdc_usd"] == 995.0
    assert result["blockers"] == "testnet_usdc_requires_spot_to_perp_transfer"


def test_single_leg_adapter_refuses_pair_trade_submission():
    fill = HyperliquidTestnetOrderAdapter().place_order(
        OrderIntent(market="BTC-USD", side="BUY", size=0.001, limit_price=64_000.0)
    )

    assert fill.status == "paper_blocked_hyperliquid_pair_executor_required"


def test_pair_adapter_denies_substituted_executor_and_keeps_single_leg_closed():
    executor = _StubPairExecutor()
    adapter = HyperliquidTestnetPairAdapter(executor=executor)
    config, _ = _test_config()

    with pytest.raises(
        EffectAuthorityError,
        match="gate00g_hyperliquid_pair_executor_denied",
    ):
        adapter.submit_pair(_pair_intents(), config)
    single = adapter.place_order(_pair_intents()[0], config)

    assert executor.calls == []
    assert single.status == "paper_blocked_hyperliquid_pair_executor_required"


def test_pair_executor_refuses_submission_without_runtime_approval():
    config, secret = _test_config(submit_orders=True)
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert "missing_explicit_hyperliquid_order_approval" in result.reason
    assert exchange.requests == []


def test_pair_executor_refuses_runtime_id_rejected_by_signed_approval_gate():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="arbitrary-id",
    )
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [
            "runtime_hyperliquid_order_approval_id_mismatch"
        ],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "runtime_hyperliquid_order_approval_id_mismatch"
    assert exchange.requests == []
    assert result.order_submission_performed is False


def test_real_exchange_path_requires_durable_execution_state_before_any_io():
    config, _ = _test_config(
        submit_orders=True,
        order_approval_id="real-path-no-journal",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("missing durable state must block before I/O")

    executor = HyperliquidTestnetPairExecutor(
        keychain_reader=forbidden,
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "hyperliquid_real_exchange_requires_durable_execution_state"
    assert result.order_submission_performed is False


def test_real_exchange_path_forbids_custom_approval_validator_before_any_io(
    tmp_path,
):
    config, _ = _test_config(
        submit_orders=True,
        order_approval_id="real-path-custom-validator",
    )
    touched = False

    def validator(approval_id, settings):
        nonlocal touched
        touched = True
        return []

    executor = HyperliquidTestnetPairExecutor(
        approval_validator=validator,
        state_path=tmp_path / "execution_state.json",
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "hyperliquid_real_exchange_custom_approval_validator_forbidden"
    assert touched is False
    assert result.order_submission_performed is False


def test_existing_os_lock_blocks_pair_submission_before_validation_or_key_access(
    tmp_path,
):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="locked-one-run-approval",
    )
    state_path = tmp_path / "execution_state.json"
    lock_path = state_path.with_suffix(state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    touched = False

    def forbidden(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("locked executor must not validate or read a key")

    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=forbidden,
        state_path=state_path,
    )
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            result = executor.submit_pair(_pair_intents(), config)
            recovery = executor.recover_incomplete_pair(config)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    assert result.status == "pair_blocked"
    assert result.reason == "hyperliquid_testnet_pair_executor_lock_present"
    assert recovery.status == "pair_recovery_blocked"
    assert recovery.reason == "hyperliquid_testnet_pair_executor_lock_present"
    assert touched is False
    assert exchange.requests == []
    assert not state_path.exists()


def test_approval_is_revalidated_immediately_before_submission_preparation(
    tmp_path,
):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="approval-drift-during-preflight",
    )
    calls = 0

    def validator(approval_id, settings):
        nonlocal calls
        calls += 1
        return [] if calls == 1 else ["testnet_smoke_approval_changed_during_preflight"]

    exchange = _FakeExchange()
    state_path = tmp_path / "execution_state.json"
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=validator,
        state_path=state_path,
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "testnet_smoke_approval_changed_during_preflight"
    assert calls == 2
    assert exchange.requests == []
    assert exchange.leverage_updates == []
    assert not state_path.exists()


def test_approval_is_revalidated_before_leverage_and_bulk_submission(
    tmp_path,
):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="approval-drift-after-leverage",
    )
    calls = 0

    def validator(approval_id, settings):
        nonlocal calls
        calls += 1
        return (
            []
            if calls < 3
            else ["testnet_smoke_approval_changed_before_bulk_submission"]
        )

    exchange = _FakeExchange()
    state_path = tmp_path / "execution_state.json"
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=validator,
        state_path=state_path,
    )

    result = executor.submit_pair(_pair_intents(), config)
    state = json.loads(state_path.read_text(encoding="utf-8"))

    assert result.status == "pair_blocked"
    assert result.reason == "testnet_smoke_approval_changed_before_bulk_submission"
    assert calls == 3
    assert exchange.leverage_updates == []
    assert exchange.requests == []
    assert state["phase"] == "PRE_SUBMISSION_FAILED"
    assert state["entry_submit_attempted"] is False


def test_pair_executor_uses_one_bulk_action_after_all_guardrails_pass():
    config, secret = _test_config(submit_orders=True, order_approval_id="testnet-smoke-001")
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_submitted"
    assert len(exchange.requests) == 1
    assert len(exchange.requests[0]) == 2
    assert exchange.requests[0][0]["coin"] == "BTC"
    assert exchange.requests[0][1]["coin"] == "ETH"
    assert exchange.leverage_updates == [
        {"leverage": 1, "coin": "BTC", "is_cross": True},
        {"leverage": 1, "coin": "ETH", "is_cross": True},
    ]
    assert all(fill.status == "paper_submitted" for fill in result.fills)


def test_pair_executor_blocks_leverage_above_one_without_proven_baseline():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-leverage-unproven",
        requested_leverage=2,
    )
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert "hyperliquid_one_x_testnet_proof_missing" in result.reason
    assert "hyperliquid_leverage_scenario_approval_missing" in result.reason
    assert exchange.leverage_updates == []
    assert exchange.requests == []


def test_pair_executor_applies_exact_approved_isolated_leverage_to_both_legs():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-leverage-proven",
        requested_leverage=3,
        margin_mode="isolated",
        one_x_testnet_proof_id="one-x-proof-1",
        leverage_scenario_id="leverage-scenario-3x",
    )
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_submitted"
    assert exchange.leverage_updates == [
        {"leverage": 3, "coin": "BTC", "is_cross": False},
        {"leverage": 3, "coin": "ETH", "is_cross": False},
    ]
    assert len(exchange.requests) == 1


def test_pair_executor_blocks_invalid_precision_and_leg_notional_before_wallet():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-invalid-rules",
    )
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )
    invalid = (
        OrderIntent(
            market="BTC-USD",
            side="BUY",
            size=0.000161,
            limit_price=64_000.0,
        ),
        OrderIntent(
            market="ETH-USD",
            side="SELL",
            size=0.001,
            limit_price=3_000.0,
        ),
    )

    result = executor.submit_pair(invalid, config)

    assert result.status == "pair_blocked"
    assert "hyperliquid_size_precision_invalid:BTC" in result.reason
    assert "hyperliquid_leg_notional_below_10_usd:ETH" in result.reason
    assert exchange.leverage_updates == []
    assert exchange.requests == []


def test_pair_executor_enforces_market_max_leverage_and_isolated_only_mode():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-market-rules",
        requested_leverage=3,
        margin_mode="cross",
        one_x_testnet_proof_id="one-x-proof-1",
        leverage_scenario_id="scenario-3x-1",
    )
    meta = _market_meta()
    meta["universe"][0]["maxLeverage"] = 2
    meta["universe"][1]["onlyIsolated"] = True
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_CustomMetaSession(config.master_address or "", meta),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert "hyperliquid_requested_leverage_above_market_max:BTC" in result.reason
    assert "hyperliquid_market_requires_isolated_margin:ETH" in result.reason
    assert exchange.leverage_updates == []
    assert exchange.requests == []


def test_pair_executor_cancels_one_leg_acceptance_and_confirms_flat():
    config, secret = _test_config(submit_orders=True, order_approval_id="testnet-smoke-partial")
    exchange = _PartialExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_recovered_flat"
    assert result.reason == "hyperliquid_pair_submission_anomaly_recovered_flat"
    assert result.reconciled is True
    assert result.order_submission_performed is True
    assert result.live_trading_authorized is False
    assert result.recovery_actions == (
        "block_duplicate_entry_retry",
        "cancel_pair_open_orders",
        "confirm_pair_flat_and_order_free",
    )
    assert exchange.cancel_requests == [[{"coin": "BTC", "oid": 123}]]
    assert exchange.close_requests == []
    assert [fill.status for fill in result.fills] == [
        "paper_submitted",
        "paper_submission_rejected:Insufficient margin",
    ]


def test_pair_executor_reduce_only_flattens_an_orphan_fill():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-orphan",
    )
    exchange = _FilledPartialExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(), _positions(BTC=0.0001), _positions()],
        open_orders=[[], [], []],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_recovered_flat"
    assert result.reconciled is True
    assert "reduce_only_flatten:BTC" in result.recovery_actions
    assert exchange.cancel_requests == [[{"coin": "BTC", "oid": 321}]]
    assert exchange.close_requests == [{"coin": "BTC", "sz": 0.0001}]


def test_pair_executor_reconciles_unconfirmed_response_without_retrying_entry():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-unconfirmed",
    )
    exchange = _UnconfirmedExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(), _positions(), _positions()],
        open_orders=[
            [],
            [{"coin": "BTC", "oid": 901}, {"coin": "ETH", "oid": 902}],
            [],
        ],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_recovered_flat"
    assert len(exchange.requests) == 1
    assert exchange.cancel_requests == [[{"coin": "BTC", "oid": 901}, {"coin": "ETH", "oid": 902}]]
    assert result.recovery_actions[0] == "block_duplicate_entry_retry"


def test_pair_executor_blocks_entry_when_pair_market_is_already_open():
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-existing",
    )
    exchange = _FakeExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(BTC=0.001)],
        open_orders=[[]],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "hyperliquid_pair_markets_not_flat_before_entry"
    assert exchange.requests == []
    assert result.order_submission_performed is False


def test_pair_executor_persists_unconfirmed_state_before_returning(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-journal",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=state_path,
    )

    result = executor.submit_pair(_pair_intents(), config)
    state = executor._read_execution_state()

    assert result.status == "pair_submitted"
    assert result.state_path == str(state_path)
    assert state["state_integrity_valid"] is True
    assert state["phase"] == "AWAITING_EXCHANGE_CONFIRMATION"
    assert state["entry_submit_attempted"] is True
    assert state["exit_submit_attempted"] is False
    assert state["submit_kind"] == "entry"
    assert state["exchange_reference_ids"] == ["123", "124"]
    assert state["live_trading_authorized"] is False


def test_pair_executor_blocks_duplicate_entry_for_consumed_one_run_approval(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-one-run",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=state_path,
    )

    first = executor.submit_pair(_pair_intents(), config)
    duplicate = executor.submit_pair(_pair_intents(), config)

    assert first.status == "pair_submitted"
    assert duplicate.status == "pair_blocked"
    assert duplicate.reason == "hyperliquid_duplicate_entry_approval_consumed"
    assert len(exchange.requests) == 1
    state = executor._read_execution_state()
    assert state["state_integrity_valid"] is True
    assert state["duplicate_entry_blocked"] is True
    assert state["duplicate_entry_blocked_at_utc"]


def test_pair_executor_requires_journaled_entry_before_reduce_only_exit(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-exit-without-entry",
    )
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=tmp_path / "execution_state.json",
    )

    result = executor.submit_pair(_pair_exit_intents(), config)

    assert result.status == "pair_blocked"
    assert result.reason == "hyperliquid_exit_entry_state_missing"
    assert result.order_submission_performed is False
    assert exchange.requests == []


def test_pair_executor_submits_one_ioc_reduce_only_exit_and_blocks_replay(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-journaled-exit",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(BTC=0.00016, ETH=-0.0034)],
        open_orders=[[]],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=state_path,
    )
    entry_state = executor._new_execution_state(_pair_intents(), config)
    entry_state["phase"] = "AWAITING_EXCHANGE_CONFIRMATION"
    entry_state["entry_submit_attempted"] = True
    executor._persist_execution_state(entry_state)

    result = executor.submit_pair(_pair_exit_intents(), config)
    replay = executor.submit_pair(_pair_exit_intents(), config)

    assert result.status == "pair_submitted"
    assert result.order_submission_performed is True
    assert len(exchange.requests) == 1
    assert all(
        request["reduce_only"] is True
        and request["order_type"] == {"limit": {"tif": "Ioc"}}
        for request in exchange.requests[0]
    )
    assert replay.status == "pair_blocked"
    assert replay.reason == "hyperliquid_duplicate_exit_submission_blocked"


def test_pair_executor_blocks_exit_that_does_not_reduce_current_position(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-wrong-way-exit",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(BTC=0.00016, ETH=-0.0034)],
        open_orders=[[]],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=state_path,
    )
    entry_state = executor._new_execution_state(_pair_intents(), config)
    entry_state["phase"] = "AWAITING_EXCHANGE_CONFIRMATION"
    entry_state["entry_submit_attempted"] = True
    executor._persist_execution_state(entry_state)
    wrong_way = (
        OrderIntent(
            market="BTC-USD",
            side="BUY",
            size=0.00016,
            limit_price=64_000.0,
            reduce_only=True,
        ),
        _pair_exit_intents()[1],
    )

    result = executor.submit_pair(wrong_way, config)

    assert result.status == "pair_blocked"
    assert "hyperliquid_exit_side_not_risk_reducing:BTC" in result.reason
    assert result.order_submission_performed is False
    assert exchange.requests == []


def test_pair_executor_recovers_journaled_restart_without_entry_retry(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-restart",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(BTC=0.0001), _positions()],
        open_orders=[[{"coin": "BTC", "oid": 808}], []],
    )

    def expired_entry_approval_must_not_be_revalidated(approval_id, settings):
        raise AssertionError("restart recovery must not revalidate expired entry approval")

    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=expired_entry_approval_must_not_be_revalidated,
        state_path=state_path,
    )
    state = executor._new_execution_state(_pair_intents(), config)
    state["phase"] = "SUBMITTING_UNCONFIRMED"
    state["entry_submit_attempted"] = True
    state["exchange_reference_ids"] = ["808"]
    state["exchange_references"] = [{"market": "BTC", "order_id": "808"}]
    executor._persist_execution_state(state)

    result = executor.recover_incomplete_pair(config)
    persisted = json.loads(state_path.read_text(encoding="utf-8"))

    assert result.status == "pair_recovered_flat"
    assert result.reason == "hyperliquid_restart_recovered_flat"
    assert result.reconciled is True
    assert exchange.requests == []
    assert exchange.cancel_requests == [[{"coin": "BTC", "oid": 808}]]
    assert exchange.close_requests == [{"coin": "BTC", "sz": 0.0001}]
    assert persisted["phase"] == "FLAT_RECONCILED"
    assert persisted["reconciled"] is True


def test_pair_executor_blocks_tampered_restart_journal_before_exchange(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-tampered-state",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        state_path=state_path,
    )
    state = executor._new_execution_state(_pair_intents(), config)
    state["phase"] = "SUBMITTING_UNCONFIRMED"
    state["entry_submit_attempted"] = True
    executor._persist_execution_state(state)
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["intents"][0]["size"] = 99.0
    state_path.write_text(json.dumps(tampered), encoding="utf-8")

    result = executor.recover_incomplete_pair(config)

    assert result.status == "pair_recovery_blocked"
    assert "hyperliquid_pair_execution_state_hash_invalid" in result.reason
    assert exchange.requests == []
    assert exchange.cancel_requests == []
    assert exchange.close_requests == []


def test_pair_executor_recovers_unknown_bulk_response_without_retry(tmp_path):
    config, secret = _test_config(
        submit_orders=True,
        order_approval_id="testnet-smoke-timeout",
    )
    state_path = tmp_path / "execution_state.json"
    exchange = _UnknownResponseExchange()
    session = _SequencedAccountSession(
        config.master_address or "",
        states=[_positions(), _positions(), _positions()],
        open_orders=[[], [], []],
    )
    executor = HyperliquidTestnetPairExecutor(
        session=session,
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
        state_path=state_path,
    )

    result = executor.submit_pair(_pair_intents(), config)

    assert result.status == "pair_recovered_flat"
    assert result.reason == "hyperliquid_unknown_submission_recovered_flat"
    assert len(exchange.requests) == 1
    assert result.recovery_actions[0] == "block_duplicate_entry_retry"
    assert executor._read_execution_state()["phase"] == "FLAT_RECONCILED"


def test_pair_executor_rejects_same_direction_entry_before_exchange_call():
    config, secret = _test_config(submit_orders=True, order_approval_id="testnet-smoke-002")
    exchange = _FakeExchange()
    executor = HyperliquidTestnetPairExecutor(
        session=_RoleSession(config.master_address or ""),
        keychain_reader=lambda service, account: secret,
        exchange_factory=lambda wallet, settings: exchange,
        approval_validator=lambda approval_id, settings: [],
    )
    invalid = (
        OrderIntent(market="BTC-USD", side="BUY", size=0.00016, limit_price=64_000.0),
        OrderIntent(market="ETH-USD", side="BUY", size=0.0034, limit_price=3_000.0),
    )

    result = executor.submit_pair(invalid, config)

    assert result.status == "pair_blocked"
    assert "hyperliquid_pair_entry_legs_must_be_opposite" in result.reason
    assert exchange.requests == []


def test_static_execution_preflight_uses_keychain_configuration_not_raw_secret(monkeypatch):
    config, _ = _test_config()
    monkeypatch.setenv("HYPERLIQUID_NETWORK", "testnet")
    monkeypatch.setenv("HYPERLIQUID_TESTNET_BASE_URL", config.base_url)
    monkeypatch.setenv("HYPERLIQUID_MASTER_ADDRESS", config.master_address or "")
    monkeypatch.setenv("HYPERLIQUID_AGENT_ADDRESS", config.agent_address or "")
    monkeypatch.setenv("HYPERLIQUID_TESTNET_AGENT_KEYCHAIN_SERVICE", config.keychain_service or "")
    monkeypatch.setenv("HYPERLIQUID_TESTNET_SUBMIT_ORDERS", "false")
    monkeypatch.delenv("HYPERLIQUID_TESTNET_AGENT_PRIVATE_KEY", raising=False)

    status = hyperliquid_testnet_order_preflight_status()

    assert status["adapter_configured"] is True
    assert status["pair_executor_available"] is True
    assert status["account_address_configured"] is True
    assert status["agent_address_configured"] is True
    assert status["keychain_service_configured"] is True
    assert status["secret_key_configured"] is True
    assert status["submit_orders_enabled"] is False
    assert "hyperliquid_testnet_submit_orders_false" in status["blocker"]
