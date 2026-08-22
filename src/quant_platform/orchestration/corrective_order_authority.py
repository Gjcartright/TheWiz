"""Gate 00G exact, one-use authority for orders and account mutations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from threading import Lock
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
    EffectConsumptionReceipt,
    EffectKind,
    EffectPermit,
    EffectRequest,
)
from quant_platform.orchestration.venue_policy_registry import (
    VENUE_POLICY_SCHEMA_VERSION,
    venue_policy,
)

GATE00G_SCHEMA_VERSION = "thewiz.gate00g_order_authority.v1"
GATE00G_EFFECT_SCOPE = "gate00g_order_account_mutation"
HYPERLIQUID_TESTNET_ADAPTER_ID = (
    "quant_platform.hyperliquid_testnet:HyperliquidTestnetPairAdapter"
)
HYPERLIQUID_LIVE_CANARY_ADAPTER_ID = (
    "quant_platform.orchestration.corrective_live_canary_execution:"
    "run_live_canary_executor"
)
HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER_ADAPTER_ID = (
    "quant_platform.orchestration.corrective_testnet_collateral_transfer:"
    "run_testnet_collateral_transfer"
)
BINANCE_SPOT_TESTNET_ADAPTER_ID = (
    "quant_platform.binance_testnet:BinanceSpotTestnetOrderAdapter"
)
BINANCE_USDM_TESTNET_ADAPTER_ID = (
    "quant_platform.binance_testnet:BinanceUsdmTestnetOrderAdapter"
)
DYDX_TESTNET_ADAPTER_ID = (
    "quant_platform.dydx_sdk_order_adapter:DydxSdkOrderAdapter"
)

_ORDER_OPERATIONS_REQUIRING_NOTIONAL = frozenset(
    {"place_order", "bulk_orders", "close_position", "market_close"}
)
_ACCOUNT_OPERATIONS_REQUIRING_NOTIONAL = frozenset({"usd_class_transfer"})
_GATE00G_AUTHORITY_SEAL = object()


def _required(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} cannot be blank")
    return normalized


def _sha256(value: str, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{field_name} must be a lowercase sha256")
    return normalized


def decimal_text(value: object, *, field_name: str, positive: bool = False) -> str:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite decimal") from exc
    if not parsed.is_finite() or parsed < 0 or (positive and parsed <= 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field_name} must be finite and {qualifier}")
    return format(parsed, "f")


def exact_notional(size: object, price: object) -> str:
    quantity = Decimal(decimal_text(size, field_name="size", positive=True))
    reference_price = Decimal(
        decimal_text(price, field_name="reference_price", positive=True)
    )
    return format(quantity * reference_price, "f")


class OrderAuthorityIdentity(BaseModel):
    """Immutable run identity supplied independently from the effect permit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    intended_slot_id: str
    account_scope_id: str
    proposal_id: str
    policy_version: str = VENUE_POLICY_SCHEMA_VERSION
    model_version: str
    formula_version: str
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str

    @field_validator(
        "run_id",
        "intended_slot_id",
        "account_scope_id",
        "proposal_id",
        "policy_version",
        "model_version",
        "formula_version",
    )
    @classmethod
    def _text_required(cls, value: str, info) -> str:
        return _required(value, field_name=info.field_name)

    @field_validator(
        "source_fingerprint_sha256",
        "runtime_fingerprint_sha256",
        "configuration_fingerprint_sha256",
    )
    @classmethod
    def _hash_required(cls, value: str, info) -> str:
        return _sha256(value, field_name=info.field_name)


class OrderEffectSpec(BaseModel):
    """The exact economic effect that a permit is allowed to perform once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[GATE00G_SCHEMA_VERSION] = GATE00G_SCHEMA_VERSION
    run_id: str
    intended_slot_id: str
    effect_kind: EffectKind
    environment: Literal["testnet", "live"]
    adapter_id: str
    target: str
    operation: str
    venue_id: str
    product_lane_id: str
    account_scope_id: str
    instrument_id: str
    side: Literal["", "buy", "sell"] = ""
    size: str = "0"
    notional: str = "0"
    leverage: str = "0"
    reduce_only: bool = False
    proposal_id: str
    client_reference: str = ""
    policy_version: str
    model_version: str
    formula_version: str
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str

    @field_validator(
        "run_id",
        "intended_slot_id",
        "adapter_id",
        "target",
        "operation",
        "venue_id",
        "product_lane_id",
        "account_scope_id",
        "instrument_id",
        "proposal_id",
        "policy_version",
        "model_version",
        "formula_version",
    )
    @classmethod
    def _text_required(cls, value: str, info) -> str:
        return _required(value, field_name=info.field_name)

    @field_validator(
        "source_fingerprint_sha256",
        "runtime_fingerprint_sha256",
        "configuration_fingerprint_sha256",
    )
    @classmethod
    def _hash_required(cls, value: str, info) -> str:
        return _sha256(value, field_name=info.field_name)

    @field_validator("venue_id", "product_lane_id")
    @classmethod
    def _lowercase_identity(cls, value: str) -> str:
        return value.lower()

    @field_validator("instrument_id")
    @classmethod
    def _uppercase_instrument(cls, value: str) -> str:
        return value.upper()

    @field_validator("size", "notional", "leverage")
    @classmethod
    def _decimal_value(cls, value: str, info) -> str:
        return decimal_text(value, field_name=info.field_name)

    @model_validator(mode="after")
    def _effect_contract(self) -> OrderEffectSpec:
        if self.effect_kind not in {
            EffectKind.ORDER_SUBMISSION,
            EffectKind.ACCOUNT_MUTATION,
        }:
            raise ValueError("Gate 00G only accepts order or account-mutation effects")
        if self.effect_kind == EffectKind.ORDER_SUBMISSION:
            if not self.side:
                raise ValueError("order side is required")
            if self.operation in _ORDER_OPERATIONS_REQUIRING_NOTIONAL:
                decimal_text(self.size, field_name="size", positive=True)
                decimal_text(self.notional, field_name="notional", positive=True)
        else:
            if self.side:
                raise ValueError("account mutation cannot declare an order side")
            if self.operation in _ACCOUNT_OPERATIONS_REQUIRING_NOTIONAL:
                decimal_text(self.size, field_name="size", positive=True)
                decimal_text(self.notional, field_name="notional", positive=True)
        return self

    def payload_sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConsumedOrderAuthorization:
    """Unforgeable-in-process proof passed from the gate to a lowest sink."""

    receipt: EffectConsumptionReceipt
    spec_sha256: str
    _owner: CorrectiveOrderAuthority = field(repr=False, compare=False)


class CorrectiveOrderAuthority:
    """Consumes caller-supplied permits; it never manufactures authority at a sink."""

    gate00g_order_authority_enforced = True

    def __init__(
        self,
        *,
        authority: EffectAuthority,
        identity: OrderAuthorityIdentity,
        permits: tuple[EffectPermit, ...],
    ) -> None:
        self.authority = authority
        self.identity = identity
        self.permits = tuple(permits)
        self._instance_seal = _GATE00G_AUTHORITY_SEAL
        self._dispatch_lock = Lock()
        self._dispatched_permit_ids: set[str] = set()

    def spec(
        self,
        *,
        effect_kind: EffectKind,
        environment: Literal["testnet", "live"],
        adapter_id: str,
        target: str,
        operation: str,
        venue_id: str,
        product_lane_id: str,
        account_scope_id: str,
        instrument_id: str,
        side: str = "",
        size: object = "0",
        notional: object = "0",
        leverage: object = "0",
        reduce_only: bool = False,
        proposal_id: str,
        client_reference: str = "",
    ) -> OrderEffectSpec:
        account = _required(account_scope_id, field_name="account_scope_id")
        proposal = _required(proposal_id, field_name="proposal_id")
        if account != self.identity.account_scope_id:
            raise EffectAuthorityError("gate00g_account_scope_mismatch")
        if proposal != self.identity.proposal_id:
            raise EffectAuthorityError("gate00g_proposal_mismatch")
        normalized_side = str(side or "").strip().lower()
        return OrderEffectSpec(
            run_id=self.identity.run_id,
            intended_slot_id=self.identity.intended_slot_id,
            effect_kind=effect_kind,
            environment=environment,
            adapter_id=adapter_id,
            target=target,
            operation=operation,
            venue_id=venue_id,
            product_lane_id=product_lane_id,
            account_scope_id=account,
            instrument_id=instrument_id,
            side=normalized_side,
            size=decimal_text(size, field_name="size"),
            notional=decimal_text(notional, field_name="notional"),
            leverage=decimal_text(leverage, field_name="leverage"),
            reduce_only=bool(reduce_only),
            proposal_id=proposal,
            client_reference=str(client_reference or "").strip(),
            policy_version=self.identity.policy_version,
            model_version=self.identity.model_version,
            formula_version=self.identity.formula_version,
            source_fingerprint_sha256=self.identity.source_fingerprint_sha256,
            runtime_fingerprint_sha256=self.identity.runtime_fingerprint_sha256,
            configuration_fingerprint_sha256=(
                self.identity.configuration_fingerprint_sha256
            ),
        )

    def consume(self, spec: OrderEffectSpec) -> ConsumedOrderAuthorization:
        _validate_effect_policy(self.authority, spec)
        if not self.permits:
            raise EffectAuthorityError("gate00g_effect_permit_missing")
        permit = self._select_permit(spec)
        if permit.max_units != 1:
            raise EffectAuthorityError("gate00g_permit_units_not_exact")
        if Decimal(permit.max_notional) != Decimal(spec.notional):
            raise EffectAuthorityError("gate00g_permit_notional_not_exact")
        if Decimal(permit.max_leverage) != Decimal(spec.leverage):
            raise EffectAuthorityError("gate00g_permit_leverage_not_exact")
        request = EffectRequest(
            run_id=spec.run_id,
            intended_slot_id=spec.intended_slot_id,
            effect_kind=spec.effect_kind,
            target=spec.target,
            operation=spec.operation,
            effect_scope=GATE00G_EFFECT_SCOPE,
            payload_sha256=spec.payload_sha256(),
            venue_id=spec.venue_id,
            product_lane_id=spec.product_lane_id,
            account_scope_id=spec.account_scope_id,
            instrument_ids=(spec.instrument_id,),
            side=spec.side,
            actual_units=1,
            actual_notional=spec.notional,
            actual_leverage=spec.leverage,
            proposal_id=spec.proposal_id,
            policy_version=spec.policy_version,
            model_version=spec.model_version,
            formula_version=spec.formula_version,
            source_fingerprint_sha256=spec.source_fingerprint_sha256,
            runtime_fingerprint_sha256=spec.runtime_fingerprint_sha256,
            configuration_fingerprint_sha256=(
                spec.configuration_fingerprint_sha256
            ),
        )
        receipt = self.authority.consume(permit, request)
        return ConsumedOrderAuthorization(
            receipt=receipt,
            spec_sha256=spec.payload_sha256(),
            _owner=self,
        )

    def consume_all(
        self, specs: tuple[OrderEffectSpec, ...]
    ) -> tuple[ConsumedOrderAuthorization, ...]:
        if not specs:
            raise EffectAuthorityError("gate00g_effect_specs_missing")
        return tuple(self.consume(spec) for spec in specs)

    def claim_dispatch(
        self,
        authorization: ConsumedOrderAuthorization | None,
        spec: OrderEffectSpec,
    ) -> None:
        """Atomically bind one consumed permit to one final sink dispatch."""

        _validate_effect_policy(self.authority, spec)
        require_consumed_authorization(
            authorization,
            owner=self,
            spec=spec,
        )
        assert authorization is not None
        permit_id = authorization.receipt.permit_id
        permit = self._permit_by_id(permit_id)
        now = self.authority.clock()
        if now < permit.issued_at_utc:
            raise EffectAuthorityError(
                "gate00g_consumed_authorization_not_yet_valid"
            )
        if now >= permit.expires_at_utc:
            raise EffectAuthorityError("gate00g_consumed_authorization_stale")
        with self._dispatch_lock:
            if permit_id in self._dispatched_permit_ids:
                raise EffectAuthorityError(
                    "gate00g_consumed_authorization_dispatch_replayed"
                )
            self._dispatched_permit_ids.add(permit_id)

    def _permit_by_id(self, permit_id: str) -> EffectPermit:
        for permit in self.permits:
            if permit.permit_id == permit_id:
                return permit
        raise EffectAuthorityError("gate00g_consumed_authorization_permit_missing")

    def _select_permit(self, spec: OrderEffectSpec) -> EffectPermit:
        scoped = [
            permit
            for permit in self.permits
            if permit.effect_kind == spec.effect_kind
            and permit.operation == spec.operation
            and permit.target == spec.target
            and permit.instrument_ids == (spec.instrument_id,)
        ]
        exact = [
            permit
            for permit in scoped
            if permit.payload_sha256 == spec.payload_sha256()
            and permit.side == spec.side
            and permit.run_id == spec.run_id
            and permit.intended_slot_id == spec.intended_slot_id
            and permit.venue_id == spec.venue_id
            and permit.product_lane_id == spec.product_lane_id
            and permit.account_scope_id == spec.account_scope_id
            and permit.proposal_id == spec.proposal_id
        ]
        candidates = exact or scoped
        if not candidates and len(self.permits) == 1:
            candidates = [self.permits[0]]
        if not candidates:
            raise EffectAuthorityError("gate00g_exact_effect_permit_missing")
        issued = [
            permit
            for permit in candidates
            if self.authority.state(permit.permit_id) == "ISSUED"
        ]
        if issued:
            return issued[0]
        return candidates[0]


def issue_gate00g_permit(
    *,
    authority: EffectAuthority,
    spec: OrderEffectSpec,
    ttl_seconds: int = 60,
) -> EffectPermit:
    """Issue through the primitive only after the current venue policy permits it."""

    _validate_effect_policy(authority, spec)
    return authority.issue(
        run_id=spec.run_id,
        intended_slot_id=spec.intended_slot_id,
        effect_kind=spec.effect_kind,
        target=spec.target,
        operation=spec.operation,
        effect_scope=GATE00G_EFFECT_SCOPE,
        payload_sha256=spec.payload_sha256(),
        policy_version=spec.policy_version,
        source_fingerprint_sha256=spec.source_fingerprint_sha256,
        runtime_fingerprint_sha256=spec.runtime_fingerprint_sha256,
        configuration_fingerprint_sha256=spec.configuration_fingerprint_sha256,
        max_units=1,
        ttl_seconds=ttl_seconds,
        max_notional=spec.notional,
        max_leverage=spec.leverage,
        venue_id=spec.venue_id,
        product_lane_id=spec.product_lane_id,
        account_scope_id=spec.account_scope_id,
        instrument_ids=(spec.instrument_id,),
        side=spec.side,
        proposal_id=spec.proposal_id,
        model_version=spec.model_version,
        formula_version=spec.formula_version,
    )


def require_consumed_authorization(
    authorization: ConsumedOrderAuthorization | None,
    *,
    owner: CorrectiveOrderAuthority,
    spec: OrderEffectSpec,
) -> None:
    if authorization is None:
        raise EffectAuthorityError("gate00g_consumed_authorization_missing")
    if authorization._owner is not owner:
        raise EffectAuthorityError("gate00g_consumed_authorization_owner_mismatch")
    if authorization.spec_sha256 != spec.payload_sha256():
        raise EffectAuthorityError("gate00g_consumed_authorization_scope_mismatch")
    permit = owner._permit_by_id(authorization.receipt.permit_id)
    if authorization.receipt.nonce != permit.nonce:
        raise EffectAuthorityError("gate00g_consumed_authorization_nonce_mismatch")
    if authorization.receipt.effect_kind != spec.effect_kind:
        raise EffectAuthorityError("gate00g_consumed_authorization_kind_mismatch")
    if owner.authority.state(authorization.receipt.permit_id) != "CONSUMED":
        raise EffectAuthorityError("gate00g_consumed_authorization_state_invalid")


def claim_effect_dispatch(
    authorization: ConsumedOrderAuthorization | None,
    *,
    owner: CorrectiveOrderAuthority,
    spec: OrderEffectSpec,
) -> None:
    owner.claim_dispatch(authorization, spec)


def claim_effect_dispatch_all(
    authorizations: tuple[ConsumedOrderAuthorization, ...],
    *,
    owner: CorrectiveOrderAuthority,
    specs: tuple[OrderEffectSpec, ...],
) -> None:
    """Claim every one-use dispatch before a multi-effect venue call."""

    if not specs or len(authorizations) != len(specs):
        raise EffectAuthorityError("gate00g_batch_authorization_cardinality_mismatch")
    for authorization, spec in zip(authorizations, specs, strict=True):
        claim_effect_dispatch(authorization, owner=owner, spec=spec)


def require_order_authority(
    authority: CorrectiveOrderAuthority | None,
) -> CorrectiveOrderAuthority:
    if authority is None:
        raise EffectAuthorityError("gate00g_order_authority_missing")
    if (
        type(authority) is not CorrectiveOrderAuthority
        or getattr(authority, "_instance_seal", None) is not _GATE00G_AUTHORITY_SEAL
    ):
        raise EffectAuthorityError("gate00g_order_authority_type_invalid")
    return authority


def _validate_effect_policy(
    authority: EffectAuthority,
    spec: OrderEffectSpec,
) -> None:
    profile_name = "".join(
        character
        for character in authority.profile.name.strip().upper()
        if character.isalnum()
    )
    if profile_name.startswith("PHASE00"):
        raise EffectAuthorityError("gate00g_phase00_profile_forbidden")
    try:
        policy = venue_policy(spec.product_lane_id)
    except KeyError as exc:
        raise EffectAuthorityError("gate00g_unknown_venue_lane") from exc
    if str(policy.venue).lower() != spec.venue_id:
        raise EffectAuthorityError("gate00g_venue_lane_mismatch")
    if not bool(policy.activation_enabled):
        raise EffectAuthorityError("gate00g_venue_activation_disabled")
    if not bool(policy.authenticated_access_allowed):
        raise EffectAuthorityError("gate00g_authenticated_access_denied")
    if spec.effect_kind == EffectKind.ORDER_SUBMISSION:
        if not bool(policy.order_submission_allowed):
            raise EffectAuthorityError("gate00g_order_submission_denied")
    elif not bool(policy.account_mutation_allowed):
        raise EffectAuthorityError("gate00g_account_mutation_denied")
    if spec.environment == "testnet":
        if not bool(policy.testnet_progression_allowed):
            raise EffectAuthorityError("gate00g_testnet_lane_denied")
        if spec.adapter_id not in tuple(policy.allowed_testnet_adapters):
            raise EffectAuthorityError("gate00g_testnet_adapter_denied")
    elif not bool(policy.live_enabled):
        raise EffectAuthorityError("gate00g_live_lane_denied")
