"""Permit-bound credential, public/authenticated-network, and credit effects."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, TypeVar
from urllib.parse import urlsplit

from quant_platform.orchestration.corrective_effect_guard import (
    _authorized_credential_effect,
    _authorized_keychain_effect,
    _authorized_network_effect,
)
from quant_platform.orchestration.effect_authority import (
    EffectAuthority,
    EffectAuthorityError,
    EffectAuthorityProfile,
    EffectConsumptionReceipt,
    EffectKind,
    EffectRequest,
)

EXTERNAL_EFFECT_POLICY_VERSION = "thewiz.external_effect_transaction.v1"
RESEARCH_EXTERNAL_EFFECT_PROFILE = EffectAuthorityProfile(
    name="RESEARCH_CREDENTIAL_PUBLIC_NETWORK_CREDIT_ONLY",
    allowed_effects=frozenset(
        {
            EffectKind.CREDENTIAL_ACCESS,
            EffectKind.PUBLIC_NETWORK,
            EffectKind.AUTHENTICATED_NETWORK,
            EffectKind.CREDIT_SPEND,
        }
    ),
    max_ttl_seconds=120,
)

T = TypeVar("T")


@dataclass(frozen=True)
class ExternalEffectCallContract:
    """One policy-reviewed provider operation and its exact unit contract."""

    operation: str
    method: str
    target: str
    credit_units_per_request: int | None


@dataclass(frozen=True)
class ExternalEffectIssuer:
    """Supervisor delegation that cannot spend until bound to a reservation."""

    authority: EffectAuthority
    run_id: str
    intended_slot_id: str
    policy_version: str
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str
    provider_id: str
    account_scope_id: str
    allowed_targets: frozenset[str]
    allowed_credential_keys: frozenset[str]
    max_total_requests: int
    max_total_credits: int
    allowed_call_contracts: frozenset[ExternalEffectCallContract] = frozenset()


@dataclass
class ExternalEffectSession:
    """Supervisor-owned budget and identity for one vendor reservation."""

    authority: EffectAuthority
    run_id: str
    intended_slot_id: str
    policy_version: str
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str
    provider_id: str
    account_scope_id: str
    reservation_id: str
    reservation_sha256: str
    reservation_binding_id: str
    allowed_targets: frozenset[str]
    allowed_credential_keys: frozenset[str]
    max_total_requests: int
    max_total_credits: int
    allowed_call_contracts: frozenset[ExternalEffectCallContract] = frozenset()
    consumed_requests: int = 0
    consumed_credits: int = 0
    credential_receipts: list[EffectConsumptionReceipt] = field(default_factory=list)
    network_receipts: list[EffectConsumptionReceipt] = field(default_factory=list)
    credit_receipts: list[EffectConsumptionReceipt] = field(default_factory=list)
    _credentials_read: set[str] = field(default_factory=set, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


_CURRENT_EXTERNAL_EFFECT_SESSION: ContextVar[ExternalEffectSession | None] = ContextVar(
    "thewiz_current_external_effect_session",
    default=None,
)
_CURRENT_EXTERNAL_EFFECT_ISSUER: ContextVar[ExternalEffectIssuer | None] = ContextVar(
    "thewiz_current_external_effect_issuer",
    default=None,
)
_CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE: ContextVar[
    dict[str, ExternalEffectIssuer] | None
] = ContextVar(
    "thewiz_current_external_effect_issuer_bundle",
    default=None,
)


@contextmanager
def external_effect_issuer_session(
    *,
    authority: EffectAuthority,
    run_id: str,
    intended_slot_id: str,
    source_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    configuration_fingerprint_sha256: str,
    provider_id: str,
    account_scope_id: str,
    allowed_targets: frozenset[str],
    allowed_credential_keys: frozenset[str],
    max_total_requests: int,
    max_total_credits: int,
    allowed_call_contracts: frozenset[ExternalEffectCallContract] = frozenset(),
    policy_version: str = EXTERNAL_EFFECT_POLICY_VERSION,
) -> Iterator[ExternalEffectIssuer]:
    """Install authority metadata without granting an unreserved vendor call."""

    if (
        _CURRENT_EXTERNAL_EFFECT_ISSUER.get() is not None
        or _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.get() is not None
    ):
        raise EffectAuthorityError("nested_external_effect_issuer_denied")
    normalized_targets = frozenset(_normalize_target(value) for value in allowed_targets)
    normalized_keys = frozenset(value.strip().upper() for value in allowed_credential_keys)
    normalized_contracts = _normalize_call_contracts(allowed_call_contracts)
    if not normalized_targets or "" in normalized_keys:
        raise ValueError("external issuer targets are required")
    if any(
        contract.target not in normalized_targets for contract in normalized_contracts
    ):
        raise ValueError("external issuer call contract target is not allowed")
    if max_total_requests <= 0 or max_total_credits < 0:
        raise ValueError("external issuer budgets are invalid")
    for value, field_name in (
        (run_id, "run_id"),
        (intended_slot_id, "intended_slot_id"),
        (provider_id, "provider_id"),
        (account_scope_id, "account_scope_id"),
        (policy_version, "policy_version"),
    ):
        _require_text(value, field_name)
    issuer = ExternalEffectIssuer(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot_id,
        policy_version=policy_version,
        source_fingerprint_sha256=source_fingerprint_sha256,
        runtime_fingerprint_sha256=runtime_fingerprint_sha256,
        configuration_fingerprint_sha256=configuration_fingerprint_sha256,
        provider_id=provider_id,
        account_scope_id=account_scope_id,
        allowed_targets=normalized_targets,
        allowed_credential_keys=normalized_keys,
        max_total_requests=max_total_requests,
        max_total_credits=max_total_credits,
        allowed_call_contracts=normalized_contracts,
    )
    token = _CURRENT_EXTERNAL_EFFECT_ISSUER.set(issuer)
    try:
        yield issuer
    finally:
        _CURRENT_EXTERNAL_EFFECT_ISSUER.reset(token)


@contextmanager
def external_effect_issuer_bundle_session(
    issuers: tuple[ExternalEffectIssuer, ...],
) -> Iterator[dict[str, ExternalEffectIssuer]]:
    """Install multiple provider delegations without selecting one implicitly."""

    if (
        _CURRENT_EXTERNAL_EFFECT_ISSUER.get() is not None
        or _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.get() is not None
    ):
        raise EffectAuthorityError("nested_external_effect_issuer_denied")
    if not issuers:
        raise ValueError("external issuer bundle cannot be empty")
    providers = {issuer.provider_id: issuer for issuer in issuers}
    if len(providers) != len(issuers):
        raise ValueError("external issuer bundle provider ids must be unique")
    authority = issuers[0].authority
    run_id = issuers[0].run_id
    intended_slot_id = issuers[0].intended_slot_id
    if any(
        issuer.authority is not authority
        or issuer.run_id != run_id
        or issuer.intended_slot_id != intended_slot_id
        for issuer in issuers
    ):
        raise ValueError("external issuer bundle identity mismatch")
    token = _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.set(providers)
    try:
        yield providers
    finally:
        _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.reset(token)


@contextmanager
def external_effect_provider_session(
    provider_id: str,
) -> Iterator[ExternalEffectIssuer]:
    """Select one exact provider delegation for a bounded stage."""

    normalized = provider_id.strip()
    active = _CURRENT_EXTERNAL_EFFECT_ISSUER.get()
    if active is not None:
        if active.provider_id != normalized:
            raise EffectAuthorityError("external_effect_provider_mismatch")
        yield active
        return
    bundle = _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.get()
    if bundle is None:
        raise EffectAuthorityError("external_effect_issuer_missing")
    issuer = bundle.get(normalized)
    if issuer is None:
        raise EffectAuthorityError("external_effect_provider_not_delegated")
    token = _CURRENT_EXTERNAL_EFFECT_ISSUER.set(issuer)
    try:
        yield issuer
    finally:
        _CURRENT_EXTERNAL_EFFECT_ISSUER.reset(token)


def current_external_effect_issuer(
    provider_id: str | None = None,
) -> ExternalEffectIssuer | None:
    active = _CURRENT_EXTERNAL_EFFECT_ISSUER.get()
    if active is not None:
        if provider_id is None or active.provider_id == provider_id:
            return active
        return None
    bundle = _CURRENT_EXTERNAL_EFFECT_ISSUER_BUNDLE.get()
    if bundle is None:
        return None
    if provider_id is not None:
        return bundle.get(provider_id)
    return next(iter(bundle.values())) if len(bundle) == 1 else None


@contextmanager
def reserved_external_effect_session(
    *,
    reservation_id: str,
    reservation_sha256: str,
    max_total_requests: int,
    max_total_credits: int,
    reservation_binding_id: str | None = None,
) -> Iterator[ExternalEffectSession]:
    """Derive exact call authority from the current supervisor and reservation."""

    issuer = current_external_effect_issuer()
    if issuer is None:
        raise EffectAuthorityError("external_effect_issuer_missing")
    if max_total_requests > issuer.max_total_requests:
        raise EffectAuthorityError("external_issuer_request_budget_exceeded")
    if max_total_credits > issuer.max_total_credits:
        raise EffectAuthorityError("external_issuer_credit_budget_exceeded")
    with external_effect_authority_session(
        authority=issuer.authority,
        run_id=issuer.run_id,
        intended_slot_id=issuer.intended_slot_id,
        source_fingerprint_sha256=issuer.source_fingerprint_sha256,
        runtime_fingerprint_sha256=issuer.runtime_fingerprint_sha256,
        configuration_fingerprint_sha256=issuer.configuration_fingerprint_sha256,
        provider_id=issuer.provider_id,
        account_scope_id=issuer.account_scope_id,
        reservation_id=reservation_id,
        reservation_sha256=reservation_sha256,
        allowed_targets=issuer.allowed_targets,
        allowed_credential_keys=issuer.allowed_credential_keys,
        max_total_requests=max_total_requests,
        max_total_credits=max_total_credits,
        allowed_call_contracts=issuer.allowed_call_contracts,
        reservation_binding_id=reservation_binding_id,
        policy_version=issuer.policy_version,
    ) as session:
        yield session


@contextmanager
def external_effect_authority_session(
    *,
    authority: EffectAuthority,
    run_id: str,
    intended_slot_id: str,
    source_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    configuration_fingerprint_sha256: str,
    provider_id: str,
    account_scope_id: str,
    reservation_id: str,
    reservation_sha256: str,
    allowed_targets: frozenset[str],
    allowed_credential_keys: frozenset[str],
    max_total_requests: int,
    max_total_credits: int,
    allowed_call_contracts: frozenset[ExternalEffectCallContract] = frozenset(),
    reservation_binding_id: str | None = None,
    policy_version: str = EXTERNAL_EFFECT_POLICY_VERSION,
) -> Iterator[ExternalEffectSession]:
    """Install exact external authority after an immutable budget reservation."""

    if _CURRENT_EXTERNAL_EFFECT_SESSION.get() is not None:
        raise EffectAuthorityError("nested_external_effect_session_denied")
    if max_total_requests <= 0 or max_total_credits < 0:
        raise ValueError("external effect budgets are invalid")
    normalized_targets = frozenset(_normalize_target(value) for value in allowed_targets)
    normalized_keys = frozenset(value.strip().upper() for value in allowed_credential_keys)
    normalized_contracts = _normalize_call_contracts(allowed_call_contracts)
    if not normalized_targets or "" in normalized_keys:
        raise ValueError("external effect targets are required")
    if any(
        contract.target not in normalized_targets for contract in normalized_contracts
    ):
        raise ValueError("external effect call contract target is not allowed")
    _require_text(run_id, "run_id")
    _require_text(intended_slot_id, "intended_slot_id")
    _require_text(provider_id, "provider_id")
    _require_text(account_scope_id, "account_scope_id")
    _require_text(reservation_id, "reservation_id")
    _require_sha256(reservation_sha256, "reservation_sha256")
    if reservation_binding_id is None:
        reservation = authority.register_external_reservation(
            run_id=run_id,
            intended_slot_id=intended_slot_id,
            provider_id=provider_id,
            account_scope_id=account_scope_id,
            reservation_id=reservation_id,
            reservation_sha256=reservation_sha256,
            max_total_requests=max_total_requests,
            max_total_credits=max_total_credits,
        )
    else:
        reservation = authority.require_open_external_reservation(
            binding_id=reservation_binding_id,
            run_id=run_id,
            intended_slot_id=intended_slot_id,
            provider_id=provider_id,
            account_scope_id=account_scope_id,
            reservation_id=reservation_id,
            reservation_sha256=reservation_sha256,
            max_total_requests=max_total_requests,
            max_total_credits=max_total_credits,
        )
    session = ExternalEffectSession(
        authority=authority,
        run_id=run_id,
        intended_slot_id=intended_slot_id,
        policy_version=policy_version,
        source_fingerprint_sha256=source_fingerprint_sha256,
        runtime_fingerprint_sha256=runtime_fingerprint_sha256,
        configuration_fingerprint_sha256=configuration_fingerprint_sha256,
        provider_id=provider_id,
        account_scope_id=account_scope_id,
        reservation_id=reservation_id,
        reservation_sha256=reservation_sha256,
        reservation_binding_id=reservation.binding_id,
        allowed_targets=normalized_targets,
        allowed_credential_keys=normalized_keys,
        max_total_requests=max_total_requests,
        max_total_credits=max_total_credits,
        allowed_call_contracts=normalized_contracts,
    )
    token = _CURRENT_EXTERNAL_EFFECT_SESSION.set(session)
    try:
        yield session
    except BaseException:
        authority.close_external_reservation(
            reservation.binding_id,
            status="FAILED",
        )
        raise
    else:
        authority.close_external_reservation(
            reservation.binding_id,
            status="CLOSED",
        )
    finally:
        _CURRENT_EXTERNAL_EFFECT_SESSION.reset(token)


def current_external_effect_session() -> ExternalEffectSession | None:
    return _CURRENT_EXTERNAL_EFFECT_SESSION.get()


def read_authorized_credential(
    key: str,
    *,
    reader: Callable[[str], str | None] | None = None,
) -> str:
    """Consume exact credential authority and return a non-empty secret in memory."""

    session = _require_session()
    normalized_key = key.strip().upper()
    if normalized_key not in session.allowed_credential_keys:
        raise EffectAuthorityError("external_credential_key_denied")
    material = _effect_material(
        session,
        target=f"environment:{normalized_key}",
        operation="read",
        payload={"credential_key": normalized_key},
    )
    with session._lock:
        if normalized_key in session._credentials_read:
            raise EffectAuthorityError("external_credential_reuse_denied")
        permit = session.authority.issue(
            **material,
            effect_kind=EffectKind.CREDENTIAL_ACCESS,
            max_units=1,
            account_scope_id=session.account_scope_id,
        )
        receipt = session.authority.consume(
            permit,
            EffectRequest.from_permit(permit, actual_units=1),
        )
        session.credential_receipts.append(receipt)
        try:
            with _authorized_credential_effect(key=normalized_key):
                secret = (reader or os.getenv)(normalized_key)
        except BaseException as exc:
            session.authority.record_outcome(
                permit.permit_id,
                status="UNKNOWN",
                detail_sha256=_outcome_hash("credential_reader_exception", exc),
            )
            raise
        if not secret or not secret.strip():
            session.authority.record_outcome(
                permit.permit_id,
                status="FAILED",
                detail_sha256=_outcome_hash("credential_missing"),
            )
            raise EffectAuthorityError("authorized_credential_missing")
        session.authority.record_outcome(
            permit.permit_id,
            status="SUCCEEDED",
            detail_sha256=_outcome_hash("credential_loaded"),
        )
        session._credentials_read.add(normalized_key)
        return secret.strip()


def read_authorized_keychain_credential(
    credential_id: str,
    *,
    service: str,
    account: str,
    reader: Callable[[str, str], str | None],
) -> str:
    """Consume exact credential authority for one macOS Keychain item."""

    session = _require_session()
    normalized_id = credential_id.strip().upper()
    normalized_service = service.strip()
    normalized_account = account.strip()
    if normalized_id not in session.allowed_credential_keys:
        raise EffectAuthorityError("external_credential_key_denied")
    if not normalized_service or not normalized_account:
        raise EffectAuthorityError("external_keychain_item_invalid")
    material = _effect_material(
        session,
        target=f"keychain:{normalized_id}",
        operation="read_keychain_item",
        payload={
            "credential_id": normalized_id,
            "service": normalized_service,
            "account": normalized_account,
        },
    )
    with session._lock:
        if normalized_id in session._credentials_read:
            raise EffectAuthorityError("external_credential_reuse_denied")
        permit = session.authority.issue(
            **material,
            effect_kind=EffectKind.CREDENTIAL_ACCESS,
            max_units=1,
            account_scope_id=session.account_scope_id,
        )
        receipt = session.authority.consume(
            permit,
            EffectRequest.from_permit(permit, actual_units=1),
        )
        session.credential_receipts.append(receipt)
        try:
            with _authorized_keychain_effect(
                service=normalized_service,
                account=normalized_account,
            ):
                secret = reader(normalized_service, normalized_account)
        except BaseException as exc:
            session.authority.record_outcome(
                permit.permit_id,
                status="UNKNOWN",
                detail_sha256=_outcome_hash("keychain_reader_exception", exc),
            )
            raise
        if not secret or not secret.strip():
            session.authority.record_outcome(
                permit.permit_id,
                status="FAILED",
                detail_sha256=_outcome_hash("keychain_credential_missing"),
            )
            raise EffectAuthorityError("authorized_credential_missing")
        session.authority.record_outcome(
            permit.permit_id,
            status="SUCCEEDED",
            detail_sha256=_outcome_hash("keychain_credential_loaded"),
        )
        session._credentials_read.add(normalized_id)
        return secret.strip()


def run_authorized_credit_call(
    *,
    target: str,
    operation: str,
    method: str | None = None,
    request_payload: bytes,
    request_count: int,
    credit_cost: int,
    callback: Callable[[], T],
    result_recorder: Callable[[T], str] | None = None,
) -> T:
    """Consume one exact network+credit budget and run the bounded callback."""

    return _run_authorized_network_call(
        target=target,
        operation=operation,
        method=method,
        request_payload=request_payload,
        request_count=request_count,
        credit_cost=credit_cost,
        callback=callback,
        result_recorder=result_recorder,
        network_effect_kind=EffectKind.AUTHENTICATED_NETWORK,
        require_credential=True,
    )


def run_authorized_public_call(
    *,
    target: str,
    operation: str,
    method: str | None = None,
    request_payload: bytes,
    request_count: int,
    callback: Callable[[], T],
    result_recorder: Callable[[T], str] | None = None,
) -> T:
    """Consume one exact public-network budget and run the bounded callback."""

    return _run_authorized_network_call(
        target=target,
        operation=operation,
        method=method,
        request_payload=request_payload,
        request_count=request_count,
        credit_cost=0,
        callback=callback,
        result_recorder=result_recorder,
        network_effect_kind=EffectKind.PUBLIC_NETWORK,
        require_credential=False,
    )


def _run_authorized_network_call(
    *,
    target: str,
    operation: str,
    method: str | None,
    request_payload: bytes,
    request_count: int,
    credit_cost: int,
    callback: Callable[[], T],
    result_recorder: Callable[[T], str] | None,
    network_effect_kind: EffectKind,
    require_credential: bool,
) -> T:
    """Apply the common one-use contract to one network callback."""

    session = _require_session()
    if network_effect_kind not in {
        EffectKind.PUBLIC_NETWORK,
        EffectKind.AUTHENTICATED_NETWORK,
    }:
        raise ValueError("external network effect kind is invalid")
    normalized_target = _normalize_target(target)
    if normalized_target not in session.allowed_targets:
        raise EffectAuthorityError("external_network_target_denied")
    normalized_operation = operation.strip()
    _require_text(normalized_operation, "operation")
    normalized_method = method.strip().upper() if method is not None else ""
    if normalized_method and normalized_method not in {"GET", "POST"}:
        raise ValueError("external call method must be GET or POST")
    if request_count <= 0 or credit_cost < 0:
        raise ValueError("request_count must be positive and credit_cost non-negative")
    if session.allowed_call_contracts:
        matching_contracts = [
            contract
            for contract in session.allowed_call_contracts
            if contract.operation == normalized_operation
            and contract.method == normalized_method
            and contract.target == normalized_target
        ]
        if len(matching_contracts) != 1:
            raise EffectAuthorityError("external_call_contract_denied")
        contract = matching_contracts[0]
        if (
            contract.credit_units_per_request is not None
            and credit_cost != contract.credit_units_per_request * request_count
        ):
            raise EffectAuthorityError("external_call_credit_contract_mismatch")
    if (
        require_credential
        and session.allowed_credential_keys
        and not session._credentials_read
    ):
        raise EffectAuthorityError("external_credential_not_authorized")
    host = urlsplit(normalized_target).hostname
    if not host:
        raise EffectAuthorityError("external_network_target_host_missing")
    payload_hash = sha256(request_payload).hexdigest()
    credit_material = _effect_material(
        session,
        target=normalized_target,
        operation=normalized_operation,
        payload_sha256=payload_hash,
        effect_scope=f"credit:{session.reservation_id}",
    )
    network_material = _effect_material(
        session,
        target=normalized_target,
        operation=normalized_operation,
        payload_sha256=payload_hash,
        effect_scope=f"network:{session.reservation_id}",
    )
    with session._lock:
        if session.consumed_requests + request_count > session.max_total_requests:
            raise EffectAuthorityError("external_request_budget_exhausted")
        if session.consumed_credits + credit_cost > session.max_total_credits:
            raise EffectAuthorityError("external_credit_budget_exhausted")
        credit_permit = (
            session.authority.issue(
                **credit_material,
                effect_kind=EffectKind.CREDIT_SPEND,
                max_units=credit_cost,
                account_scope_id=session.account_scope_id,
            )
            if credit_cost > 0
            else None
        )
        network_permit = session.authority.issue(
            **network_material,
            effect_kind=network_effect_kind,
            max_units=request_count,
            account_scope_id=session.account_scope_id,
        )
        credit_receipt = (
            session.authority.consume(
                credit_permit,
                EffectRequest.from_permit(credit_permit, actual_units=credit_cost),
            )
            if credit_permit is not None
            else None
        )
        network_receipt = session.authority.consume(
            network_permit,
            EffectRequest.from_permit(network_permit, actual_units=request_count),
        )
        session.consumed_credits += credit_cost
        session.consumed_requests += request_count
        if credit_receipt is not None:
            session.credit_receipts.append(credit_receipt)
        session.network_receipts.append(network_receipt)
    try:
        with _authorized_network_effect(hosts=frozenset({host.lower()})):
            result = callback()
        evidence_sha256 = (
            _validated_sha256(result_recorder(result))
            if result_recorder is not None
            else ""
        )
    except BaseException as exc:
        detail = _outcome_hash("external_call_exception", exc)
        session.authority.record_outcome(
            network_permit.permit_id,
            status="UNKNOWN",
            detail_sha256=detail,
        )
        if credit_permit is not None:
            session.authority.record_outcome(
                credit_permit.permit_id,
                status="UNKNOWN",
                detail_sha256=detail,
            )
        raise
    detail = evidence_sha256 or _outcome_hash("external_call_completed")
    session.authority.record_outcome(
        network_permit.permit_id,
        status="SUCCEEDED",
        detail_sha256=detail,
    )
    if credit_permit is not None:
        session.authority.record_outcome(
            credit_permit.permit_id,
            status="SUCCEEDED",
            detail_sha256=detail,
        )
    return result


def _validated_sha256(value: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise EffectAuthorityError("external_result_evidence_sha256_invalid")
    return normalized


def _require_session() -> ExternalEffectSession:
    session = current_external_effect_session()
    if session is None:
        raise EffectAuthorityError("external_effect_authority_session_missing")
    return session


def _effect_material(
    session: ExternalEffectSession,
    *,
    target: str,
    operation: str,
    payload: dict[str, Any] | None = None,
    payload_sha256: str | None = None,
    effect_scope: str | None = None,
) -> dict[str, Any]:
    if payload_sha256 is None:
        payload_sha256 = sha256(_canonical_payload(payload or {}).encode("utf-8")).hexdigest()
    return {
        "run_id": session.run_id,
        "intended_slot_id": session.intended_slot_id,
        "target": target,
        "operation": operation,
        "effect_scope": effect_scope or f"reservation:{session.reservation_id}",
        "payload_sha256": payload_sha256,
        "policy_version": session.policy_version,
        "source_fingerprint_sha256": session.source_fingerprint_sha256,
        "runtime_fingerprint_sha256": session.runtime_fingerprint_sha256,
        "configuration_fingerprint_sha256": session.configuration_fingerprint_sha256,
    }


def _normalize_target(value: str) -> str:
    target = value.strip()
    parsed = urlsplit(target)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("external target must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("external target cannot contain credentials or fragments")
    return target.rstrip("/")


def _normalize_call_contracts(
    contracts: frozenset[ExternalEffectCallContract],
) -> frozenset[ExternalEffectCallContract]:
    normalized: set[ExternalEffectCallContract] = set()
    identities: set[tuple[str, str, str]] = set()
    for contract in contracts:
        operation = contract.operation.strip()
        method = contract.method.strip().upper()
        target = _normalize_target(contract.target)
        _require_text(operation, "operation")
        if method not in {"GET", "POST"}:
            raise ValueError("external call contract method must be GET or POST")
        credits = contract.credit_units_per_request
        if credits is not None and (isinstance(credits, bool) or credits < 0):
            raise ValueError("external call contract credits must be non-negative")
        identity = (operation, method, target)
        if identity in identities:
            raise ValueError("external call contracts must be unique")
        identities.add(identity)
        normalized.add(
            ExternalEffectCallContract(
                operation=operation,
                method=method,
                target=target,
                credit_units_per_request=credits,
            )
        )
    return frozenset(normalized)


def _canonical_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _require_text(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} cannot be blank")


def _require_sha256(value: str, field_name: str) -> None:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{field_name} must be a lowercase sha256")


def _outcome_hash(label: str, error: BaseException | None = None) -> str:
    material = label if error is None else f"{label}:{type(error).__name__}"
    return sha256(material.encode("utf-8")).hexdigest()
