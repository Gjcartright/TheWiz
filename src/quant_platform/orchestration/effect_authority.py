"""Single-use authority permits for every externally visible Phase 00 effect."""

from __future__ import annotations

import hmac
import json
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from quant_platform.orchestration.corrective_redaction import (
    evidence_payload_secret_codes,
)

EFFECT_AUTHORITY_SCHEMA_VERSION = "thewiz.effect_authority.v2"
EFFECT_AUTHORITY_JOURNAL_SCHEMA_VERSION = 2
EFFECT_AUTHORITY_JOURNAL = ".runtime_control/effect_authority.sqlite3"
MAX_PERMIT_TTL_SECONDS = 15 * 60
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EffectKind(StrEnum):
    FILE_PUBLICATION = "file_publication"
    PUBLIC_NETWORK = "public_network"
    AUTHENTICATED_NETWORK = "authenticated_network"
    CREDENTIAL_ACCESS = "credential_access"
    CREDIT_SPEND = "credit_spend"
    ORDER_SUBMISSION = "order_submission"
    ACCOUNT_MUTATION = "account_mutation"


_PROVIDER_SIDE_EFFECT_KINDS = frozenset(
    {
        EffectKind.PUBLIC_NETWORK,
        EffectKind.AUTHENTICATED_NETWORK,
        EffectKind.CREDIT_SPEND,
        EffectKind.ORDER_SUBMISSION,
        EffectKind.ACCOUNT_MUTATION,
    }
)


class EffectAuthorityError(RuntimeError):
    """Raised when an effect cannot prove exact, current, one-use authority."""

    def __init__(self, reason_code: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:,=-]{0,255}", reason_code):
            raise ValueError("effect_authority_reason_code_invalid")
        super().__init__(reason_code)
        self.evidence_reason_code = reason_code


class EffectAuthorityProfile(BaseModel):
    """Deny-by-default issuance policy owned by a supervisor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    allowed_effects: frozenset[EffectKind] = frozenset()
    max_ttl_seconds: int = Field(default=300, ge=1, le=MAX_PERMIT_TTL_SECONDS)
    max_file_publication_bytes: int = Field(default=0, ge=0)

    @field_validator("name")
    @classmethod
    def _name_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("authority profile name cannot be blank")
        return normalized


PHASE00_REPAIR_PROFILE = EffectAuthorityProfile(
    name="PHASE00_REPAIR_FILE_PUBLICATION_ONLY",
    allowed_effects=frozenset({EffectKind.FILE_PUBLICATION}),
    max_ttl_seconds=300,
    max_file_publication_bytes=64 * 1024**2,
)

PHASE00_WIZARD_RESEARCH_PROFILE = EffectAuthorityProfile(
    name="PHASE00_WIZARD_RESEARCH_NO_ORDER",
    allowed_effects=frozenset(
        {
            EffectKind.FILE_PUBLICATION,
            EffectKind.PUBLIC_NETWORK,
            EffectKind.AUTHENTICATED_NETWORK,
            EffectKind.CREDENTIAL_ACCESS,
            EffectKind.CREDIT_SPEND,
        }
    ),
    max_ttl_seconds=300,
    max_file_publication_bytes=64 * 1024**2,
)

PHASE00_DENY_ALL_PROFILE = EffectAuthorityProfile(
    name="PHASE00_DENY_ALL_EXTERNAL_EFFECTS",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _hash_payload(payload: object) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _decimal(value: str, *, field: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a decimal string") from exc
    if not parsed.is_finite() or parsed < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return parsed


def _normalized_hash(value: str, *, field: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{field} must be a lowercase sha256")
    return normalized


def _normalized_required(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


class EffectPermit(BaseModel):
    """Immutable signed authority for exactly one bounded effect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[EFFECT_AUTHORITY_SCHEMA_VERSION] = (
        EFFECT_AUTHORITY_SCHEMA_VERSION
    )
    permit_id: str
    nonce: str
    issuer_id: str
    issuer_profile: str
    issued_at_utc: datetime
    expires_at_utc: datetime
    run_id: str
    intended_slot_id: str
    effect_kind: EffectKind
    target: str
    operation: str
    effect_scope: str
    payload_sha256: str
    venue_id: str = ""
    product_lane_id: str = ""
    account_scope_id: str = ""
    instrument_ids: tuple[str, ...] = ()
    side: Literal["", "buy", "sell"] = ""
    max_units: int = Field(ge=0)
    max_notional: str = "0"
    max_leverage: str = "0"
    proposal_id: str = ""
    policy_version: str
    model_version: str = ""
    formula_version: str = ""
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str
    signature: str

    @field_validator(
        "permit_id",
        "nonce",
        "issuer_id",
        "issuer_profile",
        "run_id",
        "intended_slot_id",
        "target",
        "operation",
        "effect_scope",
        "policy_version",
    )
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _normalized_required(value, field=info.field_name)

    @field_validator(
        "source_fingerprint_sha256",
        "runtime_fingerprint_sha256",
        "configuration_fingerprint_sha256",
    )
    @classmethod
    def _required_hash(cls, value: str, info) -> str:
        return _normalized_hash(value, field=info.field_name)

    @field_validator("payload_sha256")
    @classmethod
    def _payload_hash(cls, value: str) -> str:
        return _normalized_hash(value, field="payload_sha256")

    @field_validator("signature")
    @classmethod
    def _signature_hash(cls, value: str) -> str:
        return _normalized_hash(value, field="signature")

    @field_validator("instrument_ids")
    @classmethod
    def _unique_instruments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if any(not item for item in normalized):
            raise ValueError("instrument_ids cannot contain blanks")
        if len(set(normalized)) != len(normalized):
            raise ValueError("instrument_ids must be unique")
        return normalized

    @field_validator("max_notional", "max_leverage")
    @classmethod
    def _decimal_string(cls, value: str, info) -> str:
        parsed = _decimal(value, field=info.field_name)
        return format(parsed, "f")

    @model_validator(mode="after")
    def _time_and_scope(self) -> EffectPermit:
        issued = _utc(self.issued_at_utc, field="issued_at_utc")
        expires = _utc(self.expires_at_utc, field="expires_at_utc")
        if expires <= issued:
            raise ValueError("permit expiry must follow issuance")
        if expires - issued > timedelta(seconds=MAX_PERMIT_TTL_SECONDS):
            raise ValueError("permit ttl exceeds the universal maximum")
        object.__setattr__(self, "issued_at_utc", issued)
        object.__setattr__(self, "expires_at_utc", expires)
        if self.effect_kind == EffectKind.FILE_PUBLICATION:
            if not Path(self.target).is_absolute():
                raise ValueError("file publication target must be absolute")
            if self.operation not in {"create", "replace", "immutable_create"}:
                raise ValueError("unsupported file publication operation")
        if self.effect_kind in {
            EffectKind.AUTHENTICATED_NETWORK,
            EffectKind.CREDENTIAL_ACCESS,
            EffectKind.CREDIT_SPEND,
            EffectKind.ORDER_SUBMISSION,
            EffectKind.ACCOUNT_MUTATION,
        } and not self.account_scope_id:
            raise ValueError("account_scope_id is required for privileged effects")
        if self.effect_kind == EffectKind.ORDER_SUBMISSION:
            required = (
                self.venue_id,
                self.product_lane_id,
                self.instrument_ids,
                self.side,
                self.proposal_id,
                self.model_version,
                self.formula_version,
            )
            if not all(required):
                raise ValueError("order submission permit is missing execution scope")
        return self

    def signed_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"signature"})


class EffectRequest(BaseModel):
    """Actual effect parameters checked against a permit at consumption."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    intended_slot_id: str
    effect_kind: EffectKind
    target: str
    operation: str
    effect_scope: str
    payload_sha256: str
    venue_id: str = ""
    product_lane_id: str = ""
    account_scope_id: str = ""
    instrument_ids: tuple[str, ...] = ()
    side: Literal["", "buy", "sell"] = ""
    actual_units: int = Field(ge=0)
    actual_notional: str = "0"
    actual_leverage: str = "0"
    proposal_id: str = ""
    policy_version: str
    model_version: str = ""
    formula_version: str = ""
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str

    @field_validator("actual_notional", "actual_leverage")
    @classmethod
    def _decimal_string(cls, value: str, info) -> str:
        parsed = _decimal(value, field=info.field_name)
        return format(parsed, "f")

    @field_validator(
        "source_fingerprint_sha256",
        "runtime_fingerprint_sha256",
        "configuration_fingerprint_sha256",
        "payload_sha256",
    )
    @classmethod
    def _required_hash(cls, value: str, info) -> str:
        return _normalized_hash(value, field=info.field_name)

    @classmethod
    def from_permit(
        cls,
        permit: EffectPermit,
        *,
        actual_units: int,
        actual_notional: str = "0",
        actual_leverage: str = "0",
    ) -> EffectRequest:
        return cls(
            **permit.model_dump(
                exclude={
                    "schema_version",
                    "permit_id",
                    "nonce",
                    "issuer_id",
                    "issuer_profile",
                    "issued_at_utc",
                    "expires_at_utc",
                    "max_units",
                    "max_notional",
                    "max_leverage",
                    "signature",
                }
            ),
            actual_units=actual_units,
            actual_notional=actual_notional,
            actual_leverage=actual_leverage,
        )


class EffectConsumptionReceipt(BaseModel):
    """Immutable evidence that one permit nonce was consumed once."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[EFFECT_AUTHORITY_SCHEMA_VERSION] = (
        EFFECT_AUTHORITY_SCHEMA_VERSION
    )
    receipt_id: str
    permit_id: str
    nonce: str
    effect_kind: EffectKind
    request_sha256: str
    actual_units: int = Field(ge=0)
    actual_notional: str = "0"
    actual_leverage: str = "0"
    consumed_at_utc: datetime
    status: Literal["CONSUMED"] = "CONSUMED"

    @field_validator("actual_notional", "actual_leverage")
    @classmethod
    def _decimal_string(cls, value: str, info) -> str:
        parsed = _decimal(value, field=info.field_name)
        return format(parsed, "f")


class ExternalReservationRecord(BaseModel):
    """Durable run binding for a bounded provider reservation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["thewiz.effect_reservation.v1"] = (
        "thewiz.effect_reservation.v1"
    )
    binding_id: str
    run_id: str
    intended_slot_id: str
    provider_id: str
    account_scope_id: str
    reservation_id: str
    reservation_sha256: str
    binding_nonce: str = Field(default="", max_length=256)
    max_total_requests: int = Field(ge=1)
    max_total_credits: int = Field(ge=0)
    registered_at_utc: datetime

    @field_validator(
        "binding_id",
        "run_id",
        "intended_slot_id",
        "provider_id",
        "account_scope_id",
        "reservation_id",
    )
    @classmethod
    def _required_text(cls, value: str, info) -> str:
        return _normalized_required(value, field=info.field_name)

    @field_validator("reservation_sha256")
    @classmethod
    def _reservation_hash(cls, value: str) -> str:
        return _normalized_hash(value, field="reservation_sha256")

    @field_validator("binding_nonce")
    @classmethod
    def _optional_binding_nonce(cls, value: str) -> str:
        return value.strip()

    @field_validator("registered_at_utc")
    @classmethod
    def _registered_at(cls, value: datetime) -> datetime:
        return _utc(value, field="registered_at_utc")


class EffectAuthority:
    """Issue and atomically consume signed effect permits."""

    def __init__(
        self,
        *,
        root: Path,
        secret: bytes,
        issuer_id: str,
        profile: EffectAuthorityProfile,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if len(secret) < 32:
            raise ValueError("effect authority secret must contain at least 32 bytes")
        self.root = root.resolve()
        self.secret = secret
        self.issuer_id = _normalized_required(issuer_id, field="issuer_id")
        self.profile = profile
        self.clock = clock or (lambda: datetime.now(UTC))
        self.journal_path = self.root / EFFECT_AUTHORITY_JOURNAL
        self._journal_prepare_lock = threading.Lock()
        self._journal_prepared = False
        self._validate_existing_journal_identity()

    def prepare_journal(self) -> None:
        """Explicitly initialize or migrate the private authority journal."""

        self._ensure_journal()

    def issue(
        self,
        *,
        run_id: str,
        intended_slot_id: str,
        effect_kind: EffectKind,
        target: str,
        operation: str,
        effect_scope: str,
        payload_sha256: str,
        policy_version: str,
        source_fingerprint_sha256: str,
        runtime_fingerprint_sha256: str,
        configuration_fingerprint_sha256: str,
        max_units: int,
        ttl_seconds: int = 60,
        max_notional: str = "0",
        max_leverage: str = "0",
        venue_id: str = "",
        product_lane_id: str = "",
        account_scope_id: str = "",
        instrument_ids: tuple[str, ...] = (),
        side: Literal["", "buy", "sell"] = "",
        proposal_id: str = "",
        model_version: str = "",
        formula_version: str = "",
    ) -> EffectPermit:
        if effect_kind not in self.profile.allowed_effects:
            raise EffectAuthorityError(
                f"effect_denied_by_profile:{self.profile.name}:{effect_kind.value}"
            )
        if ttl_seconds <= 0 or ttl_seconds > self.profile.max_ttl_seconds:
            raise EffectAuthorityError("effect_permit_ttl_denied")
        if (
            effect_kind == EffectKind.FILE_PUBLICATION
            and max_units > self.profile.max_file_publication_bytes
        ):
            raise EffectAuthorityError("file_publication_size_denied")
        now = _utc(self.clock(), field="clock")
        unsigned = {
            "schema_version": EFFECT_AUTHORITY_SCHEMA_VERSION,
            "permit_id": f"effectpermit_{uuid4().hex}",
            "nonce": uuid4().hex,
            "issuer_id": self.issuer_id,
            "issuer_profile": self.profile.name,
            "issued_at_utc": now,
            "expires_at_utc": now + timedelta(seconds=ttl_seconds),
            "run_id": run_id,
            "intended_slot_id": intended_slot_id,
            "effect_kind": effect_kind,
            "target": target,
            "operation": operation,
            "effect_scope": effect_scope,
            "payload_sha256": payload_sha256,
            "venue_id": venue_id,
            "product_lane_id": product_lane_id,
            "account_scope_id": account_scope_id,
            "instrument_ids": instrument_ids,
            "side": side,
            "max_units": max_units,
            "max_notional": max_notional,
            "max_leverage": max_leverage,
            "proposal_id": proposal_id,
            "policy_version": policy_version,
            "model_version": model_version,
            "formula_version": formula_version,
            "source_fingerprint_sha256": source_fingerprint_sha256,
            "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
            "configuration_fingerprint_sha256": configuration_fingerprint_sha256,
        }
        unsigned_permit = EffectPermit(**unsigned, signature="0" * 64)
        signature = self._sign(unsigned_permit.signed_payload())
        permit = unsigned_permit.model_copy(update={"signature": signature})
        payload_json = _canonical_json(permit.model_dump(mode="json"))
        with self._transaction() as connection:
            connection.execute(
                """
                INSERT INTO effect_permits (
                    permit_id, nonce, payload_sha256, payload_json, signature,
                    state, issued_at_utc, expires_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'ISSUED', ?, ?)
                """,
                (
                    permit.permit_id,
                    permit.nonce,
                    sha256(payload_json.encode("utf-8")).hexdigest(),
                    payload_json,
                    permit.signature,
                    permit.issued_at_utc.isoformat(),
                    permit.expires_at_utc.isoformat(),
                ),
            )
        return permit

    def consume(
        self,
        permit: EffectPermit,
        request: EffectRequest,
    ) -> EffectConsumptionReceipt:
        self._validate_signature(permit)
        self._validate_request(permit, request)
        now = _utc(self.clock(), field="clock")
        if now < permit.issued_at_utc:
            raise EffectAuthorityError("effect_permit_not_yet_valid")
        if now >= permit.expires_at_utc:
            raise EffectAuthorityError("effect_permit_expired")
        payload_json = _canonical_json(permit.model_dump(mode="json"))
        payload_hash = sha256(payload_json.encode("utf-8")).hexdigest()
        request_hash = _hash_payload(request.model_dump(mode="json"))
        receipt_material = {
            "permit_id": permit.permit_id,
            "nonce": permit.nonce,
            "effect_kind": permit.effect_kind.value,
            "request_sha256": request_hash,
            "actual_units": request.actual_units,
            "actual_notional": request.actual_notional,
            "actual_leverage": request.actual_leverage,
            "consumed_at_utc": now.isoformat(),
        }
        receipt = EffectConsumptionReceipt(
            receipt_id=f"effectreceipt_{_hash_payload(receipt_material)[:24]}",
            permit_id=permit.permit_id,
            nonce=permit.nonce,
            effect_kind=permit.effect_kind,
            request_sha256=request_hash,
            actual_units=request.actual_units,
            actual_notional=request.actual_notional,
            actual_leverage=request.actual_leverage,
            consumed_at_utc=now,
        )
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT nonce, payload_sha256, payload_json, signature, state,
                       expires_at_utc
                FROM effect_permits WHERE permit_id = ?
                """,
                (permit.permit_id,),
            ).fetchone()
            if row is None:
                raise EffectAuthorityError("effect_permit_unknown")
            if row[0] != permit.nonce:
                raise EffectAuthorityError("effect_permit_nonce_mismatch")
            if row[1] != payload_hash or row[2] != payload_json:
                raise EffectAuthorityError("effect_permit_payload_mismatch")
            if row[3] != permit.signature:
                raise EffectAuthorityError("effect_permit_signature_mismatch")
            if row[4] != "ISSUED":
                raise EffectAuthorityError(f"effect_permit_not_usable:{row[4]}")
            journal_expiry = datetime.fromisoformat(row[5]).astimezone(UTC)
            if now >= journal_expiry:
                connection.execute(
                    """
                    UPDATE effect_permits SET state = 'EXPIRED'
                    WHERE permit_id = ? AND state = 'ISSUED'
                    """,
                    (permit.permit_id,),
                )
                raise EffectAuthorityError("effect_permit_expired")
            updated = connection.execute(
                """
                UPDATE effect_permits
                SET state = 'CONSUMED', consumed_at_utc = ?,
                    request_sha256 = ?, consumption_receipt_json = ?
                WHERE permit_id = ? AND state = 'ISSUED'
                """,
                (
                    now.isoformat(),
                    request_hash,
                    _canonical_json(receipt.model_dump(mode="json")),
                    permit.permit_id,
                ),
            )
            if updated.rowcount != 1:
                raise EffectAuthorityError("effect_permit_concurrent_replay")
        return receipt

    def register_external_reservation(
        self,
        *,
        run_id: str,
        intended_slot_id: str,
        provider_id: str,
        account_scope_id: str,
        reservation_id: str,
        reservation_sha256: str,
        max_total_requests: int,
        max_total_credits: int,
        binding_nonce: str = "",
    ) -> ExternalReservationRecord:
        """Bind one provider budget to one run before any wire attempt."""

        now = _utc(self.clock(), field="clock")
        identity_material = {
            "schema_version": "thewiz.effect_reservation.v1",
            "run_id": _normalized_required(run_id, field="run_id"),
            "intended_slot_id": _normalized_required(
                intended_slot_id,
                field="intended_slot_id",
            ),
            "provider_id": _normalized_required(provider_id, field="provider_id"),
            "account_scope_id": _normalized_required(
                account_scope_id,
                field="account_scope_id",
            ),
            "reservation_id": _normalized_required(
                reservation_id,
                field="reservation_id",
            ),
            "reservation_sha256": _normalized_hash(
                reservation_sha256,
                field="reservation_sha256",
            ),
            "binding_nonce": binding_nonce.strip(),
            "max_total_requests": max_total_requests,
            "max_total_credits": max_total_credits,
        }
        binding_id = f"effectreservation_{_hash_payload(identity_material)[:24]}"
        material = {**identity_material, "registered_at_utc": now}
        record = ExternalReservationRecord(binding_id=binding_id, **material)
        payload_json = _canonical_json(record.model_dump(mode="json"))
        payload_sha256 = sha256(payload_json.encode("utf-8")).hexdigest()
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT payload_json, state
                FROM external_reservations WHERE binding_id = ?
                """,
                (binding_id,),
            ).fetchone()
            if existing is not None:
                try:
                    existing_record = ExternalReservationRecord.model_validate_json(
                        existing[0]
                    )
                except (TypeError, ValueError) as exc:
                    raise EffectAuthorityError(
                        "external_reservation_existing_record_invalid"
                    ) from exc
                if any(
                    getattr(existing_record, field) != value
                    for field, value in identity_material.items()
                ):
                    raise EffectAuthorityError(
                        "external_reservation_identity_collision"
                    )
                if existing[1] != "OPEN":
                    raise EffectAuthorityError(
                        f"external_reservation_not_open:{existing[1]}"
                    )
                return existing_record
            connection.execute(
                """
                INSERT INTO external_reservations (
                    binding_id, run_id, intended_slot_id, payload_sha256,
                    payload_json, state, registered_at_utc
                ) VALUES (?, ?, ?, ?, ?, 'OPEN', ?)
                """,
                (
                    binding_id,
                    record.run_id,
                    record.intended_slot_id,
                    payload_sha256,
                    payload_json,
                    record.registered_at_utc.isoformat(),
                ),
            )
        return record

    def require_open_external_reservation(
        self,
        *,
        binding_id: str,
        run_id: str,
        intended_slot_id: str,
        provider_id: str,
        account_scope_id: str,
        reservation_id: str,
        reservation_sha256: str,
        max_total_requests: int,
        max_total_credits: int,
    ) -> ExternalReservationRecord:
        """Load one exact pre-registered binding without creating a substitute."""

        normalized_binding_id = _normalized_required(
            binding_id,
            field="binding_id",
        )
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT payload_json, state
                FROM external_reservations
                WHERE binding_id = ?
                """,
                (normalized_binding_id,),
            ).fetchone()
        if row is None:
            raise EffectAuthorityError("external_reservation_unknown")
        try:
            record = ExternalReservationRecord.model_validate_json(row[0])
        except (TypeError, ValueError) as exc:
            raise EffectAuthorityError(
                "external_reservation_existing_record_invalid"
            ) from exc
        expected = {
            "binding_id": normalized_binding_id,
            "run_id": _normalized_required(run_id, field="run_id"),
            "intended_slot_id": _normalized_required(
                intended_slot_id,
                field="intended_slot_id",
            ),
            "provider_id": _normalized_required(provider_id, field="provider_id"),
            "account_scope_id": _normalized_required(
                account_scope_id,
                field="account_scope_id",
            ),
            "reservation_id": _normalized_required(
                reservation_id,
                field="reservation_id",
            ),
            "reservation_sha256": _normalized_hash(
                reservation_sha256,
                field="reservation_sha256",
            ),
            "max_total_requests": max_total_requests,
            "max_total_credits": max_total_credits,
        }
        if any(getattr(record, field) != value for field, value in expected.items()):
            raise EffectAuthorityError("external_reservation_scope_mismatch")
        if row[1] != "OPEN":
            raise EffectAuthorityError(f"external_reservation_not_open:{row[1]}")
        return record

    def fail_open_external_reservations_without_provider_effects(
        self,
        *,
        run_id: str,
        intended_slot_id: str,
    ) -> int:
        """Close abandoned budget bindings only when no provider effect began."""

        accounting = self.run_accounting(
            run_id=run_id,
            intended_slot_id=intended_slot_id,
        )
        if (
            int(accounting["issued_provider_effect_permits"]) > 0
            or int(accounting["consumed_provider_effect_permits"]) > 0
        ):
            raise EffectAuthorityError(
                "external_reservation_provider_effect_activity_present"
            )
        now = _utc(self.clock(), field="clock")
        with self._transaction() as connection:
            updated = connection.execute(
                """
                UPDATE external_reservations
                SET state = 'FAILED', closed_at_utc = ?
                WHERE run_id = ? AND intended_slot_id = ? AND state = 'OPEN'
                """,
                (
                    now.isoformat(),
                    _normalized_required(run_id, field="run_id"),
                    _normalized_required(
                        intended_slot_id,
                        field="intended_slot_id",
                    ),
                ),
            )
        return int(updated.rowcount)

    def external_reservation_retry_safe(
        self,
        *,
        reservation_id: str,
        reservation_sha256: str,
    ) -> bool:
        """Return true only when every prior exact binding failed before effects."""

        normalized_id = _normalized_required(
            reservation_id,
            field="reservation_id",
        )
        normalized_sha256 = _normalized_hash(
            reservation_sha256,
            field="reservation_sha256",
        )
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT payload_json, state
                FROM external_reservations
                ORDER BY registered_at_utc, binding_id
                """
            ).fetchall()
        matches: list[tuple[ExternalReservationRecord, str]] = []
        for payload_json, state in rows:
            try:
                record = ExternalReservationRecord.model_validate_json(payload_json)
            except (TypeError, ValueError):
                return False
            if (
                record.reservation_id == normalized_id
                and record.reservation_sha256 == normalized_sha256
            ):
                matches.append((record, str(state)))
        if not matches:
            return False
        for record, state in matches:
            if state != "FAILED":
                return False
            accounting = self.run_accounting(
                run_id=record.run_id,
                intended_slot_id=record.intended_slot_id,
            )
            if (
                int(accounting["issued_provider_effect_permits"]) > 0
                or int(accounting["consumed_provider_effect_permits"]) > 0
            ):
                return False
        return True

    def close_external_reservation(
        self,
        binding_id: str,
        *,
        status: Literal["CLOSED", "FAILED"],
    ) -> None:
        """Close one reservation exactly once after its bounded session exits."""

        normalized_id = _normalized_required(binding_id, field="binding_id")
        now = _utc(self.clock(), field="clock")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM external_reservations WHERE binding_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                raise EffectAuthorityError("external_reservation_unknown")
            if row[0] == status:
                return
            if row[0] != "OPEN":
                raise EffectAuthorityError(
                    f"external_reservation_not_closable:{row[0]}"
                )
            updated = connection.execute(
                """
                UPDATE external_reservations
                SET state = ?, closed_at_utc = ?
                WHERE binding_id = ? AND state = 'OPEN'
                """,
                (status, now.isoformat(), normalized_id),
            )
            if updated.rowcount != 1:
                raise EffectAuthorityError(
                    "external_reservation_concurrent_close"
                )

    def revoke(self, permit_id: str, *, reason: str) -> None:
        normalized_reason = _normalized_required(reason, field="reason")
        now = _utc(self.clock(), field="clock")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM effect_permits WHERE permit_id = ?",
                (permit_id,),
            ).fetchone()
            if row is None:
                raise EffectAuthorityError("effect_permit_unknown")
            if row[0] != "ISSUED":
                raise EffectAuthorityError(f"effect_permit_not_revocable:{row[0]}")
            updated = connection.execute(
                """
                UPDATE effect_permits
                SET state = 'REVOKED', revoked_at_utc = ?, revocation_reason = ?
                WHERE permit_id = ? AND state = 'ISSUED'
                """,
                (now.isoformat(), normalized_reason, permit_id),
            )
            if updated.rowcount != 1:
                raise EffectAuthorityError("effect_permit_concurrent_revoke")

    def state(self, permit_id: str) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT state FROM effect_permits WHERE permit_id = ?",
                (permit_id,),
            ).fetchone()
        if row is None:
            raise EffectAuthorityError("effect_permit_unknown")
        return str(row[0])

    def record_outcome(
        self,
        permit_id: str,
        *,
        status: Literal["SUCCEEDED", "FAILED", "UNKNOWN"],
        detail_sha256: str,
    ) -> None:
        """Close one consumed effect once without storing provider or secret text."""

        detail_hash = _normalized_hash(detail_sha256, field="detail_sha256")
        recorded_at = _utc(self.clock(), field="clock")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM effect_permits WHERE permit_id = ?",
                (permit_id,),
            ).fetchone()
            if row is None:
                raise EffectAuthorityError("effect_permit_unknown")
            if row[0] != "CONSUMED":
                raise EffectAuthorityError(
                    f"effect_outcome_permit_not_consumed:{row[0]}"
                )
            try:
                connection.execute(
                    """
                    INSERT INTO effect_outcomes (
                        permit_id, status, recorded_at_utc, detail_sha256
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        permit_id,
                        status,
                        recorded_at.isoformat(),
                        detail_hash,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise EffectAuthorityError("effect_outcome_already_recorded") from exc

    def outcome(self, permit_id: str) -> str:
        """Return durable closure or explicit ambiguity for a consumed effect."""

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT p.state, o.status
                FROM effect_permits AS p
                LEFT JOIN effect_outcomes AS o ON o.permit_id = p.permit_id
                WHERE p.permit_id = ?
                """,
                (permit_id,),
            ).fetchone()
        if row is None:
            raise EffectAuthorityError("effect_permit_unknown")
        if row[1] is not None:
            return str(row[1])
        if row[0] == "CONSUMED":
            return "IN_FLIGHT_OR_UNKNOWN"
        return str(row[0])

    def run_accounting(self, *, run_id: str, intended_slot_id: str) -> dict[str, object]:
        """Reconstruct durable effect accounting for one exact scheduler run."""

        normalized_run_id = _normalized_required(run_id, field="run_id")
        normalized_slot = _normalized_required(
            intended_slot_id,
            field="intended_slot_id",
        )
        with self._connection() as connection:
            reservation_rows = connection.execute(
                """
                SELECT payload_json, state
                FROM external_reservations
                WHERE run_id = ? AND intended_slot_id = ?
                ORDER BY registered_at_utc, binding_id
                """,
                (normalized_run_id, normalized_slot),
            ).fetchall()
            rows = connection.execute(
                """
                SELECT p.payload_json, p.state, p.consumption_receipt_json,
                       o.status
                FROM effect_permits AS p
                LEFT JOIN effect_outcomes AS o ON o.permit_id = p.permit_id
                ORDER BY p.issued_at_utc, p.permit_id
                """
            ).fetchall()

        counts = {
            "external_reservations": 0,
            "external_requests_reserved": 0,
            "open_reservations": 0,
            "failed_reservations": 0,
            "matching_permits": 0,
            "issued_permits": 0,
            "consumed_permits": 0,
            "issued_provider_effect_permits": 0,
            "consumed_provider_effect_permits": 0,
            "external_calls": 0,
            "external_credits_reserved": 0,
            "external_credits_consumed": 0,
            "closed_effects": 0,
            "unknown_effects": 0,
            "failed_effects": 0,
            "order_attempts": 0,
        }
        blockers: set[str] = set()
        for payload_json, state in reservation_rows:
            try:
                reservation = ExternalReservationRecord.model_validate_json(
                    payload_json
                )
            except (TypeError, ValueError):
                blockers.add("external_reservation_record_invalid")
                continue
            if (
                reservation.run_id != normalized_run_id
                or reservation.intended_slot_id != normalized_slot
            ):
                blockers.add("external_reservation_identity_mismatch")
                continue
            counts["external_reservations"] += 1
            counts["external_requests_reserved"] += (
                reservation.max_total_requests
            )
            counts["external_credits_reserved"] += reservation.max_total_credits
            if state == "OPEN":
                counts["open_reservations"] += 1
                blockers.add("external_reservation_open")
            elif state == "FAILED":
                counts["failed_reservations"] += 1
                blockers.add("external_reservation_failed")
            elif state != "CLOSED":
                blockers.add("external_reservation_state_invalid")
        permit_reserved_credits = 0
        for payload_json, state, receipt_json, outcome_status in rows:
            try:
                raw_permit = json.loads(payload_json)
            except (TypeError, ValueError, json.JSONDecodeError):
                blockers.add("effect_journal_permit_unreadable")
                continue
            if not isinstance(raw_permit, dict):
                blockers.add("effect_journal_permit_not_object")
                continue
            if (
                raw_permit.get("run_id") != normalized_run_id
                or raw_permit.get("intended_slot_id") != normalized_slot
            ):
                continue
            counts["matching_permits"] += 1
            try:
                permit = EffectPermit.model_validate(raw_permit)
            except (TypeError, ValueError):
                blockers.add("effect_journal_matching_permit_schema_invalid")
                continue
            if permit.effect_kind == EffectKind.CREDIT_SPEND:
                permit_reserved_credits += permit.max_units
            if state == "ISSUED":
                counts["issued_permits"] += 1
                if permit.effect_kind in _PROVIDER_SIDE_EFFECT_KINDS:
                    counts["issued_provider_effect_permits"] += 1
                blockers.add("effect_permit_issued_without_consumption")
                continue
            if state != "CONSUMED":
                continue
            counts["consumed_permits"] += 1
            if permit.effect_kind in _PROVIDER_SIDE_EFFECT_KINDS:
                counts["consumed_provider_effect_permits"] += 1
            try:
                receipt = EffectConsumptionReceipt.model_validate_json(receipt_json)
            except (TypeError, ValueError):
                blockers.add("effect_consumption_receipt_missing_or_legacy")
                continue
            if receipt.permit_id != permit.permit_id or receipt.effect_kind != permit.effect_kind:
                blockers.add("effect_consumption_receipt_identity_mismatch")
                continue
            if permit.effect_kind in {
                EffectKind.PUBLIC_NETWORK,
                EffectKind.AUTHENTICATED_NETWORK,
            }:
                counts["external_calls"] += receipt.actual_units
            elif permit.effect_kind == EffectKind.CREDIT_SPEND:
                counts["external_credits_consumed"] += receipt.actual_units
            elif permit.effect_kind == EffectKind.ORDER_SUBMISSION:
                counts["order_attempts"] += receipt.actual_units
            if permit.effect_kind == EffectKind.FILE_PUBLICATION:
                continue
            if outcome_status is None or outcome_status == "UNKNOWN":
                counts["unknown_effects"] += 1
                blockers.add("effect_outcome_unknown_or_in_flight")
            else:
                counts["closed_effects"] += 1
                if outcome_status == "FAILED":
                    counts["failed_effects"] += 1
                    blockers.add("effect_outcome_failed")
        if permit_reserved_credits and not counts["external_reservations"]:
            counts["external_credits_reserved"] = permit_reserved_credits
            blockers.add("external_reservation_binding_missing")
        return {
            "schema_version": "thewiz.effect_run_accounting.v1",
            "run_id": normalized_run_id,
            "intended_slot_id": normalized_slot,
            **counts,
            "blockers": sorted(blockers),
            "accounting_complete": not blockers,
        }

    def _sign(self, payload: dict[str, object]) -> str:
        return hmac.new(
            self.secret,
            _canonical_json(payload).encode("utf-8"),
            sha256,
        ).hexdigest()

    def _validate_signature(self, permit: EffectPermit) -> None:
        expected = self._sign(permit.signed_payload())
        if not hmac.compare_digest(expected, permit.signature):
            raise EffectAuthorityError("effect_permit_signature_invalid")
        if permit.issuer_id != self.issuer_id:
            raise EffectAuthorityError("effect_permit_issuer_mismatch")
        if permit.issuer_profile != self.profile.name:
            raise EffectAuthorityError("effect_permit_profile_mismatch")
        if permit.effect_kind not in self.profile.allowed_effects:
            raise EffectAuthorityError("effect_permit_kind_not_allowed")

    @staticmethod
    def _validate_request(permit: EffectPermit, request: EffectRequest) -> None:
        exact_fields = (
            "run_id",
            "intended_slot_id",
            "effect_kind",
            "target",
            "operation",
            "effect_scope",
            "payload_sha256",
            "venue_id",
            "product_lane_id",
            "account_scope_id",
            "instrument_ids",
            "side",
            "proposal_id",
            "policy_version",
            "model_version",
            "formula_version",
            "source_fingerprint_sha256",
            "runtime_fingerprint_sha256",
            "configuration_fingerprint_sha256",
        )
        for field in exact_fields:
            if getattr(permit, field) != getattr(request, field):
                raise EffectAuthorityError(f"effect_request_scope_mismatch:{field}")
        if request.actual_units > permit.max_units:
            raise EffectAuthorityError("effect_request_units_exceed_permit")
        if _decimal(request.actual_notional, field="actual_notional") > _decimal(
            permit.max_notional,
            field="max_notional",
        ):
            raise EffectAuthorityError("effect_request_notional_exceeds_permit")
        if _decimal(request.actual_leverage, field="actual_leverage") > _decimal(
            permit.max_leverage,
            field="max_leverage",
        ):
            raise EffectAuthorityError("effect_request_leverage_exceeds_permit")

    def _prepare_journal(self) -> None:
        control = self.journal_path.parent
        if control.exists() and control.is_symlink():
            raise EffectAuthorityError("effect_authority_control_symlink_denied")
        control.mkdir(parents=True, exist_ok=True)
        if not control.is_dir() or control.resolve() != self.root / ".runtime_control":
            raise EffectAuthorityError("effect_authority_control_path_invalid")
        control.chmod(0o700)
        if self.journal_path.exists():
            identity = self.journal_path.lstat()
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise EffectAuthorityError("effect_authority_journal_not_regular")
            if identity.st_uid != os.getuid():
                raise EffectAuthorityError("effect_authority_journal_owner_mismatch")
        with self._raw_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS journal_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS effect_permits (
                    permit_id TEXT PRIMARY KEY,
                    nonce TEXT NOT NULL UNIQUE,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('ISSUED', 'CONSUMED', 'REVOKED', 'EXPIRED')
                    ),
                    issued_at_utc TEXT NOT NULL,
                    expires_at_utc TEXT NOT NULL,
                    consumed_at_utc TEXT,
                    revoked_at_utc TEXT,
                    revocation_reason TEXT,
                    request_sha256 TEXT,
                    consumption_receipt_json TEXT
                );
                CREATE TABLE IF NOT EXISTS effect_outcomes (
                    permit_id TEXT PRIMARY KEY REFERENCES effect_permits(permit_id),
                    status TEXT NOT NULL CHECK (
                        status IN ('SUCCEEDED', 'FAILED', 'UNKNOWN')
                    ),
                    recorded_at_utc TEXT NOT NULL,
                    detail_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS external_reservations (
                    binding_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    intended_slot_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('OPEN', 'CLOSED', 'FAILED')
                    ),
                    registered_at_utc TEXT NOT NULL,
                    closed_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS external_reservations_run_slot
                ON external_reservations(run_id, intended_slot_id);
                """
            )
            existing = connection.execute(
                "SELECT value FROM journal_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO journal_metadata (key, value) VALUES (?, ?)",
                    (
                        "schema_version",
                        str(EFFECT_AUTHORITY_JOURNAL_SCHEMA_VERSION),
                    ),
                )
            elif existing[0] == "1" and EFFECT_AUTHORITY_JOURNAL_SCHEMA_VERSION == 2:
                connection.execute(
                    """
                    UPDATE journal_metadata SET value = ?
                    WHERE key = 'schema_version' AND value = '1'
                    """,
                    (str(EFFECT_AUTHORITY_JOURNAL_SCHEMA_VERSION),),
                )
            elif existing[0] != str(EFFECT_AUTHORITY_JOURNAL_SCHEMA_VERSION):
                raise EffectAuthorityError("effect_authority_journal_schema_mismatch")
        self.journal_path.chmod(0o600)

    def _validate_existing_journal_identity(self) -> None:
        control = self.journal_path.parent
        if control.exists() and control.is_symlink():
            raise EffectAuthorityError("effect_authority_control_symlink_denied")
        if not self.journal_path.exists() and not self.journal_path.is_symlink():
            return
        identity = self.journal_path.lstat()
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise EffectAuthorityError("effect_authority_journal_not_regular")
        if identity.st_uid != os.getuid():
            raise EffectAuthorityError("effect_authority_journal_owner_mismatch")

    def _ensure_journal(self) -> None:
        if self._journal_prepared:
            return
        with self._journal_prepare_lock:
            if self._journal_prepared:
                return
            self._prepare_journal()
            self._journal_prepared = True

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        self._ensure_journal()
        with self._raw_connection() as connection:
            yield connection

    @contextmanager
    def _raw_connection(self) -> Iterator[sqlite3.Connection]:
        if self.journal_path.is_symlink():
            raise EffectAuthorityError("effect_authority_journal_symlink_denied")
        connection = sqlite3.connect(
            self.journal_path,
            timeout=5,
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute("PRAGMA trusted_schema = OFF")
            identity = self.journal_path.lstat()
            if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
                raise EffectAuthorityError("effect_authority_journal_identity_invalid")
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()


@dataclass
class PublicationAuthoritySession:
    """Supervisor-owned issuance context used by final publication sinks."""

    authority: EffectAuthority
    root: Path
    run_id: str
    intended_slot_id: str
    policy_version: str
    source_fingerprint_sha256: str
    runtime_fingerprint_sha256: str
    configuration_fingerprint_sha256: str
    allowed_scopes: frozenset[str]
    allowed_target_prefixes: tuple[Path, ...]
    allowed_external_targets: frozenset[Path]
    max_total_bytes: int
    consumed_bytes: int = 0
    consumed_writes: int = 0
    receipts: list[EffectConsumptionReceipt] = dataclass_field(default_factory=list)
    _lock: threading.Lock = dataclass_field(default_factory=threading.Lock, repr=False)


_CURRENT_PUBLICATION_AUTHORITY: ContextVar[PublicationAuthoritySession | None] = (
    ContextVar("thewiz_current_publication_authority", default=None)
)


@contextmanager
def publication_authority_session(
    *,
    authority: EffectAuthority,
    run_id: str,
    intended_slot_id: str,
    policy_version: str,
    source_fingerprint_sha256: str,
    runtime_fingerprint_sha256: str,
    configuration_fingerprint_sha256: str,
    allowed_scopes: frozenset[str],
    allowed_target_prefixes: tuple[Path, ...],
    max_total_bytes: int,
    allowed_external_targets: tuple[Path, ...] = (),
) -> Iterator[PublicationAuthoritySession]:
    """Install one explicit supervisor authority context for governed writes."""

    if max_total_bytes < 0:
        raise ValueError("max_total_bytes must be non-negative")
    root = authority.root.resolve()
    normalized_prefixes: list[Path] = []
    for prefix in allowed_target_prefixes:
        candidate = Path(os.path.abspath(prefix))
        if candidate != root and root not in candidate.parents:
            raise EffectAuthorityError("publication_authority_prefix_outside_root")
        normalized_prefixes.append(candidate)
    normalized_external_targets: set[Path] = set()
    for target in allowed_external_targets:
        candidate = Path(os.path.abspath(target))
        if candidate == root or root in candidate.parents:
            raise EffectAuthorityError(
                "publication_authority_external_target_inside_root"
            )
        if candidate.is_symlink():
            raise EffectAuthorityError(
                "publication_authority_external_target_symlink_denied"
            )
        normalized_external_targets.add(candidate)
    if not allowed_scopes or any(not scope.strip() for scope in allowed_scopes):
        raise EffectAuthorityError("publication_authority_scopes_required")
    existing = _CURRENT_PUBLICATION_AUTHORITY.get()
    if existing is not None:
        raise EffectAuthorityError("nested_publication_authority_session_denied")
    session = PublicationAuthoritySession(
        authority=authority,
        root=root,
        run_id=_normalized_required(run_id, field="run_id"),
        intended_slot_id=_normalized_required(
            intended_slot_id,
            field="intended_slot_id",
        ),
        policy_version=_normalized_required(policy_version, field="policy_version"),
        source_fingerprint_sha256=_normalized_hash(
            source_fingerprint_sha256,
            field="source_fingerprint_sha256",
        ),
        runtime_fingerprint_sha256=_normalized_hash(
            runtime_fingerprint_sha256,
            field="runtime_fingerprint_sha256",
        ),
        configuration_fingerprint_sha256=_normalized_hash(
            configuration_fingerprint_sha256,
            field="configuration_fingerprint_sha256",
        ),
        allowed_scopes=frozenset(scope.strip() for scope in allowed_scopes),
        allowed_target_prefixes=tuple(normalized_prefixes),
        allowed_external_targets=frozenset(normalized_external_targets),
        max_total_bytes=max_total_bytes,
    )
    token = _CURRENT_PUBLICATION_AUTHORITY.set(session)
    try:
        yield session
    finally:
        _CURRENT_PUBLICATION_AUTHORITY.reset(token)


def current_publication_authority() -> PublicationAuthoritySession | None:
    return _CURRENT_PUBLICATION_AUTHORITY.get()


def require_publication_authority(
    *,
    root: Path,
    publication_scope: str,
) -> PublicationAuthoritySession:
    """Fail unless a supervisor has installed matching publication authority."""

    session = current_publication_authority()
    if session is None:
        raise EffectAuthorityError("publication_authority_session_missing")
    canonical_root = root.resolve()
    if canonical_root != session.root and session.root not in canonical_root.parents:
        raise EffectAuthorityError("publication_authority_root_mismatch")
    if (
        "*" not in session.allowed_scopes
        and publication_scope not in session.allowed_scopes
    ):
        raise EffectAuthorityError("publication_authority_scope_denied")
    return session


def authorize_file_publication(
    *,
    root: Path,
    target: Path,
    publication_scope: str,
    operation: Literal["create", "replace", "immutable_create"],
    payload: bytes,
) -> EffectConsumptionReceipt:
    """Consume one exact file-publication permit immediately before commit."""

    session = require_file_publication_target(
        root=root,
        target=target,
        publication_scope=publication_scope,
    )
    lexical_target = Path(os.path.abspath(target))
    units = len(payload)
    secret_codes = evidence_payload_secret_codes(payload)
    if secret_codes:
        raise EffectAuthorityError(
            "publication_payload_secret_detected:" + ",".join(secret_codes)
        )
    payload_hash = sha256(payload).hexdigest()
    with session._lock:
        if session.consumed_bytes + units > session.max_total_bytes:
            raise EffectAuthorityError("publication_authority_budget_exhausted")
        permit = session.authority.issue(
            run_id=session.run_id,
            intended_slot_id=session.intended_slot_id,
            effect_kind=EffectKind.FILE_PUBLICATION,
            target=str(lexical_target),
            operation=operation,
            effect_scope=publication_scope,
            payload_sha256=payload_hash,
            policy_version=session.policy_version,
            source_fingerprint_sha256=session.source_fingerprint_sha256,
            runtime_fingerprint_sha256=session.runtime_fingerprint_sha256,
            configuration_fingerprint_sha256=(
                session.configuration_fingerprint_sha256
            ),
            max_units=units,
        )
        request = EffectRequest.from_permit(permit, actual_units=units)
        receipt = session.authority.consume(permit, request)
        session.consumed_bytes += units
        session.consumed_writes += 1
        session.receipts.append(receipt)
        return receipt


def require_file_publication_target(
    *,
    root: Path,
    target: Path,
    publication_scope: str,
) -> PublicationAuthoritySession:
    """Validate one publication target before any filesystem side effect."""

    session = require_publication_authority(
        root=root,
        publication_scope=publication_scope,
    )
    lexical_target = Path(os.path.abspath(target))
    if not any(
        lexical_target == prefix or prefix in lexical_target.parents
        for prefix in session.allowed_target_prefixes
    ) and lexical_target not in session.allowed_external_targets:
        raise EffectAuthorityError("publication_authority_target_denied")
    return session
