"""Immutable terminal accounting for governed scheduler invocations."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from quant_platform.orchestration.corrective_runtime import (
    atomic_write_text,
    create_exclusive_json,
    write_immutable_json,
)
from quant_platform.orchestration.corrective_scheduler_lock import (
    current_scheduler_process_identity,
    scheduler_terminal_write_lock,
)

RUN_INTENT_SCHEMA_VERSION = "thewiz.scheduler_run_intent.v2"
SLOT_CLAIM_SCHEMA_VERSION = "thewiz.scheduler_slot_claim.v1"
TERMINAL_RECEIPT_SCHEMA_VERSION = "thewiz.scheduler_terminal_receipt.v1"
TERMINAL_POINTER_SCHEMA_VERSION = "thewiz.scheduler_terminal_pointer.v1"
TERMINAL_PUBLICATION_SCOPE = "scheduler_terminal"
MAX_TERMINAL_POINTER_BYTES = 64 * 1024
MAX_TERMINAL_RECEIPT_BYTES = 1024 * 1024

TerminalStatus = Literal[
    "PASS",
    "BLOCKED",
    "DEFERRED",
    "INCOMPLETE",
    "TIMEOUT",
    "CHILD_FAILURE",
    "CRASH_RECOVERED",
    "FAILED",
]
ProcessHealth = Literal["HEALTHY", "DEGRADED", "FAILED", "UNKNOWN"]
BusinessState = Literal["PASS", "BLOCKED", "DEFERRED", "INCOMPLETE", "FAILED"]

_TERMINAL_STATUSES = {
    "PASS",
    "BLOCKED",
    "DEFERRED",
    "INCOMPLETE",
    "TIMEOUT",
    "CHILD_FAILURE",
    "CRASH_RECOVERED",
    "FAILED",
}
_PROCESS_HEALTH = {"HEALTHY", "DEGRADED", "FAILED", "UNKNOWN"}
_BUSINESS_STATES = {"PASS", "BLOCKED", "DEFERRED", "INCOMPLETE", "FAILED"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$")


@dataclass(frozen=True)
class SchedulerTerminalStateRule:
    terminal_status: TerminalStatus
    process_health: ProcessHealth
    business_state: BusinessState
    retryable_values: frozenset[bool]
    intended_slot_credit: bool
    exit_code: int


TERMINAL_STATE_RULES: dict[str, SchedulerTerminalStateRule] = {
    "PASS": SchedulerTerminalStateRule(
        "PASS", "HEALTHY", "PASS", frozenset({False}), True, 0
    ),
    "BLOCKED": SchedulerTerminalStateRule(
        "BLOCKED", "HEALTHY", "BLOCKED", frozenset({False, True}), False, 2
    ),
    "DEFERRED": SchedulerTerminalStateRule(
        "DEFERRED", "HEALTHY", "DEFERRED", frozenset({False, True}), False, 2
    ),
    "INCOMPLETE": SchedulerTerminalStateRule(
        "INCOMPLETE", "DEGRADED", "INCOMPLETE", frozenset({True}), False, 2
    ),
    "TIMEOUT": SchedulerTerminalStateRule(
        "TIMEOUT", "FAILED", "INCOMPLETE", frozenset({True}), False, 2
    ),
    "CHILD_FAILURE": SchedulerTerminalStateRule(
        "CHILD_FAILURE", "FAILED", "FAILED", frozenset({False, True}), False, 2
    ),
    "CRASH_RECOVERED": SchedulerTerminalStateRule(
        "CRASH_RECOVERED",
        "DEGRADED",
        "INCOMPLETE",
        frozenset({False, True}),
        False,
        2,
    ),
    "FAILED": SchedulerTerminalStateRule(
        "FAILED", "FAILED", "FAILED", frozenset({False}), False, 2
    ),
}


@dataclass(frozen=True)
class SchedulerTerminalReceipt:
    """One complete terminal account for a single scheduler run."""

    schema_version: str
    receipt_id: str
    run_id: str
    scheduler_key: str
    intended_slot: str
    started_at_utc: str
    completed_at_utc: str
    terminal_status: TerminalStatus
    process_health: ProcessHealth
    business_state: BusinessState
    retryable: bool
    intended_slot_credit: bool
    authority_advanced: bool
    promotion_authority: bool
    live_trading_authorized: bool
    external_calls: int
    external_credits_reserved: int
    external_credits_consumed: int
    external_credits_reconciled: int
    order_attempts: int
    order_submissions: int
    child_exit_code: int | None
    trigger_provenance: str
    capability_profile: str
    capability_profile_sha256: str
    runtime_contract_sha256: str
    source_fingerprint_sha256: str
    configuration_fingerprint_sha256: str
    dependency_fingerprint_sha256: str
    interpreter_fingerprint_sha256: str
    schedule_fingerprint_sha256: str
    blockers: tuple[str, ...]
    result_summary_sha256: str
    receipt_payload_sha256: str


_TERMINAL_RECEIPT_FIELDS = frozenset(field.name for field in fields(SchedulerTerminalReceipt))
_TERMINAL_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "scheduler_key",
        "run_id",
        "terminal_status",
        "intended_slot",
        "completed_at_utc",
        "receipt_id",
        "receipt_path",
        "receipt_sha256",
        "authority_advanced",
        "promotion_authority",
        "live_trading_authorized",
    }
)
_RUNTIME_IDENTITY_FIELDS = (
    "capability_profile",
    "capability_profile_sha256",
    "runtime_contract_sha256",
    "source_fingerprint_sha256",
    "configuration_fingerprint_sha256",
    "dependency_fingerprint_sha256",
    "interpreter_fingerprint_sha256",
    "schedule_fingerprint_sha256",
)


class SchedulerSlotAlreadyClaimed(RuntimeError):
    """Raised when another run already owns one deterministic scheduler slot."""

    def __init__(self, *, scheduler_key: str, intended_slot: str, owner_run_id: str):
        super().__init__("scheduler_intended_slot_already_claimed")
        self.evidence_reason_code = "scheduler_intended_slot_already_claimed"
        self.evidence_reason_only = True
        self.scheduler_key = scheduler_key
        self.intended_slot = intended_slot
        self.owner_run_id = owner_run_id


def new_scheduler_run_id(*, scheduler_key: str, now: datetime | None = None) -> str:
    """Create a collision-resistant run identifier without embedding secrets."""

    _validate_safe_id(scheduler_key, field="scheduler_key")
    observed = _as_utc(now or datetime.now(UTC))
    stamp = observed.strftime("%Y%m%dT%H%M%S%fZ")
    return f"{scheduler_key}_{stamp}_{uuid4().hex[:16]}"


def build_scheduler_run_intent(
    *,
    run_id: str,
    intended_slot: str,
    runtime_identity: dict[str, Any],
    started_at: datetime,
) -> dict[str, Any]:
    """Build the immutable start record used for abandoned-run recovery."""

    scheduler_key = str(runtime_identity.get("scheduler_key", "")).strip()
    _validate_safe_id(run_id, field="run_id")
    _validate_safe_id(scheduler_key, field="scheduler_key")
    if not intended_slot.strip():
        raise ValueError("intended_slot is required")
    material = {
        "schema_version": RUN_INTENT_SCHEMA_VERSION,
        "run_id": run_id,
        "scheduler_key": scheduler_key,
        "intended_slot": intended_slot.strip(),
        "started_at_utc": _as_utc(started_at).isoformat(),
        **current_scheduler_process_identity(),
        **_identity_fields(runtime_identity),
        "authority_advanced": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    payload_hash = _payload_sha256(material)
    return {
        **material,
        "run_intent_id": f"schedulerrunintent_{payload_hash[:24]}",
        "run_intent_payload_sha256": payload_hash,
    }


def publish_scheduler_run_intent(root: Path, intent: dict[str, Any]) -> Path:
    """Publish one immutable run intent before a supervised child starts."""

    scheduler_key, run_id = _validated_identity(intent)
    if intent.get("schema_version") != RUN_INTENT_SCHEMA_VERSION:
        raise ValueError("invalid scheduler run intent schema")
    _validate_run_intent_payload(intent)
    _validate_zero_authority(intent)
    path = _intent_path(root, scheduler_key=scheduler_key, run_id=run_id)
    with scheduler_terminal_write_lock(root, run_id=run_id):
        write_immutable_json(
            path,
            intent,
            publication_scope=TERMINAL_PUBLICATION_SCOPE,
        )
    return path


def load_validated_scheduler_run_intent(
    root: Path,
    *,
    scheduler_key: str,
    run_id: str,
) -> dict[str, Any]:
    """Load one exact self-sealed run intent for recovery decisions."""

    _validate_safe_id(scheduler_key, field="scheduler_key")
    _validate_safe_id(run_id, field="run_id")
    path = _intent_path(root, scheduler_key=scheduler_key, run_id=run_id)
    _require_regular_path(
        path,
        root=Path(root),
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="scheduler_run_intent",
    )
    intent = _load_bounded_json_object(
        path,
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="scheduler_run_intent",
    )
    observed_scheduler, observed_run = _validated_identity(intent)
    if observed_scheduler != scheduler_key or observed_run != run_id:
        raise ValueError("scheduler_run_intent_path_identity_mismatch")
    _validate_run_intent_payload(intent)
    _validate_zero_authority(intent)
    return intent


def _validate_run_intent_payload(intent: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "run_id",
        "scheduler_key",
        "intended_slot",
        "started_at_utc",
        "pid",
        "process_start_id",
        "process_start_source",
        "boot_id",
        "boot_id_source",
        "host",
        "trigger_provenance",
        *_RUNTIME_IDENTITY_FIELDS,
        "authority_advanced",
        "promotion_authority",
        "live_trading_authorized",
        "run_intent_id",
        "run_intent_payload_sha256",
    }
    if set(intent) != required:
        raise ValueError("scheduler_run_intent_schema_fields_invalid")
    if intent.get("schema_version") != RUN_INTENT_SCHEMA_VERSION:
        raise ValueError("scheduler_run_intent_schema_version_invalid")
    pid = intent.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("scheduler_run_intent_pid_invalid")
    for field in (
        "intended_slot",
        "process_start_id",
        "process_start_source",
        "boot_id",
        "boot_id_source",
        "host",
        "trigger_provenance",
        *_RUNTIME_IDENTITY_FIELDS,
    ):
        value = intent.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 512:
            raise ValueError(f"scheduler_run_intent_{field}_invalid")
    _parse_utc_timestamp(intent.get("started_at_utc"))
    material = {
        key: value
        for key, value in intent.items()
        if key not in {"run_intent_id", "run_intent_payload_sha256"}
    }
    payload_hash = _payload_sha256(material)
    if intent.get("run_intent_payload_sha256") != payload_hash:
        raise ValueError("scheduler_run_intent_payload_hash_mismatch")
    if intent.get("run_intent_id") != f"schedulerrunintent_{payload_hash[:24]}":
        raise ValueError("scheduler_run_intent_id_mismatch")


def claim_scheduler_slot(
    root: Path,
    *,
    run_id: str,
    intended_slot: str,
    runtime_identity: dict[str, Any],
    claimed_at: datetime,
) -> tuple[dict[str, Any], Path]:
    """Claim one scheduler slot durably; a different run can never replace it."""

    scheduler_key = str(runtime_identity.get("scheduler_key", "")).strip()
    _validate_safe_id(run_id, field="run_id")
    _validate_safe_id(scheduler_key, field="scheduler_key")
    slot = intended_slot.strip()
    if not slot:
        raise ValueError("intended_slot is required")
    material: dict[str, Any] = {
        "schema_version": SLOT_CLAIM_SCHEMA_VERSION,
        "scheduler_key": scheduler_key,
        "intended_slot": slot,
        "run_id": run_id,
        "claimed_at_utc": _as_utc(claimed_at).isoformat(),
        **_identity_fields(runtime_identity),
        "authority_advanced": False,
        "promotion_authority": False,
        "live_trading_authorized": False,
    }
    claim_hash = _payload_sha256(material)
    claim = {
        **material,
        "slot_claim_id": f"schedulerslot_{claim_hash[:24]}",
        "slot_claim_payload_sha256": claim_hash,
    }
    path = _slot_claim_path(
        root,
        scheduler_key=scheduler_key,
        intended_slot=slot,
    )
    with scheduler_terminal_write_lock(root, run_id=run_id):
        try:
            create_exclusive_json(
                path,
                claim,
                publication_scope=TERMINAL_PUBLICATION_SCOPE,
            )
        except FileExistsError:
            existing = load_validated_scheduler_slot_claim(
                root,
                scheduler_key=scheduler_key,
                intended_slot=slot,
            )
            if existing.get("run_id") == run_id and existing == claim:
                return existing, path
            raise SchedulerSlotAlreadyClaimed(
                scheduler_key=scheduler_key,
                intended_slot=slot,
                owner_run_id=str(existing.get("run_id", "")),
            ) from None
    return claim, path


def load_validated_scheduler_slot_claim(
    root: Path,
    *,
    scheduler_key: str,
    intended_slot: str,
    expected_run_id: str | None = None,
    expected_runtime_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load and validate the immutable owner of one deterministic slot."""

    _validate_safe_id(scheduler_key, field="scheduler_key")
    if expected_run_id is not None:
        _validate_safe_id(expected_run_id, field="run_id")
    slot = intended_slot.strip()
    if not slot:
        raise ValueError("scheduler_slot_claim_intended_slot_missing")
    path = _slot_claim_path(root, scheduler_key=scheduler_key, intended_slot=slot)
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(path)
    _require_regular_path(
        path,
        root=Path(root),
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="scheduler_slot_claim",
    )
    claim = _load_bounded_json_object(
        path,
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="scheduler_slot_claim",
    )
    required = {
        "schema_version",
        "scheduler_key",
        "intended_slot",
        "run_id",
        "claimed_at_utc",
        "trigger_provenance",
        *_RUNTIME_IDENTITY_FIELDS,
        "authority_advanced",
        "promotion_authority",
        "live_trading_authorized",
        "slot_claim_id",
        "slot_claim_payload_sha256",
    }
    if set(claim) != required:
        raise ValueError("scheduler_slot_claim_schema_fields_invalid")
    if claim.get("schema_version") != SLOT_CLAIM_SCHEMA_VERSION:
        raise ValueError("scheduler_slot_claim_schema_version_invalid")
    observed_scheduler, observed_run_id = _validated_identity(claim)
    if observed_scheduler != scheduler_key or claim.get("intended_slot") != slot:
        raise ValueError("scheduler_slot_claim_identity_mismatch")
    if expected_run_id is not None and observed_run_id != expected_run_id:
        raise ValueError("scheduler_slot_claim_run_id_mismatch")
    _validate_zero_authority(claim)
    if claim.get("trigger_provenance") != "launchd":
        raise ValueError("scheduler_slot_claim_launch_provenance_invalid")
    _parse_utc_timestamp(claim.get("claimed_at_utc"))
    material = {
        key: value
        for key, value in claim.items()
        if key not in {"slot_claim_id", "slot_claim_payload_sha256"}
    }
    claim_hash = _payload_sha256(material)
    if claim.get("slot_claim_payload_sha256") != claim_hash:
        raise ValueError("scheduler_slot_claim_payload_hash_mismatch")
    if claim.get("slot_claim_id") != f"schedulerslot_{claim_hash[:24]}":
        raise ValueError("scheduler_slot_claim_id_mismatch")
    if expected_runtime_identity is not None:
        for field in _RUNTIME_IDENTITY_FIELDS:
            if claim.get(field) != expected_runtime_identity.get(field):
                raise ValueError(f"scheduler_slot_claim_runtime_mismatch:{field}")
    return claim


def build_scheduler_terminal_receipt(
    *,
    run_id: str,
    intended_slot: str,
    runtime_identity: dict[str, Any],
    started_at: datetime,
    completed_at: datetime,
    terminal_status: TerminalStatus,
    process_health: ProcessHealth,
    business_state: BusinessState,
    retryable: bool,
    intended_slot_credit: bool,
    blockers: list[str] | tuple[str, ...] = (),
    result_summary: dict[str, Any] | None = None,
    external_calls: int = 0,
    external_credits_reserved: int = 0,
    external_credits_consumed: int = 0,
    external_credits_reconciled: int = 0,
    order_attempts: int = 0,
    order_submissions: int = 0,
    child_exit_code: int | None = None,
    authority_advanced: bool = False,
    promotion_authority: bool = False,
    live_trading_authorized: bool = False,
) -> dict[str, Any]:
    """Build and validate a complete scheduler terminal receipt."""

    scheduler_key = str(runtime_identity.get("scheduler_key", "")).strip()
    _validate_safe_id(run_id, field="run_id")
    _validate_safe_id(scheduler_key, field="scheduler_key")
    if not intended_slot.strip():
        raise ValueError("intended_slot is required")
    if terminal_status not in _TERMINAL_STATUSES:
        raise ValueError(f"invalid terminal_status: {terminal_status}")
    if process_health not in _PROCESS_HEALTH:
        raise ValueError(f"invalid process_health: {process_health}")
    if business_state not in _BUSINESS_STATES:
        raise ValueError(f"invalid business_state: {business_state}")

    started = _as_utc(started_at)
    completed = _as_utc(completed_at)
    if completed < started:
        raise ValueError("completed_at precedes started_at")
    counts = {
        "external_calls": external_calls,
        "external_credits_reserved": external_credits_reserved,
        "external_credits_consumed": external_credits_consumed,
        "external_credits_reconciled": external_credits_reconciled,
        "order_attempts": order_attempts,
        "order_submissions": order_submissions,
    }
    for name, value in counts.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if external_credits_reconciled > external_credits_consumed:
        raise ValueError("reconciled credits exceed consumed credits")
    if order_submissions > order_attempts:
        raise ValueError("order submissions exceed order attempts")

    normalized_blockers = tuple(
        sorted({str(value).strip() for value in blockers if str(value).strip()})
    )
    _validate_semantics(
        terminal_status=terminal_status,
        business_state=business_state,
        process_health=process_health,
        retryable=retryable,
        intended_slot_credit=intended_slot_credit,
        blockers=normalized_blockers,
        authority_advanced=authority_advanced,
        promotion_authority=promotion_authority,
        live_trading_authorized=live_trading_authorized,
        order_attempts=order_attempts,
        order_submissions=order_submissions,
    )

    summary_hash = _payload_sha256(result_summary or {})
    base: dict[str, Any] = {
        "schema_version": TERMINAL_RECEIPT_SCHEMA_VERSION,
        "run_id": run_id,
        "scheduler_key": scheduler_key,
        "intended_slot": intended_slot.strip(),
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "terminal_status": terminal_status,
        "process_health": process_health,
        "business_state": business_state,
        "retryable": bool(retryable),
        "intended_slot_credit": bool(intended_slot_credit),
        "authority_advanced": bool(authority_advanced),
        "promotion_authority": bool(promotion_authority),
        "live_trading_authorized": bool(live_trading_authorized),
        **counts,
        "child_exit_code": child_exit_code,
        **_identity_fields(runtime_identity),
        "blockers": list(normalized_blockers),
        "result_summary_sha256": summary_hash,
    }
    payload_hash = _payload_sha256(base)
    base["receipt_id"] = f"schedulerterminal_{payload_hash[:24]}"
    base["receipt_payload_sha256"] = payload_hash
    receipt = SchedulerTerminalReceipt(
        **{
            **base,
            "blockers": tuple(base["blockers"]),
        }
    )
    return {
        **asdict(receipt),
        "blockers": list(receipt.blockers),
    }


def publish_scheduler_terminal_receipt(
    root: Path,
    receipt: dict[str, Any],
) -> dict[str, Path]:
    """Publish exactly one immutable terminal receipt and its latest pointer."""

    scheduler_key, run_id = _validated_identity(receipt)
    if receipt.get("schema_version") != TERMINAL_RECEIPT_SCHEMA_VERSION:
        raise ValueError("invalid terminal receipt schema")
    _validate_zero_authority(receipt)
    expected_hash = str(receipt.get("receipt_payload_sha256", ""))
    material = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "receipt_payload_sha256"}
    }
    if expected_hash != _payload_sha256(material):
        raise ValueError("terminal receipt payload hash mismatch")
    expected_id = f"schedulerterminal_{expected_hash[:24]}"
    if receipt.get("receipt_id") != expected_id:
        raise ValueError("terminal receipt id mismatch")
    if receipt.get("intended_slot_credit") is True:
        load_validated_scheduler_slot_claim(
            root,
            scheduler_key=scheduler_key,
            intended_slot=str(receipt.get("intended_slot", "")),
            expected_run_id=run_id,
            expected_runtime_identity=receipt,
        )

    immutable_path = _terminal_path(
        root,
        scheduler_key=scheduler_key,
        run_id=run_id,
    )
    pointer_path = (
        root / "reports" / "active" / "scheduler_terminal_receipts" / f"{scheduler_key}_latest.json"
    )
    with scheduler_terminal_write_lock(root, run_id=run_id):
        write_immutable_json(
            immutable_path,
            receipt,
            publication_scope=TERMINAL_PUBLICATION_SCOPE,
        )
        receipt_hash = sha256(immutable_path.read_bytes()).hexdigest()
        pointer = {
            "schema_version": TERMINAL_POINTER_SCHEMA_VERSION,
            "scheduler_key": scheduler_key,
            "run_id": run_id,
            "terminal_status": receipt["terminal_status"],
            "intended_slot": receipt["intended_slot"],
            "completed_at_utc": receipt["completed_at_utc"],
            "receipt_id": receipt["receipt_id"],
            "receipt_path": immutable_path.relative_to(root).as_posix(),
            "receipt_sha256": receipt_hash,
            "authority_advanced": False,
            "promotion_authority": False,
            "live_trading_authorized": False,
        }
        atomic_write_text(
            pointer_path,
            json.dumps(pointer, indent=2, sort_keys=True) + "\n",
            publication_scope=TERMINAL_PUBLICATION_SCOPE,
        )
    return {"terminal_receipt": immutable_path, "terminal_pointer": pointer_path}


def load_validated_scheduler_terminal_receipt(
    root: Path,
    *,
    scheduler_key: str,
    expected_runtime_identity: dict[str, Any] | None = None,
    require_launchd: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load one latest terminal receipt only after its full contract is proven."""

    _validate_safe_id(scheduler_key, field="scheduler_key")
    root = Path(root)
    pointer_path = (
        root / "reports" / "active" / "scheduler_terminal_receipts" / f"{scheduler_key}_latest.json"
    )
    _require_regular_path(
        pointer_path,
        root=root,
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="terminal_pointer",
    )
    pointer = _load_bounded_json_object(
        pointer_path,
        maximum_bytes=MAX_TERMINAL_POINTER_BYTES,
        token="terminal_pointer",
    )
    if set(pointer) != _TERMINAL_POINTER_FIELDS:
        raise ValueError("terminal_pointer_schema_fields_invalid")
    if pointer.get("schema_version") != TERMINAL_POINTER_SCHEMA_VERSION:
        raise ValueError("terminal_pointer_schema_version_invalid")
    pointer_scheduler, run_id = _validated_identity(pointer)
    if pointer_scheduler != scheduler_key:
        raise ValueError("terminal_pointer_scheduler_mismatch")
    _validate_zero_authority(pointer)

    expected_path = _terminal_path(
        root,
        scheduler_key=scheduler_key,
        run_id=run_id,
    )
    expected_relative = expected_path.relative_to(root).as_posix()
    if pointer.get("receipt_path") != expected_relative:
        raise ValueError("terminal_pointer_receipt_path_invalid")
    receipt, validated_path = load_validated_scheduler_terminal_receipt_by_run(
        root,
        scheduler_key=scheduler_key,
        run_id=run_id,
        expected_runtime_identity=expected_runtime_identity,
        require_launchd=require_launchd,
    )
    receipt_bytes = validated_path.read_bytes()
    if sha256(receipt_bytes).hexdigest() != pointer.get("receipt_sha256"):
        raise ValueError("terminal_receipt_file_hash_mismatch")
    for field in (
        "scheduler_key",
        "run_id",
        "terminal_status",
        "intended_slot",
        "completed_at_utc",
        "receipt_id",
        "authority_advanced",
        "promotion_authority",
        "live_trading_authorized",
    ):
        if pointer.get(field) != receipt.get(field):
            raise ValueError(f"terminal_pointer_receipt_{field}_mismatch")
    return receipt, expected_path


def load_validated_scheduler_terminal_receipt_by_run(
    root: Path,
    *,
    scheduler_key: str,
    run_id: str,
    expected_runtime_identity: dict[str, Any] | None = None,
    require_launchd: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Load one immutable terminal receipt by its scheduler and run identity."""

    _validate_safe_id(scheduler_key, field="scheduler_key")
    _validate_safe_id(run_id, field="run_id")
    root = Path(root)
    receipt_path = _terminal_path(root, scheduler_key=scheduler_key, run_id=run_id)
    _require_regular_path(
        receipt_path,
        root=root,
        maximum_bytes=MAX_TERMINAL_RECEIPT_BYTES,
        token="terminal_receipt",
    )
    receipt = _load_bounded_json_object(
        receipt_path,
        maximum_bytes=MAX_TERMINAL_RECEIPT_BYTES,
        token="terminal_receipt",
    )
    _validate_terminal_receipt_payload(
        receipt,
        scheduler_key=scheduler_key,
        expected_runtime_identity=expected_runtime_identity,
        require_launchd=require_launchd,
    )
    if receipt.get("run_id") != run_id:
        raise ValueError("terminal_receipt_run_id_mismatch")
    if receipt.get("intended_slot_credit") is True:
        load_validated_scheduler_slot_claim(
            root,
            scheduler_key=scheduler_key,
            intended_slot=str(receipt.get("intended_slot", "")),
            expected_run_id=run_id,
            expected_runtime_identity=receipt,
        )
    return receipt, receipt_path


def _validate_terminal_receipt_payload(
    receipt: dict[str, Any],
    *,
    scheduler_key: str,
    expected_runtime_identity: dict[str, Any] | None,
    require_launchd: bool,
) -> None:
    if set(receipt) != _TERMINAL_RECEIPT_FIELDS:
        raise ValueError("terminal_receipt_schema_fields_invalid")
    if receipt.get("schema_version") != TERMINAL_RECEIPT_SCHEMA_VERSION:
        raise ValueError("terminal_receipt_schema_version_invalid")
    observed_scheduler, _ = _validated_identity(receipt)
    if observed_scheduler != scheduler_key:
        raise ValueError("terminal_receipt_scheduler_mismatch")
    _validate_zero_authority(receipt)
    material = {
        key: value
        for key, value in receipt.items()
        if key not in {"receipt_id", "receipt_payload_sha256"}
    }
    payload_hash = _payload_sha256(material)
    if receipt.get("receipt_payload_sha256") != payload_hash:
        raise ValueError("terminal_receipt_payload_hash_mismatch")
    if receipt.get("receipt_id") != f"schedulerterminal_{payload_hash[:24]}":
        raise ValueError("terminal_receipt_id_mismatch")

    count_fields = (
        "external_calls",
        "external_credits_reserved",
        "external_credits_consumed",
        "external_credits_reconciled",
        "order_attempts",
        "order_submissions",
    )
    for field in count_fields:
        value = receipt.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"terminal_receipt_{field}_invalid")
    if receipt["external_credits_reconciled"] > receipt["external_credits_consumed"]:
        raise ValueError("terminal_receipt_credit_reconciliation_invalid")
    if receipt["order_submissions"] > receipt["order_attempts"]:
        raise ValueError("terminal_receipt_order_accounting_invalid")
    blockers = receipt.get("blockers")
    if (
        not isinstance(blockers, list)
        or any(not isinstance(value, str) or not value.strip() for value in blockers)
        or blockers != sorted(set(blockers))
    ):
        raise ValueError("terminal_receipt_blockers_invalid")
    for field in ("retryable", "intended_slot_credit", "authority_advanced"):
        if not isinstance(receipt.get(field), bool):
            raise ValueError(f"terminal_receipt_{field}_invalid")  # noqa: TRY004
    child_exit_code = receipt.get("child_exit_code")
    if child_exit_code is not None and (
        isinstance(child_exit_code, bool) or not isinstance(child_exit_code, int)
    ):
        raise ValueError("terminal_receipt_child_exit_code_invalid")
    for field in ("result_summary_sha256", "receipt_payload_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get(field, ""))):
            raise ValueError(f"terminal_receipt_{field}_invalid")
    started = _parse_utc_timestamp(receipt.get("started_at_utc"))
    completed = _parse_utc_timestamp(receipt.get("completed_at_utc"))
    if completed < started:
        raise ValueError("terminal_receipt_time_order_invalid")
    _validate_semantics(
        terminal_status=str(receipt.get("terminal_status", "")),
        business_state=str(receipt.get("business_state", "")),
        process_health=str(receipt.get("process_health", "")),
        retryable=receipt["retryable"],
        intended_slot_credit=receipt["intended_slot_credit"],
        blockers=tuple(blockers),
        authority_advanced=receipt["authority_advanced"],
        promotion_authority=receipt["promotion_authority"],
        live_trading_authorized=receipt["live_trading_authorized"],
        order_attempts=receipt["order_attempts"],
        order_submissions=receipt["order_submissions"],
    )
    if require_launchd and receipt.get("trigger_provenance") != "launchd":
        raise ValueError("terminal_receipt_launch_provenance_invalid")
    if expected_runtime_identity is not None:
        for field in _RUNTIME_IDENTITY_FIELDS:
            if receipt.get(field) != expected_runtime_identity.get(field):
                raise ValueError(f"terminal_receipt_runtime_identity_mismatch:{field}")


def _parse_utc_timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError("terminal_receipt_timestamp_invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("terminal_receipt_timestamp_invalid")
    return parsed.astimezone(UTC)


def _require_regular_path(
    path: Path,
    *,
    root: Path,
    maximum_bytes: int,
    token: str,
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{token}_missing_or_nonregular")
    current = path.parent
    while current != root:
        if current.is_symlink():
            raise ValueError(f"{token}_parent_symlink_invalid")
        if root not in current.parents:
            raise ValueError(f"{token}_path_escape")
        current = current.parent
    if path.stat().st_size > maximum_bytes:
        raise ValueError(f"{token}_oversized")


def _load_bounded_json_object(
    path: Path,
    *,
    maximum_bytes: int,
    token: str,
) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > maximum_bytes:
        raise ValueError(f"{token}_oversized")
    return _decode_json_object(payload, token=token)


def _decode_json_object(payload: bytes, *, token: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{token}_duplicate_key")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{token}_unreadable") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{token}_not_object")  # noqa: TRY004
    return decoded


def scheduler_terminal_exit_code(receipt: dict[str, Any]) -> int:
    """Return zero only for a semantically complete terminal pass."""

    rule = TERMINAL_STATE_RULES.get(str(receipt.get("terminal_status", "")))
    if rule is None:
        return 2
    if (
        receipt.get("process_health") != rule.process_health
        or receipt.get("business_state") != rule.business_state
        or receipt.get("retryable") not in rule.retryable_values
        or receipt.get("intended_slot_credit") is not rule.intended_slot_credit
        or (rule.terminal_status == "PASS" and receipt.get("blockers"))
        or (rule.terminal_status != "PASS" and not receipt.get("blockers"))
    ):
        return 2
    return rule.exit_code


def terminal_receipt_exists(root: Path, *, scheduler_key: str, run_id: str) -> bool:
    _validate_safe_id(scheduler_key, field="scheduler_key")
    _validate_safe_id(run_id, field="run_id")
    return _terminal_path(root, scheduler_key=scheduler_key, run_id=run_id).is_file()


def _validate_semantics(
    *,
    terminal_status: str,
    business_state: str,
    process_health: str,
    retryable: bool,
    intended_slot_credit: bool,
    blockers: tuple[str, ...],
    authority_advanced: bool,
    promotion_authority: bool,
    live_trading_authorized: bool,
    order_attempts: int,
    order_submissions: int,
) -> None:
    rule = TERMINAL_STATE_RULES.get(terminal_status)
    if rule is None:
        raise ValueError("terminal status has no canonical state rule")
    if process_health != rule.process_health:
        raise ValueError(
            f"terminal {terminal_status} requires process health {rule.process_health}"
        )
    if business_state != rule.business_state:
        raise ValueError(
            f"terminal {terminal_status} requires business state {rule.business_state}"
        )
    if retryable not in rule.retryable_values:
        raise ValueError(f"terminal {terminal_status} has invalid retryability")
    if intended_slot_credit is not rule.intended_slot_credit:
        raise ValueError(f"terminal {terminal_status} has invalid intended-slot credit")
    passing = terminal_status == "PASS"
    if passing and blockers:
        raise ValueError("terminal PASS requires no blockers")
    if not passing and not blockers:
        raise ValueError("non-PASS terminal receipt requires at least one blocker")
    if authority_advanced and not passing:
        raise ValueError("authority cannot advance from a non-PASS terminal state")
    if promotion_authority or live_trading_authorized:
        raise ValueError("scheduler terminal receipts cannot grant trade authority")
    if order_attempts or order_submissions:
        raise ValueError("research scheduler terminal receipts cannot contain order activity")


def _identity_fields(runtime_identity: dict[str, Any]) -> dict[str, Any]:
    required = (
        "trigger_provenance",
        "capability_profile",
        "capability_profile_sha256",
        "runtime_contract_sha256",
        "source_fingerprint_sha256",
        "configuration_fingerprint_sha256",
        "dependency_fingerprint_sha256",
        "interpreter_fingerprint_sha256",
        "schedule_fingerprint_sha256",
    )
    result: dict[str, Any] = {}
    for field in required:
        value = str(runtime_identity.get(field, "")).strip()
        if not value:
            raise ValueError(f"runtime identity missing {field}")
        result[field] = value
    return result


def _validated_identity(payload: dict[str, Any]) -> tuple[str, str]:
    scheduler_key = str(payload.get("scheduler_key", "")).strip()
    run_id = str(payload.get("run_id", "")).strip()
    _validate_safe_id(scheduler_key, field="scheduler_key")
    _validate_safe_id(run_id, field="run_id")
    return scheduler_key, run_id


def _validate_safe_id(value: str, *, field: str) -> None:
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"invalid {field}")


def _validate_zero_authority(payload: dict[str, Any]) -> None:
    for field in (
        "authority_advanced",
        "promotion_authority",
        "live_trading_authorized",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"scheduler terminal publication requires {field}=false")


def _intent_path(root: Path, *, scheduler_key: str, run_id: str) -> Path:
    return root / "data" / "research" / "scheduler_run_intents" / scheduler_key / f"{run_id}.json"


def _terminal_path(root: Path, *, scheduler_key: str, run_id: str) -> Path:
    return (
        root
        / "data"
        / "research"
        / "scheduler_terminal_receipts"
        / scheduler_key
        / f"{run_id}.json"
    )


def _slot_claim_path(root: Path, *, scheduler_key: str, intended_slot: str) -> Path:
    slot_digest = sha256(f"{scheduler_key}\0{intended_slot}".encode()).hexdigest()
    return (
        Path(root)
        / "data"
        / "research"
        / "scheduler_slot_claims"
        / scheduler_key
        / f"{slot_digest}.json"
    )


def _payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
