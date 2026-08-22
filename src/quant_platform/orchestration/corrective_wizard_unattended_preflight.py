"""Fail-closed preflight for unattended Crypto Wizards vendor calls."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_canonical_status import (
    build_canonical_program_status,
    rebind_checkpoint_stage4_evidence,
)
from quant_platform.orchestration.corrective_daily_scheduler import _acquire_lock
from quant_platform.orchestration.corrective_program import complete_corrective_plan
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.corrective_wizard_api_credit_receipt import (
    capture_wizard_api_credit_receipt,
)
from quant_platform.orchestration.corrective_wizard_reset_readiness import (
    build_corrective_wizard_reset_readiness,
    validate_wizard_reset_readiness_receipt,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_unattended_external_preflight.v1"
DAILY_CREDIT_LIMIT = 1000
PROTECTED_CREDIT_RESERVE = 100
CRITICAL_LOCK_NAMES = (
    ".corrective_daily.lock",
    ".corrective_l2_capture.lock",
    ".corrective_wizard_proof.lock",
    ".corrective_registered_rerun.lock",
)
LOCK_STALE_TIMEOUT_SECONDS = 4 * 60 * 60


def build_wizard_unattended_external_preflight(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    reset_readiness_builder: Callable[..., CommandResult] = (
        build_corrective_wizard_reset_readiness
    ),
    reset_readiness_validator: Callable[..., dict[str, Any]] = (
        validate_wizard_reset_readiness_receipt
    ),
    checkpoint_refresher: Callable[..., CommandResult] = complete_corrective_plan,
    stage4_checkpoint_rebinder: Callable[..., CommandResult] = (rebind_checkpoint_stage4_evidence),
    canonical_status_builder: Callable[..., CommandResult] = build_canonical_program_status,
    credit_receipt_capturer: Callable[..., CommandResult] = capture_wizard_api_credit_receipt,
    lock_acquirer: Callable[..., None] = _acquire_lock,
    daily_credit_limit: int = DAILY_CREDIT_LIMIT,
    protected_reserve: int = PROTECTED_CREDIT_RESERVE,
    api_key: str | None = None,
) -> CommandResult:
    """Bind reset, canonical, and live credit evidence before unattended spend."""

    checked_at = _as_utc(now)
    active = root / "reports" / "active"
    manifest_path = active / "corrective_wizard_next_capture_manifest.json"
    manifest = _read_json(manifest_path)
    blockers: list[str] = []

    planned_credits = _safe_int(manifest.get("planned_credits"))
    pending_calls = _safe_int(manifest.get("pending_calls"))
    manifest_id = str(manifest.get("manifest_id", ""))
    reset_at = _try_parse_utc(str(manifest.get("next_external_attempt_eligible_at", "")))
    manifest_ready = bool(
        manifest.get("status") == "PASS"
        and manifest.get("manifest_enforced") is True
        and manifest.get("runtime_credit_preflight_required") is True
        and manifest_id.startswith("wizardcapture_")
        and pending_calls > 0
        and planned_credits > 0
        and reset_at is not None
        and checked_at >= reset_at
        and _authority_is_zero(manifest)
    )
    if not manifest_ready:
        blockers.append("unattended_frozen_manifest_not_due_or_invalid")

    reset: dict[str, Any] = {}
    reset_binding: dict[str, Any] = {"status": "BLOCKED", "blockers": ["not_run"]}
    if manifest_ready:
        try:
            reset_result = reset_readiness_builder(root=root, now=checked_at)
            reset = dict(reset_result.summary)
            reset_binding = reset_readiness_validator(root=root)
        except Exception as exc:  # noqa: BLE001 - preflight must fail closed
            blockers.append(f"unattended_reset_readiness_failed:{safe_exception_code(exc)}")
    reset_ready = bool(
        reset.get("status") == "PASS_RESET_AUTOMATION_READY"
        and reset.get("capture_window_status") == "DUE_AFTER_RESET"
        and reset.get("manifest_id") == manifest_id
        and _safe_int(reset.get("pending_calls")) == pending_calls
        and _safe_int(reset.get("planned_credits")) == planned_credits
        and reset_binding.get("status") == "PASS"
        and _authority_is_zero(reset)
    )
    if manifest_ready and not reset_ready:
        blockers.append("unattended_reset_readiness_not_bound")

    checkpoint: dict[str, Any] = {}
    if manifest_ready and reset_ready:
        try:
            checkpoint_result = checkpoint_refresher(root=root, now=checked_at)
            checkpoint = dict(checkpoint_result.summary)
        except Exception as exc:  # noqa: BLE001 - preflight must fail closed
            blockers.append(f"unattended_checkpoint_refresh_failed:{safe_exception_code(exc)}")
    checkpoint_ready = bool(
        checkpoint.get("implementation_status") == "COMPLETE"
        and checkpoint.get("generated_at_utc") == checked_at.isoformat()
        and checkpoint.get("current_decision") == "CONTINUE_RESEARCH_ONLY"
        and checkpoint.get("testnet_order_authority") is False
        and checkpoint.get("live_trading_authorized") is False
    )
    if manifest_ready and reset_ready and not checkpoint_ready:
        blockers.append("unattended_checkpoint_refresh_not_current")

    rebind: dict[str, Any] = {}
    rebind_ready = False
    canonical: dict[str, Any] = {}
    acquired_locks: list[Path] = []
    if manifest_ready and reset_ready and checkpoint_ready:
        try:
            for lock_name in CRITICAL_LOCK_NAMES:
                lock_path = active / lock_name
                lock_acquirer(
                    lock_path,
                    now=checked_at,
                    timeout_seconds=LOCK_STALE_TIMEOUT_SECONDS,
                )
                acquired_locks.append(lock_path)
            rebind_result = stage4_checkpoint_rebinder(root=root, now=checked_at)
            rebind = dict(rebind_result.summary)
            rebind_ready = bool(
                rebind.get("status") == "PASS_STAGE4_CHECKPOINT_REBOUND"
                and rebind.get("generated_at_utc") == checked_at.isoformat()
                and rebind.get("only_stage4_evidence_progress_changed") is True
                and _immutable_binding_valid(rebind, root=root)
                and _authority_is_zero(rebind)
            )
            if not rebind_ready:
                blockers.append("unattended_stage4_checkpoint_rebinding_not_current")
            else:
                canonical_result = canonical_status_builder(root=root, now=checked_at)
                canonical = dict(canonical_result.summary)
        except Exception as exc:  # noqa: BLE001 - preflight must fail closed
            blockers.append(
                f"unattended_locked_canonical_refresh_failed:{safe_exception_code(exc)}"
            )
        finally:
            for lock_path in reversed(acquired_locks):
                lock_path.unlink(missing_ok=True)
    canonical_ready = bool(
        rebind_ready
        and canonical.get("status") == "PASS_CANONICAL_CURRENT"
        and canonical.get("generated_at_utc") == checked_at.isoformat()
        and _immutable_binding_valid(canonical, root=root)
        and _authority_is_zero(canonical)
    )
    if manifest_ready and reset_ready and checkpoint_ready and not canonical_ready:
        blockers.append("unattended_canonical_status_not_current")

    credit: dict[str, Any] = {}
    if manifest_ready and reset_ready and checkpoint_ready and canonical_ready:
        try:
            credit_kwargs: dict[str, Any] = {
                "root": root,
                "now": checked_at,
                "daily_limit": daily_credit_limit,
                "protected_reserve": protected_reserve,
                "required_credits": planned_credits,
            }
            if api_key is not None:
                credit_kwargs["api_key"] = api_key
            credit_result = credit_receipt_capturer(
                **credit_kwargs,
            )
            credit = dict(credit_result.summary)
        except Exception as exc:  # noqa: BLE001 - preflight must fail closed
            blockers.append(f"unattended_credit_preflight_failed:{safe_exception_code(exc)}")
    credit_ready = bool(
        credit.get("status") == "PASS_AUTHENTICATED_CREDIT_PREFLIGHT"
        and credit.get("checked_at_utc") == checked_at.isoformat()
        and _safe_int(credit.get("credit_limit")) == daily_credit_limit
        and _safe_int(credit.get("protected_reserve")) == protected_reserve
        and _safe_int(credit.get("required_credits")) == planned_credits
        and credit.get("sufficient_after_reserve") is True
        and credit.get("response_body_stored") is False
        and credit.get("secret_value_stored") is False
        and _immutable_binding_valid(credit, root=root)
        and _authority_is_zero(credit)
    )
    if manifest_ready and reset_ready and checkpoint_ready and canonical_ready and not credit_ready:
        blockers.append("unattended_authenticated_credit_preflight_not_bound")

    blockers = list(dict.fromkeys(blockers))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": (
            "PASS_UNATTENDED_EXTERNAL_PREFLIGHT"
            if not blockers
            else "BLOCKED_UNATTENDED_EXTERNAL_PREFLIGHT"
        ),
        "manifest_id": manifest_id,
        "manifest_path": _relative(manifest_path, root),
        "pending_calls": pending_calls,
        "planned_credits": planned_credits,
        "next_external_attempt_eligible_at": reset_at.isoformat() if reset_at else "",
        "daily_credit_limit": daily_credit_limit,
        "protected_credit_reserve": protected_reserve,
        "reset_readiness_status": str(reset.get("status", "NOT_RUN")),
        "reset_readiness_receipt_id": str(reset.get("receipt_id", "")),
        "reset_readiness_binding_status": str(reset_binding.get("status", "BLOCKED")),
        "checkpoint_refresh_status": str(checkpoint.get("implementation_status", "NOT_RUN")),
        "checkpoint_operational_status": str(
            checkpoint.get("operational_acceptance_status", "NOT_RUN")
        ),
        "critical_lock_names": list(CRITICAL_LOCK_NAMES),
        "stage4_checkpoint_rebinding_status": str(rebind.get("status", "NOT_RUN")),
        "stage4_checkpoint_rebinding_receipt_id": str(rebind.get("receipt_id", "")),
        "canonical_status": str(canonical.get("status", "NOT_RUN")),
        "canonical_receipt_id": str(canonical.get("receipt_id", "")),
        "credit_preflight_status": str(credit.get("status", "NOT_RUN")),
        "credit_receipt_id": str(credit.get("receipt_id", "")),
        "credits_used": credit.get("credits_used"),
        "credits_remaining": credit.get("credits_remaining"),
        "blockers": blockers,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_id"] = (
        "wizardpreflight_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    immutable = (
        root
        / "data"
        / "research"
        / "wizard_unattended_external_preflights"
        / checked_at.date().isoformat()
        / f"{payload['receipt_id']}.json"
    )
    status_path = active / "wizard_unattended_external_preflight.json"
    _write_or_validate_immutable_json(payload, immutable)
    active_payload = {
        **payload,
        "immutable_receipt_path": _relative(immutable, root),
        "immutable_receipt_sha256": _file_sha256(immutable),
    }
    _atomic_json(active_payload, status_path)
    return CommandResult(
        paths={"status": status_path, "immutable_receipt": immutable},
        summary=active_payload,
    )


def _immutable_binding_valid(payload: dict[str, Any], *, root: Path) -> bool:
    relative = str(payload.get("immutable_receipt_path", ""))
    expected_sha = str(payload.get("immutable_receipt_sha256", ""))
    path = _safe_root_file(root, relative)
    if path is None or not path.is_file() or _file_sha256(path) != expected_sha:
        return False
    immutable = _read_json(path)
    return bool(
        immutable
        and all(payload.get(key) == value for key, value in immutable.items())
        and _authority_is_zero(immutable)
    )


def _authority_is_zero(payload: dict[str, Any]) -> bool:
    return all(
        payload.get(field) is False
        for field in (
            "candidate_promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    )


def _safe_root_file(root: Path, relative: str) -> Path | None:
    if not relative or Path(relative).is_absolute():
        return None
    try:
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _try_parse_utc(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


def _file_sha256(path: Path) -> str:
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        promote_staged_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != expected:
            raise ValueError(f"immutable unattended preflight mismatch: {path}")
        return
    _atomic_text(path, expected)
