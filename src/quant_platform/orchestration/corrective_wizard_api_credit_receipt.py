"""Redacted, immutable Crypto Wizards API credit preflight evidence."""

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
from quant_platform.api_extraction import CryptoWizardsFetchError
from quant_platform.crypto_wizards_history import fetch_credits_used
from quant_platform.crypto_wizards_sweep import parse_wizard_credit_usage
from quant_platform.env import load_selected_env_keys
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import promote_staged_file

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_api_credit_receipt.v1"
ENDPOINT = "/v1beta/credits-used"


def capture_wizard_api_credit_receipt(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    api_key: str | None = None,
    fetcher: Callable[..., Any] = fetch_credits_used,
    daily_limit: int = 1000,
    protected_reserve: int = 100,
    required_credits: int = 0,
) -> CommandResult:
    """Call the zero-trade credit endpoint and persist only redacted evidence."""

    key = (api_key or os.getenv("CRYPTO_WIZARDS_API_KEY", "")).strip()
    key_source = "argument" if api_key else "environment"
    if not key:
        load_selected_env_keys(
            root / ".env.local",
            allowed_keys={"CRYPTO_WIZARDS_API_KEY"},
        )
        key = os.getenv("CRYPTO_WIZARDS_API_KEY", "").strip()
        key_source = ".env.local" if key else "missing"
    response: Any = None
    error = ""
    if key:
        try:
            response = fetcher(api_key=key)
        except (CryptoWizardsFetchError, OSError, TypeError, ValueError) as exc:
            error = f"credit_endpoint_failed:{safe_exception_code(exc)}"
    else:
        error = "crypto_wizards_api_key_missing"
    return publish_wizard_api_credit_receipt(
        root=root,
        now=now,
        response=response,
        api_key_present=bool(key),
        api_key_source=key_source,
        error=error,
        daily_limit=daily_limit,
        protected_reserve=protected_reserve,
        required_credits=required_credits,
    )


def publish_wizard_api_credit_receipt(
    *,
    root: Path,
    response: Any,
    now: datetime | None = None,
    api_key_present: bool,
    api_key_source: str,
    error: str = "",
    daily_limit: int = 1000,
    protected_reserve: int = 100,
    required_credits: int = 0,
) -> CommandResult:
    """Persist endpoint truth without storing credentials or response contents."""

    if daily_limit <= 0 or protected_reserve < 0 or required_credits < 0:
        raise ValueError("credit limits and requirements must be non-negative")
    checked_at = _as_utc(now)
    usage = parse_wizard_credit_usage(response, configured_limit=daily_limit)
    remaining = usage.remaining
    sufficient = bool(
        not error
        and usage.known
        and remaining is not None
        and remaining >= protected_reserve + required_credits
    )
    response_type = type(response).__name__ if response is not None else "none"
    response_sha256 = (
        sha256(_canonical_json(response).encode("utf-8")).hexdigest()
        if response is not None
        else ""
    )
    blockers = []
    if error:
        blockers.append(error)
    if not usage.known:
        blockers.append("credit_usage_unknown")
    if usage.known and not sufficient:
        blockers.append("insufficient_credits_after_protected_reserve")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "endpoint": ENDPOINT,
        "response_type": response_type,
        "response_sha256": response_sha256,
        "response_body_stored": False,
        "api_key_present": api_key_present,
        "api_key_source": api_key_source,
        "secret_value_stored": False,
        "credits_used": usage.used,
        "credit_limit": usage.limit,
        "credits_remaining": remaining,
        "credit_usage_known": usage.known,
        "credit_source_fields": usage.source_fields,
        "protected_reserve": protected_reserve,
        "required_credits": required_credits,
        "sufficient_after_reserve": sufficient,
        "status": (
            "PASS_AUTHENTICATED_CREDIT_PREFLIGHT" if sufficient else "BLOCKED_CREDIT_PREFLIGHT"
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_id"] = (
        "wizardcredit_" + sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:20]
    )
    immutable = (
        root
        / "data"
        / "research"
        / "wizard_api_credit_receipts"
        / checked_at.date().isoformat()
        / f"{payload['receipt_id']}.json"
    )
    active = root / "reports" / "active" / "wizard_api_credit_receipt.json"
    _write_or_validate_immutable_json(payload, immutable)
    active_payload = {
        **payload,
        "immutable_receipt_path": _relative(immutable, root),
        "immutable_receipt_sha256": sha256(immutable.read_bytes()).hexdigest(),
    }
    _atomic_json(active_payload, active)
    return CommandResult(
        paths={"status": active, "immutable_receipt": immutable},
        summary=active_payload,
    )


def _as_utc(value: datetime | None) -> datetime:
    timestamp = value or datetime.now(UTC)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (OSError, ValueError):
        return str(path)


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
            raise ValueError(f"immutable credit receipt mismatch: {path}")
        return
    _atomic_text(path, expected)
