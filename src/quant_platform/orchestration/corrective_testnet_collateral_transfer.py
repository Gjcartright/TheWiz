"""Fail-closed Hyperliquid Testnet spot-to-perp collateral transfer control."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd
from eth_account import Account

from quant_platform.active_pipeline import CommandResult
from quant_platform.hyperliquid_testnet import (
    HyperliquidTestnetConfig,
    hyperliquid_testnet_margin_snapshot,
    read_hyperliquid_agent_key_from_keychain,
)
from quant_platform.orchestration.corrective_release_gates import (
    _validated_testnet_candidate_receipt,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_APPROVAL_VERSION,
    validate_testnet_smoke_approval,
)

ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_SCHEMA = "thewiz.hyperliquid_testnet_collateral_transfer_preflight.v1"
EXECUTION_SCHEMA = "thewiz.hyperliquid_testnet_collateral_transfer_execution.v1"
RESERVATION_SCHEMA = "thewiz.hyperliquid_testnet_collateral_transfer_reservation.v1"
ENABLE_ENV = "QPA_ENABLE_HYPERLIQUID_TESTNET_COLLATERAL_TRANSFER"
EXECUTION_ACKNOWLEDGEMENT = "TRANSFER TESTNET SPOT USDC TO PERP ONCE"
MAX_PREFLIGHT_AGE_SECONDS = 300
MINIMUM_TRANSFER_USD = 20.0
MAXIMUM_TRANSFER_USD = 25.0
RECONCILIATION_TOLERANCE_USD = 0.01


CandidateValidator = Callable[[Path, dict[str, Any]], tuple[bool, list[str]]]
ApprovalValidator = Callable[
    [str, Path, HyperliquidTestnetConfig], dict[str, object]
]
MarginReader = Callable[[HyperliquidTestnetConfig], dict[str, object]]
KeychainReader = Callable[[str, str], str]
ExchangeFactory = Callable[[Any, HyperliquidTestnetConfig], Any]


def build_testnet_collateral_transfer_preflight(
    *,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    approval_id: str = "",
    amount_usd: float = 25.0,
    now: datetime | None = None,
    candidate_validator: CandidateValidator | None = None,
    approval_validator: ApprovalValidator | None = None,
    margin_reader: MarginReader | None = None,
) -> CommandResult:
    """Build an immutable, no-key, no-transfer readiness receipt."""

    evaluated_at = _as_utc(now)
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    candidate_path = active / "testnet_candidate_receipt.json"
    approval_path = active / "hyperliquid_testnet_smoke_approval.json"
    candidate = _read_json(candidate_path)
    approval = _read_json(approval_path)
    blockers = list(resolved.configuration_blockers())

    amount = _bounded_amount(amount_usd)
    if amount is None:
        blockers.append("testnet_collateral_transfer_amount_outside_20_to_25_usd")
        amount = float(amount_usd) if _finite(amount_usd) is not None else 0.0

    validate_candidate = candidate_validator or _default_candidate_validator
    candidate_ready, candidate_blockers = validate_candidate(root, candidate)
    if not candidate_ready:
        blockers.extend(candidate_blockers or ["testnet_candidate_receipt_invalid"])

    validate_approval = approval_validator or _default_approval_validator
    approval_result = validate_approval(str(approval_id), root, resolved)
    if approval_result.get("execution_allowed") is not True:
        supplied = approval_result.get("blockers")
        if isinstance(supplied, list):
            blockers.extend(str(value) for value in supplied if str(value))
        else:
            blockers.append("testnet_smoke_approval_invalid")
    blockers.extend(_collateral_policy_blockers(approval, amount))

    read_margin = margin_reader or _default_margin_reader
    try:
        margin = read_margin(resolved)
    except Exception as exc:  # noqa: BLE001 - boundary receipt records the failure class.
        margin = {
            "status": "BLOCKED",
            "blockers": f"testnet_collateral_margin_reader_failed:{type(exc).__name__}",
        }
    margin_blockers = {
        value
        for value in str(margin.get("blockers", "")).split(";")
        if value
    }
    allowed_margin_blockers = {"testnet_usdc_requires_spot_to_perp_transfer"}
    blockers.extend(sorted(margin_blockers - allowed_margin_blockers))

    account_value = _finite(margin.get("account_value_usd"))
    spot_usdc = _finite(margin.get("spot_usdc_usd"))
    margin_used = _finite(margin.get("margin_used_usd"))
    open_positions = int(_finite(margin.get("open_positions")) or 0)
    transfer_required = bool(
        account_value is not None
        and amount > 0.0
        and account_value < amount - RECONCILIATION_TOLERANCE_USD
    )
    if account_value is None:
        blockers.append("testnet_collateral_perp_account_value_missing")
    if transfer_required and (spot_usdc is None or spot_usdc < amount):
        blockers.append("testnet_collateral_spot_usdc_below_approved_transfer")
    if margin_used is None or margin_used > RECONCILIATION_TOLERANCE_USD:
        blockers.append("testnet_collateral_margin_must_be_unused_before_transfer")
    if open_positions != 0:
        blockers.append("testnet_collateral_open_positions_present")

    reservation_path = _reservation_path(
        root,
        approval_id=str(approval_id),
        candidate_receipt_id=str(candidate.get("candidate_receipt_id", "")),
        amount_usd=amount,
    )
    if reservation_path.exists():
        blockers.append("testnet_collateral_transfer_approval_already_reserved_or_consumed")

    blockers = _unique(blockers)
    if not transfer_required and account_value is not None:
        status = "NO_TRANSFER_REQUIRED"
    elif blockers:
        status = "BLOCKED"
    else:
        status = "READY_REQUIRES_EXPLICIT_EXECUTION"

    core = {
        "schema_version": PREFLIGHT_SCHEMA,
        "evaluated_at_utc": evaluated_at.isoformat(),
        "status": status,
        "blockers": blockers,
        "network": resolved.network,
        "base_url": resolved.base_url,
        "master_address": str(resolved.master_address or ""),
        "agent_address": str(resolved.agent_address or ""),
        "candidate_receipt_id": str(candidate.get("candidate_receipt_id", "")),
        "candidate_receipt_path": _relative(candidate_path, root),
        "candidate_receipt_sha256": _path_sha256(candidate_path),
        "approval_id": str(approval_id),
        "approval_version": str(approval.get("approval_version", "")),
        "approval_path": _relative(approval_path, root),
        "approval_sha256": _path_sha256(approval_path),
        "approval_payload_hash": str(approval.get("payload_hash", "")),
        "amount_usd": amount,
        "direction": "spot_to_perp",
        "one_use": True,
        "maximum_transfer_attempts": 1,
        "transfer_required": transfer_required,
        "margin_snapshot": _safe_json_value(margin),
        "reservation_path": _relative(reservation_path, root),
        "agent_key_accessed": False,
        "transfer_attempted": False,
        "order_submission_performed": False,
        "research_only": True,
        "testnet_collateral_transfer_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    preflight_id = "testnetcollateralpreflight_" + _payload_hash(core)[:20]
    immutable_path = (
        root
        / "data"
        / "testnet"
        / "collateral_transfer_preflights"
        / f"{preflight_id}.json"
    )
    immutable = {**core, "preflight_id": preflight_id}
    _write_immutable_json(immutable, immutable_path)
    active_payload = {
        **immutable,
        "immutable_path": _relative(immutable_path, root),
        "immutable_sha256": _path_sha256(immutable_path),
    }
    active_path = active / "testnet_collateral_transfer_preflight.json"
    _write_json_atomic(active_payload, active_path)
    return CommandResult(
        paths={"preflight": active_path, "immutable_preflight": immutable_path},
        summary=active_payload,
    )


def run_testnet_collateral_transfer(
    *,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    preflight_id: str = "",
    approval_id: str = "",
    amount_usd: float = 25.0,
    acknowledgement: str = "",
    execute: bool = False,
    now: datetime | None = None,
    keychain_reader: KeychainReader | None = None,
    exchange_factory: ExchangeFactory | None = None,
    margin_reader: MarginReader | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> CommandResult:
    """Execute at most one explicitly approved Testnet collateral transfer."""

    started_at = _as_utc(now)
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    active_preflight_path = active / "testnet_collateral_transfer_preflight.json"
    active_preflight = _read_json(active_preflight_path)
    immutable_path = _safe_repo_path(
        root,
        str(active_preflight.get("immutable_path", "")),
        expected_parent=root / "data" / "testnet" / "collateral_transfer_preflights",
    )
    immutable = _read_json(immutable_path) if immutable_path else {}
    blockers: list[str] = []
    amount = _bounded_amount(amount_usd)
    if amount is None:
        blockers.append("testnet_collateral_transfer_amount_outside_20_to_25_usd")
        amount = float(amount_usd) if _finite(amount_usd) is not None else 0.0
    if active_preflight.get("preflight_id") != preflight_id or not preflight_id:
        blockers.append("testnet_collateral_preflight_id_mismatch")
    if immutable.get("preflight_id") != preflight_id:
        blockers.append("testnet_collateral_immutable_preflight_mismatch")
    if (
        immutable_path is None
        or not immutable_path.is_file()
        or _path_sha256(immutable_path)
        != str(active_preflight.get("immutable_sha256", ""))
    ):
        blockers.append("testnet_collateral_immutable_preflight_hash_invalid")
    if immutable.get("status") != "READY_REQUIRES_EXPLICIT_EXECUTION":
        blockers.append("testnet_collateral_preflight_not_ready")
    if immutable.get("approval_id") != approval_id or not approval_id:
        blockers.append("testnet_collateral_approval_id_mismatch")
    if abs(float(_finite(immutable.get("amount_usd")) or 0.0) - amount) > 1e-9:
        blockers.append("testnet_collateral_amount_mismatch")
    blockers.extend(resolved.configuration_blockers())
    if not execute:
        blockers.append("testnet_collateral_execute_flag_not_set")
    if not _truthy(os.getenv(ENABLE_ENV, "")):
        blockers.append("testnet_collateral_environment_enable_missing")
    if acknowledgement != EXECUTION_ACKNOWLEDGEMENT:
        blockers.append("testnet_collateral_exact_acknowledgement_missing")
    evaluated_at = pd.to_datetime(
        immutable.get("evaluated_at_utc"), utc=True, errors="coerce"
    )
    if (
        pd.isna(evaluated_at)
        or started_at.timestamp() - evaluated_at.timestamp() < 0
        or started_at.timestamp() - evaluated_at.timestamp()
        > MAX_PREFLIGHT_AGE_SECONDS
    ):
        blockers.append("testnet_collateral_preflight_stale_or_future")
    blockers.extend(_source_binding_blockers(root, immutable))

    reservation_path = _safe_repo_path(
        root,
        str(immutable.get("reservation_path", "")),
        expected_parent=root / "data" / "testnet" / "collateral_transfer_reservations",
    )
    if reservation_path is None:
        blockers.append("testnet_collateral_reservation_path_invalid")
    elif reservation_path.exists():
        blockers.append("testnet_collateral_transfer_approval_already_reserved_or_consumed")
    blockers = _unique(blockers)
    if blockers:
        return _blocked_execution_result(
            root=root,
            started_at=started_at,
            preflight_id=preflight_id,
            approval_id=approval_id,
            amount_usd=amount,
            blockers=blockers,
        )

    reservation = {
        "schema_version": RESERVATION_SCHEMA,
        "reserved_at_utc": started_at.isoformat(),
        "status": "RESERVED_BEFORE_AGENT_KEY_ACCESS",
        "preflight_id": preflight_id,
        "approval_id": approval_id,
        "candidate_receipt_id": str(immutable.get("candidate_receipt_id", "")),
        "amount_usd": amount,
        "direction": "spot_to_perp",
        "one_use": True,
        "transfer_attempted": False,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    assert reservation_path is not None
    try:
        _write_json_exclusive(reservation, reservation_path)
    except FileExistsError:
        return _blocked_execution_result(
            root=root,
            started_at=started_at,
            preflight_id=preflight_id,
            approval_id=approval_id,
            amount_usd=amount,
            blockers=["testnet_collateral_concurrent_or_replayed_reservation"],
        )

    read_margin = margin_reader or _default_margin_reader
    before = dict(immutable.get("margin_snapshot") or {})
    agent_key_accessed = False
    transfer_attempted = False
    response: object = {}
    error_class = ""
    after: dict[str, object] = {}
    reconciled = False
    try:
        reader = keychain_reader or read_hyperliquid_agent_key_from_keychain
        secret = reader(
            str(resolved.keychain_service or ""),
            str(resolved.agent_address or ""),
        )
        agent_key_accessed = True
        wallet = Account.from_key(secret)
        if wallet.address.lower() != str(resolved.agent_address or "").lower():
            raise ValueError("hyperliquid_agent_key_address_mismatch")
        factory = exchange_factory or _default_exchange_factory
        exchange = factory(wallet, resolved)
        transfer_attempted = True
        response = exchange.usd_class_transfer(amount, True)
    except Exception as exc:  # noqa: BLE001 - receipt must survive SDK uncertainty.
        error_class = type(exc).__name__

    if transfer_attempted:
        for index in range(3):
            try:
                after = _safe_json_value(read_margin(resolved))
            except Exception as exc:  # noqa: BLE001 - bounded reconciliation records failure.
                after = {
                    "status": "BLOCKED",
                    "blockers": f"margin_reconciliation_failed:{type(exc).__name__}",
                }
            if _transfer_reconciled(before, after, amount):
                reconciled = True
                break
            if index < 2:
                sleeper(0.5)

    response_ok = _transfer_response_ok(response)
    if reconciled and response_ok:
        status = "PASS_TRANSFER_RECONCILED"
    elif reconciled:
        status = "PASS_RECONCILED_AFTER_UNKNOWN_RESPONSE"
    elif transfer_attempted:
        status = "TRANSFER_UNCONFIRMED_NO_AUTOMATIC_RETRY"
    else:
        status = "TRANSFER_SETUP_FAILED_APPROVAL_CONSUMED"
    completed_at = datetime.now(UTC)
    execution_core = {
        "schema_version": EXECUTION_SCHEMA,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "status": status,
        "blockers": [] if reconciled else [error_class or status],
        "preflight_id": preflight_id,
        "approval_id": approval_id,
        "candidate_receipt_id": str(immutable.get("candidate_receipt_id", "")),
        "amount_usd": amount,
        "direction": "spot_to_perp",
        "agent_key_accessed": agent_key_accessed,
        "transfer_attempted": transfer_attempted,
        "transfer_response_ok": response_ok,
        "response": _safe_json_value(response),
        "error_class": error_class,
        "before_margin_snapshot": before,
        "after_margin_snapshot": after,
        "transfer_reconciled": reconciled,
        "automatic_retry_performed": False,
        "reservation_path": _relative(reservation_path, root),
        "order_submission_performed": False,
        "research_only": True,
        "testnet_collateral_transfer_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    execution_id = "testnetcollateralexecution_" + _payload_hash(execution_core)[:20]
    immutable_execution = (
        root
        / "data"
        / "testnet"
        / "collateral_transfer_executions"
        / f"{execution_id}.json"
    )
    execution = {**execution_core, "execution_id": execution_id}
    _write_immutable_json(execution, immutable_execution)
    active_execution = {
        **execution,
        "immutable_path": _relative(immutable_execution, root),
        "immutable_sha256": _path_sha256(immutable_execution),
    }
    active_execution_path = active / "testnet_collateral_transfer_execution.json"
    _write_json_atomic(active_execution, active_execution_path)
    reservation.update(
        {
            "status": "CONSUMED_NO_RETRY",
            "completed_at_utc": completed_at.isoformat(),
            "transfer_attempted": transfer_attempted,
            "execution_id": execution_id,
            "execution_path": _relative(immutable_execution, root),
            "execution_sha256": _path_sha256(immutable_execution),
        }
    )
    _write_json_atomic(reservation, reservation_path)
    return CommandResult(
        paths={
            "execution": active_execution_path,
            "immutable_execution": immutable_execution,
            "reservation": reservation_path,
        },
        summary=active_execution,
    )


def _default_candidate_validator(
    root: Path,
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    return _validated_testnet_candidate_receipt(root=root, candidate=candidate)


def _default_approval_validator(
    approval_id: str,
    root: Path,
    config: HyperliquidTestnetConfig,
) -> dict[str, object]:
    return validate_testnet_smoke_approval(
        approval_id=approval_id,
        root=root,
        config=config,
    )


def _default_margin_reader(config: HyperliquidTestnetConfig) -> dict[str, object]:
    return hyperliquid_testnet_margin_snapshot(config)


def _default_exchange_factory(wallet: Any, config: HyperliquidTestnetConfig) -> Any:
    from hyperliquid.exchange import Exchange

    return Exchange(
        wallet,
        base_url=config.base_url,
        account_address=config.master_address,
        timeout=20.0,
    )


def _collateral_policy_blockers(
    approval: dict[str, Any],
    amount_usd: float,
) -> list[str]:
    blockers: list[str] = []
    policy = approval.get("collateral_transfer_policy")
    if approval.get("approval_version") != TESTNET_APPROVAL_VERSION:
        blockers.append("testnet_collateral_approval_version_invalid")
    if not isinstance(policy, dict):
        return [*blockers, "testnet_collateral_policy_missing"]
    policy_amount = _finite(policy.get("amount_usd"))
    checks = (
        (policy.get("approved") is True, "testnet_collateral_policy_not_explicitly_approved"),
        (policy.get("direction") == "spot_to_perp", "testnet_collateral_direction_invalid"),
        (policy.get("one_use") is True, "testnet_collateral_policy_not_one_use"),
        (
            int(_finite(policy.get("maximum_transfer_attempts")) or 0) == 1,
            "testnet_collateral_transfer_attempt_limit_invalid",
        ),
        (
            policy_amount is not None and abs(policy_amount - amount_usd) <= 1e-9,
            "testnet_collateral_policy_amount_mismatch",
        ),
    )
    blockers.extend(blocker for passed, blocker in checks if not passed)
    return blockers


def _source_binding_blockers(root: Path, preflight: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    for prefix in ("candidate_receipt", "approval"):
        path = _safe_repo_path(root, str(preflight.get(f"{prefix}_path", "")))
        expected = str(preflight.get(f"{prefix}_sha256", ""))
        if path is None or not path.is_file() or _path_sha256(path) != expected:
            blockers.append(f"testnet_collateral_{prefix}_binding_changed")
    return blockers


def _transfer_reconciled(
    before: dict[str, Any],
    after: dict[str, Any],
    amount_usd: float,
) -> bool:
    before_perp = _finite(before.get("account_value_usd"))
    after_perp = _finite(after.get("account_value_usd"))
    before_spot = _finite(before.get("spot_usdc_usd"))
    after_spot = _finite(after.get("spot_usdc_usd"))
    return bool(
        before_perp is not None
        and after_perp is not None
        and before_spot is not None
        and after_spot is not None
        and after_perp
        >= before_perp + amount_usd - RECONCILIATION_TOLERANCE_USD
        and after_spot
        <= before_spot - amount_usd + RECONCILIATION_TOLERANCE_USD
    )


def _transfer_response_ok(response: object) -> bool:
    return bool(
        isinstance(response, dict)
        and response.get("status") == "ok"
        and isinstance(response.get("response"), dict)
        and response["response"].get("type") == "default"
    )


def _blocked_execution_result(
    *,
    root: Path,
    started_at: datetime,
    preflight_id: str,
    approval_id: str,
    amount_usd: float,
    blockers: list[str],
) -> CommandResult:
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    path = active / "testnet_collateral_transfer_execution.json"
    payload = {
        "schema_version": EXECUTION_SCHEMA,
        "started_at_utc": started_at.isoformat(),
        "status": "BLOCKED_NO_TRANSFER",
        "blockers": _unique(blockers),
        "preflight_id": preflight_id,
        "approval_id": approval_id,
        "amount_usd": amount_usd,
        "agent_key_accessed": False,
        "transfer_attempted": False,
        "order_submission_performed": False,
        "research_only": True,
        "testnet_collateral_transfer_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json_atomic(payload, path)
    return CommandResult(paths={"execution": path}, summary=payload)


def _reservation_path(
    root: Path,
    *,
    approval_id: str,
    candidate_receipt_id: str,
    amount_usd: float,
) -> Path:
    key = _payload_hash(
        {
            "approval_id": approval_id,
            "candidate_receipt_id": candidate_receipt_id,
            "amount_usd": amount_usd,
            "direction": "spot_to_perp",
        }
    )[:24]
    return (
        root
        / "data"
        / "testnet"
        / "collateral_transfer_reservations"
        / f"{key}.json"
    )


def _safe_repo_path(
    root: Path,
    value: str,
    *,
    expected_parent: Path | None = None,
) -> Path | None:
    if not value:
        return None
    candidate = (root / value).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        return None
    if expected_parent is not None:
        parent = expected_parent.resolve()
        if candidate.parent != parent:
            return None
    return candidate


def _bounded_amount(value: object) -> float | None:
    amount = _finite(value)
    if amount is None or not MINIMUM_TRANSFER_USD <= amount <= MAXIMUM_TRANSFER_USD:
        return None
    return amount


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(number):
        return None
    return number


def _as_utc(value: datetime | None) -> datetime:
    current = value or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC)


def _safe_json_value(value: object) -> Any:
    return json.loads(json.dumps(value, default=str))


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_exclusive(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != payload:
            raise ValueError("immutable_testnet_collateral_artifact_collision")
        return
    _write_json_exclusive(payload, path)


def _path_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
