"""Status-first operator for one governed Hyperliquid Testnet pair action."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.active_pipeline import CommandResult
from quant_platform.execution import OrderIntent
from quant_platform.hyperliquid_testnet import (
    HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON,
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
    _normalize_perp_coin,
    _price_precision_valid,
    _size_precision_valid,
)
from quant_platform.orchestration.corrective_hyperliquid_network import (
    run_authorized_hyperliquid_info_call,
)
from quant_platform.orchestration.corrective_release_gates import (
    _validated_testnet_candidate_receipt,
)
from quant_platform.orchestration.corrective_runtime import (
    create_exclusive_json,
    promote_staged_file,
    write_immutable_json,
)
from quant_platform.orchestration.hyperliquid_learning_and_risk import (
    TESTNET_CANDIDATE_BINDING_FIELDS,
    validate_testnet_smoke_approval,
)

ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_SCHEMA = "thewiz.hyperliquid_testnet_pair_preflight.v1"
EXECUTION_SCHEMA = "thewiz.hyperliquid_testnet_pair_execution.v1"
RESERVATION_SCHEMA = "thewiz.hyperliquid_testnet_pair_reservation.v1"
ENABLE_ENV = "QPA_ENABLE_HYPERLIQUID_TESTNET_PAIR_EXECUTION"
ENTRY_ACKNOWLEDGEMENT = "EXECUTE ONE HYPERLIQUID TESTNET PAIR ENTRY"
EXIT_ACKNOWLEDGEMENT = "EXECUTE ONE HYPERLIQUID TESTNET PAIR EXIT"
MAX_PREFLIGHT_AGE_SECONDS = 300

ApprovalValidator = Callable[
    [str, Path, HyperliquidTestnetConfig, str, datetime], dict[str, object]
]
CandidateValidator = Callable[[Path, dict[str, Any]], tuple[bool, list[str]]]
InfoClient = Callable[[dict[str, Any]], Any]


def build_testnet_pair_execution_preflight(
    *,
    action: str,
    approval_id: str,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    now: datetime | None = None,
    approval_validator: ApprovalValidator | None = None,
    candidate_validator: CandidateValidator | None = None,
    info_client: InfoClient | None = None,
) -> CommandResult:
    """Seal exact entry or reduce-only exit intents without touching the agent key."""

    as_of = _as_utc(now)
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    approval_path = active / "hyperliquid_testnet_smoke_approval.json"
    candidate_path = active / "testnet_candidate_receipt.json"
    state_path = active / HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON.name
    approval = _read_json(approval_path)
    candidate = _read_json(candidate_path)
    state = _read_state(state_path)
    normalized_action = str(action).strip().lower()
    blockers = list(resolved.configuration_blockers())
    if normalized_action not in {"entry", "exit"}:
        blockers.append("testnet_pair_action_invalid")

    validate_approval = approval_validator or _default_approval_validator
    approval_result = validate_approval(str(approval_id), root, resolved, normalized_action, as_of)
    if not (
        approval_result.get("status") == "PASS"
        and approval_result.get("execution_allowed") is True
        and approval_result.get("testnet_order_authority") is True
        and approval_result.get("authority_scope") == normalized_action
        and approval_result.get("live_trading_authorized") is False
    ):
        supplied = approval_result.get("blockers")
        blockers.extend(
            [str(value) for value in supplied if str(value)]
            if isinstance(supplied, list)
            else ["testnet_pair_signed_approval_invalid"]
        )
    if str(approval.get("approval_id", "")) != str(approval_id) or not approval_id:
        blockers.append("testnet_pair_approval_id_mismatch")

    if normalized_action == "entry":
        validate_candidate = candidate_validator or _default_candidate_validator
        candidate_ready, candidate_blockers = validate_candidate(root, candidate)
        if not candidate_ready:
            blockers.extend(candidate_blockers or ["testnet_pair_candidate_invalid"])
    blockers.extend(
        _state_action_blockers(
            state=state,
            action=normalized_action,
            approval_id=str(approval_id),
        )
    )

    intents: list[dict[str, object]] = []
    account_snapshot: dict[str, object] = {}
    if not blockers:
        raw_fetch = info_client or _raw_hyperliquid_testnet_info_client(
            resolved.base_url
        )
        fetch = _authorized_hyperliquid_testnet_info_client(
            base_url=resolved.base_url,
            raw_fetch=raw_fetch,
        )
        try:
            meta = fetch({"type": "meta"})
            account = fetch({"type": "clearinghouseState", "user": resolved.master_address})
            open_orders = fetch({"type": "openOrders", "user": resolved.master_address})
            mids = fetch({"type": "allMids"}) if normalized_action == "exit" else {}
            intents, account_snapshot, runtime_blockers = _derive_intents(
                action=normalized_action,
                approval=approval,
                state=state,
                meta=meta,
                account=account,
                open_orders=open_orders,
                mids=mids,
                config=resolved,
            )
            blockers.extend(runtime_blockers)
        except Exception as exc:  # noqa: BLE001 - boundary fails closed.
            blockers.append(f"testnet_pair_read_only_preflight_failed:{type(exc).__name__}")

    blockers = _unique(blockers)
    source_paths = [approval_path]
    if normalized_action == "entry":
        source_paths.append(candidate_path)
    if normalized_action == "exit" or state_path.is_file():
        source_paths.append(state_path)
    source_hashes = {
        _relative(path, root): _file_hash(path) for path in source_paths if path.is_file()
    }
    status = "READY_REQUIRES_EXPLICIT_EXECUTION" if not blockers else "BLOCKED"
    core = {
        "schema_version": PREFLIGHT_SCHEMA,
        "evaluated_at_utc": as_of.isoformat(),
        "status": status,
        "blockers": blockers,
        "action": normalized_action,
        "approval_id": str(approval_id),
        "network": resolved.network,
        "base_url": resolved.base_url,
        "master_address": str(resolved.master_address or ""),
        "agent_address": str(resolved.agent_address or ""),
        "candidate_receipt_id": str(approval.get("candidate_receipt_id", "")),
        "pair": str(approval.get("pair", "")),
        "entry_context": approval.get("entry_context", {}),
        **{
            field: str(approval.get(field, "")).strip()
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        },
        "intents": intents,
        "account_snapshot": account_snapshot,
        "source_artifact_hashes": source_hashes,
        "agent_key_accessed": False,
        "order_submission_performed": False,
        "research_only": True,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    preflight_id = "testnetpairpreflight_" + _payload_hash(core)[:20]
    immutable = {**core, "preflight_id": preflight_id}
    immutable_path = root / "data" / "testnet" / "pair_preflights" / f"{preflight_id}.json"
    _write_immutable_json(immutable, immutable_path)
    active_payload = {
        **immutable,
        "immutable_path": _relative(immutable_path, root),
        "immutable_sha256": _file_hash(immutable_path),
    }
    active_path = active / "hyperliquid_testnet_pair_preflight.json"
    _write_json_atomic(active_payload, active_path)
    return CommandResult(
        paths={"preflight": active_path, "immutable_preflight": immutable_path},
        summary=active_payload,
    )


def run_testnet_pair_execution(
    *,
    action: str,
    preflight_id: str,
    approval_id: str,
    acknowledgement: str = "",
    execute: bool = False,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    now: datetime | None = None,
    pair_executor: HyperliquidTestnetPairExecutor | None = None,
) -> CommandResult:
    """Execute one exact preflight; default invocation is status-only."""

    as_of = _as_utc(now)
    normalized_action = str(action).strip().lower()
    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    active = root / "reports" / "active"
    active_path = active / "hyperliquid_testnet_pair_preflight.json"
    pointer = _read_json(active_path)
    immutable_path = _safe_root_path(root, str(pointer.get("immutable_path", "")))
    immutable = _read_json(immutable_path) if immutable_path else {}
    blockers: list[str] = []
    if normalized_action not in {"entry", "exit"}:
        blockers.append("testnet_pair_action_invalid")
    if pointer.get("preflight_id") != preflight_id or not preflight_id:
        blockers.append("testnet_pair_preflight_id_mismatch")
    if immutable.get("preflight_id") != preflight_id:
        blockers.append("testnet_pair_immutable_preflight_mismatch")
    if not (
        immutable_path
        and immutable_path.is_file()
        and _file_hash(immutable_path) == pointer.get("immutable_sha256")
        and immutable.get("status") == "READY_REQUIRES_EXPLICIT_EXECUTION"
        and immutable.get("blockers") == []
    ):
        blockers.append("testnet_pair_preflight_not_ready_or_invalid")
    if immutable.get("action") != normalized_action:
        blockers.append("testnet_pair_action_preflight_mismatch")
    if immutable.get("approval_id") != approval_id or not approval_id:
        blockers.append("testnet_pair_approval_preflight_mismatch")
    evaluated_at = pd.to_datetime(immutable.get("evaluated_at_utc"), utc=True, errors="coerce")
    age = as_of.timestamp() - evaluated_at.timestamp() if pd.notna(evaluated_at) else math.inf
    if age < 0.0 or age > MAX_PREFLIGHT_AGE_SECONDS:
        blockers.append("testnet_pair_preflight_stale_or_future")
    blockers.extend(_source_binding_blockers(root, immutable))
    blockers.extend(resolved.configuration_blockers())
    if (
        execute
        and pair_executor is not None
        and type(pair_executor) is not HyperliquidTestnetPairExecutor
    ):
        blockers.append("gate00g_testnet_pair_executor_type_invalid")
    if not execute:
        blockers.append("testnet_pair_execute_flag_not_set")
    if os.getenv(ENABLE_ENV, "").strip().lower() != "true":
        blockers.append("testnet_pair_environment_enable_missing")
    expected_ack = ENTRY_ACKNOWLEDGEMENT if normalized_action == "entry" else EXIT_ACKNOWLEDGEMENT
    if acknowledgement != expected_ack:
        blockers.append("testnet_pair_exact_acknowledgement_missing")

    reservation_path = root / "data" / "testnet" / "pair_reservations" / f"{preflight_id}.json"
    if reservation_path.exists():
        blockers.append("testnet_pair_preflight_already_reserved_or_consumed")
    blockers = _unique(blockers)
    if blockers:
        return _execution_result(
            root=root,
            as_of=as_of,
            action=normalized_action,
            preflight_id=preflight_id,
            approval_id=approval_id,
            status="NO_SUBMISSION_STATUS_ONLY" if not execute else "BLOCKED_BEFORE_KEY_ACCESS",
            blockers=blockers,
        )

    reservation = {
        "schema_version": RESERVATION_SCHEMA,
        "reserved_at_utc": as_of.isoformat(),
        "preflight_id": preflight_id,
        "approval_id": approval_id,
        "action": normalized_action,
        "status": "RESERVED_BEFORE_AGENT_KEY_ACCESS",
        "one_use": True,
        "order_submission_performed": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    try:
        _write_json_exclusive(reservation, reservation_path)
    except FileExistsError:
        return _execution_result(
            root=root,
            as_of=as_of,
            action=normalized_action,
            preflight_id=preflight_id,
            approval_id=approval_id,
            status="BLOCKED_BEFORE_KEY_ACCESS",
            blockers=["testnet_pair_concurrent_or_replayed_reservation"],
        )

    intents = _intents_from_rows(immutable.get("intents"))
    if len(intents) != 2:
        return _execution_result(
            root=root,
            as_of=as_of,
            action=normalized_action,
            preflight_id=preflight_id,
            approval_id=approval_id,
            status="BLOCKED_AFTER_RESERVATION_BEFORE_KEY_ACCESS",
            blockers=["testnet_pair_preflight_intents_invalid"],
        )
    execution_config = replace(resolved, order_approval_id=approval_id)
    executor = _require_gate00g_testnet_pair_executor(
        pair_executor
        or HyperliquidTestnetPairExecutor(
            state_path=root
            / "reports"
            / "active"
            / HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON.name
        )
    )
    try:
        result = executor.submit_pair(intents, execution_config)
    except Exception as exc:  # noqa: BLE001 - execution boundary must leave evidence.
        reservation.update(
            {
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "status": "CONSUMED",
                "executor_status": "testnet_pair_executor_exception",
                "executor_reason": type(exc).__name__,
                "order_submission_performed": False,
            }
        )
        _write_json_atomic(reservation, reservation_path)
        return _execution_result(
            root=root,
            as_of=as_of,
            action=normalized_action,
            preflight_id=preflight_id,
            approval_id=approval_id,
            status="testnet_pair_executor_exception",
            blockers=[f"testnet_pair_executor_exception:{type(exc).__name__}"],
            result={
                "status": "testnet_pair_executor_exception",
                "reason": type(exc).__name__,
                "order_submission_performed": False,
                "reconciled": False,
                "live_trading_authorized": False,
            },
            execution_invoked=True,
            preflight=immutable,
            immutable_preflight_path=immutable_path,
        )
    reservation.update(
        {
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "status": "CONSUMED",
            "executor_status": result.status,
            "executor_reason": result.reason,
            "order_submission_performed": result.order_submission_performed,
            "reconciled": result.reconciled,
        }
    )
    _write_json_atomic(reservation, reservation_path)
    return _execution_result(
        root=root,
        as_of=as_of,
        action=normalized_action,
        preflight_id=preflight_id,
        approval_id=approval_id,
        status=result.status,
        blockers=[] if result.order_submission_performed else [result.reason],
        result=asdict(result),
        execution_invoked=True,
        preflight=immutable,
        immutable_preflight_path=immutable_path,
    )


def _derive_intents(
    *,
    action: str,
    approval: dict[str, Any],
    state: dict[str, Any],
    meta: Any,
    account: Any,
    open_orders: Any,
    mids: Any,
    config: HyperliquidTestnetConfig,
) -> tuple[list[dict[str, object]], dict[str, object], list[str]]:
    blockers: list[str] = []
    legs = approval.get("legs") if isinstance(approval.get("legs"), list) else []
    if len(legs) != 2 or not all(isinstance(leg, dict) for leg in legs):
        return [], {}, ["testnet_pair_approval_legs_invalid"]
    markets = [_normalize_perp_coin(str(leg.get("market", ""))) for leg in legs]
    rules = {
        _normalize_perp_coin(str(row.get("name", ""))): row
        for row in (meta.get("universe", []) if isinstance(meta, dict) else [])
        if isinstance(row, dict) and str(row.get("name", "")).strip()
    }
    positions = _position_sizes(account, set(markets))
    pair_orders = _pair_open_orders(open_orders, set(markets))
    if pair_orders:
        blockers.append("testnet_pair_open_orders_present")
    rows: list[dict[str, object]] = []
    if action == "entry":
        if any(abs(positions.get(market, 0.0)) > 1e-12 for market in markets):
            blockers.append("testnet_pair_markets_not_flat_before_entry")
        for leg in legs:
            rows.append(
                {
                    "market": _normalize_perp_coin(str(leg.get("market", ""))),
                    "side": str(leg.get("side", "")).strip().upper(),
                    "size": float(leg.get("size", 0.0)),
                    "limit_price": float(leg.get("limit_price", 0.0)),
                    "reduce_only": False,
                }
            )
    else:
        slippage_bps = _number(approval.get("maximum_exit_slippage_bps"))
        if slippage_bps is None or not 0.0 < slippage_bps <= 50.0:
            blockers.append("testnet_pair_exit_slippage_policy_invalid")
            slippage_bps = 0.0
        if not isinstance(mids, dict):
            blockers.append("testnet_pair_all_mids_invalid")
            mids = {}
        for leg in legs:
            market = _normalize_perp_coin(str(leg.get("market", "")))
            approved_side = str(leg.get("side", "")).strip().upper()
            approved_size = _number(leg.get("size")) or 0.0
            position = positions.get(market, 0.0)
            expected_position_sign = 1.0 if approved_side == "BUY" else -1.0
            if abs(position) <= 1e-12:
                blockers.append(f"testnet_pair_exit_position_missing:{market}")
                continue
            if position * expected_position_sign <= 0.0:
                blockers.append(f"testnet_pair_exit_position_direction_mismatch:{market}")
            if abs(position) > approved_size + 1e-12:
                blockers.append(f"testnet_pair_exit_position_exceeds_approved_size:{market}")
            mid = _number(mids.get(market))
            rule = rules.get(market, {})
            sz_decimals = _integer(rule.get("szDecimals"))
            if mid is None or mid <= 0.0:
                blockers.append(f"testnet_pair_exit_mid_missing:{market}")
                continue
            if sz_decimals is None:
                blockers.append(f"testnet_pair_market_precision_missing:{market}")
                continue
            side = "SELL" if position > 0.0 else "BUY"
            raw_price = mid * (
                1.0 - slippage_bps / 10_000.0 if side == "SELL" else 1.0 + slippage_bps / 10_000.0
            )
            price = _conservative_price(raw_price, sz_decimals, side)
            rows.append(
                {
                    "market": market,
                    "side": side,
                    "size": abs(position),
                    "limit_price": price,
                    "reduce_only": True,
                }
            )

    for row in rows:
        market = str(row["market"])
        rule = rules.get(market)
        if not isinstance(rule, dict):
            blockers.append(f"testnet_pair_market_missing:{market}")
            continue
        if str(rule.get("isDelisted", False)).strip().lower() in {"true", "1", "yes"}:
            blockers.append(f"testnet_pair_market_delisted:{market}")
        sz_decimals = _integer(rule.get("szDecimals"))
        if sz_decimals is None:
            blockers.append(f"testnet_pair_market_precision_missing:{market}")
            continue
        if not _size_precision_valid(row["size"], sz_decimals):
            blockers.append(f"testnet_pair_size_precision_invalid:{market}")
        if not _price_precision_valid(row["limit_price"], sz_decimals):
            blockers.append(f"testnet_pair_price_precision_invalid:{market}")
    snapshot = {
        "positions": positions,
        "pair_open_order_count": len(pair_orders),
        "markets": markets,
    }
    return rows, snapshot, _unique(blockers)


def _state_action_blockers(*, state: dict[str, Any], action: str, approval_id: str) -> list[str]:
    if action == "entry":
        if not state:
            return []
        if state.get("state_integrity_valid") is not True:
            return ["testnet_pair_execution_state_hash_invalid"]
        if (
            state.get("order_approval_id") == approval_id
            and state.get("entry_submit_attempted") is True
        ):
            return ["testnet_pair_entry_approval_already_consumed"]
        if str(state.get("phase", "")) not in {
            "FLAT_RECONCILED",
            "PRE_SUBMISSION_FAILED",
            "REJECTED",
        }:
            return ["testnet_pair_previous_lifecycle_not_terminal"]
        return []
    if not state:
        return ["testnet_pair_exit_entry_state_missing"]
    blockers: list[str] = []
    if state.get("state_integrity_valid") is not True:
        blockers.append("testnet_pair_execution_state_hash_invalid")
    if state.get("order_approval_id") != approval_id:
        blockers.append("testnet_pair_exit_approval_state_mismatch")
    if state.get("entry_submit_attempted") is not True:
        blockers.append("testnet_pair_exit_entry_not_recorded")
    if state.get("exit_submit_attempted") is True:
        blockers.append("testnet_pair_exit_already_attempted")
    if str(state.get("phase", "")) != "AWAITING_EXCHANGE_CONFIRMATION":
        blockers.append("testnet_pair_exit_lifecycle_not_open")
    return blockers


def _execution_result(
    *,
    root: Path,
    as_of: datetime,
    action: str,
    preflight_id: str,
    approval_id: str,
    status: str,
    blockers: list[str],
    result: dict[str, Any] | None = None,
    execution_invoked: bool = False,
    preflight: dict[str, Any] | None = None,
    immutable_preflight_path: Path | None = None,
) -> CommandResult:
    bound_preflight = preflight or {}
    payload = {
        "schema_version": EXECUTION_SCHEMA,
        "generated_at_utc": as_of.isoformat(),
        "status": status,
        "blockers": _unique(blockers),
        "action": action,
        "preflight_id": preflight_id,
        "approval_id": approval_id,
        "executor_result": result or {},
        "execution_invoked": execution_invoked,
        "immutable_preflight_path": (
            _relative(immutable_preflight_path, root)
            if immutable_preflight_path is not None
            else ""
        ),
        "immutable_preflight_sha256": (
            _file_hash(immutable_preflight_path) if immutable_preflight_path is not None else ""
        ),
        "entry_context": bound_preflight.get("entry_context", {}),
        **{
            field: str(bound_preflight.get(field, "")).strip()
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        },
        "order_submission_performed": bool((result or {}).get("order_submission_performed", False)),
        "research_only": True,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    execution_id = "testnetpairexecution_" + _payload_hash(payload)[:20]
    payload["execution_id"] = execution_id
    immutable_path = root / "data" / "testnet" / "pair_executions" / f"{execution_id}.json"
    _write_immutable_json(payload, immutable_path)
    active_path = root / "reports" / "active" / "hyperliquid_testnet_pair_execution.json"
    _write_json_atomic(
        {
            **payload,
            "immutable_path": _relative(immutable_path, root),
            "immutable_sha256": _file_hash(immutable_path),
        },
        active_path,
    )
    return CommandResult(
        paths={"execution": active_path, "immutable_execution": immutable_path},
        summary=payload,
    )


def _default_approval_validator(
    approval_id: str,
    root: Path,
    config: HyperliquidTestnetConfig,
    scope: str,
    now: datetime,
) -> dict[str, object]:
    return validate_testnet_smoke_approval(
        approval_id=approval_id,
        root=root,
        config=config,
        validation_scope=scope,
        now=now,
    )


def _default_candidate_validator(root: Path, candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    return _validated_testnet_candidate_receipt(root=root, candidate=candidate)


def _authorized_hyperliquid_testnet_info_client(
    *,
    base_url: str,
    raw_fetch: InfoClient,
) -> InfoClient:
    target = f"{base_url.rstrip('/')}/info"

    def fetch(payload: dict[str, Any]) -> Any:
        return run_authorized_hyperliquid_info_call(
            target=target,
            payload=payload,
            operation_prefix="HYPERLIQUID_TESTNET",
            transport=lambda: raw_fetch(payload),
        )

    return fetch


def _raw_hyperliquid_testnet_info_client(base_url: str) -> InfoClient:
    session = requests.Session()

    def fetch(payload: dict[str, Any]) -> Any:
        response = session.post(f"{base_url.rstrip('/')}/info", json=payload, timeout=20)
        response.raise_for_status()
        return response.json()

    return fetch


def _position_sizes(account: Any, markets: set[str]) -> dict[str, float]:
    result = {market: 0.0 for market in markets}
    rows = account.get("assetPositions", []) if isinstance(account, dict) else []
    if not isinstance(rows, list):
        raise TypeError("testnet_pair_account_positions_invalid")
    for row in rows:
        if not isinstance(row, dict):
            continue
        position = row.get("position") if isinstance(row.get("position"), dict) else row
        market = _normalize_perp_coin(str(position.get("coin", "")))
        if market in result:
            result[market] = float(position.get("szi", 0.0))
    return result


def _pair_open_orders(open_orders: Any, markets: set[str]) -> list[dict[str, Any]]:
    if not isinstance(open_orders, list):
        raise TypeError("testnet_pair_open_orders_invalid")
    return [
        row
        for row in open_orders
        if isinstance(row, dict) and _normalize_perp_coin(str(row.get("coin", ""))) in markets
    ]


def _conservative_price(raw_price: float, sz_decimals: int, side: str) -> float:
    if not math.isfinite(raw_price) or raw_price <= 0.0:
        raise ValueError("testnet_pair_exit_price_invalid")
    integer_digits = max(math.floor(math.log10(raw_price)) + 1, 1)
    decimals = max(min(6 - sz_decimals, 5 - integer_digits), 0)
    scale = 10**decimals
    scaled = raw_price * scale
    rounded = math.ceil(scaled) if side == "SELL" else math.floor(scaled)
    return rounded / scale


def _intents_from_rows(value: object) -> tuple[OrderIntent, ...]:
    rows = value if isinstance(value, list) else []
    intents: list[OrderIntent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            intents.append(
                OrderIntent(
                    market=str(row["market"]),
                    side=str(row["side"]),
                    size=float(row["size"]),
                    limit_price=float(row["limit_price"]),
                    reduce_only=bool(row["reduce_only"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(intents)


def _read_state(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if not payload:
        return {}
    supplied = str(payload.pop("state_hash", ""))
    expected = sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    payload["state_hash"] = supplied
    payload["state_integrity_valid"] = bool(supplied and supplied == expected)
    return payload


def _source_binding_blockers(root: Path, preflight: dict[str, Any]) -> list[str]:
    hashes = preflight.get("source_artifact_hashes")
    if not isinstance(hashes, dict) or not hashes:
        return ["testnet_pair_preflight_source_hashes_missing"]
    blockers: list[str] = []
    for relative, expected in hashes.items():
        path = _safe_root_path(root, str(relative))
        if path is None or not path.is_file() or _file_hash(path) != expected:
            blockers.append(f"testnet_pair_preflight_source_changed:{relative}")
    return blockers


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _require_gate00g_testnet_pair_executor(
    executor: HyperliquidTestnetPairExecutor,
) -> HyperliquidTestnetPairExecutor:
    if type(executor) is not HyperliquidTestnetPairExecutor:
        raise ValueError("gate00g_testnet_pair_executor_type_invalid")
    return executor


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _write_json_exclusive(payload: dict[str, Any], path: Path) -> None:
    create_exclusive_json(path, payload)


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    try:
        write_immutable_json(path, payload)
    except ValueError as exc:
        raise ValueError("testnet_pair_immutable_artifact_conflict") from exc


def _safe_root_path(root: Path, relative: str) -> Path | None:
    if not relative:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _payload_hash(payload: dict[str, Any]) -> str:
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _number(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: object) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() and number >= 0 else None


def _as_utc(value: datetime | None) -> datetime:
    resolved = value or datetime.now(UTC)
    return resolved.replace(tzinfo=UTC) if resolved.tzinfo is None else resolved.astimezone(UTC)


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))
