"""Read-only Hyperliquid Testnet lifecycle evidence capture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from quant_platform.hyperliquid_testnet import (
    HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON,
    HyperliquidTestnetConfig,
    HyperliquidTestnetPairExecutor,
    _normalize_perp_coin,
    _pair_open_orders,
    _pair_position_sizes,
)

ROOT = Path(__file__).resolve().parents[3]


def capture_hyperliquid_testnet_lifecycle_evidence(
    *,
    root: Path = ROOT,
    config: HyperliquidTestnetConfig | None = None,
    session: requests.Session | None = None,
    now: datetime | None = None,
) -> dict[str, object]:
    """Capture fills and account state without loading a key or submitting an action."""

    resolved = config or HyperliquidTestnetConfig.paper_testnet_from_env()
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    active = root / "reports" / "active"
    active.mkdir(parents=True, exist_ok=True)
    state_path = active / HYPERLIQUID_TESTNET_EXECUTION_STATE_JSON.name
    receipt_path = active / "hyperliquid_testnet_smoke_receipt.json"
    capture_path = active / "hyperliquid_testnet_lifecycle_evidence_capture.csv"

    state = HyperliquidTestnetPairExecutor(state_path=state_path)._read_execution_state()
    approval = _read_json(active / "hyperliquid_testnet_smoke_approval.json")
    run_manifest = _read_json(active / "hyperliquid_run_manifest.json")
    protocol = _read_json(active / "current_wizard_hyperliquid_testnet_protocol_manifest.json")
    blockers = list(resolved.configuration_blockers())
    if not state:
        blockers.append("hyperliquid_pair_execution_state_missing")
    elif state.get("state_integrity_valid") is not True:
        blockers.append("hyperliquid_pair_execution_state_hash_invalid")
    if state and state.get("network") != "testnet":
        blockers.append("hyperliquid_execution_state_not_testnet")
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        TESTNET_APPROVAL_VERSION,
        TESTNET_CANDIDATE_BINDING_FIELDS,
    )

    if not approval or approval.get("approval_version") != TESTNET_APPROVAL_VERSION:
        blockers.append("hyperliquid_testnet_candidate_bound_smoke_approval_missing")
    if state and approval.get("approval_id") != state.get("order_approval_id"):
        blockers.append("hyperliquid_lifecycle_approval_state_mismatch")
    if state and (
        state.get("master_address") != resolved.master_address
        or approval.get("master_address") != resolved.master_address
    ):
        blockers.append("hyperliquid_lifecycle_master_address_mismatch")
    if state and (
        state.get("agent_address") != resolved.agent_address
        or approval.get("agent_address") != resolved.agent_address
    ):
        blockers.append("hyperliquid_lifecycle_agent_address_mismatch")
    if not run_manifest.get("run_id") or not run_manifest.get("candidate_set_id"):
        blockers.append("hyperliquid_run_manifest_identity_missing")
    if (
        protocol.get("protocol_status") != "PASS"
        or protocol.get("simulation_is_testnet_proof") is not False
        or protocol.get("order_submission_performed") is not False
    ):
        blockers.append("hyperliquid_deterministic_protocol_not_ready")

    intents = state.get("intents") if isinstance(state.get("intents"), list) else []
    references = _ordered_exchange_references(state=state, intents=intents)
    if len(intents) != 2:
        blockers.append("hyperliquid_execution_state_intents_invalid")
    submit_kind = str(state.get("submit_kind", ""))
    if submit_kind not in {"entry", "exit"}:
        blockers.append("hyperliquid_execution_state_submit_kind_invalid")
    if submit_kind == "entry" and state.get("entry_submit_attempted") is not True:
        blockers.append("hyperliquid_entry_attempt_not_recorded")
    if submit_kind == "exit" and state.get("exit_submit_attempted") is not True:
        blockers.append("hyperliquid_exit_attempt_not_recorded")

    existing = _read_json(receipt_path)
    expected_identity = {
        "approval_id": str(approval.get("approval_id", "")),
        "run_id": str(run_manifest.get("run_id", "")),
        "candidate_set_id": str(run_manifest.get("candidate_set_id", "")),
        "protocol_id": str(protocol.get("protocol_id", "")),
        "risk_override_applied": approval.get("risk_override_applied"),
        "entry_context": approval.get("entry_context", {}),
        **{
            field: str(approval.get(field, "")).strip()
            for field in TESTNET_CANDIDATE_BINDING_FIELDS
        },
    }
    if existing and any(existing.get(key) != value for key, value in expected_identity.items()):
        blockers.append("hyperliquid_existing_receipt_identity_mismatch")

    row: dict[str, object] = {
        "checked_at_utc": checked_at.isoformat(),
        "status": "BLOCKED",
        "submit_kind": submit_kind,
        "execution_state_id": str(state.get("execution_state_id", "")),
        "approval_id": str(state.get("order_approval_id", "")),
        "leg_x_status": "",
        "leg_y_status": "",
        "events_written": 0,
        "receipt_written": False,
        "read_only": True,
        "signing_key_loaded": False,
        "order_submission_performed": False,
        "live_trading_authorized": False,
        "blockers": "",
    }
    if blockers:
        row["blockers"] = ";".join(_unique(blockers))
        _atomic_csv(pd.DataFrame([row]), capture_path)
        return {"status": "BLOCKED", "capture": capture_path, "receipt": receipt_path, **row}

    http = session or requests.Session()
    created_at = _parse_utc(str(state.get("created_at_utc", "")))
    lifecycle_started_at = (
        _parse_utc(str(existing.get("started_at_utc", "")))
        if existing.get("started_at_utc")
        else created_at
    )
    start_ms = int(lifecycle_started_at.timestamp() * 1000)
    try:
        fills = _post_info(
            http,
            resolved,
            {
                "type": "userFillsByTime",
                "user": resolved.master_address,
                "startTime": start_ms,
                "aggregateByTime": False,
            },
        )
        account_state = _post_info(
            http,
            resolved,
            {"type": "clearinghouseState", "user": resolved.master_address},
        )
        open_orders = _post_info(
            http,
            resolved,
            {"type": "openOrders", "user": resolved.master_address},
        )
        funding = _post_info(
            http,
            resolved,
            {
                "type": "userFunding",
                "user": resolved.master_address,
                "startTime": start_ms,
            },
        )
    except Exception as exc:  # noqa: BLE001 - exchange read boundary fails closed.
        row["blockers"] = f"hyperliquid_lifecycle_read_error:{type(exc).__name__}"
        _atomic_csv(pd.DataFrame([row]), capture_path)
        return {"status": "BLOCKED", "capture": capture_path, "receipt": receipt_path, **row}
    if (
        not isinstance(fills, list)
        or not isinstance(account_state, dict)
        or not isinstance(open_orders, list)
        or not isinstance(funding, list)
    ):
        row["blockers"] = "hyperliquid_lifecycle_read_payload_invalid"
        _atomic_csv(pd.DataFrame([row]), capture_path)
        return {"status": "BLOCKED", "capture": capture_path, "receipt": receipt_path, **row}

    statuses = _leg_fill_statuses(intents, references, fills)
    row["leg_x_status"], row["leg_y_status"] = statuses
    receipt = existing or _new_receipt(
        expected_identity=expected_identity,
        started_at=created_at,
    )
    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    events = [event for event in events if isinstance(event, dict)]
    coins = {
        _normalize_perp_coin(str(intent.get("market", "")))
        for intent in intents
        if isinstance(intent, dict)
    }
    positions = _pair_position_sizes(account_state, coins)
    pair_orders = _pair_open_orders(open_orders, coins)
    flat = len(positions) == 2 and all(abs(value) <= 0.0 for value in positions.values())
    new_events: list[dict[str, object]] = []
    if statuses == ["filled", "filled"]:
        new_events.append(
            _two_leg_event(
                state=state,
                submit_kind=submit_kind,
                references=references,
                intents=intents,
                fills=fills,
                checked_at=checked_at,
            )
        )
    else:
        anomaly_types = _pair_anomaly_types(statuses=statuses, positions=positions)
        anomaly_references = [str(value).strip() for value in references if str(value).strip()] or [
            str(state.get("execution_state_id", ""))
        ]
        for anomaly_type in anomaly_types:
            new_events.append(
                _event(
                    event_type="execution_anomaly",
                    state=state,
                    references=anomaly_references,
                    checked_at=checked_at,
                    extra={"anomaly_type": anomaly_type},
                )
            )
        new_events.extend(
            _recovery_events(
                state=state,
                anomaly_types=anomaly_types,
                checked_at=checked_at,
            )
        )
    if submit_kind == "exit" and statuses == ["filled", "filled"] and flat and not pair_orders:
        state_reference = (
            "state_"
            + sha256(
                _canonical_json(
                    {
                        "positions": positions,
                        "orders": pair_orders,
                        "checked_at_utc": checked_at.isoformat(),
                    }
                ).encode("utf-8")
            ).hexdigest()[:20]
        )
        new_events.append(
            _event(
                event_type="reconciled",
                state=state,
                references=[state_reference],
                checked_at=checked_at,
            )
        )
    if state.get("duplicate_entry_blocked") is True and str(
        state.get("duplicate_entry_blocked_at_utc", "")
    ):
        duplicate_at = _parse_utc(str(state["duplicate_entry_blocked_at_utc"]))
        new_events.append(
            _event(
                event_type="idempotency",
                state=state,
                references=[str(state.get("execution_state_id", ""))],
                checked_at=duplicate_at,
                extra={"duplicate_submit_blocked": True},
            )
        )

    receipt["events"] = _upsert_events(events, new_events)
    existing_funding = (
        receipt.get("funding_events") if isinstance(receipt.get("funding_events"), list) else []
    )
    receipt["funding_events"] = _upsert_funding_events(
        [row for row in existing_funding if isinstance(row, dict)],
        _funding_evidence(funding),
    )
    receipt["funding_query_complete"] = True
    receipt["completed_at_utc"] = checked_at.isoformat()
    receipt["final_state"] = {
        "reconciled": bool(submit_kind == "exit" and flat and not pair_orders),
        "position_x": float(positions.get(_normalize_perp_coin(str(intents[0]["market"])), 0.0)),
        "position_y": float(positions.get(_normalize_perp_coin(str(intents[1]["market"])), 0.0)),
        "open_order_count": len(pair_orders),
        "account_state_timestamp_utc": checked_at.isoformat(),
    }
    receipt["economics"] = _lifecycle_economics(receipt)
    from quant_platform.orchestration.hyperliquid_learning_and_risk import (
        _testnet_receipt_payload_hash,
    )

    receipt["receipt_hash"] = _testnet_receipt_payload_hash(receipt)
    _atomic_json(receipt, receipt_path)
    row["events_written"] = len(new_events)
    row["receipt_written"] = True
    complete = {
        "two_leg_entry",
        "two_leg_exit",
        "reconciled",
        "idempotency",
    }.issubset(
        {str(event.get("event_type", "")) for event in receipt["events"] if isinstance(event, dict)}
    )
    row["status"] = "COMPLETE" if complete else "PARTIAL"
    if statuses != ["filled", "filled"]:
        row["blockers"] = "hyperliquid_two_leg_terminal_fill_not_yet_proven"
    _atomic_csv(pd.DataFrame([row]), capture_path)
    return {"status": row["status"], "capture": capture_path, "receipt": receipt_path, **row}


def _new_receipt(
    *, expected_identity: dict[str, object], started_at: datetime
) -> dict[str, object]:
    return {
        "receipt_version": "hyperliquid-testnet-lifecycle-v3",
        "receipt_source": "hyperliquid_testnet_lifecycle_evidence_capture",
        "actual_testnet": True,
        "network": "testnet",
        **expected_identity,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": started_at.isoformat(),
        "events": [],
        "funding_events": [],
        "funding_query_complete": False,
        "economics": {},
        "final_state": {},
        "receipt_hash": "",
    }


def _ordered_exchange_references(*, state: dict[str, object], intents: list[object]) -> list[str]:
    """Keep one slot per leg so an orphaned leg remains observable."""

    reference_rows = (
        state.get("exchange_references")
        if isinstance(state.get("exchange_references"), list)
        else []
    )
    by_market = {
        _normalize_perp_coin(str(row.get("market", ""))): str(row.get("order_id", "")).strip()
        for row in reference_rows
        if isinstance(row, dict)
        and str(row.get("market", "")).strip()
        and str(row.get("order_id", "")).strip()
    }
    if by_market:
        return [
            by_market.get(_normalize_perp_coin(str(intent.get("market", ""))), "")
            if isinstance(intent, dict)
            else ""
            for intent in intents
        ]
    positional = (
        state.get("exchange_reference_ids")
        if isinstance(state.get("exchange_reference_ids"), list)
        else []
    )
    return [
        str(positional[index]).strip() if index < len(positional) else ""
        for index in range(len(intents))
    ]


def _pair_anomaly_types(*, statuses: list[str], positions: dict[str, float]) -> list[str]:
    anomaly_types: list[str] = []
    if "partial" in statuses:
        anomaly_types.append("partial_fill")
    filled_legs = sum(status == "filled" for status in statuses)
    nonzero_positions = sum(abs(value) > 1e-12 for value in positions.values())
    if filled_legs == 1 or nonzero_positions == 1:
        anomaly_types.append("orphan_position")
    return anomaly_types


def _recovery_events(
    *,
    state: dict[str, object],
    anomaly_types: list[str],
    checked_at: datetime,
) -> list[dict[str, object]]:
    if not anomaly_types or state.get("reconciled") is not True:
        return []
    actions = [
        str(value).strip()
        for value in (
            state.get("recovery_actions") if isinstance(state.get("recovery_actions"), list) else []
        )
        if str(value).strip()
    ]
    if not actions:
        return []
    identity_reference = (
        "recovery_"
        + sha256(
            _canonical_json(
                {
                    "execution_state_id": str(state.get("execution_state_id", "")),
                    "actions": actions,
                }
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    common = {
        "reduce_only": True,
        "recovery_actions": actions,
        "reconciled_flat": True,
    }
    event_types = ["cancel_and_unwind"]
    if "partial_fill" in anomaly_types:
        event_types.append("partial_fill_recovery")
    if "orphan_position" in anomaly_types and any(
        action.startswith("reduce_only_flatten:") for action in actions
    ):
        event_types.append("emergency_flatten")
    return [
        _event(
            event_type=event_type,
            state=state,
            references=[identity_reference],
            checked_at=checked_at,
            extra=common,
        )
        for event_type in event_types
    ]


def _two_leg_event(
    *,
    state: dict[str, object],
    submit_kind: str,
    references: list[object],
    intents: list[object],
    fills: list[object],
    checked_at: datetime,
) -> dict[str, object]:
    fill_evidence = _matched_fill_evidence(
        intents=intents,
        references=references,
        fills=fills,
    )
    return _event(
        event_type="two_leg_entry" if submit_kind == "entry" else "two_leg_exit",
        state=state,
        references=references,
        checked_at=checked_at,
        extra={
            "leg_x_status": "filled",
            "leg_y_status": "filled",
            "fill_evidence": fill_evidence,
            "fill_economics_complete": bool(
                len(fill_evidence) >= 2
                and all(row.get("economics_complete") is True for row in fill_evidence)
            ),
        },
    )


def _event(
    *,
    event_type: str,
    state: dict[str, object],
    references: list[object],
    checked_at: datetime,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "event_type": event_type,
        "anomaly_type": str((extra or {}).get("anomaly_type", "")),
        "execution_state_id": str(state.get("execution_state_id", "")),
        "references": [str(value) for value in references],
    }
    return {
        "event_id": "hltestnetevent_"
        + sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20],
        "event_type": event_type,
        "timestamp_utc": checked_at.isoformat(),
        "exchange_reference_ids": [str(value) for value in references],
        "execution_state_id": str(state.get("execution_state_id", "")),
        **(extra or {}),
    }


def _upsert_events(
    existing: list[dict[str, object]], new_events: list[dict[str, object]]
) -> list[dict[str, object]]:
    indexed = {str(event.get("event_id", "")): event for event in existing}
    for event in new_events:
        indexed[str(event.get("event_id", ""))] = event
    return sorted(indexed.values(), key=lambda event: str(event.get("timestamp_utc", "")))


def _matched_fill_evidence(
    *, intents: list[object], references: list[object], fills: list[object]
) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for intent, reference in zip(intents, references, strict=True):
        if not isinstance(intent, dict):
            continue
        coin = _normalize_perp_coin(str(intent.get("market", "")))
        expected_limit = _decimal(intent.get("limit_price"))
        intent_side = str(intent.get("side", "")).strip().upper()
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            if str(fill.get("oid", "")) != str(reference):
                continue
            if _normalize_perp_coin(str(fill.get("coin", ""))) != coin:
                continue
            size = _decimal(fill.get("sz"))
            price = _decimal(fill.get("px"))
            fee = _decimal(fill.get("fee"))
            closed_pnl = _decimal(fill.get("closedPnl"))
            fill_time = _decimal(fill.get("time"))
            shortfall: Decimal | None = None
            if (
                size is not None
                and price is not None
                and expected_limit is not None
                and intent_side in {"BUY", "SELL"}
            ):
                difference = (
                    price - expected_limit if intent_side == "BUY" else expected_limit - price
                )
                shortfall = difference * abs(size)
            complete = bool(
                size is not None
                and abs(size) > 0
                and price is not None
                and price > 0
                and fee is not None
                and closed_pnl is not None
                and fill_time is not None
                and expected_limit is not None
                and expected_limit > 0
                and shortfall is not None
            )
            evidence.append(
                {
                    "order_id": str(reference),
                    "trade_id": str(fill.get("tid", "")),
                    "transaction_hash": str(fill.get("hash", "")),
                    "coin": coin,
                    "intent_side": intent_side,
                    "exchange_side": str(fill.get("side", "")),
                    "size": float(abs(size)) if size is not None else None,
                    "price": float(price) if price is not None else None,
                    "expected_limit_price": (
                        float(expected_limit) if expected_limit is not None else None
                    ),
                    "fee_usd": float(abs(fee)) if fee is not None else None,
                    "fee_token": str(fill.get("feeToken", "USDC")),
                    "closed_pnl_usd": (float(closed_pnl) if closed_pnl is not None else None),
                    "implementation_shortfall_vs_limit_usd": (
                        float(shortfall) if shortfall is not None else None
                    ),
                    "fill_time_ms": int(fill_time) if fill_time is not None else None,
                    "economics_complete": complete,
                }
            )
    return sorted(
        evidence,
        key=lambda row: (
            str(row.get("order_id", "")),
            int(row.get("fill_time_ms") or 0),
            str(row.get("trade_id", "")),
        ),
    )


def _funding_evidence(rows: list[object]) -> list[dict[str, object]]:
    evidence: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        delta = row.get("delta") if isinstance(row.get("delta"), dict) else {}
        timestamp = _decimal(row.get("time"))
        usdc = _decimal(delta.get("usdc"))
        funding_rate = _decimal(delta.get("fundingRate"))
        position_size = _decimal(delta.get("szi"))
        coin = _normalize_perp_coin(str(delta.get("coin", "")))
        identity = {
            "time": int(timestamp) if timestamp is not None else None,
            "coin": coin,
            "usdc": float(usdc) if usdc is not None else None,
            "hash": str(row.get("hash", "")),
        }
        evidence.append(
            {
                "funding_event_id": "hlfunding_"
                + sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:20],
                "time_ms": int(timestamp) if timestamp is not None else None,
                "coin": coin,
                "funding_rate": (float(funding_rate) if funding_rate is not None else None),
                "position_size": (float(position_size) if position_size is not None else None),
                "funding_pnl_usd": float(usdc) if usdc is not None else None,
                "transaction_hash": str(row.get("hash", "")),
                "economics_complete": bool(
                    timestamp is not None
                    and coin
                    and funding_rate is not None
                    and position_size is not None
                    and usdc is not None
                ),
            }
        )
    return evidence


def _upsert_funding_events(
    existing: list[dict[str, object]], new_events: list[dict[str, object]]
) -> list[dict[str, object]]:
    indexed = {
        str(event.get("funding_event_id", "")): event
        for event in existing
        if str(event.get("funding_event_id", ""))
    }
    for event in new_events:
        indexed[str(event.get("funding_event_id", ""))] = event
    return sorted(
        indexed.values(),
        key=lambda event: (
            int(event.get("time_ms") or 0),
            str(event.get("funding_event_id", "")),
        ),
    )


def _lifecycle_economics(receipt: dict[str, object]) -> dict[str, object]:
    events = receipt.get("events") if isinstance(receipt.get("events"), list) else []
    terminal_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("event_type") in {"two_leg_entry", "two_leg_exit"}
    ]
    fills = [
        fill
        for event in terminal_events
        for fill in (
            event.get("fill_evidence") if isinstance(event.get("fill_evidence"), list) else []
        )
        if isinstance(fill, dict)
    ]
    funding_events = (
        receipt.get("funding_events") if isinstance(receipt.get("funding_events"), list) else []
    )
    gross = sum(
        float(fill["closed_pnl_usd"]) for fill in fills if fill.get("closed_pnl_usd") is not None
    )
    fees = sum(float(fill["fee_usd"]) for fill in fills if fill.get("fee_usd") is not None)
    shortfall = sum(
        float(fill["implementation_shortfall_vs_limit_usd"])
        for fill in fills
        if fill.get("implementation_shortfall_vs_limit_usd") is not None
    )
    funding_pnl = sum(
        float(event["funding_pnl_usd"])
        for event in funding_events
        if isinstance(event, dict) and event.get("funding_pnl_usd") is not None
    )
    complete = bool(
        {"two_leg_entry", "two_leg_exit"}
        == {str(event.get("event_type", "")) for event in terminal_events}
        and len(fills) >= 4
        and all(fill.get("economics_complete") is True for fill in fills)
        and receipt.get("funding_query_complete") is True
        and all(
            isinstance(event, dict) and event.get("economics_complete") is True
            for event in funding_events
        )
    )
    return {
        "economics_complete": complete,
        "terminal_fill_count": len(fills),
        "gross_realized_pnl_usd": gross,
        "fees_usd": fees,
        "funding_pnl_usd": funding_pnl,
        "implementation_shortfall_vs_limit_usd": shortfall,
        "net_realized_pnl_after_cost_usd": gross - fees + funding_pnl,
        "slippage_treatment": (
            "diagnostic_only_already_reflected_in_realized_pnl_not_double_subtracted"
        ),
    }


def _leg_fill_statuses(
    intents: list[object], references: list[object], fills: list[object]
) -> list[str]:
    statuses: list[str] = []
    for intent, reference in zip(intents, references, strict=True):
        if not isinstance(intent, dict):
            statuses.append("missing")
            continue
        coin = _normalize_perp_coin(str(intent.get("market", "")))
        expected = _decimal(intent.get("size"))
        filled = Decimal(0)
        for fill in fills:
            if not isinstance(fill, dict):
                continue
            if str(fill.get("oid", "")) != str(reference):
                continue
            if _normalize_perp_coin(str(fill.get("coin", ""))) != coin:
                continue
            size = _decimal(fill.get("sz"))
            if size is not None:
                filled += abs(size)
        if expected is None or expected <= 0 or filled <= 0:
            statuses.append("missing")
        elif filled >= expected:
            statuses.append("filled")
        else:
            statuses.append("partial")
    return statuses


def _decimal(value: object) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return number if number.is_finite() else None


def _post_info(
    session: requests.Session,
    config: HyperliquidTestnetConfig,
    payload: dict[str, object],
) -> Any:
    response = session.post(f"{config.base_url.rstrip('/')}/info", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
