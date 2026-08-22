"""Immutable shared credit reservations for scheduled Crypto Wizards lanes."""

from __future__ import annotations

import fcntl
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import (
    create_exclusive_text,
    promote_staged_file,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "thewiz.wizard_credit_ledger.v1"
RECONCILIATION_SCHEMA_VERSION = "thewiz.wizard_credit_reconciliation.v2"
DISCOVERY_LANE = "exhaustive_discovery_sweep"
DISCOVERY_REFRESH_LANE = "daily_discovery_refresh"
PROOF_LANE = "exact_mode_and_copula_proofs"
ALLOWED_LANES = frozenset({DISCOVERY_LANE, DISCOVERY_REFRESH_LANE, PROOF_LANE})


def validate_wizard_credit_lane_evidence(
    *,
    root: Path = ROOT,
    lane: str,
    credit_date_utc: str,
    reservation_id: str,
    reconciliation_id: str,
    reservation_path: str,
    reconciliation_path: str,
) -> dict[str, Any]:
    """Verify exact immutable reservation and reconciliation bindings."""

    if lane not in ALLOWED_LANES:
        return {"status": "BLOCKED", "blocker": "unsupported_credit_lane"}
    try:
        _validated_credit_date(credit_date_utc)
    except ValueError:
        return {"status": "BLOCKED", "blocker": "invalid_credit_date_utc"}
    day_root = _day_root(root, credit_date_utc)
    with _ledger_lock(root):
        state, blocker = _load_day(day_root, day=credit_date_utc)
    if blocker:
        return {"status": "BLOCKED", "blocker": blocker}
    reservation = state["reservations"].get(lane)
    if reservation is None or reservation.get("reservation_id") != reservation_id:
        return {"status": "BLOCKED", "blocker": "reservation_identity_mismatch"}
    expected_reservation_path = day_root / "reservations" / f"{lane}.json"
    if _resolve_evidence_path(root, reservation_path) != expected_reservation_path.resolve():
        return {"status": "BLOCKED", "blocker": "reservation_path_mismatch"}
    matching_reconciliations = [
        (key_hash, receipt)
        for (receipt_lane, key_hash), receipt in state["reconciliations"].items()
        if receipt_lane == lane and receipt.get("reconciliation_id") == reconciliation_id
    ]
    if len(matching_reconciliations) != 1:
        return {"status": "BLOCKED", "blocker": "reconciliation_identity_mismatch"}
    key_hash, reconciliation = matching_reconciliations[0]
    expected_reconciliation_path = day_root / "reconciliations" / f"{lane}_{key_hash}.json"
    if _resolve_evidence_path(root, reconciliation_path) != expected_reconciliation_path.resolve():
        return {"status": "BLOCKED", "blocker": "reconciliation_path_mismatch"}
    if reconciliation.get("schema_version") != RECONCILIATION_SCHEMA_VERSION:
        return {
            "status": "BLOCKED",
            "blocker": "legacy_credit_reconciliation_not_vendor_delta_proven",
        }
    if reconciliation.get("vendor_delta_status") != "PASS_VENDOR_DELTA":
        return {
            "status": "BLOCKED",
            "blocker": "credit_reconciliation_vendor_delta_not_proven",
        }
    return {
        "status": "PASS",
        "blocker": "",
        "reservation_id": reservation_id,
        "reconciliation_id": reconciliation_id,
        "reservation_path": _relative(expected_reservation_path, root),
        "reservation_sha256": sha256(expected_reservation_path.read_bytes()).hexdigest(),
        "reconciliation_path": _relative(expected_reconciliation_path, root),
        "reconciliation_sha256": sha256(expected_reconciliation_path.read_bytes()).hexdigest(),
        "planned_credits": int(reservation["planned_credits"]),
        "attempted_credits": int(reconciliation["attempted_credits"]),
        "completed_credits": int(reconciliation["completed_credits"]),
        "external_requests": int(reconciliation["external_requests"]),
        "observed_used_before": int(reconciliation["observed_used_before"]),
        "observed_used_after": int(reconciliation["observed_used_after"]),
        "observed_used_delta": int(reconciliation["observed_used_delta"]),
        "vendor_delta_status": str(reconciliation["vendor_delta_status"]),
        "activity_rows": list(reconciliation["activity_rows"]),
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def reserve_wizard_credit_lane(
    *,
    root: Path = ROOT,
    lane: str,
    planned_credits: int,
    now: datetime | None = None,
    daily_credit_limit: int = 1000,
    protected_reserve: int = 100,
    max_external_requests: int | None = None,
) -> CommandResult:
    """Reserve one lane's maximum UTC-day spend without making a vendor call."""

    _validate_contract(
        lane=lane,
        planned_credits=planned_credits,
        daily_credit_limit=daily_credit_limit,
        protected_reserve=protected_reserve,
    )
    if max_external_requests is not None and max_external_requests <= 0:
        raise ValueError("max_external_requests must be positive when provided")
    timestamp = _as_utc(now)
    day = timestamp.date().isoformat()
    day_root = _day_root(root, day)
    reservation_path = day_root / "reservations" / f"{lane}.json"
    status_path = root / "reports" / "active" / "wizard_credit_ledger_status.json"
    usable = daily_credit_limit - protected_reserve

    with _ledger_lock(root):
        state, blocker = _load_day(day_root, day=day)
        if blocker:
            return _blocked_result(
                root=root,
                status_path=status_path,
                day=day,
                lane=lane,
                planned_credits=planned_credits,
                blocker=blocker,
            )

        existing = state["reservations"].get(lane)
        if existing is not None:
            matches = all(
                (
                    int(existing["planned_credits"]) == planned_credits,
                    int(existing["daily_credit_limit"]) == daily_credit_limit,
                    int(existing["protected_reserve"]) == protected_reserve,
                )
            )
            if not matches:
                return _blocked_result(
                    root=root,
                    status_path=status_path,
                    day=day,
                    lane=lane,
                    planned_credits=planned_credits,
                    blocker="existing_lane_reservation_contract_mismatch",
                )
            lane_reconciliations = [
                receipt
                for (receipt_lane, _), receipt in state["reconciliations"].items()
                if receipt_lane == lane
            ]
            reconciled_zero_attempt_retry = bool(lane_reconciliations) and all(
                int(receipt["attempted_credits"]) == 0
                and int(receipt["completed_credits"]) == 0
                and int(receipt["external_requests"]) == 0
                for receipt in lane_reconciliations
            )
            reservation_sha256 = sha256(_encoded_json(existing)).hexdigest()
            zero_effect_authority_retry = _zero_effect_authority_retry_safe(
                reservation_id=str(existing["reservation_id"]),
                reservation_sha256=reservation_sha256,
            )
            binding = None
            if reconciled_zero_attempt_retry or zero_effect_authority_retry:
                binding = _register_current_external_effect_binding(
                    reservation=existing,
                    max_external_requests=max_external_requests,
                    binding_nonce=f"{lane}:{timestamp.isoformat()}",
                )
            external_spend_authorized = binding is not None
            summary = _day_summary(
                root=root,
                day=day,
                state=state,
                status="REUSED",
                lane=lane,
                blocker="",
                reservation=existing,
                external_spend_authorized=external_spend_authorized,
                effect_reservation_binding_id=(
                    binding.binding_id if binding is not None else ""
                ),
                zero_effect_authority_retry=zero_effect_authority_retry,
            )
            summary["zero_attempt_reconciliation_retry"] = reconciled_zero_attempt_retry
            _atomic_json(summary, status_path)
            return CommandResult(
                paths={"reservation": reservation_path, "ledger_status": status_path},
                summary=summary,
            )

        reserved_before = sum(
            int(receipt["planned_credits"]) for receipt in state["reservations"].values()
        )
        if reserved_before + planned_credits > usable:
            return _blocked_result(
                root=root,
                status_path=status_path,
                day=day,
                lane=lane,
                planned_credits=planned_credits,
                blocker="combined_lane_reservations_exceed_usable_daily_credits",
            )

        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "credit_date_utc": day,
            "reserved_at_utc": timestamp.isoformat(),
            "lane": lane,
            "planned_credits": planned_credits,
            "daily_credit_limit": daily_credit_limit,
            "protected_reserve": protected_reserve,
            "usable_daily_credits": usable,
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        body["reservation_id"] = (
            "wizardcredit_" + sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:20]
        )
        receipt = _seal(body)
        binding = _register_current_external_effect_binding(
            reservation=receipt,
            max_external_requests=max_external_requests,
            binding_nonce=f"{lane}:{timestamp.isoformat()}",
        )
        reservation_path.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive_json(receipt, reservation_path)
        state["reservations"][lane] = receipt
        summary = _day_summary(
            root=root,
            day=day,
            state=state,
            status="PASS",
            lane=lane,
            blocker="",
            reservation=receipt,
            external_spend_authorized=binding is not None,
            effect_reservation_binding_id=(
                binding.binding_id if binding is not None else ""
            ),
        )
        _atomic_json(summary, status_path)
        return CommandResult(
            paths={"reservation": reservation_path, "ledger_status": status_path},
            summary=summary,
        )


def reconcile_wizard_credit_lane(
    *,
    root: Path = ROOT,
    lane: str,
    reservation_id: str,
    reconciliation_key: str,
    attempted_credits: int,
    completed_credits: int,
    external_requests: int,
    observed_used_before: int | None = None,
    observed_used_after: int | None = None,
    activity_rows: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    now: datetime | None = None,
) -> CommandResult:
    """Bind actual lane activity to a reservation in an immutable receipt."""

    if lane not in ALLOWED_LANES:
        raise ValueError(f"unsupported Wizard credit lane: {lane}")
    if not reservation_id:
        raise ValueError("reservation_id is required")
    if not reconciliation_key.strip():
        raise ValueError("reconciliation_key is required")
    for name, value in (
        ("attempted_credits", attempted_credits),
        ("completed_credits", completed_credits),
        ("external_requests", external_requests),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if completed_credits > attempted_credits:
        raise ValueError("completed_credits cannot exceed attempted_credits")
    if observed_used_before is not None and observed_used_before < 0:
        raise ValueError("observed_used_before must be non-negative when provided")
    if observed_used_after is not None and observed_used_after < 0:
        raise ValueError("observed_used_after must be non-negative when provided")
    normalized_activity = _normalize_activity_rows(activity_rows or ())
    if sum(row["external_requests"] for row in normalized_activity) != external_requests:
        raise ValueError("activity external requests do not match reconciliation total")
    if sum(row["attempted_credits"] for row in normalized_activity) != attempted_credits:
        raise ValueError("activity attempted credits do not match reconciliation total")
    if sum(row["completed_credits"] for row in normalized_activity) != completed_credits:
        raise ValueError("activity completed credits do not match reconciliation total")
    if attempted_credits > 0 and (
        observed_used_before is None or observed_used_after is None
    ):
        raise ValueError("paid reconciliation requires vendor usage before and after")
    if observed_used_before is None:
        observed_used_before = 0
    if observed_used_after is None:
        observed_used_after = observed_used_before
    observed_used_delta = observed_used_after - observed_used_before
    if observed_used_delta < 0:
        raise ValueError("vendor used credits decreased during reconciliation")
    if observed_used_delta != attempted_credits:
        raise ValueError("vendor credit delta does not equal attempted credits")

    timestamp = _as_utc(now)
    day = timestamp.date().isoformat()
    day_root = _day_root(root, day)
    key_hash = sha256(reconciliation_key.encode("utf-8")).hexdigest()[:20]
    receipt_path = day_root / "reconciliations" / f"{lane}_{key_hash}.json"
    status_path = root / "reports" / "active" / "wizard_credit_ledger_status.json"

    with _ledger_lock(root):
        state, blocker = _load_day(day_root, day=day)
        if blocker:
            return _blocked_result(
                root=root,
                status_path=status_path,
                day=day,
                lane=lane,
                planned_credits=attempted_credits,
                blocker=blocker,
            )
        reservation = state["reservations"].get(lane)
        if reservation is None or reservation.get("reservation_id") != reservation_id:
            return _blocked_result(
                root=root,
                status_path=status_path,
                day=day,
                lane=lane,
                planned_credits=attempted_credits,
                blocker="matching_lane_reservation_missing",
            )

        existing = state["reconciliations"].get((lane, key_hash))
        if existing is not None:
            matches = all(
                (
                    existing.get("reservation_id") == reservation_id,
                    existing.get("reconciliation_key") == reconciliation_key,
                    int(existing.get("attempted_credits", -1)) == attempted_credits,
                    int(existing.get("completed_credits", -1)) == completed_credits,
                    int(existing.get("external_requests", -1)) == external_requests,
                    existing.get("observed_used_before") == observed_used_before,
                    existing.get("observed_used_after") == observed_used_after,
                    existing.get("observed_used_delta") == observed_used_delta,
                    existing.get("activity_rows") == normalized_activity,
                    existing.get("schema_version") == RECONCILIATION_SCHEMA_VERSION,
                )
            )
            if not matches:
                return _blocked_result(
                    root=root,
                    status_path=status_path,
                    day=day,
                    lane=lane,
                    planned_credits=attempted_credits,
                    blocker="existing_reconciliation_contract_mismatch",
                )
            summary = _day_summary(
                root=root,
                day=day,
                state=state,
                status="REUSED_RECONCILIATION",
                lane=lane,
                blocker="",
                reservation=reservation,
                reconciliation=existing,
            )
            _atomic_json(summary, status_path)
            return CommandResult(
                paths={"reconciliation": receipt_path, "ledger_status": status_path},
                summary=summary,
            )

        lane_attempted_before = sum(
            int(receipt["attempted_credits"])
            for (receipt_lane, _), receipt in state["reconciliations"].items()
            if receipt_lane == lane
        )
        if lane_attempted_before + attempted_credits > int(reservation["planned_credits"]):
            return _blocked_result(
                root=root,
                status_path=status_path,
                day=day,
                lane=lane,
                planned_credits=attempted_credits,
                blocker="reconciled_lane_attempts_exceed_reservation",
            )

        body = {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "credit_date_utc": day,
            "reconciled_at_utc": timestamp.isoformat(),
            "lane": lane,
            "reservation_id": reservation_id,
            "reconciliation_key": reconciliation_key,
            "reconciliation_key_sha256_prefix": key_hash,
            "attempted_credits": attempted_credits,
            "completed_credits": completed_credits,
            "external_requests": external_requests,
            "observed_used_before": observed_used_before,
            "observed_used_after": observed_used_after,
            "observed_used_delta": observed_used_delta,
            "vendor_delta_status": "PASS_VENDOR_DELTA",
            "activity_rows": normalized_activity,
            "research_only": True,
            "candidate_promotion_authority": False,
            "order_submission_included": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        body["reconciliation_id"] = (
            "wizardcreditrecon_" + sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:20]
        )
        receipt = _seal(body)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        _write_exclusive_json(receipt, receipt_path)
        state["reconciliations"][(lane, key_hash)] = receipt
        summary = _day_summary(
            root=root,
            day=day,
            state=state,
            status="PASS_RECONCILED",
            lane=lane,
            blocker="",
            reservation=reservation,
            reconciliation=receipt,
        )
        _atomic_json(summary, status_path)
        return CommandResult(
            paths={"reconciliation": receipt_path, "ledger_status": status_path},
            summary=summary,
        )


def _validate_contract(
    *, lane: str, planned_credits: int, daily_credit_limit: int, protected_reserve: int
) -> None:
    if lane not in ALLOWED_LANES:
        raise ValueError(f"unsupported Wizard credit lane: {lane}")
    if planned_credits <= 0:
        raise ValueError("planned_credits must be positive")
    if daily_credit_limit <= 0:
        raise ValueError("daily_credit_limit must be positive")
    if not 0 <= protected_reserve < daily_credit_limit:
        raise ValueError("protected_reserve must be within the daily limit")


def _load_day(day_root: Path, *, day: str) -> tuple[dict[str, Any], str]:
    state: dict[str, Any] = {"reservations": {}, "reconciliations": {}}
    reservation_dir = day_root / "reservations"
    reconciliation_dir = day_root / "reconciliations"
    try:
        for path in sorted(reservation_dir.glob("*.json")) if reservation_dir.exists() else ():
            receipt = _read_sealed(path)
            lane = str(receipt.get("lane", ""))
            if (
                path.is_symlink()
                or path.name != f"{lane}.json"
                or lane in state["reservations"]
                or not _reservation_semantics_valid(receipt, day=day, lane=lane)
            ):
                return state, f"malformed_credit_reservation:{path.name}"
            state["reservations"][lane] = receipt
        for path in (
            sorted(reconciliation_dir.glob("*.json")) if reconciliation_dir.exists() else ()
        ):
            receipt = _read_sealed(path)
            lane = str(receipt.get("lane", ""))
            key_hash = str(receipt.get("reconciliation_key_sha256_prefix", ""))
            key = (lane, key_hash)
            reservation = state["reservations"].get(lane)
            if (
                path.is_symlink()
                or path.name != f"{lane}_{key_hash}.json"
                or key in state["reconciliations"]
                or reservation is None
                or receipt.get("reservation_id") != reservation.get("reservation_id")
                or not _reconciliation_semantics_valid(
                    receipt,
                    day=day,
                    lane=lane,
                    key_hash=key_hash,
                )
            ):
                return state, f"malformed_credit_reconciliation:{path.name}"
            state["reconciliations"][key] = receipt
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return state, f"malformed_credit_ledger:{type(exc).__name__}"

    limits = {
        (int(receipt["daily_credit_limit"]), int(receipt["protected_reserve"]))
        for receipt in state["reservations"].values()
    }
    if len(limits) > 1:
        return state, "credit_reservation_contracts_disagree"
    if limits:
        limit, reserve = next(iter(limits))
        total_reserved = sum(
            int(receipt["planned_credits"]) for receipt in state["reservations"].values()
        )
        if total_reserved > limit - reserve:
            return state, "persisted_reservations_exceed_usable_daily_credits"
    for lane, reservation in state["reservations"].items():
        attempted = sum(
            int(receipt["attempted_credits"])
            for (receipt_lane, _), receipt in state["reconciliations"].items()
            if receipt_lane == lane
        )
        if attempted > int(reservation["planned_credits"]):
            return state, f"persisted_reconciliations_exceed_lane_reservation:{lane}"
    return state, ""


def _reservation_semantics_valid(receipt: dict[str, Any], *, day: str, lane: str) -> bool:
    try:
        planned = int(receipt.get("planned_credits", 0))
        limit = int(receipt.get("daily_credit_limit", 0))
        protected = int(receipt.get("protected_reserve", -1))
        usable = int(receipt.get("usable_daily_credits", -1))
        reservation_id = str(receipt.get("reservation_id", ""))
        body = dict(receipt)
        body.pop("receipt_sha256", None)
        body.pop("reservation_id", None)
        expected_id = (
            "wizardcredit_" + sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:20]
        )
    except (TypeError, ValueError):
        return False
    return bool(
        receipt.get("schema_version") == SCHEMA_VERSION
        and receipt.get("credit_date_utc") == day
        and lane in ALLOWED_LANES
        and planned > 0
        and limit > 0
        and 0 <= protected < limit
        and usable == limit - protected
        and reservation_id == expected_id
        and receipt.get("research_only") is True
        and receipt.get("candidate_promotion_authority") is False
        and receipt.get("order_submission_included") is False
        and receipt.get("testnet_order_authority") is False
        and receipt.get("live_trading_authorized") is False
    )


def _reconciliation_semantics_valid(
    receipt: dict[str, Any], *, day: str, lane: str, key_hash: str
) -> bool:
    try:
        attempted = int(receipt.get("attempted_credits", -1))
        completed = int(receipt.get("completed_credits", -1))
        external_requests = int(receipt.get("external_requests", -1))
        schema_version = str(receipt.get("schema_version", ""))
        observed_before = receipt.get("observed_used_before")
        if observed_before is not None and int(observed_before) < 0:
            return False
        reconciliation_key = str(receipt.get("reconciliation_key", ""))
        reconciliation_id = str(receipt.get("reconciliation_id", ""))
        body = dict(receipt)
        body.pop("receipt_sha256", None)
        body.pop("reconciliation_id", None)
        expected_id = (
            "wizardcreditrecon_" + sha256(_canonical_json(body).encode("utf-8")).hexdigest()[:20]
        )
    except (TypeError, ValueError):
        return False
    return bool(
        schema_version in {SCHEMA_VERSION, RECONCILIATION_SCHEMA_VERSION}
        and receipt.get("credit_date_utc") == day
        and lane in ALLOWED_LANES
        and re.fullmatch(r"[0-9a-f]{20}", key_hash)
        and key_hash == sha256(reconciliation_key.encode("utf-8")).hexdigest()[:20]
        and attempted >= 0
        and 0 <= completed <= attempted
        and external_requests >= 0
        and reconciliation_id == expected_id
        and receipt.get("research_only") is True
        and receipt.get("candidate_promotion_authority") is False
        and receipt.get("order_submission_included") is False
        and receipt.get("testnet_order_authority") is False
        and receipt.get("live_trading_authorized") is False
        and (
            schema_version == SCHEMA_VERSION
            or _v2_reconciliation_evidence_valid(
                receipt,
                attempted=attempted,
                completed=completed,
                external_requests=external_requests,
            )
        )
    )


def _normalize_activity_rows(
    rows: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in rows:
        if not isinstance(raw, dict):
            raise TypeError("credit activity row must be an object")
        lane = str(raw.get("lane", "")).strip()
        if not lane or lane in names:
            raise ValueError("credit activity lane must be non-empty and unique")
        names.add(lane)
        values: dict[str, int] = {}
        for field in (
            "external_requests",
            "credit_cost",
            "attempted_credits",
            "completed_credits",
        ):
            value = raw.get(field)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"credit activity {field} must be an integer")
            values[field] = value
        if values["external_requests"] < 0 or values["credit_cost"] <= 0:
            raise ValueError("credit activity requests and cost are invalid")
        if values["attempted_credits"] != (
            values["external_requests"] * values["credit_cost"]
        ):
            raise ValueError("credit activity attempted credits do not match calls times cost")
        if not 0 <= values["completed_credits"] <= values["attempted_credits"]:
            raise ValueError("credit activity completed credits are invalid")
        if values["completed_credits"] % values["credit_cost"]:
            raise ValueError("credit activity completed credits are not whole calls")
        normalized.append({"lane": lane, **values})
    return sorted(normalized, key=lambda row: row["lane"])


def _v2_reconciliation_evidence_valid(
    receipt: dict[str, Any],
    *,
    attempted: int,
    completed: int,
    external_requests: int,
) -> bool:
    try:
        before = int(receipt.get("observed_used_before", -1))
        after = int(receipt.get("observed_used_after", -1))
        delta = int(receipt.get("observed_used_delta", -1))
        activity = _normalize_activity_rows(receipt.get("activity_rows", []))
    except (TypeError, ValueError):
        return False
    return bool(
        before >= 0
        and after >= before
        and delta == after - before == attempted
        and sum(row["external_requests"] for row in activity) == external_requests
        and sum(row["attempted_credits"] for row in activity) == attempted
        and sum(row["completed_credits"] for row in activity) == completed
        and receipt.get("vendor_delta_status") == "PASS_VENDOR_DELTA"
    )


def _read_sealed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ledger receipt must be an object")
    expected = str(payload.get("receipt_sha256", ""))
    unsealed = dict(payload)
    unsealed.pop("receipt_sha256", None)
    if (
        not re.fullmatch(r"[0-9a-f]{64}", expected)
        or expected != sha256(_canonical_json(unsealed).encode("utf-8")).hexdigest()
    ):
        raise ValueError("ledger receipt hash mismatch")
    return payload


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "receipt_sha256": sha256(_canonical_json(body).encode("utf-8")).hexdigest(),
    }


def _day_summary(
    *,
    root: Path,
    day: str,
    state: dict[str, Any],
    status: str,
    lane: str,
    blocker: str,
    reservation: dict[str, Any] | None = None,
    reconciliation: dict[str, Any] | None = None,
    external_spend_authorized: bool = False,
    effect_reservation_binding_id: str = "",
    zero_effect_authority_retry: bool = False,
) -> dict[str, Any]:
    reservations = state["reservations"]
    reconciliations = state["reconciliations"]
    total_reserved = sum(int(item["planned_credits"]) for item in reservations.values())
    total_attempted = sum(int(item["attempted_credits"]) for item in reconciliations.values())
    lane_reserved = int((reservation or {}).get("planned_credits", 0))
    lane_attempted = sum(
        int(item["attempted_credits"])
        for (receipt_lane, _), item in reconciliations.items()
        if receipt_lane == lane
    )
    lane_reconciliation_ids = sorted(
        str(item.get("reconciliation_id", ""))
        for (receipt_lane, _), item in reconciliations.items()
        if receipt_lane == lane and item.get("reconciliation_id")
    )
    limit = int(reservation.get("daily_credit_limit", 0)) if reservation else 0
    protected = int(reservation.get("protected_reserve", 0)) if reservation else 0
    if not reservation and reservations:
        first = next(iter(reservations.values()))
        limit = int(first["daily_credit_limit"])
        protected = int(first["protected_reserve"])
    return {
        "schema_version": SCHEMA_VERSION,
        "credit_date_utc": day,
        "status": status,
        "lane": lane,
        "blocker": blocker,
        "reservation_id": str((reservation or {}).get("reservation_id", "")),
        "reconciliation_id": str((reconciliation or {}).get("reconciliation_id", "")),
        "daily_credit_limit": limit,
        "protected_reserve": protected,
        "usable_daily_credits": max(limit - protected, 0),
        "reserved_lane_count": len(reservations),
        "reconciliation_count": len(reconciliations),
        "total_reserved_credits": total_reserved,
        "total_reconciled_attempted_credits": total_attempted,
        "lane_reserved_credits": lane_reserved,
        "lane_reconciled_attempted_credits": lane_attempted,
        "lane_remaining_reserved_credits": max(lane_reserved - lane_attempted, 0),
        # A reservation grants spend authority exactly once: when it is first
        # published. Reusing an unreconciled receipt after a crash must never
        # replay vendor requests.
        "fresh_reservation": bool(status == "PASS" and external_spend_authorized),
        "external_spend_authorized": external_spend_authorized,
        "effect_reservation_binding_id": effect_reservation_binding_id,
        "zero_effect_authority_retry": zero_effect_authority_retry,
        "lane_external_authority_remaining_credits": (
            max(lane_reserved - lane_attempted, 0) if external_spend_authorized else 0
        ),
        "lane_reconciliation_ids": lane_reconciliation_ids,
        "headroom_after_reservations": max(limit - protected - total_reserved, 0),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_root": _relative(_day_root(root, day), root),
    }


def _blocked_result(
    *,
    root: Path,
    status_path: Path,
    day: str,
    lane: str,
    planned_credits: int,
    blocker: str,
) -> CommandResult:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "credit_date_utc": day,
        "status": "BLOCKED",
        "lane": lane,
        "blocker": blocker,
        "planned_credits": planned_credits,
        "reservation_id": "",
        "reconciliation_id": "",
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "fresh_reservation": False,
        "external_spend_authorized": False,
        "effect_reservation_binding_id": "",
        "zero_effect_authority_retry": False,
        "lane_external_authority_remaining_credits": 0,
        "evidence_root": _relative(_day_root(root, day), root),
    }
    _atomic_json(summary, status_path)
    return CommandResult(paths={"ledger_status": status_path}, summary=summary)


@contextmanager
def _ledger_lock(root: Path) -> Iterator[None]:
    path = root / "reports" / "active" / ".wizard_credit_ledger.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_exclusive_json(payload: dict[str, Any], path: Path) -> None:
    create_exclusive_text(
        path,
        _encoded_json(payload).decode("utf-8"),
        encoding="utf-8",
    )


def _encoded_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _register_current_external_effect_binding(
    *,
    reservation: dict[str, Any],
    max_external_requests: int | None,
    binding_nonce: str,
):
    from quant_platform.orchestration.corrective_external_effects import (
        current_external_effect_issuer,
    )

    issuer = current_external_effect_issuer()
    if issuer is None or max_external_requests is None:
        return None
    if max_external_requests > issuer.max_total_requests:
        raise ValueError("external request ceiling exceeds supervisor authority")
    planned_credits = int(reservation["planned_credits"])
    if planned_credits > issuer.max_total_credits:
        raise ValueError("credit reservation exceeds supervisor authority")
    return issuer.authority.register_external_reservation(
        run_id=issuer.run_id,
        intended_slot_id=issuer.intended_slot_id,
        provider_id=issuer.provider_id,
        account_scope_id=issuer.account_scope_id,
        reservation_id=str(reservation["reservation_id"]),
        reservation_sha256=sha256(_encoded_json(reservation)).hexdigest(),
        max_total_requests=max_external_requests,
        max_total_credits=planned_credits,
        binding_nonce=binding_nonce,
    )


def _zero_effect_authority_retry_safe(
    *,
    reservation_id: str,
    reservation_sha256: str,
) -> bool:
    from quant_platform.orchestration.corrective_external_effects import (
        current_external_effect_issuer,
    )

    issuer = current_external_effect_issuer()
    if issuer is None:
        return False
    return issuer.authority.external_reservation_retry_safe(
        reservation_id=reservation_id,
        reservation_sha256=reservation_sha256,
    )


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _day_root(root: Path, day: str) -> Path:
    return root / "data" / "research" / "wizard_credit_ledger" / day


def _validated_credit_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("credit_date_utc must be an ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError("credit_date_utc must use canonical YYYY-MM-DD form")
    return parsed


def _resolve_evidence_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve())
        return resolved
    except (OSError, ValueError):
        return root / ".invalid-wizard-credit-evidence"


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
