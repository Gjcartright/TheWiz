"""Validate a future live canary without submitting or authorizing orders."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.orchestration.corrective_runtime import promote_staged_file

EXECUTION_SCHEMA_VERSION = "thewiz.live_canary_execution_receipt.v2"
REVIEW_SCHEMA_VERSION = "thewiz.live_canary_post_canary_review.v1"
OUTCOME_SCHEMA_VERSION = "thewiz.live_canary_outcome.v4"


def evaluate_live_canary_outcome(
    *,
    root: Path,
    now: datetime,
    authorization: dict[str, Any],
    authorization_path: Path,
    approval: dict[str, Any],
    policy: dict[str, Any],
    candidate: dict[str, Any],
) -> Path:
    """Evaluate immutable execution and review receipts; never grant more authority."""

    active = root / "reports" / "active"
    execution_path = active / "live_canary_execution_receipt.json"
    outcome_path = active / "live_canary_outcome_evaluation.json"
    execution = _read_json(execution_path)
    if not execution:
        return _write_outcome(
            outcome_path,
            now=now,
            status="NOT_EXECUTED",
            authorization_path=authorization_path,
            execution={},
            execution_blockers=[],
            review_blockers=["post_canary_review_waits_for_execution"],
        )

    evidence_authorization = authorization
    evidence_authorization_path = authorization_path
    frozen_authorization_path = _safe_root_path(
        root, str(execution.get("authorization_receipt_path", ""))
    )
    if frozen_authorization_path is not None and frozen_authorization_path.is_file():
        frozen_authorization = _read_json(frozen_authorization_path)
        if frozen_authorization:
            evidence_authorization = frozen_authorization
            evidence_authorization_path = frozen_authorization_path
    execution_blockers = _validate_execution(
        root=root,
        now=now,
        execution=execution,
        authorization=evidence_authorization,
        authorization_path=evidence_authorization_path,
        approval=approval,
        policy=policy,
        candidate=candidate,
    )
    if any(
        blocker.startswith("canary_execution_without_authority") for blocker in execution_blockers
    ):
        status = "INCIDENT_UNAUTHORIZED_EXECUTION_EVIDENCE"
    elif execution_blockers:
        status = "BLOCKED_EXECUTION_RECEIPT_INVALID"
    else:
        status = "PENDING_POST_CANARY_REVIEW"

    review_path = root / "reports" / "supreme_team" / "live_canary_post_canary_review.json"
    review = _read_json(review_path)
    review_blockers = _validate_review(
        root=root,
        review=review,
        execution=execution,
        execution_path=execution_path,
    )
    if not execution_blockers and review_blockers:
        status = "BLOCKED_POST_CANARY_REVIEW"
    elif not execution_blockers and not review_blockers:
        status = "PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY"
    return _write_outcome(
        outcome_path,
        now=now,
        status=status,
        authorization_path=evidence_authorization_path,
        execution=execution,
        execution_blockers=execution_blockers,
        review_blockers=review_blockers,
        review_path=review_path,
    )


def _validate_execution(
    *,
    root: Path,
    now: datetime,
    execution: dict[str, Any],
    authorization: dict[str, Any],
    authorization_path: Path,
    approval: dict[str, Any],
    policy: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not (
        authorization.get("authorization_status") == "AUTHORIZED_FOR_ONE_LIVE_CANARY"
        and authorization.get("canary_execution_authority") is True
        and authorization.get("manual_executor_available") is True
        and authorization.get("authorization_reusable") is False
    ):
        blockers.append("canary_execution_without_authority:authorization_not_executable")
    if execution.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        blockers.append("canary_execution_schema_invalid")
    implementation_binding_fields = (
        "executor_contract_id",
        "executor_source_sha256",
        "executor_implementation_bundle_sha256",
    )
    if any(
        not str(authorization.get(field, "")).strip()
        for field in implementation_binding_fields
    ):
        blockers.append("canary_authorization_implementation_binding_missing")

    identity = _identity(execution, "execution")
    expected_id = "livecanaryexec_" + identity[:20]
    if execution.get("execution_id") != expected_id:
        blockers.append("canary_execution_id_invalid")
    if execution.get("receipt_identity_sha256") != identity:
        blockers.append("canary_execution_identity_hash_invalid")
    immutable_path = root / "data" / "live" / "canary_executions" / f"{expected_id}.json"
    if execution.get("immutable_execution_path") != _relative(immutable_path, root):
        blockers.append("canary_execution_immutable_path_invalid")
    elif not immutable_path.is_file() or _read_json(immutable_path) != execution:
        blockers.append("canary_execution_immutable_receipt_missing_or_mismatched")

    expected_bindings = {
        "authorization_receipt_sha256": _file_hash(authorization_path),
        "authorization_id": authorization.get("authorization_id"),
        "authorization_receipt_path": authorization.get(
            "immutable_authorization_path"
        ),
        "approval_id": approval.get("approval_id", ""),
        "candidate_receipt_id": candidate.get("candidate_receipt_id", ""),
        "live_canary_policy_id": authorization.get("live_canary_policy_id", ""),
        "executor_contract_id": authorization.get("executor_contract_id"),
        "executor_source_sha256": authorization.get("executor_source_sha256"),
        "executor_implementation_bundle_sha256": authorization.get(
            "executor_implementation_bundle_sha256"
        ),
        "pair": candidate.get("pair", ""),
    }
    if any(execution.get(key) != value for key, value in expected_bindings.items()):
        blockers.append("canary_execution_evidence_binding_mismatch")
    if execution.get("authorization_receipt_path"):
        expected_authorization_path = _relative(authorization_path, root)
        if (
            execution.get("authorization_receipt_path")
            != expected_authorization_path
            or _read_json(authorization_path) != authorization
        ):
            blockers.append("canary_execution_immutable_authorization_mismatch")
    if execution.get("live_canary_policy_id") != _policy_id(policy):
        blockers.append("canary_execution_policy_binding_mismatch")
    if (
        execution.get("repeat_authorized") is not False
        or execution.get("live_trading_authorized") is not False
        or execution.get("entry_retry_attempted") is not False
    ):
        blockers.append("canary_execution_claims_persistent_authority")

    started = pd.to_datetime(execution.get("started_at_utc"), utc=True, errors="coerce")
    completed = pd.to_datetime(execution.get("completed_at_utc"), utc=True, errors="coerce")
    if not (
        pd.notna(started)
        and pd.notna(completed)
        and started <= completed <= pd.Timestamp(_as_utc(now))
    ):
        blockers.append("canary_execution_timeline_invalid")

    entry_fills = execution.get("entry_fills")
    exit_fills = execution.get("exit_fills")
    approved_legs = approval.get("legs")
    if not (
        isinstance(entry_fills, list)
        and isinstance(exit_fills, list)
        and isinstance(approved_legs, list)
        and len(entry_fills) == len(exit_fills) == len(approved_legs) == 2
    ):
        blockers.append("canary_execution_requires_exact_two_leg_entry_and_exit")
        return sorted(set(blockers))
    fill_blockers, economics = _validate_fills(
        entry_fills=entry_fills,
        exit_fills=exit_fills,
        approved_legs=approved_legs,
        started=started,
        completed=completed,
    )
    blockers.extend(fill_blockers)
    for field, expected in economics.items():
        actual = _number(execution.get(field))
        if actual is None or not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8):
            blockers.append(f"canary_execution_{field}_reconciliation_mismatch")

    assets = {
        str(candidate.get("asset_x", "")).strip().upper(),
        str(candidate.get("asset_y", "")).strip().upper(),
    } - {""}
    positions = execution.get("final_positions")
    flat = bool(
        isinstance(positions, dict)
        and {str(key).upper() for key in positions} == assets
        and all(_is_zero(value) for value in positions.values())
    )
    if not (
        flat
        and execution.get("reconciled_flat") is True
        and execution.get("open_order_ids") == []
        and execution.get("unresolved_incidents") == []
    ):
        blockers.append("canary_execution_final_flat_reconciliation_failed")
    blockers.extend(
        _validate_use_ledger(
            root / "data" / "live" / "canary_authorization_use_ledger.jsonl",
            approval_id=str(approval.get("approval_id", "")),
            execution_id=expected_id,
            authorization_sha256=_file_hash(authorization_path),
        )
    )
    return sorted(set(blockers))


def _validate_fills(
    *,
    entry_fills: list[Any],
    exit_fills: list[Any],
    approved_legs: list[Any],
    started: pd.Timestamp,
    completed: pd.Timestamp,
) -> tuple[list[str], dict[str, float]]:
    blockers: list[str] = []
    approved = {
        _market(leg): leg for leg in approved_legs if isinstance(leg, dict) and _market(leg)
    }
    entries = {_market(fill): fill for fill in entry_fills if isinstance(fill, dict)}
    exits = {_market(fill): fill for fill in exit_fills if isinstance(fill, dict)}
    if len(approved) != 2 or set(entries) != set(approved) or set(exits) != set(approved):
        blockers.append("canary_execution_fill_markets_do_not_match_approval")
    fill_ids: list[str] = []
    order_ids: list[str] = []
    gross = 0.0
    fees = 0.0
    funding = 0.0
    for market, leg in approved.items():
        entry = entries.get(market, {})
        exit_fill = exits.get(market, {})
        approved_side = str(leg.get("side", "")).upper()
        entry_side = str(entry.get("side", "")).upper()
        exit_side = str(exit_fill.get("side", "")).upper()
        size = _number(entry.get("size"))
        exit_size = _number(exit_fill.get("size"))
        approved_size = _number(leg.get("size"))
        entry_price = _number(entry.get("price"))
        exit_price = _number(exit_fill.get("price"))
        if not (
            approved_side in {"BUY", "SELL"}
            and entry_side == approved_side
            and exit_side == ("SELL" if approved_side == "BUY" else "BUY")
            and size is not None
            and exit_size is not None
            and approved_size is not None
            and math.isclose(size, approved_size, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(exit_size, size, rel_tol=0.0, abs_tol=1e-12)
            and entry_price is not None
            and exit_price is not None
            and size > 0.0
            and entry_price > 0.0
            and exit_price > 0.0
        ):
            blockers.append(f"canary_execution_fill_contract_invalid:{market}")
            continue
        entry_time = pd.to_datetime(entry.get("timestamp_utc"), utc=True, errors="coerce")
        exit_time = pd.to_datetime(exit_fill.get("timestamp_utc"), utc=True, errors="coerce")
        if not (
            pd.notna(entry_time)
            and pd.notna(exit_time)
            and started <= entry_time < exit_time <= completed
        ):
            blockers.append(f"canary_execution_fill_timeline_invalid:{market}")
        for fill in (entry, exit_fill):
            fill_ids.append(str(fill.get("fill_id", "")))
            order_ids.append(str(fill.get("exchange_order_id", "")))
            fee = _number(fill.get("fee_usd"))
            funding_pnl = _number(fill.get("funding_pnl_usd"))
            if fee is None or fee < 0.0 or funding_pnl is None:
                blockers.append(f"canary_execution_fill_cost_invalid:{market}")
            else:
                fees += fee
                funding += funding_pnl
        gross += (
            (exit_price - entry_price) * size
            if approved_side == "BUY"
            else (entry_price - exit_price) * size
        )
    if (
        any(not value for value in fill_ids + order_ids)
        or len(set(fill_ids)) != 4
        or len(set(order_ids)) != 4
    ):
        blockers.append("canary_execution_fill_or_order_identity_invalid")
    return blockers, {
        "gross_pnl_usd": gross,
        "fees_usd": fees,
        "funding_pnl_usd": funding,
        "net_pnl_usd": gross - fees + funding,
    }


def _validate_use_ledger(
    path: Path, *, approval_id: str, execution_id: str, authorization_sha256: str
) -> list[str]:
    if not path.is_file():
        return ["canary_execution_one_use_ledger_missing"]
    matches = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if isinstance(row, dict) and row.get("approval_id") == approval_id:
                matches.append(row)
    except (OSError, json.JSONDecodeError):
        return ["canary_execution_one_use_ledger_invalid"]
    if len(matches) != 1:
        return ["canary_execution_one_use_ledger_count_invalid"]
    row = matches[0]
    if (
        row.get("execution_id") != execution_id
        or row.get("authorization_receipt_sha256") != authorization_sha256
    ):
        return ["canary_execution_one_use_ledger_binding_mismatch"]
    return []


def _validate_review(
    *,
    root: Path,
    review: dict[str, Any],
    execution: dict[str, Any],
    execution_path: Path,
) -> list[str]:
    if not review:
        return ["post_canary_supreme_team_review_missing"]
    blockers: list[str] = []
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        blockers.append("post_canary_review_schema_invalid")
    identity = _identity(review, "review")
    expected_id = "livecanaryreview_" + identity[:20]
    if review.get("review_id") != expected_id or review.get("receipt_identity_sha256") != identity:
        blockers.append("post_canary_review_identity_invalid")
    immutable_path = root / "data" / "live" / "canary_reviews" / f"{expected_id}.json"
    if review.get("immutable_review_path") != _relative(immutable_path, root):
        blockers.append("post_canary_review_immutable_path_invalid")
    elif not immutable_path.is_file() or _read_json(immutable_path) != review:
        blockers.append("post_canary_review_immutable_receipt_missing_or_mismatched")
    if review.get("execution_id") != execution.get("execution_id") or review.get(
        "execution_receipt_sha256"
    ) != _file_hash(execution_path):
        blockers.append("post_canary_review_execution_binding_mismatch")
    required_true = (
        "reconciliation_pass",
        "realized_cost_comparison_pass",
        "zero_unresolved_incidents",
        "new_explicit_authorization_required_for_any_next_order",
    )
    if review.get("review_status") != "PASS" or not all(
        review.get(field) is True for field in required_true
    ):
        blockers.append("post_canary_review_controls_not_passed")
    if any(
        review.get(field) is not False
        for field in (
            "repeat_authorized",
            "scaling_authorized",
            "leverage_authorized",
            "live_trading_authorized",
        )
    ):
        blockers.append("post_canary_review_claims_new_authority")
    return sorted(set(blockers))


def _write_outcome(
    path: Path,
    *,
    now: datetime,
    status: str,
    authorization_path: Path,
    execution: dict[str, Any],
    execution_blockers: list[str],
    review_blockers: list[str],
    review_path: Path | None = None,
) -> Path:
    passed = status == "PASS_ONE_CANARY_COMPLETE_NO_FURTHER_AUTHORITY"
    payload = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "generated_at_utc": _as_utc(now).isoformat(),
        "canary_status": status,
        "authorization_receipt_sha256": _file_hash(authorization_path),
        "execution_id": str(execution.get("execution_id", "")),
        "execution_receipt_path": (
            "reports/active/live_canary_execution_receipt.json" if execution else ""
        ),
        "execution_blockers": execution_blockers,
        "post_canary_review_path": _relative(review_path, path.parents[2]) if review_path else "",
        "post_canary_review_blockers": review_blockers,
        "orders_submitted": 4 if execution and not execution_blockers else 0,
        "fills_observed": 4 if execution and not execution_blockers else 0,
        "gross_pnl_usd": execution.get("gross_pnl_usd") if passed else None,
        "fees_usd": execution.get("fees_usd") if passed else None,
        "funding_pnl_usd": execution.get("funding_pnl_usd") if passed else None,
        "net_pnl_usd": execution.get("net_pnl_usd") if passed else None,
        "reconciled_flat": passed,
        "repeat_authorized": False,
        "scaling_authorized": False,
        "leverage_authorized": False,
        "new_explicit_authorization_required": True,
        "canary_execution_authority": False,
        "live_trading_authorized": False,
    }
    payload["receipt_sha256"] = _payload_hash(payload)
    _atomic_json(payload, path)
    return path


def _identity(payload: dict[str, Any], kind: str) -> str:
    excluded = {
        "receipt_identity_sha256",
        "execution_id" if kind == "execution" else "review_id",
        "immutable_execution_path" if kind == "execution" else "immutable_review_path",
    }
    core = {key: value for key, value in payload.items() if key not in excluded}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _policy_id(policy: dict[str, Any]) -> str:
    return "livecanarypolicy_" + _payload_hash(policy)[:20]


def _market(row: dict[str, Any]) -> str:
    return str(row.get("market", "")).strip().upper().removesuffix("-USD")


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_zero(value: Any) -> bool:
    number = _number(value)
    return number is not None and abs(number) <= 1e-12


def _payload_hash(payload: dict[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    return sha256(_canonical_json(core).encode("utf-8")).hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_root_path(root: Path, raw: str) -> Path | None:
    if not raw.strip():
        return None
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return path


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
