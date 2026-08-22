"""Reconcile a frozen Wizard capture manifest against landed response evidence."""

from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_redaction import safe_exception_code
from quant_platform.orchestration.corrective_runtime import (
    promote_staged_file,
    write_immutable_json,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_wizard_capture_reconciliation.v1"
MANIFEST_SCHEMA_VERSION = "thewiz.wizard_capture_cohort.v1"

ACTIVE_DETAIL = Path("reports/active/corrective_wizard_capture_reconciliation.csv")
ACTIVE_STATUS = Path("reports/active/corrective_wizard_capture_reconciliation.json")
ACTIVE_MD = Path("reports/active/corrective_wizard_capture_reconciliation.md")
IMMUTABLE_DIR = Path("data/research/wizard_capture_reconciliations")
ACTIVE_MANIFEST_STATUS = Path("reports/active/corrective_wizard_next_capture_manifest.json")
EXACT_PROOFS = Path("reports/active/hyperliquid_wizard_vendor_mode_proofs.csv")


def reconcile_corrective_wizard_capture_manifest(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    manifest_path: Path | None = None,
    proof_path: Path | None = None,
) -> CommandResult:
    """Prove each frozen call produced intact, identity-bound evidence."""

    timestamp = _as_utc(now)
    manifest_path = manifest_path or _active_manifest_path(root)
    proof_path = proof_path or root / EXACT_PROOFS
    detail_path = root / ACTIVE_DETAIL
    status_path = root / ACTIVE_STATUS
    markdown_path = root / ACTIVE_MD

    blockers: list[str] = []
    manifest: dict[str, Any] = {}
    if manifest_path is None or not manifest_path.is_file():
        blockers.append("immutable_capture_manifest_missing")
    else:
        try:
            manifest = _read_json_object(manifest_path)
            _validate_manifest(manifest=manifest, manifest_path=manifest_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"immutable_capture_manifest_invalid:{safe_exception_code(exc)}")

    proofs = _read_csv(proof_path)
    rows: list[dict[str, Any]] = []
    for call in manifest.get("calls", []) if not blockers else []:
        if not isinstance(call, dict):
            blockers.append("manifest_call_is_not_an_object")
            continue
        row = _reconcile_call(root=root, call=call, proofs=proofs)
        rows.append(row)
        if row["outcome_status"] == "BLOCKED":
            blockers.append(f"{row['call_id']}:{row['blocker']}")

    frame = pd.DataFrame(rows, columns=_detail_columns())
    required_calls = int(manifest.get("pending_calls", len(rows)) or 0)
    completed_calls = int(frame.get("outcome_status", pd.Series(dtype=str)).eq("COMPLETE").sum())
    pending_calls = int(frame.get("outcome_status", pd.Series(dtype=str)).eq("PENDING").sum())
    blocked_calls = int(frame.get("outcome_status", pd.Series(dtype=str)).eq("BLOCKED").sum())
    ou_v4_rows = frame.loc[frame.get("lane", pd.Series(dtype=str)).eq("ou_v4_holdout")]
    ou_v4_call_intents_observed = int(
        pd.to_numeric(ou_v4_rows.get("call_intent_count", pd.Series(dtype=int)), errors="coerce")
        .fillna(0)
        .sum()
    )
    all_ou_v4_call_intents_bound = bool(
        ou_v4_rows.empty
        or (
            ou_v4_rows["outcome_status"].eq("COMPLETE").all()
            and ou_v4_rows["call_intent_binding_valid"].map(_truthy).all()
        )
    )
    ou_v5_rows = frame.loc[frame.get("lane", pd.Series(dtype=str)).eq("ou_v5_holdout")]
    ou_v5_call_intents_observed = int(
        pd.to_numeric(ou_v5_rows.get("call_intent_count", pd.Series(dtype=int)), errors="coerce")
        .fillna(0)
        .sum()
    )
    all_ou_v5_call_intents_bound = bool(
        ou_v5_rows.empty
        or (
            ou_v5_rows["outcome_status"].eq("COMPLETE").all()
            and ou_v5_rows["call_intent_binding_valid"].map(_truthy).all()
        )
    )
    ou_v5_completion_receipts_observed = int(
        ou_v5_rows.get("call_completion_binding_valid", pd.Series(False, index=ou_v5_rows.index))
        .map(_truthy)
        .sum()
    )
    all_ou_v5_call_completions_bound = bool(
        ou_v5_rows.empty
        or (
            ou_v5_rows["outcome_status"].eq("COMPLETE").all()
            and ou_v5_rows["call_completion_binding_valid"].map(_truthy).all()
        )
    )
    ou_v6_rows = frame.loc[frame.get("lane", pd.Series(dtype=str)).eq("ou_v6_holdout")]
    ou_v6_call_intents_observed = int(
        pd.to_numeric(ou_v6_rows.get("call_intent_count", pd.Series(dtype=int)), errors="coerce")
        .fillna(0)
        .sum()
    )
    all_ou_v6_call_intents_bound = bool(
        ou_v6_rows.empty
        or (
            ou_v6_rows["outcome_status"].eq("COMPLETE").all()
            and ou_v6_rows["call_intent_binding_valid"].map(_truthy).all()
        )
    )
    ou_v6_completion_receipts_observed = int(
        ou_v6_rows.get("call_completion_binding_valid", pd.Series(False, index=ou_v6_rows.index))
        .map(_truthy)
        .sum()
    )
    all_ou_v6_call_completions_bound = bool(
        ou_v6_rows.empty
        or (
            ou_v6_rows["outcome_status"].eq("COMPLETE").all()
            and ou_v6_rows["call_completion_binding_valid"].map(_truthy).all()
        )
    )
    if len(rows) != required_calls:
        blockers.append(f"manifest_call_count_mismatch:{len(rows)}:{required_calls}")
    lane_totals = _lane_totals(frame)
    manifest_lane_totals = manifest.get("lane_totals", {})
    for lane, totals in lane_totals.items():
        expected = manifest_lane_totals.get(lane, {})
        if int(totals["required_calls"]) != int(expected.get("calls", 0) or 0):
            blockers.append(f"manifest_lane_call_count_mismatch:{lane}")

    blockers = sorted(set(filter(None, blockers)))
    if blockers:
        status = "BLOCKED_EVIDENCE"
    elif required_calls == 0:
        status = "NOT_REQUIRED"
    elif completed_calls == required_calls:
        status = "PASS"
    else:
        status = "PENDING"

    immutable_path: Path | None = None
    immutable_sha = ""
    manifest_id = _text(manifest.get("manifest_id"))
    if status == "PASS":
        completion_core = {
            "schema_version": "thewiz.wizard_capture_reconciliation_receipt.v1",
            "manifest_id": manifest_id,
            "manifest_path": _relative(manifest_path, root),
            "manifest_sha256": _file_hash(manifest_path),
            "required_calls": required_calls,
            "completed_calls": completed_calls,
            "lane_totals": lane_totals,
            "outcomes": frame.fillna("").to_dict("records"),
            "all_response_hashes_bound": True,
            "all_request_bindings_valid": True,
            "all_ou_v4_call_intents_bound": all_ou_v4_call_intents_bound,
            "ou_v4_call_intents_observed": ou_v4_call_intents_observed,
            "all_ou_v5_call_intents_bound": all_ou_v5_call_intents_bound,
            "ou_v5_call_intents_observed": ou_v5_call_intents_observed,
            "all_ou_v5_call_completions_bound": all_ou_v5_call_completions_bound,
            "ou_v5_completion_receipts_observed": ou_v5_completion_receipts_observed,
            "all_ou_v6_call_intents_bound": all_ou_v6_call_intents_bound,
            "ou_v6_call_intents_observed": ou_v6_call_intents_observed,
            "all_ou_v6_call_completions_bound": all_ou_v6_call_completions_bound,
            "ou_v6_completion_receipts_observed": ou_v6_completion_receipts_observed,
            "research_only": True,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        }
        reconciliation_id = "wizardcapturerecon_" + _json_hash(completion_core)[:20]
        immutable_path = root / IMMUTABLE_DIR / f"{reconciliation_id}.json"
        immutable_payload = {
            **completion_core,
            "reconciliation_id": reconciliation_id,
        }
        _write_or_validate_immutable_json(immutable_payload, immutable_path)
        immutable_sha = _file_hash(immutable_path)
    else:
        reconciliation_id = ""

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "manifest_id": manifest_id,
        "manifest_path": _relative(manifest_path, root) if manifest_path else "",
        "manifest_sha256": (
            _file_hash(manifest_path)
            if manifest_path is not None and manifest_path.is_file()
            else ""
        ),
        "required_calls": required_calls,
        "completed_calls": completed_calls,
        "pending_calls": pending_calls,
        "blocked_calls": blocked_calls,
        "lane_totals": lane_totals,
        "all_response_hashes_bound": status == "PASS",
        "all_request_bindings_valid": status == "PASS",
        "all_ou_v4_call_intents_bound": (status == "PASS" and all_ou_v4_call_intents_bound),
        "ou_v4_call_intents_observed": ou_v4_call_intents_observed,
        "all_ou_v5_call_intents_bound": (status == "PASS" and all_ou_v5_call_intents_bound),
        "ou_v5_call_intents_observed": ou_v5_call_intents_observed,
        "all_ou_v5_call_completions_bound": (status == "PASS" and all_ou_v5_call_completions_bound),
        "ou_v5_completion_receipts_observed": ou_v5_completion_receipts_observed,
        "all_ou_v6_call_intents_bound": (status == "PASS" and all_ou_v6_call_intents_bound),
        "ou_v6_call_intents_observed": ou_v6_call_intents_observed,
        "all_ou_v6_call_completions_bound": (status == "PASS" and all_ou_v6_call_completions_bound),
        "ou_v6_completion_receipts_observed": ou_v6_completion_receipts_observed,
        "blockers": blockers,
        "reconciliation_id": reconciliation_id,
        "immutable_reconciliation_path": (
            _relative(immutable_path, root) if immutable_path else ""
        ),
        "immutable_reconciliation_sha256": immutable_sha,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(detail_path, root),
    }
    _atomic_csv(frame, detail_path)
    _atomic_json(summary, status_path)
    _atomic_text(_markdown(summary, frame), markdown_path)
    paths: dict[str, Path] = {
        "detail": detail_path,
        "status": status_path,
        "markdown": markdown_path,
    }
    if immutable_path is not None:
        paths["immutable_reconciliation"] = immutable_path
    return CommandResult(paths=paths, summary=summary)


def validate_capture_reconciliation_evidence(
    *, root: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    """Re-verify immutable capture evidence before a downstream research handoff."""

    blockers: list[str] = []
    try:
        required = int(evidence.get("capture_reconciliation_required_calls", -1))
        completed = int(evidence.get("capture_reconciliation_completed_calls", -1))
        pending = int(evidence.get("capture_reconciliation_pending_calls", -1))
        blocked = int(evidence.get("capture_reconciliation_blocked_calls", -1))
    except (TypeError, ValueError):
        required = completed = pending = blocked = -1
        blockers.append("capture_reconciliation_counts_invalid")

    reconciliation_id = _text(evidence.get("capture_reconciliation_id"))
    reconciliation_path = _resolve(root, evidence.get("capture_reconciliation_immutable_path"))
    reconciliation_sha = _text(evidence.get("capture_reconciliation_immutable_sha256"))
    manifest_id = _text(evidence.get("capture_reconciliation_manifest_id"))
    manifest_path = _resolve(root, evidence.get("capture_reconciliation_manifest_path"))
    manifest_sha = _text(evidence.get("capture_reconciliation_manifest_sha256"))

    if _text(evidence.get("capture_reconciliation_status")) != "PASS":
        blockers.append("capture_reconciliation_status_not_pass")
    if not _truthy(evidence.get("capture_manifest_accounting_valid")):
        blockers.append("capture_manifest_accounting_invalid")
    if not _truthy(evidence.get("capture_reconciliation_valid")):
        blockers.append("capture_reconciliation_not_valid")
    if not _truthy(evidence.get("capture_reconciliation_complete")):
        blockers.append("capture_reconciliation_not_complete")
    if required <= 0 or completed != required or pending != 0 or blocked != 0:
        blockers.append("capture_reconciliation_counts_incomplete")

    manifest_root = root / "data" / "research" / "wizard_capture_manifests"
    if (
        manifest_path is None
        or not manifest_path.is_file()
        or not _path_within(manifest_path, manifest_root)
        or manifest_path.stem != manifest_id
        or len(manifest_sha) != 64
        or _file_hash(manifest_path) != manifest_sha
    ):
        blockers.append("capture_manifest_immutable_binding_invalid")
        manifest: dict[str, Any] = {}
    else:
        try:
            manifest = _read_json_object(manifest_path)
            _validate_manifest(manifest=manifest, manifest_path=manifest_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            manifest = {}
            blockers.append("capture_manifest_content_invalid")

    reconciliation_root = root / IMMUTABLE_DIR
    if (
        reconciliation_path is None
        or not reconciliation_path.is_file()
        or not _path_within(reconciliation_path, reconciliation_root)
        or reconciliation_path.stem != reconciliation_id
        or len(reconciliation_sha) != 64
        or _file_hash(reconciliation_path) != reconciliation_sha
    ):
        blockers.append("capture_reconciliation_immutable_binding_invalid")
        receipt: dict[str, Any] = {}
    else:
        try:
            receipt = _read_json_object(reconciliation_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            receipt = {}
            blockers.append("capture_reconciliation_content_invalid")

    if receipt:
        core = {key: value for key, value in receipt.items() if key != "reconciliation_id"}
        expected_id = "wizardcapturerecon_" + _json_hash(core)[:20]
        expected_manifest_relative = (
            _relative(manifest_path, root) if manifest_path is not None else ""
        )
        try:
            receipt_required = int(receipt.get("required_calls", -1))
            receipt_completed = int(receipt.get("completed_calls", -1))
        except (TypeError, ValueError):
            receipt_required = receipt_completed = -1
        ou_v4_intent_receipt_valid = _ou_v4_intent_receipt_valid(
            root=root,
            receipt=receipt,
        )
        ou_v5_intent_receipt_valid = _ou_v5_intent_receipt_valid(
            root=root,
            receipt=receipt,
        )
        ou_v5_completion_receipt_valid = _ou_v5_completion_receipt_valid(
            root=root,
            receipt=receipt,
        )
        ou_v6_intent_receipt_valid = _ou_v6_intent_receipt_valid(
            root=root,
            receipt=receipt,
        )
        ou_v6_completion_receipt_valid = _ou_v6_completion_receipt_valid(
            root=root,
            receipt=receipt,
        )
        if (
            _text(receipt.get("schema_version"))
            != "thewiz.wizard_capture_reconciliation_receipt.v1"
            or _text(receipt.get("reconciliation_id")) != reconciliation_id
            or reconciliation_id != expected_id
            or _text(receipt.get("manifest_id")) != manifest_id
            or _text(receipt.get("manifest_path")) != expected_manifest_relative
            or _text(receipt.get("manifest_sha256")) != manifest_sha
            or receipt_required != required
            or receipt_completed != completed
            or not _truthy(receipt.get("all_response_hashes_bound"))
            or not _truthy(receipt.get("all_request_bindings_valid"))
            or not ou_v4_intent_receipt_valid
            or not ou_v5_intent_receipt_valid
            or not ou_v5_completion_receipt_valid
            or not ou_v6_intent_receipt_valid
            or not ou_v6_completion_receipt_valid
            or not _truthy(receipt.get("research_only"))
            or any(
                _truthy(receipt.get(field))
                for field in (
                    "candidate_promotion_authority",
                    "testnet_order_authority",
                    "live_trading_authorized",
                )
            )
        ):
            blockers.append("capture_reconciliation_receipt_contract_invalid")

    if manifest and int(manifest.get("pending_calls", -1)) != required:
        blockers.append("capture_manifest_reconciliation_count_mismatch")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "manifest_id": manifest_id,
        "reconciliation_id": reconciliation_id,
        "required_calls": required,
        "completed_calls": completed,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _ou_v4_intent_receipt_valid(*, root: Path, receipt: dict[str, Any]) -> bool:
    return _ou_intent_receipt_valid(
        root=root,
        receipt=receipt,
        lane="ou_v4_holdout",
        intent_root=root / "data/research/wizard_ou_v4_holdout/call_attempts",
        all_bound_field="all_ou_v4_call_intents_bound",
        observed_field="ou_v4_call_intents_observed",
    )


def _ou_v5_intent_receipt_valid(*, root: Path, receipt: dict[str, Any]) -> bool:
    return _ou_intent_receipt_valid(
        root=root,
        receipt=receipt,
        lane="ou_v5_holdout",
        intent_root=root / "data/research/wizard_ou_v5_holdout/call_attempts",
        all_bound_field="all_ou_v5_call_intents_bound",
        observed_field="ou_v5_call_intents_observed",
    )


def _ou_v6_intent_receipt_valid(*, root: Path, receipt: dict[str, Any]) -> bool:
    return _ou_intent_receipt_valid(
        root=root,
        receipt=receipt,
        lane="ou_v6_holdout",
        intent_root=root / "data/research/wizard_ou_v6_holdout/call_attempts",
        all_bound_field="all_ou_v6_call_intents_bound",
        observed_field="ou_v6_call_intents_observed",
    )


def _ou_v5_completion_receipt_valid(*, root: Path, receipt: dict[str, Any]) -> bool:
    outcomes = receipt.get("outcomes")
    if not isinstance(outcomes, list):
        return False
    selected = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, dict) and _text(outcome.get("lane")) == "ou_v5_holdout"
    ]
    if not selected:
        return True
    if not _truthy(receipt.get("all_ou_v5_call_completions_bound")):
        return False
    observed = 0
    completion_root = root / "data/research/wizard_ou_v5_holdout/call_completions"
    for outcome in selected:
        contract_path, binding = _ou_v5_contract_binding(root=root, call=outcome)
        if contract_path is None or not binding:
            return False
        validation = _validate_ou_v5_call_completion(
            root=root,
            contract_path=contract_path,
            binding=binding,
        )
        completion_path = _resolve(root, outcome.get("call_completion_path"))
        if (
            _text(outcome.get("outcome_status")) != "COMPLETE"
            or not _truthy(outcome.get("call_completion_binding_valid"))
            or validation["status"] != "PASS"
            or validation["call_id"] != _text(outcome.get("call_id"))
            or completion_path is None
            or not completion_path.is_file()
            or completion_path.is_symlink()
            or not _path_within(completion_path, completion_root)
            or validation["completion_path"] != _text(outcome.get("call_completion_path"))
            or validation["completion_sha256"] != _text(outcome.get("call_completion_sha256"))
        ):
            return False
        observed += 1
    try:
        claimed = int(receipt.get("ou_v5_completion_receipts_observed", -1))
    except (TypeError, ValueError):
        return False
    return claimed == observed


def _ou_v6_completion_receipt_valid(*, root: Path, receipt: dict[str, Any]) -> bool:
    outcomes = receipt.get("outcomes")
    if not isinstance(outcomes, list):
        return False
    selected = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, dict) and _text(outcome.get("lane")) == "ou_v6_holdout"
    ]
    if not selected:
        return True
    if not _truthy(receipt.get("all_ou_v6_call_completions_bound")):
        return False
    observed = 0
    completion_root = root / "data/research/wizard_ou_v6_holdout/call_completions"
    for outcome in selected:
        contract_path, binding = _ou_v6_contract_binding(root=root, call=outcome)
        if contract_path is None or not binding:
            return False
        validation = _validate_ou_v6_call_completion(
            root=root,
            contract_path=contract_path,
            binding=binding,
        )
        completion_path = _resolve(root, outcome.get("call_completion_path"))
        if (
            _text(outcome.get("outcome_status")) != "COMPLETE"
            or not _truthy(outcome.get("call_completion_binding_valid"))
            or validation["status"] != "PASS"
            or validation["call_id"] != _text(outcome.get("call_id"))
            or completion_path is None
            or not completion_path.is_file()
            or completion_path.is_symlink()
            or not _path_within(completion_path, completion_root)
            or validation["completion_path"] != _text(outcome.get("call_completion_path"))
            or validation["completion_sha256"] != _text(outcome.get("call_completion_sha256"))
        ):
            return False
        observed += 1
    try:
        claimed = int(receipt.get("ou_v6_completion_receipts_observed", -1))
    except (TypeError, ValueError):
        return False
    return claimed == observed


def _ou_intent_receipt_valid(
    *,
    root: Path,
    receipt: dict[str, Any],
    lane: str,
    intent_root: Path,
    all_bound_field: str,
    observed_field: str,
) -> bool:
    outcomes = receipt.get("outcomes")
    if not isinstance(outcomes, list):
        return False
    ou_outcomes = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, dict) and _text(outcome.get("lane")) == lane
    ]
    if not ou_outcomes:
        return True
    if not _truthy(receipt.get(all_bound_field)):
        return False

    observed_count = 0
    for outcome in ou_outcomes:
        try:
            intent_count = int(outcome.get("call_intent_count", 0) or 0)
            ambiguous_attempts = int(outcome.get("ambiguous_prior_attempts", -1))
        except (TypeError, ValueError):
            return False
        paths = _split_evidence_values(outcome.get("call_intent_paths"))
        hashes = _split_evidence_values(outcome.get("call_intent_sha256s"))
        dates = _split_evidence_values(outcome.get("call_intent_dates"))
        if (
            _text(outcome.get("outcome_status")) != "COMPLETE"
            or not _truthy(outcome.get("call_intent_binding_valid"))
            or intent_count <= 0
            or len(paths) != intent_count
            or len(hashes) != intent_count
            or len(dates) != intent_count
            or ambiguous_attempts != intent_count - 1
            or _truthy(outcome.get("multi_day_recovery")) != (len(set(dates)) > 1)
        ):
            return False
        for relative, expected_hash, attempt_date in zip(paths, hashes, dates, strict=True):
            path = _resolve(root, relative)
            if (
                path is None
                or not path.is_file()
                or path.is_symlink()
                or not _path_within(path, intent_root)
                or path.parent.name != attempt_date
                or len(expected_hash) != 64
                or _file_hash(path) != expected_hash
            ):
                return False
        observed_count += intent_count
    try:
        claimed_count = int(receipt.get(observed_field, -1))
    except (TypeError, ValueError):
        return False
    return claimed_count == observed_count


def _split_evidence_values(value: object) -> list[str]:
    return [item.strip() for item in _text(value).split(";") if item.strip()]


def validate_capture_manifest_binding(
    *,
    root: Path,
    manifest_id: object,
    manifest_path: object,
    manifest_sha256: object,
    expected_pending_calls: int | None = None,
    expected_planned_credits: int | None = None,
    expected_lane_totals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-verify one immutable capture manifest before any external call."""

    blockers: list[str] = []
    resolved_id = _text(manifest_id)
    resolved_path = _resolve(root, manifest_path)
    resolved_sha = _text(manifest_sha256)
    manifest_root = root / "data" / "research" / "wizard_capture_manifests"
    manifest: dict[str, Any] = {}
    if (
        resolved_path is None
        or not resolved_path.is_file()
        or not _path_within(resolved_path, manifest_root)
        or resolved_path.stem != resolved_id
        or len(resolved_sha) != 64
        or _file_hash(resolved_path) != resolved_sha
    ):
        blockers.append("capture_manifest_immutable_binding_invalid")
    else:
        try:
            manifest = _read_json_object(resolved_path)
            _validate_manifest(manifest=manifest, manifest_path=resolved_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"capture_manifest_content_invalid:{safe_exception_code(exc)}")
            manifest = {}

    if manifest:
        if not _truthy(manifest.get("research_only")) or _truthy(
            manifest.get("order_submission_included")
        ):
            blockers.append("capture_manifest_authority_contract_invalid")
        if (
            expected_pending_calls is not None
            and int(manifest.get("pending_calls", -1)) != expected_pending_calls
        ):
            blockers.append("capture_manifest_pending_call_binding_mismatch")
        if (
            expected_planned_credits is not None
            and int(manifest.get("planned_credits", -1)) != expected_planned_credits
        ):
            blockers.append("capture_manifest_credit_binding_mismatch")
        if expected_lane_totals is not None and manifest.get("lane_totals") != expected_lane_totals:
            blockers.append("capture_manifest_lane_binding_mismatch")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "manifest_id": resolved_id,
        "manifest_path": _relative(resolved_path, root) if resolved_path else "",
        "manifest_sha256": resolved_sha,
        "pending_calls": int(manifest.get("pending_calls", 0) or 0),
        "planned_credits": int(manifest.get("planned_credits", 0) or 0),
        "lane_totals": manifest.get("lane_totals", {}),
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def frozen_capture_manifest_mutation_blockers(*, root: Path) -> list[str]:
    """Block proof-ledger mutations until the active frozen cohort is reconciled."""

    active_manifest_path = root / ACTIVE_MANIFEST_STATUS
    if not active_manifest_path.is_file():
        return []
    try:
        active_manifest = _read_json_object(active_manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ["frozen_capture_manifest_status_invalid"]
    if active_manifest.get("status") != "PASS":
        return []
    pending_calls = int(active_manifest.get("pending_calls", 0) or 0)
    if pending_calls <= 0:
        return []

    manifest_id = _text(active_manifest.get("manifest_id"))
    binding = validate_capture_manifest_binding(
        root=root,
        manifest_id=manifest_id,
        manifest_path=active_manifest.get("immutable_manifest_path"),
        manifest_sha256=active_manifest.get("immutable_manifest_sha256"),
        expected_pending_calls=pending_calls,
        expected_planned_credits=int(active_manifest.get("planned_credits", 0) or 0),
        expected_lane_totals=active_manifest.get("lane_totals", {}),
    )
    if binding["status"] != "PASS":
        return sorted(
            {
                "frozen_capture_manifest_binding_invalid",
                *[str(value) for value in binding.get("blockers", [])],
            }
        )

    reconciliation_path = root / ACTIVE_STATUS
    if not reconciliation_path.is_file():
        return [f"frozen_capture_manifest_unresolved:{manifest_id}"]
    try:
        reconciliation = _read_json_object(reconciliation_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return [
            f"frozen_capture_manifest_unresolved:{manifest_id}",
            "frozen_capture_reconciliation_status_invalid",
        ]
    complete = bool(
        reconciliation.get("status") == "PASS"
        and _text(reconciliation.get("manifest_id")) == manifest_id
    )
    evidence = {
        "capture_manifest_accounting_valid": binding["status"] == "PASS",
        "capture_reconciliation_status": reconciliation.get("status", ""),
        "capture_reconciliation_valid": complete,
        "capture_reconciliation_complete": complete,
        "capture_reconciliation_id": reconciliation.get("reconciliation_id", ""),
        "capture_reconciliation_immutable_path": reconciliation.get(
            "immutable_reconciliation_path", ""
        ),
        "capture_reconciliation_immutable_sha256": reconciliation.get(
            "immutable_reconciliation_sha256", ""
        ),
        "capture_reconciliation_manifest_id": reconciliation.get("manifest_id", ""),
        "capture_reconciliation_manifest_path": reconciliation.get("manifest_path", ""),
        "capture_reconciliation_manifest_sha256": reconciliation.get("manifest_sha256", ""),
        "capture_reconciliation_required_calls": reconciliation.get("required_calls", 0),
        "capture_reconciliation_completed_calls": reconciliation.get("completed_calls", 0),
        "capture_reconciliation_pending_calls": reconciliation.get("pending_calls", 0),
        "capture_reconciliation_blocked_calls": reconciliation.get("blocked_calls", 0),
    }
    validation = validate_capture_reconciliation_evidence(root=root, evidence=evidence)
    if validation["status"] == "PASS":
        return []
    return [f"frozen_capture_manifest_unresolved:{manifest_id}"]


def _reconcile_call(*, root: Path, call: dict[str, Any], proofs: pd.DataFrame) -> dict[str, Any]:
    lane = _text(call.get("lane"))
    if lane == "exact_mode_backtest":
        return _reconcile_exact_call(root=root, call=call, proofs=proofs)
    if lane in {
        "ou_v3_holdout",
        "ou_v4_holdout",
        "ou_v5_holdout",
        "ou_v6_holdout",
        "copula_behavioral",
    }:
        return _reconcile_deterministic_call(root=root, call=call)
    return _outcome(call, status="BLOCKED", blocker="unsupported_manifest_lane")


def _reconcile_exact_call(
    *, root: Path, call: dict[str, Any], proofs: pd.DataFrame
) -> dict[str, Any]:
    if proofs.empty:
        return _outcome(call, status="PENDING", blocker="exact_proof_ledger_empty")
    matching = proofs.loc[
        proofs.get("pair", pd.Series("", index=proofs.index)).map(_text).eq(_text(call.get("pair")))
        & proofs.get("exact_mode", pd.Series("", index=proofs.index))
        .map(_text)
        .eq(_text(call.get("exact_mode")))
        & proofs.get("orientation", pd.Series("", index=proofs.index))
        .map(_text)
        .eq(_text(call.get("orientation")))
        & proofs.apply(
            lambda row: _observations(row) == int(call.get("observations", 0) or 0),
            axis=1,
        )
    ]
    matching = matching.loc[
        matching.get("vendor_response_captured", pd.Series(False, index=matching.index)).map(
            _truthy
        )
        & matching.get("mode_proof_status", pd.Series("", index=matching.index)).eq("completed")
    ]
    if matching.empty:
        return _outcome(call, status="PENDING", blocker="exact_response_not_captured")
    if len(matching) != 1:
        return _outcome(call, status="BLOCKED", blocker="ambiguous_exact_proof_identity")
    proof = matching.iloc[0]
    request_path = _resolve(root, proof.get("request_path"))
    response_path = _resolve(root, proof.get("response_path"))
    return _validate_files(
        root=root,
        call=call,
        request_path=request_path,
        response_path=response_path,
        canonical_request_hash=True,
        evidence_path=_text(proof.get("evidence_path")),
    )


def _reconcile_deterministic_call(*, root: Path, call: dict[str, Any]) -> dict[str, Any]:
    request_path = _resolve(root, call.get("request_path"))
    response_path = _resolve(root, call.get("expected_output_path"))
    outcome = _validate_files(
        root=root,
        call=call,
        request_path=request_path,
        response_path=response_path,
        canonical_request_hash=False,
        evidence_path=_text(call.get("evidence_path")),
    )
    lane = _text(call.get("lane"))
    if lane not in {"ou_v4_holdout", "ou_v5_holdout", "ou_v6_holdout"}:
        return outcome
    outcome = _bind_ou_call_intents(root=root, call=call, outcome=outcome, lane=lane)
    if lane == "ou_v5_holdout":
        outcome = _bind_ou_v5_call_completion(root=root, call=call, outcome=outcome)
    elif lane == "ou_v6_holdout":
        outcome = _bind_ou_v6_call_completion(root=root, call=call, outcome=outcome)
    return outcome


def _bind_ou_v5_call_completion(
    *, root: Path, call: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    outcome.update(
        {
            "call_completion_path": "",
            "call_completion_sha256": "",
            "call_completion_binding_valid": False,
        }
    )
    if _text(outcome.get("outcome_status")) != "COMPLETE":
        return outcome
    contract_path, binding = _ou_v5_contract_binding(root=root, call=call)
    if contract_path is None or not binding:
        outcome["outcome_status"] = "BLOCKED"
        outcome["blocker"] = _join_blockers(
            outcome.get("blocker"), "ou_v5_contract_binding_missing_or_ambiguous"
        )
        outcome["request_binding_valid"] = False
        outcome["response_hash_bound"] = False
        return outcome
    validation = _validate_ou_v5_call_completion(
        root=root,
        contract_path=contract_path,
        binding=binding,
    )
    outcome.update(
        {
            "call_completion_path": validation["completion_path"],
            "call_completion_sha256": validation["completion_sha256"],
            "call_completion_binding_valid": validation["status"] == "PASS",
        }
    )
    if validation["status"] != "PASS" or validation["call_id"] != _text(call.get("call_id")):
        blockers = list(validation["blockers"])
        if validation["call_id"] != _text(call.get("call_id")):
            blockers.append("ou_v5_call_completion_call_id_mismatch")
        outcome["outcome_status"] = "BLOCKED"
        outcome["blocker"] = _join_blockers(outcome.get("blocker"), *blockers)
        outcome["request_binding_valid"] = False
        outcome["response_hash_bound"] = False
    return outcome


def _bind_ou_v6_call_completion(
    *, root: Path, call: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    outcome.update(
        {
            "call_completion_path": "",
            "call_completion_sha256": "",
            "call_completion_binding_valid": False,
        }
    )
    if _text(outcome.get("outcome_status")) != "COMPLETE":
        return outcome
    contract_path, binding = _ou_v6_contract_binding(root=root, call=call)
    if contract_path is None or not binding:
        outcome["outcome_status"] = "BLOCKED"
        outcome["blocker"] = _join_blockers(
            outcome.get("blocker"), "ou_v6_contract_binding_missing_or_ambiguous"
        )
        outcome["request_binding_valid"] = False
        outcome["response_hash_bound"] = False
        return outcome
    validation = _validate_ou_v6_call_completion(
        root=root,
        contract_path=contract_path,
        binding=binding,
    )
    outcome.update(
        {
            "call_completion_path": validation["completion_path"],
            "call_completion_sha256": validation["completion_sha256"],
            "call_completion_binding_valid": validation["status"] == "PASS",
        }
    )
    if validation["status"] != "PASS" or validation["call_id"] != _text(call.get("call_id")):
        blockers = list(validation["blockers"])
        if validation["call_id"] != _text(call.get("call_id")):
            blockers.append("ou_v6_call_completion_call_id_mismatch")
        outcome["outcome_status"] = "BLOCKED"
        outcome["blocker"] = _join_blockers(outcome.get("blocker"), *blockers)
        outcome["request_binding_valid"] = False
        outcome["response_hash_bound"] = False
    return outcome


def _ou_v5_contract_binding(
    *, root: Path, call: dict[str, Any]
) -> tuple[Path | None, dict[str, Any]]:
    contract_path = _resolve(root, call.get("evidence_path"))
    if contract_path is None or not contract_path.is_file():
        return None, {}
    try:
        contract = _read_json_object(contract_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, {}
    bindings = contract.get("holdout_bindings")
    if not isinstance(bindings, list):
        return None, {}
    matches = [
        binding
        for binding in bindings
        if isinstance(binding, dict)
        and _text(binding.get("pair")) == _text(call.get("pair"))
        and _text(binding.get("exact_mode")) == _text(call.get("exact_mode"))
        and _text(binding.get("orientation")) == _text(call.get("orientation"))
        and int(binding.get("proof_observations", 0) or 0) == int(call.get("observations", 0) or 0)
    ]
    return (contract_path, matches[0]) if len(matches) == 1 else (None, {})


def _ou_v6_contract_binding(
    *, root: Path, call: dict[str, Any]
) -> tuple[Path | None, dict[str, Any]]:
    return _ou_v5_contract_binding(root=root, call=call)


def _validate_ou_v5_call_completion(
    *, root: Path, contract_path: Path, binding: dict[str, Any]
) -> dict[str, Any]:
    # Imported lazily because the OU comparator stack validates frozen capture
    # reconciliation during its own module initialization path.
    from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
        validate_ou_v5_call_completion,
    )

    return validate_ou_v5_call_completion(
        root=root,
        contract_path=contract_path,
        binding=binding,
    )


def _validate_ou_v6_call_completion(
    *, root: Path, contract_path: Path, binding: dict[str, Any]
) -> dict[str, Any]:
    from quant_platform.orchestration.corrective_wizard_ou_v6_holdout import (
        validate_ou_v6_call_completion,
    )

    return validate_ou_v6_call_completion(
        root=root,
        contract_path=contract_path,
        binding=binding,
    )


def _join_blockers(*values: object) -> str:
    blockers: list[str] = []
    for value in values:
        blockers.extend(item for item in _text(value).split(";") if item)
    return ";".join(sorted(set(blockers)))


def _bind_ou_v4_call_intents(
    *, root: Path, call: dict[str, Any], outcome: dict[str, Any]
) -> dict[str, Any]:
    return _bind_ou_call_intents(
        root=root,
        call=call,
        outcome=outcome,
        lane="ou_v4_holdout",
    )


def _bind_ou_call_intents(
    *,
    root: Path,
    call: dict[str, Any],
    outcome: dict[str, Any],
    lane: str,
) -> dict[str, Any]:
    contracts = {
        "ou_v4_holdout": (
            root / "data/research/wizard_ou_v4_holdout/call_attempts",
            "thewiz.wizard_ou_v4_call_attempt_intent.v1",
            "ou_v4_response_has_no_call_attempt_intent",
        ),
        "ou_v5_holdout": (
            root / "data/research/wizard_ou_v5_holdout/call_attempts",
            "thewiz.wizard_ou_v5_call_attempt_intent.v1",
            "ou_v5_response_has_no_call_attempt_intent",
        ),
        "ou_v6_holdout": (
            root / "data/research/wizard_ou_v6_holdout/call_attempts",
            "thewiz.wizard_ou_v6_call_attempt_intent.v1",
            "ou_v6_response_has_no_call_attempt_intent",
        ),
    }
    if lane not in contracts:
        raise ValueError(f"unsupported OU intent lane: {lane}")
    intent_root, schema_version, missing_intent_blocker = contracts[lane]
    call_id = _text(call.get("call_id"))
    intent_paths = sorted(intent_root.glob(f"*/{call_id}.json"))
    intent_hashes: list[str] = []
    intent_dates: list[str] = []
    blockers: list[str] = []
    evidence_path = _resolve(root, call.get("evidence_path"))
    evidence_hash = _file_hash(evidence_path) if evidence_path and evidence_path.is_file() else ""
    for path in intent_paths:
        try:
            intent = _read_json_object(path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            blockers.append(f"call_intent_invalid:{type(exc).__name__}")
            continue
        attempt_date = path.parent.name
        try:
            registered_date = (
                _as_utc(datetime.fromisoformat(_text(intent.get("registered_at_utc"))))
                .date()
                .isoformat()
            )
        except (TypeError, ValueError):
            registered_date = ""
        valid = bool(
            not path.is_symlink()
            and _path_within(path, intent_root)
            and _text(intent.get("schema_version")) == schema_version
            and _text(intent.get("attempt_date_utc")) == attempt_date
            and registered_date == attempt_date
            and _text(intent.get("call_id")) == call_id
            and _text(intent.get("pair")) == _text(call.get("pair"))
            and _text(intent.get("exact_mode")) == _text(call.get("exact_mode"))
            and _text(intent.get("orientation")) == _text(call.get("orientation"))
            and int(intent.get("observations", 0) or 0) == int(call.get("observations", 0) or 0)
            and _text(intent.get("request_path")) == _text(call.get("request_path"))
            and _text(intent.get("request_sha256")) == _text(call.get("request_sha256"))
            and _text(intent.get("response_path")) == _text(call.get("expected_output_path"))
            and int(intent.get("credit_cost", 0) or 0) == int(call.get("credit_cost", 0) or 0)
            and _text(intent.get("contract_path")) == _text(call.get("evidence_path"))
            and len(evidence_hash) == 64
            and _text(intent.get("contract_sha256")) == evidence_hash
            and _truthy(intent.get("research_only"))
            and not any(
                _truthy(intent.get(field))
                for field in (
                    "candidate_promotion_authority",
                    "testnet_order_authority",
                    "live_trading_authorized",
                )
            )
        )
        if not valid:
            blockers.append("call_intent_binding_mismatch")
            continue
        intent_dates.append(attempt_date)
        intent_hashes.append(_file_hash(path))

    response_landed = bool(outcome.get("outcome_status") == "COMPLETE")
    if response_landed and not intent_paths:
        blockers.append(missing_intent_blocker)
    if response_landed and len(intent_hashes) != len(intent_paths):
        blockers.append("one_or_more_call_intents_unbound")
    intent_binding_valid = bool(
        intent_paths and len(intent_hashes) == len(intent_paths) and not blockers
    )
    outcome.update(
        {
            "call_intent_count": len(intent_paths),
            "call_intent_paths": ";".join(_relative(path, root) for path in intent_paths),
            "call_intent_sha256s": ";".join(intent_hashes),
            "call_intent_dates": ";".join(intent_dates),
            "call_intent_binding_valid": intent_binding_valid,
            "multi_day_recovery": len(set(intent_dates)) > 1,
            "ambiguous_prior_attempts": max(len(intent_paths) - 1, 0),
        }
    )
    if blockers:
        outcome["outcome_status"] = "BLOCKED"
        outcome["blocker"] = ";".join(sorted(set(blockers)))
        outcome["request_binding_valid"] = False
        outcome["response_hash_bound"] = False
    return outcome


def _validate_files(
    *,
    root: Path,
    call: dict[str, Any],
    request_path: Path | None,
    response_path: Path | None,
    canonical_request_hash: bool,
    evidence_path: str,
) -> dict[str, Any]:
    if request_path is None or not request_path.is_file():
        return _outcome(call, status="PENDING", blocker="request_artifact_missing")
    expected_request_hash = _text(call.get("request_sha256"))
    try:
        request_payload = _read_json_object(request_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome(
            call,
            status="BLOCKED",
            blocker=f"request_artifact_invalid:{type(exc).__name__}",
        )
    observed_request_hash = (
        _json_hash(request_payload) if canonical_request_hash else _file_hash(request_path)
    )
    if observed_request_hash != expected_request_hash:
        return _outcome(call, status="BLOCKED", blocker="request_hash_mismatch")
    if response_path is None or not response_path.is_file():
        return _outcome(
            call,
            status="PENDING",
            blocker="response_artifact_missing",
            observed_request_path=_relative(request_path, root),
            observed_request_sha256=observed_request_hash,
        )
    try:
        _read_json_object(response_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _outcome(
            call,
            status="BLOCKED",
            blocker=f"response_artifact_invalid:{type(exc).__name__}",
            observed_request_path=_relative(request_path, root),
            observed_request_sha256=observed_request_hash,
        )
    response_hash = _file_hash(response_path)
    return _outcome(
        call,
        status="COMPLETE",
        blocker="",
        observed_request_path=_relative(request_path, root),
        observed_request_sha256=observed_request_hash,
        observed_response_path=_relative(response_path, root),
        observed_response_sha256=response_hash,
        evidence_path=evidence_path,
    )


def _outcome(
    call: dict[str, Any],
    *,
    status: str,
    blocker: str,
    observed_request_path: str = "",
    observed_request_sha256: str = "",
    observed_response_path: str = "",
    observed_response_sha256: str = "",
    evidence_path: str = "",
) -> dict[str, Any]:
    return {
        "call_id": _text(call.get("call_id")),
        "lane": _text(call.get("lane")),
        "endpoint": _text(call.get("endpoint")),
        "pair": _text(call.get("pair")),
        "pair_group_id": _text(call.get("pair_group_id")),
        "exact_mode": _text(call.get("exact_mode")),
        "orientation": _text(call.get("orientation")),
        "observations": int(call.get("observations", 0) or 0),
        "repetition": int(call.get("repetition", 0) or 0),
        "credit_cost": int(call.get("credit_cost", 0) or 0),
        "outcome_status": status,
        "blocker": blocker,
        "observed_request_path": observed_request_path,
        "observed_request_sha256": observed_request_sha256,
        "observed_response_path": observed_response_path,
        "observed_response_sha256": observed_response_sha256,
        "request_binding_valid": status == "COMPLETE",
        "response_hash_bound": status == "COMPLETE",
        "call_intent_count": 0,
        "call_intent_paths": "",
        "call_intent_sha256s": "",
        "call_intent_dates": "",
        "call_intent_binding_valid": False,
        "call_completion_path": "",
        "call_completion_sha256": "",
        "call_completion_binding_valid": False,
        "multi_day_recovery": False,
        "ambiguous_prior_attempts": 0,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": evidence_path or _text(call.get("evidence_path")),
    }


def _validate_manifest(*, manifest: dict[str, Any], manifest_path: Path) -> None:
    if _text(manifest.get("schema_version")) != MANIFEST_SCHEMA_VERSION:
        raise ValueError("capture manifest schema mismatch")
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        raise TypeError("capture manifest calls must be a list")
    manifest_id = _text(manifest.get("manifest_id"))
    core = {key: value for key, value in manifest.items() if key != "manifest_id"}
    expected_id = "wizardcapture_" + _json_hash(core)[:20]
    if manifest_id != expected_id:
        raise ValueError("capture manifest content identity mismatch")
    if manifest_path.stem != manifest_id:
        raise ValueError("capture manifest filename identity mismatch")
    call_ids = [_text(call.get("call_id")) for call in calls if isinstance(call, dict)]
    if len(call_ids) != len(calls) or len(call_ids) != len(set(call_ids)):
        raise ValueError("capture manifest contains duplicate or invalid call IDs")
    if int(manifest.get("pending_calls", -1)) != len(calls):
        raise ValueError("capture manifest pending call count mismatch")
    if any(
        _truthy(manifest.get(field))
        for field in (
            "candidate_promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        raise ValueError("capture manifest must remain research-only")


def _active_manifest_path(root: Path) -> Path | None:
    status = _read_json_optional(root / ACTIVE_MANIFEST_STATUS)
    relative = _text(status.get("immutable_manifest_path"))
    return _resolve(root, relative)


def _lane_totals(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    lanes = (
        "exact_mode_backtest",
        "ou_v3_holdout",
        "ou_v4_holdout",
        "ou_v5_holdout",
        "ou_v6_holdout",
        "copula_behavioral",
    )
    totals: dict[str, dict[str, int]] = {}
    for lane in lanes:
        selected = frame.loc[frame.get("lane", pd.Series(dtype=str)).eq(lane)]
        totals[lane] = {
            "required_calls": len(selected),
            "completed_calls": int(selected["outcome_status"].eq("COMPLETE").sum())
            if not selected.empty
            else 0,
            "pending_calls": int(selected["outcome_status"].eq("PENDING").sum())
            if not selected.empty
            else 0,
            "blocked_calls": int(selected["outcome_status"].eq("BLOCKED").sum())
            if not selected.empty
            else 0,
            "completed_credits": int(
                selected.loc[selected["outcome_status"].eq("COMPLETE"), "credit_cost"].sum()
            )
            if not selected.empty
            else 0,
        }
    return totals


def _detail_columns() -> list[str]:
    return [
        "call_id",
        "lane",
        "endpoint",
        "pair",
        "pair_group_id",
        "exact_mode",
        "orientation",
        "observations",
        "repetition",
        "credit_cost",
        "outcome_status",
        "blocker",
        "observed_request_path",
        "observed_request_sha256",
        "observed_response_path",
        "observed_response_sha256",
        "request_binding_valid",
        "response_hash_bound",
        "call_intent_count",
        "call_intent_paths",
        "call_intent_sha256s",
        "call_intent_dates",
        "call_intent_binding_valid",
        "call_completion_path",
        "call_completion_sha256",
        "call_completion_binding_valid",
        "multi_day_recovery",
        "ambiguous_prior_attempts",
        "research_only",
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
        "evidence_path",
    ]


def _markdown(summary: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# Corrective Wizard Capture Reconciliation",
        "",
        f"- Status: `{summary.get('status', '')}`",
        f"- Manifest: `{summary.get('manifest_id', '')}`",
        f"- Completed calls: `{summary.get('completed_calls', 0)}/{summary.get('required_calls', 0)}`",
        f"- Pending calls: `{summary.get('pending_calls', 0)}`",
        f"- Blocked calls: `{summary.get('blocked_calls', 0)}`",
        "- No Testnet or live order authority is granted.",
        "",
    ]
    if not frame.empty:
        lines.extend(
            [
                "| Lane | Pair | Mode | Orientation | Repeat | Outcome | Blocker |",
                "| --- | --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for _, row in frame.iterrows():
            lines.append(
                f"| {row['lane']} | {row['pair']} | {row['exact_mode']} | "
                f"{row['orientation']} | {row['repetition']} | "
                f"{row['outcome_status']} | {row['blocker']} |"
            )
    blockers = summary.get("blockers", [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _observations(row: pd.Series) -> int:
    for field in ("proof_observations", "history_rows", "wizard_period"):
        try:
            value = int(float(row.get(field)))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return min(value, 360)
    return 0


def _resolve(root: Path, value: object) -> Path | None:
    text = _text(value)
    if not text or "<" in text or ">" in text:
        return None
    path = Path(text)
    return path if path.is_absolute() else root / path


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except (OSError, ValueError):
        return False
    return True


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must be an object: {path}")
    return value


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return _read_json_object(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    try:
        write_immutable_json(path, payload)
    except ValueError as exc:
        raise ValueError(f"immutable capture reconciliation mismatch: {path}") from exc


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    promote_staged_file(temporary, path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    promote_staged_file(temporary, path)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
