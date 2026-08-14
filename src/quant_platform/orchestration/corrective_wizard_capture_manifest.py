"""Fail-closed manifest for the next bounded Crypto Wizards proof capture."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _apply_source_orientation_contract,
    _artifact_paths,
    _build_capture_plan,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _identity as _copula_identity,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _read_csv as _read_copula_csv,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _read_json as _read_copula_json,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _text as _copula_text,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _truthy as _copula_truthy,
)
from quant_platform.orchestration.corrective_wizard_copula_behavioral import (
    _validate_contract as _validate_copula_contract,
)
from quant_platform.wizard_credit_budget import (
    COPULA_POST_CREDIT_COST,
    OU_V3_CUSTOM_SERIES_CREDIT_COST,
    OU_V4_CUSTOM_SERIES_CREDIT_COST,
    OU_V5_CUSTOM_SERIES_CREDIT_COST,
    OU_V6_CUSTOM_SERIES_CREDIT_COST,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    CUSTOM_SERIES_CREDIT_COST,
    _safe_filename,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_wizard_next_capture_manifest.v1"
IMMUTABLE_SCHEMA_VERSION = "thewiz.wizard_capture_cohort.v1"

ACTIVE_CSV = Path("reports/active/corrective_wizard_next_capture_manifest.csv")
ACTIVE_JSON = Path("reports/active/corrective_wizard_next_capture_manifest.json")
ACTIVE_MD = Path("reports/active/corrective_wizard_next_capture_manifest.md")
IMMUTABLE_DIR = Path("data/research/wizard_capture_manifests")
IMMUTABLE_SOURCE_DIR = Path("data/research/wizard_capture_manifest_sources")
SOURCE_RECEIPT_SCHEMA_VERSION = "thewiz.wizard_capture_source_receipt.v1"

EXACT_QUEUE = Path("reports/active/exhaustive_wizard_exact_mode_proof_queue.csv")
EXACT_PROOFS = Path("reports/active/hyperliquid_wizard_vendor_mode_proofs.csv")
INPUT_AUDIT = Path("reports/active/exhaustive_wizard_mode_proof_input_audit.csv")
CREDIT_BUDGET = Path("reports/active/wizard_credit_budget_contract.json")
OU_V3_CONTRACT = Path("config/wizard_ou_comparator_v3_holdout.json")
OU_V4_CONTRACT = Path("config/wizard_ou_comparator_v4_holdout.json")
OU_V5_CONTRACT = Path("config/wizard_ou_comparator_v5_holdout.json")
OU_V6_CONTRACT = Path("config/wizard_ou_comparator_v6_holdout.json")
SCHEDULER_STATUS = Path("reports/active/corrective_wizard_proof_scheduler_status.json")

REQUIRED_CANONICAL_QUEUE_COLUMNS = {
    "pair",
    "pair_group_id",
    "local_interval",
    "exact_mode",
    "orientation",
    "proof_observations",
    "vendor_custom_series_eligible",
}

CopulaPlanProvider = Callable[[Path], list[dict[str, Any]]]


def build_corrective_wizard_capture_manifest(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    queue_path: Path | None = None,
    proof_path: Path | None = None,
    input_audit_path: Path | None = None,
    credit_budget_path: Path | None = None,
    copula_plan_provider: CopulaPlanProvider | None = None,
) -> CommandResult:
    """Bind every pending external proof call before scheduler execution."""

    timestamp = _as_utc(now)
    queue_path = queue_path or root / EXACT_QUEUE
    proof_path = proof_path or root / EXACT_PROOFS
    input_audit_path = input_audit_path or root / INPUT_AUDIT
    credit_budget_path = credit_budget_path or root / CREDIT_BUDGET
    active_csv = root / ACTIVE_CSV
    active_json = root / ACTIVE_JSON
    active_md = root / ACTIVE_MD

    queue = _read_csv(queue_path)
    if not REQUIRED_CANONICAL_QUEUE_COLUMNS.issubset(queue.columns):
        summary = _noncanonical_fixture_summary(
            timestamp=timestamp,
            queue_path=queue_path,
            root=root,
        )
        _atomic_csv(pd.DataFrame(columns=_manifest_columns()), active_csv)
        _atomic_json(summary, active_json)
        _atomic_text(_markdown(summary, pd.DataFrame()), active_md)
        return CommandResult(
            paths={"manifest": active_csv, "status": active_json, "markdown": active_md},
            summary=summary,
        )

    blockers: list[str] = []
    source_artifacts: list[dict[str, str]] = []
    for path in (queue_path, proof_path, input_audit_path, credit_budget_path):
        _bind_source(path, root=root, output=source_artifacts, blockers=blockers)

    proofs = _read_csv(proof_path)
    audit = _read_csv(input_audit_path)
    budget = _read_json(credit_budget_path)
    if _text(budget.get("status")) != "PASS":
        blockers.append("wizard_credit_budget_contract_not_pass")

    rows: list[dict[str, Any]] = []
    exact_rows, exact_blockers = _pending_exact_rows(
        root=root,
        queue=queue,
        proofs=proofs,
        audit=audit,
    )
    rows.extend(exact_rows)
    blockers.extend(exact_blockers)

    ou_rows, ou_blockers, ou_sources = _pending_ou_rows(root=root)
    rows.extend(ou_rows)
    blockers.extend(ou_blockers)
    source_artifacts.extend(ou_sources)

    ou_v4_rows, ou_v4_blockers, ou_v4_sources = _pending_ou_v4_rows(root=root)
    rows.extend(ou_v4_rows)
    blockers.extend(ou_v4_blockers)
    source_artifacts.extend(ou_v4_sources)

    ou_v5_rows, ou_v5_blockers, ou_v5_sources = _pending_ou_v5_rows(root=root)
    rows.extend(ou_v5_rows)
    blockers.extend(ou_v5_blockers)
    source_artifacts.extend(ou_v5_sources)

    ou_v6_rows, ou_v6_blockers, ou_v6_sources = _pending_ou_v6_rows(root=root)
    rows.extend(ou_v6_rows)
    blockers.extend(ou_v6_blockers)
    source_artifacts.extend(ou_v6_sources)

    provider = copula_plan_provider or _current_copula_plans
    try:
        copula_plans = provider(root)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        copula_plans = []
        blockers.append(f"copula_capture_plan_invalid:{type(exc).__name__}:{exc}")
    copula_rows, copula_blockers = _pending_copula_rows(
        plans=copula_plans,
        exact_queue=queue,
    )
    rows.extend(copula_rows)
    blockers.extend(copula_blockers)

    frame = pd.DataFrame(rows, columns=_manifest_columns())
    blockers.extend(_manifest_integrity_blockers(frame))
    ou_v4_binding = validate_ou_v4_capture_manifest_contract(
        root=root,
        manifest={"calls": frame.fillna("").to_dict("records")},
        pending_only=True,
    )
    blockers.extend(ou_v4_binding["blockers"])
    ou_v5_binding = validate_ou_v5_capture_manifest_contract(
        root=root,
        manifest={"calls": frame.fillna("").to_dict("records")},
        pending_only=True,
    )
    blockers.extend(ou_v5_binding["blockers"])
    ou_v6_binding = validate_ou_v6_capture_manifest_contract(
        root=root,
        manifest={"calls": frame.fillna("").to_dict("records")},
        pending_only=True,
    )
    blockers.extend(ou_v6_binding["blockers"])
    lane_totals = _lane_totals(frame)
    pending_calls = len(frame)
    planned_credits = int(frame["credit_cost"].sum()) if not frame.empty else 0
    proof_lane_ceiling = sum(
        int(budget.get(field, 0) or 0)
        for field in (
            "exact_mode_proof_credit_ceiling",
            "copula_behavioral_credit_ceiling",
            "ou_v3_prospective_credit_ceiling",
            "ou_v4_prospective_credit_ceiling",
            "ou_v5_prospective_credit_ceiling",
            "ou_v6_prospective_credit_ceiling",
        )
    )
    if planned_credits > proof_lane_ceiling:
        blockers.append(
            f"manifest_credits_exceed_proof_lane_ceiling:{planned_credits}:{proof_lane_ceiling}"
        )

    blockers = sorted(set(filter(None, blockers)))
    next_eligible = _next_external_eligibility(root=root, timestamp=timestamp)
    capture_eligible_now = timestamp >= next_eligible
    manifest_core = {
        "schema_version": IMMUTABLE_SCHEMA_VERSION,
        "calls": frame.fillna("").to_dict("records"),
        "lane_totals": lane_totals,
        "pending_calls": pending_calls,
        "planned_credits": planned_credits,
        "proof_lane_credit_ceiling": proof_lane_ceiling,
        "intentional_cross_lane_overlap_policy": (
            "same_mode_cell_is_allowed_only_for_distinct_backtest_and_copula_endpoints"
        ),
        "runtime_credit_preflight_required": True,
        "order_submission_included": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    manifest_id = "wizardcapture_" + _json_hash(manifest_core)[:20]
    immutable_path = root / IMMUTABLE_DIR / f"{manifest_id}.json"
    manifest_preexisting = immutable_path.is_file()
    immutable_payload = {**manifest_core, "manifest_id": manifest_id}
    _write_or_validate_immutable_json(immutable_payload, immutable_path)
    current_source_artifacts = sorted(source_artifacts, key=lambda item: item["path"])
    source_receipt, source_receipt_path, source_receipt_created = _load_or_create_source_receipt(
        root=root,
        manifest_id=manifest_id,
        immutable_manifest_path=immutable_path,
        source_artifacts=current_source_artifacts,
        retrofit=manifest_preexisting,
    )
    source_binding = validate_capture_manifest_source_receipt(
        root=root,
        manifest_id=manifest_id,
        manifest_path=_relative(immutable_path, root),
        manifest_sha256=_file_hash(immutable_path),
    )
    if source_binding["status"] != "PASS":
        raise ValueError(
            "immutable capture source receipt invalid: " + ";".join(source_binding["blockers"])
        )
    frozen_source_artifacts = [
        {"path": item["path"], "sha256": item["sha256"]}
        for item in source_receipt["source_artifacts"]
    ]
    current_source_drift = classify_capture_manifest_source_drift(
        root=root,
        source_receipt=source_receipt,
    )
    if current_source_drift["scientific_or_structural_paths"]:
        blockers.append("current_capture_source_scientific_or_structural_drift")
    blockers = sorted(set(filter(None, blockers)))

    status = "PASS" if not blockers else "BLOCKED"
    capture_state = (
        "BLOCKED_PREREQUISITES"
        if blockers
        else "COMPLETE_NO_PENDING_CALLS"
        if pending_calls == 0
        else "READY_RUNTIME_CREDIT_PREFLIGHT_REQUIRED"
        if capture_eligible_now
        else "DEFERRED_UNTIL_UTC_RESET"
    )
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": timestamp.isoformat(),
        "status": status,
        "capture_state": capture_state,
        "manifest_enforced": True,
        "manifest_id": manifest_id,
        "immutable_manifest_path": _relative(immutable_path, root),
        "immutable_manifest_sha256": _file_hash(immutable_path),
        "pending_calls": pending_calls,
        "planned_credits": planned_credits,
        "proof_lane_credit_ceiling": proof_lane_ceiling,
        "credit_headroom_inside_proof_lane": proof_lane_ceiling - planned_credits,
        "lane_totals": lane_totals,
        "next_external_attempt_eligible_at": next_eligible.isoformat(),
        "capture_eligible_now": capture_eligible_now,
        "runtime_credit_preflight_required": True,
        "blockers": blockers,
        "source_artifacts": frozen_source_artifacts,
        "source_artifacts_sha256": source_receipt["source_artifacts_sha256"],
        "source_artifacts_current_sha256": current_source_drift["current_source_artifacts_sha256"],
        "source_artifacts_current_match": current_source_drift["current_match"],
        "source_artifacts_current_drift_classification": current_source_drift["classification"],
        "source_artifacts_current_metadata_only_paths": current_source_drift["metadata_only_paths"],
        "source_artifacts_current_scientific_or_structural_paths": (
            current_source_drift["scientific_or_structural_paths"]
        ),
        "source_receipt_id": source_receipt["receipt_id"],
        "source_receipt_path": _relative(source_receipt_path, root),
        "source_receipt_sha256": _file_hash(source_receipt_path),
        "source_receipt_binding_origin": source_receipt["binding_origin"],
        "source_receipt_created_this_run": source_receipt_created,
        "intentional_cross_lane_overlap_calls": int(
            frame.get("intentional_cross_lane_overlap", pd.Series(dtype=bool)).map(_truthy).sum()
        ),
        "order_submission_included": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(active_csv, root),
    }
    _atomic_csv(frame, active_csv)
    _atomic_json(summary, active_json)
    _atomic_text(_markdown(summary, frame), active_md)
    return CommandResult(
        paths={
            "manifest": active_csv,
            "status": active_json,
            "markdown": active_md,
            "immutable_manifest": immutable_path,
            "immutable_source_receipt": source_receipt_path,
        },
        summary=summary,
    )


def _pending_exact_rows(
    *, root: Path, queue: pd.DataFrame, proofs: pd.DataFrame, audit: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[str]]:
    eligible = queue.loc[queue["vendor_custom_series_eligible"].map(_truthy)].copy()
    captured = proofs.loc[
        proofs.get("vendor_response_captured", pd.Series(False, index=proofs.index)).map(_truthy)
        | proofs.get("mode_proof_status", pd.Series("", index=proofs.index)).eq("completed")
    ]
    captured_ids = {_identity(row) for _, row in captured.iterrows()}
    audit_by_identity = {_identity(row): row for _, row in audit.iterrows()}
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for _, candidate in eligible.iterrows():
        identity = _identity(candidate)
        if identity in captured_ids:
            continue
        audit_row = audit_by_identity.get(identity)
        pair = _text(candidate.get("pair"))
        exact_mode = _text(candidate.get("exact_mode"))
        orientation = _text(candidate.get("orientation"))
        observations = _observations(candidate)
        call_id = _call_id(
            "exact_mode_backtest",
            pair,
            exact_mode,
            orientation,
            observations,
            1,
        )
        blocker = ""
        payload_sha = ""
        if audit_row is None:
            blocker = "exact_mode_input_audit_binding_missing"
        elif _text(audit_row.get("input_audit_status")) != "READY":
            blocker = "exact_mode_input_audit_not_ready"
        else:
            payload_sha = _text(audit_row.get("request_fingerprint"))
            if len(payload_sha) != 64:
                blocker = "exact_mode_request_fingerprint_missing"
        if blocker:
            blockers.append(f"{call_id}:{blocker}")
        stem = _safe_filename(
            f"{pair}_{_text(candidate.get('local_interval'))}_{exact_mode}_{orientation}"
        )
        rows.append(
            _base_call(
                call_id=call_id,
                lane="exact_mode_backtest",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(candidate.get("pair_group_id")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=(
                    "data/raw/crypto_wizards_custom_series_proofs/"
                    f"<run_timestamp_utc>/{stem}_request.json"
                ),
                request_sha256=payload_sha,
                expected_output_path=(
                    "data/raw/crypto_wizards_custom_series_proofs/"
                    f"<run_timestamp_utc>/{stem}_response.json"
                ),
                credit_cost=CUSTOM_SERIES_CREDIT_COST,
                purpose="complete_vendor_backtest_response_ledger",
                prerequisite_status="PASS" if not blocker else "BLOCKED",
                blocker=blocker,
                evidence_path=_relative(root / INPUT_AUDIT, root),
                intentional_overlap=False,
                related_exact_status="PENDING",
            )
        )
    return rows, blockers


def _pending_ou_rows(*, root: Path) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    contract_path = root / OU_V3_CONTRACT
    if not contract_path.is_file():
        return [], [], []
    blockers: list[str] = []
    sources: list[dict[str, str]] = []
    _bind_source(contract_path, root=root, output=sources, blockers=blockers)
    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 4:
        return [], [*blockers, "ou_v3_contract_must_bind_exactly_four_cells"], sources
    rows: list[dict[str, Any]] = []
    for binding in bindings:
        response_path = root / _text(binding.get("response_path"))
        if response_path.is_file():
            continue
        request_path = root / _text(binding.get("request_path"))
        expected_hash = _text(binding.get("request_sha256"))
        blocker = ""
        if not request_path.is_file():
            blocker = "ou_v3_request_missing"
        elif _file_hash(request_path) != expected_hash:
            blocker = "ou_v3_request_hash_mismatch"
        if blocker:
            blockers.append(f"{_text(binding.get('pair'))}:{blocker}")
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        rows.append(
            _base_call(
                call_id=_call_id("ou_v3_holdout", pair, exact_mode, orientation, observations, 1),
                lane="ou_v3_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id="ou_v3_btc_eth_prospective",
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=expected_hash,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V3_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_trend_selector_holdout",
                prerequisite_status="PASS" if not blocker else "BLOCKED",
                blocker=blocker,
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )
    return rows, blockers, sources


def _pending_ou_v4_rows(
    *, root: Path
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    contract_path = root / OU_V4_CONTRACT
    if not contract_path.is_file():
        return [], [], []
    blockers: list[str] = []
    sources: list[dict[str, str]] = []
    _bind_source(contract_path, root=root, output=sources, blockers=blockers)
    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 8:
        return [], [*blockers, "ou_v4_contract_must_bind_exactly_eight_cells"], sources
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v4_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v4_contract_not_prospective")

    rows: list[dict[str, Any]] = []
    for binding in bindings:
        response_path = root / _text(binding.get("response_path"))
        if response_path.is_file():
            continue
        request_path = root / _text(binding.get("request_path"))
        expected_hash = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        call_id = _call_id("ou_v4_holdout", pair, exact_mode, orientation, observations, 1)
        blocker = ""
        if not request_path.is_file():
            blocker = "ou_v4_request_missing"
        elif len(expected_hash) != 64 or _file_hash(request_path) != expected_hash:
            blocker = "ou_v4_request_hash_mismatch"
        if blocker:
            blockers.append(f"{call_id}:{blocker}")
        rows.append(
            _base_call(
                call_id=call_id,
                lane="ou_v4_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=expected_hash,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V4_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v4_transform_trend_formula_holdout",
                prerequisite_status="PASS" if not blocker else "BLOCKED",
                blocker=blocker,
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )
    return rows, blockers, sources


def _pending_ou_v5_rows(
    *, root: Path
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    contract_path = root / OU_V5_CONTRACT
    if not contract_path.is_file():
        return [], [], []
    blockers: list[str] = []
    sources: list[dict[str, str]] = []
    _bind_source(contract_path, root=root, output=sources, blockers=blockers)
    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 8:
        return [], [*blockers, "ou_v5_contract_must_bind_exactly_eight_cells"], sources
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v5_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v5_contract_not_prospective")

    rows: list[dict[str, Any]] = []
    for binding in bindings:
        response_path = root / _text(binding.get("response_path"))
        if response_path.is_file():
            continue
        request_path = root / _text(binding.get("request_path"))
        expected_hash = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        call_id = _call_id("ou_v5_holdout", pair, exact_mode, orientation, observations, 1)
        blocker = ""
        if not request_path.is_file():
            blocker = "ou_v5_request_missing"
        elif len(expected_hash) != 64 or _file_hash(request_path) != expected_hash:
            blocker = "ou_v5_request_hash_mismatch"
        if blocker:
            blockers.append(f"{call_id}:{blocker}")
        rows.append(
            _base_call(
                call_id=call_id,
                lane="ou_v5_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=expected_hash,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V5_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v5_separate_transform_trend_profile_holdout",
                prerequisite_status="PASS" if not blocker else "BLOCKED",
                blocker=blocker,
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )
    return rows, blockers, sources


def _pending_ou_v6_rows(
    *, root: Path
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]:
    contract_path = root / OU_V6_CONTRACT
    if not contract_path.is_file():
        return [], [], []
    blockers: list[str] = []
    sources: list[dict[str, str]] = []
    _bind_source(contract_path, root=root, output=sources, blockers=blockers)
    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings", [])
    if not isinstance(bindings, list) or len(bindings) != 8:
        return [], [*blockers, "ou_v6_contract_must_bind_exactly_eight_cells"], sources
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v6_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v6_contract_not_prospective")
    if contract.get("final_successor_iteration") is not True:
        blockers.append("ou_v6_contract_not_final_successor")
    if contract.get("successor_after_v6_failure_allowed") is not False:
        blockers.append("ou_v6_contract_failure_not_terminal")

    rows: list[dict[str, Any]] = []
    for binding in bindings:
        response_path = root / _text(binding.get("response_path"))
        if response_path.is_file():
            continue
        request_path = root / _text(binding.get("request_path"))
        expected_hash = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        call_id = _call_id("ou_v6_holdout", pair, exact_mode, orientation, observations, 1)
        blocker = ""
        if not request_path.is_file():
            blocker = "ou_v6_request_missing"
        elif len(expected_hash) != 64 or _file_hash(request_path) != expected_hash:
            blocker = "ou_v6_request_hash_mismatch"
        if blocker:
            blockers.append(f"{call_id}:{blocker}")
        rows.append(
            _base_call(
                call_id=call_id,
                lane="ou_v6_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=expected_hash,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V6_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v6_final_disjoint_successor_holdout",
                prerequisite_status="PASS" if not blocker else "BLOCKED",
                blocker=blocker,
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )
    return rows, blockers, sources


def _pending_copula_rows(
    *, plans: list[dict[str, Any]], exact_queue: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[str]]:
    exact_states = {
        _overlap_identity(row): "REGISTERED_EXACT_MODE_CELL"
        for _, row in exact_queue.loc[exact_queue["exact_mode"].eq("Copula")].iterrows()
    }
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for plan in plans:
        base = plan.get("base", {})
        if plan.get("capture_state") == "COMPLETE":
            continue
        pair = _text(base.get("pair"))
        exact_mode = "Copula"
        orientation = _text(base.get("orientation"))
        observations = int(base.get("series_length", 0) or 0)
        blocker = _text(plan.get("blocker"))
        if plan.get("capture_state") == "BLOCKED" and not blocker:
            blocker = "copula_capture_plan_blocked"
        request_path = _text(base.get("request_path"))
        payload_sha = _text(plan.get("payload_sha256"))
        response_paths = [
            _text(base.get("response_1_path")),
            _text(base.get("response_2_path")),
        ]
        if not request_path or len(payload_sha) != 64 or not all(response_paths):
            blocker = blocker or "copula_capture_paths_or_payload_hash_missing"
        pair_group_id = _text(base.get("pair_group_id"))
        overlap_key = (
            pair_group_id.removesuffix(":copula_v2").lower(),
            orientation.lower(),
            observations,
        )
        related_status = exact_states.get(overlap_key, "NO_EXACT_MODE_CELL")
        for repetition, output_path in enumerate(response_paths, start=1):
            call_id = _call_id(
                "copula_behavioral",
                pair,
                exact_mode,
                orientation,
                observations,
                repetition,
            )
            if blocker:
                blockers.append(f"{call_id}:{blocker}")
            rows.append(
                _base_call(
                    call_id=call_id,
                    lane="copula_behavioral",
                    endpoint="POST /v1beta/copula",
                    pair=pair,
                    pair_group_id=pair_group_id,
                    exact_mode=exact_mode,
                    orientation=orientation,
                    observations=observations,
                    repetition=repetition,
                    request_path=request_path,
                    request_sha256=payload_sha,
                    expected_output_path=output_path,
                    credit_cost=COPULA_POST_CREDIT_COST,
                    purpose=(
                        "copula_repeatability_orientation_and_conditional_probability_symmetry"
                    ),
                    prerequisite_status="PASS" if not blocker else "BLOCKED",
                    blocker=blocker,
                    evidence_path=_text(base.get("source_backtest_request_path")),
                    intentional_overlap=related_status != "NO_EXACT_MODE_CELL",
                    related_exact_status=related_status,
                )
            )
    return rows, blockers


def _current_copula_plans(root: Path) -> list[dict[str, Any]]:
    paths = _artifact_paths(root, generation=2)
    if not paths["contract"].is_file():
        return []
    contract = _read_copula_json(paths["contract"])
    receipt = _read_copula_json(paths["contract_receipt"])
    _validate_copula_contract(
        root=root,
        contract=contract,
        contract_path=paths["contract"],
        receipt=receipt,
    )
    queue = _read_copula_csv(paths["queue"])
    proofs = _read_copula_csv(paths["proof"])
    expected = queue.loc[
        queue.get("exact_mode", pd.Series("", index=queue.index)).eq("Copula")
        & queue.get("vendor_custom_series_eligible", pd.Series(False, index=queue.index)).map(
            _copula_truthy
        )
    ]
    expected_ids = {_copula_identity(row) for _, row in expected.iterrows()}
    captured = proofs.loc[
        proofs.get("exact_mode", pd.Series("", index=proofs.index)).eq("Copula")
        & proofs.get("source_request_frozen", pd.Series(False, index=proofs.index)).map(
            _copula_truthy
        )
        & proofs.get("pair_group_id", pd.Series("", index=proofs.index)).map(_copula_text).ne("")
    ]
    captured = captured.loc[
        captured.apply(lambda row: _copula_identity(row) in expected_ids, axis=1)
    ]
    plans = [
        _build_capture_plan(
            root=root,
            proof=proof,
            contract_path=paths["contract"],
            contract_receipt_path=paths["contract_receipt"],
            queue_path=paths["queue"],
        )
        for _, proof in captured.iterrows()
    ]
    return _apply_source_orientation_contract(plans)


def _base_call(
    *,
    call_id: str,
    lane: str,
    endpoint: str,
    pair: str,
    pair_group_id: str,
    exact_mode: str,
    orientation: str,
    observations: int,
    repetition: int,
    request_path: str,
    request_sha256: str,
    expected_output_path: str,
    credit_cost: int,
    purpose: str,
    prerequisite_status: str,
    blocker: str,
    evidence_path: str,
    intentional_overlap: bool,
    related_exact_status: str,
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "lane": lane,
        "endpoint": endpoint,
        "pair": pair,
        "pair_group_id": pair_group_id,
        "exact_mode": exact_mode,
        "orientation": orientation,
        "observations": observations,
        "repetition": repetition,
        "request_path": request_path,
        "request_sha256": request_sha256,
        "expected_output_path": expected_output_path,
        "credit_cost": credit_cost,
        "purpose": purpose,
        "prerequisite_status": prerequisite_status,
        "blocker": blocker,
        "intentional_cross_lane_overlap": intentional_overlap,
        "related_exact_mode_status": related_exact_status,
        "runtime_credit_preflight_required": True,
        "order_submission_included": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": evidence_path,
    }


def _manifest_integrity_blockers(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    blockers: list[str] = []
    if frame["call_id"].duplicated().any():
        blockers.append("duplicate_manifest_call_id")
    if frame["expected_output_path"].duplicated().any():
        blockers.append("duplicate_manifest_output_path")
    if (pd.to_numeric(frame["credit_cost"], errors="coerce") <= 0).any():
        blockers.append("non_positive_manifest_credit_cost")
    if frame["prerequisite_status"].ne("PASS").any():
        blockers.append("one_or_more_manifest_prerequisites_blocked")
    return blockers


def _lane_totals(frame: pd.DataFrame) -> dict[str, dict[str, int]]:
    lanes = (
        "exact_mode_backtest",
        "ou_v3_holdout",
        "ou_v4_holdout",
        "ou_v5_holdout",
        "ou_v6_holdout",
        "copula_behavioral",
    )
    return {
        lane: {
            "calls": int(frame["lane"].eq(lane).sum()) if not frame.empty else 0,
            "credits": int(frame.loc[frame["lane"].eq(lane), "credit_cost"].sum())
            if not frame.empty
            else 0,
        }
        for lane in lanes
    }


def _next_external_eligibility(*, root: Path, timestamp: datetime) -> datetime:
    prior = _read_json(root / SCHEDULER_STATUS)
    attempted_today = bool(
        _truthy(prior.get("external_attempt_made"))
        and _text(prior.get("attempt_date_utc")) == timestamp.date().isoformat()
    )
    if not attempted_today:
        return timestamp
    return timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)


def _load_or_create_source_receipt(
    *,
    root: Path,
    manifest_id: str,
    immutable_manifest_path: Path,
    source_artifacts: list[dict[str, str]],
    retrofit: bool,
) -> tuple[dict[str, Any], Path, bool]:
    receipt_path = root / IMMUTABLE_SOURCE_DIR / f"{manifest_id}.json"
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        validation = validate_capture_manifest_source_receipt(
            root=root,
            manifest_id=manifest_id,
            manifest_path=_relative(immutable_manifest_path, root),
            manifest_sha256=_file_hash(immutable_manifest_path),
        )
        if validation["status"] != "PASS":
            raise ValueError(
                "existing immutable capture source receipt invalid: "
                + ";".join(validation["blockers"])
            )
        return receipt, receipt_path, False

    snapshot_root = root / IMMUTABLE_SOURCE_DIR / manifest_id / "artifacts"
    frozen_artifacts: list[dict[str, str]] = []
    root_resolved = root.resolve()
    for artifact in source_artifacts:
        relative = _text(artifact.get("path"))
        expected_sha256 = _text(artifact.get("sha256")).lower()
        source_path = root / relative
        try:
            source_safe = bool(
                relative
                and not Path(relative).is_absolute()
                and source_path.resolve().is_relative_to(root_resolved)
                and not source_path.is_symlink()
            )
        except (OSError, ValueError):
            source_safe = False
        if (
            not source_safe
            or not source_path.is_file()
            or len(expected_sha256) != 64
            or _file_hash(source_path) != expected_sha256
        ):
            raise ValueError(f"capture source artifact invalid: {relative or 'missing'}")
        snapshot_path = snapshot_root / relative
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = source_path.read_bytes()
        _write_or_validate_immutable_bytes(
            payload,
            snapshot_path,
            conflict_message=f"immutable capture source snapshot mismatch: {snapshot_path}",
        )
        frozen_artifacts.append(
            {
                "path": relative,
                "sha256": expected_sha256,
                "snapshot_path": _relative(snapshot_path, root),
            }
        )

    frozen_artifacts.sort(key=lambda item: item["path"])
    source_fingerprint = _source_artifact_fingerprint(frozen_artifacts)
    receipt_core = {
        "schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "immutable_manifest_path": _relative(immutable_manifest_path, root),
        "immutable_manifest_sha256": _file_hash(immutable_manifest_path),
        "source_artifacts": frozen_artifacts,
        "source_artifacts_sha256": source_fingerprint,
        "binding_origin": (
            "retrofit_before_external_execution" if retrofit else "created_with_manifest"
        ),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt = {
        **receipt_core,
        "receipt_id": "wizardcapturesources_" + _json_hash(receipt_core)[:20],
    }
    _write_or_validate_immutable_json(receipt, receipt_path)
    return receipt, receipt_path, True


def validate_capture_manifest_source_receipt(
    *,
    root: Path,
    manifest_id: object,
    manifest_path: object,
    manifest_sha256: object,
) -> dict[str, Any]:
    """Re-verify immutable source snapshots for one frozen capture cohort."""

    resolved_id = _text(manifest_id)
    resolved_manifest_path = _text(manifest_path)
    resolved_manifest_sha256 = _text(manifest_sha256).lower()
    receipt_path = root / IMMUTABLE_SOURCE_DIR / f"{resolved_id}.json"
    blockers: list[str] = []
    receipt = _read_json(receipt_path)
    expected_receipt_root = (root / IMMUTABLE_SOURCE_DIR).resolve()
    try:
        receipt_path_safe = bool(
            resolved_id.startswith("wizardcapture_")
            and receipt_path.is_file()
            and not receipt_path.is_symlink()
            and receipt_path.resolve().is_relative_to(expected_receipt_root)
        )
    except (OSError, ValueError):
        receipt_path_safe = False
    if not receipt_path_safe:
        blockers.append("capture_manifest_source_receipt_path_invalid")
    if not receipt:
        blockers.append("capture_manifest_source_receipt_missing_or_invalid")
    else:
        receipt_core = dict(receipt)
        receipt_id = _text(receipt_core.pop("receipt_id", ""))
        if (
            receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION
            or receipt_id != "wizardcapturesources_" + _json_hash(receipt_core)[:20]
            or receipt.get("manifest_id") != resolved_id
            or receipt.get("immutable_manifest_path") != resolved_manifest_path
            or receipt.get("immutable_manifest_sha256") != resolved_manifest_sha256
            or any(
                _truthy(receipt.get(field))
                for field in (
                    "candidate_promotion_authority",
                    "order_submission_included",
                    "testnet_order_authority",
                    "live_trading_authorized",
                )
            )
        ):
            blockers.append("capture_manifest_source_receipt_binding_invalid")

        artifacts = receipt.get("source_artifacts")
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        snapshot_root = (root / IMMUTABLE_SOURCE_DIR / resolved_id / "artifacts").resolve()
        if not isinstance(artifacts, list) or not artifacts:
            blockers.append("capture_manifest_source_receipt_artifacts_missing")
        else:
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    blockers.append("capture_manifest_source_receipt_artifact_invalid")
                    continue
                relative = _text(artifact.get("path"))
                expected_sha256 = _text(artifact.get("sha256")).lower()
                snapshot_relative = _text(artifact.get("snapshot_path"))
                snapshot_path = root / snapshot_relative
                try:
                    snapshot_safe = bool(
                        relative
                        and relative not in seen
                        and not Path(relative).is_absolute()
                        and snapshot_relative
                        and not Path(snapshot_relative).is_absolute()
                        and snapshot_path.is_file()
                        and not snapshot_path.is_symlink()
                        and snapshot_path.resolve().is_relative_to(snapshot_root)
                    )
                except (OSError, ValueError):
                    snapshot_safe = False
                if (
                    not snapshot_safe
                    or len(expected_sha256) != 64
                    or _file_hash(snapshot_path) != expected_sha256
                ):
                    blockers.append(
                        f"capture_manifest_source_snapshot_invalid:{relative or 'missing'}"
                    )
                    continue
                seen.add(relative)
                normalized.append(
                    {
                        "path": relative,
                        "sha256": expected_sha256,
                        "snapshot_path": snapshot_relative,
                    }
                )
        normalized.sort(key=lambda item: item["path"])
        fingerprint = _source_artifact_fingerprint(normalized)
        if receipt.get("source_artifacts_sha256") != fingerprint:
            blockers.append("capture_manifest_source_receipt_fingerprint_invalid")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "receipt_id": _text(receipt.get("receipt_id")),
        "receipt_path": _relative(receipt_path, root),
        "receipt_sha256": _file_hash(receipt_path) if receipt_path.is_file() else "",
        "source_artifacts": receipt.get("source_artifacts", []),
        "source_artifacts_sha256": _text(receipt.get("source_artifacts_sha256")),
    }


def classify_capture_manifest_source_drift(
    *, root: Path, source_receipt: dict[str, Any]
) -> dict[str, Any]:
    """Classify mutable-source drift without weakening frozen evidence lineage."""

    current_artifacts: list[dict[str, str]] = []
    metadata_only_paths: list[str] = []
    scientific_or_structural_paths: list[str] = []
    artifacts = source_receipt.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        scientific_or_structural_paths.append("source_receipt_artifacts_missing")
        artifacts = []

    root_resolved = root.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            scientific_or_structural_paths.append("source_receipt_artifact_invalid")
            continue
        relative = _text(artifact.get("path"))
        expected_sha256 = _text(artifact.get("sha256")).lower()
        snapshot_relative = _text(artifact.get("snapshot_path"))
        current_path = root / relative
        snapshot_path = root / snapshot_relative
        try:
            current_safe = bool(
                relative
                and not Path(relative).is_absolute()
                and current_path.is_file()
                and not current_path.is_symlink()
                and current_path.resolve().is_relative_to(root_resolved)
            )
        except (OSError, ValueError):
            current_safe = False
        current_sha256 = _file_hash(current_path) if current_safe else ""
        current_artifacts.append({"path": relative, "sha256": current_sha256})
        if not current_safe or len(expected_sha256) != 64:
            scientific_or_structural_paths.append(relative or "source_path_missing")
        elif current_sha256 == expected_sha256:
            continue
        elif _generated_metadata_only_json_drift(current_path, snapshot_path):
            metadata_only_paths.append(relative)
        else:
            scientific_or_structural_paths.append(relative)

    current_artifacts.sort(key=lambda item: item["path"])
    current_fingerprint = _source_artifact_fingerprint(current_artifacts)
    frozen_fingerprint = _text(source_receipt.get("source_artifacts_sha256"))
    current_match = bool(
        not scientific_or_structural_paths
        and not metadata_only_paths
        and current_fingerprint == frozen_fingerprint
    )
    classification = (
        "SCIENTIFIC_OR_STRUCTURAL_DRIFT"
        if scientific_or_structural_paths
        else "GENERATED_METADATA_ONLY"
        if metadata_only_paths
        else "EXACT_MATCH"
    )
    return {
        "classification": classification,
        "current_source_artifacts": current_artifacts,
        "current_source_artifacts_sha256": current_fingerprint,
        "current_match": current_match,
        "metadata_only_paths": sorted(set(metadata_only_paths)),
        "scientific_or_structural_paths": sorted(set(scientific_or_structural_paths)),
    }


def _generated_metadata_only_json_drift(current: Path, frozen: Path) -> bool:
    if current.suffix.lower() != ".json" or frozen.suffix.lower() != ".json":
        return False
    try:
        current_payload = json.loads(current.read_text(encoding="utf-8"))
        frozen_payload = json.loads(frozen.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(current_payload, dict) or not isinstance(frozen_payload, dict):
        return False
    if "generated_at_utc" not in current_payload or "generated_at_utc" not in frozen_payload:
        return False
    current_value = current_payload.pop("generated_at_utc")
    frozen_value = frozen_payload.pop("generated_at_utc")
    return current_value != frozen_value and current_payload == frozen_payload


def validate_ou_v4_capture_manifest_contract(
    *, root: Path, manifest: dict[str, Any], pending_only: bool = False
) -> dict[str, Any]:
    """Bind OU-v4 manifest calls to their preregistered contract cells.

    Immutable cohort validation uses the full contract. A newly built capture
    manifest contains only responses that are still pending.
    """

    blockers: list[str] = []
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        return {
            "status": "BLOCKED",
            "blockers": ["ou_v4_manifest_calls_invalid"],
            "expected_calls": 0,
            "observed_calls": 0,
        }
    observed = [
        row for row in calls if isinstance(row, dict) and row.get("lane") == "ou_v4_holdout"
    ]
    contract_path = root / OU_V4_CONTRACT
    if not contract_path.is_file():
        if observed:
            blockers.append("ou_v4_manifest_contract_missing")
        return {
            "status": "PASS_NOT_REQUIRED" if not blockers else "BLOCKED",
            "blockers": blockers,
            "expected_calls": 0,
            "observed_calls": len(observed),
        }

    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings")
    expected: list[dict[str, Any]] = []
    if not isinstance(bindings, list) or len(bindings) != 8:
        blockers.append("ou_v4_contract_must_bind_exactly_eight_cells")
        bindings = []
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v4_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v4_contract_not_prospective")

    for binding in bindings:
        if not isinstance(binding, dict):
            blockers.append("ou_v4_contract_binding_invalid")
            continue
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        request_sha = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        if (
            not request_path.is_file()
            or len(request_sha) != 64
            or _file_hash(request_path) != request_sha
        ):
            blockers.append("ou_v4_contract_request_binding_invalid")
        if pending_only and response_path.is_file():
            continue
        expected.append(
            _base_call(
                call_id=_call_id("ou_v4_holdout", pair, exact_mode, orientation, observations, 1),
                lane="ou_v4_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=request_sha,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V4_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v4_transform_trend_formula_holdout",
                prerequisite_status="PASS",
                blocker="",
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )

    expected_by_id = {_text(row.get("call_id")): row for row in expected}
    observed_by_id = {_text(row.get("call_id")): row for row in observed}
    if len(expected_by_id) != len(expected):
        blockers.append("ou_v4_contract_call_id_collision")
    if len(observed_by_id) != len(observed):
        blockers.append("ou_v4_manifest_call_id_collision")
    if set(observed_by_id) != set(expected_by_id):
        blockers.append("ou_v4_manifest_contract_call_set_mismatch")
    for call_id in sorted(set(observed_by_id) & set(expected_by_id)):
        if observed_by_id[call_id] != expected_by_id[call_id]:
            blockers.append(f"ou_v4_manifest_contract_call_mismatch:{call_id}")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "expected_calls": len(expected),
        "observed_calls": len(observed),
    }


def validate_ou_v5_capture_manifest_contract(
    *, root: Path, manifest: dict[str, Any], pending_only: bool = False
) -> dict[str, Any]:
    """Bind OU-v5 manifest calls to their preregistered contract cells."""

    blockers: list[str] = []
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        return {
            "status": "BLOCKED",
            "blockers": ["ou_v5_manifest_calls_invalid"],
            "expected_calls": 0,
            "observed_calls": 0,
        }
    observed = [
        row for row in calls if isinstance(row, dict) and row.get("lane") == "ou_v5_holdout"
    ]
    contract_path = root / OU_V5_CONTRACT
    if not contract_path.is_file():
        if observed:
            blockers.append("ou_v5_manifest_contract_missing")
        return {
            "status": "PASS_NOT_REQUIRED" if not blockers else "BLOCKED",
            "blockers": blockers,
            "expected_calls": 0,
            "observed_calls": len(observed),
        }

    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings")
    expected: list[dict[str, Any]] = []
    if not isinstance(bindings, list) or len(bindings) != 8:
        blockers.append("ou_v5_contract_must_bind_exactly_eight_cells")
        bindings = []
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v5_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v5_contract_not_prospective")

    for binding in bindings:
        if not isinstance(binding, dict):
            blockers.append("ou_v5_contract_binding_invalid")
            continue
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        request_sha = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        if (
            not request_path.is_file()
            or len(request_sha) != 64
            or _file_hash(request_path) != request_sha
        ):
            blockers.append("ou_v5_contract_request_binding_invalid")
        if pending_only and response_path.is_file():
            continue
        expected.append(
            _base_call(
                call_id=_call_id("ou_v5_holdout", pair, exact_mode, orientation, observations, 1),
                lane="ou_v5_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=request_sha,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V5_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v5_separate_transform_trend_profile_holdout",
                prerequisite_status="PASS",
                blocker="",
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )

    expected_by_id = {_text(row.get("call_id")): row for row in expected}
    observed_by_id = {_text(row.get("call_id")): row for row in observed}
    if len(expected_by_id) != len(expected):
        blockers.append("ou_v5_contract_call_id_collision")
    if len(observed_by_id) != len(observed):
        blockers.append("ou_v5_manifest_call_id_collision")
    if set(observed_by_id) != set(expected_by_id):
        blockers.append("ou_v5_manifest_contract_call_set_mismatch")
    for call_id in sorted(set(observed_by_id) & set(expected_by_id)):
        if observed_by_id[call_id] != expected_by_id[call_id]:
            blockers.append(f"ou_v5_manifest_contract_call_mismatch:{call_id}")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "expected_calls": len(expected),
        "observed_calls": len(observed),
    }


def validate_ou_v6_capture_manifest_contract(
    *, root: Path, manifest: dict[str, Any], pending_only: bool = False
) -> dict[str, Any]:
    """Bind OU-v6 manifest calls to the final preregistered contract cells."""

    blockers: list[str] = []
    calls = manifest.get("calls")
    if not isinstance(calls, list):
        return {
            "status": "BLOCKED",
            "blockers": ["ou_v6_manifest_calls_invalid"],
            "expected_calls": 0,
            "observed_calls": 0,
        }
    observed = [
        row for row in calls if isinstance(row, dict) and row.get("lane") == "ou_v6_holdout"
    ]
    contract_path = root / OU_V6_CONTRACT
    if not contract_path.is_file():
        if observed:
            blockers.append("ou_v6_manifest_contract_missing")
        return {
            "status": "PASS_NOT_REQUIRED" if not blockers else "BLOCKED",
            "blockers": blockers,
            "expected_calls": 0,
            "observed_calls": len(observed),
        }

    contract = _read_json(contract_path)
    bindings = contract.get("holdout_bindings")
    expected: list[dict[str, Any]] = []
    if not isinstance(bindings, list) or len(bindings) != 8:
        blockers.append("ou_v6_contract_must_bind_exactly_eight_cells")
        bindings = []
    if _text(contract.get("status")) != "PREREGISTERED_WAITING_VENDOR_RESPONSES":
        blockers.append("ou_v6_contract_not_preregistered")
    if int(contract.get("vendor_responses_at_registration", -1) or 0) != 0:
        blockers.append("ou_v6_contract_not_prospective")
    if contract.get("final_successor_iteration") is not True:
        blockers.append("ou_v6_contract_not_final_successor")
    if contract.get("successor_after_v6_failure_allowed") is not False:
        blockers.append("ou_v6_contract_failure_not_terminal")

    for binding in bindings:
        if not isinstance(binding, dict):
            blockers.append("ou_v6_contract_binding_invalid")
            continue
        request_path = root / _text(binding.get("request_path"))
        response_path = root / _text(binding.get("response_path"))
        request_sha = _text(binding.get("request_sha256"))
        pair = _text(binding.get("pair"))
        exact_mode = _text(binding.get("exact_mode"))
        orientation = _text(binding.get("orientation"))
        observations = int(binding.get("proof_observations", 0) or 0)
        if (
            not request_path.is_file()
            or len(request_sha) != 64
            or _file_hash(request_path) != request_sha
        ):
            blockers.append("ou_v6_contract_request_binding_invalid")
        if pending_only and response_path.is_file():
            continue
        expected.append(
            _base_call(
                call_id=_call_id("ou_v6_holdout", pair, exact_mode, orientation, observations, 1),
                lane="ou_v6_holdout",
                endpoint="POST /v1beta/backtest",
                pair=pair,
                pair_group_id=_text(binding.get("pair_group")),
                exact_mode=exact_mode,
                orientation=orientation,
                observations=observations,
                repetition=1,
                request_path=_relative(request_path, root),
                request_sha256=request_sha,
                expected_output_path=_relative(response_path, root),
                credit_cost=OU_V6_CUSTOM_SERIES_CREDIT_COST,
                purpose="prospective_ou_v6_final_disjoint_successor_holdout",
                prerequisite_status="PASS",
                blocker="",
                evidence_path=_relative(contract_path, root),
                intentional_overlap=False,
                related_exact_status="SEPARATE_PROSPECTIVE_COHORT",
            )
        )

    expected_by_id = {_text(row.get("call_id")): row for row in expected}
    observed_by_id = {_text(row.get("call_id")): row for row in observed}
    if len(expected_by_id) != len(expected):
        blockers.append("ou_v6_contract_call_id_collision")
    if len(observed_by_id) != len(observed):
        blockers.append("ou_v6_manifest_call_id_collision")
    if set(observed_by_id) != set(expected_by_id):
        blockers.append("ou_v6_manifest_contract_call_set_mismatch")
    for call_id in sorted(set(observed_by_id) & set(expected_by_id)):
        if observed_by_id[call_id] != expected_by_id[call_id]:
            blockers.append(f"ou_v6_manifest_contract_call_mismatch:{call_id}")

    blockers = sorted(set(blockers))
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
        "expected_calls": len(expected),
        "observed_calls": len(observed),
    }


def _source_artifact_fingerprint(artifacts: list[dict[str, str]]) -> str:
    normalized = [
        {"path": _text(item.get("path")), "sha256": _text(item.get("sha256")).lower()}
        for item in artifacts
    ]
    normalized.sort(key=lambda item: item["path"])
    return sha256(_json_bytes(normalized)).hexdigest()


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _bind_source(
    path: Path,
    *,
    root: Path,
    output: list[dict[str, str]],
    blockers: list[str],
) -> None:
    if not path.is_file():
        blockers.append(f"required_manifest_source_missing:{_relative(path, root)}")
        return
    output.append({"path": _relative(path, root), "sha256": _file_hash(path)})


def _identity(row: pd.Series) -> tuple[str, str, str, str, int]:
    group = _text(row.get("pair_group_id")) or _text(row.get("pair"))
    return (
        group.lower(),
        _text(row.get("local_interval")).lower(),
        _text(row.get("exact_mode")).lower(),
        _text(row.get("orientation")).lower(),
        _observations(row),
    )


def _overlap_identity(row: pd.Series) -> tuple[str, str, int]:
    return (
        _text(row.get("pair_group_id")).removesuffix(":copula_v2").lower(),
        _text(row.get("orientation")).lower(),
        _observations(row),
    )


def _observations(row: pd.Series) -> int:
    for field in (
        "proof_observations",
        "proof_observations_requested",
        "history_rows",
        "wizard_period",
    ):
        value = row.get(field)
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            continue
        if number > 0:
            return min(number, 360)
    return 0


def _call_id(
    lane: str,
    pair: str,
    exact_mode: str,
    orientation: str,
    observations: int,
    repetition: int,
) -> str:
    material = "|".join((lane, pair, exact_mode, orientation, str(observations), str(repetition)))
    return "wizcall_" + sha256(material.encode("utf-8")).hexdigest()[:20]


def _manifest_columns() -> list[str]:
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
        "request_path",
        "request_sha256",
        "expected_output_path",
        "credit_cost",
        "purpose",
        "prerequisite_status",
        "blocker",
        "intentional_cross_lane_overlap",
        "related_exact_mode_status",
        "runtime_credit_preflight_required",
        "order_submission_included",
        "research_only",
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
        "evidence_path",
    ]


def _noncanonical_fixture_summary(
    *, timestamp: datetime, queue_path: Path, root: Path
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": timestamp.isoformat(),
        "status": "SKIPPED_NONCANONICAL_FIXTURE",
        "capture_state": "NOT_ENFORCED",
        "manifest_enforced": False,
        "manifest_id": "",
        "pending_calls": 0,
        "planned_credits": 0,
        "proof_lane_credit_ceiling": 0,
        "lane_totals": _lane_totals(pd.DataFrame(columns=_manifest_columns())),
        "next_external_attempt_eligible_at": timestamp.isoformat(),
        "capture_eligible_now": True,
        "runtime_credit_preflight_required": True,
        "blockers": ["canonical_queue_columns_missing"],
        "order_submission_included": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "evidence_path": _relative(queue_path, root),
    }


def _markdown(summary: dict[str, Any], frame: pd.DataFrame) -> str:
    lines = [
        "# Corrective Wizard Next Capture Manifest",
        "",
        f"- Status: `{summary.get('status', '')}`",
        f"- Capture state: `{summary.get('capture_state', '')}`",
        f"- Pending calls: `{summary.get('pending_calls', 0)}`",
        f"- Planned credits: `{summary.get('planned_credits', 0)}`",
        f"- Next eligible UTC: `{summary.get('next_external_attempt_eligible_at', '')}`",
        f"- Current source drift: `{summary.get('source_artifacts_current_drift_classification', '')}`",
        "- Runtime credit preflight remains required.",
        "- No Testnet or live order authority is granted.",
        "",
    ]
    if frame.empty:
        lines.append("No pending canonical capture calls are listed.")
    else:
        lines.extend(
            [
                "| Lane | Pair | Mode | Orientation | Repeat | Endpoint | Credits | Purpose |",
                "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
            ]
        )
        for _, row in frame.iterrows():
            lines.append(
                "| {lane} | {pair} | {mode} | {orientation} | {repeat} | {endpoint} | {credits} | {purpose} |".format(
                    lane=row["lane"],
                    pair=row["pair"],
                    mode=row["exact_mode"],
                    orientation=row["orientation"],
                    repeat=row["repetition"],
                    endpoint=row["endpoint"],
                    credits=row["credit_cost"],
                    purpose=row["purpose"],
                )
            )
    blockers = summary.get("blockers", [])
    if blockers:
        lines.extend(["", "## Blockers", ""])
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


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
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _write_or_validate_immutable_bytes(
        encoded,
        path,
        conflict_message=f"immutable capture manifest mismatch: {path}",
    )


def _write_or_validate_immutable_bytes(
    payload: bytes,
    path: Path,
    *,
    conflict_message: str,
) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(conflict_message)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(conflict_message)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


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
    temporary.replace(path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


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
