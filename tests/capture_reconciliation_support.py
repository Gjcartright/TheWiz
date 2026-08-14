from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from quant_platform.orchestration.corrective_wizard_capture_manifest import (
    _load_or_create_source_receipt,
)


def write_valid_capture_reconciliation_evidence(root: Path) -> dict[str, object]:
    """Write a minimal immutable Stage 3 capture handoff for downstream tests."""

    manifest_core = {
        "schema_version": "thewiz.wizard_capture_cohort.v1",
        "calls": [{"call_id": "capture-call-1"}],
        "lane_totals": {},
        "pending_calls": 1,
        "planned_credits": 1,
        "proof_lane_credit_ceiling": 1,
        "intentional_cross_lane_overlap_policy": "test_fixture",
        "runtime_credit_preflight_required": True,
        "order_submission_included": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    manifest_id = "wizardcapture_" + _json_hash(manifest_core)[:20]
    manifest_path = (
        root
        / "data"
        / "research"
        / "wizard_capture_manifests"
        / f"{manifest_id}.json"
    )
    _write_json(manifest_path, {**manifest_core, "manifest_id": manifest_id})
    manifest_relative = str(manifest_path.relative_to(root))
    manifest_sha = _file_hash(manifest_path)
    source_path = root / "reports" / "active" / "capture_reconciliation_fixture_source.csv"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("call_id\ncapture-call-1\n", encoding="utf-8")
    source_receipt, source_receipt_path, _ = _load_or_create_source_receipt(
        root=root,
        manifest_id=manifest_id,
        immutable_manifest_path=manifest_path,
        source_artifacts=[
            {
                "path": str(source_path.relative_to(root)),
                "sha256": _file_hash(source_path),
            }
        ],
        retrofit=False,
    )

    reconciliation_core = {
        "schema_version": "thewiz.wizard_capture_reconciliation_receipt.v1",
        "manifest_id": manifest_id,
        "manifest_path": manifest_relative,
        "manifest_sha256": manifest_sha,
        "required_calls": 1,
        "completed_calls": 1,
        "lane_totals": {},
        "outcomes": [],
        "all_response_hashes_bound": True,
        "all_request_bindings_valid": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    reconciliation_id = "wizardcapturerecon_" + _json_hash(reconciliation_core)[:20]
    reconciliation_path = (
        root
        / "data"
        / "research"
        / "wizard_capture_reconciliations"
        / f"{reconciliation_id}.json"
    )
    _write_json(
        reconciliation_path,
        {**reconciliation_core, "reconciliation_id": reconciliation_id},
    )

    return {
        "capture_manifest_enforced": True,
        "capture_manifest_accounting_valid": True,
        "capture_manifest_id": manifest_id,
        "capture_manifest_immutable_path": manifest_relative,
        "capture_manifest_immutable_sha256": manifest_sha,
        "capture_manifest_candidate_id": manifest_id,
        "capture_manifest_candidate_immutable_path": manifest_relative,
        "capture_manifest_candidate_immutable_sha256": manifest_sha,
        "capture_manifest_candidate_binding_valid": True,
        "capture_manifest_candidate_source_binding_valid": True,
        "capture_manifest_source_receipt_id": source_receipt["receipt_id"],
        "capture_manifest_source_receipt_path": str(source_receipt_path.relative_to(root)),
        "capture_manifest_source_receipt_sha256": _file_hash(source_receipt_path),
        "capture_manifest_source_artifacts_sha256": source_receipt[
            "source_artifacts_sha256"
        ],
        "capture_manifest_continuity_valid": True,
        "capture_manifest_continuity_status": "PASS_NEW_FROZEN_COHORT",
        "capture_manifest_drift_detected": False,
        "capture_reconciliation_status": "PASS",
        "capture_reconciliation_valid": True,
        "capture_reconciliation_complete": True,
        "capture_reconciliation_id": reconciliation_id,
        "capture_reconciliation_immutable_path": str(
            reconciliation_path.relative_to(root)
        ),
        "capture_reconciliation_immutable_sha256": _file_hash(
            reconciliation_path
        ),
        "capture_reconciliation_manifest_id": manifest_id,
        "capture_reconciliation_manifest_path": manifest_relative,
        "capture_reconciliation_manifest_sha256": manifest_sha,
        "capture_reconciliation_required_calls": 1,
        "capture_reconciliation_completed_calls": 1,
        "capture_reconciliation_pending_calls": 0,
        "capture_reconciliation_blocked_calls": 0,
    }


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
