"""Reviewed, immutable activation for the Dynamic-v2 research comparator."""

from __future__ import annotations

from quant_platform.orchestration.corrective_runtime import promote_staged_file

from quant_platform.orchestration.corrective_runtime import atomic_write_bytes

import csv
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    frozen_capture_manifest_mutation_blockers,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "thewiz.wizard_dynamic_v2_activation.v2"


def build_reviewed_dynamic_v2_activation(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    apply: bool = False,
    reviewer: str = "",
    review_note: str = "",
    review_packet_id: str = "",
    now: datetime | None = None,
) -> CommandResult:
    """Plan or explicitly apply a research-only Dynamic-v2 comparator activation."""

    timestamp = _as_utc(now)
    existing = load_validated_dynamic_v2_activation(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
    )
    if existing is not None:
        status_path = root / "reports" / "active" / "wizard_dynamic_v2_activation_status.json"
        immutable_path = _resolve(root, existing.get("immutable_activation_path"))
        snapshot_path = _resolve(root, existing.get("source_proof_snapshot_path"))
        paths = {"activation_status": status_path}
        if immutable_path is not None:
            paths["immutable_activation"] = immutable_path
        if snapshot_path is not None:
            paths["proof_snapshot"] = snapshot_path
        return CommandResult(paths=paths, summary=existing)

    context, blockers = _activation_context(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
    )
    if apply:
        blockers.extend(frozen_capture_manifest_mutation_blockers(root=root))
    review_packet = build_dynamic_v2_review_packet(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        now=timestamp,
    )
    packet_id = str(review_packet.summary.get("review_packet_id", ""))
    if review_packet.summary.get("status") != "READY_FOR_EXPLICIT_REVIEW":
        blockers.append("dynamic_v2_review_packet_not_ready")
    else:
        context.update(
            {
                "review_packet_id": packet_id,
                "review_packet_path": str(
                    review_packet.summary.get("immutable_review_packet_path", "")
                ),
                "review_packet_sha256": str(
                    review_packet.summary.get("immutable_review_packet_sha256", "")
                ),
            }
        )
    if apply and not reviewer.strip():
        blockers.append("reviewer_required_for_dynamic_v2_apply")
    if apply and len(review_note.strip()) < 20:
        blockers.append("substantive_review_note_required_for_dynamic_v2_apply")
    if apply and not review_packet_id.strip():
        blockers.append("review_packet_id_required_for_dynamic_v2_apply")
    elif apply and review_packet_id.strip() != packet_id:
        blockers.append("review_packet_id_does_not_match_current_evidence")
    if apply:
        from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
            load_validated_dynamic_v2_supreme_review,
        )

        try:
            supreme_review = load_validated_dynamic_v2_supreme_review(
                root=root,
                expected_review_packet_id=packet_id,
                expected_review_packet_path=context.get("review_packet_path", ""),
                expected_review_packet_sha256=context.get("review_packet_sha256", ""),
            )
        except (OSError, TypeError, ValueError):
            supreme_review = None
            blockers.append("dynamic_v2_supreme_review_invalid_for_apply")
        if supreme_review is None and "dynamic_v2_supreme_review_invalid_for_apply" not in blockers:
            blockers.append("dynamic_v2_supreme_review_required_for_apply")
        elif supreme_review is not None:
            context.update(
                {
                    "supreme_review_id": str(supreme_review.get("review_id", "")).strip(),
                    "supreme_review_path": str(
                        supreme_review.get("immutable_review_path", "")
                    ).strip(),
                    "supreme_review_sha256": str(
                        supreme_review.get("immutable_review_sha256", "")
                    ).strip(),
                }
            )

    active = root / "reports" / "active"
    status_path = active / "wizard_dynamic_v2_activation_status.json"
    base = {
        "schema_version": SCHEMA_VERSION,
        "evaluated_at_utc": timestamp.isoformat(),
        "status": (
            "BLOCKED"
            if blockers
            else "READY_REQUIRES_EXPLICIT_APPLY"
            if not apply
            else "APPLIED_RESEARCH_COMPARATOR_ONLY"
        ),
        "apply_requested": apply,
        "reviewer": reviewer.strip() if apply else "",
        "review_note": review_note.strip() if apply else "",
        "review_packet_id_submitted": review_packet_id.strip() if apply else "",
        "blockers": list(dict.fromkeys(blockers)),
        **context,
        "comparator_generation": 2,
        "original_comparator_mutated": False,
        "raw_vendor_evidence_mutated": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if blockers or not apply:
        _atomic_json(base, status_path)
        return CommandResult(paths={"activation_status": status_path}, summary=base)

    proof_path = root / str(context["source_proof_path"])
    snapshot_hash = _file_hash(proof_path)
    snapshot_path = (
        root
        / "data"
        / "research"
        / "wizard_dynamic_comparator_activations"
        / "proof_snapshots"
        / f"{snapshot_hash}.csv"
    )
    _write_or_validate_immutable_bytes(proof_path.read_bytes(), snapshot_path)
    immutable_payload = {
        **base,
        "source_proof_snapshot_path": _relative(snapshot_path, root),
        "source_proof_snapshot_sha256": snapshot_hash,
    }
    stable = {
        key: value for key, value in immutable_payload.items() if key not in {"evaluated_at_utc"}
    }
    activation_id = (
        "dynamicv2activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    immutable_payload["activation_id"] = activation_id
    immutable_path = (
        root
        / "data"
        / "research"
        / "wizard_dynamic_comparator_activations"
        / f"{activation_id}.json"
    )
    _write_or_validate_immutable_json(immutable_payload, immutable_path)
    pointer = {
        **immutable_payload,
        "immutable_activation_path": _relative(immutable_path, root),
        "immutable_activation_sha256": _file_hash(immutable_path),
    }
    _atomic_json(pointer, status_path)
    return CommandResult(
        paths={
            "activation_status": status_path,
            "immutable_activation": immutable_path,
            "proof_snapshot": snapshot_path,
        },
        summary=pointer,
    )


def build_dynamic_v2_review_packet(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    now: datetime | None = None,
) -> CommandResult:
    """Build the evidence-bound packet required for explicit v2 review."""

    timestamp = _as_utc(now)
    context, context_blockers = _activation_context(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
    )
    active = root / "reports" / "active"
    holdout_status_path = active / "wizard_dynamic_v2_holdout_status.json"
    holdout_detail_path = active / "wizard_dynamic_v2_holdout_evaluation.csv"
    holdout = _read_json(holdout_status_path)
    contract = _read_json(root / "config" / "wizard_dynamic_comparator_v2_holdout.json")
    rows = _read_csv_rows(holdout_detail_path)
    canonical_cells = {
        (mode, orientation)
        for mode in ("Dyn (Spread)", "Dyn (ZScoreR)")
        for orientation in ("original", "reverse")
    }
    observed_cells = {
        (str(row.get("exact_mode", "")), str(row.get("orientation", ""))) for row in rows
    }
    blockers = list(context_blockers)
    if observed_cells != canonical_cells:
        blockers.append("dynamic_v2_review_packet_cell_identity_mismatch")
    cell_payloads: list[dict[str, Any]] = []
    raw_bindings_valid = True
    for row in sorted(
        rows,
        key=lambda value: (
            str(value.get("exact_mode", "")),
            str(value.get("orientation", "")),
        ),
    ):
        request_path = _resolve(root, row.get("request_path"))
        response_path = _resolve(root, row.get("response_path"))
        request_hash = str(row.get("request_sha256", "")).strip()
        response_hash = str(row.get("response_sha256", "")).strip()
        captured = _truthy(row.get("vendor_response_captured"))
        binding_valid = bool(
            captured
            and request_path is not None
            and response_path is not None
            and request_hash == _file_hash(request_path)
            and response_hash == _file_hash(response_path)
        )
        if captured and not binding_valid:
            raw_bindings_valid = False
            blockers.append("dynamic_v2_review_packet_raw_binding_mismatch")
        cell_payloads.append(
            {
                "exact_mode": str(row.get("exact_mode", "")),
                "orientation": str(row.get("orientation", "")),
                "cell_status": str(row.get("cell_status", "")),
                "vendor_response_captured": captured,
                "spread_max_abs_error": str(row.get("spread_max_abs_error", "")),
                "zscore_max_abs_error": str(row.get("zscore_max_abs_error", "")),
                "zscore_roll_max_abs_error": str(row.get("zscore_roll_max_abs_error", "")),
                "request_path": str(row.get("request_path", "")),
                "request_sha256": request_hash,
                "response_path": str(row.get("response_path", "")),
                "response_sha256": response_hash,
                "raw_binding_valid": binding_valid,
                "blocker": str(row.get("blocker", "")),
            }
        )
    all_cells_passed = bool(
        len(cell_payloads) == 4 and all(cell["cell_status"] == "PASS" for cell in cell_payloads)
    )
    all_cells_captured = bool(
        len(cell_payloads) == 4 and all(cell["vendor_response_captured"] for cell in cell_payloads)
    )
    raw_bindings_valid = bool(all_cells_captured and raw_bindings_valid)
    disjoint = bool(
        str(contract.get("derivation_cohort", {}).get("pair_group_id", ""))
        and str(contract.get("holdout_cohort", {}).get("pair_group_id", ""))
        and contract.get("derivation_cohort", {}).get("pair_group_id")
        != contract.get("holdout_cohort", {}).get("pair_group_id")
    )
    if not disjoint:
        blockers.append("dynamic_v2_review_packet_cohorts_not_disjoint")
    ready = bool(
        holdout.get("status") == "PASS"
        and int(holdout.get("passed_cells", 0) or 0) == 4
        and all_cells_passed
        and raw_bindings_valid
        and not blockers
    )
    if ready:
        status = "READY_FOR_EXPLICIT_REVIEW"
    elif holdout.get("status") == "PASS":
        status = "BLOCKED_EVIDENCE_BINDING"
    elif holdout.get("status") in {"FAIL", "INCOMPLETE"}:
        status = "BLOCKED_HOLDOUT_EVIDENCE"
    else:
        status = "WAITING_FOR_HOLDOUT"
    payload: dict[str, Any] = {
        "schema_version": "thewiz.wizard_dynamic_v2_review_packet.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "derivation_pair_group_id": str(
            contract.get("derivation_cohort", {}).get("pair_group_id", "")
        ),
        "holdout_pair_group_id": str(contract.get("holdout_cohort", {}).get("pair_group_id", "")),
        "cohorts_disjoint": disjoint,
        "required_cells": 4,
        "passed_cells": int(holdout.get("passed_cells", 0) or 0),
        "all_cells_captured": all_cells_captured,
        "all_cells_passed": all_cells_passed,
        "raw_bindings_valid": raw_bindings_valid,
        "cells": cell_payloads,
        **context,
        "holdout_status_path": _relative(holdout_status_path, root),
        "holdout_status_sha256": _file_hash(holdout_status_path)
        if holdout_status_path.is_file()
        else "",
        "holdout_detail_path": _relative(holdout_detail_path, root),
        "holdout_detail_sha256": _file_hash(holdout_detail_path)
        if holdout_detail_path.is_file()
        else "",
        "explicit_review_required": True,
        "automatic_activation": False,
        "original_comparator_mutated": False,
        "raw_vendor_evidence_mutated": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    status_path = active / "wizard_dynamic_v2_review_packet.json"
    markdown_path = active / "wizard_dynamic_v2_review_packet.md"
    paths: dict[str, Path] = {
        "review_packet": status_path,
        "review_packet_markdown": markdown_path,
    }
    if ready:
        stable = {
            key: value
            for key, value in payload.items()
            if key not in {"evaluated_at_utc", "holdout_status_sha256"}
        }
        review_packet_id = (
            "dynamicv2review_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
        )
        immutable_payload = {**stable, "review_packet_id": review_packet_id}
        immutable_path = (
            root
            / "data"
            / "research"
            / "wizard_dynamic_review_packets"
            / f"{review_packet_id}.json"
        )
        _write_or_validate_immutable_json(immutable_payload, immutable_path)
        payload.update(
            {
                "review_packet_id": review_packet_id,
                "immutable_review_packet_path": _relative(immutable_path, root),
                "immutable_review_packet_sha256": _file_hash(immutable_path),
            }
        )
        paths["immutable_review_packet"] = immutable_path
    _atomic_json(payload, status_path)
    _atomic_text(_review_packet_markdown(payload), markdown_path)
    return CommandResult(paths=paths, summary=payload)


def load_validated_dynamic_v2_activation(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
) -> dict[str, Any] | None:
    """Return the active generation-2 receipt only when every immutable binding holds."""

    status_path = root / "reports" / "active" / "wizard_dynamic_v2_activation_status.json"
    if not status_path.is_file():
        return None
    pointer = _read_json(status_path)
    if not pointer:
        raise ValueError("dynamic v2 activation status is unreadable")
    if pointer.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        immutable_dir = root / "data" / "research" / "wizard_dynamic_comparator_activations"
        if pointer.get("activation_id") or any(immutable_dir.glob("*.json")):
            raise ValueError("dynamic v2 immutable activation exists without active pointer")
        return None
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("dynamic v2 activation schema mismatch")
    if int(pointer.get("comparator_generation", 0) or 0) != 2:
        raise ValueError("dynamic v2 activation generation mismatch")
    if pointer.get("implementation_source_sha256") != implementation_source_sha256:
        raise ValueError("dynamic v2 active implementation source hash mismatch")
    if not pointer.get("reviewer") or len(str(pointer.get("review_note", "")).strip()) < 20:
        raise ValueError("dynamic v2 active review evidence missing")
    if not pointer.get("review_packet_id") or (
        pointer.get("review_packet_id_submitted") != pointer.get("review_packet_id")
    ):
        raise ValueError("dynamic v2 active review packet approval mismatch")
    if any(
        _truthy(pointer.get(key))
        for key in (
            "candidate_promotion_authority",
            "testnet_order_authority",
            "live_trading_authorized",
        )
    ):
        raise ValueError("dynamic v2 activation exceeded research authority")
    immutable_path = _resolve(root, pointer.get("immutable_activation_path"))
    if immutable_path is None:
        raise ValueError("dynamic v2 immutable activation missing")
    if _file_hash(immutable_path) != str(pointer.get("immutable_activation_sha256", "")):
        raise ValueError("dynamic v2 immutable activation hash mismatch")
    immutable = _read_json(immutable_path)
    for key, value in immutable.items():
        if pointer.get(key) != value:
            raise ValueError(f"dynamic v2 active pointer mismatch: {key}")
    activation_id = str(immutable.get("activation_id", "")).strip()
    stable = {
        key: value
        for key, value in immutable.items()
        if key not in {"activation_id", "evaluated_at_utc"}
    }
    expected_activation_id = (
        "dynamicv2activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    expected_activation_path = (
        root
        / "data"
        / "research"
        / "wizard_dynamic_comparator_activations"
        / f"{activation_id}.json"
    )
    if (
        activation_id != expected_activation_id
        or immutable_path.resolve() != expected_activation_path.resolve()
    ):
        raise ValueError("dynamic v2 immutable activation identity mismatch")
    from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
        validate_dynamic_v2_supreme_review_receipt,
    )

    supreme_review = validate_dynamic_v2_supreme_review_receipt(
        root=root,
        immutable_review_path=immutable.get("supreme_review_path"),
        immutable_review_sha256=immutable.get("supreme_review_sha256"),
        expected_review_packet_id=immutable.get("review_packet_id"),
        expected_review_packet_path=immutable.get("review_packet_path"),
        expected_review_packet_sha256=immutable.get("review_packet_sha256"),
    )
    if supreme_review.get("review_id") != immutable.get("supreme_review_id"):
        raise ValueError("dynamic v2 activation Supreme Team review ID mismatch")
    bindings = (
        ("dynamic_contract_path", "dynamic_contract_sha256"),
        ("dynamic_contract_receipt_path", "dynamic_contract_receipt_sha256"),
        ("holdout_result_path", "holdout_result_sha256"),
        ("base_comparator_contract_path", "base_comparator_contract_sha256"),
        ("base_comparator_receipt_path", "base_comparator_receipt_sha256"),
        ("review_packet_path", "review_packet_sha256"),
        ("supreme_review_path", "supreme_review_sha256"),
        ("source_proof_snapshot_path", "source_proof_snapshot_sha256"),
    )
    for path_field, hash_field in bindings:
        path = _resolve(root, immutable.get(path_field))
        if path is None or _file_hash(path) != str(immutable.get(hash_field, "")):
            raise ValueError(f"dynamic v2 activation binding mismatch: {path_field}")
    return pointer


def _activation_context(
    *, root: Path, implementation_source_sha256: str
) -> tuple[dict[str, Any], list[str]]:
    active = root / "reports" / "active"
    gate_path = active / "wizard_dynamic_v2_supersession_gate.json"
    holdout_status_path = active / "wizard_dynamic_v2_holdout_status.json"
    dynamic_contract_path = root / "config" / "wizard_dynamic_comparator_v2_holdout.json"
    dynamic_receipt_path = active / "wizard_dynamic_comparator_v2_holdout_receipt.json"
    comparator_path = active / "wizard_mode_comparator_contract.csv"
    comparator_receipt_path = active / "wizard_mode_comparator_contract_receipt.json"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    gate = _read_json(gate_path)
    holdout = _read_json(holdout_status_path)
    contract = _read_json(dynamic_contract_path)
    receipt = _read_json(dynamic_receipt_path)
    comparator_receipt = _read_json(comparator_receipt_path)
    blockers: list[str] = []
    if gate.get("status") != "READY_FOR_REVIEWED_SUPERSESSION":
        blockers.append("dynamic_v2_supersession_gate_not_ready")
    if holdout.get("status") != "PASS" or int(holdout.get("passed_cells", 0) or 0) != 4:
        blockers.append("dynamic_v2_disjoint_holdout_not_passed_four_of_four")
    holdout_result_path = _resolve(root, holdout.get("immutable_result_path"))
    holdout_result_hash = str(holdout.get("immutable_result_sha256", ""))
    if holdout_result_path is None or not holdout_result_hash:
        blockers.append("dynamic_v2_immutable_holdout_result_missing")
    elif _file_hash(holdout_result_path) != holdout_result_hash:
        blockers.append("dynamic_v2_immutable_holdout_result_hash_mismatch")
    if gate.get("source_holdout_result_sha256") != holdout_result_hash:
        blockers.append("dynamic_v2_gate_holdout_binding_mismatch")
    if not dynamic_contract_path.is_file() or not dynamic_receipt_path.is_file():
        blockers.append("dynamic_v2_contract_or_receipt_missing")
    elif receipt.get("contract_sha256") != _file_hash(dynamic_contract_path):
        blockers.append("dynamic_v2_contract_receipt_hash_mismatch")
    if contract.get("implementation", {}).get("source_sha256") != implementation_source_sha256:
        blockers.append("dynamic_v2_candidate_source_hash_mismatch")
    required_files = (comparator_path, comparator_receipt_path, proof_path)
    if not all(path.is_file() for path in required_files):
        blockers.append("dynamic_v2_base_comparator_or_proof_evidence_missing")
    forbidden = (
        "candidate_promotion_authority",
        "testnet_order_authority",
        "live_trading_authorized",
    )
    if any(
        _truthy(payload.get(key))
        for payload in (gate, holdout, contract, receipt)
        for key in forbidden
    ):
        blockers.append("dynamic_v2_source_evidence_exceeded_research_authority")
    context = {
        "implementation_source_sha256": implementation_source_sha256,
        "dynamic_contract_path": _relative(dynamic_contract_path, root),
        "dynamic_contract_sha256": _file_hash(dynamic_contract_path)
        if dynamic_contract_path.is_file()
        else "",
        "dynamic_contract_receipt_path": _relative(dynamic_receipt_path, root),
        "dynamic_contract_receipt_sha256": _file_hash(dynamic_receipt_path)
        if dynamic_receipt_path.is_file()
        else "",
        "holdout_result_id": str(holdout.get("result_id", "")),
        "holdout_result_path": _relative(holdout_result_path, root)
        if holdout_result_path is not None
        else "",
        "holdout_result_sha256": holdout_result_hash,
        "base_comparator_contract_id": str(comparator_receipt.get("contract_id", "")),
        "base_comparator_contract_path": _relative(comparator_path, root),
        "base_comparator_contract_sha256": _file_hash(comparator_path)
        if comparator_path.is_file()
        else "",
        "base_comparator_receipt_path": _relative(comparator_receipt_path, root),
        "base_comparator_receipt_sha256": _file_hash(comparator_receipt_path)
        if comparator_receipt_path.is_file()
        else "",
        "source_proof_path": _relative(proof_path, root),
        "source_proof_sha256": _file_hash(proof_path) if proof_path.is_file() else "",
        "supersession_gate_path": _relative(gate_path, root),
        "supersession_gate_sha256": _file_hash(gate_path) if gate_path.is_file() else "",
    }
    return context, blockers


def _write_or_validate_immutable_bytes(payload: bytes, path: Path) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable dynamic v2 artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, payload)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_or_validate_immutable_bytes(encoded.encode("utf-8"), path)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    promote_staged_file(temporary, path)


def _atomic_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    promote_staged_file(temporary, path)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error):
        return []


def _review_packet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Dynamic v2 Review Packet",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Review packet ID: `{payload.get('review_packet_id', '')}`",
        f"- Derivation cohort: `{payload.get('derivation_pair_group_id', '')}`",
        f"- Holdout cohort: `{payload.get('holdout_pair_group_id', '')}`",
        f"- Holdout cells passed: `{payload.get('passed_cells', 0)}/4`",
        f"- Cohorts disjoint: `{payload.get('cohorts_disjoint', False)}`",
        f"- Raw bindings valid: `{payload.get('raw_bindings_valid', False)}`",
        "- Automatic activation: `false`",
        "- Candidate promotion authority: `false`",
        "- Testnet order authority: `false`",
        "- Live trading authorized: `false`",
        "",
        "## Blockers",
        "",
    ]
    blockers = payload.get("blockers", [])
    lines.extend(
        [f"- `{blocker}`" for blocker in blockers]
        if blockers
        else ["- None. The packet is ready for explicit human review."]
    )
    lines.extend(
        [
            "",
            "## Holdout Cells",
            "",
            "| Mode | Orientation | Status | Spread Error | Z-score Error | Rolling Error | Raw Binding |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in payload.get("cells", []):
        lines.append(
            "| {exact_mode} | {orientation} | {cell_status} | "
            "{spread_max_abs_error} | {zscore_max_abs_error} | "
            "{zscore_roll_max_abs_error} | {raw_binding_valid} |".format(**cell)
        )
    lines.extend(
        [
            "",
            "## Review Action",
            "",
            "Activation requires the exact immutable review packet ID, a reviewer identity, and a substantive review note.",
            "The activation remains research-only and grants no order authority.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    path = path if path.is_absolute() else root / path
    return path if path.is_file() else None


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
