"""Reviewed, immutable activation for the OU-v5 research comparator."""

from __future__ import annotations

import csv
import math
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    frozen_capture_manifest_mutation_blockers,
)
from quant_platform.orchestration.corrective_wizard_ou_v5_holdout import (
    _implementation_source_hash,
    validate_ou_v5_evaluation_binding,
)
from quant_platform.wizard_ou_comparator_activation import (
    _as_utc,
    _atomic_json,
    _atomic_text,
    _canonical_json,
    _file_hash,
    _read_json,
    _relative,
    _resolve,
    _truthy,
    _write_or_validate_immutable_bytes,
    _write_or_validate_immutable_json,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "thewiz.wizard_ou_v5_activation.v1"
AUTHORITY_FIELDS = (
    "candidate_promotion_authority",
    "testnet_order_authority",
    "live_trading_authorized",
)


def build_ou_v5_supersession_gate(*, root: Path = ROOT) -> CommandResult:
    """Require all immutable v5 selector and formula evidence before review."""

    context, blockers = _activation_context(root=root)
    payload = {
        "schema_version": "thewiz.wizard_ou_v5_supersession_gate.v1",
        "status": "READY_FOR_REVIEWED_SUPERSESSION" if not blockers else "BLOCKED",
        "blockers": blockers,
        **context,
        "supersession_requires_reviewed_apply": True,
        "supersession_applied": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    output = root / "reports" / "active" / "wizard_ou_v5_supersession_gate.json"
    _atomic_json(payload, output)
    return CommandResult(paths={"supersession_gate": output}, summary=payload)


def build_ou_v5_review_packet(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
) -> CommandResult:
    """Build the exact eight-cell packet required for explicit v5 review."""

    timestamp = _as_utc(now)
    gate = build_ou_v5_supersession_gate(root=root)
    context, blockers = _activation_context(root=root)
    if gate.summary.get("status") != "READY_FOR_REVIEWED_SUPERSESSION":
        blockers.append("ou_v5_supersession_gate_not_ready")

    active = root / "reports" / "active"
    detail_path = active / "wizard_ou_v5_holdout_evaluation.csv"
    contract = _read_json(root / "config" / "wizard_ou_comparator_v5_holdout.json")
    expected_cells = {
        (
            _text(binding.get("pair_group")),
            _text(binding.get("exact_mode")),
            _text(binding.get("orientation")),
        )
        for binding in contract.get("holdout_bindings", [])
        if isinstance(binding, dict)
    }
    rows = _read_csv_rows(detail_path)
    observed = {
        (
            _text(row.get("pair_group")),
            _text(row.get("exact_mode")),
            _text(row.get("orientation")),
        )
        for row in rows
    }
    if len(expected_cells) != 8:
        blockers.append("ou_v5_review_packet_registered_cell_count_mismatch")
    if rows and observed != expected_cells:
        blockers.append("ou_v5_review_packet_cell_identity_mismatch")

    cells: list[dict[str, Any]] = []
    raw_bindings_valid = bool(rows)
    for row in sorted(
        rows,
        key=lambda value: (
            _text(value.get("pair_group")),
            _text(value.get("exact_mode")),
            _text(value.get("orientation")),
        ),
    ):
        request_path = _resolve(root, row.get("request_path"))
        response_path = _resolve(root, row.get("response_path"))
        request_hash = _text(row.get("request_sha256"))
        response_hash = _text(row.get("response_sha256"))
        binding_valid = bool(
            request_path is not None
            and response_path is not None
            and request_hash == _file_hash(request_path)
            and response_hash == _file_hash(response_path)
        )
        raw_bindings_valid = raw_bindings_valid and binding_valid
        if not binding_valid:
            blockers.append("ou_v5_review_packet_raw_binding_mismatch")
        cells.append(
            {
                "pair_group": _text(row.get("pair_group")),
                "pair": _text(row.get("pair")),
                "exact_mode": _text(row.get("exact_mode")),
                "orientation": _text(row.get("orientation")),
                "cell_status": _text(row.get("cell_status")),
                "formula_parity_passed": _truthy(row.get("formula_parity_passed")),
                "transform_selector_parity_passed": _truthy(
                    row.get("transform_selector_parity_passed")
                ),
                "trend_selector_parity_passed": _truthy(row.get("trend_selector_parity_passed")),
                "profile_branch_selector_parity_passed": _truthy(
                    row.get("profile_branch_selector_parity_passed")
                ),
                "predicted_profile_branch": _text(row.get("predicted_profile_branch")),
                "inferred_vendor_profile_branch": _text(row.get("inferred_vendor_profile_branch")),
                "selected_selector_cell": _text(row.get("selected_selector_cell")),
                "request_path": _text(row.get("request_path")),
                "request_sha256": request_hash,
                "response_path": _text(row.get("response_path")),
                "response_sha256": response_hash,
                "hedge_ratio_abs_error": _text(row.get("hedge_ratio_abs_error")),
                "spread_max_abs_error": _text(row.get("spread_max_abs_error")),
                "zscore_max_abs_error": _text(row.get("zscore_max_abs_error")),
                "zscore_roll_max_abs_error": _text(row.get("zscore_roll_max_abs_error")),
                "raw_binding_valid": binding_valid,
                "blocker": _text(row.get("blocker")),
            }
        )

    all_cells_passed = bool(
        len(cells) == 8
        and all(
            cell["cell_status"] == "PASS"
            and cell["formula_parity_passed"]
            and cell["transform_selector_parity_passed"]
            and cell["trend_selector_parity_passed"]
            and cell["profile_branch_selector_parity_passed"]
            for cell in cells
        )
    )
    cohorts_disjoint = bool(context.get("cohorts_disjoint", False))
    ready = bool(all_cells_passed and raw_bindings_valid and cohorts_disjoint and not blockers)
    holdout_status = _read_json(active / "wizard_ou_v5_holdout_status.json")
    status = (
        "READY_FOR_EXPLICIT_REVIEW"
        if ready
        else "BLOCKED_EVIDENCE_BINDING"
        if holdout_status.get("status") == "PASS"
        else "BLOCKED_HOLDOUT_EVIDENCE"
        if holdout_status.get("status") == "FAIL"
        else "WAITING_FOR_HOLDOUT"
    )
    payload: dict[str, Any] = {
        "schema_version": "thewiz.wizard_ou_v5_review_packet.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "required_cells": 8,
        "all_cells_passed": all_cells_passed,
        "formula_cells_passed": sum(bool(cell["formula_parity_passed"]) for cell in cells),
        "transform_selector_cells_passed": sum(
            bool(cell["transform_selector_parity_passed"]) for cell in cells
        ),
        "trend_selector_cells_passed": sum(
            bool(cell["trend_selector_parity_passed"]) for cell in cells
        ),
        "profile_branch_selector_cells_passed": sum(
            bool(cell["profile_branch_selector_parity_passed"]) for cell in cells
        ),
        "cohorts_disjoint": cohorts_disjoint,
        "raw_bindings_valid": raw_bindings_valid,
        "cells": cells,
        **context,
        "supersession_gate_path": _relative(gate.paths["supersession_gate"], root),
        "supersession_gate_sha256": _file_hash(gate.paths["supersession_gate"]),
        "holdout_detail_path": _relative(detail_path, root),
        "holdout_detail_sha256": (_file_hash(detail_path) if detail_path.is_file() else ""),
        "explicit_review_required": True,
        "automatic_activation": False,
        "raw_vendor_evidence_mutated": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    status_path = active / "wizard_ou_v5_review_packet.json"
    markdown_path = active / "wizard_ou_v5_review_packet.md"
    paths: dict[str, Path] = {
        "review_packet": status_path,
        "review_packet_markdown": markdown_path,
    }
    if ready:
        stable = {key: value for key, value in payload.items() if key != "evaluated_at_utc"}
        packet_id = "ouv5review_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
        immutable_payload = {**stable, "review_packet_id": packet_id}
        immutable_path = (
            root / "data" / "research" / "wizard_ou_v5_review_packets" / f"{packet_id}.json"
        )
        _write_or_validate_immutable_json(immutable_payload, immutable_path)
        payload.update(
            {
                "review_packet_id": packet_id,
                "immutable_review_packet_path": _relative(immutable_path, root),
                "immutable_review_packet_sha256": _file_hash(immutable_path),
            }
        )
        paths["immutable_review_packet"] = immutable_path
    _atomic_json(payload, status_path)
    _atomic_text(_review_packet_markdown(payload), markdown_path)
    return CommandResult(paths=paths, summary=payload)


def build_reviewed_ou_v5_activation(
    *,
    root: Path = ROOT,
    apply: bool = False,
    reviewer: str = "",
    review_note: str = "",
    review_packet_id: str = "",
    now: datetime | None = None,
) -> CommandResult:
    """Plan or explicitly apply the research-only OU-v5 comparator."""

    timestamp = _as_utc(now)
    existing = load_validated_ou_v5_activation(root=root)
    if existing is not None:
        status_path = root / "reports" / "active" / "wizard_ou_v5_activation_status.json"
        paths = {"activation_status": status_path}
        immutable_path = _resolve(root, existing.get("immutable_activation_path"))
        snapshot_path = _resolve(root, existing.get("source_proof_snapshot_path"))
        if immutable_path is not None:
            paths["immutable_activation"] = immutable_path
        if snapshot_path is not None:
            paths["proof_snapshot"] = snapshot_path
        return CommandResult(paths=paths, summary=existing)

    context, blockers = _activation_context(root=root)
    if apply:
        blockers.extend(frozen_capture_manifest_mutation_blockers(root=root))
    packet = build_ou_v5_review_packet(root=root, now=timestamp)
    packet_id = _text(packet.summary.get("review_packet_id"))
    if packet.summary.get("status") != "READY_FOR_EXPLICIT_REVIEW":
        blockers.append("ou_v5_review_packet_not_ready")
    else:
        context.update(
            {
                "review_packet_id": packet_id,
                "review_packet_path": _text(packet.summary.get("immutable_review_packet_path")),
                "review_packet_sha256": _text(packet.summary.get("immutable_review_packet_sha256")),
            }
        )
    if apply and not reviewer.strip():
        blockers.append("reviewer_required_for_ou_v5_apply")
    if apply and len(review_note.strip()) < 20:
        blockers.append("substantive_review_note_required_for_ou_v5_apply")
    if apply and not review_packet_id.strip():
        blockers.append("review_packet_id_required_for_ou_v5_apply")
    elif apply and review_packet_id.strip() != packet_id:
        blockers.append("review_packet_id_does_not_match_current_ou_v5_evidence")
    if apply:
        from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
            load_validated_ou_v5_supreme_review,
        )

        try:
            supreme_review = load_validated_ou_v5_supreme_review(
                root=root,
                expected_review_packet_id=packet_id,
                expected_review_packet_path=context.get("review_packet_path", ""),
                expected_review_packet_sha256=context.get("review_packet_sha256", ""),
            )
        except (OSError, TypeError, ValueError):
            supreme_review = None
            blockers.append("ou_v5_supreme_review_invalid_for_apply")
        if supreme_review is None and "ou_v5_supreme_review_invalid_for_apply" not in blockers:
            blockers.append("ou_v5_supreme_review_required_for_apply")
        elif supreme_review is not None:
            context.update(
                {
                    "supreme_review_id": _text(supreme_review.get("review_id")),
                    "supreme_review_path": _text(supreme_review.get("immutable_review_path")),
                    "supreme_review_sha256": _text(supreme_review.get("immutable_review_sha256")),
                }
            )

    status_path = root / "reports" / "active" / "wizard_ou_v5_activation_status.json"
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
        "comparator_generation": 5,
        "automatic_activation": False,
        "raw_vendor_evidence_mutated": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    if blockers or not apply:
        _atomic_json(base, status_path)
        return CommandResult(paths={"activation_status": status_path}, summary=base)

    proof_path = root / _text(context["source_proof_path"])
    snapshot_hash = _file_hash(proof_path)
    snapshot_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v5_comparator_activations"
        / "proof_snapshots"
        / f"{snapshot_hash}.csv"
    )
    _write_or_validate_immutable_bytes(proof_path.read_bytes(), snapshot_path)
    immutable_payload = {
        **base,
        "source_proof_snapshot_path": _relative(snapshot_path, root),
        "source_proof_snapshot_sha256": snapshot_hash,
    }
    stable = {key: value for key, value in immutable_payload.items() if key != "evaluated_at_utc"}
    activation_id = (
        "ouv5activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    immutable_payload["activation_id"] = activation_id
    immutable_path = (
        root / "data" / "research" / "wizard_ou_v5_comparator_activations" / f"{activation_id}.json"
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


def load_validated_ou_v5_activation(
    *,
    root: Path = ROOT,
) -> dict[str, Any] | None:
    """Return the active v5 receipt only when every immutable binding holds."""

    status_path = root / "reports" / "active" / "wizard_ou_v5_activation_status.json"
    if not status_path.is_file():
        return None
    pointer = _read_json(status_path)
    if not pointer:
        raise ValueError("OU v5 activation status is unreadable")
    immutable_dir = root / "data" / "research" / "wizard_ou_v5_comparator_activations"
    if pointer.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        if pointer.get("activation_id") or any(immutable_dir.glob("*.json")):
            raise ValueError("OU v5 immutable activation exists without active pointer")
        return None
    if pointer.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("OU v5 activation schema mismatch")
    if int(pointer.get("comparator_generation", 0) or 0) != 5:
        raise ValueError("OU v5 activation generation mismatch")
    if pointer.get("implementation_source_sha256") != _implementation_source_hash():
        raise ValueError("OU v5 implementation source hash mismatch")
    if not pointer.get("reviewer") or len(_text(pointer.get("review_note"))) < 20:
        raise ValueError("OU v5 active review evidence missing")
    if pointer.get("review_packet_id_submitted") != pointer.get("review_packet_id"):
        raise ValueError("OU v5 active review packet approval mismatch")
    try:
        profile_threshold = float(pointer.get("profile_intercept_threshold"))
        profile_separation = float(pointer.get("profile_training_separation"))
    except (TypeError, ValueError) as exc:
        raise ValueError("OU v5 profile selector activation contract invalid") from exc
    if (
        not math.isfinite(profile_threshold)
        or not math.isfinite(profile_separation)
        or profile_separation <= 0.0
        or pointer.get("transform_scale_sensitive") is not True
        or pointer.get("profile_threshold_is_fragile") is not True
    ):
        raise ValueError("OU v5 profile selector activation contract invalid")
    if any(_truthy(pointer.get(key)) for key in AUTHORITY_FIELDS):
        raise ValueError("OU v5 activation exceeded research authority")

    immutable_path = _resolve(root, pointer.get("immutable_activation_path"))
    if immutable_path is None or _file_hash(immutable_path) != _text(
        pointer.get("immutable_activation_sha256")
    ):
        raise ValueError("OU v5 immutable activation binding mismatch")
    immutable = _read_json(immutable_path)
    for key, value in immutable.items():
        if pointer.get(key) != value:
            raise ValueError(f"OU v5 active pointer mismatch: {key}")
    activation_id = _text(immutable.get("activation_id"))
    stable = {
        key: value
        for key, value in immutable.items()
        if key not in {"activation_id", "evaluated_at_utc"}
    }
    expected_activation_id = (
        "ouv5activation_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    expected_activation_path = (
        root / "data" / "research" / "wizard_ou_v5_comparator_activations" / f"{activation_id}.json"
    )
    if (
        activation_id != expected_activation_id
        or immutable_path.resolve() != expected_activation_path.resolve()
    ):
        raise ValueError("OU v5 immutable activation identity mismatch")
    from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
        validate_ou_v5_supreme_review_receipt,
    )

    supreme_review = validate_ou_v5_supreme_review_receipt(
        root=root,
        immutable_review_path=immutable.get("supreme_review_path"),
        immutable_review_sha256=immutable.get("supreme_review_sha256"),
        expected_review_packet_id=immutable.get("review_packet_id"),
        expected_review_packet_path=immutable.get("review_packet_path"),
        expected_review_packet_sha256=immutable.get("review_packet_sha256"),
    )
    if supreme_review.get("review_id") != immutable.get("supreme_review_id"):
        raise ValueError("OU v5 activation Supreme Team review ID mismatch")
    bindings = (
        ("ou_contract_path", "ou_contract_sha256"),
        ("ou_receipt_path", "ou_receipt_sha256"),
        ("derivation_path", "derivation_sha256"),
        ("failure_attribution_path", "failure_attribution_sha256"),
        ("predictions_path", "predictions_sha256"),
        ("holdout_result_path", "holdout_result_sha256"),
        ("holdout_detail_path", "holdout_detail_sha256"),
        ("base_comparator_contract_path", "base_comparator_contract_sha256"),
        ("base_comparator_receipt_path", "base_comparator_receipt_sha256"),
        ("review_packet_path", "review_packet_sha256"),
        ("supreme_review_path", "supreme_review_sha256"),
        ("source_proof_snapshot_path", "source_proof_snapshot_sha256"),
    )
    for path_field, hash_field in bindings:
        path = _resolve(root, immutable.get(path_field))
        if path is None or _file_hash(path) != _text(immutable.get(hash_field)):
            raise ValueError(f"OU v5 activation binding mismatch: {path_field}")
    if immutable.get("source_proof_sha256") != immutable.get("source_proof_snapshot_sha256"):
        raise ValueError("OU v5 activation source proof snapshot mismatch")
    return pointer


def _activation_context(*, root: Path) -> tuple[dict[str, Any], list[str]]:
    active = root / "reports" / "active"
    status_path = active / "wizard_ou_v5_holdout_status.json"
    detail_path = active / "wizard_ou_v5_holdout_evaluation.csv"
    contract_path = root / "config" / "wizard_ou_comparator_v5_holdout.json"
    receipt_path = active / "wizard_ou_v5_holdout_receipt.json"
    comparator_path = active / "wizard_mode_comparator_contract.csv"
    comparator_receipt_path = active / "wizard_mode_comparator_contract_receipt.json"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    status = _read_json(status_path)
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    comparator_receipt = _read_json(comparator_receipt_path)
    derivation_items = contract.get("derivation_evidence", [])
    predictions = contract.get("holdout_predictions", {})
    derivation_path = _resolve(
        root,
        derivation_items[0].get("path")
        if isinstance(derivation_items, list) and derivation_items
        else "",
    )
    failure_attribution_item = (
        derivation_items[1]
        if isinstance(derivation_items, list)
        and len(derivation_items) > 1
        and isinstance(derivation_items[1], dict)
        else {}
    )
    failure_attribution_path = _resolve(root, failure_attribution_item.get("path"))
    predictions_path = _resolve(root, predictions.get("path"))
    blockers: list[str] = []
    evaluation_binding = validate_ou_v5_evaluation_binding(
        root=root,
        status=status,
    )
    if evaluation_binding.get("status") != "PASS":
        blockers.append("ou_v5_evaluation_binding_invalid")
        blockers.extend(
            _text(blocker) for blocker in evaluation_binding.get("blockers", []) if _text(blocker)
        )
    if status.get("status") != "PASS":
        blockers.append("ou_v5_prospective_holdout_not_passed")
    for field in (
        "required_cells",
        "passed_cells",
        "formula_parity_passed_cells",
        "transform_selector_parity_passed_cells",
        "trend_selector_parity_passed_cells",
        "profile_branch_selector_parity_passed_cells",
    ):
        if int(status.get(field, 0) or 0) != 8:
            blockers.append(f"ou_v5_{field}_not_eight_of_eight")
    hypothesis = contract.get("hypothesis", {})
    if not _truthy(hypothesis.get("holdout_is_asset_and_pair_disjoint")):
        blockers.append("ou_v5_holdout_cohorts_not_disjoint")
    try:
        profile_threshold = float(hypothesis.get("profile_intercept_threshold"))
        profile_separation = float(hypothesis.get("profile_training_separation"))
    except (TypeError, ValueError):
        profile_threshold = None
        profile_separation = None
    if (
        profile_threshold is None
        or profile_separation is None
        or not math.isfinite(profile_threshold)
        or not math.isfinite(profile_separation)
        or profile_separation <= 0.0
    ):
        blockers.append("ou_v5_profile_selector_contract_invalid")
    holdout_result_path = _resolve(root, status.get("immutable_result_path"))
    holdout_result_hash = _text(status.get("immutable_result_sha256"))
    if holdout_result_path is None or not holdout_result_hash:
        blockers.append("ou_v5_immutable_holdout_result_missing")
    elif _file_hash(holdout_result_path) != holdout_result_hash:
        blockers.append("ou_v5_immutable_holdout_result_hash_mismatch")
    implementation_hash = _implementation_source_hash()
    if contract.get("implementation", {}).get("source_sha256") != implementation_hash:
        blockers.append("ou_v5_candidate_source_hash_mismatch")
    if receipt.get("implementation_source_sha256") != implementation_hash:
        blockers.append("ou_v5_receipt_source_hash_mismatch")
    if receipt.get("contract_sha256") != (
        _file_hash(contract_path) if contract_path.is_file() else ""
    ):
        blockers.append("ou_v5_contract_receipt_hash_mismatch")
    if derivation_path is None or not derivation_items:
        blockers.append("ou_v5_derivation_evidence_missing")
    elif _file_hash(derivation_path) != _text(derivation_items[0].get("sha256")):
        blockers.append("ou_v5_derivation_binding_mismatch")
    if failure_attribution_path is None or len(derivation_items) < 2:
        blockers.append("ou_v5_failure_attribution_evidence_missing")
    elif _file_hash(failure_attribution_path) != _text(failure_attribution_item.get("sha256")):
        blockers.append("ou_v5_failure_attribution_binding_mismatch")
    if predictions_path is None:
        blockers.append("ou_v5_predictions_missing")
    elif _file_hash(predictions_path) != _text(predictions.get("sha256")):
        blockers.append("ou_v5_predictions_binding_mismatch")
    required = (
        status_path,
        detail_path,
        contract_path,
        receipt_path,
        comparator_path,
        comparator_receipt_path,
        proof_path,
    )
    if not all(path.is_file() for path in required):
        blockers.append("ou_v5_required_activation_evidence_missing")
    if any(
        _truthy(payload.get(key))
        for payload in (status, contract, receipt)
        for key in AUTHORITY_FIELDS
    ):
        blockers.append("ou_v5_source_evidence_exceeded_research_authority")

    context = {
        "implementation_source_sha256": implementation_hash,
        "profile_intercept_threshold": profile_threshold,
        "profile_training_separation": profile_separation,
        "transform_scale_sensitive": _truthy(hypothesis.get("transform_scale_sensitive")),
        "profile_threshold_is_fragile": _truthy(hypothesis.get("profile_threshold_is_fragile")),
        "cohorts_disjoint": _truthy(hypothesis.get("holdout_is_asset_and_pair_disjoint")),
        "ou_contract_path": _relative(contract_path, root),
        "ou_contract_sha256": _file_hash(contract_path) if contract_path.is_file() else "",
        "ou_receipt_path": _relative(receipt_path, root),
        "ou_receipt_sha256": _file_hash(receipt_path) if receipt_path.is_file() else "",
        "derivation_path": _relative(derivation_path, root) if derivation_path else "",
        "derivation_sha256": _file_hash(derivation_path) if derivation_path else "",
        "failure_attribution_path": (
            _relative(failure_attribution_path, root) if failure_attribution_path else ""
        ),
        "failure_attribution_sha256": (
            _file_hash(failure_attribution_path) if failure_attribution_path else ""
        ),
        "predictions_path": _relative(predictions_path, root) if predictions_path else "",
        "predictions_sha256": _file_hash(predictions_path) if predictions_path else "",
        "holdout_result_id": _text(status.get("result_id")),
        "holdout_result_path": (
            _relative(holdout_result_path, root) if holdout_result_path else ""
        ),
        "holdout_result_sha256": holdout_result_hash,
        "holdout_detail_path": _relative(detail_path, root),
        "holdout_detail_sha256": _file_hash(detail_path) if detail_path.is_file() else "",
        "base_comparator_contract_id": _text(comparator_receipt.get("contract_id")),
        "base_comparator_contract_path": _relative(comparator_path, root),
        "base_comparator_contract_sha256": (
            _file_hash(comparator_path) if comparator_path.is_file() else ""
        ),
        "base_comparator_receipt_path": _relative(comparator_receipt_path, root),
        "base_comparator_receipt_sha256": (
            _file_hash(comparator_receipt_path) if comparator_receipt_path.is_file() else ""
        ),
        "source_proof_path": _relative(proof_path, root),
        "source_proof_sha256": _file_hash(proof_path) if proof_path.is_file() else "",
    }
    return context, list(dict.fromkeys(blockers))


def _review_packet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OU v5 Review Packet",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Review packet: `{payload.get('review_packet_id', '')}`",
        f"- Holdout cells passed: `{sum(cell.get('cell_status') == 'PASS' for cell in payload.get('cells', []))}/8`",
        f"- Formula cells passed: `{payload.get('formula_cells_passed', 0)}/8`",
        f"- Transform selector cells passed: `{payload.get('transform_selector_cells_passed', 0)}/8`",
        f"- Trend selector cells passed: `{payload.get('trend_selector_cells_passed', 0)}/8`",
        f"- Profile branch cells passed: `{payload.get('profile_branch_selector_cells_passed', 0)}/8`",
        f"- Raw bindings valid: `{payload.get('raw_bindings_valid', False)}`",
        "- Scope: research comparator only",
        "- Automatic activation: `false`",
        "- Candidate/Testnet/live authority: `false`",
        "",
        "Explicit review must confirm formula, transform, trend diagnostic, and profile branch evidence.",
        "The transform rule is scale-sensitive and the profile threshold is frozen but fragile.",
    ]
    blockers = payload.get("blockers", [])
    if blockers:
        lines.extend(["", "## Blockers", *[f"- `{item}`" for item in blockers]])
    return "\n".join(lines) + "\n"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except FileNotFoundError:
        return []


def _text(value: object) -> str:
    return str(value or "").strip()
