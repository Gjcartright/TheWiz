"""Reviewed, immutable activation for the OU-v3 research comparator."""

from __future__ import annotations

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
SCHEMA_VERSION = "thewiz.wizard_ou_v3_activation.v1"
AUTHORITY_FIELDS = (
    "candidate_promotion_authority",
    "testnet_order_authority",
    "live_trading_authorized",
)


def build_ou_v3_supersession_gate(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    selector_source_sha256: str,
) -> CommandResult:
    """Require independent formula and selector holdout proof before review."""

    context, blockers = _activation_context(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
    )
    payload = {
        "schema_version": "thewiz.wizard_ou_v3_supersession_gate.v1",
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
    output = root / "reports" / "active" / "wizard_ou_v3_supersession_gate.json"
    _atomic_json(payload, output)
    return CommandResult(paths={"supersession_gate": output}, summary=payload)


def build_ou_v3_review_packet(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    selector_source_sha256: str,
    now: datetime | None = None,
) -> CommandResult:
    """Build the exact evidence packet required for explicit OU-v3 review."""

    timestamp = _as_utc(now)
    gate = build_ou_v3_supersession_gate(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
    )
    context, blockers = _activation_context(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
    )
    if gate.summary.get("status") != "READY_FOR_REVIEWED_SUPERSESSION":
        blockers.append("ou_v3_supersession_gate_not_ready")
    active = root / "reports" / "active"
    detail_path = active / "wizard_ou_v3_holdout_evaluation.csv"
    rows = _read_csv_rows(detail_path)
    expected = {
        (mode, orientation)
        for mode in ("OU (Spread)", "OU (ZScoreR)")
        for orientation in ("original", "reverse")
    }
    observed = {
        (str(row.get("exact_mode", "")), str(row.get("orientation", "")))
        for row in rows
    }
    if observed != expected:
        blockers.append("ou_v3_review_packet_cell_identity_mismatch")

    cells: list[dict[str, Any]] = []
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
        binding_valid = bool(
            request_path is not None
            and response_path is not None
            and request_hash == _file_hash(request_path)
            and response_hash == _file_hash(response_path)
        )
        raw_bindings_valid = raw_bindings_valid and binding_valid
        if not binding_valid:
            blockers.append("ou_v3_review_packet_raw_binding_mismatch")
        cells.append(
            {
                "exact_mode": str(row.get("exact_mode", "")),
                "orientation": str(row.get("orientation", "")),
                "cell_status": str(row.get("cell_status", "")),
                "formula_parity_passed": _truthy(
                    row.get("formula_parity_passed")
                ),
                "trend_selector_parity_passed": _truthy(
                    row.get("trend_selector_parity_passed")
                ),
                "local_predicted_inc_trend": _truthy(
                    row.get("local_predicted_inc_trend")
                ),
                "vendor_inc_trend": _truthy(row.get("vendor_inc_trend")),
                "request_path": str(row.get("request_path", "")),
                "request_sha256": request_hash,
                "response_path": str(row.get("response_path", "")),
                "response_sha256": response_hash,
                "raw_binding_valid": binding_valid,
                "blocker": str(row.get("blocker", "")),
            }
        )
    all_cells_passed = bool(
        len(cells) == 4
        and all(
            cell["cell_status"] == "PASS"
            and cell["formula_parity_passed"]
            and cell["trend_selector_parity_passed"]
            for cell in cells
        )
    )
    ready = bool(all_cells_passed and raw_bindings_valid and not blockers)
    holdout_status = _read_json(active / "wizard_ou_v3_holdout_status.json")
    status = (
        "READY_FOR_EXPLICIT_REVIEW"
        if ready
        else "BLOCKED_EVIDENCE_BINDING"
        if holdout_status.get("status") == "PASS"
        else "BLOCKED_HOLDOUT_EVIDENCE"
        if holdout_status.get("status") in {"FAIL", "INCOMPLETE"}
        else "WAITING_FOR_HOLDOUT"
    )
    payload: dict[str, Any] = {
        "schema_version": "thewiz.wizard_ou_v3_review_packet.v1",
        "evaluated_at_utc": timestamp.isoformat(),
        "status": status,
        "blockers": list(dict.fromkeys(blockers)),
        "required_cells": 4,
        "all_cells_passed": all_cells_passed,
        "formula_cells_passed": sum(
            bool(cell["formula_parity_passed"]) for cell in cells
        ),
        "selector_cells_passed": sum(
            bool(cell["trend_selector_parity_passed"]) for cell in cells
        ),
        "raw_bindings_valid": raw_bindings_valid,
        "cells": cells,
        **context,
        "supersession_gate_path": _relative(
            gate.paths["supersession_gate"], root
        ),
        "supersession_gate_sha256": _file_hash(
            gate.paths["supersession_gate"]
        ),
        "holdout_detail_path": _relative(detail_path, root),
        "holdout_detail_sha256": _file_hash(detail_path)
        if detail_path.is_file()
        else "",
        "explicit_review_required": True,
        "automatic_activation": False,
        "raw_vendor_evidence_mutated": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    status_path = active / "wizard_ou_v3_review_packet.json"
    markdown_path = active / "wizard_ou_v3_review_packet.md"
    paths: dict[str, Path] = {
        "review_packet": status_path,
        "review_packet_markdown": markdown_path,
    }
    if ready:
        stable = {
            key: value
            for key, value in payload.items()
            if key != "evaluated_at_utc"
        }
        packet_id = "ouv3review_" + sha256(
            _canonical_json(stable).encode("utf-8")
        ).hexdigest()[:20]
        immutable_payload = {**stable, "review_packet_id": packet_id}
        immutable_path = (
            root
            / "data"
            / "research"
            / "wizard_ou_v3_review_packets"
            / f"{packet_id}.json"
        )
        _write_or_validate_immutable_json(immutable_payload, immutable_path)
        payload.update(
            {
                "review_packet_id": packet_id,
                "immutable_review_packet_path": _relative(
                    immutable_path, root
                ),
                "immutable_review_packet_sha256": _file_hash(immutable_path),
            }
        )
        paths["immutable_review_packet"] = immutable_path
    _atomic_json(payload, status_path)
    _atomic_text(_review_packet_markdown(payload), markdown_path)
    return CommandResult(paths=paths, summary=payload)


def build_reviewed_ou_v3_activation(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    selector_source_sha256: str,
    apply: bool = False,
    reviewer: str = "",
    review_note: str = "",
    review_packet_id: str = "",
    now: datetime | None = None,
) -> CommandResult:
    """Plan or explicitly apply the research-only OU-v3 comparator."""

    timestamp = _as_utc(now)
    existing = load_validated_ou_v3_activation(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
    )
    if existing is not None:
        status_path = root / "reports" / "active" / "wizard_ou_v3_activation_status.json"
        paths = {"activation_status": status_path}
        immutable_path = _resolve(root, existing.get("immutable_activation_path"))
        snapshot_path = _resolve(root, existing.get("source_proof_snapshot_path"))
        if immutable_path is not None:
            paths["immutable_activation"] = immutable_path
        if snapshot_path is not None:
            paths["proof_snapshot"] = snapshot_path
        return CommandResult(paths=paths, summary=existing)

    context, blockers = _activation_context(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
    )
    if apply:
        blockers.extend(frozen_capture_manifest_mutation_blockers(root=root))
    packet = build_ou_v3_review_packet(
        root=root,
        implementation_source_sha256=implementation_source_sha256,
        selector_source_sha256=selector_source_sha256,
        now=timestamp,
    )
    packet_id = str(packet.summary.get("review_packet_id", ""))
    if packet.summary.get("status") != "READY_FOR_EXPLICIT_REVIEW":
        blockers.append("ou_v3_review_packet_not_ready")
    else:
        context.update(
            {
                "review_packet_id": packet_id,
                "review_packet_path": str(
                    packet.summary.get("immutable_review_packet_path", "")
                ),
                "review_packet_sha256": str(
                    packet.summary.get("immutable_review_packet_sha256", "")
                ),
            }
        )
    if apply and not reviewer.strip():
        blockers.append("reviewer_required_for_ou_v3_apply")
    if apply and len(review_note.strip()) < 20:
        blockers.append("substantive_review_note_required_for_ou_v3_apply")
    if apply and not review_packet_id.strip():
        blockers.append("review_packet_id_required_for_ou_v3_apply")
    elif apply and review_packet_id.strip() != packet_id:
        blockers.append("review_packet_id_does_not_match_current_ou_v3_evidence")

    status_path = root / "reports" / "active" / "wizard_ou_v3_activation_status.json"
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
        "comparator_generation": 3,
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

    proof_path = root / str(context["source_proof_path"])
    snapshot_hash = _file_hash(proof_path)
    snapshot_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v3_comparator_activations"
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
        key: value
        for key, value in immutable_payload.items()
        if key != "evaluated_at_utc"
    }
    activation_id = "ouv3activation_" + sha256(
        _canonical_json(stable).encode("utf-8")
    ).hexdigest()[:20]
    immutable_payload["activation_id"] = activation_id
    immutable_path = (
        root
        / "data"
        / "research"
        / "wizard_ou_v3_comparator_activations"
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


def load_validated_ou_v3_activation(
    *,
    root: Path = ROOT,
    implementation_source_sha256: str,
    selector_source_sha256: str,
) -> dict[str, Any] | None:
    """Return the OU-v3 receipt only when every immutable binding holds."""

    status_path = root / "reports" / "active" / "wizard_ou_v3_activation_status.json"
    if not status_path.is_file():
        return None
    pointer = _read_json(status_path)
    if not pointer:
        raise ValueError("OU v3 activation status is unreadable")
    if pointer.get("status") != "APPLIED_RESEARCH_COMPARATOR_ONLY":
        immutable_dir = root / "data" / "research" / "wizard_ou_v3_comparator_activations"
        if pointer.get("activation_id") or any(immutable_dir.glob("*.json")):
            raise ValueError("OU v3 immutable activation exists without active pointer")
        return None
    if int(pointer.get("comparator_generation", 0) or 0) != 3:
        raise ValueError("OU v3 activation generation mismatch")
    if pointer.get("implementation_source_sha256") != implementation_source_sha256:
        raise ValueError("OU v3 implementation source hash mismatch")
    if pointer.get("selector_source_sha256") != selector_source_sha256:
        raise ValueError("OU v3 selector source hash mismatch")
    if not pointer.get("reviewer") or len(str(pointer.get("review_note", "")).strip()) < 20:
        raise ValueError("OU v3 active review evidence missing")
    if pointer.get("review_packet_id_submitted") != pointer.get("review_packet_id"):
        raise ValueError("OU v3 active review packet approval mismatch")
    if any(_truthy(pointer.get(key)) for key in AUTHORITY_FIELDS):
        raise ValueError("OU v3 activation exceeded research authority")
    immutable_path = _resolve(root, pointer.get("immutable_activation_path"))
    if immutable_path is None or _file_hash(immutable_path) != str(
        pointer.get("immutable_activation_sha256", "")
    ):
        raise ValueError("OU v3 immutable activation binding mismatch")
    immutable = _read_json(immutable_path)
    for key, value in immutable.items():
        if pointer.get(key) != value:
            raise ValueError(f"OU v3 active pointer mismatch: {key}")
    bindings = (
        ("ou_contract_path", "ou_contract_sha256"),
        ("ou_receipt_path", "ou_receipt_sha256"),
        ("selector_contract_path", "selector_contract_sha256"),
        ("selector_receipt_path", "selector_receipt_sha256"),
        ("selector_derivation_path", "selector_derivation_sha256"),
        ("selector_predictions_path", "selector_predictions_sha256"),
        ("holdout_result_path", "holdout_result_sha256"),
        ("base_comparator_contract_path", "base_comparator_contract_sha256"),
        ("base_comparator_receipt_path", "base_comparator_receipt_sha256"),
        ("review_packet_path", "review_packet_sha256"),
        ("source_proof_snapshot_path", "source_proof_snapshot_sha256"),
    )
    for path_field, hash_field in bindings:
        path = _resolve(root, immutable.get(path_field))
        if path is None or _file_hash(path) != str(immutable.get(hash_field, "")):
            raise ValueError(f"OU v3 activation binding mismatch: {path_field}")
    return pointer


def _activation_context(
    *,
    root: Path,
    implementation_source_sha256: str,
    selector_source_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    active = root / "reports" / "active"
    status_path = active / "wizard_ou_v3_holdout_status.json"
    contract_path = root / "config" / "wizard_ou_comparator_v3_holdout.json"
    receipt_path = active / "wizard_ou_v3_holdout_receipt.json"
    selector_contract_path = root / "config" / "wizard_ou_trend_selector_v1_holdout.json"
    selector_receipt_path = active / "wizard_ou_trend_selector_v1_receipt.json"
    selector_derivation_path = active / "wizard_ou_trend_selector_v1_derivation.csv"
    selector_predictions_path = active / "wizard_ou_trend_selector_v1_predictions.csv"
    comparator_path = active / "wizard_mode_comparator_contract.csv"
    comparator_receipt_path = active / "wizard_mode_comparator_contract_receipt.json"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    status = _read_json(status_path)
    contract = _read_json(contract_path)
    receipt = _read_json(receipt_path)
    selector_contract = _read_json(selector_contract_path)
    selector_receipt = _read_json(selector_receipt_path)
    comparator_receipt = _read_json(comparator_receipt_path)
    blockers: list[str] = []
    if status.get("status") != "PASS":
        blockers.append("ou_v3_prospective_holdout_not_passed")
    if int(status.get("required_cells", 0) or 0) != 4:
        blockers.append("ou_v3_required_cell_contract_not_four")
    if int(status.get("formula_parity_passed_cells", 0) or 0) != 4:
        blockers.append("ou_v3_formula_parity_not_four_of_four")
    if int(status.get("trend_selector_parity_passed_cells", 0) or 0) != 4:
        blockers.append("ou_v3_selector_parity_not_four_of_four")
    if not _truthy(status.get("local_point_in_time_trend_selector_proven")):
        blockers.append("ou_v3_local_point_in_time_selector_not_proven")
    holdout_result_path = _resolve(root, status.get("immutable_result_path"))
    holdout_result_hash = str(status.get("immutable_result_sha256", ""))
    if holdout_result_path is None or not holdout_result_hash:
        blockers.append("ou_v3_immutable_holdout_result_missing")
    elif _file_hash(holdout_result_path) != holdout_result_hash:
        blockers.append("ou_v3_immutable_holdout_result_hash_mismatch")
    if contract.get("implementation", {}).get("source_sha256") != implementation_source_sha256:
        blockers.append("ou_v3_candidate_source_hash_mismatch")
    if selector_contract.get("selector", {}).get("source_sha256") != selector_source_sha256:
        blockers.append("ou_v3_selector_source_hash_mismatch")
    if receipt.get("contract_sha256") != (
        _file_hash(contract_path) if contract_path.is_file() else ""
    ):
        blockers.append("ou_v3_contract_receipt_hash_mismatch")
    if selector_receipt.get("contract_sha256") != (
        _file_hash(selector_contract_path) if selector_contract_path.is_file() else ""
    ):
        blockers.append("ou_v3_selector_receipt_hash_mismatch")
    required = (
        status_path,
        contract_path,
        receipt_path,
        selector_contract_path,
        selector_receipt_path,
        selector_derivation_path,
        selector_predictions_path,
        comparator_path,
        comparator_receipt_path,
        proof_path,
    )
    if not all(path.is_file() for path in required):
        blockers.append("ou_v3_required_activation_evidence_missing")
    if any(
        _truthy(payload.get(key))
        for payload in (status, contract, receipt, selector_contract, selector_receipt)
        for key in AUTHORITY_FIELDS
    ):
        blockers.append("ou_v3_source_evidence_exceeded_research_authority")
    context = {
        "implementation_source_sha256": implementation_source_sha256,
        "selector_source_sha256": selector_source_sha256,
        "ou_contract_path": _relative(contract_path, root),
        "ou_contract_sha256": _file_hash(contract_path) if contract_path.is_file() else "",
        "ou_receipt_path": _relative(receipt_path, root),
        "ou_receipt_sha256": _file_hash(receipt_path) if receipt_path.is_file() else "",
        "selector_contract_path": _relative(selector_contract_path, root),
        "selector_contract_sha256": _file_hash(selector_contract_path) if selector_contract_path.is_file() else "",
        "selector_receipt_path": _relative(selector_receipt_path, root),
        "selector_receipt_sha256": _file_hash(selector_receipt_path) if selector_receipt_path.is_file() else "",
        "selector_derivation_path": _relative(selector_derivation_path, root),
        "selector_derivation_sha256": _file_hash(selector_derivation_path) if selector_derivation_path.is_file() else "",
        "selector_predictions_path": _relative(selector_predictions_path, root),
        "selector_predictions_sha256": _file_hash(selector_predictions_path) if selector_predictions_path.is_file() else "",
        "holdout_result_id": str(status.get("result_id", "")),
        "holdout_result_path": _relative(holdout_result_path, root) if holdout_result_path else "",
        "holdout_result_sha256": holdout_result_hash,
        "base_comparator_contract_id": str(comparator_receipt.get("contract_id", "")),
        "base_comparator_contract_path": _relative(comparator_path, root),
        "base_comparator_contract_sha256": _file_hash(comparator_path) if comparator_path.is_file() else "",
        "base_comparator_receipt_path": _relative(comparator_receipt_path, root),
        "base_comparator_receipt_sha256": _file_hash(comparator_receipt_path) if comparator_receipt_path.is_file() else "",
        "source_proof_path": _relative(proof_path, root),
        "source_proof_sha256": _file_hash(proof_path) if proof_path.is_file() else "",
    }
    return context, list(dict.fromkeys(blockers))


def _review_packet_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OU v3 Review Packet",
        "",
        f"- Status: `{payload.get('status', '')}`",
        f"- Review packet: `{payload.get('review_packet_id', '')}`",
        f"- Formula cells passed: `{payload.get('formula_cells_passed', 0)}/4`",
        f"- Selector cells passed: `{payload.get('selector_cells_passed', 0)}/4`",
        f"- Raw bindings valid: `{payload.get('raw_bindings_valid', False)}`",
        "- Scope: research comparator only",
        "- Automatic activation: `false`",
        "- Candidate/Testnet/live authority: `false`",
        "",
        "Explicit review must confirm the formula and point-in-time trend selector independently.",
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


def _as_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _write_or_validate_immutable_bytes(payload: bytes, path: Path) -> None:
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable OU v3 artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_or_validate_immutable_bytes(encoded.encode("utf-8"), path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}
