"""Unify the human-reviewed Dynamic and active OU comparator boundary."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import promote_staged_file
from quant_platform.orchestration.corrective_wizard_capture_reconciliation import (
    frozen_capture_manifest_mutation_blockers,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.corrective_wizard_comparator_review_control.v1"
DYNAMIC_V2_COMPARATOR = {
    "comparator": "dynamic_v2",
    "generation": 2,
    "required_cells": 4,
    "packet": "wizard_dynamic_v2_review_packet.json",
    "packet_schema": "thewiz.wizard_dynamic_v2_review_packet.v1",
    "packet_prefix": "dynamicv2review_",
    "gate": "wizard_dynamic_v2_supersession_gate.json",
    "gate_schema": "thewiz.wizard_dynamic_comparator_supersession_gate.v1",
    "activation": "wizard_dynamic_v2_activation_status.json",
    "activation_schema": "thewiz.wizard_dynamic_v2_activation.v2",
    "immutable_activation_root": "data/research/wizard_dynamic_comparator_activations",
    "supreme": "wizard_dynamic_v2_review.json",
    "supreme_required": True,
}
OU_V3_COMPARATOR = {
    "comparator": "ou_v3",
    "generation": 3,
    "required_cells": 4,
    "packet": "wizard_ou_v3_review_packet.json",
    "packet_schema": "thewiz.wizard_ou_v3_review_packet.v1",
    "packet_prefix": "ouv3review_",
    "gate": "wizard_ou_v3_supersession_gate.json",
    "gate_schema": "thewiz.wizard_ou_v3_supersession_gate.v1",
    "activation": "wizard_ou_v3_activation_status.json",
    "activation_schema": "thewiz.wizard_ou_v3_activation.v1",
    "immutable_activation_root": "data/research/wizard_ou_comparator_activations",
    "supreme": "wizard_ou_v3_review.json",
    "supreme_required": False,
}
OU_V4_COMPARATOR = {
    "comparator": "ou_v4",
    "generation": 4,
    "required_cells": 8,
    "packet": "wizard_ou_v4_review_packet.json",
    "packet_schema": "thewiz.wizard_ou_v4_review_packet.v1",
    "packet_prefix": "ouv4review_",
    "gate": "wizard_ou_v4_supersession_gate.json",
    "gate_schema": "thewiz.wizard_ou_v4_supersession_gate.v1",
    "activation": "wizard_ou_v4_activation_status.json",
    "activation_schema": "thewiz.wizard_ou_v4_activation.v2",
    "immutable_activation_root": "data/research/wizard_ou_v4_comparator_activations",
    "supreme": "wizard_ou_v4_review.json",
    "supreme_required": True,
}
OU_V5_COMPARATOR = {
    "comparator": "ou_v5",
    "generation": 5,
    "required_cells": 8,
    "packet": "wizard_ou_v5_review_packet.json",
    "packet_schema": "thewiz.wizard_ou_v5_review_packet.v1",
    "packet_prefix": "ouv5review_",
    "gate": "wizard_ou_v5_supersession_gate.json",
    "gate_schema": "thewiz.wizard_ou_v5_supersession_gate.v1",
    "activation": "wizard_ou_v5_activation_status.json",
    "activation_schema": "thewiz.wizard_ou_v5_activation.v1",
    "immutable_activation_root": "data/research/wizard_ou_v5_comparator_activations",
    "supreme": "wizard_ou_v5_review.json",
    "supreme_required": True,
}
OU_V6_COMPARATOR = {
    "comparator": "ou_v6",
    "generation": 6,
    "required_cells": 8,
    "packet": "wizard_ou_v6_review_packet.json",
    "packet_schema": "thewiz.wizard_ou_v6_review_packet.v1",
    "packet_prefix": "ouv6review_",
    "gate": "wizard_ou_v6_supersession_gate.json",
    "gate_schema": "thewiz.wizard_ou_v6_supersession_gate.v1",
    "activation": "wizard_ou_v6_activation_status.json",
    "activation_schema": "thewiz.wizard_ou_v6_activation.v1",
    "immutable_activation_root": "data/research/wizard_ou_v6_comparator_activations",
    "supreme": "wizard_ou_v6_review.json",
    "supreme_required": True,
}


def build_corrective_wizard_comparator_review_control(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    mutation_blocker_loader: Callable[..., list[str]] = (frozen_capture_manifest_mutation_blockers),
) -> CommandResult:
    """Publish one fail-closed queue for both explicit comparator reviews."""

    checked_at = _as_utc(now)
    active = root / "reports" / "active"
    supreme_dir = root / "reports" / "supreme_team"
    mutation_blockers = sorted(
        {str(value) for value in mutation_blocker_loader(root=root) if str(value)}
    )
    comparators = _comparators(root)
    rows = [
        _comparator_row(
            root=root,
            active=active,
            supreme_dir=supreme_dir,
            config=config,
            mutation_blockers=mutation_blockers,
        )
        for config in comparators
    ]

    control_blockers = sorted(
        {blocker for row in rows for blocker in str(row["control_blocker"]).split(";") if blocker}
    )
    review_ready = sum(_truthy(row["review_ready"]) for row in rows)
    apply_ready = sum(_truthy(row["apply_window_open"]) for row in rows)
    applied = sum(_truthy(row["activation_applied"]) for row in rows)
    if control_blockers:
        status = "BLOCKED_REVIEW_CONTROL"
        next_action = "repair_comparator_review_evidence_bindings"
    elif applied == len(rows):
        status = "PASS_RESEARCH_COMPARATORS_APPLIED"
        next_action = "refresh_activation_bound_dynamic_and_ou_proof_rows"
    elif apply_ready > 0:
        status = "READY_FOR_EXPLICIT_HUMAN_REVIEW"
        next_action = "review_exact_packet_ids_and_apply_research_comparators"
    elif review_ready > 0 and mutation_blockers:
        status = "WAITING_FOR_FROZEN_CAPTURE_RECONCILIATION"
        next_action = "complete_and_reconcile_frozen_stage3_capture_before_apply"
    else:
        status = "WAITING_FOR_PROSPECTIVE_EVIDENCE"
        next_action = "collect_and_evaluate_remaining_comparator_holdout_cells"

    findings = _supreme_findings(
        rows=rows,
        status=status,
        review_ready=review_ready,
        apply_ready=apply_ready,
        applied=applied,
        mutation_blockers=mutation_blockers,
        control_blockers=control_blockers,
    )
    recommendation = _recommendation(status)
    rows_path = active / "wizard_comparator_review_queue.csv"
    supreme_path = supreme_dir / "wizard_comparator_review_checkpoint.json"
    supreme_md_path = supreme_dir / "wizard_comparator_review_checkpoint.md"
    _write_csv(rows, rows_path)
    supreme_payload = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": status,
        "recommendation": recommendation,
        "findings": findings,
        "review_ready": review_ready,
        "apply_ready": apply_ready,
        "applied": applied,
        "comparators": len(rows),
        "control_blockers": control_blockers,
        "mutation_blockers": mutation_blockers,
        "activation_applied_by_supreme_team": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    _write_json(supreme_payload, supreme_path)
    _write_text(supreme_md_path, _supreme_markdown(supreme_payload))

    receipt_core = {
        "schema_version": SCHEMA_VERSION,
        "checked_at_utc": checked_at.isoformat(),
        "status": status,
        "next_action": next_action,
        "comparators": len(rows),
        "review_ready": review_ready,
        "apply_ready": apply_ready,
        "applied": applied,
        "control_blockers": control_blockers,
        "mutation_blockers": mutation_blockers,
        "queue_path": _relative(rows_path, root),
        "queue_sha256": _file_sha256(rows_path),
        "supreme_checkpoint_path": _relative(supreme_path, root),
        "supreme_checkpoint_sha256": _file_sha256(supreme_path),
        "supreme_recommendation": recommendation,
        "automatic_activation": False,
        "explicit_human_review_required": True,
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    receipt_id = (
        "comparatorreview_" + sha256(_canonical_json(receipt_core).encode("utf-8")).hexdigest()[:20]
    )
    immutable_path = (
        root / "data" / "research" / "wizard_comparator_review_controls" / f"{receipt_id}.json"
    )
    receipt = {**receipt_core, "receipt_id": receipt_id}
    _write_immutable_json(receipt, immutable_path)
    status_payload = {
        **receipt,
        "immutable_receipt_path": _relative(immutable_path, root),
        "immutable_receipt_sha256": _file_sha256(immutable_path),
    }
    status_path = active / "wizard_comparator_review_control.json"
    summary_path = active / "wizard_comparator_review_control.md"
    _write_json(status_payload, status_path)
    _write_text(summary_path, _control_markdown(status_payload, rows))
    paths = {
        "queue": rows_path,
        "status": status_path,
        "summary": summary_path,
        "supreme_checkpoint": supreme_path,
        "supreme_checkpoint_summary": supreme_md_path,
        "immutable_receipt": immutable_path,
    }
    return CommandResult(paths=paths, summary=status_payload)


def _comparator_row(
    *,
    root: Path,
    active: Path,
    supreme_dir: Path,
    config: dict[str, Any],
    mutation_blockers: list[str],
) -> dict[str, object]:
    packet_path = active / str(config["packet"])
    gate_path = active / str(config["gate"])
    activation_path = active / str(config["activation"])
    supreme_path = supreme_dir / str(config["supreme"])
    packet = _read_json(packet_path)
    gate = _read_json(gate_path)
    activation = _read_json(activation_path)
    supreme = _read_json(supreme_path)
    blockers: list[str] = []

    if packet.get("schema_version") != config["packet_schema"]:
        blockers.append(f"{config['comparator']}_review_packet_schema_invalid")
    if gate.get("schema_version") != config["gate_schema"]:
        blockers.append(f"{config['comparator']}_supersession_gate_schema_invalid")
    if activation.get("schema_version") != config["activation_schema"]:
        blockers.append(f"{config['comparator']}_activation_schema_invalid")
    if any(
        not _authority_is_zero(payload)
        for payload in (packet, gate, activation, supreme)
        if payload
    ):
        blockers.append(f"{config['comparator']}_authority_boundary_invalid")
    if packet.get("automatic_activation") is not False:
        blockers.append(f"{config['comparator']}_automatic_activation_not_false")
    if packet.get("explicit_review_required") is not True:
        blockers.append(f"{config['comparator']}_explicit_review_not_required")

    packet_status = _text(packet.get("status"))
    gate_status = _text(gate.get("status"))
    activation_status = _text(activation.get("status"))
    packet_id = _text(packet.get("review_packet_id"))
    packet_relative = _text(packet.get("immutable_review_packet_path"))
    packet_sha = _text(packet.get("immutable_review_packet_sha256"))
    packet_file = root / packet_relative
    packet_binding_valid = bool(
        packet_status == "READY_FOR_EXPLICIT_REVIEW"
        and packet_id.startswith(str(config["packet_prefix"]))
        and packet_relative
        and packet_file.is_file()
        and _file_sha256(packet_file) == packet_sha
    )
    review_ready = bool(
        packet_binding_valid
        and gate_status == "READY_FOR_REVIEWED_SUPERSESSION"
        and not packet.get("blockers")
        and not gate.get("blockers")
        and packet.get("all_cells_passed") is True
        and packet.get("raw_bindings_valid") is True
        and _safe_int(packet.get("required_cells")) == _safe_int(config["required_cells"])
    )
    waiting_for_evidence = bool(
        packet_status in {"WAITING_FOR_HOLDOUT", "BLOCKED_HOLDOUT_EVIDENCE", "BLOCKED"}
        and gate_status == "BLOCKED"
        and not packet_id
        and not packet_relative
        and activation_status == "BLOCKED"
        and activation.get("apply_requested") is False
    )
    if not review_ready and not waiting_for_evidence:
        blockers.append(f"{config['comparator']}_review_state_invalid")
    if packet_status == "READY_FOR_EXPLICIT_REVIEW" and not packet_binding_valid:
        blockers.append(f"{config['comparator']}_immutable_review_packet_binding_invalid")

    activation_applied = activation_status == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    if activation_applied:
        activation_packet_id = _text(activation.get("review_packet_id_submitted"))
        immutable_activation_relative = _text(activation.get("immutable_activation_path"))
        immutable_activation_path = root / immutable_activation_relative
        immutable_activation_valid = bool(
            activation.get("apply_requested") is True
            and activation_packet_id == packet_id
            and _text(activation.get("reviewer"))
            and len(_text(activation.get("review_note"))) >= 20
            and immutable_activation_relative.startswith(
                str(config["immutable_activation_root"]) + "/"
            )
            and immutable_activation_path.is_file()
            and _file_sha256(immutable_activation_path)
            == _text(activation.get("immutable_activation_sha256"))
        )
        if not immutable_activation_valid:
            blockers.append(f"{config['comparator']}_applied_activation_binding_invalid")
    elif activation.get("apply_requested") is not False:
        blockers.append(f"{config['comparator']}_unapplied_state_requested_apply")

    if supreme:
        supreme_packet_match = _text(supreme.get("review_packet_id")) == packet_id
        if review_ready and not supreme_packet_match:
            blockers.append(f"{config['comparator']}_supreme_packet_binding_invalid")
        supreme_binding_valid = _supreme_review_binding_valid(
            root=root,
            config=config,
            packet=packet,
        )
        if review_ready and config.get("supreme_required") and not supreme_binding_valid:
            blockers.append(f"{config['comparator']}_supreme_immutable_binding_invalid")
        if (
            review_ready
            and config.get("supreme_required")
            and (
                supreme.get("status") != "PASS_ADVISORY_ONLY"
                or _text(supreme.get("recommendation"))
                != "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
            )
        ):
            blockers.append(f"{config['comparator']}_supreme_review_not_passed")
        supreme_recommendation = _text(supreme.get("recommendation"))
    else:
        if review_ready and config.get("supreme_required"):
            blockers.append(f"{config['comparator']}_supreme_review_missing")
        supreme_recommendation = (
            "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
            if review_ready
            else "WAIT_FOR_PROSPECTIVE_EVIDENCE"
        )

    apply_window_open = bool(
        review_ready and not mutation_blockers and not blockers and not activation_applied
    )
    command_fields = _operator_command_fields(
        comparator=_text(config["comparator"]),
        packet_id=packet_id,
        review_ready=review_ready,
        apply_window_open=apply_window_open,
        activation_applied=activation_applied,
    )
    next_action = _row_next_action(
        review_ready=review_ready,
        apply_window_open=apply_window_open,
        activation_applied=activation_applied,
        mutation_blockers=mutation_blockers,
        blockers=blockers,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "comparator": config["comparator"],
        "generation": config["generation"],
        "packet_status": packet_status,
        "review_packet_id": packet_id,
        "review_packet_path": packet_relative,
        "review_packet_sha256": packet_sha,
        "packet_binding_valid": packet_binding_valid,
        "supersession_gate_status": gate_status,
        "activation_status": activation_status,
        "review_ready": review_ready,
        "apply_window_open": apply_window_open,
        "apply_window_blockers": ";".join(mutation_blockers),
        "activation_applied": activation_applied,
        "reviewer_recorded": bool(_text(activation.get("reviewer"))),
        "review_packet_id_submitted": _text(activation.get("review_packet_id_submitted")),
        "supreme_recommendation": supreme_recommendation,
        "explicit_review_required": True,
        "automatic_activation": False,
        **command_fields,
        "next_action": next_action,
        "control_blocker": ";".join(sorted(set(blockers))),
        "research_only": True,
        "candidate_promotion_authority": False,
        "order_submission_included": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _operator_command_fields(
    *,
    comparator: str,
    packet_id: str,
    review_ready: bool,
    apply_window_open: bool,
    activation_applied: bool,
) -> dict[str, str]:
    if activation_applied:
        return {
            "operator_command_status": "ALREADY_APPLIED",
            "preflight_command": "",
            "apply_command_template": "",
        }
    if not review_ready:
        return {
            "operator_command_status": "WAITING_FOR_REVIEW_EVIDENCE",
            "preflight_command": "",
            "apply_command_template": "",
        }

    command = {
        "dynamic_v2": "review-wizard-dynamic-v2",
        "ou_v3": "review-wizard-ou-v3",
        "ou_v4": "review-wizard-ou-v4",
        "ou_v5": "review-wizard-ou-v5",
        "ou_v6": "review-wizard-ou-v6",
    }[comparator]
    option_prefix = comparator.replace("_", "-")
    base = (
        "PYTHONPATH=src .venv/bin/python3 -m quant_platform.cli "
        f"{command} --{option_prefix}-review-packet-id {packet_id}"
    )
    if not apply_window_open:
        return {
            "operator_command_status": "PREFLIGHT_ONLY_MUTATION_WINDOW_CLOSED",
            "preflight_command": base,
            "apply_command_template": "",
        }
    return {
        "operator_command_status": "READY_FOR_EXPLICIT_HUMAN_APPLY",
        "preflight_command": base,
        "apply_command_template": (
            f"{base} --apply-{option_prefix} "
            f"--{option_prefix}-reviewer '<reviewer>' "
            f"--{option_prefix}-review-note '<substantive review note at least 20 characters>'"
        ),
    }


def _supreme_review_binding_valid(
    *,
    root: Path,
    config: dict[str, Any],
    packet: dict[str, Any],
) -> bool:
    comparator = _text(config.get("comparator"))
    kwargs = {
        "root": root,
        "expected_review_packet_id": packet.get("review_packet_id"),
        "expected_review_packet_path": packet.get("immutable_review_packet_path"),
        "expected_review_packet_sha256": packet.get("immutable_review_packet_sha256"),
    }
    try:
        if comparator == "dynamic_v2":
            from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
                load_validated_dynamic_v2_supreme_review,
            )

            return load_validated_dynamic_v2_supreme_review(**kwargs) is not None
        if comparator == "ou_v4":
            from quant_platform.orchestration.corrective_wizard_ou_v4_supreme_review import (
                load_validated_ou_v4_supreme_review,
            )

            return load_validated_ou_v4_supreme_review(**kwargs) is not None
        if comparator == "ou_v5":
            from quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review import (
                load_validated_ou_v5_supreme_review,
            )

            return load_validated_ou_v5_supreme_review(**kwargs) is not None
        if comparator == "ou_v6":
            from quant_platform.orchestration.corrective_wizard_ou_v6_supreme_review import (
                load_validated_ou_v6_supreme_review,
            )

            return load_validated_ou_v6_supreme_review(**kwargs) is not None
    except (OSError, TypeError, ValueError):
        return False
    return not bool(config.get("supreme_required"))


def _comparators(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    active = root / "reports" / "active"
    v6_registered = bool(
        (root / "config" / "wizard_ou_comparator_v6_holdout.json").is_file()
        or (active / "wizard_ou_v6_holdout_receipt.json").is_file()
    )
    v5_registered = bool(
        (root / "config" / "wizard_ou_comparator_v5_holdout.json").is_file()
        or (active / "wizard_ou_v5_holdout_receipt.json").is_file()
    )
    v4_registered = bool(
        (root / "config" / "wizard_ou_comparator_v4_holdout.json").is_file()
        or (active / "wizard_ou_v4_holdout_receipt.json").is_file()
    )
    return (
        DYNAMIC_V2_COMPARATOR,
        OU_V6_COMPARATOR
        if v6_registered
        else OU_V5_COMPARATOR
        if v5_registered
        else OU_V4_COMPARATOR
        if v4_registered
        else OU_V3_COMPARATOR,
    )


def _row_next_action(
    *,
    review_ready: bool,
    apply_window_open: bool,
    activation_applied: bool,
    mutation_blockers: list[str],
    blockers: list[str],
) -> str:
    if blockers:
        return "repair_review_control_evidence"
    if activation_applied:
        return "refresh_activation_bound_proof_rows"
    if apply_window_open:
        return "human_reviews_exact_packet_and_explicitly_applies"
    if review_ready and mutation_blockers:
        return "wait_for_frozen_capture_reconciliation"
    return "wait_for_prospective_holdout_evidence"


def _supreme_findings(
    *,
    rows: list[dict[str, object]],
    status: str,
    review_ready: int,
    apply_ready: int,
    applied: int,
    mutation_blockers: list[str],
    control_blockers: list[str],
) -> list[dict[str, str]]:
    return [
        {
            "lens": "gap_analysis",
            "severity": "medium",
            "finding": (
                f"{review_ready} of {len(rows)} comparators have immutable review-ready evidence; "
                f"{apply_ready} currently have an open apply window."
            ),
            "required_action": "Complete the missing holdout or reconciliation evidence.",
        },
        {
            "lens": "pre_mortem",
            "severity": "high",
            "finding": "A stale packet ID or pre-reconciliation apply could corrupt Stage 3 lineage.",
            "required_action": "Require the current immutable packet ID and an open mutation window.",
        },
        {
            "lens": "post_mortem",
            "severity": "low" if not control_blockers else "high",
            "finding": (
                f"Current control status is {status}; {applied} research comparators are applied."
            ),
            "required_action": "Preserve this receipt with the next parity refresh.",
        },
        {
            "lens": "red_team",
            "severity": "critical",
            "finding": (
                "Comparator activation could be misrepresented as strategy or order approval."
                + (
                    f" Mutation blockers: {','.join(mutation_blockers)}."
                    if mutation_blockers
                    else ""
                )
            ),
            "required_action": "Keep promotion, Testnet-order, and live authority false.",
        },
    ]


def _recommendation(status: str) -> str:
    return {
        "BLOCKED_REVIEW_CONTROL": "REPAIR_EVIDENCE_DO_NOT_APPLY",
        "PASS_RESEARCH_COMPARATORS_APPLIED": "REFRESH_PROOFS_RESEARCH_ONLY",
        "READY_FOR_EXPLICIT_HUMAN_REVIEW": ("RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_REVIEW"),
        "WAITING_FOR_FROZEN_CAPTURE_RECONCILIATION": (
            "WAIT_FOR_CAPTURE_RECONCILIATION_DO_NOT_APPLY"
        ),
    }.get(status, "WAIT_FOR_PROSPECTIVE_EVIDENCE_DO_NOT_APPLY")


def _authority_is_zero(payload: dict[str, Any]) -> bool:
    return all(
        not _truthy(payload.get(field))
        for field in (
            "candidate_promotion_authority",
            "promotion_authority",
            "order_submission_included",
            "testnet_order_authority",
            "live_trading_authorized",
            "activation_applied_by_supreme_team",
        )
        if field in payload
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    promote_staged_file(temporary, path)


def _write_json(payload: dict[str, Any], path: Path) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable comparator review receipt collision: {path}")
        return
    _write_text(path, encoded)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    promote_staged_file(temporary, path)


def _control_markdown(payload: dict[str, Any], rows: list[dict[str, object]]) -> str:
    lines = [
        "# Wizard Comparator Review Control",
        "",
        f"- Status: `{payload['status']}`",
        f"- Review ready: `{payload['review_ready']}/{payload['comparators']}`",
        f"- Apply ready: `{payload['apply_ready']}/{payload['comparators']}`",
        f"- Applied: `{payload['applied']}/{payload['comparators']}`",
        f"- Receipt: `{payload['receipt_id']}`",
        f"- Next action: `{payload['next_action']}`",
        "- Activation remains explicit, human-reviewed, and research-only.",
        "- This control grants no candidate, Testnet-order, or live authority.",
        "",
        "| Comparator | Packet | Review Ready | Apply Window | Activation | Next Action | Blocker |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        "| {comparator} | {packet} | {review} | {apply} | {activation} | {next_action} | {blocker} |".format(
            comparator=row["comparator"],
            packet=row["review_packet_id"] or row["packet_status"],
            review=row["review_ready"],
            apply=row["apply_window_open"],
            activation=row["activation_status"],
            next_action=row["next_action"],
            blocker=row["control_blocker"],
        )
        for row in rows
    )
    operator_rows = [row for row in rows if row["preflight_command"]]
    if operator_rows:
        lines.extend(("", "## Explicit Review Commands", ""))
        for row in operator_rows:
            lines.extend(
                (
                    f"### {row['comparator']}",
                    "",
                    f"- Command status: `{row['operator_command_status']}`",
                    "- Preflight:",
                    "",
                    "```bash",
                    str(row["preflight_command"]),
                    "```",
                )
            )
            if row["apply_command_template"]:
                lines.extend(
                    (
                        "- Explicit apply template:",
                        "",
                        "```bash",
                        str(row["apply_command_template"]),
                        "```",
                    )
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def _supreme_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Supreme Team Comparator Review Checkpoint",
        "",
        f"- Status: `{payload['status']}`",
        f"- Recommendation: `{payload['recommendation']}`",
        "- Advisory only; no activation or order authority is granted.",
        "",
    ]
    for finding in payload["findings"]:
        lines.extend(
            (
                f"## {str(finding['lens']).replace('_', ' ').title()}",
                "",
                f"- Severity: `{finding['severity']}`",
                f"- Finding: {finding['finding']}",
                f"- Required action: {finding['required_action']}",
                "",
            )
        )
    return "\n".join(lines)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_utc(value: datetime | None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)
