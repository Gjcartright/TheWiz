"""Advisory Supreme Team review for OU-v6 comparator activation."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_runtime import atomic_write_text, promote_staged_file
from quant_platform.wizard_ou_v6_comparator_activation import (
    build_ou_v6_review_packet,
    build_reviewed_ou_v6_activation,
)

ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "thewiz.wizard_ou_v6_supreme_review.v1"
AUTHORITY_FIELDS = (
    "candidate_promotion_authority",
    "testnet_order_authority",
    "live_trading_authorized",
)


def build_ou_v6_supreme_review(
    *,
    root: Path = ROOT,
    now: datetime | None = None,
    packet_builder: Callable[..., CommandResult] = build_ou_v6_review_packet,
    activation_planner: Callable[..., CommandResult] = (build_reviewed_ou_v6_activation),
) -> CommandResult:
    """Review the immutable eight-cell packet without applying authority."""

    timestamp = _as_utc(now)
    packet_result = packet_builder(root=root, now=timestamp)
    packet = packet_result.summary
    activation_result = activation_planner(root=root, apply=False, now=timestamp)
    activation = activation_result.summary
    blockers: list[str] = []
    packet_id = _text(packet.get("review_packet_id"))
    immutable_path = _resolve(root, packet.get("immutable_review_packet_path"))
    immutable_hash = _text(packet.get("immutable_review_packet_sha256"))
    immutable: dict[str, Any] = {}
    if packet.get("status") != "READY_FOR_EXPLICIT_REVIEW":
        blockers.append("ou_v6_packet_not_ready_for_explicit_review")
    if not packet_id:
        blockers.append("ou_v6_review_packet_id_missing")
    if immutable_path is None or len(immutable_hash) != 64:
        blockers.append("ou_v6_immutable_review_packet_missing")
    elif _file_hash(immutable_path) != immutable_hash:
        blockers.append("ou_v6_immutable_review_packet_hash_mismatch")
    else:
        immutable = _read_json(immutable_path)
        if _text(immutable.get("review_packet_id")) != packet_id:
            blockers.append("ou_v6_immutable_review_packet_id_mismatch")

    contract = _read_json(root / "config" / "wizard_ou_comparator_v6_holdout.json")
    expected = {
        (
            _text(binding.get("pair_group")),
            _text(binding.get("exact_mode")),
            _text(binding.get("orientation")),
        )
        for binding in contract.get("holdout_bindings", [])
        if isinstance(binding, dict)
    }
    cells = packet.get("cells") if isinstance(packet.get("cells"), list) else []
    identities = {
        (
            _text(cell.get("pair_group")),
            _text(cell.get("exact_mode")),
            _text(cell.get("orientation")),
        )
        for cell in cells
        if isinstance(cell, dict)
    }
    if len(expected) != 8 or identities != expected:
        blockers.append("ou_v6_review_cell_identity_mismatch")
    if not _truthy(packet.get("cohorts_disjoint")):
        blockers.append("ou_v6_review_cohorts_not_disjoint")
    if not _truthy(packet.get("all_cells_passed")):
        blockers.append("ou_v6_review_cells_not_all_passed")
    if not _truthy(packet.get("raw_bindings_valid")):
        blockers.append("ou_v6_review_raw_bindings_invalid")
    if packet.get("transform_scale_sensitive") is not True:
        blockers.append("ou_v6_scale_sensitivity_disclosure_missing")
    if packet.get("profile_branch_scale_sensitive") is not True:
        blockers.append("ou_v6_profile_branch_scale_sensitivity_disclosure_missing")
    if packet.get("profile_branch_rule_is_fragile") is not True:
        blockers.append("ou_v6_profile_branch_fragility_disclosure_missing")
    if (
        not _text(packet.get("profile_branch_rule"))
        or packet.get("trend_is_diagnostic_not_formula_switch") is not True
    ):
        blockers.append("ou_v6_profile_selector_contract_invalid")
    if (
        packet.get("final_successor_iteration") is not True
        or packet.get("successor_after_v6_failure_allowed") is not False
    ):
        blockers.append("ou_v6_terminal_no_successor_contract_invalid")

    max_error = 0.0
    max_tolerance_ratio = 0.0
    tolerances = contract.get("tolerances", {})
    error_limits = {
        "hedge_ratio_abs_error": tolerances.get("hedge_ratio_abs"),
        "spread_max_abs_error": tolerances.get("spread_max_abs"),
        "zscore_max_abs_error": tolerances.get("zscore_max_abs"),
        "zscore_roll_max_abs_error": tolerances.get("zscore_roll_max_abs"),
        "half_life_abs_error": tolerances.get("half_life_abs"),
    }
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cell_passed = bool(
            cell.get("cell_status") == "PASS"
            and _truthy(cell.get("formula_parity_passed"))
            and _truthy(cell.get("transform_selector_parity_passed"))
            and _truthy(cell.get("trend_selector_parity_passed"))
            and _truthy(cell.get("profile_branch_selector_parity_passed"))
            and _truthy(cell.get("raw_binding_valid"))
        )
        if not cell_passed:
            blockers.append("ou_v6_review_cell_failed")
        for field, raw_limit in error_limits.items():
            try:
                error = float(cell.get(field, "nan"))
                limit = float(raw_limit)
            except (TypeError, ValueError):
                error = float("inf")
                limit = 0.0
            max_error = max(max_error, error)
            ratio = error / limit if math.isfinite(error) and limit > 0.0 else float("inf")
            max_tolerance_ratio = max(max_tolerance_ratio, ratio)
    if not math.isfinite(max_tolerance_ratio) or max_tolerance_ratio > 1.0:
        blockers.append("ou_v6_review_numerical_tolerance_exceeded")

    if activation.get("status") not in {
        "READY_REQUIRES_EXPLICIT_APPLY",
        "APPLIED_RESEARCH_COMPARATOR_ONLY",
    }:
        blockers.append("ou_v6_activation_preflight_not_ready")
    if any(
        _truthy(payload.get(field))
        for payload in (packet, immutable, activation)
        for field in AUTHORITY_FIELDS
    ):
        blockers.append("ou_v6_review_evidence_exceeded_research_authority")
    if _truthy(activation.get("apply_requested")):
        blockers.append("ou_v6_supreme_review_must_not_apply_activation")

    passed_cells = sum(
        isinstance(cell, dict) and cell.get("cell_status") == "PASS" for cell in cells
    )
    recommendation = (
        "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
        if not blockers and activation.get("status") == "READY_REQUIRES_EXPLICIT_APPLY"
        else "ALREADY_APPLIED_RESEARCH_COMPARATOR_ONLY"
        if not blockers
        else "DO_NOT_ACTIVATE"
    )
    findings = [
        {
            "lens": "gap_analysis",
            "severity": "medium" if not blockers else "critical",
            "finding": (
                "The eight-cell v6 formula and three-selector evidence chain is complete; "
                "explicit review remains."
                if not blockers
                else "One or more OU v6 activation bindings are incomplete."
            ),
            "required_action": (
                "A named human reviewer must approve this exact packet ID."
                if not blockers
                else "Resolve every blocker and rebuild the packet."
            ),
        },
        {
            "lens": "pre_mortem",
            "severity": "high",
            "finding": (
                "A comparator could pass numerically while using the wrong transform, "
                "trend diagnostic, profile branch, or stale packet."
            ),
            "required_action": (
                "Bind all eight formula, transform, trend, profile-branch, raw, and "
                "packet identities."
            ),
        },
        {
            "lens": "post_mortem",
            "severity": (
                "low"
                if not blockers
                else "medium"
                if packet.get("status") == "WAITING_FOR_HOLDOUT"
                else "high"
            ),
            "finding": (
                f"The prospective holdout passed {passed_cells}/8 cells with maximum "
                f"absolute reconstruction error {max_error:.3e}."
                if packet.get("status") != "WAITING_FOR_HOLDOUT"
                else "The terminal OU v6 holdout is waiting for all eight vendor responses."
            ),
            "required_action": (
                "Refresh existing OU proof rows only after explicit activation."
                if not blockers
                else "Wait for the governed capture window; do not activate from missing evidence."
                if packet.get("status") == "WAITING_FOR_HOLDOUT"
                else "Preserve the failed holdout as negative evidence and do not activate v6."
            ),
        },
        {
            "lens": "red_team",
            "severity": "critical",
            "finding": (
                "Research comparator approval could be misrepresented as strategy or "
                "trading approval."
            ),
            "required_action": (
                "Keep all authority flags false and require downstream Stage 3-7 gates."
            ),
        },
    ]
    stable = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS_ADVISORY_ONLY" if not blockers else "BLOCKED",
        "recommendation": recommendation,
        "blockers": list(dict.fromkeys(blockers)),
        "review_packet_id": packet_id,
        "review_packet_path": _relative(packet_result.paths.get("review_packet", Path("")), root),
        "immutable_review_packet_path": (
            _relative(immutable_path, root) if immutable_path is not None else ""
        ),
        "immutable_review_packet_sha256": immutable_hash,
        "activation_preflight_status": _text(activation.get("status")),
        "required_cells": 8,
        "passed_cells": passed_cells,
        "formula_cells_passed": sum(
            isinstance(cell, dict) and _truthy(cell.get("formula_parity_passed")) for cell in cells
        ),
        "transform_selector_cells_passed": sum(
            isinstance(cell, dict) and _truthy(cell.get("transform_selector_parity_passed"))
            for cell in cells
        ),
        "trend_selector_cells_passed": sum(
            isinstance(cell, dict) and _truthy(cell.get("trend_selector_parity_passed"))
            for cell in cells
        ),
        "profile_branch_selector_cells_passed": sum(
            isinstance(cell, dict) and _truthy(cell.get("profile_branch_selector_parity_passed"))
            for cell in cells
        ),
        "profile_branch_rule": _text(packet.get("profile_branch_rule")),
        "transform_scale_sensitive": packet.get("transform_scale_sensitive") is True,
        "profile_branch_scale_sensitive": (
            packet.get("profile_branch_scale_sensitive") is True
        ),
        "profile_branch_rule_is_fragile": (
            packet.get("profile_branch_rule_is_fragile") is True
        ),
        "trend_is_diagnostic_not_formula_switch": (
            packet.get("trend_is_diagnostic_not_formula_switch") is True
        ),
        "final_successor_iteration": packet.get("final_successor_iteration") is True,
        "successor_after_v6_failure_allowed": (
            packet.get("successor_after_v6_failure_allowed") is True
        ),
        "cohorts_disjoint": _truthy(packet.get("cohorts_disjoint")),
        "raw_bindings_valid": _truthy(packet.get("raw_bindings_valid")),
        "max_abs_reconstruction_error": max_error,
        "max_tolerance_ratio": max_tolerance_ratio,
        "findings": findings,
        "human_approval_required": True,
        "human_approval_recorded": False,
        "activation_applied_by_supreme_team": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    review_id = "ouv6supreme_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    immutable_review = {**stable, "review_id": review_id}
    immutable_review_path = (
        root / "data" / "research" / "wizard_ou_v6_supreme_reviews" / f"{review_id}.json"
    )
    _write_or_validate_immutable_json(immutable_review, immutable_review_path)
    payload = {
        **immutable_review,
        "generated_at_utc": timestamp.isoformat(),
        "immutable_review_path": _relative(immutable_review_path, root),
        "immutable_review_sha256": _file_hash(immutable_review_path),
    }
    output_dir = root / "reports" / "supreme_team"
    status_path = output_dir / "wizard_ou_v6_review.json"
    markdown_path = output_dir / "wizard_ou_v6_review.md"
    _atomic_json(payload, status_path)
    _atomic_text(_markdown(payload), markdown_path)
    return CommandResult(
        paths={
            "status": status_path,
            "markdown": markdown_path,
            "immutable_review": immutable_review_path,
        },
        summary=payload,
    )


def validate_ou_v6_supreme_review_receipt(
    *,
    root: Path = ROOT,
    immutable_review_path: object,
    immutable_review_sha256: object,
    expected_review_packet_id: object,
    expected_review_packet_path: object,
    expected_review_packet_sha256: object,
) -> dict[str, Any]:
    """Validate one immutable, activation-recommending Supreme Team receipt."""

    review_path = _resolve(root, immutable_review_path)
    review_hash = _text(immutable_review_sha256).lower()
    if review_path is None or len(review_hash) != 64 or _file_hash(review_path) != review_hash:
        raise ValueError("OU v6 Supreme Team immutable review binding mismatch")
    review = _read_json(review_path)
    review_id = _text(review.get("review_id"))
    expected_root = (root / "data" / "research" / "wizard_ou_v6_supreme_reviews").resolve()
    try:
        path_valid = bool(
            review_id.startswith("ouv6supreme_")
            and review_path.resolve().is_relative_to(expected_root)
            and review_path.name == f"{review_id}.json"
        )
    except (OSError, ValueError):
        path_valid = False
    stable = dict(review)
    stable.pop("review_id", None)
    expected_review_id = (
        "ouv6supreme_" + sha256(_canonical_json(stable).encode("utf-8")).hexdigest()[:20]
    )
    if not path_valid or review_id != expected_review_id:
        raise ValueError("OU v6 Supreme Team review identity mismatch")

    packet_id = _text(expected_review_packet_id)
    packet_path = _text(expected_review_packet_path)
    packet_hash = _text(expected_review_packet_sha256).lower()
    if (
        not packet_id.startswith("ouv6review_")
        or review.get("review_packet_id") != packet_id
        or review.get("immutable_review_packet_path") != packet_path
        or _text(review.get("immutable_review_packet_sha256")).lower() != packet_hash
    ):
        raise ValueError("OU v6 Supreme Team review packet binding mismatch")
    bound_packet = _resolve(root, packet_path)
    if (
        bound_packet is None
        or len(packet_hash) != 64
        or _file_hash(bound_packet) != packet_hash
        or _text(_read_json(bound_packet).get("review_packet_id")) != packet_id
    ):
        raise ValueError("OU v6 Supreme Team immutable packet binding mismatch")

    required_semantics = bool(
        review.get("schema_version") == SCHEMA_VERSION
        and review.get("status") == "PASS_ADVISORY_ONLY"
        and review.get("recommendation") == "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
        and review.get("activation_preflight_status") == "READY_REQUIRES_EXPLICIT_APPLY"
        and not review.get("blockers")
        and int(review.get("required_cells", 0) or 0) == 8
        and int(review.get("passed_cells", 0) or 0) == 8
        and int(review.get("formula_cells_passed", 0) or 0) == 8
        and int(review.get("transform_selector_cells_passed", 0) or 0) == 8
        and int(review.get("trend_selector_cells_passed", 0) or 0) == 8
        and int(review.get("profile_branch_selector_cells_passed", 0) or 0) == 8
        and _truthy(review.get("cohorts_disjoint"))
        and _truthy(review.get("raw_bindings_valid"))
        and review.get("transform_scale_sensitive") is True
        and review.get("profile_branch_scale_sensitive") is True
        and review.get("profile_branch_rule_is_fragile") is True
        and _text(review.get("profile_branch_rule"))
        and review.get("trend_is_diagnostic_not_formula_switch") is True
        and review.get("final_successor_iteration") is True
        and review.get("successor_after_v6_failure_allowed") is False
        and float(review.get("max_tolerance_ratio", "inf")) <= 1.0
        and review.get("human_approval_required") is True
        and review.get("human_approval_recorded") is False
        and review.get("activation_applied_by_supreme_team") is False
        and review.get("research_only") is True
        and not any(_truthy(review.get(field)) for field in AUTHORITY_FIELDS)
    )
    if not required_semantics:
        raise ValueError("OU v6 Supreme Team review semantics invalid")
    return review


def load_validated_ou_v6_supreme_review(
    *,
    root: Path = ROOT,
    expected_review_packet_id: object,
    expected_review_packet_path: object,
    expected_review_packet_sha256: object,
) -> dict[str, Any] | None:
    """Load the current Supreme pointer only when its immutable receipt is valid."""

    status_path = root / "reports" / "supreme_team" / "wizard_ou_v6_review.json"
    if not status_path.is_file():
        return None
    pointer = _read_json(status_path)
    if not pointer:
        raise ValueError("OU v6 Supreme Team review pointer is unreadable")
    immutable = validate_ou_v6_supreme_review_receipt(
        root=root,
        immutable_review_path=pointer.get("immutable_review_path"),
        immutable_review_sha256=pointer.get("immutable_review_sha256"),
        expected_review_packet_id=expected_review_packet_id,
        expected_review_packet_path=expected_review_packet_path,
        expected_review_packet_sha256=expected_review_packet_sha256,
    )
    for key, value in immutable.items():
        if pointer.get(key) != value:
            raise ValueError(f"OU v6 Supreme Team active pointer mismatch: {key}")
    return pointer


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OU v6 Supreme Team Review",
        "",
        f"- Status: `{payload['status']}`",
        f"- Recommendation: `{payload['recommendation']}`",
        f"- Packet: `{payload['review_packet_id']}`",
        f"- Holdout cells: `{payload['passed_cells']}/8`",
        f"- Profile branch cells: `{payload['profile_branch_selector_cells_passed']}/8`",
        f"- Maximum reconstruction error: `{payload['max_abs_reconstruction_error']:.3e}`",
        f"- Maximum registered-tolerance ratio: `{payload['max_tolerance_ratio']:.6f}`",
        "- Terminal successor generation: `true`",
        "- Successor after failure allowed: `false`",
        "- Human approval recorded: `false`",
        "- Activation performed here: `false`",
        "- Candidate/Testnet/live authority: `false`",
        "",
        "| Lens | Severity | Finding | Required action |",
        "|---|---|---|---|",
    ]
    for finding in payload["findings"]:
        lines.append("| {lens} | {severity} | {finding} | {required_action} |".format(**finding))
    if payload["blockers"]:
        lines.extend(["", "## Blockers", *[f"- `{item}`" for item in payload["blockers"]]])
    return "\n".join(lines) + "\n"


def _as_utc(now: datetime | None) -> datetime:
    value = now or datetime.now(UTC)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_or_validate_immutable_json(payload: dict[str, Any], path: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"immutable OU v6 Supreme Team review changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, encoded, encoding="utf-8")


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)


def _atomic_text(payload: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    promote_staged_file(temporary, path)


def _file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, value: object) -> Path | None:
    text = _text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _relative(path: Path, root: Path) -> str:
    if not str(path):
        return ""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _text(value: object) -> str:
    return str(value or "").strip()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).lower() in {"1", "true", "yes", "y"}
