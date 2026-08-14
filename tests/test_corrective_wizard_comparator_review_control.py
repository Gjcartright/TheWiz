from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest

from quant_platform.orchestration.corrective_wizard_comparator_review_control import (
    _comparators,
    build_corrective_wizard_comparator_review_control,
)

NOW = datetime(2026, 8, 11, 17, 0, tzinfo=UTC)


def test_review_control_selects_registered_terminal_ou_v6_generation(tmp_path: Path) -> None:
    contract = tmp_path / "config" / "wizard_ou_comparator_v6_holdout.json"
    contract.parent.mkdir(parents=True)
    contract.write_text("{}\n", encoding="utf-8")

    dynamic, ou = _comparators(tmp_path)

    assert dynamic["comparator"] == "dynamic_v2"
    assert ou["comparator"] == "ou_v6"
    assert ou["generation"] == 6
    assert ou["supreme_required"] is True


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ready_packet(root: Path, comparator: str) -> dict[str, object]:
    if comparator == "dynamic_v2":
        packet_id = "dynamicv2review_0123456789abcdef0123"
        schema = "thewiz.wizard_dynamic_v2_review_packet.v1"
        directory = "wizard_dynamic_review_packets"
        required_cells = 4
    elif comparator == "ou_v3":
        packet_id = "ouv3review_0123456789abcdef0123"
        schema = "thewiz.wizard_ou_v3_review_packet.v1"
        directory = "wizard_ou_v3_review_packets"
        required_cells = 4
    elif comparator == "ou_v4":
        packet_id = "ouv4review_0123456789abcdef0123"
        schema = "thewiz.wizard_ou_v4_review_packet.v1"
        directory = "wizard_ou_v4_review_packets"
        required_cells = 8
    elif comparator == "ou_v5":
        packet_id = "ouv5review_0123456789abcdef0123"
        schema = "thewiz.wizard_ou_v5_review_packet.v1"
        directory = "wizard_ou_v5_review_packets"
        required_cells = 8
    else:
        packet_id = "ouv6review_0123456789abcdef0123"
        schema = "thewiz.wizard_ou_v6_review_packet.v1"
        directory = "wizard_ou_v6_review_packets"
        required_cells = 8
    relative = f"data/research/{directory}/{packet_id}.json"
    immutable = root / relative
    _write_json(immutable, {"review_packet_id": packet_id, "frozen": True})
    return {
        "schema_version": schema,
        "status": "READY_FOR_EXPLICIT_REVIEW",
        "review_packet_id": packet_id,
        "immutable_review_packet_path": relative,
        "immutable_review_packet_sha256": sha256(immutable.read_bytes()).hexdigest(),
        "required_cells": required_cells,
        "all_cells_passed": True,
        "raw_bindings_valid": True,
        "blockers": [],
        "explicit_review_required": True,
        "automatic_activation": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }


def _ready_dynamic_supreme(root: Path, packet: dict[str, object]) -> dict[str, object]:
    stable: dict[str, object] = {
        "schema_version": "thewiz.wizard_dynamic_v2_supreme_review.v1",
        "status": "PASS_ADVISORY_ONLY",
        "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
        "blockers": [],
        "review_packet_id": packet["review_packet_id"],
        "review_packet_path": "reports/active/wizard_dynamic_v2_review_packet.json",
        "immutable_review_packet_path": packet["immutable_review_packet_path"],
        "immutable_review_packet_sha256": packet["immutable_review_packet_sha256"],
        "activation_preflight_status": "READY_REQUIRES_EXPLICIT_APPLY",
        "required_cells": 4,
        "passed_cells": 4,
        "cohorts_disjoint": True,
        "raw_bindings_valid": True,
        "max_abs_reconstruction_error": 0.0,
        "findings": [],
        "human_approval_required": True,
        "human_approval_recorded": False,
        "activation_applied_by_supreme_team": False,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    review_id = (
        "dynamicv2supreme_"
        + sha256(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()[:20]
    )
    immutable_relative = f"data/research/wizard_dynamic_supreme_reviews/{review_id}.json"
    immutable_path = root / immutable_relative
    immutable = {**stable, "review_id": review_id}
    _write_json(immutable_path, immutable)
    return {
        **immutable,
        "generated_at_utc": NOW.isoformat(),
        "immutable_review_path": immutable_relative,
        "immutable_review_sha256": sha256(immutable_path.read_bytes()).hexdigest(),
    }


def _prepare_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    active = root / "reports" / "active"
    supreme = root / "reports" / "supreme_team"
    active.mkdir(parents=True)
    supreme.mkdir(parents=True)

    dynamic = _ready_packet(root, "dynamic_v2")
    _write_json(active / "wizard_dynamic_v2_review_packet.json", dynamic)
    _write_json(
        active / "wizard_dynamic_v2_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_dynamic_comparator_supersession_gate.v1",
            "status": "READY_FOR_REVIEWED_SUPERSESSION",
            "blockers": [],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_dynamic_v2_activation_status.json",
        {
            "schema_version": "thewiz.wizard_dynamic_v2_activation.v2",
            "status": "READY_REQUIRES_EXPLICIT_APPLY",
            "apply_requested": False,
            "reviewer": "",
            "review_packet_id_submitted": "",
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        supreme / "wizard_dynamic_v2_review.json",
        _ready_dynamic_supreme(root, dynamic),
    )

    _write_json(
        active / "wizard_ou_v3_review_packet.json",
        {
            "schema_version": "thewiz.wizard_ou_v3_review_packet.v1",
            "status": "WAITING_FOR_HOLDOUT",
            "review_packet_id": "",
            "immutable_review_packet_path": "",
            "immutable_review_packet_sha256": "",
            "required_cells": 4,
            "all_cells_passed": False,
            "raw_bindings_valid": True,
            "blockers": ["ou_v3_prospective_holdout_not_passed"],
            "explicit_review_required": True,
            "automatic_activation": False,
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_ou_v3_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_ou_v3_supersession_gate.v1",
            "status": "BLOCKED",
            "blockers": ["ou_v3_prospective_holdout_not_passed"],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_ou_v3_activation_status.json",
        {
            "schema_version": "thewiz.wizard_ou_v3_activation.v1",
            "status": "BLOCKED",
            "apply_requested": False,
            "reviewer": "",
            "review_packet_id_submitted": "",
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return root


def _run(root: Path, blockers: list[str] | None = None):
    return build_corrective_wizard_comparator_review_control(
        root=root,
        now=NOW,
        mutation_blocker_loader=lambda **_: list(blockers or []),
    )


def _make_ou_ready(root: Path) -> None:
    active = root / "reports" / "active"
    packet = _ready_packet(root, "ou_v3")
    _write_json(active / "wizard_ou_v3_review_packet.json", packet)
    _write_json(
        active / "wizard_ou_v3_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_ou_v3_supersession_gate.v1",
            "status": "READY_FOR_REVIEWED_SUPERSESSION",
            "blockers": [],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    activation = _read_json(active / "wizard_ou_v3_activation_status.json")
    activation["status"] = "READY_REQUIRES_EXPLICIT_APPLY"
    _write_json(active / "wizard_ou_v3_activation_status.json", activation)


def _make_ou_v4_ready(root: Path) -> dict[str, object]:
    active = root / "reports" / "active"
    supreme = root / "reports" / "supreme_team"
    config = root / "config" / "wizard_ou_comparator_v4_holdout.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}\n", encoding="utf-8")
    packet = _ready_packet(root, "ou_v4")
    _write_json(active / "wizard_ou_v4_review_packet.json", packet)
    _write_json(
        active / "wizard_ou_v4_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_ou_v4_supersession_gate.v1",
            "status": "READY_FOR_REVIEWED_SUPERSESSION",
            "blockers": [],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_ou_v4_activation_status.json",
        {
            "schema_version": "thewiz.wizard_ou_v4_activation.v2",
            "status": "READY_REQUIRES_EXPLICIT_APPLY",
            "apply_requested": False,
            "reviewer": "",
            "review_packet_id_submitted": "",
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        supreme / "wizard_ou_v4_review.json",
        {
            "schema_version": "thewiz.wizard_ou_v4_supreme_review.v1",
            "status": "PASS_ADVISORY_ONLY",
            "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
            "review_packet_id": packet["review_packet_id"],
            "candidate_promotion_authority": False,
            "activation_applied_by_supreme_team": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return packet


def _make_ou_v5_ready(root: Path) -> dict[str, object]:
    active = root / "reports" / "active"
    supreme = root / "reports" / "supreme_team"
    config = root / "config" / "wizard_ou_comparator_v5_holdout.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}\n", encoding="utf-8")
    packet = _ready_packet(root, "ou_v5")
    _write_json(active / "wizard_ou_v5_review_packet.json", packet)
    _write_json(
        active / "wizard_ou_v5_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_ou_v5_supersession_gate.v1",
            "status": "READY_FOR_REVIEWED_SUPERSESSION",
            "blockers": [],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_ou_v5_activation_status.json",
        {
            "schema_version": "thewiz.wizard_ou_v5_activation.v1",
            "status": "READY_REQUIRES_EXPLICIT_APPLY",
            "apply_requested": False,
            "reviewer": "",
            "review_packet_id_submitted": "",
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        supreme / "wizard_ou_v5_review.json",
        {
            "schema_version": "thewiz.wizard_ou_v5_supreme_review.v1",
            "status": "PASS_ADVISORY_ONLY",
            "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
            "review_packet_id": packet["review_packet_id"],
            "candidate_promotion_authority": False,
            "activation_applied_by_supreme_team": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return packet


def _make_ou_v6_ready(root: Path) -> dict[str, object]:
    active = root / "reports" / "active"
    supreme = root / "reports" / "supreme_team"
    config = root / "config" / "wizard_ou_comparator_v6_holdout.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}\n", encoding="utf-8")
    packet = _ready_packet(root, "ou_v6")
    _write_json(active / "wizard_ou_v6_review_packet.json", packet)
    _write_json(
        active / "wizard_ou_v6_supersession_gate.json",
        {
            "schema_version": "thewiz.wizard_ou_v6_supersession_gate.v1",
            "status": "READY_FOR_REVIEWED_SUPERSESSION",
            "blockers": [],
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        active / "wizard_ou_v6_activation_status.json",
        {
            "schema_version": "thewiz.wizard_ou_v6_activation.v1",
            "status": "READY_REQUIRES_EXPLICIT_APPLY",
            "apply_requested": False,
            "reviewer": "",
            "review_packet_id_submitted": "",
            "candidate_promotion_authority": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    _write_json(
        supreme / "wizard_ou_v6_review.json",
        {
            "schema_version": "thewiz.wizard_ou_v6_supreme_review.v1",
            "status": "PASS_ADVISORY_ONLY",
            "recommendation": "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION",
            "review_packet_id": packet["review_packet_id"],
            "candidate_promotion_authority": False,
            "activation_applied_by_supreme_team": False,
            "testnet_order_authority": False,
            "live_trading_authorized": False,
        },
    )
    return packet


def test_review_control_separates_review_ready_from_apply_window(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)

    result = _run(root, ["frozen_capture_manifest_unresolved:test"])

    assert result.summary["status"] == "WAITING_FOR_FROZEN_CAPTURE_RECONCILIATION"
    assert result.summary["review_ready"] == 1
    assert result.summary["apply_ready"] == 0
    assert result.summary["applied"] == 0
    assert result.summary["automatic_activation"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
    rows = pd.read_csv(result.paths["queue"]).fillna("")
    dynamic = rows.loc[rows["comparator"].eq("dynamic_v2")].iloc[0]
    ou = rows.loc[rows["comparator"].eq("ou_v3")].iloc[0]
    assert bool(dynamic["review_ready"]) is True
    assert bool(dynamic["apply_window_open"]) is False
    assert dynamic["operator_command_status"] == ("PREFLIGHT_ONLY_MUTATION_WINDOW_CLOSED")
    assert dynamic["review_packet_id"] in dynamic["preflight_command"]
    assert not dynamic["apply_command_template"]
    assert bool(ou["review_ready"]) is False


def test_review_control_accepts_explicit_blocked_holdout_evidence_state(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    packet_path = root / "reports" / "active" / "wizard_ou_v3_review_packet.json"
    packet = _read_json(packet_path)
    packet["status"] = "BLOCKED_HOLDOUT_EVIDENCE"
    _write_json(packet_path, packet)

    result = _run(root)

    assert result.summary["status"] == "READY_FOR_EXPLICIT_HUMAN_REVIEW"
    assert result.summary["review_ready"] == 1
    assert result.summary["apply_ready"] == 1
    assert "ou_v3_review_state_invalid" not in result.summary["control_blockers"]


def test_review_control_opens_only_after_both_evidence_and_mutation_window(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    _make_ou_ready(root)

    result = _run(root)

    assert result.summary["status"] == "READY_FOR_EXPLICIT_HUMAN_REVIEW"
    assert result.summary["review_ready"] == 2
    assert result.summary["apply_ready"] == 2
    assert result.summary["applied"] == 0
    assert result.summary["supreme_recommendation"] == (
        "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_REVIEW"
    )
    rows = pd.read_csv(result.paths["queue"]).fillna("")
    assert set(rows["operator_command_status"]) == {"READY_FOR_EXPLICIT_HUMAN_APPLY"}
    assert rows["apply_command_template"].str.contains("<reviewer>", regex=False).all()
    assert all(
        packet_id in command
        for packet_id, command in zip(
            rows["review_packet_id"],
            rows["apply_command_template"],
            strict=True,
        )
    )


def test_review_control_selects_active_ou_v4_and_builds_bound_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_root(tmp_path)
    ou_packet = _make_ou_v4_ready(root)
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_ou_v4_supreme_review."
        "load_validated_ou_v4_supreme_review",
        lambda **_: {"status": "PASS_ADVISORY_ONLY"},
    )

    result = _run(root)

    assert result.summary["status"] == "READY_FOR_EXPLICIT_HUMAN_REVIEW"
    assert result.summary["review_ready"] == 2
    assert result.summary["apply_ready"] == 2
    rows = pd.read_csv(result.paths["queue"]).fillna("")
    assert set(rows["comparator"]) == {"dynamic_v2", "ou_v4"}
    ou = rows.loc[rows["comparator"].eq("ou_v4")].iloc[0]
    assert ou["review_packet_id"] == ou_packet["review_packet_id"]
    assert "review-wizard-ou-v4" in ou["preflight_command"]
    assert "--apply-ou-v4" in ou["apply_command_template"]
    assert "--ou-v4-reviewer '<reviewer>'" in ou["apply_command_template"]
    markdown = result.paths["summary"].read_text(encoding="utf-8")
    assert "## Explicit Review Commands" in markdown
    assert ou_packet["review_packet_id"] in markdown


def test_review_control_selects_active_ou_v5_and_builds_bound_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_root(tmp_path)
    ou_packet = _make_ou_v5_ready(root)
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_ou_v5_supreme_review."
        "load_validated_ou_v5_supreme_review",
        lambda **_: {"status": "PASS_ADVISORY_ONLY"},
    )

    result = _run(root)

    assert result.summary["status"] == "READY_FOR_EXPLICIT_HUMAN_REVIEW"
    assert result.summary["review_ready"] == 2
    assert result.summary["apply_ready"] == 2
    rows = pd.read_csv(result.paths["queue"]).fillna("")
    assert set(rows["comparator"]) == {"dynamic_v2", "ou_v5"}
    ou = rows.loc[rows["comparator"].eq("ou_v5")].iloc[0]
    assert ou["review_packet_id"] == ou_packet["review_packet_id"]
    assert "review-wizard-ou-v5" in ou["preflight_command"]
    assert "--apply-ou-v5" in ou["apply_command_template"]
    assert "--ou-v5-reviewer '<reviewer>'" in ou["apply_command_template"]


def test_review_control_selects_terminal_ou_v6_and_builds_bound_commands(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _prepare_root(tmp_path)
    ou_packet = _make_ou_v6_ready(root)
    monkeypatch.setattr(
        "quant_platform.orchestration.corrective_wizard_ou_v6_supreme_review."
        "load_validated_ou_v6_supreme_review",
        lambda **_: {"status": "PASS_ADVISORY_ONLY"},
    )

    result = _run(root)

    assert result.summary["status"] == "READY_FOR_EXPLICIT_HUMAN_REVIEW"
    assert result.summary["review_ready"] == 2
    assert result.summary["apply_ready"] == 2
    rows = pd.read_csv(result.paths["queue"]).fillna("")
    assert set(rows["comparator"]) == {"dynamic_v2", "ou_v6"}
    ou = rows.loc[rows["comparator"].eq("ou_v6")].iloc[0]
    assert ou["review_packet_id"] == ou_packet["review_packet_id"]
    assert "review-wizard-ou-v6" in ou["preflight_command"]
    assert "--apply-ou-v6" in ou["apply_command_template"]
    assert "--ou-v6-reviewer '<reviewer>'" in ou["apply_command_template"]


def test_review_control_blocks_tampered_immutable_packet(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    packet = _read_json(root / "reports" / "active" / "wizard_dynamic_v2_review_packet.json")
    (root / str(packet["immutable_review_packet_path"])).write_text("tampered\n", encoding="utf-8")

    result = _run(root)

    assert result.summary["status"] == "BLOCKED_REVIEW_CONTROL"
    assert any(
        "immutable_review_packet_binding_invalid" in value
        for value in result.summary["control_blockers"]
    )


def test_review_control_blocks_automatic_activation_flag(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "wizard_dynamic_v2_review_packet.json"
    packet = _read_json(path)
    packet["automatic_activation"] = True
    _write_json(path, packet)

    result = _run(root)

    assert "dynamic_v2_automatic_activation_not_false" in result.summary["control_blockers"]


def test_review_control_blocks_mismatched_applied_packet(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    active = root / "reports" / "active"
    immutable_relative = "data/research/wizard_dynamic_comparator_activations/activation.json"
    _write_json(root / immutable_relative, {"activation": "frozen"})
    activation = _read_json(active / "wizard_dynamic_v2_activation_status.json")
    activation.update(
        {
            "status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
            "apply_requested": True,
            "reviewer": "Human Reviewer",
            "review_note": "I reviewed the exact immutable comparator evidence.",
            "review_packet_id_submitted": "dynamicv2review_wrong",
            "immutable_activation_path": immutable_relative,
            "immutable_activation_sha256": sha256(
                (root / immutable_relative).read_bytes()
            ).hexdigest(),
        }
    )
    _write_json(active / "wizard_dynamic_v2_activation_status.json", activation)

    result = _run(root)

    assert "dynamic_v2_applied_activation_binding_invalid" in result.summary["control_blockers"]


def test_review_control_blocks_authority_leak(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "wizard_ou_v3_activation_status.json"
    activation = _read_json(path)
    activation["testnet_order_authority"] = True
    _write_json(path, activation)

    result = _run(root)

    assert "ou_v3_authority_boundary_invalid" in result.summary["control_blockers"]


def test_review_control_blocks_waiting_packet_with_forged_identity(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "active" / "wizard_ou_v3_review_packet.json"
    packet = _read_json(path)
    packet["review_packet_id"] = "ouv3review_forged"
    _write_json(path, packet)

    result = _run(root)

    assert "ou_v3_review_state_invalid" in result.summary["control_blockers"]


def test_review_control_blocks_supreme_packet_mismatch(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)
    path = root / "reports" / "supreme_team" / "wizard_dynamic_v2_review.json"
    supreme = _read_json(path)
    supreme["review_packet_id"] = "dynamicv2review_stale"
    _write_json(path, supreme)

    result = _run(root)

    assert "dynamic_v2_supreme_packet_binding_invalid" in result.summary["control_blockers"]


def test_review_control_blocks_tampered_immutable_supreme_review(
    tmp_path: Path,
) -> None:
    root = _prepare_root(tmp_path)
    status = _read_json(root / "reports" / "supreme_team" / "wizard_dynamic_v2_review.json")
    immutable = root / str(status["immutable_review_path"])
    immutable.write_text("tampered\n", encoding="utf-8")

    result = _run(root)

    assert "dynamic_v2_supreme_immutable_binding_invalid" in result.summary["control_blockers"]


def test_review_control_is_idempotent_and_rejects_receipt_collision(tmp_path: Path) -> None:
    root = _prepare_root(tmp_path)

    first = _run(root, ["frozen_capture_manifest_unresolved:test"])
    second = _run(root, ["frozen_capture_manifest_unresolved:test"])

    assert first.summary["receipt_id"] == second.summary["receipt_id"]
    first.paths["immutable_receipt"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable comparator review receipt collision"):
        _run(root, ["frozen_capture_manifest_unresolved:test"])
