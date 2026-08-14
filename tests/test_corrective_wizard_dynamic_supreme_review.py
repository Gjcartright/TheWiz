from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

from quant_platform.active_pipeline import CommandResult
from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
    build_dynamic_v2_supreme_review,
)


def _fixture(tmp_path, *, tamper: bool = False):
    active = tmp_path / "reports" / "active"
    immutable = tmp_path / "data" / "research" / "packet.json"
    active.mkdir(parents=True)
    immutable.parent.mkdir(parents=True)
    cells = [
        {
            "exact_mode": mode,
            "orientation": orientation,
            "cell_status": "PASS",
            "raw_binding_valid": True,
            "spread_max_abs_error": "1e-14",
            "zscore_max_abs_error": "2e-14",
            "zscore_roll_max_abs_error": "3e-14",
        }
        for mode in ("Dyn (Spread)", "Dyn (ZScoreR)")
        for orientation in ("original", "reverse")
    ]
    packet_id = "dynamicv2review_test"
    immutable.write_text(
        json.dumps({"review_packet_id": packet_id}), encoding="utf-8"
    )
    packet_path = active / "wizard_dynamic_v2_review_packet.json"
    packet = {
        "status": "READY_FOR_EXPLICIT_REVIEW",
        "review_packet_id": packet_id,
        "cohorts_disjoint": True,
        "all_cells_captured": True,
        "all_cells_passed": True,
        "raw_bindings_valid": True,
        "cells": cells,
        "immutable_review_packet_path": str(immutable.relative_to(tmp_path)),
        "immutable_review_packet_sha256": sha256(immutable.read_bytes()).hexdigest(),
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    if tamper:
        immutable.write_text("tampered", encoding="utf-8")

    def packet_builder(**kwargs):
        return CommandResult(paths={"review_packet": packet_path}, summary=packet)

    def activation_planner(**kwargs):
        assert kwargs["apply"] is False
        return CommandResult(
            paths={"activation_status": active / "activation.json"},
            summary={
                "status": "READY_REQUIRES_EXPLICIT_APPLY",
                "apply_requested": False,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            },
        )

    return packet_builder, activation_planner


def test_dynamic_supreme_review_recommends_only_research_activation(tmp_path):
    packet_builder, activation_planner = _fixture(tmp_path)

    result = build_dynamic_v2_supreme_review(
        root=tmp_path,
        now=datetime(2026, 8, 11, tzinfo=UTC),
        packet_builder=packet_builder,
        activation_planner=activation_planner,
    )

    assert result.summary["status"] == "PASS_ADVISORY_ONLY"
    assert result.summary["recommendation"] == (
        "RECOMMEND_EXPLICIT_RESEARCH_COMPARATOR_ACTIVATION"
    )
    assert result.summary["passed_cells"] == 4
    assert result.summary["max_abs_reconstruction_error"] == 3e-14
    assert {item["lens"] for item in result.summary["findings"]} == {
        "gap_analysis",
        "pre_mortem",
        "post_mortem",
        "red_team",
    }
    assert result.summary["human_approval_recorded"] is False
    assert result.summary["activation_applied_by_supreme_team"] is False
    assert result.summary["candidate_promotion_authority"] is False
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_dynamic_supreme_review_blocks_tampered_packet(tmp_path):
    packet_builder, activation_planner = _fixture(tmp_path, tamper=True)

    result = build_dynamic_v2_supreme_review(
        root=tmp_path,
        packet_builder=packet_builder,
        activation_planner=activation_planner,
    )

    assert result.summary["status"] == "BLOCKED"
    assert result.summary["recommendation"] == "DO_NOT_ACTIVATE"
    assert "dynamic_v2_immutable_review_packet_hash_mismatch" in result.summary[
        "blockers"
    ]
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False
