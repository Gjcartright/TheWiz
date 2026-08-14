from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform import wizard_dynamic_comparator_activation
from quant_platform.orchestration.corrective_wizard_dynamic_holdout import (
    build_dynamic_v2_review_handoff,
    build_dynamic_v2_reviewed_activation,
    build_dynamic_v2_supersession_gate,
    evaluate_dynamic_v2_holdout,
)
from quant_platform.orchestration.corrective_wizard_dynamic_supreme_review import (
    build_dynamic_v2_supreme_review,
)
from quant_platform.wizard_dynamic_comparator_activation import (
    load_validated_dynamic_v2_activation,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    _dynamic_v2_source_hash,
    _formula_parity_metrics_for_generation,
    _kalman_dynamic_spread_candidate,
    _kalman_dynamic_spread_candidate_v2_holdout,
    refresh_activated_dynamic_v2_proofs,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dynamic_v2_comparator_source_remains_frozen() -> None:
    assert _dynamic_v2_source_hash() == (
        "5ee018f4c0c32033907aa2dcc65b2707dd9a3a5589f8eae2628de800d5f5af51"
    )


def _write_registration(root: Path, *, same_cohort: bool = False) -> None:
    config = root / "config"
    active = root / "reports" / "active"
    derivation_path = root / "data" / "research" / "derivation.json"
    config.mkdir(parents=True)
    active.mkdir(parents=True)
    derivation_path.parent.mkdir(parents=True)
    derivation_path.write_text('{"derivation": true}\n', encoding="utf-8")
    source_hash = hashlib.sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode()
    ).hexdigest()
    contract = {
        "schema_version": "thewiz.wizard_dynamic_comparator_holdout.v2",
        "derivation_cohort": {"pair_group_id": "holdout" if same_cohort else "derivation"},
        "holdout_cohort": {
            "pair_group_id": "holdout",
            "required_exact_modes": ["Dyn (Spread)", "Dyn (ZScoreR)"],
            "required_orientations": ["original", "reverse"],
            "required_mode_orientation_cells": 4,
        },
        "implementation": {"source_sha256": source_hash},
        "derivation_evidence": [
            {
                "path": "data/research/derivation.json",
                "sha256": _sha(derivation_path),
            }
        ],
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    contract_path = config / "wizard_dynamic_comparator_v2_holdout.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    receipt = {
        "contract_path": "config/wizard_dynamic_comparator_v2_holdout.json",
        "contract_sha256": _sha(contract_path),
        "registered_before_holdout_vendor_responses": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    (active / "wizard_dynamic_comparator_v2_holdout_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )


def _vendor_response(spread: np.ndarray, *, window: int = 20) -> dict[str, object]:
    series = pd.Series(spread)
    zscore = (spread - spread.mean()) / spread.std(ddof=1)
    rolling = (
        series.sub(series.rolling(window, min_periods=window).mean())
        .div(series.rolling(window, min_periods=window).std(ddof=1))
        .fillna(0.0)
    )
    return {
        "history": {
            "spread_stats": {
                "log_used": False,
                "spread": spread.tolist(),
                "zscore": zscore.tolist(),
                "zscore_roll": rolling.tolist(),
            }
        }
    }


def _write_proofs(root: Path, *, cells: int = 4, mutate_last: bool = False) -> None:
    raw = root / "data" / "raw"
    active = root / "reports" / "active"
    raw.mkdir(parents=True, exist_ok=True)
    rows = []
    combinations = [
        (mode, orientation)
        for mode in ("Dyn (Spread)", "Dyn (ZScoreR)")
        for orientation in ("original", "reverse")
    ][:cells]
    for index, (mode, orientation) in enumerate(combinations):
        base_x = np.linspace(0.8, 1.4, 80)
        base_y = np.linspace(0.2, 0.5, 80) + 0.01 * np.sin(np.arange(80))
        x, y = (base_x, base_y) if orientation == "original" else (base_y, base_x)
        spread = _kalman_dynamic_spread_candidate_v2_holdout(x, y)
        if mutate_last and index == len(combinations) - 1:
            spread = spread.copy()
            spread[-1] += 0.01
        request_path = raw / f"request_{index}.json"
        response_path = raw / f"response_{index}.json"
        request_path.write_text(
            json.dumps(
                {
                    "params": {
                        "series_1_closes": x.tolist(),
                        "series_2_closes": y.tolist(),
                        "roll_w": 20,
                        "strategy": ("ZScoreRoll" if mode == "Dyn (ZScoreR)" else "Spread"),
                        "spread_type": "Dynamic",
                    }
                }
            ),
            encoding="utf-8",
        )
        response_path.write_text(json.dumps(_vendor_response(spread)), encoding="utf-8")
        rows.append(
            {
                "pair_group_id": "holdout",
                "exact_mode": mode,
                "orientation": orientation,
                "vendor_response_captured": True,
                "mode_proof_status": "completed",
                "vendor_history_available": True,
                "proof_window_kind": "scanner_horizon_parity",
                "formula_comparator_generation": 1,
                "request_path": str(request_path.relative_to(root)),
                "response_path": str(response_path.relative_to(root)),
            }
        )
    pd.DataFrame(rows).to_csv(active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False)


def _write_activation_dependencies(root: Path) -> tuple[Path, Path]:
    active = root / "reports" / "active"
    comparator = active / "wizard_mode_comparator_contract.csv"
    comparator.write_text("exact_mode,version\nDyn (Spread),v1\n", encoding="utf-8")
    receipt = active / "wizard_mode_comparator_contract_receipt.json"
    receipt.write_text(json.dumps({"contract_id": "wizard-comparator-v1"}), encoding="utf-8")
    return comparator, receipt


def test_waits_for_holdout_without_granting_authority(tmp_path):
    _write_registration(tmp_path)

    result = evaluate_dynamic_v2_holdout(root=tmp_path)
    _write_activation_dependencies(tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)

    assert result.summary["status"] == "WAITING_FOR_HOLDOUT"
    assert result.summary["captured_cells"] == 0
    assert not result.summary["comparator_supersession_eligible"]
    assert not result.summary["testnet_order_authority"]
    assert packet.summary["status"] == "WAITING_FOR_HOLDOUT"
    assert len(packet.summary["cells"]) == 4
    assert not packet.summary["all_cells_captured"]
    assert not packet.summary["raw_bindings_valid"]
    assert "immutable_review_packet" not in packet.paths
    assert packet.summary["explicit_review_required"]
    assert not packet.summary["automatic_activation"]
    assert not packet.summary["testnet_order_authority"]


def test_passes_only_complete_disjoint_four_cell_holdout(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)

    result = evaluate_dynamic_v2_holdout(root=tmp_path, now=datetime(2026, 8, 11, tzinfo=UTC))

    assert result.summary["status"] == "PASS"
    assert result.summary["captured_cells"] == 4
    assert result.summary["passed_cells"] == 4
    assert result.summary["comparator_supersession_eligible"]
    assert not result.summary["comparator_supersession_automatic"]
    assert result.paths["immutable_result"].is_file()
    assert not result.summary["candidate_promotion_authority"]
    gate = build_dynamic_v2_supersession_gate(root=tmp_path)
    assert gate.summary["status"] == "READY_FOR_REVIEWED_SUPERSESSION"
    assert gate.summary["supersession_applied"] is False
    assert not gate.summary["testnet_order_authority"]
    _write_activation_dependencies(tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)
    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert packet.summary["all_cells_passed"]
    assert packet.summary["raw_bindings_valid"]
    assert packet.paths["immutable_review_packet"].is_file()
    assert str(packet.summary["review_packet_id"]).startswith("dynamicv2review_")
    assert not packet.summary["candidate_promotion_authority"]


def test_partial_holdout_remains_incomplete(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path, cells=3)

    result = evaluate_dynamic_v2_holdout(root=tmp_path)

    assert result.summary["status"] == "INCOMPLETE"
    assert result.summary["captured_cells"] == 3
    assert "immutable_result" not in result.paths
    gate = build_dynamic_v2_supersession_gate(root=tmp_path)
    assert gate.summary["status"] == "BLOCKED"
    assert "dynamic_v2_disjoint_holdout_not_passed" in gate.summary["blockers"]


def test_mutated_holdout_fails_closed(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path, mutate_last=True)

    result = evaluate_dynamic_v2_holdout(root=tmp_path)

    assert result.summary["status"] == "FAIL"
    assert result.summary["failed_cells"] == 1
    assert not result.summary["comparator_supersession_eligible"]
    assert result.paths["immutable_result"].is_file()


def test_same_cohort_cannot_validate_dynamic_v2(tmp_path):
    _write_registration(tmp_path, same_cohort=True)

    with pytest.raises(ValueError, match="disjoint"):
        evaluate_dynamic_v2_holdout(root=tmp_path)


def test_tampered_contract_is_rejected(tmp_path):
    _write_registration(tmp_path)
    contract = tmp_path / "config" / "wizard_dynamic_comparator_v2_holdout.json"
    contract.write_text(contract.read_text() + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        evaluate_dynamic_v2_holdout(root=tmp_path)


def test_reviewed_activation_is_versioned_immutable_and_research_only(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    comparator_path, _ = _write_activation_dependencies(tmp_path)
    comparator_hash_before = _sha(comparator_path)
    proof_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    proof_hash_before = _sha(proof_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)

    planned = build_dynamic_v2_reviewed_activation(root=tmp_path)
    assert planned.summary["status"] == "READY_REQUIRES_EXPLICIT_APPLY"
    assert not planned.summary["apply_requested"]
    assert str(planned.summary["review_packet_id"]).startswith("dynamicv2review_")
    supreme = build_dynamic_v2_supreme_review(root=tmp_path)

    missing_review = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
    )
    assert missing_review.summary["status"] == "BLOCKED"
    assert "reviewer_required_for_dynamic_v2_apply" in missing_review.summary["blockers"]

    applied = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note=("Reviewed all four disjoint Dynamic v2 cells and immutable lineage."),
        review_packet_id=planned.summary["review_packet_id"],
        now=datetime(2026, 8, 11, 1, 0, tzinfo=UTC),
    )
    assert applied.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert applied.paths["immutable_activation"].is_file()
    assert applied.paths["proof_snapshot"].is_file()
    assert _sha(comparator_path) == comparator_hash_before
    assert _sha(proof_path) == proof_hash_before
    assert not applied.summary["original_comparator_mutated"]
    assert not applied.summary["raw_vendor_evidence_mutated"]
    assert not applied.summary["candidate_promotion_authority"]
    assert not applied.summary["testnet_order_authority"]
    assert applied.summary["schema_version"] == "thewiz.wizard_dynamic_v2_activation.v2"
    assert applied.summary["supreme_review_id"] == supreme.summary["review_id"]
    assert applied.summary["supreme_review_path"] == supreme.summary["immutable_review_path"]
    assert applied.summary["supreme_review_sha256"] == supreme.summary["immutable_review_sha256"]

    source_hash = hashlib.sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode()
    ).hexdigest()
    validated = load_validated_dynamic_v2_activation(
        root=tmp_path,
        implementation_source_sha256=source_hash,
    )
    assert validated is not None
    assert validated["activation_id"] == applied.summary["activation_id"]

    applied.paths["proof_snapshot"].write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="activation binding mismatch"):
        load_validated_dynamic_v2_activation(
            root=tmp_path,
            implementation_source_sha256=source_hash,
        )


def test_dynamic_activation_waits_for_frozen_capture_reconciliation(tmp_path, monkeypatch):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    planned = build_dynamic_v2_reviewed_activation(root=tmp_path)
    build_dynamic_v2_supreme_review(root=tmp_path)
    blocker = "frozen_capture_manifest_unresolved:wizardcapture_test"
    monkeypatch.setattr(
        wizard_dynamic_comparator_activation,
        "frozen_capture_manifest_mutation_blockers",
        lambda *, root: [blocker],
    )

    result = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed all disjoint holdout evidence and immutable lineage.",
        review_packet_id=planned.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert blocker in result.summary["blockers"]
    assert "immutable_activation" not in result.paths


def test_reviewed_activation_cannot_apply_before_holdout_pass(tmp_path):
    _write_registration(tmp_path)
    active = tmp_path / "reports" / "active"
    (active / "hyperliquid_wizard_vendor_mode_proofs.csv").write_text(
        "pair_group_id\n", encoding="utf-8"
    )
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)

    result = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Attempted only to verify premature activation stays blocked.",
    )

    assert result.summary["status"] == "BLOCKED"
    assert "dynamic_v2_supersession_gate_not_ready" in result.summary["blockers"]
    assert "immutable_activation" not in result.paths
    assert not result.summary["testnet_order_authority"]


def test_activation_requires_exact_current_review_packet_id(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)
    build_dynamic_v2_supreme_review(root=tmp_path)

    missing = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed the immutable packet but omitted its exact identifier.",
    )
    stale = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed the immutable packet but supplied a stale identifier.",
        review_packet_id="dynamicv2review_stale",
    )

    assert missing.summary["status"] == "BLOCKED"
    assert "review_packet_id_required_for_dynamic_v2_apply" in missing.summary["blockers"]
    assert stale.summary["status"] == "BLOCKED"
    assert "review_packet_id_does_not_match_current_evidence" in stale.summary["blockers"]
    assert packet.paths["immutable_review_packet"].is_file()
    assert not stale.summary["testnet_order_authority"]


def test_review_packet_rejects_tampered_raw_holdout_evidence(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)
    response = min((tmp_path / "data" / "raw").glob("response_*.json"))
    response.write_text(response.read_text(encoding="utf-8") + " ", encoding="utf-8")

    blocked = build_dynamic_v2_review_handoff(root=tmp_path)

    assert packet.summary["status"] == "READY_FOR_EXPLICIT_REVIEW"
    assert blocked.summary["status"] == "BLOCKED_EVIDENCE_BINDING"
    assert not blocked.summary["raw_bindings_valid"]
    assert "dynamic_v2_review_packet_raw_binding_mismatch" in blocked.summary["blockers"]
    assert not blocked.summary["live_trading_authorized"]


def test_review_packet_id_survives_repeated_holdout_evaluation(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path, now=datetime(2026, 8, 11, 1, 0, tzinfo=UTC))
    build_dynamic_v2_supersession_gate(root=tmp_path)
    first = build_dynamic_v2_review_handoff(
        root=tmp_path, now=datetime(2026, 8, 11, 1, 1, tzinfo=UTC)
    )
    first_bytes = first.paths["immutable_review_packet"].read_bytes()

    evaluate_dynamic_v2_holdout(root=tmp_path, now=datetime(2026, 8, 11, 1, 10, tzinfo=UTC))
    build_dynamic_v2_supersession_gate(root=tmp_path)
    second = build_dynamic_v2_review_handoff(
        root=tmp_path, now=datetime(2026, 8, 11, 1, 11, tzinfo=UTC)
    )

    assert first.summary["review_packet_id"] == second.summary["review_packet_id"]
    assert first.paths["immutable_review_packet"] == second.paths["immutable_review_packet"]
    assert second.paths["immutable_review_packet"].read_bytes() == first_bytes


def test_dynamic_v2_formula_generation_reconstructs_holdout_exactly():
    x = np.linspace(0.8, 1.4, 80)
    y = np.linspace(0.2, 0.5, 80) + 0.01 * np.sin(np.arange(80))
    request = {
        "params": {
            "series_1_closes": x.tolist(),
            "series_2_closes": y.tolist(),
            "roll_w": 20,
            "strategy": "Spread",
            "spread_type": "Dynamic",
        }
    }
    response = _vendor_response(_kalman_dynamic_spread_candidate_v2_holdout(x, y))

    generation_one = _formula_parity_metrics_for_generation(request, response, dynamic_generation=1)
    generation_two = _formula_parity_metrics_for_generation(request, response, dynamic_generation=2)

    assert generation_one["vendor_formula_parity_status"] == "mismatch"
    assert generation_two["vendor_formula_parity_status"] == "exact_reconstruction"
    assert str(generation_two["vendor_spread_formula"]).startswith("candidate_v2:")
    with pytest.raises(ValueError, match="unsupported dynamic comparator generation"):
        _formula_parity_metrics_for_generation(request, response, dynamic_generation=3)


def test_reviewed_activation_refreshes_proofs_with_immutable_lineage(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    proof_path = tmp_path / "reports" / "active" / "hyperliquid_wizard_vendor_mode_proofs.csv"
    raw_paths = sorted((tmp_path / "data" / "raw").glob("*.json"))
    raw_hashes_before = {path: _sha(path) for path in raw_paths}
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    build_dynamic_v2_supreme_review(root=tmp_path)
    applied = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed disjoint holdout lineage before local proof refresh.",
        review_packet_id=build_dynamic_v2_review_handoff(root=tmp_path).summary["review_packet_id"],
    )

    refreshed = refresh_activated_dynamic_v2_proofs(root=tmp_path)
    proofs = pd.read_csv(proof_path)

    assert refreshed.summary["status"] == "PASS"
    assert refreshed.summary["captured_dynamic_rows"] == 4
    assert refreshed.summary["exact_dynamic_rows"] == 4
    assert proofs["formula_comparator_generation"].eq(2).all()
    assert proofs["formula_comparator_activation_id"].eq(applied.summary["activation_id"]).all()
    assert proofs["vendor_formula_parity_status"].eq("exact_reconstruction").all()
    assert proofs["formula_proof_complete"].map(bool).all()
    assert {path: _sha(path) for path in raw_paths} == raw_hashes_before
    assert not refreshed.summary["raw_vendor_evidence_mutated"]
    assert not refreshed.summary["candidate_promotion_authority"]
    assert not refreshed.summary["testnet_order_authority"]

    repeated_plan = build_dynamic_v2_reviewed_activation(root=tmp_path)
    repeated_refresh = refresh_activated_dynamic_v2_proofs(root=tmp_path)
    assert repeated_plan.summary["status"] == "APPLIED_RESEARCH_COMPARATOR_ONLY"
    assert repeated_plan.summary["activation_id"] == applied.summary["activation_id"]
    assert repeated_refresh.summary["status"] == "PASS"


def test_dynamic_v2_refresh_rejects_tampered_active_activation(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    build_dynamic_v2_supreme_review(root=tmp_path)
    build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed disjoint holdout lineage before tamper validation.",
        review_packet_id=build_dynamic_v2_review_handoff(root=tmp_path).summary["review_packet_id"],
    )
    status_path = tmp_path / "reports" / "active" / "wizard_dynamic_v2_activation_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["immutable_activation_sha256"] = "0" * 64
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="immutable activation hash mismatch"):
        refresh_activated_dynamic_v2_proofs(root=tmp_path)


def test_dynamic_v2_direct_apply_requires_current_immutable_supreme_review(
    tmp_path,
):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)

    result = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable Dynamic-v2 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert "dynamic_v2_supreme_review_required_for_apply" in result.summary["blockers"]
    assert "activation_id" not in result.summary
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_dynamic_v2_direct_apply_rejects_rehashed_tampered_supreme_review(
    tmp_path,
):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)
    supreme = build_dynamic_v2_supreme_review(root=tmp_path)
    immutable_path = supreme.paths["immutable_review"]
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    immutable["recommendation"] = "DO_NOT_ACTIVATE"
    immutable_path.write_text(
        json.dumps(immutable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = supreme.paths["status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["immutable_review_sha256"] = _sha(immutable_path)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable Dynamic-v2 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )

    assert result.summary["status"] == "BLOCKED"
    assert "dynamic_v2_supreme_review_invalid_for_apply" in result.summary["blockers"]
    assert "activation_id" not in result.summary
    assert result.summary["testnet_order_authority"] is False
    assert result.summary["live_trading_authorized"] is False


def test_dynamic_v2_loader_rejects_rehashed_activation_content(tmp_path):
    _write_registration(tmp_path)
    _write_proofs(tmp_path)
    _write_activation_dependencies(tmp_path)
    evaluate_dynamic_v2_holdout(root=tmp_path)
    build_dynamic_v2_supersession_gate(root=tmp_path)
    packet = build_dynamic_v2_review_handoff(root=tmp_path)
    build_dynamic_v2_supreme_review(root=tmp_path)
    applied = build_dynamic_v2_reviewed_activation(
        root=tmp_path,
        apply=True,
        reviewer="research-reviewer",
        review_note="Reviewed every immutable Dynamic-v2 parity cell independently.",
        review_packet_id=packet.summary["review_packet_id"],
    )
    immutable_path = applied.paths["immutable_activation"]
    immutable = json.loads(immutable_path.read_text(encoding="utf-8"))
    immutable["review_note"] = "A different but still substantive review note was inserted."
    immutable_path.write_text(
        json.dumps(immutable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    status_path = applied.paths["activation_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["review_note"] = immutable["review_note"]
    status["immutable_activation_sha256"] = _sha(immutable_path)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_hash = hashlib.sha256(
        inspect.getsource(_kalman_dynamic_spread_candidate_v2_holdout).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="immutable activation identity mismatch"):
        load_validated_dynamic_v2_activation(
            root=tmp_path,
            implementation_source_sha256=source_hash,
        )


def test_dynamic_v1_and_v2_candidates_are_not_equivalent():
    x = np.linspace(0.8, 1.4, 80)
    y = np.linspace(0.2, 0.5, 80) + 0.01 * np.sin(np.arange(80))

    assert not np.allclose(
        _kalman_dynamic_spread_candidate(x, y),
        _kalman_dynamic_spread_candidate_v2_holdout(x, y),
    )
