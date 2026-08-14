from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_platform.orchestration.corrective_wizard_parity import (
    EXACT_MODES,
    ORIENTATIONS,
    WIZARD_PRESCANNED_CONTRACT_URL,
    _active_ou_evidence_gate,
    _ou_v3_evidence_gate,
    build_ou_optimal_ordered_orientation_audit,
    build_ou_optimal_overlay_provenance,
    build_wizard_mode_authority,
    build_wizard_mode_comparator_contract,
    build_wizard_mode_evidence_completion,
    compare_series,
    run_wizard_mode_mutation_tests,
)
from quant_platform.wizard_hyperliquid_mode_proof import (
    build_comparator_formula_registration,
)


def test_source_fingerprint_modules_are_excluded_from_ruff_formatter() -> None:
    root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    exclusions = set(configuration["tool"]["ruff"]["format"]["exclude"])

    assert {
        "src/quant_platform/wizard_hyperliquid_mode_proof.py",
        "src/quant_platform/orchestration/corrective_wizard_ou_holdout.py",
        "src/quant_platform/orchestration/corrective_wizard_ou_v4_holdout.py",
        "src/quant_platform/orchestration/corrective_wizard_ou_v5_holdout.py",
    } <= exclusions


def test_parity_comparison_accepts_match_and_rejects_mutation():
    baseline = np.arange(20, dtype=float)
    assert compare_series(baseline, baseline.copy())["status"] == "PASS"
    assert compare_series(baseline, baseline * -1)["status"] == "FAIL"
    assert compare_series(baseline, baseline[:-1])["status"] == "BLOCKED"


def test_all_seven_pair_page_modes_and_both_orientations_are_accounted():
    assert len(EXACT_MODES) == 7
    assert len(ORIENTATIONS) == 2
    assert len({(mode, orientation) for mode in EXACT_MODES for orientation in ORIENTATIONS}) == 14
    assert "OU (Optimal)" not in EXACT_MODES


def test_ou_v3_gate_requires_prospective_selector_and_four_exact_cells(tmp_path):
    missing = _ou_v3_evidence_gate(root=tmp_path, expected_cells=8)
    assert missing["status"] == "BLOCKED"
    assert "ou_v3_selector_receipt_missing_or_invalid" in missing["blocker"]

    active = tmp_path / "reports" / "active"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    config.mkdir(parents=True)
    selector_contract_path = config / "wizard_ou_trend_selector_v1_holdout.json"
    selector_contract_path.write_text(
        json.dumps(
            {
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "wizard_ou_trend_selector_v1_receipt.json").write_text(
        json.dumps(
            {
                "status": "REGISTERED_WAITING_VENDOR_RESPONSES",
                "prospectively_registered": True,
                "vendor_responses_at_registration": 0,
                "derivation_matched_cells": 8,
                "holdout_prediction_cells": 4,
                "contract_sha256": hashlib.sha256(
                    selector_contract_path.read_bytes()
                ).hexdigest(),
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "wizard_ou_v3_holdout_status.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "formula_parity_passed_cells": 4,
                "trend_selector_parity_passed_cells": 4,
                "local_point_in_time_trend_selector_proven": True,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "formula_parity_passed": True,
                "trend_selector_parity_passed": True,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
            for _ in range(4)
        ]
    ).to_csv(active / "wizard_ou_v3_holdout_evaluation.csv", index=False)
    (active / "wizard_ou_v3_activation_status.json").write_text(
        json.dumps(
            {
                "status": "APPLIED_RESEARCH_COMPARATOR_ONLY",
                "activation_id": "ouv3activation_test",
                "comparator_generation": 3,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    (active / "wizard_ou_v3_proof_refresh_status.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "activation_id": "ouv3activation_test",
                "comparator_generation": 3,
                "captured_ou_rows": 8,
                "exact_ou_rows": 8,
                "refreshed_ou_rows": 8,
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        ),
        encoding="utf-8",
    )

    passed = _ou_v3_evidence_gate(root=tmp_path, expected_cells=8)
    assert passed["status"] == "PASS"
    assert passed["formula_cells_passed"] == 4
    assert passed["selector_cells_passed"] == 4
    assert passed["local_selector_proven"] is True

    status_path = active / "wizard_ou_v3_holdout_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["local_point_in_time_trend_selector_proven"] = False
    status_path.write_text(json.dumps(status), encoding="utf-8")
    blocked = _ou_v3_evidence_gate(root=tmp_path, expected_cells=8)
    assert blocked["status"] == "BLOCKED"
    assert "ou_v3_selector_cells_incomplete" in blocked["blocker"]


def test_active_ou_gate_prefers_registered_v6_and_fails_closed_while_waiting(tmp_path):
    config = tmp_path / "config" / "wizard_ou_comparator_v6_holdout.json"
    config.parent.mkdir(parents=True)
    config.write_text("{}\n", encoding="utf-8")

    result = _active_ou_evidence_gate(root=tmp_path, expected_cells=8)

    assert result["status"] == "BLOCKED"
    assert result["evidence_generation"] == 6
    assert result["successor_after_failure_allowed"] is False
    assert "ou_v6_scheduler_stage3_evidence_incomplete" in result["blocker"]


def test_formula_mutations_cannot_receive_parity(tmp_path):
    frame = run_wizard_mode_mutation_tests(root=tmp_path)
    assert frame["status"].eq("PASS").all()
    assert not frame["mislabeled_mode_can_receive_parity"].any()


def test_comparator_contract_freezes_ready_and_backlog_cells(tmp_path):
    result = build_wizard_mode_comparator_contract(root=tmp_path)
    frame = result["frame"]
    receipt = json.loads(result["receipt_path"].read_text(encoding="utf-8"))

    assert len(frame) == 14
    assert result["ready_cells"] == 4
    assert result["candidate_validation_cells"] == 8
    assert result["backlog_cells"] == 2
    assert frame.loc[
        frame["exact_mode"].str.startswith("Static"), "implementation_status"
    ].eq("READY").all()
    assert not frame["vendor_response_capture_sufficient_for_parity"].any()
    assert not frame["threshold_changes_after_vendor_response_allowed"].any()
    assert not frame["promotion_authority"].any()
    assert result["immutable_path"].is_file()
    assert result["implementation_path"].is_file()
    assert frame["comparator_specification_sha256"].str.len().eq(64).all()
    assert frame["comparator_source_sha256"].str.len().eq(64).all()
    assert frame["comparator_implementation_sha256"].str.len().eq(64).all()
    assert receipt["current_queue_vendor_responses_at_registration"] == 0
    assert receipt["registered_before_current_queue_vendor_responses"] is True
    assert receipt["threshold_changes_after_vendor_response_allowed"] is False
    assert receipt["promotion_authority"] is False
    assert receipt["testnet_order_authority"] is False
    assert receipt["live_trading_authorized"] is False
    assert receipt["implementation_bundle_sha256"] == result[
        "implementation_sha256"
    ]


def test_dynamic_v1_comparator_source_remains_frozen_under_v2_overlay():
    registration = build_comparator_formula_registration("Dyn (Spread)")

    assert registration["source_sha256"] == (
        "0d7c3a5b71126a1afee0c96dc6bafd3e627dd5f2b76415e51be08b0ebbbe53b8"
    )


def test_comparator_contract_tampering_fails_before_active_mirror_changes(tmp_path):
    result = build_wizard_mode_comparator_contract(root=tmp_path)
    active_before = result["path"].read_bytes()
    result["immutable_path"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="immutable Wizard comparator contract hash mismatch"
    ):
        build_wizard_mode_comparator_contract(root=tmp_path)

    assert result["path"].read_bytes() == active_before


def test_comparator_implementation_bundle_tampering_fails_closed(tmp_path):
    result = build_wizard_mode_comparator_contract(root=tmp_path)
    result["implementation_path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValueError, match="immutable Wizard comparator implementation hash mismatch"
    ):
        build_wizard_mode_comparator_contract(root=tmp_path)


def test_comparator_change_after_current_queue_response_is_rejected(
    tmp_path, monkeypatch
):
    from quant_platform import wizard_hyperliquid_mode_proof as proof_module

    build_wizard_mode_comparator_contract(root=tmp_path)
    active = tmp_path / "reports" / "active"
    pd.DataFrame(
        [
            {
                "pair_group_id": "current-pair-group",
                "vendor_response_captured": True,
            }
        ]
    ).to_csv(active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False)
    original = proof_module.build_comparator_formula_registration

    def changed_registration(exact_mode):
        registration = original(exact_mode)
        registration["implementation_sha256"] = "f" * 64
        return registration

    monkeypatch.setattr(
        proof_module,
        "build_comparator_formula_registration",
        changed_registration,
    )

    with pytest.raises(
        ValueError,
        match="cannot change after vendor responses",
    ):
        build_wizard_mode_comparator_contract(root=tmp_path)


def test_ou_optimal_is_accounted_as_two_orientation_scanner_overlay(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_orientation": orientation,
                "ou_optimal": flag,
                "ou_optimal_semantics": "scanner_boolean_annotation",
                "independent_pair_page_mode": False,
                "source_exact_mode": "OU (Spread)",
                "evidence_path": "raw.json",
            }
            for orientation in ORIENTATIONS
            for flag in (True, False)
        ]
    ).to_csv(active / "current_wizard_ou_optimal_overlay_ledger.csv", index=False)

    result = build_ou_optimal_overlay_provenance(root=tmp_path)

    assert result["status"] == "PASS"
    assert result["orientations_accounted"] == 2
    assert result["vendor_formula_parity_proven"] is False
    assert not result["frame"]["independent_pair_page_mode"].any()
    assert result["frame"]["vendor_contract_url"].eq(
        WIZARD_PRESCANNED_CONTRACT_URL
    ).all()
    assert result["frame"]["vendor_orientation_value_observed"].all()


def test_ou_optimal_contract_does_not_invent_missing_reverse_observation(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_orientation": "original",
                "ou_optimal": True,
                "ou_optimal_semantics": "scanner_boolean_annotation",
                "independent_pair_page_mode": False,
                "source_exact_mode": "OU (Spread)",
                "evidence_path": "raw.json",
            }
        ]
    ).to_csv(active / "current_wizard_ou_optimal_overlay_ledger.csv", index=False)

    result = build_ou_optimal_overlay_provenance(root=tmp_path)
    reverse = result["frame"].loc[
        result["frame"]["orientation"].eq("reverse")
    ].iloc[0]

    assert result["status"] == "BLOCKED"
    assert reverse["vendor_contract_url"] == WIZARD_PRESCANNED_CONTRACT_URL
    assert not reverse["vendor_orientation_value_observed"]
    assert "overlay_orientation_not_captured" in reverse["blocker"]


def test_official_ordered_snapshots_can_prove_reverse_provenance_only(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "experiment_orientation": "original",
                "ou_optimal": True,
                "ou_optimal_semantics": "scanner_boolean_annotation",
                "independent_pair_page_mode": False,
                "source_exact_mode": "OU (Spread)",
                "evidence_path": "raw.json",
            }
        ]
    ).to_csv(active / "current_wizard_ou_optimal_overlay_ledger.csv", index=False)
    raw = tmp_path / "data" / "raw" / "crypto_wizards" / "prescanned" / "2026-08-09"
    raw.mkdir(parents=True)
    (raw / "capture.json").write_text(
        json.dumps(
            {
                "capture_metadata": {"captured_at": "2026-08-09T12:00:00Z"},
                "request": {"exchange": "Coinbase", "interval": "Daily"},
                "response": [
                    {
                        "symbol_1": "ADA-USD",
                        "symbol_2": "PENGU-USD",
                        "ou_optimal": False,
                    },
                    {
                        "symbol_1": "PENGU-USD",
                        "symbol_2": "ADA-USD",
                        "ou_optimal": False,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    audit = build_ou_optimal_ordered_orientation_audit(root=tmp_path)
    result = build_ou_optimal_overlay_provenance(root=tmp_path)
    reverse = result["frame"].loc[
        result["frame"]["orientation"].eq("reverse")
    ].iloc[0]

    assert audit["status"] == "PASS"
    assert audit["bidirectional_pair_groups"] == 1
    assert result["status"] == "PASS"
    assert reverse["vendor_orientation_value_observed"]
    assert reverse["row_accounting_complete"]
    assert reverse["blocker"] == ""
    assert reverse["cross_snapshot_hindsight_aggregate"]
    assert not reverse["live_signal_eligible"]
    assert not reverse["promotion_authority"]
    assert reverse["formula_parity_status"] == "UNPROVEN"


def test_completed_orientation_matched_vendor_proof_updates_formula_authority_only(tmp_path):
    active = tmp_path / "reports" / "active"
    active.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_group_key": "dydx|daily|ADA|ALGO",
                "pair": "ADA-ALGO",
                "wizard_exchange": "dydx",
                "timeframe": "daily",
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "capture_status": "CAPTURED",
                "orientation_verified": True,
                "raw_chart_data_preserved": True,
                "capture_timestamp": "2026-08-09T12:00:00Z",
                "periods_analyzed": 360,
                "evidence_path": "raw/capture.json",
            },
            {
                "pair_group_key": "dydx|daily|ADA|ALGO",
                "pair": "ADA-ALGO",
                "wizard_exchange": "dydx",
                "timeframe": "daily",
                "asset_x": "ADA",
                "asset_y": "ALGO",
                "exact_mode": "Dyn (Spread)",
                "orientation": "original",
                "capture_status": "CAPTURED",
                "orientation_verified": True,
                "raw_chart_data_preserved": True,
                "capture_timestamp": "2026-08-09T12:00:00Z",
                "periods_analyzed": 360,
                "evidence_path": "raw/dynamic_capture.json",
            },
        ]
    ).to_csv(active / "exhaustive_wizard_pair_detail_mode_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair": "ADA-ALGO",
                "pair_group_id": "pair-1",
                "exact_mode": "Static (Spread)",
                "orientation": "original",
                "mode_proof_status": "completed",
                "proof_observations": 360,
                "proof_window_kind": "scanner_horizon_parity",
                "vendor_formula_parity_status": "exact_reconstruction",
                "vendor_history_available": True,
                "vendor_spread_max_abs_error": 1e-12,
                "vendor_zscore_max_abs_error": 1e-12,
                "vendor_zscore_roll_max_abs_error": 1e-12,
                "request_path": "raw/request.json",
                "response_path": "raw/response.json",
                "evidence_path": "raw/request.json;raw/response.json",
            },
            {
                "pair": "ADA-ALGO",
                "pair_group_id": "pair-1",
                "exact_mode": "Dyn (Spread)",
                "orientation": "original",
                "mode_proof_status": "completed",
                "vendor_response_captured": True,
                "proof_observations": 360,
                "proof_window_kind": "scanner_horizon_parity",
                "vendor_formula_parity_status": "mismatch",
                "vendor_history_available": True,
                "vendor_spread_max_abs_error": 1.25,
                "vendor_zscore_max_abs_error": 2.5,
                "vendor_zscore_roll_max_abs_error": 3.75,
                "request_path": "raw/dynamic_request.json",
                "response_path": "raw/dynamic_response.json",
                "evidence_path": (
                    "raw/dynamic_request.json;raw/dynamic_response.json"
                ),
            },
        ]
    ).to_csv(active / "hyperliquid_wizard_vendor_mode_proofs.csv", index=False)

    result = build_wizard_mode_authority(root=tmp_path)
    authority = result["frame"]
    proven = authority.loc[
        authority["exact_mode"].eq("Static (Spread)")
        & authority["orientation"].eq("original")
    ].iloc[0]
    reverse = authority.loc[
        authority["exact_mode"].eq("Static (Spread)")
        & authority["orientation"].eq("reverse")
    ].iloc[0]
    dynamic_mismatch = authority.loc[
        authority["exact_mode"].eq("Dyn (Spread)")
        & authority["orientation"].eq("original")
    ].iloc[0]

    assert proven["vendor_exact_mode_parity_proven"]
    assert proven["formula_authority"] == "vendor_custom_series_exact_reconstruction"
    assert proven["wizard_formula_confirmed_claim_allowed"]
    assert not proven["wizard_confirmed_claim_allowed"]
    assert not proven["acceptance_eligible"]
    assert not proven["live_trading_authorized"]
    assert not reverse["vendor_exact_mode_parity_proven"]
    assert dynamic_mismatch["vendor_response_captured"]
    assert not dynamic_mismatch["vendor_exact_mode_parity_proven"]
    assert dynamic_mismatch["vendor_observed_formula_parity_status"] == "mismatch"
    assert dynamic_mismatch["vendor_observed_formula_max_abs_error"] == 3.75
    assert dynamic_mismatch["blocker"] == (
        "vendor_response_captured_but_formula_parity_not_proven:mismatch"
    )
    assert "raw/dynamic_response.json" in dynamic_mismatch["evidence_path"]


def test_mixed_mode_completion_requires_formula_and_copula_provenance(tmp_path):
    active = tmp_path / "reports" / "active"
    config = tmp_path / "config"
    active.mkdir(parents=True)
    config.mkdir(parents=True)
    queue_path = active / "exhaustive_wizard_exact_mode_proof_queue.csv"
    proof_path = active / "hyperliquid_wizard_vendor_mode_proofs.csv"
    detail_path = active / "wizard_copula_behavioral_evaluation.csv"
    contract_path = config / "wizard_copula_behavioral_parity.json"
    receipt_path = active / "wizard_copula_behavioral_contract_receipt.json"
    rows = []
    for exact_mode in ("Static (Spread)", "Copula"):
        for orientation in ORIENTATIONS:
            rows.append(
                {
                    "pair_group_id": "pair-group-1",
                    "pair": "ADA-ALGO" if orientation == "original" else "ALGO-ADA",
                    "local_interval": "1d",
                    "exact_mode": exact_mode,
                    "orientation": orientation,
                    "proof_observations": 100,
                    "vendor_custom_series_eligible": True,
                }
            )
    pd.DataFrame(rows).to_csv(queue_path, index=False)
    proof_rows = []
    for row in rows:
        formula = row["exact_mode"] != "Copula"
        proof_rows.append(
            {
                **row,
                "vendor_response_captured": True,
                "formula_proof_complete": formula,
                "mode_proof_status": "completed" if formula else "response_captured",
                "vendor_formula_parity_status": (
                    "exact_reconstruction" if formula else "not_available"
                ),
                "vendor_history_available": True,
                "proof_window_kind": "scanner_horizon_parity",
            }
        )
    pd.DataFrame(proof_rows).to_csv(proof_path, index=False)
    contract_path.write_text("{}\n", encoding="utf-8")
    receipt_path.write_text("{}\n", encoding="utf-8")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    detail_rows = []
    for orientation in ORIENTATIONS:
        capture_id = f"copulacapture_{orientation}"
        capture_receipt = (
            tmp_path
            / "data"
            / "raw"
            / "crypto_wizards_copula_behavioral_proofs"
            / capture_id
            / "receipt.json"
        )
        capture_receipt.parent.mkdir(parents=True, exist_ok=True)
        capture_receipt.write_text(
            json.dumps({"capture_id": capture_id}), encoding="utf-8"
        )
        detail_rows.append(
            {
                "capture_id": capture_id,
                "pair_group_id": "pair-group-1",
                "orientation": orientation,
                "endpoint_responses_captured": 2,
                "behavioral_status": "PASS",
                "provenance_status": "PASS",
                "request_sha256": f"request-{orientation}",
                "response_1_sha256": f"response-1-{orientation}",
                "response_2_sha256": f"response-2-{orientation}",
                "capture_receipt_path": str(
                    capture_receipt.relative_to(tmp_path)
                ),
                "capture_receipt_sha256": digest(capture_receipt),
                "candidate_promotion_authority": False,
                "testnet_order_authority": False,
                "live_trading_authorized": False,
            }
        )
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    cohort_core = {
        "schema_version": "thewiz.wizard_copula_behavioral_cohort.v1",
        "evaluation_schema_version": (
            "thewiz.wizard_copula_behavioral_evaluation.v3"
        ),
        "contract_path": "config/wizard_copula_behavioral_parity.json",
        "contract_sha256": digest(contract_path),
        "contract_receipt_path": (
            "reports/active/wizard_copula_behavioral_contract_receipt.json"
        ),
        "contract_receipt_sha256": digest(receipt_path),
        "proof_queue_path": (
            "reports/active/exhaustive_wizard_exact_mode_proof_queue.csv"
        ),
        "proof_queue_sha256": digest(queue_path),
        "cells": detail_rows,
        "research_only": True,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
    }
    cohort_bytes = (
        json.dumps(cohort_core, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    cohort_id = "copulacohort_" + hashlib.sha256(cohort_bytes).hexdigest()[:20]
    cohort = {**cohort_core, "cohort_id": cohort_id}
    cohort_path = (
        tmp_path
        / "data"
        / "research"
        / "wizard_copula_behavioral_cohorts"
        / f"{cohort_id}.json"
    )
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_path.write_text(
        json.dumps(cohort, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    status = {
        "status": "PASS",
        "expected_cells": 2,
        "behavioral_cells_passed": 2,
        "provenance_cells_complete": 2,
        "formula_parity_proven": False,
        "candidate_promotion_authority": False,
        "testnet_order_authority": False,
        "live_trading_authorized": False,
        "contract_path": "config/wizard_copula_behavioral_parity.json",
        "contract_sha256": digest(contract_path),
        "contract_receipt_path": (
            "reports/active/wizard_copula_behavioral_contract_receipt.json"
        ),
        "contract_receipt_sha256": digest(receipt_path),
        "proof_queue_path": (
            "reports/active/exhaustive_wizard_exact_mode_proof_queue.csv"
        ),
        "proof_queue_sha256": digest(queue_path),
        "vendor_proof_path": (
            "reports/active/hyperliquid_wizard_vendor_mode_proofs.csv"
        ),
        "vendor_proof_sha256": digest(proof_path),
        "evidence_path": "reports/active/wizard_copula_behavioral_evaluation.csv",
        "evidence_sha256": digest(detail_path),
        "cohort_receipt_id": cohort_id,
        "cohort_receipt_path": str(cohort_path.relative_to(tmp_path)),
        "cohort_receipt_sha256": digest(cohort_path),
    }
    status_path = active / "wizard_copula_behavioral_status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")

    result = build_wizard_mode_evidence_completion(root=tmp_path)

    assert result["status"] == "PASS"
    assert result["formula_queue_cells_passed"] == 2
    assert result["formula_queue_cells_expected"] == 2
    assert result["copula_behavioral_cells_passed"] == 2
    assert result["copula_provenance_cells_complete"] == 2
    assert result["accepted_queue_cells"] == 4
    assert not result["testnet_order_authority"]

    capture_receipt_path = tmp_path / detail_rows[0]["capture_receipt_path"]
    capture_receipt_bytes = capture_receipt_path.read_bytes()
    capture_receipt_path.write_bytes(capture_receipt_bytes + b"\n")
    capture_tampered = build_wizard_mode_evidence_completion(root=tmp_path)
    assert capture_tampered["status"] == "BLOCKED"
    assert "immutable_capture_receipt_mismatch" in capture_tampered["blocker"]
    capture_receipt_path.write_bytes(capture_receipt_bytes)

    detail_path.write_text(
        detail_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    tampered = build_wizard_mode_evidence_completion(root=tmp_path)
    assert tampered["status"] == "BLOCKED"
    assert "binding_hash_mismatch:evidence_path" in tampered["blocker"]
