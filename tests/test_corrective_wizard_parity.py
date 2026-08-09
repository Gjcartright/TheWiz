from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.orchestration.corrective_wizard_parity import (
    EXACT_MODES,
    ORIENTATIONS,
    build_ou_optimal_overlay_provenance,
    compare_series,
    run_wizard_mode_mutation_tests,
)


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


def test_formula_mutations_cannot_receive_parity(tmp_path):
    frame = run_wizard_mode_mutation_tests(root=tmp_path)
    assert frame["status"].eq("PASS").all()
    assert not frame["mislabeled_mode_can_receive_parity"].any()


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
