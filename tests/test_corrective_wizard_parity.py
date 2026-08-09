from __future__ import annotations

import numpy as np

from quant_platform.orchestration.corrective_wizard_parity import (
    EXACT_MODES,
    ORIENTATIONS,
    compare_series,
    run_wizard_mode_mutation_tests,
)


def test_parity_comparison_accepts_match_and_rejects_mutation():
    baseline = np.arange(20, dtype=float)
    assert compare_series(baseline, baseline.copy())["status"] == "PASS"
    assert compare_series(baseline, baseline * -1)["status"] == "FAIL"
    assert compare_series(baseline, baseline[:-1])["status"] == "BLOCKED"


def test_all_eight_modes_and_both_orientations_are_accounted():
    assert len(EXACT_MODES) == 8
    assert len(ORIENTATIONS) == 2
    assert len({(mode, orientation) for mode in EXACT_MODES for orientation in ORIENTATIONS}) == 16


def test_formula_mutations_cannot_receive_parity(tmp_path):
    frame = run_wizard_mode_mutation_tests(root=tmp_path)
    assert frame["status"].eq("PASS").all()
    assert not frame["mislabeled_mode_can_receive_parity"].any()
