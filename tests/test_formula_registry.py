from __future__ import annotations

from quant_platform.formula_registry import FORMULAS


def test_ou_optimal_registry_does_not_invent_vendor_threshold_formula():
    registration = FORMULAS["ou_optimal"]

    assert "not exposed" in registration["formula"].lower()
    assert "boolean" in registration["formula"].lower()
    assert "never infer entries or exits" in registration["use_case"].lower()


def test_formula_dictionary_matches_math_v2_ecm_and_hurst_estimators():
    assert "hc1" in FORMULAS["ecm_strength"]["formula"].lower()
    assert "detrended fluctuation analysis" in FORMULAS["hurst"]["formula"].lower()


def test_formula_dictionary_declares_active_math_conventions():
    assert "log(y_t)" in FORMULAS["spread"]["formula"]
    assert "y-on-x" in FORMULAS["spread"]["formula"].lower()
    assert "ddof=1" in FORMULAS["zscore"]["formula"].lower()
    assert "ddof=1" in FORMULAS["sharpe"]["formula"].lower()
    assert "365-day crypto" in FORMULAS["sharpe"]["formula"].lower()
    assert "initial equity 1.0" in FORMULAS["drawdown"]["formula"].lower()
