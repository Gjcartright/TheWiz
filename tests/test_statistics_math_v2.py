from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_platform.statistics.math_v2 import (
    estimate_hurst_dfa,
    fit_ecm,
    fit_engle_granger,
    fit_gaussian_copula,
    fit_ou,
    rolling_gaussian_copula_conditionals,
    rolling_zscore_variants,
)


def _cointegrated_prices(rows: int = 600):
    rng = np.random.default_rng(5)
    log_x = 4.0 + np.cumsum(rng.normal(0.0, 0.01, rows))
    residual = np.zeros(rows)
    for index in range(1, rows):
        residual[index] = 0.8 * residual[index - 1] + rng.normal(0.0, 0.008)
    log_y = 0.3 + 1.1 * log_x + residual
    return pd.Series(np.exp(log_x)), pd.Series(np.exp(log_y)), pd.Series(residual)


def test_zscore_variants_are_both_present_and_causal():
    values = pd.Series(np.arange(20), dtype=float)
    changed = values.copy()
    changed.iloc[-1] = 10_000
    first = rolling_zscore_variants(values, window=5)
    second = rolling_zscore_variants(changed, window=5)
    assert list(first.columns) == ["zscore_ddof0", "zscore_ddof1"]
    assert first.iloc[:-1].equals(second.iloc[:-1])


def test_cointegration_and_ou_estimators_recover_known_process():
    price_x, price_y, residual = _cointegrated_prices()
    cointegration = fit_engle_granger(price_x, price_y)
    ou = fit_ou(residual)
    assert cointegration.validity_status == "valid"
    assert cointegration.values["cointegration_pvalue"] < 0.05
    assert cointegration.values["alpha"] == pytest.approx(0.3, abs=0.15)
    assert cointegration.values["hedge_ratio"] == pytest.approx(1.1, abs=0.04)
    assert cointegration.values["hedge_ratio_orientation"] == "beta_y_on_x"
    expected_residual = (
        np.log(price_y)
        - cointegration.values["alpha"]
        - cointegration.values["hedge_ratio"] * np.log(price_x)
    )
    pd.testing.assert_series_equal(cointegration.values["residual"], expected_residual)
    assert ou.validity_status == "valid"
    assert 0.7 < ou.values["phi"] < 0.9
    assert ou.values["half_life"] > 0


def test_invalid_ou_is_not_clipped_into_validity():
    result = fit_ou(pd.Series([1.01**index for index in range(200)], dtype=float))
    assert result.validity_status == "invalid"
    assert result.values["phi"] >= 1.0
    assert "half_life" not in result.values


def test_ou_rejects_nonpositive_time_step():
    result = fit_ou(pd.Series(np.arange(100), dtype=float), delta_t=0.0)

    assert result.validity_status == "invalid"
    assert result.validity_reason == "invalid_delta_t"


def test_ecm_recovers_expected_two_leg_correction_signs():
    rng = np.random.default_rng(101)
    rows = 1500
    common = 4.0 + np.cumsum(rng.normal(0.0, 0.006, rows))
    residual = np.zeros(rows)
    for index in range(1, rows):
        residual[index] = 0.65 * residual[index - 1] + rng.normal(0.0, 0.004)
    log_x = common - 0.5 * residual
    log_y = 0.2 + common + 0.5 * residual
    price_x = pd.Series(np.exp(log_x))
    price_y = pd.Series(np.exp(log_y))

    cointegration = fit_engle_granger(price_x, price_y)
    ecm = fit_ecm(price_x, price_y, cointegration)

    assert ecm.validity_status == "valid"
    assert ecm.values["gamma_x"] > 0.0
    assert ecm.values["gamma_y"] < 0.0
    assert ecm.values["ecm_strength"] == 1.0
    assert ecm.values["expected_gamma_x_sign"] == "positive"
    assert ecm.values["expected_gamma_y_sign"] == "negative"


def test_hurst_has_minimum_sample_gate():
    result = estimate_hurst_dfa(pd.Series(range(40), dtype=float))
    assert result.validity_status == "invalid"
    assert result.validity_reason == "insufficient_rows"


def test_gaussian_copula_returns_bounded_conditional_cdfs():
    rng = np.random.default_rng(13)
    samples = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], size=500)
    result = fit_gaussian_copula(pd.Series(samples[:, 0]), pd.Series(samples[:, 1]))
    assert result.validity_status == "valid"
    assert result.method_id == "gaussian_copula_empirical_cdf"
    assert result.values["u1_given_u2"].dropna().between(0, 1).all()


def test_rolling_copula_conditionals_do_not_change_past_rows():
    rng = np.random.default_rng(19)
    samples = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=180)
    x = pd.Series(samples[:, 0])
    y = pd.Series(samples[:, 1])
    changed = x.copy()
    changed.iloc[-1] = 100.0
    first = rolling_gaussian_copula_conditionals(x, y, window=80, min_rows=40)
    second = rolling_gaussian_copula_conditionals(changed, y, window=80, min_rows=40)
    pd.testing.assert_frame_equal(first.iloc[:-1], second.iloc[:-1])
    assert first["u1_given_u2"].dropna().between(0, 1).all()
