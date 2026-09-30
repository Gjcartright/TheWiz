"""Isolated RT02/03/04 unit regressions; no strategy or external side effects.

Identity fingerprints are local integrity evidence, not signatures or proof of
causal data availability. Numerical validity is not statistical eligibility.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from quant_platform.statistics import math_v2 as subject


def _prices(seed=12, beta=1.15, rows=256):
    random = np.random.default_rng(seed)
    log_x = 4.0 + np.cumsum(random.normal(0.0, 0.02, rows))
    residual = np.zeros(rows)
    for position in range(1, rows):
        residual[position] = 0.55 * residual[position - 1] + random.normal(0.0, 0.006)
    log_y = 0.2 + beta * log_x + residual
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC", name="bar")
    return pd.Series(np.exp(log_x), index=index, name="X"), pd.Series(
        np.exp(log_y), index=index, name="Y"
    )


@pytest.fixture
def pair():
    x, y = _prices()
    estimate = subject.fit_engle_granger(x, y)
    assert estimate.validity_status == "valid", estimate.validity_reason
    return x, y, estimate


def _with_values(estimate, **updates):
    return replace(estimate, values={**estimate.values, **updates})


def test_generated_relationship_binding_preserves_existing_ecm_equations(pair):
    x, y, estimate = pair
    result = subject.fit_ecm(x.copy(), y.copy(), estimate)
    assert result.validity_status == "valid", result.validity_reason
    assert estimate.estimator_version == result.estimator_version == subject.MATH_VERSION
    assert estimate.values["component_contract"] == subject.EG_ECM_COMPONENT_CONTRACT
    assert result.values["component_contract"] == subject.EG_ECM_COMPONENT_CONTRACT
    assert result.values["sample_identity"] == estimate.values["sample_identity"]
    assert result.values["relationship_id"] == estimate.values["relationship_id"]
    for key in ("sample_sha256", "price_sha256", "index_sha256"):
        assert len(result.values["sample_identity"][key]) == 64
    assert len(result.values["relationship_id"]) == 64
    # Independent design construction preserves the two original HC1 equations.
    lx, ly = np.log(x.to_numpy()), np.log(y.to_numpy())
    ect = ly - estimate.values["alpha"] - estimate.values["hedge_ratio"] * lx
    dx, dy = np.diff(lx), np.diff(ly)
    regressors = np.column_stack([np.ones(len(x) - 2), ect[1:-1], dx[:-1], dy[:-1]])
    for leg, response in (("x", dx[1:]), ("y", dy[1:])):
        oracle = sm.OLS(response, regressors).fit(cov_type="HC1")
        assert result.values[f"gamma_{leg}"] == pytest.approx(oracle.params[1], abs=1e-12)
        assert result.values[f"gamma_{leg}_standard_error"] == pytest.approx(oracle.bse[1], abs=1e-12)
        assert result.values[f"gamma_{leg}_pvalue"] == pytest.approx(oracle.pvalues[1], abs=1e-12)
    assert result.lookback_rows == len(x) - 2
    assert result.values["error_term_formula"] == "log_y-alpha-beta_y_on_x*log_x"


def test_rt02_foreign_genuine_relationship_cannot_be_reused(pair):
    _, _, estimate = pair
    foreign_x, foreign_y = _prices(seed=24, beta=0.9)
    result = subject.fit_ecm(foreign_x, foreign_y, estimate)
    assert result.validity_status == "invalid"
    assert result.validity_reason == "cointegration_sample_identity_mismatch"
    assert "ecm_strength" not in result.values


@pytest.mark.parametrize("leg", ["x", "y"])
def test_identity_binds_exact_raw_prices_not_only_rounded_log_prices(pair, leg):
    x, y, estimate = pair
    changed = x if leg == "x" else y
    changed.iloc[35] = np.nextafter(changed.iloc[35], np.inf)
    result = subject.fit_ecm(x, y, estimate)
    assert result.validity_reason == "cointegration_sample_identity_mismatch"


@pytest.mark.parametrize("defect", ["zero", "single_value", "nan", "inf", "complex", "bool", "reordered", "missing"])
def test_rt03_tampered_ect_never_yields_valid_strength(pair, defect):
    x, y, estimate = pair
    residual = estimate.values["residual"].copy()
    if defect == "zero":
        residual[:] = 0.0
    elif defect == "single_value":
        residual.iloc[20] += 0.01
    elif defect in {"nan", "inf"}:
        residual.iloc[20] = np.nan if defect == "nan" else np.inf
    elif defect == "complex":
        residual = residual.astype(complex)
        residual.iloc[20] += 1j
    elif defect == "bool":
        residual = residual.astype(bool)
    elif defect == "reordered":
        residual = residual.iloc[::-1]
    else:
        residual = residual.iloc[:-1]
    result = subject.fit_ecm(x, y, _with_values(estimate, residual=residual))
    assert result.validity_status == "invalid"
    assert "ecm_strength" not in result.values


@pytest.mark.parametrize("key,value", [
    ("component_contract", "old-or-foreign"),
    ("sample_identity", {}),
    ("relationship_id", "0" * 64),
    ("asset_order", "y_then_x"),
    ("hedge_ratio_orientation", "beta_x_on_y"),
    ("regression_formula", "y=alpha+beta*x+residual"),
    ("residual_formula", "log_x-alpha-beta_y_on_x*log_y"),
    ("alpha", np.inf),
    ("alpha", True),
    ("hedge_ratio", np.nan),
    ("hedge_ratio", 1j),
])
def test_relationship_contract_and_equation_mutations_are_rejected(pair, key, value):
    x, y, estimate = pair
    result = subject.fit_ecm(x, y, _with_values(estimate, **{key: value}))
    assert result.validity_status == "invalid"


@pytest.mark.parametrize("change", [
    {"method_id": "static_log_y_on_x_ols"},
    {"estimator_version": "old-math-version"},
    {"lookback_rows": 255},
    {"lookback_rows": True},
])
def test_wrong_estimator_identity_is_rejected(pair, change):
    x, y, estimate = pair
    assert subject.fit_ecm(x, y, replace(estimate, **change)).validity_status == "invalid"


def test_legacy_unbound_and_non_estimator_inputs_fail_closed(pair):
    x, y, estimate = pair
    legacy = {key: value for key, value in estimate.values.items()
              if key not in {"component_contract", "sample_identity", "relationship_id"}}
    result = subject.fit_ecm(x, y, replace(estimate, values=legacy))
    assert result.validity_reason == "cointegration_component_contract_missing_or_mismatched"
    assert subject.fit_ecm(x, y, legacy).validity_reason == "cointegration_estimate_type_invalid"


@pytest.mark.parametrize("change", ["asset_name", "index_name", "timezone", "index_type"])
def test_sample_metadata_changes_require_relationship_regeneration(pair, change):
    x, y, estimate = pair
    if change == "asset_name":
        x = x.rename("OTHER_X")
    elif change == "index_name":
        x, y = x.rename_axis("other_bar"), y.rename_axis("other_bar")
    elif change == "timezone":
        index = x.index.tz_convert("America/New_York")
        x, y = x.set_axis(index), y.set_axis(index)
    else:
        index = pd.RangeIndex(len(x))
        x, y = x.set_axis(index), y.set_axis(index)
    result = subject.fit_ecm(x, y, estimate)
    assert result.validity_reason == "cointegration_sample_identity_mismatch"


def test_paired_index_metadata_must_match_before_regression(pair):
    x, y, estimate = pair
    y = y.rename_axis("different_index_name")
    assert subject.fit_engle_granger(x, y).validity_reason == "price_index_identity_mismatch"
    assert subject.fit_ecm(x, y, estimate).validity_reason == "price_index_identity_mismatch"


@pytest.mark.parametrize("defect", ["shuffled", "reversed", "gap", "duplicate", "nat", "object_labels"])
def test_rt04_temporal_estimators_reject_bad_clocks_without_sorting(pair, defect):
    x, y, estimate = pair
    if defect in {"shuffled", "reversed"}:
        order = (np.random.default_rng(9).permutation(len(x)) if defect == "shuffled"
                 else np.arange(len(x) - 1, -1, -1))
        x, y = x.iloc[order], y.iloc[order]
    else:
        values = list(x.index)
        if defect == "gap":
            values[40:] = [value + pd.Timedelta(hours=1) for value in values[40:]]
        elif defect == "duplicate":
            values[40] = values[39]
        elif defect == "nat":
            values[40] = pd.NaT
        else:
            values = [str(value) for value in values]
        index = pd.Index(values, name="bar")
        x, y = x.set_axis(index), y.set_axis(index)
    for result in (subject.fit_engle_granger(x, y), subject.fit_ecm(x, y, estimate)):
        assert result.validity_status == "invalid"
        assert "ecm_strength" not in result.values


@pytest.mark.parametrize("clock", ["position", "float", "timedelta", "period"])
def test_explicit_regular_numeric_and_temporal_clocks_are_supported(pair, clock):
    x, y, _ = pair
    if clock == "position":
        index = pd.RangeIndex(len(x))
    elif clock == "float":
        index = pd.Index(np.arange(len(x), dtype=float) * 0.25)
    elif clock == "timedelta":
        index = pd.timedelta_range(start="0h", periods=len(x), freq="h")
    else:
        index = pd.period_range(start="2024-01-01", periods=len(x), freq="D")
    x, y = x.set_axis(index), y.set_axis(index)
    estimate = subject.fit_engle_granger(x, y)
    assert estimate.validity_status == "valid", estimate.validity_reason
    result = subject.fit_ecm(x, y, estimate)
    assert result.validity_status == "valid", result.validity_reason


def test_standalone_static_ols_retains_order_independence(pair):
    x, y, _ = pair
    order = np.random.default_rng(9).permutation(len(x))
    original = subject.fit_static_y_on_x_ols(x, y)
    shuffled = subject.fit_static_y_on_x_ols(x.iloc[order], y.iloc[order])
    assert original.validity_status == shuffled.validity_status == "valid"
    assert shuffled.values["alpha"] == pytest.approx(original.values["alpha"], abs=1e-12)
    assert shuffled.values["hedge_ratio"] == pytest.approx(original.values["hedge_ratio"], abs=1e-12)


def test_nonsignificance_is_not_mislabeled_numerical_invalidity(pair, monkeypatch):
    x, y, _ = pair
    # Deterministic inference-boundary substitution; not a statistical oracle.
    monkeypatch.setattr(subject, "coint", lambda *args, **kwargs: (-1.0, 0.8, [-4.0, -3.0, -2.0]))
    estimate = subject.fit_engle_granger(x, y)
    assert estimate.validity_status == "valid"
    assert estimate.values["cointegrated_5pct"] is False
    assert subject.fit_ecm(x, y, estimate).validity_status == "valid"


def test_rank_guard_is_independent_of_relationship_guard(pair, monkeypatch):
    x, y, estimate = pair
    # Isolate rank validation after the separately tested identity gate.
    monkeypatch.setattr(subject, "_ecm_relationship_reason", lambda *args: "")
    zero_ect = pd.Series(0.0, index=x.index)
    result = subject.fit_ecm(x, y, _with_values(estimate, residual=zero_ect))
    assert result.validity_reason == "unidentified_ecm_design"
    assert "ecm_strength" not in result.values


def _inference_double():
    return SimpleNamespace(
        model=SimpleNamespace(rank=4), df_resid=16,
        params=np.array([0.1, 0.2, 0.3, 0.4]),
        bse=np.array([0.1, 0.1, 0.1, 0.1]),
        pvalues=np.array([0.3, 0.04, 0.2, 0.3]),
        tvalues=np.array([1.0, 2.0, 3.0, 4.0]),
        resid=np.ones(20), cov_params=lambda: np.eye(4) * 0.01,
    )


@pytest.mark.parametrize("defect", ["params", "bse", "pvalues", "tvalues", "resid", "covariance", "negative_se", "pvalue_range", "rank", "dof", "shape", "complex"])
def test_inference_guard_rejects_invalid_or_unidentified_outputs(defect):
    regressors = pd.DataFrame(np.ones((20, 4)))  # Rank checked separately before fit.
    model = _inference_double()
    assert subject._identified_finite_inference(model, regressors)
    if defect in {"params", "bse", "pvalues", "tvalues", "resid"}:
        getattr(model, defect)[0] = np.nan
    elif defect == "covariance":
        model.cov_params = lambda: np.full((4, 4), np.inf)
    elif defect == "negative_se":
        model.bse[0] = -0.01
    elif defect == "pvalue_range":
        model.pvalues[0] = 1.1
    elif defect == "rank":
        model.model.rank = 3
    elif defect == "dof":
        model.df_resid = 0
    elif defect == "shape":
        model.params = np.ones(3)
    else:
        model.params = model.params.astype(complex)
    assert not subject._identified_finite_inference(model, regressors)


def test_fit_ecm_enforces_inference_guard(pair, monkeypatch):
    x, y, estimate = pair
    monkeypatch.setattr(subject, "_identified_finite_inference", lambda *args: False)
    result = subject.fit_ecm(x, y, estimate)
    assert result.validity_reason == "invalid_ecm_inference"
    assert "ecm_strength" not in result.values
