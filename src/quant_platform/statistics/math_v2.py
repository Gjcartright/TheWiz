"""Math V2 estimators with explicit validity and provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller, coint

from quant_platform.performance_math import MATH_VERSION


@dataclass(frozen=True)
class EstimatorResult:
    method_id: str
    estimator_version: str
    validity_status: str
    validity_reason: str
    values: dict[str, Any]
    lookback_rows: int


def rolling_zscore_variants(
    values: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Return both population and sample rolling z-scores for parity work."""

    numeric = pd.to_numeric(values, errors="coerce")
    window = max(2, int(window))
    minimum = max(2, min(int(min_periods or window), window))
    rolling = numeric.rolling(window, min_periods=minimum)
    mean = rolling.mean()
    std0 = rolling.std(ddof=0).replace(0.0, np.nan)
    std1 = rolling.std(ddof=1).replace(0.0, np.nan)
    return pd.DataFrame(
        {
            "zscore_ddof0": (numeric - mean) / std0,
            "zscore_ddof1": (numeric - mean) / std1,
        },
        index=values.index,
    )


def fit_engle_granger(
    price_x: pd.Series,
    price_y: pd.Series,
    *,
    min_rows: int = 60,
) -> EstimatorResult:
    """Fit the canonical log(Y)-on-log(X) Engle-Granger relation."""

    method = "engle_granger_log_ols"
    if not price_x.index.identical(price_y.index):
        return _invalid(method, "price_index_identity_mismatch", len(price_x))
    if len(price_x) < min_rows:
        return _invalid(method, "insufficient_rows", len(price_x))
    grid_reason = _temporal_pair_grid_reason(price_x.index)
    if grid_reason:
        return _invalid(method, grid_reason, len(price_x))
    paired = pd.concat(
        [pd.to_numeric(price_x, errors="coerce"), pd.to_numeric(price_y, errors="coerce")],
        axis=1,
        keys=["x", "y"],
    )
    if np.iscomplexobj(paired.to_numpy()):
        return _invalid(method, "non_finite_prices", len(price_x))
    try:
        prices = paired.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return _invalid(method, "non_finite_prices", len(price_x))
    if not np.isfinite(prices).all():
        return _invalid(method, "non_finite_prices", len(price_x))
    if not (prices > 0.0).all():
        return _invalid(method, "nonpositive_prices", len(price_x))
    data = _positive_log_prices(price_x, price_y)
    if len(data) < min_rows:
        return _invalid("engle_granger_log_ols", "insufficient_rows", len(data))
    try:
        design = sm.add_constant(data["log_x"], has_constant="add")
        regression = sm.OLS(data["log_y"], design).fit(cov_type="HC1")
        alpha = float(regression.params["const"])
        hedge_ratio = float(regression.params["log_x"])
        residual = data["log_y"] - alpha - hedge_ratio * data["log_x"]
        test_statistic, pvalue, critical_values = coint(
            data["log_y"], data["log_x"], trend="c", autolag="aic"
        )
        adf_statistic, adf_pvalue, used_lag, *_ = adfuller(residual, regression="c", autolag="AIC")
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _invalid(
            "engle_granger_log_ols", f"estimation_error:{type(exc).__name__}", len(data)
        )
    values = {
        "alpha": alpha,
        "hedge_ratio": hedge_ratio,
        "asset_order": "x_then_y",
        "hedge_ratio_orientation": "beta_y_on_x",
        "regression_formula": "log_y=alpha+beta_y_on_x*log_x+residual",
        "residual_formula": "log_y-alpha-beta_y_on_x*log_x",
        "residual": residual.reindex(price_x.index),
        "cointegration_test_statistic": float(test_statistic),
        "cointegration_pvalue": float(pvalue),
        "cointegration_critical_1pct": float(critical_values[0]),
        "cointegration_critical_5pct": float(critical_values[1]),
        "cointegration_critical_10pct": float(critical_values[2]),
        "residual_adf_statistic": float(adf_statistic),
        "residual_adf_pvalue": float(adf_pvalue),
        "residual_adf_used_lag": int(used_lag),
        "cointegrated_5pct": bool(pvalue < 0.05),
    }
    return EstimatorResult(
        "engle_granger_log_ols",
        MATH_VERSION,
        "valid",
        "estimated_y_on_x_with_intercept_and_aic_lag_selection",
        values,
        len(data),
    )


def _temporal_pair_grid_reason(index: pd.Index) -> str:
    """Require an ordered, regular, explicitly typed observation grid."""

    temporal = isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex))
    integer = pd.api.types.is_integer_dtype(index.dtype)
    floating = pd.api.types.is_float_dtype(index.dtype)
    if not temporal and not integer and not floating:
        return "unsupported_temporal_index"
    if len(index) < 2:
        return "insufficient_temporal_observations"
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        return "invalid_observation_grid"
    if temporal:
        coordinates = [int(value) for value in index.asi8]
    elif integer:
        coordinates = [int(value) for value in index]
    else:
        coordinates = [float(value) for value in index]
        if not all(math.isfinite(value) for value in coordinates):
            return "invalid_observation_grid"
    differences = [right - left for left, right in zip(coordinates, coordinates[1:])]
    if not all(math.isfinite(value) and value > 0 for value in differences):
        return "invalid_observation_grid"
    if floating:
        regular = all(
            math.isclose(value, differences[0], rel_tol=1e-9, abs_tol=0.0)
            for value in differences
        )
    else:
        regular = all(value == differences[0] for value in differences)
    return "" if regular else "irregular_observation_grid"


def fit_ou(spread: pd.Series, *, min_rows: int = 60, delta_t: float = 1.0) -> EstimatorResult:
    """Fit an AR(1) representation of OU and reject non-mean-reverting phi."""

    numeric = pd.to_numeric(spread, errors="coerce").dropna()
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        return _invalid("ou_ar1_with_intercept", "invalid_delta_t", len(numeric))
    if len(numeric) < min_rows:
        return _invalid("ou_ar1_with_intercept", "insufficient_rows", len(numeric))
    lagged = numeric.shift(1).dropna()
    current = numeric.loc[lagged.index]
    try:
        model = sm.OLS(current, sm.add_constant(lagged, has_constant="add")).fit(cov_type="HC1")
        intercept = float(model.params.iloc[0])
        phi = float(model.params.iloc[1])
    except (ValueError, np.linalg.LinAlgError, IndexError) as exc:
        return _invalid(
            "ou_ar1_with_intercept", f"estimation_error:{type(exc).__name__}", len(numeric)
        )
    if not 0.0 < phi < 1.0:
        return EstimatorResult(
            "ou_ar1_with_intercept",
            MATH_VERSION,
            "invalid",
            "phi_outside_open_unit_interval",
            {"intercept": intercept, "phi": phi},
            len(numeric),
        )
    theta = -math.log(phi) / delta_t
    half_life = math.log(2.0) / theta
    mu = intercept / (1.0 - phi)
    innovation_sigma = float(np.std(model.resid, ddof=1))
    continuous_sigma = innovation_sigma * math.sqrt(
        (2.0 * theta) / max(1.0 - phi**2, np.finfo(float).eps)
    )
    confidence = model.conf_int(alpha=0.05)
    return EstimatorResult(
        "ou_ar1_with_intercept",
        MATH_VERSION,
        "valid",
        "phi_in_open_unit_interval",
        {
            "intercept": intercept,
            "phi": phi,
            "phi_ci_low": float(confidence.iloc[1, 0]),
            "phi_ci_high": float(confidence.iloc[1, 1]),
            "mu": mu,
            "theta": theta,
            "half_life": half_life,
            "innovation_sigma": innovation_sigma,
            "continuous_sigma": continuous_sigma,
        },
        len(numeric),
    )


def estimate_hurst_dfa(series: pd.Series, *, min_rows: int = 128) -> EstimatorResult:
    """Estimate the DFA scaling exponent using detrended window residuals."""

    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < min_rows:
        return _invalid("hurst_dfa_linear", "insufficient_rows", len(values))
    profile = np.cumsum(values - values.mean())
    maximum_scale = len(values) // 4
    scales = np.unique(
        np.floor(np.logspace(np.log10(4), np.log10(maximum_scale), num=12)).astype(int)
    )
    fluctuations: list[float] = []
    valid_scales: list[int] = []
    for scale in scales:
        segment_count = len(profile) // scale
        if segment_count < 4:
            continue
        residual_squares: list[float] = []
        time = np.arange(scale, dtype=float)
        for segment in range(segment_count):
            values_segment = profile[segment * scale : (segment + 1) * scale]
            trend = np.polyval(np.polyfit(time, values_segment, 1), time)
            residual_squares.append(float(np.mean((values_segment - trend) ** 2)))
        fluctuation = math.sqrt(float(np.mean(residual_squares)))
        if fluctuation > 0.0 and math.isfinite(fluctuation):
            valid_scales.append(int(scale))
            fluctuations.append(fluctuation)
    if len(valid_scales) < 5:
        return _invalid("hurst_dfa_linear", "insufficient_valid_scales", len(values))
    slope, intercept = np.polyfit(np.log(valid_scales), np.log(fluctuations), 1)
    predicted = intercept + slope * np.log(valid_scales)
    residual = np.log(fluctuations) - predicted
    total = np.log(fluctuations) - np.log(fluctuations).mean()
    r_squared = 1.0 - float(np.sum(residual**2) / np.sum(total**2)) if np.sum(total**2) > 0 else 0.0
    if not math.isfinite(float(slope)):
        return _invalid("hurst_dfa_linear", "non_finite_estimate", len(values))
    return EstimatorResult(
        "hurst_dfa_linear",
        MATH_VERSION,
        "valid",
        "minimum_rows_and_scale_coverage_satisfied",
        {"hurst": float(slope), "r_squared": r_squared, "scales": tuple(valid_scales)},
        len(values),
    )


def fit_ecm(
    price_x: pd.Series, price_y: pd.Series, engle_granger: EstimatorResult
) -> EstimatorResult:
    """Estimate two robust single-lag error-correction equations."""

    if engle_granger.validity_status != "valid":
        return _invalid(
            "ecm_two_equation_hc1", "cointegration_estimate_invalid", engle_granger.lookback_rows
        )
    data = _positive_log_prices(price_x, price_y)
    residual = engle_granger.values["residual"].reindex(data.index)
    design = pd.DataFrame(
        {
            "ec_term": residual.shift(1),
            "dx_lag": data["log_x"].diff().shift(1),
            "dy_lag": data["log_y"].diff().shift(1),
            "dx": data["log_x"].diff(),
            "dy": data["log_y"].diff(),
        }
    ).dropna()
    if len(design) < 40:
        return _invalid("ecm_two_equation_hc1", "insufficient_rows", len(design))
    regressors = sm.add_constant(design[["ec_term", "dx_lag", "dy_lag"]], has_constant="add")
    try:
        model_x = sm.OLS(design["dx"], regressors).fit(cov_type="HC1")
        model_y = sm.OLS(design["dy"], regressors).fit(cov_type="HC1")
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _invalid(
            "ecm_two_equation_hc1", f"estimation_error:{type(exc).__name__}", len(design)
        )
    gamma_x = float(model_x.params["ec_term"])
    gamma_y = float(model_y.params["ec_term"])
    pvalue_x = float(model_x.pvalues["ec_term"])
    pvalue_y = float(model_y.pvalues["ec_term"])
    # e_t = log(Y_t) - alpha - beta*log(X_t). If e_t is positive, Y is
    # rich relative to X, so correction implies X rises and/or Y falls.
    supported_x = pvalue_x < 0.05 and gamma_x > 0.0
    supported_y = pvalue_y < 0.05 and gamma_y < 0.0
    strength = (float(supported_x) + float(supported_y)) / 2.0
    return EstimatorResult(
        "ecm_two_equation_hc1",
        MATH_VERSION,
        "valid",
        "robust_two_equation_fit",
        {
            "gamma_x": gamma_x,
            "gamma_y": gamma_y,
            "gamma_x_standard_error": float(model_x.bse["ec_term"]),
            "gamma_y_standard_error": float(model_y.bse["ec_term"]),
            "gamma_x_pvalue": pvalue_x,
            "gamma_y_pvalue": pvalue_y,
            "error_term_formula": "log_y-alpha-beta_y_on_x*log_x",
            "expected_gamma_x_sign": "positive",
            "expected_gamma_y_sign": "negative",
            "ecm_strength": strength,
            "ecm_strength_method": "share_of_significant_expected_sign_adjustment_coefficients",
        },
        len(design),
    )


def fit_gaussian_copula(
    return_x: pd.Series, return_y: pd.Series, *, min_rows: int = 60
) -> EstimatorResult:
    """Fit a Gaussian copula and calculate actual conditional CDF values."""

    data = pd.concat(
        [pd.to_numeric(return_x, errors="coerce"), pd.to_numeric(return_y, errors="coerce")],
        axis=1,
        keys=["x", "y"],
    ).dropna()
    if len(data) < min_rows:
        return _invalid("gaussian_copula_empirical_cdf", "insufficient_rows", len(data))
    pseudo_x = data["x"].rank(method="average") / (len(data) + 1.0)
    pseudo_y = data["y"].rank(method="average") / (len(data) + 1.0)
    normal_x = pd.Series(norm.ppf(pseudo_x), index=data.index)
    normal_y = pd.Series(norm.ppf(pseudo_y), index=data.index)
    rho = float(normal_x.corr(normal_y))
    if not math.isfinite(rho) or abs(rho) >= 1.0:
        return _invalid("gaussian_copula_empirical_cdf", "singular_dependence", len(data))
    denominator = math.sqrt(1.0 - rho**2)
    conditional_x = pd.Series(norm.cdf((normal_x - rho * normal_y) / denominator), index=data.index)
    conditional_y = pd.Series(norm.cdf((normal_y - rho * normal_x) / denominator), index=data.index)
    return EstimatorResult(
        "gaussian_copula_empirical_cdf",
        MATH_VERSION,
        "valid",
        "empirical_pseudo_observations_and_gaussian_conditional_cdf",
        {
            "copula_family": "gaussian",
            "rho": rho,
            "u1_given_u2": conditional_x.reindex(return_x.index),
            "u2_given_u1": conditional_y.reindex(return_x.index),
            "conditional_probability_distortion": (conditional_x - conditional_y).reindex(
                return_x.index
            ),
            "lower_tail_dependence": 0.0,
            "upper_tail_dependence": 0.0,
        },
        len(data),
    )


def rolling_gaussian_copula_conditionals(
    return_x: pd.Series,
    return_y: pd.Series,
    *,
    window: int = 120,
    min_rows: int = 60,
) -> pd.DataFrame:
    """Calculate causal rolling Gaussian-copula conditional CDF values."""

    window = max(int(window), int(min_rows))
    data = pd.concat(
        [pd.to_numeric(return_x, errors="coerce"), pd.to_numeric(return_y, errors="coerce")],
        axis=1,
        keys=["x", "y"],
    )
    output = pd.DataFrame(
        np.nan,
        index=data.index,
        columns=["copula_rho", "u1_given_u2", "u2_given_u1", "conditional_probability_distortion"],
    )
    for location in range(len(data)):
        sample = data.iloc[max(0, location - window + 1) : location + 1].dropna()
        if len(sample) < min_rows or data.iloc[location].isna().any():
            continue
        pseudo_x = sample["x"].rank(method="average") / (len(sample) + 1.0)
        pseudo_y = sample["y"].rank(method="average") / (len(sample) + 1.0)
        normal_x = pd.Series(norm.ppf(pseudo_x), index=sample.index)
        normal_y = pd.Series(norm.ppf(pseudo_y), index=sample.index)
        rho = float(normal_x.corr(normal_y))
        if not math.isfinite(rho) or abs(rho) >= 1.0:
            continue
        row_index = data.index[location]
        if row_index not in normal_x.index:
            continue
        denominator = math.sqrt(1.0 - rho**2)
        conditional_x = float(
            norm.cdf((normal_x.loc[row_index] - rho * normal_y.loc[row_index]) / denominator)
        )
        conditional_y = float(
            norm.cdf((normal_y.loc[row_index] - rho * normal_x.loc[row_index]) / denominator)
        )
        output.loc[row_index] = [rho, conditional_x, conditional_y, conditional_x - conditional_y]
    return output


def attach_math_v2_statistics(
    rows: list[dict[str, Any]],
    *,
    zscore_window: int,
    min_zscore_window: int,
) -> dict[str, Any]:
    """Attach namespaced Math V2 fields without promoting legacy proxies."""

    if not rows:
        return {
            "math_version": MATH_VERSION,
            "status": "invalid",
            "reason": "empty_history",
            "audit": [],
        }
    frame = pd.DataFrame(rows)
    price_x = pd.to_numeric(frame.get("price_x"), errors="coerce")
    price_y = pd.to_numeric(frame.get("price_y"), errors="coerce")
    engle_granger = fit_engle_granger(price_x, price_y)
    legacy_spread = pd.to_numeric(frame.get("spread"), errors="coerce")
    residual = engle_granger.values.get("residual")
    analysis_spread = (
        residual.reindex(frame.index) if isinstance(residual, pd.Series) else legacy_spread
    )
    frame["math_v2_log_spread"] = analysis_spread
    zscores = rolling_zscore_variants(
        analysis_spread,
        window=zscore_window,
        min_periods=min_zscore_window,
    )
    for column in zscores:
        frame[f"math_v2_{column}"] = zscores[column]

    ou = fit_ou(analysis_spread)
    hurst = estimate_hurst_dfa(analysis_spread)
    ecm = fit_ecm(price_x, price_y, engle_granger)
    copula = fit_gaussian_copula(price_x.pct_change(), price_y.pct_change())

    _attach_scalar(frame, "math_v2_cointegration_pvalue", engle_granger, "cointegration_pvalue")
    _attach_scalar(frame, "math_v2_half_life", ou, "half_life")
    _attach_scalar(frame, "math_v2_ou_phi", ou, "phi")
    _attach_scalar(frame, "math_v2_hurst", hurst, "hurst")
    _attach_scalar(frame, "math_v2_ecm_strength", ecm, "ecm_strength")
    _attach_series(frame, "math_v2_u1_given_u2", copula, "u1_given_u2")
    _attach_series(frame, "math_v2_u2_given_u1", copula, "u2_given_u1")
    _attach_series(
        frame,
        "math_v2_conditional_probability_distortion",
        copula,
        "conditional_probability_distortion",
    )
    frame["math_version"] = MATH_VERSION
    for row, (_, enriched) in zip(rows, frame.iterrows()):
        for key, value in enriched.items():
            if key.startswith("math_v2_") or key == "math_version":
                row[key] = _json_value(value)

    results = [engle_granger, ou, hurst, ecm, copula]
    return {
        "math_version": MATH_VERSION,
        "status": "valid"
        if all(result.validity_status == "valid" for result in results)
        else "partial",
        "reason": ";".join(
            f"{result.method_id}:{result.validity_reason}"
            for result in results
            if result.validity_status != "valid"
        )
        or "all_estimators_valid",
        "audit": [_audit_row(result, frame) for result in results],
        "zscore_convention_status": "pending_wizard_parity",
    }


def _positive_log_prices(price_x: pd.Series, price_y: pd.Series) -> pd.DataFrame:
    data = pd.concat(
        [pd.to_numeric(price_x, errors="coerce"), pd.to_numeric(price_y, errors="coerce")],
        axis=1,
        keys=["x", "y"],
    ).dropna()
    data = data[(data["x"] > 0.0) & (data["y"] > 0.0)]
    return pd.DataFrame({"log_x": np.log(data["x"]), "log_y": np.log(data["y"])}, index=data.index)


def _invalid(method_id: str, reason: str, rows: int) -> EstimatorResult:
    return EstimatorResult(method_id, MATH_VERSION, "invalid", reason, {}, rows)


def _attach_scalar(frame: pd.DataFrame, column: str, result: EstimatorResult, key: str) -> None:
    frame[column] = result.values.get(key, np.nan) if result.validity_status == "valid" else np.nan
    frame[f"{column}_validity"] = result.validity_status


def _attach_series(frame: pd.DataFrame, column: str, result: EstimatorResult, key: str) -> None:
    value = result.values.get(key)
    frame[column] = value.reindex(frame.index) if isinstance(value, pd.Series) else np.nan
    frame[f"{column}_validity"] = result.validity_status


def _audit_row(result: EstimatorResult, frame: pd.DataFrame) -> dict[str, Any]:
    timestamps = frame["timestamp"] if "timestamp" in frame.columns else pd.Series(dtype=object)
    return {
        "metric": result.method_id,
        "method_id": result.method_id,
        "estimator_version": result.estimator_version,
        "source_authority": "local_validated_estimator",
        "input_start": str(timestamps.iloc[0]) if not timestamps.empty else "",
        "input_end": str(timestamps.iloc[-1]) if not timestamps.empty else "",
        "lookback_rows": result.lookback_rows,
        "computed_at": datetime.now(UTC).isoformat(),
        "point_in_time": False,
        "validity_status": result.validity_status,
        "validity_reason": result.validity_reason
        + ";batch_fit_requires_walk_forward_for_signal_use",
    }


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value
