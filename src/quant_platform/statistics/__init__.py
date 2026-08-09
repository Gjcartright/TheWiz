"""Validated local statistical estimators."""

from quant_platform.statistics.math_v2 import (
    attach_math_v2_statistics,
    estimate_hurst_dfa,
    fit_engle_granger,
    fit_gaussian_copula,
    fit_ou,
    rolling_gaussian_copula_conditionals,
    rolling_zscore_variants,
)

__all__ = [
    "attach_math_v2_statistics",
    "estimate_hurst_dfa",
    "fit_engle_granger",
    "fit_gaussian_copula",
    "fit_ou",
    "rolling_gaussian_copula_conditionals",
    "rolling_zscore_variants",
]
