"""Shared, fail-closed statistical selection controls for strategy research.

The Probabilistic and Deflated Sharpe calculations follow Bailey and Lopez de
Prado (2014). They operate on per-period Sharpe values; annualized values are
used only for reporting and are converted back to the same period before the
probability calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329
MIN_INFERENCE_OBSERVATIONS = 30
DEFAULT_BOOTSTRAP_REPLICATIONS = 999


@dataclass(frozen=True)
class SharpeInference:
    observations: int
    period_sharpe: float
    annualized_sharpe: float
    skewness: float
    pearson_kurtosis: float
    probability: float
    benchmark_period_sharpe: float
    benchmark_annualized_sharpe: float
    status: str
    blocker: str


@dataclass(frozen=True)
class BlockBootstrapInference:
    observations: int
    block_length: int
    replications: int
    observed_mean: float
    pvalue: float
    lower_confidence_bound: float
    upper_confidence_bound: float
    status: str
    blocker: str


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return monotonic Benjamini-Hochberg q-values without filling missing data."""

    values = pd.to_numeric(pvalues, errors="coerce")
    result = pd.Series(float("nan"), index=values.index, dtype="float64")
    valid = values.dropna().clip(0.0, 1.0).sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = valid * count / np.arange(1, count + 1)
    adjusted = pd.Series(
        np.minimum.accumulate(adjusted.iloc[::-1])[::-1],
        index=valid.index,
    ).clip(upper=1.0)
    result.loc[adjusted.index] = adjusted
    return result


def circular_block_bootstrap_mean(
    values: Iterable[float] | pd.Series,
    *,
    replications: int = DEFAULT_BOOTSTRAP_REPLICATIONS,
    block_length: int | None = None,
    seed: int = 20260815,
) -> BlockBootstrapInference:
    """Test positive mean and form a percentile interval under serial dependence."""

    clean = _finite_array(values)
    observations = len(clean)
    if observations < MIN_INFERENCE_OBSERVATIONS:
        return BlockBootstrapInference(
            observations,
            0,
            replications,
            float(np.mean(clean)) if observations else float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            "BLOCKED",
            f"observations<{MIN_INFERENCE_OBSERVATIONS}",
        )
    if replications < 199:
        raise ValueError("block bootstrap requires at least 199 replications")
    selected_block = block_length or max(2, int(round(observations ** (1.0 / 3.0))))
    selected_block = min(selected_block, observations)
    rng = np.random.default_rng(seed)
    observed_mean = float(clean.mean())
    centered = clean - observed_mean
    null_means = np.empty(replications, dtype=float)
    sample_means = np.empty(replications, dtype=float)
    blocks_needed = math.ceil(observations / selected_block)
    offsets = np.arange(selected_block)
    for iteration in range(replications):
        starts = rng.integers(0, observations, size=blocks_needed)
        indexes = ((starts[:, None] + offsets[None, :]) % observations).reshape(-1)[:observations]
        null_means[iteration] = float(centered[indexes].mean())
        sample_means[iteration] = float(clean[indexes].mean())
    pvalue = float((1 + np.count_nonzero(null_means >= observed_mean)) / (replications + 1))
    lower, upper = np.quantile(sample_means, [0.025, 0.975])
    return BlockBootstrapInference(
        observations,
        selected_block,
        replications,
        observed_mean,
        pvalue,
        float(lower),
        float(upper),
        "VALID",
        "",
    )


def probabilistic_sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: float,
    benchmark_annualized_sharpe: float = 0.0,
) -> SharpeInference:
    """Estimate P(true Sharpe > benchmark) with skew and kurtosis adjustment."""

    clean = _finite_array(returns)
    observations = len(clean)
    if observations < MIN_INFERENCE_OBSERVATIONS:
        return _blocked_sharpe(observations, "insufficient_return_observations")
    if not math.isfinite(periods_per_year) or periods_per_year <= 0.0:
        return _blocked_sharpe(observations, "invalid_periods_per_year")
    standard_deviation = float(np.std(clean, ddof=1))
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return _blocked_sharpe(observations, "zero_or_invalid_return_volatility")
    period_sharpe = float(np.mean(clean) / standard_deviation)
    annualized_sharpe = float(period_sharpe * math.sqrt(periods_per_year))
    sample_skew = float(skew(clean, bias=False))
    sample_kurtosis = float(kurtosis(clean, fisher=False, bias=False))
    benchmark_period = float(benchmark_annualized_sharpe / math.sqrt(periods_per_year))
    probability = sharpe_exceedance_probability(
        observed_period_sharpe=period_sharpe,
        benchmark_period_sharpe=benchmark_period,
        observations=observations,
        skewness=sample_skew,
        pearson_kurtosis=sample_kurtosis,
    )
    if not math.isfinite(probability):
        return SharpeInference(
            observations,
            period_sharpe,
            annualized_sharpe,
            sample_skew,
            sample_kurtosis,
            float("nan"),
            benchmark_period,
            benchmark_annualized_sharpe,
            "BLOCKED",
            "invalid_probabilistic_sharpe_denominator",
        )
    return SharpeInference(
        observations,
        period_sharpe,
        annualized_sharpe,
        sample_skew,
        sample_kurtosis,
        probability,
        benchmark_period,
        benchmark_annualized_sharpe,
        "VALID",
        "",
    )


def expected_maximum_sharpe(
    trial_annualized_sharpes: Iterable[float] | pd.Series,
    *,
    independent_trials: int | None = None,
) -> float:
    """Estimate the expected maximum Sharpe across a declared trial family."""

    trials = _finite_array(trial_annualized_sharpes)
    if not len(trials):
        return float("nan")
    count = int(independent_trials if independent_trials is not None else len(trials))
    if count <= 1:
        return 0.0
    trial_std = float(np.std(trials, ddof=1)) if len(trials) >= 2 else 0.0
    if not math.isfinite(trial_std):
        return float("nan")
    trial_mean = float(np.mean(trials))
    if trial_std <= 0.0:
        return trial_mean
    maximum_z = (1.0 - EULER_MASCHERONI) * norm.ppf(
        1.0 - 1.0 / count
    ) + EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (count * math.e))
    return float(trial_mean + trial_std * maximum_z)


def deflated_sharpe_ratio(
    returns: Iterable[float] | pd.Series,
    *,
    periods_per_year: float,
    trial_annualized_sharpes: Iterable[float] | pd.Series,
    independent_trials: int | None = None,
) -> SharpeInference:
    """Compute DSR as PSR against the expected best Sharpe from all trials."""

    benchmark = expected_maximum_sharpe(
        trial_annualized_sharpes,
        independent_trials=independent_trials,
    )
    if not math.isfinite(benchmark):
        return _blocked_sharpe(len(_finite_array(returns)), "invalid_trial_family_sharpes")
    return probabilistic_sharpe_ratio(
        returns,
        periods_per_year=periods_per_year,
        benchmark_annualized_sharpe=benchmark,
    )


def sharpe_probability_from_moments(
    *,
    observed_period_sharpe: float,
    benchmark_annualized_sharpe: float,
    periods_per_year: float,
    observations: int,
    skewness: float,
    pearson_kurtosis: float,
) -> float:
    if periods_per_year <= 0.0:
        return float("nan")
    benchmark_period = benchmark_annualized_sharpe / math.sqrt(periods_per_year)
    return sharpe_exceedance_probability(
        observed_period_sharpe=observed_period_sharpe,
        benchmark_period_sharpe=benchmark_period,
        observations=observations,
        skewness=skewness,
        pearson_kurtosis=pearson_kurtosis,
    )


def sharpe_exceedance_probability(
    *,
    observed_period_sharpe: float,
    benchmark_period_sharpe: float,
    observations: int,
    skewness: float,
    pearson_kurtosis: float,
) -> float:
    if observations < 2 or not all(
        math.isfinite(value)
        for value in (
            observed_period_sharpe,
            benchmark_period_sharpe,
            skewness,
            pearson_kurtosis,
        )
    ):
        return float("nan")
    variance_term = (
        1.0
        - skewness * observed_period_sharpe
        + ((pearson_kurtosis - 1.0) / 4.0) * observed_period_sharpe**2
    )
    if not math.isfinite(variance_term) or variance_term <= 0.0:
        return float("nan")
    statistic = (
        (observed_period_sharpe - benchmark_period_sharpe)
        * math.sqrt(observations - 1)
        / math.sqrt(variance_term)
    )
    return float(norm.cdf(statistic))


def _finite_array(values: Iterable[float] | pd.Series) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    array = series.to_numpy(dtype=float)
    return array[np.isfinite(array)]


def _blocked_sharpe(observations: int, blocker: str) -> SharpeInference:
    return SharpeInference(
        observations,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        "BLOCKED",
        blocker,
    )
