from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.statistical_validation import (
    benjamini_hochberg,
    circular_block_bootstrap_mean,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
)


def test_probabilistic_sharpe_ranks_positive_series_above_zero_benchmark():
    rng = np.random.default_rng(17)
    returns = rng.normal(0.003, 0.01, 500)

    result = probabilistic_sharpe_ratio(returns, periods_per_year=365.0)

    assert result.status == "VALID"
    assert result.probability > 0.95
    assert result.pearson_kurtosis > 2.0


def test_deflated_sharpe_uses_trial_family_expected_maximum_as_benchmark():
    rng = np.random.default_rng(23)
    returns = rng.normal(0.0015, 0.01, 400)
    trials = pd.Series([0.3, 0.7, 0.9, 1.1, 1.4])

    result = deflated_sharpe_ratio(
        returns,
        periods_per_year=365.0,
        trial_annualized_sharpes=trials,
    )

    assert result.status == "VALID"
    assert result.benchmark_annualized_sharpe == expected_maximum_sharpe(trials)
    assert 0.0 <= result.probability <= 1.0


def test_block_bootstrap_fails_closed_on_tiny_samples():
    result = circular_block_bootstrap_mean([0.01, -0.01, 0.02, 0.0, 0.01])

    assert result.status == "BLOCKED"
    assert np.isnan(result.pvalue)
    assert "observations<" in result.blocker


def test_block_bootstrap_detects_persistent_positive_mean():
    rng = np.random.default_rng(31)
    innovations = rng.normal(0.0, 0.002, 600)
    returns = np.empty(600)
    returns[0] = 0.001 + innovations[0]
    for index in range(1, len(returns)):
        returns[index] = 0.001 + 0.45 * (returns[index - 1] - 0.001) + innovations[index]

    result = circular_block_bootstrap_mean(returns, replications=499)

    assert result.status == "VALID"
    assert result.pvalue < 0.05
    assert result.lower_confidence_bound > 0.0


def test_benjamini_hochberg_is_monotonic_in_sorted_pvalues():
    pvalues = pd.Series([0.04, 0.001, 0.03, np.nan, 0.2])
    adjusted = benjamini_hochberg(pvalues)

    valid = pd.DataFrame({"p": pvalues, "q": adjusted}).dropna().sort_values("p")
    assert valid["q"].is_monotonic_increasing
    assert adjusted.isna().sum() == 1
