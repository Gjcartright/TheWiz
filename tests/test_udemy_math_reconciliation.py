from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from quant_platform.active_pipeline import ROOT
from quant_platform.backtest import CostModel, backtest_two_leg_spread
from quant_platform.performance_math import calculate_annualized_sharpe
from quant_platform.statistics.math_v2 import (
    fit_gaussian_copula,
    fit_ou,
    rolling_zscore_variants,
)
from quant_platform.udemy_math_reconciliation import (
    CONTRACTS,
    build_udemy_math_reconciliation,
)


def test_every_udemy_math_contract_has_local_source_and_implementation(tmp_path):
    fixture_root = tmp_path / "fixture_root"
    manifest = fixture_root / "reports" / "research" / "udemy_transcript_vault_manifest.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "lecture_id": lecture_id,
                "vault_status": "captured",
                "transcript_path": "restricted_fixture_not_exported",
                "transcript_sha256": "a" * 64,
            }
            for lecture_id in sorted({contract.lecture_id for contract in CONTRACTS})
        ]
    ).to_csv(manifest, index=False)

    result = build_udemy_math_reconciliation(
        root=fixture_root,
        output_root=tmp_path,
    )

    assert result.summary["all_referenced_transcripts_captured"] is True
    for contract in CONTRACTS:
        for implementation in contract.implementation_path.split(";"):
            assert (ROOT / implementation).is_file(), implementation


def test_udemy_rolling_zscore_contract_matches_pandas_sample_std():
    values = pd.Series([2.0, 4.0, 5.0, 9.0, 12.0, 11.0])
    actual = rolling_zscore_variants(values, window=4)["zscore_ddof1"]
    expected = (values - values.rolling(4).mean()) / values.rolling(4).std(ddof=1)

    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_math_v2_half_life_is_exact_ar1_mapping_not_course_approximation():
    phi = 0.4
    spread = [1.0]
    for _ in range(1, 300):
        spread.append(phi * spread[-1] + 0.01 * math.sin(len(spread)))
    result = fit_ou(pd.Series(spread), min_rows=60)

    assert result.validity_status == "valid"
    fitted_phi = result.values["phi"]
    exact = math.log(2.0) / -math.log(fitted_phi)
    course_approximation = -math.log(2.0) / (fitted_phi - 1.0)
    assert result.values["half_life"] == pytest.approx(exact)
    assert abs(exact - course_approximation) > 0.05


def test_udemy_gaussian_h_function_matches_math_v2_conditionals():
    rng = np.random.default_rng(41)
    values = rng.multivariate_normal([0.0, 0.0], [[1.0, 0.65], [0.65, 1.0]], size=240)
    x = pd.Series(values[:, 0])
    y = pd.Series(values[:, 1])
    result = fit_gaussian_copula(x, y)

    assert result.validity_status == "valid"
    index = 137
    u1 = x.rank(method="average").iloc[index] / (len(x) + 1.0)
    u2 = y.rank(method="average").iloc[index] / (len(y) + 1.0)
    rho = result.values["rho"]
    expected = norm.cdf((norm.ppf(u1) - rho * norm.ppf(u2)) / math.sqrt(1.0 - rho**2))
    assert result.values["u1_given_u2"].iloc[index] == pytest.approx(expected)


def test_crypto_sharpe_uses_course_sample_std_with_declared_crypto_frequency():
    returns = pd.Series([0.01, -0.02, 0.03, 0.005])
    result = calculate_annualized_sharpe(returns, interval="1d")

    expected = math.sqrt(365.0) * returns.mean() / returns.std(ddof=1)
    assert result.value == pytest.approx(expected)
    assert result.periods_per_year == 365


def test_close_signal_is_applied_to_next_bar_two_leg_returns():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"),
            "price_x": [100.0, 200.0, 200.0, 200.0],
            "price_y": [100.0, 100.0, 110.0, 110.0],
            "hedge_ratio": [1.0] * 4,
        }
    )
    signal = pd.Series([0.0, 1.0, 0.0, 0.0])
    result = backtest_two_leg_spread(
        frame,
        signal,
        CostModel(
            taker_fee_bps=0.0,
            slippage_bps=0.0,
            funding_bps_per_day=0.0,
            execution_risk_bps=0.0,
        ),
        interval="1d",
    )

    # The large X move occurs before the signal can be held; only the next Y move is earned.
    assert result.gross_return == pytest.approx(0.05)
