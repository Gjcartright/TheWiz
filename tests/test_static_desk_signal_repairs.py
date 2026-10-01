"""RT01 regression and independent signal-to-exposure boundary controls."""
import numpy as np
import pandas as pd
import pytest

from quant_platform.economic_contract import normalized_two_leg_weights


@pytest.mark.parametrize("signal", ["bad", np.nan, pd.NA, None, np.inf, -np.inf,
                                  2., -2., True, False, 1+0j, 1+2j])
def test_invalid_signal_is_not_flat_or_unbounded(signal):
    with pytest.raises(ValueError, match="signal"):
        normalized_two_leg_weights(pd.Series([signal]), 1.)


@pytest.mark.parametrize("beta", [.01, 1., 2., 1e308])
def test_bounded_fractional_and_discrete_signals_preserve_orientation(beta):
    signals = pd.Series([-1., -.5, 0., .25, 1.], index=list("abcde"))
    wx, wy = normalized_two_leg_weights(signals, beta)
    np.testing.assert_allclose(abs(wx) + abs(wy), abs(signals))
    np.testing.assert_allclose(wx, -signals * (beta / (1 + beta)))
    np.testing.assert_allclose(wy, signals / (1 + beta))
    assert wx.index.equals(signals.index) and wy.index.equals(signals.index)
    assert wx.name == "weight_x" and wy.name == "weight_y"


@pytest.mark.parametrize("beta", [True, False, 1+0j, 1+2j, "bad", None])
def test_invalid_beta_is_explicit(beta):
    with pytest.raises(ValueError, match="hedge ratio"):
        normalized_two_leg_weights(pd.Series([1.]), beta)


def test_index_mismatch_or_duplicates_cannot_silently_align():
    with pytest.raises(ValueError, match="index"):
        normalized_two_leg_weights(pd.Series([1., -1.], index=[1, 0]), pd.Series([1., 2.]))
    with pytest.raises(ValueError, match="index"):
        normalized_two_leg_weights(pd.Series([1., 1.], index=[0, 0]), 1.)


def test_numeric_text_and_nullable_numeric_compatibility():
    wx, wy = normalized_two_leg_weights(pd.Series(["1", "0", "-1"]), "2")
    np.testing.assert_allclose(wx, [-2/3, 0, 2/3])
    wx, wy = normalized_two_leg_weights(pd.Series([1., 0.], dtype="Float64"), 2.)
    assert np.isfinite(wx).all() and np.isfinite(wy).all()


def test_empty_series_means_no_target_rows():
    wx, wy = normalized_two_leg_weights(pd.Series([], dtype=float), 1.)
    assert wx.empty and wy.empty


@pytest.mark.parametrize("signal", [True, False, 1+0j, 1+2j, 2., -2., np.nan])
def test_two_leg_consumer_does_not_erase_signal_type_or_leverage(signal):
    from quant_platform.backtest import backtest_two_leg_spread_with_ledger

    frame = pd.DataFrame({"price_x": [100., 100.], "price_y": [50., 50.],
                          "hedge_ratio": [1., 1.]})
    with pytest.raises(ValueError, match="signal"):
        backtest_two_leg_spread_with_ledger(frame, pd.Series([signal, signal]))


@pytest.mark.parametrize("beta", [True, False, 1+0j, 1+2j, np.nan, 0., -1.])
def test_two_leg_consumer_preserves_beta_until_canonical_validation(beta):
    from quant_platform.backtest import backtest_two_leg_spread_with_ledger

    frame = pd.DataFrame({"price_x": [100., 100.], "price_y": [50., 50.],
                          "hedge_ratio": [beta, beta]})
    with pytest.raises(ValueError, match="hedge ratio"):
        backtest_two_leg_spread_with_ledger(frame, pd.Series([1., 1.]))
