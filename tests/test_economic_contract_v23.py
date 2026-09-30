from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quant_platform.economic_contract import (
    CANONICAL_WIZARD_MODES,
    ECONOMIC_CONTRACT_VERSION,
    EXACT_MODES,
    SCANNER_OVERLAYS,
    ExactMode,
    TradeAction,
    action_for_signal,
    copula_distortion_signal,
    mode_contract,
    normalized_two_leg_weight_magnitudes,
    normalized_two_leg_weights,
    rolling_y_on_x_beta,
    signal_for_action,
    swapped_action,
    tail_actions,
    y_on_x_beta,
    y_on_x_log_spread,
)


def test_registry_has_exactly_seven_modes_and_keeps_ou_optimal_as_overlay():
    assert ECONOMIC_CONTRACT_VERSION == "wizard-seven-mode-economic-contract.v2"
    assert len(EXACT_MODES) == 7
    assert len(CANONICAL_WIZARD_MODES) == 7
    assert "OU (Optimal)" not in CANONICAL_WIZARD_MODES
    assert SCANNER_OVERLAYS == ("OU (Optimal)",)
    assert {mode_contract(mode).mode for mode in EXACT_MODES} == set(EXACT_MODES)


@pytest.mark.parametrize("mode", [mode for mode in EXACT_MODES if mode != ExactMode.COPULA])
def test_every_spread_mode_uses_same_tail_direction_and_sizing(mode: ExactMode):
    lower, upper = tail_actions(mode)
    assert lower == TradeAction.SHORT_X_LONG_Y
    assert upper == TradeAction.LONG_X_SHORT_Y
    assert signal_for_action(lower) == 1.0
    assert signal_for_action(upper) == -1.0

    signal = pd.Series([signal_for_action(lower), signal_for_action(upper)])
    weight_x, weight_y = normalized_two_leg_weights(signal, 2.0)

    assert weight_x.iloc[0] == pytest.approx(-2.0 / 3.0)
    assert weight_y.iloc[0] == pytest.approx(1.0 / 3.0)
    assert weight_x.iloc[1] == pytest.approx(2.0 / 3.0)
    assert weight_y.iloc[1] == pytest.approx(-1.0 / 3.0)
    assert (weight_x.abs() + weight_y.abs()).eq(1.0).all()


def test_spread_tail_actions_profit_when_the_dislocation_converges():
    lower, upper = tail_actions(ExactMode.DYN_SPREAD)
    lower_signal = pd.Series([signal_for_action(lower)])
    upper_signal = pd.Series([signal_for_action(upper)])
    lower_x, lower_y = normalized_two_leg_weights(lower_signal, 1.0)
    upper_x, upper_y = normalized_two_leg_weights(upper_signal, 1.0)

    lower_convergence_pnl = lower_x.iloc[0] * 0.0 + lower_y.iloc[0] * 0.02
    upper_convergence_pnl = upper_x.iloc[0] * 0.0 + upper_y.iloc[0] * -0.02

    assert lower_convergence_pnl > 0.0
    assert upper_convergence_pnl > 0.0


def test_copula_tail_actions_are_conditioned_asset_specific():
    assert tail_actions(ExactMode.COPULA, copula_direction_view="u1_given_u2") == (
        TradeAction.LONG_X_SHORT_Y,
        TradeAction.SHORT_X_LONG_Y,
    )
    assert tail_actions(ExactMode.COPULA, copula_direction_view="u2_given_u1") == (
        TradeAction.SHORT_X_LONG_Y,
        TradeAction.LONG_X_SHORT_Y,
    )

    distortion = pd.Series([0.30, -0.30, 0.0])
    assert copula_distortion_signal(distortion, 0.20).tolist() == [1.0, -1.0, 0.0]


def test_signal_action_and_asset_swap_are_involutions():
    for action in (TradeAction.LONG_X_SHORT_Y, TradeAction.SHORT_X_LONG_Y):
        assert action_for_signal(signal_for_action(action)) == action
        assert swapped_action(swapped_action(action)) == action


def test_dynamic_beta_and_spread_share_y_on_x_orientation():
    log_x = pd.Series(np.linspace(4.0, 5.0, 80))
    log_y = 0.7 + 1.6 * log_x
    beta = rolling_y_on_x_beta(log_x, log_y, window=30)
    finite = beta.dropna()

    assert not finite.empty
    assert finite.median() == pytest.approx(1.6, abs=1e-10)

    price_x = np.exp(log_x)
    price_y = np.exp(log_y)
    spread = y_on_x_log_spread(price_x, price_y, beta)
    expected = log_y - beta * log_x
    pdt.assert_series_equal(spread, expected)

    assert y_on_x_beta(log_x, log_y) == pytest.approx(1.6, abs=1e-10)
    assert normalized_two_leg_weight_magnitudes(2.0) == pytest.approx((2 / 3, 1 / 3))


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf")])
def test_canonical_two_leg_weights_reject_nonpositive_or_nonfinite_beta(beta: float):
    with pytest.raises(ValueError, match="finite and positive"):
        normalized_two_leg_weights(pd.Series([1.0]), beta)
    with pytest.raises(ValueError, match="finite and positive"):
        normalized_two_leg_weight_magnitudes(beta)
