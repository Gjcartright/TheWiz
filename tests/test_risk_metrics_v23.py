from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import numpy as np

from quant_platform.risk_metrics import (
    RESEARCH_ONLY_AUTHORITY,
    calculate_calmar_ratio,
    calculate_expectancy_r,
    calculate_historical_var_cvar,
    calculate_payoff_ratio,
    calculate_profit_factor,
    calculate_realized_r,
    calculate_sortino_ratio,
)


def test_profit_factor_uses_net_trade_pnl_and_preserves_valid_zero() -> None:
    positive = calculate_profit_factor([100.0, -50.0, 0.0])
    zero = calculate_profit_factor([-20.0, -30.0, 0.0])

    assert positive.status == "valid"
    assert positive.value == pytest.approx(2.0)
    assert positive.unit == "ratio"
    assert zero.status == "valid"
    assert zero.value == 0.0


def test_profit_factor_without_a_loss_is_unavailable_not_infinite() -> None:
    result = calculate_profit_factor([10.0, 0.0])

    assert result.status == "unavailable"
    assert result.value is None
    assert result.reason == "no_loss_observations"


def test_payoff_ratio_requires_observed_wins_and_losses() -> None:
    valid = calculate_payoff_ratio([100.0, 50.0, -25.0, -75.0, 0.0])
    no_wins = calculate_payoff_ratio([-10.0, -20.0])

    assert valid.status == "valid"
    assert valid.value == pytest.approx(1.5)
    assert no_wins.status == "unavailable"
    assert no_wins.value is None
    assert no_wins.reason == "no_win_observations"


def test_realized_r_uses_same_currency_net_pnl_and_positive_planned_risk() -> None:
    win = calculate_realized_r(100.0, 50.0)
    loss = calculate_realized_r(-25.0, 50.0)
    breakeven = calculate_realized_r(0.0, 50.0)
    invalid_risk = calculate_realized_r(10.0, 0.0)

    assert win.value == pytest.approx(2.0)
    assert loss.value == pytest.approx(-0.5)
    assert breakeven.status == "valid"
    assert breakeven.value == 0.0
    assert invalid_risk.status == "unavailable"
    assert invalid_risk.value is None


def test_expectancy_r_is_mean_completed_trade_r_and_zero_can_be_valid() -> None:
    result = calculate_expectancy_r([2.0, -1.0, 0.0])
    zero = calculate_expectancy_r([1.0, -1.0], min_observations=2)

    assert result.status == "valid"
    assert result.value == pytest.approx(1.0 / 3.0)
    assert zero.status == "valid"
    assert zero.value == 0.0
    assert result.unit == "R_per_trade"


def test_scalar_metrics_fail_closed_on_missing_nonfinite_or_weak_samples() -> None:
    nonfinite = calculate_expectancy_r([1.0, float("nan")])
    weak = calculate_profit_factor([1.0, -1.0], min_observations=3)
    empty = calculate_payoff_ratio([])

    assert nonfinite.status == "unavailable"
    assert nonfinite.reason == "input_contains_nonfinite_observation"
    assert weak.status == "unavailable"
    assert weak.observations == 2
    assert weak.required_observations == 3
    assert empty.status == "unavailable"


def test_boolean_inputs_never_become_numeric_risk_observations() -> None:
    assert calculate_profit_factor([True, -1.0]).status == "unavailable"
    assert calculate_profit_factor([np.bool_(True), -1.0]).status == "unavailable"
    assert calculate_realized_r(np.bool_(True), 1.0).status == "unavailable"
    assert calculate_realized_r(1.0, False).status == "unavailable"
    assert calculate_historical_var_cvar([False] * 100).status == "unavailable"
    assert calculate_expectancy_r([1.0, -1.0], min_observations=np.bool_(True)).status == "unavailable"


def test_sortino_uses_population_target_downside_deviation_and_explicit_scale() -> None:
    result = calculate_sortino_ratio([0.02, -0.01, 0.0], periods_per_year=1.0)

    assert result.status == "valid"
    assert result.value == pytest.approx(1.0 / 3.0**0.5)
    assert "target_downside_deviation_ddof0" in result.method
    assert result.unit == "annualized_ratio"


def test_sortino_zero_downside_or_invalid_annualization_is_unavailable() -> None:
    no_downside = calculate_sortino_ratio([0.01, 0.02], periods_per_year=365.0)
    bad_scale = calculate_sortino_ratio([0.01, -0.01], periods_per_year=0.0)

    assert no_downside.status == "unavailable"
    assert no_downside.value is None
    assert no_downside.reason == "zero_downside_deviation"
    assert bad_scale.status == "unavailable"
    assert bad_scale.reason == "invalid_periods_per_year"


def test_calmar_compounds_returns_and_preserves_a_valid_zero_ratio() -> None:
    negative = calculate_calmar_ratio([0.10, -0.10], periods_per_year=2.0)
    zero = calculate_calmar_ratio([-0.50, 1.0], periods_per_year=2.0)

    assert negative.status == "valid"
    assert negative.value == pytest.approx(-0.10)
    assert zero.status == "valid"
    assert zero.value == pytest.approx(0.0, abs=1e-15)


def test_calmar_rejects_noncompoundable_returns_and_zero_drawdown() -> None:
    ruined = calculate_calmar_ratio([0.10, -1.0], periods_per_year=2.0)
    monotone = calculate_calmar_ratio([0.01, 0.02], periods_per_year=2.0)

    assert ruined.status == "unavailable"
    assert ruined.reason == "simple_return_at_or_below_minus_one"
    assert monotone.status == "unavailable"
    assert monotone.value is None
    assert monotone.reason == "zero_max_drawdown"


def test_historical_var_cvar_has_explicit_loss_sign_and_inclusive_tail() -> None:
    result = calculate_historical_var_cvar(
        [-0.10, -0.05, 0.0, 0.02, 0.04],
        confidence=0.80,
        min_observations=5,
        min_tail_observations=1,
    )

    assert result.status == "valid"
    assert result.confidence == 0.80
    assert result.value_at_risk == pytest.approx(0.06)
    assert result.conditional_value_at_risk == pytest.approx(0.10)
    assert result.tail_observations == 1
    assert result.required_observations == 5
    assert "loss=-decimal periodic return" in result.convention


def test_historical_tail_sufficiency_accounts_for_confidence_and_tail_count() -> None:
    result = calculate_historical_var_cvar(
        [0.0] * 39,
        confidence=0.95,
        min_observations=1,
        min_tail_observations=2,
    )

    assert result.status == "unavailable"
    assert result.value_at_risk is None
    assert result.conditional_value_at_risk is None
    assert result.observations == 39
    assert result.required_observations == 40
    assert result.reason == "insufficient_observations_for_requested_tail"


def test_historical_var_cvar_can_report_real_numeric_zero() -> None:
    result = calculate_historical_var_cvar([0.0] * 100)

    assert result.status == "valid"
    assert result.value_at_risk == 0.0
    assert result.conditional_value_at_risk == 0.0
    assert result.tail_observations == 100


def test_historical_var_cvar_fails_closed_on_bad_inputs_and_method() -> None:
    nonfinite = calculate_historical_var_cvar([0.0] * 99 + [float("inf")])
    confidence = calculate_historical_var_cvar([0.0] * 100, confidence=1.0)
    method = calculate_historical_var_cvar([0.0] * 100, quantile_method="not-a-method")

    assert nonfinite.status == "unavailable"
    assert nonfinite.reason == "input_contains_nonfinite_observation"
    assert confidence.status == "unavailable"
    assert confidence.confidence == 1.0
    assert method.status == "unavailable"
    assert method.reason == "unsupported_quantile_method"


def test_results_are_frozen_and_explicitly_research_only() -> None:
    result = calculate_profit_factor([1.0, -1.0])
    tail = calculate_historical_var_cvar([0.0] * 100)

    assert result.authority == RESEARCH_ONLY_AUTHORITY
    assert tail.authority == RESEARCH_ONLY_AUTHORITY
    assert "wizard" in result.authority
    assert "live" in tail.authority
    with pytest.raises(FrozenInstanceError):
        result.value = 99.0  # type: ignore[misc]
