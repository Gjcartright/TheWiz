from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from quant_platform.wizard_mode_replay import build_local_mode_signal, mode_requirements


def _history() -> pd.DataFrame:
    index = pd.date_range("2026-08-01", periods=12, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "price_x": [100, 101, 102, 105, 103, 101, 99, 98, 97, 99, 101, 103],
            "price_y": [50, 50, 51, 51, 51, 50, 50, 49, 49, 50, 50, 51],
            "hedge_ratio": [1.8, 1.8, 1.9, 1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.9, 1.9, 1.9],
            "u1_given_u2": [0.50, 0.18, 0.12, 0.30, 0.55, 0.82, 0.90, 0.68, 0.48, 0.22, 0.12, 0.51],
            "u2_given_u1": [0.50, 0.42, 0.35, 0.45, 0.56, 0.70, 0.72, 0.61, 0.49, 0.35, 0.28, 0.51],
        },
        index=index,
    )


def _settings(**overrides: object) -> dict[str, object]:
    settings: dict[str, object] = {
        "capture_confirmed": True,
        "entry_long_operator": "<=",
        "entry_long_value": -1.0,
        "entry_long_position": "short_x_long_y",
        "entry_short_operator": ">=",
        "entry_short_value": 1.0,
        "entry_short_position": "long_x_short_y",
        "exit_long_operator": ">=",
        "exit_long_value": 0.0,
        "exit_short_operator": "<=",
        "exit_short_value": 0.0,
        "hedge_ratio": 1.9,
        "zscore_window": 3,
        "dynamic_hedge_ratio_method": "rolling_ols_log_prices",
        "dynamic_hedge_ratio_window": 3,
        "ou_mu": -3.0,
        "ou_sigma": 0.1,
        "copula_family": "gaussian",
        "copula_signal_type": "arbitrage",
        "copula_direction_view": "u1_given_u2",
        "copula_entry_lower": 0.20,
        "copula_entry_upper": 0.80,
        "copula_exit_lower": 0.45,
        "copula_exit_upper": 0.55,
    }
    settings.update(overrides)
    return settings


def _copula_settings(**overrides: object) -> dict[str, object]:
    return _settings(
        entry_long_position="long_x_short_y",
        entry_short_position="short_x_long_y",
        **overrides,
    )


def test_static_spread_and_static_zscorer_are_distinct_local_metrics():
    history = _history()
    static = build_local_mode_signal(history, _settings(), exact_mode="Static (Spread)")
    zscorer = build_local_mode_signal(history, _settings(), exact_mode="Static (ZScoreR)")

    assert static.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert zscorer.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert static.metric_name == "static_spread_expanding_zscore"
    assert zscorer.metric_name == "static_spread_rolling_zscore"
    assert not static.metric.equals(zscorer.metric)
    assert static.mode_fidelity_status == "local_formula_approximation"
    assert not static.acceptance_eligible


def test_dynamic_mode_requires_a_supported_explicit_hedge_method():
    result = build_local_mode_signal(
        _history(),
        _settings(dynamic_hedge_ratio_method="kalman"),
        exact_mode="Dyn (Spread)",
    )

    assert result.mode_replay_status == "BLOCKED_MODE_INPUTS"
    assert "unsupported_dynamic_hedge_ratio_method" in result.missing_inputs
    assert result.mode_fidelity_status == "local_formula_approximation"
    assert not result.acceptance_eligible


def test_dynamic_spread_uses_same_y_on_x_orientation_as_static_spread():
    history = _history()
    result = build_local_mode_signal(
        history,
        _settings(dynamic_hedge_ratio_method="history_captured_hedge_ratio"),
        exact_mode="Dyn (ZScoreR)",
    )
    expected_spread = np.log(history["price_y"]) - history["hedge_ratio"] * np.log(
        history["price_x"]
    )
    expected = expected_spread.sub(expected_spread.rolling(3).mean()).div(
        expected_spread.rolling(3).std(ddof=1)
    )

    pdt.assert_series_equal(result.metric, expected, check_names=False)
    assert any("y-on-x orientation" in note for note in result.computation_notes)


def test_static_zscorer_uses_sample_standard_deviation():
    history = _history()
    result = build_local_mode_signal(history, _settings(), exact_mode="Static (ZScoreR)")
    spread = np.log(history["price_y"]) - 1.9 * np.log(history["price_x"])
    expected = spread.sub(spread.rolling(3).mean()).div(spread.rolling(3).std(ddof=1))

    pdt.assert_series_equal(result.metric, expected, check_names=False)


def test_ou_mode_refuses_to_invent_missing_parameters():
    result = build_local_mode_signal(
        _history(), _settings(ou_mu="", ou_sigma=""), exact_mode="OU (Spread)"
    )

    assert result.mode_replay_status == "BLOCKED_MODE_INPUTS"
    assert {"ou_mu", "ou_sigma"}.issubset(set(result.missing_inputs))
    assert not result.acceptance_eligible


def test_ou_spread_metric_and_thresholds_use_the_same_log_spread_unit():
    history = _history()
    settings = _settings(ou_mu=-3.0, ou_sigma=0.1)
    result = build_local_mode_signal(history, settings, exact_mode="OU (Spread)")
    spread = np.log(history["price_y"]) - 1.9 * np.log(history["price_x"])

    pdt.assert_series_equal(result.metric, spread + 3.0)
    assert result.metric_name == "ou_centered_spread"


def test_copula_mode_uses_captured_direction_or_causal_rolling_fallback():
    rng = np.random.default_rng(7)
    returns_x = rng.normal(0.0005, 0.01, 140)
    returns_y = 0.55 * returns_x + rng.normal(0.0003, 0.008, 140)
    history = pd.DataFrame(
        {
            "price_x": 100.0 * np.cumprod(1.0 + returns_x),
            "price_y": 50.0 * np.cumprod(1.0 + returns_y),
        },
        index=pd.date_range("2026-01-01", periods=140, freq="h", tz="UTC"),
    )
    fallback = build_local_mode_signal(
        history,
        _copula_settings(
            entry_long_operator="",
            entry_long_value="",
            entry_short_operator="",
            entry_short_value="",
            exit_long_operator="",
            exit_long_value="",
            exit_short_operator="",
            exit_short_value="",
        ),
        exact_mode="Copula",
    )
    ready = build_local_mode_signal(_history(), _copula_settings(), exact_mode="Copula")

    assert fallback.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert fallback.metric.notna().sum() >= 60
    assert any("causal trailing Gaussian copula" in note for note in fallback.computation_notes)
    assert ready.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert ready.metric_name == "u1_given_u2"
    assert ready.trades
    assert all(
        trade["entry_rule"] in {"copula_lower_tail", "copula_upper_tail"} for trade in ready.trades
    )
    assert not ready.acceptance_eligible


def test_mode_requirements_include_explicit_direction_mapping_for_every_mode():
    requirements = mode_requirements("Static (Spread)")

    assert "entry_long_position" in requirements
    assert "entry_short_position" in requirements


def test_crypto_wizards_operator_enums_are_accepted_without_relabeling():
    result = build_local_mode_signal(
        _history(),
        _settings(
            entry_long_operator="Lte",
            entry_short_operator="Gte",
            exit_long_operator="Eq",
            exit_short_operator="Lt",
        ),
        exact_mode="Static (Spread)",
    )

    assert result.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert not any(value.startswith("invalid_") for value in result.missing_inputs)


def test_mode_replay_blocks_tail_positions_that_violate_the_economic_contract():
    result = build_local_mode_signal(
        _history(),
        _settings(
            entry_long_position="long_x_short_y",
            entry_short_position="short_x_long_y",
        ),
        exact_mode="Dyn (Spread)",
    )

    assert result.mode_replay_status == "BLOCKED_MODE_INPUTS"
    assert "lower_tail_position_contract_mismatch" in result.missing_inputs
    assert "upper_tail_position_contract_mismatch" in result.missing_inputs


def test_direction_contract_follows_operator_when_vendor_entry_fields_are_swapped():
    result = build_local_mode_signal(
        _history(),
        _settings(
            entry_long_operator=">=",
            entry_long_value=1.0,
            entry_long_position="long_x_short_y",
            entry_short_operator="<=",
            entry_short_value=-1.0,
            entry_short_position="short_x_long_y",
        ),
        exact_mode="Static (Spread)",
    )

    assert result.mode_replay_status == "READY_FOR_RESEARCH_REPLAY"
    assert not any("position_contract_mismatch" in value for value in result.missing_inputs)
