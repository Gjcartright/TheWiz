"""Canonical economic semantics for the seven Crypto Wizards research modes.

The contract separates vendor formula parity from local economic meaning. Every
local producer must agree on asset order, hedge-ratio orientation, signal sign,
tail direction, and two-leg sizing before its output can enter acceptance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

ECONOMIC_CONTRACT_VERSION = "wizard-seven-mode-economic-contract.v2"
SCANNER_OVERLAYS = ("OU (Optimal)",)


class ExactMode(StrEnum):
    STATIC_SPREAD = "Static Spread"
    STATIC_ZSCORER = "Static ZScoreR"
    DYN_SPREAD = "Dyn Spread"
    DYN_ZSCORER = "Dyn ZScoreR"
    OU_SPREAD = "OU Spread"
    OU_ZSCORER = "OU ZScoreR"
    COPULA = "Copula"


class TradeAction(StrEnum):
    LONG_X_SHORT_Y = "long_x_short_y"
    SHORT_X_LONG_Y = "short_x_long_y"
    FLAT = "flat"
    ABSTAIN = "abstain"


class MetricUnits(StrEnum):
    DIMENSIONLESS = "dimensionless"
    LOG_SPREAD = "log_spread"
    CONDITIONAL_PROBABILITY = "conditional_probability"


@dataclass(frozen=True)
class ModeEconomicContract:
    mode: ExactMode
    replay_label: str
    metric_name: str
    metric_units: MetricUnits
    tail_rule: str
    spread_orientation: str = "log_y_minus_beta_y_on_x_times_log_x"
    hedge_ratio_orientation: str = "beta_y_on_x"
    signal_convention: str = "plus_one_short_x_long_y_minus_one_long_x_short_y"
    sizing_convention: str = "gross_one_beta_normalized"


MODE_CONTRACTS: tuple[ModeEconomicContract, ...] = (
    ModeEconomicContract(
        ExactMode.STATIC_SPREAD,
        "Static (Spread)",
        "static_spread_expanding_zscore",
        MetricUnits.DIMENSIONLESS,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.STATIC_ZSCORER,
        "Static (ZScoreR)",
        "static_spread_rolling_zscore",
        MetricUnits.DIMENSIONLESS,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.DYN_SPREAD,
        "Dyn (Spread)",
        "dynamic_spread_expanding_zscore",
        MetricUnits.DIMENSIONLESS,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.DYN_ZSCORER,
        "Dyn (ZScoreR)",
        "dynamic_spread_rolling_zscore",
        MetricUnits.DIMENSIONLESS,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.OU_SPREAD,
        "OU (Spread)",
        "ou_centered_spread",
        MetricUnits.LOG_SPREAD,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.OU_ZSCORER,
        "OU (ZScoreR)",
        "ou_residual_rolling_zscore",
        MetricUnits.DIMENSIONLESS,
        "spread_reversion",
    ),
    ModeEconomicContract(
        ExactMode.COPULA,
        "Copula",
        "conditional_cdf",
        MetricUnits.CONDITIONAL_PROBABILITY,
        "copula_view_dependent",
        spread_orientation="copula_view_defines_conditioned_asset",
    ),
)

EXACT_MODES: tuple[ExactMode, ...] = tuple(contract.mode for contract in MODE_CONTRACTS)
CANONICAL_WIZARD_MODES: tuple[str, ...] = tuple(
    contract.replay_label for contract in MODE_CONTRACTS
)
MODE_CONTRACT_BY_MODE = {contract.mode: contract for contract in MODE_CONTRACTS}
MODE_CONTRACT_BY_REPLAY_LABEL = {contract.replay_label: contract for contract in MODE_CONTRACTS}


def normalize_exact_mode(value: str | ExactMode) -> ExactMode:
    if isinstance(value, ExactMode):
        return value
    key = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    aliases = {
        "static": ExactMode.STATIC_SPREAD,
        "staticspread": ExactMode.STATIC_SPREAD,
        "staticzscore": ExactMode.STATIC_ZSCORER,
        "staticzscorer": ExactMode.STATIC_ZSCORER,
        "dynspread": ExactMode.DYN_SPREAD,
        "dynamicspread": ExactMode.DYN_SPREAD,
        "dynzscore": ExactMode.DYN_ZSCORER,
        "dynzscorer": ExactMode.DYN_ZSCORER,
        "dynamiczscore": ExactMode.DYN_ZSCORER,
        "dynamiczscorer": ExactMode.DYN_ZSCORER,
        "ou": ExactMode.OU_SPREAD,
        "ouspread": ExactMode.OU_SPREAD,
        "ouzscore": ExactMode.OU_ZSCORER,
        "ouzscorer": ExactMode.OU_ZSCORER,
        "copula": ExactMode.COPULA,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported exact mode: {value}") from exc


def mode_contract(value: str | ExactMode) -> ModeEconomicContract:
    return MODE_CONTRACT_BY_MODE[normalize_exact_mode(value)]


def replay_label(value: str | ExactMode) -> str:
    return mode_contract(value).replay_label


def signal_for_action(value: str | TradeAction) -> float:
    action = TradeAction(value)
    if action == TradeAction.SHORT_X_LONG_Y:
        return 1.0
    if action == TradeAction.LONG_X_SHORT_Y:
        return -1.0
    return 0.0


def action_for_signal(value: float) -> TradeAction:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("signal must be finite")
    if numeric > 0.0:
        return TradeAction.SHORT_X_LONG_Y
    if numeric < 0.0:
        return TradeAction.LONG_X_SHORT_Y
    return TradeAction.FLAT


def swapped_action(value: str | TradeAction) -> TradeAction:
    action = TradeAction(value)
    if action == TradeAction.SHORT_X_LONG_Y:
        return TradeAction.LONG_X_SHORT_Y
    if action == TradeAction.LONG_X_SHORT_Y:
        return TradeAction.SHORT_X_LONG_Y
    return action


def tail_actions(
    value: str | ExactMode,
    *,
    copula_direction_view: str = "u1_given_u2",
) -> tuple[TradeAction, TradeAction]:
    """Return `(lower_tail, upper_tail)` actions for one canonical mode."""

    mode = normalize_exact_mode(value)
    if mode != ExactMode.COPULA:
        # spread = log(Y) - beta_y_on_x * log(X)
        # Lower tail: Y is cheap versus X -> short X, long Y.
        # Upper tail: Y is rich versus X -> long X, short Y.
        return TradeAction.SHORT_X_LONG_Y, TradeAction.LONG_X_SHORT_Y
    view = normalize_copula_view(copula_direction_view)
    if view == "u1_given_u2":
        # Lower conditional rank means X is cheap; upper means X is rich.
        return TradeAction.LONG_X_SHORT_Y, TradeAction.SHORT_X_LONG_Y
    # Lower conditional rank means Y is cheap; upper means Y is rich.
    return TradeAction.SHORT_X_LONG_Y, TradeAction.LONG_X_SHORT_Y


def action_for_threshold_operator(
    value: str | ExactMode,
    operator: object,
    *,
    copula_direction_view: str = "u1_given_u2",
) -> tuple[str, TradeAction] | None:
    """Map a one-sided threshold operator to its canonical tail and action."""

    normalized = str(operator or "").strip().lower().replace(" ", "_")
    lower_operators = {"<", "<=", "lt", "lte", "less_than", "less_than_or_equal", "less_or_equal"}
    upper_operators = {
        ">",
        ">=",
        "gt",
        "gte",
        "greater_than",
        "greater_than_or_equal",
        "greater_or_equal",
    }
    lower_action, upper_action = tail_actions(
        value,
        copula_direction_view=copula_direction_view,
    )
    if normalized in lower_operators:
        return "lower", lower_action
    if normalized in upper_operators:
        return "upper", upper_action
    return None


def normalize_copula_view(value: object) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value or "").lower())
    aliases = {
        "u1givenu2": "u1_given_u2",
        "xgiveny": "u1_given_u2",
        "conditionalu1u2": "u1_given_u2",
        "u2givenu1": "u2_given_u1",
        "ygivenx": "u2_given_u1",
        "conditionalu2u1": "u2_given_u1",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"unsupported copula direction view: {value}") from exc


def copula_distortion_signal(distortion: pd.Series, threshold: float) -> pd.Series:
    """Map positive X-rich distortion to short X and negative X-cheap to long X."""

    if not math.isfinite(float(threshold)) or float(threshold) <= 0.0:
        raise ValueError("copula threshold must be positive and finite")
    numeric = pd.to_numeric(distortion, errors="coerce")
    signal = pd.Series(0.0, index=numeric.index, dtype="float64")
    signal = signal.mask(numeric > float(threshold), 1.0)
    signal = signal.mask(numeric < -float(threshold), -1.0)
    return signal


def rolling_y_on_x_beta(
    log_x: pd.Series,
    log_y: pd.Series,
    *,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Estimate the causal OLS slope in `log(Y) = alpha + beta * log(X)`."""

    window = int(window)
    minimum = int(min_periods or window)
    if window < 2 or minimum < 2 or minimum > window:
        raise ValueError("rolling beta requires 2 <= min_periods <= window")
    x = pd.to_numeric(log_x, errors="coerce")
    y = pd.to_numeric(log_y, errors="coerce")
    variance_x = x.rolling(window, min_periods=minimum).var(ddof=1)
    covariance_xy = x.rolling(window, min_periods=minimum).cov(y, ddof=1)
    return covariance_xy.div(variance_x.where(variance_x.abs() > 1e-12))


def y_on_x_beta(log_x: pd.Series, log_y: pd.Series) -> float:
    """Estimate the full-window OLS slope in `log(Y) = alpha + beta * log(X)`."""

    values = pd.DataFrame(
        {
            "x": pd.to_numeric(log_x, errors="coerce"),
            "y": pd.to_numeric(log_y, errors="coerce"),
        }
    ).dropna()
    if len(values) < 2:
        raise ValueError("y-on-x beta requires at least two aligned observations")
    variance_x = float(values["x"].var(ddof=1))
    if not math.isfinite(variance_x) or variance_x <= 1e-12:
        raise ValueError("y-on-x beta requires nonzero finite X variance")
    beta = float(values["x"].cov(values["y"], ddof=1) / variance_x)
    if not math.isfinite(beta):
        raise ValueError("y-on-x beta must be finite")
    return beta


def y_on_x_log_spread(
    price_x: pd.Series,
    price_y: pd.Series,
    beta_y_on_x: float | pd.Series,
) -> pd.Series:
    x = pd.to_numeric(price_x, errors="coerce")
    y = pd.to_numeric(price_y, errors="coerce")
    beta = (
        pd.Series(float(beta_y_on_x), index=x.index, dtype="float64")
        if np.isscalar(beta_y_on_x)
        else pd.to_numeric(beta_y_on_x, errors="coerce").reindex(x.index)
    )
    valid = (x > 0.0) & (y > 0.0)
    return np.log(y.where(valid)) - beta * np.log(x.where(valid))


def y_on_x_log_residual(
    price_x: pd.Series,
    price_y: pd.Series,
    *,
    alpha: float,
    beta_y_on_x: float | pd.Series,
) -> pd.Series:
    """Return the fitted equilibrium residual, including the OLS intercept."""

    if not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    return y_on_x_log_spread(price_x, price_y, beta_y_on_x) - float(alpha)


def normalized_two_leg_weights(
    signal: pd.Series,
    beta_y_on_x: float | pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return gross-one X/Y target weights under the canonical signal convention."""

    position = pd.to_numeric(signal, errors="coerce").fillna(0.0)
    beta = (
        pd.Series(float(beta_y_on_x), index=position.index, dtype="float64")
        if np.isscalar(beta_y_on_x)
        else pd.to_numeric(beta_y_on_x, errors="coerce").reindex(position.index)
    )
    invalid_beta = (~np.isfinite(beta)) | beta.le(0.0)
    if bool(invalid_beta.any()):
        raise ValueError("hedge ratio must be finite and positive")
    scale = 1.0 + beta.abs()
    weight_x = -position * beta / scale
    weight_y = position / scale
    return weight_x.rename("weight_x"), weight_y.rename("weight_y")


def normalized_two_leg_weight_magnitudes(beta_y_on_x: float) -> tuple[float, float]:
    """Return gross-one absolute X/Y weights for margin and capacity calculations."""

    beta = float(beta_y_on_x)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("hedge ratio must be finite and positive")
    scale = 1.0 + abs(beta)
    return abs(beta) / scale, 1.0 / scale
