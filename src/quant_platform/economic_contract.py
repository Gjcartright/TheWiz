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
WEIGHT_SIGNAL_CONTRACT_VERSION = "bounded-two-leg-signal.v1"
ROLLING_BETA_IMPLEMENTATION_VERSION = "rolling-y-on-x-beta.exact-window.v2"
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


def positive_log_price(price: pd.Series) -> pd.Series:
    """Return the natural log of a complete, finite, strictly positive price series."""

    numeric = pd.to_numeric(price, errors="coerce")
    invalid = (~np.isfinite(numeric)) | numeric.le(0.0)
    if bool(invalid.any()):
        raise ValueError("prices must be complete, finite, and positive")
    return np.log(numeric).rename("log_price")


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
    """Estimate causal Y-on-X OLS using one paired finite mask per window.

    Partial windows use only jointly observed X/Y pairs, with the same sample
    count and ddof for covariance and variance. Rows are not compressed, so the
    window remains a window of the original observations. Each window has fresh
    centered/scaled two-pass moments: neither evicted values nor cancellation of
    large uncentered products can change a quiet window's hedge estimate.
    Complexity is O(N * window), with O(window) temporary space. A full-minimum
    call on exactly one window computes only that window.
    """

    window = int(window)
    minimum = window if min_periods is None else int(min_periods)
    if window < 2 or minimum < 2 or minimum > window:
        raise ValueError("rolling beta requires 2 <= min_periods <= window")
    if not log_x.index.equals(log_y.index) or not log_x.index.is_unique:
        raise ValueError("rolling beta requires an identical unique index")
    x = pd.to_numeric(log_x, errors="coerce")
    y = pd.to_numeric(log_y, errors="coerce")
    if np.iscomplexobj(x) or np.iscomplexobj(y):
        raise ValueError("rolling beta requires real observations")
    x_values = x.to_numpy(dtype=float, na_value=np.nan)
    y_values = y.to_numpy(dtype=float, na_value=np.nan)
    result = np.full(len(x), np.nan, dtype=float)

    def centered_scaled(values: np.ndarray) -> tuple[np.ndarray, float]:
        with np.errstate(over="ignore"):
            centered = values - values[0]
        scale = float(np.max(np.abs(centered)))
        if math.isinf(scale):
            scale = float(np.max(np.abs(values)))
            normalized = values / scale
            centered = normalized - normalized[0]
        elif scale:
            centered = centered / scale
        mean = math.fsum(centered) / len(centered)
        return centered - mean, scale

    for end in range(minimum - 1, len(x)):
        start = max(0, end - window + 1)
        x_window, y_window = x_values[start:end + 1], y_values[start:end + 1]
        paired = np.isfinite(x_window) & np.isfinite(y_window)
        count = int(paired.sum())
        if count < minimum:
            continue
        centered_x, scale_x = centered_scaled(x_window[paired])
        sxx = math.fsum(value * value for value in centered_x)
        # This remains a sample-variance floor, not a sum-of-squares floor.
        variance_x = (scale_x * (sxx / (count - 1))) * scale_x
        if not variance_x > 1e-12:
            continue
        centered_y, scale_y = centered_scaled(y_window[paired])
        sxy = math.fsum(a * b for a, b in zip(centered_x, centered_y))
        beta = ((sxy / sxx) * scale_y) / scale_x
        if math.isfinite(beta):
            result[end] = beta
    # The pre-existing rolling covariance contract returns unnamed float64,
    # including for pandas nullable numeric inputs.
    return pd.Series(result, index=log_x.index, dtype=float)


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
    """Return bounded X/Y target weights under the canonical signal convention.

    Discrete signals -1/0/+1 remain gross-one/flat. Existing fractional sizing
    in [-1, 1] remains supported and has gross exposure abs(signal); leverage
    above one is not this contract. Missing is never silently treated as flat.
    Numeric text remains accepted, but booleans and complex values do not.
    """

    if not isinstance(signal, pd.Series) or not signal.index.is_unique:
        raise ValueError("signal must have a unique Series index")
    if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
           for value in signal):
        raise ValueError("signal must be real numeric, not boolean or complex")
    position = pd.to_numeric(signal, errors="coerce").astype("float64")
    if bool((~np.isfinite(position) | position.abs().gt(1.0)).any()):
        raise ValueError("signal must be complete, finite, and within [-1, 1]")
    if isinstance(beta_y_on_x, pd.Series):
        if not beta_y_on_x.index.equals(position.index):
            raise ValueError("hedge ratio must share the signal index and order")
        if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
               for value in beta_y_on_x):
            raise ValueError("hedge ratio must be real, finite and positive")
        beta = pd.to_numeric(beta_y_on_x, errors="coerce").astype("float64")
    else:
        if isinstance(beta_y_on_x, (bool, np.bool_, complex, np.complexfloating)):
            raise ValueError("hedge ratio must be real, finite and positive")
        try:
            scalar_beta = float(beta_y_on_x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("hedge ratio must be finite and positive") from exc
        if not math.isfinite(scalar_beta) or scalar_beta <= 0.0:
            raise ValueError("hedge ratio must be finite and positive")
        beta = pd.Series(scalar_beta, index=position.index, dtype="float64")
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


def lagged_two_leg_gross_return(
    price_x: pd.Series,
    price_y: pd.Series,
    target_weight_x: pd.Series,
    target_weight_y: pd.Series,
) -> pd.Series:
    """Return causal two-leg bar PnL using positions chosen one bar earlier."""

    index = price_x.index
    if not price_y.index.equals(index):
        raise ValueError("two-leg prices must share an identical index")
    if not target_weight_x.index.equals(index) or not target_weight_y.index.equals(index):
        raise ValueError("two-leg target weights must share the price index")
    x = pd.to_numeric(price_x, errors="coerce")
    y = pd.to_numeric(price_y, errors="coerce")
    invalid = (~np.isfinite(x)) | (~np.isfinite(y)) | x.le(0.0) | y.le(0.0)
    if bool(invalid.any()):
        raise ValueError("two-leg prices must be complete, finite, and positive")
    weight_x = pd.to_numeric(target_weight_x, errors="coerce")
    weight_y = pd.to_numeric(target_weight_y, errors="coerce")
    if bool((~np.isfinite(weight_x) | ~np.isfinite(weight_y)).any()):
        raise ValueError("two-leg target weights must be complete and finite")
    returns_x = x.pct_change().fillna(0.0)
    returns_y = y.pct_change().fillna(0.0)
    return (
        weight_x.shift(1).fillna(0.0) * returns_x + weight_y.shift(1).fillna(0.0) * returns_y
    ).rename("gross_return")


def two_leg_turnover(
    target_weight_x: pd.Series,
    target_weight_y: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return X, Y, and aggregate turnover from gross-one target weights."""

    if not target_weight_x.index.equals(target_weight_y.index):
        raise ValueError("two-leg target weights must share an identical index")
    weight_x = pd.to_numeric(target_weight_x, errors="coerce")
    weight_y = pd.to_numeric(target_weight_y, errors="coerce")
    if bool((~np.isfinite(weight_x) | ~np.isfinite(weight_y)).any()):
        raise ValueError("two-leg target weights must be complete and finite")
    turnover_x = (weight_x - weight_x.shift(1).fillna(0.0)).abs().rename("turnover_x")
    turnover_y = (weight_y - weight_y.shift(1).fillna(0.0)).abs().rename("turnover_y")
    return turnover_x, turnover_y, (turnover_x + turnover_y).rename("turnover")


def turnover_rate_cost(turnover: pd.Series, rate_bps: float) -> pd.Series:
    """Apply a nonnegative basis-point rate to a turnover stream."""

    rate = float(rate_bps)
    if not math.isfinite(rate) or rate < 0.0:
        raise ValueError("turnover cost rate must be finite and nonnegative")
    numeric = pd.to_numeric(turnover, errors="coerce")
    if bool((~np.isfinite(numeric) | numeric.lt(0.0)).any()):
        raise ValueError("turnover must be complete, finite, and nonnegative")
    return numeric.mul(rate / 10_000.0)


def expected_partial_fill_cost(
    turnover: pd.Series,
    *,
    probability: float,
    fill_fraction: float,
    penalty_bps: float,
) -> pd.Series:
    """Return the declared expected partial-fill penalty on turnover."""

    probability_value = float(probability)
    fill_value = float(fill_fraction)
    if not 0.0 <= probability_value <= 1.0:
        raise ValueError("partial-fill probability must be within [0, 1]")
    if not 0.0 <= fill_value <= 1.0:
        raise ValueError("partial-fill fraction must be within [0, 1]")
    return turnover_rate_cost(
        turnover,
        probability_value * (1.0 - fill_value) * float(penalty_bps),
    )
