"""Fail-closed research risk and performance metrics.

This module has no Wizard, execution, or live-risk authority.  It provides
isolated research calculations whose results carry their method, unit, sign
convention, sample sufficiency, and explicit research-only authority.  A
metric that cannot be supported is returned with ``value=None`` and
``status="unavailable"``; unavailable is never encoded as numeric zero.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from numbers import Integral
from typing import Literal

import numpy as np

MetricStatus = Literal["valid", "unavailable"]
ResearchAuthority = Literal["research_only_no_wizard_or_live_authority"]

RESEARCH_ONLY_AUTHORITY: ResearchAuthority = "research_only_no_wizard_or_live_authority"


@dataclass(frozen=True, slots=True)
class ScalarMetricResult:
    """Typed result for one scalar research metric."""

    metric: str
    value: float | None
    status: MetricStatus
    unit: str
    method: str
    convention: str
    observations: int
    required_observations: int
    reason: str
    authority: ResearchAuthority = RESEARCH_ONLY_AUTHORITY


@dataclass(frozen=True, slots=True)
class TailRiskResult:
    """Typed joint result for historical VaR and CVaR."""

    value_at_risk: float | None
    conditional_value_at_risk: float | None
    status: MetricStatus
    confidence: float | None
    unit: str
    method: str
    convention: str
    observations: int
    required_observations: int
    tail_observations: int
    required_tail_observations: int
    reason: str
    authority: ResearchAuthority = RESEARCH_ONLY_AUTHORITY


def calculate_profit_factor(
    net_pnls: Iterable[float], *, min_observations: int = 1
) -> ScalarMetricResult:
    """Return gross positive net P&L divided by absolute gross negative net P&L.

    Inputs are trade-level net P&L in one consistent currency and should
    already include fees, funding, slippage, and other realized costs.  Zero
    P&L trades remain observations but contribute to neither sum.  No loss
    observation makes the denominator unsupported, so the result is
    unavailable rather than positive infinity.
    """

    metric = "profit_factor"
    unit = "ratio"
    method = "sum_positive_net_pnl/abs(sum_negative_net_pnl)"
    convention = "trade net P&L; positive=profit, negative=loss, zero=breakeven"
    required = _positive_integer(min_observations)
    if required is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 0, "invalid_min_observations"
        )
    values, error = _finite_vector(net_pnls)
    if error is not None or values is None:
        return _scalar_unavailable(metric, unit, method, convention, 0, required, error)
    if values.size < required:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            int(values.size),
            required,
            "insufficient_observations",
        )

    gross_profit = _finite_sum(values[values > 0.0])
    signed_gross_loss = _finite_sum(values[values < 0.0])
    if gross_profit is None or signed_gross_loss is None:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            int(values.size),
            required,
            "nonfinite_aggregate",
        )
    gross_loss = -signed_gross_loss
    if gross_loss == 0.0:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            int(values.size),
            required,
            "no_loss_observations",
        )
    return _scalar_valid(
        metric,
        gross_profit / gross_loss,
        unit,
        method,
        convention,
        int(values.size),
        required,
    )


def calculate_payoff_ratio(
    net_pnls: Iterable[float], *, min_observations: int = 1
) -> ScalarMetricResult:
    """Return average winning net P&L divided by absolute average losing net P&L.

    Inputs are trade-level net P&L in one consistent currency, net of realized
    costs.  Positive and negative trades define wins and losses; breakeven
    trades are neither.  Both a win and a loss observation are required.
    """

    metric = "payoff_ratio"
    unit = "ratio"
    method = "mean(positive_net_pnl)/abs(mean(negative_net_pnl))"
    convention = "trade net P&L; positive=win, negative=loss, zero=breakeven"
    required = _positive_integer(min_observations)
    if required is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 0, "invalid_min_observations"
        )
    values, error = _finite_vector(net_pnls)
    if error is not None or values is None:
        return _scalar_unavailable(metric, unit, method, convention, 0, required, error)
    observations = int(values.size)
    if observations < required:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "insufficient_observations",
        )
    wins = values[values > 0.0]
    losses = values[values < 0.0]
    if wins.size == 0:
        return _scalar_unavailable(
            metric, unit, method, convention, observations, required, "no_win_observations"
        )
    if losses.size == 0:
        return _scalar_unavailable(
            metric, unit, method, convention, observations, required, "no_loss_observations"
        )
    gross_wins = _finite_sum(wins)
    gross_losses = _finite_sum(losses)
    if gross_wins is None or gross_losses is None:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "nonfinite_aggregate",
        )
    average_win = gross_wins / int(wins.size)
    average_loss = gross_losses / int(losses.size)
    return _scalar_valid(
        metric,
        average_win / abs(average_loss),
        unit,
        method,
        convention,
        observations,
        required,
    )


def calculate_realized_r(net_pnl: float, initial_risk: float) -> ScalarMetricResult:
    """Return one trade's net P&L divided by its planned initial risk amount.

    ``net_pnl`` and ``initial_risk`` must use the same currency.  Net P&L is
    signed after realized costs; planned initial risk is a strictly positive
    pre-trade loss budget.  The returned unit is ``R``: positive is profit,
    negative is loss, and an exactly breakeven trade is a valid numeric zero.
    """

    metric = "realized_r"
    unit = "R_multiple"
    method = "net_pnl/planned_initial_risk"
    convention = (
        "same-currency amounts; net P&L positive=profit; planned initial risk strictly positive"
    )
    pnl = _finite_scalar(net_pnl)
    risk = _finite_scalar(initial_risk)
    if pnl is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 1, "invalid_or_nonfinite_net_pnl"
        )
    if risk is None or risk <= 0.0:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 1, "initial_risk_must_be_finite_and_positive"
        )
    return _scalar_valid(metric, pnl / risk, unit, method, convention, 1, 1)


def calculate_expectancy_r(
    realized_rs: Iterable[float], *, min_observations: int = 1
) -> ScalarMetricResult:
    """Return arithmetic mean realized R per completed trade.

    Each input is a signed realized R multiple calculated from net P&L and the
    trade's planned initial risk.  Positive is profit, negative is loss, and
    zero is breakeven.  No missing or nonfinite observation is silently dropped.
    """

    metric = "expectancy_r"
    unit = "R_per_trade"
    method = "arithmetic_mean(realized_R)"
    convention = "completed-trade realized R; positive=profit, negative=loss, zero=breakeven"
    required = _positive_integer(min_observations)
    if required is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 0, "invalid_min_observations"
        )
    values, error = _finite_vector(realized_rs)
    if error is not None or values is None:
        return _scalar_unavailable(metric, unit, method, convention, 0, required, error)
    observations = int(values.size)
    if observations < required:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "insufficient_observations",
        )
    total = _finite_sum(values)
    if total is None:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "nonfinite_aggregate",
        )
    return _scalar_valid(
        metric, total / observations, unit, method, convention, observations, required
    )


def calculate_sortino_ratio(
    periodic_returns: Iterable[float],
    *,
    periods_per_year: float,
    target_return: float = 0.0,
    min_observations: int = 2,
) -> ScalarMetricResult:
    """Return an annualized target-downside-deviation Sortino ratio.

    Inputs and ``target_return`` are decimal simple returns per the same period.
    The downside deviation is ``sqrt(mean(min(r-target, 0)^2))`` over *all*
    observations (population denominator), and the ratio is multiplied by
    ``sqrt(periods_per_year)``.  Annualization is caller-declared and is not
    inferred.  A zero downside deviation is unavailable rather than infinity.
    """

    metric = "sortino_ratio"
    unit = "annualized_ratio"
    method = "sqrt(periods_per_year)*mean(r-target)/target_downside_deviation_ddof0"
    convention = "decimal simple periodic returns; target is per-period; downside squares use all observations"
    required = _positive_integer(min_observations)
    if required is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 0, "invalid_min_observations"
        )
    annualization = _finite_scalar(periods_per_year)
    if annualization is None or annualization <= 0.0:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, required, "invalid_periods_per_year"
        )
    target = _finite_scalar(target_return)
    if target is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, required, "invalid_target_return"
        )
    values, error = _finite_vector(periodic_returns)
    if error is not None or values is None:
        return _scalar_unavailable(metric, unit, method, convention, 0, required, error)
    observations = int(values.size)
    if observations < required:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "insufficient_observations",
        )

    excess = values - target
    excess_sum = _finite_sum(excess)
    downside = np.minimum(excess, 0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_downside = np.square(downside)
    downside_sum = _finite_sum(squared_downside)
    if excess_sum is None or downside_sum is None:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "nonfinite_aggregate",
        )
    downside_deviation = math.sqrt(downside_sum / observations)
    if downside_deviation == 0.0:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "zero_downside_deviation",
        )
    value = math.sqrt(annualization) * (excess_sum / observations) / downside_deviation
    return _scalar_valid(metric, value, unit, method, convention, observations, required)


def calculate_calmar_ratio(
    periodic_returns: Iterable[float],
    *,
    periods_per_year: float,
    min_observations: int = 2,
) -> ScalarMetricResult:
    """Return compounded annual return divided by maximum drawdown magnitude.

    Inputs are decimal simple periodic returns and must all be greater than
    -100%.  Equity begins at one, compounds in sequence, and maximum drawdown is
    the largest peak-to-trough percentage magnitude including the initial
    equity.  Annual return is ``prod(1+r) ** (periods_per_year/n) - 1``.  A path
    with zero drawdown has no finite Calmar ratio and is unavailable.
    """

    metric = "calmar_ratio"
    unit = "annualized_return_per_drawdown"
    method = "compounded_annual_return/max_peak_to_trough_drawdown"
    convention = "decimal simple periodic returns; drawdown is a nonnegative magnitude"
    required = _positive_integer(min_observations)
    if required is None:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, 0, "invalid_min_observations"
        )
    annualization = _finite_scalar(periods_per_year)
    if annualization is None or annualization <= 0.0:
        return _scalar_unavailable(
            metric, unit, method, convention, 0, required, "invalid_periods_per_year"
        )
    values, error = _finite_vector(periodic_returns)
    if error is not None or values is None:
        return _scalar_unavailable(metric, unit, method, convention, 0, required, error)
    observations = int(values.size)
    if observations < required:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "insufficient_observations",
        )
    if bool(np.any(values <= -1.0)):
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "simple_return_at_or_below_minus_one",
        )

    log_growth = _finite_sum(np.log1p(values))
    if log_growth is None:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "nonfinite_compounded_growth",
        )
    try:
        annual_return = math.expm1(log_growth * annualization / observations)
    except OverflowError:
        annual_return = float("inf")
    if not math.isfinite(annual_return):
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "nonfinite_annual_return",
        )

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for periodic_return in values:
        equity *= 1.0 + float(periodic_return)
        if not math.isfinite(equity) or equity <= 0.0:
            return _scalar_unavailable(
                metric,
                unit,
                method,
                convention,
                observations,
                required,
                "nonfinite_compounded_equity",
            )
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
    if max_drawdown == 0.0:
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required,
            "zero_max_drawdown",
        )
    return _scalar_valid(
        metric,
        annual_return / max_drawdown,
        unit,
        method,
        convention,
        observations,
        required,
    )


def calculate_historical_var_cvar(
    periodic_returns: Iterable[float],
    *,
    confidence: float = 0.95,
    min_observations: int = 100,
    min_tail_observations: int = 5,
    quantile_method: str = "linear",
) -> TailRiskResult:
    """Return historical VaR and CVaR from decimal periodic returns.

    Loss is defined as ``-return``.  VaR is the requested empirical loss
    quantile and CVaR is the arithmetic mean of observed losses greater than or
    equal to that VaR threshold.  Positive outputs are losses and negative
    outputs are gains.  The method requires both ``min_observations`` and enough
    total observations for ``min_tail_observations / (1-confidence)``; tied
    threshold observations are all retained.  No missing value is dropped.
    """

    unit = "decimal_return_loss"
    method = f"historical_quantile[{quantile_method}];mean(loss>=VaR)"
    convention = "loss=-decimal periodic return; positive=loss, negative=gain; inclusive tail"
    confidence_value = _finite_scalar(confidence)
    required = _positive_integer(min_observations)
    required_tail = _positive_integer(min_tail_observations)
    if confidence_value is None or not 0.0 < confidence_value < 1.0:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            required or 0,
            0,
            required_tail or 0,
            "confidence_must_be_between_zero_and_one",
        )
    if required is None:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            0,
            0,
            required_tail or 0,
            "invalid_min_observations",
        )
    if required_tail is None:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            required,
            0,
            0,
            "invalid_min_tail_observations",
        )
    if not isinstance(quantile_method, str) or not quantile_method:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            required,
            0,
            required_tail,
            "invalid_quantile_method",
        )

    try:
        decimal_tail_probability = Decimal(1) - Decimal(str(confidence_value))
        tail_implied_required = int(
            (Decimal(required_tail) / decimal_tail_probability).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
    except (InvalidOperation, OverflowError, ZeroDivisionError):
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            required,
            0,
            required_tail,
            "confidence_tail_requirement_not_representable",
        )
    effective_required = max(required, tail_implied_required)
    values, error = _finite_vector(periodic_returns)
    if error is not None or values is None:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            0,
            effective_required,
            0,
            required_tail,
            error,
        )
    observations = int(values.size)
    if observations < effective_required:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            observations,
            effective_required,
            0,
            required_tail,
            "insufficient_observations_for_requested_tail",
        )

    losses = -values
    try:
        value_at_risk = float(np.quantile(losses, confidence_value, method=quantile_method))
    except (TypeError, ValueError):
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            observations,
            effective_required,
            0,
            required_tail,
            "unsupported_quantile_method",
        )
    if not math.isfinite(value_at_risk):
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            observations,
            effective_required,
            0,
            required_tail,
            "nonfinite_value_at_risk",
        )
    tail_losses = losses[losses >= value_at_risk]
    tail_observations = int(tail_losses.size)
    if tail_observations < required_tail:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            observations,
            effective_required,
            tail_observations,
            required_tail,
            "insufficient_observed_tail",
        )
    tail_sum = _finite_sum(tail_losses)
    if tail_sum is None:
        return _tail_unavailable(
            confidence_value,
            unit,
            method,
            convention,
            observations,
            effective_required,
            tail_observations,
            required_tail,
            "nonfinite_tail_aggregate",
        )
    conditional_value_at_risk = tail_sum / tail_observations
    return TailRiskResult(
        value_at_risk=value_at_risk,
        conditional_value_at_risk=conditional_value_at_risk,
        status="valid",
        confidence=confidence_value,
        unit=unit,
        method=method,
        convention=convention,
        observations=observations,
        required_observations=effective_required,
        tail_observations=tail_observations,
        required_tail_observations=required_tail,
        reason="calculated",
    )


def _positive_integer(value: object) -> int | None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        return None
    integer = int(value)
    return integer if integer > 0 else None


def _finite_scalar(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _finite_vector(values: Iterable[float]) -> tuple[np.ndarray | None, str | None]:
    try:
        materialized = tuple(values)
    except TypeError:
        return None, "input_is_not_an_iterable"
    if any(isinstance(value, (bool, np.bool_)) for value in materialized):
        return None, "input_contains_nonnumeric_observation"
    try:
        vector = np.asarray(materialized, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None, "input_contains_nonnumeric_observation"
    if vector.ndim != 1:
        return None, "input_must_be_one_dimensional"
    if vector.size == 0:
        return None, "no_observations"
    if bool(np.any(~np.isfinite(vector))):
        return None, "input_contains_nonfinite_observation"
    return vector, None


def _finite_sum(values: np.ndarray) -> float | None:
    try:
        total = math.fsum(float(value) for value in values)
    except (OverflowError, ValueError):
        return None
    return total if math.isfinite(total) else None


def _scalar_valid(
    metric: str,
    value: float,
    unit: str,
    method: str,
    convention: str,
    observations: int,
    required_observations: int,
) -> ScalarMetricResult:
    if not math.isfinite(value):
        return _scalar_unavailable(
            metric,
            unit,
            method,
            convention,
            observations,
            required_observations,
            "nonfinite_result",
        )
    return ScalarMetricResult(
        metric=metric,
        value=float(value),
        status="valid",
        unit=unit,
        method=method,
        convention=convention,
        observations=observations,
        required_observations=required_observations,
        reason="calculated",
    )


def _scalar_unavailable(
    metric: str,
    unit: str,
    method: str,
    convention: str,
    observations: int,
    required_observations: int,
    reason: str | None,
) -> ScalarMetricResult:
    return ScalarMetricResult(
        metric=metric,
        value=None,
        status="unavailable",
        unit=unit,
        method=method,
        convention=convention,
        observations=observations,
        required_observations=required_observations,
        reason=reason or "unavailable",
    )


def _tail_unavailable(
    confidence: float | None,
    unit: str,
    method: str,
    convention: str,
    observations: int,
    required_observations: int,
    tail_observations: int,
    required_tail_observations: int,
    reason: str | None,
) -> TailRiskResult:
    return TailRiskResult(
        value_at_risk=None,
        conditional_value_at_risk=None,
        status="unavailable",
        confidence=confidence,
        unit=unit,
        method=method,
        convention=convention,
        observations=observations,
        required_observations=required_observations,
        tail_observations=tail_observations,
        required_tail_observations=required_tail_observations,
        reason=reason or "unavailable",
    )


__all__ = [
    "RESEARCH_ONLY_AUTHORITY",
    "ScalarMetricResult",
    "TailRiskResult",
    "calculate_calmar_ratio",
    "calculate_expectancy_r",
    "calculate_historical_var_cvar",
    "calculate_payoff_ratio",
    "calculate_profit_factor",
    "calculate_realized_r",
    "calculate_sortino_ratio",
]
