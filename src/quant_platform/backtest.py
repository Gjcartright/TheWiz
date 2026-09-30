from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_platform.economic_contract import (
    expected_partial_fill_cost,
    lagged_two_leg_gross_return,
    normalized_two_leg_weights,
    turnover_rate_cost,
    two_leg_turnover,
)
from quant_platform.performance_math import (
    MATH_VERSION,
    calculate_annualized_sharpe,
    calculate_max_drawdown,
    compound_simple_returns,
    resolve_annualization,
)
from quant_platform.risk_metrics import calculate_profit_factor
from quant_platform.trade_ledger import TradeLedgerResult, build_trade_ledger


class FundingPolicy(StrEnum):
    SIGNED_REALIZED = "signed_realized"
    CONSERVATIVE_ABSOLUTE_DRAG = "conservative_absolute_drag"


class RebalancePolicy(StrEnum):
    TARGET_WEIGHTS_EVERY_BAR = "target_weights_every_bar"
    FIXED_UNITS_UNTIL_EXIT = "fixed_units_until_exit"


@dataclass(frozen=True)
class CostModel:
    taker_fee_bps: float = 5.0
    slippage_bps: float = 4.0
    execution_risk_bps: float = 2.0
    funding_bps_per_day: float = 1.0
    bars_per_day: int = 24
    partial_fill_probability: float = 0.10
    partial_fill_fraction: float = 0.5
    partial_fill_penalty_bps: float = 2.0
    funding_policy: str = FundingPolicy.CONSERVATIVE_ABSOLUTE_DRAG.value
    rebalance_policy: str = RebalancePolicy.TARGET_WEIGHTS_EVERY_BAR.value

    def __post_init__(self) -> None:
        for field_name in (
            "taker_fee_bps",
            "slippage_bps",
            "execution_risk_bps",
            "partial_fill_penalty_bps",
        ):
            raw = getattr(self, field_name)
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{field_name} must be finite and nonnegative")
            value = float(raw)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if isinstance(self.funding_bps_per_day, (bool, np.bool_)):
            raise ValueError("funding_bps_per_day must be finite")
        if not np.isfinite(float(self.funding_bps_per_day)):
            raise ValueError("funding_bps_per_day must be finite")
        bars = float(self.bars_per_day)
        if isinstance(self.bars_per_day, (bool, np.bool_)) or not np.isfinite(bars) or bars <= 0.0 or not bars.is_integer():
            raise ValueError("bars_per_day must be a finite positive integer")
        for field_name in ("partial_fill_probability", "partial_fill_fraction"):
            raw = getattr(self, field_name)
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"{field_name} must be within [0, 1]")
            value = float(raw)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
        self.normalized_funding_policy()
        self.normalized_rebalance_policy()

    def round_trip_cost(self) -> float:
        bps = 2 * (self.taker_fee_bps + self.slippage_bps + self.execution_risk_bps)
        return bps / 10_000.0

    def funding_per_bar(self, *, bars_per_day: float | None = None) -> float:
        raw_divisor = self.bars_per_day if bars_per_day is None else bars_per_day
        if isinstance(raw_divisor, (bool, np.bool_)):
            raise ValueError("bars_per_day must be finite and positive")
        divisor = float(raw_divisor)
        if not np.isfinite(divisor) or divisor <= 0.0:
            raise ValueError("bars_per_day must be finite and positive")
        return (self.funding_bps_per_day / 10_000.0) / divisor

    def normalized_funding_policy(self) -> FundingPolicy:
        try:
            return FundingPolicy(self.funding_policy)
        except ValueError as exc:
            raise ValueError(f"unsupported funding policy: {self.funding_policy}") from exc

    def normalized_rebalance_policy(self) -> RebalancePolicy:
        try:
            return RebalancePolicy(self.rebalance_policy)
        except ValueError as exc:
            raise ValueError(f"unsupported rebalance policy: {self.rebalance_policy}") from exc


@dataclass(frozen=True)
class BacktestResult:
    trades: int
    profit_factor: float
    expectancy: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_return: float
    gross_return: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_funding: float = 0.0
    total_execution_risk: float = 0.0
    total_partial_fill_cost: float = 0.0
    avg_gross_exposure: float = 0.0
    open_trades: int = 0
    interval: str = ""
    periods_per_year: float | None = None
    sharpe_status: str = "blocked"
    funding_policy: str = FundingPolicy.CONSERVATIVE_ABSOLUTE_DRAG.value
    rebalance_policy: str = RebalancePolicy.TARGET_WEIGHTS_EVERY_BAR.value
    reconciliation_error: float = 0.0
    closed_trade_return_std: float = 0.0
    expectancy_lower_95: float | None = None
    math_version: str = MATH_VERSION
    profit_factor_status: str = "unavailable"
    profit_factor_reason: str = "not_evaluated"
    profit_factor_basis: str = "closed_trade_net_pnl_in_initial_capital_currency_units"


def max_drawdown(equity: pd.Series) -> float:
    return calculate_max_drawdown(equity)


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: float | None = None,
    *,
    interval: object | None = None,
    timestamps: pd.Series | pd.Index | None = None,
) -> float:
    """Compatibility wrapper around the fail-closed Math V2 calculation."""

    return calculate_annualized_sharpe(
        returns,
        interval=interval,
        timestamps=timestamps,
        periods_per_year=periods_per_year,
    ).value


def _series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
               for value in frame[column]):
            raise ValueError(f"{column} contains non-real numeric values")
        values = pd.to_numeric(frame[column], errors="coerce")
        if bool((~np.isfinite(values)).any()):
            raise ValueError(f"{column} contains missing or nonfinite values")
        return values.astype("float64")
    return pd.Series(default, index=frame.index, dtype="float64")


def _validated_signal(frame: pd.DataFrame, signal: pd.Series) -> pd.Series:
    if not frame.index.is_unique or not signal.index.equals(frame.index):
        raise ValueError("signal must share the unique frame index and order")
    if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
           for value in signal):
        raise ValueError("signal must be real numeric, not boolean or complex")
    values = pd.to_numeric(signal, errors="coerce").astype(float)
    if bool((~np.isfinite(values)).any()):
        raise ValueError("signal must be complete and finite; missing is not flat")
    return values


def _uniform_contract_value(
    frame: pd.DataFrame,
    column: str,
    *,
    default: str,
    allowed: frozenset[str],
) -> str:
    """Require one complete declared economic interpretation per history."""

    if column not in frame.columns:
        return default
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"{column} must be complete and uniform")
    normalized = {str(value).strip().lower() for value in values}
    if len(normalized) != 1 or "" in normalized:
        raise ValueError(f"{column} must be complete and uniform")
    value = normalized.pop()
    if value not in allowed:
        raise ValueError(f"unsupported {column}: {value}")
    return value


def _two_leg_target_weights(
    signal: pd.Series,
    hedge_ratio: pd.Series,
    price_x: pd.Series,
    price_y: pd.Series,
    *,
    spread_orientation: str,
    hedge_ratio_kind: str,
) -> tuple[pd.Series, pd.Series]:
    """Convert explicit orientation and ratio units to gross-one leg weights."""

    if hedge_ratio_kind == "notional_weight" and spread_orientation == "y_on_x":
        return normalized_two_leg_weights(signal, hedge_ratio)
    # The canonical sizing helper validates missing, boolean, complex, and
    # nonpositive hedge ratios even for the alternate interpretations.
    normalized_two_leg_weights(signal, hedge_ratio)
    ratio = pd.to_numeric(hedge_ratio, errors="raise").astype("float64")
    if hedge_ratio_kind == "quantity_units":
        notional_x = ratio * price_x if spread_orientation == "y_on_x" else price_x
        notional_y = price_y if spread_orientation == "y_on_x" else ratio * price_y
    else:
        notional_x = ratio if spread_orientation == "y_on_x" else pd.Series(1.0, index=signal.index)
        notional_y = pd.Series(1.0, index=signal.index) if spread_orientation == "y_on_x" else ratio
    gross = notional_x + notional_y
    if bool((~np.isfinite(gross) | gross.le(0.0)).any()):
        raise ValueError("two-leg gross notional must be finite and positive")
    if spread_orientation == "y_on_x":
        return (-signal * notional_x / gross).rename("weight_x"), (
            signal * notional_y / gross
        ).rename("weight_y")
    return (signal * notional_x / gross).rename("weight_x"), (
        -signal * notional_y / gross
    ).rename("weight_y")


def backtest_pair(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel | None = None,
    *,
    interval: object | None = None,
) -> BacktestResult:
    """Backtest the legacy scalar spread-change proxy and its cost ledger.

    Spread differences are treated as model returns. This compatibility model
    is distinct from positive-price two-leg P&L and does not establish executed
    pair quantities, fills, or currency-profit evidence.
    """

    costs = cost_model or CostModel()
    if costs.normalized_rebalance_policy() is RebalancePolicy.FIXED_UNITS_UNTIL_EXIT:
        raise ValueError("fixed-unit rebalancing requires two-leg positive-price inputs")
    data = frame.copy()
    data["signal"] = _validated_signal(data, signal)
    spread = _series_or_default(data, "spread", float("nan"))
    if "spread" not in data:
        raise ValueError("spread is required")
    spread_return = spread.diff()
    if len(spread_return):
        spread_return.iloc[0] = 0.0
    position = data["signal"].shift(1).fillna(0.0)
    gross_return = position * spread_return
    turnover = data["signal"].diff().abs().fillna(data["signal"].abs())
    fee_cost = turnover * costs.taker_fee_bps / 10_000.0
    slippage_cost = turnover * costs.slippage_bps / 10_000.0
    execution_risk_cost = turnover * costs.execution_risk_bps / 10_000.0
    partial_fill_cost = (
        turnover
        * costs.partial_fill_probability
        * (1.0 - costs.partial_fill_fraction)
        * costs.partial_fill_penalty_bps
        / 10_000.0
    )
    bars_per_day = _bars_per_day(frame, interval, costs)
    funding_rate = costs.funding_per_bar(bars_per_day=bars_per_day)
    if costs.normalized_funding_policy() == FundingPolicy.SIGNED_REALIZED:
        funding_cost = position * funding_rate
    else:
        funding_cost = position.abs() * abs(funding_rate)
    component_costs = {
        "fees": fee_cost,
        "slippage": slippage_cost,
        "funding": funding_cost,
        "execution_risk": execution_risk_cost,
        "partial_fill": partial_fill_cost,
    }
    close_fraction = _scalar_reversal_close_fraction(data["signal"])
    timestamps = _timestamps(data)
    ledger = build_trade_ledger(
        data["signal"],
        gross_return,
        component_costs,
        timestamps=timestamps,
        reversal_close_fraction=close_fraction,
    )
    return _backtest_result(
        ledger,
        gross_return=gross_return,
        gross_exposure=position.abs(),
        costs=costs,
        interval=interval or _frame_interval(frame),
        timestamps=timestamps,
    )


def backtest_two_leg_spread_with_ledger(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel | None = None,
    *,
    interval: object | None = None,
) -> tuple[BacktestResult, TradeLedgerResult]:
    """Backtest normalized two-leg positions and return its trade ledger.

    Required columns: ``price_x`` and ``price_y``. ``hedge_ratio`` defines the
    exposure model. ``beta`` is intentionally ignored by Math V2 and remains a
    diagnostic field only.
    """

    costs = cost_model or CostModel()
    rebalance_policy = costs.normalized_rebalance_policy()
    required = {"price_x", "price_y", "hedge_ratio"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing two-leg price columns: {missing}")

    data = frame.copy()
    data["signal"] = _validated_signal(data, signal)
    if any(isinstance(value, (bool, np.bool_, complex, np.complexfloating))
           for column in ("price_x", "price_y") for value in data[column]):
        raise ValueError("two-leg prices must be real numeric, not boolean or complex")
    price_x = pd.to_numeric(data["price_x"], errors="coerce")
    price_y = pd.to_numeric(data["price_y"], errors="coerce")
    invalid_prices = (
        (~np.isfinite(price_x)) | (~np.isfinite(price_y)) | price_x.le(0.0) | price_y.le(0.0)
    )
    if bool(invalid_prices.any()):
        raise ValueError("two-leg prices must be complete, finite, and positive")
    # Preserve supplied types until the canonical sizing boundary validates
    # them; coercing first would turn bool into one or discard complex parts.
    spread_orientation = _uniform_contract_value(
        data,
        "spread_orientation",
        default="y_on_x",
        allowed=frozenset({"y_on_x", "x_on_y"}),
    )
    hedge_ratio_kind = _uniform_contract_value(
        data,
        "hedge_ratio_kind",
        default="notional_weight",
        allowed=frozenset({"notional_weight", "quantity_units"}),
    )
    target_weight_x, target_weight_y = _two_leg_target_weights(
        data["signal"],
        data["hedge_ratio"],
        price_x,
        price_y,
        spread_orientation=spread_orientation,
        hedge_ratio_kind=hedge_ratio_kind,
    )
    weight_x = target_weight_x.shift(1).fillna(0.0)
    weight_y = target_weight_y.shift(1).fillna(0.0)
    gross_exposure = weight_x.abs() + weight_y.abs()
    gross_return = lagged_two_leg_gross_return(
        price_x,
        price_y,
        target_weight_x,
        target_weight_y,
    )

    previous_weight_x = target_weight_x.shift(1).fillna(0.0)
    previous_weight_y = target_weight_y.shift(1).fillna(0.0)
    turnover_x, turnover_y, turnover = two_leg_turnover(
        target_weight_x,
        target_weight_y,
    )

    fee_cost = turnover_rate_cost(turnover, costs.taker_fee_bps)
    leg_slippage_columns = {"slippage_x_model_bps", "slippage_y_model_bps"}
    present_leg_slippage = leg_slippage_columns.intersection(data.columns)
    if present_leg_slippage and present_leg_slippage != leg_slippage_columns:
        raise ValueError("leg slippage requires both model columns")
    if present_leg_slippage:
        slippage_x = _series_or_default(data, "slippage_x_model_bps", costs.slippage_bps)
        slippage_y = _series_or_default(data, "slippage_y_model_bps", costs.slippage_bps)
        if bool((slippage_x.lt(0.0) | slippage_y.lt(0.0)).any()):
            raise ValueError("modeled leg slippage must be finite and nonnegative")
        slippage_cost = (turnover_x * slippage_x + turnover_y * slippage_y) / 10_000.0
    else:
        slippage_x = pd.Series(float(costs.slippage_bps), index=data.index)
        slippage_y = pd.Series(float(costs.slippage_bps), index=data.index)
        slippage_cost = turnover_rate_cost(turnover, costs.slippage_bps)
    execution_risk_cost = turnover_rate_cost(turnover, costs.execution_risk_bps)
    partial_fill_cost = expected_partial_fill_cost(
        turnover,
        probability=costs.partial_fill_probability,
        fill_fraction=costs.partial_fill_fraction,
        penalty_bps=costs.partial_fill_penalty_bps,
    )
    realized_columns = {"funding_x_realized_bps", "funding_y_realized_bps"}
    present_realized_columns = realized_columns.intersection(data.columns)
    if present_realized_columns and present_realized_columns != realized_columns:
        raise ValueError("realized funding requires both leg columns")
    if present_realized_columns:
        funding_x = _series_or_default(data, "funding_x_realized_bps", 0.0)
        funding_y = _series_or_default(data, "funding_y_realized_bps", 0.0)
        funding_divisor = 1.0
    else:
        funding_x = _series_or_default(data, "funding_x_bps", costs.funding_bps_per_day)
        funding_y = _series_or_default(data, "funding_y_bps", costs.funding_bps_per_day)
        funding_divisor = _bars_per_day(frame, interval, costs)
    if costs.normalized_funding_policy() == FundingPolicy.SIGNED_REALIZED:
        funding_cost = (weight_x * funding_x + weight_y * funding_y) / 10_000.0 / funding_divisor
    else:
        funding_cost = (
            weight_x.abs() * funding_x.abs() / 10_000.0 / funding_divisor
            + weight_y.abs() * funding_y.abs() / 10_000.0 / funding_divisor
        )
    if rebalance_policy is RebalancePolicy.TARGET_WEIGHTS_EVERY_BAR:
        rebalance_event = _rebalance_events(data["signal"], turnover)
        quantity_x_after = pd.Series(np.nan, index=data.index, dtype="float64")
        quantity_y_after = pd.Series(np.nan, index=data.index, dtype="float64")
        quantity_basis = "not_applicable_target_weight_proxy"
        turnover_basis = "absolute_change_in_two_leg_target_weights"
    else:
        fixed_path = _fixed_units_until_exit_path(
            signal=data["signal"],
            price_x=price_x,
            price_y=price_y,
            target_weight_x=target_weight_x,
            target_weight_y=target_weight_y,
            funding_x=funding_x,
            funding_y=funding_y,
            funding_divisor=funding_divisor,
            slippage_x=slippage_x,
            slippage_y=slippage_y,
            costs=costs,
        )
        weight_x = fixed_path["weight_x_before_return"]
        weight_y = fixed_path["weight_y_before_return"]
        gross_exposure = weight_x.abs() + weight_y.abs()
        gross_return = fixed_path["gross_return"]
        turnover_x = fixed_path["turnover_x"]
        turnover_y = fixed_path["turnover_y"]
        turnover = turnover_x + turnover_y
        fee_cost = fixed_path["fees"]
        slippage_cost = fixed_path["slippage"]
        funding_cost = fixed_path["funding"]
        execution_risk_cost = fixed_path["execution_risk"]
        partial_fill_cost = fixed_path["partial_fill"]
        rebalance_event = fixed_path["rebalance_event"]
        quantity_x_after = fixed_path["quantity_x_after_rebalance"]
        quantity_y_after = fixed_path["quantity_y_after_rebalance"]
        quantity_basis = "initial_capital_currency_units"
        turnover_basis = "absolute_change_in_fixed_leg_quantities_at_current_prices"
    component_costs = {
        "fees": fee_cost,
        "slippage": slippage_cost,
        "funding": funding_cost,
        "execution_risk": execution_risk_cost,
        "partial_fill": partial_fill_cost,
    }
    if rebalance_policy is RebalancePolicy.TARGET_WEIGHTS_EVERY_BAR:
        close_turnover = previous_weight_x.abs() + previous_weight_y.abs()
        open_turnover = target_weight_x.abs() + target_weight_y.abs()
        reversal = data["signal"].shift(1).fillna(0.0) * data["signal"] < 0.0
        close_fraction = pd.Series(0.5, index=data.index)
        denominator = (close_turnover + open_turnover).replace(0.0, np.nan)
        close_fraction.loc[reversal] = (close_turnover / denominator).fillna(0.5).loc[reversal]
    else:
        close_fraction = fixed_path["reversal_close_fraction"]
    timestamps = _timestamps(data)
    ledger = build_trade_ledger(
        data["signal"],
        gross_return,
        component_costs,
        timestamps=timestamps,
        reversal_close_fraction=close_fraction,
    )
    returns_x = price_x.pct_change().fillna(0.0)
    returns_y = price_y.pct_change().fillna(0.0)
    gross_leg_x_return = weight_x * returns_x
    gross_leg_y_return = weight_y * returns_y
    equity_after = compound_simple_returns(ledger.bar_ledger["net_return"])
    equity_before = equity_after.shift(1).fillna(1.0)
    bar_ledger = ledger.bar_ledger.copy()
    bar_ledger["price_x"] = price_x
    bar_ledger["price_y"] = price_y
    bar_ledger["return_x"] = returns_x
    bar_ledger["return_y"] = returns_y
    bar_ledger["target_weight_x"] = target_weight_x
    bar_ledger["target_weight_y"] = target_weight_y
    bar_ledger["held_weight_x"] = weight_x
    bar_ledger["held_weight_y"] = weight_y
    bar_ledger["gross_leg_x_return"] = gross_leg_x_return
    bar_ledger["gross_leg_y_return"] = gross_leg_y_return
    bar_ledger["turnover_x"] = turnover_x
    bar_ledger["turnover_y"] = turnover_y
    bar_ledger["turnover"] = turnover
    bar_ledger["rebalance_policy"] = rebalance_policy.value
    bar_ledger["rebalance_event"] = rebalance_event
    bar_ledger["spread_orientation"] = spread_orientation
    bar_ledger["hedge_ratio_kind"] = hedge_ratio_kind
    bar_ledger["quantity_x_after_rebalance"] = quantity_x_after
    bar_ledger["quantity_y_after_rebalance"] = quantity_y_after
    bar_ledger["quantity_basis"] = quantity_basis
    bar_ledger["gross_notional"] = gross_exposure
    bar_ledger["taker_fee_bps"] = float(costs.taker_fee_bps)
    bar_ledger["slippage_x_bps"] = slippage_x
    bar_ledger["slippage_y_bps"] = slippage_y
    bar_ledger["execution_risk_bps"] = float(costs.execution_risk_bps)
    bar_ledger["funding_x_bps"] = funding_x
    bar_ledger["funding_y_bps"] = funding_y
    bar_ledger["funding_divisor"] = float(funding_divisor)
    bar_ledger["partial_fill_probability"] = float(costs.partial_fill_probability)
    bar_ledger["partial_fill_fraction"] = float(costs.partial_fill_fraction)
    bar_ledger["partial_fill_penalty_bps"] = float(costs.partial_fill_penalty_bps)
    bar_ledger["equity_before"] = equity_before
    bar_ledger["equity_after"] = equity_after
    bar_ledger["return_unit"] = "simple_fraction_of_equity"
    bar_ledger["weight_unit"] = "fraction_of_gross_one_target"
    bar_ledger["cost_unit"] = "simple_fraction_of_equity"
    bar_ledger["turnover_charge_basis"] = turnover_basis
    bar_ledger["funding_charge_basis"] = "held_leg_weight_times_signed_rate_per_interval"
    bar_ledger["funding_policy"] = costs.normalized_funding_policy().value
    bar_ledger["interval"] = str(interval or _frame_interval(frame) or "")
    ledger = TradeLedgerResult(
        closed_trades=ledger.closed_trades,
        open_trades=ledger.open_trades,
        bar_ledger=bar_ledger,
        reconciliation_error=ledger.reconciliation_error,
    )
    result = _backtest_result(
        ledger,
        gross_return=gross_return,
        gross_exposure=gross_exposure,
        costs=costs,
        interval=interval or _frame_interval(frame),
        timestamps=timestamps,
    )
    return result, ledger


def backtest_two_leg_spread(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel | None = None,
    *,
    interval: object | None = None,
) -> BacktestResult:
    """Backtest normalized two-leg positions using hedge ratio, never beta."""

    result, _ = backtest_two_leg_spread_with_ledger(
        frame,
        signal,
        cost_model,
        interval=interval,
    )
    return result


def _backtest_result(
    ledger: TradeLedgerResult,
    *,
    gross_return: pd.Series,
    gross_exposure: pd.Series,
    costs: CostModel,
    interval: object | None,
    timestamps: pd.Series | pd.Index | None,
) -> BacktestResult:
    closed_returns = (
        pd.to_numeric(ledger.closed_trades["profit_after_cost"], errors="raise")
        if not ledger.closed_trades.empty
        else pd.Series(dtype="float64")
    )
    closed_pnls = (
        pd.to_numeric(ledger.closed_trades["net_pnl"], errors="raise")
        if not ledger.closed_trades.empty
        else pd.Series(dtype="float64")
    )
    if bool((~np.isfinite(closed_returns)).any()) or bool((~np.isfinite(closed_pnls)).any()):
        raise ValueError("closed trade returns and same-currency P&L must be finite")
    profit_factor_result = calculate_profit_factor(closed_pnls)
    profit_factor = (
        float(profit_factor_result.value)
        if profit_factor_result.value is not None
        else float("nan")
    )
    expectancy = float(closed_returns.mean()) if not closed_returns.empty else 0.0
    closed_trade_return_std = float(closed_returns.std(ddof=1)) if len(closed_returns) >= 2 else 0.0
    expectancy_lower_95 = (
        expectancy - 1.645 * closed_trade_return_std / np.sqrt(len(closed_returns))
        if len(closed_returns) >= 2
        else None
    )
    win_rate = float((closed_returns > 0.0).mean()) if not closed_returns.empty else 0.0
    net_return = ledger.bar_ledger["net_return"]
    equity = compound_simple_returns(net_return)
    sharpe = calculate_annualized_sharpe(net_return, interval=interval, timestamps=timestamps)
    return BacktestResult(
        trades=len(closed_returns),
        profit_factor=profit_factor,
        expectancy=expectancy,
        sharpe=sharpe.value,
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        gross_return=float((1.0 + gross_return).prod() - 1.0) if len(gross_return) else 0.0,
        total_fees=float(ledger.bar_ledger["fees"].sum()),
        total_slippage=float(ledger.bar_ledger["slippage"].sum()),
        total_funding=float(ledger.bar_ledger["funding"].sum()),
        total_execution_risk=float(ledger.bar_ledger["execution_risk"].sum()),
        total_partial_fill_cost=float(ledger.bar_ledger["partial_fill"].sum()),
        avg_gross_exposure=float(gross_exposure.mean()),
        open_trades=len(ledger.open_trades),
        interval=sharpe.interval or "",
        periods_per_year=sharpe.periods_per_year,
        sharpe_status=sharpe.status if sharpe.status == "valid" else f"blocked:{sharpe.reason}",
        funding_policy=costs.normalized_funding_policy().value,
        rebalance_policy=costs.normalized_rebalance_policy().value,
        reconciliation_error=ledger.reconciliation_error,
        closed_trade_return_std=closed_trade_return_std,
        expectancy_lower_95=float(expectancy_lower_95) if expectancy_lower_95 is not None else None,
        profit_factor_status=profit_factor_result.status,
        profit_factor_reason=(
            "" if profit_factor_result.status == "valid" else profit_factor_result.reason
        ),
    )


def _timestamps(frame: pd.DataFrame) -> pd.Series | pd.Index | None:
    for column in ("timestamp", "startedAt", "started_at", "time"):
        if column in frame.columns:
            return frame[column]
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame.index
    return None


def _frame_interval(frame: pd.DataFrame) -> object | None:
    if "interval" in frame.columns:
        values = frame["interval"].dropna().astype(str).unique()
        if len(values) == 1:
            return values[0]
    return frame.attrs.get("interval")


def _bars_per_day(frame: pd.DataFrame, interval: object | None, costs: CostModel) -> float:
    declared_interval = interval if interval is not None else _frame_interval(frame)
    timestamps = _timestamps(frame)
    annualization = resolve_annualization(interval=declared_interval, timestamps=timestamps)
    if annualization.status == "valid" and annualization.periods_per_year is not None:
        return annualization.periods_per_year / 365.0
    if declared_interval is not None or timestamps is not None:
        raise ValueError(f"funding clock is not valid: {annualization.reason}")
    # A declared cost-model divisor is allowed without a candle clock. This
    # fallback does not make Sharpe annualization authoritative.
    return float(costs.bars_per_day)


def _scalar_reversal_close_fraction(target: pd.Series) -> pd.Series:
    previous = target.shift(1).fillna(0.0)
    denominator = (previous.abs() + target.abs()).replace(0.0, np.nan)
    return (previous.abs() / denominator).fillna(0.5)


def _rebalance_events(signal: pd.Series, turnover: pd.Series) -> pd.Series:
    previous = signal.shift(1).fillna(0.0)
    previous_sign = np.sign(previous)
    current_sign = np.sign(signal)
    events = pd.Series("hold", index=signal.index, dtype="object")
    events.loc[(previous_sign == 0) & (current_sign != 0)] = "entry"
    events.loc[(previous_sign != 0) & (current_sign == 0)] = "exit"
    events.loc[(previous_sign != 0) & (current_sign != 0) & (previous_sign != current_sign)] = (
        "reversal"
    )
    target_update = turnover.gt(1e-15) & events.eq("hold")
    events.loc[target_update] = "target_update"
    return events



def _fixed_units_until_exit_path(
    *,
    signal: pd.Series,
    price_x: pd.Series,
    price_y: pd.Series,
    target_weight_x: pd.Series,
    target_weight_y: pd.Series,
    funding_x: pd.Series,
    funding_y: pd.Series,
    funding_divisor: float,
    slippage_x: pd.Series,
    slippage_y: pd.Series,
    costs: CostModel,
) -> pd.DataFrame:
    """Simulate fixed leg quantities with equity-scaled entry sizing.

    Quantities are selected only on entry or reversal and remain unchanged until
    exit. PnL, funding, turnover, and transaction costs are converted to
    fractions of prior equity on every bar so portfolio compounding reconciles.
    """

    rows: list[dict[str, object]] = []
    quantity_x = 0.0
    quantity_y = 0.0
    equity = 1.0
    previous_signal = 0.0
    funding_policy = costs.normalized_funding_policy()
    for row_number in range(len(signal)):
        equity_before = equity
        if not np.isfinite(equity_before) or equity_before <= 0.0:
            raise ValueError("fixed-unit equity must remain finite and positive")
        px = float(price_x.iloc[row_number])
        py = float(price_y.iloc[row_number])
        previous_px = px if row_number == 0 else float(price_x.iloc[row_number - 1])
        previous_py = py if row_number == 0 else float(price_y.iloc[row_number - 1])
        notional_x_before = quantity_x * previous_px
        notional_y_before = quantity_y * previous_py
        gross_pnl = quantity_x * (px - previous_px) + quantity_y * (py - previous_py)
        gross_return = gross_pnl / equity_before

        funding_rate_x = float(funding_x.iloc[row_number]) / 10_000.0 / funding_divisor
        funding_rate_y = float(funding_y.iloc[row_number]) / 10_000.0 / funding_divisor
        if funding_policy is FundingPolicy.SIGNED_REALIZED:
            funding_dollars = (
                notional_x_before * funding_rate_x
                + notional_y_before * funding_rate_y
            )
        else:
            funding_dollars = (
                abs(notional_x_before) * abs(funding_rate_x)
                + abs(notional_y_before) * abs(funding_rate_y)
            )
        funding_cost = funding_dollars / equity_before
        pre_trade_equity = equity_before + gross_pnl - funding_dollars
        if not np.isfinite(pre_trade_equity) or pre_trade_equity <= 0.0:
            raise ValueError("fixed-unit pre-trade equity must remain positive")

        current_signal = float(signal.iloc[row_number])
        previous_sign = int(np.sign(previous_signal))
        current_sign = int(np.sign(current_signal))
        if current_sign == 0:
            desired_quantity_x = 0.0
            desired_quantity_y = 0.0
        elif previous_sign == 0 or previous_sign != current_sign:
            desired_quantity_x = (
                float(target_weight_x.iloc[row_number]) * pre_trade_equity / px
            )
            desired_quantity_y = (
                float(target_weight_y.iloc[row_number]) * pre_trade_equity / py
            )
        else:
            desired_quantity_x = quantity_x
            desired_quantity_y = quantity_y

        turnover_x_dollars = abs(desired_quantity_x - quantity_x) * px
        turnover_y_dollars = abs(desired_quantity_y - quantity_y) * py
        turnover_x = turnover_x_dollars / equity_before
        turnover_y = turnover_y_dollars / equity_before
        turnover = turnover_x + turnover_y
        fee_cost = turnover * costs.taker_fee_bps / 10_000.0
        slippage_cost = (
            turnover_x * float(slippage_x.iloc[row_number])
            + turnover_y * float(slippage_y.iloc[row_number])
        ) / 10_000.0
        execution_risk_cost = turnover * costs.execution_risk_bps / 10_000.0
        partial_fill_cost = (
            turnover
            * costs.partial_fill_probability
            * (1.0 - costs.partial_fill_fraction)
            * costs.partial_fill_penalty_bps
            / 10_000.0
        )
        transaction_cost = (
            fee_cost + slippage_cost + execution_risk_cost + partial_fill_cost
        )
        net_return = gross_return - funding_cost - transaction_cost
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("fixed-unit bar net return must remain greater than -100%")
        equity = equity_before * (1.0 + net_return)

        close_turnover_dollars = abs(quantity_x) * px + abs(quantity_y) * py
        open_turnover_dollars = (
            abs(desired_quantity_x) * px + abs(desired_quantity_y) * py
        )
        reversal = previous_sign != 0 and current_sign != 0 and previous_sign != current_sign
        reversal_denominator = close_turnover_dollars + open_turnover_dollars
        reversal_close_fraction = (
            close_turnover_dollars / reversal_denominator
            if reversal and reversal_denominator > 0.0
            else 0.5
        )
        if previous_sign == 0 and current_sign != 0:
            event = "entry"
        elif previous_sign != 0 and current_sign == 0:
            event = "exit"
        elif reversal:
            event = "reversal"
        else:
            event = "hold"

        quantity_x = desired_quantity_x
        quantity_y = desired_quantity_y
        rows.append(
            {
                "gross_return": gross_return,
                "fees": fee_cost,
                "slippage": slippage_cost,
                "funding": funding_cost,
                "execution_risk": execution_risk_cost,
                "partial_fill": partial_fill_cost,
                "weight_x_before_return": notional_x_before / equity_before,
                "weight_y_before_return": notional_y_before / equity_before,
                "quantity_x_after_rebalance": quantity_x,
                "quantity_y_after_rebalance": quantity_y,
                "notional_weight_x_after_rebalance": quantity_x * px / equity,
                "notional_weight_y_after_rebalance": quantity_y * py / equity,
                "turnover_x": turnover_x,
                "turnover_y": turnover_y,
                "reversal_close_fraction": reversal_close_fraction,
                "rebalance_event": event,
            }
        )
        previous_signal = current_signal
    return pd.DataFrame(rows, index=signal.index)
