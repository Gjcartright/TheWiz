from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from quant_platform.economic_contract import normalized_two_leg_weights
from quant_platform.performance_math import MATH_VERSION, calculate_annualized_sharpe
from quant_platform.trade_ledger import TradeLedgerResult, build_trade_ledger


class FundingPolicy(StrEnum):
    SIGNED_REALIZED = "signed_realized"
    CONSERVATIVE_ABSOLUTE_DRAG = "conservative_absolute_drag"


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

    def __post_init__(self) -> None:
        for field_name in (
            "taker_fee_bps",
            "slippage_bps",
            "execution_risk_bps",
            "partial_fill_penalty_bps",
        ):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and nonnegative")
        if not np.isfinite(float(self.funding_bps_per_day)):
            raise ValueError("funding_bps_per_day must be finite")
        if int(self.bars_per_day) <= 0:
            raise ValueError("bars_per_day must be positive")
        for field_name in ("partial_fill_probability", "partial_fill_fraction"):
            value = float(getattr(self, field_name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [0, 1]")
        self.normalized_funding_policy()

    def round_trip_cost(self) -> float:
        bps = 2 * (self.taker_fee_bps + self.slippage_bps + self.execution_risk_bps)
        return bps / 10_000.0

    def funding_per_bar(self, *, bars_per_day: float | None = None) -> float:
        divisor = float(bars_per_day or self.bars_per_day)
        if divisor <= 0.0:
            raise ValueError("bars_per_day must be positive")
        return (self.funding_bps_per_day / 10_000.0) / divisor

    def normalized_funding_policy(self) -> FundingPolicy:
        try:
            return FundingPolicy(self.funding_policy)
        except ValueError as exc:
            raise ValueError(f"unsupported funding policy: {self.funding_policy}") from exc


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
    reconciliation_error: float = 0.0
    closed_trade_return_std: float = 0.0
    expectancy_lower_95: float | None = None
    math_version: str = MATH_VERSION


def max_drawdown(equity: pd.Series) -> float:
    values = pd.to_numeric(equity, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        return 0.0
    # The first observed equity value is already the result of the first bar.
    # Prepend the initial unit of capital or an immediate loss is understated.
    path = np.concatenate(([1.0], values))
    peaks = np.maximum.accumulate(path)
    drawdown = np.divide(peaks - path, peaks, out=np.zeros_like(path), where=peaks != 0.0)
    return float(np.nanmax(drawdown))


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
        values = pd.to_numeric(frame[column], errors="coerce")
        if bool((~np.isfinite(values)).any()):
            raise ValueError(f"{column} contains missing or nonfinite values")
        return values.astype("float64")
    return pd.Series(default, index=frame.index, dtype="float64")


def backtest_pair(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel | None = None,
    *,
    interval: object | None = None,
) -> BacktestResult:
    """Backtest a spread signal with explicit lifecycle and cost ledgers."""

    costs = cost_model or CostModel()
    data = frame.copy()
    data["signal"] = signal.reindex(data.index).fillna(0.0).astype(float)
    spread_return = pd.to_numeric(data["spread"], errors="coerce").diff().fillna(0.0)
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
    required = {"price_x", "price_y", "hedge_ratio"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing two-leg price columns: {missing}")

    data = frame.copy()
    data["signal"] = signal.reindex(data.index).fillna(0.0).astype(float)
    price_x = pd.to_numeric(data["price_x"], errors="coerce")
    price_y = pd.to_numeric(data["price_y"], errors="coerce")
    invalid_prices = (
        (~np.isfinite(price_x)) | (~np.isfinite(price_y)) | price_x.le(0.0) | price_y.le(0.0)
    )
    if bool(invalid_prices.any()):
        raise ValueError("two-leg prices must be complete, finite, and positive")
    returns_x = price_x.pct_change().fillna(0.0)
    returns_y = price_y.pct_change().fillna(0.0)
    hedge_ratio = _series_or_default(data, "hedge_ratio", float("nan"))
    if bool(hedge_ratio.le(0.0).any()):
        raise ValueError("hedge_ratio must be finite and positive")

    target_weight_x, target_weight_y = normalized_two_leg_weights(data["signal"], hedge_ratio)
    weight_x = target_weight_x.shift(1).fillna(0.0)
    weight_y = target_weight_y.shift(1).fillna(0.0)
    gross_exposure = weight_x.abs() + weight_y.abs()
    gross_return = weight_x * returns_x + weight_y * returns_y

    previous_weight_x = target_weight_x.shift(1).fillna(0.0)
    previous_weight_y = target_weight_y.shift(1).fillna(0.0)
    turnover_x = (target_weight_x - previous_weight_x).abs()
    turnover_y = (target_weight_y - previous_weight_y).abs()
    turnover = turnover_x + turnover_y

    fee_cost = turnover * costs.taker_fee_bps / 10_000.0
    leg_slippage_columns = {"slippage_x_model_bps", "slippage_y_model_bps"}
    present_leg_slippage = leg_slippage_columns.intersection(data.columns)
    if present_leg_slippage and present_leg_slippage != leg_slippage_columns:
        raise ValueError("leg slippage requires both model columns")
    if present_leg_slippage:
        slippage_x = _series_or_default(data, "slippage_x_model_bps", costs.slippage_bps)
        slippage_y = _series_or_default(data, "slippage_y_model_bps", costs.slippage_bps)
        slippage_cost = (turnover_x * slippage_x + turnover_y * slippage_y) / 10_000.0
    else:
        slippage_cost = turnover * costs.slippage_bps / 10_000.0
    execution_risk_cost = turnover * costs.execution_risk_bps / 10_000.0
    partial_fill_cost = (
        turnover
        * costs.partial_fill_probability
        * (1.0 - costs.partial_fill_fraction)
        * costs.partial_fill_penalty_bps
        / 10_000.0
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
    component_costs = {
        "fees": fee_cost,
        "slippage": slippage_cost,
        "funding": funding_cost,
        "execution_risk": execution_risk_cost,
        "partial_fill": partial_fill_cost,
    }
    close_turnover = previous_weight_x.abs() + previous_weight_y.abs()
    open_turnover = target_weight_x.abs() + target_weight_y.abs()
    reversal = data["signal"].shift(1).fillna(0.0) * data["signal"] < 0.0
    close_fraction = pd.Series(0.5, index=data.index)
    denominator = (close_turnover + open_turnover).replace(0.0, np.nan)
    close_fraction.loc[reversal] = (close_turnover / denominator).fillna(0.5).loc[reversal]
    timestamps = _timestamps(data)
    ledger = build_trade_ledger(
        data["signal"],
        gross_return,
        component_costs,
        timestamps=timestamps,
        reversal_close_fraction=close_fraction,
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
        pd.to_numeric(ledger.closed_trades["profit_after_cost"], errors="coerce").dropna()
        if not ledger.closed_trades.empty
        else pd.Series(dtype="float64")
    )
    wins = closed_returns[closed_returns > 0.0]
    losses = closed_returns[closed_returns < 0.0]
    if not losses.empty:
        profit_factor = float(wins.sum() / abs(losses.sum()))
    elif not wins.empty:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    expectancy = float(closed_returns.mean()) if not closed_returns.empty else 0.0
    closed_trade_return_std = float(closed_returns.std(ddof=1)) if len(closed_returns) >= 2 else 0.0
    expectancy_lower_95 = (
        expectancy - 1.645 * closed_trade_return_std / np.sqrt(len(closed_returns))
        if len(closed_returns) >= 2
        else None
    )
    win_rate = float((closed_returns > 0.0).mean()) if not closed_returns.empty else 0.0
    net_return = ledger.bar_ledger["net_return"]
    equity = (1.0 + net_return).cumprod()
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
        reconciliation_error=ledger.reconciliation_error,
        closed_trade_return_std=closed_trade_return_std,
        expectancy_lower_95=float(expectancy_lower_95) if expectancy_lower_95 is not None else None,
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
    sharpe = calculate_annualized_sharpe(
        pd.Series([0.0, 0.0]),
        interval=interval or _frame_interval(frame),
        timestamps=_timestamps(frame),
    )
    if sharpe.periods_per_year is not None:
        return sharpe.periods_per_year / 365.0
    return float(costs.bars_per_day)


def _scalar_reversal_close_fraction(target: pd.Series) -> pd.Series:
    previous = target.shift(1).fillna(0.0)
    denominator = (previous.abs() + target.abs()).replace(0.0, np.nan)
    return (previous.abs() / denominator).fillna(0.5)
