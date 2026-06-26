from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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

    def round_trip_cost(self) -> float:
        bps = 2 * (self.taker_fee_bps + self.slippage_bps + self.execution_risk_bps)
        return bps / 10_000.0

    def funding_per_bar(self) -> float:
        return (self.funding_bps_per_day / 10_000.0) / self.bars_per_day


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


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    drawdown = (peak - equity) / peak.replace(0, np.nan)
    return float(drawdown.fillna(0.0).max())


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 365 * 24) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=0))


def _series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float64")


def backtest_pair(frame: pd.DataFrame, signal: pd.Series, cost_model: CostModel | None = None) -> BacktestResult:
    """Backtest a hedged spread signal from spread returns and target position."""
    costs = cost_model or CostModel()
    data = frame.copy()
    data["signal"] = signal.reindex(data.index).fillna(0.0).astype(float)
    spread_return = data["spread"].astype(float).diff().fillna(0.0)
    position = data["signal"].shift(1).fillna(0.0)
    gross_return = position * spread_return
    turnover = data["signal"].diff().abs().fillna(data["signal"].abs())
    trading_cost = turnover * costs.round_trip_cost() / 2.0
    funding_cost = position.abs() * costs.funding_per_bar()
    net_return = gross_return - trading_cost - funding_cost
    equity = (1.0 + net_return).cumprod()

    trade_pnl = net_return.groupby((turnover > 0).cumsum()).sum()
    completed = trade_pnl[trade_pnl.index > 0]
    wins = completed[completed > 0]
    losses = completed[completed < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
    expectancy = float(completed.mean()) if len(completed) else 0.0
    win_rate = float((completed > 0).mean()) if len(completed) else 0.0

    return BacktestResult(
        trades=int(len(completed)),
        profit_factor=profit_factor,
        expectancy=expectancy,
        sharpe=annualized_sharpe(net_return),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        gross_return=float((1.0 + gross_return).prod() - 1.0) if len(gross_return) else 0.0,
        total_fees=float((turnover * costs.taker_fee_bps / 10_000.0).sum()),
        total_slippage=float((turnover * costs.slippage_bps / 10_000.0).sum()),
        total_funding=float(funding_cost.sum()),
        total_execution_risk=float((turnover * costs.execution_risk_bps / 10_000.0).sum()),
        avg_gross_exposure=float(position.abs().mean()),
    )


def backtest_two_leg_spread(
    frame: pd.DataFrame,
    signal: pd.Series,
    cost_model: CostModel | None = None,
) -> BacktestResult:
    """Backtest market-neutral two-leg spread positions.

    Required columns: `price_x`, `price_y`.
    Optional columns: `hedge_ratio`, `beta`, `funding_x_bps`, `funding_y_bps`.
    """

    costs = cost_model or CostModel()
    required = {"price_x", "price_y"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing two-leg price columns: {missing}")

    data = frame.copy()
    data["signal"] = signal.reindex(data.index).fillna(0.0).astype(float)
    price_x = pd.to_numeric(data["price_x"], errors="coerce").ffill()
    price_y = pd.to_numeric(data["price_y"], errors="coerce").ffill()
    returns_x = price_x.pct_change().fillna(0.0)
    returns_y = price_y.pct_change().fillna(0.0)
    hedge_ratio = _series_or_default(data, "hedge_ratio", 1.0)
    beta = _series_or_default(data, "beta", 1.0).replace(0, 1.0).abs()

    signal_position = data["signal"].shift(1).fillna(0.0)
    gross_scale = 1.0 + hedge_ratio.abs() * beta
    weight_y = signal_position / gross_scale
    weight_x = -signal_position * hedge_ratio * beta / gross_scale
    gross_exposure = weight_x.abs() + weight_y.abs()
    gross_return = weight_x * returns_x + weight_y * returns_y

    target_weight_y = data["signal"] / gross_scale
    target_weight_x = -data["signal"] * hedge_ratio * beta / gross_scale
    turnover_x = target_weight_x.diff().abs().fillna(target_weight_x.abs())
    turnover_y = target_weight_y.diff().abs().fillna(target_weight_y.abs())
    turnover = turnover_x + turnover_y

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
    funding_x = _series_or_default(data, "funding_x_bps", costs.funding_bps_per_day)
    funding_y = _series_or_default(data, "funding_y_bps", costs.funding_bps_per_day)
    funding_cost = (
        weight_x.abs() * funding_x.abs() / 10_000.0 / costs.bars_per_day
        + weight_y.abs() * funding_y.abs() / 10_000.0 / costs.bars_per_day
    )
    net_return = gross_return - fee_cost - slippage_cost - execution_risk_cost - partial_fill_cost - funding_cost
    equity = (1.0 + net_return).cumprod()

    trade_groups = (turnover > 0).cumsum()
    trade_pnl = net_return.groupby(trade_groups).sum()
    completed = trade_pnl[trade_pnl.index > 0]
    wins = completed[completed > 0]
    losses = completed[completed < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
    expectancy = float(completed.mean()) if len(completed) else 0.0
    win_rate = float((completed > 0).mean()) if len(completed) else 0.0

    return BacktestResult(
        trades=int(len(completed)),
        profit_factor=profit_factor,
        expectancy=expectancy,
        sharpe=annualized_sharpe(net_return),
        max_drawdown=max_drawdown(equity),
        win_rate=win_rate,
        total_return=float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        gross_return=float((1.0 + gross_return).prod() - 1.0) if len(gross_return) else 0.0,
        total_fees=float(fee_cost.sum()),
        total_slippage=float(slippage_cost.sum()),
        total_funding=float(funding_cost.sum()),
        total_execution_risk=float(execution_risk_cost.sum()),
        total_partial_fill_cost=float(partial_fill_cost.sum()),
        avg_gross_exposure=float(gross_exposure.mean()),
    )
