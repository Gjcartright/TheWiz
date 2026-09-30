"""Closed-trade lifecycle accounting for Math V2 backtests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


COST_COMPONENTS = (
    "fees",
    "slippage",
    "funding",
    "execution_risk",
    "partial_fill",
)


@dataclass(frozen=True)
class TradeLedgerResult:
    closed_trades: pd.DataFrame
    open_trades: pd.DataFrame
    bar_ledger: pd.DataFrame
    reconciliation_error: float


def build_trade_ledger(
    target_position: pd.Series,
    gross_return: pd.Series,
    costs: Mapping[str, pd.Series],
    *,
    timestamps: pd.Series | pd.Index | None = None,
    reversal_close_fraction: pd.Series | None = None,
) -> TradeLedgerResult:
    """Build a lifecycle ledger while preserving exact portfolio compounding.

    A reversal closes the old trade and opens the new trade at the same row.
    The old trade receives the holding return, all signed funding, and the
    closing share of transition costs; the new trade receives an exact residual
    factor. All supplied series must share one unique index in the same order.
    Missing/unknown inputs are rejected, not converted to flat or free trading.
    Signed fees may represent realized rebates; modeled adverse costs may not.
    Omitting a recognized cost component explicitly selects a frictionless or
    model-zero component at this pure API; it is not observed zero-cost evidence.
    Equity starts at one currency unit. ``net_pnl`` uses that common currency
    basis, while ``profit_after_cost`` remains a trade-entry-normalized return.
    """

    index = target_position.index
    if not index.is_unique or bool(index.isna().any()):
        raise ValueError("ledger index must be unique and complete")
    if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
        raise ValueError("datetime ledger index must be strictly increasing")

    def aligned_finite(values: pd.Series, name: str) -> pd.Series:
        if not values.index.equals(index):
            raise ValueError(f"{name} must share the identical ledger index and order")
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        if bool((~np.isfinite(numeric)).any()):
            raise ValueError(f"{name} must be complete and finite")
        return numeric

    target = aligned_finite(target_position, "target_position")
    gross = aligned_finite(gross_return, "gross_return")
    unknown_costs = set(costs).difference(COST_COMPONENTS)
    if unknown_costs:
        raise ValueError(f"unknown cost components: {sorted(unknown_costs)}")
    component_frame = pd.DataFrame(index=index)
    for name in COST_COMPONENTS:
        component_frame[name] = aligned_finite(
            costs.get(name, pd.Series(0.0, index=index)), name
        )
        if name in {"slippage", "execution_risk", "partial_fill"} and bool(component_frame[name].lt(0.0).any()):
            raise ValueError(f"{name} must be nonnegative; adverse costs are not rebates")
    net = gross - component_frame.sum(axis=1)
    if bool((~np.isfinite(net) | net.le(-1.0)).any()):
        raise ValueError("bar net return must be finite and remain greater than -100%")

    held = target.shift(1).fillna(0.0).ne(0.0)
    if bool((~held & (gross.ne(0.0) | component_frame["funding"].ne(0.0))).any()):
        raise ValueError("holding return and funding require a previous held position")
    if bool((~held & target.eq(0.0) & component_frame.ne(0.0).any(axis=1)).any()):
        raise ValueError("costs require an opening or held trade")

    if timestamps is None:
        timestamp_values = pd.Series(index.astype(str), index=index)
    else:
        if isinstance(timestamps, pd.Series) and not timestamps.index.equals(index):
            raise ValueError("timestamps must share the identical ledger index and order")
        if len(timestamps) != len(index):
            raise ValueError("timestamps must match the ledger length")
        parsed_timestamps = pd.DatetimeIndex(pd.to_datetime(list(timestamps), format="mixed", utc=True, errors="coerce"))
        if parsed_timestamps.hasnans or not parsed_timestamps.is_unique or not parsed_timestamps.is_monotonic_increasing:
            raise ValueError("timestamps must be complete, unique, and strictly increasing")
        if isinstance(index, pd.DatetimeIndex) and not parsed_timestamps.equals(pd.DatetimeIndex(pd.to_datetime(index, utc=True))):
            raise ValueError("timestamps must match the datetime ledger index")
        timestamp_values = pd.Series(list(timestamps), index=index).astype(str)
    reversal_fraction = (
        aligned_finite(reversal_close_fraction, "reversal_close_fraction")
        if reversal_close_fraction is not None
        else pd.Series(0.5, index=index)
    )
    if bool((~reversal_fraction.between(0.0, 1.0)).any()):
        raise ValueError("reversal_close_fraction must be within [0, 1]")

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        equity_after = (1.0 + net).cumprod()
    if bool((~np.isfinite(equity_after) | equity_after.le(0.0)).any()):
        raise ValueError("compounded equity must remain finite and positive")
    equity_before = equity_after.shift(1).fillna(1.0)

    trades: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    next_trade_id = 1

    def open_trade(row_number: int, side: float, entry_equity: float) -> dict[str, object]:
        nonlocal next_trade_id
        trade = {
            "trade_id": next_trade_id,
            "side": int(np.sign(side)),
            "entry_index": row_number,
            "entry_timestamp": timestamp_values.iloc[row_number],
            "exit_index": None,
            "exit_timestamp": None,
            "exit_reason": "",
            "bars": 0,
            "gross_factor": 1.0,
            "net_factor": 1.0,
            "entry_equity": entry_equity,
            "pnl_unit": "initial_capital_currency_units",
            **{f"total_{name}": 0.0 for name in COST_COMPONENTS},
        }
        next_trade_id += 1
        return trade

    def apply_factor(
        trade: dict[str, object],
        *,
        gross_value: float,
        net_factor: float,
        component_values: Mapping[str, float],
        count_bar: bool,
    ) -> None:
        trade["gross_factor"] = float(trade["gross_factor"]) * (1.0 + gross_value)
        trade["net_factor"] = float(trade["net_factor"]) * net_factor
        if not np.isfinite(float(trade["net_factor"])) or float(trade["net_factor"]) <= 0.0:
            raise ValueError("trade net factor must remain finite and positive")
        if count_bar:
            trade["bars"] = int(trade["bars"]) + 1
        for name, value in component_values.items():
            key = f"total_{name}"
            trade[key] = float(trade[key]) + float(value)

    def close_trade(trade: dict[str, object], row_number: int, reason: str) -> None:
        trade["exit_index"] = row_number
        trade["exit_timestamp"] = timestamp_values.iloc[row_number]
        trade["exit_reason"] = reason
        trades.append(_finalize_trade(trade, is_closed=True))

    previous = 0.0
    for row_number in range(len(index)):
        new = float(target.iloc[row_number])
        gross_value = float(gross.iloc[row_number])
        component_values = {name: float(component_frame[name].iloc[row_number]) for name in COST_COMPONENTS}
        total_net_factor = 1.0 + float(net.iloc[row_number])
        starting_equity = float(equity_before.iloc[row_number])
        previous_sign = int(np.sign(previous))
        new_sign = int(np.sign(new))

        if previous_sign == 0 and new_sign != 0:
            current = open_trade(row_number, new, starting_equity)
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=total_net_factor,
                component_values=component_values,
                count_bar=False,
            )
        elif previous_sign != 0 and new_sign == 0:
            if current is None:
                raise ValueError("held position has no open trade")
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=total_net_factor,
                component_values=component_values,
                count_bar=True,
            )
            close_trade(current, row_number, "signal_exit")
            current = None
        elif previous_sign != 0 and new_sign != 0 and previous_sign != new_sign:
            if current is None:
                raise ValueError("held position has no open trade")
            close_fraction = float(reversal_fraction.iloc[row_number])
            old_components = {
                name: value if name == "funding" else value * close_fraction
                for name, value in component_values.items()
            }
            old_net = gross_value - sum(old_components.values())
            old_factor = 1.0 + old_net
            if not np.isfinite(old_factor) or old_factor <= 0.0:
                raise ValueError("reversal closing factor must remain finite and positive")
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=old_factor,
                component_values=old_components,
                count_bar=True,
            )
            close_trade(current, row_number, "signal_reversal")
            opening_equity = starting_equity * old_factor
            if not np.isfinite(opening_equity) or opening_equity <= 0.0:
                raise ValueError("reversal opening equity must remain finite and positive")
            current = open_trade(row_number, new, opening_equity)
            new_components = {name: value - old_components[name] for name, value in component_values.items()}
            apply_factor(
                current,
                gross_value=0.0,
                net_factor=total_net_factor / old_factor,
                component_values=new_components,
                count_bar=False,
            )
        elif previous_sign != 0 and new_sign == previous_sign:
            if current is None:
                raise ValueError("held position has no open trade")
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=total_net_factor,
                component_values=component_values,
                count_bar=True,
            )
        previous = new

    closed = pd.DataFrame(trades)
    open_rows = pd.DataFrame([_finalize_trade(current, is_closed=False)]) if current is not None else pd.DataFrame()
    portfolio_factor = float((1.0 + net).prod())
    trade_factor = 1.0
    if not closed.empty:
        trade_factor *= float((1.0 + closed["profit_after_cost"]).prod())
    if not open_rows.empty:
        trade_factor *= float((1.0 + open_rows["profit_after_cost"]).prod())
    reconciliation_error = abs(portfolio_factor - trade_factor)
    bar_ledger = pd.DataFrame(
        {
            "timestamp": timestamp_values,
            "target_position": target,
            "gross_return": gross,
            **{name: component_frame[name] for name in COST_COMPONENTS},
            "net_return": net,
            "equity_before": equity_before,
            "equity_after": equity_after,
            "net_pnl": equity_after - equity_before,
            "pnl_unit": "initial_capital_currency_units",
        },
        index=index,
    )
    return TradeLedgerResult(closed, open_rows, bar_ledger, reconciliation_error)


def _finalize_trade(trade: dict[str, object], *, is_closed: bool) -> dict[str, object]:
    ending_equity = float(trade["entry_equity"]) * float(trade["net_factor"])
    if not np.isfinite(ending_equity) or ending_equity <= 0.0:
        raise ValueError("trade ending equity must remain finite and positive")
    return {
        **trade,
        "is_closed": is_closed,
        "gross_return": float(trade["gross_factor"]) - 1.0,
        "profit_after_cost": float(trade["net_factor"]) - 1.0,
        "ending_equity": ending_equity,
        "net_pnl": ending_equity - float(trade["entry_equity"]),
    }
