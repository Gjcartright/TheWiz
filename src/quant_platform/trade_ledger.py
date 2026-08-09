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
    The old trade receives the holding return and closing-cost share; the new
    trade receives an exact residual factor so the two trade factors reconcile
    to the bar-level portfolio factor.
    """

    index = target_position.index
    target = pd.to_numeric(target_position, errors="coerce").fillna(0.0).astype(float)
    gross = pd.to_numeric(gross_return.reindex(index), errors="coerce").fillna(0.0)
    component_frame = pd.DataFrame(index=index)
    for name in COST_COMPONENTS:
        component_frame[name] = pd.to_numeric(
            costs.get(name, pd.Series(0.0, index=index)).reindex(index), errors="coerce"
        ).fillna(0.0)
    net = gross - component_frame.sum(axis=1)
    if (net <= -1.0).any():
        raise ValueError("bar net return must remain greater than -100%")

    if timestamps is None:
        timestamp_values = pd.Series(index.astype(str), index=index)
    else:
        timestamp_values = pd.Series(list(timestamps), index=index).astype(str)
    reversal_fraction = (
        pd.to_numeric(reversal_close_fraction.reindex(index), errors="coerce").fillna(0.5)
        if reversal_close_fraction is not None
        else pd.Series(0.5, index=index)
    ).clip(0.0, 1.0)

    trades: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    next_trade_id = 1

    def open_trade(row_number: int, side: float) -> dict[str, object]:
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
        previous_sign = int(np.sign(previous))
        new_sign = int(np.sign(new))

        if previous_sign == 0 and new_sign != 0:
            current = open_trade(row_number, new)
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=total_net_factor,
                component_values=component_values,
                count_bar=False,
            )
        elif previous_sign != 0 and new_sign == 0:
            if current is None:
                current = open_trade(max(0, row_number - 1), previous)
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
                current = open_trade(max(0, row_number - 1), previous)
            close_fraction = float(reversal_fraction.iloc[row_number])
            old_components = {name: value * close_fraction for name, value in component_values.items()}
            old_net = gross_value - sum(old_components.values())
            old_factor = 1.0 + old_net
            if old_factor <= 0.0:
                raise ValueError("reversal closing factor must remain positive")
            apply_factor(
                current,
                gross_value=gross_value,
                net_factor=old_factor,
                component_values=old_components,
                count_bar=True,
            )
            close_trade(current, row_number, "signal_reversal")
            current = open_trade(row_number, new)
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
                current = open_trade(max(0, row_number - 1), previous)
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
        },
        index=index,
    )
    return TradeLedgerResult(closed, open_rows, bar_ledger, reconciliation_error)


def _finalize_trade(trade: dict[str, object], *, is_closed: bool) -> dict[str, object]:
    return {
        **trade,
        "is_closed": is_closed,
        "gross_return": float(trade["gross_factor"]) - 1.0,
        "profit_after_cost": float(trade["net_factor"]) - 1.0,
    }
