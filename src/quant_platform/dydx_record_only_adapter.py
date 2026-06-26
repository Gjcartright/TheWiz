from __future__ import annotations

from itertools import count

from quant_platform.execution import DydxNetworkConfig, FillReport, OrderIntent


class RecordOnlyDydxOrderAdapter:
    """Order adapter for local paper plumbing tests.

    This adapter satisfies the dYdX order-client contract while making no
    network calls. It records the order intent as a non-submitted fill so the
    journal and learning pipeline can be tested without faking exchange fills.
    """

    exchange_submission_capable = False
    record_only = True

    def __init__(self) -> None:
        self._ids = count(1)
        self.orders: list[dict[str, object]] = []

    def place_order(self, intent: OrderIntent, config: DydxNetworkConfig) -> FillReport:
        order_number = next(self._ids)
        self.orders.append(
            {
                "order_number": order_number,
                "market": intent.market,
                "side": intent.side,
                "size": intent.size,
                "limit_price": intent.limit_price,
                "reduce_only": intent.reduce_only,
                "mode": config.mode.value,
                "submit_orders": config.submit_orders,
            }
        )
        return FillReport(
            order_id=f"record-only-{order_number}",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_recorded_not_submitted",
        )
