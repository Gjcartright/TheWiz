from __future__ import annotations

from itertools import count

from quant_platform.execution import DydxNetworkConfig, FillReport, OrderIntent


class ImmediateSubmissionDydxOrderAdapter:
    """Simple exchange-side stub that marks submission attempts as accepted."""

    exchange_submission_capable = True
    record_only = False

    def __init__(self) -> None:
        self._ids = count(1)

    def place_order(self, intent: OrderIntent, config: DydxNetworkConfig) -> FillReport:
        order_id = next(self._ids)
        return FillReport(
            order_id=f"immediate-{order_id}",
            market=intent.market,
            side=intent.side,
            size=intent.size,
            avg_price=float(intent.limit_price or 0.0),
            fee=0.0,
            slippage_bps=0.0,
            status="paper_submitted",
        )
