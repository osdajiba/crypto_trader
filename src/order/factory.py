"""领域订单意图工厂。"""

from __future__ import annotations

from src.domain.models import OrderIntent, StrategySignal
from src.order.models import PositionSizingDecision


class OrderFactory:
    """把策略信号和 sizing 结果转换为 OrderIntent。"""

    def __init__(self, order_type: str = "market") -> None:
        self.order_type = order_type

    def create(
        self,
        signal: StrategySignal,
        sizing_decision: PositionSizingDecision,
        limit_price: float | None = None,
    ) -> OrderIntent:
        return OrderIntent(
            order_intent_id=f"intent:{signal.signal_id}",
            symbol=signal.symbol,
            side=signal.side,
            quantity=sizing_decision.target_quantity,
            order_type=self.order_type,
            created_at=signal.timestamp,
            limit_price=limit_price,
            source_signal_id=signal.signal_id,
        )
