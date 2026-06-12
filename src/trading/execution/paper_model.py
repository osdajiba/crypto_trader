"""模拟盘执行模型。

PaperExecutionModel 不触达交易所，也不修改组合账本；它只用最新行情生成
模拟成交，组合现金和持仓仍由 PortfolioBook 统一更新。
"""

from __future__ import annotations

from src.domain.models import Fill, MarketSlice, OrderIntent, PortfolioSnapshot


class PaperExecutionModel:
    """基于当前 close 价格模拟成交的领域执行模型。"""

    def __init__(self, commission: float, slippage: float) -> None:
        self.commission = float(commission)
        self.slippage = float(slippage)

    async def execute(self, order_intent: OrderIntent, market: MarketSlice, portfolio: PortfolioSnapshot) -> Fill:
        """返回模拟成交，不会产生真实订单。"""
        bar = market.bars_by_symbol.get(order_intent.symbol)
        if bar is None:
            raise ValueError(f"Missing market bar for {order_intent.symbol}")

        if order_intent.side == "buy":
            price = bar.close * (1 + self.slippage)
        else:
            price = bar.close * (1 - self.slippage)

        return Fill(
            fill_id=f"paper:{order_intent.order_intent_id}",
            order_intent_id=order_intent.order_intent_id,
            symbol=order_intent.symbol,
            side=order_intent.side,
            timestamp=market.timestamp,
            quantity=order_intent.quantity,
            price=price,
            commission=price * order_intent.quantity * self.commission,
            slippage=self.slippage,
            status="filled",
        )
