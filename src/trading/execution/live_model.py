"""实盘执行模型。

LiveExecutionModel 是唯一会触达交易所下单能力的领域执行实现。它只负责
把 OrderIntent 提交给交易所客户端，并把交易所回报转换成领域 Fill。
"""

from __future__ import annotations

import pandas as pd

from src.domain.models import Fill, MarketSlice, OrderIntent, PortfolioSnapshot
from src.trading.execution.order import Direction


class LiveExecutionModel:
    """真实交易所执行模型。"""

    def __init__(self, exchange_client) -> None:
        self.exchange_client = exchange_client

    async def execute(self, order_intent: OrderIntent, market: MarketSlice, portfolio: PortfolioSnapshot) -> Fill:
        """提交真实订单，并把交易所响应转成 Fill。"""
        response = self.exchange_client.create_order(
            symbol=order_intent.symbol,
            direction=self._direction(order_intent.side),
            order_type=order_intent.order_type,
            quantity=order_intent.quantity,
            price=order_intent.limit_price,
        )

        timestamp = self._timestamp(response, market)
        return Fill(
            fill_id=str(response.get("id", f"live:{order_intent.order_intent_id}")),
            order_intent_id=order_intent.order_intent_id,
            symbol=order_intent.symbol,
            side=order_intent.side,
            timestamp=timestamp,
            quantity=float(response.get("filled", order_intent.quantity) or 0),
            price=float(response.get("average", response.get("price", 0)) or 0),
            commission=self._commission(response),
            status=str(response.get("status", "filled")).lower(),
        )

    def _direction(self, side: str) -> Direction:
        """把领域 side 转成旧交易所客户端使用的 Direction。"""
        return Direction.BUY if side == "buy" else Direction.SELL

    def _commission(self, response) -> float:
        """兼容不同交易所响应里的 fee 表达。"""
        fee = response.get("fee", 0)
        if isinstance(fee, dict):
            return float(fee.get("cost", 0) or 0)
        return float(fee or 0)

    def _timestamp(self, response, market: MarketSlice):
        """优先使用交易所回报时间，缺失时退回行情时间。"""
        value = response.get("timestamp")
        if value is None:
            return market.timestamp
        timestamp = pd.to_datetime(value, unit="ms", utc=True)
        return timestamp.to_pydatetime()
