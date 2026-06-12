"""回测成交模型。

这里模拟订单在历史 K 线上的成交价格、滑点、手续费和可成交量。它只处理
“订单如何成交”，不直接修改组合现金和持仓；组合更新由 PortfolioBook 负责。
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Tuple

import pandas as pd

from src.domain.models import Fill, MarketSlice, OrderIntent, PortfolioSnapshot
from src.trading.execution.order import Direction


class BacktestExecutionModel:
    def __init__(
        self,
        historical_data: Dict[str, pd.DataFrame],
        commission: float,
        slippage: float,
        max_bar_volume_fraction: float = 0.1,
    ):
        self.historical_data = historical_data
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_bar_volume_fraction = float(max_bar_volume_fraction)

    async def execute(self, order_intent: OrderIntent, market: MarketSlice, portfolio: PortfolioSnapshot) -> Fill:
        """领域执行入口：只根据订单意图和行情生成成交，不修改组合账本。"""
        bar = market.bars_by_symbol.get(order_intent.symbol)
        if bar is None:
            raise ValueError(f"Missing market bar for {order_intent.symbol}")

        execution = self._simulate_fill(
            symbol=order_intent.symbol,
            side=order_intent.side,
            quantity=order_intent.quantity,
            high=bar.high,
            low=bar.low,
            volume=bar.volume,
        )
        if execution["filled_qty"] <= 0:
            raise ValueError(f"No liquidity available for {order_intent.symbol}")

        # partial 只表达成交量被流动性截断；账本更新仍由 PortfolioBook.apply_fill 负责。
        status = "filled" if execution["filled_qty"] == order_intent.quantity else "partial"
        return Fill(
            fill_id=f"fill:{order_intent.order_intent_id}",
            order_intent_id=order_intent.order_intent_id,
            symbol=order_intent.symbol,
            side=order_intent.side,
            timestamp=market.timestamp,
            quantity=execution["filled_qty"],
            price=execution["price"],
            commission=execution["commission"],
            slippage=self.slippage,
            status=status,
        )

    async def execute_orders(self, orders: List) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """逐笔模拟成交，并返回成交记录和扣减成交量后的历史数据。"""
        updated_data = self.historical_data.copy()
        executed = []

        for order in orders:
            if order.symbol not in updated_data:
                continue

            symbol_data = updated_data[order.symbol]
            # volume 可能从 parquet 读出为整数或字符串，先转 float 便于扣减。
            if "volume" in symbol_data.columns:
                updated_data[order.symbol]["volume"] = symbol_data["volume"].astype(float)
                symbol_data = updated_data[order.symbol]

            # 当前回测模型使用订单时间之后的第一根 K 线作为成交 K 线。
            matching_bars = symbol_data[symbol_data["timestamp"] >= order.timestamp]
            if matching_bars.empty:
                continue

            execution_bar = matching_bars.iloc[0]
            execution = self._simulate_fill(
                symbol=order.symbol,
                side=order.direction.value,
                quantity=order.quantity,
                high=execution_bar["high"],
                low=execution_bar["low"],
                volume=execution_bar["volume"],
            )
            if execution["filled_qty"] <= 0:
                continue

            updated_data[order.symbol].loc[execution_bar.name, "volume"] -= execution["filled_qty"]

            executed.append({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "timestamp": updated_data[order.symbol].loc[execution_bar.name, "timestamp"],
                "execution_time": datetime.datetime.now(),
                "direction": order.direction,
                "filled_qty": execution["filled_qty"],
                "unfilled_qty": order.quantity - execution["filled_qty"],
                "price": execution["price"],
                "commission": execution["commission"],
            })

        return pd.DataFrame(executed), updated_data

    def _simulate_fill(self, symbol: str, side: str, quantity: float, high: float, low: float, volume: float) -> Dict[str, float]:
        """统一计算保守成交价、可成交量和手续费。"""
        max_volume = float(volume) * self.max_bar_volume_fraction
        filled_qty = min(float(quantity), max_volume)
        if filled_qty <= 0:
            return {"filled_qty": 0.0, "price": 0.0, "commission": 0.0}

        if self._is_buy(side):
            price = float(high) * (1 + self.slippage)
        else:
            price = float(low) * (1 - self.slippage)

        return {
            "filled_qty": filled_qty,
            "price": price,
            "commission": price * filled_qty * self.commission,
        }

    def _is_buy(self, side: str) -> bool:
        """兼容领域 side 字符串和旧 Direction 枚举值。"""
        if isinstance(side, Direction):
            return side == Direction.BUY
        return str(side).lower() in {"buy", Direction.BUY.value}
