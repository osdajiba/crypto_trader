"""交易领域的组合账本。

PortfolioBook 是当前回测路径的资金和持仓事实来源。执行引擎只返回成交，
真正的现金、仓位、净值和回撤更新都在这里完成，避免分散在 mode 或策略里。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.domain.models import PortfolioSnapshot


@dataclass(frozen=True)
class EquityPoint:
    """权益曲线中的一个时间点。"""
    timestamp: datetime
    equity: float


class PortfolioBook:
    """维护现金、持仓、成交记录、权益曲线和最大回撤。"""

    def __init__(self, initial_cash: float):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.positions: Dict[str, float] = {}
        self.trades: List[Dict[str, Any]] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.market_prices: Dict[str, float] = {}
        self.current_equity = float(initial_cash)
        self.peak_equity = float(initial_cash)
        self.max_drawdown = 0.0

    def update_market_price(self, symbol: str, price: float) -> None:
        """更新最新市场价格，供持仓市值和权益计算使用。"""
        self.market_prices[symbol] = float(price)

    def apply_fill(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        commission: float,
        action: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """应用一笔成交。

        返回 None 表示成交不合法，例如现金不足买入、空仓卖出或卖出数量超过持仓。
        这样执行层和 pipeline 可以保持简单，由账本统一守住资金和仓位约束。
        """
        side_value = side.lower()
        quantity = float(quantity)
        price = float(price)
        commission = float(commission)

        if side_value == "buy":
            total_cost = (price * quantity) + commission
            if total_cost > self.cash:
                return None
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
        elif side_value in {"sell", "short"}:
            current_position = self.positions.get(symbol, 0.0)
            if side_value == "sell" and current_position < quantity:
                return None
            self.positions[symbol] = current_position - quantity
            if self.positions[symbol] <= 0:
                del self.positions[symbol]
            self.cash += (price * quantity) - commission
        else:
            return None

        trade = {
            "timestamp": timestamp,
            "symbol": symbol,
            "action": action if action is not None else side_value,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "cash_after": self.cash,
        }
        self.trades.append(trade)
        return trade

    def calculate_equity(self) -> float:
        """按现金加持仓市值计算当前净值。"""
        equity = self.cash
        for symbol, quantity in self.positions.items():
            if symbol in self.market_prices:
                equity += quantity * self.market_prices[symbol]
        return equity

    def record_equity(self, timestamp: datetime) -> EquityPoint:
        """记录权益曲线并同步更新峰值权益和最大回撤。"""
        equity = self.calculate_equity()
        self.current_equity = equity
        self.equity_curve.append({"timestamp": timestamp, "equity": equity})

        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)

        return EquityPoint(timestamp=timestamp, equity=equity)

    def snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """生成组合快照，供风控、报告或后续领域接口读取。"""
        equity = self.calculate_equity()
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=dict(self.positions),
            market_prices=dict(self.market_prices),
            equity=equity,
        )
