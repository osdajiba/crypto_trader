"""应用层报告收集器。

Reporter 是领域 pipeline 的事件落点，记录成交和组合快照，并生成与旧
报告字段兼容的摘要。文件写入仍由 TradingReportUseCase 统一处理。
"""

from __future__ import annotations

from src.domain.models import Fill, PortfolioSnapshot


class InMemoryReporter:
    """内存中的领域事件收集器。"""

    def __init__(self) -> None:
        self.fills: list[Fill] = []
        self.snapshots: list[PortfolioSnapshot] = []

    async def record_fill(self, fill: Fill) -> None:
        """记录一笔领域成交。"""
        self.fills.append(fill)

    async def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        """记录一个组合快照。"""
        self.snapshots.append(snapshot)

    def generate_report(self, initial_capital: float) -> dict:
        """根据已记录的领域事件生成兼容报告字段。"""
        initial_capital = float(initial_capital)
        final_snapshot = self.snapshots[-1] if self.snapshots else None
        final_equity = float(final_snapshot.equity) if final_snapshot else initial_capital
        remaining_cash = float(final_snapshot.cash) if final_snapshot else initial_capital
        positions = dict(final_snapshot.positions) if final_snapshot else {}
        total_return = final_equity - initial_capital
        total_return_pct = (total_return / initial_capital * 100) if initial_capital > 0 else 0

        trades = [self._trade_from_fill(fill) for fill in self.fills]
        equity_curve = [
            {"timestamp": snapshot.timestamp, "equity": snapshot.equity}
            for snapshot in self.snapshots
        ]

        return {
            "initial_capital": initial_capital,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": self._max_drawdown_pct(),
            "total_trades": len(trades),
            "buy_trades": len([trade for trade in trades if trade["action"] == "buy"]),
            "sell_trades": len([trade for trade in trades if trade["action"] == "sell"]),
            "current_positions": positions,
            "remaining_cash": remaining_cash,
            "trades": trades,
            "equity_curve": equity_curve,
        }

    def _trade_from_fill(self, fill: Fill) -> dict:
        """把领域 Fill 转成旧报告使用的 trade 字典。"""
        return {
            "timestamp": fill.timestamp,
            "symbol": fill.symbol,
            "action": fill.side,
            "quantity": fill.quantity,
            "price": fill.price,
            "commission": fill.commission,
        }

    def _max_drawdown_pct(self) -> float:
        """按 reporter 的权益曲线计算最大回撤百分比。"""
        peak = None
        max_drawdown = 0.0
        for snapshot in self.snapshots:
            equity = float(snapshot.equity)
            peak = equity if peak is None else max(peak, equity)
            if peak and peak > 0:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
        return max_drawdown * 100
