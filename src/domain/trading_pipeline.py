"""纯领域交易流水线。

这里不依赖 pandas、mode、config 或具体交易所实现，只按 ports 协议把
行情、策略、风控、执行、组合账本和报告串起来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.domain.models import Fill, MarketSlice, PortfolioSnapshot
from src.domain.portfolio import PortfolioBook
from src.domain.ports import ExecutionModel, Reporter, RiskPolicy, StrategyPort


@dataclass(frozen=True)
class PipelineResult:
    """单个市场切片处理后的领域结果。"""
    fills: List[Fill]
    snapshot: PortfolioSnapshot


class DomainTradingPipeline:
    """执行一轮纯领域交易决策。"""

    def __init__(
        self,
        strategy: StrategyPort,
        risk_policy: RiskPolicy,
        execution_model: ExecutionModel,
        portfolio: PortfolioBook,
        reporter: Reporter,
    ) -> None:
        self.strategy = strategy
        self.risk_policy = risk_policy
        self.execution_model = execution_model
        self.portfolio = portfolio
        self.reporter = reporter

    async def run_once(self, market: MarketSlice) -> PipelineResult:
        """处理一个 MarketSlice，并返回本轮成交和最新组合快照。"""
        self._update_market_prices(market)
        fills: List[Fill] = []

        starting_snapshot = self.portfolio.snapshot(market.timestamp)
        signals = await self.strategy.generate(market, starting_snapshot)

        for signal in signals:
            if signal.timestamp != market.timestamp:
                continue

            current_snapshot = self.portfolio.snapshot(market.timestamp)
            decision = await self.risk_policy.evaluate(signal, current_snapshot, market)
            if not decision.accepted:
                continue

            order_intent = decision.to_order_intent()
            fill = await self.execution_model.execute(order_intent, market, current_snapshot)
            trade = self.portfolio.apply_fill(
                timestamp=fill.timestamp,
                symbol=fill.symbol,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                commission=fill.commission,
                action=fill.side,
            )
            if trade is None:
                continue

            fills.append(fill)
            await self.reporter.record_fill(fill)

        self.portfolio.record_equity(market.timestamp)
        snapshot = self.portfolio.snapshot(market.timestamp)
        await self.reporter.record_snapshot(snapshot)
        return PipelineResult(fills=fills, snapshot=snapshot)

    def _update_market_prices(self, market: MarketSlice) -> None:
        """把当前市场切片价格写入组合账本，用于后续净值计算。"""
        for symbol, bar in market.bars_by_symbol.items():
            self.portfolio.update_market_price(symbol, bar.close)
