"""交易流水线的端口协议。

这些 Protocol 是完整解耦后的稳定边界：领域核心只认识行情切片、
策略信号、风控决策、订单意图、成交结果和组合快照，不认识 DataFrame、
mode、config 或具体交易所实现。
"""

from __future__ import annotations

from typing import Iterable, Protocol, Sequence

from src.domain.models import Fill, MarketSlice, OrderIntent, PortfolioSnapshot, RiskDecision, StrategySignal


class MarketDataFeed(Protocol):
    """行情输入端口：回测读取历史切片，模拟盘和实盘读取最新切片。"""
    def load_range(self, symbols: Sequence[str], timeframe: str, start, end) -> Iterable[MarketSlice]:
        ...

    async def latest(self, symbols: Sequence[str], timeframe: str) -> MarketSlice:
        ...


class StrategyPort(Protocol):
    """策略端口：输入市场切片和组合快照，输出领域信号。"""
    async def generate(self, market: MarketSlice, portfolio: PortfolioSnapshot) -> list[StrategySignal]:
        ...


class RiskPolicy(Protocol):
    """风控端口：根据组合快照和市场切片裁决信号。"""
    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        ...


class ExecutionModel(Protocol):
    """执行端口：把下单意图转成成交结果。"""
    async def execute(self, order_intent: OrderIntent, market: MarketSlice, portfolio: PortfolioSnapshot) -> Fill:
        ...


class Reporter(Protocol):
    """报告端口：记录组合快照和成交，用于生成报告。"""
    async def record_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        ...

    async def record_fill(self, fill: Fill) -> None:
        ...
