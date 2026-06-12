"""领域风控规则。

这些规则只处理领域对象，不接触 DataFrame 或旧 risk manager。
后续 pipeline 可以用组合式风控把多个规则串起来。
"""

from __future__ import annotations

from typing import Iterable

from src.domain.models import MarketSlice, PortfolioSnapshot, RiskDecision, StrategySignal


class PositionAvailabilityPolicy:
    """空仓卖出保护。"""

    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        if signal.side == "sell" and float(portfolio.positions.get(signal.symbol, 0) or 0) <= 0:
            return RiskDecision(
                accepted=False,
                reason="no_position",
                target_notional=0,
                target_quantity=0,
                adjusted_signal=signal,
            )

        return _accepted_decision(signal, portfolio, market)


class SellQuantityClampPolicy:
    """超额卖出收缩到当前持仓。"""

    def __init__(self, requested_quantity: float | None = None) -> None:
        self.requested_quantity = requested_quantity

    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        quantity = self.requested_quantity
        if quantity is None:
            quantity = float(signal.metadata.get("quantity", 0) or 0)

        if signal.side == "sell":
            current_position = float(portfolio.positions.get(signal.symbol, 0) or 0)
            quantity = min(float(quantity or 0), current_position)

        price = market.bars_by_symbol[signal.symbol].close
        return RiskDecision(
            accepted=quantity > 0,
            reason="" if quantity > 0 else "zero_quantity",
            target_notional=quantity * price,
            target_quantity=quantity,
            adjusted_signal=signal,
        )


class CompositeRiskPolicy:
    """按顺序执行多个风控规则，任一拒绝则停止。"""

    def __init__(self, policies: Iterable) -> None:
        self.policies = list(policies)

    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        current_signal = signal
        last_decision = _accepted_decision(current_signal, portfolio, market)

        for policy in self.policies:
            last_decision = await policy.evaluate(current_signal, portfolio, market)
            if not last_decision.accepted:
                return last_decision
            current_signal = last_decision.adjusted_signal

        return last_decision


def _accepted_decision(signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
    quantity = float(signal.metadata.get("quantity", 0) or 0)
    price = market.bars_by_symbol[signal.symbol].close
    return RiskDecision(
        accepted=True,
        reason="",
        target_notional=quantity * price,
        target_quantity=quantity,
        adjusted_signal=signal,
    )
