"""仓位 sizing 规则。"""

from __future__ import annotations

from src.domain.models import MarketSlice, PortfolioSnapshot, StrategySignal
from src.order.models import PositionSizingDecision


class FixedNotionalSizer:
    """按固定下单金额计算数量。"""

    def __init__(self, notional: float, commission_rate: float = 0.0) -> None:
        self.notional = float(notional)
        self.commission_rate = float(commission_rate)

    def size(
        self,
        signal: StrategySignal,
        portfolio: PortfolioSnapshot,
        market: MarketSlice,
    ) -> PositionSizingDecision:
        price = _close_price(signal, market)
        target_notional = min(self.notional, _available_notional(signal, portfolio, self.commission_rate))
        quantity = target_notional / price if price > 0 else 0.0
        return PositionSizingDecision(
            signal=signal,
            target_notional=target_notional,
            target_quantity=quantity,
        )


class FixedFractionSizer:
    """按组合权益固定比例计算数量。"""

    def __init__(self, fraction: float, commission_rate: float = 0.0) -> None:
        self.fraction = float(fraction)
        self.commission_rate = float(commission_rate)

    def size(
        self,
        signal: StrategySignal,
        portfolio: PortfolioSnapshot,
        market: MarketSlice,
    ) -> PositionSizingDecision:
        price = _close_price(signal, market)
        requested_notional = max(float(portfolio.equity), 0.0) * self.fraction
        target_notional = min(requested_notional, _available_notional(signal, portfolio, self.commission_rate))
        quantity = target_notional / price if price > 0 else 0.0
        return PositionSizingDecision(
            signal=signal,
            target_notional=target_notional,
            target_quantity=quantity,
        )


def _close_price(signal: StrategySignal, market: MarketSlice) -> float:
    return float(market.bars_by_symbol[signal.symbol].close)


def _available_notional(signal: StrategySignal, portfolio: PortfolioSnapshot, commission_rate: float) -> float:
    if signal.side == "buy":
        divisor = 1.0 + max(float(commission_rate), 0.0)
        return max(float(portfolio.cash), 0.0) / divisor
    if signal.side == "sell":
        price = float(portfolio.market_prices.get(signal.symbol, 0.0) or 0.0)
        return max(float(portfolio.positions.get(signal.symbol, 0.0) or 0.0), 0.0) * price
    return 0.0
