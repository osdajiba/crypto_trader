"""Risk policy bridge for sizing native strategy signals."""

from __future__ import annotations

from src.domain.models import MarketSlice, PortfolioSnapshot, RiskDecision, StrategySignal
from src.order.sizing import FixedFractionSizer, FixedNotionalSizer


class SizingRiskPolicy:
    """Use the order sizing layer to assign quantity to native strategy signals."""

    def __init__(self, sizer) -> None:
        self.sizer = sizer

    async def evaluate(self, signal: StrategySignal, portfolio: PortfolioSnapshot, market: MarketSlice) -> RiskDecision:
        decision = self.sizer.size(signal, portfolio, market)
        return RiskDecision(
            accepted=decision.target_quantity > 0,
            reason=decision.reason if decision.target_quantity > 0 else "zero_quantity",
            target_notional=decision.target_notional,
            target_quantity=decision.target_quantity,
            adjusted_signal=signal,
        )


def fixed_fraction_sizing_policy(fraction: float, commission_rate: float = 0.0) -> SizingRiskPolicy:
    return SizingRiskPolicy(FixedFractionSizer(fraction=fraction, commission_rate=commission_rate))


def fixed_notional_sizing_policy(notional: float, commission_rate: float = 0.0) -> SizingRiskPolicy:
    return SizingRiskPolicy(FixedNotionalSizer(notional=notional, commission_rate=commission_rate))
