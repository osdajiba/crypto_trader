"""Native domain multi-factor strategy."""

from __future__ import annotations

from src.domain.models import MarketSlice, PortfolioSnapshot, StrategySignal
from src.factor.engine import FactorEngine
from src.factor.models import FactorView
from src.strategy.domain_base import BaseDomainStrategy


class DomainMultiFactorsStrategy(BaseDomainStrategy):
    """Multi-factor strategy that emits StrategySignal objects without sizing quantity."""

    def __init__(
        self,
        threshold: float = 0.5,
        factor_engine: FactorEngine | None = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = float(threshold)
        self._last_zone_by_symbol: dict[str, int] = {}
        super().__init__(factor_engine=factor_engine)

    async def generate_with_factors(
        self,
        market: MarketSlice,
        factors: FactorView,
        portfolio: PortfolioSnapshot,
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []

        for symbol in market.bars_by_symbol:
            composite = factors.get(symbol, "composite_signal")
            if composite is None:
                continue

            side = self._side_for(symbol, float(composite), portfolio)
            if side is None:
                continue

            signals.append(
                StrategySignal(
                    signal_id=f"domain_multi_factors:{symbol}:{side}:{market.timestamp.isoformat()}",
                    symbol=symbol,
                    timestamp=market.timestamp,
                    side=side,
                    strength=min(abs(float(composite)), 1.0),
                    reason=f"composite_signal_{side}_threshold",
                    metadata={"composite_signal": float(composite)},
                )
            )

        return signals

    def _side_for(
        self,
        symbol: str,
        composite: float,
        portfolio: PortfolioSnapshot,
    ) -> str | None:
        zone = self._zone(composite)
        previous_zone = self._last_zone_by_symbol.get(symbol, 0)
        self._last_zone_by_symbol[symbol] = zone

        if zone == 0 or previous_zone == zone:
            return None

        current_position = float(portfolio.positions.get(symbol, 0.0) or 0.0)
        if zone > 0 and current_position <= 0:
            return "buy"
        if zone < 0 and current_position > 0:
            return "sell"
        return None

    def _zone(self, composite: float) -> int:
        if composite > self.threshold:
            return 1
        if composite < -self.threshold:
            return -1
        return 0
