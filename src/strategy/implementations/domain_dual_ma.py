"""Native domain Dual Moving Average strategy."""

from __future__ import annotations

from src.domain.models import MarketSlice, PortfolioSnapshot, StrategySignal
from src.factor.engine import FactorEngine
from src.factor.models import FactorView
from src.strategy.domain_base import BaseDomainStrategy


class DomainDualMAStrategy(BaseDomainStrategy):
    """Dual MA strategy that emits StrategySignal objects without sizing quantity."""

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        factor_engine: FactorEngine | None = None,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self._last_relation_by_symbol: dict[str, int] = {}
        super().__init__(
            factor_engine=factor_engine
            or FactorEngine(ma_windows=[self.short_window, self.long_window])
        )

    async def generate_with_factors(
        self,
        market: MarketSlice,
        factors: FactorView,
        portfolio: PortfolioSnapshot,
    ) -> list[StrategySignal]:
        signals: list[StrategySignal] = []

        for symbol in market.bars_by_symbol:
            short_ma = factors.get(symbol, f"ma_{self.short_window}")
            long_ma = factors.get(symbol, f"ma_{self.long_window}")
            if short_ma is None or long_ma is None:
                continue

            side = self._side_for(symbol, float(short_ma), float(long_ma), portfolio)
            if side is None:
                continue

            signals.append(
                StrategySignal(
                    signal_id=f"domain_dual_ma:{symbol}:{side}:{market.timestamp.isoformat()}",
                    symbol=symbol,
                    timestamp=market.timestamp,
                    side=side,
                    reason=f"ma_{self.short_window}_{side}_ma_{self.long_window}",
                    metadata={
                        "short_ma": float(short_ma),
                        "long_ma": float(long_ma),
                    },
                )
            )

        return signals

    def _side_for(
        self,
        symbol: str,
        short_ma: float,
        long_ma: float,
        portfolio: PortfolioSnapshot,
    ) -> str | None:
        relation = self._relation(short_ma, long_ma)
        previous_relation = self._last_relation_by_symbol.get(symbol)
        self._last_relation_by_symbol[symbol] = relation

        if relation == 0 or previous_relation == relation:
            return None

        current_position = float(portfolio.positions.get(symbol, 0.0) or 0.0)
        if relation > 0 and current_position <= 0:
            return "buy"
        if relation < 0 and current_position > 0:
            return "sell"
        return None

    def _relation(self, short_ma: float, long_ma: float) -> int:
        if short_ma > long_ma:
            return 1
        if short_ma < long_ma:
            return -1
        return 0
