"""Native domain strategy base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.models import MarketSlice, PortfolioSnapshot, StrategySignal
from src.factor.engine import FactorEngine
from src.factor.models import FactorView


class BaseDomainStrategy(ABC):
    """Base class for strategies that consume domain objects and factor views."""

    def __init__(self, factor_engine: FactorEngine | None = None) -> None:
        self.factor_engine = factor_engine

    async def generate(self, market: MarketSlice, *args) -> list[StrategySignal]:
        if len(args) == 1:
            portfolio = args[0]
            if self.factor_engine is None:
                raise ValueError("factor_engine is required when factors are not provided")
            factors = self.factor_engine.update(market)
        elif len(args) == 2:
            factors, portfolio = args
        else:
            raise TypeError("generate expects (market, portfolio) or (market, factors, portfolio)")

        return await self.generate_with_factors(market, factors, portfolio)

    @abstractmethod
    async def generate_with_factors(
        self,
        market: MarketSlice,
        factors: FactorView,
        portfolio: PortfolioSnapshot,
    ) -> list[StrategySignal]:
        ...
