import unittest
from datetime import datetime, timezone

from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot
from src.factor.models import FactorView


class DomainMultiFactorsStrategyTests(unittest.IsolatedAsyncioTestCase):
    async def test_outputs_buy_signal_without_quantity_when_composite_crosses_threshold(self):
        from src.strategy.implementations.domain_multi_factors import DomainMultiFactorsStrategy

        strategy = DomainMultiFactorsStrategy(threshold=0.5)
        market = self._market()
        portfolio = self._portfolio()

        await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": 0.1}}), portfolio)
        signals = await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": 0.8}}), portfolio)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "BTC/USDT")
        self.assertEqual(signals[0].side, "buy")
        self.assertNotIn("quantity", signals[0].metadata)

    async def test_outputs_sell_signal_when_composite_crosses_negative_threshold_and_position_exists(self):
        from src.strategy.implementations.domain_multi_factors import DomainMultiFactorsStrategy

        strategy = DomainMultiFactorsStrategy(threshold=0.5)
        market = self._market()
        portfolio = self._portfolio(positions={"BTC/USDT": 0.1})

        await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": -0.1}}), portfolio)
        signals = await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": -0.8}}), portfolio)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].side, "sell")
        self.assertNotIn("quantity", signals[0].metadata)

    async def test_does_not_repeat_same_side_signal_without_new_cross(self):
        from src.strategy.implementations.domain_multi_factors import DomainMultiFactorsStrategy

        strategy = DomainMultiFactorsStrategy(threshold=0.5)
        market = self._market()
        portfolio = self._portfolio()

        await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": 0.1}}), portfolio)
        first = await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": 0.8}}), portfolio)
        second = await strategy.generate(market, FactorView({"BTC/USDT": {"composite_signal": 0.9}}), portfolio)

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])

    def _market(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100, 101, 99, 100, 10)
            },
        )

    def _portfolio(self, positions=None):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=100000,
            positions=positions or {},
            market_prices={"BTC/USDT": 100},
            equity=100000,
        )


if __name__ == "__main__":
    unittest.main()
