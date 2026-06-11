import unittest
from datetime import datetime, timezone

from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot
from src.factor.models import FactorView


class DomainDualMAStrategyTests(unittest.IsolatedAsyncioTestCase):
    def _market(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    timestamp=timestamp,
                    open=100,
                    high=103,
                    low=99,
                    close=102,
                    volume=10,
                )
            },
        )

    def _portfolio(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=100000,
            positions={},
            market_prices={"BTC/USDT": 102},
            equity=100000,
        )

    async def test_domain_dual_ma_outputs_buy_signal_without_quantity_metadata(self):
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy

        strategy = DomainDualMAStrategy(short_window=2, long_window=3)
        factors = FactorView({"BTC/USDT": {"ma_2": 102, "ma_3": 100}})

        signals = await strategy.generate(self._market(), factors, self._portfolio())

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "BTC/USDT")
        self.assertEqual(signals[0].side, "buy")
        self.assertNotIn("quantity", signals[0].metadata)

    async def test_domain_dual_ma_outputs_sell_when_short_ma_below_long_ma_and_position_exists(self):
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy

        strategy = DomainDualMAStrategy(short_window=2, long_window=3)
        portfolio = PortfolioSnapshot(
            timestamp=self._market().timestamp,
            cash=90000,
            positions={"BTC/USDT": 0.1},
            market_prices={"BTC/USDT": 98},
            equity=99800,
        )
        factors = FactorView({"BTC/USDT": {"ma_2": 98, "ma_3": 100}})

        signals = await strategy.generate(self._market(), factors, portfolio)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].side, "sell")
        self.assertNotIn("quantity", signals[0].metadata)

    async def test_domain_dual_ma_does_not_repeat_signal_without_new_cross(self):
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy

        strategy = DomainDualMAStrategy(short_window=2, long_window=3)
        factors = FactorView({"BTC/USDT": {"ma_2": 102, "ma_3": 100}})

        first = await strategy.generate(self._market(), factors, self._portfolio())
        second = await strategy.generate(self._market(), factors, self._portfolio())

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])


if __name__ == "__main__":
    unittest.main()
