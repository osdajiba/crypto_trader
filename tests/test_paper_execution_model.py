import datetime
import unittest

from src.domain.models import MarketBar, MarketSlice, OrderIntent
from src.domain.portfolio import PortfolioBook


class PaperExecutionModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_buy_returns_simulated_fill_without_mutating_portfolio(self):
        from src.trading.execution.paper_model import PaperExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        portfolio = PortfolioBook(initial_cash=100000)
        model = PaperExecutionModel(commission=0.001, slippage=0.01)

        fill = await model.execute(
            OrderIntent("intent-1", "BTC/USDT", "buy", 0.5, "market", timestamp),
            self._market(timestamp, close=100.0),
            portfolio.snapshot(timestamp),
        )

        self.assertEqual(fill.side, "buy")
        self.assertAlmostEqual(fill.price, 101.0)
        self.assertAlmostEqual(fill.commission, 0.0505)
        self.assertEqual(portfolio.cash, 100000)
        self.assertEqual(portfolio.positions, {})

    async def test_sell_returns_simulated_fill_with_slippage(self):
        from src.trading.execution.paper_model import PaperExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        model = PaperExecutionModel(commission=0.001, slippage=0.01)

        fill = await model.execute(
            OrderIntent("intent-2", "BTC/USDT", "sell", 0.5, "market", timestamp),
            self._market(timestamp, close=100.0),
            PortfolioBook(initial_cash=100000).snapshot(timestamp),
        )

        self.assertEqual(fill.side, "sell")
        self.assertAlmostEqual(fill.price, 99.0)
        self.assertAlmostEqual(fill.commission, 0.0495)

    def _market(self, timestamp, close):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, close, close + 1, close - 1, close, 10.0)
            },
        )


if __name__ == "__main__":
    unittest.main()
