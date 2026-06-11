import unittest
from datetime import datetime, timezone


class PortfolioBookTest(unittest.TestCase):
    def test_apply_buy_fill_updates_cash_position_and_trade(self):
        from src.domain.portfolio import PortfolioBook

        book = PortfolioBook(initial_cash=1000)

        trade = book.apply_fill(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            side="buy",
            quantity=2,
            price=100,
            commission=1,
        )

        self.assertEqual(book.cash, 799)
        self.assertEqual(book.positions, {"BTC/USDT": 2})
        self.assertEqual(trade["cash_after"], 799)
        self.assertEqual(len(book.trades), 1)

    def test_apply_sell_fill_updates_cash_and_removes_closed_position(self):
        from src.domain.portfolio import PortfolioBook

        book = PortfolioBook(initial_cash=1000)
        book.apply_fill(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            side="buy",
            quantity=2,
            price=100,
            commission=1,
        )

        trade = book.apply_fill(
            timestamp=datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            side="sell",
            quantity=2,
            price=120,
            commission=1,
        )

        self.assertEqual(book.cash, 1038)
        self.assertEqual(book.positions, {})
        self.assertEqual(trade["cash_after"], 1038)
        self.assertEqual(len(book.trades), 2)

    def test_flat_sell_returns_none_without_mutating_state(self):
        from src.domain.portfolio import PortfolioBook

        book = PortfolioBook(initial_cash=1000)

        trade = book.apply_fill(
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            side="sell",
            quantity=1,
            price=100,
            commission=1,
        )

        self.assertIsNone(trade)
        self.assertEqual(book.cash, 1000)
        self.assertEqual(book.positions, {})
        self.assertEqual(book.trades, [])

    def test_equity_curve_and_drawdown_use_market_prices(self):
        from src.domain.portfolio import PortfolioBook

        book = PortfolioBook(initial_cash=1000)
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        book.apply_fill(
            timestamp=timestamp,
            symbol="BTC/USDT",
            side="buy",
            quantity=2,
            price=100,
            commission=0,
        )

        book.update_market_price("BTC/USDT", 150)
        first = book.record_equity(timestamp)
        book.update_market_price("BTC/USDT", 90)
        second = book.record_equity(datetime(2025, 1, 1, 1, tzinfo=timezone.utc))

        self.assertEqual(first.equity, 1100)
        self.assertEqual(second.equity, 980)
        self.assertAlmostEqual(book.max_drawdown, (1100 - 980) / 1100)


if __name__ == "__main__":
    unittest.main()
