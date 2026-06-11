import datetime
import types
import unittest

import pandas as pd

from src.domain.models import MarketBar, MarketSlice, OrderIntent
from src.domain.portfolio import PortfolioBook


class BacktestExecutionModelTest(unittest.IsolatedAsyncioTestCase):
    async def test_domain_execute_buy_returns_fill_with_conservative_price(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        model = BacktestExecutionModel(historical_data={}, commission=0.001, slippage=0.01)
        portfolio = PortfolioBook(initial_cash=100000)

        fill = await model.execute(
            OrderIntent("intent-1", "BTC/USDT", "buy", 1.0, "market", timestamp),
            self._market(timestamp, high=101.0, low=99.0, volume=10.0),
            portfolio.snapshot(timestamp),
        )

        self.assertEqual(fill.order_intent_id, "intent-1")
        self.assertEqual(fill.side, "buy")
        self.assertAlmostEqual(fill.quantity, 1.0)
        self.assertAlmostEqual(fill.price, 102.01)
        self.assertAlmostEqual(fill.commission, 0.10201)

    async def test_domain_execute_sell_returns_fill_with_conservative_price(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        model = BacktestExecutionModel(historical_data={}, commission=0.001, slippage=0.01)

        fill = await model.execute(
            OrderIntent("intent-2", "BTC/USDT", "sell", 1.0, "market", timestamp),
            self._market(timestamp, high=101.0, low=99.0, volume=10.0),
            PortfolioBook(initial_cash=100000).snapshot(timestamp),
        )

        self.assertEqual(fill.side, "sell")
        self.assertAlmostEqual(fill.price, 98.01)
        self.assertAlmostEqual(fill.commission, 0.09801)

    async def test_domain_execute_caps_quantity_without_mutating_portfolio(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        model = BacktestExecutionModel(
            historical_data={},
            commission=0.0,
            slippage=0.0,
            max_bar_volume_fraction=0.1,
        )
        portfolio = PortfolioBook(initial_cash=100000)
        snapshot = portfolio.snapshot(timestamp)

        fill = await model.execute(
            OrderIntent("intent-3", "BTC/USDT", "buy", 2.5, "market", timestamp),
            self._market(timestamp, high=100.0, low=90.0, volume=10.0),
            snapshot,
        )

        self.assertAlmostEqual(fill.quantity, 1.0)
        self.assertEqual(fill.status, "partial")
        self.assertEqual(portfolio.cash, 100000)
        self.assertEqual(portfolio.positions, {})

    async def test_buy_uses_high_price_plus_slippage(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 10.0,
                }])
            },
            commission=0.001,
            slippage=0.01,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.BUY,
            quantity=1.0,
            timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        executed, _ = await model.execute_orders([order])

        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(executed.iloc[0]["price"], 102.01)
        self.assertAlmostEqual(executed.iloc[0]["commission"], 0.10201)

    async def test_sell_uses_low_price_minus_slippage(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 101.0,
                    "low": 99.0,
                    "volume": 10.0,
                }])
            },
            commission=0.001,
            slippage=0.01,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.SELL,
            quantity=1.0,
            timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        executed, _ = await model.execute_orders([order])

        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(executed.iloc[0]["price"], 98.01)
        self.assertAlmostEqual(executed.iloc[0]["commission"], 0.09801)

    async def test_volume_cap_creates_partial_fill_and_updates_volume(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 100.0,
                    "low": 90.0,
                    "volume": 10.0,
                }])
            },
            commission=0.0,
            slippage=0.0,
            max_bar_volume_fraction=0.1,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.BUY,
            quantity=2.5,
            timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        executed, updated_data = await model.execute_orders([order])

        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(executed.iloc[0]["filled_qty"], 1.0)
        self.assertAlmostEqual(executed.iloc[0]["unfilled_qty"], 1.5)
        self.assertAlmostEqual(updated_data["BTC/USDT"].iloc[0]["volume"], 9.0)

    async def test_execute_orders_preserves_fractional_crypto_quantity(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 20010.0,
                    "low": 19990.0,
                    "volume": 1000.0,
                }])
            },
            commission=0.0,
            slippage=0.0,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.BUY,
            quantity=0.049,
            timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        executed, _ = await model.execute_orders([order])

        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(executed.iloc[0]["filled_qty"], 0.049)

    async def test_execute_orders_decrements_integer_volume_by_fractional_fill(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 20010.0,
                    "low": 19990.0,
                    "volume": 1000,
                }])
            },
            commission=0.0,
            slippage=0.0,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.BUY,
            quantity=0.049,
            timestamp=datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
        )

        executed, updated_data = await model.execute_orders([order])

        self.assertEqual(len(executed), 1)
        self.assertAlmostEqual(executed.iloc[0]["filled_qty"], 0.049)
        self.assertAlmostEqual(updated_data["BTC/USDT"].iloc[0]["volume"], 999.951)

    async def test_missing_execution_bar_returns_no_fill(self):
        from src.trading.execution.backtest_model import BacktestExecutionModel
        from src.trading.execution.order import Direction

        model = BacktestExecutionModel(
            historical_data={
                "BTC/USDT": pd.DataFrame([{
                    "timestamp": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                    "high": 100.0,
                    "low": 90.0,
                    "volume": 10.0,
                }])
            },
            commission=0.0,
            slippage=0.0,
        )
        order = types.SimpleNamespace(
            order_id="o1",
            symbol="BTC/USDT",
            direction=Direction.BUY,
            quantity=1.0,
            timestamp=datetime.datetime(2025, 1, 2, tzinfo=datetime.timezone.utc),
        )

        executed, _ = await model.execute_orders([order])

        self.assertTrue(executed.empty)

    def _market(self, timestamp, high, low, volume):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    timestamp=timestamp,
                    open=100.0,
                    high=high,
                    low=low,
                    close=100.0,
                    volume=volume,
                )
            },
        )


if __name__ == "__main__":
    unittest.main()
