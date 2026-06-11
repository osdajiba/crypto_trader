import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pandas as pd

from common.config import ConfigManager
from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot
from src.strategy.implementations.dual_ma import DualMAStrategy


class FakeDataFrameStrategy:
    async def process_data(self, data, symbol):
        latest = data.iloc[-1]
        return pd.DataFrame([{
            "datetime": latest["datetime"],
            "timestamp": latest["timestamp"],
            "symbol": symbol,
            "action": "buy",
            "quantity": 0.01,
            "price": latest["close"],
        }])


class MultiRowDataFrameStrategy:
    async def process_data(self, data, symbol):
        older = data.iloc[0]
        latest = data.iloc[-1]
        return pd.DataFrame([
            {
                "datetime": older["datetime"] - pd.Timedelta(minutes=1),
                "timestamp": int((older["datetime"] - pd.Timedelta(minutes=1)).timestamp() * 1000),
                "symbol": symbol,
                "action": "buy",
                "quantity": 0.01,
                "price": older["close"],
            },
            {
                "datetime": latest["datetime"],
                "timestamp": latest["timestamp"],
                "symbol": symbol,
                "action": "buy",
                "quantity": 0.01,
                "price": latest["close"],
            },
        ])


class LegacyStrategyAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_converts_market_slice_to_strategy_signal(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        market = self._market_slice(timestamp, close=100000)
        adapter = LegacyDataFrameStrategyAdapter(FakeDataFrameStrategy(), strategy_id="fake")

        signals = await adapter.generate(market, self._snapshot(timestamp))

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "BTC/USDT")
        self.assertEqual(signals[0].side, "buy")
        self.assertEqual(signals[0].signal_id, "fake:BTC/USDT:buy:2025-01-01T00:00:00+00:00")
        self.assertEqual(signals[0].metadata["quantity"], 0.01)

    async def test_dual_ma_strategy_can_generate_domain_signals_after_warmup(self):
        config = ConfigManager()
        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            strategy = DualMAStrategy(
                config=config,
                params={"short_window": 2, "long_window": 3, "trade_size": 0.01},
            )
        adapter = LegacyDataFrameStrategyAdapter(strategy, strategy_id="dual_ma")
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        closes = [10, 9, 8, 11]
        signals = []

        for index, close in enumerate(closes):
            timestamp = start + timedelta(minutes=index)
            signals.extend(
                await adapter.generate(
                    self._market_slice(timestamp, close=close),
                    self._snapshot(timestamp),
                )
            )

        self.assertTrue(signals)
        self.assertEqual(signals[-1].symbol, "BTC/USDT")
        self.assertIn(signals[-1].side, {"buy", "sell"})
        self.assertTrue(signals[-1].signal_id.startswith("dual_ma:BTC/USDT:"))

    async def test_adapter_keeps_only_current_unprocessed_signal(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        market = self._market_slice(timestamp, close=100000)
        adapter = LegacyDataFrameStrategyAdapter(MultiRowDataFrameStrategy(), strategy_id="fake")

        first = await adapter.generate(market, self._snapshot(timestamp))
        second = await adapter.generate(market, self._snapshot(timestamp))

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].timestamp, timestamp)
        self.assertEqual(second, [])

    def _market_slice(self, timestamp, close):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar(
                    symbol="BTC/USDT",
                    timeframe="1m",
                    timestamp=timestamp,
                    open=close,
                    high=close + 1,
                    low=close - 1,
                    close=close,
                    volume=100,
                )
            },
        )

    def _snapshot(self, timestamp):
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=100000,
            positions={},
            market_prices={"BTC/USDT": 100000},
            equity=100000,
        )


if __name__ == "__main__":
    unittest.main()
