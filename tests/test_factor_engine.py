import unittest
from datetime import datetime, timedelta, timezone

from src.domain.models import MarketBar, MarketSlice
from src.factor.engine import FactorEngine


class FactorEngineTests(unittest.TestCase):
    def test_factor_engine_returns_empty_view_before_warmup(self):
        engine = FactorEngine(ma_windows=[3])

        view = engine.update(self._market_slice(close=100))

        self.assertFalse(view.has("BTC/USDT", "ma_3"))

    def test_factor_engine_calculates_moving_average_after_warmup(self):
        engine = FactorEngine(ma_windows=[3])
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

        for offset, price in enumerate([100, 101, 102]):
            view = engine.update(self._market_slice(close=price, timestamp=timestamp + timedelta(minutes=offset)))

        self.assertTrue(view.has("BTC/USDT", "ma_3"))
        self.assertEqual(view.get("BTC/USDT", "ma_3"), 101)

    def test_factor_engine_tracks_symbols_independently(self):
        engine = FactorEngine(ma_windows=[2])
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)

        engine.update(self._market_slice(close=100, symbol="BTC/USDT", timestamp=timestamp))
        engine.update(self._market_slice(close=10, symbol="ETH/USDT", timestamp=timestamp))
        view = engine.update(self._market_slice(close=102, symbol="BTC/USDT", timestamp=timestamp + timedelta(minutes=1)))

        self.assertEqual(view.get("BTC/USDT", "ma_2"), 101)
        self.assertFalse(view.has("ETH/USDT", "ma_2"))

    def _market_slice(self, close, symbol="BTC/USDT", timestamp=None):
        timestamp = timestamp or datetime(2025, 1, 1, tzinfo=timezone.utc)
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                symbol: MarketBar(symbol, "1m", timestamp, close, close, close, close, 10)
            },
        )


if __name__ == "__main__":
    unittest.main()
