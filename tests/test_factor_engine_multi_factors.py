import unittest
from datetime import datetime, timedelta, timezone

from src.domain.models import MarketBar, MarketSlice
from src.factor.engine import FactorEngine


class MultiFactorEngineTests(unittest.TestCase):
    def test_from_config_calculates_rsi_after_warmup(self):
        engine = FactorEngine.from_config({
            "rsi_fast": {
                "type": "rsi",
                "params": {"period": 3},
                "window_size": 4,
                "signal_type": "standard",
                "weight": 1.0,
            }
        })

        view = None
        for offset, close in enumerate([100, 101, 102, 103]):
            view = engine.update(self._market_slice(close=close, timestamp=self._ts(offset)))

        self.assertTrue(view.has("BTC/USDT", "rsi_fast"))
        self.assertGreater(view.get("BTC/USDT", "rsi_fast"), 99.0)

    def test_from_config_returns_empty_view_before_factor_warmup(self):
        engine = FactorEngine.from_config({
            "rsi_fast": {
                "type": "rsi",
                "params": {"period": 3},
                "window_size": 4,
                "signal_type": "standard",
                "weight": 1.0,
            }
        })

        view = engine.update(self._market_slice(close=100, timestamp=self._ts(0)))

        self.assertFalse(view.has("BTC/USDT", "rsi_fast"))
        self.assertFalse(view.has("BTC/USDT", "composite_signal"))

    def test_from_config_emits_weighted_composite_signal(self):
        engine = FactorEngine.from_config({
            "rsi_fast": {
                "type": "rsi",
                "params": {"period": 3},
                "window_size": 4,
                "signal_type": "threshold",
                "upper_threshold": 70,
                "lower_threshold": 30,
                "normalize": False,
                "weight": 2.0,
            },
            "volume_momentum": {
                "type": "volume_osc",
                "params": {"fast_period": 2, "slow_period": 3},
                "window_size": 3,
                "signal_type": "momentum",
                "normalize": False,
                "weight": 1.0,
            },
        })

        view = None
        for offset, (close, volume) in enumerate([
            (100, 10),
            (99, 10),
            (98, 11),
            (97, 13),
        ]):
            view = engine.update(self._market_slice(close=close, volume=volume, timestamp=self._ts(offset)))

        self.assertTrue(view.has("BTC/USDT", "composite_signal"))
        self.assertGreater(view.get("BTC/USDT", "composite_signal"), 0)

    def _market_slice(self, close, volume=10, timestamp=None):
        timestamp = timestamp or self._ts(0)
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar(
                    "BTC/USDT",
                    "1m",
                    timestamp,
                    close,
                    close,
                    close,
                    close,
                    volume,
                )
            },
        )

    def _ts(self, offset):
        return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset)


if __name__ == "__main__":
    unittest.main()
