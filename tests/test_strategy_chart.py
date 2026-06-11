import tempfile
import unittest
from pathlib import Path

import pandas as pd


class StrategyChartWriterTests(unittest.TestCase):
    def test_writes_ma_ema_crossover_png_for_backtest_data(self):
        from src.reporting.strategy_chart import StrategyChartWriter

        timestamps = pd.date_range("2025-01-01", periods=80, freq="1min", tz="UTC")
        prices = [100 + index * 0.2 for index in range(40)] + [108 - index * 0.15 for index in range(40)]
        data = pd.DataFrame({
            "datetime": timestamps,
            "open": prices,
            "high": [price + 1 for price in prices],
            "low": [price - 1 for price in prices],
            "close": prices,
            "volume": [10] * len(prices),
            "symbol": ["BTC/USDT"] * len(prices),
        })
        report = {
            "trades": [
                {"timestamp": timestamps[25], "symbol": "BTC/USDT", "action": "buy", "price": prices[25]},
                {"timestamp": timestamps[60], "symbol": "BTC/USDT", "action": "sell", "price": prices[60]},
            ]
        }

        with tempfile.TemporaryDirectory() as tmp:
            written = StrategyChartWriter(Path(tmp)).write(
                historical_data={"BTC/USDT": data},
                report=report,
                short_window=5,
                long_window=20,
                start_date="2025-01-01",
                end_date="2025-01-02",
            )

            self.assertEqual(written.name, "ma_ema_crossovers_20250101_20250102.png")
            self.assertTrue(written.exists())
            self.assertGreater(written.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
