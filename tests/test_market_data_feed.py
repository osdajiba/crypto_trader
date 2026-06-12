import asyncio
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


class MarketDataFeedTest(unittest.IsolatedAsyncioTestCase):
    async def test_parquet_store_loads_supported_local_layouts(self):
        asyncio.get_running_loop().slow_callback_duration = 60
        from src.datasource.stores.parquet_store import ParquetHistoricalStore

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            historical_path = data_dir / "historical"
            layouts = [
                historical_path / "1m" / "BTC_USDT",
                historical_path / "binance" / "BTC_USDT" / "1m",
                data_dir / "binance" / "BTC_USDT" / "1m",
            ]
            for idx, layout in enumerate(layouts):
                layout.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([{
                    "datetime": pd.Timestamp(f"2025-01-01 00:0{idx}:00", tz="UTC"),
                    "timestamp": 1735689600000 + idx * 60000,
                    "open": 1 + idx,
                    "high": 2 + idx,
                    "low": 0.5 + idx,
                    "close": 1.5 + idx,
                    "volume": 10 + idx,
                }]).to_parquet(layout / f"BTC_USDT_1m_{idx}.parquet")

            store = ParquetHistoricalStore(str(historical_path))

            data = await store.load(
                symbol="BTC/USDT",
                timeframe="1m",
                start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, tzinfo=timezone.utc),
            )

        self.assertEqual(len(data), 3)
        self.assertEqual(data["close"].tolist(), [1.5, 2.5, 3.5])

    def test_normalizer_sorts_deduplicates_and_attaches_symbol(self):
        from src.datasource.normalizer import MarketDataNormalizer

        raw = pd.DataFrame([
            {
                "datetime": pd.Timestamp("2025-01-01 00:01:00"),
                "open": "2",
                "high": "3",
                "low": "1",
                "close": "2.5",
                "volume": "20",
            },
            {
                "datetime": pd.Timestamp("2025-01-01 00:00:00"),
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "10",
            },
            {
                "datetime": pd.Timestamp("2025-01-01 00:00:00"),
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "10",
            },
        ])

        normalized = MarketDataNormalizer.normalize_ohlcv(raw, symbol="BTC/USDT")

        self.assertEqual(len(normalized), 2)
        self.assertEqual(normalized["symbol"].tolist(), ["BTC/USDT", "BTC/USDT"])
        self.assertEqual(normalized["datetime"].iloc[0], pd.Timestamp("2025-01-01 00:00:00", tz="UTC"))
        self.assertEqual(normalized["close"].tolist(), [1.5, 2.5])

    async def test_historical_feed_loads_ordered_market_slices(self):
        from src.datasource.feeds.market_data_feed import HistoricalMarketDataFeed

        class Store:
            async def load(self, symbol, timeframe, start, end):
                return pd.DataFrame([
                    {
                        "datetime": pd.Timestamp("2025-01-01 00:01:00", tz="UTC"),
                        "open": 2,
                        "high": 3,
                        "low": 1,
                        "close": 2.5,
                        "volume": 20,
                    },
                    {
                        "datetime": pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                        "open": 1,
                        "high": 2,
                        "low": 0.5,
                        "close": 1.5,
                        "volume": 10,
                    },
                ])

        slices = await HistoricalMarketDataFeed(Store()).load_range(
            symbols=["BTC/USDT"],
            timeframe="1m",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-02", tz="UTC"),
        )

        self.assertEqual(len(slices), 2)
        self.assertEqual(
            [market_slice.timestamp for market_slice in slices],
            [
                pd.Timestamp("2025-01-01 00:00:00", tz="UTC").to_pydatetime(),
                pd.Timestamp("2025-01-01 00:01:00", tz="UTC").to_pydatetime(),
            ],
        )
        self.assertEqual(slices[0].bars_by_symbol["BTC/USDT"].close, 1.5)

    async def test_local_btc_1m_feed_loads_1441_slices_for_baseline_range(self):
        asyncio.get_running_loop().slow_callback_duration = 60
        from src.datasource.feeds.market_data_feed import HistoricalMarketDataFeed
        from src.datasource.stores.parquet_store import ParquetHistoricalStore

        historical_path = os.path.abspath("data/historical")
        if not os.path.exists(historical_path):
            self.skipTest("local historical data directory is not available")

        feed = HistoricalMarketDataFeed(ParquetHistoricalStore(historical_path))

        slices = await feed.load_range(
            symbols=["BTC/USDT"],
            timeframe="1m",
            start=pd.Timestamp("2025-01-01", tz="UTC"),
            end=pd.Timestamp("2025-01-02", tz="UTC"),
        )

        self.assertEqual(len(slices), 1441)
        self.assertEqual(slices[0].timestamp, pd.Timestamp("2025-01-01 00:00:00", tz="UTC").to_pydatetime())
        self.assertEqual(slices[-1].timestamp, pd.Timestamp("2025-01-02 00:00:00", tz="UTC").to_pydatetime())


if __name__ == "__main__":
    unittest.main()
