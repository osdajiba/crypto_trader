import unittest

import pandas as pd

from src.application.adapters.strategy_signal_adapter import StrategySignalAdapter


class StrategySignalAdapterTest(unittest.TestCase):
    def test_from_dataframe_creates_stable_signal_id_and_domain_signal(self):
        timestamp = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")
        signals = pd.DataFrame([{
            "datetime": timestamp,
            "timestamp": 1735689660000,
            "symbol": "BTC/USDT",
            "action": "buy",
            "quantity": 0.01,
            "price": 101.0,
            "reason": "ma_cross",
        }])

        adapted = StrategySignalAdapter().prepare(signals, current_timestamp=timestamp)

        self.assertEqual(len(adapted.dataframe), 1)
        self.assertEqual(
            adapted.dataframe["signal_id"].tolist(),
            ["BTC/USDT:buy:1735689660000"],
        )
        domain_signal = adapted.domain_signals[0]
        self.assertEqual(domain_signal.signal_id, "BTC/USDT:buy:1735689660000")
        self.assertEqual(domain_signal.symbol, "BTC/USDT")
        self.assertEqual(domain_signal.side, "buy")
        self.assertEqual(domain_signal.timestamp, timestamp.to_pydatetime())
        self.assertEqual(domain_signal.metadata["quantity"], 0.01)
        self.assertEqual(domain_signal.metadata["price"], 101.0)

    def test_existing_signal_id_is_preserved(self):
        timestamp = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")
        signals = pd.DataFrame([{
            "datetime": timestamp,
            "signal_id": "custom-id",
            "symbol": "BTC/USDT",
            "action": "sell",
            "quantity": 0.01,
        }])

        adapted = StrategySignalAdapter().prepare(signals, current_timestamp=timestamp)

        self.assertEqual(adapted.dataframe["signal_id"].tolist(), ["custom-id"])
        self.assertEqual(adapted.domain_signals[0].signal_id, "custom-id")


if __name__ == "__main__":
    unittest.main()
