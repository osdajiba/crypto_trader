import unittest
from datetime import datetime, timezone

import pandas as pd


class DomainModelsTest(unittest.TestCase):
    def test_market_bar_requires_timezone_aware_timestamp(self):
        from src.domain.models import MarketBar

        with self.assertRaises(ValueError):
            MarketBar(
                symbol="BTC/USDT",
                timeframe="1m",
                timestamp=datetime(2025, 1, 1),
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                volume=100,
            )

    def test_market_slice_conversion_groups_bars_by_timestamp(self):
        from src.application.adapters.dataframe_domain_adapter import market_slices_from_dataframes

        btc = pd.DataFrame([
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
        eth = pd.DataFrame([
            {
                "datetime": pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10.5,
                "volume": 100,
            }
        ])

        slices = market_slices_from_dataframes(
            {"BTC/USDT": btc, "ETH/USDT": eth},
            timeframe="1m",
        )

        self.assertEqual(
            [s.timestamp for s in slices],
            [
                pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                pd.Timestamp("2025-01-01 00:01:00", tz="UTC"),
            ],
        )
        self.assertEqual(set(slices[0].bars_by_symbol), {"BTC/USDT", "ETH/USDT"})
        self.assertEqual(slices[0].bars_by_symbol["BTC/USDT"].close, 1.5)
        self.assertEqual(set(slices[1].bars_by_symbol), {"BTC/USDT"})

    def test_strategy_signal_conversion_uses_deterministic_signal_id(self):
        from src.application.adapters.dataframe_domain_adapter import strategy_signals_from_dataframe

        signals = pd.DataFrame([
            {
                "datetime": pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                "symbol": "BTC/USDT",
                "action": "buy",
                "price": 95000,
                "quantity": 0.01,
            }
        ])

        converted = strategy_signals_from_dataframe(signals, strategy_id="dual_ma")

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].signal_id, "dual_ma:BTC/USDT:buy:2025-01-01T00:00:00+00:00")
        self.assertEqual(converted[0].symbol, "BTC/USDT")
        self.assertEqual(converted[0].side, "buy")
        self.assertEqual(converted[0].metadata["price"], 95000)
        self.assertEqual(converted[0].metadata["quantity"], 0.01)

    def test_risk_decision_can_create_order_intent(self):
        from src.domain.models import RiskDecision, StrategySignal

        signal = StrategySignal(
            signal_id="s1",
            symbol="BTC/USDT",
            timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
            side="buy",
            strength=1.0,
            reason="test",
        )
        decision = RiskDecision(
            accepted=True,
            reason="accepted",
            target_notional=1000,
            target_quantity=0.01,
            adjusted_signal=signal,
        )

        intent = decision.to_order_intent(order_type="market")

        self.assertEqual(intent.symbol, "BTC/USDT")
        self.assertEqual(intent.side, "buy")
        self.assertEqual(intent.quantity, 0.01)
        self.assertEqual(intent.source_signal_id, "s1")


if __name__ == "__main__":
    unittest.main()
