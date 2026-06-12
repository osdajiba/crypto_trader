import unittest

import pandas as pd

from src.application.adapters.risk_decision_adapter import RiskDecisionAdapter


class RiskDecisionAdapterTest(unittest.TestCase):
    def test_from_dataframe_marks_rows_accepted_by_default(self):
        timestamp = pd.Timestamp("2025-01-01 00:01:00", tz="UTC")
        signals = pd.DataFrame([{
            "datetime": timestamp,
            "signal_id": "BTC/USDT:buy:1735689660000",
            "symbol": "BTC/USDT",
            "action": "buy",
            "quantity": 0.01,
            "price": 101.0,
        }])

        adapted = RiskDecisionAdapter().prepare(signals)

        self.assertEqual(len(adapted.dataframe), 1)
        self.assertEqual(adapted.dataframe["risk_accepted"].tolist(), [True])
        decision = adapted.decisions[0]
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.adjusted_signal.signal_id, "BTC/USDT:buy:1735689660000")
        self.assertEqual(decision.target_quantity, 0.01)
        self.assertEqual(decision.target_notional, 1.01)

    def test_filter_accepted_removes_rejected_rows(self):
        signals = pd.DataFrame([
            {
                "signal_id": "accepted",
                "symbol": "BTC/USDT",
                "action": "buy",
                "quantity": 0.01,
                "risk_accepted": True,
            },
            {
                "signal_id": "rejected",
                "symbol": "BTC/USDT",
                "action": "buy",
                "quantity": 0.01,
                "risk_accepted": False,
            },
        ])

        accepted = RiskDecisionAdapter().filter_accepted(signals)

        self.assertEqual(accepted["signal_id"].tolist(), ["accepted"])


if __name__ == "__main__":
    unittest.main()
