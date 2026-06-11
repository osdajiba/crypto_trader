import unittest
from datetime import datetime, timezone

import pandas as pd

from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter
from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot, StrategySignal


class RejectingRiskManager:
    async def validate_signals(self, signals):
        result = signals.copy()
        result["risk_accepted"] = False
        result["risk_reason"] = "blocked_by_legacy"
        return result


class LegacyRiskPolicyAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_risk_rejection_becomes_risk_decision(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        adapter = LegacyRiskPolicyAdapter(RejectingRiskManager())

        decision = await adapter.evaluate(
            StrategySignal("sig-1", "BTC/USDT", timestamp, "buy", metadata={"quantity": 0.01, "price": 100000}),
            PortfolioSnapshot(timestamp, 100000, {}, {"BTC/USDT": 100000}, 100000),
            MarketSlice(
                timestamp=timestamp,
                bars_by_symbol={
                    "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100000, 100000, 100000, 100000, 10)
                },
            ),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "blocked_by_legacy")
        self.assertEqual(decision.target_quantity, 0.01)


if __name__ == "__main__":
    unittest.main()
