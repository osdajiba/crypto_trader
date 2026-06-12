import unittest
from datetime import datetime, timezone

from src.domain.models import MarketBar, MarketSlice, RiskDecision, StrategySignal
from src.domain.portfolio import PortfolioBook
from src.domain.risk_policies import CompositeRiskPolicy, PositionAvailabilityPolicy, SellQuantityClampPolicy


class DomainRiskPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_position_availability_rejects_sell_without_position(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        portfolio = PortfolioBook(initial_cash=100000)
        policy = PositionAvailabilityPolicy()

        decision = await policy.evaluate(
            self._signal(timestamp, side="sell"),
            portfolio.snapshot(timestamp),
            self._market(timestamp),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_position")

    async def test_sell_quantity_clamp_limits_sell_to_current_position(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        portfolio = PortfolioBook(initial_cash=100000)
        portfolio.update_market_price("BTC/USDT", 100000)
        portfolio.apply_fill(timestamp, "BTC/USDT", "buy", 0.02, 100000, 1)
        policy = SellQuantityClampPolicy(requested_quantity=0.05)

        decision = await policy.evaluate(
            self._signal(timestamp, side="sell"),
            portfolio.snapshot(timestamp),
            self._market(timestamp),
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.target_quantity, 0.02)

    async def test_composite_policy_stops_on_rejection(self):
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        policy = CompositeRiskPolicy([
            PositionAvailabilityPolicy(),
            SellQuantityClampPolicy(requested_quantity=0.05),
        ])

        decision = await policy.evaluate(
            self._signal(timestamp, side="sell"),
            PortfolioBook(initial_cash=100000).snapshot(timestamp),
            self._market(timestamp),
        )

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "no_position")

    def _signal(self, timestamp, side):
        return StrategySignal(
            signal_id=f"test:BTC/USDT:{side}:1",
            symbol="BTC/USDT",
            timestamp=timestamp,
            side=side,
        )

    def _market(self, timestamp):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100000, 100000, 100000, 100000, 10)
            },
        )


if __name__ == "__main__":
    unittest.main()
