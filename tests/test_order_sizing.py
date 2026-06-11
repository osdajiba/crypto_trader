import unittest
from datetime import datetime, timezone

from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot, StrategySignal
from src.order.factory import OrderFactory
from src.order.models import PositionSizingDecision
from src.order.risk_policy import fixed_fraction_sizing_policy
from src.order.sizing import FixedFractionSizer, FixedNotionalSizer


class OrderSizingTests(unittest.IsolatedAsyncioTestCase):
    def test_fixed_notional_sizer_calculates_quantity_from_price(self):
        timestamp = self._timestamp()
        signal = self._signal(timestamp)
        market = self._market(timestamp, close=100)
        portfolio = self._portfolio(timestamp, cash=10000)

        decision = FixedNotionalSizer(notional=1000).size(signal, portfolio, market)

        self.assertEqual(decision.target_notional, 1000)
        self.assertEqual(decision.target_quantity, 10)
        self.assertEqual(decision.signal, signal)

    def test_sizer_caps_buy_to_available_cash_after_commission(self):
        timestamp = self._timestamp()
        signal = self._signal(timestamp)
        market = self._market(timestamp, close=100)
        portfolio = self._portfolio(timestamp, cash=100)

        decision = FixedNotionalSizer(notional=1000, commission_rate=0.001).size(signal, portfolio, market)

        self.assertLessEqual(decision.target_notional, 100)
        self.assertAlmostEqual(decision.target_quantity, 100 / 1.001 / 100)

    def test_fixed_fraction_sizer_uses_portfolio_equity(self):
        timestamp = self._timestamp()
        signal = self._signal(timestamp)
        market = self._market(timestamp, close=50)
        portfolio = self._portfolio(timestamp, cash=5000, equity=10000)

        decision = FixedFractionSizer(fraction=0.2).size(signal, portfolio, market)

        self.assertEqual(decision.target_notional, 2000)
        self.assertEqual(decision.target_quantity, 40)

    def test_order_factory_creates_order_intent_from_sizing_decision(self):
        timestamp = self._timestamp()
        signal = self._signal(timestamp)
        sizing_decision = PositionSizingDecision(
            signal=signal,
            target_notional=1000,
            target_quantity=10,
        )

        intent = OrderFactory(order_type="market").create(signal, sizing_decision)

        self.assertEqual(intent.symbol, "BTC/USDT")
        self.assertEqual(intent.side, "buy")
        self.assertEqual(intent.quantity, 10)
        self.assertEqual(intent.order_type, "market")
        self.assertEqual(intent.source_signal_id, signal.signal_id)

    async def test_sizing_risk_policy_assigns_quantity_without_signal_metadata_quantity(self):
        timestamp = self._timestamp()
        signal = self._signal(timestamp)
        market = self._market(timestamp, close=100)
        portfolio = self._portfolio(timestamp, cash=10000, equity=10000)

        decision = await fixed_fraction_sizing_policy(fraction=0.1).evaluate(signal, portfolio, market)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.target_quantity, 10)
        self.assertNotIn("quantity", signal.metadata)

    def _timestamp(self):
        return datetime(2025, 1, 1, tzinfo=timezone.utc)

    def _signal(self, timestamp, side="buy"):
        return StrategySignal(
            signal_id=f"test:BTC/USDT:{side}:1",
            symbol="BTC/USDT",
            timestamp=timestamp,
            side=side,
        )

    def _market(self, timestamp, close):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, close, close, close, close, 10)
            },
        )

    def _portfolio(self, timestamp, cash, equity=None):
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=cash,
            positions={},
            market_prices={"BTC/USDT": 100},
            equity=cash if equity is None else equity,
        )


if __name__ == "__main__":
    unittest.main()
