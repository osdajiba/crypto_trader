import datetime
import unittest

from src.domain.models import MarketBar, MarketSlice, OrderIntent
from src.domain.portfolio import PortfolioBook


class FakeExchangeClient:
    def __init__(self):
        self.orders = []

    def create_order(self, symbol, direction, order_type, quantity, price=None):
        self.orders.append({
            "symbol": symbol,
            "direction": direction,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
        })
        return {
            "id": "exchange-order-1",
            "filled": quantity,
            "average": 101.0,
            "fee": {"cost": 0.101},
            "status": "filled",
            "timestamp": 1735689600000,
        }


class LiveExecutionModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_exchange_response_becomes_domain_fill(self):
        from src.trading.execution.live_model import LiveExecutionModel

        timestamp = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
        exchange = FakeExchangeClient()
        model = LiveExecutionModel(exchange)

        fill = await model.execute(
            OrderIntent("intent-1", "BTC/USDT", "buy", 0.01, "market", timestamp),
            self._market(timestamp),
            PortfolioBook(initial_cash=100000).snapshot(timestamp),
        )

        self.assertEqual(fill.fill_id, "exchange-order-1")
        self.assertEqual(fill.order_intent_id, "intent-1")
        self.assertEqual(fill.side, "buy")
        self.assertAlmostEqual(fill.quantity, 0.01)
        self.assertAlmostEqual(fill.price, 101.0)
        self.assertAlmostEqual(fill.commission, 0.101)
        self.assertEqual(exchange.orders[0]["order_type"], "market")

    def _market(self, timestamp):
        return MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100.0, 101.0, 99.0, 100.0, 10.0)
            },
        )


if __name__ == "__main__":
    unittest.main()
