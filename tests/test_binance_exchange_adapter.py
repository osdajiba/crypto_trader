import unittest
import importlib
import sys
from unittest.mock import Mock
from unittest.mock import patch

from src.trading.execution.order import Direction


class BinanceExchangeAdapterTest(unittest.TestCase):
    def _make_adapter(self, exchange):
        for module_name in [
            "src.exchange.adapters.binance",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("src.common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("common.logging.LogManager.get_logger", return_value=Mock()):
            binance_module = importlib.import_module("src.exchange.adapters.binance")

        adapter = object.__new__(binance_module.Binance)
        adapter.exchange = exchange
        return adapter

    def test_create_order_delegates_to_ccxt_exchange(self):
        exchange = Mock()
        exchange.create_order.return_value = {"id": "order-1", "filled": 0.1, "average": 100}
        adapter = self._make_adapter(exchange)

        response = adapter.create_order(
            symbol="BTC/USDT",
            direction=Direction.BUY,
            order_type="limit",
            quantity=0.1,
            price=100,
        )

        exchange.create_order.assert_called_once_with(
            symbol="BTC/USDT",
            type="limit",
            side="buy",
            amount=0.1,
            price=100,
        )
        self.assertEqual(response["id"], "order-1")

    def test_get_account_balance_returns_total_balances(self):
        exchange = Mock()
        exchange.fetch_balance.return_value = {"total": {"USDT": 1000, "BTC": 0.5}}
        adapter = self._make_adapter(exchange)

        balance = adapter.get_account_balance()

        exchange.fetch_balance.assert_called_once_with()
        self.assertEqual(balance, {"USDT": 1000, "BTC": 0.5})

    def test_get_open_orders_delegates_to_exchange(self):
        exchange = Mock()
        exchange.fetch_open_orders.return_value = [{"id": "order-1"}]
        adapter = self._make_adapter(exchange)

        orders = adapter.get_open_orders()

        exchange.fetch_open_orders.assert_called_once_with()
        self.assertEqual(orders, [{"id": "order-1"}])

    def test_cancel_order_delegates_to_exchange_with_optional_symbol(self):
        exchange = Mock()
        exchange.cancel_order.return_value = {"id": "order-1", "status": "canceled"}
        adapter = self._make_adapter(exchange)

        result = adapter.cancel_order("order-1", "BTC/USDT")

        exchange.cancel_order.assert_called_once_with("order-1", "BTC/USDT")
        self.assertEqual(result["status"], "canceled")


if __name__ == "__main__":
    unittest.main()
