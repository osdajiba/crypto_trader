import importlib
import sys
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch


class LiveTradingModeTest(unittest.IsolatedAsyncioTestCase):
    def _load_live_module(self):
        for module_name in [
            "src.trading.modes.live",
            "src.trading.modes.base",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            return importlib.import_module("src.trading.modes.live")

    def _make_mode(self, config_get):
        live_module = self._load_live_module()
        config = Mock()
        config.get.side_effect = config_get
        data_manager = Mock()
        strategy_factory = Mock()
        strategy_factory.create = AsyncMock()
        risk_manager = Mock()
        performance_monitor = Mock()

        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = live_module.LiveTradingMode(
                config=config,
                data_manager=data_manager,
                strategy_factory=strategy_factory,
                risk_manager=risk_manager,
                performance_monitor=performance_monitor,
            )
        return mode, live_module

    async def test_initialize_rejects_live_mode_without_explicit_enable(self):
        mode, live_module = self._make_mode(lambda *keys, default=None: default)
        self.assertFalse(hasattr(live_module, "ExecutionEngine"))

        with self.assertRaisesRegex(PermissionError, "Live trading is disabled"):
            await mode.initialize()

        self.assertFalse(mode._running)

    async def test_initialize_requires_confirmation_even_when_enabled(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        self.assertFalse(hasattr(live_module, "ExecutionEngine"))

        with self.assertRaisesRegex(PermissionError, "confirm_live_trading"):
            await mode.initialize()

        self.assertFalse(mode._running)

    async def test_initialize_continues_when_live_safety_flags_are_enabled(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            if keys == ("live_trading", "confirm_live_trading"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        mode._verify_exchange_connectivity = AsyncMock(return_value=True)
        mode._verify_account = AsyncMock(return_value=True)

        exchange_client = Mock()
        with patch.object(live_module.LiveTradingMode, "_create_exchange_client", return_value=exchange_client) as create_client:
            await mode.initialize()

        create_client.assert_called_once_with()
        mode.strategy_factory.create.assert_awaited_once()
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertIs(mode.exchange_client, exchange_client)
        self.assertTrue(mode._running)

    async def test_initialize_rejects_missing_exchange_client_without_legacy_fallback_enabled(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            if keys == ("live_trading", "confirm_live_trading"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        mode._verify_exchange_connectivity = AsyncMock(return_value=True)
        mode._verify_account = AsyncMock(return_value=True)

        with patch.object(live_module.LiveTradingMode, "_create_exchange_client", return_value=None):
            with self.assertRaisesRegex(ConnectionError, "Live exchange client initialization failed"):
                await mode.initialize()

        mode.strategy_factory.create.assert_not_awaited()
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertFalse(hasattr(mode, "exchange_client"))
        self.assertFalse(mode._running)

    async def test_initialize_rejects_missing_exchange_client_even_when_legacy_fallback_flag_enabled(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            if keys == ("live_trading", "confirm_live_trading"):
                return True
            if keys == ("live_trading", "allow_legacy_execution_engine_fallback"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        mode._verify_exchange_connectivity = AsyncMock(return_value=True)
        mode._verify_account = AsyncMock(return_value=True)

        with patch.object(live_module.LiveTradingMode, "_create_exchange_client", return_value=None):
            with self.assertRaisesRegex(ConnectionError, "Live exchange client initialization failed"):
                await mode.initialize()

        mode.strategy_factory.create.assert_not_awaited()
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertFalse(hasattr(mode, "exchange_client"))
        self.assertFalse(mode._running)

    async def test_legacy_live_execution_engine_helper_is_removed(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)

        self.assertFalse(hasattr(mode, "_create_legacy_live_execution_engine"))

    async def test_create_exchange_client_constructs_binance_adapter(self):
        mode, live_module = self._make_mode(lambda *keys, default=None: default)

        exchange_client = Mock()
        with patch.object(live_module, "Binance", return_value=exchange_client) as binance_class:
            created = mode._create_exchange_client()

        binance_class.assert_called_once_with(mode.config)
        self.assertIs(created, exchange_client)

    async def test_initialize_creates_exchange_client_without_legacy_engine_by_default(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            if keys == ("live_trading", "confirm_live_trading"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        exchange_client = Mock()
        mode._verify_exchange_connectivity = AsyncMock(return_value=True)
        mode._verify_account = AsyncMock(return_value=True)

        with patch.object(live_module.LiveTradingMode, "_create_exchange_client", return_value=exchange_client) as create_client:
            await mode.initialize()

        create_client.assert_called_once_with()
        self.assertIs(mode.exchange_client, exchange_client)
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertTrue(mode._running)

    async def test_initialize_prefers_explicit_exchange_client_without_legacy_engine(self):
        def config_get(*keys, default=None):
            if keys == ("live_trading", "enabled"):
                return True
            if keys == ("live_trading", "confirm_live_trading"):
                return True
            return default

        mode, live_module = self._make_mode(config_get)
        exchange_client = Mock()
        mode.exchange_client = exchange_client
        mode._verify_exchange_connectivity = AsyncMock(return_value=True)
        mode._verify_account = AsyncMock(return_value=True)

        self.assertFalse(hasattr(live_module, "ExecutionEngine"))
        await mode.initialize()

        self.assertIs(mode.exchange_client, exchange_client)
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertTrue(mode._running)

    async def test_execute_trading_loop_delegates_to_live_use_case(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)
        mode._running = True
        mode.status_interval = 3600
        mode.state["last_balance_check"] = datetime.now()
        mode.risk_manager.is_risk_breached.side_effect = [False, True]
        mode.risk_manager.execute_risk_control = AsyncMock(return_value=False)
        mode.live_use_case = Mock()
        mode.live_use_case.run_once = AsyncMock(return_value=Mock(fills=[]))
        mode._process_market_data = AsyncMock(side_effect=AssertionError("legacy DataFrame pipeline should not be used"))
        mode._sleep_interval = AsyncMock()

        result = await mode._execute_trading_loop(["BTC/USDT"], "1m")

        self.assertEqual(result, {})
        mode.live_use_case.run_once.assert_awaited_once_with(["BTC/USDT"], "1m")
        mode._process_market_data.assert_not_called()
        mode._sleep_interval.assert_awaited_once_with()

    async def test_prepare_run_builds_domain_pipeline_after_safety_initialized(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)
        mode.strategy = Mock()
        mode.strategy.__class__.__name__ = "FakeStrategy"
        mode.exchange_client = Mock()
        mode.risk_manager.validate_signals = AsyncMock(side_effect=lambda signals: signals)

        await mode._prepare_run(["BTC/USDT"], "1m")

        self.assertIsNotNone(mode.market_data_feed)
        self.assertIsNotNone(mode.domain_pipeline)
        self.assertIsNotNone(mode.live_use_case)

    async def test_verify_account_prefers_exchange_client_balance(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)
        mode.exchange_client = Mock()
        mode.exchange_client.get_account_balance = AsyncMock(return_value={"USDT": 100000})
        verified = await mode._verify_account()

        self.assertTrue(verified)
        mode.exchange_client.get_account_balance.assert_awaited_once_with()
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertEqual(mode.state["current_equity"], 100000)

    async def test_update_account_status_prefers_exchange_client_balance(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)
        mode.exchange_client = Mock()
        mode.exchange_client.get_account_balance = AsyncMock(return_value={"USDT": 90000})
        mode._update_performance_metrics = Mock()

        await mode._update_account_status()

        mode.exchange_client.get_account_balance.assert_awaited_once_with()
        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertEqual(mode.state["current_equity"], 90000)
        mode._update_performance_metrics.assert_called_once_with()

    async def test_shutdown_uses_exchange_client_open_orders_and_cancel(self):
        mode, _ = self._make_mode(lambda *keys, default=None: default)
        mode.exchange_client = Mock()
        mode.exchange_client.get_open_orders = AsyncMock(return_value=[{"id": "order-1", "symbol": "BTC/USDT"}])
        mode.exchange_client.cancel_order = AsyncMock()

        await mode.shutdown()

        mode.exchange_client.get_open_orders.assert_awaited_once_with()
        mode.exchange_client.cancel_order.assert_awaited_once_with("order-1", "BTC/USDT")
        self.assertFalse(hasattr(mode, "execution_engine"))


if __name__ == "__main__":
    unittest.main()
