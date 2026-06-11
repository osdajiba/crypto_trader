import importlib
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch


class PaperTradingModeTest(unittest.IsolatedAsyncioTestCase):
    def _load_paper_module(self):
        for module_name in [
            "src.trading.modes.paper",
            "src.trading.modes.base",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            return importlib.import_module("src.trading.modes.paper")

    def _make_mode(self):
        paper_module = self._load_paper_module()
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        data_manager = Mock()
        strategy_factory = Mock()
        risk_manager = Mock()
        risk_manager.is_risk_breached.side_effect = [False, True]
        risk_manager.execute_risk_control = AsyncMock(return_value=False)
        performance_monitor = Mock()

        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = paper_module.PaperTradingMode(
                config=config,
                data_manager=data_manager,
                strategy_factory=strategy_factory,
                risk_manager=risk_manager,
                performance_monitor=performance_monitor,
            )
        mode._running = True
        return mode

    async def test_execute_trading_loop_delegates_to_paper_use_case(self):
        mode = self._make_mode()
        mode.paper_use_case = Mock()
        mode.paper_use_case.run_once = AsyncMock(return_value=Mock(fills=[]))
        mode._process_market_data = AsyncMock(side_effect=AssertionError("legacy DataFrame pipeline should not be used"))
        mode._sleep_interval = AsyncMock()

        result = await mode._execute_trading_loop(["BTC/USDT"], "1m")

        self.assertEqual(result, {})
        mode.paper_use_case.run_once.assert_awaited_once_with(["BTC/USDT"], "1m")
        mode._process_market_data.assert_not_called()
        mode._sleep_interval.assert_awaited_once_with()

    async def test_prepare_run_builds_domain_pipeline(self):
        mode = self._make_mode()
        mode.strategy = Mock()
        mode.strategy.__class__.__name__ = "FakeStrategy"
        mode.risk_manager.validate_signals = AsyncMock(side_effect=lambda signals: signals)

        await mode._prepare_run(["BTC/USDT"], "1m")

        self.assertIsNotNone(mode.market_data_feed)
        self.assertIsNotNone(mode.domain_pipeline)
        self.assertIsNotNone(mode.paper_use_case)

    async def test_initialize_does_not_create_legacy_execution_engine(self):
        paper_module = self._load_paper_module()
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        strategy_factory = Mock()
        strategy_factory.create = AsyncMock()

        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = paper_module.PaperTradingMode(
                config=config,
                data_manager=Mock(),
                strategy_factory=strategy_factory,
                risk_manager=Mock(),
                performance_monitor=Mock(),
        )

        self.assertFalse(hasattr(paper_module, "ExecutionEngine"))
        self.assertFalse(hasattr(mode, "_create_legacy_execution_engine"))

        await mode.initialize()

        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertTrue(mode._running)


if __name__ == "__main__":
    unittest.main()
