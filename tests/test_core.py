import importlib
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch


class TradingCoreTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_pipeline_does_not_shutdown_mode_twice(self):
        for module_name in [
            "src.core.core",
            "src.trading.modes.base",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            core_module = importlib.import_module("src.core.core")

        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        mode = Mock()
        mode.shutdown = AsyncMock(side_effect=self._shutdown_mode(mode))
        mode.run = AsyncMock(side_effect=self._run_and_shutdown_mode(mode))
        mode.is_running = True

        with patch.object(core_module.LogManager, "get_logger", return_value=Mock()), \
             patch.object(core_module, "TradingModeFactory") as factory_class, \
             patch.object(core_module, "AsyncExecutor", return_value=Mock(close=AsyncMock())):
            factory = Mock()
            factory.get_available_modes.return_value = {"backtest": object()}
            factory.create = AsyncMock(return_value=mode)
            factory_class.return_value = factory
            core = core_module.TradingCore(config=config, mode="backtest", backtest_engine="ohlcv")
            await core.run_pipeline()

        mode.shutdown.assert_awaited_once_with()

    def _run_and_shutdown_mode(self, mode):
        async def run(symbols, timeframe):
            await mode.shutdown()
            return {"status": "ok"}
        return run

    def _shutdown_mode(self, mode):
        async def shutdown():
            mode.is_running = False
        return shutdown


if __name__ == "__main__":
    unittest.main()
