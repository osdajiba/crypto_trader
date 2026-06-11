import importlib
import asyncio
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd


class BacktestTradingModeTest(unittest.IsolatedAsyncioTestCase):
    def _load_backtest_module(self):
        for module_name in [
            "src.trading.modes.backtest",
            "src.trading.modes.base",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            return importlib.import_module("src.trading.modes.backtest")

    def _make_mode(self):
        backtest_module = self._load_backtest_module()
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        data_manager = Mock()
        strategy_factory = Mock()
        risk_manager = Mock()
        risk_manager.validate_signals = AsyncMock(side_effect=lambda signals: signals)
        risk_manager.is_risk_breached.return_value = False
        performance_monitor = Mock()
        performance_monitor.close = AsyncMock()

        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = backtest_module.BacktestTradingMode(
                config=config,
                data_manager=data_manager,
                strategy_factory=strategy_factory,
                risk_manager=risk_manager,
                performance_monitor=performance_monitor,
            )
        mode._init_state()
        mode.start_date = "2025-01-01"
        mode.end_date = "2025-01-02"
        return mode, backtest_module

    async def test_shutdown_closes_data_manager(self):
        backtest_module = self._load_backtest_module()

        data_manager = Mock()
        data_manager.close = AsyncMock()
        strategy_factory = Mock()
        risk_manager = Mock()
        performance_monitor = Mock()
        performance_monitor.close = AsyncMock()
        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = backtest_module.BacktestTradingMode(
                config=Mock(),
                data_manager=data_manager,
                strategy_factory=strategy_factory,
                risk_manager=risk_manager,
                performance_monitor=performance_monitor,
            )
        mode.strategy = None

        await mode.shutdown()

        data_manager.close.assert_awaited_once_with()
        self.assertFalse(hasattr(mode, "execution_engine"))

    async def test_base_mode_no_longer_exposes_legacy_dataframe_market_data_entrypoint(self):
        mode, _ = self._make_mode()

        self.assertFalse(hasattr(mode, "_process_market_data"))

    async def test_generate_report_counts_direction_enum_trades(self):
        mode, backtest_module = self._make_mode()
        order_module = importlib.import_module("src.trading.execution.order")
        mode.state["trades"] = [
            {"action": order_module.Direction.BUY},
            {"action": order_module.Direction.SELL},
        ]

        report = mode._generate_report()

        self.assertEqual(report["buy_trades"], 1)
        self.assertEqual(report["sell_trades"], 1)

    async def test_generate_report_delegates_to_report_use_case(self):
        mode, _ = self._make_mode()
        report_use_case = Mock()
        report_use_case.generate.return_value = {"status": "reported"}

        with patch("src.trading.modes.base.TradingReportUseCase", return_value=report_use_case) as report_use_case_class:
            report = mode._generate_report()

        report_use_case_class.assert_called_once_with(mode)
        report_use_case.generate.assert_called_once_with()
        self.assertEqual(report, {"status": "reported"})

    async def test_save_report_delegates_to_report_use_case(self):
        mode, _ = self._make_mode()
        report_use_case = Mock()
        report = {"status": "reported"}

        with patch("src.trading.modes.base.TradingReportUseCase", return_value=report_use_case) as report_use_case_class:
            mode._save_report(report)

        report_use_case_class.assert_called_once_with(mode)
        report_use_case.save.assert_called_once_with(report)

    async def test_execute_trading_loop_delegates_to_backtest_use_case(self):
        asyncio.get_running_loop().slow_callback_duration = 60
        mode, backtest_module = self._make_mode()
        use_case = Mock()
        use_case.run = AsyncMock(return_value={"status": "delegated"})

        with patch.object(backtest_module, "BacktestUseCase", return_value=use_case) as use_case_class:
            result = await mode._execute_trading_loop(["BTC/USDT"], "1m")

        use_case_class.assert_called_once_with(mode)
        use_case.run.assert_awaited_once_with(["BTC/USDT"], "1m")
        self.assertEqual(result, {"status": "delegated"})

    async def test_prepare_run_builds_market_data_feed_from_local_store(self):
        mode, _ = self._make_mode()
        mode._load_historical_data = AsyncMock(return_value={
            "BTC/USDT": pd.DataFrame([{
                "datetime": pd.Timestamp("2025-01-01 00:00:00", tz="UTC"),
                "close": 100.0,
            }])
        })
        local_source = Mock()
        local_source.historical_store = Mock()
        mode.data_manager.primary_source = local_source

        await mode._prepare_run(["BTC/USDT"], "1m")

        self.assertIs(mode.market_data_feed.store, local_source.historical_store)
        self.assertFalse(hasattr(mode, "execution_engine"))

    async def test_initialize_does_not_create_legacy_execution_engine(self):
        backtest_module = self._load_backtest_module()
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        strategy_factory = Mock()
        strategy_factory.create = AsyncMock()
        risk_manager = Mock(spec=backtest_module.BacktestRiskManager)
        risk_manager.initialize = AsyncMock()

        with patch("src.trading.modes.base.LogManager.get_logger", return_value=Mock()):
            mode = backtest_module.BacktestTradingMode(
                config=config,
                data_manager=Mock(),
                strategy_factory=strategy_factory,
                risk_manager=risk_manager,
                performance_monitor=Mock(),
        )

        self.assertFalse(hasattr(backtest_module, "ExecutionEngine"))
        self.assertFalse(hasattr(mode, "_create_legacy_execution_engine"))

        await mode.initialize()

        self.assertFalse(hasattr(mode, "execution_engine"))
        self.assertTrue(mode._running)


if __name__ == "__main__":
    unittest.main()
