import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from common.config import ConfigManager
from common.logging import LogManager


class FakeVar:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class GuiConfigFormTest(unittest.TestCase):
    def setUp(self):
        LogManager(ConfigManager())
        patcher = patch("src.ui.gui.tk.StringVar", FakeVar)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("src.ui.gui.tk.DoubleVar", FakeVar)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("src.ui.gui.tk.IntVar", FakeVar)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch("src.ui.gui.tk.BooleanVar", FakeVar)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_init_from_config_updates_existing_variables(self):
        from src.ui.gui import TradingSystemGUI

        gui = TradingSystemGUI.__new__(TradingSystemGUI)
        gui.config = _fake_config(Path("conf/config.yaml"))
        existing_config_var = FakeVar("old.yaml")
        gui.config_var = existing_config_var

        TradingSystemGUI._init_from_config(gui)

        self.assertIs(gui.config_var, existing_config_var)
        self.assertEqual(Path(gui.config_var.get()), Path("conf/config.yaml"))
        self.assertEqual(gui.mode_var.get(), "backtest")
        self.assertEqual(gui.symbol_var.get(), "BTC/USDT")

    def test_init_from_args_ignores_config_manager_object(self):
        from src.ui.gui import TradingSystemGUI

        gui = TradingSystemGUI.__new__(TradingSystemGUI)
        gui.config_var = FakeVar("conf/config.yaml")
        gui.mode_var = FakeVar("backtest")
        gui.backtest_engine_var = FakeVar("ohlcv")
        gui.strategy_var = FakeVar("dual_ma")
        gui.symbol_var = FakeVar("BTC/USDT")
        gui.timeframe_var = FakeVar("1m")
        gui.start_date_var = FakeVar("2025-01-01")
        gui.end_date_var = FakeVar("2025-01-02")
        gui.debug_var = FakeVar(False)
        gui.db_url_var = FakeVar("")
        gui.max_workers_var = FakeVar(4)

        TradingSystemGUI._init_from_args(gui, types.SimpleNamespace(config=ConfigManager()))

        self.assertEqual(gui.config_var.get(), "conf/config.yaml")


def _fake_config(path: Path):
    values = {
        ("backtest", "engine"): "ohlcv",
        ("system", "operational_mode"): "backtest",
        ("trading", "instruments"): ["BTC/USDT"],
        ("trading", "capital", "initial"): 100000,
        ("backtest", "period", "start"): "2025-01-01",
        ("backtest", "period", "end"): "2025-01-02",
        ("backtest", "costs", "commission"): 0.001,
        ("backtest", "costs", "slippage"): 0.001,
        ("strategy", "active"): "dual_ma",
        ("strategy", "timeframe"): "1m",
        ("strategy", "fallback_strategy"): "",
        ("strategy", "parameters", "fast_period"): 50,
        ("strategy", "parameters", "slow_period"): 200,
        ("strategy", "parameters", "threshold"): 0.005,
        ("risk", "stop_loss", "enabled"): True,
        ("risk", "stop_loss", "percentage"): 0.02,
        ("risk", "stop_loss", "max_daily"): 0.05,
        ("risk", "take_profit", "enabled"): True,
        ("risk", "take_profit", "percentage"): 0.05,
        ("risk", "drawdown", "max_portfolio"): 0.2,
        ("risk", "drawdown", "max_single_trade"): 0.1,
        ("trading", "limits", "position"): 0.1,
        ("trading", "capital", "min_order"): 0.001,
        ("database", "url"): "sqlite:///database/trading.sqlite",
        ("system", "performance", "max_threads"): 4,
        ("system", "performance", "max_memory"): 1073741824,
        ("system", "timezone"): "UTC",
        ("system", "performance", "task_timeout"): 300,
        ("database", "pool_size"): 5,
        ("database", "timeout"): 30,
        ("database", "echo_sql"): False,
        ("live", "safeguards", "emergency_stop", "enabled"): True,
        ("live", "safeguards", "emergency_stop", "threshold"): 0.1,
        ("live", "safeguards", "max_retries"): 3,
        ("live", "safeguards", "cooldown"): 60,
        ("api", "binance", "api_key"): "",
        ("api", "binance", "secret"): "",
        ("api", "timeout"): 30000,
        ("api", "rate_limits", "requests_per_minute"): 20,
        ("api", "retries", "max_attempts"): 3,
        ("api", "retries", "delay_seconds"): 1,
        ("proxies", "http"): "",
        ("proxies", "https"): "",
    }

    config = Mock()
    config._config_path = path

    def get(*keys, default=None):
        return values.get(tuple(keys), default)

    config.get.side_effect = get
    return config


if __name__ == "__main__":
    unittest.main()
