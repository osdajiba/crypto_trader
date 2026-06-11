import unittest
import warnings
from unittest.mock import Mock, patch

import pandas as pd

from src.backtest.performance import PerformanceMonitor


class PerformanceMonitorTest(unittest.TestCase):
    def test_update_equity_curve_accepts_timezone_aware_timestamp_without_warning(self):
        timestamp = pd.Timestamp("2023-01-01 00:00:00", tz="UTC")

        with patch("src.backtest.performance.LogManager.get_logger", return_value=Mock()):
            monitor = PerformanceMonitor(config={})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            monitor.update_equity_curve(timestamp, 100000)

        timezone_warnings = [
            warning for warning in caught
            if "no explicit representation of timezones" in str(warning.message)
        ]
        self.assertEqual(timezone_warnings, [])

    def test_calculate_metrics_uses_default_risk_free_rate_with_config_manager_style_get(self):
        timestamp = pd.Timestamp("2023-01-01 00:00:00")
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default

        with patch("src.backtest.performance.LogManager.get_logger", return_value=Mock()):
            monitor = PerformanceMonitor(config=config)

        monitor.update_equity_curve(timestamp, 100000)
        monitor.update_equity_curve(timestamp + pd.Timedelta(days=1), 100100)

        monitor.calculate_performance_metrics()

    def test_detailed_report_trade_log_uses_string_keys(self):
        timestamp = pd.Timestamp("2023-01-01 00:00:00")

        with patch("src.backtest.performance.LogManager.get_logger", return_value=Mock()):
            monitor = PerformanceMonitor(config={})

        monitor.record_trade(
            timestamp=timestamp,
            symbol="BTC/USDT",
            direction="buy",
            entry_price=20000,
            exit_price=20100,
            quantity=0.05,
            commission=1,
        )

        report = monitor.generate_detailed_report()

        self.assertIn("timestamp", report["trade_log"][0])
        self.assertTrue(all(isinstance(key, str) for key in report["trade_log"][0]))


if __name__ == "__main__":
    unittest.main()
