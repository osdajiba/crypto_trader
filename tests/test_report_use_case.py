import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd


class TradingReportUseCaseTest(unittest.TestCase):
    def _make_config(self, report_dir):
        config = Mock()
        strategy_charts_dir = Path(report_dir).parent / "final"

        def get(*keys, default=None):
            if keys == ("backtest", "initial_capital"):
                return 100000
            if keys == ("reporting", "backtest_reports_dir"):
                return str(report_dir)
            if keys == ("reporting", "strategy_charts_dir"):
                return str(strategy_charts_dir)
            if keys == ("reporting", "output_formats"):
                return ["json"]
            return default

        config.get.side_effect = get
        return config

    def _make_context(self, report_dir):
        config = self._make_config(report_dir)
        context = SimpleNamespace(
            mode_name="backtest",
            config=config,
            logger=Mock(),
            state={
                "cash": 99800.0,
                "positions": {},
                "trades": [
                    {"action": SimpleNamespace(value="buy")},
                    {"action": SimpleNamespace(value="sell")},
                ],
                "equity_curve": [],
                "max_drawdown": 0.02,
            },
        )
        context._calculate_equity = Mock(return_value=99800.0)
        context._add_mode_specific_metrics = Mock(side_effect=lambda report: report.update({"strategy": "DualMAStrategy"}))
        return context

    def test_generate_report_preserves_current_summary_fields(self):
        from src.application.report_use_case import TradingReportUseCase

        with tempfile.TemporaryDirectory() as tmp:
            context = self._make_context(Path(tmp))

            report = TradingReportUseCase(context).generate()

        self.assertEqual(report["initial_capital"], 100000)
        self.assertEqual(report["final_equity"], 99800.0)
        self.assertEqual(report["total_return"], -200.0)
        self.assertEqual(report["total_return_pct"], -0.2)
        self.assertEqual(report["max_drawdown_pct"], 2.0)
        self.assertEqual(report["total_trades"], 2)
        self.assertEqual(report["buy_trades"], 1)
        self.assertEqual(report["sell_trades"], 1)
        self.assertEqual(report["current_positions"], {})
        self.assertEqual(report["remaining_cash"], 99800.0)
        self.assertEqual(report["strategy"], "DualMAStrategy")
        self.assertIn("diagnostics", report)
        self.assertEqual(report["diagnostics"]["net_return"], -200.0)
        self.assertEqual(report["diagnostics"]["total_trades"], 2)

    def test_save_writes_json_report_to_configured_backtest_directory(self):
        from src.application.report_use_case import TradingReportUseCase

        with tempfile.TemporaryDirectory() as tmp:
            report_dir = Path(tmp) / "reports" / "backtest"
            context = self._make_context(report_dir)
            report = TradingReportUseCase(context).generate()

            TradingReportUseCase(context).save(report)

            report_files = list(report_dir.glob("backtest_report_*.json"))
            self.assertEqual(len(report_files), 1)
            saved = json.loads(report_files[0].read_text())

        self.assertEqual(saved["final_equity"], 99800.0)
        self.assertEqual(saved["total_trades"], 2)
        self.assertEqual(report["artifacts"]["report_paths"], [str(report_files[0])])
        context.logger.info.assert_any_call(f"Performance report saved to {report_files[0]}")

    def test_save_writes_strategy_chart_when_backtest_market_data_is_available(self):
        from src.application.report_use_case import TradingReportUseCase

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            report_dir = Path(tmp) / "reports" / "backtest"
            context = self._make_context(report_dir)
            context.historical_data = {
                "BTC/USDT": pd.DataFrame({
                    "datetime": pd.date_range("2025-01-01", periods=80, freq="1min", tz="UTC"),
                    "close": list(range(80)),
                })
            }
            context.start_date = "2025-01-01"
            context.end_date = "2025-01-02"
            report = TradingReportUseCase(context).generate()

            TradingReportUseCase(context).save(report)

            chart_path = Path(tmp) / "reports" / "final" / "ma_ema_crossovers_20250101_20250102.png"
            self.assertTrue(chart_path.exists())
            self.assertGreater(chart_path.stat().st_size, 0)
            self.assertEqual(report["artifacts"]["strategy_chart_path"], str(chart_path))
        context.logger.info.assert_any_call(f"Strategy chart saved to {chart_path}")

    def test_generate_report_prefers_domain_reporter_without_mode_state(self):
        from src.application.report_use_case import TradingReportUseCase

        with tempfile.TemporaryDirectory() as tmp:
            context = SimpleNamespace(
                mode_name="backtest",
                config=self._make_config(Path(tmp)),
                logger=Mock(),
                domain_reporter=Mock(),
            )
            context.domain_reporter.fills = []
            context.domain_reporter.snapshots = []
            context._add_mode_specific_metrics = Mock(side_effect=lambda report: report.update({"strategy": "DomainStrategy"}))

            report = TradingReportUseCase(context).generate()

        self.assertEqual(report["final_equity"], 100000)
        self.assertEqual(report["total_trades"], 0)
        self.assertEqual(report["strategy"], "DomainStrategy")


if __name__ == "__main__":
    unittest.main()
