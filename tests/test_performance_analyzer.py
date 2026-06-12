import unittest
from datetime import datetime, timedelta, timezone

from src.domain.models import Fill, PortfolioSnapshot


class PerformanceAnalyzerTests(unittest.TestCase):
    def test_analyzer_calculates_report_fields_from_fills_and_snapshots(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        first = datetime(2025, 1, 1, tzinfo=timezone.utc)
        second = datetime(2025, 1, 2, tzinfo=timezone.utc)
        fills = [
            Fill("fill-1", "intent-1", "BTC/USDT", "buy", first, 0.01, 100000, 1),
            Fill("fill-2", "intent-2", "BTC/USDT", "sell", second, 0.01, 110000, 1),
        ]
        snapshots = [
            PortfolioSnapshot(first, 99000, {"BTC/USDT": 0.01}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(second, 110000, {}, {"BTC/USDT": 110000}, 110000),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=fills,
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(report["initial_capital"], 100000)
        self.assertEqual(report["final_equity"], 110000)
        self.assertEqual(report["total_return"], 10000)
        self.assertEqual(report["total_return_pct"], 10)
        self.assertEqual(report["max_drawdown_pct"], 0)
        self.assertEqual(report["total_trades"], 2)
        self.assertEqual(report["buy_trades"], 1)
        self.assertEqual(report["sell_trades"], 1)
        self.assertEqual(report["current_positions"], {})
        self.assertEqual(report["remaining_cash"], 110000)
        self.assertEqual(len(report["trades"]), 2)
        self.assertEqual(len(report["equity_curve"]), 2)
        self.assertIn("costs", report)
        self.assertIn("diagnostics", report)
        self.assertEqual(report["diagnostics"]["net_return"], 10000)
        self.assertEqual(report["diagnostics"]["total_trades"], 2)
        self.assertEqual(report["diagnostics"]["turnover_pct_of_initial"], 2.1)

    def test_analyzer_includes_backtest_quality_summary(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        first = datetime(2025, 1, 1, tzinfo=timezone.utc)
        second = datetime(2025, 1, 2, tzinfo=timezone.utc)
        fills = [
            Fill("fill-1", "intent-1", "BTC/USDT", "buy", first, 0.01, 100000, 1),
            Fill("fill-2", "intent-2", "BTC/USDT", "sell", second, 0.01, 110000, 1),
        ]
        snapshots = [
            PortfolioSnapshot(first, 99000, {"BTC/USDT": 0.01}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(second, 110000, {}, {"BTC/USDT": 110000}, 110000),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=fills,
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(
            report["quality"],
            {
                "snapshot_count": 2,
                "trade_count": 2,
                "first_timestamp": first,
                "last_timestamp": second,
                "time_order_valid": True,
                "has_negative_equity": False,
                "has_open_positions": False,
                "duplicate_timestamp_count": 0,
                "expected_interval_seconds": 86400.0,
                "gap_count": 0,
                "max_gap_seconds": 0,
                "fills_outside_snapshot_range": 0,
                "invalid_fill_quantity_count": 0,
                "invalid_fill_price_count": 0,
                "invalid_fill_commission_count": 0,
                "invalid_fill_notional_count": 0,
                "invalid_fill_count": 0,
            },
        )

    def test_analyzer_quality_summary_flags_out_of_order_snapshots_and_open_positions(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        first = datetime(2025, 1, 1, tzinfo=timezone.utc)
        second = datetime(2025, 1, 2, tzinfo=timezone.utc)
        snapshots = [
            PortfolioSnapshot(second, 99000, {"BTC/USDT": 0.01}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(first, 99000, {"BTC/USDT": 0.01}, {"BTC/USDT": 100000}, -1),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=[],
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(report["quality"]["snapshot_count"], 2)
        self.assertFalse(report["quality"]["time_order_valid"])
        self.assertTrue(report["quality"]["has_negative_equity"])
        self.assertTrue(report["quality"]["has_open_positions"])

    def test_analyzer_quality_summary_flags_duplicate_gaps_and_fills_outside_snapshot_range(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        one_minute = timedelta(minutes=1)
        snapshots = [
            PortfolioSnapshot(start, 100000, {}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(start + one_minute, 100000, {}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(start + one_minute, 100000, {}, {"BTC/USDT": 100000}, 100000),
            PortfolioSnapshot(start + timedelta(minutes=4), 100000, {}, {"BTC/USDT": 100000}, 100000),
        ]
        fills = [
            Fill("fill-before", "intent-1", "BTC/USDT", "buy", start - one_minute, 0.01, 100000, 1),
            Fill("fill-inside", "intent-2", "BTC/USDT", "sell", start + one_minute, 0.01, 100000, 1),
            Fill("fill-after", "intent-3", "BTC/USDT", "buy", start + timedelta(minutes=5), 0.01, 100000, 1),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=fills,
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(report["quality"]["duplicate_timestamp_count"], 1)
        self.assertEqual(report["quality"]["expected_interval_seconds"], 60)
        self.assertEqual(report["quality"]["gap_count"], 1)
        self.assertEqual(report["quality"]["max_gap_seconds"], 180)
        self.assertEqual(report["quality"]["fills_outside_snapshot_range"], 2)

    def test_analyzer_quality_summary_flags_invalid_fill_constraints(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        snapshots = [
            PortfolioSnapshot(timestamp, 100000, {}, {"BTC/USDT": 100000}, 100000),
        ]
        fills = [
            Fill("zero-qty", "intent-1", "BTC/USDT", "buy", timestamp, 0, 100000, 1),
            Fill("negative-price", "intent-2", "BTC/USDT", "sell", timestamp, 0.01, -1, 1),
            Fill("negative-commission", "intent-3", "BTC/USDT", "buy", timestamp, 0.01, 100000, -1),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=fills,
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(report["quality"]["invalid_fill_quantity_count"], 1)
        self.assertEqual(report["quality"]["invalid_fill_price_count"], 1)
        self.assertEqual(report["quality"]["invalid_fill_commission_count"], 1)
        self.assertEqual(report["quality"]["invalid_fill_notional_count"], 2)
        self.assertEqual(report["quality"]["invalid_fill_count"], 3)

    def test_analyzer_includes_transaction_cost_explainability(self):
        from src.reporting.performance_analyzer import PerformanceAnalyzer

        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        snapshots = [
            PortfolioSnapshot(timestamp, 100000, {}, {"BTC/USDT": 100000}, 100000),
        ]
        fills = [
            Fill("fill-1", "intent-1", "BTC/USDT", "buy", timestamp, 2, 100, 0.2, slippage=0.001),
            Fill("fill-2", "intent-2", "BTC/USDT", "sell", timestamp, 1, 110, 0.11, slippage=0.002),
        ]

        report = PerformanceAnalyzer().analyze(
            fills=fills,
            snapshots=snapshots,
            initial_capital=100000,
        )

        self.assertEqual(report["costs"]["fill_count"], 2)
        self.assertEqual(report["costs"]["total_notional"], 310)
        self.assertAlmostEqual(report["costs"]["total_commission"], 0.31)
        self.assertAlmostEqual(report["costs"]["average_commission_per_fill"], 0.155)
        self.assertAlmostEqual(report["costs"]["commission_rate_bps"], 10.0)
        self.assertAlmostEqual(report["costs"]["estimated_slippage_cost"], 0.42)
        self.assertAlmostEqual(report["costs"]["average_slippage_bps"], 13.548387096774194)
        self.assertAlmostEqual(report["costs"]["max_slippage_bps"], 20.0)
        self.assertAlmostEqual(report["costs"]["total_transaction_cost"], 0.73)


if __name__ == "__main__":
    unittest.main()
