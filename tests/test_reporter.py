import unittest
from datetime import datetime, timezone

from src.domain.models import Fill, PortfolioSnapshot


class InMemoryReporterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_report_uses_recorded_fills_and_snapshots(self):
        from src.application.reporting import InMemoryReporter

        reporter = InMemoryReporter()
        first = datetime(2025, 1, 1, tzinfo=timezone.utc)
        second = datetime(2025, 1, 2, tzinfo=timezone.utc)

        await reporter.record_fill(Fill("fill-1", "intent-1", "BTC/USDT", "buy", first, 0.01, 100000, 1))
        await reporter.record_fill(Fill("fill-2", "intent-2", "BTC/USDT", "sell", second, 0.01, 99800, 1))
        await reporter.record_snapshot(PortfolioSnapshot(first, 99000, {"BTC/USDT": 0.01}, {"BTC/USDT": 100000}, 100000))
        await reporter.record_snapshot(PortfolioSnapshot(second, 99800, {}, {"BTC/USDT": 99800}, 99800))

        report = reporter.generate_report(initial_capital=100000)

        self.assertEqual(report["initial_capital"], 100000)
        self.assertEqual(report["final_equity"], 99800)
        self.assertEqual(report["total_return"], -200)
        self.assertEqual(report["total_return_pct"], -0.2)
        self.assertEqual(report["max_drawdown_pct"], 0.2)
        self.assertEqual(report["total_trades"], 2)
        self.assertEqual(report["buy_trades"], 1)
        self.assertEqual(report["sell_trades"], 1)
        self.assertEqual(report["current_positions"], {})
        self.assertEqual(report["remaining_cash"], 99800)
        self.assertEqual(len(report["trades"]), 2)
        self.assertEqual(len(report["equity_curve"]), 2)


if __name__ == "__main__":
    unittest.main()
