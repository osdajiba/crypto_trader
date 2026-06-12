import unittest
from datetime import datetime

from src.application.backtest_batch import BacktestRunResult
from src.application.backtest_planning import WalkForwardWindow


class ScanSummaryTests(unittest.TestCase):
    def test_scan_summary_ranks_successful_results_and_keeps_failures(self):
        from src.reporting.scan_summary import summarize_scan_results

        results = [
            BacktestRunResult(
                run_id="run-001",
                parameters={"short_window": 5},
                window=None,
                status="success",
                report={"final_equity": 101000, "total_trades": 3},
            ),
            BacktestRunResult(
                run_id="run-002",
                parameters={"short_window": 10},
                window=None,
                status="failed",
                report={},
                error="boom",
            ),
            BacktestRunResult(
                run_id="run-003",
                parameters={"short_window": 15},
                window=None,
                status="success",
                report={"final_equity": 102000, "total_trades": 4},
            ),
        ]

        summary = summarize_scan_results(results, metric="final_equity")

        self.assertEqual(summary["metric"], "final_equity")
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["best_run_id"], "run-003")
        self.assertEqual(summary["best_metric"], 102000)
        self.assertEqual([row["run_id"] for row in summary["results"]], ["run-003", "run-001", "run-002"])
        self.assertEqual(summary["results"][0]["rank"], 1)
        self.assertEqual(summary["results"][0]["parameters"], {"short_window": 15})
        self.assertEqual(summary["results"][2]["status"], "failed")
        self.assertEqual(summary["results"][2]["error"], "boom")

    def test_scan_summary_handles_no_successful_metric(self):
        from src.reporting.scan_summary import summarize_scan_results

        summary = summarize_scan_results([
            BacktestRunResult(
                run_id="run-001",
                parameters={},
                window=None,
                status="failed",
                report={},
                error="boom",
            )
        ])

        self.assertEqual(summary["success_count"], 0)
        self.assertEqual(summary["failed_count"], 1)
        self.assertIsNone(summary["best_run_id"])
        self.assertIsNone(summary["best_metric"])

    def test_scan_summary_can_rank_by_nested_diagnostics_metric(self):
        from src.reporting.scan_summary import summarize_scan_results

        results = [
            BacktestRunResult(
                run_id="run-001",
                parameters={"short_window": 5},
                window=None,
                status="success",
                report={
                    "final_equity": 99900,
                    "diagnostics": {
                        "cost_to_abs_net_return": 0.95,
                        "gross_return_estimate": -5,
                    },
                },
            ),
            BacktestRunResult(
                run_id="run-002",
                parameters={"short_window": 10},
                window=None,
                status="success",
                report={
                    "final_equity": 100100,
                    "diagnostics": {
                        "cost_to_abs_net_return": 0.25,
                        "gross_return_estimate": 120,
                    },
                },
            ),
            BacktestRunResult(
                run_id="run-003",
                parameters={"short_window": 15},
                window=None,
                status="success",
                report={"final_equity": 100000},
            ),
        ]

        summary = summarize_scan_results(
            results,
            metric="diagnostics.cost_to_abs_net_return",
            descending=False,
        )

        self.assertEqual(summary["best_run_id"], "run-002")
        self.assertEqual(summary["best_metric"], 0.25)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual([row["run_id"] for row in summary["results"]], ["run-002", "run-001", "run-003"])
        self.assertEqual(summary["results"][0]["diagnostics"]["gross_return_estimate"], 120)
        self.assertIsNone(summary["results"][2]["rank"])

    def test_scan_summary_includes_walk_forward_window(self):
        from src.reporting.scan_summary import summarize_scan_results

        window = WalkForwardWindow(
            train_start=datetime(2024, 12, 1),
            train_end=datetime(2025, 1, 1),
            test_start=datetime(2025, 1, 1),
            test_end=datetime(2025, 1, 2),
        )

        summary = summarize_scan_results([
            BacktestRunResult(
                run_id="run-001",
                parameters={},
                window=window,
                status="success",
                report={"final_equity": 100000},
            )
        ])

        self.assertEqual(summary["results"][0]["window"]["test_start"], "2025-01-01T00:00:00")
        self.assertEqual(summary["results"][0]["window"]["test_end"], "2025-01-02T00:00:00")

    def test_scan_summary_includes_traceability_fields(self):
        from src.reporting.scan_summary import summarize_scan_results

        summary = summarize_scan_results([
            BacktestRunResult(
                run_id="run-001",
                parameters={"short_window": 20, "long_window": 50},
                window=None,
                status="success",
                report={
                    "final_equity": 100000,
                    "research": {
                        "config_hash": "abc123",
                        "config_snapshot": {"strategy": {"active": "dual_ma"}},
                        "artifacts": {
                            "report_paths": ["reports/backtest/backtest_report_1.json"],
                            "strategy_chart_path": "reports/final/chart.png",
                        },
                    },
                },
            )
        ])

        row = summary["results"][0]

        self.assertEqual(len(row["run_hash"]), 16)
        self.assertEqual(row["config_hash"], "abc123")
        self.assertEqual(row["config_snapshot"], {"strategy": {"active": "dual_ma"}})
        self.assertEqual(row["artifacts"]["report_paths"], ["reports/backtest/backtest_report_1.json"])
        self.assertEqual(row["artifacts"]["strategy_chart_path"], "reports/final/chart.png")


if __name__ == "__main__":
    unittest.main()
