import unittest
from datetime import datetime, timedelta, timezone


class BacktestBatchRunnerTests(unittest.TestCase):
    def test_build_backtest_run_specs_combines_parameters_and_windows(self):
        from src.application.backtest_batch import build_backtest_run_specs
        from src.application.backtest_planning import WalkForwardWindow

        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        window = WalkForwardWindow(
            train_start=start,
            train_end=start + timedelta(days=3),
            test_start=start + timedelta(days=3),
            test_end=start + timedelta(days=5),
        )

        specs = build_backtest_run_specs(
            parameter_grid={"short_window": [5, 10], "long_window": [20]},
            windows=[window],
        )

        self.assertEqual([spec.run_id for spec in specs], ["run-001", "run-002"])
        self.assertEqual(specs[0].parameters, {"short_window": 5, "long_window": 20})
        self.assertEqual(specs[1].parameters, {"short_window": 10, "long_window": 20})
        self.assertEqual(specs[0].window, window)

    def test_batch_runner_records_success_and_failure_results(self):
        from src.application.backtest_batch import BacktestBatchRunner, build_backtest_run_specs

        specs = build_backtest_run_specs(
            parameter_grid={"short_window": [5, 10]},
        )

        def run_backtest(spec):
            if spec.parameters["short_window"] == 10:
                raise ValueError("boom")
            return {"final_equity": 101000, "total_trades": 3}

        results = BacktestBatchRunner(run_backtest).run(specs)

        self.assertEqual([result.run_id for result in results], ["run-001", "run-002"])
        self.assertEqual(results[0].status, "success")
        self.assertEqual(results[0].report["final_equity"], 101000)
        self.assertEqual(results[0].error, "")
        self.assertEqual(results[1].status, "failed")
        self.assertEqual(results[1].report, {})
        self.assertIn("boom", results[1].error)


if __name__ == "__main__":
    unittest.main()
