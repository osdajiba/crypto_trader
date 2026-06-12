import unittest
import tempfile
from pathlib import Path


class BacktestResearchTests(unittest.TestCase):
    def test_research_runner_composes_specs_batch_runner_and_diagnostics_summary(self):
        from src.application.backtest_research import run_backtest_research

        def run_backtest(spec):
            return {
                "final_equity": 100000 + spec.parameters["short_window"],
                "diagnostics": {
                    "cost_to_abs_net_return": 0.5 if spec.parameters["short_window"] == 5 else 0.2,
                },
            }

        summary = run_backtest_research(
            parameter_grid={"short_window": [5, 10]},
            run_backtest=run_backtest,
            metric="diagnostics.cost_to_abs_net_return",
            descending=False,
        )

        self.assertEqual(summary["metric"], "diagnostics.cost_to_abs_net_return")
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["best_run_id"], "run-002")
        self.assertEqual([row["run_id"] for row in summary["results"]], ["run-002", "run-001"])
        self.assertEqual(summary["results"][0]["parameters"], {"short_window": 10})

    def test_research_runner_keeps_failed_runs_in_summary(self):
        from src.application.backtest_research import run_backtest_research

        def run_backtest(spec):
            if spec.parameters["short_window"] == 10:
                raise RuntimeError("bad params")
            return {"final_equity": 100005}

        summary = run_backtest_research(
            parameter_grid={"short_window": [5, 10]},
            run_backtest=run_backtest,
        )

        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["results"][1]["run_id"], "run-002")
        self.assertEqual(summary["results"][1]["error"], "bad params")

    def test_research_runner_can_persist_summary_json(self):
        from src.application.backtest_research import run_backtest_research

        def run_backtest(spec):
            return {"final_equity": 100000 + spec.parameters["short_window"]}

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_backtest_research(
                parameter_grid={"short_window": [5]},
                run_backtest=run_backtest,
                summary_report_dir=Path(tmp),
            )

            summary_path = Path(summary["summary_path"])

            self.assertTrue(summary_path.exists())
            self.assertEqual(summary_path.parent, Path(tmp))


if __name__ == "__main__":
    unittest.main()
