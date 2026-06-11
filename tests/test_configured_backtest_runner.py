import asyncio
import unittest
from datetime import datetime

from src.application.backtest_batch import BacktestRunSpec
from src.application.backtest_planning import WalkForwardWindow


class ConfiguredBacktestRunnerTests(unittest.TestCase):
    def test_runner_clones_config_applies_parameters_and_window(self):
        from common.config import ConfigManager
        from src.application.configured_backtest_runner import make_configured_backtest_runner

        captured = {}

        class FakeCore:
            def __init__(self, config, mode, backtest_engine):
                captured["config"] = config
                captured["mode"] = mode
                captured["backtest_engine"] = backtest_engine

            async def run_pipeline(self):
                return {"final_equity": 100005}

        class FakeExecutor:
            def run(self, coro):
                return asyncio.run(coro)

        base_config = ConfigManager()
        base_config.set("strategy", "active", "dual_ma")
        base_config.set("api", "binance", "api_key", value="real-key")
        base_config.set("api", "binance", "secret", value="real-secret")
        runner = make_configured_backtest_runner(
            base_config,
            mode="backtest",
            backtest_engine="ohlcv",
            core_factory=FakeCore,
            executor_factory=FakeExecutor,
        )

        spec = BacktestRunSpec(
            run_id="run-001",
            parameters={
                "short_window": 5,
                "strategy.dual_ma.long_window": 20,
            },
            window=WalkForwardWindow(
                train_start=datetime(2024, 12, 1),
                train_end=datetime(2025, 1, 1),
                test_start=datetime(2025, 1, 1),
                test_end=datetime(2025, 1, 2),
            ),
        )

        report = runner(spec)
        run_config = captured["config"]

        self.assertEqual(captured["mode"], "backtest")
        self.assertEqual(captured["backtest_engine"], "ohlcv")
        self.assertEqual(run_config.get("strategy", "parameters", "short_window"), 5)
        self.assertEqual(run_config.get("strategy", "dual_ma", "short_window"), 5)
        self.assertEqual(run_config.get("strategy", "dual_ma", "long_window"), 20)
        self.assertEqual(run_config.get("backtest", "period", "start"), "2025-01-01")
        self.assertEqual(run_config.get("backtest", "period", "end"), "2025-01-02")
        self.assertEqual(report["research"]["run_id"], "run-001")
        self.assertEqual(report["research"]["parameters"], spec.parameters)
        self.assertEqual(len(report["research"]["run_hash"]), 16)
        self.assertEqual(len(report["research"]["config_hash"]), 16)
        self.assertEqual(report["research"]["config_snapshot"]["api"]["binance"]["api_key"], "<redacted>")
        self.assertEqual(report["research"]["config_snapshot"]["api"]["binance"]["secret"], "<redacted>")
        self.assertEqual(base_config.get("strategy", "parameters", "short_window", default=None), None)

    def test_runner_raises_when_pipeline_returns_error(self):
        from common.config import ConfigManager
        from src.application.configured_backtest_runner import make_configured_backtest_runner

        class FakeCore:
            def __init__(self, config, mode, backtest_engine):
                pass

            async def run_pipeline(self):
                return {"error": "bad run"}

        class FakeExecutor:
            def run(self, coro):
                return asyncio.run(coro)

        runner = make_configured_backtest_runner(
            ConfigManager(),
            core_factory=FakeCore,
            executor_factory=FakeExecutor,
        )

        with self.assertRaisesRegex(RuntimeError, "bad run"):
            runner(BacktestRunSpec(run_id="run-001", parameters={}))


if __name__ == "__main__":
    unittest.main()
