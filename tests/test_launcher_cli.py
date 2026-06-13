import asyncio
import logging
import sys
import types
import unittest
from unittest.mock import Mock, patch

import src.launcher as launcher
import src.ui.cli as cli


class LauncherCliTest(unittest.TestCase):
    def test_default_config_path_points_to_conf_config_yaml(self):
        self.assertEqual(
            launcher.default_config_path(),
            launcher.Path(launcher.project_root) / "conf" / "config.yaml",
        )

    def test_backtest_engine_argument_does_not_use_legacy_backtest_factory(self):
        from src.backtest.engine import BacktestFactory

        config = Mock()
        config.get.return_value = "INFO"
        logger = Mock()

        def fake_setup_environment(args):
            args.config = config
            return logger

        def fake_run_cli_mode(args, config_path, passed_config, passed_logger):
            self.assertEqual(args.backtest_engine, "ohlcv")
            return {"status": "ok"}

        argv = [
            "main.py",
            "--cli",
            "--mode",
            "backtest",
            "--backtest-engine",
            "ohlcv",
        ]

        with patch.object(sys, "argv", argv), \
             patch.object(launcher, "setup_environment", side_effect=fake_setup_environment), \
             patch.object(launcher.LogManager, "get_logger", return_value=logger), \
             patch.object(BacktestFactory, "__init__", side_effect=AssertionError("legacy BacktestFactory used")), \
             patch("src.ui.cli.run_cli_mode", side_effect=fake_run_cli_mode):
            result = launcher.launch()

        self.assertNotIn("error", result)

    def test_cli_mode_passes_config_to_cli_runner(self):
        config = Mock()
        config.get.return_value = "INFO"
        logger = Mock()
        captured = {}

        def fake_setup_environment(args):
            args.config = config
            return logger

        def fake_run_cli_mode(args, config_path, passed_config, passed_logger):
            captured["args"] = args
            captured["config_path"] = config_path
            captured["config"] = passed_config
            captured["logger"] = passed_logger
            return {"status": "ok"}

        argv = [
            "main.py",
            "--cli",
            "--mode",
            "backtest",
            "--backtest-engine",
            "ohlcv",
        ]

        with patch.object(sys, "argv", argv), \
             patch.object(launcher, "setup_environment", side_effect=fake_setup_environment), \
             patch.object(launcher.LogManager, "get_logger", return_value=logger), \
             patch("src.ui.cli.run_cli_mode", side_effect=fake_run_cli_mode):
            result = launcher.launch()

        self.assertNotIn("error", result)
        self.assertIs(captured["config"], config)
        self.assertIs(captured["logger"], logger)
        self.assertEqual(captured["config_path"], config)

    def test_research_cli_routes_to_research_runner(self):
        config = Mock()
        config.get.return_value = "INFO"
        logger = Mock()
        captured = {}

        def fake_setup_environment(args):
            args.config = config
            return logger

        def fake_run_research_cli_mode(args, passed_config, passed_logger):
            captured["args"] = args
            captured["config"] = passed_config
            captured["logger"] = passed_logger
            return {"status": "research-ok"}

        argv = [
            "main.py",
            "--cli",
            "--research",
            "--mode",
            "backtest",
            "--backtest-engine",
            "ohlcv",
            "--research-grid",
            "{\"short_window\":[5]}",
        ]

        with patch.object(sys, "argv", argv), \
             patch.object(launcher, "setup_environment", side_effect=fake_setup_environment), \
             patch.object(launcher.LogManager, "get_logger", return_value=logger), \
             patch("src.ui.research_cli.run_research_cli_mode", side_effect=fake_run_research_cli_mode):
            result = launcher.launch()

        self.assertNotIn("error", result)
        self.assertIs(captured["config"], config)
        self.assertIs(captured["logger"], logger)
        self.assertEqual(captured["args"].research_grid, "{\"short_window\":[5]}")


class CliRunnerTest(unittest.TestCase):
    def test_timeframe_override_updates_trading_timeframe(self):
        args = types.SimpleNamespace(
            mode="backtest",
            backtest_engine="ohlcv",
            strategy="dual_ma",
            symbol="BTC/USDT",
            timeframe="1h",
            start_date="2023-01-01",
            end_date="2023-01-02",
            db_url=None,
            max_workers=None,
        )
        config = Mock()
        logger = Mock()

        self._run_cli_with_fake_trader(args, config, logger)

        config.set.assert_any_call("trading", "timeframe", "1h")

    def test_cli_runner_does_not_shutdown_trader_twice(self):
        args = types.SimpleNamespace(
            mode="backtest",
            backtest_engine="ohlcv",
            strategy="dual_ma",
            symbol="BTC/USDT",
            timeframe=None,
            start_date=None,
            end_date=None,
            db_url=None,
            max_workers=None,
        )
        config = Mock()
        logger = Mock()

        trader = self._run_cli_with_fake_trader(args, config, logger)

        self.assertEqual(trader.shutdown_calls, 1)

    def test_cli_runner_shuts_down_logging_handlers(self):
        args = types.SimpleNamespace(
            mode="backtest",
            backtest_engine="ohlcv",
            strategy="dual_ma",
            symbol="BTC/USDT",
            timeframe=None,
            start_date=None,
            end_date=None,
            db_url=None,
            max_workers=None,
        )
        config = Mock()
        logger = Mock()

        with patch.object(logging, "shutdown") as shutdown:
            self._run_cli_with_fake_trader(args, config, logger)

        shutdown.assert_called_once_with()

    def _run_cli_with_fake_trader(self, args, config, logger):
        captured = {}

        class FakeTradingCore:
            def __init__(self, passed_config, mode, backtest_engine):
                self.config = passed_config
                self.mode = mode
                self.backtest_engine = backtest_engine
                self.shutdown_calls = 0
                captured["trader"] = self

            async def run_pipeline(self):
                await self.shutdown()
                return {"status": "ok"}

            async def shutdown(self):
                self.shutdown_calls += 1

        class FakeExecutor:
            def run(self, coro):
                return asyncio.run(coro)

        module = types.SimpleNamespace()
        spec = Mock()
        spec.loader.exec_module.side_effect = lambda loaded_module: setattr(loaded_module, "TradingCore", FakeTradingCore)

        with patch.object(cli.importlib.util, "spec_from_file_location", return_value=spec), \
             patch.object(cli.importlib.util, "module_from_spec", return_value=module), \
             patch("src.common.async_executor.AsyncExecutor", return_value=FakeExecutor()):
            result = cli.run_cli_mode(args, config, config, logger)

        self.assertEqual(result, {"status": "ok"})
        return captured["trader"]


class ResearchCliRunnerTest(unittest.TestCase):
    def test_research_cli_parses_grid_and_calls_application_entrypoint(self):
        from src.ui import research_cli

        args = types.SimpleNamespace(
            research_grid="{\"short_window\":[5]}",
            research_metric="diagnostics.cost_to_abs_net_return",
            research_ascending=True,
            research_output_dir="reports/research",
            research_walk_forward_start=None,
            research_walk_forward_end=None,
            research_train_days=None,
            research_test_days=None,
            research_step_days=None,
            mode="backtest",
            backtest_engine="ohlcv",
            strategy=None,
            symbol=None,
            timeframe=None,
            start_date=None,
            end_date=None,
        )
        config = Mock()
        logger = Mock()
        runner = object()

        with patch.object(research_cli, "make_configured_backtest_runner", return_value=runner) as make_runner, \
             patch.object(research_cli, "run_backtest_research", return_value={"summary_path": "reports/research/x.json"}) as run_research, \
             patch("builtins.print"):
            summary = research_cli.run_research_cli_mode(args, config, logger)

        make_runner.assert_called_once_with(config, mode="backtest", backtest_engine="ohlcv")
        run_research.assert_called_once_with(
            parameter_grid={"short_window": [5]},
            run_backtest=runner,
            windows=None,
            metric="diagnostics.cost_to_abs_net_return",
            descending=False,
            summary_report_dir="reports/research",
        )
        self.assertEqual(summary["summary_path"], "reports/research/x.json")

    def test_research_cli_rejects_invalid_grid_with_helpful_message(self):
        from src.ui import research_cli

        with self.assertRaisesRegex(ValueError, "--research-grid must be valid JSON"):
            research_cli._parse_parameter_grid("{short_window:[5]}")

    def test_research_cli_builds_walk_forward_windows(self):
        from src.ui import research_cli

        args = types.SimpleNamespace(
            research_grid="{}",
            research_metric="final_equity",
            research_ascending=False,
            research_output_dir="reports/research",
            research_walk_forward_start="2024-12-31",
            research_walk_forward_end="2025-01-02",
            research_train_days=1,
            research_test_days=1,
            research_step_days=1,
            mode="backtest",
            backtest_engine="ohlcv",
            strategy=None,
            symbol=None,
            timeframe=None,
            start_date=None,
            end_date=None,
        )

        with patch.object(research_cli, "make_configured_backtest_runner", return_value=object()), \
             patch.object(research_cli, "run_backtest_research", return_value={}) as run_research, \
             patch("builtins.print"):
            research_cli.run_research_cli_mode(args, Mock(), Mock())

        windows = run_research.call_args.kwargs["windows"]
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].test_start.isoformat(), "2025-01-01T00:00:00")
        self.assertEqual(windows[0].test_end.isoformat(), "2025-01-02T00:00:00")

    def test_research_cli_applies_common_backtest_overrides(self):
        from src.ui import research_cli

        args = types.SimpleNamespace(
            research_grid="{}",
            research_metric="final_equity",
            research_ascending=False,
            research_output_dir="reports/research",
            research_walk_forward_start=None,
            research_walk_forward_end=None,
            research_train_days=None,
            research_test_days=None,
            research_step_days=None,
            mode="backtest",
            backtest_engine="ohlcv",
            strategy="dual_ma",
            symbol="BTC/USDT,ETH/USDT",
            timeframe="5m",
            start_date="2025-01-01",
            end_date="2025-01-02",
        )
        config = Mock()
        logger = Mock()

        with patch.object(research_cli, "make_configured_backtest_runner", return_value=object()), \
             patch.object(research_cli, "run_backtest_research", return_value={}), \
             patch("builtins.print"):
            research_cli.run_research_cli_mode(args, config, logger)

        config.set.assert_any_call("strategy", "active", "dual_ma")
        config.set.assert_any_call("trading", "instruments", ["BTC/USDT", "ETH/USDT"])
        config.set.assert_any_call("trading", "timeframe", "5m")
        config.set.assert_any_call("backtest", "period", "start", "2025-01-01")
        config.set.assert_any_call("backtest", "period", "end", "2025-01-02")


if __name__ == "__main__":
    unittest.main()
