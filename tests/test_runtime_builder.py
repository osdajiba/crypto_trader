import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd

from src.domain.models import MarketBar, MarketSlice, PortfolioSnapshot, StrategySignal


class RuntimeBuilderTest(unittest.IsolatedAsyncioTestCase):
    def _make_config(self):
        config = Mock()
        config.get.side_effect = lambda *keys, default=None: default
        return config

    def _make_mode(self, mode_name="backtest"):
        mode = Mock()
        mode.mode_name = mode_name
        mode.config = self._make_config()
        mode.strategy = Mock()
        mode.strategy.__class__.__name__ = "FakeStrategy"
        mode.risk_manager = Mock()
        mode.risk_manager.validate_signals = AsyncMock(side_effect=lambda signals: signals)
        mode.portfolio_book = Mock()
        mode.data_manager = Mock()
        mode.state = {}
        return mode

    def test_build_backtest_runtime_defaults_to_native_domain_strategy(self):
        from src.application.backtest_use_case import BacktestUseCase
        from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter
        from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
        from src.application.reporting import InMemoryReporter
        from src.application.runtime_builder import RuntimeBuilder
        from src.datasource.feeds.market_data_feed import HistoricalMarketDataFeed
        from src.domain.trading_pipeline import DomainTradingPipeline
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy
        from src.trading.execution.backtest_model import BacktestExecutionModel

        mode = self._make_mode("backtest")
        mode.data_manager.primary_source.historical_store = Mock()
        historical_data = {"BTC/USDT": pd.DataFrame([{"timestamp": 1, "high": 101, "low": 99, "volume": 10}])}

        runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)

        self.assertIsInstance(runtime.market_data_feed, HistoricalMarketDataFeed)
        self.assertIsInstance(runtime.domain_pipeline, DomainTradingPipeline)
        self.assertIsInstance(runtime.domain_pipeline.strategy, DomainDualMAStrategy)
        self.assertNotIsInstance(runtime.domain_pipeline.strategy, LegacyDataFrameStrategyAdapter)
        self.assertFalse(
            any(isinstance(policy, LegacyRiskPolicyAdapter) for policy in runtime.domain_pipeline.risk_policy.policies)
        )
        self.assertIsInstance(runtime.domain_pipeline.execution_model, BacktestExecutionModel)
        self.assertIs(runtime.domain_pipeline.portfolio, mode.portfolio_book)
        self.assertIsInstance(runtime.reporter, InMemoryReporter)
        self.assertIsInstance(runtime.use_case, BacktestUseCase)
        self.assertIs(mode.market_data_feed, runtime.market_data_feed)
        self.assertIs(mode.domain_pipeline, runtime.domain_pipeline)
        self.assertIs(mode.domain_reporter, runtime.reporter)
        self.assertIs(mode.backtest_use_case, runtime.use_case)

    def test_build_backtest_runtime_uses_legacy_adapters_only_when_explicitly_configured(self):
        from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter
        from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
        from src.application.runtime_builder import RuntimeBuilder

        mode = self._make_mode("backtest")
        mode.config.get.side_effect = lambda *keys, default=None: {
            ("strategy", "interface"): "legacy",
        }.get(keys, default)
        mode.data_manager.primary_source.historical_store = Mock()
        historical_data = {"BTC/USDT": pd.DataFrame([{"timestamp": 1, "high": 101, "low": 99, "volume": 10}])}

        runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)

        self.assertIsInstance(runtime.domain_pipeline.strategy, LegacyDataFrameStrategyAdapter)
        self.assertTrue(
            any(isinstance(policy, LegacyRiskPolicyAdapter) for policy in runtime.domain_pipeline.risk_policy.policies)
        )

    def test_runtime_builder_can_select_native_dual_ma_without_legacy_adapter(self):
        from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
        from src.application.runtime_builder import RuntimeBuilder
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy

        mode = self._make_mode("backtest")
        mode.config.get.side_effect = lambda *keys, default=None: {
            ("strategy", "interface"): "domain",
            ("strategy", "active"): "dual_ma",
            ("strategy", "parameters", "short_window"): 2,
            ("strategy", "parameters", "long_window"): 3,
            ("trading", "position_sizing", "fraction"): 0.01,
        }.get(keys, default)
        mode.data_manager.primary_source.historical_store = Mock()
        historical_data = {"BTC/USDT": pd.DataFrame([{"timestamp": 1, "high": 101, "low": 99, "volume": 10}])}

        runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)

        self.assertIsInstance(runtime.domain_pipeline.strategy, DomainDualMAStrategy)
        self.assertNotIsInstance(runtime.domain_pipeline.strategy, LegacyDataFrameStrategyAdapter)

    def test_runtime_builder_can_select_native_multi_factors_without_legacy_adapter(self):
        from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
        from src.application.runtime_builder import RuntimeBuilder
        from src.strategy.implementations.domain_multi_factors import DomainMultiFactorsStrategy

        mode = self._make_mode("backtest")
        mode.config.get.side_effect = lambda *keys, default=None: {
            ("strategy", "interface"): "domain",
            ("strategy", "active"): "multi_factors",
            ("strategy", "parameters", "threshold"): 0.4,
            ("strategy", "factors"): {
                "rsi_fast": {
                    "type": "rsi",
                    "params": {"period": 3},
                    "window_size": 4,
                    "signal_type": "threshold",
                    "upper_threshold": 70,
                    "lower_threshold": 30,
                    "weight": 1.0,
                }
            },
            ("trading", "position_sizing", "fraction"): 0.01,
        }.get(keys, default)
        mode.data_manager.primary_source.historical_store = Mock()
        historical_data = {"BTC/USDT": pd.DataFrame([{"timestamp": 1, "high": 101, "low": 99, "volume": 10}])}

        runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)

        self.assertIsInstance(runtime.domain_pipeline.strategy, DomainMultiFactorsStrategy)
        self.assertNotIsInstance(runtime.domain_pipeline.strategy, LegacyDataFrameStrategyAdapter)
        self.assertEqual(runtime.domain_pipeline.strategy.threshold, 0.4)

    async def test_native_runtime_risk_policy_sizes_signal_without_quantity_metadata(self):
        from src.application.runtime_builder import RuntimeBuilder

        mode = self._make_mode("backtest")
        mode.config.get.side_effect = lambda *keys, default=None: {
            ("strategy", "interface"): "domain",
            ("strategy", "active"): "dual_ma",
            ("trading", "position_sizing", "fraction"): 0.01,
            ("default_config", "user_config", "commission"): 0.0,
        }.get(keys, default)
        timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
        signal = StrategySignal("native:buy:1", "BTC/USDT", timestamp, "buy")
        market = MarketSlice(
            timestamp=timestamp,
            bars_by_symbol={
                "BTC/USDT": MarketBar("BTC/USDT", "1m", timestamp, 100, 100, 100, 100, 10)
            },
        )
        portfolio = PortfolioSnapshot(
            timestamp=timestamp,
            cash=100000,
            positions={},
            market_prices={"BTC/USDT": 100},
            equity=100000,
        )

        decision = await RuntimeBuilder(mode)._create_risk_policy().evaluate(signal, portfolio, market)

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.target_quantity, 10)
        self.assertNotIn("quantity", signal.metadata)

    def test_build_paper_runtime_uses_simulated_execution_model(self):
        from src.application.paper_trading_use_case import PaperTradingUseCase
        from src.application.runtime_builder import RuntimeBuilder
        from src.datasource.feeds.market_data_feed import RealtimeMarketDataFeed
        from src.trading.execution.paper_model import PaperExecutionModel

        mode = self._make_mode("paper")

        runtime = RuntimeBuilder(mode).build_paper_runtime()

        self.assertIsInstance(runtime.market_data_feed, RealtimeMarketDataFeed)
        self.assertIsInstance(runtime.domain_pipeline.execution_model, PaperExecutionModel)
        self.assertIsInstance(runtime.use_case, PaperTradingUseCase)
        self.assertIs(mode.paper_use_case, runtime.use_case)

    def test_build_live_runtime_uses_exchange_client_after_safety_gate(self):
        from src.application.live_trading_use_case import LiveTradingUseCase
        from src.application.runtime_builder import RuntimeBuilder
        from src.datasource.feeds.market_data_feed import RealtimeMarketDataFeed
        from src.trading.execution.live_model import LiveExecutionModel

        mode = self._make_mode("live")
        mode.exchange_client = Mock()

        runtime = RuntimeBuilder(mode).build_live_runtime()

        self.assertIsInstance(runtime.market_data_feed, RealtimeMarketDataFeed)
        self.assertIsInstance(runtime.domain_pipeline.execution_model, LiveExecutionModel)
        self.assertIs(runtime.domain_pipeline.execution_model.exchange_client, mode.exchange_client)
        self.assertIsInstance(runtime.use_case, LiveTradingUseCase)
        self.assertIs(mode.live_use_case, runtime.use_case)

    def test_build_live_runtime_rejects_legacy_execution_engine_exchange_client(self):
        from src.application.runtime_builder import RuntimeBuilder

        mode = self._make_mode("live")
        mode.execution_engine = Mock()
        mode.execution_engine.binance = Mock()

        with self.assertRaisesRegex(ValueError, "Live execution requires an exchange client"):
            RuntimeBuilder(mode).build_live_runtime()

    def test_build_live_runtime_prefers_mode_exchange_client_over_legacy_execution_engine(self):
        from src.application.runtime_builder import RuntimeBuilder
        from src.trading.execution.live_model import LiveExecutionModel

        mode = self._make_mode("live")
        mode.exchange_client = Mock()
        mode.execution_engine = None

        runtime = RuntimeBuilder(mode).build_live_runtime()

        self.assertIsInstance(runtime.domain_pipeline.execution_model, LiveExecutionModel)
        self.assertIs(runtime.domain_pipeline.execution_model.exchange_client, mode.exchange_client)

    def test_build_live_runtime_fails_before_real_execution_exists(self):
        from src.application.runtime_builder import RuntimeBuilder
        from src.trading.execution.live_model import LiveExecutionModel

        mode = self._make_mode("live")
        mode.execution_engine = Mock()
        mode.execution_engine.binance = None

        with patch.object(LiveExecutionModel, "__init__", side_effect=AssertionError("真实执行模型不应被创建")):
            with self.assertRaisesRegex(ValueError, "Live execution requires an exchange client"):
                RuntimeBuilder(mode).build_live_runtime()


if __name__ == "__main__":
    unittest.main()
