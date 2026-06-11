import unittest
from pathlib import Path
from unittest.mock import Mock

import pandas as pd

from src.common.config import ConfigManager


class LegacyDeletionReadinessTest(unittest.TestCase):
    def _make_mode_with_main_config(self):
        repo_root = Path(__file__).resolve().parents[1]
        mode = Mock()
        mode.mode_name = "backtest"
        mode.config = ConfigManager(repo_root / "conf" / "config.yaml")
        mode.strategy = Mock()
        mode.risk_manager = Mock()
        mode.portfolio_book = Mock()
        mode.data_manager = Mock()
        mode.data_manager.primary_source.historical_store = Mock()
        mode.state = {}
        return mode

    def test_main_config_backtest_runtime_uses_native_strategy_and_risk_policy(self):
        from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter
        from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter
        from src.application.runtime_builder import RuntimeBuilder
        from src.strategy.implementations.domain_dual_ma import DomainDualMAStrategy

        mode = self._make_mode_with_main_config()
        historical_data = {"BTC/USDT": pd.DataFrame([{"timestamp": 1, "high": 101, "low": 99, "volume": 10}])}

        runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)

        self.assertIsInstance(runtime.domain_pipeline.strategy, DomainDualMAStrategy)
        self.assertNotIsInstance(runtime.domain_pipeline.strategy, LegacyDataFrameStrategyAdapter)
        self.assertFalse(
            any(isinstance(policy, LegacyRiskPolicyAdapter) for policy in runtime.domain_pipeline.risk_policy.policies)
        )

    def test_legacy_deletion_blockers_are_documented_before_modules_are_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for required in [
            "legacy 删除暂缓",
            "未声明 `strategy.interface`",
            "`BaseTradingMode._process_market_data`",
            "`ExecutionEngine`",
        ]:
            self.assertIn(required, design)
            self.assertIn(required, worklist)

    def test_live_execution_engine_fallback_removal_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")
        combined_docs = f"{design}\n{worklist}"

        self.assertIn("live 默认路径已解除旧 `ExecutionEngine` exchange client ownership", combined_docs)
        self.assertIn("live 默认路径已删除旧 `ExecutionEngine` fallback", combined_docs)
        self.assertIn("`live_trading.allow_legacy_execution_engine_fallback=true` 不再创建旧 `ExecutionEngine`", combined_docs)
        self.assertIn("旧配置 fallback", combined_docs)
        self.assertIn("旧 DataFrame pipeline/execution", combined_docs)

    def test_runtime_builder_no_longer_reads_legacy_execution_engine_exchange_client(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("`RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`", docs)
            self.assertIn("live runtime 只接受显式 `mode.exchange_client`", docs)

    def test_remaining_legacy_deletion_blockers_are_named_after_live_ownership_migration(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("剩余 legacy 删除 blocker", docs)
            self.assertIn("旧配置 fallback 已删除，不再作为剩余 legacy 删除 blocker", docs)
            self.assertIn("旧 DataFrame pipeline wrapper 已删除，不再作为剩余 legacy 删除 blocker", docs)
            self.assertIn("旧 DataFrame execution：`tests/test_execution_engine.py`、`LegacyExecutionAdapter` 和 `src/trading/execution/manager.py` 已删除", docs)
            self.assertIn("旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除", docs)
            self.assertIn("执行侧 legacy 删除 blocker 已清零", docs)
            self.assertIn(
                "live 显式 legacy fallback 已删除，不再作为剩余 legacy 删除 blocker",
                docs,
            )

    def test_legacy_config_fallback_removal_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("未声明 `strategy.interface` 的配置默认进入 domain runtime", docs)
            self.assertIn("`RuntimeBuilder._strategy_interface(default=\"domain\")`", docs)
            self.assertIn("只有显式 `strategy.interface=legacy` 才会启用 legacy adapter", docs)

    def test_legacy_dataframe_pipeline_behavior_is_mapped_before_wrapper_removal(self):
        repo_root = Path(__file__).resolve().parents[1]
        strategy_adapter_tests = (repo_root / "tests" / "test_legacy_strategy_adapter.py").read_text(encoding="utf-8")
        risk_policy_tests = (repo_root / "tests" / "test_domain_risk_policies.py").read_text(encoding="utf-8")
        risk_adapter_tests = (repo_root / "tests" / "test_risk_decision_adapter.py").read_text(encoding="utf-8")
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        self.assertIn("test_adapter_keeps_only_current_unprocessed_signal", strategy_adapter_tests)
        self.assertIn("test_position_availability_rejects_sell_without_position", risk_policy_tests)
        self.assertIn("test_sell_quantity_clamp_limits_sell_to_current_position", risk_policy_tests)
        self.assertIn("test_filter_accepted_removes_rejected_rows", risk_adapter_tests)
        for docs in [design, worklist]:
            self.assertIn("旧 DataFrame `TradingPipeline` 的可迁移行为已映射到 adapter/domain risk/risk adapter 测试", docs)

    def test_legacy_dataframe_trading_pipeline_wrapper_is_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        base_mode = (repo_root / "src" / "trading" / "modes" / "base.py").read_text(encoding="utf-8")
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        self.assertFalse((repo_root / "src" / "application" / "trading_pipeline.py").exists())
        self.assertNotIn("TradingPipeline", base_mode)
        self.assertNotIn("def _process_market_data", base_mode)
        self.assertFalse((repo_root / "tests" / "test_trading_pipeline.py").exists())
        for docs in [design, worklist]:
            self.assertIn("`src/application/trading_pipeline.py` 已删除", docs)
            self.assertIn("`BaseTradingMode._process_market_data` 已删除", docs)

    def test_execution_engine_fractional_fill_behavior_is_mapped_to_backtest_execution_model(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("旧 `ExecutionEngine._backtest_execution` fractional quantity/volume 行为已映射到 `BacktestExecutionModel.execute_orders()` 覆盖", docs)
            self.assertIn("`tests/test_backtest_execution_model.py` 覆盖 fractional crypto quantity 与 integer volume fractional decrement", docs)

    def test_legacy_execution_manager_removal_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        self.assertFalse((repo_root / "src" / "trading" / "execution" / "manager.py").exists())
        for docs in [design, worklist]:
            self.assertIn("`src/trading/execution/manager.py` 已删除", docs)
            self.assertIn("旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除", docs)
            self.assertIn("回测成交算法事实来源是 `BacktestExecutionModel`，不是旧 `ExecutionEngine`", docs)

    def test_legacy_dataframe_execution_engine_tests_are_removed_after_behavior_mapping(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        self.assertFalse((repo_root / "tests" / "test_execution_engine.py").exists())
        for docs in [design, worklist]:
            self.assertIn("`tests/test_execution_engine.py` 已删除", docs)
            self.assertIn("旧 `ExecutionEngine._backtest_execution` 行为事实来源已迁移到 `tests/test_backtest_execution_model.py`", docs)

    def test_legacy_execution_adapter_removal_is_documented(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("`LegacyExecutionAdapter` 已删除", docs)
            self.assertIn("旧执行桥不再作为剩余 legacy 删除 blocker", docs)

    def test_backtest_and_paper_modes_document_legacy_execution_engine_factory_removal(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("`BacktestTradingMode` 和 `PaperTradingMode` 的 `_create_legacy_execution_engine()` 已删除", docs)
            self.assertIn("backtest/paper mode 不再引用 `src.trading.execution.manager`", docs)

    def test_live_mode_documents_legacy_execution_engine_factory_removal(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("`LiveTradingMode` 的 `_create_legacy_execution_engine()` 已删除", docs)
            self.assertIn("live mode 不再引用 `src.trading.execution.manager`", docs)
            self.assertIn("`_create_legacy_live_execution_engine()` 已删除", docs)

    def test_modes_document_execution_engine_lifecycle_removal(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        for docs in [design, worklist]:
            self.assertIn("`BaseTradingMode` 不再声明 `self.execution_engine`", docs)
            self.assertIn("backtest/paper/live mode 不再读取或关闭 `self.execution_engine`", docs)
            self.assertIn("live account/order lifecycle 只通过 `exchange_client`", docs)

    def test_unused_legacy_execution_adapter_is_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        design = (repo_root / "docs" / "design" / "2026-06-09-final-four-layer-decoupling-architecture-design.md").read_text(encoding="utf-8")
        worklist = (repo_root / "docs" / "worklist" / "2026-06-09-final-four-layer-decoupling-worklist.md").read_text(encoding="utf-8")

        self.assertFalse((repo_root / "src" / "application" / "adapters" / "legacy_execution_adapter.py").exists())
        self.assertFalse((repo_root / "tests" / "test_legacy_execution_adapter.py").exists())
        for docs in [design, worklist]:
            self.assertIn("`LegacyExecutionAdapter` 已删除", docs)
            self.assertIn("旧执行桥不再作为剩余 legacy 删除 blocker", docs)


if __name__ == "__main__":
    unittest.main()
