import ast
import unittest
from pathlib import Path


class DomainBoundaryTests(unittest.TestCase):
    """守住领域层依赖边界，防止后续重构时把基础设施重新引进 domain。"""

    FORBIDDEN_IMPORTS = {
        "pandas",
        "common.config",
        "src.common.config",
        "trading.modes",
        "src.trading.modes",
        "trading.execution.manager",
        "src.trading.execution.manager",
        "strategy",
        "src.strategy",
        "risk.manager",
        "src.risk.manager",
        "datasource",
        "src.datasource",
        "exchange",
        "src.exchange",
        "reporting",
        "src.reporting",
        "order",
        "src.order",
    }

    def test_domain_package_has_no_forbidden_imports(self):
        domain_root = Path(__file__).resolve().parents[1] / "src" / "domain"

        violations = []
        for path in sorted(domain_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if self._is_forbidden(alias.name):
                            violations.append(f"{path.name}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    if self._is_forbidden(module):
                        violations.append(f"{path.name}: from {module} import ...")

        self.assertEqual([], violations)

    def test_domain_pipeline_file_does_not_use_dataframe_name(self):
        domain_root = Path(__file__).resolve().parents[1] / "src" / "domain"
        pipeline_path = domain_root / "trading_pipeline.py"
        if not pipeline_path.exists():
            self.skipTest("DomainTradingPipeline has not been created yet")

        source = pipeline_path.read_text(encoding="utf-8")
        self.assertNotIn("DataFrame", source)
        self.assertNotIn("pd.", source)

    def test_application_dataframe_pipeline_wrapper_is_removed(self):
        repo_root = Path(__file__).resolve().parents[1]

        self.assertFalse((repo_root / "src" / "application" / "trading_pipeline.py").exists())

    def test_application_adapters_package_does_not_reexport_legacy_adapters(self):
        import src.application.adapters as adapters

        self.assertFalse(hasattr(adapters, "LegacyDataFrameStrategyAdapter"))
        self.assertFalse(hasattr(adapters, "LegacyRiskPolicyAdapter"))
        self.assertEqual([], getattr(adapters, "__all__", []))

    def test_runtime_builder_direct_imports_explicit_legacy_adapters(self):
        repo_root = Path(__file__).resolve().parents[1]
        source = (repo_root / "src" / "application" / "runtime_builder.py").read_text(encoding="utf-8")

        self.assertNotIn("from src.application.adapters import", source)
        self.assertIn(
            "from src.application.adapters.legacy_strategy_adapter import LegacyDataFrameStrategyAdapter",
            source,
        )
        self.assertIn(
            "from src.application.adapters.legacy_risk_policy_adapter import LegacyRiskPolicyAdapter",
            source,
        )

    def test_legacy_execution_manager_is_removed_and_old_backtest_engine_is_marked_legacy(self):
        repo_root = Path(__file__).resolve().parents[1]
        manager_path = repo_root / "src" / "trading" / "execution" / "manager.py"
        old_backtest_engine_path = repo_root / "src" / "backtest" / "engine.py"

        self.assertFalse(manager_path.exists())

        source = old_backtest_engine_path.read_text(encoding="utf-8")
        self.assertIn("LEGACY COMPATIBILITY", source)

    def _is_forbidden(self, module_name: str) -> bool:
        return any(
            module_name == forbidden or module_name.startswith(f"{forbidden}.")
            for forbidden in self.FORBIDDEN_IMPORTS
        )
