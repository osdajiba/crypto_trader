import ast
import unittest
from pathlib import Path


class LegacyExecutionBoundaryTest(unittest.IsolatedAsyncioTestCase):
    def test_legacy_execution_manager_module_is_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        manager_path = repo_root / "src" / "trading" / "execution" / "manager.py"

        self.assertFalse(manager_path.exists())

    def test_execution_engine_imports_are_limited_to_legacy_or_fallback_modes(self):
        repo_root = Path(__file__).resolve().parents[1]
        allowed_importers = set()

        offenders = []
        for path in (repo_root / "src").rglob("*.py"):
            relative_path = path.relative_to(repo_root)
            source = path.read_text(encoding="utf-8")
            if "trading.execution.manager" not in source:
                continue
            if relative_path not in allowed_importers:
                offenders.append(relative_path.as_posix())

        self.assertEqual([], offenders)

    def test_execution_engine_tests_are_removed_after_behavior_mapping(self):
        repo_root = Path(__file__).resolve().parents[1]
        test_path = repo_root / "tests" / "test_execution_engine.py"
        backtest_model_test = (repo_root / "tests" / "test_backtest_execution_model.py").read_text(encoding="utf-8")

        self.assertFalse(test_path.exists())
        self.assertIn("test_execute_orders_preserves_fractional_crypto_quantity", backtest_model_test)
        self.assertIn("test_execute_orders_decrements_integer_volume_by_fractional_fill", backtest_model_test)

    def test_legacy_execution_adapter_is_not_package_level_export(self):
        repo_root = Path(__file__).resolve().parents[1]
        init_path = repo_root / "src" / "application" / "adapters" / "__init__.py"
        source = init_path.read_text(encoding="utf-8")

        self.assertNotIn("legacy_execution_adapter", source)
        self.assertNotIn("LegacyExecutionAdapter", source)

    def test_modes_lazy_import_legacy_execution_engine(self):
        repo_root = Path(__file__).resolve().parents[1]
        mode_paths = [
            repo_root / "src" / "trading" / "modes" / "backtest.py",
            repo_root / "src" / "trading" / "modes" / "live.py",
            repo_root / "src" / "trading" / "modes" / "paper.py",
        ]

        offenders = []
        for path in mode_paths:
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in module.body:
                if isinstance(node, ast.ImportFrom) and node.module == "src.trading.execution.manager":
                    imported_names = {alias.name for alias in node.names}
                    if "ExecutionEngine" in imported_names:
                        offenders.append(path.name)

        self.assertEqual([], offenders)

    def test_modes_no_longer_expose_legacy_execution_engine_factories(self):
        repo_root = Path(__file__).resolve().parents[1]
        mode_paths = [
            repo_root / "src" / "trading" / "modes" / "backtest.py",
            repo_root / "src" / "trading" / "modes" / "live.py",
            repo_root / "src" / "trading" / "modes" / "paper.py",
        ]

        offenders = []
        for path in mode_paths:
            source = path.read_text(encoding="utf-8")
            if "_create_legacy_execution_engine" in source or "src.trading.execution.manager" in source:
                offenders.append(path.name)

        self.assertEqual([], offenders)

    def test_modes_no_longer_reference_execution_engine_lifecycle(self):
        repo_root = Path(__file__).resolve().parents[1]
        mode_paths = [
            repo_root / "src" / "trading" / "modes" / "base.py",
            repo_root / "src" / "trading" / "modes" / "backtest.py",
            repo_root / "src" / "trading" / "modes" / "live.py",
            repo_root / "src" / "trading" / "modes" / "paper.py",
        ]

        offenders = []
        for path in mode_paths:
            source = path.read_text(encoding="utf-8")
            if "execution_engine" in source:
                offenders.append(path.name)

        self.assertEqual([], offenders)

    def test_backtest_execution_model_remains_behavior_source_after_manager_removal(self):
        repo_root = Path(__file__).resolve().parents[1]
        manager_path = repo_root / "src" / "trading" / "execution" / "manager.py"
        backtest_model_test = (repo_root / "tests" / "test_backtest_execution_model.py").read_text(encoding="utf-8")

        self.assertFalse(manager_path.exists())
        self.assertIn("test_execute_orders_preserves_fractional_crypto_quantity", backtest_model_test)
        self.assertIn("test_execute_orders_decrements_integer_volume_by_fractional_fill", backtest_model_test)


if __name__ == "__main__":
    unittest.main()
