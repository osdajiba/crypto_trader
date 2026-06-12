import importlib
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch


class DataManagerTest(unittest.IsolatedAsyncioTestCase):
    def test_local_backtest_does_not_initialize_exchange_backup_source(self):
        config = Mock()

        def fake_get(*keys, default=None):
            if keys == ("system", "operational_mode"):
                return "backtest"
            if keys == ("data", "backup_source"):
                return "exchange"
            return default

        config.get.side_effect = fake_get
        created_sources = []

        def fake_create_source(source_type, passed_config):
            created_sources.append(source_type)
            return Mock()

        for module_name in [
            "src.datasource.manager",
            "src.datasource.datasources",
            "src.common.helpers",
        ]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            manager_module = importlib.import_module("src.datasource.manager")

        with patch.object(manager_module.LogManager, "get_logger", return_value=Mock()), \
             patch.object(manager_module.DataSourceFactory, "create_source", side_effect=fake_create_source):
            manager = manager_module.DataManager(source_type="local", config=config)

        self.assertEqual(created_sources, ["local"])
        self.assertIsNone(manager.backup_source)

    async def test_close_waits_for_thread_pools(self):
        with patch("common.logging.LogManager.get_logger", return_value=Mock()), \
             patch("src.common.logging.LogManager.get_logger", return_value=Mock()):
            manager_module = importlib.import_module("src.datasource.manager")

        manager = manager_module.DataManager.__new__(manager_module.DataManager)
        manager.primary_source = None
        manager.backup_source = None
        manager.logger = Mock()
        manager.thread_pool = Mock()
        manager.integrity_checker = Mock()
        manager.integrity_checker.close = AsyncMock()
        manager.data_cache = {}
        manager.cache_last_accessed = {}

        await manager_module.DataManager.close(manager)

        manager.integrity_checker.close.assert_awaited_once_with()
        manager.thread_pool.shutdown.assert_called_once_with(wait=True)


if __name__ == "__main__":
    unittest.main()
