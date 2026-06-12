import importlib
import sys
import unittest
from unittest.mock import Mock, patch


class DataIntegrityCheckerTest(unittest.IsolatedAsyncioTestCase):
    async def test_close_waits_for_thread_pool(self):
        for module_name in ["src.datasource.integrity"]:
            sys.modules.pop(module_name, None)

        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            integrity_module = importlib.import_module("src.datasource.integrity")

        checker = integrity_module.DataIntegrityChecker.__new__(integrity_module.DataIntegrityChecker)
        checker.thread_pool = Mock()
        checker.logger = Mock()

        await integrity_module.DataIntegrityChecker.close(checker)

        checker.thread_pool.shutdown.assert_called_once_with(wait=True)


if __name__ == "__main__":
    unittest.main()
