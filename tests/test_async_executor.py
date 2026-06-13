import unittest
import asyncio
from unittest.mock import AsyncMock, Mock, patch

from src.common.async_executor import AsyncExecutor


class AsyncExecutorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        AsyncExecutor._instance = None

    async def asyncTearDown(self):
        AsyncExecutor._instance = None

    async def test_close_shuts_down_event_loop_default_executor(self):
        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            executor = AsyncExecutor()

        await executor.start()
        loop = executor._loop

        with patch.object(loop, "shutdown_default_executor", new=AsyncMock()) as shutdown_default_executor:
            await executor.close()

        shutdown_default_executor.assert_awaited_once_with()

    async def test_new_instance_after_close_is_fresh_executor(self):
        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            executor = AsyncExecutor()

        await executor.start()
        await executor.close()

        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            next_executor = AsyncExecutor()

        self.assertIsNot(next_executor, executor)
        self.assertFalse(next_executor.is_running)


class AsyncExecutorRunLifecycleTest(unittest.TestCase):
    def setUp(self):
        AsyncExecutor._instance = None

    def tearDown(self):
        AsyncExecutor._instance = None

    def test_sync_run_uses_fresh_loop_after_default_executor_shutdown(self):
        async def close_executor():
            executor = AsyncExecutor()
            await executor.start()
            await executor.close()

        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            AsyncExecutor().run(close_executor())

        with patch("common.logging.LogManager.get_logger", return_value=Mock()):
            executor = AsyncExecutor()
            result = executor.run(asyncio.to_thread(lambda: "ok"))
            executor.run(executor.close())

        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
