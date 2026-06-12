import unittest
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


if __name__ == "__main__":
    unittest.main()
