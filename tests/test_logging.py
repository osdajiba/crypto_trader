import threading
import unittest
from unittest.mock import Mock, patch

from common.logging import AsyncLogHandler


class AsyncLogHandlerTest(unittest.TestCase):
    def test_flush_does_not_deadlock_when_logging_lock_is_already_held(self):
        handler = AsyncLogHandler(Mock())
        done = threading.Event()

        def flush_while_handler_lock_is_held():
            handler.acquire()
            try:
                handler.flush()
            finally:
                handler.release()
                handler.close()
                done.set()

        thread = threading.Thread(target=flush_while_handler_lock_is_held, daemon=True)
        thread.start()
        thread.join(timeout=1)

        self.assertTrue(done.is_set())


if __name__ == "__main__":
    unittest.main()
