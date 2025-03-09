# src/common/async_executor.py

import threading
import asyncio
from typing import Optional, Callable, Any, Dict, TypeVar, Generic, Coroutine, List
import logging
from contextlib import asynccontextmanager
from src.common.log_manager import LogManager

T = TypeVar('T')
logger = LogManager.get_logger(name="trading_system")

class AsyncExecutor:
    """Enhanced async executor providing streamlined task management and resource lifecycle control"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_executor()
            return cls._instance
    
    def _init_executor(self):
        """Initialize executor resources"""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def __aenter__(self):
        """Support for async context manager pattern"""
        if not self._running:
            await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper resource cleanup"""
        await self.close()
    
    @asynccontextmanager
    async def task_context(self, task_id: Optional[str] = None):
        """
        Async context manager for automatic task tracking and cleanup
        
        Args:
            task_id: Optional task identifier
            
        Yields:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{id(object())}_{len(self._tasks)}"
            
        try:
            yield task_id
        finally:
            if task_id in self._tasks:
                await self.cancel_task(task_id)
                
    async def start(self):
        """Start the executor"""
        if self._running:
            return
            
        with self._lock:
            if not self._running:  # Double-check within lock
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(self._loop)
                    
                self._running = True
                self._shutdown_event.clear()
                self._logger.info("AsyncExecutor started")
    
    async def submit(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                    *args, task_id: Optional[str] = None, **kwargs) -> T:
        """
        Submit a coroutine function for async execution
        
        Args:
            coro_func: Coroutine function to execute
            *args: Arguments to pass to the coroutine function
            task_id: Optional task identifier for tracking and cancellation
            **kwargs: Keyword arguments to pass to the coroutine function
            
        Returns:
            Result of the coroutine execution
            
        Raises:
            RuntimeError: If executor is not running
        """
        if not self._running:
            await self.start()
        
        if task_id is None:
            task_id = f"task_{id(coro_func)}_{len(self._tasks)}"
        
        try:
            # Create and register the task
            coro = coro_func(*args, **kwargs)
            if not asyncio.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
                
            task = asyncio.create_task(coro)
            self._tasks[task_id] = task
            
            # Set completion callback to remove task from registry
            task.add_done_callback(lambda t: self._cleanup_task(task_id))
            
            # Wait for task completion and return result
            return await task
            
        except asyncio.CancelledError:
            self._logger.info(f"Task {task_id} was cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
            raise
    
    async def submit_with_cancellation_handling(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                                               *args, task_id: Optional[str] = None, **kwargs) -> T:
        """
        Submit a task with proper cancellation handling
        
        Args:
            coro_func: Coroutine function to execute
            *args: Arguments to pass to the function
            task_id: Optional task identifier
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
        """
        try:
            return await self.submit(coro_func, *args, task_id=task_id, **kwargs)
        except asyncio.CancelledError:
            self._logger.info(f"Task {coro_func.__name__} was cancelled")
            raise  # Re-raise to propagate cancellation
    
    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        Synchronously run a coroutine and wait for completion
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
            
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Ensure executor is started
            if not self._running:
                loop.run_until_complete(self.start())
            
            # Wrap coroutine to adapt to submit method
            wrapper = lambda: coro  # Note: using coroutine object, not coroutine function
            
            # Submit and execute
            return loop.run_until_complete(self.submit(wrapper))
        except Exception as e:
            self._logger.error(f"Error during execution: {str(e)}", exc_info=True)
            raise
        finally:
            # Don't close the event loop as it might be used elsewhere
            pass
    
    def _cleanup_task(self, task_id: str):
        """Remove completed task from registry"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a specific task by ID
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if task was found and cancelled, False otherwise
        """
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                if not task.done():
                    task.cancel()
                    self._logger.info(f"Task {task_id} cancelled")
                    return True
        return False
    
    async def cancel_all_tasks(self):
        """Cancel all running tasks"""
        with self._lock:
            task_ids = list(self._tasks.keys())
        
        for task_id in task_ids:
            await self.cancel_task(task_id)
    
    async def close(self):
        """Close the executor and clean up resources"""
        if not self._running:
            self._logger.debug("AsyncExecutor is already closed")
            return

        self._logger.debug("Closing AsyncExecutor...")
        
        # Set shutdown flag
        self._running = False
        
        try:
            try:
                # Cancel all running tasks
                for task_id, task in list(self._tasks.items()):
                    if not task.done():
                        task.cancel()
                        self._logger.debug(f"Task {task_id} cancelled during shutdown")
                
                # Wait for tasks to complete or be cancelled
                pending_tasks = [t for t in self._tasks.values() if not t.done()]
                if pending_tasks:
                    try:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)
                    except asyncio.CancelledError:
                        self._logger.debug("Task cancellation in progress")
                
                self._logger.debug("AsyncExecutor closed successfully")
            except asyncio.CancelledError:
                self._logger.debug("Executor closing was itself cancelled")
        except Exception as e:
            self._logger.error(f"Error while closing AsyncExecutor: {str(e)}")
        finally:
            # Clear task registry
            self._tasks.clear()
            
            try:
                # Final brief wait to ensure resources are released
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                # Ignore cancellation errors during shutdown
                pass
    
    @property
    def task_count(self) -> int:
        """Get current active task count"""
        return len(self._tasks)
    
    @property
    def is_running(self) -> bool:
        """Check if executor is running"""
        return self._running
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> List[asyncio.Task]:
        """
        Wait for all tasks to complete with optional timeout
        
        Args:
            timeout: Maximum time to wait in seconds (None means wait indefinitely)
            
        Returns:
            List of tasks that didn't complete within the timeout
        """
        if not self._tasks:
            return []
            
        with self._lock:
            pending = list(self._tasks.values())
            
        if not pending:
            return []
            
        try:
            done, pending = await asyncio.wait(
                pending, 
                timeout=timeout, 
                return_when=asyncio.ALL_COMPLETED
            )
            return list(pending)
        except asyncio.CancelledError:
            self._logger.info("wait_for_tasks was cancelled")
            raise
            
    async def sleep(self, duration: float) -> None:
        """
        Sleep that properly handles cancellation
        
        Args:
            duration: Sleep duration in seconds
        """
        try:
            await asyncio.sleep(duration)
        except asyncio.CancelledError:
            self._logger.debug(f"Sleep of {duration}s was cancelled")
            raise