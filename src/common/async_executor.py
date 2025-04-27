# src/common/async_executor.py

import threading
import asyncio
import time
from typing import Optional, Callable, Any, Dict, TypeVar, Coroutine, List, Set
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import uuid
import signal
import logging


T = TypeVar('T')
R = TypeVar('R')


class TaskPriority(Enum):
    """Task priority levels for execution ordering"""
    HIGH = 0
    NORMAL = 1
    LOW = 2
    BACKGROUND = 3


@dataclass
class TaskInfo:
    """Detailed information about a scheduled task"""
    id: str
    task: asyncio.Task
    name: str = "unnamed_task"
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    last_run_at: Optional[float] = None
    run_count: int = 0
    is_periodic: bool = False
    period: Optional[float] = None
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, TaskInfo):
            return False
        return self.id == other.id

class AsyncExecutor:
    """
    Enhanced async executor providing streamlined task management and resource lifecycle control
    with support for prioritization, scheduling, periodic tasks and metrics
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_executor(*args, **kwargs)
            return cls._instance
    
    def _init_executor(self, logger=None):
        """Initialize executor resources with injected logger"""
        # Try to get a logger - fall back to a basic logger if LogManager isn't initialized
        if logger:
            # Use the provided logger if available
            self._logger = logger
        else:
            try:
                # Try to use LogManager 
                try:
                    from common.log_manager import LogManager
                    self._logger = LogManager.get_logger("system.async_executor")
                except (ImportError, RuntimeError):
                    # Try alternative import path
                    try:
                        from src.common.log_manager import LogManager
                        self._logger = LogManager.get_logger("system.async_executor")
                    except (ImportError, RuntimeError):
                        # Fall back to basic logger
                        self._logger = self._create_default_logger()
            except Exception as e:
                # Last resort - create a very basic logger
                self._logger = self._create_default_logger()
                self._logger.warning(f"Using fallback logger due to: {str(e)}")
                
        # Initialize other attributes
        self._tasks: Dict[str, TaskInfo] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._in_cleanup = False  # Flag to prevent recursion during cleanup
        self._scheduler_task = None
        self._periodic_tasks: Set[str] = set()
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._throttle_limit = 50  # Maximum number of concurrent tasks
        self._metrics = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0,
        }
        self._signal_handlers_installed = False
    
    def _create_default_logger(self):
        """Create a default logger when LogManager is not available"""
        logger = logging.getLogger("system.async_executor")
        if not logger.handlers:  # Only add handler if none exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    async def __aenter__(self):
        """Support for async context manager pattern"""
        if not self._running:
            await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure proper resource cleanup"""
        await self.close()
    
    @asynccontextmanager
    async def task_context(self, task_id: Optional[str] = None, name: str = None, 
                          priority: TaskPriority = TaskPriority.NORMAL, 
                          timeout: Optional[float] = None):
        """
        Async context manager for automatic task tracking and cleanup
        
        Args:
            task_id: Optional task identifier
            name: Optional descriptive name
            priority: Task priority level
            timeout: Optional timeout in seconds
            
        Yields:
            Task ID
        """
        if task_id is None:
            task_id = f"task_{uuid.uuid4()}"
        
        task_name = name or f"context_task_{task_id}"
            
        try:
            yield task_id
        finally:
            if task_id in self._tasks:
                await self.cancel_task(task_id)
                
    async def start(self):
        """Start the executor and initialize background tasks"""
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
                
                # Start the task scheduler for delayed and periodic tasks
                self._scheduler_task = asyncio.create_task(
                    self._task_scheduler(),
                    name="task_scheduler"
                )
                
                # Install signal handlers if not already installed
                if not self._signal_handlers_installed:
                    self._install_signal_handlers()
                
                self._logger.info("AsyncExecutor started")
    
    def _install_signal_handlers(self):
        """Install signal handlers for graceful shutdown"""
        try:
            # Only install in main thread to avoid issues
            if threading.current_thread() is threading.main_thread():
                signals = [signal.SIGINT, signal.SIGTERM]
                for sig in signals:
                    self._loop.add_signal_handler(
                        sig,
                        lambda s=sig: asyncio.create_task(self._handle_signal(s))
                    )
                self._signal_handlers_installed = True
                self._logger.debug("Signal handlers installed")
        except (NotImplementedError, ValueError, RuntimeError) as e:
            # Signal handling is not available on this platform
            self._logger.warning(f"Signal handlers not installed: {e}")
            self._signal_handlers_installed = False
    
    async def _handle_signal(self, sig: int):
        """Handle termination signals for graceful shutdown"""
        self._logger.info(f"Received signal {sig.name}, shutting down...")
        await self.close()
    
    async def _task_scheduler(self):
        """Background task that manages scheduled and periodic tasks"""
        self._logger.debug("Task scheduler started")
        try:
            while self._running:
                # Check for delayed tasks that are due
                current_time = time.time()
                tasks_to_run = []
                
                with self._lock:
                    for task_id, task_info in list(self._tasks.items()):
                        # Skip tasks that are already running or don't have a scheduled time
                        if task_info.scheduled_at is None or task_info.task.done() is False:
                            continue
                            
                        if current_time >= task_info.scheduled_at:
                            tasks_to_run.append(task_info)
                
                # Execute due tasks
                for task_info in tasks_to_run:
                    self._execute_scheduled_task(task_info)
                
                # Sleep a short time before checking again
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self._logger.debug("Task scheduler cancelled")
        except Exception as e:
            self._logger.error(f"Error in task scheduler: {str(e)}", exc_info=True)
        finally:
            self._logger.debug("Task scheduler stopped")
    
    def _execute_scheduled_task(self, task_info: TaskInfo):
        """Execute a scheduled task when it's due"""
        if not self._running or task_info.id not in self._tasks:
            return
            
        task_info.last_run_at = time.time()
        task_info.run_count += 1
        
        # For periodic tasks, schedule the next run
        if task_info.is_periodic and task_info.period is not None:
            task_info.scheduled_at = task_info.last_run_at + task_info.period
        else:
            task_info.scheduled_at = None
            
        # Create and execute the actual task
        self._logger.debug(f"Executing scheduled task: {task_info.name} (ID: {task_info.id})")
        task_info.task = asyncio.create_task(
            self._run_task_with_timeout(
                task_info.task,
                task_info.timeout
            ),
            name=task_info.name
        )
        
        # Set completion callback
        task_info.task.add_done_callback(
            lambda t, tid=task_info.id: self._on_task_done(tid, t)
        )
    
    async def _run_task_with_timeout(self, coro: Coroutine, timeout: Optional[float] = None):
        """Run a coroutine with an optional timeout"""
        if timeout is not None:
            try:
                return await asyncio.wait_for(coro, timeout=timeout)
            except asyncio.TimeoutError:
                self._logger.warning(f"Task timed out after {timeout}s")
                raise
        else:
            return await coro
    
    def _on_task_done(self, task_id: str, task: asyncio.Task):
        """Handle task completion, rescheduling periodic tasks if needed"""
        with self._lock:
            if task_id not in self._tasks:
                return
                
            task_info = self._tasks[task_id]
            
            # Check task result
            try:
                if task.cancelled():
                    self._metrics["tasks_cancelled"] += 1
                    self._logger.debug(f"Task cancelled: {task_info.name} (ID: {task_id})")
                elif task.exception() is not None:
                    self._metrics["tasks_failed"] += 1
                    exc = task.exception()
                    self._logger.error(f"Task failed: {task_info.name} (ID: {task_id}): {exc}")
                else:
                    self._metrics["tasks_completed"] += 1
                    self._logger.debug(f"Task completed: {task_info.name} (ID: {task_id})")
            except asyncio.InvalidStateError:
                # Task is not yet done
                pass
                
            # For non-periodic tasks, remove from registry
            if not task_info.is_periodic:
                del self._tasks[task_id]
            elif task_info.is_periodic and task_info.period is not None and not task.cancelled():
                # For periodic tasks, keep in registry with new scheduled time
                # The scheduler will pick this up and execute again
                pass
    
    async def submit(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                    *args, task_id: Optional[str] = None, name: Optional[str] = None,
                    priority: TaskPriority = TaskPriority.NORMAL,
                    timeout: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None,
                    **kwargs) -> T:
        """
        Submit a coroutine function for async execution
        
        Args:
            coro_func: Coroutine function to execute
            *args: Arguments to pass to the coroutine function
            task_id: Optional task identifier for tracking and cancellation
            name: Optional descriptive name for the task
            priority: Priority level for execution ordering
            timeout: Maximum execution time in seconds
            metadata: Optional dictionary with additional task information
            **kwargs: Keyword arguments to pass to the coroutine function
            
        Returns:
            Result of the coroutine execution
            
        Raises:
            RuntimeError: If executor is not running
        """
        if not self._running:
            await self.start()
        
        if task_id is None:
            task_id = f"task_{uuid.uuid4()}"
            
        if name is None:
            name = getattr(coro_func, "__name__", "unnamed_function")
        
        # Update metrics
        self._metrics["tasks_submitted"] += 1
        
        try:
            # Create the coroutine
            coro = coro_func(*args, **kwargs)
            if not asyncio.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
                
            # Create task and register it
            task = asyncio.create_task(coro, name=name)
            
            # Create task info
            task_info = TaskInfo(
                id=task_id,
                task=task,
                name=name,
                priority=priority,
                created_at=time.time(),
                timeout=timeout,
                metadata=metadata or {}
            )
            
            # Register task
            with self._lock:
                self._tasks[task_id] = task_info
            
            # Set completion callback
            task.add_done_callback(
                lambda t, tid=task_id: self._cleanup_task(tid)
            )
            
            # Wait for task completion and return result
            if timeout is not None:
                return await asyncio.wait_for(task, timeout=timeout)
            else:
                return await task
            
        except asyncio.CancelledError:
            self._logger.info(f"Task {name} (ID: {task_id}) was cancelled")
            self._metrics["tasks_cancelled"] += 1
            raise
        except asyncio.TimeoutError:
            self._logger.warning(f"Task {name} (ID: {task_id}) timed out after {timeout}s")
            self._metrics["tasks_failed"] += 1
            
            # Cancel the task if it's still running
            with self._lock:
                if task_id in self._tasks and not self._tasks[task_id].task.done():
                    self._tasks[task_id].task.cancel()
            
            raise
        except Exception as e:
            self._logger.error(f"Error in task {name} (ID: {task_id}): {str(e)}", exc_info=True)
            self._metrics["tasks_failed"] += 1
            raise
    
    async def submit_with_cancellation_handling(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                                               *args, task_id: Optional[str] = None, name: Optional[str] = None,
                                               **kwargs) -> T:
        """
        Submit a task with proper cancellation handling
        
        Args:
            coro_func: Coroutine function to execute
            *args: Arguments to pass to the function
            task_id: Optional task identifier
            name: Optional descriptive name
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function
        """
        try:
            return await self.submit(coro_func, *args, task_id=task_id, name=name, **kwargs)
        except asyncio.CancelledError:
            task_name = name or getattr(coro_func, "__name__", "unnamed_function")
            self._logger.info(f"Task {task_name} was cancelled")
            raise  # Re-raise to propagate cancellation
    
    async def schedule(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                      delay: float, *args, task_id: Optional[str] = None, 
                      name: Optional[str] = None, priority: TaskPriority = TaskPriority.NORMAL,
                      timeout: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None,
                      **kwargs) -> str:
        """
        Schedule a task to run after a delay
        
        Args:
            coro_func: Coroutine function to execute
            delay: Delay in seconds before execution
            *args: Arguments to pass to the function
            task_id: Optional task identifier
            name: Optional descriptive name
            priority: Priority level for execution
            timeout: Maximum execution time
            metadata: Optional task metadata
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task ID for the scheduled task
        """
        if not self._running:
            await self.start()
            
        if task_id is None:
            task_id = f"scheduled_{uuid.uuid4()}"
            
        if name is None:
            name = f"scheduled_{getattr(coro_func, '__name__', 'task')}"
            
        # Create a placeholder coroutine
        async def placeholder_coro():
            # This will be replaced when the task is actually executed
            pass
            
        # Create placeholder task
        task = asyncio.create_task(placeholder_coro(), name=f"placeholder_{name}")
        
        # Create task info with scheduled time
        task_info = TaskInfo(
            id=task_id,
            task=task,
            name=name,
            priority=priority,
            created_at=time.time(),
            scheduled_at=time.time() + delay,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        # Store the actual coroutine function and arguments
        task_info.metadata["coro_func"] = coro_func
        task_info.metadata["args"] = args
        task_info.metadata["kwargs"] = kwargs
        
        # Register the task
        with self._lock:
            self._tasks[task_id] = task_info
            
        self._logger.debug(f"Scheduled task {name} (ID: {task_id}) to run in {delay}s")
        return task_id
    
    async def schedule_periodic(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                               period: float, *args, initial_delay: Optional[float] = None,
                               task_id: Optional[str] = None, name: Optional[str] = None,
                               priority: TaskPriority = TaskPriority.NORMAL, 
                               timeout: Optional[float] = None, 
                               metadata: Optional[Dict[str, Any]] = None,
                               **kwargs) -> str:
        """
        Schedule a task to run periodically
        
        Args:
            coro_func: Coroutine function to execute
            period: Time between executions in seconds
            *args: Arguments to pass to the function
            initial_delay: Optional delay before first execution
            task_id: Optional task identifier
            name: Optional descriptive name
            priority: Priority level for execution
            timeout: Maximum execution time per invocation
            metadata: Optional task metadata
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Task ID for the periodic task
        """
        if not self._running:
            await self.start()
            
        if task_id is None:
            task_id = f"periodic_{uuid.uuid4()}"
            
        if name is None:
            name = f"periodic_{getattr(coro_func, '__name__', 'task')}"
            
        # Set initial delay if not provided
        if initial_delay is None:
            initial_delay = 0
            
        # Create a placeholder coroutine
        async def placeholder_coro():
            # This will be replaced when the task is actually executed
            try:
                await coro_func(*args, **kwargs)
            except Exception as e:
                self._logger.error(f"Error in periodic task {name} (ID: {task_id}): {str(e)}", exc_info=True)
            
        # Create placeholder task
        task = asyncio.create_task(placeholder_coro(), name=f"placeholder_{name}")
        
        # Create task info with periodic settings
        task_info = TaskInfo(
            id=task_id,
            task=task,
            name=name,
            priority=priority,
            created_at=time.time(),
            scheduled_at=time.time() + initial_delay,
            is_periodic=True,
            period=period,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        # Store the actual coroutine function and arguments
        task_info.metadata["coro_func"] = coro_func
        task_info.metadata["args"] = args
        task_info.metadata["kwargs"] = kwargs
        
        # Register the task
        with self._lock:
            self._tasks[task_id] = task_info
            self._periodic_tasks.add(task_id)
            
        self._logger.debug(f"Scheduled periodic task {name} (ID: {task_id}) with period {period}s")
        return task_id
    
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
            wrapper = lambda: coro  # Using coroutine object, not coroutine function
            
            # Submit and execute
            return loop.run_until_complete(self.submit(wrapper))
        except Exception as e:
            self._logger.error(f"Async task failed: {str(e)}", exc_info=True)
            raise
        finally:
            # Don't close the event loop as it might be used elsewhere
            pass
    
    def _cleanup_task(self, task_id: str):
        """Remove completed task from registry"""
        with self._lock:
            if task_id in self._tasks:
                # Don't remove periodic tasks
                if task_id not in self._periodic_tasks:
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
                task_info = self._tasks[task_id]
                if not task_info.task.done():
                    task_info.task.cancel()
                    self._logger.info(f"Task {task_info.name} (ID: {task_id}) cancelled")
                    
                    # For periodic tasks, also remove from periodic set
                    if task_id in self._periodic_tasks:
                        self._periodic_tasks.remove(task_id)
                        
                    # Remove from task registry
                    del self._tasks[task_id]
                    
                    self._metrics["tasks_cancelled"] += 1
                    return True
                else:
                    # Task is already done, just remove it
                    if task_id in self._periodic_tasks:
                        self._periodic_tasks.remove(task_id)
                    del self._tasks[task_id]
        return False
    
    async def cancel_all_tasks(self):
        """Cancel all running tasks"""
        with self._lock:
            task_ids = list(self._tasks.keys())
        
        cancelled_count = 0
        for task_id in task_ids:
            if await self.cancel_task(task_id):
                cancelled_count += 1
                
        self._logger.info(f"Cancelled {cancelled_count} tasks")
        return cancelled_count
    
    async def close(self):
        """Close the executor and clean up resources"""
        if not self._running:
            self._logger.debug("AsyncExecutor is already closed")
            return

        # Prevent recursive calls
        if self._in_cleanup:
            self._logger.debug("Already in cleanup process, skipping")
            return
            
        self._in_cleanup = True
        self._logger.debug("Closing AsyncExecutor...")
        
        # Set shutdown flag
        self._running = False
        self._shutdown_event.set()
        
        try:
            # Cancel the task scheduler first
            if self._scheduler_task and not self._scheduler_task.done():
                if self._scheduler_task.get_loop() != asyncio.get_running_loop():
                    self._logger.warning("Scheduler task belongs to a different loop, skipping cancel")
                else:
                    self._scheduler_task.cancel()
                    try:
                        await asyncio.wait_for(asyncio.shield(self._scheduler_task), timeout=0.5)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
            
            # Get task snapshot to avoid modification during iteration
            with self._lock:
                tasks_to_cancel = list(self._tasks.items())
                self._periodic_tasks.clear()
            
            # Cancel each task individually
            for task_id, task_info in tasks_to_cancel:
                if not task_info.task.done():
                    if task_info.task.get_loop() != asyncio.get_running_loop():
                        self._logger.warning(f"Task {task_info.name} belongs to a different loop, skipping cancel")
                        continue
                    task_info.task.cancel()
            
            # Wait for a short period for tasks to cancel
            if tasks_to_cancel:
                try:
                    await asyncio.sleep(0.2)
                except asyncio.CancelledError:
                    pass
            
            # Clear task registry
            with self._lock:
                self._tasks.clear()
                
            self._logger.debug("AsyncExecutor closed successfully")
            
        except Exception as e:
            self._logger.error(f"Error while closing AsyncExecutor: {str(e)}", exc_info=True)
        finally:
            self._in_cleanup = False
    
    @property
    def task_count(self) -> int:
        """Get current active task count"""
        return len(self._tasks)
    
    @property
    def is_running(self) -> bool:
        """Check if executor is running"""
        return self._running
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            return dict(self._metrics)
    
    async def get_active_tasks(self) -> List[Dict[str, Any]]:
        """
        Get information about all active tasks
        
        Returns:
            List of task information dictionaries
        """
        result = []
        with self._lock:
            for task_id, task_info in self._tasks.items():
                # Skip completed tasks
                if task_info.task.done():
                    continue
                    
                # Create serializable task info
                task_data = {
                    "id": task_info.id,
                    "name": task_info.name,
                    "priority": task_info.priority.name,
                    "created_at": task_info.created_at,
                    "is_periodic": task_info.is_periodic,
                    "run_count": task_info.run_count,
                }
                
                if task_info.scheduled_at:
                    task_data["scheduled_at"] = task_info.scheduled_at
                    task_data["time_until_next_run"] = max(0, task_info.scheduled_at - time.time())
                    
                if task_info.last_run_at:
                    task_data["last_run_at"] = task_info.last_run_at
                    
                if task_info.period:
                    task_data["period"] = task_info.period
                    
                # Add safe metadata (omit callable objects)
                safe_metadata = {}
                for k, v in task_info.metadata.items():
                    if k not in ("coro_func", "args", "kwargs") and not callable(v):
                        safe_metadata[k] = v
                        
                task_data["metadata"] = safe_metadata
                
                result.append(task_data)
                
        return result
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> List[TaskInfo]:
        """
        Wait for all tasks to complete with optional timeout
        
        Args:
            timeout: Maximum time to wait in seconds (None means wait indefinitely)
            
        Returns:
            List of TaskInfo objects for tasks that didn't complete within the timeout
        """
        if not self._tasks:
            return []
            
        with self._lock:
            task_infos = list(self._tasks.values())
            pending_tasks = [info.task for info in task_infos if not info.task.done()]
            
        if not pending_tasks:
            return []
            
        try:
            done, pending = await asyncio.wait(
                pending_tasks, 
                timeout=timeout, 
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Find task infos for pending tasks
            pending_infos = []
            with self._lock:
                for task_id, task_info in self._tasks.items():
                    if task_info.task in pending:
                        pending_infos.append(task_info)
                        
            return pending_infos
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
            
    @asynccontextmanager
    async def throttle(self, limit: int = None):
        """
        Context manager for throttling task execution
        
        Args:
            limit: Maximum number of concurrent tasks (uses default if None)
            
        Yields:
            None
        """
        if limit is None:
            limit = self._throttle_limit
            
        semaphore = asyncio.Semaphore(limit)
        await semaphore.acquire()
        try:
            yield
        finally:
            semaphore.release()
            
    async def batch_submit(self, coro_funcs: List[Callable[..., Coroutine[Any, Any, R]]], 
                          *args, max_concurrency: int = None, 
                          timeout: Optional[float] = None, **kwargs) -> List[R]:
        """
        Submit a batch of coroutine functions for execution with controlled concurrency
        
        Args:
            coro_funcs: List of coroutine functions to execute
            *args: Arguments to pass to each function
            max_concurrency: Maximum number of concurrent executions
            timeout: Maximum total execution time
            **kwargs: Keyword arguments to pass to each function
            
        Returns:
            List of results in the same order as the input functions
        """
        if not self._running:
            await self.start()
            
        if max_concurrency is None:
            max_concurrency = self._throttle_limit
            
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def run_with_semaphore(func, *inner_args, **inner_kwargs):
            async with semaphore:
                return await func(*inner_args, **inner_kwargs)
        
        # Create tasks with semaphore
        tasks = [
            self.submit(
                run_with_semaphore, 
                func, *args, 
                name=f"batch_task_{i}_{getattr(func, '__name__', 'func')}", 
                **kwargs
            )
            for i, func in enumerate(coro_funcs)
        ]
        
        # Wait for all tasks with optional timeout
        if timeout is not None:
            return await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        else:
            return await asyncio.gather(*tasks)