# src/common/async_executor.py

import threading
import asyncio
from typing import Optional, Callable, Any, Dict, TypeVar, Generic, Coroutine
import logging
from contextlib import suppress, asynccontextmanager

T = TypeVar('T')

class AsyncExecutor:
    """增强的异步执行器，提供简洁的任务管理和资源生命周期控制"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_executor()
            return cls._instance
    
    def _init_executor(self):
        """初始化执行器资源"""
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tasks: Dict[str, asyncio.Task] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    async def __aenter__(self):
        """支持异步上下文管理器"""
        if not self._running:
            await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """确保资源正确清理"""
        await self.close()
    
    @asynccontextmanager
    async def task_context(self, task_id: Optional[str] = None):
        """
        异步上下文管理器，用于自动跟踪和清理任务
        
        Args:
            task_id: 可选的任务ID
            
        Yields:
            任务ID
        """
        if task_id is None:
            task_id = f"task_{id(object())}_{len(self._tasks)}"
            
        try:
            yield task_id
        finally:
            if task_id in self._tasks:
                await self.cancel_task(task_id)
                
    async def start(self):
        """启动执行器"""
        if self._running:
            return
            
        with self._lock:
            if not self._running:  # 锁内二次检查
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
        提交协程函数异步执行
        
        Args:
            coro_func: 要执行的协程函数
            *args: 传递给协程函数的参数
            task_id: 可选的任务标识符，用于跟踪和取消
            **kwargs: 传递给协程函数的关键字参数
            
        Returns:
            协程执行的结果
            
        Raises:
            RuntimeError: 如果执行器未运行
        """
        if not self._running:
            await self.start()
        
        if task_id is None:
            task_id = f"task_{id(coro_func)}_{len(self._tasks)}"
        
        try:
            # 创建并注册任务
            coro = coro_func(*args, **kwargs)
            if not asyncio.iscoroutine(coro):
                raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
                
            task = asyncio.create_task(coro)
            self._tasks[task_id] = task
            
            # 设置完成回调以从注册表中删除任务
            task.add_done_callback(lambda t: self._cleanup_task(task_id))
            
            # 等待任务完成并返回结果
            return await task
            
        except asyncio.CancelledError:
            self._logger.info(f"Task {task_id} was cancelled")
            raise
        except Exception as e:
            self._logger.error(f"Error in task {task_id}: {str(e)}", exc_info=True)
            raise
    
    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        """
        同步运行一个协程，等待其完成
        
        Args:
            coro: 要运行的协程
            
        Returns:
            协程的结果
        """
        if not asyncio.iscoroutine(coro):
            raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
            
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # 确保执行器启动
            if not self._running:
                loop.run_until_complete(self.start())
            
            # 包装协程为不需要参数的lambda以适配submit方法
            wrapper = lambda: coro  # 注意这里使用协程对象，非协程函数
            
            # 提交并执行
            return loop.run_until_complete(self.submit(wrapper))
        except Exception as e:
            self._logger.error(f"Error during execution: {str(e)}", exc_info=True)
            raise
        finally:
            # 不关闭事件循环，因为它可能在其他地方使用
            pass
    
    def _cleanup_task(self, task_id: str):
        """从注册表中移除已完成的任务"""
        with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        通过ID取消特定任务
        
        Args:
            task_id: 要取消的任务ID
            
        Returns:
            如果找到并取消了任务则为True，否则为False
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
        """取消所有运行中的任务"""
        with self._lock:
            task_ids = list(self._tasks.keys())
        
        for task_id in task_ids:
            await self.cancel_task(task_id)
    
    async def close(self):
        """关闭执行器并清理所有资源"""
        if not self._running:
            return
            
        with self._lock:
            if self._running:
                self._running = False
                self._shutdown_event.set()
                
                # 取消所有待处理的任务
                await self.cancel_all_tasks()
                
                # 等待一段时间让任务正确取消
                await asyncio.sleep(0.1)
                
                # 最终清理任何剩余的任务
                for task_id, task in list(self._tasks.items()):
                    if not task.done():
                        with suppress(asyncio.CancelledError):
                            self._logger.warning(f"强制取消任务 {task_id}")
                            task.cancel()
                
                self._tasks.clear()
                self._logger.info("AsyncExecutor shutdown complete")
    
    @property
    def task_count(self) -> int:
        """获取当前活动任务数量"""
        return len(self._tasks)
    
    @property
    def is_running(self) -> bool:
        """检查执行器是否在运行"""
        return self._running