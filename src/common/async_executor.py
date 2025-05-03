#!/usr/bin/env python3
# src/common/async_executor.py

import asyncio
import time
import uuid
from typing import Dict, List, Set, Any, Optional, Callable, TypeVar, Coroutine
from enum import Enum
import logging
from collections import deque

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.async_executor")

# 类型变量
T = TypeVar('T')

class TaskPriority(Enum):
    """任务优先级"""
    HIGH = 0
    NORMAL = 1
    LOW = 2
    BACKGROUND = 3

class AsyncExecutor:
    """
    优化的异步执行器，提供任务管理和资源控制
    
    特点:
    - 支持任务优先级
    - 支持任务超时
    - 支持任务取消和中断
    - 支持资源限制
    - 支持批量任务提交
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取执行器单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """初始化执行器"""
        self._running = False
        self._loop = None
        self._tasks = {}  # 类型: Dict[str, asyncio.Task]
        self._task_info = {}  # 类型: Dict[str, Dict[str, Any]]
        self._pending_by_priority = {p: deque() for p in TaskPriority}
        
        # 资源限制
        self._max_concurrent_tasks = 100
        self._active_tasks = set()
        self._task_semaphore = None
        
        # 监控数据
        self._completed_count = 0
        self._failed_count = 0
        self._cancelled_count = 0
        
        # 状态标志
        self._initialized = False
        self._shutdown_event = None
    
    async def start(self):
        """启动执行器"""
        if self._running:
            return
            
        try:
            # 获取或创建事件循环
            if asyncio.get_event_loop().is_running():
                self._loop = asyncio.get_event_loop()
            else:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                
            # 初始化状态
            self._running = True
            self._shutdown_event = asyncio.Event()
            self._task_semaphore = asyncio.Semaphore(self._max_concurrent_tasks)
            self._initialized = True
            
            logger.info("AsyncExecutor started")
        except Exception as e:
            logger.error(f"Error starting AsyncExecutor: {e}")
            raise
    
    async def submit(self, coro_func: Callable[..., Coroutine[Any, Any, T]], 
                   *args, task_id: Optional[str] = None, name: Optional[str] = None,
                   priority: TaskPriority = TaskPriority.NORMAL,
                   timeout: Optional[float] = None, **kwargs) -> T:
        """
        提交异步任务
        
        Args:
            coro_func: 异步函数
            *args: 函数参数
            task_id: 任务ID
            name: 任务名称
            priority: 任务优先级
            timeout: 任务超时时间(秒)
            **kwargs: 函数关键字参数
            
        Returns:
            T: 任务结果
            
        Raises:
            TimeoutError: 任务超时
            asyncio.CancelledError: 任务被取消
        """
        if not self._running:
            await self.start()
            
        # 生成任务ID
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        # 生成任务名称
        if name is None:
            name = f"{coro_func.__name__}_{task_id[:8]}"
            
        # 创建协程
        coro = coro_func(*args, **kwargs)
        
        # 创建任务
        task = self._loop.create_task(self._execute_with_timeout(coro, timeout), name=name)
        
        # 存储任务信息
        self._tasks[task_id] = task
        self._task_info[task_id] = {
            'id': task_id,
            'name': name,
            'priority': priority,
            'created_at': time.time(),
            'timeout': timeout,
            'status': 'running'
        }
        
        # 设置完成回调
        task.add_done_callback(lambda t: self._on_task_done(task_id, t))
        
        # 等待任务完成
        try:
            return await task
        finally:
            # 清理任务
            if task_id in self._tasks:
                del self._tasks[task_id]
    
    async def _execute_with_timeout(self, coro: Coroutine, timeout: Optional[float] = None) -> T:
        """
        使用超时执行协程
        
        Args:
            coro: 协程
            timeout: 超时时间(秒)
            
        Returns:
            T: 协程结果
            
        Raises:
            TimeoutError: 超时异常
        """
        # 获取任务执行权
        async with self._task_semaphore:
            task_id = id(coro)
            self._active_tasks.add(task_id)
            
            try:
                if timeout is not None:
                    return await asyncio.wait_for(coro, timeout=timeout)
                else:
                    return await coro
            finally:
                self._active_tasks.discard(task_id)
    
    def _on_task_done(self, task_id: str, task: asyncio.Task) -> None:
        """
        任务完成回调
        
        Args:
            task_id: 任务ID
            task: 任务对象
        """
        if task_id not in self._task_info:
            return
            
        info = self._task_info[task_id]
        
        try:
            if task.cancelled():
                info['status'] = 'cancelled'
                self._cancelled_count += 1
                logger.debug(f"Task {info['name']} (ID: {task_id}) was cancelled")
            elif task.exception() is not None:
                info['status'] = 'failed'
                info['error'] = str(task.exception())
                self._failed_count += 1
                logger.error(f"Task {info['name']} (ID: {task_id}) failed: {task.exception()}")
            else:
                info['status'] = 'completed'
                self._completed_count += 1
                logger.debug(f"Task {info['name']} (ID: {task_id}) completed successfully")
                
            # 记录执行时间
            info['completed_at'] = time.time()
            info['duration'] = info['completed_at'] - info['created_at']
            
        except asyncio.InvalidStateError:
            # 任务状态无效
            pass
        finally:
            # 从活动任务中移除
            self._active_tasks.discard(id(task))
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务ID
            
        Returns:
            bool: 是否成功取消
        """
        if task_id not in self._tasks:
            return False
            
        task = self._tasks[task_id]
        
        if task.done():
            return False
            
        task.cancel()
        
        # 等待任务取消完成
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
            
        return True
    
    async def cancel_all_tasks(self) -> int:
        """
        取消所有任务
        
        Returns:
            int: 取消的任务数量
        """
        task_ids = list(self._tasks.keys())
        count = 0
        
        for task_id in task_ids:
            if await self.cancel_task(task_id):
                count += 1
                
        return count
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> List[str]:
        """
        等待所有任务完成
        
        Args:
            timeout: 等待超时时间(秒)
            
        Returns:
            List[str]: 未完成的任务ID列表
        """
        if not self._tasks:
            return []
            
        # 获取所有未完成的任务
        pending_tasks = [task for task in self._tasks.values() if not task.done()]
        
        if not pending_tasks:
            return []
            
        # 等待任务完成
        done, pending = await asyncio.wait(pending_tasks, timeout=timeout)
        
        # 查找未完成任务的ID
        pending_ids = []
        for task_id, task in self._tasks.items():
            if task in pending:
                pending_ids.append(task_id)
                
        return pending_ids
    
    async def batch_submit(self, coro_funcs: List[Callable[..., Coroutine[Any, Any, T]]], 
                          *args, max_concurrency: Optional[int] = None, 
                          timeout: Optional[float] = None, **kwargs) -> List[T]:
        """
        批量提交任务
        
        Args:
            coro_funcs: 异步函数列表
            *args: 函数参数
            max_concurrency: 最大并发数
            timeout: 总体超时时间(秒)
            **kwargs: 函数关键字参数
            
        Returns:
            List[T]: 任务结果列表
            
        Raises:
            TimeoutError: 批处理超时
        """
        if not self._running:
            await self.start()
            
        if max_concurrency is None:
            max_concurrency = self._max_concurrent_tasks
            
        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def run_with_semaphore(func, *inner_args, **inner_kwargs):
            async with semaphore:
                return await func(*inner_args, **inner_kwargs)
        
        # 创建任务
        tasks = [
            self.submit(
                run_with_semaphore, 
                func, *args, 
                name=f"batch_{i}_{func.__name__}", 
                **kwargs
            )
            for i, func in enumerate(coro_funcs)
        ]
        
        # 等待所有任务完成
        if timeout is not None:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
            return results
        else:
            return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取执行器统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'running': self._running,
            'active_tasks': len(self._active_tasks),
            'completed_count': self._completed_count,
            'failed_count': self._failed_count,
            'cancelled_count': self._cancelled_count,
            'total_tasks': len(self._tasks),
            'max_concurrent_tasks': self._max_concurrent_tasks
        }
    
    def get_task_info(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        获取任务信息
        
        Args:
            task_id: 任务ID，None表示获取所有任务信息
            
        Returns:
            Dict[str, Any]: 任务信息
        """
        if task_id:
            return self._task_info.get(task_id, {})
        else:
            return {tid: info.copy() for tid, info in self._task_info.items()}
    
    def get_running_tasks(self) -> List[Dict[str, Any]]:
        """
        获取运行中的任务
        
        Returns:
            List[Dict[str, Any]]: 任务信息列表
        """
        return [
            info for tid, info in self._task_info.items()
            if tid in self._tasks and not self._tasks[tid].done()
        ]
    
    def set_max_concurrent_tasks(self, max_tasks: int) -> None:
        """
        设置最大并发任务数
        
        Args:
            max_tasks: 最大任务数
        """
        if max_tasks < 1:
            raise ValueError("Maximum concurrent tasks must be at least 1")
            
        self._max_concurrent_tasks = max_tasks
        
        # 更新信号量
        if self._task_semaphore:
            self._task_semaphore = asyncio.Semaphore(max_tasks)
    
    async def close(self) -> None:
        """关闭执行器"""
        if not self._running:
            return
            
        self._running = False
        
        # 取消所有任务
        await self.cancel_all_tasks()
        
        # 清空状态
        self._tasks.clear()
        self._task_info.clear()
        self._active_tasks.clear()
        
        # 信号关闭完成
        if self._shutdown_event:
            self._shutdown_event.set()
            
        logger.info("AsyncExecutor closed")