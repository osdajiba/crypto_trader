#!/usr/bin/env python3
# src/common/resource_pool.py

import asyncio
import time
from typing import Dict, List, Any, Callable, Optional, TypeVar, Generic
import logging
from collections import deque

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.resource_pool")

T = TypeVar('T')

class ResourcePool(Generic[T]):
    """
    通用资源池，用于管理连接等有限资源
    
    特点:
    - 支持自动扩展和收缩
    - 支持健康检查
    - 支持资源超时和重置
    - 支持最大闲置时间
    """
    
    def __init__(self, 
                 factory: Callable[[], T],
                 close_func: Callable[[T], None] = None,
                 health_check: Optional[Callable[[T], bool]] = None,
                 min_size: int = 1,
                 max_size: int = 10,
                 max_idle_time: float = 60.0,  # 秒
                 acquisition_timeout: float = 10.0,  # 秒
                 cleanup_interval: float = 30.0):  # 秒
        """
        初始化资源池
        
        Args:
            factory: 创建资源的工厂函数
            close_func: 关闭资源的函数
            health_check: 资源健康检查函数
            min_size: 池中最小资源数
            max_size: 池中最大资源数
            max_idle_time: 资源最大闲置时间(秒)
            acquisition_timeout: 获取资源超时时间(秒)
            cleanup_interval: 清理间隔(秒)
        """
        self.factory = factory
        self.close_func = close_func
        self.health_check = health_check
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.acquisition_timeout = acquisition_timeout
        
        # 资源池状态
        self._resources = deque()  # 可用资源
        self._in_use = set()       # 使用中的资源
        self._resource_metadata = {}  # 资源元数据
        
        # 同步原语
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # 控制标志
        self._closed = False
        self._initialized = False
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(cleanup_interval))
        
    async def initialize(self):
        """初始化资源池，预创建最小数量的资源"""
        async with self._lock:
            if self._initialized:
                return
                
            logger.info(f"Initializing resource pool with min_size={self.min_size}")
            
            # 预创建资源
            for _ in range(self.min_size):
                try:
                    resource = await self._create_resource()
                    self._resources.append(resource)
                except Exception as e:
                    logger.error(f"Error creating initial resource: {e}")
                    
            self._initialized = True
        
    async def _create_resource(self) -> T:
        """创建新资源并记录元数据"""
        resource = await self.factory()
        
        self._resource_metadata[id(resource)] = {
            'created_at': time.time(),
            'last_used': time.time(),
            'use_count': 0
        }
        
        return resource
    
    async def _cleanup_loop(self, interval: float):
        """定期清理过期资源"""
        try:
            while not self._closed:
                await asyncio.sleep(interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            logger.debug("Cleanup task cancelled")
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
            
    async def _cleanup_expired(self):
        """清理过期的闲置资源"""
        if self._closed:
            return
            
        now = time.time()
        to_remove = []
        
        async with self._lock:
            # 保持最小池大小
            if len(self._resources) <= self.min_size:
                return
                
            # 检查过期资源
            for i, resource in enumerate(self._resources):
                metadata = self._resource_metadata.get(id(resource))
                
                if not metadata:
                    to_remove.append(i)
                    continue
                    
                # 检查闲置时间
                idle_time = now - metadata['last_used']
                if idle_time > self.max_idle_time:
                    to_remove.append(i)
                    
            # 从后向前移除资源
            for i in sorted(to_remove, reverse=True):
                if len(self._resources) <= self.min_size:
                    break
                    
                try:
                    resource = self._resources[i]
                    self._resources.remove(resource)
                    
                    # 关闭资源
                    if self.close_func:
                        await self.close_func(resource)
                        
                    # 清理元数据
                    self._resource_metadata.pop(id(resource), None)
                    
                except (IndexError, ValueError):
                    pass
                except Exception as e:
                    logger.error(f"Error closing expired resource: {e}")
    
    async def acquire(self) -> T:
        """
        获取资源
        
        Returns:
            T: 池中资源
            
        Raises:
            TimeoutError: 超时未获取到资源
            RuntimeError: 资源池已关闭
        """
        if self._closed:
            raise RuntimeError("Resource pool is closed")
            
        if not self._initialized:
            await self.initialize()
            
        # 尝试获取资源
        async with self._lock:
            # 首先尝试获取可用资源
            while len(self._resources) == 0:
                # 如果未达到最大大小，则创建新资源
                if len(self._in_use) < self.max_size:
                    try:
                        resource = await self._create_resource()
                        self._resource_metadata[id(resource)]['use_count'] += 1
                        self._resource_metadata[id(resource)]['last_used'] = time.time()
                        self._in_use.add(resource)
                        return resource
                    except Exception as e:
                        logger.error(f"Error creating resource: {e}")
                        # 如果创建失败，继续等待
                
                # 等待其他资源释放
                try:
                    # 使用超时等待
                    await asyncio.wait_for(
                        self._not_empty.wait(), 
                        timeout=self.acquisition_timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError("Timeout waiting for available resource")
            
            # 获取一个可用资源
            resource = self._resources.popleft()
            
            # 检查资源健康状况
            if self.health_check and not await self.health_check(resource):
                logger.warning("Resource failed health check, creating new one")
                
                # 尝试关闭不健康的资源
                if self.close_func:
                    try:
                        await self.close_func(resource)
                    except Exception as e:
                        logger.error(f"Error closing unhealthy resource: {e}")
                
                # 创建新资源
                resource = await self._create_resource()
            
            # 更新资源元数据
            if id(resource) in self._resource_metadata:
                self._resource_metadata[id(resource)]['use_count'] += 1
                self._resource_metadata[id(resource)]['last_used'] = time.time()
            
            # 将资源标记为使用中
            self._in_use.add(resource)
            
            return resource
    
    async def release(self, resource: T) -> None:
        """
        释放资源回池
        
        Args:
            resource: 要释放的资源
        """
        if self._closed:
            # 如果池已关闭，直接关闭资源
            if self.close_func and resource:
                try:
                    await self.close_func(resource)
                except Exception as e:
                    logger.error(f"Error closing resource: {e}")
            return
            
        async with self._lock:
            if resource in self._in_use:
                self._in_use.remove(resource)
                
                # 更新使用时间
                if id(resource) in self._resource_metadata:
                    self._resource_metadata[id(resource)]['last_used'] = time.time()
                
                # 放回池中
                self._resources.append(resource)
                
                # 通知等待的获取者
                self._not_empty.notify()
                
    async def close(self) -> None:
        """关闭资源池并释放所有资源"""
        if self._closed:
            return
            
        self._closed = True
        
        # 取消清理任务
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            
        async with self._lock:
            # 关闭所有资源
            resources = list(self._resources) + list(self._in_use)
            self._resources.clear()
            self._in_use.clear()
            
            # 通知任何等待的获取者
            self._not_empty.notify_all()
            
        # 关闭所有资源
        if self.close_func:
            for resource in resources:
                try:
                    await self.close_func(resource)
                except Exception as e:
                    logger.error(f"Error closing resource: {e}")
        
        self._resource_metadata.clear()
        logger.info("Resource pool closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取资源池统计信息"""
        return {
            'available': len(self._resources),
            'in_use': len(self._in_use),
            'total': len(self._resources) + len(self._in_use),
            'min_size': self.min_size,
            'max_size': self.max_size,
            'initialized': self._initialized,
            'closed': self._closed
        }