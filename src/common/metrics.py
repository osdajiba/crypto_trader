#!/usr/bin/env python3
# src/common/metrics.py

import time
import asyncio
import functools
from typing import Dict, List, Any, Optional, Callable, TypeVar, Union
import threading
import logging
import statistics
from collections import deque

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.metrics")

# 类型变量定义
F = TypeVar('F', bound=Callable)

class Metrics:
    """
    系统性能指标收集器
    
    特点:
    - 低开销的指标收集
    - 支持函数执行时间测量
    - 支持自定义指标
    - 支持指标导出
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取指标收集器单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        # 计数器指标
        self._counters = {}
        # 仪表盘指标(最后一个值)
        self._gauges = {}
        # 直方图指标(值分布)
        self._histograms = {}
        # 计时器
        self._timers = {}
        
        # 历史数据
        self._histories = {}
        self._history_max_size = 1000
        
        # 同步锁
        self._lock = threading.RLock()
        
        # 指标导出任务
        self._export_task = None
        self._export_interval = 60  # 秒
        self._closed = False
        
    def increment(self, name: str, value: float = 1, tags: Dict[str, str] = None) -> None:
        """
        增加计数器值
        
        Args:
            name: 指标名称
            value: 增加值
            tags: 标签
        """
        with self._lock:
            key = self._get_key(name, tags)
            
            if key not in self._counters:
                self._counters[key] = 0
                
            self._counters[key] += value
            
            # 添加到历史
            self._add_to_history(key, self._counters[key], 'counter')
    
    def decrement(self, name: str, value: float = 1, tags: Dict[str, str] = None) -> None:
        """
        减少计数器值
        
        Args:
            name: 指标名称
            value: 减少值
            tags: 标签
        """
        self.increment(name, -value, tags)
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """
        设置仪表盘值
        
        Args:
            name: 指标名称
            value: 当前值
            tags: 标签
        """
        with self._lock:
            key = self._get_key(name, tags)
            self._gauges[key] = value
            
            # 添加到历史
            self._add_to_history(key, value, 'gauge')
    
    def histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """
        记录直方图值
        
        Args:
            name: 指标名称
            value: 值
            tags: 标签
        """
        with self._lock:
            key = self._get_key(name, tags)
            
            if key not in self._histograms:
                self._histograms[key] = deque(maxlen=self._history_max_size)
                
            self._histograms[key].append(value)
            
            # 添加到历史
            self._add_to_history(key, value, 'histogram')
    
    def timer(self, name: str, duration: float, tags: Dict[str, str] = None) -> None:
        """
        记录计时器值
        
        Args:
            name: 指标名称
            duration: 持续时间(秒)
            tags: 标签
        """
        with self._lock:
            key = self._get_key(name, tags)
            
            if key not in self._timers:
                self._timers[key] = deque(maxlen=self._history_max_size)
                
            self._timers[key].append(duration)
            
            # 添加到历史
            self._add_to_history(key, duration, 'timer')
    
    def _get_key(self, name: str, tags: Dict[str, str] = None) -> str:
        """生成指标键"""
        if not tags:
            return name
            
        tag_str = ','.join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}[{tag_str}]"
    
    def _add_to_history(self, key: str, value: float, metric_type: str) -> None:
        """添加值到历史记录"""
        if key not in self._histories:
            self._histories[key] = deque(maxlen=self._history_max_size)
            
        self._histories[key].append({
            'timestamp': time.time(),
            'value': value,
            'type': metric_type
        })
    
    def get_counter(self, name: str, tags: Dict[str, str] = None) -> float:
        """
        获取计数器值
        
        Args:
            name: 指标名称
            tags: 标签
            
        Returns:
            float: 计数器值
        """
        with self._lock:
            key = self._get_key(name, tags)
            return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, tags: Dict[str, str] = None) -> Optional[float]:
        """
        获取仪表盘值
        
        Args:
            name: 指标名称
            tags: 标签
            
        Returns:
            Optional[float]: 仪表盘值
        """
        with self._lock:
            key = self._get_key(name, tags)
            return self._gauges.get(key)
    
    def get_histogram_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """
        获取直方图统计信息
        
        Args:
            name: 指标名称
            tags: 标签
            
        Returns:
            Dict[str, float]: 统计信息
        """
        with self._lock:
            key = self._get_key(name, tags)
            values = list(self._histograms.get(key, []))
            
            if not values:
                return {}
                
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stddev': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': sorted(values)[int(len(values) * 0.95)] if values else 0,
                'p99': sorted(values)[int(len(values) * 0.99)] if values else 0
            }
    
    def get_timer_stats(self, name: str, tags: Dict[str, str] = None) -> Dict[str, float]:
        """
        获取计时器统计信息
        
        Args:
            name: 指标名称
            tags: 标签
            
        Returns:
            Dict[str, float]: 统计信息
        """
        with self._lock:
            key = self._get_key(name, tags)
            values = list(self._timers.get(key, []))
            
            if not values:
                return {}
                
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stddev': statistics.stdev(values) if len(values) > 1 else 0,
                'p95': sorted(values)[int(len(values) * 0.95)] if values else 0,
                'p99': sorted(values)[int(len(values) * 0.99)] if values else 0
            }
    
    def get_history(self, name: str, tags: Dict[str, str] = None, 
                   count: int = None) -> List[Dict[str, Any]]:
        """
        获取指标历史数据
        
        Args:
            name: 指标名称
            tags: 标签
            count: 返回数量
            
        Returns:
            List[Dict[str, Any]]: 历史数据
        """
        with self._lock:
            key = self._get_key(name, tags)
            history = list(self._histories.get(key, []))
            
            if count is not None:
                history = history[-count:]
                
            return history
    
    def clear(self, name: str = None, tags: Dict[str, str] = None) -> None:
        """
        清除指标数据
        
        Args:
            name: 指标名称
            tags: 标签
        """
        with self._lock:
            if name is None:
                # 清除所有
                self._counters.clear()
                self._gauges.clear()
                self._histograms.clear()
                self._timers.clear()
                self._histories.clear()
            else:
                # 清除特定指标
                key = self._get_key(name, tags)
                self._counters.pop(key, None)
                self._gauges.pop(key, None)
                self._histograms.pop(key, None)
                self._timers.pop(key, None)
                self._histories.pop(key, None)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标
        
        Returns:
            Dict[str, Any]: 所有指标数据
        """
        with self._lock:
            result = {
                'counters': self._counters.copy(),
                'gauges': self._gauges.copy(),
                'histograms': {},
                'timers': {}
            }
            
            # 计算直方图统计
            for key, values in self._histograms.items():
                if values:
                    result['histograms'][key] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values)
                    }
            
            # 计算计时器统计
            for key, values in self._timers.items():
                if values:
                    result['timers'][key] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values)
                    }
                    
            return result
    
    async def start_export(self, export_func: Callable, interval: float = 60) -> None:
        """
        启动定期导出指标
        
        Args:
            export_func: 导出函数
            interval: 导出间隔(秒)
        """
        self._export_interval = interval
        self._closed = False
        self._export_task = asyncio.create_task(self._export_loop(export_func))
        logger.info(f"Metrics export started with interval {interval}s")
    
    async def _export_loop(self, export_func: Callable) -> None:
        """定期导出指标循环"""
        try:
            while not self._closed:
                await asyncio.sleep(self._export_interval)
                
                try:
                    metrics = self.get_all_metrics()
                    
                    if asyncio.iscoroutinefunction(export_func):
                        await export_func(metrics)
                    else:
                        export_func(metrics)
                        
                except Exception as e:
                    logger.error(f"Error exporting metrics: {e}")
                    
        except asyncio.CancelledError:
            logger.debug("Metrics export task cancelled")
        except Exception as e:
            logger.error(f"Error in metrics export loop: {e}")
    
    async def close(self) -> None:
        """关闭指标收集器"""
        if self._closed:
            return
            
        self._closed = True
        
        # 取消导出任务
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Metrics collector closed")

# 装饰器：测量函数执行时间
def timed(name: Optional[str] = None, tags: Dict[str, str] = None):
    """
    测量函数执行时间的装饰器
    
    Args:
        name: 指标名称
        tags: 标签
    """
    def decorator(func: F) -> F:
        # 获取默认指标名
        metric_name = name or f"function.execution_time.{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                Metrics.get_instance().timer(metric_name, duration, tags)
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                Metrics.get_instance().timer(metric_name, duration, tags)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore
    
    return decorator