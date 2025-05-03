#!/usr/bin/env python3
# src/common/error_handling.py

import asyncio
import functools
import time
import random
import traceback
from typing import Callable, Type, Union, List, Optional, Dict, Any, TypeVar
import logging
from enum import Enum

from src.common.log_manager import LogManager

logger = LogManager.get_logger("common.error_handling")

# 类型变量定义
T = TypeVar('T')
F = TypeVar('F', bound=Callable)

# 错误类型分类
class ErrorSeverity(Enum):
    """错误严重性分类"""
    CRITICAL = 3    # 关键错误，需要立即处理
    ERROR = 2       # 普通错误，可能需要系统干预
    WARNING = 1     # 警告，可以继续运行
    INFO = 0        # 信息性错误，记录但不干预
    
# 自定义异常基类
class TradingSystemError(Exception):
    """交易系统错误基类"""
    severity = ErrorSeverity.ERROR
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.timestamp = time.time()
        self.traceback = traceback.format_exc()
    
    def to_dict(self) -> Dict[str, Any]:
        """将错误转换为字典格式"""
        return {
            'type': self.__class__.__name__,
            'message': self.message,
            'details': self.details,
            'severity': self.severity.name,
            'timestamp': self.timestamp,
            'traceback': self.traceback
        }
    
# 常见异常类型
class NetworkError(TradingSystemError):
    """网络相关错误"""
    severity = ErrorSeverity.WARNING
    
class ExchangeError(TradingSystemError):
    """交易所API错误"""
    severity = ErrorSeverity.ERROR
    
class RateLimitError(TradingSystemError):
    """速率限制错误"""
    severity = ErrorSeverity.WARNING
    
class ConfigError(TradingSystemError):
    """配置错误"""
    severity = ErrorSeverity.ERROR
    
class DataError(TradingSystemError):
    """数据错误"""
    severity = ErrorSeverity.WARNING
    
class ValidationError(TradingSystemError):
    """数据验证错误"""
    severity = ErrorSeverity.WARNING
    
class SystemError(TradingSystemError):
    """系统级错误"""
    severity = ErrorSeverity.CRITICAL
    
# 重试机制
async def retry_async(
    func: Callable,
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    on_retry: Optional[Callable] = None,
) -> Any:
    """
    异步重试函数，支持指数回退和抖动
    
    Args:
        func: 异步函数
        max_retries: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        retry_exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        Any: 函数返回值
        
    Raises:
        Exception: 超过最大重试次数后的异常
    """
    retry_count = 0
    last_exception = None
    
    while True:
        try:
            return await func()
        except retry_exceptions as e:
            retry_count += 1
            last_exception = e
            
            if retry_count > max_retries:
                logger.warning(f"Maximum retries ({max_retries}) reached")
                raise
            
            # 计算延迟时间
            delay = min(base_delay * (backoff_factor ** (retry_count - 1)), max_delay)
            
            # 添加随机抖动 (±25%)
            if jitter:
                delay = delay * (random.uniform(0.75, 1.25))
                
            logger.debug(f"Retry {retry_count}/{max_retries} after {delay:.2f}s: {str(e)}")
            
            # 调用重试回调
            if on_retry:
                try:
                    if asyncio.iscoroutinefunction(on_retry):
                        await on_retry(retry_count, delay, e)
                    else:
                        on_retry(retry_count, delay, e)
                except Exception as callback_err:
                    logger.error(f"Error in retry callback: {callback_err}")
            
            # 等待后重试
            await asyncio.sleep(delay)

def retry_decorator(
    max_retries: int = 3, 
    base_delay: float = 1.0, 
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retry_exceptions: Union[Type[Exception], List[Type[Exception]]] = Exception,
    on_retry: Optional[Callable] = None,
) -> Callable[[F], F]:
    """
    异步重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间(秒)
        max_delay: 最大延迟时间(秒)
        backoff_factor: 退避因子
        jitter: 是否添加随机抖动
        retry_exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def _execute():
                return await func(*args, **kwargs)
            
            return await retry_async(
                _execute,
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retry_exceptions=retry_exceptions,
                on_retry=on_retry
            )
        
        return wrapper  # type: ignore
    
    return decorator

class ErrorHandler:
    """错误处理类，用于集中处理系统错误"""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取错误处理器单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self.error_count = 0
        self.error_history = []
        self.max_history = 100
        self.error_handlers = {}
        
    def register_handler(self, error_type: Type[Exception], handler: Callable):
        """
        注册错误处理函数
        
        Args:
            error_type: 错误类型
            handler: 处理函数
        """
        if error_type not in self.error_handlers:
            self.error_handlers[error_type] = []
        self.error_handlers[error_type].append(handler)
        
    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> bool:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            
        Returns:
            bool: 是否成功处理
        """
        handled = False
        self.error_count += 1
        
        # 记录错误
        error_info = {
            'timestamp': time.time(),
            'type': type(error).__name__,
            'message': str(error),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        # 添加到历史记录
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        # 获取错误严重性
        severity = getattr(error, 'severity', ErrorSeverity.ERROR)
        
        # 根据严重性记录日志
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error}", exc_info=True)
        elif severity == ErrorSeverity.ERROR:
            logger.error(f"ERROR: {error}", exc_info=True)
        elif severity == ErrorSeverity.WARNING:
            logger.warning(f"WARNING: {error}")
        else:
            logger.info(f"INFO: {error}")
        
        # 查找并执行错误处理器
        for error_type, handlers in self.error_handlers.items():
            if isinstance(error, error_type):
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(error, context)
                        else:
                            result = handler(error, context)
                        
                        # 如果处理器返回True，表示错误已处理
                        if result:
                            handled = True
                    except Exception as e:
                        logger.error(f"Error in error handler: {e}")
        
        return handled
    
    def get_recent_errors(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的错误
        
        Args:
            count: 返回的错误数量
            
        Returns:
            List[Dict[str, Any]]: 错误列表
        """
        return self.error_history[-count:]
    
    def clear_history(self) -> None:
        """清空错误历史"""
        self.error_history.clear()