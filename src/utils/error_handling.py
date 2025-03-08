# src/utils/error_handling.py

import time
import asyncio
from typing import Callable, Type, Tuple, Optional, Any

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager

logger = LogManager.get_logger(name="trading_system")

def retry(
    max_retries: Optional[int] = None,
    delay: Optional[float] = None,
    exceptions: Tuple[Type[Exception]] = (Exception,),
    backoff_factor: float = 2.0,
    on_error: Optional[Callable[[Exception, int], None]] = None,
    on_success: Optional[Callable[[Any, int], None]] = None,
    should_retry: Optional[Callable[[Exception, int], bool]] = None,
    log_errors: bool = True,
    config_path: Optional[str] = None
) -> Callable:
    """
    带指数退避的重试装饰器，支持同步和异步函数。
    
    Args:
        max_retries (Optional[int]): 最大重试次数，默认从配置读取或 3
        delay (Optional[float]): 初始重试延迟（秒），默认从配置读取或 1.0
        exceptions (Tuple[Type[Exception]]): 需要捕获的异常类型，默认 (Exception,)
        backoff_factor (float): 指数退避因子，每次重试延迟乘以此值，默认 2.0
        on_error (Optional[Callable[[Exception, int], None]]): 发生错误时的回调，接收异常和重试次数
        on_success (Optional[Callable[[Any, int], None]]): 重试成功时的回调，接收结果和尝试次数
        should_retry (Optional[Callable[[Exception, int], bool]]): 自定义重试条件，返回是否继续重试
        log_errors (bool): 是否记录错误日志，默认 True
        config_path (Optional[str]): 配置文件路径，用于读取默认参数
    
    Returns:
        Callable: 装饰后的函数
    
    Raises:
        RuntimeError: 重试次数耗尽后仍失败
    """
    # 从配置加载默认参数（如果未显式指定）
    if config_path:
        config = ConfigManager.get_instance(config_path)
        max_retries = max_retries if max_retries is not None else config.get("default_config", "misc_config", "max_retries", default=3)
        delay = delay if delay is not None else config.get("default_config", "misc_config", "retry_delay", default=1.0)
    else:
        max_retries = max_retries if max_retries is not None else 3
        delay = delay if delay is not None else 1.0

    def decorator(func: Callable) -> Callable:
        # 判断函数是否为异步
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            async def async_wrapper(*args, **kwargs) -> Any:
                retries = 0
                current_delay = delay
                while retries < max_retries:
                    try:
                        result = await func(*args, **kwargs)
                        if on_success:
                            on_success(result, retries + 1)
                        return result
                    except exceptions as e:
                        retries += 1
                        msg = f"Attempt {retries}/{max_retries} failed for '{func.__name__}': {str(e)}"
                        if log_errors:
                            logger.warning(msg)
                        if on_error:
                            on_error(e, retries)
                        if retries == max_retries or (should_retry and not should_retry(e, retries)):
                            raise RuntimeError(f"Async operation '{func.__name__}' failed after {max_retries} retries") from e
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff_factor
                return None  # 理论上不会到达，但为类型安全添加
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs) -> Any:
                retries = 0
                current_delay = delay
                while retries < max_retries:
                    try:
                        result = func(*args, **kwargs)
                        if on_success:
                            on_success(result, retries + 1)
                        return result
                    except exceptions as e:
                        retries += 1
                        msg = f"Attempt {retries}/{max_retries} failed for '{func.__name__}': {str(e)}"
                        if log_errors:
                            logger.warning(msg)
                        if on_error:
                            on_error(e, retries)
                        if retries == max_retries or (should_retry and not should_retry(e, retries)):
                            raise RuntimeError(f"Operation '{func.__name__}' failed after {max_retries} retries") from e
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                return None  # 理论上不会到达，但为类型安全添加
            return sync_wrapper
    return decorator
