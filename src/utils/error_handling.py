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
    Retry decorator with exponential backoff, supports synchronous and asynchronous functions. 
    
    Args: 
        max_retries (Optional[int]): Maximum number of retries, default reading from configuration or 3 
        delay (Optional[float]): Initial retry delay (seconds), read from configuration by default or 1.0 
        exceptions (Tuple[Type[Exception]]): Type of Exception to catch; default (Exception,) 
        backoff_factor (float): Exponential backoff factor, multiplied by the delay per retry, default 2.0
        on_error (Optional[Callable[[Exception, int], None]]): Callback in case of an error, receiving the exception and the number of retries 
        on_success (Optional[Callable[[Any, int], None]]): Callback ona successful retry, receiving the result and the number of attempts 
        should_retry (Optional[Callable[[Exception, int], bool]]): Define a custom retry condition that returns whether to continue retrying 
        log_errors (bool): Default True 
        config_path (Optional[str]): Configuration file path for reading default arguments 
        
    Returns: 
        Callable: decorated function 
        Raises: RuntimeError: fails even after exhausting the number of retries
    """
    
    # Load default parameters from the configuration (if not explicitly specified)
    if config_path:
        config = ConfigManager.get_instance(config_path)
        max_retries = max_retries if max_retries is not None else config.get("default_config", "misc_config", "max_retries", default=3)
        delay = delay if delay is not None else config.get("default_config", "misc_config", "retry_delay", default=1.0)
    else:
        max_retries = max_retries if max_retries is not None else 3
        delay = delay if delay is not None else 1.0

    def decorator(func: Callable) -> Callable:
        # Determines whether the function is asynchronous
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
                return None  # Theoretically not reached, but added for type safety
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
                return None  # Theoretically not reached, but added for type safety
            return sync_wrapper
    return decorator
