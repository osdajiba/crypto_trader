#!/usr/bin/env python3
# src/exchange/base.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import random
from functools import wraps

from src.common.log_manager import LogManager
from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.common.config import ConfigManager


logger = LogManager.get_logger("trading_system")


def retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0):
    """Retry decorator with exponential backoff and jitter"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    last_exception = e
                    
                    # Calculate exponential backoff delay with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% random jitter
                    adjusted_delay = delay * jitter
                    
                    logger.info(f"Waiting {adjusted_delay:.2f} seconds before retry...")
                    await asyncio.sleep(adjusted_delay)
                except Exception as e:
                    logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # All attempts failed
            logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    last_exception = e
                    
                    # Calculate exponential backoff delay with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% random jitter
                    adjusted_delay = delay * jitter
                    
                    logger.info(f"Waiting {adjusted_delay:.2f} seconds before retry...")
                    time.sleep(adjusted_delay)
                except Exception as e:
                    logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # All attempts failed
            logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
        
        # Choose appropriate wrapper based on whether the function is a coroutine
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class TokenBucket:
    """Token bucket for rate limiting with smooth request distribution"""
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket
        
        Args:
            rate: Token refill rate (tokens per second)
            capacity: Maximum bucket capacity
        """
        self.rate = rate  # tokens per second
        self.capacity = capacity  # maximum tokens
        self.tokens = capacity  # current tokens
        self.last_refill = time.time()  # last refill timestamp
    
    async def consume(self, tokens: int = 1) -> float:
        """
        Consume tokens from the bucket, waiting if necessary
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            float: Wait time in seconds (0 if no wait)
        """
        # Refill tokens based on elapsed time
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now
        
        # If not enough tokens, calculate wait time
        if self.tokens < tokens:
            # Time needed to refill required tokens
            wait_time = (tokens - self.tokens) / self.rate
            await asyncio.sleep(wait_time)    # After waiting, we'll have exactly the tokens we need
            self.tokens = 0
            self.last_refill = time.time()
            return wait_time
        else:
            self.tokens -= tokens    # Consume tokens and continue immediately
            return 0
        
        
class Exchange:
    """Basic exchange interface"""
    
    def __init__(self, config: ConfigManager):
        self.config = config