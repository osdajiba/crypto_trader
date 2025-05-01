#!/usr/bin/env python3
# src/exchange/base.py

"""
Base exchange implementation.
Provides abstract base class for all exchange interfaces.
"""

import pandas as pd
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Union, List, Callable
from datetime import datetime

from src.common.config import ConfigManager
from src.common.log_manager import LogManager


class ExchangeError(Exception):
    """Base exception for exchange errors"""
    pass


class ExchangeConnectionError(ExchangeError):
    """Error establishing connection to exchange"""
    pass


class ExchangeAPIError(ExchangeError):
    """Error interacting with exchange API"""
    pass


class ExchangeRateLimitError(ExchangeError):
    """Rate limit exceeded"""
    pass


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
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_refill = time.time()
            return wait_time
        else:
            self.tokens -= tokens
            return 0


class Exchange(ABC):
    """
    Abstract base exchange interface.
    
    All exchange implementations should inherit from this class
    to provide a consistent interface across the application.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize exchange
        
        Args:
            config: Configuration manager
            params: Additional parameters for initialization
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"exchange.{self.__class__.__name__.lower()}")
        
        # Track initialization and connection state
        self._initialized = False
        self._connected = False
        
        # Rate limiting
        self._setup_rate_limiting()
    
    def _setup_rate_limiting(self) -> None:
        """Set up rate limiting using token bucket"""
        rate_limit = self.config.get('api', 'rate_limits', 'requests', default=20)
        # Slightly lower rate (95%) to provide safety margin
        tokens_per_second = (rate_limit * 0.95) / 60.0
        self.token_bucket = TokenBucket(rate=tokens_per_second, capacity=rate_limit)
    
    async def initialize(self) -> None:
        """
        Initialize exchange (template method)
        
        Performs common initialization and calls exchange-specific init
        """
        if self._initialized:
            return
            
        try:
            self.logger.info(f"Initializing {self.__class__.__name__} exchange")
            
            # Check if offline mode is enabled
            offline_mode = self.config.get('system', 'offline_mode', default=False)
            if offline_mode:
                self.logger.info("Running in offline mode, using minimal functionality")
                await self._initialize_offline()
            else:
                await self._initialize_exchange()
                
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} exchange initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Exchange initialization failed: {str(e)}")
            
            # Attempt fallback initialization for backtest support if configured
            if self.config.get('backtest', 'fallback_to_local', default=True):
                self.logger.warning("Attempting fallback initialization for backtest support")
                try:
                    await self._initialize_fallback()
                    self._initialized = True
                    self.logger.info("Fallback initialization successful")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback initialization failed: {str(fallback_error)}")
                    raise ExchangeConnectionError(f"Failed to initialize exchange in fallback mode: {str(fallback_error)}")
            else:
                raise ExchangeConnectionError(f"Failed to initialize exchange: {str(e)}")
    
    @abstractmethod
    async def _initialize_exchange(self) -> None:
        """
        Exchange-specific initialization (to be implemented by subclasses)
        """
        pass
    
    async def _initialize_offline(self) -> None:
        """
        Initialize exchange in offline mode
        
        Default implementation that can be overridden by subclasses
        """
        self.logger.info("Using default offline mode initialization")
        # Basic offline initialization
        pass
    
    async def _initialize_fallback(self) -> None:
        """
        Fallback initialization for backtest support
        
        Default implementation that can be overridden by subclasses
        """
        self.logger.info("Using default fallback initialization")
        # Basic fallback initialization
        pass
    
    async def shutdown(self) -> None:
        """
        Shutdown exchange connection and cleanup resources
        """
        if not self._initialized:
            return
            
        try:
            self.logger.info(f"Shutting down {self.__class__.__name__} exchange")
            await self._shutdown_exchange()
            self._initialized = False
            self._connected = False
            self.logger.info(f"{self.__class__.__name__} exchange shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during exchange shutdown: {str(e)}")
    
    async def _shutdown_exchange(self) -> None:
        """
        Exchange-specific shutdown (to be implemented by subclasses)
        
        Default implementation for subclasses to override
        """
        pass
    
    @abstractmethod
    async def fetch_historical_ohlcv(self, 
                                symbol: str, 
                                timeframe: str = '1m',
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period (e.g., '1m', '1h', '1d')
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            
        Returns:
            pd.DataFrame: OHLCV data
            
        Raises:
            ExchangeAPIError: If API call fails
            ExchangeRateLimitError: If rate limit is exceeded
        """
        pass
    
    @abstractmethod
    async def fetch_latest_ohlcv(self, 
                            symbol: str, 
                            timeframe: str = '1m', 
                            limit: int = 100) -> pd.DataFrame:
        """
        Fetch latest OHLCV data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period (e.g., '1m', '1h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            pd.DataFrame: OHLCV data
            
        Raises:
            ExchangeAPIError: If API call fails
            ExchangeRateLimitError: If rate limit is exceeded
        """
        pass
    
    @abstractmethod
    async def watch_ohlcv(self, 
                    symbol: str, 
                    timeframe: str, 
                    callback: Optional[Callable] = None) -> bool:
        """
        Watch OHLCV data in real-time
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period (e.g., '1m', '1h', '1d')
            callback: Callback function to receive updates
            
        Returns:
            bool: Success status
            
        Raises:
            ExchangeAPIError: If subscription fails
        """
        pass
    
    @abstractmethod
    async def stop_watching(self, 
                        symbol: str, 
                        timeframe: str) -> bool:
        """
        Stop watching OHLCV data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period (e.g., '1m', '1h', '1d')
            
        Returns:
            bool: Success status
        """
        pass
    
    async def _handle_rate_limit(self) -> None:
        """
        Handle rate limiting for API requests
        
        Uses token bucket for smooth request distribution
        
        Raises:
            ExchangeRateLimitError: If waiting for rate limit would exceed reasonable time
        """
        wait_time = await self.token_bucket.consume(1)
        if wait_time > 0:
            if wait_time > 30:  # If wait time is excessive, raise error
                raise ExchangeRateLimitError(f"Rate limit exceeded, would need to wait {wait_time:.2f} seconds")
            self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict[str, Any]: Ticker data with price information
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("fetch_ticker not implemented by this exchange")
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch order book for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Order book depth
            
        Returns:
            Dict[str, Any]: Order book data
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("fetch_order_book not implemented by this exchange")
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch recent trades for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            
        Returns:
            List[Dict[str, Any]]: Recent trades
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("fetch_trades not implemented by this exchange")
    
    def is_initialized(self) -> bool:
        """
        Check if exchange is initialized
        
        Returns:
            bool: Initialization status
        """
        return self._initialized
    
    def is_connected(self) -> bool:
        """
        Check if exchange is connected
        
        Returns:
            bool: Connection status
        """
        return self._connected