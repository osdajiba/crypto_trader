#!/usr/bin/env python3
# src/exchange/implementations/binance.py

"""
Binance exchange implementation.
Provides interface to Binance exchange API.
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import random
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps

from src.common.config_manager import ConfigManager
from src.common.abstract_factory import register_factory_class
from src.common.helpers import TimeUtils
from src.exchange.base import Exchange, ExchangeError, ExchangeAPIError, ExchangeRateLimitError, ExchangeConnectionError


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
                    self = args[0]  # Extract self from args
                    self.logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    last_exception = e
                    
                    # Calculate exponential backoff delay with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% random jitter
                    adjusted_delay = delay * jitter
                    
                    self.logger.info(f"Waiting {adjusted_delay:.2f} seconds before retry...")
                    await asyncio.sleep(adjusted_delay)
                except Exception as e:
                    self = args[0]  # Extract self from args
                    self.logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # All attempts failed
            self = args[0]  # Extract self from args
            self.logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
        return async_wrapper
    return decorator


@register_factory_class('exchange_factory', 'binance', 
                        description="Binance exchange integration",
                        features=["spot", "futures", "websocket"],
                        category="exchange")
class BinanceExchange(Exchange):
    """
    Binance exchange implementation
    
    This class provides a full implementation of the Exchange interface
    for the Binance cryptocurrency exchange, supporting both REST API
    and WebSocket connections.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Binance exchange
        
        Args:
            config: Configuration manager
            params: Additional parameters
        """
        super().__init__(config, params)
        
        # Exchange-specific attributes
        self.exchange = None
        self.async_exchange = None
        self.request_semaphore = None
        self.ws_subscriptions = {}
        
        # Settings from config
        self.download_chunk_size = self.config.get('api', 'download_chunk_size', default=5000)
        self.max_retry_attempts = self.config.get('network', 'connection', 'retry_attempts', default=3)
        self.retry_delay = self.config.get('network', 'connection', 'retry_delay', default=2.0)
        
        # Set up cache directory
        self.cache_dir = self._get_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_dir(self) -> str:
        """Get cache directory from configuration"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        return os.path.abspath(os.path.expanduser(cache_dir))
    
    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration
        
        Returns:
            Dict: CCXT parameters
        """
        params = {
            'apiKey': self.config.get('api', 'binance', 'api_key', default=''),
            'secret': self.config.get('api', 'binance', 'secret', default=''),
            'timeout': self.config.get('api', 'timeout', default=60000),
            'enableRateLimit': self.config.get('api', 'enable_rate_limit', default=True),
            'options': {
                'adjustForTimeDifference': self.config.get(
                    'api', 'binance', 'options', 'adjust_time_diff', default=True),
                'recvWindow': self.config.get(
                    'api', 'binance', 'options', 'recv_window', default=60000),
                'defaultType': self.config.get(
                    'api', 'binance', 'options', 'default_type', default='spot'),
                'keepAlive': True,
            },
            'headers': {
                'Connection': 'keep-alive',
                'Keep-Alive': '60',
            }
        }
        
        # Check for proxy configuration
        use_proxy = self.config.get('api', 'useproxy', default=None)
        
        if use_proxy is True or use_proxy == "true":
            # Get proxy from configuration
            http_proxy = self.config.get('proxies', 'http', default=None)
            https_proxy = self.config.get('proxies', 'https', default=None)
            
            proxies = {}
            if http_proxy or https_proxy:
                proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
                self.logger.info(f"Using configured proxies: {proxies}")
                if proxies:
                    params['proxies'] = proxies
        
        return params
    
    async def _initialize_exchange(self) -> None:
        """
        Initialize Binance exchange connection
        
        Raises:
            ExchangeConnectionError: If initialization fails
        """
        try:
            self.logger.info("Initializing Binance exchange")
            
            # Build params from config
            params = self._build_params()
            
            # Initialize synchronous exchange
            self.exchange = ccxt.binance(params)
            self.logger.info("Synchronous exchange initialized successfully")
            
            # Set up async request semaphore
            self.request_semaphore = asyncio.Semaphore(5)  # Limit max concurrent requests
            
            # Preload market data
            try:
                self.exchange.load_markets()
                self.logger.info("Market data loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load market data: {str(e)}, will load on first call")
            
            self._connected = True
            
        except Exception as e:
            self.logger.error(f"Error initializing Binance exchange: {str(e)}")
            raise ExchangeConnectionError(f"Cannot initialize exchange: {str(e)}")
    
    async def _initialize_offline(self) -> None:
        """Initialize exchange in offline mode"""
        self.logger.info("Initializing Binance exchange in offline mode")
        self.exchange = None
        self._connected = False
    
    async def _initialize_fallback(self) -> None:
        """Initialize fallback mode for backtesting"""
        self.logger.warning("Initializing minimal Binance exchange for backtest support")
        self.exchange = ccxt.binance()
        self.exchange.markets = {}
        self.exchange.markets_by_id = {}
        self._connected = False
    
    async def _init_async_exchange(self) -> None:
        """
        Initialize async exchange (lazy initialization) optimized for high-latency networks
        
        Raises:
            ExchangeConnectionError: If initialization fails
        """
        if self.async_exchange is not None:
            return

        try:
            # Create optimized aiohttp session
            import aiohttp
            from aiohttp import ClientTimeout
            
            # Optimize HTTP connector for high-latency networks
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,
                limit_per_host=10,
                limit=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
            
            # Set longer timeouts for high-latency
            timeout = ClientTimeout(
                total=60,
                connect=20,
                sock_read=45,
                sock_connect=20
            )
            
            # Create session
            session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive', 'Keep-Alive': '60'}
            )
            
            # Build params from config
            params = self._build_params()
            
            # Initialize async exchange
            self.async_exchange = ccxt_async.binance(params)
            
            # Apply optimized session
            self.async_exchange.session = session
            
            # Initialize concurrency limiting semaphore if not already set
            if not self.request_semaphore:
                self.request_semaphore = asyncio.Semaphore(5)
            
            self.logger.info("Async Binance exchange initialized with high-latency network optimizations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async Binance exchange: {str(e)}")
            self.async_exchange = None
            raise ExchangeConnectionError(f"Failed to initialize async exchange: {str(e)}")
    
    async def _shutdown_exchange(self) -> None:
        """Clean up exchange resources"""
        # Close all WebSocket subscriptions
        for sub_key, sub_info in list(self.ws_subscriptions.items()):
            try:
                sub_info['active'] = False
                exchange = sub_info.get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket subscription {sub_key}: {str(e)}")
        
        # Close async exchange
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                # Close underlying session
                if hasattr(self.async_exchange, 'session') and hasattr(self.async_exchange.session, 'close'):
                    await self.async_exchange.session.close()
                self.async_exchange = None
                self.logger.info("Async exchange connection closed")
        except Exception as e:
            self.logger.error(f"Error closing async exchange: {str(e)}")
    
    async def _exponential_backoff(self, attempt: int) -> float:
        """
        Implement exponential backoff with jitter
        
        Args:
            attempt: Current retry attempt number
            
        Returns:
            float: Delay time in seconds
        """
        base_delay = 1.0
        max_delay = 30.0
        
        # Calculate exponential backoff time with random jitter
        delay = min(max_delay, base_delay * (2 ** attempt)) 
        jitter = random.uniform(0.5, 1.0)  # 50-100% random jitter
        final_delay = delay * jitter
        self.logger.info(f"Will wait {final_delay:.2f} seconds before retry (attempt {attempt+1})")
        await asyncio.sleep(final_delay)
        return final_delay
        
    async def watch_ohlcv(self, 
                        symbol: str, 
                        timeframe: str, 
                        callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to real-time OHLCV data using WebSocket
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period (e.g., '1m', '1h', '1d')
            callback: Callback function to receive updates
            
        Returns:
            bool: Success status
            
        Raises:
            ExchangeAPIError: If subscription fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Check if ccxt.pro is installed
            try:
                import ccxtpro
            except ImportError:
                self.logger.error("ccxt.pro not installed. Please install ccxt.pro to use WebSocket functionality.")
                return False
            
            # Subscription key
            sub_key = f"{symbol}_{timeframe}"
            
            # If already subscribed, avoid duplicate subscription
            if sub_key in self.ws_subscriptions and self.ws_subscriptions[sub_key]['active']:
                self.logger.info(f"Already subscribed to WebSocket data stream for {symbol} {timeframe}")
                return True
            
            self.logger.info(f"Starting WebSocket connection to monitor {symbol} {timeframe} in real-time")
            
            # Get params from config
            params = self._build_params()
            
            # Initialize exchange
            exchange = ccxtpro.binance(params)
            
            # Configure WebSocket options
            exchange.options['ws'] = {
                'heartbeat': True,         # Enable heartbeat
                'ping_interval': 30000,    # 30 second ping interval
                'reconnect_rate': 5000,    # 5 second reconnect rate
                'max_reconnects': 100,     # Maximum reconnect attempts
            }
            
            # Mark as active
            self.ws_subscriptions[sub_key] = {
                'active': True,
                'exchange': exchange,
                'last_data': None,
                'errors': 0,
            }
            
            # Launch background task to manage WebSocket connection
            asyncio.create_task(self._ws_manager(sub_key, symbol, timeframe, callback))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up WebSocket connection: {str(e)}")
            raise ExchangeAPIError(f"Failed to set up WebSocket connection: {str(e)}")
    
    async def _ws_manager(self, sub_key: str, symbol: str, timeframe: str, callback):
        """
        Manage the continuous operation of a WebSocket connection
        
        Args:
            sub_key: Subscription key
            symbol: Trading pair symbol
            timeframe: Time period
            callback: Callback function to receive updates
        """
        if sub_key not in self.ws_subscriptions:
            self.logger.error(f"WebSocket subscription for {sub_key} not found")
            return
        
        sub_info = self.ws_subscriptions[sub_key]
        exchange = sub_info['exchange']
        
        retry_delay = 1.0  # Initial retry delay
        max_retry_delay = 30.0  # Maximum retry delay
        
        while sub_info['active']:
            try:
                self.logger.debug(f"Waiting for WebSocket update for {symbol} {timeframe}")
                ohlcv = await exchange.watchOHLCV(symbol, timeframe)
                
                # Data processing
                sub_info['last_data'] = ohlcv
                sub_info['errors'] = 0  # Reset error count
                
                # Reset retry delay
                retry_delay = 1.0
                
                # Call callback
                if callback and callable(callback):
                    try:
                        await callback(ohlcv)
                    except Exception as callback_error:
                        self.logger.error(f"Callback processing error: {str(callback_error)}")
                
            except Exception as e:
                sub_info['errors'] += 1
                self.logger.error(f"WebSocket error ({sub_info['errors']}): {str(e)}")
                
                # If too many errors, may need complete reconnection
                if sub_info['errors'] > 10:
                    self.logger.warning(f"Too many errors, reinitializing WebSocket connection")
                    try:
                        await exchange.close()
                        # Recreate exchange instance
                        import ccxtpro
                        exchange = ccxtpro.binance(self._build_params())
                        exchange.options['ws'] = {
                            'heartbeat': True,
                            'ping_interval': 30000,
                            'reconnect_rate': 5000,
                            'max_reconnects': 100,
                        }
                        sub_info['exchange'] = exchange
                        sub_info['errors'] = 0
                    except Exception as reset_error:
                        self.logger.error(f"Error resetting WebSocket connection: {str(reset_error)}")
                
                # Use exponential backoff
                retry_delay = min(retry_delay * 2, max_retry_delay)
                jitter = random.uniform(0.5, 1.0)
                actual_delay = retry_delay * jitter
                self.logger.info(f"Waiting {actual_delay:.2f} seconds before retrying WebSocket connection")
                await asyncio.sleep(actual_delay)
    
    async def stop_watching(self, symbol: str, timeframe: str) -> bool:
        """
        Stop WebSocket data subscription
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period
            
        Returns:
            bool: Success status
        """
        sub_key = f"{symbol}_{timeframe}"
        
        if sub_key in self.ws_subscriptions:
            try:
                # Mark as inactive
                self.ws_subscriptions[sub_key]['active'] = False
                
                # Close exchange connection
                exchange = self.ws_subscriptions[sub_key].get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
                    
                self.logger.info(f"Stopped WebSocket subscription for {symbol} {timeframe}")
                return True
            except Exception as e:
                self.logger.error(f"Error stopping WebSocket subscription: {str(e)}")
                return False
        
        self.logger.warning(f"No active WebSocket subscription found for {symbol} {timeframe}")
        return False
    
    def _get_date_chunks(self, start_dt: datetime, end_dt: datetime, 
                        timeframe: str, chunk_size_days: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """
        Break date range into manageable chunks
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            timeframe: Time period (e.g., '1m', '1h', '1d')
            chunk_size_days: Optional override for chunk size in days
            
        Returns:
            List[Tuple[datetime, datetime]]: List of (start, end) pairs for each chunk
        """
        # Calculate total time span and expected data points
        total_seconds = (end_dt - start_dt).total_seconds()
        seconds_per_candle = TimeUtils.timeframe_to_seconds(timeframe)
        expected_candles = total_seconds / seconds_per_candle
        
        # Determine optimal chunk size based on timeframe and total expected candles
        # Aim for chunks that will yield no more than ~70% of download_chunk_size
        target_candles_per_chunk = self.download_chunk_size * 0.7
        
        # Calculate ideal chunk size in seconds
        ideal_chunk_seconds = target_candles_per_chunk * seconds_per_candle
        safety_factor = 0.8
        ideal_chunk_seconds *= safety_factor  # Add a safety factor
        ideal_chunk_days = max(0.5, min(30, ideal_chunk_seconds / 86400))  # Convert to days with bounds
        
        if chunk_size_days is not None:
            chunk_size = timedelta(days=chunk_size_days)
        else:
            # Round to a half-day or full day for cleaner chunks
            if ideal_chunk_days < 1:
                chunk_size = timedelta(hours=12)
            else:
                chunk_size = timedelta(days=round(ideal_chunk_days))
        
        # For very small timeframes with expected high data volume, use smaller chunks
        if timeframe in ['1m', '5m'] and expected_candles > 5000:
            chunk_size = min(chunk_size, timedelta(hours=12))
        elif timeframe in ['15m', '30m'] and expected_candles > 3000:
            chunk_size = min(chunk_size, timedelta(days=1))
        
        self.logger.info(f"Using {chunk_size} chunk size for {timeframe} data from {start_dt} to {end_dt}")
        
        # Generate chunks
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end
            
        self.logger.info(f"Split request into {len(chunks)} chunks for {expected_candles:.0f} expected data points")
        
        return chunks
        
    @staticmethod
    def _process_ohlcv_data(ohlcv_data: List) -> pd.DataFrame:
        """
        Convert OHLCV data to DataFrame
        
        Args:
            ohlcv_data: Raw OHLCV data from exchange
            
        Returns:
            pd.DataFrame: Processed OHLCV data
        """
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            # No logger here - static method
            print(f"Error processing OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    @retry_exchange_operation(max_attempts=3, base_delay=2.0, max_delay=30.0)
    async def fetch_latest_ohlcv(self, 
                           symbol: str, 
                           timeframe: str = '1m', 
                           limit: int = 100) -> pd.DataFrame:
        """
        Fetch latest OHLCV data with retry mechanism
        
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
        if not self._initialized:
            await self.initialize()
        
        if not self.exchange:
            self.logger.error("Exchange not initialized, cannot fetch data")
            return pd.DataFrame()
        
        try:
            # Handle rate limit
            await self._handle_rate_limit()
            
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Fetch data
            ohlcv = await self.async_exchange.fetch_ohlcv(
                symbol=symbol, 
                timeframe=timeframe, 
                limit=limit
            )

            df = self._process_ohlcv_data(ohlcv)
            if not df.empty:
                self.logger.info(f"Successfully fetched {len(df)} latest records for {symbol} {timeframe}")
                return df
            
            self.logger.warning(f"Empty response when fetching {symbol} data")
            return pd.DataFrame()
                
        except ccxt.RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded: {str(e)}")
            raise ExchangeRateLimitError(f"Rate limit exceeded when fetching latest OHLCV: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise ExchangeAPIError(f"API error when fetching latest OHLCV: {str(e)}")
    
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
        if not self._initialized:
            await self.initialize()
        
        try:
            start_dt = TimeUtils.parse_timestamp(start_date, default_days_ago=30)
            end_dt = TimeUtils.parse_timestamp(end_date, default_days_ago=0)
            self.logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")
            
            # Initialize async exchange if needed
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Split request into multiple chunks to handle large date ranges
            chunks = self._get_date_chunks(start_dt, end_dt, timeframe)
            self.logger.info(f"Split request into {len(chunks)} chunks")
            
            all_data = []
            
            # Re-evaluate the chunking strategy for very large requests
            if len(chunks) > 20:  # Arbitrary threshold for large requests
                self.logger.warning(f"Large number of chunks ({len(chunks)}), implementing special handling")
                
                batch_size = 5  # Process 5 chunks at a time
                
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
                    
                    # Create tasks for this batch
                    batch_tasks = []
                    for j, (chunk_start, chunk_end) in enumerate(batch_chunks):
                        task = asyncio.create_task(
                            self._fetch_chunk(i+j, chunk_start, chunk_end, symbol, timeframe)
                        )
                        batch_tasks.append(task)
                    
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Chunk processing failed: {str(result)}")
                        elif isinstance(result, list) and result:
                            all_data.extend(result)
                    
                    # Pause between batches to avoid rate limit issues
                    if i + batch_size < len(chunks):
                        pause_time = 3  # 3 seconds between batches
                        self.logger.info(f"Pausing {pause_time} seconds between batch processing")
                        await asyncio.sleep(pause_time)
            else:
                tasks = []
                # Process chunks concurrently, but limited by semaphore
                for i, (chunk_start, chunk_end) in enumerate(chunks):
                    task = asyncio.create_task(
                        self._fetch_chunk(i, chunk_start, chunk_end, symbol, timeframe)
                    )
                    tasks.append(task)
        
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in chunk_results:
                    if isinstance(result, Exception):
                        self.logger.error(f"Chunk processing failed: {str(result)}")
                    elif isinstance(result, list) and result:
                        all_data.extend(result)
            
            if not all_data:
                self.logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
                
            df = self._process_ohlcv_data(all_data)
            
            if not df.empty:
                # Ensure datetime column is datetime type
                if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
                if 'datetime' in df.columns:
                    df = df.drop_duplicates(subset=['datetime'])
                df = df.sort_values('datetime')
            
            self.logger.info(f"Downloaded {len(df)} candle data points for {symbol}")
            
            return df
            
        except ccxt.RateLimitExceeded as e:
            self.logger.error(f"Rate limit exceeded: {str(e)}")
            raise ExchangeRateLimitError(f"Rate limit exceeded when fetching historical OHLCV: {str(e)}")
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            raise ExchangeAPIError(f"API error when fetching historical OHLCV: {str(e)}")
    
    async def _fetch_chunk(self, 
                       chunk_index: int, 
                       chunk_start: datetime, 
                       chunk_end: datetime, 
                       symbol: str, 
                       timeframe: str) -> List:
        """
        Fetch a single historical data chunk with retries and improved rate limiting
        
        Args:
            chunk_index: Index of the chunk (for logging)
            chunk_start: Start datetime
            chunk_end: End datetime
            symbol: Trading pair symbol
            timeframe: Time period
            
        Returns:
            List: OHLCV data for the chunk
            
        Raises:
            ExchangeAPIError: If API call fails
            ExchangeRateLimitError: If rate limit is exceeded
        """
        # Use semaphore to limit concurrency
        async with self.request_semaphore:
            # Calculate chunk durations and expected data points to determine priority
            chunk_duration = (chunk_end - chunk_start).total_seconds()
            seconds_per_candle = TimeUtils.timeframe_to_seconds(timeframe)
            expected_points = int(chunk_duration / seconds_per_candle)
            
            # If this is a large chunk, further subdivide it to avoid rate limit issues
            if expected_points > self.download_chunk_size / 2 and chunk_duration > 86400:  # More than half max size and >1 day
                self.logger.debug(f"Splitting large chunk {chunk_index+1} into smaller pieces")
                
                mid_point = chunk_start + timedelta(seconds=chunk_duration/2)
                
                # Process first half
                first_half = await self._fetch_chunk(
                    chunk_index * 2, 
                    chunk_start, 
                    mid_point, 
                    symbol, 
                    timeframe
                )
                
                await asyncio.sleep(0.5)  # Small delay between sub-chunks
                
                # Process second half
                second_half = await self._fetch_chunk(
                    chunk_index * 2 + 1, 
                    mid_point, 
                    chunk_end, 
                    symbol, 
                    timeframe
                )
                
                return first_half + second_half
            
            # Regular chunk processing with retries
            for retry in range(self.max_retry_attempts):
                try:
                    await self._handle_rate_limit()
                    
                    chunk_since = int(chunk_start.timestamp() * 1000)
                    chunk_until = int(chunk_end.timestamp() * 1000)
                    
                    if chunk_index == 0 or chunk_index % 5 == 0:
                        self.logger.info(f"Fetching chunk {chunk_index+1}: {chunk_start} to {chunk_end}")
                    
                    # Ensure async exchange is initialized
                    if not self.async_exchange:
                        await self._init_async_exchange()
                    
                    ohlcv = await self.async_exchange.fetch_ohlcv(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        since=chunk_since,
                        limit=self.download_chunk_size,
                        params={
                            "endTime": chunk_until,
                            "recvWindow": 60000  # Use maximum receive window for high-latency
                        }
                    )
                    
                    if not ohlcv or len(ohlcv) == 0:
                        self.logger.debug(f"Chunk {chunk_index+1} returned no data")
                        return []
                    
                    self.logger.debug(f"Chunk {chunk_index+1} successfully fetched {len(ohlcv)} records")
                    
                    # Adaptive delay based on how many records we got relative to limit
                    ratio = len(ohlcv) / self.download_chunk_size
                    delay = 0.2 + (0.5 * ratio) if ratio > 0.8 else 0.2
                    await asyncio.sleep(delay)
                    
                    return ohlcv
                    
                except ccxt.RateLimitExceeded as e:
                    # Special handling for explicit rate limit errors
                    self.logger.warning(f"Rate limit exceeded on chunk {chunk_index+1}, attempt {retry+1}")
                    wait_time = 60.0  # Full minute wait
                    self.logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    
                except Exception as e:
                    self.logger.warning(f"Chunk {chunk_index+1}, attempt {retry+1} failed: {str(e)}")
                    if retry < self.max_retry_attempts - 1:
                        await self._exponential_backoff(retry)
            
            self.logger.error(f"All retry attempts for chunk {chunk_index+1} failed")
            raise ExchangeAPIError(f"Failed to fetch chunk {chunk_index+1} after {self.max_retry_attempts} attempts")
            
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker data for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict[str, Any]: Ticker data with price information
            
        Raises:
            ExchangeAPIError: If API call fails
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Handle rate limit
            await self._handle_rate_limit()
            
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
                
            # Fetch ticker
            ticker = await self.async_exchange.fetch_ticker(symbol)
            
            return ticker
        except Exception as e:
            self.logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise ExchangeAPIError(f"Failed to fetch ticker for {symbol}: {str(e)}")