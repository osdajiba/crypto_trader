# src/exchange/binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import random
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import TimeUtils


# Get or create logger
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
        
        
class Binance:
    """Binance exchange interface optimized for high-latency networks"""
    
    def __init__(self, config: ConfigManager):
        """Initialize Binance interface using configuration"""
        self.config = config
        
        # Set up cache directory
        self.cache_dir = self._get_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract configuration parameters
        self.params = self._build_params()
        
        # Initialize exchange objects
        self.exchange = None
        self.async_exchange = None
        
        # Set up concurrency limiting semaphore
        self.request_semaphore = None
        
        # Initialize exchange
        self._init_exchange()
        
        # Settings
        self.rate_limit = self.config.get('api', 'rate_limits', 'requests', default=20)
        self.current_requests = 0
        self.rate_limit_reset = time.time() + 60
        self.download_chunk_size = self.config.get('api', 'download_chunk_size', default=5000)
        self.max_retry_attempts = self.config.get('network', 'connection', 'retry_attempts', default=3)
        self.retry_delay = self.config.get('network', 'connection', 'retry_delay', default=2.0)
        
        # WebSocket status tracking
        self.ws_subscriptions = {}
    
        self.rate_limit = self.config.get('api', 'rate_limits', 'requests', default=20)    # Set up rate limiting using token bucket
        tokens_per_second = (self.rate_limit * 0.95) / 60.0    # Initialize with slightly lower rate to provide safety margin
        self.token_bucket = TokenBucket(rate=tokens_per_second, capacity=self.rate_limit)
        
        logger.info("Binance interface initialized with high-latency network optimizations")
    
    def _get_cache_dir(self) -> str:
        """Get cache directory from configuration"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        return os.path.abspath(os.path.expanduser(cache_dir))
    
    def _build_params(self) -> Dict:
        """
        从配置构建CCXT参数，增强安全性和网络性能
        
        Returns:
        CCXT参数字典
        """
        params = {
            'apiKey': self.config.get('api', 'binance', 'api_key', default=''),
            'secret': self.config.get('api', 'binance', 'secret', default=''),
            'timeout': self.config.get('api', 'timeout', default=60000),  # 增加到60秒
            'enableRateLimit': self.config.get('api', 'enable_rate_limit', default=True),
            'options': {
                'adjustForTimeDifference': self.config.get(
                    'api', 'binance', 'options', 'adjust_time_diff', default=True),
                'recvWindow': self.config.get(
                    'api', 'binance', 'options', 'recv_window', default=60000),  # 最大值
                'defaultType': self.config.get(
                    'api', 'binance', 'options', 'default_type', default='spot'),
                'keepAlive': True,  # 启用TCP Keep-Alive
            },
            'headers': {
                'Connection': 'keep-alive',  # 重要: 保持连接打开
                'Keep-Alive': '60',          # 保持60秒
            }
        }
        
        # 检查配置是否明确启用或禁用代理
        use_proxy = self.config.get('api', 'useproxy', default=None)
        
        # 如果没有明确禁用代理，尝试使用代理
        if use_proxy is True or use_proxy == "true":
            # 从配置中获取代理
            http_proxy = self.config.get('proxies', 'http', default=None)
            https_proxy = self.config.get('proxies', 'https', default=None)
            
            proxies = {}
            if http_proxy or https_proxy:
                # 使用配置中指定的代理
                proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
                logger.info(f"使用配置中指定的代理: {proxies}")
                if proxies:
                    params['proxies'] = proxies
            
        return params

    def _init_exchange(self):
        """Initialize exchange connection with improved error handling and recovery mechanisms"""
        
        # Check if offline mode is enabled
        offline_mode = self.config.get('system', 'offline_mode', default=False)
        if offline_mode:
            logger.info("Running in offline mode, skipping Binance exchange initialization")
            self.exchange = None
            return
        
        # Track proxy status
        using_proxy = False
        
        try:
            logger.info("Initializing Binance exchange")
            
            # Add custom async HTTP client settings
            import aiohttp
            
            # Optimize HTTP connector for high-latency networks
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,      # Keep connections for 60 seconds
                limit_per_host=10,         # Max 10 connections per host
                limit=20,                  # Total connection limit
                ttl_dns_cache=300,         # DNS cache for 5 minutes
                use_dns_cache=True,        # Use DNS cache
                enable_cleanup_closed=True # Auto-cleanup closed connections
            )
            
            # Try to initialize exchange
            try:
                # First test synchronous exchange
                self.exchange = ccxt.binance(self.params)
                logger.info("Synchronous exchange initialized successfully")
                
                # Set up async request semaphore
                self.request_semaphore = asyncio.Semaphore(5)  # Limit max concurrent requests
                
                # Preload market data
                try:
                    self.exchange.load_markets()
                    logger.info("Market data loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load market data: {str(e)}, will load on first call")
                
            except Exception as e:
                logger.error(f"Error initializing exchange: {str(e)}")
                raise RuntimeError(f"Cannot initialize exchange: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {str(e)}")
            
            # Special handling for offline backtest support
            if self.config.get('backtest', 'fallback_to_local', default=True):
                logger.warning("Exchange initialization failed but fallback_to_local enabled. Creating minimal exchange instance.")
                self.exchange = ccxt.binance()
                self.exchange.markets = {}
                self.exchange.markets_by_id = {}
                return
                
            raise RuntimeError(f"Binance initialization failed: {str(e)}")

    async def _init_async_exchange(self) -> None:
        """Initialize async exchange (lazy initialization), optimized for high-latency networks"""
        if self.async_exchange is not None:
            return

        try:
            # Create optimized aiohttp session
            import aiohttp
            from aiohttp import ClientTimeout
            
            # Optimize HTTP connector for high-latency networks
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,      # Keep connections for 60 seconds
                limit_per_host=10,         # Max 10 connections per host
                limit=20,                  # Total connection limit
                ttl_dns_cache=300,         # DNS cache for 5 minutes
                use_dns_cache=True,        # Use DNS cache
                enable_cleanup_closed=True # Auto-cleanup closed connections
            )
            
            # Set longer timeouts to accommodate high-latency
            timeout = ClientTimeout(
                total=60,                 # Total timeout 60 seconds
                connect=20,               # Connection timeout 20 seconds
                sock_read=45,             # Socket read timeout 45 seconds
                sock_connect=20           # Socket connect timeout 20 seconds
            )
            
            # Create session
            session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive', 'Keep-Alive': '60'}
            )
            
            # Initialize async exchange
            self.async_exchange = ccxt_async.binance(self.params)
            
            # Apply optimized session
            self.async_exchange.session = session
            
            # Initialize concurrency limiting semaphore
            if not self.request_semaphore:
                self.request_semaphore = asyncio.Semaphore(5)
            
            logger.info("Async Binance exchange initialized with high-latency network optimizations")
        except Exception as e:
            logger.error(f"Failed to initialize async Binance exchange: {str(e)}")
            self.async_exchange = None
    
    async def _handle_rate_limit(self) -> None:
        """Manage rate limits for API requests with improved distribution"""
        wait_time = await self.token_bucket.consume(1)    # Consume a token (wait if necessary)
        if wait_time > 0:
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
        
        current_time = time.time()
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # Instead of hitting the limit and waiting, try to spread requests evenly
        elapsed_time = current_time - (self.rate_limit_reset - 60)
        elapsed_ratio = elapsed_time / 60.0
        
        # Calculate ideal max requests at this point in time. Calculate time to wait
        ideal_max_requests = min(self.rate_limit, int(self.rate_limit * elapsed_ratio) + 1)
        
        if self.current_requests >= ideal_max_requests:
            
            time_per_request = 60.0 / self.rate_limit
            wait_time = max(0.1, time_per_request - (random.random() * 0.1))  # Small jitter
            
            logger.debug(f"Smoothing API request rate, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
        
        # If we're very close to the limit, implement a longer wait
        if self.current_requests >= self.rate_limit - 2:
            remaining_time = max(0, self.rate_limit_reset - current_time)
            if remaining_time > 0:
                logger.warning(f"Rate limit nearly reached, waiting {remaining_time:.2f} seconds")
                await asyncio.sleep(remaining_time)
                
                self.current_requests = 0    # Reset counter and window after waiting
                self.rate_limit_reset = time.time() + 60
                return
        
        self.current_requests += 1
    
    async def _exponential_backoff(self, attempt: int) -> float:
        """Implement exponential backoff with jitter"""
        base_delay = 1.0
        max_delay = 30.0
        
        # Calculate exponential backoff time with random jitter
        delay = min(max_delay, base_delay * (2 ** attempt)) 
        jitter = random.uniform(0.5, 1.0)  # 50-100% random jitter
        final_delay = delay * jitter
        logger.info(f"Will wait {final_delay:.2f} seconds before retry (attempt {attempt+1})")
        await asyncio.sleep(final_delay)
        return final_delay
    
    @staticmethod
    def _process_ohlcv_data(ohlcv_data: List) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame"""
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    @retry_exchange_operation(max_attempts=3, base_delay=2.0, max_delay=30.0)
    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """Get latest OHLCV data with optimized retry mechanism"""
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange not initialized, cannot fetch data")
                return pd.DataFrame()
        
        try:
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol, 
                timeframe=timeframe, 
                limit=limit
            )

            df = self._process_ohlcv_data(ohlcv)
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} latest records for {symbol} {timeframe}")
                return df
            
            logger.warning(f"Empty response when fetching {symbol} data")
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise  # Let decorator handle retries
    
    async def fetch_historical_ohlcv(self, 
                                symbol: str, 
                                timeframe: str = '1m',
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical OHLCV data, optimized for high-latency networks"""
        # Ensure exchange is initialized
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange initialization failed")
                return pd.DataFrame()
        
        try:
            start_dt = TimeUtils.parse_timestamp(start_date, default_days_ago=30)
            end_dt = TimeUtils.parse_timestamp(end_date, default_days_ago=0)
            logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Split request into multiple chunks to handle large date ranges
            chunks = self._get_date_chunks(start_dt, end_dt, timeframe)
            logger.info(f"Split request into {len(chunks)} chunks")
            
            all_data = []
            
            # Re-evaluate the chunking strategy for very large requests
            if len(chunks) > 20:  # Arbitrary threshold
                logger.warning(f"Large number of chunks ({len(chunks)}), implementing special handling")
                
                all_data = []
                batch_size = 5  # Process 5 chunks at a time
                
                for i in range(0, len(chunks), batch_size):
                    batch_chunks = chunks[i:i+batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size} ({len(batch_chunks)} chunks)")
                    
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
                            logger.error(f"Chunk processing failed: {str(result)}")
                        elif isinstance(result, list) and result:
                            all_data.extend(result)
                    
                    # Pause between batches to avoid rate limit issues
                    if i + batch_size < len(chunks):
                        pause_time = 3  # 3 seconds between batches
                        logger.info(f"Pausing {pause_time} seconds between batch processing")
                        await asyncio.sleep(pause_time)
            else:
                tasks = []
                # Process chunks concurrently, but limited by semaphore
                for i, (chunk_start, chunk_end) in enumerate(chunks):
                    task = asyncio.create_task(self._fetch_chunk(i, chunk_start, chunk_end, symbol, timeframe))
                    tasks.append(task)
        
                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in chunk_results:
                    if isinstance(result, Exception):
                        logger.error(f"Chunk processing failed: {str(result)}")
                    elif isinstance(result, list) and result:
                        all_data.extend(result)
            
            if not all_data:
                logger.warning(f"No historical data found for {symbol}")
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
            
            logger.info(f"Downloaded {len(df)} candle data points for {symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _fetch_chunk(self, chunk_index: int, chunk_start: datetime, 
                        chunk_end: datetime, symbol: str, timeframe: str) -> List:
        """Fetch a single historical data chunk with retries and improved rate limiting"""
        # Use semaphore to limit concurrency
        async with self.request_semaphore:
            # Calculate chunk durations and expected data points to determine priority
            chunk_duration = (chunk_end - chunk_start).total_seconds()
            seconds_per_candle = TimeUtils.timeframe_to_seconds(timeframe)
            expected_points = int(chunk_duration / seconds_per_candle)
            
            # If this is a large chunk, further subdivide it to avoid rate limit issues
            if expected_points > self.download_chunk_size / 2 and chunk_duration > 86400:  # More than half max size and >1 day
                logger.debug(f"Splitting large chunk {chunk_index+1} into smaller pieces")
                
                mid_point = chunk_start + timedelta(seconds=chunk_duration/2)
                
                # Process first half
                first_half = await self._fetch_chunk(
                    chunk_index * 2, 
                    chunk_start, 
                    mid_point, 
                    symbol, 
                    timeframe
                )
                
                await asyncio.sleep(0.5)    # Small delay between sub-chunks
                
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
                        logger.info(f"Fetching chunk {chunk_index+1}: {chunk_start} to {chunk_end}")
                    
                    exchange = self.async_exchange if self.async_exchange else self.exchange
                    
                    ohlcv = await exchange.fetch_ohlcv(
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
                        logger.debug(f"Chunk {chunk_index+1} returned no data")
                        return []
                    
                    logger.debug(f"Chunk {chunk_index+1} successfully fetched {len(ohlcv)} records")
                    
                    # Adaptive delay based on how many records we got relative to limit
                    ratio = len(ohlcv) / self.download_chunk_size
                    delay = 0.2 + (0.5 * ratio) if ratio > 0.8 else 0.2
                    await asyncio.sleep(delay)
                    
                    return ohlcv
                    
                except ccxt.RateLimitExceeded as e:
                    # Special handling for explicit rate limit errors
                    logger.warning(f"Rate limit exceeded on chunk {chunk_index+1}, attempt {retry+1}")
                    wait_time = 60.0  # Full minute wait
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    self.current_requests = 0
                    self.rate_limit_reset = time.time() + 60
                    
                except Exception as e:
                    logger.warning(f"Chunk {chunk_index+1}, attempt {retry+1} failed: {str(e)}")
                    if retry < self.max_retry_attempts - 1:
                        await self._exponential_backoff(retry)
            
            logger.error(f"All retry attempts for chunk {chunk_index+1} failed")
            return []
    
    def _get_date_chunks(self, start_dt: datetime, end_dt: datetime, 
                        timeframe: str, chunk_size_days: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """Break date range into manageable chunks with adaptive sizing based on timeframe"""
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
        ideal_chunk_seconds *= safety_factor    # Add a safety factor to account for API inconsistencies
        ideal_chunk_days = max(0.5, min(30, ideal_chunk_seconds / 86400))    # Convert to days for easier handling but enforce reasonable bounds

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
        logger.info(f"Using {chunk_size} chunk size for {timeframe} data from {start_dt} to {end_dt}")

        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end
        logger.info(f"Split request into {len(chunks)} chunks for {expected_candles:.0f} expected data points")
        
        return chunks
        
    async def watch_ohlcv(self, symbol: str, timeframe: str, callback=None):
        """
        Subscribe to real-time OHLCV data using WebSocket (requires ccxt.pro)
        
        Args:
            symbol: Trading pair symbol
            timeframe: Time period
            callback: Callback function to receive data
            
        Returns:
            None: Continuously running task
        """
        try:
            # Check if ccxt.pro is installed
            try:
                import ccxtpro
            except ImportError:
                logger.error("ccxt.pro not installed. Please install ccxt.pro to use WebSocket functionality.")
                return False
            
            # Subscription key
            sub_key = f"{symbol}_{timeframe}"
            
            # If already subscribed, avoid duplicate subscription
            if sub_key in self.ws_subscriptions and self.ws_subscriptions[sub_key]['active']:
                logger.info(f"Already subscribed to WebSocket data stream for {symbol} {timeframe}")
                return True
            
            logger.info(f"Starting WebSocket connection to monitor {symbol} {timeframe} in real-time")
            
            # Initialize exchange
            exchange = ccxtpro.binance(self.params)
            
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
            logger.error(f"Error setting up WebSocket connection: {str(e)}")
            return False
    
    async def _ws_manager(self, sub_key: str, symbol: str, timeframe: str, callback):
        """Manage the continuous operation of a WebSocket connection"""
        if sub_key not in self.ws_subscriptions:
            logger.error(f"WebSocket subscription for {sub_key} not found")
            return
        
        sub_info = self.ws_subscriptions[sub_key]
        exchange = sub_info['exchange']
        
        retry_delay = 1.0  # Initial retry delay
        max_retry_delay = 30.0  # Maximum retry delay
        
        while sub_info['active']:
            try:
                logger.debug(f"Waiting for WebSocket update for {symbol} {timeframe}")
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
                        logger.error(f"Callback processing error: {str(callback_error)}")
                
            except Exception as e:
                sub_info['errors'] += 1
                logger.error(f"WebSocket error ({sub_info['errors']}): {str(e)}")
                
                # If too many errors, may need complete reconnection
                if sub_info['errors'] > 10:
                    logger.warning(f"Too many errors, reinitializing WebSocket connection")
                    try:
                        await exchange.close()
                        # Recreate exchange instance
                        import ccxtpro
                        exchange = ccxtpro.binance(self.params)
                        exchange.options['ws'] = {
                            'heartbeat': True,
                            'ping_interval': 30000,
                            'reconnect_rate': 5000,
                            'max_reconnects': 100,
                        }
                        sub_info['exchange'] = exchange
                        sub_info['errors'] = 0
                    except Exception as reset_error:
                        logger.error(f"Error resetting WebSocket connection: {str(reset_error)}")
                
                # Use exponential backoff
                retry_delay = min(retry_delay * 2, max_retry_delay)
                jitter = random.uniform(0.5, 1.0)
                actual_delay = retry_delay * jitter
                logger.info(f"Waiting {actual_delay:.2f} seconds before retrying WebSocket connection")
                await asyncio.sleep(actual_delay)
    
    async def stop_watching(self, symbol: str, timeframe: str):
        """Stop WebSocket data subscription"""
        sub_key = f"{symbol}_{timeframe}"
        
        if sub_key in self.ws_subscriptions:
            try:
                # Mark as inactive
                self.ws_subscriptions[sub_key]['active'] = False
                
                # Close exchange connection
                exchange = self.ws_subscriptions[sub_key].get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
                    
                logger.info(f"Stopped WebSocket subscription for {symbol} {timeframe}")
                return True
            except Exception as e:
                logger.error(f"Error stopping WebSocket subscription: {str(e)}")
                return False
        
        logger.warning(f"No active WebSocket subscription found for {symbol} {timeframe}")
        return False
        
    async def close(self):
        """Clean up resources"""
        # Close all WebSocket subscriptions
        for sub_key, sub_info in list(self.ws_subscriptions.items()):
            try:
                sub_info['active'] = False
                exchange = sub_info.get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket subscription {sub_key}: {str(e)}")
        
        # Close async exchange
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                # Close underlying session
                if hasattr(self.async_exchange, 'session') and hasattr(self.async_exchange.session, 'close'):
                    await self.async_exchange.session.close()
                self.async_exchange = None
                logger.info("Async exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing async exchange: {str(e)}")