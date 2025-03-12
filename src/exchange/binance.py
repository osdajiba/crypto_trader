# src/exchange/binance.py
import platform
import socket
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import requests
import functools
import concurrent.futures
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from functools import lru_cache, wraps
from src.common.config_manager import ConfigManager

from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")


def retry_exchange_operation(max_attempts=3, delay_base=1):
    """
    Decorator for retrying exchange operations with exponential backoff
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay_base: Base delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    last_exception = e
                    
                    # Exponential backoff with jitter
                    delay = delay_base * (2 ** attempt)
                    time.sleep(delay)
                except Exception as e:
                    logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # If all attempts fail
            logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
        return wrapper
    return decorator


class Binance:
    """
    High-performance Binance interface with optimized data retrieval and caching
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Binance interface with configuration
        
        Args:
            config: Configuration dictionary with API credentials and settings
        """
        
        self.config = config
        
        # Set up cache directory
        self.cache_dir = self._safe_get_config('data', 'paths', 'historical_data_path', default='data/historical')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract configuration parameters with safe defaults
        self.params = self._build_params()
        
        # Initialize exchange objects
        self.exchange = None
        self.async_exchange = None
        
        # Advanced initialization with retry
        self._init_exchange()
        
        # Rate limiting configuration
        self.rate_limit = self._safe_get_config('api', 'rate_limits', 'max_calls_per_second', default=10) * 60
        self.current_requests = 0
        self.rate_limit_reset = time.time() + 60
        
        # Download settings
        self.download_chunk_size = 500  # Balanced for stability
        self.max_retry_attempts = 3
        self.retry_delay = 2
        
        # Latency tracking
        self.latency_history = []
        self.max_latency_history = 20  # Keep last 20 measurements
        
        # Measure initial latency
        self._measure_latency()
        
        logger.info("Binance interface initialized successfully")
    
    def _init_exchange(self):
        """
        Robust exchange initialization with comprehensive error handling and fallback
        """
        def check_network_connectivity():
            """Check basic network connectivity"""
            try:
                # Try to ping multiple endpoints
                endpoints = [
                    'https://api.binance.com/api/v3/ping',
                    'https://api1.binance.com/api/v3/ping',
                    'https://api2.binance.com/api/v3/ping'
                ]
                
                for endpoint in endpoints:
                    try:
                        response = requests.get(endpoint, timeout=10)
                        if response.status_code == 200:
                            logger.info(f"Network connectivity verified: {endpoint}")
                            return True
                    except Exception as e:
                        logger.warning(f"Endpoint {endpoint} failed: {str(e)}")
                
                logger.error("No Binance API endpoints are reachable")
                return False
            except Exception as e:
                logger.error(f"Network connectivity check failed: {str(e)}")
                return False

        # First, check network connectivity
        if not check_network_connectivity():
            logger.critical("No network connectivity to Binance API")
            raise RuntimeError("Unable to establish network connection to Binance")

        # Try each parameter set
        try:
            logger.info(f"Attempting exchange initialization")
            
            # Create exchange instance
            self.exchange = ccxt.binance(self.params)
            
            # Attempt to load markets with timeout
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self.exchange.load_markets)
                    market_data = future.result(timeout=30)
                
                # Additional verification
                if not market_data:
                    logger.warning(f"No markets loaded in strategy")
                    self.exchange = None
                
                logger.info(f"Successfully initialized exchange with strategy")
                return
            
            except concurrent.futures.TimeoutError:
                logger.warning(f"Market loading timed out in strategy")
                self.exchange = None
            
            except Exception as market_error:
                logger.warning(f"Market loading failed in strategy: {str(market_error)}")
                self.exchange = None
        
        except (ccxt.NetworkError, ccxt.ExchangeError) as init_error:
            logger.warning(f"Exchange initialization failed in strategy: {str(init_error)}")
            self.exchange = None
        
        except Exception as unexpected_error:
            logger.error(f"Unexpected error in strategy: {str(unexpected_error)}")
            self.exchange = None

        # If all strategies fail
        error_msg = "Failed to initialize Binance exchange after all attempts. " \
                    "Check network, API credentials, and proxy settings."
        logger.critical(error_msg)
        
        # Optionally, you can add more diagnostic information
        import platform
        import socket
        
        logger.info(f"System Information:")
        logger.info(f"OS: {platform.platform()}")
        logger.info(f"Python Version: {platform.python_version()}")
        try:
            logger.info(f"Hostname: {socket.gethostname()}")
            logger.info(f"IP Address: {socket.gethostbyname(socket.gethostname())}")
        except Exception:
            pass

        raise RuntimeError(error_msg)

    def _safe_get_config(self, *keys, default=None):
        """
        Safely retrieve a value from nested dictionaries
        
        Args:
            *keys: The keys to navigate through
            default: Default value if key doesn't exist
            
        Returns:
            Value or default
        """
        current = self.config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current
    
    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration with enhanced security
        
        Returns:
        Dictionary of CCXT parameters
        """

        params = {
            'apiKey': self._safe_get_config('api', 'binance', 'apiKey', default=''),
            'secret': self._safe_get_config('api', 'binance', 'secret', default=''),
            'timeout': self._safe_get_config('api', 'timeout', default=30000),
            'enableRateLimit': self._safe_get_config('api', 'enableRateLimit', default=True),
            'options': {
                'adjustForTimeDifference': self._safe_get_config(
                    'default_config', 'options', 'adjustForTimeDifference', default=True),
                'recvWindow': self._safe_get_config(
                    'default_config', 'options', 'recvWindow', default=60000),
                'defaultType': self._safe_get_config(
                    'default_config', 'options', 'defaultType', default='spot')
            },
            'proxies': {
                'http': self._safe_get_config('proxies', 'http', default="http://127.0.0.1:7890"), 
                'https': self._safe_get_config('proxies', 'https', default="http://127.0.0.1:7890")
            }
        }

        return params
    
    def _validate_proxy(self, proxy: str) -> bool:
        """
        Validate proxy configuration
        
        Args:
            proxy: Proxy URL to validate
        
        Returns:
            Boolean indicating proxy validity
        """
        try:
            # Basic proxy URL validation
            if not proxy.startswith(('http://', 'https://')):
                logger.warning(f"Invalid proxy format: {proxy}")
                return False
            
            # Optional: Add actual proxy connectivity test
            requests.get('https://api.binance.com/api/v3/ping', 
                         proxies={'http': proxy, 'https': proxy}, 
                         timeout=5)
            return True
        except Exception as e:
            logger.warning(f"Proxy validation failed: {str(e)}")
            return False

    async def _init_async_exchange(self) -> None:
        """Initialize the asynchronous exchange connection (lazy initialization)"""
        if self.async_exchange is not None:
            return

        try:
            self.async_exchange = ccxt_async.binance(self.params)
            logger.info("CCXT Async Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Async Binance exchange: {str(e)}")
            self.async_exchange = None
    
    def _measure_latency(self) -> float:
        """
        Measure API latency to Binance
        
        Returns:
            Latency in milliseconds
        """
        try:
            proxies = {}
            http_proxy = self._safe_get_config('proxies', 'http')
            https_proxy = self._safe_get_config('proxies', 'https')
            
            if http_proxy:
                proxies['http'] = http_proxy
            if https_proxy:
                proxies['https'] = https_proxy
                
            test_url = "https://api.binance.com/api/v3/ping"
            
            start_time = time.time()
            response = requests.get(test_url, proxies=proxies, timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                latency = (end_time - start_time) * 1000  # Convert to ms
                
                # Store in history
                self.latency_history.append(latency)
                # Keep history size limited
                if len(self.latency_history) > self.max_latency_history:
                    self.latency_history.pop(0)
                    
                logger.info(f"Current API latency: {latency:.2f}ms")
                return latency
            else:
                logger.warning(f"Latency test failed with status code: {response.status_code}")
                return float('inf')
        except Exception as e:
            logger.warning(f"Latency measurement failed: {str(e)}")
            return float('inf')
    
    def get_average_latency(self) -> float:
        """
        Get average latency from history
        
        Returns:
            Average latency in milliseconds
        """
        if not self.latency_history:
            # Measure now if we don't have data
            return self._measure_latency()
            
        return sum(self.latency_history) / len(self.latency_history)
    
    def _handle_rate_limit(self) -> None:
        """
        Handle rate limiting for API requests
        """
        current_time = time.time()
        
        # Check if the rate limit window has reset
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # Check if we're at the rate limit
        if self.current_requests >= self.rate_limit:
            # Calculate sleep time
            sleep_time = max(0, self.rate_limit_reset - current_time)
            logger.warning(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
            
            # Reset after sleeping
            self.current_requests = 0
            self.rate_limit_reset = time.time() + 60
        
        # Increment the request counter
        self.current_requests += 1

    async def _async_handle_rate_limit(self) -> None:
        """
        Handle rate limiting for asynchronous API requests
        """
        current_time = time.time()
        
        # Check if the rate limit window has reset
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # Check if we're at the rate limit
        if self.current_requests >= self.rate_limit:
            # Calculate sleep time
            sleep_time = max(0, self.rate_limit_reset - current_time)
            logger.warning(f"Async rate limit reached, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)
            
            # Reset after sleeping
            self.current_requests = 0
            self.rate_limit_reset = time.time() + 60
        
        # Increment the request counter
        self.current_requests += 1
    
    @staticmethod
    def _process_ohlcv_data(ohlcv_data: List) -> pd.DataFrame:
        """
        Process OHLCV data into a DataFrame
        
        Args:
            ohlcv_data: Raw OHLCV data from CCXT
        
        Returns:
            Processed DataFrame
        """
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime with UTC timezone
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # Sort by timestamp to ensure data is in chronological order
            df = df.sort_values('timestamp')
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            return df
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """
        Fetch recent OHLCV data synchronously
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '1h', '1d')
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV data
        """
        # First check recent cache to avoid unnecessary API calls
        cache_data = self._check_recent_cache(symbol, timeframe)
        if not cache_data.empty:
            return cache_data[:limit]  # Return the most recent entries
        
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange not initialized, can't fetch data")
                return pd.DataFrame()
        
        for attempt in range(self.max_retry_attempts):
            try:
                # Handle rate limiting
                self._handle_rate_limit()
                
                # Fetch OHLCV data
                ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
                
                # Convert to DataFrame
                df = self._process_ohlcv_data(ohlcv)
                
                # Save to cache if successful
                if not df.empty:
                    self._append_to_cache(df, symbol, timeframe)
                    return df
                
                logger.warning(f"Empty response on attempt {attempt+1}")
                time.sleep(self.retry_delay)
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV data (attempt {attempt+1}): {str(e)}")
                time.sleep(self.retry_delay * (attempt + 1))
        
        # If all attempts fail, try to return cached data even if not recent
        return self._get_fallback_cache_data(symbol, timeframe, limit)
    
    def _check_recent_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Check if we have recent data in cache
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            
        Returns:
            DataFrame with recent cached data or empty DataFrame
        """
        safe_symbol = symbol.replace('/', '_')
        cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
        
        if not os.path.exists(cache_file):
            return pd.DataFrame()
            
        try:
            # Load the CSV file - use efficient reading with only necessary columns
            df = pd.read_csv(cache_file, index_col=None)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            else:
                return pd.DataFrame()
                
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Check if cache is recent enough
            if not df.empty:
                last_time = df.index.max()
                now = pd.Timestamp.now(tz='UTC')
                
                # For smaller timeframes, need more recent data
                if timeframe in ['1m', '5m', '15m']:
                    max_age = timedelta(hours=1)  # Cache valid for 1 hour
                elif timeframe in ['30m', '1h', '2h']:
                    max_age = timedelta(hours=6)  # Cache valid for 6 hours
                else:
                    max_age = timedelta(days=1)   # Cache valid for 1 day
                
                if now - last_time < max_age:
                    logger.info(f"Using recent cache data for {symbol} {timeframe}")
                    return df.sort_index()
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Error checking recent cache: {str(e)}")
            return pd.DataFrame()
    
    def _get_fallback_cache_data(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Get fallback data from cache even if not recent
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of entries to return
            
        Returns:
            DataFrame with cached data
        """
        safe_symbol = symbol.replace('/', '_')
        cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
        
        if not os.path.exists(cache_file):
            return pd.DataFrame()
            
        try:
            # Load the CSV file
            df = pd.read_csv(cache_file, index_col=None)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            else:
                return pd.DataFrame()
                
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            if not df.empty:
                logger.warning(f"Using outdated cache data for {symbol} {timeframe} as fallback")
                return df.sort_index().tail(limit)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Error getting fallback cache data: {str(e)}")
            return pd.DataFrame()
            
    def _append_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Append new data to cache, avoiding duplicates
        
        Args:
            df: New data to append
            symbol: Trading pair symbol
            timeframe: Timeframe
        """
        if df.empty:
            return
            
        safe_symbol = symbol.replace('/', '_')
        cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
        
        try:
            existing_df = pd.DataFrame()
            
            # Load existing data if available
            if os.path.exists(cache_file):
                existing_df = pd.read_csv(cache_file, index_col=None)
                
                # Convert timestamp to datetime
                if 'timestamp' in existing_df.columns:
                    existing_df['datetime'] = pd.to_datetime(existing_df['timestamp'], unit='ms', utc=True)
                elif 'datetime' in existing_df.columns:
                    existing_df['datetime'] = pd.to_datetime(existing_df['datetime'], utc=True)
                
                # Set datetime as index
                existing_df.set_index('datetime', inplace=True)
            
            # Combine existing and new data
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, df])
                # Remove duplicates efficiently
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                # Sort by datetime
                combined_df = combined_df.sort_index()
            else:
                combined_df = df
            
            # Save to file
            combined_df.to_csv(cache_file)
            logger.debug(f"Updated cache for {symbol} {timeframe}, total records: {len(combined_df)}")
            
        except Exception as e:
            logger.warning(f"Error appending to cache: {str(e)}")
    
    @lru_cache(maxsize=128)
    def _parse_date(self, date_input: Optional[Union[str, datetime, int, float]], default_days_ago: int = 0) -> datetime:
        """
        Parse various date inputs into datetime object with timezone
        
        Args:
            date_input: Date input (string, datetime object, timestamp)
            default_days_ago: Days ago to use if input is None
            
        Returns:
            datetime object with UTC timezone
        """
        # If input is None, return relative to current time
        if date_input is None:
            return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # If already a datetime object
        if isinstance(date_input, datetime):
            # Ensure it has timezone info
            return date_input if date_input.tzinfo else date_input.replace(tzinfo=timezone.utc)
        
        # If integer or float (Unix timestamp)
        if isinstance(date_input, (int, float)):
            # Determine if milliseconds or seconds
            if date_input > 1e10:  # Usually millisecond timestamps are > 1e10
                date_input = date_input / 1000  # Convert to seconds
            return datetime.fromtimestamp(date_input, tz=timezone.utc)
        
        # String handling
        if isinstance(date_input, str):
            try:
                # Try automatic parsing with pandas
                dt = pd.to_datetime(date_input, utc=True)
                return dt.to_pydatetime()
            except Exception as e:
                # If parsing fails, try common formats
                formats = [
                    '%Y-%m-%d',            # 2023-01-31
                    '%Y/%m/%d',            # 2023/01/31
                    '%d-%m-%Y',            # 31-01-2023
                    '%d/%m/%Y',            # 31/01/2023
                    '%Y-%m-%d %H:%M:%S',   # 2023-01-31 14:30:00
                    '%Y-%m-%dT%H:%M:%S',   # 2023-01-31T14:30:00
                    '%Y%m%d',              # 20230131
                ]
                
                for fmt in formats:
                    try:
                        dt = datetime.strptime(date_input, fmt)
                        # Add UTC timezone
                        return dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        continue
                
                # All attempts failed
                logger.warning(f"Could not parse date '{date_input}'. Using default.")
                return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # Fallback for unsupported types
        logger.warning(f"Unsupported date type {type(date_input)}. Using default.")
        return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
    
    def _get_date_chunks(self, start_dt: datetime, end_dt: datetime, timeframe: str, chunk_size_days: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """
        Split date range into manageable chunks for fetching
        
        Args:
            start_dt: Start datetime
            end_dt: End datetime
            timeframe: Timeframe
            chunk_size_days: Optional override for chunk size in days
            
        Returns:
            List of (start, end) datetime tuples for each chunk
        """
        # Determine chunk size based on timeframe (if not overridden)
        if chunk_size_days is None:
            if timeframe in ['1m', '5m']:
                # For small timeframes, use smaller chunks
                chunk_size = timedelta(days=1)
            elif timeframe in ['15m', '30m', '1h']:
                # For medium timeframes, use medium chunks
                chunk_size = timedelta(days=7)
            else:
                # For larger timeframes, use larger chunks
                chunk_size = timedelta(days=30)
        else:
            chunk_size = timedelta(days=chunk_size_days)
        
        # Create chunks
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end
        
        return chunks

    async def fetch_historical_ohlcv(self, 
                                symbol: str, 
                                timeframe: str = '1h',
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical OHLCV data with optimized performance
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_date: Start date/time
            end_date: End date/time
        
        Returns:
            DataFrame with historical OHLCV data
        """
        # Make sure exchange is initialized
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("Exchange initialization failed")
                return pd.DataFrame()
        
        try:
            # Process dates
            start_dt = self._parse_date(start_date, default_days_ago=30)
            end_dt = self._parse_date(end_date, default_days_ago=0)
            
            logger.info(f"Fetching historical data for {symbol} from {start_dt} to {end_dt}")
            
            # First check if data is in cache
            cache_data = self._get_from_cache(symbol, timeframe, start_dt, end_dt)
            if not cache_data.empty:
                logger.info(f"Using cached data for {symbol} {timeframe}")
                return cache_data
            
            # Get data chunks based on timeframe
            chunks = self._get_date_chunks(start_dt, end_dt, timeframe)
            logger.info(f"Splitting request into {len(chunks)} chunks")
            
            # Track progress and accumulated data
            chunks_processed = 0
            chunks_successful = 0
            all_data = []
            
            # Measure latency before fetching to optimize timings
            current_latency = self._measure_latency()
            delay_between_chunks = max(0.5, min(2.0, current_latency / 1000 * 2))
            
            # Pre-check if the exchange is properly loaded and markets are available
            try:
                if not hasattr(self.exchange, 'markets') or not self.exchange.markets:
                    logger.info("Loading markets before fetching OHLCV data")
                    self.exchange.load_markets()
            except Exception as e:
                logger.warning(f"Failed to load markets: {str(e)}. Continuing anyway...")
            
            # Process each chunk with improved error handling
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                for retry in range(self.max_retry_attempts):
                    try:
                        # Handle rate limiting
                        await self._async_handle_rate_limit()
                        
                        # Convert to millisecond timestamp
                        chunk_since = int(chunk_start.timestamp() * 1000)
                        
                        # Log for first chunk or every 5th chunk to reduce log noise
                        if i == 0 or i % 5 == 0:
                            logger.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                        
                        # Fetch the chunk with more explicit parameters
                        self.async_exchange = self._init_async_exchange()
                        ohlcv = await self.async_exchange.fetch_ohlcv(
                            symbol=symbol, 
                            timeframe=timeframe, 
                            since=chunk_since,
                            limit=self.download_chunk_size,
                            params={"endTime": int(chunk_end.timestamp() * 1000)}
                        )
                        
                        chunks_processed += 1
                        
                        # Check if we got data
                        if not ohlcv or len(ohlcv) == 0:
                            logger.debug(f"No data returned for chunk {i+1}")
                            break  # No need to retry, just move to next chunk
                        
                        # Add data to accumulated list
                        all_data.extend(ohlcv)
                        chunks_successful += 1
                        
                        # Short delay between chunks, adjusted based on measured latency
                        await asyncio.sleep(delay_between_chunks)
                        
                        # Successful fetch, break retry loop
                        break
                        
                    except ccxt.RateLimitExceeded as e:
                        logger.warning(f"Rate limit exceeded on chunk {i+1} (attempt {retry+1}): {str(e)}")
                        # Exponential backoff for rate limit errors
                        await asyncio.sleep(self.retry_delay * (2 ** retry))
                        
                    except ccxt.NetworkError as e:
                        logger.warning(f"Network error on chunk {i+1} (attempt {retry+1}): {str(e)}")
                        await asyncio.sleep(self.retry_delay)
                        
                    except Exception as e:
                        error_message = str(e)
                        logger.error(f"Error fetching chunk {i+1} (attempt {retry+1}): {error_message}")
                        
                        # Specific handling for common error types
                        if "exchangeInfo" in error_message:
                            logger.info("Attempting to refresh exchange info...")
                            try:
                                self.exchange.load_markets(reload=True)
                                logger.info("Successfully refreshed exchange info")
                            except Exception as refresh_error:
                                logger.warning(f"Failed to refresh exchange info: {str(refresh_error)}")
                        
                        # Wait before retry with progressive backoff
                        await asyncio.sleep(self.retry_delay * (retry + 1))
                
                # If we've tried all retries for this chunk, continue to the next chunk
            
            # Process all data
            if not all_data:
                logger.warning(f"No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = self._process_ohlcv_data(all_data)
            
            # Filter by date range more elegantly
            if not df.empty:
                # Use index filtering (more reliable)
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            logger.info(f"Downloaded {len(df)} candles for {symbol} ({chunks_successful}/{len(chunks)} chunks successful)")
            
            # Save to cache for future use
            if not df.empty:
                self._save_to_cache(df, symbol, timeframe)
            
            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _get_from_cache(self, symbol: str, timeframe: str, 
                       start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
        """
        Get data from cache efficiently
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            DataFrame with data from cache
        """
        # Generate cache path
        safe_symbol = symbol.replace('/', '_')
        cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
        
        if not os.path.exists(cache_file):
            return pd.DataFrame()
        
        try:
            # Use faster reading method with chunksize for large files
            file_size = os.path.getsize(cache_file)
            
            # For small files, read directly
            if file_size < 10_000_000:  # 10MB threshold
                df = pd.read_csv(cache_file)
            else:
                # For large files, use chunked reading
                chunks = []
                for chunk in pd.read_csv(cache_file, chunksize=100000):
                    chunks.append(chunk)
                df = pd.concat(chunks)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            elif 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            else:
                return pd.DataFrame()
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Sort to ensure order is correct
            df = df.sort_index()
            
            # Filter by requested date range
            filtered_df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            
            # Determine if cache data is sufficient
            if len(filtered_df) > 0:
                # Check if the data covers the entire range with reasonable density
                range_seconds = (end_dt - start_dt).total_seconds()
                
                if timeframe.endswith('m'):  # minutes
                    # For minute timeframes, expect at least one candle per hour on average
                    expected_candles = range_seconds / 3600
                elif timeframe.endswith('h'):  # hours
                    # For hourly timeframes, expect at least 4 candles per day
                    expected_candles = range_seconds / 86400 * 4
                else:  # days or larger
                    # For daily timeframes, expect at least one candle per day
                    expected_candles = range_seconds / 86400
                
                if len(filtered_df) >= min(10, expected_candles * 0.5):
                    logger.info(f"Found {len(filtered_df)} records in cache for {symbol} {timeframe}")
                    return filtered_df
                else:
                    logger.info(f"Cache data insufficient: found {len(filtered_df)}, expected ~{expected_candles}")
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Error reading from cache: {str(e)}")
            return pd.DataFrame()
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Save data to cache
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Timeframe
        """
        try:
            # Ensure DataFrame is sorted
            df = df.sort_index()
            
            # Create cache path
            safe_symbol = symbol.replace('/', '_')
            cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # Save to CSV
            df.to_csv(cache_file)
            logger.info(f"Saved {len(df)} records to cache for {symbol} {timeframe}")
            
        except Exception as e:
            logger.warning(f"Error saving to cache: {str(e)}")
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                           start: Optional[Union[str, datetime]] = None,
                           end: Optional[Union[str, datetime]] = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        High-level method to fetch historical data with caching
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start: Start date/time
            end: End date/time
            use_cache: Whether to use cache system
            
        Returns:
            DataFrame with historical data
        """
        # Parse dates
        start_dt = self._parse_date(start, default_days_ago=30)
        end_dt = self._parse_date(end, default_days_ago=0)
        
        logger.info(f"Fetching historical data for {symbol} {timeframe} from {start_dt} to {end_dt}")
        
        # Check if we should use cache
        if use_cache:
            cache_data = self._get_from_cache(symbol, timeframe, start_dt, end_dt)
            if not cache_data.empty:
                logger.info(f"Using {len(cache_data)} records from cache for {symbol} {timeframe}")
                return cache_data
        
        # Get the data from exchange
        df = await self.fetch_historical_ohlcv(symbol, timeframe, start_dt, end_dt)
        
        # Save to cache if successful and caching is enabled
        if not df.empty and use_cache:
            self._save_to_cache(df, symbol, timeframe)
        
        return df
    
    async def fetch_realtime(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch real-time data with recent data from cache or API
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            limit: Number of candles to return
            
        Returns:
            DataFrame with real-time data
        """
        # Use the synchronous fetch which is optimized and reliable
        return self.fetch_latest_ohlcv(symbol, timeframe, limit)
    
    async def smart_fetch_ohlcv(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None,
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Smart fetching of OHLCV data with optimized performance and error handling
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start: Start date/time
            end: End date/time
            
        Returns:
            DataFrame with OHLCV data
        """
        # Reuse the optimized fetch_historical method
        return await self.fetch_historical(symbol, timeframe, start, end)
    
    def optimize_performance(self):
        """
        Optimize performance based on current conditions
        """
        # Measure current latency
        latency = self._measure_latency()
        
        # Adjust chunk size based on latency
        if latency < 100:  # Very fast connection
            self.download_chunk_size = 1000
            logger.info(f"Great connection speed ({latency:.2f}ms), using large chunk size: {self.download_chunk_size}")
        elif latency < 500:  # Good connection
            self.download_chunk_size = 500
            logger.info(f"Good connection speed ({latency:.2f}ms), using medium chunk size: {self.download_chunk_size}")
        else:  # Slow connection
            self.download_chunk_size = 200
            logger.info(f"Slow connection ({latency:.2f}ms), using small chunk size: {self.download_chunk_size}")
        
        # Adjust retry delay based on latency
        self.retry_delay = max(1, min(5, latency / 200))
    
    def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None):
        """
        Clear cache data
        
        Args:
            symbol: Optional specific symbol to clear
            timeframe: Optional specific timeframe to clear
        """
        try:
            if symbol and timeframe:
                # Clear specific cache file
                safe_symbol = symbol.replace('/', '_')
                cache_file = os.path.join(self.cache_dir, f"{safe_symbol}_{timeframe}.csv")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"Cleared cache for {symbol} {timeframe}")
            elif symbol:
                # Clear all files for a symbol
                safe_symbol = symbol.replace('/', '_')
                pattern = f"{safe_symbol}_"
                for file in os.listdir(self.cache_dir):
                    if file.startswith(pattern):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info(f"Cleared all cache files for {symbol}")
            else:
                # Clear all cache
                for file in os.listdir(self.cache_dir):
                    if file.endswith('.csv'):
                        os.remove(os.path.join(self.cache_dir, file))
                logger.info("Cleared all cache files")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    async def close(self):
        """Close exchange connections and free resources"""
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                self.async_exchange = None
                logger.info("Async exchange connection closed")
        except Exception as e:
            logger.error(f"Error closing async exchange: {str(e)}")