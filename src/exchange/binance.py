# src/exchange/binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import requests
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")

class Binance:
    """
    High-performance Binance interface with optimized data retrieval and caching
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize Binance interface with configuration
        
        Args:
            config: Configuration dictionary with API credentials and settings
        """
        self.config = config or {}
        
        # Set up cache directory
        self.cache_dir = self.config.get('data', 'paths', 'historical_data_path', default='data/historical')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Extract configuration parameters with safe defaults
        self.params = self._build_params()
        
        # Initialize exchange objects
        self.exchange = None
        self.async_exchange = None
        self._init_exchange()
        
        # Rate limiting configuration
        self.rate_limit = self.config.get('api', 'rate_limits', 'max_calls_per_second', default=10) * 60
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
    
    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration

        Returns:
        Dictionary of CCXT parameters
        """
        # Extract API credentials safely
        api_key = self.config.get('api', 'binance', 'apiKey')
        api_secret = self.config.get('api', 'binance', 'secret')

        # Basic parameters
        params = {
            'timeout': self.config.get('api', 'timeout', default=30000),
            'enableRateLimit': self.config.get('api', 'enableRateLimit', default=True),
            'options': {
            'adjustForTimeDifference': self.config.get(
            'default_config', 'options', 'adjustForTimeDifference', default=True),
            'recvWindow': self.config.get(
            'default_config', 'options', 'recvWindow', default=60000),
            'defaultType': self.config.get(
            'default_config', 'options', 'defaultType', default='spot')
            },
            'proxies': {
                'http': self.config.get('proxies', 'http'),
                'https': self.config.get('proxies', 'https')                
            }
        }

        # Add API credentials if provided
        if api_key and api_secret:
            params['apiKey'] = api_key
            params['secret'] = api_secret

        return params
    
    def _init_exchange(self) -> None:
        """Initialize the synchronous exchange connection"""
        try:
            # Try with configured parameters
            self.exchange = ccxt.binance(self.params)
            logger.info("CCXT Binance exchange initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {str(e)}")
            
            # Try with minimal parameters as fallback
            try:
                minimal_params = {
                    'timeout': 30000,
                    'enableRateLimit': True
                }
                # Add proxies if configured
                if 'proxies' in self.params:
                    minimal_params['proxies'] = self.params['proxies']
                
                self.exchange = ccxt.binance(minimal_params)
                logger.info("CCXT Binance exchange initialized with minimal parameters")
            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {str(e2)}")
                self.exchange = None
    
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
            proxies = self._nested_get(self.config, ['proxies'])
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
            
            # Process each chunk
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                try:
                    # Handle rate limiting
                    self._handle_rate_limit()
                    
                    # Convert to millisecond timestamp
                    chunk_since = int(chunk_start.timestamp() * 1000)
                    
                    # Log for first chunk or every 5th chunk to reduce log noise
                    if i == 0 or i % 5 == 0:
                        logger.info(f"Fetching chunk {i+1}/{len(chunks)}: {chunk_start} to {chunk_end}")
                    
                    # Fetch the chunk
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        since=chunk_since,
                        limit=self.download_chunk_size
                    )
                    
                    chunks_processed += 1
                    
                    # Check if we got data
                    if not ohlcv or len(ohlcv) == 0:
                        logger.debug(f"No data returned for chunk {i+1}")
                        continue
                    
                    # Add data to accumulated list
                    all_data.extend(ohlcv)
                    chunks_successful += 1
                    
                    # Short delay between chunks, adjusted based on measured latency
                    await asyncio.sleep(delay_between_chunks)
                    
                except Exception as e:
                    logger.error(f"Error fetching chunk {i+1}: {str(e)}")
                    await asyncio.sleep(self.retry_delay)
            
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
        # 修改缓存存储结构
        def get_cache_path(ts: pd.Timestamp) -> str:
            return os.path.join(
                self.cache_dir,
                timeframe,
                symbol.replace('/', '_'),
                f"{ts.year:04d}",
                f"{ts.month:02d}",
                f"{ts.timestamp():.0f}_{(ts + pd.Timedelta(hours=1)).timestamp():.0f}.parquet"
            )

        try:
            # 按小时分片存储
            for ts, group in df.groupby(pd.Grouper(freq='H')):
                if group.empty:
                    continue
                    
                file_path = get_cache_path(ts)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 使用Parquet格式
                group.to_parquet(
                    file_path,
                    engine='pyarrow',
                    compression='zstd',
                    index=True
                )
                
        except Exception as e:
            logger.warning(f"缓存保存失败: {str(e)}")
            
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