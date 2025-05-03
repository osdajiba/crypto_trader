#!/usr/bin/env python3
# src/datasource/sources/exchange.py

"""
Exchange data source implementation.
Provides access to real-time and historical market data from exchanges.
"""

from typing import Dict, Optional, Union, Any, List
import pandas as pd
import os
import asyncio
import time
from datetime import datetime, timezone, timedelta

from src.common.abstract_factory import register_factory_class
from src.common.helpers import TimeUtils, FileUtils
from src.common.config_manager import ConfigManager
from src.datasource.sources.base import BaseDataSource, DataSourceConnectionError, DataSourceFetchError
from src.exchange.factory import get_exchange_factory


@register_factory_class('datasource_factory', 'exchange', 
    description="Direct exchange connection data source",
    features=["historical", "realtime", "online_access"],
    category="datasource")
class ExchangeDataSource(BaseDataSource):
    """
    Exchange data source for real-time and historical data
    
    This data source connects directly to cryptocurrency exchanges to fetch
    real-time and historical market data. It supports rate limiting,
    connection management, and optional local caching.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize exchange data source
        
        Args:
            config: Configuration manager
            params: Additional parameters
        """
        super().__init__(config, params)
        
        self.exchange = None
        self.cache_dir = None
        self.data_path = None
        
        # Rate limit settings
        self.max_requests_per_minute = 0
        self.request_delay = 0
        self.last_request_time = 0
        
        # Retry settings
        self.max_retries = 0
        self.retry_delay = 0
        
        # Cache settings
        self.cache_enabled = True
    
    async def _initialize_source_specific(self) -> None:
        """Initialize exchange connection and settings"""
        # Initialize exchange connection
        await self._init_exchange()
        
        # Setup cache directory
        self.cache_dir = self._setup_cache_dir()
        
        # Configure rate limiting
        self.max_requests_per_minute = self.config.get("api", "rate_limits", "requests_per_minute", default=20)
        self.request_delay = 60.0 / self.max_requests_per_minute
        
        # Configure retry settings
        self.max_retries = self.config.get("api", "retries", "max_attempts", default=3)
        self.retry_delay = self.config.get("api", "retries", "delay_seconds", default=1)
        
        # Setup data storage
        self.data_path = self._get_data_path()
        os.makedirs(self.data_path, exist_ok=True)
        
        # Configure caching behavior
        self.cache_enabled = self.config.get("data", "storage", "cache_enabled", default=True)
        
        self.logger.info(f"ExchangeDataSource initialized, max request rate: {self.max_requests_per_minute}/minute")
    
    def _setup_cache_dir(self) -> str:
        """
        Set up and return cache directory
        
        Returns:
            str: Path to cache directory
        """
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_data_path(self) -> str:
        """
        Get data storage path from config
        
        Returns:
            str: Path to data storage directory
        """
        paths_to_check = [
            ('data', 'paths', 'historical_data_path'),
            ('data', 'storage', 'historical'),
            ('data_paths', 'historical_data')
        ]
        
        for path_keys in paths_to_check:
            path = self.config.get(*path_keys, default=None)
            if path:
                return os.path.abspath(os.path.expanduser(path))
        
        return os.path.abspath('data/historical')
        
    async def _init_exchange(self) -> None:
        """
        Initialize the exchange connection
        
        Raises:
            DataSourceConnectionError: If exchange connection fails
        """
        try:
            exchange_name = self.config.get("exchange", "name", default="binance")
            
            exchange_factory = get_exchange_factory(self.config)
            self.exchange = await exchange_factory.create(exchange_name)
            
            self.logger.info(f"Exchange connection initialized: {exchange_name}")
        except Exception as e:
            error_msg = f"Failed to initialize exchange: {str(e)}"
            self.logger.error(error_msg)
            raise DataSourceConnectionError(error_msg)
    
    def _generate_storage_path(self, symbol: str, timeframe: str, 
                             start_dt: datetime, end_dt: datetime) -> str:
        """
        Generate standardized storage path for data caching
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            str: Directory path for data storage
        """
        symbol_safe = symbol.replace('/', '_')
        source_name = self.config.get("exchange", "name", default="binance")
        
        year = start_dt.year
        month = f"{start_dt.month:02d}"
        
        dir_path = os.path.join(
            self.data_path,
            source_name,
            symbol_safe,
            timeframe,
            str(year),
            month
        )
        
        os.makedirs(dir_path, exist_ok=True)
        return dir_path
    
    async def _apply_rate_limit(self) -> None:
        """Apply rate limiting to prevent API request throttling"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.request_delay:
            delay = self.request_delay - time_since_last_request
            if delay > 0:
                await asyncio.sleep(delay)
                
        self.last_request_time = time.time()
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                             start: Optional[Union[str, datetime]] = None, 
                             end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical data from exchange with caching
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start: Start datetime or string
            end: End datetime or string
            
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails
        """
        if not self._is_initialized:
            await self.initialize()
            
        self.logger.info(f"Fetching historical data for: {symbol} {timeframe}")
        
        try:
            # Parse date parameters
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now(timezone.utc)
            start_dt = TimeUtils.parse_timestamp(start) if start else end_dt - timedelta(days=30)
            
            # Check if data is available in cache
            if self.cache_enabled:
                cached_data = await self._try_load_from_cache(symbol, timeframe, start_dt, end_dt)
                if cached_data is not None and not cached_data.empty:
                    self.logger.info(f"Using cached data for {symbol} {timeframe}")
                    return cached_data
            
            # Get optimal data ranges to respect exchange limits
            data_ranges = self.get_optimal_data_ranges(start_dt, end_dt, timeframe)
            
            all_data = []
            
            # Fetch data for each range
            for range_start, range_end in data_ranges:
                for attempt in range(self.max_retries):
                    try:
                        # Apply rate limiting
                        await self._apply_rate_limit()
                        
                        # Fetch from exchange
                        self.logger.debug(f"Fetching {symbol} {timeframe} from {range_start} to {range_end}")
                        
                        # Convert dates to millisecond timestamps for exchange API
                        start_ms = int(range_start.timestamp() * 1000)
                        end_ms = int(range_end.timestamp() * 1000)
                        
                        # Call exchange API
                        ohlcv_data = await self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=start_ms,
                            limit=1000  # Typical exchange limit
                        )
                        
                        # Convert to DataFrame
                        if ohlcv_data and len(ohlcv_data) > 0:
                            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            
                            # Convert timestamp to datetime
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                            
                            # Add symbol column
                            df['symbol'] = symbol
                            
                            all_data.append(df)
                            break  # Success, exit retry loop
                            
                    except Exception as e:
                        self.logger.warning(f"Fetch attempt {attempt+1} failed: {str(e)}")
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                        else:
                            self.logger.error(f"All {self.max_retries} attempts failed for {symbol} {timeframe}")
                            # Continue with other ranges instead of failing completely
            
            # Combine all data
            if not all_data:
                self.logger.warning(f"No data fetched for {symbol} {timeframe}")
                return pd.DataFrame()
                
            result_df = pd.concat(all_data, ignore_index=True)
            
            # Remove duplicates and sort
            if not result_df.empty:
                result_df = result_df.drop_duplicates(subset='datetime').sort_values('datetime').reset_index(drop=True)
                
                # Validate the data
                self.validate_ohlcv_data(result_df)
                
                # Cache result if enabled
                if self.cache_enabled:
                    await self._cache_data(result_df, symbol, timeframe, start_dt, end_dt)
                
                self.logger.info(f"Fetched {len(result_df)} rows for {symbol} {timeframe}")
                
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {str(e)}")
            raise DataSourceFetchError(f"Exchange data fetch failed for {symbol} {timeframe}: {str(e)}")
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Fetch real-time market data from exchange
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            
        Returns:
            pd.DataFrame: DataFrame with latest OHLCV data
            
        Raises:
            DataSourceFetchError: If data fetching fails
        """
        if not self._is_initialized:
            await self.initialize()
            
        self.logger.info(f"Fetching real-time data for: {symbol} {timeframe}")
        
        try:
            # Apply rate limiting
            await self._apply_rate_limit()
            
            # Fetch latest candle from exchange
            ticker = await self.exchange.fetch_ticker(symbol)
            
            if not ticker:
                self.logger.warning(f"No ticker data available for {symbol}")
                return pd.DataFrame()
            
            # Create OHLCV data from ticker
            now = datetime.now(timezone.utc)
            
            df = pd.DataFrame([{
                'datetime': now,
                'timestamp': int(now.timestamp() * 1000),
                'open': ticker.get('open', ticker.get('last', 0)),
                'high': ticker.get('high', ticker.get('last', 0)),
                'low': ticker.get('low', ticker.get('last', 0)),
                'close': ticker.get('last', 0),
                'volume': ticker.get('volume', 0),
                'symbol': symbol
            }])
            
            self.logger.debug(f"Real-time data fetched for {symbol}: {ticker.get('last', 0)}")
            
            # If enabled, add to cache
            if self.cache_enabled:
                await self._append_to_cache(df, symbol, timeframe)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch real-time data: {str(e)}")
            raise DataSourceFetchError(f"Exchange real-time fetch failed for {symbol}: {str(e)}")
    
    async def _try_load_from_cache(self, symbol: str, timeframe: str, 
                                 start_dt: datetime, end_dt: datetime) -> Optional[pd.DataFrame]:
        """
        Try to load data from cache
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            Optional[pd.DataFrame]: Cached data or None if not available
        """
        try:
            symbol_safe = symbol.replace('/', '_')
            source_name = self.config.get("exchange", "name", default="binance")
            
            # Generate possible cache file paths
            cache_paths = []
            
            # Determine date range
            current_year = start_dt.year
            current_month = start_dt.month
            end_year = end_dt.year
            end_month = end_dt.month
            
            # Generate paths for all months in range
            while current_year < end_year or (current_year == end_year and current_month <= end_month):
                dir_path = os.path.join(
                    self.data_path,
                    source_name,
                    symbol_safe,
                    timeframe,
                    str(current_year),
                    f"{current_month:02d}"
                )
                
                if os.path.exists(dir_path):
                    for filename in os.listdir(dir_path):
                        if filename.endswith('.parquet'):
                            cache_paths.append(os.path.join(dir_path, filename))
                
                # Move to next month
                current_month += 1
                if current_month > 12:
                    current_month = 1
                    current_year += 1
            
            if not cache_paths:
                return None
                
            # Load and filter data from cache files
            all_data = []
            for path in cache_paths:
                try:
                    df = pd.read_parquet(path)
                    all_data.append(df)
                except Exception as e:
                    self.logger.warning(f"Failed to read cache file {path}: {str(e)}")
            
            if not all_data:
                return None
                
            # Combine data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Filter by date range
            combined_df = combined_df[(combined_df['datetime'] >= start_dt) & 
                                    (combined_df['datetime'] <= end_dt)]
            
            # Sort by datetime
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            
            if combined_df.empty:
                return None
                
            self.logger.info(f"Loaded {len(combined_df)} rows from cache for {symbol} {timeframe}")
            return combined_df
            
        except Exception as e:
            self.logger.warning(f"Cache loading failed: {str(e)}")
            return None
    
    async def _cache_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                        start_dt: datetime, end_dt: datetime) -> bool:
        """
        Cache data to local storage
        
        Args:
            df: DataFrame to cache
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            bool: Success status
        """
        if df.empty:
            return False
            
        try:
            # Generate storage path
            dir_path = self._generate_storage_path(symbol, timeframe, start_dt, end_dt)
            
            # Generate filename with ISO timestamp range
            start_iso = start_dt.astimezone(timezone.utc).isoformat()
            end_iso = end_dt.astimezone(timezone.utc).isoformat()
            
            filename = f"{start_iso}to{end_iso}.parquet"
            file_path = os.path.join(dir_path, filename)
            
            # Save to parquet file
            df.to_parquet(file_path, index=False)
            
            self.logger.info(f"Cached {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache data: {str(e)}")
            return False
    
    async def _append_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Append real-time data to cache
        
        Args:
            df: DataFrame to append
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            
        Returns:
            bool: Success status
        """
        if df.empty:
            return False
            
        try:
            # Get current date for path
            now = datetime.now(timezone.utc)
            
            # Generate storage path
            dir_path = self._generate_storage_path(symbol, timeframe, now, now)
            
            # Use daily cache file for real-time data
            date_str = now.strftime('%Y-%m-%d')
            file_path = os.path.join(dir_path, f"realtime_{date_str}.parquet")
            
            # Append to existing file or create new one
            if os.path.exists(file_path):
                existing_df = pd.read_parquet(file_path)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=['datetime']).sort_values('datetime')
                combined_df.to_parquet(file_path, index=False)
            else:
                df.to_parquet(file_path, index=False)
            
            self.logger.debug(f"Appended real-time data to {file_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"Failed to append to cache: {str(e)}")
            return False
    
    async def _shutdown_source_specific(self) -> None:
        """Close exchange connection and clean up resources"""
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close()
                self.logger.info("Exchange connection closed")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {str(e)}")