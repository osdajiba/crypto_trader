# src/data/data_manager.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from .integrity_checker import DataIntegrityChecker

class DataManager:
    """
    Enhanced Data Manager responsible for fetching, caching, and managing market data
    with improved error handling, data validation, and paginated data handling
    """
    
    def __init__(self, source_type: str = "local", config: Optional[ConfigManager] = None):
        """
        Initialize the Data Manager
        
        Args:
            source_type: Type of data source ("local", "exchange", "hybrid")
            config: Configuration manager instance
        """
        self.config = config
        self.source_type = source_type.lower()
        self.logger = LogManager.get_logger("data.manager")
        
        # Initialize async executor
        self.executor = AsyncExecutor()
        
        # Data sources (to be initialized later)
        self.primary_source = None
        self.backup_source = None
        
        # Data cache
        self.data_cache = {}
        self.use_cache = config.get("data", "cache", "enabled", default=True) if config else True
        
        # Data integrity checker
        self.integrity_checker = None
        self.validate_data = config.get("data", "validation", "enabled", default=True) if config else True
        
        # Pagination settings
        self.page_size = config.get("data", "pagination", "page_size", default=1000) if config else 1000
        self.max_concurrent_pages = config.get("data", "pagination", "max_concurrent", default=5) if config else 5
        
        # Create data sources based on source_type
        self._initialize_data_sources()
        
        self.logger.info(f"DataManager initialized: source={self.source_type}, cache={self.use_cache}, mode={self._get_trading_mode()}")
    
    def _initialize_data_sources(self):
        """Initialize appropriate data sources based on configuration"""
        try:
            # Create integrity checker
            self.integrity_checker = DataIntegrityChecker(
                timeframe=self.config.get("data", "default_timeframe", default="1h") if self.config else "1h"
            )
            
            # Initialize primary source based on source_type
            if self.source_type == "local":
                from src.datasource.sources import LocalSource
                self.primary_source = LocalSource(config=self.config)
                self.logger.info(f"Created local data source")
                
            elif self.source_type == "exchange":
                from src.datasource.sources import ExchangeSource
                self.primary_source = ExchangeSource(config=self.config)
                self.logger.info(f"Created exchange data source")
                
            elif self.source_type == "hybrid":
                # In hybrid mode, primary is exchange and backup is local
                from src.datasource.sources import ExchangeSource
                from src.datasource.sources import LocalSource
                self.primary_source = ExchangeSource(config=self.config)
                self.backup_source = LocalSource(config=self.config)
                self.logger.info(f"Created hybrid data sources (primary: exchange, backup: local)")
                
            else:
                raise ValueError(f"Unsupported data source type: {self.source_type}")
                
            # Create backup source for auto-download of missing data if in backtest mode
            if self._get_trading_mode() == "BACKTEST" and self.source_type == "local" and not self.backup_source:
                from src.datasource.sources import ExchangeSource
                self.backup_source = ExchangeSource(config=self.config)
                self.logger.info(f"Created exchange backup source for auto-downloading missing data")
                
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {e}", exc_info=True)
            raise
    
    def _get_trading_mode(self) -> str:
        """Get current trading mode from config"""
        if not self.config:
            return "backtest"  # Default to backtest
            
        mode = self.config.get("system", "operational_mode", default="backtest")
        return mode.upper()
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: Optional[Union[str, datetime]] = None, 
        end: Optional[Union[str, datetime]] = None,
        use_pagination: bool = True
    ) -> pd.DataFrame:
        """
        Get historical market data with improved pagination and error handling
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Data timeframe/interval (e.g., "1h", "1d")
            start: Start datetime or date string
            end: End datetime or date string
            use_pagination: Whether to use pagination for large data requests
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Check cache first
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        if self.use_cache and cache_key in self.data_cache:
            self.logger.info(f"Historical data loaded from cache: {cache_key}")
            return self.data_cache[cache_key]
        
        try:
            # Start async executor if not running
            if not self.executor.is_running:
                await self.executor.start()
            
            # Convert string dates to datetime objects if needed
            start_dt = self._parse_datetime(start) if start else None
            end_dt = self._parse_datetime(end) if end else None
            
            # Try primary source first
            try:
                if use_pagination and start_dt and end_dt:
                    # Use paginated fetching for large date ranges
                    data = await self._fetch_paginated_data(
                        source=self.primary_source, 
                        symbol=symbol, 
                        timeframe=timeframe, 
                        start_dt=start_dt, 
                        end_dt=end_dt
                    )
                else:
                    # Use standard fetch for smaller requests
                    data = await self.executor.submit(
                        self.primary_source.fetch_historical,
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt
                    )
                
                if not data.empty:
                    # Validate data if enabled
                    if self.validate_data and self.integrity_checker:
                        valid, results = await self.integrity_checker.check(data)
                        if not valid:
                            self.logger.warning(f"Data integrity issues found for {symbol} {timeframe}")
                            data = await self._fix_data_issues(data, results)
                    
                    # Cache the result
                    if self.use_cache:
                        self.data_cache[cache_key] = data
                        self.logger.info(f"Historical data cached: {cache_key}")
                    
                    return data
                else:
                    self.logger.warning(f"No historical data found in local database for {symbol} {timeframe}")
                    self.logger.info(f"Auto try: fetch historical data from exchange ...")
                
            except Exception as e:
                self.logger.warning(f"Primary source failed for {symbol} {timeframe}: {e}")
                
            # If primary source fails or returns empty and we have a backup, try it
            if self.backup_source:
                self.logger.info(f"Trying download source from exchange for {symbol} {timeframe}")
                try:
                    if use_pagination and start_dt and end_dt:
                        # Use paginated fetching for large date ranges with backup source
                        data = await self._fetch_paginated_data(
                            source=self.backup_source, 
                            symbol=symbol, 
                            timeframe=timeframe, 
                            start_dt=start_dt, 
                            end_dt=end_dt
                        )
                    else:
                        # Use standard fetch for smaller requests
                        data = await self.executor.submit(
                            self.backup_source.fetch_historical,
                            symbol=symbol,
                            timeframe=timeframe,
                            start=start_dt,
                            end=end_dt
                        )
                    
                    if not data.empty:
                        # Validate data if enabled
                        if self.validate_data and self.integrity_checker:
                            valid, results = await self.integrity_checker.check(data)
                            if not valid:
                                self.logger.warning(f"Data integrity issues found in backup data for {symbol} {timeframe}")
                                data = await self._fix_data_issues(data, results)
                        
                        # Cache the result
                        if self.use_cache:
                            self.data_cache[cache_key] = data
                            self.logger.info(f"Backup historical data cached: {cache_key}")
                        
                        return data
                
                except Exception as e:
                    self.logger.error(f"Backup source also failed for {symbol} {timeframe}: {e}")
            
            # If we got here, both sources failed or returned empty
            self.logger.error(f"Failed to get historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()

    async def _fetch_paginated_data(
        self, 
        source: Any, 
        symbol: str, 
        timeframe: str, 
        start_dt: datetime, 
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Fetch data in pages and combine, optimized for Parquet files
        
        Args:
            source: Data source to use
            symbol: Symbol to fetch
            timeframe: Timeframe to fetch
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            pd.DataFrame: Combined data from all pages
        """
        # Calculate time ranges for pagination - optimized for timestamp-based files
        time_chunks = self._calculate_time_chunks(timeframe, start_dt, end_dt, self.page_size)
        
        if not time_chunks:
            self.logger.error(f"Failed to calculate time chunks for pagination")
            return pd.DataFrame()
            
        self.logger.info(f"Fetching {symbol} {timeframe} data in {len(time_chunks)} pages")
        
        # Fetch data in chunks with concurrency control
        all_chunks = []
        chunk_sets = [time_chunks[i:i+self.max_concurrent_pages] for i in range(0, len(time_chunks), self.max_concurrent_pages)]
        
        for chunk_set_idx, chunk_set in enumerate(chunk_sets):
            self.logger.info(f"Fetching page set {chunk_set_idx+1}/{len(chunk_sets)} ({len(chunk_set)} pages)")
            
            # Create tasks for each chunk in this set
            tasks = []
            for chunk_start, chunk_end in chunk_set:
                tasks.append(self.executor.submit(
                    source.fetch_historical,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=chunk_start,
                    end=chunk_end
                ))
            
            # Wait for all tasks in this set to complete
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, handling any exceptions
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    self.logger.error(f"Error fetching page {chunk_set_idx * self.max_concurrent_pages + i}: {result}")
                    continue
                    
                if not isinstance(result, pd.DataFrame) or result.empty:
                    self.logger.warning(f"Empty result for page {chunk_set_idx * self.max_concurrent_pages + i}")
                    continue
                    
                all_chunks.append(result)
                
            # Throttle between chunk sets if needed
            if chunk_set_idx < len(chunk_sets) - 1:
                await asyncio.sleep(0.5)  # Small delay between chunk sets
        
        # Combine all chunks
        if not all_chunks:
            self.logger.warning(f"No data returned from any page.")
            return pd.DataFrame()
            
        combined_data = pd.concat(all_chunks, ignore_index=True)
        
        # Ensure datetime column exists and is properly formatted
        if 'timestamp' in combined_data.columns and 'datetime' not in combined_data.columns:
            combined_data['datetime'] = pd.to_datetime(combined_data['timestamp'], unit='ms')
        
        # Sort by datetime and remove duplicates
        if 'datetime' in combined_data.columns:
            # Sort by timestamp
            combined_data = combined_data.sort_values(by='datetime')
            
            # Remove duplicates keeping last occurrence
            before_count = len(combined_data)
            combined_data = combined_data.drop_duplicates(subset=['datetime'], keep='last')
            after_count = len(combined_data)
            
            if before_count > after_count:
                self.logger.info(f"Removed {before_count - after_count} duplicate records")
        
        self.logger.info(f"Successfully fetched and combined {len(combined_data)} records from {len(all_chunks)}/{len(time_chunks)} pages")
        return combined_data

    def _calculate_time_chunks(
        self, 
        timeframe: str, 
        start_dt: datetime, 
        end_dt: datetime, 
        chunk_size: int
    ) -> List[Tuple[datetime, datetime]]:
        """
        Calculate time chunks for paginated fetching
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "1d")
            start_dt: Start datetime
            end_dt: End datetime
            chunk_size: Maximum records per chunk
            
        Returns:
            List[Tuple[datetime, datetime]]: List of (start, end) datetime pairs
        """
        try:
            # Convert timeframe to seconds
            timeframe_seconds = self._timeframe_to_seconds(timeframe)
            
            # Calculate total time range in seconds
            total_seconds = (end_dt - start_dt).total_seconds()
            
            # Calculate estimated number of candles
            estimated_candles = int(total_seconds / timeframe_seconds) + 1
            
            # Calculate number of chunks needed
            num_chunks = (estimated_candles + chunk_size - 1) // chunk_size
            
            # Safety cap on number of chunks
            max_chunks = 100  # Prevent excessive chunking
            if num_chunks > max_chunks:
                self.logger.warning(f"Requested date range would require {num_chunks} chunks, limiting to {max_chunks}")
                num_chunks = max_chunks
            
            # Calculate duration of each chunk
            chunk_seconds = total_seconds / num_chunks
            
            # Create time range chunks
            chunks = []
            for i in range(num_chunks):
                chunk_start = start_dt + timedelta(seconds=i * chunk_seconds)
                
                # For the last chunk, use the exact end time
                if i == num_chunks - 1:
                    chunk_end = end_dt
                else:
                    chunk_end = start_dt + timedelta(seconds=(i + 1) * chunk_seconds)
                
                chunks.append((chunk_start, chunk_end))
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error calculating time chunks: {e}")
            return []
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        Convert timeframe string to seconds
        
        Args:
            timeframe: Timeframe string (e.g., "1h", "1d")
            
        Returns:
            int: Number of seconds
        """
        # Extract number and unit
        import re
        match = re.match(r"(\d+)([a-zA-Z]+)", timeframe)
        if not match:
            self.logger.warning(f"Invalid timeframe format: {timeframe}, using 1h as default")
            return 3600  # Default to 1 hour
            
        quantity = int(match.group(1))
        unit = match.group(2).lower()
        
        # Convert to seconds
        if unit == 'm' or unit == 'min':
            return quantity * 60
        elif unit == 'h' or unit == 'hour':
            return quantity * 3600
        elif unit == 'd' or unit == 'day':
            return quantity * 86400
        elif unit == 'w' or unit == 'week':
            return quantity * 86400 * 7
        else:
            self.logger.warning(f"Unknown time unit: {unit}, assuming minutes")
            return quantity * 60
    
    async def _fix_data_issues(self, df: pd.DataFrame, check_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply automatic fixes to common data issues
        
        Args:
            df: Original DataFrame with issues
            check_results: Results from integrity checker
            
        Returns:
            pd.DataFrame: Fixed DataFrame
        """
        fixed_df = df.copy()
        
        try:
            # Get timestamp column
            timestamp_col = None
            for col in ['datetime', 'timestamp', 'date', 'time']:
                if col in fixed_df.columns:
                    timestamp_col = col
                    break
            
            # Fix: Sort by timestamp
            if timestamp_col and check_results.get("timestamp_sequence", {}).get("passed") is False:
                self.logger.info("Fixing timestamp sequence by sorting")
                fixed_df = fixed_df.sort_values(by=timestamp_col)
            
            # Fix: Remove duplicates
            if timestamp_col and check_results.get("duplicate_timestamps", {}).get("passed") is False:
                self.logger.info("Fixing duplicate timestamps by keeping last occurrence")
                fixed_df = fixed_df.drop_duplicates(subset=[timestamp_col], keep='last')
            
            # Fix: Handle missing values
            if check_results.get("missing_values", {}).get("passed") is False:
                self.logger.info("Fixing missing values using forward fill")
                fixed_df = fixed_df.ffill()
            
            # Fix: OHLC consistency
            if check_results.get("ohlc_consistency", {}).get("passed") is False:
                self.logger.info("Fixing OHLC consistency issues")
                
                # Ensure high is highest
                fixed_df['high'] = fixed_df[['high', 'open', 'close']].max(axis=1)
                
                # Ensure low is lowest
                fixed_df['low'] = fixed_df[['low', 'open', 'close']].min(axis=1)
            
            # Fix: Zero prices
            if check_results.get("zero_values", {}).get("passed") is False:
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in fixed_df.columns]
                for col in price_cols:
                    zero_mask = fixed_df[col] == 0
                    if zero_mask.any():
                        self.logger.info(f"Replacing zero values in {col}")
                        fixed_df.loc[zero_mask, col] = None  # Convert zeros to NaN
                        fixed_df[col] = fixed_df[col].ffill()  # Forward fill NaNs
            
            return fixed_df
            
        except Exception as e:
            self.logger.error(f"Error fixing data issues: {e}")
            return df  # Return original if fixes fail
    
    def _parse_datetime(self, dt_input: Union[str, datetime]) -> datetime:
        """Parse datetime from various input formats"""
        if isinstance(dt_input, datetime):
            return dt_input
            
        try:
            # Try different formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y%m%d", "%Y%m%d%H%M%S"]:
                try:
                    return datetime.strptime(dt_input, fmt)
                except ValueError:
                    continue
                    
            # Try pandas to_datetime as fallback
            return pd.to_datetime(dt_input).to_pydatetime()
            
        except Exception as e:
            self.logger.error(f"Error parsing datetime {dt_input}: {e}")
            raise ValueError(f"Invalid datetime format: {dt_input}")
    
    async def get_latest_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """
        Get the most recent market data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            limit: Maximum number of records to fetch
            
        Returns:
            pd.DataFrame: Latest market data
        """
        # Only relevant for exchange or hybrid sources
        if self.source_type in ["exchange", "hybrid"]:
            try:
                data = await self.executor.submit(
                    self.primary_source.get_latest_data,
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )
                
                if self.validate_data and self.integrity_checker and not data.empty:
                    valid, results = await self.integrity_checker.check(data)
                    if not valid:
                        self.logger.warning(f"Data integrity issues found in latest data for {symbol} {timeframe}")
                        data = await self._fix_data_issues(data, results)
                        
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching latest data: {e}")
                return pd.DataFrame()
        else:
            # For local source, get most recent data from historical
            end = datetime.now()
            start = end - timedelta(days=7)  # Fetch last 7 days
            return await self.get_historical_data(symbol, timeframe, start, end)
    
    async def update_local_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Update local data storage using a consolidated daily file format
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            data: New data to store
            
        Returns:
            bool: Success status
        """
        # Only relevant if we have a local source (primary or backup)
        local_source = None
        if self.source_type == "local":
            local_source = self.primary_source
        elif self.backup_source and hasattr(self.backup_source, "update_data"):
            local_source = self.backup_source
            
        if local_source and hasattr(local_source, "update_data"):
            try:
                # Validate data before storing
                if self.validate_data and self.integrity_checker and not data.empty:
                    valid, results = await self.integrity_checker.check(data)
                    if not valid:
                        self.logger.warning(f"Fixing data integrity issues before updating local storage")
                        data = await self._fix_data_issues(data, results)
                
                # Update local storage
                success = await self.executor.submit(
                    local_source.update_data,
                    symbol=symbol,
                    timeframe=timeframe,
                    data=data
                )
                
                # Update cache if successful
                if success and self.use_cache:
                    # Update any relevant cache entries
                    for key in list(self.data_cache.keys()):
                        if key.startswith(f"{symbol}|{timeframe}"):
                            del self.data_cache[key]  # Invalidate cached data
                    
                    self.logger.info(f"Invalidated cache for {symbol} {timeframe} after update")
                
                return success
                
            except Exception as e:
                self.logger.error(f"Error updating local data: {e}")
                return False
        else:
            self.logger.warning(f"No local source available for updating data")
            return False
    
    async def download_and_store_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: Optional[Union[str, datetime]] = None, 
        end: Optional[Union[str, datetime]] = None
    ) -> bool:
        """
        Download data from exchange and store it locally
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            
        Returns:
            bool: Success status
        """
        if not self.backup_source:
            self.logger.warning(f"No exchange source available for downloading {symbol} {timeframe}")
            return False
            
        try:
            # Parse dates
            start_dt = self._parse_datetime(start) if start else None
            end_dt = self._parse_datetime(end) if end else datetime.now()
            
            # Default start date is 1 year ago if not specified
            if not start_dt:
                start_dt = end_dt - timedelta(days=365)
                
            self.logger.info(f"Downloading data for {symbol} {timeframe} from {start_dt} to {end_dt}")
            
            # Use exchange source to fetch data with pagination
            data = await self._fetch_paginated_data(
                source=self.backup_source, 
                symbol=symbol, 
                timeframe=timeframe, 
                start_dt=start_dt, 
                end_dt=end_dt
            )
            
            if data.empty:
                self.logger.error(f"Failed to download data for {symbol} {timeframe}")
                return False
                
            # Store the data locally
            success = await self.update_local_data(symbol, timeframe, data)
            
            if success:
                self.logger.info(f"Successfully downloaded and stored {len(data)} records for {symbol} {timeframe}")
            else:
                self.logger.error(f"Failed to store downloaded data for {symbol} {timeframe}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error downloading and storing data: {e}")
            return False
    
    async def clear_cache(self, symbol: Optional[str] = None, timeframe: Optional[str] = None) -> None:
        """
        Clear the data cache for specific symbol/timeframe or all
        
        Args:
            symbol: Symbol to clear (None for all)
            timeframe: Timeframe to clear (None for all)
        """
        if not self.use_cache:
            return
            
        if symbol and timeframe:
            # Clear specific symbol and timeframe
            prefix = f"{symbol}|{timeframe}"
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for {symbol} {timeframe} ({len(keys_to_remove)} entries)")
            
        elif symbol:
            # Clear all timeframes for a symbol
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(f"{symbol}|")]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for {symbol} ({len(keys_to_remove)} entries)")
            
        elif timeframe:
            # Clear all symbols for a timeframe
            keys_to_remove = [k for k in self.data_cache.keys() if f"|{timeframe}|" in k]
            for key in keys_to_remove:
                del self.data_cache[key]
            self.logger.info(f"Cleared cache for timeframe {timeframe} ({len(keys_to_remove)} entries)")
            
        else:
            # Clear all cache
            count = len(self.data_cache)
            self.data_cache.clear()
            self.logger.info(f"Cleared entire data cache ({count} entries)")
    
    async def preload_data(self, symbols: List[str], timeframe: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Preload data for multiple symbols to improve performance
        
        Args:
            symbols: List of symbols to preload
            timeframe: Timeframe to preload
            days: Number of days to preload (from now)
            
        Returns:
            Dict[str, pd.DataFrame]: Preloaded data keyed by symbol
        """
        if not symbols:
            return {}
            
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Create tasks to load data for each symbol
        results = {}
        for symbol in symbols:
            try:
                data = await self.get_historical_data(symbol, timeframe, start, end)
                if not data.empty:
                    results[symbol] = data
                    self.logger.debug(f"Preloaded {len(data)} records for {symbol} {timeframe}")
                else:
                    self.logger.warning(f"No data available when preloading {symbol} {timeframe}")
            except Exception as e:
                self.logger.error(f"Error preloading data for {symbol}: {e}")
                
        self.logger.info(f"Preloaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def close(self) -> None:
        """Clean up resources"""
        # Close data sources
        if hasattr(self.primary_source, 'close'):
            try:
                await self.executor.submit(self.primary_source.close)
                self.logger.debug("Closed primary data source")
            except Exception as e:
                self.logger.error(f"Error closing primary data source: {e}")
                
        if self.backup_source and hasattr(self.backup_source, 'close'):
            try:
                await self.executor.submit(self.backup_source.close)
                self.logger.debug("Closed backup data source")
            except Exception as e:
                self.logger.error(f"Error closing backup data source: {e}")
                
        # Clear cache
        self.data_cache.clear()