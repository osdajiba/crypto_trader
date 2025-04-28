# src/datasource/manager.py

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import TimeUtils
from src.datasource.sources import DataSource, DataSourceFactory
from src.datasource.integrity import DataIntegrityChecker
from src.datasource.downloader import DataDownloader
from src.datasource.processor import DataProcessor


class DataManager:
    """
    Unified Data Manager responsible for fetching, caching, and managing market data
    across all trading modes (backtest, paper, live)
    """
    
    def __init__(self, config: ConfigManager, source_type: str = "hybrid"):
        """
        Initialize the Data Manager
        
        Args:
            config: Configuration manager
            source_type: Type of data source ("local", "exchange", "hybrid")
        """
        self.config = config
        self.source_type = source_type.lower()
        self.logger = LogManager.get_logger("data.manager")
        
        self.mode = self.config.get("system", "operational_mode", default="backtest").upper()
        self._init_data_sources()    # Create appropriate data sources
        
        # Data cache settings
        self.use_cache = config.get("data", "cache", "enabled", default=True)
        self.cache_ttl_seconds = config.get("data", "cache", "ttl_seconds", default=3600)
        self.cache_max_size = config.get("data", "cache", "max_size", default=100)
        self.data_cache = {}
        self.cache_last_accessed = {}
        
        # Data integrity checker with settings
        self.integrity_checker = DataIntegrityChecker(
            timeframe=config.get("data", "default_timeframe", default="1m"),
            parallel=config.get("data", "validation", "parallel", default=True)
        )
        self.validate_data = config.get("data", "validation", "enabled", default=True)
        
        # Thread pool for CPU-bound operations
        max_workers = config.get("system", "max_workers", default=min(32, (os.cpu_count() or 4) + 4))
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)    
        self.batch_size = config.get("data", "batch_size", default=10000)    # Batch processing settings

        self.downloader = self._create_downloader()    # Downloader for fetching new data when needed
        
        self.processor = DataProcessor()    # Processor for process data to specific format
        
        self.logger.info(f"DataManager initialized: source={self.source_type}, mode={self.mode}, cache={'enabled' if self.use_cache else 'disabled'}")
    
    def _init_data_sources(self) -> None:
        """Initialize appropriate data sources based on mode and source_type"""
        try:
            factory = DataSourceFactory()
            
            # For backtest mode
            if self.mode == "BACKTEST":
                if self.source_type == "hybrid":
                    self.primary_source = factory.create_source("local", self.config)
                    self.backup_source = factory.create_source("exchange", self.config)
                else:
                    self.primary_source = factory.create_source(self.source_type, self.config)
                    self.backup_source = None
            
            # For live and paper trading
            elif self.mode in ["LIVE", "PAPER"]:
                if self.source_type == "local":
                    self.logger.warning(f"Local source not ideal for {self.mode} mode, using hybrid source")
                    self.primary_source = factory.create_source("hybrid", self.config)
                else:
                    self.primary_source = factory.create_source(self.source_type, self.config)
                
                # Always have a local backup
                if not isinstance(self.primary_source, DataSource):
                    self.backup_source = factory.create_source("local", self.config)
                else:
                    self.backup_source = None
            
            # Unknown mode
            else:
                self.logger.warning(f"Unknown trading mode: {self.mode}, defaulting to local source")
                self.primary_source = factory.create_source("local", self.config)
                self.backup_source = None
                
        except Exception as e:
            self.logger.error(f"Error initializing data sources: {str(e)}")
            raise
    
    def _create_downloader(self) -> DataDownloader:
        """Create and initialize data downloader"""
        api_key = self.config.get("api", "binance", "api_key", default=None)
        api_secret = self.config.get("api", "binance", "secret", default=None)
        
        # Get base URLs from config
        base_url = self.config.get("api", "base_url", default="https://api.binance.com")
        futures_url = self.config.get("api", "binance", "futures_url", 
                                     default="https://fapi.binance.com")
        
        # Rate limit settings
        rate_limit = self.config.get("api", "rate_limits", "requests", default=10)
        
        # Create downloader with our data path
        data_path = os.path.abspath('data/historical')
        
        # Check if using futures
        use_futures = self.config.get("api", "binance", "options", "default_type", default="spot") == "future"
        
        downloader = DataDownloader(
            base_url=base_url,
            futures_url=futures_url,
            api_key=api_key,
            api_secret=api_secret,
            rate_limit=rate_limit,
            storage_path=data_path,
            use_futures=use_futures
        )
        
        return downloader
        
    async def get_historical_data(self, 
                                symbol: str, 
                                timeframe: str, 
                                start: Optional[Union[str, datetime]] = None, 
                                end: Optional[Union[str, datetime]] = None,
                                use_cache: bool = True) -> pd.DataFrame:
        """
        Get historical market data with caching and validation
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Data timeframe/interval (e.g., "1h", "1d")
            start: Start datetime or date string
            end: End datetime or date string
            use_cache: Whether to use caching
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Check cache first if enabled
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        if use_cache and self.use_cache and cache_key in self.data_cache:
            now = datetime.now().timestamp()
            last_accessed = self.cache_last_accessed.get(cache_key, 0)
            
            if now - last_accessed < self.cache_ttl_seconds:
                self.cache_last_accessed[cache_key] = now
                self.logger.info(f"Historical data loaded from cache: {cache_key}")
                return self.data_cache[cache_key]
            else:
                # Cache expired, remove it
                del self.data_cache[cache_key]
                del self.cache_last_accessed[cache_key]
        
        try:
            # Fetch data from primary source
            data = await self._fetch_and_validate_data(
                source=self.primary_source,
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                source_name="primary"
            )
            
            if not data.empty:
                # Cache result if enabled
                if self.use_cache and use_cache:
                    self._update_cache(cache_key, data)
                return data
            
            self.logger.warning(f"No historical data found in primary source for {symbol} {timeframe}")
            
            # Try backup source if available
            if self.backup_source:
                self.logger.info(f"Trying backup source for {symbol} {timeframe}")
                data = await self._fetch_and_validate_data(
                    source=self.backup_source,
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    source_name="backup"
                )
                
                if not data.empty:
                    # Cache result if enabled
                    if self.use_cache and use_cache:
                        self._update_cache(cache_key, data)
                    return data
                
                self.logger.error(f"Backup source also failed for {symbol} {timeframe}")
            
            # If all sources failed, try downloading the data directly
            self.logger.info(f"All sources failed for {symbol} {timeframe}, trying direct download...")
            data = await self._download_and_save_data(symbol, timeframe, start, end)
            
            if not data.empty:
                if self.use_cache and use_cache:
                    self._update_cache(cache_key, data)
                return data
            
            # All attempts failed
            self.logger.error(f"Failed to get historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def _download_and_save_data(self, 
                                    symbol: str, 
                                    timeframe: str, 
                                    start: Optional[Union[str, datetime]], 
                                    end: Optional[Union[str, datetime]]) -> pd.DataFrame:
        """Download data directly using downloader and save to storage"""
        try:
            await self.downloader.initialize()
            
            df = await self.downloader.download_data(symbol, timeframe, start, end)
            
            if df.empty:
                self.logger.warning(f"Download returned no data for {symbol} {timeframe}")
                return df
                
            # Parse dates for storage
            start_dt = TimeUtils.parse_timestamp(start) if start else (df['datetime'].min() if not df.empty else None)
            end_dt = TimeUtils.parse_timestamp(end) if end else (df['datetime'].max() if not df.empty else None)
                
            # Save the data
            if start_dt and end_dt:
                file_path = await self.downloader.save_data(df, symbol, timeframe, start_dt, end_dt)
                if file_path:
                    self.logger.info(f"Downloaded and saved {len(df)} records to {file_path}")
            
            # Validate and fix if needed
            if self.validate_data and not df.empty:
                valid, results = await self.integrity_checker.check(df, symbol, timeframe)
                if not valid:
                    self.logger.warning(f"Data integrity issues found in downloaded data for {symbol} {timeframe}")
                    df = await self._fix_data_issues(df, results)
                    
            return df
                
        except Exception as e:
            self.logger.error(f"Error downloading data: {e}")
            return pd.DataFrame()
        finally:
            await self.downloader.close()
            
    async def _fetch_and_validate_data(self,
                                      source: Any,
                                      symbol: str,
                                      timeframe: str,
                                      start: Optional[Union[str, datetime]],
                                      end: Optional[Union[str, datetime]],
                                      source_name: str = "unknown") -> pd.DataFrame:
        """
        Fetch data from a source and validate/fix it if needed
        
        Args:
            source: Data source to fetch from
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start: Start datetime or date string
            end: End datetime or date string
            source_name: Name of source for logging
            
        Returns:
            pd.DataFrame: Validated data
        """
        try:
            data = await source.fetch_historical(
                symbol=symbol,
                timeframe=timeframe,
                start=start,
                end=end
            )
            
            if not data.empty and self.validate_data:
                # Validate data in smaller batches for large datasets to reduce memory pressure
                if len(data) > self.batch_size:
                    data = await self._validate_large_dataframe(data, symbol, timeframe)
                else:
                    valid, results = await self.integrity_checker.check(data, symbol, timeframe)
                    if not valid:
                        self.logger.warning(f"Data integrity issues found in {source_name} data for {symbol} {timeframe}")
                        data = await self._fix_data_issues(data, results)
            return data
            
        except Exception as e:
            self.logger.warning(f"{source_name.capitalize()} source failed for {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    async def _validate_large_dataframe(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Validate large dataframes in chunks to reduce memory pressure
        
        Args:
            df: Large DataFrame to validate
            symbol: Symbol for logging
            timeframe: Timeframe for logging
            
        Returns:
            pd.DataFrame: Validated DataFrame
        """
        self.logger.info(f"Processing large dataset ({len(df)} rows) for {symbol} {timeframe} in chunks")
        
        # Process in chunks
        chunks = [df.iloc[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        fixed_chunks = []
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {symbol} {timeframe}")
            valid, results = await self.integrity_checker.check(chunk, symbol, timeframe)
            
            if not valid:
                self.logger.warning(f"Data integrity issues found in chunk {i+1}/{len(chunks)} for {symbol} {timeframe}")
                fixed_chunk = await self._fix_data_issues(chunk, results)
                fixed_chunks.append(fixed_chunk)
            else:
                fixed_chunks.append(chunk)
        
        # Combine chunks
        combined_df = pd.concat(fixed_chunks, ignore_index=False)
        
        # Final check for consistency between chunks
        if len(chunks) > 1:
            # Find timestamp column
            timestamp_col = None
            for col in ['datetime', 'timestamp', 'date', 'time']:
                if col in combined_df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                # Ensure sorted by timestamp
                combined_df = combined_df.sort_values(by=timestamp_col)
                
                # Remove any duplicates that might have been at chunk boundaries
                combined_df = combined_df.drop_duplicates(subset=[timestamp_col], keep='last')
        
        return combined_df
    
    async def _fix_data_issues(self, df: pd.DataFrame, check_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply automated fixes to common data issues
        
        Args:
            df: Original DataFrame with issues
            check_results: Results from integrity checker
            
        Returns:
            pd.DataFrame: Fixed DataFrame
        """
        fixed_df = df.copy()
        fixes_applied = []
        
        try:
            # Get timestamp column
            timestamp_col = None
            for col in ['datetime', 'timestamp', 'date', 'time']:
                if col in fixed_df.columns:
                    timestamp_col = col
                    break
            
            # Fix: Sort by timestamp sequence
            if check_results.get("timestamp_sequence", {}).get("passed") is False:
                fixed_df = fixed_df.sort_values(by=timestamp_col)
                fixes_applied.append("Sorted timestamps in ascending order")
            
            # Fix: Remove duplicate timestamps
            if check_results.get("duplicate_timestamps", {}).get("passed") is False:
                duplicate_count = check_results.get("duplicate_timestamps", {}).get("duplicate_count", 0)
                fixed_df = fixed_df.drop_duplicates(subset=[timestamp_col], keep='last')
                fixes_applied.append(f"Removed {duplicate_count} duplicate timestamps")
            
            # Fix: Missing values - use vectorized operations
            if check_results.get("missing_values", {}).get("passed") is False:
                missing_counts = check_results.get("missing_values", {}).get("missing_counts", {})
                total_missing = sum(missing_counts.values())
                
                # Process only columns with missing values
                for col, count in missing_counts.items():
                    if count > 0:
                        # Forward fill each column individually 
                        fixed_df[col] = fixed_df[col].ffill()
                        
                        # Check if still has NaNs at the beginning
                        if fixed_df[col].isna().any():
                            fixed_df[col] = fixed_df[col].bfill()
                
                fixes_applied.append(f"Filled {total_missing} missing values")
            
            # Fix: OHLC consistency using vectorized operations
            consistency_key = "ohlc_consistency" if "ohlc_consistency" in check_results else "consistency"
            if check_results.get(consistency_key, {}).get("passed") is False:
                # Use numpy for vectorized performance
                open_array = fixed_df['open'].to_numpy()
                close_array = fixed_df['close'].to_numpy()
                
                # Ensure high is the highest value
                high_array = np.maximum.reduce([
                    fixed_df['high'].to_numpy(), 
                    open_array, 
                    close_array
                ])
                fixed_df['high'] = high_array
                
                # Ensure low is the lowest value
                low_array = np.minimum.reduce([
                    fixed_df['low'].to_numpy(), 
                    open_array, 
                    close_array
                ])
                fixed_df['low'] = low_array
                
                fixes_applied.append("Corrected OHLC inconsistencies")
            
            # Fix: Zero price values
            if check_results.get("zero_values", {}).get("passed") is False:
                zero_counts = check_results.get("zero_values", {}).get("zero_counts", {})
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in fixed_df.columns]
                
                for col in price_cols:
                    if col in zero_counts:
                        zero_mask = fixed_df[col] == 0
                        zeros_in_col = zero_mask.sum()
                        
                        if zeros_in_col > 0:
                            fixed_df.loc[zero_mask, col] = None  # Convert zeros to NaN
                            fixed_df[col] = fixed_df[col].ffill().bfill()  # Fill NaNs
                
                fixes_applied.append("Replaced zero values")
            
            # Fix: Price anomalies (extreme values)
            if check_results.get("price_anomalies", {}).get("passed") is False:
                anomalies = check_results.get("price_anomalies", {}).get("anomalies", {})
                
                for col, info in anomalies.items():
                    bounds = info.get("bounds", (None, None))
                    if bounds[0] is not None and bounds[1] is not None:
                        # Identify outliers with vectorized operations
                        outliers_mask = (fixed_df[col] < bounds[0]) | (fixed_df[col] > bounds[1])
                        outlier_count = outliers_mask.sum()
                        
                        if outlier_count > 0:
                            # Convert outliers to NaN
                            fixed_df.loc[outliers_mask, col] = None
                            # Interpolate NaNs
                            fixed_df[col] = fixed_df[col].interpolate(method='linear').ffill().bfill()
                
                fixes_applied.append("Corrected price anomalies")
            
            # Log summary of all fixes
            if fixes_applied:
                fixes_summary = ", ".join(fixes_applied)
                self.logger.info(f"Data fixes summary for {timestamp_col}: {fixes_summary}")
            
            return fixed_df
            
        except Exception as e:
            self.logger.error(f"Error fixing data issues: {e}", exc_info=True)
            return df  # If fixing fails, return original data
    
    def _update_cache(self, key: str, data: pd.DataFrame) -> None:
        """
        Update cache with size limit enforcement
        
        Args:
            key: Cache key
            data: DataFrame to cache
        """
        now = datetime.now().timestamp()
        
        # Check if cache is at capacity
        if len(self.data_cache) >= self.cache_max_size:
            # Find oldest accessed item
            oldest_key = min(self.cache_last_accessed.items(), key=lambda x: x[1])[0]
            del self.data_cache[oldest_key]
            del self.cache_last_accessed[oldest_key]
        
        # Add to cache
        self.data_cache[key] = data
        self.cache_last_accessed[key] = now
        self.logger.info(f"Added to cache: {key} ({len(data)} rows)")
    
    async def get_real_time_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get the latest market data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            pd.DataFrame: Latest market data
        """
        # For live and paper trading, get real-time data
        if self.mode in ["LIVE", "PAPER"]:
            try:
                data = await self.primary_source.fetch_realtime(symbol, timeframe)
                
                if self.validate_data and not data.empty:
                    valid, results = await self.integrity_checker.check(data, symbol, timeframe)
                    if not valid:
                        self.logger.warning(f"Data integrity issues found in latest data for {symbol} {timeframe}")
                        data = await self._fix_data_issues(data, results)
                        
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching latest data: {e}")
                return pd.DataFrame()
                
        # For backtest mode, get latest historical data
        else:
            end = datetime.now()
            start = end - timedelta(hours=1)  # Get last hour
            return await self.get_historical_data(symbol, timeframe, start, end)
            
    async def get_orderbook(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get latest orderbook data (for live trading modes)
        
        Args:
            symbol: Trading pair symbol
            limit: Depth limit
            
        Returns:
            Dict: Orderbook data with bids and asks
        """
        if self.mode not in ["LIVE", "PAPER"]:
            self.logger.warning(f"Orderbook data not available in {self.mode} mode")
            return {"bids": [], "asks": []}
            
        try:
            if hasattr(self.primary_source, 'exchange') and self.primary_source.exchange:
                # Use AsyncExecutor to run synchronous method if needed
                orderbook = await self.primary_source.executor.submit(
                    self.primary_source.exchange.fetch_order_book,
                    symbol=symbol, limit=limit
                )
                return orderbook
            else:
                self.logger.error("Exchange not available to fetch orderbook")
                return {"bids": [], "asks": []}
                
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            return {"bids": [], "asks": []}
            
    async def fetch_all_data_for_symbols(self, symbols: List[str], timeframe: str,
                                       start: Optional[Union[str, datetime]] = None,
                                       end: Optional[Union[str, datetime]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Data timeframe
            start: Start datetime
            end: End datetime
            
        Returns:
            Dict: Symbol -> DataFrame mapping
        """
        # Create tasks to load data for each symbol concurrently
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_historical_data(symbol, timeframe, start, end))
        
        # Run all tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        results = {}
        for i, result in enumerate(results_list):
            symbol = symbols[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Error loading data for {symbol}: {result}")
            elif not isinstance(result, pd.DataFrame) or result.empty:
                self.logger.warning(f"No data available for {symbol} {timeframe}")
            else:
                results[symbol] = result
        
        return results
    
    async def preload_data(self, symbols: List[str], timeframe: str, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Preload data for multiple symbols to improve performance
        
        Args:
            symbols: List of symbols to preload
            timeframe: Timeframe to preload
            days: Number of days to preload
            
        Returns:
            Dict[str, pd.DataFrame]: Preloaded data keyed by symbol
        """
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days)
        
        # Fetch all data
        results = await self.fetch_all_data_for_symbols(symbols, timeframe, start, end)
        
        self.logger.info(f"Preloaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def download_historical_data(self, symbol: str, timeframe: str,
                                     start: Optional[Union[str, datetime]] = None,
                                     end: Optional[Union[str, datetime]] = None) -> bool:
        """
        Download historical data from exchange and store it locally
        
        Args:
            symbol: Symbol to download
            timeframe: Timeframe to download
            start: Start datetime
            end: End datetime
            
        Returns:
            bool: Success status
        """
        try:
            # Initialize downloader
            await self.downloader.initialize()
            
            # Download data
            success = await self.downloader.download_and_save(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start,
                end_date=end
            )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error downloading historical data: {e}")
            return False
        finally:
            await self.downloader.close()
    
    async def clear_cache(self) -> None:
        """Clear the data cache"""
        count = len(self.data_cache)
        self.data_cache.clear()
        self.cache_last_accessed.clear()
        self.logger.info(f"Cleared data cache ({count} entries)")
    
    async def ensure_data_available(self, symbol: str, timeframe: str,
                                  start: Optional[Union[str, datetime]] = None,
                                  end: Optional[Union[str, datetime]] = None) -> bool:
        """
        Ensure data is available locally, downloading if needed
        
        Args:
            symbol: Symbol to check
            timeframe: Timeframe to check
            start: Start datetime
            end: End datetime
            
        Returns:
            bool: True if data is available
        """
        # Try to get from local source first
        data = await self.get_historical_data(symbol, timeframe, start, end)
        
        if not data.empty:
            return True
            
        # If no data, try to download
        self.logger.info(f"No data available for {symbol} {timeframe}, attempting to download...")
        success = await self.download_historical_data(symbol, timeframe, start, end)
        
        if success:
            # Verify download worked
            data = await self.get_historical_data(symbol, timeframe, start, end, use_cache=False)
            return not data.empty
            
        return False
    
    async def close(self) -> None:
        """Clean up resources"""
        # Close data sources
        if hasattr(self, 'primary_source') and self.primary_source:
            try:
                await self.primary_source.close()
                self.logger.debug("Closed primary data source")
            except Exception as e:
                self.logger.error(f"Error closing primary data source: {e}")
                
        if hasattr(self, 'backup_source') and self.backup_source:
            try:
                await self.backup_source.close()
                self.logger.debug("Closed backup data source")
            except Exception as e:
                self.logger.error(f"Error closing backup data source: {e}")
        
        # Close downloader
        if hasattr(self, 'downloader'):
            try:
                await self.downloader.close()
                self.logger.debug("Closed data downloader")
            except Exception as e:
                self.logger.error(f"Error closing data downloader: {e}")
        
        # Shutdown thread pool
        try:
            self.thread_pool.shutdown(wait=False)
            self.logger.debug("Thread pool shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")
                
        # Clear cache
        self.data_cache.clear()
        self.cache_last_accessed.clear()