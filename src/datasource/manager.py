# src/data/data_manager.py

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools

from src.common.config import ConfigManager
from src.common.logging import LogManager
from src.common.helpers import TimeUtils
from datasource.datasources import *
from src.datasource.integrity import DataIntegrityChecker


class DataManager:
    """
    Optimized Data Manager responsible for fetching, caching, and managing market data
    with enhanced performance and reduced resource consumption
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
        
        # Data sources
        self.primary_source = None
        self.backup_source = None
        
        # Data cache with TTL and size limits for better memory management
        self.data_cache = {}
        self.cache_last_accessed = {}
        self.cache_max_size = config.get("data", "cache", "max_size", default=100) if config else 100
        self.cache_ttl_seconds = config.get("data", "cache", "ttl_seconds", default=3600) if config else 3600
        self.use_cache = config.get("data", "cache", "enabled", default=True) if config else True
        
        # Thread pool for CPU-bound operations
        max_workers = config.get("system", "max_workers", default=min(32, (os.cpu_count() or 4) + 4)) if config else None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Data integrity checker with optimized settings
        self.integrity_checker = DataIntegrityChecker(
            timeframe=config.get("data", "default_timeframe", default="1h") if config else "1h",
            parallel=config.get("data", "validation", "parallel", default=True) if config else True
        )
        self.validate_data = config.get("data", "validation", "enabled", default=True) if config else True
        
        # Pagination settings
        self.page_size = config.get("data", "pagination", "page_size", default=1000) if config else 1000
        self.max_concurrent_pages = config.get("data", "pagination", "max_concurrent", default=5) if config else 5
        
        # Create data sources based on source_type
        self._initialize_data_sources()
        
        # Data prefetch settings for performance optimization
        self.prefetch_enabled = config.get("data", "prefetch", "enabled", default=False) if config else False
        self.prefetch_days = config.get("data", "prefetch", "days", default=7) if config else 7
        
        # Batch processing settings
        self.batch_size = config.get("data", "batch_size", default=10000) if config else 10000
        
        self.logger.info(f"DataManager initialized: source={self.source_type}, cache={self.use_cache}, mode={self._get_trading_mode()}")
    
    def _initialize_data_sources(self):
        """Initialize appropriate data sources based on configuration"""
        try:
            factory = DataSourceFactory()
            primary_type = self.config.get("data", "primary_source", default="local")
            backup_type = self.config.get("data", "backup_source", default="exchange")
            
            # Initialize primary_source based on source_type
            if self.source_type == "local":
                self.primary_source = factory.create_source("local", self.config)
                    
            elif self.source_type == "exchange":
                self.primary_source = factory.create_source("exchange", self.config)
                
            elif self.source_type == "database":
                self.primary_source = factory.create_source("database", self.config)
                    
            elif self.source_type == "hybrid":
                self.primary_source = factory.create_source(primary_type, self.config)
                self.backup_source = factory.create_source(backup_type, self.config)
                    
            else:
                raise ValueError(f"Unsupported data source type: {self.source_type}")
                    
            if self._get_trading_mode() == "BACKTEST" and not self.backup_source:
                if self.source_type == "local":
                    self.backup_source = factory.create_source(backup_type, self.config)
                    
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
        Get historical market data, fix any issues detected
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            timeframe: Data timeframe/interval (e.g., "1h", "1d")
            start: Start datetime or date string
            end: End datetime or date string
            use_pagination: Whether to use pagination for large data requests
            
        Returns:
            pd.DataFrame: Historical market data
        """
        # Check cache first (with expiration)
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        if self.use_cache and cache_key in self.data_cache:
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
            # First try primary source
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
                if self.use_cache:
                    self._update_cache(cache_key, data)
                return data
            else:
                self.logger.warning(f"No historical data found in primary source for {symbol} {timeframe}")
            
            # If primary source failed or returned empty and we have a backup source, try it
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
                    if self.use_cache:
                        self._update_cache(cache_key, data)
                    
                    # If local is backup source and exchange is primary source, store downloaded data
                    if isinstance(self.backup_source, LocalSource) and hasattr(self.backup_source, 'update_data'):
                        await self.backup_source.update_data(symbol, timeframe, data)
                        self.logger.info(f"Data saved to local storage: {symbol} {timeframe}")
                    
                    return data
                else:
                    self.logger.error(f"Backup source also failed for {symbol} {timeframe}")
            
            # If we got here, both sources failed or returned empty
            self.logger.error(f"Failed to get historical data for {symbol} {timeframe}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def _fetch_and_validate_data(
        self,
        source: Any,
        symbol: str,
        timeframe: str,
        start: Optional[Union[str, datetime]],
        end: Optional[Union[str, datetime]],
        source_name: str = "unknown"
    ) -> pd.DataFrame:
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
                        data = await self._fix_data_issues(data, results, symbol, timeframe)
            
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
                fixed_chunk = await self._fix_data_issues(chunk, results, symbol, timeframe)
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
    
    async def get_latest_data(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get recent market data, fixing any detected issues
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            limit: Max records to fetch
            
        Returns:
            pd.DataFrame: Latest market data
        """
        # Only makes sense for exchange or hybrid sources
        if self.source_type in ["exchange", "hybrid"]:
            try:
                data = await self.primary_source.fetch_realtime(symbol, timeframe)
                
                if self.validate_data and not data.empty:
                    valid, results = await self.integrity_checker.check(data, symbol, timeframe)
                    if not valid:
                        self.logger.warning(f"Data integrity issues found in latest data for {symbol} {timeframe}")
                        data = await self._fix_data_issues(data, results, symbol, timeframe)
                        
                return data
                
            except Exception as e:
                self.logger.error(f"Error fetching latest data: {e}")
                return pd.DataFrame()
        else:
            # For local sources, get recent data from historical
            end = datetime.now()
            start = end - timedelta(days=7)  # Get last 7 days
            return await self.get_historical_data(symbol, timeframe, start, end)
    
    async def download_and_store_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: Optional[Union[str, datetime]] = None, 
        end: Optional[Union[str, datetime]] = None,
        validate_before_storage: bool = True
    ) -> bool:
        """
        Download data from exchange, validate and store it, fixing any issues
        
        Args:
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start: Start date
            end: End date
            validate_before_storage: Whether to validate and fix data before storage
            
        Returns:
            bool: Success status
        """
        # Need exchange data source
        exchange_source = None
        if self.source_type == "exchange":
            exchange_source = self.primary_source
        elif self.backup_source and hasattr(self.backup_source, 'fetch_historical'):
            exchange_source = self.backup_source
            
        if not exchange_source:
            self.logger.warning(f"No exchange source available for downloading {symbol} {timeframe}")
            return False
            
        try:
            # Parse dates
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # Default start date is one year ago if not specified
            if not start_dt:
                start_dt = end_dt - timedelta(days=365)
                
            self.logger.info(f"Downloading data for {symbol} {timeframe} from {start_dt} to {end_dt}")
            
            # Calculate duration and check if we should use chunked downloads
            duration_days = (end_dt - start_dt).days
            use_chunks = duration_days > 30  # Use chunks for long periods
            
            if use_chunks:
                data = await self._download_in_chunks(exchange_source, symbol, timeframe, start_dt, end_dt)
            else:
                # Simple download for shorter periods
                data = await exchange_source.fetch_historical(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_dt,
                    end=end_dt
                )
            
            if data.empty:
                self.logger.error(f"Failed to download data for {symbol} {timeframe}")
                return False
                
            # Validate and fix data if enabled
            if validate_before_storage and self.validate_data:
                valid, results = await self.integrity_checker.check(data, symbol, timeframe)
                if not valid:
                    self.logger.warning(f"Data integrity issues found in downloaded data for {symbol} {timeframe}")
                    data = await self._fix_data_issues(
                        data, 
                        results, 
                        symbol=None,  # Don't persist here, will save directly below
                        timeframe=None
                    )
                    
                    # Log recommendations for future data quality improvement
                    recommendations = self.integrity_checker.get_correction_recommendations(results)
                    if recommendations:
                        self.logger.info(f"Recommendations for {symbol} {timeframe}: {', '.join(recommendations)}")
                
            # Get local source to store data
            local_source = None
            if self.source_type == "local":
                local_source = self.primary_source
            elif hasattr(self.backup_source, 'update_data'):
                local_source = self.backup_source
                
            if not local_source:
                self.logger.error(f"No local source available for storing data")
                return False
                
            # Store data
            success = await local_source.update_data(symbol, timeframe, data)
            
            if success:
                self.logger.info(f"Successfully downloaded and stored {len(data)} records for {symbol} {timeframe}")
            else:
                self.logger.error(f"Failed to store downloaded data for {symbol} {timeframe}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error downloading and storing data: {e}")
            return False
    
    async def _download_in_chunks(
        self,
        source: Any,
        symbol: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> pd.DataFrame:
        """
        Download data in chunks for better performance and reduced memory usage
        
        Args:
            source: Data source to use
            symbol: Trading pair symbol
            timeframe: Data timeframe
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            pd.DataFrame: Combined data from all chunks
        """
        # Calculate chunk size based on timeframe
        timeframe_seconds = TimeUtils.timeframe_to_seconds(timeframe)
        
        # For daily data, use 30-day chunks
        # For hourly data, use 7-day chunks
        # For minute data, use 1-day chunks
        if timeframe_seconds >= 86400:  # Daily or higher
            chunk_days = 30
        elif timeframe_seconds >= 3600:  # Hourly
            chunk_days = 7
        else:  # Minute data
            chunk_days = 1
        
        # Create chunks
        chunks = []
        current_start = start_dt
        
        tasks = []
        while current_start < end_dt:
            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
            
            self.logger.debug(f"Creating download task for {symbol} {timeframe} from {current_start} to {current_end}")
            tasks.append(source.fetch_historical(
                symbol=symbol,
                timeframe=timeframe,
                start=current_start,
                end=current_end
            ))
            
            current_start = current_end
        
        # Execute all tasks concurrently for better performance
        chunk_results = await asyncio.gather(*tasks)
        
        # Combine chunks, ignoring empty ones
        valid_chunks = [chunk for chunk in chunk_results if not chunk.empty]
        
        if not valid_chunks:
            return pd.DataFrame()
        
        # Use pandas.concat which is optimized for this operation
        combined_data = pd.concat(valid_chunks, ignore_index=False)
        
        # Find timestamp column
        timestamp_col = None
        for col in ['datetime', 'timestamp', 'date', 'time']:
            if col in combined_data.columns:
                timestamp_col = col
                break
        
        if timestamp_col:
            # Sort by timestamp
            combined_data = combined_data.sort_values(by=timestamp_col)
            
            # Remove duplicates that might exist at chunk boundaries
            combined_data = combined_data.drop_duplicates(subset=[timestamp_col], keep='last')
        
        return combined_data

    async def scan_and_repair_data(self, symbol: str, timeframe: str,
                           start: Optional[Union[str, datetime]] = None,
                           end: Optional[Union[str, datetime]] = None) -> Dict[str, Any]:
        """
        Scan and repair data for a specific symbol and timeframe
        
        Args:
            symbol: Symbol to scan and repair
            timeframe: Timeframe to scan and repair
            start: Start date for scan
            end: End date for scan
            
        Returns:
            Dict: Results summary
        """
        results = {
            "symbol": symbol,
            "timeframe": timeframe,
            "scanned": 0,
            "with_issues": 0,
            "fixed": 0,
            "failed": 0,
            "issue_types": {}
        }
        
        try:
            # Load data without validation for initial scan
            self.logger.info(f"Loading data for {symbol} {timeframe} to scan for issues")
            
            # Temporarily disable validation for initial load
            original_validate_setting = self.validate_data
            self.validate_data = False
            
            data = await self.get_historical_data(symbol, timeframe, start, end)
            
            # Restore validation setting
            self.validate_data = original_validate_setting
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol} {timeframe}, skipping repair")
                return results
                
            results["scanned"] = 1
            
            # Use vectorized operations for large datasets
            if len(data) > 10000:
                self.logger.info(f"Using optimized processing for large dataset ({len(data)} rows)")
                return await self._scan_and_repair_large_data(data, symbol, timeframe, results)
            
            # Check data integrity
            valid, check_results = await self.integrity_checker.check(data, symbol, timeframe)
            
            if valid:
                self.logger.info(f"No integrity issues found for {symbol} {timeframe}")
                return results
                
            # Record issues found
            results["with_issues"] = 1
            
            # Count issue types
            for issue_type, issue_result in check_results.items():
                if isinstance(issue_result, dict) and issue_result.get("passed") is False:
                    results["issue_types"].setdefault(issue_type, 0)
                    results["issue_types"][issue_type] += 1
            
            # Fix the issues
            fixed_data = await self._fix_data_issues(data, check_results, symbol, timeframe)
            
            # Recheck to verify fixes
            recheck_valid, recheck_results = await self.integrity_checker.check(fixed_data, symbol, timeframe)
            
            if recheck_valid:
                # Save the fixed data
                success = await self._persist_fixed_data(fixed_data, symbol, timeframe)
                
                if success:
                    results["fixed"] = 1
                    self.logger.info(f"Successfully fixed and saved data for {symbol} {timeframe}")
                else:
                    results["failed"] = 1
                    self.logger.error(f"Failed to save fixed data for {symbol} {timeframe}")
            else:
                results["failed"] = 1
                issue_types = [k for k, v in recheck_results.items() 
                              if isinstance(v, dict) and v.get("passed") is False]
                self.logger.error(f"Data integrity issues remain after attempted fix for {symbol} {timeframe}: {', '.join(issue_types)}")
            
            return results
            
        except Exception as e:
            results["failed"] = 1
            self.logger.error(f"Error during scan and repair for {symbol} {timeframe}: {e}", exc_info=True)
            return results
    
    async def _scan_and_repair_large_data(self, df: pd.DataFrame, symbol: str, timeframe: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimized handling for large datasets using chunking and parallel processing
        
        Args:
            df: Large DataFrame to process
            symbol: Symbol for logging
            timeframe: Timeframe for logging
            results: Current results dictionary
            
        Returns:
            Dict: Updated results
        """
        # Process in chunks
        chunks = [df.iloc[i:i + self.batch_size] for i in range(0, len(df), self.batch_size)]
        fixed_chunks = []
        
        issues_found = False
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            self.logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {symbol} {timeframe}")
            valid, check_results = await self.integrity_checker.check(chunk, symbol, timeframe)
            
            if not valid:
                issues_found = True
                # Track issue types
                for issue_type, issue_result in check_results.items():
                    if isinstance(issue_result, dict) and issue_result.get("passed") is False:
                        results["issue_types"].setdefault(issue_type, 0)
                        results["issue_types"][issue_type] += 1
                
                # Fix this chunk
                fixed_chunk = await self._fix_data_issues(chunk, check_results, symbol, timeframe)
                fixed_chunks.append(fixed_chunk)
            else:
                fixed_chunks.append(chunk)
        
        if not issues_found:
            self.logger.info(f"No integrity issues found for {symbol} {timeframe}")
            return results
        
        # Record that issues were found
        results["with_issues"] = 1
        
        # Combine fixed chunks
        combined_df = pd.concat(fixed_chunks, ignore_index=False)
        
        # Final check across all chunks
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
        
        # Verify the fixed data
        recheck_valid, _ = await self.integrity_checker.check(combined_df, symbol, timeframe)
        
        if recheck_valid:
            # Save the fixed data
            success = await self._persist_fixed_data(combined_df, symbol, timeframe)
            
            if success:
                results["fixed"] = 1
                self.logger.info(f"Successfully fixed and saved large dataset for {symbol} {timeframe}")
            else:
                results["failed"] = 1
                self.logger.error(f"Failed to save fixed large dataset for {symbol} {timeframe}")
        else:
            results["failed"] = 1
            self.logger.error(f"Data integrity issues remain after attempted fix for large dataset {symbol} {timeframe}")
        
        return results
            
    async def verify_data_integrity(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify data integrity after fixes
        
        Args:
            df: DataFrame to verify
            symbol: Symbol for logging
            timeframe: Timeframe for logging
            
        Returns:
            Tuple[bool, Dict]: (passed, results) indicating if verification passed
        """
        try:
            valid, results = await self.integrity_checker.check(df, symbol, timeframe)
            
            if valid:
                self.logger.info(f"Data integrity verification passed for {symbol} {timeframe}")
            else:
                issue_types = [k for k, v in results.items() 
                              if isinstance(v, dict) and v.get("passed") is False]
                self.logger.warning(f"Data integrity verification failed for {symbol} {timeframe}: {', '.join(issue_types)}")
            
            return valid, results
        
        except Exception as e:
            self.logger.error(f"Error verifying data integrity: {e}", exc_info=True)
            return False, {"error": str(e)}

    async def _fix_data_issues(self, df: pd.DataFrame, check_results: Dict[str, Any], 
                        symbol: Optional[str] = None, timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Apply automated fixes to common data issues
        
        Args:
            df: Original DataFrame with issues
            check_results: Results from integrity checker
            symbol: Trading pair symbol (for logging and persistence)
            timeframe: Data timeframe (for logging and persistence)
            
        Returns:
            pd.DataFrame: Fixed DataFrame
        """
        fixed_df = df.copy()
        fixes_applied = []
        issue_types_fixed = []
        
        try:
            # Get timestamp column
            timestamp_col = None
            for col in ['datetime', 'timestamp', 'date', 'time']:
                if col in fixed_df.columns:
                    timestamp_col = col
                    break
            
            # Fix: Sort by timestamp sequence
            if check_results.get("timestamp_sequence", {}).get("passed") is False:
                self.logger.info(f"Fixing timestamp sequence by sorting for {symbol} {timeframe}")
                fixed_df = fixed_df.sort_values(by=timestamp_col)
                fixes_applied.append("Sorted timestamps in ascending order")
                issue_types_fixed.append("timestamp_sequence")
            
            # Fix: Remove duplicate timestamps
            if check_results.get("duplicate_timestamps", {}).get("passed") is False:
                duplicate_count = check_results.get("duplicate_timestamps", {}).get("duplicate_count", 0)
                self.logger.info(f"Fixing {duplicate_count} duplicate timestamps by keeping last occurrence for {symbol} {timeframe}")
                fixed_df = fixed_df.drop_duplicates(subset=[timestamp_col], keep='last')
                fixes_applied.append(f"Removed {duplicate_count} duplicate timestamps")
                issue_types_fixed.append("duplicate_timestamps")
            
            # Fix: Missing values - use vectorized operations for performance
            if check_results.get("missing_values", {}).get("passed") is False:
                missing_counts = check_results.get("missing_values", {}).get("missing_counts", {})
                total_missing = sum(missing_counts.values())
                self.logger.info(f"Fixing {total_missing} missing values using forward fill for {symbol} {timeframe}")
                
                # Optimize by processing only columns with missing values
                for col, count in missing_counts.items():
                    if count > 0:
                        # Forward fill each column individually 
                        fixed_df[col] = fixed_df[col].ffill()
                        
                        # Check if still has NaNs at the beginning
                        if fixed_df[col].isna().any():
                            fixed_df[col] = fixed_df[col].bfill()
                
                fixes_applied.append(f"Filled {total_missing} missing values")
                issue_types_fixed.append("missing_values")
            
            # Fix: OHLC consistency using vectorized operations
            consistency_key = "ohlc_consistency" if "ohlc_consistency" in check_results else "consistency"
            if check_results.get(consistency_key, {}).get("passed") is False:
                inconsistency_count = check_results.get(consistency_key, {}).get("count", 0)
                self.logger.info(f"Fixing {inconsistency_count} OHLC consistency issues for {symbol} {timeframe}")
                
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
                
                fixes_applied.append(f"Corrected {inconsistency_count} OHLC inconsistencies")
                issue_types_fixed.append(consistency_key)
            
            # Fix: Zero price values
            if check_results.get("zero_values", {}).get("passed") is False:
                zero_counts = check_results.get("zero_values", {}).get("zero_counts", {})
                price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in fixed_df.columns]
                
                total_zeros = 0
                for col in price_cols:
                    if col in zero_counts:
                        zero_mask = fixed_df[col] == 0
                        zeros_in_col = zero_mask.sum()
                        total_zeros += zeros_in_col
                        
                        if zeros_in_col > 0:
                            self.logger.info(f"Replacing {zeros_in_col} zero values in {col} for {symbol} {timeframe}")
                            fixed_df.loc[zero_mask, col] = None  # Convert zeros to NaN
                            fixed_df[col] = fixed_df[col].ffill().bfill()  # Fill NaNs
                
                fixes_applied.append(f"Replaced {total_zeros} zero values")
                issue_types_fixed.append("zero_values")
            
            # Fix: Price anomalies (extreme values)
            if check_results.get("price_anomalies", {}).get("passed") is False:
                anomalies = check_results.get("price_anomalies", {}).get("anomalies", {})
                total_anomalies = sum(info.get("count", 0) for info in anomalies.values())
                
                if total_anomalies > 0:
                    self.logger.info(f"Fixing {total_anomalies} price anomalies for {symbol} {timeframe}")
                    
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
                    
                    fixes_applied.append(f"Corrected {total_anomalies} price anomalies")
                    issue_types_fixed.append("price_anomalies")
            
            # Log summary of all fixes
            if fixes_applied:
                fixes_summary = ", ".join(fixes_applied)
                self.logger.info(f"Data fixes summary for {symbol} {timeframe}: {fixes_summary}")
                
                # Register fixes with integrity checker for statistics
                if hasattr(self.integrity_checker, 'register_fix') and issue_types_fixed:
                    self.integrity_checker.register_fix(issue_types_fixed)
                
                # If symbol and timeframe provided, persist the fixed data
                if symbol and timeframe:
                    await self._persist_fixed_data(fixed_df, symbol, timeframe)
            
            return fixed_df
            
        except Exception as e:
            self.logger.error(f"Error fixing data issues for {symbol} {timeframe}: {e}", exc_info=True)
            return df  # If fixing fails, return original data
    
    async def _persist_fixed_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Save fixed data directly to the data source
        
        Args:
            df: Fixed DataFrame
            symbol: Trading pair symbol
            timeframe: Data timeframe
            
        Returns:
            bool: Success status
        """
        if df.empty:
            self.logger.warning(f"Attempted to persist empty dataframe for {symbol} {timeframe}")
            return False
        
        try:
            # Find suitable data source for saving
            local_source = None
            
            # Try to find a LocalSource
            if self.source_type == "local" and isinstance(self.primary_source, LocalSource):
                local_source = self.primary_source
            elif hasattr(self, 'backup_source') and isinstance(self.backup_source, ExchangeSource):
                local_source = self.backup_source
                
            # If we have a LocalSource, save data directly
            if local_source and hasattr(local_source, 'update_data'):
                success = await local_source.update_data(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    data=df
                )
                
                if success:
                    self.logger.info(f"Successfully persisted fixed data for {symbol} {timeframe}")
                else:
                    self.logger.error(f"Failed to persist fixed data for {symbol} {timeframe}")
                    
                return success
                
            # If we have a DatabaseSource, save to database
            db_source = None
            if self.source_type == "database" and isinstance(self.primary_source, DatabaseSource):
                db_source = self.primary_source
            elif hasattr(self, 'backup_source') and isinstance(self.backup_source, DatabaseSource):
                db_source = self.backup_source
                
            if db_source and hasattr(db_source, 'update_data'):
                success = await db_source.update_data(symbol, timeframe, df)
                if success:
                    self.logger.info(f"Successfully persisted fixed data to database for {symbol} {timeframe}")
                else:
                    self.logger.error(f"Failed to persist fixed data to database for {symbol} {timeframe}")
                return success
                
            self.logger.warning(f"No suitable source found to persist fixed data for {symbol} {timeframe}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error persisting fixed data for {symbol} {timeframe}: {e}", exc_info=True)
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
                if key in self.cache_last_accessed:
                    del self.cache_last_accessed[key]
            self.logger.info(f"Cleared cache for {symbol} {timeframe} ({len(keys_to_remove)} entries)")
            
        elif symbol:
            # Clear all timeframes for a symbol
            keys_to_remove = [k for k in self.data_cache.keys() if k.startswith(f"{symbol}|")]
            for key in keys_to_remove:
                del self.data_cache[key]
                if key in self.cache_last_accessed:
                    del self.cache_last_accessed[key]
            self.logger.info(f"Cleared cache for {symbol} ({len(keys_to_remove)} entries)")
            
        elif timeframe:
            # Clear all symbols for a timeframe
            keys_to_remove = [k for k in self.data_cache.keys() if f"|{timeframe}|" in k]
            for key in keys_to_remove:
                del self.data_cache[key]
                if key in self.cache_last_accessed:
                    del self.cache_last_accessed[key]
            self.logger.info(f"Cleared cache for timeframe {timeframe} ({len(keys_to_remove)} entries)")
            
        else:
            # Clear all cache
            count = len(self.data_cache)
            self.data_cache.clear()
            self.cache_last_accessed.clear()
            self.logger.info(f"Cleared entire data cache ({count} entries)")
    
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
        if not symbols:
            return {}
            
        # Calculate date range
        end = datetime.now()
        start = end - timedelta(days=days)
        
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
                self.logger.error(f"Error preloading data for {symbol}: {result}")
            elif not isinstance(result, pd.DataFrame) or result.empty:
                self.logger.warning(f"No data available when preloading {symbol} {timeframe}")
            else:
                results[symbol] = result
                self.logger.debug(f"Preloaded {len(result)} records for {symbol} {timeframe}")
        
        self.logger.info(f"Preloaded data for {len(results)}/{len(symbols)} symbols")
        return results
    
    async def close(self) -> None:
        """Clean up resources"""
        # Close data sources
        if self.primary_source and hasattr(self.primary_source, 'close'):
            try:
                await self.primary_source.close()
                self.logger.debug("Closed primary data source")
            except Exception as e:
                self.logger.error(f"Error closing primary data source: {e}")
                
        if self.backup_source and hasattr(self.backup_source, 'close'):
            try:
                await self.backup_source.close()
                self.logger.debug("Closed backup data source")
            except Exception as e:
                self.logger.error(f"Error closing backup data source: {e}")
        
        # Shutdown thread pool
        try:
            self.thread_pool.shutdown(wait=False)
            self.logger.debug("Thread pool shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")
                
        # Clear cache
        self.data_cache.clear()
        self.cache_last_accessed.clear()