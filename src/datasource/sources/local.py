#!/usr/bin/env python3
# src/datasource/sources/local.py

"""
Local file-based data source implementation.
Provides access to historical market data stored in local files.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import asyncio

from src.common.abstract_factory import register_factory_class
from src.common.helpers import TimeUtils, ParquetFileManager
from src.common.config_manager import ConfigManager
from src.datasource.sources.base import BaseDataSource, DataSourceFetchError


@register_factory_class('datasource_factory', 'local', 
    description="Local file-based data source",
    features=["historical", "offline_access"],
    category="datasource")
class LocalDataSource(BaseDataSource):
    """
    Local file data source with standardized path structure
    
    This data source provides access to market data stored in local files
    using a standardized directory structure. It supports efficient lookup
    and retrieval of historical data for backtesting.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize local data source
        
        Args:
            config: Configuration manager
            params: Additional parameters
        """
        super().__init__(config, params)
        
        self.data_path = None
        self.missing_symbols = set()
        self.cache_enabled = True
    
    async def _initialize_source_specific(self) -> None:
        """Initialize local data source specific settings"""
        # Get and create data path
        self.data_path = self._get_data_path()
        os.makedirs(self.data_path, exist_ok=True)
        
        # Configure caching behavior
        self.cache_enabled = self.config.get("data", "storage", "cache_enabled", default=True)
        
        # Optional: scan for available data
        self._scan_available_data() if self.params.get("scan_on_init", False) else None
        
        self.logger.info(f"LocalDataSource initialized, data path: {self.data_path}")
    
    def _get_data_path(self) -> str:
        """
        Get and validate data path from config
        
        Returns:
            str: Absolute path to data directory
        """
        # Try multiple possible config paths for backward compatibility
        paths_to_check = [
            ('data', 'paths', 'historical_data_path'),
            ('data', 'storage', 'historical'),
            ('data_paths', 'historical_data')
        ]
        
        for path_keys in paths_to_check:
            path = self.config.get(*path_keys, default=None)
            if path:
                return os.path.abspath(os.path.expanduser(path))
        
        # Fallback to default path
        default_path = os.path.abspath('data/historical')
        self.logger.warning(f"Data path not found in config, using default: {default_path}")
        return default_path
    
    def _scan_available_data(self) -> None:
        """Scan data directory to identify available data"""
        self.logger.info("Scanning available data (not implemented)")
        # This would be implemented to catalog available symbols and timeframes
    
    def _generate_file_pattern(self, symbol: str, timeframe: str, 
                             start_dt: datetime, end_dt: datetime) -> List[str]:
        """
        Generate file path patterns based on standardized structure
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            List[str]: List of file path patterns to search
        """
        symbol_safe = symbol.replace('/', '_')
        sources = ['binance']  # Default source
        patterns = []
        
        start_year, end_year = start_dt.year, end_dt.year
        start_month, end_month = start_dt.month, end_dt.month
        
        for year in range(start_year, end_year + 1):
            month_start = start_month if year == start_year else 1
            month_end = end_month if year == end_year else 12
            
            for month in range(month_start, month_end + 1):
                for source in sources:
                    pattern = os.path.join(
                        self.data_path,
                        source,
                        symbol_safe,
                        timeframe,
                        str(year),
                        f"{month:02d}",
                        "*.parquet"
                    )
                    patterns.append(pattern)
        
        return patterns
            
    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical data with automatic integrity validation
        
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
            # Parse dates
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            if not start_dt:
                start_dt = end_dt - timedelta(days=30)
            
            # Generate file patterns
            file_patterns = self._generate_file_pattern(symbol, timeframe, start_dt, end_dt)
            
            # Find matching files
            matched_files = []
            for pattern in file_patterns:
                matched_files.extend(glob.glob(pattern))
            
            if not matched_files:
                self.logger.debug(f"No matching files found for {symbol} {timeframe} between {start_dt} and {end_dt}")
                self.missing_symbols.add(f"{symbol}_{timeframe}")
                return pd.DataFrame()
            
            # Filter files by date range based on filename
            filtered_files = []
            for file_path in matched_files:
                file_name = os.path.basename(file_path)
                if "to" in file_name:
                    try:
                        file_start_str, file_end_str = file_name.split('.')[0].split('to')
                        file_start = pd.to_datetime(file_start_str)
                        file_end = pd.to_datetime(file_end_str)
                        
                        if (start_dt <= file_end and end_dt >= file_start):
                            filtered_files.append(file_path)
                    except:
                        # If parsing fails, include file anyway
                        filtered_files.append(file_path)
                else:
                    # For files without date info in name, include them
                    filtered_files.append(file_path)
            
            if not filtered_files:
                self.logger.debug(f"No files in date range for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Load and combine files
            df = await ParquetFileManager.load_and_combine_files(
                filtered_files,
                date_filter=(start_dt, end_dt)
            )
            
            if not df.empty:
                self.logger.info(f"Retrieved {len(df)} rows for {symbol} {timeframe} from {len(filtered_files)} files")
                
                # Validate data
                self.validate_ohlcv_data(df)
                
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            self.logger.error(f"Data loading failure: {str(e)}")
            raise DataSourceFetchError(f"Failed to fetch historical data for {symbol} {timeframe}: {str(e)}")
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get the latest available data as "real-time" data
        
        For a local data source, this retrieves the most recent data point
        available in local storage.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            
        Returns:
            pd.DataFrame: DataFrame with latest data point
        """
        if not self._is_initialized:
            await self.initialize()
            
        # Get today's data as a proxy for "real-time"
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        df = await self.fetch_historical(symbol, timeframe, yesterday, today)
        
        if df.empty:
            self.logger.warning(f"LocalDataSource cannot provide real-time data: {symbol} {timeframe}")
            return df
            
        # Return only the last row
        last_row = df.iloc[[-1]].copy()
        self.logger.info(f"LocalDataSource provided last record as real-time data: {symbol} {timeframe}")
        
        return last_row
    
    async def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """
        Save data to local storage
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            timeframe: Timeframe for candles
            
        Returns:
            bool: Success status
        """
        if df.empty:
            self.logger.warning(f"Cannot save empty DataFrame for {symbol} {timeframe}")
            return False
            
        try:
            # Ensure datetime column is present and in proper format
            if 'datetime' not in df.columns:
                self.logger.error("DataFrame must have 'datetime' column to save")
                return False
                
            # Convert datetime to proper timezone-aware format if needed
            if not pd.api.types.is_datetime64_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Determine date range for filename
            start_dt = df['datetime'].min() if not df['datetime'].empty else datetime.now()
            end_dt = df['datetime'].max() if not df['datetime'].empty else datetime.now()
            
            # Generate storage path
            symbol_safe = symbol.replace('/', '_')
            source = 'binance'  # Default source name
            
            year = start_dt.year
            month = start_dt.month
            
            # Create directory structure
            dir_path = os.path.join(
                self.data_path,
                source,
                symbol_safe,
                timeframe,
                str(year),
                f"{month:02d}"
            )
            os.makedirs(dir_path, exist_ok=True)
            
            # Generate filename with ISO timestamp range
            start_iso = start_dt.isoformat()
            end_iso = end_dt.isoformat()
            filename = f"{start_iso}to{end_iso}.parquet"
            file_path = os.path.join(dir_path, filename)
            
            # Save to parquet file
            df.to_parquet(file_path, index=False)
            
            self.logger.info(f"Saved {len(df)} rows to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            return False
    
    async def _shutdown_source_specific(self) -> None:
        """Clean up local data source resources"""
        self.logger.debug("No specific resources to clean up for LocalDataSource")