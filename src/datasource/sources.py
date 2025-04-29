#!/usr/bin/env python3
# src/datasource/sources.py

from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import os
import asyncio
from datetime import datetime, timedelta, timezone
import logging

from src.common.helpers import TimeUtils, ParquetFileManager
from src.common.log_manager import LogManager
from src.common.config import ConfigManager
from src.common.async_executor import AsyncExecutor
from src.datasource.integrity import DataIntegrityChecker

logger = LogManager.get_logger("datasource")

class DataSource:
    """Abstract base class defining interfaces for historical and real-time data"""
    
    def __init__(self, config: ConfigManager):
        """Initialize data source"""
        self.config = config
        self.executor = AsyncExecutor()
        self.logger = logger
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                             start: Optional[Union[str, datetime]] = None, 
                             end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Get historical OHLCV data"""
        raise NotImplementedError

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get real-time OHLCV data"""
        raise NotImplementedError

    async def close(self) -> None:
        """Close data source connection"""
        if hasattr(self.executor, 'close'):
            await self.executor.close()
    
    @staticmethod
    def get_optimal_data_ranges(start_dt: datetime, end_dt: datetime, 
                              timeframe: str, max_points: int = 1000) -> List[Tuple[datetime, datetime]]:
        """Break large date ranges into smaller chunks"""
        # Calculate seconds per timeframe
        seconds_per_candle = TimeUtils.timeframe_to_seconds(timeframe)
        
        # Calculate total seconds
        total_seconds = (end_dt - start_dt).total_seconds()
        
        # Estimate total data points
        estimated_points = total_seconds / seconds_per_candle
        
        # If points fewer than max, return entire range
        if estimated_points <= max_points:
            return [(start_dt, end_dt)]
            
        # Calculate number of chunks needed
        num_chunks = int(estimated_points / max_points) + 1
        
        # Calculate size of each chunk (seconds)
        chunk_seconds = total_seconds / num_chunks
        
        # Create date range list
        ranges = []
        for i in range(num_chunks):
            chunk_start = start_dt + timedelta(seconds=i * chunk_seconds)
            chunk_end = start_dt + timedelta(seconds=(i+1) * chunk_seconds)
            
            # Ensure last chunk includes endpoint
            if i == num_chunks - 1:
                chunk_end = end_dt
                
            # Add to range list
            ranges.append((chunk_start, chunk_end))
            
        return ranges

class LocalSource(DataSource):
    """Local file data source with standardized path structure"""

    def __init__(self, config: ConfigManager):
        """Initialize local data source"""
        super().__init__(config)
        
        # Get historical data path
        self.data_path = self._get_data_path()
        os.makedirs(self.data_path, exist_ok=True)
        
        # Track missing data
        self.missing_symbols = set()
        logger.info(f"LocalSource initialized, data path: {self.data_path}")

    def _get_data_path(self) -> str:
        """Get and validate data path from config"""
        paths_to_check = [
            ('data', 'paths', 'historical_data_path'),
            ('data', 'storage', 'historical'),
            ('data_paths', 'historical_data')
        ]
        
        for path_keys in paths_to_check:
            path = self.config.get(*path_keys, default=None)
            if path:
                return os.path.abspath(os.path.expanduser(path))
        
        # Default fallback
        return os.path.abspath('data/historical')
    
    def _generate_file_pattern(self, symbol: str, timeframe: str, 
                             start_dt: datetime, end_dt: datetime) -> List[str]:
        """Generate file path patterns based on standardized structure"""
        symbol_safe = symbol.replace('/', '_')
        
        # Get source names to check
        sources = ['binance']  # Default to binance, could be expanded
        
        patterns = []
        
        # Format: source_name/symbol/timeframe/year/month/*.parquet
        start_year, end_year = start_dt.year, end_dt.year
        start_month, end_month = start_dt.month, end_dt.month
        
        # Generate all year/month combinations in range
        for year in range(start_year, end_year + 1):
            # Determine month range for this year
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
                            end: Optional[Union[str, datetime]] = None,
                            auto_fix: bool = True) -> pd.DataFrame:
        """Fetch historical data with automatic integrity validation"""
        logger.info(f"Fetching historical data for: {symbol} {timeframe}")

        try:
            # Parse time parameters
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # Default to 30 days before end if no start
            if not start_dt:
                start_dt = end_dt - timedelta(days=30)
            
            # Find files matching the date range
            file_patterns = self._generate_file_pattern(symbol, timeframe, start_dt, end_dt)
            
            # Find matching files
            matched_files = []
            for pattern in file_patterns:
                import glob
                matched_files.extend(glob.glob(pattern))
            
            if not matched_files:
                logger.debug(f"No matching files found for {symbol} {timeframe} between {start_dt} and {end_dt}")
                self.missing_symbols.add(f"{symbol}_{timeframe}")
                return pd.DataFrame()
            
            # Filter files by date range from filenames
            filtered_files = []
            for file_path in matched_files:
                file_name = os.path.basename(file_path)
                # Extract date range if filename has ISO format with "to"
                if "to" in file_name:
                    try:
                        file_start_str, file_end_str = file_name.split('.')[0].split('to')
                        file_start = pd.to_datetime(file_start_str)
                        file_end = pd.to_datetime(file_end_str)
                        
                        # Check if file overlaps with requested range
                        if (start_dt <= file_end and end_dt >= file_start):
                            filtered_files.append(file_path)
                    except:
                        # If we can't parse, include file just in case
                        filtered_files.append(file_path)
                else:
                    # Can't determine range from filename, include it
                    filtered_files.append(file_path)
            
            # If no files after filtering, return empty DataFrame
            if not filtered_files:
                logger.debug(f"No files in date range for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Load and combine files
            df = await ParquetFileManager.load_and_combine_files(
                filtered_files,
                date_filter=(start_dt, end_dt)
            )
            
            if not df.empty:
                logger.info(f"Retrieved {len(df)} rows for {symbol} {timeframe} from {len(filtered_files)} files")
                return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"Data loading failure: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get the latest available data as "real-time" data"""
        # Get today's data
        today = datetime.now()
        start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get today's data from historical storage
        df = await self.fetch_historical(symbol, timeframe, start, today)
        
        if df.empty:
            logger.warning(f"LocalSource cannot provide real-time data: {symbol} {timeframe}")
            return df
            
        # Return last record
        last_row = df.iloc[[-1]].copy()
        logger.info(f"LocalSource provided last record as real-time data: {symbol} {timeframe}")
        return last_row

class ExchangeSource(DataSource):
    """Exchange data source for real-time and historical data"""
    
    def __init__(self, config: ConfigManager):
        """Initialize exchange data source"""
        super().__init__(config)
        
        # Create exchange connection
        self.exchange = None
        self._init_exchange()
        
        # Cache directory
        self.cache_dir = self._setup_cache_dir()
        
        # Rate limit settings
        self.max_requests_per_minute = config.get("api", "rate_limits", "requests_per_minute", default=20)
        self.request_delay = 60.0 / self.max_requests_per_minute
        
        # Retry settings
        self.max_retries = config.get("api", "retries", "max_attempts", default=3)
        self.retry_delay = config.get("api", "retries", "delay_seconds", default=1)
        
        # Local storage for saving fetched data
        self.data_path = self._get_data_path()
        os.makedirs(self.data_path, exist_ok=True)
        
        logger.info(f"ExchangeSource initialized, max request rate: {self.max_requests_per_minute}/minute")
    
    def _setup_cache_dir(self) -> str:
        """Set up and return cache directory"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    
    def _get_data_path(self) -> str:
        """Get data storage path from config"""
        paths_to_check = [
            ('data', 'paths', 'historical_data_path'),
            ('data', 'storage', 'historical'),
            ('data_paths', 'historical_data')
        ]
        
        for path_keys in paths_to_check:
            path = self.config.get(*path_keys, default=None)
            if path:
                return os.path.abspath(os.path.expanduser(path))
        
        # Default fallback
        return os.path.abspath('data/historical')
        
    def _init_exchange(self) -> None:
        """Initialize the exchange connection"""
        from src.exchange.adapters.binance import Binance
        
        try:
            self.exchange = Binance(self.config)
            logger.info("Exchange connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise RuntimeError(f"Exchange initialization failed: {str(e)}")
    
    def _generate_storage_path(self, symbol: str, timeframe: str, 
                             start_dt: datetime, end_dt: datetime) -> str:
        """Generate standardized storage path"""
        symbol_safe = symbol.replace('/', '_')
        source_name = "binance"  # Default to binance, could be dynamic
        
        # Convert to ISO format
        start_iso = start_dt.astimezone(timezone.utc).isoformat()
        end_iso = end_dt.astimezone(timezone.utc).isoformat()
        
        # Format: data_source/symbol/timeframe/year/month/start_to_end.parquet
        year = start_dt.year
        month = f"{start_dt.month:02d}"
        
        # Create filename with date range
        filename = f"{start_iso}to{end_iso}.parquet"
        
        path = os.path.join(
            self.data_path,
            source_name,
            symbol_safe,
            timeframe,
            str(year),
            month
        )
        
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, filename)

    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical data from exchange and save to local storage"""
        logger.info(f"Fetching historical data from exchange: {symbol} {timeframe} {start} - {end}")
        
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Parse dates
            start_dt = TimeUtils.parse_timestamp(start)
            end_dt = TimeUtils.parse_timestamp(end, default_days_ago=0)  # Default to now
            
            # Check for local file first
            file_path = self._generate_storage_path(symbol, timeframe, start_dt, end_dt)
            if os.path.exists(file_path):
                logger.info(f"Loading {symbol} {timeframe} data from local cache: {file_path}")
                try:
                    df = pd.read_parquet(file_path)
                    if not df.empty:
                        return df
                except Exception as e:
                    logger.warning(f"Error reading cached file {file_path}: {e}")
            
            # Fetch from exchange if not available locally
            df = await self.exchange.fetch_historical_ohlcv(symbol, timeframe, start_dt, end_dt)
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
            else:
                logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} from exchange")
                
                # Save data to local storage
                try:
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    df.to_parquet(file_path, index=False)
                    logger.info(f"Saved {symbol} {timeframe} data to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save data to {file_path}: {e}")
                
            return df
            
        except Exception as e:
            logger.error(f"Exchange fetch failed for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get real-time data from exchange"""
        logger.info(f"Fetching real-time data from exchange: {symbol} {timeframe}")
        
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Use AsyncExecutor to run synchronous method 
            df = await self.executor.submit(
                self.exchange.fetch_latest_ohlcv,
                symbol=symbol, timeframe=timeframe, limit=1
            )
            
            logger.info(f"Real-time data fetch successful: {symbol} {timeframe}, {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch real-time data: {str(e)}")
            return pd.DataFrame()

    async def close(self) -> None:
        """Close exchange connection and executor"""
        try:
            if self.exchange and hasattr(self.exchange, 'close'):
                await self.exchange.close()
                
            await super().close()
            logger.info("ExchangeSource closed")
            
        except Exception as e:
            logger.error(f"Error closing ExchangeSource: {str(e)}")

class HybridSource(DataSource):
    """Hybrid data source that combines local and exchange data"""
    
    def __init__(self, config: ConfigManager):
        """Initialize hybrid data source"""
        super().__init__(config)
        
        # Create both local and exchange sources
        self.local_source = LocalSource(config)
        self.exchange_source = ExchangeSource(config)
        
        # Set preferred source for historical and realtime
        self.prefer_local_for_historical = config.get("data", "source", "prefer_local", default=True)
        
        logger.info(f"HybridSource initialized, prefer_local_for_historical={self.prefer_local_for_historical}")
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical data, preferring local source if available"""
        if self.prefer_local_for_historical:
            # Try local first, then exchange
            df = await self.local_source.fetch_historical(symbol, timeframe, start, end)
            if not df.empty:
                return df
            
            logger.info(f"Local data not available for {symbol} {timeframe}, trying exchange...")
            df = await self.exchange_source.fetch_historical(symbol, timeframe, start, end)
            return df
        else:
            # Try exchange first, then local
            df = await self.exchange_source.fetch_historical(symbol, timeframe, start, end)
            if not df.empty:
                return df
            
            logger.info(f"Exchange data not available for {symbol} {timeframe}, trying local...")
            df = await self.local_source.fetch_historical(symbol, timeframe, start, end)
            return df
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get real-time data, always from exchange if possible"""
        # Always try exchange first for real-time data
        df = await self.exchange_source.fetch_realtime(symbol, timeframe)
        if not df.empty:
            return df
        
        logger.info(f"Exchange real-time data not available for {symbol} {timeframe}, trying local...")
        df = await self.local_source.fetch_realtime(symbol, timeframe)
        return df
    
    async def close(self) -> None:
        """Close all data sources"""
        await self.local_source.close()
        await self.exchange_source.close()
        await super().close()

class DataSourceFactory:
    """Factory for creating data sources"""
    
    @staticmethod
    def create_source(source_type: str, config: ConfigManager) -> DataSource:
        """Create a data source instance"""
        sources = {
            'local': LocalSource,
            'exchange': ExchangeSource,
            'hybrid': HybridSource
        }
        
        # Check if source type is valid
        source_type = source_type.lower()
        if source_type not in sources:
            available = ", ".join(sources.keys())
            raise ValueError(f"Unknown data source type: {source_type}, available options: {available}")
            
        # Create and return data source
        try:
            source = sources[source_type](config)
            logger.info(f"Created {source_type} data source")
            return source
        except Exception as e:
            logger.error(f"Failed to create {source_type} data source: {str(e)}")
            raise