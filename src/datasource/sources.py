# src/data/sources.py

import shutil
from typing import Dict, Optional, List, Union, Tuple
import pandas as pd
import os
import traceback
import json
import asyncio
from datetime import datetime, timedelta

from src.utils.time_utils import TimeUtils
from src.utils.file_utils import FileUtils
from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor
from src.exchange.binance import Binance
from src.utils.file_utils import ParquetFileManager

logger = LogManager.get_logger("trading_system")


class DataSource:
    """Data source abstract base class, defines interfaces for historical and real-time data"""
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                              start: Optional[Union[str, datetime]] = None, 
                              end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get historical OHLCV data
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            start: Start time
            end: End time
            
        Returns:
            DataFrame: OHLCV data
        """
        raise NotImplementedError

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get real-time OHLCV data
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            
        Returns:
            DataFrame: OHLCV data
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Close data source connection (if applicable)"""
        pass
        
    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """Convert timeframe to seconds"""
        units = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        if not timeframe:
            return 60  # Default to 1 minute
            
        # Extract number and unit
        num = int(''.join(filter(str.isdigit, timeframe)))
        unit = timeframe[-1].lower()
        
        # Check if unit is valid
        if unit not in units:
            logger.warning(f"Unknown timeframe unit: {unit}, using minutes")
            unit = 'm'
            
        return num * units.get(unit, 60)
        
    @staticmethod
    def get_optimal_data_ranges(start_dt: datetime, end_dt: datetime, 
                              timeframe: str, max_points: int = 1000) -> List[Tuple[datetime, datetime]]:
        """
        Break large date ranges into smaller chunks to optimize data retrieval
        
        Args:
            start_dt: Start date
            end_dt: End date
            timeframe: Time interval
            max_points: Max data points per request
            
        Returns:
            List[Tuple]: List of time ranges
        """
        # Calculate seconds per timeframe
        seconds_per_candle = DataSource.timeframe_to_seconds(timeframe)
        
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
    """Local file data source, supporting timestamp-based parquet files"""

    def __init__(self, config: ConfigManager):
        """
        Initialize local data source.
        
        Args:
            config (Dict): Configuration dictionary with 'data_path' etc.
        """
        # Get historical data path
        self.data_path = self._get_validated_data_path(config)

        if not self.data_path:
            # Try other possible config paths
            self.data_path = config.get('data_paths', 'historical_data', default="data/historical")
            
        # Ensure directory exists
        if not os.path.exists(self.data_path):
            try:
                os.makedirs(self.data_path)
                logger.info(f"Created historical data directory: {self.data_path}")
            except Exception as e:
                logger.error(f"Failed to create historical data directory: {str(e)}")
                
        # Async executor
        self.executor = AsyncExecutor()
        
        # Add file extension support
        self.supported_formats = {
            '.csv': self._read_csv,
            '.parquet': self._read_parquet,
            '.json': self._read_json
        }
        
        # Track data status
        self.missing_symbols = set()
        logger.info(f"LocalSource initialized, data path: {self.data_path}")

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Local source attempts to extract last record from latest data as "real-time" data
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            
        Returns:
            DataFrame: Last OHLCV data
        """
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
        
    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file"""
        return pd.read_csv(file_path)
        
    def _read_parquet(self, file_path: str) -> pd.DataFrame:
        """Read Parquet file"""
        return pd.read_parquet(file_path)
        
    def _read_json(self, file_path: str) -> pd.DataFrame:
        """Read JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
        
    async def get_missing_data_info(self) -> Dict[str, List[str]]:
        """
        Get missing data information
        
        Returns:
            Dict: Missing data information grouped by trading pair
        """
        missing_info = {}
        
        for item in self.missing_symbols:
            parts = item.split('_')
            if len(parts) >= 2:
                symbol = parts[0]
                if symbol not in missing_info:
                    missing_info[symbol] = []
                timeframe = parts[1]
                missing_info[symbol].append(timeframe)
                
        return missing_info
        
    async def migrate_legacy_data(self, symbol: str, timeframe: str) -> bool:
        """
        Migrate data from legacy format (individual candle files) to daily files
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            
        Returns:
            bool: Success status
        """
        try:
            # Get base directory for this symbol/timeframe
            base_dir = os.path.join(
                self.data_path,
                timeframe,
                symbol.replace('/', '_')
            )
            
            if not os.path.exists(base_dir):
                logger.warning(f"No data directory found to migrate: {base_dir}")
                return False
                
            # Scan for all parquet files with legacy timestamp-based names
            legacy_files = []
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if file.endswith('.parquet'):
                        # Check if it's a legacy format file (contains underscore between timestamps)
                        filename = os.path.splitext(file)[0]
                        if '_' in filename and not filename.startswith(timeframe):
                            file_path = os.path.join(root, file)
                            legacy_files.append(file_path)
            
            if not legacy_files:
                logger.info(f"No legacy format files found to migrate for {symbol} {timeframe}")
                return True  # Nothing to migrate is still a success
                
            logger.info(f"Found {len(legacy_files)} legacy format files to migrate for {symbol} {timeframe}")
            
            # Process files in batches to avoid memory issues
            batch_size = 100
            batches = [legacy_files[i:i + batch_size] for i in range(0, len(legacy_files), batch_size)]
            
            total_migrated = 0
            for batch_idx, file_batch in enumerate(batches):
                logger.info(f"Processing batch {batch_idx+1}/{len(batches)} ({len(file_batch)} files)")
                
                # Read all files in batch
                dfs = []
                for file_path in file_batch:
                    try:
                        df = pd.read_parquet(file_path)
                        if not df.empty:
                            dfs.append(df)
                    except Exception as e:
                        logger.error(f"Error reading file {file_path}: {str(e)}")
                
                if not dfs:
                    logger.warning(f"No valid data in batch {batch_idx+1}")
                    continue
                    
                # Combine all data
                combined_df = pd.concat(dfs, ignore_index=True)
                
                # Ensure datetime column exists
                if 'datetime' not in combined_df.columns:
                    if 'timestamp' in combined_df.columns:
                        combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
                    else:
                        logger.error(f"Cannot migrate data: no datetime or timestamp column found")
                        return False
                
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(combined_df['datetime']):
                    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
                
                # Update with new format
                success = await self.update_data(symbol, timeframe, combined_df)
                
                if success:
                    total_migrated += len(file_batch)
                    
                    # Optionally, backup old files after migration
                    backup_dir = os.path.join(base_dir, "legacy_backup")
                    os.makedirs(backup_dir, exist_ok=True)
                    
                    for file_path in file_batch:
                        try:
                            file_name = os.path.basename(file_path)
                            backup_path = os.path.join(backup_dir, file_name)
                            # Move file to backup directory
                            shutil.move(file_path, backup_path)
                        except Exception as e:
                            logger.error(f"Error backing up old file {file_path}: {str(e)}")
            
            logger.info(f"Migration complete for {symbol} {timeframe}: {total_migrated}/{len(legacy_files)} files processed")
            return total_migrated > 0
            
        except Exception as e:
            logger.error(f"Data migration failed: {str(e)}\n{traceback.format_exc()}")
            return False

    def _get_validated_data_path(self, config: Dict) -> str:
        """Validate and create data storage path"""
        # Obtained from the configuration path, with multiple layers of rollback
        path = config.get('data', 'paths', 'historical_data_path', default='data/historical')
        abs_path = os.path.abspath(os.path.expanduser(path))
        
        try:
            # Create a directory synchronously (at initialization)
            os.makedirs(abs_path, exist_ok=True)
            logger.info(f"Ensuring data directory exists: {abs_path}")
        except Exception as e:
            logger.critical(f"Cannot create data directory {abs_path}: {str(e)}")
            raise RuntimeError(f"Data directory initialization failed: {str(e)}")
        
        return abs_path

    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical data from local storage using timestamp-based Parquet files
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            start: Start time
            end: End time
            
        Returns:
            DataFrame: OHLCV data
        """
        
        logger.info(f"Fetching historical data from local database: {symbol} {timeframe} {start} - {end}")

        try:
            # Parse time parameters
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # If no start time, default to 30 days before end
            if not start_dt:
                start_dt = end_dt - timedelta(days=30)
            
            # Find files matching the date range
            file_paths = ParquetFileManager.find_files_in_date_range(
                self.data_path, timeframe, symbol, start_dt, end_dt
            )
            
            if not file_paths:
                logger.debug(f"No matching files found for {symbol} {timeframe} between {start_dt} and {end_dt}")
                self.missing_symbols.add(f"{symbol}_{timeframe}")
                return pd.DataFrame()
            
            # Load and combine files
            df = await ParquetFileManager.load_and_combine_files(
                file_paths,
                date_filter=(start_dt, end_dt),
                drop_duplicates=True
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe} in the specified date range")
                return df
            
            # Set datetime as index if not already
            if 'datetime' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                df.set_index('datetime', inplace=True)
            
            logger.info(f"Retrieved {len(df)} rows for {symbol} {timeframe} from {len(file_paths)} files")
            return df
            
        except Exception as e:
            logger.error(f"Data loading failure: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    async def update_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Update local data storage with new data using timestamp-based file format
        
        Args:
            symbol: Trading pair
            timeframe: Time interval 
            data: New data to store
            
        Returns:
            bool: Success status
        """
        from src.utils.file_utils import ParquetFileManager
        
        if data.empty:
            logger.warning(f"Attempted to update with empty data: {symbol} {timeframe}")
            return False
        
        # Save data using the simplified ParquetFileManager
        success = await ParquetFileManager.save_dataframe(
            df=data,
            base_path=self.data_path,
            timeframe=timeframe,
            symbol=symbol
        )
        
        return success
    
class ExchangeSource(DataSource):
    """Exchange data source, supports smart pagination for large historical datasets"""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize exchange data source.
        
        Args:
            config (Dict): Configuration dictionary with exchange API settings
        """
        # Create exchange connection
        self.exchange = Binance(config)
            
        self.executor = AsyncExecutor()
        
        # Load rate limit config
        self.max_requests_per_minute = config.get("api", "rate_limits", "requests_per_minute", default=20) if config else 20
        self.request_delay = 60.0 / self.max_requests_per_minute
        
        # Retry config
        self.max_retries = config.get("api", "retries", "max_attempts", default=3) if config else 3
        self.retry_delay = config.get("api", "retries", "delay_seconds", default=1) if config else 1
        
        logger.info(f"ExchangeSource initialized, max request rate: {self.max_requests_per_minute}/minute")        

    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Fetch historical data from exchange with smart chunking for large datasets
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            start: Start time
            end: End time
            
        Returns:
            DataFrame: OHLCV data
        """
        logger.info(f"Fetching historical data from exchange: {symbol} {timeframe} {start} - {end}")
        
        # Ensure executor is started
        await self.executor.start()
        
        # If no time range specified, get recent data
        if not start and not end:
            try:
                # Use AsyncExecutor to wrap synchronous method
                df = await self.executor.submit(
                    self.exchange.fetch_latest_ohlcv,
                    symbol=symbol, timeframe=timeframe, limit=100
                )
                logger.info(f"Fetched {len(df)} recent records for {symbol} {timeframe}")
                return df
            except Exception as e:
                logger.error(f"Failed to fetch recent data for {symbol}: {str(e)}")
                return pd.DataFrame()
        
        # Parse dates
        start_dt = TimeUtils.parse_timestamp(start)
        end_dt = TimeUtils.parse_timestamp(end, default_days_ago=0)  # Default to now
        
        try:
            # Use the smart_fetch_ohlcv method directly instead of chunking manually
            logger.info(f"Using smart fetch for {symbol} {timeframe} from {start_dt} to {end_dt}")
            df = await self.exchange.fetch_historical_ohlcv(symbol, timeframe, start_dt, end_dt)
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
            else:
                logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} from exchange")
                
            return df
            
        except Exception as e:
            logger.error(f"Smart fetch failed for {symbol} {timeframe}: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Get real-time data from exchange
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            
        Returns:
            DataFrame: Latest OHLCV data
        """
        logger.info(f"Fetching real-time data from exchange: {symbol} {timeframe}")
        
        try:
            # Use AsyncExecutor to run synchronous method 
            df = await self.executor.submit(
                self.exchange.fetch_latest_ohlcv,
                symbol=symbol, timeframe=timeframe, limit=1
            )
            
            logger.info(f"Real-time data fetch successful: {symbol} {timeframe}, {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch real-time data: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()

    async def close(self) -> None:
        """Close exchange connection and executor"""
        try:
            if hasattr(self.exchange, 'close'):
                await self.exchange.close()
                
            await self.executor.close()
            logger.info("ExchangeSource closed")
            
        except Exception as e:
            logger.error(f"Error closing ExchangeSource: {str(e)}")


class DataSourceFactory:
    """Data source factory, responsible for creating different types of data sources"""
    
    @staticmethod
    def create_source(source_type: str, config: ConfigManager) -> DataSource:
        """
        Create a data source instance.
        
        Args:
            source_type (str): Data source type ('local', 'exchange', or other)
            config (Dict): Configuration dictionary
        
        Returns:
            DataSource: Data source instance
        
        Raises:
            ValueError: If source_type is invalid
        """
        sources = {
            'local': LocalSource,
            'exchange': ExchangeSource
        }
        
        # Check if source type is valid
        source_type = source_type.lower()
        if source_type not in sources:
            available = ", ".join(sources.keys())
            logger.error(f"Unknown data source type: {source_type}, available options: {available}")
            raise ValueError(f"Unknown data source type: {source_type}, available options: {available}")
            
        # Create and return data source
        try:
            source = sources[source_type](config)
            logger.info(f"Created {source_type} data source")
            return source
        except Exception as e:
            logger.error(f"Failed to create {source_type} data source: {str(e)}\n{traceback.format_exc()}")
            raise