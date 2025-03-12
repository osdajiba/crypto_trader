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
    """Local file data source, supports daily file format for efficient storage and retrieval"""

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
        Fetch historical data from local storage using daily files
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            start: Start time
            end: End time
            
        Returns:
            DataFrame: OHLCV data
        """
        # Use verified absolute paths
        base_dir = os.path.join(
            self.data_path,
            timeframe,
            symbol.replace('/', '_')
        )
        logger.info(f"Fetching historical data from local database: {symbol} {timeframe} {start} - {end}")

        if not os.path.exists(base_dir):
            logger.debug(f"The local data directory does not exist: {base_dir}")
            self.missing_symbols.add(f"{symbol}_{timeframe}")
            return pd.DataFrame()
        
        try:
            # Parse time parameters
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # If no start time, default to 30 days before end
            if not start_dt:
                start_dt = end_dt - timedelta(days=30)
                
            # Get list of days between start and end
            days_list = []
            current_day = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            end_day = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            while current_day <= end_day:
                days_list.append(current_day)
                current_day += timedelta(days=1)
                
            # Find all daily files that match our date range
            matched_files = []
            
            for day in days_list:
                # Format: 1h-YYYY-MM-DD.parquet
                year_dir = os.path.join(base_dir, str(day.year))
                month_dir = os.path.join(year_dir, f"{day.month:02d}")
                file_pattern = f"{timeframe}-{day.year}-{day.month:02d}-{day.day:02d}.parquet"
                file_path = os.path.join(month_dir, file_pattern)
                
                if os.path.exists(file_path):
                    matched_files.append(file_path)
                else:
                    # Fall back to legacy format if available
                    legacy_files = self._find_legacy_files(base_dir, day)
                    if legacy_files:
                        matched_files.extend(legacy_files)
                    else:
                        logger.debug(f"Daily file not found: {file_path}")
        except Exception as e:
            logger.error(f"Failed to determine date range: {str(e)}")
            return pd.DataFrame()
        
        if not matched_files:
            logger.debug(f"No matching daily files found for {symbol} {timeframe} between {start_dt} and {end_dt}")
            return pd.DataFrame()
            
        try:
            # Use the executor to read files concurrently
            dfs = await asyncio.gather(*[
                self.executor.submit(FileUtils.async_read_parquet, f) 
                for f in matched_files
            ])
            
            # Filter out empty DataFrames
            dfs = [df for df in dfs if not df.empty]
            
            if not dfs:
                logger.warning(f"All files were empty for {symbol} {timeframe}")
                return pd.DataFrame()
                
            # Combine all DataFrames
            combined_df = pd.concat(dfs, ignore_index=False)
            
            # Check index type before comparison
            if not pd.api.types.is_datetime64_any_dtype(combined_df.index):
                # If index is not datetime, convert it
                combined_df.index = pd.to_datetime(combined_df.index, utc=True)
                
            # Now apply time range filtering
            time_filter = pd.Series(True, index=combined_df.index)
            if start_dt:
                time_filter &= (combined_df.index >= start_dt)
            if end_dt:
                time_filter &= (combined_df.index <= end_dt)
            filtered_df = combined_df[time_filter]

            # Sort by index
            sorted_df = filtered_df.sort_index()

            
            logger.info(f"Retrieved {len(sorted_df)} rows for {symbol} {timeframe} from {len(matched_files)} files")
            return sorted_df
            
        except Exception as e:
            logger.error(f"Data loading failure: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()

    def _find_legacy_files(self, base_dir: str, day: datetime) -> List[str]:
        """Find legacy format files for a given day"""
        try:
            # Calculate day boundaries in timestamp format
            day_start = int(day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            day_end = int(day.replace(hour=23, minute=59, second=59, microsecond=999999).timestamp())
            
            matched_files = []
            
            # Scan directory for legacy files
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if not file.endswith('.parquet'):
                        continue
                        
                    try:
                        # Legacy format: start_ts_end_ts.parquet
                        filename = os.path.splitext(file)[0]
                        parts = filename.split('_')
                        if len(parts) < 2:
                            continue
                            
                        file_start = int(float(parts[0]))
                        file_end = int(float(parts[1]))
                        
                        # Check if file overlaps with the day
                        if (file_end >= day_start) and (file_start <= day_end):
                            matched_files.append(os.path.join(root, file))
                            
                    except (ValueError, IndexError):
                        continue
                        
            return matched_files
            
        except Exception as e:
            logger.error(f"Error finding legacy files: {str(e)}")
            return []
    
    async def update_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """
        Update local data storage with new data using daily file format
        
        Args:
            symbol: Trading pair
            timeframe: Time interval 
            data: New data to store
            
        Returns:
            bool: Success status
        """
        if data.empty:
            logger.warning(f"Attempted to update with empty data: {symbol} {timeframe}")
            return False
            
        try:
            # Ensure 'datetime' column exists and is datetime type
            if 'datetime' not in data.columns:
                if isinstance(data.index, pd.DatetimeIndex):
                    # Copy index to datetime column
                    data = data.reset_index()
                else:
                    # Try to convert timestamp to datetime
                    if 'timestamp' in data.columns:
                        data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
                    else:
                        logger.error(f"Cannot update data: no datetime or timestamp column in data")
                        return False
            
            # Convert datetime column to pandas datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                data['datetime'] = pd.to_datetime(data['datetime'])
            
            # Group data by day
            data['date'] = data['datetime'].dt.date
            grouped = data.groupby('date')
            
            # Prepare base directory
            base_dir = os.path.join(
                self.data_path,
                timeframe,
                symbol.replace('/', '_')
            )
            
            success_count = 0
            for date, group_df in grouped:
                # Create date-based directory structure
                year = date.year
                month = date.month
                
                year_dir = os.path.join(base_dir, str(year))
                month_dir = os.path.join(year_dir, f"{month:02d}")
                
                # Create directories if they don't exist
                os.makedirs(month_dir, exist_ok=True)
                
                # Create filename: 1h-2025-02-09.parquet
                filename = f"{timeframe}-{year}-{month:02d}-{date.day:02d}.parquet"
                file_path = os.path.join(month_dir, filename)
                
                # Drop the date column we added for grouping
                daily_df = group_df.drop(columns=['date'])
                
                # Check if file already exists
                if os.path.exists(file_path):
                    # Read existing file
                    existing_df = pd.read_parquet(file_path)
                    
                    # Ensure 'datetime' column is datetime type in existing data
                    if 'datetime' in existing_df.columns and not pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                    
                    # Set datetime as index if it's not already
                    if not isinstance(existing_df.index, pd.DatetimeIndex) and 'datetime' in existing_df.columns:
                        existing_df.set_index('datetime', inplace=True)
                    
                    # Set datetime as index for new data if it's not already
                    daily_df_copy = daily_df.copy()
                    if not isinstance(daily_df_copy.index, pd.DatetimeIndex) and 'datetime' in daily_df_copy.columns:
                        daily_df_copy.set_index('datetime', inplace=True)
                    
                    # Combine existing and new data
                    combined_df = pd.concat([existing_df, daily_df_copy])
                    
                    # Remove duplicates, keeping latest data
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    
                    # Sort by datetime
                    combined_df = combined_df.sort_index()
                    
                    # Reset index to have datetime as a column
                    if isinstance(combined_df.index, pd.DatetimeIndex):
                        combined_df = combined_df.reset_index()
                    
                    # Save to file
                    combined_df.to_parquet(file_path, index=False)
                    logger.info(f"Updated existing file: {file_path} with {len(daily_df)} new records")
                else:
                    # Save new file
                    daily_df.to_parquet(file_path, index=False)
                    logger.info(f"Created new file: {file_path} with {len(daily_df)} records")
                
                success_count += 1
            
            logger.info(f"Data update complete for {symbol} {timeframe}: updated {success_count} daily files")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update local data: {str(e)}\n{traceback.format_exc()}")
            return False
    
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