# src/data/sources.py

from typing import Dict, Optional, List, Union, Tuple
import pandas as pd
import os
from datetime import datetime, timedelta

from src.utils.time_utils import TimeUtils
from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor

logger = LogManager.get_logger("trading_system")


class DataSource:
    """Data source abstract base class, defines interfaces for historical and real-time data"""
    
    def __init__(self, config: ConfigManager):
        """Initialize the data source"""
        self.config = config
        self.executor = AsyncExecutor()
        self.logger = logger
    
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
        if hasattr(self.executor, 'close'):
            await self.executor.close()
        
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
        """Initialize local data source"""
        super().__init__(config)
        
        # Get historical data path and create if needed
        self.data_path = self._get_data_path()
        os.makedirs(self.data_path, exist_ok=True)
        
        # Track missing data info
        self.missing_symbols = set()
        logger.info(f"LocalSource initialized, data path: {self.data_path}")

    def _get_data_path(self) -> str:
        """Get and validate data path from config"""
        # Try different config paths to find the data directory
        paths_to_check = [
            ('data', 'paths', 'historical_data_path'),
            ('data_paths', 'historical_data'),
            ('data', 'storage', 'historical')
        ]
        
        for path_keys in paths_to_check:
            path = self.config.get(*path_keys, default=None)
            if path:
                return os.path.abspath(os.path.expanduser(path))
        
        # Default fallback
        return os.path.abspath('data/historical')
 
    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical data from local storage using timestamp-based Parquet files"""
        logger.info(f"Fetching historical data from local database: {symbol} {timeframe} {start} - {end}")

        try:
            # Parse time parameters
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
            
            # If no start time, default to 30 days before end
            if not start_dt:
                start_dt = end_dt - timedelta(days=30)
            
            # Find files matching the date range
            from src.utils.file_utils import ParquetFileManager
            file_paths = ParquetFileManager.find_files_in_date_range(self.data_path, timeframe, symbol, start_dt, end_dt)
            
            if not file_paths:
                logger.debug(f"No matching files found for {symbol} {timeframe} between {start_dt} and {end_dt}")
                self.missing_symbols.add(f"{symbol}_{timeframe}")
                return pd.DataFrame()
            
            # Load and combine files
            df = await ParquetFileManager.load_and_combine_files(
                file_paths,
                date_filter=(start_dt, end_dt)
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol} {timeframe} in the specified date range")
                return df
            
            logger.info(f"Retrieved {len(df)} rows for {symbol} {timeframe} from {len(file_paths)} files")
            return df
            
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

    async def update_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Update local data storage with new data"""
        if data.empty:
            logger.warning(f"Attempted to update with empty data: {symbol} {timeframe}")
            return False
        
        from src.utils.file_utils import ParquetFileManager
        
        # Save data using the ParquetFileManager
        success = await ParquetFileManager.save_dataframe(
            df=data,
            base_path=self.data_path,
            timeframe=timeframe,
            symbol=symbol
        )
        
        return success

    async def get_missing_data_info(self) -> Dict[str, List[str]]:
        """Get missing data information"""
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
    
class ExchangeSource(DataSource):
    """Exchange data source using CCXT"""
    
    def __init__(self, config: ConfigManager):
        """Initialize exchange data source"""
        super().__init__(config)
        
        # Create exchange connection
        self.exchange = None
        self._init_exchange()
        
        # Cache directory setup
        self.cache_dir = self._setup_cache_dir()
        
        # Rate limit settings
        self.max_requests_per_minute = config.get("api", "rate_limits", "requests_per_minute", default=20)
        self.request_delay = 60.0 / self.max_requests_per_minute
        
        # Retry settings
        self.max_retries = config.get("api", "retries", "max_attempts", default=3)
        self.retry_delay = config.get("api", "retries", "delay_seconds", default=1)
        
        logger.info(f"ExchangeSource initialized, max request rate: {self.max_requests_per_minute}/minute")
    
    def _setup_cache_dir(self) -> str:
        """Set up and return cache directory"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
        
    def _init_exchange(self) -> None:
        """Initialize the exchange connection"""
        from src.exchange.binance import Binance
        
        try:
            self.exchange = Binance(self.config)
            logger.info("Exchange connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {str(e)}")
            raise RuntimeError(f"Exchange initialization failed: {str(e)}")

    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical data from exchange"""
        logger.info(f"Fetching historical data from exchange: {symbol} {timeframe} {start} - {end}")
        
        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()
        
        try:
            # Parse dates
            start_dt = TimeUtils.parse_timestamp(start)
            end_dt = TimeUtils.parse_timestamp(end, default_days_ago=0)  # Default to now
            
            # Use the exchange method directly
            df = await self.exchange.fetch_historical_ohlcv(symbol, timeframe, start_dt, end_dt)
            
            if df.empty:
                logger.warning(f"No data fetched for {symbol} {timeframe}")
            else:
                logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} from exchange")
                
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


class DatabaseSource(DataSource):
    """Database data source, supporting SQL databases like MySQL, PostgreSQL, etc."""

    def __init__(self, config: ConfigManager):
        """Initialize database data source"""
        super().__init__(config)
        
        # Get database connection parameters
        self.db_type = self.config.get("database", "type", default="mysql")
        self.db_host = self.config.get("database", "host", default="localhost")
        self.db_port = self.config.get("database", "port", default=3306)
        self.db_name = self.config.get("database", "name", default="trading")
        self.db_user = self.config.get("database", "user", default="root")
        self.db_password = self.config.get("database", "password", default="")
        self.db_table_prefix = self.config.get("database", "table_prefix", default="")
        
        # Initialize connection
        self.connection = None
        self._init_connection()
        
        logger.info(f"DatabaseSource initialized, db_type: {self.db_type}, host: {self.db_host}")

    def _init_connection(self) -> None:
        """Initialize database connection"""
        try:
            if self.db_type.lower() == "mysql":
                import mysql.connector
                self.connection = mysql.connector.connect(
                    host=self.db_host,
                    port=self.db_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
            elif self.db_type.lower() == "postgresql":
                import psycopg2
                self.connection = psycopg2.connect(
                    host=self.db_host,
                    port=self.db_port,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
            else:
                raise ValueError(f"Unsupported database type: {self.db_type}")
                
            logger.info(f"Connected to {self.db_type} database at {self.db_host}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}")

    async def fetch_historical(self, symbol: str, timeframe: str, 
                            start: Optional[Union[str, datetime]] = None, 
                            end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """Fetch historical data from database"""
        logger.info(f"Fetching historical data from database: {symbol} {timeframe} {start} - {end}")
        
        if not self.connection:
            logger.error("Database not connected")
            return pd.DataFrame()
        
        try:
            # Parse dates
            start_dt = TimeUtils.parse_timestamp(start)
            end_dt = TimeUtils.parse_timestamp(end, default_days_ago=0)  # Default to now
            
            # Build the query
            table_name = f"{self.db_table_prefix}{symbol.replace('/', '_')}_{timeframe}"
            
            # Create a cursor
            cursor = self.connection.cursor(dictionary=True)
            
            # Build query
            query = f"SELECT * FROM {table_name} WHERE datetime BETWEEN %s AND %s ORDER BY datetime"
            cursor.execute(query, (start_dt, end_dt))
            
            # Fetch data
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                logger.warning(f"No data found in database for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to dataframe
            df = pd.DataFrame(rows)
            
            # Make sure datetime is a datetime type
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe} from database")
            return df
            
        except Exception as e:
            logger.error(f"Database fetch failed for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get real-time data from database (latest record)"""
        logger.info(f"Fetching real-time data from database: {symbol} {timeframe}")
        
        if not self.connection:
            logger.error("Database not connected")
            return pd.DataFrame()
        
        try:
            # Build the query
            table_name = f"{self.db_table_prefix}{symbol.replace('/', '_')}_{timeframe}"
            
            # Create a cursor
            cursor = self.connection.cursor(dictionary=True)
            
            # Build query to get latest record
            query = f"SELECT * FROM {table_name} ORDER BY datetime DESC LIMIT 1"
            cursor.execute(query)
            
            # Fetch data
            row = cursor.fetchone()
            cursor.close()
            
            if not row:
                logger.warning(f"No data found in database for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to dataframe
            df = pd.DataFrame([row])
            
            # Make sure datetime is a datetime type
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            logger.info(f"Fetched latest data for {symbol} {timeframe} from database")
            return df
            
        except Exception as e:
            logger.error(f"Database fetch failed for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()

    async def update_data(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        """Update database with new data"""
        if data.empty:
            logger.warning(f"Attempted to update with empty data: {symbol} {timeframe}")
            return False
        
        try:
            # Build the table name
            table_name = f"{self.db_table_prefix}{symbol.replace('/', '_')}_{timeframe}"
            
            # Create a cursor
            cursor = self.connection.cursor()
            
            # Check if table exists, if not create it
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                datetime DATETIME PRIMARY KEY,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume FLOAT
            )
            """
            cursor.execute(create_table_query)
            
            # Insert data
            inserted = 0
            updated = 0
            
            for _, row in data.iterrows():
                # Extract values
                values = {}
                for col in ['datetime', 'open', 'high', 'low', 'close', 'volume']:
                    if col in row:
                        values[col] = row[col]
                
                if 'datetime' not in values:
                    continue
                
                # Convert datetime to str if needed
                if isinstance(values['datetime'], (datetime, pd.Timestamp)):
                    values['datetime'] = values['datetime'].strftime('%Y-%m-%d %H:%M:%S')
                
                # Build column list and placeholders
                columns = ', '.join(values.keys())
                placeholders = ', '.join(['%s'] * len(values))
                update_parts = ', '.join([f"{col} = %s" for col in values.keys() if col != 'datetime'])
                
                # Build query with upsert logic
                if self.db_type.lower() == "mysql":
                    query = f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})
                    ON DUPLICATE KEY UPDATE {update_parts}
                    """
                    # For MySQL, duplicate values are listed twice (for insert and update)
                    query_values = list(values.values())
                    query_values.extend([values[col] for col in values.keys() if col != 'datetime'])
                    
                elif self.db_type.lower() == "postgresql":
                    query = f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES ({placeholders})
                    ON CONFLICT (datetime) DO UPDATE SET {update_parts}
                    """
                    # For PostgreSQL, duplicate values are listed once
                    query_values = list(values.values())
                    query_values.extend([values[col] for col in values.keys() if col != 'datetime'])
                
                try:
                    cursor.execute(query, query_values)
                    if cursor.rowcount == 1:
                        inserted += 1
                    else:
                        updated += 1
                except Exception as e:
                    logger.error(f"Error inserting/updating row: {str(e)}")
            
            # Commit changes
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Database update successful: {symbol} {timeframe} | Inserted: {inserted}, Updated: {updated}")
            return True
            
        except Exception as e:
            logger.error(f"Database update failed: {str(e)}")
            return False

    async def close(self) -> None:
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database connection: {str(e)}")
            
        await super().close()
        
class DataSourceFactory:
    """Factory for creating data sources"""
    
    @staticmethod
    def create_source(source_type: str, config: ConfigManager) -> DataSource:
        """Create a data source instance"""
        sources = {
            'local': LocalSource,
            'exchange': ExchangeSource,
            'database': DatabaseSource
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