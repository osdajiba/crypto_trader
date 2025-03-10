# src/data/sources.py

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
    """Local file data source, supports multiple file formats and smart partial merging"""

    def __init__(self, config: Dict):
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
        """验证并创建数据存储路径"""
        # 从配置获取路径，带多层回退
        path = config.get('data', 'paths', 'historical_data_path', default='db/historical')
        
        # 转换为绝对路径
        abs_path = os.path.abspath(os.path.expanduser(path))
        
        try:
            # 同步创建目录（在初始化时完成）
            os.makedirs(abs_path, exist_ok=True)
            logger.info(f"确保数据目录存在: {abs_path}")
        except Exception as e:
            logger.critical(f"无法创建数据目录 {abs_path}: {str(e)}")
            raise RuntimeError(f"数据目录初始化失败: {str(e)}")
        
        return abs_path

    async def fetch_historical(self, symbol: str, timeframe: str, 
                              start: Optional[Union[str, datetime]] = None, 
                              end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        # 使用验证过的绝对路径
        base_dir = os.path.join(
            self.data_path,
            timeframe,
            symbol.replace('/', '_')
        )
        
        # 新增路径存在性检查
        if not os.path.exists(base_dir):
            logger.debug(f"本地数据目录不存在: {base_dir}")
            return pd.DataFrame()
        
        # 时间戳转换增加错误处理
        try:
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else None
            start_ts = start_dt.timestamp() if start_dt else 0
            end_ts = end_dt.timestamp() if end_dt else float('inf')
        except Exception as e:
            logger.error(f"时间参数解析失败: {str(e)}")
            return pd.DataFrame()
        
        # 文件匹配逻辑优化
        matched_files = []
        try:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    if not file.endswith('.parquet'):
                        continue
                    try:
                        # 文件名解析增强容错
                        filename = os.path.splitext(file)[0]
                        parts = filename.split('_')
                        if len(parts) < 2:
                            continue
                            
                        file_start = float(parts[0])
                        file_end = float(parts[1])
                        
                        # 时间范围判断逻辑优化
                        if (file_end >= start_ts) and (file_start <= end_ts):
                            matched_files.append(os.path.join(root, file))
                    except ValueError:
                        logger.warning(f"无效文件名格式: {file}")
                        continue
        except Exception as e:
            logger.error(f"遍历目录失败: {base_dir} - {str(e)}")
            return pd.DataFrame()
        
        # 异步读取优化
        if not matched_files:
            logger.debug(f"未找到匹配文件: {base_dir}")
            return pd.DataFrame()
            
        try:
            # 使用统一文件工具类进行读取
            dfs = await asyncio.gather(*[
                self.executor.submit(FileUtils.async_read_parquet, f) 
                for f in matched_files
            ])
            
            combined_df = pd.concat(dfs, ignore_index=False)
            
            # 时间范围过滤
            if start_dt or end_dt:
                time_filter = True
                if start_dt:
                    time_filter &= (combined_df.index >= start_dt)
                if end_dt:
                    time_filter &= (combined_df.index <= end_dt)
                combined_df = combined_df[time_filter]
            
            return combined_df.sort_index()
            
        except Exception as e:
            logger.error(f"数据加载失败: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Local source attempts to extract last record from latest data as "real-time" data
        
        Args:
            symbol: Trading pair
            timeframe: Time interval
            
        Returns:
            DataFrame: Last OHLCV data
        """
        # Get all data from historical
        df = await self.fetch_historical(symbol, timeframe)
        
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

class ExchangeSource(DataSource):
    """Exchange data source, supports smart pagination for large historical datasets"""
    
    def __init__(self, config: Dict):
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
    def create_source(source_type: str, config: Dict) -> DataSource:
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