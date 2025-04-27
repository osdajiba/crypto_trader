# src/common/helpers.py

import pytz
import asyncio
import os
import json
import csv
import functools
import pandas as pd
import aiofiles
import pyarrow as pa
import pyarrow.parquet as pq
from enum import Enum
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple, Union


from common.log_manager import LogManager
logger = LogManager.get_logger("helpers")


class TradingMode(Enum):
    """Centralize the definition of transaction mode types"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    
    
class TimeUtils:
    """Enhanced time processing utility class"""
    
    @staticmethod
    def parse_timestamp(timestamp: Union[str, datetime, int, float, None], 
                    default_days_ago: int = 30) -> Optional[datetime]:
        """Parse various timestamp formats into a datetime object"""
        # 定义默认返回值函数
        def default_time():
            return datetime.now(timezone.utc) - timedelta(days=default_days_ago)
        
        # 处理None
        if timestamp is None:
            return default_time()
        
        # 处理datetime对象
        if isinstance(timestamp, datetime):
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        
        # 处理数字时间戳
        if isinstance(timestamp, (int, float)):
            # 使用三元表达式检查是否为毫秒
            divisor = 1000 if timestamp > 1e11 else 1
            return datetime.fromtimestamp(timestamp / divisor, tz=timezone.utc)
        
        # 处理字符串时间戳
        if isinstance(timestamp, str):
            # 清理常见的时区指示符
            clean_ts = timestamp.replace('Z', '+00:00')
            
            # 尝试ISO格式
            has_iso_indicators = ('T' in clean_ts or 
                                ('+' in clean_ts or '-' in clean_ts and 'T' in clean_ts))
            if has_iso_indicators:
                try:
                    dt = datetime.fromisoformat(clean_ts)
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
            
            # 尝试常见日期格式
            date_formats = {
                '%Y-%m-%d': None,
                '%Y/%m/%d': None,
                '%m/%d/%Y': None,
                '%d-%m-%Y': None,
                '%Y-%m-%d %H:%M:%S': None,
                '%Y-%m-%dT%H:%M:%S': None,
                '%Y%m%d': None
            }
            
            for fmt in date_formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
        
        # 对于所有其他情况记录警告并返回默认值
        logger.warning(f"Couldn't parse timestamp: {timestamp}")
        return default_time()
    
    @staticmethod
    def to_timestamp(dt: datetime, milliseconds: bool = True) -> int:
        """Convert datetime to a timestamp"""
        if dt.tzinfo is None:
            dt = pytz.utc.localize(dt)
        factor = 1000 if milliseconds else 1
        return int(dt.timestamp() * factor)
    
    @staticmethod
    def from_timestamp(timestamp: Union[int, float], milliseconds: bool = False) -> datetime:
        """Convert timestamp to datetime object"""
        if milliseconds:
            timestamp = timestamp / 1000
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def ensure_tz_aware(dt: Union[datetime, pd.Timestamp]) -> Union[datetime, pd.Timestamp]:
        """Ensure a datetime object has timezone information (UTC)"""
        if dt is None:
            return datetime.now(timezone.utc)
            
        # 处理pandas Timestamp
        if isinstance(dt, pd.Timestamp):
            return dt if dt.tzinfo else dt.tz_localize('UTC')
            
        # 处理Python datetime
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    @staticmethod
    def is_in_date_range(dt: datetime, start_dt: datetime, end_dt: datetime) -> bool:
        """Check if a datetime is within a date range"""
        # Ensure all datetimes are timezone-aware
        dt = TimeUtils.ensure_tz_aware(dt)
        start_dt = TimeUtils.ensure_tz_aware(start_dt)
        end_dt = TimeUtils.ensure_tz_aware(end_dt)
        
        return start_dt <= dt <= end_dt
    
    @staticmethod
    def is_timestamp_in_range(ts: int, start_ts: int, end_ts: int) -> bool:
        """Check if a timestamp is within a timestamp range"""
        return start_ts <= ts <= end_ts
    
    @staticmethod
    def round_datetime(dt: datetime, interval_minutes: int = 5) -> datetime:
        """Round a datetime to the nearest interval"""
        minutes = dt.minute
        rounded_minutes = ((minutes + interval_minutes // 2) // interval_minutes) * interval_minutes
        
        # Create new datetime with rounded minutes
        rounded_dt = dt.replace(minute=rounded_minutes % 60, second=0, microsecond=0)
        
        # Add an hour if needed
        if rounded_minutes >= 60:
            rounded_dt = rounded_dt + timedelta(hours=rounded_minutes // 60)
            
        return rounded_dt
    
    @staticmethod
    def get_time_intervals(start_dt: datetime, end_dt: datetime, 
                        interval: Union[str, timedelta]) -> List[datetime]:
        """Generate a list of datetime points at regular intervals"""
        # 解析字符串间隔为timedelta
        if isinstance(interval, str):
            unit_map = {'d': 'days', 'h': 'hours', 'm': 'minutes', 's': 'seconds'}
            unit = interval[-1].lower()
            value = int(interval[:-1])
            
            if unit not in unit_map:
                raise ValueError(f"Unsupported interval unit: {unit}")
                
            delta = timedelta(**{unit_map[unit]: value})
        else:
            delta = interval
        
        # 使用列表推导式生成时间点
        result = []
        current = start_dt
        while current <= end_dt:
            result.append(current)
            current += delta
            
        return result
    
    @staticmethod
    def get_start_end_of_day(dt: datetime) -> Tuple[datetime, datetime]:
        """Get start and end of day for a given datetime"""
        start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return start, end
    
    @staticmethod
    def get_start_end_of_month(dt: datetime) -> Tuple[datetime, datetime]:
        """Get start and end of month for a given datetime"""
        start = dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Find the last day of the month
        if dt.month == 12:
            end_month = dt.replace(year=dt.year+1, month=1, day=1)
        else:
            end_month = dt.replace(month=dt.month+1, day=1)
            
        end = end_month - timedelta(microseconds=1)
        return start, end
    
    @staticmethod
    def get_current_timestamp(milliseconds: bool = True) -> int:
        """Get the current UTC timestamp"""
        dt = datetime.now(timezone.utc)
        return TimeUtils.to_timestamp(dt, milliseconds)
    
    @staticmethod
    def format_datetime(dt: datetime, fmt: str = '%Y-%m-%d %H:%M:%S') -> str:
        """Format a datetime object to string"""
        return dt.strftime(fmt)
    
    @staticmethod
    def parse_timeframe(timeframe: str) -> timedelta:
        """Parse a timeframe string to timedelta"""
        unit = timeframe[-1].lower()
        
        try:
            value = int(timeframe[:-1])
        except ValueError:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        
        if unit == 'd':
            return timedelta(days=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'm':
            return timedelta(minutes=value)
        elif unit == 's':
            return timedelta(seconds=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
        

class FileUtils:
    """File operation utility class with optimized async operations"""
    
    PARQUET_COMPRESSION = 'SNAPPY'
    CSV_ENCODING = 'utf-8'
    DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB

    @staticmethod
    async def _async_io_executor(func: Callable, *args, **kwargs) -> Any:
        """General purpose asynchronous IO executor"""
        loop = asyncio.get_running_loop()
        bound_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    @staticmethod
    async def async_makedirs(path: str) -> None:
        """Asynchronous directory creation"""
        await FileUtils._async_io_executor(os.makedirs, path, exist_ok=True)

    @staticmethod
    async def async_write_parquet(file_path: str, df: pd.DataFrame) -> None:
        """Optimized asynchronous Parquet writes"""
        try:
            dir_path = os.path.dirname(file_path)
            await FileUtils.async_makedirs(dir_path)

            # Handle timezone
            if 'datetime' in df.columns and df['datetime'].dt.tz is not None:
                df = df.copy()
                df['datetime'] = df['datetime'].dt.tz_convert(None)

            table = pa.Table.from_pandas(df)
            await FileUtils._async_io_executor(
                pq.write_table,
                table,
                file_path,
                compression=FileUtils.PARQUET_COMPRESSION
            )
        except Exception as e:
            logger.error(f"Parquet write failed: {str(e)}")

    @staticmethod
    async def async_read_parquet(file_path: str) -> pd.DataFrame:
        """Asynchronously read a parquet file"""
        try:
            df = await FileUtils._async_io_executor(pd.read_parquet, file_path)
            return df
        except Exception as e:
            logger.error(f"Error reading parquet file {file_path}: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def timestamp_to_path(data_path: str, timeframe: str, symbol: str, 
                        timestamp: datetime) -> str:
        """Generate directory path for a specific timestamp"""
        safe_symbol = symbol.replace('/', '_')
        
        year = timestamp.year
        month = timestamp.month
        
        path = os.path.join(
            data_path,
            timeframe,
            safe_symbol,
            str(year),
            f"{month:02d}"
        )
        
        return path
    
    @staticmethod
    def generate_timestamp_filename(start: datetime, end: datetime) -> str:
        """Generate timestamp-based filename for parquet files"""
        return f"{int(start.timestamp())}_{int(end.timestamp())}.parquet"

    @staticmethod
    async def _async_file_operation(mode: str, file_path: str, 
                                   data: Optional[Any] = None) -> Any:
        """Generic asynchronous file operation"""
        await FileUtils.async_makedirs(os.path.dirname(file_path))
        
        if mode == 'read':
            async with aiofiles.open(file_path, 'r', encoding=FileUtils.CSV_ENCODING) as f:
                return await f.read()
        
        async with aiofiles.open(file_path, 'w', encoding=FileUtils.CSV_ENCODING) as f:
            await f.write(data)

    @staticmethod
    async def async_read_json(file_path: str) -> Dict[str, Any]:
        """Read JSON file asynchronously"""
        content = await FileUtils._async_file_operation('read', file_path)
        return json.loads(content)

    @staticmethod
    async def async_write_json(file_path: str, data: Dict[str, Any]) -> None:
        """Write JSON file asynchronously"""
        await FileUtils._async_file_operation('write', file_path, json.dumps(data, indent=2))

    @staticmethod
    async def async_read_csv(file_path: str) -> List[Dict[str, str]]:
        """Read CSV file asynchronously"""
        content = await FileUtils._async_file_operation('read', file_path)
        return list(csv.DictReader(content.splitlines()))

    @staticmethod
    async def async_write_csv(file_path: str, data: List[Dict[str, Any]]) -> None:
        """Write CSV file asynchronously"""
        if not data:
            return

        headers = data[0].keys()
        csv_content = ','.join(headers) + '\n'
        csv_content += '\n'.join(
            ','.join(str(item) for item in row.values())
            for row in data
        )
        await FileUtils._async_file_operation('write', file_path, csv_content)

    @staticmethod
    def sync_operation(async_func: Callable) -> Callable:
        """Decorator to run async functions synchronously"""
        @functools.wraps(async_func)
        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
        return wrapper

    @staticmethod
    def ensure_data_directory(data_path: str, timeframe: str, symbol: str, 
                            timestamp: datetime) -> str:
        """Ensure directory structure exists and return path"""
        path = FileUtils.timestamp_to_path(data_path, timeframe, symbol, timestamp)
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    async def batch_process_parquet_files(file_paths: List[str],
                                        processor_func: Callable,
                                        batch_size: int = 10) -> List[Any]:
        """Process multiple parquet files concurrently in batches"""
        results = []
        
        # 批量处理文件以避免资源过载
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            batch_results = await asyncio.gather(
                *[processor_func(path) for path in batch], 
                return_exceptions=True
            )
            
            # 过滤出异常并添加有效结果
            results.extend(result for result in batch_results 
                        if not isinstance(result, Exception))
        
        return results


class ParquetFileManager:
    """Manages Parquet file operations for time-series data"""

    @staticmethod
    def find_files_in_date_range(
        base_path: str, 
        timeframe: str, 
        symbol: str, 
        start_dt: datetime, 
        end_dt: datetime
    ) -> List[str]:
        """Find Parquet files for symbol/timeframe within a date range"""
        try:
            symbol_dir = os.path.join(base_path, timeframe, symbol.replace('/', '_'))
            
            # 确保时区感知的日期时间并转换为时间戳
            start_dt, end_dt = map(TimeUtils.ensure_tz_aware, (start_dt, end_dt))
            start_ts, end_ts = int(start_dt.timestamp()), int(end_dt.timestamp())
            
            # 如果目录不存在则提前返回
            if not os.path.exists(symbol_dir):
                logger.warning(f"Directory does not exist: {symbol_dir}")
                return []
                
            logger.info(f"Searching for files in {symbol_dir} from timestamp {start_ts} to {end_ts}")
            
            file_paths = []
            
            # 生成要搜索的年月组合
            date_ranges = []
            current = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
            end_month = datetime(end_dt.year, end_dt.month, 1, tzinfo=timezone.utc)
            
            while current <= end_month:
                date_ranges.append((current.year, current.month))
                # 移至下个月
                current = (datetime(current.year + 1, 1, 1) if current.month == 12 
                        else datetime(current.year, current.month + 1, 1)).replace(tzinfo=timezone.utc)
            
            # 检查各月目录中的文件
            for year, month in date_ranges:
                month_dir = os.path.join(symbol_dir, str(year), f"{month:02d}")
                if not os.path.exists(month_dir):
                    continue
                    
                # 使用列表推导式和条件表达式处理文件
                for file in (f for f in os.listdir(month_dir) if f.endswith('.parquet')):
                    file_path = os.path.join(month_dir, file)
                    
                    # 时间戳格式文件名处理
                    if '_' in file:
                        parts = file.split('.')[0].split('_')
                        if len(parts) >= 2:
                            try:
                                file_start_ts, file_end_ts = map(int, parts[:2])
                                # 使用布尔表达式简化重叠检查
                                overlaps = (
                                    (start_ts <= file_start_ts <= end_ts) or
                                    (start_ts <= file_end_ts <= end_ts) or
                                    (file_start_ts <= start_ts and file_end_ts >= end_ts)
                                )
                                if overlaps:
                                    file_paths.append(file_path)
                            except ValueError:
                                pass
                    
                    # 时间段格式文件名处理
                    elif file.startswith(f"{timeframe}-"):
                        try:
                            date_str = file.split('.')[0].replace(f"{timeframe}-", "")
                            date_dt = datetime.strptime(date_str, "%Y-%m-%d")
                            file_start_ts = int(date_dt.replace(tzinfo=timezone.utc).timestamp())
                            file_end_ts = int((date_dt + timedelta(days=1)).replace(tzinfo=timezone.utc).timestamp())
                            
                            overlaps = (
                                (start_ts <= file_start_ts <= end_ts) or
                                (start_ts <= file_end_ts <= end_ts) or
                                (file_start_ts <= start_ts and file_end_ts >= end_ts)
                            )
                            if overlaps:
                                file_paths.append(file_path)
                        except ValueError:
                            pass
            
            logger.info(f"Found {len(file_paths)} files for {symbol} {timeframe} in date range")
            return file_paths
            
        except Exception as e:
            logger.error(f"Error in find_files_in_date_range for {symbol} {timeframe}: {str(e)}")
            raise
        
    @staticmethod
    async def load_and_combine_files(
        file_paths: List[str], 
        date_filter: Optional[Tuple[datetime, datetime]] = None
    ) -> pd.DataFrame:
        """Load and combine multiple Parquet files into a single DataFrame"""
        if not file_paths:
            return pd.DataFrame()
            
        # 定义异步加载单个文件的函数
        async def load_file(path):
            try:
                return await FileUtils._async_io_executor(pd.read_parquet, path)
            except Exception as e:
                logger.error(f"Error loading file {path}: {str(e)}")
                return pd.DataFrame()
        
        # 并发加载所有文件
        dfs = await asyncio.gather(*[load_file(path) for path in file_paths])
        
        # 过滤出非空DataFrame
        dfs = [df for df in dfs if not df.empty]
        
        if not dfs:
            return pd.DataFrame()
        
        # 应用日期过滤器
        if date_filter and 'datetime' in dfs[0].columns:
            start_dt, end_dt = map(TimeUtils.ensure_tz_aware, date_filter)
            
            # 将Python datetime转换为pandas Timestamp对象，保证类型兼容
            pd_start = pd.Timestamp(start_dt)
            pd_end = pd.Timestamp(end_dt)
            
            filtered_dfs = []
            for df in dfs:
                # 确保时区感知
                if df['datetime'].dt.tz is None:
                    df = df.copy()
                    df['datetime'] = df['datetime'].dt.tz_localize('UTC')
                # 使用pandas Timestamp对象进行过滤
                filtered_df = df[(df['datetime'] >= pd_start) & (df['datetime'] <= pd_end)]
                if not filtered_df.empty:
                    filtered_dfs.append(filtered_df)
            
            dfs = filtered_dfs
        
        # 合并并排序
        if not dfs:
            return pd.DataFrame()
            
        result = pd.concat(dfs, ignore_index=True)
        
        # 按datetime排序（如果存在）
        if not result.empty and 'datetime' in result.columns:
            result = result.sort_values('datetime').reset_index(drop=True)
        
        return result
        
    @staticmethod
    async def save_dataframe_as_daily_files(
        df: pd.DataFrame, 
        base_path: str, 
        timeframe: str, 
        symbol: str
    ) -> bool:
        """Save DataFrame to daily Parquet files using timeframe-YYYY-MM-DD format"""
        if df.empty or 'datetime' not in df.columns:
            return False
            
        try:
            symbol_safe = symbol.replace('/', '_')
            
            # 添加日期列用于分组
            df = df.copy()
            df['date'] = df['datetime'].dt.date
            
            # 定义异步保存函数
            async def save_daily_file(date, group_df):
                daily_df = group_df.drop(columns=['date'])
                
                # 构建路径
                date_obj = pd.Timestamp(date)
                dir_path = os.path.join(
                    base_path, 
                    timeframe, 
                    symbol_safe, 
                    str(date_obj.year), 
                    f"{date_obj.month:02d}"
                )
                
                # 创建文件名
                filename = f"{timeframe}-{date_obj.year}-{date_obj.month:02d}-{date_obj.day:02d}.parquet"
                file_path = os.path.join(dir_path, filename)
                
                # 创建目录并保存文件
                await FileUtils.async_makedirs(dir_path)
                await FileUtils._async_io_executor(daily_df.to_parquet, file_path, index=False)
                return True
            
            # 并发执行所有保存操作
            tasks = [save_daily_file(date, group) for date, group in df.groupby('date')]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 检查是否有任何保存失败
            return all(not isinstance(r, Exception) for r in results)
                
        except Exception as e:
            logger.error(f"Error saving DataFrame: {str(e)}")
            return False
    
    @staticmethod
    async def save_dataframe_as_timestamp_file(
        df: pd.DataFrame,
        base_path: str,
        timeframe: str,
        symbol: str
    ) -> bool:
        """Save DataFrame to a Parquet file using timestamp-based filename"""
        if df.empty:
            return False
            
        try:
            if 'datetime' not in df.columns:
                raise ValueError("DataFrame must have a datetime column")
                
            # Get min and max timestamps
            min_dt = df['datetime'].min()
            max_dt = df['datetime'].max()
            
            # Ensure timezone-aware
            min_dt = TimeUtils.ensure_tz_aware(min_dt)
            max_dt = TimeUtils.ensure_tz_aware(max_dt)
            
            # Create directory path
            dir_path = FileUtils.timestamp_to_path(base_path, timeframe, symbol, min_dt)
            os.makedirs(dir_path, exist_ok=True)
            
            # Create filename
            filename = FileUtils.generate_timestamp_filename(min_dt, max_dt)
            file_path = os.path.join(dir_path, filename)
            
            # Save file
            df.to_parquet(file_path, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving timestamp dataframe: {str(e)}")
            return False
        
    @staticmethod
    def generate_date_ranges(start_dt: datetime, end_dt: datetime) -> List[Tuple[int, int]]:
        """生成年月组合，用于查找文件"""
        # 确保时区感知
        start_dt = TimeUtils.ensure_tz_aware(start_dt)
        end_dt = TimeUtils.ensure_tz_aware(end_dt)
        
        date_ranges = []
        current = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
        end_month = datetime(end_dt.year, end_dt.month, 1, tzinfo=timezone.utc)
        
        while current <= end_month:
            date_ranges.append((current.year, current.month))
            # 计算下一个月（处理年末情况）
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                current = datetime(current.year, current.month + 1, 1, tzinfo=timezone.utc)
                
        return date_ranges
    
    @staticmethod
    def is_file_in_range(file_start_ts: int, file_end_ts: int, search_start_ts: int, search_end_ts: int) -> bool:
        """判断文件的时间范围是否与搜索范围重叠"""
        # 简化重叠检查逻辑，提高可读性和性能
        return (
            (search_start_ts <= file_start_ts <= search_end_ts) or  # 文件开始在搜索范围内
            (search_start_ts <= file_end_ts <= search_end_ts) or    # 文件结束在搜索范围内
            (file_start_ts <= search_start_ts and file_end_ts >= search_end_ts)  # 文件跨越整个搜索范围
        )
        
    @staticmethod
    async def _process_file_safely(func, file_path, *args, **kwargs):
        """安全处理单个文件，捕获并记录异常"""
        try:
            result = await func(file_path, *args, **kwargs)
            return result
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            # 返回空DataFrame或适当的默认值，而不是抛出异常
            return pd.DataFrame() if 'read' in func.__name__ else False

    @staticmethod
    async def save_dataframe(
        df: pd.DataFrame, 
        base_path: str, 
        timeframe: str, 
        symbol: str
    ) -> bool:
        """
        Save a DataFrame to appropriate parquet files, using a timestamp-based approach
        
        Args:
            df: DataFrame to save
            base_path: Base directory for storing data
            timeframe: Timeframe of the data (e.g., "1h", "1d")
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            bool: Success status
        """
        if df.empty or 'datetime' not in df.columns:
            logger.warning(f"Cannot save empty DataFrame or DataFrame without datetime column")
            return False
            
        try:
            symbol_safe = symbol.replace('/', '_')
            
            # Ensure datetime column is datetime type
            if not pd.api.types.is_datetime64_dtype(df['datetime']):
                df = df.copy()
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Group by year and month for organization
            df = df.copy()
            df['year'] = df['datetime'].dt.year
            df['month'] = df['datetime'].dt.month
            df['day'] = df['datetime'].dt.day
            
            # Group by day for daily files
            success_count = 0
            total_groups = 0
            
            # Process each day's data
            for (year, month, day), day_df in df.groupby(['year', 'month', 'day']):
                total_groups += 1
                
                # Remove grouping columns
                day_df = day_df.drop(['year', 'month', 'day'], axis=1)
                
                # Sort by datetime
                day_df = day_df.sort_values('datetime')
                
                # Create directory path
                dir_path = os.path.join(base_path, timeframe, symbol_safe, str(year), f"{month:02d}")
                await FileUtils.async_makedirs(dir_path)
                
                # Get time range for this day
                min_dt = day_df['datetime'].min()
                max_dt = day_df['datetime'].max()
                
                # Ensure timezone-aware
                min_dt = TimeUtils.ensure_tz_aware(min_dt)
                max_dt = TimeUtils.ensure_tz_aware(max_dt)
                
                # Create timestamp-based filename
                start_ts = int(min_dt.timestamp())
                end_ts = int(max_dt.timestamp())
                filename = f"{start_ts}_{end_ts}.parquet"
                file_path = os.path.join(dir_path, filename)
                
                # Save to parquet file
                try:
                    await FileUtils.async_write_parquet(file_path, day_df)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error saving data to {file_path}: {e}")
            
            logger.info(f"Successfully saved data for {symbol} {timeframe} to {success_count}/{total_groups} files")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error saving dataframe: {e}")
            return False