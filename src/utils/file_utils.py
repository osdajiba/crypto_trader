# src\utils\file_utils.py

import asyncio
import os
import json
import csv
import functools
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Callable, Tuple
import pandas as pd
import aiofiles
import pyarrow as pa
import pyarrow.parquet as pq

from .time_utils import TimeUtils
from src.common.log_manager import LogManager

logger = LogManager.get_logger("data.source")

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

