# src\utils\file_utils.py

import asyncio
import os
import json
import csv
import functools
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
import pandas as pd
import aiofiles
import pyarrow as pa
import pyarrow.parquet as pq

class FileUtils:
    """优化后的文件操作工具类，修复异步目录创建问题"""
    
    # 配置常量
    PARQUET_COMPRESSION = 'SNAPPY'
    CSV_ENCODING = 'utf-8'
    DEFAULT_CHUNK_SIZE = 1024 * 1024  # 1MB

    @staticmethod
    async def _async_io_executor(func: Callable, *args, **kwargs) -> Any:
        """通用异步IO执行器"""
        loop = asyncio.get_running_loop()
        bound_func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(None, bound_func)

    @staticmethod
    async def async_makedirs(path: str) -> None:
        """修复后的异步目录创建"""
        await FileUtils._async_io_executor(os.makedirs, path, exist_ok=True)

    @staticmethod
    async def async_write_parquet(file_path: str, df: pd.DataFrame) -> None:
        """优化后的异步Parquet写入"""
        try:
            dir_path = os.path.dirname(file_path)
            await FileUtils.async_makedirs(dir_path)

            # 时区处理优化
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
            raise IOError(f"Parquet写入失败: {str(e)}") from e

    @staticmethod
    async def async_read_parquet(file_path: str) -> pd.DataFrame:
        """优化的异步Parquet读取"""
        try:
            table = await FileUtils._async_io_executor(pq.read_table, file_path)
            return table.to_pandas()
        except Exception as e:
            raise IOError(f"Parquet读取失败: {str(e)}") from e

    @staticmethod
    def generate_timestamp_filename(start: datetime, end: datetime) -> str:
        """时间戳文件名生成优化"""
        return f"{int(start.timestamp())}_{int(end.timestamp())}.parquet"

    @staticmethod
    async def _async_file_operation(mode: str, file_path: str, 
                                   data: Optional[Any] = None) -> Any:
        """通用异步文件操作"""
        await FileUtils.async_makedirs(os.path.dirname(file_path))
        
        if mode == 'read':
            async with aiofiles.open(file_path, 'r', encoding=FileUtils.CSV_ENCODING) as f:
                return await f.read()
        
        async with aiofiles.open(file_path, 'w', encoding=FileUtils.CSV_ENCODING) as f:
            await f.write(data)

    @staticmethod
    async def async_read_json(file_path: str) -> Dict[str, Any]:
        """优化的JSON读取"""
        content = await FileUtils._async_file_operation('read', file_path)
        return json.loads(content)

    @staticmethod
    async def async_write_json(file_path: str, data: Dict[str, Any]) -> None:
        """优化的JSON写入"""
        await FileUtils._async_file_operation('write', file_path, json.dumps(data, indent=2))

    @staticmethod
    async def async_read_csv(file_path: str) -> List[Dict[str, str]]:
        """优化的CSV读取"""
        content = await FileUtils._async_file_operation('read', file_path)
        return list(csv.DictReader(content.splitlines()))

    @staticmethod
    async def async_write_csv(file_path: str, data: List[Dict[str, Any]]) -> None:
        """优化的CSV写入"""
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
        """自动同步包装装饰器"""
        @functools.wraps(async_func)
        def wrapper(*args, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(async_func(*args, **kwargs))
            finally:
                loop.close()
        return wrapper
