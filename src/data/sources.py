# src/data/sources.py

from typing import Dict, Optional
import pandas as pd
import os
import asyncio

from src.utils.time_utils import TimeUtils
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.exchange.binance import Binance

logger = LogManager.get_logger("trading_system")

class DataSource:
    """数据源基类"""
    async def fetch_historical(self, symbol: str, timeframe: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        """获取历史 OHLCV 数据"""
        raise NotImplementedError

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """获取实时 OHLCV 数据"""
        raise NotImplementedError

    async def close(self) -> None:
        """关闭数据源连接（如果适用）"""
        pass

class LocalSource(DataSource):
    """本地文件数据源"""
    def __init__(self, config: Dict):
        """
        初始化本地数据源。
        
        Args:
            config (Dict): 配置字典，包含 'data_path' 等
        """
        self.data_path = config.get('data', 'paths', 'historical_data_path')
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        logger.info("LocalSource initialized with data_path=%s", self.data_path)

    async def fetch_historical(self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """从本地文件加载历史数据"""
        file_name = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        file_path = os.path.join(self.data_path, file_name)
        
        if not os.path.exists(file_path):
            logger.warning("本地文件未找到: %s", file_path)
            return pd.DataFrame()
        
        df = pd.read_csv(file_path)
        if 'datetime' not in df.columns:
            logger.error("本地文件 %s 缺少 'datetime' 列", file_path)
            return pd.DataFrame()
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        if start or end:
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else None
            mask = True
            if start_dt:
                mask &= df['datetime'] >= start_dt
            if end_dt:
                mask &= df['datetime'] <= end_dt
            df = df[mask]
        
        logger.info("从本地加载历史数据: %s %s %s to %s, 行数: %d", symbol, timeframe, start, end, len(df))
        return df

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """本地源不支持实时数据"""
        logger.warning("LocalSource 不支持实时数据获取: %s %s", symbol, timeframe)
        return pd.DataFrame()

class ExchangeSource(DataSource):
    """交易所数据源"""
    def __init__(self, config: Dict):
        """
        初始化交易所数据源。
        
        Args:
            config (Dict): 配置字典，包含 'config_path' 等
        """
        self.exchange = Binance(config=config)
        self.async_executor = AsyncExecutor()
        logger.info("ExchangeSource initialized with Binance")

    async def fetch_historical(self, symbol: str, timeframe: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
        """从交易所获取历史数据"""
        logger.info("从交易所获取历史数据: %s %s %s to %s", symbol, timeframe, start, end)
        if not start or not end:
            df = await asyncio.to_thread(self.exchange.fetch_ohlcv, symbol=symbol, timeframe=timeframe, limit=1000)
        else:
            df = await self.async_executor.run_async(
                self.exchange.smart_fetch_ohlcv(symbol, timeframe, start, end)
            )
        logger.info("历史数据获取完成: %s %s %s to %s, 行数: %d", symbol, timeframe, start, end, len(df))
        return df

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """从交易所获取实时数据"""
        logger.info("从交易所获取实时数据: %s %s", symbol, timeframe)
        df = await self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=1)
        logger.info("实时数据获取完成: %s %s, 行数: %d", symbol, timeframe, len(df))
        return df

    async def close(self) -> None:
        """关闭交易所连接"""
        if hasattr(self.exchange, 'close'):
            await self.exchange.close()
        logger.info("ExchangeSource 已关闭")

class DataSourceFactory:
    """数据源工厂"""
    @staticmethod
    def create_source(source_type: str, config: Dict) -> 'DataSource':
        """
        创建数据源实例。
        
        Args:
            source_type (str): 数据源类型 ('local' 或 'exchange')
            config (Dict): 配置字典
        
        Returns:
            DataSource: 数据源实例
        
        Raises:
            ValueError: 如果 source_type 无效
        """
        sources = {
            'local': LocalSource,
            'exchange': ExchangeSource
        }
        if source_type not in sources:
            raise ValueError(f"未知的数据源类型: {source_type}")
        return sources[source_type](config)