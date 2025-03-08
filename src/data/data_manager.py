# src/data/data_manager.py

import pandas as pd
from enum import Enum
from typing import Optional, Dict, List

from src.common.log_manager import LogManager
from src.common.data_processor import DataProcessor
from .sources import DataSourceFactory

logger = LogManager.get_logger("trading_system")

class TradingMode(Enum):
    """Supported trading modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    
class DataManager:
    """数据管理器，负责数据获取和预处理，支持异步操作"""
    
    def __init__(self, source_type: str, config: Optional[Dict] = None):
        """
        初始化数据管理器。
        
        Args:
            source_type (str): 数据源类型 ('local' 或 'exchange')
            cache_enabled (bool): 是否启用缓存，默认 True
            config (Optional[Dict]): 配置字典，包含 'data_path', 'max_cache_size' 等
        """
        self.source_type = source_type
        self.source = DataSourceFactory.create_source(source_type, config or {})
        
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_enabled = config.get("data", "cache_enabled", default=True)
        self.max_cache_size = config.get('max_cache_size', 100) if config else 100
        self.config = config
        
        logger.info("DataManager initialized with source_type=%s, cache_enabled=%s", source_type, self.cache_enabled)

    async def fetch_all_data_for_symbols(self, symbols: List[str], timeframe: str, 
                                        start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all symbols concurrently.
        
        Args:
            symbols (List[str]): List of symbols to fetch data for
            timeframe (str): Timeframe for data
            start (Optional[str]): Start date for historical data
            end (Optional[str]): End date for historical data
            
        Returns:
            Dict[str, pd.DataFrame]: Market data for each symbol
        """
        # Create tasks for each symbol
        tasks = []
        for symbol in symbols:
            if self.mode == TradingMode.BACKTEST and start and end:
                task = self.get_historical_data(symbol, timeframe, start, end)
            else:
                task = self.get_real_time_data(symbol, timeframe)
            tasks.append(task)
        
        # Execute all tasks concurrently
        from common.async_executor import AsyncExecutor
        async_exec = AsyncExecutor()
        results = await async_exec.gather(*tasks, return_exceptions=True)
        
        # Process results
        data_map = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {symbol}: {result}")
                continue
            data_map[symbol] = result
        
        return data_map

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取历史 OHLCV 数据（用于回测）。
        
        Args:
            symbol (str): 交易对
            timeframe (str): 时间框架
            start (Optional[str]): 开始时间
            end (Optional[str]): 结束时间
            use_cache (bool): 是否使用缓存
        
        Returns:
            pd.DataFrame: OHLCV 数据
        """
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        if self.cache_enabled and use_cache and cache_key in self.cache:
            logger.info("从缓存中获取历史数据: %s", cache_key)
            return self.cache[cache_key]
        
        raw_data = await self.source.fetch_historical(symbol, timeframe, start, end)
        if raw_data.empty:
            logger.warning("获取的历史数据为空: %s", cache_key)
            return raw_data
        
        processed_data = self.preprocess_data(raw_data)
        if self.cache_enabled and use_cache:
            if len(self.cache) >= self.max_cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = processed_data
            logger.info("历史数据已缓存: %s", cache_key)
        return processed_data

    async def get_real_time_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        获取实时 OHLCV 数据（用于 paper/live 模式）。
        
        Args:
            symbol (str): 交易对
            timeframe (str): 时间框架
        
        Returns:
            pd.DataFrame: OHLCV 数据
        """
        raw_data = await self.source.fetch_realtime(symbol, timeframe)
        if raw_data.empty:
            logger.warning("获取的实时数据为空: %s|%s", symbol, timeframe)
            return raw_data
        
        processed_data = self.preprocess_data(raw_data)
        logger.info("实时数据获取完成: %s|%s, 行数=%d", symbol, timeframe, len(processed_data))
        return processed_data

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理流水线。
        
        Args:
            raw_data (pd.DataFrame): 原始数据
        
        Returns:
            pd.DataFrame: 处理后的数据
        """
        if raw_data.empty:
            return raw_data

        # 清洗数据
        processed = DataProcessor.clean_ohlcv(raw_data)

        # 时区标准化
        if 'datetime' in processed.columns:
            timezone = self.config.get("default_config", "misc_config", "timezone", default="Asia/Shanghai") if self.config else "Asia/Shanghai"
            processed['datetime'] = pd.to_datetime(processed['datetime']).dt.tz_convert(timezone)

        # 过滤异常值
        processed = processed[
            (processed['volume'] > 0) &
            (processed['high'] >= processed['low']) &
            (processed['close'].between(processed['low'], processed['high']))
        ].reset_index(drop=True)

        logger.info("数据预处理完成，行数: %d", len(processed))
        return processed

    async def close(self) -> None:
        """关闭数据源连接（如果适用）"""
        if hasattr(self.source, 'close'):
            await self.source.close()
        self.clear_cache()
        logger.info("DataManager 已关闭")

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")