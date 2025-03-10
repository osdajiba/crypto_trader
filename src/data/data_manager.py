# src/data/data_manager.py

import pandas as pd
import os
import traceback
from enum import Enum
from typing import Optional, Dict, List, Union
from datetime import datetime
import aiofiles

from src.common.log_manager import LogManager
from src.common.data_processor import DataProcessor
from src.common.async_executor import AsyncExecutor
from src.data.integrity_checker import DataIntegrityChecker
from .sources import DataSourceFactory

logger = LogManager.get_logger("trading_system")

class TradingMode(Enum):
    """支持的交易模式"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    
class DataManager:
    """
    增强的数据管理器，负责数据获取和预处理
    
    支持自动下载缺失的历史数据、数据完整性检查和异步操作。
    """
    
    def __init__(self, source_type: str, config: Optional[Dict] = None, trading_mode: TradingMode = TradingMode.BACKTEST):
        """
        初始化数据管理器。
        
        Args:
            source_type (str): 数据源类型 ('local' 或 'exchange')
            config (Optional[Dict]): 配置字典，包含 'data_path', 'max_cache_size' 等
            trading_mode (TradingMode): 交易模式，影响数据获取策略
        """
        self.source_type = source_type
        self.config = config
        self.mode = trading_mode
        
        # 创建数据源
        self.primary_source = DataSourceFactory.create_source(source_type, config or {})
        
        # 如果主数据源是本地，创建备用交易所数据源用于回退
        self.fallback_source = None
        if source_type == 'local':
            try:
                self.fallback_source = DataSourceFactory.create_source('exchange', config or {})
                logger.info("创建了交易所备用数据源用于自动下载缺失数据")
            except Exception as e:
                logger.warning(f"无法创建备用交易所数据源: {str(e)}")
        
        # 缓存设置
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_enabled = config.get("data", "cache_enabled", default=True) if config else True
        self.max_cache_size = config.get("data", "max_cache_size", default=100) if config else 100
        
        # 数据完整性检查
        self.integrity_check_enabled = config.get("data", "integrity_check", "enabled", default=True) if config else True
        
        # 异步执行器
        self.executor = AsyncExecutor()
        
        # 数据路径配置
        self.data_path = config.get("data", "storage", "historical") if config else "db/historical"
        if not os.path.exists(self.data_path):
            try:
                os.makedirs(self.data_path)
                logger.info(f"创建数据目录: {self.data_path}")
            except Exception as e:
                logger.error(f"创建数据目录失败 {self.data_path}: {str(e)}")
        
        logger.info(f"DataManager初始化完成: 数据源={source_type}, 缓存={self.cache_enabled}, 交易模式={self.mode.name}")

    async def fetch_all_data_for_symbols(self, symbols: List[str], timeframe: str, 
                                         start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        并发获取所有交易对的数据。
        
        Args:
            symbols (List[str]): 要获取数据的交易对列表
            timeframe (str): 时间框架
            start (Optional[str]): 历史数据的开始时间
            end (Optional[str]): 历史数据的结束时间
            
        Returns:
            Dict[str, pd.DataFrame]: 每个交易对的市场数据字典
        """
        # 确保执行器已启动
        await self.executor.start()
        
        # 为每个交易对创建任务
        tasks = []
        for symbol in symbols:
            if self.mode == TradingMode.BACKTEST and start and end:
                # 使用executor.task_context进行任务跟踪
                async with self.executor.task_context(f"fetch_{symbol}") as task_id:
                    task = self.get_historical_data(symbol, timeframe, start, end, auto_download=True)
                    tasks.append(task)
            else:
                # 实时模式或没有时间范围的情况
                async with self.executor.task_context(f"fetch_rt_{symbol}") as task_id:
                    task = self.get_real_time_data(symbol, timeframe)
                    tasks.append(task)
        
        # 并发执行所有任务
        results = []
        for task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception as e:
                logger.error(f"获取数据失败: {str(e)}\n{traceback.format_exc()}")
                results.append(pd.DataFrame())  # 失败时添加空DataFrame
        
        # 处理结果
        data_map = {}
        for symbol, result in zip(symbols, results):
            data_map[symbol] = result
            logger.info(f"获取到{symbol}的数据: {len(result)}行")
        
        return data_map

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
        auto_download: bool = True
    ) -> pd.DataFrame:
        """
        获取历史OHLCV数据，自动从本地或交易所获取。
        
        Args:
            symbol (str): 交易对
            timeframe (str): 时间框架
            start (Optional[str]): 开始时间
            end (Optional[str]): 结束时间
            use_cache (bool): 是否使用缓存
            auto_download (bool): 如果本地数据不存在，是否自动从交易所下载
        
        Returns:
            pd.DataFrame: OHLCV数据
        """
        cache_key = f"{symbol}|{timeframe}|{start}|{end}"
        
        # 1. 尝试从缓存获取
        if self.cache_enabled and use_cache and cache_key in self.cache:
            logger.info(f"从缓存中获取历史数据: {cache_key}")
            return self.cache[cache_key]
        
        # 2. 尝试从主数据源获取
        raw_data = await self.primary_source.fetch_historical(symbol, timeframe, start, end)
        
        # 3. 如果主数据源没有数据且允许自动下载，尝试从交易所获取
        if (raw_data.empty or len(raw_data) < 2) and auto_download and self.fallback_source is not None:
            logger.warning(f"本地历史数据不存在或不完整: {symbol} {timeframe} {start}-{end}，尝试从交易所下载")
            try:
                exchange_data = await self.fallback_source.fetch_historical(symbol, timeframe, start, end)
                
                if not exchange_data.empty:
                    logger.info(f"从交易所成功下载 {symbol} {timeframe} 数据: {len(exchange_data)}行")
                    raw_data = exchange_data
                    
                    # 保存到本地文件
                    await self._save_to_local(symbol, timeframe, exchange_data)
                else:
                    logger.error(f"无法从交易所获取 {symbol} {timeframe} 数据")
            except Exception as e:
                logger.error(f"从交易所下载数据失败: {str(e)}\n{traceback.format_exc()}")
        
        # 4. 如果仍然没有数据，返回空DataFrame
        if raw_data.empty:
            logger.warning(f"无法获取历史数据: {symbol} {timeframe} {start}-{end}")
            return pd.DataFrame()
        
        # 5. 数据处理和完整性检查
        processed_data = await self._process_and_validate(raw_data, timeframe)
        
        # 6. 更新缓存
        if self.cache_enabled and use_cache:
            if len(self.cache) >= self.max_cache_size:
                # 移除最早添加的缓存项
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = processed_data
            logger.info(f"历史数据已缓存: {cache_key}")
            
        return processed_data

    async def get_real_time_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        获取实时OHLCV数据（用于paper/live模式）。
        
        Args:
            symbol (str): 交易对
            timeframe (str): 时间框架
        
        Returns:
            pd.DataFrame: OHLCV数据
        """
        raw_data = await self.primary_source.fetch_realtime(symbol, timeframe)
        
        # 如果主数据源没有数据且有备用数据源，尝试从备用源获取
        if raw_data.empty and self.fallback_source is not None:
            logger.warning(f"主数据源无法获取实时数据: {symbol} {timeframe}，尝试备用数据源")
            try:
                raw_data = await self.fallback_source.fetch_realtime(symbol, timeframe)
            except Exception as e:
                logger.error(f"从备用数据源获取实时数据失败: {str(e)}")
        
        if raw_data.empty:
            logger.warning(f"获取的实时数据为空: {symbol}|{timeframe}")
            return pd.DataFrame()
        
        # 处理和验证数据
        processed_data = self.preprocess_data(raw_data)
        logger.info(f"实时数据获取完成: {symbol}|{timeframe}, 行数={len(processed_data)}")
        return processed_data

    async def _process_and_validate(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        处理并验证数据质量。
        
        Args:
            data (pd.DataFrame): 原始数据
            timeframe (str): 时间框架，用于完整性检查
            
        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 预处理数据
        processed = self.preprocess_data(data)
        
        # 如果启用了完整性检查，执行检查
        if self.integrity_check_enabled and not processed.empty:
            try:
                checker = DataIntegrityChecker(timeframe, self.config)
                check_results = await checker.check(processed)
                
                # 记录完整性问题
                stats = check_results.get('stats', {})
                if stats.get('missing_count', 0) > 0 or stats.get('outlier_count', 0) > 0:
                    logger.warning(f"数据完整性问题: 缺失={stats.get('missing_count', 0)}, 异常值={stats.get('outlier_count', 0)}")
                    
                # 如果需要，可以在这里添加修复数据的逻辑
            except Exception as e:
                logger.error(f"数据完整性检查失败: {str(e)}")
        
        return processed

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
        if hasattr(DataProcessor, 'clean_ohlcv'):
            processed = DataProcessor.clean_ohlcv(raw_data)
        else:
            # 如果DataProcessor不可用，使用基本清洗
            processed = raw_data.copy()
            
            # 标准化列名
            column_map = {
                'open_time': 'datetime', 'timestamp': 'datetime',
                'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'
            }
            processed.rename(columns={k: v for k, v in column_map.items() if k in processed.columns}, inplace=True)

        # 时区标准化
        if 'datetime' in processed.columns:
            timezone = self.config.get("default_config", "misc_config", "timezone", default="Asia/Shanghai") if self.config else "Asia/Shanghai"
            try:
                processed['datetime'] = pd.to_datetime(processed['datetime'])
                # 检查是否已经有时区信息
                if processed['datetime'].dt.tz is None:
                    processed['datetime'] = processed['datetime'].dt.tz_localize('UTC').dt.tz_convert(timezone)
                elif str(processed['datetime'].dt.tz) != timezone:
                    processed['datetime'] = processed['datetime'].dt.tz_convert(timezone)
            except Exception as e:
                logger.warning(f"时区转换失败: {str(e)}")

        # 过滤异常值
        try:
            processed = processed[
                (processed['volume'] > 0) &
                (processed['high'] >= processed['low']) &
                (processed['close'].between(processed['low'], processed['high']))
            ].reset_index(drop=True)
        except Exception as e:
            logger.warning(f"过滤异常值失败: {str(e)}")

        logger.info(f"数据预处理完成，行数: {len(processed)}")
        return processed

    async def _save_to_local(self, symbol: str, timeframe: str, data: pd.DataFrame) -> bool:
        # 修改文件路径生成逻辑
        def generate_file_path(row: pd.Series) -> str:
            ts = row.name if isinstance(row.name, pd.Timestamp) else row['datetime']
            start_ts = ts.floor('D')  # 按天分片
            end_ts = start_ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            dir_path = os.path.join(
                self.data_path,
                timeframe,
                symbol.replace('/', '_'),
                f"{start_ts.year:04d}",
                f"{start_ts.month:02d}"
            )
            return os.path.join(
                dir_path,
                f"{start_ts.timestamp():.0f}_{end_ts.timestamp():.0f}.parquet"
            )

        try:
            # 新增分片保存逻辑
            grouped = data.groupby(pd.Grouper(freq='D'))
            
            for day, daily_df in grouped:
                if daily_df.empty:
                    continue
                
                file_path = generate_file_path(daily_df.iloc[0])
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 使用列式存储
                daily_df = daily_df.reset_index()
                daily_df['start_ts'] = day.timestamp()
                daily_df['end_ts'] = (day + pd.Timedelta(days=1)).timestamp() - 1
                
                # 异步保存
                async def async_save():
                    daily_df.to_parquet(
                        file_path,
                        engine='pyarrow',
                        compression='snappy',
                        index=False
                    )
                await self.executor.submit(async_save)
                
            return True
        except Exception as e:
            logger.error(f"保存失败: {str(e)}")
            return False

    async def close(self) -> None:
        """关闭数据源连接和清理资源"""
        # 关闭主数据源
        if hasattr(self.primary_source, 'close'):
            await self.primary_source.close()
            
        # 关闭备用数据源
        if self.fallback_source is not None and hasattr(self.fallback_source, 'close'):
            await self.fallback_source.close()
            
        # 清空缓存
        self.clear_cache()
        
        # 关闭执行器
        await self.executor.close()
        
        logger.info("DataManager 已关闭")

    def clear_cache(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")
        
    async def update_local_data(self, symbol: str, timeframe: str, 
                               start: Optional[Union[str, datetime]] = None, 
                               end: Optional[Union[str, datetime]] = None) -> bool:
        """
        更新本地历史数据，从交易所获取最新数据并保存。
        
        Args:
            symbol (str): 交易对
            timeframe (str): 时间框架
            start (Optional[Union[str, datetime]]): 开始时间
            end (Optional[Union[str, datetime]]): 结束时间
            
        Returns:
            bool: 更新成功返回True，否则返回False
        """
        if self.fallback_source is None:
            logger.error("无法更新本地数据: 未配置交易所数据源")
            return False
            
        try:
            # 从交易所获取数据
            logger.info(f"从交易所更新 {symbol} {timeframe} 数据: {start} - {end}")
            exchange_data = await self.fallback_source.fetch_historical(symbol, timeframe, start, end)
            
            if exchange_data.empty:
                logger.warning(f"从交易所获取的数据为空: {symbol} {timeframe}")
                return False
                
            # 保存到本地
            success = await self._save_to_local(symbol, timeframe, exchange_data)
            return success
            
        except Exception as e:
            logger.error(f"更新本地数据失败: {str(e)}\n{traceback.format_exc()}")
            return False