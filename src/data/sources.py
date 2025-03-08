# src/data/sources.py

from typing import Dict, Optional, List, Union, Tuple
import pandas as pd
import os
import traceback
import json
import asyncio
from datetime import datetime, timedelta

from src.utils.time_utils import TimeUtils
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.exchange.binance import Binance

logger = LogManager.get_logger("trading_system")

class DataSource:
    """数据源抽象基类，定义了获取历史和实时数据的接口"""
    
    async def fetch_historical(self, symbol: str, timeframe: str, 
                              start: Optional[Union[str, datetime]] = None, 
                              end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        获取历史OHLCV数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start: 开始时间
            end: 结束时间
            
        Returns:
            DataFrame: OHLCV数据
        """
        raise NotImplementedError

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        获取实时OHLCV数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            DataFrame: OHLCV数据
        """
        raise NotImplementedError

    async def close(self) -> None:
        """关闭数据源连接（如果适用）"""
        pass
        
    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """将时间框架转换为秒数"""
        units = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        if not timeframe:
            return 60  # 默认为1分钟
            
        # 提取数字和单位
        num = int(''.join(filter(str.isdigit, timeframe)))
        unit = timeframe[-1].lower()
        
        # 检查单位是否有效
        if unit not in units:
            logger.warning(f"未知的时间框架单位: {unit}，使用分钟")
            unit = 'm'
            
        return num * units.get(unit, 60)
        
    @staticmethod
    def get_optimal_data_ranges(start_dt: datetime, end_dt: datetime, 
                              timeframe: str, max_points: int = 1000) -> List[Tuple[datetime, datetime]]:
        """
        将大型日期范围分解为更小的块以优化数据获取
        
        Args:
            start_dt: 开始日期
            end_dt: 结束日期
            timeframe: 时间框架
            max_points: 每个请求的最大数据点数
            
        Returns:
            List[Tuple]: 时间范围列表
        """
        # 计算每个时间框架的秒数
        seconds_per_candle = DataSource.timeframe_to_seconds(timeframe)
        
        # 计算总秒数
        total_seconds = (end_dt - start_dt).total_seconds()
        
        # 估计总数据点数
        estimated_points = total_seconds / seconds_per_candle
        
        # 如果点数少于最大值，返回整个范围
        if estimated_points <= max_points:
            return [(start_dt, end_dt)]
            
        # 计算需要的块数
        num_chunks = int(estimated_points / max_points) + 1
        
        # 计算每个块的大小（秒）
        chunk_seconds = total_seconds / num_chunks
        
        # 创建日期范围列表
        ranges = []
        for i in range(num_chunks):
            chunk_start = start_dt + timedelta(seconds=i * chunk_seconds)
            chunk_end = start_dt + timedelta(seconds=(i+1) * chunk_seconds)
            
            # 确保最后一个块包含终点
            if i == num_chunks - 1:
                chunk_end = end_dt
                
            # 添加到范围列表
            ranges.append((chunk_start, chunk_end))
            
        return ranges

class LocalSource(DataSource):
    """本地文件数据源，支持多种文件格式和智能合并部分数据"""
    
    def __init__(self, config: Dict):
        """
        初始化本地数据源。
        
        Args:
            config (Dict): 配置字典，包含 'data_path' 等
        """
        # 获取历史数据路径
        self.data_path = config.get('data', 'paths', 'historical_data_path')
        if not self.data_path:
            # 尝试其他可能的配置路径
            self.data_path = config.get('data_paths', 'historical_data', default="data/historical")
            
        # 确保目录存在
        if not os.path.exists(self.data_path):
            try:
                os.makedirs(self.data_path)
                logger.info(f"创建历史数据目录: {self.data_path}")
            except Exception as e:
                logger.error(f"创建历史数据目录失败: {str(e)}")
                
        # 异步执行器
        self.executor = AsyncExecutor()
        
        # 添加文件扩展名支持
        self.supported_formats = {
            '.csv': self._read_csv,
            '.parquet': self._read_parquet,
            '.json': self._read_json
        }
        
        # 跟踪数据状态
        self.missing_symbols = set()
        
        logger.info(f"LocalSource初始化完成，数据路径: {self.data_path}")

    async def fetch_historical(self, symbol: str, timeframe: str, 
                              start: Optional[Union[str, datetime]] = None, 
                              end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        从本地文件加载历史数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start: 开始时间
            end: 结束时间
            
        Returns:
            DataFrame: OHLCV数据，如果找不到则返回空DataFrame
            
        Note:
            返回空DataFrame表示数据缺失，调用方可以决定是否从交易所获取数据
        """
        # 确保异步执行器已启动
        await self.executor.start()
        
        # 生成文件名
        safe_symbol = symbol.replace('/', '_')
        
        # 尝试各种可能的文件格式
        df = pd.DataFrame()
        
        for ext, reader_func in self.supported_formats.items():
            file_name = f"{safe_symbol}_{timeframe}{ext}"
            file_path = os.path.join(self.data_path, file_name)
            
            if os.path.exists(file_path):
                try:
                    # 使用异步执行器读取文件
                    df = await self.executor.submit(reader_func, file_path)
                    logger.info(f"从本地文件加载数据: {file_path}")
                    break
                except Exception as e:
                    logger.error(f"读取{file_path}失败: {str(e)}")
                    continue
        
        # 如果没有找到任何文件
        if df.empty:
            # 记录此交易对缺失数据
            self.missing_symbols.add(f"{symbol}_{timeframe}")
            logger.warning(f"本地文件未找到或加载失败: {safe_symbol}_{timeframe}")
            return df
            
        # 确保datetime列的类型正确
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
            df.drop('timestamp', axis=1, inplace=True, errors='ignore')
        else:
            logger.info(f"本地文件缺少datetime列: {safe_symbol}_{timeframe}")
            return pd.DataFrame()
        
        # 如果指定了时间范围，过滤数据
        if start or end:
            start_dt = TimeUtils.parse_timestamp(start) if start else None
            end_dt = TimeUtils.parse_timestamp(end) if end else None
            
            # 应用过滤器
            mask = True
            if start_dt:
                mask &= df['datetime'] >= start_dt
            if end_dt:
                mask &= df['datetime'] <= end_dt
                
            filtered_df = df[mask]
            
            # 如果过滤后的数据为空或太少，可能需要获取更多数据
            if len(filtered_df) < 10:
                logger.warning(f"本地数据在指定时间范围内不足: {symbol} {timeframe} {start}-{end}")
                return pd.DataFrame()  # 返回空DataFrame以触发从交易所获取
                
            df = filtered_df
        
        logger.info(f"从本地加载历史数据: {symbol} {timeframe}, 行数: {len(df)}")
        return df

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        本地源尝试从最新数据中提取最后一条作为"实时"数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            DataFrame: 最后一条OHLCV数据
        """
        # 从历史数据中获取所有数据
        df = await self.fetch_historical(symbol, timeframe)
        
        if df.empty:
            logger.warning(f"LocalSource不能提供实时数据: {symbol} {timeframe}")
            return df
            
        # 返回最后一条数据
        last_row = df.iloc[[-1]].copy()
        logger.info(f"LocalSource提供了最后一条数据作为实时数据: {symbol} {timeframe}")
        return last_row
        
    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """读取CSV文件"""
        return pd.read_csv(file_path)
        
    def _read_parquet(self, file_path: str) -> pd.DataFrame:
        """读取Parquet文件"""
        return pd.read_parquet(file_path)
        
    def _read_json(self, file_path: str) -> pd.DataFrame:
        """读取JSON文件"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
        
    async def get_missing_data_info(self) -> Dict[str, List[str]]:
        """
        获取缺失数据信息
        
        Returns:
            Dict: 按交易对分组的缺失数据信息
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
    """交易所数据源，支持智能分页获取大型历史数据"""
    
    def __init__(self, config: Dict):
        """
        初始化交易所数据源。
        
        Args:
            config (Dict): 配置字典，包含交易所API配置
        """
        self.exchange = Binance(config=config)
        self.executor = AsyncExecutor()
        
        # 加载速率限制配置
        self.max_requests_per_minute = config.get("api", "rate_limits", "requests_per_minute", default=20) if config else 20
        self.request_delay = 60.0 / self.max_requests_per_minute
        
        # 重试配置
        self.max_retries = config.get("api", "retries", "max_attempts", default=3) if config else 3
        self.retry_delay = config.get("api", "retries", "delay_seconds", default=2) if config else 2
        
        logger.info(f"ExchangeSource初始化完成，最大请求频率: {self.max_requests_per_minute}/分钟")

    async def fetch_historical(self, symbol: str, timeframe: str, 
                              start: Optional[Union[str, datetime]] = None, 
                              end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        从交易所获取历史数据，支持智能分块获取大型数据集
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start: 开始时间
            end: 结束时间
            
        Returns:
            DataFrame: OHLCV数据
        """
        logger.info(f"从交易所获取历史数据: {symbol} {timeframe} {start} - {end}")
        
        # 确保执行器已启动
        await self.executor.start()
        
        # 如果没有指定时间范围，获取最近的数据
        if not start and not end:
            try:
                # 使用AsyncExecutor包装同步方法
                df = await self.executor.submit(
                    self.exchange.fetch_ohlcv,
                    symbol=symbol, timeframe=timeframe, limit=1000
                )
                logger.info(f"获取了{symbol} {timeframe}的最近{len(df)}条数据")
                return df
            except Exception as e:
                logger.error(f"获取{symbol}最近数据失败: {str(e)}")
                return pd.DataFrame()
        
        # 解析时间范围
        start_dt = TimeUtils.parse_timestamp(start) if start else datetime.now() - timedelta(days=30)
        end_dt = TimeUtils.parse_timestamp(end) if end else datetime.now()
        
        # 计算合适的数据获取范围
        date_ranges = self.get_optimal_data_ranges(start_dt, end_dt, timeframe)
        logger.info(f"将{symbol} {timeframe}的数据获取分为{len(date_ranges)}个部分")
        
        # 批量获取数据
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(date_ranges):
            try:
                logger.debug(f"获取第{i+1}/{len(date_ranges)}部分: {chunk_start} - {chunk_end}")
                
                # 使用智能获取方法
                chunk_data = await self.exchange.smart_fetch_ohlcv(
                    symbol, timeframe, 
                    chunk_start.strftime('%Y-%m-%d %H:%M:%S'), 
                    chunk_end.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                if not chunk_data.empty:
                    all_data.append(chunk_data)
                    logger.debug(f"第{i+1}部分获取了{len(chunk_data)}行数据")
                
                # 添加延迟以避免超过速率限制
                if i < len(date_ranges) - 1:
                    await asyncio.sleep(self.request_delay)
                    
            except Exception as e:
                logger.error(f"获取第{i+1}部分数据失败: {str(e)}")
                # 继续获取下一部分，而不是完全失败
                continue
        
        # 合并所有数据
        if not all_data:
            logger.warning(f"没有获取到{symbol} {timeframe}的数据")
            return pd.DataFrame()
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # 去除重复数据
        if 'datetime' in combined_df.columns:
            combined_df.drop_duplicates(subset=['datetime'], inplace=True)
            combined_df.sort_values('datetime', inplace=True)
            
        logger.info(f"从交易所获取了{symbol} {timeframe}的{len(combined_df)}行数据")
        return combined_df

    async def fetch_realtime(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        从交易所获取实时数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            
        Returns:
            DataFrame: 最新的OHLCV数据
        """
        logger.info(f"从交易所获取实时数据: {symbol} {timeframe}")
        
        try:
            # 使用AsyncExecutor运行同步方法
            df = await self.executor.submit(
                self.exchange.fetch_ohlcv,
                symbol=symbol, timeframe=timeframe, limit=1
            )
            
            logger.info(f"实时数据获取成功: {symbol} {timeframe}, {len(df)}行")
            return df
            
        except Exception as e:
            logger.error(f"获取实时数据失败: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()

    async def close(self) -> None:
        """关闭交易所连接和执行器"""
        try:
            if hasattr(self.exchange, 'close'):
                await self.exchange.close()
                
            await self.executor.close()
            logger.info("ExchangeSource已关闭")
            
        except Exception as e:
            logger.error(f"关闭ExchangeSource时出错: {str(e)}")

class DataSourceFactory:
    """数据源工厂，负责创建不同类型的数据源"""
    
    @staticmethod
    def create_source(source_type: str, config: Dict) -> DataSource:
        """
        创建数据源实例。
        
        Args:
            source_type (str): 数据源类型 ('local', 'exchange', 或其他)
            config (Dict): 配置字典
        
        Returns:
            DataSource: 数据源实例
        
        Raises:
            ValueError: 如果source_type无效
        """
        sources = {
            'local': LocalSource,
            'exchange': ExchangeSource
        }
        
        # 检查源类型是否有效
        source_type = source_type.lower()
        if source_type not in sources:
            available = ", ".join(sources.keys())
            logger.error(f"未知的数据源类型: {source_type}，可用选项: {available}")
            raise ValueError(f"未知的数据源类型: {source_type}，可用选项: {available}")
            
        # 创建并返回数据源
        try:
            source = sources[source_type](config)
            logger.info(f"创建了{source_type}数据源")
            return source
        except Exception as e:
            logger.error(f"创建{source_type}数据源失败: {str(e)}\n{traceback.format_exc()}")
            raise