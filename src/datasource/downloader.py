import asyncio
import aiohttp
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

class DataDownloader:
    """
    针对低性能环境优化的交易数据下载器
    专注于稳定性和低资源消耗
    """
    
    def __init__(self, 
                 base_url: str = "https://api.binance.com",
                 futures_url: str = "https://fapi.binance.com",
                 api_key: str = None,
                 api_secret: str = None,
                 max_retries: int = 5,
                 timeout: int = 60,
                 rate_limit: int = 2,  # 每秒请求数
                 chunk_size: int = 500,  # 每次请求数据点数
                 storage_path: str = "data",
                 use_futures: bool = False,
                 log_level: str = "INFO"):
        """初始化下载器"""
        # 基本设置
        self.base_url = futures_url if use_futures else base_url
        self.api_key = api_key
        self.api_secret = api_secret
        self.use_futures = use_futures
        
        # 性能相关设置
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.request_interval = 1.0 / rate_limit
        self.last_request_time = 0
        self.chunk_size = min(chunk_size, 1000)  # Binance最大支持1000
        
        # 存储设置
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # 设置日志
        self._setup_logger(log_level)
        
        # HTTP会话设置
        self.session = None
        self.request_semaphore = asyncio.Semaphore(2)  # 限制并发请求数
        
        self.logger.info(f"数据下载器初始化完成: chunk_size={self.chunk_size}, rate_limit={self.rate_limit}/s")
    
    def _setup_logger(self, log_level: str):
        """配置日志器"""
        self.logger = logging.getLogger("data_downloader")
        self.logger.setLevel(getattr(logging, log_level))
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(module)-18s | [%(filename)s:%(lineno)d] | %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # 添加文件处理器
            os.makedirs("logs", exist_ok=True)
            file_handler = logging.FileHandler("logs/data_download.log")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    async def initialize(self):
        """初始化HTTP会话"""
        if self.session is not None:
            await self.close()
            
        # 创建TCP连接池
        conn = aiohttp.TCPConnector(
            limit=3,  # 最大连接数
            limit_per_host=2,  # 每个主机的最大连接数
            ttl_dns_cache=300,  # DNS缓存TTL
            keepalive_timeout=60,  # 保持连接活跃时间
            force_close=False  # 允许连接重用
        )
        
        # 设置默认超时
        timeout = aiohttp.ClientTimeout(
            total=self.timeout,
            sock_connect=30,
            sock_read=30
        )
        
        # 创建会话
        self.session = aiohttp.ClientSession(
            connector=conn,
            timeout=timeout,
            headers={
                'User-Agent': 'OptimizedDataDownloader/1.0',
                'Accept': 'application/json',
                'Connection': 'keep-alive'
            }
        )
        
        self.logger.info("HTTP会话初始化完成")
        return self.session
    
    async def close(self):
        """关闭会话和资源"""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("HTTP会话已关闭")
    
    def _get_endpoint(self, symbol: str, timeframe: str) -> str:
        """构建API端点URL"""
        if self.use_futures:
            return f"{self.base_url}/fapi/v1/klines"
        else:
            return f"{self.base_url}/api/v3/klines"
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """将时间周期转换为Binance格式"""
        # 已经符合Binance格式则直接返回
        if timeframe in ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']:
            return timeframe
            
        # 转换其他格式
        mapping = {
            '1min': '1m', '3min': '3m', '5min': '5m', '15min': '15m', '30min': '30m',
            '1hour': '1h', '2hour': '2h', '4hour': '4h', '6hour': '6h', '12hour': '12h',
            '1day': '1d', '3day': '3d', '1week': '1w', '1month': '1M'
        }
        return mapping.get(timeframe, '1h')  # 默认返回1h
    
    def _calculate_chunks(self, start_time: int, end_time: int, timeframe: str) -> List[Tuple[int, int]]:
        """
        将日期范围分成合适的数据块
        
        Args:
            start_time: 开始时间戳(毫秒)
            end_time: 结束时间戳(毫秒)
            timeframe: 数据时间周期
            
        Returns:
            日期范围块列表
        """
        # 计算单个时间周期的毫秒数
        tf_milliseconds = {
            '1m': 60000, '3m': 180000, '5m': 300000, '15m': 900000, '30m': 1800000,
            '1h': 3600000, '2h': 7200000, '4h': 14400000, '6h': 21600000, '8h': 28800000, 
            '12h': 43200000, '1d': 86400000, '3d': 259200000, '1w': 604800000, '1M': 2592000000
        }
        
        ms_per_candle = tf_milliseconds.get(self._timeframe_to_interval(timeframe), 3600000)
        
        # 计算单个块的时间范围 (chunk_size个蜡烛图)
        chunk_ms = ms_per_candle * self.chunk_size
        
        # 创建时间块
        chunks = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + chunk_ms, end_time)
            chunks.append((current_start, current_end))
            current_start = current_end
            
        self.logger.info(f"将时间范围分割为{len(chunks)}个块")
        return chunks
    
    async def _wait_rate_limit(self):
        """
        遵守速率限制(使用令牌桶算法)
        """
        now = time.time()
        elapsed = now - self.last_request_time
        
        if elapsed < self.request_interval:
            delay = self.request_interval - elapsed
            await asyncio.sleep(delay)
            
        self.last_request_time = time.time()
    
    async def _fetch_chunk(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List]:
        """
        获取单个数据块
        
        Args:
            symbol: 交易对
            interval: 时间周期
            start_ms: 开始时间戳(毫秒)
            end_ms: 结束时间戳(毫秒)
            
        Returns:
            OHLCV数据列表
        """
        endpoint = self._get_endpoint(symbol, interval)
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': self.chunk_size
        }
        
        # 避免速率限制
        await self._wait_rate_limit()
        
        # 限制并发请求数量
        async with self.request_semaphore:
            # 重试机制
            for attempt in range(1, self.max_retries + 1):
                try:
                    async with self.session.get(endpoint, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.logger.debug(f"成功获取数据: {symbol} {interval} ({len(data)}条记录)")
                            return data
                        else:
                            error_text = await response.text()
                            self.logger.warning(f"API请求失败(尝试 {attempt}/{self.max_retries}): HTTP {response.status}, {error_text}")
                            
                            # 特殊处理某些错误
                            if response.status == 429:  # 速率限制
                                wait_time = int(response.headers.get('Retry-After', 5))
                                self.logger.warning(f"遇到速率限制, 等待{wait_time}秒")
                                await asyncio.sleep(wait_time)
                            elif response.status >= 500:  # 服务器错误
                                await asyncio.sleep(2 ** attempt)  # 指数退避
                            else:
                                await asyncio.sleep(1)  # 其他错误暂停1秒
                
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.logger.warning(f"请求异常(尝试 {attempt}/{self.max_retries}): {e}")
                    # 指数退避 + 随机抖动
                    backoff = (2 ** attempt) + (0.1 * np.random.random())
                    await asyncio.sleep(backoff)
            
            # 所有重试都失败
            self.logger.error(f"所有重试都失败: {symbol} {interval}")
            return []
    
    def _parse_ohlcv(self, data: List[List]) -> pd.DataFrame:
        """解析OHLCV数据成DataFrame"""
        if not data:
            return pd.DataFrame()
            
        # Binance API返回格式:
        # [
        #   [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, 主动买入成交量, 主动买入成交额, 忽略]
        # ]
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignored'
        ])
        
        # 转换类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 转换时间戳
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 只保留必要列
        keep_cols = ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
        return df[keep_cols]
    
    async def download_data(self, 
                           symbol: str, 
                           timeframe: str, 
                           start_date: Union[str, datetime, int] = None,
                           end_date: Union[str, datetime, int] = None) -> pd.DataFrame:
        """
        下载历史OHLCV数据
        
        Args:
            symbol: 交易对 (如 "BTC/USDT")
            timeframe: 时间周期 (如 "1m", "1h")
            start_date: 开始日期 (字符串, datetime或时间戳)
            end_date: 结束日期 (字符串, datetime或时间戳)
            
        Returns:
            DataFrame: OHLCV数据
        """
        # 确保会话已初始化
        if self.session is None:
            await self.initialize()
            
        # 处理日期
        if start_date is None:
            start_date = datetime.now() - timedelta(days=7)
            
        if end_date is None:
            end_date = datetime.now()
            
        # 转换为时间戳(毫秒)
        start_ms = self._to_timestamp_ms(start_date)
        end_ms = self._to_timestamp_ms(end_date)
        
        # 标准化时间周期
        interval = self._timeframe_to_interval(timeframe)
        
        self.logger.info(f"开始下载 {symbol} {interval} 数据: {datetime.fromtimestamp(start_ms/1000)} 到 {datetime.fromtimestamp(end_ms/1000)}")
        
        # 分块下载
        chunks = self._calculate_chunks(start_ms, end_ms, interval)
        all_data = []
        
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            self.logger.debug(f"下载块 {i+1}/{len(chunks)}: {datetime.fromtimestamp(chunk_start/1000)} 到 {datetime.fromtimestamp(chunk_end/1000)}")
            chunk_data = await self._fetch_chunk(symbol, interval, chunk_start, chunk_end)
            all_data.extend(chunk_data)
            
            # 避免长时间运行的请求导致的资源泄漏
            if i > 0 and i % 10 == 0:
                # 每10个块后强制GC
                import gc
                gc.collect()
        
        # 解析数据
        df = self._parse_ohlcv(all_data)
        
        if df.empty:
            self.logger.warning(f"未获取到任何数据: {symbol} {interval}")
            return df
            
        # 排序和去重
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])
        
        self.logger.info(f"成功下载 {symbol} {interval} 数据: {len(df)}条记录, {df['datetime'].min()} 到 {df['datetime'].max()}")
        return df
    
    def _to_timestamp_ms(self, date_input: Union[str, datetime, int]) -> int:
        """
        将各种日期格式转换为毫秒时间戳
        
        Args:
            date_input: 输入日期(字符串、datetime或时间戳)
            
        Returns:
            int: 毫秒时间戳
        """
        if isinstance(date_input, int):
            # 确保是毫秒级
            if date_input < 10000000000:  # 秒级时间戳
                return date_input * 1000
            return date_input
            
        elif isinstance(date_input, datetime):
            return int(date_input.timestamp() * 1000)
            
        elif isinstance(date_input, str):
            try:
                dt = pd.to_datetime(date_input)
                return int(dt.timestamp() * 1000)
            except:
                raise ValueError(f"无法解析日期: {date_input}")
                
        else:
            raise TypeError(f"不支持的日期类型: {type(date_input)}")
    
    async def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str, 
                      format: str = 'parquet', append: bool = True) -> str:
        """
        保存数据到文件
        
        Args:
            df: OHLCV数据
            symbol: 交易对
            timeframe: 时间周期
            format: 保存格式 ('parquet', 'csv')
            append: 是否附加到现有文件
            
        Returns:
            str: 保存的文件路径
        """
        if df.empty:
            self.logger.warning(f"尝试保存空数据: {symbol} {timeframe}")
            return None
            
        # 创建目录结构
        symbol_dir = os.path.join(self.storage_path, symbol.replace('/', '_'), timeframe)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # 文件路径
        safe_symbol = symbol.replace('/', '_')
        filename = f"{safe_symbol}_{timeframe}"
        
        if format.lower() == 'parquet':
            file_path = os.path.join(symbol_dir, f"{filename}.parquet")
            
            # 如果需要附加
            if append and os.path.exists(file_path):
                try:
                    existing_df = pd.read_parquet(file_path)
                    # 合并并去重
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])
                    combined_df.to_parquet(file_path, index=False)
                    self.logger.info(f"更新数据到: {file_path} (总共 {len(combined_df)} 条记录)")
                except Exception as e:
                    self.logger.error(f"合并现有数据时出错: {e}")
                    # 如果合并失败, 保存为新文件
                    file_path = os.path.join(symbol_dir, f"{filename}_{int(time.time())}.parquet")
                    df.to_parquet(file_path, index=False)
                    self.logger.info(f"保存数据到新文件: {file_path} ({len(df)} 条记录)")
            else:
                df.to_parquet(file_path, index=False)
                self.logger.info(f"保存数据到: {file_path} ({len(df)} 条记录)")
                
        elif format.lower() == 'csv':
            file_path = os.path.join(symbol_dir, f"{filename}.csv")
            
            # 如果需要附加
            if append and os.path.exists(file_path):
                try:
                    existing_df = pd.read_csv(file_path, parse_dates=['datetime'])
                    # 合并并去重
                    combined_df = pd.concat([existing_df, df])
                    combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])
                    combined_df.to_csv(file_path, index=False)
                    self.logger.info(f"更新数据到: {file_path} (总共 {len(combined_df)} 条记录)")
                except Exception as e:
                    self.logger.error(f"合并现有数据时出错: {e}")
                    # 如果合并失败, 保存为新文件
                    file_path = os.path.join(symbol_dir, f"{filename}_{int(time.time())}.csv")
                    df.to_csv(file_path, index=False)
                    self.logger.info(f"保存数据到新文件: {file_path} ({len(df)} 条记录)")
            else:
                df.to_csv(file_path, index=False)
                self.logger.info(f"保存数据到: {file_path} ({len(df)} 条记录)")
                
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        return file_path
    
    async def download_and_save(self, symbol: str, timeframe: str, 
                             start_date: Union[str, datetime, int] = None,
                             end_date: Union[str, datetime, int] = None,
                             format: str = 'parquet', append: bool = True) -> bool:
        """
        下载并保存数据的便捷方法
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            format: 保存格式
            append: 是否附加到现有文件
            
        Returns:
            bool: 是否成功
        """
        try:
            df = await self.download_data(symbol, timeframe, start_date, end_date)
            
            if df.empty:
                self.logger.warning(f"未获取到数据, 无法保存: {symbol} {timeframe}")
                return False
                
            file_path = await self.save_data(df, symbol, timeframe, format, append)
            return file_path is not None
            
        except Exception as e:
            self.logger.error(f"下载保存数据过程中出错: {e}", exc_info=True)
            return False
    
    async def download_many(self, symbols: List[str], timeframes: List[str],
                          start_date: Union[str, datetime, int] = None,
                          end_date: Union[str, datetime, int] = None,
                          format: str = 'parquet', 
                          append: bool = True,
                          max_concurrency: int = 2) -> Dict[str, bool]:
        """
        下载多个交易对和时间周期的数据
        
        Args:
            symbols: 交易对列表
            timeframes: 时间周期列表
            start_date: 开始日期
            end_date: 结束日期
            format: 保存格式
            append: 是否附加到现有文件
            max_concurrency: 最大并发数
            
        Returns:
            Dict: 每个任务的结果
        """
        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                tasks.append((symbol, timeframe))
                
        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def download_with_semaphore(symbol, timeframe):
            async with semaphore:
                key = f"{symbol}_{timeframe}"
                try:
                    success = await self.download_and_save(
                        symbol, timeframe, start_date, end_date, format, append
                    )
                    results[key] = success
                except Exception as e:
                    self.logger.error(f"下载任务失败 {key}: {e}")
                    results[key] = False
        
        # 创建所有任务
        download_tasks = [download_with_semaphore(s, tf) for s, tf in tasks]
        
        # 执行所有任务
        await asyncio.gather(*download_tasks)
        
        # 统计成功和失败的数量
        success_count = sum(1 for success in results.values() if success)
        fail_count = len(results) - success_count
        
        self.logger.info(f"批量下载完成: {success_count}个成功, {fail_count}个失败")
        return results


# 使用示例
async def main():
    # 创建下载器
    downloader = DataDownloader(
        chunk_size=10000,  # 每次请求10000条数据
        rate_limit=3,    # 每秒最多3个请求
        storage_path="data/binance",
        log_level="INFO"
    )
    
    try:
        # 初始化会话
        await downloader.initialize()
        
        # 下载Bitcoin 1分钟数据(过去7天)
        symbol = "BTC/USDT"
        timeframe = "1m"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        # 单个交易对下载
        success = await downloader.download_and_save(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            format="parquet",
            append=True
        )
        
        print(f"下载状态: {'成功' if success else '失败'}")
        
        # 多个交易对和时间周期批量下载
        symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
        timeframes = ["1m", "5m", "1h"]
        
        results = await downloader.download_many(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date, 
            end_date=end_date,
            max_concurrency=1  # 最多1个并发任务
        )
        
        # 打印结果
        for key, success in results.items():
            print(f"{key}: {'成功' if success else '失败'}")
            
    finally:
        # 确保资源被释放
        await downloader.close()

# 运行示例
if __name__ == "__main__":
    asyncio.run(main())