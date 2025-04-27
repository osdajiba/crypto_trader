# src/exchange/binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import random
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
from functools import wraps

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import TimeUtils

# 获取或创建日志记录器
logger = LogManager.get_logger("trading_system")

def retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0):
    """带有指数退避和随机抖动的重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.warning(f"尝试 {attempt + 1}/{max_attempts} 失败: {str(e)}")
                    last_exception = e
                    
                    # 计算指数退避延迟加随机抖动
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% 随机抖动
                    adjusted_delay = delay * jitter
                    
                    logger.info(f"等待 {adjusted_delay:.2f}秒后重试...")
                    await asyncio.sleep(adjusted_delay)
                except Exception as e:
                    logger.error(f"未处理的错误: {str(e)}")
                    raise
            
            # 所有尝试都失败
            logger.error(f"所有尝试均失败。最后错误: {last_exception}")
            raise last_exception
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    logger.warning(f"尝试 {attempt + 1}/{max_attempts} 失败: {str(e)}")
                    last_exception = e
                    
                    # 计算指数退避延迟加随机抖动
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% 随机抖动
                    adjusted_delay = delay * jitter
                    
                    logger.info(f"等待 {adjusted_delay:.2f}秒后重试...")
                    time.sleep(adjusted_delay)
                except Exception as e:
                    logger.error(f"未处理的错误: {str(e)}")
                    raise
            
            # 所有尝试都失败
            logger.error(f"所有尝试均失败。最后错误: {last_exception}")
            raise last_exception
        
        # 根据函数是否是协程选择适当的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class Binance:
    """针对高延迟网络优化的Binance交易所接口"""
    
    def __init__(self, config: ConfigManager):
        """初始化Binance接口，使用配置"""
        self.config = config
        
        # 设置缓存目录
        self.cache_dir = self._get_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 提取配置参数
        self.params = self._build_params()
        
        # 初始化交易所对象
        self.exchange = None
        self.async_exchange = None
        
        # 设置并发限制信号量
        self.request_semaphore = None
        
        # 初始化交易所
        self._init_exchange()
        
        # 设置
        self.rate_limit = self.config.get('api', 'rate_limits', 'requests', default=20)
        self.current_requests = 0
        self.rate_limit_reset = time.time() + 60
        self.download_chunk_size = self.config.get('api', 'download_chunk_size', default=500)
        self.max_retry_attempts = self.config.get('network', 'connection', 'retry_attempts', default=3)
        self.retry_delay = self.config.get('network', 'connection', 'retry_delay', default=2.0)
        
        # WebSocket状态跟踪
        self.ws_subscriptions = {}
        
        logger.info("Binance接口已初始化，针对高延迟网络进行了优化")
    
    def _get_cache_dir(self) -> str:
        """从配置中获取缓存目录"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        return os.path.abspath(os.path.expanduser(cache_dir))
    
    def _build_params(self) -> Dict:
        """
        从配置构建CCXT参数，增强安全性和网络性能
        
        Returns:
        CCXT参数字典
        """
        params = {
            'apiKey': self.config.get('api', 'binance', 'api_key', default=''),
            'secret': self.config.get('api', 'binance', 'secret', default=''),
            'timeout': self.config.get('api', 'timeout', default=60000),  # 增加到60秒
            'enableRateLimit': self.config.get('api', 'enable_rate_limit', default=True),
            'options': {
                'adjustForTimeDifference': self.config.get(
                    'api', 'binance', 'options', 'adjust_time_diff', default=True),
                'recvWindow': self.config.get(
                    'api', 'binance', 'options', 'recv_window', default=60000),  # 最大值
                'defaultType': self.config.get(
                    'api', 'binance', 'options', 'default_type', default='spot'),
                'keepAlive': True,  # 启用TCP Keep-Alive
            },
            'headers': {
                'Connection': 'keep-alive',  # 重要: 保持连接打开
                'Keep-Alive': '60',          # 保持60秒
            }
        }
        
        # 检查配置是否明确启用或禁用代理
        use_proxy = self.config.get('api', 'useproxy', default=None)
        
        # 如果没有明确禁用代理，尝试使用代理
        if use_proxy is True or use_proxy == "true":
            # 从配置中获取代理
            http_proxy = self.config.get('proxies', 'http', default=None)
            https_proxy = self.config.get('proxies', 'https', default=None)
            
            proxies = {}
            if http_proxy or https_proxy:
                # 使用配置中指定的代理
                proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
                logger.info(f"使用配置中指定的代理: {proxies}")
                if proxies:
                    params['proxies'] = proxies
            
        return params

    def _init_exchange(self):
        """初始化交易所连接，改进错误处理和恢复机制"""
        
        # 检查是否启用离线模式
        offline_mode = self.config.get('system', 'offline_mode', default=False)
        if offline_mode:
            logger.info("运行在离线模式，跳过Binance交易所初始化")
            self.exchange = None
            return
        
        # 跟踪代理状态
        using_proxy = False
        
        try:
            logger.info("初始化Binance交易所")
            
            # 添加自定义异步HTTP客户端设置
            import aiohttp
            
            # 针对高延迟网络优化HTTP连接器
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,      # 保持连接60秒
                limit_per_host=10,         # 每个主机最多10个连接
                limit=20,                  # 总连接限制
                ttl_dns_cache=300,         # DNS缓存5分钟
                use_dns_cache=True,        # 使用DNS缓存
                enable_cleanup_closed=True # 自动清理关闭的连接
            )
            
            # 尝试初始化交易所
            try:
                # 首先测试同步交易所
                self.exchange = ccxt.binance(self.params)
                logger.info("同步交易所初始化成功")
                
                # 设置异步请求信号量
                self.request_semaphore = asyncio.Semaphore(5)  # 限制最大并发请求数
                
                # 预加载市场数据
                try:
                    self.exchange.load_markets()
                    logger.info("成功加载市场数据")
                except Exception as e:
                    logger.warning(f"加载市场数据失败: {str(e)}，将在首次调用时加载")
                
            except Exception as e:
                logger.error(f"初始化交易所时出错: {str(e)}")
                raise RuntimeError(f"无法初始化交易所: {str(e)}")
                
        except Exception as e:
            logger.error(f"初始化Binance交易所失败: {str(e)}")
            
            # 特殊处理离线回测支持
            if self.config.get('backtest', 'fallback_to_local', default=True):
                logger.warning("交易所初始化失败但启用了fallback_to_local。创建最小交易所实例。")
                self.exchange = ccxt.binance()
                self.exchange.markets = {}
                self.exchange.markets_by_id = {}
                return
                
            raise RuntimeError(f"Binance初始化失败: {str(e)}")

    async def _init_async_exchange(self) -> None:
        """初始化异步交易所（延迟初始化），针对高延迟网络进行优化"""
        if self.async_exchange is not None:
            return

        try:
            # 创建优化的aiohttp会话
            import aiohttp
            from aiohttp import ClientTimeout
            
            # 针对高延迟网络优化HTTP连接器
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,      # 保持连接60秒
                limit_per_host=10,         # 每个主机最多10个连接
                limit=20,                  # 总连接限制
                ttl_dns_cache=300,         # DNS缓存5分钟
                use_dns_cache=True,        # 使用DNS缓存
                enable_cleanup_closed=True # 自动清理关闭的连接
            )
            
            # 设置更长的超时以适应高延迟
            timeout = ClientTimeout(
                total=60,                 # 总超时60秒
                connect=20,               # 连接超时20秒
                sock_read=45,             # 读取超时45秒
                sock_connect=20           # 套接字连接超时20秒
            )
            
            # 创建会话
            session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive', 'Keep-Alive': '60'}
            )
            
            # 初始化异步交易所
            self.async_exchange = ccxt_async.binance(self.params)
            
            # 应用优化的会话
            self.async_exchange.session = session
            
            # 初始化并发限制信号量
            if not self.request_semaphore:
                self.request_semaphore = asyncio.Semaphore(5)
            
            logger.info("异步Binance交易所初始化完成，已针对高延迟网络优化")
        except Exception as e:
            logger.error(f"初始化异步Binance交易所失败: {str(e)}")
            self.async_exchange = None
    
    async def _handle_rate_limit(self) -> None:
        """为API请求管理速率限制，支持高延迟环境"""
        current_time = time.time()
        
        # 检查速率限制窗口是否已重置
        if current_time > self.rate_limit_reset:
            self.current_requests = 0
            self.rate_limit_reset = current_time + 60
        
        # 检查是否达到速率限制
        if self.current_requests >= self.rate_limit:
            # 计算睡眠时间
            sleep_time = max(0, self.rate_limit_reset - current_time)
            logger.warning(f"达到速率限制，等待 {sleep_time:.2f} 秒")
            await asyncio.sleep(sleep_time)
            
            # 重置计数器
            self.current_requests = 0
            self.rate_limit_reset = time.time() + 60
        
        # 增加请求计数器
        self.current_requests += 1
    
    async def _exponential_backoff(self, attempt: int) -> float:
        """实现指数退避与抖动"""
        base_delay = 1.0
        max_delay = 30.0
        # 计算指数退避时间并添加随机抖动
        delay = min(max_delay, base_delay * (2 ** attempt)) 
        jitter = random.uniform(0.5, 1.0)  # 50-100% 随机抖动
        final_delay = delay * jitter
        logger.info(f"将等待 {final_delay:.2f}秒后重试 (尝试 {attempt+1})")
        await asyncio.sleep(final_delay)
        return final_delay
    
    @staticmethod
    def _process_ohlcv_data(ohlcv_data: List) -> pd.DataFrame:
        """将OHLCV数据转换为DataFrame"""
        if not ohlcv_data:
            return pd.DataFrame()
        
        try:
            # 创建DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 将时间戳转换为datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            
            # 按时间戳排序
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            logger.error(f"处理OHLCV数据时出错: {str(e)}")
            return pd.DataFrame()
    
    @retry_exchange_operation(max_attempts=3, base_delay=2.0, max_delay=30.0)
    def fetch_latest_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """获取最近的OHLCV数据，使用优化的重试机制"""
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("交易所未初始化，无法获取数据")
                return pd.DataFrame()
        
        try:
            # 获取数据
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol, 
                timeframe=timeframe, 
                limit=limit,
                params={
                    'recvWindow': 60000,  # 最大接收窗口
                }
            )
            
            # 转换为DataFrame
            df = self._process_ohlcv_data(ohlcv)
            
            if not df.empty:
                logger.info(f"成功获取 {symbol} {timeframe} 的最新 {len(df)} 条数据")
                return df
            
            logger.warning(f"获取 {symbol} 数据时返回空响应")
            return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"获取OHLCV数据时出错: {str(e)}")
            raise  # 让装饰器处理重试
    
    async def fetch_historical_ohlcv(self, 
                                symbol: str, 
                                timeframe: str = '1m',
                                start_date: Optional[Union[str, datetime]] = None,
                                end_date: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """获取历史OHLCV数据，针对高延迟网络进行了性能优化"""
        # 确保交易所已初始化
        if not self.exchange:
            self._init_exchange()
            if not self.exchange:
                logger.error("交易所初始化失败")
                return pd.DataFrame()
        
        try:
            # 处理日期
            start_dt = TimeUtils.parse_timestamp(start_date, default_days_ago=30)
            end_dt = TimeUtils.parse_timestamp(end_date, default_days_ago=0)
            
            logger.info(f"获取 {symbol} 从 {start_dt} 到 {end_dt} 的历史数据")
            
            # 初始化异步交易所（如果需要）
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # 将请求分成多个块以处理大日期范围
            chunks = self._get_date_chunks(start_dt, end_dt, timeframe)
            logger.info(f"将请求分成 {len(chunks)} 个块")
            
            # 跟踪进度
            all_data = []
            chunks_processed = 0
            tasks = []
            
            # 并发处理块，但受信号量限制
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                task = asyncio.create_task(
                    self._fetch_chunk(i, chunk_start, chunk_end, symbol, timeframe)
                )
                tasks.append(task)
            
            # 等待所有任务完成
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in chunk_results:
                if isinstance(result, Exception):
                    logger.error(f"块处理失败: {str(result)}")
                elif isinstance(result, list) and result:
                    all_data.extend(result)
            
            # 处理所有数据
            if not all_data:
                logger.warning(f"未找到 {symbol} 的历史数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = self._process_ohlcv_data(all_data)
            
            # 按日期范围过滤
            if not df.empty:
                # 确保datetime列是datetime类型
                if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                    df['datetime'] = pd.to_datetime(df['datetime'])
                
                df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)]
                
                # 删除重复项
                if 'datetime' in df.columns:
                    df = df.drop_duplicates(subset=['datetime'])
                    
                # 按时间戳排序
                df = df.sort_values('datetime')
            
            logger.info(f"为 {symbol} 下载了 {len(df)} 条蜡烛图数据")
            
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 的历史数据失败: {str(e)}")
            return pd.DataFrame()
    
    async def _fetch_chunk(self, chunk_index: int, chunk_start: datetime, 
                          chunk_end: datetime, symbol: str, timeframe: str) -> List:
        """获取单个历史数据块，带有重试和速率限制"""
        # 使用信号量限制并发
        async with self.request_semaphore:
            for retry in range(self.max_retry_attempts):
                try:
                    # 处理速率限制
                    await self._handle_rate_limit()
                    
                    # 转换为毫秒时间戳
                    chunk_since = int(chunk_start.timestamp() * 1000)
                    chunk_until = int(chunk_end.timestamp() * 1000)
                    
                    # 记录进度
                    if chunk_index == 0 or chunk_index % 5 == 0:
                        logger.info(f"获取块 {chunk_index+1}: {chunk_start} 到 {chunk_end}")
                    
                    # 获取此块的数据
                    exchange = self.async_exchange if self.async_exchange else self.exchange
                    
                    ohlcv = await exchange.fetch_ohlcv(
                        symbol=symbol, 
                        timeframe=timeframe, 
                        since=chunk_since,
                        limit=self.download_chunk_size,
                        params={
                            "endTime": chunk_until,
                            "recvWindow": 60000  # 使用最大接收窗口以适应高延迟
                        }
                    )
                    
                    # 检查我们是否得到了数据
                    if not ohlcv or len(ohlcv) == 0:
                        logger.debug(f"块 {chunk_index+1} 未返回数据")
                        return []
                    
                    logger.debug(f"块 {chunk_index+1} 成功获取了 {len(ohlcv)} 条记录")
                    
                    # 短暂延迟以避免请求风暴
                    await asyncio.sleep(0.2)
                    
                    # 成功，返回数据
                    return ohlcv
                    
                except Exception as e:
                    logger.warning(f"块 {chunk_index+1}, 尝试 {retry+1} 出错: {str(e)}")
                    if retry < self.max_retry_attempts - 1:
                        # 使用指数退避
                        await self._exponential_backoff(retry)
            
            # 所有重试都失败
            logger.error(f"块 {chunk_index+1} 的所有重试尝试均失败")
            return []
    
    def _get_date_chunks(self, start_dt: datetime, end_dt: datetime, 
                        timeframe: str, chunk_size_days: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """将日期范围分解为可管理的块，为高延迟网络优化"""
        # 根据时间周期确定块大小 - 为高延迟减小块大小
        if chunk_size_days is None:
            if timeframe in ['1m', '5m']:
                chunk_size = timedelta(hours=12)  # 减小为半天
            elif timeframe in ['15m', '30m']:
                chunk_size = timedelta(days=1)    # 减小为1天
            elif timeframe == '1h':
                chunk_size = timedelta(days=3)    # 减小为3天
            else:
                chunk_size = timedelta(days=7)    # 减小为7天
        else:
            chunk_size = timedelta(days=chunk_size_days)
        
        # 创建块
        chunks = []
        current_start = start_dt
        
        while current_start < end_dt:
            current_end = min(current_start + chunk_size, end_dt)
            chunks.append((current_start, current_end))
            current_start = current_end
        
        return chunks
        
    async def watch_ohlcv(self, symbol: str, timeframe: str, callback=None):
        """
        使用WebSocket实时订阅OHLCV数据（需要ccxt.pro）
        
        Args:
            symbol: 交易对符号
            timeframe: 时间周期
            callback: 接收数据的回调函数
            
        Returns:
            None: 持续运行的任务
        """
        try:
            # 检查是否已安装ccxt.pro
            try:
                import ccxtpro
            except ImportError:
                logger.error("未安装ccxt.pro。请安装ccxt.pro以使用WebSocket功能。")
                return False
            
            # 订阅键
            sub_key = f"{symbol}_{timeframe}"
            
            # 如果已经订阅，避免重复订阅
            if sub_key in self.ws_subscriptions and self.ws_subscriptions[sub_key]['active']:
                logger.info(f"已经订阅了 {symbol} {timeframe} 的WebSocket数据流")
                return True
            
            logger.info(f"开始WebSocket连接以实时监控 {symbol} {timeframe}")
            
            # 初始化交易所
            exchange = ccxtpro.binance(self.params)
            
            # 配置WebSocket选项
            exchange.options['ws'] = {
                'heartbeat': True,         # 启用心跳
                'ping_interval': 30000,    # 30秒ping间隔
                'reconnect_rate': 5000,    # 5秒重连率
                'max_reconnects': 100,     # 最大重连次数
            }
            
            # 标记为活动状态
            self.ws_subscriptions[sub_key] = {
                'active': True,
                'exchange': exchange,
                'last_data': None,
                'errors': 0,
            }
            
            # 启动后台任务管理WebSocket连接
            asyncio.create_task(self._ws_manager(sub_key, symbol, timeframe, callback))
            
            return True
            
        except Exception as e:
            logger.error(f"设置WebSocket连接时出错: {str(e)}")
            return False
    
    async def _ws_manager(self, sub_key: str, symbol: str, timeframe: str, callback):
        """管理WebSocket连接的持续运行"""
        if sub_key not in self.ws_subscriptions:
            logger.error(f"找不到 {sub_key} 的WebSocket订阅")
            return
        
        sub_info = self.ws_subscriptions[sub_key]
        exchange = sub_info['exchange']
        
        retry_delay = 1.0  # 初始重试延迟
        max_retry_delay = 30.0  # 最大重试延迟
        
        while sub_info['active']:
            try:
                logger.debug(f"等待 {symbol} {timeframe} 的WebSocket更新")
                ohlcv = await exchange.watchOHLCV(symbol, timeframe)
                
                # 数据处理
                sub_info['last_data'] = ohlcv
                sub_info['errors'] = 0  # 重置错误计数
                
                # 重置重试延迟
                retry_delay = 1.0
                
                # 调用回调
                if callback and callable(callback):
                    try:
                        await callback(ohlcv)
                    except Exception as callback_error:
                        logger.error(f"回调处理出错: {str(callback_error)}")
                
            except Exception as e:
                sub_info['errors'] += 1
                logger.error(f"WebSocket错误 ({sub_info['errors']}): {str(e)}")
                
                # 如果错误太多，可能需要彻底重新连接
                if sub_info['errors'] > 10:
                    logger.warning(f"错误过多，重新初始化WebSocket连接")
                    try:
                        await exchange.close()
                        # 重新创建交易所实例
                        import ccxtpro
                        exchange = ccxtpro.binance(self.params)
                        exchange.options['ws'] = {
                            'heartbeat': True,
                            'ping_interval': 30000,
                            'reconnect_rate': 5000,
                            'max_reconnects': 100,
                        }
                        sub_info['exchange'] = exchange
                        sub_info['errors'] = 0
                    except Exception as reset_error:
                        logger.error(f"重置WebSocket连接时出错: {str(reset_error)}")
                
                # 使用指数退避
                retry_delay = min(retry_delay * 2, max_retry_delay)
                jitter = random.uniform(0.5, 1.0)
                actual_delay = retry_delay * jitter
                logger.info(f"等待 {actual_delay:.2f}秒后重试WebSocket连接")
                await asyncio.sleep(actual_delay)
    
    async def stop_watching(self, symbol: str, timeframe: str):
        """停止WebSocket数据订阅"""
        sub_key = f"{symbol}_{timeframe}"
        
        if sub_key in self.ws_subscriptions:
            try:
                # 标记为非活动
                self.ws_subscriptions[sub_key]['active'] = False
                
                # 关闭交易所连接
                exchange = self.ws_subscriptions[sub_key].get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
                    
                logger.info(f"已停止 {symbol} {timeframe} 的WebSocket订阅")
                return True
            except Exception as e:
                logger.error(f"停止WebSocket订阅时出错: {str(e)}")
                return False
        
        logger.warning(f"找不到 {symbol} {timeframe} 的活动WebSocket订阅")
        return False
        
    async def close(self):
        """清理资源"""
        # 关闭所有WebSocket订阅
        for sub_key, sub_info in list(self.ws_subscriptions.items()):
            try:
                sub_info['active'] = False
                exchange = sub_info.get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
            except Exception as e:
                logger.error(f"关闭WebSocket订阅 {sub_key} 时出错: {str(e)}")
        
        # 关闭异步交易所
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                # 关闭底层会话
                if hasattr(self.async_exchange, 'session') and hasattr(self.async_exchange.session, 'close'):
                    await self.async_exchange.session.close()
                self.async_exchange = None
                logger.info("异步交易所连接已关闭")
        except Exception as e:
            logger.error(f"关闭异步交易所时出错: {str(e)}")


# 以下部分包含辅助函数，ProxyDetector类和网络工具

# src/utils/network.py
import os
import socket
import requests
from typing import Dict, Optional, Union
import subprocess
import re
import platform
from urllib.parse import urlparse


logger = LogManager.get_logger("data.manager")


class ProxyDetector:
    """
    用于检测和验证代理配置的实用程序类，
    用于外部API（如Binance）。
    """
    
    def __init__(self, test_url: str = 'https://api.binance.com/api/v3/ping', timeout: int = 5):
        """
        初始化代理检测器
        
        Args:
            test_url: 用于测试代理连接的URL
            timeout: 以秒为单位的连接超时
        """
        self.test_url = test_url
        self.timeout = timeout
        self.common_proxy_ports = [7890, 8080, 1080, 8118, 3128, 10809, 7070, 9090]
    
    def detect_proxy(self, check_env: bool = True, check_system: bool = True, 
                   check_common_ports: bool = True) -> Optional[Dict[str, str]]:
        """
        使用多种方法检测可用的代理配置
        
        Args:
            check_env: 是否检查环境变量
            check_system: 是否检查系统代理设置
            check_common_ports: 是否检查常见代理端口
            
        Returns:
            Dict包含http/https代理URL或者如果没有找到工作代理则为None
        """
        proxy_sources = []
        
        # 1. 检查环境变量
        if check_env:
            env_proxies = self._check_environment_variables()
            if env_proxies:
                proxy_sources.append(("环境变量", env_proxies))
        
        # 2. 检查系统代理设置
        if check_system:
            system_proxies = self._check_system_proxy()
            if system_proxies:
                proxy_sources.append(("系统设置", system_proxies))
        
        # 3. 检查requests库的自动检测
        requests_proxies = self._check_requests_proxy()
        if requests_proxies:
            proxy_sources.append(("Requests自动检测", requests_proxies))
        
        # 4. 检查常见代理端口
        if check_common_ports:
            for port in self.common_proxy_ports:
                common_proxy = self._check_common_proxy_port(port)
                if common_proxy:
                    proxy_sources.append((f"常见端口 {port}", common_proxy))
                    break
        
        # 测试每个检测到的代理并返回第一个工作的代理
        for source, proxy in proxy_sources:
            if self.test_proxy(proxy):
                logger.info(f"从{source}找到工作代理: {proxy}")
                return proxy
                
        logger.info("未找到工作代理，使用直接连接")
        return None
    
    def _check_environment_variables(self) -> Optional[Dict[str, str]]:
        """检查环境变量中的代理配置"""
        env_vars = {
            'http': ['HTTP_PROXY', 'http_proxy'],
            'https': ['HTTPS_PROXY', 'https_proxy', 'HTTP_PROXY', 'http_proxy']
        }
        
        proxies = {}
        
        for protocol, var_names in env_vars.items():
            for var in var_names:
                if var in os.environ and os.environ[var]:
                    proxies[protocol] = os.environ[var]
                    break
        
        if proxies:
            logger.debug(f"在环境变量中找到代理设置: {proxies}")
            return proxies
        
        return None
    
    def _check_system_proxy(self) -> Optional[Dict[str, str]]:
        """使用特定于平台的方法检查系统代理配置"""
        proxies = None
        system = platform.system()
        
        try:
            if system == 'Windows':
                proxies = self._get_windows_proxy()
            elif system == 'Darwin':  # macOS
                proxies = self._get_macos_proxy()
            elif system == 'Linux':
                proxies = self._get_linux_proxy()
            
            if proxies:
                logger.debug(f"找到系统代理设置: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"检查系统代理时出错: {e}")
        
        return None
    
    def _get_windows_proxy(self) -> Optional[Dict[str, str]]:
        """从Windows注册表获取代理设置"""
        try:
            import winreg
            
            proxy_key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings"
            )
            
            # 检查代理是否启用
            proxy_enable, _ = winreg.QueryValueEx(proxy_key, "ProxyEnable")
            
            if proxy_enable:
                proxy_server, _ = winreg.QueryValueEx(proxy_key, "ProxyServer")
                
                # 处理不同的代理格式
                if "=" in proxy_server:  # 协议特定格式
                    protocols = proxy_server.split(";")
                    proxies = {}
                    
                    for protocol in protocols:
                        if "=" in protocol:
                            proto_name, address = protocol.split("=", 1)
                            if proto_name.lower() in ('http', 'https'):
                                proxies[proto_name.lower()] = address
                    
                    return proxies if proxies else None
                else:  # 所有协议使用相同代理
                    return {'http': proxy_server, 'https': proxy_server}
            
        except Exception as e:
            logger.debug(f"读取Windows注册表时出错: {e}")
        
        return None
    
    def _get_macos_proxy(self) -> Optional[Dict[str, str]]:
        """从macOS系统偏好设置获取代理设置"""
        try:
            # 检查HTTP代理
            result = subprocess.run(
                ['networksetup', '-getwebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            http_enabled = "Enabled: Yes" in result.stdout
            http_proxy = None
            
            if http_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    http_proxy = f"http://{server}:{port}"
            
            # 检查HTTPS代理
            result = subprocess.run(
                ['networksetup', '-getsecurewebproxy', 'Wi-Fi'],
                capture_output=True, text=True
            )
            
            https_enabled = "Enabled: Yes" in result.stdout
            https_proxy = None
            
            if https_enabled:
                server_match = re.search(r'Server: (.*)', result.stdout)
                port_match = re.search(r'Port: (\d+)', result.stdout)
                
                if server_match and port_match:
                    server = server_match.group(1)
                    port = port_match.group(1)
                    https_proxy = f"http://{server}:{port}"
            
            if http_proxy or https_proxy:
                return {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
            
        except Exception as e:
            logger.debug(f"检查macOS代理时出错: {e}")
        
        return None
    
    def _get_linux_proxy(self) -> Optional[Dict[str, str]]:
        """从Linux环境获取代理设置"""
        # 检查常见环境变量
        proxies = self._check_environment_variables()
        if proxies:
            return proxies
            
        # 检查GNOME设置
        try:
            result = subprocess.run(
                ['gsettings', 'get', 'org.gnome.system.proxy', 'mode'],
                capture_output=True, text=True
            )
            
            if 'manual' in result.stdout:
                http_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                http_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.http', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                https_host = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'host'],
                    capture_output=True, text=True
                ).stdout.strip().strip("'")
                
                https_port = subprocess.run(
                    ['gsettings', 'get', 'org.gnome.system.proxy.https', 'port'],
                    capture_output=True, text=True
                ).stdout.strip()
                
                proxies = {}
                
                if http_host and http_port:
                    proxies['http'] = f"http://{http_host}:{http_port}"
                
                if https_host and https_port:
                    proxies['https'] = f"http://{https_host}:{https_port}"
                elif 'http' in proxies:
                    proxies['https'] = proxies['http']
                
                if proxies:
                    return proxies
        
        except Exception as e:
            logger.debug(f"检查Linux代理设置时出错: {e}")
        
        return None
    
    def _check_requests_proxy(self) -> Optional[Dict[str, str]]:
        """使用requests库的自动检测检查代理设置"""
        try:
            session = requests.Session()
            proxies = session.proxies
            
            # 过滤掉空代理
            proxies = {k: v for k, v in proxies.items() if v}
            
            if proxies:
                logger.debug(f"通过requests自动检测找到代理设置: {proxies}")
                return proxies
        except Exception as e:
            logger.debug(f"检查requests代理时出错: {e}")
        
        return None
    
    def _check_common_proxy_port(self, port: int) -> Optional[Dict[str, str]]:
        """检查本地主机上是否有常见代理端口可用"""
        proxy_url = f"http://127.0.0.1:{port}"
        
        try:
            # 尝试连接到端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:  # 端口打开
                logger.debug(f"在以下位置找到开放代理端口: {proxy_url}")
                return {'http': proxy_url, 'https': proxy_url}
        except Exception as e:
            logger.debug(f"检查代理端口 {port} 时出错: {e}")
        
        return None
    
    def test_proxy(self, proxy_settings: Dict[str, str]) -> bool:
        """
        测试代理配置是否适用于目标API
        
        Args:
            proxy_settings: Dict带有'http'和/或'https'代理URL
            
        Returns:
            bool: 如果代理工作则为True，否则为False
        """
        if not proxy_settings:
            return False
            
        try:
            logger.debug(f"测试代理配置: {proxy_settings}")
            response = requests.get(
                self.test_url,
                proxies=proxy_settings,
                timeout=self.timeout
            )
            
            success = response.status_code == 200
            logger.debug(f"代理测试 {'成功' if success else '失败'} 状态码 {response.status_code}")
            return success
        except Exception as e:
            logger.debug(f"代理测试失败，错误: {e}")
            return False
    
    def get_proxy_info(self, proxy_url: str) -> Dict[str, Union[str, int]]:
        """
        解析代理URL并提取组件
        
        Args:
            proxy_url: 代理URL字符串
            
        Returns:
            Dict包含协议，主机，端口和认证信息
        """
        if not proxy_url:
            return {}
            
        try:
            parsed = urlparse(proxy_url)
            
            # 处理认证信息（如果存在）
            auth = None
            if parsed.username and parsed.password:
                auth = f"{parsed.username}:{parsed.password}"
            
            return {
                'protocol': parsed.scheme,
                'host': parsed.hostname,
                'port': parsed.port,
                'auth': auth
            }
        except Exception as e:
            logger.error(f"解析代理URL时出错: {e}")
            return {}

def detect_proxy(test_url: str = 'https://api.binance.com/api/v3/ping') -> Optional[Dict[str, str]]:
    """
    便捷函数，用于检测目标URL的可用代理设置
    
    Args:
        test_url: 用于测试代理的URL
        
    Returns:
        Dict带有代理配置或者如果没有找到工作代理则为None
    """
    detector = ProxyDetector(test_url=test_url)
    return detector.detect_proxy()

def test_connection(url: str = 'https://api.binance.com/api/v3/ping', 
                   proxy: Optional[Dict[str, str]] = None, 
                   timeout: int = 5) -> bool:
    """
    测试与URL的连接，可选使用代理
    
    Args:
        url: 要测试的URL
        proxy: 可选代理配置
        timeout: 以秒为单位的连接超时
        
    Returns:
        bool: 如果连接成功则为True，否则为False
    """
    try:
        response = requests.get(url, proxies=proxy, timeout=timeout)
        return response.status_code == 200
    except Exception as e:
        logger.debug(f"连接测试失败: {e}")
        return False