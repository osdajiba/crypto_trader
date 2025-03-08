# src\exchange\binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import os
import requests
import asyncio
import traceback
from typing import Optional, Union, Dict, List, Any, Tuple
from datetime import datetime, timedelta

from src.utils.error_handling import retry
from src.utils.time_utils import TimeUtils
from src.common.async_executor import AsyncExecutor
from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")

class Direction:
    """交易方向枚举"""
    BUY = "buy"
    SELL = "sell"

class Binance:
    """
    增强的Binance交易所接口，支持账户管理、订单操作和智能数据获取
    
    提供自动重试、速率限制管理和异步操作支持
    """
    
    def __init__(self, config):
        """
        初始化Binance接口
        
        Args:
            config: 配置对象，包含API密钥和其他设置
        """
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = config
        
        # 本地数据库路径
        self.local_database = os.path.join(self.base_dir, config.get('data', 'data_paths', 'base_path', default="db/")) if self.config else None
        
        # 初始化交易所连接
        self.exchange = None
        self.async_exchange = None
        
        # 创建异步执行器
        self.executor = AsyncExecutor()
        
        # 连接交易所
        self._connect_exchange()
        
        # 速率限制设置
        self.rate_limit_weight = 0
        self.rate_limit_last_reset = datetime.now()
        self.rate_limit_max = 1200  # 每分钟的最大权重
        self.rate_limit_reset_interval = 60  # 秒
        
        # 下载设置
        self.download_chunk_size = config.get('api', 'download', 'chunk_size', default=500) if config else 500
        self.download_max_retries = config.get('api', 'download', 'max_retries', default=3) if config else 3
        self.download_retry_delay = config.get('api', 'download', 'retry_delay', default=2) if config else 2
        
        logger.info("Binance接口初始化完成")

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError, ccxt.AuthenticationError))
    def _connect_exchange(self) -> None:
        """初始化同步和异步交易所连接"""
        # 检查配置
        if not self.config or not self.config.get('api', 'binance'):
            logger.error("未找到Binance配置")
            return
        
        try:
            # 获取认证和连接参数
            exchange_params = {
                'apiKey': self.config.get('api', 'binance', 'apiKey'),
                'secret': self.config.get('api', 'binance', 'secret'),
                'timeout': self.config.get('api', 'timeout', 30000),
                'enableRateLimit': self.config.get('api', 'enableRateLimit', True),
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000
                }
            }
            
            # 添加可选的HTTP头信息
            headers = self.config.get('api', 'headers', 'Connection', {})
            if headers:
                exchange_params['headers'] = headers
                
            # 添加可选的代理设置
            proxies = self.config.get('api', 'proxies', {})
            if proxies:
                exchange_params['proxies'] = proxies
                
            # 创建同步和异步客户端
            self.exchange = ccxt.binance(exchange_params)
            
            # 验证认证
            self.exchange.check_required_credentials()
            logger.info("同步客户端API密钥验证成功")
            
            # 创建异步客户端
            self.async_exchange = ccxt_async.binance(exchange_params)
            logger.info("异步客户端初始化成功")
            
        except Exception as e:
            logger.error(f"连接Binance失败: {str(e)}\n{traceback.format_exc()}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(requests.RequestException,))
    def get_current_ip(self) -> Optional[str]:
        """
        获取当前IP地址，支持代理
        
        Returns:
            Optional[str]: IP地址
        """
        try:
            proxy = None
            if self.config and 'proxies' in self.config and 'http' in self.config['proxies']:
                proxy = self.config['proxies']['http']
                
            response = requests.get(
                'https://api.ipify.org', 
                proxies={'http': proxy, 'https': proxy} if proxy else None, 
                timeout=10
            )
            response.raise_for_status()
            ip = response.text
            logger.info(f"当前IP地址: {ip}")
            return ip
            
        except Exception as e:
            logger.error(f"获取IP地址失败: {str(e)}")
            return None

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_balance(self) -> Optional[Dict[str, float]]:
        """
        获取账户总余额
        
        Returns:
            Optional[Dict[str, float]]: 账户余额
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return None
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(10)
            
            # 获取余额
            balance = self.exchange.fetch_balance()
            return balance.get('total', {})
            
        except Exception as e:
            logger.error(f"获取账户余额失败: {str(e)}")
            return None

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """
        获取最近的OHLCV数据
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            limit: 获取的数据点数量
            
        Returns:
            pd.DataFrame: OHLCV数据
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return pd.DataFrame()
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(5)
            
            # 获取OHLCV数据
            ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
            return self._to_dataframe(ohlcv)
            
        except Exception as e:
            logger.error(f"获取OHLCV数据失败 ({symbol} {timeframe}): {str(e)}")
            return pd.DataFrame()

    async def smart_fetch_ohlcv(self, symbol: str, timeframe: str, 
                              start: Union[str, datetime, int], 
                              end: Union[str, datetime, int]) -> pd.DataFrame:
        """
        智能分页获取历史OHLCV数据，避免速率限制和超时
        
        Args:
            symbol: 交易对
            timeframe: 时间框架
            start: 开始时间
            end: 结束时间
        
        Returns:
            pd.DataFrame: OHLCV数据
        """
        if not self.async_exchange:
            logger.error("异步客户端未初始化")
            return pd.DataFrame()
            
        try:
            # 解析时间
            start_dt = TimeUtils.parse_timestamp(start)
            end_dt = TimeUtils.parse_timestamp(end)
            
            # 确保时间顺序正确
            if start_dt > end_dt:
                logger.warning(f"开始时间晚于结束时间，交换时间范围: {start_dt} > {end_dt}")
                start_dt, end_dt = end_dt, start_dt
                
            # 计算时间框架毫秒数
            timeframe_ms = self._timeframe_to_ms(timeframe)
            
            logger.info(f"开始智能获取OHLCV数据: {symbol} {timeframe} {start_dt} - {end_dt}")
            
            # 计算数据获取块
            chunks = self._calculate_fetch_chunks(start_dt, end_dt, timeframe)
            logger.info(f"将请求分为{len(chunks)}个块")
            
            # 存储所有获取的数据
            all_data = []
            successful_chunks = 0
            
            # 确保异步执行器已启动
            await self.executor.start()
            
            # 分块获取数据
            for i, (chunk_start, chunk_end) in enumerate(chunks):
                logger.debug(f"获取第{i+1}/{len(chunks)}块: {chunk_start} - {chunk_end}")
                
                # 转换时间戳格式
                since = int(chunk_start.timestamp() * 1000)
                
                # 计算该块的最佳limit值
                limit = self._calculate_optimal_limit(chunk_start, chunk_end, timeframe)
                
                try:
                    for attempt in range(self.download_max_retries):
                        try:
                            # 获取数据
                            data = await self.async_exchange.fetch_ohlcv(
                                symbol=symbol,
                                timeframe=timeframe,
                                since=since,
                                limit=limit
                            )
                            
                            # 如果成功获取数据，更新速率限制
                            await self._async_update_rate_limit(5)
                            
                            # 过滤数据以匹配结束时间
                            end_ts = int(end_dt.timestamp() * 1000)
                            filtered = [d for d in data if d[0] <= end_ts]
                            
                            if filtered:
                                all_data.extend(filtered)
                                successful_chunks += 1
                                
                            # 可能的提前结束条件
                            if len(data) < limit:
                                logger.debug(f"第{i+1}块获取的数据少于请求的限制，可能达到了数据结尾")
                                break
                                
                            # 成功获取，跳出重试循环
                            break
                            
                        except Exception as e:
                            if attempt < self.download_max_retries - 1:
                                logger.warning(f"第{i+1}块数据获取失败，尝试重试({attempt+1}/{self.download_max_retries}): {str(e)}")
                                await asyncio.sleep(self.download_retry_delay * (attempt + 1))
                            else:
                                logger.error(f"第{i+1}块数据获取失败，已达到最大重试次数: {str(e)}")
                                raise
                    
                    # 添加请求间隔以避免速率限制
                    if i < len(chunks) - 1:
                        interval = self._calculate_request_interval()
                        await asyncio.sleep(interval)
                        
                except Exception as e:
                    logger.error(f"获取第{i+1}块数据时出错: {str(e)}")
                    # 继续获取下一块，而不是完全失败
                    continue
            
            # 检查是否获取了任何数据
            if not all_data:
                logger.warning(f"没有获取到任何数据: {symbol} {timeframe} {start_dt} - {end_dt}")
                return pd.DataFrame()
                
            # 将数据转换为DataFrame
            df = self._to_dataframe(all_data)
            
            # 去除重复的时间戳
            if not df.empty and 'datetime' in df.columns:
                df.drop_duplicates(subset=['datetime'], inplace=True)
                df.sort_values('datetime', inplace=True)
                df.reset_index(drop=True, inplace=True)
                
            # 统计摘要
            total_expected = sum(self._expected_candles(s, e, timeframe) for s, e in chunks)
            missing_rate = 1.0 - (len(df) / total_expected) if total_expected > 0 else 0
            
            logger.info(f"智能获取完成: {symbol} {timeframe} {start_dt} - {end_dt}, "
                       f"获取了{len(df)}行数据，成功{successful_chunks}/{len(chunks)}块, "
                       f"缺失率: {missing_rate:.2%}")
            
            return df
            
        except Exception as e:
            logger.error(f"智能获取OHLCV数据失败: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: Optional[float] = None) -> Dict:
        """
        创建市价或限价订单
        
        Args:
            symbol: 交易对
            side: 订单方向 ('buy' 或 'sell')
            order_type: 订单类型 ('market' 或 'limit')
            amount: 数量
            price: 价格(限价订单必需)
            
        Returns:
            Dict: 订单信息
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
            
        if order_type not in ('market', 'limit'):
            raise ValueError("订单类型必须是'market'或'limit'")
            
        if order_type == 'limit' and price is None:
            raise ValueError("限价订单需要提供价格")
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(1)
            
            # 创建订单
            order = self.exchange.create_order(
                symbol=symbol, 
                type=order_type, 
                side=side, 
                amount=amount, 
                price=price
            )
            logger.info(f"订单创建成功: {order['id']}")
            return order
            
        except Exception as e:
            logger.error(f"创建订单失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        获取未完成订单
        
        Args:
            symbol: 可选的交易对过滤器
            
        Returns:
            List[Dict]: 订单列表
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(3)
            
            # 获取未完成订单
            orders = self.exchange.fetch_open_orders(symbol=symbol)
            logger.info(f"获取未完成订单: {len(orders)}条")
            return orders
            
        except Exception as e:
            logger.error(f"获取未完成订单失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """
        取消指定订单
        
        Args:
            order_id: 订单ID
            symbol: 交易对(某些交易所要求)
            
        Returns:
            Dict: 取消订单的响应
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(1)
            
            # 取消订单
            canceled = self.exchange.cancel_order(order_id, symbol=symbol)
            logger.info(f"订单已取消: {order_id}")
            return canceled
            
        except Exception as e:
            logger.error(f"取消订单失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_trades(self, symbol: str = 'BTC/USDT', since: Optional[Union[str, int]] = None, limit: int = 100) -> List[Dict]:
        """
        获取最近交易记录
        
        Args:
            symbol: 交易对
            since: 开始时间
            limit: 数量限制
            
        Returns:
            List[Dict]: 交易记录列表
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(5)
            
            # 解析时间
            since_ts = TimeUtils.parse_timestamp(since).timestamp() * 1000 if since else None
            
            # 获取交易记录
            trades = self.exchange.fetch_my_trades(
                symbol=symbol, 
                since=int(since_ts) if since_ts else None, 
                limit=limit
            )
            logger.info(f"获取交易记录: {len(trades)}条")
            return trades
            
        except Exception as e:
            logger.error(f"获取交易记录失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_order_book(self, symbol: str = 'BTC/USDT', limit: int = 100) -> Dict:
        """
        获取订单簿数据
        
        Args:
            symbol: 交易对
            limit: 深度限制
            
        Returns:
            Dict: 订单簿数据
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(5)
            
            # 获取订单簿
            return self.exchange.fetch_order_book(symbol=symbol, limit=limit)
            
        except Exception as e:
            logger.error(f"获取订单簿失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_funding_rate(self, symbol: str = 'BTC/USDT') -> Dict:
        """
        获取当前资金费率
        
        Args:
            symbol: 交易对
            
        Returns:
            Dict: 资金费率信息
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
            
        try:
            # 更新速率限制计数
            self._update_rate_limit(5)
            
            # 获取资金费率
            rate = self.exchange.fetch_funding_rate(symbol=symbol)
            logger.info(f"资金费率: {rate.get('fundingRate')}")
            return rate
            
        except Exception as e:
            logger.error(f"获取资金费率失败: {str(e)}")
            raise

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def batch_create_orders(self, orders: List[Dict[str, Any]]) -> List[Dict]:
        """
        批量创建订单
        
        Args:
            orders: 订单参数列表
            
        Returns:
            List[Dict]: 创建的订单列表
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
            
        results = []
        for order in orders:
            try:
                result = self.create_order(
                    symbol=order.get('symbol', 'BTC/USDT'),
                    side=order.get('side', 'buy'),
                    order_type=order.get('type', 'limit'),
                    amount=order.get('amount', 0.001),
                    price=order.get('price')
                )
                results.append(result)
            except Exception as e:
                logger.error(f"批量创建订单时出错: {str(e)}")
                results.append({'error': str(e)})
                
        logger.info(f"批量创建了{len(results)}个订单")
        return results

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """
        将时间框架转换为毫秒
        
        Args:
            timeframe: 时间框架字符串
            
        Returns:
            int: 毫秒数
        """
        seconds_per_unit = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400, '1w': 604800}
        if timeframe not in seconds_per_unit:
            logger.warning(f"未知的时间框架: {timeframe}，使用默认值1m")
            timeframe = '1m'
        return seconds_per_unit.get(timeframe, 60) * 1000

    def _calculate_optimal_limit(self, start: datetime, end: datetime, timeframe: str) -> int:
        """
        动态计算最佳请求数量，避免请求过多数据
        
        Args:
            start: 开始时间
            end: 结束时间
            timeframe: 时间框架
            
        Returns:
            int: 数据点数量
        """
        seconds_per_unit = self._timeframe_to_ms(timeframe) // 1000
        total_seconds = (end - start).total_seconds()
        
        # 估算所需K线数
        estimated_bars = int(total_seconds / seconds_per_unit)
        
        # 添加额外的K线以处理可能的差异
        limit = min(1000, max(100, estimated_bars + 10))
        
        return limit

    def _next_start_point(self, last_ts: int, timeframe_ms: int) -> datetime:
        """
        计算下一次请求的起始时间
        
        Args:
            last_ts: 上次请求的最后时间戳
            timeframe_ms: 时间框架毫秒数
            
        Returns:
            datetime: 下一个起始时间
        """
        # 添加1ms以避免重复获取最后一根K线
        return datetime.fromtimestamp((last_ts + 1) / 1000)

    def _to_dataframe(self, data: List[List[Any]]) -> pd.DataFrame:
        """
        将OHLCV数据转换为DataFrame
        
        Args:
            data: OHLCV数据列表
            
        Returns:
            pd.DataFrame: 格式化的DataFrame
        """
        if not data:
            return pd.DataFrame()
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # 转换时间戳
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 应用时区
            timezone = self.config.get("default_config", "misc_config", "timezone", default="Asia/Shanghai") if self.config else "Asia/Shanghai"
            try:
                df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert(timezone)
            except:
                logger.warning(f"时区转换失败，使用UTC时间")
                df['datetime'] = df['datetime'].dt.tz_localize('UTC')
            
            # 选择最终列
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"DataFrame转换失败: {str(e)}")
            return pd.DataFrame()
    
    def create_order_with_direction(
        self, 
        symbol: str, 
        direction: Union[str, Direction],
        order_type: str, 
        quantity: float, 
        price: Optional[float] = None
    ) -> Dict:
        """
        使用方向枚举创建订单
        
        Args:
            symbol: 交易对
            direction: 方向枚举或字符串
            order_type: 订单类型
            quantity: 数量
            price: 价格
            
        Returns:
            Dict: 订单信息
        """
        # 转换方向枚举为字符串
        if isinstance(direction, Direction):
            side = direction.value
        else:
            side = direction.lower()
        
        # 格式化数量和价格精度
        formatted_qty = self._format_quantity(symbol, quantity)
        formatted_price = self._format_price(symbol, price) if price else None
        
        # 创建订单
        return self.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            amount=formatted_qty,
            price=formatted_price
        )

    def _format_price(self, symbol: str, price: float) -> float:
        """
        格式化价格精度
        
        Args:
            symbol: 交易对
            price: 价格
            
        Returns:
            float: 格式化后的价格
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return price
            
        try:
            # 获取市场信息
            market = self.exchange.market(symbol)
            return self.exchange.price_to_precision(symbol, price)
        except Exception as e:
            logger.warning(f"格式化价格精度失败: {str(e)}")
            return price

    def _format_quantity(self, symbol: str, quantity: float) -> float:
        """
        格式化数量精度
        
        Args:
            symbol: 交易对
            quantity: 数量
            
        Returns:
            float: 格式化后的数量
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return quantity
            
        try:
            # 获取市场信息
            market = self.exchange.market(symbol)
            return self.exchange.amount_to_precision(symbol, quantity)
        except Exception as e:
            logger.warning(f"格式化数量精度失败: {str(e)}")
            return quantity

    def _update_rate_limit(self, weight: int = 1) -> None:
        """
        更新速率限制计数
        
        Args:
            weight: 请求权重
        """
        current_time = datetime.now()
        
        # 检查是否需要重置计数
        time_diff = (current_time - self.rate_limit_last_reset).total_seconds()
        if time_diff >= self.rate_limit_reset_interval:
            self.rate_limit_weight = 0
            self.rate_limit_last_reset = current_time
            
        # 更新权重
        self.rate_limit_weight += weight
        
        # 检查是否接近限制
        if self.rate_limit_weight > self.rate_limit_max * 0.8:
            logger.warning(f"接近速率限制: {self.rate_limit_weight}/{self.rate_limit_max}")
            
        # 如果超过限制，等待
        if self.rate_limit_weight >= self.rate_limit_max:
            wait_time = self.rate_limit_reset_interval - time_diff
            if wait_time > 0:
                logger.warning(f"达到速率限制，等待{wait_time:.2f}秒")
                time.sleep(wait_time)
                # 重置计数
                self.rate_limit_weight = weight
                self.rate_limit_last_reset = datetime.now()

    async def _async_update_rate_limit(self, weight: int = 1) -> None:
        """
        异步更新速率限制计数
        
        Args:
            weight: 请求权重
        """
        current_time = datetime.now()
        
        # 检查是否需要重置计数
        time_diff = (current_time - self.rate_limit_last_reset).total_seconds()
        if time_diff >= self.rate_limit_reset_interval:
            self.rate_limit_weight = 0
            self.rate_limit_last_reset = current_time
            
        # 更新权重
        self.rate_limit_weight += weight
        
        # 检查是否接近限制
        if self.rate_limit_weight > self.rate_limit_max * 0.8:
            logger.warning(f"接近速率限制: {self.rate_limit_weight}/{self.rate_limit_max}")
            
        # 如果超过限制，等待
        if self.rate_limit_weight >= self.rate_limit_max:
            wait_time = self.rate_limit_reset_interval - time_diff
            if wait_time > 0:
                logger.warning(f"达到速率限制，等待{wait_time:.2f}秒")
                await asyncio.sleep(wait_time)
                # 重置计数
                self.rate_limit_weight = weight
                self.rate_limit_last_reset = datetime.now()

    def _calculate_fetch_chunks(self, start: datetime, end: datetime, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """
        将时间范围分割为多个可manageable的块
        
        Args:
            start: 开始时间
            end: 结束时间
            timeframe: 时间框架
            
        Returns:
            List[Tuple]: 时间范围块列表
        """
        # 计算每个K线的毫秒数
        ms_per_candle = self._timeframe_to_ms(timeframe)
        
        # 计算总毫秒数
        total_ms = (end - start).total_seconds() * 1000
        
        # 估计总K线数
        total_candles = int(total_ms / ms_per_candle) + 1
        
        # 如果总数低于阈值，直接返回整个范围
        if total_candles <= self.download_chunk_size:
            return [(start, end)]
            
        # 计算需要的块数
        num_chunks = (total_candles + self.download_chunk_size - 1) // self.download_chunk_size
        
        # 计算每个块的持续时间
        chunk_duration = timedelta(milliseconds=ms_per_candle * self.download_chunk_size)
        
        # 创建块列表
        chunks = []
        chunk_start = start
        
        for i in range(num_chunks):
            # 计算块的结束时间
            chunk_end = chunk_start + chunk_duration
            
            # 确保不超过总结束时间
            if chunk_end > end:
                chunk_end = end
                
            # 添加到列表
            chunks.append((chunk_start, chunk_end))
            
            # 更新下一块的开始时间
            chunk_start = chunk_end
            
            # 如果已经达到结束时间，退出循环
            if chunk_start >= end:
                break
                
        return chunks

    def _calculate_request_interval(self) -> float:
        """
        计算请求之间的间隔，以避免速率限制
        
        Returns:
            float: 秒数
        """
        # 基于当前速率限制状态计算
        percentage_used = self.rate_limit_weight / self.rate_limit_max
        
        if percentage_used > 0.9:
            # 接近限制，使用更长间隔
            return 1.5
        elif percentage_used > 0.7:
            # 中等使用，适中间隔
            return 1.0
        else:
            # 低使用，最小间隔
            return 0.5

    def _expected_candles(self, start: datetime, end: datetime, timeframe: str) -> int:
        """
        计算给定时间范围内预期的K线数量
        
        Args:
            start: 开始时间
            end: 结束时间
            timeframe: 时间框架
            
        Returns:
            int: 预期K线数量
        """
        # 计算每个K线的秒数
        seconds_per_candle = self._timeframe_to_ms(timeframe) // 1000
        
        # 计算总秒数
        total_seconds = (end - start).total_seconds()
        
        # 计算预期K线数
        return int(total_seconds / seconds_per_candle) + 1

    async def close(self) -> None:
        """关闭交易所连接和资源"""
        try:
            # 关闭异步交易所连接
            if self.async_exchange:
                await self.async_exchange.close()
                logger.info("关闭了异步交易所连接")
                
            # 关闭异步执行器
            await self.executor.close()
            logger.info("Binance接口已关闭")
            
        except Exception as e:
            logger.error(f"关闭Binance接口时出错: {str(e)}")

    def get_exchange_info(self) -> Dict:
        """
        获取交易所信息和限制
        
        Returns:
            Dict: 交易所信息
        """
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
            
        try:
            # 加载市场
            markets = self.exchange.load_markets()
            
            # 准备结果
            result = {
                'name': self.exchange.name,
                'versions': self.exchange.versions,
                'timeout': self.exchange.timeout,
                'rate_limit': self.exchange.rateLimit,
                'markets_count': len(markets)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"获取交易所信息失败: {str(e)}")
            return {}