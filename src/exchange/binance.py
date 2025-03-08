# src\exchange\binance.py

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import os
import requests
from typing import Optional, Union, Dict, List, Any
from datetime import datetime
from src.utils.error_handling import retry
from src.utils.time_utils import TimeUtils
from src.common.async_executor import AsyncExecutor
from src.common.log_manager import LogManager
from src.execution.order import Direction  # 添加Direction枚举导入

logger = LogManager.get_logger("trading_system")

class Binance:
    """增强的 Binance 交易所接口，支持账户管理、订单操作和智能数据获取"""
    
    def __init__(self, config):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = config
        self.local_database = os.path.join(self.base_dir, config.get('data', 'data_paths', 'base_path', default="db/")) if self.config else None
        self.exchange = None
        self.async_exchange = None
        self.async_executor = AsyncExecutor()
        self._connect_exchange()

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError, ccxt.AuthenticationError))
    def _connect_exchange(self) -> None:
        """初始化同步和异步交易所连接"""
        if not self.config or 'binance' not in self.config.get('api'):
            logger.error("未找到 Binance 配置")
            return
        exchange_params = {
            'apiKey': self.config.get('api', 'binance', 'apiKey'),
            'secret': self.config.get('api', 'binance', 'secret'),
            'timeout': self.config.get('api', 'timeout', 30000),
            'enableRateLimit': self.config.get('api', 'enableRateLimit', True),
            'headers': self.config.get('api', 'headers', {}),
            'proxies': self.config.get('api', 'proxies', {}),
        }
        self.exchange = ccxt.binance(exchange_params)
        self.exchange.check_required_credentials()
        logger.info("同步客户端 API 密钥验证成功")
        self.async_exchange = ccxt_async.binance(exchange_params)
        logger.info("异步客户端初始化成功")

    @retry(max_retries=3, delay=1, exceptions=(requests.RequestException,))
    def get_current_ip(self) -> Optional[str]:
        """获取当前 IP 地址，支持代理"""
        proxy = self.config['proxies']['http'] if self.config else None
        response = requests.get('https://api.ipify.org', proxies={'http': proxy, 'https': proxy} if proxy else None, timeout=10)
        response.raise_for_status()
        logger.info(f"当前IP地址为: {response.text}")
        return response.text

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_balance(self) -> Optional[Dict[str, float]]:
        """获取账户总余额"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return None
        balance = self.exchange.fetch_balance()
        return balance.get('total', {})

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_ohlcv(self, symbol: str = 'BTC/USDT', timeframe: str = '1m', limit: int = 100) -> pd.DataFrame:
        """获取最近的 OHLCV 数据"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return pd.DataFrame()
        ohlcv = self.exchange.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=limit)
        return self._to_dataframe(ohlcv)

    async def smart_fetch_ohlcv(self, symbol: str, timeframe: str, start: Union[str, datetime, int], end: Union[str, datetime, int]) -> pd.DataFrame:
        """
        智能分页获取历史 OHLCV 数据。
        
        Args:
            symbol (str): 交易对。
            timeframe (str): 时间框架。
            start (Union[str, datetime, int]): 开始时间。
            end (Union[str, datetime, int]): 结束时间。
        
        Returns:
            pd.DataFrame: OHLCV 数据。
        """
        if not self.async_exchange:
            logger.error("异步客户端未初始化")
            return pd.DataFrame()
        
        start_dt = TimeUtils.parse_timestamp(start)
        end_dt = TimeUtils.parse_timestamp(end)
        timeframe_ms = self._timeframe_to_ms(timeframe)
        current = start_dt
        all_data = []

        while current < end_dt:
            limit = self._calculate_optimal_limit(current, end_dt, timeframe)
            data = await self.async_exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=int(current.timestamp() * 1000),
                limit=limit
            )
            if not data:
                break
            filtered = [d for d in data if d[0] <= int(end_dt.timestamp() * 1000)]
            all_data.extend(filtered)
            if len(data) < limit:
                break
            current = self._next_start_point(data[-1][0], timeframe_ms)
        
        return self._to_dataframe(all_data)

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def create_order(self, symbol: str, side: str, order_type: str, amount: float, price: Optional[float] = None) -> Dict:
        """创建市价或限价订单"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
        if order_type not in ('market', 'limit'):
            raise ValueError("订单类型必须是 'market' 或 'limit'")
        if order_type == 'limit' and price is None:
            raise ValueError("限价订单需要提供价格")
        order = self.exchange.create_order(symbol=symbol, type=order_type, side=side, amount=amount, price=price)
        logger.info("订单创建成功: %s", order['id'])
        return order

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """获取未完成订单"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
        orders = self.exchange.fetch_open_orders(symbol=symbol)
        logger.info("获取未完成订单: %d 条", len(orders))
        return orders

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """取消指定订单"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
        canceled = self.exchange.cancel_order(order_id, symbol=symbol)
        logger.info("订单已取消: %s", order_id)
        return canceled

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_trades(self, symbol: str = 'BTC/USDT', since: Optional[Union[str, int]] = None, limit: int = 100) -> List[Dict]:
        """获取最近交易记录"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
        since_ts = TimeUtils.parse_timestamp(since).timestamp() * 1000 if since else None
        trades = self.exchange.fetch_my_trades(symbol=symbol, since=int(since_ts) if since_ts else None, limit=limit)
        logger.info("获取交易记录: %d 条", len(trades))
        return trades

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_order_book(self, symbol: str = 'BTC/USDT', limit: int = 100) -> Dict:
        """获取订单簿数据"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
        return self.exchange.fetch_order_book(symbol=symbol, limit=limit)

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def fetch_funding_rate(self, symbol: str = 'BTC/USDT') -> Dict:
        """获取当前资金费率"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return {}
        rate = self.exchange.fetch_funding_rate(symbol=symbol)
        logger.info("资金费率: %s", rate.get('fundingRate'))
        return rate

    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def batch_create_orders(self, orders: List[Dict[str, Any]]) -> List[Dict]:
        """批量创建订单"""
        if not self.exchange:
            logger.error("同步客户端未初始化")
            return []
        results = []
        for order in orders:
            result = self.create_order(
                symbol=order.get('symbol', 'BTC/USDT'),
                side=order.get('side', 'buy'),
                order_type=order.get('type', 'limit'),
                amount=order.get('amount', 0.001),
                price=order.get('price')
            )
            results.append(result)
        logger.info("批量创建 %d 个订单", len(results))
        return results

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """将时间框架转换为毫秒"""
        seconds_per_unit = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400, '1d': 86400}
        return seconds_per_unit.get(timeframe, 60) * 1000

    def _calculate_optimal_limit(self, start: datetime, end: datetime, timeframe: str) -> int:
        """动态计算最佳请求数量"""
        seconds_per_unit = self._timeframe_to_ms(timeframe) // 1000
        total_seconds = (end - start).total_seconds()
        estimated_bars = int(total_seconds / seconds_per_unit)
        return min(1000, max(100, estimated_bars + 100))

    def _next_start_point(self, last_ts: int, timeframe_ms: int) -> datetime:
        """计算下一次请求的起始时间"""
        return datetime.fromtimestamp((last_ts + 1) / 1000)

    def _to_dataframe(self, data: List[List[Any]]) -> pd.DataFrame:
        """将 OHLCV 数据转换为 DataFrame"""
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
        return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    @retry(max_retries=3, delay=1, exceptions=(ccxt.NetworkError,))
    def create_order(
        self, 
        symbol: str, 
        direction: Union[Direction, str],  # 支持枚举或字符串
        order_type: str, 
        quantity: float, 
        price: Optional[float] = None
    ) -> Dict:
        """处理方向枚举转换"""
        if isinstance(direction, Direction):
            side = direction.value  # 转换为字符串
        else:
            side = direction.lower()
        
        # 格式化数量和价格精度
        formatted_qty = self._format_quantity(symbol, quantity)
        formatted_price = self._format_price(symbol, price) if price else None
        
        return self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=formatted_qty,
            price=formatted_price
        )

    def _format_price(self, symbol: str, price: float) -> float:
        """格式化价格精度"""
        market = self.exchange.market(symbol)
        return self.exchange.price_to_precision(symbol, price)

    def _format_quantity(self, symbol: str, quantity: float) -> float:
        """格式化数量精度"""
        market = self.exchange.market(symbol)
        return self.exchange.amount_to_precision(symbol, quantity)