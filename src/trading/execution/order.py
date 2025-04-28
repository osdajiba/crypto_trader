# src/execution/order.py

import datetime
import time
import functools
from enum import Enum
import uuid
from collections import defaultdict
from typing import Callable, Optional

from src.common.log_manager import LogManager


# 定义枚举类型
class OrderType(Enum):
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    LIMIT_SLTP = 'limit_sltp'  # 限价单带止盈止损
    MARKET_SLTP = 'market_sltp'  # 市价单带止盈止损
    TRAILING_STOP = 'trailing_stop'  # 跟踪止损类型


class Direction(Enum):
    BUY = 'buy'
    SELL = 'sell'
    SHORT_SELL = 'short'


class OrderStatus(Enum):
    CREATED = 'created'
    SUBMITTED = 'submitted'
    PARTIAL = 'partial'
    FILLED = 'filled'
    CANCELED = 'canceled'


class Validity(Enum):
    DAY = 'day'
    GOOD_TILL_CANCEL = 'gtc'


class OrderEventBus:
    """订单事件发布订阅系统"""
    def __init__(self):
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable):
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, order: 'Order'):
        for callback in self.subscribers.get(event_type, []):
            callback(order)


def monitor_performance(func: Callable) -> Callable:
    """订单方法执行监控"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            duration = time.time() - start_time
            self.logger.info(f"{func.__name__} executed in {duration:.4f}s")
            return result
        except Exception as e:
            self.logger.error(f"{func.__name__} failed: {str(e)}")
            raise
    return wrapper


class Order:
    """订单基类"""
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        direction: Direction,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None
    ):
        # 初始化日志和事件总线
        self.logger = LogManager.get_logger("system.order")
        self.event_bus = event_bus or OrderEventBus()

        # 参数验证
        if not isinstance(order_type, OrderType):
            raise TypeError("order_type must be an OrderType enum.")
        if not isinstance(direction, Direction):
            raise TypeError("direction must be a Direction enum.")
        if not isinstance(validity, Validity):
            raise TypeError("validity must be a Validity enum.")
        if not isinstance(quantity, float) or quantity == 0.0:
            raise ValueError("quantity must be a positive integer.")

        # 处理市价单的特殊情况
        if order_type == OrderType.MARKET:
            price = None  # 市价单不指定价格
        elif order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit order must specify a price.")

        # 条件订单验证
        if order_type in (OrderType.STOP_LOSS, OrderType.LIMIT_SLTP, OrderType.MARKET_SLTP):
            if stop_price is None:
                raise ValueError(f"{order_type.value} order must specify a stop_price.")
            if direction == Direction.BUY and stop_price >= (price if price else float('inf')):
                raise ValueError("BUY stop orders require stop_price < entry price")
            if direction == Direction.SELL and stop_price <= (price if price else 0):
                raise ValueError("SELL stop orders require stop_price > entry price")

        if order_type in (OrderType.TAKE_PROFIT, OrderType.LIMIT_SLTP, OrderType.MARKET_SLTP):
            if take_profit_price is None:
                raise ValueError(f"{order_type.value} order must specify a take_profit_price.")
            if direction == Direction.BUY and take_profit_price <= (price if price else 0):
                raise ValueError("BUY take-profit requires price < take_profit_price")
            if direction == Direction.SELL and take_profit_price >= (price if price else float('inf')):
                raise ValueError("SELL take-profit requires price > take_profit_price")

        # 价格精度处理
        # TODO: function _get_price_precision() should be improved
        self._price_precision = self._get_price_precision(symbol)
        self.price = round(price, self._price_precision) if price is not None else None
        self.stop_price = round(stop_price, self._price_precision) if stop_price is not None else None
        self.take_profit_price = round(take_profit_price, self._price_precision) if take_profit_price is not None else None

        # 初始化基础属性
        self.order_id = order_id or str(uuid.uuid4())
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.timestamp = timestamp or datetime.datetime.now()
        # set datetime from timestamp
        if isinstance(self.timestamp, (int, float)):
            timestamp = timestamp / 1000 if self.timestamp > 1e12 else self.timestamp
            self.datetime = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        self.validity = validity
        self.status = OrderStatus.CREATED
        self.filled_quantity = 0
        self.avg_filled_price = 0.0
        self.expiry_time = self._calculate_expiry()
        self.logger.info(f"Order created: {self.order_id} for {self.symbol}")

    def _get_price_precision(self, symbol: str) -> int:
        """获取价格精度（示例实现）"""
        return 2 if "BTC" in symbol else 4

    def _calculate_expiry(self) -> datetime:
        """计算订单过期时间"""
        if self.validity == Validity.DAY:
            if isinstance(self.timestamp, (int, float, datetime)):
                return self.datetime.replace(hour=23, minute=59, second=59)
            else:
                raise TypeError("timestamp must be a datetime, int, or float.")
        return datetime.max  # GTC orders never expire (permanent order)

    def check_validity(self) -> bool:
        """检查订单是否有效"""
        return datetime.datetime.now() < self.expiry_time

    @property
    def remaining_quantity(self) -> int:
        """剩余未成交数量"""
        return self.quantity - self.filled_quantity

    @monitor_performance
    def cancel(self) -> None:
        """取消订单"""
        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELED):
            self.logger.error(f"Cannot cancel order {self.order_id} in {self.status.value} state.")
            raise ValueError(f"Cannot cancel order in {self.status.value} state.")
        
        self.status = OrderStatus.CANCELED
        self.logger.info(f"Order {self.order_id} canceled.")
        self.event_bus.publish('CANCEL', self)

    @monitor_performance
    def fill(self, quantity: int, fill_price: float, liquidity: float) -> None:
        """执行订单填充"""
        if self.status in (OrderStatus.CANCELED, OrderStatus.FILLED):
            self.logger.error(f"Cannot fill order {self.order_id} in {self.status.value} state.")
            raise ValueError(f"Cannot fill order in {self.status.value} state.")
        
        if quantity <= 0 or fill_price <= 0:
            self.logger.error("Fill quantity and price must be positive.")
            raise ValueError("Fill quantity and price must be positive.")

        # 计算实际可成交量
        max_fillable = min(quantity, liquidity, self.remaining_quantity)
        if max_fillable <= 0:
            self.logger.warning("No available liquidity for order %s", self.order_id)
            return

        # 更新成交详情
        total_cost = (self.avg_filled_price * self.filled_quantity) + (fill_price * max_fillable)
        self.filled_quantity += max_fillable
        self.avg_filled_price = total_cost / self.filled_quantity

        # 更新订单状态
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
            self.logger.info(f"Order {self.order_id} fully filled at {self.avg_filled_price:.2f}")
        else:
            self.status = OrderStatus.PARTIAL
            self.logger.info(f"Order {self.order_id} partially filled: {self.filled_quantity}/{self.quantity}")

        self.event_bus.publish('FILL', self)


class MarketOrder(Order):
    """市价单"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        order_id: Optional[str] = None,
        timestamp: Optional[int] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.MARKET,
            direction=direction,
            quantity=quantity,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )


class LimitOrder(Order):
    """限价单"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            direction=direction,
            quantity=quantity,
            price=price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )

    def fill(self, quantity: int, fill_price: float, liquidity: float) -> None:
        """限价单专用填充逻辑"""
        # 验证价格条件
        if self.direction == Direction.BUY and fill_price > self.price:
            self.logger.warning("Fill price exceeds limit price for BUY order")
            return
        if self.direction == Direction.SELL and fill_price < self.price:
            self.logger.warning("Fill price below limit price for SELL order")
            return
        
        super().fill(quantity, fill_price, liquidity)


class StopLossOrder(Order):
    """止损单"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        stop_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.STOP_LOSS,
            direction=direction,
            quantity=quantity,
            stop_price=stop_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )


class TakeProfitOrder(Order):
    """止盈单"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        take_profit_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.TAKE_PROFIT,
            direction=direction,
            quantity=quantity,
            take_profit_price=take_profit_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )


class LimitOrderWithSLTP(Order):
    """限价单带止盈止损"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        price: float,
        stop_price: float,
        take_profit_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.LIMIT_SLTP,
            direction=direction,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )


class MarketOrderWithSLTP(Order):
    """市价单带止盈止损"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        stop_price: float,
        take_profit_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.MARKET_SLTP,
            direction=direction,
            quantity=quantity,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )


class TrailingStopOrder(Order):
    """跟踪止损单"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        trail_value: float,
        is_percentage: bool = True,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.TRAILING_STOP,
            direction=direction,
            quantity=quantity,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity
        )
        self.trail_value = trail_value
        self.is_percentage = is_percentage
        self.activation_price: Optional[float] = None
        self.current_stop_price: Optional[float] = None

    def update_stop_price(self, current_price: float) -> None:
        """更新动态止损价"""
        if self.activation_price is None:
            # 首次激活逻辑
            if (self.direction == Direction.BUY and current_price >= self.activation_price) or \
               (self.direction == Direction.SELL and current_price <= self.activation_price):
                self.activation_price = current_price
        else:
            # 计算偏移量
            offset = self.trail_value * (self.activation_price if self.is_percentage else 1.0)
            new_stop = (
                self.activation_price - offset if self.direction == Direction.BUY else
                self.activation_price + offset
            )
            
            # 更新止损价（只向有利方向移动）
            if self.direction == Direction.BUY:
                self.current_stop_price = max(new_stop, self.current_stop_price or -float('inf'))
            else:
                self.current_stop_price = min(new_stop, self.current_stop_price or float('inf'))