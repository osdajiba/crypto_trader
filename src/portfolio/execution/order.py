#!/usr/bin/env python3
# src/portfolio/execution/order.py

import datetime
import time
import functools
from enum import Enum
import uuid
from collections import defaultdict
from typing import Callable, Optional, Dict, Any, Union, List

from src.common.log_manager import LogManager
from src.exchange.base import Exchange


class OrderType(Enum):
    """Order type enumeration for categorizing trading orders"""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    LIMIT_SLTP = 'limit_sltp'      # Limit order with stop loss and take profit
    MARKET_SLTP = 'market_sltp'    # Market order with stop loss and take profit
    TRAILING_STOP = 'trailing_stop' # Trailing stop order type


class Direction(Enum):
    """Order direction enumeration for specifying trade direction"""
    BUY = 'buy'
    SELL = 'sell'
    SHORT_SELL = 'short'


class OrderStatus(Enum):
    """Order status enumeration for tracking order lifecycle"""
    CREATED = 'created'      # Initial state
    SUBMITTED = 'submitted'  # Submitted to exchange
    PARTIAL = 'partial'      # Partially filled
    FILLED = 'filled'        # Completely filled
    CANCELED = 'canceled'    # Canceled before complete fill
    REJECTED = 'rejected'    # Rejected by exchange
    EXPIRED = 'expired'      # Expired based on validity period
    FAILED = 'failed'        # Failed due to error


class Validity(Enum):
    """Order validity enumeration for time in force settings"""
    DAY = 'day'                   # Valid for the current trading day
    GOOD_TILL_CANCEL = 'gtc'      # Valid until explicitly canceled
    IMMEDIATE_OR_CANCEL = 'ioc'   # Fill immediately or cancel
    FILL_OR_KILL = 'fok'          # Fill completely or cancel


class OrderEventBus:
    """
    Event system for order-related events with pub/sub pattern.
    
    Provides a simple event bus for handling order lifecycle events,
    allowing components to subscribe to order events and react accordingly.
    """
    def __init__(self):
        """Initialize the event bus with empty subscriber lists"""
        self.subscribers = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to a specific event type
        
        Args:
            event_type: The event type to subscribe to
            callback: Function to call when the event occurs
        """
        self.subscribers[event_type].append(callback)

    def publish(self, event_type: str, order: 'Order'):
        """
        Publish an event to all subscribers
        
        Args:
            event_type: The event type being published
            order: The order object associated with the event
        """
        for callback in self.subscribers.get(event_type, []):
            callback(order)


def monitor_performance(func: Callable) -> Callable:
    """
    Decorator for monitoring order method performance
    
    Args:
        func: The function to monitor
        
    Returns:
        Wrapped function with performance monitoring
    """
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
    """
    Base class for all trading orders
    
    Represents a trade order with all necessary properties and
    lifecycle management methods.
    """
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
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None        
    ):
        # Initialize logger and event bus
        self.logger = LogManager.get_logger("execution.order")
        self.event_bus = event_bus or OrderEventBus()
        self.exchange = exchange

        # Parameter validation
        if not isinstance(order_type, OrderType):
            raise TypeError("order_type must be an OrderType enum.")
        if not isinstance(direction, Direction):
            raise TypeError("direction must be a Direction enum.")
        if not isinstance(validity, Validity):
            raise TypeError("validity must be a Validity enum.")
        if not isinstance(quantity, float) or quantity <= 0.0:
            raise ValueError("quantity must be a positive number.")

        # Handle special cases for market orders
        if order_type == OrderType.MARKET:
            price = None  # Market orders don't specify price
        elif order_type == OrderType.LIMIT and price is None:
            raise ValueError("Limit order must specify a price.")

        # Conditional order validation
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

        # Price precision handling
        self._price_precision = self._get_price_precision(symbol)
        self.price = round(price, self._price_precision) if price is not None else None
        self.stop_price = round(stop_price, self._price_precision) if stop_price is not None else None
        self.take_profit_price = round(take_profit_price, self._price_precision) if take_profit_price is not None else None

        # Initialize basic properties
        self.order_id = order_id or str(uuid.uuid4())
        self.symbol = symbol
        self.order_type = order_type
        self.direction = direction
        self.quantity = quantity
        self.timestamp = timestamp or datetime.datetime.now()
        
        # Convert timestamp to datetime if needed
        if isinstance(self.timestamp, (int, float)):
            timestamp_val = self.timestamp / 1000 if self.timestamp > 1e12 else self.timestamp
            self.datetime = datetime.datetime.fromtimestamp(timestamp_val, tz=datetime.timezone.utc)
        else:
            self.datetime = self.timestamp
            
        self.validity = validity
        self.status = OrderStatus.CREATED
        self.filled_quantity = 0.0
        self.avg_filled_price = 0.0
        self.expiry_time = self._calculate_expiry()
        
        # Reduce-only flag (for futures trading)
        self.reduce_only = False
        
        # External reference IDs
        self.exchange_order_id = None
        
        self.logger.info(f"Order created: {self.order_id} for {self.symbol}")

    def _get_price_precision(self, symbol: str) -> int:
        """
        Get price precision for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Price precision (decimal places)
        """
        # This is a simplified implementation
        # In production, should fetch from exchange info
        return 2 if "BTC" in symbol.upper() else 4

    def _calculate_expiry(self) -> datetime.datetime:
        """
        Calculate order expiry time based on validity
        
        Returns:
            Expiry datetime
        """
        if self.validity == Validity.DAY:
            if isinstance(self.datetime, datetime.datetime):
                # Set to end of current day
                return self.datetime.replace(hour=23, minute=59, second=59, microsecond=999999)
            else:
                raise TypeError("timestamp must be a datetime")
        
        # For GTC orders, set far future date
        return datetime.datetime.max

    def check_validity(self) -> bool:
        """
        Check if order is still valid (not expired)
        
        Returns:
            True if order is valid, False otherwise
        """
        return datetime.datetime.now() < self.expiry_time

    @property
    def remaining_quantity(self) -> float:
        """
        Get remaining unfilled quantity
        
        Returns:
            Unfilled quantity
        """
        return self.quantity - self.filled_quantity

    def set_exchange_id(self, exchange_id: str) -> None:
        """
        Set exchange order ID after submission
        
        Args:
            exchange_id: Exchange-assigned order ID
        """
        self.exchange_order_id = exchange_id
        if self.status == OrderStatus.CREATED:
            self.status = OrderStatus.SUBMITTED
            
    def set_status(self, status: OrderStatus) -> None:
        """
        Explicitly set order status
        
        Args:
            status: New order status
        """
        if status != self.status:
            old_status = self.status
            self.status = status
            self.logger.info(f"Order {self.order_id} status changed: {old_status.value} -> {status.value}")

    @monitor_performance
    def cancel(self) -> None:
        """
        Cancel the order
        
        Raises:
            ValueError: If order cannot be canceled
        """
        if self.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
            self.logger.error(f"Cannot cancel order {self.order_id} in {self.status.value} state.")
            raise ValueError(f"Cannot cancel order in {self.status.value} state.")
        
        self.status = OrderStatus.CANCELED
        self.logger.info(f"Order {self.order_id} canceled.")
        self.event_bus.publish('CANCEL', self)

    @monitor_performance
    def fill(self, quantity: float, fill_price: float, liquidity: float) -> None:
        """
        Execute order fill (partial or complete)
        
        Args:
            quantity: Quantity to fill
            fill_price: Fill price
            liquidity: Available liquidity
            
        Raises:
            ValueError: If order cannot be filled
        """
        if self.status in (OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED):
            self.logger.error(f"Cannot fill order {self.order_id} in {self.status.value} state.")
            raise ValueError(f"Cannot fill order in {self.status.value} state.")
        
        if quantity <= 0 or fill_price <= 0:
            self.logger.error("Fill quantity and price must be positive.")
            raise ValueError("Fill quantity and price must be positive.")

        # Calculate maximum fillable quantity based on available liquidity and remaining quantity
        max_fillable = min(quantity, liquidity, self.remaining_quantity)
        if max_fillable <= 0:
            self.logger.warning(f"No available liquidity for order {self.order_id}")
            return

        # Update fill details
        total_cost = (self.avg_filled_price * self.filled_quantity) + (fill_price * max_fillable)
        self.filled_quantity += max_fillable
        self.avg_filled_price = total_cost / self.filled_quantity if self.filled_quantity > 0 else 0.0

        # Update order status
        if self.filled_quantity >= self.quantity:
            # Round to handle floating point precision issues
            self.filled_quantity = self.quantity
            self.status = OrderStatus.FILLED
            self.logger.info(f"Order {self.order_id} fully filled at {self.avg_filled_price:.6f}")
        else:
            self.status = OrderStatus.PARTIAL
            self.logger.info(f"Order {self.order_id} partially filled: {self.filled_quantity:.6f}/{self.quantity:.6f}")

        # Publish fill event
        self.event_bus.publish('FILL', self)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert order to dictionary
        
        Returns:
            Dictionary representation of order
        """
        return {
            'order_id': self.order_id,
            'exchange_order_id': self.exchange_order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.value,
            'direction': self.direction.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'take_profit_price': self.take_profit_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'avg_filled_price': self.avg_filled_price,
            'timestamp': self.timestamp,
            'validity': self.validity.value,
            'reduce_only': self.reduce_only,
        }


class MarketOrder(Order):
    """Market order implementation"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None,
        reduce_only: bool = False
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.MARKET,
            direction=direction,
            quantity=quantity,
            price=None,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        self.reduce_only = reduce_only


class LimitOrder(Order):
    """Limit order implementation"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None,
        reduce_only: bool = False
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.LIMIT,
            direction=direction,
            quantity=quantity,
            price=price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        self.reduce_only = reduce_only

    def fill(self, quantity: float, fill_price: float, liquidity: float) -> None:
        """
        Special fill logic for limit orders
        
        Args:
            quantity: Quantity to fill
            fill_price: Fill price
            liquidity: Available liquidity
        """
        # Validate price conditions
        if self.direction == Direction.BUY and fill_price > self.price:
            self.logger.warning(f"Fill price {fill_price} exceeds limit price {self.price} for BUY order")
            return
        
        if self.direction == Direction.SELL and fill_price < self.price:
            self.logger.warning(f"Fill price {fill_price} below limit price {self.price} for SELL order")
            return
        
        # Use parent class fill implementation
        super().fill(quantity, fill_price, liquidity)


class StopLossOrder(Order):
    """Stop-loss order implementation"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        stop_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None,
        reduce_only: bool = True  # Stop losses are usually reduce-only
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.STOP_LOSS,
            direction=direction,
            quantity=quantity,
            stop_price=stop_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        self.reduce_only = reduce_only
        self.triggered = False
        
    def check_trigger(self, current_price: float) -> bool:
        """
        Check if stop order should be triggered
        
        Args:
            current_price: Current market price
            
        Returns:
            True if triggered, False otherwise
        """
        if self.triggered:
            return True
            
        if (self.direction == Direction.SELL and current_price <= self.stop_price) or \
           (self.direction == Direction.BUY and current_price >= self.stop_price):
            self.triggered = True
            self.logger.info(f"Stop order {self.order_id} triggered at {current_price}")
            return True
            
        return False


class TakeProfitOrder(Order):
    """Take-profit order implementation"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        take_profit_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None,
        reduce_only: bool = True  # Take profits are usually reduce-only
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.TAKE_PROFIT,
            direction=direction,
            quantity=quantity,
            take_profit_price=take_profit_price,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        self.reduce_only = reduce_only
        self.triggered = False
        
    def check_trigger(self, current_price: float) -> bool:
        """
        Check if take profit order should be triggered
        
        Args:
            current_price: Current market price
            
        Returns:
            True if triggered, False otherwise
        """
        if self.triggered:
            return True
            
        if (self.direction == Direction.SELL and current_price >= self.take_profit_price) or \
           (self.direction == Direction.BUY and current_price <= self.take_profit_price):
            self.triggered = True
            self.logger.info(f"Take profit order {self.order_id} triggered at {current_price}")
            return True
            
        return False


class LimitOrderWithSLTP(Order):
    """Limit order with stop-loss and take-profit levels"""
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
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None
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
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        # Child orders created after the main order fills
        self.stop_loss_order = None
        self.take_profit_order = None
        
    def on_fill(self) -> List[Order]:
        """
        Create child orders when parent order is filled
        
        Returns:
            List of child orders (SL and TP)
        """
        if self.status != OrderStatus.FILLED:
            return []
            
        child_orders = []
        
        # Create stop loss order
        self.stop_loss_order = StopLossOrder(
            symbol=self.symbol,
            direction=Direction.SELL if self.direction == Direction.BUY else Direction.BUY,
            quantity=self.quantity,
            stop_price=self.stop_price,
            order_id=f"{self.order_id}_sl",
            timestamp=datetime.datetime.now(),
            validity=self.validity,
            event_bus=self.event_bus,
            exchange=self.exchange,
            reduce_only=True
        )
        child_orders.append(self.stop_loss_order)
        
        # Create take profit order
        self.take_profit_order = TakeProfitOrder(
            symbol=self.symbol,
            direction=Direction.SELL if self.direction == Direction.BUY else Direction.BUY,
            quantity=self.quantity,
            take_profit_price=self.take_profit_price,
            order_id=f"{self.order_id}_tp",
            timestamp=datetime.datetime.now(),
            validity=self.validity,
            event_bus=self.event_bus,
            exchange=self.exchange,
            reduce_only=True
        )
        child_orders.append(self.take_profit_order)
        
        return child_orders


class MarketOrderWithSLTP(Order):
    """Market order with stop-loss and take-profit levels"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        stop_price: float,
        take_profit_price: float,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None
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
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        # Child orders created after the main order fills
        self.stop_loss_order = None
        self.take_profit_order = None
        
    def on_fill(self) -> List[Order]:
        """
        Create child orders when parent order is filled
        
        Returns:
            List of child orders (SL and TP)
        """
        if self.status != OrderStatus.FILLED:
            return []
            
        child_orders = []
        
        # Create stop loss order
        self.stop_loss_order = StopLossOrder(
            symbol=self.symbol,
            direction=Direction.SELL if self.direction == Direction.BUY else Direction.BUY,
            quantity=self.quantity,
            stop_price=self.stop_price,
            order_id=f"{self.order_id}_sl",
            timestamp=datetime.datetime.now(),
            validity=self.validity,
            event_bus=self.event_bus,
            exchange=self.exchange,
            reduce_only=True
        )
        child_orders.append(self.stop_loss_order)
        
        # Create take profit order
        self.take_profit_order = TakeProfitOrder(
            symbol=self.symbol,
            direction=Direction.SELL if self.direction == Direction.BUY else Direction.BUY,
            quantity=self.quantity,
            take_profit_price=self.take_profit_price,
            order_id=f"{self.order_id}_tp",
            timestamp=datetime.datetime.now(),
            validity=self.validity,
            event_bus=self.event_bus,
            exchange=self.exchange,
            reduce_only=True
        )
        child_orders.append(self.take_profit_order)
        
        return child_orders


class TrailingStopOrder(Order):
    """Trailing stop order implementation"""
    def __init__(
        self,
        symbol: str,
        direction: Direction,
        quantity: float,
        trail_value: float,
        is_percentage: bool = True,
        order_id: Optional[str] = None,
        timestamp: Optional[datetime.datetime] = None,
        validity: Validity = Validity.DAY,
        event_bus: Optional[OrderEventBus] = None,
        exchange: Optional[Exchange] = None,
        reduce_only: bool = True
    ):
        super().__init__(
            symbol=symbol,
            order_type=OrderType.TRAILING_STOP,
            direction=direction,
            quantity=quantity,
            order_id=order_id,
            timestamp=timestamp,
            validity=validity,
            event_bus=event_bus,
            exchange=exchange
        )
        self.trail_value = trail_value
        self.is_percentage = is_percentage
        self.reduce_only = reduce_only
        self.activation_price = None
        self.current_stop_price = None
        self.triggered = False

    def update_stop_price(self, current_price: float) -> None:
        """
        Update trailing stop price based on current market price
        
        Args:
            current_price: Current market price
        """
        if self.activation_price is None:
            # Initialize activation price on first update
            self.activation_price = current_price
            
            # Set initial stop price
            offset = self.trail_value * (current_price if self.is_percentage else 1.0)
            if self.direction == Direction.BUY:
                self.current_stop_price = current_price - offset
            else:
                self.current_stop_price = current_price + offset
                
            self.logger.info(f"Trailing stop initialized: activation={current_price}, stop={self.current_stop_price}")
            return
            
        # For BUY trailing stop, only move the stop price up when the price increases
        if self.direction == Direction.BUY:
            if current_price > self.activation_price:
                # Calculate new stop price
                offset = self.trail_value * (current_price if self.is_percentage else 1.0)
                new_stop = current_price - offset
                
                # Only update if new stop is higher
                if new_stop > self.current_stop_price:
                    self.current_stop_price = new_stop
                    self.activation_price = current_price
                    self.logger.debug(f"Updated BUY trailing stop: activation={current_price}, stop={self.current_stop_price}")
        
        # For SELL trailing stop, only move the stop price down when the price decreases
        else:
            if current_price < self.activation_price:
                # Calculate new stop price
                offset = self.trail_value * (current_price if self.is_percentage else 1.0)
                new_stop = current_price + offset
                
                # Only update if new stop is lower
                if new_stop < self.current_stop_price:
                    self.current_stop_price = new_stop
                    self.activation_price = current_price
                    self.logger.debug(f"Updated SELL trailing stop: activation={current_price}, stop={self.current_stop_price}")

    def check_trigger(self, current_price: float) -> bool:
        """
        Check if trailing stop should be triggered
        
        Args:
            current_price: Current market price
            
        Returns:
            True if triggered, False otherwise
        """
        if self.triggered or self.current_stop_price is None:
            return self.triggered
            
        if (self.direction == Direction.BUY and current_price <= self.current_stop_price) or \
           (self.direction == Direction.SELL and current_price >= self.current_stop_price):
            self.triggered = True
            self.logger.info(f"Trailing stop {self.order_id} triggered at {current_price}, stop level: {self.current_stop_price}")
            return True
            
        return False