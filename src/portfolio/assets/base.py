#!/usr/bin/env python3
# src/portfolio/assets/base.py

from abc import ABC, abstractmethod
from decimal import Decimal
import asyncio
from typing import Dict, Any, Optional, List, Union, Callable
import uuid
import time

import pandas as pd

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.execution.order import (
    Order, OrderType, Direction, OrderStatus, Validity,
    MarketOrder, LimitOrder, StopLossOrder, TakeProfitOrder, 
    OrderEventBus
)


class Asset(ABC):
    """
    Base class for all portfolio assets with trading capabilities.
    
    This class provides a unified interface for different asset types,
    with optional trading functionality that leverages the execution system.
    """
    
    def __init__(self, name: str, 
                 exchange: Exchange, 
                 execution_engine: BaseExecutionEngine, 
                 config: Optional[ConfigManager] = None, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize an asset.

        Args:
            name: Asset name/symbol
            exchange: Exchange interface (optional)
            config: Configuration manager (optional)
            params: Additional parameters (optional)
        """
        self.name = name
        self.exchange = exchange
        self.execution_engine = execution_engine
        self.config = config if config else ConfigManager()
        self.params = params or {}
        self.logger = LogManager.get_logger(f"asset.{name}")
        
        # Asset state
        self._value = Decimal('0')
        self._position_size = Decimal('0')
        self._last_update_time = 0
        self._last_price = Decimal('0')
        self._initialized = False
        self._update_in_progress = False
        
        # Trading capabilities (optional)
        self.is_tradable = params.get('tradable', False)
        if self.is_tradable:
            self._setup_trading()
        
        # Event subscribers
        self._subscribers = {}
        
        # Status tracking
        self._status = 'created'
        self._errors = []
        
        # Integration with risk system
        self._risk_limits = {}
        
    def _setup_trading(self):
        """Initialize trading capabilities if asset is tradable"""
        # Determine execution mode
        system_mode = self.config.get("system", "operational_mode", default="backtest")
        self.execution_mode = self.params.get('execution_mode', system_mode)
        
        # Initialize execution engine
        self.execution_engine = None  # Will be initialized asynchronously
        self._init_execution_task = None
        
        # Order management
        self.order_event_bus = OrderEventBus()
        self.active_orders = {}
        self.filled_orders = {}
        self.canceled_orders = {}
        self.failed_orders = {}
        
        # Set up order event listeners
        self.order_event_bus.subscribe('FILL', self._on_order_fill)
        self.order_event_bus.subscribe('CANCEL', self._on_order_cancel)
        self.order_event_bus.subscribe('REJECT', self._on_order_reject)
        
        self.logger.info(f"Asset {self.name} initialized with trading capabilities in {self.execution_mode} mode")
    
    async def initialize(self):
        """
        Initialize the asset and its components.
        
        Should be called after creation to properly set up async components.
        """
        if self._initialized:
            return self
            
        try:
            # Additional asset-specific initialization
            await self._initialize_asset()
            
            self._initialized = True
            self._status = 'initialized'
            self.logger.info(f"Asset {self.name} initialized")
        except Exception as e:
            self._status = 'initialization_failed'
            self._errors.append(str(e))
            self.logger.error(f"Failed to initialize asset {self.name}: {str(e)}")
            raise
            
        return self
    
    async def _initialize_asset(self):
        """
        Asset-specific initialization.
        
        Override in subclasses to provide custom initialization logic.
        """
        pass
    
    def _on_order_fill(self, order):
        """
        Handle order fill events
        
        Args:
            order: The order that was filled
        """
        if order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return
            
        if order.status == OrderStatus.FILLED and order.order_id in self.active_orders:
            self.filled_orders[order.order_id] = order
            del self.active_orders[order.order_id]
            
        self._update_position_from_filled_order(order)
        self.logger.info(f"Order {order.order_id} filled: {order.filled_quantity} @ {order.avg_filled_price}")
        
        # Notify subscribers
        self._notify_subscribers('order_filled', {
            'order_id': order.order_id,
            'symbol': self.name,
            'direction': order.direction.value,
            'filled_qty': order.filled_quantity,
            'avg_price': order.avg_filled_price,
            'status': order.status.value
        })
    
    def _on_order_cancel(self, order):
        """
        Handle order cancel events
        
        Args:
            order: The order that was canceled
        """
        if order.status == OrderStatus.CANCELED and order.order_id in self.active_orders:
            self.canceled_orders[order.order_id] = order
            del self.active_orders[order.order_id]
            self.logger.info(f"Order {order.order_id} canceled")
            
            # Notify subscribers
            self._notify_subscribers('order_canceled', {
                'order_id': order.order_id,
                'symbol': self.name,
                'direction': order.direction.value,
                'status': 'canceled'
            })
    
    def _on_order_reject(self, order):
        """
        Handle order rejection events
        
        Args:
            order: The rejected order
        """
        if order.status == OrderStatus.REJECTED and order.order_id in self.active_orders:
            self.failed_orders[order.order_id] = order
            del self.active_orders[order.order_id]
            self.logger.warning(f"Order {order.order_id} rejected")
            
            # Notify subscribers
            self._notify_subscribers('order_rejected', {
                'order_id': order.order_id,
                'symbol': self.name,
                'direction': order.direction.value,
                'reason': getattr(order, 'rejection_reason', 'Unknown reason')
            })

    def _update_position_from_filled_order(self, order):
        """
        Update asset position based on filled order
        
        This method should be implemented by asset subclasses to update
        position size and value based on trade executions.
        
        Args:
            order: The filled order
        """
        # Basic implementation for spot assets
        filled_amount = order.filled_quantity
        filled_price = order.avg_filled_price
        
        if order.direction == Direction.BUY:
            self._position_size += Decimal(str(filled_amount))
            self._value = self._position_size * Decimal(str(filled_price))
        else:  # SELL
            self._position_size -= Decimal(str(filled_amount))
            self._value = self._position_size * Decimal(str(filled_price))
        
        # Update last price
        self._last_price = Decimal(str(filled_price))
        
        # Update last update time
        self._last_update_time = time.time()
        
        # Notify subscribers of position update
        self._notify_subscribers('position_updated', {
            'symbol': self.name,
            'position_size': float(self._position_size),
            'value': float(self._value),
            'last_price': float(self._last_price)
        })
    
    def get_value(self) -> float:
        """
        Get current asset value
        
        Returns:
            Current asset value
        """
        return float(self._value)
    
    def get_position_size(self) -> float:
        """
        Get current position size
        
        Returns:
            Current position size
        """
        return float(self._position_size)
    
    async def update_data(self, data: pd.DataFrame) -> None:
        """
        Update asset with market data
        
        This method should be implemented by asset subclasses to 
        update their state based on market data.
        
        Args:
            data: DataFrame containing market data for this asset
        """
        if data.empty:
            return
            
        try:
            # Avoid concurrent updates
            if self._update_in_progress:
                return
                
            self._update_in_progress = True
            
            # Get latest price from data (typically close price from the latest candle)
            if 'close' in data.columns and len(data) > 0:
                last_row = data.iloc[-1]
                old_price = self._last_price
                self._last_price = Decimal(str(last_row['close']))
                
                # Update value based on new price
                self._value = self._position_size * self._last_price
                self._last_update_time = time.time()
                
                # Calculate price change percentage
                price_change_pct = 0
                if old_price > 0:
                    price_change_pct = (self._last_price - old_price) * 100 / old_price
                
                # Notify subscribers of value changes
                if abs(price_change_pct) > 0:
                    self._notify_subscribers('value_changed', {
                        'symbol': self.name,
                        'old_price': float(old_price),
                        'new_price': float(self._last_price),
                        'change_pct': float(price_change_pct),
                        'position_size': float(self._position_size),
                        'value': float(self._value)
                    })
                    
                self.logger.debug(f"Updated {self.name} with latest price: {float(self._last_price):.2f}, value: {float(self._value):.2f}")
        
        except Exception as e:
            self.logger.error(f"Error updating {self.name} with market data: {str(e)}")
        finally:
            self._update_in_progress = False
            
    async def update_value(self) -> float:
        """
        Update and return current asset value
        
        This method should be implemented by asset subclasses to
        calculate the current value based on market data.
        
        Returns:
            Updated asset value
        """
        # Avoid concurrent updates
        if self._update_in_progress:
            return float(self._value)
            
        self._update_in_progress = True
        
        try:
            # Basic implementation - subclasses should override
            if self._position_size > 0 and self._last_price > 0:
                old_value = self._value
                self._value = self._position_size * self._last_price
                self._last_update_time = time.time()
                
                # Notify subscribers of significant value changes
                if old_value > 0 and abs((self._value - old_value) / old_value) > 0.0:
                    self._notify_subscribers('value_changed', {
                        'symbol': self.name,
                        'old_value': float(old_value),
                        'new_value': float(self._value),
                        'change_pct': float((self._value - old_value) / old_value),
                        'position_size': float(self._position_size),
                        'price': float(self._last_price)
                    })
                    
            return float(self._value)
        finally:
            self._update_in_progress = False
    
    async def _create_order(self, order_type: str, direction: Direction, 
                          quantity: float, **kwargs) -> Order:
        """
        Create an order of the specified type
        
        Args:
            order_type: Type of order ('market', 'limit', etc.)
            direction: Order direction (BUY or SELL)
            quantity: Order quantity
            **kwargs: Additional order parameters
            
        Returns:
            Created order object
            
        Raises:
            ValueError: If asset is not tradable or parameters are invalid
        """
        if not self.is_tradable:
            raise ValueError(f"Asset {self.name} is not tradable")
        
        # Apply risk limits if provided
        if self._risk_limits and 'max_order_size' in self._risk_limits:
            max_order_size = float(self._risk_limits['max_order_size'])
            if quantity > max_order_size:
                self.logger.warning(f"Order size {quantity} exceeds risk limit {max_order_size}, limiting order")
                quantity = max_order_size
        
        order = None
        
        if order_type == 'market':
            order = MarketOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                reduce_only=kwargs.get('reduce_only', False)
            )
        elif order_type == 'limit':
            price = kwargs.get('price')
            if price is None:
                raise ValueError("Limit order requires a price")
            order = LimitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                price=float(price),
                reduce_only=kwargs.get('reduce_only', False)
            )
        elif order_type == 'stop':
            stop_price = kwargs.get('stop_price')
            if stop_price is None:
                raise ValueError("Stop order requires a stop_price")
            order = StopLossOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                stop_price=float(stop_price),
                reduce_only=kwargs.get('reduce_only', True)
            )
        elif order_type == 'take_profit':
            take_profit_price = kwargs.get('take_profit_price')
            if take_profit_price is None:
                raise ValueError("Take profit order requires a take_profit_price")
            order = TakeProfitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                take_profit_price=float(take_profit_price),
                reduce_only=kwargs.get('reduce_only', True)
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Register with event bus
        order.event_bus = self.order_event_bus
        
        # Add to active orders
        self.active_orders[order.order_id] = order
        
        # Notify subscribers
        self._notify_subscribers('order_created', {
            'order_id': order.order_id,
            'symbol': self.name,
            'direction': direction.value,
            'quantity': quantity,
            'order_type': order_type,
            'params': kwargs
        })
        
        return order
    
    async def _execute_order(self, order: Order) -> Dict[str, Any]:
        """
        Execute an order using the execution engine
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result dictionary
        """
        if not self.execution_engine:
            self.logger.warning(f"No execution engine for {self.name}, skipping execution")
            return {"success": False, "error": "No execution engine"}
        
        try:
            # Execute the order
            result = await self.execution_engine.execute_order(order)
            
            # Log result
            if result.get('success', False):
                self.logger.info(f"Order {order.order_id} executed successfully")
                
                # Notify subscribers
                self._notify_subscribers('order_executed', {
                    'order_id': order.order_id,
                    'symbol': self.name,
                    'direction': order.direction.value,
                    'result': result
                })
            else:
                self.logger.warning(f"Order {order.order_id} execution failed: {result.get('error', 'Unknown error')}")
                
                # Notify subscribers
                self._notify_subscribers('order_failed', {
                    'order_id': order.order_id,
                    'symbol': self.name,
                    'direction': order.direction.value,
                    'error': result.get('error', 'Unknown error')
                })
                
            if result["success"]: self.set_quantity(order["amount"])

            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {str(e)}")
            
            # Notify subscribers
            self._notify_subscribers('order_error', {
                'order_id': order.order_id,
                'symbol': self.name,
                'error': str(e)
            })
            
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if canceled successfully, False otherwise
        """
        if not self.is_tradable:
            return False
            
        # Check if order exists in active orders
        if order_id not in self.active_orders:
            return False
        
        try:
            order = self.active_orders[order_id]
            
            # Use execution engine to cancel if available
            if self.execution_engine:
                result = await self.execution_engine.cancel_order(order_id, self.name)
                if result.get('success', False):
                    # Order was canceled by execution engine
                    
                    # Notify subscribers
                    self._notify_subscribers('order_canceled', {
                        'order_id': order_id,
                        'symbol': self.name,
                        'direction': order.direction.value
                    })
                    
                    return True
                    
            # If no execution engine or cancellation failed, update order locally
            order.cancel()
            
            # Notify subscribers
            self._notify_subscribers('order_canceled', {
                'order_id': order_id,
                'symbol': self.name,
                'direction': order.direction.value
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def cancel_all_orders(self) -> Dict[str, bool]:
        """
        Cancel all active orders
        
        Returns:
            Dictionary mapping order IDs to cancellation results
        """
        if not self.is_tradable:
            return {}
        
        results = {}
        for order_id in list(self.active_orders.keys()):
            results[order_id] = await self.cancel_order(order_id)
            
        # Notify subscribers
        if results:
            self._notify_subscribers('all_orders_canceled', {
                'symbol': self.name,
                'order_count': len(results),
                'success_count': sum(1 for success in results.values() if success)
            })
            
        return results
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID to look up
            
        Returns:
            Order object if found, None otherwise
        """
        if not self.is_tradable:
            return None
        
        return (self.active_orders.get(order_id) or 
                self.filled_orders.get(order_id) or 
                self.canceled_orders.get(order_id) or
                self.failed_orders.get(order_id))
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders
        
        Returns:
            List of active order objects
        """
        if not self.is_tradable:
            return []
        
        return list(self.active_orders.values())
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the current status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        if not self.is_tradable:
            return {"success": False, "error": "Trading not available"}
            
        # Check local status first
        order = self.get_order(order_id)
        if order:
            return {
                "success": True,
                "order_id": order_id,
                "symbol": order.symbol,
                "status": order.status.value,
                "filled_qty": order.filled_quantity,
                "unfilled_qty": order.quantity - order.filled_quantity,
                "price": getattr(order, 'price', None),
                "avg_price": order.avg_filled_price,
                "direction": order.direction.value,
                "timestamp": order.timestamp
            }
            
        # If not found locally, check with execution engine
        if self.execution_engine:
            return await self.execution_engine.get_order_status(order_id, self.name)
            
        return {"success": False, "error": "Order not found"}
    
    async def buy(self, kwargs) -> Dict[str, Any]:
        """
        Buy this asset
        
        Args:
            amount: Amount to buy
            **kwargs: Additional parameters including:
                - order_type: Type of order ('market', 'limit', etc.)
                - price: Limit price (required for limit orders)
                - stop_price: Stop price (for stop orders)
                - take_profit_price: Take profit price
                - reduce_only: Whether the order reduces position only
                
        Returns:
            Buy operation result dictionary
        """
        if not self.is_tradable:
            return {"success": False, "error": f"Asset {self.name} is not tradable"}
        
        try:
            amount = kwargs['quantity']
            
            # Basic validation
            if amount <= 0:
                return {
                    "success": False,
                    "error": "Amount must be positive",
                    "direction": "buy",
                    "amount": amount
                }
                
            kwargs_copy = kwargs.copy()
            if 'order_type' in kwargs_copy:
                order_type = kwargs.get('order_type', 'market').lower()
                del kwargs_copy['order_type']
                del kwargs_copy['quantity']
                del kwargs_copy['direction']
                
            order = await self._create_order(order_type, Direction.BUY, amount, **kwargs_copy)
            
            # Execute order if execution engine is available
            execute_result = await self._execute_order(order)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "sell",
                "price": order.price if not order_type=='market' else kwargs.price,
                "amount": amount,
                "status": order.status.value,
                "execute_result": execute_result
            }
            
        except Exception as e:
            self.logger.error(f"Buy error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "buy",
                "amount": amount
            }
    
    async def sell(self, kwargs) -> Dict[str, Any]:
        """
        Sell this asset
        
        Args:
            amount: Amount to sell
            **kwargs: Additional parameters including:
                - order_type: Type of order ('market', 'limit', etc.)
                - price: Limit price (required for limit orders)
                - stop_price: Stop price (for stop orders)
                - take_profit_price: Take profit price
                - reduce_only: Whether the order reduces position only
                
        Returns:
            Sell operation result dictionary
        """
        if not self.is_tradable:
            return {"success": False, "error": f"Asset {self.name} is not tradable"}
        
        try:
            # Basic validation
            amount = kwargs['quantity']
            if amount <= 0:
                return {
                    "success": False,
                    "error": "Amount must be positive",
                    "direction": "sell",
                    "amount": amount
                }
                
            # Check if we have enough to sell (for spot assets)
            if not kwargs.get('allow_partial', False) and isinstance(self._position_size, Decimal) and amount > float(self._position_size):
                return {
                    "success": False,
                    "error": f"Insufficient position: have {float(self._position_size)}, need {amount}",
                    "direction": "sell",
                    "amount": amount
                }
                
            kwargs_copy = kwargs.copy()
            if 'order_type' in kwargs_copy:
                order_type = kwargs.get('order_type', 'market').lower()
                del kwargs_copy['order_type']
                del kwargs_copy['quantity']
                del kwargs_copy['direction']
                
                
            order = await self._create_order(order_type, Direction.BUY, amount, **kwargs_copy)
            
            # Execute order if execution engine is available
            if self.execution_engine:
                execute_result = await self._execute_order(order)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "sell",
                "price": order.price if not order_type=='market' else kwargs.price,
                "amount": amount,
                "status": order.status.value,
                "execute_result": execute_result
            }
        except Exception as e:
            self.logger.error(f"Sell error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "sell",
                "amount": amount
            }
    
    def set_exchange(self, exchange) -> None:
        """
        Set/update the exchange interface
        
        Args:
            exchange: Exchange interface
        """
        self.exchange = exchange
        
        # Notify subscribers
        self._notify_subscribers('exchange_updated', {
            'symbol': self.name,
            'exchange': exchange.__class__.__name__
        })
    
    def set_risk_limits(self, limits: Dict[str, Any]) -> None:
        """
        Set risk limits for this asset
        
        Args:
            limits: Risk limits dictionary
        """
        self._risk_limits = limits
        self.logger.info(f"Updated risk limits for {self.name}")
        
        # Notify subscribers
        self._notify_subscribers('risk_limits_updated', {
            'symbol': self.name,
            'limits': limits
        })
        
    def set_quantity(self, new_quantity):
        self.quantity = new_quantity
        
    def subscribe(self, event_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Subscribe to asset events
        
        Args:
            event_type: Event type to subscribe to
            callback: Callback function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
            
        self._subscribers[event_type].append(callback)
        self.logger.debug(f"Added subscriber for {event_type} events")
    
    def _notify_subscribers(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Notify subscribers of an event
        
        Args:
            event_type: Event type
            event_data: Event data
        """
        if event_type not in self._subscribers:
            return
            
        for callback in self._subscribers[event_type]:
            try:
                callback(event_data)
            except Exception as e:
                self.logger.error(f"Error in subscriber callback for {event_type}: {str(e)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert asset to dictionary representation
        
        Returns:
            Asset information dictionary
        """
        data = {
            'name': self.name,
            'type': self.__class__.__name__,
            'value': float(self._value),
            'position_size': float(self._position_size),
            'last_price': float(self._last_price),
            'last_update': self._last_update_time,
            'tradable': self.is_tradable,
            'status': self._status
        }
        
        if self.is_tradable:
            data.update({
                'active_orders': len(self.active_orders),
                'filled_orders': len(self.filled_orders),
                'canceled_orders': len(self.canceled_orders),
                'failed_orders': len(self.failed_orders),
                'execution_mode': self.execution_mode
            })
            
        return data
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed asset status
        
        Returns:
            Detailed status dictionary
        """
        return {
            'name': self.name,
            'status': self._status,
            'initialized': self._initialized,
            'last_update': self._last_update_time,
            'errors': self._errors,
            'position_size': float(self._position_size),
            'value': float(self._value),
            'last_price': float(self._last_price)
        }
    
    async def close(self):
        """
        Close the asset and clean up resources
        """
        if self.is_tradable:
            # Cancel all active orders
            await self.cancel_all_orders()
            
            # Close execution engine
            if self.execution_engine and hasattr(self.execution_engine, 'close'):
                await self.execution_engine.close()
        
        # Reset subscriptions
        self._subscribers = {}
        
        # Update status
        self._status = 'closed'
        self._initialized = False
        
        self.logger.info(f"Closed asset {self.name}")