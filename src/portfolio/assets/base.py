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
    Abstract base class for all portfolio assets with trading capabilities.
    Defines the interface all tradable assets must implement.
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
            exchange: Exchange interface
            execution_engine: Execution engine for order management 
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
        
        # Trading capabilities
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
        
    @abstractmethod
    async def get_value(self) -> float:
        """
        Get current asset value
        
        Returns:
            Current asset value
        """
        pass
    
    @abstractmethod
    async def update_value(self) -> float:
        """
        Update and return current asset value
        
        Returns:
            Updated asset value
        """
        pass
    
    @abstractmethod
    async def update_data(self, data: pd.DataFrame) -> None:
        """
        Update asset with market data
        
        Args:
            data: DataFrame containing market data for this asset
        """
        pass
        
    async def buy(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Buy this asset
        
        Args:
            params: Dictionary with order parameters:
                quantity: Amount to buy
                price: Price for limit orders (optional)
                order_type: Order type (market, limit, etc.)
                
        Returns:
            Dict with order result
        """
        if not self.is_tradable:
            return {"success": False, "error": f"Asset {self.name} is not tradable"}
        
        try:
            # Extract parameters
            quantity = params['quantity']
            if quantity <= 0:
                return {
                    "success": False,
                    "error": "Quantity must be positive",
                    "direction": "buy",
                    "quantity": quantity
                }
            
            # Create order
            order_type = params.get('order_type', 'market').lower()
            
            # Create a copy of params for passing to _create_order
            order_params = params.copy()
            # Remove parameters that will be passed directly
            if 'order_type' in order_params:
                del order_params['order_type']
            if 'quantity' in order_params:
                del order_params['quantity']
            if 'direction' in order_params:
                del order_params['direction']
            
            # Create the order object
            order = await self._create_order(order_type, Direction.BUY, quantity, **order_params)
            
            # Execute order if execution engine is available
            execute_result = await self._execute_order(order)
            
            # Return result
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "buy",
                "price": getattr(order, 'price', params.get('price')),
                "quantity": quantity,
                "status": order.status.value,
                "execute_result": execute_result
            }
        except Exception as e:
            self.logger.error(f"Buy error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "buy",
                "quantity": params.get('quantity', 0)
            }
    
    async def sell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sell this asset
        
        Args:
            params: Dictionary with order parameters:
                quantity: Amount to sell
                price: Price for limit orders (optional)
                order_type: Order type (market, limit, etc.)
                
        Returns:
            Dict with order result
        """
        if not self.is_tradable:
            return {"success": False, "error": f"Asset {self.name} is not tradable"}
        
        try:
            # Extract parameters
            quantity = params['quantity']
            if quantity <= 0:
                return {
                    "success": False,
                    "error": "Quantity must be positive",
                    "direction": "sell",
                    "quantity": quantity
                }
            
            # Check if we have enough to sell (for spot assets)
            if not params.get('allow_partial', False) and isinstance(self._position_size, Decimal) and quantity > float(self._position_size):
                return {
                    "success": False,
                    "error": f"Insufficient position: have {float(self._position_size)}, need {quantity}",
                    "direction": "sell",
                    "quantity": quantity
                }
            
            # Create order
            order_type = params.get('order_type', 'market').lower()
            
            # Create a copy of params for passing to _create_order
            order_params = params.copy()
            # Remove parameters that will be passed directly
            if 'order_type' in order_params:
                del order_params['order_type']
            if 'quantity' in order_params:
                del order_params['quantity']
            if 'direction' in order_params:
                del order_params['direction']
            
            # Create the order object
            order = await self._create_order(order_type, Direction.SELL, quantity, **order_params)
            
            # Execute order if execution engine is available
            execute_result = await self._execute_order(order)
            
            # Return result
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "sell",
                "price": getattr(order, 'price', params.get('price')),
                "quantity": quantity,
                "status": order.status.value,
                "execute_result": execute_result
            }
        except Exception as e:
            self.logger.error(f"Sell error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "sell",
                "quantity": params.get('quantity', 0)
            }
                
    def get_position_size(self) -> float:
        """
        Get current position size
        
        Returns:
            Current position size
        """
        return float(self._position_size)
    
    def get_position_type(self) -> str:
        """
        Get current position type
        
        Returns:
            Position type ('long', 'short', 'flat')
        """
        # Default implementation - subclasses should override if needed
        if hasattr(self, '_position_type'):
            return self._position_type
        
        # For simple assets, infer from position size
        if self._position_size > 0:
            return 'long'
        return 'flat'
        
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an active order
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            Dict with cancellation result
        """
        if not self.is_tradable:
            return {"success": False, "error": "Trading not available"}
            
        # Check if order exists in active orders
        if order_id not in self.active_orders:
            return {"success": False, "error": "Order not found"}
        
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
                    
                    return result
                    
            # If no execution engine or cancellation failed, update order locally
            order.cancel()
            
            # Notify subscribers
            self._notify_subscribers('order_canceled', {
                'order_id': order_id,
                'symbol': self.name,
                'direction': order.direction.value
            })
            
            return {
                "success": True,
                "order_id": order_id,
                "symbol": self.name,
                "status": "canceled"
            }
            
        except Exception as e:
            self.logger.error(f"Error canceling order {order_id}: {str(e)}")
            return {
                "success": False,
                "order_id": order_id,
                "error": str(e)
            }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get current status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict with order status information
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
    
    async def open_long_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a long position
        
        Args:
            quantity: Position size
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        return await self.buy({'quantity': quantity, 'symbol': self.name, **kwargs})
        
    async def close_long_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a long position
        
        Args:
            quantity: Position size to close (default: all)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        # If position is not long, nothing to close
        if self._position_size <= 0:
            return {"success": False, "error": "No long position to close"}
            
        # Default to closing entire position
        if quantity is None:
            quantity = float(self._position_size)
            
        return await self.sell({'quantity': quantity, 'symbol': self.name, **kwargs})
        
    async def open_short_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a short position
        
        Args:
            quantity: Position size
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        return await self.sell({'quantity': quantity, 'symbol': self.name, **kwargs})
        
    async def close_short_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a short position
        
        Args:
            quantity: Position size to close (default: all)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        # If position is not short, nothing to close
        if self._position_size >= 0:
            return {"success": False, "error": "No short position to close"}
            
        # Default to closing entire position
        if quantity is None:
            quantity = abs(float(self._position_size))
            
        return await self.buy({'quantity': quantity, 'symbol': self.name, **kwargs})
    
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