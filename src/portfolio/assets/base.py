#!/usr/bin/env python3
# src/portfolio/assets/base.py

from abc import ABC, abstractmethod
from decimal import Decimal
import asyncio
from typing import Dict, Any, Optional, List, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.execution.factory import get_execution_factory
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
    
    def __init__(self, name: str, exchange=None, config: Optional[ConfigManager] = None, 
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
        self.config = config if config else ConfigManager()
        self.params = params or {}
        self.logger = LogManager.get_logger(f"asset.{name}")
        
        # Asset state
        self._value = Decimal('0')
        self._position_size = Decimal('0')
        self._last_update_time = 0
        self._last_price = Decimal('0')
        
        # Trading capabilities (optional)
        self.is_tradable = params.get('tradable', False)
        if self.is_tradable:
            self._setup_trading()
        
    def _setup_trading(self):
        """Initialize trading capabilities if asset is tradable"""
        # Get execution engine factory
        execution_factory = get_execution_factory(self.config)
        
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
        
        # Set up order event listeners
        self.order_event_bus.subscribe('FILL', self._on_order_fill)
        self.order_event_bus.subscribe('CANCEL', self._on_order_cancel)
        
        self.logger.info(f"Asset {self.name} initialized with trading capabilities in {self.execution_mode} mode")
    
    async def initialize(self):
        """
        Initialize the asset and its components.
        
        Should be called after creation to properly set up async components.
        """
        if self.is_tradable:
            await self._initialize_execution_engine()
            
        # Additional asset-specific initialization
        await self._initialize_asset()
        
        self.logger.info(f"Asset {self.name} initialized")
        return self
    
    async def _initialize_asset(self):
        """
        Asset-specific initialization.
        
        Override in subclasses to provide custom initialization logic.
        """
        pass
    
    async def _initialize_execution_engine(self):
        """Initialize the execution engine"""
        try:
            # Get execution factory
            execution_factory = get_execution_factory(self.config)
            
            # Create appropriate engine based on mode
            historical_data = self.params.get('historical_data', {})
            if self.name in historical_data:
                # Single asset data
                data = {self.name: historical_data[self.name]}
            elif isinstance(historical_data, dict) and historical_data:
                # Multi-asset data
                data = historical_data
            else:
                # No data
                data = None
            
            # Create the execution engine
            self.execution_engine = await execution_factory.create(
                name=self.execution_mode,
                params={
                    "historical_data": data,
                    "exchange": self.exchange
                }
            )
            
            self.logger.info(f"Execution engine initialized for {self.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize execution engine: {str(e)}")
            raise
    
    def _on_order_fill(self, order):
        """
        Handle order fill events
        
        Args:
            order: The order that was filled
        """
        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            if order.status == OrderStatus.FILLED and order.order_id in self.active_orders:
                self.filled_orders[order.order_id] = order
                del self.active_orders[order.order_id]
            
            self._update_position_from_filled_order(order)
            self.logger.info(f"Order {order.order_id} filled: {order.filled_quantity} @ {order.avg_filled_price}")
    
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
    
    async def update_value(self) -> float:
        """
        Update and return current asset value
        
        This method should be implemented by asset subclasses to
        calculate the current value based on market data.
        
        Returns:
            Updated asset value
        """
        # Basic implementation - subclasses should override
        if self._position_size > 0 and self._last_price > 0:
            self._value = self._position_size * self._last_price
        return float(self._value)
    
    async def create_order(self, order_type: str, direction: Direction, 
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
        
        # Execute order if execution engine is available
        if self.execution_engine:
            await self._execute_order(order)
        
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
            else:
                self.logger.warning(f"Order {order.order_id} execution failed: {result.get('error', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {str(e)}")
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
                    return True
                    
            # If no execution engine or cancellation failed, update order locally
            order.cancel()
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
                self.canceled_orders.get(order_id))
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders
        
        Returns:
            List of active order objects
        """
        if not self.is_tradable:
            return []
        
        return list(self.active_orders.values())
    
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
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
            order_type = kwargs.get('order_type', 'market').lower()
            order = await self.create_order(order_type, Direction.BUY, amount, **kwargs)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "buy",
                "amount": amount,
                "status": order.status.value
            }
        except Exception as e:
            self.logger.error(f"Buy error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "buy",
                "amount": amount
            }
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
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
            order_type = kwargs.get('order_type', 'market').lower()
            order = await self.create_order(order_type, Direction.SELL, amount, **kwargs)
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "sell",
                "amount": amount,
                "status": order.status.value
            }
        except Exception as e:
            self.logger.error(f"Sell error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "sell",
                "amount": amount
            }
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the current status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dictionary
        """
        if not self.is_tradable or not self.execution_engine:
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
    
    def set_exchange(self, exchange) -> None:
        """
        Set/update the exchange interface
        
        Args:
            exchange: Exchange interface
        """
        self.exchange = exchange
    
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
            'tradable': self.is_tradable
        }
        
        if self.is_tradable:
            data.update({
                'active_orders': len(self.active_orders),
                'filled_orders': len(self.filled_orders),
                'execution_mode': self.execution_mode
            })
            
        return data
    
    async def close(self):
        """
        Close the asset and clean up resources
        """
        if self.is_tradable:
            # Cancel all active orders
            await self.cancel_all_orders()
            
            # Close execution engine
            if self.execution_engine:
                await self.execution_engine.close()
        
        self.logger.info(f"Closed asset {self.name}")
        
    async def shutdown(self) -> None:
        """
        Clean up resources
        """
        # Call subclass-specific shutdown
        await self._shutdown_specific()
        
        # Reset state
        self._initialized = False
        
        self.logger.info(f"{self.__class__.__name__} shutdown completed")
    
    async def _shutdown_specific(self) -> None:
        """
        Specific shutdown operations for subclasses
        """
        pass