#!/usr/bin/env python3
# src/portfolio/assets/tradable_asset.py

import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union

from src.portfolio.assets.base import Asset
from src.common.log_manager import LogManager
from src.portfolio.execution.order import *
from src.portfolio.execution.engine import ExecutionEngine


logger = LogManager.get_logger("trading_system.assets.tradable")


class TradableAsset(Asset):
    """
    Base class for assets that can be traded using the execution engine
    """
    
    def __init__(self, name: str, exchange=None, config=None, params=None):
        """
        Initialize tradable asset
        
        Args:
            name: Asset identifier
            exchange: Exchange interface (optional)
            config: Configuration manager (optional)
            params: Additional parameters (optional)
        """
        super().__init__(name, exchange, config, params)
        
        # Initialize execution engine
        self.execution_mode = params.get('execution_mode', 'live' if exchange else 'simple_backtest')
        self.execution_engine = None
        self._initialize_execution_engine()
        
        # Order tracking
        self.order_event_bus = OrderEventBus()
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: Dict[str, Order] = {}
        self.canceled_orders: Dict[str, Order] = {}
        
        # Set up order event listeners
        self._setup_order_listeners()
    
    def _initialize_execution_engine(self):
        """Initialize the appropriate execution engine based on the mode"""
        if not self.execution_engine and hasattr(self, 'config'):
            self.execution_engine = ExecutionEngine(
                config=self.config,
                mode=self.execution_mode,
                # Can provide historical data if in backtest mode
                historical_data={self.name: self.params.get('historical_data', None)} if 'backtest' in self.execution_mode else None
            )
    
    def _setup_order_listeners(self):
        """Set up event listeners for order status changes"""
        def on_order_fill(order):
            """Handle order fill events"""
            if order.status == OrderStatus.FILLED:
                # Move from active to filled
                if order.order_id in self.active_orders:
                    self.filled_orders[order.order_id] = order
                    del self.active_orders[order.order_id]
                
                # Update asset position based on order
                self._update_position_from_filled_order(order)
                logger.info(f"Order {order.order_id} filled: {order.filled_quantity} @ {order.avg_filled_price}")
            
        def on_order_cancel(order):
            """Handle order cancel events"""
            if order.status == OrderStatus.CANCELED:
                # Move from active to canceled
                if order.order_id in self.active_orders:
                    self.canceled_orders[order.order_id] = order
                    del self.active_orders[order.order_id]
                logger.info(f"Order {order.order_id} canceled")
        
        # Subscribe to order events
        self.order_event_bus.subscribe('FILL', on_order_fill)
        self.order_event_bus.subscribe('CANCEL', on_order_cancel)
    
    def _update_position_from_filled_order(self, order: Order):
        """
        Update asset position based on filled order
        
        Args:
            order: Filled order
        """
        # To be implemented by specific asset classes
        pass
    
    async def create_market_order(self, direction: Direction, quantity: float) -> Order:
        """
        Create a market order
        
        Args:
            direction: Order direction (BUY or SELL)
            quantity: Order quantity
            
        Returns:
            Order: Created market order
        """
        try:
            # Create order object
            order = MarketOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity)
            )
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Execute order if exchange is available
            if self.exchange and self.execution_engine:
                await self._execute_order(order)
            
            return order
        except Exception as e:
            logger.error(f"Error creating market order: {str(e)}")
            raise
    
    async def create_limit_order(self, direction: Direction, quantity: float, price: float) -> Order:
        """
        Create a limit order
        
        Args:
            direction: Order direction (BUY or SELL)
            quantity: Order quantity
            price: Limit price
            
        Returns:
            Order: Created limit order
        """
        try:
            # Create order object
            order = LimitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                price=float(price)
            )
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Execute order if exchange is available
            if self.exchange and self.execution_engine:
                await self._execute_order(order)
            
            return order
        except Exception as e:
            logger.error(f"Error creating limit order: {str(e)}")
            raise
    
    async def create_stop_loss_order(self, direction: Direction, quantity: float, 
                                  stop_price: float) -> Order:
        """
        Create a stop loss order
        
        Args:
            direction: Order direction (BUY or SELL)
            quantity: Order quantity
            stop_price: Stop price
            
        Returns:
            Order: Created stop loss order
        """
        try:
            # Create order object
            order = StopLossOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                stop_price=float(stop_price)
            )
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Execute order if exchange is available
            if self.exchange and self.execution_engine:
                await self._execute_order(order)
            
            return order
        except Exception as e:
            logger.error(f"Error creating stop loss order: {str(e)}")
            raise
    
    async def create_take_profit_order(self, direction: Direction, quantity: float, 
                                    take_profit_price: float) -> Order:
        """
        Create a take profit order
        
        Args:
            direction: Order direction (BUY or SELL)
            quantity: Order quantity
            take_profit_price: Take profit price
            
        Returns:
            Order: Created take profit order
        """
        try:
            # Create order object
            order = TakeProfitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                take_profit_price=float(take_profit_price)
            )
            
            # Add to active orders
            self.active_orders[order.order_id] = order
            
            # Execute order if exchange is available
            if self.exchange and self.execution_engine:
                await self._execute_order(order)
            
            return order
        except Exception as e:
            logger.error(f"Error creating take profit order: {str(e)}")
            raise
    
    async def _execute_order(self, order: Order) -> Dict[str, Any]:
        """
        Execute an order using the execution engine
        
        Args:
            order: Order to execute
            
        Returns:
            Dict[str, Any]: Execution result
        """
        if not self.execution_engine:
            logger.warning(f"No execution engine available for {self.name}, skipping order execution")
            return {"success": False, "error": "No execution engine available"}
        
        try:
            # Convert order to a format acceptable by the execution engine
            import pandas as pd
            signals = pd.DataFrame([{
                'timestamp': order.timestamp if not isinstance(order.timestamp, (int, float)) else order.timestamp,
                'symbol': order.symbol,
                'action': 'buy' if order.direction == Direction.BUY else 'sell',
                'quantity': order.quantity,
                'price': getattr(order, 'price', None)
            }])
            
            # Current prices for live execution
            prices = {self.name: float(self.price)} if hasattr(self, 'price') else None
            
            # Execute the order
            executed_orders, _ = await self.execution_engine.execute(signals, prices)
            
            if not executed_orders.empty:
                # Update order status based on execution result
                for _, executed in executed_orders.iterrows():
                    order_id = executed.get('order_id', None)
                    if order_id == order.order_id:
                        # Update filled quantity and price
                        filled_qty = executed.get('filled_qty', 0)
                        avg_price = executed.get('avg_price', 0)
                        
                        if filled_qty > 0:
                            # Use fill method to update order
                            order.fill(filled_qty, avg_price, float('inf'))  # Assume sufficient liquidity
                    
                logger.info(f"Order {order.order_id} execution result: {executed_orders.to_dict('records')}")
                return executed_orders.to_dict('records')
            
            logger.warning(f"No execution results for order {order.order_id}")
            return {"success": False, "error": "No execution results"}
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if canceled, False otherwise
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found or not active")
            return False
        
        order = self.active_orders[order_id]
        
        try:
            # If connected to exchange, send cancel request
            if self.exchange and hasattr(self.exchange, 'cancel_order'):
                await self.exchange.cancel_order(order_id, self.name)
            
            # Update local order state
            order.cancel()
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def cancel_all_orders(self) -> Dict[str, bool]:
        """
        Cancel all active orders
        
        Returns:
            Dict[str, bool]: Mapping of order IDs to cancel success
        """
        results = {}
        
        # Make a copy of order IDs as we'll be modifying the dictionary during iteration
        active_order_ids = list(self.active_orders.keys())
        
        for order_id in active_order_ids:
            results[order_id] = await self.cancel_order(order_id)
        
        return results
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Order]: Order if found, None otherwise
        """
        return (self.active_orders.get(order_id) or 
                self.filled_orders.get(order_id) or 
                self.canceled_orders.get(order_id))
    
    def get_active_orders(self) -> List[Order]:
        """
        Get all active orders
        
        Returns:
            List[Order]: List of active orders
        """
        return list(self.active_orders.values())
    
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy the asset (implements the Asset interface)
        
        Args:
            amount: Amount to buy
            **kwargs: Additional buy parameters:
                order_type: Type of order ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                stop_price: Stop price (for stop orders)
                take_profit_price: Take profit price (for take profit orders)
                
        Returns:
            Dict[str, Any]: Buy result
        """
        order_type = kwargs.get('order_type', 'market').lower()
        
        try:
            if order_type == 'market':
                order = await self.create_market_order(Direction.BUY, amount)
            elif order_type == 'limit':
                price = kwargs.get('price')
                if price is None:
                    raise ValueError("Limit order requires a price")
                order = await self.create_limit_order(Direction.BUY, amount, price)
            elif order_type == 'stop':
                stop_price = kwargs.get('stop_price')
                if stop_price is None:
                    raise ValueError("Stop order requires a stop_price")
                order = await self.create_stop_loss_order(Direction.BUY, amount, stop_price)
            elif order_type == 'take_profit':
                take_profit_price = kwargs.get('take_profit_price')
                if take_profit_price is None:
                    raise ValueError("Take profit order requires a take_profit_price")
                order = await self.create_take_profit_order(Direction.BUY, amount, take_profit_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "buy",
                "amount": amount,
                "status": order.status.value
            }
        except Exception as e:
            logger.error(f"Error in buy operation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "buy",
                "amount": amount
            }
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell the asset (implements the Asset interface)
        
        Args:
            amount: Amount to sell
            **kwargs: Additional sell parameters:
                order_type: Type of order ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                stop_price: Stop price (for stop orders)
                take_profit_price: Take profit price (for take profit orders)
                
        Returns:
            Dict[str, Any]: Sell result
        """
        order_type = kwargs.get('order_type', 'market').lower()
        
        try:
            if order_type == 'market':
                order = await self.create_market_order(Direction.SELL, amount)
            elif order_type == 'limit':
                price = kwargs.get('price')
                if price is None:
                    raise ValueError("Limit order requires a price")
                order = await self.create_limit_order(Direction.SELL, amount, price)
            elif order_type == 'stop':
                stop_price = kwargs.get('stop_price')
                if stop_price is None:
                    raise ValueError("Stop order requires a stop_price")
                order = await self.create_stop_loss_order(Direction.SELL, amount, stop_price)
            elif order_type == 'take_profit':
                take_profit_price = kwargs.get('take_profit_price')
                if take_profit_price is None:
                    raise ValueError("Take profit order requires a take_profit_price")
                order = await self.create_take_profit_order(Direction.SELL, amount, take_profit_price)
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return {
                "success": True,
                "order_id": order.order_id,
                "order_type": order_type,
                "direction": "sell",
                "amount": amount,
                "status": order.status.value
            }
        except Exception as e:
            logger.error(f"Error in sell operation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "sell",
                "amount": amount
            }
    
    def close(self):
        """Close the asset and clean up resources"""
        # Clean up execution engine
        if self.execution_engine:
            asyncio.create_task(self.execution_engine.close())
            self.execution_engine = None
        
        # Clear order tracking
        self.active_orders.clear()
        self.filled_orders.clear()
        self.canceled_orders.clear()
        logger.info(f"Closed tradable asset {self.name} and cleaned up resources")