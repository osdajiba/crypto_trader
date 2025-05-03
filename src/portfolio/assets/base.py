#!/usr/bin/env python3
# src/portfolio/assets/base.py

from abc import ABC
from decimal import Decimal
from typing import Dict, Any, Optional, List, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.portfolio.execution.order import (
    Order, OrderType, Direction, OrderStatus, Validity,
    MarketOrder, LimitOrder, StopLossOrder, TakeProfitOrder, 
    OrderEventBus
)

logger = LogManager.get_logger("portfolio.asset")
async_executor = AsyncExecutor()

class Asset(ABC):
    """Base class for all portfolio assets with trading capabilities"""
    
    def __init__(self, name: str, exchange=None, config: Optional[ConfigManager] = None, 
                params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.exchange = exchange
        self.config = config if config else ConfigManager()
        self.params = params or {}
        
        # Asset state
        self._value = Decimal('0')
        self._position_size = Decimal('0')
        self._last_update_time = 0
        
        # Trading capabilities (optional)
        self.is_tradable = params.get('tradable', False)
        if self.is_tradable:
            self._setup_trading()
        
    def _setup_trading(self):
        """Initialize trading capabilities if asset is tradable"""
        # Execution engine
        from portfolio.execution.base import ExecutionEngine
        self.execution_mode = self.params.get('execution_mode', 'live' if self.exchange else 'simple_backtest')
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode=self.execution_mode,
            historical_data={self.name: self.params.get('historical_data')} if 'backtest' in self.execution_mode else None
        )
        
        # Order management
        self.order_event_bus = OrderEventBus()
        self.active_orders = {}
        self.filled_orders = {}
        self.canceled_orders = {}
        
        # Set up order event listeners
        self.order_event_bus.subscribe('FILL', self._on_order_fill)
        self.order_event_bus.subscribe('CANCEL', self._on_order_cancel)
    
    def _on_order_fill(self, order):
        """Handle order fill events"""
        if order.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            if order.status == OrderStatus.FILLED and order.order_id in self.active_orders:
                self.filled_orders[order.order_id] = order
                del self.active_orders[order.order_id]
            
            self._update_position_from_filled_order(order)
            logger.info(f"{self.name} order {order.order_id} filled: {order.filled_quantity} @ {order.avg_filled_price}")
    
    def _on_order_cancel(self, order):
        """Handle order cancel events"""
        if order.status == OrderStatus.CANCELED and order.order_id in self.active_orders:
            self.canceled_orders[order.order_id] = order
            del self.active_orders[order.order_id]
            logger.info(f"{self.name} order {order.order_id} canceled")

    def _update_position_from_filled_order(self, order):
        """Update asset position based on filled order - to be implemented by subclasses"""
        pass
    
    def get_value(self) -> float:
        """Get current asset value"""
        return float(self._value)
    
    def get_position_size(self) -> float:
        """Get current position size"""
        return float(self._position_size)
    
    async def update_value(self) -> float:
        """Update and return current asset value - to be implemented by subclasses"""
        return float(self._value)
    
    async def create_order(self, order_type: str, direction: Direction, 
                          quantity: float, **kwargs) -> Order:
        """Create an order of the specified type"""
        if not self.is_tradable:
            raise ValueError(f"Asset {self.name} is not tradable")
        
        order = None
        
        if order_type == 'market':
            order = MarketOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity)
            )
        elif order_type == 'limit':
            price = kwargs.get('price')
            if price is None:
                raise ValueError("Limit order requires a price")
            order = LimitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                price=float(price)
            )
        elif order_type == 'stop':
            stop_price = kwargs.get('stop_price')
            if stop_price is None:
                raise ValueError("Stop order requires a stop_price")
            order = StopLossOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                stop_price=float(stop_price)
            )
        elif order_type == 'take_profit':
            take_profit_price = kwargs.get('take_profit_price')
            if take_profit_price is None:
                raise ValueError("Take profit order requires a take_profit_price")
            order = TakeProfitOrder(
                symbol=self.name,
                direction=direction,
                quantity=float(quantity),
                take_profit_price=float(take_profit_price)
            )
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Register with event bus
        order.event_bus = self.order_event_bus
        
        # Add to active orders
        self.active_orders[order.order_id] = order
        
        # Execute if exchange available
        if self.exchange and self.execution_engine:
            await self._execute_order(order)
        
        return order
    
    async def _execute_order(self, order: Order) -> Dict[str, Any]:
        """Execute an order using the execution engine"""
        if not self.execution_engine:
            logger.warning(f"No execution engine for {self.name}, skipping execution")
            return {"success": False, "error": "No execution engine"}
        
        try:
            # Convert order to signal format
            import pandas as pd
            signals = pd.DataFrame([{
                'timestamp': order.timestamp,
                'symbol': order.symbol,
                'action': 'buy' if order.direction == Direction.BUY else 'sell',
                'quantity': order.quantity,
                'price': getattr(order, 'price', None)
            }])
            
            # Current prices for live execution
            prices = {self.name: float(self.price)} if hasattr(self, 'price') else None
            
            # Execute order
            executed_orders, _ = await self.execution_engine.execute(signals, prices)
            
            if not executed_orders.empty:
                # Update order status
                for _, executed in executed_orders.iterrows():
                    order_id = executed.get('order_id', None)
                    if order_id == order.order_id:
                        filled_qty = executed.get('filled_qty', 0)
                        avg_price = executed.get('avg_price', 0)
                        
                        if filled_qty > 0:
                            order.fill(filled_qty, avg_price, float('inf'))
                
                return executed_orders.to_dict('records')
            
            return {"success": False, "error": "No execution results"}
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        if not self.is_tradable or order_id not in self.active_orders:
            return False
        
        try:
            order = self.active_orders[order_id]
            
            # Cancel on exchange if connected
            if self.exchange and hasattr(self.exchange, 'cancel_order'):
                await self.exchange.cancel_order(order_id, self.name)
            
            # Update local state
            order.cancel()
            return True
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {str(e)}")
            return False
    
    async def cancel_all_orders(self) -> Dict[str, bool]:
        """Cancel all active orders"""
        if not self.is_tradable:
            return {}
        
        results = {}
        for order_id in list(self.active_orders.keys()):
            results[order_id] = await self.cancel_order(order_id)
        return results
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        if not self.is_tradable:
            return None
        
        return (self.active_orders.get(order_id) or 
                self.filled_orders.get(order_id) or 
                self.canceled_orders.get(order_id))
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        if not self.is_tradable:
            return []
        
        return list(self.active_orders.values())
    
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """Buy this asset"""
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
            logger.error(f"Buy error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "buy",
                "amount": amount
            }
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """Sell this asset"""
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
            logger.error(f"Sell error for {self.name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "direction": "sell",
                "amount": amount
            }
    
    def set_exchange(self, exchange) -> None:
        """Set/update the exchange interface"""
        self.exchange = exchange
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert asset to dictionary representation"""
        data = {
            'name': self.name,
            'type': self.__class__.__name__,
            'value': float(self._value),
            'position_size': float(self._position_size),
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
        """Close the asset and clean up resources"""
        if self.is_tradable:
            await self.cancel_all_orders()
            if self.execution_engine:
                await self.execution_engine.close()
        
        logger.info(f"Closed asset {self.name}")