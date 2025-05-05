#!/usr/bin/env python3
# src/portfolio/execution/base.py

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
import pandas as pd
import uuid
import time
from typing import Dict, Optional, List, Tuple, Any, Union, Set, Callable

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange
from src.portfolio.execution.order import Order, OrderStatus, Direction, MarketOrder, LimitOrder, OrderType


class BaseExecutionEngine(ABC):
    """
    Abstract base class for execution engines.
    
    This class defines the common interface and shared functionality for all
    execution engines, providing a consistent way to execute orders across
    different trading modes (live, paper, backtest).
    """
    
    def __init__(self, config: ConfigManager, mode: str = "backtest", 
                 historical_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the base execution engine.

        Args:
            config: Configuration manager instance.
            mode: Trading mode name.
            historical_data: Historical data for backtesting.
        """
        self.config = config
        self.mode = mode.lower()
        self.logger = LogManager.get_logger(f"execution.{self.mode}")
        self._running = True
        self._initialized = False
        self.historical_data = historical_data or {}
        
        # Load settings from configuration
        self.commission_taker = config.get("trading", "fees", "commission_taker", default=0.001)
        self.commission_maker = config.get("trading", "fees", "commission_maker", default=0.0005)
        self.slippage = config.get("trading", "execution", "slippage", default=0.0001)
        
        # Exchange interface (initialized lazily)
        self._exchange = None
        
        # Order tracking
        self._order_cache = {}
        self._order_execution_time = {}
        
        # Event handlers
        self._event_handlers = {}
        
        # Execution stats
        self._stats = {
            'executed_orders': 0,
            'failed_orders': 0,
            'canceled_orders': 0,
            'total_commission': 0.0,
            'avg_execution_time': 0.0
        }
        
        # Custom parameters for subclasses
        self._params = {}
    
    async def initialize(self) -> None:
        """
        Initialize execution engine components and connections.
        
        This method establishes necessary connections and resources based on
        the execution mode. Subclasses should override this method to perform
        mode-specific initialization.
        """
        if self._initialized:
            return
            
        try:
            # Initialize subclass-specific components
            await self._initialize_specific()
            
            self._initialized = True
            self.logger.info(f"Execution engine initialized in {self.mode} mode")
        except Exception as e:
            self.logger.error(f"Error initializing execution engine: {e}")
            raise
    
    async def _initialize_specific(self) -> None:
        """
        Initialize components specific to this execution engine.
        Subclasses should implement this method.
        """
        pass
    
    def set_param(self, key: str, value: Any) -> None:
        """
        Set a custom parameter for the execution engine
        
        Args:
            key: Parameter key
            value: Parameter value
        """
        self._params[key] = value
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """
        Get a custom parameter from the execution engine
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        return self._params.get(key, default)
    
    @abstractmethod
    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals and return executed orders and updated historical data.

        Args:
            signals: Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices: Current prices for each symbol.

        Returns:
            Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]: Executed orders and updated historical data.
        """
        pass
    
    async def execute_order(self, order: Order) -> Dict[str, Any]:
        """
        Execute a single order directly.
        
        This method converts the order to a signal and uses the execute method.
        
        Args:
            order: The order object to execute
            
        Returns:
            Dict containing execution results
        """
        # Record execution start time
        start_time = time.time()
        
        try:
            # Convert order to signal format for uniform processing
            signal_data = {
                'timestamp': order.timestamp,
                'symbol': order.symbol,
                'action': order.direction.value,
                'quantity': order.quantity
            }
            
            # Add price for limit orders
            if hasattr(order, 'price') and order.price is not None:
                signal_data['price'] = order.price
                
            signals = pd.DataFrame([signal_data])
            
            # Execute using the standard method
            executed_df, _ = await self.execute(signals)
            
            # If no execution results, create an error response
            if executed_df.empty:
                self._stats['failed_orders'] += 1
                return {
                    'success': False,
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'error': 'Execution failed'
                }
                
            # Cache the order for future reference
            self._order_cache[order.order_id] = order
                
            # Extract execution results for the order
            result = executed_df.iloc[0].to_dict()
            result['success'] = result.get('status', 'failed') != 'failed'
            
            # Update statistics
            if result['success']:
                self._stats['executed_orders'] += 1
                if 'commission' in result:
                    self._stats['total_commission'] += result['commission']
            else:
                self._stats['failed_orders'] += 1
                
            # Fire event
            self._fire_event('order_executed', {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'success': result['success'],
                'result': result
            })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {e}")
            
            # Update statistics
            self._stats['failed_orders'] += 1
            
            # Fire event
            self._fire_event('order_error', {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'error': str(e)
            })
            
            return {
                'success': False,
                'order_id': order.order_id,
                'symbol': order.symbol,
                'error': str(e)
            }
        finally:
            # Record execution time
            execution_time = time.time() - start_time
            self._order_execution_time[order.order_id] = execution_time
            
            # Update average execution time
            if self._stats['executed_orders'] > 0:
                total_execution_time = sum(self._order_execution_time.values())
                self._stats['avg_execution_time'] = total_execution_time / len(self._order_execution_time)

    async def _create_orders(self, signals: pd.DataFrame) -> List[Order]:
        """
        Create Order objects from signals.
        
        Args:
            signals: DataFrame containing trading signals
            
        Returns:
            List of Order objects
        """
        orders = []
        
        for _, signal in signals.iterrows():
            try:
                # Extract signal data
                symbol = signal['symbol']
                action = signal['action'].lower() if 'action' in signal else 'buy'
                timestamp = signal.get('timestamp', None)
                quantity = float(signal['quantity']) if 'quantity' in signal else 0.0
                price = float(signal['price']) if 'price' in signal else None
                
                # Normalize timestamp if needed
                if timestamp is not None and not isinstance(timestamp, (int, float)):
                    if hasattr(timestamp, 'timestamp'):
                        timestamp = timestamp.timestamp()
                
                # Skip invalid signals
                if quantity <= 0:
                    self.logger.warning(f"Skipping signal with invalid quantity: {quantity}")
                    continue
                
                # Determine order direction
                if action in ('buy', 'long'):
                    direction = Direction.BUY
                elif action in ('sell', 'short'):
                    direction = Direction.SELL
                else:
                    self.logger.warning(f"Skipping signal with unknown action: {action}")
                    continue
                
                # Generate an order ID if not provided
                order_id = signal.get('order_id', str(uuid.uuid4()))
                
                # Create appropriate order type
                if price is not None:
                    # Limit order
                    order = LimitOrder(
                        symbol=symbol,
                        direction=direction,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        order_id=order_id
                    )
                else:
                    # Market order
                    order = MarketOrder(
                        symbol=symbol,
                        direction=direction,
                        quantity=quantity,
                        timestamp=timestamp,
                        order_id=order_id
                    )
                
                orders.append(order)
                self.logger.debug(f"Created {order.order_type.value} order for {symbol}, {direction.value} {quantity}")
                
            except Exception as e:
                self.logger.error(f"Failed to create order for signal: {e}")
                continue

        self.logger.info(f"Created {len(orders)} orders from {len(signals)} signals")
        return orders
    
    def add_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add an event handler
        
        Args:
            event_type: Event type to handle
            handler: Event handler function
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = set()
            
        self._event_handlers[event_type].add(handler)
        self.logger.debug(f"Added event handler for {event_type}")
    
    def remove_event_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove an event handler
        
        Args:
            event_type: Event type
            handler: Event handler function
        """
        if event_type in self._event_handlers:
            self._event_handlers[event_type].discard(handler)
            self.logger.debug(f"Removed event handler for {event_type}")
    
    def _fire_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Fire an event to all registered handlers
        
        Args:
            event_type: Event type
            event_data: Event data
        """
        if event_type not in self._event_handlers:
            return
            
        for handler in self._event_handlers[event_type]:
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get current status of an order, either from cache or exchange.
        
        Args:
            order_id: The order ID to check
            symbol: Trading symbol
            
        Returns:
            Dict containing order status information
        """
        # Check if order is in local cache
        if order_id in self._order_cache:
            order = self._order_cache[order_id]
            return {
                'success': True,
                'order_id': order_id,
                'symbol': order.symbol,
                'status': order.status.value,
                'filled_qty': order.filled_quantity,
                'unfilled_qty': order.quantity - order.filled_quantity,
                'avg_price': order.avg_filled_price,
                'direction': order.direction.value,
                'timestamp': order.timestamp,
                'from_cache': True
            }
        
        # Subclasses should override to implement exchange-specific status checking
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'status': 'unknown',
            'error': 'Order not found in cache, and subclass did not implement exchange status checking'
        }

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an open order.
        
        Args:
            order_id: The order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Dict containing cancellation result
        """
        pass

    async def close(self) -> None:
        """
        Close the execution engine and clean up resources.
        """
        if not self._running:
            return
        
        self._running = False
        
        try:
            # Perform subclass-specific cleanup
            await self._close_specific()
            
            # Clear data
            self.historical_data.clear()
            self._order_cache.clear()
            self._order_execution_time.clear()
            self._event_handlers.clear()
            
            # Close exchange connection if applicable
            if self._exchange and hasattr(self._exchange, 'close'):
                try:
                    await self._exchange.close()
                except Exception as e:
                    self.logger.error(f"Error closing exchange connection: {e}")
            
            self.logger.info("Execution engine closed")
        except Exception as e:
            self.logger.error(f"Error closing execution engine: {e}")

    async def _close_specific(self) -> None:
        """
        Close resources specific to this execution engine.
        Subclasses should implement this method.
        """
        pass

    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Set historical data for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrames
        """
        self.historical_data = data
        self.logger.debug(f"Set historical data for {len(data)} symbols")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics
        
        Returns:
            Dict containing execution statistics
        """
        return self._stats
    
    @staticmethod
    def _map_exchange_status(status: str) -> str:
        """
        Map exchange order status to internal status.
        
        Args:
            status: Exchange order status
            
        Returns:
            Internal order status
        """
        status_map = {
            'new': 'submitted',
            'open': 'submitted',
            'closed': 'filled',
            'filled': 'filled',
            'partially_filled': 'partial',
            'partial': 'partial',
            'canceled': 'canceled',
            'cancelled': 'canceled',
            'expired': 'canceled',
            'rejected': 'failed'
        }
        
        return status_map.get(status.lower(), 'unknown')