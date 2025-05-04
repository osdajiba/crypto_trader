#!/usr/bin/env python3
# src/portfolio/execution/base.py

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal
import pandas as pd
import uuid
from typing import Dict, Optional, List, Tuple, Any, Union

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
            config (ConfigManager): Configuration manager instance.
            mode (str): Trading mode name.
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical data for backtesting.
        """
        self.config = config
        self.mode = mode.lower()
        self.logger = LogManager.get_logger(f"execution.{self.mode}")
        self._running = True
        self.historical_data = historical_data or {}
        
        # Load settings from configuration
        self.commission_taker = config.get("trading", "fees", "commission_taker", default=0.001)
        self.commission_maker = config.get("trading", "fees", "commission_maker", default=0.0005)
        self.slippage = config.get("trading", "execution", "slippage", default=0.0001)
        
        # Exchange interface (initialized lazily)
        self._exchange = None
        
        # Order tracking
        self._order_cache = {}
        
        self.logger.info(f"Base execution engine initialized in {self.mode} mode")
    
    async def initialize(self) -> None:
        """
        Initialize execution engine components and connections.
        
        This method establishes necessary connections and resources based on
        the execution mode. Subclasses should override this method to perform
        mode-specific initialization.
        """
        self.logger.info("Initializing base execution engine")
    
    @abstractmethod
    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals and return executed orders and updated historical data.

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Current prices for each symbol.

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
        
        return result

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
                
                # Create appropriate order type
                if price is not None:
                    # Limit order
                    order = LimitOrder(
                        symbol=symbol,
                        direction=direction,
                        quantity=quantity,
                        price=price,
                        timestamp=timestamp,
                        order_id=str(uuid.uuid4())
                    )
                else:
                    # Market order
                    order = MarketOrder(
                        symbol=symbol,
                        direction=direction,
                        quantity=quantity,
                        timestamp=timestamp,
                        order_id=str(uuid.uuid4())
                    )
                
                orders.append(order)
                self.logger.debug(f"Created {order.order_type.value} order for {symbol}, {direction.value} {quantity}")
                
            except Exception as e:
                self.logger.error(f"Failed to create order for signal {signal.to_dict() if hasattr(signal, 'to_dict') else signal}: {str(e)}")
                continue

        self.logger.info(f"Created {len(orders)} orders from {len(signals)} signals")
        return orders
    
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
        self.historical_data.clear()
        self._order_cache.clear()
        
        # Close exchange connection if applicable
        if self._exchange and hasattr(self._exchange, 'close'):
            try:
                await self._exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {str(e)}")
        
        self.logger.info("Execution engine closed")

    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Set historical data for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrames
        """
        self.historical_data = data
        self.logger.debug(f"Historical data set for {len(data)} symbols")
    
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