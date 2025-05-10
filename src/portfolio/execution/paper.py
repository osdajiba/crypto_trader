#!/usr/bin/env python3
# src/portfolio/execution/paper.py

import asyncio
from decimal import Decimal
import pandas as pd
import uuid
import time
from typing import Dict, Optional, List, Tuple, Any, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from exchange.factory import get_exchange_factory
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.execution.order import Order, OrderStatus, Direction, MarketOrder, LimitOrder


class PaperExecutionEngine(BaseExecutionEngine):
    """
    Paper execution engine for simulated trading.
    
    This class simulates order execution using real market data but without 
    placing actual orders on the exchange. It provides a realistic simulation
    of trading with features like slippage, fees, and partial fills.
    """
    
    def __init__(self, config: ConfigManager, mode: str = "paper"):
        """
        Initialize the paper execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Should be "paper".
            historical_data (Optional[Dict[str, pd.DataFrame]]): Not used in paper mode.
        """
        super().__init__(config, mode)
        
        # Paper-specific configuration
        self.simulate_latency = config.get("trading", "paper", "simulate_latency", default=True)
        self.min_latency = config.get("trading", "paper", "min_latency", default=0.1)  # seconds
        self.max_latency = config.get("trading", "paper", "max_latency", default=0.5)  # seconds
        
        # Fill simulation
        self.fill_probability = config.get("trading", "paper", "fill_probability", default=0.95)  # Probability of full fill
        self.partial_fill_range = config.get("trading", "paper", "partial_fill_range", default=[0.7, 1.0])  # % range for partial fills
        self.reject_probability = config.get("trading", "paper", "reject_probability", default=0.01)  # Probability of rejection
        
        # Pending orders storage
        self.pending_orders = {}  # Stores orders not immediately filled
        self.order_fills = {}  # Tracks partial fills
        
        # Periodic order processing
        self._fill_task = None
        
        self.logger.info(f"Paper execution engine initialized with latency simulation {'enabled' if self.simulate_latency else 'disabled'}")

    async def initialize(self) -> None:
        """
        Initialize the paper execution engine and establish exchange connection for price data.
        """
        await super().initialize()
        
        try:
            # Create exchange instance
            if self.exchange is None:
                self.exchange_factory = get_exchange_factory(self.config)
            self.exchange = await self.exchange_factory.create()
            
            # Check connection
            if not self.exchange.is_initialized() or not self.exchange.is_connected():
                raise ConnectionError("Exchange is not properly initialized or connected")
            self.logger.info(f"Paper execution engine connected to exchange {self.exchange.__class__.__name__}")
            
            # Start background task for processing pending orders
            self._fill_task = asyncio.create_task(self._process_pending_orders_loop())
        except Exception as e:
            self.logger.warning(f"Failed to initialize paper execution engine exchange connection: {str(e)}")
            self.logger.info("Paper execution engine will continue without live price data")

    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals in paper trading mode.

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Current market prices (optional, will fetch if needed).

        Returns:
            Tuple[pd.DataFrame, None]: Executed order results and None (no historical data updates).
        """
        if signals.empty:
            self.logger.info("No signals to execute")
            return pd.DataFrame(), None
            
        self.logger.info(f"Executing {len(signals)} signals in paper mode")
        
        # Convert signals to order objects
        orders = await self._create_orders(signals)
        
        # Simulate execution
        executed_orders = await self._paper_execution(orders, prices)
        
        # No historical data updates in paper mode
        return executed_orders, None

    async def _paper_execution(self, orders: List[Order], prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Simulate order execution for paper trading.
        
        Args:
            orders: List of orders to simulate
            prices: Current market prices (optional, will fetch if needed)
            
        Returns:
            DataFrame of executed order results
        """
        # Get current prices if needed
        if not prices:
            prices = await self._fetch_current_prices([order.symbol for order in orders])
        
        executed_orders = []
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Skip if no price available
                if symbol not in prices:
                    self.logger.warning(f"No price available for {symbol}, skipping order")
                    executed_orders.append(self._create_rejected_order_result(
                        order, "No price available"
                    ))
                    continue
                
                current_price = prices[symbol]
                
                # Simulate exchange latency
                if self.simulate_latency:
                    await self._simulate_latency()
                
                # Simulate order rejection
                if self._should_reject_order():
                    executed_orders.append(self._create_rejected_order_result(
                        order, "Order rejected (simulated exchange behavior)"
                    ))
                    continue
                
                # For limit orders, check price conditions
                if hasattr(order, 'price') and order.price is not None:
                    # For buy limit orders, only execute if current price <= limit price
                    if order.direction == Direction.BUY and current_price > order.price:
                        self.logger.info(f"Buy limit order for {symbol} not executed: current price {current_price} > limit {order.price}")
                        
                        # Add to pending orders for possible future execution
                        self.pending_orders[order.order_id] = {
                            'order': order,
                            'added_time': time.time()
                        }
                        
                        executed_orders.append({
                            'order_id': order.order_id,
                            'symbol': symbol,
                            'direction': order.direction.value,
                            'filled_qty': 0,
                            'unfilled_qty': order.quantity,
                            'price': order.price,
                            'avg_price': 0,
                            'status': 'submitted',
                            'timestamp': order.timestamp
                        })
                        continue
                    
                    # For sell limit orders, only execute if current price >= limit price
                    if order.direction == Direction.SELL and current_price < order.price:
                        self.logger.info(f"Sell limit order for {symbol} not executed: current price {current_price} < limit {order.price}")
                        
                        # Add to pending orders for possible future execution
                        self.pending_orders[order.order_id] = {
                            'order': order,
                            'added_time': time.time()
                        }
                        
                        executed_orders.append({
                            'order_id': order.order_id,
                            'symbol': symbol,
                            'direction': order.direction.value,
                            'filled_qty': 0,
                            'unfilled_qty': order.quantity,
                            'price': order.price,
                            'avg_price': 0,
                            'status': 'submitted',
                            'timestamp': order.timestamp
                        })
                        continue
                
                # Apply slippage
                execution_price = self._apply_slippage(current_price, order.direction)
                
                # Decide how much of the order to fill
                filled_qty, status = self._decide_fill_quantity(order.quantity)
                unfilled_qty = order.quantity - filled_qty
                
                # Calculate commission
                commission_rate = self.commission_taker if order.order_type == 'market' else self.commission_maker
                commission = filled_qty * execution_price * commission_rate
                
                # Update order status
                if hasattr(order, 'set_status') and hasattr(order, 'fill') and filled_qty > 0:
                    # Update order with fill information
                    if status == 'filled':
                        order.fill(filled_qty, execution_price, float('inf'))
                    elif status == 'partial':
                        order.fill(filled_qty, execution_price, float('inf'))
                        
                        # Add to order fills for tracking partial fills
                        self.order_fills[order.order_id] = {
                            'order': order,
                            'filled_so_far': filled_qty,
                            'last_fill_time': time.time()
                        }
                
                # Record the simulated execution
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'direction': order.direction.value,
                    'filled_qty': filled_qty,
                    'unfilled_qty': unfilled_qty,
                    'price': execution_price,
                    'avg_price': execution_price,
                    'commission': commission,
                    'status': status,
                    'timestamp': order.timestamp
                })
                
                # Save order in cache
                self._order_cache[order.order_id] = order
                
                # Log execution
                self.logger.info(f"Paper execution: {order.direction.value} {filled_qty}/{order.quantity} {symbol} @ {execution_price:.6f} ({status})")
                
                # If partially filled, add to pending orders
                if status == 'partial':
                    self.pending_orders[order.order_id] = {
                        'order': order,
                        'added_time': time.time()
                    }
                
            except Exception as e:
                self.logger.error(f"Paper execution failed for {order.symbol}: {str(e)}")
                # Record the failed order
                executed_orders.append(self._create_failed_order_result(order, str(e)))
        
        return pd.DataFrame(executed_orders)

    async def _process_pending_orders_loop(self) -> None:
        """
        Background task to process pending orders over time.
        """
        while self._running:
            try:
                if self.pending_orders:
                    self.logger.debug(f"Processing {len(self.pending_orders)} pending orders")
                    
                    # Fetch current prices for all symbols in pending orders
                    symbols = list(set(order_data['order'].symbol for order_data in self.pending_orders.values()))
                    prices = await self._fetch_current_prices(symbols)
                    
                    # Process each pending order
                    orders_to_process = list(self.pending_orders.items())
                    for order_id, order_data in orders_to_process:
                        if order_id not in self.pending_orders:
                            continue  # Order might have been removed by another task
                            
                        order = order_data['order']
                        added_time = order_data['added_time']
                        
                        # Skip if price not available
                        if order.symbol not in prices:
                            continue
                        
                        current_price = prices[order.symbol]
                        
                        # Check if limit order can be executed
                        if hasattr(order, 'price') and order.price is not None:
                            # Buy limit order
                            if order.direction == Direction.BUY and current_price > order.price:
                                continue  # Price still too high
                            
                            # Sell limit order
                            if order.direction == Direction.SELL and current_price < order.price:
                                continue  # Price still too low
                        
                        # Order can be executed - simulate a fill
                        await self._fill_pending_order(order_id, current_price)
                
            except Exception as e:
                self.logger.error(f"Error in pending orders processing: {str(e)}")
            
            # Sleep before next check
            await asyncio.sleep(1.0)  # Check every second
    
    async def _fill_pending_order(self, order_id: str, current_price: float) -> None:
        """
        Fill a pending order at the current price.
        
        Args:
            order_id: ID of the pending order
            current_price: Current market price
        """
        if order_id not in self.pending_orders:
            return
            
        order_data = self.pending_orders[order_id]
        order = order_data['order']
        
        # Remove from pending orders
        del self.pending_orders[order_id]
        
        # Check if it's in the fills dictionary (partially filled before)
        previous_fill = self.order_fills.get(order_id, None)
        already_filled = previous_fill['filled_so_far'] if previous_fill else 0
        remaining_qty = order.quantity - already_filled
        
        # Apply slippage
        execution_price = self._apply_slippage(current_price, order.direction)
        
        # Decide how much to fill now
        fill_qty, status = self._decide_fill_quantity(remaining_qty)
        
        # Update order status
        if hasattr(order, 'fill') and fill_qty > 0:
            order.fill(fill_qty, execution_price, float('inf'))
            
            if status == 'partial':
                # Update tracking for partial fills
                self.order_fills[order_id] = {
                    'order': order,
                    'filled_so_far': already_filled + fill_qty,
                    'last_fill_time': time.time()
                }
                
                # Put back in pending orders
                self.pending_orders[order_id] = {
                    'order': order,
                    'added_time': time.time()
                }
            elif status == 'filled' and order_id in self.order_fills:
                # Remove from fills tracking if fully filled
                del self.order_fills[order_id]
        
        # Log the fill
        total_filled = already_filled + fill_qty
        self.logger.info(f"Filled pending order {order_id}: {fill_qty}/{remaining_qty} {order.symbol} @ {execution_price:.6f} ({total_filled}/{order.quantity} total)")

    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for a list of symbols.
        
        Args:
            symbols: List of symbols to fetch prices for
            
        Returns:
            Dictionary mapping symbols to prices
        """
        prices = {}
        if not self.exchange:
            return prices
            
        for symbol in symbols:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                prices[symbol] = ticker['last']
            except Exception as e:
                self.logger.error(f"Failed to fetch price for {symbol}: {str(e)}")
                
        return prices

    async def _simulate_latency(self) -> None:
        """
        Simulate network latency for order execution.
        """
        if not self.simulate_latency:
            return
            
        import random
        latency = random.uniform(self.min_latency, self.max_latency)
        await asyncio.sleep(latency)

    def _should_reject_order(self) -> bool:
        """
        Determine if an order should be rejected in simulation.
        
        Returns:
            True if order should be rejected, False otherwise
        """
        import random
        return random.random() < self.reject_probability

    def _decide_fill_quantity(self, quantity: float) -> Tuple[float, str]:
        """
        Decide how much of an order to fill based on simulation parameters.
        
        Args:
            quantity: Total order quantity
            
        Returns:
            Tuple of (filled quantity, status)
        """
        import random
        
        # Check for full fill
        if random.random() < self.fill_probability:
            return quantity, 'filled'
            
        # Partial fill
        fill_percent = random.uniform(self.partial_fill_range[0], self.partial_fill_range[1])
        filled_qty = quantity * fill_percent
        
        return filled_qty, 'partial'

    def _apply_slippage(self, price: float, direction: Direction) -> float:
        """
        Apply simulated slippage to a price.
        
        Args:
            price: Base price
            direction: Order direction
            
        Returns:
            Price with slippage applied
        """
        if direction == Direction.BUY:
            # Buy orders get filled at slightly higher prices
            return price * (1 + self.slippage)
        else:
            # Sell orders get filled at slightly lower prices
            return price * (1 - self.slippage)

    def _create_failed_order_result(self, order: Order, error_message: str) -> Dict[str, Any]:
        """
        Create result entry for a failed order.
        
        Args:
            order: The failed order
            error_message: Error message
            
        Returns:
            Order result dictionary with failure information
        """
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'direction': order.direction.value,
            'filled_qty': 0,
            'unfilled_qty': order.quantity,
            'price': getattr(order, 'price', 0),
            'avg_price': 0,
            'status': 'failed',
            'timestamp': order.timestamp,
            'error': error_message
        }

    def _create_rejected_order_result(self, order: Order, reason: str) -> Dict[str, Any]:
        """
        Create result entry for a rejected order.
        
        Args:
            order: The rejected order
            reason: Rejection reason
            
        Returns:
            Order result dictionary with rejection information
        """
        # Update order status
        if hasattr(order, 'set_status'):
            order.set_status(OrderStatus.REJECTED)
            
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'direction': order.direction.value,
            'filled_qty': 0,
            'unfilled_qty': order.quantity,
            'price': getattr(order, 'price', 0),
            'avg_price': 0,
            'status': 'rejected',
            'timestamp': order.timestamp,
            'reason': reason
        }

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel a paper trading order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        # Check if it's a pending order
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]['order']
            del self.pending_orders[order_id]
            
            # Update order status
            if hasattr(order, 'set_status'):
                order.set_status(OrderStatus.CANCELED)
                
            # Remove from fills tracking if present
            if order_id in self.order_fills:
                del self.order_fills[order_id]
                
            self.logger.info(f"Canceled pending order {order_id}")
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'status': 'canceled'
            }
            
        # Check if it's in order cache
        if order_id in self._order_cache:
            order = self._order_cache[order_id]
            
            # Only allow canceling if not in a final state
            if order.status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                # Update order status
                if hasattr(order, 'set_status'):
                    order.set_status(OrderStatus.CANCELED)
                    
                self.logger.info(f"Canceled order {order_id}")
                return {
                    'success': True,
                    'order_id': order_id,
                    'symbol': symbol,
                    'status': 'canceled'
                }
            else:
                return {
                    'success': False,
                    'order_id': order_id,
                    'symbol': symbol,
                    'error': f"Cannot cancel order in {order.status.value} state"
                }
                
        # Order not found
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'error': 'Order not found'
        }

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get current status of a paper trading order.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            Order status dictionary
        """
        # First try from cache using parent method
        cache_result = await super().get_order_status(order_id, symbol)
        if 'error' not in cache_result:
            return cache_result
            
        # Check pending orders
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]['order']
            return {
                'success': True,
                'order_id': order_id,
                'symbol': order.symbol,
                'status': order.status.value,
                'filled_qty': order.filled_quantity,
                'unfilled_qty': order.quantity - order.filled_quantity,
                'avg_price': order.avg_filled_price,
                'direction': order.direction.value,
                'timestamp': order.timestamp
            }
            
        # Order not found
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'error': 'Order not found'
        }

    async def close(self) -> None:
        """
        Close the paper execution engine and clean up resources.
        """
        # Cancel the pending orders processing task
        if self._fill_task and not self._fill_task.done():
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
                
        # Clear pending orders and fills
        self.pending_orders.clear()
        self.order_fills.clear()
        
        # Call parent close method
        await super().close()