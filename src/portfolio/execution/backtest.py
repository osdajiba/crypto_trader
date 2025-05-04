#!/usr/bin/env python3
# src/portfolio/execution/backtest.py

import asyncio
from decimal import Decimal
import pandas as pd
import uuid
import time
import numpy as np
from typing import Dict, Optional, List, Tuple, Any, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.execution.order import Order, OrderStatus, Direction, MarketOrder, LimitOrder


class BacktestExecutionEngine(BaseExecutionEngine):
    """
    Backtest execution engine for simulating trades on historical data.
    
    This class simulates order execution using historical price data, allowing
    for realistic trading simulation with features like slippage, fees, and
    volume-based partial fills. It can modify the historical data to account 
    for market impact of orders.
    """
    
    def __init__(self, config: ConfigManager, mode: str = "backtest", 
                 historical_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the backtest execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Should be "backtest".
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical OHLCV data for backtesting.
        """
        super().__init__(config, mode, historical_data)
        
        # Backtest-specific configuration
        self.volume_participation = config.get("trading", "backtest", "volume_participation", default=0.1)  # Max volume percent per bar
        self.use_market_impact = config.get("trading", "backtest", "use_market_impact", default=True)  # Simulate market impact
        self.market_impact_factor = config.get("trading", "backtest", "market_impact_factor", default=0.1)  # Impact strength
        self.realistic_slippage = config.get("trading", "backtest", "realistic_slippage", default=True)  # More realistic slippage model
        
        # Need historical data for backtesting
        if not historical_data:
            self.logger.warning("No historical data provided for backtest engine")
        
        self.logger.info(f"Backtest execution engine initialized with market impact {'enabled' if self.use_market_impact else 'disabled'}")

    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals in backtest mode using historical data.

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Not used in backtest mode, as we use historical data.

        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: Executed order results and updated historical data.
        """
        if signals.empty:
            self.logger.info("No signals to execute")
            return pd.DataFrame(), self.historical_data
            
        if not self.historical_data:
            self.logger.error("No historical data available for backtest execution")
            return pd.DataFrame(), None
            
        self.logger.info(f"Executing {len(signals)} signals in backtest mode")
        
        # Convert signals to order objects
        orders = await self._create_orders(signals)
        
        # Execute orders against historical data
        executed_orders, updated_data = await self._backtest_execution(orders)
        
        return executed_orders, updated_data

    async def _backtest_execution(self, orders: List[Order]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Execute orders against historical data in backtest mode.
        
        Args:
            orders: List of orders to simulate
            
        Returns:
            Tuple of (executed orders DataFrame, updated historical data)
        """
        executed_orders = []
        updated_data = {k: v.copy() for k, v in self.historical_data.items()}
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Skip if no historical data available
                if symbol not in updated_data:
                    self.logger.warning(f"No historical data for {symbol}, skipping order")
                    executed_orders.append(self._create_rejected_order_result(
                        order, "No historical data available"
                    ))
                    continue
                
                # Get relevant data for execution
                symbol_data = updated_data[symbol]
                
                # Find execution bar based on timestamp
                if order.timestamp is not None:
                    # Find bars on or after the timestamp
                    execution_bars = symbol_data[symbol_data['timestamp'] >= order.timestamp]
                    if execution_bars.empty:
                        self.logger.warning(f"No data after timestamp {order.timestamp} for {symbol}")
                        executed_orders.append(self._create_rejected_order_result(
                            order, "No data available after order timestamp"
                        ))
                        continue
                    execution_bar = execution_bars.iloc[0]
                else:
                    # Use last bar if no timestamp
                    execution_bar = symbol_data.iloc[-1]
                
                # Calculate maximum execution volume based on participation rate
                max_volume = execution_bar['volume'] * self.volume_participation
                filled_qty = min(order.quantity, max_volume)
                
                if filled_qty <= 0:
                    self.logger.info(f"Not enough volume to execute order for {symbol}")
                    executed_orders.append(self._create_rejected_order_result(
                        order, "Insufficient volume"
                    ))
                    continue
                
                # Determine execution price with realistic slippage
                if self.realistic_slippage:
                    exec_price = self._calculate_realistic_execution_price(order, execution_bar, filled_qty)
                else:
                    # Simple slippage model
                    if order.direction == Direction.BUY:
                        # Buy: use high price with positive slippage
                        exec_price = execution_bar['high'] * (1 + self.slippage)
                    else:
                        # Sell: use low price with negative slippage
                        exec_price = execution_bar['low'] * (1 - self.slippage)
                
                # For limit orders, respect limit price
                if hasattr(order, 'price') and order.price is not None:
                    if order.direction == Direction.BUY and exec_price > order.price:
                        self.logger.info(f"Buy limit price {order.price} exceeded by execution price {exec_price}")
                        executed_orders.append(self._create_rejected_order_result(
                            order, "Price condition not met"
                        ))
                        continue
                    if order.direction == Direction.SELL and exec_price < order.price:
                        self.logger.info(f"Sell limit price {order.price} not reached by execution price {exec_price}")
                        executed_orders.append(self._create_rejected_order_result(
                            order, "Price condition not met"
                        ))
                        continue
                
                # Apply market impact if enabled
                if self.use_market_impact:
                    updated_data[symbol] = self._apply_market_impact(
                        symbol_data, 
                        execution_bar, 
                        order.direction, 
                        filled_qty
                    )
                
                # Reduce available volume in the execution bar
                bar_index = execution_bar.name
                updated_data[symbol].loc[bar_index, 'volume'] -= filled_qty
                
                # Update order status
                if hasattr(order, 'fill') and filled_qty > 0:
                    # Fill the order
                    order.fill(filled_qty, exec_price, max_volume)
                    
                # Save order in cache
                self._order_cache[order.order_id] = order
                
                # Calculate commission
                commission_rate = self.commission_taker if order.order_type.value == 'market' else self.commission_maker
                commission = filled_qty * exec_price * commission_rate
                
                # Determine final status
                status = 'filled' if filled_qty == order.quantity else 'partial'
                
                # Record the execution
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'direction': order.direction.value,
                    'filled_qty': filled_qty,
                    'unfilled_qty': order.quantity - filled_qty,
                    'price': exec_price,
                    'avg_price': exec_price,
                    'commission': commission,
                    'status': status,
                    'timestamp': execution_bar['timestamp'],
                    'bar_timestamp': execution_bar['timestamp']
                })
                
                self.logger.debug(f"Backtest execution: {order.direction.value} {filled_qty}/{order.quantity} {symbol} @ {exec_price:.6f} ({status})")
                
            except Exception as e:
                self.logger.error(f"Backtest execution failed for {order.symbol}: {str(e)}")
                # Record the failed order
                executed_orders.append(self._create_failed_order_result(order, str(e)))
        
        return pd.DataFrame(executed_orders), updated_data

    def _calculate_realistic_execution_price(self, order: Order, bar: pd.Series, quantity: float) -> float:
        """
        Calculate a realistic execution price based on order type, direction, and size.
        
        Args:
            order: The order being executed
            bar: OHLCV bar for execution
            quantity: Quantity being executed
            
        Returns:
            Realistic execution price with slippage
        """
        # Extract bar data
        open_price = bar['open']
        high_price = bar['high']
        low_price = bar['low']
        close_price = bar['close']
        volume = bar['volume']
        
        # Calculate volume ratio (order size relative to bar volume)
        volume_ratio = quantity / volume
        
        # Limit the impact of very large orders
        volume_ratio = min(volume_ratio, 0.5)
        
        # Calculate price range for this bar
        price_range = high_price - low_price
        
        # Add randomness to execution (market noise)
        import random
        noise_factor = random.uniform(0.0, 0.3)  # Random noise between 0-30%
        
        # For market orders: use VWAP-like price plus slippage
        if order.order_type.value == 'market':
            # Estimate VWAP (we don't have tick data, so this is approximate)
            vwap = (open_price + high_price + low_price + close_price) / 4
            
            # Apply directional slippage based on order size and direction
            if order.direction == Direction.BUY:
                # Buy orders - execution price is higher than VWAP
                slippage_impact = price_range * volume_ratio * (1 + noise_factor) * self.slippage
                return vwap + slippage_impact
            else:
                # Sell orders - execution price is lower than VWAP
                slippage_impact = price_range * volume_ratio * (1 + noise_factor) * self.slippage
                return vwap - slippage_impact
        
        # For limit orders: use limit price with minimal favorable slippage
        elif order.order_type.value == 'limit' and hasattr(order, 'price'):
            if order.direction == Direction.BUY:
                # Buy limit - execution at or better than limit price
                best_possible = max(low_price, order.price * (1 - self.slippage * noise_factor))
                return min(order.price, best_possible)
            else:
                # Sell limit - execution at or better than limit price
                best_possible = min(high_price, order.price * (1 + self.slippage * noise_factor))
                return max(order.price, best_possible)
        
        # Default fallback - simple slippage model
        if order.direction == Direction.BUY:
            return close_price * (1 + self.slippage)
        else:
            return close_price * (1 - self.slippage)

    def _apply_market_impact(self, data: pd.DataFrame, bar: pd.Series, direction: Direction, quantity: float) -> pd.DataFrame:
        """
        Apply market impact of an order to future bars in the data.
        
        Args:
            data: Full DataFrame of historical data
            bar: The execution bar
            direction: Order direction
            quantity: Executed quantity
            
        Returns:
            Updated DataFrame with market impact applied
        """
        # Copy data to avoid modifying the original
        updated_data = data.copy()
        bar_index = bar.name
        
        # If this is the last bar, no future impact
        if bar_index >= len(updated_data) - 1:
            return updated_data
        
        # Calculate impact factor based on order size relative to bar volume
        volume_ratio = min(quantity / bar['volume'], 0.5)  # Cap at 50% for very large orders
        impact = volume_ratio * self.market_impact_factor
        
        # Direction of impact depends on order direction
        impact_direction = 1 if direction == Direction.BUY else -1
        
        # Apply impact to future bars with decay
        decay_rate = 0.7  # Each subsequent bar has 70% of previous impact
        current_impact = impact
        
        # Apply to next 3-5 bars with decay
        impact_periods = min(5, len(updated_data) - bar_index - 1)
        
        for i in range(1, impact_periods + 1):
            future_idx = bar_index + i
            
            # Skip if out of bounds
            if future_idx >= len(updated_data):
                break
                
            # Apply impact proportionally to open, high, low, close
            impact_amount = current_impact * impact_direction
            
            updated_data.loc[future_idx, 'open'] *= (1 + impact_amount)
            updated_data.loc[future_idx, 'high'] *= (1 + impact_amount)
            updated_data.loc[future_idx, 'low'] *= (1 + impact_amount)
            updated_data.loc[future_idx, 'close'] *= (1 + impact_amount)
            
            # Increase volume slightly to reflect increased activity
            updated_data.loc[future_idx, 'volume'] *= (1 + abs(impact_amount) * 0.5)
            
            # Decay impact for next bar
            current_impact *= decay_rate
        
        return updated_data

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
        Cancel an order in backtest simulation.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        # In backtest mode, we can only cancel orders that we have in our cache
        if order_id in self._order_cache:
            order = self._order_cache[order_id]
            
            # Only allow canceling if not in a final state
            if order.status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                # Update order status
                if hasattr(order, 'set_status'):
                    order.set_status(OrderStatus.CANCELED)
                    
                self.logger.info(f"Canceled order {order_id} in backtest")
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
        Get current status of a backtest order.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            Order status dictionary
        """
        # In backtest, we only know about orders in our cache
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
                'timestamp': order.timestamp
            }
            
        # Order not found
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'error': 'Order not found'
        }

    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Set historical data for backtesting.
        
        Args:
            data: Dictionary of symbol -> DataFrames
        """
        # Ensure all DataFrames have required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        for symbol, df in data.items():
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Historical data for {symbol} missing columns: {missing_columns}")
                continue
                
            # Make sure timestamp is parsed as datetime
            if df['timestamp'].dtype != 'datetime64[ns]':
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    self.logger.warning(f"Could not parse timestamps for {symbol}: {str(e)}")
        
        # Store the validated data
        self.historical_data = data
        self.logger.info(f"Historical data set for {len(data)} symbols")