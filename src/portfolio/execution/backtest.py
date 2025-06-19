#!/usr/bin/env python3
# src/portfolio/execution/backtest.py (改进版)

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
    
    def __init__(self, config: ConfigManager, mode: str = "backtest"):
        """
        Initialize the backtest execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Should be "backtest".
        """
        super().__init__(config, mode)
        
        # 回测专用配置
        self.volume_participation = config.get("trading", "backtest", "volume_participation", default=0.1)
        self.use_market_impact = config.get("trading", "backtest", "use_market_impact", default=True)
        self.market_impact_factor = config.get("trading", "backtest", "market_impact_factor", default=0.1)
        self.realistic_slippage = config.get("trading", "backtest", "realistic_slippage", default=True)
        
        # 滑点和手续费设置
        self.slippage_buy = config.get("trading", "execution", "slippage_buy", default=0.0001)
        self.slippage_sell = config.get("trading", "execution", "slippage_sell", default=0.0001)
        
        # 历史数据和状态跟踪
        self.historical_data = None
        self.current_timestamp = None
        self.asset_states = {}  # symbol -> {quantity, avg_price, total_cost}
        
        # 成交记录
        self.execution_log = []
        
        self.logger.info(f"Backtest execution engine initialized with volume_participation={self.volume_participation}")

    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals with volume-based matching
        
        Args:
            signals: DataFrame containing trading signals
            prices: Optional current prices (not used in backtest)
            
        Returns:
            Tuple of (executed orders DataFrame, updated historical data)
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
        
        # Execute orders with volume matching
        executed_orders = await self._execute_orders_with_matching(orders)
        
        return executed_orders, self.historical_data

    async def _execute_orders_with_matching(self, orders: List[Order]) -> pd.DataFrame:
        """
        Execute orders with realistic volume matching and asset state updates
        
        Args:
            orders: List of orders to execute
            
        Returns:
            DataFrame of execution results
        """
        executed_results = []
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Get market data for this symbol
                if symbol not in self.historical_data:
                    result = self._create_rejected_order_result(order, "No market data available")
                    executed_results.append(result)
                    continue
                
                # Find the appropriate bar for execution
                execution_bar = self._find_execution_bar(order, self.historical_data[symbol])
                if execution_bar is None:
                    result = self._create_rejected_order_result(order, "No valid execution bar found")
                    executed_results.append(result)
                    continue
                
                # Execute order with volume matching
                execution_result = await self._execute_single_order(order, execution_bar)
                executed_results.append(execution_result)
                
                # Update asset state if execution was successful
                if execution_result.get('status') in ['filled', 'partial']:
                    await self._update_asset_state(order, execution_result)
                
            except Exception as e:
                self.logger.error(f"Error executing order {order.order_id}: {str(e)}")
                result = self._create_failed_order_result(order, str(e))
                executed_results.append(result)
        
        return pd.DataFrame(executed_results)

    async def _execute_single_order(self, order: Order, execution_bar: pd.Series) -> Dict[str, Any]:
        """
        Execute a single order against a market bar
        
        Args:
            order: Order to execute
            execution_bar: Market data bar for execution
            
        Returns:
            Execution result dictionary
        """
        try:
            # Calculate available liquidity
            available_volume = execution_bar['volume'] * self.volume_participation
            
            # Determine maximum fill quantity
            max_fill_qty = min(order.quantity, available_volume)
            
            if max_fill_qty <= 0:
                return self._create_rejected_order_result(order, "Insufficient market liquidity")
            
            # Calculate execution price with slippage
            execution_price = self._calculate_execution_price(order, execution_bar)
            
            # Validate limit order price constraints
            if not self._validate_limit_price(order, execution_price):
                return self._create_rejected_order_result(order, "Limit price not met")
            
            # Calculate fees
            commission = max_fill_qty * execution_price * self.commission_taker
            
            # Fill the order
            if hasattr(order, 'fill'):
                order.fill(max_fill_qty, execution_price, available_volume)
            
            # Cache order
            self._order_cache[order.order_id] = order
            
            # Record execution
            self.execution_log.append({
                'timestamp': execution_bar.get('timestamp', execution_bar.name),
                'order_id': order.order_id,
                'symbol': order.symbol,
                'direction': order.direction.value,
                'quantity': max_fill_qty,
                'price': execution_price,
                'commission': commission,
                'bar_volume': execution_bar['volume'],
                'used_volume': max_fill_qty
            })
            
            # Determine final status
            status = 'filled' if max_fill_qty >= order.quantity else 'partial'
            
            return {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'direction': order.direction.value,
                'filled_qty': max_fill_qty,
                'unfilled_qty': order.quantity - max_fill_qty,
                'price': execution_price,
                'avg_price': execution_price,
                'commission': commission,
                'status': status,
                'timestamp': execution_bar.get('timestamp', execution_bar.name),
                'execution_bar_volume': execution_bar['volume'],
                'liquidity_used': max_fill_qty / available_volume if available_volume > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error executing order {order.order_id}: {str(e)}")
            return self._create_failed_order_result(order, str(e))

    def _calculate_execution_price(self, order: Order, bar: pd.Series) -> float:
        """
        Calculate realistic execution price with slippage
        
        Args:
            order: Order being executed
            bar: Market data bar
            
        Returns:
            Execution price
        """
        base_price = bar['close']  # Use close price as base
        
        if order.order_type.value == 'market':
            # Market orders: apply slippage based on direction
            if order.direction == Direction.BUY:
                # Buy at higher price (adverse slippage)
                slippage_factor = 1 + self.slippage_buy
                execution_price = min(base_price * slippage_factor, bar['high'])
            else:
                # Sell at lower price (adverse slippage)
                slippage_factor = 1 - self.slippage_sell
                execution_price = max(base_price * slippage_factor, bar['low'])
        
        elif order.order_type.value == 'limit' and hasattr(order, 'price'):
            # Limit orders: use limit price but apply minimal slippage
            if order.direction == Direction.BUY:
                # Can get filled at or below limit price
                best_price = max(bar['low'], order.price * (1 - self.slippage_buy * 0.1))
                execution_price = min(order.price, best_price)
            else:
                # Can get filled at or above limit price
                best_price = min(bar['high'], order.price * (1 + self.slippage_sell * 0.1))
                execution_price = max(order.price, best_price)
        
        else:
            # Fallback to close price
            execution_price = base_price
        
        return execution_price

    def _validate_limit_price(self, order: Order, execution_price: float) -> bool:
        """
        Validate that limit order price constraints are met
        
        Args:
            order: Order to validate
            execution_price: Proposed execution price
            
        Returns:
            True if price is valid, False otherwise
        """
        if order.order_type.value != 'limit' or not hasattr(order, 'price'):
            return True
        
        if order.direction == Direction.BUY:
            return execution_price <= order.price
        else:
            return execution_price >= order.price

    def _find_execution_bar(self, order: Order, symbol_data: pd.DataFrame) -> Optional[pd.Series]:
        """
        Find the appropriate market data bar for order execution
        
        Args:
            order: Order to execute
            symbol_data: Historical data for the symbol
            
        Returns:
            Market data bar or None if not found
        """
        try:
            if order.timestamp is not None:
                # Find bar at or after order timestamp
                if 'timestamp' in symbol_data.columns:
                    valid_bars = symbol_data[symbol_data['timestamp'] >= order.timestamp]
                    if not valid_bars.empty:
                        return valid_bars.iloc[0]
                elif isinstance(symbol_data.index, pd.DatetimeIndex):
                    try:
                        timestamp = pd.to_datetime(order.timestamp)
                        valid_indices = symbol_data.index[symbol_data.index >= timestamp]
                        if len(valid_indices) > 0:
                            return symbol_data.loc[valid_indices[0]]
                    except Exception:
                        pass
            
            # Fallback to last available bar
            return symbol_data.iloc[-1]
            
        except Exception as e:
            self.logger.error(f"Error finding execution bar: {str(e)}")
            return None

    async def _update_asset_state(self, order: Order, execution_result: Dict[str, Any]) -> None:
        """
        Update asset state after successful execution
        
        Args:
            order: Executed order
            execution_result: Execution result details
        """
        try:
            symbol = order.symbol
            filled_qty = execution_result['filled_qty']
            avg_price = execution_result['avg_price']
            commission = execution_result['commission']
            
            # Initialize asset state if not exists
            if symbol not in self.asset_states:
                self.asset_states[symbol] = {
                    'quantity': 0.0,
                    'avg_price': 0.0,
                    'total_cost': 0.0,
                    'total_commission': 0.0
                }
            
            asset_state = self.asset_states[symbol]
            
            if order.direction == Direction.BUY:
                # Update for buy order
                old_quantity = asset_state['quantity']
                old_total_cost = asset_state['total_cost']
                
                new_cost = filled_qty * avg_price + commission
                new_quantity = old_quantity + filled_qty
                new_total_cost = old_total_cost + new_cost
                
                asset_state['quantity'] = new_quantity
                asset_state['total_cost'] = new_total_cost
                asset_state['total_commission'] += commission
                
                if new_quantity > 0:
                    asset_state['avg_price'] = new_total_cost / new_quantity
                
            else:  # SELL
                # Update for sell order
                old_quantity = asset_state['quantity']
                
                if old_quantity >= filled_qty:
                    # Sufficient quantity to sell
                    asset_state['quantity'] -= filled_qty
                    asset_state['total_commission'] += commission
                    
                    # Proportionally reduce total cost
                    if old_quantity > 0:
                        cost_reduction = (filled_qty / old_quantity) * asset_state['total_cost']
                        asset_state['total_cost'] -= cost_reduction
                    
                    self.logger.debug(f"Sold {filled_qty} of {symbol}, remaining: {asset_state['quantity']}")
                else:
                    self.logger.warning(f"Oversold {symbol}: tried to sell {filled_qty}, had {old_quantity}")
            
            # Notify asset if it exists in portfolio manager
            await self._notify_asset_update(symbol, asset_state, execution_result)
            
        except Exception as e:
            self.logger.error(f"Error updating asset state for {order.symbol}: {str(e)}")

    async def _notify_asset_update(self, symbol: str, asset_state: Dict[str, Any], 
                                 execution_result: Dict[str, Any]) -> None:
        """
        Notify the corresponding asset about the state change
        
        Args:
            symbol: Asset symbol
            asset_state: Updated asset state
            execution_result: Execution details
        """
        try:
            # This would typically be injected or accessed through a registry
            # For now, we'll implement a notification mechanism
            
            # Create a trade record for the asset to process
            trade_record = {
                'symbol': symbol,
                'direction': execution_result['direction'],
                'quantity': execution_result['filled_qty'],
                'price': execution_result['avg_price'],
                'commission': execution_result['commission'],
                'timestamp': execution_result['timestamp'],
                'order_id': execution_result['order_id']
            }
            
            # Store for later retrieval by portfolio manager
            if not hasattr(self, 'trade_updates'):
                self.trade_updates = []
            
            self.trade_updates.append(trade_record)
            
            self.logger.debug(f"Recorded trade update for {symbol}: {trade_record}")
            
        except Exception as e:
            self.logger.error(f"Error notifying asset update for {symbol}: {str(e)}")

    def get_asset_state(self, symbol: str) -> Dict[str, Any]:
        """
        Get current state of an asset
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Asset state dictionary
        """
        return self.asset_states.get(symbol, {
            'quantity': 0.0,
            'avg_price': 0.0,
            'total_cost': 0.0,
            'total_commission': 0.0
        })

    def get_all_asset_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get states of all tracked assets
        
        Returns:
            Dictionary of symbol -> asset state
        """
        return self.asset_states.copy()

    def get_execution_log(self) -> List[Dict[str, Any]]:
        """
        Get complete execution log
        
        Returns:
            List of execution records
        """
        return self.execution_log.copy()

    def get_trade_updates(self) -> List[Dict[str, Any]]:
        """
        Get and clear pending trade updates for portfolio manager
        
        Returns:
            List of trade update records
        """
        if hasattr(self, 'trade_updates'):
            updates = self.trade_updates.copy()
            self.trade_updates.clear()
            return updates
        return []

    def set_historical_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Set historical data for backtesting with validation
        
        Args:
            data: Dictionary of symbol -> DataFrames
        """
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        validated_data = {}
        for symbol, df in data.items():
            # Check required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                self.logger.warning(f"Historical data for {symbol} missing columns: {missing_columns}")
                continue
                
            # Ensure timestamp is datetime
            if df['timestamp'].dtype != 'datetime64[ns]':
                try:
                    df = df.copy()
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception as e:
                    self.logger.warning(f"Could not parse timestamps for {symbol}: {str(e)}")
                    continue
            
            # Validate data integrity
            if df['volume'].sum() <= 0:
                self.logger.warning(f"No volume data for {symbol}")
                continue
                
            validated_data[symbol] = df.sort_values('timestamp')
        
        self.historical_data = validated_data
        self.logger.info(f"Historical data set for {len(validated_data)} symbols")

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order in backtest (limited functionality)"""
        if order_id in self._order_cache:
            order = self._order_cache[order_id]
            
            if order.status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                order.status = OrderStatus.CANCELED
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
                
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'error': 'Order not found'
        }

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status in backtest"""
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
            
        return {
            'success': False,
            'order_id': order_id,
            'symbol': symbol,
            'error': 'Order not found'
        }

    def _create_failed_order_result(self, order: Order, error_message: str) -> Dict[str, Any]:
        """Create result entry for a failed order"""
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'direction': order.direction.value,
            'filled_qty': 0,
            'unfilled_qty': order.quantity,
            'price': getattr(order, 'price', 0),
            'avg_price': 0,
            'commission': 0,
            'status': 'failed',
            'timestamp': order.timestamp,
            'error': error_message
        }

    def _create_rejected_order_result(self, order: Order, reason: str) -> Dict[str, Any]:
        """Create result entry for a rejected order"""
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
            'commission': 0,
            'status': 'rejected',
            'timestamp': order.timestamp,
            'reason': reason
        }

    async def _close_specific(self):
        """Clean up backtest-specific resources"""
        if self.historical_data:
            self.historical_data.clear()
        self.asset_states.clear()
        self.execution_log.clear()
        if hasattr(self, 'trade_updates'):
            self.trade_updates.clear()