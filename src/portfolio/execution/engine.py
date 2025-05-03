#!/usr/bin/env python3
# src/portfolio/execution/engine.py

import asyncio
from decimal import Decimal
import pandas as pd
import uuid
from typing import Dict, Optional, List, Tuple, Any, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange
from src.exchange.factory import get_exchange_factory
from src.portfolio.execution.order import Order, OrderStatus, Direction, MarketOrder, LimitOrder


class ExecutionEngine:
    """Engine for executing trading orders with multiple execution modes."""
    
    def __init__(self, config: ConfigManager, mode: str = "backtest", 
                 historical_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Trading mode ("live", "paper", "backtest", "simple_backtest").
            historical_data (Optional[Dict[str, pd.DataFrame]]): Historical data for backtesting.
        """
        self.config = config
        self.mode = mode.lower()
        self.logger = LogManager.get_logger("execution.engine")
        self._running = True
        self.historical_data = historical_data or {}
        
        # Load settings from configuration
        self.commission = config.get_float("trading", "fees", "commission", 
                                          default=config.get_float("default_config", "user_config", "commission", default=0.001))
        self.slippage = config.get_float("trading", "execution", "slippage", 
                                        default=config.get_float("default_config", "user_config", "slippage", default=0.001))
        
        # Exchange interface (initialized lazily)
        self._exchange = None
        self._exchange_factory = None
        
        self.logger.info(f"Execution engine initialized in {self.mode} mode")

    async def initialize(self) -> None:
        """Initialize execution engine components"""
        if self.mode in ("live", "paper"):
            # Initialize exchange connection for live and paper trading
            self._exchange_factory = get_exchange_factory(self.config)
            
            try:
                self._exchange = await self._exchange_factory.create()
                self.logger.info(f"Exchange connection established for {self.mode} mode")
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange connection: {str(e)}")
                if self.mode == "live":
                    raise  # Re-raise for live mode
    
    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals and return executed orders and updated historical data (if applicable).

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Current prices for each symbol (required for live/paper modes).

        Returns:
            Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]: Executed orders and updated historical data (None for live/paper).
        """
        if signals.empty:
            self.logger.info("No signals to execute")
            return pd.DataFrame(), None

        self.logger.info(f"Executing {len(signals)} signals in {self.mode} mode")
        orders = await self._create_orders(signals)

        if self.mode == "live":
            # Execute orders on the exchange
            if not self._exchange:
                raise ValueError("Exchange interface required for live mode")
            executed_orders = await self._live_execution(orders)
            updated_data = None
            
        elif self.mode == "paper":
            # Simulate execution using real market data but without actual orders
            if not prices and not self._exchange:
                raise ValueError("Current prices or exchange required for paper trading")
            executed_orders = await self._paper_execution(orders, prices)
            updated_data = None
            
        elif self.mode == "backtest":
            # Full backtest with historical data updates
            if not self.historical_data:
                raise ValueError("Historical data required for backtest mode")
            executed_orders, updated_data = await self._backtest_execution(orders)
            
        elif self.mode == "simple_backtest":
            # Simple backtest without historical data updates
            if not self.historical_data:
                raise ValueError("Historical data required for simple_backtest mode")
            executed_orders = await self._simple_backtest_execution(orders)
            updated_data = None
            
        else:
            raise ValueError(f"Unsupported execution mode: {self.mode}")

        return executed_orders, updated_data

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

    async def _live_execution(self, orders: List[Order]) -> pd.DataFrame:
        """
        Execute orders on real exchange in live trading mode.
        
        Args:
            orders: List of orders to execute
            
        Returns:
            DataFrame of executed order results
        """
        executed_orders = []
        
        for order in orders:
            try:
                # Ensure exchange is available
                if not self._exchange:
                    raise ValueError(f"Exchange not available for live execution of {order.symbol}")
                
                # Convert to the parameter format required by the exchange
                params = {}
                
                # Add order-specific parameters
                if hasattr(order, 'reduce_only') and order.reduce_only:
                    params['reduceOnly'] = True
                
                # Create order on exchange
                response = await self._exchange.create_order(
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    side=order.direction.value,
                    amount=order.quantity,
                    price=getattr(order, 'price', None),
                    params=params
                )
                
                # Record transaction details
                executed_orders.append({
                    'order_id': response.get('id', order.order_id),
                    'symbol': order.symbol,
                    'direction': order.direction.value,
                    'filled_qty': float(response.get('filled', 0)),
                    'unfilled_qty': order.quantity - float(response.get('filled', 0)),
                    'price': float(response.get('price', 0)),
                    'avg_price': float(response.get('average', 0)) if response.get('average') else None,
                    'status': self._map_exchange_status(response.get('status', 'unknown')),
                    'timestamp': order.timestamp,
                    'exchange_order_id': response.get('id')
                })
                
                self.logger.info(f"Order {response.get('id', order.order_id)} submitted to exchange")
                
            except Exception as e:
                self.logger.error(f"Order execution failed: {str(e)}")
                # Record the failed order
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'direction': order.direction.value,
                    'filled_qty': 0,
                    'unfilled_qty': order.quantity,
                    'price': getattr(order, 'price', 0),
                    'avg_price': 0,
                    'status': 'failed',
                    'timestamp': order.timestamp,
                    'error': str(e)
                })

        return pd.DataFrame(executed_orders)

    async def _paper_execution(self, orders: List[Order], prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Simulate order execution for paper trading.
        
        Args:
            orders: List of orders to simulate
            prices: Current market prices (optional, will fetch from exchange if not provided)
            
        Returns:
            DataFrame of executed order results
        """
        executed_orders = []
        
        # Fetch current prices if not provided
        if not prices:
            prices = {}
            if self._exchange:
                for order in orders:
                    try:
                        ticker = await self._exchange.fetch_ticker(order.symbol)
                        prices[order.symbol] = ticker['last']
                    except Exception as e:
                        self.logger.error(f"Failed to fetch price for {order.symbol}: {str(e)}")
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Skip if no price available
                if symbol not in prices:
                    self.logger.warning(f"No price available for {symbol}, skipping order")
                    continue
                
                current_price = prices[symbol]
                
                # For limit orders, check price conditions
                if hasattr(order, 'price') and order.price is not None:
                    # For buy limit orders, only execute if current price <= limit price
                    if order.direction == Direction.BUY and current_price > order.price:
                        self.logger.info(f"Buy limit order for {symbol} not executed: current price {current_price} > limit {order.price}")
                        continue
                    
                    # For sell limit orders, only execute if current price >= limit price
                    if order.direction == Direction.SELL and current_price < order.price:
                        self.logger.info(f"Sell limit order for {symbol} not executed: current price {current_price} < limit {order.price}")
                        continue
                
                # Apply slippage
                execution_price = current_price
                if order.direction == Direction.BUY:
                    execution_price *= (1 + self.slippage)
                else:
                    execution_price *= (1 - self.slippage)
                
                # Calculate commission
                commission = order.quantity * execution_price * self.commission
                
                # Record the simulated execution
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'direction': order.direction.value,
                    'filled_qty': order.quantity,
                    'unfilled_qty': 0,
                    'price': execution_price,
                    'avg_price': execution_price,
                    'commission': commission,
                    'status': 'filled',
                    'timestamp': order.timestamp
                })
                
                self.logger.info(f"Paper execution: {order.direction.value} {order.quantity} {symbol} @ {execution_price:.6f}")
                
            except Exception as e:
                self.logger.error(f"Paper execution failed for {order.symbol}: {str(e)}")
                # Record the failed order
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'direction': order.direction.value,
                    'filled_qty': 0,
                    'unfilled_qty': order.quantity,
                    'price': getattr(order, 'price', 0),
                    'status': 'failed',
                    'timestamp': order.timestamp,
                    'error': str(e)
                })
        
        return pd.DataFrame(executed_orders)

    async def _backtest_execution(self, orders: List[Order]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Full backtest execution with historical data modification.
        
        Args:
            orders: List of orders to simulate
            
        Returns:
            Tuple of (executed orders DataFrame, updated historical data)
        """
        executed_orders = []
        updated_data = self.historical_data.copy()
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Skip if no historical data available
                if symbol not in updated_data:
                    self.logger.warning(f"No historical data for {symbol}, skipping order")
                    continue
                
                # Get relevant data for execution
                symbol_data = updated_data[symbol]
                
                # Find execution bar based on timestamp
                if order.timestamp is not None:
                    execution_bars = symbol_data[symbol_data['timestamp'] >= order.timestamp]
                    if execution_bars.empty:
                        self.logger.warning(f"No data after timestamp {order.timestamp} for {symbol}")
                        continue
                    execution_bar = execution_bars.iloc[0]
                else:
                    execution_bar = symbol_data.iloc[-1]  # Use last bar if no timestamp
                
                # Calculate execution volume
                max_volume = execution_bar['volume'] * 0.1  # Assume max 10% of bar volume tradable
                filled_qty = min(order.quantity, max_volume)
                
                if filled_qty <= 0:
                    self.logger.info(f"Not enough volume to execute order for {symbol}")
                    continue
                
                # Calculate execution price with slippage
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
                        continue
                    if order.direction == Direction.SELL and exec_price < order.price:
                        self.logger.info(f"Sell limit price {order.price} not reached by execution price {exec_price}")
                        continue
                
                # Reduce available volume in the bar
                updated_data[symbol].loc[execution_bar.name, 'volume'] -= filled_qty
                
                # Calculate commission
                commission = filled_qty * exec_price * self.commission
                
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
                    'status': 'filled' if filled_qty == order.quantity else 'partial',
                    'timestamp': execution_bar['timestamp'],
                    'bar_timestamp': execution_bar['timestamp']
                })
                
                self.logger.debug(f"Backtest execution: {order.direction.value} {filled_qty} {symbol} @ {exec_price:.6f}")
                
            except Exception as e:
                self.logger.error(f"Backtest execution failed for {order.symbol}: {str(e)}")
                # Record the failed order
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'direction': order.direction.value,
                    'filled_qty': 0,
                    'unfilled_qty': order.quantity,
                    'price': getattr(order, 'price', 0),
                    'status': 'failed',
                    'timestamp': order.timestamp,
                    'error': str(e)
                })
        
        return pd.DataFrame(executed_orders), updated_data

    async def _simple_backtest_execution(self, orders: List[Order]) -> pd.DataFrame:
        """
        Simple backtest execution without modifying historical data.
        
        Args:
            orders: List of orders to simulate
            
        Returns:
            DataFrame of executed order results
        """
        executed_orders = []
        
        for order in orders:
            try:
                symbol = order.symbol
                
                # Skip if no historical data available
                if symbol not in self.historical_data:
                    self.logger.warning(f"No historical data for {symbol}, skipping order")
                    continue
                
                # Get relevant data for execution
                symbol_data = self.historical_data[symbol]
                
                # Find execution bar based on timestamp
                if order.timestamp is not None:
                    matching_data = symbol_data[symbol_data['timestamp'] == order.timestamp]
                    if matching_data.empty:
                        # Try to find closest bar after timestamp
                        after_data = symbol_data[symbol_data['timestamp'] >= order.timestamp]
                        if after_data.empty:
                            self.logger.warning(f"No data after timestamp {order.timestamp} for {symbol}")
                            continue
                        execution_bar = after_data.iloc[0]
                    else:
                        execution_bar = matching_data.iloc[0]
                else:
                    execution_bar = symbol_data.iloc[-1]  # Use last bar if no timestamp
                
                # For limit orders, check if limit price is hit
                if hasattr(order, 'price') and order.price is not None:
                    # Check if the price has been reached during this bar
                    if order.direction == Direction.BUY and execution_bar['low'] > order.price:
                        self.logger.info(f"Buy limit price {order.price} not reached (low: {execution_bar['low']})")
                        continue
                    if order.direction == Direction.SELL and execution_bar['high'] < order.price:
                        self.logger.info(f"Sell limit price {order.price} not reached (high: {execution_bar['high']})")
                        continue
                
                # Simple execution uses the closing price
                exec_price = execution_bar['close']
                
                # Apply simulated slippage
                if order.direction == Direction.BUY:
                    exec_price *= (1 + self.slippage)
                else:
                    exec_price *= (1 - self.slippage)
                
                # Calculate commission
                commission = order.quantity * exec_price * self.commission
                
                # Record the execution
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': symbol,
                    'direction': order.direction.value,
                    'filled_qty': order.quantity,
                    'unfilled_qty': 0,
                    'price': exec_price,
                    'avg_price': exec_price,
                    'commission': commission,
                    'status': 'filled',
                    'timestamp': execution_bar['timestamp']
                })
                
                self.logger.debug(f"Simple backtest execution: {order.direction.value} {order.quantity} {symbol} @ {exec_price:.6f}")
                
            except Exception as e:
                self.logger.error(f"Simple backtest execution failed for {order.symbol}: {str(e)}")
                # Record the failed order
                executed_orders.append({
                    'order_id': order.order_id,
                    'symbol': order.symbol,
                    'direction': order.direction.value,
                    'filled_qty': 0,
                    'unfilled_qty': order.quantity,
                    'price': getattr(order, 'price', 0),
                    'status': 'failed',
                    'timestamp': order.timestamp,
                    'error': str(e)
                })
        
        return pd.DataFrame(executed_orders)

    async def close(self) -> None:
        """Close the execution engine and clean up resources."""
        if not self._running:
            return
        
        self._running = False
        self.historical_data.clear()
        
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
    
    def _map_exchange_status(self, status: str) -> str:
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