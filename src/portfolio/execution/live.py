#!/usr/bin/env python3
# src/portfolio/execution/live.py

import asyncio
from decimal import Decimal
import pandas as pd
import uuid
from typing import Dict, Optional, List, Tuple, Any, Union

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange, ExchangeAPIError
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.execution.order import Order, OrderStatus, Direction, MarketOrder, LimitOrder


class LiveExecutionEngine(BaseExecutionEngine):
    """
    Live execution engine implementation for real trading.
    
    This class handles order execution with real exchanges using the CCXT library,
    managing order creation, submission, and status tracking.
    """
    
    def __init__(self, config: ConfigManager, mode: str = "live", 
                 historical_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize the live execution engine.

        Args:
            config (ConfigManager): Configuration manager instance.
            mode (str): Should be "live".
            historical_data (Optional[Dict[str, pd.DataFrame]]): Not used in live mode.
        """
        super().__init__(config, mode, historical_data)
        
        # Live-specific configuration
        self.max_retry_attempts = config.get("trading", "execution", "max_retry_attempts", default=3)
        self.retry_delay = config.get("trading", "execution", "retry_delay", default=1.0)
        
        # Safety limits
        self.use_safety_limits = config.get("trading", "safety", "enabled", default=True)
        self.max_order_value = config.get("trading", "safety", "max_order_value", default=1000.0)  # Max order value in base currency
        self.max_slippage_percent = config.get("trading", "safety", "max_slippage_percent", default=2.0)  # Max acceptable slippage in percent
        
        self.logger.info("Live execution engine initialized with safety limits enabled" if self.use_safety_limits else "Live execution engine initialized with safety limits DISABLED")

    async def initialize(self) -> None:
        """
        Initialize the live execution engine and establish exchange connection.
        """
        await super().initialize()
        
        try:
            # Create exchange instance
            self._exchange = await self._exchange_factory.create()
            
            # Check connection
            if not self._exchange.is_initialized() or not self._exchange.is_connected():
                raise ConnectionError("Exchange is not properly initialized or connected")
                
            self.logger.info(f"Live execution engine connected to exchange {self._exchange.__class__.__name__}")
            
            # Load exchange-specific configurations
            await self._load_exchange_config()
        except Exception as e:
            self.logger.error(f"Failed to initialize live execution engine: {str(e)}")
            raise  # Re-raise as this is critical for live trading

    async def _load_exchange_config(self) -> None:
        """
        Load exchange-specific configurations and limits.
        """
        if not self._exchange:
            return
            
        try:
            # Some exchanges provide market information like precision, limits, etc.
            if hasattr(self._exchange, 'fetch_markets') and callable(self._exchange.fetch_markets):
                markets = await self._exchange.fetch_markets()
                self.logger.info(f"Loaded configuration for {len(markets)} markets from exchange")
                
                # Parse market data to extract useful information
                # This is exchange-specific and depends on what data is available
                
        except Exception as e:
            self.logger.warning(f"Could not load exchange configuration: {str(e)}")

    async def execute(self, signals: pd.DataFrame, prices: Optional[Dict[str, float]] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.DataFrame]]]:
        """
        Execute trading signals on the live exchange.

        Args:
            signals (pd.DataFrame): Signals with 'timestamp', 'symbol', 'action', 'quantity', and optional 'price'.
            prices (Optional[Dict[str, float]]): Current market prices (optional, will fetch if needed).

        Returns:
            Tuple[pd.DataFrame, None]: Executed order results and None (no historical data updates).
        """
        if signals.empty:
            self.logger.info("No signals to execute")
            return pd.DataFrame(), None
            
        self.logger.info(f"Executing {len(signals)} signals in live mode")
        
        # Convert signals to order objects
        orders = await self._create_orders(signals)
        
        # Execute orders on exchange
        executed_orders = await self._live_execution(orders, prices)
        
        # No historical data updates in live mode
        return executed_orders, None

    async def _live_execution(self, orders: List[Order], prices: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Execute orders on the live exchange.
        
        Args:
            orders: List of orders to execute
            prices: Current market prices (optional)
            
        Returns:
            DataFrame of executed order results
        """
        # Ensure exchange is available
        if not self._exchange:
            self.logger.error("No exchange available for live execution")
            return self._create_failed_execution_results(orders, "No exchange available")
            
        # Get current prices if needed
        if not prices:
            prices = await self._fetch_current_prices([order.symbol for order in orders])
        
        executed_orders = []
        
        for order in orders:
            try:
                # Apply safety checks
                if self.use_safety_limits and not self._check_safety_limits(order, prices.get(order.symbol)):
                    executed_orders.append(self._create_rejected_order_result(
                        order, "Order rejected due to safety limits"
                    ))
                    continue
                
                # Convert to exchange parameters
                params = self._prepare_exchange_params(order)
                
                # Execute the order with retry logic
                for attempt in range(1, self.max_retry_attempts + 1):
                    try:
                        # Create order on exchange
                        response = await self._exchange.create_order(
                            symbol=order.symbol,
                            order_type=order.order_type.value,
                            side=order.direction.value,
                            amount=order.quantity,
                            price=getattr(order, 'price', None),
                            params=params
                        )
                        
                        # Process successful response
                        executed_orders.append(self._process_exchange_response(order, response))
                        self.logger.info(f"Order {response.get('id', order.order_id)} submitted to exchange")
                        
                        # Store order in cache with exchange ID
                        order.exchange_order_id = response.get('id')
                        self._order_cache[order.order_id] = order
                        
                        # Update order status based on response
                        if hasattr(order, 'set_status'):
                            if response.get('status') == 'filled':
                                order.set_status(OrderStatus.FILLED)
                            elif response.get('status') == 'partial':
                                order.set_status(OrderStatus.PARTIAL)
                            else:
                                order.set_status(OrderStatus.SUBMITTED)
                        
                        break  # Success, exit retry loop
                    except ExchangeAPIError as e:
                        self.logger.warning(f"Exchange API error on attempt {attempt}/{self.max_retry_attempts}: {str(e)}")
                        if attempt < self.max_retry_attempts:
                            await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff
                        else:
                            # Final attempt failed
                            executed_orders.append(self._create_failed_order_result(order, str(e)))
                    except Exception as e:
                        # Non-API errors are not retried
                        self.logger.error(f"Order execution failed: {str(e)}")
                        executed_orders.append(self._create_failed_order_result(order, str(e)))
                        break
                        
            except Exception as e:
                self.logger.error(f"Unexpected error during order execution: {str(e)}")
                executed_orders.append(self._create_failed_order_result(order, str(e)))

        return pd.DataFrame(executed_orders)

    async def _fetch_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Fetch current prices for a list of symbols.
        
        Args:
            symbols: List of symbols to fetch prices for
            
        Returns:
            Dictionary mapping symbols to prices
        """
        prices = {}
        if not self._exchange:
            return prices
            
        for symbol in symbols:
            try:
                ticker = await self._exchange.fetch_ticker(symbol)
                prices[symbol] = ticker['last']
            except Exception as e:
                self.logger.error(f"Failed to fetch price for {symbol}: {str(e)}")
                
        return prices

    def _check_safety_limits(self, order: Order, current_price: Optional[float]) -> bool:
        """
        Check if an order passes safety limits.
        
        Args:
            order: Order to check
            current_price: Current market price
            
        Returns:
            True if order passes safety checks, False otherwise
        """
        # Skip checks if current price is not available
        if current_price is None:
            self.logger.warning(f"Cannot verify safety limits for {order.symbol} without current price")
            return True
            
        # Check order value
        order_value = order.quantity * current_price
        if order_value > self.max_order_value:
            self.logger.warning(f"Order value ${order_value:.2f} exceeds maximum ${self.max_order_value:.2f}")
            return False
            
        # For limit orders, check slippage from current price
        if hasattr(order, 'price') and order.price is not None:
            # Calculate percent difference between limit price and current price
            if order.direction == Direction.BUY:
                price_diff_pct = (order.price - current_price) / current_price * 100
            else:  # SELL
                price_diff_pct = (current_price - order.price) / current_price * 100
                
            if price_diff_pct > self.max_slippage_percent:
                self.logger.warning(f"Order price slippage {price_diff_pct:.2f}% exceeds maximum {self.max_slippage_percent:.2f}%")
                return False
                
        return True

    def _prepare_exchange_params(self, order: Order) -> Dict[str, Any]:
        """
        Prepare exchange-specific parameters for an order.
        
        Args:
            order: Order to prepare parameters for
            
        Returns:
            Exchange-specific parameters dictionary
        """
        params = {}
        
        # Add reduce-only flag for futures trading if applicable
        if hasattr(order, 'reduce_only') and order.reduce_only:
            params['reduceOnly'] = True
            
        # Add time-in-force parameter based on validity
        if hasattr(order, 'validity'):
            if order.validity.value == 'day':
                params['timeInForce'] = 'DAY'
            elif order.validity.value == 'gtc':
                params['timeInForce'] = 'GTC'
            elif order.validity.value == 'ioc':
                params['timeInForce'] = 'IOC'
            elif order.validity.value == 'fok':
                params['timeInForce'] = 'FOK'
                
        return params

    def _process_exchange_response(self, order: Order, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process exchange response into standardized order result.
        
        Args:
            order: The original order
            response: Exchange API response
            
        Returns:
            Standardized order result dictionary
        """
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'direction': order.direction.value,
            'filled_qty': float(response.get('filled', 0)),
            'unfilled_qty': order.quantity - float(response.get('filled', 0)),
            'price': float(response.get('price', 0)) if response.get('price') else None,
            'avg_price': float(response.get('average', 0)) if response.get('average') else None,
            'status': self._map_exchange_status(response.get('status', 'unknown')),
            'timestamp': order.timestamp,
            'exchange_order_id': response.get('id')
        }

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

    def _create_failed_execution_results(self, orders: List[Order], error_message: str) -> pd.DataFrame:
        """
        Create failed execution results for all orders.
        
        Args:
            orders: List of orders that couldn't be executed
            error_message: Common error message
            
        Returns:
            DataFrame with failure information for all orders
        """
        results = []
        for order in orders:
            results.append(self._create_failed_order_result(order, error_message))
        return pd.DataFrame(results)

    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel an order on the exchange.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        if not self._exchange:
            return {'success': False, 'error': 'No exchange available'}
            
        try:
            # Check if we have this order in our cache
            exchange_order_id = None
            if order_id in self._order_cache:
                order = self._order_cache[order_id]
                exchange_order_id = order.exchange_order_id
                
                # If the order is already in a final state, no need to cancel
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                    return {
                        'success': True, 
                        'order_id': order_id,
                        'status': 'already_' + order.status.value,
                        'message': f"Order already in {order.status.value} state"
                    }
            
            # Try to cancel on the exchange
            response = await self._exchange.cancel_order(
                exchange_order_id or order_id,  # Use exchange ID if available
                symbol
            )
            
            # Update cached order if available
            if order_id in self._order_cache:
                self._order_cache[order_id].set_status(OrderStatus.CANCELED)
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'status': 'canceled',
                'exchange_response': response
            }
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return {
                'success': False,
                'order_id': order_id,
                'symbol': symbol,
                'error': str(e)
            }

    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get current status of an order from the exchange.
        
        Args:
            order_id: Order ID to check
            symbol: Trading symbol
            
        Returns:
            Order status dictionary
        """
        # First try from cache
        cache_result = await super().get_order_status(order_id, symbol)
        if 'from_cache' in cache_result and cache_result['from_cache']:
            # For non-final states, verify with exchange
            if cache_result['status'] not in ('filled', 'canceled', 'rejected'):
                pass  # Continue to exchange check
            else:
                # Return cache result for final states
                return cache_result
                
        # Try to fetch from exchange
        if not self._exchange:
            return {'success': False, 'error': 'No exchange available'}
            
        try:
            # Get exchange order ID if available
            exchange_order_id = None
            if order_id in self._order_cache:
                exchange_order_id = self._order_cache[order_id].exchange_order_id
                
            # Fetch from exchange
            response = await self._exchange.fetch_order(
                exchange_order_id or order_id,  # Use exchange ID if available
                symbol
            )
            
            # Update cache if order found in our cache
            if order_id in self._order_cache:
                order = self._order_cache[order_id]
                status = self._map_exchange_status(response.get('status', 'unknown'))
                order.set_status(OrderStatus(status))
                
                # Update fill information if available
                filled_quantity = float(response.get('filled', 0))
                if filled_quantity > 0 and hasattr(order, 'fill') and callable(order.fill):
                    avg_price = float(response.get('average', 0))
                    if avg_price > 0:
                        order.fill(filled_quantity, avg_price, float('inf'))
            
            # Return standardized status
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'status': self._map_exchange_status(response.get('status', 'unknown')),
                'filled_qty': float(response.get('filled', 0)),
                'unfilled_qty': float(response.get('remaining', 0)),
                'avg_price': float(response.get('average', 0)) if response.get('average') else 0,
                'direction': response.get('side', 'unknown'),
                'timestamp': response.get('timestamp'),
                'exchange_order_id': response.get('id')
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch order status for {order_id}: {str(e)}")
            return {
                'success': False,
                'order_id': order_id,
                'symbol': symbol,
                'error': str(e)
            }