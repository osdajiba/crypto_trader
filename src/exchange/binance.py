#!/usr/bin/env python3
# src/exchange/implementations/binance.py

"""
Binance exchange implementation.
Provides interface to Binance exchange API.
"""

import ccxtpro
import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import asyncio
import time
import os
import random
from typing import Dict, List, Union, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from functools import wraps
from decimal import Decimal

from src.common.config_manager import ConfigManager
from src.common.abstract_factory import register_factory_class
from src.common.helpers import TimeUtils
from src.exchange.base import (
    Exchange, ExchangeError, ExchangeAPIError, ExchangeRateLimitError, 
    ExchangeConnectionError, ExchangeOrderError, retry_exchange_operation
)


def retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0):
    """Retry decorator with exponential backoff and jitter"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, ccxt.RequestTimeout) as e:
                    self = args[0]  # Extract self from args
                    self.logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    last_exception = e
                    
                    # Calculate exponential backoff delay with jitter
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0.5, 1.0)  # 50%-100% random jitter
                    adjusted_delay = delay * jitter
                    
                    self.logger.info(f"Waiting {adjusted_delay:.2f} seconds before retry...")
                    await asyncio.sleep(adjusted_delay)
                except Exception as e:
                    self = args[0]  # Extract self from args
                    self.logger.error(f"Unhandled error: {str(e)}")
                    raise
            
            # All attempts failed
            self = args[0]  # Extract self from args
            self.logger.error(f"All attempts failed. Last error: {last_exception}")
            raise last_exception
        return async_wrapper
    return decorator


@register_factory_class('exchange_factory', 'binance', 
                        description="Binance exchange integration",
                        features=["spot", "futures", "websocket"],
                        category="exchange")
class BinanceExchange(Exchange):
    """
    Binance exchange implementation
    
    This class provides a full implementation of the Exchange interface
    for the Binance cryptocurrency exchange, supporting both REST API
    and WebSocket connections.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Binance exchange
        
        Args:
            config: Configuration manager
            params: Additional parameters
        """
        super().__init__(config, params)
        
        # Exchange-specific attributes
        self.exchange = None
        self.async_exchange = None
        self.request_semaphore = None
        self.ws_subscriptions = {}
        
        # Settings from config
        self.download_chunk_size = self.config.get('api', 'download_chunk_size', default=5000)
        self.max_retry_attempts = self.config.get('network', 'connection', 'retry_attempts', default=3)
        self.retry_delay = self.config.get('network', 'connection', 'retry_delay', default=2.0)
        
        # Set up cache directory
        self.cache_dir = self._get_cache_dir()
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Market types and precisions
        self._market_precisions = {}
        self._market_limits = {}
        
        # Order type mappings
        self._order_type_map = {
            'market': 'MARKET',
            'limit': 'LIMIT',
            'stop': 'STOP',
            'stop_market': 'STOP_MARKET',
            'stop_limit': 'STOP',
            'take_profit': 'TAKE_PROFIT',
            'take_profit_market': 'TAKE_PROFIT_MARKET',
            'take_profit_limit': 'TAKE_PROFIT',
            'trailing_stop_market': 'TRAILING_STOP_MARKET'
        }
        
        # Order side mappings
        self._order_side_map = {
            'buy': 'BUY',
            'sell': 'SELL'
        }
        
        # Trading mode
        self._trading_mode = self.config.get('system', 'operational_mode', default='backtest')
    
    def _get_cache_dir(self) -> str:
        """Get cache directory from configuration"""
        cache_dir = self.config.get('data', 'cache', 'dir', default='./cache')
        return os.path.abspath(os.path.expanduser(cache_dir))
    
    def _build_params(self) -> Dict:
        """
        Build CCXT parameters from configuration
        
        Returns:
            Dict: CCXT parameters
        """
        params = {
            'apiKey': self.config.get('api', 'binance', 'api_key', default=''),
            'secret': self.config.get('api', 'binance', 'secret', default=''),
            'timeout': self.config.get('api', 'timeout', default=60000),
            'enableRateLimit': self.config.get('api', 'enable_rate_limit', default=True),
            'options': {
                'adjustForTimeDifference': self.config.get(
                    'api', 'binance', 'options', 'adjust_time_diff', default=True),
                'recvWindow': self.config.get(
                    'api', 'binance', 'options', 'recv_window', default=60000),
                'defaultType': self.config.get(
                    'api', 'binance', 'options', 'default_type', default='spot'),
                'keepAlive': True,
            },
            'headers': {
                'Connection': 'keep-alive',
                'Keep-Alive': '60',
            }
        }
        
        # Check for proxy configuration
        use_proxy = self.config.get('api', 'useproxy', default=None)
        
        if use_proxy is True or use_proxy == "true":
            # Get proxy from configuration
            http_proxy = self.config.get('proxies', 'http', default=None)
            https_proxy = self.config.get('proxies', 'https', default=None)
            
            proxies = {}
            if http_proxy or https_proxy:
                proxies = {
                    'http': http_proxy,
                    'https': https_proxy or http_proxy
                }
                self.logger.info(f"Using configured proxies: {proxies}")
                if proxies:
                    params['proxies'] = proxies
        
        return params
    
    async def _initialize_exchange(self) -> None:
        """
        Initialize Binance exchange connection
        
        Raises:
            ExchangeConnectionError: If initialization fails
        """
        try:
            self.logger.info("Initializing Binance exchange")
            
            # Build params from config
            params = self._build_params()
            
            # Initialize synchronous exchange
            self.exchange = ccxt.binance(params)
            self.logger.info("Synchronous exchange initialized successfully")
            
            # Set up async request semaphore
            self.request_semaphore = asyncio.Semaphore(5)  # Limit max concurrent requests
            
            # Preload market data
            try:
                self.exchange.load_markets()
                self.logger.info("Market data loaded successfully")
                self._load_market_info()
            except Exception as e:
                self.logger.warning(f"Failed to load market data: {str(e)}, will load on first call")
            
            self._connected = True
            
        except Exception as e:
            self.logger.error(f"Error initializing Binance exchange: {str(e)}")
            raise ExchangeConnectionError(f"Cannot initialize exchange: {str(e)}")
    
    def _load_market_info(self) -> None:
        """Load market information including precision and limits"""
        if not self.exchange or not self.exchange.markets:
            return
            
        for symbol, market in self.exchange.markets.items():
            # Extract precision info
            if 'precision' in market:
                self._market_precisions[symbol] = {
                    'price': market['precision'].get('price', 8),
                    'amount': market['precision'].get('amount', 8),
                    'cost': market['precision'].get('cost', 8)
                }
            
            # Extract limits
            if 'limits' in market:
                self._market_limits[symbol] = {
                    'amount': {
                        'min': market['limits']['amount'].get('min', 0),
                        'max': market['limits']['amount'].get('max', None)
                    },
                    'price': {
                        'min': market['limits']['price'].get('min', 0),
                        'max': market['limits']['price'].get('max', None)
                    },
                    'cost': {
                        'min': market['limits']['cost'].get('min', 0),
                        'max': market['limits']['cost'].get('max', None)
                    },
                    'market': {
                        'min': market['limits'].get('market', {}).get('min', 0),
                        'max': market['limits'].get('market', {}).get('max', None)
                    }
                }
    
    async def _initialize_offline(self) -> None:
        """Initialize exchange in offline mode"""
        self.logger.info("Initializing Binance exchange in offline mode")
        self.exchange = None
        self._connected = False
    
    async def _initialize_fallback(self) -> None:
        """Initialize fallback mode for backtesting"""
        self.logger.warning("Initializing minimal Binance exchange for backtest support")
        self.exchange = ccxt.binance()
        self.exchange.markets = {}
        self.exchange.markets_by_id = {}
        self._connected = False
    
    async def _init_async_exchange(self) -> None:
        """
        Initialize async exchange (lazy initialization) optimized for high-latency networks
        
        Raises:
            ExchangeConnectionError: If initialization fails
        """
        if self.async_exchange is not None:
            return

        try:
            # Create optimized aiohttp session
            import aiohttp
            from aiohttp import ClientTimeout
            
            # Optimize HTTP connector for high-latency networks
            connector = aiohttp.TCPConnector(
                keepalive_timeout=60,
                limit_per_host=10,
                limit=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
            
            # Set longer timeouts for high-latency
            timeout = ClientTimeout(
                total=60,
                connect=20,
                sock_read=45,
                sock_connect=20
            )
            
            # Create session
            session = aiohttp.ClientSession(
                connector=connector, 
                timeout=timeout,
                headers={'Connection': 'keep-alive', 'Keep-Alive': '60'}
            )
            
            # Build params from config
            params = self._build_params()
            
            # Initialize async exchange
            self.async_exchange = ccxt_async.binance(params)
            
            # Apply optimized session
            self.async_exchange.session = session
            
            # Initialize concurrency limiting semaphore if not already set
            if not self.request_semaphore:
                self.request_semaphore = asyncio.Semaphore(5)
            
            self.logger.info("Async Binance exchange initialized with high-latency network optimizations")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize async Binance exchange: {str(e)}")
            self.async_exchange = None
            raise ExchangeConnectionError(f"Failed to initialize async exchange: {str(e)}")
    
    async def _shutdown_exchange(self) -> None:
        """Clean up exchange resources"""
        # Close all WebSocket subscriptions
        for sub_key, sub_info in list(self.ws_subscriptions.items()):
            try:
                sub_info['active'] = False
                exchange = sub_info.get('exchange')
                if exchange and hasattr(exchange, 'close'):
                    await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing WebSocket subscription {sub_key}: {str(e)}")
        
        # Close async exchange
        try:
            if self.async_exchange:
                await self.async_exchange.close()
                # Close underlying session
                if hasattr(self.async_exchange, 'session') and hasattr(self.async_exchange.session, 'close'):
                    await self.async_exchange.session.close()
                self.async_exchange = None
                self.logger.info("Async exchange connection closed")
        except Exception as e:
            self.logger.error(f"Error closing async exchange: {str(e)}")
    
    @retry_exchange_operation(max_attempts=3, base_delay=2.0, max_delay=30.0)
    async def create_order(self, 
                     symbol: str, 
                     order_type: str, 
                     side: str, 
                     amount: float,
                     price: Optional[float] = None, 
                     params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an order on the exchange with enhanced error handling and validation
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type (e.g., 'market', 'limit')
            side: Order side ('buy' or 'sell')
            amount: Order amount
            price: Order price (required for limit orders)
            params: Additional parameters for the order
            
        Returns:
            Dict[str, Any]: Order information with standardized format
            
        Raises:
            ExchangeOrderError: If order creation fails
        """
        if not self._initialized:
            await self.initialize()
        
        # Validate parameters
        self._validate_order_params(symbol, order_type, side, amount, price)
        
        # Apply precision and limits
        amount = self._apply_amount_precision(symbol, amount)
        if price:
            price = self._apply_price_precision(symbol, price)
        
        # Standardize order type and side
        std_order_type = self._order_type_map.get(order_type.lower(), 'LIMIT')
        std_side = self._order_side_map.get(side.lower(), 'BUY')
        
        try:
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Handle rate limit
            await self._handle_rate_limit()
            
            # Create order with Binance
            result = await self.async_exchange.create_order(
                symbol=symbol,
                type=std_order_type.lower(),
                side=std_side.lower(),
                amount=amount,
                price=price,
                params=params or {}
            )
            
            # Standardize response format
            standardized_result = self._standardize_order_response(result)
            
            # Cache order information
            order_id = standardized_result.get('order_id') or standardized_result.get('id')
            if order_id:
                self._active_orders[order_id] = standardized_result
            
            # Emit order created event
            self.emit_event('order_created', {
                'order_id': order_id,
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': price,
                'status': standardized_result.get('status', 'unknown')
            })
            
            return standardized_result
            
        except ccxt.OrderNotFound:
            error_msg = f"Order not found: {symbol} {side} {amount}"
            self.logger.error(error_msg)
            raise ExchangeOrderError(error_msg)
        except ccxt.InsufficientFunds:
            error_msg = f"Insufficient funds for order: {symbol} {side} {amount}"
            self.logger.error(error_msg)
            raise ExchangeOrderError(error_msg)
        except ccxt.InvalidOrder as e:
            error_msg = f"Invalid order: {symbol} {side} {amount} - {str(e)}"
            self.logger.error(error_msg)
            raise ExchangeOrderError(error_msg)
        except Exception as e:
            error_msg = f"Order creation failed: {symbol} {side} {amount} - {str(e)}"
            self.logger.error(error_msg)
            raise ExchangeOrderError(error_msg)
    
    def _validate_order_params(self, symbol: str, order_type: str, 
                              side: str, amount: float, price: Optional[float]) -> None:
        """
        Validate order parameters against market limits
        
        Args:
            symbol: Trading pair symbol
            order_type: Order type
            side: Order side
            amount: Order amount
            price: Order price (optional)
            
        Raises:
            ExchangeOrderError: If validation fails
        """
        # Check if symbol exists
        if symbol not in self._market_limits:
            raise ExchangeOrderError(f"Symbol not found: {symbol}")
        
        # Check amount limits
        limits = self._market_limits[symbol]
        min_amount = limits['amount']['min']
        max_amount = limits['amount']['max']
        
        if amount < min_amount:
            raise ExchangeOrderError(f"Amount {amount} below minimum {min_amount} for {symbol}")
        
        if max_amount and amount > max_amount:
            raise ExchangeOrderError(f"Amount {amount} exceeds maximum {max_amount} for {symbol}")
        
        # Check price limits for limit orders
        if price and order_type.lower() in ['limit', 'stop_limit']:
            min_price = limits['price']['min']
            max_price = limits['price']['max']
            
            if price < min_price:
                raise ExchangeOrderError(f"Price {price} below minimum {min_price} for {symbol}")
            
            if max_price and price > max_price:
                raise ExchangeOrderError(f"Price {price} exceeds maximum {max_price} for {symbol}")
        
        # Check cost (notional) limits
        if price and order_type.lower() in ['limit', 'market']:
            estimated_cost = amount * price
            min_cost = limits['cost']['min']
            
            if estimated_cost < min_cost:
                raise ExchangeOrderError(f"Order value {estimated_cost} below minimum {min_cost} for {symbol}")
    
    def _apply_amount_precision(self, symbol: str, amount: float) -> float:
        """Apply precision to order amount"""
        if symbol in self._market_precisions:
            precision = self._market_precisions[symbol]['amount']
            return round(amount, precision)
        return amount
    
    def _apply_price_precision(self, symbol: str, price: float) -> float:
        """Apply precision to order price"""
        if symbol in self._market_precisions:
            precision = self._market_precisions[symbol]['price']
            return round(price, precision)
        return price
    
    def _standardize_order_response(self, raw_order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize order response format for consistency
        
        Args:
            raw_order: Raw order response from exchange
            
        Returns:
            Dict[str, Any]: Standardized order information
        """
        standardized = {
            'order_id': raw_order.get('id'),
            'client_order_id': raw_order.get('clientOrderId'),
            'symbol': raw_order.get('symbol'),
            'order_type': raw_order.get('type', '').lower(),
            'side': raw_order.get('side', '').lower(),
            'amount': raw_order.get('amount', 0),
            'price': raw_order.get('price', 0),
            'filled': raw_order.get('filled', 0),
            'remaining': raw_order.get('remaining', 0),
            'cost': raw_order.get('cost', 0),
            'average_price': raw_order.get('average', 0),
            'fee': {
                'cost': raw_order.get('fee', {}).get('cost', 0),
                'currency': raw_order.get('fee', {}).get('currency', '')
            },
            'status': self.map_order_status(raw_order.get('status', 'unknown')),
            'timestamp': raw_order.get('timestamp'),
            'datetime': raw_order.get('datetime'),
            'raw': raw_order  # Keep original for reference
        }
        
        return standardized
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Cancel an order on the exchange
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Cancellation result
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Cancel order
            result = await self.async_exchange.cancel_order(order_id, symbol, params)
            
            # Update order cache
            if order_id in self._active_orders:
                self._active_orders[order_id]['status'] = 'canceled'
                self._order_history[order_id] = self._active_orders.pop(order_id)
            
            # Emit order canceled event
            self.emit_event('order_canceled', {
                'order_id': order_id,
                'symbol': symbol,
                'status': 'canceled'
            })
            
            return self._standardize_order_response(result)
            
        except ccxt.OrderNotFound:
            self.logger.warning(f"Order {order_id} not found for cancellation")
            if order_id in self._active_orders:
                # If locally tracked but not found on exchange, mark as completed
                self._active_orders[order_id]['status'] = 'filled'
                self._order_history[order_id] = self._active_orders.pop(order_id)
            return {'success': False, 'order_id': order_id, 'error': 'Order not found'}
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            raise ExchangeOrderError(f"Cancel order failed: {str(e)}")
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def fetch_order(self, order_id: str, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch order details with caching
        
        Args:
            order_id: Order ID to fetch
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Order details
        """
        # Check cache first if order might be completed
        if order_id in self._order_history:
            return self._order_history[order_id]
        elif order_id in self._active_orders:
            cached_order = self._active_orders[order_id]
            # If cached order is marked as final state and not too old
            if (cached_order.get('status') in ['filled', 'canceled', 'rejected', 'expired'] 
                and time.time() - cached_order.get('cache_time', 0) < 300):  # 5 minutes
                return cached_order
        
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Fetch order from exchange
            result = await self.async_exchange.fetch_order(order_id, symbol, params)
            
            # Standardize response
            standardized_result = self._standardize_order_response(result)
            
            # Update cache
            standardized_result['cache_time'] = time.time()
            
            if standardized_result['status'] in ['filled', 'canceled', 'rejected', 'expired']:
                # Move to history if completed
                if order_id in self._active_orders:
                    del self._active_orders[order_id]
                self._order_history[order_id] = standardized_result
            else:
                # Keep in active orders
                self._active_orders[order_id] = standardized_result
            
            return standardized_result
            
        except ccxt.OrderNotFound:
            self.logger.warning(f"Order {order_id} not found when fetching")
            return {'success': False, 'order_id': order_id, 'error': 'Order not found'}
        except Exception as e:
            self.logger.error(f"Failed to fetch order {order_id}: {str(e)}")
            raise ExchangeOrderError(f"Fetch order failed: {str(e)}")
    
    async def fetch_open_orders(self, symbol: Optional[str] = None, 
                         since: Optional[int] = None, 
                         limit: Optional[int] = None, 
                         params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Fetch open orders"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure async exchange is initialized
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Fetch open orders
            result = await self.async_exchange.fetch_open_orders(symbol, since, limit, params)
            
            # Standardize and cache results
            standardized_orders = []
            for order in result:
                standardized = self._standardize_order_response(order)
                standardized['cache_time'] = time.time()
                
                order_id = standardized.get('order_id')
                if order_id:
                    self._active_orders[order_id] = standardized
                    
                standardized_orders.append(standardized)
            
            return standardized_orders
            
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {str(e)}")
            return []
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def fetch_position(self, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch position information for futures trading
        
        Args:
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Position information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure async exchange is initialized and in futures mode
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Switch to futures mode if needed
            original_type = self.async_exchange.options.get('defaultType', 'spot')
            self.async_exchange.options['defaultType'] = 'future'
            
            try:
                # Fetch position
                result = await self.async_exchange.fetch_position(symbol, params)
                
                # Cache position
                self._position_cache[symbol] = result
                
                return result
            finally:
                # Restore original mode
                self.async_exchange.options['defaultType'] = original_type
                
        except Exception as e:
            self.logger.error(f"Failed to fetch position for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch position failed: {str(e)}")
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def set_leverage(self, leverage: float, symbol: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set leverage for futures trading
        
        Args:
            leverage: Leverage level
            symbol: Trading pair symbol
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Response from exchange
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Ensure async exchange is initialized and in futures mode
            if not self.async_exchange:
                await self._init_async_exchange()
            
            # Switch to futures mode if needed
            original_type = self.async_exchange.options.get('defaultType', 'spot')
            self.async_exchange.options['defaultType'] = 'future'
            
            try:
                # Set leverage
                result = await self.async_exchange.set_leverage(leverage, symbol, params)
                
                self.logger.info(f"Set leverage for {symbol} to {leverage}x")
                
                return result
            finally:
                # Restore original mode
                self.async_exchange.options['defaultType'] = original_type
                
        except Exception as e:
            self.logger.error(f"Failed to set leverage for {symbol}: {str(e)}")
            raise ExchangeError(f"Set leverage failed: {str(e)}")
    
    async def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get market information for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict[str, Any]: Market information
        """
        if symbol in self._market_cache:
            return self._market_cache[symbol]
            
        if not self.exchange or not self.exchange.markets:
            try:
                await self.initialize()
            except Exception as e:
                self.logger.error(f"Failed to initialize exchange: {e}")
                return {}
        
        if symbol in self.exchange.markets:
            market_info = self.exchange.markets[symbol].copy()
            # Add precision and limits
            if symbol in self._market_precisions:
                market_info['precision'] = self._market_precisions[symbol]
            if symbol in self._market_limits:
                market_info['limits'] = self._market_limits[symbol]
            
            self._market_cache[symbol] = market_info
            return market_info
        
        return {}
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Fetch ticker data with caching"""
        # Check cache
        if symbol in self._ticker_cache:
            cache_data, cache_time = self._ticker_cache[symbol]
            if time.time() - cache_time < 5:  # 5 second cache
                return cache_data
        
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.async_exchange:
                await self._init_async_exchange()
            
            ticker = await self.async_exchange.fetch_ticker(symbol)
            
            # Cache ticker data
            self._ticker_cache[symbol] = (ticker, time.time())
            
            return ticker
        except Exception as e:
            self.logger.error(f"Failed to fetch ticker for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch ticker failed: {str(e)}")
    
    @staticmethod
    def _map_order_status(binance_status: str) -> str:
        """Map Binance order status to standard status"""
        binance_status_map = {
            'NEW': 'created',
            'PARTIALLY_FILLED': 'partial',
            'FILLED': 'filled',
            'CANCELED': 'canceled',
            'REJECTED': 'rejected',
            'EXPIRED': 'expired',
            'PENDING_CANCEL': 'canceling',
            'PENDING_NEW': 'submitting'
        }
        
        return binance_status_map.get(binance_status.upper(), 'unknown')
    
    async def fetch_balance(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch account balance
        
        Args:
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Account balance information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.async_exchange:
                await self._init_async_exchange()
            
            balance = await self.async_exchange.fetch_balance(params)
            
            # Standardize balance format
            standardized = {
                'info': balance.get('info', {}),
                'currencies': {}
            }
            
            for currency, data in balance.items():
                if currency not in ['info', 'free', 'used', 'total']:
                    standardized['currencies'][currency] = {
                        'free': data.get('free', 0),
                        'used': data.get('used', 0),
                        'total': data.get('total', 0)
                    }
            
            return standardized
            
        except Exception as e:
            self.logger.error(f"Failed to fetch balance: {str(e)}")
            raise ExchangeError(f"Fetch balance failed: {str(e)}")
    
    async def fetch_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch recent public trades for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Number of trades to fetch
            
        Returns:
            List[Dict[str, Any]]: Recent trades
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.async_exchange:
                await self._init_async_exchange()
            
            trades = await self.async_exchange.fetch_trades(symbol, limit=limit)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to fetch trades for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch trades failed: {str(e)}")
    
    async def fetch_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Fetch order book for a symbol
        
        Args:
            symbol: Trading pair symbol
            limit: Order book depth
            
        Returns:
            Dict[str, Any]: Order book data
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.async_exchange:
                await self._init_async_exchange()
            
            order_book = await self.async_exchange.fetch_order_book(symbol, limit=limit)
            
            # Cache order book data
            self._order_book_cache[symbol] = (order_book, time.time())
            
            return order_book
            
        except Exception as e:
            self.logger.error(f"Failed to fetch order book for {symbol}: {str(e)}")
            raise ExchangeError(f"Fetch order book failed: {str(e)}")
    
    def set_market_type(self, market_type: str) -> None:
        """
        Set market type for trading (override from base)
        
        Args:
            market_type: Market type ('spot', 'future', 'margin', etc.)
        """
        super().set_market_type(market_type)
        
        # Update exchange options if needed
        if self.exchange:
            default_type = 'future' if market_type == 'future' else 'spot'
            self.exchange.options['defaultType'] = default_type
        
        if self.async_exchange:
            async_default_type = 'future' if market_type == 'future' else 'spot'
            self.async_exchange.options['defaultType'] = async_default_type
    
    async def create_stop_loss_order(self, 
                              symbol: str, 
                              side: str, 
                              amount: float,
                              stop_price: float,
                              price: Optional[float] = None,
                              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a stop loss order
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            stop_price: Stop price to trigger the order
            price: Limit price (for stop limit orders)
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Order information
        """
        params = params or {}
        
        # Add stop price parameter
        params['stopPrice'] = stop_price
        
        # Determine order type
        order_type = 'stop_limit' if price else 'stop_market'
        
        return await self.create_order(symbol, order_type, side, amount, price, params)
    
    async def create_take_profit_order(self, 
                                symbol: str, 
                                side: str, 
                                amount: float,
                                stop_price: float,
                                price: Optional[float] = None,
                                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a take profit order
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            stop_price: Stop price to trigger the order
            price: Limit price (for take profit limit orders)
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Order information
        """
        params = params or {}
        
        # Add stop price parameter
        params['stopPrice'] = stop_price
        
        # Determine order type
        order_type = 'take_profit_limit' if price else 'take_profit_market'
        
        return await self.create_order(symbol, order_type, side, amount, price, params)
    
    async def create_trailing_stop_order(self, 
                                  symbol: str, 
                                  side: str, 
                                  amount: float,
                                  callback_rate: float,
                                  activation_price: Optional[float] = None,
                                  params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a trailing stop market order
        
        Args:
            symbol: Trading pair symbol
            side: Order side ('buy' or 'sell')
            amount: Order amount
            callback_rate: Callback rate in percentage
            activation_price: Activation price for trailing stop
            params: Additional parameters
            
        Returns:
            Dict[str, Any]: Order information
        """
        params = params or {}
        
        # Add trailing stop parameters
        params['callbackRate'] = callback_rate
        if activation_price:
            params['activationPrice'] = activation_price
        
        return await self.create_order(symbol, 'trailing_stop_market', side, amount, None, params)
    
    async def get_order_status_from_exchange(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get real-time order status directly from exchange (not cached)
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol
            
        Returns:
            Dict[str, Any]: Order status information
        """
        try:
            order = await self.fetch_order(order_id, symbol)
            
            return {
                'success': True,
                'order_id': order_id,
                'symbol': symbol,
                'status': order.get('status', 'unknown'),
                'filled_qty': order.get('filled', 0),
                'unfilled_qty': order.get('remaining', 0),
                'avg_price': order.get('average_price', 0),
                'timestamp': order.get('timestamp'),
                'raw': order
            }
        except Exception as e:
            return {
                'success': False,
                'order_id': order_id,
                'symbol': symbol,
                'error': str(e)
            }
    
    async def cancel_all_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancel all open orders for a symbol or all symbols
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            Dict[str, Any]: Cancellation results
        """
        results = {
            'success': True,
            'canceled': [],
            'failed': [],
            'errors': []
        }
        
        try:
            # Fetch all open orders
            open_orders = await self.fetch_open_orders(symbol)
            
            # Cancel each order
            for order in open_orders:
                order_id = order.get('order_id')
                order_symbol = order.get('symbol')
                
                if order_id and order_symbol:
                    try:
                        cancel_result = await self.cancel_order(order_id, order_symbol)
                        if cancel_result.get('success', True):
                            results['canceled'].append(order_id)
                        else:
                            results['failed'].append(order_id)
                            results['errors'].append(cancel_result.get('error', 'Unknown error'))
                    except Exception as e:
                        results['failed'].append(order_id)
                        results['errors'].append(str(e))
                        results['success'] = False
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to cancel all open orders: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'canceled': [],
                'failed': [],
                'errors': [str(e)]
            }
    
    def get_order_from_cache(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order from local cache without hitting the API
        
        Args:
            order_id: Order ID
            
        Returns:
            Optional[Dict[str, Any]]: Order data if found, None otherwise
        """
        if order_id in self._active_orders:
            return self._active_orders[order_id]
        elif order_id in self._order_history:
            return self._order_history[order_id]
        return None
    
    def clear_order_cache(self, order_id: Optional[str] = None) -> None:
        """
        Clear order cache
        
        Args:
            order_id: Specific order ID to clear, or None to clear all
        """
        if order_id:
            self._active_orders.pop(order_id, None)
            self._order_history.pop(order_id, None)
        else:
            self._active_orders.clear()
            self._order_history.clear()
        
        self.logger.info(f"Cleared order cache for {order_id or 'all orders'}")
    
    async def fetch_my_trades(self, 
                             symbol: Optional[str] = None, 
                             since: Optional[int] = None, 
                             limit: Optional[int] = None,
                             params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch user's trade history
        
        Args:
            symbol: Trading pair symbol (optional)
            since: Timestamp to fetch trades from
            limit: Maximum number of trades to fetch
            params: Additional parameters
            
        Returns:
            List[Dict[str, Any]]: User's trade history
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            if not self.async_exchange:
                await self._init_async_exchange()
            
            trades = await self.async_exchange.fetch_my_trades(symbol, since, limit, params)
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Failed to fetch my trades: {str(e)}")
            raise ExchangeError(f"Fetch my trades failed: {str(e)}")
    
    async def get_trading_fees(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading fees for a symbol
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Dict[str, Any]: Trading fees information
        """
        try:
            market_info = await self.get_market_info(symbol)
            
            fees = {
                'maker': self.config.get('trading', 'fees', 'commission_maker', default=0.001),
                'taker': self.config.get('trading', 'fees', 'commission_taker', default=0.001)
            }
            
            # Override with market specific fees if available
            if 'fees' in market_info:
                if 'trading' in market_info['fees']:
                    fees.update({
                        'maker': market_info['fees']['trading'].get('maker', fees['maker']),
                        'taker': market_info['fees']['trading'].get('taker', fees['taker'])
                    })
            
            return fees
            
        except Exception as e:
            self.logger.error(f"Failed to get trading fees for {symbol}: {str(e)}")
            return {
                'maker': 0.001,
                'taker': 0.001
            }
    
    async def watch_order_updates(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """
        Subscribe to order updates via WebSocket
        
        Args:
            callback: Callback function to receive order updates
            
        Returns:
            bool: Success status
        """
        try:
            # Check if ccxt.pro is installed
            try:
            except ImportError:
                self.logger.error("ccxt.pro not installed. Please install ccxt.pro to use WebSocket functionality.")
                return False
            
            self.logger.info("Starting WebSocket connection for order updates")
            
            # Get params from config
            params = self._build_params()
            
            # Initialize WebSocket exchange
            ws_exchange = ccxtpro.binance(params)
            
            # Configure WebSocket options
            ws_exchange.options['ws'] = {
                'heartbeat': True,
                'ping_interval': 30000,
                'reconnect_rate': 5000,
                'max_reconnects': 100,
            }
            
            # Start WebSocket task
            asyncio.create_task(self._watch_orders_manager(ws_exchange, callback))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up order updates WebSocket: {str(e)}")
            return False
    
    async def _watch_orders_manager(self, exchange, callback):
        """
        Manage the continuous operation of order updates WebSocket
        
        Args:
            exchange: CCXT pro exchange instance
            callback: Callback function to receive updates
        """
        retry_delay = 1.0
        max_retry_delay = 30.0
        error_count = 0
        
        while True:
            try:
                self.logger.debug("Waiting for order update...")
                orders = await exchange.watch_orders()
                
                error_count = 0  # Reset error count on success
                retry_delay = 1.0  # Reset retry delay
                
                # Process and call callback for each order
                for order in orders:
                    try:
                        standardized_order = self._standardize_order_response(order)
                        callback(standardized_order)
                    except Exception as callback_error:
                        self.logger.error(f"Error in callback processing: {str(callback_error)}")
                
            except Exception as e:
                error_count += 1
                self.logger.error(f"WebSocket error ({error_count}): {str(e)}")
                
                # If too many errors, reinitialize connection
                if error_count > 10:
                    self.logger.warning("Too many errors, reinitializing WebSocket connection")
                    try:
                        await exchange.close()
                        # Recreate exchange instance
                        exchange = ccxtpro.binance(self._build_params())
                        exchange.options['ws'] = {
                            'heartbeat': True,
                            'ping_interval': 30000,
                            'reconnect_rate': 5000,
                            'max_reconnects': 100,
                        }
                        error_count = 0
                    except Exception as reset_error:
                        self.logger.error(f"Error resetting WebSocket connection: {str(reset_error)}")
                
                # Use exponential backoff
                retry_delay = min(retry_delay * 2, max_retry_delay)
                jitter = random.uniform(0.5, 1.0)
                actual_delay = retry_delay * jitter
                self.logger.info(f"Waiting {actual_delay:.2f} seconds before retrying WebSocket connection")
                await asyncio.sleep(actual_delay)