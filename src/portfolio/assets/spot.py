#!/usr/bin/env python3
# src/portfolio/assets/spot.py

import time
import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, Union

from src.exchange.adapters.base import retry_exchange_operation
from src.common.abstract_factory import register_factory_class
from src.portfolio.execution.order import Order, Direction, OrderStatus
from src.portfolio.assets.tradable_asset import TradableAsset, logger


@register_factory_class('asset_factory', 'spot')
class Spot(TradableAsset):
    """
    Spot asset implementation for cryptocurrency trading (stocks, currencies, etc.)
    Integrates with CCXT exchange API and execution engine
    """
    
    def __init__(self, config, params):
        """
        Initialize spot asset
        
        Args:
            config: Configuration manager
            params: Parameters including:
                name: Asset symbol (e.g. 'BTC/USDT')
                quantity: Initial quantity (optional)
                price: Initial price (optional)
                exchange: Exchange interface (optional)
                execution_mode: Execution mode ('live', 'backtest', 'simple_backtest')
        """
        name = params.get('name', '')
        exchange = params.get('exchange', None)
        super().__init__(name, exchange, config, params)
        
        # Asset specific properties
        self.quantity = Decimal(str(params.get('quantity', 0.0)))
        self.price = Decimal(str(params.get('price', 0.0)))
        self.symbol = name  # For spot assets, name is usually the trading symbol
        
        # Trading parameters
        self.precision = params.get('precision', 8)
        self.min_notional = Decimal(str(params.get('min_notional', 10.0)))  # Min order value
        self.min_quantity = Decimal(str(params.get('min_quantity', 0.0001)))  # Min order quantity
        
        # Position tracking
        self._position_size = self.quantity
        self._value = self.quantity * self.price
        self._last_update_time = time.time()
        
        logger.info(f"Initialized {self.symbol} spot asset with {float(self.quantity)} units at ${float(self.price):.2f}")

    def get_value(self) -> float:
        """
        Get the current value of the spot position
        
        Returns:
            float: Current position value (quantity * price)
        """
        return float(self._value)
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def update_value(self) -> float:
        """
        Update the asset value by fetching latest price from exchange
        
        Returns:
            float: Updated asset value
        """
        if not self.exchange:
            logger.warning(f"No exchange available for {self.symbol}, using last known price")
            return float(self._value)
        
        try:
            # Fetch ticker for latest price
            ticker = await self.exchange.async_exchange.fetch_ticker(self.symbol)
            
            # Update price and calculated value
            if ticker and 'last' in ticker and ticker['last']:
                self.price = Decimal(str(ticker['last']))
                self._value = self.quantity * self.price
                self._last_update_time = time.time()
                
                logger.debug(f"Updated {self.symbol} price to ${float(self.price):.2f}, " 
                           f"value: ${float(self._value):.2f}")
                
            return float(self._value)
        except Exception as e:
            logger.error(f"Error updating {self.symbol} value: {str(e)}")
            # Re-raise to allow retry decorator to work
            raise
    
    def _update_position_from_filled_order(self, order: Order):
        """
        Update position based on a filled order
        
        Args:
            order: Filled order
        """
        if order.status != OrderStatus.FILLED and order.status != OrderStatus.PARTIAL:
            return
        
        filled_qty = Decimal(str(order.filled_quantity))
        avg_price = Decimal(str(order.avg_filled_price))
        
        if order.direction == Direction.BUY:
            # Buying increases quantity
            self.quantity += filled_qty
        else:  # SELL
            # Selling decreases quantity
            self.quantity -= filled_qty
        
        # Update position size
        self._position_size = self.quantity
        
        # Update price if it's a significant trade
        if filled_qty > self.quantity * Decimal('0.05'):  # More than 5% of position
            self.price = avg_price
        
        # Update value
        self._value = self.quantity * self.price
        
        logger.info(f"Updated {self.symbol} position after order fill: {float(self.quantity)} @ ${float(self.price):.2f}")
    
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy spot asset using the execution engine
        
        Args:
            amount: Amount to buy (quantity)
            **kwargs: Additional parameters:
                order_type: Order type ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Ensure sufficient minimum amounts
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_quantity:
            raise ValueError(f"Buy amount {amount} below minimum {float(self.min_quantity)}")
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            raise ValueError(f"Buy value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}")
        
        # Execute through parent class implementation (which uses execution engine)
        return await super().buy(amount, **kwargs)
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell spot asset using the execution engine
        
        Args:
            amount: Amount to sell (quantity)
            **kwargs: Additional parameters:
                order_type: Order type ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Check available quantity
        amount_dec = Decimal(str(amount))
        if amount_dec > self.quantity:
            raise ValueError(f"Insufficient {self.symbol} balance: have {float(self.quantity)}, need {amount}")
        
        if amount_dec < self.min_quantity:
            raise ValueError(f"Sell amount {amount} below minimum {float(self.min_quantity)}")
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            raise ValueError(f"Sell value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}")
        
        # Execute through parent class implementation (which uses execution engine)
        return await super().sell(amount, **kwargs)
            
    async def sync_balance(self) -> Dict[str, Any]:
        """
        Sync asset balance with exchange
        
        Returns:
            Dict[str, Any]: Balance information
        """
        if not self.exchange or not self.exchange.async_exchange:
            logger.warning(f"No exchange available for {self.symbol}, cannot sync balance")
            return {'symbol': self.symbol, 'quantity': float(self.quantity)}
        
        try:
            # Extract base currency from symbol (e.g., BTC from BTC/USDT)
            base_currency = self.symbol.split('/')[0]
            
            # Fetch balances from exchange
            await self.exchange._init_async_exchange()
            balance = await self.exchange.async_exchange.fetch_balance()
            
            if balance and base_currency in balance and 'free' in balance[base_currency]:
                # Update local quantity from exchange balance
                self.quantity = Decimal(str(balance[base_currency]['free']))
                self._position_size = self.quantity
                
                # Also update price and value
                await self.update_value()
                
                logger.info(f"Synced {self.symbol} balance: {float(self.quantity)} units")
                
            return {
                'symbol': self.symbol,
                'quantity': float(self.quantity),
                'position_size': float(self._position_size),
                'price': float(self.price),
                'value': float(self._value)
            }
        except Exception as e:
            logger.error(f"Error syncing {self.symbol} balance: {str(e)}")
            return {'symbol': self.symbol, 'error': str(e)}