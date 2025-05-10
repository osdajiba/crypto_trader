#!/usr/bin/env python3
# src/portfolio/assets/spot.py

from decimal import Decimal
import time
from typing import Dict, Any, Optional
import pandas as pd

from common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.abstract_factory import register_factory_class
from exchange.base import Exchange
from portfolio.execution.base import BaseExecutionEngine
from src.portfolio.assets.base import Asset
from src.portfolio.execution.order import Direction, OrderStatus

logger = LogManager.get_logger("portfolio.assets.spot")

@register_factory_class('asset_factory', 'spot')
class Spot(Asset):
    """Spot asset implementation for cryptocurrency trading"""
    
    def __init__(self, name: str, 
                 exchange: Exchange = None, 
                 execution_engine: BaseExecutionEngine = None, 
                 config: Optional[ConfigManager] = None, 
                 params: Optional[Dict[str, Any]] = None):
        params = params or {}
        # Ensure spot assets are tradable
        params['tradable'] = True
        super().__init__(name, exchange, execution_engine, config, params)
        
        # Spot specific properties
        self.quantity = Decimal(str(params.get('quantity', 0.0)))
        self.price = Decimal(str(params.get('price', 0.0)))
        self.symbol = name
        
        # Trading parameters
        self.precision = params.get('precision', 8)
        self.min_notional = Decimal(str(params.get('min_notional', 10.0)))
        self.min_quantity = Decimal(str(params.get('min_quantity', 0.0001)))
        
        # Position tracking
        self._position_size = self.quantity
        self._value = self.quantity * self.price
        
        logger.info(f"Initialized {self.symbol} spot with {float(self.quantity)} units at ${float(self.price):.2f}")

    def _update_position_from_filled_order(self, order):
        """Update position based on filled order"""
        if order.status != OrderStatus.FILLED and order.status != OrderStatus.PARTIAL:
            return
        
        filled_qty = Decimal(str(order.filled_quantity))
        avg_price = Decimal(str(order.avg_filled_price))
        
        if order.direction == Direction.BUY:
            self.quantity += filled_qty
        else:  # SELL
            self.quantity -= filled_qty
        
        # Update position tracking
        self._position_size = self.quantity
        
        # Update price if significant trade
        if filled_qty > self.quantity * Decimal('0.05'):
            self.price = avg_price
        
        # Update value
        self._value = self.quantity * self.price
        
        logger.info(f"Updated {self.symbol} position: {float(self.quantity)} @ ${float(self.price):.2f}")

    async def update_data(self, data: pd.DataFrame) -> None:
        """
        Update spot asset with market data
        
        Args:
            data: DataFrame containing market data
        """
        if data.empty:
            return
            
        try:
            # Get latest price and update asset value
            if 'close' in data.columns and len(data) > 0:
                last_row = data.iloc[-1]
                old_price = self.price
                
                # Update price
                self.price = Decimal(str(last_row['close']))
                
                # Update position value
                self._value = self.quantity * self.price
                self._last_update_time = time.time()
                
                # Calculate and log price change if significant
                price_change_pct = 0
                if old_price > 0:
                    price_change_pct = (self.price - old_price) * 100 / old_price
                    
                if abs(price_change_pct) > 0.1:  # Only log significant changes
                    self.logger.info(f"Updated {self.symbol} price: ${float(self.price):.2f} " 
                                f"({float(price_change_pct):.2f}%), value: ${float(self._value):.2f}")
                else:
                    self.logger.debug(f"Updated {self.symbol} price: ${float(self.price):.2f}, value: ${float(self._value):.2f}")
                
                # Notify subscribers of value changes
                self._notify_subscribers('value_changed', {
                    'symbol': self.symbol,
                    'old_price': float(old_price),
                    'new_price': float(self.price),
                    'change_pct': float(price_change_pct),
                    'position_size': float(self.quantity),
                    'value': float(self._value)
                })
        
        except Exception as e:
            self.logger.error(f"Error updating {self.symbol} with market data: {str(e)}")
            
    async def update_value(self) -> float:
        """Update the asset's value by fetching current market data"""
        try:
            if self.exchange:
                ticker = await self.exchange.fetch_ticker(self.symbol)
                if ticker and 'last' in ticker:
                    self._last_price = Decimal(str(ticker['last']))
                    self._value = self._position_size * self._last_price
                    return float(self._value)
            
            # If no exchange or failed to get ticker, try to use provided last price
            return float(self._value)
        except Exception as e:
            # More informative error message
            error_msg = str(e)
            if "does not have market symbol" in error_msg:
                self.logger.warning(f"Symbol {self.symbol} not available on exchange, using last known price")
                # Still return the current value based on last known price
                return float(self._value)
            else:
                self.logger.error(f"Error updating {self.symbol} value: {e}")
                # Return current value even on error
                return float(self._value)
            
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """Buy spot asset with validation"""
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_quantity:
            return {"success": False, "error": f"Buy amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            return {"success": False, "error": f"Buy value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}"}
        
        return await super().buy(amount, **kwargs)
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """Sell spot asset with validation"""
        amount_dec = Decimal(str(amount))
        if amount_dec > self.quantity:
            return {"success": False, "error": f"Insufficient {self.symbol} balance: have {float(self.quantity)}, need {amount}"}
        
        if amount_dec < self.min_quantity:
            return {"success": False, "error": f"Sell amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            return {"success": False, "error": f"Sell value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}"}
        
        return await super().sell(amount, **kwargs)
    
    async def sync_balance(self) -> Dict[str, Any]:
        """Sync asset balance with exchange"""
        if not self.exchange or not hasattr(self.exchange, 'async_exchange'):
            return {'symbol': self.symbol, 'quantity': float(self.quantity)}
        
        try:
            # Extract base currency from symbol
            base_currency = self.symbol.split('/')[0]
            
            # Fetch balances from exchange
            if hasattr(self.exchange, '_init_async_exchange'):
                await self.exchange._init_async_exchange()
                
            balance = await self.exchange.async_exchange.fetch_balance()
            
            if balance and base_currency in balance and 'free' in balance[base_currency]:
                self.quantity = Decimal(str(balance[base_currency]['free']))
                self._position_size = self.quantity
                
                # Update price and value
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