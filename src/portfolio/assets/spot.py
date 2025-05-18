# src/portfolio/assets/spot.py (Completed Implementation)

from decimal import Decimal
import time
from typing import Dict, Any, Optional
import pandas as pd

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.abstract_factory import register_factory_class
from src.exchange.base import Exchange
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.assets.base import Asset
from src.portfolio.execution.order import Direction, OrderStatus


logger = LogManager.get_logger("portfolio.assets.spot")


@register_factory_class('asset_factory', 'spot')
class Spot(Asset):
    """
    Spot asset implementation for cryptocurrency/standard asset trading
    
    This is the basic asset type for simple buy/sell operations.
    """
    
    def __init__(self, name: str, 
                 exchange: Exchange = None, 
                 execution_engine: BaseExecutionEngine = None, 
                 config: Optional[ConfigManager] = None, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize spot asset
        
        Args:
            name: Asset symbol
            exchange: Exchange interface
            execution_engine: Execution engine
            config: Configuration manager
            params: Additional parameters
        """
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
        
        # For leveraged spot trading
        self.leverage = Decimal(str(params.get('leverage', 1)))
        self._position_type = 'long' if self._position_size > 0 else 'flat'
        
        self.logger = LogManager.get_logger(f"asset.spot.{name}")
        self.logger.info(f"Initialized {self.symbol} spot with {float(self.quantity)} units at ${float(self.price):.2f}")
    
    async def get_value(self) -> float:
        """
        Get current asset value
        
        Returns:
            Current asset value
        """
        return float(self._value)
    
    def get_position_type(self) -> str:
        """
        Get current position type
        
        Returns:
            Position type ('long' or 'flat' for normal spot, can be 'short' for leveraged spot)
        """
        return self._position_type
            
    async def open_long_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a long position (buy)
        
        Args:
            quantity: Amount to buy
            **kwargs: Additional parameters:
                price: Limit price (optional)
                leverage: Leverage to use (optional, for leveraged spot)
                
        Returns:
            Dict with order result
        """
        # Update leverage if specified and supported
        if 'leverage' in kwargs and hasattr(self, 'set_leverage') and callable(self.set_leverage):
            try:
                await self.set_leverage(float(kwargs['leverage']))
            except Exception as e:
                self.logger.error(f"Failed to set leverage to {kwargs['leverage']}: {e}")
        
        # Prepare buy parameters
        buy_params = {
            'symbol': self.symbol,
            'quantity': quantity,
            'direction': 'buy'
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                buy_params[param] = kwargs[param]
        
        # Execute buy
        result = await self.buy(buy_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Opened long position: {quantity} units of {self.symbol}")
        
        return result

    async def close_long_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a long position (sell)
        
        Args:
            quantity: Amount to sell (defaults to all if None)
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if we have a position to close
        if self._position_size <= 0:
            return {
                "success": False,
                "error": f"No position to close. Current position: {float(self._position_size)}",
                "direction": "sell",
                "quantity": quantity or 0
            }
        
        # If quantity not specified, close the entire position
        if quantity is None:
            quantity = float(self._position_size)
        elif quantity > float(self._position_size):
            # Limit quantity to current position size
            quantity = float(self._position_size)
        
        # Prepare sell parameters
        sell_params = {
            'symbol': self.symbol,
            'quantity': quantity,
            'direction': 'sell'
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                sell_params[param] = kwargs[param]
        
        # Execute sell
        result = await self.sell(sell_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Closed long position: {quantity} units of {self.symbol}")
        
        return result
    
    # For leveraged spot trading, which supports short positions
    
    async def open_short_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a short position (sell short) - leveraged spot only
        
        Args:
            quantity: Amount to sell short
            **kwargs: Additional parameters:
                price: Limit price (optional)
                leverage: Leverage to use (optional)
                
        Returns:
            Dict with order result
        """
        # Check if leveraged spot is supported
        if not hasattr(self, 'leverage') or self.leverage <= 1:
            return {
                "success": False,
                "error": "Short positions not supported for standard spot assets. Requires leveraged spot.",
                "direction": "sell",
                "quantity": quantity
            }
        
        # Update leverage if specified
        if 'leverage' in kwargs and hasattr(self, 'set_leverage') and callable(self.set_leverage):
            try:
                await self.set_leverage(float(kwargs['leverage']))
            except Exception as e:
                self.logger.error(f"Failed to set leverage to {kwargs['leverage']}: {e}")
        
        # Prepare sell parameters for short position
        sell_params = {
            'symbol': self.symbol,
            'quantity': quantity,
            'direction': 'sell',
            'is_short': True  # Flag to indicate short position
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                sell_params[param] = kwargs[param]
        
        # Execute sell
        result = await self.sell(sell_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Opened short position: {quantity} units of {self.symbol}")
        
        return result

    async def close_short_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a short position (buy to cover) - leveraged spot only
        
        Args:
            quantity: Amount to buy to cover (defaults to all if None)
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if leveraged spot is supported and we have a short position
        if not hasattr(self, 'leverage') or self.leverage <= 1 or self._position_type != 'short':
            return {
                "success": False,
                "error": "No short position to close",
                "direction": "buy",
                "quantity": quantity or 0
            }
        
        # If quantity not specified, close the entire position
        if quantity is None:
            quantity = float(abs(self._position_size))
        elif quantity > float(abs(self._position_size)):
            # Limit quantity to current position size
            quantity = float(abs(self._position_size))
        
        # Prepare buy parameters to close short
        buy_params = {
            'symbol': self.symbol,
            'quantity': quantity,
            'direction': 'buy',
            'close_short': True  # Flag to indicate closing short
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                buy_params[param] = kwargs[param]
        
        # Execute buy
        result = await self.buy(buy_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Closed short position: {quantity} units of {self.symbol}")
        
        return result
        
    def _update_position_from_filled_order(self, order):
        """
        Update position based on filled order
        
        Args:
            order: The filled order
        """
        if order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return
        
        filled_qty = Decimal(str(order.filled_quantity))
        avg_price = Decimal(str(order.avg_filled_price))
        
        # Handle special flags for leveraged spot
        is_short = getattr(order, 'is_short', False)
        close_short = getattr(order, 'close_short', False)
        
        if order.direction == Direction.BUY:
            if close_short and self._position_type == 'short':
                # Closing a short position
                if filled_qty >= abs(self._position_size):
                    # Close entire short position
                    realized_pnl = (self.price - avg_price) * abs(self._position_size)
                    self._position_size = Decimal('0')
                    self._position_type = 'flat'
                    self.logger.info(f"Closed entire short position with realized PNL: ${float(realized_pnl):.2f}")
                else:
                    # Partially close short position
                    self._position_size += filled_qty  # Reduces the negative position
                    self.logger.info(f"Reduced short position by {float(filled_qty)} units")
            else:
                # Standard buy - increasing long position
                self.quantity += filled_qty
                self._position_size = self.quantity
                self._position_type = 'long'
        else:  # SELL
            if is_short and (self._position_type == 'flat' or self._position_type == 'short'):
                # Opening or adding to short position (leveraged spot)
                if self._position_type == 'short':
                    # Adding to existing short
                    self._position_size -= filled_qty  # More negative position
                else:
                    # Opening new short
                    self._position_size = -filled_qty  # Negative position for short
                    self._position_type = 'short'
            else:
                # Standard sell - reducing long position
                self.quantity -= filled_qty
                self._position_size = self.quantity
                if self._position_size <= 0:
                    self._position_type = 'flat'
        
        # Update price if significant trade
        if filled_qty > self.quantity * Decimal('0.05'):
            self.price = avg_price
        
        # Update value
        self._value = self.quantity * self.price
        
        self.logger.info(f"Updated {self.symbol} position: {float(self.quantity)} @ ${float(self.price):.2f}")
        
        # Notify subscribers of position update
        self._notify_subscribers('position_updated', {
            'symbol': self.symbol,
            'position_type': self._position_type,
            'position_size': float(self._position_size),
            'current_price': float(self.price),
            'value': float(self._value)
        })

    def _update_position_from_trade(self, trade):
        """
        Update position from a trade dictionary
        
        Args:
            trade: Trade execution result
        """
        # Extract trade details
        direction = trade.get('direction', '').lower()
        quantity = Decimal(str(trade.get('quantity', 0.0)))
        price = Decimal(str(trade.get('price', 0.0)))
        
        # Update position based on direction
        if direction in ['buy', 'long']:
            self.quantity += quantity
        elif direction in ['sell', 'short']:
            self.quantity -= quantity
        
        # Update position tracking
        self._position_size = self.quantity
        
        # Update price if significant trade or no price set
        if quantity > self.quantity * Decimal('0.05') or self.price == 0:
            self.price = price
            self._last_price = price
        
        # Update value
        self._value = self.quantity * self.price
        
        logger.info(f"Updated {self.symbol} position from trade: {float(self.quantity)} @ ${float(self.price):.2f}")

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
                self._last_price = self.price
                
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
                    price = Decimal(str(ticker['last']))
                    self._last_price = price
                    self.price = price
                    self._value = self._position_size * price
                    return float(self._value)
            
            # If no exchange or failed to get ticker, use the current price
            self._value = self._position_size * self.price
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
            
    async def buy(self, kwargs) -> Dict[str, Any]:
        """Buy spot asset with validation"""
        amount = kwargs['quantity']
        if amount < self.min_quantity:
            return {"success": False, "error": f"Buy amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount < self.min_notional:
            return {"success": False, "error": f"Buy value ${float(self.price * amount)} below minimum ${float(self.min_notional)}"}
        
        result = await super().buy(kwargs)
        
        # Update local quantity for tracking without relying on execution engine
        if result.get('success', True):
            self.quantity += Decimal(str(amount))
            
        return result
    
    async def sell(self, kwargs) -> Dict[str, Any]:
        """Sell spot asset with validation"""
        amount = kwargs['quantity']
        if amount > self.quantity and not kwargs.get('allow_short', False):
            return {"success": False, "error": f"Insufficient {self.symbol} position: have {float(self.quantity)}, need {amount}"}
        
        if amount < self.min_quantity:
            return {"success": False, "error": f"Sell amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount < self.min_notional:
            return {"success": False, "error": f"Sell value ${float(self.price * amount)} below minimum ${float(self.min_notional)}"}
        
        result = await super().sell(kwargs)
        
        # Update local quantity for tracking without relying on execution engine
        if result.get('success', True):
            self.quantity -= Decimal(str(amount))
            
        return result
    
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
    
    async def open_long(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Open a long position (buy)
        
        Args:
            amount: Position size to open
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        buy_params = {'quantity': amount, 'symbol': self.symbol, **kwargs}
        return await self.buy(buy_params)
        
    async def close_long(self, amount: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a long position (sell)
        
        Args:
            amount: Position size to close (default: all)
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Operation result
        """
        if self.quantity <= 0:
            return {"success": False, "error": "No long position to close"}
            
        # Default to closing entire position
        if amount is None:
            amount = float(self.quantity)
        else:
            # Make sure we don't try to close more than we have
            amount = min(amount, float(self.quantity))
            
        sell_params = {'quantity': amount, 'symbol': self.symbol, **kwargs}
        return await self.sell(sell_params)