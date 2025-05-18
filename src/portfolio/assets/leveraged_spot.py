#!/usr/bin/env python3
# src/portfolio/assets/leveraged_spot.py

from decimal import Decimal
import time
from typing import Dict, Any, Optional
import pandas as pd

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.abstract_factory import register_factory_class
from src.exchange.base import Exchange, retry_exchange_operation
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.assets.base import Asset
from src.portfolio.execution.order import Direction, OrderStatus

logger = LogManager.get_logger("portfolio.assets.leveraged_spot")

@register_factory_class('asset_factory', 'leveraged_spot')
class LeveragedSpot(Asset):
    """Leveraged spot asset implementation for margin trading on Binance"""
    
    def __init__(self, name: str, 
                 exchange: Exchange = None, 
                 execution_engine: BaseExecutionEngine = None, 
                 config: Optional[ConfigManager] = None, 
                 params: Optional[Dict[str, Any]] = None):
        params = params or {}
        # Ensure leveraged spot assets are tradable
        params['tradable'] = True
        super().__init__(name, exchange, execution_engine, config, params)
        
        # Spot specific properties
        self.quantity = Decimal(str(params.get('quantity', 0.0)))
        self.price = Decimal(str(params.get('price', 0.0)))
        self.symbol = name
        
        # Leveraged specific properties
        self.leverage = Decimal(str(params.get('leverage', 1.0)))  # Default to 1x (no leverage)
        self.borrowed_amount = Decimal(str(params.get('borrowed_amount', 0.0)))
        self.position_type = params.get('position_type', 'long')  # 'long' or 'short'
        self.entry_price = Decimal(str(params.get('entry_price', 0.0)))
        self.maintenance_margin = Decimal(str(params.get('maintenance_margin', 0.05)))  # 5% default
        self.liquidation_price = Decimal('0')
        self.unrealized_pnl = Decimal('0')
        self.interest_rate = Decimal(str(params.get('interest_rate', 0.0001)))  # Daily interest rate (0.01% default)
        self.last_interest_update = time.time()
        
        # Trading parameters
        self.precision = params.get('precision', 8)
        self.min_notional = Decimal(str(params.get('min_notional', 10.0)))
        self.min_quantity = Decimal(str(params.get('min_quantity', 0.0001)))
        
        # Position tracking
        self._position_size = self.quantity
        self._value = self.quantity * self.price
        
        # Calculate liquidation price
        self._calculate_liquidation_price()
        
        logger.info(f"Initialized {self.symbol} leveraged spot with {float(self.quantity)} units at ${float(self.price):.2f}, leverage: {float(self.leverage)}x")

    def _calculate_liquidation_price(self) -> None:
        """
        Calculate the liquidation price based on position and leverage
        """
        if self.quantity == 0 or self.entry_price == 0:
            self.liquidation_price = Decimal('0')
            return
            
        if self.position_type == 'long':
            # For longs: entry_price * (1 - 1/leverage + maintenance_margin)
            self.liquidation_price = self.entry_price * (Decimal('1') - Decimal('1') / self.leverage + self.maintenance_margin)
        else:
            # For shorts: entry_price * (1 + 1/leverage - maintenance_margin)
            self.liquidation_price = self.entry_price * (Decimal('1') + Decimal('1') / self.leverage - self.maintenance_margin)
            
        self.logger.debug(f"Calculated liquidation price for {self.symbol}: ${float(self.liquidation_price):.2f}")

    def _update_position_from_filled_order(self, order):
        """Update position based on filled order"""
        if order.status != OrderStatus.FILLED and order.status != OrderStatus.PARTIAL:
            return
        
        filled_qty = Decimal(str(order.filled_quantity))
        avg_price = Decimal(str(order.avg_filled_price))
        
        # Calculate fill price from the order result
        fill_price = avg_price if avg_price > 0 else self.price
        
        # Handle position update based on direction
        if order.direction == Direction.BUY:
            if self.position_type == 'short' and self.quantity > 0:
                # Reducing a short position
                if filled_qty >= self.quantity:
                    # Closed entire short position
                    self.borrowed_amount = Decimal('0')
                    self.quantity = Decimal('0')
                    self.entry_price = Decimal('0')
                    self.position_type = 'flat'
                else:
                    # Partially reduced short position
                    # Reduce borrowed amount proportionally
                    reduction_ratio = filled_qty / self.quantity
                    self.borrowed_amount -= self.borrowed_amount * reduction_ratio
                    self.quantity -= filled_qty
            else:
                # Adding to long position or opening new long
                old_value = self.quantity * self.entry_price if self.entry_price > 0 else Decimal('0')
                new_value = filled_qty * fill_price
                
                # Calculate new entry price as weighted average
                total_qty = self.quantity + filled_qty
                if total_qty > 0:
                    self.entry_price = (old_value + new_value) / total_qty
                
                self.quantity += filled_qty
                self.position_type = 'long'
                
                # Update borrowed amount based on leverage
                if self.leverage > 1:
                    self.borrowed_amount = self.quantity * self.entry_price * (self.leverage - 1) / self.leverage
        else:  # SELL
            if self.position_type == 'long' and self.quantity > 0:
                # Reducing a long position
                if filled_qty >= self.quantity:
                    # Closed entire long position
                    self.borrowed_amount = Decimal('0')
                    self.quantity = Decimal('0')
                    self.entry_price = Decimal('0')
                    self.position_type = 'flat'
                else:
                    # Partially reduced long position
                    # Reduce borrowed amount proportionally
                    reduction_ratio = filled_qty / self.quantity
                    self.borrowed_amount -= self.borrowed_amount * reduction_ratio
                    self.quantity -= filled_qty
            else:
                # Opening or adding to short position
                old_value = self.quantity * self.entry_price if self.entry_price > 0 else Decimal('0')
                new_value = filled_qty * fill_price
                
                # Calculate new entry price as weighted average
                total_qty = self.quantity + filled_qty
                if total_qty > 0:
                    self.entry_price = (old_value + new_value) / total_qty
                
                self.quantity += filled_qty
                self.position_type = 'short'
                
                # Update borrowed amount based on leverage
                if self.leverage > 1:
                    self.borrowed_amount = self.quantity * self.entry_price * (self.leverage - 1) / self.leverage
        
        # Update position tracking
        self._position_size = self.quantity
        
        # Update position value
        self._value = self.quantity * self.price
        
        # Recalculate liquidation price
        self._calculate_liquidation_price()
        
        # Log the update
        position_info = f"{float(self.quantity)} {self.symbol} ({self.position_type}"
        if self.leverage > 1:
            position_info += f", {float(self.leverage)}x leverage"
        position_info += f") at ${float(self.entry_price):.2f}"
        
        self.logger.info(f"Updated position: {position_info}")

    async def update_data(self, data: pd.DataFrame) -> None:
        """
        Update leveraged spot asset with market data
        
        Args:
            data: DataFrame containing market data
        """
        if data.empty:
            return
            
        try:
            # Get latest price from data
            if 'close' in data.columns and len(data) > 0:
                last_row = data.iloc[-1]
                old_price = self.price
                
                # Update price
                self.price = Decimal(str(last_row['close']))
                
                # Update position value
                self._value = self.quantity * self.price
                self._last_update_time = time.time()
                
                # Update unrealized PnL
                self._update_unrealized_pnl()
                
                # Process interest accrual
                self._process_interest()
                
                # Check for liquidation
                if self._check_liquidation():
                    self.logger.warning(f"Position liquidated: {self.symbol} price {float(self.price)} crossed liquidation price {float(self.liquidation_price)}")
                    
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
                    'position_type': self.position_type,
                    'leverage': float(self.leverage),
                    'value': float(self._value),
                    'liquidation_price': float(self.liquidation_price)
                })
        
        except Exception as e:
            self.logger.error(f"Error updating {self.symbol} with market data: {str(e)}")

    def _update_unrealized_pnl(self) -> None:
        """Calculate and update unrealized PnL"""
        if self.quantity == 0 or self.entry_price == 0:
            self.unrealized_pnl = Decimal('0')
            return
            
        if self.position_type == 'long':
            # For long: current_value - entry_value
            self.unrealized_pnl = self.quantity * (self.price - self.entry_price)
        else:  # short
            # For short: entry_value - current_value
            self.unrealized_pnl = self.quantity * (self.entry_price - self.price)
            
        # Log significant PnL changes
        if abs(self.unrealized_pnl) > 0:
            pnl_pct = self.unrealized_pnl * 100 / (self.quantity * self.entry_price) if self.entry_price > 0 else Decimal('0')
            if abs(pnl_pct) > 1:  # Only log changes > 1%
                self.logger.info(f"Unrealized PnL for {self.symbol}: ${float(self.unrealized_pnl):.2f} ({float(pnl_pct):.2f}%)")

    def _process_interest(self) -> None:
        """Calculate and apply interest on borrowed funds"""
        if self.borrowed_amount <= 0:
            return
            
        current_time = time.time()
        time_diff = current_time - self.last_interest_update
        
        # Only process interest once per hour to avoid excessive calculations
        if time_diff < 3600:  # 1 hour in seconds
            return
            
        # Calculate interest for the time period (convert daily rate to hourly)
        hourly_rate = self.interest_rate / 24
        hours_elapsed = time_diff / 3600
        
        interest_amount = self.borrowed_amount * hourly_rate * Decimal(str(hours_elapsed))
        
        # Add interest to borrowed amount
        self.borrowed_amount += interest_amount
        
        # Update last interest time
        self.last_interest_update = current_time
        
        # Recalculate liquidation price with new borrowed amount
        self._calculate_liquidation_price()
        
        self.logger.debug(f"Applied ${float(interest_amount):.4f} interest on borrowed amount for {self.symbol}")

    def _check_liquidation(self) -> bool:
        """
        Check if position should be liquidated based on current price
        
        Returns:
            bool: True if liquidated, False otherwise
        """
        if self.quantity == 0 or self.liquidation_price == 0:
            return False
            
        if self.position_type == 'long' and self.price <= self.liquidation_price:
            # Liquidate long position
            self._liquidate_position()
            return True
        elif self.position_type == 'short' and self.price >= self.liquidation_price:
            # Liquidate short position
            self._liquidate_position()
            return True
            
        return False

    def _liquidate_position(self) -> None:
        """Liquidate the current position"""
        # Reset position to zero
        self.quantity = Decimal('0')
        self.borrowed_amount = Decimal('0')
        self.entry_price = Decimal('0')
        self.position_type = 'flat'
        self._position_size = Decimal('0')
        self._value = Decimal('0')
        self.liquidation_price = Decimal('0')
        
        # Notify subscribers
        self._notify_subscribers('position_liquidated', {
            'symbol': self.symbol,
            'price': float(self.price),
            'liquidation_price': float(self.liquidation_price)
        })
        
        self.logger.warning(f"Position liquidated for {self.symbol} at ${float(self.price):.2f}")

    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def set_leverage(self, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for leveraged spot trading
        
        Args:
            leverage: Leverage value (e.g., 1, 2, 3, 5, etc.)
            
        Returns:
            Dict[str, Any]: Result of leverage change
        """
        if not self.exchange or not self.exchange.async_exchange:
            raise ValueError(f"No exchange available for {self.symbol}")
        
        try:
            # Make sure we're in margin mode
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                current_type = self.exchange.async_exchange.options['defaultType']
                self.exchange.async_exchange.options['defaultType'] = 'margin'
            
            # Set leverage with exchange if supported
            result = {}
            if hasattr(self.exchange.async_exchange, 'set_leverage'):
                result = await self.exchange.async_exchange.set_leverage(leverage, self.symbol)
            
            # Revert market type if changed
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                self.exchange.async_exchange.options['defaultType'] = current_type
            
            # Update local leverage value
            old_leverage = self.leverage
            self.leverage = Decimal(str(leverage))
            
            # Update borrowed amount if we have a position
            if self.quantity > 0 and self.entry_price > 0:
                # Recalculate borrowed amount based on new leverage
                self.borrowed_amount = self.quantity * self.entry_price * (self.leverage - 1) / self.leverage
                
                # Recalculate liquidation price
                self._calculate_liquidation_price()
            
            self.logger.info(f"Set leverage for {self.symbol} from {float(old_leverage)}x to {float(self.leverage)}x")
            
            return result
        except Exception as e:
            self.logger.error(f"Error setting leverage for {self.symbol}: {str(e)}")
            raise

    async def buy(self, kwargs) -> Dict[str, Any]:
        """
        Buy leveraged spot asset
        
        Args:
            amount: Amount to buy
            **kwargs: Additional parameters for the buy operation
            
        Returns:
            Dict[str, Any]: Buy operation result
        """
        # Ensure minimum amounts
        amount = kwargs['quantity']
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_quantity:
            return {"success": False, "error": f"Buy amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            return {"success": False, "error": f"Buy value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}"}
        
        # Set market type for execution
        if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
            current_type = self.exchange.exchange.options.get('defaultType', 'spot')
            self.exchange.exchange.options['defaultType'] = 'margin'
        
        # Set leverage if provided
        leverage = kwargs.get('leverage', None)
        if leverage is not None and float(leverage) != float(self.leverage):
            try:
                await self.set_leverage(float(leverage))
            except Exception as e:
                self.logger.error(f"Failed to set leverage: {e}")
                # Continue with current leverage
        
        try:
            # For short positions, handle differently
            if self.position_type == 'short' and self.quantity > 0:
                # Buying to close a short position
                kwargs['reduce_only'] = True
            
            # Execute through parent class implementation
            result = await super().buy(kwargs)
            return result
        finally:
            # Revert market type
            if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
                self.exchange.exchange.options['defaultType'] = current_type

    async def sell(self, kwargs) -> Dict[str, Any]:
        """
        Sell leveraged spot asset
        
        Args:
            amount: Amount to sell
            **kwargs: Additional parameters for the sell operation
            
        Returns:
            Dict[str, Any]: Sell operation result
        """
        amount = kwargs['quantity']
        # For long positions, validate available quantity
        if self.position_type == 'long' and self.quantity < amount:
            return {"success": False, "error": f"Insufficient {self.symbol} position: have {float(self.quantity)}, need {amount}"}
        
        # Ensure minimum amounts
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_quantity:
            return {"success": False, "error": f"Sell amount {amount} below minimum {float(self.min_quantity)}"}
        
        # For market orders, check estimated value
        order_type = kwargs.get('order_type', 'market')
        if order_type == 'market' and self.price * amount_dec < self.min_notional:
            return {"success": False, "error": f"Sell value ${float(self.price * amount_dec)} below minimum ${float(self.min_notional)}"}
        
        # Set market type for execution
        if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
            current_type = self.exchange.exchange.options.get('defaultType', 'spot')
            self.exchange.exchange.options['defaultType'] = 'margin'
        
        # Set leverage if provided
        leverage = kwargs.get('leverage', None)
        if leverage is not None and float(leverage) != float(self.leverage):
            try:
                await self.set_leverage(float(leverage))
            except Exception as e:
                self.logger.error(f"Failed to set leverage: {e}")
                # Continue with current leverage
        
        try:
            # For long positions, handle differently
            if self.position_type == 'long' and self.quantity > 0:
                # Selling to close a long position
                kwargs['reduce_only'] = True
            
            # Execute through parent class implementation
            result = await super().sell(kwargs)
            return result
        finally:
            # Revert market type
            if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
                self.exchange.exchange.options['defaultType'] = current_type

    async def sync_balance(self) -> Dict[str, Any]:
        """Sync leveraged asset balance with exchange"""
        if not self.exchange or not hasattr(self.exchange, 'async_exchange'):
            return {'symbol': self.symbol, 'quantity': float(self.quantity)}
        
        try:
            # Make sure we're in margin mode
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                current_type = self.exchange.async_exchange.options['defaultType']
                self.exchange.async_exchange.options['defaultType'] = 'margin'
            
            # Extract base currency from symbol
            base_currency = self.symbol.split('/')[0]
            
            # Fetch margin balances from exchange
            if hasattr(self.exchange, '_init_async_exchange'):
                await self.exchange._init_async_exchange()
                
            # Different exchanges have different methods for fetching margin balances
            margin_balance = None
            
            if hasattr(self.exchange.async_exchange, 'fetch_margin_balance'):
                balance = await self.exchange.async_exchange.fetch_margin_balance()
                if balance and base_currency in balance:
                    margin_balance = balance[base_currency]
            elif hasattr(self.exchange.async_exchange, 'fetch_balance'):
                # Try with options set to margin
                balance = await self.exchange.async_exchange.fetch_balance({'type': 'margin'})
                if balance and base_currency in balance:
                    margin_balance = balance[base_currency]
            
            # Revert market type if changed
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                self.exchange.async_exchange.options['defaultType'] = current_type
            
            # Update local state if we got balance info
            if margin_balance:
                # Get net position
                free = Decimal(str(margin_balance.get('free', 0)))
                used = Decimal(str(margin_balance.get('used', 0)))
                total = Decimal(str(margin_balance.get('total', 0)))
                debt = Decimal(str(margin_balance.get('debt', 0)))
                
                # Update position based on net balance
                self.quantity = total - debt if total > debt else Decimal('0')
                self.borrowed_amount = debt
                
                # Update position type based on net position
                if self.borrowed_amount > 0:
                    if self.quantity > 0:
                        self.position_type = 'long'
                    else:
                        self.position_type = 'short'
                else:
                    self.position_type = 'flat' if self.quantity == 0 else 'long'
                
                # Update position tracking
                self._position_size = self.quantity
                
                # Update price and value
                await self.update_value()
                
                # Recalculate liquidation price
                self._calculate_liquidation_price()
                
                self.logger.info(f"Synced {self.symbol} margin balance: {float(self.quantity)} units, "
                                f"borrowed: {float(self.borrowed_amount)}")
                
            return {
                'symbol': self.symbol,
                'quantity': float(self.quantity),
                'position_type': self.position_type,
                'borrowed_amount': float(self.borrowed_amount),
                'leverage': float(self.leverage),
                'entry_price': float(self.entry_price),
                'liquidation_price': float(self.liquidation_price),
                'price': float(self.price),
                'value': float(self._value)
            }
        except Exception as e:
            self.logger.error(f"Error syncing {self.symbol} margin balance: {str(e)}")
            return {'symbol': self.symbol, 'error': str(e)}

    async def update_value(self) -> float:
        """Update the asset's value by fetching current market data"""
        try:
            if self.exchange:
                ticker = await self.exchange.fetch_ticker(self.symbol)
                if ticker and 'last' in ticker:
                    old_price = self.price
                    self.price = Decimal(str(ticker['last']))
                    
                    # Update position value
                    self._value = self.quantity * self.price
                    self._last_update_time = time.time()
                    
                    # Update unrealized PnL
                    self._update_unrealized_pnl()
                    
                    # Process interest accrual
                    self._process_interest()
                    
                    # Check for liquidation
                    self._check_liquidation()
                    
                    # Log significant price changes
                    price_change_pct = 0
                    if old_price > 0:
                        price_change_pct = (self.price - old_price) * 100 / old_price
                    
                    if abs(price_change_pct) > 0.1:
                        self.logger.info(f"Updated {self.symbol} price to ${float(self.price):.2f} "
                                    f"({float(price_change_pct):.2f}%)")
                    
                    return float(self._value)
            
            # If no exchange or failed to get ticker, use current value
            return float(self._value)
        except Exception as e:
            self.logger.error(f"Error updating {self.symbol} value: {str(e)}")
            return float(self._value)
            
    def get_margin_level(self) -> float:
        """
        Calculate current margin level as a percentage
        
        Returns:
            float: Margin level percentage
        """
        if self.borrowed_amount <= 0:
            return float('inf')  # No borrowed funds, infinite margin
            
        # Calculate margin level: (position value / borrowed amount) * 100
        margin_level = (self.quantity * self.price / self.borrowed_amount) * 100
        return float(margin_level)