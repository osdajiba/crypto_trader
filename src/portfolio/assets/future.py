#!/usr/bin/env python3
# src/portfolio/assets/future.py

import time
import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional, Union

from src.exchange.base import retry_exchange_operation
from src.common.abstract_factory import register_factory_class
from src.common.log_manager import LogManager
from src.portfolio.execution.order import Order, Direction, OrderStatus
from src.portfolio.assets.base import Asset


logger = LogManager.get_logger("asset.future")


@register_factory_class('asset_factory', 'future')
class Future(Asset):
    """
    Future contract asset implementation for derivatives trading
    Integrates with CCXT exchange API futures markets and execution engine
    """
    
    def __init__(self, config, params):
        """
        Initialize future asset
        
        Args:
            config: Configuration manager
            params: Parameters including:
                name: Future contract symbol (e.g. 'BTC/USDT:USDT')
                contract_size: Contract size
                price: Current contract price
                leverage: Trading leverage (default 1 = no leverage)
                position_type: 'long' or 'short'
                exchange: Exchange interface (optional)
                execution_mode: Execution mode ('live', 'backtest', 'simple_backtest')
        """
        name = params.get('name', '')
        exchange = params.get('exchange', None)
        super().__init__(name, exchange, config, params)
        
        # Asset specific properties
        self.contract_size = Decimal(str(params.get('contract_size', 1)))
        self.price = Decimal(str(params.get('price', 0.0)))
        self.symbol = name  # Contract symbol
        self.leverage = Decimal(str(params.get('leverage', 1)))
        self.position_type = params.get('position_type', 'long')  # 'long' or 'short'
        
        # Trading parameters
        self.precision = params.get('precision', 8)
        self.min_notional = Decimal(str(params.get('min_notional', 10.0)))
        self.min_contracts = Decimal(str(params.get('min_contracts', 0.001)))
        self.maintenance_margin = Decimal(str(params.get('maintenance_margin', 0.05)))
        
        # Position tracking
        self._contracts = Decimal(str(params.get('contracts', 0)))
        self._position_size = self._contracts
        self._entry_price = Decimal(str(params.get('entry_price', 0.0)))
        self._liquidation_price = Decimal('0')
        self._unrealized_pnl = Decimal('0')
        self._value = self._contracts * self.contract_size * self.price
        self._last_update_time = time.time()
        
        # Calculate liquidation price
        self._calculate_liquidation_price()
        
        logger.info(f"Initialized {self.symbol} future contract with {float(self._contracts)} contracts "
                    f"at ${float(self.price):.2f}, leverage: {float(self.leverage)}x")

    def get_value(self) -> float:
        """
        Get the current notional value of the futures position
        
        Returns:
            float: Current position value (contracts * contract_size * price)
        """
        return float(self._value)
    
    def get_exposure(self) -> float:
        """
        Get the actual exposure considering leverage
        
        Returns:
            float: Position exposure
        """
        return float(self._value * self.leverage)
    
    def get_margin_required(self) -> float:
        """
        Get required margin for current position
        
        Returns:
            float: Required margin
        """
        return float(self._value / self.leverage)
    
    def _calculate_liquidation_price(self) -> None:
        """
        Calculate the liquidation price based on position and leverage
        """
        if self._contracts == 0 or self._entry_price == 0:
            self._liquidation_price = Decimal('0')
            return
            
        # Simplified liquidation price calculation
        if self.position_type == 'long':
            # For longs: entry_price * (1 - 1/leverage + maintenance_margin)
            self._liquidation_price = self._entry_price * (Decimal('1') - Decimal('1') / self.leverage + self.maintenance_margin)
        else:
            # For shorts: entry_price * (1 + 1/leverage - maintenance_margin)
            self._liquidation_price = self._entry_price * (Decimal('1') + Decimal('1') / self.leverage - self.maintenance_margin)
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def update_value(self) -> float:
        """
        Update the future contract value by fetching latest price from exchange
        
        Returns:
            float: Updated position value
        """
        if not self.exchange:
            logger.warning(f"No exchange available for {self.symbol}, using last known price")
            return float(self._value)
        
        try:
            # Make sure we're in futures mode
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                current_type = self.exchange.async_exchange.options['defaultType']
                self.exchange.async_exchange.options['defaultType'] = 'future'
            
            # Fetch latest price and position info
            ticker = await self.exchange.async_exchange.fetch_ticker(self.symbol)
            
            # Revert market type if changed
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                self.exchange.async_exchange.options['defaultType'] = current_type
            
            # Update price and value
            if ticker and 'last' in ticker and ticker['last']:
                old_price = self.price
                self.price = Decimal(str(ticker['last']))
                
                # If we have an open position, calculate unrealized PnL
                if self._contracts > 0 and self._entry_price > 0:
                    if self.position_type == 'long':
                        price_change = self.price - self._entry_price
                    else:  # short
                        price_change = self._entry_price - self.price
                        
                    self._unrealized_pnl = self._contracts * self.contract_size * price_change
                
                # Update position value
                self._value = self._contracts * self.contract_size * self.price
                self._last_update_time = time.time()
                
                price_change_pct = 0
                if old_price > 0:
                    price_change_pct = (self.price - old_price) * 100 / old_price
                
                logger.debug(f"Updated {self.symbol} price to ${float(self.price):.2f} "
                           f"({float(price_change_pct):.2f}%), value: ${float(self._value):.2f}")
                
            return float(self._value)
        except Exception as e:
            logger.error(f"Error updating {self.symbol} future value: {str(e)}")
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
        
        # Calculate fill price from the order result
        fill_price = avg_price if avg_price > 0 else self.price
        
        # Update position tracking
        if filled_qty > 0:
            if order.direction == Direction.BUY:
                # For futures, buying could mean opening long or closing short
                if self.position_type == 'short' and self._contracts > 0:
                    # Reducing a short position
                    if filled_qty >= self._contracts:
                        # Closed entire short position
                        self._contracts = Decimal('0')
                        self._entry_price = Decimal('0')
                        self.position_type = 'flat'
                    else:
                        # Partially reduced short position
                        self._contracts -= filled_qty
                else:
                    # Adding to long position
                    if self._contracts > 0 and self.position_type == 'long':
                        # Calculate average entry price
                        total_cost = (self._entry_price * self._contracts) + (fill_price * filled_qty)
                        self._contracts += filled_qty
                        self._entry_price = total_cost / self._contracts
                    else:
                        # Opening new long position
                        self._contracts = filled_qty
                        self._entry_price = fill_price
                        self.position_type = 'long'
            else:  # SELL
                # For futures, selling could mean opening short or closing long
                if self.position_type == 'long' and self._contracts > 0:
                    # Reducing a long position
                    if filled_qty >= self._contracts:
                        # Closed entire long position
                        self._contracts = Decimal('0')
                        self._entry_price = Decimal('0')
                        self.position_type = 'flat'
                    else:
                        # Partially reduced long position
                        self._contracts -= filled_qty
                else:
                    # Adding to short position
                    if self._contracts > 0 and self.position_type == 'short':
                        # Calculate average entry price
                        total_cost = (self._entry_price * self._contracts) + (fill_price * filled_qty)
                        self._contracts += filled_qty
                        self._entry_price = total_cost / self._contracts
                    else:
                        # Opening new short position
                        self._contracts = filled_qty
                        self._entry_price = fill_price
                        self.position_type = 'short'
        
        # Update position size and value
        self._position_size = self._contracts
        self._value = self._contracts * self.contract_size * self.price
        
        # Recalculate liquidation price if we have a position
        if self._contracts > 0:
            self._calculate_liquidation_price()
        
        logger.info(f"Updated {self.symbol} future position after order fill: {float(self._contracts)} contracts "
                   f"at ${float(self._entry_price):.2f}, {self.position_type}")

    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def set_leverage(self, leverage: float) -> Dict[str, Any]:
        """
        Set leverage for trading
        
        Args:
            leverage: Leverage value (e.g., 1, 3, 5, 10, etc.)
            
        Returns:
            Dict[str, Any]: Result of leverage change
        """
        if not self.exchange or not self.exchange.async_exchange:
            raise ValueError(f"No exchange available for {self.symbol}")
        
        try:
            # Make sure we're in futures mode
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                current_type = self.exchange.async_exchange.options['defaultType']
                self.exchange.async_exchange.options['defaultType'] = 'future'
            
            # Set leverage on exchange
            result = await self.exchange.async_exchange.set_leverage(leverage, self.symbol)
            
            # Revert market type if changed
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                self.exchange.async_exchange.options['defaultType'] = current_type
            
            # Update local leverage value
            self.leverage = Decimal(str(leverage))
            logger.info(f"Set leverage for {self.symbol} to {leverage}x")
            
            # Recalculate liquidation price
            self._calculate_liquidation_price()
            
            return result
        except Exception as e:
            logger.error(f"Error setting leverage for {self.symbol}: {str(e)}")
            raise
    
    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy futures contracts using the execution engine
        
        Args:
            amount: Number of contracts to buy
            **kwargs: Additional parameters:
                order_type: Order type ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                reduce_only: Whether to reduce existing position only
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Ensure sufficient minimum amounts
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_contracts:
            raise ValueError(f"Order size {float(amount_dec)} below minimum {float(self.min_contracts)} contracts")
        
        # For market orders, check estimated value
        if amount_dec * self.contract_size * self.price < self.min_notional:
            raise ValueError(f"Order value ${float(amount_dec * self.contract_size * self.price)} "
                           f"below minimum ${float(self.min_notional)}")
        
        # Check if trying to reduce more than position size
        reduce_only = kwargs.get('reduce_only', False)
        if reduce_only and self.position_type == 'short' and amount_dec > self._contracts:
            raise ValueError(f"Cannot reduce more than current position size: {float(self._contracts)} contracts")
        
        # Set market type for execution
        if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
            current_type = self.exchange.exchange.options.get('defaultType', 'spot')
            self.exchange.exchange.options['defaultType'] = 'future'
        
        # Add futures-specific parameters
        futures_params = {
            'reduce_only': reduce_only
        }
        kwargs.update(futures_params)
        
        try:
            # Execute through parent class implementation (which uses execution engine)
            result = await super().buy(amount, **kwargs)
            return result
        finally:
            # Revert market type
            if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
                self.exchange.exchange.options['defaultType'] = current_type
    
    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell futures contracts using the execution engine
        
        Args:
            amount: Number of contracts to sell
            **kwargs: Additional parameters:
                order_type: Order type ('market', 'limit', etc.)
                price: Limit price (for limit orders)
                reduce_only: Whether to reduce existing position only
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Ensure sufficient minimum amounts
        amount_dec = Decimal(str(amount))
        if amount_dec < self.min_contracts:
            raise ValueError(f"Order size {float(amount_dec)} below minimum {float(self.min_contracts)} contracts")
        
        # For market orders, check estimated value
        if amount_dec * self.contract_size * self.price < self.min_notional:
            raise ValueError(f"Order value ${float(amount_dec * self.contract_size * self.price)} "
                           f"below minimum ${float(self.min_notional)}")
        
        # Check if trying to reduce more than position size
        reduce_only = kwargs.get('reduce_only', False)
        if reduce_only and self.position_type == 'long' and amount_dec > self._contracts:
            raise ValueError(f"Cannot reduce more than current position size: {float(self._contracts)} contracts")
        
        # Set market type for execution
        if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
            current_type = self.exchange.exchange.options.get('defaultType', 'spot')
            self.exchange.exchange.options['defaultType'] = 'future'
        
        # Add futures-specific parameters
        futures_params = {
            'reduce_only': reduce_only
        }
        kwargs.update(futures_params)
        
        try:
            # Execute through parent class implementation (which uses execution engine)
            result = await super().sell(amount, **kwargs)
            return result
        finally:
            # Revert market type
            if self.exchange and hasattr(self.exchange, 'exchange') and hasattr(self.exchange.exchange, 'options'):
                self.exchange.exchange.options['defaultType'] = current_type
    
    @retry_exchange_operation(max_attempts=3, base_delay=1.0, max_delay=30.0)
    async def sync_position(self) -> Dict[str, Any]:
        """
        Sync futures position with exchange
        
        Returns:
            Dict[str, Any]: Position information
        """
        if not self.exchange or not self.exchange.async_exchange:
            logger.warning(f"No exchange available for {self.symbol}, cannot sync position")
            return {
                'symbol': self.symbol,
                'contracts': float(self._contracts),
                'entry_price': float(self._entry_price),
                'position_type': self.position_type
            }
        
        try:
            # Make sure we're in futures mode
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                current_type = self.exchange.async_exchange.options['defaultType']
                self.exchange.async_exchange.options['defaultType'] = 'future'
            
            # Fetch position information
            position = await self.exchange.async_exchange.fetch_position(self.symbol)
            
            # Revert market type if changed
            if hasattr(self.exchange.async_exchange, 'options') and 'defaultType' in self.exchange.async_exchange.options:
                self.exchange.async_exchange.options['defaultType'] = current_type
            
            if position:
                # Parse position info
                side = position.get('side', 'flat')
                contracts = Decimal(str(position.get('contracts', 0)))
                entry_price = Decimal(str(position.get('entryPrice', 0)))
                
                # Update local state
                if side == 'long':
                    self.position_type = 'long'
                    self._contracts = contracts
                    self._entry_price = entry_price
                elif side == 'short':
                    self.position_type = 'short'
                    self._contracts = contracts
                    self._entry_price = entry_price
                else:  # flat - no position
                    self.position_type = 'flat'
                    self._contracts = Decimal('0')
                    self._entry_price = Decimal('0')
                
                # Update position size and recalculate values
                self._position_size = self._contracts
                
                # Also update price and value
                await self.update_value()
                
                # Recalculate liquidation price
                self._calculate_liquidation_price()
                
                logger.info(f"Synced {self.symbol} position: {float(self._contracts)} contracts "
                           f"at ${float(self._entry_price):.2f}, {self.position_type}")
                
            return {
                'symbol': self.symbol,
                'contracts': float(self._contracts),
                'entry_price': float(self._entry_price),
                'current_price': float(self.price),
                'position_type': self.position_type,
                'liquidation_price': float(self._liquidation_price),
                'value': float(self._value),
                'leverage': float(self.leverage),
                'unrealized_pnl': float(self._unrealized_pnl)
            }
        except Exception as e:
            logger.error(f"Error syncing {self.symbol} position: {str(e)}")
            return {
                'symbol': self.symbol,
                'error': str(e),
                'contracts': float(self._contracts),
                'position_type': self.position_type
            }