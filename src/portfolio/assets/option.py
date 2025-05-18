import asyncio
from decimal import Decimal
import time
from typing import Dict, Any, Optional, List, Union
import pandas as pd
from datetime import datetime

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.abstract_factory import register_factory_class
from src.exchange.base import Exchange
from src.portfolio.execution.base import BaseExecutionEngine
from src.portfolio.assets.base import Asset
from src.portfolio.execution.order import Direction, Order, OrderStatus


@register_factory_class('asset_factory', 'option')
class Option(Asset):
    """
    Option contract asset with full trading capabilities
    
    Supports both long and short positions, option pricing, and position management.
    """
    
    PRICING_METHODS = ['Black-Scholes', 'Monte-Carlo', 'Binomial-Tree', 'BAW', 'Heston']
    
    def __init__(self, name: str, 
                 exchange: Exchange = None, 
                 execution_engine: BaseExecutionEngine = None, 
                 config: Optional[ConfigManager] = None, 
                 params: Optional[Dict[str, Any]] = None):
        """
        Initialize option asset
        
        Args:
            name: Option contract symbol
            exchange: Exchange interface
            execution_engine: Execution engine
            config: Configuration manager
            params: Additional parameters
        """
        params = params or {}
        # Ensure option assets are tradable
        params['tradable'] = True
        super().__init__(name, exchange, execution_engine, config, params)
        
        # Basic option properties
        self.option_type = params.get('option_type', 'call')  # 'call' or 'put'
        self.option_price = Decimal(str(params.get('option_price', 0.0)))
        self.underlying_price = Decimal(str(params.get('underlying_price', 0.0)))
        self.strike_price = Decimal(str(params.get('strike_price', 0.0)))
        self.risk_free_rate = Decimal(str(params.get('risk_free_rate', 0.0)))
        
        # Expiration details
        self.end_date = params.get('end_date')
        if isinstance(self.end_date, str):
            self.end_date = datetime.fromisoformat(self.end_date)
        
        self.tau = params.get('tau')  # Time to expiration in years
        if self.tau is None:
            time_to_maturity = params.get('time_to_maturity')
            if time_to_maturity:
                self.tau = time_to_maturity / 365.0
            elif self.end_date:
                self.tau = (self.end_date - datetime.now()).days / 365.0
            else:
                self.tau = 0.0
        
        # Trading parameters
        self.contract_size = Decimal(str(params.get('contract_size', 100)))  # Typically 100 shares per contract
        self.min_contracts = Decimal(str(params.get('min_contracts', 1)))    # Minimum contracts to trade
        self.premium = Decimal(str(params.get('premium', 0.0)))              # Premium paid/received
        
        # Position tracking
        self._position_size = Decimal(str(params.get('position_size', 0)))   # Number of contracts
        self._position_type = params.get('position_type', 'flat')            # 'long', 'short', or 'flat'
        self._entry_price = Decimal(str(params.get('entry_price', 0.0)))     # Average entry price
        self._last_price = self.option_price                                 # Last known price
        
        # Volatility
        self.volatility = params.get('volatility', 0.2)  # Historical/expected volatility
        self.implied_volatility = params.get('implied_volatility')  # Implied volatility from market
        
        # For pricing
        self.pricing_method = params.get('pricing_method', 'Black-Scholes')
        
        # Value calculation
        self._value = self._position_size * self.contract_size * self.option_price
        self._last_update_time = time.time()
        
        self.logger = LogManager.get_logger(f"asset.option.{name}")
        self.logger.info(f"Initialized {self.name} option ({self.option_type}) with {float(self._position_size)} contracts, "
                        f"Strike: ${float(self.strike_price)}, Expiry: {self.end_date}")
    
    async def _initialize_asset(self):
        """Asset-specific initialization"""
        # Calculate implied volatility if not provided
        if self.implied_volatility is None and self.option_price > 0:
            try:
                self.implied_volatility = self._calculate_implied_volatility()
                self.logger.info(f"Calculated implied volatility: {self.implied_volatility:.4f}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate implied volatility: {e}")
                self.implied_volatility = self.volatility

    async def get_value(self) -> float:
        """
        Calculate the current value of the option position
        
        Returns:
            Current position value
        """
        # Calculate current option price
        current_price = self.calculate_price()
        
        # Calculate position value based on direction
        if self._position_type == 'long':
            return float(self._position_size * self.contract_size * Decimal(str(current_price)))
        elif self._position_type == 'short':
            # For short positions, value is negative of long position
            return float(-self._position_size * self.contract_size * Decimal(str(current_price)))
        else:
            return 0.0
        
    def calculate_price(self) -> float:
        """Calculate option price using the specified pricing method"""
        if self.pricing_method == "Black-Scholes":
            return self._calculate_black_scholes_price()
        elif self.pricing_method == "Monte-Carlo":
            return self._calculate_monte_carlo_price()
        else:
            return self._calculate_black_scholes_price()
        
    def get_position_type(self) -> str:
        """
        Get current position type
        
        Returns:
            Position type ('long', 'short', 'flat')
        """
        return self._position_type
        
    async def open_long_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a long position in the option (buy to open)
        
        Args:
            quantity: Number of contracts to buy
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if we have a short position that this would reduce
        reduce_only = False
        if self._position_type == 'short' and self._position_size > 0:
            reduce_only = True
            self.logger.info(f"Opening long will reduce existing short position")
        
        # Prepare buy parameters
        buy_params = {
            'symbol': self.name,
            'quantity': quantity,
            'direction': 'buy',
            'reduce_only': reduce_only
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                buy_params[param] = kwargs[param]
        
        # Execute buy
        result = await self.buy(buy_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Opened long position: {quantity} contracts of {self.name}")
        
        return result

    async def close_long_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a long position in the option (sell to close)
        
        Args:
            quantity: Number of contracts to sell (defaults to all if None)
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if we have a long position to close
        if self._position_type != 'long' or self._position_size <= 0:
            return {
                "success": False,
                "error": f"No long position to close. Current position: {self._position_type} with {float(self._position_size)} contracts",
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
            'symbol': self.name,
            'quantity': quantity,
            'direction': 'sell',
            'reduce_only': True  # Always reduce_only when closing
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                sell_params[param] = kwargs[param]
        
        # Execute sell
        result = await self.sell(sell_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Closed long position: {quantity} contracts of {self.name}")
        
        return result

    async def open_short_position(self, quantity: float, **kwargs) -> Dict[str, Any]:
        """
        Open a short position in the option (sell to open)
        
        Args:
            quantity: Number of contracts to sell short
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if we have a long position that this would reduce
        reduce_only = False
        if self._position_type == 'long' and self._position_size > 0:
            reduce_only = True
            self.logger.info(f"Opening short will reduce existing long position")
        
        # Prepare sell parameters
        sell_params = {
            'symbol': self.name,
            'quantity': quantity,
            'direction': 'sell',
            'reduce_only': reduce_only
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                sell_params[param] = kwargs[param]
        
        # Execute sell
        result = await self.sell(sell_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Opened short position: {quantity} contracts of {self.name}")
        
        return result

    async def close_short_position(self, quantity: float = None, **kwargs) -> Dict[str, Any]:
        """
        Close a short position in the option (buy to close)
        
        Args:
            quantity: Number of contracts to buy to cover (defaults to all if None)
            **kwargs: Additional parameters:
                price: Limit price (optional)
                
        Returns:
            Dict with order result
        """
        # Check if we have a short position to close
        if self._position_type != 'short' or self._position_size <= 0:
            return {
                "success": False,
                "error": f"No short position to close. Current position: {self._position_type} with {float(self._position_size)} contracts",
                "direction": "buy",
                "quantity": quantity or 0
            }
        
        # If quantity not specified, close the entire position
        if quantity is None:
            quantity = float(self._position_size)
        elif quantity > float(self._position_size):
            # Limit quantity to current position size
            quantity = float(self._position_size)
        
        # Prepare buy parameters
        buy_params = {
            'symbol': self.name,
            'quantity': quantity,
            'direction': 'buy',
            'reduce_only': True  # Always reduce_only when closing
        }
        
        # Add optional parameters
        for param in ['price', 'order_type']:
            if param in kwargs:
                buy_params[param] = kwargs[param]
        
        # Execute buy
        result = await self.buy(buy_params)
        
        # Log position update if successful
        if result.get('success', False):
            self.logger.info(f"Closed short position: {quantity} contracts of {self.name}")
        
        return result
        
    def _update_position_from_filled_order(self, order):
        """
        Update position based on a filled order
        
        Args:
            order: The filled order
        """
        if order.status not in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            return
        
        filled_qty = Decimal(str(order.filled_quantity))
        avg_price = Decimal(str(order.avg_filled_price))
        
        # Calculate fill price from the order result
        fill_price = avg_price if avg_price > 0 else self._last_price
        
        # Update position tracking
        if filled_qty > 0:
            if order.direction == Direction.BUY:
                if self._position_type == 'short' and self._position_size > 0:
                    # Reducing a short position (buying to close)
                    if filled_qty >= self._position_size:
                        # Closed entire short position
                        realized_pnl = (self._entry_price - fill_price) * self._position_size * self.contract_size
                        self._position_size = Decimal('0')
                        self._entry_price = Decimal('0')
                        self._position_type = 'flat'
                        self.logger.info(f"Closed entire short position with realized PNL: ${float(realized_pnl):.2f}")
                    else:
                        # Partially reduced short position
                        realized_pnl = (self._entry_price - fill_price) * filled_qty * self.contract_size
                        self._position_size -= filled_qty
                        self.logger.info(f"Reduced short position by {float(filled_qty)} contracts with realized PNL: ${float(realized_pnl):.2f}")
                else:
                    # Adding to or opening a long position
                    if self._position_size > 0 and self._position_type == 'long':
                        # Calculate average entry price for additional contracts
                        total_cost = (self._entry_price * self._position_size) + (fill_price * filled_qty)
                        self._position_size += filled_qty
                        self._entry_price = total_cost / self._position_size
                        self.logger.info(f"Added {float(filled_qty)} contracts to long position at ${float(fill_price):.4f}")
                    else:
                        # Opening new long position
                        self._position_size = filled_qty
                        self._entry_price = fill_price
                        self._position_type = 'long'
                        self.logger.info(f"Opened new long position: {float(filled_qty)} contracts at ${float(fill_price):.4f}")
            else:  # SELL
                if self._position_type == 'long' and self._position_size > 0:
                    # Reducing a long position (selling to close)
                    if filled_qty >= self._position_size:
                        # Closed entire long position
                        realized_pnl = (fill_price - self._entry_price) * self._position_size * self.contract_size
                        self._position_size = Decimal('0')
                        self._entry_price = Decimal('0')
                        self._position_type = 'flat'
                        self.logger.info(f"Closed entire long position with realized PNL: ${float(realized_pnl):.2f}")
                    else:
                        # Partially reduced long position
                        realized_pnl = (fill_price - self._entry_price) * filled_qty * self.contract_size
                        self._position_size -= filled_qty
                        self.logger.info(f"Reduced long position by {float(filled_qty)} contracts with realized PNL: ${float(realized_pnl):.2f}")
                else:
                    # Adding to or opening a short position
                    if self._position_size > 0 and self._position_type == 'short':
                        # Calculate average entry price for additional contracts
                        total_cost = (self._entry_price * self._position_size) + (fill_price * filled_qty)
                        self._position_size += filled_qty
                        self._entry_price = total_cost / self._position_size
                        self.logger.info(f"Added {float(filled_qty)} contracts to short position at ${float(fill_price):.4f}")
                    else:
                        # Opening new short position
                        self._position_size = filled_qty
                        self._entry_price = fill_price
                        self._position_type = 'short'
                        self.logger.info(f"Opened new short position: {float(filled_qty)} contracts at ${float(fill_price):.4f}")
        
        # Update value based on position
        if self._position_type == 'long':
            self._value = self._position_size * self.contract_size * self._last_price
        elif self._position_type == 'short':
            self._value = -self._position_size * self.contract_size * self._last_price
        else:
            self._value = Decimal('0')
        
        # Notify subscribers of position update
        self._notify_subscribers('position_updated', {
            'symbol': self.name,
            'position_type': self._position_type,
            'position_size': float(self._position_size),
            'entry_price': float(self._entry_price),
            'current_price': float(self._last_price),
            'value': float(self._value)
        })
        
    def get_exposure(self) -> float:
        """
        Get the maximum potential exposure
        
        Returns:
            Maximum potential exposure
        """
        # For long call options, exposure is limited to premium paid
        if self.option_type == 'call' and self._position_type == 'long':
            return float(self._position_size * self.contract_size * self._entry_price)
        
        # For short call options, exposure is theoretically unlimited
        elif self.option_type == 'call' and self._position_type == 'short':
            return float('inf')  # Unlimited risk
        
        # For long put options, exposure is limited to premium paid
        elif self.option_type == 'put' and self._position_type == 'long':
            return float(self._position_size * self.contract_size * self._entry_price)
        
        # For short put options, exposure is limited to strike price
        elif self.option_type == 'put' and self._position_type == 'short':
            return float(self._position_size * self.contract_size * self.strike_price)
        
        return 0.0

    async def update_data(self, data: pd.DataFrame) -> None:
        """
        Update option with market data
        
        Args:
            data: DataFrame containing market data
        """
        if data.empty:
            return
            
        try:
            # Get latest price from data
            if 'close' in data.columns and len(data) > 0:
                last_row = data.iloc[-1]
                old_price = self._last_price
                
                # Update price (option premium)
                self._last_price = Decimal(str(last_row['close']))
                
                # If underlying price data available, update that too
                if 'underlying_price' in data.columns:
                    self.underlying_price = Decimal(str(last_row['underlying_price']))
                
                # Update position value
                old_value = self._value
                
                if self._position_type == 'long':
                    self._value = self._position_size * self.contract_size * self._last_price
                elif self._position_type == 'short':
                    self._value = -self._position_size * self.contract_size * self._last_price
                else:
                    self._value = Decimal('0')
                
                self._last_update_time = time.time()
                
                # Calculate price change
                price_change_pct = 0
                if old_price > 0:
                    price_change_pct = (self._last_price - old_price) * 100 / old_price
                
                # Calculate value change
                value_change_pct = 0
                if old_value > 0:
                    value_change_pct = (self._value - old_value) * 100 / old_value
                
                # Log significant changes
                if abs(price_change_pct) > 1.0:  # Only log significant changes
                    self.logger.info(f"Updated {self.name} price: ${float(self._last_price):.4f} "
                                f"({float(price_change_pct):.2f}%), value: ${float(self._value):.2f}")
                
                # Notify subscribers of value changes
                self._notify_subscribers('value_changed', {
                    'symbol': self.name,
                    'old_price': float(old_price),
                    'new_price': float(self._last_price),
                    'change_pct': float(price_change_pct),
                    'position_size': float(self._position_size),
                    'position_type': self._position_type,
                    'value': float(self._value),
                    'value_change_pct': float(value_change_pct)
                })
                
                # Update implied volatility based on new market data
                if self._last_price > 0:
                    try:
                        self.implied_volatility = self._calculate_implied_volatility()
                    except Exception as e:
                        self.logger.debug(f"Failed to update implied volatility: {e}")
                
        except Exception as e:
            self.logger.error(f"Error updating {self.name} with market data: {str(e)}")

    async def buy(self, kwargs) -> Dict[str, Any]:
        """
        Buy option contracts
        
        Args:
            kwargs: Order parameters:
                quantity: Number of contracts to buy
                price: Limit price (optional)
                order_type: Order type (market, limit)
                reduce_only: Whether to reduce existing position only
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Ensure minimum contract requirements
        quantity = kwargs['quantity']
        if quantity < float(self.min_contracts):
            return {
                "success": False,
                "error": f"Order size {quantity} below minimum {float(self.min_contracts)} contracts",
                "direction": "buy",
                "quantity": quantity
            }
        
        # Use the parent class implementation
        return await super().buy(kwargs)

    async def sell(self, kwargs) -> Dict[str, Any]:
        """
        Sell option contracts
        
        Args:
            kwargs: Order parameters:
                quantity: Number of contracts to sell
                price: Limit price (optional)
                order_type: Order type (market, limit)
                reduce_only: Whether to reduce existing position only
                
        Returns:
            Dict[str, Any]: Order result
        """
        # Ensure minimum contract requirements
        quantity = kwargs['quantity']
        if quantity < float(self.min_contracts):
            return {
                "success": False,
                "error": f"Order size {quantity} below minimum {float(self.min_contracts)} contracts",
                "direction": "sell",
                "quantity": quantity
            }
        
        # If reduce_only is set, validate we have a position to reduce
        if kwargs.get('reduce_only', False):
            if self._position_type == 'long' and quantity > float(self._position_size):
                return {
                    "success": False,
                    "error": f"Reduce-only sell order size {quantity} exceeds long position size {float(self._position_size)}",
                    "direction": "sell",
                    "quantity": quantity
                }
        
        # Use the parent class implementation
        return await super().sell(kwargs)

    async def update_value(self) -> float:
        """
        Update the option value
        
        Returns:
            Updated asset value
        """
        try:
            if self.exchange:
                # First try to get current price from exchange
                ticker = await self.exchange.fetch_ticker(self.name)
                if ticker and 'last' in ticker:
                    self._last_price = Decimal(str(ticker['last']))
                    
                    # If we also have underlying ticker info, update that too
                    if 'info' in ticker and 'underlyingPrice' in ticker['info']:
                        self.underlying_price = Decimal(str(ticker['info']['underlyingPrice']))
                    
                    # Update implied volatility if price changed
                    try:
                        self.implied_volatility = self._calculate_implied_volatility()
                    except Exception as e:
                        self.logger.debug(f"Failed to update implied volatility: {e}")
            
            # Calculate option value
            if self._position_type == 'long':
                self._value = self._position_size * self.contract_size * self._last_price
            elif self._position_type == 'short':
                self._value = -self._position_size * self.contract_size * self._last_price
            else:
                self._value = Decimal('0')
                
            # Update time to expiry
            if self.end_date:
                self.tau = (self.end_date - datetime.now()).days / 365.0
                # Check if expired
                if self.tau <= 0:
                    self.tau = 0.0
                    self._last_price = Decimal('0') if self.is_out_of_the_money() else Decimal(str(self.intrinsic_value()))
                    self.logger.warning(f"Option {self.name} has expired")
            
            return float(self._value)
        
        except Exception as e:
            self.logger.error(f"Error updating {self.name} value: {str(e)}")
            return float(self._value)

    def _calculate_implied_volatility(self, precision: float = 0.0001, max_iterations: int = 100) -> float:
        """
        Calculate implied volatility using the Newton-Raphson method
        
        Args:
            precision: Desired precision level
            max_iterations: Maximum iterations
            
        Returns:
            Calculated implied volatility
        """
        from math import exp, log, sqrt
        from scipy.stats import norm
        
        # Use current option price
        market_price = float(self._last_price)
        if market_price <= 0:
            return self.volatility  # Default to historical volatility
            
        # Initial volatility guess
        vol = self.volatility or 0.2
        
        # Newton-Raphson iterations
        for i in range(max_iterations):
            # Calculate option price with current volatility estimate
            price = self._calculate_black_scholes_price(vol)
            
            # Calculate vega (derivative of price with respect to volatility)
            S = float(self.underlying_price)
            K = float(self.strike_price)
            T = float(self.tau)
            r = float(self.risk_free_rate)
            
            d1 = (log(S / K) + (r + 0.5 * vol**2) * T) / (vol * sqrt(T))
            vega = S * norm.pdf(d1) * sqrt(T)
            
            # Price difference
            diff = price - market_price
            
            # Check for convergence
            if abs(diff) < precision:
                return vol
                
            # Update volatility estimate
            if abs(vega) < 1e-6:  # Avoid division by near-zero
                break
                
            vol = vol - diff / vega
            
            # Ensure volatility is positive
            if vol <= 0:
                vol = 0.01
                
        # If no convergence, return best estimate
        return vol

    def _calculate_black_scholes_price(self, volatility: float = None) -> float:
        """
        Calculate option price using Black-Scholes model
        
        Args:
            volatility: Volatility to use (uses implied_volatility or volatility if None)
            
        Returns:
            Option price
        """
        from math import exp, log, sqrt
        from scipy.stats import norm
        
        # Use provided volatility or default to implied or historical
        sigma = volatility if volatility is not None else (
            self.implied_volatility if self.implied_volatility is not None else self.volatility
        )
        
        # Extract parameters
        S = float(self.underlying_price)
        K = float(self.strike_price)
        T = float(self.tau)
        r = float(self.risk_free_rate)
        
        # Handle expired options
        if T <= 0:
            if self.option_type == "call":
                return max(0, S - K)
            else:  # put
                return max(0, K - S)
                
        # Calculate d1 and d2
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # Calculate call or put price
        if self.option_type == "call":
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        return price

    def _calculate_monte_carlo_price(self, num_simulations: int = 10000) -> float:
        """
        Calculate option price using Monte Carlo simulation
        
        Args:
            num_simulations: Number of price path simulations
            
        Returns:
            Option price
        """
        import numpy as np
        from math import exp
        
        # Use implied or historical volatility
        sigma = self.implied_volatility if self.implied_volatility is not None else self.volatility
        
        # Extract parameters
        S = float(self.underlying_price)
        K = float(self.strike_price)
        T = float(self.tau)
        r = float(self.risk_free_rate)
        
        # Handle expired options
        if T <= 0:
            if self.option_type == "call":
                return max(0, S - K)
            else:  # put
                return max(0, K - S)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate random price paths
        z = np.random.normal(0, 1, num_simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        
        # Calculate payoffs
        if self.option_type == "call":
            payoffs = np.maximum(ST - K, 0)
        else:  # put
            payoffs = np.maximum(K - ST, 0)
            
        # Discount payoffs to present value
        option_price = exp(-r * T) * np.mean(payoffs)
        
        return float(option_price)

    def is_in_the_money(self) -> bool:
        """
        Check if option is in-the-money
        
        Returns:
            True if in-the-money, False otherwise
        """
        if self.option_type == "call":
            return self.underlying_price > self.strike_price
        else:  # put
            return self.underlying_price < self.strike_price

    def is_out_of_the_money(self) -> bool:
        """
        Check if option is out-of-the-money
        
        Returns:
            True if out-of-the-money, False otherwise
        """
        return not self.is_in_the_money()

    def intrinsic_value(self) -> float:
        """
        Calculate intrinsic value of the option
        
        Returns:
            Intrinsic value
        """
        if self.option_type == "call":
            return max(0, float(self.underlying_price - self.strike_price))
        else:  # put
            return max(0, float(self.strike_price - self.underlying_price))

    def time_value(self) -> float:
        """
        Calculate time value of the option
        
        Returns:
            Time value
        """
        return float(self._last_price) - self.intrinsic_value()

    def calculate_greeks(self) -> Dict[str, float]:
        """
        Calculate option greeks (delta, gamma, vega, theta, rho)
        
        Returns:
            Dictionary of greek values
        """
        from math import exp, log, sqrt
        from scipy.stats import norm
        
        # Use implied or historical volatility
        sigma = self.implied_volatility if self.implied_volatility is not None else self.volatility
        
        # Extract parameters
        S = float(self.underlying_price)
        K = float(self.strike_price)
        T = float(self.tau)
        r = float(self.risk_free_rate)
        
        # Handle expired options
        if T <= 0:
            return {
                'delta': 1.0 if self.is_in_the_money() else 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0
            }
        
        # Calculate d1 and d2
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        # Calculate greeks
        if self.option_type == "call":
            delta = norm.cdf(d1)
            theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(d2)
        else:  # put
            delta = norm.cdf(d1) - 1
            theta = -(S * norm.pdf(d1) * sigma) / (2 * sqrt(T)) + r * K * exp(-r * T) * norm.cdf(-d2)
            
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        vega = S * norm.pdf(d1) * sqrt(T) / 100  # Divided by 100 to get per 1% change
        
        if self.option_type == "call":
            rho = K * T * exp(-r * T) * norm.cdf(d2) / 100  # Divided by 100 to get per 1% change
        else:  # put
            rho = -K * T * exp(-r * T) * norm.cdf(-d2) / 100  # Divided by 100 to get per 1% change
            
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

    def get_position_info(self) -> Dict[str, Any]:
        """
        Get detailed position information
        
        Returns:
            Dictionary with position details
        """
        # Calculate greeks
        greeks = self.calculate_greeks()
        
        # Get intrinsic and time values
        intrinsic = self.intrinsic_value()
        time_val = self.time_value()
        
        # Days to expiry
        dte = int(self.tau * 365) if self.tau > 0 else 0
        
        return {
            'symbol': self.name,
            'option_type': self.option_type,
            'position_type': self._position_type,
            'position_size': float(self._position_size),
            'entry_price': float(self._entry_price),
            'current_price': float(self._last_price),
            'underlying_price': float(self.underlying_price),
            'strike_price': float(self.strike_price),
            'value': float(self._value),
            'expiry_date': self.end_date.isoformat() if self.end_date else None,
            'days_to_expiry': dte,
            'in_the_money': self.is_in_the_money(),
            'intrinsic_value': intrinsic,
            'time_value': time_val,
            'implied_volatility': self.implied_volatility,
            'greeks': greeks
        }