#!/usr/bin/env python3
# src/portfolio/risk/backtest.py

"""
Improved backtest risk manager implementation.
Provides enhanced risk management functionality for backtesting.
"""

import time
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Any, Optional

from src.common.abstract_factory import register_factory_class
from src.portfolio.risk.base import BaseRiskManager


@register_factory_class('risk_manager_factory', 'backtest', 
    description="Enhanced risk management for backtesting",
    features=["position_limits", "basic_validation", "dynamic_sizing"],
    category="risk")
class BacktestRiskManager(BaseRiskManager):
    """
    Backtest risk manager implementation
    
    This is a simplified risk manager optimized for backtesting with reduced
    computational overhead while still providing essential risk controls.
    """
    
    async def _initialize_specific(self) -> None:
        """Initialize backtest-specific settings"""
        # Load more permissive risk limits specific to backtest mode
        backtest_limits = {
            'max_drawdown': self.config.get("backtest", "risk", "max_drawdown", default=0.3),
            'max_position_size': self.config.get("backtest", "risk", "max_position", default=0.25),
            'max_leverage': self.config.get("backtest", "risk", "max_leverage", default=10.0),
            'max_open_positions': self.config.get("backtest", "risk", "max_positions", default=20),
            'position_limit_pct': self.config.get("backtest", "risk", "position_limit", default=0.1),
            'trade_size_limit': self.config.get("backtest", "risk", "trade_size", default=0.05),
            'risk_per_trade': self.config.get("backtest", "risk", "risk_per_trade", default=0.02),
            'signal_threshold': self.config.get("backtest", "risk", "signal_threshold", default=0.005)
        }
        
        # Update limits with backtest-specific values
        self._risk_limits.update(backtest_limits)
        
        # Activate relevant controls for backtest
        self.activate_control('position_limits')
        self.activate_control('dynamic_sizing')
        
        # No need for temporary value tracking in backtest mode
        self._last_check_time = 0
        
        # Track consecutive losses for adaptive sizing
        self._consecutive_losses = 0
        self._max_consecutive_losses = self.config.get("backtest", "risk", "max_consecutive_losses", default=5)
        
        # Initialize portfolio tracking if portfolio is available
        if self._portfolio:
            portfolio_value = self._portfolio.get_total_value()
            self._peak_value = Decimal(str(portfolio_value))
            self._last_portfolio_value = Decimal(str(portfolio_value))
            self.logger.info(f"BacktestRiskManager initialized with portfolio value: {portfolio_value}")
        else:
            self.logger.info("BacktestRiskManager initialized without portfolio")
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate trading signals with enhanced logic for backtesting
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            pd.DataFrame: Filtered signals that pass risk checks
        """
        # Fast path for empty signals
        if signals.empty:
            return signals
        
        # Skip validation if no portfolio manager
        if not self._portfolio:
            return signals
            
        # Skip full validation if risk already breached
        if self._risk_breached:
            return signals[signals['action'].str.lower() == 'sell'].copy()
        
        try:
            # Get portfolio value
            portfolio_value = Decimal(str(self._portfolio.get_total_value()))
            
            # Handle edge cases for portfolio value
            if portfolio_value <= 0:
                self.logger.warning("Portfolio value is zero or negative, using minimal position sizing")
                portfolio_value = Decimal('1000.0')  # Use a minimal default value
            
            # Update peak value if needed
            if portfolio_value > self._peak_value:
                self._peak_value = portfolio_value
                
            # Calculate current drawdown
            current_drawdown = Decimal('0')
            if self._peak_value > 0:
                current_drawdown = (self._peak_value - portfolio_value) / self._peak_value
            
            # Check if drawdown limit is breached
            max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.3)))
            if current_drawdown > max_drawdown:
                self._risk_breached = True
                self.logger.info(f"Max drawdown limit breached in backtest: {float(current_drawdown*100):.2f}% > {float(max_drawdown*100):.2f}%")
                return signals[signals['action'].str.lower() == 'sell'].copy()
                
            # Update consecutive losses tracking for dynamic sizing
            if portfolio_value < self._last_portfolio_value:
                self._consecutive_losses += 1
                if self._consecutive_losses >= self._max_consecutive_losses:
                    self.logger.warning(f"Max consecutive losses reached ({self._max_consecutive_losses}), reducing position sizes")
                    signals = self._reduce_position_sizes(signals)
            else:
                self._consecutive_losses = 0
                
            # Update last portfolio value
            self._last_portfolio_value = portfolio_value
            
            # Apply signal threshold filtering
            signals = self._apply_signal_threshold(signals)
            
            # Make a copy to avoid warnings
            valid_signals = signals.copy()
            
            # Calculate position limits for all assets
            total_exposure = Decimal('0')
            position_count = 0
            
            # Get current positions and exposure
            for asset_name, asset in self._portfolio.assets.items():
                if hasattr(asset, 'get_value') and callable(asset.get_value):
                    asset_value = Decimal(str(asset.get_value()))
                    if asset_value > 0:
                        position_count += 1
                        # Add leveraged exposure for futures
                        if hasattr(asset, 'get_exposure') and callable(asset.get_exposure):
                            total_exposure += Decimal(str(asset.get_exposure()))
                        else:
                            total_exposure += asset_value
                            
            # Check if we've exceeded max positions
            max_open_positions = int(self.get_risk_limit('max_open_positions', 20))
            
            # Check total exposure vs limits
            max_total_exposure = portfolio_value * Decimal(str(self.get_risk_limit('max_leverage', 2.0)))
            
            # List of indices to remove
            indices_to_drop = []
            
            # Process signals to apply position sizing and limits
            for idx, signal in valid_signals.iterrows():
                action = signal['action'].lower()
                symbol = signal['symbol']
                current_position_size = 0
                
                # Check if this is an existing asset
                asset = self._portfolio.assets.get(symbol)
                if asset:
                    current_position_size = asset.get_position_size() if hasattr(asset, 'get_position_size') else 0
                
                # Process buy signals
                if action == 'buy':
                    # Check if we've hit position count limit
                    if position_count >= max_open_positions and current_position_size == 0:
                        self.logger.info(f"Skipping new position for {symbol}, already at max positions ({position_count}/{max_open_positions})")
                        indices_to_drop.append(idx)
                        continue
                    
                    # Apply position size limits
                    if 'quantity' in signal:
                        # Get asset price
                        price = signal.get('price', None)
                        if price is None and asset and hasattr(asset, 'price'):
                            price = asset.price
                        elif price is None:
                            self.logger.warning(f"No price available for {symbol}, using nominal value")
                            price = Decimal('1.0')
                        else:
                            price = Decimal(str(price))
                            
                        # Calculate order value
                        quantity = Decimal(str(signal['quantity']))
                        order_value = quantity * price
                        
                        # Calculate trade size limit
                        trade_size_limit = portfolio_value * Decimal(str(self.get_risk_limit('trade_size_limit', 0.05)))
                        
                        # Apply limit
                        if order_value > trade_size_limit:
                            # Calculate new quantity
                            new_quantity = float(trade_size_limit / price)
                            valid_signals.at[idx, 'quantity'] = new_quantity
                            self.logger.info(f"Adjusted position size for {symbol} from {float(quantity)} to {new_quantity}")
                
                # Process sell signals - ensure not selling more than we have
                elif action == 'sell' and asset and 'quantity' in signal:
                    quantity = Decimal(str(signal['quantity']))
                    if quantity > current_position_size and current_position_size > 0:
                        valid_signals.at[idx, 'quantity'] = float(current_position_size)
                        self.logger.info(f"Adjusted sell quantity for {symbol} from {float(quantity)} to {float(current_position_size)}")
            
            # Remove invalid signals
            if indices_to_drop:
                valid_signals = valid_signals.drop(indices_to_drop)
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error in signal validation: {e}")
            return signals  # Return original signals on error
    
    def _apply_signal_threshold(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply signal threshold to filter weak signals
        
        Args:
            signals: Trading signals to filter
            
        Returns:
            pd.DataFrame: Filtered signals
        """
        if signals.empty or 'ma_spread' not in signals.columns:
            return signals
            
        # Get signal threshold
        threshold = float(self.get_risk_limit('signal_threshold', 0.005))
        
        # Apply threshold
        return signals[signals['ma_spread'].abs() > threshold]
    
    def _reduce_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Reduce position sizes due to consecutive losses
        
        Args:
            signals: Trading signals to adjust
            
        Returns:
            pd.DataFrame: Adjusted signals
        """
        if signals.empty or 'quantity' not in signals.columns:
            return signals
            
        # Reduce by 50%
        signals_copy = signals.copy()
        signals_copy.loc[signals_copy['action'].str.lower() == 'buy', 'quantity'] *= 0.5
        return signals_copy
    
    async def execute_risk_control(self) -> bool:
        """
        Execute simplified risk control for backtesting
        
        Returns:
            bool: True if trading should continue, False if trading should stop
        """
        # Fast path if no portfolio
        if not self._portfolio:
            return True
            
        try:
            # In backtesting, we only check basic limits periodically
            current_time = time.time()
            
            # Check only every 5 seconds in simulation time to reduce overhead
            if (current_time - self._last_check_time) < 5:
                return True
                
            self._last_check_time = current_time
            
            # Calculate current drawdown
            current_value = Decimal(str(self._portfolio.get_total_value()))
            
            if current_value > self._peak_value:
                self._peak_value = current_value
            
            current_drawdown = Decimal('0')
            if self._peak_value > 0:
                current_drawdown = (self._peak_value - current_value) / self._peak_value
            
            # Check if max drawdown is breached
            max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.3)))
            if current_drawdown > max_drawdown:
                self._risk_breached = True
                self.logger.info(f"Max drawdown limit breached in backtest: {float(current_drawdown*100):.2f}% > {float(max_drawdown*100):.2f}%")
                return False
            
            # Check position limits
            await self.check_position_limits()
            
            return not self._risk_breached
            
        except Exception as e:
            self.logger.error(f"Error in backtest risk control: {e}")
            return True  # Continue on error for backtest
    
    async def validate_order(self, kwargs) -> Dict[str, Any]:
        """
        Validate an order against risk parameters with enhanced checks
        
        Args:
            kwargs: Order parameters including symbol, quantity, direction, etc.
            
        Returns:
            Dict[str, Any]: Validation result with 'allowed' flag and reasons
        """
        if not self._portfolio:
            return {'allowed': True, 'reasons': ['No portfolio manager provided']}
            
        symbol = kwargs['symbol']
        amount = kwargs['quantity']
        direction = kwargs['direction'].lower()
            
        # Get asset
        asset = self._portfolio.assets.get(symbol)
        if not asset:
            return {'allowed': False, 'reasons': [f"Asset {symbol} not found"]}
            
        # Get price
        price = kwargs.get('price')
        if price is None and hasattr(asset, 'price'):
            price = asset.price
        elif price is None:
            return {'allowed': False, 'reasons': ["No price available for order validation"]}
            
        # Convert to Decimal for consistent calculations
        price = Decimal(str(price))
        amount = Decimal(str(amount))
            
        # Calculate order value
        # Different calculation for futures vs spot
        if hasattr(asset, 'contract_size'):
            order_value = amount * price * Decimal(str(asset.contract_size))
        else:
            order_value = amount * price
            
        # Get portfolio value
        portfolio_value = Decimal(str(self._portfolio.get_total_value()))
            
        # Calculate value as percentage of portfolio
        value_pct = order_value / portfolio_value if portfolio_value > 0 else Decimal('0.01')
            
        # List of validation checks
        validations = []
            
        # Check trade size limit
        trade_size_limit = Decimal(str(self.get_risk_limit('trade_size_limit', 0.05)))
        if value_pct > trade_size_limit and direction == 'buy':
            validations.append({
                'check': 'trade_size_limit',
                'passed': False,
                'message': f"Order size {float(value_pct*100)}% exceeds limit of {float(trade_size_limit*100)}%"
            })
        else:
            validations.append({
                'check': 'trade_size_limit',
                'passed': True,
                'message': f"Order size within limit: {float(value_pct*100)}% of portfolio"
            })
            
        # Check drawdown limits
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.3)))
            
        if current_drawdown > max_drawdown:
            validations.append({
                'check': 'drawdown_limit',
                'passed': False,
                'message': f"Current drawdown {float(current_drawdown*100)}% exceeds limit of {float(max_drawdown*100)}%"
            })
        else:
            validations.append({
                'check': 'drawdown_limit',
                'passed': True,
                'message': f"Drawdown within limit: {float(current_drawdown*100)}%"
            })
            
        # Check position count limit (if buying)
        if direction == 'buy':
            position_count = len([a for a in self._portfolio.assets.values() if hasattr(a, 'get_position_size') and a.get_position_size() > 0])
            max_positions = int(self.get_risk_limit('max_open_positions', 20))
                
            # Only apply this check for new positions
            current_position_size = asset.get_position_size() if hasattr(asset, 'get_position_size') else 0
            if current_position_size <= 0 and position_count >= max_positions:
                validations.append({
                    'check': 'position_count',
                    'passed': False,
                    'message': f"Position count {position_count} at maximum of {max_positions}"
                })
            else:
                validations.append({
                    'check': 'position_count',
                    'passed': True,
                    'message': f"Position count {position_count} within limit of {max_positions}"
                })
                
        # For sell orders, check if we have enough to sell
        if direction == 'sell':
            current_position_size = asset.get_position_size() if hasattr(asset, 'get_position_size') else 0
            if amount > current_position_size:
                validations.append({
                    'check': 'sell_limit',
                    'passed': False,
                    'message': f"Insufficient position: have {float(current_position_size)}, trying to sell {float(amount)}"
                })
            else:
                validations.append({
                    'check': 'sell_limit',
                    'passed': True,
                    'message': f"Sufficient position for sale: {float(current_position_size)} >= {float(amount)}"
                })
            
        # Determine if order is allowed
        passed_all = all(v['passed'] for v in validations)
        failed_reasons = [v['message'] for v in validations if not v['passed']]
            
        return {
            'allowed': passed_all,
            'reasons': failed_reasons if not passed_all else ["All risk checks passed"],
            'checks': validations,
            'asset': symbol,
            'direction': direction,
            'amount': float(amount),
            'value': float(order_value),
            'value_pct': float(value_pct)
        }
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report for backtesting
        
        Returns:
            Dict[str, Any]: Risk report information
        """
        if not self._portfolio:
            return {"error": "No portfolio manager provided"}
        
        try:
            # Get portfolio value
            portfolio_value = self._portfolio.get_total_value()
            
            # Calculate drawdown
            current_drawdown = Decimal('0')
            if self._peak_value > 0:
                current_drawdown = (self._peak_value - Decimal(str(portfolio_value))) / self._peak_value
            
            # Calculate portfolio exposure
            total_exposure = Decimal('0')
            position_count = 0
            
            # Detailed position information
            positions = {}
            asset_weights = self._portfolio.get_asset_weights()
            
            for asset_name, asset in self._portfolio.assets.items():
                asset_value = Decimal(str(asset.get_value()))
                
                if asset_value > 0:
                    position_count += 1
                    
                    # Add to positions dictionary
                    positions[asset_name] = {
                        "value": float(asset_value),
                        "weight": asset_weights.get(asset_name, 0)
                    }
                    
                    # Add to total exposure (using leveraged value for futures)
                    if hasattr(asset, 'get_exposure') and callable(asset.get_exposure):
                        exposure = Decimal(str(asset.get_exposure()))
                        positions[asset_name]["exposure"] = float(exposure)
                        positions[asset_name]["leverage"] = float(getattr(asset, 'leverage', 1.0))
                        total_exposure += exposure
                    else:
                        total_exposure += asset_value
            
            # For backtesting, return enhanced report
            report = {
                "portfolio_value": float(portfolio_value),
                "peak_value": float(self._peak_value),
                "drawdown": float(current_drawdown),
                "total_exposure": float(total_exposure),
                "exposure_ratio": float(total_exposure / portfolio_value) if portfolio_value > 0 else 0,
                "risk_breached": self._risk_breached,
                "position_count": position_count,
                "consecutive_losses": self._consecutive_losses,
                "max_consecutive_losses": self._max_consecutive_losses,
                "risk_limits": {k: float(v) if isinstance(v, Decimal) else v 
                               for k, v in self._risk_limits.items()},
                "positions": positions
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating backtest risk report: {e}")
            return {"error": str(e)}