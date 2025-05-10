#!/usr/bin/env python3
# src/portfolio/risk/backtest.py

"""
Backtest risk manager implementation.
Provides simplified risk management functionality for backtesting.
"""

import time
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Any, Optional

from src.common.abstract_factory import register_factory_class
from src.portfolio.risk.base import BaseRiskManager


@register_factory_class('risk_manager_factory', 'backtest', 
    description="Simplified risk management for backtesting",
    features=["position_limits", "basic_validation"],
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
            'risk_per_trade': self.config.get("backtest", "risk", "risk_per_trade", default=0.02)
        }
        
        # Update limits with backtest-specific values
        self._risk_limits.update(backtest_limits)
        
        # Activate minimal controls for backtest
        self.activate_control('position_limits')
        
        # No need for temporary value tracking in backtest mode
        self._last_check_time = 0
        
        # Initialize portfolio tracking if portfolio is available
        if self._portfolio:
            portfolio_value = self._portfolio.get_total_value()
            self._peak_value = Decimal(str(portfolio_value))
            self.logger.info(f"BacktestRiskManager initialized with portfolio value: {portfolio_value}")
        else:
            self.logger.info("BacktestRiskManager initialized without portfolio")
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate trading signals with simplified logic for backtesting
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            pd.DataFrame: Filtered signals that pass risk checks
        """
        # For backtesting, we can use a simplified validation process
        # that is optimized for performance
        
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
            if portfolio_value <= 0:
                return signals  # Continue in backtest mode even with low portfolio
            
            # Update peak value if needed
            if portfolio_value > self._peak_value:
                self._peak_value = portfolio_value
            
            # Calculate current drawdown
            current_drawdown = Decimal('0')
            if self._peak_value > 0:
                current_drawdown = (self._peak_value - portfolio_value) / self._peak_value
            
            # Check if max drawdown is breached
            max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.3)))
            if current_drawdown > max_drawdown:
                self._risk_breached = True
                self.logger.info(f"Max drawdown limit breached in backtest: {float(current_drawdown*100):.2f}% > {float(max_drawdown*100):.2f}%")
                return signals[signals['action'].str.lower() == 'sell'].copy()
            
            # For buy signals, limit position sizing
            if not signals.empty and 'quantity' in signals.columns:
                trade_size_limit = Decimal(str(self.get_risk_limit('trade_size_limit', 0.05)))
                
                # Make a copy to avoid warnings
                valid_signals = signals.copy()
                
                # Apply position sizing to buy signals
                for idx, signal in valid_signals.iterrows():
                    if signal['action'].lower() == 'buy':
                        symbol = signal['symbol']
                        
                        # Skip if no asset info
                        asset = self._portfolio.assets.get(symbol)
                        if not asset or not hasattr(asset, 'price') or asset.price <= 0:
                            continue
                        
                        # Calculate max quantity based on portfolio
                        max_quantity = portfolio_value * trade_size_limit / Decimal(str(asset.price))
                        
                        # Adjust quantity if needed
                        if Decimal(str(signal['quantity'])) > max_quantity:
                            valid_signals.at[idx, 'quantity'] = float(max_quantity)
                
                return valid_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error in backtest signal validation: {e}")
            return signals  # Return original signals on error for backtesting
    
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
            
            # For backtesting, we skip more complex checks for performance
            # but this could be customized based on backtest needs
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in backtest risk control: {e}")
            return True  # Continue on error for backtest
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a simplified risk report for backtest mode
        
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
            
            # For backtesting, return simplified report
            report = {
                "portfolio_value": float(portfolio_value),
                "peak_value": float(self._peak_value),
                "drawdown": float(current_drawdown),
                "risk_breached": self._risk_breached,
                "position_count": len(self._portfolio.assets)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating backtest risk report: {e}")
            return {"error": str(e)}