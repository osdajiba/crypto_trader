#!/usr/bin/env python3
# src/portfolio/risk/standard.py

"""
Standard risk manager implementation.
Provides default risk management functionality for trading.
"""

import pandas as pd
import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Set

from src.common.abstract_factory import register_factory_class
from src.portfolio.risk.base import BaseRiskManager


@register_factory_class('risk_manager_factory', 'standard', 
    description="Standard risk management implementation",
    features=["position_limits", "drawdown_control", "leverage_limits"],
    category="risk")
class StandardRiskManager(BaseRiskManager):
    """
    Standard risk manager implementation
    
    This is the main implementation of risk manager for normal trading operations.
    It provides standard risk controls including position limits, drawdown control,
    and leverage limits.
    """
    
    async def _initialize_specific(self) -> None:
        """Initialize standard risk manager specific settings"""
        # Load additional risk limits specific to standard implementation
        self._risk_limits.update({
            'max_daily_loss': self.config.get("risk", "limits", "max_daily_loss", default=0.03),
            'max_consecutive_losses': self.config.get("risk", "limits", "max_consecutive_losses", default=5),
            'max_single_asset_exposure': self.config.get("risk", "limits", "max_single_asset_exposure", default=0.3)
        })
        
        # Initialize standard risk controls
        self._last_portfolio_value = Decimal('0')
        self._consecutive_losses = 0
        self._daily_high_value = Decimal('0')
        self._daily_low_value = Decimal('0')
        
        # Activate standard risk controls
        self.activate_control('position_limits')
        self.activate_control('drawdown_control')
        
        # Check if volatility control should be enabled
        if self.config.get("risk", "volatility_control", "enabled", default=False):
            self.activate_control('volatility_control')
            
        # Initialize the start portfolio value
        if self._portfolio:
            portfolio_value = self._portfolio.get_total_value()
            self._peak_value = Decimal(str(portfolio_value))
            self._last_portfolio_value = Decimal(str(portfolio_value))
            self._daily_high_value = Decimal(str(portfolio_value))
            self._daily_low_value = Decimal(str(portfolio_value))
    
    async def _add_additional_validations(self, validations: List[Dict[str, Any]], kwargs) -> None:
        """
        Add standard-specific validations
        
        Args:
            asset_name: Asset name
            direction: Order direction
            amount: Order amount
            validations: List of validation dictionaries to append to
            **kwargs: Additional parameters
        """
        if not self._portfolio:
            return
            
        asset_name = kwargs["symbol"]
        amount = kwargs["quantity"]
            
        try:
            # Calculate portfolio value
            portfolio_value = Decimal(str(self._portfolio.get_total_value()))
            
            # Check single asset exposure limit
            max_single_asset_exposure = Decimal(str(self.get_risk_limit('max_single_asset_exposure', 0.3)))
            
            # Get asset price
            asset = self._portfolio.assets.get(asset_name)
            if not asset:
                return
                
            asset_price = Decimal(str(getattr(asset, 'price', 1.0)))
            
            # Calculate order value
            if hasattr(asset, 'contract_size'):
                order_value = Decimal(str(amount)) * asset_price * Decimal(str(asset.contract_size))
            else:
                order_value = Decimal(str(amount)) * asset_price
                
            # Get current position value if any
            current_position_value = Decimal('0')
            if asset_name in self._portfolio.assets:
                current_position_value = Decimal(str(self._portfolio.assets[asset_name].get_value()))
                
            # Calculate new position value
            new_position_value = current_position_value + order_value
            
            # Check against limit
            if portfolio_value > 0:
                exposure_pct = new_position_value / portfolio_value
                
                if exposure_pct > max_single_asset_exposure:
                    validations.append({
                        'check': 'single_asset_exposure',
                        'passed': False,
                        'message': f"Asset exposure {float(exposure_pct*100):.2f}% exceeds limit of {float(max_single_asset_exposure*100):.2f}%"
                    })
                else:
                    validations.append({
                        'check': 'single_asset_exposure',
                        'passed': True,
                        'message': f"Asset exposure {float(exposure_pct*100):.2f}% within limit"
                    })
        except Exception as e:
            self.logger.error(f"Error in additional validations: {e}")
    
    async def _process_signals_specific(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Process signals with standard risk management logic
        
        Args:
            signals: Signals to process
            
        Returns:
            pd.DataFrame: Processed signals
        """
        if signals.empty:
            return signals
            
        # Check if we've exceeded consecutive loss limit
        max_consecutive_losses = int(self.get_risk_limit('max_consecutive_losses', 5))
        
        if self._consecutive_losses >= max_consecutive_losses:
            self.logger.warning(f"Max consecutive losses reached ({max_consecutive_losses}), reducing position sizes")
            
            # Reduce position sizes by 50%
            if 'quantity' in signals.columns:
                signals['quantity'] = signals['quantity'] * 0.5
                self.logger.info(f"Reduced all position sizes by 50% due to consecutive losses")
        
        # Check daily loss limit if we're in a drawdown for the day
        if self._portfolio:
            current_value = Decimal(str(self._portfolio.get_total_value()))
            
            # Calculate daily loss percentage
            if self._daily_high_value > 0:
                daily_loss_pct = (self._daily_high_value - current_value) / self._daily_high_value
                max_daily_loss = Decimal(str(self.get_risk_limit('max_daily_loss', 0.03)))
                
                if daily_loss_pct >= max_daily_loss:
                    self.logger.warning(f"Daily loss limit reached: {float(daily_loss_pct*100):.2f}%, rejecting buy signals")
                    
                    # Only allow sell signals
                    return signals[signals['action'].str.lower() == 'sell'].copy()
        
        return signals
    
    async def _execute_risk_control_specific(self) -> bool:
        """
        Execute standard risk control logic
        
        Returns:
            bool: True if trading should continue, False if trading should stop
        """
        if not self._portfolio:
            return True
            
        try:
            # Get current portfolio value
            current_value = Decimal(str(self._portfolio.get_total_value()))
            
            # Skip further checks if portfolio value is zero or negative
            if current_value <= 0:
                return True
                
            # Update high/low values for the day
            if current_value > self._daily_high_value:
                self._daily_high_value = current_value
            
            if self._daily_low_value == 0 or current_value < self._daily_low_value:
                self._daily_low_value = current_value
            
            # Check for consecutive losses
            if self._last_portfolio_value > 0:
                if current_value < self._last_portfolio_value:
                    self._consecutive_losses += 1
                else:
                    self._consecutive_losses = 0
            
            # Update last value
            self._last_portfolio_value = current_value
            
            # Check position limits
            position_limits_ok = await self.check_position_limits()
            if not position_limits_ok:
                return False
                
            # Check volatility control if active
            if self.is_control_active('volatility_control'):
                volatility_ok = await self._check_volatility()
                if not volatility_ok:
                    return False
            
            # Continue if not breached
            return not self._risk_breached
            
        except Exception as e:
            self.logger.error(f"Error in risk control execution: {e}")
            return True
    
    async def _check_additional_limits(self) -> bool:
        """
        Check additional risk limits specific to standard implementation
        
        Returns:
            bool: True if all limits are within bounds, False otherwise
        """
        if not self._portfolio:
            return True
            
        # Get all assets
        asset_names = self._portfolio.list_assets()
        portfolio_value = Decimal(str(self._portfolio.get_total_value()))
        
        # Calculate sector/asset type exposure
        sector_exposure = {}
        asset_type_exposure = {}
        
        for asset_name in asset_names:
            asset = self._portfolio.assets.get(asset_name)
            
            # Get asset value
            asset_value = Decimal(str(asset.get_value()))
            
            # Get sector and asset type if available
            sector = getattr(asset, 'sector', 'unknown')
            asset_type = getattr(asset, 'asset_type', 'unknown')
            
            # Sum exposure by sector
            sector_exposure[sector] = sector_exposure.get(sector, Decimal('0')) + asset_value
            
            # Sum exposure by asset type
            asset_type_exposure[asset_type] = asset_type_exposure.get(asset_type, Decimal('0')) + asset_value
        
        # Check sector concentration
        max_sector_exposure = Decimal(str(self.get_risk_limit('max_sector_exposure', 0.4)))
        
        for sector, exposure in sector_exposure.items():
            sector_pct = exposure / portfolio_value if portfolio_value > 0 else Decimal('0')
            
            if sector_pct > max_sector_exposure:
                self.logger.warning(f"Sector {sector} exposure ({float(sector_pct*100):.2f}%) exceeds limit ({float(max_sector_exposure*100):.2f}%)")
                self._risk_breached = True
                return False
        
        # Check asset type concentration
        max_asset_type_exposure = Decimal(str(self.get_risk_limit('max_asset_type_exposure', 0.7)))
        
        for asset_type, exposure in asset_type_exposure.items():
            asset_type_pct = exposure / portfolio_value if portfolio_value > 0 else Decimal('0')
            
            if asset_type_pct > max_asset_type_exposure:
                self.logger.warning(f"Asset type {asset_type} exposure ({float(asset_type_pct*100):.2f}%) exceeds limit ({float(max_asset_type_exposure*100):.2f}%)")
                self._risk_breached = True
                return False
        
        return True
    
    async def _check_volatility(self) -> bool:
        """
        Check portfolio volatility
        
        Returns:
            bool: True if volatility is within limits, False otherwise
        """
        # This method would normally compute volatility-based metrics
        # For this example, we'll use a simple check based on daily range
        
        if self._daily_high_value > 0 and self._daily_low_value > 0:
            daily_range = (self._daily_high_value - self._daily_low_value) / self._daily_high_value
            max_daily_range = Decimal(str(self.get_risk_limit('max_daily_range', 0.1)))
            
            if daily_range > max_daily_range:
                self.logger.warning(f"Daily range ({float(daily_range*100):.2f}%) exceeds limit ({float(max_daily_range*100):.2f}%)")
                return False
        
        return True
    
    def _add_to_risk_report(self, report: Dict[str, Any]) -> None:
        """
        Add standard-specific information to risk report
        
        Args:
            report: Report dictionary to modify in-place
        """
        report.update({
            "consecutive_losses": self._consecutive_losses,
            "daily_high_value": float(self._daily_high_value),
            "daily_low_value": float(self._daily_low_value),
            "daily_range_pct": float((self._daily_high_value - self._daily_low_value) / self._daily_high_value * 100) if self._daily_high_value > 0 else 0
        })