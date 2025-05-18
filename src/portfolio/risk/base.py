#!/usr/bin/env python3
# src/portfolio/risk/base.py

"""
Base risk manager implementation.
Provides abstract base class for all risk management implementations.
"""

import pandas as pd
import asyncio
import time
from decimal import Decimal
from typing import Callable, Dict, List, Any, Optional, Set, Union
from abc import ABC, abstractmethod
from datetime import datetime

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager


class RiskValidationError(Exception):
    """Exception raised when risk validation fails"""
    pass


class RiskLimitBreachedError(Exception):
    """Exception raised when a risk limit is breached"""
    pass


class BaseRiskManager(ABC):
    """
    Base risk management class that defines common interface
    
    All risk manager implementations should inherit from this class to provide
    a consistent interface for risk management across the system.
    """
    
    def __init__(self, config: Optional[ConfigManager] = None, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk manager
        
        Args:
            config: Configuration manager
            params: Additional parameters including portfolio_manager
        """
        self.config = config or ConfigManager()
        self.params = params or {}
        self.logger = LogManager.get_logger(f"risk.{self.__class__.__name__.lower()}")
        
        # Extract portfolio manager from params
        self._portfolio = self.params.get('portfolio_manager')
        
        # Core risk management state
        self._risk_limits = {}
        self._active_controls: Set[str] = set()
        self._initialized = False
        self._risk_breached = False
        self._peak_value = Decimal('0')
        self._current_positions = {}
        self._risk_events = []
        self._last_check_time = 0
        
        # Get trading mode from config
        self._trading_mode = self.config.get("system", "operational_mode", default="backtest").lower()
        
        # Load default risk limits from config
        self._load_risk_limits()
    
    async def initialize(self) -> None:
        """
        Initialize the risk manager
        
        This method performs standard initialization and then calls
        initialize_specific for subclass-specific initialization.
        
        Subclasses should call super().initialize() when overriding
        """
        if self._initialized:
            return
            
        try:
            # Perform subclass-specific initialization
            await self._initialize_specific()
            
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized in {self._trading_mode} mode")
        except Exception as e:
            self.logger.error(f"Error initializing risk manager: {e}")
            raise
    
    async def _initialize_specific(self) -> None:
        """
        Specific initialization to be implemented by subclasses
        
        This method should be overridden by subclasses to perform any
        specific initialization tasks.
        """
        pass
    
    def _load_risk_limits(self) -> None:
        """
        Load risk limits from configuration
        
        This method loads default risk limits from configuration and
        can be extended by subclasses for additional limits.
        """
        # Basic limits
        self._risk_limits.update({
            'max_drawdown': self.config.get("risk", "drawdown", "max_portfolio", default=0.2),
            'max_position_size': self.config.get("trading", "limits", "position", default=0.1),
            'max_leverage': self.config.get("trading", "limits", "max_leverage", default=10.0),
            'max_open_positions': self.config.get("risk", "limits", "max_open_positions", default=10),
            'max_concentration': self.config.get("risk", "limits", "max_concentration", default=0.3),
            'position_limit_pct': self.config.get("risk", "limits", "position_limit_pct", default=0.05),
            'trade_size_limit': self.config.get("risk", "limits", "trade_size_limit", default=0.02),
            'risk_per_trade': self.config.get("risk", "limits", "risk_per_trade", default=0.01)
        })
        
        # Stop loss / take profit defaults
        if self.config.get("risk", "stop_loss", "enabled", default=False):
            self._risk_limits['stop_loss_percentage'] = self.config.get("risk", "stop_loss", "percentage", default=0.02)
            self.activate_control('stop_loss')
            
        if self.config.get("risk", "take_profit", "enabled", default=False):
            self._risk_limits['take_profit_percentage'] = self.config.get("risk", "take_profit", "percentage", default=0.05)
            self.activate_control('take_profit')
        
        # Log loaded limits
        self.logger.debug(f"Loaded risk limits: {self._risk_limits}")
    
    async def validate_order(self, kwargs) -> Dict[str, Any]:
        """
        Validate an order against risk parameters
        
        Args:
            asset_name: Asset name
            direction: Order direction ('buy' or 'sell')
            amount: Order amount
            **kwargs: Additional order parameters
            
        Returns:
            Dict[str, Any]: Validation result with 'allowed' flag and reasons
        """
        if not self._initialized:
            await self.initialize()
            
        if not self._portfolio:
            return {'allowed': True, 'reasons': ['No portfolio manager provided']}
        
        asset_name = kwargs['symbol']
        amount = kwargs['quantity']
        direction = kwargs['direction'].lower()   
         
        # Get asset
        asset = self._portfolio.assets.get(asset_name)
        if not asset:
            return {'allowed': False, 'reasons': [f"Asset {asset_name} not found"]}
        
        # Get asset price and calculate order value
        asset_price = Decimal(str(getattr(asset, 'price', 1.0)))
        
        # Different calculation for futures vs spot
        if hasattr(asset, 'contract_size'):
            order_value = Decimal(str(amount)) * asset_price * Decimal(str(asset.contract_size))
        else:
            order_value = Decimal(str(amount)) * asset_price
        
        # Get portfolio value
        portfolio_value = Decimal(str(self._portfolio.get_total_value()))
        
        # Calculate value as percentage of portfolio
        value_pct = order_value / portfolio_value if portfolio_value > 0 else Decimal('0.01')
        
        # List of validation checks
        validations = []
        
        # 1. Check trade size limit
        trade_size_limit = Decimal(str(self.get_risk_limit('trade_size_limit', 0.0001)))
        if value_pct > trade_size_limit:
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
        
        # 2. Check position exposure limit (only for buys)
        if direction.lower() == 'buy':
            # Calculate current position value
            position_value = Decimal('0')
            if asset_name in self._portfolio.assets:
                position_value = Decimal(str(self._portfolio.assets[asset_name].get_value()))
            
            # Calculate new position value after order
            new_position_value = position_value + order_value
            
            # Calculate as percentage of portfolio
            position_limit_pct = Decimal(str(self.get_risk_limit('position_limit_pct', 0.05)))
            new_position_pct = new_position_value / portfolio_value if portfolio_value > 0 else Decimal('0')
            
            if new_position_pct > position_limit_pct:
                validations.append({
                    'check': 'position_limit',
                    'passed': False,
                    'message': f"Position would exceed limit: {float(new_position_pct*100)}% > {float(position_limit_pct*100)}%"
                })
            else:
                validations.append({
                    'check': 'position_limit',
                    'passed': True,
                    'message': f"Position within limit: {float(new_position_pct*100)}% of portfolio"
                })
        
        # 3. Check leverage limits for futures
        if hasattr(asset, 'leverage') and hasattr(asset, 'get_exposure'):
            leverage = Decimal(str(asset.leverage))
            max_leverage = Decimal(str(self.get_risk_limit('max_leverage', 10.0)))
            
            if leverage > max_leverage:
                validations.append({
                    'check': 'leverage_limit',
                    'passed': False,
                    'message': f"Leverage {float(leverage)}x exceeds limit of {float(max_leverage)}x"
                })
            else:
                validations.append({
                    'check': 'leverage_limit',
                    'passed': True,
                    'message': f"Leverage within limit: {float(leverage)}x"
                })
            
            # Calculate total exposure with leverage
            exposure = Decimal(str(asset.get_exposure()))
            if portfolio_value > 0:
                exposure_pct = exposure / portfolio_value
                max_position_size = Decimal(str(self.get_risk_limit('max_position_size', 0.2)))
                
                if exposure_pct > max_position_size:
                    validations.append({
                        'check': 'exposure_limit',
                        'passed': False,
                        'message': f"Total exposure {float(exposure_pct*100)}% exceeds limit of {float(max_position_size*100)}%"
                    })
                else:
                    validations.append({
                        'check': 'exposure_limit',
                        'passed': True,
                        'message': f"Total exposure within limit: {float(exposure_pct*100)}%"
                    })
        
        # 4. Check drawdown limits
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.2)))
        
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
        
        # 5. Check position count limit (if buying)
        if direction.lower() == 'buy':
            current_positions = len(self._portfolio.assets)
            max_positions = int(self.get_risk_limit('max_open_positions', 10))
            
            if current_positions >= max_positions:
                validations.append({
                    'check': 'position_count',
                    'passed': False,
                    'message': f"Position count {current_positions} at maximum of {max_positions}"
                })
            else:
                validations.append({
                    'check': 'position_count',
                    'passed': True,
                    'message': f"Position count {current_positions} within limit of {max_positions}"
                })
        
        # Allow subclasses to add additional validations
        await self._add_additional_validations(validations, kwargs)
        
        # Determine if order is allowed
        passed_all = all(v['passed'] for v in validations)
        failed_reasons = [v['message'] for v in validations if not v['passed']]
        
        return {
            'allowed': passed_all,
            'reasons': failed_reasons if not passed_all else ["All risk checks passed"],
            'checks': validations,
            'asset': asset_name,
            'direction': direction,
            'amount': float(amount),
            'value': float(order_value),
            'value_pct': float(value_pct)
        }
    
    async def _add_additional_validations(self, validations, kwargs) -> None:
        """
        Add additional validations specific to subclass implementation
        
        Args:
            asset_name: Asset name
            direction: Order direction
            amount: Order amount
            validations: List of validation dictionaries to append to
            **kwargs: Additional parameters
        """
        pass
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate trading signals against risk rules
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            pd.DataFrame: Filtered signals that pass risk checks
            
        Raises:
            RiskValidationError: If signals don't meet risk rules
        """
        if signals.empty:
            return signals

        # Apply risk filtering to create validated signals
        valid_signals = signals.copy()
        
        try:
            # Check if risk is already breached (stop trading if true)
            if self._risk_breached:
                self.logger.warning("Risk limits already breached, rejecting all new long positions")
                return valid_signals[valid_signals['action'].str.lower() == 'sell'].copy()
            
            # Get portfolio value
            portfolio_value = Decimal(str(self._portfolio.get_total_value() if self._portfolio else 0))
            if portfolio_value <= 0:
                self.logger.error("Invalid portfolio value - cannot validate position sizing")
                return pd.DataFrame()  # Return empty dataframe to reject all signals
            
            drop_indices = []
            
            for idx, signal in valid_signals.iterrows():
                direction = signal['action'].lower() if 'action' in signal else 'unknown'
                symbol = signal['symbol'] if 'symbol' in signal else 'unknown'
                kwargs = {
                    "symbol": signal['symbol'],
                    "price": None,
                    "quantity": signal['quantity'],
                    "order_type": None,
                    "direction": signal['action']
                }
                if direction == 'sell':
                    # Validate the order
                    validation = await self.validate_order(kwargs)
                    
                    if not validation.get('allowed', False):
                        reasons = validation.get('reasons', ['Failed risk check'])
                        self.logger.warning(f"Signal for {symbol} rejected: {reasons}")
                        drop_indices.append(idx)
                
                # Validate buy signals
                if direction == 'buy':
                    # Validate the order
                    validation = await self.validate_order(kwargs)
                    
                    if not validation.get('allowed', False):
                        reasons = validation.get('reasons', ['Failed risk check'])
                        self.logger.warning(f"Signal for {symbol} rejected: {reasons}")
                        drop_indices.append(idx)
            
            # Drop rejected signals
            if drop_indices:
                valid_signals = valid_signals.drop(drop_indices)
            
            # Update signal quantity based on risk limits
            if not valid_signals.empty and 'quantity' in valid_signals.columns:
                for idx, signal in valid_signals.iterrows():
                    if signal['action'].lower() == 'buy':
                        # Get the asset
                        asset_name = signal['symbol']
                        asset = self._portfolio.assets.get(asset_name)
                        
                        if asset and hasattr(asset, 'price') and asset.price > 0:
                            # Calculate risk-adjusted position size
                            risk_per_trade = Decimal(str(self.get_risk_limit('risk_per_trade', 0.01)))
                            max_trade_size = portfolio_value * risk_per_trade / Decimal(str(asset.price))
                            
                            # Adjust quantity if needed
                            if Decimal(str(signal['quantity'])) > max_trade_size:
                                valid_signals.at[idx, 'quantity'] = float(max_trade_size)
                                self.logger.info(f"Adjusted {asset_name} quantity from {signal['quantity']} to {float(max_trade_size)}")
            
            # Allow subclasses to further process signals
            valid_signals = await self._process_signals_specific(valid_signals)
            
            return valid_signals
            
        except Exception as e:
            self.logger.error(f"Error during signal validation: {e}")
            raise RiskValidationError(f"Failed to validate signals: {str(e)}")
    
    async def _process_signals_specific(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Process signals in a way specific to subclass implementation
        
        Args:
            signals: Signals to process
            
        Returns:
            pd.DataFrame: Processed signals
        """
        return signals
    
    async def execute_risk_control(self) -> bool:
        """
        Execute risk control logic
        
        Checks for limit breaches and takes appropriate actions
        
        Returns:
            bool: True if trading should continue, False if trading should stop
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Refresh portfolio data
            if self._portfolio:
                await self._portfolio.update_all_values()
            
            # Check drawdown limits
            current_drawdown = self._calculate_current_drawdown()
            max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.2)))
            
            if current_drawdown >= max_drawdown and not self._risk_breached:
                self._risk_breached = True
                self.logger.warning(f"Max drawdown limit breached: {float(current_drawdown*100):.2f}% >= {float(max_drawdown*100):.2f}%")
                
                # If stop loss control is active, close positions
                if self.is_control_active('stop_loss') and self._portfolio:
                    await self._emergency_position_reduction()
                    
                return False
            
            # Execute subclass-specific risk controls
            return await self._execute_risk_control_specific()
                
        except Exception as e:
            self.logger.error(f"Error in risk control execution: {e}")
            return True  # Continue trading on error (safer not to interrupt)
    
    async def _execute_risk_control_specific(self) -> bool:
        """
        Execute risk control logic specific to subclass implementation
        
        Returns:
            bool: True if trading should continue, False if trading should stop
        """
        return not self._risk_breached
    
    async def _emergency_position_reduction(self, reduction_pct: float = 0.5) -> None:
        """
        Reduce positions in emergency situations
        
        Args:
            reduction_pct: Percentage of positions to reduce (0.0-1.0)
        """
        if not self._portfolio:
            return
            
        self.logger.warning(f"Executing emergency position reduction ({reduction_pct*100}%)")
        
        try:
            # Get all assets
            asset_names = self._portfolio.list_assets()
            
            for asset_name in asset_names:
                asset = self._portfolio.assets.get(asset_name)
                
                # Skip if not a tradable asset
                if not hasattr(asset, 'get_position_size'):
                    continue
                
                # Get position size
                position_size = asset.get_position_size()
                
                if position_size > 0:
                    # Calculate reduction amount
                    reduction_amount = position_size * reduction_pct
                    
                    if reduction_amount > 0:
                        self.logger.warning(f"Emergency sell: {asset_name} - {reduction_amount} units")
                        
                        # Execute the sell
                        try:
                            await self._portfolio.sell_asset(asset_name, float(reduction_amount))
                        except Exception as e:
                            self.logger.error(f"Error in emergency sell for {asset_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error in emergency position reduction: {e}")
    
    def _calculate_current_drawdown(self) -> Decimal:
        """
        Calculate current portfolio drawdown
        
        Returns:
            Decimal: Current drawdown as a fraction (0.0-1.0)
        """
        if not self._portfolio:
            return Decimal('0')
            
        current_value = Decimal(str(self._portfolio.get_total_value()))
        peak_value = self._peak_value
        
        # Update peak value if needed
        if current_value > peak_value:
            self._peak_value = current_value
            peak_value = current_value
        
        # Calculate drawdown
        if peak_value > 0:
            return (peak_value - current_value) / peak_value
        
        return Decimal('0')
    
    def set_risk_limit(self, limit_name: str, value: Any) -> None:
        """
        Set a specific risk limit
        
        Args:
            limit_name: Name of the limit
            value: Limit value
        """
        self._risk_limits[limit_name] = value
        self.logger.info(f"Risk limit '{limit_name}' set to {value}")
    
    def get_risk_limit(self, limit_name: str, default: Any = None) -> Any:
        """
        Get a specific risk limit
        
        Args:
            limit_name: Name of the limit
            default: Default value if limit doesn't exist
            
        Returns:
            Limit value or default
        """
        return self._risk_limits.get(limit_name, default)
    
    def activate_control(self, control_name: str) -> None:
        """
        Activate a specific risk control
        
        Args:
            control_name: Name of the control to activate
        """
        self._active_controls.add(control_name)
        self.logger.info(f"Risk control '{control_name}' activated")
    
    def deactivate_control(self, control_name: str) -> None:
        """
        Deactivate a specific risk control
        
        Args:
            control_name: Name of the control to deactivate
        """
        if control_name in self._active_controls:
            self._active_controls.remove(control_name)
            self.logger.info(f"Risk control '{control_name}' deactivated")
    
    def is_control_active(self, control_name: str) -> bool:
        """
        Check if a specific control is active
        
        Args:
            control_name: Name of the control to check
            
        Returns:
            bool: True if control is active, False otherwise
        """
        return control_name in self._active_controls
    
    def is_risk_breached(self) -> bool:
        """
        Check if any risk limits have been breached
        
        Returns:
            bool: True if risk limits are breached, False otherwise
        """
        return self._risk_breached
    
    async def check_position_limits(self) -> bool:
        """
        Check if positions are within limits
        
        Returns:
            bool: True if positions are within limits, False otherwise
        """
        if not self._portfolio:
            return True
            
        try:
            # Get all assets
            asset_names = self._portfolio.list_assets()
            portfolio_value = Decimal(str(self._portfolio.get_total_value()))
            
            if portfolio_value <= 0:
                return False
            
            # Check total exposure
            total_exposure = Decimal('0')
            for asset_name in asset_names:
                asset = self._portfolio.assets.get(asset_name)
                
                # If it's a futures asset, add leveraged exposure
                if hasattr(asset, 'get_exposure'):
                    total_exposure += Decimal(str(asset.get_exposure()))
                else:
                    # For spot assets, add direct value
                    total_exposure += Decimal(str(asset.get_value()))
            
            # Check total exposure as percentage of portfolio
            max_exposure = Decimal(str(self.get_risk_limit('max_total_exposure', 2.0)))
            current_exposure_pct = total_exposure / portfolio_value
            
            if current_exposure_pct > max_exposure:
                self.logger.warning(f"Total exposure ({float(current_exposure_pct):.2f}x) exceeds limit ({float(max_exposure):.2f}x)")
                self._risk_breached = True
                return False
            
            # Check individual position limits
            for asset_name in asset_names:
                asset = self._portfolio.assets.get(asset_name)
                asset_value = Decimal(str(asset.get_value()))
                
                # Check asset value as percentage of portfolio
                position_pct = asset_value / portfolio_value
                max_position_size = Decimal(str(self.get_risk_limit('max_position_size', 0.2)))
                
                if position_pct > max_position_size:
                    self.logger.warning(f"{asset_name} position size ({float(position_pct*100):.2f}%) exceeds limit ({float(max_position_size*100):.2f}%)")
                    self._risk_breached = True
                    return False
                
                # Check leverage for futures
                if hasattr(asset, 'leverage'):
                    leverage = Decimal(str(asset.leverage))
                    max_leverage = Decimal(str(self.get_risk_limit('max_leverage', 10.0)))
                    
                    if leverage > max_leverage:
                        self.logger.warning(f"{asset_name} leverage ({float(leverage):.2f}x) exceeds limit ({float(max_leverage):.2f}x)")
                        self._risk_breached = True
                        return False
            
            # Allow subclasses to check additional limits
            if not await self._check_additional_limits():
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return True  # Don't block trading due to calculation errors
    
    async def _check_additional_limits(self) -> bool:
        """
        Check additional limits specific to subclass implementation
        
        Returns:
            bool: True if all limits are within bounds, False otherwise
        """
        return True
    
    def get_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report
        
        Returns:
            Dict[str, Any]: Risk report information
        """
        if not self._portfolio:
            return {"error": "No portfolio manager provided"}
        
        try:
            # Get portfolio value
            portfolio_value = self._portfolio.get_total_value()
            
            # Get drawdown
            current_drawdown = float(self._calculate_current_drawdown())
            
            # Get position details
            positions = {}
            weights = self._portfolio.get_asset_weights()
            
            for asset_name, asset in self._portfolio.assets.items():
                position_info = {
                    "value": asset.get_value(),
                    "weight": weights.get(asset_name, 0),
                }
                
                # Add futures-specific info
                if hasattr(asset, 'leverage'):
                    position_info.update({
                        "leverage": float(asset.leverage),
                        "exposure": asset.get_exposure(),
                        "contract_size": float(getattr(asset, 'contract_size', 1.0)),
                        "liquidation_price": float(getattr(asset, '_liquidation_price', 0.0))
                    })
                
                positions[asset_name] = position_info
            
            # Get total exposure
            total_exposure = sum(
                pos.get("exposure", pos["value"]) 
                for pos in positions.values()
            )
            
            # Create report
            report = {
                "portfolio_value": float(portfolio_value),
                "drawdown": current_drawdown,
                "total_exposure": float(total_exposure),
                "leverage_ratio": float(total_exposure / portfolio_value) if portfolio_value > 0 else 0,
                "risk_breached": self._risk_breached,
                "active_controls": list(self._active_controls),
                "risk_limits": {k: float(v) if isinstance(v, Decimal) else v for k, v in self._risk_limits.items()},
                "positions": positions,
                "position_count": len(positions)
            }
            
            # Allow subclasses to add additional report information
            self._add_to_risk_report(report)
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {"error": str(e)}
    
    def _add_to_risk_report(self, report: Dict[str, Any]) -> None:
        """
        Add additional information to risk report (subclass specific)
        
        Args:
            report: Report dictionary to modify in-place
        """
        pass

    def update_portfolio_value(self, value: float) -> None:
        """
        Update portfolio value without recalculating drawdown
        
        Args:
            value: New portfolio value
        """
        self._last_portfolio_value = Decimal(str(value))
        
    def register_event_handler(self, event_type: str, handler: Callable) -> None:
        """
        Register an event handler
        
        Args:
            event_type: Event type
            handler: Event handler function
        """
        if not hasattr(self, '_event_handlers'):
            self._event_handlers = {}
            
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
            
        self._event_handlers[event_type].append(handler)
        self.logger.debug(f"Registered handler for {event_type} events")
        
    async def trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Trigger an event to notify registered handlers
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not hasattr(self, '_event_handlers'):
            return
            
        handlers = self._event_handlers.get(event_type, [])
        
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
        
    async def handle_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle an event from the portfolio manager
        
        Args:
            event_type: Event type
            data: Event data
        """
        # Process different event types
        if event_type == 'order_executed':
            # Update risk state based on new order
            await self._process_order_execution(data)
        elif event_type == 'significant_value_change':
            # Handle significant value changes
            await self._process_value_change(data)
        elif event_type == 'asset_added' or event_type == 'asset_removed':
            # Update asset tracking
            await self._update_asset_tracking()
        elif event_type == 'sync_completed':
            # Update risk after portfolio sync
            await self.execute_risk_control()
        
    async def _process_order_execution(self, data: Dict[str, Any]) -> None:
        """
        Process order execution event
        
        Args:
            data: Order execution data
        """
        asset_name = data.get('asset_name')
        direction = data.get('direction')
        amount = data.get('amount')
        
        # Update position tracking
        if asset_name in self._current_positions:
            if direction == 'buy':
                self._current_positions[asset_name] += amount
            elif direction == 'sell':
                self._current_positions[asset_name] -= amount
        else:
            if direction == 'buy':
                self._current_positions[asset_name] = amount
            
        # Check for risk limit breaches
        await self.check_position_limits()
        
    async def _process_value_change(self, data: Dict[str, Any]) -> None:
        """
        Process significant value change event
        
        Args:
            data: Value change data
        """
        change_pct = data.get('change_pct', 0)
        
        # Check for large drawdowns
        if change_pct < -0.05:  # More than 5% drop
            drawdown = self._calculate_current_drawdown()
            max_drawdown = Decimal(str(self.get_risk_limit('max_drawdown', 0.2)))
            
            # Send notification for significant drawdowns
            if drawdown > max_drawdown * Decimal('0.7'):  # Over 70% of max
                await self.trigger_event('risk_breach', {
                    'type': 'approaching_max_drawdown',
                    'severity': 'warning',
                    'current': float(drawdown),
                    'max': float(max_drawdown),
                    'percentage': float(drawdown / max_drawdown * 100)
                })

    async def _update_asset_tracking(self) -> None:
        """Update internal asset tracking after asset changes"""
        if not self._portfolio:
            return
            
        # Update position tracking
        self._current_positions = {}
        for asset_name, asset in self._portfolio.assets.items():
            if hasattr(asset, 'get_position_size'):
                self._current_positions[asset_name] = asset.get_position_size()
                
        
            
    async def shutdown(self) -> None:
        """
        Clean up resources
        """
        # Call subclass-specific shutdown
        await self._shutdown_specific()
        
        # Reset state
        self._initialized = False
        
        self.logger.info(f"{self.__class__.__name__} shutdown completed")
    
    async def _shutdown_specific(self) -> None:
        """
        Specific shutdown operations for subclasses
        """
        pass
    