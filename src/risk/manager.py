# src/risk/risk_manager.py

import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from datetime import datetime

from src.common.log_manager import LogManager


class RiskValidationError(Exception):
    """Exception raised when risk validation fails"""
    pass


class BaseRiskManager(ABC):
    """
    Base risk management class that defines common interface
    All risk manager implementations should inherit from this class
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk manager
        
        Args:
            config: Risk management configuration
        """
        self._logger = LogManager.get_logger(f"risk.{self.__class__.__name__.lower()}")
        self._config = config or {}
        self._risk_limits = {}
        self._active_controls: Set[str] = set()
        self._initialized = False
        self._risk_breached = False
    
    async def initialize(self) -> None:
        """
        Initialize the risk manager
        Subclasses should call super().initialize() when overriding
        """
        # Load risk limits from configuration
        self._load_risk_limits()
        
        self._initialized = True
        self._logger.info(f"{self.__class__.__name__} initialized")
    
    def _load_risk_limits(self) -> None:
        """Load risk limits from configuration"""
        if not self._config:
            return
            
        # Get risk limits from config
        risk_limits = self._config.get('risk_limits', {})
        
        # If no specific risk limits defined, look for structured config
        if not risk_limits:
            # Basic limits
            self._risk_limits.update({
                'max_drawdown': self._config.get("risk", "drawdown", "max_portfolio", default=0.2),
                'max_position_size': self._config.get("trading", "limits", "position", default=0.1),
                'max_leverage': self._config.get("trading", "limits", "max_leverage", default=1.0),
                'max_open_positions': self._config.get("risk", "limits", "max_open_positions", default=10),
                'max_concentration': self._config.get("risk", "limits", "max_concentration", default=0.3)
            })
            
            # Stop loss / take profit defaults
            if self._config.get("risk", "stop_loss", "enabled", default=False):
                self._risk_limits['stop_loss_percentage'] = self._config.get("risk", "stop_loss", "percentage", default=0.02)
                self.activate_control('stop_loss')
                
            if self._config.get("risk", "take_profit", "enabled", default=False):
                self._risk_limits['take_profit_percentage'] = self._config.get("risk", "take_profit", "percentage", default=0.05)
                self.activate_control('take_profit')
        else:
            # Use risk limits directly
            self._risk_limits.update(risk_limits)
        
        # Log loaded limits
        self._logger.debug(f"Loaded risk limits: {self._risk_limits}")
    
    @abstractmethod
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Common interface for validating trading signals
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            Filtered or modified signals that pass risk checks
            
        Raises:
            RiskValidationError: If signals don't meet risk rules
        """
        pass
    
    @abstractmethod
    async def execute_risk_control(self) -> None:
        """
        Common interface for executing risk control
        Should check current market and position state and take corrective actions if needed
        """
        pass
    
    def set_risk_limit(self, limit_name: str, value: Any) -> None:
        """
        Set a specific risk limit
        
        Args:
            limit_name: Name of the limit
            value: Limit value
        """
        self._risk_limits[limit_name] = value
        self._logger.info(f"Risk limit '{limit_name}' set to {value}")
    
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
        self._logger.info(f"Risk control '{control_name}' activated")
    
    def deactivate_control(self, control_name: str) -> None:
        """
        Deactivate a specific risk control
        
        Args:
            control_name: Name of the control to deactivate
        """
        if control_name in self._active_controls:
            self._active_controls.remove(control_name)
            self._logger.info(f"Risk control '{control_name}' deactivated")
    
    def is_control_active(self, control_name: str) -> bool:
        """
        Check if a specific control is active
        
        Args:
            control_name: Name of the control to check
            
        Returns:
            True if control is active, False otherwise
        """
        return control_name in self._active_controls
    
    def is_risk_breached(self) -> bool:
        """
        Check if any risk limits have been breached
        
        Returns:
            True if risk limits are breached, False otherwise
        """
        return self._risk_breached
    
    @abstractmethod
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        Check if positions are within limits
        
        Args:
            positions: Current positions
            
        Returns:
            True if positions are within limits, False otherwise
        """
        pass


class RiskManagerFactory:
    """Factory for creating appropriate risk manager instances"""
    
    @staticmethod
    def create_risk_manager(mode: str, config: Dict[str, Any]) -> BaseRiskManager:
        """
        Creates appropriate risk manager based on mode
        
        Args:
            mode: Trading mode ('backtest', 'paper', 'live')
            config: Configuration dictionary
            
        Returns:
            Appropriate risk manager instance
        """
        
        if mode.lower() == "backtest":
            return BacktestRiskManager(config)
        elif mode.lower() == "paper":
            # Paper trading can use the backtest risk manager
            return BacktestRiskManager(config)
        elif mode.lower() == "live":
            return LiveRiskManager(config)
        else:
            # Default to backtest risk manager
            return BacktestRiskManager(config)


class BacktestRiskManager(BaseRiskManager):
    """
    Risk management implementation for backtesting and paper trading
    
    Provides appropriate risk controls and signal validation
    with a focus on performance and realistic simulation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtest risk manager
        
        Args:
            config: Risk management configuration
        """
        super().__init__(config)
        
        # Backtest-specific state
        self._portfolio_equity = []
        self._current_positions = {}
        self._risk_events = []
        initial_capital = self._config.get("trading", "capital", "initial", default=100000.0)
        self._current_equity = float(initial_capital)  # Ensure it's a float
        self._max_equity = self._current_equity
        
        # Add initial equity point
        self._portfolio_equity.append({
            'timestamp': pd.Timestamp.now(),
            'equity': initial_capital,
            'drawdown': 0.0
        })
        
        self._logger.info(f"BacktestRiskManager initialized with {initial_capital} initial capital")
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Validate trading signals for backtesting"""
        if signals.empty:
            return signals

        # Apply risk filtering to create validated signals
        valid_signals = signals.copy()

        try:
            # Check if risk is already breached (stop trading if true)
            if self._risk_breached:
                self._logger.warning("Risk limits already breached, rejecting all new long positions")
                return valid_signals[valid_signals['action'].str.lower() == 'sell'].copy()

            # Check if we have valid equity value
            if not hasattr(self, '_current_equity') or self._current_equity <= 0:
                self._logger.error("Invalid portfolio equity value - cannot validate position sizing")
                return pd.DataFrame()  # Return empty dataframe to reject all signals

            drop_indices = []
            
            for idx, signal in valid_signals.iterrows():
                action = signal['action'].lower() if 'action' in signal else 'unknown'
                symbol = signal['symbol'] if 'symbol' in signal else 'unknown'
                
                # Skip invalid signals
                if action not in ['buy', 'sell', 'short']:
                    self._logger.warning(f"Unknown action type: {action}, skipping")
                    drop_indices.append(idx)
                    continue
                    
                # Apply position limits for buy orders only (don't restrict selling)
                if action == 'buy':
                    # Apply position allocation limit with safe division
                    max_allocation = self.get_risk_limit('max_position_size', 0.1)
                    if max_allocation < 1.0 and 'quantity' in signal and 'price' in signal:
                        try:
                            new_allocation = signal['quantity'] * signal['price'] / self._current_equity
                            if new_allocation > max_allocation:
                                self._logger.warning(
                                    f"Signal for {symbol} exceeds max position allocation "
                                    f"({new_allocation:.2%} > {max_allocation:.2%})"
                                )
                                drop_indices.append(idx)
                                continue
                        except ZeroDivisionError:
                            self._logger.error("Division by zero in position allocation calculation")
                            drop_indices.append(idx)
                            continue
                            
                    # Check max open positions
                    max_positions = self.get_risk_limit('max_open_positions', 10)
                    if len(self._current_positions) >= max_positions:
                        self._logger.warning(
                            f"Signal for {symbol} rejected: max positions limit reached "
                            f"({len(self._current_positions)} >= {max_positions})"
                        )
                        drop_indices.append(idx)
                        continue
            
            # Drop rejected signals
            if drop_indices:
                valid_signals = valid_signals.drop(drop_indices)
            
            # Update signal quantity based on risk limits with safe division
            if 'quantity' in valid_signals.columns and 'price' in valid_signals.columns:
                for idx, signal in valid_signals.iterrows():
                    if signal['action'].lower() == 'buy' and self._current_equity > 0:
                        risk_per_trade = self.get_risk_limit('risk_per_trade', 0.01)
                        try:
                            adjusted_quantity = risk_per_trade * self._current_equity / signal['price']
                            min_quantity = self.get_risk_limit('min_trade_quantity', 0.001)
                            max_quantity = self.get_risk_limit('max_trade_quantity', float('inf'))
                            
                            valid_signals.at[idx, 'quantity'] = max(
                                min_quantity, 
                                min(adjusted_quantity, max_quantity, signal['quantity'])
                            )
                        except ZeroDivisionError:
                            self._logger.error(f"Zero price for {signal.get('symbol', 'unknown')}, skipping quantity adjustment")
                            continue
            
            return valid_signals
            
        except Exception as e:
            self._logger.error(f"Error during signal validation: {e}", exc_info=True)
            return pd.DataFrame()
    
    async def execute_risk_control(self) -> None:
        """Execute risk control logic for backtest"""
        # In backtest mode, just update drawdown calculations
        if self._portfolio_equity:
            current_equity = self._portfolio_equity[-1]['equity']
            if current_equity > self._max_equity:
                self._max_equity = current_equity
            
            # Calculate current drawdown
            if self._max_equity > 0:
                current_drawdown = (self._max_equity - current_equity) / self._max_equity
                # Update state
                if current_drawdown > 0:
                    self._portfolio_equity[-1]['drawdown'] = current_drawdown
                    
                # Check if max drawdown is breached
                max_drawdown_limit = self.get_risk_limit('max_drawdown', 0.25)
                if current_drawdown >= max_drawdown_limit:
                    if not self._risk_breached:
                        self._risk_breached = True
                        self._risk_events.append({
                            'timestamp': datetime.now(),
                            'type': 'drawdown_breach',
                            'value': current_drawdown,
                            'limit': max_drawdown_limit
                        })
                        self._logger.warning(
                            f"Max drawdown limit breached: {current_drawdown:.2%} >= {max_drawdown_limit:.2%}"
                        )
    
    def update_portfolio_state(self, equity: float, positions: Dict[str, Any], timestamp: Any = None) -> None:
        """
        Update portfolio state for risk calculations
        
        Args:
            equity: Current portfolio equity
            positions: Current positions
            timestamp: Current timestamp (optional)
        """
        # Default timestamp to now if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update equity curve
        self._current_equity = equity
        
        # Calculate drawdown
        drawdown = 0.0
        if self._max_equity > 0:
            drawdown = (self._max_equity - equity) / self._max_equity
            
        # Update max equity
        if equity > self._max_equity:
            self._max_equity = equity
            
        # Add new equity point
        self._portfolio_equity.append({
            'timestamp': timestamp,
            'equity': equity,
            'drawdown': drawdown
        })
        
        # Update positions
        self._current_positions = positions
        
        # Check position limits
        position_value = sum(p.get('value', 0) for p in positions.values())
        max_position_value = self.get_risk_limit('max_position_value', equity * 0.9)  # Default 90% of equity
        
        if position_value > max_position_value:
            self._risk_breached = True
            self._risk_events.append({
                'timestamp': timestamp,
                'type': 'position_limit_breach',
                'value': position_value,
                'limit': max_position_value
            })
            self._logger.warning(
                f"Total position value exceeds limit: {position_value:.2f} > {max_position_value:.2f}"
            )
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown experienced during backtest
        
        Returns:
            float: Maximum drawdown as a fraction (0.0-1.0)
        """
        if not self._portfolio_equity:
            return 0.0
            
        drawdowns = [entry['drawdown'] for entry in self._portfolio_equity]
        return max(drawdowns) if drawdowns else 0.0
    
    def get_risk_events(self) -> List[Dict[str, Any]]:
        """
        Get list of risk breach events
        
        Returns:
            List[Dict]: Risk events that occurred during backtest
        """
        return self._risk_events
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        Check backtest position limits
        
        Args:
            positions: Current positions
            
        Returns:
            True if positions are within limits, False otherwise
        """
        try:
            # Check total position value
            total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
            max_total = self.get_risk_limit('max_total_position', float('inf'))
            
            if total_value > max_total:
                self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
                self._risk_breached = True
                return False
            
            # Check if any single position exceeds limit
            for symbol, pos in positions.items():
                position_value = abs(pos.get('value', 0))
                max_position = self.get_risk_limit('max_position_value', float('inf'))
                
                if position_value > max_position:
                    self._logger.warning(
                        f"Position value for {symbol} ({position_value:.2f}) exceeds maximum ({max_position:.2f})"
                    )
                    self._risk_breached = True
                    return False
                    
                # Check concentration only if we have positions
                if total_value > 0:
                    concentration = position_value / total_value
                    max_concentration = self.get_risk_limit('max_concentration', 1.0)
                    
                    if concentration > max_concentration:
                        self._logger.warning(
                            f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                        )
                        self._risk_breached = True
                        return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error checking position limits: {e}")
            return True  # Don't block trading due to calculation errors
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate current system drawdown
        
        Returns:
            Current drawdown percentage (as decimal)
        """
        if not self._portfolio_equity:
            return 0.0
            
        current_equity = self._portfolio_equity[-1]['equity']
        
        if self._max_equity <= 0:
            return 0.0
            
        return (self._max_equity - current_equity) / self._max_equity


class LiveRiskManager(BaseRiskManager):
    """
    Risk management implementation for backtesting and paper trading
    
    Provides appropriate risk controls and signal validation
    with a focus on performance and realistic simulation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the backtest risk manager
        
        Args:
            config: Risk management configuration
        """
        super().__init__(config)
        
        # Backtest-specific state
        self._portfolio_equity = []
        self._max_equity = 0
        self._current_equity = 0
        self._current_positions = {}
        self._risk_events = []
    
    async def initialize(self) -> None:
        """Initialize the risk manager"""
        await super().initialize()
        initial_capital = self._config.get("trading", {}).get("capital", {}).get("initial", 100000)
        self._current_equity = float(initial_capital)  # Ensure it's a float
        self._max_equity = self._current_equity
        
        # Add initial equity point
        self._portfolio_equity.append({
            'timestamp': pd.Timestamp.now(),
            'equity': initial_capital,
            'drawdown': 0.0
        })
        
        self._logger.info(f"BacktestRiskManager initialized with {initial_capital} initial capital")
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate trading signals for backtesting
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            pd.DataFrame: Validated trading signals
        """
        if signals.empty:
            return signals
            
        # Apply risk filtering to create validated signals
        valid_signals = signals.copy()
        
        try:
            # Check if risk is already breached (stop trading if true)
            if self._risk_breached:
                self._logger.warning("Risk limits already breached, rejecting all new long positions")
                # When risk is breached, only allow closing positions (sell signals)
                valid_signals = valid_signals[valid_signals['action'].str.lower() == 'sell'].copy()
                return valid_signals
            
            # Apply signal filters one by one
            drop_indices = []
            
            for idx, signal in valid_signals.iterrows():
                action = signal['action'].lower() if 'action' in signal else 'unknown'
                symbol = signal['symbol'] if 'symbol' in signal else 'unknown'
                
                # Skip invalid signals
                if action not in ['buy', 'sell']:
                    self._logger.warning(f"Unknown action type: {action}, skipping")
                    drop_indices.append(idx)
                    continue
                    
                # Apply position limits for buy orders only (don't restrict selling)
                if action == 'buy':
                    # Apply position allocation limit
                    max_allocation = self.get_risk_limit('max_position_size', 0.1)
                    if max_allocation < 1.0 and 'quantity' in signal and 'price' in signal:
                        # Calculate allocation as fraction of total portfolio
                        new_allocation = signal['quantity'] * signal['price'] / self._current_equity
                        
                        if new_allocation > max_allocation:
                            self._logger.warning(
                                f"Signal for {symbol} exceeds max position allocation "
                                f"({new_allocation:.2%} > {max_allocation:.2%})"
                            )
                            drop_indices.append(idx)
                            continue
                            
                    # Check max open positions
                    max_positions = self.get_risk_limit('max_open_positions', 10)
                    if len(self._current_positions) >= max_positions:
                        self._logger.warning(
                            f"Signal for {symbol} rejected: max positions limit reached "
                            f"({len(self._current_positions)} >= {max_positions})"
                        )
                        drop_indices.append(idx)
                        continue
            
            # Drop rejected signals
            if drop_indices:
                valid_signals = valid_signals.drop(drop_indices)
            
            # Update signal quantity based on risk limits
            if 'quantity' in valid_signals.columns and 'price' in valid_signals.columns:
                for idx, signal in valid_signals.iterrows():
                    if signal['action'].lower() == 'buy':
                        # Calculate risk-adjusted position size
                        risk_per_trade = self.get_risk_limit('risk_per_trade', 0.01)
                        adjusted_quantity = risk_per_trade * self._current_equity / signal['price']
                        
                        # Ensure within min/max quantity
                        min_quantity = self.get_risk_limit('min_trade_quantity', 0.001)
                        max_quantity = self.get_risk_limit('max_trade_quantity', float('inf'))
                        
                        valid_signals.at[idx, 'quantity'] = max(
                            min_quantity, 
                            min(adjusted_quantity, max_quantity, signal['quantity'])
                        )
            
            return valid_signals
            
        except Exception as e:
            self._logger.error(f"Error during signal validation: {e}")
            return pd.DataFrame()
    
    async def execute_risk_control(self) -> None:
        """Execute risk control logic for backtest"""
        # In backtest mode, just update drawdown calculations
        if self._portfolio_equity:
            current_equity = self._portfolio_equity[-1]['equity']
            if current_equity > self._max_equity:
                self._max_equity = current_equity
            
            # Calculate current drawdown
            if self._max_equity > 0:
                current_drawdown = (self._max_equity - current_equity) / self._max_equity
                # Update state
                if current_drawdown > 0:
                    self._portfolio_equity[-1]['drawdown'] = current_drawdown
                    
                # Check if max drawdown is breached
                max_drawdown_limit = self.get_risk_limit('max_drawdown', 0.25)
                if current_drawdown >= max_drawdown_limit:
                    if not self._risk_breached:
                        self._risk_breached = True
                        self._risk_events.append({
                            'timestamp': datetime.now(),
                            'type': 'drawdown_breach',
                            'value': current_drawdown,
                            'limit': max_drawdown_limit
                        })
                        self._logger.warning(
                            f"Max drawdown limit breached: {current_drawdown:.2%} >= {max_drawdown_limit:.2%}"
                        )
    
    def update_portfolio_state(self, equity: float, positions: Dict[str, Any], timestamp: Any = None) -> None:
        """
        Update portfolio state for risk calculations
        
        Args:
            equity: Current portfolio equity
            positions: Current positions
            timestamp: Current timestamp (optional)
        """
        # Default timestamp to now if not provided
        if timestamp is None:
            timestamp = datetime.now()
            
        # Update equity curve
        self._current_equity = equity
        
        # Calculate drawdown
        drawdown = 0.0
        if self._max_equity > 0:
            drawdown = (self._max_equity - equity) / self._max_equity
            
        # Update max equity
        if equity > self._max_equity:
            self._max_equity = equity
            
        # Add new equity point
        self._portfolio_equity.append({
            'timestamp': timestamp,
            'equity': equity,
            'drawdown': drawdown
        })
        
        # Update positions
        self._current_positions = positions
        
        # Check position limits
        position_value = sum(p.get('value', 0) for p in positions.values())
        max_position_value = self.get_risk_limit('max_position_value', equity * 0.9)  # Default 90% of equity
        
        if position_value > max_position_value:
            self._risk_breached = True
            self._risk_events.append({
                'timestamp': timestamp,
                'type': 'position_limit_breach',
                'value': position_value,
                'limit': max_position_value
            })
            self._logger.warning(
                f"Total position value exceeds limit: {position_value:.2f} > {max_position_value:.2f}"
            )
    
    def get_max_drawdown(self) -> float:
        """
        Get maximum drawdown experienced during backtest
        
        Returns:
            float: Maximum drawdown as a fraction (0.0-1.0)
        """
        if not self._portfolio_equity:
            return 0.0
            
        drawdowns = [entry['drawdown'] for entry in self._portfolio_equity]
        return max(drawdowns) if drawdowns else 0.0
    
    def get_risk_events(self) -> List[Dict[str, Any]]:
        """
        Get list of risk breach events
        
        Returns:
            List[Dict]: Risk events that occurred during backtest
        """
        return self._risk_events
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        Check backtest position limits
        
        Args:
            positions: Current positions
            
        Returns:
            True if positions are within limits, False otherwise
        """
        try:
            # Check total position value
            total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
            max_total = self.get_risk_limit('max_total_position', float('inf'))
            
            if total_value > max_total:
                self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
                self._risk_breached = True
                return False
            
            # Check if any single position exceeds limit
            for symbol, pos in positions.items():
                position_value = abs(pos.get('value', 0))
                max_position = self.get_risk_limit('max_position_value', float('inf'))
                
                if position_value > max_position:
                    self._logger.warning(
                        f"Position value for {symbol} ({position_value:.2f}) exceeds maximum ({max_position:.2f})"
                    )
                    self._risk_breached = True
                    return False
                    
                # Check concentration only if we have positions
                if total_value > 0:
                    concentration = position_value / total_value
                    max_concentration = self.get_risk_limit('max_concentration', 1.0)
                    
                    if concentration > max_concentration:
                        self._logger.warning(
                            f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                        )
                        self._risk_breached = True
                        return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error checking position limits: {e}")
            return True  # Don't block trading due to calculation errors
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate current system drawdown
        
        Returns:
            Current drawdown percentage (as decimal)
        """
        if not self._portfolio_equity:
            return 0.0
            
        current_equity = self._portfolio_equity[-1]['equity']
        
        if self._max_equity <= 0:
            return 0.0
            
        return (self._max_equity - current_equity) / self._max_equity