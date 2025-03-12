# risk_manager.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any, Optional, Set
import asyncio

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
        self._logger = LogManager.get_logger(f"mode.{self.__class__.__name__.lower()}")
        self._config = config or {}
        self._risk_limits = self._config.get('risk_limits', {})
        self._active_controls: Set[str] = set()
        self._initialized = False
        self._risk_breached = False
    
    async def initialize(self) -> None:
        """
        Initialize the risk manager
        Subclasses should call super().initialize() when overriding
        """
        self._initialized = True
        self._logger.info(f"{self.__class__.__name__} initialized")
    
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

class BacktestRiskManager(BaseRiskManager):
    """
    Backtesting-specific risk management implementation
    Simplifies certain risk checks while maintaining compatible interface
    """
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate backtest signals
        
        Args:
            signals: Trading signals to validate
            
        Returns:
            Validated signals
        """
        if signals.empty:
            return pd.DataFrame()
        
        # Apply basic risk filtering logic
        try:
            # Filter out invalid signals
            valid_signals = signals.copy()
            
            # Check for risk breaches that would prevent new positions
            max_drawdown = self.get_risk_limit('max_drawdown', -1.0)
            if max_drawdown > 0 and hasattr(self, '_calculate_current_drawdown'):
                current_drawdown = self._calculate_current_drawdown()
                if current_drawdown >= max_drawdown:
                    self._risk_breached = True
                    self._logger.warning(f"Max drawdown limit reached: {current_drawdown:.2%} >= {max_drawdown:.2%}")
                    # Only allow closing positions when risk is breached
                    valid_signals = valid_signals[valid_signals['action'].str.lower() == 'sell']
            
            # Apply position size limits
            if 'quantity' in valid_signals.columns:
                max_position_size = self.get_risk_limit('max_position_size', float('inf'))
                valid_signals.loc[valid_signals['quantity'] > max_position_size, 'quantity'] = max_position_size
            
            self._logger.debug(f"Validated {len(valid_signals)} of {len(signals)} signals")
            return valid_signals
            
        except Exception as e:
            self._logger.error(f"Error validating signals: {e}")
            return pd.DataFrame()  # Return empty frame on error
    
    async def execute_risk_control(self) -> None:
        """
        Execute risk control in backtest
        In backtest mode, this is a simplified implementation
        """
        # Most controls are passive in backtest mode
        # Most control is done in validate_signals
        pass
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        Check backtest position limits
        
        Args:
            positions: Current backtest positions
            
        Returns:
            True if positions are within limits, False otherwise
        """
        # Check total position value
        total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        max_total = self.get_risk_limit('max_total_position', float('inf'))
        
        if total_value > max_total:
            self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
            self._risk_breached = True
            return False
        
        # Check single asset concentration
        for symbol, pos in positions.items():
            concentration = abs(pos.get('value', 0)) / total_value if total_value > 0 else 0
            max_concentration = self.get_risk_limit('max_concentration', 1.0)
            
            if concentration > max_concentration:
                self._logger.warning(
                    f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                )
                self._risk_breached = True
                return False
        
        return True
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate current system drawdown
        
        Returns:
            Current drawdown percentage (as decimal)
        """
        # In a real implementation, this would calculate based on equity curve
        # This is a simplified version
        return 0.0

class LiveRiskManager(BaseRiskManager):
    """
    Live trading-specific risk management implementation
    Includes stricter risk controls and real-time alerts
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._position_history = []
        self._market_data_cache = {}
        self._alert_callbacks = []
    
    async def initialize(self) -> None:
        """Initialize live risk manager"""
        await super().initialize()
        
        # Set up default controls
        default_controls = [
            'max_drawdown', 'max_exposure', 'volatility_control',
            'correlation_control', 'liquidity_control'
        ]
        
        for control in default_controls:
            if self.get_risk_limit(control, None) is not None:
                self.activate_control(control)
    
    async def validate_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Validate live trading signals
        
        Args:
            signals: Signals to validate
            
        Returns:
            Validated signals
            
        Raises:
            RiskValidationError: If signals have serious risk issues
        """
        if signals.empty:
            return signals
        
        valid_signals = signals.copy()
        
        # Check global risk state
        if self.is_control_active('emergency_stop') and self.get_risk_limit('emergency_stop', False):
            self._risk_breached = True
            raise RiskValidationError("Emergency stop is active - all trading suspended")
        
        # Apply risk checks to each signal
        to_drop = []
        for idx, signal in valid_signals.iterrows():
            try:
                # Basic validation
                if not self._is_valid_signal(signal):
                    to_drop.append(idx)
                    continue
                
                # Check each active risk control
                for control in self._active_controls:
                    control_method = getattr(self, f"_check_{control}", None)
                    if control_method and callable(control_method):
                        result = await control_method(signal)
                        if result is None:  # Signal rejected
                            to_drop.append(idx)
                            break
                            
            except Exception as e:
                self._logger.error(f"Error during risk validation: {str(e)}")
                to_drop.append(idx)
        
        # Drop rejected signals
        valid_signals = valid_signals.drop(to_drop)
        self._logger.info(f"Validated {len(valid_signals)} of {len(signals)} signals")
        return valid_signals
    
    def _is_valid_signal(self, signal: pd.Series) -> bool:
        """Check if signal format is valid"""
        required_fields = {'symbol', 'action'}
        return all(field in signal for field in required_fields)
    
    async def _check_max_exposure(self, signal: pd.Series) -> Optional[pd.Series]:
        """
        Check if signal exceeds maximum risk exposure
        
        Args:
            signal: Signal to check
            
        Returns:
            Modified signal or None if rejected
        """
        max_exposure = self.get_risk_limit('max_exposure', float('inf'))
        
        # Calculate current total exposure
        current_exposure = self._calculate_total_exposure()
        
        # Calculate new signal's incremental exposure
        signal_exposure = 0
        if 'quantity' in signal and 'price' in signal:
            signal_exposure = abs(signal['quantity'] * signal['price'])
        
        if current_exposure + signal_exposure > max_exposure:
            self._logger.warning(
                f"Signal for {signal['symbol']} rejected: would exceed max exposure "
                f"({current_exposure + signal_exposure} > {max_exposure})"
            )
            return None
        
        return signal
    
    async def _check_max_drawdown(self, signal: pd.Series) -> Optional[pd.Series]:
        """
        Check if maximum drawdown limit is exceeded
        
        Args:
            signal: Signal to check
            
        Returns:
            Signal if accepted, None if rejected
        """
        max_drawdown = self.get_risk_limit('max_drawdown', -1.0)
        if max_drawdown <= 0:  # No valid drawdown limit set
            return signal
        
        # Calculate current drawdown
        current_drawdown = self._calculate_current_drawdown()
        
        if current_drawdown >= max_drawdown:
            self._risk_breached = True
            self._logger.warning(
                f"Signal for {signal['symbol']} rejected: max drawdown limit reached "
                f"({current_drawdown:.2%} >= {max_drawdown:.2%})"
            )
            # Allow only risk-reducing trades when in drawdown
            if signal['action'].lower() == 'buy':
                return None
        
        return signal
    
    def _calculate_current_drawdown(self) -> float:
        """
        Calculate current system drawdown
        
        Returns:
            Current drawdown percentage (as decimal)
        """
        # In actual implementation, this would be calculated from equity curve
        # This is a simplified placeholder
        return 0.05  # Example: 5% drawdown
    
    def _calculate_total_exposure(self) -> float:
        """
        Calculate current total risk exposure
        
        Returns:
            Total exposure value
        """
        # In actual implementation, this would be from API or local position tracking
        # This is a simplified placeholder
        return sum(abs(pos.get('value', 0)) for pos in self._position_history)
    
    async def execute_risk_control(self) -> None:
        """
        Execute live risk control
        Check current market and position state and take corrective actions if needed
        """
        try:
            # Check global risk metrics
            await self._check_global_risk_metrics()
            
            # Check each control
            for control in self._active_controls:
                control_method = getattr(self, f"_execute_{control}_control", None)
                if control_method and callable(control_method):
                    await control_method()
            
        except Exception as e:
            self._logger.error(f"Error during risk control execution: {str(e)}")
    
    async def _check_global_risk_metrics(self) -> None:
        """Check global risk metrics and respond"""
        # Check account drawdown
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = self.get_risk_limit('max_drawdown', 1.0)
        
        if current_drawdown >= max_drawdown:
            self._risk_breached = True
            self._logger.warning(f"Maximum drawdown reached: {current_drawdown:.2%} >= {max_drawdown:.2%}")
            self.activate_control('emergency_stop')
            await self._send_alert(f"EMERGENCY: Maximum drawdown limit reached ({current_drawdown:.2%})")
        
        # Other global risk metric checks...
    
    async def _execute_volatility_control(self) -> None:
        """Execute volatility control logic"""
        if 'volatility_control' not in self._active_controls:
            return
        
        # Get current market volatility
        market_volatility = await self._get_market_volatility()
        vol_threshold = self.get_risk_limit('volatility_threshold', 0.15)
        
        if market_volatility > vol_threshold:
            self._logger.warning(f"High market volatility detected: {market_volatility:.2%} > {vol_threshold:.2%}")
            
            # Execute volatility response strategy
            vol_response = self.get_risk_limit('volatility_response', 'reduce')
            
            if vol_response == 'halt':
                self._risk_breached = True
                self.activate_control('emergency_stop')
                await self._send_alert("EMERGENCY: Trading halted due to high market volatility")
            elif vol_response == 'reduce':
                # Implement position reduction strategy
                self._logger.info("Implementing position reduction due to high volatility")
                # Actual implementation would call trading API to reduce positions
    
    async def _get_market_volatility(self) -> float:
        """
        Get current market volatility
        
        Returns:
            Market volatility value
        """
        # In actual implementation, this would be from market data provider
        # This is a simplified placeholder
        return 0.08  # Example: 8% volatility
    
    async def _send_alert(self, message: str) -> None:
        """
        Send risk alert
        
        Args:
            message: Alert message
        """
        self._logger.critical(f"RISK ALERT: {message}")
        
        # Execute all registered alert callbacks
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self._logger.error(f"Error in alert callback: {str(e)}")
    
    def register_alert_callback(self, callback: callable) -> None:
        """
        Register alert callback
        
        Args:
            callback: Function to call when alert is triggered
        """
        self._alert_callbacks.append(callback)
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        Check live position limits
        
        Args:
            positions: Current live positions
            
        Returns:
            True if positions are within limits, False otherwise
        """
        # Check total position value
        total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        max_total = self.get_risk_limit('max_total_position', float('inf'))
        
        if total_value > max_total:
            self._risk_breached = True
            self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
            await self._send_alert(f"Position limit exceeded: total value {total_value} > {max_total}")
            return False
        
        # Check single asset concentration
        for symbol, pos in positions.items():
            concentration = abs(pos.get('value', 0)) / total_value if total_value > 0 else 0
            max_concentration = self.get_risk_limit('max_concentration', 1.0)
            
            if concentration > max_concentration:
                self._risk_breached = True
                self._logger.warning(
                    f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                )
                await self._send_alert(
                    f"Concentration limit exceeded: {symbol} at {concentration:.2%} > {max_concentration:.2%}"
                )
                return False
        
        return True