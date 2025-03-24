# src/risk/risk_manager.py

import pandas as pd
import asyncio
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod

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
        # Import implementations here to avoid circular imports
        from src.risk.backtest_risk_manager import BacktestRiskManager
        from src.risk.live_risk_manager import LiveRiskManager
        
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