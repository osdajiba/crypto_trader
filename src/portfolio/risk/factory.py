#!/usr/bin/env python3
# src/portfolio/risk/factory.py

"""
Risk manager factory module.
Provides factory methods for creating risk manager instances according to the factory pattern standard.
"""

from enum import Enum
from typing import Dict, Optional, Any, Type, List

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.risk.base import BaseRiskManager


class RiskManager(Enum):
    """Centralize the definition of backtest engine types"""
    BACKTEST = "backtest"
    STANDARD = "standard"
    CONSERVATIVE = "conservative"
    

class RiskManagerFactory(AbstractFactory):
    """
    Factory for creating risk manager instances
    
    This factory creates and manages instances of risk managers, providing
    consistent interface for risk manager creation, discovery, and metadata access.
    """
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """
        Initialize risk manager factory
        
        Args:
            config: Configuration manager instance
        """
        super().__init__(config)
        
        # Initialize logger with proper category
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Register built-in risk managers
        self._register_default_risk_managers()
        
        # Auto-discover additional risk managers
        self._discover_risk_managers()
    
    def _register_default_risk_managers(self) -> None:
        """Register default risk managers with consistent metadata"""
        self.register(
            RiskManager.STANDARD.value, 
            "src.portfolio.risk.standard.StandardRiskManager",
            {
                "description": "Standard risk management implementation",
                "features": ["position_limits", "drawdown_control", "leverage_limits"],
                "category": "risk"
            }
        )
        
        self.register(
            RiskManager.BACKTEST.value, 
            "src.portfolio.risk.backtest.BacktestRiskManager",
            {
                "description": "Simplified risk management for backtesting",
                "features": ["position_limits", "basic_validation"],
                "category": "risk"
            }
        )
        
        self.register(
            RiskManager.CONSERVATIVE.value,
            "src.portfolio.risk.conservative.ConservativeRiskManager",
            {
                "description": "Conservative risk management with stricter limits",
                "features": ["position_limits", "drawdown_control", "leverage_limits", "volatility_control"],
                "category": "risk"
            }
        )
        
        self.logger.info("Registered default risk managers")
    
    def _discover_risk_managers(self) -> None:
        """Auto-discover additional risk manager implementations"""
        try:
            # Discover implementations from implementations directory
            module_path = "src.portfolio.risk"
            self.discover_registrable_classes(
                BaseRiskManager, 
                module_path, 
                "risk_manager_factory"
            )
            self.logger.debug(f"Discovered additional risk managers from {module_path}")
        except Exception as e:
            self.logger.error(f"Error during risk manager discovery: {str(e)}")
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve risk manager name, falling back to default from config if none provided
        
        Args:
            name: Risk manager name or None
            
        Returns:
            str: Resolved risk manager name
            
        Raises:
            ValueError: If no valid risk manager name could be resolved
        """
        if name:
            return name.lower()
            
        # Get default from config based on operational mode
        op_mode = self.config.get("system", "operational_mode", default="backtest")
        
        # Default risk managers for different modes
        if op_mode.lower() == "backtest":
            default_risk_manager = RiskManager.BACKTEST.value
        elif op_mode.lower() == "live":
            default_risk_manager = RiskManager.CONSERVATIVE.value
        else:
            default_risk_manager = RiskManager.STANDARD.value
            
        # Override from config if specified
        default_risk_manager = self.config.get("risk", "default_manager", default=default_risk_manager)
        
        if not default_risk_manager:
            raise ValueError("No risk manager name provided and no default risk manager configured")
            
        self.logger.debug(f"Using default risk manager: {default_risk_manager}")
        return default_risk_manager.lower()
    
    async def _get_concrete_class(self, name: str) -> Type[BaseRiskManager]:
        """
        Get concrete risk manager class by name
        
        Args:
            name: Risk manager name
            
        Returns:
            Type[BaseRiskManager]: Risk manager class
            
        Raises:
            ComponentLoadError: If the component could not be loaded
        """
        return await self._load_class_from_path(name, BaseRiskManager)
    
    def get_available_risk_managers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of all available risk managers with metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Risk manager names mapped to their metadata
        """
        return self.get_registered_items()
    
    def get_risk_manager_features(self, risk_manager_name: str) -> List[str]:
        """
        Get features of a specific risk manager
        
        Args:
            risk_manager_name: Name of the risk manager
            
        Returns:
            List[str]: List of features
        """
        metadata = self._metadata.get(risk_manager_name.lower(), {})
        return metadata.get('features', [])
    
    async def create_with_config_params(self, 
                                      portfolio_manager,
                                      name: Optional[str] = None, 
                                      ) -> BaseRiskManager:
        """
        Create a risk manager with parameters from configuration
        
        Args:
            name: Optional risk manager name
            portfolio_manager: Optional portfolio manager instance
            
        Returns:
            BaseRiskManager: Risk manager instance
        """
        resolved_name = await self._resolve_name(name)
        
        # Get risk manager-specific configuration
        risk_params = self.config.get("risk", "managers", resolved_name, default={})
        
        # Set portfolio manager in params
        params = {"portfolio_manager": portfolio_manager}
        
        # Add config params
        if risk_params:
            params.update(risk_params)
            
        return await self.create(resolved_name, params=params)


def get_risk_factory(config: ConfigManager) -> RiskManagerFactory:
    """
    Get or create singleton instance of RiskManagerFactory
    
    Args:
        config: Configuration manager
    
    Returns:
        RiskManagerFactory: Singleton instance
    """
    return RiskManagerFactory.get_instance(config)