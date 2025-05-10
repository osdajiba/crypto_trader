#!/usr/bin/env python3
# src/strategy/factory.py

from enum import Enum
from typing import Dict, Optional, Any, Type, List, Union
import asyncio

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.base import BaseStrategy


class Strategy(Enum):
    """Enumeration of built-in strategy types"""
    DUAL_MA = "dual_ma"
    MULTI_FACTORS = "multi_factors"
    NEURAL_NETWORK = "neural_network"
    

class StrategyFactory(AbstractFactory):
    """
    Factory for creating trading strategy instances.
    
    Provides a unified interface for registering, discovering, and creating
    strategy instances with automatic configuration management.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize strategy factory.
        
        Args:
            config: Configuration manager instance
        """
        super().__init__(config)
        
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Get default strategy from config
        self.default_strategy_type = config.get("strategy", "active", default="dual_ma")
        
        # Register built-in strategies and discover custom ones
        self._register_default_strategies()
        self._discover_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register built-in strategy types with metadata."""
        # Dual Moving Average Strategy
        self.register(Strategy.DUAL_MA.value, "src.strategy.implementations.dual_ma.DualMAStrategy", {
            "description": "Dual Moving Average Crossover Strategy",
            "features": ["moving_averages", "crossover", "trend_following"],
            "category": "trend",
            "parameters": [
                {"name": "short_window", "type": "int", "default": 20, "description": "Short moving average period"},
                {"name": "long_window", "type": "int", "default": 60, "description": "Long moving average period"},
                {"name": "signal_threshold", "type": "float", "default": 0.005, "description": "Signal generation threshold"},
                {"name": "position_size", "type": "float", "default": 0.01, "description": "Position size as fraction of capital"},
                {"name": "use_risk_based_sizing", "type": "bool", "default": True, "description": "Use risk-based position sizing"}
            ]
        })
        
        # Multi-Factor Strategy
        self.register(Strategy.MULTI_FACTORS.value, "src.strategy.implementations.multi_factors.MultiFactorsStrategy", {
            "description": "Multi-Factor Strategy with custom factor combinations",
            "features": ["multi_factor", "customizable", "technical_indicators"],
            "category": "custom",
            "parameters": [
                {"name": "factors", "type": "list", "default": ["rsi", "macd", "bollinger"], "description": "List of factors to use"},
                {"name": "weights", "type": "list", "default": [0.4, 0.3, 0.3], "description": "Weights for each factor"},
                {"name": "signal_threshold", "type": "float", "default": 0.1, "description": "Threshold for signal generation"}
            ]
        })
        
        # Neural Network Strategy
        self.register(Strategy.NEURAL_NETWORK.value, "src.strategy.implementations.neural_network.NeuralNetworkStrategy", {
            "description": "Neural Network based prediction strategy",
            "features": ["machine_learning", "prediction", "advanced"],
            "category": "ml",
            "parameters": [
                {"name": "epochs", "type": "int", "default": 100, "description": "Training epochs"},
                {"name": "batch_size", "type": "int", "default": 32, "description": "Training batch size"},
                {"name": "layers", "type": "list", "default": [64, 32, 16], "description": "Hidden layer sizes"},
                {"name": "dropout", "type": "float", "default": 0.2, "description": "Dropout rate"}
            ]
        })
    
    def _discover_strategies(self) -> None:
        """Auto-discover additional strategy modules from implementations directory."""
        try:
            strategy_dir = "src.strategy.implementations"
            self.discover_registrable_classes(BaseStrategy, strategy_dir, "strategy_factory")
            self.logger.info(f"Discovered {len(self._registry)} strategy implementations")
        except Exception as e:
            self.logger.error(f"Error auto-discovering strategies: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseStrategy]:
        """
        Get the strategy class for the given name.
        
        Args:
            name: Strategy name to load
            
        Returns:
            Type[BaseStrategy]: Strategy class
        """
        return await self._load_class_from_path(name, BaseStrategy)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Validate and resolve strategy name.
        
        Args:
            name: Requested strategy name or None
            
        Returns:
            str: Resolved strategy name
            
        Raises:
            ValueError: If strategy not found and no fallback available
        """
        # Use provided name or default from config
        resolved_name = (name or self.default_strategy_type).lower()
        
        # Check if strategy exists
        if resolved_name not in self._registry:
            available_strategies = list(self._registry.keys())
            self.logger.warning(f"Strategy '{resolved_name}' not found, available strategies: {available_strategies}")
            
            # Try fallback strategy from config
            fallback_strategy = self.config.get("strategy", "fallback_strategy", default=None)
            if fallback_strategy and fallback_strategy in self._registry:
                self.logger.info(f"Using fallback strategy: {fallback_strategy}")
                return fallback_strategy
                
            # Try default strategy
            if self.default_strategy_type in self._registry:
                self.logger.info(f"Using default strategy: {self.default_strategy_type}")
                return self.default_strategy_type
                
            # Last resort: use first available strategy
            if available_strategies:
                fallback = available_strategies[0]
                self.logger.info(f"Using first available strategy: {fallback}")
                return fallback
                
            # No strategies available
            raise ValueError(f"Strategy '{resolved_name}' not found and no fallbacks available")
            
        return resolved_name
    
    def get_available_strategies(self) -> List[str]:
        """
        Get list of all available strategy names.
        
        Returns:
            List[str]: List of strategy names
        """
        return list(self._registry.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dict[str, Any]: Strategy information
        """
        if strategy_name not in self._registry:
            return {"error": f"Strategy '{strategy_name}' not found"}
            
        metadata = self._metadata.get(strategy_name, {})
        return {
            "name": strategy_name,
            "class_path": self._registry[strategy_name],
            "description": metadata.get("description", "No description available"),
            "category": metadata.get("category", "uncategorized"),
            "features": metadata.get("features", []),
            "parameters": metadata.get("parameters", [])
        }
    
    def get_strategies_by_category(self, category: str) -> List[str]:
        """
        Get list of strategies filtered by category.
        
        Args:
            category: Category to filter by
            
        Returns:
            List[str]: List of strategy names in the category
        """
        result = []
        for name, metadata in self._metadata.items():
            if metadata.get("category", "").lower() == category.lower():
                result.append(name)
        return result
    
    async def create_strategy(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        Create a strategy instance.
        
        Args:
            name: Strategy name (optional, uses default if not provided)
            params: Strategy parameters (optional)
            
        Returns:
            BaseStrategy: Strategy instance
        """
        return await self.create(name, params)


def get_strategy_factory(config: ConfigManager) -> StrategyFactory:
    """
    Get or create singleton instance of StrategyFactory.
    
    Args:
        config: Configuration manager instance
        
    Returns:
        StrategyFactory: Singleton factory instance
    """
    return StrategyFactory.get_instance(config)