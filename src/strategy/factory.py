#!/usr/bin/env python3
# src/strategy/factory.py

from enum import Enum
from typing import Dict, Optional, Any, Type, List, Union

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.base import BaseStrategy


class Strategy(Enum):
    """Centralize the definition of backtest engine types"""
    DUAL_MA = "dual_ma"
    MF = "multi_factors"
    NN = "neural_network"
    
    
class StrategyFactory(AbstractFactory):
    """Factory for creating strategy instances"""
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """Initialize strategy factory"""
        super().__init__(config)
        
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        self.default_strategy_type = config.get("strategy", "active", default="dual_ma")
        
        # Register built-in strategies
        self._register_default_strategies()
        self._discover_strategies()
    
    def _register_default_strategies(self) -> None:
        """Register default strategies with consistent metadata"""
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
        
        self.register(Strategy.MF.value, "src.strategy.implementations.multi_factors.MultiFactorsStrategy", {
            "description": "Multi-Factor Strategy with custom factor combinations",
            "features": ["multi_factor", "customizable", "technical_indicators"],
            "category": "custom"
        })
        
        self.register(Strategy.NN.value, "src.strategy.implementations.neural_network.NeuralNetworkStrategy", {
            "description": "Neural Network based prediction strategy",
            "features": ["machine_learning", "prediction", "advanced"],
            "category": "ml"
        })
    
    def _discover_strategies(self) -> None:
        """Auto-discover strategy modules"""
        try:
            strategy_dir = "src.strategy.implementations"
            self.discover_registrable_classes(BaseStrategy, strategy_dir, "strategy_factory")
        except Exception as e:
            self.logger.error(f"Error auto-discovering strategies: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class"""
        return await self._load_class_from_path(name, BaseStrategy)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """Validate and resolve strategy name"""
        resolved_name = (name or self.default_strategy_type).lower()
        
        if resolved_name not in self._registry:
            available_strategies = list(self._registry.keys())
            self.logger.warning(f"Strategy '{resolved_name}' not found, available strategies: {available_strategies}")
            
            # Fall back to default if specified strategy not found
            if resolved_name != self.default_strategy_type and self.default_strategy_type in self._registry:
                self.logger.info(f"Falling back to default strategy: {self.default_strategy_type}")
                return self.default_strategy_type
            
            # If default not found either, use first available strategy
            if available_strategies:
                fallback = available_strategies[0]
                self.logger.info(f"Using first available strategy: {fallback}")
                return fallback
                
            raise ValueError(f"Strategy '{resolved_name}' not found and no fallbacks available")
            
        return resolved_name
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self._registry.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific strategy"""
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
        """Get strategies filtered by category"""
        result = []
        for name, metadata in self._metadata.items():
            if metadata.get("category", "").lower() == category.lower():
                result.append(name)
        return result
    
    async def create_strategy(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        Create a strategy instance
        
        Args:
            name: Strategy name (optional, uses default if not provided)
            params: Strategy parameters (optional)
            
        Returns:
            BaseStrategy: Strategy instance
        """
        return await self.create(name, params)


def get_strategy_factory(config: ConfigManager) -> StrategyFactory:
    """Get or create singleton instance of StrategyFactory"""
    return StrategyFactory.get_instance(config)