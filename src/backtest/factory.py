#!/usr/bin/env python3
# src/backtest/factory.py

"""
Backtest engine factory module.
Provides factory methods for creating backtest engine instances according to the factory pattern standard.
"""

from enum import Enum
from typing import Dict, Optional, Any, Type, List

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.backtest.base import BaseBacktestEngine


class BacktestEngine(Enum):
    """Centralize the definition of backtest engine types"""
    MARKETREPLAY = "market_replay"
    OHLCV = "ohlcv"
    
    
class BacktestEngineFactory(AbstractFactory):
    """
    Factory for creating backtest engines following the standard factory pattern.
    
    This factory creates and manages instances of backtest engines, providing
    consistent interface for engine creation, discovery, and metadata access.
    """
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """
        Initialize backtest engine factory
        
        Args:
            config: Configuration manager instance
        """
        super().__init__(config)
        
        # Initialize logger with proper category
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Register built-in engines
        self._register_default_engines()
        
        # Auto-discover additional engines
        self._discover_backtest_engines()
    
    def _register_default_engines(self) -> None:
        """Register default backtest engines with consistent metadata"""
        # Register OHLCV backtest engine
        self.register(
            BacktestEngine.OHLCV.value,
            "src.backtest.engine.ohlcv.OHLCVEngine", 
            {
                "description": "OHLCV Backtest Engine for vectorized backtesting",
                "category": "backtest",
                "features": ["vectorized", "ohlcv", "performance_optimized"]
            }
        )
        
        # Register Market Replay engine
        self.register(
            BacktestEngine.MARKETREPLAY.value,
            "src.backtest.engine.market_replay.MarketReplayEngine", 
            {
                "description": "Market Replay Engine for sequential data processing",
                "category": "backtest",
                "features": ["sequential", "realistic_execution", "detailed_simulation"]
            }
        )
        
        # Optional: Register any other built-in engines
        self.register(
            "event_driven",  # Consider adding to BacktestEngine enum if widely used
            "src.backtest.engine.event_driven.EventDrivenEngine", 
            {
                "description": "Event-Driven Backtest Engine for complex event processing",
                "category": "backtest",
                "features": ["event_based", "advanced_simulation", "multi_asset"]
            }
        )
        
        self.logger.info("Registered default backtest engines")
    
    def _discover_backtest_engines(self) -> None:
        """Auto-discover additional backtest engine modules"""
        try:
            # Discover implementations from the engines directory
            module_path = "src.backtest"
            self.discover_registrable_classes(
                BaseBacktestEngine, 
                module_path, 
                "backtest_engine_factory"
            )
            self.logger.debug(f"Discovered additional backtest engines from {module_path}")
        except Exception as e:
            self.logger.error(f"Error during backtest engine discovery: {str(e)}")
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve engine name, falling back to default from config if none provided
        
        Args:
            name: Engine name or None
            
        Returns:
            str: Resolved engine name
            
        Raises:
            ValueError: If no valid engine name could be resolved
        """
        # If name is provided, normalize it
        if name:
            return name.lower()
            
        # Get default from config with hierarchical path
        default_engine = self.config.get("backtest", "engine", "default", default=BacktestEngine.OHLCV.value)
        
        if not default_engine:
            raise ValueError("No engine name provided and no default engine configured")
            
        self.logger.debug(f"Using default backtest engine: {default_engine}")
        return default_engine.lower()
    
    async def _get_concrete_class(self, name: str) -> Type[BaseBacktestEngine]:
        """
        Get concrete backtest engine class by name
        
        Args:
            name: Engine name
            
        Returns:
            Type[BaseBacktestEngine]: Engine class
            
        Raises:
            ComponentLoadError: If the component could not be loaded
        """
        return await self._load_class_from_path(name, BaseBacktestEngine)
    
    def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of all available backtest engines with metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Engine names mapped to their metadata
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = metadata
            
        # Ensure all enum values are included
        for engine in BacktestEngine:
            if engine.value not in result:
                result[engine.value] = {
                    "description": self._get_default_description(engine),
                    "features": [],
                    "category": "backtest"
                }
                
        return result
    
    def _get_default_description(self, engine: BacktestEngine) -> str:
        """
        Get default description for backtest engine when not available
        
        Args:
            engine: Backtest engine enum
            
        Returns:
            str: Default description
        """
        descriptions = {
            BacktestEngine.OHLCV: "OHLCV Backtest Engine for vectorized backtesting",
            BacktestEngine.MARKETREPLAY: "Market Replay Engine for sequential data processing"
        }
        return descriptions.get(engine, "Unknown backtest engine")
    
    def get_engine_features(self, engine_name: str) -> List[str]:
        """
        Get features of a specific backtest engine
        
        Args:
            engine_name: Name of the engine
            
        Returns:
            List[str]: List of features
        """
        metadata = self._metadata.get(engine_name.lower(), {})
        return metadata.get('features', [])
    
    async def create_backtest_engine(self, name: Optional[str] = None) -> BaseBacktestEngine:
        """
        Create a backtest engine with parameters from configuration
        
        Args:
            name: Optional engine name
            
        Returns:
            BaseBacktestEngine: Backtest engine instance
        """
        resolved_name = await self._resolve_name(name)
        
        # Get engine-specific configuration
        engine_config = self.config.get("backtest", "engines", resolved_name, default={})
        
        return await self.create(resolved_name, params=engine_config)


def get_backtest_engine_factory(config: ConfigManager) -> BacktestEngineFactory:
    """
    Get or create singleton instance of BacktestEngineFactory
    
    Args:
        config: Configuration manager
    
    Returns:
        BacktestEngineFactory: Singleton instance
    """
    return BacktestEngineFactory.get_instance(config)