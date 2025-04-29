#!/usr/bin/env python3
# src/backtest/factory.py

from typing import Dict, Any, Optional, Type
import importlib

from src.common.abstract_factory import AbstractFactory
from src.common.config import ConfigManager
from src.common.log_manager import LogManager


class BacktestFactory(AbstractFactory):
    """Factory for creating backtest engines"""
    
    def __init__(self, config: ConfigManager):
        """Initialize backtest factory"""
        super().__init__(config)
        self.logger = LogManager.get_logger("backtest.factory")
        self.default_engine = config.get("backtest", "engine", default="standard")
        
        # Register default backtest engines
        self._register_default_engines()
    
    def _register_default_engines(self):
        """Register default backtest engines"""
        # Register standard backtest engine
        self.register("standard", "src.backtest.engine.base.BaseBacktestEngine", {
            "description": "Standard Backtest Engine with factor-based data management"
        })
        
        # Register additional engines
        self.register("ohlcv", "src.backtest.engine.ohlcv.OHLCVEngine", {
            "description": "OHLCV Backtest Engine for vectorized backtesting"
        })
        self.register("market_replay", "src.backtest.engine.market_replay.MarketReplayEngine", {
            "description": "Market Replay Engine for sequential data processing"
        })
    
    async def _get_concrete_class(self, name: str):
        """Get concrete backtest engine class"""
        try:
            # Try loading by name
            if name in self._registry:
                class_path = self._registry[name]
                
                if isinstance(class_path, str):
                    # Load from path
                    module_path, class_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    return getattr(module, class_name)
                else:
                    # Return already loaded class
                    return class_path
            
            self.logger.error(f"Unknown backtest engine: {name}")
            raise ValueError(f"Unknown backtest engine: {name}")
            
        except (ImportError, AttributeError) as e:
            self.logger.error(f"Error loading backtest engine '{name}': {e}")
            raise
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """Resolve engine name with default fallback"""
        name = name or self.default_engine
        if not name:
            raise ValueError("No engine name provided and no default in config")
        return name.lower()
    
    async def create_engine(self, engine_name: str, params: Optional[Dict[str, Any]] = None):
        """Create and initialize backtest engine"""
        engine = await self.create(engine_name, params)
        await engine.initialize()
        return engine
    
    def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """Get all available backtest engines with metadata"""
        engines = {}
        for name, info in self._registry.items():
            if name in self._metadata:
                engines[name] = self._metadata[name]
            else:
                engines[name] = {"description": "Backtest engine"}
        return engines


# Singleton instance getter
def get_backtest_factory(config: ConfigManager) -> BacktestFactory:
    """Get or create singleton instance of BacktestFactory"""
    return BacktestFactory.get_instance(config)