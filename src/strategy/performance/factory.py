#!/usr/bin/env python3
# src/strategy/performance/factory.py

from typing import Dict, Optional, Any, Type
import asyncio

from src.common.abstract_factory import AbstractFactory
from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import PerformanceAnalyzer
from src.strategy.performance.base import BasePerformanceAnalyzer


class PerformanceAnalyzerFactory(AbstractFactory):
    """Factory for creating performance analyzer instances"""
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """Initialize performance analyzer factory"""
        super().__init__(config)
        
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Register default analyzers
        self._register_default_analyzers()
        
        # Discover additional analyzers
        self._discover_analyzers()
    
    def _register_default_analyzers(self) -> None:
        """Register default performance analyzers with metadata"""
        self.register(PerformanceAnalyzer.BACKTEST.value, "src.strategy.performance.backtest.BacktestPerformanceAnalyzer", {
            "description": "Performance analyzer for backtesting",
            "features": ["historical_analysis", "equity_tracking", "drawdown_calculation"],
            "category": "simulation"
        })
        
        self.register(PerformanceAnalyzer.TRADING.value, "src.strategy.performance.trading.TradingPerformanceAnalyzer", {
            "description": "Performance analyzer for live and paper trading",
            "features": ["real_time_analysis", "equity_tracking", "drawdown_calculation", "trade_tracking"],
            "category": "production"
        })
    
    def _discover_analyzers(self) -> None:
        """Auto-discover performance analyzers from modules"""
        try:
            analyzer_dir = "src.strategy.performance"
            self.discover_registrable_classes(BasePerformanceAnalyzer, analyzer_dir, "performance_factory")
        except Exception as e:
            self.logger.error(f"Error auto-discovering performance analyzers: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BasePerformanceAnalyzer]:
        """Get performance analyzer class"""
        return await self._load_class_from_path(name, BasePerformanceAnalyzer)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """Validate and resolve performance analyzer name"""
        if name is None:
            # Default to trading mode specific analyzer
            trading_mode = self.config.get("system", "operational_mode", default="backtest")
            
            if trading_mode in ["live", "paper"]:
                return PerformanceAnalyzer.TRADING.value
            
            return PerformanceAnalyzer.BACKTEST.value
        
        # Check if name is a valid enum value
        try:
            analyzer = PerformanceAnalyzer(name.lower())
            return analyzer.value
        except ValueError:
            # If not an enum value, assume it's a direct class name
            if name.lower() in self._registry:
                return name.lower()
            
            # Fallback
            self.logger.warning(f"Unknown analyzer name: {name}, falling back to backtest analyzer")
            return PerformanceAnalyzer.BACKTEST.value
    
    def get_available_analyzers(self) -> Dict[str, Dict[str, Any]]:
        """Get available performance analyzers with metadata"""
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = metadata
            
        # Add enum values that might not be registered
        for analyzer in PerformanceAnalyzer:
            if analyzer.value not in result:
                result[analyzer.value] = {
                    "description": self._get_default_description(analyzer),
                    "features": [],
                    "category": "default"
                }
                
        return result
    
    def _get_default_description(self, analyzer: PerformanceAnalyzer) -> str:
        """Get default description for analyzer type"""
        descriptions = {
            PerformanceAnalyzer.BACKTEST: "Backtest performance analysis",
            PerformanceAnalyzer.TRADING: "Live/paper trading performance analysis"
        }
        return descriptions.get(analyzer, "Unknown performance analyzer")
    
    async def create_with_config_params(self, name: Optional[str] = None) -> BasePerformanceAnalyzer:
        """Create a performance analyzer with parameters from configuration"""
        resolved_name = await self._resolve_name(name)
        
        # Get mode-specific analyzer config
        analyzer_config = self.config.get("performance", resolved_name, default={})
        
        return await self.create(resolved_name, params=analyzer_config)


def get_analyzer_factory(config: ConfigManager) -> PerformanceAnalyzerFactory:
    """Get or create singleton instance of PerformanceAnalyzerFactory"""
    return PerformanceAnalyzerFactory.get_instance(config)