#!/usr/bin/env python3
# src/trading/factory.py

from enum import Enum
from typing import Dict, Optional, Any, Type, List

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.mode.base import BaseTradingMode


class TradingMode(Enum):    
    """Centralize the definition of transaction mode types"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    

class TradingModeFactory(AbstractFactory):
    """Factory for creating trading mode instances"""
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """Initialize trading mode factory"""
        super().__init__(config)
        
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        self._register_default_modes()
        self._discover_trading_modes()
    
    def _register_default_modes(self) -> None:
        """Register default trading modes with consistent metadata"""
        self.register(TradingMode.BACKTEST.value, "src.mode.backtest.BacktestMode", {
            "description": "Historical data backtesting",
            "features": ["historical_data", "performance_analysis"],
            "category": "simulation"
        })
        
        self.register(TradingMode.PAPER.value, "src.mode.paper.PaperMode", {
            "description": "Paper trading (uses real market data without real funds)",
            "features": ["real_time_data", "virtual_execution"],
            "category": "simulation"
        })
        
        self.register(TradingMode.LIVE.value, "src.mode.live.LiveMode", {
            "description": "Live trading (uses real market data on exchange)",
            "features": ["real_time_data", "real_execution", "risk_management"],
            "category": "production"
        })
    
    def _discover_trading_modes(self) -> None:
        """Auto-discover trading mode modules"""
        try:
            mode_dir = "src.mode"
            self.discover_registrable_classes(BaseTradingMode, mode_dir, "trading_mode_factory")
        except Exception as e:
            self.logger.error(f"Error auto-discovering trading modes: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseTradingMode]:
        """Get trading mode class"""
        return await self._load_class_from_path(name, BaseTradingMode)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """Validate and resolve trading mode name"""
        try:
            name_value = name.lower() if name else TradingMode.BACKTEST.value
            mode = TradingMode(name_value)
        except ValueError:
            valid_modes = [m.value for m in TradingMode]
            raise ValueError(f"Unsupported trading mode: {name}. Must be one of: {valid_modes}")
            
        use_live_data = self.config.get("data", "use_live_data", default=False)
        if use_live_data and mode != TradingMode.LIVE:
            self.logger.info(f"Overriding mode to {TradingMode.LIVE.value} due to use_live_data=True")
            return TradingMode.LIVE.value
            
        return mode.value
    
    def get_available_modes(self) -> Dict[str, Dict[str, Any]]:
        """Get available trading modes with complete metadata"""
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = metadata
        
        for mode in TradingMode:
            if mode.value not in result:
                result[mode.value] = {
                    "description": self._get_default_description(mode),
                    "features": [],
                    "category": "default"
                }
                
        return result
    
    def _get_default_description(self, mode: TradingMode) -> str:
        """Get default description for trading mode"""
        descriptions = {
            TradingMode.BACKTEST: "Historical data backtesting",
            TradingMode.PAPER: "Paper trading (uses real market data without real funds)",
            TradingMode.LIVE: "Live trading (uses real funds on exchange)"
        }
        return descriptions.get(mode, "Unknown trading mode")
    
    def get_mode_features(self, mode_name: str) -> List[str]:
        """Get features of a trading mode"""
        metadata = self._metadata.get(mode_name.lower(), {})
        return metadata.get('features', [])
    
    async def create_trading_mode(self, name: Optional[str] = None) -> BaseTradingMode:
        """Create a trading mode with parameters from configuration"""
        resolved_name = await self._resolve_name(name)
        
        mode_config = self.config.get("trading", "modes", resolved_name, default={})
        
        return await self.create(resolved_name, params=mode_config)


def get_trading_mode_factory(config: ConfigManager) -> TradingModeFactory:
    """Get or create singleton instance of TradingModeFactory"""
    return TradingModeFactory.get_instance(config)