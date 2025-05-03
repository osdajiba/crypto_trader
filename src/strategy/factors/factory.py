#!/usr/bin/env python3
# src/strategy/factors/factory.py

from typing import Dict, Optional, Any, Type, List
import asyncio

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.factors.base import FactorBase


class FactorFactory(AbstractFactory):
    """
    Factory for creating and managing technical indicator factors
    
    This factory provides a centralized way to access all available factors
    and create instances with specific parameters.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the factory
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.logger = LogManager.get_logger("strategy.factors.factory")
        
        # Register default factors
        self._register_default_factors()
        
        # Discover additional factors
        self._discover_factors()
    
    def _register_default_factors(self) -> None:
        """Register default factor classes by category"""
        
        # Momentum factors
        self.register('rsi', 'src.strategy.factors.momentum.RSI',
                     {'category': 'momentum', 'description': 'Relative Strength Index'})
        self.register('macd', 'src.strategy.factors.momentum.MACD',
                     {'category': 'momentum', 'description': 'Moving Average Convergence Divergence'})
        self.register('stochastic', 'src.strategy.factors.momentum.Stochastic',
                     {'category': 'momentum', 'description': 'Stochastic Oscillator'})
        
        # Volatility factors
        self.register('bollinger', 'src.strategy.factors.volatility.BollingerBands',
                     {'category': 'volatility', 'description': 'Bollinger Bands'})
        self.register('atr', 'src.strategy.factors.volatility.ATR',
                     {'category': 'volatility', 'description': 'Average True Range'})
        self.register('adx', 'src.strategy.factors.volatility.ADX',
                     {'category': 'volatility', 'description': 'Average Directional Index'})
        
        # Volume factors
        self.register('obv', 'src.strategy.factors.volume.OBV',
                     {'category': 'volume', 'description': 'On-Balance Volume'})
        self.register('mfi', 'src.strategy.factors.volume.MoneyFlowIndex',
                     {'category': 'volume', 'description': 'Money Flow Index'})
        self.register('volume_osc', 'src.strategy.factors.volume.VolumeOscillator',
                     {'category': 'volume', 'description': 'Volume Oscillator'})
        
        # Trend factors
        self.register('sma', 'src.strategy.factors.trend.SMA',
                     {'category': 'trend', 'description': 'Simple Moving Average'})
        self.register('ema', 'src.strategy.factors.trend.EMA',
                     {'category': 'trend', 'description': 'Exponential Moving Average'})
        self.register('ichimoku', 'src.strategy.factors.trend.IchimokuCloud',
                     {'category': 'trend', 'description': 'Ichimoku Cloud'})
    
    def _discover_factors(self) -> None:
        """Discover additional factors from modules"""
        try:
            factor_dir = "src.strategy.factors"
            self.discover_registrable_classes(FactorBase, factor_dir, "factors")
        except Exception as e:
            self.logger.error(f"Error auto-discovering strategies: {e}")

    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve factor name to a registered name
        
        Args:
            name: Factor name or None to use default
            
        Returns:
            str: Resolved factor name
        """
        if name is None:
            # Get default factor from config
            default_factor = self.config.get("strategy", "factors", "default", default="sma")
            self.logger.debug(f"Using default factor: {default_factor}")
            return default_factor.lower()
        
        return name.lower()
    
    async def _get_concrete_class(self, name: str) -> Type[FactorBase]:
        """
        Get concrete factor class
        
        Args:
            name: Factor name
            
        Returns:
            Type[FactorBase]: Factor class
        """
        return await self._load_class_from_path(name, FactorBase)
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """
        Get list of factor names in specified category
        
        Args:
            category: Factor category
            
        Returns:
            List[str]: List of factor names
        """
        return [name for name, info in self.get_registered_items().items() 
                if info['metadata'].get('category', '').lower() == category.lower()]
    
    def get_factor_info(self, name: str) -> Dict[str, Any]:
        """
        Get information about a specific factor
        
        Args:
            name: Factor name
            
        Returns:
            Dict[str, Any]: Factor information
        """
        items = self.get_registered_items()
        name_lower = name.lower()
        
        if name_lower in items:
            return items[name_lower]
        return {}
    
    def get_available_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available factors
        
        Returns:
            Dict[str, Dict[str, Any]]: Factor information by name
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = {
                'category': metadata.get('category', 'custom'),
                'description': metadata.get('description', '')
            }
        return result


# Singleton access function
def get_factor_factory(config: ConfigManager) -> FactorFactory:
    """
    Get the factor factory singleton
    
    Args:
        config: Configuration manager
        
    Returns:
        FactorFactory: Factor factory instance
    """
    return FactorFactory.get_instance(config)


# Convenience function for creating factors
async def create_factor(config: ConfigManager, name: str, **kwargs) -> Optional[FactorBase]:
    """
    Create a factor instance with specified parameters
    
    Args:
        config: Configuration manager
        name: Factor name
        **kwargs: Parameters for factor initialization
        
    Returns:
        Optional[FactorBase]: Factor instance
    """
    factory = get_factor_factory(config)
    return await factory.create(name, kwargs)