#!/usr/bin/env python3
# src/strategy/factors/registry.py

from typing import Dict, List, Any, Type, Optional, Union, Callable

from src.common.log_manager import LogManager
from src.strategy.factors.base import FactorBase, SignalType
from src.strategy.factors.momentum import RSI, MACD, Stochastic
from src.strategy.factors.volatility import BollingerBands, ATR, ADX
from src.strategy.factors.volume import OBV, MoneyFlowIndex, VolumeOscillator
from src.strategy.factors.trend import SMA, EMA, IchimokuCloud


logger = LogManager.get_logger("strategy.factors.registry")


class FactorRegistry:
    """
    Registry for managing and accessing technical indicators and factors.
    
    This registry provides a centralized way to access all available factors
    and create instances with specific parameters.
    """
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FactorRegistry, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the registry with available factors"""
        self._factor_classes = {}
        self._factor_instances = {}
        
        # Register available factor classes by category
        self._register_default_factors()
    
    def _register_default_factors(self) -> None:
        """Register default factor classes"""
        # Momentum factors
        self.register_factor_class('rsi', RSI, category='momentum', 
                                  description="Relative Strength Index")
        self.register_factor_class('macd', MACD, category='momentum', 
                                  description="Moving Average Convergence Divergence")
        self.register_factor_class('stochastic', Stochastic, category='momentum', 
                                  description="Stochastic Oscillator")
        
        # Volatility factors
        self.register_factor_class('bollinger', BollingerBands, category='volatility', 
                                  description="Bollinger Bands")
        self.register_factor_class('atr', ATR, category='volatility', 
                                  description="Average True Range")
        self.register_factor_class('adx', ADX, category='volatility', 
                                  description="Average Directional Index")
        
        # Volume factors
        self.register_factor_class('obv', OBV, category='volume', 
                                  description="On-Balance Volume")
        self.register_factor_class('mfi', MoneyFlowIndex, category='volume', 
                                  description="Money Flow Index")
        self.register_factor_class('volume_osc', VolumeOscillator, category='volume', 
                                  description="Volume Oscillator")
        
        # Trend factors
        self.register_factor_class('sma', SMA, category='trend', 
                                  description="Simple Moving Average")
        self.register_factor_class('ema', EMA, category='trend', 
                                  description="Exponential Moving Average")
        self.register_factor_class('ichimoku', IchimokuCloud, category='trend', 
                                  description="Ichimoku Cloud")
    
    def register_factor_class(self, name: str, factor_class: Type[FactorBase], 
                            category: str = 'custom', description: str = "") -> None:
        """
        Register a factor class in the registry
        
        Args:
            name: Unique name for the factor
            factor_class: Factor class (subclass of FactorBase)
            category: Factor category
            description: Factor description
        """
        if not issubclass(factor_class, FactorBase):
            logger.error(f"Failed to register {name}: Class must inherit from FactorBase")
            return
            
        self._factor_classes[name.lower()] = {
            'class': factor_class,
            'category': category,
            'description': description
        }
        logger.debug(f"Registered factor class {name} in category {category}")
    
    def create_factor(self, name: str, **kwargs) -> Optional[FactorBase]:
        """
        Create a factor instance with specified parameters
        
        Args:
            name: Factor name
            **kwargs: Parameters for factor initialization
            
        Returns:
            Optional[FactorBase]: Created factor instance or None if factor not found
        """
        name_lower = name.lower()
        if name_lower not in self._factor_classes:
            logger.warning(f"Factor '{name}' not found in registry")
            return None
            
        factor_info = self._factor_classes[name_lower]
        try:
            # Create instance
            instance = factor_info['class'](**kwargs)
            # Store in instances cache with unique identifier
            instance_key = f"{name_lower}_{hash(frozenset(kwargs.items()))}"
            self._factor_instances[instance_key] = instance
            return instance
        except Exception as e:
            logger.error(f"Error creating factor '{name}': {e}")
            return None
    
    def get_or_create_factor(self, name: str, **kwargs) -> Optional[FactorBase]:
        """
        Get existing factor instance or create a new one
        
        Args:
            name: Factor name
            **kwargs: Parameters for factor initialization
            
        Returns:
            Optional[FactorBase]: Factor instance or None if factor not found
        """
        # Generate a unique key for this factor configuration
        name_lower = name.lower()
        instance_key = f"{name_lower}_{hash(frozenset(kwargs.items()))}"
        
        # Return existing instance if available
        if instance_key in self._factor_instances:
            return self._factor_instances[instance_key]
            
        # Create new instance
        return self.create_factor(name, **kwargs)
    
    def get_available_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available factors
        
        Returns:
            Dict[str, Dict[str, Any]]: Factor information by name
        """
        return {name: {'category': info['category'], 'description': info['description']} 
                for name, info in self._factor_classes.items()}
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """
        Get list of factor names in specified category
        
        Args:
            category: Factor category
            
        Returns:
            List[str]: List of factor names
        """
        return [name for name, info in self._factor_classes.items() 
                if info['category'].lower() == category.lower()]


# Singleton instance for easy access
_registry = None

def get_factor_registry() -> FactorRegistry:
    """
    Get singleton instance of factor registry
    
    Returns:
        FactorRegistry: Factor registry instance
    """
    global _registry
    if _registry is None:
        _registry = FactorRegistry()
    return _registry


def create_factor(name: str, **kwargs) -> Optional[FactorBase]:
    """
    Convenience function to create a factor instance
    
    Args:
        name: Factor name
        **kwargs: Factor parameters
        
    Returns:
        Optional[FactorBase]: Factor instance
    """
    return get_factor_registry().create_factor(name, **kwargs)