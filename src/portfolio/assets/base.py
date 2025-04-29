#!/usr/bin/env python3
# src/portfolio/assets/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from decimal import Decimal

from src.common.config import ConfigManager
from src.common.log_manager import LogManager


logger = LogManager.get_logger("trading_system.assets")


class Asset(ABC):
    """
    Base class for all portfolio assets with exchange integration
    """
    
    def __init__(self, name: str, exchange=None, config: Optional[ConfigManager] = None, 
                params: Optional[Dict[str, Any]] = None):
        """
        Initialize asset
        
        Args:
            name: Asset identifier
            exchange: Exchange interface (optional)
            config: Configuration manager (optional)
            params: Additional parameters (optional)
        """
        self.name = name
        self.exchange = exchange
        self.config = config if config else ConfigManager()
        self.params = params or {}
        
        # Initialize asset state
        self._value = Decimal('0')
        self._last_update_time = 0
        self._position_size = Decimal('0')
        
    @abstractmethod
    def get_value(self) -> float:
        """
        Return the current value of the asset
        
        Returns:
            float: Current asset value
        """
        pass

    @abstractmethod
    def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy the asset
        
        Args:
            amount: Amount to buy
            **kwargs: Additional parameters for the buy operation
            
        Returns:
            Dict[str, Any]: Transaction result
        """
        pass

    @abstractmethod
    def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell the asset
        
        Args:
            amount: Amount to sell
            **kwargs: Additional parameters for the sell operation
            
        Returns:
            Dict[str, Any]: Transaction result
        """
        pass
    
    async def update_value(self) -> float:
        """
        Update and return current asset value, usually by fetching from exchange
        
        Returns:
            float: Updated asset value
        """
        # To be implemented by subclasses for real-time value updates
        # Default implementation just returns current stored value
        return float(self._value)
        
    def get_position_size(self) -> float:
        """
        Get current position size
        
        Returns:
            float: Position size
        """
        return float(self._position_size)
        
    def set_exchange(self, exchange) -> None:
        """
        Set/update the exchange interface
        
        Args:
            exchange: Exchange interface
        """
        self.exchange = exchange
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert asset to dictionary representation
        
        Returns:
            Dict[str, Any]: Asset as dictionary
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'value': float(self._value),
            'position_size': float(self._position_size)
        }
        
    def __str__(self) -> str:
        """String representation of the asset"""
        return f"{self.__class__.__name__}(name={self.name}, value={float(self._value):.2f})"