#!/usr/bin/env python3
# src/exchange/factory.py

"""
Exchange factory module.
Provides factory methods for creating exchange instances according to the factory pattern standard.
"""

from typing import Dict, Optional, Any, Type, List

from src.common.abstract_factory import AbstractFactory
from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange


class ExchangeFactory(AbstractFactory):
    """
    Factory for creating exchange instances
    
    This factory creates and manages instances of exchange interfaces, providing
    consistent interface for exchange creation, discovery, and metadata access.
    """
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """
        Initialize exchange factory
        
        Args:
            config: Configuration manager instance
        """
        super().__init__(config)
        
        # Initialize logger with proper category
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Register built-in exchanges
        self._register_default_exchanges()
        
        # Auto-discover additional exchanges
        self._discover_exchanges()
    
    def _register_default_exchanges(self) -> None:
        """Register default exchanges with consistent metadata"""
        self.register(
            "binance", 
            "src.exchange.implementations.binance.BinanceExchange",
            {
                "description": "Binance exchange integration",
                "features": ["spot", "futures", "websocket"],
                "category": "exchange"
            }
        )
        
        # You can register other exchanges similarly
        self.register(
            "bybit",
            "src.exchange.implementations.bybit.BybitExchange",
            {
                "description": "Bybit exchange integration",
                "features": ["spot", "futures", "websocket"],
                "category": "exchange"
            }
        )
        
        self.logger.info("Registered default exchanges")
    
    def _discover_exchanges(self) -> None:
        """Auto-discover additional exchange implementations"""
        try:
            # Discover implementations from exchange directory
            module_path = "src.exchange.implementations"
            self.discover_registrable_classes(
                Exchange, 
                module_path, 
                "exchange_factory"
            )
            self.logger.debug(f"Discovered additional exchanges from {module_path}")
        except Exception as e:
            self.logger.error(f"Error during exchange discovery: {str(e)}")
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve exchange name, falling back to default from config if none provided
        
        Args:
            name: Exchange name or None
            
        Returns:
            str: Resolved exchange name
            
        Raises:
            ValueError: If no valid exchange name could be resolved
        """
        if name:
            return name.lower()
            
        # Get default from config
        default_exchange = self.config.get("exchange", "name", default="binance")
        
        if not default_exchange:
            raise ValueError("No exchange name provided and no default exchange configured")
            
        self.logger.debug(f"Using default exchange: {default_exchange}")
        return default_exchange.lower()
    
    async def _get_concrete_class(self, name: str) -> Type[Exchange]:
        """
        Get concrete exchange class by name
        
        Args:
            name: Exchange name
            
        Returns:
            Type[Exchange]: Exchange class
            
        Raises:
            ComponentLoadError: If the component could not be loaded
        """
        return await self._load_class_from_path(name, Exchange)
    
    def get_available_exchanges(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of all available exchanges with metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Exchange names mapped to their metadata
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = metadata
                
        return result
    
    def get_exchange_features(self, exchange_name: str) -> List[str]:
        """
        Get features of a specific exchange
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List[str]: List of features
        """
        metadata = self._metadata.get(exchange_name.lower(), {})
        return metadata.get('features', [])
    
    async def create_with_config_params(self, name: Optional[str] = None) -> Exchange:
        """
        Create an exchange with parameters from configuration
        
        Args:
            name: Optional exchange name
            
        Returns:
            Exchange: Exchange instance
        """
        resolved_name = await self._resolve_name(name)
        
        # Get exchange-specific configuration
        exchange_config = self.config.get("exchange", "exchanges", resolved_name, default={})
        
        return await self.create(resolved_name, params=exchange_config)


def get_exchange_factory(config: ConfigManager) -> ExchangeFactory:
    """
    Get or create singleton instance of ExchangeFactory
    
    Args:
        config: Configuration manager
    
    Returns:
        ExchangeFactory: Singleton instance
    """
    return ExchangeFactory.get_instance(config)