#!/usr/bin/env python3
# src/exchange/factory.py

"""
Exchange factory module.
Provides factory methods for creating exchange instances according to the factory pattern standard.
"""

import time
from typing import Dict, Optional, Any, Type, List, Set, Callable

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.base import Exchange, ExchangeError


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
        self._active_connections = {}    # ensure singleton implenmentation
        
        # Initialize logger with proper category
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Track created exchanges
        self._created_exchanges: Dict[str, Exchange] = {}
        
        # Exchange lifecycle hooks
        self._initialization_hooks: Dict[str, Set[Callable]] = {}
        self._shutdown_hooks: Dict[str, Set[Callable]] = {}
        
        # Connection pool for reusable exchange instances
        self._connection_pool: Dict[str, Dict[str, Any]] = {}
        
        # Register built-in exchanges
        self._register_default_exchanges()
        
        # Auto-discover additional exchanges
        self._discover_exchanges()
    
    def _register_default_exchanges(self) -> None:
        """Register default exchanges with consistent metadata"""
        self.register(
            "binance", 
            "src.exchange.binance.BinanceExchange",
            {
                "description": "Binance exchange integration",
                "features": ["spot", "futures", "websocket", "margin"],
                "category": "exchange",
                "supported_order_types": [
                    "market", "limit", "stop", "stop_market", "stop_limit",
                    "take_profit", "take_profit_market", "take_profit_limit",
                    "trailing_stop_market"
                ],
                "supported_market_types": ["spot", "future", "margin"]
            }
        )
        
        # You can register other exchanges similarly
        self.register(
            "bybit",
            "src.exchange.bybit.BybitExchange",
            {
                "description": "Bybit exchange integration",
                "features": ["spot", "futures", "websocket"],
                "category": "exchange",
                "supported_order_types": ["market", "limit", "stop", "stop_limit"],
                "supported_market_types": ["spot", "future"]
            }
        )
        
        self.logger.info("Registered default exchanges")
    
    def _discover_exchanges(self) -> None:
        """Auto-discover additional exchange implementations"""
        try:
            # Discover implementations from exchange directory
            module_path = "src.exchange"
            self.discover_registrable_classes(
                Exchange, 
                module_path, 
                "exchange_factory"
            )
            self.logger.debug(f"Discovered additional exchanges from {module_path}")
        except Exception as e:
            self.logger.error(f"Error during exchange discovery: {str(e)}")
    
    def register_initialization_hook(self, exchange_name: str, hook: Callable) -> None:
        """
        Register a hook to be called after exchange initialization
        
        Args:
            exchange_name: Exchange name (or '*' for all exchanges)
            hook: Hook function to call
        """
        if exchange_name not in self._initialization_hooks:
            self._initialization_hooks[exchange_name] = set()
            
        self._initialization_hooks[exchange_name].add(hook)
        self.logger.debug(f"Registered initialization hook for {exchange_name}")
    
    def register_shutdown_hook(self, exchange_name: str, hook: Callable) -> None:
        """
        Register a hook to be called before exchange shutdown
        
        Args:
            exchange_name: Exchange name (or '*' for all exchanges)
            hook: Hook function to call
        """
        if exchange_name not in self._shutdown_hooks:
            self._shutdown_hooks[exchange_name] = set()
            
        self._shutdown_hooks[exchange_name].add(hook)
        self.logger.debug(f"Registered shutdown hook for {exchange_name}")
    
    async def _run_initialization_hooks(self, exchange_name: str, exchange: Exchange) -> None:
        """
        Run initialization hooks for an exchange
        
        Args:
            exchange_name: Exchange name
            exchange: Exchange instance
        """
        # Run exchange-specific hooks
        for hook in self._initialization_hooks.get(exchange_name, set()):
            try:
                await hook(exchange)
            except Exception as e:
                self.logger.error(f"Error in {exchange_name} initialization hook: {e}")
        
        # Run global hooks
        for hook in self._initialization_hooks.get('*', set()):
            try:
                await hook(exchange)
            except Exception as e:
                self.logger.error(f"Error in global initialization hook: {e}")
    
    async def _run_shutdown_hooks(self, exchange_name: str, exchange: Exchange) -> None:
        """
        Run shutdown hooks for an exchange
        
        Args:
            exchange_name: Exchange name
            exchange: Exchange instance
        """
        # Run exchange-specific hooks
        for hook in self._shutdown_hooks.get(exchange_name, set()):
            try:
                await hook(exchange)
            except Exception as e:
                self.logger.error(f"Error in {exchange_name} shutdown hook: {e}")
        
        # Run global hooks
        for hook in self._shutdown_hooks.get('*', set()):
            try:
                await hook(exchange)
            except Exception as e:
                self.logger.error(f"Error in global shutdown hook: {e}")
    
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
    
    def get_exchange_order_types(self, exchange_name: str) -> List[str]:
        """
        Get supported order types for an exchange
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List[str]: List of supported order types
        """
        metadata = self._metadata.get(exchange_name.lower(), {})
        return metadata.get('supported_order_types', [])
    
    def get_exchange_market_types(self, exchange_name: str) -> List[str]:
        """
        Get supported market types for an exchange
        
        Args:
            exchange_name: Name of the exchange
            
        Returns:
            List[str]: List of supported market types
        """
        metadata = self._metadata.get(exchange_name.lower(), {})
        return metadata.get('supported_market_types', [])
    
    async def create_exchange(self, name: Optional[str] = None, force_recreate: bool = False) -> Exchange:
        """
        Get existing exchange instance or create new one
        
        Args:
            name: Optional exchange name
            force_recreate: Force creation of new instance
            
        Returns:
            Exchange: Exchange instance
        """
        resolved_name = await self._resolve_name(name)
        
        # Return existing instance if available
        if not force_recreate and resolved_name in self._created_exchanges:
            exchange = self._created_exchanges[resolved_name]
            
            # Check if exchange is still connected
            if exchange.is_connected():
                return exchange
            else:
                # Try to reconnect or create new instance
                try:
                    await exchange.initialize()
                    if exchange.is_connected():
                        return exchange
                except Exception as e:
                    self.logger.warning(f"Failed to reconnect exchange {resolved_name}: {e}")
                
        # Get exchange-specific configuration
        exchange_config = self.config.get("exchange", "exchanges", resolved_name, default={})
        
        # Create exchange instance
        exchange = await self.create(resolved_name, params=exchange_config)
        
        # Store instance for reuse
        self._created_exchanges[resolved_name] = exchange
        
        return exchange
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Exchange:
        """
        Create exchange instance
        
        Args:
            name: Exchange name or None to use default
            params: Additional parameters for initialization
            
        Returns:
            Exchange: Initialized exchange instance
        """
        resolved_name = await self._resolve_name(name)
        params = params or {}
        
        # Get concrete class
        concrete_class = await self._get_concrete_class(resolved_name)
        
        # Create instance
        try:
            # Create exchange
            exchange = concrete_class(self.config, params)
            
            # Initialize exchange
            await exchange.initialize()
            
            # Run initialization hooks
            await self._run_initialization_hooks(resolved_name, exchange)
            
            # Store instance
            self._created_exchanges[resolved_name] = exchange
            
            self.logger.info(f"Created and initialized exchange: {resolved_name}")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to create exchange {resolved_name}: {e}")
            raise ExchangeError(f"Exchange creation failed: {str(e)}")
    
    async def shutdown_exchange(self, name: str) -> None:
        """
        Shutdown specific exchange instance
        
        Args:
            name: Exchange name
        """
        name = name.lower()
        
        if name in self._created_exchanges:
            exchange = self._created_exchanges[name]
            
            try:
                # Run shutdown hooks
                await self._run_shutdown_hooks(name, exchange)
                
                # Shutdown exchange
                await exchange.shutdown()
                
                # Remove from created exchanges
                del self._created_exchanges[name]
                
                self.logger.info(f"Shutdown exchange: {name}")
                
            except Exception as e:
                self.logger.error(f"Error shutting down exchange {name}: {e}")
    
    async def shutdown_all(self) -> None:
        """Shutdown all created exchange instances"""
        for name in list(self._created_exchanges.keys()):
            await self.shutdown_exchange(name)
    
    def get_created_exchanges(self) -> Dict[str, Exchange]:
        """
        Get all created exchange instances
        
        Returns:
            Dict[str, Exchange]: Created exchanges
        """
        return self._created_exchanges.copy()
    
    def get_exchange_instance(self, name: str) -> Optional[Exchange]:
        """
        Get exchange instance by name
        
        Args:
            name: Exchange name
            
        Returns:
            Optional[Exchange]: Exchange instance if exists
        """
        return self._created_exchanges.get(name.lower())
    
    async def validate_exchange_compatibility(self, exchange_name: str, 
                                            required_features: List[str] = None,
                                            required_order_types: List[str] = None,
                                            required_market_types: List[str] = None) -> bool:
        """
        Validate if exchange supports required features
        
        Args:
            exchange_name: Exchange name
            required_features: List of required features
            required_order_types: List of required order types
            required_market_types: List of required market types
            
        Returns:
            bool: True if compatible, False otherwise
        """
        exchange_name = exchange_name.lower()
        
        if exchange_name not in self._metadata:
            self.logger.error(f"Exchange not found: {exchange_name}")
            return False
        
        metadata = self._metadata[exchange_name]
        
        # Check features
        if required_features:
            exchange_features = set(metadata.get('features', []))
            required_features_set = set(required_features)
            
            if not required_features_set.issubset(exchange_features):
                missing_features = required_features_set - exchange_features
                self.logger.warning(f"Exchange {exchange_name} missing features: {missing_features}")
                return False
        
        # Check order types
        if required_order_types:
            exchange_order_types = set(metadata.get('supported_order_types', []))
            required_order_types_set = set(required_order_types)
            
            if not required_order_types_set.issubset(exchange_order_types):
                missing_order_types = required_order_types_set - exchange_order_types
                self.logger.warning(f"Exchange {exchange_name} missing order types: {missing_order_types}")
                return False
        
        # Check market types
        if required_market_types:
            exchange_market_types = set(metadata.get('supported_market_types', []))
            required_market_types_set = set(required_market_types)
            
            if not required_market_types_set.issubset(exchange_market_types):
                missing_market_types = required_market_types_set - exchange_market_types
                self.logger.warning(f"Exchange {exchange_name} missing market types: {missing_market_types}")
                return False
        
        return True
    
    def set_connection_pooling(self, enabled: bool = True, max_idle_time: int = 300) -> None:
        """
        Enable or disable connection pooling for exchanges
        
        Args:
            enabled: Whether to enable connection pooling
            max_idle_time: Maximum idle time in seconds before closing connection
        """
        self._connection_pool_enabled = enabled
        self._connection_pool_max_idle = max_idle_time
        
        self.logger.info(f"Connection pooling {'enabled' if enabled else 'disabled'}")
    
    async def get_pooled_connection(self, exchange_name: str) -> Optional[Exchange]:
        """
        Get exchange instance from connection pool if available
        
        Args:
            exchange_name: Exchange name
            
        Returns:
            Optional[Exchange]: Exchange instance if available
        """
        if not getattr(self, '_connection_pool_enabled', False):
            return None
        
        exchange_name = exchange_name.lower()
        
        if exchange_name in self._connection_pool:
            pool_data = self._connection_pool[exchange_name]
            exchange = pool_data.get('exchange')
            
            # Check if exchange is still connected and not too old
            if exchange and exchange.is_connected():
                idle_time = time.time() - pool_data.get('last_used', 0)
                
                if idle_time < getattr(self, '_connection_pool_max_idle', 300):
                    pool_data['last_used'] = time.time()
                    return exchange
        
        return None


def get_exchange_factory(config: ConfigManager) -> ExchangeFactory:
    """
    Get or create singleton instance of ExchangeFactory
    
    Args:
        config: Configuration manager
    
    Returns:
        ExchangeFactory: Singleton instance
    """
    return ExchangeFactory.get_instance(config)