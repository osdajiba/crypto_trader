#!/usr/bin/env python3
# src/portfolio/assets/factory.py

import asyncio
from typing import Type, Dict, Any, Optional, List, Set
import inspect
import importlib

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.assets.base import Asset


logger = LogManager.get_logger("portfolio.assets.factory")


class AssetFactory(AbstractFactory):
    """
    Factory for creating different types of assets with execution integration.
    
    This factory provides a centralized mechanism for creating, discovering, and
    managing various asset types within the portfolio. It supports both synchronous
    and asynchronous asset creation with automatic execution engine setup.
    """
    
    @classmethod
    def instance(cls, config):
        """
        Get singleton instance of the factory
        
        Args:
            config: Configuration manager
            
        Returns:
            AssetFactory: Singleton factory instance
        """
        return cls.get_instance(config)
    
    def __init__(self, config: ConfigManager):
        """
        Initialize asset factory
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.logger = LogManager.get_logger("factory.asset")
        
        # Asset creation handlers
        self._creation_hooks = {}
        
        # Track created assets
        self._created_assets = {}
        
        # Register default assets
        self._register_default_assets()
    
    def _register_default_assets(self) -> None:
        """Register default asset types"""
        # Register spot asset as default
        self.register("spot", "src.portfolio.assets.spot.Spot")
        
        # Register other built-in assets
        self.register("future", "src.portfolio.assets.future.Future")
        
        logger.info("Registered default asset types")
    
    async def _get_concrete_class(self, name: str) -> Type[Asset]:
        """
        Get concrete asset class by name
        
        Args:
            name: Asset type name
            
        Returns:
            Type[Asset]: Asset class
            
        Raises:
            ValueError: If asset type not found
        """
        if name not in self._registry:
            supported = ", ".join(self._registry.keys())
            raise ValueError(f"Asset type not found: {name}. Supported types: {supported}")
            
        return await self._load_class_from_path(name, Asset)
        
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve asset type name
        
        Args:
            name: Asset type name
            
        Returns:
            str: Resolved asset type name
            
        Note:
            Default to spot asset if none specified
        """
        if name is None:
            return 'spot'
        return name.lower()
    
    def register_creation_hook(self, asset_type: str, hook: callable) -> None:
        """
        Register a hook to be called before asset creation
        
        Args:
            asset_type: Asset type
            hook: Hook function to call
        """
        if asset_type not in self._creation_hooks:
            self._creation_hooks[asset_type] = set()
            
        self._creation_hooks[asset_type].add(hook)
        logger.debug(f"Registered creation hook for {asset_type} assets")
    
    async def _run_creation_hooks(self, asset_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run creation hooks for an asset type
        
        Args:
            asset_type: Asset type
            params: Asset creation parameters
            
        Returns:
            Dict[str, Any]: Modified parameters
        """
        # Run global hooks first
        global_hooks = self._creation_hooks.get('*', set())
        for hook in global_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    params = await hook(asset_type, params)
                else:
                    params = hook(asset_type, params)
            except Exception as e:
                logger.warning(f"Error in global creation hook: {e}")
        
        # Run asset type specific hooks
        type_hooks = self._creation_hooks.get(asset_type, set())
        for hook in type_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    params = await hook(params)
                else:
                    params = hook(params)
            except Exception as e:
                logger.warning(f"Error in {asset_type} creation hook: {e}")
                
        return params
    
    def get_asset_instance(self, asset_name: str) -> Optional[Asset]:
        """
        Get a previously created asset instance by name
        
        Args:
            asset_name: Asset name
            
        Returns:
            Optional[Asset]: Asset instance if found, None otherwise
        """
        asset_info = self._created_assets.get(asset_name)
        if asset_info:
            return asset_info['instance']
        return None
    
    def discover_assets(self) -> None:
        """
        Auto-discover asset classes with the asset_factory registration
        """
        try:
            # Discover and register asset implementations
            self.discover_registrable_classes(Asset, 'src.portfolio.assets', 'asset_factory')
            logger.info(f"Discovered {len(self._registry)} asset types")
        except Exception as e:
            logger.error(f"Error during asset discovery: {str(e)}")
            logger.info("Registering default assets only")
            
    async def create_asset(self, params: Optional[Dict[str, Any]] = None) -> Asset:
        """Create an asset instance asynchronously"""
        try:
            params_copy = params.copy()
            asset_type = await self._resolve_name(params.pop('type', 'spot'))
            exchange_instance = params_copy.get('exchange', None)
            execution_engine_instance = params_copy.get('execution_engine', None)
            
            asset_class = await self._get_concrete_class(asset_type)
            
            modified_params = await self._run_creation_hooks(asset_type, params_copy)
            asset_name = modified_params.get('name', '')
                        
            asset = asset_class(
                name=asset_name,
                exchange=exchange_instance,
                execution_engine=execution_engine_instance,
                config=self.config,
                params=params_copy
            )
            
            if hasattr(asset, 'initialize') and callable(asset.initialize):
                await asset.initialize()
            
            self._created_assets[asset_name] = {
                'instance': asset,
                'type': asset_type,
                'params': modified_params
            }
            
            self.logger.info(f"Created asset {asset_name} of type {asset_type}")
            return asset
            
        except Exception as e:
            self.logger.error(f"Error creating asset: {str(e)}")
            raise

    async def create_multi_assets(self, assets_config: List[Dict[str, Any]], 
                                exchange=None, execution_engine=None) -> Dict[str, Asset]:
        """Create multiple assets from configuration asynchronously"""
        assets = {}
                
        # Prepare asset configurations
        for config in assets_config:
            asset_name = config.get('symbol') or config.get('name')
            if not asset_name:
                self.logger.warning(f"Skipping asset with no symbol or name: {config}")
                continue
            
            config['name'] = asset_name
            config['symbol'] = asset_name
            config['tradable'] = True

            if exchange is not None:
                config['exchange'] = exchange
            
            if execution_engine is not None:
                config['execution_engine'] = execution_engine
            
            try:
                self.logger.debug(f"Creating asset {asset_name} with execution_engine: {config.get('execution_engine') is not None}")
                asset = await self.create_asset(params=config)
                assets[asset_name] = asset
                self.logger.info(f"Successfully created asset {asset_name}")
            except Exception as e:
                self.logger.error(f"Error creating asset {asset_name}: {e}")
        
        return assets

    def get_supported_asset_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported asset types
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping asset types to their metadata
        """
        result = {}
        for name, class_path in self._registry.items():
            metadata = self._metadata.get(name, {})
            result[name] = {
                'class_path': class_path,
                'metadata': metadata
            }
        return result
    
    def get_asset_type_info(self, asset_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific asset type
        
        Args:
            asset_type: Asset type name
            
        Returns:
            Optional[Dict[str, Any]]: Asset type metadata or None if not found
        """
        asset_type = asset_type.lower()
        if asset_type in self._registry:
            return {
                'class_path': self._registry[asset_type],
                'metadata': self._metadata.get(asset_type, {})
            }
        return None
        
    async def initialize_markets(self, exchange) -> None:
        """
        Initialize markets information from exchange
        
        Args:
            exchange: Exchange interface
        """
        if not exchange:
            logger.warning("No exchange provided, skipping market initialization")
            return
            
        try:
            # Ensure exchange is initialized
            if hasattr(exchange, 'initialize') and callable(exchange.initialize):
                await exchange.initialize()
            
            # Load markets
            if hasattr(exchange, 'fetch_markets') and callable(exchange.fetch_markets):
                markets = await exchange.fetch_markets()
                logger.info(f"Initialized {len(markets)} markets from exchange")
            else:
                logger.warning("Exchange does not support market fetching")
                
        except Exception as e:
            logger.error(f"Error initializing markets: {e}")


def get_asset_factory(config: ConfigManager) -> AssetFactory:
    """
    Get the asset factory singleton instance
    
    Args:
        config: Configuration manager
        
    Returns:
        AssetFactory: Singleton instance
    """
    return AssetFactory.instance(config)