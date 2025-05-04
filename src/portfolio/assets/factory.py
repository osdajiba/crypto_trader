#!/usr/bin/env python3
# src/portfolio/assets/factory.py

import asyncio
from typing import Type, Dict, Any, Optional

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.assets.base import Asset
from src.portfolio.execution.factory import get_execution_factory


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
            raise ValueError(f"Asset type not found: {name}")
            
        return await self._load_class_from_path(name, Asset)
        
    async def _resolve_name(self, name: str) -> str:
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
    
    def create_asset(self, asset_type: str, params: dict) -> Asset:
        """
        Create an asset synchronously
        
        Args:
            asset_type: Type of asset to create
            params: Asset creation parameters
            
        Returns:
            Asset: Created asset instance
        """
        return self.create_sync(asset_type, params)
        
    async def create(self, asset_type: str, params: Optional[Dict[str, Any]] = None) -> Asset:
        """
        Create an asset instance asynchronously
        
        Args:
            asset_type: Type of asset to create
            params: Asset creation parameters (optional)
            
        Returns:
            Asset: Created and initialized asset instance
        """
        params = params or {}
        
        # Resolve asset type name
        resolved_name = await self._resolve_name(asset_type)
        
        # Get concrete asset class
        concrete_class = await self._get_concrete_class(resolved_name)
        
        # Get metadata for asset type
        asset_metadata = self._metadata.get(resolved_name, {})
        
        # Create combined parameters
        combined_params = dict(asset_metadata)
        if params:
            combined_params.update(params)
            
        # Set up config if needed
        if 'config' not in combined_params:
            combined_params['config'] = self.config
        
        # Create asset instance
        asset = concrete_class(**combined_params)
        
        # Initialize asset if it has an async initialize method
        if hasattr(asset, 'initialize') and callable(asset.initialize):
            try:
                await asset.initialize()
            except Exception as e:
                logger.error(f"Error initializing asset {asset.name}: {str(e)}")
                raise
        
        logger.info(f"Created {resolved_name} asset: {asset.name}")
        return asset
        
    def discover_assets(self) -> None:
        """
        Auto-discover asset classes with the asset_factory registration
        """
        try:
            self.discover_registrable_classes(Asset, 'src.portfolio.assets', 'asset_factory')
            logger.info(f"Discovered {len(self._registry)} asset types")
        except Exception as e:
            logger.error(f"Error during asset discovery: {str(e)}")
            logger.info("Registering default assets only")
            self._register_default_assets()
    
    def _register_default_assets(self) -> None:
        """Register default asset types"""
        # Register spot asset as default
        self.register("spot", "src.portfolio.assets.spot.SpotAsset")
        # Register other built-in assets
        self.register("future", "src.portfolio.assets.future.FutureAsset")
        
        logger.info("Registered default asset types")
        
    async def create_multi_assets(self, asset_configs: Dict[str, Dict[str, Any]], 
                                 exchange=None) -> Dict[str, Asset]:
        """
        Create multiple assets from configuration asynchronously
        
        Args:
            asset_configs: Dictionary mapping asset names to their configurations
            exchange: Exchange interface to use for all assets (optional)
            
        Returns:
            Dict[str, Asset]: Dictionary of created assets
        """
        assets = {}
        create_tasks = []
        
        for asset_name, config in asset_configs.items():
            # Make sure the asset has a name
            if 'name' not in config:
                config['name'] = asset_name
                
            # Add exchange if provided
            if exchange and 'exchange' not in config:
                config['exchange'] = exchange
            
            # Get asset type, default to spot
            asset_type = config.pop('type', 'spot')
            
            # Create task for asset creation
            create_tasks.append(
                self.create(asset_type, config)
            )
        
        # Wait for all assets to be created
        if create_tasks:
            created_assets = await asyncio.gather(*create_tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(created_assets):
                asset_name = list(asset_configs.keys())[i]
                if isinstance(result, Exception):
                    logger.error(f"Error creating asset {asset_name}: {str(result)}")
                else:
                    assets[asset_name] = result
        
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
            logger.error(f"Error initializing markets: {str(e)}")


def get_asset_factory(config: ConfigManager) -> AssetFactory:
    """
    Get the asset factory singleton instance
    
    Args:
        config: Configuration manager
        
    Returns:
        AssetFactory: Singleton instance
    """
    return AssetFactory.instance(config)