#!/usr/bin/env python3
# src/portfolio/assets/factory.py

import asyncio
from typing import Type, Dict, Any, Optional

from src.common.abstract_factory import AbstractFactory
from src.common.log_manager import LogManager
from src.portfolio.assets.base import Asset


logger = LogManager.get_logger("trading_system.assets.factory")


class AssetFactory(AbstractFactory):
    """Factory for creating different types of assets with exchange integration"""
    
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
        
    async def create_asset_async(self, asset_type: str, params: dict) -> Asset:
        """
        Create an asset asynchronously
        
        Args:
            asset_type: Type of asset to create
            params: Asset creation parameters
            
        Returns:
            Asset: Created asset instance
        """
        return await self.create(asset_type, params)
        
    def discover_assets(self) -> None:
        """
        Auto-discover asset classes with the asset_factory registration
        """
        self.discover_registrable_classes(Asset, 'portfolio.assets', 'asset_factory')
        logger.info(f"Discovered {len(self._registry)} asset types")
        
    def create_multi_assets(self, asset_configs: Dict[str, Dict[str, Any]], 
                           exchange=None) -> Dict[str, Asset]:
        """
        Create multiple assets from configuration
        
        Args:
            asset_configs: Dictionary mapping asset names to their configurations
            exchange: Exchange interface to use for all assets (optional)
            
        Returns:
            Dict[str, Asset]: Dictionary of created assets
        """
        assets = {}
        
        for asset_name, config in asset_configs.items():
            try:
                # Make sure the asset has a name
                if 'name' not in config:
                    config['name'] = asset_name
                    
                # Add exchange if provided
                if exchange and 'exchange' not in config:
                    config['exchange'] = exchange
                
                # Get asset type, default to spot
                asset_type = config.pop('type', 'spot')
                
                # Create the asset
                asset = self.create_asset(asset_type, config)
                assets[asset_name] = asset
                
                logger.info(f"Created {asset_type} asset: {asset_name}")
            except Exception as e:
                logger.error(f"Error creating asset {asset_name}: {str(e)}")
        
        return assets
        
    def get_supported_asset_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all supported asset types
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping asset types to their metadata
        """
        return self.get_registered_items()
    
    def get_asset_type_info(self, asset_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific asset type
        
        Args:
            asset_type: Asset type name
            
        Returns:
            Optional[Dict[str, Any]]: Asset type metadata or None if not found
        """
        all_types = self.get_registered_items()
        return all_types.get(asset_type.lower())
        
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
            # Initialize async exchange if needed
            await exchange._init_async_exchange()
            
            # Load markets
            markets = await exchange.async_exchange.load_markets()
            
            logger.info(f"Initialized {len(markets)} markets from exchange")
        except Exception as e:
            logger.error(f"Error initializing markets: {str(e)}")