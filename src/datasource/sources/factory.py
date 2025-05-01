#!/usr/bin/env python3
# src/datasource/sources/factory.py

"""
Data source factory module.
Provides factory methods for creating data source instances according to the factory pattern standard.
"""

from typing import Dict, Optional, Any, Type, List, Union
import asyncio

from src.common.abstract_factory import AbstractFactory
from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import DataSource
from src.datasource.sources.base import BaseDataSource, DataSourceError


class DataSourceFactory(AbstractFactory):
    """
    Factory for creating data source instances
    
    This factory creates and manages instances of data sources, providing
    consistent interface for source creation, discovery, and metadata access.
    """
    
    _instances = {}
    
    def __init__(self, config: ConfigManager):
        """
        Initialize data source factory
        
        Args:
            config: Configuration manager instance
        """
        super().__init__(config)
        
        # Initialize logger with proper category
        self.logger = LogManager.get_logger(f"factory.{self.__class__.__name__.lower()}")
        
        # Register built-in data sources
        self._register_default_sources()
        
        # Auto-discover additional data sources
        self._discover_sources()
    
    def _register_default_sources(self) -> None:
        """Register default data sources with consistent metadata"""
        self.register(DataSource.LOCAL.value, "src.datasource.sources.local.LocalDataSource", {
            "description": "Local data source for historical data",
            "features": ["historical", "offline_access", "file_storage"],
            "category": "datasource"
        })
        
        self.register(DataSource.EXCHANGE.value, "src.datasource.sources.exchange.ExchangeDataSource", {
            "description": "Exchange data source for real-time market data",
            "features": ["historical", "realtime", "online_access"],
            "category": "datasource"
        })
        
        self.register(DataSource.HIBRID.value, "src.datasource.sources.hybrid.HybridDataSource", {
            "description": "Hybrid data source combining local and exchange data",
            "features": ["historical", "realtime", "online_access", "offline_access"],
            "category": "datasource"
        })
        
        self.logger.info("Registered default data sources")
    
    def _discover_sources(self) -> None:
        """Auto-discover additional data source modules"""
        try:
            # Discover implementations from sources directory
            module_path = "src.datasource.sources"
            self.discover_registrable_classes(
                BaseDataSource, 
                module_path, 
                "datasource_factory"
            )
            self.logger.debug(f"Discovered additional data sources from {module_path}")
        except Exception as e:
            self.logger.error(f"Error during data source discovery: {str(e)}")
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve source name, falling back to default from config if none provided
        
        Args:
            name: Source name or None
            
        Returns:
            str: Resolved source name
            
        Raises:
            ValueError: If no valid source name could be resolved
        """
        if name is None:
            # Get default from configuration
            default_source = self.config.get("data", "source", "type", default=DataSource.LOCAL.value)
            self.logger.info(f"Using default data source: {default_source}")
            return default_source
            
        try:
            # Try to convert string to enum value for validation
            if isinstance(name, str):
                source_value = name.lower()
                source = DataSource(source_value)
            else:
                # Handle case where enum is passed directly
                source = name
                source_value = source.value
        except ValueError:
            valid_sources = [src.value for src in DataSource]
            raise ValueError(f"Unsupported data source: {name}. Must be one of: {valid_sources}")
            
        # Handle live data override
        use_live_data = self.config.get("data", "source", "live", default=False)
        if use_live_data and source_value == DataSource.LOCAL.value:
            source = DataSource.EXCHANGE
            self.logger.info(f"Overriding data source to {source.value} due to use_live_data=True")
            
        return source.value
    
    async def _get_concrete_class(self, name: str) -> Type[BaseDataSource]:
        """
        Get concrete data source class by name
        
        Args:
            name: Source name
            
        Returns:
            Type[BaseDataSource]: Source class
            
        Raises:
            ComponentLoadError: If the component could not be loaded
        """
        return await self._load_class_from_path(name, BaseDataSource)
    
    def get_available_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of all available data sources with metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Source names mapped to their metadata
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            result[name] = metadata
            
        # Ensure all enum values are included
        for source in DataSource:
            if source.value not in result:
                result[source.value] = {
                    "description": self._get_default_description(source),
                    "features": [],
                    "category": "datasource"
                }
                
        return result
    
    def _get_default_description(self, source: DataSource) -> str:
        """
        Get default description for data source when not available
        
        Args:
            source: Data source enum
            
        Returns:
            str: Default description
        """
        descriptions = {
            DataSource.LOCAL: "Local data source for historical data",
            DataSource.EXCHANGE: "Exchange data source for real-time market data",
            DataSource.HIBRID: "Hybrid data source combining local and exchange data"
        }
        return descriptions.get(source, "Unknown data source")
    
    def get_source_features(self, source_name: str) -> List[str]:
        """
        Get features of a specific data source
        
        Args:
            source_name: Name of the source
            
        Returns:
            List[str]: List of features
        """
        metadata = self._metadata.get(source_name.lower(), {})
        return metadata.get('features', [])
    
    async def create_with_config_params(self, name: Optional[str] = None) -> BaseDataSource:
        """
        Create a data source with parameters from configuration
        
        Args:
            name: Optional source name
            
        Returns:
            BaseDataSource: Data source instance
        """
        resolved_name = await self._resolve_name(name)
        
        # Get source-specific configuration
        source_config = self.config.get("data", "sources", resolved_name, default={})
        
        return await self.create(resolved_name, params=source_config)


def get_datasource_factory(config: ConfigManager) -> DataSourceFactory:
    """
    Get or create singleton instance of DataSourceFactory
    
    Args:
        config: Configuration manager
    
    Returns:
        DataSourceFactory: Singleton instance
    """
    return DataSourceFactory.get_instance(config)