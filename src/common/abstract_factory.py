#!/usr/bin/env python3
# src/common/abstract_factory.py

import copy
import json
import hashlib
import inspect
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, ClassVar, Set, List
import asyncio
import logging

from .config import ConfigManager


# Generic type variable
T = TypeVar('T')


class AbstractFactory(ABC):
    """Enhanced abstract factory base class with singleton support"""
    
    # Class-level cache for singleton instances
    _instances: ClassVar[Dict[str, 'AbstractFactory']] = {}
    
    def __init__(self, config: ConfigManager):
        """
        Initialize abstract factory
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(f"system.{self.__class__.__name__.lower()}")
        self._registry: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_instance(cls, config) -> 'AbstractFactory':
        """
        Get singleton factory instance
        
        Args:
            config: Configuration object
            
        Returns:
            AbstractFactory: Factory instance
        """
        key = cls.__name__
        if key not in cls._instances:
            cls._instances[key] = cls(config)
        return cls._instances[key]
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> T:
        """
        Create object (template method)
        
        Args:
            name: Object name
            params: Creation parameters
            
        Returns:
            T: Created object
        """
        try:
            # 1. Resolve object name
            name = await self._resolve_name(name)
            
            # 2. Get concrete class
            concrete_class = await self._get_concrete_class(name)
            
            # 3. Get or create cached instance
            return await self._get_or_create(name, concrete_class, params)
            
        except Exception as e:
            self.logger.error(f"Creation failed: {e}")
            raise
    
    def create_sync(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> T:
        """
        Synchronous creation interface
        
        Args:
            name: Object name
            params: Creation parameters
            
        Returns:
            T: Created object
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.create(name, params))
    
    @abstractmethod
    async def _get_concrete_class(self, name: str) -> Type[T]:
        """
        Get concrete class (to be implemented by subclasses)
        
        Args:
            name: Object name
            
        Returns:
            Type[T]: Object type
        """
        pass
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve object name (can be overridden by subclasses)
        
        Args:
            name: Object name
            
        Returns:
            str: Resolved name
        """
        if name is None:
            raise ValueError("Name must be provided")
        return name.lower()
    
    def register(self, name: str, creator: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register creator
        
        Args:
            name: Object name
            creator: Creator function or class
            metadata: Associated metadata
        """
        name = name.lower()
        self._registry[name] = creator
        if metadata:
            self._metadata[name] = metadata
        self.logger.debug(f"Registered {name} to {self.__class__.__name__}")
    
    def get_registered_items(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered items
        
        Returns:
            Dict[str, Dict[str, Any]]: Registered items dictionary with creators and metadata
        """
        result = {}
        for name, creator in self._registry.items():
            item_info = {
                'creator': str(creator),
                'metadata': self._metadata.get(name, {})
            }
            result[name] = item_info
        return result
    
    def _create_cache_key(self, name: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Create reliable cache key
        
        Args:
            name: Object name
            params: Parameters
            
        Returns:
            str: Cache key
        """
        if not params:
            return name
            
        # Use JSON serialization for consistency
        try:
            # Sort keys for consistency
            param_json = json.dumps(params, sort_keys=True)
            # Generate MD5 hash
            param_hash = hashlib.md5(param_json.encode()).hexdigest()
            return f"{name}_{param_hash}"
        except TypeError:
            # Fallback for non-serializable parameters
            return f"{name}_{id(params)}"
    
    async def _get_or_create(self, name: str, class_type: Type[T], params: Optional[Dict[str, Any]] = None,
                            dependencies: Optional[Dict[str, Any]] = None) -> T:
        """
        Get cached object or create new instance
        
        Args:
            name: Object name
            class_type: Object type
            params: Creation parameters
            dependencies: Dependency objects to inject
            
        Returns:
            T: Created object
        """
        cache_key = self._create_cache_key(name, params)
        
        # Check cache
        if cache_key in self._cache:
            self.logger.debug(f"Returning {name} from cache")
            return self._cache[cache_key]
        
        # Create new instance
        self.logger.info(f"Creating new {name} instance")
        
        try:
            # Merge default and provided parameters
            merged_params = self._get_default_params(name).copy() if params is None else {}
            if params:
                merged_params.update(params)
            
            # Create instance
            instance = class_type(self.config, merged_params)
            
            # Inject dependencies
            if dependencies:
                for attr_name, dependency in dependencies.items():
                    setattr(instance, attr_name, dependency)
            
            # Initialize if needed
            if hasattr(instance, 'initialize'):
                await instance.initialize()
            
            # Cache instance
            self._cache[cache_key] = instance
            
            return instance
            
        except Exception as e:
            self.logger.error(f"Failed to create {name}: {e}")
            raise ValueError(f"Failed to create {name}: {e}")
    
    def _get_default_params(self, name: str) -> Dict[str, Any]:
        """
        Get default parameters
        
        Args:
            name: Object name
            
        Returns:
            Dict[str, Any]: Default parameters
        """
        return {}
    
    async def clear_cache(self, name: Optional[str] = None) -> None:
        """
        Clear cache
        
        Args:
            name: Object name to clear (None clears all)
        """
        if name:
            # Remove specific entries
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{name}_")]
            
            for key in keys_to_remove:
                instance = self._cache[key]
                
                # Shutdown if applicable
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
                
                del self._cache[key]
            
            self.logger.info(f"Cleared cached instances of {name}")
        else:
            # Clear all cache
            for instance in self._cache.values():
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
            
            self._cache.clear()
            self.logger.info("Cleared all cached instances")
    
    def with_config_scope(self, scope: str) -> 'AbstractFactory':
        """
        Create factory view with configuration subset
        
        Args:
            scope: Configuration scope path
            
        Returns:
            AbstractFactory: Factory instance with scoped configuration
        """
        scoped_config = self.config.get(scope, default={})
        factory_copy = copy.copy(self)
        factory_copy.config = scoped_config
        return factory_copy
    
    async def _load_class_from_path(self, name: str, base_class: Type) -> Type[T]:
        """
        Load class from path or reference
        
        Args:
            name: Registered name
            base_class: Expected base class type
        
        Returns:
            Type[T]: Loaded class
        """
        if name not in self._registry:
            raise ValueError(f"Unknown type: {name}")
        
        class_path = self._registry[name]
        
        # Return class if valid
        if inspect.isclass(class_path) and issubclass(class_path, base_class):
            return class_path
        
        # Import from string path
        if isinstance(class_path, str):
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                loaded_class = getattr(module, class_name)
                
                if not inspect.isclass(loaded_class) or not issubclass(loaded_class, base_class):
                    raise ValueError(f"{class_name} is not a valid {base_class.__name__} subclass")
                
                # Update registry
                self._registry[name] = loaded_class
                
                return loaded_class
                
            except ImportError as e:
                self.logger.error(f"Module import error {module_path}: {str(e)}")
                raise ValueError(f"Module import failed: {str(e)}")
            except AttributeError as e:
                self.logger.error(f"Class {class_name} not found in {module_path}: {str(e)}")
                raise ValueError(f"Class not found: {str(e)}")
        
        raise TypeError(f"Invalid registration type: {type(class_path)}")
    
    def discover_registrable_classes(self, base_class: Type, package_path: str, factory_type: str = None) -> None:
        """
        Auto-discover classes with registration decorator
        
        Args:
            base_class: Base class type
            package_path: Package path
            factory_type: Factory type identifier (defaults to lowercase class name)
        """
        if factory_type is None:
            factory_type = self.__class__.__name__.lower()
            
        processed_modules: Set[str] = set()
        self._process_package(package_path, base_class, factory_type, processed_modules)
        self.logger.info(f"Auto-discovery completed: Registered {len(self._registry)} items")
    
    def _process_package(self, package_path: str, base_class: Type, factory_type: str, processed_modules: Set[str]) -> None:
        """
        Process package recursively
        
        Args:
            package_path: Package path
            base_class: Base class type
            factory_type: Factory type identifier
            processed_modules: Set of processed modules
        """
        if package_path in processed_modules:
            return
            
        processed_modules.add(package_path)
        
        try:
            package = importlib.import_module(package_path)
        except ImportError:
            self.logger.warning(f"Failed to import package {package_path}")
            return
            
        if not hasattr(package, '__path__'):
            return
        
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            if is_pkg:
                self._process_package(name, base_class, factory_type, processed_modules)
            else:
                self._process_module(name, base_class, factory_type)
    
    def _process_module(self, module_name: str, base_class: Type, factory_type: str) -> None:
        """
        Process single module
        
        Args:
            module_name: Module name
            base_class: Base class type
            factory_type: Factory type identifier
        """
        try:
            module = importlib.import_module(module_name)
            
            for item_name, obj in inspect.getmembers(module, inspect.isclass):
                if (hasattr(obj, '_factory_registry') and 
                    issubclass(obj, base_class) and 
                    obj != base_class):
                    
                    factory_info = obj._factory_registry.get(factory_type)
                    if factory_info:
                        reg_name = factory_info['name']
                        metadata = factory_info.get('metadata', {})
                        self.register(reg_name, obj, metadata)
                        self.logger.info(f"Auto-registered {reg_name} ({module_name}.{item_name})")
                        
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Module processing error {module_name}: {e}")


# Factory registration decorator
def register_factory_class(factory_type: str, name: Optional[str] = None, **metadata):
    """
    Decorator for auto-registering factory classes
    
    Args:
        factory_type: Factory type (e.g., 'strategy_factory', 'trading_mode_factory')
        name: Registration name (defaults to transformed class name)
        **metadata: Additional metadata
    """
    def decorator(cls):
        if not hasattr(cls, '_factory_registry'):
            cls._factory_registry = {}
        
        reg_name = name
        if reg_name is None:
            reg_name = cls.__name__
            for suffix in ['Strategy', 'TradingMode']:
                reg_name = reg_name.replace(suffix, '')
            reg_name = reg_name.lower()
        
        cls._factory_registry[factory_type] = {
            'name': reg_name,
            'metadata': metadata
        }
        
        return cls
    return decorator
