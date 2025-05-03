#!/usr/bin/env python3
# src/common/abstract_factory.py

import importlib
import inspect
from typing import Dict, Any, Optional, Type, ClassVar, List, Callable
import asyncio
import functools
import pkgutil

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager


class AbstractFactoryError(Exception):
    """Base class for abstract factory errors"""
    pass


class ComponentRegistrationError(AbstractFactoryError):
    """Error raised when component registration fails"""
    pass


class ComponentLoadError(AbstractFactoryError):
    """Error raised when component loading fails"""
    pass


def register_factory_class(factory_type: str, name: str, **metadata):
    """
    Decorator to register classes with the appropriate factory
    
    Args:
        factory_type: Type of factory to register with
        name: Name to register component as
        **metadata: Additional metadata for the component
    """
    def decorator(cls):
        if not hasattr(cls, '__factory_registrations'):
            cls.__factory_registrations = []
        
        cls.__factory_registrations.append({
            'factory_type': factory_type,
            'name': name,
            'metadata': metadata
        })
        return cls
    return decorator


class AbstractFactory:
    """Abstract base class for component factories with dynamic component discovery"""
    
    _instances: ClassVar[Dict[Type, Any]] = {}
    
    @classmethod
    def get_instance(cls, config: ConfigManager):
        """
        Get or create singleton instance of factory
        
        Args:
            config: Configuration manager
            
        Returns:
            AbstractFactory: Singleton instance
        """
        if cls not in cls._instances:
            cls._instances[cls] = cls(config)
        return cls._instances[cls]
    
    def __init__(self, config: ConfigManager):
        """
        Initialize abstract factory
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.logger = LogManager.get_logger(self.__class__.__name__)
        self._registry = {}
        self._metadata = {}
    
    def register(self, name: str, class_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Register component with factory
        
        Args:
            name: Component name
            class_path: Fully qualified class path
            metadata: Component metadata
        """
        if not isinstance(name, str):
            name = str(name)
            
        self._registry[name.lower()] = class_path
        if metadata:
            self._metadata[name.lower()] = metadata
        
        self.logger.debug(f"Registered component: {name} -> {class_path}")
    
    def get_registered_items(self) -> Dict[str, Dict[str, Any]]:
        """
        Get registered items with metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Component registry
        """
        result = {}
        for name, class_path in self._registry.items():
            metadata = self._metadata.get(name, {})
            result[name] = {
                'class_path': class_path,
                'metadata': metadata
            }
        return result
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Create component instance
        
        Args:
            name: Component name
            params: Parameters for component initialization
            
        Returns:
            Any: Component instance
        """
        try:
            resolved_name = await self._resolve_name(name)
            concrete_class = await self._get_concrete_class(resolved_name)
            
            if not concrete_class:
                raise ComponentLoadError(f"Failed to load component class for {resolved_name}")
                
            instance = concrete_class(self.config, params)
            
            # Initialize if method exists
            if hasattr(instance, 'initialize') and callable(instance.initialize):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    instance.initialize()
                    
            self.logger.info(f"Created component instance: {resolved_name}")
            return instance
            
        except Exception as e:
            self.logger.error(f"Error creating component {name}: {str(e)}")
            raise
    
    async def _get_concrete_class(self, name: str) -> Type:
        """
        Get concrete class for component (to be implemented by subclasses)
        
        Args:
            name: Component name
            
        Returns:
            Type: Component class
        """
        return await self._load_class_from_path(name)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve component name (to be implemented by subclasses)
        
        Args:
            name: Component name or None
            
        Returns:
            str: Resolved component name
        """
        raise NotImplementedError("Subclasses must implement _resolve_name")
    
    async def _load_class_from_path(self, name: str, base_class: Optional[Type] = None) -> Type:
        """
        Dynamically load class from registry path
        
        Args:
            name: Component name
            base_class: Expected base class for validation
            
        Returns:
            Type: Component class
        """
        if name not in self._registry:
            raise ComponentLoadError(f"Component not registered: {name}")
            
        class_path = self._registry[name]
        
        # Extract module path and class name
        try:
            module_path, class_name = class_path.rsplit('.', 1)
        except ValueError:
            raise ComponentLoadError(f"Invalid class path: {class_path}")
        
        try:
            # Import module
            loop = asyncio.get_event_loop()
            module = await loop.run_in_executor(
                None, 
                functools.partial(importlib.import_module, module_path)
            )
            
            # Get class
            if not hasattr(module, class_name):
                raise ComponentLoadError(f"Class not found in module: {class_name}")
                
            component_class = getattr(module, class_name)
            
            # Validate base class if provided
            if base_class and not issubclass(component_class, base_class):
                raise ComponentLoadError(
                    f"Component class {class_name} must be subclass of {base_class.__name__}"
                )
                
            return component_class
            
        except ImportError as e:
            raise ComponentLoadError(f"Error importing component module {module_path}: {str(e)}")
        except Exception as e:
            raise ComponentLoadError(f"Error loading component class {class_path}: {str(e)}")
    
    def discover_registrable_classes(self, base_class: Type, module_path: str, 
                                   log_prefix: str = "factory") -> None:
        """
        Auto-discover registrable classes
        
        Args:
            base_class: Base class that components must derive from
            module_path: Base module path to search
            log_prefix: Prefix for log messages
        """
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Scan for submodules
            if hasattr(module, '__path__'):
                for _, name, _ in pkgutil.iter_modules(module.__path__, module.__name__ + '.'):
                    try:
                        submodule = importlib.import_module(name)
                        
                        # Scan classes in submodule
                        for attr_name in dir(submodule):
                            attr = getattr(submodule, attr_name)
                            
                            # Check if it's a class and subclass of base_class
                            if (inspect.isclass(attr) and issubclass(attr, base_class) and 
                                attr != base_class and not inspect.isabstract(attr)):
                                
                                # Check for registration metadata
                                if hasattr(attr, '_AbstractFactory_'):
                                    for reg in attr._AbstractFactory_:
                                        if reg['factory_type'] == log_prefix:
                                            self.register(
                                                reg['name'], 
                                                f"{attr.__module__}.{attr.__name__}", 
                                                reg['metadata']
                                            )
                                            self.logger.debug(
                                                f"Auto-discovered component: {reg['name']} -> {attr.__module__}.{attr.__name__}"
                                            )
                    except Exception as e:
                        self.logger.warning(f"Error scanning submodule {name}: {str(e)}")
                        continue
                        
        except ImportError as e:
            self.logger.warning(f"Error importing base module {module_path}: {str(e)}")
        except Exception as e:
            self.logger.warning(f"Error during component discovery: {str(e)}")