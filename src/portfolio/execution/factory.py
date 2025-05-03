#!/usr/bin/env python3
# src/portfolio/execution/factory.py

import asyncio
from enum import Enum
import importlib
import inspect
from typing import Dict, Optional, Any, Type

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.execution.engine import ExecutionEngine


class Execution(Enum):
    """Centralize the definition of data source types"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    
    
class ExecutionFactory(AbstractFactory):
    """Factory for creating execution engines"""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize execution factory
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.logger = LogManager.get_logger("execution.factory")
        
        # Register default execution engines
        self._register_default_execution_engines()
        
        # Discover additional execution engines
        self._discover_execution_engines()
    
    def _register_default_execution_engines(self) -> None:
        """Register built-in execution engines"""
        # Register main execution engine
        self.register("default", "src.portfolio.execution.engine.ExecutionEngine")
        
        # Map mode names to the same engine (different modes handled internally)
        self.register("live", "src.portfolio.execution.engine.ExecutionEngine", {"mode": "live"})
        self.register("paper", "src.portfolio.execution.engine.ExecutionEngine", {"mode": "paper"})
        self.register("backtest", "src.portfolio.execution.engine.ExecutionEngine", {"mode": "backtest"})
        self.register("simple_backtest", "src.portfolio.execution.engine.ExecutionEngine", {"mode": "simple_backtest"})
        
        self.logger.info("Registered default execution engines")
    
    def _discover_execution_engines(self) -> None:
        """Discover additional execution engines using auto-discovery"""
        try:
            self.discover_registrable_classes(
                ExecutionEngine, 
                "src.portfolio.execution", 
                "execution_factory"
            )
            self.logger.debug("Completed execution engine discovery")
        except Exception as e:
            self.logger.warning(f"Error during execution engine discovery: {str(e)}")
    
    async def _resolve_name(self, name: Optional[str] = None) -> str:
        """
        Resolve execution engine name
        
        Args:
            name: Engine name or None to use default
            
        Returns:
            Resolved engine name
        """
        if name:
            return name.lower()
            
        # Determine appropriate execution mode from config
        operational_mode = self.config.get("system", "operational_mode", default="backtest")
        
        # Map system mode to execution engine
        mode_mapping = {
            "backtest": Execution.BACKTEST.value,
            "paper": Execution.PAPER.value,
            "live": Execution.LIVE.value
        }
        
        return mode_mapping.get(operational_mode, "default")
    
    async def _get_concrete_class(self, name: str) -> Type:
        """
        Get concrete execution engine class
        
        Args:
            name: Execution engine name
            
        Returns:
            Execution engine class
            
        Raises:
            ValueError: If engine not found
        """
        if name not in self._registry:
            raise ValueError(f"Execution engine not found: {name}")
            
        # Get class path and load class
        return await self._load_class_from_path(name, ExecutionEngine)
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> ExecutionEngine:
        """
        Create execution engine
        
        Args:
            name: Execution engine name or None to use default
            params: Additional parameters
            
        Returns:
            Execution engine instance
        """
        resolved_name = await self._resolve_name(name)
        concrete_class = await self._get_concrete_class(resolved_name)
        
        # Get metadata for the engine
        engine_metadata = self._metadata.get(resolved_name, {})
        
        # Create combined parameters
        combined_params = dict(engine_metadata)
        if params:
            combined_params.update(params)
        
        # Create instance
        engine = concrete_class(self.config, combined_params.get("mode", resolved_name), None)
        
        # Initialize if needed
        if hasattr(engine, 'initialize') and callable(engine.initialize):
            if asyncio.iscoroutinefunction(engine.initialize):
                await engine.initialize()
            else:
                engine.initialize()
                
        self.logger.info(f"Created execution engine: {resolved_name}")
        return engine
    
    def get_available_engines(self) -> Dict[str, str]:
        """
        Get available execution engines
        
        Returns:
            Dict of engine names and class paths
        """
        return {name: path for name, path in self._registry.items()}


def get_execution_factory(config: ConfigManager) -> ExecutionFactory:
    """
    Get execution factory singleton instance
    
    Args:
        config: Configuration manager
        
    Returns:
        ExecutionFactory instance
    """
    return ExecutionFactory.get_instance(config)