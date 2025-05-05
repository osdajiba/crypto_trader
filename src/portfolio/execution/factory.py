#!/usr/bin/env python3
# src/portfolio/execution/factory.py

import asyncio
from enum import Enum
import importlib
import inspect
from typing import Dict, Optional, Any, Type, Set

from src.common.abstract_factory import AbstractFactory
from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.execution.base import BaseExecutionEngine


class ExecutionMode(Enum):
    """Centralize the definition of execution modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    

class ExecutionFactory(AbstractFactory):
    """
    Factory for creating execution engines.
    
    This factory manages the creation of execution engines based on the desired mode
    (live, paper, backtest) and provides a consistent interface for accessing and
    configuring execution engines.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize execution factory
        
        Args:
            config: Configuration manager
        """
        super().__init__(config)
        self.logger = LogManager.get_logger("execution.factory")
        
        # Track created engines
        self._created_engines = {}
        
        # Creation hooks
        self._creation_hooks = {}
        
        # Register default execution engines
        self._register_default_execution_engines()
        
        # Discover additional execution engines
        self._discover_execution_engines()
    
    def _register_default_execution_engines(self) -> None:
        """Register built-in execution engines"""
        # Register main execution engine types with their class paths
        self.register("backtest", "src.portfolio.execution.backtest.BacktestExecutionEngine", {"mode": "backtest"})
        self.register("paper", "src.portfolio.execution.paper.PaperExecutionEngine", {"mode": "paper"})
        self.register("live", "src.portfolio.execution.live.LiveExecutionEngine", {"mode": "live"})
        
        # Register alias for simple_backtest -> backtest
        self.register("simple_backtest", "src.portfolio.execution.backtest.BacktestExecutionEngine", {"mode": "simple_backtest"})
        
        self.logger.info("Registered default execution engines")
    
    def _discover_execution_engines(self) -> None:
        """Discover additional execution engines using auto-discovery"""
        try:
            # Use AbstractFactory's discovery mechanism to find additional engine implementations
            self.discover_registrable_classes(
                BaseExecutionEngine, 
                "src.portfolio.execution", 
                "execution_factory"
            )
            self.logger.debug("Completed execution engine discovery")
        except Exception as e:
            self.logger.warning(f"Error during execution engine discovery: {e}")
    
    def register_creation_hook(self, mode: str, hook: callable) -> None:
        """
        Register a hook to be called before engine creation
        
        Args:
            mode: Execution mode
            hook: Hook function to call
        """
        if mode not in self._creation_hooks:
            self._creation_hooks[mode] = set()
            
        self._creation_hooks[mode].add(hook)
        self.logger.debug(f"Registered creation hook for {mode} execution engines")
    
    async def _run_creation_hooks(self, mode: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run creation hooks for an execution mode
        
        Args:
            mode: Execution mode
            params: Engine creation parameters
            
        Returns:
            Dict[str, Any]: Modified parameters
        """
        # Run global hooks first
        global_hooks = self._creation_hooks.get('*', set())
        for hook in global_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    params = await hook(mode, params)
                else:
                    params = hook(mode, params)
            except Exception as e:
                self.logger.warning(f"Error in global creation hook: {e}")
        
        # Run mode specific hooks
        mode_hooks = self._creation_hooks.get(mode, set())
        for hook in mode_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    params = await hook(params)
                else:
                    params = hook(params)
            except Exception as e:
                self.logger.warning(f"Error in {mode} creation hook: {e}")
                
        return params
    
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
            "backtest": ExecutionMode.BACKTEST.value,
            "paper": ExecutionMode.PAPER.value,
            "live": ExecutionMode.LIVE.value
        }
        
        return mode_mapping.get(operational_mode, ExecutionMode.BACKTEST.value)
    
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
            supported = ", ".join(self._registry.keys())
            raise ValueError(f"Execution engine not found: {name}. Supported engines: {supported}")
            
        # Get class path and load class
        return await self._load_class_from_path(name, BaseExecutionEngine)
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> BaseExecutionEngine:
        """
        Create execution engine
        
        Args:
            name: Execution engine name or None to use default
            params: Additional parameters
            
        Returns:
            Execution engine instance
        """
        params = params or {}
        
        # Resolve execution engine name
        resolved_name = await self._resolve_name(name)
        
        # Apply creation hooks
        params = await self._run_creation_hooks(resolved_name, params)
        
        # Get concrete class and metadata
        concrete_class = await self._get_concrete_class(resolved_name)
        engine_metadata = self._metadata.get(resolved_name, {})
        
        # Create combined parameters
        combined_params = dict(engine_metadata)
        if params:
            combined_params.update(params)
        
        # Create instance
        mode = combined_params.get("mode", resolved_name)
        historical_data = combined_params.get("historical_data")
        
        self.logger.info(f"Creating {resolved_name} execution engine with mode: {mode}")
        
        try:
            # Create engine instance
            engine = concrete_class(self.config, mode, historical_data)
            
            # Initialize engine
            await engine.initialize()
                
            # Store in created engines
            self._created_engines[resolved_name] = engine
                
            self.logger.info(f"Created execution engine: {resolved_name}")
            return engine
        except Exception as e:
            self.logger.error(f"Error creating execution engine {resolved_name}: {e}")
            raise
    
    def get_engine_instance(self, name: str) -> Optional[BaseExecutionEngine]:
        """
        Get a previously created execution engine by name
        
        Args:
            name: Engine name
            
        Returns:
            Optional[BaseExecutionEngine]: Engine instance if found, None otherwise
        """
        return self._created_engines.get(name)
    
    def get_available_engines(self) -> Dict[str, str]:
        """
        Get available execution engines
        
        Returns:
            Dict of engine names and class paths
        """
        return {name: path for name, path in self._registry.items()}
    
    @staticmethod
    def map_execution_mode(system_mode: str) -> str:
        """
        Map system operational mode to execution mode
        
        Args:
            system_mode: System operational mode
            
        Returns:
            Corresponding execution mode
        """
        mode_mapping = {
            "backtest": ExecutionMode.BACKTEST.value,
            "paper": ExecutionMode.PAPER.value,
            "live": ExecutionMode.LIVE.value
        }
        
        return mode_mapping.get(system_mode, ExecutionMode.BACKTEST.value)


def get_execution_factory(config: ConfigManager) -> ExecutionFactory:
    """
    Get execution factory singleton instance
    
    Args:
        config: Configuration manager
        
    Returns:
        ExecutionFactory instance
    """
    return ExecutionFactory.get_instance(config)