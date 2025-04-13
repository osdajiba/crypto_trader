# src/strategy/base_strategy.py

import pandas as pd
from typing import Dict, Optional, Callable, Any, List, Type, Union
from abc import ABC, abstractmethod

from src.common.abstract_factory import AbstractFactory, register_factory_class
from common.config import ConfigManager
from common.logging import LogManager


class BaseStrategy(ABC):
    """
    Base class for trading strategies with factor management.
    Concrete strategies inherit from this and implement _generate_signals.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize base strategy.
        
        Args:
            config: Configuration manager instance
            params: Strategy-specific parameters
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"strategy.{self.__class__.__name__.lower()}")
        self._is_initialized = False
        
        # Factor management
        self._factor_registry = {}  # Factor registry
        self._data_buffer = {}      # Data buffer
        self._required_data_points = 0  # Will be computed based on factors
        self._has_sufficient_history = {}  # Track history per symbol
        self._factor_cache = {}     # Cache for factor values
        
        # Basic configuration
        self.lookback_period = self.params.get("lookback_period", 60)
    
    def register_factor(self, name: str, window_size: int, func: Callable = None, 
                       depends_on: List[str] = None, is_differential: bool = False) -> None:
        """
        Register a factor with its required window size
        
        Args:
            name: Factor name
            window_size: Window size required for this factor
            func: Function to calculate the factor
            depends_on: List of factors this factor depends on
            is_differential: If True, factor requires an extra point for differencing
        """
        self._factor_registry[name] = {
            'window_size': window_size,
            'func': func,
            'depends_on': depends_on or [],
            'is_differential': is_differential
        }
        
        self._update_required_data_points()
    
    def _update_required_data_points(self) -> None:
        """Calculate required data points based on registered factors"""
        max_window = 0
        
        for name, info in self._factor_registry.items():
            window_size = info['window_size']
            if info.get('is_differential', False):
                window_size += 1
            max_window = max(max_window, window_size)
            
            # Check dependencies
            for dep_name in info.get('depends_on', []):
                if dep_name in self._factor_registry:
                    dep_window = self._factor_registry[dep_name]['window_size']
                    if self._factor_registry[dep_name].get('is_differential', False):
                        dep_window += 1
                    max_window = max(max_window, dep_window)
        
        # Set the required data points (minimum 2)
        self._required_data_points = max(max_window, 2)
        
        # Update lookback period if needed
        if self._required_data_points > self.lookback_period:
            self.lookback_period = self._required_data_points
    
    async def process_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process data for a symbol and generate signals.
        
        Args:
            data: Market data
            symbol: Trading pair being processed
            
        Returns:
            Trading signals
        """
        try:
            # Manage data buffer
            sufficient_history = await self._manage_data_buffer(data, symbol)
            
            # Check if we have enough data to generate signals
            if not sufficient_history:
                return pd.DataFrame()
            
            # Get data for signal generation
            combined_data = self._get_combined_data(symbol)
            
            # Generate signals
            signals = await self._generate_signals(combined_data)
            
            # Add symbol if not present
            if not signals.empty and 'symbol' not in signals.columns:
                signals['symbol'] = symbol
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol} data: {str(e)}")
            return pd.DataFrame()
    
    async def _manage_data_buffer(self, new_data: pd.DataFrame, symbol: str) -> bool:
        """
        Manage data buffer for initial collection and updates
        
        Args:
            new_data: New data to add to buffer
            symbol: Symbol being processed
            
        Returns:
            True if we have sufficient history, False otherwise
        """
        # Initialize buffers if needed
        if symbol not in self._data_buffer:
            self._data_buffer[symbol] = pd.DataFrame()
            self._has_sufficient_history[symbol] = False
            self._factor_cache[symbol] = {}
        
        # Add new data to buffer
        self._data_buffer[symbol] = pd.concat([self._data_buffer[symbol], new_data])
        
        # If we already have sufficient history, just keep the window we need
        if self._has_sufficient_history[symbol]:
            # Keep only required data points
            self._data_buffer[symbol] = self._data_buffer[symbol].tail(self._required_data_points)
            return True
        
        # Check if we have enough data now
        if len(self._data_buffer[symbol]) >= self._required_data_points:
            self._has_sufficient_history[symbol] = True
            return True
        
        # Not enough data yet
        return False
    
    def _get_combined_data(self, symbol: str) -> pd.DataFrame:
        """Get data for signal generation"""
        return self._data_buffer[symbol].copy()
    
    def calculate_factor(self, data: pd.DataFrame, factor_name: str, symbol: str = None) -> pd.Series:
        """
        Calculate factor values with caching for efficiency
        
        Args:
            data: Data for factor calculation
            factor_name: Name of factor to calculate
            symbol: Symbol for caching purposes
            
        Returns:
            Calculated factor values
        """
        if factor_name not in self._factor_registry:
            return pd.Series(index=data.index)
        
        factor_info = self._factor_registry[factor_name]
        
        # Check dependencies
        for dep_name in factor_info.get('depends_on', []):
            if dep_name in self._factor_registry and symbol and symbol in self._factor_cache:
                if dep_name not in self._factor_cache[symbol]:
                    self.calculate_factor(data, dep_name, symbol)
        
        # Calculate factor
        if factor_info['func'] and callable(factor_info['func']):
            try:
                # Prepare dependency values
                kwargs = {}
                for dep_name in factor_info.get('depends_on', []):
                    if symbol and symbol in self._factor_cache and dep_name in self._factor_cache[symbol]:
                        kwargs[dep_name] = self._factor_cache[symbol][dep_name]
                
                # Calculate
                factor_values = factor_info['func'](data, **kwargs)
                
                # Cache result
                if symbol:
                    if symbol not in self._factor_cache:
                        self._factor_cache[symbol] = {}
                    self._factor_cache[symbol][factor_name] = factor_values
                
                return factor_values
                
            except Exception as e:
                self.logger.error(f"Error calculating factor '{factor_name}': {str(e)}")
        
        return pd.Series(index=data.index)
    
    @abstractmethod
    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data.
        Must be implemented by concrete strategy classes.
        
        Args:
            data: Market data for signal generation
            
        Returns:
            Signals with 'timestamp', 'symbol', 'action' columns
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize strategy resources"""
        if not self._is_initialized:
            self._is_initialized = True
    
    async def shutdown(self) -> None:
        """Clean up strategy resources"""
        if self._is_initialized:
            self._data_buffer.clear()
            self._factor_cache.clear()
            self._has_sufficient_history.clear()
            self._is_initialized = False


class StrategyFactory(AbstractFactory):
    """Factory for creating strategy instances"""
    
    def __init__(self, config):
        """Initialize strategy factory"""
        super().__init__(config)
        self.default_strategy_type = config.get("strategy", "active", default="dual_ma")
        
        # Register built-in strategies
        self._register_default_strategies()
        self._discover_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies"""
        self.register("dual_ma", "src.strategy.DualMA.DualMAStrategy", {
            "description": "Dual Moving Average Crossover Strategy",
        })
    
    def _discover_strategies(self):
        """Auto-discover strategy modules"""
        try:
            # Auto-register decorated strategies
            self.discover_registrable_classes(BaseStrategy, "src.strategy", "strategy_factory")
        except Exception as e:
            self.logger.error(f"Error during strategy discovery: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseStrategy]:
        """Get strategy class"""
        return await self._load_class_from_path(name, BaseStrategy)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """Resolve strategy name with default fallback"""
        name = name or self.default_strategy_type
        if not name:
            raise ValueError("No strategy name provided and no default in config")
        return name.lower()


# Decorator for registering strategies
def register_strategy(name: Optional[str] = None, **metadata):
    """Decorator for registering strategies"""
    return register_factory_class('strategy_factory', name, **metadata)