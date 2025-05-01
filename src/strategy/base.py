#!/usr/bin/env python3
# src/strategy/base.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable, Any, List, Type, Union
from abc import ABC, abstractmethod
import time
import asyncio

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor


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
        
        # Performance monitoring
        self._performance_stats = {
            'signals_generated': 0,
            'execution_time': 0,
            'last_run': 0,
            'avg_execution_time': 0,
            'runs': 0
        }
        
        # Basic configuration
        self.lookback_period = self.params.get("lookback_period", 60)
        
        # Async execution
        self.executor = AsyncExecutor()
        
        # Initialize factors
        self._init_factors()
    
    def _init_factors(self) -> None:
        """Initialize strategy factors - to be implemented by subclasses"""
        pass
    
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
        self._required_data_points = max(max_window, 2) + 1
        
        # Update lookback period if needed
        if self._required_data_points > self.lookback_period:
            self.lookback_period = self._required_data_points + 5  # Add buffer
    
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
            start_time = time.time()
            
            # Manage data buffer
            sufficient_history = await self._manage_data_buffer(data, symbol)
            
            # Check if we have enough data to generate signals
            if not sufficient_history:
                self.logger.debug(f"Insufficient history for {symbol}, need {self._required_data_points} points")
                return pd.DataFrame()
            
            # Generate signals
            signals = await self._generate_signals(self._data_buffer[symbol])
            
            # Add symbol if not present
            if not signals.empty and 'symbol' not in signals.columns:
                signals['symbol'] = symbol
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._performance_stats['execution_time'] += execution_time
            self._performance_stats['runs'] += 1
            self._performance_stats['avg_execution_time'] = self._performance_stats['execution_time'] / self._performance_stats['runs']
            self._performance_stats['last_run'] = time.time()
            
            if not signals.empty:
                self._performance_stats['signals_generated'] += len(signals)
                self.logger.info(f"Generated {len(signals)} signals for {symbol} in {execution_time:.3f}s")
            
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
        
        # Ensure new_data has datetime column
        if 'datetime' not in new_data.columns and 'timestamp' in new_data.columns:
            try:
                # Try to convert timestamp to datetime
                new_data = new_data.copy()
                if pd.api.types.is_numeric_dtype(new_data['timestamp']):
                    new_data['datetime'] = pd.to_datetime(new_data['timestamp'], unit='ms')
                else:
                    new_data['datetime'] = pd.to_datetime(new_data['timestamp'])
            except Exception as e:
                self.logger.warning(f"Error converting timestamp to datetime: {e}")
        
        # Add new data to buffer
        if not new_data.empty:
            # Handle different index types gracefully
            if isinstance(new_data.index, pd.DatetimeIndex) and not isinstance(self._data_buffer[symbol].index, pd.DatetimeIndex):
                # Convert existing buffer to use DatetimeIndex if new data uses it
                if not self._data_buffer[symbol].empty:
                    self._data_buffer[symbol] = self._data_buffer[symbol].set_index('datetime')
            
            # Handle DataFrame concatenation
            try:
                self._data_buffer[symbol] = pd.concat([self._data_buffer[symbol], new_data]).drop_duplicates()
            except Exception as e:
                self.logger.error(f"Error concatenating data: {e}")
                # Try a more robust approach
                buffer_cols = set(self._data_buffer[symbol].columns)
                new_cols = set(new_data.columns)
                
                # Use common columns only
                common_cols = buffer_cols.intersection(new_cols)
                if common_cols:
                    self._data_buffer[symbol] = pd.concat([
                        self._data_buffer[symbol][list(common_cols)], 
                        new_data[list(common_cols)]
                    ]).drop_duplicates()
        
        # Sort data by datetime if present
        if 'datetime' in self._data_buffer[symbol].columns:
            self._data_buffer[symbol] = self._data_buffer[symbol].sort_values('datetime')
        
        # If we already have sufficient history, just keep the window we need
        if self._has_sufficient_history[symbol]:
            # Keep only required data points plus some buffer
            buffer_size = min(1000, max(100, self._required_data_points * 2))  # Keep reasonable buffer
            self._data_buffer[symbol] = self._data_buffer[symbol].tail(buffer_size)
            return True
        
        # Check if we have enough data now
        if len(self._data_buffer[symbol]) >= self._required_data_points:
            self._has_sufficient_history[symbol] = True
            self.logger.info(f"Sufficient history collected for {symbol}: {len(self._data_buffer[symbol])} points")
            return True
        
        # Not enough data yet
        self.logger.debug(f"Collecting history for {symbol}: {len(self._data_buffer[symbol])}/{self._required_data_points} points")
        return False

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
            self.logger.warning(f"Factor '{factor_name}' not registered")
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
                # Check cache first
                if symbol and symbol in self._factor_cache and factor_name in self._factor_cache[symbol]:
                    cached_factor = self._factor_cache[symbol][factor_name]
                    # Check if cache is valid for the current data
                    if isinstance(cached_factor, pd.Series) and len(cached_factor) == len(data):
                        return cached_factor
                
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
        
        # Return empty series on error
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
            await self.executor.start()
            self._is_initialized = True
            self.logger.info(f"Strategy {self.__class__.__name__} initialized")
    
    async def shutdown(self) -> None:
        """Clean up strategy resources"""
        if self._is_initialized:
            # Clear data
            self._data_buffer.clear()
            self._factor_cache.clear()
            self._has_sufficient_history.clear()
            
            # Shutdown executor
            await self.executor.shutdown()
            
            self._is_initialized = False
            self.logger.info(f"Strategy {self.__class__.__name__} shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        return {
            'strategy': self.__class__.__name__,
            'runs': self._performance_stats['runs'],
            'signals_generated': self._performance_stats['signals_generated'],
            'avg_execution_time': self._performance_stats['avg_execution_time'],
            'last_run': self._performance_stats['last_run'],
            'factors_registered': len(self._factor_registry),
            'symbols_monitored': len(self._data_buffer),
            'required_data_points': self._required_data_points
        }
    
    def register_ma_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Utility function to register a moving average factor
        
        Args:
            name: Factor name
            window_size: Window size for the moving average
            price_column: Column to calculate MA on (default: close)
        """
        def calculate_ma(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column in data.columns:
                return data[price_column].rolling(window=window_size, min_periods=1).mean()
            return pd.Series(index=data.index)
        
        self.register_factor(name, window_size, calculate_ma)
        self.logger.debug(f"Registered MA factor '{name}' with window size {window_size}")
    
    def register_ema_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Utility function to register an exponential moving average factor
        
        Args:
            name: Factor name
            window_size: Window size for the EMA
            price_column: Column to calculate EMA on (default: close)
        """
        def calculate_ema(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column in data.columns:
                return data[price_column].ewm(span=window_size, adjust=False).mean()
            return pd.Series(index=data.index)
        
        self.register_factor(name, window_size, calculate_ema)
        self.logger.debug(f"Registered EMA factor '{name}' with window size {window_size}")