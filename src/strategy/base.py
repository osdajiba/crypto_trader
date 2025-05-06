#!/usr/bin/env python3
# src/strategy/base.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Callable, Any, List, Union
from abc import ABC, abstractmethod
import time
import asyncio

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    Provides a unified framework for strategy implementation with
    built-in factor management, data buffering, and performance tracking.
    Concrete strategies inherit from this class and implement the
    _generate_signals method to create trading signals.
    """
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the base strategy.
        
        Args:
            config: Configuration manager instance
            params: Optional strategy parameters
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"strategy.{self.__class__.__name__.lower()}")
        self.id = self.__class__.__name__.lower()
        
        # Internal state tracking
        self._is_initialized = False
        
        # Factor management system
        self._factor_registry = {}
        self._factor_cache = {}
        self._data_buffer = {}
        self._required_data_points = 0
        self._has_sufficient_history = {}
        
        # Performance monitoring
        self._performance_stats = {
            'signals_generated': 0,
            'execution_time': 0,
            'last_run': 0,
            'avg_execution_time': 0,
            'runs': 0
        }
        
        # Set default lookback period
        self.lookback_period = self.params.get("lookback_period", 60)
        
        # Async executor for background tasks
        self.executor = AsyncExecutor()
        
        # Initialize factors
        self._init_factors()
    
    def _init_factors(self) -> None:
        """
        Initialize strategy factors.
        
        This method should be implemented by concrete strategy classes
        to register their required factors.
        """
        pass
    
    def get_strategy_id(self):
        return self.id
        
    def register_factor(self, name: str, window_size: int, func: Callable = None, 
                       depends_on: List[str] = None, is_differential: bool = False) -> None:
        """
        Register a calculation factor with the strategy.
        
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
        
        # Update required data points based on new factor
        self._update_required_data_points()
    
    def _update_required_data_points(self) -> None:
        """
        Calculate required data points based on registered factors.
        
        This determines how much historical data is needed for proper
        factor calculation.
        """
        max_window = 0
        
        # Find the largest window size from all factors
        for name, info in self._factor_registry.items():
            window_size = info['window_size']
            
            # If factor uses differencing, it needs an extra point
            if info.get('is_differential', False):
                window_size += 1
            
            max_window = max(max_window, window_size)
            
            # Also consider dependencies
            for dep_name in info.get('depends_on', []):
                if dep_name in self._factor_registry:
                    dep_window = self._factor_registry[dep_name]['window_size']
                    if self._factor_registry[dep_name].get('is_differential', False):
                        dep_window += 1
                    max_window = max(max_window, dep_window)
        
        # Add 1 for safety and set minimum of 2 for crossover detection
        self._required_data_points = max(max_window, 2) + 1
        
        # Update lookback period if needed
        if self._required_data_points > self.lookback_period:
            self.lookback_period = self._required_data_points + 5  # Add buffer
    
    async def process_data(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        Process market data for a symbol and generate signals.
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol (optional, extracted from data if available)
            
        Returns:
            pd.DataFrame: Trading signals
        """
        try:
            # Track performance
            start_time = time.time()
            
            # Determine symbol from data if not provided
            if symbol is None and 'symbol' in data.columns and not data.empty:
                symbol = data['symbol'].iloc[0]
            elif symbol is None:
                symbol = self.params.get("symbol", "unknown")
            
            # Manage data buffer for the symbol
            sufficient_history = await self._manage_data_buffer(data, symbol)
            
            # Check if we have enough data to generate signals
            if not sufficient_history:
                self.logger.debug(f"Insufficient history for {symbol}, need {self._required_data_points} points")
                return pd.DataFrame()
            
            # Generate signals using strategy-specific logic
            signals = await self._generate_signals(self._data_buffer[symbol])
            
            # Ensure symbol is set in signals
            if not signals.empty and 'symbol' not in signals.columns:
                signals['symbol'] = symbol
            
            # Update performance stats
            execution_time = time.time() - start_time
            self._performance_stats['execution_time'] += execution_time
            self._performance_stats['runs'] += 1
            self._performance_stats['avg_execution_time'] = (
                self._performance_stats['execution_time'] / 
                self._performance_stats['runs']
            )
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
        Manage data buffer for a symbol.
        
        Args:
            new_data: New data to add to buffer
            symbol: Symbol being processed
            
        Returns:
            bool: True if we have sufficient history for signal generation
        """
        # Initialize buffers if needed
        if symbol not in self._data_buffer:
            self._data_buffer[symbol] = pd.DataFrame()
            self._has_sufficient_history[symbol] = False
            self._factor_cache[symbol] = {}
        
        # Ensure datetime column exists
        if 'datetime' not in new_data.columns and 'timestamp' in new_data.columns:
            try:
                # Copy to avoid modifying the original data
                new_data = new_data.copy()
                
                # Convert timestamp to datetime
                if pd.api.types.is_numeric_dtype(new_data['timestamp']):
                    new_data['datetime'] = pd.to_datetime(new_data['timestamp'], unit='ms')
                else:
                    new_data['datetime'] = pd.to_datetime(new_data['timestamp'])
                    
            except Exception as e:
                self.logger.warning(f"Error converting timestamp to datetime: {e}")
        
        # Add new data to buffer if not empty
        if not new_data.empty:
            try:
                # Handle different index types
                if isinstance(new_data.index, pd.DatetimeIndex) and not isinstance(self._data_buffer[symbol].index, pd.DatetimeIndex):
                    if not self._data_buffer[symbol].empty:
                        # Convert existing buffer to use DatetimeIndex
                        if 'datetime' in self._data_buffer[symbol].columns:
                            self._data_buffer[symbol] = self._data_buffer[symbol].set_index('datetime')
                
                # Concatenate data
                self._data_buffer[symbol] = pd.concat([self._data_buffer[symbol], new_data]).drop_duplicates()
                
            except Exception as e:
                self.logger.error(f"Error concatenating data: {e}")
                
                # Try a more robust approach with common columns
                buffer_cols = set(self._data_buffer[symbol].columns)
                new_cols = set(new_data.columns)
                
                common_cols = buffer_cols.intersection(new_cols)
                if common_cols:
                    self._data_buffer[symbol] = pd.concat([
                        self._data_buffer[symbol][list(common_cols)], 
                        new_data[list(common_cols)]
                    ]).drop_duplicates()
        
        # Sort data by datetime if present
        if 'datetime' in self._data_buffer[symbol].columns:
            self._data_buffer[symbol] = self._data_buffer[symbol].sort_values('datetime')
        
        # If we already have sufficient history, trim the buffer
        if self._has_sufficient_history[symbol]:
            # Keep a reasonable buffer size to avoid memory issues
            buffer_size = min(1000, max(100, self._required_data_points * 2))
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
        Calculate a factor with efficient caching.
        
        Args:
            data: Data for factor calculation
            factor_name: Name of factor to calculate
            symbol: Symbol for caching purposes
            
        Returns:
            pd.Series: Calculated factor values
        """
        if factor_name not in self._factor_registry:
            self.logger.warning(f"Factor '{factor_name}' not registered")
            return pd.Series(index=data.index)
        
        factor_info = self._factor_registry[factor_name]
        
        # Check dependencies first
        for dep_name in factor_info.get('depends_on', []):
            if dep_name in self._factor_registry and symbol and symbol in self._factor_cache:
                if dep_name not in self._factor_cache[symbol]:
                    # Calculate dependency if not in cache
                    self.calculate_factor(data, dep_name, symbol)
        
        # Calculate factor if function available
        if factor_info['func'] and callable(factor_info['func']):
            try:
                # Check if factor is already in cache
                if symbol and symbol in self._factor_cache and factor_name in self._factor_cache[symbol]:
                    cached_factor = self._factor_cache[symbol][factor_name]
                    # Check if cache is valid for the current data
                    if isinstance(cached_factor, pd.Series) and len(cached_factor) == len(data):
                        return cached_factor
                
                # Prepare dependency values for calculation
                kwargs = {}
                for dep_name in factor_info.get('depends_on', []):
                    if symbol and symbol in self._factor_cache and dep_name in self._factor_cache[symbol]:
                        kwargs[dep_name] = self._factor_cache[symbol][dep_name]
                
                # Calculate the factor
                factor_values = factor_info['func'](data, **kwargs)
                
                # Cache the result if symbol provided
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
        
        This method must be implemented by concrete strategy classes.
        
        Args:
            data: Market data with sufficient history
            
        Returns:
            pd.DataFrame: Signals with columns like timestamp, symbol, action
        """
        pass
    
    async def initialize(self) -> None:
        """Initialize strategy resources."""
        if not self._is_initialized:
            # Start async executor
            await self.executor.start()
            
            # Mark as initialized
            self._is_initialized = True
            self.logger.info(f"Strategy {self.__class__.__name__} initialized")
    
    async def shutdown(self) -> None:
        """Clean up strategy resources."""
        if self._is_initialized:
            # Clear data caches
            self._data_buffer.clear()
            self._factor_cache.clear()
            self._has_sufficient_history.clear()
            
            # Shutdown executor
            await self.executor.shutdown()
            
            # Mark as not initialized
            self._is_initialized = False
            self.logger.info(f"Strategy {self.__class__.__name__} shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get strategy performance statistics.
        
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
    
    async def shutdown(self) -> None:
        """
        Clean up resources
        """
        # Call subclass-specific shutdown
        await self._shutdown_specific()
        
        # Reset state
        self._initialized = False
        
        self.logger.info(f"{self.__class__.__name__} shutdown completed")
    
    async def _shutdown_specific(self) -> None:
        """
        Specific shutdown operations for subclasses
        """
        pass