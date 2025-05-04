#!/usr/bin/env python3
"""
Factor Manager

A utility class to manage and calculate multiple technical factors efficiently.
Provides caching, batched calculation, and factor dependency resolution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Set
import logging
import asyncio
from datetime import datetime

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.factors.base import BaseFactor, SignalType
from src.strategy.factors.factory import get_factor_factory


class FactorManager:
    """
    Manager for technical factors with efficient calculation and caching
    
    This class provides a centralized way to manage technical factors used by
    strategies, with support for caching, dependency resolution, and batch
    calculation to improve performance.
    """
    
    def __init__(self, config: ConfigManager):
        """
        Initialize factor manager
        
        Args:
            config: System configuration manager
        """
        self.config = config
        self.logger = LogManager.get_logger("strategy.factor_manager")
        
        # Initialize factor factory
        self.factor_factory = get_factor_factory(config)
        
        # Factor instances
        self._factors = {}
        
        # Value cache
        self._factor_values = {}
        
        # Signal cache
        self._factor_signals = {}
        
        # Dependencies between factors
        self._dependencies = {}
        
        # Factor metadata
        self._metadata = {}
        
        self.logger.info("Factor manager initialized")
    
    async def add_factor(self, name: str, factor_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a factor to the manager
        
        Args:
            name: Factor name
            factor_params: Factor parameters (optional)
            
        Returns:
            bool: True if factor was added successfully
        """
        if name in self._factors:
            return True  # Already added
            
        try:
            # Create factor instance
            factor_instance = await self.factor_factory.create(name, factor_params)
            
            if factor_instance:
                self._factors[name] = factor_instance
                
                # Get factor metadata
                factor_info = self.factor_factory.get_factor_info(name)
                self._metadata[name] = factor_info.get('metadata', {})
                
                self.logger.debug(f"Added factor: {name}")
                return True
            else:
                self.logger.warning(f"Failed to create factor: {name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding factor {name}: {e}")
            return False
    
    async def add_factors(self, factor_names: List[str]) -> int:
        """
        Add multiple factors to the manager
        
        Args:
            factor_names: List of factor names
            
        Returns:
            int: Number of factors added successfully
        """
        added_count = 0
        
        for name in factor_names:
            if await self.add_factor(name):
                added_count += 1
                
        return added_count
    
    def get_factor(self, name: str) -> Optional[BaseFactor]:
        """
        Get a factor instance by name
        
        Args:
            name: Factor name
            
        Returns:
            Optional[BaseFactor]: Factor instance or None if not found
        """
        return self._factors.get(name)
    
    def get_factors_by_category(self, category: str) -> List[str]:
        """
        Get list of factor names in a category
        
        Args:
            category: Factor category
            
        Returns:
            List[str]: Factor names in the category
        """
        return [name for name, metadata in self._metadata.items()
                if metadata.get('category', '').lower() == category.lower()]
    
    def calculate_factor(self, data: pd.DataFrame, factor_name: str) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate a single factor
        
        Args:
            data: Input data
            factor_name: Factor name
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Calculated factor values
        """
        # Get factor instance
        factor = self._factors.get(factor_name)
        if not factor:
            self.logger.warning(f"Factor not found: {factor_name}")
            return pd.Series(index=data.index)
        
        # Check data requirements
        if data.empty:
            return pd.Series(index=data.index)
            
        try:
            # Calculate factor
            result = factor.calculate(data)
            
            # Cache the result
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'default'
            if symbol not in self._factor_values:
                self._factor_values[symbol] = {}
            self._factor_values[symbol][factor_name] = {
                'timestamp': datetime.now(),
                'values': result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating factor {factor_name}: {e}")
            return pd.Series(index=data.index)
    
    def calculate_factors(self, data: pd.DataFrame, factor_names: List[str]) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
        """
        Calculate multiple factors
        
        Args:
            data: Input data
            factor_names: List of factor names
            
        Returns:
            Dict[str, Union[pd.Series, pd.DataFrame]]: Calculated factors
        """
        results = {}
        
        # Calculate each factor
        for name in factor_names:
            results[name] = self.calculate_factor(data, name)
            
        return results
    
    def generate_signal(self, data: pd.DataFrame, factor_name: str, 
                      signal_type: Union[SignalType, str] = SignalType.STANDARD,
                      signal_params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Generate a trading signal from a factor
        
        Args:
            data: Input data
            factor_name: Factor name
            signal_type: Signal type
            signal_params: Signal parameters
            
        Returns:
            pd.Series: Trading signal
        """
        # Get factor instance
        factor = self._factors.get(factor_name)
        if not factor:
            self.logger.warning(f"Factor not found: {factor_name}")
            return pd.Series(index=data.index)
        
        # Check data requirements
        if data.empty:
            return pd.Series(index=data.index)
            
        try:
            # Convert signal type string to enum if needed
            if isinstance(signal_type, str):
                try:
                    signal_type = SignalType(signal_type)
                except ValueError:
                    signal_type = SignalType.STANDARD
            
            # Generate signal
            signal = factor.generate_signal(data, signal_type, signal_params)
            
            # Cache the signal
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'default'
            if symbol not in self._factor_signals:
                self._factor_signals[symbol] = {}
            self._factor_signals[symbol][factor_name] = {
                'timestamp': datetime.now(),
                'signal': signal,
                'type': signal_type.value if isinstance(signal_type, SignalType) else signal_type,
                'params': signal_params
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {factor_name}: {e}")
            return pd.Series(index=data.index)
    
    def generate_signals(self, data: pd.DataFrame, factor_config: Dict[str, Dict[str, Any]]) -> Dict[str, pd.Series]:
        """
        Generate signals for multiple factors
        
        Args:
            data: Input data
            factor_config: Factor configuration with signal type and parameters
                Format: {factor_name: {'signal_type': SignalType, 'params': {}}}
            
        Returns:
            Dict[str, pd.Series]: Generated signals
        """
        results = {}
        
        # Generate signal for each factor
        for name, config in factor_config.items():
            signal_type = config.get('signal_type', SignalType.STANDARD)
            params = config.get('params', {})
            
            results[name] = self.generate_signal(data, name, signal_type, params)
            
        return results
    
    def get_cached_values(self, symbol: str, factor_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cached factor values
        
        Args:
            symbol: Symbol
            factor_name: Factor name (optional, returns all factors if None)
            
        Returns:
            Dict[str, Any]: Cached values
        """
        if symbol not in self._factor_values:
            return {}
            
        if factor_name:
            return self._factor_values[symbol].get(factor_name, {})
            
        return self._factor_values[symbol]
    
    def get_cached_signals(self, symbol: str, factor_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cached factor signals
        
        Args:
            symbol: Symbol
            factor_name: Factor name (optional, returns all factors if None)
            
        Returns:
            Dict[str, Any]: Cached signals
        """
        if symbol not in self._factor_signals:
            return {}
            
        if factor_name:
            return self._factor_signals[symbol].get(factor_name, {})
            
        return self._factor_signals[symbol]
    
    def clear_cache(self, symbol: Optional[str] = None, factor_name: Optional[str] = None) -> None:
        """
        Clear factor cache
        
        Args:
            symbol: Symbol (optional, clears all symbols if None)
            factor_name: Factor name (optional, clears all factors if None)
        """
        if symbol is None:
            # Clear all caches
            self._factor_values = {}
            self._factor_signals = {}
            
            # Clear factor instance caches
            for factor in self._factors.values():
                factor.clear_cache()
                
            return
        
        # Clear specific symbol cache
        if factor_name is None:
            if symbol in self._factor_values:
                del self._factor_values[symbol]
            if symbol in self._factor_signals:
                del self._factor_signals[symbol]
        else:
            # Clear specific factor cache
            if symbol in self._factor_values and factor_name in self._factor_values[symbol]:
                del self._factor_values[symbol][factor_name]
            if symbol in self._factor_signals and factor_name in self._factor_signals[symbol]:
                del self._factor_signals[symbol][factor_name]
                
            # Clear factor instance cache
            factor = self._factors.get(factor_name)
            if factor:
                factor.clear_cache()
    
    def get_available_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about available factors
        
        Returns:
            Dict[str, Dict[str, Any]]: Factor information
        """
        return {name: self._metadata.get(name, {}) for name in self._factors.keys()}
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear caches
        self._factor_values.clear()
        self._factor_signals.clear()
        
        # Clear factor instances
        self._factors.clear()
        
        self.logger.info("Factor manager shutdown complete")