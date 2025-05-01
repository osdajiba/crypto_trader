#!/usr/bin/env python3
# src/strategy/factors/custom.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable, Type

from src.strategy.factors.base import FactorBase


class PriceChangeRate(FactorBase):
    """Price Change Rate indicator"""
    
    def __init__(self, period: int = 5, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize Price Change Rate indicator
        
        Args:
            period: Look-back period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"PCR_{period}")
        self.period = period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Price Change Rate
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: Price Change Rate values
        """
        if self.price_col not in data.columns or len(data) < self.period + 1:
            return pd.Series(index=data.index)
        
        # Calculate percentage change over period
        pcr = data[self.price_col].pct_change(self.period)
        
        return pcr


class VolatilityIndex(FactorBase):
    """Volatility Index indicator"""
    
    def __init__(self, period: int = 20, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize Volatility Index indicator
        
        Args:
            period: Volatility calculation period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"VOLX_{period}")
        self.period = period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volatility Index
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: Volatility Index values
        """
        if self.price_col not in data.columns or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate returns
        returns = data[self.price_col].pct_change()
        
        # Calculate volatility as rolling standard deviation of returns
        volatility = returns.rolling(window=self.period).std()
        
        # Annualize volatility (assuming daily data)
        annualized_volatility = volatility * np.sqrt(252)
        
        return annualized_volatility


class PriceChannel(FactorBase):
    """Price Channel indicator"""
    
    def __init__(self, period: int = 20, name: Optional[str] = None):
        """
        Initialize Price Channel indicator
        
        Args:
            period: Channel period
            name: Indicator name
        """
        super().__init__(name or f"PC_{period}")
        self.period = period
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Channel
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'upper', 'lower', 'middle' columns
        """
        required_cols = ['high', 'low']
        if not all(col in data.columns for col in required_cols) or len(data) < self.period:
            return pd.DataFrame(index=data.index, columns=['upper', 'lower', 'middle'])
        
        # Calculate upper and lower bands
        upper = data['high'].rolling(window=self.period).max()
        lower = data['low'].rolling(window=self.period).min()
        
        # Calculate middle line
        middle = (upper + lower) / 2
        
        # Create result DataFrame
        result = pd.DataFrame({
            'upper': upper,
            'lower': lower,
            'middle': middle
        }, index=data.index)
        
        # Calculate width
        result['width'] = (upper - lower) / middle
        
        # Calculate position within channel
        if 'close' in data.columns:
            result['position'] = (data['close'] - lower) / (upper - lower).replace(0, np.finfo(float).eps)
        
        return result


class RelativeStrengthFactor(FactorBase):
    """Relative Strength Factor compares current instrument to a benchmark or another instrument"""
    
    def __init__(self, period: int = 20, price_col: str = 'close', 
                benchmark_col: str = 'benchmark', name: Optional[str] = None):
        """
        Initialize Relative Strength Factor
        
        Args:
            period: Calculation period
            price_col: Price column name for main instrument
            benchmark_col: Price column name for benchmark
            name: Indicator name
        """
        super().__init__(name or f"RS_{period}")
        self.period = period
        self.price_col = price_col
        self.benchmark_col = benchmark_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Relative Strength
        
        Args:
            data: Input price data with both instrument and benchmark
            
        Returns:
            pd.Series: Relative Strength values
        """
        if self.price_col not in data.columns or self.benchmark_col not in data.columns or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate percentage change of both instrument and benchmark
        instrument_change = data[self.price_col].pct_change(self.period)
        benchmark_change = data[self.benchmark_col].pct_change(self.period)
        
        # Calculate relative strength (instrument performance relative to benchmark)
        rs = (1 + instrument_change) / (1 + benchmark_change).replace(0, np.finfo(float).eps)
        
        return rs


class CustomFactorBuilder:
    """Helper class to build custom factors using function composition"""
    
    @staticmethod
    def create_custom_factor(name: str, func: Callable[[pd.DataFrame], pd.Series], 
                          required_columns: List[str] = None, 
                          min_data_points: int = 0) -> Type[FactorBase]:
        """
        Create a custom factor class using provided function
        
        Args:
            name: Factor name
            func: Function to calculate factor values
            required_columns: Required data columns
            min_data_points: Minimum required data points
            
        Returns:
            Type[FactorBase]: Custom factor class
        """
        class CustomFactor(FactorBase):
            def __init__(self, custom_name=None):
                super().__init__(custom_name or name)
                self._func = func
                self._required_columns = required_columns or []
                self._min_data_points = min_data_points
            
            def _calculate(self, data: pd.DataFrame) -> pd.Series:
                # Check requirements
                if self._required_columns and not all(col in data.columns for col in self._required_columns):
                    return pd.Series(index=data.index)
                    
                if len(data) < self._min_data_points:
                    return pd.Series(index=data.index)
                
                # Call provided function
                try:
                    return self._func(data)
                except Exception as e:
                    self.logger.error(f"Error in custom factor '{self.name}': {e}")
                    return pd.Series(index=data.index)
        
        return CustomFactor