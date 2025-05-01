#!/usr/bin/env python3
# src/strategy/factors/base.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union, Tuple, List, Callable
from abc import ABC, abstractmethod
import hashlib
from enum import Enum

from src.common.log_manager import LogManager
logger = LogManager.get_logger("strategy.factors")


class SignalType(Enum):
    """Signal type enumeration"""
    STANDARD = "standard"       # Use indicator value directly
    CROSSOVER = "crossover"     # Use crossover signals
    THRESHOLD = "threshold"     # Use threshold-based signals
    MOMENTUM = "momentum"       # Use momentum-based signals
    REVERSAL = "reversal"       # Use reversal signals
    VOLATILITY = "volatility"   # Use volatility-based signals


class FactorBase(ABC):
    """Base class for all technical indicators and factors"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize indicator
        
        Args:
            name: Indicator name
        """
        self._cache = {}
        self._name = name or self.__class__.__name__
        self.logger = LogManager.get_logger(f"strategy.factors.{self._name.lower()}")
    
    def clear_cache(self) -> None:
        """Clear calculation cache"""
        self._cache = {}
    
    def _get_cache_key(self, data: pd.DataFrame, variant_key: str = "") -> str:
        """
        Generate cache key for dataframe
        
        Args:
            data: Data frame
            variant_key: Optional variant identifier
            
        Returns:
            str: Cache key
        """
        if data.empty:
            return ""
            
        # Create unique cache key
        try:
            # Use first 3 rows and last 3 rows as fingerprint
            front = data.head(3)
            back = data.tail(3)
            
            # Create fingerprint
            hash_input = ""
            
            if isinstance(data.index, pd.DatetimeIndex):
                start_date = data.index[0].strftime('%Y%m%d%H%M%S')
                end_date = data.index[-1].strftime('%Y%m%d%H%M%S')
                hash_input += f"{start_date}_{end_date}_{len(data)}"
            else:
                hash_input += f"{len(data)}"
                
            # Add head/tail data hash
            for df in [front, back]:
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        values = df[col].fillna(0).values
                        hash_input += "_" + np.array2string(values, precision=4)
            
            # Add variant key
            if variant_key:
                hash_input += f"_{variant_key}"
                
            # Generate hash
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Error creating cache key: {e}")
            return f"{len(data)}_{variant_key}"
    
    def calculate(self, data: pd.DataFrame, variant_key: str = "") -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate indicator (with caching)
        
        Args:
            data: Input price data
            variant_key: Optional variant identifier
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
        """
        cache_key = self._get_cache_key(data, variant_key)
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            result = self._calculate(data)
            
            if cache_key:
                self._cache[cache_key] = result
                
            return result
        except Exception as e:
            self.logger.error(f"{self._name} calculation error: {e}")
            if isinstance(data, pd.DataFrame):
                return pd.Series(index=data.index)
            return pd.Series()
    
    @abstractmethod
    def _calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Internal calculation method (to be implemented by subclasses)
        
        Args:
            data: Input price data
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
        """
        pass
    
    def generate_signal(self, data: pd.DataFrame, signal_type: SignalType = SignalType.STANDARD, 
                       params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Generate trading signals based on indicator
        
        Args:
            data: Input price data
            signal_type: Signal type
            params: Signal parameters
            
        Returns:
            pd.Series: Trading signals (positive for buy, negative for sell, zero for no action)
        """
        params = params or {}
        result = self.calculate(data)
        
        # Handle DataFrame case
        if isinstance(result, pd.DataFrame):
            # Use specified column
            column = params.get("column", result.columns[0])
            if column in result.columns:
                values = result[column]
            else:
                values = result.iloc[:, 0]  # Use first column
        else:
            values = result
            
        if values.empty:
            return pd.Series(index=data.index)
            
        signal = pd.Series(0, index=values.index)
        
        # Generate signals based on signal type
        if signal_type == SignalType.CROSSOVER or signal_type.value == "crossover":
            # Crossover signal (with zero or specified level)
            zero_level = params.get("level", 0)
            signal = ((values > zero_level) & (values.shift(1) <= zero_level)).astype(float)
            signal -= ((values < zero_level) & (values.shift(1) >= zero_level)).astype(float)
            
        elif signal_type == SignalType.THRESHOLD or signal_type.value == "threshold":
            # Threshold signal
            upper = params.get("upper_threshold", 0.7)
            lower = params.get("lower_threshold", 0.3)
            signal = ((values < lower)).astype(float)  # Buy when below lower
            signal -= ((values > upper)).astype(float)  # Sell when above upper
            
        elif signal_type == SignalType.MOMENTUM or signal_type.value == "momentum":
            # Momentum signal
            signal = values.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            
        elif signal_type == SignalType.REVERSAL or signal_type.value == "reversal":
            # Reversal signal
            lookback = params.get("lookback", 3)
            signal = values.rolling(window=lookback).apply(
                lambda x: 1 if x[0] > x[-1] else (-1 if x[0] < x[-1] else 0), raw=True)
                
        elif signal_type == SignalType.VOLATILITY or signal_type.value == "volatility":
            # Volatility signal
            threshold = params.get("vol_threshold", 0.02)
            volatility = values.rolling(window=params.get("vol_window", 10)).std()
            signal = ((volatility < threshold) & (volatility.shift(1) >= threshold)).astype(float)  # Volatility decreasing
            signal -= ((volatility > threshold) & (volatility.shift(1) <= threshold)).astype(float)  # Volatility increasing
            
        else:  # SignalType.STANDARD
            # Standard signal (use indicator value directly)
            signal = values
            
        return signal
    
    @property
    def name(self) -> str:
        """Get indicator name"""
        return self._name