#!/usr/bin/env python3
# src/strategy/factors/momentum.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

from src.strategy.factors.base import BaseFactor


class RSI(BaseFactor):
    """Relative Strength Index (RSI) indicator"""
    
    def __init__(self, period: int = 14, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize RSI indicator
        
        Args:
            period: RSI period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"RSI_{period}")
        self.period = period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate RSI
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: RSI values
        """
        if self.price_col not in data.columns or len(data) < self.period + 1:
            return pd.Series(index=data.index)
        
        # Calculate price changes
        delta = data[self.price_col].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate averages
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MACD(BaseFactor):
    """Moving Average Convergence Divergence (MACD) indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"MACD_{fast_period}_{slow_period}_{signal_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'macd', 'signal', 'histogram' columns
        """
        if self.price_col not in data.columns or len(data) < self.slow_period + self.signal_period:
            return pd.DataFrame(index=data.index, columns=['macd', 'signal', 'histogram'])
        
        # Calculate EMAs
        fast_ema = data[self.price_col].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.price_col].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Create result DataFrame
        result = pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)
        
        return result


class Stochastic(BaseFactor):
    """Stochastic Oscillator indicator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3, name: Optional[str] = None):
        """
        Initialize Stochastic Oscillator indicator
        
        Args:
            k_period: K period
            d_period: D period
            slowing: Slowing period
            name: Indicator name
        """
        super().__init__(name or f"Stoch_{k_period}_{d_period}_{slowing}")
        self.k_period = k_period
        self.d_period = d_period
        self.slowing = slowing
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'k' and 'd' columns
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols) or len(data) < self.k_period + self.d_period:
            return pd.DataFrame(index=data.index, columns=['k', 'd'])
        
        # Calculate %K
        low_min = data['low'].rolling(window=self.k_period).min()
        high_max = data['high'].rolling(window=self.k_period).max()
        
        # Handle division by zero
        denom = high_max - low_min
        denom = denom.replace(0, np.finfo(float).eps)
        
        k_fast = 100 * ((data['close'] - low_min) / denom)
        
        # Apply slowing if specified
        if self.slowing > 1:
            k = k_fast.rolling(window=self.slowing).mean()
        else:
            k = k_fast
        
        # Calculate %D
        d = k.rolling(window=self.d_period).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'k': k,
            'd': d
        }, index=data.index)
        
        return result