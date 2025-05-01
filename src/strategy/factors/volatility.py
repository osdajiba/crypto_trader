#!/usr/bin/env python3
# src/strategy/factors/volatility.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

from src.strategy.factors.base import FactorBase


class BollingerBands(FactorBase):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize Bollinger Bands indicator
        
        Args:
            period: SMA period
            std_dev: Standard deviation multiplier
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"BB_{period}_{std_dev}")
        self.period = period
        self.std_dev = std_dev
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'upper', 'middle', 'lower' columns
        """
        if self.price_col not in data.columns or len(data) < self.period:
            return pd.DataFrame(index=data.index, columns=['upper', 'middle', 'lower', 'width', 'percent_b'])
        
        # Calculate middle band (SMA)
        middle = data[self.price_col].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        std = data[self.price_col].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        upper = middle + (std * self.std_dev)
        lower = middle - (std * self.std_dev)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        }, index=data.index)
        
        # Calculate percentage bandwidth
        result['width'] = (upper - lower) / middle
        
        # Calculate price relative to bands (%B)
        result['percent_b'] = (data[self.price_col] - lower) / (upper - lower).replace(0, np.finfo(float).eps)
        
        return result


class ATR(FactorBase):
    """Average True Range (ATR) indicator"""
    
    def __init__(self, period: int = 14, name: Optional[str] = None):
        """
        Initialize ATR indicator
        
        Args:
            period: ATR period
            name: Indicator name
        """
        super().__init__(name or f"ATR_{period}")
        self.period = period
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate ATR
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: ATR values
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols) or len(data) < self.period + 1:
            return pd.Series(index=data.index)
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR using simple moving average
        atr = tr.rolling(window=self.period).mean()
        
        return atr


class ADX(FactorBase):
    """Average Directional Index (ADX) indicator"""
    
    def __init__(self, period: int = 14, name: Optional[str] = None):
        """
        Initialize ADX indicator
        
        Args:
            period: ADX period
            name: Indicator name
        """
        super().__init__(name or f"ADX_{period}")
        self.period = period
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'adx', 'di_plus', 'di_minus' columns
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols) or len(data) < self.period * 2:
            return pd.DataFrame(index=data.index, columns=['adx', 'di_plus', 'di_minus'])
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = data['high'] - data['high'].shift(1)
        down_move = data['low'].shift(1) - data['low']
        
        # Calculate +DM and -DM
        plus_dm = pd.Series(0, index=data.index)
        plus_dm.loc[(up_move > down_move) & (up_move > 0)] = up_move
        
        minus_dm = pd.Series(0, index=data.index)
        minus_dm.loc[(down_move > up_move) & (down_move > 0)] = down_move
        
        # Calculate smoothed TR, +DM, and -DM
        tr_smoothed = tr.rolling(window=self.period).sum()
        plus_dm_smoothed = plus_dm.rolling(window=self.period).sum()
        minus_dm_smoothed = minus_dm.rolling(window=self.period).sum()
        
        # Calculate +DI and -DI
        di_plus = 100 * plus_dm_smoothed / tr_smoothed.replace(0, np.finfo(float).eps)
        di_minus = 100 * minus_dm_smoothed / tr_smoothed.replace(0, np.finfo(float).eps)
        
        # Calculate DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.finfo(float).eps)
        
        # Calculate ADX
        adx = dx.rolling(window=self.period).mean()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }, index=data.index)
        
        return result