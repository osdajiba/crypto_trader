class IchimokuCloud(FactorBase):
    """Ichimoku Cloud indicator"""
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                senkou_b_period: int = 52, displacement: int = 26, name: Optional[str] = None):
        """
        Initialize Ichimoku Cloud indicator
        
        Args:
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
            displacement: Displacement period
            name: Indicator name
        """
        super().__init__(name or "Ichimoku")
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with Ichimoku components
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols) or len(data) < self.senkou_b_period:
            return pd.DataFrame(index=data.index, columns=[
                'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'
            ])
        
        # Calculate Tenkan-sen (Conversion Line)
        tenkan_high = data['high'].rolling(window=self.tenkan_period).max()
        tenkan_low = data['low'].rolling(window=self.tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Calculate Kijun-sen (Base Line)
        kijun_high = data['high'].rolling(window=self.kijun_period).max()
        kijun_low = data['low'].rolling(window=self.kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Calculate Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self.displacement)
        
        # Calculate Senkou Span B (Leading Span B)
        senkou_high = data['high'].rolling(window=self.senkou_b_period).max()
        senkou_low = data['low'].rolling(window=self.senkou_b_period).min()
        senkou_span_b = ((senkou_high + senkou_low) / 2).shift(self.displacement)
        
        # Calculate Chikou Span (Lagging Span)
        chikou_span = data['close'].shift(-self.displacement)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }, index=data.index)
        
        return result


class EMA(FactorBase):
    """Exponential Moving Average (EMA) indicator"""
    
    def __init__(self, period: int = 20, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize EMA indicator
        
        Args:
            period: EMA period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"EMA_{period}")
        self.period = period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate EMA
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: EMA values
        """
        if self.price_col not in data.columns or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate EMA
        ema = data[self.price_col].ewm(span=self.period, adjust=False).mean()
        
        return ema#!/usr/bin/env python3
# src/strategy/factors/trend.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

from src.strategy.factors.base import FactorBase


class SMA(FactorBase):
    """Simple Moving Average (SMA) indicator"""
    
    def __init__(self, period: int = 20, price_col: str = 'close', name: Optional[str] = None):
        """
        Initialize SMA indicator
        
        Args:
            period: SMA period
            price_col: Price column name
            name: Indicator name
        """
        super().__init__(name or f"SMA_{period}")
        self.period = period
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate SMA
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: SMA values
        """
        if self.price_col not in data.columns or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate SMA
        sma = data[self.price_col].rolling(window=self.period).mean()
        
        return sma