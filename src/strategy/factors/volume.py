#!/usr/bin/env python3
# src/strategy/factors/volume.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union

from src.strategy.factors.base import BaseFactor


class OBV(BaseFactor):
    """On-Balance Volume (OBV) indicator"""
    
    def __init__(self, price_col: str = 'close', vol_col: str = 'volume', name: Optional[str] = None):
        """
        Initialize OBV indicator
        
        Args:
            price_col: Price column name
            vol_col: Volume column name
            name: Indicator name
        """
        super().__init__(name or "OBV")
        self.price_col = price_col
        self.vol_col = vol_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate OBV
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: OBV values
        """
        if self.price_col not in data.columns or self.vol_col not in data.columns or len(data) < 2:
            return pd.Series(index=data.index)
        
        # Calculate price change direction
        price_change = data[self.price_col].diff()
        
        # Create OBV signal
        obv_signal = pd.Series(0, index=data.index)
        obv_signal.loc[price_change > 0] = 1
        obv_signal.loc[price_change < 0] = -1
        
        # Calculate OBV
        obv = (obv_signal * data[self.vol_col]).cumsum()
        
        return obv


class MoneyFlowIndex(BaseFactor):
    """Money Flow Index (MFI) indicator"""
    
    def __init__(self, period: int = 14, vol_col: str = 'volume', name: Optional[str] = None):
        """
        Initialize MFI indicator
        
        Args:
            period: MFI period
            vol_col: Volume column name
            name: Indicator name
        """
        super().__init__(name or f"MFI_{period}")
        self.period = period
        self.vol_col = vol_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate MFI
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: MFI values
        """
        required_cols = ['high', 'low', 'close', self.vol_col]
        if not all(col in data.columns for col in required_cols) or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data[self.vol_col]
        
        # Calculate money flow direction
        price_change = typical_price.diff()
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        positive_flow.loc[price_change > 0] = raw_money_flow.loc[price_change > 0]
        negative_flow.loc[price_change < 0] = raw_money_flow.loc[price_change < 0]
        
        # Sum positive and negative flows over period
        positive_sum = positive_flow.rolling(window=self.period).sum()
        negative_sum = negative_flow.rolling(window=self.period).sum()
        
        # Calculate money flow ratio
        money_flow_ratio = positive_sum / negative_sum.replace(0, np.finfo(float).eps)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        
        return mfi


class VolumeOscillator(BaseFactor):
    """Volume Oscillator indicator"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 14, vol_col: str = 'volume', name: Optional[str] = None):
        """
        Initialize Volume Oscillator indicator
        
        Args:
            fast_period: Fast volume MA period
            slow_period: Slow volume MA period
            vol_col: Volume column name
            name: Indicator name
        """
        super().__init__(name or f"VolOsc_{fast_period}_{slow_period}")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.vol_col = vol_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Oscillator
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: Volume Oscillator values
        """
        if self.vol_col not in data.columns or len(data) < self.slow_period:
            return pd.Series(index=data.index)
        
        # Calculate fast and slow MAs
        fast_ma = data[self.vol_col].rolling(window=self.fast_period).mean()
        slow_ma = data[self.vol_col].rolling(window=self.slow_period).mean()
        
        # Calculate percentage difference
        vol_osc = 100 * ((fast_ma - slow_ma) / slow_ma.replace(0, np.finfo(float).eps))
        
        return vol_osc