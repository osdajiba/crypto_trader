#!/usr/bin/env python3
# src/strategy/factors_lib.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union, Tuple


class IndicatorBase:
    """Base class for all technical indicators"""
    
    def __init__(self):
        """Initialize indicator"""
        self._cache = {}
    
    def clear_cache(self):
        """Clear calculation cache"""
        self._cache = {}
    
    def _get_cache_key(self, data: pd.DataFrame) -> str:
        """Generate a cache key for the dataframe"""
        if data.empty:
            return ""
        # Use the first and last timestamps and data length for the key
        first = data.index[0] if isinstance(data.index, pd.DatetimeIndex) else data.index[0]
        last = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else data.index[-1]
        return f"{first}_{last}_{len(data)}"
    
    def calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate indicator (with caching)
        
        Args:
            data: Input price data
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
        """
        cache_key = self._get_cache_key(data)
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        
        result = self._calculate(data)
        
        if cache_key:
            self._cache[cache_key] = result
            
        return result
    
    def _calculate(self, data: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """
        Internal calculation method (to be implemented by subclasses)
        
        Args:
            data: Input price data
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Indicator values
        """
        raise NotImplementedError("Subclasses must implement _calculate method")


class RsiIndicator(IndicatorBase):
    """Relative Strength Index (RSI) indicator"""
    
    def __init__(self, period: int = 14, price_col: str = 'close'):
        """
        Initialize RSI indicator
        
        Args:
            period: RSI period
            price_col: Price column to use
        """
        super().__init__()
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
        
        # First average values
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class MacdIndicator(IndicatorBase):
    """Moving Average Convergence Divergence (MACD) indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                price_col: str = 'close'):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            price_col: Price column to use
        """
        super().__init__()
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
            pd.DataFrame: DataFrame with 'macd', 'signal', and 'histogram' columns
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


class BollingerBand(IndicatorBase):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, price_col: str = 'close'):
        """
        Initialize Bollinger Bands indicator
        
        Args:
            period: SMA period
            std_dev: Standard deviation multiplier
            price_col: Price column to use
        """
        super().__init__()
        self.period = period
        self.std_dev = std_dev
        self.price_col = price_col
    
    def _calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            data: Input price data
            
        Returns:
            pd.DataFrame: DataFrame with 'upper', 'middle', and 'lower' columns
        """
        if self.price_col not in data.columns or len(data) < self.period:
            return pd.DataFrame(index=data.index, columns=['upper', 'middle', 'lower'])
        
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
        
        return result


class VolumeOscillator(IndicatorBase):
    """Volume Oscillator indicator"""
    
    def __init__(self, fast_period: int = 5, slow_period: int = 14, vol_col: str = 'volume'):
        """
        Initialize Volume Oscillator indicator
        
        Args:
            fast_period: Fast volume MA period
            slow_period: Slow volume MA period
            vol_col: Volume column to use
        """
        super().__init__()
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


class StochasticOscillator(IndicatorBase):
    """Stochastic Oscillator indicator"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, slowing: int = 3):
        """
        Initialize Stochastic Oscillator indicator
        
        Args:
            k_period: K period
            d_period: D period
            slowing: Slowing period
        """
        super().__init__()
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


class AverageTrueRange(IndicatorBase):
    """Average True Range (ATR) indicator"""
    
    def __init__(self, period: int = 14):
        """
        Initialize ATR indicator
        
        Args:
            period: ATR period
        """
        super().__init__()
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
        
        # Calculate ATR
        atr = tr.rolling(window=self.period).mean()
        
        return atr


class OnBalanceVolume(IndicatorBase):
    """On-Balance Volume (OBV) indicator"""
    
    def __init__(self, price_col: str = 'close', vol_col: str = 'volume'):
        """
        Initialize OBV indicator
        
        Args:
            price_col: Price column to use
            vol_col: Volume column to use
        """
        super().__init__()
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


class IchimokuCloud(IndicatorBase):
    """Ichimoku Cloud indicator"""
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                senkou_b_period: int = 52, displacement: int = 26):
        """
        Initialize Ichimoku Cloud indicator
        
        Args:
            tenkan_period: Tenkan-sen (Conversion Line) period
            kijun_period: Kijun-sen (Base Line) period
            senkou_b_period: Senkou Span B period
            displacement: Displacement period
        """
        super().__init__()
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


class MovingAverageConvergenceDivergence(MacdIndicator):
    """Alias for MACD indicator for backward compatibility"""
    pass


class ExponentialMovingAverage(IndicatorBase):
    """Exponential Moving Average (EMA) indicator"""
    
    def __init__(self, period: int = 20, price_col: str = 'close'):
        """
        Initialize EMA indicator
        
        Args:
            period: EMA period
            price_col: Price column to use
        """
        super().__init__()
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
        
        return ema


class SimpleMovingAverage(IndicatorBase):
    """Simple Moving Average (SMA) indicator"""
    
    def __init__(self, period: int = 20, price_col: str = 'close'):
        """
        Initialize SMA indicator
        
        Args:
            period: SMA period
            price_col: Price column to use
        """
        super().__init__()
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


class RelativeStrengthIndex(RsiIndicator):
    """Alias for RSI indicator for backward compatibility"""
    pass


class CommodityChannelIndex(IndicatorBase):
    """Commodity Channel Index (CCI) indicator"""
    
    def __init__(self, period: int = 20, constant: float = 0.015):
        """
        Initialize CCI indicator
        
        Args:
            period: CCI period
            constant: CCI constant (typically 0.015)
        """
        super().__init__()
        self.period = period
        self.constant = constant
    
    def _calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate CCI
        
        Args:
            data: Input price data
            
        Returns:
            pd.Series: CCI values
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols) or len(data) < self.period:
            return pd.Series(index=data.index)
        
        # Calculate typical price
        tp = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate simple moving average of typical price
        tp_sma = tp.rolling(window=self.period).mean()
        
        # Calculate mean deviation
        mean_deviation = tp.rolling(window=self.period).apply(
            lambda x: pd.Series(x).mad()  # Mean absolute deviation
        )
        
        # Handle zero mean deviation
        mean_deviation = mean_deviation.replace(0, np.finfo(float).eps)
        
        # Calculate CCI
        cci = (tp - tp_sma) / (self.constant * mean_deviation)
        
        return cci


class AverageDirectionalIndex(IndicatorBase):
    """Average Directional Index (ADX) indicator"""
    
    def __init__(self, period: int = 14):
        """
        Initialize ADX indicator
        
        Args:
            period: ADX period
        """
        super().__init__()
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