#!/usr/bin/env python3
# src/strategy/factors_lib.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Union, Tuple, List, Callable, Type
from abc import ABC, abstractmethod
import hashlib
from enum import Enum
import logging

# Get logger
logger = logging.getLogger("strategy.factors")


class SignalType(Enum):
    """Signal type enumeration"""
    STANDARD = "standard"       # Use indicator value directly
    CROSSOVER = "crossover"     # Use crossover signals
    THRESHOLD = "threshold"     # Use threshold-based signals
    MOMENTUM = "momentum"       # Use momentum-based signals
    REVERSAL = "reversal"       # Use reversal signals
    VOLATILITY = "volatility"   # Use volatility-based signals


class BaseIndicator(ABC):
    """Base class for all technical indicators"""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize indicator
        
        Args:
            name: Indicator name
        """
        self._cache = {}
        self._name = name or self.__class__.__name__
    
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
            logger.warning(f"Error creating cache key: {e}")
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
            logger.error(f"{self._name} calculation error: {e}")
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


class RsiIndicator(BaseIndicator):
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


class MacdIndicator(BaseIndicator):
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


class BollingerBand(BaseIndicator):
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
        
        # Calculate percentage bandwidth
        result['width'] = (upper - lower) / middle
        
        # Calculate price relative to bands (%B)
        result['percent_b'] = (data[self.price_col] - lower) / (upper - lower)
        
        return result


class VolumeOscillator(BaseIndicator):
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


class StochasticOscillator(BaseIndicator):
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


class AverageTrueRange(BaseIndicator):
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


class AverageDirectionalIndex(BaseIndicator):
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


class OnBalanceVolume(BaseIndicator):
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


class IchimokuCloud(BaseIndicator):
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


class ExponentialMovingAverage(BaseIndicator):
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
        
        return ema


class SimpleMovingAverage(BaseIndicator):
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


class MoneyFlowIndex(BaseIndicator):
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