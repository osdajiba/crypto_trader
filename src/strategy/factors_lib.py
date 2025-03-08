# src/strategy/factor_lib.py

import numpy as np
import pandas as pd

def sma(data: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
        window (int): Moving average window size (default: 10).

    Returns:
        pd.Series: SMA values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If 'close' column is missing.
    """
    if 'close' not in data.columns:
        raise KeyError("DataFrame must contain 'close' column for SMA calculation")
    return data['close'].rolling(window=window, min_periods=1).mean()

def rsi(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
        window (int): RSI window size (default: 14).

    Returns:
        pd.Series: RSI values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If 'close' column is missing.
    """
    if 'close' not in data.columns:
        raise KeyError("DataFrame must contain 'close' column for RSI calculation")
    delta = data['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)  # Avoid division by zero
    return 100 - (100 / (1 + rs)).fillna(50)  # Default to 50 when undefined

def vwap(data: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    Args:
        data (pd.DataFrame): DataFrame with 'close', 'high', 'low', and 'volume' columns.
    
    Returns:
        pd.Series: VWAP values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If required columns ('close', 'high', 'low', 'volume') are missing.
    """
    required_cols = {'close', 'high', 'low', 'volume'}
    if not required_cols.issubset(data.columns):
        raise KeyError(f"DataFrame must contain {required_cols} for VWAP calculation")
    typical_price = (data['close'] + data['high'] + data['low']) / 3
    return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()

def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
        period (int): ATR calculation period (default: 14).

    Returns:
        pd.Series: ATR values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If required columns ('high', 'low', 'close') are missing.
    """
    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(data.columns):
        raise KeyError(f"DataFrame must contain {required_cols} for ATR calculation")
    tr0 = abs(data['high'] - data['low'])
    tr1 = abs(data['high'] - data['close'].shift())
    tr2 = abs(data['low'] - data['close'].shift())
    true_range = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()

def supertrend(data: pd.DataFrame, period: int = 14, multiplier: float = 3.0) -> pd.Series:
    """
    Calculate SuperTrend indicator.

    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
        period (int): ATR calculation period (default: 14).
        multiplier (float): Multiplier for band width (default: 3.0).

    Returns:
        pd.Series: SuperTrend values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If required columns ('high', 'low', 'close') are missing.
    """
    required_cols = {'high', 'low', 'close'}
    if not required_cols.issubset(data.columns):
        raise KeyError(f"DataFrame must contain {required_cols} for SuperTrend calculation")
    
    atr_val = atr(data, period)
    basic_upper = (data['high'] + data['low']) / 2 + multiplier * atr_val.shift()
    basic_lower = (data['high'] + data['low']) / 2 - multiplier * atr_val.shift()
    
    supertrend = pd.Series(index=data.index, dtype=float)
    supertrend.iloc[0] = basic_upper.iloc[0]  # Initial value
    
    for i in range(1, len(data)):
        if data['close'].iloc[i-1] <= supertrend.iloc[i-1]:
            supertrend.iloc[i] = basic_upper.iloc[i]
        elif data['close'].iloc[i-1] >= supertrend.iloc[i-1]:
            supertrend.iloc[i] = basic_lower.iloc[i]
        else:
            supertrend.iloc[i] = supertrend.iloc[i-1]
    
    return supertrend

def aroon(data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Calculate Aroon indicator (Aroon Up and Aroon Down).

    Args:
        data (pd.DataFrame): DataFrame with 'high' and 'low' columns.
        window (int): Aroon calculation period (default: 14).

    Returns:
        pd.DataFrame: DataFrame with 'aroon_up' and 'aroon_down' columns.

    Raises:
        KeyError: If required columns ('high', 'low') are missing.
    """
    required_cols = {'high', 'low'}
    if not required_cols.issubset(data.columns):
        raise KeyError(f"DataFrame must contain {required_cols} for Aroon calculation")
    
    result = pd.DataFrame(index=data.index)
    result['aroon_up'] = (window - data['high'].rolling(window=window).apply(np.argmax, raw=True) + 1) / window * 100
    result['aroon_down'] = (window - data['low'].rolling(window=window).apply(np.argmin, raw=True) + 1) / window * 100
    return result

def ma(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Moving Average (MA).

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
        window (int): Moving average window size (default: 20).

    Returns:
        pd.Series: MA values aligned with the input DataFrame’s index.

    Raises:
        KeyError: If 'close' column is missing.
    """
    if 'close' not in data.columns:
        raise KeyError("DataFrame must contain 'close' column for MA calculation")
    return data['close'].rolling(window=window, min_periods=1).mean()

def bollinger_bands(data: pd.DataFrame, window: int = 20, num_std_dev: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands (Upper, Middle, Lower).

    Args:
        data (pd.DataFrame): DataFrame with at least a 'close' column.
        window (int): Moving average window size (default: 20).
        num_std_dev (float): Number of standard deviations for band width (default: 2.0).

    Returns:
        pd.DataFrame: DataFrame with 'upper_band', 'middle_band', and 'lower_band' columns.

    Raises:
        KeyError: If 'close' column is missing.
    """
    if 'close' not in data.columns:
        raise KeyError("DataFrame must contain 'close' column for Bollinger Bands calculation")
    
    result = pd.DataFrame(index=data.index)
    middle_band = ma(data, window=window)
    std = data['close'].rolling(window=window, min_periods=1).std()
    
    result['middle_band'] = middle_band
    result['upper_band'] = middle_band + num_std_dev * std
    result['lower_band'] = middle_band - num_std_dev * std
    
    return result

