#!/usr/bin/env python3
# src/datacource/processor.py

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta


class DataProcessor:
    """Centralized data preprocessing utilities"""
    
    @staticmethod
    def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and clean OHLCV data
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned data
        """
        if data.empty:
            return data
        
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Standardize column names
        rename_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if ('time' in lower_col or 'date' in lower_col) and 'timestamp' not in lower_col:
                rename_map[col] = 'datetime'
            elif lower_col in ['o', 'open']:
                rename_map[col] = 'open'
            elif lower_col in ['h', 'high']:
                rename_map[col] = 'high'
            elif lower_col in ['l', 'low']:
                rename_map[col] = 'low'
            elif lower_col in ['c', 'close']:
                rename_map[col] = 'close'
            elif lower_col in ['v', 'volume']:
                rename_map[col] = 'volume'
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Ensure datetime column is present
        required_datetime = ['datetime', 'timestamp']
        has_datetime = any(col in df.columns for col in required_datetime)
        if not has_datetime:
            raise ValueError(f"DataFrame must have a datetime column (one of {required_datetime})")
        
        # Ensure required columns exist
        required_price_cols = ['open', 'high', 'low', 'close']
        has_required_cols = all(col in df.columns for col in required_price_cols)
        if not has_required_cols:
            for col in required_price_cols:
                if col not in df.columns:
                    df[col] = np.nan
        
        # Add volume if missing
        if 'volume' not in df.columns:
            df['volume'] = np.nan
        
        # Standardize datetime
        if 'datetime' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        
        # Add timestamp column if not present
        if 'timestamp' not in df.columns and 'datetime' in df.columns:
            df['timestamp'] = df['datetime'].map(lambda x: int(x.timestamp() * 1000) if pd.notnull(x) else None)
        
        # Sort and remove duplicates
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # Remove invalid data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to specified timeframe
        
        Args:
            data: Raw OHLCV data
            timeframe: Target timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            pd.DataFrame: Resampled data
        """
        if data.empty:
            return data
        
        if 'datetime' not in data.columns:
            raise ValueError("Data must contain 'datetime' column")
        
        # Ensure datetime is the index
        df = data.copy()
        if df.index.name != 'datetime':
            df = df.set_index('datetime')
        
        # Map timeframe to pandas frequency strings
        time_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        pandas_timeframe = time_map.get(timeframe, timeframe)
        
        # Perform resampling
        resampled = df.resample(pandas_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Reset index
        resampled = resampled.reset_index()
        
        # Add timestamp column
        if 'timestamp' not in resampled.columns:
            resampled['timestamp'] = resampled['datetime'].map(lambda x: int(x.timestamp() * 1000))
        
        return resampled
    
    @staticmethod
    def fill_missing_periods(data: pd.DataFrame, timeframe: str, 
                           start: Optional[datetime] = None, 
                           end: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fill missing time periods in data
        
        Args:
            data: Input data
            timeframe: Time frequency
            start: Start time (optional)
            end: End time (optional)
            
        Returns:
            pd.DataFrame: Filled data
        """
        if data.empty:
            return data
        
        if 'datetime' not in data.columns:
            raise ValueError("Data must contain 'datetime' column")
        
        # Ensure datetime is properly typed
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Set index
        df = df.set_index('datetime')
        
        # Determine start and end times
        start_time = start if start is not None else df.index.min()
        end_time = end if end is not None else df.index.max()
        
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.to_datetime(start_time)
        if not isinstance(end_time, pd.Timestamp):
            end_time = pd.to_datetime(end_time)
        
        # Create complete time sequence
        time_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        pandas_timeframe = time_map.get(timeframe, timeframe)
        full_index = pd.date_range(start=start_time, end=end_time, freq=pandas_timeframe)
        
        # Reindex data
        filled_df = df.reindex(full_index)
        
        # Forward fill OHLC data
        for col in ['open', 'high', 'low', 'close']:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].ffill()
        
        # Set missing volume to 0
        if 'volume' in filled_df.columns:
            filled_df['volume'] = filled_df['volume'].fillna(0)
        
        # Reset index
        filled_df = filled_df.reset_index().rename(columns={'index': 'datetime'})
        
        # Add timestamp column
        if 'timestamp' not in filled_df.columns:
            filled_df['timestamp'] = filled_df['datetime'].map(lambda x: int(x.timestamp() * 1000))
        
        return filled_df
    
    @staticmethod
    def merge_dataframes(main_df: pd.DataFrame, other_df: pd.DataFrame, 
                       on: str = 'datetime', suffixes: Tuple[str, str] = ('', '_right')) -> pd.DataFrame:
        """
        Merge two dataframes with duplicate column handling
        
        Args:
            main_df: Primary dataframe
            other_df: Secondary dataframe to merge
            on: Merge key column
            suffixes: Suffix tuple
            
        Returns:
            pd.DataFrame: Merged dataframe
        """
        if main_df.empty:
            return other_df.copy()
        if other_df.empty:
            return main_df.copy()
        
        # Process datetime columns
        for df in [main_df, other_df]:
            if on in df.columns and not pd.api.types.is_datetime64_any_dtype(df[on]):
                df[on] = pd.to_datetime(df[on])
        
        # Perform merge
        merged = pd.merge(main_df, other_df, on=on, how='outer', suffixes=suffixes)
        
        return merged
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, columns: Optional[List[str]] = None, 
                      method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and flag outliers
        
        Args:
            data: Input data
            columns: Columns to check (if None, all numeric columns)
            method: Detection method ('iqr' or 'zscore')
            threshold: Outlier threshold
            
        Returns:
            pd.DataFrame: Data with outlier flags
        """
        if data.empty:
            return data
        
        # Use all numeric columns if none specified
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
        else:
            # Keep only existing columns
            columns = [col for col in columns if col in data.columns]
        
        df = data.copy()
        
        # Create flags for each column
        for col in columns:
            outlier_col = f"{col}_is_outlier"
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[outlier_col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                if std == 0:  # Avoid division by zero
                    df[outlier_col] = False
                else:
                    z_scores = (df[col] - mean) / std
                    df[outlier_col] = abs(z_scores) > threshold
            
            else:
                raise ValueError(f"Unsupported method: {method}")
        
        # Add global outlier flag
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        if outlier_cols:
            df['is_outlier'] = df[outlier_cols].any(axis=1)
        
        return df
    
    @staticmethod
    def create_train_test_split(data: pd.DataFrame, train_ratio: float = 0.7, 
                              validation_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into training, validation, and test sets
        
        Args:
            data: Input data
            train_ratio: Ratio of training data
            validation_ratio: Ratio of validation data
            
        Returns:
            Tuple of (train_df, validation_df, test_df)
        """
        if data.empty:
            return data, pd.DataFrame(), pd.DataFrame()
        
        # Ensure data is sorted by datetime
        if 'datetime' in data.columns:
            data = data.sort_values('datetime').reset_index(drop=True)
        
        # Calculate split points
        n = len(data)
        train_end = int(n * train_ratio)
        validation_end = train_end + int(n * validation_ratio)
        
        # Split data
        train_df = data.iloc[:train_end].copy()
        validation_df = data.iloc[train_end:validation_end].copy()
        test_df = data.iloc[validation_end:].copy()
        
        return train_df, validation_df, test_df
    
    @staticmethod
    def extract_features(data: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """
        Extract basic technical features from OHLCV data
        
        Args:
            data: OHLCV data
            window_size: Window size for feature calculations
            
        Returns:
            pd.DataFrame: Data with extracted features
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # Ensure datetime is index for calculations
        if 'datetime' in df.columns and df.index.name != 'datetime':
            df = df.set_index('datetime')
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate moving averages
        df['sma_10'] = df['close'].rolling(window=window_size).mean()
        df['sma_20'] = df['close'].rolling(window=window_size*2).mean()
        
        # Calculate volatility
        df['volatility'] = df['returns'].rolling(window=window_size).std()
        
        # Calculate relative strength
        df['rsi'] = DataProcessor._calculate_rsi(df['close'], window=window_size)
        
        # Calculate MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = DataProcessor._calculate_macd(df['close'])
        
        # Reset index if needed
        if df.index.name == 'datetime':
            df = df.reset_index()
        
        return df
    
    @staticmethod
    def _calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI technical indicator"""
        # Calculate price changes
        delta = series.diff()
        
        # Create gain/loss series
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain/loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def _calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD, signal line, and histogram"""
        # Calculate EMAs
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram