# src/utils/data_processor.py

import pandas as pd
import numpy as np
from typing import List


class DataProcessor:
    """Centralized data preprocessing functionalities"""
    
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
        
        # Ensure required columns exist
        required_columns = ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'start_ts', 'end_ts']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        def convert_timestamp(ts):
            try:
                ts_str = str(ts)
                if len(ts_str) == 10:
                    return int(ts) * 1000
                elif len(ts_str) == 13:
                    return int(ts)
                else:
                    return pd.NaT
            except:
                return pd.NaT

        # Standardize timestamp units
        df['datetime'] = df['datetime'].apply(convert_timestamp)
        df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', errors='coerce')
        
        # Sort and remove duplicates
        df = df.reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # Remove invalid data
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        present_numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        if present_numeric_columns:
            # Remove rows with NaN
            df = df.dropna(subset=present_numeric_columns)
            
            # Ensure numeric columns are float type
            for col in present_numeric_columns:
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
        
        return resampled
    
    @staticmethod
    def fill_missing_periods(data: pd.DataFrame, timeframe: str, start=None, end=None) -> pd.DataFrame:
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
        
        return filled_df
    
    @staticmethod
    def merge_dataframes(main_df: pd.DataFrame, other_df: pd.DataFrame, on: str = 'datetime', 
                         suffixes: tuple = ('', '_right')) -> pd.DataFrame:
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
        
        # Clean column names
        return merged
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', 
                      threshold: float = 1.5) -> pd.DataFrame:
        """
        Detect and flag outliers
        
        Args:
            data: Input data
            columns: Columns to check
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