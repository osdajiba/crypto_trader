# src/utils/data_processor.py

import pandas as pd
import numpy as np
from typing import List


class DataProcessor:
    """集中数据预处理功能"""
    
    @staticmethod
    def clean_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化OHLCV数据清理
        
        Args:
            data: 原始OHLCV数据
            
        Returns:
            pd.DataFrame: 清理后的数据
        """
        if data.empty:
            return data
        
        # 创建副本避免修改原始数据
        df = data.copy()
        
        # 标准化列名
        rename_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if 'time' in lower_col or 'date' in lower_col:
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
        
        # 确保必要的列存在
        required_columns = ['datetime']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必要的列: {col}")
        
        # 处理日期时间列
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 排序并去除重复
        df = df.sort_values('datetime').reset_index(drop=True)
        df = df.drop_duplicates(subset=['datetime'], keep='last')
        
        # 移除无效数据
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        present_numeric_columns = [col for col in numeric_columns if col in df.columns]
        
        if present_numeric_columns:
            # 移除包含NaN的行
            df = df.dropna(subset=present_numeric_columns)
            
            # 确保数值列为浮点型
            for col in present_numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def resample_ohlcv(data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        重采样OHLCV数据到指定的时间周期
        
        Args:
            data: 原始OHLCV数据
            timeframe: 目标时间周期 (例如 '1m', '1h', '1d')
            
        Returns:
            pd.DataFrame: 重采样后的数据
        """
        if data.empty:
            return data
        
        if 'datetime' not in data.columns:
            raise ValueError("数据必须包含'datetime'列")
        
        # 确保datetime是索引
        df = data.copy()
        if df.index.name != 'datetime':
            df = df.set_index('datetime')
        
        # 将字符串时间周期转换为pandas周期字符串
        time_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        pandas_timeframe = time_map.get(timeframe, timeframe)
        
        # 执行重采样
        resampled = df.resample(pandas_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # 重置索引
        resampled = resampled.reset_index()
        
        return resampled
    
    @staticmethod
    def fill_missing_periods(data: pd.DataFrame, timeframe: str, start=None, end=None) -> pd.DataFrame:
        """
        填充缺失的时间周期
        
        Args:
            data: 原始数据
            timeframe: 时间周期
            start: 开始时间 (可选)
            end: 结束时间 (可选)
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        if data.empty:
            return data
        
        if 'datetime' not in data.columns:
            raise ValueError("数据必须包含'datetime'列")
        
        # 确保datetime是正确的类型
        df = data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 设置索引
        df = df.set_index('datetime')
        
        # 确定开始和结束时间
        start_time = start if start is not None else df.index.min()
        end_time = end if end is not None else df.index.max()
        
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.to_datetime(start_time)
        if not isinstance(end_time, pd.Timestamp):
            end_time = pd.to_datetime(end_time)
        
        # 创建完整的时间序列
        time_map = {
            '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
        }
        
        pandas_timeframe = time_map.get(timeframe, timeframe)
        full_index = pd.date_range(start=start_time, end=end_time, freq=pandas_timeframe)
        
        # 重新索引数据
        filled_df = df.reindex(full_index)
        
        # 向前填充数据 (OHLC)
        for col in ['open', 'high', 'low', 'close']:
            if col in filled_df.columns:
                filled_df[col] = filled_df[col].ffill()
        
        # 将缺失的成交量设为0
        if 'volume' in filled_df.columns:
            filled_df['volume'] = filled_df['volume'].fillna(0)
        
        # 重置索引
        filled_df = filled_df.reset_index().rename(columns={'index': 'datetime'})
        
        return filled_df
    
    @staticmethod
    def merge_dataframes(main_df: pd.DataFrame, other_df: pd.DataFrame, on: str = 'datetime', 
                         suffixes: tuple = ('', '_right')) -> pd.DataFrame:
        """
        合并两个数据框，处理重复列
        
        Args:
            main_df: 主数据框
            other_df: 要合并的数据框
            on: 合并列名
            suffixes: 后缀元组
            
        Returns:
            pd.DataFrame: 合并后的数据框
        """
        if main_df.empty:
            return other_df.copy()
        if other_df.empty:
            return main_df.copy()
        
        # 处理时间列
        for df in [main_df, other_df]:
            if on in df.columns and not pd.api.types.is_datetime64_any_dtype(df[on]):
                df[on] = pd.to_datetime(df[on])
        
        # 执行合并
        merged = pd.merge(main_df, other_df, on=on, how='outer', suffixes=suffixes)
        
        # 整理列名
        return merged
    
    @staticmethod
    def detect_outliers(data: pd.DataFrame, columns: List[str] = None, method: str = 'iqr', 
                      threshold: float = 1.5) -> pd.DataFrame:
        """
        检测并标记异常值
        
        Args:
            data: 输入数据
            columns: 要检查的列
            method: 检测方法 ('iqr' 或 'zscore')
            threshold: 异常值阈值
            
        Returns:
            pd.DataFrame: 带有异常值标记的数据
        """
        if data.empty:
            return data
        
        # 如果没有指定列，使用所有数值列
        if columns is None:
            columns = data.select_dtypes(include=np.number).columns.tolist()
        else:
            # 仅保留数据中存在的列
            columns = [col for col in columns if col in data.columns]
        
        df = data.copy()
        
        # 为每列创建标志
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
                if std == 0:  # 避免除零错误
                    df[outlier_col] = False
                else:
                    z_scores = (df[col] - mean) / std
                    df[outlier_col] = abs(z_scores) > threshold
            
            else:
                raise ValueError(f"不支持的方法: {method}")
        
        # 添加全局异常标志
        outlier_cols = [col for col in df.columns if col.endswith('_is_outlier')]
        if outlier_cols:
            df['is_outlier'] = df[outlier_cols].any(axis=1)
        
        return df