# src/data/integrity_checker.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.common.log_manager import LogManager

logger = LogManager.get_logger("trading_system")

class DataIntegrityChecker:
    """数据完整性检查器，验证 OHLCV 数据的时间、交易量和价格一致性"""
    
    def __init__(self, timeframe: str, config: Optional[Dict] = None):
        """
        初始化检查器。
        
        Args:
            timeframe (str): 时间框架（例如 '1m', '1h'）
            config (Optional[Dict]): 配置字典，可包含异常检测参数
        """
        self.expected_interval = self._timeframe_to_seconds(timeframe)
        self.config = config or {}
        self.iqr_multiplier = self.config.get("data", "integrity", "iqr_multiplier", default=3.0)  # IQR 倍数
        self.max_gap_factor = self.config.get("data", "integrity", "max_gap_factor", default=1.5)  # 最大允许时间间隔倍数
        logger.info("DataIntegrityChecker initialized with timeframe=%s, iqr_multiplier=%s", timeframe, self.iqr_multiplier)

    async def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行数据完整性检查（异步接口）。
        
        Args:
            df (pd.DataFrame): 输入数据，需包含 'datetime', 'volume', 'high', 'low', 'close' 等列
        
        Returns:
            Dict[str, Any]: 检查结果，包括缺失周期、交易量异常和价格一致性问题
        """
        if df.empty or 'datetime' not in df.columns:
            logger.warning("输入数据为空或缺少 'datetime' 列")
            return {
                'missing_periods': [],
                'volume_outliers': [],
                'price_consistency': [],
                'duplicate_timestamps': [],
                'unsorted_timestamps': False,
                'stats': {'total_rows': 0, 'missing_count': 0, 'outlier_count': 0}
            }

        # 使用 asyncio.to_thread 将计算密集型任务转为异步
        results = await asyncio.to_thread(self._perform_checks, df)
        
        logger.info("完整性检查完成: %s", results['stats'])
        return results

    def _perform_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """执行具体的完整性检查"""
        results = {
            'missing_periods': self._check_missing_periods(df),
            'volume_outliers': self._check_volume_anomalies(df),
            'price_consistency': self._check_price_consistency(df),
            'duplicate_timestamps': self._check_duplicate_timestamps(df),
            'unsorted_timestamps': self._check_timestamp_order(df),
            'stats': {
                'total_rows': len(df),
                'missing_count': 0,
                'outlier_count': 0
            }
        }
        
        # 更新统计信息
        results['stats']['missing_count'] = len(results['missing_periods'])
        results['stats']['outlier_count'] = len(results['volume_outliers']) + len(results['price_consistency']) + len(results['duplicate_timestamps'])
        
        return results

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """将时间框架转换为秒"""
        units = {'m': 60, 'h': 3600, 'd': 86400}
        num = int(''.join(filter(str.isdigit, timeframe)))
        unit = timeframe[-1]
        return num * units.get(unit, 60)

    def _check_missing_periods(self, df: pd.DataFrame) -> List[int]:
        """检测缺失时间段"""
        time_diff = df['datetime'].diff().dt.total_seconds().dropna()
        gaps = time_diff[time_diff > self.expected_interval * self.max_gap_factor]
        missing_indices = gaps.index.tolist()
        if missing_indices:
            logger.debug("检测到 %d 个缺失时间段", len(missing_indices))
        return missing_indices

    def _check_volume_anomalies(self, df: pd.DataFrame) -> List[int]:
        """检测交易量异常（基于 IQR 方法）"""
        if 'volume' not in df.columns:
            logger.warning("缺少 'volume' 列，无法检查交易量异常")
            return []
        
        q25, q75 = np.percentile(df['volume'], [25, 75])
        iqr = q75 - q25
        upper_bound = q75 + self.iqr_multiplier * iqr
        lower_bound = q25 - self.iqr_multiplier * iqr  # 添加下限检测
        outliers = df[(df['volume'] > upper_bound) | (df['volume'] < lower_bound)].index.tolist()
        if outliers:
            logger.debug("检测到 %d 个交易量异常", len(outliers))
        return outliers

    def _check_price_consistency(self, df: pd.DataFrame) -> List[int]:
        """检查价格合理性"""
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("缺少必要价格列 (%s)，跳过价格一致性检查", required_cols)
            return []
        
        anomalies = df[
            (df['high'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        ]
        anomaly_indices = anomalies.index.tolist()
        if anomaly_indices:
            logger.debug("检测到 %d 个价格一致性问题", len(anomaly_indices))
        return anomaly_indices

    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> List[int]:
        """检测重复时间戳"""
        duplicates = df[df['datetime'].duplicated(keep=False)].index.tolist()
        if duplicates:
            logger.debug("检测到 %d 个重复时间戳", len(duplicates))
        return duplicates

    def _check_timestamp_order(self, df: pd.DataFrame) -> bool:
        """检查时间戳是否按升序排列"""
        is_sorted = df['datetime'].is_monotonic_increasing
        if not is_sorted:
            logger.warning("时间戳未按升序排列")
        return not is_sorted