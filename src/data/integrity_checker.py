# src/data/integrity_checker.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import traceback

from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor

logger = LogManager.get_logger("trading_system")

class DataIntegrityChecker:
    """
    增强版数据完整性检查器，用于验证OHLCV数据的质量和完整性
    
    提供时间序列完整性检查、异常值检测和数据修复建议。
    """
    
    def __init__(self, timeframe: str, config: Optional[Dict] = None):
        """
        初始化检查器。
        
        Args:
            timeframe (str): 时间框架（例如 '1m', '1h', '1d'）
            config (Optional[Dict]): 配置字典，可包含异常检测参数
        """
        self.timeframe = timeframe
        self.expected_interval = self._timeframe_to_seconds(timeframe)
        self.config = config or {}
        
        # 检测参数
        self.iqr_multiplier = self.config.get("data", "integrity", "iqr_multiplier", default=3.0)  # IQR倍数用于异常检测
        self.max_gap_factor = self.config.get("data", "integrity", "max_gap_factor", default=1.5)  # 最大允许时间间隔倍数
        self.min_volume_threshold = self.config.get("data", "integrity", "min_volume", default=0)  # 最小交易量
        
        # 修复选项
        self.auto_repair = self.config.get("data", "integrity", "auto_repair", default=False)
        
        # 异步执行器
        self.executor = AsyncExecutor()
        
        logger.info(f"DataIntegrityChecker初始化完成: timeframe={timeframe}, iqr_multiplier={self.iqr_multiplier}, max_gap_factor={self.max_gap_factor}")

    async def check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行数据完整性检查（异步接口）。
        
        Args:
            df (pd.DataFrame): 输入数据，需包含'datetime', 'volume', 'high', 'low', 'close'等列
        
        Returns:
            Dict[str, Any]: 检查结果，包括缺失周期、异常值和修复建议
        """
        # 基本验证
        if df.empty or 'datetime' not in df.columns:
            logger.warning("输入数据为空或缺少'datetime'列")
            return self._empty_result()
            
        # 确保执行器已启动
        await self.executor.start()

        # 使用AsyncExecutor运行计算密集型任务
        try:
            results = await self.executor.submit(self._perform_checks, df)
            
            # 如果启用了自动修复，执行修复
            if self.auto_repair and (results['missing_periods'] or results['volume_outliers'] or results['price_consistency']):
                logger.info("检测到数据问题，尝试自动修复")
                df_fixed = await self.executor.submit(self._repair_data, df, results)
                
                # 再次检查修复后的数据
                post_repair_results = await self.executor.submit(self._perform_checks, df_fixed)
                
                # 添加修复信息到结果
                results['fixed_data'] = df_fixed
                results['post_repair'] = post_repair_results
                results['repair_summary'] = {
                    'missing_fixed': len(results['missing_periods']) - len(post_repair_results['missing_periods']),
                    'outliers_fixed': len(results['volume_outliers']) - len(post_repair_results['volume_outliers']),
                    'price_issues_fixed': len(results['price_consistency']) - len(post_repair_results['price_consistency'])
                }
            
            logger.info(f"完整性检查完成: {results['stats']}")
            return results
            
        except Exception as e:
            logger.error(f"数据完整性检查失败: {str(e)}\n{traceback.format_exc()}")
            return self._empty_result()

    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果结构"""
        return {
            'missing_periods': [],
            'volume_outliers': [],
            'price_consistency': [],
            'duplicate_timestamps': [],
            'unsorted_timestamps': False,
            'stats': {'total_rows': 0, 'missing_count': 0, 'outlier_count': 0}
        }

    def _perform_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行所有数据完整性检查
        
        Args:
            df: 要检查的数据
            
        Returns:
            Dict: 包含各种检查结果的字典
        """
        # 确保日期时间列是datetime类型
        if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
            df['datetime'] = pd.to_datetime(df['datetime'])
            
        # 排序数据以确保正确的检查
        sorted_df = df.sort_values('datetime').reset_index(drop=True)
        was_unsorted = not df['datetime'].equals(sorted_df['datetime'])
        
        # 执行各种检查
        missing_periods = self._check_missing_periods(sorted_df)
        volume_outliers = self._check_volume_anomalies(sorted_df)
        price_consistency = self._check_price_consistency(sorted_df)
        duplicate_timestamps = self._check_duplicate_timestamps(sorted_df)
        
        # 收集结果
        results = {
            'missing_periods': missing_periods,
            'volume_outliers': volume_outliers,
            'price_consistency': price_consistency,
            'duplicate_timestamps': duplicate_timestamps,
            'unsorted_timestamps': was_unsorted,
            'stats': {
                'total_rows': len(df),
                'missing_count': len(missing_periods),
                'outlier_count': len(volume_outliers) + len(price_consistency) + len(duplicate_timestamps),
                'period_coverage': self._calculate_coverage(sorted_df)
            }
        }
        
        # 添加详细信息
        if missing_periods:
            results['missing_details'] = self._get_missing_period_details(sorted_df, missing_periods)
            
        if price_consistency:
            results['price_details'] = self._get_price_issue_details(sorted_df, price_consistency)
            
        # 添加总体质量评分
        results['quality_score'] = self._calculate_quality_score(results)
        
        return results

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """
        将时间框架转换为秒数
        
        Args:
            timeframe: 时间框架字符串(如'1m', '1h', '1d')
            
        Returns:
            int: 等效的秒数
        """
        units = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        
        # 确保有效的时间框架
        if not timeframe or not any(unit in timeframe for unit in units):
            logger.warning(f"无效的时间框架: {timeframe}，使用默认值60秒")
            return 60
        
        try:
            # 提取数字和单位
            num = int(''.join(filter(str.isdigit, timeframe)))
            unit = timeframe[-1].lower()
            
            if unit not in units:
                logger.warning(f"未知的时间单位: {unit}，使用分钟")
                unit = 'm'
                
            return num * units.get(unit, 60)
            
        except Exception as e:
            logger.error(f"时间框架转换失败: {str(e)}")
            return 60  # 默认为1分钟

    def _check_missing_periods(self, df: pd.DataFrame) -> List[int]:
        """
        检测缺失的时间段
        
        Args:
            df: 要检查的数据框
            
        Returns:
            List[int]: 缺失期间开始的索引列表
        """
        if df.empty or len(df) < 2:
            return []
            
        # 计算时间差
        time_diff = df['datetime'].diff().dt.total_seconds().dropna()
        
        # 查找异常大的间隔
        max_allowed_gap = self.expected_interval * self.max_gap_factor
        gaps = time_diff[time_diff > max_allowed_gap]
        
        # 获取索引
        missing_indices = gaps.index.tolist()
        
        if missing_indices:
            logger.debug(f"检测到{len(missing_indices)}个缺失时间段")
            
        return missing_indices

    def _check_volume_anomalies(self, df: pd.DataFrame) -> List[int]:
        """
        检测交易量异常（基于IQR方法）
        
        Args:
            df: 要检查的数据框
            
        Returns:
            List[int]: 包含异常交易量的行索引
        """
        if df.empty or 'volume' not in df.columns:
            logger.warning("缺少'volume'列，无法检查交易量异常")
            return []
        
        try:
            # 计算四分位数
            q25, q75 = np.percentile(df['volume'], [25, 75])
            iqr = q75 - q25
            
            # 计算上下边界
            upper_bound = q75 + self.iqr_multiplier * iqr
            # 使用配置的最小值或计算的下限（取较大值）
            lower_bound = max(self.min_volume_threshold, q25 - self.iqr_multiplier * iqr)
            
            # 查找异常值
            outliers = df[(df['volume'] > upper_bound) | (df['volume'] < lower_bound)].index.tolist()
            
            if outliers:
                logger.debug(f"检测到{len(outliers)}个交易量异常")
                
            return outliers
            
        except Exception as e:
            logger.error(f"交易量异常检测失败: {str(e)}")
            return []

    def _check_price_consistency(self, df: pd.DataFrame) -> List[int]:
        """
        检查价格合理性
        
        Args:
            df: 要检查的数据框
            
        Returns:
            List[int]: 包含价格不一致的行索引
        """
        required_cols = ['high', 'low', 'close', 'open']
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"缺少必要价格列({required_cols})，跳过价格一致性检查")
            return []
        
        try:
            # 检查各种价格关系
            anomalies = df[
                (df['high'] < df['low']) |                   # 最高价低于最低价
                (df['close'] > df['high']) |                 # 收盘价高于最高价
                (df['close'] < df['low']) |                  # 收盘价低于最低价
                (df['open'] > df['high']) |                  # 开盘价高于最高价
                (df['open'] < df['low'])                     # 开盘价低于最低价
            ]
            
            anomaly_indices = anomalies.index.tolist()
            
            if anomaly_indices:
                logger.debug(f"检测到{len(anomaly_indices)}个价格一致性问题")
                
            return anomaly_indices
            
        except Exception as e:
            logger.error(f"价格一致性检查失败: {str(e)}")
            return []

    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> List[int]:
        """
        检测重复时间戳
        
        Args:
            df: 要检查的数据框
            
        Returns:
            List[int]: 包含重复时间戳的行索引
        """
        try:
            # 查找重复的日期时间
            duplicates = df[df['datetime'].duplicated(keep=False)].index.tolist()
            
            if duplicates:
                logger.debug(f"检测到{len(duplicates)}个重复时间戳")
                
            return duplicates
            
        except Exception as e:
            logger.error(f"重复时间戳检查失败: {str(e)}")
            return []

    def _check_timestamp_order(self, df: pd.DataFrame) -> bool:
        """
        检查时间戳是否按升序排列
        
        Args:
            df: 要检查的数据框
            
        Returns:
            bool: 如果时间戳未按升序排列则为True
        """
        try:
            is_sorted = df['datetime'].is_monotonic_increasing
            
            if not is_sorted:
                logger.warning("时间戳未按升序排列")
                
            return not is_sorted
            
        except Exception as e:
            logger.error(f"时间戳顺序检查失败: {str(e)}")
            return False

    def _get_missing_period_details(self, df: pd.DataFrame, missing_indices: List[int]) -> List[Dict[str, Any]]:
        """
        获取缺失期间的详细信息
        
        Args:
            df: 数据框
            missing_indices: 缺失期间开始的索引列表
            
        Returns:
            List[Dict]: 包含缺失期间详细信息的字典列表
        """
        details = []
        
        for idx in missing_indices:
            if idx < len(df) and idx > 0:
                # 获取缺失前后的时间戳
                before_time = df.iloc[idx-1]['datetime']
                after_time = df.iloc[idx]['datetime']
                
                # 计算缺失的周期数
                gap_seconds = (after_time - before_time).total_seconds()
                missing_periods = int(gap_seconds / self.expected_interval) - 1
                
                details.append({
                    'index': idx,
                    'before_time': before_time,
                    'after_time': after_time,
                    'gap_seconds': gap_seconds,
                    'missing_periods': missing_periods,
                    'expected_interval': self.expected_interval
                })
                
        return details

    def _get_price_issue_details(self, df: pd.DataFrame, price_indices: List[int]) -> List[Dict[str, Any]]:
        """
        获取价格问题的详细信息
        
        Args:
            df: 数据框
            price_indices: 价格问题的索引列表
            
        Returns:
            List[Dict]: 包含价格问题详细信息的字典列表
        """
        details = []
        
        for idx in price_indices:
            if idx < len(df):
                row = df.iloc[idx]
                
                # 检查具体的价格问题
                issues = []
                if row['high'] < row['low']:
                    issues.append('high_below_low')
                if row['close'] > row['high']:
                    issues.append('close_above_high')
                if row['close'] < row['low']:
                    issues.append('close_below_low')
                if row['open'] > row['high']:
                    issues.append('open_above_high')
                if row['open'] < row['low']:
                    issues.append('open_below_low')
                
                details.append({
                    'index': idx,
                    'datetime': row['datetime'],
                    'open': row['open'],
                    'high': row['high'],
                    'low': row['low'],
                    'close': row['close'],
                    'issues': issues
                })
                
        return details

    def _calculate_coverage(self, df: pd.DataFrame) -> float:
        """
        计算数据覆盖率
        
        Args:
            df: 数据框
            
        Returns:
            float: 0到1之间的覆盖率
        """
        if df.empty or len(df) < 2:
            return 0.0
            
        # 计算理想情况下的周期数
        start_time = df['datetime'].min()
        end_time = df['datetime'].max()
        total_seconds = (end_time - start_time).total_seconds()
        ideal_periods = int(total_seconds / self.expected_interval) + 1
        
        # 计算实际周期数
        actual_periods = len(df)
        
        # 计算覆盖率
        coverage = min(1.0, actual_periods / max(1, ideal_periods))
        
        return coverage

    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """
        计算数据质量得分
        
        Args:
            results: 检查结果字典
            
        Returns:
            float: 0到100之间的质量得分
        """
        if results['stats']['total_rows'] == 0:
            return 0.0
            
        # 计算各项指标的权重
        total_rows = results['stats']['total_rows']
        coverage_weight = 0.4
        missing_weight = 0.3
        consistency_weight = 0.2
        outlier_weight = 0.1
        
        # 计算各项得分
        coverage_score = results['stats'].get('period_coverage', 0) * 100
        
        missing_score = 100 * (1 - len(results['missing_periods']) / max(1, total_rows))
        consistency_score = 100 * (1 - len(results['price_consistency']) / max(1, total_rows))
        outlier_score = 100 * (1 - len(results['volume_outliers']) / max(1, total_rows))
        
        # 计算加权总分
        total_score = (
            coverage_weight * coverage_score +
            missing_weight * missing_score + 
            consistency_weight * consistency_score +
            outlier_weight * outlier_score
        )
        
        return round(max(0, min(100, total_score)), 2)

    def _repair_data(self, df: pd.DataFrame, check_results: Dict[str, Any]) -> pd.DataFrame:
        """
        修复数据问题
        
        Args:
            df: 原始数据框
            check_results: 检查结果字典
            
        Returns:
            pd.DataFrame: 修复后的数据框
        """
        if df.empty:
            return df
            
        # 创建数据的副本
        fixed_df = df.copy()
        
        # 1. 修复数据排序问题
        if check_results['unsorted_timestamps']:
            fixed_df = fixed_df.sort_values('datetime').reset_index(drop=True)
            logger.info("已按时间戳对数据进行排序")
            
        # 2. 修复重复的时间戳
        if check_results['duplicate_timestamps']:
            fixed_df = fixed_df.drop_duplicates(subset=['datetime'], keep='first').reset_index(drop=True)
            logger.info(f"已移除{len(check_results['duplicate_timestamps'])}个重复时间戳")
            
        # 3. 修复价格一致性问题
        if check_results['price_consistency']:
            for idx in check_results['price_consistency']:
                if idx < len(fixed_df):
                    row = fixed_df.loc[idx]
                    
                    # 确保合理的价格关系
                    high = row['high']
                    low = row['low']
                    close = row['close']
                    open_price = row['open']
                    
                    # 修复高低价关系
                    if high < low:
                        # 交换高低价
                        fixed_df.at[idx, 'high'] = low
                        fixed_df.at[idx, 'low'] = high
                        high, low = low, high
                    
                    # 修复收盘价超出范围
                    if close > high:
                        fixed_df.at[idx, 'close'] = high
                    elif close < low:
                        fixed_df.at[idx, 'close'] = low
                        
                    # 修复开盘价超出范围
                    if open_price > high:
                        fixed_df.at[idx, 'open'] = high
                    elif open_price < low:
                        fixed_df.at[idx, 'open'] = low
            
            logger.info(f"已修复{len(check_results['price_consistency'])}个价格一致性问题")
            
        # 4. 修复交易量异常
        if check_results['volume_outliers']:
            # 计算中位数交易量
            median_volume = fixed_df['volume'].median()
            
            for idx in check_results['volume_outliers']:
                if idx < len(fixed_df):
                    # 将异常交易量替换为中位数
                    fixed_df.at[idx, 'volume'] = median_volume
            
            logger.info(f"已修复{len(check_results['volume_outliers'])}个交易量异常")
            
        # 5. 填充缺失的时间段
        if check_results['missing_periods'] and 'missing_details' in check_results:
            new_rows = []
            
            for detail in check_results['missing_details']:
                before_time = detail['before_time']
                missing_periods = detail['missing_periods']
                
                # 生成缺失的时间点
                for i in range(1, missing_periods + 1):
                    new_time = before_time + timedelta(seconds=i * self.expected_interval)
                    
                    # 创建估算的OHLCV数据
                    # 使用前一行为基础，但交易量设为0
                    before_idx = detail['index'] - 1
                    if before_idx >= 0 and before_idx < len(fixed_df):
                        before_row = fixed_df.iloc[before_idx].copy()
                        new_row = before_row.copy()
                        new_row['datetime'] = new_time
                        new_row['volume'] = 0  # 缺失的周期交易量为0
                        
                        # 将开高低收都设置为前一个收盘价
                        new_row['open'] = before_row['close']
                        new_row['high'] = before_row['close']
                        new_row['low'] = before_row['close']
                        new_row['close'] = before_row['close']
                        
                        new_rows.append(new_row)
            
            # 将新行添加到数据框
            if new_rows:
                # 合并新行到原始数据框
                extended_df = pd.concat([fixed_df] + [pd.DataFrame([row]) for row in new_rows], ignore_index=True)
                
                # 重新排序
                fixed_df = extended_df.sort_values('datetime').reset_index(drop=True)
                
                logger.info(f"已填充{len(new_rows)}个缺失的时间段")
                
        return fixed_df