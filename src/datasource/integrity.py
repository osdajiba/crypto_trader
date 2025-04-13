# src/data/integrity_checker.py

from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Coroutine
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os

from common.logging import LogManager


class DataIntegrityChecker:
    """
    Optimized Data Integrity Checker
    
    Checks for common data issues with enhanced performance:
    - Missing values
    - Duplicate timestamps
    - Price anomalies
    - Out-of-sequence data
    - Volume anomalies
    
    Includes hardware acceleration, vectorized operations, and parallel processing.
    """
    
    def __init__(self, timeframe: str = "1h", iqr_multiplier: float = 3.0, parallel: bool = True):
        """
        Initialize the data integrity checker
        
        Args:
            timeframe: Data timeframe for context-appropriate checks
            iqr_multiplier: Multiplier for IQR-based outlier detection
            parallel: Whether to use parallel processing for checks
        """
        self.logger = LogManager.get_logger("data.integrity")
        self.timeframe = timeframe
        self.iqr_multiplier = iqr_multiplier
        self.parallel = parallel
        
        # Configure thread pool for CPU-bound operations
        max_workers = min(os.cpu_count() or 4, 8)  # Limit to reasonable number
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Check function mapping
        self.check_functions = {
            "missing_values": self._check_missing_values,
            "duplicate_timestamps": self._check_duplicate_timestamps,
            "timestamp_sequence": self._check_timestamp_sequence,
            "price_anomalies": self._check_price_anomalies,
            "volume_anomalies": self._check_volume_anomalies,
            "zero_values": self._check_zero_values,
            "ohlc_consistency": self._check_ohlc_consistency
        }
        
        # Stats tracking
        self._validation_stats = {
            "checks_performed": 0,
            "issues_detected": 0,
            "issues_fixed": 0,
            "issue_types": {}
        }
        
        self.logger.info(f"DataIntegrityChecker initialized with timeframe={timeframe}, iqr_multiplier={iqr_multiplier}, parallel={parallel}")
    
    async def check(self, df: pd.DataFrame, symbol: str = None, timeframe: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all integrity checks on the dataframe
        
        Args:
            df: DataFrame to check
            symbol: Symbol being checked (for logging)
            timeframe: Timeframe being checked (for logging)
            
        Returns:
            Tuple[bool, Dict]: (passed, results) where results contains details of checks
        """
        try:
            if df.empty:
                self.logger.warning(f"Empty dataframe provided for integrity check {symbol} {timeframe}")
                return False, {"error": "Empty dataframe"}
            
            # Use different approaches based on dataframe size and settings
            if len(df) > 10000 and self.parallel:
                results = await self._perform_parallel_checks(df)
            else:
                results = await self._perform_checks(df)
            
            # Update stats
            self._validation_stats["checks_performed"] += 1
            
            # Count issues by type
            issues_found = 0
            for check_name, check_result in results.items():
                if isinstance(check_result, dict) and check_result.get("passed") is False:
                    issues_found += 1
                    self._validation_stats["issue_types"].setdefault(check_name, 0)
                    self._validation_stats["issue_types"][check_name] += 1
            
            if issues_found > 0:
                self._validation_stats["issues_detected"] += 1
                
                # Log a summary of issues
                issue_types = [name for name, result in results.items() 
                              if isinstance(result, dict) and result.get("passed") is False]
                
                entity_info = ""
                if symbol and timeframe:
                    entity_info = f" for {symbol} {timeframe}"
                
                self.logger.warning(f"Found {issues_found} data integrity issues{entity_info}: {', '.join(issue_types)}")
            
            # Determine overall pass/fail
            passed = all(result.get("passed", False) for result in results.values() if isinstance(result, dict))
            
            return passed, results
            
        except Exception as e:
            self.logger.error(f"Data integrity check failed: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def _perform_parallel_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform integrity checks in parallel for large dataframes
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict: Results of all checks
        """
        # Ensure we have a copy to avoid modifications
        df_copy = df.copy()
        
        # Convert to datetime if needed
        if 'datetime' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['datetime']):
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        
        # Create tasks for each check
        tasks = []
        for check_name, check_func in self.check_functions.items():
            tasks.append(asyncio.ensure_future(check_func(df_copy)))
        
        # Run all checks concurrently
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        results = {}
        for i, (check_name, _) in enumerate(self.check_functions.items()):
            result = check_results[i]
            if isinstance(result, Exception):
                self.logger.error(f"Error in {check_name} check: {str(result)}")
                results[check_name] = {"passed": False, "error": str(result)}
            else:
                results[check_name] = result
        
        return results
    
    async def _perform_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform integrity checks sequentially
        
        Args:
            df: DataFrame to check
            
        Returns:
            Dict: Results of all checks
        """
        # Ensure we have a copy to avoid modifications
        df_copy = df.copy()
        
        # Convert to datetime if needed
        if 'datetime' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['datetime']):
            df_copy['datetime'] = pd.to_datetime(df_copy['datetime'])
        
        # Run checks
        results = {}
        for check_name, check_func in self.check_functions.items():
            try:
                results[check_name] = await check_func(df_copy)
            except Exception as e:
                self.logger.error(f"Error in {check_name} check: {str(e)}")
                results[check_name] = {"passed": False, "error": str(e)}
        
        return results
    
    async def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in critical columns using vectorized operations"""
        critical_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in critical_columns if col in df.columns]
        
        if not available_columns:
            return {"passed": False, "error": "No OHLC columns found"}
        
        # Use vectorized operations
        missing_counts = df[available_columns].isna().sum().to_dict()
        missing_total = sum(missing_counts.values())
        
        if missing_total > 0:
            missing_percentage = missing_total / (len(df) * len(available_columns)) * 100
            return {
                "passed": missing_percentage < 1.0,  # Allow up to 1% missing
                "missing_counts": missing_counts,
                "missing_percentage": missing_percentage,
                "message": f"Found {missing_total} missing values ({missing_percentage:.2f}%)"
            }
        
        return {"passed": True, "missing_counts": missing_counts}
    
    async def _check_duplicate_timestamps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for duplicate timestamps with optimized algorithm"""
        timestamp_col = self._identify_timestamp_column(df)
        
        if timestamp_col is None:
            return {"passed": False, "error": "No timestamp column found"}
        
        # For large dataframes, use hash-based approach for better performance
        if len(df) > 100000:
            # Create value counts series
            value_counts = df[timestamp_col].value_counts()
            
            # Filter to get only duplicates (count > 1)
            duplicates = value_counts[value_counts > 1]
            duplicate_count = duplicates.sum() - len(duplicates)
            
            if duplicate_count > 0:
                # Get a sample of duplicate timestamps for reporting
                duplicate_values = duplicates.index[:10].tolist()
                
                return {
                    "passed": False,
                    "duplicate_count": duplicate_count,
                    "duplicate_timestamps": duplicate_values,
                    "message": f"Found {duplicate_count} duplicate timestamps"
                }
        else:
            # For smaller dataframes, use standard duplicated method
            duplicates = df[timestamp_col].duplicated()
            duplicate_count = duplicates.sum()
            
            if duplicate_count > 0:
                duplicate_timestamps = df.loc[duplicates, timestamp_col].head(10).tolist()
                
                return {
                    "passed": False,
                    "duplicate_count": duplicate_count,
                    "duplicate_timestamps": duplicate_timestamps,
                    "message": f"Found {duplicate_count} duplicate timestamps"
                }
        
        return {"passed": True, "duplicate_count": 0}
    
    async def _check_timestamp_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if timestamps are in ascending order using optimized approach"""
        timestamp_col = self._identify_timestamp_column(df)
        
        if timestamp_col is None:
            return {"passed": False, "error": "No timestamp column found"}
        
        # Use faster numpy operations for large dataframes
        if len(df) > 10000:
            # Convert to numpy array for faster processing
            timestamps = df[timestamp_col].to_numpy()
            
            # Check if sorted
            is_sorted = np.all(np.diff(timestamps) >= np.timedelta64(0))
            
            if not is_sorted:
                self.logger.warning("Timestamps not in ascending order")
                return {
                    "passed": False,
                    "message": "Timestamps not in ascending order",
                    "suggestion": "Data should be sorted by timestamp"
                }
        else:
            # For smaller dataframes, use pandas is_monotonic_increasing
            if not df[timestamp_col].is_monotonic_increasing:
                self.logger.warning("Timestamps not in ascending order")
                return {
                    "passed": False,
                    "message": "Timestamps not in ascending order",
                    "suggestion": "Data should be sorted by timestamp"
                }
        
        return {"passed": True}
    
    async def _check_price_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price anomalies using robust IQR method with vectorized operations"""
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if not price_cols:
            return {"passed": False, "error": "No price columns found"}
        
        anomalies = {}
        
        for col in price_cols:
            # Calculate IQR using numpy for performance
            values = df[col].to_numpy()
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            # Find outliers using vectorized operations
            outliers_mask = (values < lower_bound) | (values > upper_bound)
            outlier_count = np.sum(outliers_mask)
            
            if outlier_count > 0:
                # Get example outlier values
                outlier_indices = np.where(outliers_mask)[0][:5]  # Get up to 5 indices
                example_values = values[outlier_indices].tolist()
                
                anomalies[col] = {
                    "count": int(outlier_count),
                    "percentage": (outlier_count / len(df)) * 100,
                    "bounds": (float(lower_bound), float(upper_bound)),
                    "examples": example_values
                }
        
        if anomalies:
            total_anomalies = sum(info["count"] for info in anomalies.values())
            total_percentage = sum(info["percentage"] for info in anomalies.values()) / len(price_cols)
            
            return {
                "passed": total_percentage < 1.0,  # Allow up to 1% anomalies
                "anomalies": anomalies,
                "message": f"Found {total_anomalies} price anomalies ({total_percentage:.2f}%)"
            }
        
        return {"passed": True, "anomalies": {}}
    
    async def _check_volume_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for volume anomalies using vectorized operations"""
        if 'volume' not in df.columns:
            return {"passed": True, "message": "No volume column to check"}
        
        # Use numpy for performance
        volumes = df['volume'].to_numpy()
        
        # Check for negative volumes
        negative_mask = volumes < 0
        negative_count = np.sum(negative_mask)
        
        # Check for extreme volumes using IQR
        q1 = np.percentile(volumes, 25)
        q3 = np.percentile(volumes, 75)
        iqr = q3 - q1
        
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        extreme_mask = volumes > upper_bound
        extreme_count = np.sum(extreme_mask)
        
        issues = {}
        
        if negative_count > 0:
            negative_indices = np.where(negative_mask)[0][:5]  # Get up to 5 indices
            issues["negative"] = {
                "count": int(negative_count),
                "examples": volumes[negative_indices].tolist()
            }
        
        if extreme_count > 0:
            extreme_indices = np.where(extreme_mask)[0][:5]  # Get up to 5 indices
            issues["extreme"] = {
                "count": int(extreme_count),
                "percentage": (extreme_count / len(df)) * 100,
                "threshold": float(upper_bound),
                "examples": volumes[extreme_indices].tolist()
            }
        
        if issues:
            is_severe = "negative" in issues or (
                "extreme" in issues and issues["extreme"]["percentage"] > 5.0
            )
            
            return {
                "passed": not is_severe,
                "issues": issues,
                "message": "Volume anomalies detected"
            }
        
        return {"passed": True}
    
    async def _check_zero_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for zero values in critical columns using vectorized operations"""
        check_columns = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
        
        if not check_columns:
            return {"passed": True, "message": "No columns to check for zeros"}
        
        zero_counts = {}
        
        for col in check_columns:
            # Use numpy for performance
            values = df[col].to_numpy()
            zero_mask = values == 0
            zero_count = np.sum(zero_mask)
            
            if zero_count > 0:
                zero_counts[col] = {
                    "count": int(zero_count),
                    "percentage": (zero_count / len(df)) * 100
                }
        
        if zero_counts:
            # Allow some zero volumes but no zero prices
            has_zero_prices = any(col in zero_counts for col in ['open', 'high', 'low', 'close'])
            high_zero_volume = 'volume' in zero_counts and zero_counts['volume']["percentage"] > 10.0
            
            return {
                "passed": not (has_zero_prices or high_zero_volume),
                "zero_counts": zero_counts,
                "message": "Zero values detected in critical columns"
            }
        
        return {"passed": True, "zero_counts": {}}
    
    async def _check_ohlc_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for OHLC price consistency using vectorized operations"""
        ohlc_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if len(ohlc_cols) < 4:
            return {"passed": True, "message": "Not enough OHLC columns for consistency check"}
        
        # Use numpy arrays for better performance
        high_array = df['high'].to_numpy()
        low_array = df['low'].to_numpy()
        open_array = df['open'].to_numpy()
        close_array = df['close'].to_numpy()
        
        # Create masks for inconsistencies
        high_lt_low_mask = high_array < low_array
        high_lt_open_mask = high_array < open_array
        high_lt_close_mask = high_array < close_array
        low_gt_open_mask = low_array > open_array
        low_gt_close_mask = low_array > close_array
        
        # Count inconsistencies
        counts = {
            "high_vs_low": int(np.sum(high_lt_low_mask)),
            "high_vs_open": int(np.sum(high_lt_open_mask)),
            "high_vs_close": int(np.sum(high_lt_close_mask)),
            "low_vs_open": int(np.sum(low_gt_open_mask)),
            "low_vs_close": int(np.sum(low_gt_close_mask))
        }
        
        # Get indices of inconsistencies
        inconsistencies = {}
        for key, mask in [
            ("high_vs_low", high_lt_low_mask),
            ("high_vs_open", high_lt_open_mask),
            ("high_vs_close", high_lt_close_mask),
            ("low_vs_open", low_gt_open_mask),
            ("low_vs_close", low_gt_close_mask)
        ]:
            count = counts[key]
            if count > 0:
                inconsistencies[key] = np.where(mask)[0].tolist()[:10]  # Limit to first 10
        
        if inconsistencies:
            total_rows = sum(counts.values())
            return {
                "passed": False,
                "inconsistencies": inconsistencies,
                "counts": counts,
                "count": total_rows,
                "message": f"Found {total_rows} OHLC consistency issues"
            }
        
        return {"passed": True}
    
    def _identify_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Identify the timestamp column in the dataframe"""
        # Try common timestamp column names
        timestamp_candidates = ['datetime', 'timestamp', 'date', 'time']
        
        for col in timestamp_candidates:
            if col in df.columns:
                return col
        
        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            return df.index.name or 'index'
        
        return None
    
    def register_fix(self, issue_types: List[str]) -> None:
        """
        Register that issues have been fixed to track statistics
        
        Args:
            issue_types: List of issue types that were fixed
        """
        self._validation_stats["issues_fixed"] += 1
        
        for issue_type in issue_types:
            self._validation_stats.setdefault("fixed_types", {})
            self._validation_stats["fixed_types"].setdefault(issue_type, 0)
            self._validation_stats["fixed_types"][issue_type] += 1
        
        self.logger.info(f"Registered fixes for issue types: {', '.join(issue_types)}")
        
        # Update detailed statistics
        self._validation_stats.setdefault("fix_history", []).append({
            "timestamp": datetime.now().isoformat(),
            "issue_types": issue_types,
            "count": len(issue_types)
        })
        
        # Trim history if it gets too large
        if len(self._validation_stats.get("fix_history", [])) > 1000:
            self._validation_stats["fix_history"] = self._validation_stats["fix_history"][-1000:]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about data integrity checks
        
        Returns:
            Dict: Validation statistics
        """
        return self._validation_stats.copy()
    
    def reset_stats(self) -> None:
        """Reset statistics counters"""
        self._validation_stats = {
            "checks_performed": 0,
            "issues_detected": 0,
            "issues_fixed": 0,
            "issue_types": {}
        }
    
    def get_correction_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for correcting data issues
        
        Args:
            results: Results from integrity check
            
        Returns:
            List[str]: List of correction recommendations
        """
        recommendations = []
        
        # Missing values recommendations
        if results.get("missing_values", {}).get("passed") is False:
            recommendations.append("Fill missing values using forward fill or interpolation")
        
        # Duplicate timestamps recommendations
        if results.get("duplicate_timestamps", {}).get("passed") is False:
            recommendations.append("Remove or aggregate duplicate timestamp entries")
        
        # Timestamp sequence recommendations
        if results.get("timestamp_sequence", {}).get("passed") is False:
            recommendations.append("Sort data by timestamp in ascending order")
        
        # Price anomalies recommendations
        if results.get("price_anomalies", {}).get("passed") is False:
            recommendations.append("Investigate and filter extreme price values")
        
        # Volume anomalies recommendations
        if results.get("volume_anomalies", {}).get("passed") is False:
            issues = results.get("volume_anomalies", {}).get("issues", {})
            if "negative" in issues:
                recommendations.append("Correct negative volume values")
            if "extreme" in issues:
                recommendations.append("Verify extreme volume spikes")
        
        # Zero values recommendations
        if results.get("zero_values", {}).get("passed") is False:
            recommendations.append("Replace zero prices with previous valid values")
        
        # OHLC consistency recommendations
        consistency_key = "ohlc_consistency" if "ohlc_consistency" in results else "consistency"
        if results.get(consistency_key, {}).get("passed") is False:
            recommendations.append("Fix OHLC price inconsistencies (high should be highest, low should be lowest)")
        
        return recommendations
    
    def get_fix_recommendations(self, results: Dict[str, Any], detailed: bool = False) -> Dict[str, Any]:
        """
        Generate detailed recommendations for fixing data issues
        
        Args:
            results: Results from integrity check
            detailed: Whether to include detailed fix instructions
            
        Returns:
            Dict: Fix recommendations with specific steps for each issue type
        """
        recommendations = {}
        
        # Missing values recommendations
        if results.get("missing_values", {}).get("passed") is False:
            missing_counts = results.get("missing_values", {}).get("missing_counts", {})
            total_missing = sum(missing_counts.values()) if missing_counts else 0
            
            recommendations["missing_values"] = {
                "description": f"Found {total_missing} missing values in dataset",
                "fix_strategy": "forward_fill_then_back",
                "affected_columns": list(missing_counts.keys()) if missing_counts else [],
                "severity": "medium" if total_missing < len(results.get("missing_values", {}).get("data", [])) * 0.05 else "high"
            }
        
        # Duplicate timestamps recommendations
        if results.get("duplicate_timestamps", {}).get("passed") is False:
            duplicate_count = results.get("duplicate_timestamps", {}).get("duplicate_count", 0)
            
            recommendations["duplicate_timestamps"] = {
                "description": f"Found {duplicate_count} duplicate timestamps",
                "fix_strategy": "keep_last",
                "severity": "high"
            }
        
        # Timestamp sequence recommendations
        if results.get("timestamp_sequence", {}).get("passed") is False:
            recommendations["timestamp_sequence"] = {
                "description": "Timestamps not in ascending order",
                "fix_strategy": "sort_ascending",
                "severity": "high"
            }
        
        # Price anomalies recommendations
        if results.get("price_anomalies", {}).get("passed") is False:
            anomalies = results.get("price_anomalies", {}).get("anomalies", {})
            total_anomalies = sum(info.get("count", 0) for info in anomalies.values())
            
            recommendations["price_anomalies"] = {
                "description": f"Found {total_anomalies} price anomalies",
                "fix_strategy": "interpolate_outliers",
                "affected_columns": list(anomalies.keys()),
                "severity": "medium"
            }
        
        # Zero values recommendations
        if results.get("zero_values", {}).get("passed") is False:
            zero_counts = results.get("zero_values", {}).get("zero_counts", {})
            
            recommendations["zero_values"] = {
                "description": f"Found zero values in columns: {', '.join(zero_counts.keys())}",
                "fix_strategy": "replace_with_adjacent",
                "affected_columns": list(zero_counts.keys()),
                "severity": "medium" if "close" in zero_counts or "open" in zero_counts else "low"
            }
        
        # OHLC consistency recommendations
        consistency_key = "ohlc_consistency" if "ohlc_consistency" in results else "consistency"
        if results.get(consistency_key, {}).get("passed") is False:
            inconsistencies = results.get(consistency_key, {}).get("inconsistencies", {})
            
            recommendations[consistency_key] = {
                "description": "OHLC price inconsistencies detected",
                "fix_strategy": "enforce_high_low_rules",
                "severity": "high",
                "inconsistency_types": list(inconsistencies.keys()) if inconsistencies else []
            }
        
        # If detailed instructions are requested, add code examples
        if detailed:
            for issue_type, rec in recommendations.items():
                if issue_type == "duplicate_timestamps":
                    rec["fix_code"] = "df = df.drop_duplicates(subset=['datetime'], keep='last')"
                elif issue_type == "timestamp_sequence":
                    rec["fix_code"] = "df = df.sort_values(by='datetime')"
                elif issue_type == "missing_values":
                    rec["fix_code"] = """# Forward fill then backward fill
for col in ['open', 'high', 'low', 'close']:
    if col in df.columns:
        df[col] = df[col].ffill().bfill()"""
                elif issue_type in ["ohlc_consistency", "consistency"]:
                    rec["fix_code"] = """# Use numpy for vectorized performance
high_array = np.maximum.reduce([df['high'].to_numpy(), df['open'].to_numpy(), df['close'].to_numpy()])
low_array = np.minimum.reduce([df['low'].to_numpy(), df['open'].to_numpy(), df['close'].to_numpy()])
df['high'] = high_array
df['low'] = low_array"""
                elif issue_type == "zero_values":
                    rec["fix_code"] = """# Replace zeros with NaN then fill
for col in ['open', 'high', 'low', 'close']:
    zero_mask = df[col] == 0
    if zero_mask.any():
        df.loc[zero_mask, col] = None
        df[col] = df[col].ffill().bfill()"""
                elif issue_type == "price_anomalies":
                    rec["fix_code"] = """# Identify and replace outliers with vectorized operations
for col in ['open', 'high', 'low', 'close']:
    values = df[col].to_numpy()
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - (3.0 * iqr)
    upper_bound = q3 + (3.0 * iqr)
    outlier_mask = (values < lower_bound) | (values > upper_bound)
    if np.any(outlier_mask):
        df.loc[outlier_mask, col] = None
        df[col] = df[col].interpolate(method='linear').ffill().bfill()"""
        
        return recommendations
    
    async def close(self):
        """Clean up resources"""
        try:
            self.thread_pool.shutdown(wait=False)
            self.logger.debug("Thread pool shutdown")
        except Exception as e:
            self.logger.error(f"Error shutting down thread pool: {e}")