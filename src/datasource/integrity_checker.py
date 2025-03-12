# src/data/integrity_checker.py

import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Coroutine
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor


class DataIntegrityChecker:
    """
    Enhanced Data Integrity Checker
    
    Checks for common data issues:
    - Missing values
    - Duplicate timestamps
    - Price anomalies
    - Out-of-sequence data
    - Volume anomalies
    """
    
    def __init__(self, timeframe: str = "1h", iqr_multiplier: float = 3.0):
        """
        Initialize the data integrity checker
        
        Args:
            timeframe: Data timeframe for context-appropriate checks
            iqr_multiplier: Multiplier for IQR-based outlier detection
        """
        self.logger = LogManager.get_logger("data.integrity")
        self.timeframe = timeframe
        self.iqr_multiplier = iqr_multiplier
        self.executor = AsyncExecutor()
        self.logger.info(f"DataIntegrityChecker initialized with timeframe={timeframe}, iqr_multiplier={iqr_multiplier}")
    
    async def check(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Run all integrity checks on the dataframe
        
        Args:
            df: DataFrame to check
            
        Returns:
            Tuple[bool, Dict]: (passed, results) where results contains details of checks
        """
        try:
            if df.empty:
                self.logger.warning("Empty dataframe provided for integrity check")
                return False, {"error": "Empty dataframe"}
            
            # Create async task to perform checks
            results = await self.executor.submit(self._perform_checks, df)
            
            # Determine overall pass/fail
            passed = all(result.get("passed", False) for result in results.values())
            return passed, results
            
        except Exception as e:
            self.logger.error(f"Data integrity check failed: {e}", exc_info=True)
            return False, {"error": str(e)}
    
    async def _perform_checks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform all integrity checks asynchronously
        
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
        
        # Define check coroutines
        check_tasks = {
            "missing_values": self._check_missing_values(df_copy),
            "duplicate_timestamps": self._check_duplicate_timestamps(df_copy),
            "timestamp_sequence": self._check_timestamp_sequence(df_copy),
            "price_anomalies": self._check_price_anomalies(df_copy),
            "volume_anomalies": self._check_volume_anomalies(df_copy),
            "zero_values": self._check_zero_values(df_copy),
            "consistency": self._check_ohlc_consistency(df_copy)
        }
        
        # Run checks concurrently
        results = {}
        for check_name, check_coro in check_tasks.items():
            try:
                results[check_name] = await check_coro
            except Exception as e:
                self.logger.error(f"Error in {check_name} check: {str(e)}")
                results[check_name] = {"passed": False, "error": str(e)}
                
        return results
    
    async def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for missing values in critical columns"""
        critical_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in critical_columns if col in df.columns]
        
        if not available_columns:
            return {"passed": False, "error": "No OHLC columns found"}
            
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
        """Check for duplicate timestamps"""
        timestamp_col = self._identify_timestamp_column(df)
        
        if timestamp_col is None:
            return {"passed": False, "error": "No timestamp column found"}
            
        duplicates = df[timestamp_col].duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            duplicate_timestamps = df.loc[duplicates, timestamp_col].tolist()
            return {
                "passed": False,
                "duplicate_count": duplicate_count,
                "duplicate_timestamps": duplicate_timestamps[:10],  # Limit to first 10
                "message": f"Found {duplicate_count} duplicate timestamps"
            }
            
        return {"passed": True, "duplicate_count": 0}
    
    async def _check_timestamp_sequence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check if timestamps are in ascending order"""
        timestamp_col = self._identify_timestamp_column(df)
        
        if timestamp_col is None:
            return {"passed": False, "error": "No timestamp column found"}
        
        # Sort a copy to compare
        df_sorted = df.sort_values(by=timestamp_col)
        
        if not df[timestamp_col].equals(df_sorted[timestamp_col]):
            self.logger.warning("Timestamps not in ascending order")
            return {
                "passed": False,
                "message": "Timestamps not in ascending order",
                "suggestion": "Data should be sorted by timestamp"
            }
        
        return {"passed": True}
    
    async def _check_price_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect price anomalies using IQR method"""
        price_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if not price_cols:
            return {"passed": False, "error": "No price columns found"}
            
        anomalies = {}
        
        for col in price_cols:
            # Calculate IQR
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            
            # Define bounds
            lower_bound = q1 - (self.iqr_multiplier * iqr)
            upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            # Find outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            if not outliers.empty:
                anomalies[col] = {
                    "count": len(outliers),
                    "percentage": len(outliers) / len(df) * 100,
                    "bounds": (lower_bound, upper_bound),
                    "examples": outliers[col].head(5).tolist() if len(outliers) > 0 else []
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
        """Check for volume anomalies"""
        if 'volume' not in df.columns:
            return {"passed": True, "message": "No volume column to check"}
            
        # Check for negative volumes
        negative_volumes = df[df['volume'] < 0]
        
        # Check for extreme volumes using IQR
        q1 = df['volume'].quantile(0.25)
        q3 = df['volume'].quantile(0.75)
        iqr = q3 - q1
        
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        extreme_volumes = df[df['volume'] > upper_bound]
        
        issues = {}
        if not negative_volumes.empty:
            issues["negative"] = {
                "count": len(negative_volumes),
                "examples": negative_volumes['volume'].head(5).tolist()
            }
            
        if not extreme_volumes.empty:
            issues["extreme"] = {
                "count": len(extreme_volumes),
                "percentage": len(extreme_volumes) / len(df) * 100,
                "threshold": upper_bound,
                "examples": extreme_volumes['volume'].head(5).tolist()
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
        """Check for zero values in critical columns"""
        check_columns = [col for col in ['close', 'volume'] if col in df.columns]
        
        if not check_columns:
            return {"passed": True, "message": "No columns to check for zeros"}
            
        zero_counts = {}
        for col in check_columns:
            zero_count = (df[col] == 0).sum()
            if zero_count > 0:
                zero_counts[col] = {
                    "count": zero_count,
                    "percentage": zero_count / len(df) * 100
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
        """Check for OHLC price consistency"""
        ohlc_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df.columns]
        
        if len(ohlc_cols) < 4:
            return {"passed": True, "message": "Not enough OHLC columns for consistency check"}
            
        # Check: high >= open, high >= close, high >= low, low <= open, low <= close
        inconsistencies = {
            "high_vs_low": df[df['high'] < df['low']].index.tolist(),
            "high_vs_open": df[df['high'] < df['open']].index.tolist(),
            "high_vs_close": df[df['high'] < df['close']].index.tolist(),
            "low_vs_open": df[df['low'] > df['open']].index.tolist(),
            "low_vs_close": df[df['low'] > df['close']].index.tolist()
        }
        
        # Filter out empty lists
        inconsistencies = {k: v for k, v in inconsistencies.items() if v}
        
        if inconsistencies:
            total_rows = sum(len(indices) for indices in inconsistencies.values())
            return {
                "passed": False,
                "inconsistencies": inconsistencies,
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
        if results.get("ohlc_consistency", {}).get("passed") is False:
            recommendations.append("Fix OHLC price inconsistencies (high should be highest, low should be lowest)")
            
        return recommendations