# src/utils/file_utils.py
# Consolidated file utility functions

import os
import pandas as pd
import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime
import traceback

from src.common.log_manager import LogManager

logger = LogManager.get_logger("utils.file")

class ParquetFileManager:
    """Unified class for handling timestamp-based Parquet files"""
    
    @staticmethod
    def get_directory_path(base_path: str, timeframe: str, symbol: str, 
                         timestamp: datetime) -> str:
        """
        Get directory path for a specific timestamp
        
        Args:
            base_path: Base data path
            timeframe: Timeframe (e.g., '1h')
            symbol: Symbol (e.g., 'BTC/USDT')
            timestamp: Datetime
            
        Returns:
            str: Directory path for the timestamp
        """
        # Convert symbol format
        safe_symbol = symbol.replace('/', '_')
        
        # Extract year and month
        year = timestamp.year
        month = timestamp.month
        
        # Build path
        return os.path.join(
            base_path,
            timeframe,
            safe_symbol,
            str(year),
            f"{month:02d}"
        )
    
    @staticmethod
    def ensure_directory_exists(path: str) -> None:
        """
        Ensure a directory exists
        
        Args:
            path: Directory path
        """
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def generate_filename(start_time: datetime, end_time: datetime) -> str:
        """
        Generate timestamp-based filename
        
        Args:
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            str: Filename in format 'start_timestamp_end_timestamp.parquet'
        """
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())
        
        return f"{start_ts}_{end_ts}.parquet"
    
    @staticmethod
    def parse_filename_timestamps(filename: str) -> tuple:
        """
        Parse timestamps from filename
        
        Args:
            filename: Filename in format 'start_timestamp_end_timestamp.parquet'
            
        Returns:
            tuple: (start_timestamp, end_timestamp)
        """
        try:
            # Remove extension
            basename = os.path.splitext(filename)[0]
            
            # Split by underscore
            parts = basename.split('_')
            
            # Parse timestamps
            start_ts = int(parts[0])
            end_ts = int(parts[1])
            
            return start_ts, end_ts
        except (ValueError, IndexError):
            return None, None
    
    @staticmethod
    async def read_parquet(file_path: str) -> pd.DataFrame:
        """
        Read a Parquet file asynchronously
        
        Args:
            file_path: Path to Parquet file
            
        Returns:
            DataFrame: Data from file
        """
        try:
            # This is wrapped in a loop to make it non-blocking
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, pd.read_parquet, file_path)
            
            # Process DataFrame - ensure datetime column is present
            if 'timestamp' in df.columns and 'datetime' not in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert datetime column to datetime type if needed
            if 'datetime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                df['datetime'] = pd.to_datetime(df['datetime'])
                
            return df
        except Exception as e:
            logger.error(f"Error reading Parquet file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    async def write_parquet(df: pd.DataFrame, file_path: str) -> bool:
        """
        Write DataFrame to Parquet file asynchronously
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure datetime column exists and is properly formatted
            if 'datetime' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                elif 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure parent directory exists
            parent_dir = os.path.dirname(file_path)
            os.makedirs(parent_dir, exist_ok=True)
            
            # Write file
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, df.to_parquet, file_path, False)
            
            return True
        except Exception as e:
            logger.error(f"Error writing Parquet file {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def find_files_in_date_range(
        base_dir: str, 
        timeframe: str,
        symbol: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[str]:
        """
        Find Parquet files in a date range
        
        Args:
            base_dir: Base directory
            timeframe: Timeframe (e.g., '1h')
            symbol: Symbol (e.g., 'BTC/USDT')
            start_dt: Start datetime
            end_dt: End datetime
            
        Returns:
            List[str]: List of file paths
        """
        symbol_dir = os.path.join(base_dir, timeframe, symbol.replace('/', '_'))
        if not os.path.exists(symbol_dir):
            return []
            
        # Convert to Unix timestamps
        start_ts = int(start_dt.timestamp())
        end_ts = int(end_dt.timestamp())
        
        matched_files = []
        
        # Get year range to search
        start_year = start_dt.year
        end_year = end_dt.year
        
        # Iterate through years and months
        for year in range(start_year, end_year + 1):
            year_dir = os.path.join(symbol_dir, str(year))
            if not os.path.exists(year_dir):
                continue
                
            # Get month range
            start_month = 1
            end_month = 12
            
            if year == start_year:
                start_month = start_dt.month
            if year == end_year:
                end_month = end_dt.month
                
            for month in range(start_month, end_month + 1):
                month_dir = os.path.join(year_dir, f"{month:02d}")
                if not os.path.exists(month_dir):
                    continue
                    
                # Check each file in the month directory
                for file in os.listdir(month_dir):
                    if not file.endswith('.parquet'):
                        continue
                        
                    # Check if file overlaps with date range
                    file_start_ts, file_end_ts = ParquetFileManager.parse_filename_timestamps(file)
                    
                    if file_start_ts is not None and file_end_ts is not None:
                        if file_end_ts >= start_ts and file_start_ts <= end_ts:
                            matched_files.append(os.path.join(month_dir, file))
        
        return matched_files
    
    @staticmethod
    async def batch_process_files(
        file_paths: List[str],
        processor_func: Callable,
        batch_size: int = 10
    ) -> List[Any]:
        """
        Process files in batches
        
        Args:
            file_paths: List of file paths
            processor_func: Function to process each file
            batch_size: Batch size
            
        Returns:
            List[Any]: Processing results
        """
        results = []
        
        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            
            # Process batch concurrently
            tasks = [processor_func(f) for f in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            for result in batch_results:
                if not isinstance(result, Exception) and result is not None:
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        results.append(result)
        
        return results
    
    @staticmethod
    async def load_and_combine_files(
        file_paths: List[str],
        date_filter: Optional[tuple] = None,
        drop_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Load and combine multiple Parquet files
        
        Args:
            file_paths: List of file paths
            date_filter: Optional (start_dt, end_dt) for filtering
            drop_duplicates: Whether to drop duplicate timestamps
            
        Returns:
            DataFrame: Combined data
        """
        if not file_paths:
            return pd.DataFrame()
            
        # Load all files
        dfs = await ParquetFileManager.batch_process_files(
            file_paths,
            ParquetFileManager.read_parquet
        )
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure datetime column exists
        if 'datetime' not in combined_df.columns and 'timestamp' in combined_df.columns:
            combined_df['datetime'] = pd.to_datetime(combined_df['timestamp'], unit='ms')
            
        # Apply date filter if specified
        if date_filter and 'datetime' in combined_df.columns:
            start_dt, end_dt = date_filter
            combined_df = combined_df[
                (combined_df['datetime'] >= pd.Timestamp(start_dt)) &
                (combined_df['datetime'] <= pd.Timestamp(end_dt))
            ]
            
        # Sort by datetime
        if 'datetime' in combined_df.columns:
            combined_df = combined_df.sort_values('datetime')
            
            # Drop duplicates if requested
            if drop_duplicates:
                combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
        
        return combined_df
    
    @staticmethod
    async def save_dataframe(
        df: pd.DataFrame,
        base_path: str,
        timeframe: str,
        symbol: str,
        folder_suffix: str = ""
    ) -> bool:
        """
        Save DataFrame to Parquet file in timestamp-based format
        
        Args:
            df: DataFrame to save
            base_path: Base directory
            timeframe: Timeframe
            symbol: Symbol
            folder_suffix: Optional suffix for folder name
            
        Returns:
            bool: Success status
        """
        if df.empty:
            logger.warning("Cannot save empty DataFrame")
            return False
            
        try:
            # Ensure datetime column exists
            if 'datetime' not in df.columns:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                elif 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                else:
                    logger.error("Cannot save DataFrame: no datetime column")
                    return False
                    
            # Get time range
            min_time = df['datetime'].min()
            max_time = df['datetime'].max()
            
            # Build folder path with optional suffix
            tf_folder = f"{timeframe}{folder_suffix}"
            folder_path = ParquetFileManager.get_directory_path(
                base_path, tf_folder, symbol, min_time
            )
            
            # Ensure directory exists
            ParquetFileManager.ensure_directory_exists(folder_path)
            
            # Generate filename
            filename = ParquetFileManager.generate_filename(min_time, max_time)
            file_path = os.path.join(folder_path, filename)
            
            # Save DataFrame
            return await ParquetFileManager.write_parquet(df, file_path)
            
        except Exception as e:
            logger.error(f"Error saving DataFrame: {str(e)}")
            return False
    
    @staticmethod
    async def update_or_merge_file(
        new_data: pd.DataFrame,
        file_path: str
    ) -> bool:
        """
        Update existing file or create new one
        
        Args:
            new_data: New data to save
            file_path: File path
            
        Returns:
            bool: Success status
        """
        if new_data.empty:
            return False
            
        try:
            if os.path.exists(file_path):
                # Read existing file
                existing_df = await ParquetFileManager.read_parquet(file_path)
                
                if not existing_df.empty:
                    # Combine with new data
                    combined_df = pd.concat([existing_df, new_data], ignore_index=True)
                    
                    # Remove duplicates
                    if 'datetime' in combined_df.columns:
                        combined_df = combined_df.sort_values('datetime')
                        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                        
                    # Save combined data
                    return await ParquetFileManager.write_parquet(combined_df, file_path)
            
            # If file doesn't exist or reading failed, just save new data
            return await ParquetFileManager.write_parquet(new_data, file_path)
            
        except Exception as e:
            logger.error(f"Error updating file {file_path}: {str(e)}")
            return False