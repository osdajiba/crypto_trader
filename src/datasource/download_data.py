#!/usr/bin/env python
# src/data/migrate_to_timestamp.py

import asyncio
import os
import sys
import traceback
import pandas as pd
import shutil
from datetime import datetime, timedelta

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from .data_manager import DataManager


def setup_logging():
    """
    Set up logging for data download operations.
    Creates necessary directory structure for logs.
    
    Returns:
        logging.Logger: Configured logger for data download
    """
    import logging
    import os
    from pathlib import Path
    
    # Define log paths
    log_dir = Path('logs/data')
    log_file = log_dir / 'data_download.log'
    
    # Create the directory structure if it doesn't exist
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logger
    logger = LogManager.get_logger('data_download')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

async def download_data(data_manager, symbol, timeframe, start_date, end_date):
    """Download and store data for a specific symbol and timeframe using timestamp format"""
    logger = LogManager.get_logger('data_download')
    logger.info(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}...")
    
    try:
        # Use exchange source to fetch data with pagination
        source = data_manager.backup_source if data_manager.backup_source else data_manager.primary_source
        
        if not source:
            logger.error(f"No data source available for downloading {symbol} {timeframe}")
            return False
            
        # Fetch data from exchange with pagination
        data = await data_manager._fetch_paginated_data(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            start_dt=start_date,
            end_dt=end_date
        )
        
        if data.empty:
            logger.error(f"Failed to download data for {symbol} {timeframe}")
            return False
            
        # Validate the data
        if data_manager.validate_data and data_manager.integrity_checker:
            logger.info(f"Validating data integrity for {symbol} {timeframe}")
            valid, results = await data_manager.integrity_checker.check(data)
            if not valid:
                logger.warning(f"Data integrity issues found for {symbol} {timeframe}")
                data = await data_manager._fix_data_issues(data, results)
                logger.info(f"Data issues fixed for {symbol} {timeframe}")
        
        # Ensure 'datetime' column exists and is datetime type
        if 'datetime' not in data.columns:
            if 'timestamp' in data.columns:
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                logger.error(f"Cannot store data: no datetime or timestamp column in data")
                return False
        
        # Convert datetime column to pandas datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(data['datetime']):
            data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Store data using timestamp file format
        logger.info(f"Storing {len(data)} records for {symbol} {timeframe} in timestamp format")
        
        # Prepare base directory
        data_path = data_manager.config.get('data', 'paths', 'historical_data_path', default='data/historical')
        base_dir = os.path.join(
            data_path,
            timeframe,
            symbol.replace('/', '_')
        )
        
        # Get time range of the data
        min_time = data['datetime'].min()
        max_time = data['datetime'].max()
        
        # Convert to Unix timestamps for filename
        start_ts = int(min_time.timestamp())
        end_ts = int(max_time.timestamp())
        
        # Create year/month directory structure
        year = min_time.year
        month = min_time.month
        
        year_dir = os.path.join(base_dir, str(year))
        month_dir = os.path.join(year_dir, f"{month:02d}")
        
        # Create directories if they don't exist
        os.makedirs(month_dir, exist_ok=True)
        
        # Create filename: 1672531200_1672617599.parquet
        filename = f"{start_ts}_{end_ts}.parquet"
        file_path = os.path.join(month_dir, filename)
        
        # Save data to parquet file
        data.to_parquet(file_path, index=False)
        logger.info(f"Successfully saved {len(data)} records to {file_path}")
        
        return True
            
    except Exception as e:
        logger.error(f"Error downloading {symbol} {timeframe}: {str(e)}")
        traceback.print_exc()
        return False

async def run_downloads(config_path, symbols, timeframes, start_date, end_date, max_concurrent=3):
    """Manage downloading data for multiple symbols and timeframes"""
    logger = setup_logging()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = ConfigManager(config_path)
        
        # Initialize logging
        LogManager(config)
        
        # Create data manager in hybrid mode to ensure we can download and store
        logger.info("Initializing data manager in hybrid mode")
        data_manager = DataManager(source_type="hybrid", config=config)
        
        # Parse dates
        start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        
        # Default start date is 30 days ago if not specified
        if not start_dt:
            start_dt = end_dt - timedelta(days=30)
            
        # Create download tasks for each symbol and timeframe
        all_tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                all_tasks.append((symbol, timeframe))
        
        logger.info(f"Prepared {len(all_tasks)} download tasks")
        
        # Process tasks in batches to control concurrency
        results = {"success": 0, "failed": 0}
        for i in range(0, len(all_tasks), max_concurrent):
            batch = all_tasks[i:i+max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}/{(len(all_tasks) + max_concurrent - 1)//max_concurrent} ({len(batch)} tasks)")
            
            tasks = []
            for symbol, timeframe in batch:
                tasks.append(download_data(data_manager, symbol, timeframe, start_dt, end_dt))
                
            batch_results = await asyncio.gather(*tasks)
            
            # Count successes and failures
            for success in batch_results:
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
        
        # Close data manager
        logger.info("Shutting down data manager")
        await data_manager.close()
        
        logger.info(f"Download completed. Summary: {results['success']} succeeded, {results['failed']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        traceback.print_exc()
        return {"success": 0, "failed": len(all_tasks) if 'all_tasks' in locals() else 0}

async def migrate_to_timestamp_format(data_path, symbol, timeframe, backup=True, delete_old=False):
    """Migrate daily and legacy files to timestamp-based parquet format"""
    logger = LogManager.get_logger('migrate_timestamp')
    
    try:
        # Generate path for symbol/timeframe using new structure
        symbol_dir = os.path.join(data_path, timeframe, symbol.replace('/', '_'))
        
        if not os.path.exists(symbol_dir):
            logger.warning(f"Directory not found for {symbol} {timeframe}: {symbol_dir}")
            return False
            
        # Create a backup directory if needed
        backup_dir = os.path.join(symbol_dir, "pre_timestamp_backup")
        if backup:
            os.makedirs(backup_dir, exist_ok=True)
            
        logger.info(f"Scanning for files to migrate in {symbol_dir}")
        
        # Find all files to migrate (daily format and legacy format)
        files_to_migrate = []
        
        # First, check for daily files and any non-timestamp format files
        for root, _, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith('.parquet') or file.endswith('.csv'):
                    # Skip if already in timestamp format (contains underscore in filename)
                    filename = os.path.splitext(file)[0]
                    if '_' in filename and filename.split('_')[0].isdigit() and filename.split('_')[1].isdigit():
                        # Already in timestamp format
                        continue
                    
                    # Add to migration list
                    files_to_migrate.append(os.path.join(root, file))
        
        # Skip if no files to migrate
        if not files_to_migrate:
            logger.info(f"No files found to migrate for {symbol} {timeframe}")
            return True
            
        logger.info(f"Found {len(files_to_migrate)} files to migrate for {symbol} {timeframe}")
        
        # Process files in chunks to avoid memory issues
        chunk_size = 100
        file_chunks = [files_to_migrate[i:i+chunk_size] for i in range(0, len(files_to_migrate), chunk_size)]
        
        migrated_count = 0
        
        # Process each chunk of files
        for chunk_idx, file_chunk in enumerate(file_chunks):
            logger.info(f"Processing chunk {chunk_idx+1}/{len(file_chunks)} ({len(file_chunk)} files)")
            
            # Read all files in this chunk
            for file_idx, file_path in enumerate(file_chunk):
                try:
                    # Read the file based on extension
                    if file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    elif file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        logger.warning(f"Unsupported file format: {file_path}")
                        continue
                    
                    if df.empty:
                        logger.warning(f"Empty file: {file_path}")
                        continue
                    
                    # Ensure datetime column exists
                    if 'datetime' not in df.columns:
                        if 'timestamp' in df.columns:
                            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                        else:
                            logger.warning(f"No datetime or timestamp column in file: {file_path}")
                            continue
                    
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                        df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    # Get time range of data
                    min_time = df['datetime'].min()
                    max_time = df['datetime'].max()
                    
                    # Convert to Unix timestamps for filename
                    start_ts = int(min_time.timestamp())
                    end_ts = int(max_time.timestamp())
                    
                    # Create year/month directory structure
                    year = min_time.year
                    month = min_time.month
                    
                    year_dir = os.path.join(symbol_dir, str(year))
                    month_dir = os.path.join(year_dir, f"{month:02d}")
                    
                    # Create directories if they don't exist
                    os.makedirs(month_dir, exist_ok=True)
                    
                    # Create filename: 1677715200_1677801599.parquet
                    new_filename = f"{start_ts}_{end_ts}.parquet"
                    new_file_path = os.path.join(month_dir, new_filename)
                    
                    # Check if new file already exists
                    if os.path.exists(new_file_path):
                        # Merge with existing file
                        existing_df = pd.read_parquet(new_file_path)
                        
                        # Ensure datetime column is datetime type
                        if 'datetime' in existing_df.columns and not pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                            existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                        
                        # Combine with existing data
                        combined_df = pd.concat([existing_df, df])
                        
                        # Remove duplicates
                        combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                        
                        # Sort by datetime
                        combined_df = combined_df.sort_values('datetime')
                        
                        # Save to new file
                        combined_df.to_parquet(new_file_path, index=False)
                        logger.info(f"Updated existing timestamp file: {new_file_path} with {len(df)} records")
                    else:
                        # Sort by datetime before saving
                        df = df.sort_values('datetime')
                        
                        # Save to new file
                        df.to_parquet(new_file_path, index=False)
                        logger.info(f"Created new timestamp file: {new_file_path} with {len(df)} records")
                    
                    # Move original file to backup directory if requested
                    if backup:
                        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, backup_path)
                        
                    # Delete original file if requested
                    if delete_old:
                        os.remove(file_path)
                        
                    migrated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    traceback.print_exc()
        
        logger.info(f"Successfully migrated {migrated_count}/{len(files_to_migrate)} files for {symbol} {timeframe}")
        
        # Clean up empty directories
        if delete_old:
            try:
                # Remove backup if empty
                if os.path.exists(backup_dir) and not os.listdir(backup_dir):
                    os.rmdir(backup_dir)
                    logger.info(f"Removed empty backup directory: {backup_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up directories: {str(e)}")
                
        return True
        
    except Exception as e:
        logger.error(f"Migration failed for {symbol} {timeframe}: {str(e)}")
        traceback.print_exc()
        return False

async def scan_and_migrate(config_path, symbol=None, timeframe=None, backup=True, delete_old=False):
    """Scan data directory and migrate all files to timestamp format"""
    logger = LogManager.get_logger('migrate_timestamp')
    
    try:
        # Load configuration
        config = ConfigManager(config_path)
        logger.info(f"Loaded configuration from {config_path}")
        
        # Initialize logging
        LogManager(config)
        
        # Get data path from config
        data_path = config.get('data', 'paths', 'historical_data_path', default='data/historical')
        logger.info(f"Using data path: {data_path}")
        
        if not os.path.exists(data_path):
            logger.error(f"Data path does not exist: {data_path}")
            return
            
        # If specific symbol and timeframe provided, just migrate those
        if symbol and timeframe:
            logger.info(f"Migrating specific symbol/timeframe to timestamp format: {symbol} {timeframe}")
            success = await migrate_to_timestamp_format(data_path, symbol, timeframe, backup, delete_old)
            if success:
                logger.info(f"Successfully migrated {symbol} {timeframe} to timestamp format")
            else:
                logger.error(f"Failed to migrate {symbol} {timeframe}")
            return
            
        # Otherwise, scan and migrate all
        # Get all timeframes (subdirectories of data_path)
        timeframes = []
        for item in os.listdir(data_path):
            if os.path.isdir(os.path.join(data_path, item)):
                timeframes.append(item)
                
        if not timeframes:
            logger.error(f"No timeframe directories found in {data_path}")
            return
            
        logger.info(f"Found {len(timeframes)} timeframes: {', '.join(timeframes)}")
        
        # For each timeframe, get all symbols
        total_success = 0
        total_failure = 0
        
        for tf in timeframes:
            if timeframe and tf != timeframe:
                continue  # Skip if not the requested timeframe
                
            tf_path = os.path.join(data_path, tf)
            
            # Get all symbol directories
            symbols = []
            for item in os.listdir(tf_path):
                if os.path.isdir(os.path.join(tf_path, item)):
                    symbols.append(item.replace('_', '/'))
                    
            logger.info(f"Found {len(symbols)} symbols for timeframe {tf}")
            
            # Migrate each symbol
            for sym in symbols:
                if symbol and sym != symbol:
                    continue  # Skip if not the requested symbol
                    
                logger.info(f"Migrating {sym} {tf} to timestamp format...")
                success = await migrate_to_timestamp_format(data_path, sym, tf, backup, delete_old)
                
                if success:
                    total_success += 1
                    logger.info(f"Successfully migrated {sym} {tf}")
                else:
                    total_failure += 1
                    logger.error(f"Failed to migrate {sym} {tf}")
        
        logger.info(f"Migration summary: {total_success} successful, {total_failure} failed")
        
    except Exception as e:
        logger.error(f"Error during migration scan: {str(e)}")
        traceback.print_exc()