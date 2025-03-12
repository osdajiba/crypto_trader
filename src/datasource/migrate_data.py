#!/usr/bin/env python
# src/data/migrate_data.py

import asyncio
import argparse
import logging
import os
import sys
import traceback
import pandas as pd
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_migration.log')
        ]
    )
    return LogManager.get_logger('data_migration')

async def migrate_legacy_files(data_path, symbol, timeframe, backup=True, delete_old=False):
    """Migrate timestamp-based files to daily files"""
    logger = LogManager.get_logger('data_migration')
    
    try:
        # Generate path for symbol/timeframe
        symbol_dir = os.path.join(data_path, timeframe, symbol.replace('/', '_'))
        
        if not os.path.exists(symbol_dir):
            logger.warning(f"Directory not found for {symbol} {timeframe}: {symbol_dir}")
            return False
            
        # Create a backup directory if needed
        backup_dir = os.path.join(symbol_dir, "legacy_backup")
        if backup:
            os.makedirs(backup_dir, exist_ok=True)
            
        logger.info(f"Scanning for legacy files in {symbol_dir}")
        
        # Find all legacy files (timestamp_timestamp.parquet format)
        legacy_files = []
        for root, _, files in os.walk(symbol_dir):
            for file in files:
                if file.endswith('.parquet'):
                    # Check if it's a legacy format (contains underscore in name)
                    filename = os.path.splitext(file)[0]
                    if '_' in filename and not filename.startswith(timeframe):
                        legacy_files.append(os.path.join(root, file))
        
        # Skip if no legacy files found
        if not legacy_files:
            logger.info(f"No legacy files found for {symbol} {timeframe}")
            return True
            
        logger.info(f"Found {len(legacy_files)} legacy files for {symbol} {timeframe}")
        
        # Process files in chunks to avoid memory issues
        chunk_size = 100
        file_chunks = [legacy_files[i:i+chunk_size] for i in range(0, len(legacy_files), chunk_size)]
        
        daily_data = {}  # Dictionary to hold data by day
        processed_count = 0
        
        # Process each chunk of files
        for chunk_idx, file_chunk in enumerate(file_chunks):
            logger.info(f"Processing chunk {chunk_idx+1}/{len(file_chunks)} ({len(file_chunk)} files)")
            
            # Read all files in this chunk
            for file_idx, file_path in enumerate(file_chunk):
                try:
                    # Read the file
                    df = pd.read_parquet(file_path)
                    
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
                    
                    # Group by day
                    df['date'] = df['datetime'].dt.date
                    for date, day_df in df.groupby('date'):
                        day_key = date.strftime('%Y-%m-%d')
                        if day_key not in daily_data:
                            daily_data[day_key] = []
                        daily_data[day_key].append(day_df)
                    
                    processed_count += 1
                    
                    # Move file to backup directory if requested
                    if backup:
                        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
                        shutil.copy2(file_path, backup_path)
                        
                    # Delete original file if requested
                    if delete_old:
                        os.remove(file_path)
                        
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
        
        logger.info(f"Successfully read {processed_count}/{len(legacy_files)} legacy files")
        
        # Save data in daily files
        saved_days = 0
        for day_key, day_dfs in daily_data.items():
            try:
                # Parse date components
                year, month, day = map(int, day_key.split('-'))
                
                # Create directory structure
                year_dir = os.path.join(symbol_dir, str(year))
                month_dir = os.path.join(year_dir, f"{month:02d}")
                os.makedirs(month_dir, exist_ok=True)
                
                # Create filename: 1h-2025-02-09.parquet
                filename = f"{timeframe}-{year}-{month:02d}-{day:02d}.parquet"
                file_path = os.path.join(month_dir, filename)
                
                # Combine all DataFrames for this day
                combined_df = pd.concat(day_dfs)
                
                # Drop the date column we added for grouping
                combined_df = combined_df.drop(columns=['date'])
                
                # Remove duplicates based on datetime
                combined_df = combined_df.drop_duplicates(subset=['datetime'], keep='last')
                
                # Sort by datetime
                combined_df = combined_df.sort_values('datetime')
                
                # Check if file already exists
                if os.path.exists(file_path):
                    # Read existing file
                    existing_df = pd.read_parquet(file_path)
                    
                    # Ensure datetime column is datetime type
                    if 'datetime' in existing_df.columns and not pd.api.types.is_datetime64_any_dtype(existing_df['datetime']):
                        existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                    
                    # Combine with existing data
                    final_df = pd.concat([existing_df, combined_df])
                    
                    # Remove duplicates
                    final_df = final_df.drop_duplicates(subset=['datetime'], keep='last')
                    
                    # Sort by datetime
                    final_df = final_df.sort_values('datetime')
                    
                    # Save to file
                    final_df.to_parquet(file_path, index=False)
                    logger.info(f"Updated existing file: {file_path} with {len(combined_df)} records")
                else:
                    # Save new file
                    combined_df.to_parquet(file_path, index=False)
                    logger.info(f"Created new file: {file_path} with {len(combined_df)} records")
                
                saved_days += 1
                
            except Exception as e:
                logger.error(f"Error saving daily file for {day_key}: {str(e)}")
        
        logger.info(f"Migration complete for {symbol} {timeframe}: saved data for {saved_days} days")
        
        # Clean up empty directories
        if delete_old:
            try:
                # Remove legacy backup if empty
                if os.path.exists(backup_dir) and len(os.listdir(backup_dir)) == 0:
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
    """Scan data directory and migrate all legacy files found"""
    logger = LogManager.get_logger('data_migration')
    
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
            logger.info(f"Migrating specific symbol/timeframe: {symbol} {timeframe}")
            success = await migrate_legacy_files(data_path, symbol, timeframe, backup, delete_old)
            if success:
                logger.info(f"Successfully migrated {symbol} {timeframe}")
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
                    
                logger.info(f"Migrating {sym} {tf}...")
                success = await migrate_legacy_files(data_path, sym, tf, backup, delete_old)
                
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

def main():
    """Parse arguments and run the migration"""
    parser = argparse.ArgumentParser(description='Migrate market data from legacy format to daily files')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--symbol', type=str, help='Specific symbol to migrate (e.g., BTC/USDT)')
    parser.add_argument('--timeframe', type=str, help='Specific timeframe to migrate (e.g., 1h)')
    parser.add_argument('--no-backup', action='store_true', help='Do not create backups of legacy files')
    parser.add_argument('--delete-old', action='store_true', help='Delete legacy files after migration')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    try:
        start_time = datetime.now()
        logger.info(f"Starting migration at {start_time}")
        
        asyncio.run(scan_and_migrate(
            args.config,
            args.symbol,
            args.timeframe,
            not args.no_backup,
            args.delete_old
        ))
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Migration completed at {end_time}. Total duration: {duration}")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Migration interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())