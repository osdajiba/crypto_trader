#!/usr/bin/env python
# src/data/download_data.py

import asyncio
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta

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
    """Download and store data for a specific symbol and timeframe"""
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
        
        # Store data using daily file format
        logger.info(f"Storing {len(data)} records for {symbol} {timeframe}")
        success = await data_manager.update_local_data(symbol, timeframe, data)
        
        if success:
            logger.info(f"Successfully downloaded and stored {len(data)} records for {symbol} {timeframe}")
            return True
        else:
            logger.error(f"Failed to store downloaded data for {symbol} {timeframe}")
            return False
            
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
