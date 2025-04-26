# src/data/download_data.py

import asyncio
import os
import sys
import traceback
from pathlib import Path
import re
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional, Union
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from common.config import ConfigManager
from common.logging import LogManager
from src.datasource.manager import DataManager
from datasource.datasources import LocalSource


def setup_logging():
    """
    Set up logging for data download operations.
    Creates necessary directory structure for logs.
    
    Returns:
        logging.Logger: Configured logger for data download
    """
    # Define log paths
    log_dir = 'logs/data'
    log_file = f"{log_dir}/data_download.log"
    
    # Create the directory structure if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure logger
    logger = LogManager.get_logger('data_download')
    
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

async def run_downloads(config_path: str, symbols: List[str], timeframes: List[str], 
                      start_date: str, end_date: str, max_concurrent: int = 3) -> Dict[str, int]:
    """
    Download data for multiple symbols and timeframes with data validation
    
    Args:
        config_path: Path to configuration file
        symbols: List of symbols to download
        timeframes: List of timeframes to download
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_concurrent: Maximum concurrent downloads
        validate_data: Whether to validate and fix data before storage
        
    Returns:
        Dict: Results summary
    """
    logger = setup_logging()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = ConfigManager(config_path)
        
        # Create data manager in hybrid mode to ensure we can download and store
        logger.info("Initializing data manager in hybrid mode")
        
        data_manager = DataManager(source_type="hybrid", config=config)
        
        # Parse dates
        from src.common.helpers import TimeUtils
        start_dt = TimeUtils.parse_timestamp(start_date) if start_date else None
        end_dt = TimeUtils.parse_timestamp(end_date) if end_date else datetime.now()
        
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
        results = {"success": 0, "failed": 0, "validated": 0, "fixed": 0}
        for i in range(0, len(all_tasks), max_concurrent):
            batch = all_tasks[i:i+max_concurrent]
            logger.info(f"Processing batch {i//max_concurrent + 1}/{(len(all_tasks) + max_concurrent - 1)//max_concurrent} ({len(batch)} tasks)")
            
            tasks = []
            for symbol, timeframe in batch:
                tasks.append(data_manager.download_and_store_data(
                    symbol=symbol, 
                    timeframe=timeframe, 
                    start=start_dt, 
                    end=end_dt,
                    validate_before_storage=validate_data
                ))
                
            batch_results = await asyncio.gather(*tasks)
            
            # Count successes and failures
            for success in batch_results:
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
        
        # If validation was enabled, log data integrity status
        if validate_data and hasattr(data_manager.integrity_checker, '_validation_stats'):
            results["validated"] = data_manager.integrity_checker._validation_stats.get("checks_performed", 0)
            results["fixed"] = data_manager.integrity_checker._validation_stats.get("issues_fixed", 0)
            logger.info(f"Data validation stats: {results['validated']} datasets checked, {results['fixed']} datasets fixed")
        
        # Close data manager
        logger.info("Shutting down data manager")
        await data_manager.close()
        
        logger.info(f"Download completed. Summary: {results['success']} succeeded, {results['failed']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        traceback.print_exc()
        return {"success": 0, "failed": len(all_tasks) if 'all_tasks' in locals() else 0}

async def scan_and_migrate(config_path: str, symbol: Optional[str] = None, 
                         timeframe: Optional[str] = None, backup: bool = True, 
                         delete_old: bool = False) -> Dict[str, int]:
    """
    Scan for legacy data files and migrate them to the new format
    
    Args:
        config_path: Path to configuration file
        symbol: Symbol to migrate (None for all)
        timeframe: Timeframe to migrate (None for all)
        backup: Whether to create backups before migration
        delete_old: Whether to delete legacy files after migration
        
    Returns:
        Dict: Results summary
    """
    logger = setup_logging()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {config_path}")
        config = ConfigManager(config_path)
        
        # Create data manager for migration
        logger.info("Initializing data manager for migration")
        data_manager = DataManager(source_type="local", config=config)
        
        # Get data path from config
        data_path = config.get('data', 'paths', 'historical_data_path', default='data/historical')
        
        # Scan for legacy files (timestamp-based files)
        logger.info(f"Scanning for legacy files in {data_path}")
        
        legacy_files = []
        scanned_count = 0
        
        # Walk through the data directory
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    path_parts = Path(file_path).parts
                    
                    # Try to determine symbol and timeframe from path
                    file_symbol = None
                    file_timeframe = None
                    
                    # Search for timeframe in path parts
                    for part in path_parts:
                        if part in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
                            file_timeframe = part
                            break
                    
                    # Extract symbol (assuming format like BTC_USDT)
                    for part in path_parts:
                        if "_" in part and not part.startswith("."): 
                            file_symbol = part
                            break
                    
                    # Skip if we couldn't determine symbol or timeframe
                    if not file_symbol or not file_timeframe:
                        continue
                        
                    # Apply filters
                    if symbol and file_symbol != symbol.replace("/", "_"):
                        continue
                        
                    if timeframe and file_timeframe != timeframe:
                        continue
                    
                    scanned_count += 1
                    
                    # Check if it's a legacy file (has timestamp in name)
                    if re.search(r'\d{10,}', file):
                        legacy_files.append({
                            "path": file_path,
                            "symbol": file_symbol,
                            "timeframe": file_timeframe
                        })
        
        if not legacy_files:
            logger.info(f"No legacy files found. Scanned {scanned_count} files.")
            return {"scanned": scanned_count, "migrated": 0, "failed": 0}
        
        logger.info(f"Found {len(legacy_files)} legacy files to migrate out of {scanned_count} scanned")
        
        # Migrate files
        results = {"scanned": scanned_count, "migrated": 0, "failed": 0}
        
        for file_info in legacy_files:
            try:
                file_path = file_info['path']
                file_symbol = file_info['symbol']
                file_timeframe = file_info['timeframe']
                
                logger.info(f"Migrating {file_path}")
                
                # Create backup if requested
                if backup:
                    backup_path = f"{file_path}.bak"
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                
                # Load file data
                data = pd.read_parquet(file_path)
                
                # Convert symbol format from BTC_USDT to BTC/USDT
                symbol_for_api = file_symbol.replace("_", "/")
                
                local_source = None
                
                if isinstance(data_manager.primary_source, LocalSource):
                    local_source = data_manager.primary_source
                elif hasattr(data_manager, 'backup_source') and isinstance(data_manager.backup_source, LocalSource):
                    local_source = data_manager.backup_source
                
                if local_source:
                    success = await local_source.update_data(symbol_for_api, file_timeframe, data)
                    
                    if success:
                        results["migrated"] += 1
                        logger.info(f"Successfully migrated {file_path}")
                        
                        # Delete old file if requested
                        if delete_old:
                            os.remove(file_path)
                            logger.info(f"Deleted legacy file {file_path}")
                    else:
                        results["failed"] += 1
                        logger.error(f"Failed to migrate {file_path}")
                else:
                    results["failed"] += 1
                    logger.error("No LocalSource available for migration")
                
            except Exception as e:
                results["failed"] += 1
                logger.error(f"Error migrating file {file_info['path']}: {str(e)}")
        
        # Close data manager
        logger.info("Shutting down data manager")
        await data_manager.close()
        
        logger.info(f"Migration completed. Summary: {results['migrated']} succeeded, {results['failed']} failed")
        return results
        
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        traceback.print_exc()
        return {"scanned": 0, "migrated": 0, "failed": len(legacy_files) if 'legacy_files' in locals() else 0}