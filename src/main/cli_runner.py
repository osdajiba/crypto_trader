# src/scripts/download_data.py
#!/usr/bin/env python

import asyncio
import argparse
import logging
import os
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any
import importlib.util

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.datasource.data_manager import DataManager


def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('data_download.log')
        ]
    )
    return LogManager.get_logger('data_download')


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


async def run_downloads(config, symbols, timeframes, start_date, end_date, max_concurrent=3):
    """Manage downloading data for multiple symbols and timeframes"""
    logger = setup_logging()
    
    try:
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
    

def run_cli_mode(args: argparse.Namespace, logger) -> Dict[str, Any]:
    """Run the system in CLI mode with the provided arguments"""
    logger.info(f"Starting trading system in CLI mode with {args.mode} mode")
    
    # Apply command line overrides to config
    config = apply_config_overrides(args, logger)
    
    # Initialize trading core
    try:
        logger.info("Initializing trading core...")
        # Dynamically import the core module
        spec = importlib.util.spec_from_file_location(
            "core", 
            os.path.join(project_root, "src/core/core.py")
        )
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        
        # Create core instance with backtest engine type if specified
        TradingCore = core_module.TradingCore
        core = TradingCore(config, mode=args.mode, backtest_engine=args.backtest_engine)
        
        # Register progress callback for CLI display
        core.set_progress_callback(cli_progress_callback)
        
        # Run the trading pipeline
        logger.info("Starting trading pipeline...")
        
        # Use async executor to run the async pipeline
        from src.common.async_executor import AsyncExecutor
        executor = AsyncExecutor()
        result = executor.run(core.run_pipeline())
        
        logger.info("Trading pipeline completed")
        
        # Display summary results if available
        if isinstance(result, dict) and "error" not in result:
            display_cli_summary(result)
        
        # Shutdown properly
        executor.run(core.shutdown())
        return result
    
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import required module: {str(e)}")
        return {"error": f"Failed to import required module: {str(e)}"}
    except Exception as e:
        logger.error(f"Error running trading pipeline: {str(e)}", exc_info=True)
        traceback.print_exc()
        return {"error": f"Error running trading pipeline: {str(e)}"}


def apply_config_overrides(args: argparse.Namespace, logger) -> ConfigManager:
    """Apply command line argument overrides to the configuration"""
    logger.info("Applying command line overrides to configuration...")
    
    # Initialize config if not provided
    config = None
    if hasattr(args, 'config') and args.config:
        if isinstance(args.config, ConfigManager):
            config = args.config
        else:
            config = ConfigManager(str(args.config))
    else:
        default_config_path = Path(project_root) / "conf" / "config.yaml"
        config = ConfigManager(str(default_config_path))
          
    # Apply mode override
    if hasattr(args, 'mode') and args.mode:
        config.set("system", "operational_mode", args.mode)
        logger.info(f"Overriding operational mode: {args.mode}")
    
    # Apply backtest engine type override
    if hasattr(args, 'backtest_engine') and args.backtest_engine:
        config.set("backtest", "engine_type", args.backtest_engine)
        logger.info(f"Overriding backtest engine type: {args.backtest_engine}")
    
    # Apply strategy override
    if hasattr(args, 'strategy') and args.strategy:
        config.set("strategy", "active", args.strategy)
        logger.info(f"Overriding strategy: {args.strategy}")
    
    # Apply symbol override
    if hasattr(args, 'symbol') and args.symbol:
        # Handle multiple symbols if comma-separated
        symbols = [s.strip() for s in args.symbol.split(',')]
        config.set("trading", "instruments", symbols)
        logger.info(f"Overriding trading symbols: {symbols}")
    
    # Apply timeframe override
    if hasattr(args, 'timeframe') and args.timeframe:
        # Set both relevant timeframe configurations
        config.set("strategy", "timeframe", args.timeframe)
        config.set("trading", "timeframe", args.timeframe)
        logger.info(f"Overriding timeframe: {args.timeframe}")
    
    # Apply backtest date range overrides
    if hasattr(args, 'start_date') and args.start_date:
        config.set("backtest", "period", "start", args.start_date)
        logger.info(f"Overriding backtest start date: {args.start_date}")
    
    if hasattr(args, 'end_date') and args.end_date:
        config.set("backtest", "period", "end", args.end_date)
        logger.info(f"Overriding backtest end date: {args.end_date}")
    
    # Apply database URL override
    if hasattr(args, 'db_url') and args.db_url:
        config.set("database", "url", args.db_url)
        logger.info(f"Overriding database URL: {args.db_url}")
    
    # Apply max workers override
    if hasattr(args, 'max_workers') and args.max_workers:
        config.set("system", "performance", "max_threads", args.max_workers)
        logger.info(f"Overriding max threads: {args.max_workers}")
    
    return config


def cli_progress_callback(percent: float, message: str) -> None:
    """Display progress in the CLI"""
    # Calculate the width based on terminal size
    try:
        import shutil
        width = shutil.get_terminal_size().columns
    except (ImportError, AttributeError):
        width = 80
    
    # Progress bar width (adjust as needed)
    bar_width = max(10, width - 40)
    
    # Create the progress bar
    filled_width = int(bar_width * percent / 100)
    bar = '█' * filled_width + '░' * (bar_width - filled_width)
    
    # Format and print the progress
    sys.stdout.write(f"\r[{bar}] {percent:.1f}% | {message}")
    sys.stdout.flush()
    
    # Add newline if complete
    if percent >= 100:
        sys.stdout.write("\n")


def display_cli_summary(result: Dict[str, Any]) -> None:
    """Display a formatted summary of results in the CLI"""
    print("\n" + "="*80)
    print(f"{'TRADING SUMMARY':^80}")
    print("="*80)
    
    # Extract key metrics
    if "initial_capital" in result:
        print(f"Initial Capital:    ${result.get('initial_capital', 0):,.2f}")
    if "final_equity" in result:
        print(f"Final Equity:       ${result.get('final_equity', 0):,.2f}")
    if "total_return" in result:
        print(f"Total Return:       ${result.get('total_return', 0):,.2f} ({result.get('total_return_pct', 0):.2f}%)")
    if "max_drawdown_pct" in result:
        print(f"Maximum Drawdown:   {result.get('max_drawdown_pct', 0):.2f}%")
    if "sharpe_ratio" in result:
        print(f"Sharpe Ratio:       {result.get('sharpe_ratio', 0):.2f}")
    if "total_trades" in result:
        print(f"Total Trades:       {result.get('total_trades', 0)}")
        print(f"Win Rate:           {result.get('win_rate', 0):.2f}%")
    
    # Display backtest parameters
    if "backtest_params" in result:
        print("\n" + "-"*80)
        print(f"{'BACKTEST PARAMETERS':^80}")
        print("-"*80)
        
        params = result["backtest_params"]
        print(f"Symbols:            {', '.join(params.get('symbols', []))}")
        print(f"Timeframe:          {params.get('timeframe', 'N/A')}")
        print(f"Period:             {params.get('start_date', 'N/A')} to {params.get('end_date', 'N/A')}")
        print(f"Engine Type:        {result.get('engine_type', 'N/A')}")
        print(f"Commission Rate:    {params.get('commission_rate', 0)*100:.2f}%")
        print(f"Slippage:           {params.get('slippage', 0)*100:.2f}%")
    
    print("\n" + "="*80)
    
    # Ask if user wants to see detailed trades
    if "trades" in result and result["trades"]:
        show_details = input("\nDo you want to see detailed trade information? (y/n): ").lower() == 'y'
        if show_details:
            print("\n" + "-"*100)
            print(f"{'TRADE DETAILS':^100}")
            print("-"*100)
            
            # Display trade header
            print(f"{'DATE':^20} | {'SYMBOL':^8} | {'ACTION':^6} | {'QUANTITY':^10} | {'PRICE':^10} | {'COMMISSION':^10} | {'CASH AFTER':^15}")
            print("-"*100)
            
            # Display trades (limit to 20 for readability)
            max_trades = 20
            for i, trade in enumerate(result["trades"][:max_trades]):
                print(f"{str(trade.get('timestamp', 'N/A')):^20} | {trade.get('symbol', ''):^8} | "
                      f"{trade.get('action', ''):^6} | {trade.get('quantity', 0):^10.5f} | "
                      f"{trade.get('price', 0):^10.2f} | {trade.get('commission', 0):^10.5f} | "
                      f"${trade.get('cash_after', 0):^15,.2f}")
            
            # Show message if there are more trades
            if len(result["trades"]) > max_trades:
                print(f"\n... and {len(result['trades']) - max_trades} more trades not shown ...")
    
    print("\n")
    
    
def main():
    """Parse arguments and run the data download"""
    parser = argparse.ArgumentParser(description='Download and store market data in daily files')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--symbols', type=str, nargs='+', default=['BTC/USDT'], help='Symbols to download (e.g., BTC/USDT ETH/USDT)')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['1h'], help='Timeframes to download (e.g., 1h 4h 1d)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Maximum concurrent downloads')
    
    args = parser.parse_args()
    
    # Set up basic logging before main logging initialization
    logger = setup_logging()
    
    try:
        start_time = datetime.now()
        logger.info(f"Starting data download at {start_time}")
        
        # Run the download process
        results = asyncio.run(run_downloads(
            args.config,
            args.symbols,
            args.timeframes,
            args.start,
            args.end,
            args.max_concurrent
        ))
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"Download completed at {end_time}. Total duration: {duration}")
        logger.info(f"Results: {results['success']} succeeded, {results['failed']} failed")
        
        # Return success if at least one download succeeded
        return 0 if results['success'] > 0 else 1
        
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Download failed: {str(e)}", exc_info=True)
        return 1