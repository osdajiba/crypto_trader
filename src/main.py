#!/usr/bin/env python
# src/main.py - Professional Crypto Trading System CLI

import os
import sys
import argparse
import asyncio
import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Set up basic logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("cli")

def run_command():
    """Entry point to run CLI commands"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Default to trade command if none provided
    if not hasattr(args, 'command') or not args.command:
        args.command = 'trade'
    
    # Set logging level based on verbosity - safely check attributes
    if hasattr(args, 'debug') and args.debug:
        logger.setLevel(logging.DEBUG)
    elif hasattr(args, 'verbose') and args.verbose:
        level = logging.INFO if args.verbose == 1 else logging.DEBUG
        logger.setLevel(level)
    
    # Load configuration
    config_path = args.config if hasattr(args, 'config') and args.config else os.path.join(project_root, "conf", "config.yaml")
    
    try:
        # Import and initialize configuration
        from src.common.config import ConfigManager
        config = ConfigManager(config_path)
        config.load()
        
        # Initialize logging system with config
        setup_logging(config)
        
        # Execute command
        logger.info(f"{'='*20} COMMAND: {args.command.upper()} {'='*20}")
        
        # Apply command-line overrides to config
        apply_args_to_config(args, config)
        
        # Run the appropriate command
        command_map = {
            'trade': handle_trade,
            'download': handle_download,
            'migrate': handle_migrate,
            'optimize': handle_optimize
        }
        
        handler = command_map.get(args.command)
        if not handler:
            logger.error(f"Unknown command: {args.command}")
            return 1
            
        result = handler(args, config)
        
        # Save output to file if specified
        if hasattr(args, 'output') and args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        
        logger.info(f"{'='*20} COMMAND COMPLETED {'='*20}")
        return 0 if not result or "error" not in result else 1
    
    except KeyboardInterrupt:
        logger.warning("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}", exc_info=True)
        return 1

def setup_logging(config):
    """Setup advanced logging if available"""
    try:
        # Try to initialize LogManager
        from src.common.log_manager import LogManager
        LogManager(config)
        
        # Get a new logger after initialization
        global logger
        logger = logging.getLogger("cli")
        logger.info("Advanced logging initialized")
    except Exception as e:
        logger.warning(f"Could not initialize advanced logging: {e}")
        logger.info("Using basic logging")

def apply_args_to_config(args, config):
    """Apply command line arguments to configuration"""
    # Direct mappings from args to config
    mappings = {
        'mode': ("system", "operational_mode"),
        'backtest_engine': ("backtest", "engine_type"),
        'strategy': ("strategy", "active"),
        'timeframe': ("strategy", "timeframe"),
        'db_url': ("database", "url"),
        'max_workers': ("system", "performance", "max_threads")
    }
    
    # Apply mappings
    for arg_name, config_path in mappings.items():
        if hasattr(args, arg_name) and getattr(args, arg_name):
            config.set(*config_path, getattr(args, arg_name))
            logger.info(f"Config override: {config_path} = {getattr(args, arg_name)}")
    
    # Handle symbol (comma-separated)
    if hasattr(args, 'symbol') and args.symbol:
        symbols = [s.strip() for s in args.symbol.split(',')]
        config.set("trading", "instruments", symbols)
        logger.info(f"Override trading symbols: {symbols}")
    
    # Handle date range
    if hasattr(args, 'start_date') and args.start_date:
        config.set("backtest", "period", "start", args.start_date)
    if hasattr(args, 'end_date') and args.end_date:
        config.set("backtest", "period", "end", args.end_date)
    
    # Handle JSON parameters
    if hasattr(args, 'params') and args.params:
        try:
            params = json.loads(args.params)
            for key, value in params.items():
                config.set("strategy", "parameters", key, value)
        except json.JSONDecodeError:
            logger.warning(f"Invalid strategy parameters JSON: {args.params}")
    
    # Handle risk parameters
    if hasattr(args, 'risk_params') and args.risk_params:
        try:
            risk_params = json.loads(args.risk_params)
            for key, value in risk_params.items():
                if key.startswith("stop_loss") and len(key.split("_")) > 2:
                    config.set("risk", "stop_loss", "_".join(key.split("_")[2:]), value)
                elif key.startswith("take_profit") and len(key.split("_")) > 2:
                    config.set("risk", "take_profit", "_".join(key.split("_")[2:]), value)
                else:
                    config.set("risk", key, value)
        except json.JSONDecodeError:
            logger.warning(f"Invalid risk parameters JSON: {args.risk_params}")

def handle_trade(args, config):
    """Run the trading system"""
    logger.info("Starting trading pipeline...")
    
    try:
        # Import required modules
        from src.common.async_executor import AsyncExecutor
        
        # Use importlib to avoid dependency issues
        import importlib
        try:
            core_module = importlib.import_module("src.core.core")
        except Exception as e:
            logger.error(f"Failed to import core module: {e}")
            return {"error": f"Failed to import core module: {e}"}
        
        # Create and initialize trading core
        trader = core_module.TradingCore(
            config,
            getattr(args, 'mode', None),
            getattr(args, 'backtest_engine', None)
        )
        
        # Set progress callback
        trader.set_progress_callback(show_progress)
        
        # Execute trading pipeline
        executor = AsyncExecutor()
        result = executor.run(trader.run_pipeline())
        
        # Clean shutdown
        executor.run(trader.shutdown())
        
        return result
    except Exception as e:
        logger.error(f"Trading error: {str(e)}", exc_info=True)
        return {"error": f"Trading error: {str(e)}"}

def handle_download(args, config):
    """Download historical market data"""
    logger.info("Starting data download...")
    
    try:
        # Import downloader
        from src.datasource.downloader import run_downloads
        
        # Parse parameters
        symbols = args.symbol.split(',') if hasattr(args, 'symbol') and args.symbol else ["BTC/USDT"]
        timeframes = args.timeframes.split(',') if hasattr(args, 'timeframes') and args.timeframes else ["1h"]
        
        # Set date range (default to last 30 days)
        start_date = args.start_date if hasattr(args, 'start_date') and args.start_date else \
            (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        end_date = args.end_date if hasattr(args, 'end_date') and args.end_date else \
            datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Set concurrent downloads
        concurrent = int(args.concurrent) if hasattr(args, 'concurrent') and args.concurrent else 3
        
        # Log parameters
        logger.info(f"Downloading data for: {symbols}, timeframes: {timeframes}")
        logger.info(f"Date range: {start_date} to {end_date}, concurrent: {concurrent}")
        
        # Run download
        result = asyncio.run(run_downloads(
            config_path=str(config._config_path),
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            concurrent=concurrent
        ))
        
        logger.info(f"Download completed: {result['success']} successful, {result['failed']} failed")
        if result['failed'] > 0 and 'errors' in result:
            for error in result['errors'][:5]:
                logger.error(f"Download error: {error}")
        
        return result
    except Exception as e:
        logger.error(f"Download error: {str(e)}", exc_info=True)
        return {"error": f"Download error: {str(e)}"}

def handle_migrate(args, config):
    """Migrate data formats"""
    logger.info("Starting data migration...")
    
    try:
        # Import migration function
        from src.datasource.downloader import scan_and_migrate
        
        # Get parameters
        symbol = args.symbol if hasattr(args, 'symbol') and args.symbol else None
        timeframe = args.timeframe if hasattr(args, 'timeframe') and args.timeframe else None
        
        # Set flags
        backup = not args.no_backup if hasattr(args, 'no_backup') else True
        delete_old = args.delete if hasattr(args, 'delete') else False
        
        # Confirm if deleting
        if delete_old:
            confirmation = input("Warning: This will delete original files after migration. Continue? (y/n): ")
            if confirmation.lower() != 'y':
                logger.info("Migration cancelled by user")
                return {"status": "cancelled"}
        
        # Log parameters
        logger.info(f"Migrating data{' for '+symbol if symbol else ''}{' with timeframe '+timeframe if timeframe else ''}")
        logger.info(f"Backup: {backup}, Delete original: {delete_old}")
        
        # Run migration
        result = asyncio.run(scan_and_migrate(
            config_path=str(config._config_path),
            symbol=symbol,
            timeframe=timeframe,
            backup=backup,
            delete_original=delete_old
        ))
        
        return result
    except Exception as e:
        logger.error(f"Migration error: {str(e)}", exc_info=True)
        return {"error": f"Migration error: {str(e)}"}

def handle_optimize(args, config):
    """Optimize strategy parameters"""
    logger.info("Starting strategy optimization...")
    
    try:
        # Validate required parameters
        if not hasattr(args, 'optimize_params') or not args.optimize_params:
            return {"error": "No parameters specified for optimization"}
        
        # Parse parameter ranges
        try:
            param_ranges = json.loads(args.optimize_params)
        except json.JSONDecodeError:
            return {"error": "Invalid optimization parameters format"}
        
        # Get optimization metric
        metric = args.metric if hasattr(args, 'metric') and args.metric else "sharpe_ratio"
        logger.info(f"Optimization metric: {metric}")
        
        # Generate parameter combinations
        params_list = generate_param_combinations(param_ranges)
        logger.info(f"Testing {len(params_list)} parameter combinations")
        
        # Initialize tracking
        results = []
        best_result = None
        best_params = None
        best_value = float('-inf')
        
        # Check if metric should be minimized
        minimize_metric = metric in ["max_drawdown", "volatility"]
        if minimize_metric:
            best_value = float('inf')
        
        # Import required modules
        from src.common.config import ConfigManager
        from src.common.async_executor import AsyncExecutor
        import importlib
        
        try:
            core_module = importlib.import_module("src.core.core")
        except Exception as e:
            logger.error(f"Failed to import core module: {e}")
            return {"error": f"Failed to import core module: {e}"}
        
        # Run tests for each parameter combination
        for i, params in enumerate(params_list):
            logger.info(f"Testing combination {i+1}/{len(params_list)}: {params}")
            
            # Create test config
            run_config = ConfigManager(str(config._config_path))
            run_config.load()
            
            # Apply parameters
            for key, value in params.items():
                run_config.set("strategy", "parameters", key, value)
            
            # Ensure backtest mode
            run_config.set("system", "operational_mode", "backtest")
            
            # Create trader and run backtest
            trader = core_module.TradingCore(run_config, "backtest", getattr(args, 'backtest_engine', None))
            trader.set_progress_callback(
                lambda percent, message: show_progress(percent, f"[Run {i+1}/{len(params_list)}] {message}")
            )
            
            executor = AsyncExecutor()
            run_result = executor.run(trader.run_pipeline())
            executor.run(trader.shutdown())
            
            # Extract result metric
            if "metrics" in run_result and metric in run_result["metrics"]:
                metric_value = run_result["metrics"][metric]
                
                # Store result
                result_entry = {
                    "params": params,
                    "result": run_result["metrics"],
                    "value": metric_value
                }
                results.append(result_entry)
                
                # Check if best result
                is_better = (not minimize_metric and metric_value > best_value) or \
                          (minimize_metric and metric_value < best_value)
                if is_better:
                    best_value = metric_value
                    best_params = params
                    best_result = run_result
                    
                logger.info(f"Run {i+1} completed: {metric}={metric_value}")
        
        logger.info(f"Optimization completed. Best {metric}: {best_value}")
        logger.info(f"Best parameters: {json.dumps(best_params)}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "best_result": best_result,
            "all_results": results
        }
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}", exc_info=True)
        return {"error": f"Optimization error: {str(e)}"}

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(description="Crypto Trading System CLI")
    
    # Add global arguments
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, help="Output file for results (JSON format)")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Common arguments for subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    
    # Trade command
    trade_parser = subparsers.add_parser('trade', parents=[parent_parser], help='Run trading system')
    trade_parser.add_argument("--mode", type=str, choices=["backtest", "paper", "live"], help="Trading mode")
    trade_parser.add_argument("--strategy", type=str, help="Strategy to use")
    trade_parser.add_argument("--symbol", type=str, help="Trading symbol (comma-separated)")
    trade_parser.add_argument("--timeframe", type=str, help="Timeframe for analysis")
    trade_parser.add_argument("--backtest-engine", type=str, choices=["ohlcv", "market_replay"], help="Backtest engine type")
    trade_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    trade_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    trade_parser.add_argument("--params", type=str, help='Strategy parameters as JSON')
    trade_parser.add_argument("--risk-params", type=str, help='Risk parameters as JSON')
    trade_parser.add_argument("--max-workers", type=int, help="Async worker threads")
    trade_parser.add_argument("--db-url", type=str, help="Database connection URL")
    
    # Download command
    download_parser = subparsers.add_parser('download', parents=[parent_parser], help='Download historical data')
    download_parser.add_argument("--symbol", type=str, help="Symbols to download (comma-separated)")
    download_parser.add_argument("--timeframes", type=str, help="Timeframes (comma-separated)")
    download_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    download_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    download_parser.add_argument("--concurrent", type=int, help="Max concurrent downloads")
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', parents=[parent_parser], help='Migrate data formats')
    migrate_parser.add_argument("--symbol", type=str, help="Symbol to migrate (optional)")
    migrate_parser.add_argument("--timeframe", type=str, help="Timeframe to migrate (optional)")
    migrate_parser.add_argument("--no-backup", action="store_true", help="Skip creating backup files")
    migrate_parser.add_argument("--delete", action="store_true", help="Delete original files")
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', parents=[parent_parser], help='Optimize strategy parameters')
    optimize_parser.add_argument("--optimize-params", type=str, required=True, help='Parameters to optimize as JSON with arrays of values')
    optimize_parser.add_argument("--metric", type=str, default="sharpe_ratio", 
                               choices=["sharpe_ratio", "total_return", "max_drawdown", "win_rate", "profit_factor"],
                               help="Metric to optimize")
    # Add trading arguments to optimizer
    optimize_parser.add_argument("--strategy", type=str, help="Strategy to use")
    optimize_parser.add_argument("--symbol", type=str, help="Trading symbol (comma-separated)")
    optimize_parser.add_argument("--timeframe", type=str, help="Timeframe for analysis")
    optimize_parser.add_argument("--backtest-engine", type=str, choices=["ohlcv", "market_replay"], help="Backtest engine type")
    optimize_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    optimize_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    return parser

def show_progress(percent: float, message: str) -> None:
    """Display progress bar in console"""
    bar_length = 40
    filled_length = int(bar_length * percent / 100)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    percent_display = f"{percent:.1f}" if percent < 100 else "100"
    sys.stdout.write(f'\r[{bar}] {percent_display}% {message}')
    sys.stdout.flush()
    
    # Add newline at 100%
    if percent >= 100:
        sys.stdout.write('\n')
        sys.stdout.flush()

def generate_param_combinations(param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from provided ranges"""
    if not param_ranges:
        return [{}]
    
    # Extract parameter names and values
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = []
    
    # Recursive function to generate combinations
    def generate(current_params, index):
        if index == len(param_names):
            combinations.append(current_params.copy())
            return
        
        param_name = param_names[index]
        values = param_values[index]
        
        for value in values:
            current_params[param_name] = value
            generate(current_params, index + 1)
    
    generate({}, 0)
    return combinations

if __name__ == "__main__":
    sys.exit(run_command())