# src/ui/cli_runner.py

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import importlib.util

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def run_cli_mode(args: argparse.Namespace, config_path: Path, config, logger) -> Dict[str, Any]:
    """Run the system in CLI mode with the provided arguments"""
    logger.info(f"Starting trading system in CLI mode with {args.mode} mode")
    
    # Validate configuration
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        return {"error": f"Configuration file not found: {config_path}"}
    
    # Apply command line overrides to config
    apply_config_overrides(args, config, logger)
    
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
        
        # Create core instance
        TradingCore = core_module.TradingCore
        core = TradingCore(config)
        
        # Run the trading pipeline
        logger.info("Starting trading pipeline...")
        
        # Use async executor to run the async pipeline
        from src.common.async_executor import AsyncExecutor
        executor = AsyncExecutor()
        result = executor.run(core.run_pipeline())
        
        logger.info("Trading pipeline completed")
        logger.info(f"Result: {result}")
        
        # Shutdown properly
        executor.run(core.shutdown())
        return result
    
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import required module: {str(e)}")
        return {"error": f"Failed to import required module: {str(e)}"}
    except Exception as e:
        logger.error(f"Error running trading pipeline: {str(e)}", exc_info=True)
        return {"error": f"Error running trading pipeline: {str(e)}"}

def apply_config_overrides(args: argparse.Namespace, config, logger) -> None:
    """Apply command line argument overrides to the configuration"""
    logger.info("Applying command line overrides to configuration...")
    
    # Apply mode override
    if args.mode:
        config.set("system", "operational_mode", args.mode)
        logger.info(f"Overriding operational mode: {args.mode}")
    
    # Apply strategy override
    if args.strategy:
        config.set("strategy", "active", args.strategy)
        logger.info(f"Overriding strategy: {args.strategy}")
    
    # Apply symbol override
    if args.symbol:
        # Handle multiple symbols if comma-separated
        symbols = [s.strip() for s in args.symbol.split(',')]
        config.set("trading", "instruments", symbols)
        logger.info(f"Overriding trading symbols: {symbols}")
    
    # Apply timeframe override
    if args.timeframe:
        # Assuming there's a timeframe configuration
        config.set("strategy", "timeframe", args.timeframe)
        logger.info(f"Overriding timeframe: {args.timeframe}")
    
    # Apply backtest date range overrides
    if args.start_date:
        config.set("backtest", "period", "start", args.start_date)
        logger.info(f"Overriding backtest start date: {args.start_date}")
    
    if args.end_date:
        config.set("backtest", "period", "end", args.end_date)
        logger.info(f"Overriding backtest end date: {args.end_date}")
    
    # Apply database URL override
    if args.db_url:
        config.set("database", "url", args.db_url)
        logger.info(f"Overriding database URL: {args.db_url}")
    
    # Apply max workers override
    if args.max_workers:
        config.set("system", "performance", "max_threads", args.max_workers)
        logger.info(f"Overriding max threads: {args.max_workers}")