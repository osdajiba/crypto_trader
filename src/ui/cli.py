#!/usr/bin/env python
# src/main/cli_runner.py

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
    """
    Run the system in CLI mode with the provided arguments.
    
    Args:
        args: Command line arguments
        config_path: Path to configuration file
        config: Configuration manager instance
        logger: Logger instance
        
    Returns:
        Dict[str, Any]: Results dictionary or error information
    """
    # Apply mode override
    if hasattr(args, 'mode') and args.mode:
        config.set("system", "operational_mode", args.mode)
        logger.info(f"Overriding operational mode: {args.mode}")

    # Apply backtest engine override
    if hasattr(args, 'backtest_engine') and args.backtest_engine:
        config.set("backtest", "engine_type", args.backtest_engine)
        logger.info(f"Overriding backtest engine: {args.backtest_engine}")

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
        # Assuming there's a timeframe configuration
        config.set("strategy", "timeframe", args.timeframe)
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
    
    # Initialize trading core
    try:
        # Import the core module dynamically
        core_module_path = os.path.join(project_root, "src", "core", "core.py")
        module_name = "src.core.core"
        
        spec = importlib.util.spec_from_file_location(module_name, core_module_path)
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        
        # Create core instance
        TradingCore = core_module.TradingCore
        
        # Create core instance with mode and backtest engine if specified
        mode = args.mode if hasattr(args, 'mode') else None
        backtest_engine = args.backtest_engine if hasattr(args, 'backtest_engine') else None
        trader = TradingCore(config, mode, backtest_engine)
        
        # Import AsyncExecutor
        from src.common.async_executor import AsyncExecutor
        executor = AsyncExecutor()
        
        # Run the trading pipeline
        logger.info("Starting trading pipeline...")
        result = executor.run(trader.run_pipeline())
        logger.info("Trading pipeline completed")
        logger.info(f"Result: {result}")
        
        # Shutdown properly
        executor.run(trader.shutdown())
        
        return result
    except ModuleNotFoundError as e:
        logger.error(f"Failed to import required module: {str(e)}", exc_info=True)
        return {"error": f"Failed to import required module: {str(e)}"}
    except Exception as e:
        logger.error(f"Error running trading pipeline: {str(e)}", exc_info=True)
        return {"error": f"Error running trading pipeline: {str(e)}"}