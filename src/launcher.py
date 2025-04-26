#!/usr/bin/env python
# src/main/launcher.py

import os
import sys
import argparse
from pathlib import Path
import tkinter as tk

from common.logging import LogManager, LogInitializer
from common.config import ConfigManager


# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the launcher"""
    parser = argparse.ArgumentParser(description="Trading System Launcher")
    
    # Mode selection
    parser.add_argument("--gui", action="store_true", 
                        help="Start in GUI mode")
    parser.add_argument("--cli", action="store_true", 
                        help="Start in CLI mode")
    
    # Configuration
    parser.add_argument("--config", type=str, 
                      help="Path to configuration file")
    parser.add_argument("--mode", type=str, 
                      choices=["backtest", "paper", "live"], 
                      help="Trading mode")
    parser.add_argument("--strategy", type=str, 
                      help="Strategy to use")
    parser.add_argument("--symbol", type=str, 
                      help="Trading symbol (comma-separated for multiple)")
    parser.add_argument("--timeframe", type=str, 
                      help="Timeframe for analysis")
    parser.add_argument("--start-date", type=str, 
                      help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, 
                      help="End date for backtesting (YYYY-MM-DD)")
    
    # Backtest engine selection
    parser.add_argument("--backtest-engine", type=str,
                      choices=["ohlcv", "market_replay", "basic", "advanced"],
                      help="Backtest engine to use")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    parser.add_argument("--verbose", "-v", action="count", default=0, 
                      help="Increase verbosity")
    parser.add_argument("--db-url", type=str, 
                      help="Database connection URL")
    parser.add_argument("--max-workers", type=int, 
                      help="Async worker threads")
    
    return parser.parse_args()
    
def prompt_mode_selection() -> str:
    """Command-line prompt for trading mode selection"""
    print("\nTrading Mode Selection")
    print("========================================")
    print("1. Backtest    - Historical data simulation")
    print("2. Paper       - Real-time simulation")
    print("3. Live        - Real order execution (CAUTION)")
    print("========================================")
    print("Warning: Live trading requires proper configuration!\n")
    
    while True:
        try:
            choice = int(input("Select trading mode (1-3): "))
            if 1 <= choice <= 3:
                modes = {1: "backtest", 2: "paper", 3: "live"}
                return modes[choice]
            else:
                print("Invalid selection. Please choose 1-3.")
        except ValueError:
            print("Please enter a number from 1 to 3.")

def prompt_backtest_engine_selection() -> str:
    """Command-line prompt for backtest engine selection"""
    print("\nBacktest Engine Selection")
    print("========================================")
    print("1. OHLCV (Basic)          - Standard backtesting with OHLCV data")
    print("2. Market Replay (Advanced) - High-fidelity market replay with realistic order execution")
    print("========================================")
    
    while True:
        try:
            choice = int(input("Select backtest engine type (1-2): "))
            if choice == 1:
                return "ohlcv"
            elif choice == 2:
                return "market_replay"
            else:
                print("Invalid selection. Please choose 1 or 2.")
        except ValueError:
            print("Please enter either 1 or 2.")

def setup_environment(args):
    """Setup environment variables and paths with enhanced logging"""
    # Constants
    base_dir = Path(project_root)
    default_config_path = base_dir / "conf/bt_config.yaml"
    config_path = Path(args.config) if args.config else default_config_path
    
    config_manager = ConfigManager(config_path=config_path)
    config_manager.load()
    
    from common.logging import LogManager
    LogManager(config_manager)
    
    log_init = LogInitializer(config_manager)
    log_init.initialize()
    
    args.config = config_manager
    
    logger = LogManager.get_logger("trading_system")
    
    if args.debug:
        LogManager.set_level("DEBUG")
        logger.debug("Debug logging enabled")
    
    return logger

def launch():
    """Main entry point for the launcher"""
    args = parse_arguments()
    logger = setup_environment(args)    

    if not args.gui and not args.cli:
        args.gui = True
    
    log_level = "DEBUG" if args.debug else args.config.get("logging", "level", default="INFO")
    logger = LogManager.get_logger(name="trading_system", level=log_level)
    logger.info(f"Trading system launcher initialized with {'GUI' if args.gui else 'CLI'} mode")
    
    try:
        if args.gui:
            # Start GUI mode
            root = tk.Tk()
            root.title("Trading System")
            from src.ui.gui import TradingSystemGUI
            app = TradingSystemGUI(root, args)
            root.mainloop()
        else:
            # Start CLI mode
            # If no trading mode specified, prompt for selection
            if not args.mode:
                args.mode = prompt_mode_selection()
                
            # For backtest mode, prompt for engine type if not specified
            if args.mode == "backtest" and not args.backtest_engine:
                args.backtest_engine = prompt_backtest_engine_selection()
                
            # Run in CLI mode
            from src.ui.cli import run_cli_mode
            run_cli_mode(args, logger)
            
    except Exception as e:
        logger.error(f"Critical error in launcher: {str(e)}", exc_info=True)
        return {"error": f"Critical error: {str(e)}"}
    
    logger.info("Trading system shutting down")
    return {"status": "completed"}