#!/usr/bin/env python
# src/ui/launcher.py

import os
import sys
import argparse
from pathlib import Path
import tkinter as tk

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.main.CLI_runner import run_cli_mode
from src.main.GUI_app import TradingSystemGUI

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

def setup_environment(args):
    """Setup environment variables and paths"""
    # Constants
    base_dir = Path(project_root)
    default_config_path = base_dir / "conf/bt_config.yaml"
    config_path = Path(args.config) if args.config else default_config_path
    
    # Initialize config
    config = ConfigManager(config_path=str(config_path))
    
    # Initialize logging
    LogManager(config)
    log_level = "DEBUG" if args.debug else config.get("logging", "level", default="INFO")
    logger = LogManager.get_logger(name="trading_system", level=log_level)
    logger.info(f"Trading system launcher initialized with {'GUI' if args.gui else 'CLI'} mode")
    
    return config_path, config, logger

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
            
def launch():
    """Main entry point for the launcher"""
    args = parse_arguments()
    
    # Check if the user specified a mode
    if not args.gui and not args.cli:
        # Default to CLI mode if neither CLI nor GUI is specified
        args.cli = True
    
    # Setup environment
    config_path, config, logger = setup_environment(args)
    
    try:
        if args.gui:
            # Start GUI mode
            root = tk.Tk()
            root.title("Trading System")
            app = TradingSystemGUI(root, config, args)
            root.mainloop()
        else:
            # Start CLI mode
            # If no trading mode specified, prompt for selection
            if not args.mode:
                args.mode = prompt_mode_selection()
                
            # Run in CLI mode
            run_cli_mode(args, config_path, config, logger)
    except Exception as e:
        logger.error(f"Critical error in launcher: {str(e)}", exc_info=True)
        return {"error": f"Critical error: {str(e)}"}
    
    logger.info("Trading system shutting down")
    return {"status": "completed"}

if __name__ == "__main__":
    try:
        result = launch()  # Fixed: Changed from main() to launch()
        sys.exit(0 if not result or "error" not in result else 1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        sys.exit(1)