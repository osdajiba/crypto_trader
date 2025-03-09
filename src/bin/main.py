# src/bin/main.py

import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor
from src.core.core import TradingCore

# Constants
BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "conf/bt_config.yaml"

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Trading System")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", 
                          help="Start in GUI mode")
    mode_group.add_argument("--mode", type=str, 
                          choices=["backtest", "paper", "live"], 
                          help="Trading mode")
    
    # Configuration
    parser.add_argument("--config", type=str, 
                      help="Path to configuration file")
    parser.add_argument("--strategy", type=str, 
                      help="Strategy to use")
    parser.add_argument("--symbol", type=str, 
                      help="Trading symbol")
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
    
    return parser.parse_args()

async def async_main(config_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Async main function entry point"""
    # Check if config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
        
    # Initialize config and validate
    config = ConfigManager(config_path=str(config_path))
    log_dir = BASE_DIR / config.get("log_paths", "error_log", default="logs/error.log")

    # Initialize logging system
    LogManager(base_dir=log_dir)
    log_level = "DEBUG" if args.debug else config.get("logging", "level", default="INFO")
    logger = LogManager.get_logger(name="trading_system", level=log_level)
    logger.info("Trading system initializing...")

    # Apply command line overrides to config
    if args.mode:
        config.set("mode", "type", args.mode)
        
    if args.strategy:
        config.set("strategy", "type", args.strategy)
        
    if args.symbol:
        # Handle multiple symbols if comma-separated
        symbols = [s.strip() for s in args.symbol.split(',')]
        config.set("symbols", "list", symbols)
        
    if args.timeframe:
        config.set("timeframe", args.timeframe)
        
    if args.start_date:
        config.set("backtest", "start_date", args.start_date)
        
    if args.end_date:
        config.set("backtest", "end_date", args.end_date)
    
    # Initialize trading core
    core = TradingCore(config)
    
    try:
        logger.info("Starting trading pipeline...")
        result = await core.run_pipeline()
        return result
    finally:
        logger.info("Shutting down trading system...")
        await core.shutdown()

def start_gui(args: argparse.Namespace):
    """Start the GUI version of the application"""
    try:
        import tkinter as tk
    except ImportError:
        return {"error": "Tkinter not available"}
    
    try:
        # Import GUI module - assuming it's been installed or is in path
        from trading_system_gui import TradingSystemGUI
            
        # Create root window
        root = tk.Tk()
        
        # Initialize GUI
        app = TradingSystemGUI(root)
        
        # Apply command line arguments if provided
        if args.config:
            app.config_file = args.config
        if args.mode:
            app.mode_var.set(args.mode)
        if args.strategy:
            app.strategy_var.set(args.strategy)
        if args.start_date:
            app.start_date_var.set(args.start_date)
        if args.end_date:
            app.end_date_var.set(args.end_date)
            
        # Start main loop
        root.mainloop()
        return {"status": "GUI closed"}
        
    except Exception as e:
        return {"error": f"GUI error: {str(e)}"}
    
def main() -> Optional[Dict[str, Any]]:
    """Main entry point"""
    try:
        args = parse_arguments()
        config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        
        if args.gui:
            return start_gui(args)
        else:
            executor = AsyncExecutor()
            return executor.run(async_main(config_path, args))

    except FileNotFoundError as e:
        return {"error": f"Configuration error: {str(e)}"}
    except Exception as e:
        return {"error": f"Critical error: {str(e)}"}

if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0 if not result or "error" not in result else 1)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        sys.exit(1)