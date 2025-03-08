# src/bin/main.py

import sys
import asyncio
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.common.log_manager import LogManager
from src.common.config_manager import ConfigManager
from src.common.async_executor import AsyncExecutor
from src.core.core import TradingCore

BASE_DIR = Path(__file__).parent.parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "conf/bt_config.yaml"

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Trading System")
    
    # Mode selection
    parser.add_argument("--gui", action="store_true", help="Start in GUI mode")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["backtest", "paper", "live"], 
                      help="Trading mode")
    parser.add_argument("--strategy", type=str, help="Strategy to use")
    parser.add_argument("--symbol", type=str, help="Trading symbol")
    parser.add_argument("--timeframe", type=str, help="Timeframe for analysis")
    parser.add_argument("--start-date", type=str, help="Start date for backtesting (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for backtesting (YYYY-MM-DD)")
    
    # Debug options
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity")
    
    return parser.parse_args()

async def async_main(config_path: Path, args: argparse.Namespace) -> Dict[str, Any]:
    """Async main function entry point for CLI mode"""
    # Check if config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
        
    # 1. Initialize config and validate
    config = ConfigManager(config_path=str(config_path))
    log_dir = BASE_DIR / config.get("log_paths", "error_log", default="logs/error.log")

    # 2. Initialize logging system
    LogManager(base_dir=log_dir)  # Create LogManager singleton
    log_level = "DEBUG" if args.debug else config.get("logging", "level", default="INFO")
    logger = LogManager.get_logger(name="trading_system", level=log_level)
    logger.info("Trading system initializing...")

    # Apply command line overrides to config
    if args.mode:
        config.set("mode", "type", args.mode)
        logger.info(f"Overriding mode from arguments: {args.mode}")
        
    if args.strategy:
        config.set("strategy", "type", args.strategy)
        logger.info(f"Overriding strategy from arguments: {args.strategy}")
        
    if args.symbol:
        symbols = args.symbol.split(',')
        config.set("symbols", symbols)
        logger.info(f"Overriding symbols from arguments: {symbols}")
        
    if args.timeframe:
        config.set("timeframe", args.timeframe)
        logger.info(f"Overriding timeframe from arguments: {args.timeframe}")
        
    if args.start_date:
        config.set("backtest", "start_date", args.start_date)
        logger.info(f"Overriding start date from arguments: {args.start_date}")
        
    if args.end_date:
        config.set("backtest", "end_date", args.end_date)
        logger.info(f"Overriding end date from arguments: {args.end_date}")

    # 3. Initialize trading core
    core = TradingCore(config)
    
    # 4. Run trading pipeline
    logger.info("Starting trading pipeline...")
    try:
        result = await core.run_pipeline()
        if "error" in result:
            logger.error(f"Trading pipeline failed: {result['error']}")
        else:
            logger.info("Trading pipeline completed successfully")
        return result
    finally:
        # Ensure resources are properly released
        logger.info("Shutting down trading system...")
        await core.shutdown()

def start_gui(args: argparse.Namespace):
    """Start the GUI version of the application"""
    try:
        import tkinter as tk
        from tkinter import messagebox
    except ImportError:
        print("Error: Tkinter is not installed or not available.")
        print("Please install tkinter to run the GUI application.")
        return {"error": "Tkinter not available"}
    
    try:
        # Look for GUI module
        gui_paths = [
            os.path.join(current_dir, 'trading_system_gui.py'),
            os.path.join(parent_dir, 'src', 'bin', 'trading_system_gui.py'),
            os.path.join(parent_dir, 'bin', 'trading_system_gui.py'),
            os.path.join(parent_dir, 'src', 'gui', 'trading_system_gui.py')
        ]
        
        gui_module_found = False
        for path in gui_paths:
            if os.path.exists(path):
                gui_module_found = True
                # Add the directory to path
                sys.path.append(os.path.dirname(path))
                break
        
        if not gui_module_found:
            print("Error: Could not find trading_system_gui.py module.")
            print("Please ensure it exists in one of these locations:")
            for path in gui_paths:
                print(f"  - {path}")
            return {"error": "GUI module not found"}
        
        # Import GUI
        try:
            from trading_system_gui import TradingSystemGUI
        except ImportError:
            # Try direct import assuming it's in the current directory
            sys.path.append(current_dir)
            from trading_system_gui import TradingSystemGUI
            
        # Create root window
        try:
            # Try themed Tk if available
            import ttkthemes
            root = ttkthemes.ThemedTk(theme="arc")
        except ImportError:
            # Fall back to standard Tk
            root = tk.Tk()
        
        # Initialize GUI
        app = TradingSystemGUI(root)
        
        # Apply command line arguments
        if args.config:
            # Override config file path
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
        print(f"Error starting GUI: {str(e)}")
        return {"error": f"GUI error: {str(e)}"}

def main() -> Optional[Dict[str, Any]]:
    """Synchronous main entry point"""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Determine config path
        config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
        
        # Choose mode based on arguments
        if args.gui:
            # Start GUI mode
            print("Starting Trading System in GUI mode...")
            return start_gui(args)
        else:
            # Start CLI mode
            print("Starting Trading System in CLI mode...")
            
            # Use AsyncExecutor to run async main function
            executor = AsyncExecutor()
            return executor.run(async_main(config_path, args))

    except FileNotFoundError as e:
        print(f"Configuration error: {str(e)}")
        return {"error": f"Configuration error: {str(e)}"}

    except Exception as e:
        print(f"Critical error: {str(e)}")
        return {"error": f"Critical error: {str(e)}"}


if __name__ == "__main__":
    try:
        # Run main function and get result
        report = main()

        # Display result summary
        if report:
            if isinstance(report, dict) and "error" not in report:
                print(f"\nTrading Results Summary:")
                print(f"-------------------------")
                for key, value in report.items():
                    if key not in ["trades", "equity_curve"] and not key.startswith("_"):
                        print(f"{key}: {value}")
            elif isinstance(report, dict) and "error" in report:
                print(f"\nError occurred: {report['error']}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
        
    except Exception as e:
        print(f"Critical error: {str(e)}")
        sys.exit(1)