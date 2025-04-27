#!/usr/bin/env python3
# src/main.py - Combined launcher and main entry point for CLI-only mode

import os
import sys
import signal
import traceback
import argparse
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import local cli_runner module
from cli_runner import run_cli_mode

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the launcher"""
    parser = argparse.ArgumentParser(description="Trading System CLI")
    
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
                      choices=["ohlcv", "market_replay"],
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
    
    # Handle config path with OS compatibility
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = default_config_path
    
    from src.common.config import ConfigManager
    config_manager = ConfigManager(config_path)
    
    # Load configuration
    try:
        config_manager.load()
    except Exception as e:
        print(f"Warning: Failed to load config: {e}")
        print(f"Using default configuration.")
    
    # Setup logging based on config
    from src.common.log_manager import LogManager, LogInitializer
    log_init = LogInitializer(config_manager)
    log_init.initialize()
    
    logger = LogManager.get_logger("trading_system")
    
    # Override log level if debug flag is set
    if args.debug:
        LogManager.set_level("DEBUG")
        logger.debug("Debug logging enabled")
    
    # Store config manager in args
    args.config = config_manager
    args.config_path = config_path
    
    return logger, config_manager

def handle_sigint(sig, frame):
    """Handle keyboard interrupt gracefully"""
    print("\nOperation cancelled by user.")
    sys.exit(130)

def main():
    """
    Main entry point - processes command line arguments and launches the CLI mode
    """
    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_sigint)
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Setup environment
        logger, config_manager = setup_environment(args)
        
        # If no trading mode specified, prompt for selection
        if not args.mode:
            args.mode = prompt_mode_selection()
            
        # For backtest mode, prompt for engine type if not specified
        if args.mode == "backtest" and not args.backtest_engine:
            args.backtest_engine = prompt_backtest_engine_selection()
            
        # Run in CLI mode
        result = run_cli_mode(args, args.config_path, config_manager, logger)
        return result
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return {"status": "cancelled"}
    except Exception as e:
        print(f"Critical error: {str(e)}")
        traceback.print_exc()
        return {"error": f"Critical error: {str(e)}"}

if __name__ == "__main__":
    try:
        result = main()
        exit_code = 0 if not result or "error" not in result else 1
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Unhandled error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)