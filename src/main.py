#!/usr/bin/env python3
# src/main.py - Unified entry point for the trading system

import os
import sys
import signal
import traceback
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from core.core import TradingCore

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.config import ConfigManager
from common.log_manager import LogManager, LogInitializer
from common.cli import *


class TradingSystemLauncher:
    """Unified launcher for trading system with CLI interface"""
    
    def __init__(self):
        """Initialize the launcher with default values"""
        self.args = None
        self.config = None
        self.logger = None
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Set up signal handlers for graceful termination"""
        signal.signal(signal.SIGINT, self.handle_sigint)
    
    def handle_sigint(self, sig, frame):
        """Handle keyboard interrupt gracefully"""
        print("\nOperation cancelled by user.")
        sys.exit(130)
    
    def setup_argument_parser(self):
        """Create and configure the command line argument parser"""
        parser = argparse.ArgumentParser(description="Trading System")
        
        # Configuration options
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
        
        # Backtest options
        parser.add_argument("--start-date", type=str, 
                        help="Start date for backtesting (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, 
                        help="End date for backtesting (YYYY-MM-DD)")
        parser.add_argument("--backtest-engine", type=str,
                        choices=["ohlcv", "market_replay"],
                        help="Backtest engine to use")
        parser.add_argument("--commission", type=float,
                        help="Commission rate (e.g., 0.001 for 0.1%)")
        parser.add_argument("--slippage", type=float,
                        help="Slippage (e.g., 0.001 for 0.1%)")
        parser.add_argument("--capital", type=float,
                        help="Initial capital")
        
        # Live trading options
        parser.add_argument("--api-key", type=str,
                        help="Exchange API key")
        parser.add_argument("--api-secret", type=str,
                        help="Exchange API secret")
        
        # System options
        parser.add_argument("--debug", action="store_true", 
                        help="Enable debug logging")
        parser.add_argument("--verbose", "-v", action="count", default=0, 
                        help="Increase verbosity")
        parser.add_argument("--db-url", type=str, 
                        help="Database connection URL")
        parser.add_argument("--max-workers", type=int, 
                        help="Async worker threads")
        
        return parser
    
    def prompt_mode_selection(self) -> str:
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

    def prompt_backtest_engine_selection(self) -> str:
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
    
    def setup_environment(self):
        """Setup environment with config and logging"""
        # Constants
        base_dir = Path(project_root)
        default_config_path = base_dir / "conf/config.yaml"
        
        # Handle config path
        if hasattr(self.args, 'config') and self.args.config and isinstance(self.args.config, str):
            config_path = Path(self.args.config)
        else:
            config_path = default_config_path
        
        # Initialize config manager
        config_manager = ConfigManager(str(config_path))
        
        # Load configuration
        try:
            config_manager.load()
            self.config = config_manager
        except Exception as e:
            print(f"Warning: Failed to load config: {e}")
            print(f"Using default configuration.")
            self.config = config_manager
        
        # Initialize logging
        log_init = LogInitializer(self.config)
        log_init.initialize()
        
        # Get logger for main module
        self.logger = LogManager.get_logger("trading_system")
        
        # Override log level if debug flag is set
        if self.args.debug:
            LogManager.set_level("DEBUG")
            self.logger.debug("Debug logging enabled")
        
        self.logger.info(f"Environment setup complete. Using config: {config_path}")
        return config_path
    
    def run_trading_core(self):
        """Initialize and run the trading core"""
        
        self.logger.info(f"Starting trading system in {self.args.mode} mode")
        
        # Initialize the trading core
        trading_core = TradingCore(
            config=self.config,
            mode=self.args.mode,
            backtest_engine=getattr(self.args, 'backtest_engine', None)
        )
        
        # Create and run event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run trading pipeline
            result = loop.run_until_complete(trading_core.run_pipeline())
            
            # Ensure proper shutdown
            loop.run_until_complete(trading_core.shutdown())
            
            return result
        finally:
            loop.close()
    
    def run(self):
        """Main entry point for the trading system"""
        try:
            # Parse command line arguments
            parser = self.setup_argument_parser()
            self.args = parser.parse_args()
            
            # Setup environment (config and logging)
            config_path = self.setup_environment()
            
            # Interactive mode - prompt for missing critical options
            if not self.args.mode:
                self.args.mode = self.prompt_mode_selection()
                
            if self.args.mode == "backtest" and not getattr(self.args, 'backtest_engine', None):
                self.args.backtest_engine = self.prompt_backtest_engine_selection()
            
            # Initialize progress display
            progress_display = setup_progress_display(self.args.verbose)
            
            # Run the trading core
            result = self.run_trading_core()
            
            # Display results
            display_results(result)
            
            self.logger.info("Trading system execution completed")
            return result
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            if self.logger:
                self.logger.warning("Operation cancelled by user")
            return {"status": "cancelled"}
            
        except Exception as e:
            error_msg = f"Critical error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            
            if self.logger:
                self.logger.error(error_msg, exc_info=True)
                
            return {"error": error_msg}


def main():
    """Application entry point"""
    launcher = TradingSystemLauncher()
    result = launcher.run()
    return result


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