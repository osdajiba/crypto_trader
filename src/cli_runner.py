#!/usr/bin/env python3
# src/ui/cli_runner.py

import os
import sys
import asyncio
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
from tqdm import tqdm  # Make sure to install this with pip install tqdm

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = str(Path(current_dir).parent)
sys.path.append(project_root)

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor


class CliProgressBar:
    """Simple progress bar for CLI mode"""
    
    def __init__(self, total=100, desc="Progress"):
        """Initialize the progress bar"""
        self.pbar = tqdm(total=total, desc=desc, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        self.last_percent = 0
        
    def update(self, percent, message):
        """Update the progress bar"""
        delta = int(percent) - self.last_percent
        if delta > 0:
            self.pbar.set_description(message)
            self.pbar.update(delta)
            self.last_percent = int(percent)
            
    def close(self):
        """Close the progress bar"""
        self.pbar.close()


class CliRunner:
    """Enhanced CLI Runner for Trading System"""
    
    def __init__(self, args: argparse.Namespace, config_path: Path, config: ConfigManager, logger):
        """Initialize with configuration and arguments"""
        self.args = args
        self.config_path = config_path
        self.config = config
        self.logger = logger

        # Pass the logger explicitly to AsyncExecutor to avoid LogManager initialization issues
        self.executor = AsyncExecutor(logger=logger)
        self.progress_bar = None
        
    async def run(self) -> Dict[str, Any]:
        """Run the trading system in CLI mode"""
        try:
            self._apply_arg_overrides()
            
            # Import the core module
            try:
                from src.core.core import TradingCore
            except ImportError as e:
                self.logger.error(f"Failed to import TradingCore: {e}")
                return {"error": f"Failed to import required modules: {e}"}
            
            # Create progress bar
            self.progress_bar = CliProgressBar(desc="Trading Pipeline")
            
            # Create core instance with specified mode and engine
            mode = self.args.mode
            backtest_engine = getattr(self.args, 'backtest_engine', None)
            self.logger.info(f"Starting {mode} mode" + (f" with {backtest_engine} engine" if backtest_engine else ""))
            
            # Initialize trading core
            trader = TradingCore(self.config, mode, backtest_engine)
            
            # Add progress callback if supported
            if hasattr(trader, 'set_progress_callback'):
                trader.set_progress_callback(self._update_progress)
            
            # Run the trading pipeline
            self.logger.info("Starting trading pipeline...")
            result = await trader.run_pipeline()
            
            # Close progress bar
            if self.progress_bar:
                self.progress_bar.close()
                
            # Display results
            self._display_results(result)
            
            # Shutdown properly
            await trader.shutdown()
            
            self.logger.info("Trading pipeline completed successfully")
            return result
            
        except asyncio.CancelledError:
            self.logger.warning("Trading pipeline cancelled")
            return {"status": "cancelled"}
            
        except Exception as e:
            self.logger.error(f"Error running trading pipeline: {e}", exc_info=True)
            return {"error": f"Error running trading pipeline: {e}"}
    
    def _apply_arg_overrides(self):
        """Apply command line argument overrides to config"""
        # Apply mode override
        if hasattr(self.args, 'mode') and self.args.mode:
            self.config.set("system", "operational_mode", self.args.mode)
            self.logger.info(f"Overriding operational mode: {self.args.mode}")

        # Apply backtest engine override
        if hasattr(self.args, 'backtest_engine') and self.args.backtest_engine:
            self.config.set("backtest", "engine_type", self.args.backtest_engine)
            self.logger.info(f"Overriding backtest engine: {self.args.backtest_engine}")

        # Apply strategy override
        if hasattr(self.args, 'strategy') and self.args.strategy:
            self.config.set("strategy", "active", self.args.strategy)
            self.logger.info(f"Overriding strategy: {self.args.strategy}")

        # Apply symbol override
        if hasattr(self.args, 'symbol') and self.args.symbol:
            # Handle multiple symbols if comma-separated
            symbols = [s.strip() for s in self.args.symbol.split(',')]
            self.config.set("trading", "instruments", symbols)
            self.logger.info(f"Overriding trading symbols: {symbols}")

        # Apply timeframe override
        if hasattr(self.args, 'timeframe') and self.args.timeframe:
            self.config.set("strategy", "timeframe", self.args.timeframe)
            self.logger.info(f"Overriding timeframe: {self.args.timeframe}")

        # Apply backtest date range overrides
        if hasattr(self.args, 'start_date') and self.args.start_date:
            self.config.set("backtest", "period", "start", self.args.start_date)
            self.logger.info(f"Overriding backtest start date: {self.args.start_date}")

        if hasattr(self.args, 'end_date') and self.args.end_date:
            self.config.set("backtest", "period", "end", self.args.end_date)
            self.logger.info(f"Overriding backtest end date: {self.args.end_date}")

        # Apply database URL override
        if hasattr(self.args, 'db_url') and self.args.db_url:
            self.config.set("database", "url", self.args.db_url)
            self.logger.info(f"Overriding database URL: {self.args.db_url}")

        # Apply max workers override
        if hasattr(self.args, 'max_workers') and self.args.max_workers:
            self.config.set("system", "performance", "max_threads", self.args.max_workers)
            self.logger.info(f"Overriding max threads: {self.args.max_workers}")
    
    def _update_progress(self, percent, message):
        """Update progress bar callback"""
        if self.progress_bar:
            self.progress_bar.update(percent, message)
    
    def _display_results(self, result):
        """Display results in a readable format"""
        if not result:
            print("No results to display")
            return
            
        # Print formatted results
        print("\n" + "="*80)
        print(" "*30 + "TRADING RESULTS")
        print("="*80)
        
        if isinstance(result, dict):
            # Check for error
            if "error" in result:
                print(f"\nERROR: {result['error']}")
                return
                
            # Display metrics
            if "metrics" in result:
                print("\nPERFORMANCE METRICS:")
                print("-" * 40)
                for key, value in result["metrics"].items():
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, float):
                        print(f"{formatted_key}: {value:.4f}")
                    else:
                        print(f"{formatted_key}: {value}")
            
            # Display summary statistics
            if "summary" in result:
                print("\nSUMMARY:")
                print("-" * 40)
                for key, value in result["summary"].items():
                    formatted_key = key.replace("_", " ").title()
                    if isinstance(value, float):
                        print(f"{formatted_key}: {value:.4f}")
                    else:
                        print(f"{formatted_key}: {value}")
            
            # Display trade statistics
            if "trades" in result:
                trades = result["trades"]
                win_trades = sum(1 for t in trades if t.get('pnl', 0) > 0)
                loss_trades = sum(1 for t in trades if t.get('pnl', 0) <= 0)
                
                print(f"\nTRADES SUMMARY: {len(trades)} total ({win_trades} wins, {loss_trades} losses)")
                
                # Show sample of trades (first 5)
                if trades:
                    print("\nSAMPLE TRADES:")
                    print("-" * 80)
                    print(f"{'DATE':<20} {'SYMBOL':<10} {'TYPE':<5} {'PRICE':<10} {'QUANTITY':<10} {'P&L':<10}")
                    print("-" * 80)
                    
                    for trade in trades[:5]:
                        date = trade.get('timestamp', 'N/A')
                        if isinstance(date, (int, float)):
                            date = datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M')
                        
                        print(f"{str(date):<20} {trade.get('symbol', 'N/A'):<10} "
                              f"{str(trade.get('action', 'N/A')):<5} {trade.get('price', 0):<10.2f} "
                              f"{trade.get('quantity', 0):<10.2f} {trade.get('pnl', 0):<10.2f}")
                    
                    if len(trades) > 5:
                        print(f"\n... and {len(trades) - 5} more trades")
            
            # Save detailed results to file
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = f"results_{timestamp}.json"
            
            try:
                with open(result_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nDetailed results saved to: {result_file}")
            except Exception as e:
                print(f"\nFailed to save detailed results: {e}")
        
        else:
            # Just print as string for non-dict results
            print(str(result))
        
        print("\n" + "="*80)


def run_cli_mode(args: argparse.Namespace, config_path: Path, config: ConfigManager, logger) -> Dict[str, Any]:
    """Run the trading system in CLI mode with the provided arguments"""
    # Create CLI runner
    cli_runner = CliRunner(args, config_path, config, logger)
    
    # Create and run event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(cli_runner.run())
        return result
    finally:
        loop.close()


if __name__ == "__main__":
    # For testing the CLI runner directly
    from src.common.config import ConfigManager
    
    parser = argparse.ArgumentParser(description="Trading System CLI")
    parser.add_argument("--mode", type=str, choices=["backtest", "paper", "live"], default="backtest")
    parser.add_argument("--backtest-engine", type=str, choices=["ohlcv", "market_replay"], default="ohlcv")
    parser.add_argument("--strategy", type=str, default="dual_ma")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str, default="2023-12-31")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    
    # Initialize config
    config_path = Path(project_root) / "conf" / "config.yaml"
    config = ConfigManager(str(config_path))
    
    # Set up logger
    from src.common.log_manager import LogManager, LogInitializer
    log_init = LogInitializer(config)
    log_init.initialize()
    logger = LogManager.get_logger("cli_runner")
    
    # Run CLI mode
    result = run_cli_mode(args, config_path, config, logger)
    print(f"Result: {result}")