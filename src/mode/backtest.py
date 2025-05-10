#!/usr/bin/env python3
# src/trading/backtest.py

import traceback
from typing import Dict, Optional, Any, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from tqdm import tqdm
import time

from src.common.config_manager import ConfigManager
from src.common.helpers import TimeUtils
from src.common.abstract_factory import register_factory_class
from src.backtest.factory import get_backtest_engine_factory
from src.mode.base import BaseTradingMode, PerformanceCategory
from src.common.helpers import time_func


@register_factory_class('trading_mode_factory', 'backtest', 
    description="Historical data backtesting trading mode",
    features=["historical_data", "performance_analysis", "fast_execution"],
    category="simulation")
class BacktestMode(BaseTradingMode):
    """Trading mode for backtesting strategies on historical data"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize backtest mode"""
        super().__init__(config, params)
        
        self.historical_data = {}
        self.start_date = None
        self.end_date = None
        self.show_progress = True
        self.backtest_engine = None
        self.backtest_engine_factory = get_backtest_engine_factory(config)
        
        # Get engine type from system configuration
        self.engine_name = self.config.get("system", "backtest_mode", default="ohlcv")
        
        # Enhanced performance tracking
        self.detailed_performance_tracking = self.config.get(
            "backtest", "detailed_performance_tracking", default=True
        )
        
        # Performance metrics specific to backtest
        self.engine_timing = {
            'data_preparation': 0.0,
            'engine_execution': 0.0,
            'results_processing': 0.0,
            'total_runtime': 0.0
        }
    
    @time_func("backtest_initialize")
    async def _initialize_mode_specific(self) -> None:
        """Initialize backtest-specific components with engine from config"""
        # Get strategy from config
        strategy_name = self.config.get("backtest", "strategy", default=None)
        if not strategy_name:
            strategy_name = self.config.get("strategy", "active", default="dual_ma")
        
        # Get strategy parameters from config
        strategy_params = self.config.get("backtest", "strategy_params", default={})
        if not strategy_params:
            strategy_params = self.config.get("strategy", "parameters", default={})
        
        # Get engine parameters from config
        engine_params = {
            'strategy': strategy_name,
            'strategy_params': strategy_params,
            'initial_capital': self.config.get("trading", "capital", "initial", default=100000),
        }
        
        # Add any other engine-specific parameters from config
        engine_config_params = self.config.get("backtest", "engines", self.engine_name, default={})
        for key, value in engine_config_params.items():
            if key not in engine_params:
                engine_params[key] = value
                
        # Create the engine using factory
        start_time = time.time()
        self.backtest_engine = await self.backtest_engine_factory.create_backtest_engine(self.engine_name)
        engine_init_time = time.time() - start_time
        self.logger.info(f"Backtest engine initialized: {self.engine_name} (took {engine_init_time:.4f}s)")
        
        # Connect portfolio to engine
        if hasattr(self.backtest_engine, 'set_portfolio') and callable(getattr(self.backtest_engine, 'set_portfolio')):
            self.backtest_engine.set_portfolio(self.portfolio)
            self.logger.debug("Portfolio connected to backtest engine")
        
        # Connect strategy to engine
        if hasattr(self.backtest_engine, 'set_strategy') and callable(getattr(self.backtest_engine, 'set_strategy')):
            self.backtest_engine.set_strategy(self.strategy)
            self.logger.debug("Strategy connected to backtest engine")
            
        # Connect performance analyzer to portfolio for tracking
        if self.performance_analyzer and self.portfolio:
            self.portfolio.performance_analyzer = self.performance_analyzer
            self.logger.debug("Connected performance analyzer to portfolio")
            
        # Also share the performance analyzer with the backtest engine
        if self.performance_analyzer and hasattr(self.backtest_engine, 'set_performance_analyzer'):
            self.backtest_engine.set_performance_analyzer(self.performance_analyzer)
            self.logger.debug("Connected performance analyzer to backtest engine")
        
        # Configure backtest parameters
        self.show_progress = self.config.get("backtest", "show_progress", default=True)
        
        # Parse date range from config
        date_config = self.config.get("backtest", "period", default={})
        self.start_date = TimeUtils.parse_timestamp(date_config.get("start", None))
        self.end_date = TimeUtils.parse_timestamp(date_config.get("end", None))
            
        if not self.start_date:
            self.start_date = datetime.now() - timedelta(days=365)
        if not self.end_date:
            self.end_date = datetime.now()
            
        self.logger.info(f"Backtest period: {self.start_date} to {self.end_date}")
    
    @time_func("prepare_backtest_data")
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Prepare backtest data"""
        self.logger.info(f"Loading historical data for {len(symbols)} symbols")
        
        data_load_start = time.time()
        
        # Load all historical data
        for symbol in symbols:
            try:
                # Track individual symbol loading time
                symbol_load_start = time.time()
                
                df = await self.data_manager.get_historical_data(
                    symbol, 
                    timeframe, 
                    start=self.start_date,
                    end=self.end_date
                )
                
                symbol_load_time = time.time() - symbol_load_start
                
                if df.empty:
                    self.logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Ensure datetime column is datetime type
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    
                # Store data
                self.historical_data[symbol] = df
                self.logger.info(f"Loaded {len(df)} bars for {symbol} (took {symbol_load_time:.4f}s)")
                
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.DATA_FETCH,
                        symbol_load_time
                    )
                
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                
        total_load_time = time.time() - data_load_start
        self.logger.info(f"Total data loading time: {total_load_time:.4f}s")
        
        self.engine_timing['data_preparation'] = total_load_time
        
        if not self.historical_data:
            raise ValueError("No historical data loaded for any symbol")
    
    @time_func("execute_backtest")
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute the backtest using the backtest engine"""
        self.logger.info(f"Starting backtest execution using engine: {self.engine_name}")
        
        # Record start time for total execution
        total_start_time = time.time()
        
        # Report initial progress
        self._report_progress(10, f"Starting backtest using {self.engine_name} engine")
        
        # Set up progress display if enabled
        if self.show_progress:
            # Initialize progress bar for data preparation
            total_symbols = len(symbols)
            with tqdm(total=total_symbols, desc="Preparing data") as pbar:
                # Prepare the data in the backtest engine
                data_prep_start = time.time()
                sufficient_data = await self.backtest_engine.prepare_data(self.historical_data)
                data_prep_time = time.time() - data_prep_start
                self.engine_timing['data_preparation'] += data_prep_time
                
                pbar.update(total_symbols)
                
                # Report progress
                self._report_progress(30, f"Data preparation completed (took {data_prep_time:.4f}s)")
                
                if not sufficient_data:
                    self.logger.warning("Insufficient data for backtest")
                    self._report_progress(100, "Backtest aborted: insufficient data")
                    return {"status": "failed", "reason": "insufficient_data"}
        else:
            # Prepare the data without progress display
            data_prep_start = time.time()
            sufficient_data = await self.backtest_engine.prepare_data(self.historical_data)
            data_prep_time = time.time() - data_prep_start
            self.engine_timing['data_preparation'] += data_prep_time
            
            self._report_progress(30, f"Data preparation completed (took {data_prep_time:.4f}s)")
            
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                self._report_progress(100, "Backtest aborted: insufficient data")
                return {"status": "failed", "reason": "insufficient_data"}
        
        # Run the backtest
        try:
            self._report_progress(40, "Starting backtest execution")
            
            # Execute the backtest with performance tracking
            engine_exec_start = time.time()
            result = await self.backtest_engine.run_backtest(self.historical_data)
            engine_exec_time = time.time() - engine_exec_start
            self.engine_timing['engine_execution'] = engine_exec_time
            # Check if we got partial results due to error
            if result.get('status') == 'partial':
                self.logger.warning("Received partial results due to error during backtest")
                self._report_progress(70, f"Backtest partially completed with errors, processing available results")
            else:
                self._report_progress(70, f"Backtest execution completed in {engine_exec_time:.4f}s, processing results")
            
            self._report_progress(70, f"Backtest execution completed in {engine_exec_time:.4f}s, processing results")
            
            # Process results with performance tracking
            processing_start = time.time()
            
            # Update state with results from engine
            if 'equity_curve' in result:
                self.state['equity_curve'] = result['equity_curve']
            
            if 'trades' in result:
                self.state['trades'] = result['trades']
                
                # Update portfolio with trades
                trades_record_start = time.time()
                await self.portfolio.record_batch_trades(result['trades'])
                trades_record_time = time.time() - trades_record_start
                
                if self.detailed_performance_tracking:
                    self.logger.debug(f"PERF: Recording {len(result['trades'])} trades took {trades_record_time:.4f}s")
                
                # Report progress
                self._report_progress(80, f"Processed {len(result['trades'])} trades")
            
            # Add metadata to result
            result.update({
                "status": "completed",
                "symbols": list(self.historical_data.keys()),
                "timeframe": timeframe,
                "strategy": self.strategy.__class__.__name__,
                "engine": self.backtest_engine.__class__.__name__,
                "engine_type": self.engine_name,
                "start_date": self.start_date,
                "end_date": self.end_date
            })
            
            processing_time = time.time() - processing_start
            self.engine_timing['results_processing'] = processing_time
            
            # Calculate total runtime
            total_runtime = time.time() - total_start_time
            self.engine_timing['total_runtime'] = total_runtime
            
            # Add performance timing to result
            if self.detailed_performance_tracking:
                result['performance_timing'] = self.engine_timing
                
                # Log performance summary
                self.logger.info(f"Backtest performance summary:")
                self.logger.info(f"  Data preparation: {self.engine_timing['data_preparation']:.4f}s")
                self.logger.info(f"  Engine execution: {self.engine_timing['engine_execution']:.4f}s")
                self.logger.info(f"  Results processing: {self.engine_timing['results_processing']:.4f}s")
                self.logger.info(f"  Total runtime: {self.engine_timing['total_runtime']:.4f}s")
            
            # Final progress report
            self._report_progress(100, f"Backtest completed successfully in {total_runtime:.2f}s with {len(result.get('trades', []))} trades")
            
            self.logger.info(f"Backtest completed successfully using {self.engine_name} engine in {total_runtime:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during backtest execution: {e}", exc_info=True)
            self._report_progress(100, f"Backtest failed: {str(e)}")
            
            # Return a structured error result
            return {
                "status": "failed",
                "reason": str(e),
                "symbols": list(self.historical_data.keys()),
                "timeframe": timeframe,
                "engine_type": self.engine_name,
                "error_details": {
                    "exception_type": type(e).__name__,
                    "traceback": traceback.format_exc()
                }
            }
        
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add backtest-specific metrics to report"""
        # Calculate annualized return
        if self.start_date and self.end_date:
            days = (self.end_date - self.start_date).days
            if days > 0 and 'total_return_pct' in report:
                annualized_return = (1 + report['total_return_pct']/100) ** (365/days) - 1
                report['annualized_return_pct'] = annualized_return * 100
                
        # Add backtest parameters
        report['backtest_params'] = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'symbols': list(self.historical_data.keys()),
            'strategy': self.strategy.__class__.__name__,
            'engine': self.backtest_engine.__class__.__name__,
            'engine_type': self.engine_name,
            'initial_capital': self.config.get("trading", "capital", "initial", default=100000)
        }
        
        # Add performance timing metrics
        if self.detailed_performance_tracking:
            report['performance_timing'] = self.engine_timing
        
        # Include engine-specific metrics if available
        if hasattr(self.backtest_engine, 'get_metrics') and callable(getattr(self.backtest_engine, 'get_metrics')):
            engine_metrics = self.backtest_engine.get_metrics()
            report.update(engine_metrics)
        
        # Add engine-specific parameters if available
        engine_config = self.config.get("backtest", "engines", self.engine_name, default={})
        if engine_config:
            report['engine_config'] = engine_config
    
    @time_func("get_available_engines")
    async def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available backtest engines with their metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of engine names with metadata
        """
        return self.backtest_engine_factory.get_available_engines()
    
    @time_func("set_engine")
    async def set_engine(self, engine_name: str) -> None:
        """
        Set the backtest engine to use
        
        Args:
            engine_name: Name of the engine to use
        """
        # Validate engine exists
        available_engines = await self.get_available_engines()
        if engine_name not in available_engines:
            self.logger.error(f"Engine {engine_name} not found in available engines")
            raise ValueError(f"Invalid engine name: {engine_name}")
        
        # Set engine name
        self.engine_name = engine_name
        self.logger.info(f"Set backtest engine to: {engine_name}")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        shutdown_start = time.time()
        
        # First call parent implementation
        await super().shutdown()
        
        # Close backtest engine
        if self.backtest_engine:
            try:
                if hasattr(self.backtest_engine, 'shutdown'):
                    await self.backtest_engine.shutdown()
                elif hasattr(self.backtest_engine, 'close'):
                    await self.backtest_engine.close()
            except Exception as e:
                self.logger.error(f"Error shutting down backtest engine: {e}")
            
        shutdown_time = time.time() - shutdown_start
        self.logger.info(f"BacktestMode shutdown complete (took {shutdown_time:.4f}s)")
        
        # Log final performance summary if tracking is enabled
        if self.detailed_performance_tracking:
            self.logger.info("Final backtest performance summary:")
            for category, time_value in self.engine_timing.items():
                self.logger.info(f"  {category}: {time_value:.4f}s")