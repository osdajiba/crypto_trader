#!/usr/bin/env python3
# src/mode/base.py

from abc import ABC, abstractmethod
import asyncio
from enum import Enum
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
import time

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import time_func, get_performance_metrics, reset_performance_metrics
from src.portfolio.manager import PortfolioManager
from src.datasource.manager import DataManager
from src.strategy.factory import get_strategy_factory
from src.strategy.performance.factory import get_analyzer_factory


class TradingModeError(Exception):
    """Base exception for trading mode errors"""
    pass


class TradingModeInitError(TradingModeError):
    """Error during trading mode initialization"""
    pass


class TradingModeExecutionError(TradingModeError):
    """Error during trading mode execution"""
    pass


class PerformanceCategory(Enum):
    """Performance tracking categories"""
    DATA_FETCH = "data_fetch"
    SIGNAL_GENERATION = "signal_generation"
    TRADE_EXECUTION = "trade_execution"
    PORTFOLIO_UPDATE = "portfolio_update"
    RISK_MANAGEMENT = "risk_management"
    TOTAL_ITERATION = "total_iteration"


class BaseTradingMode(ABC):
    """Base class for all trading modes"""
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize trading mode"""
        self.config = config
        self.params = params or {}
        
        self.logger = LogManager.get_logger(f"mode.{self.__class__.__name__.lower()}")
        self._running = False
        
        # Initialize progress callback
        self._progress_callback = None
        
        # Component factories
        self.performance_factory = get_analyzer_factory(config)
        self.strategy_factory = get_strategy_factory(config)
        
        # Components will be initialized later
        self.portfolio = None
        self.data_manager = None
        self.performance_analyzer = None
        self.strategy = None
        
        # Common state tracking for reporting and analysis
        self.state = {
            'timestamp': None,
            'equity_curve': [],
            'market_prices': {}
        }
        
        # Performance tracking
        self.performance_metrics = {category.value: [] for category in PerformanceCategory}
        self.performance_summaries = {}
        self.enable_performance_tracking = self.config.get(
            "system", "performance", "enable_tracking", default=True
        )
        self.performance_log_interval = self.config.get(
            "system", "performance", "log_interval", default=10
        )
        self._iteration_count = 0
        
        self.logger.info(f"Creating {self.mode_name} trading mode instance")
        
    @time_func("initialize")
    async def initialize(self) -> None:
        """Initialize trading mode components"""
        try:
            # Initialize portfolio manager, and data manager 
            source_type = {"live": "exchange", "paper": "hybrid", "backtest": "local"}.get(self.mode_name, "local") # use local data default source
            self.data_manager = DataManager(self.config, source_type)
            await self.data_manager.initialize()
            
            self.portfolio = PortfolioManager(self.config)
            await self.portfolio.initialize()
                   
            # Create performance analyzer
            self.performance_analyzer = await self.performance_factory.create_performance_analyzer(self.mode_name)
            self.logger.info("Performance analyzer initialized")
            
            # Create strategy instance
            strategy_name = self.params.get("strategy_name") or self.config.get("live_trading", "strategy", default=None)
            strategy_params = self.params.get("strategy_params") or self.config.get("live_trading", "strategy_params", default={})
            self.strategy = await self.strategy_factory.create_strategy(strategy_name, strategy_params)
            self.logger.info(f"Strategy initialized: {self.strategy.__class__.__name__}")
            
            # Mode-specific initialization
            await self._initialize_mode_specific()
            
            # Reset performance metrics
            reset_performance_metrics()
            
            # Mark initialization complete
            self._running = True
            self.logger.info(f"{self.mode_name} mode initialization complete")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.mode_name} mode: {e}")
            raise TradingModeInitError(f"Trading mode initialization failed: {str(e)}")

    @abstractmethod
    async def _initialize_mode_specific(self) -> None:
        """Initialize mode-specific components"""
        pass
    
    @property
    def mode_name(self) -> str:
        """Get mode name derived from class name"""
        return self.__class__.__name__.replace('Mode', '').lower()
    
    @property
    def is_running(self) -> bool:
        """Check if the mode is currently running"""
        return self._running
    
    @time_func("run_trading_mode")
    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Run the trading mode - Template Method pattern implementation"""
        self.logger.info(f"Starting {self.mode_name} mode | Symbols: {symbols} | Timeframe: {timeframe}")
        
        try:
            # 1. Pre-run preparation
            self._init_state()
            await self._prepare_run(symbols, timeframe)
            
            # 2. Main trading loop
            result = await self._execute_trading_loop(symbols, timeframe)
            
            # 3. Generate report with performance analyzer
            report = self.performance_analyzer.generate_performance_report()
            
            # 4. Add mode-specific metrics
            self._add_mode_specific_metrics(report)
            
            # 5. Add performance tracking metrics
            if self.enable_performance_tracking:
                report['performance_tracking'] = self._get_performance_summary()
            
            return report
            
        except asyncio.CancelledError:
            self.logger.warning(f"{self.mode_name} mode cancelled")
            return {"status": "cancelled"}
            
        except Exception as e:
            self.logger.error(f"{self.mode_name} mode error: {e}", exc_info=True)
            raise TradingModeExecutionError(f"Trading mode execution failed: {str(e)}")
            
        finally:
            await self.shutdown()
    
    def _init_state(self) -> None:
        """Initialize the trading state with default values"""
        initial_capital = self.config.get("trading", "capital", "initial", default=100000)
        self.state.update({
            'timestamp': None,
            'equity_curve': [],
            'market_prices': {},
            'peak_equity': initial_capital,
            'max_drawdown': 0.0
        })
        
        self.logger.info(f"State initialized with {initial_capital} initial capital")
    
    @abstractmethod
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Mode-specific preparation steps"""
        self._report_progress(10, f"Preparing {self.mode_name} mode for {len(symbols)} symbols")
        return {"status": "not_implemented"}
    
    @abstractmethod
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute the main trading loop"""
        self._report_progress(20, f"Starting execution for {len(symbols)} symbols with {timeframe} timeframe")
    
    @time_func("fetch_market_data")
    async def _fetch_market_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for symbols based on the current trading mode
        
        Args:
            symbols: List of symbols to fetch data for
            timeframe: Timeframe for candles
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their market data
        """
        # Track performance
        start_time = time.time()
        
        if not self.data_manager:
            self.logger.error("Data source not initialized")
            return {}
        
        try:
            # Get market data based on trading mode
            if self.mode_name in ["live", "paper"]:
                # For live and paper trading, get real-time data
                data_map = {}
                for symbol in symbols:
                    df = await self.data_manager.get_latest_data_for_symbols(symbol, timeframe)
                    if not df.empty:
                        data_map[symbol] = df
                        
                # Record performance metric
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.DATA_FETCH,
                        time.time() - start_time
                    )
                        
                return data_map
                
            else:  # backtest mode
                # For backtest, get the next batch of historical data
                # This is typically handled by a specific implementation in the backtest mode
                return {}
        
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {}
    
    @time_func("process_market_data")
    async def _process_market_data(self, data_map: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Process market data and generate signals"""
        executed_trades = []
        
        if not data_map or all(df.empty for df in data_map.values()):
            self.logger.warning("No market data to process")
            return executed_trades
        
        # Update current time from data
        self._update_timestamp_from_data(data_map)
        
        # Update market prices in state
        self._update_market_prices(data_map)
        
        # Update portfolio with latest market data
        portfolio_update_start = time.time()
        await self.portfolio.update_market_data(data_map)
        if self.enable_performance_tracking:
            self._record_performance_metric(
                PerformanceCategory.PORTFOLIO_UPDATE,
                time.time() - portfolio_update_start
            )
        
        # Process data for each symbol
        for symbol, data in data_map.items():
            if data.empty:
                self.logger.warning(f"No data for {symbol}")
                continue
                
            # Generate signals from strategy
            signal_gen_start = time.time()
            signals = await self.strategy.process_data(data, symbol)
            if self.enable_performance_tracking:
                self._record_performance_metric(
                    PerformanceCategory.SIGNAL_GENERATION,
                    time.time() - signal_gen_start
                )
            
            if not signals.empty:
                # Pass signals to portfolio for validation, risk management and execution
                trade_exec_start = time.time()
                trades = await self.portfolio.process_signals(signals, data)
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.TRADE_EXECUTION,
                        time.time() - trade_exec_start
                    )
                    
                if trades:
                    executed_trades.extend(trades)
        
        # Update equity curve and drawdown
        self._update_performance_metrics()
        
        # Increment iteration counter
        self._iteration_count += 1
        
        # Log performance summary at intervals
        if self.enable_performance_tracking and self._iteration_count % self.performance_log_interval == 0:
            self._log_performance_summary()
        
        return executed_trades

    def _update_timestamp_from_data(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """Update current timestamp from data"""
        for df in data_map.values():
            if not df.empty and 'timestamp' in df.columns:
                latest_time = df['timestamp'].iloc[-1]
                self.state['timestamp'] = latest_time
                break
    
    def _update_market_prices(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """Update current market prices in state"""
        for symbol, df in data_map.items():
            if not df.empty and 'close' in df.columns:
                self.state['market_prices'][symbol] = df['close'].iloc[-1]
    
    def _update_performance_metrics(self) -> None:
        """Update equity curve and drawdown"""
        # Get current equity value from portfolio
        equity = self.portfolio.get_total_value()
        
        # Record equity point
        equity_point = {
            'timestamp': self.state['timestamp'],
            'equity': equity
        }
        self.state['equity_curve'].append(equity_point)
        
        # Update peak equity
        if equity > self.state.get('peak_equity', 0):
            self.state['peak_equity'] = equity
        
        # Calculate drawdown
        if self.state.get('peak_equity', 0) > 0:
            drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
            self.state['max_drawdown'] = max(self.state.get('max_drawdown', 0), drawdown)
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add mode-specific metrics to report"""
        pass
    
    def _should_continue(self) -> bool:
        """Check if the mode should continue running"""
        # Check if we should continue based on portfolio risk status
        return self._running and not self.portfolio.is_risk_breached()
    
    async def _sleep_interval(self, interval: Optional[int] = None) -> None:
        """Wait for the specified time"""
        if interval is None:
            interval = self.config.get("data", "polling_interval", default=60)
        
        if interval > 0:
            await asyncio.sleep(interval)
    
    def _record_performance_metric(self, category: PerformanceCategory, duration: float) -> None:
        """
        Record a performance metric
        
        Args:
            category: Performance category
            duration: Time duration in seconds
        """
        if not self.enable_performance_tracking:
            return
            
        self.performance_metrics[category.value].append(duration)
    
    def _get_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get a summary of all performance metrics
        
        Returns:
            Dict with categories as keys and statistics as values
        """
        # Get the function-level performance metrics
        function_metrics = get_performance_metrics()
        
        # Calculate statistics for category metrics
        category_metrics = {}
        for category, times in self.performance_metrics.items():
            if not times:
                continue
                
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            total_time = sum(times)
            call_count = len(times)
            
            category_metrics[category] = {
                'avg_time': avg_time,
                'max_time': max_time,
                'min_time': min_time,
                'total_time': total_time,
                'call_count': call_count
            }
        
        # Combine both types of metrics
        return {
            'categories': category_metrics,
            'functions': function_metrics
        }
    
    def _log_performance_summary(self) -> None:
        """Log a summary of performance metrics"""
        if not self.enable_performance_tracking:
            return
            
        summary = self._get_performance_summary()
        
        # Log category summaries
        self.logger.info("==== Performance Metrics Summary ====")
        for category, metrics in summary['categories'].items():
            if 'avg_time' in metrics:
                self.logger.info(f"{category}: avg={metrics['avg_time']:.4f}s, "
                               f"max={metrics['max_time']:.4f}s, "
                               f"calls={metrics['call_count']}")
        
        # Log most important function metrics
        self.logger.info("==== Function Performance ====")
        for func_name, metrics in summary['functions'].items():
            if 'avg_time' in metrics and metrics['avg_time'] > 0.01:  # Only log significant functions
                self.logger.info(f"{func_name}: avg={metrics['avg_time']:.4f}s, "
                               f"max={metrics['max_time']:.4f}s, "
                               f"calls={metrics['call_count']}")
        
        self.logger.info("====================================")
            
    def set_progress_callback(self, callback: callable) -> None:
        """
        Set a callback for progress updates during execution.
        
        Args:
            callback: Function taking (percentage, message) parameters
        """
        self._progress_callback = callback
        self.logger.debug("Progress callback registered in trading mode")
        
    def _report_progress(self, percent: float, message: str) -> None:
        """
        Report progress using the registered callback.
        
        Args:
            percent: Progress percentage (0-100)
            message: Progress message
        """
        if hasattr(self, '_progress_callback') and self._progress_callback:
            try:
                self._progress_callback(percent, message)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
        else:
            # Fallback to logging if no callback is set
            self.logger.info(f"Progress: {percent:.1f}% - {message}")
            
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown mode specific components"""
        self._running = False
        
        try:
            if self.portfolio:
                await self.portfolio.close()
            if self.data_manager:
                await self.data_manager.close()
            if self.strategy:
                await self.strategy.shutdown()
            if self.performance_analyzer:
                await self.performance_analyzer.shutdown()                
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")