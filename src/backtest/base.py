#!/usr/bin/env python3
# src/backtest/base.py

from abc import ABC, abstractmethod
import asyncio
from collections import deque
import pandas as pd
import time
from typing import Dict, Any, Optional, List, Set, Callable

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager


class BacktestEngineError(Exception):
    """Base exception for backtest engine errors"""
    pass


class BaseBacktestEngine(ABC):
    """
    Base class for backtest engines that defines a common interface and shared functionality.
    
    Backtest engines simulate trading on historical data with different approaches,
    but share common operations for data preparation, position tracking, and performance analysis.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the base backtest engine.
        
        Args:
            config: Configuration manager
            params: Engine parameters
        """
        self.config = config
        self.params = params or {}
        
        # Initialize logger with appropriate category
        self.logger = LogManager.get_logger(f"backtest.{self.__class__.__name__.lower()}")
        
        # Set state tracking
        self._is_initialized = False
        self._is_running = False
        
        # Data buffers for processing
        self.data_buffers = {}
        self.data_queues = {}
        self.has_sufficient_history = {}
        
        # Window size for data processing (different engines can use this differently)
        self.required_window_size = self.params.get('required_window_size', 0)
        
        # Strategy instance (set by trading mode)
        self.strategy = None
        
        # Portfolio reference (set by trading mode)
        self.portfolio = None
        
        # Set metrics tracking
        self.metrics = {}
        
        self.logger.info(f"Initializing {self.__class__.__name__} backtest engine")
    
    async def initialize(self) -> None:
        """
        Initialize the backtest engine.
        
        Should be called after creation to properly set up components.
        """
        if self._is_initialized:
            return
            
        try:
            # Initialize strategy if provided in params
            strategy_name = self.params.get('strategy')
            strategy_params = self.params.get('strategy_params', {})
            
            if strategy_name and 'strategy_factory' in self.params:
                strategy_factory = self.params['strategy_factory']
                self.strategy = await strategy_factory.create_strategy(strategy_name, strategy_params)
                self.logger.info(f"Initialized strategy: {strategy_name}")
            
            # Initialize engine-specific components
            await self._initialize_engine()
            
            self._is_initialized = True
            self.logger.info(f"{self.__class__.__name__} initialization complete")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            raise BacktestEngineError(f"Engine initialization failed: {str(e)}")
            
    async def _initialize_engine(self) -> None:
        """
        Initialize engine-specific components.
        
        Override in subclasses to provide custom initialization logic.
        """
        pass
    
    def set_strategy(self, strategy) -> None:
        """
        Set the trading strategy for the backtest.
        
        Args:
            strategy: Strategy instance
        """
        self.strategy = strategy
        self.logger.info(f"Set strategy: {self.strategy.__class__.__name__}")
    
    def set_portfolio(self, portfolio) -> None:
        """
        Set the portfolio for the backtest.
        
        Args:
            portfolio: Portfolio manager instance
        """
        self.portfolio = portfolio
        self.logger.info(f"Set portfolio manager")
    
    async def prepare_data(self, historical_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prepare historical data for backtesting.
        
        This base implementation validates data and checks for sufficient history.
        Subclasses may override to provide additional preprocessing.
        
        Args:
            historical_data: Dictionary mapping symbols to historical data DataFrames
            
        Returns:
            bool: True if data is sufficient, False otherwise
        """
        if not historical_data:
            self.logger.error("No historical data provided")
            return False
            
        # Create data buffers for each symbol
        for symbol, df in historical_data.items():
            # Skip empty dataframes
            if df.empty:
                self.logger.warning(f"Empty data for {symbol}, skipping")
                continue
                
            # Create buffer and queue
            self.data_buffers[symbol] = pd.DataFrame()
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
            
            # Calculate required history
            self.has_sufficient_history[symbol] = len(df) >= self.required_window_size
            
            if not self.has_sufficient_history[symbol] and self.required_window_size > 0:
                self.logger.warning(f"Insufficient history for {symbol}: {len(df)} < {self.required_window_size} required")
            else:
                self.logger.info(f"Prepared data for {symbol} with {len(df)} bars")
        
        # Check if we have sufficient data for all symbols if window size specified
        if self.required_window_size > 0:
            if not all(self.has_sufficient_history.values()):
                insufficient = [s for s, v in self.has_sufficient_history.items() if not v]
                self.logger.warning(f"Insufficient history for symbols: {', '.join(insufficient)}")
                return False
                
        self.logger.info("Data preparation complete")
        return True
    
    @abstractmethod
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol.
        
        Subclasses must implement this method to handle data point processing,
        signal generation, and potentially trade execution depending on the engine type.
        
        Args:
            data_point: DataFrame containing a single data point
            symbol: Symbol being processed
            
        Returns:
            pd.DataFrame: Generated signals, if any
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the backtest.
        
        Returns:
            Dict[str, Any]: Performance metrics dictionary
        """
        return self.metrics
    
    def set_performance_analyzer(self, performance_analyzer) -> None:
        """
        Set the performance analyzer for tracking metrics.
        
        Args:
            performance_analyzer: Performance analyzer instance
        """
        self.performance_analyzer = performance_analyzer
        self.logger.info(f"Set performance analyzer: {performance_analyzer.__class__.__name__}")
    
    @abstractmethod
    async def run_backtest(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run a backtest using the configured strategy and parameters.
        
        Args:
            historical_data: Dictionary mapping symbols to historical data DataFrames
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        pass
    
    def _get_data_at_timestamp(self, data: Dict[str, pd.DataFrame], timestamp) -> Dict[str, pd.DataFrame]:
        """
        Get data for all symbols at a specific timestamp
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            timestamp: Timestamp to fetch data for
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> data point at timestamp
        """
        result = {}
        
        for symbol, df in data.items():
            # Find data at timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                if timestamp in df.index:
                    result[symbol] = df.loc[[timestamp]]
            elif 'datetime' in df.columns:
                mask = df['datetime'] == timestamp
                if mask.any():
                    result[symbol] = df[mask]
            elif 'timestamp' in df.columns:
                mask = df['timestamp'] == timestamp
                if mask.any():
                    result[symbol] = df[mask]
        
        return result
        
    def calculate_performance_metrics(self, initial_capital: float, final_value: float, 
                                      baseline_start_price: Optional[float] = None, 
                                      baseline_end_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest
        
        Args:
            initial_capital: Initial capital
            final_value: Final portfolio value
            baseline_start_price: Optional starting price of baseline asset
            baseline_end_price: Optional ending price of baseline asset
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Calculate basic performance metrics
        total_return = final_value - initial_capital
        return_pct = (final_value / initial_capital - 1) * 100
        
        metrics = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': return_pct
        }
        
        # Add baseline comparison if available
        if baseline_start_price is not None and baseline_end_price is not None:
            baseline_return = (baseline_end_price - baseline_start_price) / baseline_start_price
            baseline_return_pct = baseline_return * 100
            alpha = return_pct - baseline_return_pct
            
            metrics.update({
                'baseline_return_pct': baseline_return_pct,
                'alpha': alpha,
                'baseline_start_price': baseline_start_price,
                'baseline_end_price': baseline_end_price
            })
            
        return metrics
    
    async def shutdown(self) -> None:
        """
        Clean up resources.
        """
        self._is_running = False
        
        # Clear data structures
        self.data_buffers.clear()
        self.data_queues.clear()
        self.has_sufficient_history.clear()
        
        self.logger.info(f"{self.__class__.__name__} shutdown complete")