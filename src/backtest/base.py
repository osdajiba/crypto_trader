#!/usr/bin/env python3
# src/backtest/engine/base.py

"""
Base backtest engine implementation.
Provides abstract base class for all backtest engines.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional
from collections import deque

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager

class BacktestEngineError(Exception):
    """Base exception for backtest engine errors"""
    pass


class DataPreparationError(BacktestEngineError):
    """Error raised when data preparation fails"""
    pass


class StrategyExecutionError(BacktestEngineError):
    """Error raised when strategy execution fails"""
    pass


class BaseBacktestEngine(ABC):
    """
    Base abstract backtest engine with efficient data management.
    
    Provides the foundation for all backtest engine implementations with
    standardized data handling, strategy integration, and backtest execution.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest engine base class
        
        Args:
            config: Configuration manager
            params: Engine parameters dictionary
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"backtest.{self.__class__.__name__.lower()}")
        self.strategy = None
        
        # Data management
        self.data_buffers = {}
        self.factor_cache = {}
        self.has_sufficient_history = {}
        self.data_queues = {}
        self.required_window_size = 0
        
        # Performance tracking
        self.metrics = {
            'total_signals': 0,
            'processing_time': 0,
            'data_points_processed': 0
        }
        
        # State tracking
        self._is_initialized = False
        self._is_running = False
    
    async def initialize(self) -> None:
        """
        Initialize backtest engine and load strategy
        
        This method performs standard initialization including strategy loading
        and window size calculation.
        
        Raises:
            BacktestEngineError: If initialization fails
        """
        if self._is_initialized:
            return
            
        self.logger.info("Initializing backtest engine")
        
        try:
            await self._load_strategy()
            
            if self.strategy:
                self._update_required_window_size()
            
            self._is_initialized = True
            self.logger.info(f"Backtest initialization complete, window size: {self.required_window_size}")
        except Exception as e:
            self.logger.error(f"Backtest engine initialization failed: {str(e)}")
            raise BacktestEngineError(f"Initialization failed: {str(e)}")
    
    def _update_required_window_size(self) -> None:
        """
        Determine required window size from strategy factors
        
        This method analyzes the strategy's factor requirements to determine
        the minimum required data window size.
        """
        if not hasattr(self.strategy, '_factor_registry'):
            self.required_window_size = 30  # Default
            return
            
        max_window = 0
        for name, info in self.strategy._factor_registry.items():
            window_size = info.get('window_size', 0)
            if info.get('is_differential', False):
                window_size += 1
            max_window = max(max_window, window_size)
            
            # Consider dependencies
            for dep_name in info.get('depends_on', []):
                if dep_name in self.strategy._factor_registry:
                    dep_window = self.strategy._factor_registry[dep_name].get('window_size', 0)
                    if self.strategy._factor_registry[dep_name].get('is_differential', False):
                        dep_window += 1
                    max_window = max(max_window, dep_window)
        
        self.required_window_size = max(max_window, 2)
    
    async def _load_strategy(self) -> None:
        """
        Load strategy from parameters
        
        This method loads the strategy specified in parameters or
        falls back to the default strategy from configuration.
        
        Raises:
            BacktestEngineError: If strategy loading fails
        """
        strategy_name = self.params.get('strategy')
        if not strategy_name:
            self.logger.info("No strategy specified for backtest, using default")
            # Get default strategy from config
            strategy_name = self.config.get("strategy", "default", default="dual_ma")
        
        strategy_params = self.params.get('strategy_params', {})
        
        try:
            # Import using factory pattern
            from src.strategy.factory import get_strategy_factory
            
            strategy_factory = get_strategy_factory(self.config)
            self.strategy = await strategy_factory.create_strategy(strategy_name, strategy_params)
            
            self.logger.info(f"Strategy '{strategy_name}' loaded and initialized")
        except Exception as e:
            self.logger.error(f"Failed to load strategy '{strategy_name}': {str(e)}")
            raise BacktestEngineError(f"Strategy loading failed: {str(e)}")
    
    async def prepare_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prepare and validate data for backtesting
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            bool: True if data is sufficient for backtest
            
        Raises:
            DataPreparationError: If data preparation encounters a critical error
        """
        if not data:
            self.logger.warning("No data provided for backtest")
            return False
        
        try:
            # Initialize data structures
            self.data_buffers = {}
            self.factor_cache = {}
            self.has_sufficient_history = {}
            self.data_queues = {}
            
            # Process each symbol
            sufficient_data = True
            for symbol, df in data.items():
                # Validate data
                if df.empty:
                    self.logger.warning(f"Empty data for {symbol}")
                    sufficient_data = False
                    continue
                
                # Ensure symbol column exists    
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                
                # Initialize buffers
                self.data_buffers[symbol] = pd.DataFrame()
                self.factor_cache[symbol] = {}
                self.has_sufficient_history[symbol] = False
                self.data_queues[symbol] = deque(maxlen=self.required_window_size)
                
                # Preload data if sufficient history exists
                if len(df) >= self.required_window_size:
                    initial_data = df.iloc[:self.required_window_size]
                    
                    for _, row in initial_data.iterrows():
                        self.data_queues[symbol].append(pd.DataFrame([row]))
                    
                    self.data_buffers[symbol] = initial_data.copy()
                    self.has_sufficient_history[symbol] = True
                    
                    self.logger.debug(f"Preloaded {len(initial_data)} data points for {symbol}")
                else:
                    self.logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.required_window_size}")
                    sufficient_data = False
            
            # Pre-calculate factors if strategy is available
            if self.strategy and sufficient_data:
                await self._precalculate_factors()
            
            return sufficient_data
            
        except Exception as e:
            self.logger.error(f"Data preparation error: {str(e)}")
            raise DataPreparationError(f"Failed to prepare data: {str(e)}")
    
    async def _precalculate_factors(self) -> None:
        """
        Precalculate factors for all symbols with sufficient history
        
        This method calculates strategy factors for efficient processing
        during the backtest.
        """
        if not hasattr(self.strategy, '_factor_registry') or not self.strategy._factor_registry:
            return
            
        self.logger.debug("Precalculating strategy factors")
        
        for symbol in self.has_sufficient_history:
            if not self.has_sufficient_history[symbol]:
                continue
                
            # Calculate all registered factors
            for factor_name in self.strategy._factor_registry:
                try:
                    factor_values = self.strategy.calculate_factor(
                        self.data_buffers[symbol], 
                        factor_name, 
                        symbol
                    )
                    
                    # Cache result
                    if symbol not in self.factor_cache:
                        self.factor_cache[symbol] = {}
                    self.factor_cache[symbol][factor_name] = factor_values
                    
                except Exception as e:
                    self.logger.error(f"Error calculating factor '{factor_name}' for {symbol}: {str(e)}")
    
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol
        
        Args:
            data_point: Single data point as DataFrame
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Generated signals
            
        Raises:
            StrategyExecutionError: If strategy execution fails
        """
        if not self._is_initialized:
            await self.initialize()
        
        if not self.strategy:
            raise StrategyExecutionError("No strategy loaded")
        
        try:
            # Add data point to buffer
            sufficient_history = await self._add_data_point(data_point, symbol)
            
            if not sufficient_history:
                self.logger.debug(f"Still collecting data for {symbol}, "
                                f"{len(self.data_queues.get(symbol, deque()))} of "
                                f"{self.required_window_size} required")
                return pd.DataFrame()  # Not enough data yet
            
            # Process data with strategy
            signals = await self.strategy.process_data(data_point, symbol)
            
            # Track metrics
            if not signals.empty:
                self.metrics['total_signals'] += len(signals)
            self.metrics['data_points_processed'] += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing data point for {symbol}: {str(e)}")
            raise StrategyExecutionError(f"Strategy execution failed: {str(e)}")
    
    async def _add_data_point(self, data_point: pd.DataFrame, symbol: str) -> bool:
        """
        Add a data point to the buffer
        
        Args:
            data_point: Single data point as DataFrame
            symbol: Trading symbol
            
        Returns:
            bool: True if sufficient history exists
        """
        # Initialize buffers if needed
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame()
            self.factor_cache[symbol] = {}
            self.has_sufficient_history[symbol] = False
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
        
        # Add to queue
        self.data_queues[symbol].append(data_point)
        
        # Update DataFrame buffer
        self.data_buffers[symbol] = pd.concat(list(self.data_queues[symbol]))
        
        # Update status
        sufficient = len(self.data_queues[symbol]) >= self.required_window_size
        
        # If status changed, log it
        if sufficient and not self.has_sufficient_history[symbol]:
            self.has_sufficient_history[symbol] = True
            self.logger.info(f"Collected sufficient history for {symbol}: {len(self.data_queues[symbol])} data points")
        
        return self.has_sufficient_history[symbol]
    
    @abstractmethod
    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run backtest on provided data
        
        This abstract method must be implemented by subclasses to perform
        the actual backtest execution.
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            Dict: Backtest results
        """
        pass
    
    def stop(self) -> None:
        """Stop backtest execution"""
        self._is_running = False
        self.logger.info("Stopping backtest")
    
    async def shutdown(self) -> None:
        """
        Clean up resources
        
        This method performs cleanup of resources used during backtesting.
        """
        if self.strategy and hasattr(self.strategy, 'shutdown'):
            await self.strategy.shutdown()
        
        # Clear buffers
        self.data_buffers.clear()
        self.factor_cache.clear()
        self.has_sufficient_history.clear()
        self.data_queues.clear()
        
        self._is_initialized = False
        self._is_running = False
        
        self.logger.info("Backtest engine shut down")