#!/usr/bin/env python3
# src/backtest/base.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Type
from collections import deque
import time
import traceback

from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.common.async_executor import AsyncExecutor
from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.base import StrategyFactory


class BaseBacktestEngine:
    """
    Base backtest engine with efficient factor-based data management
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest engine
        
        Args:
            config: Configuration manager
            params: Backtest parameters
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"backtest.{self.__class__.__name__.lower()}")
        self.strategy = None
        self.executor = AsyncExecutor()  # Get singleton instance
        
        # Data management
        self.data_buffers = {}  # Symbol -> DataFrame
        self.factor_cache = {}  # Symbol -> {Factor -> Series}
        self.has_sufficient_history = {}  # Symbol -> bool
        
        # Use deque for efficient window management
        self.data_queues = {}  # Symbol -> deque of data points
        self.required_window_size = 0  # Will be determined from strategy
        
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
        """Initialize backtest engine and load strategy"""
        if self._is_initialized:
            return
            
        self.logger.info("Initializing backtest engine")
        
        # Start async executor
        await self.executor.start()
        
        # Load strategy
        await self._load_strategy()
        
        # Get required window size from strategy factors
        if self.strategy:
            self._update_required_window_size()
        
        self._is_initialized = True
        self.logger.info(f"Backtest initialization complete, required window size: {self.required_window_size}")
    
    def _update_required_window_size(self) -> None:
        """Determine required window size from strategy factors"""
        if not hasattr(self.strategy, '_factor_registry'):
            self.required_window_size = 30  # Default
            return
            
        # Find maximum window size from factors
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
        
        # Set required window size (minimum 2)
        self.required_window_size = max(max_window, 2)
        
        self.logger.info(f"Required window size determined from factors: {self.required_window_size}")
    
    async def _load_strategy(self) -> None:
        """Load strategy from parameters"""
        strategy_name = self.params.get('strategy')
        if not strategy_name:
            raise ValueError("No strategy specified for backtest")
        
        strategy_params = self.params.get('strategy_params', {})
        
        try:
            # Get strategy factory
            factory = StrategyFactory.get_instance(self.config)
            
            # Create strategy
            self.strategy = await factory.create(strategy_name, strategy_params)
            
            # Initialize strategy
            await self.strategy.initialize()
            
            self.logger.info(f"Strategy '{strategy_name}' loaded and initialized")
        except Exception as e:
            self.logger.error(f"Failed to load strategy '{strategy_name}': {str(e)}")
            raise
    
    async def prepare_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prepare and validate data for backtesting
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            bool: True if data is valid and sufficient
        """
        if not data:
            self.logger.warning("No data provided for backtest")
            return False
        
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
            df[symbol] = symbol
            
            # Initialize buffers
            self.data_buffers[symbol] = pd.DataFrame()
            self.factor_cache[symbol] = {}
            self.has_sufficient_history[symbol] = False
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
            
            # Preload data if sufficient history exists
            if len(df) >= self.required_window_size:
                # Take the most recent required_window_size rows for initial buffer
                initial_data = df.iloc[-self.required_window_size:]
                
                # Store in deque and dataframe
                for _, row in initial_data.iterrows():
                    self.data_queues[symbol].append(pd.DataFrame([row]))
                
                self.data_buffers[symbol] = initial_data.copy()
                self.has_sufficient_history[symbol] = True
                
                self.logger.info(f"Preloaded {len(initial_data)} data points for {symbol}")
            else:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.required_window_size}")
                sufficient_data = False
        
        # Pre-calculate factors if strategy is available
        if self.strategy and sufficient_data:
            await self._precalculate_factors()
        
        return sufficient_data
    
    async def _precalculate_factors(self) -> None:
        """Precalculate factors for all symbols with sufficient history"""
        if not hasattr(self.strategy, '_factor_registry') or not self.strategy._factor_registry:
            return
            
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
        
        self.logger.info("Precalculated factors for all symbols with sufficient history")
    
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol
        
        Args:
            data_point: New data point
            symbol: Symbol
            
        Returns:
            DataFrame: Generated signals or empty DataFrame
        """
        if not self._is_initialized:
            await self.initialize()
        
        if not self.strategy:
            raise ValueError("No strategy loaded")
        
        try:
            # Add data point to buffer
            sufficient_history = await self._add_data_point(data_point, symbol)
            
            if not sufficient_history:
                self.logger.debug(f"Still collecting data for {symbol}, "
                                f"{len(self.data_queues.get(symbol, deque()))} of "
                                f"{self.required_window_size} required data points")
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
            return pd.DataFrame()
    
    async def _add_data_point(self, data_point: pd.DataFrame, symbol: str) -> bool:
        """
        Add a data point to the buffer
        
        Args:
            data_point: New data point
            symbol: Symbol
            
        Returns:
            bool: True if we have sufficient history
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
    
    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run backtest on all data
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dict: Backtest results
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self._is_running = True
        
        results = {
            'signals': {},
            'metrics': {}
        }
        
        try:
            # Prepare data
            sufficient_data = await self.prepare_data(data)
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                return {
                    'error': 'Insufficient data for backtest',
                    'metrics': self.metrics
                }
            
            # Process each symbol
            for symbol, df in data.items():
                if symbol not in self.has_sufficient_history or not self.has_sufficient_history[symbol]:
                    continue
                
                signals = []
                
                # Process data points after initial window
                for i in range(self.required_window_size, len(df)):
                    if not self._is_running:
                        self.logger.info("Backtest stopped early")
                        break
                    
                    # Get data point
                    data_point = df.iloc[[i]]
                    
                    # Process data point
                    signal = await self.process_data_point(data_point, symbol)
                    
                    if not signal.empty:
                        signals.append(signal)
                    
                    # Allow asyncio to process other tasks
                    if i % 100 == 0:
                        await asyncio.sleep(0)
                
                # Combine signals
                if signals:
                    results['signals'][symbol] = pd.concat(signals)
                else:
                    results['signals'][symbol] = pd.DataFrame()
                    
                self.logger.info(f"Generated {len(results['signals'][symbol])} signals for {symbol}")
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.metrics['processing_time'] = execution_time
            
            results['metrics'] = self.metrics
            
            self.logger.info(f"Backtest completed in {execution_time:.2f}s")
            self.logger.info(f"Generated {self.metrics['total_signals']} signals in total")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}\n{traceback.format_exc()}")
            return {'error': str(e), 'metrics': self.metrics}
        finally:
            self._is_running = False
    
    def stop(self) -> None:
        """Stop backtest execution"""
        self._is_running = False
        self.logger.info("Stopping backtest")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
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


class BacktestFactory(AbstractFactory):
    """Factory for creating backtest engines"""
    
    def __init__(self, config):
        """Initialize backtest factory"""
        super().__init__(config)
        self.default_engine = config.get("backtest", "engine", default="standard")
        
        # Register default backtest engines
        self._register_default_engines()
    
    def _register_default_engines(self):
        """Register default backtest engines"""
        # Register standard backtest engine
        self.register("standard", "src.backtest.base.BaseBacktestEngine", {
            "description": "Standard Backtest Engine with factor-based data management"
        })
        
        # Register any additional engines
        try:
            self.register("ohlcv", "src.backtest.ohlcv.OHLCVEngine", {
                "description": "OHLCV Backtest Engine"
            })
            self.register("market_replay", "src.backtest.market_replay.MarketReplayEngine", {
                "description": "Market Replay Backtest Engine"
            })
        except Exception as e:
            self.logger.debug(f"Optional engines not registered: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseBacktestEngine]:
        """
        Get concrete backtest engine class
        
        Args:
            name: Engine name
            
        Returns:
            Type[BaseBacktestEngine]: Backtest engine class
        """
        return await self._load_class_from_path(name, BaseBacktestEngine)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve engine name with default fallback
        
        Args:
            name: Engine name
            
        Returns:
            str: Resolved engine name
        """
        name = name or self.default_engine
        if not name:
            raise ValueError("No engine name provided and no default in config")
        return name.lower()
    
    async def create_engine(self, engine_name: str, params: Optional[Dict[str, Any]] = None) -> BaseBacktestEngine:
        """
        Create and initialize backtest engine
        
        Args:
            engine_name: Engine name
            params: Engine parameters
            
        Returns:
            BaseBacktestEngine: Initialized backtest engine
        """
        engine = await self.create(engine_name, params)
        await engine.initialize()
        return engine


# Decorator for registering backtest engines
def register_backtest_engine(name: Optional[str] = None, **metadata):
    """
    Decorator for registering backtest engines
    
    Args:
        name: Engine name (defaults to class name without 'Engine' suffix)
        **metadata: Additional metadata
    """
    return register_factory_class('backtest_factory', name, **metadata)