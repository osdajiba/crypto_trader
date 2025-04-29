#!/usr/bin/env python3
# src/backtest/engine/base.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from collections import deque
import time
import traceback

from src.common.abstract_factory import register_factory_class
from src.common.async_executor import AsyncExecutor
from src.common.config import ConfigManager
from src.common.log_manager import LogManager


class BaseBacktestEngine:
    """Base backtest engine with efficient data management"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize backtest engine"""
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"backtest.{self.__class__.__name__.lower()}")
        self.strategy = None
        self.executor = AsyncExecutor()
        
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
        """Initialize backtest engine and load strategy"""
        if self._is_initialized:
            return
            
        self.logger.info("Initializing backtest engine")
        await self.executor.start()
        await self._load_strategy()
        
        if self.strategy:
            self._update_required_window_size()
        
        self._is_initialized = True
        self.logger.info(f"Backtest initialization complete, window size: {self.required_window_size}")
    
    def _update_required_window_size(self) -> None:
        """Determine required window size from strategy factors"""
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
        """Load strategy from parameters"""
        strategy_name = self.params.get('strategy')
        if not strategy_name:
            raise ValueError("No strategy specified for backtest")
        
        strategy_params = self.params.get('strategy_params', {})
        
        try:
            # Import here to avoid circular imports
            from src.strategy.base import StrategyFactory
            
            factory = StrategyFactory.get_instance(self.config)
            self.strategy = await factory.create(strategy_name, strategy_params)
            await self.strategy.initialize()
            
            self.logger.info(f"Strategy '{strategy_name}' loaded and initialized")
        except Exception as e:
            self.logger.error(f"Failed to load strategy '{strategy_name}': {str(e)}")
            raise
    
    async def prepare_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Prepare and validate data for backtesting"""
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
                initial_data = df.iloc[-self.required_window_size:]
                
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
    
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process a single data point for a symbol"""
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
            return pd.DataFrame()
    
    async def _add_data_point(self, data_point: pd.DataFrame, symbol: str) -> bool:
        """Add a data point to the buffer"""
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
        """Run backtest on all data"""
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