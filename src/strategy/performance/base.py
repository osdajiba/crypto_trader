#!/usr/bin/env python3
# src/strategy/performance/base.py

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json

from src.common.config import ConfigManager
from src.common.log_manager import LogManager


class BasePerformanceAnalyzer(ABC):
    """Base class for performance analyzers with standardized interface"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize performance analyzer
        
        Args:
            config: Configuration manager
            params: Optional parameters dictionary
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"performance.{self.__class__.__name__.lower()}")
        
        # Initialize storage paths
        self.storage_path = self.config.get("performance", "storage_path", default="./data/performance")
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Initial capital (can be overridden in params)
        self.initial_capital = self.params.get("initial_balance", 
                                self.config.get("trading", "capital", "initial", default=10000.0))
        
        # Risk-free rate for performance calculations
        self.risk_free_rate = self.config.get("performance", "risk_free_rate", default=0.0)
        
        # Data containers
        self.equity_history = {}  # Equity curve by strategy/portfolio
        self.trade_history = {}   # Trade history by strategy/portfolio
        self.drawdown_history = {} # Drawdown history by strategy/portfolio
        self.strategy_metrics = {} # Performance metrics by strategy
        self.benchmark_data = {}  # Benchmark data for comparison
        
        # Initialize benchmark data if available
        self._load_benchmark_data()
    
    async def initialize(self) -> None:
        """Initialize the performance analyzer"""
        self.logger.info("Initializing performance analyzer")
        # Create system portfolio tracking
        self.equity_history['system'] = pd.Series(dtype=float)
        self.trade_history['system'] = pd.DataFrame()
        self.drawdown_history['system'] = pd.Series(dtype=float)
        
        # Perform any additional initialization
        await self._initialize_analyzer()
        
        self.logger.info("Performance analyzer initialized")
    
    @abstractmethod
    async def _initialize_analyzer(self) -> None:
        """Initialize analyzer-specific components"""
        pass
    
    def register_strategy(self, strategy_id: str) -> None:
        """Register a strategy for performance tracking
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id not in self.equity_history:
            self.equity_history[strategy_id] = pd.Series(dtype=float)
            self.trade_history[strategy_id] = pd.DataFrame()
            self.drawdown_history[strategy_id] = pd.Series(dtype=float)
            self.strategy_metrics[strategy_id] = {}
            self.logger.info(f"Registered strategy {strategy_id} for performance tracking")
    
    def update_equity(self, strategy_id: str, timestamp: Union[str, datetime], value: float) -> None:
        """Update equity point for a strategy
        
        Args:
            strategy_id: Strategy identifier
            timestamp: Timestamp for the equity point
            value: Equity value
        """
        # Ensure strategy is registered
        if strategy_id not in self.equity_history:
            self.register_strategy(strategy_id)
            
        # Convert timestamp to pandas Timestamp if needed
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
            
        # Update equity history
        self.equity_history[strategy_id].loc[timestamp] = value
        
        # Sort by timestamp
        self.equity_history[strategy_id] = self.equity_history[strategy_id].sort_index()
        
        # Update drawdown
        self._update_drawdown(strategy_id)
        
        # Keep system equity updated
        if strategy_id != 'system':
            # Merge all strategies for system equity
            all_equity = pd.concat([df for strat_id, df in self.equity_history.items() 
                                if strat_id != 'system'], axis=1)
            if not all_equity.empty:
                # Sum across all strategies
                system_equity = all_equity.sum(axis=1)
                self.equity_history['system'] = system_equity
                
                # Update system drawdown
                self._update_drawdown('system')
    
    def record_trade(self, strategy_id: str, trade_data: Dict[str, Any]) -> None:
        """Record a trade for performance analysis
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Trade information
        """
        # Ensure strategy is registered
        if strategy_id not in self.trade_history:
            self.register_strategy(strategy_id)
            
        # Create DataFrame from trade data
        trade_df = pd.DataFrame([trade_data])
        
        # Append to trade history
        self.trade_history[strategy_id] = pd.concat([self.trade_history[strategy_id], trade_df])
        
        # Sort by timestamp if present
        if 'timestamp' in self.trade_history[strategy_id].columns:
            self.trade_history[strategy_id] = self.trade_history[strategy_id].sort_values('timestamp')
            
        # Add to system trade history
        if strategy_id != 'system':
            # Add strategy ID column
            trade_df['strategy_id'] = strategy_id
            
            # Append to system trade history
            self.trade_history['system'] = pd.concat([self.trade_history['system'], trade_df])
            
            # Sort by timestamp if present
            if 'timestamp' in self.trade_history['system'].columns:
                self.trade_history['system'] = self.trade_history['system'].sort_values('timestamp')
    
    def _update_drawdown(self, strategy_id: str) -> None:
        """Update drawdown for a strategy
        
        Args:
            strategy_id: Strategy identifier
        """
        if strategy_id not in self.equity_history:
            return
            
        equity = self.equity_history[strategy_id]
        if equity.empty:
            return
            
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Update drawdown history
        self.drawdown_history[strategy_id] = drawdown
    
    def _load_benchmark_data(self) -> None:
        """Load benchmark data if configured"""
        benchmark_symbol = self.config.get("performance", "benchmark_symbol", default="SPY")
        benchmark_file = self.config.get("performance", "benchmark_data", default=None)
        
        if benchmark_file and os.path.exists(benchmark_file):
            try:
                self.benchmark_data[benchmark_symbol] = pd.read_csv(
                    benchmark_file, parse_dates=['date'], index_col='date'
                )
                self.logger.info(f"Loaded benchmark data for {benchmark_symbol}")
            except Exception as e:
                self.logger.error(f"Error loading benchmark data: {e}")
    
    def calculate_returns(self, strategy_id: str, period: str = 'daily') -> pd.Series:
        """Calculate returns for a strategy
        
        Args:
            strategy_id: Strategy identifier
            period: Return period ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.Series: Returns series
        """
        if strategy_id not in self.equity_history:
            return pd.Series()
            
        equity = self.equity_history[strategy_id]
        if equity.empty:
            return pd.Series()
            
        # Resample to desired frequency if needed
        if period == 'daily':
            equity_resampled = equity.resample('D').last().dropna()
        elif period == 'weekly':
            equity_resampled = equity.resample('W').last().dropna()
        elif period == 'monthly':
            equity_resampled = equity.resample('M').last().dropna()
        else:
            equity_resampled = equity
            
        # Calculate period returns
        returns = equity_resampled.pct_change().dropna()
        return returns
    
    @abstractmethod
    def calculate_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        pass
    
    @abstractmethod
    def save_to_file(self, strategy_id: str, filepath: Optional[str] = None) -> str:
        """Save performance data to file
        
        Args:
            strategy_id: Strategy identifier
            filepath: Optional file path
            
        Returns:
            str: Path to saved file
        """
        pass
    
    @abstractmethod
    def load_from_file(self, filepath: str) -> str:
        """Load performance data from file
        
        Args:
            filepath: File path
            
        Returns:
            str: Strategy ID loaded
        """
        pass
    
    @abstractmethod
    def generate_performance_report(self, strategy_id: str) -> Dict[str, Any]:
        """Generate a comprehensive performance report
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance report
        """
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the performance analyzer"""
        self.logger.info("Shutting down performance analyzer")
        
        # Save system performance data if configured
        auto_save = self.config.get("performance", "auto_save_on_shutdown", default=True)
        if auto_save and 'system' in self.equity_history and not self.equity_history['system'].empty:
            try:
                filepath = self.save_to_file('system')
                self.logger.info(f"System performance data auto-saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Error auto-saving system performance data: {e}")
        
        # Additional cleanup
        await self._shutdown_analyzer()
    
    @abstractmethod
    async def _shutdown_analyzer(self) -> None:
        """Analyzer-specific shutdown operations"""
        pass