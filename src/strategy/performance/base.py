#!/usr/bin/env python3
# src/strategy/performance/base.py

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import os
import json

from src.common.config_manager import ConfigManager
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
        
        self.initial_capital = self.params.get("initial_balance", 
                                self.config.get("trading", "capital", "initial", default=10000.0))
        
        self.strategy_id = self.config.get("strategy", "id", default="dual_ma")
        
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
        self.equity_history[self.strategy_id] = pd.Series(dtype=float)
        self.trade_history[self.strategy_id] = pd.DataFrame()
        self.drawdown_history[self.strategy_id] = pd.Series(dtype=float)
        
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
            # Initialize with empty series with float dtype to avoid type issues
            self.equity_history[strategy_id] = pd.Series(dtype=float)
            self.trade_history[strategy_id] = pd.DataFrame()
            self.drawdown_history[strategy_id] = pd.Series(dtype=float)
            self.strategy_metrics[strategy_id] = {}
            self.logger.info(f"Registered strategy {strategy_id} for performance tracking")
        
        # Also ensure 'system' is registered for aggregation
        if 'system' not in self.equity_history:
            self.equity_history['system'] = pd.Series(dtype=float)
            self.trade_history['system'] = pd.DataFrame()
            self.drawdown_history['system'] = pd.Series(dtype=float)
            self.strategy_metrics['system'] = {}
            
    def update_equity(self, timestamp: Union[str, datetime], value: float) -> None:
        """Update equity point for a strategy
        
        Args:
            self.strategy_id: Strategy identifier
            timestamp: Timestamp for the equity point
            value: Equity value
        """
        # Ensure strategy is registered
        if self.strategy_id not in self.equity_history:
            self.register_strategy(self.strategy_id)
            
        # Convert timestamp to pandas Timestamp if needed
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)
            
        # Update equity history
        self.equity_history[self.strategy_id].loc[timestamp] = value
        
        # Sort by timestamp
        self.equity_history[self.strategy_id] = self.equity_history[self.strategy_id].sort_index()
        
        # Update drawdown
        self._update_drawdown()
        
        # Keep system equity updated - THIS IS WHERE THE ERROR OCCURS
        if self.strategy_id != 'system':
            # Check if we have any equity data before attempting to concatenate
            valid_dfs = [df for strat_id, df in self.equity_history.items() 
                        if strat_id != 'system' and not df.empty]
            
            if valid_dfs:  # Only concatenate if we have valid DataFrames
                # Merge all strategies for system equity
                system_equity = pd.concat(valid_dfs, axis=1)
                if not system_equity.empty:
                    # Sum across all strategies
                    system_equity = system_equity.sum(axis=1)
                    self.equity_history['system'] = system_equity
                    
                    # Update system drawdown
                    self._update_drawdown()
    
    def record_trade(self, trade_data: Dict[str, Any]) -> None:
        """Record a trade for performance analysis
        
        Args:
            self.strategy_id: Strategy identifier
            trade_data: Trade information
        """
        # Ensure strategy is registered
        if self.strategy_id not in self.trade_history:
            self.register_strategy(self.strategy_id)
            
        # Create DataFrame from trade data
        trade_df = pd.DataFrame([trade_data])
        
        # Append to trade history
        self.trade_history[self.strategy_id] = pd.concat([self.trade_history[self.strategy_id], trade_df])
        
        # Sort by timestamp if present
        if 'timestamp' in self.trade_history[self.strategy_id].columns:
            self.trade_history[self.strategy_id] = self.trade_history[self.strategy_id].sort_values('timestamp')
            
        # Add to system trade history
        if self.strategy_id is not None:
            # Add strategy ID column
            trade_df['self.strategy_id'] = self.strategy_id
            
            # Append to system trade history
            self.trade_history[self.strategy_id] = pd.concat([self.trade_history[self.strategy_id], trade_df])
            
            # Sort by timestamp if present
            if 'timestamp' in self.trade_history[self.strategy_id].columns:
                self.trade_history[self.strategy_id] = self.trade_history[self.strategy_id].sort_values('timestamp')
    
    def _safe_timestamp(self, timestamp) -> Optional[pd.Timestamp]:
        """
        Convert various formats to pandas Timestamp, handling None and NaT values
        
        Args:
            timestamp: Input timestamp in various formats
            
        Returns:
            Optional[pd.Timestamp]: Converted timestamp or None if invalid
        """
        if timestamp is None or pd.isna(timestamp):
            return None
            
        try:
            return pd.Timestamp(timestamp)
        except:
            self.logger.warning(f"Could not convert {timestamp} to Timestamp")
            return None

    def _safe_value(self, value, default=0.0) -> float:
        """
        Safely convert a value to float, handling None and NaN values
        
        Args:
            value: Input value
            default: Default value if conversion fails
            
        Returns:
            float: Converted value or default
        """
        if value is None or pd.isna(value):
            return default
            
        try:
            return float(value)
        except:
            self.logger.warning(f"Could not convert {value} to float")
            return default
        
    def _update_drawdown(self) -> None:
        """Update drawdown for a strategy
        
        Args:
            self.strategy_id: Strategy identifier
        """
        if self.strategy_id not in self.equity_history:
            return
            
        equity = self.equity_history[self.strategy_id]
        if equity.empty:
            return
            
        # Calculate running maximum
        running_max = equity.cummax()
        
        # Calculate drawdown
        drawdown = (equity - running_max) / running_max
        
        # Update drawdown history
        self.drawdown_history[self.strategy_id] = drawdown
    
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
    
    def calculate_returns(self, period: str = 'daily') -> pd.Series:
        """Calculate returns for a strategy
        
        Args:
            self.strategy_id: Strategy identifier
            period: Return period ('daily', 'weekly', 'monthly')
            
        Returns:
            pd.Series: Returns series
        """
        if self.strategy_id not in self.equity_history:
            return pd.Series()
            
        equity = self.equity_history[self.strategy_id]
        if len(equity) == 0:  # Use len() instead of empty property
            return pd.Series()
        
        # Convert millisecond timestamps to proper datetime index
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index, unit='ms')
                
        # Resample to desired frequency if needed
        if period == 'daily':
            equity_resampled = equity.resample('D').last().dropna()
        elif period == 'weekly':
            equity_resampled = equity.resample('W').last().dropna()
        elif period == 'monthly':
            equity_resampled = equity.resample('ME').last().dropna()
        else:
            equity_resampled = equity            
        
        # Calculate period returns
        returns = equity_resampled.pct_change().dropna()
        return returns

    @abstractmethod
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy
        
        Args:
            self.strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        pass
    
    @abstractmethod
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save performance data to file
        
        Args:
            self.strategy_id: Strategy identifier
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
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report
        
        Args:
            self.strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance report
        """
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the performance analyzer"""
        self.logger.info("Shutting down performance analyzer")
        
        # Save system performance data if configured
        auto_save = self.config.get("performance", "auto_save_on_shutdown", default=True)
        if auto_save and self.strategy_id in self.equity_history:
            try:
                filepath = self.save_to_file()
                self.logger.info(f"System performance data auto-saved to {filepath}")
            except Exception as e:
                self.logger.error(f"Error auto-saving system performance data: {e}")
        
        # Additional cleanup
        await self._shutdown_analyzer()
    
    @abstractmethod
    async def _shutdown_analyzer(self) -> None:
        """Analyzer-specific shutdown operations"""
        pass