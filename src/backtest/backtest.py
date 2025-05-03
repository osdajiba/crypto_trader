#!/usr/bin/env python3
# src/backtest/backtest.py

from typing import Dict, Optional, Any, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from tqdm import tqdm

from src.common.config_manager import ConfigManager
from src.common.helpers import TimeUtils
from src.common.abstract_factory import register_factory_class
from src.trading.base import BaseTradingMode


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
        self.current_index = 0
        self.start_date = None
        self.end_date = None
        self.show_progress = True
    
    async def _initialize_mode_specific(self) -> None:
        """Initialize backtest-specific components"""
        # Get strategy
        strategy_name = self.config.get("backtest", "strategy", default=None)
        strategy_params = self.config.get("backtest", "strategy_params", default={})
        
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        self.logger.info(f"Strategy initialized: {self.strategy.__class__.__name__}")
        
        # Create execution engine for backtesting
        execution_params = {
            "mode": "backtest",
            "allow_partial_fills": False
        }
        self.execution_engine = await self.execution_factory.create("backtest_execution", execution_params)
        self.logger.info("Backtest execution engine initialized")
        
        # Configure backtest parameters
        self.show_progress = self.config.get("backtest", "show_progress", default=True)
        
        # Parse date range
        date_config = self.config.get("backtest", "period", default={})
        self.start_date = TimeUtils.parse_timestamp(date_config.get("start", None))
        self.end_date = TimeUtils.parse_timestamp(date_config.get("end", None))
        
        if not self.start_date:
            self.start_date = datetime.now() - timedelta(days=365)
        if not self.end_date:
            self.end_date = datetime.now()
            
        self.logger.info(f"Backtest period: {self.start_date} to {self.end_date}")
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Prepare backtest data"""
        self.logger.info(f"Loading historical data for {len(symbols)} symbols")
        
        # Load all historical data
        for symbol in symbols:
            try:
                df = await self.data_source.fetch_historical(
                    symbol, 
                    timeframe, 
                    start=self.start_date,
                    end=self.end_date
                )
                
                if df.empty:
                    self.logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                # Ensure datetime column is datetime type
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    
                # Store data
                self.historical_data[symbol] = df
                self.logger.info(f"Loaded {len(df)} bars for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
                
        if not self.historical_data:
            raise ValueError("No historical data loaded for any symbol")
            
        # Prepare indexes by aligning all data to common dates
        self._align_data_indexes()
    
    def _align_data_indexes(self) -> None:
        """Align data from different symbols to common timestamps"""
        # Get unique timestamps across all data
        all_timestamps = set()
        
        for symbol, df in self.historical_data.items():
            if 'datetime' in df.columns:
                all_timestamps.update(df['datetime'].tolist())
        
        # Sort timestamps
        self.common_timestamps = sorted(all_timestamps)
        self.logger.info(f"Aligned data with {len(self.common_timestamps)} unique timestamps")
        
        # Reset index to start at 0
        self.current_index = 0
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute the main backtesting loop"""
        self.logger.info(f"Starting backtest execution with {len(self.common_timestamps)} bars")
        
        # Use tqdm for progress reporting if enabled
        iterator = tqdm(range(len(self.common_timestamps))) if self.show_progress else range(len(self.common_timestamps))
        
        for i in iterator:
            if not self._should_continue():
                self.logger.warning("Backtest stopped due to risk breach or manual stop")
                break
                
            self.current_index = i
            current_time = self.common_timestamps[i]
            
            # Get data slice for current timestamp
            data_slice = self._get_data_slice(current_time)
            
            if not data_slice:
                continue
                
            # Process this slice
            trades = await self._process_market_data(data_slice)
            
            # Update progress description if using tqdm
            if self.show_progress and isinstance(iterator, tqdm):
                equity = self.state.get('current_equity', 0)
                iterator.set_description(f"Equity: ${equity:.2f}")
        
        # Calculate final metrics
        result = {
            "status": "completed",
            "bars_processed": self.current_index,
            "total_bars": len(self.common_timestamps),
            "symbols": list(self.historical_data.keys()),
            "timeframe": timeframe
        }
        
        self.logger.info(f"Backtest completed. Processed {self.current_index} of {len(self.common_timestamps)} bars")
        return result
    
    def _get_data_slice(self, timestamp) -> Dict[str, pd.DataFrame]:
        """Get data slice for all symbols at specified timestamp"""
        result = {}
        
        for symbol, df in self.historical_data.items():
            # Find row with matching timestamp
            if 'datetime' in df.columns:
                matching_rows = df[df['datetime'] == timestamp]
                
                if not matching_rows.empty:
                    result[symbol] = matching_rows
        
        return result
    
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
            'strategy': self.strategy.__class__.__name__
        }
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        shutdown_tasks = []
        
        # Close strategy
        if self.strategy:
            shutdown_tasks.append(self.strategy.shutdown())
            
        # Close execution engine
        if self.execution_engine:
            shutdown_tasks.append(self.execution_engine.shutdown())
            
        # Close data source
        if self.data_source:
            shutdown_tasks.append(self.data_source.shutdown())
            
        # Close risk manager
        if self.risk_manager:
            shutdown_tasks.append(self.risk_manager.shutdown())
            
        # Close performance analyzer
        if self.performance_analyzer:
            shutdown_tasks.append(self.performance_analyzer.shutdown())
        
        # Wait for all shutdown tasks to complete
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
        self._running = False
        self.logger.info("BacktestMode shutdown complete")