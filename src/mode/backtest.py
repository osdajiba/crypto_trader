#!/usr/bin/env python3
# src/trading/backtest.py

from typing import Dict, Optional, Any, List
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from tqdm import tqdm

from src.common.config_manager import ConfigManager
from src.common.helpers import TimeUtils
from src.common.abstract_factory import register_factory_class
from src.backtest.factory import get_backtest_engine_factory
from src.mode.base import BaseTradingMode


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
        self.backtest_engine = await self.backtest_engine_factory.create_backtest_engine(self.engine_name)
        self.logger.info(f"Backtest engine initialized: {self.engine_name}")
        
        # Connect portfolio to engine
        if hasattr(self.backtest_engine, 'set_portfolio') and callable(getattr(self.backtest_engine, 'set_portfolio')):
            self.backtest_engine.set_portfolio(self.portfolio)
            self.logger.debug("Portfolio connected to backtest engine")
        
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
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute the backtest using the backtest engine"""
        self.logger.info(f"Starting backtest execution using engine: {self.engine_name}")
        
        # Set up progress display if enabled
        if self.show_progress:
            # Initialize progress bar for data preparation
            total_symbols = len(symbols)
            with tqdm(total=total_symbols, desc="Preparing data") as pbar:
                # Prepare the data in the backtest engine
                sufficient_data = await self.backtest_engine.prepare_data(self.historical_data)
                pbar.update(total_symbols)
                
                if not sufficient_data:
                    self.logger.warning("Insufficient data for backtest")
                    return {"status": "failed", "reason": "insufficient_data"}
        else:
            # Prepare the data without progress display
            sufficient_data = await self.backtest_engine.prepare_data(self.historical_data)
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                return {"status": "failed", "reason": "insufficient_data"}
        
        # Run the backtest
        try:
            # Run the actual backtest using the engine
            result = await self.backtest_engine.run_backtest(self.historical_data)
            
            # Update state with results from engine
            if 'equity_curve' in result:
                self.state['equity_curve'] = result['equity_curve']
            
            if 'trades' in result:
                self.state['trades'] = result['trades']
                
                # Update portfolio with trades
                await self.portfolio.record_batch_trades(result['trades'])
            
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
            
            self.logger.info(f"Backtest completed successfully using {self.engine_name} engine")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during backtest execution: {e}", exc_info=True)
            return {
                "status": "failed",
                "reason": str(e),
                "symbols": list(self.historical_data.keys()),
                "timeframe": timeframe,
                "engine_type": self.engine_name
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
        
        # Include engine-specific metrics if available
        if hasattr(self.backtest_engine, 'get_metrics') and callable(getattr(self.backtest_engine, 'get_metrics')):
            engine_metrics = self.backtest_engine.get_metrics()
            report.update(engine_metrics)
        
        # Add engine-specific parameters if available
        engine_config = self.config.get("backtest", "engines", self.engine_name, default={})
        if engine_config:
            report['engine_config'] = engine_config
    
    async def get_available_engines(self) -> Dict[str, Dict[str, Any]]:
        """
        Get list of available backtest engines with their metadata
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of engine names with metadata
        """
        return self.backtest_engine_factory.get_available_engines()
    
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
        # First call parent implementation
        await super().shutdown()
        
        # Close backtest engine
        if self.backtest_engine:
            if hasattr(self.backtest_engine, 'shutdown'):
                await self.backtest_engine.shutdown()
            elif hasattr(self.backtest_engine, 'close'):
                await self.backtest_engine.close()
            
        self.logger.info("BacktestMode shutdown complete")