#!/usr/bin/env python3
# src/backtest/mode.py

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path

from src.trading.modes.base import BaseTradingMode
from src.trading.execution.engine import ExecutionEngine
from src.risk.manager import BacktestRiskManager
from src.trading.performance.manager import PerformanceManager
from src.backtest.factory import get_backtest_factory
from src.common.config import ConfigManager


class BacktestMode(BaseTradingMode):
    """Backtesting trading mode with modular engine support"""
    
    def __init__(self, config, data_manager=None, strategy_factory=None, 
                risk_manager=None, performance_monitor=None):
        """Initialize backtest mode"""
        super().__init__(config, data_manager, strategy_factory, risk_manager, performance_monitor)
        
        # Backtest-specific attributes
        self.start_date = None
        self.end_date = None
        self.historical_data = {}
        self.timestamps = []
        self.backtest_engine = None
        self.backtest_engine_name = config.get("backtest", "engine", default="ohlcv")
    
    async def initialize(self) -> None:
        """Initialize backtest mode specific components"""
        self.logger.info("Initializing backtest mode")
        
        # Create backtest-specific execution engine
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="backtest"
        )
        
        # Initialize risk manager
        if not self.risk_manager or not isinstance(self.risk_manager, BacktestRiskManager):
            self.risk_manager = BacktestRiskManager(config=self.config)
            await self.risk_manager.initialize()
        
        # Get strategy information from config
        strategy_name = self.config.get("strategy", "active", default='dual_ma')
        strategy_params = self.config.get("strategy", "parameters", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        
        # Initialize performance monitoring
        if not self.performance_monitor:
            initial_capital = self.config.get("trading", "capital", "initial", default=100000)
            self.performance_monitor = PerformanceManager(
                config=self.config,
                initial_balance=initial_capital
            )
        
        # Initialize backtest engine
        await self._initialize_backtest_engine(strategy_name, strategy_params)
        
        # Mark as initialization complete
        self._running = True
        self.logger.info("Backtest mode initialization complete")
    
    async def _initialize_backtest_engine(self, strategy_name: str, strategy_params: Dict) -> None:
        """Initialize the appropriate backtest engine based on configuration"""
        engine_name = self.backtest_engine_name
        self.logger.info(f"Initializing backtest engine: {engine_name}")
        
        factory = get_backtest_factory(self.config)
        
        engine_params = {
            'strategy': strategy_name,
            'strategy_params': strategy_params,
            'initial_capital': self.config.get("trading", "capital", "initial", default=100000),
            'commission_rate': self.config.get("trading", "execution", "commission", default=0.001),
            'slippage': self.config.get("trading", "execution", "slippage", default=0.001),
        }
        
        self.backtest_engine = await factory.create_engine(engine_name, engine_params)
        self.logger.info(f"Backtest engine {engine_name} initialized")
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Backtest-specific preparation"""
        # Initialize backtest parameters from config
        self.start_date = self.config.get("backtest", "period", "start", default=None)
        self.end_date = self.config.get("backtest", "period", "end", default=None)
        
        # If not specified in backtest section, try trading section
        if not self.start_date:
            self.start_date = self.config.get("trading", "backtest_start", default=None)
        
        if not self.end_date:
            self.end_date = self.config.get("trading", "backtest_end", default=None)
        
        # If still None, use reasonable defaults (last month)
        if not self.start_date:
            end = datetime.now()
            start = end.replace(day=1)  # Start of current month
            self.start_date = start.strftime("%Y-%m-%d")
            self.end_date = end.strftime("%Y-%m-%d")
            self.logger.info(f"Using default date range: {self.start_date} to {self.end_date}")
        
        # Load historical data
        self.historical_data = await self._load_historical_data(symbols, timeframe)
        if not self.historical_data:
            raise ValueError("Failed to get historical data")
        
        # Get timestamps from historical data
        self.timestamps = self._get_combined_timestamps(self.historical_data)
        self.logger.info(f"Backtest contains {len(self.timestamps)} time periods")
        
        # Set execution engine's historical data
        if self.execution_engine:
            self.execution_engine.set_historical_data(self.historical_data)
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """Load historical data"""
        data_map = {}
        
        for symbol in symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=self.start_date,
                    end=self.end_date
                )
                
                if not data.empty:
                    data['symbol'] = symbol  # Add symbol column for later processing
                    data_map[symbol] = data
                    self.logger.info(f"Loaded {symbol} historical data: {len(data)} records")
                else:
                    self.logger.warning(f"{symbol} has no historical data")
                
            except Exception as e:
                self.logger.error(f"Failed to load {symbol} historical data: {e}")
        
        return data_map
    
    def _get_combined_timestamps(self, data_map: Dict[str, pd.DataFrame]) -> List[Any]:
        """Get all unique timestamps and sort them"""
        all_timestamps = []
        
        for df in data_map.values():
            # Use 'datetime' column if available, otherwise try index
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
            elif isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.extend(df.index.tolist())
        
        # Sort timestamps and remove duplicates
        return sorted(set(all_timestamps))
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute backtest trading loop"""
        try:
            # Run backtest using the initialized engine
            backtest_results = await self.backtest_engine.run_backtest(self.historical_data)
            
            # Process results
            if 'error' in backtest_results:
                self.logger.error(f"Backtest error: {backtest_results['error']}")
                return {'error': backtest_results['error']}
            
            # Update performance monitor with trades
            if 'trades' in backtest_results and self.performance_monitor:
                for trade in backtest_results['trades']:
                    self.performance_monitor.record_trade(
                        timestamp=trade.get('timestamp'),
                        symbol=trade.get('symbol', ''),
                        direction=trade.get('action', ''),
                        entry_price=trade.get('price', 0),
                        exit_price=trade.get('price', 0),
                        quantity=trade.get('quantity', 0),
                        commission=trade.get('commission', 0)
                    )
            
            # Update equity curve in performance monitor
            if 'equity_curve' in backtest_results and not backtest_results['equity_curve'].empty:
                equity_df = backtest_results['equity_curve']
                
                for _, row in equity_df.iterrows():
                    if 'timestamp' in row and 'portfolio_value' in row:
                        self.performance_monitor.update_equity_curve(
                            row['timestamp'], 
                            row['portfolio_value']
                        )
            
            # Generate comprehensive report
            self.performance_monitor.calculate_performance_metrics()
            report = self.performance_monitor.generate_detailed_report()
            
            # Add backtest engine results
            report.update({
                'signals': backtest_results.get('signals', {}),
                'trades': backtest_results.get('trades', []),
                'engine_metrics': backtest_results.get('metrics', {})
            })
            
            # Generate visual reports if configured
            if self.config.get("backtest", "generate_charts", default=True):
                self._generate_charts(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Backtest execution error: {e}", exc_info=True)
            raise
    
    def _generate_charts(self, report: Dict[str, Any]) -> None:
        """Generate performance charts if matplotlib is available"""
        try:
            # Check if matplotlib is available
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            reports_dir = Path(self.config.get("backtest", "reports_dir", default="reports/backtest"))
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for filenames
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Draw equity curve
            if 'equity_curve' in report:
                equity_df = pd.DataFrame(report['equity_curve'])
                if not equity_df.empty and 'timestamps' in equity_df and 'balance' in equity_df:
                    plt.figure(figsize=(12, 6))
                    plt.plot(equity_df['timestamps'], equity_df['balance'])
                    plt.title('Equity Curve')
                    plt.xlabel('Time')
                    plt.ylabel('Portfolio Value')
                    plt.grid(True)
                    plt.savefig(os.path.join(reports_dir, f"equity_curve_{timestamp}.png"))
                    plt.close()
            
            # Draw drawdown curve
            if 'equity_curve' in report:
                equity_df = pd.DataFrame(report['equity_curve'])
                if not equity_df.empty and 'timestamps' in equity_df and 'balance' in equity_df:
                    # Calculate drawdown
                    equity = equity_df['balance'].values
                    peak = np.maximum.accumulate(equity)
                    drawdown = (equity - peak) / peak
                    
                    plt.figure(figsize=(12, 6))
                    plt.plot(equity_df['timestamps'], drawdown * 100)
                    plt.title('Drawdown (%)')
                    plt.xlabel('Time')
                    plt.ylabel('Drawdown (%)')
                    plt.grid(True)
                    plt.savefig(os.path.join(reports_dir, f"drawdown_{timestamp}.png"))
                    plt.close()
                    
            self.logger.info(f"Performance charts saved to {reports_dir}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available - skipping chart generation")
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add backtest-specific metrics to report"""
        # Add backtest parameters
        report['backtest_params'] = {
            'engine': self.backtest_engine_name,
            'symbols': self.config.get("trading", "instruments", default=[]),
            'timeframe': self.config.get("data", "default_timeframe", default="1m"),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'transaction_costs': {
                'commission_rate': self.config.get("trading", "execution", "commission", default=0.001),
                'slippage': self.config.get("trading", "execution", "slippage", default=0.001)
            }
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown backtest mode"""
        self.logger.info("Shutting down backtest mode")
        
        # Shutdown order of components is important
        
        # First shutdown backtest engine
        if self.backtest_engine:
            await self.backtest_engine.shutdown()
        
        # Then shutdown the strategy
        if hasattr(self, 'strategy') and self.strategy:
            await self.strategy.shutdown()
        
        # Then shutdown execution engine
        if hasattr(self, 'execution_engine') and self.execution_engine:
            await self.execution_engine.close()
        
        # Finally shutdown performance monitor
        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            await self.performance_monitor.close()
        
        self._running = False
        self.logger.info("Backtest mode shutdown complete")