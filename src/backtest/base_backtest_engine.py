# src/backtest/base_backtest_engine.py

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from datetime import datetime
import json
import os
from pathlib import Path

from src.common.log_manager import LogManager

class BacktestEngine(ABC):
    """Enhanced unified backtest engine base class"""
    
    def __init__(self, config, strategy, data_manager):
        """
        Initialize base backtest engine
        
        Args:
            config: Configuration manager
            strategy: Trading strategy instance
            data_manager: Data manager instance
        """
        self.config = config
        self.strategy = strategy
        self.data_manager = data_manager
        self.logger = LogManager.get_logger("system.backtest")
        
        # Common parameters
        self.initial_capital = self.config.get("backtest", "initial_capital", default=100000)
        self.commission_rate = self.config.get("backtest", "transaction_costs", "commission_rate", default=0.001)
        self.slippage = self.config.get("backtest", "transaction_costs", "slippage", default=0.001)
        
        # Backtest state
        self.state = {
            'timestamp': None,
            'cash': self.initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': [],
            'market_prices': {},
            'current_equity': self.initial_capital,
            'peak_equity': self.initial_capital,
            'max_drawdown': 0.0
        }
        
        # Component references (to be set by subclasses)
        self.execution_engine = None
        self.risk_manager = None
        self.performance_monitor = None
        
        # Store symbols and timeframe
        self.symbols = []
        self.timeframe = ""
        
    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Execute generic backtest workflow
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for backtesting
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        self.logger.info(f"Starting backtest | Symbols: {symbols} | Timeframe: {timeframe}")
        
        try:
            # 1. Initialize backtest parameters
            await self._initialize_backtest(symbols, timeframe)
            
            # 2. Load historical data
            data = await self._load_historical_data(symbols, timeframe)
            if not data:
                raise ValueError("Failed to load historical data")
                
            # 3. Generate time points sequence
            time_points = self._get_time_points(data)
            self.logger.info(f"Backtest contains {len(time_points)} time periods")
            
            # 4. Execute backtest loop
            for i, time_point in enumerate(time_points):
                if i % 100 == 0:  # Periodic progress logging
                    self.logger.info(f"Backtest progress: {i}/{len(time_points)}")
                
                # Get data for current time point
                current_data = self._get_data_at_time_point(data, time_point)
                
                # Process current time point
                await self._process_data_point(time_point, current_data)
                
                # Update performance metrics
                self._update_performance_metrics(time_point)
                
            # 5. Close any open positions
            await self._close_positions()
                
            # 6. Generate backtest report
            results = self._generate_backtest_report()
            
            # 7. Save backtest report
            self._save_backtest_report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest execution error: {e}", exc_info=True)
            return {"error": str(e)}
            
        finally:
            # 8. Cleanup resources
            await self._cleanup_resources()
    
    def _update_performance_metrics(self, timestamp):
        """
        Update performance metrics at current timestamp
        
        Args:
            timestamp: Current timestamp
        """
        if self.performance_monitor:
            equity = self._calculate_equity()
            self.performance_monitor.update_equity_curve(timestamp, equity)
            
            # Update state
            self.state['current_equity'] = equity
            
            # Update peak equity and drawdown
            if equity > self.state['peak_equity']:
                self.state['peak_equity'] = equity
            
            if self.state['peak_equity'] > 0:
                drawdown = (self.state['peak_equity'] - equity) / self.state['peak_equity']
                self.state['max_drawdown'] = max(self.state['max_drawdown'], drawdown)
    
    def _calculate_equity(self) -> float:
        """
        Calculate current equity
        
        Returns:
            float: Current equity value
        """
        equity = self.state['cash']
        
        # Add position values
        for symbol, position in self.state['positions'].items():
            # Support both simple quantity and dict position formats
            if isinstance(position, dict) and 'quantity' in position:
                quantity = position['quantity']
            else:
                quantity = position
                
            if symbol in self.state['market_prices']:
                price = self.state['market_prices'][symbol]
                equity += quantity * price
        
        return equity
    
    def _update_market_prices(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Update current market prices
        
        Args:
            data: Current time point data
        """
        for symbol, df in data.items():
            if not df.empty and 'close' in df.columns:
                self.state['market_prices'][symbol] = df['close'].iloc[0]
    
    async def _close_positions(self) -> None:
        """Close all open positions at the end of backtest"""
        # This is a placeholder that should be implemented by subclasses
        pass

    @abstractmethod
    async def _initialize_backtest(self, symbols: List[str], timeframe: str) -> None:
        """
        Initialize backtest parameters (must be implemented by subclasses)
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for backtesting
        """
        pass

    @abstractmethod
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data (must be implemented by subclasses)
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe for backtesting
            
        Returns:
            Dict[str, pd.DataFrame]: Historical data
        """
        pass

    @abstractmethod
    def _get_time_points(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Get sequence of time points (must be implemented by subclasses)
        
        Args:
            data: Historical data
            
        Returns:
            List[Any]: Sequence of time points
        """
        pass

    @abstractmethod
    def _get_data_at_time_point(self, data: Dict[str, pd.DataFrame], time_point: Any) -> Dict[str, pd.DataFrame]:
        """
        Get data at specified time point (must be implemented by subclasses)
        
        Args:
            data: Historical data
            time_point: Time point
            
        Returns:
            Dict[str, pd.DataFrame]: Data at time point
        """
        pass

    @abstractmethod
    async def _process_data_point(self, time_point: Any, data: Dict[str, pd.DataFrame]) -> None:
        """
        Process single data point (must be implemented by subclasses)
        
        Args:
            time_point: Time point
            data: Data at time point
        """
        pass

    @abstractmethod
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        Generate backtest report (must be implemented by subclasses)
        
        Returns:
            Dict[str, Any]: Backtest report
        """
        pass

    def _save_backtest_report(self, results: Dict[str, Any]) -> None:
        """
        Default implementation for saving backtest reports
        
        Args:
            results: Backtest results
        """
        # Create report directory
        report_dir = self.config.get("reporting", "backtest_reports_dir", default="reports/backtest/")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trades
        output_formats = self.config.get("reporting", "output_formats", default=["json"])
        
        # Save CSV
        if 'csv' in output_formats and 'trades' in results:
            trades_df = pd.DataFrame(results['trades'])
            trades_csv = os.path.join(report_dir, f"trades_{timestamp}.csv")
            trades_df.to_csv(trades_csv, index=False)
            self.logger.info(f"Trades records saved to {trades_csv}")
        
        # Save JSON
        if 'json' in output_formats:
            # Prepare for serialization
            clean_results = self._prepare_for_serialization(results)
            
            summary_json = os.path.join(report_dir, f"report_{timestamp}.json")
            with open(summary_json, 'w') as f:
                json.dump(clean_results, f, indent=4, default=str)
            
            self.logger.info(f"Backtest report saved to {summary_json}")
        
        # Log performance metrics
        self._log_performance_metrics(results)

    def _prepare_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for JSON serialization
        
        Args:
            data: Raw data
            
        Returns:
            Dict[str, Any]: Serializable data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items() 
                if k not in ['strategy', 'equity_curve'] or k == 'strategy' and hasattr(v, '__name__')}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return data

    def _log_performance_metrics(self, results: Dict[str, Any]) -> None:
        """
        Log performance metrics
        
        Args:
            results: Backtest results
        """
        self.logger.info("==== Backtest Performance Summary ====")
        self.logger.info(f"Initial Capital: ${results.get('initial_capital', 0):,.2f}")
        self.logger.info(f"Final Equity: ${results.get('final_equity', 0):,.2f}")
        self.logger.info(f"Total Return: ${results.get('total_return', 0):,.2f} ({results.get('total_return_pct', 0):.2f}%)")
        self.logger.info(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        self.logger.info(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        self.logger.info("=======================================")

    async def _cleanup_resources(self) -> None:
        """
        Cleanup resources
        """
        # Clean up strategy
        if hasattr(self.strategy, 'shutdown'):
            await self.strategy.shutdown()
            
        # Clean up other components
        if self.performance_monitor and hasattr(self.performance_monitor, 'close'):
            await self.performance_monitor.close()
            
        if self.execution_engine and hasattr(self.execution_engine, 'close'):
            await self.execution_engine.close()