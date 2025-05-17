#!/usr/bin/env python3
# src/strategy/performance/backtest.py

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.helpers import format_timestamp
from src.strategy.performance.base import BasePerformanceAnalyzer
from src.strategy.performance.metrics import PerformanceMetrics
from src.common.abstract_factory import register_factory_class


@register_factory_class('performance_factory', 'backtest', 
                       description="Performance analyzer for backtesting",
                       features=["historical_analysis", "equity_tracking", "drawdown_calculation"],
                       category="simulation")
class BacktestPerformanceAnalyzer(BasePerformanceAnalyzer):
    """Performance analyzer implementation for backtesting scenarios"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize backtest performance analyzer"""
        super().__init__(config, params)
        
        # Backtest specific settings
        self.output_formats = self.config.get("performance", "output_formats", default=["json", "csv"])
        self.include_charts = self.config.get("performance", "include_charts", default=True)
        self.report_dir = self.config.get("performance", "report_dir", default="./reports/backtest")
        
        # Create report directory
        os.makedirs(self.report_dir, exist_ok=True)
    
    async def _initialize_analyzer(self) -> None:
        """Initialize analyzer-specific components"""
        # Load historical benchmark data if specified
        benchmark_file = self.config.get("performance", "backtest", "benchmark_file", default=None)
        if benchmark_file and os.path.exists(benchmark_file):
            try:
                benchmark_symbol = self.config.get("performance", "benchmark_symbol", default="SPY")
                self.benchmark_data[benchmark_symbol] = pd.read_csv(
                    benchmark_file, parse_dates=['date'], index_col='date'
                )
                self.logger.info(f"Loaded backtest benchmark data from {benchmark_file}")
            except Exception as e:
                self.logger.error(f"Error loading backtest benchmark data: {e}")

    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a strategy
        
        Args:
            self.strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if self.strategy_id not in self.equity_history:
            return {}
            
        # Get equity and returns
        equity = self.equity_history[self.strategy_id]
        if len(equity) == 0:  # Use len() instead of empty property
            return {}
            
        # Convert to datetime index if needed
        if not isinstance(equity.index, pd.DatetimeIndex):
            equity.index = pd.to_datetime(equity.index, unit='ms')
        
        # Calculate returns
        daily_returns = self.calculate_returns('daily')
        monthly_returns = self.calculate_returns('monthly')
        
        # Calculate drawdown
        max_drawdown, drawdown_start, drawdown_end = PerformanceMetrics.calculate_max_drawdown(equity)
        
        # Calculate trade metrics
        trade_metrics = {}
        if self.strategy_id in self.trade_history:
            trades = self.trade_history[self.strategy_id]
            if len(trades) > 0 and 'profit_pct' in trades.columns:
                trade_metrics = PerformanceMetrics.calculate_trade_metrics(trades)
        
        # Calculate CAGR if sufficient history
        cagr = 0.0
        if len(equity) >= 2:
            start_date = equity.index[0]
            end_date = equity.index[-1]
            years = (end_date - start_date).days / 365.25
            start_value = float(equity.iloc[0])  # Convert to float to handle potential Series
            end_value = float(equity.iloc[-1])   # Convert to float to handle potential Series
            
            if years > 0 and start_value > 0:
                cagr = PerformanceMetrics.calculate_cagr(start_value, end_value, years)
        
        # Calculate other metrics - handle potential divide by zero issues
        total_return = 0.0
        if len(equity) >= 2:
            start_value = float(equity.iloc[0])
            end_value = float(equity.iloc[-1])
            if start_value > 0:
                total_return = (end_value / start_value) - 1
        
        metrics = {
            'total_return': total_return,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'drawdown_start': drawdown_start.strftime('%Y-%m-%d') if drawdown_start else None,
            'drawdown_end': drawdown_end.strftime('%Y-%m-%d') if drawdown_end else None,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(daily_returns, self.risk_free_rate / 252),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(daily_returns, self.risk_free_rate / 252),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(daily_returns, max_drawdown),
            'volatility': PerformanceMetrics.calculate_volatility(daily_returns),
            'monthly_volatility': PerformanceMetrics.calculate_volatility(monthly_returns, 'monthly'),
            'last_value': float(equity.iloc[-1]) if len(equity) > 0 else 0.0,
            'first_date': equity.index[0].strftime('%Y-%m-%d') if len(equity) > 0 else None,
            'last_date': equity.index[-1].strftime('%Y-%m-%d') if len(equity) > 0 else None,
            **trade_metrics
        }
        
        # Add underwater periods analysis
        underwater_periods = PerformanceMetrics.calculate_underwater_periods(equity)
        if isinstance(underwater_periods, pd.DataFrame) and len(underwater_periods) > 0:
            metrics['underwater_periods_count'] = len(underwater_periods)
            metrics['avg_underwater_duration'] = underwater_periods['duration_days'].mean()
            metrics['max_underwater_duration'] = underwater_periods['duration_days'].max()
        
        # Calculate benchmark comparison if available
        benchmark_symbol = self.config.get("performance", "benchmark_symbol", default="SPY")
        if benchmark_symbol in self.benchmark_data:
            benchmark_data = self.benchmark_data[benchmark_symbol]
            
            # Align benchmark data with strategy data
            if 'close' in benchmark_data.columns:
                benchmark_prices = benchmark_data['close']
                aligned_dates = equity.index.intersection(benchmark_prices.index)
                
                if len(aligned_dates) >= 2:
                    # Calculate benchmark returns
                    bench_equity = benchmark_prices.loc[aligned_dates]
                    bench_returns = bench_equity.pct_change().dropna()
                    
                    # Calculate comparison metrics
                    strat_returns = equity.loc[aligned_dates].pct_change().dropna()
                    
                    # Safe calculation of benchmark return
                    bench_start = float(bench_equity.iloc[0])
                    bench_end = float(bench_equity.iloc[-1])
                    bench_return = (bench_end / bench_start - 1) if bench_start > 0 else 0.0
                    
                    metrics.update({
                        'beta': PerformanceMetrics.calculate_beta(strat_returns, bench_returns),
                        'alpha': PerformanceMetrics.calculate_alpha(strat_returns, bench_returns, self.risk_free_rate / 252),
                        'benchmark_return': bench_return,
                        'benchmark_sharpe': PerformanceMetrics.calculate_sharpe_ratio(bench_returns, self.risk_free_rate / 252),
                        'benchmark_volatility': PerformanceMetrics.calculate_volatility(bench_returns),
                        'correlation': strat_returns.corr(bench_returns)
                    })
        
        # Save metrics
        self.strategy_metrics[self.strategy_id] = metrics
        
        return metrics
    
    def save_to_file(self, filepath: Optional[str] = None) -> str:
        """Save performance data to file
        
        Args:
            filepath: Optional file path
            
        Returns:
            str: Path to saved file
        """
        if self.strategy_id not in self.equity_history:
            raise ValueError(f"Strategy {self.strategy_id} not found")
            
        # Generate default filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(
                self.storage_path, 
                f"{self.strategy_id}_performance_{timestamp}.json"
            )
            
        # Calculate metrics
        metrics = self.calculate_metrics()
                
        # Helper function to make data JSON-serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            if isinstance(obj, (pd.Timestamp, datetime)):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            if isinstance(obj, pd.Series):
                return make_serializable(obj.to_dict())
            if isinstance(obj, pd.DataFrame):
                return make_serializable(obj.to_dict(orient='records'))
            if isinstance(obj, np.ndarray):
                return make_serializable(obj.tolist())
            if isinstance(obj, np.generic):
                return obj.item()
                
            try:
                if pd.isna(obj):
                    return None
            except TypeError:
                pass
                
            return obj

        # Helper to safely get history records (修复版)
        def get_history_records(history_dict, reset_index=False):
            entry = history_dict.get(self.strategy_id)
            
            # 确保返回列表类型
            if isinstance(entry, pd.DataFrame):
                df = entry.reset_index() if reset_index else entry
                return df.to_dict(orient='records')
            if isinstance(entry, pd.Series):
                s = entry.reset_index() if reset_index else entry
                return s.to_dict()
            return []  # 统一返回空列表而不是字典

        # Prepare data for serialization
        data = {
            'strategy_id': self.strategy_id,
            'metrics': make_serializable(metrics),
            'equity': make_serializable(
                get_history_records(self.equity_history, reset_index=True)
            ),
            'trades': make_serializable(
                get_history_records(self.trade_history)
            ),
            'drawdown': make_serializable(
                get_history_records(self.drawdown_history, reset_index=True)
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Performance data for {self.strategy_id} saved to {filepath}")
        
        # Save additional CSV files if configured
        if hasattr(self, 'output_formats') and 'csv' in self.output_formats:
            self._save_csv_data(filepath)
        
        return filepath
    
    def _save_csv_data(self, base_filepath: str) -> None:
        """Save additional data in CSV format
        
        Args:
            self.strategy_id: Strategy identifier
            base_filepath: Base file path
        """
        base_path = Path(base_filepath).with_suffix('')
        
        # Save equity curve
        if not self.equity_history[self.strategy_id].empty:
            equity_df = self.equity_history[self.strategy_id].reset_index()
            equity_df.columns = ['timestamp', 'equity']
            equity_csv = f"{base_path}_equity.csv"
            equity_df.to_csv(equity_csv, index=False)
            self.logger.debug(f"Equity curve saved to {equity_csv}")
        
        # Save trades
        if not self.trade_history[self.strategy_id].empty:
            trades_csv = f"{base_path}_trades.csv"
            self.trade_history[self.strategy_id].to_csv(trades_csv, index=False)
            self.logger.debug(f"Trades saved to {trades_csv}")
        
        # Save drawdown
        if not self.drawdown_history[self.strategy_id].empty:
            drawdown_df = self.drawdown_history[self.strategy_id].reset_index()
            drawdown_df.columns = ['timestamp', 'drawdown']
            drawdown_csv = f"{base_path}_drawdown.csv"
            drawdown_df.to_csv(drawdown_csv, index=False)
            self.logger.debug(f"Drawdown history saved to {drawdown_csv}")
    
    def load_from_file(self, filepath: str) -> str:
        """Load performance data from file
        
        Args:
            filepath: File path
            
        Returns:
            str: Strategy ID loaded
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.strategy_id = data.get('self.strategy_id')
        if not self.strategy_id:
            raise ValueError("Invalid performance data file: missing self.strategy_id")
            
        # Register strategy if needed
        if self.strategy_id not in self.equity_history:
            self.register_strategy(self.strategy_id)
            
        # Load equity data
        if 'equity' in data and data['equity']:
            equity_df = pd.DataFrame(data['equity'])
            
            # Handle various timestamp formats
            timestamp_col = None
            for col in ['timestamp', 'date', 'datetime']:
                if col in equity_df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col:
                equity_df[timestamp_col] = pd.to_datetime(equity_df[timestamp_col])
                value_col = [col for col in equity_df.columns if col != timestamp_col][0]
                equity_series = pd.Series(equity_df[value_col].values, index=equity_df[timestamp_col])
                self.equity_history[self.strategy_id] = equity_series
            
        # Load trade data
        if 'trades' in data and data['trades']:
            trade_df = pd.DataFrame(data['trades'])
            
            # Convert timestamp columns to datetime
            for col in ['timestamp', 'entry_time', 'exit_time']:
                if col in trade_df.columns:
                    trade_df[col] = pd.to_datetime(trade_df[col])
                    
            self.trade_history[self.strategy_id] = trade_df
            
        # Load drawdown data
        if 'drawdown' in data and data['drawdown']:
            drawdown_df = pd.DataFrame(data['drawdown'])
            timestamp_col = 'timestamp' if 'timestamp' in drawdown_df.columns else 'date'
            
            if timestamp_col in drawdown_df.columns:
                drawdown_df[timestamp_col] = pd.to_datetime(drawdown_df[timestamp_col])
                value_col = [col for col in drawdown_df.columns if col != timestamp_col][0]
                drawdown_series = pd.Series(drawdown_df[value_col].values, index=drawdown_df[timestamp_col])
                self.drawdown_history[self.strategy_id] = drawdown_series
            
        # Load metrics
        if 'metrics' in data:
            self.strategy_metrics[self.strategy_id] = data['metrics']
            
        self.logger.info(f"Performance data for {self.strategy_id} loaded from {filepath}")
        return self.strategy_id
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report
        
        Args:
            self.strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance report
        """
        # Calculate or retrieve metrics
        metrics = self.calculate_metrics()
                
        # Incorporate additional metrics from backtest engine if available
        if hasattr(self, 'backtest_engine') and self.backtest_engine:
            engine_metrics = self.backtest_engine.get_metrics()
            if engine_metrics:
                # Add engine metrics but avoid overriding existing ones
                for key, value in engine_metrics.items():
                    if key not in metrics:
                        metrics[key] = value
        
        # Generate summary statistics
        summary = {
            'self.strategy_id': self.strategy_id,
            'performance_summary': {
                'total_return': f"{metrics.get('total_return', 0.0) * 100:.2f}%",
                'cagr': f"{metrics.get('cagr', 0.0) * 100:.2f}%",
                'max_drawdown': f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                'sortino_ratio': f"{metrics.get('sortino_ratio', 0.0):.2f}",
                'volatility': f"{metrics.get('volatility', 0.0) * 100:.2f}%"
            },
            'trade_statistics': {
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': f"{metrics.get('win_rate', 0.0) * 100:.2f}%",
                'profit_factor': f"{metrics.get('profit_factor', 0.0):.2f}",
                'avg_trade': f"{metrics.get('avg_trade', 0.0) * 100:.2f}%",
                'expectancy': f"{metrics.get('expectancy', 0.0) * 100:.2f}%"
            },
            'time_period': f"{format_timestamp(metrics.get('first_date'))} to {format_timestamp(metrics.get('last_date'))}",
            'current_value': metrics.get('last_value', 0.0)
        }
        
        # Add benchmark comparison if available
        if 'benchmark_return' in metrics:
            summary['benchmark_comparison'] = {
                'benchmark_return': f"{metrics.get('benchmark_return', 0.0) * 100:.2f}%",
                'alpha': f"{metrics.get('alpha', 0.0) * 100:.2f}%",
                'beta': f"{metrics.get('beta', 0.0):.2f}",
                'correlation': f"{metrics.get('correlation', 0.0):.2f}"
            }
        
        # Add drawdown analysis
        if 'drawdown_start' in metrics and metrics['drawdown_start']:
            summary['drawdown_analysis'] = {
                'max_drawdown': f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%",
                'period': f"{metrics.get('drawdown_start', 'N/A')} to {metrics.get('drawdown_end', 'N/A')}",
                'calmar_ratio': f"{metrics.get('calmar_ratio', 0.0):.2f}"
            }
        
        # Add monthly returns table if available
        monthly_returns = self._calculate_monthly_returns()
        if not monthly_returns.empty:
            summary['monthly_returns'] = monthly_returns.to_dict()
            
            # Add monthly statistics
            positive_months = (monthly_returns > 0).sum().sum()
            total_months = monthly_returns.count().sum()
            if total_months > 0:
                summary['monthly_statistics'] = {
                    'positive_months': positive_months,
                    'negative_months': total_months - positive_months,
                    'positive_months_pct': f"{positive_months / total_months * 100:.2f}%",
                    'best_month': f"{monthly_returns.max().max() * 100:.2f}%",
                    'worst_month': f"{monthly_returns.min().min() * 100:.2f}%"
                }
    
        return {
            'summary': summary,
            'metrics': metrics,
            'trade_count': len(self.trade_history[self.strategy_id]),
            'equity_points': len(self.equity_history[self.strategy_id]),
            'generation_time': datetime.now().isoformat()
        }
    
    def _calculate_monthly_returns(self) -> pd.DataFrame:
        """Calculate monthly returns table
        
        Args:
            self.strategy_id: Strategy identifier
            
        Returns:
            pd.DataFrame: Monthly returns table
        """
        if self.strategy_id not in self.equity_history or self.equity_history[self.strategy_id].empty:
            return pd.DataFrame()
            
        # Get equity curve
        equity = self.equity_history[self.strategy_id]
        
        # Resample to month-end values
        monthly_equity = equity.resample('ME').last()
        
        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Create a table with years as rows and months as columns
        years = sorted(set(monthly_returns.index.year))
        months = range(1, 13)
        
        table = pd.DataFrame(index=years, columns=months)
        
        # Fill the table with returns
        for date, value in monthly_returns.items():
            year = date.year
            month = date.month
            table.loc[year, month] = value
        
        return table
    
    async def _shutdown_analyzer(self) -> None:
        """Analyzer-specific shutdown operations"""
        # Auto-generate final report for system if enabled
        auto_report = self.config.get("performance", "backtest", "auto_generate_report", default=True)
        
        if auto_report and self.strategy_id in self.equity_history:
            try:
                # Generate report for system portfolio
                report = self.generate_performance_report()
                
                # Save report to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = os.path.join(self.report_dir, f"backtest_report_{timestamp}.json")
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
                self.logger.info(f"Final backtest report generated: {report_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate final backtest report: {e}")
    