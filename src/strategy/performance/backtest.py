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
    
    def calculate_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics for a strategy
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance metrics
        """
        if strategy_id not in self.equity_history:
            return {}
            
        # Get equity and returns
        equity = self.equity_history[strategy_id]
        if equity.empty:
            return {}
            
        daily_returns = self.calculate_returns(strategy_id, 'daily')
        monthly_returns = self.calculate_returns(strategy_id, 'monthly')
        
        # Calculate drawdown
        max_drawdown, drawdown_start, drawdown_end = PerformanceMetrics.calculate_max_drawdown(equity)
        
        # Calculate trade metrics
        trades = self.trade_history[strategy_id]
        trade_metrics = {}
        
        if not trades.empty and 'profit_pct' in trades.columns:
            trade_metrics = PerformanceMetrics.calculate_trade_metrics(trades)
        
        # Calculate CAGR if sufficient history
        cagr = 0.0
        if len(equity) >= 2:
            start_date = equity.index[0]
            end_date = equity.index[-1]
            years = (end_date - start_date).days / 365.25
            start_value = equity.iloc[0]
            end_value = equity.iloc[-1]
            
            if years > 0 and start_value > 0:
                cagr = PerformanceMetrics.calculate_cagr(start_value, end_value, years)
        
        # Calculate other metrics
        metrics = {
            'total_return': (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) >= 2 else 0.0,
            'cagr': cagr,
            'max_drawdown': max_drawdown,
            'drawdown_start': drawdown_start.strftime('%Y-%m-%d') if drawdown_start else None,
            'drawdown_end': drawdown_end.strftime('%Y-%m-%d') if drawdown_end else None,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(daily_returns, self.risk_free_rate / 252),
            'sortino_ratio': PerformanceMetrics.calculate_sortino_ratio(daily_returns, self.risk_free_rate / 252),
            'calmar_ratio': PerformanceMetrics.calculate_calmar_ratio(daily_returns, max_drawdown),
            'volatility': PerformanceMetrics.calculate_volatility(daily_returns),
            'monthly_volatility': PerformanceMetrics.calculate_volatility(monthly_returns, 'monthly'),
            'last_value': equity.iloc[-1] if not equity.empty else 0.0,
            'first_date': equity.index[0].strftime('%Y-%m-%d') if not equity.empty else None,
            'last_date': equity.index[-1].strftime('%Y-%m-%d') if not equity.empty else None,
            **trade_metrics
        }
        
        # Add underwater periods analysis
        underwater_periods = PerformanceMetrics.calculate_underwater_periods(equity)
        if not underwater_periods.empty:
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
                    bench_equity = benchmark_prices[aligned_dates]
                    bench_returns = bench_equity.pct_change().dropna()
                    
                    # Calculate comparison metrics
                    strat_returns = equity[aligned_dates].pct_change().dropna()
                    
                    metrics.update({
                        'beta': PerformanceMetrics.calculate_beta(strat_returns, bench_returns),
                        'alpha': PerformanceMetrics.calculate_alpha(strat_returns, bench_returns, self.risk_free_rate / 252),
                        'benchmark_return': (bench_equity.iloc[-1] / bench_equity.iloc[0] - 1),
                        'benchmark_sharpe': PerformanceMetrics.calculate_sharpe_ratio(bench_returns, self.risk_free_rate / 252),
                        'benchmark_volatility': PerformanceMetrics.calculate_volatility(bench_returns),
                        'correlation': strat_returns.corr(bench_returns)
                    })
        
        # Save metrics
        self.strategy_metrics[strategy_id] = metrics
        
        return metrics
    
    def save_to_file(self, strategy_id: str, filepath: Optional[str] = None) -> str:
        """Save performance data to file
        
        Args:
            strategy_id: Strategy identifier
            filepath: Optional file path
            
        Returns:
            str: Path to saved file
        """
        if strategy_id not in self.equity_history:
            raise ValueError(f"Strategy {strategy_id} not found")
            
        # Generate default filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.storage_path, f"{strategy_id}_performance_{timestamp}.json")
            
        # Calculate metrics
        metrics = self.calculate_metrics(strategy_id)
        
        # Prepare data for serialization
        data = {
            'strategy_id': strategy_id,
            'metrics': metrics,
            'equity': self.equity_history[strategy_id].reset_index().to_dict(orient='records') if not self.equity_history[strategy_id].empty else [],
            'trades': self.trade_history[strategy_id].to_dict(orient='records') if not self.trade_history[strategy_id].empty else [],
            'drawdown': self.drawdown_history[strategy_id].reset_index().to_dict(orient='records') if not self.drawdown_history[strategy_id].empty else [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Performance data for {strategy_id} saved to {filepath}")
        
        # Save additional CSV files if configured
        if 'csv' in self.output_formats:
            self._save_csv_data(strategy_id, filepath)
        
        return filepath
    
    def _save_csv_data(self, strategy_id: str, base_filepath: str) -> None:
        """Save additional data in CSV format
        
        Args:
            strategy_id: Strategy identifier
            base_filepath: Base file path
        """
        base_path = Path(base_filepath).with_suffix('')
        
        # Save equity curve
        if not self.equity_history[strategy_id].empty:
            equity_df = self.equity_history[strategy_id].reset_index()
            equity_df.columns = ['timestamp', 'equity']
            equity_csv = f"{base_path}_equity.csv"
            equity_df.to_csv(equity_csv, index=False)
            self.logger.debug(f"Equity curve saved to {equity_csv}")
        
        # Save trades
        if not self.trade_history[strategy_id].empty:
            trades_csv = f"{base_path}_trades.csv"
            self.trade_history[strategy_id].to_csv(trades_csv, index=False)
            self.logger.debug(f"Trades saved to {trades_csv}")
        
        # Save drawdown
        if not self.drawdown_history[strategy_id].empty:
            drawdown_df = self.drawdown_history[strategy_id].reset_index()
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
            
        strategy_id = data.get('strategy_id')
        if not strategy_id:
            raise ValueError("Invalid performance data file: missing strategy_id")
            
        # Register strategy if needed
        if strategy_id not in self.equity_history:
            self.register_strategy(strategy_id)
            
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
                self.equity_history[strategy_id] = equity_series
            
        # Load trade data
        if 'trades' in data and data['trades']:
            trade_df = pd.DataFrame(data['trades'])
            
            # Convert timestamp columns to datetime
            for col in ['timestamp', 'entry_time', 'exit_time']:
                if col in trade_df.columns:
                    trade_df[col] = pd.to_datetime(trade_df[col])
                    
            self.trade_history[strategy_id] = trade_df
            
        # Load drawdown data
        if 'drawdown' in data and data['drawdown']:
            drawdown_df = pd.DataFrame(data['drawdown'])
            timestamp_col = 'timestamp' if 'timestamp' in drawdown_df.columns else 'date'
            
            if timestamp_col in drawdown_df.columns:
                drawdown_df[timestamp_col] = pd.to_datetime(drawdown_df[timestamp_col])
                value_col = [col for col in drawdown_df.columns if col != timestamp_col][0]
                drawdown_series = pd.Series(drawdown_df[value_col].values, index=drawdown_df[timestamp_col])
                self.drawdown_history[strategy_id] = drawdown_series
            
        # Load metrics
        if 'metrics' in data:
            self.strategy_metrics[strategy_id] = data['metrics']
            
        self.logger.info(f"Performance data for {strategy_id} loaded from {filepath}")
        return strategy_id
    
    def generate_performance_report(self, strategy_id: str) -> Dict[str, Any]:
        """Generate a comprehensive performance report
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            Dict[str, Any]: Performance report
        """
        # Calculate or retrieve metrics
        metrics = self.calculate_metrics(strategy_id)
        
        # Generate summary statistics
        summary = {
            'strategy_id': strategy_id,
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
            'time_period': f"{metrics.get('first_date', 'N/A')} to {metrics.get('last_date', 'N/A')}",
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
        monthly_returns = self._calculate_monthly_returns(strategy_id)
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
            'trade_count': len(self.trade_history[strategy_id]),
            'equity_points': len(self.equity_history[strategy_id]),
            'generation_time': datetime.now().isoformat()
        }
    
    def _calculate_monthly_returns(self, strategy_id: str) -> pd.DataFrame:
        """Calculate monthly returns table
        
        Args:
            strategy_id: Strategy identifier
            
        Returns:
            pd.DataFrame: Monthly returns table
        """
        if strategy_id not in self.equity_history or self.equity_history[strategy_id].empty:
            return pd.DataFrame()
            
        # Get equity curve
        equity = self.equity_history[strategy_id]
        
        # Resample to month-end values
        monthly_equity = equity.resample('M').last()
        
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
        
        if auto_report and 'system' in self.equity_history and not self.equity_history['system'].empty:
            try:
                # Generate report for system portfolio
                report = self.generate_performance_report('system')
                
                # Save report to file
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                report_path = os.path.join(self.report_dir, f"backtest_report_{timestamp}.json")
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                    
                self.logger.info(f"Final backtest report generated: {report_path}")
                
            except Exception as e:
                self.logger.error(f"Failed to generate final backtest report: {e}")