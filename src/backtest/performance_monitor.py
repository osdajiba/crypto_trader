# src/backtest/performance_monitor.py

import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import scipy.stats as stats
from dataclasses import dataclass
from src.common.log_manager import LogManager

@dataclass
class PerformanceMetrics:
    """
    Dataclass to hold comprehensive performance metrics
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    commission_paid: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

class PerformanceMonitor:
    """
    Advanced Performance Monitoring System with Optimized Metrics Calculation
    """
    def __init__(
        self, 
        config: Dict, 
        initial_balance: float = 100000, 
        benchmark: Optional[str] = None
    ):
        """
        Initialize Performance Monitor with enhanced tracking capabilities
        
        Args:
            config (Dict): Configuration dictionary
            initial_balance (float): Starting capital
            benchmark (Optional[str]): Benchmark symbol for comparison
        """
        # Logging
        self.logger = LogManager.get_logger("performance_monitor")
        
        # Configuration
        self.config = config
        self.initial_balance = initial_balance
        self.benchmark = benchmark
        
        # Performance Tracking with NumPy for Efficiency
        self.performance_metrics = PerformanceMetrics()
        
        # Time Series Tracking with NumPy Arrays
        self.equity_curve = {
            'timestamps': np.array([], dtype='datetime64[ns]'),
            'balance': np.array([], dtype=np.float64),
            'daily_return': np.array([], dtype=np.float64),
            'cumulative_return': np.array([], dtype=np.float64)
        }
        
        # Trade Log with NumPy Structured Array
        self.trade_log_dtype = [
            ('timestamp', 'datetime64[ns]'),
            ('symbol', 'U20'),
            ('direction', 'U10'),
            ('entry_price', np.float64),
            ('exit_price', np.float64),
            ('quantity', np.float64),
            ('profit_loss', np.float64),
            ('return_pct', np.float64)
        ]
        self.trade_log = np.array([], dtype=self.trade_log_dtype)
    
    def record_trade(
        self, 
        timestamp: pd.Timestamp, 
        symbol: str, 
        direction: str, 
        entry_price: float, 
        exit_price: float, 
        quantity: float, 
        commission: float
    ):
        """
        Record individual trade with optimized NumPy storage
        
        Args:
            timestamp (pd.Timestamp): Trade timestamp
            symbol (str): Trading symbol
            direction (str): Trade direction (buy/sell)
            entry_price (float): Entry price
            exit_price (float): Exit price
            quantity (float): Trade quantity
            commission (float): Transaction commission
        """
        # Calculate trade profit/loss and return percentage
        profit_loss = (exit_price - entry_price) * quantity * (1 if direction == 'buy' else -1)
        return_pct = profit_loss / (entry_price * quantity)
        
        # Create trade record
        trade_record = np.array([
            (timestamp, symbol, direction, entry_price, exit_price, 
             quantity, profit_loss, return_pct)
        ], dtype=self.trade_log_dtype)
        
        # Append trade record efficiently
        self.trade_log = np.append(self.trade_log, trade_record)
        
        # Update performance metrics
        self.performance_metrics.total_trades += 1
        self.performance_metrics.commission_paid += commission
        
        if profit_loss > 0:
            self.performance_metrics.winning_trades += 1
            self.performance_metrics.total_profit += profit_loss
        else:
            self.performance_metrics.losing_trades += 1
            self.performance_metrics.total_loss += abs(profit_loss)
    
    def update_equity_curve(
        self, 
        timestamp: pd.Timestamp, 
        current_balance: float
    ):
        """
        Update equity curve with NumPy-based tracking
        
        Args:
            timestamp (pd.Timestamp): Current timestamp
            current_balance (float): Current portfolio balance
        """
        # Calculate returns
        last_balance = (
            self.equity_curve['balance'][-1] 
            if len(self.equity_curve['balance']) > 0 
            else self.initial_balance
        )
        daily_return = (current_balance - last_balance) / last_balance
        cumulative_return = current_balance / self.initial_balance - 1
        
        # Append to NumPy arrays
        self.equity_curve['timestamps'] = np.append(
            self.equity_curve['timestamps'], 
            np.datetime64(timestamp)
        )
        self.equity_curve['balance'] = np.append(
            self.equity_curve['balance'], 
            current_balance
        )
        self.equity_curve['daily_return'] = np.append(
            self.equity_curve['daily_return'], 
            daily_return
        )
        self.equity_curve['cumulative_return'] = np.append(
            self.equity_curve['cumulative_return'], 
            cumulative_return
        )
    
    def calculate_performance_metrics(self):
        """
        Calculate comprehensive performance metrics using NumPy
        """
        # Basic metrics
        if self.performance_metrics.total_trades > 0:
            self.performance_metrics.win_rate = (
                self.performance_metrics.winning_trades / 
                self.performance_metrics.total_trades
            )
            
            # Profit Factor
            total_profit = self.performance_metrics.total_profit
            total_loss = self.performance_metrics.total_loss
            self.performance_metrics.profit_factor = (
                total_profit / total_loss if total_loss > 0 else float('inf')
            )
        
        # Returns analysis (using NumPy for efficiency)
        if len(self.equity_curve['daily_return']) > 0:
            returns = self.equity_curve['daily_return']
            
            # Risk-free rate from configuration
            risk_free_rate = self.config.get('risk_free_rate', 0.02)
            
            # Annualized return and volatility
            annualized_return = (1 + returns.mean()) ** 252 - 1
            annualized_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            self.performance_metrics.sharpe_ratio = (
                (annualized_return - risk_free_rate) / annualized_volatility 
                if annualized_volatility > 0 else 0
            )
            
            # Sortino Ratio (downside risk)
            downside_returns = returns[returns < 0]
            downside_volatility = (
                downside_returns.std() * np.sqrt(252) 
                if len(downside_returns) > 0 else 0
            )
            
            self.performance_metrics.sortino_ratio = (
                (annualized_return - risk_free_rate) / downside_volatility 
                if downside_volatility > 0 else 0
            )
    
    def generate_detailed_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dict[str, Any]: Detailed performance report
        """
        # Ensure metrics are up to date
        self.calculate_performance_metrics()
        
        # Convert NumPy structures to standard Python types
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': (
                    self.equity_curve['balance'][-1] 
                    if len(self.equity_curve['balance']) > 0 
                    else self.initial_balance
                ),
                'total_return_pct': (
                    (self.equity_curve['balance'][-1] / self.initial_balance - 1) * 100 
                    if len(self.equity_curve['balance']) > 0 else 0
                )
            },
            'performance_metrics': {
                'total_trades': self.performance_metrics.total_trades,
                'winning_trades': self.performance_metrics.winning_trades,
                'losing_trades': self.performance_metrics.losing_trades,
                'total_profit': self.performance_metrics.total_profit,
                'total_loss': self.performance_metrics.total_loss,
                'commission_paid': self.performance_metrics.commission_paid,
                'win_rate': self.performance_metrics.win_rate,
                'profit_factor': self.performance_metrics.profit_factor,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                'sortino_ratio': self.performance_metrics.sortino_ratio
            },
            'trade_log': [
                {name: trade[i] for i, name in enumerate(self.trade_log_dtype)} 
                for trade in self.trade_log
            ],
            'equity_curve': {
                'timestamps': [pd.Timestamp(ts) for ts in self.equity_curve['timestamps']],
                'balance': self.equity_curve['balance'].tolist(),
                'daily_return': self.equity_curve['daily_return'].tolist(),
                'cumulative_return': self.equity_curve['cumulative_return'].tolist()
            }
        }
        
        return report
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Returns:
            Dict[str, Any]: Detailed performance report
        """
        # Ensure metrics are up to date
        self.calculate_performance_metrics()
        
        # Convert NumPy structures to standard Python types
        report = {
            'summary': {
                'initial_balance': self.initial_balance,
                'final_balance': (
                    self.equity_curve['balance'][-1] 
                    if len(self.equity_curve['balance']) > 0 
                    else self.initial_balance
                ),
                'total_return_pct': (
                    (self.equity_curve['balance'][-1] / self.initial_balance - 1) * 100 
                    if len(self.equity_curve['balance']) > 0 else 0
                )
            },
            'performance_metrics': {
                'total_trades': self.performance_metrics.total_trades,
                'winning_trades': self.performance_metrics.winning_trades,
                'losing_trades': self.performance_metrics.losing_trades,
                'total_profit': self.performance_metrics.total_profit,
                'total_loss': self.performance_metrics.total_loss,
                'commission_paid': self.performance_metrics.commission_paid,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
            }
        }
        
        return report
    
    def analyze_trade_distribution(self) -> Dict[str, Any]:
        """
        Perform advanced statistical analysis of trade returns
        
        Returns:
            Dict[str, Any]: Comprehensive trade return distribution analysis
        """
        if len(self.trade_log) == 0:
            return {}
        
        # Extract returns using NumPy
        returns = np.array([trade['return_pct'] for trade in self.trade_log])
        
        try:
            # Comprehensive statistical analysis
            distribution_analysis = {
                'descriptive_stats': {
                    'mean': np.mean(returns),
                    'median': np.median(returns),
                    'std_dev': np.std(returns),
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns)
                },
                'normality_tests': {
                    # Shapiro-Wilk test for normality
                    'shapiro_test': {
                        'statistic': stats.shapiro(returns)[0],
                        'p_value': stats.shapiro(returns)[1]
                    }
                },
                'percentiles': {
                    '5th': np.percentile(returns, 5),
                    '25th': np.percentile(returns, 25),
                    '50th': np.percentile(returns, 50),
                    '75th': np.percentile(returns, 75),
                    '95th': np.percentile(returns, 95)
                }
            }
            
            return distribution_analysis
        except Exception as e:
            self.logger.error(f"Trade distribution analysis error: {e}")
            return {}
    
    def visualize_performance(self, output_dir: str = 'reports'):
        """
        Generate advanced performance visualizations
        
        Args:
            output_dir (str): Directory to save visualization files
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Color palette
        colors = sns.color_palette("husl", 3)
        
        # 1. Equity Curve with Matplotlib
        plt.figure(figsize=(14, 7))
        plt.plot(
            self.equity_curve['timestamps'], 
            self.equity_curve['balance'], 
            color=colors[0], 
            linewidth=2
        )
        plt.title('Portfolio Equity Curve', fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Portfolio Value', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'), dpi=300)
        plt.close()
        
        # 2. Trade Returns Distribution with Seaborn
        plt.figure(figsize=(14, 7))
        returns = [trade['return_pct'] for trade in self.trade_log]
        sns.histplot(returns, kde=True, color=colors[1])
        plt.title('Trade Returns Distribution', fontsize=16)
        plt.xlabel('Return Percentage', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'returns_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Drawdown Chart
        cumulative_max = np.maximum.accumulate(self.equity_curve['balance'])
        drawdown = (self.equity_curve['balance'] - cumulative_max) / cumulative_max
        
        plt.figure(figsize=(14, 7))
        plt.plot(
            self.equity_curve['timestamps'], 
            drawdown * 100, 
            color=colors[2], 
            linewidth=2
        )
        plt.title('Portfolio Drawdown', fontsize=16)
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'drawdown.png'), dpi=300)
        plt.close()
        
    async def close(self):
        """
        Perform cleanup and finalization of performance tracking
        
        This method can be used to:
        - Generate final performance report
        - Export performance logs
        - Clean up resources
        """
        try:
            # Ensure all metrics are calculated
            self.calculate_performance_metrics()
            
            # Generate final detailed report
            final_report = self.generate_detailed_report()
            
            # Log final performance metrics
            self.logger.info("Closing Performance Monitor")
            self.logger.info(f"Final Performance Summary: {final_report['summary']}")
            
            # Export performance report
            self.export_report(format='json', output_dir='reports/final')
            
            # Optional: Visualize final performance
            self.visualize_performance(output_dir='reports/final')
            
            # Reset or clean up resources
            self.trade_log = np.array([], dtype=self.trade_log_dtype)
            self.equity_curve = {
                'timestamps': np.array([], dtype='datetime64[ns]'),
                'balance': np.array([], dtype=np.float64),
                'daily_return': np.array([], dtype=np.float64),
                'cumulative_return': np.array([], dtype=np.float64)
            }
            
            # Reset performance metrics
            self.performance_metrics = PerformanceMetrics()
        
        except Exception as e:
            self.logger.error(f"Error during performance monitor close: {e}")

    @property
    def snapshot(self) -> Dict[str, float]:
        """
        Get a snapshot of current performance metrics
        
        Returns:
            Dict[str, float]: Current performance snapshot
        """
        try:
            # Calculate current portfolio value
            current_balance = (
                self.equity_curve['balance'][-1] 
                if len(self.equity_curve['balance']) > 0 
                else self.initial_balance
            )
            
            # Calculate daily return (most recent)
            daily_return = (
                self.equity_curve['daily_return'][-1] 
                if len(self.equity_curve['daily_return']) > 0 
                else 0
            )
            
            # Calculate maximum drawdown
            cumulative_max = self.equity_curve['balance'].cummax() if len(self.equity_curve['balance']) > 0 else self.initial_balance
            drawdown = (
                (self.equity_curve['balance'][-1] - cumulative_max) / cumulative_max 
                if len(self.equity_curve['balance']) > 0 
                else 0
            )
            
            return {
                'current_equity': current_balance,
                'daily_return': daily_return,
                'max_drawdown': abs(drawdown)
            }
        
        except Exception as e:
            self.logger.error(f"Error generating performance snapshot: {e}")
            return {
                'current_equity': self.initial_balance,
                'daily_return': 0,
                'max_drawdown': 0
            }