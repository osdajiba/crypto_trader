#!/usr/bin/env python3
# src/strategy/performance/metrics.py

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from datetime import datetime


class PerformanceMetrics:
    """Static class providing common performance metric calculations"""
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, period: str = 'daily') -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of period returns
            risk_free_rate: Risk-free rate (default: 0)
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            float: Sharpe ratio
        """
        if returns.empty:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Calculate Sharpe ratio
        annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }.get(period.lower(), 252)
        
        std_dev = returns.std()
        if std_dev == 0:
            return 0.0
            
        sharpe = excess_returns.mean() / std_dev * np.sqrt(annualization_factor)
        return float(sharpe)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, period: str = 'daily') -> float:
        """
        Calculate Sortino ratio (only considers downside risk)
        
        Args:
            returns: Series of period returns
            risk_free_rate: Risk-free rate (default: 0)
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            float: Sortino ratio
        """
        if returns.empty:
            return 0.0
            
        # Calculate excess returns
        excess_returns = returns - risk_free_rate
        
        # Calculate downside deviation (standard deviation of negative returns only)
        downside_returns = returns[returns < 0]
        if downside_returns.empty:
            return float('inf')  # No downside risk
            
        downside_deviation = downside_returns.std()
        if downside_deviation == 0:
            return 0.0
            
        # Calculate Sortino ratio
        annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }.get(period.lower(), 252)
        
        sortino = excess_returns.mean() / downside_deviation * np.sqrt(annualization_factor)
        return float(sortino)
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """
        Calculate maximum drawdown and drawdown period
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            Tuple[float, pd.Timestamp, pd.Timestamp]: (max drawdown, start date, end date)
        """
        if equity_curve.empty:
            return 0.0, None, None
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown start and end dates
        end_date = drawdown.idxmin()
        
        # Find the last peak before the maximum drawdown
        drawdown_start = drawdown[:end_date]
        start_date = running_max[:end_date].idxmax()
        
        return float(max_drawdown), start_date, end_date
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame, pnl_column: str = 'profit_pct') -> float:
        """
        Calculate win rate from trades
        
        Args:
            trades: DataFrame of trade records
            pnl_column: Column name for profit/loss
            
        Returns:
            float: Win rate (0.0-1.0)
        """
        if trades.empty:
            return 0.0
            
        # Count winning trades
        winning_trades = trades[trades[pnl_column] > 0]
        
        # Calculate win rate
        win_rate = len(winning_trades) / len(trades)
        return float(win_rate)
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame, pnl_column: str = 'profit_pct') -> float:
        """
        Calculate profit factor (gross profit / gross loss)
        
        Args:
            trades: DataFrame of trade records
            pnl_column: Column name for profit/loss
            
        Returns:
            float: Profit factor
        """
        if trades.empty:
            return 0.0
            
        # Separate winning and losing trades
        winning_trades = trades[trades[pnl_column] > 0]
        losing_trades = trades[trades[pnl_column] < 0]
        
        # Calculate gross profit and loss
        gross_profit = winning_trades[pnl_column].sum()
        gross_loss = abs(losing_trades[pnl_column].sum())
        
        # Calculate profit factor
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
            
        profit_factor = gross_profit / gross_loss
        return float(profit_factor)
    
    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame, pnl_column: str = 'profit_pct') -> float:
        """
        Calculate expectancy (expected profit per trade)
        
        Args:
            trades: DataFrame of trade records
            pnl_column: Column name for profit/loss
            
        Returns:
            float: Expectancy
        """
        if trades.empty:
            return 0.0
            
        # Calculate win rate
        win_rate = PerformanceMetrics.calculate_win_rate(trades, pnl_column)
        
        # Calculate average win and loss
        winning_trades = trades[trades[pnl_column] > 0]
        losing_trades = trades[trades[pnl_column] < 0]
        
        avg_win = winning_trades[pnl_column].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades[pnl_column].mean() if not losing_trades.empty else 0
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        return float(expectancy)
    
    @staticmethod
    def calculate_cagr(start_value: float, end_value: float, years: float) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            start_value: Initial investment value
            end_value: Final investment value
            years: Investment period in years
            
        Returns:
            float: CAGR
        """
        if start_value <= 0 or years <= 0:
            return 0.0
            
        cagr = (end_value / start_value) ** (1 / years) - 1
        return float(cagr)
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_drawdown: float, period: str = 'daily') -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown)
        
        Args:
            returns: Series of period returns
            max_drawdown: Maximum drawdown (positive value)
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            float: Calmar ratio
        """
        if returns.empty or max_drawdown >= 0:
            return 0.0
            
        # Annualize returns
        annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }.get(period.lower(), 252)
        
        annualized_return = returns.mean() * annualization_factor
        
        # Calculate Calmar ratio
        calmar = annualized_return / abs(max_drawdown)
        return float(calmar)
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, period: str = 'daily') -> float:
        """
        Calculate annualized volatility
        
        Args:
            returns: Series of period returns
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            float: Annualized volatility
        """
        if returns.empty:
            return 0.0
            
        # Annualize volatility
        annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }.get(period.lower(), 252)
        
        volatility = returns.std() * np.sqrt(annualization_factor)
        return float(volatility)
    
    @staticmethod
    def calculate_beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """
        Calculate beta (sensitivity to market)
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            float: Beta
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
            
        # Ensure index alignment
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if aligned_returns.empty:
            return 0.0
            
        strategy_returns = aligned_returns.iloc[:, 0]
        market_returns = aligned_returns.iloc[:, 1]
        
        # Calculate covariance and variance
        covariance = strategy_returns.cov(market_returns)
        variance = market_returns.var()
        
        if variance == 0:
            return 0.0
            
        beta = covariance / variance
        return float(beta)
    
    @staticmethod
    def calculate_alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0, period: str = 'daily') -> float:
        """
        Calculate alpha (excess return over the benchmark)
        
        Args:
            returns: Series of strategy returns
            benchmark_returns: Series of benchmark returns
            risk_free_rate: Risk-free rate (default: 0)
            period: Return period ('daily', 'weekly', 'monthly', 'annual')
            
        Returns:
            float: Alpha
        """
        if returns.empty or benchmark_returns.empty:
            return 0.0
            
        # Calculate beta
        beta = PerformanceMetrics.calculate_beta(returns, benchmark_returns)
        
        # Ensure index alignment
        aligned_returns = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if aligned_returns.empty:
            return 0.0
            
        strategy_returns = aligned_returns.iloc[:, 0]
        market_returns = aligned_returns.iloc[:, 1]
        
        # Annualize returns
        annualization_factor = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'annual': 1
        }.get(period.lower(), 252)
        
        strategy_return = strategy_returns.mean() * annualization_factor
        market_return = market_returns.mean() * annualization_factor
        
        # Calculate alpha
        alpha = strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))
        return float(alpha)
    
    @staticmethod
    def convert_returns_to_equity(returns: pd.Series, initial_capital: float = 10000.0) -> pd.Series:
        """
        Convert returns series to equity curve
        
        Args:
            returns: Series of period returns
            initial_capital: Initial capital
            
        Returns:
            pd.Series: Equity curve
        """
        if returns.empty:
            return pd.Series()
            
        # Calculate compounded returns
        compounded_returns = (1 + returns).cumprod()
        
        # Convert to equity curve
        equity_curve = compounded_returns * initial_capital
        
        return equity_curve
    
    @staticmethod
    def calculate_recovery_time(drawdown_series: pd.Series, start_date: pd.Timestamp) -> Optional[int]:
        """
        Calculate recovery time from drawdown (in days)
        
        Args:
            drawdown_series: Series of drawdown values
            start_date: Start date of drawdown
            
        Returns:
            Optional[int]: Recovery time in days or None if not recovered
        """
        if drawdown_series.empty or start_date not in drawdown_series.index:
            return None
            
        # Get drawdown values after start date
        post_drawdown = drawdown_series[drawdown_series.index >= start_date]
        
        # Find first date when drawdown returns to 0
        recovery_dates = post_drawdown[post_drawdown >= 0].index
        
        if len(recovery_dates) == 0:
            return None  # Not recovered yet
            
        first_recovery = recovery_dates[0]
        
        # Calculate days between drawdown start and recovery
        recovery_days = (first_recovery - start_date).days
        
        return recovery_days
    
    @staticmethod
    def calculate_underwater_periods(equity_curve: pd.Series) -> pd.DataFrame:
        """
        Identify underwater periods (drawdowns)
        
        Args:
            equity_curve: Series of equity values
            
        Returns:
            pd.DataFrame: Underwater periods with start, end, depth
        """
        if equity_curve.empty:
            return pd.DataFrame()
            
        # Calculate running maximum
        running_max = equity_curve.cummax()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        # Identify periods where drawdown < 0
        is_underwater = drawdown < 0
        
        # Find transition points
        transitions = is_underwater.astype(int).diff().fillna(0)
        
        # Extract underwater periods
        underwater_periods = []
        
        # Track current period
        period_start = None
        
        # Iterate through transitions
        for date, value in transitions.items():
            if value == 1:  # Start of underwater period
                period_start = date
            elif value == -1 and period_start is not None:  # End of underwater period
                # Calculate max drawdown during this period
                period_drawdown = drawdown[period_start:date].min()
                underwater_periods.append({
                    'start_date': period_start,
                    'end_date': date,
                    'duration_days': (date - period_start).days,
                    'max_drawdown': float(period_drawdown)
                })
                period_start = None
        
        # Check if still underwater at the end
        if period_start is not None:
            period_drawdown = drawdown[period_start:].min()
            underwater_periods.append({
                'start_date': period_start,
                'end_date': equity_curve.index[-1],
                'duration_days': (equity_curve.index[-1] - period_start).days,
                'max_drawdown': float(period_drawdown),
                'recovered': False
            })
        
        return pd.DataFrame(underwater_periods)
    
    @staticmethod
    def calculate_rolling_metrics(returns: pd.Series, window: int = 30) -> Dict[str, pd.Series]:
        """
        Calculate rolling performance metrics
        
        Args:
            returns: Series of returns
            window: Rolling window size
            
        Returns:
            Dict[str, pd.Series]: Dictionary of rolling metrics
        """
        if returns.empty or len(returns) < window:
            return {}
            
        # Calculate rolling mean
        rolling_return = returns.rolling(window=window).mean() * 252
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
        
        # Calculate rolling Sharpe
        rolling_sharpe = rolling_return / rolling_vol
        
        # Calculate rolling drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_max = equity_curve.rolling(window=window).max()
        rolling_drawdown = (equity_curve - rolling_max) / rolling_max
        
        return {
            'return': rolling_return,
            'volatility': rolling_vol,
            'sharpe': rolling_sharpe,
            'drawdown': rolling_drawdown.min()
        }
    
    @staticmethod
    def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive trade metrics
        
        Args:
            trades: DataFrame of trade records
            
        Returns:
            Dict[str, Any]: Trade metrics
        """
        if trades.empty:
            return {}
            
        # Ensure profit_pct column exists
        if 'profit_pct' not in trades.columns:
            return {}
            
        # Basic metrics
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
        expectancy = PerformanceMetrics.calculate_expectancy(trades)
        
        # Winning and losing trades
        winning_trades = trades[trades['profit_pct'] > 0]
        losing_trades = trades[trades['profit_pct'] < 0]
        
        # Calculate average metrics
        avg_win = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0.0
        avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0.0
        
        # Calculate trade durations if timestamp columns exist
        duration_metrics = {}
        if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
            trades_with_duration = trades.copy()
            trades_with_duration['duration'] = (
                pd.to_datetime(trades_with_duration['exit_time']) - 
                pd.to_datetime(trades_with_duration['entry_time'])
            )
            
            duration_metrics = {
                'avg_trade_duration': trades_with_duration['duration'].mean().total_seconds() / 3600,  # in hours
                'avg_win_duration': trades_with_duration.loc[trades_with_duration['profit_pct'] > 0, 'duration'].mean().total_seconds() / 3600 if not winning_trades.empty else 0.0,
                'avg_loss_duration': trades_with_duration.loc[trades_with_duration['profit_pct'] < 0, 'duration'].mean().total_seconds() / 3600 if not losing_trades.empty else 0.0
            }
        
        # Combine all metrics
        metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'avg_trade': trades['profit_pct'].mean(),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'largest_win': winning_trades['profit_pct'].max() if not winning_trades.empty else 0.0,
            'largest_loss': losing_trades['profit_pct'].min() if not losing_trades.empty else 0.0,
            'consecutive_wins': PerformanceMetrics._calculate_max_consecutive(trades, 'profit_pct', lambda x: x > 0),
            'consecutive_losses': PerformanceMetrics._calculate_max_consecutive(trades, 'profit_pct', lambda x: x < 0),
            **duration_metrics
        }
        
        return metrics
    
    @staticmethod
    def _calculate_max_consecutive(df: pd.DataFrame, column: str, condition: callable) -> int:
        """Helper method to calculate maximum consecutive occurrences"""
        if df.empty:
            return 0
            
        # Create series indicating if condition is met
        condition_met = condition(df[column])
        
        # Create groups by condition change
        groups = (condition_met != condition_met.shift()).cumsum()
        
        # Count consecutive occurrences
        counts = df.groupby(groups).apply(lambda x: len(x) if condition(x[column].iloc[0]) else 0)
        
        # Return maximum
        return counts.max() if not counts.empty else 0