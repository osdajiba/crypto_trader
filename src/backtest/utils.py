#!/usr/bin/env python3
# src/backtest/utils.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime, timedelta


def calculate_metrics(equity_curve: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate various performance metrics from an equity curve
    
    Args:
        equity_curve: DataFrame with 'timestamp' and 'portfolio_value' columns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Dictionary of performance metrics
    """
    if equity_curve.empty or 'portfolio_value' not in equity_curve.columns:
        return {}
    
    # Calculate returns
    equity_curve['return'] = equity_curve['portfolio_value'].pct_change()
    equity_curve['log_return'] = np.log(equity_curve['portfolio_value'] / equity_curve['portfolio_value'].shift(1))
    equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(equity_curve) < 2:
        return {}
    
    # Get values
    returns = equity_curve['return'].values
    log_returns = equity_curve['log_return'].values
    
    # Time period adjustments
    period = estimate_period(equity_curve['timestamp'])
    periods_per_year = periods_in_year(period)
    
    # Basic metrics
    total_return = (equity_curve['portfolio_value'].iloc[-1] / equity_curve['portfolio_value'].iloc[0]) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(equity_curve)) - 1
    
    # Calculate drawdown
    cumulative_max = np.maximum.accumulate(equity_curve['portfolio_value'])
    drawdown = (equity_curve['portfolio_value'] - cumulative_max) / cumulative_max
    max_drawdown = abs(min(drawdown))
    
    # Risk metrics
    daily_std = np.std(returns)
    annualized_vol = daily_std * np.sqrt(periods_per_year)
    
    # Only calculate Sharpe if we have a positive standard deviation
    if annualized_vol > 0:
        daily_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol
    else:
        sharpe_ratio = np.nan
    
    # Downside deviation (only negative returns)
    downside_returns = np.where(returns < 0, returns, 0)
    downside_std = np.std(downside_returns)
    
    # Sortino ratio (if we have negative returns)
    if downside_std > 0:
        sortino_ratio = (annualized_return - risk_free_rate) / (downside_std * np.sqrt(periods_per_year))
    else:
        sortino_ratio = np.nan
    
    # Calmar ratio (return / max drawdown)
    if max_drawdown > 0:
        calmar_ratio = annualized_return / max_drawdown
    else:
        calmar_ratio = np.nan
    
    return {
        'total_return': total_return * 100,  # Convert to percentage
        'annualized_return': annualized_return * 100,  # Convert to percentage
        'annualized_volatility': annualized_vol * 100,  # Convert to percentage
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown': max_drawdown * 100,  # Convert to percentage
    }


def estimate_period(timestamps) -> str:
    """
    Estimate the period of a time series (daily, hourly, etc.)
    
    Args:
        timestamps: Series or list of timestamps
        
    Returns:
        String indicating the period ('D' for daily, 'H' for hourly, etc.)
    """
    if len(timestamps) < 2:
        return 'D'  # Default to daily
    
    # Convert to pandas datetime if not already
    if not isinstance(timestamps, pd.DatetimeIndex):
        timestamps = pd.to_datetime(timestamps)
    
    # Calculate differences between consecutive timestamps
    diffs = pd.Series(timestamps).diff().dropna()
    
    # Get the most common difference
    if len(diffs) == 0:
        return 'D'
    
    median_diff = diffs.median()
    
    # Classify based on the median difference
    if median_diff < timedelta(minutes=5):
        return 'min'
    elif median_diff < timedelta(hours=1):
        return 'min'
    elif median_diff < timedelta(days=1):
        return 'H'
    elif median_diff < timedelta(weeks=1):
        return 'D'
    elif median_diff < timedelta(days=90):
        return 'W'
    else:
        return 'M'


def periods_in_year(period: str) -> int:
    """
    Get the number of periods in a year for a given period
    
    Args:
        period: Period string ('D' for daily, 'H' for hourly, etc.)
        
    Returns:
        Number of periods in a year
    """
    period_map = {
        'min': 525600,  # 365 * 24 * 60
        'H': 8760,      # 365 * 24
        'D': 365,
        'W': 52,
        'M': 12
    }
    return period_map.get(period, 252)  # Default to 252 trading days


def calculate_trade_metrics(trades: List[Dict]) -> Dict[str, Any]:
    """
    Calculate trade-based performance metrics
    
    Args:
        trades: List of trade dictionaries
        
    Returns:
        Dictionary of trade metrics
    """
    if not trades:
        return {}
    
    # Extract profit/loss from trades
    buy_trades = [t for t in trades if t.get('action') == 'buy']
    sell_trades = [t for t in trades if t.get('action') == 'sell']
    
    # Calculate profit/loss for sell trades
    profits = []
    losses = []
    
    for trade in sell_trades:
        if 'value' in trade and 'cost' in trade:
            pnl = trade['value'] - trade['cost']
            if pnl > 0:
                profits.append(pnl)
            else:
                losses.append(abs(pnl))
    
    # Calculate basic metrics
    total_trades = len(trades)
    win_count = len(profits)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0
    
    # Calculate profit metrics
    total_profit = sum(profits) if profits else 0
    total_loss = sum(losses) if losses else 0
    
    # Avoid division by zero
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate averages
    avg_profit = total_profit / win_count if win_count > 0 else 0
    avg_loss = total_loss / loss_count if loss_count > 0 else 0
    
    # Expectancy
    expectancy = (win_rate * avg_profit) - ((1 - win_rate) * avg_loss)
    
    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': win_rate * 100,  # Convert to percentage
        'profit_factor': profit_factor,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'avg_profit': avg_profit,
        'avg_loss': avg_loss,
        'expectancy': expectancy
    }


def resample_equity_curve(equity_curve: pd.DataFrame, rule: str = 'D') -> pd.DataFrame:
    """
    Resample an equity curve to a specified frequency
    
    Args:
        equity_curve: DataFrame with 'timestamp' and 'portfolio_value' columns
        rule: Resampling rule ('D' for daily, 'W' for weekly, etc.)
        
    Returns:
        Resampled equity curve
    """
    if equity_curve.empty or 'timestamp' not in equity_curve.columns:
        return equity_curve
    
    # Set timestamp as index if it's not already
    if equity_curve.index.name != 'timestamp':
        equity_curve = equity_curve.set_index('timestamp')
    
    # Resample
    resampled = equity_curve['portfolio_value'].resample(rule).last()
    
    # Reset index to get timestamp as a column again
    return resampled.reset_index()


def align_timestamps(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Align multiple dataframes to a common set of timestamps using forward fill
    
    Args:
        dataframes: Dictionary mapping symbol to DataFrame
        
    Returns:
        Dictionary of aligned dataframes
    """
    if not dataframes:
        return {}
    
    # Get all unique timestamps
    all_timestamps = set()
    for df in dataframes.values():
        if isinstance(df.index, pd.DatetimeIndex):
            all_timestamps.update(df.index)
        elif 'timestamp' in df.columns:
            all_timestamps.update(df['timestamp'])
    
    all_timestamps = sorted(all_timestamps)
    
    aligned_dfs = {}
    for symbol, df in dataframes.items():
        # Determine how timestamps are stored (index or column)
        if isinstance(df.index, pd.DatetimeIndex):
            # Set a unique index to avoid reindexing issues
            temp_df = df.copy()
            temp_df = temp_df.reindex(all_timestamps, method='ffill')
            aligned_dfs[symbol] = temp_df
        elif 'timestamp' in df.columns:
            # For dataframes with timestamp column
            temp_df = df.set_index('timestamp')
            temp_df = temp_df.reindex(all_timestamps, method='ffill')
            temp_df = temp_df.reset_index()
            aligned_dfs[symbol] = temp_df
    
    return aligned_dfs


def calculate_alpha_beta(equity_curve: pd.DataFrame, benchmark_prices: pd.DataFrame, 
                         risk_free_rate: float = 0.02) -> Tuple[float, float]:
    """
    Calculate Alpha and Beta of a strategy versus a benchmark
    
    Args:
        equity_curve: DataFrame with 'timestamp' and 'portfolio_value' columns
        benchmark_prices: DataFrame with 'timestamp' and 'price' columns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Tuple containing (alpha, beta)
    """
    # Check inputs
    if (equity_curve.empty or benchmark_prices.empty or 
        'portfolio_value' not in equity_curve.columns or 
        'price' not in benchmark_prices.columns):
        return np.nan, np.nan
    
    # Align timestamps
    equity = equity_curve.set_index('timestamp')['portfolio_value']
    benchmark = benchmark_prices.set_index('timestamp')['price']
    
    # Get common dates and align series
    common_dates = equity.index.intersection(benchmark.index)
    if len(common_dates) < 10:  # Need enough data points
        return np.nan, np.nan
        
    equity = equity.loc[common_dates]
    benchmark = benchmark.loc[common_dates]
    
    # Calculate returns
    equity_returns = equity.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    
    # Get common dates again after calculating returns
    common_dates = equity_returns.index.intersection(benchmark_returns.index)
    equity_returns = equity_returns.loc[common_dates]
    benchmark_returns = benchmark_returns.loc[common_dates]
    
    if len(equity_returns) < 10:
        return np.nan, np.nan
    
    # Calculate Beta (slope of regression line)
    covariance = np.cov(equity_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    beta = covariance / benchmark_variance if benchmark_variance > 0 else np.nan
    
    # Calculate Alpha (intercept of regression line)
    # Alpha is annualized excess return
    period = estimate_period(common_dates)
    periods = periods_in_year(period)
    
    daily_rf = (1 + risk_free_rate) ** (1/periods) - 1
    alpha = np.mean(equity_returns - daily_rf - beta * (benchmark_returns - daily_rf))
    alpha_annualized = alpha * periods
    
    return alpha_annualized, beta