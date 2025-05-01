#!/usr/bin/env python3
# src/strategy/performance/trading.py

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List, Tuple
import asyncio

from src.common.config import ConfigManager
from src.common.log_manager import LogManager
from src.strategy.performance.base import BasePerformanceAnalyzer
from src.strategy.performance.metrics import PerformanceMetrics
from src.common.abstract_factory import register_factory_class


@register_factory_class('performance_factory', 'trading', 
                       description="Performance analyzer for live and paper trading",
                       features=["real_time_analysis", "equity_tracking", "drawdown_calculation", "trade_tracking"],
                       category="production")
class TradingPerformanceAnalyzer(BasePerformanceAnalyzer):
    """Performance analyzer implementation for live and paper trading scenarios"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize trading performance analyzer"""
        super().__init__(config, params)
        
        # Trading specific settings
        self.auto_save_interval = self.config.get("performance", "trading", "auto_save_interval", default=3600)  # Default 1 hour
        self.alert_thresholds = self.config.get("performance", "trading", "alert_thresholds", default={
            "drawdown": 0.1,  # 10% drawdown alert
            "losing_trades": 5,  # Alert after 5 consecutive losing trades
            "daily_loss": 0.05  # 5% daily loss alert
        })
        
        # Status tracking
        self.last_save_time = None
        self.last_alert_check = None
        self.consecutive_losing_trades = 0
        self.daily_tracking = {}
        
        # Monitoring task
        self.monitoring_task = None
    
    async def _initialize_analyzer(self) -> None:
        """Initialize analyzer-specific components"""
        # Initialize real-time monitoring
        self.last_save_time = datetime.now()
        self.last_alert_check = datetime.now()
        
        # Reset tracking
        self.daily_tracking = {
            'date': datetime.now().date(),
            'starting_equity': self.initial_capital,
            'current_equity': self.initial_capital,
            'high_equity': self.initial_capital,
            'low_equity': self.initial_capital,
            'trades': []
        }
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.logger.info("Real-time performance monitoring started")
        
        # Initialize live metrics storage
        self.intraday_metrics = {
            'equity_updates': [],
            'trade_metrics': {}
        }
    
    async def _monitoring_loop(self) -> None:
        """Background task for periodic monitoring and alerts"""
        try:
            while True:
                current_time = datetime.now()
                
                # Check if a new day has started
                if current_time.date() > self.daily_tracking['date']:
                    await self._process_day_end()
                    
                # Periodic saves
                if (self.last_save_time is None or 
                    (current_time - self.last_save_time).total_seconds() >= self.auto_save_interval):
                    self._auto_save_data()
                    self.last_save_time = current_time
                
                # Check for alerts
                if (self.last_alert_check is None or 
                    (current_time - self.last_alert_check).total_seconds() >= 60):  # Check alerts every minute
                    await self._check_performance_alerts()
                    self.last_alert_check = current_time
                
                # Sleep for a short period to reduce CPU usage
                await asyncio.sleep(5)
                
        except asyncio.CancelledError:
            self.logger.info("Performance monitoring task cancelled")
            
        except Exception as e:
            self.logger.error(f"Error in performance monitoring: {e}")
    
    async def _process_day_end(self) -> None:
        """Process end of trading day and rotate daily tracking"""
        # Log daily summary
        previous_date = self.daily_tracking['date']
        starting_equity = self.daily_tracking['starting_equity']
        ending_equity = self.daily_tracking.get('current_equity', starting_equity)
        daily_return = (ending_equity / starting_equity) - 1 if starting_equity > 0 else 0
        
        self.logger.info(f"Daily performance summary for {previous_date}:")
        self.logger.info(f"  Starting equity: ${starting_equity:,.2f}")
        self.logger.info(f"  Ending equity: ${ending_equity:,.2f}")
        self.logger.info(f"  Daily return: {daily_return:.2%}")
        self.logger.info(f"  High: ${self.daily_tracking.get('high_equity', ending_equity):,.2f}")
        self.logger.info(f"  Low: ${self.daily_tracking.get('low_equity', ending_equity):,.2f}")
        self.logger.info(f"  Trades: {len(self.daily_tracking.get('trades', []))}")
        
        # Add to daily history
        daily_record = self.daily_tracking.copy()
        daily_record['return'] = daily_return
        
        # Initialize new day tracking
        current_date = datetime.now().date()
        self.daily_tracking = {
            'date': current_date,
            'starting_equity': ending_equity,
            'current_equity': ending_equity,
            'high_equity': ending_equity,
            'low_equity': ending_equity,
            'trades': []
        }
        
        self.logger.info(f"Started new daily tracking for {current_date}")
        
        # Save daily data
        daily_save_enabled = self.config.get("performance", "trading", "save_daily_data", default=True)
        if daily_save_enabled:
            self._save_daily_data(daily_record, previous_date)
    
    def _save_daily_data(self, daily_record: Dict[str, Any], date: datetime.date) -> None:
        """Save daily performance data to file"""
        try:
            # Create daily data directory if needed
            daily_dir = os.path.join(self.storage_path, "daily")
            os.makedirs(daily_dir, exist_ok=True)
            
            # Create filename with date
            date_str = date.strftime('%Y-%m-%d')
            filepath = os.path.join(daily_dir, f"trading_day_{date_str}.json")
            
            # Save data
            with open(filepath, 'w') as f:
                json.dump(daily_record, f, indent=2, default=str)
                
            self.logger.info(f"Daily performance data saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving daily performance data: {e}")
    
    def _auto_save_data(self) -> None:
        """Automatically save current performance data"""
        try:
            # Only save if we have equity data
            if 'system' in self.equity_history and not self.equity_history['system'].empty:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = os.path.join(self.storage_path, f"trading_performance_{timestamp}.json")
                
                self.save_to_file('system', filepath)
                self.logger.info(f"Performance data auto-saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error in auto-save: {e}")
    
    async def _check_performance_alerts(self) -> None:
        """Check for performance alerts that may require attention"""
        try:
            # Skip if no data available
            if 'system' not in self.equity_history or self.equity_history['system'].empty:
                return
                
            current_equity = self.equity_history['system'].iloc[-1]
            
            # Check drawdown alert
            drawdown_threshold = self.alert_thresholds.get('drawdown', 0.1)
            max_drawdown = 0.0
            
            if 'system' in self.drawdown_history and not self.drawdown_history['system'].empty:
                max_drawdown = abs(self.drawdown_history['system'].min())
                
                if max_drawdown >= drawdown_threshold:
                    self.logger.warning(f"ALERT: Maximum drawdown ({max_drawdown:.2%}) has exceeded threshold ({drawdown_threshold:.2%})")
            
            # Check daily loss alert
            daily_loss_threshold = self.alert_thresholds.get('daily_loss', 0.05)
            starting_equity = self.daily_tracking.get('starting_equity', self.initial_capital)
            
            if starting_equity > 0:
                daily_return = (current_equity / starting_equity) - 1
                
                if daily_return <= -daily_loss_threshold:
                    self.logger.warning(f"ALERT: Daily loss ({daily_return:.2%}) has exceeded threshold ({-daily_loss_threshold:.2%})")
            
            # Update daily tracking
            self.daily_tracking['current_equity'] = current_equity
            self.daily_tracking['high_equity'] = max(self.daily_tracking.get('high_equity', 0), current_equity)
            self.daily_tracking['low_equity'] = min(self.daily_tracking.get('low_equity', float('inf')), current_equity)
            
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    def record_trade(self, strategy_id: str, trade_data: Dict[str, Any]) -> None:
        """Record a trade for performance analysis, with additional tracking for live trading
        
        Args:
            strategy_id: Strategy identifier
            trade_data: Trade information
        """
        # Call base method
        super().record_trade(strategy_id, trade_data)
        
        # Track consecutive losing trades
        is_profitable = trade_data.get('profit_pct', 0) > 0
        
        if is_profitable:
            self.consecutive_losing_trades = 0
        else:
            self.consecutive_losing_trades += 1
            
            # Check for consecutive losses alert
            losing_trades_threshold = self.alert_thresholds.get('losing_trades', 5)
            if self.consecutive_losing_trades >= losing_trades_threshold:
                self.logger.warning(f"ALERT: {self.consecutive_losing_trades} consecutive losing trades (threshold: {losing_trades_threshold})")
        
        # Add to daily tracking
        if 'trades' in self.daily_tracking:
            self.daily_tracking['trades'].append(trade_data)
            
        # Update intraday metrics
        self._update_intraday_metrics(trade_data)
    
    def _update_intraday_metrics(self, trade_data: Dict[str, Any]) -> None:
        """Update intraday trading metrics"""
        trade_metrics = self.intraday_metrics['trade_metrics']
        
        # Initialize symbol if needed
        symbol = trade_data.get('symbol', 'unknown')
        if symbol not in trade_metrics:
            trade_metrics[symbol] = {
                'trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0
            }
        
        # Update statistics
        profit = trade_data.get('profit', 0)
        trade_metrics[symbol]['trades'] += 1
        
        if profit > 0:
            trade_metrics[symbol]['winning_trades'] += 1
            trade_metrics[symbol]['total_profit'] += profit
        else:
            trade_metrics[symbol]['losing_trades'] += 1
            trade_metrics[symbol]['total_loss'] += abs(profit)
    
    def update_equity(self, strategy_id: str, timestamp: Union[str, datetime], value: float) -> None:
        """Update equity with additional tracking for trading
        
        Args:
            strategy_id: Strategy identifier
            timestamp: Timestamp for the equity point
            value: Equity value
        """
        # Call base method
        super().update_equity(strategy_id, timestamp, value)
        
        # Track equity updates for intraday metrics
        if strategy_id == 'system':
            timestamp_obj = pd.Timestamp(timestamp) if isinstance(timestamp, str) else timestamp
            
            self.intraday_metrics['equity_updates'].append({
                'timestamp': timestamp_obj,
                'equity': value
            })
            
            # Limit number of equity updates tracked
            max_updates = 1000
            if len(self.intraday_metrics['equity_updates']) > max_updates:
                self.intraday_metrics['equity_updates'] = self.intraday_metrics['equity_updates'][-max_updates:]
    
    def calculate_metrics(self, strategy_id: str) -> Dict[str, Any]:
        """Calculate performance metrics for a strategy
        
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
        
        # Calculate drawdown
        max_drawdown, drawdown_start, drawdown_end = PerformanceMetrics.calculate_max_drawdown(equity)
        
        # Calculate trade metrics
        trades = self.trade_history[strategy_id]
        trade_metrics = {}
        
        if not trades.empty and 'profit_pct' in trades.columns:
            trade_metrics = PerformanceMetrics.calculate_trade_metrics(trades)
        
        # Calculate intraday volatility
        intraday_volatility = 0.0
        if len(self.intraday_metrics['equity_updates']) > 1:
            equity_values = [update['equity'] for update in self.intraday_metrics['equity_updates']]
            equity_series = pd.Series(equity_values)
            equity_returns = equity_series.pct_change().dropna()
            intraday_volatility = equity_returns.std() * np.sqrt(252 * 6.5)  # Assuming 6.5 trading hours
        
        # Assemble metrics
        metrics = {
            'current_equity': equity.iloc[-1],
            'initial_equity': equity.iloc[0],
            'total_return': (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) >= 2 else 0.0,
            'max_drawdown': max_drawdown,
            'current_drawdown': self.drawdown_history[strategy_id].iloc[-1] if not self.drawdown_history[strategy_id].empty else 0.0,
            'sharpe_ratio': PerformanceMetrics.calculate_sharpe_ratio(daily_returns, self.risk_free_rate / 252) if not daily_returns.empty else 0.0,
            'volatility': PerformanceMetrics.calculate_volatility(daily_returns) if not daily_returns.empty else 0.0,
            'intraday_volatility': intraday_volatility,
            'monitoring_since': equity.index[0].strftime('%Y-%m-%d %H:%M:%S') if not equity.empty else None,
            'last_update': equity.index[-1].strftime('%Y-%m-%d %H:%M:%S') if not equity.empty else None,
            **trade_metrics
        }
        
        # Add today's data
        today = datetime.now().date()
        metrics['today'] = {
            'starting_equity': self.daily_tracking.get('starting_equity', 0.0),
            'current_equity': self.daily_tracking.get('current_equity', 0.0),
            'high': self.daily_tracking.get('high_equity', 0.0),
            'low': self.daily_tracking.get('low_equity', 0.0),
            'trades': len(self.daily_tracking.get('trades', [])),
            'return': ((self.daily_tracking.get('current_equity', 0.0) / 
                       self.daily_tracking.get('starting_equity', 1.0)) - 1)
                       if self.daily_tracking.get('starting_equity', 0) > 0 else 0.0
        }
        
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
            filepath = os.path.join(self.storage_path, f"{strategy_id}_trading_performance_{timestamp}.json")
            
        # Calculate metrics
        metrics = self.calculate_metrics(strategy_id)
        
        # Prepare data for serialization
        data = {
            'strategy_id': strategy_id,
            'metrics': metrics,
            'equity': self.equity_history[strategy_id].reset_index().to_dict(orient='records') if not self.equity_history[strategy_id].empty else [],
            'trades': self.trade_history[strategy_id].to_dict(orient='records') if not self.trade_history[strategy_id].empty else [],
            'drawdown': self.drawdown_history[strategy_id].reset_index().to_dict(orient='records') if not self.drawdown_history[strategy_id].empty else [],
            'intraday_metrics': self.intraday_metrics,
            'daily_tracking': self.daily_tracking,
            'alert_thresholds': self.alert_thresholds,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            
        self.logger.info(f"Trading performance data for {strategy_id} saved to {filepath}")
        return filepath
    
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
            
            timestamp_col = None
            for col in ['timestamp', 'date', 'datetime']:
                if col in drawdown_df.columns:
                    timestamp_col = col
                    break
                    
            if timestamp_col:
                drawdown_df[timestamp_col] = pd.to_datetime(drawdown_df[timestamp_col])
                value_col = [col for col in drawdown_df.columns if col != timestamp_col][0]
                drawdown_series = pd.Series(drawdown_df[value_col].values, index=drawdown_df[timestamp_col])
                self.drawdown_history[strategy_id] = drawdown_series
            
        # Load metrics
        if 'metrics' in data:
            self.strategy_metrics[strategy_id] = data['metrics']
            
        # Load intraday metrics
        if 'intraday_metrics' in data:
            self.intraday_metrics = data['intraday_metrics']
            
            # Convert equity update timestamps
            if 'equity_updates' in self.intraday_metrics:
                for update in self.intraday_metrics['equity_updates']:
                    if 'timestamp' in update:
                        update['timestamp'] = pd.Timestamp(update['timestamp'])
        
        # Load daily tracking
        if 'daily_tracking' in data:
            self.daily_tracking = data['daily_tracking']
            
            # Convert date string to date object
            if 'date' in self.daily_tracking and isinstance(self.daily_tracking['date'], str):
                self.daily_tracking['date'] = datetime.strptime(
                    self.daily_tracking['date'], '%Y-%m-%d').date()
        
        # Load alert thresholds
        if 'alert_thresholds' in data:
            self.alert_thresholds = data['alert_thresholds']
            
        self.logger.info(f"Trading performance data for {strategy_id} loaded from {filepath}")
        return strategy_id
    
    def generate_performance_report(self, strategy_id: str) -> Dict[str, Any]:
        """Generate a comprehensive trading performance report
        
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
                'current_equity': f"${metrics.get('current_equity', 0.0):,.2f}",
                'total_return': f"{metrics.get('total_return', 0.0) * 100:.2f}%",
                'max_drawdown': f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%",
                'current_drawdown': f"{metrics.get('current_drawdown', 0.0) * 100:.2f}%",
                'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0.0):.2f}",
                'volatility': f"{metrics.get('volatility', 0.0) * 100:.2f}%"
            },
            'trade_statistics': {
                'total_trades': metrics.get('total_trades', 0),
                'win_rate': f"{metrics.get('win_rate', 0.0) * 100:.2f}%",
                'profit_factor': f"{metrics.get('profit_factor', 0.0):.2f}",
                'avg_trade': f"{metrics.get('avg_trade', 0.0) * 100:.2f}%",
                'expectancy': f"{metrics.get('expectancy', 0.0) * 100:.2f}%"
            },
            'today': {
                'return': f"{metrics.get('today', {}).get('return', 0.0) * 100:.2f}%",
                'high': f"${metrics.get('today', {}).get('high', 0.0):,.2f}",
                'low': f"${metrics.get('today', {}).get('low', 0.0):,.2f}",
                'trades': metrics.get('today', {}).get('trades', 0)
            },
            'monitoring_since': metrics.get('monitoring_since', 'N/A'),
            'last_update': metrics.get('last_update', 'N/A')
        }
        
        # Add symbol-specific stats
        symbol_stats = {}
        for symbol, stats in self.intraday_metrics.get('trade_metrics', {}).items():
            if stats['trades'] > 0:
                win_rate = stats['winning_trades'] / stats['trades'] if stats['trades'] > 0 else 0
                profit_factor = stats['total_profit'] / stats['total_loss'] if stats['total_loss'] > 0 else float('inf')
                
                symbol_stats[symbol] = {
                    'trades': stats['trades'],
                    'win_rate': f"{win_rate * 100:.2f}%",
                    'profit_factor': f"{profit_factor:.2f}",
                    'net_profit': f"${stats['total_profit'] - stats['total_loss']:,.2f}"
                }
        
        if symbol_stats:
            summary['symbol_performance'] = symbol_stats
        
        # Add alert status
        alerts_active = {
            'drawdown': metrics.get('max_drawdown', 0.0) >= self.alert_thresholds.get('drawdown', 0.1),
            'daily_loss': metrics.get('today', {}).get('return', 0) <= -self.alert_thresholds.get('daily_loss', 0.05),
            'consecutive_losses': self.consecutive_losing_trades >= self.alert_thresholds.get('losing_trades', 5)
        }
        
        summary['alerts'] = {
            'active_alerts': sum(1 for alert in alerts_active.values() if alert),
            'alert_status': alerts_active,
            'thresholds': self.alert_thresholds
        }
        
        return {
            'summary': summary,
            'metrics': metrics,
            'trade_count': len(self.trade_history[strategy_id]),
            'equity_points': len(self.equity_history[strategy_id]),
            'generation_time': datetime.now().isoformat()
        }
    
    async def _shutdown_analyzer(self) -> None:
        """Analyzer-specific shutdown operations"""
        # Cancel monitoring task
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            
        # Save final performance data
        if 'system' in self.equity_history and not self.equity_history['system'].empty:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = os.path.join(self.storage_path, f"final_trading_performance_{timestamp}.json")
                
                self.save_to_file('system', filepath)
                self.logger.info(f"Final trading performance data saved to {filepath}")
                
            except Exception as e:
                self.logger.error(f"Error saving final performance data: {e}")