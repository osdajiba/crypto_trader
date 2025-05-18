#!/usr/bin/env python3
# src/backtest/engine/market_replay.py

"""
Market Replay backtest engine implementation.
Simulates trading with sequential processing of market data bar by bar.
"""

import asyncio
from collections import deque
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time
from tqdm import tqdm

from src.common.abstract_factory import register_factory_class
from src.backtest.base import BaseBacktestEngine, BacktestEngineError


class MarketReplayEngineError(BacktestEngineError):
    """Error specific to market replay engine operations"""
    pass


@register_factory_class('backtest_engine_factory', "market_replay", 
    description="Market Replay Backtest Engine for sequential bar-by-bar data processing",
    features=["sequential", "realistic_execution", "detailed_simulation"],
    category="backtest")
class MarketReplayEngine(BaseBacktestEngine):
    """
    Market Replay Backtest Engine
    
    Processes data sequentially bar by bar, generating signals and executing trades
    in real-time sequence. This engine provides a more accurate simulation of 
    live trading conditions compared to vectorized approaches.
    """
    
    def __init__(self, config, params=None):
        """
        Initialize market replay backtest engine
        
        Args:
            config: Configuration manager
            params: Engine parameters
        """
        super().__init__(config, params)
        
        # Market replay specific settings from params with defaults
        self.replay_speed = self.params.get('replay_speed', 0)  # 0 means as fast as possible
        self.initial_capital = self.params.get(
            'initial_capital', 
            self.config.get("trading", "capital", "initial", default=100000)
        )
        self.commission_rate = self.params.get(
            'commission_rate', 
            self.config.get("trading", "fees", "commission_taker", default=0.005)
        )
        self.slippage = self.params.get(
            'slippage', 
            self.config.get("trading", "fees", "slippage", default=0.0001)
        )
        
        # Execution and tracking
        self.trades = []
        self.portfolio_history = []
        self.baseline_history = []
        
        # Track the current timestamp for each symbol
        self.symbol_timestamps = {}
        
        # Window size for tracking required history (will be determined from strategy)
        self.window_sizes = {}
        
        self.logger.info(f"Market Replay Engine initialized with capital={self.initial_capital}")

    async def initialize(self) -> None:
        """
        Initialize the market replay engine
        """
        await super().initialize()
        
        # If strategy is set, determine required window sizes
        if self.strategy:
            self.determine_window_sizes()
            
    def determine_window_sizes(self) -> None:
        """
        Determine required window sizes for each factor in the strategy
        """
        # Reset window sizes
        self.window_sizes = {}
        
        # Default window size from params
        default_window = self.params.get('window_size', 100)
        
        # Try to get window sizes from strategy factors
        if hasattr(self.strategy, '_factor_registry'):
            for factor_name, factor_info in self.strategy._factor_registry.items():
                window = factor_info.get('window', default_window)
                self.window_sizes[factor_name] = window
                
            # Find the largest window size
            if self.window_sizes:
                max_window = max(self.window_sizes.values())
                # Add extra padding
                self.required_window_size = max_window + 1
                self.logger.info(f"Set required window size to {self.required_window_size} based on strategy factors")
                return
                
        # If no factor registry or no window sizes, use default
        self.required_window_size = default_window
        self.logger.info(f"Using default window size of {self.required_window_size}")

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run market replay backtest, processing data bar by bar in sequence
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if not self._is_initialized:
            await self.initialize()
        
        # Check for portfolio manager
        if not self.portfolio:
            raise BacktestEngineError("Portfolio manager not set, cannot run backtest")
            
        # Check for strategy
        if not self.strategy:
            raise BacktestEngineError("Strategy not set, cannot run backtest")
        
        start_time = time.time()
        self._is_running = True
        
        # Reset state for clean run
        self.trades = []
        self.portfolio_history = []
        self.baseline_history = []
        self.symbol_timestamps = {}
        
        try:
            # Prepare initial data buffers
            sufficient_data = await self.prepare_data(data)
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                return {'error': 'Insufficient data for backtest'}
            
            # Get common timeline for all symbols
            timeline = self._get_common_timeline(data)
            self.logger.info(f"Running market replay with {len(timeline)} time points")
            
            # Get main symbol (first in list if multiple) for baseline tracking
            symbols = list(data.keys())
            main_symbol = symbols[0] if symbols else None
            self.logger.info(f"Using {main_symbol} as the baseline asset")
            
            # Set up progress display if enabled
            show_progress = self.params.get('show_progress', True)
            progress_bar = tqdm(total=len(timeline), desc="Market Replay") if show_progress else None
            
            # Make sure we have assets in the portfolio for each symbol
            await self._ensure_assets_in_portfolio(symbols)
            
            # Track initial portfolio value
            initial_value = self.portfolio.get_total_value()
            self.portfolio_history.append({
                'timestamp': timeline[0] if timeline else pd.Timestamp.now(),
                'portfolio_value': initial_value
            })
            
            # Process each time point sequentially
            self.logger.info("Beginning market replay simulation")
            for i, timestamp in enumerate(timeline):
                if not self._is_running:
                    self.logger.info("Backtest stopped early")
                    break
                
                # Get data for current timestamp across all symbols
                current_data = self._get_data_at_timestamp(data, timestamp)
                
                # Update portfolio with current market data
                if current_data:
                    await self.portfolio.update_market_data(current_data)
                
                # Track baseline (underlying asset) price
                if main_symbol in current_data and not current_data[main_symbol].empty:
                    baseline_price = current_data[main_symbol]['close'].iloc[0]
                    self.baseline_history.append({
                        'timestamp': timestamp,
                        'price': baseline_price
                    })
                
                # Process each symbol's data point to generate signals immediately
                all_signals_for_timestamp = []
                
                for symbol, data_point in current_data.items():
                    # Process data point to generate signals
                    signals = await self.process_data_point(data_point, symbol)
                    
                    # If signals were generated, execute them immediately
                    if not signals.empty:
                        # Execute the signals
                        executed_trades = await self.portfolio.process_signals(signals, data_point)
                        if executed_trades:
                            self.trades.extend(executed_trades)
                            self.logger.debug(f"Executed {len(executed_trades)} trades for {symbol} at {timestamp}")
                            # Update performance analyzer if available
                            if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                                for trade in executed_trades:
                                    self.performance_analyzer.record_trade(trade)
                    
                # Update portfolio value after processing all symbols at this timestamp
                portfolio_value = self.portfolio.get_total_value()
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value
                })

                # Update progress bar
                if progress_bar:
                    progress_bar.update(1)
                
                # Simulate real-time delay if specified
                if self.replay_speed > 0 and i % 10 == 0:
                    await asyncio.sleep(self.replay_speed / 1000)  # Convert to seconds
                
                try:
                    if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                        self.performance_analyzer.update_equity(timestamp, portfolio_value)
                except Exception as analyzer_error:
                    self.logger.warning(f"Error updating performance metrics: {analyzer_error}")
                    
            # Clean up progress bar
            if progress_bar:
                progress_bar.close()
            
            # Prepare results
            execution_time = time.time() - start_time
            
            # Create equity curve DataFrame
            equity_curve_df = pd.DataFrame(self.portfolio_history)
            
            # Create baseline prices DataFrame
            baseline_df = pd.DataFrame(self.baseline_history) if self.baseline_history else pd.DataFrame()
            
            # Calculate performance metrics
            final_value = self.portfolio.get_total_value()
            
            # Calculate baseline performance
            baseline_return_pct = None
            if not baseline_df.empty:
                initial_price = baseline_df['price'].iloc[0]
                final_price = baseline_df['price'].iloc[-1]
                baseline_return = (final_price - initial_price) / initial_price
                baseline_return_pct = baseline_return * 100
            
            # Prepare result object
            results = {
                'status': 'completed',
                'trades': self.trades,
                'equity_curve': equity_curve_df,
                'baseline_prices': baseline_df,
                'metrics': {
                    'processing_time': execution_time,
                    'initial_capital': self.initial_capital,
                    'final_value': final_value,
                    'total_return': final_value - self.initial_capital,
                    'total_return_pct': ((final_value / self.initial_capital) - 1) * 100,
                    'total_trades': len(self.trades),
                    'symbols_traded': len(set(trade.get('symbol', '') for trade in self.trades)) if self.trades else 0
                },
                'symbols': list(data.keys()),
                'strategy': self.strategy.__class__.__name__,
                'engine': self.__class__.__name__,
                'start_timestamp': timeline[0] if timeline else None,
                'end_timestamp': timeline[-1] if timeline else None
            }
            
            # Add baseline metrics if available
            if baseline_return_pct is not None:
                results['metrics']['baseline_initial_price'] = initial_price
                results['metrics']['baseline_final_price'] = final_price
                results['metrics']['baseline_return_pct'] = baseline_return_pct
                
                # Calculate alpha (strategy outperformance)
                strategy_return_pct = results['metrics']['total_return_pct']
                results['metrics']['alpha'] = strategy_return_pct - baseline_return_pct
            
            self.logger.info(f"Market replay completed in {execution_time:.2f}s with "
                           f"{len(self.trades)} trades and "
                           f"{results['metrics']['total_return_pct']:.2f}% return")
            
            return results
        
        except Exception as e:
            self.logger.error(f"Error during market replay: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            # More graceful failure - return partial results if available
            if self.portfolio_history:
                # Try to salvage what data we have
                self.logger.warning("Returning partial results despite error")
                return {
                    'status': 'partial',
                    'error': str(e),
                    'trades': self.trades,
                    'equity_curve': pd.DataFrame(self.portfolio_history),
                    'baseline_prices': pd.DataFrame(self.baseline_history) if self.baseline_history else pd.DataFrame(),
                    'metrics': {
                        'error_occurred': True,
                        'initial_capital': self.initial_capital,
                        'processed_bars': i if 'i' in locals() else 0,
                        'total_bars': len(timeline)
                    }
                }
            else:
                raise BacktestEngineError(f"Market replay failed: {str(e)}")
        finally:
            self._is_running = False

    async def _ensure_assets_in_portfolio(self, symbols: List[str]) -> None:
        """
        Make sure all symbols are present as assets in the portfolio
        
        Args:
            symbols: List of symbol names
        """
        if not self.portfolio:
            return
            
        # Get current assets in portfolio
        portfolio_assets = self.portfolio.list_assets()
        
        # For each symbol, ensure it's in the portfolio
        for symbol in symbols:
            if symbol not in portfolio_assets:
                self.logger.info(f"Adding {symbol} to portfolio for backtesting")
                
                # Create asset params
                asset_params = {
                    'name': symbol,
                    'symbol': symbol,
                    'type': 'spot',
                    'tradable': True,
                    'quantity': 0.0,
                    'price': 100.0  # Default price, will be updated with market data
                }
                
                try:
                    # Create asset through the portfolio
                    if hasattr(self.portfolio, 'asset_factory'):
                        asset = await self.portfolio.asset_factory.create_asset(asset_params)
                        await self.portfolio.add_asset(asset)
                except Exception as e:
                    self.logger.error(f"Error adding {symbol} to portfolio: {e}")


    def _get_common_timeline(self, data: Dict[str, pd.DataFrame]) -> List:
        """
        Extract a common timeline from all data sources
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            List: Sorted unique timestamps
        """
        # Extract all timestamps
        all_timestamps = []
        for df in data.values():
            if isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.extend(df.index.tolist())
            elif 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
            elif 'timestamp' in df.columns:
                all_timestamps.extend(df['timestamp'].tolist())
        
        # Sort and remove duplicates
        return sorted(set(all_timestamps))
    
    def _get_data_at_timestamp(self, data: Dict[str, pd.DataFrame], timestamp) -> Dict[str, pd.DataFrame]:
        """
        Get data for all symbols at a specific timestamp
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            timestamp: Timestamp to fetch data for
            
        Returns:
            Dict[str, pd.DataFrame]: Symbol -> data point at timestamp
        """
        result = {}
        
        for symbol, df in data.items():
            # Find data at timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                if timestamp in df.index:
                    result[symbol] = df.loc[[timestamp]]
            elif 'datetime' in df.columns:
                mask = df['datetime'] == timestamp
                if mask.any():
                    result[symbol] = df[mask]
            elif 'timestamp' in df.columns:
                mask = df['timestamp'] == timestamp
                if mask.any():
                    result[symbol] = df[mask]
            
            # Update the current timestamp for this symbol
            if symbol in result:
                self.symbol_timestamps[symbol] = timestamp
        
        return result

    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol, immediately generating signals
        
        This implementation maintains a rolling window of data specifically sized
        for the strategy's factor calculation needs, then calls the strategy to
        generate signals based on the accumulated data.
        
        Args:
            data_point: DataFrame containing a single data point
            symbol: Symbol being processed
            
        Returns:
            pd.DataFrame: Generated signals, if any
        """
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame()
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
            self.has_sufficient_history[symbol] = False
        
        # Skip empty data points
        if data_point.empty:
            return pd.DataFrame()
        
        # Add to queue if using window
        if self.required_window_size > 0:
            self.data_queues[symbol].append(data_point)
            
            # If we don't have enough data yet, no signal
            if len(self.data_queues[symbol]) < self.required_window_size:
                return pd.DataFrame()  # Empty DataFrame = no signal
                
            # Construct window from queue
            self.data_buffers[symbol] = pd.concat(list(self.data_queues[symbol]))
        else:
            # No windowing, just use the current data point
            self.data_buffers[symbol] = data_point
        
        # Now we have sufficient history
        self.has_sufficient_history[symbol] = True
        
        # Process with strategy
        if self.strategy:
            signals = await self.strategy.process_data(self.data_buffers[symbol], symbol)
            
            # Ensure signals have timestamp and symbol
            if not signals.empty:
                # Get the timestamp from this data point
                if 'timestamp' in data_point.columns:
                    timestamp = data_point['timestamp'].iloc[0]
                elif 'datetime' in data_point.columns:
                    timestamp = data_point['datetime'].iloc[0]
                else:
                    timestamp = pd.Timestamp.now()
                
                # Add timestamp if missing
                if 'timestamp' not in signals.columns:
                    signals['timestamp'] = timestamp
                
                # Add symbol if missing
                if 'symbol' not in signals.columns:
                    signals['symbol'] = symbol
                
                self.logger.debug(f"Generated {len(signals)} signals for {symbol} at {timestamp}")
            
            return signals
        
        # No strategy, no signals
        return pd.DataFrame()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the market replay backtest
        
        Returns:
            Dict[str, Any]: Performance metrics dictionary
        """
        metrics = super().get_metrics()
        
        # Get the latest portfolio value
        if self.portfolio:
            current_value = self.portfolio.get_total_value()
            
            # Calculate additional metrics
            metrics.update({
                'current_value': current_value,
                'return_pct': ((current_value / self.initial_capital) - 1) * 100,
                'trade_count': len(self.trades)
            })
            
            # Calculate drawdown if we have portfolio history
            if self.portfolio_history:
                equity_values = [point['portfolio_value'] for point in self.portfolio_history]
                peak_equity = max(equity_values)
                drawdown = (peak_equity - current_value) / peak_equity if peak_equity > 0 else 0
                metrics['max_drawdown_pct'] = drawdown * 100
                
                # Win rate if we have trades
                if self.trades:
                    profitable_trades = sum(1 for trade in self.trades if trade.get('realized_pnl', 0) > 0)
                    total_trades = len(self.trades)
                    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
                    metrics['win_rate_pct'] = win_rate
        
        return metrics
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Reset state
        self.trades = []
        self.portfolio_history = []
        self.baseline_history = []
        self.symbol_timestamps = {}
        
        # Call parent shutdown
        await super().shutdown()