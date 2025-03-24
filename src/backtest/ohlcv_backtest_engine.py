# src/backtest/ohlcv_backtest_engine.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import traceback

from base_backtest_engine import BaseBacktestEngine, register_backtest_engine
from src.common.config_manager import ConfigManager


@register_backtest_engine('ohlcv', 
    description="OHLCV Backtest Engine for vectorized backtesting",
    category="backtest")
class OHLCVBacktestEngine(BaseBacktestEngine):
    """
    OHLCV Backtest Engine
    
    Performs vectorized backtest operations on OHLCV data,
    processing all data at once for maximum performance.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize OHLCV backtest engine"""
        super().__init__(config, params)
        
        # OHLCV specific settings
        self.initial_capital = self.params.get('initial_capital', 100000)
        self.commission_rate = self.params.get('commission_rate', 0.001)
        self.slippage = self.params.get('slippage', 0.001)
        
        # Factor calculation cache
        self.factor_values = {}  # Symbol -> Factor -> Series

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run market replay backtest, processing data sequentially
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dict: Backtest results with baseline data
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self._is_running = True
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        results = {
            'signals': {},
            'trades': [],
            'equity_curve': pd.DataFrame(),
            'metrics': {}
        }
        
        try:
            # Prepare initial data buffers
            sufficient_data = await self.prepare_data(data)
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                return {'error': 'Insufficient data for backtest'}
            
            # Get common timeline for all symbols
            timeline = self._get_common_timeline(data)
            self.logger.info(f"Running market replay with {len(timeline)} time points")
            
            # Track signals by symbol
            all_signals = {}
            
            # NEW: Track baseline performance (underlying asset prices)
            baseline_history = []
            
            # Get main symbol (first in list if multiple)
            symbols = list(data.keys())
            main_symbol = symbols[0] if symbols else None
            self.logger.info(f"Using {main_symbol} as the baseline asset")
            
            # Process each time point sequentially
            self.logger.info("Beginning market replay simulation")
            for i, timestamp in enumerate(timeline):
                if not self._is_running:
                    self.logger.info("Backtest stopped early")
                    break
                
                # Get data for current timestamp across all symbols
                current_data = self._get_data_at_timestamp(data, timestamp)
                
                # Update portfolio value based on current prices
                portfolio_value = self._calculate_portfolio_value(current_data)
                self.portfolio_history.append({
                    'timestamp': timestamp,
                    'cash': self.cash,
                    'portfolio_value': portfolio_value
                })
                
                # NEW: Track baseline (underlying asset) price
                if main_symbol in current_data and not current_data[main_symbol].empty:
                    baseline_price = current_data[main_symbol]['close'].iloc[0]
                    baseline_history.append({
                        'timestamp': timestamp,
                        'price': baseline_price
                    })
                
                # Process each symbol's data point
                for symbol, data_point in current_data.items():
                    # Process data through strategy
                    signals = await self.process_data_point(data_point, symbol)
                    
                    # Store signals
                    if not signals.empty:
                        if symbol not in all_signals:
                            all_signals[symbol] = []
                        all_signals[symbol].append(signals)
                        
                        # Execute signals
                        new_trades = self._execute_signals(signals, current_data)
                        self.trades.extend(new_trades)
                
                # Log progress
                if i % 100 == 0:
                    self.logger.debug(f"Processed {i}/{len(timeline)} time points")
                
                # Simulate real-time delay if specified
                if self.replay_speed > 0 and i % 10 == 0:
                    await asyncio.sleep(self.replay_speed / 1000)  # Convert to seconds
            
            # Combine signals
            for symbol, signals_list in all_signals.items():
                if signals_list:
                    results['signals'][symbol] = pd.concat(signals_list)
                else:
                    results['signals'][symbol] = pd.DataFrame()
            
            # Create equity curve
            if self.portfolio_history:
                results['equity_curve'] = pd.DataFrame(self.portfolio_history)
            
            # NEW: Add baseline prices to results
            if baseline_history:
                results['baseline_prices'] = pd.DataFrame(baseline_history)
                self.logger.info(f"Stored {len(baseline_history)} baseline price points")
            
            # Store trades
            results['trades'] = self.trades
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.metrics['processing_time'] = execution_time
            
            # Calculate final portfolio value
            final_value = self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else self.initial_capital
            
            # NEW: Calculate baseline performance metrics
            if baseline_history:
                initial_price = baseline_history[0]['price']
                final_price = baseline_history[-1]['price']
                
                baseline_return = (final_price - initial_price) / initial_price
                baseline_return_pct = baseline_return * 100
                
                # Add baseline metrics to results
                results['metrics']['baseline_initial_price'] = initial_price
                results['metrics']['baseline_final_price'] = final_price
                results['metrics']['baseline_return_pct'] = baseline_return_pct
                
                # Calculate alpha (strategy outperformance)
                strategy_return_pct = ((final_value / self.initial_capital) - 1) * 100
                results['metrics']['alpha'] = strategy_return_pct - baseline_return_pct
            
            # Calculate performance metrics
            results['metrics'] = {
                **self.metrics,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': final_value - self.initial_capital,
                'total_return_pct': ((final_value / self.initial_capital) - 1) * 100,
                'total_trades': len(self.trades),
                'symbols_traded': len(set(trade['symbol'] for trade in self.trades)) if self.trades else 0
            }
            
            # Calculate additional metrics if equity curve exists
            if not results['equity_curve'].empty:
                results['metrics'].update(self._calculate_advanced_metrics(results['equity_curve']))
            
            self.logger.info(f"Market replay completed in {execution_time:.2f}s, "
                        f"processed {self.metrics.get('data_points_processed', 0)} data points, "
                        f"generated {len(self.trades)} trades")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during market replay: {str(e)}\n{traceback.format_exc()}")
            return {'error': str(e)}
        finally:
            self._is_running = False
    
    async def _calculate_all_factors(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Calculate all factors for a symbol's data in vectorized manner
        
        Args:
            data: Full historical data
            symbol: Symbol
        """
        if not self.strategy or not hasattr(self.strategy, '_factor_registry'):
            return
            
        # Initialize cache for this symbol
        if symbol not in self.factor_values:
            self.factor_values[symbol] = {}
        
        # Calculate each registered factor
        for factor_name in self.strategy._factor_registry:
            try:
                # Calculate factor values
                factor_values = self._calculate_factor_vectorized(data, factor_name, symbol)
                
                # Store in cache
                if factor_values is not None:
                    self.factor_values[symbol][factor_name] = factor_values
                
            except Exception as e:
                self.logger.error(f"Error calculating factor '{factor_name}' for {symbol}: {str(e)}")
    
    def _calculate_factor_vectorized(self, data: pd.DataFrame, factor_name: str, symbol: str) -> Optional[pd.Series]:
        """
        Calculate factor values for entire dataset
        
        Args:
            data: Full historical data
            factor_name: Factor name
            symbol: Symbol
            
        Returns:
            pd.Series: Factor values or None if error
        """
        # Check if already calculated
        if symbol in self.factor_values and factor_name in self.factor_values[symbol]:
            return self.factor_values[symbol][factor_name]
        
        try:
            # Get factor info
            factor_info = self.strategy._factor_registry.get(factor_name)
            if not factor_info:
                return None
            
            # Calculate dependencies first
            for dep_name in factor_info.get('depends_on', []):
                if dep_name not in self.factor_values.get(symbol, {}):
                    self._calculate_factor_vectorized(data, dep_name, symbol)
            
            # Calculate factor
            if factor_info.get('func') and callable(factor_info['func']):
                # Prepare kwargs with dependencies
                kwargs = {}
                for dep_name in factor_info.get('depends_on', []):
                    if symbol in self.factor_values and dep_name in self.factor_values[symbol]:
                        kwargs[dep_name] = self.factor_values[symbol][dep_name]
                
                # Calculate factor
                return factor_info['func'](data, **kwargs)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating factor '{factor_name}' for {symbol}: {str(e)}")
            return None
    
    async def _generate_signals_vectorized(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate signals for entire dataset at once
        
        Args:
            data: Processed data
            symbol: Symbol
            
        Returns:
            pd.DataFrame: Signals
        """
        try:
            # Check if strategy has vectorized signal generation
            if hasattr(self.strategy, '_generate_signals_vectorized'):
                # Use vectorized method if available
                signals = await self.strategy._generate_signals_vectorized(data, symbol, self.factor_values.get(symbol, {}))
                
                # Add symbol if not present
                if not signals.empty and 'symbol' not in signals.columns:
                    signals['symbol'] = symbol
                
                return signals
            
            # If no vectorized method, use regular strategy logic but apply to the entire dataset
            # Add factor columns to dataset
            enriched_data = data.copy()
            for factor_name, factor_values in self.factor_values.get(symbol, {}).items():
                factor_column = f"factor_{factor_name}"
                enriched_data[factor_column] = factor_values
            
            # Create a method that applies the strategy's signal generation logic to the entire dataset
            signals = await self._apply_strategy_vectorized(enriched_data, symbol)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating vectorized signals for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _apply_strategy_vectorized(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply strategy logic to entire dataset
        
        Args:
            data: Dataset with factor values
            symbol: Symbol
            
        Returns:
            pd.DataFrame: Signals
        """
        # This is a generic implementation that needs to be customized
        # for the specific strategy logic
        
        # For strategies like DualMA, we'd calculate crossovers directly
        if 'factor_short_ma' in data.columns and 'factor_long_ma' in data.columns:
            # Create signal DataFrame
            signals = pd.DataFrame(index=data.index)
            signals['timestamp'] = data.index
            signals['symbol'] = symbol
            
            # Buy signal: short MA crosses above long MA
            crossover_up = (data['factor_short_ma'] > data['factor_long_ma']) & \
                           (data['factor_short_ma'].shift(1) <= data['factor_long_ma'].shift(1))
            
            # Sell signal: short MA crosses below long MA
            crossover_down = (data['factor_short_ma'] < data['factor_long_ma']) & \
                             (data['factor_short_ma'].shift(1) >= data['factor_long_ma'].shift(1))
            
            # Set action based on crossovers
            signals['action'] = None
            signals.loc[crossover_up, 'action'] = 'buy'
            signals.loc[crossover_down, 'action'] = 'sell'
            
            # Add price
            if 'close' in data.columns:
                signals['price'] = data['close']
            
            # Drop rows with no action
            signals = signals.dropna(subset=['action'])
            
            return signals
        
        # For other strategies, we might need to process data point by point
        # This is less efficient but more general
        signals_list = []
        for i in range(len(data)):
            data_point = data.iloc[[i]]
            signal = await self.strategy.process_data(data_point, symbol)
            if not signal.empty:
                signals_list.append(signal)
        
        if signals_list:
            return pd.concat(signals_list)
        return pd.DataFrame()
    
    def _simulate_portfolio(self, all_signals: Dict[str, pd.DataFrame], price_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Simulate portfolio performance based on generated signals
        
        Args:
            all_signals: Dictionary of symbol -> signals DataFrame
            price_data: Dictionary of symbol -> price DataFrame
            
        Returns:
            Tuple[List[Dict], pd.DataFrame]: Trades and equity curve
        """
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},  # symbol -> quantity
            'trades': [],
            'history': []     # portfolio value history
        }
        
        try:
            # Create unified timeline
            all_timestamps = []
            for df in price_data.values():
                if isinstance(df.index, pd.DatetimeIndex):
                    all_timestamps.extend(df.index.tolist())
                elif 'datetime' in df.columns:
                    all_timestamps.extend(df['datetime'].tolist())
            
            timeline = sorted(set(all_timestamps))
            
            # Organize signals by timestamp
            signal_by_time = {}
            for symbol, signals in all_signals.items():
                if signals.empty:
                    continue
                
                for _, signal in signals.iterrows():
                    timestamp = signal.get('timestamp')
                    if timestamp is None and isinstance(signal.name, pd.Timestamp):
                        timestamp = signal.name
                        
                    if timestamp:
                        if timestamp not in signal_by_time:
                            signal_by_time[timestamp] = []
                        signal_by_time[timestamp].append(signal.to_dict())
            
            # Initialize portfolio history
            for timestamp in timeline:
                # Get current prices
                current_prices = {}
                for symbol, df in price_data.items():
                    if isinstance(df.index, pd.DatetimeIndex):
                        if timestamp in df.index:
                            current_prices[symbol] = df.loc[timestamp, 'close']
                    elif 'datetime' in df.columns:
                        mask = df['datetime'] == timestamp
                        if mask.any():
                            current_prices[symbol] = df.loc[mask, 'close'].iloc[0]
                
                # Calculate portfolio value
                portfolio_value = portfolio['cash']
                for symbol, quantity in portfolio['positions'].items():
                    if symbol in current_prices:
                        portfolio_value += quantity * current_prices[symbol]
                
                # Execute any signals at this timestamp
                if timestamp in signal_by_time:
                    for signal in signal_by_time[timestamp]:
                        self._execute_signal(signal, current_prices, portfolio)
                
                # Record portfolio history
                portfolio['history'].append({
                    'timestamp': timestamp,
                    'cash': portfolio['cash'],
                    'portfolio_value': portfolio_value
                })
            
            # Create equity curve DataFrame
            equity_curve = pd.DataFrame(portfolio['history'])
            
            # Add returns calculations
            if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
                equity_curve['return'] = equity_curve['portfolio_value'].pct_change()
                equity_curve['cumulative_return'] = (1 + equity_curve['return'].fillna(0)).cumprod() - 1
            
            return portfolio['trades'], equity_curve
            
        except Exception as e:
            self.logger.error(f"Error simulating portfolio: {str(e)}")
            return [], pd.DataFrame()
    
    def _execute_signal(self, signal: Dict, current_prices: Dict[str, float], portfolio: Dict) -> None:
        """
        Execute a trading signal
        
        Args:
            signal: Signal dictionary
            current_prices: Current prices by symbol
            portfolio: Portfolio state
        """
        symbol = signal['symbol']
        action = signal['action'].lower()
        timestamp = signal.get('timestamp')
        
        # Skip if we don't have price data
        if symbol not in current_prices:
            return
            
        price = current_prices[symbol]
        
        # Apply slippage
        if action == 'buy':
            execution_price = price * (1 + self.slippage)
        else:  # sell
            execution_price = price * (1 - self.slippage)
        
        # Determine quantity
        quantity = signal.get('quantity')
        if quantity is None:
            # Use 1% of portfolio value by default
            portfolio_value = portfolio['cash']
            for sym, qty in portfolio['positions'].items():
                if sym in current_prices:
                    portfolio_value += qty * current_prices[sym]
            
            risk_pct = 0.01  # 1% risk per trade
            quantity = (portfolio_value * risk_pct) / execution_price
        
        # Execute trade
        if action == 'buy':
            # Calculate cost
            cost = quantity * execution_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            
            # Check if we have enough cash
            if total_cost > portfolio['cash']:
                quantity = portfolio['cash'] / (execution_price * (1 + self.commission_rate))
                cost = quantity * execution_price
                commission = cost * self.commission_rate
                total_cost = cost + commission
            
            if quantity > 0:
                # Update cash
                portfolio['cash'] -= total_cost
                
                # Update position
                if symbol not in portfolio['positions']:
                    portfolio['positions'][symbol] = 0
                portfolio['positions'][symbol] += quantity
                
                # Record trade
                portfolio['trades'].append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'buy',
                    'quantity': quantity,
                    'price': execution_price,
                    'commission': commission,
                    'cost': total_cost
                })
        
        elif action == 'sell':
            # Check if we have the position
            if symbol in portfolio['positions'] and portfolio['positions'][symbol] > 0:
                # Limit to current position
                quantity = min(quantity, portfolio['positions'][symbol])
                
                # Calculate value
                value = quantity * execution_price
                commission = value * self.commission_rate
                net_value = value - commission
                
                # Update cash
                portfolio['cash'] += net_value
                
                # Update position
                portfolio['positions'][symbol] -= quantity
                
                # Remove position if zero
                if portfolio['positions'][symbol] <= 0:
                    del portfolio['positions'][symbol]
                
                # Record trade
                portfolio['trades'].append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': execution_price,
                    'commission': commission,
                    'value': net_value
                })
    
    def _calculate_performance_metrics(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate performance metrics
        
        Args:
            equity_curve: Portfolio equity curve
            trades: List of trades
            
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            if not equity_curve.empty and 'portfolio_value' in equity_curve.columns:
                initial_value = equity_curve['portfolio_value'].iloc[0]
                final_value = equity_curve['portfolio_value'].iloc[-1]
                
                metrics['initial_capital'] = initial_value
                metrics['final_value'] = final_value
                metrics['total_return'] = final_value - initial_value
                metrics['total_return_pct'] = (final_value / initial_value - 1) * 100
                
                # Calculate drawdown
                equity_curve['peak'] = equity_curve['portfolio_value'].cummax()
                equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['peak']) / equity_curve['peak']
                metrics['max_drawdown_pct'] = equity_curve['drawdown'].min() * 100 if not equity_curve['drawdown'].empty else 0
                
                # Calculate Sharpe ratio (assuming 252 trading days per year)
                if 'return' in equity_curve.columns and len(equity_curve) > 1:
                    daily_returns = equity_curve['return'].fillna(0)
                    avg_return = daily_returns.mean()
                    std_return = daily_returns.std()
                    if std_return > 0:
                        metrics['sharpe_ratio'] = (avg_return / std_return) * (252 ** 0.5)
                    else:
                        metrics['sharpe_ratio'] = 0
            
            # Trade metrics
            metrics['total_trades'] = len(trades)
            if trades:
                # Calculate win rate
                buy_trades = {t['timestamp'].strftime('%Y-%m-%d %H:%M:%S'): t for t in trades if t['action'] == 'buy'}
                sell_trades = {t['timestamp'].strftime('%Y-%m-%d %H:%M:%S'): t for t in trades if t['action'] == 'sell'}
                
                # This is a simplified calculation - it doesn't account for matching buys/sells
                if buy_trades and sell_trades:
                    # Calculate profit/loss per trade
                    profitable_trades = 0
                    total_matched = 0
                    
                    for timestamp, sell in sell_trades.items():
                        # Find closest buy
                        if timestamp in buy_trades:
                            buy = buy_trades[timestamp]
                            if sell['price'] > buy['price']:
                                profitable_trades += 1
                            total_matched += 1
                    
                    if total_matched > 0:
                        metrics['win_rate'] = (profitable_trades / total_matched) * 100
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear factor cache
        self.factor_values.clear()
        
        # Call parent shutdown
        await super().shutdown()