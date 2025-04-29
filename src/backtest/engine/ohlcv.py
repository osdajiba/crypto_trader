#!/usr/bin/env python3
# src/backtest/engine/ohlcv.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time

from src.common.abstract_factory import register_factory_class
from src.backtest.engine.base import BaseBacktestEngine


@register_factory_class('backtest_factory', 'ohlcv', 
    description="OHLCV Engine for vectorized backtesting",
    category="backtest")
class OHLCVEngine(BaseBacktestEngine):
    """OHLCV Engine for vectorized backtesting on OHLCV data"""
    
    def __init__(self, config, params=None):
        """Initialize OHLCV backtest engine"""
        super().__init__(config, params)
        
        # OHLCV specific settings
        self.initial_capital = self.params.get('initial_capital', 100000)
        self.commission_rate = self.params.get('commission_rate', 0.001)
        self.slippage = self.params.get('slippage', 0.001)
        
        # Factor calculation cache
        self.factor_values = {}  # Symbol -> Factor -> Series

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run vectorized backtest on OHLCV data"""
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self._is_running = True
        
        # Initialize portfolio state
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
            # Calculate all factors for each symbol in vectorized manner
            for symbol, df in data.items():
                if not df.empty:
                    await self._calculate_all_factors(df, symbol)
            
            # Generate signals in vectorized manner
            all_signals = {}
            for symbol, df in data.items():
                if not df.empty:
                    signals = await self._generate_signals_vectorized(df, symbol)
                    if not signals.empty:
                        all_signals[symbol] = signals
            
            # Simulate portfolio
            trades, equity_curve = self._simulate_portfolio(all_signals, data)
            
            # Store results
            results['signals'] = all_signals
            results['trades'] = trades
            results['equity_curve'] = equity_curve
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.metrics['processing_time'] = execution_time
            
            # Calculate final portfolio value
            final_value = self.portfolio_history[-1]['portfolio_value'] if self.portfolio_history else self.initial_capital
            
            # Calculate baseline performance (buy & hold)
            main_symbol = list(data.keys())[0] if data else None
            if main_symbol and main_symbol in data and not data[main_symbol].empty:
                initial_price = data[main_symbol]['close'].iloc[0]
                final_price = data[main_symbol]['close'].iloc[-1]
                baseline_return = (final_price - initial_price) / initial_price
                baseline_return_pct = baseline_return * 100
                
                # Calculate alpha (strategy outperformance)
                strategy_return_pct = ((final_value / self.initial_capital) - 1) * 100
                
                results['metrics'].update({
                    'baseline_initial_price': initial_price,
                    'baseline_final_price': final_price,
                    'baseline_return_pct': baseline_return_pct,
                    'alpha': strategy_return_pct - baseline_return_pct
                })
            
            # Add performance metrics
            results['metrics'].update({
                **self.metrics,
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': final_value - self.initial_capital,
                'total_return_pct': ((final_value / self.initial_capital) - 1) * 100,
                'total_trades': len(trades),
                'symbols_traded': len(set(trade['symbol'] for trade in trades)) if trades else 0
            })
            
            self.logger.info(f"OHLCV vectorized backtest completed in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during OHLCV backtest: {str(e)}")
            return {'error': str(e), 'metrics': self.metrics}
        finally:
            self._is_running = False
    
    async def _calculate_all_factors(self, data: pd.DataFrame, symbol: str) -> None:
        """Calculate all factors for a symbol's data in vectorized manner"""
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
        """Calculate factor values for entire dataset"""
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
        """Generate signals for entire dataset at once"""
        try:
            # Check if strategy has vectorized signal generation
            if hasattr(self.strategy, '_generate_signals_vectorized'):
                # Use vectorized method if available
                signals = await self.strategy._generate_signals_vectorized(data, symbol, self.factor_values.get(symbol, {}))
                
                # Add symbol if not present
                if not signals.empty and 'symbol' not in signals.columns:
                    signals['symbol'] = symbol
                
                return signals
            
            # Enrich data with factor values
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
        """Apply strategy logic to entire dataset"""
        # For strategies with specific factors (e.g. dual MA), optimize signal generation
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
        
        # For other strategies, fallback to point-by-point processing
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
        """Simulate portfolio performance based on generated signals"""
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
            
            # Simulate each timestamp sequentially
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
            
            # Store for class access
            self.portfolio_history = portfolio['history']
            
            return portfolio['trades'], equity_curve
            
        except Exception as e:
            self.logger.error(f"Error simulating portfolio: {str(e)}")
            return [], pd.DataFrame()
    
    def _execute_signal(self, signal: Dict, current_prices: Dict[str, float], portfolio: Dict) -> None:
        """Execute a trading signal"""
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
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear factor cache
        self.factor_values.clear()
        
        # Call parent shutdown
        await super().shutdown()