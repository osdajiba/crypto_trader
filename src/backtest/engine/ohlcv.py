#!/usr/bin/env python3
# src/backtest/engine/ohlcv.py

"""
OHLCV backtest engine implementation.
Provides vectorized backtesting for OHLCV (Open, High, Low, Close, Volume) data.
"""

from collections import deque
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import asyncio

from src.common.abstract_factory import register_factory_class
from src.backtest.engine.base import BaseBacktestEngine, BacktestEngineError


class OHLCVEngineError(BacktestEngineError):
    """Error specific to OHLCV engine operations"""
    pass


@register_factory_class('backtest_engine_factory', "ohlcv", 
    description="OHLCV Engine for vectorized backtesting",
    features=["vectorized", "ohlcv", "performance_optimized"],
    category="backtest")
class OHLCVEngine(BaseBacktestEngine):
    """
    OHLCV Engine for vectorized backtesting on OHLCV data
    
    This engine provides optimized performance for backtesting strategies 
    on standard OHLCV (Open, High, Low, Close, Volume) data, with optional
    vectorized processing for maximum speed.
    """
    
    def __init__(self, config, params=None):
        """
        Initialize OHLCV backtest engine
        
        Args:
            config: Configuration manager
            params: Engine parameters
        """
        super().__init__(config, params)
        
        # OHLCV specific settings from config with params override
        self.initial_capital = self.params.get(
            'initial_capital', 
            self.config.get("trading", "capital", "initial", default=100000)
        )
        self.commission_rate = self.params.get(
            'commission_rate', 
            self.config.get("trading", "fees", "commission", default=0.001)
        )
        self.slippage = self.params.get(
            'slippage', 
            self.config.get("trading", "fees", "slippage", default=0.001)
        )
        
        # Factor calculation cache
        self.factor_values = {}  # Symbol -> Factor -> Series
        
        self.logger.info(f"OHLCV Engine initialized with capital={self.initial_capital}, "
                        f"commission={self.commission_rate}, slippage={self.slippage}")

    async def initialize(self) -> None:
        """
        Initialize backtest engine with specific OHLCV requirements
        
        This method extends the base initialization with OHLCV-specific
        settings and configurations.
        """
        await super().initialize()
        
        # Additional OHLCV-specific initialization
        if 'use_vectorized' not in self.params:
            self.params['use_vectorized'] = True
        
        self.use_vectorized = self.params.get('use_vectorized')
        self.logger.info(f"OHLCV Engine vectorized mode: {self.use_vectorized}")

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run vectorized backtest on OHLCV data
        
        This method implements the backtest execution on OHLCV data, optionally
        using vectorized processing for improved performance.
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            Dict[str, Any]: Backtest results
            
        Raises:
            OHLCVEngineError: If backtest execution fails
        """
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
            self.logger.info("Starting OHLCV backtest run")
            
            # Calculate all factors for each symbol in vectorized manner
            if self.use_vectorized:
                for symbol, df in data.items():
                    if not df.empty:
                        await self._calculate_all_factors(df, symbol)
            
            # Generate signals in vectorized manner
            all_signals = {}
            for symbol, df in data.items():
                if not df.empty:
                    if self.use_vectorized:
                        signals = await self._generate_signals_vectorized(df, symbol)
                    else:
                        signals = await self._generate_signals_sequential(df, symbol)
                        
                    if not signals.empty:
                        all_signals[symbol] = signals
                        self.logger.info(f"Generated {len(signals)} signals for {symbol}")
            
            # Simulate portfolio
            trades, equity_curve = await self._simulate_portfolio(all_signals, data)
            
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
            
            self.logger.info(f"OHLCV vectorized backtest completed in {execution_time:.2f}s with "
                           f"return {results['metrics']['total_return_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during OHLCV backtest: {str(e)}")
            raise OHLCVEngineError(f"Backtest execution failed: {str(e)}")
        finally:
            self._is_running = False
    
    async def _calculate_all_factors(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Calculate all factors for a symbol's data in vectorized manner
        
        Args:
            data: OHLCV data for a symbol
            symbol: Trading symbol
        """
        if not self.strategy or not hasattr(self.strategy, '_factor_registry'):
            return
            
        self.logger.debug(f"Calculating factors for {symbol}")
            
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
            data: OHLCV data for a symbol
            factor_name: Factor name to calculate
            symbol: Trading symbol
            
        Returns:
            Optional[pd.Series]: Factor values or None if calculation fails
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
            data: OHLCV data for a symbol
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Generated signals
        """
        try:
            # Check if strategy has vectorized signal generation
            if hasattr(self.strategy, 'generate_signals_vectorized'):
                # Use vectorized method if available
                signals = await self.strategy.generate_signals_vectorized(data, symbol, self.factor_values.get(symbol, {}))
                
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
    
    async def _generate_signals_sequential(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate signals by processing data sequentially
        
        Args:
            data: OHLCV data for a symbol
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Generated signals
        """
        signals_list = []
        
        # Reset buffer for clean run
        self.data_buffers[symbol] = pd.DataFrame()
        self.has_sufficient_history[symbol] = False
        self.data_queues[symbol] = deque(maxlen=self.required_window_size)
        
        # Process data sequentially
        for i in range(len(data)):
            data_point = data.iloc[[i]]
            signal = await self.process_data_point(data_point, symbol)
            if not signal.empty:
                signals_list.append(signal)
                
            # Log progress periodically
            if i % 1000 == 0 and i > 0:
                self.logger.debug(f"Processed {i}/{len(data)} data points for {symbol}")
        
        if signals_list:
            return pd.concat(signals_list)
        return pd.DataFrame()
    
    async def _apply_strategy_vectorized(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Apply strategy logic to entire dataset
        
        Args:
            data: Enriched OHLCV data with factor values
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Generated signals
        """
        # For strategies with specific factors (e.g. dual MA), optimize signal generation
        if 'factor_short_ma' in data.columns and 'factor_long_ma' in data.columns:
            # Create signal DataFrame
            signals = pd.DataFrame(index=data.index)
            signals['timestamp'] = data.index if isinstance(data.index, pd.DatetimeIndex) else data.get('datetime', data.index)
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
    
    async def _simulate_portfolio(self, all_signals: Dict[str, pd.DataFrame], price_data: Dict[str, pd.DataFrame]) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Simulate portfolio performance based on generated signals
        
        Args:
            all_signals: Dictionary of symbol -> signals DataFrame
            price_data: Dictionary of symbol -> OHLCV data
            
        Returns:
            Tuple[List[Dict], pd.DataFrame]: Trades and equity curve
        """
        try:
            trades = []
            equity_curve = []
            self.cash = self.initial_capital
            self.positions = {}
            self.portfolio_history = []
            
            # Combine all signals from all symbols into a single timeline
            combined_signals = []
            for symbol, signals_df in all_signals.items():
                if not signals_df.empty:
                    # Ensure datetime column exists
                    if 'datetime' not in signals_df.columns and 'timestamp' in signals_df.columns:
                        signals_df['datetime'] = signals_df['timestamp']
                    
                    # Convert to records for easier timeline merging
                    for _, row in signals_df.iterrows():
                        signal_dict = row.to_dict()
                        if 'datetime' not in signal_dict and 'timestamp' in signal_dict:
                            signal_dict['datetime'] = signal_dict['timestamp']
                        combined_signals.append(signal_dict)
            
            # Sort signals by datetime
            if combined_signals:
                combined_signals.sort(key=lambda x: x.get('datetime', pd.Timestamp.min))
                
                # Process signals sequentially
                for signal in combined_signals:
                    symbol = signal.get('symbol')
                    action = signal.get('action', '').lower()
                    price = signal.get('price')
                    
                    if not symbol or not action or not price:
                        continue
                    
                    # Calculate quantity based on portfolio allocation
                    quantity = signal.get('quantity')
                    if not quantity:
                        if action == 'buy':
                            # Default to using 10% of available cash
                            allocation = self.params.get('position_size', 0.1)
                            quantity = (self.cash * allocation) / price
                            
                    # Execute trade
                    if action == 'buy':
                        trade = self._execute_buy(symbol, price, quantity)
                        if trade:
                            trades.append(trade)
                    elif action == 'sell':
                        trade = self._execute_sell(symbol, price, quantity)
                        if trade:
                            trades.append(trade)
                    
                    # Update portfolio value at each signal
                    portfolio_value = self._calculate_portfolio_value(price_data, signal.get('datetime'))
                    self.portfolio_history.append({
                        'timestamp': signal.get('datetime'),
                        'portfolio_value': portfolio_value,
                        'cash': self.cash
                    })
            
            # Create equity curve DataFrame
            if self.portfolio_history:
                equity_curve = pd.DataFrame(self.portfolio_history)
            else:
                # Create empty equity curve with structure
                equity_curve = pd.DataFrame(columns=['timestamp', 'portfolio_value', 'cash'])
            
            self.logger.info(f"Portfolio simulation completed with {len(trades)} trades")
            return trades, equity_curve
        
        except Exception as e:
            self.logger.error(f"Error in portfolio simulation: {str(e)}")
            return [], pd.DataFrame()
    
    def _execute_buy(self, symbol: str, price: float, quantity: float) -> Optional[Dict]:
        """
        Execute a buy order
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Quantity to buy
            
        Returns:
            Optional[Dict]: Trade record or None if trade couldn't be executed
        """
        if quantity <= 0:
            return None
            
        # Apply slippage
        execution_price = price * (1 + self.slippage)
        
        # Calculate total cost including commission
        cost = quantity * execution_price
        commission = cost * self.commission_rate
        total_cost = cost + commission
        
        # Check if we have enough cash
        if total_cost > self.cash:
            # Adjust quantity to available cash
            quantity = (self.cash / execution_price) * (1 - self.commission_rate)
            if quantity <= 0:
                return None
                
            # Recalculate costs
            cost = quantity * execution_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
        
        # Update cash
        self.cash -= total_cost
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
        
        # Create trade record
        trade = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'action': 'buy',
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'total_cost': total_cost,
            'cash_after': self.cash
        }
        
        return trade
    
    def _execute_sell(self, symbol: str, price: float, quantity: float = None) -> Optional[Dict]:
        """
        Execute a sell order
        
        Args:
            symbol: Trading symbol
            price: Execution price
            quantity: Quantity to sell (None for all)
            
        Returns:
            Optional[Dict]: Trade record or None if trade couldn't be executed
        """
        current_position = self.positions.get(symbol, 0)
        
        # If no position, can't sell
        if current_position <= 0:
            return None
            
        # If quantity not specified, sell all
        if quantity is None or quantity > current_position:
            quantity = current_position
            
        # Apply slippage
        execution_price = price * (1 - self.slippage)
        
        # Calculate proceeds
        value = quantity * execution_price
        commission = value * self.commission_rate
        net_proceeds = value - commission
        
        # Update cash
        self.cash += net_proceeds
        
        # Update position
        self.positions[symbol] -= quantity
        if self.positions[symbol] <= 0:
            del self.positions[symbol]
        
        # Create trade record
        trade = {
            'timestamp': pd.Timestamp.now(),
            'symbol': symbol,
            'action': 'sell',
            'quantity': quantity,
            'price': execution_price,
            'commission': commission,
            'net_proceeds': net_proceeds,
            'cash_after': self.cash
        }
        
        return trade
    
    def _calculate_portfolio_value(self, price_data: Dict[str, pd.DataFrame], timestamp) -> float:
        """
        Calculate portfolio value at a specific timestamp
        
        Args:
            price_data: Dictionary of symbol -> OHLCV data
            timestamp: Timestamp to calculate value at
            
        Returns:
            float: Portfolio value
        """
        # Start with cash
        portfolio_value = self.cash
        
        # Add value of all positions
        for symbol, quantity in self.positions.items():
            if symbol in price_data:
                symbol_data = price_data[symbol]
                
                # Find closest price data
                if isinstance(timestamp, pd.Timestamp) and 'datetime' in symbol_data.columns:
                    closest_row = symbol_data.iloc[symbol_data['datetime'].searchsorted(timestamp, side='right') - 1]
                    price = closest_row['close']
                elif len(symbol_data) > 0:
                    # Use latest price if can't match timestamp
                    price = symbol_data['close'].iloc[-1]
                else:
                    continue
                    
                # Add position value
                position_value = quantity * price
                portfolio_value += position_value
        
        return portfolio_value
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear factor cache
        self.factor_values.clear()
        
        # Call parent shutdown
        await super().shutdown()