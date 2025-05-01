#!/usr/bin/env python3
# src/backtest/engine/market_replay.py

"""
Market Replay backtest engine implementation.
Simulates trading with sequential processing of market data.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import time

from src.common.abstract_factory import register_factory_class
from src.common.helpers import BacktestEngine
from src.backtest.engine.base import BaseBacktestEngine, BacktestEngineError


class MarketReplayEngineError(BacktestEngineError):
    """Error specific to market replay engine operations"""
    pass


@register_factory_class('backtest_engine_factory', BacktestEngine.MARKETREPLAY.value, 
    description="Market Replay Backtest Engine for sequential data processing",
    features=["sequential", "realistic_execution", "detailed_simulation"],
    category="backtest")
class MarketReplayEngine(BaseBacktestEngine):
    """
    Market Replay Backtest Engine
    
    Processes data sequentially, simulating real-time market conditions with
    realistic order execution and position tracking. This engine provides
    a more accurate simulation of live trading conditions compared to
    vectorized approaches.
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
            self.config.get("trading", "fees", "commission", default=0.001)
        )
        self.slippage = self.params.get(
            'slippage', 
            self.config.get("trading", "fees", "slippage", default=0.001)
        )
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_history = []  # Snapshots of portfolio value over time

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run market replay backtest, processing data sequentially
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
            
        Returns:
            Dict[str, Any]: Backtest results
            
        Raises:
            MarketReplayEngineError: If backtest execution fails
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
            
            # Track baseline performance (underlying asset prices)
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
                
                # Track baseline (underlying asset) price
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
            
            # Add baseline prices to results
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
            
            # Calculate baseline performance metrics
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
            
            self.logger.info(f"Market replay completed in {execution_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during market replay: {str(e)}")
            raise MarketReplayEngineError(f"Market replay failed: {str(e)}")
        finally:
            self._is_running = False

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
        
        return result
    
    def _execute_signals(self, signals: pd.DataFrame, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Execute trading signals and return transactions
        
        Args:
            signals: Trading signals DataFrame
            current_data: Current market data for all symbols
            
        Returns:
            List[Dict]: List of executed trades
        """
        if signals.empty:
            return []
        
        trades = []
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action'].lower()
            timestamp = signal.get('timestamp')
            
            # Get current price
            if symbol not in current_data or current_data[symbol].empty:
                continue
            
            price_data = current_data[symbol]
            price = price_data['close'].iloc[0]
            
            # Apply slippage
            if action == 'buy':
                execution_price = price * (1 + self.slippage)
            else:  # sell
                execution_price = price * (1 - self.slippage)
            
            # Determine quantity
            quantity = signal.get('quantity')
            if quantity is None:
                # Calculate default position size (1% of portfolio)
                risk_pct = self.params.get('risk_per_trade', 0.01)
                quantity = (self.cash * risk_pct) / execution_price
            
            # Execute trade
            if action == 'buy':
                # Check if we have enough cash
                cost = quantity * execution_price
                commission = cost * self.commission_rate
                total_cost = cost + commission
                
                if total_cost > self.cash:
                    # Adjust quantity to available cash
                    max_quantity = self.cash / (execution_price * (1 + self.commission_rate))
                    quantity = max_quantity
                    cost = quantity * execution_price
                    commission = cost * self.commission_rate
                    total_cost = cost + commission
                
                if quantity > 0:
                    # Update cash and position
                    self.cash -= total_cost
                    
                    # Update position
                    if symbol not in self.positions:
                        self.positions[symbol] = 0
                    self.positions[symbol] += quantity
                    
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'buy',
                        'quantity': quantity,
                        'price': execution_price,
                        'commission': commission,
                        'cost': total_cost,
                        'cash_after': self.cash
                    }
                    trades.append(trade)
            
            elif action == 'sell':
                # Check if we have the position
                current_position = self.positions.get(symbol, 0)
                
                if current_position > 0:
                    # Limit quantity to current position
                    quantity = min(quantity, current_position)
                    
                    # Calculate value
                    value = quantity * execution_price
                    commission = value * self.commission_rate
                    net_value = value - commission
                    
                    # Update cash and position
                    self.cash += net_value
                    self.positions[symbol] -= quantity
                    
                    # Remove position if zero
                    if self.positions[symbol] <= 0:
                        del self.positions[symbol]
                    
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': 'sell',
                        'quantity': quantity,
                        'price': execution_price,
                        'commission': commission,
                        'value': net_value,
                        'cash_after': self.cash
                    }
                    trades.append(trade)
        
        return trades
    
    def _calculate_portfolio_value(self, current_data: Dict[str, pd.DataFrame]) -> float:
        """
        Calculate current portfolio value
        
        Args:
            current_data: Current market data for all symbols
            
        Returns:
            float: Current portfolio value
        """
        portfolio_value = self.cash
        
        # Add position values
        for symbol, quantity in self.positions.items():
            if symbol in current_data and not current_data[symbol].empty:
                price = current_data[symbol]['close'].iloc[0]
                position_value = quantity * price
                portfolio_value += position_value
        
        return portfolio_value
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Reset portfolio state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Call parent shutdown
        await super().shutdown()