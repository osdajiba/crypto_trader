# src/backtest/ohlcv_backtest_engine.py

from typing import Dict, List, Any
import pandas as pd

from src.backtest.base_backtest_engine import BacktestEngine
from src.execution.execution_engine import ExecutionEngine
from src.risk.risk_manager import BacktestRiskManager
from src.backtest.performance_monitor import PerformanceMonitor


class OHLCVBacktestEngine(BacktestEngine):
    """OHLCV data-based backtesting engine implementation"""
    
    async def _initialize_backtest(self, symbols: List[str], timeframe: str) -> None:
        """
        Initialize OHLCV backtest
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
        """
        # Save backtest parameters
        self.symbols = symbols
        self.timeframe = timeframe
        
        # Initialize component references
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="backtest"
        )
        
        self.risk_manager = BacktestRiskManager(config=self.config)
        await self.risk_manager.initialize()
        
        self.performance_monitor = PerformanceMonitor(
            config=self.config,
            initial_balance=self.initial_capital
        )
        
        # Initialize strategy
        await self.strategy.initialize()
        
        self.logger.info(f"Initializing OHLCV backtest | Initial capital: {self.initial_capital}")
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load OHLCV historical data
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            
        Returns:
            Dict[str, pd.DataFrame]: Historical data
        """
        data_map = {}
        
        # Set backtest date range
        start_date = self.config.get("backtest", "period", "start", default=None)
        if not start_date:
            start_date = self.config.get("backtest", "start_date", default=None)
            
        end_date = self.config.get("backtest", "period", "end", default=None)
        if not end_date:
            end_date = self.config.get("backtest", "end_date", default=None)
        
        # Load historical data for each symbol
        for symbol in symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date
                )
                
                if not data.empty:
                    data_map[symbol] = data
                    self.logger.info(f"Loaded {symbol} historical data: {len(data)} records")
                else:
                    self.logger.warning(f"No historical data found for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Failed to load {symbol} historical data: {e}")
        
        return data_map
    
    def _get_time_points(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Get all time points
        
        Args:
            data: Historical data
            
        Returns:
            List[Any]: Sorted unique time points
        """
        all_timestamps = []
        
        for df in data.values():
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
            elif isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.extend(df.index.tolist())
        
        # Return sorted unique timestamps
        return sorted(set(all_timestamps))
    
    def _get_data_at_time_point(self, data: Dict[str, pd.DataFrame], time_point: Any) -> Dict[str, pd.DataFrame]:
        """
        Get data at specified time point
        
        Args:
            data: Historical data
            time_point: Time point
            
        Returns:
            Dict[str, pd.DataFrame]: Data at time point
        """
        result = {}
        
        for symbol, df in data.items():
            if 'datetime' in df.columns:
                data_at_time = df[df['datetime'] == time_point]
                if not data_at_time.empty:
                    result[symbol] = data_at_time
            elif isinstance(df.index, pd.DatetimeIndex):
                try:
                    # Try to get exact timestamp match from index
                    data_at_time = df.loc[[time_point]]
                    if not data_at_time.empty:
                        result[symbol] = data_at_time
                except KeyError:
                    # No exact match - try the nearest timestamp if needed
                    pass
        
        return result
    
    async def _process_data_point(self, time_point: Any, data: Dict[str, pd.DataFrame]) -> None:
        """
        Process single time point data
        
        Args:
            time_point: Time point
            data: Time point data
        """
        # Update current timestamp
        self.state['timestamp'] = time_point
        
        # Update market prices
        self._update_market_prices(data)
        
        # Process each symbol
        for symbol, symbol_data in data.items():
            # Generate signals
            signals = await self.strategy.process_data(symbol_data, symbol)
            
            # Validate signals
            valid_signals = await self.risk_manager.validate_signals(signals)
            
            # Execute valid signals
            if not valid_signals.empty:
                executed_trades = self._execute_trades(valid_signals, data)
                
                # Record trades
                if executed_trades:
                    self.state['trades'].extend(executed_trades)
    
    def _execute_trades(self, signals: pd.DataFrame, current_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Execute trading signals
        
        Args:
            signals: Trading signals
            current_data: Current market data
            
        Returns:
            List[Dict[str, Any]]: List of executed trades
        """
        executed_trades = []
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action'].lower()
            
            # Only process buy/sell signals
            if action not in ['buy', 'sell']:
                continue
            
            # Get current price
            if symbol not in current_data or current_data[symbol].empty:
                continue
                
            current_price = current_data[symbol]['close'].iloc[0]
            
            # Calculate execution price (considering slippage)
            execution_price = current_price * (1 + self.slippage) if action == 'buy' else current_price * (1 - self.slippage)
            
            # Calculate quantity
            if 'quantity' in signal:
                quantity = signal['quantity']
            else:
                # Calculate position size (based on risk parameters)
                risk_per_trade = self.config.get("risk", "risk_per_trade", default=0.01)
                max_position = self.config.get("trading", "limits", "position", default=0.1)
                
                # Calculate based on portfolio value
                portfolio_value = self._calculate_equity()
                max_trade_value = portfolio_value * max_position
                risk_adjusted_value = portfolio_value * risk_per_trade
                
                # Use the smaller of the two values
                trade_value = min(max_trade_value, risk_adjusted_value)
                quantity = trade_value / execution_price
            
            # Validate trade
            if action == 'buy':
                # Check available cash
                total_cost = quantity * execution_price * (1 + self.commission_rate)
                if total_cost > self.state['cash']:
                    quantity = self.state['cash'] / (execution_price * (1 + self.commission_rate))
                    if quantity <= 0:
                        continue
            elif action == 'sell':
                # Check available position
                current_position = self.state['positions'].get(symbol, {}).get('quantity', 0)
                if current_position <= 0:
                    continue
                quantity = min(quantity, current_position)
            
            # Execute trade
            commission_cost = quantity * execution_price * self.commission_rate
            
            if action == 'buy':
                # Update cash
                self.state['cash'] -= (quantity * execution_price + commission_cost)
                
                # Create or update position
                if symbol not in self.state['positions']:
                    self.state['positions'][symbol] = {
                        'quantity': 0,
                        'avg_price': 0
                    }
                
                position = self.state['positions'][symbol]
                total_qty = position['quantity'] + quantity
                avg_price = ((position['quantity'] * position['avg_price']) + (quantity * execution_price)) / total_qty if total_qty > 0 else 0
                
                position['quantity'] = total_qty
                position['avg_price'] = avg_price
                
            else:  # sell
                # Update cash
                self.state['cash'] += (quantity * execution_price - commission_cost)
                
                # Update position
                position = self.state['positions'][symbol]
                position['quantity'] -= quantity
                
                if position['quantity'] <= 0:
                    del self.state['positions'][symbol]
            
            # Record trade
            trade = {
                'timestamp': current_data[symbol]['datetime'].iloc[0],
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': execution_price,
                'commission': commission_cost,
                'slippage': abs(execution_price - current_price),
                'cash_after': self.state['cash']
            }
            
            executed_trades.append(trade)
            
            # Record trade in performance monitor
            if self.performance_monitor:
                # For performance monitor, we use dummy prices depending on action
                if action == 'buy':
                    self.performance_monitor.record_trade(
                        timestamp=self.state['timestamp'],
                        symbol=symbol,
                        direction=action,
                        entry_price=execution_price,
                        exit_price=execution_price,  # Dummy for buy
                        quantity=quantity,
                        commission=commission_cost
                    )
                else:  # sell
                    # For sell, we use the position average price as the entry
                    if symbol in self.state['positions']:
                        entry_price = self.state['positions'][symbol]['avg_price']
                    else:
                        # Position already closed
                        entry_price = execution_price * 0.9  # Placeholder
                        
                    self.performance_monitor.record_trade(
                        timestamp=self.state['timestamp'],
                        symbol=symbol,
                        direction=action,
                        entry_price=entry_price,
                        exit_price=execution_price,
                        quantity=quantity,
                        commission=commission_cost
                    )
        
        return executed_trades
    
    async def _close_positions(self) -> None:
        """Close all open positions at the end of backtest"""
        for symbol, position in list(self.state['positions'].items()):
            # Check if we have a current price
            if symbol not in self.state['market_prices']:
                continue
                
            current_price = self.state['market_prices'][symbol]
            quantity = position['quantity']
            
            # Calculate commission
            commission = quantity * current_price * self.commission_rate
            
            # Update cash
            self.state['cash'] += (quantity * current_price - commission)
            
            # Record trade
            trade = {
                'timestamp': self.state['timestamp'],
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': current_price,
                'commission': commission,
                'slippage': 0,
                'cash_after': self.state['cash'],
                'note': 'position_close'
            }
            
            self.state['trades'].append(trade)
            self.logger.info(f"Closed position: {quantity} {symbol} at {current_price}")
            
            # Record in performance monitor
            if self.performance_monitor:
                self.performance_monitor.record_trade(
                    timestamp=self.state['timestamp'],
                    symbol=symbol,
                    direction='sell',
                    entry_price=position['avg_price'],
                    exit_price=current_price,
                    quantity=quantity,
                    commission=commission
                )
        
        # Clear positions
        self.state['positions'] = {}
    
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        Generate backtest report
        
        Returns:
            Dict[str, Any]: Backtest report
        """
        final_equity = self._calculate_equity()
        
        # Calculate performance metrics
        total_return = final_equity - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Calculate trade statistics
        trades = self.state['trades']
        buy_trades = len([t for t in trades if t['action'] == 'buy'])
        sell_trades = len([t for t in trades if t['action'] == 'sell'])
        
        # Get advanced metrics from performance monitor
        performance_metrics = {}
        if self.performance_monitor:
            self.performance_monitor.calculate_performance_metrics()
            performance_report = self.performance_monitor.generate_detailed_report()
            performance_metrics = performance_report.get('performance_metrics', {})
        
        # Compile report
        report = {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': self.state['max_drawdown'] * 100,
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'sortino_ratio': performance_metrics.get('sortino_ratio', 0),
            'win_rate': performance_metrics.get('win_rate', 0) * 100 if isinstance(performance_metrics.get('win_rate', 0), float) else 0,
            'profit_factor': performance_metrics.get('profit_factor', 0),
            'total_trades': len(trades),
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'backtest_params': {
                'symbols': self.symbols,
                'timeframe': self.timeframe,
                'start_date': self.config.get("backtest", "period", "start", default=""),
                'end_date': self.config.get("backtest", "period", "end", default=""),
                'commission_rate': self.commission_rate,
                'slippage': self.slippage
            },
            'strategy': self.strategy.__class__.__name__,
            'open_positions': self.state['positions'],
            'remaining_cash': self.state['cash'],
            'trades': trades,
            'engine_type': 'ohlcv'
        }
        
        return report