# backtest_engine_factory/market_replay_backtest_engine.py

from typing import Dict, List, Any, Optional
import pandas as pd

from src.backtest.base_backtest_engine import BacktestEngine
from src.execution.execution_engine import ExecutionEngine
from src.risk.risk_manager import BacktestRiskManager
from src.backtest.performance_monitor import PerformanceMonitor


class MarketReplayBacktestEngine(BacktestEngine):
    """
    Market Replay Backtest Engine
    
    Provides high-fidelity market simulation with realistic order execution,
    market impact, and order queuing.
    """
    
    async def _initialize_backtest(self, symbols: List[str], timeframe: str) -> None:
        """
        Initialize market replay backtest
        
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
        
        # Order management
        self.pending_orders = []
        self.executed_orders = []
        self.order_history = []
        
        # Data storage
        self.historical_data = {}
        self.modified_data = {}  # Data modified by order execution
        
        # Market impact simulation
        self.market_impact_factor = self.config.get("backtest", "market_impact", default=0.0001)
        self.modify_data = self.config.get("backtest", "modify_data", default=True)
        
        # Initialize strategy
        await self.strategy.initialize()
        
        self.logger.info(f"Initializing Market Replay backtest | Initial capital: {self.initial_capital}")
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data for market replay
        
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
        
        # Store the original data
        self.historical_data = data_map
        
        # Create a deep copy for modifications during replay
        self.modified_data = {symbol: df.copy() for symbol, df in data_map.items()}
        
        return data_map
    
    def _get_time_points(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Get all time points from historical data
        
        Args:
            data: Historical data
            
        Returns:
            List[Any]: Sorted unique time points
        """
        all_timestamps = []
        
        for df in data.values():
            # Use 'datetime' column if available, otherwise try index
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
            elif isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.extend(df.index.tolist())
        
        # Return sorted unique timestamps
        return sorted(set(all_timestamps))
    
    def _get_data_at_time_point(self, data: Dict[str, pd.DataFrame], time_point: Any) -> Dict[str, pd.DataFrame]:
        """
        Get data for all symbols at the specified timestamp
        
        Args:
            data: Historical data
            time_point: Time point
            
        Returns:
            Dict[str, pd.DataFrame]: Data at time point
        """
        result = {}
        
        # We use modified_data to incorporate market impact
        for symbol, df in self.modified_data.items():
            # Get data at the specific timestamp
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
        
        # Process pending orders first (realistic order execution)
        await self._process_pending_orders(data)
        
        # Process each symbol
        for symbol, symbol_data in data.items():
            # Generate signals
            signals = await self.strategy.process_data(symbol_data, symbol)
            
            # Validate signals through risk management
            valid_signals = await self.risk_manager.validate_signals(signals)
            
            # Create orders from valid signals
            if not valid_signals.empty:
                await self._create_orders(valid_signals, data)
    
    async def _process_pending_orders(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Process all pending orders against current market data
        
        Args:
            data: Current market data
        """
        still_pending = []
        
        for order in self.pending_orders:
            symbol = order['symbol']
            
            # Skip if we don't have data for this symbol
            if symbol not in data or data[symbol].empty:
                still_pending.append(order)
                continue
            
            # Get current price data
            current_bar = data[symbol].iloc[0]
            
            # Check if order can be executed
            executed = await self._try_execute_order(order, current_bar)
            
            if executed:
                # Order was executed
                self.executed_orders.append(executed)
                self.order_history.append({**order, 'status': 'executed', 'execution_time': self.state['timestamp']})
                
                # Record trade
                trade_record = self._create_trade_record(executed)
                if trade_record:
                    self.state['trades'].append(trade_record)
                
                # Update modified data to reflect the execution (market impact)
                if self.modify_data:
                    self._apply_market_impact(symbol, executed)
            else:
                # Order still pending
                still_pending.append(order)
        
        # Update pending orders list
        self.pending_orders = still_pending
    
    async def _try_execute_order(self, order: Dict[str, Any], bar_data: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Try to execute an order against current bar data
        
        Args:
            order: Order details
            bar_data: Current price bar
            
        Returns:
            Optional[Dict[str, Any]]: Executed order details or None if not executed
        """
        action = order['action'].lower()
        price = order.get('price')  # Limit price if specified
        quantity = order['quantity']
        
        # For market orders or if no price specified
        if price is None:
            # Market order - use current close with slippage
            if action == 'buy':
                execution_price = bar_data['close'] * (1 + self.slippage)
            else:  # sell
                execution_price = bar_data['close'] * (1 - self.slippage)
                
            # Execute the order
            executed_order = {
                **order,
                'execution_price': execution_price,
                'execution_time': self.state['timestamp'],
                'status': 'filled'
            }
            
            return executed_order
            
        else:
            # Limit order - check if price level was reached
            if action == 'buy' and bar_data['low'] <= price:
                # Buy limit triggered
                execution_price = min(price, bar_data['open'])  # Best case is open price
                
                executed_order = {
                    **order,
                    'execution_price': execution_price,
                    'execution_time': self.state['timestamp'],
                    'status': 'filled'
                }
                
                return executed_order
                
            elif action == 'sell' and bar_data['high'] >= price:
                # Sell limit triggered
                execution_price = max(price, bar_data['open'])  # Best case is open price
                
                executed_order = {
                    **order,
                    'execution_price': execution_price,
                    'execution_time': self.state['timestamp'],
                    'status': 'filled'
                }
                
                return executed_order
        
        # Order not executed
        return None
    
    def _create_trade_record(self, executed_order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Create a trade record from an executed order
        
        Args:
            executed_order: Executed order details
            
        Returns:
            Dict[str, Any]: Trade record
        """
        symbol = executed_order['symbol']
        action = executed_order['action'].lower()
        quantity = executed_order['quantity']
        price = executed_order['execution_price']
        
        # Calculate commission
        commission = price * quantity * self.commission_rate
        
        # Update portfolio state
        if action == 'buy':
            # Check if we have enough cash
            total_cost = quantity * price + commission
            if total_cost > self.state['cash']:
                self.logger.warning(f"Insufficient cash for {action} {quantity} {symbol} @ {price}")
                # Adjust quantity to available cash
                quantity = max(0, (self.state['cash'] - commission) / price)
                if quantity <= 0:
                    return None
            
            # Update cash
            self.state['cash'] -= quantity * price + commission
            
            # Update position
            if symbol not in self.state['positions']:
                self.state['positions'][symbol] = {
                    'quantity': 0,
                    'avg_price': 0
                }
            
            position = self.state['positions'][symbol]
            total_qty = position['quantity'] + quantity
            avg_price = ((position['quantity'] * position['avg_price']) + (quantity * price)) / total_qty if total_qty > 0 else 0
            
            position['quantity'] = total_qty
            position['avg_price'] = avg_price
            
        elif action == 'sell':
            # Check if we have the position
            if symbol not in self.state['positions'] or self.state['positions'][symbol]['quantity'] < quantity:
                self.logger.warning(f"Insufficient position for {action} {quantity} {symbol}")
                # Adjust quantity to available position
                quantity = self.state['positions'].get(symbol, {}).get('quantity', 0)
                if quantity <= 0:
                    return None
                
            # Calculate trade PnL
            position = self.state['positions'][symbol]
            entry_price = position['avg_price']
            trade_pnl = (price - entry_price) * quantity
            
            # Update cash
            self.state['cash'] += quantity * price - commission
            
            # Update position
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                del self.state['positions'][symbol]
        
        # Create trade record
        trade = {
            'timestamp': self.state['timestamp'],
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'slippage': abs(price - self.state['market_prices'].get(symbol, price)),
            'cash_after': self.state['cash'],
            'order_id': executed_order.get('order_id', '')
        }
        
        # Record trade in performance monitor
        if self.performance_monitor:
            # For performance monitor, we use dummy prices depending on action
            # This isn't perfect but handles the performance_monitor API
            if action == 'buy':
                self.performance_monitor.record_trade(
                    timestamp=self.state['timestamp'],
                    symbol=symbol,
                    direction=action,
                    entry_price=price,
                    exit_price=price,  # Dummy for buy
                    quantity=quantity,
                    commission=commission
                )
            else:  # sell
                entry_price = self.state['positions'].get(symbol, {}).get('avg_price', price)
                self.performance_monitor.record_trade(
                    timestamp=self.state['timestamp'],
                    symbol=symbol,
                    direction=action,
                    entry_price=entry_price,
                    exit_price=price,
                    quantity=quantity,
                    commission=commission
                )
        
        return trade
    
    def _apply_market_impact(self, symbol: str, executed_order: Dict[str, Any]) -> None:
        """
        Apply market impact to modify market data
        
        Args:
            symbol: Trading symbol
            executed_order: Executed order details
        """
        if symbol not in self.modified_data:
            return
            
        df = self.modified_data[symbol]
        if df.empty:
            return
            
        # Get the current bar
        if 'datetime' in df.columns:
            current_bar_mask = df['datetime'] == self.state['timestamp']
        else:
            current_bar_mask = df.index == self.state['timestamp']
            
        if not any(current_bar_mask):
            return
            
        # Get action and quantity
        action = executed_order['action'].lower()
        quantity = executed_order['quantity']
        
        # Update volume
        if 'volume' in df.columns:
            df.loc[current_bar_mask, 'volume'] += quantity
        
        # Calculate price impact based on quantity and impact factor
        price_impact = quantity * self.market_impact_factor
        
        # Apply market impact to prices
        if action == 'buy':
            # Buy order pushes prices up
            if 'high' in df.columns:
                df.loc[current_bar_mask, 'high'] *= (1 + price_impact)
            if 'close' in df.columns:
                df.loc[current_bar_mask, 'close'] *= (1 + price_impact * 0.5)
        else:
            # Sell order pushes prices down
            if 'low' in df.columns:
                df.loc[current_bar_mask, 'low'] *= (1 - price_impact)
            if 'close' in df.columns:
                df.loc[current_bar_mask, 'close'] *= (1 - price_impact * 0.5)
    
    async def _create_orders(self, signals: pd.DataFrame, current_data: Dict[str, pd.DataFrame]) -> None:
        """
        Create orders from trading signals
        
        Args:
            signals: Trading signals
            current_data: Current market data
        """
        for _, signal in signals.iterrows():
            try:
                symbol = signal['symbol']
                action = signal['action'].lower()
                
                # Skip if we don't have data for this symbol
                if symbol not in current_data or current_data[symbol].empty:
                    continue
                
                # Get current price
                current_price = current_data[symbol]['close'].iloc[0]
                
                # Calculate quantity if not specified
                if 'quantity' not in signal:
                    # Calculate quantity based on portfolio value and risk settings
                    risk_per_trade = self.config.get("risk", "risk_per_trade", default=0.01)  # 1% risk per trade
                    max_position = self.config.get("trading", "limits", "position", default=0.1)  # Max 10% in one position
                    
                    # Calculate position size based on risk
                    portfolio_value = self._calculate_equity()
                    max_trade_value = portfolio_value * max_position
                    risk_adjusted_value = portfolio_value * risk_per_trade
                    
                    # Use the smaller of the two values
                    trade_value = min(max_trade_value, risk_adjusted_value)
                    
                    # Convert to quantity
                    quantity = trade_value / current_price
                else:
                    quantity = signal['quantity']
                
                # Check position limits for buys (don't restrict selling)
                if action == 'buy':
                    max_position_size = self.config.get("trading", "limits", "position", default=0.1)
                    new_position_size = (quantity * current_price) / self._calculate_equity()
                    
                    if new_position_size > max_position_size:
                        # Reduce quantity to limit
                        old_quantity = quantity
                        quantity = (max_position_size * self._calculate_equity()) / current_price
                        self.logger.warning(f"Reduced position size from {old_quantity} to {quantity} due to position limit")
                
                # Create order
                order = {
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'order_type': 'market',
                    'creation_time': self.state['timestamp'],
                    'price': signal.get('price'),  # Limit price if specified
                    'order_id': f"order_{len(self.pending_orders) + len(self.executed_orders)}"
                }
                
                # Add to pending orders
                self.pending_orders.append(order)
                self.logger.debug(f"Created {action} order for {quantity} {symbol}")
            
            except Exception as e:
                self.logger.error(f"Error creating order: {e}")
    
    async def _close_positions(self) -> None:
        """Close all open positions at the end of backtest"""
        for symbol, position in list(self.state['positions'].items()):
            # Check if we have a current price
            if symbol not in self.state['market_prices']:
                continue
                
            current_price = self.state['market_prices'][symbol]
            
            # Create a market sell order
            order = {
                'symbol': symbol,
                'action': 'sell',
                'quantity': position['quantity'],
                'order_type': 'market',
                'creation_time': self.state['timestamp'],
                'price': None,  # Market order
                'order_id': f"close_{symbol}"
            }
            
            # Add to order history
            self.order_history.append({
                **order,
                'status': 'closed_at_end',
                'execution_time': self.state['timestamp'],
                'execution_price': current_price
            })
            
            # Calculate trade value
            trade_value = position['quantity'] * current_price
            
            # Calculate commission
            commission = trade_value * self.commission_rate
            
            # Update cash
            self.state['cash'] += trade_value - commission
            
            # Record the trade
            trade = {
                'timestamp': self.state['timestamp'],
                'symbol': symbol,
                'action': 'sell',
                'quantity': position['quantity'],
                'price': current_price,
                'commission': commission,
                'slippage': 0,
                'cash_after': self.state['cash'],
                'order_id': order['order_id'],
                'note': 'position_close'
            }
            
            self.state['trades'].append(trade)
            self.logger.info(f"Closed position: {position['quantity']} {symbol} at {current_price}")
            
            # Record in performance monitor
            if self.performance_monitor:
                self.performance_monitor.record_trade(
                    timestamp=self.state['timestamp'],
                    symbol=symbol,
                    direction='sell',
                    entry_price=position['avg_price'],
                    exit_price=current_price,
                    quantity=position['quantity'],
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
        
        # Calculate performance metrics from internal state
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
                'slippage': self.slippage,
                'market_impact': self.market_impact_factor
            },
            'strategy': self.strategy.__class__.__name__,
            'open_positions': self.state['positions'],
            'remaining_cash': self.state['cash'],
            'trades': trades,
            'orders': self.order_history,
            'engine_type': 'market_replay'
        }
        
        return report