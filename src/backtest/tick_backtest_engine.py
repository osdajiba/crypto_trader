# src/backtest/tick_backtest_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from src.risk.risk_manager import RiskManager
from src.backtest.performance_monitor import PerformanceMonitor
from src.common.log_manager import LogManager

class TickLevelBacktestEngine:
    """
    Advanced Tick-Level Backtesting Engine
    Simulates market conditions with high-fidelity replay
    """
    def __init__(
        self, 
        tick_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
        strategy,
        config: Dict,
        initial_capital: float = 100000
    ):
        """
        Initialize Tick-Level Backtest Engine
        
        Args:
            tick_data (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): Tick-level market data
            strategy: Trading strategy to backtest
            config (Dict): Backtest configuration
            initial_capital (float): Starting capital
        """
        # Logging
        self.logger = LogManager.get_logger("tick_backtest")
        
        # Data Preparation
        self.tick_data = self._prepare_tick_data(tick_data)
        self.symbols = list(self.tick_data.keys())
        
        # Configuration
        self.config = config
        self.initial_capital = initial_capital
        
        # Components
        self.risk_manager = RiskManager(
            config, 
            initial_capital=initial_capital, 
            symbols=self.symbols
        )
        self.performance_monitor = PerformanceMonitor(
            config, 
            initial_balance=initial_capital
        )
        
        # Strategy
        self.strategy = strategy
        
        # Simulation State
        self.current_state = {
            'timestamp': None,
            'market_prices': {},
            'portfolio_value': initial_capital,
            'positions': {},
            'cash': initial_capital
        }
        
        # Trading Parameters
        self.trading_params = {
            'commission_rate': config.get('commission_rate', 0.001),
            'slippage': config.get('slippage', 0.001),
            'max_position_size': config.get('max_position_size', 0.1)
        }
    
    def _prepare_tick_data(
        self, 
        tick_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare and validate tick-level data
        
        Args:
            tick_data: Raw tick data
        
        Returns:
            Dict of prepared tick DataFrames
        """
        # If single DataFrame, convert to dictionary
        if isinstance(tick_data, pd.DataFrame):
            tick_data = {'primary': tick_data}
        
        # Validate and prepare each DataFrame
        prepared_data = {}
        for symbol, df in tick_data.items():
            # Ensure required columns
            required_columns = ['timestamp', 'price', 'volume']
            missing_cols = set(required_columns) - set(df.columns)
            
            if missing_cols:
                raise ValueError(f"Missing columns in {symbol} data: {missing_cols}")
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Convert timestamp if needed
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            prepared_data[symbol] = df
        
        return prepared_data
    
    def run(self) -> Dict:
        """
        Execute full tick-level backtest
        
        Returns:
            Dict: Comprehensive backtest results
        """
        try:
            # Prepare for backtest
            self._initialize_backtest()
            
            # Main backtest loop
            for tick in self._generate_ticks():
                # Update current market state
                self._update_market_state(tick)
                
                # Generate trading signals
                signals = self._generate_signals()
                
                # Execute signals
                self._process_signals(signals)
                
                # Update performance tracking
                self._update_performance()
            
            # Finalize backtest
            return self._finalize_backtest()
        
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise
    
    def _initialize_backtest(self):
        """
        Prepare system for backtesting
        """
        # Initialize strategy
        self.strategy.initialize()
        
        # Reset state
        self.current_state = {
            'timestamp': None,
            'market_prices': {},
            'portfolio_value': self.initial_capital,
            'positions': {},
            'cash': self.initial_capital
        }
    
    def _generate_ticks(self):
        """
        Generate a unified tick stream across all symbols
        
        Yields:
            Dict: Tick information
        """
        # Combine and sort ticks from all symbols
        all_ticks = []
        for symbol, df in self.tick_data.items():
            for _, row in df.iterrows():
                all_ticks.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    'price': row['price'],
                    'volume': row['volume']
                })
        
        # Sort ticks by timestamp
        all_ticks.sort(key=lambda x: x['timestamp'])
        
        for tick in all_ticks:
            yield tick
    
    def _update_market_state(self, tick: Dict):
        """
        Update current market state based on tick
        
        Args:
            tick (Dict): Current market tick
        """
        symbol = tick['symbol']
        
        # Update market prices
        self.current_state['timestamp'] = tick['timestamp']
        self.current_state['market_prices'][symbol] = tick['price']

    def _generate_signals(self) -> List[Dict]:
            """
            Generate trading signals based on current market state
            
            Returns:
                List[Dict]: Trading signals
            """
            try:
                # Prepare data for strategy
                market_data = {
                    symbol: pd.DataFrame({
                        'timestamp': [self.current_state['timestamp']],
                        'price': [self.current_state['market_prices'][symbol]]
                    }) 
                    for symbol in self.symbols
                }
                
                # Generate signals using strategy
                signals = self.strategy.generate_signals(market_data)
                
                return signals
            except Exception as e:
                self.logger.error(f"Signal generation error: {e}")
                return []
    
    def _process_signals(self, signals: List[Dict]):
        """
        Process and execute trading signals
        
        Args:
            signals (List[Dict]): Trading signals to process
        """
        for signal in signals:
            try:
                # Validate signal
                validation = self._validate_signal(signal)
                
                if validation['is_valid']:
                    # Execute trade
                    self._execute_trade(signal)
            except Exception as e:
                self.logger.error(f"Signal processing error: {signal} - {e}")
    
    def _validate_signal(self, signal: Dict) -> Dict:
        """
        Validate trading signal against risk parameters
        
        Args:
            signal (Dict): Trading signal
        
        Returns:
            Dict: Validation result
        """
        symbol = signal['symbol']
        current_price = self.current_state['market_prices'][symbol]
        
        # Use risk manager to validate signal
        validation = self.risk_manager.validate_trade_signal(
            symbol=symbol,
            signal_type=signal['action'],
            quantity=signal.get('quantity', 0),
            current_price=current_price
        )
        
        return validation
    
    def _execute_trade(self, signal: Dict):
        """
        Execute a validated trading signal
        
        Args:
            signal (Dict): Validated trading signal
        """
        symbol = signal['symbol']
        action = signal['action'].lower()
        current_price = self.current_state['market_prices'][symbol]
        
        # Calculate trade quantity
        quantity = self._calculate_position_size(symbol, current_price)
        
        # Apply slippage
        executed_price = self._apply_slippage(current_price, action)
        
        # Calculate transaction costs
        commission = self._calculate_commission(quantity, executed_price)
        
        # Update positions
        if action == 'buy':
            self._execute_buy(symbol, quantity, executed_price, commission)
        elif action == 'sell':
            self._execute_sell(symbol, quantity, executed_price, commission)
    
    def _calculate_position_size(self, symbol: str, current_price: float) -> float:
        """
        Calculate optimal position size
        
        Args:
            symbol (str): Trading symbol
            current_price (float): Current market price
        
        Returns:
            float: Position size
        """
        # Calculate position size based on risk parameters
        max_risk_amount = self.current_state['cash'] * self.trading_params['max_position_size']
        position_size = max_risk_amount / current_price
        
        return position_size
    
    def _apply_slippage(self, price: float, action: str) -> float:
        """
        Apply slippage to trade execution
        
        Args:
            price (float): Original price
            action (str): Trade action (buy/sell)
        
        Returns:
            float: Price after slippage
        """
        slippage = self.trading_params['slippage']
        
        if action == 'buy':
            return price * (1 + slippage)
        else:  # sell
            return price * (1 - slippage)
    
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """
        Calculate trade commission
        
        Args:
            quantity (float): Trade quantity
            price (float): Execution price
        
        Returns:
            float: Commission amount
        """
        total_trade_value = quantity * price
        return total_trade_value * self.trading_params['commission_rate']
    
    def _execute_buy(self, symbol: str, quantity: float, price: float, commission: float):
        """
        Execute buy trade
        
        Args:
            symbol (str): Trading symbol
            quantity (float): Trade quantity
            price (float): Execution price
            commission (float): Transaction commission
        """
        # Calculate total cost
        total_cost = quantity * price + commission
        
        # Check if sufficient cash
        if total_cost > self.current_state['cash']:
            self.logger.warning(f"Insufficient funds for buy order: {symbol}")
            return
        
        # Update cash and positions
        self.current_state['cash'] -= total_cost
        
        # Update or create position
        if symbol not in self.current_state['positions']:
            self.current_state['positions'][symbol] = {
                'quantity': quantity,
                'avg_price': price
            }
        else:
            existing_pos = self.current_state['positions'][symbol]
            total_quantity = existing_pos['quantity'] + quantity
            avg_price = (
                (existing_pos['quantity'] * existing_pos['avg_price']) + 
                (quantity * price)
            ) / total_quantity
            
            existing_pos.update({
                'quantity': total_quantity,
                'avg_price': avg_price
            })
        
        # Record trade
        self.performance_monitor.record_trade(
            timestamp=self.current_state['timestamp'],
            symbol=symbol,
            direction='buy',
            entry_price=price,
            exit_price=None,  # Not applicable for buy
            quantity=quantity,
            commission=commission
        )
    
    def _execute_sell(self, symbol: str, quantity: float, price: float, commission: float):
        """
        Execute sell trade
        
        Args:
            symbol (str): Trading symbol
            quantity (float): Trade quantity
            price (float): Execution price
            commission (float): Transaction commission
        """
        # Check if position exists and has sufficient quantity
        if symbol not in self.current_state['positions']:
            self.logger.warning(f"No position to sell: {symbol}")
            return
        
        position = self.current_state['positions'][symbol]
        if position['quantity'] < quantity:
            self.logger.warning(f"Insufficient position to sell: {symbol}")
            return
        
        # Calculate profit/loss
        entry_price = position['avg_price']
        profit_loss = (price - entry_price) * quantity
        
        # Update cash
        total_revenue = quantity * price - commission
        self.current_state['cash'] += total_revenue
        
        # Update position
        position['quantity'] -= quantity
        
        # Remove position if fully closed
        if position['quantity'] == 0:
            del self.current_state['positions'][symbol]
        
        # Record trade
        self.performance_monitor.record_trade(
            timestamp=self.current_state['timestamp'],
            symbol=symbol,
            direction='sell',
            entry_price=entry_price,
            exit_price=price,
            quantity=quantity,
            commission=commission
        )
    
    def _update_performance(self):
        """
        Update performance tracking at each tick
        """
        # Calculate current portfolio value
        portfolio_value = self.current_state['cash']
        
        # Add positions value
        for symbol, position in self.current_state['positions'].items():
            current_price = self.current_state['market_prices'][symbol]
            portfolio_value += position['quantity'] * current_price
        
        # Update equity curve
        self.performance_monitor.update_equity_curve(
            timestamp=self.current_state['timestamp'],
            current_balance=portfolio_value
        )
    
    def _finalize_backtest(self) -> Dict:
        """
        Finalize backtest and generate comprehensive report
        
        Returns:
            Dict: Comprehensive backtest results
        """
        # Close any remaining positions
        self._liquidate_positions()
        
        # Generate performance reports
        performance_report = self.performance_monitor.generate_detailed_report()
        risk_report = self.risk_manager.generate_risk_report()
        
        # Combine reports
        final_report = {
            'performance': performance_report,
            'risk': risk_report,
            'configuration': {
                'initial_capital': self.initial_capital,
                'trading_params': self.trading_params
            }
        }
        
        # Generate visualizations
        self.performance_monitor.visualize_performance()
        self.performance_monitor.export_report()
        
        return final_report
    
    def _liquidate_positions(self):
        """
        Liquidate all open positions at the end of backtest
        """
        for symbol in list(self.current_state['positions'].keys()):
            position = self.current_state['positions'][symbol]
            current_price = self.current_state['market_prices'][symbol]
            
            # Execute sell trade
            self._execute_sell(
                symbol=symbol,
                quantity=position['quantity'],
                price=current_price,
                commission=self._calculate_commission(position['quantity'], current_price)
            )