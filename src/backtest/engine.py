# src/backtest/base_backtest_engine.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Type, Callable, Union, Set
from collections import deque
import time
import traceback

from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.common.async_executor import AsyncExecutor
from common.config import ConfigManager
from common.logging import LogManager
from strategy.base import BaseStrategy, StrategyFactory


class BaseBacktestEngine:
    """
    Base backtest engine with efficient factor-based data management
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize backtest engine
        
        Args:
            config: Configuration manager
            params: Backtest parameters
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"backtest.{self.__class__.__name__.lower()}")
        self.strategy = None
        self.executor = AsyncExecutor()  # Get singleton instance
        
        # Data management
        self.data_buffers = {}  # Symbol -> DataFrame
        self.factor_cache = {}  # Symbol -> {Factor -> Series}
        self.has_sufficient_history = {}  # Symbol -> bool
        
        # Use deque for efficient window management
        self.data_queues = {}  # Symbol -> deque of data points
        self.required_window_size = 0  # Will be determined from strategy
        
        # Performance tracking
        self.metrics = {
            'total_signals': 0,
            'processing_time': 0,
            'data_points_processed': 0
        }
        
        # State tracking
        self._is_initialized = False
        self._is_running = False
    
    async def initialize(self) -> None:
        """Initialize backtest engine and load strategy"""
        if self._is_initialized:
            return
            
        self.logger.info("Initializing backtest engine")
        
        # Start async executor
        await self.executor.start()
        
        # Load strategy
        await self._load_strategy()
        
        # Get required window size from strategy factors
        if self.strategy:
            self._update_required_window_size()
        
        self._is_initialized = True
        self.logger.info(f"Backtest initialization complete, required window size: {self.required_window_size}")
    
    def _update_required_window_size(self) -> None:
        """Determine required window size from strategy factors"""
        if not hasattr(self.strategy, '_factor_registry'):
            self.required_window_size = 30  # Default
            return
            
        # Find maximum window size from factors
        max_window = 0
        for name, info in self.strategy._factor_registry.items():
            window_size = info.get('window_size', 0)
            if info.get('is_differential', False):
                window_size += 1
            max_window = max(max_window, window_size)
            
            # Consider dependencies
            for dep_name in info.get('depends_on', []):
                if dep_name in self.strategy._factor_registry:
                    dep_window = self.strategy._factor_registry[dep_name].get('window_size', 0)
                    if self.strategy._factor_registry[dep_name].get('is_differential', False):
                        dep_window += 1
                    max_window = max(max_window, dep_window)
        
        # Set required window size (minimum 2)
        self.required_window_size = max(max_window, 2)
        
        self.logger.info(f"Required window size determined from factors: {self.required_window_size}")
    
    async def _load_strategy(self) -> None:
        """Load strategy from parameters"""
        strategy_name = self.params.get('strategy')
        if not strategy_name:
            raise ValueError("No strategy specified for backtest")
        
        strategy_params = self.params.get('strategy_params', {})
        
        try:
            # Get strategy factory
            factory = StrategyFactory.get_instance(self.config)
            
            # Create strategy
            self.strategy = await factory.create(strategy_name, strategy_params)
            
            # Initialize strategy
            await self.strategy.initialize()
            
            self.logger.info(f"Strategy '{strategy_name}' loaded and initialized")
        except Exception as e:
            self.logger.error(f"Failed to load strategy '{strategy_name}': {str(e)}")
            raise
    
    async def prepare_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Prepare and validate data for backtesting
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            bool: True if data is valid and sufficient
        """
        if not data:
            self.logger.warning("No data provided for backtest")
            return False
        
        # Initialize data structures
        self.data_buffers = {}
        self.factor_cache = {}
        self.has_sufficient_history = {}
        self.data_queues = {}
        
        # Process each symbol
        sufficient_data = True
        for symbol, df in data.items():
            # Validate data
            if df.empty:
                self.logger.warning(f"Empty data for {symbol}")
                sufficient_data = False
                continue
            
            # Initialize buffers
            self.data_buffers[symbol] = pd.DataFrame()
            self.factor_cache[symbol] = {}
            self.has_sufficient_history[symbol] = False
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
            
            # Preload data if sufficient history exists
            if len(df) >= self.required_window_size:
                # Take the most recent required_window_size rows for initial buffer
                initial_data = df.iloc[-self.required_window_size:]
                
                # Store in deque and dataframe
                for _, row in initial_data.iterrows():
                    self.data_queues[symbol].append(pd.DataFrame([row]))
                
                self.data_buffers[symbol] = initial_data.copy()
                self.has_sufficient_history[symbol] = True
                
                self.logger.info(f"Preloaded {len(initial_data)} data points for {symbol}")
            else:
                self.logger.warning(f"Insufficient data for {symbol}: {len(df)} < {self.required_window_size}")
                sufficient_data = False
        
        # Pre-calculate factors if strategy is available
        if self.strategy and sufficient_data:
            await self._precalculate_factors()
        
        return sufficient_data
    
    async def _precalculate_factors(self) -> None:
        """Precalculate factors for all symbols with sufficient history"""
        if not hasattr(self.strategy, '_factor_registry') or not self.strategy._factor_registry:
            return
            
        for symbol in self.has_sufficient_history:
            if not self.has_sufficient_history[symbol]:
                continue
                
            # Calculate all registered factors
            for factor_name in self.strategy._factor_registry:
                try:
                    factor_values = self.strategy.calculate_factor(
                        self.data_buffers[symbol], 
                        factor_name, 
                        symbol
                    )
                    
                    # Cache result
                    if symbol not in self.factor_cache:
                        self.factor_cache[symbol] = {}
                    self.factor_cache[symbol][factor_name] = factor_values
                    
                except Exception as e:
                    self.logger.error(f"Error calculating factor '{factor_name}' for {symbol}: {str(e)}")
        
        self.logger.info("Precalculated factors for all symbols with sufficient history")
    
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol
        
        Args:
            data_point: New data point
            symbol: Symbol
            
        Returns:
            DataFrame: Generated signals or empty DataFrame
        """
        if not self._is_initialized:
            await self.initialize()
        
        if not self.strategy:
            raise ValueError("No strategy loaded")
        
        try:
            # Add data point to buffer
            sufficient_history = await self._add_data_point(data_point, symbol)
            
            if not sufficient_history:
                self.logger.debug(f"Still collecting data for {symbol}, "
                                f"{len(self.data_queues.get(symbol, deque()))} of "
                                f"{self.required_window_size} required data points")
                return pd.DataFrame()  # Not enough data yet
            
            # Process data with strategy
            signals = await self.strategy.process_data(data_point, symbol)
            
            # Track metrics
            if not signals.empty:
                self.metrics['total_signals'] += len(signals)
            self.metrics['data_points_processed'] += 1
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error processing data point for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def _add_data_point(self, data_point: pd.DataFrame, symbol: str) -> bool:
        """
        Add a data point to the buffer
        
        Args:
            data_point: New data point
            symbol: Symbol
            
        Returns:
            bool: True if we have sufficient history
        """
        # Initialize buffers if needed
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame()
            self.factor_cache[symbol] = {}
            self.has_sufficient_history[symbol] = False
            self.data_queues[symbol] = deque(maxlen=self.required_window_size)
        
        # Add to queue
        self.data_queues[symbol].append(data_point)
        
        # Update DataFrame buffer
        self.data_buffers[symbol] = pd.concat(list(self.data_queues[symbol]))
        
        # Update status
        sufficient = len(self.data_queues[symbol]) >= self.required_window_size
        
        # If status changed, log it
        if sufficient and not self.has_sufficient_history[symbol]:
            self.has_sufficient_history[symbol] = True
            self.logger.info(f"Collected sufficient history for {symbol}: {len(self.data_queues[symbol])} data points")
        
        return self.has_sufficient_history[symbol]
    
    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run backtest on all data
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
        Returns:
            Dict: Backtest results
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self._is_running = True
        
        results = {
            'signals': {},
            'metrics': {}
        }
        
        try:
            # Prepare data
            sufficient_data = await self.prepare_data(data)
            if not sufficient_data:
                self.logger.warning("Insufficient data for backtest")
                return {
                    'error': 'Insufficient data for backtest',
                    'metrics': self.metrics
                }
            
            # Process each symbol
            for symbol, df in data.items():
                if symbol not in self.has_sufficient_history or not self.has_sufficient_history[symbol]:
                    continue
                
                signals = []
                
                # Process data points after initial window
                for i in range(self.required_window_size, len(df)):
                    if not self._is_running:
                        self.logger.info("Backtest stopped early")
                        break
                    
                    # Get data point
                    data_point = df.iloc[[i]]
                    
                    # Process data point
                    signal = await self.process_data_point(data_point, symbol)
                    
                    if not signal.empty:
                        signals.append(signal)
                    
                    # Allow asyncio to process other tasks
                    if i % 100 == 0:
                        await asyncio.sleep(0)
                
                # Combine signals
                if signals:
                    results['signals'][symbol] = pd.concat(signals)
                else:
                    results['signals'][symbol] = pd.DataFrame()
                    
                self.logger.info(f"Generated {len(results['signals'][symbol])} signals for {symbol}")
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.metrics['processing_time'] = execution_time
            
            results['metrics'] = self.metrics
            
            self.logger.info(f"Backtest completed in {execution_time:.2f}s")
            self.logger.info(f"Generated {self.metrics['total_signals']} signals in total")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during backtest: {str(e)}\n{traceback.format_exc()}")
            return {'error': str(e), 'metrics': self.metrics}
        finally:
            self._is_running = False
    
    def stop(self) -> None:
        """Stop backtest execution"""
        self._is_running = False
        self.logger.info("Stopping backtest")
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        if self.strategy and hasattr(self.strategy, 'shutdown'):
            await self.strategy.shutdown()
        
        # Clear buffers
        self.data_buffers.clear()
        self.factor_cache.clear()
        self.has_sufficient_history.clear()
        self.data_queues.clear()
        
        self._is_initialized = False
        self._is_running = False
        
        self.logger.info("Backtest engine shut down")


class BacktestFactory(AbstractFactory):
    """Factory for creating backtest engines"""
    
    def __init__(self, config):
        """Initialize backtest factory"""
        super().__init__(config)
        self.default_engine = config.get("backtest", "engine", default="standard")
        
        # Register default backtest engines
        self._register_default_engines()
    
    def _register_default_engines(self):
        """Register default backtest engines"""
        # Register standard backtest engine
        self.register("standard", "src.backtest.base_backtest.BaseBacktestEngine", {
            "description": "Standard Backtest Engine with factor-based data management"
        })
        
        # Register any additional engines
        try:
            self.register("ohlcv", "src.backtest.ohlcv_backtest_engine.OHLCVBacktestEngine", {
                "description": "OHLCV Backtest Engine"
            })
            self.register("market_replay", "src.backtest.market_replay_backtest_engine.MarketReplayBacktestEngine", {
                "description": "Market Replay Backtest Engine"
            })
        except Exception as e:
            self.logger.debug(f"Optional engines not registered: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseBacktestEngine]:
        """
        Get concrete backtest engine class
        
        Args:
            name: Engine name
            
        Returns:
            Type[BaseBacktestEngine]: Backtest engine class
        """
        return await self._load_class_from_path(name, BaseBacktestEngine)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        Resolve engine name with default fallback
        
        Args:
            name: Engine name
            
        Returns:
            str: Resolved engine name
        """
        name = name or self.default_engine
        if not name:
            raise ValueError("No engine name provided and no default in config")
        return name.lower()
    
    async def create_engine(self, engine_name: str, params: Optional[Dict[str, Any]] = None) -> BaseBacktestEngine:
        """
        Create and initialize backtest engine
        
        Args:
            engine_name: Engine name
            params: Engine parameters
            
        Returns:
            BaseBacktestEngine: Initialized backtest engine
        """
        engine = await self.create(engine_name, params)
        await engine.initialize()
        return engine


# Decorator for registering backtest engines
def register_backtest_engine(name: Optional[str] = None, **metadata):
    """
    Decorator for registering backtest engines
    
    Args:
        name: Engine name (defaults to class name without 'Engine' suffix)
        **metadata: Additional metadata
    """
    return register_factory_class('backtest_factory', name, **metadata)


# src/backtest/market_replay_backtest_engine.py

import pandas as pd
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import time
import traceback

from backtest.engine import BaseBacktestEngine, register_backtest_engine
from common.config import ConfigManager


@register_backtest_engine('market_replay', 
    description="Market Replay Backtest Engine for sequential data processing",
    category="backtest")
class MarketReplayBacktestEngine(BaseBacktestEngine):
    """
    Market Replay Backtest Engine
    
    Processes data sequentially, one data point at a time,
    simulating real-time market conditions with position tracking.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize market replay backtest engine"""
        super().__init__(config, params)
        
        # Market replay specific settings
        self.replay_speed = self.params.get('replay_speed', 0)  # 0 means as fast as possible
        self.initial_capital = self.params.get('initial_capital', 100000)
        self.commission_rate = self.params.get('commission_rate', 0.001)
        self.slippage = self.params.get('slippage', 0.001)
        
        # Portfolio state
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.portfolio_history = []  # Snapshots of portfolio value over time

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

    def _get_common_timeline(self, data: Dict[str, pd.DataFrame]) -> List:
        """
        Extract a common timeline from all data sources
        
        Args:
            data: Dictionary of symbol -> DataFrame
            
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
            data: Dictionary of symbol -> DataFrame
            timestamp: Timestamp to get data for
            
        Returns:
            Dict: Symbol -> DataFrame with data at timestamp
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
            signals: DataFrame with signals
            current_data: Current market data
            
        Returns:
            List: Trade records
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
            current_data: Current market data
            
        Returns:
            float: Portfolio value (cash + positions)
        """
        portfolio_value = self.cash
        
        # Add position values
        for symbol, quantity in self.positions.items():
            if symbol in current_data and not current_data[symbol].empty:
                price = current_data[symbol]['close'].iloc[0]
                position_value = quantity * price
                portfolio_value += position_value
        
        return portfolio_value
    
    def _calculate_advanced_metrics(self, equity_curve: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate advanced performance metrics
        
        Args:
            equity_curve: DataFrame with portfolio history
            
        Returns:
            Dict: Performance metrics
        """
        metrics = {}
        
        try:
            if 'portfolio_value' in equity_curve.columns:
                # Calculate returns
                equity_curve['return'] = equity_curve['portfolio_value'].pct_change()
                
                # Calculate drawdown
                equity_curve['peak'] = equity_curve['portfolio_value'].cummax()
                equity_curve['drawdown'] = (equity_curve['portfolio_value'] - equity_curve['peak']) / equity_curve['peak']
                
                # Maximum drawdown
                metrics['max_drawdown_pct'] = equity_curve['drawdown'].min() * 100 if not equity_curve['drawdown'].empty else 0
                
                # Sharpe ratio (assuming 252 trading days)
                if len(equity_curve) > 1:
                    avg_return = equity_curve['return'].mean()
                    std_return = equity_curve['return'].std()
                    if std_return > 0:
                        metrics['sharpe_ratio'] = (avg_return / std_return) * (252 ** 0.5)
                    else:
                        metrics['sharpe_ratio'] = 0
                
                # Win rate
                if self.trades:
                    winning_trades = [t for t in self.trades if 
                                    t['action'] == 'sell' and 
                                    t.get('value', 0) > t.get('cost', 0)]
                    metrics['win_rate'] = len(winning_trades) / len(self.trades) * 100
        
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {str(e)}")
        
        return metrics
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Reset portfolio state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []
        
        # Call parent shutdown
        await super().shutdown()
        

# src/backtest/ohlcv_backtest_engine.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import traceback

from backtest.engine import BaseBacktestEngine, register_backtest_engine
from common.config import ConfigManager


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