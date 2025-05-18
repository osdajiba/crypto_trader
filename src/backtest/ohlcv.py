# src/backtest/engine/ohlcv.py (Fixed Implementation)

from collections import deque
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import time
import asyncio

from src.common.abstract_factory import register_factory_class
from src.backtest.base import BaseBacktestEngine, BacktestEngineError


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
    on standard OHLCV (Open, High, Low, Close, Volume) data, with vectorized
    processing for maximum speed.
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
        
        # Execution settings
        self.commission_rate = self.params.get(
            'commission_rate', 
            self.config.get("trading", "fees", "commission_taker", default=0.005)
        )
        self.slippage = self.params.get(
            'slippage', 
            self.config.get("trading", "fees", "slippage", default=0.0001)
        )
        
        # Factor calculation cache
        self.factor_values = {}  # Symbol -> Factor -> Series
        
        # Default to vectorized mode
        self.use_vectorized = self.params.get('use_vectorized', True)
        
        self.logger.info(f"OHLCV Engine initialized with capital={self.initial_capital}, "
                        f"commission={self.commission_rate}, slippage={self.slippage}, "
                        f"vectorized={self.use_vectorized}")

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run vectorized backtest on OHLCV data
        
        This method implements the backtest execution on OHLCV data, with full
        vectorized processing for improved performance.
        
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
        
        # Check for portfolio manager
        if not self.portfolio:
            raise OHLCVEngineError("Portfolio manager not set, cannot run backtest")
            
        # Check for strategy
        if not self.strategy:
            raise OHLCVEngineError("Strategy not set, cannot run backtest")
        
        results = {
            'signals': {},
            'trades': [],
            'equity_curve': pd.DataFrame(),
            'metrics': {}
        }
        
        try:
            self.logger.info("Starting OHLCV backtest run")
            
            # Ensure all required symbols are in the portfolio
            await self._ensure_assets_in_portfolio(list(data.keys()))
            
            # Calculate all factors for each symbol in vectorized manner
            if self.use_vectorized:
                for symbol, df in data.items():
                    if not df.empty:
                        await self._calculate_all_factors(df, symbol)
            
            # Generate signals for all data at once (vectorized approach)
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
            
            # Create a timeline of all unique timestamps across all signals
            signal_timeline = self._create_signal_timeline(all_signals)
            
            # Process signals in chronological order to simulate realistic execution
            trades = []
            equity_history = []
            
            # Track initial portfolio value
            initial_value = self.portfolio.get_total_value()
            equity_history.append({
                'timestamp': signal_timeline[0] if signal_timeline else pd.Timestamp.now(),
                'portfolio_value': initial_value
            })
            
            # Process each timestamp of signals
            for timestamp in signal_timeline:
                # Get signals for this timestamp
                current_signals = self._get_signals_at_timestamp(all_signals, timestamp)
                
                if not current_signals.empty:
                    # Get market data at this timestamp for proper execution
                    market_data = self._get_data_at_timestamp(data, timestamp)
                    
                    # Execute signals through portfolio
                    executed_trades = await self.portfolio.process_signals(current_signals, market_data)
                    if executed_trades:
                        trades.extend(executed_trades)
                        symbols_traded = set(trade.get('symbol', '') for trade in executed_trades)
                        self.logger.debug(f"Executed {len(executed_trades)} trades for {', '.join(symbols_traded)} at {timestamp}")
    
                        # Update performance analyzer if available
                        if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                            for trade in executed_trades:
                                self.performance_analyzer.record_trade(trade)
                
                # Update all assets with the latest market data
                if market_data:
                    await self.portfolio.update_market_data(market_data)
                
                # Record portfolio value at this timestamp
                portfolio_value = self.portfolio.get_total_value()
                equity_history.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value
                })
                
                # Update performance analyzer with equity update
                if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                    self.performance_analyzer.update_equity(timestamp, portfolio_value)
            
            # Create equity curve DataFrame
            equity_curve = pd.DataFrame(equity_history)
            
            # Calculate metrics
            execution_time = time.time() - start_time
            self.metrics['processing_time'] = execution_time
            
            # Calculate final portfolio value
            final_value = self.portfolio.get_total_value()
            
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
                'symbols_traded': len(set(trade.get('symbol', '') for trade in trades)) if trades else 0
            })
            
            # Prepare final results
            results['signals'] = all_signals
            results['trades'] = trades
            results['equity_curve'] = equity_curve
            
            self.logger.info(f"OHLCV vectorized backtest completed in {execution_time:.2f}s with "
                           f"return {results['metrics']['total_return_pct']:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during OHLCV backtest: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise OHLCVEngineError(f"Backtest execution failed: {str(e)}")
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
    
    def _create_signal_timeline(self, all_signals: Dict[str, pd.DataFrame]) -> List:
        """
        Create a timeline of all unique timestamps from signals
        
        Args:
            all_signals: Dictionary of symbol -> signals DataFrame
            
        Returns:
            List: Sorted unique timestamps
        """
        all_timestamps = []
        
        for symbol, signals in all_signals.items():
            if signals.empty:
                continue
                
            # Get timestamp column - could be 'timestamp' or 'datetime'
            timestamp_col = None
            if 'timestamp' in signals.columns:
                timestamp_col = 'timestamp'
            elif 'datetime' in signals.columns:
                timestamp_col = 'datetime'
                
            if timestamp_col:
                all_timestamps.extend(signals[timestamp_col].tolist())
        
        # Sort and remove duplicates
        return sorted(set(all_timestamps))
    
    def _get_signals_at_timestamp(self, all_signals: Dict[str, pd.DataFrame], timestamp) -> pd.DataFrame:
        """
        Get all signals at a specific timestamp
        
        Args:
            all_signals: Dictionary of symbol -> signals DataFrame
            timestamp: Timestamp to get signals for
            
        Returns:
            pd.DataFrame: Combined signals at the timestamp
        """
        timestamp_signals = []
        
        for symbol, signals in all_signals.items():
            if signals.empty:
                continue
                
            # Check for timestamp or datetime column
            timestamp_col = None
            if 'timestamp' in signals.columns:
                timestamp_col = 'timestamp'
            elif 'datetime' in signals.columns:
                timestamp_col = 'datetime'
                
            if timestamp_col:
                mask = signals[timestamp_col] == timestamp
                if mask.any():
                    timestamp_signals.append(signals[mask])
        
        if timestamp_signals:
            return pd.concat(timestamp_signals)
        return pd.DataFrame()
    
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
                # Use strategy's vectorized method
                signals = await self.strategy.generate_signals_vectorized(
                    data, symbol, self.factor_values.get(symbol, {})
                )
                
                # Add symbol if not present
                if not signals.empty and 'symbol' not in signals.columns:
                    signals['symbol'] = symbol
                
                return signals
            
            # Enrich data with factor values
            enriched_data = data.copy()
            for factor_name, factor_values in self.factor_values.get(symbol, {}).items():
                factor_column = f"factor_{factor_name}"
                enriched_data[factor_column] = factor_values
            
            # Fall back to sequential processing if vectorized method not available
            return await self._generate_signals_sequential(enriched_data, symbol)
            
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
    
    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Process a single data point for a symbol.
        
        In the OHLCV Engine, this method is mainly used for sequential processing
        when vectorized processing is not chosen.
        
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
            try:
                # Try to use the strategy's process_data method first
                if hasattr(self.strategy, 'process_data') and callable(self.strategy.process_data):
                    signals = await self.strategy.process_data(self.data_buffers[symbol], symbol)
                # Fall back to _generate_signals if process_data is not available
                elif hasattr(self.strategy, '_generate_signals') and callable(self.strategy._generate_signals):
                    signals = await self.strategy._generate_signals(self.data_buffers[symbol])
                else:
                    self.logger.warning(f"Strategy doesn't have process_data or _generate_signals methods")
                    return pd.DataFrame()
                
                # Add symbol if not present
                if not signals.empty and 'symbol' not in signals.columns:
                    signals['symbol'] = symbol
                
                # Add timestamp if not present
                if not signals.empty and 'timestamp' not in signals.columns:
                    if 'timestamp' in data_point.columns:
                        signals['timestamp'] = data_point['timestamp'].iloc[0]
                    elif 'datetime' in data_point.columns:
                        signals['timestamp'] = data_point['datetime'].iloc[0]
                
                return signals
            
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}: {str(e)}")
        
        # No strategy or error occurred, no signals
        return pd.DataFrame()
    
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
        
        return result
    
    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear factor cache
        self.factor_values.clear()
        
        # Call parent shutdown
        await super().shutdown()