#!/usr/bin/env python3
# src/backtest/engines/ohlcv.py (Enhanced Vectorized Engine)

import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import warnings

from src.common.abstract_factory import register_factory_class
from src.backtest.base import BaseBacktestEngine, BacktestEngineError


@register_factory_class('backtest_engine_factory', "ohlcv", 
    description="Enhanced OHLCV Engine for vectorized factor generation and batch execution",
    features=["vectorized", "factor_computation", "batch_execution", "performance_optimized"],
    category="backtest")
class EnhancedOHLCVEngine(BaseBacktestEngine):
    """
    Enhanced OHLCV Engine specialized for:
    1. Vectorized factor computation across entire datasets
    2. Batch signal generation with optimized performance
    3. Efficient execution through portfolio integration
    4. Advanced caching and memory management
    
    This engine is optimized for speed and handles large datasets efficiently
    using vectorized operations and parallel processing where beneficial.
    """
    
    def __init__(self, config, params=None):
        """
        Initialize Enhanced OHLCV Engine
        
        Args:
            config: Configuration manager
            params: Engine parameters
        """
        super().__init__(config, params)
        
        # Core configuration
        self.initial_capital = self.params.get(
            'initial_capital', 
            self.config.get("trading", "capital", "initial", default=100000)
        )
        
        # Vectorization settings
        self.use_vectorized = self.params.get('use_vectorized', True)
        self.parallel_processing = self.params.get('parallel_processing', True)
        self.max_workers = self.params.get('max_workers', 4)
        self.chunk_size = self.params.get('chunk_size', 1000)
        
        # Factor computation cache with advanced management
        self.factor_cache = {}  # symbol -> factor_name -> values
        self.cache_metadata = {}  # symbol -> metadata
        self.computation_stats = {}
        
        # Performance optimization settings
        self.enable_jit_compilation = self.params.get('enable_jit_compilation', False)
        self.memory_efficient_mode = self.params.get('memory_efficient_mode', False)
        self.cache_management = self.params.get('cache_management', True)
        
        # Execution integration
        self.batch_execution = self.params.get('batch_execution', True)
        self.execution_optimization = self.params.get('execution_optimization', True)
        
        # Initialize thread pool for parallel operations
        if self.parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = None
        
        self.logger.info(f"Enhanced OHLCV Engine initialized: "
                        f"vectorized={self.use_vectorized}, "
                        f"parallel={self.parallel_processing}, "
                        f"capital={self.initial_capital}")

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run enhanced vectorized backtest with optimized factor computation
        
        Args:
            data: Dictionary of symbol -> DataFrame market data
                
        Returns:
            Dict[str, Any]: Comprehensive backtest results
        """
        if not self._is_initialized:
            await self.initialize()
        
        start_time = time.time()
        self._is_running = True
        
        # Validate prerequisites
        if not self.portfolio:
            raise BacktestEngineError("Portfolio manager not set")
        if not self.strategy:
            raise BacktestEngineError("Strategy not set")
        
        try:
            self.logger.info(f"Starting enhanced OHLCV backtest on {len(data)} symbols")
            
            # Performance tracking
            timing_stats = {}
            
            # 1. Data validation and preparation
            prep_start = time.time()
            validated_data = await self._validate_and_prepare_data(data)
            timing_stats['data_preparation'] = time.time() - prep_start
            
            # 2. Asset initialization
            asset_start = time.time()
            await self._ensure_assets_in_portfolio(list(validated_data.keys()))
            timing_stats['asset_initialization'] = time.time() - asset_start
            
            # 3. Vectorized factor computation
            factor_start = time.time()
            factor_results = await self._compute_all_factors_vectorized(validated_data)
            timing_stats['factor_computation'] = time.time() - factor_start
            
            # 4. Batch signal generation
            signal_start = time.time()
            signal_results = await self._generate_signals_batch(validated_data, factor_results)
            timing_stats['signal_generation'] = time.time() - signal_start
            
            # 5. Execution through portfolio manager
            execution_start = time.time()
            execution_results = await self._execute_signals_batch(signal_results, validated_data)
            timing_stats['execution'] = time.time() - execution_start
            
            # 6. Performance analysis and reporting
            analysis_start = time.time()
            performance_metrics = await self._analyze_performance(execution_results, validated_data)
            timing_stats['performance_analysis'] = time.time() - analysis_start
            
            # 7. Compile comprehensive results
            total_time = time.time() - start_time
            
            results = {
                'status': 'completed',
                'engine_type': 'enhanced_ohlcv',
                'strategy': self.strategy.__class__.__name__,
                'symbols': list(validated_data.keys()),
                'data_summary': self._get_data_summary(validated_data),
                'factor_results': factor_results,
                'signal_results': signal_results,
                'execution_results': execution_results,
                'performance_metrics': performance_metrics,
                'timing_stats': timing_stats,
                'computation_stats': self.computation_stats,
                'cache_stats': self._get_cache_statistics(),
                'total_execution_time': total_time
            }
            
            self.logger.info(f"Enhanced OHLCV backtest completed in {total_time:.2f}s "
                           f"with {performance_metrics.get('total_return_pct', 0):.2f}% return")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced OHLCV backtest failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise BacktestEngineError(f"Backtest execution failed: {str(e)}")
        finally:
            self._is_running = False

    async def _validate_and_prepare_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate and prepare data for vectorized processing
        
        Args:
            data: Raw market data
            
        Returns:
            Validated and prepared data
        """
        validated_data = {}
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for symbol, df in data.items():
            if df.empty:
                self.logger.warning(f"Empty data for {symbol}, skipping")
                continue
            
            # Check required columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                self.logger.warning(f"Missing columns for {symbol}: {missing_cols}")
                continue
            
            # Clean and validate data
            clean_df = df.copy()
            
            # Handle datetime column
            if 'timestamp' not in clean_df.columns and 'datetime' in clean_df.columns:
                clean_df['timestamp'] = pd.to_datetime(clean_df['datetime'])
            elif 'timestamp' in clean_df.columns:
                clean_df['timestamp'] = pd.to_datetime(clean_df['timestamp'])
            else:
                # Create timestamp index if missing
                clean_df['timestamp'] = pd.date_range(
                    start='2023-01-01', periods=len(clean_df), freq='1H'
                )
            
            # Validate numeric data
            for col in required_columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
            
            # Remove invalid rows
            clean_df = clean_df.dropna(subset=required_columns)
            
            # Sort by timestamp
            clean_df = clean_df.sort_values('timestamp').reset_index(drop=True)
            
            if len(clean_df) < 10:  # Minimum data requirement
                self.logger.warning(f"Insufficient data for {symbol}: {len(clean_df)} bars")
                continue
            
            validated_data[symbol] = clean_df
            self.logger.debug(f"Validated {symbol}: {len(clean_df)} bars")
        
        self.logger.info(f"Data validation completed: {len(validated_data)} valid symbols")
        return validated_data

    async def _compute_all_factors_vectorized(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Vectorized computation of all factors across all symbols
        
        Args:
            data: Validated market data
            
        Returns:
            Factor computation results
        """
        if not hasattr(self.strategy, '_factor_registry') or not self.strategy._factor_registry:
            self.logger.info("No factors registered in strategy, skipping factor computation")
            return {'factors_computed': 0, 'computation_time': 0}
        
        self.logger.info(f"Computing factors for {len(data)} symbols using vectorized approach")
        
        factor_results = {
            'factors_computed': 0,
            'symbols_processed': 0,
            'computation_time': 0,
            'factor_cache': {}
        }
        
        start_time = time.time()
        
        if self.parallel_processing and len(data) > 1:
            # Parallel processing for multiple symbols
            tasks = []
            for symbol, df in data.items():
                task = self._compute_symbol_factors_async(symbol, df)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                symbol = list(data.keys())[i]
                if isinstance(result, Exception):
                    self.logger.error(f"Factor computation failed for {symbol}: {result}")
                else:
                    factor_results['factor_cache'][symbol] = result
                    factor_results['symbols_processed'] += 1
        else:
            # Sequential processing
            for symbol, df in data.items():
                try:
                    symbol_factors = await self._compute_symbol_factors_async(symbol, df)
                    factor_results['factor_cache'][symbol] = symbol_factors
                    factor_results['symbols_processed'] += 1
                except Exception as e:
                    self.logger.error(f"Factor computation failed for {symbol}: {e}")
        
        factor_results['computation_time'] = time.time() - start_time
        factor_results['factors_computed'] = len(self.strategy._factor_registry)
        
        # Update global cache
        self.factor_cache.update(factor_results['factor_cache'])
        
        self.logger.info(f"Factor computation completed: "
                        f"{factor_results['symbols_processed']} symbols, "
                        f"{factor_results['factors_computed']} factors, "
                        f"{factor_results['computation_time']:.2f}s")
        
        return factor_results

    async def _compute_symbol_factors_async(self, symbol: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Asynchronously compute all factors for a single symbol
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Dictionary of factor_name -> factor_values
        """
        loop = asyncio.get_event_loop()
        
        if self.executor:
            # Run in thread pool for CPU-intensive operations
            return await loop.run_in_executor(
                self.executor, self._compute_symbol_factors_sync, symbol, data
            )
        else:
            return self._compute_symbol_factors_sync(symbol, data)

    def _compute_symbol_factors_sync(self, symbol: str, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Synchronously compute all factors for a symbol
        
        Args:
            symbol: Trading symbol
            data: OHLCV data
            
        Returns:
            Dictionary of factor_name -> factor_values
        """
        symbol_factors = {}
        
        # Suppress warnings during vectorized operations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for factor_name, factor_info in self.strategy._factor_registry.items():
                try:
                    start_time = time.time()
                    
                    # Compute dependencies first
                    dependencies = {}
                    for dep_name in factor_info.get('depends_on', []):
                        if dep_name in symbol_factors:
                            dependencies[dep_name] = symbol_factors[dep_name]
                        elif dep_name in self.factor_cache.get(symbol, {}):
                            dependencies[dep_name] = self.factor_cache[symbol][dep_name]
                    
                    # Compute the factor
                    if factor_info.get('func') and callable(factor_info['func']):
                        factor_values = factor_info['func'](data, **dependencies)
                        
                        if factor_values is not None and len(factor_values) > 0:
                            symbol_factors[factor_name] = factor_values
                            
                            # Track computation stats
                            computation_time = time.time() - start_time
                            if symbol not in self.computation_stats:
                                self.computation_stats[symbol] = {}
                            self.computation_stats[symbol][factor_name] = {
                                'computation_time': computation_time,
                                'data_points': len(factor_values),
                                'success': True
                            }
                        else:
                            self.logger.warning(f"Factor {factor_name} returned empty result for {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"Error computing factor {factor_name} for {symbol}: {e}")
                    if symbol not in self.computation_stats:
                        self.computation_stats[symbol] = {}
                    self.computation_stats[symbol][factor_name] = {
                        'success': False,
                        'error': str(e)
                    }
        
        return symbol_factors

    async def _generate_signals_batch(self, data: Dict[str, pd.DataFrame], 
                                    factor_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate signals in batch mode using vectorized approach
        
        Args:
            data: Market data
            factor_results: Computed factor results
            
        Returns:
            Signal generation results
        """
        self.logger.info("Generating signals using batch vectorized approach")
        
        signal_results = {
            'total_signals': 0,
            'signals_by_symbol': {},
            'generation_time': 0,
            'symbols_processed': 0
        }
        
        start_time = time.time()
        
        for symbol, df in data.items():
            try:
                symbol_factors = factor_results['factor_cache'].get(symbol, {})
                
                # Check if strategy has vectorized signal generation
                if hasattr(self.strategy, 'generate_signals_vectorized'):
                    signals = await self.strategy.generate_signals_vectorized(
                        df, symbol, symbol_factors
                    )
                else:
                    # Fallback to sequential generation
                    signals = await self._generate_signals_sequential(df, symbol, symbol_factors)
                
                if not signals.empty:
                    # Ensure required columns
                    if 'symbol' not in signals.columns:
                        signals['symbol'] = symbol
                    if 'timestamp' not in signals.columns and 'timestamp' in df.columns:
                        # Map signals to timestamps
                        signal_indices = signals.index
                        signals['timestamp'] = df.iloc[signal_indices]['timestamp'].values
                    
                    signal_results['signals_by_symbol'][symbol] = signals
                    signal_results['total_signals'] += len(signals)
                    signal_results['symbols_processed'] += 1
                    
                    self.logger.debug(f"Generated {len(signals)} signals for {symbol}")
                
            except Exception as e:
                self.logger.error(f"Signal generation failed for {symbol}: {e}")
        
        signal_results['generation_time'] = time.time() - start_time
        
        self.logger.info(f"Signal generation completed: "
                        f"{signal_results['total_signals']} total signals, "
                        f"{signal_results['symbols_processed']} symbols, "
                        f"{signal_results['generation_time']:.2f}s")
        
        return signal_results

    async def _generate_signals_sequential(self, data: pd.DataFrame, symbol: str, 
                                         factors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Fallback sequential signal generation
        
        Args:
            data: Market data for symbol
            symbol: Trading symbol
            factors: Computed factors
            
        Returns:
            Generated signals DataFrame
        """
        signals_list = []
        
        # Create enriched data with factors
        enriched_data = data.copy()
        for factor_name, factor_values in factors.items():
            enriched_data[f'factor_{factor_name}'] = factor_values
        
        # Process data sequentially using existing method
        for i in range(len(enriched_data)):
            try:
                data_point = enriched_data.iloc[[i]]
                signal = await self.process_data_point(data_point, symbol)
                if not signal.empty:
                    signals_list.append(signal)
            except Exception as e:
                self.logger.debug(f"Error processing data point {i} for {symbol}: {e}")
        
        if signals_list:
            return pd.concat(signals_list, ignore_index=True)
        return pd.DataFrame()

    async def _execute_signals_batch(self, signal_results: Dict[str, Any], 
                                data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Execute signals through portfolio manager with batch optimization
        Fixed: Ensure historical data is properly passed to execution engine
        """
        if signal_results['total_signals'] == 0:
            return {'successful': 0, 'failed': 0, 'orders': []}

        self.logger.info(f"Executing {signal_results['total_signals']} signals through portfolio manager")

        # Combine all signals into single DataFrame
        all_signals = []
        for symbol, signals in signal_results['signals_by_symbol'].items():
            all_signals.append(signals)

        combined_signals = pd.concat(all_signals, ignore_index=True)

        # FIXED: Ensure execution engine has historical data before processing signals
        if (hasattr(self.portfolio, 'execution_engine') and 
            hasattr(self.portfolio.execution_engine, 'set_historical_data')):
            
            # Validate data format before setting
            validated_data = {}
            for symbol, df in data.items():
                if not df.empty and all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
                    # Ensure timestamp column exists
                    if 'timestamp' not in df.columns:
                        if isinstance(df.index, pd.DatetimeIndex):
                            df = df.reset_index()
                            df.rename(columns={'index': 'timestamp'}, inplace=True)
                        else:
                            df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
                    
                    validated_data[symbol] = df
                    self.logger.debug(f"Set historical data for {symbol}: {len(df)} bars")
            
            if validated_data:
                self.portfolio.execution_engine.set_historical_data(validated_data)
                self.logger.info(f"Historical data set for execution engine: {len(validated_data)} symbols")
            else:
                self.logger.warning("No valid historical data to set for execution engine")

        # Execute through portfolio manager
        execution_results = await self.portfolio.process_signals(combined_signals, data)

        self.logger.info(f"Execution completed: "
                        f"{execution_results.get('successful', 0)} successful, "
                        f"{execution_results.get('failed', 0)} failed")

        return execution_results

    async def _analyze_performance(self, execution_results: Dict[str, Any], 
                                 data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze backtest performance with comprehensive metrics
        
        Args:
            execution_results: Execution results
            data: Market data for baseline calculation
            
        Returns:
            Performance metrics
        """
        # Get final portfolio value
        final_value = await self.portfolio.get_total_value()
        
        # Calculate basic metrics
        metrics = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': final_value - self.initial_capital,
            'total_return_pct': ((final_value / self.initial_capital) - 1) * 100,
            'total_trades': execution_results.get('successful', 0),
            'failed_trades': execution_results.get('failed', 0)
        }
        
        # Calculate baseline (buy and hold) performance
        main_symbol = list(data.keys())[0] if data else None
        if main_symbol and not data[main_symbol].empty:
            baseline = self._calculate_baseline_performance(data[main_symbol])
            metrics.update(baseline)
            
            # Calculate alpha (outperformance vs baseline)
            if 'baseline_return_pct' in baseline:
                metrics['alpha'] = metrics['total_return_pct'] - baseline['baseline_return_pct']
        
        # Get portfolio summary
        if hasattr(self.portfolio, 'get_portfolio_summary'):
            portfolio_summary = await self.portfolio.get_portfolio_summary()
            metrics['portfolio_summary'] = portfolio_summary
        
        return metrics

    def _calculate_baseline_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate buy-and-hold baseline performance"""
        try:
            initial_price = data['close'].iloc[0]
            final_price = data['close'].iloc[-1]
            
            baseline_return = (final_price - initial_price) / initial_price
            baseline_return_pct = baseline_return * 100
            
            return {
                'baseline_initial_price': initial_price,
                'baseline_final_price': final_price,
                'baseline_return_pct': baseline_return_pct
            }
        except Exception as e:
            self.logger.error(f"Error calculating baseline performance: {e}")
            return {}

    def _get_data_summary(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        summary = {
            'total_symbols': len(data),
            'total_bars': sum(len(df) for df in data.values()),
            'date_range': {},
            'symbols': list(data.keys())
        }
        
        if data:
            all_timestamps = []
            for df in data.values():
                if 'timestamp' in df.columns:
                    all_timestamps.extend(df['timestamp'].tolist())
            
            if all_timestamps:
                summary['date_range'] = {
                    'start': min(all_timestamps),
                    'end': max(all_timestamps)
                }
        
        return summary

    def _get_cache_statistics(self) -> Dict[str, Any]:
        """Get factor cache statistics"""
        cache_stats = {
            'symbols_cached': len(self.factor_cache),
            'total_factors_cached': 0,
            'cache_size_mb': 0,
            'cache_hit_rate': 0
        }
        
        for symbol_factors in self.factor_cache.values():
            cache_stats['total_factors_cached'] += len(symbol_factors)
        
        # Estimate cache size (rough approximation)
        cache_stats['cache_size_mb'] = cache_stats['total_factors_cached'] * 0.1  # MB estimate
        
        return cache_stats

    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process single data point (maintained for compatibility)"""
        if symbol not in self.data_buffers:
            self.data_buffers[symbol] = pd.DataFrame()
            self.has_sufficient_history[symbol] = False
        
        if data_point.empty:
            return pd.DataFrame()
        
        # Maintain data buffer
        self.data_buffers[symbol] = pd.concat([self.data_buffers[symbol], data_point])
        
        if self.required_window_size > 0 and len(self.data_buffers[symbol]) > self.required_window_size:
            self.data_buffers[symbol] = self.data_buffers[symbol].tail(self.required_window_size)
        
        # Check sufficient history
        if self.required_window_size > 0:
            self.has_sufficient_history[symbol] = len(self.data_buffers[symbol]) >= self.required_window_size
        else:
            self.has_sufficient_history[symbol] = True
        
        if not self.has_sufficient_history[symbol]:
            return pd.DataFrame()
        
        # Generate signals using strategy
        if self.strategy:
            try:
                signals = await self.strategy.process_data(self.data_buffers[symbol], symbol)
                
                if not signals.empty:
                    # Ensure proper signal format
                    if 'timestamp' not in signals.columns:
                        if 'timestamp' in data_point.columns:
                            signals['timestamp'] = data_point['timestamp'].iloc[0]
                    
                    if 'symbol' not in signals.columns:
                        signals['symbol'] = symbol
                
                return signals
            except Exception as e:
                self.logger.error(f"Error processing data point for {symbol}: {e}")
        
        return pd.DataFrame()

    async def _ensure_assets_in_portfolio(self, symbols: List[str]) -> None:
        """Ensure all symbols exist as assets in portfolio"""
        if not self.portfolio:
            return
            
        existing_assets = self.portfolio.list_assets()
        
        for symbol in symbols:
            if symbol not in existing_assets:
                self.logger.info(f"Adding {symbol} to portfolio")
                
                asset_params = {
                    'name': symbol,
                    'symbol': symbol,
                    'type': 'spot',
                    'tradable': True,
                    'quantity': 0.0,
                    'price': 100.0
                }
                
                try:
                    if hasattr(self.portfolio, 'asset_factory'):
                        asset = await self.portfolio.asset_factory.create_asset(asset_params)
                        await self.portfolio.add_asset(asset)
                except Exception as e:
                    self.logger.error(f"Error adding {symbol} to portfolio: {e}")

    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clear caches
        self.factor_cache.clear()
        self.cache_metadata.clear()
        self.computation_stats.clear()
        
        # Close thread pool
        if self.executor:
            self.executor.shutdown(wait=True)
        
        await super().shutdown()

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced engine metrics"""
        metrics = super().get_metrics()
        
        metrics.update({
            'vectorized_mode': self.use_vectorized,
            'parallel_processing': self.parallel_processing,
            'cache_statistics': self._get_cache_statistics(),
            'computation_statistics': self.computation_stats,
            'memory_efficient_mode': self.memory_efficient_mode
        })
        
        return metrics