#!/usr/bin/env python3
# src/backtest/engines/market_replay.py (Enhanced with Orderbook & Tick Support)

import asyncio
from collections import deque, OrderedDict
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import time
from tqdm import tqdm
from dataclasses import dataclass
from enum import Enum
import heapq

from src.common.abstract_factory import register_factory_class
from src.backtest.base import BaseBacktestEngine, BacktestEngineError


class DataType(Enum):
    """Types of market data supported"""
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDERBOOK = "orderbook"
    TRADE = "trade"


@dataclass
class TickData:
    """Tick data structure"""
    timestamp: pd.Timestamp
    symbol: str
    price: float
    size: float
    side: str  # 'buy' or 'sell'
    trade_id: Optional[str] = None


@dataclass
class OrderBookLevel:
    """Order book level structure"""
    price: float
    size: float
    count: Optional[int] = None


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""
    timestamp: pd.Timestamp
    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    sequence: Optional[int] = None


class AdvancedOrderBookEngine:
    """Advanced order book simulation engine"""
    
    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.orderbooks = {}  # symbol -> current orderbook
        self.last_trades = {}  # symbol -> last trade info
        
    def update_orderbook(self, snapshot: OrderBookSnapshot):
        """Update order book with new snapshot"""
        self.orderbooks[snapshot.symbol] = {
            'timestamp': snapshot.timestamp,
            'bids': sorted(snapshot.bids, key=lambda x: x.price, reverse=True)[:self.max_depth],
            'asks': sorted(snapshot.asks, key=lambda x: x.price)[:self.max_depth],
            'sequence': snapshot.sequence
        }
    
    def get_best_bid_ask(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get best bid and ask prices"""
        if symbol not in self.orderbooks:
            return None, None
            
        book = self.orderbooks[symbol]
        best_bid = book['bids'][0].price if book['bids'] else None
        best_ask = book['asks'][0].price if book['asks'] else None
        
        return best_bid, best_ask
    
    def get_market_depth(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        """Get market depth information"""
        if symbol not in self.orderbooks:
            return {'bids': [], 'asks': [], 'spread': None}
            
        book = self.orderbooks[symbol]
        bids = book['bids'][:depth]
        asks = book['asks'][:depth]
        
        spread = None
        if bids and asks:
            spread = asks[0].price - bids[0].price
            
        return {
            'bids': [(level.price, level.size) for level in bids],
            'asks': [(level.price, level.size) for level in asks],
            'spread': spread,
            'timestamp': book['timestamp']
        }
    
    def simulate_order_matching(self, symbol: str, side: str, quantity: float, 
                               order_type: str = 'market') -> Dict[str, Any]:
        """Simulate order matching against order book"""
        if symbol not in self.orderbooks:
            return {'success': False, 'reason': 'No orderbook data'}
            
        book = self.orderbooks[symbol]
        
        if side.lower() == 'buy':
            # Buy order matches against asks
            levels = book['asks']
            is_aggressive = True
        else:
            # Sell order matches against bids
            levels = book['bids']
            is_aggressive = True
            
        if not levels:
            return {'success': False, 'reason': 'No liquidity available'}
        
        fills = []
        remaining_qty = quantity
        total_cost = 0
        
        for level in levels:
            if remaining_qty <= 0:
                break
                
            available_qty = min(level.size, remaining_qty)
            fill_price = level.price
            
            # Apply market impact based on order size
            impact_factor = min(available_qty / level.size, 1.0)
            if impact_factor > 0.5:  # Large order, apply price impact
                if side.lower() == 'buy':
                    fill_price *= (1 + impact_factor * 0.0001)  # 0.01% impact
                else:
                    fill_price *= (1 - impact_factor * 0.0001)
            
            fills.append({
                'price': fill_price,
                'quantity': available_qty,
                'level': level.price,
                'impact': impact_factor
            })
            
            total_cost += fill_price * available_qty
            remaining_qty -= available_qty
        
        avg_price = total_cost / (quantity - remaining_qty) if quantity > remaining_qty else 0
        
        return {
            'success': remaining_qty < quantity,
            'filled_quantity': quantity - remaining_qty,
            'remaining_quantity': remaining_qty,
            'avg_price': avg_price,
            'fills': fills,
            'total_cost': total_cost
        }


@register_factory_class('backtest_engine_factory', "market_replay", 
    description="Enhanced Market Replay Engine with orderbook and tick data support",
    features=["real_time_simulation", "orderbook_matching", "tick_data", "market_microstructure"],
    category="backtest")
class EnhancedMarketReplayEngine(BaseBacktestEngine):
    """
    Enhanced Market Replay Engine that supports:
    1. Real-time sequential data processing
    2. Order book simulation and matching
    3. Tick-by-tick data processing
    4. Advanced market microstructure modeling
    5. Realistic execution with slippage and market impact
    
    This engine provides the most realistic trading simulation by processing
    data in chronological order with full market depth information.
    """
    
    def __init__(self, config, params=None):
        """
        Initialize Enhanced Market Replay Engine
        
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
        
        # Market replay specific settings
        self.replay_speed = self.params.get('replay_speed', 0)  # 0 = max speed
        self.real_time_mode = self.params.get('real_time_mode', False)
        self.market_hours_only = self.params.get('market_hours_only', False)
        
        # Advanced features
        self.enable_orderbook = self.params.get('enable_orderbook', True)
        self.enable_tick_data = self.params.get('enable_tick_data', True)
        self.orderbook_depth = self.params.get('orderbook_depth', 20)
        self.tick_aggregation_ms = self.params.get('tick_aggregation_ms', 100)
        
        # Market microstructure settings
        self.enable_market_impact = self.params.get('enable_market_impact', True)
        self.latency_simulation = self.params.get('latency_simulation', True)
        self.latency_ms = self.params.get('latency_ms', 5)
        self.slippage_model = self.params.get('slippage_model', 'advanced')
        
        # Data processing
        self.data_types = set()
        self.data_streams = {}  # data_type -> symbol -> data
        self.event_queue = []  # Priority queue for chronological processing
        
        # Market simulation components
        self.orderbook_engine = AdvancedOrderBookEngine(self.orderbook_depth)
        self.tick_processor = TickDataProcessor(self.tick_aggregation_ms)
        self.execution_simulator = AdvancedExecutionSimulator(
            enable_impact=self.enable_market_impact,
            latency_ms=self.latency_ms
        )
        
        # State tracking
        self.current_time = None
        self.market_state = {}  # symbol -> current market state
        self.execution_log = []
        self.performance_metrics = {}
        
        self.logger.info(f"Enhanced Market Replay Engine initialized: "
                        f"orderbook={self.enable_orderbook}, "
                        f"tick_data={self.enable_tick_data}, "
                        f"latency={self.latency_ms}ms")

    async def run_backtest(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run enhanced market replay backtest with multi-data support
        
        Args:
            data: Dictionary containing various data types:
                  - 'ohlcv': Traditional OHLCV data
                  - 'orderbook': Order book snapshots
                  - 'ticks': Tick data
                  - 'trades': Trade data
                
        Returns:
            Comprehensive backtest results with microstructure analysis
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
            self.logger.info(f"Starting enhanced market replay backtest")
            
            # 1. Prepare and validate multi-stream data
            preparation_start = time.time()
            prepared_data = await self._prepare_multi_stream_data(data)
            preparation_time = time.time() - preparation_start
            
            # 2. Initialize assets and market state
            initialization_start = time.time()
            await self._initialize_market_simulation(prepared_data)
            initialization_time = time.time() - initialization_start
            
            # 3. Build chronological event timeline
            timeline_start = time.time()
            event_timeline = self._build_event_timeline(prepared_data)
            timeline_time = time.time() - timeline_start
            
            # 4. Execute market replay simulation
            simulation_start = time.time()
            simulation_results = await self._execute_market_replay(event_timeline)
            simulation_time = time.time() - simulation_start
            
            # 5. Analyze performance and microstructure
            analysis_start = time.time()
            analysis_results = await self._analyze_microstructure_performance(simulation_results)
            analysis_time = time.time() - analysis_start
            
            # 6. Compile comprehensive results
            total_time = time.time() - start_time
            
            results = {
                'status': 'completed',
                'engine_type': 'enhanced_market_replay',
                'strategy': self.strategy.__class__.__name__,
                'simulation_results': simulation_results,
                'microstructure_analysis': analysis_results,
                'data_summary': self._get_data_stream_summary(prepared_data),
                'execution_log': self.execution_log,
                'performance_metrics': self.performance_metrics,
                'timing_breakdown': {
                    'data_preparation': preparation_time,
                    'initialization': initialization_time,
                    'timeline_building': timeline_time,
                    'simulation': simulation_time,
                    'analysis': analysis_time,
                    'total': total_time
                },
                'market_state_final': self.market_state.copy()
            }
            
            final_value = await self.portfolio.get_total_value()
            return_pct = ((final_value / self.initial_capital) - 1) * 100
            
            self.logger.info(f"Enhanced market replay completed in {total_time:.2f}s "
                           f"with {return_pct:.2f}% return")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Enhanced market replay failed: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise BacktestEngineError(f"Market replay execution failed: {str(e)}")
        finally:
            self._is_running = False

    async def _prepare_multi_stream_data(self, data: Dict[str, Any]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Prepare and validate multi-stream market data
        
        Args:
            data: Raw data containing different data types
            
        Returns:
            Organized data streams by type and symbol
        """
        prepared_streams = {
            DataType.OHLCV.value: {},
            DataType.TICK.value: {},
            DataType.ORDERBOOK.value: {},
            DataType.TRADE.value: {}
        }
        
        # Handle different input formats
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower() in ['ohlcv', 'bars', 'candles']:
                    prepared_streams[DataType.OHLCV.value] = self._prepare_ohlcv_data(value)
                elif key.lower() in ['orderbook', 'book', 'depth']:
                    prepared_streams[DataType.ORDERBOOK.value] = self._prepare_orderbook_data(value)
                elif key.lower() in ['ticks', 'tick_data', 'trades']:
                    prepared_streams[DataType.TICK.value] = self._prepare_tick_data(value)
                elif isinstance(value, pd.DataFrame):
                    # Try to auto-detect data type
                    detected_type = self._detect_data_type(value)
                    if detected_type:
                        if detected_type == DataType.OHLCV:
                            prepared_streams[DataType.OHLCV.value][key] = value
                        elif detected_type == DataType.TICK:
                            prepared_streams[DataType.TICK.value][key] = value
                        elif detected_type == DataType.ORDERBOOK:
                            prepared_streams[DataType.ORDERBOOK.value][key] = value
        
        # Remove empty streams
        prepared_streams = {k: v for k, v in prepared_streams.items() if v}
        
        # Log data summary
        for data_type, streams in prepared_streams.items():
            self.logger.info(f"Prepared {data_type} data: {len(streams)} symbols")
            self.data_types.add(data_type)
        
        return prepared_streams

    def _prepare_ohlcv_data(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare OHLCV data streams"""
        if isinstance(data, pd.DataFrame):
            # Single symbol data
            return {'default': self._validate_ohlcv_dataframe(data)}
        elif isinstance(data, dict):
            # Multi-symbol data
            validated_data = {}
            for symbol, df in data.items():
                validated_df = self._validate_ohlcv_dataframe(df)
                if not validated_df.empty:
                    validated_data[symbol] = validated_df
            return validated_data
        return {}

    def _prepare_orderbook_data(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare orderbook data streams"""
        if isinstance(data, pd.DataFrame):
            return {'default': self._validate_orderbook_dataframe(data)}
        elif isinstance(data, dict):
            validated_data = {}
            for symbol, df in data.items():
                validated_df = self._validate_orderbook_dataframe(df)
                if not validated_df.empty:
                    validated_data[symbol] = validated_df
            return validated_data
        return {}

    def _prepare_tick_data(self, data: Union[Dict[str, pd.DataFrame], pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare tick data streams"""
        if isinstance(data, pd.DataFrame):
            return {'default': self._validate_tick_dataframe(data)}
        elif isinstance(data, dict):
            validated_data = {}
            for symbol, df in data.items():
                validated_df = self._validate_tick_dataframe(df)
                if not validated_df.empty:
                    validated_data[symbol] = validated_df
            return validated_data
        return {}

    def _validate_ohlcv_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV DataFrame format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if df.empty:
            return df
        
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing OHLCV columns: {missing_cols}")
            return pd.DataFrame()
        
        # Ensure timestamp column
        if 'timestamp' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'timestamp'}, inplace=True)
            else:
                # Create timestamp if missing
                df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='1H')
        
        # Convert and validate data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid rows
        df = df.dropna(subset=required_columns)
        
        return df.sort_values('timestamp').reset_index(drop=True)

    def _validate_orderbook_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate orderbook DataFrame format"""
        required_columns = ['timestamp', 'bid_price', 'bid_size', 'ask_price', 'ask_size']
        
        if df.empty:
            return df
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing orderbook columns: {missing_cols}")
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df.sort_values('timestamp').reset_index(drop=True)

    def _validate_tick_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate tick DataFrame format"""
        required_columns = ['timestamp', 'price', 'size']
        
        if df.empty:
            return df
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing tick columns: {missing_cols}")
            return pd.DataFrame()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add side column if missing
        if 'side' not in df.columns:
            df['side'] = 'unknown'
        
        return df.sort_values('timestamp').reset_index(drop=True)

    def _detect_data_type(self, df: pd.DataFrame) -> Optional[DataType]:
        """Auto-detect data type from DataFrame columns"""
        columns = set(df.columns.str.lower())
        
        # Check for OHLCV
        ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}
        if ohlcv_cols.issubset(columns):
            return DataType.OHLCV
        
        # Check for orderbook
        book_cols = {'bid_price', 'ask_price', 'bid_size', 'ask_size'}
        if book_cols.issubset(columns) or 'bids' in columns or 'asks' in columns:
            return DataType.ORDERBOOK
        
        # Check for tick data
        tick_cols = {'price', 'size'}
        if tick_cols.issubset(columns):
            return DataType.TICK
        
        return None

    async def _initialize_market_simulation(self, prepared_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Initialize market simulation components"""
        all_symbols = set()
        
        # Collect all symbols from all data streams
        for data_type, streams in prepared_data.items():
            all_symbols.update(streams.keys())
        
        # Initialize market state for each symbol
        for symbol in all_symbols:
            self.market_state[symbol] = {
                'last_price': 0,
                'last_volume': 0,
                'best_bid': None,
                'best_ask': None,
                'last_trade_time': None,
                'daily_volume': 0,
                'daily_high': 0,
                'daily_low': float('inf')
            }
        
        # Ensure assets exist in portfolio
        await self._ensure_assets_in_portfolio(list(all_symbols))
        
        self.logger.info(f"Initialized market simulation for {len(all_symbols)} symbols")

    def _build_event_timeline(self, prepared_data: Dict[str, Dict[str, pd.DataFrame]]) -> List[Dict[str, Any]]:
        """Build chronological event timeline from all data streams"""
        events = []
        
        # Process each data type and symbol
        for data_type, streams in prepared_data.items():
            for symbol, df in streams.items():
                for _, row in df.iterrows():
                    event = {
                        'timestamp': row['timestamp'],
                        'data_type': data_type,
                        'symbol': symbol,
                        'data': row.to_dict()
                    }
                    events.append(event)
        
        # Sort all events chronologically
        events.sort(key=lambda x: x['timestamp'])
        
        self.logger.info(f"Built event timeline with {len(events)} events")
        return events

    async def _execute_market_replay(self, event_timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the market replay simulation"""
        simulation_results = {
            'events_processed': 0,
            'signals_generated': 0,
            'orders_executed': 0,
            'execution_details': [],
            'market_updates': []
        }
        
        # Set up progress tracking
        total_events = len(event_timeline)
        progress_bar = tqdm(total=total_events, desc="Market Replay") if total_events > 1000 else None
        
        try:
            for i, event in enumerate(event_timeline):
                if not self._is_running:
                    break
                
                # Update current time
                self.current_time = event['timestamp']
                
                # Process event based on data type
                await self._process_market_event(event, simulation_results)
                
                # Update progress
                simulation_results['events_processed'] += 1
                if progress_bar and i % 100 == 0:
                    progress_bar.update(100)
                
                # Simulate latency if enabled
                if self.latency_simulation and self.replay_speed > 0:
                    await asyncio.sleep(self.replay_speed / 1000)
            
            if progress_bar:
                progress_bar.close()
            
            self.logger.info(f"Market replay completed: {simulation_results['events_processed']} events processed")
            return simulation_results
            
        except Exception as e:
            if progress_bar:
                progress_bar.close()
            raise e

    async def _process_market_event(self, event: Dict[str, Any], results: Dict[str, Any]):
        """Process a single market event"""
        data_type = event['data_type']
        symbol = event['symbol']
        data = event['data']
        timestamp = event['timestamp']
        
        try:
            if data_type == DataType.OHLCV.value:
                await self._process_ohlcv_event(symbol, data, timestamp, results)
            elif data_type == DataType.ORDERBOOK.value:
                await self._process_orderbook_event(symbol, data, timestamp, results)
            elif data_type == DataType.TICK.value:
                await self._process_tick_event(symbol, data, timestamp, results)
            
            # Update market state
            self._update_market_state(symbol, data_type, data)
            
            # Check for signal generation opportunities
            signals = await self._check_signal_generation(symbol, timestamp)
            if signals and not signals.empty:
                results['signals_generated'] += len(signals)
                
                # Execute signals through portfolio
                execution_result = await self.portfolio.process_signals(signals)
                if execution_result and execution_result.get('successful', 0) > 0:
                    results['orders_executed'] += execution_result['successful']
                    results['execution_details'].append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'signals': len(signals),
                        'executions': execution_result['successful']
                    })
            
        except Exception as e:
            self.logger.error(f"Error processing event for {symbol} at {timestamp}: {e}")

    async def _process_ohlcv_event(self, symbol: str, data: Dict[str, Any], 
                                 timestamp: pd.Timestamp, results: Dict[str, Any]):
        """Process OHLCV data event"""
        # Update market state with OHLCV data
        self.market_state[symbol].update({
            'last_price': data['close'],
            'last_volume': data['volume'],
            'daily_high': max(self.market_state[symbol]['daily_high'], data['high']),
            'daily_low': min(self.market_state[symbol]['daily_low'], data['low']),
            'last_trade_time': timestamp
        })
        
        # Update portfolio with market data if asset exists
        asset = self._find_asset_by_symbol(symbol)
        if asset:
            # Create single-row DataFrame for asset update
            market_data = pd.DataFrame([data])
            market_data['timestamp'] = timestamp
            await asset.update_data(market_data)

    async def _process_orderbook_event(self, symbol: str, data: Dict[str, Any], 
                                     timestamp: pd.Timestamp, results: Dict[str, Any]):
        """Process order book update event"""
        if not self.enable_orderbook:
            return
        
        # Create orderbook snapshot
        bids = []
        asks = []
        
        # Handle different orderbook formats
        if 'bid_price' in data and 'ask_price' in data:
            # Single level orderbook
            bids.append(OrderBookLevel(data['bid_price'], data['bid_size']))
            asks.append(OrderBookLevel(data['ask_price'], data['ask_size']))
        elif 'bids' in data and 'asks' in data:
            # Multi-level orderbook
            bids = [OrderBookLevel(price, size) for price, size in data['bids']]
            asks = [OrderBookLevel(price, size) for price, size in data['asks']]
        
        if bids and asks:
            snapshot = OrderBookSnapshot(timestamp, symbol, bids, asks)
            self.orderbook_engine.update_orderbook(snapshot)
            
            # Update market state
            best_bid, best_ask = self.orderbook_engine.get_best_bid_ask(symbol)
            self.market_state[symbol].update({
                'best_bid': best_bid,
                'best_ask': best_ask,
                'last_trade_time': timestamp
            })

    async def _process_tick_event(self, symbol: str, data: Dict[str, Any], 
                                timestamp: pd.Timestamp, results: Dict[str, Any]):
        """Process tick data event"""
        if not self.enable_tick_data:
            return
        
        # Create tick data object
        tick = TickData(
            timestamp=timestamp,
            symbol=symbol,
            price=data['price'],
            size=data['size'],
            side=data.get('side', 'unknown'),
            trade_id=data.get('trade_id')
        )
        
        # Process through tick processor
        self.tick_processor.process_tick(tick)
        
        # Update market state
        self.market_state[symbol].update({
            'last_price': tick.price,
            'last_volume': tick.size,
            'last_trade_time': timestamp,
            'daily_volume': self.market_state[symbol]['daily_volume'] + tick.size
        })

    def _update_market_state(self, symbol: str, data_type: str, data: Dict[str, Any]):
        """Update market state with new data"""
        if symbol not in self.market_state:
            self.market_state[symbol] = {
                'last_price': 0,
                'last_volume': 0,
                'best_bid': None,
                'best_ask': None,
                'last_trade_time': None,
                'daily_volume': 0,
                'daily_high': 0,
                'daily_low': float('inf')
            }
        
        # Update based on data type
        if 'price' in data:
            self.market_state[symbol]['last_price'] = data['price']
        
        if 'volume' in data or 'size' in data:
            volume = data.get('volume', data.get('size', 0))
            self.market_state[symbol]['last_volume'] = volume
            self.market_state[symbol]['daily_volume'] += volume

    async def _check_signal_generation(self, symbol: str, timestamp: pd.Timestamp) -> Optional[pd.DataFrame]:
        """Check if conditions are met for signal generation"""
        if not self.strategy:
            return None
        
        # Get sufficient data for strategy
        asset = self._find_asset_by_symbol(symbol)
        if not asset:
            return None
        
        # Create market data point for strategy
        market_state = self.market_state[symbol]
        data_point = pd.DataFrame([{
            'timestamp': timestamp,
            'close': market_state['last_price'],
            'volume': market_state['last_volume'],
            'best_bid': market_state['best_bid'],
            'best_ask': market_state['best_ask']
        }])
        
        # Process through strategy
        try:
            signals = await self.strategy.process_data(data_point, symbol)
            return signals if not signals.empty else None
        except Exception as e:
            self.logger.debug(f"Signal generation failed for {symbol}: {e}")
            return None

    def _find_asset_by_symbol(self, symbol: str):
        """Find asset by symbol in portfolio"""
        if hasattr(self.portfolio, '_find_asset_by_symbol'):
            return self.portfolio._find_asset_by_symbol(symbol)
        elif hasattr(self.portfolio, 'assets') and symbol in self.portfolio.assets:
            return self.portfolio.assets[symbol]
        return None

    async def _analyze_microstructure_performance(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze microstructure and performance metrics"""
        analysis = {
            'execution_quality': {},
            'market_impact_analysis': {},
            'latency_analysis': {},
            'orderbook_statistics': {},
            'tick_statistics': {}
        }
        
        # Analyze execution quality
        if simulation_results['execution_details']:
            executions = simulation_results['execution_details']
            analysis['execution_quality'] = {
                'total_executions': len(executions),
                'avg_execution_time': np.mean([e.get('execution_time', 0) for e in executions]),
                'execution_success_rate': simulation_results['orders_executed'] / max(simulation_results['signals_generated'], 1)
            }
        
        # Analyze market impact
        analysis['market_impact_analysis'] = self.execution_simulator.get_impact_statistics()
        
        # Analyze orderbook statistics
        if self.enable_orderbook:
            analysis['orderbook_statistics'] = self._analyze_orderbook_statistics()
        
        # Get final portfolio metrics
        final_value = await self.portfolio.get_total_value()
        analysis['portfolio_performance'] = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': ((final_value / self.initial_capital) - 1) * 100
        }
        
        return analysis

    def _analyze_orderbook_statistics(self) -> Dict[str, Any]:
        """Analyze order book statistics"""
        stats = {
            'symbols_with_orderbook': len(self.orderbook_engine.orderbooks),
            'avg_spread': 0,
            'avg_depth': 0
        }
        
        if self.orderbook_engine.orderbooks:
            spreads = []
            depths = []
            
            for symbol, book in self.orderbook_engine.orderbooks.items():
                if book['bids'] and book['asks']:
                    spread = book['asks'][0].price - book['bids'][0].price
                    spreads.append(spread)
                    depths.append(len(book['bids']) + len(book['asks']))
            
            if spreads:
                stats['avg_spread'] = np.mean(spreads)
                stats['avg_depth'] = np.mean(depths)
        
        return stats

    def _get_data_stream_summary(self, prepared_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Get summary of data streams"""
        summary = {
            'data_types': list(prepared_data.keys()),
            'total_symbols': 0,
            'total_events': 0,
            'date_range': None
        }
        
        all_timestamps = []
        for data_type, streams in prepared_data.items():
            summary['total_symbols'] += len(streams)
            for symbol, df in streams.items():
                summary['total_events'] += len(df)
                if 'timestamp' in df.columns:
                    all_timestamps.extend(df['timestamp'].tolist())
        
        if all_timestamps:
            summary['date_range'] = {
                'start': min(all_timestamps),
                'end': max(all_timestamps),
                'duration': max(all_timestamps) - min(all_timestamps)
            }
        
        return summary

    async def _ensure_assets_in_portfolio(self, symbols: List[str]) -> None:
        """Ensure all symbols exist as assets in portfolio"""
        if not self.portfolio:
            return
            
        existing_assets = self.portfolio.list_assets()
        
        for symbol in symbols:
            if symbol not in existing_assets:
                self.logger.info(f"Adding {symbol} to portfolio for market replay")
                
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

    async def process_data_point(self, data_point: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process single data point (maintained for compatibility)"""
        # This method is maintained for compatibility with the base class
        # but the main processing happens in the event-driven _process_market_event method
        
        if data_point.empty:
            return pd.DataFrame()
        
        # For real-time compatibility, process as OHLCV event
        if self.strategy:
            try:
                signals = await self.strategy.process_data(data_point, symbol)
                
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

    async def shutdown(self) -> None:
        """Clean up resources"""
        # Clean up market simulation components
        self.orderbook_engine = None
        self.tick_processor = None
        self.execution_simulator = None
        
        # Clear state
        self.market_state.clear()
        self.execution_log.clear()
        self.event_queue.clear()
        self.data_streams.clear()
        
        await super().shutdown()

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced engine metrics"""
        metrics = super().get_metrics()
        
        metrics.update({
            'market_replay_mode': True,
            'orderbook_enabled': self.enable_orderbook,
            'tick_data_enabled': self.enable_tick_data,
            'latency_simulation': self.latency_simulation,
            'data_types_supported': list(self.data_types),
            'symbols_tracked': len(self.market_state),
            'current_time': self.current_time,
            'events_processed': len(self.execution_log)
        })
        
        return metrics


class TickDataProcessor:
    """Processor for tick data aggregation and analysis"""
    
    def __init__(self, aggregation_window_ms: int = 100):
        self.aggregation_window_ms = aggregation_window_ms
        self.tick_buffers = {}  # symbol -> list of ticks
        self.last_aggregation = {}  # symbol -> timestamp
        
    def process_tick(self, tick: TickData):
        """Process a single tick"""
        symbol = tick.symbol
        
        if symbol not in self.tick_buffers:
            self.tick_buffers[symbol] = []
            self.last_aggregation[symbol] = tick.timestamp
        
        self.tick_buffers[symbol].append(tick)
        
        # Check if aggregation window has passed
        time_diff = (tick.timestamp - self.last_aggregation[symbol]).total_seconds() * 1000
        if time_diff >= self.aggregation_window_ms:
            self._aggregate_ticks(symbol, tick.timestamp)
    
    def _aggregate_ticks(self, symbol: str, current_time: pd.Timestamp):
        """Aggregate ticks for a symbol"""
        if symbol not in self.tick_buffers or not self.tick_buffers[symbol]:
            return
        
        ticks = self.tick_buffers[symbol]
        
        # Calculate aggregated metrics
        prices = [tick.price for tick in ticks]
        sizes = [tick.size for tick in ticks]
        
        aggregated = {
            'symbol': symbol,
            'timestamp': current_time,
            'count': len(ticks),
            'volume': sum(sizes),
            'vwap': sum(p * s for p, s in zip(prices, sizes)) / sum(sizes) if sizes else 0,
            'min_price': min(prices) if prices else 0,
            'max_price': max(prices) if prices else 0,
            'last_price': prices[-1] if prices else 0
        }
        
        # Clear buffer and update last aggregation time
        self.tick_buffers[symbol] = []
        self.last_aggregation[symbol] = current_time
        
        return aggregated


class AdvancedExecutionSimulator:
    """Advanced execution simulator with market impact and latency"""
    
    def __init__(self, enable_impact: bool = True, latency_ms: int = 5):
        self.enable_impact = enable_impact
        self.latency_ms = latency_ms
        self.impact_history = []
        self.execution_stats = {
            'total_executions': 0,
            'avg_impact': 0,
            'avg_latency': 0
        }
    
    async def simulate_execution(self, order: Dict[str, Any], market_state: Dict[str, Any],
                               orderbook_engine: AdvancedOrderBookEngine) -> Dict[str, Any]:
        """Simulate order execution with realistic market effects"""
        symbol = order['symbol']
        side = order['direction']
        quantity = order['quantity']
        order_type = order.get('order_type', 'market')
        
        # Simulate network latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)
        
        # Get market state
        current_price = market_state.get('last_price', 0)
        if current_price <= 0:
            return {'success': False, 'reason': 'No market price available'}
        
        # Simulate execution based on order type
        if order_type == 'market':
            # Market order - use orderbook matching if available
            if orderbook_engine and symbol in orderbook_engine.orderbooks:
                match_result = orderbook_engine.simulate_order_matching(symbol, side, quantity)
                if match_result['success']:
                    # Record impact
                    impact = abs(match_result['avg_price'] - current_price) / current_price
                    self._record_impact(symbol, impact, quantity)
                    
                    return {
                        'success': True,
                        'filled_quantity': match_result['filled_quantity'],
                        'avg_price': match_result['avg_price'],
                        'fills': match_result['fills'],
                        'market_impact': impact,
                        'execution_method': 'orderbook_matching'
                    }
            
            # Fallback to simple execution with slippage
            slippage = self._calculate_market_slippage(quantity, current_price)
            if side.lower() == 'buy':
                execution_price = current_price * (1 + slippage)
            else:
                execution_price = current_price * (1 - slippage)
            
            impact = abs(execution_price - current_price) / current_price
            self._record_impact(symbol, impact, quantity)
            
            return {
                'success': True,
                'filled_quantity': quantity,
                'avg_price': execution_price,
                'market_impact': impact,
                'execution_method': 'slippage_model'
            }
        
        elif order_type == 'limit':
            # Limit order - check if price is achievable
            limit_price = order.get('price', current_price)
            
            if side.lower() == 'buy' and limit_price >= current_price:
                # Buy limit at or above market - can execute
                execution_price = min(limit_price, current_price)
            elif side.lower() == 'sell' and limit_price <= current_price:
                # Sell limit at or below market - can execute
                execution_price = max(limit_price, current_price)
            else:
                # Limit order not marketable
                return {
                    'success': False,
                    'reason': 'Limit price not marketable',
                    'limit_price': limit_price,
                    'market_price': current_price
                }
            
            return {
                'success': True,
                'filled_quantity': quantity,
                'avg_price': execution_price,
                'market_impact': 0,  # Limit orders typically have minimal impact
                'execution_method': 'limit_order'
            }
        
        return {'success': False, 'reason': f'Unsupported order type: {order_type}'}
    
    def _calculate_market_slippage(self, quantity: float, price: float) -> float:
        """Calculate market slippage based on order size"""
        if not self.enable_impact:
            return 0.0001  # Fixed 1bp slippage
        
        # Simple slippage model based on order size
        notional = quantity * price
        
        if notional < 1000:
            return 0.0001  # 1bp for small orders
        elif notional < 10000:
            return 0.0005  # 5bp for medium orders
        else:
            return 0.001   # 10bp for large orders
    
    def _record_impact(self, symbol: str, impact: float, quantity: float):
        """Record market impact for analysis"""
        self.impact_history.append({
            'symbol': symbol,
            'impact': impact,
            'quantity': quantity,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update running statistics
        self.execution_stats['total_executions'] += 1
        total_impact = sum(record['impact'] for record in self.impact_history)
        self.execution_stats['avg_impact'] = total_impact / len(self.impact_history)
    
    def get_impact_statistics(self) -> Dict[str, Any]:
        """Get market impact statistics"""
        if not self.impact_history:
            return {'no_data': True}
        
        impacts = [record['impact'] for record in self.impact_history]
        quantities = [record['quantity'] for record in self.impact_history]
        
        return {
            'total_executions': len(self.impact_history),
            'avg_impact_bps': np.mean(impacts) * 10000,
            'max_impact_bps': np.max(impacts) * 10000,
            'min_impact_bps': np.min(impacts) * 10000,
            'avg_quantity': np.mean(quantities),
            'impact_volatility': np.std(impacts) * 10000
        }