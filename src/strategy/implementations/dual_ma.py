#!/usr/bin/env python3
# src/strategy/implementations/dual_ma.py

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import asyncio

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.strategy.base import BaseStrategy


@register_factory_class('strategy_factory', 'dual_ma',
                       description="Dual Moving Average Crossover Strategy",
                       category="trend",
                       features=["moving_averages", "crossover", "trend_following"])
class DualMAStrategy(BaseStrategy):
    """
    Implementation of Dual Moving Average crossover strategy.
    
    This strategy generates trading signals based on crossovers between
    short-term and long-term moving averages. Additional filters are 
    applied to improve signal quality and reduce false positives.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dual Moving Average strategy.
        
        Args:
            config: Configuration manager instance
            params: Optional strategy parameters (will override config values)
        """
        # Load parameters from config with fallbacks
        self.short_window = config.get("strategy", "parameters", "fast_period", default=20)
        self.long_window = config.get("strategy", "parameters", "slow_period", default=60)
        self.signal_threshold = config.get("strategy", "parameters", "threshold", default=0.005)
        self.primary_symbol = config.get("trading", "instruments", 0, default="BTC/USDT")
        self.position_size = config.get("strategy", "parameters", "position_size", default=0.01)
        self.use_risk_based_sizing = config.get("strategy", "parameters", "use_risk_based_sizing", default=True)
        
        # Override with params if provided
        if params:
            self.short_window = params.get("short_window", self.short_window)
            self.long_window = params.get("long_window", self.long_window)
            self.signal_threshold = params.get("signal_threshold", self.signal_threshold)
            self.primary_symbol = params.get("symbol", self.primary_symbol)
            self.position_size = params.get("position_size", self.position_size)
            self.use_risk_based_sizing = params.get("use_risk_based_sizing", self.use_risk_based_sizing)
            
    
        self.lookback_period = max(self.long_window * 2, 100) + 1    # Set required lookback period based on window sizes
        
        super().__init__(config, params)
        
        self.logger.info(
            f"DualMAStrategy initialized with short_window={self.short_window}, "
            f"long_window={self.long_window}, threshold={self.signal_threshold}"
        )

    def _init_factors(self) -> None:
        """Register strategy factors for calculation."""
        # Register primary moving average factors
        self.register_ma_factor('short_ma', self.short_window)
        self.register_ma_factor('long_ma', self.long_window)
        
        # Register EMA factors for signal confirmation
        self.register_ema_factor('short_ema', self.short_window)
        self.register_ema_factor('long_ema', self.long_window)
        
        # Register price change factor for volatility measure
        self.register_factor('price_change', 10, self._calculate_price_change, is_differential=True)
        
        self.logger.debug(f"Registered factors for DualMAStrategy")

    def _calculate_price_change(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate percentage price change over window.
        
        Args:
            data: Price data DataFrame
            **kwargs: Additional parameters
            
        Returns:
            pd.Series: Percentage price change
        """
        if 'close' not in data.columns:
            self.logger.warning("Close price column not found in data")
            return pd.Series(index=data.index)
            
        return data['close'].pct_change(10)

    async def initialize(self) -> None:
        """Initialize strategy and validate parameters."""
        # Parameter validation
        if self.short_window >= self.long_window:
            self.logger.error(f"short_window ({self.short_window}) must be less than long_window ({self.long_window})")
            raise ValueError("short_window must be less than long_window")
            
        # Initialize parent class (registers factors)
        await super().initialize()
            
    def register_ma_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Register a simple moving average factor.
        
        Args:
            name: Factor name
            window_size: Window size for the moving average
            price_column: Column to calculate MA on (default: close)
        """
        def calculate_ma(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column in data.columns:
                return data[price_column].rolling(window=window_size, min_periods=1).mean()
            return pd.Series(index=data.index)
        
        self.register_factor(name, window_size, calculate_ma)
    
    def register_ema_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Register an exponential moving average factor.
        
        Args:
            name: Factor name
            window_size: Window size for the EMA
            price_column: Column to calculate EMA on (default: close)
        """
        def calculate_ema(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column in data.columns:
                return data[price_column].ewm(span=window_size, adjust=False).mean()
            return pd.Series(index=data.index)
        
        self.register_factor(name, window_size, calculate_ema)
        
    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data.
        
        Args:
            data: Market data DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with generated trading signals
        """        
        if data.empty:
            self.logger.warning("Empty data, cannot generate signals")
            return pd.DataFrame()
        
        # Determine symbol from data or use default
        symbol = self.primary_symbol
        if 'symbol' in data.columns and not data.empty:
            symbol = data['symbol'].iloc[0]
        
        try:
            # Calculate moving averages
            short_ma = self.calculate_factor(data, 'short_ma')
            long_ma = self.calculate_factor(data, 'long_ma')
            
            # Calculate EMAs for confirmation
            short_ema = self.calculate_factor(data, 'short_ema')
            long_ema = self.calculate_factor(data, 'long_ema')
            
            # Calculate price change for volatility
            price_change = self.calculate_factor(data, 'price_change')
            
            # Add factors to data for analysis
            processed_data = data.copy()
            processed_data['short_ma'] = short_ma
            processed_data['long_ma'] = long_ma
            processed_data['short_ema'] = short_ema
            processed_data['long_ema'] = long_ema
            processed_data['price_change'] = price_change
            
            # Calculate MA spread (percentage difference)
            processed_data['ma_spread'] = (short_ma - long_ma) / long_ma
            
            # Need at least 2 rows to detect crossovers
            if len(processed_data) < 2:
                return pd.DataFrame()
                
            # Get last two rows for signal generation
            last_rows = processed_data.tail(2)
            signals = []
            
            # Get values for comparison
            prev_spread = last_rows['ma_spread'].iloc[0]
            curr_spread = last_rows['ma_spread'].iloc[1]
            
            # Get confirmation from EMA
            prev_ema_diff = (last_rows['short_ema'].iloc[0] - last_rows['long_ema'].iloc[0])
            curr_ema_diff = (last_rows['short_ema'].iloc[1] - last_rows['long_ema'].iloc[1])
            
            # Get price confirmation
            curr_price = last_rows['close'].iloc[1]
            prev_price = last_rows['close'].iloc[0]
            
            # Buy signal conditions
            buy_signal = (
                prev_spread < 0 and curr_spread > 0 and  # MA crossover
                prev_ema_diff < 0 and curr_ema_diff > 0 and  # EMA confirmation
                abs(curr_spread) > self.signal_threshold and  # Sufficient separation
                curr_price > prev_price  # Price confirmation
            )
            
            # Sell signal conditions
            sell_signal = (
                prev_spread > 0 and curr_spread < 0 and  # MA crossover
                prev_ema_diff > 0 and curr_ema_diff < 0 and  # EMA confirmation
                abs(curr_spread) > self.signal_threshold and  # Sufficient separation
                curr_price < prev_price  # Price confirmation
            )
            
            # Generate signal if conditions are met
            if buy_signal or sell_signal:
                row = last_rows.iloc[1]  # Use latest row for signal data
                action = 'buy' if buy_signal else 'sell'
                
                # Calculate position size
                quantity = self.calculate_position_size(
                    curr_price,
                    self.config.get("trading", "capital", "initial", default=100000)
                )
                
                # Create signal data
                signal_data = {
                    'timestamp': row.name if isinstance(row.name, pd.Timestamp) else row.get('timestamp', pd.Timestamp.now()),
                    'symbol': symbol,
                    'action': action,
                    'price': float(curr_price),
                    'quantity': quantity,
                    'ma_spread': float(curr_spread),
                    'reason': 'MA Crossover with EMA confirmation',
                    'short_ma': float(short_ma.iloc[-1]),
                    'long_ma': float(long_ma.iloc[-1])
                }
                
                signals.append(signal_data)
                
                self.logger.info(
                    f"Generated {action} signal for {symbol} @ ${float(curr_price):.2f}, "
                    f"quantity: {quantity:.6f}, spread: {float(curr_spread)*100:.2f}%"
                )
            
            return pd.DataFrame(signals)
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    async def generate_signals_vectorized(self, data: pd.DataFrame, symbol: str, factor_values: Dict[str, pd.Series] = None) -> pd.DataFrame:
        """
        Generate trading signals for an entire dataset at once.
        
        This is a vectorized version of signal generation for improved performance
        in backtesting, especially with the OHLCVEngine.
        
        Args:
            data: Market data DataFrame for the entire period
            symbol: Trading symbol
            factor_values: Pre-calculated factor values (optional)
            
        Returns:
            pd.DataFrame: DataFrame with trading signals
        """
        if data.empty:
            self.logger.warning("Empty data, cannot generate signals")
            return pd.DataFrame()
        
        # Determine symbol from data or use provided symbol
        if symbol is None and 'symbol' in data.columns and not data.empty:
            symbol = data['symbol'].iloc[0]
        elif symbol is None:
            symbol = self.primary_symbol
        
        try:
            # Use pre-calculated factors if provided, otherwise calculate them
            if factor_values and 'short_ma' in factor_values and 'long_ma' in factor_values:
                short_ma = factor_values['short_ma']
                long_ma = factor_values['long_ma']
                short_ema = factor_values.get('short_ema')
                long_ema = factor_values.get('long_ema')
                price_change = factor_values.get('price_change')
            else:
                # Calculate moving averages
                short_ma = data['close'].rolling(window=self.short_window, min_periods=1).mean()
                long_ma = data['close'].rolling(window=self.long_window, min_periods=1).mean()
                
                # Calculate EMAs for confirmation
                short_ema = data['close'].ewm(span=self.short_window, adjust=False).mean()
                long_ema = data['close'].ewm(span=self.long_window, adjust=False).mean()
                
                # Calculate price change for volatility
                price_change = data['close'].pct_change(10)
            
            # Add indicators to data for analysis
            processed_data = data.copy()
            processed_data['short_ma'] = short_ma
            processed_data['long_ma'] = long_ma
            processed_data['short_ema'] = short_ema if short_ema is not None else short_ma
            processed_data['long_ema'] = long_ema if long_ema is not None else long_ma
            processed_data['price_change'] = price_change if price_change is not None else data['close'].pct_change(10)
            
            # Calculate MA spread (percentage difference)
            processed_data['ma_spread'] = (processed_data['short_ma'] - processed_data['long_ma']) / processed_data['long_ma']
            
            # Need at least 2 rows to detect crossovers
            if len(processed_data) < 2:
                return pd.DataFrame()
            
            # Calculate conditions vectorized
            # Buy condition: MA crossover from below to above
            ma_cross_up = (processed_data['ma_spread'] > 0) & (processed_data['ma_spread'].shift(1) < 0)
            
            # Sell condition: MA crossover from above to below
            ma_cross_down = (processed_data['ma_spread'] < 0) & (processed_data['ma_spread'].shift(1) > 0)
            
            # EMA confirmation
            ema_diff = processed_data['short_ema'] - processed_data['long_ema']
            ema_cross_up = (ema_diff > 0) & (ema_diff.shift(1) < 0)
            ema_cross_down = (ema_diff < 0) & (ema_diff.shift(1) > 0)
            
            # Price confirmation
            price_up = processed_data['close'] > processed_data['close'].shift(1)
            price_down = processed_data['close'] < processed_data['close'].shift(1)
            
            # Spread threshold
            spread_significant = processed_data['ma_spread'].abs() > self.signal_threshold
            
            # Combined conditions
            buy_signals = ma_cross_up & ema_cross_up & price_up & spread_significant
            sell_signals = ma_cross_down & ema_cross_down & price_down & spread_significant
            
            # Extract signal rows
            buy_rows = processed_data[buy_signals].copy()
            sell_rows = processed_data[sell_signals].copy()
            
            # Create signals DataFrame
            signals_list = []
            
            # Process buy signals
            if not buy_rows.empty:
                for idx, row in buy_rows.iterrows():
                    # Calculate position size
                    quantity = self.calculate_position_size(
                        row['close'],
                        self.config.get("trading", "capital", "initial", default=100000)
                    )
                    
                    # Create signal data
                    signal_data = {
                        'timestamp': idx if isinstance(idx, pd.Timestamp) else row.get('timestamp', row.get('datetime', pd.Timestamp.now())),
                        'symbol': symbol,
                        'action': 'buy',
                        'price': float(row['close']),
                        'quantity': quantity,
                        'ma_spread': float(row['ma_spread']),
                        'reason': 'MA Crossover with EMA confirmation',
                        'short_ma': float(row['short_ma']),
                        'long_ma': float(row['long_ma'])
                    }
                    signals_list.append(signal_data)
            
            # Process sell signals
            if not sell_rows.empty:
                for idx, row in sell_rows.iterrows():
                    # Calculate position size
                    quantity = self.calculate_position_size(
                        row['close'],
                        self.config.get("trading", "capital", "initial", default=100000)
                    )
                    
                    # Create signal data
                    signal_data = {
                        'timestamp': idx if isinstance(idx, pd.Timestamp) else row.get('timestamp', row.get('datetime', pd.Timestamp.now())),
                        'symbol': symbol,
                        'action': 'sell',
                        'price': float(row['close']),
                        'quantity': quantity,
                        'ma_spread': float(row['ma_spread']),
                        'reason': 'MA Crossover with EMA confirmation',
                        'short_ma': float(row['short_ma']),
                        'long_ma': float(row['long_ma'])
                    }
                    signals_list.append(signal_data)
            
            # Create DataFrame from signals list
            signals_df = pd.DataFrame(signals_list) if signals_list else pd.DataFrame()
            
            if not signals_df.empty:
                self.logger.info(f"Generated {len(signals_df)} vectorized signals for {symbol}")
            
            return signals_df
            
        except Exception as e:
            self.logger.error(f"Error generating vectorized signals: {str(e)}")
            return pd.DataFrame()
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        Calculate position size based on price and available capital.
        
        Args:
            price: Current asset price
            capital: Available capital
            
        Returns:
            float: Position size (quantity to trade)
        """
        if self.use_risk_based_sizing:
            # Get risk settings from config
            risk_per_trade = self.config.get("risk", "managers", "standard", "exposure", "risk_per_trade", default=0.01)
            max_position = self.config.get("trading", "limits", "position", default=0.1)
            
            # Calculate position size based on risk
            risk_amount = capital * risk_per_trade
            max_amount = capital * max_position
            
            # Use smaller of the two values
            position_value = min(risk_amount, max_amount)
            
        else:
            # Use fixed position size as percentage of capital
            position_value = capital * self.position_size
            
        # Convert to quantity
        quantity = position_value / price if price > 0 else 0
        return quantity