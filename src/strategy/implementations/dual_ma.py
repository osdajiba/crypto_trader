#!/usr/bin/env python3
# src/strategy/implementations/dual_ma.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List
import asyncio

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.strategy.base import BaseStrategy


@register_factory_class('strategy_factory', 'dual_ma', 
                      description="Dual Moving Average Crossover Strategy",
                      category="trend",
                      features=["moving_averages", "crossover", "trend_following"],
                      parameters=[
                          {"name": "short_window", "type": "int", "default": 20, "description": "Short moving average period"},
                          {"name": "long_window", "type": "int", "default": 60, "description": "Long moving average period"},
                          {"name": "signal_threshold", "type": "float", "default": 0.005, "description": "Signal generation threshold"},
                          {"name": "position_size", "type": "float", "default": 0.01, "description": "Position size as fraction of capital"},
                          {"name": "use_risk_based_sizing", "type": "bool", "default": True, "description": "Use risk-based position sizing"}
                      ])
class DualMAStrategy(BaseStrategy):
    """
    Implementation of Dual Moving Average crossover strategy generating trading signals 
    using short-term and long-term moving averages.
    
    Enhanced with factor registration and efficient data management.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Dual MA strategy
        
        Args:
            config: Configuration manager instance
            params: Strategy-specific parameters
                Expected parameters: short_window, long_window, symbol, lookback_period
        """
        # Load parameters from config if not provided in params
        params = params or {}
        config_params = config.get("strategy", "parameters", default={})
        
        # Override with params if provided
        params.setdefault("short_window", config_params.get("fast_period", 20))
        params.setdefault("long_window", config_params.get("slow_period", 60))
        params.setdefault("symbol", config.get("trading", "instruments", 0, default="BTC/USDT"))
        params.setdefault("signal_threshold", config_params.get("threshold", 0.005))
        params.setdefault("position_size", 0.01)  # Default 1%
        params.setdefault("use_risk_based_sizing", True)
        
        # Initialize parent class
        super().__init__(config, params)
        
        # Store parameters as instance variables for easy access
        self.short_window = self.params["short_window"]
        self.long_window = self.params["long_window"]
        self.primary_symbol = self.params["symbol"]
        self.signal_threshold = self.params["signal_threshold"]
        self.position_size = self.params["position_size"]
        self.use_risk_based_sizing = self.params["use_risk_based_sizing"]

    def _init_factors(self) -> None:
        """Initialize strategy factors"""
        # Register short MA factor
        self.register_ma_factor('short_ma', self.short_window)
        
        # Register long MA factor
        self.register_ma_factor('long_ma', self.long_window)
        
        # Register EMA factors for smoother crossovers
        self.register_ema_factor('short_ema', self.short_window)
        self.register_ema_factor('long_ema', self.long_window)
        
        # Register price percentage change factor for volatility measure
        self.register_factor('price_change', 10, self._calculate_price_change, is_differential=True)
        
        # Log registration
        self.logger.debug(f"Registered factors: short_ma (window={self.short_window}), long_ma (window={self.long_window})")

    def _calculate_price_change(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """
        Calculate percentage price change over window
        
        Args:
            data: Price data
            **kwargs: Additional parameters
            
        Returns:
            pd.Series: Percentage price change
        """
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
            
        # Calculate percentage change over 10 periods
        return data['close'].pct_change(10)

    async def initialize(self) -> None:
        """Initialize resources and validate parameters"""
        # Parameter validation
        if self.short_window >= self.long_window:
            self.logger.error(f"short_window ({self.short_window}) must be less than long_window ({self.long_window})")
            raise ValueError("short_window must be less than long_window")
        
        # Adjust lookback period if needed
        min_lookback = self.long_window * 2  # At least double the long window for meaningful signals
        if self.lookback_period < min_lookback:
            self.logger.warning(f"lookback_period ({self.lookback_period}) is less than recommended ({min_lookback}), adjusting")
            self.lookback_period = min_lookback
            
        # Parent class initialization - includes factor initialization from _init_factors
        await super().initialize()
            
        self.logger.info(f"Initialized DualMAStrategy with short_window={self.short_window}, long_window={self.long_window}")

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            data: Market data for signal generation (e.g. OHLCV)

        Returns:
            pd.DataFrame: Signals containing 'timestamp', 'symbol', 'action' (buy/sell) columns
        """        
        if data.empty:
            self.logger.warning("Empty data, cannot generate signals")
            return pd.DataFrame()
        
        # Get the symbol
        symbol = self.primary_symbol
        if 'symbol' in data.columns:
            # If data has symbol column, use the first one (assuming it's all the same)
            if not data.empty:
                symbol = data['symbol'].iloc[0]
        
        # Clone data to avoid modifying the original
        working_data = data.copy()
        
        try:
            # Calculate moving averages using optimized factor calculation
            short_ma = self.calculate_factor(working_data, 'short_ma', symbol)
            long_ma = self.calculate_factor(working_data, 'long_ma', symbol)
            
            # Calculate EMAs for confirmation
            short_ema = self.calculate_factor(working_data, 'short_ema', symbol)
            long_ema = self.calculate_factor(working_data, 'long_ema', symbol)
            
            # Calculate price change for volatility
            price_change = self.calculate_factor(working_data, 'price_change', symbol)
            
            # Add factors to working data
            working_data['short_ma'] = short_ma
            working_data['long_ma'] = long_ma
            working_data['short_ema'] = short_ema
            working_data['long_ema'] = long_ema
            working_data['price_change'] = price_change
            
            # Calculate moving average spread as percentage
            working_data['ma_spread'] = (short_ma - long_ma) / long_ma
            
            # Create signal DataFrame using last 2 rows for potential signals
            last_rows = working_data.tail(2)
            if len(last_rows) < 2:
                return pd.DataFrame()  # Need at least 2 rows for crossover detection
                
            signals = pd.DataFrame()
            
            # Detect crossovers with improved reliability
            prev_spread = last_rows['ma_spread'].iloc[0]
            curr_spread = last_rows['ma_spread'].iloc[1]
            
            # Get confirmation from EMA
            prev_ema_diff = (last_rows['short_ema'].iloc[0] - last_rows['long_ema'].iloc[0])
            curr_ema_diff = (last_rows['short_ema'].iloc[1] - last_rows['long_ema'].iloc[1])
            
            # Get confirmation from price
            curr_price = last_rows['close'].iloc[1]
            prev_price = last_rows['close'].iloc[0]
            
            # Signal conditions with improved reliability
            buy_signal = (
                prev_spread < 0 and curr_spread > 0  # MA crossover
                and prev_ema_diff < 0 and curr_ema_diff > 0  # EMA confirmation
                and abs(curr_spread) > self.signal_threshold  # Sufficient separation
                and curr_price > prev_price  # Price confirmation
            )
            
            sell_signal = (
                prev_spread > 0 and curr_spread < 0  # MA crossover
                and prev_ema_diff > 0 and curr_ema_diff < 0  # EMA confirmation
                and abs(curr_spread) > self.signal_threshold  # Sufficient separation
                and curr_price < prev_price  # Price confirmation
            )
            
            # Create signal if conditions met
            if buy_signal or sell_signal:
                row = last_rows.iloc[1]  # Use latest row for signal data
                
                # Create signal
                signal_data = {
                    'timestamp': row.name if isinstance(row.name, pd.Timestamp) else (
                        row['datetime'] if 'datetime' in row else row.get('timestamp', pd.Timestamp.now())
                    ),
                    'symbol': symbol,
                    'action': 'buy' if buy_signal else 'sell',
                    'price': float(row['close']),
                    'ma_spread': float(curr_spread),
                    'reason': 'MA Crossover with EMA confirmation',
                    'short_ma': float(short_ma.iloc[-1]),
                    'long_ma': float(long_ma.iloc[-1])
                }
                
                # Calculate quantity
                quantity = self.calculate_position_size(
                    float(curr_price),
                    self.config.get("trading", "capital", "initial", default=100000)
                )
                
                signal_data['quantity'] = quantity
                signals = pd.DataFrame([signal_data])
                
                self.logger.info(
                    f"Generated {signal_data['action']} signal for {symbol} @ ${float(curr_price):.2f}, "
                    f"quantity: {quantity:.6f}, spread: {float(curr_spread)*100:.2f}%"
                )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        Calculate position size based on price and available capital
        
        Args:
            price: Current asset price
            capital: Available capital
            
        Returns:
            float: Position size
        """
        if self.use_risk_based_sizing:
            # Get risk settings from config
            risk_per_trade = self.config.get("risk", "exposure", "risk_per_trade", default=0.01)  # Default 1% risk
            max_position = self.config.get("trading", "limits", "position", default=0.1)  # Default max 10%
            
            # Calculate position size
            risk_amount = capital * risk_per_trade
            max_amount = capital * max_position
            
            # Use smaller of the two values
            position_value = min(risk_amount, max_amount)
            
            # Convert to quantity
            quantity = position_value / price if price > 0 else 0
            
            return quantity
        else:
            # Use fixed position size as percentage of capital
            position_value = capital * self.position_size
            
            # Convert to quantity
            quantity = position_value / price if price > 0 else 0
            
            return quantity