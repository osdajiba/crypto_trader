#!/usr/bin/env python3
# src/strategy/implementations/multi_factors.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Callable
import asyncio

from src.common.abstract_factory import register_factory_class
from src.common.config import ConfigManager
from src.strategy.base import BaseStrategy


@register_factory_class('strategy_factory', 'multi_factors', 
                      description="Multi-Factor Strategy with custom factor combinations",
                      category="custom",
                      features=["multi_factor", "customizable", "technical_indicators"],
                      parameters=[
                          {"name": "factors", "type": "list", "default": ["ma_20", "ma_50", "rsi"], "description": "List of factors to use"},
                          {"name": "weights", "type": "dict", "default": {}, "description": "Factor weights for scoring"},
                          {"name": "threshold", "type": "float", "default": 0.7, "description": "Signal generation threshold"}
                      ])
class MultiFactorsStrategy(BaseStrategy):
    """
    Multi-Factor Strategy that combines multiple technical indicators and signals.
    
    This strategy offers a flexible framework to combine various factors with custom weights
    to generate trading signals based on a composite score.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Multi-Factor strategy
        
        Args:
            config: Configuration manager instance
            params: Strategy-specific parameters
        """
        # Initialize parameters with defaults
        params = params or {}
        
        # Load factors and weights from config if not in params
        strategy_config = config.get("strategy", "multi_factors", default={})
        
        # Set default factors if not provided
        params.setdefault("factors", strategy_config.get("factors", ["ma_20", "ma_50", "rsi"]))
        params.setdefault("weights", strategy_config.get("weights", {}))
        params.setdefault("threshold", strategy_config.get("threshold", 0.7))
        params.setdefault("lookback_period", 100)  # Larger default lookback for multiple factors
        
        # Initialize parent class
        super().__init__(config, params)
        
        # Factor definitions - mapping factor names to calculation functions
        self.factor_definitions = {
            "ma_20": {"window": 20, "type": "ma", "column": "close"},
            "ma_50": {"window": 50, "type": "ma", "column": "close"},
            "ma_200": {"window": 200, "type": "ma", "column": "close"},
            "ema_20": {"window": 20, "type": "ema", "column": "close"},
            "ema_50": {"window": 50, "type": "ema", "column": "close"},
            "rsi": {"window": 14, "type": "rsi", "column": "close"},
            "volume_sma": {"window": 20, "type": "ma", "column": "volume"},
            "price_change": {"window": 5, "type": "custom", "column": "close"},
            "volatility": {"window": 20, "type": "custom", "column": "close"}
        }
        
        # Extract parameters to instance variables
        self.selected_factors = self.params["factors"]
        self.weights = self.params["weights"]
        self.threshold = self.params["threshold"]
        
        # Initialize signal score components
        self.factor_signals = {}

    def _init_factors(self) -> None:
        """Initialize selected factors"""
        for factor_name in self.selected_factors:
            if factor_name in self.factor_definitions:
                factor_def = self.factor_definitions[factor_name]
                
                # Register the appropriate factor type
                if factor_def["type"] == "ma":
                    self.register_ma_factor(factor_name, factor_def["window"], factor_def["column"])
                elif factor_def["type"] == "ema":
                    self.register_ema_factor(factor_name, factor_def["window"], factor_def["column"])
                elif factor_def["type"] == "rsi":
                    self._register_rsi_factor(factor_name, factor_def["window"], factor_def["column"])
                elif factor_name == "price_change":
                    self.register_factor(factor_name, factor_def["window"], self._calculate_price_change, is_differential=True)
                elif factor_name == "volatility":
                    self.register_factor(factor_name, factor_def["window"], self._calculate_volatility)
                    
                self.logger.debug(f"Registered factor: {factor_name}")
            else:
                self.logger.warning(f"Unknown factor '{factor_name}', skipping")
    
    def _register_rsi_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Register a Relative Strength Index (RSI) factor
        
        Args:
            name: Factor name
            window_size: RSI calculation period
            price_column: Price column to use
        """
        def calculate_rsi(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column not in data.columns:
                return pd.Series(index=data.index)
                
            # Calculate price changes
            delta = data[price_column].diff()
            
            # Create gain and loss series
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=window_size, min_periods=1).mean()
            avg_loss = loss.rolling(window=window_size, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        self.register_factor(name, window_size, calculate_rsi)
        self.logger.debug(f"Registered RSI factor '{name}' with window size {window_size}")
    
    def _calculate_price_change(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate percentage price change over window"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].pct_change(5)
    
    def _calculate_volatility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate price volatility (standard deviation)"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].rolling(window=20).std() / data['close']
    
    async def initialize(self) -> None:
        """Initialize strategy and validate parameters"""
        # Validate factors
        invalid_factors = [f for f in self.selected_factors if f not in self.factor_definitions]
        if invalid_factors:
            self.logger.warning(f"Unknown factors: {invalid_factors}")
            # Filter out invalid factors
            self.selected_factors = [f for f in self.selected_factors if f in self.factor_definitions]
        
        # Validate weights
        if not self.weights:
            # Set default equal weights
            self.weights = {factor: 1.0 / len(self.selected_factors) for factor in self.selected_factors}
            self.logger.info(f"Using default equal weights: {self.weights}")
        else:
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        # Parent initialization
        await super().initialize()
        
        self.logger.info(f"Initialized MultiFactorsStrategy with {len(self.selected_factors)} factors")
    
def _calculate_factor_signal(self, factor_name: str, data: pd.DataFrame, symbol: str) -> float:
        """
        Calculate signal strength (-1 to 1) for a given factor
        
        Args:
            factor_name: Name of the factor
            data: Market data
            symbol: Trading symbol
            
        Returns:
            Signal strength from -1 (strong sell) to 1 (strong buy)
        """
        if factor_name not in self.factors:
            return 0.0
            
        # Get factor instance
        factor = self.factors[factor_name]
        
        # Calculate factor values
        factor_result = factor.calculate(data)
        
        # Handle different return types
        if isinstance(factor_result, pd.DataFrame):
            # For multi-column factors, get relevant column
            if factor_name == "bbands":
                # For Bollinger Bands, use percent_b
                if "percent_b" in factor_result.columns:
                    factor_value = factor_result["percent_b"].iloc[-1]
                    # Normalize to -1 to 1 range
                    return (factor_value - 0.5) * 2
                elif "width" in factor_result.columns:
                    # If percent_b not available, use width for volatility signal
                    width = factor_result["width"].iloc[-1]
                    # High width means high volatility (potentially bearish)
                    return -1 * min(width * 10, 1) if width > 0.05 else 0.2
            elif factor_name == "macd":
                # For MACD, use histogram
                if "histogram" in factor_result.columns:
                    histogram = factor_result["histogram"].iloc[-1]
                    # Normalize to -1 to 1 range with 0.01 threshold
                    return np.clip(histogram / 0.01, -1, 1)
            
            # Default to first column if specific handling not defined
            factor_value = factor_result.iloc[-1, 0]
        else:
            factor_value = factor_result.iloc[-1]
        
        # Different interpretation based on factor type
        factor_def = self.factor_definitions[factor_name]
        factor_type = factor_def["factor_type"]
        
        if factor_type == "sma" or factor_type == "ema":
            # For MA/EMA, compare to current price
            if "close" in data.columns and len(data) > 0:
                current_price = data["close"].iloc[-1]
                # Calculate distance as percentage
                if factor_value > 0:
                    distance = (current_price - factor_value) / factor_value
                    # Normalize to -1 to 1 range with 5% threshold
                    normalized = np.clip(distance / 0.05, -1, 1)
                    return normalized
        
        elif factor_type == "rsi":
            # RSI interpretation: 
            # < 30: oversold (buy), > 70: overbought (sell)
            if 0 <= factor_value <= 100:
                if factor_value < 30:
                    # Buy signal strength increases as RSI drops below 30
                    return (30 - factor_value) / 30
                elif factor_value > 70:
                    # Sell signal strength increases as RSI rises above 70
                    return -1 * (factor_value - 70) / 30
                else:
                    # Neutral zone
                    return (50 - factor_value) / 50
        
        elif factor_name == "price_change":
            # Price change interpretation
            # Limit to -1 to 1 range
            return np.clip(factor_value * 10, -1, 1)
            
        elif factor_name == "volatility":
            # Volatility interpretation 
            # High volatility (> 0.02) gives sell signal, low volatility gives weak buy signal
            if factor_value > 0.02:
                return -1 * min(factor_value * 25, 1)
            else:
                return 0.2  # Slight bullish bias in low volatility
        
        return 0.0#!/usr/bin/env python3
# src/strategy/implementations/multi_factors.py

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, List, Callable
import asyncio

from src.common.abstract_factory import register_factory_class
from src.common.config import ConfigManager
from src.strategy.base import BaseStrategy
from src.strategy.factors import get_factor_registry, FactorBase, SignalType
from src.strategy.factors.momentum import RSI
from src.strategy.factors.trend import SMA, EMA
from src.strategy.factors.volatility import BollingerBands
from src.strategy.factors.custom import PriceChangeRate, VolatilityIndex


@register_factory_class('strategy_factory', 'multi_factors', 
                      description="Multi-Factor Strategy with custom factor combinations",
                      category="custom",
                      features=["multi_factor", "customizable", "technical_indicators"],
                      parameters=[
                          {"name": "factors", "type": "list", "default": ["ma_20", "ma_50", "rsi"], "description": "List of factors to use"},
                          {"name": "weights", "type": "dict", "default": {}, "description": "Factor weights for scoring"},
                          {"name": "threshold", "type": "float", "default": 0.7, "description": "Signal generation threshold"}
                      ])
class MultiFactorsStrategy(BaseStrategy):
    """
    Multi-Factor Strategy that combines multiple technical indicators and signals.
    
    This strategy offers a flexible framework to combine various factors with custom weights
    to generate trading signals based on a composite score.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Multi-Factor strategy
        
        Args:
            config: Configuration manager instance
            params: Strategy-specific parameters
        """
        # Initialize parameters with defaults
        params = params or {}
        
        # Load factors and weights from config if not in params
        strategy_config = config.get("strategy", "multi_factors", default={})
        
        # Set default factors if not provided
        params.setdefault("factors", strategy_config.get("factors", ["ma_20", "ma_50", "rsi"]))
        params.setdefault("weights", strategy_config.get("weights", {}))
        params.setdefault("threshold", strategy_config.get("threshold", 0.7))
        params.setdefault("lookback_period", 100)  # Larger default lookback for multiple factors
        
        # Initialize parent class
        super().__init__(config, params)
        
        # Factor definitions - mapping factor names to calculation functions
        self.factor_definitions = {
            "ma_20": {"factor_type": "sma", "params": {"period": 20, "price_col": "close"}},
            "ma_50": {"factor_type": "sma", "params": {"period": 50, "price_col": "close"}},
            "ma_200": {"factor_type": "sma", "params": {"period": 200, "price_col": "close"}},
            "ema_20": {"factor_type": "ema", "params": {"period": 20, "price_col": "close"}},
            "ema_50": {"factor_type": "ema", "params": {"period": 50, "price_col": "close"}},
            "rsi": {"factor_type": "rsi", "params": {"period": 14, "price_col": "close"}},
            "volume_sma": {"factor_type": "sma", "params": {"period": 20, "price_col": "volume"}},
            "price_change": {"factor_type": "custom", "params": {"period": 5, "price_col": "close"}},
            "volatility": {"factor_type": "custom", "params": {"period": 20, "price_col": "close"}},
            "bbands": {"factor_type": "bollinger", "params": {"period": 20, "std_dev": 2.0, "price_col": "close"}}
        }
        
        # Get factor registry
        self.factor_registry = get_factor_registry()
        
        # Initialize factor instances
        self.factors = {}
        
        # Extract parameters to instance variables
        self.selected_factors = self.params["factors"]
        self.weights = self.params["weights"]
        self.threshold = self.params["threshold"]
        
        # Initialize signal score components
        self.factor_signals = {}

    def _init_factors(self) -> None:
        """Initialize selected factors"""
        for factor_name in self.selected_factors:
            if factor_name in self.factor_definitions:
                factor_def = self.factor_definitions[factor_name]
                factor_type = factor_def["factor_type"]
                params = factor_def["params"]
                
                # Create factor instance from registry
                if factor_type == "custom":
                    # Handle custom factors
                    if factor_name == "price_change":
                        self.factors[factor_name] = PriceChangeRate(**params)
                    elif factor_name == "volatility":
                        self.factors[factor_name] = VolatilityIndex(**params)
                else:
                    # Create standard factor from registry
                    self.factors[factor_name] = self.factor_registry.create_factor(factor_type, **params)
                
                if self.factors[factor_name]:
                    self.logger.debug(f"Created factor: {factor_name}")
                else:
                    self.logger.warning(f"Failed to create factor '{factor_name}'")
            else:
                self.logger.warning(f"Unknown factor '{factor_name}', skipping")
    
    def _register_rsi_factor(self, name: str, window_size: int, price_column: str = 'close') -> None:
        """
        Register a Relative Strength Index (RSI) factor
        
        Args:
            name: Factor name
            window_size: RSI calculation period
            price_column: Price column to use
        """
        def calculate_rsi(data: pd.DataFrame, **kwargs) -> pd.Series:
            if price_column not in data.columns:
                return pd.Series(index=data.index)
                
            # Calculate price changes
            delta = data[price_column].diff()
            
            # Create gain and loss series
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = abs(loss)
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=window_size, min_periods=1).mean()
            avg_loss = loss.rolling(window=window_size, min_periods=1).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        self.register_factor(name, window_size, calculate_rsi)
        self.logger.debug(f"Registered RSI factor '{name}' with window size {window_size}")
    
    def _calculate_price_change(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate percentage price change over window"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].pct_change(5)
    
    def _calculate_volatility(self, data: pd.DataFrame, **kwargs) -> pd.Series:
        """Calculate price volatility (standard deviation)"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index)
        
        return data['close'].rolling(window=20).std() / data['close']
    
    async def initialize(self) -> None:
        """Initialize strategy and validate parameters"""
        # Validate factors
        invalid_factors = [f for f in self.selected_factors if f not in self.factor_definitions]
        if invalid_factors:
            self.logger.warning(f"Unknown factors: {invalid_factors}")
            # Filter out invalid factors
            self.selected_factors = [f for f in self.selected_factors if f in self.factor_definitions]
        
        # Initialize factors
        self._init_factors()
        
        # Validate weights
        if not self.weights:
            # Set default equal weights
            self.weights = {factor: 1.0 / len(self.selected_factors) for factor in self.selected_factors}
            self.logger.info(f"Using default equal weights: {self.weights}")
        else:
            # Normalize weights to sum to 1
            weight_sum = sum(self.weights.values())
            if weight_sum > 0:
                self.weights = {k: v / weight_sum for k, v in self.weights.items()}
        
        # Parent initialization
        await super().initialize()
        
        self.logger.info(f"Initialized MultiFactorsStrategy with {len(self.selected_factors)} factors")
    
    def _calculate_factor_signal(self, factor_name: str, data: pd.DataFrame, symbol: str) -> float:
        """
        Calculate signal strength (-1 to 1) for a given factor
        
        Args:
            factor_name: Name of the factor
            data: Market data
            symbol: Trading symbol
            
        Returns:
            Signal strength from -1 (strong sell) to 1 (strong buy)
        """
        if factor_name not in self.factor_definitions:
            return 0.0
            
        # Get factor values
        factor_values = self.calculate_factor(data, factor_name, symbol)
        if factor_values.empty:
            return 0.0
            
        factor_value = factor_values.iloc[-1]
        
        # Different interpretation based on factor type
        factor_def = self.factor_definitions[factor_name]
        
        if factor_def["type"] == "ma" or factor_def["type"] == "ema":
            # For MA/EMA, compare to current price
            if factor_def["column"] in data.columns and len(data) > 0:
                current_price = data[factor_def["column"]].iloc[-1]
                # Calculate distance as percentage
                if factor_value > 0:
                    distance = (current_price - factor_value) / factor_value
                    # Normalize to -1 to 1 range with 5% threshold
                    normalized = np.clip(distance / 0.05, -1, 1)
                    return normalized
        
        elif factor_name == "rsi":
            # RSI interpretation: 
            # < 30: oversold (buy), > 70: overbought (sell)
            if 0 <= factor_value <= 100:
                if factor_value < 30:
                    # Buy signal strength increases as RSI drops below 30
                    return (30 - factor_value) / 30
                elif factor_value > 70:
                    # Sell signal strength increases as RSI rises above 70
                    return -1 * (factor_value - 70) / 30
                else:
                    # Neutral zone
                    return (50 - factor_value) / 50
        
        elif factor_name == "price_change":
            # Price change interpretation
            # Limit to -1 to 1 range
            return np.clip(factor_value * 10, -1, 1)
            
        elif factor_name == "volatility":
            # Volatility interpretation 
            # High volatility (> 0.02) gives sell signal, low volatility gives weak buy signal
            if factor_value > 0.02:
                return -1 * min(factor_value * 25, 1)
            else:
                return 0.2  # Slight bullish bias in low volatility
        
        return 0.0
    
    def _calculate_composite_score(self, data: pd.DataFrame, symbol: str) -> float:
        """
        Calculate composite signal score from all factors
        
        Args:
            data: Market data
            symbol: Trading symbol
            
        Returns:
            float: Composite score from -1 (strong sell) to 1 (strong buy)
        """
        factor_signals = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor_name in self.selected_factors:
            if factor_name not in self.factors:
                continue
                
            weight = self.weights.get(factor_name, 1.0 / len(self.selected_factors))
            signal = self._calculate_factor_signal(factor_name, data, symbol)
            
            factor_signals[factor_name] = signal
            weighted_sum += signal * weight
            total_weight += weight
        
        # Store factor signals for logging
        self.factor_signals = factor_signals
        
        # Return normalized weighted score
        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on composite factor score
        
        Args:
            data: Market data for signal generation
            
        Returns:
            pd.DataFrame: Trading signals
        """
        if data.empty:
            return pd.DataFrame()
        
        # Get symbol
        symbol = self.params.get("symbol", "unknown")
        if 'symbol' in data.columns and not data.empty:
            symbol = data['symbol'].iloc[0]
        
        try:
            # Calculate the composite score
            current_score = self._calculate_composite_score(data, symbol)
            
            # Get previous score if available
            if hasattr(self, 'previous_score'):
                score_change = current_score - self.previous_score
            else:
                score_change = 0
                
            # Store current score for next iteration
            self.previous_score = current_score
            
            # Generate signal based on threshold
            signals = pd.DataFrame()
            
            # Determine action based on score and threshold
            action = None
            if current_score > self.threshold:
                action = 'buy'
            elif current_score < -self.threshold:
                action = 'sell'
            
            # Only generate a signal if score crosses threshold with sufficient momentum
            if action and abs(score_change) > 0.1:
                # Create signal with current price
                if not data.empty and 'close' in data.columns:
                    current_price = data['close'].iloc[-1]
                    
                    # Calculate reasonable quantity
                    capital = self.config.get("trading", "capital", "initial", default=100000)
                    position_size = 0.02  # Default 2% position size
                    quantity = capital * position_size / current_price
                    
                    timestamp = data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else (
                        data['datetime'].iloc[-1] if 'datetime' in data.columns else pd.Timestamp.now()
                    )
                    
                    signal_data = {
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'action': action,
                        'price': float(current_price),
                        'quantity': float(quantity),
                        'score': float(current_score),
                        'reason': f"Multi-factor score: {current_score:.2f}",
                        'factors': self.factor_signals
                    }
                    
                    signals = pd.DataFrame([signal_data])
                    
                    # Log the signal generation with factor breakdown
                    factor_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.factor_signals.items()])
                    self.logger.info(
                        f"Generated {action} signal for {symbol} @ ${float(current_price):.2f}, "
                        f"score: {current_score:.2f}, factors: [{factor_str}]"
                    )
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return pd.DataFrame()