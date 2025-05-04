#!/usr/bin/env python3
# src/strategy/implementations/multi_factors.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import asyncio
import logging

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.strategy.base import BaseStrategy
from src.strategy.factors.factory import get_factor_factory
from src.strategy.factors.base import SignalType


@register_factory_class('strategy_factory', 'multi_factors', 
                      description="Multi-Factor Strategy combining various technical indicators",
                      category="custom",
                      features=["multi_factor", "customizable", "technical_indicators"],
                      parameters=[
                          {"name": "factors", "type": "list", "default": ["rsi", "macd", "bollinger"], "description": "List of factors to use"},
                          {"name": "weights", "type": "list", "default": [0.4, 0.3, 0.3], "description": "Weights for each factor"},
                          {"name": "signal_threshold", "type": "float", "default": 0.6, "description": "Signal strength threshold (0-1)"}
                      ])
class MultiFactorsStrategy(BaseStrategy):
    """
    Multi-Factor Strategy that combines multiple technical indicators
    
    This strategy allows flexible combination of technical factors with
    customizable weights and signal generation methods. It integrates
    with the factor library for advanced technical analysis.
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Multi-Factor strategy with configuration and parameters
        
        Args:
            config: System configuration manager
            params: Strategy-specific parameters (optional)
        """
        # Set default parameters if not provided
        params = params or {}
        
        # Load factors and weights from config if not provided in params
        params.setdefault("factors", config.get("strategy", "multi_factors", "factors", 
                                             default=["rsi", "macd", "bollinger"]))
        params.setdefault("weights", config.get("strategy", "multi_factors", "weights", 
                                             default=[0.4, 0.3, 0.3]))
        params.setdefault("signal_threshold", config.get("strategy", "multi_factors", "signal_threshold", 
                                                     default=0.6))
        params.setdefault("symbol", config.get("trading", "instruments", 0, default="BTC/USDT"))
        
        # Initialize base class
        super().__init__(config, params)
        
        # Store parameters as instance variables
        self.factors = self.params["factors"]
        self.weights = self.params["weights"]
        self.signal_threshold = self.params["signal_threshold"]
        self.primary_symbol = self.params["symbol"]
        
        # Factor signal configuration
        self.factor_signals = self._get_factor_signal_config()
        
        # Factor instances
        self._factor_instances = {}
        
        # Signal cache
        self._signal_cache = {}
    
    def _determine_required_history(self) -> int:
        """
        Determine required historical data points for strategy
        
        Returns:
            int: Number of required data points
        """
        # Different factors require different amounts of history
        # Use conservative estimate based on common factor requirements
        return 100
    
    def _get_factor_signal_config(self) -> Dict[str, Dict[str, Any]]:
        """
        Get factor signal configuration with optimal settings for each factor
        
        Returns:
            Dict[str, Dict[str, Any]]: Factor signal configuration
        """
        # Default configuration - optimized settings for each factor
        return {
            # Momentum factors
            "rsi": {
                "signal_type": SignalType.THRESHOLD,
                "params": {"upper_threshold": 70, "lower_threshold": 30}
            },
            "macd": {
                "signal_type": SignalType.CROSSOVER,
                "params": {"column": "histogram"}
            },
            "stochastic": {
                "signal_type": SignalType.THRESHOLD,
                "params": {"upper_threshold": 80, "lower_threshold": 20, "column": "k"}
            },
            
            # Trend factors
            "sma": {
                "signal_type": SignalType.STANDARD,
                "params": {}
            },
            "ema": {
                "signal_type": SignalType.STANDARD,
                "params": {}
            },
            "ichimoku": {
                "signal_type": SignalType.STANDARD,
                "params": {"column": "tenkan_sen"}
            },
            
            # Volatility factors
            "bollinger": {
                "signal_type": SignalType.THRESHOLD,
                "params": {"upper_threshold": 1.0, "lower_threshold": 0.0, "column": "percent_b"}
            },
            "atr": {
                "signal_type": SignalType.STANDARD,
                "params": {}
            },
            "adx": {
                "signal_type": SignalType.THRESHOLD,
                "params": {"upper_threshold": 25, "lower_threshold": 20, "column": "adx"}
            },
            
            # Volume factors
            "obv": {
                "signal_type": SignalType.MOMENTUM,
                "params": {}
            },
            "mfi": {
                "signal_type": SignalType.THRESHOLD,
                "params": {"upper_threshold": 80, "lower_threshold": 20}
            },
            "volume_osc": {
                "signal_type": SignalType.CROSSOVER,
                "params": {}
            }
        }
    
    async def _initialize_strategy(self) -> None:
        """Perform strategy-specific initialization"""
        # Create factor factory
        self.factor_factory = get_factor_factory(self.config)
        
        # Validate factors list
        available_factors = self.factor_factory.get_available_factors()
        for factor in self.factors:
            if factor not in available_factors:
                self.logger.warning(
                    f"Unknown factor '{factor}'. Available factors: {list(available_factors.keys())}"
                )
        
        # Validate weights
        if len(self.weights) != len(self.factors):
            self.logger.warning(
                f"Number of weights ({len(self.weights)}) does not match "
                f"number of factors ({len(self.factors)}). Using equal weights."
            )
            self.weights = [1.0 / len(self.factors)] * len(self.factors)
        
        # Normalize weights to sum to 1
        weight_sum = sum(self.weights)
        if weight_sum != 1.0:
            self.weights = [w / weight_sum for w in self.weights]
        
        # Initialize factor instances
        for factor_name in self.factors:
            try:
                factor_instance = await self.factor_factory.create(factor_name)
                if factor_instance:
                    self._factor_instances[factor_name] = factor_instance
                    self.logger.debug(f"Initialized factor: {factor_name}")
                else:
                    self.logger.warning(f"Failed to create factor: {factor_name}")
            except Exception as e:
                self.logger.error(f"Error creating factor {factor_name}: {e}")
        
        self.logger.info(
            f"Initialized MultiFactorsStrategy with factors: {self.factors}, "
            f"weights: {self.weights}, threshold: {self.signal_threshold}"
        )
    
    async def shutdown(self) -> None:
        """Clean up strategy resources"""
        # Clear factor instances and caches
        self._factor_instances.clear()
        self._signal_cache.clear()
        
        # Call base class shutdown
        await super().shutdown()
    
    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on multiple factors
        
        Args:
            data: Market data with sufficient history
            
        Returns:
            pd.DataFrame: Trading signals
        """
        if data.empty:
            return pd.DataFrame()
        
        # Get symbol from data
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else self.primary_symbol
        
        # Calculate factor signals and weights
        factor_signals = self._calculate_factor_signals(data)
        
        # Calculate combined signal strength for most recent data point
        signal_strength = 0.0
        buy_factors = []
        sell_factors = []
        
        for i, factor_name in enumerate(self.factors):
            if factor_name in factor_signals and not factor_signals[factor_name].empty:
                factor_value = factor_signals[factor_name].iloc[-1]
                factor_weight = self.weights[i]
                
                # Accumulate signal strength
                signal_strength += factor_value * factor_weight
                
                # Track contributing factors
                if factor_value > 0:
                    buy_factors.append(factor_name)
                elif factor_value < 0:
                    sell_factors.append(factor_name)
        
        # Generate signal if strength exceeds threshold
        current_price = data['close'].iloc[-1]
        signals = []
        
        if signal_strength >= self.signal_threshold:
            # Buy signal
            quantity = self.calculate_position_size(float(current_price), symbol)
            
            signal_data = {
                'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else (
                    data['datetime'].iloc[-1] if 'datetime' in data.columns else 
                    data['timestamp'].iloc[-1] if 'timestamp' in data.columns else 
                    pd.Timestamp.now()
                ),
                'symbol': symbol,
                'action': 'buy',
                'price': float(current_price),
                'quantity': quantity,
                'signal_strength': float(signal_strength),
                'reason': f"Multi-factor buy signal: {', '.join(buy_factors)}",
                'factors': buy_factors
            }
            
            signals.append(signal_data)
            
            self.logger.info(
                f"Generated BUY signal for {symbol} @ ${float(current_price):.2f}, "
                f"quantity: {quantity:.6f}, strength: {float(signal_strength):.2f}"
            )
            
        elif signal_strength <= -self.signal_threshold:
            # Sell signal
            quantity = self.calculate_position_size(float(current_price), symbol)
            
            signal_data = {
                'timestamp': data.index[-1] if isinstance(data.index, pd.DatetimeIndex) else (
                    data['datetime'].iloc[-1] if 'datetime' in data.columns else 
                    data['timestamp'].iloc[-1] if 'timestamp' in data.columns else 
                    pd.Timestamp.now()
                ),
                'symbol': symbol,
                'action': 'sell',
                'price': float(current_price),
                'quantity': quantity,
                'signal_strength': float(signal_strength),
                'reason': f"Multi-factor sell signal: {', '.join(sell_factors)}",
                'factors': sell_factors
            }
            
            signals.append(signal_data)
            
            self.logger.info(
                f"Generated SELL signal for {symbol} @ ${float(current_price):.2f}, "
                f"quantity: {quantity:.6f}, strength: {float(signal_strength):.2f}"
            )
        
        return pd.DataFrame(signals)
    
    def _calculate_factor_signals(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate signals for all factors
        
        Args:
            data: Market data
            
        Returns:
            Dict[str, pd.Series]: Factor signals
        """
        factor_signals = {}
        
        # Calculate signals for each factor
        for factor_name in self.factors:
            if factor_name in self._factor_instances:
                try:
                    factor = self._factor_instances[factor_name]
                    
                    # Get signal configuration for this factor
                    signal_config = self.factor_signals.get(factor_name, {
                        "signal_type": SignalType.STANDARD,
                        "params": {}
                    })
                    
                    # Generate signal
                    signal = factor.generate_signal(
                        data, 
                        signal_type=signal_config["signal_type"],
                        params=signal_config["params"]
                    )
                    
                    factor_signals[factor_name] = signal
                    
                except Exception as e:
                    self.logger.error(f"Error calculating signal for {factor_name}: {e}")
        
        # Update signal cache with most recent values
        self._update_signal_cache(data, factor_signals)
        
        return factor_signals
    
    def _update_signal_cache(self, data: pd.DataFrame, factor_signals: Dict[str, pd.Series]) -> None:
        """
        Update signal cache with most recent values
        
        Args:
            data: Market data
            factor_signals: Factor signals
        """
        # Get symbol
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else self.primary_symbol
        
        # Initialize cache entry if needed
        if symbol not in self._signal_cache:
            self._signal_cache[symbol] = {
                'last_update': pd.Timestamp.now(),
                'signals': {},
                'combined_strength': 0.0
            }
        
        # Update cache with latest signal values
        signals = {}
        combined_strength = 0.0
        
        for i, factor_name in enumerate(self.factors):
            if factor_name in factor_signals and not factor_signals[factor_name].empty:
                signal_value = factor_signals[factor_name].iloc[-1]
                signals[factor_name] = float(signal_value)
                combined_strength += signal_value * self.weights[i]
        
        self._signal_cache[symbol]['last_update'] = pd.Timestamp.now()
        self._signal_cache[symbol]['signals'] = signals
        self._signal_cache[symbol]['combined_strength'] = float(combined_strength)
    
    def get_factor_signals(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Get current factor signals
        
        Args:
            symbol: Trading symbol (optional, returns all symbols if None)
            
        Returns:
            Dict with factor signals
        """
        if symbol:
            return {symbol: self._signal_cache.get(symbol, {})}
        return self._signal_cache
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get strategy status and configuration
        
        Returns:
            Dict with strategy status
        """
        return {
            'name': 'multi_factors',
            'description': 'Multi-Factor Strategy combining various technical indicators',
            'parameters': {
                'factors': self.factors,
                'weights': self.weights,
                'signal_threshold': self.signal_threshold
            },
            'performance': self.get_performance_stats(),
            'factor_signals': self.get_factor_signals(),
            'data_buffers': {
                symbol: len(df) for symbol, df in self._data_buffer.items()
            }
        }