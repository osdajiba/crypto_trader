#!/usr/bin/env python3
# src/strategy/multi_factors.py

import pandas as pd
from typing import Dict, Optional, Any, List
from src.strategy.base import BaseStrategy
from src.common.config import ConfigManager
from src.common.abstract_factory import register_factory_class
from src.common.async_executor import AsyncExecutor
from src.strategy.factor_lib import IndicatorBase, RsiIndicator, MacdIndicator, BollingerBand, VolumeOscillator


class FactorFactory:
    """Factory for creating technical indicators based on configuration"""
    
    @staticmethod
    def create_factor(factor_type: str, params: Dict[str, Any]) -> IndicatorBase:
        """
        Create a factor indicator based on type and parameters
        
        Args:
            factor_type: Type of indicator to create
            params: Parameters for the indicator
            
        Returns:
            IndicatorBase: Instantiated indicator
        """
        if factor_type == 'rsi':
            period = params.get('period', 14)
            return RsiIndicator(period=period)
            
        elif factor_type == 'macd':
            fast_period = params.get('fast_period', 12)
            slow_period = params.get('slow_period', 26)
            signal_period = params.get('signal_period', 9)
            return MacdIndicator(fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)
            
        elif factor_type == 'bollinger':
            period = params.get('period', 20)
            std_dev = params.get('std_dev', 2.0)
            return BollingerBand(period=period, std_dev=std_dev)
            
        elif factor_type == 'volume_osc':
            fast_period = params.get('fast_period', 5)
            slow_period = params.get('slow_period', 14)
            return VolumeOscillator(fast_period=fast_period, slow_period=slow_period)
            
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")


@register_factory_class('strategy_factory', 'multi_factors', 
                       description="Multi-Factor Trading Strategy",
                       category="adaptive",
                       parameters=["factor_config"])
class MultiFactorsStrategy(BaseStrategy):
    """
    Implementation of a Multi-Factor strategy that generates trading signals
    based on a combination of configurable technical indicators.
    
    Enhanced with factor registration and efficient data management.
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        Initialize Multi-Factor Strategy
        
        Args:
            config (ConfigManager): Configuration manager instance
            params (Optional[Dict[str, Any]]): Strategy-specific parameters
                Expected parameters: factor_config, threshold, symbol, lookback_period
        """
        super().__init__(config, params)
        
        # Initialize parameters
        self.threshold = self.params.get("threshold", 0.5)
        self.primary_symbol = self.params.get("symbol", "unknown")
        
        # Get factor configuration
        self.factor_config = self.params.get("factor_config", {})
        if not self.factor_config:
            # Use default config from config file if not specified in params
            self.factor_config = self.config.get("strategy", "factors", default={})
        
        # Async executor
        self.executor = AsyncExecutor()
        self._running_tasks: List[str] = []
        
        # Create indicators
        self.indicators = self._create_indicators()
        
        # Register factors during initialization
        self._register_default_factors()

    def _create_indicators(self) -> Dict[str, IndicatorBase]:
        """
        Create indicator instances based on configuration
        
        Returns:
            Dict[str, IndicatorBase]: Mapping of indicator names to instances
        """
        indicators = {}
        
        for factor_name, factor_config in self.factor_config.items():
            factor_type = factor_config.get('type')
            factor_params = factor_config.get('params', {})
            
            try:
                indicator = FactorFactory.create_factor(factor_type, factor_params)
                indicators[factor_name] = indicator
                self.logger.debug(f"Created indicator {factor_name} of type {factor_type}")
            except Exception as e:
                self.logger.error(f"Failed to create indicator {factor_name}: {e}")
        
        return indicators

    def _register_default_factors(self) -> None:
        """
        Register technical factors based on configuration
        """
        # Register all configured factors
        for factor_name, indicator in self.indicators.items():
            factor_config = self.factor_config.get(factor_name, {})
            window_size = factor_config.get('window_size', 0)
            is_differential = factor_config.get('is_differential', False)
            
            # Get dependencies
            depends_on = factor_config.get('depends_on', [])
            
            # Generate factor calculation function
            def create_calc_func(indicator, factor_name):
                def calc_func(data, **kwargs):
                    result = indicator.calculate(data)
                    
                    # Handle different return types (Series or DataFrame)
                    if isinstance(result, pd.DataFrame):
                        if factor_name in result.columns:
                            return result[factor_name]
                        else:
                            # For indicators that return multiple columns (like MACD)
                            # Use the first column by default
                            return result.iloc[:, 0]
                    return result
                return calc_func
            
            # Register the factor
            self.register_factor(
                factor_name, 
                window_size, 
                create_calc_func(indicator, factor_name),
                depends_on=depends_on,
                is_differential=is_differential
            )
            
            self.logger.debug(f"Registered factor: {factor_name}")
        
        # Register composite factor that combines all other factors
        factor_names = list(self.indicators.keys())
        self.register_composite_factor(factor_names)
        
        self.logger.info(f"Registered {len(self.indicators)} factors and a composite signal")

    def register_composite_factor(self, factor_names: List[str]) -> None:
        """
        Register a composite factor that combines individual factors
        
        Args:
            factor_names: List of factor names to combine
        """
        def calculate_composite(data: pd.DataFrame, **kwargs) -> pd.Series:
            # Get factor weights from config
            weights = {}
            scores = pd.Series(0, index=data.index)
            
            for factor_name in factor_names:
                if factor_name not in kwargs:
                    continue
                    
                factor_config = self.factor_config.get(factor_name, {})
                weight = factor_config.get('weight', 1.0)
                weights[factor_name] = weight
                
                factor_value = kwargs[factor_name]
                
                # Apply normalization if configured
                normalize = factor_config.get('normalize', True)
                if normalize:
                    # Simple min-max normalization
                    min_val = factor_value.min()
                    max_val = factor_value.max()
                    if max_val > min_val:
                        factor_value = (factor_value - min_val) / (max_val - min_val)
                
                # Apply signals based on factor type
                signal_type = factor_config.get('signal_type', 'standard')
                
                if signal_type == 'crossover':
                    # Zero crossover
                    signal = ((factor_value > 0) & (factor_value.shift(1) <= 0)).astype(float)
                    signal -= ((factor_value < 0) & (factor_value.shift(1) >= 0)).astype(float)
                
                elif signal_type == 'threshold':
                    upper = factor_config.get('upper_threshold', 0.7)
                    lower = factor_config.get('lower_threshold', 0.3)
                    signal = ((factor_value < lower)).astype(float)  # Bullish when below lower
                    signal -= ((factor_value > upper)).astype(float)  # Bearish when above upper
                
                elif signal_type == 'momentum':
                    signal = factor_value.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
                
                else:  # standard
                    signal = factor_value
                
                scores += signal * weight
            
            # Normalize final score if we have weights
            if weights:
                total_weight = sum(weights.values())
                if total_weight > 0:
                    scores = scores / total_weight
            
            return scores
        
        # Register composite factor with dependencies on all individual factors
        self.register_factor('composite_signal', 1, calculate_composite, depends_on=factor_names)

    async def initialize(self) -> None:
        """Initialize resources and validate parameters"""
        # Start executor
        await self.executor.start()
        
        # Parent class initialization
        await super().initialize()
        
        # Check that we have at least one factor
        if not self.indicators:
            self.logger.error("No valid factors configured")
            raise ValueError("At least one valid factor must be configured")
        
        # Calculate required lookback from factors
        required_lookback = max(
            [config.get('window_size', 30) for config in self.factor_config.values()],
            default=30
        ) * 2
        
        # Adjust lookback period if needed
        if self.lookback_period < required_lookback:
            self.logger.warning(f"lookback_period ({self.lookback_period}) is less than recommended ({required_lookback}), adjusting")
            self.lookback_period = required_lookback
            
        self.logger.info(f"Initialized MultiFactorsStrategy with {len(self.indicators)} factors")

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            data (pd.DataFrame): Market data for signal generation (e.g. OHLCV)

        Returns:
            pd.DataFrame: Signals containing 'timestamp', 'symbol', 'action' (buy/sell) columns
        """
        self.logger.info("Generating signals for MultiFactorsStrategy")
        
        if data.empty:
            self.logger.warning("Empty data, cannot generate signals")
            return pd.DataFrame()
        
        # Get the symbol
        symbol = self.primary_symbol
        if '_symbol' in data.columns:
            if not data.empty:
                symbol = data['_symbol'].iloc[0]
        
        # Calculate composite signal
        composite = self.calculate_factor(data, 'composite_signal', symbol)
        
        # Create signal DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['symbol'] = symbol
        signals['action'] = None  # Initialize with None
        
        # Generate buy/sell signals based on composite score
        buy_threshold = self.threshold
        sell_threshold = -self.threshold
        
        # Buy signal when composite score crosses above threshold
        buy_signal = (composite > buy_threshold) & (composite.shift(1) <= buy_threshold)
        signals.loc[buy_signal, 'action'] = 'buy'
        
        # Sell signal when composite score crosses below negative threshold
        sell_signal = (composite < sell_threshold) & (composite.shift(1) >= sell_threshold)
        signals.loc[sell_signal, 'action'] = 'sell'
        
        # Drop rows with no action
        signals = signals.dropna(subset=['action'])
        
        # Add price information if needed for order sizing
        if 'close' in data.columns:
            signals['price'] = data.loc[signals.index, 'close']
        
        # Calculate position size based on available capital
        if 'price' in signals.columns:
            for idx, row in signals.iterrows():
                if row['action'] == 'buy':
                    signals.at[idx, 'quantity'] = self.calculate_position_size(row['price'], 
                                                                          self.config.get("trading", "capital", "initial", default=100000))
        
        self.logger.info(f"Generated {len(signals)} signals for {symbol}")
        return signals
    
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        Calculate position size based on price and available capital
        
        Args:
            price (float): Current asset price
            capital (float): Available capital
            
        Returns:
            float: Position size
        """
        # Get risk settings from params or config
        risk_per_trade = self.params.get("risk_per_trade", 0.01)  # Default 1% risk
        max_position = self.config.get("trading", "limits", "position", default=0.1)  # Default max 10%
        
        # Calculate position size
        risk_amount = capital * risk_per_trade
        max_amount = capital * max_position
        
        # Use smaller of the two values
        position_value = min(risk_amount, max_amount)
        
        # Convert to quantity
        quantity = position_value / price if price > 0 else 0
        
        return quantity