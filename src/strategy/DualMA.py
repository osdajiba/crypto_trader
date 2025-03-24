# src/strategy/DualMA.py

import pandas as pd
from typing import Dict, Optional, Any, List
from src.strategy.base_strategy import BaseStrategy
from src.common.config_manager import ConfigManager
from src.common.abstract_factory import register_factory_class
from src.common.async_executor import AsyncExecutor


@register_factory_class('strategy_factory', 'dual_ma', 
                       description="Dual Moving Average Crossover Strategy",
                       category="trend",
                       parameters=["short_window", "long_window"])
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
            config (ConfigManager): Configuration manager instance
            params (Optional[Dict[str, Any]]): Strategy-specific parameters
                Expected parameters: short_window, long_window, symbol, multi_symbol_data, lookback_period, period
        """
        super().__init__(config, params)
        self.short_window = self.params.get("short_window", 10)
        self.long_window = self.params.get("long_window", 30)
        self.primary_symbol = self.params.get("symbol", "unknown")
        self.multi_symbol_data = self.params.get("multi_symbol_data", {})
        self.executor = AsyncExecutor()  # Get singleton instance
        self._running_tasks: List[str] = []
        
        # Register factors during initialization
        self._register_default_factors()

    def _register_default_factors(self) -> None:
        """
        Register MA factors based on strategy parameters
        """
        # Register short MA factor
        self.register_ma_factor('short_ma', self.short_window)
        
        # Register long MA factor
        self.register_ma_factor('long_ma', self.long_window)
        
        # Log registration
        self.logger.debug(f"Registered factors: short_ma (window={self.short_window}), long_ma (window={self.long_window})")

    def register_ma_factor(self, name: str, window_size: int) -> None:
        """
        Register a moving average factor
        
        Args:
            name: Factor name
            window_size: Window size for the moving average
        """
        # Define MA calculation function
        def calculate_ma(data: pd.DataFrame, **kwargs) -> pd.Series:
            if 'close' in data.columns:
                return data['close'].rolling(window=window_size).mean()
            return pd.Series(index=data.index)
        
        # Register the factor with the base class
        self.register_factor(name, window_size, calculate_ma)

    async def initialize(self) -> None:
        """Initialize resources and validate parameters"""
        # Start executor
        await self.executor.start()
        
        # Parent class initialization
        await super().initialize()
        
        # Parameter validation
        if self.short_window >= self.long_window:
            self.logger.error(f"short_window ({self.short_window}) must be less than long_window ({self.long_window})")
            raise ValueError("short_window must be less than long_window")
        
        # Adjust lookback period if needed
        min_lookback = self.long_window * 2  # At least double the long window for meaningful signals
        if self.lookback_period < min_lookback:
            self.logger.warning(f"lookback_period ({self.lookback_period}) is less than recommended ({min_lookback}), adjusting")
            self.lookback_period = min_lookback
            
        self.logger.info(f"Initialized DualMAStrategy with short_window={self.short_window}, long_window={self.long_window}")

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on market data
        
        Args:
            data (pd.DataFrame): Market data for signal generation (e.g. OHLCV)

        Returns:
            pd.DataFrame: Signals containing 'timestamp', 'symbol', 'action' (buy/sell) columns
        """        
        if data.empty:
            self.logger.warning("Empty data, cannot generate signals")
            return pd.DataFrame()
        
        # Get the symbol
        symbol = self.primary_symbol
        if '_symbol' in data.columns:
            # If data has symbol column, use the first one (assuming it's all the same)
            if not data.empty:
                symbol = data['_symbol'].iloc[0]
        
        # Calculate moving averages using optimized factor calculation
        short_ma = self.calculate_factor(data, 'short_ma', symbol)
        long_ma = self.calculate_factor(data, 'long_ma', symbol)
        
        # Create signal DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['timestamp'] = data.index
        signals['symbol'] = symbol
        signals['action'] = None  # Initialize with None
        
        # Buy signal: short MA crosses above long MA
        crossover_up = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        signals.loc[crossover_up, 'action'] = 'buy'
        
        # Sell signal: short MA crosses below long MA
        crossover_down = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        signals.loc[crossover_down, 'action'] = 'sell'
        
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