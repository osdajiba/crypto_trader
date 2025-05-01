#!/usr/bin/env python3
# src/strategy/factors/adapter.py

import pandas as pd
from typing import Dict, Optional, Any, List, Union, Type, Callable

from src.common.log_manager import LogManager
from src.strategy.base import BaseStrategy
from src.strategy.factors.base import FactorBase, SignalType
from src.strategy.factors.registry import get_factor_registry


class StrategyFactorAdapter:
    """
    Adapter class to integrate the Factor library with existing strategies.
    
    This adapter allows the enhanced factor library to be used with BaseStrategy 
    implementations through a bridge pattern.
    """
    
    def __init__(self, strategy: BaseStrategy):
        """
        Initialize the adapter with the strategy instance
        
        Args:
            strategy: The strategy instance to adapt
        """
        self.strategy = strategy
        self.logger = LogManager.get_logger(f"strategy.adapter.{strategy.__class__.__name__.lower()}")
        self.factor_registry = get_factor_registry()
        self.factors = {}
    
    def register_factor_from_strategy(self, name: str, window_size: int, 
                                     func: Optional[Callable] = None,
                                     price_col: str = 'close',
                                     factor_type: str = 'custom',
                                     **kwargs) -> None:
        """
        Register a factor from the strategy's register_factor method
        
        Args:
            name: Factor name
            window_size: Factor window size
            func: Original calculation function (optional)
            price_col: Price column name
            factor_type: Type of factor ('ma', 'ema', 'rsi', etc.)
            **kwargs: Additional parameters for factor creation
        """
        try:
            # Map to appropriate factor type
            if factor_type == 'ma':
                factor = self.factor_registry.create_factor('sma', period=window_size, price_col=price_col)
            elif factor_type == 'ema':
                factor = self.factor_registry.create_factor('ema', period=window_size, price_col=price_col)
            elif factor_type == 'rsi':
                factor = self.factor_registry.create_factor('rsi', period=window_size, price_col=price_col)
            elif factor_type == 'macd':
                fast_period = kwargs.get('fast_period', 12)
                slow_period = kwargs.get('slow_period', 26)
                signal_period = kwargs.get('signal_period', 9)
                factor = self.factor_registry.create_factor('macd', fast_period=fast_period, 
                                                        slow_period=slow_period, 
                                                        signal_period=signal_period,
                                                        price_col=price_col)
            elif factor_type == 'bbands':
                std_dev = kwargs.get('std_dev', 2.0)
                factor = self.factor_registry.create_factor('bollinger', period=window_size, 
                                                         std_dev=std_dev, price_col=price_col)
            elif factor_type == 'custom':
                # For custom factors, use original function if provided
                from src.strategy.factors.custom import CustomFactorBuilder
                if func:
                    custom_factor_class = CustomFactorBuilder.create_custom_factor(
                        name, func, kwargs.get('required_columns', [price_col]), window_size
                    )
                    factor = custom_factor_class(custom_name=name)
                else:
                    # Fall back to price change rate if no function provided
                    from src.strategy.factors.custom import PriceChangeRate
                    factor = PriceChangeRate(period=window_size, price_col=price_col, name=name)
            else:
                # Try to create from registry with generic params
                factor = self.factor_registry.create_factor(factor_type, period=window_size, 
                                                         price_col=price_col, **kwargs)
            
            if factor:
                self.factors[name] = factor
                self.logger.debug(f"Registered factor '{name}' of type '{factor_type}'")
            else:
                self.logger.warning(f"Failed to create factor '{name}' of type '{factor_type}'")
                
        except Exception as e:
            self.logger.error(f"Error registering factor '{name}': {e}")
    
    def calculate_factor(self, data: pd.DataFrame, factor_name: str) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate factor values using the enhanced factor library
        
        Args:
            data: Input market data
            factor_name: Name of the factor to calculate
            
        Returns:
            Union[pd.Series, pd.DataFrame]: Factor values
        """
        if factor_name not in self.factors:
            self.logger.warning(f"Factor '{factor_name}' not registered in adapter")
            return pd.Series(index=data.index)
            
        return self.factors[factor_name].calculate(data)
    
    def generate_signal(self, data: pd.DataFrame, factor_name: str, 
                       signal_type: SignalType = SignalType.STANDARD,
                       params: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Generate signals from a factor
        
        Args:
            data: Input market data
            factor_name: Factor name
            signal_type: Signal generation type
            params: Signal generation parameters
            
        Returns:
            pd.Series: Signal series (-1 for sell, 0 for neutral, 1 for buy)
        """
        if factor_name not in self.factors:
            self.logger.warning(f"Factor '{factor_name}' not registered in adapter")
            return pd.Series(0, index=data.index)
            
        return self.factors[factor_name].generate_signal(data, signal_type, params)


def adapt_strategy_factors(strategy: BaseStrategy) -> StrategyFactorAdapter:
    """
    Create a factor adapter for a strategy instance
    
    Args:
        strategy: Strategy instance to adapt
        
    Returns:
        StrategyFactorAdapter: Factor adapter for the strategy
    """
    return StrategyFactorAdapter(strategy)