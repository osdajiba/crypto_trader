#!/usr/bin/env python3
# src/strategy/factors/__init__.py

from src.strategy.factors.base import BaseFactor, SignalType
from src.strategy.factors.factory import FactorFactory
from src.strategy.factors.momentum import RSI, MACD, Stochastic
from src.strategy.factors.volatility import BollingerBands, ATR, ADX
from src.strategy.factors.volume import OBV, MoneyFlowIndex, VolumeOscillator
from src.strategy.factors.trend import SMA, EMA, IchimokuCloud

# Export all factor classes for easy imports
__all__ = [
    # Base classes
    'BaseFactor',
    'SignalType',
    'FactorFactory',
    
    # Momentum indicators
    'RSI',
    'MACD',
    'Stochastic',
    
    # Volatility indicators
    'BollingerBands',
    'ATR',
    'ADX',
    
    # Volume indicators
    'OBV',
    'MoneyFlowIndex',
    'VolumeOscillator',
    
    # Trend indicators
    'SMA',
    'EMA',
    'IchimokuCloud'
]