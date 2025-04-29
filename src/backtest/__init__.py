#!/usr/bin/env python3
# src/backtest/__init__.py

"""
Backtest package for trading strategy evaluation.

This package provides components for backtesting trading strategies
with different engines:
- OHLCV Engine: Vectorized backtesting using OHLCV data
- Market Replay Engine: Sequential data processing for realistic simulation

Usage:
    from src.backtest.factory import get_backtest_factory
    from src.backtest.mode import BacktestMode
    from src.common.config import ConfigManager
    
    # Create backtest mode
    config = ConfigManager('config.yaml')
    backtest_mode = BacktestMode(config)
    
    # Or create a specific engine directly
    factory = get_backtest_factory(config)
    engine = await factory.create_engine('ohlcv', {'strategy': 'dual_ma'})
"""

from src.backtest.factory import get_backtest_factory, BacktestFactory
from src.backtest.mode import BacktestMode

# Engine imports
from src.backtest.engine.base import BaseBacktestEngine
from src.backtest.engine.ohlcv import OHLCVEngine
from src.backtest.engine.market_replay import MarketReplayEngine