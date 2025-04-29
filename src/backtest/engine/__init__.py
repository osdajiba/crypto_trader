#!/usr/bin/env python3
# src/backtest/engine/__init__.py

"""
Backtest engines for different backtesting strategies.

Available engines:
- BaseBacktestEngine: Core backtest functionality with efficient data management
- OHLCVEngine: Vectorized backtesting using OHLCV data
- MarketReplayEngine: Sequential simulation for realistic market conditions
"""

from src.backtest.engine.base import BaseBacktestEngine
from src.backtest.engine.ohlcv import OHLCVEngine
from src.backtest.engine.market_replay import MarketReplayEngine