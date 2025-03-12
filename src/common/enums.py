# src/common/enums.py

from enum import Enum

class TradingMode(Enum):
    """Centralize the definition of transaction mode types"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"