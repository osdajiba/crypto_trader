# src/common/enums.py

from enum import Enum

class TradingMode(Enum):
    """集中定义交易模式类型"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"