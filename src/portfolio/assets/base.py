# src/portfolio/asset.py

from abc import ABC, abstractmethod

class Asset(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_value(self) -> float:
        """返回资产的当前价值"""
        pass

    @abstractmethod
    def buy(self, amount: float):
        """买入资产"""
        pass

    @abstractmethod
    def sell(self, amount: float):
        """卖出资产"""
        pass