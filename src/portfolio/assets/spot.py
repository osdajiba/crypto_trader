# src/portfolio/spot.py

from .base import Asset

class Spot(Asset):
    def __init__(self, name: str, quantity: float, price: float):
        super().__init__(name)
        self.quantity = quantity  # 持有数量
        self.price = price  # 当前价格

    def get_value(self) -> float:
        # 计算现货当前价值
        return self.quantity * self.price

    def buy(self, amount: float):
        # 买入现货逻辑（可根据实际需求实现）
        pass

    def sell(self, amount: float):
        # 卖出现货逻辑（可根据实际需求实现）
        pass