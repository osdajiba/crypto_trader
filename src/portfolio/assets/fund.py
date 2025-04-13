# src/portfolio/fund.py

from .base import Asset

class Fund(Asset):
    def __init__(self, name: str, units: float, nav: float):
        super().__init__(name)
        self.units = units  # 持有单位数
        self.nav = nav  # 净资产值 (Net Asset Value)

    def get_value(self) -> float:
        # 计算基金当前价值
        return self.units * self.nav

    def buy(self, amount: float):
        # 买入基金逻辑（可根据实际需求实现）
        pass

    def sell(self, amount: float):
        # 卖出基金逻辑（可根据实际需求实现）
        pass