# src/portfolio/portfolio.py

from src.portfolio.asset import Asset
from src.portfolio.portfolio_manager import PortfolioManager

class Portfolio(Asset):
    def __init__(self, name: str):
        super().__init__(name)
        self.sub_portfolio = PortfolioManager()  # 子投资组合管理器

    def get_value(self) -> float:
        # 计算子投资组合的总价值
        return self.sub_portfolio.get_total_value()

    def buy(self, amount: float):
        # 买入子投资组合逻辑（可根据实际需求实现）
        pass

    def sell(self, amount: float):
        # 卖出子投资组合逻辑（可根据实际需求实现）
        pass