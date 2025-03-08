# src/portfolio/future.py

from src.portfolio.asset import Asset

class Future(Asset):
    def __init__(self, name: str, contract_size: int, price: float):
        super().__init__(name)
        self.contract_size = contract_size  # 合约规模
        self.price = price  # 当前价格

    def get_value(self) -> float:
        # 计算期货当前价值
        return self.contract_size * self.price

    def buy(self, amount: float):
        # 买入期货逻辑（可根据实际需求实现）
        pass

    def sell(self, amount: float):
        # 卖出期货逻辑（可根据实际需求实现）
        pass