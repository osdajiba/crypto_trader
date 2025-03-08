# src/portfolio/bond.py

from src.portfolio.asset import Asset

class Bond(Asset):
    def __init__(self, name: str, face_value: float, coupon_rate: float, maturity: int):
        super().__init__(name)
        self.face_value = face_value  # 面值
        self.coupon_rate = coupon_rate  # 票面利率
        self.maturity = maturity  # 到期时间（年）

    def get_value(self) -> float:
        # 计算当前债券价值（可根据实际需求补充具体公式）
        return self.face_value  # 简单示例，返回面值

    def buy(self, amount: float):
        # 买入债券逻辑（可根据实际需求实现）
        pass

    def sell(self, amount: float):
        # 卖出债券逻辑（可根据实际需求实现）
        pass