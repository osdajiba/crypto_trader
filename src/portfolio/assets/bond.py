#!/usr/bin/env python3
# src/portfolio/assets/bond.py

from src.portfolio.assets.base import Asset
from src.common.abstract_factory import register_factory_class


@register_factory_class('asset_factory', 'bond')
class Bond(Asset):
    """Bond asset"""
    
    def __init__(self, config, params):
        name = params.get('name', '')
        super().__init__(name)
        self.face_value = params.get('face_value', 0.0)
        self.coupon_rate = params.get('coupon_rate', 0.0)
        self.maturity = params.get('maturity', 0)

    def get_value(self) -> float:
        """Calculate the current value of the bond
        
        Note: In a real implementation, this would calculate present value
        based on yield curve, time to maturity, etc.
        """
        return self.face_value  # Simplified implementation

    def buy(self, amount: float):
        """Buy bond (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass

    def sell(self, amount: float):
        """Sell bond (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass