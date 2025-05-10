#!/usr/bin/env python3
# src/portfolio/assets/fund.py


from src.portfolio.assets.base import Asset
from src.common.abstract_factory import register_factory_class


@register_factory_class('asset_factory', 'fund')
class Fund(Asset):
    """Fund asset (mutual funds, ETFs, etc.)"""
    
    def __init__(self, config, params):
        name = params.get('name', '')
        super().__init__(name)
        self.units = params.get('units', 0.0)
        self.nav = params.get('nav', 0.0)  # Net Asset Value

    def get_value(self) -> float:
        """Calculate the current value of the fund"""
        return self.units * self.nav

    def buy(self, amount: float):
        """Buy fund units (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass

    def sell(self, amount: float):
        """Sell fund units (implementation depends on business logic)"""
        # This would be implemented with actual trading logic
        pass