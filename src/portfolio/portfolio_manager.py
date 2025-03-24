# src/portfolio/portfolio_manager.py

from typing import Dict, List
from src.portfolio.asset import Asset

class PortfolioManager:
    def __init__(self):
        self.assets: Dict[str, Asset] = {}

    def add_asset(self, asset: Asset):
        if asset.name in self.assets:
            raise ValueError(f"Asset '{asset.name}' already exists")
        self.assets[asset.name] = asset

    def remove_asset(self, asset_name: str):
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        del self.assets[asset_name]

    def get_total_value(self) -> float:
        return sum(asset.get_value() for asset in self.assets.values())

    def buy_asset(self, asset_name: str, amount: float):
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        self.assets[asset_name].buy(amount)

    def sell_asset(self, asset_name: str, amount: float):
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        self.assets[asset_name].sell(amount)

    def list_assets(self) -> List[str]:
        return list(self.assets.keys())

    # extended functions
    def get_asset_weights(self) -> Dict[str, float]:
        total_value = self.get_total_value()
        if total_value == 0:
            return {}
        return {name: asset.get_value() / total_value for name, asset in self.assets.items()}

    def calculate_portfolio_risk(self, method: str = 'var') -> float:
        # Implement risk calculation logic
        pass

    def get_portfolio_return(self, start_date: str, end_date: str) -> float:
        # Implement yield calculation logic
        pass