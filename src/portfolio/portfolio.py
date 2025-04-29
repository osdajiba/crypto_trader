#!/usr/bin/env python3
# src/portfolio/portfolio.py

import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union

import pandas as pd
from src.common.abstract_factory import register_factory_class
from src.common.log_manager import LogManager
from src.portfolio.assets.tradable_asset import TradableAsset
from src.portfolio.execution.order import Direction, OrderStatus


logger = LogManager.get_logger("portfolio")


@register_factory_class('asset_factory', 'portfolio')
class Portfolio(TradableAsset):
    """
    Portfolio asset that contains other assets
    
    This allows for hierarchical portfolio structures where a portfolio
    can contain other portfolios as assets.
    """
    
    def __init__(self, config, params):
        """
        Initialize portfolio asset
        
        Args:
            config: Configuration manager
            params: Parameters including:
                name: Portfolio name
                exchange: Exchange interface (optional)
                sub_portfolio_manager: Sub-portfolio manager (optional)
        """
        name = params.get('name', 'Portfolio')
        exchange = params.get('exchange', None)
        super().__init__(name, exchange, config, params)
        
        # Get sub-portfolio manager if provided, otherwise create a new one
        from .manager import PortfolioManager
        self.sub_portfolio = params.get('sub_portfolio_manager', None)
        if not self.sub_portfolio:
            self.sub_portfolio = PortfolioManager(config, exchange)
        
        # Portfolio specific properties
        self.description = params.get('description', '')
        self.risk_level = params.get('risk_level', 'moderate')
        self.base_currency = params.get('base_currency', 'USDT')

    def get_value(self) -> float:
        """
        Calculate the total value of the sub-portfolio
        
        Returns:
            float: Total value of all assets in the sub-portfolio
        """
        return float(Decimal(str(self.sub_portfolio.get_total_value())))

    async def update_value(self) -> float:
        """
        Update and return current portfolio value by updating all contained assets
        
        Returns:
            float: Updated portfolio value
        """
        # Update values of all assets in the portfolio
        await self.sub_portfolio.update_all_values()
        
        # Get the updated total value
        total_value = self.sub_portfolio.get_total_value()
        self._value = Decimal(str(total_value))
        
        return float(self._value)

    def _update_position_from_filled_order(self, order):
        """
        Update portfolio position based on filled order
        
        For a Portfolio, this is a no-op as position is calculated from sub-assets
        """
        # Nothing to do - portfolio value is calculated from sub-assets
        pass

    async def buy(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy operation for the portfolio
        
        For a portfolio, this might involve buying assets according to 
        target allocation weights.
        
        Args:
            amount: Amount to invest across portfolio
            **kwargs: Additional parameters:
                asset_allocations: Dictionary mapping asset names to allocation percentages
                order_type: Order type (market, limit, etc.)
                
        Returns:
            Dict[str, Any]: Transaction results
        """
        # Extract parameters
        asset_allocations = kwargs.get('asset_allocations', None)
        
        if not asset_allocations:
            # If no specific allocations provided, use current weights or equal weight
            allocations = self.get_asset_weights()
            
            # If no current positions, use equal weighting
            if not allocations:
                assets = self.sub_portfolio.list_assets()
                if not assets:
                    logger.warning(f"No assets in {self.name} portfolio to allocate {amount}")
                    return {'success': False, 'error': 'No assets in portfolio'}
                
                # Equal weight allocation
                weight = 1.0 / len(assets)
                allocations = {asset: weight for asset in assets}
        else:
            allocations = asset_allocations
        
        # Normalize allocations to ensure they sum to 100%
        total_allocation = sum(allocations.values())
        if total_allocation <= 0:
            logger.error(f"Invalid allocation weights: {allocations}")
            return {'success': False, 'error': 'Invalid allocation weights'}
            
        normalized_allocations = {k: v / total_allocation for k, v in allocations.items()}
        
        # Execute buys according to allocations
        results = {}
        amount_decimal = Decimal(str(amount))
        
        for asset_name, allocation in normalized_allocations.items():
            if asset_name not in self.sub_portfolio.assets:
                logger.warning(f"Asset {asset_name} not found in portfolio, skipping allocation")
                continue
                
            # Calculate amount to allocate to this asset
            asset_amount = float(amount_decimal * Decimal(str(allocation)))
            
            try:
                # Buy the asset
                asset_result = await self.sub_portfolio.buy_asset(asset_name, asset_amount, **kwargs)
                results[asset_name] = asset_result
            except Exception as e:
                logger.error(f"Error buying {asset_name}: {str(e)}")
                results[asset_name] = {'success': False, 'error': str(e)}
        
        return {
            'success': True,
            'amount': float(amount_decimal),
            'allocations': {k: float(v) for k, v in normalized_allocations.items()},
            'results': results
        }

    async def sell(self, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell operation for the portfolio
        
        For a portfolio, this might involve selling assets according to
        target allocation weights or rebalancing rules.
        
        Args:
            amount: Amount to divest from portfolio
            **kwargs: Additional parameters:
                proportional: Whether to sell proportionally across assets (default True)
                from_assets: List of specific assets to sell from
                
        Returns:
            Dict[str, Any]: Transaction results
        """
        # Extract parameters
        proportional = kwargs.get('proportional', True)
        from_assets = kwargs.get('from_assets', None)
        
        total_value = Decimal(str(self.get_value()))
        amount_decimal = Decimal(str(amount))
        
        # Check if we're trying to sell more than we have
        if amount_decimal > total_value:
            logger.warning(f"Attempting to sell {float(amount_decimal)} from portfolio "
                         f"worth {float(total_value)}, reducing to available amount")
            amount_decimal = total_value
        
        # If no specific assets provided, sell proportionally from all
        if proportional and not from_assets:
            # Calculate current weights
            weights = self.sub_portfolio.get_asset_weights()
            
            # Sell according to weights
            results = {}
            for asset_name, weight in weights.items():
                asset_amount = float(amount_decimal * Decimal(str(weight)))
                
                try:
                    # Sell the asset
                    asset_result = await self.sub_portfolio.sell_asset(asset_name, asset_amount, **kwargs)
                    results[asset_name] = asset_result
                except Exception as e:
                    logger.error(f"Error selling {asset_name}: {str(e)}")
                    results[asset_name] = {'success': False, 'error': str(e)}
            
            return {
                'success': True,
                'amount': float(amount_decimal),
                'proportional': True,
                'results': results
            }
        elif from_assets:
            # Sell from specific assets
            results = {}
            remaining_amount = amount_decimal
            
            for asset_name in from_assets:
                if asset_name not in self.sub_portfolio.assets:
                    logger.warning(f"Asset {asset_name} not found in portfolio, skipping")
                    continue
                
                asset = self.sub_portfolio.assets[asset_name]
                asset_value = Decimal(str(asset.get_value()))
                
                # Determine how much to sell from this asset
                asset_amount = min(asset_value, remaining_amount)
                if asset_amount <= 0:
                    continue
                
                try:
                    # Sell the asset
                    asset_result = await self.sub_portfolio.sell_asset(asset_name, float(asset_amount), **kwargs)
                    results[asset_name] = asset_result
                    remaining_amount -= asset_amount
                except Exception as e:
                    logger.error(f"Error selling {asset_name}: {str(e)}")
                    results[asset_name] = {'success': False, 'error': str(e)}
                
                # Stop if we've sold enough
                if remaining_amount <= 0:
                    break
            
            return {
                'success': True,
                'amount': float(amount_decimal - remaining_amount),
                'proportional': False,
                'from_assets': from_assets,
                'results': results
            }
        else:
            # Sell from most liquid assets first (implementation dependent on asset types)
            logger.warning(f"Non-proportional selling without specified assets not implemented")
            return {'success': False, 'error': 'Method not implemented'}
        
    def add_asset(self, asset: TradableAsset) -> None:
        """
        Add an asset to the sub-portfolio
        
        Args:
            asset: Asset to add
        """
        self.sub_portfolio.add_asset(asset)
        
    def create_asset(self, asset_type: str, params: dict) -> TradableAsset:
        """
        Create and add an asset to the sub-portfolio
        
        Args:
            asset_type: Type of asset to create
            params: Parameters for asset creation
            
        Returns:
            Asset: The created asset
        """
        return self.sub_portfolio.create_asset(asset_type, params)
        
    def get_asset_weights(self) -> Dict[str, float]:
        """
        Get the weight of each asset in the sub-portfolio
        
        Returns:
            Dict[str, float]: Dictionary mapping asset names to their weight percentages
        """
        return self.sub_portfolio.get_asset_weights()
    
    def get_allocation_by_type(self) -> Dict[str, float]:
        """
        Get allocation breakdown by asset type
        
        Returns:
            Dict[str, float]: Dictionary mapping asset types to their allocation percentages
        """
        # Get total portfolio value
        total_value = Decimal(str(self.get_value()))
        if total_value == 0:
            return {}
        
        # Group by asset type
        type_values = {}
        
        for asset_name, asset in self.sub_portfolio.assets.items():
            asset_type = asset.__class__.__name__
            asset_value = Decimal(str(asset.get_value()))
            
            if asset_type not in type_values:
                type_values[asset_type] = Decimal('0')
            
            type_values[asset_type] += asset_value
        
        # Calculate percentages
        return {asset_type: float(value / total_value) for asset_type, value in type_values.items()}
    
    async def rebalance(self, target_allocations: Dict[str, float]) -> Dict[str, Any]:
        """
        Rebalance portfolio to target allocations
        
        Args:
            target_allocations: Dictionary mapping asset names to target allocation percentages
            
        Returns:
            Dict[str, Any]: Rebalancing results
        """
        # First, update all asset values
        await self.update_value()
        
        # Get current values and allocations
        total_value = Decimal(str(self.get_value()))
        if total_value == 0:
            logger.warning("Cannot rebalance empty portfolio")
            return {'success': False, 'error': 'Empty portfolio'}
        
        current_allocations = self.sub_portfolio.get_asset_weights()
        
        # Normalize target allocations
        target_sum = sum(target_allocations.values())
        normalized_targets = {k: Decimal(str(v)) / Decimal(str(target_sum)) 
                             for k, v in target_allocations.items()}
        
        # Calculate trades needed
        trades = []
        for asset_name, target_weight in normalized_targets.items():
            current_weight = Decimal(str(current_allocations.get(asset_name, 0)))
            
            # Calculate target value
            target_value = total_value * target_weight
            
            # Get current asset value
            current_value = Decimal('0')
            if asset_name in self.sub_portfolio.assets:
                current_value = Decimal(str(self.sub_portfolio.assets[asset_name].get_value()))
            
            # Calculate difference
            difference = target_value - current_value
            
            if abs(difference) > Decimal('0.01'):  # Tolerance for small differences
                trades.append({
                    'asset': asset_name,
                    'current_value': float(current_value),
                    'current_weight': float(current_weight),
                    'target_value': float(target_value),
                    'target_weight': float(target_weight),
                    'difference': float(difference),
                    'action': 'buy' if difference > 0 else 'sell',
                    'amount': abs(float(difference))
                })
        
        # Execute rebalancing trades
        results = []
        for trade in trades:
            try:
                if trade['action'] == 'buy':
                    # Check if asset exists, create if not
                    if trade['asset'] not in self.sub_portfolio.assets:
                        # Would need logic to determine asset type and parameters
                        logger.warning(f"Asset {trade['asset']} not found for rebalancing buy")
                        continue
                    
                    # Buy asset
                    result = await self.sub_portfolio.buy_asset(trade['asset'], trade['amount'])
                    
                else:  # sell
                    if trade['asset'] not in self.sub_portfolio.assets:
                        logger.warning(f"Asset {trade['asset']} not found for rebalancing sell")
                        continue
                    
                    # Sell asset
                    result = await self.sub_portfolio.sell_asset(trade['asset'], trade['amount'])
                
                trade['result'] = result
                results.append(trade)
                
            except Exception as e:
                logger.error(f"Error rebalancing {trade['asset']}: {str(e)}")
                trade['error'] = str(e)
                trade['success'] = False
                results.append(trade)
        
        return {
            'success': True,
            'initial_value': float(total_value),
            'initial_allocations': {k: float(v) for k, v in current_allocations.items()},
            'target_allocations': {k: float(v) for k, v in normalized_targets.items()},
            'trades': results
        }
    
    async def execute_signal_batch(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a batch of trading signals across the portfolio
        
        Args:
            signals: DataFrame with signal information
            
        Returns:
            Dict[str, Any]: Execution results
        """
        # Use sub-portfolio manager to execute signals
        return await self.sub_portfolio.execute_signal_batch(signals)
    
    async def sync_with_exchange(self) -> Dict[str, Any]:
        """
        Sync all sub-assets with exchange
        
        Returns:
            Dict[str, Any]: Sync results
        """
        return await self.sub_portfolio.sync_with_exchange()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert portfolio to dictionary representation including sub-assets
        
        Returns:
            Dict[str, Any]: Portfolio as dictionary
        """
        # Get base asset information
        base_info = super().to_dict()
        
        # Add portfolio specific information
        portfolio_info = {
            'description': self.description,
            'risk_level': self.risk_level,
            'base_currency': self.base_currency,
            'assets': {name: asset.to_dict() for name, asset in self.sub_portfolio.assets.items()},
            'total_value': self.get_value(),
            'asset_count': len(self.sub_portfolio.assets)
        }
        
        # Merge dictionaries
        return {**base_info, **portfolio_info}
    
    def close(self):
        """
        Close the portfolio and its assets
        """
        # Close all sub-assets
        for asset in self.sub_portfolio.assets.values():
            if hasattr(asset, 'close') and callable(asset.close):
                asset.close()
        
        # Close sub-portfolio
        asyncio.create_task(self.sub_portfolio.close())
        
        # Call parent close method
        super().close()