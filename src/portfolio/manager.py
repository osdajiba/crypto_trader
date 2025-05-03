#!/usr/bin/env python3
# src/portfolio/manager.py

from enum import Enum
import time
import asyncio
from decimal import Decimal
from typing import Dict, List, Any, Optional
import pandas as pd

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.assets.base import Asset
from src.portfolio.assets.factory import *
from src.exchange.factory import get_exchange_factory

    
logger = LogManager.get_logger("portfolio.manager")


class TradableAsset(Enum):
    """Centralize the definition of backtest engine types"""
    SPOT = "spot"
    FUTURE = "future"
    

class PortfolioManager:
    """
    Manager for handling a collection of assets in a portfolio
    with exchange integration and order execution capabilities
    """
    
    def __init__(self, config: Optional[ConfigManager] = None, exchange=None):
        """
        Initialize portfolio manager
        
        Args:
            config: Configuration manager (optional)
            exchange: Exchange interface (optional)
        """
        self.assets: Dict[str, Asset] = {}
        self.config = config if config else ConfigManager()
        self.exchange = exchange if exchange is not None else get_exchange_factory(config)
        
        # Initialize asset factory
        self.asset_factory = AssetFactory.instance(self.config)
        
        # Auto-discover all asset types
        self.asset_factory.discover_assets()
        
        # Portfolio state tracking
        self._total_value = Decimal('0')
        self._last_update_time = 0
        self._is_syncing = False
        
        # Order tracking
        self._all_orders: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Portfolio manager initialized")

    def add_asset(self, asset: Asset) -> None:
        """
        Add an asset to the portfolio
        
        Args:
            asset: Asset instance to add
            
        Raises:
            ValueError: If asset with the same name already exists
        """
        if asset.name in self.assets:
            raise ValueError(f"Asset '{asset.name}' already exists")
        
        # Set the exchange if asset doesn't have one
        if not asset.exchange and self.exchange:
            asset.set_exchange(self.exchange)
            
        self.assets[asset.name] = asset
        logger.info(f"Added asset {asset.name} to portfolio")

    def create_asset(self, asset_type: str, params: dict) -> Asset:
        """
        Create and add a new asset to the portfolio
        
        Args:
            asset_type: Type of asset to create
            params: Parameters for asset creation
            
        Returns:
            Asset: The created asset
            
        Raises:
            ValueError: If asset with the same name already exists
        """
        # Add exchange to parameters if available
        if self.exchange and 'exchange' not in params:
            params['exchange'] = self.exchange
            
        # Create the asset
        asset = self.asset_factory.create_asset(asset_type, params)
        
        # Add to portfolio
        self.add_asset(asset)
        return asset

    def remove_asset(self, asset_name: str) -> None:
        """
        Remove an asset from the portfolio
        
        Args:
            asset_name: Name of the asset to remove
            
        Raises:
            ValueError: If asset doesn't exist
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
            
        del self.assets[asset_name]
        logger.info(f"Removed asset {asset_name} from portfolio")

    def get_total_value(self) -> float:
        """
        Calculate the total value of all assets in the portfolio
        
        Returns:
            float: Total portfolio value
        """
        total = sum(asset.get_value() for asset in self.assets.values())
        self._total_value = Decimal(str(total))
        return float(self._total_value)

    async def update_all_values(self) -> float:
        """
        Update values for all assets in the portfolio
        
        Returns:
            float: Updated total portfolio value
        """
        if self._is_syncing:
            logger.debug("Already syncing values, skipping duplicate call")
            return float(self._total_value)
            
        self._is_syncing = True
        
        try:
            # Use asyncio.gather to update all assets concurrently
            update_tasks = []
            for asset in self.assets.values():
                if hasattr(asset, 'update_value') and callable(asset.update_value):
                    update_tasks.append(asset.update_value())
                
            if update_tasks:
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Check for exceptions
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        asset_name = list(self.assets.keys())[i]
                        logger.error(f"Error updating {asset_name} value: {str(result)}")
                
            # Calculate total value
            total = Decimal('0')
            for asset in self.assets.values():
                total += Decimal(str(asset.get_value()))
                
            self._total_value = total
            self._last_update_time = time.time()
            
            return float(self._total_value)
        finally:
            self._is_syncing = False

    async def buy_asset(self, asset_name: str, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Buy an existing asset
        
        Args:
            asset_name: Name of the asset to buy
            amount: Amount to buy (currency amount or quantity depending on asset type)
            **kwargs: Additional parameters for the buy operation
            
        Returns:
            Dict[str, Any]: Buy operation result
            
        Raises:
            ValueError: If asset doesn't exist
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        
        asset = self.assets[asset_name]
        
        # Check if the asset is a TradableAsset
        if isinstance(asset, TradableAsset):
            result = await asset.buy(amount, **kwargs)
            
            # Track order in portfolio manager
            if result.get('success', False) and 'order_id' in result:
                self._all_orders[result['order_id']] = {
                    'asset_name': asset_name,
                    'action': 'buy',
                    'amount': amount,
                    'result': result,
                    'timestamp': time.time(),
                    'params': kwargs
                }
                
            return result
        elif hasattr(asset, 'buy') and callable(asset.buy):
            # Support for non-TradableAsset buy methods
            if asyncio.iscoroutinefunction(asset.buy):
                return await asset.buy(amount, **kwargs)
            else:
                return asset.buy(amount, **kwargs)
        else:
            raise ValueError(f"Asset {asset_name} does not support buy operations")

    async def sell_asset(self, asset_name: str, amount: float, **kwargs) -> Dict[str, Any]:
        """
        Sell an existing asset
        
        Args:
            asset_name: Name of the asset to sell
            amount: Amount to sell (currency amount or quantity depending on asset type)
            **kwargs: Additional parameters for the sell operation
            
        Returns:
            Dict[str, Any]: Sell operation result
            
        Raises:
            ValueError: If asset doesn't exist
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        
        asset = self.assets[asset_name]
        
        # Check if the asset is a TradableAsset
        if isinstance(asset, TradableAsset):
            result = await asset.sell(amount, **kwargs)
            
            # Track order in portfolio manager
            if result.get('success', False) and 'order_id' in result:
                self._all_orders[result['order_id']] = {
                    'asset_name': asset_name,
                    'action': 'sell',
                    'amount': amount,
                    'result': result,
                    'timestamp': time.time(),
                    'params': kwargs
                }
                
            return result
        elif hasattr(asset, 'sell') and callable(asset.sell):
            # Support for non-TradableAsset sell methods
            if asyncio.iscoroutinefunction(asset.sell):
                return await asset.sell(amount, **kwargs)
            else:
                return asset.sell(amount, **kwargs)
        else:
            raise ValueError(f"Asset {asset_name} does not support sell operations")

    async def place_order(self, asset_name: str, order_type: str, direction: str, 
                      amount: float, **kwargs) -> Dict[str, Any]:
        """
        Place an order for an asset
        
        Args:
            asset_name: Name of the asset
            order_type: Type of order ('market', 'limit', 'stop', etc.)
            direction: Order direction ('buy', 'sell')
            amount: Order amount
            **kwargs: Additional order parameters
            
        Returns:
            Dict[str, Any]: Order result
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        
        asset = self.assets[asset_name]
        
        # Check if the asset is a TradableAsset
        if not isinstance(asset, TradableAsset):
            raise ValueError(f"Asset {asset_name} does not support advanced order operations")
        
        # Normalize order type and direction
        order_type = order_type.lower()
        direction = direction.lower()
        
        # Place different order types
        try:
            if direction == 'buy':
                # Add order type to kwargs
                kwargs['order_type'] = order_type
                return await self.buy_asset(asset_name, amount, **kwargs)
            elif direction == 'sell':
                # Add order type to kwargs
                kwargs['order_type'] = order_type
                return await self.sell_asset(asset_name, amount, **kwargs)
            else:
                raise ValueError(f"Invalid direction: {direction}. Must be 'buy' or 'sell'")
        except Exception as e:
            logger.error(f"Error placing {order_type} {direction} order for {asset_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'asset_name': asset_name,
                'order_type': order_type,
                'direction': direction,
                'amount': amount
            }

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Cancel result
        """
        # Find the asset that owns this order
        if order_id in self._all_orders:
            asset_name = self._all_orders[order_id]['asset_name']
            asset = self.assets.get(asset_name)
            
            if asset and isinstance(asset, TradableAsset):
                try:
                    success = await asset.cancel_order(order_id)
                    
                    if success:
                        return {
                            'success': True,
                            'order_id': order_id,
                            'asset_name': asset_name,
                            'status': 'canceled'
                        }
                    else:
                        return {
                            'success': False,
                            'order_id': order_id,
                            'asset_name': asset_name,
                            'error': 'Cancel failed'
                        }
                except Exception as e:
                    logger.error(f"Error canceling order {order_id}: {str(e)}")
                    return {
                        'success': False,
                        'order_id': order_id,
                        'asset_name': asset_name,
                        'error': str(e)
                    }
            else:
                return {
                    'success': False,
                    'order_id': order_id,
                    'error': f"Asset {asset_name} not found or does not support order operations"
                }
        
        # If order not tracked, try canceling on all tradable assets
        for asset_name, asset in self.assets.items():
            if isinstance(asset, TradableAsset):
                try:
                    success = await asset.cancel_order(order_id)
                    if success:
                        return {
                            'success': True,
                            'order_id': order_id,
                            'asset_name': asset_name,
                            'status': 'canceled'
                        }
                except Exception:
                    continue
        
        return {
            'success': False,
            'order_id': order_id,
            'error': 'Order not found'
        }

    async def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order details by ID
        
        Args:
            order_id: Order ID
            
        Returns:
            Dict[str, Any]: Order details or empty dict if not found
        """
        # Check if we're tracking this order in portfolio manager
        if order_id in self._all_orders:
            asset_name = self._all_orders[order_id]['asset_name']
            asset = self.assets.get(asset_name)
            
            if asset and isinstance(asset, TradableAsset):
                order = asset.get_order(order_id)
                if order:
                    return {
                        'order_id': order_id,
                        'asset_name': asset_name,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'direction': order.direction.value,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'avg_filled_price': order.avg_filled_price,
                        'order_type': order.order_type.value,
                        'timestamp': order.timestamp
                    }
        
        # If order not tracked, search in all tradable assets
        for asset_name, asset in self.assets.items():
            if isinstance(asset, TradableAsset):
                order = asset.get_order(order_id)
                if order:
                    return {
                        'order_id': order_id,
                        'asset_name': asset_name,
                        'status': order.status.value,
                        'symbol': order.symbol,
                        'direction': order.direction.value,
                        'quantity': order.quantity,
                        'filled_quantity': order.filled_quantity,
                        'avg_filled_price': order.avg_filled_price,
                        'order_type': order.order_type.value,
                        'timestamp': order.timestamp
                    }
        
        return {}

    def list_assets(self) -> List[str]:
        """
        Get a list of all asset names
        
        Returns:
            List[str]: List of asset names
        """
        return list(self.assets.keys())

    def get_asset_weights(self) -> Dict[str, float]:
        """
        Calculate the weight of each asset in the portfolio
        
        Returns:
            Dict[str, float]: Dictionary mapping asset names to their weight percentages
        """
        total_value = Decimal(str(self.get_total_value()))
        
        if total_value == 0:
            return {}
            
        return {name: float(Decimal(str(asset.get_value())) / total_value) 
                for name, asset in self.assets.items()}

    def set_exchange(self, exchange) -> None:
        """
        Set exchange for the portfolio and all assets
        
        Args:
            exchange: Exchange interface
        """
        self.exchange = exchange
        
        # Update exchange for all assets
        for asset in self.assets.values():
            asset.set_exchange(exchange)
            
        logger.info("Updated exchange for portfolio and all assets")

    async def sync_with_exchange(self) -> Dict[str, Any]:
        """
        Synchronize all assets with exchange data
        
        Returns:
            Dict[str, Any]: Sync results by asset
        """
        if not self.exchange:
            logger.warning("No exchange available, cannot sync portfolio")
            return {'success': False, 'error': 'No exchange available'}
        
        if self._is_syncing:
            logger.debug("Already syncing, skipping duplicate call")
            return {'success': False, 'error': 'Sync already in progress'}
            
        self._is_syncing = True
        
        try:
            results = {}
            
            # Sync each asset type accordingly
            for name, asset in self.assets.items():
                try:
                    # Different sync methods based on asset type
                    if hasattr(asset, 'sync_balance'):
                        result = await asset.sync_balance()
                    elif hasattr(asset, 'sync_position'):
                        result = await asset.sync_position()
                    else:
                        # Basic value update
                        value = await asset.update_value()
                        result = {'value': value}
                        
                    results[name] = result
                except Exception as e:
                    logger.error(f"Error syncing {name}: {str(e)}")
                    results[name] = {'error': str(e)}
            
            # Update total value
            total = sum(asset.get_value() for asset in self.assets.values())
            self._total_value = Decimal(str(total))
            
            return {
                'success': True,
                'total_value': float(self._total_value),
                'asset_results': results
            }
        finally:
            self._is_syncing = False
    
    async def execute_signal_batch(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute a batch of trading signals
        
        Args:
            signals: DataFrame with columns: 'timestamp', 'symbol', 'action', 'quantity', 'price'
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if signals.empty:
            return {'success': True, 'executed': 0, 'message': 'No signals to execute'}
        
        required_cols = ['timestamp', 'symbol', 'action', 'quantity']
        missing = [col for col in required_cols if col not in signals.columns]
        if missing:
            return {'success': False, 'error': f"Missing required columns: {missing}"}
        
        results = []
        
        # Process each signal
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            action = signal['action'].lower()
            amount = float(signal['quantity'])
            price = signal.get('price', None)
            
            # Find the corresponding asset
            asset = self.assets.get(symbol)
            if not asset:
                results.append({
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'error': f"Asset {symbol} not found in portfolio"
                })
                continue
            
            try:
                # Execute buy or sell
                if action in ('buy', 'long'):
                    kwargs = {}
                    if price:
                        kwargs['price'] = price
                        kwargs['order_type'] = 'limit'
                    
                    result = await self.buy_asset(symbol, amount, **kwargs)
                elif action in ('sell', 'short'):
                    kwargs = {}
                    if price:
                        kwargs['price'] = price
                        kwargs['order_type'] = 'limit'
                    
                    result = await self.sell_asset(symbol, amount, **kwargs)
                else:
                    result = {
                        'success': False,
                        'symbol': symbol,
                        'action': action,
                        'error': f"Unsupported action: {action}"
                    }
                
                # Add signal info to result
                result.update({
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'timestamp': signal['timestamp']
                })
                
                results.append(result)
            except Exception as e:
                results.append({
                    'success': False,
                    'symbol': symbol,
                    'action': action,
                    'amount': amount,
                    'error': str(e)
                })
        
        # Summarize results
        success_count = sum(1 for r in results if r.get('success', False))
        
        return {
            'success': True,
            'executed': len(results),
            'successful': success_count,
            'failed': len(results) - success_count,
            'results': results
        }
        
    def calculate_portfolio_risk(self, method: str = 'var') -> float:
        """
        Calculate portfolio risk using specified method
        
        Args:
            method: Risk calculation method ('var', 'std', etc.)
            
        Returns:
            float: Risk measure
            
        Note:
            This implementation can be extended with actual risk calculations
        """
        # Simple risk calculation based on asset allocation and volatility
        # This is a placeholder - actual implementation would be more complex
        
        if method == 'var':
            # Value at Risk calculation would go here
            # Placeholder: sum of weighted asset volatilities
            return 0.05  # Example 5% VaR
        elif method == 'std':
            # Standard deviation calculation
            return 0.15  # Example 15% portfolio volatility
        else:
            logger.warning(f"Unsupported risk calculation method: {method}")
            return 0.0

    def get_portfolio_return(self, start_date: str, end_date: str) -> float:
        """
        Calculate portfolio return over a specified period
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            float: Portfolio return as a decimal
            
        Note:
            This implementation can be extended with actual return calculations
        """
        # In a real implementation, this would calculate returns from historical data
        # Placeholder for now
        return 0.0
        
    async def close(self):
        """
        Close the portfolio manager and clean up resources
        """
        # Cancel all orders
        for asset_name, asset in self.assets.items():
            if isinstance(asset, TradableAsset):
                try:
                    await asset.cancel_all_orders()
                    if hasattr(asset, 'close') and callable(asset.close):
                        asset.close()
                except Exception as e:
                    logger.error(f"Error closing asset {asset_name}: {str(e)}")
        
        # Close exchange connection if we own it
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close()
                logger.info("Closed exchange connection")
            except Exception as e:
                logger.error(f"Error closing exchange connection: {str(e)}")
        
        logger.info("Portfolio manager closed")