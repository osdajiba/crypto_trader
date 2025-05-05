#!/usr/bin/env python3
# src/portfolio/manager.py

import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from common.abstract_factory import AbstractFactory

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.exchange.factory import get_exchange_factory
from src.portfolio.assets.base import Asset
from src.portfolio.assets.factory import get_asset_factory
from src.portfolio.risk.factory import get_risk_factory
from src.portfolio.execution.factory import get_execution_factory


class PortfolioManager:
    """
    Manager for handling a collection of assets in a portfolio
    with exchange integration and order execution capabilities.
    
    This class provides a centralized way to manage multiple assets,
    execute trades, track positions, and handle risk management.
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
        self.logger = LogManager.get_logger("portfolio.manager")
        
        # Initialize exchange
        self.exchange = exchange
        self._exchange_factory = None
        
        # Initialize asset factory
        self.asset_factory = get_asset_factory(self.config)
        self.execution_factory = get_execution_factory(self.config)
        self.risk_management_factory = get_risk_factory(self.config)
        
        # Portfolio state tracking
        self._total_value = Decimal('0')
        self._last_update_time = 0
        self._is_syncing = False
        
        # Order tracking
        self._all_orders: Dict[str, Dict[str, Any]] = {}
        
        self._initialize()
                
        self.logger.info("Portfolio manager initialized")

    async def _initialize(self) -> None:
        """
        Initialize the portfolio manager and its components.
        
        Should be called after creation to properly set up async resources.
        """
        # Initialize exchange
        if not self.exchange:
            self.exchange_factory = get_exchange_factory(self.config)
            self.exchange = await self.exchange_factory.create()
            self.logger.info(f"Exchange initialized: {self.exchange.__class__.__name__}")
        
        # Initialize all asset in asset_list
        self.asset_factory.discover_assets()
        self.logger.info("Asset types discovered")
        assets_config = self.config.get("portfolio", "assets", default={})
        
        for asset_name, asset_config in assets_config.items():
            try:
                asset_type = asset_config.get("type", "spot")
                
                # Add exchange to params
                params = asset_config.get("params", {})
                if self.exchange and 'exchange' not in params:
                    params['exchange'] = self.exchange
                
                # Create and add the asset
                asset = await self.create_asset(asset_type, {
                    "name": asset_name,
                    **params
                })
                self.logger.info(f"Initialized pre-configured asset: {asset_name} ({self.exchange.__class__.__name__} : {asset_type})")
            except Exception as e:
                self.logger.error(f"Failed to initialize asset {asset_name}: {str(e)}")
        
        # Initialize risk manager
        self.risk_manager = await self.risk_management_factory.create_with_config_params(self)
        self.logger.info(f"Exchange initialized: {self.exchange.__class__.__name__}")
        
        # Initialize execution
        self.execution_engine = await self.execution_factory.create()
        self.logger.info(f"Exchange initialized: {self.exchange.__class__.__name__}")        
        
        self.logger.info("Portfolio manager initialization complete")
        return self
    
    def _init_risk_management_params(self) -> None:
        """Initialize risk management parameters"""
        risk_config = self.config.get("portfolio", "risk", default={})
        
        # Maximum allowed drawdown before stopping trading
        self.max_drawdown = Decimal(str(risk_config.get("max_drawdown", 0.25)))
        
        # Maximum percentage of portfolio in a single position
        self.max_position_pct = Decimal(str(risk_config.get("max_position_pct", 0.25)))
        
        # Maximum leverage allowed (if applicable)
        self.max_leverage = Decimal(str(risk_config.get("max_leverage", 1.0)))
        
        # Position sizing method
        self.position_sizing_method = risk_config.get("position_sizing", "fixed")
        
        # Risk per trade (percentage of portfolio)
        self.risk_per_trade = Decimal(str(risk_config.get("risk_per_trade", 0.01)))
        
        # Historical drawdown values
        self.peak_value = Decimal('0')
        self.current_drawdown = Decimal('0')
        
        self.logger.info(f"Risk management initialized: max drawdown {self.max_drawdown}, max position {self.max_position_pct}")

    async def add_asset(self, asset: Asset) -> None:
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
            
        # Initialize the asset if needed
        if hasattr(asset, 'initialize') and callable(asset.initialize):
            await asset.initialize()
            
        self.assets[asset.name] = asset
        self.logger.info(f"Added asset {asset.name} to portfolio")

    async def create_asset(self, asset_type: str, params: dict) -> Asset:
        """
        Create and add a new asset to the portfolio
        
        Args:
            asset_type: Type of asset to create
            params: Parameters for asset creation
            
        Returns:
            Asset: The created asset
            
        Raises:
            ValueError: If asset creation fails
        """
        # Add exchange to parameters if available
        if self.exchange and 'exchange' not in params:
            params['exchange'] = self.exchange
            
        # Add configuration
        if 'config' not in params:
            params['config'] = self.config
            
        # Create the asset
        asset = await self.asset_factory.create(asset_type, params)
        
        # Add to portfolio
        await self.add_asset(asset)
        return asset

    async def remove_asset(self, asset_name: str) -> None:
        """
        Remove an asset from the portfolio
        
        Args:
            asset_name: Name of the asset to remove
            
        Raises:
            ValueError: If asset doesn't exist
        """
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
            
        # Close the asset if it has a close method
        asset = self.assets[asset_name]
        if hasattr(asset, 'close') and callable(asset.close):
            await asset.close()
            
        del self.assets[asset_name]
        self.logger.info(f"Removed asset {asset_name} from portfolio")

    def get_total_value(self) -> float:
        """
        Calculate the total value of all assets in the portfolio
        
        Returns:
            float: Total portfolio value
        """
        total = sum(asset.get_value() for asset in self.assets.values())
        self._total_value = Decimal(str(total))
        
        # Update peak value and drawdown
        self._update_drawdown()
        
        return float(self._total_value)

    def _update_drawdown(self) -> None:
        """Update peak value and current drawdown"""
        if self._total_value > self.peak_value:
            self.peak_value = self._total_value
            self.current_drawdown = Decimal('0')
        elif self.peak_value > 0:
            self.current_drawdown = (self.peak_value - self._total_value) / self.peak_value
            
        # Check if drawdown exceeds limit
        if self.current_drawdown > self.max_drawdown:
            self.logger.warning(f"DRAWDOWN ALERT: Current drawdown {self.current_drawdown:.2%} exceeds maximum {self.max_drawdown:.2%}")

    async def update_all_values(self) -> float:
        """
        Update values for all assets in the portfolio
        
        Returns:
            float: Updated total portfolio value
        """
        if self._is_syncing:
            self.logger.debug("Already syncing values, skipping duplicate call")
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
                        self.logger.error(f"Error updating {asset_name} value: {str(result)}")
                
            # Calculate total value
            total = Decimal('0')
            for asset in self.assets.values():
                total += Decimal(str(asset.get_value()))
                
            self._total_value = total
            self._last_update_time = time.time()
            
            # Update drawdown metrics
            self._update_drawdown()
            
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
        
        # Check if the asset is tradable
        if not hasattr(asset, 'buy') or not callable(asset.buy):
            raise ValueError(f"Asset {asset_name} does not support buy operations")
        
        # Apply risk checks
        if not await self._check_position_risk(asset_name, amount, 'buy'):
            return {
                'success': False,
                'error': 'Trade rejected due to risk limits',
                'asset_name': asset_name,
                'action': 'buy',
                'amount': amount
            }
        
        # Execute buy
        result = await asset.buy(amount, **kwargs)
        
        # Track order if successful
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
        
        # Check if the asset is tradable
        if not hasattr(asset, 'sell') or not callable(asset.sell):
            raise ValueError(f"Asset {asset_name} does not support sell operations")
        
        # Apply risk checks (simplified for sells)
        if not await self._check_position_risk(asset_name, amount, 'sell'):
            return {
                'success': False,
                'error': 'Trade rejected due to risk limits',
                'asset_name': asset_name,
                'action': 'sell',
                'amount': amount
            }
        
        # Execute sell
        result = await asset.sell(amount, **kwargs)
        
        # Track order if successful
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

    async def _check_position_risk(self, asset_name: str, amount: float, action: str) -> bool:
        """
        Check if a trade would violate risk management rules
        
        Args:
            asset_name: Asset name
            amount: Trade amount
            action: Trade action ('buy' or 'sell')
            
        Returns:
            True if trade is acceptable, False if it should be rejected
        """
        # Skip checks for sales (risk reduction)
        if action.lower() == 'sell':
            return True
            
        # Check overall portfolio drawdown
        if self.current_drawdown > self.max_drawdown:
            self.logger.warning(f"Trade rejected: current drawdown ({self.current_drawdown:.2%}) exceeds maximum ({self.max_drawdown:.2%})")
            return False
            
        # Get asset's current value and portfolio total
        asset = self.assets[asset_name]
        current_value = Decimal(str(asset.get_value()))
        total_value = self._total_value if self._total_value > 0 else self.get_total_value()
        
        # Estimate the new position size based on the trade
        # Note: This is a simplification and should be refined based on asset type
        if hasattr(asset, '_last_price') and asset._last_price > 0:
            # Use last price for estimate
            trade_value = Decimal(str(amount)) * asset._last_price
        else:
            # Rough approximation
            trade_value = Decimal(str(amount))
        
        new_position_value = current_value + trade_value
        
        # Check position size as percentage of portfolio
        if total_value > 0:
            position_pct = new_position_value / total_value
            if position_pct > self.max_position_pct:
                self.logger.warning(f"Trade rejected: position size ({position_pct:.2%}) would exceed maximum ({self.max_position_pct:.2%})")
                return False
        
        return True

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
            self.logger.error(f"Error placing {order_type} {direction} order for {asset_name}: {str(e)}")
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
            
            if asset and hasattr(asset, 'cancel_order') and callable(asset.cancel_order):
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
                    self.logger.error(f"Error canceling order {order_id}: {str(e)}")
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
        
        # If order not tracked, try canceling on all assets
        for asset_name, asset in self.assets.items():
            if hasattr(asset, 'cancel_order') and callable(asset.cancel_order):
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
            
            if asset and hasattr(asset, 'get_order_status') and callable(asset.get_order_status):
                return await asset.get_order_status(order_id)
                
            elif asset and hasattr(asset, 'get_order') and callable(asset.get_order):
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
        
        # If order not tracked, search in all assets
        for asset_name, asset in self.assets.items():
            # Try get_order_status first (preferred)
            if hasattr(asset, 'get_order_status') and callable(asset.get_order_status):
                result = await asset.get_order_status(order_id)
                if result.get('success', False):
                    return result
                    
            # Fall back to get_order
            elif hasattr(asset, 'get_order') and callable(asset.get_order):
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
        
        return {
            'success': False,
            'error': 'Order not found'
        }

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
            
        self.logger.info("Updated exchange for portfolio and all assets")

    async def sync_with_exchange(self) -> Dict[str, Any]:
        """
        Synchronize all assets with exchange data
        
        Returns:
            Dict[str, Any]: Sync results by asset
        """
        if not self.exchange:
            self.logger.warning("No exchange available, cannot sync portfolio")
            return {'success': False, 'error': 'No exchange available'}
        
        if self._is_syncing:
            self.logger.debug("Already syncing, skipping duplicate call")
            return {'success': False, 'error': 'Sync already in progress'}
            
        self._is_syncing = True
        
        try:
            results = {}
            
            # Sync each asset accordingly
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
                    self.logger.error(f"Error syncing {name}: {str(e)}")
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
        
    def calculate_portfolio_risk(self, method: str = 'var') -> Dict[str, float]:
        """
        Calculate portfolio risk using specified method
        
        Args:
            method: Risk calculation method ('var', 'std', 'drawdown', etc.)
            
        Returns:
            Dict with risk metrics
        """
        # Get current drawdown
        current_drawdown = float(self.current_drawdown)
        
        # Get portfolio weights
        weights = self.get_asset_weights()
        
        # Basic portfolio risk metrics
        if method == 'var':
            # Simple Value at Risk estimate
            # In a real implementation, would use historical returns and proper VaR calculation
            var_estimate = 0.05  # 5% VaR (placeholder)
            return {
                'var': var_estimate,
                'drawdown': current_drawdown,
                'max_position': max(weights.values()) if weights else 0.0
            }
        elif method == 'std':
            # Standard deviation (volatility) estimate
            # Placeholder - real implementation would calculate from return series
            volatility = 0.15  # 15% volatility
            return {
                'volatility': volatility,
                'drawdown': current_drawdown,
                'max_position': max(weights.values()) if weights else 0.0
            }
        elif method == 'drawdown':
            # Just return current and maximum drawdown
            return {
                'current_drawdown': current_drawdown,
                'max_drawdown': float(self.max_drawdown),
                'max_position': max(weights.values()) if weights else 0.0
            }
        else:
            self.logger.warning(f"Unsupported risk calculation method: {method}")
            return {
                'drawdown': current_drawdown,
                'error': f"Unsupported method: {method}"
            }

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the portfolio state
        
        Returns:
            Dict with portfolio summary
        """
        return {
            'total_value': float(self._total_value),
            'assets': len(self.assets),
            'asset_values': {name: asset.get_value() for name, asset in self.assets.items()},
            'weights': self.get_asset_weights(),
            'drawdown': float(self.current_drawdown),
            'last_update': self._last_update_time
        }
        
    async def close(self):
        """
        Close the portfolio manager and clean up resources
        """
        # Cancel all orders
        for asset_name, asset in self.assets.items():
            try:
                if hasattr(asset, 'cancel_all_orders') and callable(asset.cancel_all_orders):
                    await asset.cancel_all_orders()
                
                if hasattr(asset, 'close') and callable(asset.close):
                    await asset.close()
            except Exception as e:
                self.logger.error(f"Error closing asset {asset_name}: {str(e)}")
        
        # Close exchange connection if we own it
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close()
                self.logger.info("Closed exchange connection")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {str(e)}")
        
        self.logger.info("Portfolio manager closed")