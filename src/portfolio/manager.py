#!/usr/bin/env python3
# src/portfolio/manager.py

import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd

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
        
        # Initialize factories
        self.asset_factory = get_asset_factory(self.config)
        self.execution_factory = get_execution_factory(self.config)
        self.risk_management_factory = get_risk_factory(self.config)
        
        # Portfolio state tracking
        self._total_value = Decimal('0')
        self._last_update_time = 0
        self._is_syncing = False
        
        # Order tracking
        self._all_orders: Dict[str, Dict[str, Any]] = {}
        
        # Risk manager (will be initialized in _initialize)
        self.risk_manager = None
        
        # Risk event callbacks
        self._risk_callbacks = {}
        
        # Execution engine
        self.execution_engine = None
        
        # Initialize components
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
        
        # Initialize assets from configuration
        self.asset_factory.discover_assets()
        self.logger.info("Asset types discovered")
        await self._initialize_configured_assets()
        
        # Initialize risk manager
        await self._initialize_risk_manager()
        
        # Initialize execution engine
        self.execution_engine = await self.execution_factory.create()
        self.logger.info(f"Execution engine initialized: {self.execution_engine.__class__.__name__}")
        
        # Register for risk events
        self._register_risk_events()
        
        self.logger.info("Portfolio manager initialization complete")
        return self

    async def _initialize_configured_assets(self) -> None:
        """Initialize assets defined in configuration"""
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

    async def _initialize_risk_manager(self) -> None:
        """Initialize risk manager component"""
        try:
            # Create risk manager instance
            self.risk_manager = await self.risk_management_factory.create_with_config_params(self)
            
            # Perform initial risk assessment
            if self.risk_manager:
                await self.risk_manager.execute_risk_control()
                self.logger.info(f"Risk manager initialized: {self.risk_manager.__class__.__name__}")
            else:
                self.logger.warning("Failed to initialize risk manager")
        except Exception as e:
            self.logger.error(f"Error initializing risk manager: {str(e)}")
            raise

    def _register_risk_events(self) -> None:
        """Register for risk event notifications"""
        if not self.risk_manager:
            return
            
        # Register for risk breach events
        self._risk_callbacks['risk_breach'] = self._on_risk_breach
        self._risk_callbacks['emergency_action'] = self._on_emergency_action
        self._risk_callbacks['position_limit'] = self._on_position_limit
        
        # Set up event handlers in risk manager if it supports the interface
        if hasattr(self.risk_manager, 'register_event_handler'):
            for event, callback in self._risk_callbacks.items():
                self.risk_manager.register_event_handler(event, callback)

    async def _on_risk_breach(self, event_data: Dict[str, Any]) -> None:
        """
        Handle risk breach event
        
        Args:
            event_data: Risk breach event details
        """
        breach_type = event_data.get('type', 'unknown')
        severity = event_data.get('severity', 'high')
        
        self.logger.warning(f"Risk breach detected: {breach_type} (severity: {severity})")
        
        # Take appropriate action based on breach type and severity
        if severity == 'critical' and event_data.get('auto_action', False):
            # For critical breaches, we might want to reduce positions automatically
            if 'action' in event_data and event_data['action'] == 'reduce_positions':
                reduction_pct = event_data.get('reduction_pct', 0.5)
                await self._reduce_positions(reduction_pct)

    async def _on_emergency_action(self, event_data: Dict[str, Any]) -> None:
        """
        Handle emergency action event
        
        Args:
            event_data: Emergency action details
        """
        action = event_data.get('action', 'unknown')
        
        self.logger.warning(f"Emergency action required: {action}")
        
        if action == 'reduce_positions':
            reduction_pct = event_data.get('reduction_pct', 0.5)
            await self._reduce_positions(reduction_pct)
        elif action == 'close_all_positions':
            await self._close_all_positions()
        elif action == 'pause_trading':
            # Implement pause trading functionality
            pass

    async def _on_position_limit(self, event_data: Dict[str, Any]) -> None:
        """
        Handle position limit event
        
        Args:
            event_data: Position limit details
        """
        asset_name = event_data.get('asset_name', 'unknown')
        limit_type = event_data.get('limit_type', 'unknown')
        
        self.logger.warning(f"Position limit reached for {asset_name}: {limit_type}")
        
        # Take appropriate action based on limit type
        if limit_type == 'max_position' and event_data.get('auto_action', False):
            # Reduce position if auto action is enabled
            if asset_name in self.assets:
                reduction_pct = event_data.get('reduction_pct', 0.25)
                await self._reduce_asset_position(asset_name, reduction_pct)

    async def _reduce_positions(self, reduction_pct: float) -> None:
        """
        Reduce positions by a percentage
        
        Args:
            reduction_pct: Percentage to reduce (0.0-1.0)
        """
        self.logger.info(f"Reducing all positions by {reduction_pct * 100:.0f}%")
        
        for asset_name, asset in self.assets.items():
            await self._reduce_asset_position(asset_name, reduction_pct)

    async def _reduce_asset_position(self, asset_name: str, reduction_pct: float) -> None:
        """
        Reduce a specific asset position
        
        Args:
            asset_name: Asset name
            reduction_pct: Percentage to reduce (0.0-1.0)
        """
        if asset_name not in self.assets:
            return
            
        asset = self.assets[asset_name]
        
        # Skip if not a tradable asset
        if not hasattr(asset, 'get_position_size'):
            return
            
        # Get position size
        position_size = asset.get_position_size()
        
        if position_size > 0:
            # Calculate reduction amount
            reduction_amount = position_size * reduction_pct
            
            if reduction_amount > 0:
                self.logger.info(f"Reducing {asset_name} position by {reduction_amount} units")
                
                # Execute the sell
                try:
                    await self.sell_asset(asset_name, float(reduction_amount))
                except Exception as e:
                    self.logger.error(f"Error reducing position for {asset_name}: {e}")

    async def _close_all_positions(self) -> None:
        """Close all open positions"""
        self.logger.warning("Closing all positions")
        
        for asset_name, asset in self.assets.items():
            # Skip if not a tradable asset
            if not hasattr(asset, 'get_position_size'):
                continue
                
            # Get position size
            position_size = asset.get_position_size()
            
            if position_size > 0:
                self.logger.info(f"Closing position for {asset_name}: {position_size} units")
                
                # Execute the sell
                try:
                    await self.sell_asset(asset_name, float(position_size))
                except Exception as e:
                    self.logger.error(f"Error closing position for {asset_name}: {e}")

    async def notify_risk_manager(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Notify risk manager of an event
        
        Args:
            event_type: Event type
            data: Event data
        """
        if not self.risk_manager:
            return
            
        # Check if risk manager has the appropriate method
        if hasattr(self.risk_manager, 'handle_event'):
            try:
                await self.risk_manager.handle_event(event_type, data)
            except Exception as e:
                self.logger.error(f"Error notifying risk manager of {event_type}: {e}")

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
        
        # Notify risk manager of new asset
        await self.notify_risk_manager('asset_added', {
            'asset_name': asset.name,
            'asset_type': asset.__class__.__name__
        })

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
            
        # Remove from portfolio
        del self.assets[asset_name]
        self.logger.info(f"Removed asset {asset_name} from portfolio")
        
        # Notify risk manager of asset removal
        await self.notify_risk_manager('asset_removed', {
            'asset_name': asset_name
        })

    def get_total_value(self) -> float:
        """
        Calculate the total value of all assets in the portfolio
        
        Returns:
            float: Total portfolio value
        """
        total = sum(asset.get_value() for asset in self.assets.values())
        self._total_value = Decimal(str(total))
        
        # Notify risk manager of value update
        if self.risk_manager:
            # Use appropriate method if available
            if hasattr(self.risk_manager, 'update_portfolio_value'):
                self.risk_manager.update_portfolio_value(float(self._total_value))
        
        return float(self._total_value)

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
                
            old_value = self._total_value
            self._total_value = total
            self._last_update_time = time.time()
            
            # Notify risk manager of value update
            if self.risk_manager:
                # Use appropriate method if available
                if hasattr(self.risk_manager, 'update_portfolio_value'):
                    self.risk_manager.update_portfolio_value(float(self._total_value))
                
                # Notify of significant value changes
                if old_value > 0 and abs((self._total_value - old_value) / old_value) > 0.05:
                    await self.notify_risk_manager('significant_value_change', {
                        'old_value': float(old_value),
                        'new_value': float(self._total_value),
                        'change_pct': float((self._total_value - old_value) / old_value)
                    })
            
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
        
        # Apply risk checks via risk manager
        if self.risk_manager:
            validation = await self.risk_manager.validate_order(asset_name, 'buy', amount, **kwargs)
            if not validation.get('allowed', False):
                return {
                    'success': False,
                    'error': 'Trade rejected due to risk limits',
                    'asset_name': asset_name,
                    'action': 'buy',
                    'amount': amount,
                    'reasons': validation.get('reasons', ['Risk check failed'])
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
            
            # Notify risk manager of new order
            await self.notify_risk_manager('order_executed', {
                'asset_name': asset_name,
                'direction': 'buy',
                'amount': amount,
                'order_id': result['order_id'],
                'result': result
            })
                
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
        
        # Apply risk checks via risk manager (simplified for sells)
        if self.risk_manager:
            validation = await self.risk_manager.validate_order(asset_name, 'sell', amount, **kwargs)
            if not validation.get('allowed', False):
                return {
                    'success': False,
                    'error': 'Trade rejected due to risk limits',
                    'asset_name': asset_name,
                    'action': 'sell',
                    'amount': amount,
                    'reasons': validation.get('reasons', ['Risk check failed'])
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
            
            # Notify risk manager of new order
            await self.notify_risk_manager('order_executed', {
                'asset_name': asset_name,
                'direction': 'sell',
                'amount': amount,
                'order_id': result['order_id'],
                'result': result
            })
                
        return result

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
                        result = {
                            'success': True,
                            'order_id': order_id,
                            'asset_name': asset_name,
                            'status': 'canceled'
                        }
                        
                        # Notify risk manager of order cancellation
                        await self.notify_risk_manager('order_canceled', {
                            'asset_name': asset_name,
                            'order_id': order_id
                        })
                        
                        return result
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
                        # Notify risk manager of order cancellation
                        await self.notify_risk_manager('order_canceled', {
                            'asset_name': asset_name,
                            'order_id': order_id
                        })
                        
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
            
            # Update risk manager
            if self.risk_manager:
                # Update portfolio value
                if hasattr(self.risk_manager, 'update_portfolio_value'):
                    self.risk_manager.update_portfolio_value(float(self._total_value))
                    
                # Execute risk control
                await self.risk_manager.execute_risk_control()
                
                # Notify of sync completion
                await self.notify_risk_manager('sync_completed', {
                    'total_value': float(self._total_value),
                    'asset_count': len(self.assets)
                })
            
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
        
        # Validate signals with risk manager if available
        valid_signals = signals
        if self.risk_manager:
            try:
                valid_signals = await self.risk_manager.validate_signals(signals)
            except Exception as e:
                self.logger.error(f"Error validating signals: {e}")
                # Continue with original signals if validation fails
        
        # Notify risk manager of batch execution start
        if self.risk_manager:
            await self.notify_risk_manager('batch_execution_start', {
                'signal_count': len(valid_signals),
                'symbols': valid_signals['symbol'].unique().tolist() if not valid_signals.empty else []
            })
        
        results = []
        
        # Process each signal
        for _, signal in valid_signals.iterrows():
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
        failed_count = len(results) - success_count
        
        # Notify risk manager of batch execution completion
        if self.risk_manager:
            await self.notify_risk_manager('batch_execution_complete', {
                'total': len(results),
                'successful': success_count,
                'failed': failed_count
            })
        
        return {
            'success': True,
            'executed': len(results),
            'successful': success_count,
            'failed': failed_count,
            'results': results
        }
        
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the portfolio state
        
        Returns:
            Dict with portfolio summary
        """
        # Basic portfolio information
        summary = {
            'total_value': float(self._total_value),
            'assets': len(self.assets),
            'asset_values': {name: asset.get_value() for name, asset in self.assets.items()},
            'weights': self.get_asset_weights(),
            'last_update': self._last_update_time
        }
        
        # Add risk information if available
        if self.risk_manager:
            risk_report = self.risk_manager.get_risk_report()
            # Only add risk metrics, not the full position details which are already in the summary
            risk_metrics = {k: v for k, v in risk_report.items() 
                           if k not in ['positions', 'position_count']}
            summary['risk'] = risk_metrics
        
        return summary
        
    async def close(self):
        """
        Close the portfolio manager and clean up resources
        """
        # Notify risk manager of shutdown
        if self.risk_manager:
            await self.notify_risk_manager('portfolio_shutdown', {
                'total_value': float(self._total_value),
                'asset_count': len(self.assets)
            })
        
        # Cancel all orders
        for asset_name, asset in self.assets.items():
            try:
                if hasattr(asset, 'cancel_all_orders') and callable(asset.cancel_all_orders):
                    await asset.cancel_all_orders()
                
                if hasattr(asset, 'close') and callable(asset.close):
                    await asset.close()
            except Exception as e:
                self.logger.error(f"Error closing asset {asset_name}: {str(e)}")
        
        # Close risk manager
        if self.risk_manager:
            try:
                await self.risk_manager.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down risk manager: {str(e)}")
        
        # Close execution engine
        if self.execution_engine and hasattr(self.execution_engine, 'close'):
            try:
                await self.execution_engine.close()
            except Exception as e:
                self.logger.error(f"Error closing execution engine: {str(e)}")
        
        # Close exchange connection if we own it
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close()
                self.logger.info("Closed exchange connection")
            except Exception as e:
                self.logger.error(f"Error closing exchange connection: {str(e)}")
        
        self.logger.info("Portfolio manager closed")