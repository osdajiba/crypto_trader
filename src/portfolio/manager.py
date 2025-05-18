#!/usr/bin/env python3
# src/portfolio/manager.py

import asyncio
import time
from decimal import Decimal
from typing import Dict, List, Any, Optional, Union, Callable
import pandas as pd
from portfolio.execution.order import Direction

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.portfolio.assets.base import Asset
from src.exchange.base import Exchange
from src.exchange.factory import get_exchange_factory
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
    
    def __init__(self, config: Optional[ConfigManager] = None, exchange: Exchange = None):
        """
        Initialize portfolio manager
        
        Args:
            config: Configuration manager (optional)
            exchange: Exchange interface (optional)
        """
        self.assets: Dict[str, Asset] = {}
        self.config = config if config else ConfigManager()
        self.logger = LogManager.get_logger("portfolio.manager")
        
        # Initialize plugins
        self.exchange = exchange
        self.exchange_factory = None
        self._all_orders: Dict[str, Dict[str, Any]] = {}    # Order tracking
        self.risk_manager = None
        self._risk_callbacks = {}
        self.execution_engine = None        
        
        # Initialize factories
        self.asset_factory = get_asset_factory(self.config)
        self.execution_factory = get_execution_factory(self.config)
        self.risk_management_factory = get_risk_factory(self.config)
        
        # Portfolio state tracking
        self._last_update_time = 0
        self._is_syncing = False
        
        self.logger.info("Portfolio manager initialized")

    async def initialize(self) -> None:
        """
        Initialize the portfolio manager and its components.
        
        Should be called after creation to properly set up async resources.
        """
        try:
            # Initialize exchange
            if not self.exchange:
                self.exchange_factory = get_exchange_factory(self.config)
                self.exchange = await self.exchange_factory.create_exchange('binance')
                self.logger.info(f"Exchange initialized: {self.exchange.__class__.__name__}")
            
            # Initialize execution engine
            self.execution_engine = await self.execution_factory.create_execution_engine()
            self.logger.info(f"Execution engine initialized: {self.execution_engine.__class__.__name__}")
        
            # Initialize risk manager and register for risk events
            await self._initialize_risk_manager()            
            self._register_risk_events()
            
            # Initialize assets from configuration
            self.asset_factory.discover_assets()
            self.logger.info("Asset types discovered")
            await self._initialize_configured_assets()      
            
            # Initialize portfolio value 
            initial_capital = float(self.config.get("trading", "initial_capital", default=100000.0))
            self._total_value = {'cash': initial_capital}
            
            for asset_name, _ in self.assets.items():
                self._total_value[asset_name] = 0.0  # This should remain float
                  
            self.logger.info("Portfolio manager initialization complete")

        except Exception as e:
            self.logger.error(f"Failed to initialize portfolio manager: {str(e)}")
            raise

    async def _initialize_configured_assets(self) -> None:
        """Initialize assets defined in configuration"""
        
        assets_config = self.config.get("trading", "assets", default={})
        if not assets_config:
            self.logger.info("No assets defined in configuration")
            return
            
        self.logger.info(f"Initializing {len(assets_config)} assets from configuration...")
        
        try:
            assets = await self.asset_factory.create_multi_assets(assets_config, self.exchange, self.execution_engine)
            self.logger.info(f"Successfully initialized {len(assets)} assets")
            
            # Add assets to portfolio manager and ensure execution engine is set
            for asset_name, asset in assets.items():
                try:
                    # Explicitly set execution engine if it's missing
                    if hasattr(asset, 'execution_engine') and asset.execution_engine is None:
                        asset.execution_engine = self.execution_engine
                        self.logger.info(f"Set missing execution engine for {asset_name}")
                    
                    self.assets[asset_name] = asset
                    self.logger.info(f"Added {asset_name} to portfolio")
                except Exception as e:
                    self.logger.error(f"Failed to add asset {asset_name} to portfolio: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize assets from configuration: {str(e)}")
            
    async def _initialize_risk_manager(self) -> None:
        """Initialize risk manager component"""
        try:
            # Create risk manager instance
            self.mode_name = self.config.get("system", "operational_mode", default=None)
            risk_management_type = {"backtest": "backtest", "paper": "standard", "live": "conservative"}.get(self.mode_name)
            self.risk_manager = await self.risk_management_factory.create_risk_mamager(self, risk_management_type)
            
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
            # TODO: Implement pause trading functionality
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
                
                # Close all positions the sell with market order
                try:
                    kwargs = {
                        "symbol": asset_name,
                        "price": None,
                        "quantity": position_size,
                        "order_type": "market",
                        "direction": "sell"
                    }
                    await self.sell_asset(kwargs)
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
                
                # Close all positions the sell with market order
                try:
                    kwargs = {
                        "symbol": asset_name,
                        "price": None,
                        "quantity": position_size,
                        "order_type": "market",
                        "direction": "sell"
                    }
 
                    await self.sell_asset(kwargs)
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
        # Check for existing asset with same name
        if asset.name in self.assets:
            self.logger.warning(f"Asset '{asset.name}' already exists, replacing")
            # Close existing asset if needed
            if hasattr(self.assets[asset.name], 'close') and callable(self.assets[asset.name].close):
                await self.assets[asset.name].close()
        
        # Set the exchange if asset doesn't have one
        if not asset.exchange and self.exchange:
            asset.set_exchange(self.exchange)
            
        # Initialize the asset if needed
        if hasattr(asset, 'initialize') and callable(asset.initialize):
            try:
                await asset.initialize()
            except Exception as e:
                self.logger.error(f"Failed to initialize asset {asset.name}: {e}")
                raise
            
        # Register asset with its canonical name
        self.assets[asset.name] = asset
        
        # Also register asset with alternate names for better lookups
        self._register_asset_aliases(asset)
        
        self.logger.info(f"Added asset {asset.name} to portfolio")
        
        # Notify risk manager of new asset
        await self.notify_risk_manager('asset_added', {
            'asset_name': asset.name,
            'asset_type': asset.__class__.__name__
        })
    
    def _register_asset_aliases(self, asset: Asset) -> None:
        """
        Register asset with alternate names/formats for more robust lookups
        
        Args:
            asset: Asset to register
        """
        # Skip for non-standard asset names
        if '/' not in asset.name:
            return
            
        # Create common variants
        base, quote = asset.name.split('/')
        variants = [
            asset.name.replace('/', ''),  # No separator
            f"{base}_{quote}",            # Underscore separator
            f"{base}-{quote}"             # Dash separator
        ]
        
        # Register variants if they don't conflict with existing assets
        for variant in variants:
            if variant not in self.assets:
                # Store reference to the original asset
                self.logger.debug(f"Registered alias {variant} for asset {asset.name}")
                self.assets[variant] = asset

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

    async def update_market_data(self, data_map: Dict[str, pd.DataFrame]) -> None:
        """
        Update assets with the latest market data
        
        Args:
            data_map: Dictionary mapping symbols to their market data DataFrames
        """
        if not data_map:
            self.logger.warning("Empty data map provided to update_market_data")
            return
            
        # Create update tasks for each asset with corresponding data
        update_tasks = []
        
        for asset_name, asset in self.assets.items():
            # Direct match - asset name matches a symbol in data_map
            if asset_name in data_map:
                df = data_map[asset_name]
                if not df.empty:
                    update_tasks.append(asset.update_data(df))
            else:
                # Try alternative matching for pairs like BTC/USDT matching BTC asset
                matched = False
                for symbol, df in data_map.items():
                    # Check if symbol starts with asset name (e.g., BTC/USDT for BTC asset)
                    # Or if asset name contains symbol (e.g., "BTCUSDT" asset for "BTC/USDT" symbol)
                    if (symbol.startswith(asset_name + "/") or 
                        asset_name.startswith(symbol + "/") or
                        asset_name.replace("/", "") == symbol.replace("/", "")):
                        if not df.empty:
                            update_tasks.append(asset.update_data(df))
                            matched = True
                            break
                
                if not matched:
                    self.logger.debug(f"No market data provided for asset {asset_name}")
        
        # Execute all updates concurrently
        if update_tasks:
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Check for exceptions
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    asset_name = list(self.assets.keys())[i]
                    self.logger.error(f"Error updating {asset_name} with market data: {str(result)}")
        
        # After updating all assets, recalculate portfolio total value
        await self.update_all_values()

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
                if old_value > 0 and abs((self._total_value - old_value) / old_value) > 0.001:
                    await self.notify_risk_manager('significant_value_change', {
                        'old_value': float(old_value),
                        'new_value': float(self._total_value),
                        'change_pct': float((self._total_value - old_value) / old_value)
                    })
            
            return float(self._total_value)
        finally:
            self._is_syncing = False

    async def buy_asset(self, kwargs) -> Dict[str, Any]:
        """
        Buy an existing asset
        
        Args:
            kwargs: Dictionary containing order parameters:
                symbol: Name of the asset to buy
                quantity: Amount to buy
                price: Limit price (optional)
                order_type: Order type (e.g., 'market', 'limit')
                reduce_only: Whether to reduce position only (optional)
                    
        Returns:
            Dict[str, Any]: Buy operation result
                
        Raises:
            ValueError: If asset doesn't exist
        """
        asset_name = kwargs['symbol']
        amount = kwargs['quantity']
        
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        
        asset = self.assets[asset_name]
        
        # Check if the asset is tradable
        if not hasattr(asset, 'buy') or not callable(asset.buy):
            raise ValueError(f"Asset {asset_name} does not support buy operations")
        
        # Apply risk checks via risk manager
        if self.risk_manager:
            validation = await self.risk_manager.validate_order(kwargs)
            if not validation.get('allowed', False):
                return {
                    'success': False,
                    'error': 'Trade rejected due to risk limits',
                    'asset_name': asset_name,
                    'action': 'buy',
                    'amount': amount,
                    'reasons': validation.get('reasons', ['Risk check failed'])
                }
        
        # Make a copy of kwargs to avoid modifying the original
        order_params = kwargs.copy()
        
        # Ensure the direction is correctly set to 'buy'
        order_params['direction'] = 'buy'
        
        # Execute buy
        try:
            result = await asset.buy(order_params)
            
            # Track order if successful
            if result.get('success', False) and 'order_id' in result:
                self._all_orders[result['order_id']] = {
                    'asset_name': asset_name,
                    'action': 'buy',
                    'price': result.get('price', 0.0),
                    'amount': amount,
                    'result': result,
                    'timestamp': time.time(),
                    'params': kwargs
                }
                    
                # Notify risk manager of new order
                await self.notify_risk_manager('order_executed', {
                    'asset_name': asset_name,
                    'direction': 'buy',
                    'price': result.get('price', 0.0),
                    'amount': amount,
                    'order_id': result['order_id'],
                    'result': result
                })
                        
            return result
        except Exception as e:
            self.logger.warning(f"Failed to execute buy for {asset_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'asset_name': asset_name,
                'action': 'buy',
                'amount': amount
            }

    async def sell_asset(self, kwargs) -> Dict[str, Any]:
        """
        Sell an existing asset
        
        Args:
            kwargs: Dictionary containing order parameters:
                symbol: Name of the asset to sell
                quantity: Amount to sell
                price: Limit price (optional)
                order_type: Order type (e.g., 'market', 'limit')
                reduce_only: Whether to reduce position only (optional)
                    
        Returns:
            Dict[str, Any]: Sell operation result
                
        Raises:
            ValueError: If asset doesn't exist
        """
        asset_name = kwargs['symbol']
        amount = kwargs['quantity']
        
        if asset_name not in self.assets:
            raise ValueError(f"Asset '{asset_name}' does not exist")
        
        asset = self.assets[asset_name]
        
        # Check if the asset is tradable
        if not hasattr(asset, 'sell') or not callable(asset.sell):
            raise ValueError(f"Asset {asset_name} does not support sell operations")
        
        # Apply risk checks via risk manager (simplified for sells)
        if self.risk_manager:
            validation = await self.risk_manager.validate_order(kwargs)
            if not validation.get('allowed', False):
                return {
                    'success': False,
                    'error': 'Trade rejected due to risk limits',
                    'asset_name': asset_name,
                    'action': 'sell',
                    'amount': amount,
                    'reasons': validation.get('reasons', ['Risk check failed'])
                }
        
        # Make a copy of kwargs to avoid modifying the original
        order_params = kwargs.copy()
        
        # Ensure the direction is correctly set to 'sell'
        order_params['direction'] = 'sell'
        
        # Execute sell
        try:
            result = await asset.sell(order_params)
            
            # Track order if successful
            if result.get('success', False) and 'order_id' in result:
                self._all_orders[result['order_id']] = {
                    'asset_name': asset_name,
                    'action': 'sell',
                    'price': result.get('price', 0.0),
                    'amount': amount,
                    'result': result,
                    'timestamp': time.time(),
                    'params': kwargs
                }
                    
                # Notify risk manager of new order
                await self.notify_risk_manager('order_executed', {
                    'asset_name': asset_name,
                    'direction': 'sell',
                    'price': result.get('price', 0.0),
                    'amount': amount,
                    'order_id': result['order_id'],
                    'result': result
                })
                        
            return result
        except Exception as e:
            self.logger.warning(f"Failed to execute sell for {asset_name}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'asset_name': asset_name,
                'action': 'sell',
                'amount': amount
            }
        
    async def process_signals(self, signals: pd.DataFrame, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Process trading signals and route to appropriate asset operations
        
        Args:
            signals: DataFrame containing trading signals with columns:
                - symbol: Asset symbol
                - action: Trading action (buy, sell, long, short, close_long, close_short)
                - quantity: Amount to trade
                - price: Optional price for limit orders
                - order_type: Optional order type (default: market)
                - additional parameters depending on asset type
            data: Optional market data for reference
            
        Returns:
            Dict containing execution results
        """
        if signals.empty:
            self.logger.info("No signals to process")
            return {"success": True, "message": "No signals to process", "orders": []}
        
        # Validate required columns
        required_columns = ['symbol', 'action']
        missing_columns = [col for col in required_columns if col not in signals.columns]
        if missing_columns:
            return {
                "success": False, 
                "error": f"Missing required columns in signals: {missing_columns}",
                "orders": []
            }
        
        # Process each signal and collect results
        order_results = []
        for _, signal in signals.iterrows():
            try:
                # Extract basic signal information
                symbol = signal['symbol']
                action = signal['action'].lower()
                quantity = signal.get('quantity')
                
                # Handle auto-sizing if quantity not provided
                if quantity is None and data is not None:
                    quantity = self._auto_size_position(symbol, action, data)
                    
                if quantity is None or quantity <= 0:
                    self.logger.warning(f"Invalid quantity for {symbol}: {quantity}")
                    order_results.append({
                        "success": False,
                        "symbol": symbol,
                        "action": action,
                        "error": "Invalid or missing quantity"
                    })
                    continue
                    
                # Find corresponding asset
                asset = self._find_asset_by_symbol(symbol)
                if asset is None:
                    self.logger.warning(f"Asset not found: {symbol}")
                    order_results.append({
                        "success": False,
                        "symbol": symbol,
                        "action": action,
                        "error": "Asset not found"
                    })
                    continue
                    
                # Prepare order parameters
                order_params = self._prepare_order_params(signal, quantity)
                
                # Apply risk checks before execution
                if self.risk_manager:
                    validation = await self.risk_manager.validate_order(order_params)
                    if not validation.get('allowed', False):
                        reasons = validation.get('reasons', ['Risk check failed'])
                        self.logger.warning(f"Order rejected by risk manager for {symbol}: {reasons}")
                        order_results.append({
                            "success": False,
                            "symbol": symbol,
                            "action": action,
                            "quantity": quantity,
                            "error": f"Risk check failed: {reasons}"
                        })
                        continue
                
                # Route to appropriate asset method based on action
                result = await self._route_order_to_asset(asset, action, order_params)
                
                # Track successful orders
                if result.get('success', False) and 'order_id' in result:
                    self._all_orders[result['order_id']] = {
                        'asset_name': symbol,
                        'action': action,
                        'price': result.get('price', 0.0),
                        'quantity': quantity,
                        'result': result,
                        'timestamp': time.time(),
                        'params': order_params
                    }
                    
                    # Notify risk manager of executed order
                    await self.notify_risk_manager('order_executed', {
                        'asset_name': symbol,
                        'action': action,
                        'price': result.get('price', 0.0),
                        'quantity': quantity,
                        'order_id': result['order_id'],
                        'result': result
                    })
                
                order_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing signal for {signal.get('symbol', 'unknown')}: {str(e)}")
                order_results.append({
                    "success": False,
                    "symbol": signal.get('symbol', 'unknown'),
                    "action": signal.get('action', 'unknown'),
                    "error": str(e)
                })
        
        # Summarize results
        success_count = sum(1 for r in order_results if r.get('success', False))
        self.logger.info(f"Processed {len(order_results)} signals: {success_count} successful, {len(order_results) - success_count} failed")
        
        return {
            "success": True,
            "orders": order_results,
            "successful": success_count,
            "failed": len(order_results) - success_count
        }

    def _prepare_order_params(self, signal: pd.Series, quantity: float) -> Dict[str, Any]:
        """
        Prepare order parameters from signal data
        
        Args:
            signal: Signal series from DataFrame row
            quantity: Order quantity (may be auto-calculated)
            
        Returns:
            Dict with order parameters
        """
        params = {
            'symbol': signal['symbol'],
            'quantity': quantity,
            'direction': signal['action'].lower()  # Initially set to action, will be mapped if needed
        }
        
        # Add optional parameters if present
        if 'price' in signal and not pd.isna(signal['price']):
            params['price'] = signal['price']
            
        if 'order_type' in signal and not pd.isna(signal['order_type']):
            params['order_type'] = signal['order_type'].lower()
        else:
            params['order_type'] = 'market'  # Default to market orders
            
        # Handle other common parameters
        for param in ['leverage', 'stop_price', 'take_profit', 'reduce_only', 'time_in_force']:
            if param in signal and not pd.isna(signal[param]):
                params[param] = signal[param]
        
        # Add any remaining parameters that might be asset-specific
        for column, value in signal.items():
            if column not in params and not pd.isna(value):
                params[column] = value
                
        return params

    def _auto_size_position(self, symbol: str, action: str, data: pd.DataFrame) -> Optional[float]:
        """
        Calculate position size based on portfolio value and risk settings
        
        Args:
            symbol: Asset symbol
            action: Trading action
            data: Market data DataFrame
            
        Returns:
            Calculated quantity or None if calculation fails
        """
        try:
            # Get latest price from data
            if data is None or data.empty or 'close' not in data.columns:
                self.logger.warning(f"Cannot auto-size position for {symbol}: No price data available")
                return None
                
            latest_price = data['close'].iloc[-1]
            if latest_price <= 0:
                self.logger.warning(f"Invalid price for auto-sizing {symbol}: {latest_price}")
                return None
                
            # Get portfolio value
            portfolio_value = self.get_total_value()
            
            # Get position size percentage from config or use default
            position_size_pct = self.config.get("trading", "position_size_pct", default=0.02)
            
            # Calculate quantity
            quantity = (portfolio_value * position_size_pct) / latest_price
            
            # Round to appropriate precision (could be asset-specific)
            asset = self._find_asset_by_symbol(symbol)
            precision = 8  # Default precision
            if asset and hasattr(asset, 'precision'):
                precision = asset.precision
                
            rounded_quantity = round(quantity, precision)
            
            self.logger.info(f"Auto-sized {action} for {symbol}: {rounded_quantity} units ({position_size_pct*100:.1f}% of portfolio)")
            return rounded_quantity
            
        except Exception as e:
            self.logger.error(f"Error auto-sizing position for {symbol}: {str(e)}")
            return None

    async def _route_order_to_asset(self, asset: Asset, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route order to appropriate asset method based on action
        
        Args:
            asset: Asset instance
            action: Trading action (buy, sell, long, short, close_long, close_short)
            params: Order parameters
            
        Returns:
            Order execution result
        """
        # Map action to asset method
        action = action.lower()
        
        # Basic spot trading operations
        if action in ['buy', 'sell']:
            # For spot assets, just use the basic buy/sell methods
            method = getattr(asset, action, None)
            if method and callable(method):
                return await method(params)
                
        # Futures/options position operations
        elif action == 'long' or action == 'open_long':
            # For futures/options, use the position opening methods
            method = getattr(asset, 'open_long_position', None)
            if method and callable(method):
                return await method(params['quantity'], **params)
                
        elif action == 'short' or action == 'open_short':
            # For futures/options, use the position opening methods
            method = getattr(asset, 'open_short_position', None)
            if method and callable(method):
                return await method(params['quantity'], **params)
                
        elif action == 'close_long':
            # For futures/options, use the position closing methods
            method = getattr(asset, 'close_long_position', None)
            if method and callable(method):
                return await method(params.get('quantity'), **params)
                
        elif action == 'close_short':
            # For futures/options, use the position closing methods
            method = getattr(asset, 'close_short_position', None)
            if method and callable(method):
                return await method(params.get('quantity'), **params)
                
        # If action not recognized or method not available, return error
        self.logger.error(f"Unsupported action {action} for asset {asset.name} or method not implemented")
        return {
            "success": False,
            "symbol": params['symbol'],
            "action": action,
            "error": f"Unsupported action or method not implemented: {action}"
        }

    async def execute_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single order with the specified parameters
        
        Args:
            order_params: Dictionary containing:
                - symbol: Asset symbol
                - action: Trading action (buy, sell, long, short, etc.)
                - quantity: Amount to trade
                - price: Optional price for limit orders
                - Additional parameters depending on asset type
                
        Returns:
            Dict with order execution result
        """
        if 'symbol' not in order_params:
            return {"success": False, "error": "Missing 'symbol' parameter"}
            
        if 'action' not in order_params:
            return {"success": False, "error": "Missing 'action' parameter"}
            
        symbol = order_params['symbol']
        action = order_params['action'].lower()
        
        # Find the asset
        asset = self._find_asset_by_symbol(symbol)
        if asset is None:
            return {"success": False, "error": f"Asset not found: {symbol}"}
        
        # Apply risk validation if needed
        if self.risk_manager:
            validation = await self.risk_manager.validate_order(order_params)
            if not validation.get('allowed', False):
                reasons = validation.get('reasons', ['Risk check failed'])
                return {
                    "success": False,
                    "symbol": symbol,
                    "action": action,
                    "error": f"Risk check failed: {reasons}"
                }
        
        # Route to appropriate asset method
        result = await self._route_order_to_asset(asset, action, order_params)
        
        # Track successful orders
        if result.get('success', False) and 'order_id' in result:
            self._all_orders[result['order_id']] = {
                'asset_name': symbol,
                'action': action,
                'price': result.get('price', 0.0),
                'quantity': order_params.get('quantity', 0.0),
                'result': result,
                'timestamp': time.time(),
                'params': order_params
            }
            
            # Notify risk manager of executed order
            await self.notify_risk_manager('order_executed', {
                'asset_name': symbol,
                'action': action,
                'price': result.get('price', 0.0),
                'quantity': order_params.get('quantity', 0.0),
                'order_id': result['order_id'],
                'result': result
            })
        
        return result

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an active order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            Dict with cancellation result
        """
        # Check if we're tracking this order
        if order_id in self._all_orders:
            order_info = self._all_orders[order_id]
            asset_name = order_info['asset_name']
            
            # Find the asset
            asset = self._find_asset_by_symbol(asset_name)
            if asset is None:
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": f"Asset not found for order: {asset_name}"
                }
                
            # Cancel the order through the asset
            if hasattr(asset, 'cancel_order') and callable(asset.cancel_order):
                result = await asset.cancel_order(order_id)
                
                # Notify risk manager of cancellation
                if result.get('success', False):
                    await self.notify_risk_manager('order_canceled', {
                        'asset_name': asset_name,
                        'order_id': order_id
                    })
                    
                return result
        
        # If order not tracked, try canceling on all assets
        for asset_name, asset in self.assets.items():
            if hasattr(asset, 'cancel_order') and callable(asset.cancel_order):
                try:
                    result = await asset.cancel_order(order_id)
                    if result.get('success', False):
                        # Notify risk manager of cancellation
                        await self.notify_risk_manager('order_canceled', {
                            'asset_name': asset_name,
                            'order_id': order_id
                        })
                        
                        return result
                except Exception:
                    continue
        
        return {
            "success": False,
            "order_id": order_id,
            "error": "Order not found"
        }

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get current status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Dict with order status information
        """
        # Check if we're tracking this order
        if order_id in self._all_orders:
            order_info = self._all_orders[order_id]
            asset_name = order_info['asset_name']
            
            # Find the asset
            asset = self._find_asset_by_symbol(asset_name)
            if asset is None:
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": f"Asset not found for order: {asset_name}"
                }
                
            # Get order status through the asset
            if hasattr(asset, 'get_order_status') and callable(asset.get_order_status):
                return await asset.get_order_status(order_id)
        
        # If order not tracked, try checking on all assets
        for asset_name, asset in self.assets.items():
            if hasattr(asset, 'get_order_status') and callable(asset.get_order_status):
                try:
                    result = await asset.get_order_status(order_id)
                    if result.get('success', False):
                        return result
                except Exception:
                    continue
        
        return {
            "success": False,
            "order_id": order_id,
            "error": "Order not found"
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
        
    async def get_account_balance(self) -> Dict[str, float]:
        """
        Get account balance from the exchange
        
        Returns:
            Dict[str, float]: Currency balances where key is currency code and value is amount
        """
        if not self.exchange:
            self.logger.error("No exchange available to fetch account balance")
            return {}
        
        try:
            # Ensure exchange is initialized
            if not self.exchange.is_initialized():
                await self.exchange.initialize()
            
            # Fetch balance from exchange
            balance_data = await self.exchange.fetch_balance()
            
            if not balance_data or 'currencies' not in balance_data:
                self.logger.error("Invalid balance data format returned from exchange")
                return {}
            
            # Extract free balances for each currency
            balances = {}
            for currency, data in balance_data['currencies'].items():
                if 'free' in data and data['free'] > 0:
                    balances[currency] = data['free']
            
            self.logger.info(f"Retrieved account balance for {len(balances)} currencies")
            return balances
        
        except Exception as e:
            self.logger.error(f"Failed to fetch account balance: {str(e)}")
            return {}

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Get all open orders from the exchange
        
        Returns:
            List[Dict[str, Any]]: List of open orders
        """
        if not self.exchange:
            self.logger.error("No exchange available to fetch open orders")
            return []
        
        try:
            # Ensure exchange is initialized
            if not self.exchange.is_initialized():
                await self.exchange.initialize()
            
            # Fetch open orders from exchange
            open_orders = await self.exchange.fetch_open_orders()
            
            self.logger.info(f"Retrieved {len(open_orders)} open orders from exchange")
            return open_orders
        
        except Exception as e:
            self.logger.error(f"Failed to fetch open orders: {str(e)}")
            return []
    
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
        
    def is_risk_breached(self) -> bool:
        """
        Check if any risk limits have been breached
        
        This method checks with the risk manager to determine if any risk 
        parameters have been violated that would require stopping the trading.
        
        Returns:
            bool: True if risk limits are breached, False otherwise
        """
        # If no risk manager is available, assume no risk breach
        if not self.risk_manager:
            return False
        
        # Check if risk manager has a method to check risk breach
        if hasattr(self.risk_manager, 'is_risk_breached') and callable(getattr(self.risk_manager, 'is_risk_breached')):
            try:
                return self.risk_manager.is_risk_breached()
            except Exception as e:
                self.logger.error(f"Error checking risk breach status: {e}")
                # Default to stopping trading if we can't check risk (safety first)
                return True
        
        # Alternative check: look for critical risk levels
        if hasattr(self.risk_manager, 'get_risk_level') and callable(getattr(self.risk_manager, 'get_risk_level')):
            try:
                risk_level = self.risk_manager.get_risk_level()
                # Assume risk levels might be numeric (higher = more risky) or string-based
                if isinstance(risk_level, (int, float)):
                    critical_threshold = self.config.get("risk", "critical_threshold", default=0.8)
                    return risk_level >= critical_threshold
                elif isinstance(risk_level, str):
                    critical_levels = ['critical', 'emergency', 'severe', 'extreme']
                    return risk_level.lower() in critical_levels
            except Exception as e:
                self.logger.error(f"Error getting risk level: {e}")
                return True
        
        # If we can't determine risk status, default to safe (continue trading)
        return False

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
    
    async def record_batch_trades(self, trades: List[Dict[str, Any]]) -> None:
        """
        Record a batch of trades for portfolio tracking.
        
        Args:
            trades: List of trade dictionaries
        """
        if not trades:
            return
            
        self.logger.info(f"Recording batch of {len(trades)} trades")
        
        # Process each trade individually
        for trade in trades:
            # Extract trade details
            symbol = trade.get('symbol')
            direction = trade.get('direction', 'unknown').lower()
            quantity = trade.get('quantity', 0.0)
            price = trade.get('price', 0.0)
            timestamp = trade.get('timestamp')
            
            # Skip invalid trades
            if not symbol or not quantity or not price:
                self.logger.warning(f"Skipping invalid trade: {trade}")
                continue
                
            # Find the corresponding asset with flexible symbol matching
            asset = self._find_asset_by_symbol(symbol)
            
            if not asset:
                self.logger.warning(f"Asset {symbol} not found, skipping trade")
                continue
                
            # Update asset position
            if hasattr(asset, '_update_position_from_trade'):
                asset._update_position_from_trade(trade)
            elif hasattr(asset, '_update_position_from_filled_order'):
                # Create a synthetic order for the asset to process
                from src.portfolio.execution.order import Order, Direction, OrderStatus, OrderType
                
                # Create synthetic order - adding OrderType.MARKET
                synth_order = Order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,  # Add this missing parameter
                    direction=Direction.BUY if direction == 'buy' else Direction.SELL,
                    quantity=quantity
                )
                
                # Set order as filled
                synth_order.status = OrderStatus.FILLED
                synth_order.filled_quantity = quantity
                synth_order.avg_filled_price = price
                
                asset._update_position_from_filled_order(synth_order)
            else:
                # Basic update for any asset type
                if direction == 'buy':
                    asset._position_size += Decimal(str(quantity))
                elif direction == 'sell':
                    asset._position_size -= Decimal(str(quantity))
                    
                asset._last_price = Decimal(str(price))
                asset._value = asset._position_size * asset._last_price
                
            # Notify risk manager
            await self.notify_risk_manager('trade_recorded', {
                'asset_name': symbol,
                'direction': direction,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp
            })
                
        # Update portfolio value after processing each trades
        await self.update_all_values()
        
        self.logger.info(f"Batch of {len(trades)} trades recorded successfully")

    def _find_asset_by_symbol(self, symbol: str) -> Optional[Asset]:
        """
        Find an asset by symbol with flexible matching to handle different symbol formats.
        
        Args:
            symbol: Symbol to look for (e.g., 'BTC/USDT', 'BTCUSDT', etc.)
            
        Returns:
            Optional[Asset]: Asset instance if found, None otherwise
        """
        # Direct match
        if symbol in self.assets:
            return self.assets[symbol]
        
        # Try with different separators and formats
        symbol_variants = [
            symbol,                           # Original (e.g., "BTC/USDT")
            symbol.replace("/", ""),          # No separator (e.g., "BTCUSDT")
            symbol.replace("/", "_"),         # Underscore separator (e.g., "BTC_USDT")
            symbol.replace("/", "-"),         # Dash separator (e.g., "BTC-USDT")
            symbol.upper(),                   # Uppercase (e.g., "BTC/USDT")
            symbol.lower()                    # Lowercase (e.g., "btc/usdt")
        ]
        
        # Try with different base/quote arrangements
        if "/" in symbol:
            base, quote = symbol.split("/")
            symbol_variants.extend([
                f"{quote}/{base}",            # Reversed (e.g., "USDT/BTC")
                f"{base.upper()}/{quote.upper()}",  # All uppercase
                f"{base.lower()}/{quote.lower()}"   # All lowercase
            ])
        
        # Check all variants
        for variant in symbol_variants:
            if variant in self.assets:
                self.logger.debug(f"Found asset for {symbol} using variant {variant}")
                return self.assets[variant]
        
        # Try partial matching for assets that might be registered with additional info
        symbol_base = symbol.split("/")[0] if "/" in symbol else symbol
        for asset_name, asset in self.assets.items():
            # Check if asset name starts with the base symbol
            if asset_name.startswith(symbol_base):
                self.logger.debug(f"Found partial match for {symbol}: {asset_name}")
                return asset
            # Check if the base symbol is in the asset name
            elif symbol_base in asset_name:
                self.logger.debug(f"Found partial match for {symbol} in {asset_name}")
                return asset
        
        # No match found
        return None
        
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