#!/usr/bin/env python3
# src/trading/live.py

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.mode.base import BaseTradingMode


@register_factory_class('trading_mode_factory', 'live', 
                       description="Live trading (uses real funds on exchange)", 
                       features=["real_time_data", "real_execution", "risk_management"],
                       category="production")
class LiveMode(BaseTradingMode):
    """Live trading mode implementation with exchange connectivity"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize live trading mode"""
        super().__init__(config, params)
        self.status_interval = 3600  # Default hourly status check
        self.execution_factory = None  # Will be initialized later
    
    async def _initialize_mode_specific(self) -> None:
        """Initialize live trading mode components"""
        self.logger.info("Initializing live-specific components")

        # Verify exchange connectivity
        if not await self._verify_exchange_connectivity():
            raise ConnectionError("Failed to connect to exchange API")
        
        # Verify account credentials and balances
        if not await self._verify_account():
            raise ValueError("Account verification failed")
    
    async def _verify_exchange_connectivity(self) -> bool:
        """Verify connectivity to the exchange API"""
        try:
            # Get exchange name from config
            exchange_name = self.config.get("exchange", "name", default="")
            if not exchange_name:
                self.logger.error("Exchange name not configured")
                return False
            
            # Try to fetch a small amount of public data as a connectivity test
            test_symbol = self.config.get("exchange", "test_symbol", default="BTC/USDT")
            test_data = await self.data_source.fetch_realtime(test_symbol, "1m")

            if test_data is None or test_data.empty:
                self.logger.error(f"Failed to fetch test data from {exchange_name}")
                return False
            
            self.logger.info(f"Successfully connected to {exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Exchange connectivity error: {e}")
            return False
    
    async def _verify_account(self) -> bool:
        """Verify account credentials and check balances"""
        try:
            # Fetch account balance
            balance = await self.portfolio.get_account_balance()
            
            if not balance:
                self.logger.error("Failed to fetch account balance")
                return False
            
            # Check minimum required balance
            min_balance = self.config.get("live_trading", "min_required_balance", default=0)
            
            # Calculate total equity value (simple sum of all balances)
            total_equity = sum(balance.values())
            
            # Store initial balance data
            self.state['initial_balance'] = balance
            self.state['current_equity'] = total_equity
            self.state['peak_equity'] = total_equity
            self.state['last_balance_check'] = datetime.now()
            
            # Log balance information
            self.logger.info(f"Account balance verified: {balance}")
            
            if total_equity < min_balance:
                self.logger.warning(f"Account balance below minimum required: {total_equity} < {min_balance}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Account verification error: {e}")
            return False
                
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Live mode specific preparation"""
        # Initialize state tracking
        if 'last_balance_check' not in self.state:
            self.state['last_balance_check'] = datetime.now()
        
        # Status check interval (how often to check account status)
        self.status_interval = self.config.get("live_trading", "status_interval", default=3600)  # Default hourly
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute live trading loop"""
        try:
            # Trading loop
            while self._should_continue():
                current_time = datetime.now()
                
                # Periodically update account status
                if (current_time - self.state['last_balance_check']).total_seconds() > self.status_interval:
                    await self._update_account_status()
                    self.state['last_balance_check'] = current_time
                
                # Fetch market data using the new method
                data_map = await self._fetch_market_data(symbols, timeframe)
                
                # Process market data if we have any
                if data_map:
                    await self._process_market_data(data_map)
                else:
                    self.logger.warning("No market data received, skipping trading cycle")
                
                # Check risk control
                if hasattr(self, 'risk_manager') and self.risk_manager and hasattr(self.risk_manager, 'execute_risk_control'):
                    if await self.risk_manager.execute_risk_control():
                        self.logger.critical("Risk control triggered, stopping live trading")
                        self._running = False
                        break
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return basic result
            return {
                "status": "completed" if self._running else "stopped_by_risk_control", 
                "symbols": symbols,
                "timeframe": timeframe
            }
            
        except asyncio.CancelledError:
            self.logger.warning("Live trading cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Live trading error: {e}")
            raise
    
    async def _update_account_status(self) -> None:
        """Update account balance and position information"""
        try:
            # Fetch current account balance
            balance = await self.portfolio.get_account_balance()
            
            if balance:
                # Calculate total equity (simple sum of all balances)
                total_equity = sum(balance.values())
                
                # Update state with current equity
                self.state['current_equity'] = total_equity
                
                # Update peak equity if new peak reached
                if total_equity > self.state.get('peak_equity', 0):
                    self.state['peak_equity'] = total_equity
                
                # Calculate drawdown
                peak_equity = self.state.get('peak_equity', total_equity)
                if peak_equity > 0:
                    drawdown = (peak_equity - total_equity) / peak_equity
                    self.state['max_drawdown'] = max(self.state.get('max_drawdown', 0), drawdown)
                
                # Log updated balance
                self.logger.info(f"Account status updated | Equity: ${total_equity:,.2f} | " +
                            f"Drawdown: {self.state.get('max_drawdown', 0)*100:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to update account status: {e}")
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add live trading specific metrics to report"""
        # Add exchange info
        report['exchange'] = self.config.get("exchange", "name", default="unknown")
        
        # Add live trading parameters
        report['live_trading_params'] = {
            'status_interval': self.status_interval,
            'min_required_balance': self.config.get("live_trading", "min_required_balance", default=0)
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown live trading mode"""
        self.logger.info("Shutting down live trading mode")
        
        # Cancel any open orders if configured
        should_cancel_orders = self.config.get("live_trading", "cancel_orders_on_shutdown", default=True)
        if should_cancel_orders and self.portfolio and hasattr(self.portfolio, 'get_open_orders'):
            try:
                # Get open orders
                open_orders = await self.portfolio.get_open_orders()
                if open_orders:
                    self.logger.info(f"Cancelling {len(open_orders)} open orders")
                    for order in open_orders:
                        order_id = order.get('id') or order.get('order_id')
                        symbol = order.get('symbol')
                        if order_id:
                            await self.portfolio.cancel_order(order_id)
            except Exception as e:
                self.logger.error(f"Error cancelling open orders: {e}")
                
        # Close execution engine
        if self.portfolio and hasattr(self.portfolio, 'shutdown'):
            await self.portfolio.shutdown()
            
        # Call parent shutdown
        await super().shutdown()
            
        self.logger.info("Live trading mode shutdown complete")