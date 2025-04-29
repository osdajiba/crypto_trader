#!/usr/bin/env python3
# src/trading/modes/live.py

from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
from datetime import datetime

from src.trading.base import BaseTradingMode
from src.portfolio.execution.engine import ExecutionEngine


class LiveMode(BaseTradingMode):
    """Live trading mode implementation with exchange connectivity"""
    
    async def initialize(self) -> None:
        """Initialize live trading mode components"""
        self.logger.info("Initializing live trading mode")
        
        # Create live trading specific execution engine
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="live"
        )
        
        # Get strategy configuration
        strategy_name = self.config.get("live_trading", "strategy", default=None)
        strategy_params = self.config.get("live_trading", "strategy_params", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        
        # Verify exchange connectivity
        if not await self._verify_exchange_connectivity():
            raise ConnectionError("Failed to connect to exchange API")
        
        # Verify account credentials and balances
        if not await self._verify_account():
            raise ValueError("Account verification failed")
        
        # Mark as initialized
        self._running = True
        self.logger.info("Live trading mode initialization complete")
    
    async def _verify_exchange_connectivity(self) -> bool:
        """
        Verify connectivity to the exchange API
        
        Returns:
            bool: True if connection is successful
        """
        try:
            # Get exchange name from config
            exchange_name = self.config.get("exchange", "name", default="")
            if not exchange_name:
                self.logger.error("Exchange name not configured")
                return False
            
            # Try to fetch a small amount of public data as a connectivity test
            test_symbol = self.config.get("exchange", "test_symbol", default="BTC/USDT")
            test_data = await self.data_manager.get_real_time_data(test_symbol, "1m")

            if test_data is None or test_data.empty:
                self.logger.error(f"Failed to fetch test data from {exchange_name}")
                return False
            
            self.logger.info(f"Successfully connected to {exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Exchange connectivity error: {e}", exc_info=True)
            return False
    
    async def _verify_account(self) -> bool:
        """
        Verify account credentials and check balances
        
        Returns:
            bool: True if account verification is successful
        """
        try:
            # Fetch account balance
            balance = await self.execution_engine.get_account_balance()
            
            if balance is None:
                self.logger.error("Failed to fetch account balance")
                return False
            
            # Check minimum required balance
            min_balance = self.config.get("live_trading", "min_required_balance", default=0)
            
            # Store initial balance data
            self.state['initial_balance'] = balance
            self.state['current_equity'] = sum(balance.values()) if isinstance(balance, dict) else 0
            self.state['peak_equity'] = self.state['current_equity']
            self.state['last_balance_check'] = datetime.now()
            
            # Log balance information
            self.logger.info(f"Account balance verified: {balance}")
            
            if self.state['current_equity'] < min_balance:
                self.logger.warning(f"Account balance below minimum required: {self.state['current_equity']} < {min_balance}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Account verification error: {e}", exc_info=True)
            return False
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """
        Live mode specific preparation
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
        """
        # Initialize state tracking
        if 'last_balance_check' not in self.state:
            self.state['last_balance_check'] = datetime.now()
        
        # Status check interval (how often to check account status)
        self.status_interval = self.config.get("live_trading", "status_interval", default=3600)  # Default hourly
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Execute live trading loop
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
            
        Returns:
            Dict: Live trading results
        """
        try:
            # Trading loop
            while self._should_continue():
                current_time = datetime.now()
                
                # Periodically update account status
                if (current_time - self.state['last_balance_check']).total_seconds() > self.status_interval:
                    await self._update_account_status()
                    self.state['last_balance_check'] = current_time
                
                # Get market data
                data_map = await self.data_manager.fetch_all_data_for_symbols(symbols, timeframe)
                
                # Process market data
                await self._process_market_data(data_map)
                
                # Check risk control
                if await self.risk_manager.execute_risk_control():
                    self.logger.critical("Risk control triggered, stopping live trading")
                    self._running = False
                    break
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return default report structure
            return {}
            
        except asyncio.CancelledError:
            self.logger.warning("Live trading cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Live trading error: {e}", exc_info=True)
            raise
    
    async def _update_account_status(self) -> None:
        """Update account balance and position information"""
        try:
            # Fetch current account balance
            balance = await self.execution_engine.get_account_balance()
            
            if balance is not None:
                # Update equity value
                self.state['current_equity'] = sum(balance.values()) if isinstance(balance, dict) else 0
                
                # Update peak equity and drawdown (handled by base class method)
                self._update_performance_metrics()
                
                # Log updated balance
                self.logger.info(f"Account status updated | Equity: ${self.state['current_equity']:,.2f} | " +
                               f"Drawdown: {self.state['max_drawdown']*100:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to update account status: {e}", exc_info=True)
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """
        Add live trading specific metrics to report
        
        Args:
            report: Performance report to update
        """
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
        if should_cancel_orders:
            try:
                # Get open orders
                open_orders = await self.execution_engine.get_open_orders()
                if open_orders:
                    self.logger.info(f"Cancelling {len(open_orders)} open orders")
                    for order in open_orders:
                        await self.execution_engine.cancel_order(order['id'], order.get('symbol'))
            except Exception as e:
                self.logger.error(f"Error cancelling open orders: {e}", exc_info=True)
        
        # Close components
        if hasattr(self, 'strategy') and self.strategy:
            await self.strategy.shutdown()
        
        if hasattr(self, 'execution_engine') and self.execution_engine:
            await self.execution_engine.close()
        
        self._running = False
        self.logger.info("Live trading mode shutdown complete")