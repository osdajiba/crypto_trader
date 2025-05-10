#!/usr/bin/env python3
# src/trading/live.py

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime
import time

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.mode.base import BaseTradingMode, PerformanceCategory
from src.common.helpers import time_func


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
        
        # Live-specific performance metrics
        self.exchange_latency = []
        self.order_execution_latency = []
        
        # Enable real-time latency monitoring
        self.monitor_exchange_latency = self.config.get(
            "live_trading", "monitor_exchange_latency", default=True
        )
        
        # Set latency thresholds for logging
        self.high_latency_threshold = self.config.get(
            "live_trading", "high_latency_threshold", default=1.0
        )  # seconds
        
        # Latency monitoring intervals
        self.latency_log_interval = self.config.get(
            "live_trading", "latency_log_interval", default=10
        )  # iterations
        self._latency_counter = 0
    
    @time_func("live_initialize")
    async def _initialize_mode_specific(self) -> None:
        """Initialize live trading mode components"""
        self.logger.info("Initializing live-specific components")

        # Verify exchange connectivity
        init_start = time.time()
        if not await self._verify_exchange_connectivity():
            raise ConnectionError("Failed to connect to exchange API")
        
        # Record initial exchange latency
        init_latency = time.time() - init_start
        self.exchange_latency.append(init_latency)
        self.logger.info(f"Initial exchange connection latency: {init_latency:.4f}s")
        
        # Verify account credentials and balances
        account_start = time.time()
        if not await self._verify_account():
            raise ValueError("Account verification failed")
        account_latency = time.time() - account_start
        self.logger.info(f"Account verification latency: {account_latency:.4f}s")
    
    @time_func("verify_exchange")
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
            
            # Measure latency
            fetch_start = time.time()
            test_data = await self.data_manager.fetch_realtime(test_symbol, "1m")
            fetch_latency = time.time() - fetch_start
            
            # Record latency for monitoring
            if self.monitor_exchange_latency:
                self.exchange_latency.append(fetch_latency)
                
                # Log high latency
                if fetch_latency > self.high_latency_threshold:
                    self.logger.warning(f"High exchange latency detected: {fetch_latency:.4f}s")

            if test_data is None or test_data.empty:
                self.logger.error(f"Failed to fetch test data from {exchange_name}")
                return False
            
            self.logger.info(f"Successfully connected to {exchange_name} (latency: {fetch_latency:.4f}s)")
            return True
            
        except Exception as e:
            self.logger.error(f"Exchange connectivity error: {e}")
            return False
    
    @time_func("verify_account")
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
                
    @time_func("prepare_live_trading")
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Live mode specific preparation"""
        # Initialize state tracking
        if 'last_balance_check' not in self.state:
            self.state['last_balance_check'] = datetime.now()
        
        # Status check interval (how often to check account status)
        self.status_interval = self.config.get("live_trading", "status_interval", default=3600)  # Default hourly
        
        # Initialize latency tracking
        self.state['latency'] = {
            'exchange_api': [],
            'order_execution': [],
            'data_processing': []
        }
    
    @time_func("live_trading_loop")
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute live trading loop"""
        try:
            # Trading loop
            while self._should_continue():
                # Record iteration start time for latency tracking
                iteration_start = time.time()
                current_time = datetime.now()
                
                # Periodically update account status
                if (current_time - self.state['last_balance_check']).total_seconds() > self.status_interval:
                    status_start = time.time()
                    await self._update_account_status()
                    status_latency = time.time() - status_start
                    
                    if self.enable_performance_tracking:
                        self._record_performance_metric(
                            PerformanceCategory.PORTFOLIO_UPDATE,
                            status_latency
                        )
                    
                    self.state['last_balance_check'] = current_time
                
                # Fetch market data using the base method with timing already included
                data_map = await self._fetch_market_data(symbols, timeframe)
                
                # Process market data if we have any
                if data_map:
                    # Process market data with performance tracking
                    trades = await self._process_market_data(data_map)
                    
                    # Track order execution latency if trades were made
                    if trades and self.monitor_exchange_latency:
                        avg_exec_time = self.performance_metrics.get(PerformanceCategory.TRADE_EXECUTION.value, [0])[-1]
                        self.order_execution_latency.append(avg_exec_time)
                else:
                    self.logger.warning("No market data received, skipping trading cycle")
                
                # Check risk control
                risk_start = time.time()
                if hasattr(self, 'risk_manager') and self.risk_manager and hasattr(self.risk_manager, 'execute_risk_control'):
                    if await self.risk_manager.execute_risk_control():
                        self.logger.critical("Risk control triggered, stopping live trading")
                        self._running = False
                        break
                
                risk_latency = time.time() - risk_start
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.RISK_MANAGEMENT,
                        risk_latency
                    )
                
                # Calculate total iteration time
                iteration_time = time.time() - iteration_start
                if self.enable_performance_tracking:
                    self._record_performance_metric(
                        PerformanceCategory.TOTAL_ITERATION,
                        iteration_time
                    )
                
                # Increment latency counter and log if needed
                self._latency_counter += 1
                if self.monitor_exchange_latency and self._latency_counter % self.latency_log_interval == 0:
                    self._log_latency_metrics()
                
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
    
    @time_func("update_account")
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
    
    def _log_latency_metrics(self) -> None:
        """Log current latency metrics"""
        if not self.exchange_latency or not self.enable_performance_tracking:
            return
            
        # Calculate exchange latency stats
        avg_exchange = sum(self.exchange_latency) / len(self.exchange_latency) if self.exchange_latency else 0
        max_exchange = max(self.exchange_latency) if self.exchange_latency else 0
        
        # Calculate order execution latency if available
        avg_execution = sum(self.order_execution_latency) / len(self.order_execution_latency) if self.order_execution_latency else 0
        max_execution = max(self.order_execution_latency) if self.order_execution_latency else 0
        
        # Get data latency from performance metrics
        data_latency = self.performance_metrics.get(PerformanceCategory.DATA_FETCH.value, [0])
        avg_data = sum(data_latency) / len(data_latency) if data_latency else 0
        max_data = max(data_latency) if data_latency else 0
        
        # Log the metrics
        self.logger.info("==== Latency Metrics ====")
        self.logger.info(f"Exchange API: avg={avg_exchange:.4f}s, max={max_exchange:.4f}s")
        self.logger.info(f"Order Execution: avg={avg_execution:.4f}s, max={max_execution:.4f}s")
        self.logger.info(f"Data Fetch: avg={avg_data:.4f}s, max={max_data:.4f}s")
        
        # Check for potential issues
        if avg_exchange > self.high_latency_threshold:
            self.logger.warning(f"HIGH LATENCY ALERT: Exchange API latency ({avg_exchange:.4f}s) exceeds threshold ({self.high_latency_threshold}s)")
        
        if avg_execution > self.high_latency_threshold:
            self.logger.warning(f"HIGH LATENCY ALERT: Order execution latency ({avg_execution:.4f}s) exceeds threshold ({self.high_latency_threshold}s)")
    
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
            
        # Add latency metrics if available
        if self.monitor_exchange_latency and self.exchange_latency:
            report['latency_metrics'] = {
                'exchange_api': {
                    'avg': sum(self.exchange_latency) / len(self.exchange_latency),
                    'max': max(self.exchange_latency),
                    'min': min(self.exchange_latency),
                    'samples': len(self.exchange_latency)
                }
            }
            
            # Add order execution latency if available
            if self.order_execution_latency:
                report['latency_metrics']['order_execution'] = {
                    'avg': sum(self.order_execution_latency) / len(self.order_execution_latency),
                    'max': max(self.order_execution_latency),
                    'min': min(self.order_execution_latency),
                    'samples': len(self.order_execution_latency)
                }
    
    @time_func("live_shutdown")
    async def shutdown(self) -> None:
        """Shutdown live trading mode"""
        self.logger.info("Shutting down live trading mode")
        
        # Cancel any open orders if configured
        should_cancel_orders = self.config.get("live_trading", "cancel_orders_on_shutdown", default=True)
        if should_cancel_orders and self.portfolio and hasattr(self.portfolio, 'get_open_orders'):
            try:
                # Get open orders
                orders_start = time.time()
                open_orders = await self.portfolio.get_open_orders()
                orders_latency = time.time() - orders_start
                
                if open_orders:
                    self.logger.info(f"Cancelling {len(open_orders)} open orders (fetch latency: {orders_latency:.4f}s)")
                    for order in open_orders:
                        order_id = order.get('id') or order.get('order_id')
                        symbol = order.get('symbol')
                        if order_id:
                            cancel_start = time.time()
                            await self.portfolio.cancel_order(order_id)
                            cancel_latency = time.time() - cancel_start
                            self.logger.debug(f"Cancelled order {order_id} (latency: {cancel_latency:.4f}s)")
            except Exception as e:
                self.logger.error(f"Error cancelling open orders: {e}")
                
        # Close execution engine
        if self.portfolio and hasattr(self.portfolio, 'shutdown'):
            await self.portfolio.shutdown()
            
        # Call parent shutdown
        await super().shutdown()
            
        self.logger.info("Live trading mode shutdown complete")