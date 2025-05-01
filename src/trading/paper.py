#!/usr/bin/env python3
# src/trading/paper.py

from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
from datetime import datetime

from src.common.abstract_factory import register_factory_class
from src.common.config import ConfigManager
from src.trading.base import BaseTradingMode


@register_factory_class('trading_mode_factory', 'paper', 
                       description="Paper trading (uses real market data without real funds)", 
                       features=["real_time_data", "virtual_execution"],
                       category="simulation")
class PaperMode(BaseTradingMode):
    """Paper trading mode implementation"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize paper trading mode"""
        super().__init__(config, params)
        self.commission_rate = 0
        self.data_source_type = "hybrid"
    
    async def _initialize_mode_specific(self) -> None:
        """Initialize paper trading mode components"""
        self.logger.info("Initializing paper-specific components")
        
        # Create paper trading specific execution engine
        execution_params = {
            "mode": "paper",
            "allow_partial_fills": True
        }
        self.execution_engine = await self.execution_factory.create("paper_execution", execution_params)
        
        # Get strategy configuration from params first, then config
        strategy_name = self.params.get("strategy_name") or self.config.get("paper_trading", "strategy", default=None)
        strategy_params = self.params.get("strategy_params") or self.config.get("paper_trading", "strategy_params", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        self.logger.info(f"Strategy initialized: {self.strategy.__class__.__name__}")
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Paper trading specific preparation"""
        # Initialize performance tracking data
        self.state['performance_data'] = {
            'start_time': datetime.now(),
            'commission_paid': 0.0,
            'trade_count': 0
        }
        
        # Get commission rate
        self.commission_rate = self.config.get(
            "paper_trading", 
            "commission_rate", 
            default=self.config.get("trading", "commission", default=0.001)
        )
        
        # Get data source type
        self.data_source_type = self.config.get("paper_trading", "data_source", default="hybrid")
        
        self.logger.info(f"Paper trading prepared with {self.data_source_type} data source")
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute paper trading loop"""
        try:
            # Trading loop
            while self._should_continue():
                # Get market data
                data_map = await self.data_source.fetch_all_data_for_symbols(symbols, timeframe)
                
                # Process market data
                executed_trades = await self._process_market_data(data_map)
                
                # Update performance data
                if executed_trades:
                    self.state['performance_data']['trade_count'] += len(executed_trades)
                    # Sum up commissions
                    for trade in executed_trades:
                        self.state['performance_data']['commission_paid'] += trade.get('commission', 0)
                
                # Check risk control
                if self.risk_manager and hasattr(self.risk_manager, 'execute_risk_control'):
                    if await self.risk_manager.execute_risk_control():
                        self.logger.critical("Risk control triggered, stopping paper trading")
                        self._running = False
                        break
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return performance report
            return self._generate_report()
            
        except asyncio.CancelledError:
            self.logger.warning("Paper trading cancelled")
            raise
            
        except Exception as e:
            self.logger.error(f"Paper trading error: {e}", exc_info=True)
            raise
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """Add paper trading specific metrics to report"""
        # Add paper trading parameters
        report['paper_trading_params'] = {
            'commission_rate': self.commission_rate,
            'data_source': self.data_source_type,
            'run_duration': (datetime.now() - self.state['performance_data']['start_time']).total_seconds() / 60,  # minutes
            'total_commission_paid': self.state['performance_data']['commission_paid']
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown paper trading mode"""
        self.logger.info("Shutting down paper trading mode")
        
        shutdown_tasks = []
        
        # Close strategy
        if self.strategy:
            shutdown_tasks.append(self.strategy.shutdown())
            
        # Close execution engine
        if self.execution_engine and hasattr(self.execution_engine, 'shutdown'):
            shutdown_tasks.append(self.execution_engine.shutdown())
            
        # Close data source
        if self.data_source:
            shutdown_tasks.append(self.data_source.shutdown())
            
        # Close risk manager
        if self.risk_manager:
            shutdown_tasks.append(self.risk_manager.shutdown())
            
        # Close performance analyzer
        if self.performance_analyzer:
            shutdown_tasks.append(self.performance_analyzer.shutdown())
        
        # Wait for all shutdown tasks to complete
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
            
        self._running = False
        self.logger.info("Paper trading mode shutdown complete")