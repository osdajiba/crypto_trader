#!/usr/bin/env python3
# src/trading/paper.py

from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from src.common.abstract_factory import register_factory_class
from src.common.config_manager import ConfigManager
from src.mode.base import BaseTradingMode
from src.portfolio.execution.factory import get_execution_factory


@register_factory_class('trading_mode_factory', 'paper', 
                       description="Paper trading (uses real market data without real funds)", 
                       features=["real_time_data", "virtual_execution"],
                       category="simulation")
class PaperMode(BaseTradingMode):
    """Paper trading mode implementation"""
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """Initialize paper trading mode"""
        super().__init__(config, params)
        self.data_source_type = "hybrid"
        self.execution_factory = None
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """Paper trading specific preparation"""
        # Initialize performance tracking data
        self.state['performance_data'] = {
            'start_time': datetime.now(),
            'trade_count': 0
        }
        
        # Get data source type
        self.data_source_type = self.config.get("paper_trading", "data_source", default="hybrid")
        
        self.logger.info(f"Paper trading prepared with {self.data_source_type} data source")
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Execute paper trading loop"""
        try:
            # Trading loop
            while self._should_continue():
                # Get market data
                data_map = await self._fetch_market_data(symbols, timeframe)
                
                # Process market data
                executed_trades = await self._process_market_data(data_map)
                
                # Update performance data
                if executed_trades:
                    self.state['performance_data']['trade_count'] += len(executed_trades)
                
                # Check risk control
                if hasattr(self, 'risk_manager') and self.risk_manager and hasattr(self.risk_manager, 'execute_risk_control'):
                    if await self.risk_manager.execute_risk_control():
                        self.logger.critical("Risk control triggered, stopping paper trading")
                        self._running = False
                        break
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return basic results
            return {
                "status": "completed" if self._running else "stopped_by_risk_control",
                "symbols": symbols, 
                "timeframe": timeframe
            }
            
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
            'data_source': self.data_source_type,
            'run_duration': (datetime.now() - self.state['performance_data']['start_time']).total_seconds() / 60,  # minutes
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown paper trading mode"""
        self.logger.info("Shutting down paper trading mode")
        
        # Call parent shutdown
        await super().shutdown()
            
        self.logger.info("Paper trading mode shutdown complete")