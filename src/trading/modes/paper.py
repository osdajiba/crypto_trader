# src/mode/paper_trading_mode.py

from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
from datetime import datetime

from src.trading.modes.base import BaseTradingMode
from src.application.runtime_builder import RuntimeBuilder


class PaperTradingMode(BaseTradingMode):
    """Paper trading mode implementation"""
    
    async def initialize(self) -> None:
        """Initialize paper trading mode components"""
        self.logger.info("Initializing paper trading mode")
        
        # Get strategy configuration
        strategy_name = self.config.get("paper_trading", "strategy", default=None)
        strategy_params = self.config.get("paper_trading", "strategy_params", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        
        # Mark as initialized
        self._running = True
        self.logger.info("Paper trading mode initialization complete")
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """
        Paper trading specific preparation
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
        """
        RuntimeBuilder(self).build_paper_runtime()
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Execute paper trading loop
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
            
        Returns:
            Dict: Paper trading results
        """
        try:
            # Trading loop
            while self._should_continue():
                # paper mode 只编排 use case，真实交易决策交给 DomainTradingPipeline。
                await self.paper_use_case.run_once(symbols, timeframe)
                
                # Check risk control
                if await self.risk_manager.execute_risk_control():
                    self.logger.critical("Risk control triggered, stopping paper trading")
                    self._running = False
                    break
                
                # Wait for next interval
                await self._sleep_interval()
            
            # Return default report structure
            return {}
            
        except asyncio.CancelledError:
            self.logger.warning("Paper trading cancelled")
            raise

        except Exception as e:
            self.logger.error(f"Paper trading error: {e}", exc_info=True)
            raise

    def _create_domain_pipeline(self):
        """兼容旧入口：模拟盘领域流水线由 RuntimeBuilder 统一组装。"""
        return RuntimeBuilder(self).build_paper_runtime().domain_pipeline
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """
        Add paper trading specific metrics to report
        
        Args:
            report: Performance report to update
        """
        # Add paper trading parameters
        report['paper_trading_params'] = {
            'commission_rate': self.config.get("paper_trading", "commission_rate", 
                                              default=self.config.get("default_config", "user_config", "commission", default=0.001)),
            'data_source': self.config.get("paper_trading", "data_source", default="real-time")
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown paper trading mode"""
        self.logger.info("Shutting down paper trading mode")
        
        if hasattr(self, 'strategy') and self.strategy:
            await self.strategy.shutdown()
        
        self._running = False
        self.logger.info("Paper trading mode shutdown complete")
