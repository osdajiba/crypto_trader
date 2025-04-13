# src/mode/paper_trading_mode.py

from typing import Dict, List, Any, Optional
import asyncio
import pandas as pd
from datetime import datetime

from src.trading.modes.base import BaseTradingMode
from execution.manager import ExecutionEngine


class PaperTradingMode(BaseTradingMode):
    """Paper trading mode implementation"""
    
    async def initialize(self) -> None:
        """Initialize paper trading mode components"""
        self.logger.info("Initializing paper trading mode")
        
        # Create paper trading specific execution engine
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="paper"
        )
        
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
        # Paper trading doesn't need additional setup
        pass
    
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
                # Get market data
                data_map = await self.data_manager.fetch_all_data_for_symbols(symbols, timeframe)
                
                # Process market data
                await self._process_market_data(data_map)
                
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
        
        if hasattr(self, 'execution_engine') and self.execution_engine:
            await self.execution_engine.close()
        
        self._running = False
        self.logger.info("Paper trading mode shutdown complete")