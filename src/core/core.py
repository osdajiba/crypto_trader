#!/usr/bin/env python3
# src/core/core.py

import asyncio
from typing import Dict, List, Optional, Callable, Any

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.trading.factory import get_trading_mode_factory


class TradingCore:
    """Core coordinator for multi-asset concurrent trading system"""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize the trading core with configuration and mode.
        """
        self.config = config
        self.logger = LogManager.get_logger(name="trading_system")
        
        self.mode = config.get("system", "operational_mode", default="backtest")
        self.backtest_engine = config.get("backtest", "engine", default="ohlcv")
        
        self.async_exec = AsyncExecutor()
        self._running = False
        self.trading_mode = None
        
        # Add this line to store the callback
        self._progress_callback = None
        
        # Create trading mode factory
        self.mode_factory = get_trading_mode_factory(config=self.config)
        
        # Get available pattern
        available_modes = self.mode_factory.get_available_modes()
        
        self.logger.info(
            f"Trading core initialized | Mode: {self.mode} | "
            f"Version: {self.config.get('system', 'version', default='unknown')}"
        )
        self.logger.info(f"Available trading modes: {list(available_modes.keys())}")
        
        # Log backtest engine type if in backtest mode
        if self.mode == "backtest":
            self.logger.info(f"Backtest engine: {self.backtest_engine}")

    async def run_pipeline(self) -> Optional[Dict[str, Any]]:
        """
        Run the complete trading pipeline.

        Returns:
            Optional[Dict[str, Any]]: Performance report, None if error occurs
        """
        # Get symbols and timeframe from config or set default values
        symbols = self.config.get("trading", "instruments", default=["BTC/USDT"])
        if isinstance(symbols, str):
            symbols = [symbols.strip()]
        timeframe = self.config.get("trading", "timeframe", default="1m")
        self.logger.info(f"Starting trading pipeline | Symbols: {symbols} | Timeframe: {timeframe}")
        self._running = True
        
        try:
            self.trading_mode = await self.mode_factory.create(self.mode)            
            
            # Apply stored progress callback if available
            if hasattr(self, '_progress_callback') and self._progress_callback and hasattr(self.trading_mode, 'set_progress_callback'):
                self.trading_mode.set_progress_callback(self._progress_callback)
                self.logger.debug("Progress callback applied to initialized trading mode")
                        
            # Ensure trading mode is fully initialized
            if self.trading_mode is None:
                raise ValueError(f"Failed to create trading mode: {self.mode}")
            
            # If backtest mode, log the engine type and set it
            if self.mode == "backtest" and hasattr(self.trading_mode, 'set_engine_type'):
                self.logger.info(f"Using backtest engine: {self.backtest_engine}")
                self.trading_mode.set_engine_type(self.backtest_engine)
            
            # Progress callback setup (moved after mode initialization)
            def default_progress_callback(percent: float, message: str) -> None:
                """Default progress callback if no specific one is set"""
                self.logger.info(f"Progress: {percent}% - {message}")
            
            # Set a default progress callback if method exists
            if hasattr(self.trading_mode, 'set_progress_callback'):
                try:
                    self.trading_mode.set_progress_callback(default_progress_callback)
                    self.logger.debug("Default progress callback registered")
                except Exception as e:
                    self.logger.warning(f"Failed to set default progress callback: {str(e)}")
            
            # Run the trading mode
            results = await self.trading_mode.run(symbols, timeframe)
            return results
            
        except asyncio.CancelledError:
            self.logger.warning("Trading pipeline cancelled")
            return None
            
        except Exception as e:
            self.logger.error(f"Error in trading pipeline: {e}", exc_info=True)
            return {"error": str(e)}
            
        finally:
            await self.shutdown()
        
    async def shutdown(self) -> None:
        """Safely shut down all components."""
        if not self._running:
            return
        
        self.logger.info("Initiating system shutdown...")
        self._running = False
        
        # Shutdown trading mode if it exists
        if self.trading_mode is not None and hasattr(self.trading_mode, 'shutdown'):
            await self.trading_mode.shutdown()
        else:
            self.logger.warning("Trading mode not properly initialized, skipping shutdown")
        
        # Close async executor
        if hasattr(self.async_exec, 'close'):
            await self.async_exec.close()
        
        self.logger.info("System shutdown completed")

    def register_hooks(self, hooks: Dict[str, Callable]) -> None:
        """Register lifecycle hooks for the strategy."""
        if self.trading_mode and hasattr(self.trading_mode, 'strategy'):
            self.trading_mode.strategy.register_hooks(hooks)
            self.logger.debug(f"Registered strategy hooks: {list(hooks.keys())}")
        else:
            self.logger.warning("Cannot register hooks: trading mode or strategy not initialized")
            
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """
        Set a callback for progress updates during execution.
        
        Args:
            callback: Function taking (percentage, message) parameters
        """
        # Store the callback for later use
        self._progress_callback = callback
        self.logger.debug("Progress callback stored for later use")
        
        # If trading mode is already initialized, apply immediately
        if self.trading_mode and hasattr(self.trading_mode, 'set_progress_callback'):
            self.trading_mode.set_progress_callback(callback)
            self.logger.debug("Progress callback registered immediately")
            
    def stop(self) -> None:
        """Request to stop the trading process"""
        self.logger.info("Stop request received")
        self._running = False
        
        # Propagate stop request to trading mode if possible
        if self.trading_mode and hasattr(self.trading_mode, 'stop'):
            self.trading_mode.stop()
            
    def pause(self) -> None:
        """Pause the trading process if supported"""
        if self.trading_mode and hasattr(self.trading_mode, 'pause'):
            self.trading_mode.pause()
            self.logger.info("Trading process paused")
        else:
            self.logger.warning("Pause not supported by current trading mode")
            
    def resume(self) -> None:
        """Resume the trading process if supported"""
        if self.trading_mode and hasattr(self.trading_mode, 'resume'):
            self.trading_mode.resume()
            self.logger.info("Trading process resumed")
        else:
            self.logger.warning("Resume not supported by current trading mode")