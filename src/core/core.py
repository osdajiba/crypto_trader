# src/core/core.py

import asyncio
from typing import Dict, List, Optional, Callable, Any

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.mode.trading_mode_factory import TradingModeFactory


class TradingCore:
    """Core coordinator for multi-asset concurrent trading system"""

    def __init__(self, config: ConfigManager, mode: str = None):
        """
        Initialize the trading core with configuration and mode.

        Args:
            config (ConfigManager): Initialized configuration manager instance.
            mode (str, optional): Trading mode ("backtest", "paper", "live"). 
                                  If None, uses the value from config.

        Raises:
            ValueError: If an unsupported mode is provided.
        """
        self.config: ConfigManager = config
        self.logger = LogManager.get_logger(name="trading_system")
        
        # 从配置中获取交易模式，如果提供了参数则使用参数
        self.mode = mode if mode is not None else config.get("default_config", "mode", default="backtest")
        
        self.async_exec = AsyncExecutor()
        self._running = False
        self.trading_mode = None  # 初始化为None，而不是字符串
        
        # Create trading mode factory
        self.mode_factory = TradingModeFactory(config=self.config)
        
        # 获取可用模式
        available_modes = self.mode_factory.get_available_modes()
        
        self.logger.info(
            f"Trading core initialized | Mode: {self.mode} | "
            f"Version: {self.config.get('system', 'version', default='unknown')}"
        )
        self.logger.info(f"Available trading modes: {list(available_modes.keys())}")

    async def run_pipeline(self) -> Optional[Dict[str, Any]]:
        """
        Run the complete trading pipeline.

        Returns:
            Optional[Dict[str, Any]]: Performance report, None if error occurs
        """
        # Get symbols and timeframe from config or set default values
        symbols = self.config.get("trading", "symbols", default=["BTC/USDT"])
        timeframe = self.config.get("trading", "timeframe", default="1h")
        
        self.logger.info(f"Starting trading pipeline | Symbols: {symbols} | Timeframe: {timeframe}")
        self._running = True
        
        try:
            # Create and initialize the appropriate trading mode
            self.trading_mode = await self.mode_factory.create(self.mode)
            
            # Run the selected trading mode
            if self.trading_mode is None:
                raise ValueError(f"Failed to create trading mode: {self.mode}")
                
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