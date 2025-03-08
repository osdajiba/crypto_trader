# src/mode/backtest_trading_mode.py

from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from src.mode.base_trading_mode import BaseTradingMode
from src.execution.execution_engine import ExecutionEngine


class BacktestTradingMode(BaseTradingMode):
    """Backtesting trading mode implementation"""
    
    async def initialize(self) -> None:
        """Initialize backtest mode specific components"""
        self.logger.info("Initializing backtest mode")
        
        # Create backtest-specific execution engine
        self.execution_engine = ExecutionEngine(
            config=self.config,
            mode="backtest"
        )
        
        # Get strategy information from config
        strategy_name = self.config.get("backtest", "strategy", default=None)
        strategy_params = self.config.get("backtest", "strategy_parameters", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        
        # Mark as initialization complete
        self._running = True
        self.logger.info("Backtest mode initialization complete")
    
    async def _prepare_run(self, symbols: List[str], timeframe: str) -> None:
        """
        Backtest-specific preparation
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
        """
        # Initialize backtest parameters from config
        self.start_date = self.config.get("backtest", "start_date")
        self.end_date = self.config.get("backtest", "end_date")
        
        # Load historical data
        self.historical_data = await self._load_historical_data(symbols, timeframe)
        if not self.historical_data:
            raise ValueError("Failed to get historical data")
        
        # Get timestamps from historical data
        self.timestamps = self._get_combined_timestamps(self.historical_data)
        self.logger.info(f"Backtest contains {len(self.timestamps)} time periods")
    
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        Load historical data
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
            
        Returns:
            Dict: Mapping of symbols to data
        """
        data_map = {}
        
        for symbol in symbols:
            try:
                data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start=self.start_date,
                    end=self.end_date
                )
                
                if not data.empty:
                    data_map[symbol] = data
                    self.logger.info(f"Loaded {symbol} historical data: {len(data)} records")
                else:
                    self.logger.warning(f"{symbol} has no historical data")
                
            except Exception as e:
                self.logger.error(f"Failed to load {symbol} historical data: {e}")
        
        return data_map
    
    def _get_combined_timestamps(self, data_map: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        Get all unique timestamps and sort them
        
        Args:
            data_map: Mapping of symbols to data
            
        Returns:
            List: Sorted unique timestamps
        """
        all_timestamps = []
        
        for df in data_map.values():
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
        
        return sorted(set(all_timestamps))
    
    def _get_data_at_timestamp(self, timestamp) -> Dict[str, pd.DataFrame]:
        """
        Get data for specified timestamp
        
        Args:
            timestamp: Timestamp
            
        Returns:
            Dict: Mapping of symbols to single-timestamp data
        """
        result = {}
        
        for symbol, df in self.historical_data.items():
            if 'datetime' in df.columns:
                data_at_timestamp = df[df['datetime'] == timestamp]
                if not data_at_timestamp.empty:
                    result[symbol] = data_at_timestamp
        
        return result
    
    async def _execute_trading_loop(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        Execute backtest trading loop
        
        Args:
            symbols: List of trading symbols
            timeframe: Time period
            
        Returns:
            Dict: Backtest results
        """
        try:
            # Main backtest loop
            for i, timestamp in enumerate(self.timestamps):
                if i % 100 == 0:  # Log progress periodically
                    self.logger.info(f"Backtest progress: {i}/{len(self.timestamps)}")
                
                # Get data for current timestamp
                current_data = self._get_data_at_timestamp(timestamp)
                
                # Process this data point
                await self._process_market_data(current_data)
                
                # Check if should continue
                if not self._should_continue():
                    self.logger.info("Stopping backtest due to risk breach or user request")
                    break
            
            # Return default report structure
            return {}
            
        except Exception as e:
            self.logger.error(f"Backtest execution error: {e}", exc_info=True)
            raise
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """
        Add backtest-specific metrics to report
        
        Args:
            report: Performance report to update
        """
        # Add backtest parameters
        report['backtest_params'] = {
            'symbols': self.config.get("backtest", "symbols", default=[]),
            'timeframe': self.config.get("backtest", "timeframe", default=""),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'transaction_costs': {
                'commission_rate': self.config.get("backtest", "transaction_costs", "commission_rate", default=0.001),
                'slippage': self.config.get("backtest", "transaction_costs", "slippage", default=0.001)
            }
        }
        
        # Add strategy info
        if self.strategy:
            report['strategy'] = self.strategy.__class__.__name__
    
    async def shutdown(self) -> None:
        """Shutdown backtest mode"""
        self.logger.info("Shutting down backtest mode")
        
        if hasattr(self, 'strategy') and self.strategy:
            await self.strategy.shutdown()
        
        if hasattr(self, 'execution_engine') and self.execution_engine:
            await self.execution_engine.close()
        
        self._running = False
        self.logger.info("Backtest mode shutdown complete")