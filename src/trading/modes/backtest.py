# src/mode/backtest_trading_mode.py

from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from src.trading.modes.base import BaseTradingMode
from src.risk.manager import BacktestRiskManager
from src.backtest.performance import PerformanceMonitor
from src.application.backtest_use_case import BacktestUseCase
from src.application.runtime_builder import RuntimeBuilder


class BacktestTradingMode(BaseTradingMode):
    """Backtesting trading mode implementation"""
    
    async def initialize(self) -> None:
        """Initialize backtest mode specific components"""
        self.logger.info("Initializing backtest mode")
        
        # Initialize risk manager
        if not self.risk_manager or not isinstance(self.risk_manager, BacktestRiskManager):
            self.risk_manager = BacktestRiskManager(config=self.config)
            await self.risk_manager.initialize()
        
        # Get strategy information from config
        strategy_name = self.config.get("strategy", "active", default='dual_ma')
        strategy_params = self.config.get("strategy", "parameters", default={})
        
        # Create strategy instance
        self.strategy = await self.strategy_factory.create(strategy_name, strategy_params)
        
        # Initialize performance monitoring
        if not self.performance_monitor:
            initial_capital = self.config.get("trading", "capital", "initial", default=100000)
            self.performance_monitor = PerformanceMonitor(
                config=self.config,
                initial_balance=initial_capital
            )
        
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
        self.start_date = self.config.get("backtest", "period", "start", default=None)
        self.end_date = self.config.get("backtest", "period", "end", default=None)
        
        # If not specified in backtest section, try trading section
        if not self.start_date:
            self.start_date = self.config.get("trading", "backtest_start", default=None)
        
        if not self.end_date:
            self.end_date = self.config.get("trading", "backtest_end", default=None)
        
        # If still None, use reasonable defaults (last 30 days)
        if not self.start_date:
            end = datetime.now()
            start = end.replace(day=1)  # Start of current month
            self.start_date = start.strftime("%Y-%m-%d")
            self.end_date = end.strftime("%Y-%m-%d")
            self.logger.info(f"Using default date range: {self.start_date} to {self.end_date}")

        # 回测主循环优先走 MarketDataFeed；这里仍加载 historical_data，是为了 RuntimeBuilder 构造回测执行模型。
        self.historical_data = await self._load_historical_data(symbols, timeframe)
        if not self.historical_data:
            raise ValueError("Failed to get historical data")
        
        # Get timestamps from historical data
        self.timestamps = self._get_combined_timestamps(self.historical_data)
        self.logger.info(f"Backtest contains {len(self.timestamps)} time periods")
        
        runtime = RuntimeBuilder(self).build_backtest_runtime(self.historical_data)
        if runtime is None:
            self.domain_pipeline = None

    def _create_market_data_feed(self):
        # 兼容旧测试入口；真实组装逻辑集中在 RuntimeBuilder。
        return RuntimeBuilder(self).create_historical_market_data_feed()

    def _create_domain_pipeline(self):
        """兼容旧入口：回测领域流水线由 RuntimeBuilder 统一组装。"""
        runtime = RuntimeBuilder(self).build_backtest_runtime(self.historical_data)
        return runtime.domain_pipeline if runtime is not None else None
    
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
                    data['symbol'] = symbol  # Add symbol column for later processing
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
            # Use 'datetime' column if available, otherwise try index
            if 'datetime' in df.columns:
                all_timestamps.extend(df['datetime'].tolist())
            elif isinstance(df.index, pd.DatetimeIndex):
                all_timestamps.extend(df.index.tolist())
        
        # Sort timestamps and remove duplicates
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
            # 这是 feed 迁移前的旧切片路径，目前保留给兼容场景和旧测试使用。
            # Get data at the specific timestamp
            if 'datetime' in df.columns:
                data_at_timestamp = df[df['datetime'] == timestamp]
                if not data_at_timestamp.empty:
                    result[symbol] = data_at_timestamp
            elif isinstance(df.index, pd.DatetimeIndex):
                try:
                    # Try to get exact timestamp match from index
                    data_at_timestamp = df.loc[[timestamp]]
                    if not data_at_timestamp.empty:
                        result[symbol] = data_at_timestamp
                except KeyError:
                    # No exact match - try the nearest timestamp if needed
                    pass
        
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
        # 回测编排已抽到 BacktestUseCase，mode 只负责准备依赖和委托执行。
        return await BacktestUseCase(self).run(symbols, timeframe)
    
    def _add_mode_specific_metrics(self, report: Dict[str, Any]) -> None:
        """
        Add backtest-specific metrics to report
        
        Args:
            report: Performance report to update
        """
        # Add backtest parameters
        report['backtest_params'] = {
            'symbols': self.config.get("trading", "instruments", default=[]),
            'timeframe': self.config.get("data", "default_timeframe", default="1m"),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'transaction_costs': {
                'commission_rate': self.config.get("trading", "execution", "commission", default=0.001),
                'slippage': self.config.get("trading", "execution", "slippage", default=0.001)
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
        
        if hasattr(self, 'data_manager') and self.data_manager:
            # DataManager 持有线程池和数据源，回测结束必须关闭，否则 CLI 会停在后台线程上。
            await self.data_manager.close()

        if hasattr(self, 'performance_monitor') and self.performance_monitor:
            await self.performance_monitor.close()
        
        self._running = False
        self.logger.info("Backtest mode shutdown complete")
