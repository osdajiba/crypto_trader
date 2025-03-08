# src/strategy/DualMA.py

import pandas as pd
from typing import Dict, Optional, Any, List
from src.strategy.base_strategy import BaseStrategy
from src.common.config_manager import ConfigManager
from src.common.abstract_factory import register_factory_class
from src.common.async_executor import AsyncExecutor
from src.common.data_processor import DataProcessor


@register_factory_class('strategy_factory', 'dual_ma', 
                       description="双均线交叉策略",
                       category="trend",
                       parameters=["short_window", "long_window"])
class DualMAStrategy(BaseStrategy):
    """
    双均线交叉策略实现，使用短期和长期移动平均线的交叉生成交易信号。
    
    通过AsyncExecutor管理异步任务，提供更高效的资源使用和更好的错误处理。
    """

    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        初始化双均线策略
        
        Args:
            config (ConfigManager): 配置管理器实例
            params (Optional[Dict[str, Any]]): 策略特定参数
                期望参数: short_window, long_window, symbol, multi_symbol_data, lookback_period, period
        """
        super().__init__(config, params)
        self.short_window = self.params.get("short_window", 10)
        self.long_window = self.params.get("long_window", 30)
        self.primary_symbol = self.params.get("symbol", "unknown")
        self.multi_symbol_data = self.params.get("multi_symbol_data", {})
        self.executor = AsyncExecutor()  # 获取单例实例
        self._running_tasks: List[str] = []

    async def initialize(self) -> None:
        """初始化资源并验证参数，使用AsyncExecutor管理任务生命周期"""
        # 启动执行器
        await self.executor.start()
        
        # 使用父类初始化
        await self.executor.submit(super().initialize)
        
        # 验证参数
        if self.short_window >= self.long_window:
            self.logger.error("short_window (%d) must be less than long_window (%d)", 
                              self.short_window, self.long_window)
            raise ValueError("short_window must be less than long_window")
        
        if self.lookback_period < self.long_window:
            self.logger.warning("lookback_period (%d) is less than long_window (%d), adjusting to %d", 
                                self.lookback_period, self.long_window, self.long_window)
            self.lookback_period = self.long_window
            
        self.logger.info("Initialized DualMAStrategy with short_window=%d, long_window=%d", 
                         self.short_window, self.long_window)

    async def shutdown(self) -> None:
        """关闭策略并清理资源"""
        # 取消所有运行中的任务
        for task_id in self._running_tasks.copy():
            await self.executor.cancel_task(task_id)
        self._running_tasks.clear()
        
        # 调用父类的shutdown方法
        await super().shutdown()
        
        self.logger.info("DualMAStrategy shutdown complete")

    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成基于市场数据的交易信号，实现BaseStrategy的抽象方法。
        
        Args:
            data (pd.DataFrame): 用于信号生成的市场数据（例如，OHLCV）

        Returns:
            pd.DataFrame: 包含'timestamp', 'symbol', 'action'（买/卖）等列的信号
        """
        self.logger.info("Generating signals for DualMAStrategy")
        
        # 执行前置钩子
        async with self.executor.task_context("pre_signal_hook") as task_id:
            self._running_tasks.append(task_id)
            await self._execute_hook("pre_signal_generation", data=data)
            self._running_tasks.remove(task_id)

        # 清理和验证数据
        if DataProcessor is not None:
            real_time_data = DataProcessor.clean_ohlcv(data)
            real_time_data = DataProcessor.resample(real_time_data, self.period)
        else:
            real_time_data = self._preprocess_data(data)
            
        valid_data = await self.executor.submit(self._validate_data, real_time_data)
        if not valid_data:
            return pd.DataFrame()

        # 准备分析数据
        if DataProcessor is not None and hasattr(DataProcessor, "splice_data"):
            all_data = DataProcessor.splice_data(
                self.historical_data, real_time_data, self.primary_symbol, 
                self.lookback_period, self.multi_symbol_data
            )
        else:
            # 如果DataUtils不可用，使用内部方法
            all_data = self._combine_historical_and_new_data(real_time_data)
            
        if all_data.empty:
            self.logger.warning("No valid data after preparation")
            return pd.DataFrame()

        # 计算信号
        async with self.executor.task_context("signal_calculation") as task_id:
            self._running_tasks.append(task_id)
            signals = await self._calculate_signals(all_data)
            signals = self._filter_signals(signals)
            self._running_tasks.remove(task_id)

        # 更新历史数据缓存
        if DataProcessor is not None and hasattr(DataProcessor, "update_historical_data"):
            DataProcessor.update_historical_data(self.historical_data, real_time_data, 
                                          self.primary_symbol, self.lookback_period)
        else:
            self._update_historical_data(real_time_data)
            
        self.logger.debug("Updated historical data for %s with %d new rows", 
                        self.primary_symbol, len(real_time_data))

        # 执行后置钩子
        async with self.executor.task_context("post_signal_hook") as task_id:
            self._running_tasks.append(task_id)
            await self._execute_hook("post_signal_generation", signals=signals)
            self._running_tasks.remove(task_id)
            
        return signals

    def _combine_historical_and_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        内部方法，在DataUtils不可用时合并历史和新数据
        
        Args:
            new_data (pd.DataFrame): 新的市场数据
            
        Returns:
            pd.DataFrame: 合并的历史和新数据
        """
        if self.primary_symbol in self.historical_data:
            combined = pd.concat([self.historical_data[self.primary_symbol], new_data])
            combined['_symbol'] = self.primary_symbol
            return combined
        else:
            new_data['_symbol'] = self.primary_symbol
            return new_data

    def _update_historical_data(self, new_data: pd.DataFrame) -> None:
        """
        内部方法，在DataUtils不可用时更新历史数据
        
        Args:
            new_data (pd.DataFrame): 要添加到历史缓存的新市场数据
        """
        if self.primary_symbol not in self.historical_data:
            self.historical_data[self.primary_symbol] = new_data
        else:
            self.historical_data[self.primary_symbol] = pd.concat(
                [self.historical_data[self.primary_symbol], new_data]
            ).tail(self.lookback_period)

    async def _calculate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算基于双均线交叉的信号
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的数据框
            
        Returns:
            pd.DataFrame: 生成的交易信号
        """
        # 过滤主要符号
        if '_symbol' in df.columns:
            df = df[df['_symbol'] == self.primary_symbol].copy()
        else:
            df = df.copy()
            
        if df.empty:
            self.logger.warning("No data for primary symbol %s", self.primary_symbol)
            return pd.DataFrame()

        # 计算移动平均线
        df['short_ma'] = df['close'].rolling(window=self.short_window, min_periods=1).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_window, min_periods=1).mean()

        # 生成信号
        signals = pd.DataFrame(index=df.index)
        signals['timestamp'] = df['time'] if 'time' in df.columns else df.index
        signals['symbol'] = self.primary_symbol
        signals['action'] = None

        # 买入信号：短期均线向上穿过长期均线
        signals.loc[(df['short_ma'] > df['long_ma']) & 
                    (df['short_ma'].shift(1) <= df['long_ma'].shift(1)), 'action'] = 'buy'
        
        # 卖出信号：短期均线向下穿过长期均线
        signals.loc[(df['short_ma'] < df['long_ma']) & 
                    (df['short_ma'].shift(1) >= df['long_ma'].shift(1)), 'action'] = 'sell'

        signals = signals.dropna(subset=['action'])
        self.logger.debug("Generated %d signals for %s", len(signals), self.primary_symbol)
        return signals

    async def _calculate_factor(self, df: pd.DataFrame, factor: str) -> pd.Series:
        """
        计算因子值（在此策略中未使用）
        
        Args:
            df (pd.DataFrame): 包含OHLCV数据的数据框
            factor (str): 因子名称
            
        Returns:
            pd.Series: 计算的因子值
        """
        self.logger.warning("Factor calculation not implemented in DualMAStrategy")
        return pd.Series(0, index=df.index)



