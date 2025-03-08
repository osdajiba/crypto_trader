# src/strategy/base_strategy.py

import pandas as pd
from typing import Dict, Optional, Callable, Any, List
from abc import ABC, abstractmethod
import traceback

from src.common.config_manager import ConfigManager
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor
from src.common.data_processor import DataProcessor


class BaseStrategy(ABC):
    """
    增强的交易策略抽象基类，提供完整的生命周期管理和共享功能。
    
    所有特定策略必须继承此类并实现_generate_signals方法。
    """
    
    def __init__(self, config: ConfigManager, params: Optional[Dict[str, Any]] = None):
        """
        初始化基础策略。

        Args:
            config (ConfigManager): 配置管理器实例
            params (Optional[Dict[str, Any]]): 策略特定参数
        """
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(f"system.strategy.{self.__class__.__name__.lower()}")
        self.hooks: Dict[str, Callable] = {}
        self._state: Dict[str, Any] = {}
        self._is_initialized: bool = False
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.lookback_period = self.params.get("lookback_period", 60)
        self.period = self.params.get("period", "24H")
        self.executor = AsyncExecutor()  # 获取单例执行器实例
        self._running_tasks: List[str] = []  # 跟踪当前运行的任务
        
        # 尝试优雅地处理配置错误
        try:
            self._load_config_settings()
        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}，使用默认值")

    def _load_config_settings(self) -> None:
        """从配置中加载设置"""
        # 从配置加载通用策略设置
        strategy_config = self.config.get("strategy", default={})
        
        # 合并特定策略配置
        strategy_name = self.__class__.__name__.replace('Strategy', '').lower()
        specific_config = self.config.get("strategy", strategy_name, default={})
        
        # 应用通用设置
        if strategy_config:
            for key, value in strategy_config.items():
                if key not in self.params and key != "parameters":
                    self.params[key] = value
        
        # 应用特定设置（优先级更高）
        if specific_config:
            for key, value in specific_config.items():
                if key not in self.params:
                    self.params[key] = value

    async def process_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        处理特定交易对的数据并生成信号。
        
        Args:
            data (pd.DataFrame): 市场数据
            symbol (str): 处理的交易对
            
        Returns:
            pd.DataFrame: 交易信号
        """
        self.logger.info(f"处理{symbol}的数据")
        
        try:
            # 使用AsyncExecutor处理任务
            async with self.executor.task_context("preprocess") as task_id:
                self._running_tasks.append(task_id)
                clean_data = self._preprocess_data(data)
                self._running_tasks.remove(task_id)
                
            # 验证数据
            valid = await self.executor.submit(self._validate_data, clean_data)
            if not valid:
                self.logger.warning(f"{symbol}的数据无效")
                return pd.DataFrame()
            
            # 执行前置信号钩子
            async with self.executor.task_context("pre_hook") as task_id:
                self._running_tasks.append(task_id)
                await self._execute_hook("pre_signal_generation", data=clean_data, symbol=symbol)
                self._running_tasks.remove(task_id)
            
            # 生成信号
            async with self.executor.task_context("generate_signals") as task_id:
                self._running_tasks.append(task_id)
                signals = await self._generate_signals(clean_data)
                self._running_tasks.remove(task_id)
            
            # 添加交易对到信号表
            if not signals.empty and 'symbol' not in signals.columns:
                signals['symbol'] = symbol
            
            # 过滤信号
            filtered_signals = self._filter_signals(signals)
            
            # 执行后置信号钩子
            async with self.executor.task_context("post_hook") as task_id:
                self._running_tasks.append(task_id)
                await self._execute_hook("post_signal_generation", signals=filtered_signals, symbol=symbol)
                self._running_tasks.remove(task_id)
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"处理{symbol}数据时出错: {str(e)}\n{traceback.format_exc()}")
            return pd.DataFrame()  # 出错时返回空DataFrame
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预处理市场数据，为信号生成做准备。

        Args:
            data (pd.DataFrame): 原始市场数据

        Returns:
            pd.DataFrame: 处理后的数据
        """
        processed_data = data.copy()
        
        # 确保必要的列存在
        required_columns = {'open', 'high', 'low', 'close'}
        missing = required_columns - set(processed_data.columns)
        
        if missing:
            self.logger.warning(f"数据中缺少列 {missing}，填充为NaN")
            for col in missing:
                processed_data[col] = pd.NA
        
        # 确保索引为时间索引
        if not isinstance(processed_data.index, pd.DatetimeIndex):
            try:
                # 尝试从timestamp列或索引创建时间索引
                if 'timestamp' in processed_data.columns:
                    processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                    processed_data.set_index('timestamp', inplace=True)
                else:
                    processed_data.index = pd.to_datetime(processed_data.index)
            except Exception as e:
                self.logger.warning(f"转换为时间索引失败: {str(e)}")
        
        # 去除close列为空的行
        return processed_data.dropna(subset=['close'])
    
    @abstractmethod
    async def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        基于市场数据生成交易信号，子类必须实现此方法。

        Args:
            data (pd.DataFrame): 用于信号生成的市场数据（如OHLCV）

        Returns:
            pd.DataFrame: 包含'timestamp', 'symbol', 'action'(buy/sell)等列的信号
        """
        pass

    # --- 生命周期管理 ---
    async def initialize(self) -> None:
        """
        初始化策略资源（如加载模型、预计算数据）。
        
        子类可以重写此方法以实现特定的初始化逻辑。
        """
        if self._is_initialized:
            self.logger.warning(f"策略 {self.__class__.__name__} 已初始化")
            return
            
        # 启动执行器
        await self.executor.start()
        
        self.logger.debug(f"初始化策略: {self.__class__.__name__}")
        
        # 使用AsyncExecutor执行异步任务
        async with self.executor.task_context("load_resources") as task_id:
            self._running_tasks.append(task_id)
            await self._load_resources()
            self._running_tasks.remove(task_id)
            
        async with self.executor.task_context("preload_data") as task_id:
            self._running_tasks.append(task_id)
            await self._preload_historical_data()
            self._running_tasks.remove(task_id)
            
        self._is_initialized = True
        self.logger.info(f"策略 {self.__class__.__name__} 初始化完成")

    async def shutdown(self) -> None:
        """
        清理策略资源。
        
        子类可以重写此方法以实现特定的清理逻辑。
        """
        if not self._is_initialized:
            self.logger.warning(f"策略 {self.__class__.__name__} 未初始化，跳过关闭")
            return
            
        self.logger.debug(f"关闭策略: {self.__class__.__name__}")
        
        # 取消所有正在运行的任务
        for task_id in self._running_tasks.copy():
            await self.executor.cancel_task(task_id)
        self._running_tasks.clear()
        
        # 释放资源
        async with self.executor.task_context("release_resources") as task_id:
            await self._release_resources()
            
        # 清理状态和数据
        self._state.clear()
        self.historical_data.clear()
        self._is_initialized = False
        
        self.logger.info(f"策略 {self.__class__.__name__} 已关闭")

    async def _load_resources(self) -> None:
        """加载资源如模型或缓存数据的占位符方法"""
        pass

    async def _release_resources(self) -> None:
        """释放资源的占位符方法"""
        pass

    async def _preload_historical_data(self) -> None:
        """从multi_symbol_data或其他来源预加载历史数据"""
        # 获取多交易对数据
        multi_symbol_data = self.params.get("multi_symbol_data", {})
        if not multi_symbol_data:
            return
        
        # 处理每个交易对的数据
        for symbol, df in multi_symbol_data.items():
            try:
                # 清理和重采样数据
                cleaned_df = DataProcessor.clean_ohlcv(df)
                resampled_df = DataProcessor.resample(cleaned_df, self.period)

                # 存储足够长度的历史数据
                if len(resampled_df) >= self.lookback_period:
                    self.historical_data[symbol] = resampled_df.tail(self.lookback_period)
                else:
                    self.logger.warning(f"{symbol}的历史数据不足，需要{self.lookback_period}个周期")
            except Exception as e:
                self.logger.error(f"处理{symbol}的历史数据时出错: {str(e)}")
                
        self.logger.debug(f"为{len(self.historical_data)}个交易对预加载了历史数据")
    
    def _filter_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        基于规则过滤生成的信号（如最小信号强度）。

        Args:
            signals (pd.DataFrame): 原始信号

        Returns:
            pd.DataFrame: 过滤后的信号
        """
        if signals.empty:
            return signals
            
        # 基本过滤：移除action为空的行
        filtered = signals.dropna(subset=['action'])
        
        # 应用自定义过滤器（由子类设置）
        if hasattr(self, '_custom_filter') and callable(self._custom_filter):
            try:
                filtered = self._custom_filter(filtered)
            except Exception as e:
                self.logger.error(f"应用自定义过滤器时出错: {str(e)}")
        
        self.logger.debug(f"过滤从{len(signals)}个信号到{len(filtered)}个")
        return filtered

    async def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证输入数据。

        Args:
            data (pd.DataFrame): 要验证的数据

        Returns:
            bool: 有效为True，否则为False
        """
        # 检查基本条件
        if data.empty:
            self.logger.warning("数据为空")
            return False
            
        if 'close' not in data.columns:
            self.logger.warning("数据缺少'close'列")
            return False
            
        # 检查最小数据点数量
        min_data_points = max(self.lookback_period // 10, 5)  # 至少需要5个点或回溯期的10%
        if len(data) < min_data_points:
            self.logger.warning(f"数据点太少: {len(data)} < {min_data_points}")
            return False
            
        return True
    
    # --- 钩子管理 ---
    def register_hooks(self, hooks: Dict[str, Callable]) -> None:
        """
        注册生命周期钩子。

        Args:
            hooks (Dict[str, Callable]): 钩子名称到回调函数的字典
        """
        # 验证钩子是否合法
        for name, hook in hooks.items():
            if not callable(hook):
                self.logger.warning(f"钩子 '{name}' 不可调用，已跳过")
                continue
            self.hooks[name] = hook
            
        self.logger.debug(f"注册了钩子: {list(self.hooks.keys())}")

    async def _execute_hook(self, hook_name: str, *args, **kwargs) -> None:
        """
        执行已注册的钩子（如果存在）。

        Args:
            hook_name (str): 要执行的钩子名称
            *args: 钩子的位置参数
            **kwargs: 钩子的关键字参数
        """
        if hook_name not in self.hooks:
            return
            
        try:
            hook = self.hooks[hook_name]
            
            # 检查是否是协程函数并执行
            if callable(hook):
                # 异步函数直接await
                if hasattr(hook, '__await__'):
                    await hook(*args, **kwargs)
                # 使用AsyncExecutor包装同步函数
                else:
                    await self.executor.submit(lambda: hook(*args, **kwargs))
        except Exception as e:
            self.logger.error(f"执行钩子 {hook_name} 时出错: {str(e)}\n{traceback.format_exc()}")

    # --- 状态管理 ---
    def set_state(self, key: str, value: Any) -> None:
        """
        设置策略的状态变量。

        Args:
            key (str): 状态变量名称
            value (Any): 要存储的值
        """
        self._state[key] = value
        self.logger.debug(f"设置状态: {key} = {value}")

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        获取状态变量值。

        Args:
            key (str): 状态变量名称
            default (Any): 如果键不存在时的默认值

        Returns:
            Any: 存储的值或默认值
        """
        return self._state.get(key, default)

    def clear_state(self) -> None:
        """清除所有状态变量"""
        self._state.clear()
        self.logger.debug("清除了策略状态")

    # --- 工具方法 ---
    def calculate_position_size(self, price: float, capital: float) -> float:
        """
        基于价格和可用资金计算仓位大小。

        Args:
            price (float): 当前资产价格
            capital (float): 可用资金

        Returns:
            float: 计算的仓位大小（数量）
        """
        # 从参数或配置获取风险设置
        risk_per_trade = self.params.get("risk_per_trade", 0.01)  # 默认1%风险
        max_position = self.config.get("default_config", "user_config", "max_position", default=0.1)  # 默认最大10%
        
        # 计算仓位大小
        size = min((capital * risk_per_trade) / price, capital * max_position / price)
        
        # 确保大于最小订单大小
        min_order_size = self.config.get("default_config", "user_config", "min_order_size", default=0.001)
        return max(size, min_order_size)

    def calculate_performance_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        基于交易历史计算策略的性能指标。
        
        Args:
            trades (List[Dict]): 交易记录列表
        
        Returns:
            Dict[str, Any]: 性能指标字典
        """
        # 检查是否有交易
        if not trades:
            self.logger.warning("没有交易记录，无法计算性能指标")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # 转换为DataFrame以便分析
        try:
            df = pd.DataFrame(trades)
            
            # 计算关键指标
            metrics = {
                'total_trades': len(df),
                'win_rate': self._calculate_win_rate(df),
                'profit_factor': self._calculate_profit_factor(df),
                'max_drawdown': self._calculate_max_drawdown(df),
                'sharpe_ratio': self._calculate_sharpe_ratio(df),
                'total_return': df['profit'].sum() if 'profit' in df.columns else 0,
                'avg_trade_return': df['profit'].mean() if 'profit' in df.columns else 0
            }
            
            # 添加高级指标
            metrics.update(self._calculate_advanced_metrics(df))
            
            self.logger.info(f"策略性能指标: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"计算性能指标时出错: {str(e)}")
            return {'error': str(e), 'total_trades': len(trades)}

    def _calculate_win_rate(self, trades_df: pd.DataFrame) -> float:
        """计算盈利交易的百分比"""
        if 'profit' not in trades_df.columns or trades_df.empty:
            return 0
            
        profitable_trades = trades_df[trades_df['profit'] > 0]
        return len(profitable_trades) / len(trades_df) if len(trades_df) > 0 else 0

    def _calculate_profit_factor(self, trades_df: pd.DataFrame) -> float:
        """
        计算利润因子（总盈利/总亏损）。
        
        利润因子显示盈亏之间的关系，较高的值表示更好的策略表现。
        """
        if 'profit' not in trades_df.columns or trades_df.empty:
            return 0
            
        gross_profits = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_losses = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
        
        return gross_profits / gross_losses if gross_losses != 0 else float('inf')

    def _calculate_max_drawdown(self, trades_df: pd.DataFrame) -> float:
        """计算最大百分比回撤"""
        if 'profit' not in trades_df.columns or trades_df.empty:
            return 0
            
        # 计算累计盈亏曲线
        cumulative_profits = trades_df['profit'].cumsum()
        
        # 计算回撤
        peak = cumulative_profits.cummax()
        drawdown = (cumulative_profits - peak) / peak.replace(0, 1e-10)  # 避免除以零
        
        return abs(drawdown.min()) if not drawdown.empty else 0

    def _calculate_sharpe_ratio(self, trades_df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """
        计算策略表现的夏普比率。
        
        Args:
            trades_df (pd.DataFrame): 交易历史
            risk_free_rate (float): 年化无风险利率，默认2%
        
        Returns:
            float: 夏普比率
        """
        if 'profit_percentage' not in trades_df.columns or trades_df.empty:
            return 0
            
        # 假设日收益率，调整无风险利率
        daily_risk_free_rate = risk_free_rate / 252
        
        # 计算交易收益率
        trade_returns = trades_df['profit_percentage']
        
        # 计算平均超额收益率和标准差
        avg_excess_return = trade_returns.mean() - daily_risk_free_rate
        std_dev = trade_returns.std()
        
        return avg_excess_return / std_dev if std_dev != 0 else 0.0

    def _calculate_advanced_metrics(self, trades_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算高级性能指标
        
        Args:
            trades_df (pd.DataFrame): 交易历史DataFrame
            
        Returns:
            Dict[str, float]: 高级指标字典
        """
        advanced_metrics = {}
        
        try:
            if 'profit' in trades_df.columns and not trades_df.empty:
                # 计算最大连续盈利/亏损交易数
                profit_streak = self._calculate_streak(trades_df, 'profit', '>')
                loss_streak = self._calculate_streak(trades_df, 'profit', '<')
                
                # 计算盈亏比（平均盈利/平均亏损）
                avg_win = trades_df[trades_df['profit'] > 0]['profit'].mean() if any(trades_df['profit'] > 0) else 0
                avg_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].mean()) if any(trades_df['profit'] < 0) else 1
                win_loss_ratio = avg_win / avg_loss if avg_loss != 0 else float('inf')
                
                # 添加到结果
                advanced_metrics.update({
                    'max_profit_streak': profit_streak,
                    'max_loss_streak': loss_streak,
                    'win_loss_ratio': win_loss_ratio,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss
                })
        except Exception as e:
            self.logger.error(f"计算高级指标时出错: {str(e)}")
            
        return advanced_metrics

    def _calculate_streak(self, df: pd.DataFrame, column: str, operator: str) -> int:
        """
        计算最大连续事件序列
        
        Args:
            df (pd.DataFrame): 数据框
            column (str): 用于比较的列
            operator (str): 比较运算符（'>'或'<'）
            
        Returns:
            int: 最大连续序列长度
        """
        if operator == '>':
            streaks = (df[column] > 0).astype(int)
        else:
            streaks = (df[column] < 0).astype(int)
            
        # 计算连续序列
        max_streak = 0
        current_streak = 0
        
        for value in streaks:
            if value == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak

    def generate_backtest_report(self, trades: List[Dict], config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        生成全面的回测报告。
        
        Args:
            trades (List[Dict]): 完整的交易历史
            config (Optional[Dict]): 回测配置
        
        Returns:
            Dict[str, Any]: 全面的回测报告
        """
        # 性能指标
        performance = self.calculate_performance_metrics(trades)
        
        # 可选的交易可视化或统计分析
        visualizations = {}
        if config and config.get('generate_visualization', False):
            try:
                visualizations = self._generate_trade_visualizations(trades)
            except Exception as e:
                self.logger.error(f"生成可视化时出错: {str(e)}")
                visualizations = {'error': str(e)}
        
        # 策略参数
        strategy_params = {
            'class': self.__class__.__name__,
            'parameters': self.params
        }
        
        # 详细的交易日志和摘要
        trade_summary = self._generate_trade_summary(trades)
        
        # 详细报告
        detailed_report = {
            'strategy': strategy_params,
            'performance_metrics': performance,
            'trade_summary': trade_summary,
            'visualizations': visualizations,
            'trade_log': trades,
            'config': config or {}
        }
        
        self.logger.info(f"为{len(trades)}个交易生成了回测报告")
        return detailed_report

    def _generate_trade_visualizations(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        生成交易表现可视化的方法。
        
        仅作为占位符，子类可以实现具体功能。
        
        Args:
            trades (List[Dict]): 交易历史
            
        Returns:
            Dict[str, Any]: 包含可视化数据的字典
        """
        # 这只是个占位符，实现可能使用matplotlib,seaborn等
        return {'message': '可视化需要在子类中实现'}

    def _generate_trade_summary(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        生成交易摘要数据
        
        Args:
            trades (List[Dict]): 交易历史
            
        Returns:
            Dict[str, Any]: 交易摘要
        """
        if not trades:
            return {'message': '无交易记录'}
            
        try:
            df = pd.DataFrame(trades)
            
            # 基本摘要
            summary = {
                'total_trades': len(df),
                'first_trade_date': df['timestamp'].min() if 'timestamp' in df.columns else 'unknown',
                'last_trade_date': df['timestamp'].max() if 'timestamp' in df.columns else 'unknown',
            }
            
            # 按交易对分组统计
            if 'symbol' in df.columns:
                by_symbol = df.groupby('symbol').agg({
                    'profit': ['sum', 'mean', 'count']
                })
                
                # 转换为字典格式
                symbol_stats = {}
                for symbol, data in by_symbol.iterrows():
                    symbol_stats[symbol] = {
                        'total_profit': data[('profit', 'sum')],
                        'avg_profit': data[('profit', 'mean')],
                        'trade_count': data[('profit', 'count')]
                    }
                
                summary['by_symbol'] = symbol_stats
            
            # 按月/年分组统计
            if 'timestamp' in df.columns:
                df['month'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m')
                monthly = df.groupby('month').agg({
                    'profit': ['sum', 'mean', 'count']
                })
                
                # 转换为字典
                monthly_stats = {}
                for month, data in monthly.iterrows():
                    monthly_stats[month] = {
                        'total_profit': data[('profit', 'sum')],
                        'avg_profit': data[('profit', 'mean')],
                        'trade_count': data[('profit', 'count')]
                    }
                
                summary['by_month'] = monthly_stats
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成交易摘要时出错: {str(e)}")
            return {'error': str(e)}
