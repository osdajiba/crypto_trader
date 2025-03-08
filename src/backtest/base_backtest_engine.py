# src/backtest/base_backtest_engine.py

import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os
from pathlib import Path

from common.log_manager import LogManager


class BacktestEngine(ABC):
    """统一回测引擎基类"""
    
    def __init__(self, config, strategy, data_manager):
        """
        初始化回测引擎基类
        
        Args:
            config: 配置管理器
            strategy: 交易策略实例
            data_manager: 数据管理器实例
        """
        self.config = config
        self.strategy = strategy
        self.data_manager = data_manager
        self.logger = LogManager.get_logger("system.backtest")
        
        # 回测状态
        self.state = {}
        
    async def run(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """
        执行回测的通用流程
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
            
        Returns:
            Dict[str, Any]: 回测结果
        """
        self.logger.info(f"开始回测 | 交易品种: {symbols} | 时间周期: {timeframe}")
        
        try:
            # 1. 初始化回测参数
            await self._initialize_backtest(symbols, timeframe)
            
            # 2. 加载历史数据
            data = await self._load_historical_data(symbols, timeframe)
            if not data:
                raise ValueError("无法获取历史数据")
                
            # 3. 获取时间点序列
            time_points = self._get_time_points(data)
            self.logger.info(f"回测包含 {len(time_points)} 个时间周期")
            
            # 4. 执行回测循环
            for i, time_point in enumerate(time_points):
                if i % 100 == 0:  # 定期记录进度
                    self.logger.info(f"回测进度: {i}/{len(time_points)}")
                
                # 获取当前时间点的数据
                current_data = self._get_data_at_time_point(data, time_point)
                
                # 处理当前时间点
                await self._process_data_point(time_point, current_data)
                
            # 5. 生成回测报告
            results = self._generate_backtest_report()
            
            # 6. 保存回测报告
            self._save_backtest_report(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"回测执行错误: {e}", exc_info=True)
            return {"error": str(e)}
            
        finally:
            # 7. 清理资源
            await self._cleanup_resources()
    
    @abstractmethod
    async def _initialize_backtest(self, symbols: List[str], timeframe: str) -> None:
        """
        初始化回测参数，子类必须实现
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
        """
        pass
    
    @abstractmethod
    async def _load_historical_data(self, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
        """
        加载历史数据，子类必须实现
        
        Args:
            symbols: 交易品种列表
            timeframe: 时间周期
            
        Returns:
            Dict[str, pd.DataFrame]: 历史数据
        """
        pass
    
    @abstractmethod
    def _get_time_points(self, data: Dict[str, pd.DataFrame]) -> List[Any]:
        """
        获取时间点序列，子类必须实现
        
        Args:
            data: 历史数据
            
        Returns:
            List[Any]: 时间点序列
        """
        pass
    
    @abstractmethod
    def _get_data_at_time_point(self, data: Dict[str, pd.DataFrame], time_point: Any) -> Dict[str, pd.DataFrame]:
        """
        获取指定时间点的数据，子类必须实现
        
        Args:
            data: 历史数据
            time_point: 时间点
            
        Returns:
            Dict[str, pd.DataFrame]: 时间点数据
        """
        pass
    
    @abstractmethod
    async def _process_data_point(self, time_point: Any, data: Dict[str, pd.DataFrame]) -> None:
        """
        处理单个数据点，子类必须实现
        
        Args:
            time_point: 时间点
            data: 时间点数据
        """
        pass
    
    @abstractmethod
    def _generate_backtest_report(self) -> Dict[str, Any]:
        """
        生成回测报告，子类必须实现
        
        Returns:
            Dict[str, Any]: 回测报告
        """
        pass
    
    def _save_backtest_report(self, results: Dict[str, Any]) -> None:
        """
        保存回测报告的通用实现
        
        Args:
            results: 回测报告
        """
        # 创建报告目录
        report_dir = self.config.get("reporting", "backtest_reports_dir", default="reports/backtest/")
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳作为文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存交易记录
        output_formats = self.config.get("reporting", "output_formats", default=["json"])
        
        # 保存为CSV
        if 'csv' in output_formats and 'trades' in results:
            trades_df = pd.DataFrame(results['trades'])
            trades_csv = os.path.join(report_dir, f"trades_{timestamp}.csv")
            trades_df.to_csv(trades_csv, index=False)
            self.logger.info(f"交易记录已保存到 {trades_csv}")
        
        # 保存为JSON
        if 'json' in output_formats:
            # 清理不可序列化的对象
            clean_results = self._prepare_for_serialization(results)
            
            summary_json = os.path.join(report_dir, f"report_{timestamp}.json")
            with open(summary_json, 'w') as f:
                json.dump(clean_results, f, indent=4, default=str)
            
            self.logger.info(f"回测报告已保存到 {summary_json}")
        
        # 输出主要性能指标
        self._log_performance_metrics(results)
    
    def _prepare_for_serialization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备数据用于JSON序列化
        
        Args:
            data: 原始数据
            
        Returns:
            Dict[str, Any]: 可序列化的数据
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items() 
                   if k not in ['strategy', 'equity_curve'] or k == 'strategy' and hasattr(v, '__name__')}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return data
    
    def _log_performance_metrics(self, results: Dict[str, Any]) -> None:
        """
        日志输出性能指标
        
        Args:
            results: 回测结果
        """
        self.logger.info("==== 回测性能摘要 ====")
        self.logger.info(f"初始资金: ${results.get('initial_capital', 0):,.2f}")
        self.logger.info(f"最终权益: ${results.get('final_equity', 0):,.2f}")
        self.logger.info(f"总收益: ${results.get('total_return', 0):,.2f} ({results.get('total_return_pct', 0):.2f}%)")
        self.logger.info(f"最大回撤: {results.get('max_drawdown_pct', 0):.2f}%")
        self.logger.info(f"夏普比率: {results.get('sharpe_ratio', 0):.2f}")
        self.logger.info(f"交易次数: {results.get('total_trades', 0)}")
        self.logger.info("=======================")
    
    async def _cleanup_resources(self) -> None:
        """
        清理资源
        """
        if hasattr(self.strategy, 'shutdown'):
            await self.strategy.shutdown()