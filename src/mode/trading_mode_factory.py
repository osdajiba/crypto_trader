# src/mode/trading_mode_factory.py

import importlib
import inspect
from typing import Dict, Optional, Any, Type, List
from pathlib import Path

from src.common.abstract_factory import AbstractFactory, register_factory_class
from src.mode.base_trading_mode import BaseTradingMode
from src.common.enums import TradingMode


class TradingModeFactory(AbstractFactory):
    """优化的交易模式工厂"""
    
    def __init__(self, config):
        """
        初始化交易模式工厂
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        
        # 注册内置交易模式
        self._register_default_modes()
        
        # 自动发现其他交易模式
        self._discover_trading_modes()
    
    def _register_default_modes(self):
        """注册默认交易模式"""
        self.register(TradingMode.BACKTEST.value, "src.mode.backtest_trading_mode.BacktestTradingMode", {
            "description": "历史数据回测",
            "features": ["historical_data", "performance_analysis"]
        })
        self.register(TradingMode.PAPER.value, "src.mode.paper_trading_mode.PaperTradingMode", {
            "description": "模拟交易（使用真实市场数据但不使用真实资金）",
            "features": ["real_time_data", "virtual_execution"]
        })
        self.register(TradingMode.LIVE.value, "src.mode.live_trading_mode.LiveTradingMode", {
            "description": "实盘交易（在交易所使用真实资金）",
            "features": ["real_time_data", "real_execution", "risk_management"]
        })
    
    def _discover_trading_modes(self):
        """自动发现交易模式模块"""
        try:
            mode_dir = "src.mode"
            self.discover_registrable_classes(BaseTradingMode, mode_dir, "trading_mode_factory")
        except Exception as e:
            self.logger.error(f"自动发现交易模式时出错: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseTradingMode]:
        """
        获取交易模式类
        
        Args:
            name: 交易模式名称
            
        Returns:
            Type[BaseTradingMode]: 交易模式类
        """
        return await self._load_class_from_path(name, BaseTradingMode)
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        解析并验证交易模式名称
        
        Args:
            name: 交易模式名称
            
        Returns:
            str: 解析后的交易模式名称
        """
        # 验证模式类型
        try:
            name_value = name.lower() if name else TradingMode.BACKTEST.value
            mode = TradingMode(name_value)
        except ValueError:
            valid_modes = [m.value for m in TradingMode]
            raise ValueError(f"不支持的交易模式: {name}. 必须是以下之一: {valid_modes}")
            
        # 如果配置中启用了实时数据，强制使用实盘模式
        use_live_data = self.config.get("data", "use_live_data", default=False)
        if use_live_data:
            mode = TradingMode.LIVE
            self.logger.info(f"由于 use_live_data=True，覆盖模式为 {mode.value}")
            
        return mode.value
    
    def get_available_modes(self) -> Dict[str, str]:
        """
        获取可用的交易模式
        
        Returns:
            Dict[str, str]: 交易模式名称和描述
        """
        result = {}
        for name, info in self.get_registered_items().items():
            metadata = info.get('metadata', {})
            description = metadata.get('description', '')
            result[name] = description
        
        # 确保至少包含枚举中的基本模式
        for mode in TradingMode:
            if mode.value not in result:
                result[mode.value] = self._get_default_description(mode)
                
        return result
    
    def _get_default_description(self, mode: TradingMode) -> str:
        """
        获取交易模式的默认描述
        
        Args:
            mode: 交易模式
            
        Returns:
            str: 默认描述
        """
        descriptions = {
            TradingMode.BACKTEST: "历史数据回测",
            TradingMode.PAPER: "模拟交易（使用真实市场数据但不使用真实资金）",
            TradingMode.LIVE: "实盘交易（在交易所使用真实资金）"
        }
        return descriptions.get(mode, "未知交易模式")
    
    def get_mode_features(self, mode_name: str) -> List[str]:
        """
        获取交易模式的功能特性
        
        Args:
            mode_name: 交易模式名称
            
        Returns:
            List[str]: 功能特性列表
        """
        metadata = self._metadata.get(mode_name.lower(), {})
        return metadata.get('features', [])


# 在交易模式实现中使用装饰器的示例
@register_factory_class('trading_mode_factory', 'custom_mode', 
                       description="自定义交易模式",
                       features=["feature1", "feature2"])
class CustomTradingMode(BaseTradingMode):
    """
    自定义交易模式，展示自动注册的使用
    
    注意：这个类应该在单独的文件中定义，这里仅作示例
    """
    pass