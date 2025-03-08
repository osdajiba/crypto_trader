# risk_manager.py
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Any, Optional, Set
import asyncio

class RiskValidationError(Exception):
    """风险验证失败时引发的异常"""
    pass

class BaseRiskManager(ABC):
    """
    风险管理基类，定义了风险管理组件的通用接口
    所有风险管理器实现都应继承此类
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化风险管理器
        
        Args:
            config: 风险管理配置
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._config = config or {}
        self._risk_limits = self._config.get('risk_limits', {})
        self._active_controls: Set[str] = set()
        self._initialized = False
    
    async def initialize(self) -> None:
        """
        初始化风险管理器
        子类应在覆盖此方法时调用 super().initialize()
        """
        self._initialized = True
        self._logger.info(f"{self.__class__.__name__} initialized")
    
    @abstractmethod
    async def validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证交易信号的通用接口
        
        Args:
            signals: 要验证的交易信号列表
            
        Returns:
            经过风险验证的信号列表（可能被过滤或修改）
            
        Raises:
            RiskValidationError: 如果信号不符合风险规则
        """
        pass
    
    @abstractmethod
    async def execute_risk_control(self) -> None:
        """
        执行风险控制的通用接口
        应检查当前市场和仓位状态，并在必要时采取纠正行动
        """
        pass
    
    def set_risk_limit(self, limit_name: str, value: Any) -> None:
        """
        设置特定风险限制
        
        Args:
            limit_name: 限制名称
            value: 限制值
        """
        self._risk_limits[limit_name] = value
        self._logger.info(f"Risk limit '{limit_name}' set to {value}")
    
    def get_risk_limit(self, limit_name: str, default: Any = None) -> Any:
        """
        获取特定风险限制
        
        Args:
            limit_name: 限制名称
            default: 如果限制不存在则返回的默认值
            
        Returns:
            限制值或默认值
        """
        return self._risk_limits.get(limit_name, default)
    
    def activate_control(self, control_name: str) -> None:
        """
        激活特定风险控制
        
        Args:
            control_name: 要激活的控制名称
        """
        self._active_controls.add(control_name)
        self._logger.info(f"Risk control '{control_name}' activated")
    
    def deactivate_control(self, control_name: str) -> None:
        """
        停用特定风险控制
        
        Args:
            control_name: 要停用的控制名称
        """
        if control_name in self._active_controls:
            self._active_controls.remove(control_name)
            self._logger.info(f"Risk control '{control_name}' deactivated")
    
    def is_control_active(self, control_name: str) -> bool:
        """
        检查特定控制是否处于活动状态
        
        Args:
            control_name: 要检查的控制名称
            
        Returns:
            如果控制处于活动状态则为True，否则为False
        """
        return control_name in self._active_controls
    
    @abstractmethod
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        检查仓位是否在限制范围内
        
        Args:
            positions: 当前仓位
            
        Returns:
            如果仓位在限制范围内则为True，否则为False
        """
        pass

class BacktestRiskManager(BaseRiskManager):
    """
    回测专用风险管理实现
    简化了某些风险检查，但保持兼容的接口
    """
    
    async def validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证回测信号
        
        Args:
            signals: 要验证的信号列表
            
        Returns:
            验证后的信号列表
        """
        if not signals:
            return []
        
        valid_signals = []
        
        # 应用基本风险筛选逻辑
        for signal in signals:
            # 检查信号有效性
            if not self._is_valid_signal(signal):
                self._logger.warning(f"Invalid signal format: {signal}")
                continue
            
            # 应用仓位大小限制
            signal = self._apply_position_size_limits(signal)
            
            # 其他回测特定的风险检查...
            
            valid_signals.append(signal)
        
        self._logger.debug(f"Validated {len(valid_signals)} of {len(signals)} signals")
        return valid_signals
    
    def _is_valid_signal(self, signal: Dict[str, Any]) -> bool:
        """
        检查信号格式是否有效
        
        Args:
            signal: 要检查的信号
            
        Returns:
            如果信号格式有效则为True，否则为False
        """
        required_fields = {'symbol', 'direction', 'size'}
        return all(field in signal for field in required_fields)
    
    def _apply_position_size_limits(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        应用仓位大小限制
        
        Args:
            signal: 要修改的信号
            
        Returns:
            应用了限制的信号
        """
        max_size = self.get_risk_limit('max_position_size', float('inf'))
        if signal.get('size', 0) > max_size:
            self._logger.info(f"Limiting position size for {signal['symbol']} from {signal['size']} to {max_size}")
            signal_copy = signal.copy()
            signal_copy['size'] = max_size
            return signal_copy
        return signal
    
    async def execute_risk_control(self) -> None:
        """
        在回测中执行风险控制
        在回测模式下，这是一个简化的实现
        """
        # 回测模式下的风险控制通常是被动的
        # 大多数控制在validate_signals中完成
        pass
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        检查回测仓位限制
        
        Args:
            positions: 当前回测仓位
            
        Returns:
            如果仓位在限制范围内则为True，否则为False
        """
        # 检查总仓位价值
        total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        max_total = self.get_risk_limit('max_total_position', float('inf'))
        
        if total_value > max_total:
            self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
            return False
        
        # 检查单个资产集中度
        for symbol, pos in positions.items():
            concentration = abs(pos.get('value', 0)) / total_value if total_value > 0 else 0
            max_concentration = self.get_risk_limit('max_concentration', 1.0)
            
            if concentration > max_concentration:
                self._logger.warning(
                    f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                )
                return False
        
        return True

class LiveRiskManager(BaseRiskManager):
    """
    实盘专用风险管理实现
    包含更严格的风险控制和实时警报
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._position_history = []
        self._market_data_cache = {}
        self._alert_callbacks = []
    
    async def initialize(self) -> None:
        """初始化实盘风险管理器"""
        await super().initialize()
        
        # 设置默认控制
        default_controls = [
            'max_drawdown', 'max_exposure', 'volatility_control',
            'correlation_control', 'liquidity_control'
        ]
        
        for control in default_controls:
            if self.get_risk_limit(control, None) is not None:
                self.activate_control(control)
    
    async def validate_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证实盘交易信号
        
        Args:
            signals: 要验证的信号列表
            
        Returns:
            验证后的信号列表
            
        Raises:
            RiskValidationError: 如果信号出现严重风险问题
        """
        if not signals:
            return []
        
        valid_signals = []
        
        # 检查全局风险状态
        if self.is_control_active('emergency_stop') and self.get_risk_limit('emergency_stop', False):
            raise RiskValidationError("Emergency stop is active - all trading suspended")
        
        # 应用复杂的风险筛选逻辑
        for signal in signals:
            # 基本验证
            if not self._is_valid_signal(signal):
                self._logger.warning(f"Invalid signal format: {signal}")
                continue
            
            # 应用风险控制器
            try:
                # 检查每个活动的风险控制
                for control in self._active_controls:
                    control_method = getattr(self, f"_check_{control}", None)
                    if control_method and callable(control_method):
                        signal = await control_method(signal)
                        if signal is None:  # 信号被拒绝
                            break
                
                if signal is not None:  # 如果信号通过了所有控制
                    valid_signals.append(signal)
                    
            except Exception as e:
                self._logger.error(f"Error during risk validation: {str(e)}", exc_info=True)
                # 不添加此信号
        
        self._logger.info(f"Validated {len(valid_signals)} of {len(signals)} signals")
        return valid_signals
    
    def _is_valid_signal(self, signal: Dict[str, Any]) -> bool:
        """检查信号格式是否有效"""
        required_fields = {'symbol', 'direction', 'size', 'price'}
        return all(field in signal for field in required_fields)
    
    async def _check_max_exposure(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        检查信号是否超过最大风险敞口
        
        Args:
            signal: 要检查的信号
            
        Returns:
            可能修改过的信号，如果信号被拒绝则为None
        """
        max_exposure = self.get_risk_limit('max_exposure', float('inf'))
        
        # 计算当前总敞口
        current_exposure = self._calculate_total_exposure()
        
        # 计算新信号的增量敞口
        signal_exposure = abs(signal.get('size', 0) * signal.get('price', 0))
        
        if current_exposure + signal_exposure > max_exposure:
            self._logger.warning(
                f"Signal for {signal['symbol']} rejected: would exceed max exposure "
                f"({current_exposure + signal_exposure} > {max_exposure})"
            )
            return None
        
        return signal
    
    async def _check_max_drawdown(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        检查是否超过最大回撤限制
        
        Args:
            signal: 要检查的信号
            
        Returns:
            如果信号被接受则为信号，否则为None
        """
        max_drawdown = self.get_risk_limit('max_drawdown', -1.0)
        if max_drawdown <= 0:  # 未设置有效的回撤限制
            return signal
        
        # 计算当前回撤
        current_drawdown = self._calculate_current_drawdown()
        
        if current_drawdown >= max_drawdown:
            self._logger.warning(
                f"Signal for {signal['symbol']} rejected: max drawdown limit reached "
                f"({current_drawdown:.2%} >= {max_drawdown:.2%})"
            )
            return None
        
        return signal
    
    def _calculate_current_drawdown(self) -> float:
        """
        计算当前系统回撤
        
        Returns:
            当前回撤百分比（作为小数）
        """
        # 在实际实现中，这应根据账户的权益高水位线计算
        # 这是一个简化的实现
        return 0.05  # 示例：5%的回撤
    
    def _calculate_total_exposure(self) -> float:
        """
        计算当前总风险敞口
        
        Returns:
            总风险敞口值
        """
        # 在实际实现中，这应该从交易API或本地仓位跟踪获取
        # 这是一个简化的实现
        return sum(abs(pos.get('value', 0)) for pos in self._position_history)
    
    async def execute_risk_control(self) -> None:
        """
        执行实盘风险控制
        检查当前市场和仓位状态，并在必要时采取纠正行动
        """
        try:
            # 检查全局风险指标
            await self._check_global_risk_metrics()
            
            # 检查各个控制器
            for control in self._active_controls:
                control_method = getattr(self, f"_execute_{control}_control", None)
                if control_method and callable(control_method):
                    await control_method()
            
        except Exception as e:
            self._logger.error(f"Error during risk control execution: {str(e)}", exc_info=True)
    
    async def _check_global_risk_metrics(self) -> None:
        """检查全局风险指标并响应"""
        # 检查账户回撤
        current_drawdown = self._calculate_current_drawdown()
        max_drawdown = self.get_risk_limit('max_drawdown', 1.0)
        
        if current_drawdown >= max_drawdown:
            self._logger.warning(f"Maximum drawdown reached: {current_drawdown:.2%} >= {max_drawdown:.2%}")
            self.activate_control('emergency_stop')
            await self._send_alert(f"EMERGENCY: Maximum drawdown limit reached ({current_drawdown:.2%})")
        
        # 其他全局风险指标检查...
    
    async def _execute_volatility_control(self) -> None:
        """执行波动率控制逻辑"""
        if 'volatility_control' not in self._active_controls:
            return
        
        # 获取当前市场波动率
        market_volatility = await self._get_market_volatility()
        vol_threshold = self.get_risk_limit('volatility_threshold', 0.15)
        
        if market_volatility > vol_threshold:
            self._logger.warning(f"High market volatility detected: {market_volatility:.2%} > {vol_threshold:.2%}")
            
            # 根据波动率执行风险响应策略
            vol_response = self.get_risk_limit('volatility_response', 'reduce')
            
            if vol_response == 'halt':
                self.activate_control('emergency_stop')
                await self._send_alert("EMERGENCY: Trading halted due to high market volatility")
            elif vol_response == 'reduce':
                # 实施仓位减少策略
                self._logger.info("Implementing position reduction due to high volatility")
                # 实际实现会调用交易API来减少仓位
    
    async def _get_market_volatility(self) -> float:
        """
        获取当前市场波动率
        
        Returns:
            市场波动率值
        """
        # 在实际实现中，这应该从市场数据提供商获取
        # 这是一个简化的实现
        return 0.08  # 示例：8%的波动率
    
    async def _send_alert(self, message: str) -> None:
        """
        发送风险警报
        
        Args:
            message: 警报消息
        """
        self._logger.critical(f"RISK ALERT: {message}")
        
        # 执行所有已注册的警报回调
        for callback in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                self._logger.error(f"Error in alert callback: {str(e)}")
    
    def register_alert_callback(self, callback: callable) -> None:
        """
        注册警报回调
        
        Args:
            callback: 警报触发时要调用的函数
        """
        self._alert_callbacks.append(callback)
    
    async def check_position_limits(self, positions: Dict[str, Any]) -> bool:
        """
        检查实盘仓位限制
        
        Args:
            positions: 当前实盘仓位
            
        Returns:
            如果仓位在限制范围内则为True，否则为False
        """
        # 检查总仓位价值
        total_value = sum(abs(pos.get('value', 0)) for pos in positions.values())
        max_total = self.get_risk_limit('max_total_position', float('inf'))
        
        if total_value > max_total:
            self._logger.warning(f"Total position value {total_value} exceeds maximum {max_total}")
            await self._send_alert(f"Position limit exceeded: total value {total_value} > {max_total}")
            return False
        
        # 检查单个资产集中度
        for symbol, pos in positions.items():
            concentration = abs(pos.get('value', 0)) / total_value if total_value > 0 else 0
            max_concentration = self.get_risk_limit('max_concentration', 1.0)
            
            if concentration > max_concentration:
                self._logger.warning(
                    f"Position concentration for {symbol} ({concentration:.2%}) exceeds maximum ({max_concentration:.2%})"
                )
                await self._send_alert(
                    f"Concentration limit exceeded: {symbol} at {concentration:.2%} > {max_concentration:.2%}"
                )
                return False
        
        return True