# src/strategy/strategy_factory.py

from typing import Dict, Optional, Any, Type, List, Union
import importlib
import inspect
import traceback

from src.common.abstract_factory import AbstractFactory
from src.common.async_executor import AsyncExecutor
from src.strategy.base_strategy import BaseStrategy


class StrategyFactory(AbstractFactory):
    """增强的策略工厂，支持自动发现、动态加载和缓存策略实例"""
    
    def __init__(self, config):
        """
        初始化策略工厂
        
        Args:
            config: 配置对象
        """
        super().__init__(config)
        self.default_strategy_type = config.get("strategy", "active", default="dual_ma")
        self.default_params = config.get("strategy", "parameters", default={})
        self.executor = AsyncExecutor()  # 获取单例执行器
        
        # 更全面的策略注册表
        self._strategy_registry = {}  # 名称到类路径映射
        self._strategy_cache = {}  # 实例缓存
        self._strategy_metadata = {}  # 策略元数据
        
        # 注册内置策略
        self._register_default_strategies()
        
        # 自动发现策略
        self._discover_strategies()
    
    def _register_default_strategies(self):
        """注册默认策略"""
        # 注册双均线策略
        self.register("dual_ma", "src.strategy.DualMA.DualMAStrategy", {
            "description": "双均线交叉策略",
            "category": "trend",
            "parameters": ["short_window", "long_window"],
            "version": "1.0.0"
        })
        
        # 注册神经网络策略
        self.register("neural_network", "src.strategy.nn.NeuralNetStrategy", {
            "description": "神经网络策略",
            "category": "ml",
            "parameters": ["hidden_layers", "learning_rate"],
            "version": "1.0.0"
        })
        
        self.logger.info(f"注册了{len(self._strategy_registry)}个默认策略")
    
    def _discover_strategies(self):
        """自动发现并注册策略模块"""
        try:
            # 使用AsyncExecutor运行耗时操作
            strategy_dir = "src.strategy"
            
            # 1. 使用装饰器自动注册的策略
            self.discover_registrable_classes(BaseStrategy, strategy_dir, "strategy_factory")
            
            # 2. 搜索继承了BaseStrategy但没有使用装饰器的类
            self._discover_strategy_classes()
            
            self.logger.info(f"自动发现完成，共注册{len(self._strategy_registry)}个策略")
        except Exception as e:
            self.logger.error(f"自动发现策略时出错: {e}\n{traceback.format_exc()}")
    
    def _discover_strategy_classes(self):
        """扫描项目寻找未使用装饰器的策略类"""
        try:
            import pkgutil
            import importlib.util
            import os
            
            # 查找策略目录
            strategy_path = None
            for path in ["src/strategy", "strategy"]:
                if os.path.exists(path) and os.path.isdir(path):
                    strategy_path = path
                    break
                    
            if not strategy_path:
                self.logger.warning("找不到策略目录，跳过手动发现")
                return
                
            # 遍历策略目录中的所有Python文件
            for root, _, files in os.walk(strategy_path):
                for file in files:
                    if file.endswith('.py') and file not in ['__init__.py', 'base_strategy.py', 'strategy_factory.py', 'strategy_loader.py']:
                        try:
                            # 构建模块路径
                            rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(strategy_path))
                            module_path = rel_path.replace(os.sep, '.').replace('.py', '')
                            
                            if strategy_path == "src/strategy":
                                module_path = f"src.{module_path}"
                            
                            # 加载模块
                            spec = importlib.util.spec_from_file_location(module_path, os.path.join(root, file))
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                
                                # 检查模块中的类
                                for name, obj in inspect.getmembers(module, inspect.isclass):
                                    if (issubclass(obj, BaseStrategy) and 
                                        obj != BaseStrategy and 
                                        not hasattr(obj, '_factory_registry')):
                                        
                                        # 从类名中生成策略名称
                                        strategy_name = name.replace('Strategy', '').lower()
                                        
                                        # 检查是否已注册
                                        if strategy_name not in self._strategy_registry:
                                            self.register(strategy_name, f"{module_path}.{name}", {
                                                "description": obj.__doc__ or f"{name} 策略",
                                                "category": "其他",
                                                "parameters": [],
                                                "auto_discovered": True
                                            })
                                            self.logger.debug(f"自动发现并注册策略: {strategy_name} ({module_path}.{name})")
                        except Exception as e:
                            self.logger.warning(f"处理文件 {file} 时出错: {e}")
        except Exception as e:
            self.logger.error(f"扫描策略类时出错: {e}")
    
    async def _get_concrete_class(self, name: str) -> Type[BaseStrategy]:
        """
        获取策略类
        
        Args:
            name: 策略名称
            
        Returns:
            Type[BaseStrategy]: 策略类
        """
        try:
            # 1. 首先尝试从AbstractFactory方法加载
            return await self._load_class_from_path(name, BaseStrategy)
        except Exception as e:
            # 2. 如果失败，使用更健壮的方法
            self.logger.warning(f"使用标准方法加载策略 {name} 失败: {e}，尝试替代方法")
            
            if name not in self._strategy_registry:
                raise ValueError(f"未知策略: {name}")
                
            class_path = self._strategy_registry[name]
            
            # 如果已经是类引用，直接返回
            if inspect.isclass(class_path) and issubclass(class_path, BaseStrategy):
                return class_path
                
            # 加载类
            if isinstance(class_path, str):
                try:
                    module_path, class_name = class_path.rsplit('.', 1)
                    module = importlib.import_module(module_path)
                    loaded_class = getattr(module, class_name)
                    
                    if not inspect.isclass(loaded_class) or not issubclass(loaded_class, BaseStrategy):
                        raise TypeError(f"{class_name} 不是 BaseStrategy 的有效子类")
                        
                    # 更新注册表以备将来使用
                    self._strategy_registry[name] = loaded_class
                    return loaded_class
                except Exception as e:
                    self.logger.error(f"加载策略类 {class_path} 失败: {e}")
                    raise ValueError(f"加载策略类失败: {e}")
                    
            raise TypeError(f"无效的策略类型: {type(class_path)}")
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        解析策略名称，支持默认策略
        
        Args:
            name: 策略名称
            
        Returns:
            str: 解析后的策略名称
        """
        name = name or self.default_strategy_type
        if not name:
            raise ValueError("未提供策略名称且配置中无默认策略")
        return name.lower()
    
    def _get_default_params(self, name: str) -> Dict[str, Any]:
        """
        获取默认策略参数，整合全局和特定策略配置
        
        Args:
            name: 策略名称
            
        Returns:
            Dict[str, Any]: 默认参数
        """
        # 获取全局策略参数
        global_params = self.config.get("strategy", "parameters", default={})
        
        # 获取特定策略参数
        strategy_params = global_params.get(name, {})
        
        # 从策略元数据中获取默认参数
        metadata = self.get_strategy_metadata(name)
        default_params = metadata.get("default_params", {})
        
        # 合并参数（优先级：全局 < 元数据 < 特定策略）
        result = {}
        result.update(default_params)
        result.update(strategy_params)
        
        return result
    
    def register(self, name: str, creator: Union[str, Type[BaseStrategy]], metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        注册策略
        
        Args:
            name: 策略名称
            creator: 策略类或类路径
            metadata: 策略元数据
        """
        name = name.lower()
        self._strategy_registry[name] = creator
        if metadata:
            self._strategy_metadata[name] = metadata
        self.logger.debug(f"注册策略: {name}")
    
    def get_strategies_by_category(self, category: Optional[str] = None) -> List[str]:
        """
        按类别获取策略列表
        
        Args:
            category: 策略类别（如果为None则返回所有）
            
        Returns:
            List[str]: 策略名称列表
        """
        result = []
        for name in self._strategy_registry.keys():
            metadata = self.get_strategy_metadata(name)
            if category is None or metadata.get('category') == category:
                result.append(name)
        return sorted(result)
    
    def get_strategy_metadata(self, name: str) -> Dict[str, Any]:
        """
        获取策略元数据
        
        Args:
            name: 策略名称
            
        Returns:
            Dict[str, Any]: 策略元数据
        """
        return self._strategy_metadata.get(name, {})
    
    def get_all_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有策略及其元数据
        
        Returns:
            Dict[str, Dict[str, Any]]: 策略名称到元数据的映射
        """
        result = {}
        for name in self._strategy_registry.keys():
            result[name] = self.get_strategy_metadata(name)
        return result
    
    async def create_strategy(self, strategy_name: str, params: Optional[Dict[str, Any]] = None) -> BaseStrategy:
        """
        创建策略实例的便捷方法
        
        Args:
            strategy_name: 策略名称
            params: 策略参数
            
        Returns:
            BaseStrategy: 策略实例
        """
        # 确保执行器已启动
        await self.executor.start()
        
        # 使用AsyncExecutor创建策略
        try:
            strategy = await self.executor.submit(
                lambda: self.create_sync(strategy_name, params)
            )
            return strategy
        except Exception as e:
            self.logger.error(f"创建策略 {strategy_name} 失败: {e}")
            raise ValueError(f"创建策略失败: {e}")
    
    async def initialize_strategy(self, strategy: BaseStrategy) -> BaseStrategy:
        """
        初始化策略的便捷方法
        
        Args:
            strategy: 策略实例
            
        Returns:
            BaseStrategy: 初始化后的策略实例
        """
        # 使用AsyncExecutor初始化策略
        try:
            await self.executor.submit(strategy.initialize)
            return strategy
        except Exception as e:
            self.logger.error(f"初始化策略 {strategy.__class__.__name__} 失败: {e}")
            raise ValueError(f"初始化策略失败: {e}")
    
    async def shutdown_strategy(self, strategy: BaseStrategy) -> None:
        """
        关闭策略的便捷方法
        
        Args:
            strategy: 策略实例
        """
        # 使用AsyncExecutor关闭策略
        try:
            await self.executor.submit(strategy.shutdown)
        except Exception as e:
            self.logger.error(f"关闭策略 {strategy.__class__.__name__} 失败: {e}")
    
    async def load_strategies_from_config(self) -> Dict[str, BaseStrategy]:
        """
        从配置中加载所有已启用的策略
        
        Returns:
            Dict[str, BaseStrategy]: 策略名称到实例的映射
        """
        strategies = {}
        
        try:
            # 获取配置中已启用的策略
            enabled_strategies = self.config.get("strategy", "enabled", default=[])
            if isinstance(enabled_strategies, str):
                enabled_strategies = [enabled_strategies]
                
            if not enabled_strategies:
                self.logger.warning("配置中没有启用的策略")
                return strategies
                
            # 加载每个策略
            for strategy_name in enabled_strategies:
                try:
                    params = self._get_default_params(strategy_name)
                    strategy = await self.create_strategy(strategy_name, params)
                    await self.initialize_strategy(strategy)
                    strategies[strategy_name] = strategy
                    self.logger.info(f"已加载并初始化策略: {strategy_name}")
                except Exception as e:
                    self.logger.error(f"加载策略 {strategy_name} 失败: {e}")
        except Exception as e:
            self.logger.error(f"从配置加载策略时出错: {e}")
            
        return strategies
    
    def update_registry(self, strategy_registry: Dict[str, str]) -> None:
        """
        更新策略注册表，与StrategyLoader集成
        
        Args:
            strategy_registry (Dict[str, str]): 策略名称到路径的映射
        """
        self._strategy_registry.update(strategy_registry)
        self.logger.info(f"更新了策略注册表，现有{len(self._strategy_registry)}个策略")
