# src/strategy/strategy_loader.py

from typing import Dict, List, Set, Any, Type
import os
import importlib.util
import inspect
import traceback
from pathlib import Path
import time

from src.strategy.base_strategy import BaseStrategy
from src.strategy.strategy_factory import StrategyFactory
from src.common.log_manager import LogManager
from src.common.async_executor import AsyncExecutor


class StrategyLoader:
    """
    负责发现、加载和验证策略模块的增强工具
    
    提供延迟加载、缓存、自动重新加载和多路径发现功能。
    """
    
    def __init__(self, strategy_dirs: List[str] = None):
        """
        初始化策略加载器
        
        Args:
            strategy_dirs (List[str], optional): 包含策略模块的目录列表
        """
        self.logger = LogManager.get_logger("system.strategy_loader")
        
        # 如果未提供目录，使用默认列表
        self.strategy_dirs = []
        if strategy_dirs:
            self.strategy_dirs = [Path(d) for d in strategy_dirs]
        else:
            # 尝试多个可能的策略目录
            possible_dirs = ["src/strategy", "strategy", "src/strategies", "strategies"]
            for d in possible_dirs:
                p = Path(d)
                if p.exists() and p.is_dir():
                    self.strategy_dirs.append(p)
                    
        if not self.strategy_dirs:
            self.logger.warning("找不到任何策略目录")
            
        # 初始化注册表和元数据存储
        self.strategy_registry: Dict[str, str] = {}  # 名称 -> 模块路径
        self.strategy_metadata: Dict[str, Dict[str, Any]] = {}  # 名称 -> 元数据
        self.module_timestamps: Dict[str, float] = {}  # 文件路径 -> 最后修改时间
        self.processed_files: Set[str] = set()  # 已处理的文件集合
        
        # 获取AsyncExecutor单例
        self.executor = AsyncExecutor()
        
        # 自动发现策略
        self._discover_strategies()
        
    def _discover_strategies(self) -> None:
        """
        在策略目录中发现可用的策略模块
        """
        try:
            # 查找不是特定文件的Python文件
            excluded_files = {'BaseStrategy.py', '__init__.py', 'strategy_factory.py', 'strategy_loader.py'}
            strategy_files = []
            
            # 遍历策略目录及子目录
            for strategy_dir in self.strategy_dirs:
                self.logger.debug(f"在 {strategy_dir} 中搜索策略")
                
                # 验证目录
                if not strategy_dir.exists() or not strategy_dir.is_dir():
                    self.logger.warning(f"策略目录 {strategy_dir} 不存在")
                    continue
                    
                # 遍历目录和子目录
                for root, _, files in os.walk(strategy_dir):
                    for file in files:
                        if file.endswith('.py') and file not in excluded_files:
                            strategy_files.append(os.path.join(root, file))
            
            # 处理所有策略文件
            start_time = time.time()
            for file_path in strategy_files:
                self._load_strategies_from_file(file_path)
                
            # 报告发现结果
            elapsed = time.time() - start_time
            self.logger.info(f"发现了 {len(self.strategy_registry)} 个策略: {list(self.strategy_registry.keys())} (用时 {elapsed:.2f} 秒)")
        
        except Exception as e:
            self.logger.error(f"发现策略时出错: {str(e)}\n{traceback.format_exc()}")
    
    def _load_strategies_from_file(self, file_path: str) -> None:
        """
        从Python文件加载策略类
        
        Args:
            file_path (str): Python文件的路径
        """
        # 避免重复处理
        if file_path in self.processed_files:
            return
            
        # 跟踪文件修改时间
        file_timestamp = os.path.getmtime(file_path)
        
        # 检查是否需要重新加载
        if file_path in self.module_timestamps:
            if file_timestamp <= self.module_timestamps[file_path]:
                return  # 文件未更改，跳过
                
        # 更新时间戳和已处理标记
        self.module_timestamps[file_path] = file_timestamp
        self.processed_files.add(file_path)
        
        try:
            # 转换文件路径为模块路径
            for strategy_dir in self.strategy_dirs:
                if Path(file_path).is_relative_to(strategy_dir):
                    rel_path = os.path.relpath(file_path, os.path.dirname(strategy_dir.parent))
                    break
            else:
                rel_path = os.path.relpath(file_path)
                
            module_path = rel_path.replace(os.sep, '.').replace('.py', '')
            
            # 确保父路径是src.strategy或strategy
            if not (module_path.startswith('src.strategy') or module_path.startswith('strategy')):
                parts = module_path.split('.')
                if 'strategy' in parts:
                    idx = parts.index('strategy')
                    if idx > 0 and parts[idx-1] == 'src':
                        module_path = '.'.join(parts[idx-1:])
                    else:
                        module_path = '.'.join(parts[idx:])
            
            # 加载模块
            spec = importlib.util.spec_from_file_location(module_path, file_path)
            if spec is None or spec.loader is None:
                self.logger.warning(f"无法加载 {file_path} 的规范")
                return
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # 在模块中查找策略类
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseStrategy) and 
                    obj != BaseStrategy):
                    
                    # 注册策略，移除Strategy后缀
                    strategy_name = name.replace('Strategy', '').lower()
                    full_class_path = f"{module_path}.{name}"
                    self.strategy_registry[strategy_name] = full_class_path
                    
                    # 提取文档和元数据
                    doc = inspect.getdoc(obj) or f"{name} strategy"
                    
                    # 收集策略元数据
                    metadata = {
                        "name": strategy_name,
                        "class": name, 
                        "module": module_path,
                        "file": file_path,
                        "description": doc,
                        "last_updated": file_timestamp
                    }
                    
                    # 检查类是否有元数据属性
                    if hasattr(obj, '_factory_registry'):
                        factory_info = obj._factory_registry.get('strategy_factory', {})
                        if factory_info and 'metadata' in factory_info:
                            metadata.update(factory_info['metadata'])
                    
                    # 存储元数据
                    self.strategy_metadata[strategy_name] = metadata
                    
                    self.logger.debug(f"注册策略 {strategy_name}: {full_class_path}")
        
        except Exception as e:
            self.logger.error(f"从 {file_path} 加载策略时出错: {str(e)}\n{traceback.format_exc()}")
    
    def get_strategy_registry(self) -> Dict[str, str]:
        """
        获取策略注册表
        
        Returns:
            Dict[str, str]: 策略名称到模块路径的映射字典
        """
        return self.strategy_registry.copy()
    
    def get_strategy_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        获取策略元数据
        
        Returns:
            Dict[str, Dict[str, Any]]: 策略名称到元数据的映射字典
        """
        return self.strategy_metadata.copy()
    
    def validate_strategy(self, strategy_name: str) -> bool:
        """
        验证策略名称
        
        Args:
            strategy_name (str): 要验证的策略名称
            
        Returns:
            bool: 如果策略有效则为True，否则为False
        """
        return strategy_name.lower() in self.strategy_registry
    
    def update_factory(self, factory: StrategyFactory) -> None:
        """
        用发现的策略更新StrategyFactory
        
        Args:
            factory (StrategyFactory): 要更新的策略工厂
        """
        # 更新注册表
        factory.update_registry(self.strategy_registry)
        
        # 更新元数据
        for name, metadata in self.strategy_metadata.items():
            # 确保工厂使用的元数据格式正确
            factory_metadata = {
                "description": metadata.get("description", ""),
                "category": metadata.get("category", "其他"),
                "parameters": metadata.get("parameters", []),
                "source": metadata.get("file", "unknown")
            }
            
            # 添加额外元数据
            for key, value in metadata.items():
                if key not in ["description", "category", "parameters", "source"]:
                    factory_metadata[key] = value
                    
            # 注册带有元数据的策略
            factory.register(name, self.strategy_registry[name], factory_metadata)
            
        self.logger.info(f"使用 {len(self.strategy_registry)} 个策略更新了策略工厂")
    
    async def reload_strategies(self) -> None:
        """
        重新加载所有策略，检查更改
        """
        # 保存先前的注册表
        previous_strategies = set(self.strategy_registry.keys())
        
        # 清除处理文件集以强制重新加载
        self.processed_files.clear()
        
        # 重新发现策略
        self._discover_strategies()
        
        # 报告变化
        current_strategies = set(self.strategy_registry.keys())
        new_strategies = current_strategies - previous_strategies
        removed_strategies = previous_strategies - current_strategies
        
        if new_strategies:
            self.logger.info(f"发现了 {len(new_strategies)} 个新策略: {new_strategies}")
            
        if removed_strategies:
            self.logger.info(f"移除了 {len(removed_strategies)} 个策略: {removed_strategies}")
            
        return len(new_strategies) > 0 or len(removed_strategies) > 0
    
    async def load_strategy_class(self, strategy_name: str) -> Type[BaseStrategy]:
        """
        加载特定策略类
        
        Args:
            strategy_name (str): 策略名称
            
        Returns:
            Type[BaseStrategy]: 策略类
            
        Raises:
            ValueError: 如果策略未找到或加载失败
        """
        strategy_name = strategy_name.lower()
        
        if strategy_name not in self.strategy_registry:
            raise ValueError(f"未知策略: {strategy_name}")
            
        # 获取类路径
        class_path = self.strategy_registry[strategy_name]
        
        # 如果已经是类引用，直接返回
        if inspect.isclass(class_path) and issubclass(class_path, BaseStrategy):
            return class_path
            
        # 加载类
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            
            # 使用AsyncExecutor运行导入操作
            def import_module():
                try:
                    module = importlib.import_module(module_path)
                    return module
                except ImportError:
                    # 尝试修复常见的导入问题
                    if module_path.startswith('src.'):
                        # 如果src不在路径中，尝试不带src
                        try:
                            return importlib.import_module(module_path[4:])
                        except ImportError:
                            pass
                    elif not module_path.startswith('src.'):
                        # 如果没有src前缀，尝试添加
                        try:
                            return importlib.import_module(f"src.{module_path}")
                        except ImportError:
                            pass
                    raise
            
            # 导入模块
            module = await self.executor.submit(import_module)
            
            # 获取类
            strategy_class = getattr(module, class_name)
            
            # 验证类
            if not inspect.isclass(strategy_class) or not issubclass(strategy_class, BaseStrategy):
                raise TypeError(f"{class_name} 不是 BaseStrategy 的有效子类")
                
            # 更新注册表以使用类引用而不是字符串
            self.strategy_registry[strategy_name] = strategy_class
            
            return strategy_class
            
        except Exception as e:
            self.logger.error(f"加载策略类 {strategy_name} 失败: {str(e)}\n{traceback.format_exc()}")
            raise ValueError(f"加载策略类失败: {str(e)}")
    
    def get_available_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        获取可用策略的全面信息
        
        Returns:
            Dict[str, Dict[str, Any]]: 策略名称到策略信息的字典
        """
        result = {}
        
        for name in self.strategy_registry:
            if name in self.strategy_metadata:
                result[name] = self.strategy_metadata[name]
            else:
                result[name] = {
                    "name": name,
                    "class": self.strategy_registry[name].split('.')[-1] if isinstance(self.strategy_registry[name], str) else self.strategy_registry[name].__name__,
                    "description": "No description available"
                }
                
        return result