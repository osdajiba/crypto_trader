# src/common/abstract_factory.py

import copy
import json
import hashlib
import inspect
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, ClassVar, Set, List
import asyncio

from src.common.log_manager import LogManager

# 泛型类型变量
T = TypeVar('T')


class AbstractFactory(ABC):
    """增强的抽象工厂基类"""
    
    # 类级缓存以支持单例模式
    _instances: ClassVar[Dict[str, 'AbstractFactory']] = {}
    
    def __init__(self, config):
        """
        初始化抽象工厂
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.logger = LogManager.get_logger(f"system.{self.__class__.__name__.lower()}")
        self._registry: Dict[str, Any] = {}
        self._cache: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def get_instance(cls, config) -> 'AbstractFactory':
        """
        获取工厂单例实例
        
        Args:
            config: 配置对象
            
        Returns:
            AbstractFactory: 工厂实例
        """
        key = cls.__name__
        if key not in cls._instances:
            cls._instances[key] = cls(config)
        return cls._instances[key]
    
    async def create(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> T:
        """
        创建对象（模板方法）
        
        Args:
            name: 对象名称
            params: 创建参数
            
        Returns:
            T: 创建的对象
        """
        try:
            # 1. 获取要创建的类型名称
            name = await self._resolve_name(name)
            
            # 2. 获取具体类
            concrete_class = await self._get_concrete_class(name)
            
            # 3. 创建或获取缓存的实例
            return await self._get_or_create(name, concrete_class, params)
            
        except Exception as e:
            self.logger.error(f"创建失败: {e}")
            raise
    
    def create_sync(self, name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> T:
        """
        同步创建接口（方便调用）
        
        Args:
            name: 对象名称
            params: 创建参数
            
        Returns:
            T: 创建的对象
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # 如果没有运行中的事件循环，创建一个新的
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.create(name, params))
    
    @abstractmethod
    async def _get_concrete_class(self, name: str) -> Type[T]:
        """
        获取具体类型，由子类实现
        
        Args:
            name: 对象名称
            
        Returns:
            Type[T]: 对象类型
        """
        pass
    
    async def _resolve_name(self, name: Optional[str]) -> str:
        """
        解析对象名称，可被子类重写
        
        Args:
            name: 对象名称
            
        Returns:
            str: 解析后的名称
        """
        if name is None:
            raise ValueError("必须提供名称")
        return name.lower()
    
    def register(self, name: str, creator: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        注册创建器
        
        Args:
            name: 对象名称
            creator: 创建函数或类
            metadata: 关联的元数据
        """
        name = name.lower()
        self._registry[name] = creator
        if metadata:
            self._metadata[name] = metadata
        self.logger.debug(f"注册了 {name} 到 {self.__class__.__name__}")
    
    def get_registered_items(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有已注册项
        
        Returns:
            Dict[str, Dict[str, Any]]: 已注册项字典，包含创建器和元数据
        """
        result = {}
        for name, creator in self._registry.items():
            item_info = {
                'creator': str(creator),
                'metadata': self._metadata.get(name, {})
            }
            result[name] = item_info
        return result
    
    def _create_cache_key(self, name: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        创建更可靠的缓存键
        
        Args:
            name: 对象名称
            params: 参数
            
        Returns:
            str: 缓存键
        """
        if not params:
            return name
            
        # 使用JSON序列化确保一致的字符串表示
        try:
            # 排序键以确保一致性
            param_json = json.dumps(params, sort_keys=True)
            # 使用MD5生成固定长度哈希
            param_hash = hashlib.md5(param_json.encode()).hexdigest()
            return f"{name}_{param_hash}"
        except TypeError:
            # 如果参数不可JSON序列化，回退到简单方法
            return f"{name}_{id(params)}"
    
    async def _get_or_create(self, name: str, class_type: Type[T], params: Optional[Dict[str, Any]] = None,
                            dependencies: Optional[Dict[str, Any]] = None) -> T:
        """
        获取缓存的对象或创建新对象
        
        Args:
            name: 对象名称
            class_type: 对象类型
            params: 创建参数
            dependencies: 要注入的依赖对象
            
        Returns:
            T: 创建的对象
        """
        cache_key = self._create_cache_key(name, params)
        
        # 检查缓存
        if cache_key in self._cache:
            self.logger.debug(f"从缓存返回 {name}")
            return self._cache[cache_key]
        
        # 未缓存，创建新实例
        self.logger.info(f"创建新的 {name} 实例")
        
        try:
            # 合并默认参数和传入参数
            merged_params = self._get_default_params(name).copy() if params is None else {}
            if params:
                merged_params.update(params)
            
            # 创建实例
            instance = class_type(self.config, merged_params)
            
            # 注入依赖
            if dependencies:
                for attr_name, dependency in dependencies.items():
                    setattr(instance, attr_name, dependency)
            
            # 如果有初始化方法，则调用
            if hasattr(instance, 'initialize'):
                await instance.initialize()
            
            # 缓存实例
            self._cache[cache_key] = instance
            
            return instance
            
        except Exception as e:
            self.logger.error(f"创建 {name} 失败: {e}")
            raise ValueError(f"创建 {name} 失败: {e}")
    
    def _get_default_params(self, name: str) -> Dict[str, Any]:
        """
        获取默认参数
        
        Args:
            name: 对象名称
            
        Returns:
            Dict[str, Any]: 默认参数
        """
        return {}
    
    async def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除缓存
        
        Args:
            name: 要清除的对象名称，如果为None则清除所有
        """
        if name:
            # 清除特定对象的所有缓存
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(f"{name}_")]
            
            for key in keys_to_remove:
                instance = self._cache[key]
                
                # 如果实例有关闭方法，则调用
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
                
                del self._cache[key]
            
            self.logger.info(f"已清除 {name} 的缓存实例")
        else:
            # 清除所有缓存
            for instance in self._cache.values():
                if hasattr(instance, 'shutdown'):
                    await instance.shutdown()
            
            self._cache.clear()
            self.logger.info("已清除所有缓存实例")
    
    def with_config_scope(self, scope: str) -> 'AbstractFactory':
        """
        创建带有配置子集的工厂视图
        
        Args:
            scope: 配置范围路径
            
        Returns:
            配置范围受限的工厂实例
        """
        # 创建一个新工厂，仅传递配置的子集
        scoped_config = self.config.get(scope, default={})
        factory_copy = copy.copy(self)
        factory_copy.config = scoped_config
        return factory_copy
    
    async def _load_class_from_path(self, name: str, base_class: Type) -> Type[T]:
        """
        从路径或类引用加载类
        
        Args:
            name: 注册名称
            base_class: 期望的基类类型
        
        Returns:
            加载的类
        """
        if name not in self._registry:
            raise ValueError(f"未知类型: {name}")
        
        class_path = self._registry[name]
        
        # 如果已经是类，验证并返回
        if inspect.isclass(class_path) and issubclass(class_path, base_class):
            return class_path
        
        # 如果是字符串路径，导入相应类
        if isinstance(class_path, str):
            try:
                module_path, class_name = class_path.rsplit('.', 1)
                module = importlib.import_module(module_path)
                loaded_class = getattr(module, class_name)
                
                if not inspect.isclass(loaded_class) or not issubclass(loaded_class, base_class):
                    raise ValueError(f"{class_name} 不是有效的 {base_class.__name__} 子类")
                
                # 更新注册表，使用实际类而不是字符串
                self._registry[name] = loaded_class
                
                return loaded_class
                
            except ImportError as e:
                self.logger.error(f"导入模块 {module_path} 出错: {str(e)}")
                raise ValueError(f"无法导入模块: {str(e)}")
            except AttributeError as e:
                self.logger.error(f"在模块 {module_path} 中未找到类 {class_name}: {str(e)}")
                raise ValueError(f"未找到类: {str(e)}")
        
        raise TypeError(f"无效的注册类型: {type(class_path)}")
    
    def discover_registrable_classes(self, base_class: Type, package_path: str, factory_type: str = None) -> None:
        """
        自动发现并注册带有装饰器的类
        
        Args:
            base_class: 基类类型
            package_path: 包路径
            factory_type: 工厂类型标识符（默认为工厂类名小写）
        """
        if factory_type is None:
            factory_type = self.__class__.__name__.lower()
            
        # 记录已处理的模块，防止重复处理
        processed_modules: Set[str] = set()
        
        # 递归处理包及其子包
        self._process_package(package_path, base_class, factory_type, processed_modules)
        
        self.logger.info(f"自动发现完成，注册了 {len(self._registry)} 个项目")
    
    def _process_package(self, package_path: str, base_class: Type, factory_type: str, processed_modules: Set[str]) -> None:
        """
        递归处理包及其子包
        
        Args:
            package_path: 包路径
            base_class: 基类类型
            factory_type: 工厂类型标识符
            processed_modules: 已处理的模块集合
        """
        if package_path in processed_modules:
            return
            
        processed_modules.add(package_path)
        
        try:
            package = importlib.import_module(package_path)
        except ImportError:
            self.logger.warning(f"无法导入包 {package_path}")
            return
            
        # 获取包的物理路径
        if not hasattr(package, '__path__'):
            return
        
        # 遍历包中的所有模块和子包
        for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            if is_pkg:
                # 递归处理子包
                self._process_package(name, base_class, factory_type, processed_modules)
            else:
                # 处理模块
                self._process_module(name, base_class, factory_type)
    
    def _process_module(self, module_name: str, base_class: Type, factory_type: str) -> None:
        """
        处理单个模块，查找并注册符合条件的类
        
        Args:
            module_name: 模块名称
            base_class: 基类类型
            factory_type: 工厂类型标识符
        """
        try:
            module = importlib.import_module(module_name)
            
            # 查找所有带有_factory_registry属性的类
            for item_name, obj in inspect.getmembers(module, inspect.isclass):
                if (hasattr(obj, '_factory_registry') and 
                    issubclass(obj, base_class) and 
                    obj != base_class):
                    
                    # 获取注册信息
                    factory_info = obj._factory_registry.get(factory_type)
                    if factory_info:
                        reg_name = factory_info['name']
                        metadata = factory_info.get('metadata', {})
                        self.register(reg_name, obj, metadata)
                        self.logger.info(f"自动注册 {reg_name} ({module_name}.{item_name})")
                        
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"加载模块 {module_name} 时出错: {e}")


# 工厂注册装饰器
def register_factory_class(factory_type: str, name: Optional[str] = None, **metadata):
    """
    用于自动注册工厂创建项的装饰器
    
    Args:
        factory_type: 工厂类型（如'strategy_factory', 'trading_mode_factory'）
        name: 注册名称（默认为类名转换）
        **metadata: 其他元数据
    """
    def decorator(cls):
        # 保存注册信息
        if not hasattr(cls, '_factory_registry'):
            cls._factory_registry = {}
        
        # 确定注册名称
        reg_name = name
        if reg_name is None:
            reg_name = cls.__name__
            for suffix in ['Strategy', 'TradingMode']:
                reg_name = reg_name.replace(suffix, '')
            reg_name = reg_name.lower()
        
        # 添加到注册表
        cls._factory_registry[factory_type] = {
            'name': reg_name,
            'metadata': metadata
        }
        
        return cls
    return decorator