# 工厂模式架构规范文档

## 1. 概述

本文档定义了交易系统中工厂类的实现规范，确保各组件工厂采用一致的设计模式和接口。该规范基于抽象工厂模式和单例模式，实现了组件的灵活创建、自动发现和注册机制。

## 2. 核心原则

- **抽象工厂模式**: 使用抽象工厂创建一系列相关的组件
- **单例模式**: 工厂类使用单例模式确保系统中只有一个实例
- **自动发现**: 支持自动发现和注册组件
- **配置驱动**: 使用统一的配置管理器驱动组件的创建和配置
- **一致的接口**: 所有工厂类提供一致的创建和管理接口
- **日志集成**: 集成标准化的日志记录机制

## 3. 基础结构

### 3.1 AbstractFactory 基类

所有工厂类继承自 `AbstractFactory` 基类，该基类提供以下核心功能：

- 单例模式实现
- 组件注册机制
- 组件创建和解析接口
- 动态类加载功能
- 元数据管理

```python
class AbstractFactory:
    """抽象工厂基类"""
    
    # 单例实例字典
    _instances = {}
    
    @classmethod
    def get_instance(cls, config):
        """获取单例实例"""
        if cls not in cls._instances:
            cls._instances[cls] = cls(config)
        return cls._instances[cls]
    
    def __init__(self, config):
        """初始化工厂"""
        self.config = config
        self.logger = LogManager.get_logger(self.__class__.__name__)
        self._registry = {}
        self._metadata = {}
        
    def register(self, name, class_path, metadata=None):
        """注册组件"""
        self._registry[name] = class_path
        if metadata:
            self._metadata[name] = metadata
            
    async def create(self, name=None, params=None):
        """创建组件实例"""
        resolved_name = await self._resolve_name(name)
        concrete_class = await self._get_concrete_class(resolved_name)
        instance = concrete_class(self.config, params)
        await instance.initialize()
        return instance
        
    async def _load_class_from_path(self, name, base_class):
        """从路径加载类"""
        # 动态导入实现
        
    def discover_registrable_classes(self, base_class, module_path, log_prefix):
        """自动发现可注册类"""
        # 自动发现实现
```

### 3.2 组件基类

每个工厂对应的组件应当有一个抽象基类，定义了该组件的通用接口：

```python
class BaseComponent(ABC):
    """组件基类"""
    
    def __init__(self, config, params=None):
        """初始化组件"""
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger(self.__class__.__name__)
        
    @abstractmethod
    async def initialize(self):
        """初始化组件"""
        pass
        
    @abstractmethod
    async def shutdown(self):
        """关闭组件"""
        pass
```

## 4. 工厂类规范

### 4.1 命名规范

- 工厂类命名: `[组件名]Factory`
- 获取工厂方法命名: `get_[组件名]_factory`
- 基类命名: `Base[组件名]`

### 4.2 工厂类结构

每个工厂类应当包含以下方法和属性：

1. **构造函数**:
   - 接收 `ConfigManager` 实例
   - 调用 `_register_default_[组件名]s()` 注册默认组件
   - 调用 `_discover_[组件名]s()` 自动发现其他组件

2. **组件注册方法**:
   - `_register_default_[组件名]s()`: 注册内置组件
   - `_discover_[组件名]s()`: 自动发现外部组件

3. **实例创建方法**:
   - `_get_concrete_class(name)`: 获取组件类
   - `_resolve_name(name)`: 解析组件名称

4. **辅助方法**:
   - `get_available_[组件名]s()`: 获取可用组件列表
   - 其他特定于组件的辅助方法

### 4.3 获取工厂函数

每个工厂类应当提供一个全局函数用于获取单例实例：

```python
def get_[组件名]_factory(config: ConfigManager) -> [组件名]Factory:
    """获取[组件名]工厂单例实例"""
    return [组件名]Factory.get_instance(config)
```

## 5. 组件基类规范

### 5.1 基本方法

每个组件基类应当定义以下方法：

1. **构造函数**:
   - 接收 `ConfigManager` 实例和可选参数字典
   - 初始化日志记录器
   - 设置初始状态

2. **初始化方法**:
   - `async def initialize()`: 异步初始化方法
   - 连接外部资源
   - 加载配置
   - 设置内部状态

3. **关闭方法**:
   - `async def shutdown()`: 异步关闭方法
   - 释放资源
   - 清理状态

### 5.2 特定功能方法

根据组件类型，基类应当定义适当的功能方法：

- 数据源组件: `fetch_data()`, `subscribe()`, 等
- 策略组件: `process_data()`, `generate_signals()`, 等
- 风险管理组件: `validate_signals()`, `check_risk_limits()`, 等
- 执行组件: `execute()`, `cancel_order()`, 等

## 6. 配置集成

### 6.1 配置访问规范

- 使用路径形式访问配置: `config.get("section", "subsection", "key", default=value)`
- 提供合理默认值
- 处理缺失配置的情况

### 6.2 组件特定配置

工厂和组件应当从配置中读取特定于它们的配置部分：

```python
# 工厂配置示例
strategy_config = config.get("strategy", default={})
default_strategy = strategy_config.get("default", "dual_ma")

# 组件配置示例
risk_limit = config.get("risk", "max_drawdown", default=0.1)
```

## 7. 日志记录规范

### 7.1 日志记录器获取

- 使用一致的日志记录器命名: `self.logger = LogManager.get_logger(f"{category}.{class_name}")`
- 工厂类使用其类名作为记录器名称
- 组件类使用类别前缀和类名组合作为记录器名称

### 7.2 日志记录级别

- **ERROR**: 阻止正常操作的错误
- **WARNING**: 需要注意但不影响操作的问题
- **INFO**: 重要的操作信息，如组件初始化、交易执行等
- **DEBUG**: 详细的调试信息

## 8. 异常处理

工厂和组件应当实现一致的异常处理策略：

- 定义特定的异常类: `[组件名]Error`, `[组件名]ConfigError`, 等
- 在适当的位置捕获和处理异常
- 记录详细的错误信息
- 在必要时优雅降级

## 9. 单元测试指南

为工厂和组件编写单元测试时，应遵循以下原则：

- 使用模拟的 `ConfigManager` 进行测试
- 测试组件的创建和初始化
- 测试特定功能和边缘情况
- 测试异常处理
- 测试配置变更的影响

## 10. 示例实现

示例工厂类和组件基类的实现请参考项目代码库中的：

- `TradingModeFactory` 和 `BaseTradingMode`
- `StrategyFactory` 和 `BaseStrategy`
- `DataSourceFactory` 和 `BaseDataSource`
- `RiskManagerFactory` 和 `BaseRiskManager`
- `ExecutionFactory` 和 `BaseExecution`

## 11.完整的项目目录
├── src
│   ├── backtest
│   │   ├── backtest.py
│   │   ├── engine
│   │   │   ├── base.py
│   │   │   ├── factory.py
│   │   │   ├── init.py
│   │   │   ├── market_replay.py
│   │   │   └── ohlcv.py
│   │   ├── init.py
│   │   └── utils.py
│   ├── common
│   │   ├── abstract_factory.py
│   │   ├── async_executor.py
│   │   ├── cli.py
│   │   ├── config.py
│   │   ├── helpers.py
│   │   ├── init.py
│   │   └── log_manager.py
│   ├── core
│   │   ├── core.py
│   │   └── init.py
│   ├── datasource
│   │   ├── engine.py
│   │   ├── init.py
│   │   ├── integrity.py
│   │   ├── manager.py
│   │   ├── processor.py
│   │   └── sources
│   │       ├── base.py
│   │       ├── exchange.py
│   │       ├── factory.py
│   │       ├── hybrid.py
│   │       └── local.py
│   ├── exchange
│   │   ├── base.py
│   │   ├── binance.py
│   │   ├── factory.py
│   │   └── init.py
│   ├── main.py
│   ├── portfolio
│   │   ├── assets
│   │   │   ├── base.py
│   │   │   ├── bond.py
│   │   │   ├── factory.py
│   │   │   ├── fund.py
│   │   │   ├── future.py
│   │   │   ├── init.py
│   │   │   ├── option.py
│   │   │   ├── spot.py
│   │   │   └── tradable_asset.py
│   │   ├── execution
│   │   │   ├── engine.py
│   │   │   ├── init.py
│   │   │   └── order.py
│   │   ├── init.py
│   │   ├── manager.py
│   │   ├── portfolio.py
│   │   └── risk.py
│   ├── strategy
│   │   ├── base.py
│   │   ├── factor_lib.py
│   │   ├── factory.py
│   │   ├── implementations
│   │   │   ├── dual_ma.py
│   │   │   ├── init.py
│   │   │   ├── multi_factors.py
│   │   │   └── neural_network.py
│   │   ├── init.py
│   │   ├── performance.py
│   │   └── registry.py
│   └── trading
│       ├── base.py
│       ├── factory.py
│       ├── init.py
│       ├── live.py
│       └── paper.py
└── tests
    └── network.py