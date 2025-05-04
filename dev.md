# 量化交易系统：架构与设计文档

## 1. 系统概述

### 1.1 设计目标

该量化交易系统旨在提供一个轻量级、高性能、易维护的框架，用于开发、测试和部署量化交易策略。系统设计专为个人项目优化，可在单核单线程服务器环境中高效运行。

### 1.2 核心需求

- **轻量化**：最小化外部依赖，降低资源占用
- **易维护**：模块化设计，清晰的组件边界，一致的接口
- **高性能**：在单线程环境中最大化执行效率
- **可扩展**：支持添加新的资产类型、策略和交易所

### 1.3 系统架构概览

![系统架构图](https://placeholder-for-architecture-diagram)

系统采用分层架构，主要包含以下核心模块：

- **核心层(Core)**: 系统核心协调器
- **通用层(Common)**: 基础设施与工具
- **数据层(DataSource)**: 市场数据获取与处理
- **交易所层(Exchange)**: 交易所API集成
- **策略层(Strategy)**: 交易策略实现
- **资产层(Portfolio)**: 资产管理与订单执行
- **交易层(Trading)**: 交易模式实现
- **回测层(Backtest)**: 策略回测框架

## 2. 核心设计理念

### 2.1 异步事件驱动架构

系统基于`async/await`异步编程模型构建，充分利用Python协程特性在单线程环境中实现高效并发。通过事件驱动架构，系统可在等待I/O操作（如网络请求）期间并发处理其他任务。

### 2.2 工厂模式与依赖注入

采用抽象工厂模式和装饰器注册机制创建和管理组件，简化组件初始化和依赖管理。工厂类使用单例模式确保全局唯一实例，提高资源利用效率。

### 2.3 组合模式

资产和策略采用组合模式设计，允许构建复杂的组合结构（如多资产组合、多策略组合），同时保持接口一致性。

### 2.4 资源池化与延迟初始化

通过资源池化复用昂贵资源（如数据库连接、网络连接），并采用延迟初始化策略只在需要时创建组件实例，减少资源占用。

### 2.5 轻量级消息传递

使用内存事件总线实现组件间松耦合通信，避免直接依赖，便于单元测试和组件替换。

## 3. 模块详细设计

### 3.1 通用模块 (Common)

#### 3.1.1 核心组件

- **EventLoop**: 中央事件循环管理器，基于`async_executor.py`优化
- **ConfigManager**: 分层配置管理器，支持默认值和配置覆盖
- **LogManager**: 高效日志系统，支持缓冲写入和日志旋转
- **AbstractFactory**: 工厂抽象基类，提供组件注册和创建机制
- **EventBus**: 轻量级事件总线，用于组件间通信

#### 3.1.2 实现要点

```python
# 优化的EventLoop设计
class EventLoop:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        self._loop = None
        self._tasks = set()
        self._subscriptions = {}
    
    async def run_task(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task
    
    def subscribe(self, event_type, handler):
        if event_type not in self._subscriptions:
            self._subscriptions[event_type] = set()
        self._subscriptions[event_type].add(handler)
        
    async def publish(self, event_type, data):
        if event_type in self._subscriptions:
            for handler in self._subscriptions[event_type]:
                await self.run_task(handler(data))
```

### 3.2 核心模块 (Core)

#### 3.2.1 核心组件

- **Core**: 系统核心协调器，负责初始化和关联各模块
- **EngineFactory**: 创建和管理交易引擎实例

#### 3.2.2 实现要点

```python
class Core:
    def __init__(self, config):
        self.config = config
        self.logger = LogManager.get_logger("core")
        self.event_loop = EventLoop.get_instance()
        self.exchange = None
        self.data_source = None
        self.portfolio = None
        self.strategy = None
        self.trading_mode = None
        
    async def initialize(self):
        # 初始化各模块
        self.exchange = await ExchangeFactory.get_instance(self.config).create()
        self.data_source = await DataSourceFactory.get_instance(self.config).create()
        self.portfolio = await PortfolioFactory.get_instance(self.config).create()
        self.strategy = await StrategyFactory.get_instance(self.config).create()
        self.trading_mode = await TradingModeFactory.get_instance(self.config).create()
        
        # 关联模块
        self.trading_mode.set_portfolio(self.portfolio)
        self.trading_mode.set_strategy(self.strategy)
        self.strategy.set_data_source(self.data_source)
```

### 3.3 数据源模块 (DataSource)

#### 3.3.1 核心组件

- **DataSourceFactory**: 数据源工厂，创建不同数据源
- **BaseDataSource**: 数据源抽象基类
- **LocalDataSource**: 本地数据源实现
- **ExchangeDataSource**: 交易所数据源实现
- **HybridDataSource**: 混合数据源实现
- **DataProcessor**: 数据处理器，负责数据清洗和转换
- **IntegrityChecker**: 数据完整性检查器

#### 3.3.2 实现要点

```python
# 优化的数据源基类
class DataSource:
    def __init__(self, config, params=None):
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger("datasource")
        self.cache = {}
        self.event_bus = EventBus.get_instance()
        
    async def initialize(self):
        # 初始化数据源
        pass
        
    async def get_data(self, symbol, timeframe, start_time, end_time):
        # 从缓存读取或请求新数据
        cache_key = f"{symbol}_{timeframe}_{start_time}_{end_time}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 获取新数据
        data = await self._fetch_data(symbol, timeframe, start_time, end_time)
        
        # 缓存数据
        self.cache[cache_key] = data
        return data
    
    @abstractmethod
    async def _fetch_data(self, symbol, timeframe, start_time, end_time):
        # 子类实现具体数据获取逻辑
        pass
```

### 3.4 交易所模块 (Exchange)

#### 3.4.1 核心组件

- **ExchangeFactory**: 交易所工厂，创建不同交易所连接
- **Exchange**: 交易所抽象基类
- **BinanceExchange**: 币安交易所实现
- **ConnectionPool**: 交易所连接池管理器

#### 3.4.2 实现要点

```python
# 优化的交易所连接池
class ConnectionPool:
    def __init__(self, exchange_class, config, max_connections=5):
        self.exchange_class = exchange_class
        self.config = config
        self.max_connections = max_connections
        self.available = []
        self.in_use = set()
        
    async def get_connection(self):
        if not self.available and len(self.in_use) < self.max_connections:
            # 创建新连接
            conn = self.exchange_class(self.config)
            await conn.initialize()
        else:
            # 从池中获取
            conn = self.available.pop()
            
        self.in_use.add(conn)
        return conn
        
    async def release_connection(self, conn):
        if conn in self.in_use:
            self.in_use.remove(conn)
            self.available.append(conn)
```

### 3.5 策略模块 (Strategy)

#### 3.5.1 核心组件

- **StrategyFactory**: 策略工厂，创建不同交易策略
- **Strategy**: 策略抽象基类
- **DualMAStrategy**: 双均线策略实现
- **MultiFactorStrategy**: 多因子策略实现
- **NeuralNetworkStrategy**: 神经网络策略实现
- **FactorLib**: 因子库，提供常用因子计算

#### 3.5.2 实现要点

```python
# 优化的策略基类
class Strategy:
    def __init__(self, config, params=None):
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger("strategy")
        self.data_source = None
        self.event_bus = EventBus.get_instance()
        
    def set_data_source(self, data_source):
        self.data_source = data_source
        
    @abstractmethod
    async def generate_signals(self, symbol, timeframe, start_time, end_time):
        # 子类实现信号生成逻辑
        pass
        
    async def process_data(self, symbol, timeframe, start_time, end_time):
        # 获取数据
        data = await self.data_source.get_data(symbol, timeframe, start_time, end_time)
        
        # 生成信号
        signals = await self.generate_signals(data)
        
        # 发布信号事件
        await self.event_bus.publish('signals_generated', signals)
        
        return signals
```

### 3.6 资产模块 (Portfolio)

#### 3.6.1 核心组件

- **PortfolioManager**: 资产管理器，管理多个资产
- **Asset**: 资产抽象基类，统一接口
- **SpotAsset**: 现货资产实现
- **FutureAsset**: 期货资产实现
- **BondAsset**: 债券资产实现
- **OptionAsset**: 期权资产实现
- **FundAsset**: 基金资产实现
- **ExecutionEngine**: 订单执行引擎
- **Order**: 订单基类及其子类
- **RiskManager**: 风险管理器

#### 3.6.2 实现要点

```python
# 优化的资产基类
class Asset:
    def __init__(self, name, exchange=None, config=None, params=None):
        self.name = name
        self.exchange = exchange
        self.config = config or ConfigManager()
        self.params = params or {}
        
        # 资产状态
        self._value = Decimal('0')
        self._position = Decimal('0')
        
        # 交易功能（可选）
        self.is_tradable = params.get('tradable', False)
        if self.is_tradable:
            self._setup_trading()
    
    def _setup_trading(self):
        # 初始化交易组件
        self.orders = {}
        self.execution_engine = ExecutionEngine(self.config)
        self.event_bus = EventBus.get_instance()
        
        # 订阅订单事件
        self.event_bus.subscribe('order_filled', self._on_order_filled)
    
    async def _on_order_filled(self, order_data):
        # 处理订单成交事件
        if order_data['asset'] == self.name:
            order_id = order_data['order_id']
            if order_id in self.orders:
                order = self.orders[order_id]
                # 更新订单状态
                # 更新资产仓位
            
    async def get_value(self):
        return float(self._value)
    
    async def update_value(self):
        # 子类实现价值更新逻辑
        return float(self._value)
    
    async def buy(self, amount, **kwargs):
        if not self.is_tradable:
            raise ValueError(f"Asset {self.name} is not tradable")
        # 创建买入订单
        # ...
    
    async def sell(self, amount, **kwargs):
        if not self.is_tradable:
            raise ValueError(f"Asset {self.name} is not tradable")
        # 创建卖出订单
        # ...
```

### 3.7 交易模块 (Trading)

#### 3.7.1 核心组件

- **TradingModeFactory**: 交易模式工厂
- **TradingMode**: 交易模式抽象基类
- **LiveTradingMode**: 实盘交易模式
- **PaperTradingMode**: 模拟交易模式

#### 3.7.2 实现要点

```python
# 优化的交易模式基类
class TradingMode:
    def __init__(self, config, params=None):
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger("trading")
        self.event_bus = EventBus.get_instance()
        self.portfolio = None
        self.strategy = None
        
    def set_portfolio(self, portfolio):
        self.portfolio = portfolio
        
    def set_strategy(self, strategy):
        self.strategy = strategy
        # 订阅策略信号
        self.event_bus.subscribe('signals_generated', self._on_signals)
        
    async def _on_signals(self, signals):
        # 处理策略信号
        if not self.portfolio:
            self.logger.warning("Cannot process signals: portfolio not set")
            return
            
        # 转换信号为订单
        orders = await self._signals_to_orders(signals)
        
        # 执行订单
        for order in orders:
            asset_name = order['symbol']
            action = order['action']
            amount = order['amount']
            
            if action.lower() in ('buy', 'long'):
                await self.portfolio.buy_asset(asset_name, amount)
            elif action.lower() in ('sell', 'short'):
                await self.portfolio.sell_asset(asset_name, amount)
    
    @abstractmethod
    async def _signals_to_orders(self, signals):
        # 子类实现信号转换为订单的逻辑
        pass
```

### 3.8 回测模块 (Backtest)

#### 3.8.1 核心组件

- **BacktestEngine**: 回测引擎抽象基类
- **BacktestEngineFactory**: 回测引擎工厂
- **OHLCVBacktestEngine**: 基于OHLCV数据的回测引擎
- **MarketReplayBacktestEngine**: 市场回放回测引擎
- **BacktestResults**: 回测结果分析器

#### 3.8.2 实现要点

```python
# 优化的回测引擎基类
class BacktestEngine:
    def __init__(self, config, params=None):
        self.config = config
        self.params = params or {}
        self.logger = LogManager.get_logger("backtest")
        self.event_bus = EventBus.get_instance()
        self.data_source = None
        self.strategy = None
        self.portfolio = None
        self.trading_mode = None
        
    async def initialize(self):
        # 初始化回测环境
        factory_params = {'mode': 'backtest'}
        
        # 创建回测专用组件
        self.data_source = await DataSourceFactory.get_instance(self.config).create(**factory_params)
        self.portfolio = await PortfolioFactory.get_instance(self.config).create(**factory_params)
        self.strategy = await StrategyFactory.get_instance(self.config).create(**factory_params)
        self.trading_mode = await TradingModeFactory.get_instance(self.config).create(**factory_params)
        
        # 关联组件
        self.trading_mode.set_portfolio(self.portfolio)
        self.trading_mode.set_strategy(self.strategy)
        self.strategy.set_data_source(self.data_source)
        
    async def run(self, symbols, timeframe, start_time, end_time):
        # 运行回测
        results = {}
        
        for symbol in symbols:
            # 获取历史数据
            data = await self.data_source.get_data(symbol, timeframe, start_time, end_time)
            
            # 运行策略
            signals = await self.strategy.process_data(data)
            
            # 模拟交易执行
            trades = await self._simulate_trades(signals)
            
            # 计算绩效
            performance = await self._calculate_performance(trades)
            
            results[symbol] = {
                'trades': trades,
                'performance': performance
            }
            
        return results
    
    @abstractmethod
    async def _simulate_trades(self, signals):
        # 子类实现交易模拟逻辑
        pass
        
    @abstractmethod
    async def _calculate_performance(self, trades):
        # 子类实现绩效计算逻辑
        pass
```

## 4. 组件交互模式

### 4.1 信号流动路径

1. **数据获取**: 数据源模块获取市场数据
2. **策略处理**: 策略模块处理数据并生成交易信号
3. **信号发布**: 策略模块通过事件总线发布信号
4. **交易转换**: 交易模块将信号转换为交易指令
5. **订单执行**: 资产模块执行交易指令
6. **结果反馈**: 执行结果通过事件总线反馈给系统各组件

### 4.2 事件类型定义

| 事件类型 | 描述 | 数据格式 |
|---------|------|---------|
| `data_updated` | 市场数据更新 | `{symbol, timeframe, data}` |
| `signals_generated` | 策略生成信号 | `[{timestamp, symbol, action, amount, price}]` |
| `order_created` | 订单创建 | `{order_id, symbol, direction, quantity, price, type}` |
| `order_submitted` | 订单提交到交易所 | `{order_id, exchange_order_id, status}` |
| `order_filled` | 订单成交 | `{order_id, filled_quantity, fill_price, timestamp}` |
| `order_canceled` | 订单取消 | `{order_id, reason}` |
| `position_updated` | 仓位更新 | `{symbol, quantity, value}` |
| `error` | 错误事件 | `{module, error_type, message, details}` |

### 4.3 序列图

![信号流转序列图](https://placeholder-for-sequence-diagram)

```
DataSource -> Strategy: 提供市场数据
Strategy -> EventBus: 发布signals_generated事件
EventBus -> TradingMode: 转发signals_generated事件
TradingMode -> Portfolio: 执行buy_asset/sell_asset
Portfolio -> Asset: 执行buy/sell
Asset -> ExecutionEngine: 执行订单
ExecutionEngine -> Exchange: 提交订单
Exchange -> ExecutionEngine: 返回订单状态
ExecutionEngine -> EventBus: 发布order_filled事件
EventBus -> Asset: 转发order_filled事件
Asset -> Portfolio: 更新资产价值
Portfolio -> TradingMode: 更新组合状态
```

## 5. 数据流设计

### 5.1 数据结构定义

#### 5.1.1 市场数据格式

```python
# OHLCV数据格式
{
    'timestamp': pd.DatetimeIndex,  # 时间戳
    'open': pd.Series,              # 开盘价
    'high': pd.Series,              # 最高价
    'low': pd.Series,               # 最低价
    'close': pd.Series,             # 收盘价
    'volume': pd.Series             # 成交量
}
```

#### 5.1.2 信号数据格式

```python
# 信号数据格式
[
    {
        'timestamp': datetime,      # 信号时间
        'symbol': str,              # 交易对
        'action': str,              # 动作：buy, sell, hold
        'quantity': float,          # 数量
        'price': float,             # 价格（可选）
        'stop_loss': float,         # 止损价（可选）
        'take_profit': float        # 止盈价（可选）
    }
]
```

#### 5.1.3 订单数据格式

```python
# 订单基本结构
{
    'order_id': str,               # 订单ID
    'symbol': str,                 # 交易对
    'direction': str,              # 方向：buy, sell
    'quantity': float,             # 数量
    'price': float,                # 价格（市价单为None）
    'type': str,                   # 类型：market, limit, stop, etc.
    'status': str,                 # 状态：created, submitted, partial, filled, canceled
    'timestamp': datetime,         # 创建时间
    'filled_quantity': float,      # 已成交数量
    'avg_filled_price': float,     # 平均成交价格
    'exchange_order_id': str       # 交易所订单ID（可选）
}
```

### 5.2 数据缓存策略

1. **分层缓存**:
   - 内存缓存: 使用LRU策略缓存最近使用的数据
   - 本地缓存: 使用Parquet格式在本地存储数据

2. **缓存失效策略**:
   - 基于时间: 缓存数据超过设定时间后失效
   - 基于容量: 缓存达到最大容量时，移除最不常用的数据

3. **预加载机制**:
   - 根据交易策略预测可能需要的数据并提前加载
   - 异步加载减少等待时间

### 5.3 数据持久化

1. **数据存储格式**:
   - 市场数据: Parquet格式存储，按时间和交易对分区
   - 交易记录: JSON或CSV格式存储

2. **文件命名规则**:
   - 市场数据: `{symbol}_{timeframe}_{start_time}_{end_time}.parquet`
   - 交易记录: `trades_{strategy_name}_{start_date}_{end_date}.json`

3. **目录结构**:
   - 市场数据: `data/{source}/{symbol}/{timeframe}/{year}/{month}/`
   - 交易记录: `records/{strategy}/{year}/{month}/`

## 6. 优化策略

### 6.1 内存优化

1. **使用`__slots__`**: 为关键数据类定义`__slots__`减少内存使用
2. **数据结构选择**: 使用更紧凑的数据结构（如NumPy数组替代列表）
3. **惰性加载**: 只在需要时加载数据
4. **按需计算**: 计算值只在需要时计算，并使用`@property`缓存

```python
class CompactDataObject:
    __slots__ = ('field1', 'field2', 'field3')
    
    def __init__(self, field1, field2, field3):
        self.field1 = field1
        self.field2 = field2
        self.field3 = field3
```

### 6.2 计算优化

1. **向量化计算**: 使用NumPy和Pandas进行向量化操作而非循环
2. **缓存计算结果**: 使用`functools.lru_cache`缓存复杂计算结果
3. **延迟计算**: 使用生成器和惰性求值减少不必要的计算
4. **最小化数据拷贝**: 尽可能使用视图和引用而非拷贝

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def calculate_indicators(data_hash):
    # 复杂指标计算
    pass
```

### 6.3 I/O优化

1. **异步I/O**: 使用异步I/O避免阻塞
2. **批量读写**: 合并小的I/O操作为批量操作
3. **缓冲写入**: 使用缓冲区收集写入操作
4. **本地缓存**: 缓存远程数据到本地

```python
class BufferedWriter:
    def __init__(self, file_path, buffer_size=1000):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.buffer = []
        
    async def write(self, data):
        self.buffer.append(data)
        if len(self.buffer) >= self.buffer_size:
            await self.flush()
            
    async def flush(self):
        if not self.buffer:
            return
        # 批量写入数据
        # ...
        self.buffer = []
```

### 6.4 网络优化

1. **连接池**: 复用网络连接
2. **请求合并**: 合并多个请求为单个批量请求
3. **压缩传输**: 压缩请求和响应数据
4. **智能重试**: 使用指数回退策略进行重试

```python
class RequestBatcher:
    def __init__(self, max_batch_size=10, max_wait_time=0.1):
        self.queue = []
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.last_flush = time.time()
        
    async def add_request(self, request):
        self.queue.append(request)
        
        if len(self.queue) >= self.max_batch_size or time.time() - self.last_flush > self.max_wait_time:
            return await self.flush()
            
        return None
        
    async def flush(self):
        if not self.queue:
            return None
            
        batch = self.queue
        self.queue = []
        self.last_flush = time.time()
        
        # 执行批量请求
        # ...
        
        return results
```

## 7. 错误处理与恢复

### 7.1 错误分类

| 错误类型 | 描述 | 处理策略 |
|---------|------|---------|
| 网络错误 | 连接超时、断开等 | 重试机制 |
| API错误 | 请求失败、响应错误 | 降级服务 |
| 数据错误 | 数据缺失、格式错误 | 插补、平滑 |
| 系统错误 | 资源不足、崩溃 | 恢复机制 |

### 7.2 重试策略

1. **指数回退**: 重试间隔随失败次数指数增长
2. **抖动**: 添加随机抖动避免重试风暴
3. **最大重试次数**: 限制最大重试次数
4. **故障转移**: 在多个服务间故障转移

```python
async def retry_with_backoff(func, max_retries=3, base_delay=1.0, max_delay=30.0):
    retries = 0
    while True:
        try:
            return await func()
        except RetryableError as e:
            retries += 1
            if retries > max_retries:
                raise
                
            # 计算延迟（指数回退+抖动）
            delay = min(base_delay * (2 ** (retries - 1)), max_delay)
            jitter = random.uniform(0.8, 1.2)
            adjusted_delay = delay * jitter
            
            logger.warning(f"Retry {retries}/{max_retries} after {adjusted_delay:.2f}s: {str(e)}")
            await asyncio.sleep(adjusted_delay)
```

### 7.3 降级策略

1. **功能降级**: 禁用非核心功能
2. **数据源切换**: 从备用数据源获取数据
3. **限流**: 减少请求频率
4. **缓存扩展**: 延长缓存有效期

```python
class ServiceDegrader:
    def __init__(self):
        self.degradation_level = 0
        self.degraded_features = set()
        
    def degrade(self, level=1):
        self.degradation_level += level
        
        # 根据降级级别禁用特性
        if self.degradation_level >= 1:
            self.degraded_features.add('real_time_data')
        if self.degradation_level >= 2:
            self.degraded_features.add('market_depth')
        if self.degradation_level >= 3:
            self.degraded_features.add('trading')
            
    def is_degraded(self, feature):
        return feature in self.degraded_features
```

### 7.4 状态恢复

1. **检查点**: 定期保存系统状态
2. **日志重放**: 通过日志恢复操作
3. **状态文件**: 使用JSON/YAML文件存储关键状态
4. **自动重连**: 网络断开后自动重连

```python
class StateManager:
    def __init__(self, state_file):
        self.state_file = state_file
        self.state = {}
        self.last_save = 0
        self.save_interval = 60  # 1分钟
        
    async def load(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    self.state = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            
    async def save(self, force=False):
        now = time.time()
        if force or now - self.last_save > self.save_interval:
            try:
                with open(self.state_file, 'w') as f:
                    json.dump(self.state, f)
                self.last_save = now
            except Exception as e:
                logger.error(f"Failed to save state: {e}")
                
    def update(self, key, value):
        self.state[key] = value
        
    def get(self, key, default=None):
        return self.state.get(key, default)
```

## 8. 配置管理

### 8.1 配置层次结构

1. **默认配置**: 系统内置的默认设置
2. **用户配置**: 用户定义的全局设置
3. **模块配置**: 特定模块的设置
4. **运行时配置**: 运行时覆盖的临时设置

### 8.2 配置文件格式

使用YAML格式的配置文件，支持配置继承和变量替换。

```yaml
# 默认配置示例 (default_config.yaml)
system:
  offline_mode: false
  timezone: UTC
  performance:
    max_threads: 1
    max_memory: 1073741824  # 1GB

logging:
  base_path: ./logs
  level: INFO
  
datasource:
  default: local
  cache:
    enabled: true
    max_size: 104857600  # 100MB
    
exchange:
  default: binance
  
trading:
  mode: backtest
  instruments:
    - BTC/USDT
    - ETH/USDT
  
# 用户配置示例 (user_config.yaml)
system:
  timezone: Asia/Shanghai
  
exchange:
  default: binance
  binance:
    api_key: YOUR_API_KEY
    secret: YOUR_API_SECRET
    
trading:
  mode: paper
  capital:
    initial: 10000
```

### 8.3 配置访问模式

使用统一的配置访问接口，支持默认值和类型转换。

```python
# 配置访问示例
api_key = config.get("exchange", "binance", "api_key", default="")
log_level = config.get("logging", "level", default="INFO")
initial_capital = config.get("trading", "capital", "initial", default=10000, cast=float)
```

## 9. 日志系统

### 9.1 日志级别

| 级别 | 用途 |
|------|------|
| ERROR | 严重错误，影响系统运行 |
| WARNING | 警告，潜在问题 |
| INFO | 重要系统事件 |
| DEBUG | 详细调试信息 |

### 9.2 日志格式

```
[时间戳] | [级别] | [模块] | [文件:行号] | 消息
```

### 9.3 日志输出目标

1. **控制台**: 实时日志显示
2. **文件**: 按日期和模块分割的日志文件
3. **旋转文件**: 按大小或时间旋转的日志文件
4. **性能日志**: 专门记录性能数据的日志文件

### 9.4 日志配置示例

```yaml
logging:
  base_path: ./logs
  level: INFO
  combined_log: true
  format: "%(asctime)s | %(levelname)s | %(module)-18s | [%(filename)s:%(lineno)d] | %(message)s"
  date_format: '%Y-%m-%d %H:%M:%S%z'
  handlers:
    console:
      enabled: true
      level: WARNING
    file:
      enabled: true
      compression: gz
      encoding: utf-8
      buffer:
        enabled: true
        capacity: 1000
        flush_interval: 5
  categories:
    core:
      level: INFO
    data:
      level: INFO
    exchange:
      level: WARNING
    strategy:
      level: INFO
    trading:
      level: INFO
```

## 10. 测试策略

### 10.1 单元测试

使用`pytest`框架进行单元测试，重点测试各组件的核心功能。

```python
# 策略单元测试示例
def test_dual_ma_strategy():
    # 准备测试数据
    test_data = pd.DataFrame({
        'timestamp': pd.date_range(start='2023-01-01', periods=100),
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100,
        'volume': np.random.rand(100) * 1000
    })
    
    # 创建策略实例
    config = ConfigManager('test_config.yaml')
    strategy = DualMAStrategy(config, {
        'fast_period': 5,
        'slow_period': 20
    })
    
    # 生成信号
    signals = strategy.generate_signals(test_data)
    
    # 验证信号
    assert isinstance(signals, list)
    assert all('timestamp' in s and 'action' in s for s in signals)
    assert any(s['action'] == 'buy' for s in signals)
    assert any(s['action'] == 'sell' for s in signals)
```

### 10.2 集成测试

测试多个组件的协作，验证完整流程。

```python
# 集成测试示例
@pytest.mark.asyncio
async def test_strategy_portfolio_integration():
    # 准备测试环境
    config = ConfigManager('test_config.yaml')
    data_source = MockDataSource(config)
    strategy = DualMAStrategy(config)
    portfolio = PortfolioManager(config)
    
    # 配置组件
    strategy.set_data_source(data_source)
    
    # 执行测试
    symbol = 'BTC/USDT'
    signals = await strategy.process_data(symbol, '1h', '2023-01-01', '2023-01-31')
    
    # 将信号发送到投资组合
    trades = []
    for signal in signals:
        if signal['action'] == 'buy':
            result = await portfolio.buy_asset(symbol, signal['quantity'])
        elif signal['action'] == 'sell':
            result = await portfolio.sell_asset(symbol, signal['quantity'])
        trades.append(result)
    
    # 验证结果
    assert len(trades) > 0
    assert all(t['success'] for t in trades)
```

### 10.3 回测测试

使用历史数据测试策略性能。

```python
# 回测测试示例
@pytest.mark.asyncio
async def test_strategy_backtest():
    # 准备测试环境
    config = ConfigManager('test_config.yaml')
    backtest = OHLCVBacktestEngine(config)
    
    # 设置回测参数
    symbols = ['BTC/USDT']
    timeframe = '1h'
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    strategy_type = 'dual_ma'
    strategy_params = {
        'fast_period': 5,
        'slow_period': 20
    }
    
    # 运行回测
    results = await backtest.run(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        strategy_type=strategy_type,
        strategy_params=strategy_params
    )
    
    # 验证结果
    assert 'BTC/USDT' in results
    assert 'trades' in results['BTC/USDT']
    assert 'performance' in results['BTC/USDT']
    
    # 验证性能指标
    performance = results['BTC/USDT']['performance']
    assert 'total_return' in performance
    assert 'sharpe_ratio' in performance
    assert 'max_drawdown' in performance
```

## 11. 部署指南

### 11.1 环境要求

- Python 3.8+
- 单核单线程服务器
- 最小内存: 1GB
- 存储空间: 10GB+

### 11.2 监控与维护

1. **日志监控**:
   - 检查 `logs/system.log` 中的错误和警告
   - 使用 `tail -f logs/system.log` 实时查看日志

2. **性能监控**:
   - 监控内存使用: `ps aux | grep python`
   - 监控CPU使用: `top -p <PID>`

3. **备份**:
   - 定期备份配置文件和数据
   - 使用 `cron` 作业自动备份

```bash
# 备份脚本示例 (backup.sh)
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d)
mkdir -p "$BACKUP_DIR/$DATE"
cp -r config/* "$BACKUP_DIR/$DATE/config"
cp -r data/* "$BACKUP_DIR/$DATE/data"
tar -czf "$BACKUP_DIR/$DATE/logs.tar.gz" logs/
```

## 12. 扩展指南

### 12.1 添加新的资产类型

1. 在 `portfolio/assets/` 目录创建新的资产类文件
2. 继承 `Asset` 基类并实现所需方法
3. 使用 `@register_factory_class` 装饰器注册资产类

### 12.2 添加新的策略

1. 在 `strategy/implementations/` 目录创建新的策略文件
2. 继承 `Strategy` 基类并实现所需方法
3. 使用 `@register_factory_class` 装饰器注册策略

### 12.3 添加新的交易所接口

1. 在 `exchange/` 目录创建新的交易所文件
2. 继承 `Exchange` 基类并实现所需方法
3. 使用 `@register_factory_class` 装饰器注册交易所


## 13. 总结

本文档详细描述了一个针对个人项目优化的量化交易系统架构，该系统采用异步事件驱动架构，结合工厂模式、组合模式等设计模式，实现了轻量级、高性能的交易框架。系统各模块通过事件总线松耦合通信，支持市场数据获取、交易策略执行、资产管理和回测分析等核心功能。

核心优势包括：

1. **轻量化**：最小化外部依赖，优化内存使用，适合单核单线程服务器
2. **易维护**：模块化设计，清晰的组件边界，一致的工厂模式
3. **高性能**：异步事件驱动架构，资源池化，内存优化
4. **可扩展**：支持便捷添加新资产类型、交易策略和交易所接口

该系统提供了从数据获取、策略开发到回测分析和实盘交易的完整解决方案，同时针对个人项目和单线程环境进行了特别优化。

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