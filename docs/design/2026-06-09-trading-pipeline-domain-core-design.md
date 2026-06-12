# 交易流水线领域核心方案设计

日期：2026-06-09

## 1. 目的

本文档定义 `crypto_trader` 交易流水线的渐进式解耦方案。

当前目标不是重写系统，而是让现有回测、模拟盘和实盘路径逐步依赖同一套领域流水线边界。这样后续替换策略、数据源、执行模型和报告输出时，不需要牵动整条链路。

本文档基于当前项目状态编写：`2025-01-01` 到 `2025-01-02` 的 `BTC/USDT 1m` 本地回测已经可以运行并完成验证。

## 2. 当前问题

项目已经有不少可用模块，但职责边界穿透比较明显。

### 2.1 Trading mode 承担了太多职责

`src/trading/modes/base.py` 当前同时协调：

- 行情数据处理。
- 策略调用。
- 信号过滤。
- 风控校验。
- 执行下单。
- 现金和持仓变更。
- 净值曲线追踪。
- 绩效报告生成。
- 报告文件写入。

这会让 `BaseTradingMode` 既像应用用例，又像领域核心。一个职责里的问题容易泄漏到另一个职责里。

### 2.2 策略输出已经接近执行层数据

`DualMAStrategy` 当前返回包含 `action`、`price`、`quantity` 的 DataFrame。

这意味着策略不只是表达“想做多”或“想退出”，还部分决定了订单大小。更清晰的设计是：策略只输出领域信号，风控和组合策略决定是否交易、交易多少。

### 2.3 执行模型混合了模式和适配器

`ExecutionEngine` 在一个类里同时包含 live、backtest、simple_backtest 逻辑，并且在 live 模式下直接知道 Binance。

领域流水线应该调用 `ExecutionModel` 接口。回测模拟成交、模拟盘成交和 Binance 实盘成交应分别作为接口背后的不同实现。

### 2.4 数据加载有效，但还不够优雅

当前本地数据路径发现逻辑已经可以读取下载好的 Binance parquet 数据。已验证的回测从本地文件加载了 `BTC/USDT 1m` 的 1441 行数据。

但路径约定仍然嵌在 helper/source 代码里。交易流水线不应该关心数据来自：

- `data/historical/binance/<symbol>/<timeframe>/...`
- `data/binance/<symbol>/<timeframe>/...`
- 其他本地或远程存储

这些细节应该被数据 feed 或历史数据 store 适配器屏蔽。

## 3. 目标架构

目标架构保留现有外层形态：CLI/config 创建运行模式，运行模式执行 backtest/paper/live。变化在于中间层变成真正的领域流水线。

```mermaid
flowchart LR
    A["MarketDataFeed"] --> B["MarketData"]
    B --> C["Strategy"]
    C --> D["StrategySignal"]
    D --> E["RiskPolicy"]
    E --> F["OrderIntent"]
    F --> G["ExecutionModel"]
    G --> H["Fill"]
    H --> I["PortfolioBook"]
    I --> J["PerformanceReporter"]
```

依赖方向应该是：

```text
UI / CLI
  -> Application / Use Cases
    -> Domain Core
      -> Ports
        <- Infrastructure Adapters
```

领域核心可以依赖接口和纯领域模型，但不能依赖 Binance、parquet 路径、具体报告写入器或 CLI 细节。

## 4. 领域模型

应先引入这些模型，再迁移行为。第一阶段它们可以通过 adapter 函数和 DataFrame 共存。

### 4.1 MarketBar

表示一根标准化后的 OHLCV K 线。

字段：

- `symbol`
- `timeframe`
- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

规则：

- `timestamp` 必须带时区。
- `symbol` 使用统一交易对格式，例如 `BTC/USDT`。
- 数值字段可以是 float 或 decimal，但内部必须保持一致。

### 4.2 MarketSlice

表示某一个流水线时间点可用的所有 K 线。

字段：

- `timestamp`
- `bars_by_symbol`

回测用例后续应遍历 `MarketSlice`，而不是直接遍历原始 DataFrame。

### 4.3 StrategySignal

表示策略观点，不是可执行订单。

字段：

- `signal_id`
- `symbol`
- `timestamp`
- `side`：`buy`、`sell`、`short`、`cover` 或 `hold`
- `strength`
- `reason`
- `metadata`

规则：

- 策略可以附带建议价格或信心分数，但不应拥有最终仓位大小的决定权。
- 信号去重应基于 `signal_id`，或基于 `(strategy_id, symbol, side, timestamp)`。

### 4.4 RiskDecision

表示风控策略作用于策略信号后的结果。

字段：

- `accepted`
- `reason`
- `target_notional`
- `target_quantity`
- `adjusted_signal`

风控可以拒绝信号、原样放行信号，或调整信号对应的交易数量。

### 4.5 OrderIntent

表示领域层的交易意图。

字段：

- `order_intent_id`
- `symbol`
- `side`
- `quantity`
- `order_type`
- `created_at`
- `limit_price`
- `source_signal_id`

这还不是交易所订单，而是执行模型的输入。

### 4.6 Fill

表示执行结果。

字段：

- `fill_id`
- `order_intent_id`
- `symbol`
- `side`
- `timestamp`
- `quantity`
- `price`
- `commission`
- `slippage`
- `status`

组合状态应该根据 `Fill` 更新，而不是根据原始订单更新。

### 4.7 PortfolioSnapshot

表示某一时间点的账户状态。

字段：

- `timestamp`
- `cash`
- `positions`
- `market_prices`
- `equity`
- `realized_pnl`
- `unrealized_pnl`

## 5. Ports 接口

Ports 是应用层或用例层依赖的接口。

### 5.1 MarketDataFeed

职责：

- 为回测提供历史行情切片。
- 为模拟盘和实盘提供最新行情切片。
- 隐藏文件、数据库或交易所细节。

建议方法：

```python
load_range(symbols, timeframe, start, end) -> Iterable[MarketSlice]
latest(symbols, timeframe) -> MarketSlice
```

### 5.2 StrategyPort

职责：

- 接收行情历史或当前市场上下文。
- 输出策略信号。

建议方法：

```python
generate(context) -> list[StrategySignal]
```

### 5.3 RiskPolicy

职责：

- 校验策略信号。
- 应用仓位大小规则。
- 防止空仓无效卖出。
- 执行回撤、敞口、集中度等限制。

建议方法：

```python
evaluate(signal, portfolio, market) -> RiskDecision
```

### 5.4 ExecutionModel

职责：

- 将下单意图转换为成交结果。
- 建模手续费、滑点、部分成交和失败。
- 隐藏成交来自模拟还是实盘。

建议方法：

```python
execute(order_intent, market, portfolio) -> Fill
```

### 5.5 Reporter

职责：

- 消费组合快照、成交记录和运行元数据。
- 生成 JSON、CSV、图表或 HTML 报告。

报告模块不应修改组合状态，也不应影响交易决策。

## 6. 应用用例

三种运行模式后续应变成较薄的编排层。

### 6.1 BacktestUseCase

输入：

- config
- symbols
- timeframe
- start/end date
- market data feed
- strategy
- risk policy
- execution model
- portfolio book
- reporter

循环：

1. 加载历史行情切片。
2. 对每个切片更新当前市场价格。
3. 请求策略生成信号。
4. 请求风控评估每个信号。
5. 将通过的决策转换成下单意图。
6. 通过回测执行模型执行下单意图。
7. 将成交结果应用到组合账本。
8. 记录快照和事件。
9. 生成最终报告。

### 6.2 PaperUseCase

与回测使用同一条领域流程，但行情来自最新或实时数据，执行使用模拟成交。

### 6.3 LiveUseCase

与回测使用同一条领域流程，但执行使用真实交易所适配器。实盘用例必须增加更强的安全控制：

- 显式 dry-run/live 开关。
- 交易前现金和持仓检查。
- 交易所响应标准化。
- 每个 `OrderIntent` 对应幂等键。
- 实盘订单审计日志。

## 7. 数据层设计

数据层应拆成三个概念。

### 7.1 HistoricalStore

底层存储抽象。

职责：

- 理解本地 parquet 布局。
- 发现文件。
- 读取文件。
- 应用日期过滤。
- 返回标准化 DataFrame 或 `MarketBar`。

初始实现：

- `ParquetHistoricalStore`

它应支持当前本地布局：

- `data/historical/<timeframe>/<symbol>/...`
- `data/historical/binance/<symbol>/<timeframe>/...`
- `data/binance/<symbol>/<timeframe>/...`

### 7.2 MarketDataNormalizer

职责：

- 统一时间戳列。
- 确保时间戳带时区。
- 确保 OHLCV 数值列可用。
- 按时间排序。
- 去除重复 K 线。
- 附加统一交易对符号。

这样可以让策略和执行代码不依赖原始文件 schema 的差异。

### 7.3 MarketDataFeed Adapter

供用例层使用的高层适配器。

职责：

- 从 `HistoricalStore` 加载数据。
- 标准化数据。
- 生成 `MarketSlice` 对象。

该适配器后续应成为回测流水线唯一直接依赖的数据组件。

## 8. 错误处理

流水线应对结构性错误快速失败，对市场运行事件安全跳过并记录。

结构性错误：

- 配置的数据路径缺失。
- 缺少必要 OHLCV 列。
- timeframe 无效。
- 日期区间无效。
- 策略参数无效。

这些错误应停止运行，并输出清晰错误。

市场或运行事件：

- 某个 symbol 在某个时间点没有 K 线。
- 风控拒绝信号。
- 订单未成交。
- 部分成交。
- 信号重复。

这些事件应被记录，并出现在报告或事件日志中。

## 9. 迁移计划

### Phase 1：添加领域模型和适配器，不改变运行行为

创建领域模型类和转换 helper。

验证：

- 当前所有单元测试通过。
- `2025-01-01` 到 `2025-01-02` 的 `BTC/USDT 1m` 回测仍然返回 60 笔交易、30 笔买入、30 笔卖出，并且结束无持仓。

### Phase 2：抽出 PortfolioBook

将现金、持仓、交易记录、净值曲线和回撤逻辑从 `BaseTradingMode` 迁出。

验证：

- 单元测试覆盖买入、卖出、空仓卖出拒绝、手续费和净值计算。
- 同一个 1 天回测报告在数值容忍范围内保持一致。

### Phase 3：抽出 BacktestExecutionModel

将模拟成交逻辑从 `ExecutionEngine` 中迁出。

验证：

- 单元测试覆盖买入滑点、卖出滑点、成交量上限、部分成交和缺失 K 线。
- 当前 CLI 回测仍然可运行。

### Phase 4：引入 MarketDataFeed 和 ParquetHistoricalStore

将本地数据路径发现逻辑移到 store/feed 适配器背后。

验证：

- 测试覆盖所有当前数据布局。
- `2025-01-01` 到 `2025-01-02` 的本地 `BTC/USDT 1m` 数据在切片前能加载 1441 行。
- CLI 回测不依赖具体 parquet 路径规则。

### Phase 5：让 Trading Modes 变薄

让 `BacktestTradingMode`、`PaperTradingMode` 和 `LiveTradingMode` 组装并调用 use case，而不是自己拥有整条流水线行为。

验证：

- backtest、paper、live 初始化可以分别测试。
- live 模式没有显式实盘安全配置时不能启动。

## 10. 非目标

本设计不要求：

- 重写每个策略。
- 全局替换 pandas。
- 立即修改配置格式。
- 立即修改报告输出格式。
- 添加数据库。
- 修改已验证的运行命令。

第一阶段实现应保留当前 CLI 行为。

## 11. 成功标准

架构工作成功的标准：

- 替换策略代码时，不需要改执行逻辑或组合状态逻辑。
- 替换数据源实现时，不需要改策略或执行逻辑。
- 回测和实盘共享同一条 `OrderIntent -> Fill -> PortfolioBook` 流程。
- 组合状态只在一个地方发生变更。
- 本地 parquet 数据可以通过 feed/store 适配器发现和读取。
- 已验证的 `2025-01-01` 到 `2025-01-02` 的 `BTC/USDT 1m` 回测可复现。

## 12. 推荐下一步

先只实施 Phase 1 和 Phase 2。

这两个阶段会建立核心边界，并移除当前风险最高的耦合点，同时保留已经跑通的回测。执行层和数据层抽取可以在组合状态集中之后再推进。
