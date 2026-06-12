# 交易流水线领域核心需求文档

日期：2026-06-09

关联方案设计文档：

- `docs/design/2026-06-09-trading-pipeline-domain-core-design.md`

## 1. 背景

当前项目已经可以使用本地 Binance 历史数据，完成 `BTC/USDT 1m` 在 `2025-01-01` 到 `2025-01-02` 区间的回测。前一阶段已经修复了主要执行逻辑问题：历史信号不会被重复执行，空仓卖出会被跳过，最终报告显示 60 笔交易，并且结束时没有残留持仓。

下一阶段的目标是让交易流水线更容易演进。后续会替换策略，因此系统需要更清晰的执行逻辑、更独立的组合状态管理，以及更优雅的本地数据读取层。

## 2. 目标

- 解耦策略、风控、执行、组合状态、报告和数据访问。
- 在迁移过程中保留当前 CLI 回测流程，确保仍然可以运行。
- 让策略替换不需要修改执行逻辑或组合状态逻辑。
- 通过清晰的 feed/store 边界访问本地 parquet 数据。
- 每个迁移阶段都必须可测试、可回退、可验证。
- 维护一份 worklist，记录进度、验证证据和下一次继续工作的入口。

## 3. 非目标

- 不一次性重写整个项目。
- 不在第一阶段全局替换 pandas。
- 除非必要，不修改现有 CLI 命令。
- 第一轮迁移不引入数据库。
- 第一轮迁移不实现完整生产级实盘安全框架。
- 架构迁移阶段不优化策略盈利能力。

## 4. 功能需求

### 4.1 领域流水线

系统应逐步暴露一条清晰的交易流水线：

```text
MarketData -> StrategySignal -> RiskDecision -> OrderIntent -> Fill -> PortfolioSnapshot -> Report
```

含义：

- `MarketData`：标准化后的行情数据。
- `StrategySignal`：策略表达的交易观点。
- `RiskDecision`：风控和仓位规则处理后的决策。
- `OrderIntent`：领域层的下单意图。
- `Fill`：执行层返回的成交结果。
- `PortfolioSnapshot`：账户、现金、持仓和净值快照。
- `Report`：绩效和交易结果报告。

### 4.2 策略边界

策略后续应输出策略信号，而不是直接输出接近可执行订单的数据。仓位大小、是否允许交易、是否需要缩放数量，应逐步交给风控策略和组合策略处理。

### 4.3 组合状态

现金、持仓、交易记录、净值和回撤，应由统一的 `PortfolioBook` 或等价领域组件管理。

### 4.4 执行边界

回测、模拟盘、实盘执行应成为共享接口背后的不同执行模型实现，而不是混在同一个执行类里。

### 4.5 数据边界

本地 Binance parquet 数据应通过 `HistoricalStore` 和 `MarketDataFeed` 抽象访问。现有文件布局必须继续支持。

### 4.6 报告边界

报告模块只消费快照、成交记录和运行元数据。报告不应修改交易状态，也不应参与交易决策。

## 5. 验证需求

每个实现阶段都必须保持以下基线回测可运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv
```

基线预期特征：

- 数据从本地存储加载。
- 最终净值接近 `99863.31403628497`。
- 总交易数：`60`。
- 买入交易数：`30`。
- 卖出交易数：`30`。
- 结束持仓：`{}`。

每个阶段也应运行完整单元测试：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m unittest discover tests
```

预期结果：

- 所有测试通过。

## 6. 文档需求

所有项目规划文档都应放在 `E:\myProgram\crypto_trader\docs` 下。

文档目录：

- `docs/requirements`：需求和背景文档。
- `docs/design`：架构和技术方案设计文档。
- `docs/worklist`：实施清单和进度记录。

每完成一个实施任务后，必须更新 worklist，记录：

- 当前状态。
- 修改过的文件。
- 执行过的验证命令。
- 验证结果。
- 下一次继续工作的入口。
