# 交易流水线领域核心实施清单

日期：2026-06-09

关联文档：

- 需求文档：`docs/requirements/2026-06-09-trading-pipeline-domain-core-requirements.md`
- 方案设计文档：`docs/design/2026-06-09-trading-pipeline-domain-core-design.md`

## 状态说明

- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[!]` 已阻塞

## 当前基线

执行逻辑修复后的当前基线：

- 回测区间：`2025-01-01` 到 `2025-01-02`
- 交易对和周期：`BTC/USDT 1m`
- 策略：`dual_ma`
- 回测引擎：`ohlcv`
- 最近已知报告：`reports/backtest/backtest_report_20260609_093604.json`
- 最终净值：`99863.31403628497`
- 总收益率：`-0.13668596371503372%`
- 总交易数：`60`
- 买入交易数：`30`
- 卖出交易数：`30`
- 结束持仓：`{}`

基线回测验证命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv
```

完整测试命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m unittest discover tests
```

## 实施项

### Phase 0：文档结构整理

- [x] 创建 `docs/requirements`、`docs/design` 和 `docs/worklist`。
- [x] 将领域核心方案设计文档移动到 `docs/design`。
- [x] 创建需求文档。
- [x] 创建本文档作为实施清单和进度记录。

完成证据：

- 已创建文件：
  - `docs/requirements/2026-06-09-trading-pipeline-domain-core-requirements.md`
  - `docs/design/2026-06-09-trading-pipeline-domain-core-design.md`
  - `docs/worklist/2026-06-09-trading-pipeline-domain-core-worklist.md`

### Phase 1：领域模型和适配器

目标：添加领域模型和转换 helper，不改变当前运行行为。

- [x] 创建领域包。
  - 候选路径：`src/domain`
  - 预期文件：
    - `src/domain/__init__.py`
    - `src/domain/models.py`
    - `src/domain/ports.py`
- [x] 定义 `MarketBar`、`MarketSlice`、`StrategySignal`、`RiskDecision`、`OrderIntent`、`Fill` 和 `PortfolioSnapshot`。
- [x] 添加 DataFrame 到领域对象的转换 helper。
- [x] 添加模型构造和转换相关的单元测试。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期验证：

- 单元测试通过。
- 基线回测仍然产生 60 笔交易，并且结束无持仓。

完成证据：

- 新增文件：
  - `src/domain/__init__.py`
  - `src/domain/models.py`
  - `src/domain/ports.py`
  - `src/domain/adapters.py`
  - `tests/test_domain_models.py`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_models`
  - 结果：4 个测试因 `ModuleNotFoundError: No module named 'src.domain'` 失败，符合预期。
- GREEN 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_models`
  - 结果：`Ran 4 tests in 0.063s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 25 tests in 6.201s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_091850.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 2：抽出 PortfolioBook

目标：集中管理现金、持仓、交易记录、净值和回撤行为。

- [x] 添加 `PortfolioBook` 领域组件。
  - 候选路径：`src/domain/portfolio.py`
- [x] 编写买入成交应用测试。
- [x] 编写卖出成交应用测试。
- [x] 编写空仓卖出拒绝或无操作处理测试。
- [x] 编写手续费和净值计算测试。
- [x] 将 `PortfolioBook` 接入当前 `BaseTradingMode` 行为背后。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期验证：

- 现有报告数值在小范围数值误差内保持一致。
- 组合状态变更集中到一个组件中，而不是散落在 mode 逻辑里。

完成证据：

- 新增/修改文件：
  - `src/domain/portfolio.py`
  - `tests/test_portfolio_book.py`
  - `src/domain/__init__.py`
  - `src/trading/modes/base.py`
- 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_portfolio_book tests.test_backtest_mode`
  - 结果：`Ran 8 tests in 6.490s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 29 tests in 5.764s`，`OK`。
- 基线回测：
  - 报告：`reports/backtest/backtest_report_20260609_092427.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 3：抽出 BacktestExecutionModel

目标：将模拟成交行为从混合型 `ExecutionEngine` 中迁出。

- [x] 如果 `src/domain/ports.py` 中还没有执行模型接口，则添加该接口。
- [x] 添加 `BacktestExecutionModel`。
  - 候选路径：`src/trading/execution/backtest_model.py`
- [x] 编写买入滑点测试。
- [x] 编写卖出滑点测试。
- [x] 编写成交量上限和部分成交测试。
- [x] 编写缺失 K 线行为测试。
- [x] 将 backtest mode 接入新的执行模型，同时保留 CLI 行为。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期验证：

- 回测执行逻辑独立于 Binance 实盘适配器。
- 基线 CLI 命令仍然可用。

完成证据：

- 新增/修改文件：
  - `src/trading/execution/backtest_model.py`
  - `tests/test_backtest_execution_model.py`
  - `src/trading/execution/manager.py`
- 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model tests.test_execution_engine`
  - 结果：`Ran 6 tests in 5.172s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 33 tests in 5.806s`，`OK`。
- 基线回测：
  - 报告：`reports/backtest/backtest_report_20260609_092731.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 4：MarketDataFeed 和 ParquetHistoricalStore

目标：将本地文件布局隐藏在数据 store/feed 适配器背后。

- [x] 添加 `HistoricalStore` 抽象。
- [x] 添加 `ParquetHistoricalStore`。
  - 候选路径：`src/datasource/stores/parquet_store.py`
- [x] 添加 `MarketDataNormalizer`。
  - 候选路径：`src/datasource/normalizer.py`
- [x] 添加回测用 `MarketDataFeed` 适配器。
  - 候选路径：`src/datasource/feeds/market_data_feed.py`
- [x] 用测试覆盖当前本地布局：
  - `data/historical/<timeframe>/<symbol>/...`
  - `data/historical/binance/<symbol>/<timeframe>/...`
  - `data/binance/<symbol>/<timeframe>/...`
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期验证：

- `2025-01-01` 到 `2025-01-02` 的 `BTC/USDT 1m` 本地数据在时间切片前能加载 1441 行。
- 交易流水线不依赖具体 parquet 路径规则。

完成证据：

- 新增/修改文件：
  - `src/datasource/stores/__init__.py`
  - `src/datasource/stores/parquet_store.py`
  - `src/datasource/normalizer.py`
  - `src/datasource/feeds/__init__.py`
  - `src/datasource/feeds/market_data_feed.py`
  - `src/datasource/datasources.py`
  - `tests/test_market_data_feed.py`
- 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_market_data_feed tests.test_parquet_file_manager tests.test_data_manager`
  - 结果：`Ran 6 tests in 1.988s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 35 tests in 7.491s`，`OK`。
- 基线回测：
  - 报告：`reports/backtest/backtest_report_20260609_093034.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 5：将 Trading Modes 变成薄用例编排层

目标：让 backtest、paper、live modes 负责组装 use case，而不是拥有核心交易流水线行为。

- [x] 添加 `BacktestUseCase`。
  - 候选路径：`src/application/backtest_use_case.py`
- [x] 让 `BacktestTradingMode` 通过 `BacktestUseCase` 运行。
- [x] 只有当前代码需要时，才添加 paper/live use case 骨架。
- [x] 保留现有 CLI 命令行为。
- [x] 添加 backtest use case 集成测试。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期验证：

- `BaseTradingMode` 不再拥有整条交易流水线。
- 不需要阅读策略、执行和报告内部，也能理解回测编排流程。

完成证据：

- 新增/修改文件：
  - `src/application/__init__.py`
  - `src/application/backtest_use_case.py`
  - `src/trading/modes/backtest.py`
  - `tests/test_backtest_use_case.py`
  - `tests/test_backtest_mode.py`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case tests.test_backtest_mode`
  - 结果：2 个错误，分别是 `ModuleNotFoundError: No module named 'src.application'` 和 `BacktestUseCase` 未在 backtest mode 模块暴露，符合预期。
- GREEN 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case tests.test_backtest_mode`
  - 结果：`Ran 6 tests in 6.072s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 37 tests in 6.215s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_093604.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

## 进度记录

### 2026-06-09

- 创建文档目录结构。
- 将领域核心方案设计文档移动到 `docs/design`。
- 添加需求文档和 worklist 文档。
- 将三份文档翻译为中文，保留必要英文领域模型名、路径和命令。
- 完成 Phase 1：新增领域模型、ports 和 DataFrame adapter，当前未接入运行路径，因此不改变现有回测行为。
- 完成 Phase 2：新增 `PortfolioBook`，并将现金、持仓、交易记录、净值和回撤同步接入 `BaseTradingMode`。
- 完成 Phase 3：新增 `BacktestExecutionModel`，让回测成交模型从 `ExecutionEngine` 的混合逻辑中独立出来。
- 完成 Phase 4：新增 parquet 本地历史数据 store、normalizer 和 market data feed，并让 `LocalSource` 通过 store 读取本地数据。
- 完成 Phase 5：新增 `BacktestUseCase`，将回测 timestamp loop 从 `BacktestTradingMode` 中抽到应用层；当前没有新增 paper/live use case 骨架，因为本阶段只需要保护回测主路径。
- 当前架构状态：领域模型、组合账本、回测成交模型、本地历史数据读取和回测用例编排已经有独立边界；`BaseTradingMode` 仍保留信号生成、风险校验、订单执行和报告生成等兼容逻辑，后续可继续按风险较低的方式拆分。
- 下一次继续入口：讨论是否进入下一轮解耦，候选方向包括 `SignalPipeline`、`RiskUseCase`、`ReportUseCase`，以及 paper/live mode 是否要复用相同应用层接口。
