# 交易流水线完整解耦实施清单

日期：2026-06-09

关联文档：

- 完整解耦设计文档：`docs/design/2026-06-09-trading-pipeline-full-decoupling-design.md`
- 第二阶段设计文档：`docs/design/2026-06-09-trading-pipeline-phase-2-design.md`
- 第二阶段实施清单：`docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`

## 状态说明

- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[!]` 已阻塞

## 当前基线

- 回测区间：`2025-01-01` 到 `2025-01-02`
- 交易对和周期：`BTC/USDT 1m`
- 策略：`dual_ma`
- 回测引擎：`ohlcv`
- 最近已知报告：`reports/backtest/backtest_report_20260609_104133.json`
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

## 解耦完成标准

- [x] `src/domain` 不依赖 `pandas`、mode、config、具体策略、具体风控、执行引擎、交易所或文件系统。
- [x] 主链路使用 `MarketSlice -> StrategySignal -> RiskDecision -> OrderIntent -> Fill -> PortfolioSnapshot`。
- [x] `DataFrame` 只存在于 legacy adapter、基础设施读取和报告表格导出边界内。
- [x] backtest、paper、live 共用同一个 `DomainTradingPipeline`。
- [x] `PortfolioBook` 是现金、持仓、净值和回撤的唯一事实来源。
- [x] report 不再读取 mode/context 私有状态。
- [x] Runtime Builder 统一组装 feed、strategy、risk、execution、reporter、portfolio 和 use case。

## 实施项

### Phase 11：解耦边界和 import boundary tests

目标：先定义并用测试守住“真正解耦”的边界。

- [x] 新增 import boundary tests。
- [x] 验证 `src/domain` 不 import `pandas`。
- [x] 验证 `src/domain` 不 import `src.trading.modes`。
- [x] 验证 `src/domain` 不 import `ConfigManager`。
- [x] 验证 `src/domain` 不 import `ExecutionEngine`。
- [x] 验证 `src/domain` 不 import 具体 strategy/risk/data source/exchange/report writer。
- [x] 更新 `src/domain/ports.py`，让 ports 与完整链路一致。
- [x] 明确 DataFrame 允许存在的边界目录。
- [x] 运行聚焦测试。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/domain/ports.py`
- `tests/test_domain_boundaries.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- boundary test 能在 domain 误 import pandas 时失败。
- 当前基线回测数值不变。

完成证据：

- 新增/修改文件：
  - `tests/test_domain_boundaries.py`
  - `src/domain/ports.py`
  - `src/application/adapters/dataframe_domain_adapter.py`
  - `src/datasource/feeds/market_data_feed.py`
  - `tests/test_domain_models.py`
  - `src/domain/adapters.py`：已删除，避免 pandas 留在 domain 包内。
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries tests.test_domain_models tests.test_portfolio_book`
  - 结果：1 个失败，`src/domain/adapters.py` 仍 `import pandas`，说明边界测试成功抓到 domain 依赖污染。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries tests.test_domain_models tests.test_portfolio_book tests.test_market_data_feed`
  - 结果：`Ran 14 tests in 1.192s`，`OK (skipped=1)`；跳过项为 Phase 12 预留的 `DomainTradingPipeline` 文件检查。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 61 tests in 11.880s`，`OK (skipped=1)`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_112826.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 12：新增纯 DomainTradingPipeline

目标：建立不依赖 DataFrame 和 mode 的领域流水线。

- [x] 新增 `DomainTradingPipeline`。
- [x] pipeline 接收 `MarketSlice`。
- [x] pipeline 依赖 `StrategyPort`、`RiskPolicy`、`ExecutionModel`、`PortfolioBook`、`Reporter`。
- [x] pipeline 将 `RiskDecision` 转成 `OrderIntent`。
- [x] pipeline 将 `Fill` 应用到 `PortfolioBook`。
- [x] pipeline 记录 `PortfolioSnapshot` 和 `Fill` 到 reporter。
- [x] 使用 fake strategy/fake risk/fake execution/fake reporter 写纯领域单测。
- [x] 验证 domain pipeline 测试不需要 pandas。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/domain/trading_pipeline.py`
- `src/domain/models.py`
- `src/domain/portfolio.py`
- `tests/test_domain_trading_pipeline.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 纯 fake 组件可以跑通：

```text
MarketSlice -> StrategySignal -> RiskDecision -> OrderIntent -> Fill -> PortfolioBook -> Reporter
```

完成证据：

- 新增/修改文件：
  - `src/domain/trading_pipeline.py`
  - `tests/test_domain_trading_pipeline.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：1 个错误，`ModuleNotFoundError: No module named 'src.domain.trading_pipeline'`，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_trading_pipeline tests.test_domain_boundaries tests.test_domain_models tests.test_portfolio_book`
  - 结果：`Ran 12 tests in 0.251s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 63 tests in 11.706s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_113121.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 13：Legacy Strategy Adapter

目标：旧 DataFrame 策略不重写，但对领域 pipeline 表现为 `StrategyPort`。

- [x] 新增 `LegacyDataFrameStrategyAdapter`。
- [x] 将 `MarketSlice` 转成旧策略需要的 DataFrame。
- [x] 调用 `legacy_strategy.process_data(data, symbol)`。
- [x] 将输出 DataFrame 转成 `StrategySignal`。
- [x] 兼容旧策略的 warmup/data buffer 行为。
- [x] 兼容 `datetime` 和毫秒级 `timestamp`。
- [x] 测试 `DualMAStrategy` 可以通过 adapter 产生稳定 `StrategySignal`。
- [x] 测试 signal_id 去重稳定。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/adapters/legacy_strategy_adapter.py`
- `src/application/adapters/__init__.py`
- `tests/test_legacy_strategy_adapter.py`
- `tests/test_domain_trading_pipeline.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 不修改 `DualMAStrategy` 也能进入领域 pipeline。
- DataFrame 不泄漏到 domain pipeline 接口。

完成证据：

- 新增/修改文件：
  - `src/application/adapters/legacy_strategy_adapter.py`
  - `src/application/adapters/__init__.py`
  - `tests/test_legacy_strategy_adapter.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_strategy_adapter`
  - 结果：1 个错误，`ModuleNotFoundError: No module named 'src.application.adapters.legacy_strategy_adapter'`，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_strategy_adapter tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：`Ran 6 tests in 0.790s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 65 tests in 12.350s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_113821.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 14：Legacy Risk Adapter 和领域风控规则

目标：把风控从 DataFrame 过滤迁到 `RiskDecision`。

- [x] 新增 `LegacyRiskPolicyAdapter`。
- [x] 新增 `CompositeRiskPolicy`。
- [x] 新增 `PositionAvailabilityPolicy`，空仓卖出拒绝。
- [x] 新增 `SellQuantityClampPolicy`，超额卖出收缩到当前持仓。
- [x] 兼容旧 `risk_manager.validate_signals(signals_df)`。
- [x] 测试空仓卖出不会进入执行。
- [x] 测试超额卖出会调整到当前持仓。
- [x] 测试 `risk_accepted=False` 不会生成 `OrderIntent`。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/domain/risk_policies.py`
- `src/application/adapters/legacy_risk_policy_adapter.py`
- `tests/test_domain_risk_policies.py`
- `tests/test_legacy_risk_policy_adapter.py`
- `tests/test_domain_trading_pipeline.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 基础持仓约束不再写在 pipeline 内部。
- 风控只通过 `RiskDecision` 影响下游。

完成证据：

- 新增/修改文件：
  - `src/domain/risk_policies.py`
  - `src/application/adapters/legacy_risk_policy_adapter.py`
  - `src/application/adapters/__init__.py`
  - `tests/test_domain_risk_policies.py`
  - `tests/test_legacy_risk_policy_adapter.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_risk_policies tests.test_legacy_risk_policy_adapter tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：`Ran 8 tests in 0.955s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 69 tests in 11.177s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_114257.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 15：ExecutionModel 领域化

目标：执行入口从 DataFrame 信号迁到 `OrderIntent -> Fill`。

- [x] 扩展或新增领域化 `BacktestExecutionModel`。
- [x] 支持 `execute(order_intent, market, portfolio) -> Fill`。
- [x] 保留旧 `ExecutionEngine.execute(signals_df)` 兼容入口。
- [x] 买入使用保守成交价。
- [x] 卖出使用保守成交价。
- [x] 手续费和滑点计算与当前配置一致。
- [x] 测试成交量限制。
- [x] 测试成交后不直接修改 `PortfolioBook`。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/trading/execution/backtest_model.py`
- `src/application/adapters/legacy_execution_adapter.py`
- `tests/test_backtest_execution_model.py`
- `tests/test_legacy_execution_adapter.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 领域执行模型只返回 `Fill`。
- 组合状态只由 `PortfolioBook.apply_fill()` 修改。

完成证据：

- 新增/修改文件：
  - `src/trading/execution/backtest_model.py`
  - `src/application/adapters/legacy_execution_adapter.py`
  - `src/application/adapters/__init__.py`
  - `tests/test_backtest_execution_model.py`
  - `tests/test_legacy_execution_adapter.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model tests.test_legacy_execution_adapter`
  - 结果：4 个错误，`BacktestExecutionModel` 缺少 `execute()`，且 `src.application.adapters.legacy_execution_adapter` 尚不存在，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model tests.test_legacy_execution_adapter`
  - 结果：`Ran 8 tests in 1.235s`，`OK`。
- 边界和兼容聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model tests.test_legacy_execution_adapter tests.test_execution_engine tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：`Ran 14 tests in 6.593s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 73 tests in 12.361s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_115344.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 16：BacktestUseCase 切换到 DomainTradingPipeline

目标：回测路径真正进入领域 pipeline。

- [x] `BacktestUseCase` 组装或接收 `DomainTradingPipeline`。
- [x] 回测循环消费 `MarketDataFeed.load_range()` 输出的 `MarketSlice`。
- [x] 使用 legacy strategy/risk adapter 维持旧策略和旧风控行为。
- [x] 使用领域化 backtest execution model。
- [x] `BacktestTradingMode` 只负责准备依赖和委托 use case。
- [x] 移除 `BacktestUseCase` 对 mode 私有切片方法的依赖。
- [x] 测试回测路径使用 domain pipeline。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/backtest_use_case.py`
- `src/trading/modes/backtest.py`
- `tests/test_backtest_use_case.py`
- `tests/test_backtest_mode.py`
- `tests/test_domain_trading_pipeline.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 基线回测仍然产生 60 笔交易。
- 最终净值保持 `99863.31403628497`，如有差异必须说明原因。
- `BacktestUseCase` 不再调用 mode 私有切片方法。

完成证据：

- 新增/修改文件：
  - `src/application/backtest_use_case.py`
  - `src/trading/modes/backtest.py`
  - `src/application/reporting.py`
  - `src/application/adapters/legacy_strategy_adapter.py`
  - `tests/test_backtest_use_case.py`
  - `tests/test_legacy_strategy_adapter.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_strategy_adapter`
  - 结果：1 个失败，legacy 策略适配器返回了历史信号和当前信号共 2 条，说明切换 domain pipeline 前必须补当前时间过滤和去重。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case`
  - 结果：1 个失败，`BacktestUseCase` 仍调用旧 `_process_market_data`，说明测试抓到回测主路径尚未进入 `DomainTradingPipeline`。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case tests.test_backtest_mode tests.test_legacy_strategy_adapter tests.test_domain_trading_pipeline`
  - 结果：`Ran 18 tests in 10.891s`，`OK`。
- 边界测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries`
  - 结果：`Ran 2 tests in 0.020s`，`OK`；同时 `rg` 未在 `src/domain` 搜到 `pandas`、mode、`ConfigManager` 或 `ExecutionEngine`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 75 tests in 11.046s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_120022.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 17：PaperTradingUseCase 独立化

目标：paper mode 不再自己拥有交易循环，改为 use case + domain pipeline。

- [x] 新增 `PaperTradingUseCase`。
- [x] 新增或适配实时 `MarketDataFeed`。
- [x] 新增 `PaperExecutionModel`。
- [x] `PaperTradingMode` 委托给 use case。
- [x] 保证 paper 不真实下单。
- [x] 测试 paper 一轮循环经过 domain pipeline。
- [x] 运行完整单元测试。
- [x] 视情况运行 paper 聚焦集成测试。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/paper_trading_use_case.py`
- `src/trading/modes/paper.py`
- `src/trading/execution/paper_model.py`
- `tests/test_paper_trading_use_case.py`
- `tests/test_paper_mode.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- paper 和 backtest 共用同一个 `DomainTradingPipeline`。
- paper execution 只返回模拟 `Fill`。

完成证据：

- 新增/修改文件：
  - `src/application/paper_trading_use_case.py`
  - `src/trading/execution/paper_model.py`
  - `src/datasource/feeds/market_data_feed.py`
  - `src/trading/modes/paper.py`
  - `tests/test_paper_execution_model.py`
  - `tests/test_paper_trading_use_case.py`
  - `tests/test_paper_mode.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_execution_model tests.test_paper_trading_use_case`
  - 结果：3 个错误，`src.trading.execution.paper_model` 和 `src.application.paper_trading_use_case` 尚不存在，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_execution_model tests.test_paper_trading_use_case tests.test_paper_mode`
  - 结果：`Ran 5 tests in 8.443s`，`OK`。
- 相关模块测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_execution_model tests.test_paper_trading_use_case tests.test_paper_mode tests.test_backtest_use_case tests.test_backtest_mode tests.test_domain_trading_pipeline tests.test_market_data_feed tests.test_domain_boundaries`
  - 结果：`Ran 26 tests in 11.504s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 79 tests in 12.145s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_120737.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 18：LiveTradingUseCase 独立化

目标：live mode 复用领域 pipeline，只替换实时 feed 和真实 execution model。

- [x] 新增 `LiveTradingUseCase`。
- [x] 新增 `LiveExecutionModel`。
- [x] 将交易所回报转成 `Fill`。
- [x] 保留 live 双安全门。
- [x] `LiveTradingMode` 委托给 use case。
- [x] 使用 fake exchange 测试 live use case。
- [x] 测试未确认 live 安全开关时不会创建真实 execution model。
- [x] 运行完整单元测试。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/live_trading_use_case.py`
- `src/trading/modes/live.py`
- `src/trading/execution/live_model.py`
- `tests/test_live_trading_use_case.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- live 不绕过 domain pipeline。
- live 安全门仍在创建真实执行模型前生效。

完成证据：

- 新增/修改文件：
  - `src/application/live_trading_use_case.py`
  - `src/trading/execution/live_model.py`
  - `src/trading/modes/live.py`
  - `tests/test_live_execution_model.py`
  - `tests/test_live_trading_use_case.py`
  - `tests/test_live_mode.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_execution_model tests.test_live_trading_use_case`
  - 结果：2 个错误，`src.trading.execution.live_model` 和 `src.application.live_trading_use_case` 尚不存在，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_execution_model tests.test_live_trading_use_case tests.test_live_mode`
  - 结果：`Ran 7 tests in 6.938s`，`OK`。
- 相关模块测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_execution_model tests.test_live_trading_use_case tests.test_live_mode tests.test_paper_execution_model tests.test_paper_trading_use_case tests.test_paper_mode tests.test_backtest_use_case tests.test_backtest_mode tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：`Ran 29 tests in 9.170s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 83 tests in 10.570s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_121249.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 19：Reporter 端口化

目标：报告不再读取 mode/context 状态。

- [x] 新增或扩展 `Reporter` 实现。
- [x] reporter 记录 `PortfolioSnapshot`。
- [x] reporter 记录 `Fill`。
- [x] reporter 生成当前兼容报告字段。
- [x] `TradingReportUseCase` 改为读取 reporter，或降级为兼容 wrapper。
- [x] 保持当前 JSON 字段和文件名兼容。
- [x] 测试报告生成不需要 mode.state。
- [x] 测试报告仍写入 `reports/backtest`。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/report_use_case.py`
- `src/reporting/` 或 `src/application/reporting/`
- `tests/test_report_use_case.py`
- `tests/test_reporter.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- 报告不再依赖 mode 私有状态。
- 报告字段与当前报告兼容。

完成证据：

- 新增/修改文件：
  - `src/application/reporting.py`
  - `src/application/report_use_case.py`
  - `tests/test_reporter.py`
  - `tests/test_report_use_case.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_reporter tests.test_report_use_case`
  - 结果：最初失败，`InMemoryReporter` 尚无 `generate_report()`，且 `TradingReportUseCase` 仍依赖 `_calculate_equity()`，符合 Phase 19 预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_reporter tests.test_report_use_case`
  - 结果：`Ran 4 tests in 1.154s`，`OK`。
- 相关模块测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_reporter tests.test_report_use_case tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode tests.test_domain_trading_pipeline`
  - 结果：`Ran 22 tests in 7.828s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 85 tests in 9.420s`，`OK`。
- 边界检查：
  - 命令：`rg -n "import pandas|from pandas|trading\.modes|ConfigManager|ExecutionEngine" src\domain`
  - 结果：无匹配，domain 边界仍干净。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_1216xx.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 20：Runtime Builder / Component Factory

目标：实现架构图中的运行时组装层。

- [x] 新增 `RuntimeBuilder`。
- [x] 根据 config 创建 `MarketDataFeed`。
- [x] 根据 config 创建 `StrategyPort`。
- [x] 根据 config 创建 `RiskPolicy`。
- [x] 根据 mode 创建 `ExecutionModel`。
- [x] 创建 `PortfolioBook`。
- [x] 创建 `Reporter`。
- [x] 创建 backtest/paper/live use case。
- [x] mode 改为调用 builder，减少手工组装。
- [x] 测试 builder 可以组装 backtest 运行时。
- [x] 测试 builder 不会在未确认 live 时创建真实 live execution。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/runtime_builder.py`
- `src/trading/modes/backtest.py`
- `src/trading/modes/paper.py`
- `src/trading/modes/live.py`
- `tests/test_runtime_builder.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

预期验证：

- mode 文件明显变薄。
- backtest/paper/live 的依赖组装入口统一。
- CLI 使用方式不变。

完成证据：

- 新增/修改文件：
  - `src/application/runtime_builder.py`
  - `src/trading/modes/backtest.py`
  - `src/trading/modes/paper.py`
  - `src/trading/modes/live.py`
  - `tests/test_runtime_builder.py`
  - `docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder`
  - 结果：4 个错误，`ModuleNotFoundError: No module named 'src.application.runtime_builder'`，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder`
  - 结果：`Ran 4 tests in 0.171s`，`OK`。
- mode 兼容聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode`
  - 结果：`Ran 20 tests in 5.957s`，`OK`。
- 相关模块测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_backtest_use_case tests.test_backtest_mode tests.test_paper_trading_use_case tests.test_paper_mode tests.test_live_trading_use_case tests.test_live_mode tests.test_domain_trading_pipeline tests.test_domain_boundaries`
  - 结果：`Ran 30 tests in 8.989s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 89 tests in 9.016s`，`OK`；输出中有 1 条 asyncio 慢回调提示，不影响测试结果。
- 边界检查：
  - 命令：`rg -n "import pandas|from pandas|trading\.modes|ConfigManager|ExecutionEngine" src\domain`
  - 结果：无匹配，domain 边界仍干净。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_122627.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`，剩余现金 `99863.31403628497`。

## 进度记录

### 2026-06-09

- 创建完整解耦设计文档：`docs/design/2026-06-09-trading-pipeline-full-decoupling-design.md`。
- 创建完整解耦实施清单：`docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`。
- 审查 Phase 11-20 方案后补充三个关键约束：import boundary tests、async/sync port 约定、DataFrame 只允许留在 legacy adapter 和基础设施边界。
- 完成 Phase 11：新增 domain import boundary tests，将 DataFrame 兼容转换从 `src/domain/adapters.py` 迁到 `src/application/adapters/dataframe_domain_adapter.py`，并更新 ports 为完整领域协议；基线回测保持最终净值 `99863.31403628497` 和总交易数 `60`。
- 完成 Phase 12：新增纯 `DomainTradingPipeline`，用 fake strategy/risk/execution/reporter 验证领域链路可以从 `MarketSlice` 跑到 `Fill` 和 `PortfolioSnapshot`；domain 边界测试确认没有 DataFrame 泄漏。
- 完成 Phase 13：新增 `LegacyDataFrameStrategyAdapter`，将旧 DataFrame 策略包装成 `StrategyPort`，并验证 `DualMAStrategy` 无需重写即可输出领域 `StrategySignal`。
- 完成 Phase 14：新增领域风控规则和 `LegacyRiskPolicyAdapter`，将空仓卖出、超额卖出、legacy risk reject 转成 `RiskDecision`；domain 边界仍无 DataFrame 泄漏；基线回测不变。
- 完成 Phase 15：`BacktestExecutionModel` 新增领域执行入口 `execute(order_intent, market, portfolio) -> Fill`，复用旧回测成交价、滑点、手续费和成交量限制逻辑；新增 `LegacyExecutionAdapter` 作为旧 DataFrame 执行入口到领域 `Fill` 的兼容桥；旧 CLI 基线回测不变。
- 完成 Phase 16：`BacktestUseCase` 在 `domain_pipeline` 存在时直接消费 `MarketDataFeed.load_range()` 的 `MarketSlice` 并调用 `DomainTradingPipeline`；`BacktestTradingMode` 负责组装 legacy strategy/risk adapter、领域回测执行模型、`PortfolioBook` 和 `InMemoryReporter`；旧 CLI 基线回测结果不变。
- 完成 Phase 17：新增 `PaperTradingUseCase`、`PaperExecutionModel` 和 `RealtimeMarketDataFeed.latest()`；`PaperTradingMode` 现在委托 use case 跑同一个 `DomainTradingPipeline`，paper execution 只返回模拟 `Fill`，不会创建真实订单；旧 CLI 基线回测结果不变。
- 完成 Phase 18：新增 `LiveTradingUseCase` 和 `LiveExecutionModel`，将交易所回报转换成领域 `Fill`；`LiveTradingMode` 在安全门通过后组装 live domain pipeline，并在循环中委托 use case；未启用/未确认 live 时仍不会创建 `ExecutionEngine` 或真实下单能力；旧 CLI 基线回测结果不变。
- 完成 Phase 19：`InMemoryReporter` 扩展为正式 reporter 端口实现，记录 `Fill` 和 `PortfolioSnapshot` 后可直接生成兼容报告字段；`TradingReportUseCase` 优先读取 `domain_reporter`，没有领域 reporter 时保留旧 `mode.state` 兼容路径；旧 CLI 基线回测结果不变。
- 完成 Phase 20：新增 `RuntimeBuilder` 和 `TradingRuntime`，统一组装 historical/realtime feed、legacy strategy adapter、组合风控策略、backtest/paper/live execution model、`InMemoryReporter` 和 use case；backtest、paper、live mode 改为调用 builder，保留少量兼容薄入口；旧 CLI 基线回测结果不变。
- Phase 11-20 已完成，当前架构已达到本 worklist 定义的完整解耦标准。后续建议另开清单处理 legacy `src/application/trading_pipeline.py` 的删除条件、CLI 输出过长和 performance timezone warning。
