# 最终四层解耦收敛实施清单

日期：2026-06-09

关联文档：

- 设计文档：`docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- 实施计划：`docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
- 历史完整解耦清单：`docs/worklist/2026-06-09-trading-pipeline-full-decoupling-worklist.md`

## 状态说明

- `[ ]` 未开始
- `[~]` 进行中
- `[x]` 已完成
- `[!]` 已阻塞

## Worklist 更新规则

每完成一个任务，必须同步更新本文档：

1. 将任务状态改为 `[x]`，如果中途停下则改为 `[~]`。
2. 在该任务的“完成证据”中记录修改文件。
3. 记录聚焦测试命令和结果。
4. 记录完整测试命令和结果。
5. 记录基线回测命令、报告路径和关键数值。
6. 如果基线数值变化，必须写明旧值、新值和原因。
7. 更新“当前断点”，写清楚下一次从哪个任务、哪个文件继续。

## 当前断点

- 当前阶段：Task 1 到 Task 52 已完成。旧 DataFrame pipeline 的可迁移行为已映射到 adapter/domain risk/risk adapter 覆盖；`src/application/trading_pipeline.py` 已删除，`BaseTradingMode._process_market_data` 已删除，`tests/test_trading_pipeline.py` 已删除；旧 `ExecutionEngine._backtest_execution` 的 fractional quantity/volume 行为已映射到 `BacktestExecutionModel.execute_orders()` 覆盖；`tests/test_execution_engine.py` 已删除；`LegacyExecutionAdapter` 已删除；`RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`，live runtime 只接受显式 `mode.exchange_client`；backtest/paper/live mode 已删除旧 `_create_legacy_execution_engine()` factory 且不再引用 `src.trading.execution.manager`；`BaseTradingMode` 不再声明 `self.execution_engine`，backtest/paper/live mode 不再读取或关闭 `self.execution_engine`；`src/trading/execution/manager.py` 已删除；旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除；执行侧 legacy 删除 blocker 已清零；未声明 `strategy.interface` 的配置默认进入 domain runtime；`src.application.adapters` 不再聚合导出 legacy strategy/risk adapter，显式 legacy 兼容只能 direct import；回测报告已加入 `quality` 摘要，并扩展到重复时间戳、预期间隔、缺口、成交越界和成交约束检测；报告已加入 `costs` 摘要用于滑点/手续费可解释性；已新增 walk-forward 窗口、参数网格规划底座、可注入批量 backtest runner、扫描结果汇总报告，并把 MA/EMA crossover 策略直观图接回 backtest 报告链路；默认 backtest JSON 已新增 `diagnostics` 研究诊断摘要；扫描汇总已支持 `diagnostics.*` 嵌套指标排序并透传 walk-forward window；本地回测研究入口已可组合参数网格、真实 backtest runner、walk-forward 窗口、扫描汇总、JSON summary 落盘和 CLI 薄入口。
- 下一步入口：本地回测研究闭环第一版已达到 `100%`；下一步建议从“研究产物可读性/可复现性”继续，例如给 summary 增加配置快照、报告路径索引、多参数示例脚本，或进入真实策略/风控改良前的实验清单整理。
- 当前主风险：默认 CLI 回测现在是 native `dual_ma` 基线；旧 legacy 基线仍可作为兼容对照，但不再是主配置默认结果。legacy strategy/risk adapter 仍作为显式兼容层保留；执行侧旧 DataFrame pipeline wrapper、`tests/test_execution_engine.py`、`LegacyExecutionAdapter` 和旧 `ExecutionEngine` 本体已删除。
- 旧 DataFrame execution：`tests/test_execution_engine.py`、`LegacyExecutionAdapter` 和 `src/trading/execution/manager.py` 已删除；旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除；执行侧 legacy 删除 blocker 已清零。
- live runtime exchange client：`RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`，live runtime 只接受显式 `mode.exchange_client`。

## 当前基线

- 回测区间：`2025-01-01` 到 `2025-01-02`
- 交易对和周期：`BTC/USDT 1m`
- 策略：`dual_ma`
- 回测执行模型参数：`--backtest-engine ohlcv`
- 最近已知报告：`reports/backtest/backtest_report_20260610_170737.json`
- 最近研究汇总：`reports/research/research_summary_20260610_170742.json`
- 最终净值：`99953.35539553566`
- 总交易数：`27`
- 买入交易数：`11`
- 卖出交易数：`16`
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

## 最终验收标准

- [x] `src/domain` 不依赖 `pandas`、mode、config、具体 strategy/risk/data source/exchange/report writer。
- [x] backtest、paper、live 主路径全部通过 `RuntimeBuilder -> DomainTradingPipeline`。
- [x] 新策略不再必须使用 `pandas.DataFrame` 输入输出。
- [x] 原生 `dual_ma` 不经过 `LegacyDataFrameStrategyAdapter` 即可跑通 domain 路径；主配置默认已切换为 `strategy.interface=domain`。
- [x] 策略不再计算最终 `quantity`。
- [x] `PositionSizer / OrderFactory` 是 native 路径订单数量和 `OrderIntent` 的主要来源。
- [x] 因子计算从 `BaseStrategy` 和具体策略中迁出到 `src/factor`。
- [x] performance/reporting 层只消费 `Fill` 和 `PortfolioSnapshot`，不依赖 mode 私有状态计算指标。
- [x] `src/application/trading_pipeline.py` 被删除、归档或标记为明确 legacy wrapper。
- [x] `ExecutionEngine.execute(signals_df)` 不再是主执行入口。
- [x] CLI 参数命名不会让开发者误以为当前主路径调用旧 `src/backtest/engine.py`。
- [x] 完整测试通过。
- [x] 基线回测通过；若结果变化，已记录原因。

## 实施项

### [x] Task 1：架构边界和 CLI 命名审计

目标：先避免误解和误用，确认 CLI 主路径不使用旧 `BacktestFactory.run_backtest()`。

计划来源：`docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md` 的 Task 1。

预期修改文件：

- `src/launcher.py`
- `tests/test_launcher_cli.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] 默认配置路径指向 `conf/config.yaml`。
- [x] `--backtest-engine` 被解释为兼容执行模型参数，而不是旧大型回测引擎入口。
- [x] 测试能证明 CLI 主路径没有调用旧 `BacktestFactory`。
- [x] 完整测试通过。
- [x] 基线回测通过。

完成证据：

- 修改文件：
  - `src/launcher.py`
  - `tests/test_launcher_cli.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli -v`
  - 结果：`test_default_config_path_points_to_conf_config_yaml` 因 `src.launcher` 缺少 `default_config_path` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli -v`
  - 结果：`Ran 6 tests in 2.718s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 91 tests in 17.248s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_145258.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。
- 下一步：
  - 进入 Task 2：最小 `PositionSizer / OrderFactory`。

### [x] Task 2：最小 PositionSizer 和 OrderFactory

目标：先给订单数量一个策略外归属，避免原生策略继续计算 `quantity`。

计划来源：实施计划 Task 2。

预期修改文件：

- `src/order/__init__.py`
- `src/order/models.py`
- `src/order/sizing.py`
- `src/order/factory.py`
- `src/application/runtime_builder.py`
- `tests/test_order_sizing.py`

验收：

- [x] `FixedNotionalSizer` 可根据价格计算数量。
- [x] sizing 可按现金和手续费限制买入数量。
- [x] `OrderFactory` 可生成 `OrderIntent`。
- [x] 策略可以不提供最终 quantity。
- [x] 完整测试通过。
- [x] 基线回测通过。

完成证据：

- 修改文件：
  - `src/order/__init__.py`
  - `src/order/models.py`
  - `src/order/sizing.py`
  - `src/order/factory.py`
  - `tests/test_order_sizing.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_order_sizing -v`
  - 结果：`ModuleNotFoundError: No module named 'src.order'`，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_order_sizing -v`
  - 结果：`Ran 4 tests in 0.001s`，`OK`。
  - 相关领域测试：`.\.venv\Scripts\python.exe -m unittest tests.test_order_sizing tests.test_domain_trading_pipeline tests.test_domain_risk_policies -v`
  - 结果：`Ran 9 tests in 0.634s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 95 tests in 14.930s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_145735.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。
- 下一步：
  - 进入 Task 3：最小 `FactorView / FactorEngine`。

### [x] Task 3：最小 FactorView 和 FactorEngine

目标：把策略所需的历史窗口和基础 MA 因子从 `BaseStrategy` 迁到 `src/factor`。

计划来源：实施计划 Task 3。

预期修改文件：

- `src/factor/__init__.py`
- `src/factor/models.py`
- `src/factor/engine.py`
- `src/domain/ports.py`
- `src/application/runtime_builder.py`
- `tests/test_factor_engine.py`

验收：

- [x] `FactorEngine` 可以生成基础 MA factor view。
- [x] warmup 不足时不会生成错误信号。
- [x] `src/domain` 仍不 import pandas。
- [x] 策略层有了可迁移到 factor 层的基础窗口能力。
- [x] 完整测试通过。
- [x] 基线回测通过。

完成证据：

- 修改文件：
  - `src/factor/__init__.py`
  - `src/factor/models.py`
  - `src/factor/engine.py`
  - `tests/test_factor_engine.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine -v`
  - 结果：`ModuleNotFoundError: No module named 'src.factor'`，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine tests.test_domain_boundaries -v`
  - 结果：`Ran 5 tests in 0.022s`，`OK`。
  - 边界检查：`rg -n "import pandas|from pandas|trading\.modes|ConfigManager|ExecutionEngine" src\domain`
  - 结果：无匹配，domain 边界仍干净。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 98 tests in 13.040s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_150112.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。
- 下一步：
  - 进入 Task 4：原生领域版 `DualMA Strategy`。

### [x] Task 4：原生领域版 DualMA Strategy

目标：让 `dual_ma` 不经过 `LegacyDataFrameStrategyAdapter` 即可跑通，并且只输出信号，不计算最终 quantity。

计划来源：实施计划 Task 4。

预期修改文件：

- `src/strategy/domain_base.py`
- `src/strategy/implementations/domain_dual_ma.py`
- `src/application/runtime_builder.py`
- `conf/strategy.yaml`
- `tests/test_domain_dual_ma_strategy.py`
- `tests/test_runtime_builder.py`

验收：

- [x] `DomainDualMAStrategy` 不 import pandas。
- [x] `DomainDualMAStrategy` 输出 `StrategySignal`。
- [x] 输出信号不包含最终 `quantity`。
- [x] `RuntimeBuilder` 可以根据 config 选择原生策略或 legacy adapter。
- [x] 原生 `dual_ma` 可跑通基线回测。
- [x] 如基线数值变化，已说明是 sizing 迁移导致还是 bug 修正导致。

完成证据：

- 修改文件：
  - `src/strategy/domain_base.py`
  - `src/strategy/implementations/domain_dual_ma.py`
  - `src/application/runtime_builder.py`
  - `src/order/risk_policy.py`
  - `src/order/__init__.py`
  - `conf/strategy.yaml`
  - `tests/test_domain_dual_ma_strategy.py`
  - `tests/test_runtime_builder.py`
  - `tests/test_order_sizing.py`
  - `tests/test_domain_boundaries.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy tests.test_runtime_builder -v`
  - 结果：`ModuleNotFoundError: No module named 'src.strategy.implementations.domain_dual_ma'`，符合原生策略尚未实现的预期。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder -v`
  - 结果：新增 `test_native_runtime_risk_policy_sizes_signal_without_quantity_metadata` 失败，确认 native 风控链仍被旧 `SellQuantityClampPolicy` 的 `metadata["quantity"]` 依赖阻断。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy -v`
  - 结果：新增“不重复发同方向信号”用例失败，确认原生策略需要记录 MA 关系变化。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy tests.test_runtime_builder tests.test_order_sizing tests.test_domain_trading_pipeline -v`
  - 结果：`Ran 16 tests in 2.144s`，`OK`。
  - 边界测试：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy tests.test_runtime_builder tests.test_order_sizing tests.test_domain_trading_pipeline tests.test_domain_boundaries -v`
  - 结果：`Ran 16 tests in 1.132s`，`OK`，domain 仍不依赖 pandas/order/strategy 等外部层。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 104 tests in 13.032s`，`OK`。
- 基线回测：
  - legacy 默认命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - legacy 报告：`reports/backtest/backtest_report_20260609_152039.json`
  - legacy 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`，与既有基线一致。
  - native domain 验证：临时内存配置设置 `strategy.interface=domain`、`strategy.active=dual_ma`、`short_window=10`、`long_window=30`、`trading.position_sizing.fraction=0.01` 后运行 `TradingCore(...).run_pipeline()`。
  - native 报告：`reports/backtest/backtest_report_20260609_151912.json`
  - native 结果：最终净值 `99911.08496522238`，总交易数 `49`，买入 `20`，卖出 `29`，结束持仓 `{}`。
  - 差异说明：默认 legacy 基线未变化；native 结果不同是预期迁移差异，原生策略不再计算最终 `quantity`，订单数量由 `FixedFractionSizer` 按组合权益比例生成，同时策略按 MA 关系变化去重发信号。
- 下一步：
  - 进入 Task 5：`PerformanceAnalyzer / ReportWriter`。

### [x] Task 5：PerformanceAnalyzer 和 ReportWriter

目标：让报告层只消费事件和快照，指标计算与文件写入分离。

计划来源：实施计划 Task 5。

预期修改文件：

- `src/reporting/__init__.py`
- `src/reporting/performance_analyzer.py`
- `src/reporting/writers.py`
- `src/application/reporting.py`
- `src/application/report_use_case.py`
- `tests/test_performance_analyzer.py`
- `tests/test_report_writer.py`
- `tests/test_report_use_case.py`

验收：

- [x] `PerformanceAnalyzer` 从 fills/snapshots 计算指标。
- [x] `ReportWriter` 独立处理 JSON/CSV 输出。
- [x] `TradingReportUseCase` 优先通过 reporter 的 fills/snapshots 走 analyzer；legacy mode 私有状态只保留为兼容 fallback。
- [x] 报告字段与当前报告兼容。
- [x] 完整测试通过。
- [x] 基线回测报告可正常写入。

完成证据：

- 修改文件：
  - `src/reporting/__init__.py`
  - `src/reporting/performance_analyzer.py`
  - `src/reporting/writers.py`
  - `src/application/report_use_case.py`
  - `tests/test_performance_analyzer.py`
  - `tests/test_report_writer.py`
  - `tests/test_report_use_case.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_writer tests.test_report_use_case -v`
  - 结果：`ModuleNotFoundError: No module named 'src.reporting'`；同时 `test_generate_report_prefers_domain_reporter_without_mode_state` 暴露旧 `domain_reporter.generate_report` 路径仍被直接调用。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_writer tests.test_report_use_case tests.test_reporter -v`
  - 结果：`Ran 6 tests in 1.734s`，`OK`。
  - 相邻测试：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_writer tests.test_report_use_case tests.test_reporter tests.test_backtest_mode tests.test_domain_trading_pipeline -v`
  - 结果：`Ran 17 tests in 10.108s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 106 tests in 16.469s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_152633.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`，与既有基线一致。
- 下一步：
  - 进入 Task 6：清理 legacy 兼容链路。

### [x] Task 6：清理 legacy 兼容链路

目标：从“adapter 兼容”进入“新接口原生运行”，只保留明确标记的 legacy wrapper。

计划来源：实施计划 Task 6。

预期修改文件：

- `src/application/trading_pipeline.py`
- `src/trading/modes/base.py`
- `src/trading/execution/manager.py`
- `src/backtest/engine.py`
- `tests/test_domain_boundaries.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `tests/test_live_mode.py`

验收：

- [x] backtest 主路径不调用 `src/application/trading_pipeline.py`。
- [x] paper 主路径不调用旧 DataFrame pipeline。
- [x] live 主路径不绕过 `DomainTradingPipeline`。
- [x] `BaseTradingMode` 不再暴露旧 DataFrame 执行入口，或只保留明确 legacy wrapper。
- [x] `src/backtest/engine.py` 被标记、移动或归档为 legacy。
- [x] `ExecutionEngine.execute(signals_df)` 不再是主执行入口。
- [x] 完整测试通过。
- [x] 基线回测通过。

完成证据：

- 修改文件：
  - `src/application/backtest_use_case.py`
  - `src/application/trading_pipeline.py`
  - `src/trading/execution/manager.py`
  - `src/backtest/engine.py`
  - `tests/test_backtest_use_case.py`
  - `tests/test_domain_boundaries.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case -v`
  - 结果：新增缺失 `domain_pipeline` 用例失败，旧实现仍尝试 legacy `_load_steps` fallback，报错 `TypeError: 'Mock' object is not iterable`，证明 fallback 仍存在。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries -v`
  - 结果：`src/application/trading_pipeline.py`、`src/trading/execution/manager.py`、`src/backtest/engine.py` 缺少 `LEGACY COMPATIBILITY` 标记，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode tests.test_domain_boundaries tests.test_runtime_builder tests.test_trading_pipeline -v`
  - 结果：`Ran 32 tests in 10.654s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 106 tests in 16.172s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_153816.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`，与既有基线一致。
- 下一步：
  - 本轮六项任务已完成；下一步只需最终自检和验收。如继续深化，可评估是否将 `strategy.interface=domain` 设为默认。

### [x] Task 7：原生 multi_factors 和可配置 FactorEngine

目标：让 `multi_factors` 也能在 `strategy.interface=domain` 下不经过 `LegacyDataFrameStrategyAdapter` 进入 `DomainTradingPipeline`，并把多因子窗口和计算迁到 `src/factor`。

计划来源：实施计划 Task 7。

预期修改文件：

- `src/factor/engine.py`
- `src/factor/models.py`
- `src/strategy/implementations/domain_multi_factors.py`
- `src/application/runtime_builder.py`
- `tests/test_factor_engine_multi_factors.py`
- `tests/test_domain_multi_factors_strategy.py`
- `tests/test_runtime_builder.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `FactorEngine.from_config()` 可按配置计算 `rsi`、`macd`、`bollinger`、`volume_osc` 和 `composite_signal`。
- [x] 因子 warmup 不足时不会输出误导性信号。
- [x] `DomainMultiFactorsStrategy` 不 import pandas。
- [x] `DomainMultiFactorsStrategy` 只输出 `StrategySignal`，不输出最终 `quantity`。
- [x] `RuntimeBuilder` 可在 `strategy.interface=domain` 且 `strategy.active=multi_factors` 时选择原生策略。
- [x] 旧 `dual_ma` 基线回测保持不变。
- [x] 完整测试通过。

完成证据：

- 修改文件：
  - `src/factor/engine.py`
  - `src/strategy/implementations/domain_multi_factors.py`
  - `src/application/runtime_builder.py`
  - `tests/test_factor_engine_multi_factors.py`
  - `tests/test_domain_multi_factors_strategy.py`
  - `tests/test_runtime_builder.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine_multi_factors -v`
  - 结果：`FactorEngine` 缺少 `from_config`，3 个用例均因 `AttributeError: type object 'FactorEngine' has no attribute 'from_config'` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine_multi_factors tests.test_factor_engine -v`
  - 结果：`Ran 6 tests in 0.003s`，`OK`。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_multi_factors_strategy -v`
  - 结果：`ModuleNotFoundError: No module named 'src.strategy.implementations.domain_multi_factors'`，符合原生策略尚未实现的预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_multi_factors_strategy tests.test_domain_boundaries -v`
  - 结果：`Ran 6 tests in 0.557s`，`OK`。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder -v`
  - 结果：新增 native `multi_factors` builder 用例因 `ValueError: Unsupported domain strategy: multi_factors` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_domain_multi_factors_strategy tests.test_factor_engine_multi_factors tests.test_domain_dual_ma_strategy tests.test_order_sizing -v`
  - 结果：`Ran 21 tests in 1.790s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 113 tests in 16.138s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_160237.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`，与既有基线一致。
- 下一步：
  - Task 7 已完成；下一步可评估是否将 `strategy.interface` 默认值切换为 `domain`，或继续迁移/删除旧 legacy 兼容层。

### [x] Task 8：主配置默认使用 domain 策略接口

目标：让 `conf/config.yaml` 的默认运行路径使用 `strategy.interface=domain`，但保留 `RuntimeBuilder` 代码层 `default="legacy"`，保护未声明 interface 的旧配置。

计划来源：实施计划 Task 8。

预期修改文件：

- `conf/config.yaml`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- `tests/test_config_defaults.py`
- `tests/test_runtime_builder.py`

验收：

- [x] `conf/config.yaml` 明确写入 `strategy.interface: domain`。
- [x] `conf/config.yaml` 明确写入 `strategy.active: dual_ma`。
- [x] 未声明 `strategy.interface` 的旧配置仍通过 `RuntimeBuilder` 回落 legacy adapter。
- [x] 默认 CLI 回测进入 native domain `dual_ma` 路径。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过；若结果与旧 legacy 基线不同，记录差异原因。

完成证据：

- 修改文件：
  - `conf/config.yaml`
  - `tests/test_config_defaults.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_config_defaults tests.test_runtime_builder -v`
  - 结果：`test_main_config_defaults_strategy_interface_to_domain` 因 `KeyError: 'interface'` 失败，证明主配置尚未声明 domain interface。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_config_defaults tests.test_runtime_builder tests.test_domain_dual_ma_strategy tests.test_domain_multi_factors_strategy -v`
  - 结果：`Ran 14 tests in 2.025s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 114 tests in 22.558s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_161000.json`
  - native 默认结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 旧 legacy 对照：最近报告 `reports/backtest/backtest_report_20260609_160237.json`，最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。
  - 差异说明：主配置从 `strategy.interface=legacy` 等效路径切到 `strategy.interface=domain` 后，默认回测使用原生 `DomainDualMAStrategy`、`FactorEngine` 和 native sizing/risk path；原生策略按 MA 关系变化去重，且订单数量由 sizing policy 生成，因此交易次数和净值与旧 DataFrame 策略基线不同，属于预期迁移差异。
- 下一步：
  - Task 8 已完成；下一步可评估删除 legacy 兼容层，或继续完善 native `multi_factors` 的实盘/回测样例配置。

### [x] Task 9：legacy 兼容层删除就绪度门禁

目标：确认当前主配置默认 backtest runtime 已经不依赖 legacy strategy/risk adapter，同时把暂不删除 legacy 模块的阻塞项写成文档和测试门禁。

计划来源：实施计划 Task 9。

预期修改文件：

- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- `tests/test_legacy_deletion_readiness.py`

验收：

- [x] 主配置 `conf/config.yaml` 组装出的 backtest runtime 使用 `DomainDualMAStrategy`，不使用 `LegacyDataFrameStrategyAdapter`。
- [x] 主配置 native risk policy 不包含 `LegacyRiskPolicyAdapter`。
- [x] 设计文档和 worklist 明确记录 legacy 删除暂缓原因。
- [x] 暂缓原因至少覆盖：未声明 `strategy.interface` 的旧配置 fallback、`BaseTradingMode._process_market_data`、旧 DataFrame pipeline 测试、live `ExecutionEngine` exchange client ownership。
- [x] 本任务不删除 legacy 模块；后续删除必须先迁移对应兼容测试或 ownership。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`test_legacy_deletion_blockers_are_documented_before_modules_are_removed` 因设计/worklist 尚未包含 `legacy 删除暂缓`、未声明 `strategy.interface`、`BaseTradingMode._process_market_data`、`ExecutionEngine` 等删除阻塞项失败；`test_main_config_backtest_runtime_uses_native_strategy_and_risk_policy` 已通过，证明生产代码主路径无需改动。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 2 tests in 0.230s`，`OK`。
- 相关边界测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_config_defaults tests.test_runtime_builder tests.test_domain_boundaries -v`
  - 结果：`Ran 13 tests in 0.959s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 116 tests in 15.992s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_162437.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务未修改生产代码，结果与 Task 8 native 默认基线一致。
- 下一步：
  - Task 9 已完成；下一轮优先评估 live exchange client ownership 是否能从旧 `ExecutionEngine` 迁出，或迁移旧 DataFrame pipeline 测试后删除 `BaseTradingMode._process_market_data`。

### [x] Task 10：live exchange client ownership 迁出旧 ExecutionEngine

目标：让 live domain runtime 不再必须从旧 `ExecutionEngine.binance` 获取交易所 client；`RuntimeBuilder` 优先使用 `mode.exchange_client`，旧 `ExecutionEngine` 仅保留为兼容 fallback。

计划来源：Task 9 删除门禁的下一步 blocker。

预期修改文件：

- `src/application/runtime_builder.py`
- `src/trading/modes/live.py`
- `tests/test_runtime_builder.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `RuntimeBuilder.build_live_runtime()` 在 `mode.exchange_client` 存在时使用该 client，不需要 `mode.execution_engine.binance`。
- [x] 旧 `mode.execution_engine.binance` fallback 仍保留，保护旧调用方。
- [x] `LiveTradingMode.initialize()` 将 live exchange client 暴露为 `self.exchange_client`。
- [x] live 安全门、账号校验和 shutdown 行为不在本任务重写。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/application/runtime_builder.py`
  - `src/trading/modes/live.py`
  - `tests/test_runtime_builder.py`
  - `tests/test_live_mode.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_live_mode -v`
  - 结果：新增 `test_build_live_runtime_prefers_mode_exchange_client_over_legacy_execution_engine` 因 `RuntimeBuilder` 仍只读取旧 `execution_engine.binance` 报 `ValueError: Live execution requires an exchange client`；新增 live 初始化断言因 `LiveTradingMode` 尚无 `exchange_client` 属性报 `AttributeError`。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_live_mode -v`
  - 结果：`Ran 13 tests in 7.021s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_live_mode tests.test_live_execution_model tests.test_execution_engine tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 18 tests in 6.583s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 117 tests in 13.920s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_163229.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只影响 live runtime 获取 exchange client 的 ownership，默认 backtest native 基线不变。
- 下一步：
  - Task 10 已完成；继续评估是否把 live 账号校验/关闭也迁出旧 `ExecutionEngine`，再决定 `src/trading/execution/manager.py` 能否移动或删除。

### [x] Task 11：live 账号校验和订单关闭优先使用 exchange_client

目标：让 live mode 的账号余额检查、状态更新、open orders 查询和撤单不再必须依赖旧 `ExecutionEngine`；优先调用 `mode.exchange_client` 的账号/订单能力，旧 `execution_engine` 仅作为兼容 fallback。

计划来源：Task 10 后续 blocker。

预期修改文件：

- `src/trading/modes/live.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `_verify_account()` 优先通过 `exchange_client` 获取余额。
- [x] `_update_account_status()` 优先通过 `exchange_client` 获取余额。
- [x] `shutdown()` 优先通过 `exchange_client` 查询 open orders 并撤单。
- [x] 旧 `execution_engine` fallback 保留，避免旧调用方断裂。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_live_mode.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：新增 3 个用例失败，`exchange_client.get_account_balance/get_open_orders/cancel_order` 未被调用，说明 live mode 仍直接依赖旧 `execution_engine`。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：`Ran 8 tests in 7.555s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model tests.test_execution_engine tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 21 tests in 8.653s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 120 tests in 12.517s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_163858.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只影响 live 账号/订单生命周期调用入口，默认 backtest native 基线不变。
- 下一步：
  - Task 11 已完成；继续评估 `ExecutionEngine` 是否只剩 legacy DataFrame 执行和旧测试使用。

### [x] Task 12：backtest/paper 旧 ExecutionEngine 懒加载

目标：backtest 和 paper 主路径已经通过 `RuntimeBuilder -> DomainTradingPipeline` 使用领域执行模型，不再在 initialize 阶段主动创建旧 `ExecutionEngine`；旧 engine 仅在 legacy DataFrame 入口需要时懒加载。

计划来源：Task 11 后续旧 execution manager 收敛。

预期修改文件：

- `src/trading/modes/base.py`
- `src/trading/modes/backtest.py`
- `src/trading/modes/paper.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BacktestTradingMode.initialize()` 不主动实例化旧 `ExecutionEngine`。
- [x] `PaperTradingMode.initialize()` 不主动实例化旧 `ExecutionEngine`。
- [x] 旧 `_process_market_data` 路径仍可在需要时懒加载旧 engine。
- [x] backtest prepare 阶段只有旧 engine 已存在时才写入 historical data。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/trading/modes/base.py`
  - `src/trading/modes/backtest.py`
  - `src/trading/modes/paper.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_paper_mode.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_paper_mode -v`
  - 结果：新增 backtest/paper initialize 不创建旧 `ExecutionEngine` 的用例因 initialize 仍实例化旧 engine 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_paper_mode -v`
  - 结果：`Ran 13 tests in 8.542s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_paper_mode tests.test_trading_pipeline tests.test_execution_engine tests.test_runtime_builder tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 29 tests in 9.317s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 122 tests in 13.248s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_164946.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只迁移 backtest/paper initialize 阶段的旧 engine ownership，默认 backtest native 基线不变。
- 下一步：
  - Task 13 审计旧 `src/trading/execution/manager.py` 的剩余引用，并把它限定为 legacy DataFrame/live fallback/test 边界。

### [x] Task 13：旧 ExecutionEngine 剩余引用边界门禁

目标：确认 `src/trading/execution/manager.py` 现在只作为 legacy DataFrame 执行、live fallback 或旧测试对象存在，避免新的主路径重新把 `ExecutionEngine.execute(signals_df)` 当成默认入口。

计划来源：Task 12 完成后旧 execution manager 收敛。

预期修改文件：

- `tests/test_legacy_execution_boundary.py`
- `src/trading/modes/live.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `src/trading/execution/manager.py` 文件头明确标记为 `LEGACY COMPATIBILITY`。
- [x] 测试列出 `ExecutionEngine` 允许的生产代码引用点，新增主路径引用会失败。
- [x] `RuntimeBuilder`、domain pipeline、order/reporting/factor/domain 层不 import 旧 `ExecutionEngine`。
- [x] live initialize 在存在显式 `exchange_client` 时不创建旧 `ExecutionEngine`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_live_mode.py`
  - `src/trading/modes/live.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode -v`
  - 结果：新增 `test_initialize_prefers_explicit_exchange_client_without_legacy_engine` 因 live initialize 仍无条件创建旧 `ExecutionEngine` 失败，边界测试本身已通过。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode -v`
  - 结果：`Ran 11 tests in 6.698s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model tests.test_execution_engine tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 24 tests in 7.291s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 125 tests in 13.484s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_165834.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只收窄旧 `ExecutionEngine` 边界和 live 显式 exchange client 优先级，默认 backtest native 基线不变。
- 下一步：
  - Task 14 收窄 `tests/test_execution_engine.py` 的命名和说明，把它明确为 legacy DataFrame execution 测试，避免未来误以为旧 `ExecutionEngine.execute(signals_df)` 是主执行入口。

### [x] Task 14：旧 ExecutionEngine 测试命名和范围收窄

目标：将仍覆盖旧 `ExecutionEngine.execute(signals_df)` 的测试命名为 legacy 行为测试，并补充门禁，避免测试名和类名暗示旧执行器仍是主执行模型。

计划来源：Task 13 完成后的 legacy execution manager 收敛。

预期修改文件：

- `tests/test_execution_engine.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] 旧执行器测试类名或文件内容明确包含 legacy/DataFrame 语义。
- [x] 测试只覆盖旧 DataFrame backtest execution 的兼容行为，不新增主路径入口依赖。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_execution_engine.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_execution_engine -v`
  - 结果：新增 `test_execution_engine_tests_are_named_as_legacy_dataframe_coverage` 因 `tests/test_execution_engine.py` 缺少 legacy DataFrame docstring 和类名失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_execution_engine -v`
  - 结果：`Ran 5 tests in 5.157s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_execution_engine tests.test_legacy_execution_adapter tests.test_runtime_builder tests.test_domain_boundaries -v`
  - 结果：`Ran 17 tests in 10.339s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 126 tests in 22.719s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_170320.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只收窄旧执行器测试命名和边界说明，默认 backtest native 基线不变。
- 下一步：
  - Task 15 评估并最小迁移 live exchange client 的默认构造，目标是让 `LiveTradingMode.initialize()` 不再需要通过旧 `ExecutionEngine` 创建 exchange client。

### [x] Task 15：live exchange client 默认构造迁出旧 ExecutionEngine

目标：在不扩大交易所 adapter API 设计的前提下，给 live mode 一个独立的 exchange client 构造入口；显式注入优先，其次独立工厂，旧 `ExecutionEngine` 只保留为最后 fallback。

计划来源：Task 14 完成后的 live fallback 收敛。

预期修改文件：

- `src/trading/modes/live.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `LiveTradingMode.initialize()` 在没有显式 `exchange_client` 时优先调用 `_create_exchange_client()`。
- [x] `_create_exchange_client()` 默认返回 `Binance(config)` 或等价 exchange adapter，不经旧 `ExecutionEngine`。
- [x] 旧 `ExecutionEngine` live fallback 仍可保留，但只在独立 client 构造失败或显式不可用时使用。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_live_mode.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：新增 `test_initialize_creates_exchange_client_without_legacy_engine_by_default` 因 live initialize 仍创建旧 `ExecutionEngine` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：`Ran 13 tests in 8.417s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model tests.test_legacy_execution_boundary tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 27 tests in 8.178s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 130 tests in 13.436s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_171056.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只迁移 live exchange client 的默认构造，默认 backtest native 基线不变。
- 下一步：
  - Task 16 给独立 `Binance` adapter 补齐 live 路径需要的 `create_order/get_account_balance/get_open_orders/cancel_order` 薄方法，进一步减少旧 live fallback。

### [x] Task 16：独立 exchange adapter 补齐 live 薄方法

目标：让 live 主路径默认构造出来的 `Binance` adapter 能直接承担下单、余额查询、open orders 查询和撤单能力，不再因为这些方法缺失而依赖旧 `ExecutionEngine`。

计划来源：Task 15 完成后的 live fallback 收敛。

预期修改文件：

- `src/exchange/adapters/binance.py`
- `tests/test_binance_exchange_adapter.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `Binance.create_order()` 代理到底层 ccxt exchange 的 `create_order()`。
- [x] `Binance.get_account_balance()` 可从 `fetch_balance()` 返回可求和的 total balance。
- [x] `Binance.get_open_orders()` 代理到底层 exchange 的 `fetch_open_orders()`。
- [x] `Binance.cancel_order()` 代理到底层 exchange 的 `cancel_order()`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/exchange/adapters/binance.py`
  - `tests/test_binance_exchange_adapter.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_binance_exchange_adapter -v`
  - 结果：初次运行先因测试夹具未 patch `LogManager` 触发导入错误；补夹具后再次运行，4 个用例因 `Binance` 缺少 `create_order/get_account_balance/get_open_orders/cancel_order` 报 `AttributeError`，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_binance_exchange_adapter -v`
  - 结果：`Ran 4 tests in 6.093s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_binance_exchange_adapter tests.test_live_mode tests.test_live_execution_model tests.test_runtime_builder tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 29 tests in 11.620s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 134 tests in 13.548s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_171656.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只补独立 exchange adapter 的 live 薄接口，默认 backtest native 基线不变。
- 下一步：
  - Task 17 评估是否可以删除 live mode 中旧 `ExecutionEngine` fallback，或至少把 fallback 改成显式 legacy 配置才启用。

### [x] Task 17：live 旧 ExecutionEngine fallback 显式化或删除

目标：在 `Binance` adapter 已具备 live 主路径薄接口后，进一步确认 live mode 是否还需要默认旧 `ExecutionEngine` fallback；若仍保留，必须通过显式 legacy 配置启用。

计划来源：Task 16 完成后的 live fallback 收敛。

预期修改文件：

- `src/trading/modes/live.py`
- `tests/test_live_mode.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] live 默认初始化不再创建旧 `ExecutionEngine` fallback。
- [x] 如保留 fallback，必须由显式 legacy 配置开启。
- [x] 旧测试仍能覆盖 fallback 兼容行为。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- 修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_live_mode.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：新增 `test_initialize_rejects_missing_exchange_client_without_legacy_fallback_enabled` 因默认路径未抛出 `ConnectionError` 失败，证明 live 仍会默认进入旧 `ExecutionEngine` fallback。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：`Ran 14 tests in 8.848s`，`OK`。
- 相关测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model tests.test_legacy_execution_boundary tests.test_binance_exchange_adapter -v`
  - 结果：`Ran 30 tests in 8.403s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 135 tests in 14.143s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_172537.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只把 live 旧 `ExecutionEngine` fallback 改为显式配置 `live_trading.allow_legacy_execution_engine_fallback=true` 才启用，默认 backtest native 基线不变。
- 下一步：
  - Task 18 更新 legacy 删除门禁，把 live 默认 exchange client ownership blocker 从删除阻塞中移除或降级为显式 legacy fallback 兼容项。

### [x] Task 18：更新 legacy 删除门禁和剩余阻塞清单

目标：Task 17 已经让 live 默认路径不再依赖旧 `ExecutionEngine` fallback；本任务复查设计文档、worklist 和 readiness 测试，把 live ownership blocker 改成当前事实，并明确下一批真正阻塞 legacy 删除的项目。

计划来源：Task 17 完成后的 legacy 删除门禁复查。

预期修改文件：

- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- `tests/test_legacy_deletion_readiness.py`

验收：

- [x] legacy 删除暂缓原因不再声称 live 默认路径通过旧 `ExecutionEngine` 持有 exchange client。
- [x] readiness 测试覆盖 live fallback 已显式化这一事实。
- [x] 剩余 blocker 清楚指向旧配置 fallback、旧 DataFrame pipeline/execution 兼容测试和 live 显式 legacy fallback。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

剩余 legacy 删除 blocker：

- 旧配置 fallback：未声明 `strategy.interface` 的配置仍需兼容或迁移。
- 旧 DataFrame pipeline/execution：兼容测试迁移前不能删除 legacy wrapper。
- live 显式 legacy fallback：`live_trading.allow_legacy_execution_engine_fallback=true` 仍需保留测试或迁移策略。

完成证据：
- 修改文件：
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
  - `tests/test_legacy_deletion_readiness.py`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_remaining_legacy_deletion_blockers_are_named_after_live_ownership_migration` 因设计文档和 worklist 缺少“剩余 legacy 删除 blocker”精确清单失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 4 tests in 0.132s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 137 tests in 12.375s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_082532.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只更新 legacy 删除门禁、readiness 测试和文档清单，默认 native backtest 基线不变。
  - 下一步：
    - 按剩余 blocker 选择迁移旧配置 fallback，或迁移旧 DataFrame pipeline/execution 兼容测试；live 显式 legacy fallback 继续作为独立兼容项保留测试或迁移策略。

### [x] Task 19：收紧旧配置 fallback 兼容边界

目标：在不改变生产默认行为的前提下，把未声明 `strategy.interface` 的旧配置 fallback 从“删除阻塞项”收紧成显式 legacy 兼容边界；后续删除前必须先有迁移策略或保留兼容测试。

计划来源：Task 18 完成后的剩余 legacy 删除 blocker 清单。

预期修改文件：

- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- `tests/test_legacy_deletion_readiness.py`

验收：

- [x] readiness 测试要求文档和 worklist 明确旧配置 fallback 保留为显式兼容边界。
- [x] 文档说明未声明 `strategy.interface` 的配置继续回落 legacy adapter，但不再代表主路径默认行为。
- [x] 删除前置条件明确为：提供旧配置迁移策略，或继续保留兼容测试。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- 旧配置 fallback 保留为显式兼容边界。
- 未声明 `strategy.interface` 的配置继续回落 legacy adapter。
- 删除前必须先提供旧配置迁移策略或继续保留兼容测试。

完成证据：
- 修改文件：
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
  - `tests/test_legacy_deletion_readiness.py`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_legacy_config_fallback_is_documented_as_explicit_compatibility_boundary` 因设计文档缺少“旧配置 fallback 保留为显式兼容边界”失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 5 tests in 0.164s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 138 tests in 15.397s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_083054.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只收紧旧配置 fallback 的兼容边界说明和 readiness 门禁，不改变生产默认行为，默认 native backtest 基线不变。
  - 下一步：
    - 进入旧 DataFrame pipeline/execution 兼容测试迁移评估，优先审计 `tests/test_trading_pipeline.py`、`tests/test_execution_engine.py` 和 `src/application/trading_pipeline.py` / `BaseTradingMode` 旧入口。

### [x] Task 20：标记旧 DataFrame pipeline/execution 兼容边界

目标：在不改变旧 DataFrame 行为的前提下，把 `BaseTradingMode` 的 DataFrame market-data/execution 入口、`src/application/trading_pipeline.py` 和旧 `ExecutionEngine` 测试明确标记为 legacy 兼容边界；后续删除前必须先迁移对应兼容测试。

计划来源：Task 19 完成后的剩余 legacy 删除 blocker 清单。

预期修改文件：

- `src/trading/modes/base.py`
- `tests/test_trading_pipeline.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BaseTradingMode._process_market_data` 标记为 `LEGACY COMPATIBILITY: DataFrame market-data wrapper`。
- [x] `BaseTradingMode._execute_signals` 标记为 `LEGACY COMPATIBILITY: DataFrame signal execution wrapper`。
- [x] `tests/test_trading_pipeline.py` 明确是 legacy DataFrame `TradingPipeline` 兼容测试。
- [x] readiness 测试要求文档和 worklist 记录：迁移 `tests/test_trading_pipeline.py` 和 `tests/test_execution_engine.py` 前不能删除 legacy wrapper。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- 旧 DataFrame pipeline/execution 保留为显式兼容边界。
- 迁移 `tests/test_trading_pipeline.py` 和 `tests/test_execution_engine.py` 前不能删除 legacy wrapper。

完成证据：
- 修改文件：
  - `src/trading/modes/base.py`
  - `tests/test_trading_pipeline.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_legacy_dataframe_pipeline_and_execution_boundaries_are_labeled` 先因 `BaseTradingMode` 和 `tests/test_trading_pipeline.py` 缺少 legacy 标识失败；补齐代码/测试标识后，又因 worklist 缺少“旧 DataFrame pipeline/execution 保留为显式兼容边界”精确说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 6 tests in 0.189s`，`OK`。
  - 相邻 legacy 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_trading_pipeline tests.test_execution_engine tests.test_backtest_mode tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 19 tests in 10.319s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 139 tests in 17.020s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_083753.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只标记旧 DataFrame pipeline/execution 兼容边界，不改变生产默认行为，默认 native backtest 基线不变。
- 下一步：
  - 评估旧 DataFrame pipeline/execution 兼容测试是否可迁移为 domain pipeline 覆盖，优先从 `tests/test_trading_pipeline.py` 的行为用例映射到 `tests/test_domain_trading_pipeline.py`；无法映射的继续保留为 legacy-only 测试直到删除策略明确。

### [x] Task 21：迁移旧 DataFrame TradingPipeline 可迁移行为覆盖

目标：把旧 `tests/test_trading_pipeline.py` 中能迁移的行为映射到 domain pipeline/risk policy 覆盖，减少删除旧 DataFrame pipeline 前的未知行为债务。

计划来源：Task 20 完成后的旧 DataFrame pipeline/execution 兼容测试迁移评估。

预期修改文件：

- `src/domain/trading_pipeline.py`
- `tests/test_domain_trading_pipeline.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] domain pipeline 跳过非当前 market timestamp 的信号。
- [x] 旧 `TradingPipeline` 当前时间戳过滤行为已映射到 domain pipeline 覆盖。
- [x] 空仓卖出、超额卖出 clamp、risk reject 映射为既有 domain risk/pipeline 覆盖。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容映射说明：

- 旧 DataFrame `TradingPipeline` 的当前 timestamp 过滤迁移到 `DomainTradingPipeline.run_once()`。
- risk reject 已由 `tests/test_domain_trading_pipeline.py::test_rejected_risk_decision_does_not_execute` 覆盖。
- 空仓卖出已由 `tests/test_domain_risk_policies.py::test_position_availability_rejects_sell_without_position` 覆盖。
- 超额卖出 clamp 已由 `tests/test_domain_risk_policies.py::test_sell_quantity_clamp_limits_sell_to_current_position` 覆盖。
- 旧 DataFrame pipeline/execution 的不可迁移接口形态继续作为 legacy-only 兼容测试保留。

完成证据：
- 修改文件：
  - `src/domain/trading_pipeline.py`
  - `tests/test_domain_trading_pipeline.py`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_trading_pipeline -v`
  - 结果：新增 `test_ignores_signals_that_do_not_match_current_market_timestamp` 因 stale signal 也被执行失败，`AssertionError: 2 != 1`，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_trading_pipeline -v`
  - 结果：`Ran 3 tests in 0.616s`，`OK`。
  - 相邻测试：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_trading_pipeline tests.test_domain_risk_policies tests.test_trading_pipeline tests.test_backtest_use_case -v`
  - 结果：`Ran 13 tests in 1.848s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 140 tests in 16.012s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_084249.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只迁移旧 pipeline 时间戳过滤行为到 domain pipeline，不改变默认 native backtest 基线。
- 下一步：
  - 评估旧 `ExecutionEngine` DataFrame 兼容测试是否可由 `BacktestExecutionModel` 覆盖，优先审计 fractional quantity/volume 行为。

### [x] Task 22：映射旧 ExecutionEngine fractional fill 行为到 BacktestExecutionModel 覆盖

目标：确认旧 `ExecutionEngine._backtest_execution` 中 fractional crypto quantity 和 integer volume fractional decrement 行为已由新 `BacktestExecutionModel` 覆盖，减少旧执行引擎删除 blocker。

计划来源：Task 21 完成后的旧 DataFrame execution 兼容测试迁移评估。

预期修改文件：

- `tests/test_backtest_execution_model.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BacktestExecutionModel.execute_orders()` 覆盖 fractional crypto quantity 保留行为。
- [x] `BacktestExecutionModel.execute_orders()` 覆盖整数 volume 被 fractional fill 扣减行为。
- [x] readiness 测试要求设计文档和 worklist 记录该映射。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容映射说明：

- 旧 `ExecutionEngine._backtest_execution` fractional quantity/volume 行为已映射到 `BacktestExecutionModel.execute_orders()` 覆盖。
- `tests/test_backtest_execution_model.py` 覆盖 fractional crypto quantity 与 integer volume fractional decrement。
- 旧 `tests/test_execution_engine.py` 暂时保留为 legacy DataFrame `ExecutionEngine` 兼容对照，删除前需继续保留或完成明确迁移策略。

完成证据：
- 修改文件：
  - `tests/test_backtest_execution_model.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - 迁移覆盖：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model -v`
  - 结果：`Ran 9 tests in 1.041s`，`OK`。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_execution_engine_fractional_fill_behavior_is_mapped_to_backtest_execution_model` 因设计文档和 worklist 缺少映射说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 7 tests in 0.197s`，`OK`。
  - 相邻 execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_execution_model tests.test_execution_engine tests.test_legacy_execution_boundary tests.test_backtest_use_case -v`
  - 结果：`Ran 17 tests in 3.971s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 143 tests in 10.823s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_084805.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只迁移旧执行引擎 fractional fill 行为覆盖和文档门禁，不改变默认 native backtest 基线。
- 下一步：
  - 继续收敛旧 DataFrame pipeline/execution 删除门禁，评估 `tests/test_execution_engine.py` 是否只保留 legacy 对照，或进一步迁移到 domain `ExecutionModel`/adapter 覆盖。

### [x] Task 23：将旧 backtest ExecutionEngine 收窄为 wrapper-only 边界

目标：确认旧 `ExecutionEngine._backtest_execution()` 不再拥有独立回测成交算法，只作为 `BacktestExecutionModel.execute_orders()` 的委托 wrapper；后续删除或移动旧执行引擎时，成交算法事实来源保持在新执行模型。

计划来源：Task 22 完成后的旧 `ExecutionEngine` 删除门禁继续收敛。

预期修改文件：

- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] 边界测试证明 `ExecutionEngine._backtest_execution()` 委托 `BacktestExecutionModel.execute_orders()`。
- [x] readiness 测试要求设计文档和 worklist 记录 wrapper-only 边界。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- 旧 `ExecutionEngine._backtest_execution` 仅作为 `BacktestExecutionModel.execute_orders()` 委托 wrapper 保留。
- 回测成交算法事实来源是 `BacktestExecutionModel`，不是旧 `ExecutionEngine`。
- `tests/test_execution_engine.py` 继续作为 legacy DataFrame `ExecutionEngine` 兼容对照，不再代表独立成交算法所有权。

完成证据：
- 修改文件：
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - 边界测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 4 tests in 3.519s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_backtest_execution_engine_is_documented_as_wrapper_only` 因设计文档和 worklist 缺少 wrapper-only 说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 8 tests in 0.140s`，`OK`。
  - 相邻 execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_backtest_execution_model tests.test_execution_engine tests.test_legacy_execution_adapter -v`
  - 结果：`Ran 16 tests in 4.087s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 145 tests in 10.865s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_085227.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只收紧旧 backtest `ExecutionEngine` wrapper-only 边界和文档门禁，不改变默认 native backtest 基线。
- 下一步：
  - 继续评估 `LegacyExecutionAdapter` 是否仍有生产引用，或是否只作为旧 `ExecutionEngine.execute(signals_df)` 到 domain `Fill` 的兼容桥保留。

### [x] Task 24：收窄 LegacyExecutionAdapter 为 direct-import-only 兼容桥

目标：确认 `LegacyExecutionAdapter` 没有主路径生产引用，并从 `src.application.adapters` 包级默认导出中移除，避免新 application 代码通过聚合导入误把旧 `ExecutionEngine.execute(signals_df)` 桥接器带回主路径。

计划来源：Task 23 完成后的旧 execution adapter 删除门禁继续收敛。

预期修改文件：

- `src/application/adapters/__init__.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `LegacyExecutionAdapter` 不再从 `src.application.adapters` 包级导出。
- [x] 旧 adapter 自身仍可通过 `src.application.adapters.legacy_execution_adapter` 直接导入并通过兼容测试。
- [x] readiness 测试要求设计文档和 worklist 记录 direct-import-only 边界。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `LegacyExecutionAdapter` 不再从 `src.application.adapters` 包级导出。
- 如需兼容旧 `ExecutionEngine.execute(signals_df)`，只能直接从 `src.application.adapters.legacy_execution_adapter` 导入。
- 主 `RuntimeBuilder` 当前不使用 `LegacyExecutionAdapter`；native backtest/paper/live 分别使用 `BacktestExecutionModel`、`PaperExecutionModel` 和 `LiveExecutionModel`。

完成证据：
- 修改文件：
  - `src/application/adapters/__init__.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary -v`
  - 结果：新增 `test_legacy_execution_adapter_is_not_package_level_export` 因 `src/application/adapters/__init__.py` 仍导出 `LegacyExecutionAdapter` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_legacy_execution_adapter tests.test_runtime_builder -v`
  - 结果：`Ran 14 tests in 3.725s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_legacy_execution_adapter_is_documented_as_direct_import_only` 因设计文档和 worklist 缺少 direct-import-only 说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 9 tests in 0.173s`，`OK`。
  - 相邻 execution/runtime 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_legacy_execution_adapter tests.test_runtime_builder tests.test_domain_boundaries -v`
  - 结果：`Ran 17 tests in 3.938s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 147 tests in 11.200s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_085622.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只收窄 `LegacyExecutionAdapter` 包级导出和文档门禁，不改变默认 native backtest 基线。
- 下一步：
  - 评估 `src/trading/modes/backtest.py` 和 `src/trading/modes/paper.py` 中旧 `ExecutionEngine` import 是否能延迟到 legacy factory 内部，减少模块加载期旧执行引擎可见性。

### [x] Task 25：backtest/paper 旧 ExecutionEngine 延迟导入

目标：把 `BacktestTradingMode` 和 `PaperTradingMode` 对旧 `ExecutionEngine` 的可见性从模块加载期收窄到 legacy factory 内部；默认 initialize 和 native runtime 组装不应暴露或触发旧执行引擎。

计划来源：Task 24 完成后的旧 DataFrame pipeline/execution 删除门禁继续收敛。

预期修改文件：

- `src/trading/modes/backtest.py`
- `src/trading/modes/paper.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BacktestTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- [x] `PaperTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- [x] 旧 `ExecutionEngine` 仅在 `_create_legacy_execution_engine()` 内部按需导入。
- [x] 默认 initialize 不触发 legacy factory。
- [x] readiness 测试要求设计文档和 worklist 记录 lazy-import 边界。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `BacktestTradingMode` 和 `PaperTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- 旧 `ExecutionEngine` 仅在 `_create_legacy_execution_engine()` 内部按需导入。
- 默认 initialize 和 native runtime 组装不会触发 legacy factory；旧 DataFrame 执行入口仍可按需创建旧 engine。

完成证据：
- 修改文件：
  - `src/trading/modes/backtest.py`
  - `src/trading/modes/paper.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_paper_mode.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary -v`
  - 结果：新增 `test_backtest_and_paper_modes_lazy_import_legacy_execution_engine` 因 `backtest.py` 和 `paper.py` 仍顶层 import 旧 `ExecutionEngine` 失败，符合预期。
  - 旧测试调整：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_backtest_mode tests.test_paper_mode -v`
  - 结果：首次运行因 `test_initialize_does_not_create_legacy_execution_engine` 仍 patch 模块顶层 `ExecutionEngine` 报 `AttributeError`；已改为断言模块顶层没有 `ExecutionEngine`，并确认 initialize 不调用 legacy factory。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_backtest_mode tests.test_paper_mode -v`
  - 结果：`Ran 19 tests in 5.790s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_backtest_and_paper_modes_document_legacy_execution_engine_lazy_import` 因设计文档和 worklist 缺少 lazy-import 说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 10 tests in 0.189s`，`OK`。
  - 相邻 mode/execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_backtest_mode tests.test_paper_mode tests.test_execution_engine -v`
  - 结果：`Ran 21 tests in 6.357s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 149 tests in 13.685s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_090114.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只把 backtest/paper 的旧 `ExecutionEngine` import 延迟到 legacy factory 内部，不改变默认 native backtest 基线。
- 下一步：
  - 评估 live mode 的旧 `ExecutionEngine` import 是否还必须保持顶层可见，或是否可在显式 legacy fallback 内部延迟导入。

### [x] Task 26：live 旧 ExecutionEngine 延迟导入

目标：把 `LiveTradingMode` 对旧 `ExecutionEngine` 的可见性从模块加载期收窄到显式 legacy fallback helper 内部；默认 live 初始化、独立 `Binance` adapter 和 native runtime 组装不应暴露旧执行引擎。

计划来源：Task 25 完成后的下一步删除门禁继续收敛。

预期修改文件：

- `src/trading/modes/live.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_live_mode.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `LiveTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- [x] 旧 `ExecutionEngine` 仅在 `_create_legacy_live_execution_engine()` 和 `_create_legacy_execution_engine()` 内部按需导入。
- [x] 默认 live initialize 不触发 legacy factory。
- [x] 显式 `live_trading.allow_legacy_execution_engine_fallback=true` fallback 仍可 patch 旧 `src.trading.execution.manager.ExecutionEngine`。
- [x] readiness 测试要求设计文档和 worklist 记录 live lazy-import 边界。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `LiveTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- 旧 `ExecutionEngine` 仅在 `_create_legacy_live_execution_engine()` 和 `_create_legacy_execution_engine()` 内部按需导入。
- 默认 live 路径优先使用独立 `Binance` adapter；旧 engine 只服务显式 legacy fallback。

完成证据：
- 修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_live_mode.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary -v`
  - 结果：扩展后的 lazy-import 边界测试因 `live.py` 仍顶层 import 旧 `ExecutionEngine` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode -v`
  - 结果：`Ran 20 tests in 9.755s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：新增 `test_live_mode_documents_legacy_execution_engine_lazy_import` 因设计文档和 worklist 缺少 live lazy-import 说明失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 11 tests in 0.136s`，`OK`。
  - 相邻 mode/runtime/execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model -v`
  - 结果：`Ran 29 tests in 8.253s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 150 tests in 13.322s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_093812.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只把 live 的旧 `ExecutionEngine` import 延迟到显式 legacy fallback helper 内部，不改变默认 native backtest 基线。
- 下一步：
  - 评估 `live_trading.allow_legacy_execution_engine_fallback=true` 兼容路径是否仍需保留，或是否可迁移/删除显式 live legacy fallback。

### [x] Task 27：删除 live 显式 legacy ExecutionEngine fallback

目标：在 live 默认路径已经使用独立 `Binance` adapter 后，删除 `live_trading.allow_legacy_execution_engine_fallback=true` 对旧 `ExecutionEngine` 的初始化入口；缺少 exchange client 时直接失败，不再回落旧执行引擎。

计划来源：Task 26 完成后的下一步删除门禁继续收敛。

预期修改文件：

- `src/trading/modes/live.py`
- `tests/test_live_mode.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `live_trading.allow_legacy_execution_engine_fallback=true` 不再创建旧 `ExecutionEngine`。
- [x] `LiveTradingMode.initialize()` 在缺少 exchange client 时直接抛出 `ConnectionError`。
- [x] `_create_legacy_live_execution_engine()` 已删除。
- [x] `LiveTradingMode` 不再顶层 import 旧 `ExecutionEngine`。
- [x] 旧 `ExecutionEngine` 仅在 `_create_legacy_execution_engine()` 内部按需导入，用于旧 DataFrame execution wrapper 兼容入口。
- [x] readiness 测试要求设计文档和 worklist 记录 live fallback 删除事实。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- live 默认路径已删除旧 `ExecutionEngine` fallback。
- `live_trading.allow_legacy_execution_engine_fallback=true` 不再创建旧 `ExecutionEngine`。
- live 显式 legacy fallback 已删除，不再作为剩余 legacy 删除 blocker。
- 剩余 legacy 删除 blocker 收敛为旧配置 fallback 和旧 DataFrame pipeline/execution。

完成证据：
- 修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_live_mode.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：新增 `test_initialize_rejects_missing_exchange_client_even_when_legacy_fallback_flag_enabled` 和 `test_legacy_live_execution_engine_helper_is_removed` 失败，证明 live 显式 fallback 仍会创建旧 engine，且 helper 仍存在。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_live_mode -v`
  - 结果：`Ran 14 tests in 9.434s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：更新后的 readiness 因设计文档和 worklist 尚未记录 live fallback 删除事实失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 11 tests in 0.223s`，`OK`。
  - 相邻 mode/runtime/execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode tests.test_runtime_builder tests.test_live_execution_model -v`
  - 结果：`Ran 29 tests in 10.384s`，`OK`；输出中有 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 150 tests in 19.202s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_094856.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除 live 显式旧 `ExecutionEngine` fallback，不改变默认 native backtest 基线。
- 下一步：
  - 继续收敛旧 DataFrame pipeline/execution 删除门禁，优先评估旧配置 fallback 是否可迁移，或迁移 `tests/test_trading_pipeline.py` / `tests/test_execution_engine.py` 后删除旧 DataFrame wrapper。

### [x] Task 28：删除未声明 strategy.interface 的旧配置 fallback

目标：把 `RuntimeBuilder` 的代码层默认策略接口从 `legacy` 切到 `domain`；未声明 `strategy.interface` 的配置默认进入 native domain runtime，只有显式 `strategy.interface=legacy` 才会启用 legacy adapter。

计划来源：Task 27 完成后的剩余 legacy 删除 blocker 清单。

预期修改文件：

- `src/application/runtime_builder.py`
- `tests/test_runtime_builder.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] 未声明 `strategy.interface` 的配置默认进入 domain runtime。
- [x] `RuntimeBuilder._strategy_interface(default="domain")` 是代码层默认行为。
- [x] 只有显式 `strategy.interface=legacy` 才会启用 legacy adapter。
- [x] readiness 测试要求设计文档和 worklist 记录旧配置 fallback 删除事实。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- 旧配置 fallback 已删除，不再作为剩余 legacy 删除 blocker。
- 未声明 `strategy.interface` 的配置默认进入 domain runtime。
- 只有显式 `strategy.interface=legacy` 才会启用 legacy adapter，继续保护明确声明旧接口的配置和测试。
- 剩余 legacy 删除 blocker 收敛为旧 DataFrame pipeline/execution。

完成证据：
- 修改文件：
  - `src/application/runtime_builder.py`
  - `tests/test_runtime_builder.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder -v`
  - 结果：新增 `test_build_backtest_runtime_defaults_to_native_domain_strategy` 因默认 runtime 仍创建 `LegacyDataFrameStrategyAdapter` 失败，符合预期；显式 `strategy.interface=legacy` 兼容测试通过。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder -v`
  - 结果：`Ran 9 tests in 1.067s`，`OK`。
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：更新后的 readiness 因设计文档和 worklist 尚未记录旧配置 fallback 删除事实失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 11 tests in 0.294s`，`OK`。
  - 相邻测试：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_legacy_strategy_adapter tests.test_legacy_risk_policy_adapter tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 24 tests in 2.097s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 151 tests in 17.204s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_095831.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除未声明 `strategy.interface` 的旧配置 fallback，不改变默认 native backtest 基线。
- 下一步：
  - 继续迁移旧 DataFrame pipeline/execution 删除门禁，优先审计 `tests/test_trading_pipeline.py`、`tests/test_execution_engine.py`、`src/application/trading_pipeline.py`、`BaseTradingMode._process_market_data()` 和 `BaseTradingMode._execute_signals()`。

### [x] Task 29：收窄 BaseTradingMode 的旧 DataFrame pipeline 入口

目标：删除 `BaseTradingMode` 中重复的旧 DataFrame signal/execution helper，只保留 `_process_market_data()` 作为 thin legacy entrypoint 委托 `src/application/trading_pipeline.py`；旧 DataFrame 兼容逻辑集中在 application legacy wrapper。

计划来源：Task 28 完成后的旧 DataFrame pipeline/execution 删除门禁继续收敛。

预期修改文件：

- `src/trading/modes/base.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BaseTradingMode._process_market_data` 只保留 thin legacy entrypoint，并委托到 `src/application/trading_pipeline.py`。
- [x] `BaseTradingMode` 不再定义旧 DataFrame `_prepare_signals_for_execution`、`_filter_executable_signals`、`_update_timestamp_from_data`、`_update_market_prices`、`_execute_signals`、`_process_executed_order`。
- [x] 旧 DataFrame signal 过滤、持仓约束、执行和成交入账兼容逻辑仍由 `src/application/trading_pipeline.py` 与 `tests/test_trading_pipeline.py` 覆盖。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `BaseTradingMode._process_market_data` 只保留 thin legacy entrypoint，并委托到 `src/application/trading_pipeline.py`。
- DataFrame signal/execution helper 已从 mode 层移除，旧兼容行为集中在 application legacy wrapper。
- 剩余 legacy 删除 blocker 仍是 `src/application/trading_pipeline.py` 和旧 `ExecutionEngine.execute(signals_df)` 的兼容测试迁移。

完成证据：
- 修改文件：
  - `src/trading/modes/base.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_base_mode_keeps_only_thin_legacy_dataframe_pipeline_entrypoint -v`
  - 结果：因 `BaseTradingMode` 仍定义 `_prepare_signals_for_execution` 等旧 DataFrame helper 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_base_mode_keeps_only_thin_legacy_dataframe_pipeline_entrypoint -v`
  - 结果：`Ran 1 test in 0.003s`，`OK`。
  - 相邻测试：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_trading_pipeline -v`
  - 结果：`Ran 14 tests in 4.422s`，`OK`。
  - readiness GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 12 tests in 0.268s`，`OK`。
  - 相邻 legacy deletion/backtest/trading/execution 边界测试：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_trading_pipeline tests.test_execution_engine tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 22 tests in 9.796s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 152 tests in 16.066s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_101244.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只把 `BaseTradingMode` 的旧 DataFrame helper 收窄到 `src/application/trading_pipeline.py`，不改变默认 native backtest 基线。
- 下一步：
  - 继续收敛旧 DataFrame pipeline/execution 删除门禁，优先评估 `src/application/trading_pipeline.py` / `tests/test_trading_pipeline.py` 是否还能进一步迁移或删除，然后处理 `tests/test_execution_engine.py`。

### [x] Task 30：将旧 TradingPipeline 测试收窄为 smoke-only

目标：把 `tests/test_trading_pipeline.py` 从旧 DataFrame 行为事实来源降级为单个 legacy wrapper smoke；当前时间/去重、空仓卖出、超额卖出 clamp、risk reject 分别由 adapter/domain risk/risk adapter 测试承接。

计划来源：Task 29 完成后的旧 DataFrame pipeline/execution 删除门禁继续收敛。

预期修改文件：

- `tests/test_trading_pipeline.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `tests/test_trading_pipeline.py` 只保留 1 个 legacy wrapper smoke 测试。
- [x] 当前时间/去重由 `tests/test_legacy_strategy_adapter.py::test_adapter_keeps_only_current_unprocessed_signal` 覆盖。
- [x] 空仓卖出和超额卖出 clamp 由 `tests/test_domain_risk_policies.py` 覆盖。
- [x] risk reject 由 `tests/test_risk_decision_adapter.py::test_filter_accepted_removes_rejected_rows` 覆盖。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `tests/test_trading_pipeline.py` 已收窄为 smoke-only legacy wrapper 测试。
- `src/application/trading_pipeline.py` 仍作为 legacy wrapper 保留，但不再承载细行为事实来源。
- 下一步删除门禁重点从“迁移旧 pipeline 行为”转为“决定旧 DataFrame wrapper 的删除、归档或显式兼容策略”。
- 迁移 `tests/test_execution_engine.py` 并决定 `src/application/trading_pipeline.py` 的删除或归档策略前不能删除 legacy wrapper。

完成证据：
- 修改文件：
  - `tests/test_trading_pipeline.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_legacy_dataframe_trading_pipeline_tests_are_smoke_only -v`
  - 结果：因 `tests/test_trading_pipeline.py` 仍有 4 个 `async def test_` 失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_trading_pipeline tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_legacy_dataframe_trading_pipeline_tests_are_smoke_only -v`
  - 结果：`Ran 2 tests in 0.126s`，`OK`。
  - 相邻测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_trading_pipeline tests.test_legacy_strategy_adapter tests.test_domain_risk_policies tests.test_risk_decision_adapter -v`
  - 结果：`Ran 22 tests in 0.958s`，`OK`。
  - readiness GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 13 tests in 0.198s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 150 tests in 11.507s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_102353.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只将旧 `TradingPipeline` 兼容测试收窄为 smoke-only，不改变默认 native backtest 基线。
- 下一步：
  - 继续评估 `src/application/trading_pipeline.py` 的删除、归档或显式兼容策略。

### [x] Task 31：删除旧 DataFrame TradingPipeline wrapper

目标：在旧 DataFrame `TradingPipeline` 的可迁移行为已经迁移到 adapter/domain risk/risk adapter 覆盖后，删除 `src/application/trading_pipeline.py`、`BaseTradingMode._process_market_data` 和对应 smoke-only wrapper 测试，避免 mode/application 层继续暴露旧 DataFrame pipeline 入口。

计划来源：Task 30 完成后的旧 DataFrame wrapper 删除决策。

预期修改文件：

- `src/trading/modes/base.py`
- `src/application/trading_pipeline.py`
- `tests/test_trading_pipeline.py`
- `tests/test_backtest_mode.py`
- `tests/test_domain_boundaries.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `src/application/trading_pipeline.py` 已删除。
- [x] `BaseTradingMode._process_market_data` 已删除。
- [x] `tests/test_trading_pipeline.py` 已删除。
- [x] 旧 DataFrame `TradingPipeline` 的可迁移行为已映射到 adapter/domain risk/risk adapter 测试。
- [x] 旧 DataFrame pipeline wrapper 已删除，不再作为剩余 legacy 删除 blocker。
- [x] 旧 DataFrame execution：迁移 `tests/test_execution_engine.py` 前不能删除旧 `ExecutionEngine.execute(signals_df)` 兼容入口。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `src/application/trading_pipeline.py` 已删除。
- `BaseTradingMode._process_market_data` 已删除。
- `tests/test_trading_pipeline.py` 已删除。
- 旧 DataFrame pipeline wrapper 已删除，不再作为剩余 legacy 删除 blocker。
- 剩余 legacy 删除 blocker 是旧 DataFrame execution：迁移 `tests/test_execution_engine.py` 前不能删除旧 `ExecutionEngine.execute(signals_df)` 兼容入口。

完成证据：
- 修改文件：
  - `src/trading/modes/base.py`
  - `src/application/trading_pipeline.py`
  - `tests/test_trading_pipeline.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_domain_boundaries.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_legacy_dataframe_trading_pipeline_wrapper_is_removed -v`
  - 结果：因 `src/application/trading_pipeline.py` 仍存在失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_domain_boundaries tests.test_backtest_mode -v`
  - 结果：`Ran 24 tests in 4.144s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
  - 相邻 legacy/domain/backtest/paper/live/execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_domain_boundaries tests.test_backtest_mode tests.test_backtest_use_case tests.test_paper_mode tests.test_live_mode tests.test_legacy_execution_boundary tests.test_execution_engine tests.test_legacy_strategy_adapter tests.test_domain_risk_policies tests.test_risk_decision_adapter -v`
  - 结果：`Ran 60 tests in 11.120s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 147 tests in 13.762s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_103811.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除旧 DataFrame pipeline wrapper、mode 入口和 smoke-only wrapper 测试，不改变默认 native backtest 基线。
- 下一步：
  - 继续处理 `tests/test_execution_engine.py` 和旧 `ExecutionEngine.execute(signals_df)` 兼容边界。

### [x] Task 32：删除旧 ExecutionEngine 对照测试

目标：在旧 `ExecutionEngine._backtest_execution` fractional quantity/volume 行为已经由 `BacktestExecutionModel.execute_orders()` 覆盖后，删除 `tests/test_execution_engine.py`，让旧回测成交算法事实来源只保留在新执行模型测试里。

计划来源：Task 31 完成后的旧 DataFrame execution 删除门禁继续收敛。

预期修改文件：

- `tests/test_execution_engine.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `tests/test_execution_engine.py` 已删除。
- [x] 旧 `ExecutionEngine._backtest_execution` 行为事实来源已迁移到 `tests/test_backtest_execution_model.py`。
- [x] `tests/test_legacy_execution_boundary.py` 不再要求旧 execution 对照测试存在。
- [x] 旧 DataFrame execution 剩余 blocker 收敛为旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `tests/test_execution_engine.py` 已删除。
- 旧 `ExecutionEngine._backtest_execution` 行为事实来源已迁移到 `tests/test_backtest_execution_model.py`。
- 剩余 legacy 删除 blocker 是旧 DataFrame execution：`tests/test_execution_engine.py` 已删除，剩余 blocker 是旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体。

完成证据：
- 修改文件：
  - `tests/test_execution_engine.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_legacy_dataframe_execution_engine_tests_are_removed_after_behavior_mapping -v`
  - 结果：因 `tests/test_execution_engine.py` 仍存在失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_legacy_execution_boundary tests.test_backtest_execution_model -v`
  - 结果：`Ran 28 tests in 4.598s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
  - 相邻 execution/mode/runtime 边界测试：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_legacy_execution_boundary tests.test_backtest_execution_model tests.test_legacy_execution_adapter tests.test_backtest_mode tests.test_backtest_use_case tests.test_paper_mode tests.test_live_mode tests.test_runtime_builder tests.test_domain_boundaries -v`
  - 结果：`Ran 70 tests in 11.375s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 146 tests in 11.642s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_104727.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除旧 `ExecutionEngine` 对照测试并把行为事实来源收敛到 `BacktestExecutionModel`，不改变默认 native backtest 基线。
- 下一步：
  - 继续决定旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体和 `LegacyExecutionAdapter` 的删除、移动或显式保留策略。

### [x] Task 33：删除未使用的 LegacyExecutionAdapter

目标：在 `LegacyExecutionAdapter` 已无生产引用且旧 `_backtest_execution` 行为事实来源已迁移后，删除 `src/application/adapters/legacy_execution_adapter.py` 和对应兼容测试，避免 application adapter 层继续暴露旧 `ExecutionEngine.execute(signals_df)` 桥接入口。

计划来源：Task 32 完成后的旧 DataFrame execution 删除门禁继续收敛。

预期修改文件：

- `src/application/adapters/legacy_execution_adapter.py`
- `tests/test_legacy_execution_adapter.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `src/application/adapters/legacy_execution_adapter.py` 已删除。
- [x] `tests/test_legacy_execution_adapter.py` 已删除。
- [x] `src.application.adapters` 仍不导出 `LegacyExecutionAdapter`。
- [x] `src/` 中无 `LegacyExecutionAdapter` 生产引用。
- [x] 旧执行桥不再作为剩余 legacy 删除 blocker。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `LegacyExecutionAdapter` 已删除。
- 旧执行桥不再作为剩余 legacy 删除 blocker。
- 剩余 legacy 删除 blocker 是旧 DataFrame execution：旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体。

完成证据：
- 修改文件：
  - `src/application/adapters/legacy_execution_adapter.py`
  - `tests/test_legacy_execution_adapter.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_unused_legacy_execution_adapter_is_removed -v`
  - 结果：因 `src/application/adapters/legacy_execution_adapter.py` 仍存在失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness tests.test_legacy_execution_boundary tests.test_domain_boundaries tests.test_runtime_builder -v`
  - 结果：`Ran 33 tests in 7.476s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
  - 相邻 mode/runtime/execution 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode tests.test_runtime_builder tests.test_backtest_execution_model tests.test_backtest_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 70 tests in 15.003s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 146 tests in 13.967s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_110150.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除无生产引用的 `LegacyExecutionAdapter` 和其兼容测试，不改变默认 native backtest 基线。
- 进度估算：约 `90%`。
- 下一步：
  - 继续收敛旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体，优先评估 mode 中 `_create_legacy_execution_engine()` 懒加载入口和 `src/trading/execution/manager.py` 的删除、移动或显式归档策略。

### [x] Task 34：删除 RuntimeBuilder live legacy exchange client fallback

目标：让 live runtime 组装只接受显式 `mode.exchange_client`，不再从旧 `mode.execution_engine.binance` 获取交易所 client，进一步切断新 application runtime 对旧 `ExecutionEngine` 的隐式依赖。

计划来源：Task 33 完成后的旧 DataFrame execution 删除门禁继续收敛。

预期修改文件：

- `src/application/runtime_builder.py`
- `tests/test_runtime_builder.py`
- `tests/test_live_mode.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`。
- [x] live runtime 只接受显式 `mode.exchange_client`。
- [x] 只有旧 `execution_engine.binance` 时，`RuntimeBuilder.build_live_runtime()` 直接失败。
- [x] live mode 准备 domain pipeline 的测试使用显式 `exchange_client`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`，live runtime 只接受显式 `mode.exchange_client`。
- 旧 `ExecutionEngine` 不再拥有 live runtime 的 exchange client fallback 权限。
- 剩余 legacy 删除 blocker 是旧 DataFrame execution：旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体，以及 mode 中 `_create_legacy_execution_engine()` 懒加载入口。

完成证据：
- 修改文件：
  - `src/application/runtime_builder.py`
  - `tests/test_runtime_builder.py`
  - `tests/test_live_mode.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder.RuntimeBuilderTest.test_build_live_runtime_rejects_legacy_execution_engine_exchange_client -v`
  - 结果：因 `RuntimeBuilder` 仍接受 `mode.execution_engine.binance`，未抛出 `ValueError`，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder -v`
  - 结果：`Ran 10 tests in 0.992s`，`OK`。
  - 文档 RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_runtime_builder_no_longer_reads_legacy_execution_engine_exchange_client -v`
  - 结果：因设计文档和 worklist 尚未记录 runtime builder fallback 删除失败，符合预期。
  - 文档 GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_runtime_builder_no_longer_reads_legacy_execution_engine_exchange_client -v`
  - 结果：`Ran 1 test in 0.005s`，`OK`。
  - 相邻 runtime/live/legacy 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_live_mode tests.test_legacy_deletion_readiness tests.test_legacy_execution_boundary -v`
  - 结果：`Ran 45 tests in 8.278s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 148 tests in 20.268s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_111044.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除 live runtime 的旧 `ExecutionEngine.binance` fallback，不改变默认 native backtest 基线。
- 进度估算：约 `92%`。
- 下一步：
  - 继续删除或归档 mode 中 `_create_legacy_execution_engine()` 懒加载入口，并评估 `src/trading/execution/manager.py` 是否还能保留为旧 DataFrame execution 本体。

### [x] Task 35：删除 mode 级旧 ExecutionEngine factory

目标：在 backtest、paper、live 主路径已经统一通过 `RuntimeBuilder -> DomainTradingPipeline` 后，删除 mode 内部 `_create_legacy_execution_engine()` 和 live 专用 `_create_legacy_live_execution_engine()` 懒加载入口，让 mode 层彻底不再引用旧 `src.trading.execution.manager`。

计划来源：Task 34 完成后的旧 DataFrame execution 删除门禁继续收敛。

预期修改文件：

- `src/trading/modes/backtest.py`
- `src/trading/modes/paper.py`
- `src/trading/modes/live.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BacktestTradingMode` 和 `PaperTradingMode` 的 `_create_legacy_execution_engine()` 已删除。
- [x] backtest/paper mode 不再引用 `src.trading.execution.manager`。
- [x] `LiveTradingMode` 的 `_create_legacy_execution_engine()` 已删除。
- [x] `_create_legacy_live_execution_engine()` 已删除。
- [x] live mode 不再引用 `src.trading.execution.manager`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `BacktestTradingMode` 和 `PaperTradingMode` 的 `_create_legacy_execution_engine()` 已删除，backtest/paper mode 不再引用 `src.trading.execution.manager`；默认 initialize 和 native runtime 组装不会触发旧执行引擎。
- `LiveTradingMode` 的 `_create_legacy_execution_engine()` 已删除，live mode 不再引用 `src.trading.execution.manager`；`_create_legacy_live_execution_engine()` 已删除。
- 剩余 legacy 删除 blocker 收敛为旧 DataFrame execution 本体：旧 `ExecutionEngine.execute(signals_df)` 兼容入口，以及旧 `ExecutionEngine._backtest_execution()` wrapper-only 兼容边界。

完成证据：
- 修改文件：
  - `src/trading/modes/backtest.py`
  - `src/trading/modes/paper.py`
  - `src/trading/modes/live.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_paper_mode.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary.LegacyExecutionBoundaryTest.test_modes_no_longer_expose_legacy_execution_engine_factories -v`
  - 结果：因 `backtest.py`、`paper.py`、`live.py` 仍暴露 `_create_legacy_execution_engine` 或引用 `src.trading.execution.manager` 失败，符合预期。
  - 文档 RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_backtest_and_paper_modes_document_legacy_execution_engine_factory_removal tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_live_mode_documents_legacy_execution_engine_factory_removal -v`
  - 结果：因 worklist 尚未记录 mode 级 legacy factory 删除失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_legacy_deletion_readiness tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode -v`
  - 结果：`Ran 47 tests in 11.780s`，`OK`；输出中有 2 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 149 tests in 18.753s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_113321.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除 mode 级旧 `ExecutionEngine` factory 和旧 manager 引用，不改变默认 native backtest 基线。
- 进度估算：约 `94%`。
- 下一步：
  - 继续清理 mode 中直接读取或关闭 `self.execution_engine` 的剩余兼容分支，并评估 `src/trading/execution/manager.py` 的删除、移动或显式归档策略。

### [x] Task 36：删除 mode 生命周期中的 execution_engine 引用

目标：在 mode 级旧 factory 已删除后，继续移除 `BaseTradingMode` 默认 `self.execution_engine` 字段、backtest/paper/live shutdown 关闭旧 engine 的分支、backtest prepare 对旧 engine 灌历史数据的分支，以及 live account/order helper 对旧 engine 的 fallback。

计划来源：Task 35 完成后的 mode 生命周期剩余旧执行引擎触点清理。

预期修改文件：

- `src/trading/modes/base.py`
- `src/trading/modes/backtest.py`
- `src/trading/modes/paper.py`
- `src/trading/modes/live.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `tests/test_backtest_mode.py`
- `tests/test_paper_mode.py`
- `tests/test_live_mode.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `BaseTradingMode` 不再声明 `self.execution_engine`。
- [x] backtest/paper/live mode 不再读取或关闭 `self.execution_engine`。
- [x] live account/order lifecycle 只通过 `exchange_client`。
- [x] mode 生命周期边界测试覆盖 `base.py`、`backtest.py`、`paper.py`、`live.py`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `BaseTradingMode` 不再声明 `self.execution_engine`；backtest/paper/live mode 不再读取或关闭 `self.execution_engine`。
- live account/order lifecycle 只通过 `exchange_client`；缺少对应 exchange client 方法时返回空结果或失败，不再 fallback 到旧执行引擎。
- 剩余 legacy 删除 blocker 进一步收敛为旧 DataFrame execution 本体：旧 `ExecutionEngine.execute(signals_df)` 兼容入口，以及旧 `ExecutionEngine._backtest_execution()` wrapper-only 兼容边界。

完成证据：
- 修改文件：
  - `src/trading/modes/base.py`
  - `src/trading/modes/backtest.py`
  - `src/trading/modes/paper.py`
  - `src/trading/modes/live.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_paper_mode.py`
  - `tests/test_live_mode.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary.LegacyExecutionBoundaryTest.test_modes_no_longer_reference_execution_engine_lifecycle -v`
  - 结果：因 `base.py`、`backtest.py`、`paper.py`、`live.py` 仍包含 `execution_engine` 生命周期引用失败，符合预期。
  - 代码 GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode tests.test_runtime_builder -v`
  - 结果：`Ran 43 tests in 11.816s`，`OK`；输出中有 3 条 Windows asyncio 慢回调提示，不影响测试结果。
  - 文档 RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_modes_document_execution_engine_lifecycle_removal -v`
  - 结果：因设计文档和 worklist 尚未记录 mode 生命周期删除失败，符合预期。
  - 文档 GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness.LegacyDeletionReadinessTest.test_modes_document_execution_engine_lifecycle_removal -v`
  - 结果：`Ran 1 test in 0.005s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 151 tests in 20.763s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_114347.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除 mode 生命周期中的旧 `self.execution_engine` 引用，不改变默认 native backtest 基线。
- 进度估算：约 `96%`。
- 下一步：
  - 决定旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体的删除、移动或显式保留策略。

### [x] Task 37：删除旧 ExecutionEngine manager 本体

目标：在旧 DataFrame pipeline wrapper、`LegacyExecutionAdapter`、mode factory 和 mode 生命周期引用都删除后，删除最后的旧执行本体 `src/trading/execution/manager.py`，让旧 `ExecutionEngine.execute(signals_df)` 不再作为任何入口存在。

计划来源：Task 36 完成后的最终执行侧 legacy 删除门禁。

预期修改文件：

- `src/trading/execution/manager.py`
- `tests/test_legacy_execution_boundary.py`
- `tests/test_legacy_deletion_readiness.py`
- `tests/test_live_mode.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：

- [x] `src/trading/execution/manager.py` 已删除。
- [x] 旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除。
- [x] 旧 `ExecutionEngine._backtest_execution` 行为事实来源保持在 `tests/test_backtest_execution_model.py` 和 `BacktestExecutionModel`。
- [x] 执行侧 legacy 删除 blocker 已清零。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

兼容边界说明：

- `src/trading/execution/manager.py` 已删除。
- 旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除。
- 回测成交算法事实来源是 `BacktestExecutionModel`，不是旧 `ExecutionEngine`。
- 执行侧 legacy 删除 blocker 已清零；显式 legacy strategy/risk adapter 仍作为策略/风控兼容层保留。

完成证据：
- 修改文件：
  - `src/trading/execution/manager.py`
  - `tests/test_legacy_execution_boundary.py`
  - `tests/test_legacy_deletion_readiness.py`
  - `tests/test_live_mode.py`
  - `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- 聚焦测试：
  - RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary.LegacyExecutionBoundaryTest.test_legacy_execution_manager_module_is_removed -v`
  - 结果：因 `src/trading/execution/manager.py` 仍存在失败，符合预期。
  - 文档 RED：`.\.venv\Scripts\python.exe -m unittest tests.test_legacy_execution_boundary tests.test_live_mode tests.test_legacy_deletion_readiness -v`
  - 结果：代码边界和 live 测试已通过，readiness 因设计文档和 worklist 尚未记录旧 manager 删除失败，符合预期。
  - GREEN：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries tests.test_legacy_execution_boundary tests.test_legacy_deletion_readiness tests.test_live_mode -v`
  - 结果：`Ran 42 tests in 10.182s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 151 tests in 16.701s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_115617.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 差异说明：本任务只删除无生产引用的旧 `ExecutionEngine` manager 本体，不改变默认 native backtest 基线。
- 最终扫描：
  - `src/trading/execution/manager.py` 不存在。
  - `rg -n "trading\.execution\.manager|src\.trading\.execution\.manager" src tests` 仅命中边界测试和 readiness 测试自身，没有生产引用。
- 进度估算：`100%`。
- 下一步：
  - 本轮四层解耦收尾完成；后续若继续推进，应转入显式 legacy strategy/risk adapter 的长期兼容策略或旧 backtest engine 归档策略。

### [x] Task 38：后解耦硬化一：边界门禁、legacy adapter 收窄和回测质量摘要

目标：在四层解耦收尾到 `100%` 后，先做第一批架构硬化：把边界规则自动化，把剩余 legacy strategy/risk adapter 从包级默认导出中撤下，并让默认回测报告携带可检查的数据质量摘要。

计划来源：完成四层解耦后的下一步建议 1、2、3。

修改文件：
- `src/application/adapters/__init__.py`
- `src/application/runtime_builder.py`
- `src/reporting/performance_analyzer.py`
- `tests/test_domain_boundaries.py`
- `tests/test_performance_analyzer.py`
- `tests/test_runtime_builder.py`
- `tests/test_legacy_deletion_readiness.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `src.application.adapters` 不再聚合导出 `LegacyDataFrameStrategyAdapter` 和 `LegacyRiskPolicyAdapter`。
- [x] `RuntimeBuilder` 对显式 legacy strategy/risk adapter 只使用 direct import。
- [x] 默认 domain runtime 仍不使用 legacy strategy/risk adapter。
- [x] 显式 `strategy.interface=legacy` 仍能进入 direct-import legacy adapter 兼容路径。
- [x] `PerformanceAnalyzer` 输出 `quality` 摘要，包含 snapshot/trade 数量、首尾时间、时间顺序、负权益和未平仓标记。
- [x] 默认 native 基线回测报告包含 `quality` 摘要。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries tests.test_performance_analyzer -v`
  - 结果：`test_application_adapters_package_does_not_reexport_legacy_adapters`、`test_runtime_builder_direct_imports_explicit_legacy_adapters` 按预期失败；`test_analyzer_includes_backtest_quality_summary` 和 `test_analyzer_quality_summary_flags_out_of_order_snapshots_and_open_positions` 因缺少 `quality` 字段按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_domain_boundaries tests.test_performance_analyzer tests.test_runtime_builder tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 35 tests in 1.323s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 155 tests in 19.437s`，`OK`；输出中有 1 条 Windows asyncio 慢回调提示，不影响测试结果。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_141448.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 质量摘要：`snapshot_count=1441`，`trade_count=27`，`time_order_valid=true`，`has_negative_equity=false`，`has_open_positions=false`。
- 进度估算：`100%`；本任务属于 100% 后的第一批硬化，不改变原四层解耦完成状态。
- 下一步：
  - 可以继续做 backtest credibility 第二批：数据缺口/重复时间戳检测、成交约束、walk-forward 和参数扫描。

### [x] Task 39：后解耦硬化二：回测时间连续性质量检测

目标：扩展回测报告 `quality` 摘要，让默认 native 基线报告能直接暴露重复时间戳、K 线预期间隔、时间缺口、最大缺口和成交时间越界问题，为后续成交约束、walk-forward 和参数扫描提供更可信的输入检查。

计划来源：Task 38 后的 backtest credibility 第二批建议。

修改文件：
- `src/reporting/performance_analyzer.py`
- `tests/test_performance_analyzer.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `PerformanceAnalyzer.quality` 输出 `duplicate_timestamp_count`。
- [x] `PerformanceAnalyzer.quality` 输出 `expected_interval_seconds`。
- [x] `PerformanceAnalyzer.quality` 输出 `gap_count` 和 `max_gap_seconds`。
- [x] `PerformanceAnalyzer.quality` 输出 `fills_outside_snapshot_range`。
- [x] 正常默认 native 基线回测报告中新增质量字段均为干净状态。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：新增 `test_analyzer_quality_summary_flags_duplicate_gaps_and_fills_outside_snapshot_range` 因缺少 `duplicate_timestamp_count` 字段按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：`Ran 4 tests in 2.133s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_use_case tests.test_report_writer tests.test_domain_boundaries -v`
  - 结果：`Ran 14 tests in 1.768s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 156 tests in 17.746s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_144147.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 质量摘要：`snapshot_count=1441`，`trade_count=27`，`time_order_valid=true`，`duplicate_timestamp_count=0`，`expected_interval_seconds=60.0`，`gap_count=0`，`max_gap_seconds=0`，`fills_outside_snapshot_range=0`，`has_negative_equity=false`，`has_open_positions=false`。
- 进度估算：`100%`；本任务属于 100% 后的第二批硬化，不改变原四层解耦完成状态。
- 下一步：
  - 可以继续做 backtest credibility 第三批：成交约束、滑点/手续费可解释性、walk-forward 和参数扫描。

### [x] Task 40：后解耦硬化三：成交约束质量检测

目标：继续扩展回测报告 `quality` 摘要，让报告层能直接暴露明显不可信的成交记录，包括非正数量、非正价格、负手续费和非正名义金额；本任务只增加报告质量检查，不改变执行模型和基线交易结果。

计划来源：Task 39 后的 backtest credibility 第三批建议。

修改文件：
- `src/reporting/performance_analyzer.py`
- `tests/test_performance_analyzer.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `PerformanceAnalyzer.quality` 输出 `invalid_fill_quantity_count`。
- [x] `PerformanceAnalyzer.quality` 输出 `invalid_fill_price_count`。
- [x] `PerformanceAnalyzer.quality` 输出 `invalid_fill_commission_count`。
- [x] `PerformanceAnalyzer.quality` 输出 `invalid_fill_notional_count`。
- [x] `PerformanceAnalyzer.quality` 输出去重后的 `invalid_fill_count`。
- [x] 正常默认 native 基线回测报告中新增成交约束字段均为 `0`。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：新增 `test_analyzer_quality_summary_flags_invalid_fill_constraints` 因缺少 `invalid_fill_quantity_count` 字段按预期失败；既有精确 `quality` 断言也因缺少新增字段按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：`Ran 5 tests in 2.181s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_use_case tests.test_report_writer tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 31 tests in 0.447s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 157 tests in 16.916s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_144701.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 质量摘要：`snapshot_count=1441`，`trade_count=27`，`time_order_valid=true`，`duplicate_timestamp_count=0`，`expected_interval_seconds=60.0`，`gap_count=0`，`max_gap_seconds=0`，`fills_outside_snapshot_range=0`，`invalid_fill_quantity_count=0`，`invalid_fill_price_count=0`，`invalid_fill_commission_count=0`，`invalid_fill_notional_count=0`，`invalid_fill_count=0`，`has_negative_equity=false`，`has_open_positions=false`。
- 进度估算：`100%`；本任务属于 100% 后的第三批硬化，不改变原四层解耦完成状态。
- 下一步：
  - 可以继续做 backtest credibility 第四批：滑点/手续费可解释性、walk-forward 和参数扫描。

### [x] Task 41：后解耦硬化四：滑点和手续费成本可解释性

目标：在不改变执行模型和基线交易结果的前提下，为回测报告新增 `costs` 摘要，让手续费、滑点和总交易成本可以直接被检查和对比，避免只看到最终净值而看不到交易成本贡献。

计划来源：Task 40 后的 backtest credibility 第四批建议。

修改文件：
- `src/reporting/performance_analyzer.py`
- `tests/test_performance_analyzer.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] 报告输出 `costs.fill_count`。
- [x] 报告输出 `costs.total_notional`。
- [x] 报告输出 `costs.total_commission` 和 `costs.average_commission_per_fill`。
- [x] 报告输出 `costs.commission_rate_bps`。
- [x] 报告输出 `costs.estimated_slippage_cost`、`costs.average_slippage_bps` 和 `costs.max_slippage_bps`。
- [x] 报告输出 `costs.total_transaction_cost`。
- [x] 默认 native 基线回测报告包含 `costs` 摘要。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：新增 `test_analyzer_includes_transaction_cost_explainability` 因缺少 `costs` 字段按预期失败；既有基础报告字段测试新增 `costs` 存在断言也按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer -v`
  - 结果：`Ran 6 tests in 1.713s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_use_case tests.test_report_writer tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 32 tests in 0.244s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 158 tests in 15.829s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_145135.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 成本摘要：`fill_count=27`，`total_notional=21995.377138197546`，`total_commission=21.995377138197547`，`average_commission_per_fill=0.8146435977110202`，`commission_rate_bps=10.0`，`estimated_slippage_cost=21.995377138197547`，`average_slippage_bps=10.0`，`max_slippage_bps=10.0`，`total_transaction_cost=43.990754276395094`。
  - 质量摘要：`snapshot_count=1441`，`trade_count=27`，`time_order_valid=true`，`duplicate_timestamp_count=0`，`expected_interval_seconds=60.0`，`gap_count=0`，`max_gap_seconds=0`，`fills_outside_snapshot_range=0`，`invalid_fill_quantity_count=0`，`invalid_fill_price_count=0`，`invalid_fill_commission_count=0`，`invalid_fill_notional_count=0`，`invalid_fill_count=0`，`has_negative_equity=false`，`has_open_positions=false`。
- 进度估算：`100%`；本任务属于 100% 后的第四批硬化，不改变原四层解耦完成状态。
- 下一步：
  - 可以继续做 walk-forward 和参数扫描；也可以先补报告层成本字段的展示/导出格式增强。

### [x] Task 42：后解耦硬化五：walk-forward 和参数扫描规划底座

目标：先为 walk-forward 和参数扫描建立独立、可测试的 application planning helper，只生成参数组合和训练/测试窗口，不接 CLI、不执行回测、不改变默认基线。后续批量 backtest 执行器可以复用这层稳定输入。

计划来源：Task 41 后的 walk-forward 和参数扫描建议。

修改文件：
- `src/application/backtest_planning.py`
- `tests/test_backtest_planning.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `expand_parameter_grid()` 按调用方给定 key 顺序稳定展开参数组合。
- [x] `expand_parameter_grid()` 对空参数值列表直接报错，并指出参数名。
- [x] `generate_walk_forward_windows()` 基于 `start/end/train_size/test_size` 生成半开区间训练/测试窗口。
- [x] `generate_walk_forward_windows()` 默认按 `test_size` 步进。
- [x] `generate_walk_forward_windows()` 拒绝非正 duration。
- [x] 本任务不接入默认 CLI 回测主路径，默认基线结果不变。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_planning -v`
  - 结果：4 个新增测试均因 `src.application.backtest_planning` 模块不存在按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_planning -v`
  - 结果：`Ran 4 tests in 0.028s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_planning tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 26 tests in 0.271s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 162 tests in 16.019s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_145849.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 成本摘要：`fill_count=27`，`total_notional=21995.377138197546`，`total_commission=21.995377138197547`，`commission_rate_bps=10.0`，`estimated_slippage_cost=21.995377138197547`，`average_slippage_bps=10.0`，`max_slippage_bps=10.0`，`total_transaction_cost=43.990754276395094`。
  - 质量摘要：异常字段均为 `0`。
- 进度估算：`100%`；本任务属于 100% 后的第五批硬化，不改变原四层解耦完成状态。
- 后解耦硬化阶段进度：已完成边界门禁、质量摘要、时间连续性、成交约束、成本解释、walk-forward/参数扫描规划底座；按当前拆分约 `75%`，剩余主要是批量执行器、扫描结果汇总报告和可选 CLI 入口。
- 下一步：
  - 可以继续做参数扫描/ walk-forward 批量执行器，将 planning helper 与现有 backtest use case 连接起来。

### [x] Task 43：后解耦硬化六：参数扫描批量执行底座

目标：在 Task 42 的参数组合和 walk-forward 窗口规划之上，新增一个可注入的批量 backtest runner。当前 runner 只负责按 `BacktestRunSpec` 顺序调用传入的 `run_backtest(spec)` 并结构化记录成功/失败，不直接绑定 CLI、配置系统或真实交易链路，避免污染默认主路径。

计划来源：Task 42 后的参数扫描 / walk-forward 批量执行器建议。

修改文件：
- `src/application/backtest_batch.py`
- `tests/test_backtest_batch_runner.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `build_backtest_run_specs()` 能把参数网格和 walk-forward 窗口组合成稳定顺序的 run specs。
- [x] run id 使用 `run-001` 这种稳定递增格式。
- [x] `BacktestBatchRunner` 能记录成功结果。
- [x] `BacktestBatchRunner` 能捕获单个 run 的异常并记录失败结果，不中断整批扫描。
- [x] 本任务不接入默认 CLI 回测主路径，默认基线结果不变。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_batch_runner -v`
  - 结果：2 个新增测试均因 `src.application.backtest_batch` 模块不存在按预期失败。
- 聚焦 GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_batch_runner tests.test_backtest_planning -v`
  - 结果：`Ran 6 tests in 0.060s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 164 tests in 19.985s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_151000.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
- 进度估算：`100%`；本任务属于 100% 后的第六批硬化，不改变原四层解耦完成状态。
- 后解耦硬化阶段进度：已完成到可注入批量执行底座；按当前拆分约 `90%`，剩余主要是扫描结果汇总报告和可选 CLI 入口。
- 下一步：
  - 继续做扫描结果汇总报告，完成后本轮后解耦硬化阶段可视为 `100%`。

### [x] Task 44：后解耦硬化七：扫描结果汇总报告

目标：在 Task 43 的批量 backtest runner 输出之上，新增轻量扫描结果汇总能力。该 helper 负责按指定指标排序成功 run、保留失败 run、输出 best run 和计数摘要，便于后续参数扫描 / walk-forward 报告层复用；本任务不接入默认 CLI 回测主路径。

计划来源：Task 43 后的扫描结果汇总报告建议。

修改文件：
- `src/reporting/scan_summary.py`
- `tests/test_scan_summary.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `summarize_scan_results()` 能按指定 metric 对成功结果排序并分配 rank。
- [x] 汇总结果包含 `run_count`、`success_count`、`failed_count`、`best_run_id` 和 `best_metric`。
- [x] 单个 run 失败或缺少目标 metric 时不会丢失记录，失败项保留在结果尾部且 rank 为 `None`。
- [x] 本任务不接入默认 CLI 回测主路径，默认基线结果不变。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`
  - 结果：新增测试因 `src.reporting.scan_summary` 模块不存在按预期失败。
- GREEN：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`
  - 结果：`Ran 2 tests in 2.319s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_planning tests.test_backtest_batch_runner tests.test_scan_summary tests.test_report_writer tests.test_report_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 34 tests in 0.513s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 166 tests in 15.561s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_151334.json`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 说明：输出中仍有既有非阻塞提示 `UserWarning: no explicit representation of timezones available for np.datetime64`，不影响测试或回测结果。
- 最终复验：
  - 聚焦命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_planning tests.test_backtest_batch_runner tests.test_scan_summary tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 聚焦结果：`Ran 30 tests`，`OK`。
  - 全量命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 全量结果：`Ran 166 tests`，`OK`。
  - 基线报告：`reports/backtest/backtest_report_20260610_151655.json`
  - 基线结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
- 进度估算：`100%`；本任务属于 100% 后的第七批硬化，不改变原四层解耦完成状态。
- 后解耦硬化阶段进度：已完成边界门禁、质量摘要、时间连续性、成交约束、成本解释、walk-forward/参数扫描规划底座、可注入批量执行器和扫描结果汇总报告；按当前拆分为 `100%`。可选 CLI 入口不再计入本轮硬化完成条件。
- 下一步：
  - 本轮后解耦硬化阶段已完成。后续可以选择可选 CLI 扫描入口、显式 legacy strategy/risk adapter 长期兼容策略，或继续增强因子 / 组合 / 风控能力。

### [x] Task 45：后解耦体验补齐：策略直观图接回 backtest 报告链路

目标：恢复用户可直接肉眼检查策略行为的图形输出。默认 backtest 在保存 JSON 报告后，如果 mode 上存在 `historical_data`，自动生成 `reports/final/ma_ema_crossovers_YYYYMMDD_YYYYMMDD.png`，图中包含价格、短/长 MA、短/长 EMA、MA/EMA crossover 标记和 MA spread 百分比。

计划来源：用户指出本次完整运行没有看到 `ma_ema_crossovers_20250101_20250102.png` 这类直观图片。

修改文件：
- `src/reporting/strategy_chart.py`
- `src/application/report_use_case.py`
- `tests/test_strategy_chart.py`
- `tests/test_report_use_case.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `StrategyChartWriter` 能基于 historical OHLCV 生成 MA/EMA crossover PNG。
- [x] `TradingReportUseCase.save()` 在 backtest 且存在 `historical_data` 时自动生成策略图。
- [x] 策略图默认输出到 `reports/final`，文件名格式为 `ma_ema_crossovers_YYYYMMDD_YYYYMMDD.png`。
- [x] 图生成失败时只记录 warning，不阻断 JSON 报告保存。
- [x] 默认 native 基线回测会生成 `reports/final/ma_ema_crossovers_20250101_20250102.png`。
- [x] 默认 native 基线数值不变。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_strategy_chart tests.test_report_use_case -v`
  - 结果：因 `src.reporting.strategy_chart` 模块不存在和 report save 尚未生成策略图按预期失败。
- GREEN 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_strategy_chart tests.test_report_use_case -v`
  - 结果：`Ran 5 tests in 5.266s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_strategy_chart tests.test_report_use_case tests.test_report_writer tests.test_performance_analyzer tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 34 tests in 3.518s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 168 tests`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_153433.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本任务属于 100% 后体验补齐，不改变解耦完成状态。
- 下一步：
  - 可以继续把策略图扩展为多 symbol、多策略图层，或把图路径写入 JSON 报告元数据。

### [x] Task 46：本地回测研究闭环一：单次 backtest 研究诊断摘要

目标：按照方案文档 Priority 1/2/3，先不修改策略逻辑，而是在默认 backtest 报告中补充研究诊断摘要。诊断摘要用于解释净收益、估算毛收益、交易成本拖累、换手、最大回撤和收益 / 回撤比，为后续参数扫描、walk-forward 和风险 / 仓位优化提供可比较字段。

计划来源：用户确认暂时不修改策略，要求把“研究评估能力、成本和过度交易诊断、风险和仓位诊断”写入方案文档和 worklist，并开始实施。

预期修改文件：
- `src/reporting/backtest_diagnostics.py`
- `src/reporting/performance_analyzer.py`
- `src/application/report_use_case.py`
- `tests/test_backtest_diagnostics.py`
- `tests/test_performance_analyzer.py`
- `tests/test_report_use_case.py`
- `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] 默认 backtest 报告包含 `diagnostics` 摘要。
- [x] `diagnostics` 包含净收益、估算毛收益、总交易成本、成本拖累比例、成本 / 绝对净收益比、平均成交名义金额、换手率、最大回撤和收益 / 回撤比。
- [x] 该任务只增强报告与研究诊断，不修改策略信号、风控执行或仓位计算行为。
- [x] 默认 native 基线净值、交易数和策略图输出不变。
- [x] 完整测试通过。
- [x] 默认 native 基线回测通过。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_diagnostics tests.test_performance_analyzer -v`
  - 结果：因 `src.reporting.backtest_diagnostics` 模块不存在，且 `PerformanceAnalyzer` 尚未输出 `diagnostics` 按预期失败。
- fallback 路径 RED：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_report_use_case.TradingReportUseCaseTest.test_generate_report_preserves_current_summary_fields -v`
  - 结果：旧 mode-state fallback 报告尚未包含 `diagnostics`，按预期失败。
- GREEN 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_report_use_case -v`
  - 结果：`Ran 12 tests in 1.543s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_report_use_case tests.test_report_writer tests.test_scan_summary tests.test_backtest_batch_runner tests.test_backtest_planning tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 43 tests in 2.019s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 170 tests`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_155744.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 诊断摘要：`gross_return_estimate=-2.6538501879429504`，`total_transaction_cost=43.990754276395094`，`cost_to_abs_net_return=0.9431048838677163`，`turnover_pct_of_initial=21.995377138197544`，`return_to_drawdown=-0.9365338889740047`，`has_open_positions=False`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本任务属于本地回测研究闭环第一步，不改变解耦完成状态。
- 下一步：
  - 将 `diagnostics` 接入扫描结果汇总排序，支持按成本拖累、估算毛收益和收益 / 回撤比比较参数或窗口。

### [x] Task 47：本地回测研究闭环二：扫描汇总支持 diagnostics 嵌套指标排序

目标：让 Task 46 新增的 `diagnostics` 字段真正服务批量研究。`summarize_scan_results()` 支持通过点路径读取嵌套指标，例如 `diagnostics.cost_to_abs_net_return`、`diagnostics.gross_return_estimate` 和 `diagnostics.return_to_drawdown`，并把每个 run 的 `diagnostics` 透传到汇总行中。

计划来源：Task 46 完成后的下一步建议：将诊断摘要接入扫描结果汇总排序，支持按成本拖累、估算毛收益和收益 / 回撤比比较参数或窗口。

修改文件：
- `src/reporting/scan_summary.py`
- `tests/test_scan_summary.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] 扫描汇总支持 `diagnostics.*` 点路径指标。
- [x] 可通过 `descending=False` 按成本拖累这类“越小越好”的指标升序排序。
- [x] 缺少目标诊断指标的 run 不会丢失，会保留在结果尾部且 rank 为 `None`。
- [x] 汇总行透传 `diagnostics`，供后续报告或 CLI 输出使用。
- [x] 本任务不修改策略信号、风控执行或仓位计算行为。
- [x] 默认 native 基线净值、交易数和策略图输出不变。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`
  - 结果：新增嵌套指标测试因 `scan_summary` 尚不支持点路径读取而按预期失败。
- GREEN 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`
  - 结果：`Ran 3 tests in 2.278s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_backtest_batch_runner tests.test_backtest_planning tests.test_report_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 43 tests in 2.649s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 171 tests`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_160716.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本任务属于本地回测研究闭环第二步，不改变解耦完成状态。
- 下一步：
  - 增加研究批量入口，把参数网格、默认回测命令、报告路径、策略图路径和扫描汇总串起来。

### [x] Task 48：本地回测研究闭环三：应用层批量研究入口

目标：在不接 CLI、不修改策略逻辑的前提下，新增一个可测试的应用层研究入口，把参数网格规划、可注入批量执行器和扫描汇总串起来。后续 CLI 或脚本只需要作为薄入口调用该编排函数。

计划来源：Task 47 完成后的下一步建议：增加研究批量入口，把参数网格、默认回测命令、报告路径、策略图路径和扫描汇总串起来。

修改文件：
- `src/application/backtest_research.py`
- `tests/test_backtest_research.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `run_backtest_research()` 能从参数网格构建稳定 run specs。
- [x] `run_backtest_research()` 使用注入的 `run_backtest(spec)` 执行每个 run，不绑定 CLI 或真实交易链路。
- [x] 成功和失败 run 都进入扫描汇总。
- [x] 研究入口可直接按 `diagnostics.*` 嵌套指标排序。
- [x] 本任务不修改策略信号、风控执行或仓位计算行为。
- [x] 默认 native 基线净值、交易数和策略图输出不变。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research -v`
  - 结果：因 `src.application.backtest_research` 模块不存在按预期失败。
- GREEN 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research -v`
  - 结果：`Ran 2 tests in 1.664s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research tests.test_backtest_batch_runner tests.test_backtest_planning tests.test_scan_summary tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_report_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 45 tests in 1.707s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 173 tests in 12.961s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260610_164308.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本任务属于本地回测研究闭环第三步，不改变解耦完成状态。
- 下一步：
  - 增加 CLI 薄入口或 JSON summary writer，使研究入口可以直接从命令行运行并保存汇总报告。

### [x] Task 49：本地回测研究闭环四：研究汇总 JSON writer 和应用层落盘

目标：在不接 CLI、不修改策略逻辑的前提下，让 Task 48 的应用层研究入口具备可选的 JSON summary 落盘能力。后续 CLI 或脚本可以直接调用 `run_backtest_research(..., summary_report_dir=...)`，把扫描汇总写入 `reports/research/research_summary_YYYYMMDD_HHMMSS.json`。

计划来源：Task 48 完成后的下一步建议：增加 CLI 薄入口或 JSON summary writer，使研究入口可以直接保存汇总报告。

修改文件：
- `src/reporting/research_writer.py`
- `src/reporting/__init__.py`
- `src/application/backtest_research.py`
- `tests/test_research_writer.py`
- `tests/test_backtest_research.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `ResearchSummaryWriter.write(summary)` 会创建输出目录，写入 `research_summary_*.json`，并返回 `Path`。
- [x] `run_backtest_research(..., summary_report_dir=...)` 会在完成扫描汇总后保存 JSON，并在返回 summary 中带上 `summary_path`。
- [x] summary 落盘不绑定 CLI，不执行真实交易链路，不修改策略信号、风控执行或仓位计算行为。
- [x] 默认 native 基线净值、交易数和策略图输出不变。

完成证据：
- RED 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_research_writer -v`
  - 结果：因 `src.reporting.research_writer` 模块不存在按预期失败。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research -v`
  - 结果：因 `run_backtest_research()` 不支持 `summary_report_dir` 参数按预期失败。
- GREEN 测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research tests.test_research_writer -v`
  - 结果：`Ran 4 tests in 1.816s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_research tests.test_research_writer tests.test_backtest_batch_runner tests.test_backtest_planning tests.test_scan_summary tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_report_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 47 tests in 2.323s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 175 tests in 19.868s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 日志：`reports/run_logs/task49_backtest_20260610.log`
  - 报告：`reports/backtest/backtest_report_20260610_165047.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`。
  - 诊断摘要：`cost_to_abs_net_return=0.9431048838677163`，`turnover_pct_of_initial=21.995377138197544`，`return_to_drawdown=-0.9365338889740047`，`has_open_positions=False`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本地回测研究闭环按当前拆分约 `80%`，已经完成诊断、诊断排序、应用层研究入口和 JSON summary 落盘。
- 下一步：
  - 增加 CLI 薄入口或真实 backtest run adapter，让 `reports/research` 产出来自真实批量 backtest，而不是只具备可注入测试入口。

### [x] Task 50：本地回测研究闭环五：真实 backtest run adapter

目标：新增可注入但能连接真实 `TradingCore` 的 backtest runner，把 `BacktestRunSpec` 转换成克隆配置、参数覆盖、窗口覆盖和真实回测执行，避免 CLI 层直接理解交易主链路。

修改文件：
- `src/application/configured_backtest_runner.py`
- `tests/test_configured_backtest_runner.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] 每个 run 使用 `ConfigManager.get_all()` 克隆配置，不污染原始配置。
- [x] 参数名无点路径时同时写入 `strategy.parameters.*` 和当前 active strategy 配置。
- [x] 参数名为点路径时按显式配置路径写入。
- [x] walk-forward window 存在时使用 test window 覆盖 `backtest.period.start/end`。
- [x] pipeline 返回 `error` 时转换为失败 run，由 batch runner 记录，不中断整批研究。

完成证据：
- RED 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_configured_backtest_runner -v`，因 `src.application.configured_backtest_runner` 模块不存在按预期失败。
- GREEN 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_configured_backtest_runner -v`，`Ran 2 tests in 0.169s`，`OK`。

### [x] Task 51：本地回测研究闭环六：research CLI 薄入口

目标：给 launcher 增加 `--research` 薄入口，让命令行可以解析参数网格、排序指标和输出目录，然后委托 application 层运行真实研究，不把研究逻辑放进 UI 层。

修改文件：
- `src/launcher.py`
- `src/ui/research_cli.py`
- `tests/test_launcher_cli.py`
- `src/reporting/research_writer.py`
- `tests/test_research_writer.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] `--cli --research` 会分流到 `src.ui.research_cli.run_research_cli_mode()`。
- [x] `--research-grid` 解析 JSON 参数网格，`--research-metric` 支持传入 `diagnostics.*` 指标。
- [x] research CLI 会应用普通 backtest 覆盖参数：strategy、symbol、timeframe、start-date、end-date。
- [x] `ResearchSummaryWriter` 落盘 JSON 中包含 `summary_path`，研究产物可自描述。
- [x] 真实 research CLI 可生成 `reports/research/research_summary_*.json`。

完成证据：
- RED 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli -v`，因 `src.ui.research_cli` 模块不存在按预期失败；新增 common override 测试曾因未应用 CLI 覆盖按预期失败。
- GREEN 测试：`.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli.ResearchCliRunnerTest tests.test_launcher_cli.LauncherCliTest -v`，`Ran 6 tests in 1.403s`，`OK`。
- 真实 research CLI：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --research --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv --research-grid "{\"short_window\":[20]}" --research-metric diagnostics.cost_to_abs_net_return --research-ascending`
  - 日志：`reports/run_logs/task51_research_cli_20260610.log`
  - 研究汇总：`reports/research/research_summary_20260610_170215.json`
  - backtest 报告：`reports/backtest/backtest_report_20260610_170211.json`
  - 结果：`run_count=1`，`success_count=1`，`best_metric=0.9431048838677163`，最终净值 `99953.35539553566`，总交易数 `27`。

### [x] Task 52：本地回测研究闭环七：research CLI walk-forward 窗口和 summary window 透传

目标：把 application 层已有的 walk-forward 能力暴露到 research CLI，并让扫描汇总每行明确记录 train/test window，保证研究产物可追溯。

修改文件：
- `src/launcher.py`
- `src/ui/research_cli.py`
- `src/reporting/scan_summary.py`
- `tests/test_launcher_cli.py`
- `tests/test_scan_summary.py`
- `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

验收：
- [x] CLI 支持 `--research-walk-forward-start/end`、`--research-train-days`、`--research-test-days`、`--research-step-days`。
- [x] research CLI 会构建 `WalkForwardWindow` 并传给 `run_backtest_research()`。
- [x] `summarize_scan_results()` 每行输出 `window`，包含 `train_start/train_end/test_start/test_end`。
- [x] 真实 walk-forward research CLI 可生成包含 window 字段的 summary JSON。
- [x] 默认 native 基线净值、交易数和策略图输出不变。

完成证据：
- RED 测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli.ResearchCliRunnerTest -v`，因未传 `windows` 按预期失败。
  - `.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`，因 summary row 未输出 `window` 按预期失败。
- GREEN 测试：
  - `.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli.ResearchCliRunnerTest tests.test_launcher_cli.LauncherCliTest -v`，`Ran 7 tests in 2.248s`，`OK`。
  - `.\.venv\Scripts\python.exe -m unittest tests.test_scan_summary -v`，`Ran 4 tests in 2.796s`，`OK`。
- 相关聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_configured_backtest_runner tests.test_backtest_research tests.test_research_writer tests.test_launcher_cli tests.test_backtest_batch_runner tests.test_backtest_planning tests.test_scan_summary tests.test_backtest_diagnostics tests.test_performance_analyzer tests.test_report_use_case tests.test_domain_boundaries tests.test_legacy_deletion_readiness -v`
  - 结果：`Ran 58 tests in 2.805s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 182 tests in 22.519s`，`OK`。
- 真实 walk-forward research CLI：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --research --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --backtest-engine ohlcv --research-grid "{\"short_window\":[20]}" --research-metric diagnostics.cost_to_abs_net_return --research-ascending --research-walk-forward-start 2024-12-31 --research-walk-forward-end 2025-01-02 --research-train-days 1 --research-test-days 1 --research-step-days 1`
  - 日志：`reports/run_logs/task52_research_cli_walk_forward_20260610.log`
  - 研究汇总：`reports/research/research_summary_20260610_170742.json`
  - backtest 报告：`reports/backtest/backtest_report_20260610_170737.json`
  - 策略图：`reports/final/ma_ema_crossovers_20250101_20250102.png`
  - window：`train_start=2024-12-31T00:00:00`，`train_end=2025-01-01T00:00:00`，`test_start=2025-01-01T00:00:00`，`test_end=2025-01-02T00:00:00`。
  - 结果：最终净值 `99953.35539553566`，总交易数 `27`，买入 `11`，卖出 `16`，结束持仓 `{}`，`cost_to_abs_net_return=0.9431048838677163`。
- 进度估算：四层解耦主阶段仍为 `100%`；后解耦硬化阶段仍为 `100%`；本地回测研究闭环第一版为 `100%`。已经完成 Priority 1 的批量研究入口/CLI/walk-forward，Priority 2 的成本和过度交易诊断，Priority 3 的基础风险和仓位诊断。
- 下一步：
  - 不建议立刻改策略。更好的下一轮是提升研究产物质量：summary 中加入每个 run 的 backtest 报告路径和策略图路径索引、配置快照 hash、参数扫描示例脚本，然后再基于多参数/多窗口证据讨论策略或风控修改。

## 进度记录

### 2026-06-09

- 小修设计文档：`docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`。
  - 明确当前是“新 RuntimeBuilder 路径已共用 DomainTradingPipeline，旧 fallback 仍存在”。
  - 明确当前 `StrategyPort` 是 v1：`generate(market, portfolio)`；Factor 层落地后再升级到 v2：`generate(market, factors, portfolio)`。
  - 调整推荐优先级为：Phase A -> D-min -> C -> B -> E -> F。
- 新增实施计划：`docs/plans/2026-06-09-final-four-layer-decoupling-implementation-plan.md`。
- 新增本文档作为本轮可续接 worklist。
- 完成 Task 1：新增 launcher 默认配置路径 helper，将默认配置从 `conf/bt_config.yaml` 改为 `conf/config.yaml`；更新 `--backtest-engine` help 文案，明确它只是兼容执行模型参数；补充 CLI 测试证明 launcher 主路径不会实例化旧 `BacktestFactory`。
- 完成 Task 2：新增最小 `src/order` 层，包括 `PositionSizingDecision`、`FixedNotionalSizer`、`FixedFractionSizer` 和 `OrderFactory`；当前未接入主链路，先为后续原生策略迁移提供策略外 quantity 落点，基线回测不变。
- 完成 Task 3：新增最小 `src/factor` 层，包括 `FactorView` 和 `FactorEngine`；当前支持按 symbol 维护窗口并生成基础 `ma_N` 因子，domain 边界保持无 pandas 依赖，基线回测不变。
- 完成 Task 4：新增原生领域版 `DomainDualMAStrategy`，支持 `strategy.interface=domain` 显式选择；默认仍保留 legacy adapter 以保护既有基线，native 路径不再由策略输出最终 `quantity`。
- 完成 Task 5：新增 `PerformanceAnalyzer` 和 `ReportWriter`，`TradingReportUseCase` 优先从 domain reporter 的 fills/snapshots 计算报告，mode 私有状态只作为 legacy fallback。
- 完成 Task 6：`BacktestUseCase` 主运行入口收敛到 `DomainTradingPipeline`，缺少 `domain_pipeline` 时直接失败；旧 DataFrame pipeline、旧 execution manager 和旧 backtest engine 已标记为 `LEGACY COMPATIBILITY`。
- 开始 Task 7：按确认后的“增量 native 迁移”方案，先补设计文档、实施计划和 worklist；目标是 native `multi_factors` 与可配置 `FactorEngine`，暂不删除旧策略和 adapter。
- 完成 Task 7：新增配置化 `FactorEngine.from_config()`、原生 `DomainMultiFactorsStrategy`，并让 `RuntimeBuilder` 支持 `strategy.interface=domain` + `strategy.active=multi_factors`。
- 开始 Task 8：按保守默认化方案，将主配置默认切到 `strategy.interface=domain`，代码层默认仍保留 legacy 兜底。
- 完成 Task 8：`conf/config.yaml` 已明确设置 `strategy.active=dual_ma` 和 `strategy.interface=domain`；默认 CLI 基线已更新为 native domain 结果，代码层未声明 interface 时仍回落 legacy。
- 开始 Task 9：补充 legacy 删除就绪度门禁测试，确认默认 native 主路径不使用 legacy strategy/risk adapter，并把 legacy 删除暂缓 blockers 写入设计文档和 worklist。
- 完成 Task 9：新增 `tests/test_legacy_deletion_readiness.py`，主配置默认 runtime 已验证为 native strategy/risk path；设计文档和 worklist 已明确 legacy 删除暂缓原因，本轮未删除任何 legacy 模块。
- 开始 Task 10：迁移 live exchange client ownership 的最小步，目标是 `RuntimeBuilder` 优先使用 `mode.exchange_client`，旧 `ExecutionEngine.binance` 只作为兼容 fallback。
- 完成 Task 10：`RuntimeBuilder` 已优先使用显式 `mode.exchange_client`，`LiveTradingMode.initialize()` 会暴露 `self.exchange_client`；旧 `execution_engine.binance` fallback 保留。
- 开始 Task 11：迁移 live 账号校验和订单关闭逻辑，目标是优先使用 `exchange_client`，旧 `ExecutionEngine` 仅保留 fallback。
- 完成 Task 11：live 账号余额、状态更新、open orders 查询和撤单均已优先通过 `exchange_client`，旧 `execution_engine` fallback 保留。
- 开始 Task 12：把 backtest/paper 旧 `ExecutionEngine` 从 initialize 主路径迁到 legacy 懒加载入口。
- 完成 Task 12：backtest/paper initialize 不再主动创建旧 `ExecutionEngine`；旧 DataFrame 执行入口需要时通过 `_create_legacy_execution_engine()` 懒加载，默认 native 回测基线保持不变。
- 开始 Task 13：审计旧 `ExecutionEngine` 剩余引用，并为 legacy-only 边界建立测试门禁。
- 完成 Task 13：新增 legacy execution boundary 测试，限定旧 `ExecutionEngine` 的生产引用点；live initialize 在已有显式 `exchange_client` 时不再创建旧 `ExecutionEngine`，旧 engine 仅作为兼容 fallback。
- 开始 Task 14：收窄旧 `tests/test_execution_engine.py` 的命名和范围，让它明确是 legacy DataFrame execution 测试。
- 完成 Task 14：`tests/test_execution_engine.py` 已加 legacy DataFrame docstring，测试类名已改为 `LegacyDataFrameExecutionEngineTest`，并由边界测试守住命名语义。
- 开始 Task 15：评估并最小迁移 live exchange client 默认构造，减少 live 对旧 `ExecutionEngine` 的 fallback 依赖。
- 完成 Task 15：live initialize 无显式 client 时优先通过 `_create_exchange_client()` 构造 `Binance(config)`，旧 `ExecutionEngine` live 构造只作为最后 fallback 保留。
- 开始 Task 16：给独立 `Binance` adapter 补齐 live order/account 薄方法。
- 完成 Task 16：`Binance` adapter 已补齐 `create_order/get_account_balance/get_open_orders/cancel_order` 薄代理方法，并由 fake exchange 单测覆盖；live 默认 exchange client 不再因这些方法缺失而需要旧 engine。
- 开始 Task 17：评估 live 旧 `ExecutionEngine` fallback 是否可以显式化或删除。
- 完成 Task 17：live 默认初始化在 exchange client 构造失败时直接失败，不再自动创建旧 `ExecutionEngine`；旧 fallback 保留为 `live_trading.allow_legacy_execution_engine_fallback=true` 显式兼容路径，并由 live mode 测试覆盖。
- 完成 Task 18：更新 legacy 删除门禁和剩余阻塞清单，live 默认 exchange client ownership blocker 已降级为显式 legacy fallback 兼容项，剩余 blocker 收敛为旧配置 fallback、旧 DataFrame pipeline/execution 和 live 显式 legacy fallback。
- 完成 Task 19：旧配置 fallback 已收紧为显式兼容边界，未声明 `strategy.interface` 的配置继续回落 legacy adapter，但不再代表主路径默认行为。
- 完成 Task 20：旧 DataFrame pipeline/execution 已标记为显式兼容边界，`BaseTradingMode` 旧入口和 `tests/test_trading_pipeline.py` 均有 legacy 标识，并由 readiness 门禁守住。
- 完成 Task 21：旧 DataFrame `TradingPipeline` 的当前 timestamp 过滤行为已迁移到 `DomainTradingPipeline` 覆盖，risk reject、空仓卖出、超额卖出 clamp 映射到既有 domain pipeline/risk policy 覆盖。
- 完成 Task 22：旧 `ExecutionEngine._backtest_execution` fractional quantity/volume 行为已映射到 `BacktestExecutionModel.execute_orders()` 覆盖，并由 readiness 文档门禁守住。
- 完成 Task 23：旧 `ExecutionEngine._backtest_execution()` 已由边界测试限定为 `BacktestExecutionModel.execute_orders()` 的 wrapper-only 兼容入口，回测成交算法事实来源保持在新执行模型。
- 完成 Task 24：`LegacyExecutionAdapter` 已从 `src.application.adapters` 包级默认导出中移除，只能直接从 `src.application.adapters.legacy_execution_adapter` 导入作为旧执行桥兼容入口。
- 完成 Task 25：`BacktestTradingMode` 和 `PaperTradingMode` 已不再顶层 import 旧 `ExecutionEngine`；旧 engine 仅在 `_create_legacy_execution_engine()` 内部按需导入。
- 完成 Task 26：`LiveTradingMode` 已不再顶层 import 旧 `ExecutionEngine`；旧 engine 仅在 `_create_legacy_live_execution_engine()` 和 `_create_legacy_execution_engine()` 内部按需导入。
- 完成 Task 27：live 显式 legacy fallback 已删除；`live_trading.allow_legacy_execution_engine_fallback=true` 不再创建旧 `ExecutionEngine`，缺少 exchange client 时 live initialize 直接失败。
- 完成 Task 28：未声明 `strategy.interface` 的配置默认进入 domain runtime；只有显式 `strategy.interface=legacy` 才启用 legacy adapter。
- 完成 Task 29：`BaseTradingMode._process_market_data` 只保留 thin legacy entrypoint，并委托到 `src/application/trading_pipeline.py`；旧 DataFrame signal/execution helper 已从 mode 层移除。
- 完成 Task 30：`tests/test_trading_pipeline.py` 已收窄为 smoke-only legacy wrapper 测试，旧 pipeline 细行为事实来源由 adapter/domain risk/risk adapter 测试承接。
- 完成 Task 31：`src/application/trading_pipeline.py`、`BaseTradingMode._process_market_data` 和 `tests/test_trading_pipeline.py` 已删除；旧 DataFrame pipeline wrapper 已不再作为剩余 legacy 删除 blocker。
- 完成 Task 32：`tests/test_execution_engine.py` 已删除；旧 `ExecutionEngine._backtest_execution` 行为事实来源已迁移到 `tests/test_backtest_execution_model.py`。
- 完成 Task 33：`LegacyExecutionAdapter` 和 `tests/test_legacy_execution_adapter.py` 已删除；旧执行桥不再作为剩余 legacy 删除 blocker。
- 完成 Task 34：`RuntimeBuilder._exchange_client()` 不再读取 `mode.execution_engine.binance`；live runtime 只接受显式 `mode.exchange_client`。
- 完成 Task 35：backtest/paper/live mode 的旧 `_create_legacy_execution_engine()` factory 已删除；live 专用 `_create_legacy_live_execution_engine()` 已删除；mode 层不再引用 `src.trading.execution.manager`。
- 完成 Task 36：`BaseTradingMode` 不再声明 `self.execution_engine`；backtest/paper/live mode 不再读取或关闭 `self.execution_engine`；live account/order lifecycle 只通过 `exchange_client`。
- 完成 Task 37：`src/trading/execution/manager.py` 已删除；旧 `ExecutionEngine.execute(signals_df)` 兼容入口本体已删除；执行侧 legacy 删除 blocker 已清零。
- 完成 Task 38：`src.application.adapters` 不再聚合导出 legacy strategy/risk adapter；`RuntimeBuilder` 显式 legacy 兼容路径改为 direct import；默认回测报告已加入基础 `quality` 摘要。
- 完成 Task 39：`PerformanceAnalyzer.quality` 已扩展重复时间戳、预期间隔、缺口、最大缺口和成交越界检测；默认 native 基线报告质量字段干净。
- 完成 Task 40：`PerformanceAnalyzer.quality` 已扩展成交约束检测，覆盖非正数量、非正价格、负手续费、非正名义金额和去重后的异常成交数；默认 native 基线报告质量字段干净。
- 完成 Task 41：回测报告已新增 `costs` 摘要，覆盖成交名义金额、总手续费、手续费 bps、估算滑点成本、滑点 bps 和总交易成本；默认 native 基线报告成本字段可解释。
- 完成 Task 42：新增 `src.application.backtest_planning`，提供参数网格稳定展开和 walk-forward 训练/测试窗口生成；当前只作为规划底座，不接入默认 CLI 回测主路径。
- 完成 Task 43：新增 `src.application.backtest_batch`，提供稳定 run spec 构建和可注入批量 backtest runner；单个 run 失败会被记录为失败结果，不中断整批扫描，当前不接入默认 CLI 回测主路径。
- 完成 Task 44：新增 `src.reporting.scan_summary`，提供扫描结果汇总报告 helper；成功 run 按指定 metric 排序并生成 rank，失败或缺 metric 的 run 保留在结果尾部，当前不接入默认 CLI 回测主路径。
- 完成 Task 45：新增 `src.reporting.strategy_chart`，并让 `TradingReportUseCase.save()` 在 backtest 保存报告时自动生成 `reports/final/ma_ema_crossovers_YYYYMMDD_YYYYMMDD.png`；默认 native 基线已验证生成 `reports/final/ma_ema_crossovers_20250101_20250102.png`。
- 完成 Task 46：新增 `src.reporting.backtest_diagnostics`，并把 `diagnostics` 写入默认 backtest JSON 报告；诊断摘要覆盖净收益、估算毛收益、交易成本拖累、换手、最大回撤和收益 / 回撤比，为后续扫描汇总排序提供研究字段。
- 完成 Task 47：`summarize_scan_results()` 已支持 `diagnostics.*` 点路径嵌套指标排序，并在汇总行透传 `diagnostics`，后续批量研究可以按成本拖累、估算毛收益和收益 / 回撤比排序。
- 完成 Task 48：新增 `src.application.backtest_research.run_backtest_research()`，把参数网格规划、可注入批量执行器和扫描汇总组合成应用层研究入口；当前不接 CLI、不修改策略逻辑。
- 完成 Task 49：新增 `src.reporting.research_writer.ResearchSummaryWriter`，并让 `run_backtest_research(..., summary_report_dir=...)` 可选保存研究汇总 JSON；当前仍不接 CLI、不修改策略逻辑。
- 完成 Task 50：新增 `src.application.configured_backtest_runner.make_configured_backtest_runner()`，可把 `BacktestRunSpec` 转为真实 `TradingCore` backtest run，并隔离每个 run 的配置覆盖。
- 完成 Task 51：新增 `--cli --research` 薄入口，支持 JSON 参数网格、诊断指标排序、输出目录和普通 backtest CLI 覆盖；真实 research CLI 已生成 `reports/research/research_summary_20260610_170215.json`。
- 完成 Task 52：research CLI 已支持 walk-forward 参数；扫描汇总每行已透传 `window`，真实 walk-forward research CLI 已生成 `reports/research/research_summary_20260610_170742.json`。
- 当前断点：Task 52 已完成；完整测试 `Ran 182 tests, OK`，默认 native / research CLI 基线回测报告 `reports/backtest/backtest_report_20260610_170737.json`，研究汇总 `reports/research/research_summary_20260610_170742.json`，策略图 `reports/final/ma_ema_crossovers_20250101_20250102.png`，最终净值 `99953.35539553566`、总交易数 `27`、买入 `11`、卖出 `16`、结束持仓 `{}`；`diagnostics.cost_to_abs_net_return=0.9431048838677163`。本轮四层解耦仍为 `100%`；100% 后硬化阶段也已达到 `100%`；本地回测研究闭环第一版已达到 `100%`。
