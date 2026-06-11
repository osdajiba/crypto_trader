# 交易流水线第二阶段实施清单

日期：2026-06-09

关联文档：

- 第二阶段设计文档：`docs/design/2026-06-09-trading-pipeline-phase-2-design.md`
- 第一阶段实施清单：`docs/worklist/2026-06-09-trading-pipeline-domain-core-worklist.md`

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

## 实施项

### Phase 6：抽出 TradingPipeline

目标：将 `BaseTradingMode._process_market_data()` 的主体迁出 mode，让 mode 只保留兼容委托入口。

- [x] 创建 `TradingPipeline`。
  - 候选路径：`src/application/trading_pipeline.py`
- [x] 添加 `TradingPipeline.run_once(data_map)` 测试，覆盖当前 timestamp 只执行当前信号。
  - 候选测试：`tests/test_trading_pipeline.py`
- [x] 添加同一信号不会重复执行的测试。
- [x] 添加空仓卖出跳过测试。
- [x] 添加超额卖出缩小到当前持仓测试。
- [x] 添加有效成交进入 `PortfolioBook` 的测试。
- [x] 让 `BaseTradingMode._process_market_data()` 委托给 `TradingPipeline`。
- [x] 保留当前策略、风控和执行引擎 DataFrame 协议。
- [x] 运行聚焦测试。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/trading_pipeline.py`
- `src/trading/modes/base.py`
- `tests/test_trading_pipeline.py`
- `tests/test_backtest_mode.py`
- `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`

预期验证：

- `BaseTradingMode._process_market_data()` 不再拥有完整交易决策流程。
- `TradingPipeline` 可被单独测试。
- 基线回测仍然产生 60 笔交易，并且结束无持仓。

完成证据：

- 新增/修改文件：
  - `src/application/trading_pipeline.py`
  - `src/trading/modes/base.py`
  - `tests/test_trading_pipeline.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_market_data_feed.py`
  - `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_trading_pipeline tests.test_backtest_mode`
  - 结果：2 个错误，分别是 `ModuleNotFoundError: No module named 'src.application.trading_pipeline'` 和 `BaseTradingMode` 模块中没有 `TradingPipeline`，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_trading_pipeline tests.test_backtest_mode`
  - 结果：`Ran 9 tests in 7.392s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 41 tests in 8.615s`，`OK`。
  - 备注：Windows asyncio debug 输出了一条 `BaseProactorEventLoop._loop_self_reading()` 慢回调提示，但测试结果为通过。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_095141.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 7：引入 Signal/Risk 兼容 Adapter

目标：为未来替换策略接口做准备，但不破坏当前 DataFrame 策略和风控。

- [x] 添加 `StrategySignalAdapter`。
  - 候选路径：`src/application/adapters/strategy_signal_adapter.py`
- [x] 添加 `RiskDecisionAdapter`。
  - 候选路径：`src/application/adapters/risk_decision_adapter.py`
- [x] 将当前策略 DataFrame 输出转换为稳定 `StrategySignal` 或兼容 signal key。
- [x] 将当前风控 DataFrame 输出转换为兼容 `RiskDecision`。
- [x] 让 `TradingPipeline` 通过 adapter 调用策略和风控。
- [x] 测试 DataFrame 策略输出可以生成稳定 signal id。
- [x] 测试重复 signal id 不会重复执行。
- [x] 测试风控拒绝信号不会进入执行。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/adapters/__init__.py`
- `src/application/adapters/strategy_signal_adapter.py`
- `src/application/adapters/risk_decision_adapter.py`
- `src/application/trading_pipeline.py`
- `tests/test_strategy_signal_adapter.py`
- `tests/test_risk_decision_adapter.py`
- `tests/test_trading_pipeline.py`

预期验证：

- 当前 `DualMAStrategy` 不需要修改。
- 当前 risk manager 不需要修改。
- 策略替换时可以从 adapter 边界进入，而不是直接改执行逻辑。

完成证据：

- 新增/修改文件：
  - `src/application/adapters/__init__.py`
  - `src/application/adapters/strategy_signal_adapter.py`
  - `src/application/adapters/risk_decision_adapter.py`
  - `src/application/trading_pipeline.py`
  - `tests/test_strategy_signal_adapter.py`
  - `tests/test_risk_decision_adapter.py`
  - `tests/test_trading_pipeline.py`
  - `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_strategy_signal_adapter tests.test_risk_decision_adapter tests.test_trading_pipeline`
  - 结果：2 个模块导入错误，`src.application.adapters` 不存在；另有 1 个失败，说明 `risk_accepted=False` 尚未阻止执行，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_strategy_signal_adapter tests.test_risk_decision_adapter tests.test_trading_pipeline`
  - 结果：`Ran 8 tests in 0.626s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 46 tests in 7.665s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_095815.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 8：BacktestUseCase 直接消费 MarketDataFeed

目标：让回测用例不再依赖 mode 的历史数据切片私有方法。

- [x] 扩展或确认 `MarketDataFeed` 可以按时间顺序产生回测切片。
- [x] 让 `BacktestUseCase` 支持从 feed 遍历 market slices。
- [x] 让 `BacktestTradingMode` 负责组装 feed 和 pipeline。
- [x] 保留 `_load_historical_data()`、`_get_combined_timestamps()` 和 `_get_data_at_timestamp()` 作为过渡兼容，直到新路径稳定。
- [x] 测试 `BTC/USDT 1m` 在 `2025-01-01` 到 `2025-01-02` 可以产生 1441 个时间点。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/backtest_use_case.py`
- `src/application/trading_pipeline.py`
- `src/datasource/feeds/market_data_feed.py`
- `src/trading/modes/backtest.py`
- `tests/test_backtest_use_case.py`
- `tests/test_market_data_feed.py`

预期验证：

- 回测主循环不再需要 mode 私有数据切片方法。
- 本地 parquet 数据仍能优雅读取。
- 基线回测数值不变。

完成证据：

- 本阶段完成说明：
  - 已完成 `BacktestUseCase` 的 feed 路径：当 mode 设置 `market_data_feed` 时，use case 会调用 `market_data_feed.load_range()`，并将 `MarketSlice` 转成当前 `TradingPipeline` 兼容的单行 DataFrame data map。
  - `BacktestTradingMode` 已自动从本地 parquet store 组装 `HistoricalMarketDataFeed`，CLI 基线回测默认进入 feed 路径。
  - feed 转换会同时保留 `datetime` 列、毫秒级 `timestamp` 列，并使用唯一 `DatetimeIndex`，避免策略内部增量 buffer 因重复索引触发 `cannot reindex on an axis with duplicate labels`。
  - 旧的 `_load_historical_data()`、`_get_combined_timestamps()` 和 `_get_data_at_timestamp()` 路径仍保留为过渡兼容；执行引擎仍复用 `historical_data`。
- 新增/修改文件：
  - `src/application/backtest_use_case.py`
  - `src/trading/modes/backtest.py`
  - `tests/test_backtest_use_case.py`
  - `tests/test_backtest_mode.py`
  - `tests/test_market_data_feed.py`
  - `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case tests.test_market_data_feed`
  - 结果：1 个错误，`BacktestUseCase` 仍强依赖 `mode.timestamps`，有 feed 时不会走 feed 路径，符合预期。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_use_case`
  - 结果：1 个失败，feed 转换出的 DataFrame 索引为默认 `[0]`，不能保证策略 buffer 中每根 K 线索引唯一，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_market_data_feed tests.test_backtest_use_case`
  - 结果：`Ran 14 tests in 9.355s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 51 tests in 7.582s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_101903.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 9：抽出 ReportUseCase / Reporter

目标：让报告生成、保存和摘要日志离开 `BaseTradingMode`。

- [x] 添加报告用例或 reporter。
  - 候选路径：`src/application/report_use_case.py`
  - 候选路径：`src/reporting/backtest_reporter.py`
- [x] 让 `BaseTradingMode._generate_report()` 委托给 reporter。
- [x] 让 `BaseTradingMode._save_report()` 委托给 reporter 或 report writer。
- [x] 保留当前 JSON 字段和文件命名规则。
- [x] 测试报告字段包含当前关键字段。
- [x] 测试报告写入 `reports/backtest`。
- [x] 运行完整单元测试。
- [x] 运行 1 天基线回测。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/application/report_use_case.py`
- `src/reporting/backtest_reporter.py`
- `src/trading/modes/base.py`
- `tests/test_report_use_case.py`
- `tests/test_backtest_mode.py`

预期验证：

- 报告不再修改组合状态。
- 报告字段与当前结果兼容。
- 基线回测数值不变。

完成证据：

- 本阶段完成说明：
  - 新增 `TradingReportUseCase`，负责生成报告、写入报告文件、序列化清洗和摘要日志。
  - `BaseTradingMode._generate_report()`、`_save_report()`、`_prepare_report_for_serialization()` 和 `_log_report_summary()` 已改为薄委托，保留旧入口兼容现有调用。
  - 当前先放在 `src/application/report_use_case.py`；没有额外创建 `src/reporting/backtest_reporter.py`，避免在本阶段引入第二层空转抽象。
- 新增/修改文件：
  - `src/application/report_use_case.py`
  - `src/trading/modes/base.py`
  - `tests/test_report_use_case.py`
  - `tests/test_backtest_mode.py`
  - `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_report_use_case`
  - 结果：2 个错误，`ModuleNotFoundError: No module named 'src.application.report_use_case'`，符合预期。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode`
  - 结果：2 个错误，`src.trading.modes.base` 尚无 `TradingReportUseCase` 可供委托，符合预期。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_report_use_case tests.test_backtest_mode`
  - 结果：`Ran 11 tests in 9.930s`，`OK`。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_report_use_case tests.test_backtest_mode tests.test_backtest_use_case tests.test_trading_pipeline`
  - 结果：`Ran 18 tests in 9.359s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 55 tests in 8.570s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_102910.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

### Phase 10：paper/live 复用同一 TradingPipeline

目标：让 paper/live 和 backtest 共享交易决策流程，同时保护实盘安全。

- [x] 评估当前 `PaperTradingMode` 的数据和执行路径。
- [x] 评估当前 `LiveTradingMode` 的数据和执行路径。
- [x] 让 paper mode 使用 `TradingPipeline` 跑一次循环。
- [x] 为 live mode 添加显式 live enable 安全开关验证。
- [x] live mode 没有显式安全配置时必须拒绝启动。
- [x] 保持 backtest 基线不变。
- [x] 更新本文档，记录修改文件和验证证据。

预期修改文件：

- `src/trading/modes/paper.py`
- `src/trading/modes/live.py`
- `src/application/trading_pipeline.py`
- `tests/test_paper_mode.py`
- `tests/test_live_mode.py`
- `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`

预期验证：

- paper mode 可以在不真实下单的情况下跑通一次交易循环。
- live mode 默认安全，不会因配置误用而真实下单。
- backtest 基线回测仍然可复现。

完成证据：

- 本阶段完成说明：
  - `PaperTradingMode` 当前循环已经通过 `self._process_market_data(data_map)` 进入 `BaseTradingMode`，再委托到 `TradingPipeline`；本阶段补测试证明该路径可跑通一轮，没有重写 paper mode。
  - `LiveTradingMode.initialize()` 已新增 `_verify_live_trading_enabled()` 安全门，必须同时满足 `live_trading.enabled=true` 和 `live_trading.confirm_live_trading=true` 才会继续创建 live `ExecutionEngine`、连接交易所、验证账户。
  - `conf/config.yaml` 和 `conf/example.yaml` 已将 live 安全开关默认设置为 `false`，防止误用配置真实下单。
- 新增/修改文件：
  - `src/trading/modes/live.py`
  - `tests/test_paper_mode.py`
  - `tests/test_live_mode.py`
  - `conf/config.yaml`
  - `conf/example.yaml`
  - `docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`
- 未修改但已验证：
  - `src/trading/modes/paper.py`：现有实现已经通过 base 入口复用 `TradingPipeline`。
  - `src/application/trading_pipeline.py`：本阶段不需要改动 pipeline 主流程。
- RED 验证：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_mode tests.test_live_mode`
  - 结果：live 相关测试出现 2 个错误，默认未显式启用 live 时仍继续到了交易所连接检查并抛出 `ConnectionError`，说明缺少安全门；paper 测试初版夹具未设置 `_running=True`，随后修正测试夹具。
- GREEN 聚焦测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_mode tests.test_live_mode`
  - 结果：`Ran 4 tests in 7.058s`，`OK`。
  - 命令：`.\.venv\Scripts\python.exe -m unittest tests.test_paper_mode tests.test_live_mode tests.test_backtest_mode tests.test_trading_pipeline`
  - 结果：`Ran 17 tests in 9.323s`，`OK`。
- 完整测试：
  - 命令：`.\.venv\Scripts\python.exe -m unittest discover tests`
  - 结果：`Ran 59 tests in 10.500s`，`OK`。
- 基线回测：
  - 命令：`.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv`
  - 报告：`reports/backtest/backtest_report_20260609_104133.json`
  - 结果：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。

## 进度记录

### 2026-06-09

- 创建第二阶段设计文档：`docs/design/2026-06-09-trading-pipeline-phase-2-design.md`。
- 创建第二阶段实施清单：`docs/worklist/2026-06-09-trading-pipeline-phase-2-worklist.md`。
- 完成 Phase 6：新增 `TradingPipeline`，将 `BaseTradingMode._process_market_data()` 的交易决策主流程迁出到应用层；当前仍保留 DataFrame 策略、风控和执行协议。
- 完成 Phase 7：新增 `StrategySignalAdapter` 和 `RiskDecisionAdapter`，让 DataFrame 策略输出具备稳定 `signal_id`，并让风控结果可通过 `risk_accepted` 阻止执行；当前没有重写 `DualMAStrategy`。
- 完成 Phase 8：`BacktestUseCase` 已直接消费 `MarketDataFeed`，`BacktestTradingMode` 已自动组装真实 `HistoricalMarketDataFeed`；本地 `BTC/USDT 1m` 在 `2025-01-01` 到 `2025-01-02` 可产生 1441 个切片，CLI 基线回测恢复到 60 笔交易和最终净值 `99863.31403628497`。
- Phase 8 修复说明：feed 转换出的 DataFrame 需要同时提供旧策略依赖的毫秒级 `timestamp` 列和唯一 `DatetimeIndex`，否则会分别触发 `'timestamp'` 缺失和 `cannot reindex on an axis with duplicate labels`。
- 完成 Phase 9：新增 `TradingReportUseCase`，将报告生成、保存、序列化清洗和摘要日志从 `BaseTradingMode` 迁出；`BaseTradingMode` 保留旧方法作为薄委托入口，基线回测数值不变。
- 完成 Phase 10：确认 `PaperTradingMode` 已通过 base 入口复用 `TradingPipeline`，并为 `LiveTradingMode` 增加显式 live 安全开关；没有同时设置 `live_trading.enabled=true` 和 `live_trading.confirm_live_trading=true` 时，live mode 会在创建真实执行引擎前拒绝启动。
- 补充 Phase 1-10 相关代码中文注释，覆盖领域模型、数据规范化、parquet store、MarketDataFeed、回测成交模型、执行引擎委托、回测用例、交易 pipeline、报告用例、mode 委托和 live 安全门；本次只补说明，不改变业务逻辑。
- 中文注释补充验证：聚焦测试 `Ran 34 tests in 8.782s`，`OK`；完整测试 `Ran 59 tests in 7.977s`，`OK`。
- 第二阶段完成说明：Phase 6 到 Phase 10 已全部完成，当前基线回测保持最终净值 `99863.31403628497`、总交易数 `60`、结束持仓 `{}`。
- 推荐下一次继续入口：进入第三阶段设计，优先处理策略接口从 DataFrame 协议迁移到领域对象协议，以及执行引擎 live/paper/backtest 的更清晰分层。
