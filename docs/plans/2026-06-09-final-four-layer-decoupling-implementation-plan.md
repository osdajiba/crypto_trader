# Final Four-Layer Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前“领域化主链路 + legacy adapter”收敛为稳定的 Data + Factor、Strategy、Order + Execution、Performance + Report 四层架构。

**Architecture:** 保留 `DomainTradingPipeline` 作为薄编排主链路，但把策略数量计算、因子 buffer、旧 DataFrame pipeline、report context fallback 和 CLI 旧命名逐步移出主路径。每个阶段必须更新 `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`，记录修改文件、测试命令、测试结果和下一次继续入口。

**Tech Stack:** Python 3.8+、unittest、pandas 仅限 legacy adapter/data/report 边界、PyYAML config、当前本地 parquet 历史数据。

---

## File Structure

- Modify: `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
  - 职责：记录最终四层架构、当前状态、阶段顺序和验收标准。
- Create: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
  - 职责：作为本轮可续接进度清单。每完成一个任务都必须同步更新。
- Modify: `src/launcher.py`
  - 职责：清理 `--backtest-engine` 旧语义和默认 config 路径。
- Modify: `tests/test_launcher_cli.py`
  - 职责：验证 CLI 参数语义和旧 `BacktestFactory` 不再进入主路径。
- Create: `src/order/__init__.py`
  - 职责：暴露新的订单和仓位 sizing 模块。
- Create: `src/order/models.py`
  - 职责：定义 `PositionSizingDecision` 等订单层辅助模型。
- Create: `src/order/sizing.py`
  - 职责：根据信号、组合、行情和风险预算计算目标数量。
- Create: `src/order/factory.py`
  - 职责：把 `StrategySignal + PositionSizingDecision` 转成 `OrderIntent`。
- Test: `tests/test_order_sizing.py`
  - 职责：覆盖固定金额、固定比例、现金不足、最小订单金额等 sizing 规则。
- Create: `src/factor/__init__.py`
  - 职责：暴露 factor 层。
- Create: `src/factor/models.py`
  - 职责：定义 `FactorView`，作为策略读取因子的稳定对象。
- Create: `src/factor/engine.py`
  - 职责：维护市场历史窗口并生成基础 MA factor view。
- Test: `tests/test_factor_engine.py`
  - 职责：验证 MA 计算、warmup 和策略不再维护原始行情 buffer。
- Create: `src/strategy/domain_base.py`
  - 职责：定义新策略基类，面向领域对象和 factor view。
- Create: `src/strategy/implementations/domain_dual_ma.py`
  - 职责：原生领域版 dual_ma，只输出 `StrategySignal`，不输出 DataFrame，不计算 quantity。
- Modify: `src/application/runtime_builder.py`
  - 职责：优先组装原生策略和订单层组件；旧策略继续走 legacy adapter。
- Test: `tests/test_domain_dual_ma_strategy.py`
  - 职责：验证原生 dual_ma 信号、无 quantity 输出、可进入 domain pipeline。
- Create: `src/reporting/__init__.py`
  - 职责：暴露 reporting 层。
- Create: `src/reporting/performance_analyzer.py`
  - 职责：从 fills 和 snapshots 计算收益、回撤、胜率、Sharpe 等指标。
- Create: `src/reporting/writers.py`
  - 职责：处理 JSON/CSV/HTML 等输出。
- Modify: `src/application/report_use_case.py`
  - 职责：降级为 reporter/analyzer/writer 的协调器，不再直接计算指标或依赖 mode 私有状态。
- Test: `tests/test_performance_analyzer.py`
  - 职责：覆盖指标计算。
- Test: `tests/test_report_writer.py`
  - 职责：覆盖报告落盘和 JSON 清洗。
- Modify: `src/application/trading_pipeline.py`
  - 职责：标记 legacy 或迁移到 legacy 包。
- Modify: `src/trading/modes/base.py`
  - 职责：删除或降级旧 DataFrame 执行入口。
- Modify: `src/trading/execution/manager.py`
  - 职责：标记旧 `ExecutionEngine.execute(signals_df)` 为 legacy，不作为主执行入口。
- Test: `tests/test_domain_boundaries.py`
  - 职责：扩展 import boundary 和 DataFrame 泄漏检查。

## Baseline Commands

每个任务完成后至少运行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m unittest discover tests
```

基线回测命令：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='E:\myProgram\crypto_trader\src;E:\myProgram\crypto_trader'
.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv
```

预期基线：最终净值 `99863.31403628497`，总交易数 `60`，买入 `30`，卖出 `30`，结束持仓 `{}`。如数值变化，必须在 worklist 记录旧值、新值和原因。

### Task 1: Architecture Boundary And CLI Naming Audit

**Files:**
- Modify: `src/launcher.py`
- Modify: `tests/test_launcher_cli.py`
- Modify: `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing CLI tests**

Add tests that assert the default config path is `conf/config.yaml`, and that the main CLI path does not call `BacktestFactory.run_backtest()`.

```python
def test_launcher_default_config_points_to_conf_config_yaml(self):
    from src.launcher import default_config_path

    self.assertTrue(str(default_config_path()).endswith("conf\\config.yaml") or str(default_config_path()).endswith("conf/config.yaml"))


def test_cli_backtest_engine_is_execution_model_name_not_legacy_factory(self):
    import src.backtest.engine as legacy_engine
    from unittest.mock import patch

    with patch.object(legacy_engine.BacktestFactory, "run_backtest", side_effect=AssertionError("legacy factory used")):
        # run_cli_mode should complete through TradingCore/RuntimeBuilder path.
        # Use existing CLI helper fixture from tests/test_launcher_cli.py.
        self.run_backtest_cli_once(backtest_engine="ohlcv")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli -v
```

Expected: FAIL because `launcher.py` still hardcodes `conf/bt_config.yaml`, and helper functions may not exist yet.

- [ ] **Step 3: Implement minimal CLI naming cleanup**

Add a small helper in `src/launcher.py`:

```python
def default_config_path() -> Path:
    return Path(project_root) / "conf" / "config.yaml"
```

Replace the old default:

```python
default_config_path = base_dir / "conf/bt_config.yaml"
config_path = Path(args.config) if args.config else default_config_path
```

with:

```python
config_path = Path(args.config) if args.config else default_config_path()
```

Keep `--backtest-engine` temporarily, but update help text:

```python
help="Execution model name for backtest compatibility; does not call legacy BacktestFactory"
```

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_launcher_cli -v
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected: PASS.

- [ ] **Step 5: Run baseline backtest**

Run the baseline command above. Expected: final equity and trade counts unchanged.

- [ ] **Step 6: Update worklist**

In `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`, mark Task 1 completed and record commands/results/report path.

### Task 2: Minimal PositionSizer And OrderFactory

**Files:**
- Create: `src/order/__init__.py`
- Create: `src/order/models.py`
- Create: `src/order/sizing.py`
- Create: `src/order/factory.py`
- Modify: `src/application/runtime_builder.py`
- Test: `tests/test_order_sizing.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing sizing tests**

Create `tests/test_order_sizing.py` with these cases:

```python
def test_fixed_notional_sizer_calculates_quantity_from_price(self):
    decision = FixedNotionalSizer(notional=1000).size(signal, portfolio, market)
    self.assertEqual(decision.target_notional, 1000)
    self.assertEqual(decision.target_quantity, 10)


def test_sizer_caps_buy_to_available_cash_after_commission(self):
    portfolio = snapshot(cash=100)
    decision = FixedNotionalSizer(notional=1000, commission_rate=0.001).size(signal, portfolio, market)
    self.assertLessEqual(decision.target_notional, 100)


def test_order_factory_creates_order_intent_from_sizing_decision(self):
    intent = OrderFactory(order_type="market").create(signal, sizing_decision)
    self.assertEqual(intent.symbol, "BTC/USDT")
    self.assertEqual(intent.side, "buy")
    self.assertGreater(intent.quantity, 0)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_order_sizing -v
```

Expected: FAIL with missing `src.order`.

- [ ] **Step 3: Implement order models**

Create `src/order/models.py`:

```python
from dataclasses import dataclass
from src.domain.models import StrategySignal


@dataclass(frozen=True)
class PositionSizingDecision:
    signal: StrategySignal
    target_notional: float
    target_quantity: float
    reason: str = ""
```

- [ ] **Step 4: Implement minimal sizer and factory**

Create `src/order/sizing.py` with `FixedNotionalSizer` and `FixedFractionSizer`. Create `src/order/factory.py` with `OrderFactory.create(signal, sizing_decision) -> OrderIntent`.

Key rule: strategies do not set final quantity. Quantity comes from `PositionSizer`.

- [ ] **Step 5: Wire runtime builder**

Modify `RuntimeBuilder` so the default risk/order path can convert accepted signals into sized `OrderIntent`. Keep legacy behavior behind config until baseline is protected.

- [ ] **Step 6: Run verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_order_sizing tests.test_domain_trading_pipeline tests.test_domain_risk_policies -v
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected: PASS.

- [ ] **Step 7: Run baseline backtest and update worklist**

Run baseline backtest. Update worklist with result and any numeric difference.

### Task 3: Minimal FactorView And FactorEngine

**Files:**
- Create: `src/factor/__init__.py`
- Create: `src/factor/models.py`
- Create: `src/factor/engine.py`
- Modify: `src/domain/ports.py`
- Modify: `src/application/runtime_builder.py`
- Test: `tests/test_factor_engine.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing factor tests**

Create tests for a rolling MA view:

```python
def test_factor_engine_returns_empty_view_before_warmup(self):
    engine = FactorEngine()
    view = engine.update(market_slice(close=100))
    self.assertFalse(view.has("BTC/USDT", "ma_3"))


def test_factor_engine_calculates_moving_average_after_warmup(self):
    engine = FactorEngine(ma_windows=[3])
    for price in [100, 101, 102]:
        view = engine.update(market_slice(close=price))
    self.assertEqual(view.get("BTC/USDT", "ma_3"), 101)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine -v
```

Expected: FAIL because `src.factor` does not exist.

- [ ] **Step 3: Implement FactorView and FactorEngine**

`FactorEngine` may use pandas internally, but `src/domain` must not import pandas. Store per-symbol rolling bars inside factor layer, not strategy layer.

- [ ] **Step 4: Update StrategyPort migration path**

Keep current `StrategyPort.generate(market, portfolio)` compatible, but introduce an optional factor-aware protocol or adapter for `generate(market, factors, portfolio)`. Do not break legacy adapter.

- [ ] **Step 5: Run boundary tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine tests.test_domain_boundaries -v
```

Expected: PASS; `src/domain` still has no pandas import.

- [ ] **Step 6: Run full tests, baseline backtest, update worklist**

Run full tests and baseline backtest. Record results.

### Task 4: Native Domain DualMA Strategy

**Files:**
- Create: `src/strategy/domain_base.py`
- Create: `src/strategy/implementations/domain_dual_ma.py`
- Modify: `src/application/runtime_builder.py`
- Modify: `conf/strategy.yaml`
- Test: `tests/test_domain_dual_ma_strategy.py`
- Test: `tests/test_runtime_builder.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing native strategy tests**

Expected behaviors:

```python
def test_domain_dual_ma_outputs_signal_without_quantity_metadata(self):
    strategy = DomainDualMAStrategy(short_window=2, long_window=3)
    signals = await strategy.generate(market, factors, portfolio)
    self.assertEqual(signals[0].side, "buy")
    self.assertNotIn("quantity", signals[0].metadata)


def test_runtime_builder_can_select_native_dual_ma_without_legacy_adapter(self):
    runtime = RuntimeBuilder(mode).build_backtest_runtime(historical_data)
    self.assertNotEqual(runtime.domain_pipeline.strategy.__class__.__name__, "LegacyDataFrameStrategyAdapter")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy tests.test_runtime_builder -v
```

Expected: FAIL because native strategy does not exist.

- [ ] **Step 3: Implement domain strategy**

Create `BaseDomainStrategy` and `DomainDualMAStrategy`. The strategy reads `FactorView`; it must not import pandas, mutate `PortfolioBook`, call execution, or calculate final quantity.

- [ ] **Step 4: Wire config**

Support a strategy config flag such as:

```yaml
strategy:
  active: dual_ma
  interface: domain
```

Fallback to `LegacyDataFrameStrategyAdapter` when `interface` is absent or set to `legacy`.

- [ ] **Step 5: Verify native path**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_domain_dual_ma_strategy tests.test_runtime_builder tests.test_domain_trading_pipeline -v
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected: PASS.

- [ ] **Step 6: Run baseline in both legacy and native strategy modes**

Run baseline with legacy config and native config. If native results differ because sizing moved out of strategy, record old/new values and reason in worklist.

### Task 5: PerformanceAnalyzer And ReportWriter

**Files:**
- Create: `src/reporting/__init__.py`
- Create: `src/reporting/performance_analyzer.py`
- Create: `src/reporting/writers.py`
- Modify: `src/application/reporting.py`
- Modify: `src/application/report_use_case.py`
- Test: `tests/test_performance_analyzer.py`
- Test: `tests/test_report_writer.py`
- Test: `tests/test_report_use_case.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing reporting tests**

Add tests:

```python
def test_performance_analyzer_calculates_total_return_and_drawdown(self):
    result = PerformanceAnalyzer().analyze(fills=[], snapshots=snapshots)
    self.assertEqual(result["final_equity"], 110000)
    self.assertEqual(result["total_return"], 10000)


def test_report_use_case_does_not_require_mode_state_when_reporter_exists(self):
    report = TradingReportUseCase(context_with_domain_reporter_only).generate()
    self.assertEqual(report["total_trades"], 1)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_performance_analyzer tests.test_report_writer tests.test_report_use_case -v
```

Expected: FAIL with missing `src.reporting`.

- [ ] **Step 3: Implement analyzer and writers**

Move reusable metric math out of `src/backtest/performance.py` into `PerformanceAnalyzer`. Keep `src/backtest/performance.py` as legacy compatibility until Task 6.

- [ ] **Step 4: Refactor report use case**

`TradingReportUseCase` should coordinate reporter, analyzer and writer. It may keep a legacy fallback but must clearly label it.

- [ ] **Step 5: Verify**

Run focused tests, full tests, and baseline backtest. Confirm reports still write to `reports/backtest`.

- [ ] **Step 6: Update worklist**

Record report path, JSON field compatibility, and any metric differences.

### Task 6: Legacy Compatibility Chain Cleanup

**Files:**
- Modify or move: `src/application/trading_pipeline.py`
- Modify: `src/trading/modes/base.py`
- Modify or move: `src/trading/execution/manager.py`
- Modify or move: `src/backtest/engine.py`
- Modify: `tests/test_domain_boundaries.py`
- Modify: `tests/test_backtest_mode.py`
- Modify: `tests/test_paper_mode.py`
- Modify: `tests/test_live_mode.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing no-legacy-main-path tests**

Add tests that prove:

```python
def test_backtest_mode_does_not_call_application_dataframe_pipeline(self):
    with patch("src.application.trading_pipeline.TradingPipeline.run_once", side_effect=AssertionError("legacy pipeline used")):
        run_backtest_one_step_through_domain_pipeline()


def test_base_mode_no_longer_exposes_dataframe_execution_entrypoint(self):
    self.assertFalse(hasattr(BaseTradingMode, "_execute_signals"))
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_domain_boundaries -v
```

Expected: FAIL because old methods still exist.

- [ ] **Step 3: Delete or archive legacy modules**

Only delete a legacy module when all tests prove no main path uses it. Safer first step:

```text
src/legacy/application_trading_pipeline.py
src/legacy/backtest_engine.py
```

If moving files causes import churn, keep original files as thin deprecated wrappers with explicit warning comments.

- [ ] **Step 4: Remove DataFrame execution methods from BaseTradingMode**

Remove `_process_market_data`, `_filter_executable_signals`, `_execute_signals` only after backtest/paper/live tests prove use cases call `DomainTradingPipeline`.

- [ ] **Step 5: Verify all paths**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_backtest_mode tests.test_paper_mode tests.test_live_mode tests.test_domain_boundaries tests.test_runtime_builder -v
.\.venv\Scripts\python.exe -m unittest discover tests
```

Expected: PASS.

- [ ] **Step 6: Run baseline backtest and update final status**

Run baseline backtest. Mark final acceptance criteria in worklist. If all criteria pass, record that the project has moved from adapter-compatible architecture to native four-layer architecture.

### Task 7: Native Multi-Factors Strategy And Configurable FactorEngine

**Files:**
- Modify: `src/factor/engine.py`
- Modify: `src/factor/models.py`
- Create: `src/strategy/implementations/domain_multi_factors.py`
- Modify: `src/application/runtime_builder.py`
- Test: `tests/test_factor_engine_multi_factors.py`
- Test: `tests/test_domain_multi_factors_strategy.py`
- Modify: `tests/test_runtime_builder.py`
- Update: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`

- [ ] **Step 1: Write failing configurable factor tests**

Add tests that prove `FactorEngine.from_config()` can calculate a configured RSI factor, keeps warmup empty until enough bars exist, and emits a `composite_signal`.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_factor_engine_multi_factors -v
```

Expected: FAIL with missing `from_config` or missing configured factor output.

- [ ] **Step 2: Implement configured factors in FactorEngine**

Extend `FactorEngine` without importing legacy strategy modules. It may keep rolling `MarketBar` windows and calculate:

- `rsi`
- `macd`
- `bollinger`
- `volume_osc`
- `composite_signal`

Use simple numeric helpers in `src/factor/engine.py`; do not move old DataFrame indicator classes into the native path.

- [ ] **Step 3: Write failing native multi_factors strategy tests**

Create tests that pass explicit `FactorView` values and assert:

- composite crossing above threshold emits a buy signal.
- composite crossing below negative threshold emits a sell signal when a position exists.
- repeated same-side composite state does not emit duplicate signals.
- signal metadata does not contain `quantity`.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_domain_multi_factors_strategy -v
```

Expected: FAIL because `DomainMultiFactorsStrategy` does not exist.

- [ ] **Step 4: Implement DomainMultiFactorsStrategy**

Create `src/strategy/implementations/domain_multi_factors.py`. The class must extend `BaseDomainStrategy`, consume `FactorView`, track previous composite relation per symbol, and emit `StrategySignal` without final quantity.

- [ ] **Step 5: Wire RuntimeBuilder**

Support `strategy.interface=domain` and `strategy.active=multi_factors`. Build `DomainMultiFactorsStrategy` with:

- `threshold` from `strategy.parameters.threshold`, default `0.5`.
- factor config from `strategy.factors` or `strategy.multi_factors.factors` only when available.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_runtime_builder tests.test_domain_multi_factors_strategy tests.test_factor_engine_multi_factors -v
```

Expected: PASS.

- [ ] **Step 6: Run full verification and baseline**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv
```

Expected: full tests pass and `dual_ma` baseline remains final equity `99863.31403628497`, total trades `60`, buys `30`, sells `30`, positions `{}`.

- [ ] **Step 7: Update worklist**

Mark Task 7 completed, record RED/GREEN commands, full test result, baseline report path and next breakpoint.

### Task 8: Default Domain Strategy Interface In Main Config

**Files:**
- Modify: `conf/config.yaml`
- Modify: `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- Modify: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- Test: `tests/test_config_defaults.py`
- Test: `tests/test_runtime_builder.py`

- [ ] **Step 1: Write failing config default test**

Add a test that loads `conf/config.yaml` and asserts:

```python
def test_main_config_defaults_strategy_interface_to_domain(self):
    data = yaml.safe_load((repo_root / "conf" / "config.yaml").read_text(encoding="utf-8"))
    self.assertEqual(data["strategy"]["interface"], "domain")
    self.assertEqual(data["strategy"].get("active", data["strategy"].get("default")), "dual_ma")
```

Also keep a runtime builder test showing omitted `strategy.interface` still falls back to legacy.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config_defaults tests.test_runtime_builder -v
```

Expected: FAIL because `conf/config.yaml` has no `strategy.interface`.

- [ ] **Step 2: Update main config**

Add this under the top-level `strategy:` section in `conf/config.yaml`:

```yaml
  active: dual_ma
  interface: domain
```

Do not change `RuntimeBuilder._strategy_interface()` code default; it should remain `default="legacy"` for old configuration compatibility.

- [ ] **Step 3: Run focused verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config_defaults tests.test_runtime_builder tests.test_domain_dual_ma_strategy tests.test_domain_multi_factors_strategy -v
```

Expected: PASS.

- [ ] **Step 4: Run full verification and default native baseline**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
.\.venv\Scripts\python.exe -m src.main --cli --config conf/config.yaml --mode backtest --strategy dual_ma --symbol "BTC/USDT" --timeframe 1m --start-date 2025-01-01 --end-date 2025-01-02 --backtest-engine ohlcv
```

Expected: tests pass. Baseline may change from legacy because default CLI now uses native domain `dual_ma`; record the new report path and explain the difference in worklist.

- [ ] **Step 5: Update worklist**

Mark Task 8 completed, record RED/GREEN commands, full test result, default native baseline report path and next breakpoint.

### Task 9: Legacy Compatibility Deletion Readiness Gate

**Files:**
- Modify: `docs/design/2026-06-09-final-four-layer-decoupling-architecture-design.md`
- Modify: `docs/worklist/2026-06-09-final-four-layer-decoupling-worklist.md`
- Test: `tests/test_legacy_deletion_readiness.py`

- [ ] **Step 1: Write failing deletion-readiness tests**

Add tests that prove the main config default runtime uses native strategy/risk policy, and that deletion blockers are documented before legacy modules are removed.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_legacy_deletion_readiness -v
```

Expected: FAIL until design/worklist document the explicit deletion blockers.

- [ ] **Step 2: Document deletion blockers**

Record that legacy deletion is temporarily deferred because:

- 未声明 `strategy.interface` 的旧配置仍 fallback 到 legacy adapter。
- `BaseTradingMode._process_market_data` 和旧 DataFrame pipeline 仍有兼容测试。
- `ExecutionEngine` 仍被 mode 初始化，live 路径还通过它持有 exchange client。
- 旧 adapter / pipeline / execution tests 仍在保护兼容行为。

- [ ] **Step 3: Verify**

Run focused tests, full tests, and the default native baseline backtest. If no production code changed, baseline should stay on the Task 8 native numbers unless market data or config changed.

- [ ] **Step 4: Update worklist**

Mark Task 9 completed, record that no legacy modules were deleted in this pass, and set the next breakpoint to either exchange-client ownership migration or old DataFrame pipeline test migration.

## Self-Review

- Spec coverage: The plan covers Phase A, D-min, C, B-min/B, E and F from the design document. Task 7 covers the next-round native `multi_factors` migration and configurable FactorEngine follow-up. Task 8 covers conservative defaulting of the main config to domain while preserving code-level legacy fallback. Task 9 adds a deletion-readiness gate so legacy removal is driven by tests instead of manual guessing.
- Placeholder scan: No `TBD`/`TODO` placeholders are used; each task has concrete files, tests and commands.
- Type consistency: `StrategySignal`, `OrderIntent`, `PortfolioSnapshot`, `MarketSlice`, `DomainTradingPipeline`, `RuntimeBuilder` names match the current codebase. The only planned new protocol surface is factor-aware strategy generation, explicitly marked as a migration from current `StrategyPort` v1.
